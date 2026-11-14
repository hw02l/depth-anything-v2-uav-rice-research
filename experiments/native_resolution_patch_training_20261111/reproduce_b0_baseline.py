from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "local_huber_20261105"))
sys.path.insert(0, str(HERE))

from data_patch import read_manifest  # noqa: E402
from evaluate_patch import _pool_absolute, evaluate_one  # noqa: E402
from patch_engine import file_sha256  # noqa: E402
from train_patch import check_initial_hashes, load_model, verify_manifest  # noqa: E402

BASELINE_PAIR_SHA256 = "bcb56eab03bec4a44f0626da0f4814876fae67774e536e00cd9f1ad16b1f5728"
ABSOLUTE_TOLERANCE = 1e-7


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty baseline output: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def pooled_large_pairs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [r for r in rows if r["stride_px"] == 14 and r["group"] == "gt_ge20cm"]
    count = sum(int(r["pair_count"]) for r in rows)
    error = sum(float(r["abs_delta_error_sum"]) for r in rows)
    gt_abs = sum(float(r["abs_gt_delta_sum"]) for r in rows)
    pred_abs = sum(float(r["abs_prediction_delta_sum"]) for r in rows)
    nonzero = sum(int(r["nonzero_gt_pair_count"]) for r in rows)
    sign_match = sum(int(r["sign_match_count"]) for r in rows)
    if not count or not gt_abs or not nonzero:
        raise ValueError("B0 primary local-pair statistics are empty")
    return {
        "pair_count": count,
        "local_mae_m": error / count,
        "absolute_error_sum_m": error,
        "absolute_gt_delta_sum_m": gt_abs,
        "absolute_prediction_delta_sum_m": pred_abs,
        "amplitude_ratio": pred_abs / gt_abs,
        "nonzero_gt_pair_count": nonzero,
        "sign_match_count": sign_match,
        "sign_match_rate": sign_match / nonzero,
    }


def compare(actual: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    absolute = actual["absolute"]
    expected_absolute = expected["absolute"]
    actual_large = actual["gt_ge20cm_14px"]
    expected_large = expected["gt_ge20cm_14px"]
    fields = {
        "valid_pixels": (absolute["valid_pixels"], expected_absolute["valid_pixels"], 0),
        "mae_m": (absolute["mae_m"], expected_absolute["mae_m"], ABSOLUTE_TOLERANCE),
        "rmse_m": (absolute["rmse_m"], expected_absolute["rmse_m"], ABSOLUTE_TOLERANCE),
        "abs_rel": (absolute["abs_rel"], expected_absolute["abs_rel"], ABSOLUTE_TOLERANCE),
        "pearson_r": (absolute["pearson_r"], expected_absolute["pearson_r"], ABSOLUTE_TOLERANCE),
        "full14_pair_count": (absolute["full14_pair_count"], expected_absolute["local_pair_count"], 0),
        "full14_local_mae_m": (absolute["full14_mae_m"], expected_absolute["local_depth_delta_mae_m"], ABSOLUTE_TOLERANCE),
        "gt_ge20_pair_count": (actual_large["pair_count"], expected_large["pair_count"], 0),
        "gt_ge20_local_mae_m": (actual_large["local_mae_m"], expected_large["local_mae_m"], ABSOLUTE_TOLERANCE),
        "gt_ge20_amplitude_ratio": (actual_large["amplitude_ratio"], expected_large["amplitude_ratio"], ABSOLUTE_TOLERANCE),
        "gt_ge20_sign_match_rate": (actual_large["sign_match_rate"], expected_large["sign_match_rate"], ABSOLUTE_TOLERANCE),
    }
    if "bias_m" in expected_absolute:
        fields["bias_m"] = (absolute["bias_m"], expected_absolute["bias_m"], ABSOLUTE_TOLERANCE)
    results = {}
    for key, (value, reference, tolerance) in fields.items():
        if tolerance == 0:
            passed = int(value) == int(reference)
        else:
            passed = abs(float(value) - float(reference)) <= tolerance
        results[key] = {
            "actual": value, "reference": reference,
            "absolute_tolerance": tolerance, "passed": bool(passed),
        }
    return {"passed": all(item["passed"] for item in results.values()), "checks": results}


def run(args) -> None:
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("B0 gate requires the verified CUDA device")
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to reuse or overwrite baseline output: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir(exist_ok=False)

    rows = read_manifest(args.validation_manifest)
    verify_manifest(rows, args.validation_manifest,
                    ROOT / "configs/uav3dcrop_verified/validation.csv", 128, "validation")
    pair_hash = file_sha256(args.pair_manifest)
    if pair_hash != BASELINE_PAIR_SHA256:
        raise ValueError(f"Fixed pair manifest SHA256 mismatch: {pair_hash}")
    initial_hashes = check_initial_hashes(args.initial_checkpoint)
    model, load_audit = load_model(args.initial_checkpoint, args.device)
    model.eval()
    reference = json.loads(args.reference_summary.read_text(encoding="utf-8"))
    if reference.get("model_id") != "E0" or int(reference.get("epoch", -1)) != 20:
        raise ValueError("Reference must be the saved epoch-20 E0/B0 baseline")
    reference_rows = list(csv.DictReader(args.reference_per_image.open("r", encoding="utf-8", newline="")))
    if len(reference_rows) != 128:
        raise ValueError("Saved E0 baseline must contain exactly 128 image rows")
    expected_images = json.loads(args.pair_manifest.read_text(encoding="utf-8"))["images"]
    limit = args.max_images or len(rows)
    if not 1 <= limit <= len(rows):
        raise ValueError("max-images must be in [1, 128]")

    absolute_rows, local_rows, boundary_rows = [], [], []
    for index, row in enumerate(rows[:limit]):
        absolute, local, boundary, _ = evaluate_one(
            model, row, index, args.rgb_root, args.depth_root, args.device, "P0",
            expected_images[index])
        absolute_rows.append(absolute)
        local_rows.extend(local)
        boundary_rows.extend(boundary)
        if (index + 1) % 8 == 0 or index + 1 == limit:
            print(json.dumps({"phase": "epoch20_B0", "images": index + 1,
                              "total": limit}, ensure_ascii=False), flush=True)
    del model
    torch.cuda.empty_cache()

    if limit != 128:
        write_csv(out / "smoke_per_image_absolute.csv", absolute_rows)
        write_csv(out / "smoke_per_image_local.csv", local_rows)
        write_json(out / "b0_smoke.json", {
            "status": "single_image_gpu_smoke_only", "image_count": limit,
            "method": "corrected P0; uint8 RGB; frozen B0 preprocess/restore/mask/pair sampler",
            "checkpoint_sha256": initial_hashes["model.safetensors"],
            "pair_manifest_sha256": pair_hash,
        })
        return

    actual_absolute = _pool_absolute(absolute_rows)
    actual = {
        "absolute": actual_absolute,
        "gt_ge20cm_14px": pooled_large_pairs(local_rows),
    }
    comparison = compare(actual, reference)
    per_image_comparisons = []
    for actual_row, expected_row in zip(absolute_rows, reference_rows):
        if actual_row["image_index"] != int(expected_row["image_index"]):
            raise ValueError("Per-image reference order differs")
        if actual_row["image_relpath"] != expected_row["image_relpath"]:
            raise ValueError("Per-image reference identity differs")
        large_rows = [r for r in local_rows if r["image_index"] == actual_row["image_index"]
                      and r["stride_px"] == 14 and r["group"] == "gt_ge20cm"]
        large_count = sum(int(r["pair_count"]) for r in large_rows)
        large_error = sum(float(r["abs_delta_error_sum"]) for r in large_rows)
        large_gt = sum(float(r["abs_gt_delta_sum"]) for r in large_rows)
        large_pred = sum(float(r["abs_prediction_delta_sum"]) for r in large_rows)
        large_nonzero = sum(int(r["nonzero_gt_pair_count"]) for r in large_rows)
        large_sign = sum(int(r["sign_match_count"]) for r in large_rows)
        comparisons = {
            "valid_pixels": (int(actual_row["valid_pixels"]), int(expected_row["valid_pixel_count"]), 0),
            "full14_pair_count": (int(actual_row["full14_pair_count"]), int(expected_row["full14_pair_count"]), 0),
            "gt_ge20_pair_count": (large_count, int(expected_row["gt_ge20_pair_count"]), 0),
            "mae_m": (float(actual_row["mae_m"]), float(expected_row["mae_m"]), ABSOLUTE_TOLERANCE),
            "rmse_m": (float(actual_row["rmse_m"]), float(expected_row["rmse_m"]), ABSOLUTE_TOLERANCE),
            "abs_rel": (float(actual_row["abs_rel"]), float(expected_row["abs_rel"]), ABSOLUTE_TOLERANCE),
            "full14_local_mae_m": (float(actual_row["full14_mae_m"]), float(expected_row["full14_local_mae_m"]), ABSOLUTE_TOLERANCE),
            "gt_ge20_local_mae_m": (large_error / large_count, float(expected_row["gt_ge20_local_mae_m"]), ABSOLUTE_TOLERANCE),
            "gt_ge20_amplitude_ratio": (large_pred / large_gt, float(expected_row["gt_ge20_amplitude_ratio"]), ABSOLUTE_TOLERANCE),
            "gt_ge20_sign_match_rate": (large_sign / large_nonzero, float(expected_row["gt_ge20_sign_match_rate"]), ABSOLUTE_TOLERANCE),
        }
        checks = {}
        for name, (value, reference_value, tolerance) in comparisons.items():
            passed = int(value) == int(reference_value) if tolerance == 0 else abs(value - reference_value) <= tolerance
            checks[name] = {"actual": value, "reference": reference_value,
                            "tolerance": tolerance, "passed": bool(passed)}
        per_image_comparisons.append({"image_index": actual_row["image_index"],
                                      "image_relpath": actual_row["image_relpath"],
                                      "checks": checks,
                                      "passed": all(v["passed"] for v in checks.values())})
    comparison["per_image_count_checks_passed"] = all(
        record["passed"] for record in per_image_comparisons)
    comparison["passed"] = comparison["passed"] and comparison["per_image_count_checks_passed"]
    if not comparison["passed"]:
        write_csv(out / "b0_per_image_absolute.csv", absolute_rows)
        write_csv(out / "b0_per_image_local.csv", local_rows)
        write_json(out / "b0_baseline_comparison_FAILED.json", {
            "status": "failed_hard_gate", "actual": actual,
            "comparison": comparison, "per_image_count_checks": per_image_comparisons,
        })
        raise RuntimeError("Epoch-20 B0 reproduction did not meet the predeclared tolerance")

    write_csv(out / "b0_per_image_absolute.csv", absolute_rows)
    write_csv(out / "b0_per_image_local.csv", local_rows)
    payload = {
        "status": "passed_hard_gate", "model_id": "E0", "epoch": 20,
        "method": "P0 whole-frame 518px; corrected uint8 RGB; frozen B0 preprocess/restore/valid mask/fixed pairs",
        "image_count": 128, "independent_test_used": False,
        "input_contract": "load_pair float32 [0,1] -> np.rint(*255) uint8 [0,255]; strict infer_depth uint8",
        "checkpoint_sha256": initial_hashes["model.safetensors"],
        "initial_config_sha256": initial_hashes["config.json"],
        "validation_manifest_sha256": sha256(args.validation_manifest),
        "pair_manifest_sha256": pair_hash,
        "reference_summary_sha256": sha256(args.reference_summary),
        "reference_per_image_sha256": sha256(args.reference_per_image),
        "comparison_tolerance": {"metric_absolute_tolerance": ABSOLUTE_TOLERANCE,
                                  "integer_counts": "exact"},
        "model_load_audit": load_audit, "actual": actual,
        "comparison": comparison,
    }
    write_json(out / "b0_baseline_reproduction.json", payload)
    print(json.dumps({"status": payload["status"], "absolute": actual["absolute"],
                      "gt_ge20cm_14px": actual["gt_ge20cm_14px"]}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Reproduce the saved epoch-20 E0/B0 Validation128 baseline as a hard gate.")
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--rgb-root", type=Path, required=True)
    parser.add_argument("--depth-root", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--reference-summary", type=Path, required=True)
    parser.add_argument("--reference-per-image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max-images", type=int, help="Single-image smoke only; does not pass the baseline gate")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

