#!/usr/bin/env python3
"""Re-evaluate final E0-E3 models with the frozen B0 path and save per-image sufficient statistics.

This is evaluation-only. It never trains, changes existing checkpoints, or reads the independent test set.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

import cv2
import numpy as np
import tifffile
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForDepthEstimation

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "src"))
from uav_metric_research import multiscale as m  # noqa: E402

BASE_MODEL_SHA256 = "52769e2e95704d0862e708e8772195b861c647dd4812058eaf698a44305c3bcb"
BASE_CONFIG_SHA256 = "3c6ab05881e6a76edbfd33ff393fdfb082582084e9262826bb1ac0fd2b1dac7c"
GROUPS = ("E0", "E1", "E2", "E3")
SCALES = (14, 28, 56, 107, 214)
SUFFICIENT = (
    "valid_pixel_count", "abs_error_sum_m", "squared_error_sum_m2",
    "abs_relative_error_sum", "prediction_sum_m", "gt_sum_m",
    "prediction_squared_sum_m2", "gt_squared_sum_m2", "prediction_gt_cross_sum_m2",
    "full14_pair_count", "full14_abs_error_sum_m",
    "gt_ge20_pair_count", "gt_ge20_abs_error_sum_m", "gt_ge20_abs_gt_sum_m",
    "gt_ge20_abs_prediction_sum_m", "gt_ge20_nonzero_sign_count",
    "gt_ge20_sign_match_count",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload) -> None:
    temporary = path.with_name("." + path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_csv(path: Path, rows) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    temporary = path.with_name("." + path.name + ".tmp")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def new_model(base: Path, device: str):
    if sha256(base / "model.safetensors") != BASE_MODEL_SHA256:
        raise ValueError("Base model SHA256 mismatch")
    if sha256(base / "config.json") != BASE_CONFIG_SHA256:
        raise ValueError("Base config SHA256 mismatch")
    config = AutoConfig.from_pretrained(str(base), local_files_only=True)
    if config.max_depth != 80 or config.depth_estimation_type != "metric":
        raise ValueError("Expected metric 80m model configuration")
    model = AutoModelForDepthEstimation.from_config(config)
    audit = m.strict_attention_state(model, load_file(str(base / "model.safetensors"), device="cpu"))
    model.to(device)
    return model, audit


def load_group(model, group: str, base: Path, run_root: Path):
    if group == "E0":
        return {
            "checkpoint": str(base / "model.safetensors"),
            "checkpoint_sha256": BASE_MODEL_SHA256,
            "epoch": 20,
        }
    path = run_root / group / "epoch_0005.pt"
    expected_hash = {
        "E1": "d912edb89b4d42cfeabef5051d97e756210019414116f60e0961084525f3a19c",
        "E2": "b6875d59fd4ee32bbacce32664c493c885645acec906b074f2f615f27ef892b4",
        "E3": "223b75d1f5d81844bc5702565a8291270a8ae951d6e8e7c12be9f1b3ad12e886",
    }[group]
    if len(expected_hash) != 64:
        raise AssertionError(f"Invalid recorded SHA256 length for {group}: {len(expected_hash)}")
    if sha256(path) != expected_hash:
        raise ValueError(f"{group} final checkpoint SHA256 mismatch")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("epoch") != 5 or checkpoint.get("config", {}).get("group") != group:
        raise ValueError(f"{group} is not the selected epoch-5 checkpoint")
    state = checkpoint.get("model")
    if not isinstance(state, dict):
        raise ValueError(f"{group} checkpoint lacks model state")
    target = model.state_dict()
    if set(state) != set(target):
        raise ValueError(f"{group} state dict keys differ")
    for name, tensor in state.items():
        if tensor.shape != target[name].shape or tensor.dtype != target[name].dtype:
            raise ValueError(f"{group} state tensor mismatch: {name}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{group} state tensor is nonfinite: {name}")
    model.load_state_dict(state, strict=True)
    loaded = model.state_dict()
    # Compare exact checkpoint values on CPU after strict load, regardless of model device.
    if any(not torch.equal(state[name], loaded[name].detach().cpu()) for name in state):
        raise ValueError(f"{group} strict load did not preserve values")
    return {"checkpoint": str(path), "checkpoint_sha256": expected_hash, "epoch": 5}


def empty_sufficient():
    metrics = m.frozen("metrics").MetricAccumulator()
    return metrics


def restore_absolute(rows):
    total = empty_sufficient()
    for row in rows:
        total.count += int(row["valid_pixel_count"])
        total.abs_error_sum += float(row["abs_error_sum_m"])
        total.squared_error_sum += float(row["squared_error_sum_m2"])
        total.abs_relative_sum += float(row["abs_relative_error_sum"])
        total.pred_sum += float(row["prediction_sum_m"])
        total.target_sum += float(row["gt_sum_m"])
        total.pred_sq_sum += float(row["prediction_squared_sum_m2"])
        total.target_sq_sum += float(row["gt_squared_sum_m2"])
        total.cross_sum += float(row["prediction_gt_cross_sum_m2"])
        total.local_pair_count += int(row["full14_pair_count"])
        total.local_abs_delta_sum += float(row["full14_abs_error_sum_m"])
    return total


def row_metrics(row):
    n = int(row["valid_pixel_count"])
    p = int(row["full14_pair_count"])
    large = int(row["gt_ge20_pair_count"])
    return {
        "valid_pixels": n,
        "mae_m": float(row["abs_error_sum_m"]) / n if n else None,
        "rmse_m": math.sqrt(float(row["squared_error_sum_m2"]) / n) if n else None,
        "abs_rel": float(row["abs_relative_error_sum"]) / n if n else None,
        "bias_m": (float(row["prediction_sum_m"]) - float(row["gt_sum_m"])) / n if n else None,
        "full14_pair_count": p,
        "full14_local_mae_m": float(row["full14_abs_error_sum_m"]) / p if p else None,
        "gt_ge20_pair_count": large,
        "gt_ge20_local_mae_m": float(row["gt_ge20_abs_error_sum_m"]) / large if large else None,
        "gt_ge20_amplitude_ratio": (
            float(row["gt_ge20_abs_prediction_sum_m"]) / float(row["gt_ge20_abs_gt_sum_m"])
            if float(row["gt_ge20_abs_gt_sum_m"]) > 0 else None
        ),
        "gt_ge20_sign_match_rate": (
            int(row["gt_ge20_sign_match_count"]) / int(row["gt_ge20_nonzero_sign_count"])
            if int(row["gt_ge20_nonzero_sign_count"]) else None
        ),
    }


def load_existing_progress(progress: Path, group: str):
    rows = []
    if not progress.exists():
        return rows
    with progress.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("model_id") != group:
                raise ValueError("Progress file contains a different model")
            rows.append(row)
    indices = [int(row["image_index"]) for row in rows]
    if indices != list(range(len(indices))):
        raise ValueError("Progress is not a contiguous canonical image prefix")
    return rows


def eval_group(model, group, rows, rgb_root, depth_root, pair_manifest, outdir, reference_root, metadata, device):
    progress = outdir / f"{group}_per_image.jsonl"
    per_image = load_existing_progress(progress, group)
    if len(per_image) > len(rows):
        raise ValueError(f"{group} progress exceeds validation image count")
    expected = json.loads(pair_manifest.read_text(encoding="utf-8"))["images"]
    metric_module = m.frozen("metrics")
    model.eval()
    with progress.open("a", encoding="utf-8") as stream:
        for i in range(len(per_image), len(rows)):
            row = rows[i]
            bgr = cv2.imread(str(Path(rgb_root) / row["image_relpath"]), cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(row["image_relpath"])
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            gt = np.asarray(tifffile.imread(Path(depth_root) / row["depth_relpath"]), dtype=np.float32)
            if rgb.shape[:2] != gt.shape:
                raise ValueError(f"RGB/GT shape mismatch: {row['image_relpath']}")
            with torch.inference_mode():
                native = model(pixel_values=m.preprocess(rgb, 518)[None].to(device)).predicted_depth[0].float().cpu().numpy()
            pred = m.restore(native, gt.shape)
            m.validate_prediction(pred, gt)
            mask = m.valid_gt(gt)
            acc = metric_module.MetricAccumulator()
            acc.update(torch.from_numpy(pred), torch.from_numpy(np.where(mask, gt, 0.0)))
            stats, coordinates = m.multiscale(pred, gt, i, scales=(14,))
            expected_pairs = [item for item in expected[i]["pairs"] if item["stride"] == 14]
            if coordinates != expected_pairs:
                raise ValueError(f"Fixed 14px sample hash mismatch at image {i}")
            large = stats[14]["gt_ge20cm"].legacy.total
            record = {
                "model_id": group,
                "image_index": i,
                "image_relpath": row["image_relpath"],
                "crop": row["crop"],
                "view": row["view"],
                "valid_pixel_count": acc.count,
                "abs_error_sum_m": acc.abs_error_sum,
                "squared_error_sum_m2": acc.squared_error_sum,
                "abs_relative_error_sum": acc.abs_relative_sum,
                "prediction_sum_m": acc.pred_sum,
                "gt_sum_m": acc.target_sum,
                "prediction_squared_sum_m2": acc.pred_sq_sum,
                "gt_squared_sum_m2": acc.target_sq_sum,
                "prediction_gt_cross_sum_m2": acc.cross_sum,
                "full14_pair_count": acc.local_pair_count,
                "full14_abs_error_sum_m": acc.local_abs_delta_sum,
                "gt_ge20_pair_count": int(large[0]),
                "gt_ge20_abs_error_sum_m": float(large[1]),
                "gt_ge20_abs_gt_sum_m": float(large[2]),
                "gt_ge20_abs_prediction_sum_m": float(large[3]),
                "gt_ge20_nonzero_sign_count": int(large[4]),
                "gt_ge20_sign_match_count": int(large[5]),
            }
            record.update(row_metrics(record))
            if not math.isfinite(record["mae_m"]) or not math.isfinite(record["gt_ge20_local_mae_m"] or 0.0):
                raise FloatingPointError(f"Nonfinite metric at {group}/{i}")
            stream.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            per_image.append(record)
            if (i + 1) % 16 == 0 or i + 1 == len(rows):
                print(f"{group}: evaluated {i+1}/{len(rows)} images", flush=True)
            del pred, native, rgb, gt, bgr
            if str(device).startswith("cuda") and (i + 1) % 8 == 0:
                torch.cuda.empty_cache()

    aggregate = restore_absolute(per_image)
    absolute = aggregate.result()
    primary_pairs = sum(int(row["gt_ge20_pair_count"]) for row in per_image)
    primary_error_sum = sum(float(row["gt_ge20_abs_error_sum_m"]) for row in per_image)
    primary_gt_abs_sum = sum(float(row["gt_ge20_abs_gt_sum_m"]) for row in per_image)
    primary_pred_abs_sum = sum(float(row["gt_ge20_abs_prediction_sum_m"]) for row in per_image)
    sign_nonzero = sum(int(row["gt_ge20_nonzero_sign_count"]) for row in per_image)
    sign_correct = sum(int(row["gt_ge20_sign_match_count"]) for row in per_image)
    reference_path = Path(reference_root) / group / ("validation_epoch_0020.json" if group == "E0" else "validation_epoch_0005.json")
    reference = json.loads(reference_path.read_text(encoding="utf-8"))["validation"]
    reference_absolute = reference["absolute"]
    comparisons = {}
    for key in ("valid_pixels", "mae_m", "rmse_m", "abs_rel", "pearson_r", "local_pair_count", "local_depth_delta_mae_m"):
        actual_value, expected_value = absolute[key], reference_absolute[key]
        if key in ("valid_pixels", "local_pair_count"):
            ok = int(actual_value) == int(expected_value)
        else:
            ok = actual_value is not None and expected_value is not None and abs(float(actual_value) - float(expected_value)) <= 1e-7
        comparisons[key] = {"actual": actual_value, "reference": expected_value, "passed": ok}
    reference_large = next(x for x in reference["sampled"] if x["scope"] == "all" and x["stride_px"] == 14 and x["group"] == "gt_ge20cm")
    comparisons["gt_ge20_pair_count"] = {
        "actual": primary_pairs, "reference": int(reference_large["pair_count"]),
        "passed": primary_pairs == int(reference_large["pair_count"]),
    }
    comparisons["gt_ge20_local_mae_m"] = {
        "actual": primary_error_sum / primary_pairs if primary_pairs else None,
        "reference": reference_large["delta_mae_m"],
        "passed": primary_pairs == int(reference_large["pair_count"]) and abs(primary_error_sum / primary_pairs - float(reference_large["delta_mae_m"])) <= 1e-7,
    }
    if not all(item["passed"] for item in comparisons.values()):
        raise RuntimeError(f"{group} B0 re-evaluation differs from saved validation: {comparisons}")
    summary = {
        "model_id": group, "epoch": metadata["epoch"],
        "checkpoint": metadata["checkpoint"], "checkpoint_sha256": metadata["checkpoint_sha256"],
        "reference_validation_json_sha256": sha256(reference_path),
        "image_count": len(per_image), "absolute": absolute,
        "full14_local_mae_m": aggregate.local_abs_delta_sum / aggregate.local_pair_count,
        "gt_ge20cm_14px": {
            "pair_count": primary_pairs,
            "absolute_error_sum_m": primary_error_sum,
            "local_mae_m": primary_error_sum / primary_pairs if primary_pairs else None,
            "absolute_gt_delta_sum_m": primary_gt_abs_sum,
            "absolute_pred_delta_sum_m": primary_pred_abs_sum,
            "amplitude_ratio": primary_pred_abs_sum / primary_gt_abs_sum if primary_gt_abs_sum else None,
            "nonzero_gt_pair_count": sign_nonzero,
            "sign_match_count": sign_correct,
            "sign_match_rate": sign_correct / sign_nonzero if sign_nonzero else None,
        },
        "saved_validation_comparison": comparisons,
        "pair_manifest_sha256": sha256(pair_manifest),
    }
    atomic_json(outdir / f"{group}_summary.json", summary)
    atomic_csv(outdir / f"{group}_per_image.csv", per_image)
    print(json.dumps({"group": group, "absolute": absolute, "gt_ge20cm_14px": summary["gt_ge20cm_14px"]}, ensure_ascii=False), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--rgb-root", type=Path, required=True)
    parser.add_argument("--depth-root", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This final full-validation re-evaluation requires the verified CUDA device")
    project = args.project_root.resolve()
    out = args.output.resolve()
    if out.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite existing evaluation output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    run_root = project / "runs"
    rows = m.validation_rows(args.validation_manifest)
    if len(rows) != 128:
        raise ValueError("Only canonical Validation128 is allowed")
    if m.sha256(args.pair_manifest) != "bcb56eab03bec4a44f0626da0f4814876fae67774e536e00cd9f1ad16b1f5728":
        raise ValueError("Fixed pair manifest SHA256 mismatch")
    if sha256(args.base_checkpoint / "model.safetensors") != BASE_MODEL_SHA256:
        raise ValueError("Base model SHA256 mismatch")
    if sha256(args.base_checkpoint / "config.json") != BASE_CONFIG_SHA256:
        raise ValueError("Base config SHA256 mismatch")
    code_hash = sha256(HERE)
    signature = {
        "method": "B0全体518px; frozen preprocess/restore/valid_gt/fixed pair sampler; final model only",
        "groups": list(GROUPS), "validation_images": 128, "scales": list(SCALES),
        "validation_manifest_sha256": sha256(args.validation_manifest),
        "pair_manifest_sha256": sha256(args.pair_manifest),
        "base_model_sha256": BASE_MODEL_SHA256, "base_config_sha256": BASE_CONFIG_SHA256,
        "final_checkpoint_sha256": {
            g: (BASE_MODEL_SHA256 if g == "E0" else sha256(run_root / g / "epoch_0005.pt"))
            for g in GROUPS
        },
        "code_sha256": code_hash,
    }
    signature_path = out / "evaluation_signature.json"
    if signature_path.exists():
        previous = json.loads(signature_path.read_text(encoding="utf-8"))
        if previous != signature:
            previous_core = dict(previous)
            current_core = dict(signature)
            previous_code = previous_core.pop("code_sha256", None)
            current_code = current_core.pop("code_sha256", None)
            if not args.resume or previous_core != current_core or not previous_code or not current_code:
                raise ValueError("Resume signature mismatch outside the code revision")
            audit_path = out / "resume_code_revision_audit.json"
            if audit_path.exists():
                raise FileExistsError(f"Refusing to overwrite resume audit: {audit_path}")
            atomic_json(audit_path, {
                "reason": "Strict-load value comparison now moves loaded tensors to CPU before exact equality check; evaluation definition, data, pairs, and checkpoints are unchanged.",
                "previous_code_sha256": previous_code,
                "resumed_code_sha256": current_code,
                "prior_signature_preserved": True,
                "resume_authorized_by_cli_flag": True,
            })
        if not args.resume:
            raise FileExistsError(f"Existing evaluation signature: {signature_path}")
    else:
        atomic_json(signature_path, signature)
    model, attention_audit = new_model(args.base_checkpoint, args.device)
    if not (out / "model_load_audit.json").exists():
        atomic_json(out / "model_load_audit.json", attention_audit)
    summaries = []
    for group in GROUPS:
        model_meta = load_group(model, group, args.base_checkpoint, run_root)
        group_summary_path = out / f"{group}_summary.json"
        if group_summary_path.exists():
            if not args.resume:
                raise FileExistsError(group_summary_path)
            summaries.append(json.loads(group_summary_path.read_text(encoding="utf-8")))
            continue
        summaries.append(eval_group(model, group, rows, args.rgb_root, args.depth_root,
                                    args.pair_manifest, out, run_root, model_meta, args.device))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    all_rows = []
    for group in GROUPS:
        with (out / f"{group}_per_image.csv").open("r", encoding="utf-8", newline="") as stream:
            all_rows.extend(csv.DictReader(stream))
    atomic_csv(out / "per_image_primary.csv", all_rows)
    atomic_json(out / "final_summary.json", {
        "experiment_id": "local_huber_20261105",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_only": True,
        "independent_test_used": False,
        "signature": signature,
        "attention_load_audit": attention_audit,
        "models": summaries,
        "per_image_csv_sha256": sha256(out / "per_image_primary.csv"),
    })
    print(f"FINAL_EVALUATION_COMPLETE output={out} rows={len(all_rows)}", flush=True)


if __name__ == "__main__":
    main()

\r\n