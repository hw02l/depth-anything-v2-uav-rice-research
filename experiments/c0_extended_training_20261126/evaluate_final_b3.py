from __future__ import annotations

"""Evaluate the fixed B3 fusion on the C0 epoch20 continuation checkpoint."""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NATIVE = ROOT / "experiments" / "native_resolution_patch_training_20261111"
BEST = ROOT / "experiments" / "best_model_inference_combination_20261122"
sys.path.insert(0, str(NATIVE))
sys.path.insert(0, str(BEST))

import train_patch as native  # noqa: E402
import run_b3 as b3  # noqa: E402
from continue_c0 import (  # noqa: E402
    assert_no_new_gpu_compute_processes,
    check_gpu_receipt,
    check_paths_inside_gpu_work,
    git_revision,
)
from data_patch import rgb_unit_float_to_uint8  # noqa: E402
from patch_engine import canonical_signature, file_sha256  # noqa: E402


def write_json(path: Path, value: dict) -> None:
    tmp = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    with tmp.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_epoch20_model(base_dir: Path, checkpoint: Path, device: str):
    actual_sha = file_sha256(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if int(payload.get("epoch", -1)) != 20:
        raise ValueError("B3 evaluation requires the fixed epoch20 checkpoint")
    if int(payload.get("global_step", -1)) < 960:
        raise ValueError("Checkpoint global_step moved backwards")
    signature_payload = payload.get("signature_payload")
    if not isinstance(signature_payload, dict) or signature_payload.get("group") != "C0":
        raise ValueError("Checkpoint is not signed as the continued C0 run")
    if payload.get("checkpoint_signature") != canonical_signature(signature_payload):
        raise ValueError("Checkpoint signature verification failed")
    progress_path = checkpoint.parent / "progress.json"
    completion_path = checkpoint.parent / "run_complete.json"
    if not completion_path.is_file():
        raise ValueError("Completed-run record is missing")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if (
        completion.get("status") != "complete"
        or int(completion.get("epoch", -1)) != 20
        or int(completion.get("global_step", -1)) != int(payload["global_step"])
        or completion.get("checkpoint_sha256") != actual_sha
    ):
        raise ValueError("Completed-run record does not match the epoch20 checkpoint")
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if (
            progress.get("status") != "complete"
            or int(progress.get("epoch", -1)) != 20
            or int(progress.get("global_step", -1)) != int(payload["global_step"])
        ):
            raise ValueError("Completed-run progress state does not match the epoch20 checkpoint")
        progress_sha = progress.get("checkpoint_sha256")
        if progress_sha is not None and progress_sha != actual_sha:
            raise ValueError("Optional progress checkpoint hash differs from epoch20 checkpoint")

    model, base_audit = native.load_model(base_dir, device)
    state = payload.get("model")
    target = model.state_dict()
    if not isinstance(state, dict) or set(state) != set(target):
        raise ValueError("Epoch20 state_dict keys differ from the model")
    for key, tensor in state.items():
        if tensor.shape != target[key].shape or tensor.dtype != target[key].dtype:
            raise ValueError(f"Epoch20 state_dict shape/dtype mismatch: {key}")
        if tensor.is_floating_point() and not torch.isfinite(tensor).all().item():
            raise FloatingPointError(f"Non-finite checkpoint tensor: {key}")
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("Strict epoch20 state_dict load was not exact")
    model.eval()
    audit = {
        "checkpoint_sha256": actual_sha,
        "completion_record_sha256": file_sha256(completion_path),
        "epoch": 20,
        "global_step": int(payload["global_step"]),
        "model_key_count": len(state),
        "strict_load": True,
        "missing_keys": [],
        "unexpected_keys": [],
        "base_model_load_audit": base_audit,
        "run_signature": payload["checkpoint_signature"],
    }
    del payload, state
    return model, audit


def run(args) -> dict:
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Final B3 evaluation requires CUDA")
    receipt_audit = check_gpu_receipt(
        args.preflight_receipt, args.gpu_permission_record,
        args.gpu_work_root, args.checkpoint,
        reserved_checkpoint_copies=0, output_reserve_bytes=64 * 1024**2,
    )
    check_paths_inside_gpu_work([
        ("checkpoint", args.checkpoint), ("base model", args.base_model_dir),
        ("validation manifest", args.validation_manifest), ("pair manifest", args.pair_manifest),
        ("RGB root", args.rgb_root), ("depth root", args.depth_root),
        ("preflight receipt", args.preflight_receipt),
        ("GPU permission record", args.gpu_permission_record), ("output", args.output),
    ], args.gpu_work_root)
    if args.output.exists():
        raise FileExistsError(f"Existing output is preserved: {args.output}")
    args.output.mkdir(parents=True, exist_ok=False)

    rows = native.read_manifest(args.validation_manifest)
    native.verify_manifest(
        rows, args.validation_manifest,
        ROOT / "configs/uav3dcrop_verified/validation.csv", 128, "validation",
    )
    native.verify_files(rows, args.rgb_root, args.depth_root)
    if b3.sha256(args.pair_manifest) != b3.PAIR_SHA256:
        raise ValueError("Fixed pair manifest differs from the approved hash")
    pair_images = json.loads(args.pair_manifest.read_text(encoding="utf-8"))["images"]
    if len(pair_images) != 128:
        raise ValueError("Expected exactly 128 fixed pair-manifest entries")

    model, load_audit = load_epoch20_model(args.base_model_dir, args.checkpoint, args.device)
    write_json(args.output / "provenance.json", {
        "started_unix_time": time.time(),
        "command": sys.argv,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_state_load_audit": load_audit,
        "git": git_revision(),
        "code_sha256": {
            str(path.relative_to(ROOT)): file_sha256(path)
            for path in (HERE / "evaluate_final_b3.py", BEST / "run_b3.py",
                         NATIVE / "train_patch.py", NATIVE / "data_patch.py",
                         NATIVE / "patch_engine.py")
        },
        "validation_manifest_sha256": file_sha256(args.validation_manifest),
        "pair_manifest_sha256": file_sha256(args.pair_manifest),
        "rgb_input_contract": "uint8 RGB HxWx3 [0,255]; shared B3 preprocessing normalizes once",
        "gt_mask": "finite and 0 < depth <= 65m",
        "gpu_preflight": receipt_audit,
        "gpu_name": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "fusion_method": "B3 same-checkpoint global518 + Hann-blended original-coordinate 2000px tiles; box215 reflect101 residual",
        "independent_test319": False,
        "training_or_optimizer_update": False,
    })

    absolute_rows, local_rows, runtime_rows = [], [], []
    expected_other_pids = set(receipt_audit["other_compute_pids_authorized"])
    expected_other_memory = receipt_audit["other_compute_memory_mib_at_preflight"]
    for index, row in enumerate(rows):
        assert_no_new_gpu_compute_processes(expected_other_pids, expected_other_memory)
        rgb, gt = native.load_pair(row, args.rgb_root, args.depth_root)
        rgb_u8 = rgb_unit_float_to_uint8(rgb)
        if rgb_u8.dtype != np.uint8 or rgb_u8.ndim != 3 or rgb_u8.shape[2] != 3:
            raise TypeError("RGB input contract failed before B3 inference")
        if int(rgb_u8.max()) <= 1:
            raise ValueError("RGB appears to be a black [0,1]-to-uint8 cast")
        if rgb_u8.shape[:2] != gt.shape:
            raise ValueError(f"RGB/GT coordinate mismatch at image {index}")

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        with torch.inference_mode():
            prediction, fusion_audit = b3.predict_b3(model, rgb_u8, args.device)
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        if prediction.shape != gt.shape or not np.isfinite(prediction).all():
            raise FloatingPointError(f"B3 output invalid at image {index}")

        valid = native.m.valid_gt(gt)
        absolute = b3.absolute_stats(prediction, gt, valid)
        dense = b3.dense14_stats(prediction, gt, valid)
        sampled, _ = b3.fixed_pair_stats(prediction, gt, valid, index, pair_images)
        absolute_rows.append({
            "condition": "C0-epoch20-B3", "image_index": index,
            "image_relpath": row["image_relpath"], "crop": row["crop"], "view": row["view"],
            "rgb_dtype": str(rgb_u8.dtype), "rgb_min": int(rgb_u8.min()), "rgb_max": int(rgb_u8.max()),
            **absolute, **dense,
            "fusion_nonpositive_count": fusion_audit["fusion_nonpositive_count"],
            "fusion_above_80m_count": fusion_audit["fusion_above_80m_count"],
        })
        local_rows.extend({
            "condition": "C0-epoch20-B3", "image_relpath": row["image_relpath"],
            "crop": row["crop"], "view": row["view"], **item,
        } for item in sampled)
        runtime_rows.append({
            "condition": "C0-epoch20-B3", "image_index": index,
            "seconds_global_tiles_fusion": elapsed,
            "tile_count": fusion_audit["tile_count"],
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            **fusion_audit,
        })
        if (index + 1) % 8 == 0:
            print(json.dumps({"images_done": index + 1, "total": 128,
                              "elapsed_seconds": elapsed,
                              "rmse_image_m": absolute["rmse_m"]}), flush=True)
            torch.cuda.empty_cache()
            assert_no_new_gpu_compute_processes(expected_other_pids, expected_other_memory)
        del prediction, rgb, rgb_u8, gt

    summary = b3.pooled_metrics(absolute_rows, local_rows)
    summary.update({
        "condition": "C0-epoch20-B3",
        "runtime_seconds": sum(float(row["seconds_global_tiles_fusion"]) for row in runtime_rows),
        "checkpoint_sha256": load_audit["checkpoint_sha256"],
        "validation_manifest_sha256": file_sha256(args.validation_manifest),
        "pair_manifest_sha256": file_sha256(args.pair_manifest),
        "fusion_out_of_range_nonpositive_count": sum(int(row["fusion_nonpositive_count"]) for row in runtime_rows),
        "fusion_out_of_range_above80_count": sum(int(row["fusion_above_80m_count"]) for row in runtime_rows),
    })
    write_csv(args.output / "per_image_absolute_sufficient.csv", absolute_rows)
    write_csv(args.output / "per_image_local_sufficient.csv", local_rows)
    write_csv(args.output / "runtime.csv", runtime_rows)
    write_json(args.output / "summary.json", summary)
    write_json(args.output / "run_complete.json", {
        "status": "complete", "images": len(absolute_rows), "summary": summary,
        "checkpoint_state_load_audit": load_audit,
        "nonfinite_or_out_of_range_values_were_not_corrected": True,
    })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-model-dir", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--rgb-root", type=Path, required=True)
    parser.add_argument("--depth-root", type=Path, required=True)
    parser.add_argument("--preflight-receipt", type=Path, required=True)
    parser.add_argument("--gpu-permission-record", type=Path, required=True)
    parser.add_argument("--gpu-work-root", type=Path, default=Path.home() / "gpu_work")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    print(json.dumps(run(parser.parse_args()), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
