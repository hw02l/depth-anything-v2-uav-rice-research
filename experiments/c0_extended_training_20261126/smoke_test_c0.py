from __future__ import annotations

"""Run one transient C0 FP16 update; save audit only, never model weights."""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NATIVE = ROOT / "experiments" / "native_resolution_patch_training_20261111"
sys.path.insert(0, str(NATIVE))

import train_patch as native  # noqa: E402
import continue_c0 as continuation  # noqa: E402
from patch_engine import file_sha256, restore_rng  # noqa: E402


def run(args) -> dict:
    receipt = continuation.check_gpu_receipt(
        args.preflight_receipt, args.gpu_permission_record,
        args.gpu_work_root, args.initial_checkpoint,
        reserved_checkpoint_copies=16,
    )
    continuation.check_paths_inside_gpu_work([
        ("checkpoint", args.initial_checkpoint), ("base model", args.base_model_dir),
        ("train manifest", args.train_manifest), ("RGB root", args.rgb_root),
        ("depth root", args.depth_root), ("preflight receipt", args.preflight_receipt),
        ("GPU permission record", args.gpu_permission_record), ("audit output", args.audit_output),
    ], args.gpu_work_root)
    checkpoint_sha = file_sha256(args.initial_checkpoint)
    state5 = torch.load(args.initial_checkpoint, map_location="cpu", weights_only=False)
    state_audit = continuation.audit_epoch5(state5)
    model, model_audit = continuation.load_model_from_epoch5(
        args.base_model_dir, state5, "cuda")
    model.train()
    optimizer = native.make_optimizer(model)
    optimizer.load_state_dict(state5["optimizer"])
    continuation.reset_group_lrs(optimizer)
    native._copy_optimizer_state_to_device(optimizer, "cuda")
    scheduler = continuation.make_continuation_scheduler(optimizer)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    scaler.load_state_dict(state5["scaler"])
    restore_rng(state5["rng"])

    rows = native.read_manifest(args.train_manifest)
    native.verify_manifest(rows, args.train_manifest,
                           ROOT / "configs/uav3dcrop_verified/train.csv", 384, "train")
    native.verify_files(rows, args.rgb_root, args.depth_root)
    dataset = native.NativePatchDataset(
        rows, args.rgb_root, args.depth_root, "C0",
        repeats=2, seed=42, training=True,
    )
    own_processes = set(receipt["other_compute_pids_authorized"])
    other_memory = receipt["other_compute_memory_mib_at_preflight"]
    started = time.monotonic()
    result = native._train_epoch(
        model, dataset, optimizer, scheduler, scaler,
        native.m.frozen("losses").silog_loss,
        epoch=6, seed=42, device="cuda", amp=True, accumulation=4,
        max_updates=1,
        progress_callback=lambda *_: continuation.assert_no_new_gpu_compute_processes(
            own_processes, other_memory),
    )
    torch.cuda.synchronize()
    continuation.assert_no_new_gpu_compute_processes(own_processes, other_memory)
    if result["optimizer_updates"] != 1 or result["amp_skips"] != 0:
        raise RuntimeError(f"Expected one successful transient update, got {result}")
    if file_sha256(args.initial_checkpoint) != checkpoint_sha:
        raise RuntimeError("The protected epoch5 checkpoint changed during smoke test")
    return {
        "status": "passed",
        "test": "one transient C0 FP16 optimizer update, no weights saved",
        "training_checkpoint_modified": False,
        "initial_checkpoint_sha256": checkpoint_sha,
        "checkpoint_state_audit": state_audit,
        "strict_model_load_audit": model_audit,
        "train_epoch_result": result,
        "gpu_name": torch.cuda.get_device_name(0),
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "elapsed_seconds": time.monotonic() - started,
        "gpu_preflight_audit": receipt,
        "command": sys.argv,
        "code_sha256": {
            str(path.relative_to(ROOT)): file_sha256(path)
            for path in (HERE / "smoke_test_c0.py", HERE / "continue_c0.py",
                         NATIVE / "train_patch.py", NATIVE / "patch_engine.py",
                         NATIVE / "data_patch.py",
                         ROOT / "experiments" / "metric_outdoor_full_20261014_s42_518_cap80_gt65"
                         / "source_full_v1" / "losses.py")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--base-model-dir", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--rgb-root", type=Path, required=True)
    parser.add_argument("--depth-root", type=Path, required=True)
    parser.add_argument("--preflight-receipt", type=Path, required=True)
    parser.add_argument("--gpu-permission-record", type=Path, required=True)
    parser.add_argument("--gpu-work-root", type=Path, default=Path.home() / "gpu_work")
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    if args.audit_output.exists():
        raise FileExistsError(f"Smoke audit is preserved: {args.audit_output}")
    args.audit_output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
