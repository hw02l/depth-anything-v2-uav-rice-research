from __future__ import annotations

"""Continue the fixed C0 run from epoch 5 through epoch 20, without retraining from the base model."""

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
NATIVE = ROOT / "experiments" / "native_resolution_patch_training_20261111"
sys.path.insert(0, str(NATIVE))

import train_patch as native  # noqa: E402
from patch_engine import (  # noqa: E402
    canonical_signature,
    capture_rng,
    cpu_model_state,
    file_sha256,
    latest_epoch,
    load_epoch,
    restore_rng,
    save_epoch,
)

EXPECTED_EPOCH5_SHA256 = "2fd190727c7a5124c56c1e0ea2dd8cb09c6cb62a0f92b34fce97ad394281b35a"
EXPECTED_MODEL_KEYS = 287
EPOCH_START = 6
EPOCH_END = 20
GROUP_LRS = (1e-6, 1e-5)
TOTAL_SCHEDULER_UPDATES = 2880
MIN_SHARED_FREE_VRAM_BYTES = 5 * 1024**3
LOCAL_QUOTA_BYTES = 50 * 1024**3


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    with tmp.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def code_hashes() -> dict[str, str]:
    paths = [
        HERE / "continue_c0.py",
        NATIVE / "train_patch.py",
        NATIVE / "patch_engine.py",
        NATIVE / "data_patch.py",
        NATIVE / "evaluate_patch.py",
        ROOT / "src" / "uav_metric_research" / "multiscale.py",
    ]
    return {str(path.relative_to(ROOT)): file_sha256(path) for path in paths}


def git_revision() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", "-C", str(ROOT), *args],
                check=True, capture_output=True, text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    return {
        "head": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "status_porcelain": run("status", "--porcelain"),
    }


def assert_finite_tree(value: Any, name: str) -> None:
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() and not torch.isfinite(value).all().item():
            raise FloatingPointError(f"Non-finite tensor in {name}")
    elif isinstance(value, dict):
        for key, item in value.items():
            assert_finite_tree(item, f"{name}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_finite_tree(item, f"{name}[{index}]")


def _gpu_compute_pids() -> set[int]:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=20,
    )
    pids = set()
    for line in result.stdout.splitlines():
        token = line.strip().split(",", 1)[0].strip()
        if token.isdigit():
            pids.add(int(token))
    return pids


def check_gpu_receipt(path: Path, permission_path: Path,
                       gpu_work_root: Path, epoch5_checkpoint: Path,
                       *, reserved_checkpoint_copies: int = 16,
                       output_reserve_bytes: int = 256 * 1024**2) -> dict[str, Any]:
    """Validate a fresh GPU receipt, allowing only explicitly authorized safe sharing."""
    receipt = json.loads(path.read_text(encoding="utf-8"))
    permission = json.loads(permission_path.read_text(encoding="utf-8"))
    if permission.get("status") != "granted" or "explicitly authorizes starting now" not in permission.get("source", ""):
        raise RuntimeError("Explicit permission record for shared GPU use is missing")
    age = time.time() - float(receipt.get("unix_time", 0))
    if not 0 <= age <= 300:
        raise RuntimeError(f"GPU preflight receipt is stale ({age:.1f}s)")
    if receipt.get("hostname", "").split(".")[0].lower() != "gpgpu4":
        raise RuntimeError("Receipt is not from GPGPU4")
    if receipt.get("work_root_realpath") != str(gpu_work_root.resolve()):
        raise RuntimeError("Receipt work root does not match the requested local GPU directory")
    if receipt.get("singularity_version") != "4.2.2":
        raise RuntimeError("Unexpected Singularity runtime; stop without bypass")
    checks = receipt.get("checks", {})
    failed = [name for name, passed in checks.items()
              if not passed and name != "no_gpu_compute_processes"]
    if failed:
        raise RuntimeError(f"GPGPU4 preflight checks failed: {failed}")

    processes = receipt.get("compute_processes", [])
    if len(processes) > 1:
        raise RuntimeError("Shared GPU receipt lists more than one other compute process")
    free_vram = int(receipt.get("gpu_free_bytes", 0))
    if processes:
        if int(receipt.get("gpu_utilization_percent") or 100) > 5:
            raise RuntimeError("GPU utilization exceeds the shared-use idle threshold")
        if free_vram < MIN_SHARED_FREE_VRAM_BYTES:
            raise RuntimeError("Shared-use preflight requires at least 5GiB free VRAM")

    expected_pids = {int(line.strip().split(",", 1)[0].strip())
                     for line in processes
                     if line.strip().split(",", 1)[0].strip().isdigit()}
    current_pids = _gpu_compute_pids()
    if current_pids != expected_pids:
        raise RuntimeError("GPU compute process set changed since preflight; refresh the receipt")

    local_root = gpu_work_root.resolve()
    usage = subprocess.run(["du", "-sb", str(local_root)], check=True,
                           capture_output=True, text=True, timeout=300)
    used_bytes = int(usage.stdout.split()[0])
    checkpoint_bytes = epoch5_checkpoint.stat().st_size
    # The training run reserves 16 complete checkpoints (epoch5 restart plus
    # epochs6–20); evaluation callers may pass zero and reserve only tables.
    projected_bytes = (used_bytes + reserved_checkpoint_copies * checkpoint_bytes
                       + output_reserve_bytes)
    if projected_bytes > LOCAL_QUOTA_BYTES:
        raise RuntimeError(
            f"Projected local use {projected_bytes / 1024**3:.2f}GiB exceeds 50GiB quota"
        )
    df = subprocess.run(["df", "-PB1", str(local_root)], check=True,
                        capture_output=True, text=True, timeout=20)
    disk_rows = [line.split() for line in df.stdout.splitlines()[1:] if line.split()]
    if len(disk_rows) != 1 or int(disk_rows[0][3]) < 10 * 1024**3:
        raise RuntimeError("Less than 10GiB filesystem space is available")
    return {
        "receipt_sha256": file_sha256(path),
        "permission_record_sha256": file_sha256(permission_path),
        "age_seconds": age,
        "gpu": receipt["gpu_name"],
        "other_compute_pids_authorized": sorted(expected_pids),
        "free_vram_bytes_at_preflight": free_vram,
        "local_work_bytes": used_bytes,
        "projected_work_bytes": projected_bytes,
        "free_filesystem_bytes": int(disk_rows[0][3]),
        "shared_gpu_permission": "user-reported permission; preflight snapshot unchanged",
    }


def assert_no_new_gpu_compute_processes(expected_other_pids: set[int]) -> None:
    current = _gpu_compute_pids()
    own_pid = os.getpid()
    current.discard(own_pid)
    if current != expected_other_pids:
        raise RuntimeError(
            f"GPU compute process set changed during run; expected other={sorted(expected_other_pids)}, "
            f"current other={sorted(current)}. Stop and resume from the last epoch checkpoint."
        )


def audit_epoch5(state: dict[str, Any]) -> dict[str, Any]:
    required = {
        "epoch", "global_step", "model", "optimizer", "scheduler",
        "scaler", "rng", "history", "signature_payload", "checkpoint_signature",
    }
    missing = sorted(required - set(state))
    if missing:
        raise ValueError(f"Epoch5 checkpoint lacks required state: {missing}")
    if int(state["epoch"]) != 5 or int(state["global_step"]) != 960:
        raise ValueError("Input checkpoint must be C0 epoch5 at global_step 960")
    if not isinstance(state["model"], dict) or len(state["model"]) != EXPECTED_MODEL_KEYS:
        raise ValueError(f"Expected {EXPECTED_MODEL_KEYS} model keys")
    if state.get("checkpoint_signature") != canonical_signature(state["signature_payload"]):
        raise ValueError("Input checkpoint signature is invalid")
    if not isinstance(state["optimizer"], dict) or len(state["optimizer"].get("param_groups", [])) != 2:
        raise ValueError("C0 checkpoint must contain both AdamW parameter groups")
    if not state["optimizer"].get("state"):
        raise ValueError("AdamW moment state is empty")
    if not isinstance(state["scaler"], dict):
        raise ValueError("GradScaler state is missing")
    if not isinstance(state["rng"], dict) or not {
        "python", "numpy", "torch_cpu", "torch_cuda"
    }.issubset(state["rng"]):
        raise ValueError("Full RNG state is missing")
    if not isinstance(state["history"], list) or len(state["history"]) < 5:
        raise ValueError("Epoch1–5 history is missing")
    signature_payload = state["signature_payload"]
    if signature_payload.get("group") != "C0":
        raise ValueError("Input checkpoint signature is not for C0")
    if signature_payload.get("epoch") not in (None, 5):
        raise ValueError("Input checkpoint signature has an unexpected epoch")
    assert_finite_tree(state["model"], "model")
    assert_finite_tree(state["optimizer"], "optimizer")
    return {
        "epoch": int(state["epoch"]),
        "global_step": int(state["global_step"]),
        "model_key_count": len(state["model"]),
        "optimizer_parameter_groups": len(state["optimizer"]["param_groups"]),
        "optimizer_state_entries": len(state["optimizer"]["state"]),
        "scheduler_keys": len(state["scheduler"]),
        "scaler_keys": len(state["scaler"]),
        "rng_fields": sorted(state["rng"]),
        "model_and_optimizer_finite": True,
        "signature": state["checkpoint_signature"],
    }


def reset_group_lrs(optimizer: torch.optim.Optimizer,
                    lrs: tuple[float, float] = GROUP_LRS) -> None:
    if len(optimizer.param_groups) != 2:
        raise ValueError("Expected encoder and neck/head optimizer groups")
    for group, lr in zip(optimizer.param_groups, lrs, strict=True):
        group["lr"] = float(lr)
        group["initial_lr"] = float(lr)


def make_continuation_scheduler(
    optimizer: torch.optim.Optimizer,
    total_updates: int = TOTAL_SCHEDULER_UPDATES,
) -> torch.optim.lr_scheduler.PolynomialLR:
    if total_updates <= 0:
        raise ValueError("Scheduler horizon must be positive")
    reset_group_lrs(optimizer)
    scheduler = torch.optim.lr_scheduler.PolynomialLR(
        optimizer, total_iters=total_updates, power=0.9
    )
    current = tuple(float(group["lr"]) for group in optimizer.param_groups)
    if not np.allclose(current, GROUP_LRS, rtol=0, atol=1e-15):
        raise RuntimeError(f"Scheduler did not start at the locked LR: {current}")
    return scheduler


def load_model_from_epoch5(base_model_dir: Path, checkpoint: dict[str, Any],
                           device: str) -> tuple[torch.nn.Module, dict[str, Any]]:
    model, base_audit = native.load_model(base_model_dir, device)
    model_state = checkpoint["model"]
    expected_state = model.state_dict()
    if set(model_state) != set(expected_state):
        missing = sorted(set(expected_state) - set(model_state))
        extra = sorted(set(model_state) - set(expected_state))
        raise ValueError(f"Checkpoint model keys differ: missing={missing[:8]}, extra={extra[:8]}")
    shape_mismatches = [
        key for key in expected_state
        if tuple(model_state[key].shape) != tuple(expected_state[key].shape)
    ]
    if shape_mismatches:
        raise ValueError(f"Checkpoint tensor shape mismatch: {shape_mismatches[:8]}")
    result = model.load_state_dict(model_state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("Strict checkpoint model load was not exact")
    return model, {
        "base_metric_model_load_audit": base_audit,
        "epoch5_model_load": "strict",
        "model_keys": len(model_state),
        "missing_keys": [],
        "unexpected_keys": [],
        "shape_mismatches": [],
    }


def build_signature(args, config, checkpoint_sha: str,
                    train_rows, val_rows) -> dict[str, Any]:
    return {
        "experiment_id": config["experiment_id"],
        "group": "C0",
        "seed": int(config["seed"]),
        "config_sha256": file_sha256(args.config),
        "epoch5_checkpoint_sha256": checkpoint_sha,
        "train_manifest_sha256": file_sha256(args.train_manifest),
        "validation_manifest_sha256": file_sha256(args.val_manifest),
        "pair_manifest_sha256": file_sha256(args.pair_manifest),
        "code_sha256": code_hashes(),
        "git": git_revision(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "train_images": len(train_rows),
        "validation_images": len(val_rows),
        "continuation": config["continuation"],
    }


def check_paths_inside_gpu_work(paths: list[tuple[str, Path]], root: Path) -> None:
    root = root.resolve()
    if not str(root).startswith("/localdata/"):
        raise ValueError(f"GPU work must resolve to /localdata: {root}")
    for label, path in paths:
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(f"{label} is outside ~/gpu_work: {resolved}")


def make_initial_state(args, config, output: Path, model: torch.nn.Module,
                       optimizer, scheduler, scaler, state5: dict[str, Any],
                       signature: dict[str, Any]) -> tuple[int, int, list[dict[str, Any]]]:
    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            raise FileExistsError(f"Existing output is preserved: {output}")
    else:
        output.mkdir(parents=True, exist_ok=False)
    scaler.load_state_dict(state5["scaler"])
    restore_rng(state5["rng"])
    history = list(state5["history"])
    initial_continuation = {
        "epoch": 5,
        "global_step": 960,
        "model": cpu_model_state(model),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "rng": capture_rng(),
        "history": history,
        "signature_payload": signature,
        "device_type": "cuda",
        "resume_origin": {
            "epoch5_checkpoint_sha256": EXPECTED_EPOCH5_SHA256,
            "optimizer_moments_restored": True,
            "scheduler_reset_at_epoch6": True,
        },
    }
    atomic_json(output / "run_signature.json", {
        "signature": canonical_signature(signature),
        "payload": signature,
        "initial_checkpoint_sha256": EXPECTED_EPOCH5_SHA256,
        "initial_state_audit": audit_epoch5(state5),
        "initial_gpu_preflight": args._gpu_receipt_audit,
        "optimizer_policy": "AdamW moments restored from epoch5; only LR and initial_lr reset",
        "scheduler_policy": {
            "class": "PolynomialLR", "power": 0.9,
            "total_successful_updates": TOTAL_SCHEDULER_UPDATES,
            "initial_lrs": list(GROUP_LRS),
        },
    })
    atomic_json(output / "provenance.json", {
        "started_unix_time": time.time(),
        "command": sys.argv,
        "git": signature["git"],
        "code_sha256": signature["code_sha256"],
        "initial_checkpoint_sha256": EXPECTED_EPOCH5_SHA256,
        "initial_state_audit": audit_epoch5(state5),
        "gpu_receipt_audit": args._gpu_receipt_audit,
        "gpu_name": signature["gpu_name"],
        "torch": signature["torch"],
        "cuda_runtime": signature["cuda_runtime"],
        "train_manifest_sha256": signature["train_manifest_sha256"],
        "validation_manifest_sha256": signature["validation_manifest_sha256"],
        "pair_manifest_sha256": signature["pair_manifest_sha256"],
    })
    initial_checkpoint, initial_checkpoint_sha = save_epoch(output, initial_continuation)
    atomic_json(output / "progress.json", {
        "status": "ready", "group": "C0", "epoch": 5,
        "next_epoch": 6, "global_step": 960,
        "checkpoint": initial_checkpoint.name,
        "checkpoint_sha256": initial_checkpoint_sha,
        "scheduler_total_updates": TOTAL_SCHEDULER_UPDATES,
    })
    return 6, 960, history


def restore_run(args, output: Path, model, optimizer, scheduler, scaler,
                signature: dict[str, Any], state5: dict[str, Any]) -> tuple[int, int, list[dict[str, Any]]]:
    if not output.is_dir():
        raise FileNotFoundError(f"Cannot resume absent run directory: {output}")
    saved_signature = json.loads((output / "run_signature.json").read_text(encoding="utf-8"))
    if saved_signature.get("signature") != canonical_signature(saved_signature.get("payload", {})):
        raise ValueError("Run signature file is internally inconsistent")
    if saved_signature.get("signature") != canonical_signature(signature):
        raise ValueError("Config, code, manifests, initial checkpoint, or environment changed")
    if (output / "run_complete.json").exists():
        raise FileExistsError("This run is already complete; preserve it and do not resume it")
    candidates = sorted(output.glob("epoch_[0-9][0-9][0-9][0-9].pt"))
    if not candidates:
        # A stop before the first epoch checkpoint is resumable from the
        # verified epoch5 state. The fixed signature still guards all inputs.
        optimizer.load_state_dict(state5["optimizer"])
        reset_group_lrs(optimizer)
        native._copy_optimizer_state_to_device(optimizer, args.device)
        scaler.load_state_dict(state5["scaler"])
        restore_rng(state5["rng"])
        return 6, 960, list(state5["history"])
    checkpoint_path = latest_epoch(output)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    loaded = load_epoch(checkpoint_path, model, optimizer, scheduler, scaler,
                        signature, args.device)
    native._copy_optimizer_state_to_device(optimizer, args.device)
    epoch = int(loaded["epoch"])
    step = int(loaded["global_step"])
    if epoch < 5 or epoch > EPOCH_END:
        raise ValueError(f"Unexpected resume epoch {epoch}")
    if int(state.get("global_step", -1)) != step:
        raise ValueError("Resume global_step mismatch")
    return epoch + 1, step, list(loaded.get("history", []))


def run(args) -> dict[str, Any]:
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    if config.get("status") != "locked" or config.get("group") != "C0":
        raise RuntimeError("Only the locked C0 experiment may be run")
    if int(config["continuation"]["start_epoch"]) != EPOCH_START:
        raise ValueError("Config epoch range differs from the fixed experiment")
    if int(config["continuation"]["end_epoch"]) != EPOCH_END:
        raise ValueError("Config epoch range differs from the fixed experiment")
    if int(config["continuation"]["scheduler_total_successful_updates"]) != TOTAL_SCHEDULER_UPDATES:
        raise ValueError("Config scheduler horizon differs from the fixed experiment")
    if args.device != "cuda" or not torch.cuda.is_available() or not args.amp:
        raise RuntimeError("Formal continuation requires CUDA FP16 AMP")
    if not args.preflight_receipt or not args.gpu_permission_record:
        raise RuntimeError("A fresh GPGPU4 preflight receipt is required")
    receipt_audit = check_gpu_receipt(
        args.preflight_receipt, args.gpu_permission_record,
        args.gpu_work_root, args.initial_checkpoint,
    )
    args._gpu_receipt_audit = receipt_audit

    root = args.gpu_work_root
    check_paths_inside_gpu_work([
        ("config", args.config), ("initial checkpoint", args.initial_checkpoint),
        ("base model", args.base_model_dir), ("train manifest", args.train_manifest),
        ("validation manifest", args.val_manifest), ("pair manifest", args.pair_manifest),
        ("RGB root", args.rgb_root), ("depth root", args.depth_root),
        ("preflight receipt", args.preflight_receipt),
        ("GPU permission record", args.gpu_permission_record), ("output", args.output),
    ], root)

    train_rows = native.read_manifest(args.train_manifest)
    val_rows = native.read_manifest(args.val_manifest)
    native.verify_manifest(train_rows, args.train_manifest,
                           ROOT / "configs/uav3dcrop_verified/train.csv", 384, "train")
    native.verify_manifest(val_rows, args.val_manifest,
                           ROOT / "configs/uav3dcrop_verified/validation.csv", 128, "validation")
    native.verify_files(train_rows + val_rows, args.rgb_root, args.depth_root)
    pair_data = json.loads(args.pair_manifest.read_text(encoding="utf-8"))
    if len(pair_data.get("images", [])) != 128:
        raise ValueError("Fixed Validation128 pair manifest is required")

    checkpoint_sha = file_sha256(args.initial_checkpoint)
    if checkpoint_sha != EXPECTED_EPOCH5_SHA256:
        raise ValueError(f"Epoch5 checkpoint SHA256 mismatch: {checkpoint_sha}")
    state5 = torch.load(args.initial_checkpoint, map_location="cpu", weights_only=False)
    state_audit = audit_epoch5(state5)
    model, model_audit = load_model_from_epoch5(args.base_model_dir, state5, args.device)
    model.eval()
    optimizer = native.make_optimizer(model)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    scheduler = make_continuation_scheduler(optimizer)
    signature = build_signature(args, config, checkpoint_sha, train_rows, val_rows)

    if args.resume:
        if (args.output.is_dir() and not (args.output / "run_signature.json").exists()
                and not any(args.output.iterdir())):
            # A process may be interrupted after creating the output directory
            # but before writing its first atomic metadata file. An empty
            # directory contains no state to overwrite; bootstrap it from the
            # verified epoch5 checkpoint and retain the requested resume event.
            optimizer.load_state_dict(state5["optimizer"])
            reset_group_lrs(optimizer)
            native._copy_optimizer_state_to_device(optimizer, args.device)
            scheduler = make_continuation_scheduler(optimizer)
            start_epoch, global_step, history = make_initial_state(
                args, config, args.output, model, optimizer, scheduler, scaler,
                state5, signature)
        else:
            scheduler = make_continuation_scheduler(optimizer)
            start_epoch, global_step, history = restore_run(
                args, args.output, model, optimizer, scheduler, scaler, signature, state5)
        event_path = args.output / "execution_events.jsonl"
        with event_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({
                "event": "resume_preflight_passed",
                "time_unix": time.time(),
                "gpu_preflight": receipt_audit,
            }, ensure_ascii=False, allow_nan=False) + "\n")
    else:
        optimizer.load_state_dict(state5["optimizer"])
        reset_group_lrs(optimizer)
        native._copy_optimizer_state_to_device(optimizer, args.device)
        # Construct the new schedule only after restoring optimizer state so
        # its base_lrs are exactly the locked continuation learning rates.
        scheduler = make_continuation_scheduler(optimizer)
        start_epoch, global_step, history = make_initial_state(
            args, config, args.output, model, optimizer, scheduler, scaler,
            state5, signature)

    dataset = native.NativePatchDataset(
        train_rows, args.rgb_root, args.depth_root, "C0",
        repeats=2, seed=int(config["seed"]), training=True,
    )
    silog_loss = native.m.frozen("losses").silog_loss
    epoch_rows: list[dict[str, Any]] = []
    expected_other_pids = set(receipt_audit["other_compute_pids_authorized"])
    for epoch in range(start_epoch, EPOCH_END + 1):
        assert_no_new_gpu_compute_processes(expected_other_pids)
        result = native._train_epoch(
            model, dataset, optimizer, scheduler, scaler, silog_loss,
            epoch=epoch, seed=int(config["seed"]), device=args.device,
            amp=True, accumulation=4,
            progress_callback=lambda *_: assert_no_new_gpu_compute_processes(expected_other_pids),
        )
        global_step += int(result["optimizer_updates"])
        atomic_json(args.output / "progress.json", {
            "status": "validating", "epoch": epoch, "global_step": global_step,
            "successful_updates_in_epoch": result["optimizer_updates"],
            "amp_skips_in_epoch": result["amp_skips"],
            "training_seconds": result["training_seconds"],
        })
        model.eval()
        assert_no_new_gpu_compute_processes(expected_other_pids)
        val_dir = args.output / "validation" / f"epoch_{epoch:04d}_attempt_{time.time_ns()}"
        validation = native.evaluate_combo(
            model, val_rows, args.rgb_root, args.depth_root, args.device, "P0",
            args.pair_manifest, val_dir, group="C0", epoch=epoch,
        )
        result.update({
            "group": "C0", "global_step": global_step,
            "scheduler_lr_encoder": float(optimizer.param_groups[0]["lr"]),
            "scheduler_lr_neck_head": float(optimizer.param_groups[1]["lr"]),
            "validation": validation,
            "cuda_max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "cuda_max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "checkpoint": f"epoch_{epoch:04d}.pt",
            "phase": "continued_C0_epoch5_to_epoch20",
        })
        history.append(result)
        state = {
            "epoch": epoch, "global_step": global_step,
            "model": cpu_model_state(model),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "rng": capture_rng(),
            "history": history,
            "signature_payload": signature,
            "device_type": "cuda",
            "train": result,
            "validation": validation,
            "resume_origin": {
                "epoch5_checkpoint_sha256": checkpoint_sha,
                "optimizer_moments_restored": True,
                "scheduler_reset_at_epoch6": True,
            },
        }
        checkpoint, checkpoint_sha_epoch = save_epoch(args.output, state)
        result["checkpoint_sha256"] = checkpoint_sha_epoch
        native._write_json_atomic(args.output / f"epoch_{epoch:04d}.json", result)
        native._write_epoch_csv(args.output / "epoch_metrics.csv", history)
        epoch_rows.append(result)
        print(json.dumps({
            "epoch": epoch, "global_step": global_step,
            "optimizer_updates": result["optimizer_updates"],
            "amp_skips": result["amp_skips"],
            "mean_silog_m": result["mean_silog_m"],
            "validation_rmse_m": validation["absolute"]["rmse_m"],
            "validation_high_delta14_mae_m": next(
                (row["delta_mae_m"] for row in validation["sampled_local"]
                 if row["stride_px"] == 14 and row["group"] == "gt_abs_ge_0.2m"), None
            ),
            "checkpoint_sha256": checkpoint_sha_epoch,
        }, ensure_ascii=False), flush=True)
        atomic_json(args.output / "progress.json", {
            "status": "checkpointed", "epoch": epoch, "global_step": global_step,
            "checkpoint": checkpoint.name, "checkpoint_sha256": checkpoint_sha_epoch,
            "updates": result["optimizer_updates"], "amp_skips": result["amp_skips"],
        })
        model.train()

    if start_epoch <= EPOCH_END:
        if global_step < 960:
            raise RuntimeError("Cumulative optimizer update count moved backwards")
    atomic_json(args.output / "run_complete.json", {
        "status": "complete", "group": "C0", "epoch": EPOCH_END,
        "global_step": global_step, "completed_unix_time": time.time(),
        "initial_checkpoint_sha256": checkpoint_sha,
        "checkpoint_sha256": file_sha256(args.output / f"epoch_{EPOCH_END:04d}.pt"),
    })
    atomic_json(args.output / "progress.json", {
        "status": "complete", "epoch": EPOCH_END, "global_step": global_step,
    })
    return {"status": "complete", "epoch": EPOCH_END, "global_step": global_step}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--base-model-dir", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--rgb-root", type=Path, required=True)
    parser.add_argument("--depth-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-receipt", type=Path, required=True)
    parser.add_argument("--gpu-permission-record", type=Path, required=True)
    parser.add_argument("--gpu-work-root", type=Path, default=Path.home() / "gpu_work")
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--amp", action="store_true", required=True)
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
