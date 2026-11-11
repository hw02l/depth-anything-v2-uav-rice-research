from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "local_huber_20261105"))

from uav_metric_research import multiscale as m
from data_patch import NativePatchDataset, read_manifest, load_pair
from patch_engine import (atomic_torch_save, capture_rng, canonical_signature,
                          cpu_model_state, file_sha256, latest_epoch, load_epoch,
                          restore_rng, save_epoch, seed_everything)
from evaluate_patch import evaluate_combo


def sha256_file(path: str | Path) -> str:
    return file_sha256(path)


def verify_manifest(rows, path: Path, canonical_path: Path, count: int, split: str) -> None:
    if len(rows) != count or any(row["split"] != split for row in rows):
        raise ValueError(f"Expected canonical {split}{count} manifest")
    if sha256_file(path) != sha256_file(canonical_path):
        raise ValueError(f"Manifest bytes differ from canonical file: {path}")


def verify_files(rows, rgb_root: Path, depth_root: Path) -> None:
    missing = []
    for row in rows:
        for root, key in ((rgb_root, "image_relpath"), (depth_root, "depth_relpath")):
            path = (root / row[key]).resolve()
            if not path.is_relative_to(root.resolve()) or not path.is_file():
                missing.append(str(path))
    if missing:
        raise FileNotFoundError(f"{len(missing)} dataset files missing or unsafe; first: {missing[:4]}")


def load_model(checkpoint: Path, device: str):
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForDepthEstimation
    config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
    if getattr(config, "max_depth", None) != 80 or getattr(config, "depth_estimation_type", None) != "metric":
        raise ValueError("Initial model must be Metric Outdoor with max_depth=80")
    model = AutoModelForDepthEstimation.from_config(config)
    audit = m.strict_attention_state(model, load_file(str(checkpoint / "model.safetensors"), device="cpu"))
    model.to(device)
    return model, audit


def check_initial_hashes(checkpoint: Path) -> dict[str, str]:
    expected = {
        "model.safetensors": "52769e2e95704d0862e708e8772195b861c647dd4812058eaf698a44305c3bcb",
        "config.json": "3c6ab05881e6a76edbfd33ff393fdfb082582084e9262826bb1ac0fd2b1dac7c",
    }
    actual = {name: sha256_file(checkpoint / name) for name in expected}
    for name, value in expected.items():
        if actual[name] != value:
            raise ValueError(f"Initial checkpoint hash mismatch: {name}")
    return actual


def check_gpu_receipt(path: Path) -> dict[str, Any]:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    age = time.time() - float(receipt.get("unix_time", 0))
    if receipt.get("status") != "available" or receipt.get("hostname", "").lower() != "gpgpu4":
        raise RuntimeError("Fresh available GPGPU4 receipt is required")
    if not 0 <= age <= 300:
        raise RuntimeError(f"GPU receipt is stale ({age:.1f}s)")
    if receipt.get("compute_processes"):
        raise RuntimeError("GPU compute process is active; do not compete")
    if receipt.get("gpu_free_bytes", 0) < 1024**3:
        raise RuntimeError("Less than 1GiB VRAM is free")
    if receipt.get("work_root_realpath", "").startswith("/localdata/") is False:
        raise RuntimeError("~/gpu_work is not on local computation storage")
    if receipt.get("singularity_version") != "4.2.2":
        raise RuntimeError("Unexpected Singularity runtime; stop without bypass")
    return {"receipt_sha256": sha256_file(path), "age_seconds": age, "gpu": receipt["gpu_name"]}


def source_hashes() -> dict[str, str]:
    return {
        path.name: sha256_file(path)
        for path in sorted(HERE.glob("*.py"))
        if not path.name.startswith("test_")
    }


def make_optimizer(model):
    encoder = [parameter for name, parameter in model.named_parameters() if name.startswith("backbone.")]
    head = [parameter for name, parameter in model.named_parameters() if not name.startswith("backbone.")]
    if not encoder or not head:
        raise ValueError("Missing encoder or neck/head parameter groups")
    return torch.optim.AdamW(
        [{"params": encoder, "lr": 2e-6}, {"params": head, "lr": 2e-5}],
        weight_decay=0.01,
    )


def _train_epoch(model, dataset, optimizer, scheduler, scaler, silog_loss,
                 *, epoch: int, seed: int, device: str, amp: bool,
                 accumulation: int, max_updates: int | None = None) -> dict[str, Any]:
    dataset.set_epoch(epoch)
    random.seed(seed + epoch - 1)
    np.random.seed(seed + epoch - 1)
    generator = torch.Generator().manual_seed(seed + epoch - 1)
    loader = DataLoader(dataset, batch_size=None, shuffle=True, num_workers=0, generator=generator)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    valid_fractions = []
    batch_count = updates = amp_skips = 0
    started = time.perf_counter()
    for index, sample in enumerate(loader):
        pixels = sample["pixel_values"].unsqueeze(0).to(device, non_blocking=False)
        target = sample["depth"].unsqueeze(0).to(device, non_blocking=False)
        valid = torch.isfinite(target) & (target >= 0.001) & (target <= 65)
        context = torch.autocast("cuda", dtype=torch.float16) if amp else torch.autocast("cpu", enabled=False)
        with context:
            prediction = model(pixel_values=pixels).predicted_depth
        loss = silog_loss(prediction.float(), target.float(), valid)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite SiLog loss")
        scaler.scale(loss / accumulation).backward()
        running_loss += float(loss.detach())
        valid_fractions.append(float(sample["metadata"]["valid_gt_fraction"]))
        batch_count += 1
        should_step = (index + 1) % accumulation == 0 or index + 1 == len(loader)
        if should_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=not scaler.is_enabled())
            prior_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= prior_scale:
                scheduler.step()
                updates += 1
            else:
                amp_skips += 1
            optimizer.zero_grad(set_to_none=True)
            if max_updates is not None and updates >= max_updates:
                break
    elapsed = time.perf_counter() - started
    if updates == 0:
        raise RuntimeError("No optimizer update completed")
    return {
        "epoch": epoch, "samples": batch_count, "optimizer_updates": updates,
        "amp_skips": amp_skips, "mean_silog_m": running_loss / batch_count,
        "mean_valid_gt_fraction": float(np.mean(valid_fractions)),
        "min_valid_gt_fraction": float(np.min(valid_fractions)),
        "mean_seconds_per_crop": elapsed / batch_count,
        "training_seconds": elapsed,
    }


def _copy_optimizer_state_to_device(optimizer, device: str) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_name("." + path.name + "." + str(time.time_ns()) + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _write_epoch_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    temporary = path.with_name("." + path.name + "." + str(time.time_ns()) + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _formal_signature(args, config, checkpoint_hashes, train_manifest, val_manifest, pairs,
                      validation_rows) -> dict[str, Any]:
    return {
        "experiment_id": config["experiment_id"], "group": args.group,
        "config_sha256": sha256_file(args.config),
        "train_manifest_sha256": sha256_file(train_manifest),
        "validation_manifest_sha256": sha256_file(val_manifest),
        "pair_manifest_sha256": sha256_file(pairs),
        "initial_checkpoint_hashes": checkpoint_hashes,
        "code_sha256": source_hashes(),
        "group_mode": args.group, "seed": config["seed"],
        "loss": config["loss"], "training": config.get("training", {}),
        "python": sys.version, "torch": torch.__version__,
        "validation_rows": len(validation_rows),
        "device": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else "cpu",
    }


def run(args) -> dict[str, Any]:
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    if args.mode == "formal" and config.get("status") != "locked":
        raise RuntimeError("Formal training is blocked until config.json is locked")
    if args.mode == "formal":
        if args.device != "cuda" or not torch.cuda.is_available() or not args.amp:
            raise RuntimeError("Formal runs require verified CUDA + FP16 AMP")
        if not args.preflight_receipt:
            raise RuntimeError("A fresh GPU preflight receipt is required")
        receipt = check_gpu_receipt(args.preflight_receipt)
        root = args.gpu_work_root.resolve()
        for label, path in (("config", args.config), ("checkpoint", args.initial_checkpoint),
                            ("train manifest", args.train_manifest), ("validation manifest", args.val_manifest),
                            ("pair manifest", args.pair_manifest), ("RGB root", args.rgb_root),
                            ("depth root", args.depth_root), ("preflight receipt", args.preflight_receipt),
                            ("output", args.output)):
            if not path.resolve().is_relative_to(root):
                raise ValueError(f"{label} must reside inside ~/gpu_work: {path}")
    else:
        receipt = None
        if args.mode == "pilot":
            if args.device != "cuda" or not torch.cuda.is_available() or not args.amp or not args.preflight_receipt:
                raise RuntimeError("GPU pilot requires CUDA, AMP, and a live receipt")
            receipt = check_gpu_receipt(args.preflight_receipt)
        elif args.device != "cpu":
            raise ValueError("CPU smoke mode must use --device cpu")

    train_rows = read_manifest(args.train_manifest)
    val_rows = read_manifest(args.val_manifest)
    canonical_train = ROOT / "configs/uav3dcrop_verified/train.csv"
    canonical_val = ROOT / "configs/uav3dcrop_verified/validation.csv"
    verify_manifest(train_rows, args.train_manifest, canonical_train, 384, "train")
    verify_manifest(val_rows, args.val_manifest, canonical_val, 128, "validation")
    if args.mode != "smoke":
        verify_files(train_rows + val_rows, args.rgb_root, args.depth_root)
    checkpoint_hashes = check_initial_hashes(args.initial_checkpoint)
    model, load_audit = load_model(args.initial_checkpoint, args.device)
    pairs = json.loads(args.pair_manifest.read_text(encoding="utf-8"))
    if len(pairs.get("images", [])) != 128:
        raise ValueError("Canonical fixed pair manifest must contain 128 images")
    if args.mode == "formal":
        training = config["training"]
        repeats = int(training["crops_per_image_per_epoch"])
        epochs = int(training["epochs"])
        accumulation = int(training["gradient_accumulation"])
        updates_per_epoch = math.ceil(384 * repeats / accumulation)
        if updates_per_epoch * epochs != training["total_optimizer_updates"]:
            raise ValueError("Locked optimizer budget does not match sample/repeat count")
        selected_rows = train_rows
        expected_updates = updates_per_epoch
    elif args.mode == "pilot":
        repeats, epochs, accumulation, expected_updates = 1, 1, 4, 2
        selected_rows = args.pilot_rows
        if not selected_rows or len(selected_rows) != 8:
            raise ValueError("Pilot train set must be the fixed 8-image list")
    else:
        repeats, epochs, accumulation, expected_updates = 1, 1, 1, 1
        selected_rows = args.pilot_rows[:1] if args.pilot_rows else train_rows[:1]

    scheduler = torch.optim.lr_scheduler.PolynomialLR(
        make_optimizer(model), total_iters=expected_updates * epochs, power=0.9)
    optimizer = scheduler.optimizer
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    silog_loss = m.frozen("losses").silog_loss
    dataset = NativePatchDataset(selected_rows, args.rgb_root, args.depth_root, args.group,
                                 repeats=repeats, seed=int(config.get("seed", 42)), training=True)
    signature_payload = _formal_signature(args, config, checkpoint_hashes,
                                          args.train_manifest, args.val_manifest,
                                          args.pair_manifest, val_rows)
    signature_payload.update({
        "mode": args.mode, "epochs": epochs, "repeats": repeats,
        "accumulation": accumulation, "updates_per_epoch": expected_updates,
        "amp": args.amp, "load_audit": load_audit,
        "gpu_receipt_sha256": receipt["receipt_sha256"] if receipt else None,
    })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.resume:
        prior = torch.load(latest_epoch(args.output), map_location="cpu", weights_only=False)
        if prior["checkpoint_signature"] != canonical_signature(signature_payload):
            raise ValueError("Resume signature mismatch")
        state = load_epoch(latest_epoch(args.output), model, optimizer, scheduler, scaler,
                           signature_payload, args.device)
        _copy_optimizer_state_to_device(optimizer, args.device)
        start_epoch = int(state["epoch"]) + 1
        global_step = int(state["global_step"])
        history = state.get("history", [])
    else:
        if args.output.exists():
            raise FileExistsError(f"Existing run directory preserved: {args.output}")
        args.output.mkdir(parents=True, exist_ok=False)
        seed_everything(int(config.get("seed", 42)))
        start_epoch, global_step, history = 1, 0, []
        _write_json_atomic(args.output / "run_signature.json", {
            "signature": canonical_signature(signature_payload), "payload": signature_payload,
            "initial_checkpoint": checkpoint_hashes, "load_audit": load_audit,
        })
        if receipt:
            _write_json_atomic(args.output / f"gpu_receipt_{time.time_ns()}.json", receipt)

    for epoch in range(start_epoch, epochs + 1):
        result = _train_epoch(
            model, dataset, optimizer, scheduler, scaler, silog_loss, epoch=epoch,
            seed=int(config.get("seed", 42)), device=args.device, amp=args.amp,
            accumulation=accumulation, max_updates=2 if args.mode == "pilot" else None,
        )
        global_step += result["optimizer_updates"]
        if args.mode == "formal":
            model.eval()
            val_dir = args.output / "validation" / f"epoch_{epoch:04d}"
            validation = evaluate_combo(model, val_rows, args.rgb_root, args.depth_root,
                                        args.device, "P0", args.pair_manifest, val_dir,
                                        group=args.group, epoch=epoch)
        else:
            validation = {
                "mode": args.mode, "validation_performed": False,
                "pilot_images": len(selected_rows), "research_result": False,
            }
        result.update({
            "group": args.group, "global_step": global_step,
            "scheduler_lr_encoder": optimizer.param_groups[0]["lr"],
            "scheduler_lr_neck_head": optimizer.param_groups[1]["lr"],
            "validation": validation,
            "cuda_max_allocated_bytes": int(torch.cuda.max_memory_allocated()) if args.device.startswith("cuda") else 0,
            "cuda_max_reserved_bytes": int(torch.cuda.max_memory_reserved()) if args.device.startswith("cuda") else 0,
        })
        history.append(result)
        state = {
            "epoch": epoch, "global_step": global_step, "model": cpu_model_state(model),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(), "rng": capture_rng(),
            "history": history, "signature_payload": signature_payload,
            "device_type": "cuda" if args.device.startswith("cuda") else "cpu",
            "train": result, "validation": validation,
        }
        checkpoint, digest = save_epoch(args.output, state)
        result["checkpoint"] = checkpoint.name
        result["checkpoint_sha256"] = digest
        if args.mode == "pilot":
            restored = load_epoch(checkpoint, model, optimizer, scheduler, scaler,
                                  signature_payload, args.device)
            if int(restored["epoch"]) != epoch or int(restored["global_step"]) != global_step:
                raise RuntimeError("GPU pilot checkpoint resume round-trip returned the wrong epoch/step")
            result["checkpoint_resume_roundtrip"] = "pass"
        _write_json_atomic(args.output / f"epoch_{epoch:04d}.json", result)
        _write_epoch_csv(args.output / "epoch_metrics.csv", history)
        print(json.dumps({
            "mode": args.mode, "group": args.group, "epoch": epoch,
            "global_step": global_step, "updates": result["optimizer_updates"],
            "amp_skips": result["amp_skips"], "mean_silog_m": result["mean_silog_m"],
            "training_seconds": result["training_seconds"], "checkpoint_sha256": digest,
        }, ensure_ascii=False), flush=True)
        if args.mode == "pilot":
            break
    return {"group": args.group, "epoch": epoch, "global_step": global_step,
            "history": history, "output": str(args.output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "formal", "smoke"), required=True)
    parser.add_argument("--group", choices=("C0", "F1", "F2"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--rgb-root", type=Path, required=True)
    parser.add_argument("--depth-root", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-receipt", type=Path)
    parser.add_argument("--gpu-work-root", type=Path, default=Path("/work"))
    parser.add_argument("--pilot-rows-json", type=Path)
    args = parser.parse_args()
    args.pilot_rows = json.loads(args.pilot_rows_json.read_text(encoding="utf-8")) if args.pilot_rows_json else None
    run(args)


if __name__ == "__main__":
    main()
