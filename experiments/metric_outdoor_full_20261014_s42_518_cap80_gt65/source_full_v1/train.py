from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import sys
import time
import uuid
from pathlib import Path, PurePosixPath
from typing import Any

import cv2
import huggingface_hub
import numpy as np
import requests
import safetensors
import tifffile
import torch
import transformers
from safetensors.torch import load_file
from torch.optim import AdamW
from torch.utils.data import DataLoader

if __package__:
    from .data import UAVDepthDataset, read_manifest, worker_seed
    from .losses import silog_loss
    from .metrics import evaluate_model
    from .modeling import METRIC_OUTDOOR_MODEL_ID, RELATIVE_MODEL_ID, build_metric_model
else:
    from data import UAVDepthDataset, read_manifest, worker_seed
    from losses import silog_loss
    from metrics import evaluate_model
    from modeling import METRIC_OUTDOOR_MODEL_ID, RELATIVE_MODEL_ID, build_metric_model


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if not required.issubset(state):
        raise ValueError("完全checkpointに必要な乱数状態がありません")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA乱数状態を復元できません。GPUランタイムへ接続してください。")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def should_pause_after_epoch(epoch_number: int, total_epochs: int, stop_after_epoch: int | None) -> bool:
    return stop_after_epoch is not None and epoch_number >= stop_after_epoch and epoch_number < total_epochs


def check_split_isolation(train_rows: list[dict[str, str]], val_rows: list[dict[str, str]]) -> None:
    if not train_rows or not val_rows:
        raise ValueError("Train and validation manifests must both contain rows")
    if any(row["split"] != "train" for row in train_rows):
        raise ValueError("--train-manifest contains a non-train split label")
    if any(row["split"] != "validation" for row in val_rows):
        raise ValueError("--val-manifest contains a non-validation split label")
    train_scenes = {row["scene"] for row in train_rows}
    val_scenes = {row["scene"] for row in val_rows}
    train_days = {row["day"] for row in train_rows}
    val_days = {row["day"] for row in val_rows}
    if train_scenes & val_scenes or train_days & val_days:
        raise ValueError("Train/validation must be disjoint by both scene and acquisition day")
    protected = ("2025/Day001_", "2025/Day007_")
    if any(row["scene"].startswith(protected) for row in train_rows + val_rows):
        raise ValueError("The historical 52-pair evaluation days are quarantined from training and validation")
    if any(row["split"] == "independent_test" for row in train_rows + val_rows):
        raise ValueError("Independent test rows may never be used for training or validation")
    for label, rows in (("train", train_rows), ("validation", val_rows)):
        missing = []
        # Files are checked again after roots are known in main; this protects malformed paths here.
        for row in rows:
            image_rel = PurePosixPath(row["image_relpath"])
            depth_rel = PurePosixPath(row["depth_relpath"])
            scene = PurePosixPath(row["scene"])
            if image_rel.parts[:2] != scene.parts or image_rel.parent.name != "images":
                raise ValueError(f"RGB path does not match the declared scene: {row['scene']} / {row['image_relpath']}")
            if depth_rel.parts[:2] != (scene.parts[0], scene.name + "_depth") or depth_rel.stem != image_rel.stem + "_depth":
                raise ValueError(f"Depth path does not match the RGB/scene: {row['image_relpath']} / {row['depth_relpath']}")
            for key, path in (("image_relpath", Path(row["image_relpath"])), ("depth_relpath", Path(row["depth_relpath"]))):
                if path.is_absolute() or ".." in path.parts:
                    missing.append(row[key])
        if missing:
            raise ValueError(f"Unsafe relative paths in {label} manifest: {missing[:3]}")


def verify_local_pairs(rows: list[dict[str, str]], rgb_root: Path, depth_root: Path, label: str) -> None:
    missing: list[str] = []
    for row in rows:
        for root, key in ((rgb_root, "image_relpath"), (depth_root, "depth_relpath")):
            candidate = (root / Path(row[key])).resolve()
            if not candidate.is_relative_to(root.resolve()):
                raise ValueError(f"Manifest path escapes {root}: {row[key]}")
            if not candidate.is_file():
                missing.append(str(candidate))
    if missing:
        raise FileNotFoundError(
            f"{label} data is incomplete: {len(missing)} missing files. First paths: {missing[:5]}"
        )


def validate_preflight(path: Path, args: argparse.Namespace, init_summary: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"GPU preflight report not found: {path}. Run memory_preflight.py first.")
    report = json.loads(path.read_text(encoding="utf-8"))
    checks = {
        "status": report.get("status") == "pass",
        "training_feasible": report.get("training_feasible") is True,
        "initialization": report.get("initialization", {}).get("initialization") == args.initialization,
        "image_size": report.get("image_size") == args.image_size,
        "batch_size": report.get("batch_size") == args.batch_size,
        "amp": report.get("amp") == args.amp,
        "source": report.get("initialization", {}).get("source") == init_summary.get("source"),
        "source_revision": report.get("initialization", {}).get("source_revision") == init_summary.get("source_revision"),
        "model_max_depth": report.get("initialization", {}).get("max_depth_m") == init_summary["max_depth_m"],
        "gt_max_depth": report.get("gt_max_depth_m", report.get("initialization", {}).get("max_depth_m")) == args.max_depth,
        "gpu": report.get("device") == torch.cuda.get_device_name(0),
    }
    failed = [key for key, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"GPU preflight does not match this training run ({', '.join(failed)}). Rerun it with identical settings/model.")
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    measured_reserved = int(report["peak_reserved_bytes_one_synthetic_step"])
    baseline_reserved = int(report.get("baseline_model_reserved_bytes", 0))
    additional_step_memory = max(measured_reserved - baseline_reserved, 0)
    # The model itself is already resident in this process; compare only extra step memory plus headroom.
    required_free = int(additional_step_memory * 1.15) + 512 * 1024**2
    report["training_start_free_vram_bytes"] = int(free_bytes)
    report["training_start_total_vram_bytes"] = int(total_bytes)
    report["preflight_additional_step_memory_bytes"] = additional_step_memory
    report["required_free_vram_bytes_with_safety_margin"] = required_free
    if free_bytes < required_free:
        raise MemoryError(
            f"GPU safety stop: {free_bytes / 1024**3:.2f} GiB free, but preflight peak reserved "
            f"plus safety margin requires {required_free / 1024**3:.2f} GiB. Close other GPU jobs or reduce resolution, then rerun preflight."
        )
    return report


def checkpoint_directories(run_dir: Path) -> list[Path]:
    return sorted(
        [path for path in run_dir.glob("checkpoint_epoch_[0-9][0-9][0-9][0-9]") if (path / "complete.json").is_file()],
        key=lambda path: int(path.name.rsplit("_", 1)[1]),
    )


def save_checkpoint(
    run_dir: Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    best_rmse: float,
    best_epoch: int | None,
) -> Path:
    name = f"checkpoint_epoch_{epoch:04d}"
    final_dir = run_dir / name
    staging_dir = run_dir / ("." + name + "." + uuid.uuid4().hex + ".tmp")
    if final_dir.exists() or staging_dir.exists():
        raise FileExistsError(f"Checkpoint path already exists; preserving it: {final_dir if final_dir.exists() else staging_dir}")
    staging_dir.mkdir(parents=False)
    try:
        model.save_pretrained(staging_dir, safe_serialization=True)
        torch.save(
            {
                "completed_epoch": epoch,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "rng_state": capture_rng_state(),
                "best_validation_rmse_m": best_rmse if math.isfinite(best_rmse) else None,
                "best_epoch": best_epoch,
            },
            staging_dir / "trainer_state.pt",
        )
        (staging_dir / "complete.json").write_text(
            json.dumps({"completed_epoch": epoch, "file_sha256": {
                filename: sha256_file(staging_dir / filename)
                for filename in ("model.safetensors", "config.json", "trainer_state.pt")
            }}, indent=2), encoding="utf-8"
        )
        staging_dir.replace(final_dir)
    except Exception:
        # Incomplete staging directories are intentionally retained for inspection.
        raise
    return final_dir


def prune_checkpoints(run_dir: Path, best_epoch: int | None, keep_recent: int) -> None:
    if keep_recent == 0:
        return  # Preserve all completed checkpoints unless retention was explicitly requested.
    checkpoints = checkpoint_directories(run_dir)
    keep = {path.name for path in checkpoints[-keep_recent:]}
    if best_epoch is not None:
        keep.add(f"checkpoint_epoch_{best_epoch:04d}")
    for path in checkpoints:
        if path.name not in keep:
            shutil.rmtree(path)


def validate_resume_signature(previous: dict, current: dict) -> None:
    if previous != current:
        raise ValueError("再開設定・manifest・コードが元runと一致しません。試験から本学習へ移る場合も新しいrunを作成してください。")


def scaled_optimizer_step(optimizer, scheduler, scaler):
    previous_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    updated = scaler.get_scale() >= previous_scale
    if updated:
        scheduler.step()
    return updated


def restore_checkpoint(checkpoint: Path, model, optimizer, scheduler, scaler, device):
    marker = json.loads((checkpoint / "complete.json").read_text(encoding="utf-8"))
    hashes = marker.get("file_sha256", {})
    for filename in ("model.safetensors", "config.json", "trainer_state.pt"):
        path = checkpoint / filename
        if not path.is_file() or hashes.get(filename) != sha256_file(path):
            raise ValueError(f"checkpointの完全性を確認できません。保存ファイルは変更しません: {path}")
    # CPU staging avoids allocating a duplicate model/optimizer state on the GPU.
    model.load_state_dict(load_file(str(checkpoint / "model.safetensors"), device="cpu"), strict=True)
    state = torch.load(checkpoint / "trainer_state.pt", map_location="cpu", weights_only=False)
    if state["completed_epoch"] != marker["completed_epoch"]:
        raise ValueError("checkpointのepoch情報が一致しません。")
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    restore_rng_state(state.get("rng_state", {}))
    return state


def truncate_epoch_log(path: Path, completed_epoch: int) -> None:
    if not path.exists():
        return
    retained = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            if int(json.loads(line)["epoch"]) <= completed_epoch:
                retained.append(line)
        except (ValueError, KeyError, json.JSONDecodeError):
            continue
    path.write_text("\n".join(retained) + ("\n" if retained else ""), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune UAV3DCrop Metric Depth from one Small checkpoint.")
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--rgb-root", type=Path, required=True)
    parser.add_argument("--depth-root", type=Path, required=True)
    parser.add_argument("--initialization", choices=("relative_encoder", "metric_outdoor"), required=True)
    parser.add_argument("--relative-model", default=RELATIVE_MODEL_ID)
    parser.add_argument("--metric-model", default=METRIC_OUTDOOR_MODEL_ID)
    parser.add_argument("--max-depth", type=float, required=True, help="Metres; derive using train-only analyze_gt.py")
    parser.add_argument("--model-max-depth", type=float, default=None, help="Model output ceiling; --max-depth remains the GT mask limit")
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accumulation", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--run-kind", choices=("trial", "full"), default="trial")
    parser.add_argument("--encoder-lr", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-every-epochs", type=int, default=1)
    parser.add_argument("--keep-recent-checkpoints", type=int, default=0, help="0 preserves all; positive retention is explicit")
    parser.add_argument("--memory-preflight-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Resume only this run directory and identical configuration")
    parser.add_argument("--stop-after-epoch", type=int, default=None, help="Pause cleanly at this completed epoch; excluded from the 20-epoch schedule signature")
    args = parser.parse_args()
    if args.stop_after_epoch is not None and not 1 <= args.stop_after_epoch <= args.epochs:
        parser.error("stop-after-epoch must be between 1 and the fixed total epoch count")
    if args.model_max_depth is not None and args.model_max_depth <= 0:
        parser.error("model-max-depth must be positive")

    if not torch.cuda.is_available():
        parser.error("CUDA is required for training. CPU preparation is supported; use cpu_smoke.py for a tiny CPU check.")
    if args.max_depth <= 0 or args.epochs < 1 or args.batch_size < 1 or args.grad_accumulation < 1:
        parser.error("Depth, epochs, batch size, and gradient accumulation must be positive")
    if args.image_size < 56 or args.image_size % 14:
        parser.error("image-size must be >=56 and divisible by 14")
    if args.run_kind == "trial" and args.epochs != 1:
        parser.error("試験runはepochs=1です。本学習は--run-kind fullと新しいoutput-dirを指定してください。")
    if args.save_every_epochs < 1 or args.keep_recent_checkpoints < 0:
        parser.error("Checkpoint intervals/retention must be positive")
    if args.workers < 0:
        parser.error("workers cannot be negative")

    run_dir = args.output_dir.resolve()
    if run_dir.exists() and not args.resume:
        parser.error(f"Run directory exists. Use --resume only for the same run, or choose a new directory: {run_dir}")
    if not run_dir.exists() and args.resume:
        parser.error(f"Cannot resume a run that does not exist: {run_dir}")
    train_rows = read_manifest(args.train_manifest)
    val_rows = read_manifest(args.val_manifest)
    check_split_isolation(train_rows, val_rows)
    verify_local_pairs(train_rows, args.rgb_root, args.depth_root, "train")
    verify_local_pairs(val_rows, args.rgb_root, args.depth_root, "validation")

    seed_all(args.seed)
    device = torch.device("cuda")
    model, init_summary = build_metric_model(
        args.initialization,
        max_depth=args.model_max_depth if args.model_max_depth is not None else args.max_depth,
        relative_model=args.relative_model,
        metric_model=args.metric_model,
        local_files_only=True,
    )
    model_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    free_before_load, _ = torch.cuda.mem_get_info(device)
    if free_before_load < model_bytes + 256 * 1024**2:
        raise MemoryError(
            f"GPU safety stop before model transfer: {free_before_load / 1024**3:.2f} GiB free; "
            f"model parameters need about {model_bytes / 1024**3:.2f} GiB plus 0.25 GiB margin."
        )
    try:
        model.to(device)
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        raise MemoryError("CUDA OOM while placing the model on GPU; no training began. Close other GPU workloads and rerun preflight.") from exc
    preflight = validate_preflight(args.memory_preflight_report, args, init_summary)

    train_dataset = UAVDepthDataset(train_rows, args.rgb_root, args.depth_root, image_size=args.image_size, training=True)
    val_dataset = UAVDepthDataset(val_rows, args.rgb_root, args.depth_root, image_size=args.image_size, training=False)
    # Recreated once per epoch with seed+epoch; this makes restarts deterministic at epoch boundaries.
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=args.workers,
        pin_memory=True, worker_init_fn=worker_seed, persistent_workers=False,
    )

    backbone_params = [p for name, p in model.named_parameters() if name.startswith("backbone.")]
    head_params = [p for name, p in model.named_parameters() if not name.startswith("backbone.")]
    optimizer = AdamW(
        [{"params": backbone_params, "lr": args.encoder_lr}, {"params": head_params, "lr": args.encoder_lr * 10.0}],
        betas=(0.9, 0.999), weight_decay=args.weight_decay,
    )
    optimizer_steps_per_epoch = math.ceil(math.ceil(len(train_dataset) / args.batch_size) / args.grad_accumulation)
    total_steps = optimizer_steps_per_epoch * args.epochs
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=total_steps, power=0.9)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    code_files = ("train.py", "modeling.py", "data.py", "losses.py", "metrics.py", "audit_data.py",
                  "memory_preflight.py", "analyze_gt.py", "evaluate.py")
    run_metadata = {
        "arguments": json_safe(vars(args)),
        "initialization": init_summary,
        "seed": args.seed,
        "versions": {
            "python": sys.version, "torch": torch.__version__, "transformers": transformers.__version__,
            "numpy": np.__version__, "opencv": cv2.__version__, "tifffile": tifffile.__version__,
            "huggingface_hub": huggingface_hub.__version__, "requests": requests.__version__,
            "safetensors": safetensors.__version__,
        },
        "device": torch.cuda.get_device_name(0),
        "cuda_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "memory_preflight": preflight,
        "train_manifest_sha256": sha256_file(args.train_manifest),
        "validation_manifest_sha256": sha256_file(args.val_manifest),
        "code_sha256": {filename: sha256_file(Path(__file__).with_name(filename)) for filename in code_files}
        | {"scene_split_plan.json": sha256_file(Path(__file__).parent / "config" / "scene_split_plan.json")},
        "train_pairs": len(train_rows), "validation_pairs": len(val_rows),
        "metrics": ["mae_m", "rmse_m", "abs_rel", "local_depth_delta_mae_m", "pearson_r"],
        "loss": "SiLog with finite positive targets not exceeding GT max_depth",
        "gt_max_depth_m": args.max_depth,
        "model_max_depth_m": init_summary["max_depth_m"],
        "test_data_read": False,
    }
    signature_keys = (
        "initialization", "max_depth", "model_max_depth", "image_size", "batch_size", "grad_accumulation", "epochs", "run_kind",
        "encoder_lr", "weight_decay", "workers", "seed", "amp", "save_every_epochs",
        "keep_recent_checkpoints", "relative_model", "metric_model",
    )
    signature = {key: run_metadata["arguments"].get(key) for key in signature_keys}
    signature.update({
        "runtime_versions": {key: run_metadata["versions"][key] for key in ("torch", "transformers", "numpy", "safetensors")},
        "initialization_source": init_summary["source"],
        "initialization_source_revision": init_summary.get("source_revision"),
        "train_manifest_sha256": run_metadata["train_manifest_sha256"],
        "validation_manifest_sha256": run_metadata["validation_manifest_sha256"],
        "code_sha256": run_metadata["code_sha256"],
    })
    metadata_path = run_dir / "run_metadata.json"
    if args.resume:
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Run metadata is missing; refusing unsafe resume: {metadata_path}")
        old_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        old_signature = old_metadata.get("resume_signature")
        validate_resume_signature(old_signature, signature)
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        run_metadata["resume_signature"] = signature
        write_json_atomic(metadata_path, run_metadata)

    log_path = run_dir / "epoch_metrics.jsonl"
    best_rmse = float("inf")
    best_epoch: int | None = None
    completed_epoch = 0
    epoch_durations: list[float] = []
    if args.resume:
        checkpoints = checkpoint_directories(run_dir)
        if not checkpoints:
            raise FileNotFoundError(f"No complete checkpoint to resume from in {run_dir}")
        latest = checkpoints[-1]
        trainer_state = restore_checkpoint(latest, model, optimizer, scheduler, scaler, device)
        completed_epoch = int(trainer_state["completed_epoch"])
        best_rmse_value = trainer_state.get("best_validation_rmse_m")
        best_rmse = float(best_rmse_value) if best_rmse_value is not None else float("inf")
        best_epoch = trainer_state.get("best_epoch")
        if best_epoch is not None:
            best_path = run_dir / f"checkpoint_epoch_{int(best_epoch):04d}"
            if not (best_path / "complete.json").is_file():
                raise FileNotFoundError(f"Best checkpoint referenced by training state is missing: {best_path}")
            write_json_atomic(run_dir / "best_checkpoint.json", {"path": best_path.name, "epoch": int(best_epoch)})
        truncate_epoch_log(log_path, completed_epoch)
        if log_path.is_file():
            for line in log_path.read_text(encoding="utf-8").splitlines():
                try:
                    duration = float(json.loads(line).get("epoch_seconds", 0.0))
                    if duration > 0:
                        epoch_durations.append(duration)
                except (ValueError, json.JSONDecodeError):
                    continue
        print(f"Resumed from epoch {completed_epoch}/{args.epochs}; remaining {max(args.epochs - completed_epoch, 0)} epoch(s).")
    else:
        checkpoint = save_checkpoint(run_dir, 0, model, optimizer, scheduler, scaler, best_rmse, best_epoch)
        write_json_atomic(run_dir / "latest_checkpoint.json", {"path": checkpoint.name, "completed_epoch": 0})
        write_json_atomic(run_dir / "run_status.json", {
            "status": "running", "completed_epochs": 0, "remaining_epochs": args.epochs,
            "best_validation_rmse_m": None, "best_epoch": None,
        })

    for epoch_index in range(completed_epoch, args.epochs):
        epoch_number = epoch_index + 1
        print(f"Starting epoch {epoch_number}/{args.epochs}; completed {epoch_index}, remaining including this epoch {args.epochs - epoch_index}.")
        train_generator = torch.Generator().manual_seed(args.seed + epoch_index)
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
            pin_memory=True, drop_last=False, worker_init_fn=worker_seed,
            generator=train_generator, persistent_workers=False,
        )
        epoch_start = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(device)
        model.train()
        running_loss = 0.0
        optimizer_steps_completed = 0
        amp_skipped_steps = 0
        amp_scale_start = scaler.get_scale()
        optimizer.zero_grad(set_to_none=True)
        oom = False
        try:
            number_of_batches = len(train_loader)
            for batch_index, batch in enumerate(train_loader):
                pixels = batch["pixel_values"].to(device, non_blocking=True)
                target = batch["depth"].to(device, non_blocking=True)
                valid = torch.isfinite(target) & (target >= 0.001) & (target <= args.max_depth)
                group_start = (batch_index // args.grad_accumulation) * args.grad_accumulation
                group_size = min(args.grad_accumulation, number_of_batches - group_start)
                with torch.autocast("cuda", dtype=torch.float16, enabled=args.amp):
                    prediction = model(pixel_values=pixels).predicted_depth
                    loss = silog_loss(prediction, target, valid)
                    scaled_loss = loss / group_size
                scaler.scale(scaled_loss).backward()
                running_loss += float(loss.detach().cpu())
                is_update = ((batch_index + 1) % args.grad_accumulation == 0) or (batch_index + 1 == number_of_batches)
                if is_update:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    updated = scaled_optimizer_step(optimizer, scheduler, scaler)
                    optimizer_steps_completed += int(updated)
                    amp_skipped_steps += int(not updated)
                    optimizer.zero_grad(set_to_none=True)
                if (batch_index + 1) % 32 == 0:
                    print(f"TRAIN_BATCH {batch_index+1}/{number_of_batches}; updates={optimizer_steps_completed}; AMP skipped={amp_skipped_steps}; scale={scaler.get_scale()}", flush=True)
            val_metrics = evaluate_model(model, val_loader, device, args.max_depth)
            torch.cuda.synchronize(device)
        except torch.cuda.OutOfMemoryError as exc:
            oom = True
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            write_json_atomic(run_dir / "run_status.json", {
                "status": "stopped_cuda_oom", "failed_epoch": epoch_number,
                "last_completed_epoch": completed_epoch,
                "message": str(exc)[:1000],
                "resume": "Rerun the identical command with --resume after closing other GPU workloads. If resolution or another signed setting changes, start a new run and rerun preflight.",
            })
            print(f"CUDA out of memory in epoch {epoch_number}; stopped safely. Latest complete checkpoint: epoch {completed_epoch}. Close other GPU workloads and resume with identical settings; rerun preflight for a new lower-resolution run.")
        if oom:
            break

        epoch_seconds = time.perf_counter() - epoch_start
        epoch_durations.append(epoch_seconds)
        mean_epoch_seconds = sum(epoch_durations) / len(epoch_durations)
        remaining_after = args.epochs - epoch_number
        row = {
            "epoch": epoch_number,
            "optimizer_steps_completed": optimizer_steps_completed,
            "amp_skipped_steps": amp_skipped_steps,
            "amp_scale_start": amp_scale_start,
            "amp_scale_end": scaler.get_scale(),
            "train_silog": running_loss / max(len(train_loader), 1),
            "validation": val_metrics,
            "encoder_lr": optimizer.param_groups[0]["lr"],
            "head_lr": optimizer.param_groups[1]["lr"],
            "epoch_seconds": epoch_seconds,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "free_vram_after_epoch_bytes": torch.cuda.mem_get_info(device)[0],
            "mean_epoch_seconds_this_session": mean_epoch_seconds,
            "estimated_total_seconds_at_current_rate": mean_epoch_seconds * args.epochs,
            "estimated_remaining_seconds": mean_epoch_seconds * remaining_after,
            "completed_epochs": epoch_number,
            "remaining_epochs": remaining_after,
        }
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row, allow_nan=False))

        improved = val_metrics["rmse_m"] < best_rmse
        if improved:
            best_rmse = float(val_metrics["rmse_m"])
            best_epoch = epoch_number
            write_json_atomic(run_dir / "best_validation_metrics.json", val_metrics)
        should_save = epoch_number % args.save_every_epochs == 0 or epoch_number == args.epochs or improved
        if should_save:
            checkpoint = save_checkpoint(run_dir, epoch_number, model, optimizer, scheduler, scaler, best_rmse, best_epoch)
            write_json_atomic(run_dir / "latest_checkpoint.json", {"path": checkpoint.name, "completed_epoch": epoch_number})
            if improved:
                write_json_atomic(run_dir / "best_checkpoint.json", {"path": checkpoint.name, "epoch": epoch_number})
            prune_checkpoints(run_dir, best_epoch, args.keep_recent_checkpoints)
            completed_epoch = epoch_number
        should_pause = should_pause_after_epoch(epoch_number, args.epochs, args.stop_after_epoch)
        status = "completed" if epoch_number == args.epochs else ("paused_after_epoch" if should_pause else "running")
        write_json_atomic(run_dir / "run_status.json", {
            "status": status,
            "completed_epochs": completed_epoch,
            "remaining_epochs": max(args.epochs - completed_epoch, 0),
            "best_validation_rmse_m": best_rmse if math.isfinite(best_rmse) else None,
            "best_epoch": best_epoch,
        })
        if should_pause:
            print(f"Paused safely after epoch {completed_epoch}/{args.epochs}; scheduler remains configured for the full schedule.")
            break
    else:
        write_json_atomic(run_dir / "run_status.json", {
            "status": "completed", "completed_epochs": completed_epoch,
            "remaining_epochs": max(args.epochs - completed_epoch, 0),
            "best_validation_rmse_m": best_rmse if math.isfinite(best_rmse) else None,
            "best_epoch": best_epoch,
        })
    print(f"Run status: {(run_dir / 'run_status.json').read_text(encoding='utf-8')}")
    if best_epoch is not None:
        print(f"Best validation RMSE: {best_rmse:.6f} m at epoch {best_epoch}; test data was not read.")
    if json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))["status"] == "stopped_cuda_oom":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
