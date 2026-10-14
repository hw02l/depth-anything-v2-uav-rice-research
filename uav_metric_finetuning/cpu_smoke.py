from __future__ import annotations

import argparse
import gc
import json
import random
import tempfile
from pathlib import Path

import cv2
import numpy as np
import tifffile
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForDepthEstimation

if __package__:
    from .data import UAVDepthDataset
    from .losses import silog_loss
    from .modeling import build_metric_model
    from .train import save_checkpoint
else:
    from data import UAVDepthDataset
    from losses import silog_loss
    from modeling import build_metric_model
    from train import save_checkpoint

REPO_ROOT = Path(__file__).resolve().parents[1]


def cached_snapshot(cache_root: Path, model_folder: str) -> Path:
    snapshots = cache_root / model_folder / "snapshots"
    candidates = sorted(path for path in snapshots.glob("*") if (path / "model.safetensors").is_file())
    if not candidates:
        raise FileNotFoundError(
            f"Local checkpoint is not cached: {snapshots}. CPU smoke is offline and will not download weights."
        )
    return candidates[-1]


def make_tiny_pair(rgb_root: Path, depth_root: Path, seed: int) -> dict[str, str]:
    generator = np.random.default_rng(seed)
    image_rel = Path("2026/Smoke_Corn/images/synthetic_1.jpg")
    depth_rel = Path("2026/Smoke_Corn_depth/synthetic_1_depth.tif")
    image_path = rgb_root / image_rel
    depth_path = depth_root / depth_rel
    image_path.parent.mkdir(parents=True, exist_ok=True)
    depth_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = 72, 84
    rgb = generator.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    depth = np.linspace(1.0, 18.0, num=height * width, dtype=np.float32).reshape(height, width).astype(np.float16)
    if not cv2.imwrite(str(image_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise IOError(f"Could not create synthetic smoke RGB: {image_path}")
    tifffile.imwrite(depth_path, depth)
    return {
        "split": "train",
        "scene": "2026/Smoke_Corn",
        "day": "2026/Smoke",
        "crop": "corn",
        "view": "nadir",
        "image_relpath": image_rel.as_posix(),
        "depth_relpath": depth_rel.as_posix(),
    }


def one_training_step(
    initialization: str,
    *,
    relative_model: Path,
    metric_model: Path,
    rgb_root: Path,
    depth_root: Path,
    checkpoint_dir: Path,
    seed: int,
) -> dict[str, object]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model, init_report = build_metric_model(
        initialization,
        max_depth=20.0,
        relative_model=str(relative_model),
        metric_model=str(metric_model),
        local_files_only=True,
    )
    model.to("cpu").train()
    dataset = UAVDepthDataset(
        [make_tiny_pair(rgb_root, depth_root, seed)], rgb_root, depth_root,
        image_size=56, training=True,
    )
    batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)))
    pixels = batch["pixel_values"]
    target = batch["depth"]
    valid = torch.isfinite(target) & (target > 0) & (target <= 20)

    optimizer = AdamW(model.parameters(), lr=1e-6)
    optimizer.zero_grad(set_to_none=True)
    prediction = model(pixel_values=pixels).predicted_depth
    loss = silog_loss(prediction, target, valid)
    loss.backward()
    finite_gradients = [p.grad.detach().float() for p in model.parameters() if p.grad is not None]
    if not finite_gradients or not all(torch.isfinite(gradient).all().item() for gradient in finite_gradients):
        raise RuntimeError(f"Backward pass produced no finite gradients for {initialization}")
    grad_norm = float(torch.sqrt(sum(gradient.square().sum() for gradient in finite_gradients)))
    optimizer.step()
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=1, power=0.9)
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    model.eval()
    with torch.inference_mode():
        reference = model(pixel_values=pixels).predicted_depth.detach().clone()
    checkpoint_run_dir = checkpoint_dir / "run"
    checkpoint_run_dir.mkdir(parents=True, exist_ok=False)
    complete_checkpoint = save_checkpoint(
        checkpoint_run_dir, 1, model, optimizer, scheduler, scaler, best_rmse=1.0, best_epoch=1
    )
    reloaded = AutoModelForDepthEstimation.from_pretrained(complete_checkpoint, local_files_only=True).eval()
    with torch.inference_mode():
        restored = reloaded(pixel_values=pixels).predicted_depth
    max_abs_difference = float((reference - restored).abs().max())
    if max_abs_difference > 1e-6:
        raise RuntimeError(f"Checkpoint reload changed predictions by {max_abs_difference}")
    trainer_state = torch.load(complete_checkpoint / "trainer_state.pt", map_location="cpu", weights_only=False)
    if trainer_state["completed_epoch"] != 1 or trainer_state["best_epoch"] != 1:
        raise RuntimeError(f"Trainer-state checkpoint did not reload expected epoch: {trainer_state}")
    report = {
        "initialization": initialization,
        "input_shape": list(pixels.shape),
        "synthetic_pairs": 1,
        "max_depth_m": 20.0,
        "init": init_report,
        "silog_loss": float(loss.detach()),
        "finite_gradient_tensors": len(finite_gradients),
        "gradient_l2_norm": grad_norm,
        "optimizer_step": "passed",
        "checkpoint_reload": "passed",
        "trainer_state_reload": "passed",
        "reload_max_abs_prediction_difference": max_abs_difference,
        "checkpoint_format": "Hugging Face config.json + model.safetensors",
    }
    del model, reloaded, optimizer, dataset, batch, finite_gradients, reference, restored
    gc.collect()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU-only one-step smoke test using synthetic pairs and cached weights.")
    parser.add_argument("--model-cache", type=Path, default=REPO_ROOT / "work" / "hf_cache")
    parser.add_argument(
        "--scratch-root", type=Path,
        default=REPO_ROOT / "work" / "uav_metric_finetuning" / "cpu_smoke_tmp",
        help="Writable location for disposable synthetic data and temporary checkpoints",
    )
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    relative = cached_snapshot(args.model_cache, "models--depth-anything--Depth-Anything-V2-Small-hf")
    metric = cached_snapshot(args.model_cache, "models--depth-anything--Depth-Anything-V2-Metric-Outdoor-Small-hf")
    reports = []
    args.scratch_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="da2-uav-cpu-smoke-", dir=args.scratch_root) as temporary:
        root = Path(temporary)
        rgb_root, depth_root = root / "rgb", root / "depth"
        for initialization in ("relative_encoder", "metric_outdoor"):
            checkpoint_dir = root / f"checkpoint_{initialization}"
            reports.append(one_training_step(
                initialization,
                relative_model=relative,
                metric_model=metric,
                rgb_root=rgb_root,
                depth_root=depth_root,
                checkpoint_dir=checkpoint_dir,
                seed=args.seed,
            ))
    result = {
        "status": "pass",
        "device": "cpu",
        "real_uav3dcrop_frames_used": 0,
        "downloads": 0,
        "package_installs": 0,
        "arms": reports,
    }
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
