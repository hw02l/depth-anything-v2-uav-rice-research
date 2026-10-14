from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

if __package__:
    from .losses import silog_loss
    from .modeling import METRIC_OUTDOOR_MODEL_ID, RELATIVE_MODEL_ID, build_metric_model
else:
    from losses import silog_loss
    from modeling import METRIC_OUTDOOR_MODEL_ID, RELATIVE_MODEL_ID, build_metric_model


def require_finite_gradients(loss, parameters):
    gradients = [p.grad for p in parameters if p.grad is not None]
    if not torch.isfinite(loss).all().item() or not gradients or not all(torch.isfinite(g).all().item() for g in gradients):
        raise RuntimeError("損失または勾配に数値異常があります。AMPによる更新スキップをメモリ計測の成功として扱えません。設定を確認して再計測してください。")


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure one synthetic CUDA training step before a real run.")
    parser.add_argument("--initialization", choices=("relative_encoder", "metric_outdoor"), required=True)
    parser.add_argument("--relative-model", default=RELATIVE_MODEL_ID, help="Resolved local cache path for Relative Small")
    parser.add_argument("--metric-model", default=METRIC_OUTDOOR_MODEL_ID, help="Resolved local cache path for Metric Outdoor Small")
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-depth", type=float, required=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable. This is a GPU-only cell; prior CPU preparation can still be completed.")
    if args.image_size < 56 or args.image_size % 14 or args.batch_size < 1 or args.max_depth <= 0:
        parser.error("image-size must be >=56 and divisible by 14; batch-size and max-depth must be positive")
    if args.output.exists():
        parser.error(f"Refusing to overwrite existing preflight report: {args.output}")

    device = torch.device("cuda")
    free_before, total_before = torch.cuda.mem_get_info(device)
    result = {
        "status": "running",
        "device": torch.cuda.get_device_name(0),
        "total_vram_bytes": torch.cuda.get_device_properties(0).total_memory,
        "total_vram_gib": torch.cuda.get_device_properties(0).total_memory / 1024**3,
        "free_vram_before_model_bytes": free_before,
        "training_feasible": False,
        "feasibility_scope": "Synthetic square training step with headroom; real data/validation can still OOM.",
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "amp": args.amp,
        "initialization": None,
        "peak_allocated_bytes_one_synthetic_step": None,
        "peak_reserved_bytes_one_synthetic_step": None,
        "optimizer_step": "synthetic-only; no dataset samples or checkpoint were used",
    }
    started = time.perf_counter()
    try:
        model, init_summary = build_metric_model(
            args.initialization,
            max_depth=args.max_depth,
            relative_model=args.relative_model,
            metric_model=args.metric_model,
            local_files_only=True,
        )
        result["initialization"] = init_summary
        model = model.to(device).train()
        total_params = sum(parameter.numel() for parameter in model.parameters())
        trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        result["parameters"] = total_params
        result["trainable_parameters"] = trainable_params
        result["static_parameter_gradient_adam_bytes_lower_bound"] = 4 * total_params + 12 * trainable_params
        result["static_lower_bound_gib_excludes_activations"] = (4 * total_params + 12 * trainable_params) / 1024**3
        torch.cuda.synchronize(device)
        baseline_allocated = torch.cuda.memory_allocated(device)
        torch.cuda.empty_cache()
        baseline_reserved = torch.cuda.memory_reserved(device)
        torch.cuda.reset_peak_memory_stats(device)
        result["baseline_model_allocated_bytes"] = baseline_allocated
        result["baseline_model_reserved_bytes"] = baseline_reserved

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
        pixels = torch.randn(args.batch_size, 3, args.image_size, args.image_size, device=device)
        target = (torch.rand(args.batch_size, args.image_size, args.image_size, device=device) * args.max_depth).clamp_min(0.1)
        valid = torch.isfinite(target) & (target > 0)
        scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=args.amp):
            prediction = model(pixel_values=pixels).predicted_depth
            loss = silog_loss(prediction, target, valid)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        require_finite_gradients(loss, model.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        if not optimizer.state:
            raise RuntimeError("optimizer状態が作成されませんでした。事前計測を成功と判定しません。")
        torch.cuda.synchronize(device)
        result.update({
            "status": "pass",
            "peak_allocated_bytes_one_synthetic_step": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes_one_synthetic_step": torch.cuda.max_memory_reserved(device),
            "loss": float(loss.detach().cpu()),
            "elapsed_seconds": time.perf_counter() - started,
        })
        required = baseline_reserved + int(max(result["peak_reserved_bytes_one_synthetic_step"] - baseline_reserved, 0) * 1.15) + 512 * 1024**2
        result["required_vram_with_margin_bytes"] = required
        result["free_vram_after_step_bytes"] = torch.cuda.mem_get_info(device)[0]
        result["training_feasible"] = free_before >= required
        if not result["training_feasible"]:
            result["status"] = "insufficient_headroom"
            result["message"] = "合成stepは完了しましたがVRAMの安全余裕が不足しています。学習を開始せず設定を確認してください。"
    except torch.cuda.OutOfMemoryError as exc:
        result["peak_allocated_bytes_one_synthetic_step"] = torch.cuda.max_memory_allocated(device)
        result["peak_reserved_bytes_one_synthetic_step"] = torch.cuda.max_memory_reserved(device)
        torch.cuda.empty_cache()
        result.update({"status": "stopped_cuda_oom", "message": str(exc)[:1000], "elapsed_seconds": time.perf_counter() - started})
    except Exception as exc:
        result.update({"status": "failed", "message": f"{type(exc).__name__}: {exc}", "elapsed_seconds": time.perf_counter() - started})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        raise
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))
    if result["status"] != "pass":
        raise SystemExit("GPU事前計測が通りませんでした。学習を開始せずレポートを確認してください。392px/280pxは手動変更後に再計測してください。")


if __name__ == "__main__":
    main()
