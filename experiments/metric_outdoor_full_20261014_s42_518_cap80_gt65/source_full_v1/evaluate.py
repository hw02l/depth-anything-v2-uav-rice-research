from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForDepthEstimation

if __package__:
    from .data import UAVDepthDataset, read_manifest
    from .metrics import evaluate_model
else:
    from data import UAVDepthDataset, read_manifest
    from metrics import evaluate_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved metric checkpoint on validation or held-out test data.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--rgb-root", type=Path, required=True)
    parser.add_argument("--depth-root", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--max-depth", type=float, default=None)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-train", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Refusing to overwrite evaluation output: {args.output}")
    rows = read_manifest(args.manifest)
    split = rows[0]["split"]
    if split == "train" and not args.allow_train:
        parser.error("Training rows are not an evaluation split; pass --allow-train only for diagnostics.")
    if split not in {"validation", "independent_test", "train"}:
        parser.error(f"Unknown split label: {split}")

    model = AutoModelForDepthEstimation.from_pretrained(args.model, local_files_only=True)
    max_depth = float(args.max_depth if args.max_depth is not None else model.config.max_depth)
    model.config.max_depth = max_depth
    model.head.max_depth = max_depth
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    dataset = UAVDepthDataset(rows, args.rgb_root, args.depth_root, image_size=args.image_size, training=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")
    metrics = evaluate_model(model, loader, device, max_depth)
    payload = {
        "model": str(args.model.resolve()),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "split": split,
        "pairs": len(rows),
        "max_depth_m": max_depth,
        "device": str(device),
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
