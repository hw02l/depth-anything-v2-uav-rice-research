from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))
from data_patch import NativePatchDataset, read_manifest
from train_patch import _train_epoch, make_optimizer
from uav_metric_research import multiscale as m


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Conv2d(3, 4, 1)
        self.head = torch.nn.Conv2d(4, 1, 1)

    def forward(self, pixel_values):
        prediction = torch.nn.functional.softplus(self.head(self.backbone(pixel_values)))[:, 0] + 1.0
        return SimpleNamespace(predicted_depth=prediction)


def run(args):
    rows = read_manifest(args.train_manifest)
    if len(rows) != 384:
        raise ValueError("Expected canonical Train384")
    row = rows[0]
    results = []
    for group in ("C0", "F1", "F2"):
        dataset = NativePatchDataset([row], args.rgb_root, args.depth_root, group, seed=42)
        dataset.set_epoch(1)
        sample = dataset[0]
        model = ToyModel()
        optimizer = make_optimizer(model)
        scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=1, power=0.9)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        summary = _train_epoch(model, dataset, optimizer, scheduler, scaler,
                               m.frozen("losses").silog_loss, epoch=1, seed=42,
                               device="cpu", amp=False, accumulation=1, max_updates=1)
        meta = sample["metadata"]
        results.append({
            "mode": "real_data_cpu_smoke", "group": group,
            "image_relpath": row["image_relpath"],
            "original_hw": "x".join(map(str, meta["original_hw"])),
            "input_hw": "x".join(map(str, sample["depth"].shape)),
            "crop_box_yxyx": json.dumps(meta["crop_box_yxyx"]),
            "crop_space": meta["crop_space"],
            "native_px_per_input_px_yx": json.dumps(meta["native_px_per_input_px_yx"]),
            "flipped_horizontal": meta["flipped_horizontal"],
            "valid_gt_fraction": meta["valid_gt_fraction"],
            "valid_gt_median_m": meta["valid_gt_median_m"],
            "loss_m": summary["mean_silog_m"],
            "optimizer_updates": summary["optimizer_updates"],
            "amp_skips": summary["amp_skips"],
            "training_seconds": summary["training_seconds"],
            "research_result": False,
        })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "cpu_smoke_results.json"
    csv_path = args.output_dir / "cpu_smoke_results.csv"
    for path in (json_path, csv_path):
        if path.exists():
            raise FileExistsError(f"Smoke output already exists; preserving it: {path}")
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    with csv_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(json.dumps({"mode": "real_data_cpu_smoke", "groups": [r["group"] for r in results],
                      "files": [str(json_path), str(csv_path)], "research_result": False},
                     ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", type=Path, default=ROOT / "configs/uav3dcrop_verified/train.csv")
    parser.add_argument("--rgb-root", type=Path, default=ROOT / "data/uav3dcrop_rgb")
    parser.add_argument("--depth-root", type=Path, default=ROOT / "data/uav3dcrop_depth")
    parser.add_argument("--output-dir", type=Path, default=HERE / "cpu_smoke")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
