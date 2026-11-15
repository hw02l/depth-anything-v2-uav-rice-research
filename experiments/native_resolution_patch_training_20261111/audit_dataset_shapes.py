from __future__ import annotations

import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from data_patch import read_manifest


def main():
    train_path = ROOT / "configs/uav3dcrop_verified/train.csv"
    val_path = ROOT / "configs/uav3dcrop_verified/validation.csv"
    rows = read_manifest(train_path) + read_manifest(val_path)
    if len(rows) != 512:
        raise ValueError(f"Expected exactly 512 train/validation pairs; got {len(rows)}")
    rgb_root, depth_root = ROOT / "data/uav3dcrop_rgb", ROOT / "data/uav3dcrop_depth"
    shape_counts = Counter()
    output = []
    started = time.perf_counter()
    for index, row in enumerate(rows, 1):
        image_path = rgb_root / row["image_relpath"]
        depth_path = depth_root / row["depth_relpath"]
        rgb = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if rgb is None:
            raise FileNotFoundError(image_path)
        depth = np.asarray(tifffile.imread(depth_path))
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        if depth.ndim != 2:
            raise ValueError(f"GT is not one-channel: {row['depth_relpath']} {depth.shape}")
        if rgb.shape[:2] != depth.shape:
            raise ValueError(f"RGB/GT dimensions differ at {row['image_relpath']}: {rgb.shape[:2]} vs {depth.shape}")
        if min(depth.shape) < 1036:
            raise ValueError(f"Image too small for F1 at {row['image_relpath']}: {depth.shape}")
        shape = f"{depth.shape[0]}x{depth.shape[1]}"
        shape_counts[(row["split"], shape)] += 1
        output.append({
            "split": row["split"], "image_relpath": row["image_relpath"],
            "depth_relpath": row["depth_relpath"], "height": depth.shape[0], "width": depth.shape[1],
            "rgb_gt_match": True,
        })
        if index % 64 == 0:
            print(json.dumps({"verified_pairs": index, "elapsed_seconds": time.perf_counter() - started}), flush=True)
    audit_dir = HERE / "audit"
    audit_dir.mkdir(exist_ok=True)
    csv_path = audit_dir / "train_validation_pair_dimensions.csv"
    json_path = audit_dir / "train_validation_pair_dimensions.json"
    if csv_path.exists() or json_path.exists():
        raise FileExistsError("Pair-dimension audit outputs already exist; preserving them")
    with csv_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader(); writer.writerows(output)
    summary = {
        "status": "pass", "train_pairs": sum(1 for r in rows if r["split"] == "train"),
        "validation_pairs": sum(1 for r in rows if r["split"] == "validation"),
        "independent_test_read": False,
        "shape_counts_by_split": {f"{split}:{shape}": count for (split, shape), count in sorted(shape_counts.items())},
        "pair_count": len(output), "rgb_gt_dimension_mismatches": 0,
        "all_crops_support_1036px": True, "elapsed_seconds": time.perf_counter() - started,
        "csv": csv_path.name,
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
