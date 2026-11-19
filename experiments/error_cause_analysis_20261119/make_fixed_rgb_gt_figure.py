#!/usr/bin/env python3
"""Render RGB/GT-only context sheets for the previously fixed B-series examples.

No model prediction is generated. Representative image IDs are reused exactly
from the existing B-series comparison figure set.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile

FIXED_IMAGE_IDS = (7, 20, 34, 48, 71, 89, 97, 112)


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    repo, out = args.repo_root.resolve(), args.output_dir.resolve()
    rgb_root = repo / "data/uav3dcrop_rgb"
    depth_root = repo / "data/uav3dcrop_depth"
    inv_path = repo / "experiments/multiscale_tile_inference_20261016/image_inventory.csv"
    with inv_path.open(encoding="utf-8-sig", newline="") as f:
        inventory = list(csv.DictReader(f))
    by_index = {int(r["image_index"]): r for r in inventory}
    if set(FIXED_IMAGE_IDS) - set(by_index):
        raise ValueError("one or more fixed B-series image IDs are absent from image_inventory.csv")
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    fig, axes = plt.subplots(len(FIXED_IMAGE_IDS), 2, figsize=(10, 20), constrained_layout=True)
    integrity = []
    for row_i, image_id in enumerate(FIXED_IMAGE_IDS):
        entry = by_index[image_id]
        rgb_path = (rgb_root / entry["image_relpath"]).resolve()
        depth_path = (depth_root / entry["depth_relpath"]).resolve()
        if rgb_root.resolve() not in rgb_path.parents or depth_root.resolve() not in depth_path.parents:
            raise ValueError("manifest path escaped the fixed dataset roots")
        bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(rgb_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        depth = np.asarray(tifffile.imread(depth_path), dtype=np.float32)
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        if depth.ndim != 2 or depth.shape != rgb.shape[:2]:
            raise ValueError(f"RGB/GT shape mismatch for image index {image_id}: {rgb.shape} / {depth.shape}")
        valid = np.isfinite(depth) & (depth > 0.0) & (depth <= 65.0)
        # The inventory hashes are source-file hashes; verify before rendering.
        rgb_hash, depth_hash = file_sha(rgb_path), file_sha(depth_path)
        if rgb_hash != entry["rgb_sha256"] or depth_hash != entry["gt_sha256"]:
            raise AssertionError(f"raw file hash mismatch for image index {image_id}")
        axes[row_i, 0].imshow(rgb)
        axes[row_i, 0].set_title(f"{entry['crop'].title()} / {entry['view']}  |  RGB  |  idx={image_id}", loc="left")
        masked = np.ma.masked_where(~valid, depth)
        axes[row_i, 1].imshow(masked, cmap="viridis", vmin=10.0, vmax=45.0, interpolation="nearest")
        p05, p95 = np.percentile(depth[valid], [5, 95])
        axes[row_i, 1].set_title(f"GT camera Z-depth (m)  |  p05–p95={p05:.2f}–{p95:.2f} m", loc="left")
        for ax in axes[row_i]:
            ax.set_xticks([]); ax.set_yticks([])
        integrity.append({
            "image_index": image_id, "image_relpath": entry["image_relpath"],
            "depth_relpath": entry["depth_relpath"], "crop": entry["crop"], "view": entry["view"],
            "rgb_shape": list(rgb.shape), "depth_shape": list(depth.shape),
            "rgb_dtype": str(rgb.dtype), "depth_dtype_source": str(tifffile.imread(depth_path).dtype),
            "valid_gt_pixels": int(valid.sum()), "rgb_file_sha256": rgb_hash,
            "depth_file_sha256": depth_hash, "expected_rgb_sha256": entry["rgb_sha256"],
            "expected_gt_sha256": entry["gt_sha256"],
        })
    fig.suptitle("Previously fixed B-series examples: RGB and GT only\nGT panels share a fixed 10–45 m display scale; invalid pixels are masked", fontsize=12)
    fig.colorbar(axes[0, 1].images[0], ax=axes[:, 1].tolist(), shrink=0.65, label="Camera Z-depth (m)")
    fig.savefig(out / "fixed_rgb_gt_examples.png", dpi=140)
    fig.savefig(out / "fixed_rgb_gt_examples.jpg", dpi=105, pil_kwargs={"quality": 90, "optimize": True})
    plt.close(fig)
    with (out / "fixed_rgb_gt_integrity.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(integrity[0]))
        w.writeheader(); w.writerows(integrity)
    (out / "fixed_rgb_gt_figure_metadata.json").write_text(json.dumps({
        "fixed_image_indices": FIXED_IMAGE_IDS,
        "selection_source": "reused image indices from multiscale_tile_inference_20261016/figures/depth_comparison_ep20_* and pilot_selection.csv",
        "display_only": "RGB and GT; no new predictions or inference",
        "gt_mask": "finite and 0 < depth <= 65m",
        "shared_display_range_m": [10, 45],
        "values_outside_display_range": "saturated at display limits",
        "per_image_file_hashes_verified_against": "multiscale_tile_inference_20261016/image_inventory.csv",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
