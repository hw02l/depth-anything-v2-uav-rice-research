from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create representative GT/DA2 depth profiles for UAV3DCrop")
    parser.add_argument("--selection-csv", default="results/uav3dcrop/all_selected_images.csv")
    parser.add_argument("--output-root", default="outputs/uav3dcrop")
    parser.add_argument("--profile-dir", default="outputs/uav3dcrop/all_depth_profiles")
    args = parser.parse_args()
    with (ROOT / args.selection_csv).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected: dict[tuple[str, str], dict] = {}
    for row in rows:
        selected.setdefault((row["scene"], row["view_type"]), row)
    destination = ROOT / args.profile_dir
    destination.mkdir(parents=True, exist_ok=True)
    for row in selected.values():
        stem = Path(row["filename"]).stem
        directory = ROOT / args.output_root / row["scene"].replace("/", "_") / row["view_type"] / stem
        pred = np.load(directory / f"{stem}_metric_depth_m.npy").astype(np.float32)
        gt = tifffile.imread(ROOT / row["local_depth_path"]).astype(np.float32)
        y = gt.shape[0] // 2
        stride = max(1, gt.shape[1] // 1600)
        x = np.arange(gt.shape[1])
        figure, axis = plt.subplots(figsize=(14, 5))
        axis.plot(x[::stride], gt[y, ::stride], label="UAV3DCrop GT z-depth", linewidth=1.2)
        axis.plot(x[::stride], pred[y, ::stride], label="DA2 metric depth", linewidth=1.0)
        bias = float(np.median((pred - gt)[np.isfinite(gt) & (gt > 0) & np.isfinite(pred) & (pred > 0)]))
        axis.plot(x[::stride], (pred - bias)[y, ::stride], label="Bias-corrected DA2", linewidth=1.0)
        axis.set_title(f"Depth profile: {row['scene']} | {row['view_type']} | {row['filename']} | y={y}")
        axis.set_xlabel("Pixel x")
        axis.set_ylabel("Camera-frame z-depth (m)")
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(destination / f"{row['scene'].replace('/', '_')}_{row['view_type']}_{stem}_profile.png", dpi=170)
        plt.close(figure)
    print(f"[uav3dcrop] wrote {len(selected)} representative profiles to {destination}")


if __name__ == "__main__":
    main()
