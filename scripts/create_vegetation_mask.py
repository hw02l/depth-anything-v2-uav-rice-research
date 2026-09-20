from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from da2_experiments.io import read_rgb, write_csv, write_json  # noqa: E402


def exg_mask(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[..., 0].astype(np.float32), rgb[..., 1].astype(np.float32), rgb[..., 2].astype(np.float32)
    return (2.0 * g - r - b) > 20.0


def vari_mask(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[..., 0].astype(np.float32), rgb[..., 1].astype(np.float32), rgb[..., 2].astype(np.float32)
    denominator = g + r - b
    vari = np.divide(g - r, denominator, out=np.zeros_like(g), where=np.abs(denominator) > 1e-6)
    return vari > 0.05


def hsv_mask(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    # Include green through yellow-green foliage, while requiring chroma.
    return (hsv[..., 0] >= 20) & (hsv[..., 0] <= 100) & (hsv[..., 1] >= 45) & (hsv[..., 2] >= 30)


def save_mask(mask: np.ndarray, path: Path) -> None:
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare explainable vegetation masks for rice UAV RGB frames")
    parser.add_argument("--selection-csv", default="results/rice_uav/selected_images.csv")
    parser.add_argument("--output-root", default="outputs/rice_uav")
    parser.add_argument("--results-dir", default="results/rice_uav")
    parser.add_argument("--method", choices=["vari", "exg", "hsv"], default="vari")
    args = parser.parse_args()
    with Path(args.selection_csv).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    summary = []
    for row in rows:
        sample_dir = Path(args.output_root) / row["growth_stage"] / row["date"] / Path(row["filename"]).stem
        rgb = np.asarray(read_rgb(ROOT / row["local_path"]))
        masks = {"exg": exg_mask(rgb), "vari": vari_mask(rgb), "hsv": hsv_mask(rgb)}
        for method, mask in masks.items():
            save_mask(mask, sample_dir / f"{Path(row['filename']).stem}_vegetation_mask_{method}.png")
            summary.append({"date": row["date"], "filename": row["filename"], "growth_stage": row["growth_stage"], "method": method, "vegetation_pixels": int(mask.sum()), "total_pixels": int(mask.size), "vegetation_fraction": float(mask.mean())})
        chosen = masks[args.method]
        save_mask(chosen, sample_dir / f"{Path(row['filename']).stem}_vegetation_mask.png")
        figure, axes = plt.subplots(1, 4, figsize=(18, 5), constrained_layout=True)
        axes[0].imshow(rgb); axes[0].set_title("RGB")
        for axis, method in zip(axes[1:], ["exg", "vari", "hsv"]):
            axis.imshow(masks[method], cmap="gray", vmin=0, vmax=1); axis.set_title(f"{method.upper()} mask ({masks[method].mean():.2f})")
        for axis in axes:
            axis.set_xlabel("pixel x"); axis.set_ylabel("pixel y")
        figure.suptitle(f"Vegetation mask comparison: {row['date']} {row['filename']}")
        figure.savefig(sample_dir / f"{Path(row['filename']).stem}_mask_comparison.png", dpi=120)
        plt.close(figure)
    write_csv(Path(args.results_dir) / "vegetation_mask_method_comparison.csv", summary, ["date", "filename", "growth_stage", "method", "vegetation_pixels", "total_pixels", "vegetation_fraction"])
    write_json(Path(args.results_dir) / "vegetation_mask_selection.json", {"selected_method": args.method, "thresholds": {"exg": "2G-R-B > 20", "vari": "(G-R)/(G+R-B) > 0.05", "hsv": "H 20-100, S >= 45, V >= 30 in OpenCV HSV"}, "reason": "VARI is used as the default because it is dimensionless and less directly dependent on overall brightness; method fractions and visual comparisons are retained for failure analysis."})
    print(f"[rice] wrote vegetation masks for {len(rows)} images; selected method={args.method}")


if __name__ == "__main__":
    main()
