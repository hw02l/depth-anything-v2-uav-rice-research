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


# The late-flight frames are visually dominated by dense/yellow canopy.  The
# appearance-only ground heuristic then labels yellow leaves as non-vegetation
# and would produce a misleading proxy.  Keep the candidate mask for visual
# inspection, but do not use it as a ground reference in the numeric result.
MANUAL_NO_GROUND_REFERENCE_DATES = {"2025-10-03"}


def paths(row: dict, output_root: Path) -> tuple[Path, Path, Path]:
    sample = output_root / row["growth_stage"] / row["date"] / Path(row["filename"]).stem
    stem = Path(row["filename"]).stem
    return sample / f"{stem}_metric_depth_m.npy", sample / f"{stem}_vegetation_mask.png", sample


def ground_candidate(rgb: np.ndarray, vegetation: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    nonvegetation = ~vegetation
    # Conservative soil/water/ground proxy: exclude the outer 2% border,
    # saturated bright yellow/green objects, and very bright reflections.
    height, width = vegetation.shape
    interior = np.zeros_like(vegetation, dtype=bool)
    y0, y1 = int(height * 0.02), int(height * 0.98)
    x0, x1 = int(width * 0.02), int(width * 0.98)
    interior[y0:y1, x0:x1] = True
    low_or_neutral_chroma = (hsv[..., 1] < 150) | (hsv[..., 2] < 90)
    valid_brightness = (hsv[..., 2] > 20) & (hsv[..., 2] < 245)
    candidate = nonvegetation & interior & low_or_neutral_chroma & valid_brightness
    # Remove isolated single pixels but preserve narrow gaps between rows.
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(candidate.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)


def sample_profile(depth: np.ndarray, vegetation: np.ndarray, ground: np.ndarray, y: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.arange(depth.shape[1])
    values = depth[y]
    return x, values, vegetation[y], ground[y]


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate conservative ground candidates and canopy-ground depth difference proxies")
    parser.add_argument("--selection-csv", default="results/rice_uav/selected_images.csv")
    parser.add_argument("--output-root", default="outputs/rice_uav")
    parser.add_argument("--results-dir", default="results/metrics")
    parser.add_argument("--vegetation-method", default="vari")
    parser.add_argument("--min-ground-pixels", type=int, default=2000)
    args = parser.parse_args()
    with Path(args.selection_csv).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    output_root = Path(args.output_root)
    result_rows = []
    for row in rows:
        depth_path, mask_path, sample_dir = paths(row, output_root)
        rgb = np.asarray(read_rgb(ROOT / row["local_path"]))
        depth = np.load(depth_path).astype(np.float32)
        vegetation = np.asarray(Image.open(mask_path).convert("L")) > 127
        ground = ground_candidate(rgb, vegetation)
        stem = Path(row["filename"]).stem
        Image.fromarray((ground.astype(np.uint8) * 255), mode="L").save(sample_dir / f"{stem}_ground_candidate.png")

        valid_depth = np.isfinite(depth) & (depth > 0)
        canopy_valid = vegetation & valid_depth
        ground_valid = ground & valid_depth
        ground_values = depth[ground_valid]
        canopy_values = depth[canopy_valid]
        manual_no_ground_reference = row["date"] in MANUAL_NO_GROUND_REFERENCE_DATES
        if manual_no_ground_reference:
            flag = "NO_GROUND_REFERENCE"
            proxy = {"median_m": np.nan, "p25_m": np.nan, "p10_m": np.nan, "p05_m": np.nan}
        elif len(ground_values) < args.min_ground_pixels:
            flag = "NO_GROUND_REFERENCE"
            proxy = {"median_m": np.nan, "p25_m": np.nan, "p10_m": np.nan, "p05_m": np.nan}
        elif len(canopy_values) == 0:
            flag = "NO_VEGETATION_REFERENCE"
            proxy = {"median_m": np.nan, "p25_m": np.nan, "p10_m": np.nan, "p05_m": np.nan}
        else:
            ground_median = float(np.median(ground_values))
            proxy = {
                "median_m": ground_median - float(np.median(canopy_values)),
                "p25_m": ground_median - float(np.percentile(canopy_values, 25)),
                "p10_m": ground_median - float(np.percentile(canopy_values, 10)),
                "p05_m": ground_median - float(np.percentile(canopy_values, 5)),
            }
            ground_spread = float(np.percentile(ground_values, 95) - np.percentile(ground_values, 5))
            flag = "GROUND_REFERENCE_AVAILABLE" if ground_spread < 4.0 else "GROUND_REFERENCE_HIGH_VARIATION"

        overlay = rgb.copy()
        overlay[vegetation] = (0.55 * overlay[vegetation] + 0.45 * np.array([0, 220, 0])).astype(np.uint8)
        overlay[ground] = (0.55 * overlay[ground] + 0.45 * np.array([220, 40, 0])).astype(np.uint8)
        Image.fromarray(overlay).save(sample_dir / f"{stem}_ground_vegetation_overlay.png")

        y = rgb.shape[0] // 2
        x, profile, profile_vegetation, profile_ground = sample_profile(depth, vegetation, ground, y)
        figure, axis = plt.subplots(figsize=(12, 4.5))
        finite = np.isfinite(profile) & (profile > 0)
        axis.plot(x[finite], profile[finite], color="#333333", linewidth=0.8, label="metric depth")
        axis.scatter(x[profile_vegetation & finite], profile[profile_vegetation & finite], s=2, color="green", label="vegetation")
        axis.scatter(x[profile_ground & finite], profile[profile_ground & finite], s=4, color="red", label="ground candidate")
        axis.set_title(f"Depth profile at y={y}: {row['date']} {row['filename']}")
        axis.set_xlabel("Pixel x along horizontal profile")
        axis.set_ylabel("Camera-to-surface depth (m)")
        axis.legend()
        figure.tight_layout()
        profile_dir = Path("outputs/rice_uav/plots/profiles")
        profile_dir.mkdir(parents=True, exist_ok=True)
        figure.savefig(profile_dir / f"{row['date']}_{stem}_depth_profile.png", dpi=160)
        plt.close(figure)

        ground_median = float(np.median(ground_values)) if len(ground_values) else np.nan
        result_rows.append({
            "date": row["date"],
            "filename": row["filename"],
            "growth_stage": row["growth_stage"],
            "ground_depth_m": ground_median,
            "canopy_depth_m": float(np.median(canopy_values)) if len(canopy_values) else np.nan,
            "canopy_depth_p25_m": float(np.percentile(canopy_values, 25)) if len(canopy_values) else np.nan,
            "canopy_depth_p10_m": float(np.percentile(canopy_values, 10)) if len(canopy_values) else np.nan,
            "canopy_depth_p05_m": float(np.percentile(canopy_values, 5)) if len(canopy_values) else np.nan,
            "depth_difference_median_m": proxy["median_m"],
            "depth_difference_p25_m": proxy["p25_m"],
            "depth_difference_p10_m": proxy["p10_m"],
            "depth_difference_p05_m": proxy["p05_m"],
            "vegetation_pixel_count": int(canopy_valid.sum()),
            "ground_candidate_pixel_count": int(ground_valid.sum()),
            "ground_candidate_fraction": float(ground.mean()),
            "reliability_flag": flag,
            "interpretation": "canopy-ground depth difference proxy; not plant height",
        })

    fields = list(result_rows[0].keys()) if result_rows else []
    write_csv(Path(args.results_dir) / "rice_uav_depth_difference.csv", result_rows, fields)
    write_json(Path(args.results_dir) / "rice_uav_ground_method.json", {"vegetation_method": args.vegetation_method, "ground_rule": "non-vegetation AND interior 96% AND HSV saturation<150 OR value<90 AND 20<value<245, 3x3 opening", "min_ground_pixels": args.min_ground_pixels, "manual_no_ground_reference_dates": sorted(MANUAL_NO_GROUND_REFERENCE_DATES), "caution": "ground candidate is a conservative appearance-based proxy; it is not a ground-truth label. The late 2025-10-03 frames were manually excluded from proxy computation after visual inspection because candidate pixels were dominated by yellow/brown canopy rather than exposed ground.", "rows": result_rows})
    print(f"[rice] wrote canopy-ground proxies for {len(result_rows)} images")


if __name__ == "__main__":
    main()
