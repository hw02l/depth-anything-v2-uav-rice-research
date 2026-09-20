from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "work" / "mplconfig"))
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


STAGE_ORDER = ["early", "middle", "late"]
STAGE_COLORS = {"early": "#2ca02c", "middle": "#ff7f0e", "late": "#d62728"}


def sample_dir(row: dict, output_root: Path) -> Path:
    return output_root / row["growth_stage"] / row["date"] / Path(row["filename"]).stem


def load_rows(selection_csv: Path) -> list[dict]:
    with selection_csv.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def open_small(path: Path, max_size: tuple[int, int] = (1200, 700)) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    image.thumbnail(max_size, Image.Resampling.LANCZOS)
    return np.asarray(image)


def show_image(axis, path: Path, title: str, cmap: str | None = None) -> None:
    image = Image.open(path)
    image.thumbnail((1200, 700), Image.Resampling.LANCZOS)
    axis.imshow(np.asarray(image), cmap=cmap)
    axis.set_title(title)
    axis.axis("off")


def representative_rows(rows: list[dict]) -> list[dict]:
    return [next(row for row in rows if row["growth_stage"] == stage) for stage in STAGE_ORDER]


def save_figure(figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def figure_1_rgb(rows: list[dict], output_root: Path, report_dir: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(18, 6))
    for axis, row in zip(axes, representative_rows(rows)):
        directory = sample_dir(row, output_root)
        show_image(axis, directory / row["filename"], f"{row['growth_stage']} | {row['date']} | {row['filename']}")
    figure.suptitle("Figure 1. Public rice-field UAV RGB frames", fontsize=15)
    save_figure(figure, report_dir / "figure_1_public_rgb.png")


def figure_2_depth(rows: list[dict], output_root: Path, report_dir: Path) -> None:
    figure, axes = plt.subplots(3, 3, figsize=(18, 15))
    for row_index, row in enumerate(representative_rows(rows)):
        directory = sample_dir(row, output_root)
        stem = Path(row["filename"]).stem
        show_image(axes[row_index, 0], directory / row["filename"], f"{row['growth_stage']} RGB")
        show_image(axes[row_index, 1], directory / f"{stem}_relative_depth.png", "Relative depth")
        show_image(axes[row_index, 2], directory / f"{stem}_metric_depth_m.png", "Outdoor metric depth (m)")
    figure.suptitle("Figure 2. RGB, relative depth, and outdoor metric depth", fontsize=15)
    save_figure(figure, report_dir / "figure_2_rgb_relative_metric.png")


def figure_3_mask(rows: list[dict], output_root: Path, report_dir: Path) -> None:
    figure, axes = plt.subplots(3, 3, figsize=(18, 15))
    for row_index, row in enumerate(representative_rows(rows)):
        directory = sample_dir(row, output_root)
        stem = Path(row["filename"]).stem
        show_image(axes[row_index, 0], directory / row["filename"], f"{row['growth_stage']} RGB")
        show_image(axes[row_index, 1], directory / f"{stem}_vegetation_mask.png", "Selected VARI mask", cmap="gray")
        show_image(axes[row_index, 2], directory / f"{stem}_mask_comparison.png", "ExG / VARI / HSV comparison")
    figure.suptitle("Figure 3. Vegetation-mask selection and visual comparison", fontsize=15)
    save_figure(figure, report_dir / "figure_3_vegetation_mask.png")


def figure_4_ground(rows: list[dict], output_root: Path, report_dir: Path) -> None:
    figure, axes = plt.subplots(3, 3, figsize=(18, 15))
    for row_index, row in enumerate(representative_rows(rows)):
        directory = sample_dir(row, output_root)
        stem = Path(row["filename"]).stem
        show_image(axes[row_index, 0], directory / row["filename"], f"{row['growth_stage']} RGB")
        show_image(axes[row_index, 1], directory / f"{stem}_ground_candidate.png", "Ground candidate", cmap="gray")
        show_image(axes[row_index, 2], directory / f"{stem}_ground_vegetation_overlay.png", "Green=vegetation, red=candidate")
    figure.suptitle("Figure 4. Vegetation and conservative ground-candidate regions", fontsize=15)
    save_figure(figure, report_dir / "figure_4_ground_candidate.png")


def figure_5_profiles(rows: list[dict], output_root: Path, report_dir: Path, results_dir: Path) -> None:
    with (results_dir / "rice_uav_depth_difference.csv").open(encoding="utf-8", newline="") as handle:
        proxy_rows = list(csv.DictReader(handle))
    valid = [row for row in proxy_rows if row["reliability_flag"] != "NO_GROUND_REFERENCE"]
    selected = [next((row for row in valid if row["growth_stage"] == stage), valid[0]) for stage in STAGE_ORDER]
    figure, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    for axis, proxy_row in zip(axes, selected):
        profile = ROOT / "outputs" / "rice_uav" / "plots" / "profiles" / f"{proxy_row['date']}_{Path(proxy_row['filename']).stem}_depth_profile.png"
        image = Image.open(profile)
        image.thumbnail((1400, 700), Image.Resampling.LANCZOS)
        axis.imshow(np.asarray(image))
        axis.set_title(f"{proxy_row['growth_stage']} | {proxy_row['date']} {proxy_row['filename']}")
        axis.axis("off")
    figure.suptitle("Figure 5. Horizontal metric-depth profiles (illustrative local transitions)", fontsize=15)
    save_figure(figure, report_dir / "figure_5_ground_canopy_depth_profile.png")


def figure_6_stage_comparison(rows: list[dict], output_root: Path, report_dir: Path, results_dir: Path) -> None:
    rng = np.random.default_rng(7)
    pixel_values: dict[str, list[float]] = {stage: [] for stage in STAGE_ORDER}
    for row in rows:
        depth = np.load(sample_dir(row, output_root) / f"{Path(row['filename']).stem}_metric_depth_m.npy")
        finite = depth[np.isfinite(depth) & (depth > 0)]
        if len(finite) > 6000:
            finite = rng.choice(finite, 6000, replace=False)
        pixel_values[row["growth_stage"]].extend(finite.astype(float).tolist())
    with (results_dir / "rice_uav_depth_difference.csv").open(encoding="utf-8", newline="") as handle:
        proxy_rows = list(csv.DictReader(handle))
    figure, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].boxplot([pixel_values[stage] for stage in STAGE_ORDER], tick_labels=STAGE_ORDER, showfliers=False, patch_artist=True,
                    boxprops={"facecolor": "#9ecae1"})
    axes[0].set_title("Metric-depth distribution by growth-stage group")
    axes[0].set_xlabel("Growth-stage group (different images/locations)")
    axes[0].set_ylabel("Estimated camera-to-surface depth (m)")
    proxy_values = []
    for stage in STAGE_ORDER:
        values = [float(row["depth_difference_median_m"]) for row in proxy_rows
                  if row["growth_stage"] == stage and row["depth_difference_median_m"] not in ("", "nan", "NaN")]
        proxy_values.append(values)
    axes[1].boxplot(proxy_values, tick_labels=STAGE_ORDER, showfliers=True, patch_artist=True,
                    boxprops={"facecolor": "#fdd0a2"})
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("Canopy-ground depth-difference proxy")
    axes[1].set_xlabel("Growth-stage group (not longitudinally matched)")
    axes[1].set_ylabel("Ground median − canopy median (m)")
    figure.suptitle("Figure 6. Exploratory comparison across public image groups", fontsize=15)
    save_figure(figure, report_dir / "figure_6_growth_stage_comparison.png")


def figure_7_failure(rows: list[dict], output_root: Path, report_dir: Path) -> None:
    row = next(row for row in rows if row["date"] == "2025-10-03" and Path(row["filename"]).stem == "frame132")
    directory = sample_dir(row, output_root)
    stem = Path(row["filename"]).stem
    figure, axes = plt.subplots(1, 4, figsize=(20, 5.5))
    show_image(axes[0], directory / row["filename"], "RGB: dense/yellow canopy")
    show_image(axes[1], directory / f"{stem}_relative_depth.png", "Relative depth")
    show_image(axes[2], directory / f"{stem}_metric_depth_m.png", "Metric depth (m)")
    show_image(axes[3], directory / f"{stem}_ground_vegetation_overlay.png", "Candidate failure: red is not trusted ground")
    figure.suptitle("Figure 7. Failure case: dense late-stage canopy and false ground candidates", fontsize=15)
    save_figure(figure, report_dir / "figure_7_failure_case.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create report-ready figures for the public rice UAV experiment")
    parser.add_argument("--selection-csv", default="results/rice_uav/selected_images.csv")
    parser.add_argument("--output-root", default="outputs/rice_uav")
    parser.add_argument("--results-dir", default="results/metrics")
    parser.add_argument("--report-dir", default="outputs/rice_uav/report_figures")
    args = parser.parse_args()
    rows = load_rows(ROOT / args.selection_csv)
    output_root = ROOT / args.output_root
    report_dir = ROOT / args.report_dir
    results_dir = ROOT / args.results_dir
    figure_1_rgb(rows, output_root, report_dir)
    figure_2_depth(rows, output_root, report_dir)
    figure_3_mask(rows, output_root, report_dir)
    figure_4_ground(rows, output_root, report_dir)
    figure_5_profiles(rows, output_root, report_dir, results_dir)
    figure_6_stage_comparison(rows, output_root, report_dir, results_dir)
    figure_7_failure(rows, output_root, report_dir)
    print(f"[rice] wrote report figures to {report_dir}")


if __name__ == "__main__":
    main()
