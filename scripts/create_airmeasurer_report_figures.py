from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def limits(array: np.ndarray, valid: np.ndarray) -> tuple[float, float]:
    values = array[valid & np.isfinite(array)]
    return tuple(np.percentile(values, [1, 99])) if len(values) else (0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create AirMeasurer report and failure-analysis figures")
    parser.add_argument("--selection-csv", default="results/airmeasurer/selected_dates.csv")
    parser.add_argument("--inference-dir", default="outputs/airmeasurer/inference")
    parser.add_argument("--chm-dir", default="outputs/airmeasurer/chm")
    parser.add_argument("--results-dir", default="results/metrics/airmeasurer")
    parser.add_argument("--output-dir", default="outputs/airmeasurer/report_figures")
    args = parser.parse_args()
    rows = read_rows(ROOT / args.selection_csv)
    report = ROOT / args.output_dir
    report.mkdir(parents=True, exist_ok=True)
    for row in rows:
        date = row["date"]
        with rasterio.open(ROOT / row["orthomosaic_path"]) as source:
            rgb = np.moveaxis(source.read([1, 2, 3]), 0, -1)
        with rasterio.open(ROOT / args.chm_dir / date / f"{date}_chm_on_orthomosaic_grid.tif") as source:
            chm = source.read(1).astype(np.float32)
            chm[chm <= -9990] = np.nan
        relative = np.load(ROOT / args.inference_dir / date / f"{date}_relative_depth.npy").astype(np.float32)
        pixel_metrics = read_rows(ROOT / args.results_dir / "airmeasurer_pixel_metrics.csv")
        metric_row = next(item for item in pixel_metrics if item["date"] == date)
        scale = float(metric_row["scale"])
        shift = float(metric_row["shift_m"])
        aligned = scale * relative + shift
        rgb_mask = np.any(rgb > 0, axis=2)
        valid = np.isfinite(chm) & np.isfinite(relative) & rgb_mask
        stride = max(1, int(max(rgb.shape[:2]) / 1000))
        rgb_small = rgb[::stride, ::stride]
        chm_small = chm[::stride, ::stride]
        rel_small = relative[::stride, ::stride]
        aligned_small = aligned[::stride, ::stride]

        # Figure 1: orthomosaic.
        fig, ax = plt.subplots(figsize=(10, 9))
        ax.imshow(rgb_small)
        ax.set_title(f"Figure 1: AirMeasurer rice orthomosaic ({date})")
        ax.set_xlabel("Orthomosaic pixel x")
        ax.set_ylabel("Orthomosaic pixel y")
        fig.tight_layout(); fig.savefig(report / "figure_1_rice_orthomosaic.png", dpi=160); plt.close(fig)

        # Figure 2: RGB / CHM / relative.
        fig, axes = plt.subplots(1, 3, figsize=(18, 7))
        axes[0].imshow(rgb_small); axes[0].set_title("RGB orthomosaic")
        vmin, vmax = limits(chm_small, np.isfinite(chm_small)); plot = axes[1].imshow(chm_small, cmap="viridis", vmin=vmin, vmax=vmax); axes[1].set_title("CHM height (m)"); fig.colorbar(plot, ax=axes[1], fraction=0.046)
        vmin, vmax = limits(rel_small, np.isfinite(rel_small)); plot = axes[2].imshow(rel_small, cmap="turbo", vmin=vmin, vmax=vmax); axes[2].set_title("DA2 relative depth (raw, near-large)"); fig.colorbar(plot, ax=axes[2], fraction=0.046)
        for axis in axes: axis.set_xlabel("Orthomosaic pixel x"); axis.set_ylabel("Orthomosaic pixel y")
        fig.suptitle("Figure 2: RGB / CHM / Depth Anything V2 relative depth")
        fig.tight_layout(); fig.savefig(report / "figure_2_rgb_chm_relative.png", dpi=160); plt.close(fig)

        # Pixel samples for scatter figures.
        rng = np.random.default_rng(20260920)
        indices = np.flatnonzero(valid)
        if len(indices) > 100000:
            indices = rng.choice(indices, 100000, replace=False)
        gt = chm.flat[indices].astype(float)
        raw = relative.flat[indices].astype(float)
        calibrated = aligned.flat[indices].astype(float)

        fig, ax = plt.subplots(figsize=(8, 7))
        ax.scatter(raw, gt, s=2, alpha=0.15)
        ax.set_title("Figure 3: CHM height vs DA2 relative score")
        ax.set_xlabel("DA2 relative depth (raw, arbitrary scale)")
        ax.set_ylabel("CHM canopy height (m)")
        ax.grid(alpha=0.25); fig.tight_layout(); fig.savefig(report / "figure_3_chm_vs_relative_scatter.png", dpi=170); plt.close(fig)

        bins = read_rows(ROOT / args.results_dir / "airmeasurer_height_bins.csv")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar([row["height_bin_m"] for row in bins], [float(row["aligned_mae_m"]) for row in bins], color="tab:orange")
        ax.set_title("Figure 4: Height-bin diagnostic error")
        ax.set_xlabel("CHM height bin (m)")
        ax.set_ylabel("Aligned DA2 MAE (m)")
        ax.grid(axis="y", alpha=0.25); fig.tight_layout(); fig.savefig(report / "figure_4_height_bin_error.png", dpi=170); plt.close(fig)

        # Figure 5 explicitly records unavailable plot-level mapping.
        fig, ax = plt.subplots(figsize=(10, 5)); ax.axis("off")
        ax.text(0.02, 0.72, "Plot-level evaluation: NOT AVAILABLE", fontsize=20, weight="bold")
        ax.text(0.02, 0.48, "The downloaded small-field shapefile contains four PointZ RTK control points.\nIt does not contain plot polygons, plot IDs, or a matched height table.", fontsize=15, va="top")
        ax.text(0.02, 0.20, "No plot-level correlation or held-out plot height error is reported.", fontsize=15, va="top", color="firebrick")
        fig.suptitle("Figure 5: Plot-level CHM height vs DA2 score")
        fig.tight_layout(); fig.savefig(report / "figure_5_plot_level_unavailable.png", dpi=170); plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 7))
        ax.scatter(gt, calibrated, s=2, alpha=0.15, label="pixels")
        lo, hi = float(min(np.min(gt), np.min(calibrated))), float(max(np.max(gt), np.max(calibrated)))
        ax.plot([lo, hi], [lo, hi], "k--", label="ideal y=x")
        ax.set_title("Figure 6: Calibrated DA2 diagnostic height vs CHM")
        ax.set_xlabel("CHM height (m)")
        ax.set_ylabel("DA2 scale+shift diagnostic height (m)")
        ax.legend(); ax.grid(alpha=0.25); fig.tight_layout(); fig.savefig(report / "figure_6_calibrated_height_scatter.png", dpi=170); plt.close(fig)

        # Figure 7: resample local pairs at 32 px.
        separation = 32
        y = rng.integers(0, chm.shape[0] - separation, 100000)
        x = rng.integers(0, chm.shape[1] - separation, 100000)
        pair_valid = valid[y, x] & valid[y, x + separation]
        d_gt = chm[y[pair_valid], x[pair_valid]] - chm[y[pair_valid], x[pair_valid] + separation]
        d_pred = aligned[y[pair_valid], x[pair_valid]] - aligned[y[pair_valid], x[pair_valid] + separation]
        if len(d_gt) > 10000:
            keep = rng.choice(len(d_gt), 10000, replace=False); d_gt, d_pred = d_gt[keep], d_pred[keep]
        fig, ax = plt.subplots(figsize=(8, 7)); ax.scatter(d_gt, d_pred, s=3, alpha=0.2); lo, hi = float(min(np.min(d_gt), np.min(d_pred))), float(max(np.max(d_gt), np.max(d_pred))); ax.plot([lo, hi], [lo, hi], "k--", label="ideal y=x"); ax.set_title("Figure 7: GT local height difference vs DA2 difference (32 px)"); ax.set_xlabel("Δh_GT (m)"); ax.set_ylabel("Δh_DA2 aligned (m)"); ax.legend(); ax.grid(alpha=0.25); fig.tight_layout(); fig.savefig(report / "figure_7_local_height_difference.png", dpi=170); plt.close(fig)

        magnitude = read_rows(ROOT / args.results_dir / "airmeasurer_local_height_difference_by_magnitude.csv")
        if magnitude:
            fig, axes = plt.subplots(1, 2, figsize=(13, 5))
            labels = [f"{float(item['target_gt_difference_m']):.1f} m" for item in magnitude]
            axes[0].bar(labels, [float(item["mae_m"]) for item in magnitude], color="tab:purple")
            axes[0].set_title("Figure 10: Local height-difference error by GT magnitude")
            axes[0].set_xlabel("GT local height difference target")
            axes[0].set_ylabel("Local Difference MAE (m)")
            axes[1].bar(labels, [float(item["sign_agreement"]) for item in magnitude], color="tab:green")
            axes[1].set_title("Sign agreement by GT magnitude")
            axes[1].set_xlabel("GT local height difference target")
            axes[1].set_ylabel("Sign agreement (fraction)")
            axes[1].set_ylim(0, 1)
            for axis in axes: axis.grid(axis="y", alpha=0.25)
            fig.tight_layout(); fig.savefig(report / "figure_10_local_height_difference_magnitude.png", dpi=170); plt.close(fig)

        # Figure 8: only one downloaded date; do not invent a growth-stage comparison.
        fig, ax = plt.subplots(figsize=(10, 5)); ax.axis("off"); ax.text(0.02, 0.70, "Cross-date comparison: NOT AVAILABLE", fontsize=20, weight="bold"); ax.text(0.02, 0.48, "Only 20220505 was downloaded from the small-field release.\nThe full release lists additional dates, but they were not downloaded because the archive is multi-GB.", fontsize=15, va="top"); fig.suptitle("Figure 8: Early / middle / late comparison"); fig.tight_layout(); fig.savefig(report / "figure_8_cross_date_unavailable.png", dpi=170); plt.close(fig)

        # Figure 9: highest-error spatial block as an explicit failure case.
        error = np.where(valid, np.abs(aligned - chm), np.nan)
        block = 512
        best = (-1.0, 0, 0)
        for y0 in range(0, rgb.shape[0] - block + 1, block):
            for x0 in range(0, rgb.shape[1] - block + 1, block):
                value = np.nanmean(error[y0:y0 + block, x0:x0 + block])
                if np.isfinite(value) and value > best[0]: best = (value, y0, x0)
        _, y0, x0 = best
        y1, x1 = y0 + block, x0 + block
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(rgb[y0:y1, x0:x1]); axes[0].set_title("RGB")
        axes[1].imshow(chm[y0:y1, x0:x1], cmap="viridis"); axes[1].set_title("CHM (m)")
        axes[2].imshow(relative[y0:y1, x0:x1], cmap="turbo"); axes[2].set_title("DA2 relative")
        plot = axes[3].imshow(error[y0:y1, x0:x1], cmap="magma"); axes[3].set_title("|aligned error| (m)"); fig.colorbar(plot, ax=axes[3], fraction=0.046)
        for axis in axes: axis.set_xlabel("x"); axis.set_ylabel("y")
        fig.suptitle(f"Figure 9: representative high-error block (mean error={best[0]:.3f} m)")
        fig.tight_layout(); failure_dir = ROOT / "outputs/airmeasurer/failure_analysis"; failure_dir.mkdir(parents=True, exist_ok=True); fig.savefig(report / "figure_9_failure_case.png", dpi=170); fig.savefig(failure_dir / f"{date}_high_error_block.png", dpi=170); plt.close(fig)
        print(f"[airmeasurer] wrote report figures for {date}; failure block={best}")


if __name__ == "__main__":
    main()
