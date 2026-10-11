from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_kitti import find_ground_truth, load_kitti_gt


METRICS = ("mae_m", "rmse_m", "absrel", "delta_1_25")
MODEL_COLORS = {"outdoor": "#4472c4", "indoor": "#ed7d31"}


def read_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_depth_plot(outdoor: dict, indoor: dict, output_path: Path) -> None:
    depth_ranges = [row["depth_range_m"] for row in outdoor["by_depth"]]
    if depth_ranges != [row["depth_range_m"] for row in indoor["by_depth"]]:
        raise ValueError("The two models have different depth bins")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    metric_titles = {
        "mae_m": "MAE by GT depth band (m)",
        "rmse_m": "RMSE by GT depth band (m)",
        "absrel": "AbsRel by GT depth band",
        "delta_1_25": "δ < 1.25 by GT depth band",
    }
    x = np.arange(len(depth_ranges))
    width = 0.34
    for axis, metric in zip(axes.flat, METRICS):
        for offset, label, results in ((-width / 2, "outdoor", outdoor), (width / 2, "indoor", indoor)):
            values = [float(row[metric]) for row in results["by_depth"]]
            bars = axis.bar(x + offset, values, width, color=MODEL_COLORS[label], label=label)
            axis.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
        axis.set_title(metric_titles[metric])
        axis.set_xticks(x, depth_ranges)
        axis.set_xlabel("GT depth band (m)")
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
    axes[0, 0].set_ylabel("Error (m)")
    axes[0, 1].set_ylabel("Error (m)")
    axes[1, 0].set_ylabel("Relative error")
    axes[1, 1].set_ylabel("Fraction of valid pixels")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("KITTI 50-image comparison, GT depth ≤ 20 m", fontsize=14, y=1.02)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_mae_by_depth_plot(outdoor: dict, indoor: dict, output_path: Path) -> None:
    outdoor_rows = outdoor["by_depth"]
    indoor_rows = indoor["by_depth"]
    if [row["depth_range_m"] for row in outdoor_rows] != [row["depth_range_m"] for row in indoor_rows]:
        raise ValueError("The two models have different depth-bin order")

    depth_ranges = [row["depth_range_m"] for row in outdoor_rows]
    valid_pixels = [int(row["valid_pixels"]) for row in outdoor_rows]
    outdoor_mae = [float(row["mae_m"]) for row in outdoor_rows]
    indoor_mae = [float(row["mae_m"]) for row in indoor_rows]

    fig, axis = plt.subplots(figsize=(9.2, 5.8), facecolor="white")
    fig.subplots_adjust(left=0.11, right=0.98, top=0.75, bottom=0.21)
    x = np.arange(len(depth_ranges))
    width = 0.32
    bars_outdoor = axis.bar(x - width / 2, outdoor_mae, width, color=MODEL_COLORS["outdoor"], label="Outdoor model", zorder=3)
    bars_indoor = axis.bar(x + width / 2, indoor_mae, width, color=MODEL_COLORS["indoor"], label="Indoor model", zorder=3)
    axis.bar_label(bars_outdoor, labels=[f"{value:.3f} m" for value in outdoor_mae], padding=4, fontsize=10, fontweight="bold")
    axis.bar_label(bars_indoor, labels=[f"{value:.3f} m" for value in indoor_mae], padding=4, fontsize=10, fontweight="bold")
    axis.set_xticks(x, [f"{band} m\n(n={count:,} pixels)" for band, count in zip(depth_ranges, valid_pixels)])
    axis.set_ylabel("Mean Absolute Error (MAE) [m]")
    axis.set_ylim(0, max(indoor_mae) * 1.15)
    axis.grid(axis="y", color="#d9dee8", linewidth=0.8, zorder=0)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#9aa3b2")
    axis.tick_params(axis="both", colors="#303846")
    fig.suptitle("KITTI: MAE by Ground-Truth Depth Band (≤20 m)", y=0.98, fontsize=15, fontweight="bold")
    fig.text(0.5, 0.93, "Same 50 KITTI images and identical valid-pixel masks for both models", ha="center", color="#596273", fontsize=10)
    fig.legend(*axis.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.875), ncol=2, frameon=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, facecolor="white")
    plt.close(fig)


def colorize(values: np.ndarray, cmap_name: str, vmax: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    normalized = np.clip(values / max(vmax, 1e-6), 0.0, 1.0)
    rgb = (plt.get_cmap(cmap_name)(normalized)[..., :3] * 255).astype(np.uint8)
    rgb[~np.isfinite(values)] = 0
    return rgb


def make_same_frame_example(
    image_path: Path,
    gt_path: Path,
    outdoor_path: Path,
    indoor_path: Path,
    output_path: Path,
) -> None:
    rgb = np.asarray(Image.open(image_path).convert("RGB"))
    gt = load_kitti_gt(gt_path)
    outdoor = np.load(outdoor_path)
    indoor = np.load(indoor_path)
    if outdoor.shape != gt.shape:
        outdoor = cv2.resize(outdoor, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
    if indoor.shape != gt.shape:
        indoor = cv2.resize(indoor, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)

    valid = np.isfinite(gt) & (gt > 0) & (gt <= 20)
    valid &= np.isfinite(outdoor) & (outdoor > 0) & np.isfinite(indoor) & (indoor > 0)
    gt_view = np.where(valid, gt, np.nan)
    outdoor_view = np.where(valid, outdoor, np.nan)
    indoor_view = np.where(valid, indoor, np.nan)
    outdoor_error = np.where(valid, np.abs(outdoor - gt), np.nan)
    indoor_error = np.where(valid, np.abs(indoor - gt), np.nan)
    combined_error = np.concatenate((outdoor_error[np.isfinite(outdoor_error)], indoor_error[np.isfinite(indoor_error)]))
    error_max = float(np.percentile(combined_error, 99)) if len(combined_error) else 1.0

    panels = [
        (rgb, "RGB"),
        (colorize(gt_view, "turbo", 20), "GT depth (m)"),
        (colorize(outdoor_view, "turbo", 20), "Outdoor prediction (m)"),
        (colorize(indoor_view, "turbo", 20), "Indoor prediction (m)"),
        (colorize(outdoor_error, "magma", error_max), "Outdoor absolute error (m)"),
        (colorize(indoor_error, "magma", error_max), "Indoor absolute error (m)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(17, 5), constrained_layout=True)
    for axis, (panel, title) in zip(axes.flat, panels):
        axis.imshow(panel)
        axis.set_title(title)
        axis.set_axis_off()
    fig.suptitle(f"Matched KITTI frame, GT depth ≤ 20 m: {image_path.name}", fontsize=13)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare matched indoor/outdoor KITTI metric-depth evaluations")
    parser.add_argument("--out-root", default="outputs/kitti_20m")
    parser.add_argument("--data-root", default="data/kitti_validation_50")
    parser.add_argument("--outdir", default="outputs/kitti_20m/comparison")
    parser.add_argument("--results-outdir", default="results/metrics/kitti_20m/comparison")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    outdir = Path(args.outdir)
    results_outdir = Path(args.results_outdir)
    outdoor = read_metrics(out_root / "outdoor" / "metrics.json")
    indoor = read_metrics(out_root / "indoor" / "metrics.json")

    outdoor_images = outdoor["per_image"]
    indoor_images = indoor["per_image"]
    image_dir = Path(args.data_root) / "val_selection_cropped" / "image"
    gt_dir = Path(args.data_root) / "val_selection_cropped" / "groundtruth_depth"
    if [row["image"] for row in outdoor_images] != [row["image"] for row in indoor_images]:
        raise ValueError("The two models were not evaluated on the same image sequence")
    if any(int(a["valid_pixels"]) != int(b["valid_pixels"]) for a, b in zip(outdoor_images, indoor_images)):
        raise ValueError("Per-image valid-pixel counts differ; a matched-pixel comparison is not possible")

    bias_values = {"outdoor": [], "indoor": []}
    ratio_values = {"outdoor": [], "indoor": []}
    for outdoor_row, indoor_row in zip(outdoor_images, indoor_images):
        image_path = image_dir / outdoor_row["image"]
        gt = load_kitti_gt(find_ground_truth(gt_dir, image_path))
        outdoor_pred = np.load(Path(args.out_root) / "outdoor" / "predictions" / f"{image_path.stem}_metric_depth_m.npy")
        indoor_pred = np.load(Path(args.out_root) / "indoor" / "predictions" / f"{image_path.stem}_metric_depth_m.npy")
        if outdoor_pred.shape != gt.shape:
            outdoor_pred = cv2.resize(outdoor_pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
        if indoor_pred.shape != gt.shape:
            indoor_pred = cv2.resize(indoor_pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
        base_valid = np.isfinite(gt) & (gt > 0) & (gt <= 20)
        outdoor_valid = base_valid & np.isfinite(outdoor_pred) & (outdoor_pred > 0)
        indoor_valid = base_valid & np.isfinite(indoor_pred) & (indoor_pred > 0)
        if not np.array_equal(outdoor_valid, indoor_valid):
            raise ValueError(f"The two models have different valid-pixel masks for {image_path.name}")
        bias_values["outdoor"].append(outdoor_pred[outdoor_valid] - gt[outdoor_valid])
        bias_values["indoor"].append(indoor_pred[indoor_valid] - gt[indoor_valid])
        ratio_values["outdoor"].append(outdoor_pred[outdoor_valid] / gt[outdoor_valid])
        ratio_values["indoor"].append(indoor_pred[indoor_valid] / gt[indoor_valid])

    summary = []
    for metric in METRICS:
        outdoor_value = float(outdoor["summary"][metric])
        indoor_value = float(indoor["summary"][metric])
        summary.append({
            "metric": metric,
            "outdoor": outdoor_value,
            "indoor": indoor_value,
            "indoor_minus_outdoor": indoor_value - outdoor_value,
            "indoor_over_outdoor": indoor_value / outdoor_value if outdoor_value else float("nan"),
        })
    write_csv(results_outdir / "summary_comparison.csv", summary, list(summary[0]))

    depth_rows = []
    for outdoor_row, indoor_row in zip(outdoor["by_depth"], indoor["by_depth"]):
        if outdoor_row["depth_range_m"] != indoor_row["depth_range_m"]:
            raise ValueError("The two models have different depth-bin order")
        row = {
            "depth_range_m": outdoor_row["depth_range_m"],
            "valid_pixels": int(outdoor_row["valid_pixels"]),
        }
        for metric in METRICS:
            o_value = float(outdoor_row[metric])
            i_value = float(indoor_row[metric])
            row[f"outdoor_{metric}"] = o_value
            row[f"indoor_{metric}"] = i_value
            row[f"indoor_minus_outdoor_{metric}"] = i_value - o_value
        depth_rows.append(row)
    write_csv(results_outdir / "depth_comparison.csv", depth_rows, list(depth_rows[0]))

    per_image_rows = []
    for outdoor_row, indoor_row in zip(outdoor_images, indoor_images):
        row = {"image": outdoor_row["image"], "valid_pixels": int(outdoor_row["valid_pixels"])}
        for metric in METRICS:
            o_value = float(outdoor_row[metric])
            i_value = float(indoor_row[metric])
            row[f"outdoor_{metric}"] = o_value
            row[f"indoor_{metric}"] = i_value
            row[f"indoor_minus_outdoor_{metric}"] = i_value - o_value
        per_image_rows.append(row)
    write_csv(results_outdir / "per_image_comparison.csv", per_image_rows, list(per_image_rows[0]))
    make_depth_plot(outdoor, indoor, outdir / "metrics_by_depth.png")
    make_mae_by_depth_plot(outdoor, indoor, outdir / "mae_by_depth_band.png")

    diagnostics = []
    for model in ("outdoor", "indoor"):
        biases = np.concatenate(bias_values[model])
        ratios = np.concatenate(ratio_values[model])
        diagnostics.append({
            "model": model,
            "mean_bias_m": float(np.mean(biases)),
            "median_predicted_to_gt_ratio": float(np.median(ratios)),
        })
    write_csv(results_outdir / "scale_diagnostics.csv", diagnostics, list(diagnostics[0]))

    median_image = outdoor["representatives"]["median"]
    example_image_path = image_dir / median_image
    make_same_frame_example(
        example_image_path,
        find_ground_truth(gt_dir, example_image_path),
        Path(args.out_root) / "outdoor" / "predictions" / f"{example_image_path.stem}_metric_depth_m.npy",
        Path(args.out_root) / "indoor" / "predictions" / f"{example_image_path.stem}_metric_depth_m.npy",
        outdir / "same_frame_median_example.png",
    )

    comparison = {
        "condition": "same 50 KITTI images; valid GT and prediction pixels; GT depth <= 20 m",
        "models": {
            "outdoor": outdoor["summary"],
            "indoor": indoor["summary"],
        },
        "summary_comparison": summary,
        "by_depth": depth_rows,
        "same_images": True,
        "same_valid_pixels_per_image": True,
        "scale_diagnostics": diagnostics,
    }
    (outdir / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[compare] wrote matched comparison for {len(per_image_rows)} images to {outdir}")


if __name__ == "__main__":
    main()
