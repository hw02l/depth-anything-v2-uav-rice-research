from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from da2_experiments.io import write_csv, write_json  # noqa: E402


SEPARATIONS = (8, 16, 32, 64, 128)
VARIANTS = ("metric_raw", "metric_bias_corrected", "metric_scale_shift", "relative_scale_shift")
THRESHOLDS = (0.02, 0.05, 0.10, 0.20, 0.50)
MAGNITUDE_BINS = ((0.0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.50), (0.50, 1.0), (1.0, float("inf")))
TARGETS = (0.10, 0.20, 0.50, 1.00)


def sample_dir(row: dict, output_root: Path) -> Path:
    return output_root / row["scene"].replace("/", "_") / row["view_type"] / Path(row["filename"]).stem


def fit_scale_shift(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    indices = np.flatnonzero(valid)
    if len(indices) > 200_000:
        indices = rng.choice(indices, 200_000, replace=False)
    x = pred.flat[indices].astype(np.float64)
    y = gt.flat[indices].astype(np.float64)
    scale, shift = np.linalg.lstsq(np.column_stack((x, np.ones_like(x))), y, rcond=None)[0]
    return float(scale), float(shift)


def unique_pair_coordinates(height: int, width: int, separation: int, count: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample horizontal and vertical pixel pairs without replacement."""
    horizontal_width = width - separation
    vertical_height = height - separation
    total_horizontal = max(0, height * horizontal_width)
    total_vertical = max(0, vertical_height * width)
    horizontal_count = min(count // 2, total_horizontal)
    vertical_count = min(count - horizontal_count, total_vertical)
    if horizontal_count + vertical_count < count:
        remaining = count - horizontal_count - vertical_count
        extra_horizontal = min(remaining, total_horizontal - horizontal_count)
        horizontal_count += extra_horizontal
        remaining -= extra_horizontal
        vertical_count += min(remaining, total_vertical - vertical_count)

    h_indices = rng.choice(total_horizontal, size=horizontal_count, replace=False) if horizontal_count else np.empty(0, dtype=np.int64)
    v_indices = rng.choice(total_vertical, size=vertical_count, replace=False) if vertical_count else np.empty(0, dtype=np.int64)
    y_h = h_indices // horizontal_width if horizontal_count else np.empty(0, dtype=np.int64)
    x_h = h_indices % horizontal_width if horizontal_count else np.empty(0, dtype=np.int64)
    y_v = v_indices // width if vertical_count else np.empty(0, dtype=np.int64)
    x_v = v_indices % width if vertical_count else np.empty(0, dtype=np.int64)
    y1 = np.concatenate((y_h, y_v))
    x1 = np.concatenate((x_h, x_v))
    y2 = np.concatenate((y_h, y_v + separation))
    x2 = np.concatenate((x_h + separation, x_v))
    return y1, x1, y2, x2


def sample_difference_pair(gt: np.ndarray, predictions: dict[str, np.ndarray], valid: np.ndarray, separation: int, samples: int, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Use one shared, duplicate-free pair support for every variant."""
    requested = min(samples * 8, max(1, (gt.shape[0] * max(0, gt.shape[1] - separation)) + (max(0, gt.shape[0] - separation) * gt.shape[1])))
    y1, x1, y2, x2 = unique_pair_coordinates(*gt.shape, separation, requested, rng)
    pair_valid = valid[y1, x1] & valid[y2, x2]
    keep = np.flatnonzero(pair_valid)[:samples]
    y1, x1, y2, x2 = y1[keep], x1[keep], y2[keep], x2[keep]
    d_gt = gt[y1, x1].astype(np.float64) - gt[y2, x2].astype(np.float64)
    d_predictions = {
        name: prediction[y1, x1].astype(np.float64) - prediction[y2, x2].astype(np.float64)
        for name, prediction in predictions.items()
    }
    return d_gt, d_predictions


def rank_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks_sorted = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks_sorted[start:end] = (start + end - 1) / 2.0 + 1.0
        start = end
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = ranks_sorted
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    return pearson(rank_average(x), rank_average(y))


def difference_metrics(d_gt: np.ndarray, d_pred: np.ndarray) -> dict[str, float | int]:
    error = d_pred - d_gt
    nonzero_gt = np.abs(d_gt) > 1e-9
    return {
        "pairs": int(len(d_gt)),
        "mean_abs_gt_difference_m": float(np.mean(np.abs(d_gt))) if len(d_gt) else float("nan"),
        "local_difference_mae_m": float(np.mean(np.abs(error))) if len(error) else float("nan"),
        "local_difference_rmse_m": float(np.sqrt(np.mean(error**2))) if len(error) else float("nan"),
        "local_difference_bias_m": float(np.mean(error)) if len(error) else float("nan"),
        "local_difference_pearson": pearson(d_gt, d_pred),
        "local_difference_spearman": spearman(d_gt, d_pred),
        "local_difference_sign_agreement": float(np.mean(np.sign(d_gt[nonzero_gt]) == np.sign(d_pred[nonzero_gt]))) if np.any(nonzero_gt) else float("nan"),
        "gt_difference_std_m": float(np.std(d_gt)) if len(d_gt) else float("nan"),
    }


def regression_metrics(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    if len(x) < 2 or np.std(x) == 0:
        return float("nan"), float("nan"), float("nan")
    slope, intercept = np.linalg.lstsq(np.column_stack((x, np.ones_like(x))), y, rcond=None)[0]
    fitted = slope * x + intercept
    sst = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - np.sum((y - fitted) ** 2) / sst) if sst > 0 else float("nan")
    return float(slope), float(intercept), r2


def aggregate_chunks(chunks: list[tuple[np.ndarray, dict[str, np.ndarray]]], variant: str) -> tuple[np.ndarray, np.ndarray]:
    return np.concatenate([item[0] for item in chunks]), np.concatenate([item[1][variant] for item in chunks])


def write_profiles(rows: list[dict], loaded_profiles: dict[str, tuple[dict, np.ndarray, np.ndarray, np.ndarray]], profile_dir: Path) -> None:
    profile_dir.mkdir(parents=True, exist_ok=True)
    for row, gt, pred, corrected in loaded_profiles.values():
        y = gt.shape[0] // 2
        stride = max(1, gt.shape[1] // 1600)
        x = np.arange(gt.shape[1])
        figure, axis = plt.subplots(figsize=(14, 5))
        axis.plot(x[::stride], gt[y, ::stride], label="UAV3DCrop GT z-depth", linewidth=1.2)
        axis.plot(x[::stride], pred[y, ::stride], label="DA2 metric depth", linewidth=1.0)
        axis.plot(x[::stride], corrected[y, ::stride], label="Bias-corrected DA2", linewidth=1.0)
        axis.set_title(f"Depth profile: {row['scene']} | {row['view_type']} | {row['filename']} | y={y}")
        axis.set_xlabel("Pixel x")
        axis.set_ylabel("Camera-frame z-depth (m)")
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(profile_dir / f"{row['scene'].replace('/', '_')}_{row['view_type']}_{Path(row['filename']).stem}_profile.png", dpi=170)
        plt.close(figure)


def write_histogram(gt_chunks: dict[int, list[np.ndarray]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 6))
    for separation, chunks in gt_chunks.items():
        values = np.concatenate(chunks)
        axis.hist(values, bins=100, alpha=0.32, density=True, label=f"{separation} px")
    axis.set_title("UAV3DCrop GT local depth-difference magnitude")
    axis.set_xlabel("|Δd_GT| (m)")
    axis.set_ylabel("Density")
    axis.set_yscale("log")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def write_report_figures(result_dir: Path, report_dir: Path, gt_chunks: dict[int, list[np.ndarray]], magnitude_rows: list[dict], baseline_rows: list[dict], threshold_rows: list[dict], summary_rows: list[dict], target_data: dict[float, dict[str, tuple[np.ndarray, np.ndarray]]], view_rows: list[dict], scene_rows: list[dict], crop_rows: list[dict]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    # Figure 1: GT difference distribution.
    write_histogram(gt_chunks, report_dir / "figure_1_gt_difference_distribution.png")

    # Figure 2: magnitude versus local error.
    figure, axis = plt.subplots(figsize=(10, 6))
    for variant in ("metric_raw", "metric_scale_shift", "relative_scale_shift"):
        subset = [row for row in magnitude_rows if row["variant"] == variant and np.isfinite(float(row["mean_abs_gt_difference_m"]))]
        grouped = {}
        for row in subset:
            grouped.setdefault(row["gt_magnitude_bin"], []).append(row)
        xs = [np.mean([float(r["mean_abs_gt_difference_m"]) for r in rs]) for rs in grouped.values()]
        ys = [np.mean([float(r["local_difference_mae_m"]) for r in rs]) for rs in grouped.values()]
        axis.plot(xs, ys, marker="o", label=variant)
    axis.set_title("GT difference magnitude vs local-difference MAE")
    axis.set_xlabel("Mean |Δd_GT| (m)")
    axis.set_ylabel("Local Difference MAE (m)")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(report_dir / "figure_2_gt_magnitude_vs_local_mae.png", dpi=180)
    plt.close(figure)

    # Figure 3: target magnitude scatter, one panel per target.
    figure, axes = plt.subplots(2, 2, figsize=(13, 11))
    colors = {"metric_raw": "tab:blue", "metric_scale_shift": "tab:orange", "relative_scale_shift": "tab:green"}
    for axis, target in zip(axes.flat, TARGETS):
        data = target_data.get(target, {})
        all_values = []
        for variant, (x, y) in data.items():
            if len(x) == 0:
                continue
            keep = np.linspace(0, len(x) - 1, min(4000, len(x)), dtype=int)
            axis.scatter(x[keep], y[keep], s=3, alpha=0.18, color=colors.get(variant), label=variant)
            all_values.extend([float(np.min(x)), float(np.max(x)), float(np.min(y)), float(np.max(y))])
        if all_values:
            lo, hi = min(all_values), max(all_values)
            axis.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="ideal y=x")
        axis.set_title(f"Target |Δd_GT| ≈ {target:.2f} m")
        axis.set_xlabel("Signed Δd_GT (m)")
        axis.set_ylabel("Signed Δd_pred (m)")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.suptitle("GT local depth difference vs predicted difference")
    figure.tight_layout()
    figure.savefig(report_dir / "figure_3_gt_vs_pred_difference_scatter.png", dpi=180)
    plt.close(figure)

    # Figure 4: zero baseline.
    figure, axis = plt.subplots(figsize=(10, 6))
    for variant in ("zero_baseline", "metric_raw", "metric_scale_shift", "relative_scale_shift"):
        subset = [r for r in baseline_rows if r["variant"] == variant]
        subset.sort(key=lambda r: int(r["pixel_separation"]))
        axis.plot([int(r["pixel_separation"]) for r in subset], [float(r["local_difference_mae_m"]) for r in subset], marker="o", label=variant)
    axis.set_title("Depth-difference MAE versus zero baseline")
    axis.set_xlabel("Pixel separation (px)")
    axis.set_ylabel("Local Difference MAE (m)")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(report_dir / "figure_4_zero_baseline_comparison.png", dpi=180)
    plt.close(figure)

    # Figure 5: sign accuracy by threshold at 32 px.
    figure, axis = plt.subplots(figsize=(10, 6))
    for variant in ("zero_baseline", "metric_raw", "metric_scale_shift", "relative_scale_shift"):
        subset = [r for r in threshold_rows if r["variant"] == variant and int(r["pixel_separation"]) == 32]
        subset.sort(key=lambda r: float(r["threshold_m"]))
        axis.plot([float(r["threshold_m"]) for r in subset], [float(r["sign_agreement"]) for r in subset], marker="o", label=variant)
    axis.set_title("Sign agreement versus GT difference threshold (32 px)")
    axis.set_xlabel("Minimum |Δd_GT| (m)")
    axis.set_ylabel("Sign agreement")
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(report_dir / "figure_5_threshold_sign_agreement.png", dpi=180)
    plt.close(figure)

    # Figure 6: separation curve.
    figure, axis = plt.subplots(figsize=(10, 6))
    for variant in ("metric_raw", "metric_scale_shift", "relative_scale_shift"):
        subset = [r for r in summary_rows if r["variant"] == variant and r["metric"] == "local_difference_mae_m"]
        subset.sort(key=lambda r: int(r["pixel_separation"]))
        axis.plot([int(r["pixel_separation"]) for r in subset], [float(r["mean"]) for r in subset], marker="o", label=variant)
    axis.set_title("Pixel separation versus local-difference error")
    axis.set_xlabel("Pixel separation (px)")
    axis.set_ylabel("Mean Local Difference MAE (m)")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(report_dir / "figure_6_pixel_separation_vs_error.png", dpi=180)
    plt.close(figure)

    def grouped_bar(rows: list[dict], filename: str, title: str, group_key: str, metric_key: str) -> None:
        subset = [r for r in rows if int(r["pixel_separation"]) == 32]
        groups = sorted({r[group_key] for r in subset})
        variants = ["metric_raw", "metric_scale_shift", "relative_scale_shift"]
        figure, axis = plt.subplots(figsize=(11, 6))
        x = np.arange(len(groups))
        width = 0.25
        for offset, variant in enumerate(variants):
            values = [float(next(r[metric_key] for r in subset if r[group_key] == group and r["variant"] == variant)) for group in groups]
            axis.bar(x + (offset - 1) * width, values, width, label=variant)
        axis.set_xticks(x, groups, rotation=25, ha="right")
        axis.set_title(title)
        axis.set_xlabel(group_key)
        axis.set_ylabel("Pearson correlation" if metric_key == "local_difference_pearson" else "Local Difference MAE (m)")
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(report_dir / filename, dpi=180)
        plt.close(figure)

    grouped_bar(view_rows, "figure_7_nadir_oblique_local_geometry.png", "Nadir versus oblique local geometry (32 px)", "view_type", "local_difference_pearson")
    grouped_bar(scene_rows, "figure_8_scene_local_geometry.png", "Scene-wise local geometry (32 px)", "scene", "local_difference_pearson")
    grouped_bar(crop_rows, "figure_9_crop_local_geometry.png", "Crop-wise local geometry (32 px)", "crop", "local_difference_pearson")

    target_rows = [r for r in magnitude_rows if r["variant"] == "relative_scale_shift" and r["gt_magnitude_bin"] == "0.10-0.20 m"]
    figure, axis = plt.subplots(figsize=(10, 6))
    if target_rows:
        axis.bar([r["pixel_separation"] for r in target_rows], [float(r["local_difference_mae_m"]) for r in target_rows])
    axis.set_title("Local error for GT differences around 0.2 m")
    axis.set_xlabel("Pixel separation (px)")
    axis.set_ylabel("Local Difference MAE (m)")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(report_dir / "figure_10_around_0p2m_difference.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate local depth differences by GT magnitude and geometry baselines")
    parser.add_argument("--selection-csv", default="results/uav3dcrop/selected_images.csv")
    parser.add_argument("--output-root", default="outputs/uav3dcrop")
    parser.add_argument("--results-dir", default="results/metrics")
    parser.add_argument("--analysis-dir", default="outputs/uav3dcrop/local_analysis")
    parser.add_argument("--profile-dir", default="outputs/uav3dcrop/depth_profiles")
    parser.add_argument("--pair-samples", type=int, default=20000)
    args = parser.parse_args()

    with (ROOT / args.selection_csv).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    output_root = ROOT / args.output_root
    rng = np.random.default_rng(20260920)
    corrected_rows: list[dict] = []
    local_rows: list[dict] = []
    samples_by_sep: dict[int, list[tuple[np.ndarray, dict[str, np.ndarray]]]] = {sep: [] for sep in SEPARATIONS}
    gt_chunks: dict[int, list[np.ndarray]] = {sep: [] for sep in SEPARATIONS}
    group_chunks: dict[tuple[str, str, int], list[tuple[np.ndarray, dict[str, np.ndarray]]]] = defaultdict(list)
    profile_data: dict[str, tuple[dict, np.ndarray, np.ndarray, np.ndarray]] = {}

    for row in rows:
        directory = sample_dir(row, output_root)
        stem = Path(row["filename"]).stem
        pred = np.load(directory / f"{stem}_metric_depth_m.npy").astype(np.float32)
        relative = np.load(directory / f"{stem}_relative_depth.npy").astype(np.float32)
        gt = tifffile.imread(ROOT / row["local_depth_path"]).astype(np.float32)
        if pred.shape != gt.shape or relative.shape != gt.shape:
            raise ValueError(f"shape mismatch in {row['filename']}: {pred.shape}, {relative.shape}, {gt.shape}")
        valid = np.isfinite(pred) & (pred > 0) & np.isfinite(relative) & np.isfinite(gt) & (gt > 0)
        raw_error = pred[valid].astype(np.float64) - gt[valid].astype(np.float64)
        bias = float(np.median(raw_error))
        corrected = pred - bias
        corrected_valid = valid & np.isfinite(corrected)
        scale, shift = fit_scale_shift(pred, gt, valid, rng)
        aligned = scale * pred + shift
        relative_like = -relative
        relative_scale, relative_shift = fit_scale_shift(relative_like, gt, valid, rng)
        relative_aligned = relative_scale * relative_like + relative_shift
        corrected_rows.append({
            **row,
            "valid_pixels": int(valid.sum()),
            "raw_mae_m": float(np.mean(np.abs(raw_error))),
            "raw_rmse_m": float(np.sqrt(np.mean(raw_error**2))),
            "raw_absrel": float(np.mean(np.abs(raw_error) / gt[valid])),
            "global_bias_m": bias,
            "corrected_mae_m": float(np.mean(np.abs(corrected[corrected_valid] - gt[corrected_valid]))),
            "corrected_rmse_m": float(np.sqrt(np.mean((corrected[corrected_valid] - gt[corrected_valid]) ** 2))),
            "scale_shift_scale": scale,
            "scale_shift_shift_m": shift,
            "scale_shift_mae_m": float(np.mean(np.abs(aligned[valid] - gt[valid]))),
            "scale_shift_rmse_m": float(np.sqrt(np.mean((aligned[valid] - gt[valid]) ** 2))),
            "scale_shift_absrel": float(np.mean(np.abs(aligned[valid] - gt[valid]) / gt[valid])),
            "relative_orientation": "inverse_depth; negated before z-depth alignment",
            "relative_scale_shift_scale": relative_scale,
            "relative_scale_shift_shift_m": relative_shift,
            "relative_aligned_mae_m": float(np.mean(np.abs(relative_aligned[valid] - gt[valid]))),
            "relative_aligned_rmse_m": float(np.sqrt(np.mean((relative_aligned[valid] - gt[valid]) ** 2))),
            "relative_aligned_absrel": float(np.mean(np.abs(relative_aligned[valid] - gt[valid]) / gt[valid])),
        })
        predictions = {
            "metric_raw": pred,
            "metric_bias_corrected": corrected,
            "metric_scale_shift": aligned,
            "relative_scale_shift": relative_aligned,
        }
        if row["view_type"] not in {item[0]["view_type"] for item in profile_data.values()}:
            profile_data[f"{row['view_type']}_{row['filename']}"] = (row, gt, pred, corrected)
        for separation in SEPARATIONS:
            d_gt, d_predictions = sample_difference_pair(gt, predictions, valid, separation, args.pair_samples, rng)
            samples_by_sep[separation].append((d_gt, d_predictions))
            gt_chunks[separation].append(np.abs(d_gt))
            group_chunks[(row["scene"], row["view_type"], separation)].append((d_gt, d_predictions))
            for variant in VARIANTS:
                local_rows.append({**row, "variant": variant, "pixel_separation": separation, **difference_metrics(d_gt, d_predictions[variant])})

    result_dir = ROOT / args.results_dir
    write_csv(result_dir / "uav3dcrop_bias_corrected.csv", corrected_rows, list(corrected_rows[0].keys()))
    write_csv(result_dir / "uav3dcrop_local_depth_difference.csv", local_rows, list(local_rows[0].keys()))

    summary_rows = []
    for variant in VARIANTS:
        for separation in SEPARATIONS:
            subset = [r for r in local_rows if r["variant"] == variant and int(r["pixel_separation"]) == separation]
            for metric_name in ("local_difference_mae_m", "local_difference_rmse_m", "local_difference_pearson", "local_difference_spearman", "local_difference_sign_agreement"):
                values = np.asarray([float(r[metric_name]) for r in subset], dtype=np.float64)
                values = values[np.isfinite(values)]
                summary_rows.append({"variant": variant, "pixel_separation": separation, "metric": metric_name, "images": len(subset), "mean": float(np.mean(values)) if len(values) else float("nan"), "median": float(np.median(values)) if len(values) else float("nan"), "std": float(np.std(values)) if len(values) else float("nan")})
    write_csv(result_dir / "uav3dcrop_local_depth_difference_summary.csv", summary_rows, list(summary_rows[0].keys()))

    distribution_rows = []
    for separation in SEPARATIONS:
        values = np.concatenate(gt_chunks[separation])
        distribution_rows.append({
            "pixel_separation": separation,
            "pairs": len(values),
            "mean_abs_gt_difference_m": float(np.mean(values)),
            "median_abs_gt_difference_m": float(np.median(values)),
            "std_abs_gt_difference_m": float(np.std(values)),
            "p05_abs_gt_difference_m": float(np.percentile(values, 5)),
            "p25_abs_gt_difference_m": float(np.percentile(values, 25)),
            "p75_abs_gt_difference_m": float(np.percentile(values, 75)),
            "p95_abs_gt_difference_m": float(np.percentile(values, 95)),
        })
    write_csv(result_dir / "local_gt_difference_distribution.csv", distribution_rows, list(distribution_rows[0].keys()))

    magnitude_rows = []
    for separation in SEPARATIONS:
        d_gt_all = np.concatenate([item[0] for item in samples_by_sep[separation]])
        for lower, upper in MAGNITUDE_BINS:
            label = f"{lower:.2f}+ m" if np.isinf(upper) else f"{lower:.2f}-{upper:.2f} m"
            mask = (np.abs(d_gt_all) >= lower) & ((np.abs(d_gt_all) <= upper) if np.isinf(upper) else (np.abs(d_gt_all) < upper))
            for variant in VARIANTS:
                d_pred = np.concatenate([item[1][variant] for item in samples_by_sep[separation]])[mask]
                d_gt = d_gt_all[mask]
                values = difference_metrics(d_gt, d_pred)
                magnitude_rows.append({"pixel_separation": separation, "gt_magnitude_lower_m": lower, "gt_magnitude_upper_m": upper, "gt_magnitude_bin": label, "variant": variant, **values})
    write_csv(result_dir / "local_difference_by_gt_magnitude.csv", magnitude_rows, list(magnitude_rows[0].keys()))

    baseline_rows = []
    for separation in SEPARATIONS:
        d_gt_all = np.concatenate([item[0] for item in samples_by_sep[separation]])
        variant_arrays = {variant: np.concatenate([item[1][variant] for item in samples_by_sep[separation]]) for variant in VARIANTS}
        variant_arrays["zero_baseline"] = np.zeros_like(d_gt_all)
        zero_mae = float(np.mean(np.abs(d_gt_all)))
        for variant, d_pred in variant_arrays.items():
            values = difference_metrics(d_gt_all, d_pred)
            baseline_rows.append({"pixel_separation": separation, "variant": variant, **values, "zero_baseline_mae_m": zero_mae, "improvement_vs_zero_mae_m": zero_mae - float(values["local_difference_mae_m"])})
    write_csv(result_dir / "local_difference_baseline_comparison.csv", baseline_rows, list(baseline_rows[0].keys()))

    threshold_rows = []
    for separation in SEPARATIONS:
        d_gt_all = np.concatenate([item[0] for item in samples_by_sep[separation]])
        variant_arrays = {variant: np.concatenate([item[1][variant] for item in samples_by_sep[separation]]) for variant in VARIANTS}
        variant_arrays["zero_baseline"] = np.zeros_like(d_gt_all)
        for threshold in THRESHOLDS:
            mask = np.abs(d_gt_all) >= threshold
            for variant, d_pred in variant_arrays.items():
                gt_selected, pred_selected = d_gt_all[mask], d_pred[mask]
                nonzero = np.abs(gt_selected) > 1e-9
                threshold_rows.append({"pixel_separation": separation, "variant": variant, "threshold_m": threshold, "pairs": int(mask.sum()), "sign_agreement": float(np.mean(np.sign(gt_selected[nonzero]) == np.sign(pred_selected[nonzero]))) if np.any(nonzero) else float("nan")})
    write_csv(result_dir / "local_difference_sign_thresholds.csv", threshold_rows, list(threshold_rows[0].keys()))

    ratio_rows = []
    for separation in SEPARATIONS:
        d_gt_all = np.concatenate([item[0] for item in samples_by_sep[separation]])
        mask = np.abs(d_gt_all) >= 0.05
        for variant in VARIANTS:
            d_pred = np.concatenate([item[1][variant] for item in samples_by_sep[separation]])
            ratio = d_pred[mask] / d_gt_all[mask]
            ratio_rows.append({"pixel_separation": separation, "variant": variant, "threshold_m": 0.05, "pairs": len(ratio), "ratio_mean": float(np.mean(ratio)), "ratio_median": float(np.median(ratio)), "ratio_p25": float(np.percentile(ratio, 25)), "ratio_p75": float(np.percentile(ratio, 75))})
    write_csv(result_dir / "local_difference_ratio.csv", ratio_rows, list(ratio_rows[0].keys()))

    target_data: dict[float, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    target_rows = []
    for target in TARGETS:
        target_data[target] = {}
        all_gt = np.concatenate([item[0] for chunks in samples_by_sep.values() for item in chunks])
        for variant in VARIANTS:
            all_pred = np.concatenate([item[1][variant] for chunks in samples_by_sep.values() for item in chunks])
            mask = (np.abs(all_gt) >= target * 0.8) & (np.abs(all_gt) <= target * 1.2)
            x, y = all_gt[mask], all_pred[mask]
            target_data[target][variant] = (x, y)
            slope, intercept, r2 = regression_metrics(x, y)
            values = difference_metrics(x, y)
            target_rows.append({"target_abs_gt_difference_m": target, "range_lower_m": target * 0.8, "range_upper_m": target * 1.2, "variant": variant, "pairs": len(x), "slope": slope, "intercept_m": intercept, "r2": r2, **values})
    write_csv(result_dir / "local_difference_target_magnitude.csv", target_rows, list(target_rows[0].keys()))

    def aggregate_group(group_chunks_local: dict[tuple[str, str, int], list[tuple[np.ndarray, dict[str, np.ndarray]]]], key_index: int) -> list[dict]:
        output = []
        groups = sorted({key[key_index] for key in group_chunks_local})
        for group in groups:
            for separation in SEPARATIONS:
                chunks = [item for key, item in group_chunks_local.items() if key[key_index] == group and key[2] == separation]
                if not chunks:
                    continue
                for variant in VARIANTS:
                    d_gt = np.concatenate([pair[0] for chunk in chunks for pair in chunk])
                    d_pred = np.concatenate([pair[1][variant] for chunk in chunks for pair in chunk])
                    output.append({"group": group, "pixel_separation": separation, "variant": variant, **difference_metrics(d_gt, d_pred)})
        return output

    # Make one group chunk dictionary per scene/view and crop for reusable reports.
    view_rows = [{**r, "view_type": r["group"]} for r in aggregate_group(group_chunks, 1)]
    scene_rows = [{**r, "scene": r["group"]} for r in aggregate_group(group_chunks, 0)]
    crop_chunks: dict[tuple[str, str, int], list[tuple[np.ndarray, dict[str, np.ndarray]]]] = defaultdict(list)
    for key, chunks in group_chunks.items():
        crop = next((row["crop"] for row in rows if row["scene"] == key[0]), "unknown")
        crop_chunks[(crop, key[1], key[2])].extend(chunks)
    crop_rows = [{**r, "crop": r["group"]} for r in aggregate_group(crop_chunks, 0)]
    write_csv(result_dir / "local_difference_by_view.csv", view_rows, list(view_rows[0].keys()) if view_rows else [])
    write_csv(result_dir / "local_difference_by_scene.csv", scene_rows, list(scene_rows[0].keys()) if scene_rows else [])
    write_csv(result_dir / "local_difference_by_crop.csv", crop_rows, list(crop_rows[0].keys()) if crop_rows else [])

    write_histogram(gt_chunks, ROOT / args.analysis_dir / "gt_difference_histogram.png")
    write_report_figures(result_dir, ROOT / args.analysis_dir / "report_figures", gt_chunks, magnitude_rows, baseline_rows, threshold_rows, summary_rows, target_data, view_rows, scene_rows, crop_rows)
    write_profiles(rows, profile_data, ROOT / args.profile_dir)
    write_json(result_dir / "uav3dcrop_local_method.json", {
        "separations_px": list(SEPARATIONS),
        "pair_samples_per_image_variant_separation": args.pair_samples,
        "sampling": "horizontal and vertical candidate pairs sampled without replacement; one shared valid pair set is reused for all variants for each image and separation",
        "valid_mask": "finite GT, metric, relative; GT>0 and metric>0",
        "difference_definition": "Delta d = d(pixel 1) - d(pixel 2); positive means pixel 1 has larger camera-frame z-depth and is farther",
        "sign_agreement": "computed over nonzero GT differences; predicted zero counts as disagreement",
        "relative_convention": "official relative output treated as affine-invariant inverse depth; negated before scale-shift alignment to GT z-depth",
        "aligned_results": "diagnostic geometry scores only, not raw metric accuracy",
    })
    write_json(result_dir / "local_difference_method.json", {
        "gt_magnitude_bins_m": [[lower, upper if np.isfinite(upper) else None] for lower, upper in MAGNITUDE_BINS],
        "target_ranges_m": {str(target): [target * 0.8, target * 1.2] for target in TARGETS},
        "thresholds_m": list(THRESHOLDS),
        "ratio_threshold_m": 0.05,
        "sampling": "same duplicate-free pair support across variants",
    })
    print(f"[local] evaluated {len(rows)} images, wrote magnitude/baseline/threshold/ratio/scatter outputs")


if __name__ == "__main__":
    main()
