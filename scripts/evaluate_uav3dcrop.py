from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from da2_experiments.io import write_csv, write_json  # noqa: E402


METRICS = (
    "mae_m",
    "rmse_m",
    "absrel",
    "bias_m",
    "median_abs_error_m",
    "delta_1_25",
    "delta_1_25_2",
    "delta_1_25_3",
    "valid_pixels",
)


def sample_dir(row: dict, output_root: Path) -> Path:
    return output_root / row["scene"].replace("/", "_") / row["view_type"] / Path(row["filename"]).stem


def read_pair(row: dict, output_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, Path]:
    directory = sample_dir(row, output_root)
    prediction = np.load(directory / f"{Path(row['filename']).stem}_metric_depth_m.npy").astype(np.float32)
    ground_truth = tifffile.imread(ROOT / row["local_depth_path"]).astype(np.float32)
    rgb = np.asarray(Image.open(ROOT / row["local_rgb_path"]).convert("RGB"))
    if prediction.shape != ground_truth.shape or rgb.shape[:2] != ground_truth.shape:
        raise ValueError(f"RGB/GT/prediction shape mismatch for {row['filename']}: {rgb.shape}, {ground_truth.shape}, {prediction.shape}")
    valid = np.isfinite(prediction) & (prediction > 0) & np.isfinite(ground_truth) & (ground_truth > 0)
    return prediction, ground_truth, valid, directory


def metric_values(prediction: np.ndarray, ground_truth: np.ndarray, valid: np.ndarray) -> dict[str, float | int]:
    pred = prediction[valid].astype(np.float64)
    gt = ground_truth[valid].astype(np.float64)
    error = pred - gt
    ratio = np.maximum(pred / gt, gt / pred)
    return {
        "mae_m": float(np.mean(np.abs(error))),
        "rmse_m": float(np.sqrt(np.mean(error**2))),
        "absrel": float(np.mean(np.abs(error) / gt)),
        "bias_m": float(np.mean(error)),
        "median_abs_error_m": float(np.median(np.abs(error))),
        "delta_1_25": float(np.mean(ratio < 1.25)),
        "delta_1_25_2": float(np.mean(ratio < 1.25**2)),
        "delta_1_25_3": float(np.mean(ratio < 1.25**3)),
        "valid_pixels": int(valid.sum()),
    }


def row_stats(rows: list[dict], group_key: str | None = None) -> list[dict]:
    output = []
    groups = sorted({row[group_key] for row in rows}) if group_key else [None]
    for group in groups:
        subset = [row for row in rows if group is None or row[group_key] == group]
        for metric in METRICS:
            values = np.asarray([float(row[metric]) for row in subset], dtype=np.float64)
            output.append({
                **({group_key: group, "images": len(subset)} if group_key else {"images": len(subset)}),
                "metric": metric,
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            })
    return output


def pooled_metrics(records: list[dict]) -> dict[str, float | int]:
    total = sum(int(row["valid_pixels"]) for row in records)
    pooled = {"valid_pixels": total}
    if total == 0:
        return {metric: float("nan") for metric in METRICS} | pooled
    for metric in ("mae_m", "absrel", "delta_1_25", "delta_1_25_2", "delta_1_25_3"):
        pooled[metric] = float(sum(float(row[metric]) * int(row["valid_pixels"]) for row in records) / total)
    pooled["rmse_m"] = float(np.sqrt(sum(float(row["rmse_m"]) ** 2 * int(row["valid_pixels"]) for row in records) / total))
    pooled["bias_m"] = float(sum(float(row["bias_m"]) * int(row["valid_pixels"]) for row in records) / total)
    pooled["median_abs_error_m"] = float("nan")
    return pooled


def depth_edges(rows: list[dict], output_root: Path) -> tuple[np.ndarray, int]:
    samples = []
    for row in rows:
        _, gt, _, _ = read_pair(row, output_root)
        valid = np.isfinite(gt) & (gt > 0)
        values = gt[valid]
        step = max(1, len(values) // 50000)
        samples.append(values[::step])
    all_values = np.concatenate(samples)
    edges = np.unique(np.quantile(all_values, np.linspace(0.0, 1.0, 6)))
    if len(edges) < 2:
        edges = np.array([float(all_values.min()), float(all_values.max()) + 1e-6])
    return edges, len(all_values)


def depth_bin_rows(rows: list[dict], output_root: Path, edges: np.ndarray) -> list[dict]:
    result = []
    for index in range(len(edges) - 1):
        lower, upper = float(edges[index]), float(edges[index + 1])
        count = 0
        sum_abs = 0.0
        sum_sq = 0.0
        sum_absrel = 0.0
        sum_error = 0.0
        delta_counts = [0, 0, 0]
        for row in rows:
            pred, gt, valid, _ = read_pair(row, output_root)
            in_bin = valid & (gt >= lower) & ((gt <= upper) if index == len(edges) - 2 else (gt < upper))
            if np.any(in_bin):
                pred_bin = pred[in_bin].astype(np.float64)
                gt_bin = gt[in_bin].astype(np.float64)
                error = pred_bin - gt_bin
                ratio = np.maximum(pred_bin / gt_bin, gt_bin / pred_bin)
                count += int(gt_bin.size)
                sum_abs += float(np.abs(error).sum())
                sum_sq += float(np.square(error).sum())
                sum_absrel += float((np.abs(error) / gt_bin).sum())
                sum_error += float(error.sum())
                delta_counts[0] += int((ratio < 1.25).sum())
                delta_counts[1] += int((ratio < 1.25**2).sum())
                delta_counts[2] += int((ratio < 1.25**3).sum())
        if count == 0:
            continue
        result.append({
            "depth_bin_lower_m": lower,
            "depth_bin_upper_m": upper,
            "depth_range_m": f"{lower:.3f}-{upper:.3f}",
            "bin_method": "five empirical GT quantile bins",
            "valid_pixels": count,
            "mae_m": sum_abs / count,
            "rmse_m": float(np.sqrt(sum_sq / count)),
            "absrel": sum_absrel / count,
            "bias_m": sum_error / count,
            "delta_1_25": delta_counts[0] / count,
            "delta_1_25_2": delta_counts[1] / count,
            "delta_1_25_3": delta_counts[2] / count,
        })
    return result


def save_error_figure(row: dict, prediction: np.ndarray, ground_truth: np.ndarray, valid: np.ndarray, directory: Path, outdir: Path) -> None:
    stride = max(1, int(max(ground_truth.shape) / 900))
    rgb = np.asarray(Image.open(ROOT / row["local_rgb_path"]).convert("RGB"))[::stride, ::stride]
    pred = prediction[::stride, ::stride]
    gt = ground_truth[::stride, ::stride]
    valid_small = valid[::stride, ::stride]
    error = np.where(valid_small, pred - gt, np.nan)
    bias = float(np.median((prediction - ground_truth)[valid]))
    bias_error = np.where(valid_small, (pred - bias) - gt, np.nan)
    absolute = np.abs(error)
    figure, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("RGB")
    images = [
        (gt, "GT z-depth (m)", "viridis"),
        (pred, "DA2 metric depth (m)", "viridis"),
        (absolute, "Absolute error (m)", "magma"),
        (error, "Signed error (m)", "coolwarm"),
        (bias_error, "Bias-corrected signed error (m)", "coolwarm"),
    ]
    for axis, (image, title, cmap) in zip(axes.flat[1:], images):
        plot = axis.imshow(image, cmap=cmap)
        axis.set_title(title)
        axis.figure.colorbar(plot, ax=axis, fraction=0.046, pad=0.04)
    for axis in axes.flat:
        axis.axis("off")
    scene_key = row["scene"].replace("/", "_")
    destination = outdir / f"{scene_key}_{row['view_type']}_{Path(row['filename']).stem}_error_analysis.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.suptitle(f"{row['scene']} | {row['view_type']} | {row['filename']} | median bias={bias:.3f} m")
    figure.tight_layout()
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate UAV3DCrop GT z-depth against DA2 metric depth")
    parser.add_argument("--selection-csv", default="results/uav3dcrop/selected_images.csv")
    parser.add_argument("--output-root", default="outputs/uav3dcrop")
    parser.add_argument("--results-dir", default="results/metrics")
    parser.add_argument("--error-dir", default="outputs/uav3dcrop/error_analysis")
    parser.add_argument("--error-examples", type=int, default=6)
    args = parser.parse_args()
    with (ROOT / args.selection_csv).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    output_root = ROOT / args.output_root
    records = []
    for row in rows:
        prediction, ground_truth, valid, directory = read_pair(row, output_root)
        values = metric_values(prediction, ground_truth, valid)
        record = {**row, **values, "prediction_path": str(directory / f"{Path(row['filename']).stem}_metric_depth_m.npy")}
        records.append(record)

    per_image_fields = list(rows[0].keys()) + list(METRICS) + ["prediction_path"]
    write_csv(ROOT / args.results_dir / "uav3dcrop_per_image.csv", records, per_image_fields)
    summary = row_stats(records)
    pooled = pooled_metrics(records)
    for row in summary:
        row["pooled"] = pooled.get(row["metric"], float("nan"))
    write_csv(ROOT / args.results_dir / "uav3dcrop_summary.csv", summary, ["images", "metric", "mean", "median", "std", "min", "max", "pooled"])
    write_csv(ROOT / args.results_dir / "uav3dcrop_by_view.csv", row_stats(records, "view_type"), ["view_type", "images", "metric", "mean", "median", "std", "min", "max"])
    edges, sampled_gt_count = depth_edges(rows, output_root)
    write_csv(ROOT / args.results_dir / "uav3dcrop_by_depth.csv", depth_bin_rows(rows, output_root, edges), [
        "depth_bin_lower_m", "depth_bin_upper_m", "depth_range_m", "bin_method", "valid_pixels", "mae_m", "rmse_m", "absrel", "bias_m", "delta_1_25", "delta_1_25_2", "delta_1_25_3"
    ])
    write_json(ROOT / args.results_dir / "uav3dcrop_evaluation_manifest.json", {
        "depth_definition": "UAV3DCrop photogrammetry-referenced camera-frame z-depth along optical axis, meters",
        "valid_mask": "finite GT and prediction, GT>0, prediction>0",
        "resize": "none; RGB, GT TIFF, and prediction all matched 5280x3956 / 3956x5280 raster",
        "gt_interpolation": "none",
        "depth_bin_method": "five empirical GT quantile bins derived from sampled valid GT pixels",
        "sampled_gt_pixels_for_edges": sampled_gt_count,
        "records": records,
    })

    ranked = sorted(records, key=lambda row: float(row["rmse_m"]))
    chosen = []
    for row in ranked[:2] + ranked[len(ranked) // 2 : len(ranked) // 2 + 2] + ranked[-2:]:
        if row["filename"] not in {item["filename"] for item in chosen}:
            chosen.append(row)
    for row in chosen[: args.error_examples]:
        prediction, ground_truth, valid, directory = read_pair(row, output_root)
        save_error_figure(row, prediction, ground_truth, valid, directory, ROOT / args.error_dir)
    print(f"[uav3dcrop] evaluated {len(records)} images; pooled MAE={pooled['mae_m']:.4f} m, RMSE={pooled['rmse_m']:.4f} m, AbsRel={pooled['absrel']:.4f}")


if __name__ == "__main__":
    main()
