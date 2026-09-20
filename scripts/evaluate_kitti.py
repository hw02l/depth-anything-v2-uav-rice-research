from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from da2_experiments.io import load_depth, read_rgb, write_csv, write_json  # noqa: E402
from da2_experiments.metrics import (  # noqa: E402
    METRIC_NAMES,
    aggregate_depth_metrics,
    depth_bin_metrics,
    depth_metrics,
    summary_statistics,
)
from da2_experiments.modeling import METRIC_OUTDOOR_MODEL_ID, load_estimator  # noqa: E402


BINS = [(0.0, 10.0), (10.0, 20.0), (20.0, 40.0), (40.0, 60.0), (60.0, 80.0)]
PER_IMAGE_FIELDS = ["image", "valid_pixels", *METRIC_NAMES]
SUMMARY_FIELDS = ["metric", "mean", "median", "std", "min", "max", "pooled"]
BIN_FIELDS = ["depth_min_m", "depth_max_m", "depth_range_m", "images", "valid_pixels", *METRIC_NAMES]


def find_prediction(pred_dir: Path | None, stem: str) -> Path | None:
    if pred_dir is None:
        return None
    for suffix in ("_metric_depth_m.npy", "_raw_depth_meter.npy", "_raw_depth.npy", ".npy", ".png"):
        candidate = pred_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def find_ground_truth(gt_dir: Path, image_path: Path) -> Path:
    candidates = [
        gt_dir / image_path.name.replace("_sync_image_", "_sync_groundtruth_depth_", 1),
        gt_dir / image_path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    image_token = image_path.name.replace("_sync_image_", "_sync_", 1)
    matches = sorted(p for p in gt_dir.glob("*.png") if p.name.replace("_sync_groundtruth_depth_", "_sync_", 1) == image_token)
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Missing ground truth for {image_path.name} under {gt_dir}")


def load_kitti_gt(path: Path) -> np.ndarray:
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise OSError(f"Could not read KITTI ground truth: {path}")
    return np.asarray(raw, dtype=np.float32) / 256.0


def colorize(values: np.ndarray, cmap_name: str, vmin: float | None = None, vmax: float | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    valid = np.isfinite(array)
    if vmin is None or vmax is None:
        finite = array[valid]
        vmin, vmax = (np.percentile(finite, [1, 99]) if len(finite) else (0.0, 1.0))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    norm = np.clip((array - vmin) / (vmax - vmin), 0.0, 1.0)
    rgb = (plt.get_cmap(cmap_name)(norm)[..., :3] * 255).astype(np.uint8)
    rgb[~valid] = 0
    return rgb


def save_representative(record: dict, name: str, output_dir: Path) -> None:
    rgb = np.asarray(read_rgb(record["image_path"]))
    gt = record["gt"]
    pred = record["pred"]
    valid = np.isfinite(gt) & np.isfinite(pred) & (gt > 0) & (pred > 0) & (gt <= 80.0)
    common_max = float(np.percentile(gt[valid], 99)) if np.any(valid) else 80.0
    error = np.abs(pred - gt)
    error[~valid] = np.nan
    error_max = float(np.percentile(error[np.isfinite(error)], 99)) if np.any(np.isfinite(error)) else 1.0
    images = [rgb, colorize(gt, "turbo", 0.0, common_max), colorize(pred, "turbo", 0.0, common_max), colorize(error, "magma", 0.0, error_max)]
    titles = ["RGB", "GT depth (m)", "Predicted depth (m)", "Absolute error (m)"]
    figure, axes = plt.subplots(1, 4, figsize=(18, 4.5), constrained_layout=True)
    for axis, image, title in zip(axes, images, titles):
        axis.imshow(image)
        axis.set_title(title)
        axis.set_xlabel("pixel x")
        axis.set_ylabel("pixel y")
    figure.suptitle(f"KITTI {name}: {record['image']}")
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / f"kitti_{name}_example.png", dpi=160)
    plt.close(figure)


def make_plots(records: list[dict], rows: list[dict], bin_rows: list[dict], plot_dir: Path) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260920)
    scatter_gt, scatter_pred, error_gt, error_values, histogram_values = [], [], [], [], []
    for record in records:
        gt = record["gt"]
        pred = record["pred"]
        valid = np.isfinite(gt) & np.isfinite(pred) & (gt > 0) & (pred > 0) & (gt <= 80.0)
        gt_values, pred_values = gt[valid], pred[valid]
        if len(gt_values) > 5000:
            selected = rng.choice(len(gt_values), 5000, replace=False)
            gt_values, pred_values = gt_values[selected], pred_values[selected]
        scatter_gt.append(gt_values)
        scatter_pred.append(pred_values)
        error_gt.append(gt_values)
        error_values.append(np.abs(pred_values - gt_values))
        histogram_values.append(np.abs(pred_values - gt_values))
    gt_values = np.concatenate(scatter_gt)
    pred_values = np.concatenate(scatter_pred)
    depth_values = np.concatenate(error_gt)
    abs_errors = np.concatenate(error_values)
    hist_errors = np.concatenate(histogram_values)

    fig, axis = plt.subplots(figsize=(7, 6))
    axis.scatter(gt_values, pred_values, s=2, alpha=0.12, rasterized=True)
    axis.plot([0, 80], [0, 80], "k--", linewidth=1, label="y=x")
    axis.set(title="KITTI GT depth vs predicted depth", xlabel="GT depth (m)", ylabel="Predicted depth (m)")
    axis.set_xlim(0, 80); axis.set_ylim(0, 80); axis.legend(); fig.tight_layout(); fig.savefig(plot_dir / "gt_vs_pred_scatter.png", dpi=180); plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.scatter(depth_values, abs_errors, s=2, alpha=0.12, rasterized=True)
    axis.set(title="Absolute error as a function of GT depth", xlabel="GT depth (m)", ylabel="Absolute error (m)")
    axis.set_xlim(0, 80); fig.tight_layout(); fig.savefig(plot_dir / "absolute_error_vs_gt_depth.png", dpi=180); plt.close(fig)

    labels = [str(index + 1) for index in range(len(rows))]
    fig, axis = plt.subplots(figsize=(max(8, len(rows) * 0.22), 5))
    axis.bar(labels, [float(row["rmse_m"]) for row in rows], color="#4472c4")
    axis.set(title="RMSE by KITTI image", xlabel="Image order in selected validation subset", ylabel="RMSE (m)")
    axis.tick_params(axis="x", labelrotation=90); fig.tight_layout(); fig.savefig(plot_dir / "rmse_per_image.png", dpi=180); plt.close(fig)

    fig, axis = plt.subplots(figsize=(max(8, len(rows) * 0.22), 5))
    axis.bar(labels, [float(row["absrel"]) for row in rows], color="#ed7d31")
    axis.set(title="AbsRel by KITTI image", xlabel="Image order in selected validation subset", ylabel="AbsRel (dimensionless)")
    axis.tick_params(axis="x", labelrotation=90); fig.tight_layout(); fig.savefig(plot_dir / "absrel_per_image.png", dpi=180); plt.close(fig)

    clipped_max = float(np.percentile(hist_errors, 99.5)) if len(hist_errors) else 1.0
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.hist(np.clip(hist_errors, 0, clipped_max), bins=50, color="#70ad47", alpha=0.85)
    axis.set(title="KITTI absolute error histogram", xlabel="Absolute error (m; clipped at 99.5th percentile)", ylabel="Pixel count")
    fig.tight_layout(); fig.savefig(plot_dir / "absolute_error_histogram.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    x = np.arange(len(bin_rows))
    labels = [str(row["depth_range_m"]) for row in bin_rows]
    axes[0].plot(x, [row["mae_m"] for row in bin_rows], "o-", label="MAE")
    axes[0].plot(x, [row["rmse_m"] for row in bin_rows], "o-", label="RMSE")
    axes[0].set(title="Error by GT depth band", xlabel="GT depth band (m)", ylabel="Error (m)"); axes[0].set_xticks(x, labels); axes[0].legend()
    axes[1].plot(x, [row["absrel"] for row in bin_rows], "o-", color="#ed7d31")
    axes[1].set(title="Relative error by GT depth band", xlabel="GT depth band (m)", ylabel="AbsRel (dimensionless)"); axes[1].set_xticks(x, labels)
    axes[2].bar(x, [row["valid_pixels"] for row in bin_rows], color="#a5a5a5")
    axes[2].set(title="Valid pixels by GT depth band", xlabel="GT depth band (m)", ylabel="Valid pixel count"); axes[2].set_xticks(x, labels)
    fig.savefig(plot_dir / "error_by_depth.png", dpi=180); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Depth Anything V2 metric depth on multiple KITTI validation frames")
    parser.add_argument("--data-root", required=True, help="KITTI root containing val_selection_cropped")
    parser.add_argument("--pred-dir", help="Existing prediction directory; otherwise infer with the metric model")
    parser.add_argument("--outdir", default="outputs/kitti")
    parser.add_argument("--results-dir", default="results/metrics")
    parser.add_argument("--model-id", default=METRIC_OUTDOOR_MODEL_ID)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-depth", type=float, default=80.0)
    args = parser.parse_args()

    root = Path(args.data_root)
    image_dir = root / "val_selection_cropped" / "image"
    gt_dir = root / "val_selection_cropped" / "groundtruth_depth"
    images = sorted(image_dir.glob("*.png"))
    if args.limit is not None:
        images = images[: args.limit]
    if not images:
        raise FileNotFoundError(f"No KITTI images found under {image_dir}")
    batch_size = max(1, args.batch_size)
    pred_dir = Path(args.pred_dir) if args.pred_dir else None
    predictions: list[np.ndarray | None] = [None] * len(images)
    outdir = Path(args.outdir)
    prediction_output = outdir / "predictions"
    prediction_output.mkdir(parents=True, exist_ok=True)
    pending: list[int] = []
    for index, image_path in enumerate(images):
        existing = find_prediction(pred_dir, image_path.stem)
        if existing is not None:
            predictions[index] = load_depth(existing, png_scale=256.0 if existing.suffix.lower() == ".png" else 1.0)
        else:
            pending.append(index)
    estimator = None
    if pending:
        estimator = load_estimator("metric", args.model_id, args.device)
        for start in range(0, len(pending), batch_size):
            indices = pending[start : start + batch_size]
            batch = estimator.predict_batch([read_rgb(images[index]) for index in indices])
            for index, prediction in zip(indices, batch):
                predictions[index] = prediction
                np.save(prediction_output / f"{images[index].stem}_metric_depth_m.npy", prediction.astype(np.float32))
            print(f"[kitti] inferred {min(start + batch_size, len(pending))}/{len(pending)} images")

    records, rows, per_image_bin_rows = [], [], []
    for image_path, prediction in zip(images, predictions):
        if prediction is None:
            raise RuntimeError(f"No prediction was produced for {image_path}")
        gt_path = find_ground_truth(gt_dir, image_path)
        gt = load_kitti_gt(gt_path)
        if prediction.shape != gt.shape:
            prediction = cv2.resize(prediction, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
        saved_prediction = prediction_output / f"{image_path.stem}_metric_depth_m.npy"
        np.save(saved_prediction, prediction.astype(np.float32))
        metrics = depth_metrics(prediction, gt, max_depth=args.max_depth)
        row = {"image": image_path.name, **metrics}
        rows.append(row)
        per_image_bin_rows.append(depth_bin_metrics(prediction, gt, BINS, max_depth=args.max_depth))
        records.append({"image": image_path.name, "image_path": image_path, "gt": gt, "pred": prediction})
        print(f"[kitti] {image_path.name}: MAE={metrics['mae_m']:.4f} RMSE={metrics['rmse_m']:.4f} AbsRel={metrics['absrel']:.4f}")

    summary = aggregate_depth_metrics(rows)
    summary_stats = summary_statistics(rows)
    summary_rows = [{**row, "pooled": summary.get(row["metric"], float("nan"))} for row in summary_stats]
    write_csv(Path(args.results_dir) / "kitti_per_image.csv", rows, PER_IMAGE_FIELDS)
    write_csv(Path(args.results_dir) / "kitti_summary.csv", summary_rows, SUMMARY_FIELDS)

    bin_rows = []
    for bin_index, bounds in enumerate(BINS):
        bin_records = [rows_for_image[bin_index] for rows_for_image in per_image_bin_rows]
        aggregate = aggregate_depth_metrics(bin_records)
        bin_rows.append({"depth_min_m": bounds[0], "depth_max_m": bounds[1], "depth_range_m": f"{bounds[0]:g}-{bounds[1]:g}", **aggregate})
    write_csv(Path(args.results_dir) / "kitti_by_depth.csv", bin_rows, BIN_FIELDS)

    make_plots(records, rows, bin_rows, outdir / "plots")
    order = np.argsort([float(row["rmse_m"]) for row in rows])
    selected = {"best": int(order[0]), "median": int(order[len(order) // 2]), "worst": int(order[-1])}
    representative_dir = outdir / "representatives"
    for name, index in selected.items():
        save_representative(records[index], name, representative_dir)
    write_json(representative_dir / "selection.json", {name: rows[index] for name, index in selected.items()})

    write_json(outdir / "metrics.json", {
        "summary": {**summary, "model_id": args.model_id, "prediction_mode": "inferred_or_cached", "max_depth_m": args.max_depth},
        "per_image": rows,
        "summary_statistics": summary_rows,
        "by_depth": bin_rows,
        "representatives": {name: rows[index]["image"] for name, index in selected.items()},
    })
    print(f"[kitti] evaluated {len(rows)} images")
    print(f"[kitti] pooled summary: {summary}")


if __name__ == "__main__":
    main()
