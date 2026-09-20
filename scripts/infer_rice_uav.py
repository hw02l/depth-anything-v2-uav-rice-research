from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from da2_experiments.io import read_rgb, save_depth_outputs, write_json  # noqa: E402
from da2_experiments.modeling import METRIC_OUTDOOR_MODEL_ID, RELATIVE_MODEL_ID, load_estimator  # noqa: E402


def comparison_figure(rgb: np.ndarray, relative: np.ndarray, metric: np.ndarray, path: Path, title: str) -> None:
    relative_valid = np.isfinite(relative)
    metric_valid = np.isfinite(metric) & (metric > 0)
    relative_values = relative[relative_valid]
    metric_values = metric[metric_valid]
    relative_limits = np.percentile(relative_values, [1, 99]) if len(relative_values) else (0, 1)
    metric_limits = np.percentile(metric_values, [1, 99]) if len(metric_values) else (0, 1)
    figure, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    axes[0].imshow(rgb)
    axes[0].set_title("Original RGB")
    axes[1].imshow(relative, cmap="turbo", vmin=float(relative_limits[0]), vmax=float(relative_limits[1]))
    axes[1].set_title("Relative depth")
    axes[2].imshow(metric, cmap="turbo", vmin=float(metric_limits[0]), vmax=float(metric_limits[1]))
    axes[2].set_title("Metric depth (m)")
    for axis in axes:
        axis.set_xlabel("pixel x")
        axis.set_ylabel("pixel y")
    figure.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run relative and outdoor metric Depth Anything V2 on selected rice UAV frames")
    parser.add_argument("--selection-csv", default="results/rice_uav/selected_images.csv")
    parser.add_argument("--outdir", default="outputs/rice_uav")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    with Path(args.selection_csv).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise FileNotFoundError(f"No selected images in {args.selection_csv}")
    batch_size = max(1, args.batch_size)
    relative_estimator = load_estimator("relative", RELATIVE_MODEL_ID, args.device)
    metric_estimator = load_estimator("metric", METRIC_OUTDOOR_MODEL_ID, args.device)
    output_root = Path(args.outdir)
    output_root.mkdir(parents=True, exist_ok=True)
    run_rows = []

    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        images = [read_rgb(ROOT / row["local_path"]) for row in batch_rows]
        relative_predictions = relative_estimator.predict_batch(images)
        metric_predictions = metric_estimator.predict_batch(images)
        for row, image, relative, metric in zip(batch_rows, images, relative_predictions, metric_predictions):
            sample_dir = output_root / row["growth_stage"] / row["date"] / Path(row["filename"]).stem
            sample_dir.mkdir(parents=True, exist_ok=True)
            original_path = sample_dir / row["filename"]
            shutil.copy2(ROOT / row["local_path"], original_path)
            relative_saved = save_depth_outputs(relative, Path(row["filename"]).stem, sample_dir, "relative_depth")
            metric_saved = save_depth_outputs(metric, Path(row["filename"]).stem, sample_dir, "metric_depth_m")
            comparison_path = sample_dir / f"{Path(row['filename']).stem}_rgb_relative_metric.png"
            comparison_figure(np.asarray(image), relative, metric, comparison_path, f"{row['date']} {row['filename']}")
            metadata = {
                "date": row["date"],
                "filename": row["filename"],
                "growth_stage": row["growth_stage"],
                "rgb": str(original_path),
                "relative_model_id": RELATIVE_MODEL_ID,
                "metric_model_id": METRIC_OUTDOOR_MODEL_ID,
                "relative_outputs": relative_saved,
                "metric_outputs": metric_saved,
                "comparison": str(comparison_path),
                "shape": [int(metric.shape[0]), int(metric.shape[1])],
                "metric_depth_unit": "meter",
            }
            write_json(sample_dir / "run.json", metadata)
            run_rows.append(metadata)
            print(f"[rice] {row['date']}/{row['filename']} complete")

    write_json(output_root / "run_manifest.json", {
        "selection_csv": str(Path(args.selection_csv)),
        "relative_model_id": RELATIVE_MODEL_ID,
        "metric_model_id": METRIC_OUTDOOR_MODEL_ID,
        "images": run_rows,
    })
    print(f"[rice] wrote {len(run_rows)} image results under {output_root}")


if __name__ == "__main__":
    main()
