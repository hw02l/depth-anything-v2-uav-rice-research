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
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from da2_experiments.io import read_rgb, save_depth_outputs, write_json  # noqa: E402
from da2_experiments.modeling import (  # noqa: E402
    METRIC_OUTDOOR_MODEL_ID,
    RELATIVE_MODEL_ID,
    load_estimator,
)


def sample_dir(row: dict, output_root: Path) -> Path:
    scene_key = row["scene"].replace("/", "_")
    return output_root / scene_key / row["view_type"] / Path(row["filename"]).stem


def comparison_figure(rgb_path: Path, relative_path: Path, metric_path: Path, destination: Path, title: str) -> None:
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
    relative = np.asarray(Image.open(relative_path).convert("RGB"))
    metric = np.asarray(Image.open(metric_path).convert("RGB"))
    figure, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(rgb)
    axes[0].set_title("RGB")
    axes[1].imshow(relative)
    axes[1].set_title("Relative depth")
    axes[2].imshow(metric)
    axes[2].set_title("Outdoor metric depth (m)")
    for axis in axes:
        axis.axis("off")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Depth Anything V2 on a UAV3DCrop subset")
    parser.add_argument("--selection-csv", default="results/uav3dcrop/selected_images.csv")
    parser.add_argument("--output-root", default="outputs/uav3dcrop")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    with (ROOT / args.selection_csv).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Selection CSV contains no images")

    relative_estimator = load_estimator("relative", model_id=RELATIVE_MODEL_ID, device=args.device)
    metric_estimator = load_estimator("metric", model_id=METRIC_OUTDOOR_MODEL_ID, device=args.device)
    output_root = ROOT / args.output_root
    manifest_rows = []

    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start : start + args.batch_size]
        images = [read_rgb(ROOT / row["local_rgb_path"]) for row in batch_rows]
        relative_depths = relative_estimator.predict_batch(images)
        metric_depths = metric_estimator.predict_batch(images)
        for row, image, relative_depth, metric_depth in zip(batch_rows, images, relative_depths, metric_depths):
            destination = sample_dir(row, output_root)
            destination.mkdir(parents=True, exist_ok=True)
            stem = Path(row["filename"]).stem
            rgb_destination = destination / Path(row["filename"]).name
            # Preserve the downloaded source bytes for the "original RGB"
            # artifact; PIL is used only for inference and visualization.
            shutil.copy2(ROOT / row["local_rgb_path"], rgb_destination)
            relative_paths = save_depth_outputs(relative_depth, stem, destination, "relative_depth")
            metric_paths = save_depth_outputs(metric_depth, stem, destination, "metric_depth_m")
            comparison_path = destination / f"{stem}_rgb_relative_metric.png"
            comparison_figure(
                rgb_destination,
                Path(relative_paths["visualization"]),
                Path(metric_paths["visualization"]),
                comparison_path,
                f"{row['scene']} | {row['view_type']} | {row['filename']}",
            )
            sample_manifest = {
                "scene": row["scene"],
                "filename": row["filename"],
                "view_type": row["view_type"],
                "rgb_path": str(rgb_destination),
                "relative_depth_npy": relative_paths["raw"],
                "relative_depth_visualization": relative_paths["visualization"],
                "metric_depth_npy": metric_paths["raw"],
                "metric_depth_visualization": metric_paths["visualization"],
                "comparison_figure": str(comparison_path),
                "relative_model_id": RELATIVE_MODEL_ID,
                "metric_model_id": METRIC_OUTDOOR_MODEL_ID,
                "metric_max_depth_m": 80.0,
            }
            write_json(destination / "run.json", sample_manifest)
            manifest_rows.append(sample_manifest)
            print(f"[uav3dcrop] {row['view_type']} {row['filename']}")

    write_json(output_root / "run_manifest.json", {
        "relative_model_id": RELATIVE_MODEL_ID,
        "metric_model_id": METRIC_OUTDOOR_MODEL_ID,
        "metric_max_depth_m": 80.0,
        "images": manifest_rows,
    })
    print(f"[uav3dcrop] wrote predictions for {len(manifest_rows)} images")


if __name__ == "__main__":
    main()
