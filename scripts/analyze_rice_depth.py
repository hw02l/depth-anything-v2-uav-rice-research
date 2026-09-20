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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from da2_experiments.io import read_rgb, write_csv, write_json  # noqa: E402


def metric_path(row: dict, output_root: Path) -> Path:
    return output_root / row["growth_stage"] / row["date"] / Path(row["filename"]).stem / f"{Path(row['filename']).stem}_metric_depth_m.npy"


def finite_depth(depth: np.ndarray) -> np.ndarray:
    return depth[np.isfinite(depth) & (depth > 0)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Depth Anything V2 metric depth on rice UAV frames")
    parser.add_argument("--selection-csv", default="results/rice_uav/selected_images.csv")
    parser.add_argument("--output-root", default="outputs/rice_uav")
    parser.add_argument("--results-dir", default="results/metrics")
    args = parser.parse_args()
    with Path(args.selection_csv).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    statistics = []
    distributions = []
    for row in rows:
        depth_file = metric_path(row, Path(args.output_root))
        depth = np.load(depth_file).astype(np.float32)
        values = finite_depth(depth)
        if len(values) == 0:
            raise ValueError(f"No valid metric depth in {depth_file}")
        statistics.append({
            "date": row["date"],
            "filename": row["filename"],
            "growth_stage": row["growth_stage"],
            "valid_pixels": int(len(values)),
            "minimum_m": float(np.min(values)),
            "p01_m": float(np.percentile(values, 1)),
            "p05_m": float(np.percentile(values, 5)),
            "median_m": float(np.median(values)),
            "mean_m": float(np.mean(values)),
            "p95_m": float(np.percentile(values, 95)),
            "p99_m": float(np.percentile(values, 99)),
            "maximum_m": float(np.max(values)),
            "std_m": float(np.std(values)),
            "gps_altitude_recorded_m": float(json.loads(row["exif_values"]).get("GPS_GPSAltitude", "nan")),
            "documented_flight_altitude": row["documented_flight_altitude"],
            "metric_depth_source": "Depth-Anything-V2-Metric-Outdoor-Small-hf",
        })
        sample = values[::max(1, len(values) // 10000)]
        distributions.append({"date": row["date"], "growth_stage": row["growth_stage"], "values": sample})

    output_dir = Path(args.results_dir)
    fields = [
        "date", "filename", "growth_stage", "valid_pixels", "minimum_m", "p01_m", "p05_m", "median_m",
        "mean_m", "p95_m", "p99_m", "maximum_m", "std_m", "gps_altitude_recorded_m",
        "documented_flight_altitude", "metric_depth_source",
    ]
    write_csv(output_dir / "rice_uav_depth_statistics.csv", statistics, fields)

    figure, axis = plt.subplots(figsize=(10, 6))
    for distribution in distributions:
        axis.hist(distribution["values"], bins=60, density=True, alpha=0.25, label=f"{distribution['date']} ({distribution['growth_stage']})")
    axis.set_title("Rice UAV metric depth distribution by date")
    axis.set_xlabel("Depth from camera (m)")
    axis.set_ylabel("Density")
    axis.legend(fontsize=8)
    figure.tight_layout()
    plot_dir = Path("outputs/rice_uav/plots")
    plot_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_dir / "metric_depth_distribution_by_date.png", dpi=160)
    plt.close(figure)

    grouped = {}
    for item in statistics:
        grouped.setdefault(item["growth_stage"], []).append(item["median_m"])
    figure, axis = plt.subplots(figsize=(8, 5))
    labels = ["early", "middle", "late"]
    axis.boxplot([grouped.get(label, []) for label in labels], tick_labels=labels, showmeans=True)
    axis.set_title("Rice UAV per-image median metric depth by growth-stage group")
    axis.set_xlabel("Growth stage group")
    axis.set_ylabel("Per-image median depth (m)")
    figure.tight_layout()
    figure.savefig(plot_dir / "growth_stage_metric_depth_boxplot.png", dpi=160)
    plt.close(figure)
    write_json(Path(args.output_root) / "depth_statistics_manifest.json", {"statistics_csv": str(output_dir / "rice_uav_depth_statistics.csv"), "images": statistics})
    print(f"[rice] wrote depth statistics for {len(statistics)} images")


if __name__ == "__main__":
    main()
