from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a cautious cross-dataset metric summary")
    parser.add_argument("--results-dir", default="results/metrics")
    parser.add_argument("--uav-results-dir", default="", help="directory containing the preferred UAV3DCrop summary; defaults to uav3dcrop_all when present")
    args = parser.parse_args()
    directory = ROOT / args.results_dir
    kitti = next(row for row in read_csv(directory / "kitti_summary.csv") if row["metric"] == "mae_m")
    kitti_rmse = next(row for row in read_csv(directory / "kitti_summary.csv") if row["metric"] == "rmse_m")
    kitti_absrel = next(row for row in read_csv(directory / "kitti_summary.csv") if row["metric"] == "absrel")
    eth = read_csv(directory / "eth3d_metrics.csv")[0]
    preferred_uav = ROOT / args.uav_results_dir if args.uav_results_dir else directory / "uav3dcrop_all"
    if not (preferred_uav / "uav3dcrop_summary.csv").exists():
        preferred_uav = directory
    uav_summary = read_csv(preferred_uav / "uav3dcrop_summary.csv")
    uav_metrics = {row["metric"]: row["pooled"] for row in uav_summary}
    uav_per_image = read_csv(preferred_uav / "uav3dcrop_per_image.csv")
    uav_views = read_csv(preferred_uav / "uav3dcrop_by_view.csv")
    view_counts = {
        row["view_type"]: int(row["images"])
        for row in uav_views
        if row["metric"] == "mae_m"
    }
    uav_viewpoint = " + ".join(f"{view_counts.get(view, 0)} {view}" for view in ("nadir", "oblique"))
    rows = [
        {"dataset": "KITTI", "domain": "driving", "camera_viewpoint": "forward road camera", "number_of_images": 50, "mae_m": float(kitti["pooled"]), "rmse_m": float(kitti_rmse["pooled"]), "absrel": float(kitti_absrel["pooled"]), "note": "50-image validation subset; GT depth selection crop"},
        {"dataset": "ETH3D", "domain": "indoor multi-view", "camera_viewpoint": "rectified stereo image", "number_of_images": 1, "mae_m": float(eth["mae_m"]), "rmse_m": float(eth["rmse_m"]), "absrel": float(eth["absrel"]), "note": "one low-res two-view image; disparity-derived metric depth"},
        {"dataset": "UAV3DCrop", "domain": "field crop UAV", "camera_viewpoint": uav_viewpoint, "number_of_images": len(uav_per_image), "mae_m": float(uav_metrics["mae_m"]), "rmse_m": float(uav_metrics["rmse_m"]), "absrel": float(uav_metrics["absrel"]), "note": "2025 wheat/oat/corn scenes; photogrammetry-referenced z-depth; raw metric prediction; domain-specific comparison only"},
    ]
    path = directory / "cross_dataset_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[summary] wrote {path}")


if __name__ == "__main__":
    main()
