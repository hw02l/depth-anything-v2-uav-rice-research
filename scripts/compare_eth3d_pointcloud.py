from __future__ import annotations

import argparse
import sys

import open3d as o3d

from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from da2_experiments.io import write_json  # noqa: E402
from da2_experiments.pointcloud_eval import compare_point_clouds  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two ETH3D point clouds in the common im0 camera frame")
    parser.add_argument("--pred-ply", required=True)
    parser.add_argument("--gt-ply", required=True)
    parser.add_argument("--out-json", default="outputs/eth3d_gt/im0_pointcloud_comparison.json")
    parser.add_argument("--max-points", type=int, default=50000)
    parser.add_argument("--visualize", action="store_true", help="Open an interactive Open3D window")
    args = parser.parse_args()
    result = compare_point_clouds(args.pred_ply, args.gt_ply, max_points=args.max_points)
    result.update({"coordinate_frame": "same ETH3D im0 rectified camera coordinates", "alignment": "none; same intrinsics and per-pixel depth coordinates"})
    write_json(args.out_json, result)
    print(result)
    if args.visualize:
        pred = o3d.io.read_point_cloud(args.pred_ply)
        gt = o3d.io.read_point_cloud(args.gt_ply)
        pred.paint_uniform_color([0.1, 0.7, 1.0])
        gt.paint_uniform_color([1.0, 0.2, 0.1])
        o3d.visualization.draw_geometries([pred, gt], window_name="ETH3D predicted (blue) vs GT (red)")


if __name__ == "__main__":
    main()
