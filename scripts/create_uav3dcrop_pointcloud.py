from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import tifffile
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from da2_experiments.io import write_csv, write_json  # noqa: E402
from da2_experiments.pointcloud import write_ascii_ply  # noqa: E402


def sample_dir(row: dict, output_root: Path) -> Path:
    return output_root / row["scene"].replace("/", "_") / row["view_type"] / Path(row["filename"]).stem


def load_camera(row: dict, data_root: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    transforms_path = data_root / row["scene"] / "transforms.json"
    transforms = json.loads(transforms_path.read_text(encoding="utf-8-sig"))
    K = np.array([[transforms["fl_x"], 0.0, transforms["cx"]], [0.0, transforms["fl_y"], transforms["cy"]], [0.0, 0.0, 1.0]], dtype=np.float64)
    distortion = np.array([transforms.get("k1", 0.0), transforms.get("k2", 0.0), transforms.get("p1", 0.0), transforms.get("p2", 0.0), transforms.get("k3", 0.0)], dtype=np.float64)
    return transforms, K, distortion


def camera_points_from_z(depth: np.ndarray, rgb: np.ndarray, K: np.ndarray, distortion: np.ndarray, stride: int, transform: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = depth.shape
    yy, xx = np.mgrid[0:height:stride, 0:width:stride]
    z = depth[::stride, ::stride].astype(np.float64)
    colors = rgb[::stride, ::stride, :3].astype(np.uint8)
    valid = np.isfinite(z) & (z > 0)
    pixels = np.column_stack((xx[valid].astype(np.float64), yy[valid].astype(np.float64)))
    normalized = cv2.undistortPoints(pixels.reshape(-1, 1, 2), K, distortion).reshape(-1, 2)
    z_values = z[valid]
    points_cv = np.column_stack((normalized[:, 0] * z_values, normalized[:, 1] * z_values, z_values))
    if transform is not None:
        # transforms.json uses a Nerfstudio/OpenGL camera-to-world frame. The
        # reference depth is OpenCV-style positive optical-axis z-depth, so
        # convert x-right/y-down/z-forward to x-right/y-up/z-back first.
        camera_opengl = points_cv * np.array([1.0, -1.0, -1.0])
        points = (transform[:3, :3] @ camera_opengl.T).T + transform[:3, 3]
    else:
        points = points_cv
    return points.astype(np.float32), colors[valid], points_cv.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate UAV3DCrop GT/DA2 point clouds and same-pixel 3D errors")
    parser.add_argument("--selection-csv", default="results/uav3dcrop/selected_images.csv")
    parser.add_argument("--output-root", default="outputs/uav3dcrop")
    parser.add_argument("--data-root", default="data/uav3dcrop")
    parser.add_argument("--results-dir", default="results/metrics")
    parser.add_argument("--pointcloud-dir", default="outputs/uav3dcrop/pointcloud")
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--ply-stride", type=int, default=32)
    parser.add_argument("--ply-per-view", type=int, default=2)
    args = parser.parse_args()
    with (ROOT / args.selection_csv).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    output_root = ROOT / args.output_root
    pointcloud_dir = ROOT / args.pointcloud_dir
    selected_for_ply = {view: [row for row in rows if row["view_type"] == view][: args.ply_per_view] for view in ("nadir", "oblique")}
    ply_filenames = {row["filename"] for group in selected_for_ply.values() for row in group}
    error_rows = []
    for row in rows:
        transforms, K, distortion = load_camera(row, ROOT / args.data_root)
        frame = next(item for item in transforms["frames"] if Path(item["file_path"]).name == row["filename"])
        T = np.asarray(frame["transform_matrix"], dtype=np.float64)
        rgb = np.asarray(Image.open(ROOT / row["local_rgb_path"]).convert("RGB"))
        gt = tifffile.imread(ROOT / row["local_depth_path"]).astype(np.float32)
        pred = np.load(sample_dir(row, output_root) / f"{Path(row['filename']).stem}_metric_depth_m.npy").astype(np.float32)
        common_valid = np.isfinite(gt) & (gt > 0) & np.isfinite(pred) & (pred > 0)
        # Use the same valid pixel support for correspondence errors.  A few
        # sparse invalid GT pixels otherwise make independently filtered point
        # arrays have different lengths even though the rasters are aligned.
        gt_common = np.where(common_valid, gt, 0.0)
        pred_common = np.where(common_valid, pred, 0.0)
        gt_world, colors, gt_camera = camera_points_from_z(gt_common, rgb, K, distortion, args.stride, T)
        pred_world, _, pred_camera = camera_points_from_z(pred_common, rgb, K, distortion, args.stride, T)
        if len(gt_camera) != len(pred_camera):
            raise ValueError(f"GT/pred valid sampled point count mismatch for {row['filename']}")
        difference = pred_camera.astype(np.float64) - gt_camera.astype(np.float64)
        distances = np.linalg.norm(difference, axis=1)
        error_rows.append({
            **row,
            "sample_stride": args.stride,
            "corresponding_points": int(len(distances)),
            "mean_3d_error_m": float(np.mean(distances)),
            "median_3d_error_m": float(np.median(distances)),
            "rmse_3d_error_m": float(np.sqrt(np.mean(distances**2))),
            "p95_3d_error_m": float(np.percentile(distances, 95)),
            "distortion_correction": "cv2.undistortPoints with OPENCV k1,k2,p1,p2,k3",
            "world_transform": "camera OpenCV x-right/y-down/z-forward converted to Nerfstudio x-right/y-up/z-back, then transforms.json camera-to-world",
        })
        if row["filename"] in ply_filenames:
            stem = Path(row["filename"]).stem
            destination = pointcloud_dir / row["scene"].replace("/", "_") / row["view_type"]
            destination.mkdir(parents=True, exist_ok=True)
            gt_points, gt_colors, _ = camera_points_from_z(gt, rgb, K, distortion, args.ply_stride, T)
            pred_points, pred_colors, _ = camera_points_from_z(pred, rgb, K, distortion, args.ply_stride, T)
            write_ascii_ply(destination / f"{stem}_gt_world.ply", gt_points, gt_colors)
            write_ascii_ply(destination / f"{stem}_pred_world.ply", pred_points, pred_colors)
            # Camera-coordinate copies make single-view inspection independent
            # of the world-frame convention.
            write_ascii_ply(destination / f"{stem}_gt_camera.ply", gt_camera, colors)
            _, pred_colors_camera, pred_camera_full = camera_points_from_z(pred, rgb, K, distortion, args.ply_stride, None)
            write_ascii_ply(destination / f"{stem}_pred_camera.ply", pred_camera_full, pred_colors_camera)

    fields = list(error_rows[0].keys()) if error_rows else []
    write_csv(ROOT / args.results_dir / "uav3dcrop_point_error.csv", error_rows, fields)
    write_json(ROOT / args.results_dir / "uav3dcrop_pointcloud_method.json", {
        "depth_convention": "UAV3DCrop GT and DA2 are compared as camera-frame z-depth along the optical axis in meters",
        "intrinsics": "transforms.json fl_x, fl_y, cx, cy",
        "distortion": "OPENCV k1,k2,p1,p2,k3 applied with cv2.undistortPoints; no depth raster interpolation",
        "world_transform": "T_c2w @ [x_cv, -y_cv, -z_cv, 1] because transforms.json uses Nerfstudio/OpenGL camera axes",
        "alignment": "none; raw metric prediction and GT only",
        "point_error": "same-pixel corresponding 3D Euclidean error, not nearest-neighbor or ICP",
        "ply_stride": args.ply_stride,
        "images_with_ply": sorted(ply_filenames),
    })
    print(f"[uav3dcrop] wrote same-pixel 3D errors for {len(error_rows)} images and PLYs for {len(ply_filenames)} images")


if __name__ == "__main__":
    main()
