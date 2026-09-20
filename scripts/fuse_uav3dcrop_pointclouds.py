from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import tifffile
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from create_uav3dcrop_pointcloud import camera_points_from_z, load_camera  # noqa: E402
from da2_experiments.io import write_json  # noqa: E402
from da2_experiments.pointcloud import write_ascii_ply  # noqa: E402


def sample_dir(row: dict, output_root: Path) -> Path:
    return output_root / row["scene"].replace("/", "_") / row["view_type"] / Path(row["filename"]).stem


def choose_rows(rows: list[dict], view_type: str, count: int) -> list[dict]:
    candidates = [row for row in rows if row["view_type"] == view_type]
    if len(candidates) <= count:
        return candidates
    indices = np.linspace(0, len(candidates) - 1, count, dtype=int)
    return [candidates[int(index)] for index in indices]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse a small UAV3DCrop subset into transforms.json world coordinates")
    parser.add_argument("--selection-csv", default="results/uav3dcrop/selected_images.csv")
    parser.add_argument("--output-root", default="outputs/uav3dcrop")
    parser.add_argument("--data-root", default="data/uav3dcrop")
    parser.add_argument("--pointcloud-dir", default="outputs/uav3dcrop/pointcloud")
    parser.add_argument("--view-type", default="nadir", choices=("nadir", "oblique"))
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--stride", type=int, default=32)
    args = parser.parse_args()

    import csv

    with (ROOT / args.selection_csv).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    chosen = choose_rows(rows, args.view_type, args.count)
    if not chosen:
        raise ValueError(f"No {args.view_type} rows found")

    gt_points_all, gt_colors_all = [], []
    pred_points_all, pred_colors_all = [], []
    for row in chosen:
        transforms, K, distortion = load_camera(row, ROOT / args.data_root)
        frame = next(item for item in transforms["frames"] if Path(item["file_path"]).name == row["filename"])
        T = np.asarray(frame["transform_matrix"], dtype=np.float64)
        rgb = np.asarray(Image.open(ROOT / row["local_rgb_path"]).convert("RGB"))
        gt = tifffile.imread(ROOT / row["local_depth_path"]).astype(np.float32)
        pred = np.load(sample_dir(row, ROOT / args.output_root) / f"{Path(row['filename']).stem}_metric_depth_m.npy").astype(np.float32)
        gt_points, gt_colors, _ = camera_points_from_z(gt, rgb, K, distortion, args.stride, T)
        pred_points, pred_colors, _ = camera_points_from_z(pred, rgb, K, distortion, args.stride, T)
        gt_points_all.append(gt_points)
        gt_colors_all.append(gt_colors)
        pred_points_all.append(pred_points)
        pred_colors_all.append(pred_colors)

    gt_points = np.concatenate(gt_points_all)
    gt_colors = np.concatenate(gt_colors_all)
    pred_points = np.concatenate(pred_points_all)
    pred_colors = np.concatenate(pred_colors_all)
    destination = ROOT / args.pointcloud_dir / chosen[0]["scene"].replace("/", "_") / "fused"
    destination.mkdir(parents=True, exist_ok=True)
    write_ascii_ply(destination / f"{args.view_type}_{args.count}_gt_world_fused.ply", gt_points, gt_colors)
    write_ascii_ply(destination / f"{args.view_type}_{args.count}_pred_world_fused.ply", pred_points, pred_colors)
    gt_overlay = np.clip(0.45 * gt_colors.astype(np.float32) + np.array([0, 80, 160], dtype=np.float32), 0, 255).astype(np.uint8)
    pred_overlay = np.clip(0.45 * pred_colors.astype(np.float32) + np.array([180, 20, 0], dtype=np.float32), 0, 255).astype(np.uint8)
    write_ascii_ply(destination / f"{args.view_type}_{args.count}_gt_pred_overlay.ply", np.concatenate([gt_points, pred_points]), np.concatenate([gt_overlay, pred_overlay]))

    sparse_path = ROOT / args.data_root / chosen[0]["scene"] / "sparse_pc.ply"
    sparse = o3d.io.read_point_cloud(str(sparse_path))
    sparse_points = np.asarray(sparse.points)
    sparse_colors = np.asarray(sparse.colors)
    if len(sparse_points) > 100_000:
        indices = np.linspace(0, len(sparse_points) - 1, 100_000, dtype=int)
        sparse_points = sparse_points[indices]
        sparse_colors = sparse_colors[indices]
    write_ascii_ply(destination / "sparse_pc_sample.ply", sparse_points.astype(np.float32), np.clip(sparse_colors * 255, 0, 255).astype(np.uint8))

    write_json(destination / "fusion_manifest.json", {
        "scene": chosen[0]["scene"],
        "view_type": args.view_type,
        "images": [row["filename"] for row in chosen],
        "count": len(chosen),
        "stride": args.stride,
        "coordinate_convention": "camera OpenCV x-right/y-down/z-forward -> Nerfstudio x-right/y-up/z-back -> transforms.json camera-to-world",
        "alignment": "none; no ICP, rigid alignment, or scale alignment",
        "sparse_pc": str(sparse_path),
        "gt_points": int(len(gt_points)),
        "pred_points": int(len(pred_points)),
    })
    print(f"[uav3dcrop] fused {len(chosen)} {args.view_type} images in world coordinates")


if __name__ == "__main__":
    main()
