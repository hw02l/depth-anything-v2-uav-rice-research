from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import tifffile
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def sample_dir(row: dict, output_root: Path) -> Path:
    return output_root / row["scene"].replace("/", "_") / row["view_type"] / Path(row["filename"]).stem


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def show_depth(axis, array: np.ndarray, title: str, cmap: str = "viridis", stride: int = 8) -> None:
    image = array[::stride, ::stride]
    plot = axis.imshow(image, cmap=cmap)
    axis.set_title(title)
    axis.axis("off")
    axis.figure.colorbar(plot, ax=axis, fraction=0.046, pad=0.04)


def show_rgb(axis, path: Path, title: str, stride: int = 8) -> None:
    image = np.asarray(Image.open(path).convert("RGB"))[::stride, ::stride]
    axis.imshow(image)
    axis.set_title(title)
    axis.axis("off")


def save(figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def figure_1(row: dict, output_root: Path, report_dir: Path) -> None:
    directory = sample_dir(row, output_root)
    gt = tifffile.imread(ROOT / row["local_depth_path"]).astype(np.float32)
    figure, axes = plt.subplots(1, 2, figsize=(13, 6))
    show_rgb(axes[0], ROOT / row["local_rgb_path"], f"RGB | {row['view_type']} | {row['filename']}")
    show_depth(axes[1], gt, "UAV3DCrop GT z-depth (m)")
    figure.suptitle("Figure 1. UAV3DCrop RGB and photogrammetry-referenced GT depth")
    save(figure, report_dir / "figure_1_rgb_gt_depth.png")


def figure_2(rows: list[dict], output_root: Path, report_dir: Path) -> None:
    chosen = [next(row for row in rows if row["view_type"] == view) for view in ("nadir", "oblique")]
    figure, axes = plt.subplots(2, 3, figsize=(18, 11))
    for i, row in enumerate(chosen):
        directory = sample_dir(row, output_root)
        stem = Path(row["filename"]).stem
        gt = tifffile.imread(ROOT / row["local_depth_path"]).astype(np.float32)
        pred = np.load(directory / f"{stem}_metric_depth_m.npy")
        show_rgb(axes[i, 0], ROOT / row["local_rgb_path"], f"{row['view_type']} RGB")
        show_depth(axes[i, 1], gt, "GT z-depth (m)")
        show_depth(axes[i, 2], pred, "DA2 metric depth (m)")
    figure.suptitle("Figure 2. RGB / UAV3DCrop GT depth / Depth Anything V2 metric depth")
    save(figure, report_dir / "figure_2_rgb_gt_pred.png")


def figure_3(row: dict, output_root: Path, report_dir: Path) -> None:
    directory = sample_dir(row, output_root)
    stem = Path(row["filename"]).stem
    gt = tifffile.imread(ROOT / row["local_depth_path"]).astype(np.float32)
    pred = np.load(directory / f"{stem}_metric_depth_m.npy")
    valid = np.isfinite(gt) & (gt > 0) & np.isfinite(pred) & (pred > 0)
    error = np.where(valid, np.abs(pred - gt), np.nan)
    figure, axes = plt.subplots(1, 4, figsize=(20, 5.5))
    show_rgb(axes[0], ROOT / row["local_rgb_path"], "RGB")
    show_depth(axes[1], gt, "GT z-depth (m)")
    show_depth(axes[2], pred, "DA2 metric depth (m)")
    show_depth(axes[3], error, "Absolute error (m)", cmap="magma")
    figure.suptitle(f"Figure 3. Absolute error map | {row['view_type']} | {row['filename']}")
    save(figure, report_dir / "figure_3_absolute_error.png")


def figure_4(row: dict, output_root: Path, report_dir: Path) -> None:
    directory = sample_dir(row, output_root)
    stem = Path(row["filename"]).stem
    gt = tifffile.imread(ROOT / row["local_depth_path"]).astype(np.float32)
    pred = np.load(directory / f"{stem}_metric_depth_m.npy")
    valid = np.isfinite(gt) & (gt > 0) & np.isfinite(pred) & (pred > 0)
    bias = float(np.median((pred - gt)[valid]))
    corrected = pred - bias
    figure, axes = plt.subplots(1, 4, figsize=(20, 5.5))
    show_depth(axes[0], gt, "GT z-depth (m)")
    show_depth(axes[1], pred, "Raw DA2 metric (m)")
    show_depth(axes[2], corrected, f"Bias-corrected (m), bias={bias:.3f}")
    show_depth(axes[3], np.where(valid, corrected - gt, np.nan), "Corrected signed error (m)", cmap="coolwarm")
    figure.suptitle(f"Figure 4. Raw and global-bias-corrected depth | {row['filename']}")
    save(figure, report_dir / "figure_4_raw_bias_corrected.png")


def figure_5(report_dir: Path) -> None:
    profiles = sorted((ROOT / "outputs/uav3dcrop/depth_profiles").glob("*.png"))[:3]
    figure, axes = plt.subplots(1, max(1, len(profiles)), figsize=(18, 5))
    axes = np.atleast_1d(axes)
    for axis, path in zip(axes, profiles):
        image = Image.open(path)
        image.thumbnail((1400, 700), Image.Resampling.LANCZOS)
        axis.imshow(np.asarray(image))
        axis.set_title(path.stem)
        axis.axis("off")
    figure.suptitle("Figure 5. GT depth profile versus DA2 and bias-corrected profiles")
    save(figure, report_dir / "figure_5_depth_profiles.png")


def figure_6(results_dir: Path, report_dir: Path) -> None:
    rows = load_rows(results_dir / "uav3dcrop_local_depth_difference_summary.csv")
    figure, axis = plt.subplots(figsize=(10, 6))
    for variant in ("metric_raw", "metric_scale_shift", "relative_scale_shift"):
        subset = [row for row in rows if row["variant"] == variant and row["metric"] == "local_difference_mae_m"]
        if subset:
            subset.sort(key=lambda row: int(row["pixel_separation"]))
            axis.plot([int(row["pixel_separation"]) for row in subset], [float(row["mean"]) for row in subset], marker="o", label=variant)
    axis.set_title("Figure 6. Local depth-difference error versus pixel separation")
    axis.set_xlabel("Pixel separation (px)")
    axis.set_ylabel("Local difference MAE (m)")
    axis.grid(alpha=0.25)
    axis.legend()
    save(figure, report_dir / "figure_6_local_depth_difference.png")


def figure_7(results_dir: Path, report_dir: Path) -> None:
    rows = load_rows(results_dir / "uav3dcrop_by_view.csv")
    metrics = ["mae_m", "rmse_m", "absrel", "bias_m"]
    figure, axes = plt.subplots(1, 4, figsize=(18, 5.5))
    for axis, metric in zip(axes, metrics):
        subset = [row for row in rows if row["metric"] == metric]
        views = [row["view_type"] for row in subset]
        values = [float(row["mean"]) for row in subset]
        axis.bar(views, values, color=["#4c78a8", "#f58518"][: len(values)])
        axis.set_title(metric)
        axis.set_xlabel("View type")
        axis.set_ylabel("Mean per-image value")
    figure.suptitle("Figure 7. Nadir versus oblique raw metric-depth errors")
    save(figure, report_dir / "figure_7_nadir_oblique.png")


def figure_8(pointcloud_dir: Path, report_dir: Path) -> None:
    gt_files = sorted(pointcloud_dir.glob("**/*_gt_world.ply"))
    if not gt_files:
        return
    gt_file = gt_files[0]
    pred_file = gt_file.with_name(gt_file.name.replace("_gt_world.ply", "_pred_world.ply"))
    gt = np.asarray(o3d.io.read_point_cloud(str(gt_file)).points)
    pred = np.asarray(o3d.io.read_point_cloud(str(pred_file)).points)
    figure, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].scatter(gt[:, 0][::4], gt[:, 2][::4], s=0.2, c="#2ca02c", label="GT")
    axes[0].set_title("GT world point cloud projection")
    axes[1].scatter(pred[:, 0][::4], pred[:, 2][::4], s=0.2, c="#d62728", label="DA2")
    axes[1].set_title("DA2 world point cloud projection")
    for axis in axes:
        axis.set_xlabel("World X (m)")
        axis.set_ylabel("World Z (m)")
        axis.legend()
    figure.suptitle("Figure 8. GT and Depth Anything V2 single-view point clouds")
    save(figure, report_dir / "figure_8_pointclouds.png")


def figure_9(pointcloud_dir: Path, report_dir: Path) -> None:
    fused = sorted((pointcloud_dir).glob("**/*_gt_pred_overlay.ply"))
    if not fused:
        return
    points = np.asarray(o3d.io.read_point_cloud(str(fused[0])).points)
    colors = np.asarray(o3d.io.read_point_cloud(str(fused[0])).colors)
    figure, axis = plt.subplots(figsize=(9, 7))
    axis.scatter(points[:, 0][::8], points[:, 2][::8], s=0.2, c=colors[::8])
    axis.set_title("Figure 9. Multi-view fused GT/predicted point clouds")
    axis.set_xlabel("World X (m)")
    axis.set_ylabel("World Z (m)")
    save(figure, report_dir / "figure_9_multiview_fused.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create report figures for UAV3DCrop experiments")
    parser.add_argument("--selection-csv", default="results/uav3dcrop/selected_images.csv")
    parser.add_argument("--output-root", default="outputs/uav3dcrop")
    parser.add_argument("--results-dir", default="results/metrics")
    parser.add_argument("--report-dir", default="outputs/uav3dcrop/report_figures")
    args = parser.parse_args()
    rows = load_rows(ROOT / args.selection_csv)
    output_root = ROOT / args.output_root
    report_dir = ROOT / args.report_dir
    per_image = load_rows(ROOT / args.results_dir / "uav3dcrop_per_image.csv")
    best = min(per_image, key=lambda row: float(row["rmse_m"]))
    worst = max(per_image, key=lambda row: float(row["rmse_m"]))
    median = sorted(per_image, key=lambda row: float(row["rmse_m"]))[len(per_image) // 2]
    representative = next(row for row in rows if row["filename"] == median["filename"])
    figure_1(next(row for row in rows if row["view_type"] == "nadir"), output_root, report_dir)
    figure_2(rows, output_root, report_dir)
    figure_3(next(row for row in rows if row["filename"] == worst["filename"]), output_root, report_dir)
    figure_4(representative, output_root, report_dir)
    figure_5(report_dir)
    figure_6(ROOT / args.results_dir, report_dir)
    figure_7(ROOT / args.results_dir, report_dir)
    figure_8(ROOT / "outputs/uav3dcrop/pointcloud", report_dir)
    figure_9(ROOT / "outputs/uav3dcrop/pointcloud", report_dir)
    print(f"[uav3dcrop] wrote report figures to {report_dir}")


if __name__ == "__main__":
    main()
