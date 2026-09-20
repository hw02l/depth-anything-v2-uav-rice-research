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

from da2_experiments.eth3d import disparity_to_depth_m, read_eth3d_calib, read_pfm  # noqa: E402
from da2_experiments.io import load_depth, read_rgb, save_depth_outputs, write_csv, write_json  # noqa: E402
from da2_experiments.metrics import depth_metrics  # noqa: E402
from da2_experiments.modeling import METRIC_INDOOR_MODEL_ID, load_estimator  # noqa: E402
from da2_experiments.pointcloud import depth_to_points, write_ascii_ply  # noqa: E402
from da2_experiments.pointcloud_eval import compare_point_clouds  # noqa: E402


def colorize(values: np.ndarray, cmap: str, vmin: float, vmax: float) -> np.ndarray:
    data = np.asarray(values, dtype=np.float32)
    valid = np.isfinite(data)
    scaled = np.clip((data - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    rgb = (plt.get_cmap(cmap)(scaled)[..., :3] * 255).astype(np.uint8)
    rgb[~valid] = 0
    return rgb


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Depth Anything V2 against ETH3D low-res two-view disparity converted to metric depth")
    parser.add_argument("--scene", default="data/eth3d/delivery_area_1l")
    parser.add_argument("--rgb", default="im0.png")
    parser.add_argument("--disparity", default="disp0GT.pfm")
    parser.add_argument("--calib", default="calib.txt")
    parser.add_argument("--pred-depth", help="Existing predicted metric depth .npy/.png; otherwise infer")
    parser.add_argument("--model-id", default=METRIC_INDOOR_MODEL_ID)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--outdir", default="outputs/eth3d_gt")
    parser.add_argument("--results-dir", default="results/metrics")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--max-depth", type=float, default=80.0)
    parser.add_argument("--pointcloud-max-points", type=int, default=50000)
    args = parser.parse_args()

    scene = Path(args.scene)
    rgb_path = scene / args.rgb
    disparity_path = scene / args.disparity
    calib_path = scene / args.calib
    rgb_image = read_rgb(rgb_path)
    rgb = np.asarray(rgb_image)
    calibration = read_eth3d_calib(calib_path)
    disparity = read_pfm(disparity_path)
    gt_depth = disparity_to_depth_m(disparity, calibration["fx"], calibration["baseline_mm"], calibration["doffs_px"])
    if args.pred_depth:
        pred_path = Path(args.pred_depth)
        predicted = load_depth(pred_path)
        source = str(pred_path)
    else:
        estimator = load_estimator("metric", args.model_id, args.device)
        predicted = estimator.predict(rgb_image)
        source = args.model_id
    if predicted.shape != gt_depth.shape:
        predicted = cv2.resize(predicted, (gt_depth.shape[1], gt_depth.shape[0]), interpolation=cv2.INTER_LINEAR)
    if predicted.shape != rgb.shape[:2]:
        predicted = cv2.resize(predicted, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    if gt_depth.shape != rgb.shape[:2]:
        raise ValueError(f"ETH3D RGB and GT shape mismatch: {rgb.shape[:2]} vs {gt_depth.shape}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    gt_saved = save_depth_outputs(gt_depth, "im0", outdir, "gt_depth_m")
    pred_saved = save_depth_outputs(predicted, "im0", outdir, "pred_depth_m")
    metrics = depth_metrics(predicted, gt_depth, max_depth=args.max_depth)
    write_csv(Path(args.results_dir) / "eth3d_metrics.csv", [{"scene": scene.name, "image": args.rgb, **metrics}], ["scene", "image", "valid_pixels", "mae_m", "rmse_m", "absrel", "sqrel", "rmse_log", "delta_1_25", "delta_1_25_2", "delta_1_25_3"])

    valid = np.isfinite(gt_depth) & np.isfinite(predicted) & (gt_depth > 0) & (predicted > 0) & (gt_depth <= args.max_depth)
    depth_max = float(np.percentile(gt_depth[valid], 99)) if np.any(valid) else args.max_depth
    error = np.abs(predicted - gt_depth)
    error[~valid] = np.nan
    figure, axes = plt.subplots(1, 4, figsize=(18, 4.5), constrained_layout=True)
    error_max = float(np.nanpercentile(error, 99)) if np.any(np.isfinite(error)) else 1.0
    images = [rgb, colorize(gt_depth, "turbo", 0, depth_max), colorize(predicted, "turbo", 0, depth_max), colorize(error, "magma", 0, error_max)]
    for axis, image, title in zip(axes, images, ["RGB", "ETH3D GT depth (m)", "Predicted depth (m)", "Absolute error (m)"]):
        axis.imshow(image); axis.set_title(title); axis.set_xlabel("pixel x"); axis.set_ylabel("pixel y")
    figure.suptitle(f"ETH3D low-res two-view: {scene.name}/{args.rgb}")
    figure.savefig(outdir / "im0_depth_comparison.png", dpi=160); plt.close(figure)

    common = {"fx": calibration["fx"], "fy": calibration["fy"], "cx": calibration["cx"], "cy": calibration["cy"]}
    points_pred, colors_pred = depth_to_points(predicted, rgb, **common, stride=max(1, args.stride), max_depth=args.max_depth)
    points_gt, colors_gt = depth_to_points(gt_depth, rgb, **common, stride=max(1, args.stride), max_depth=args.max_depth)
    pred_ply = outdir / "im0_pred.ply"
    gt_ply = outdir / "im0_gt.ply"
    write_ascii_ply(pred_ply, points_pred, colors_pred)
    write_ascii_ply(gt_ply, points_gt, colors_gt)
    cloud_metrics = compare_point_clouds(pred_ply, gt_ply, max_points=args.pointcloud_max_points)
    write_json(outdir / "im0_pointcloud_comparison.json", {"coordinate_frame": "same ETH3D im0 rectified camera coordinates", "alignment": "none; same intrinsics and per-pixel depth coordinates", **cloud_metrics})
    write_json(outdir / "metrics.json", {"scene": str(scene), "rgb": str(rgb_path), "disparity": str(disparity_path), "calibration": calibration, "pred_depth_source": source, "gt_depth_conversion": "Z_m = fx_px * baseline_mm / (disparity_px + doffs_px) / 1000", "depth_metrics": metrics, "depth_outputs": {"gt": gt_saved, "pred": pred_saved}, "pointcloud": {"pred": str(pred_ply), "gt": str(gt_ply), **cloud_metrics}})
    print(f"[eth3d] metrics: {metrics}")
    print(f"[eth3d] point cloud comparison: {cloud_metrics}")


if __name__ == "__main__":
    main()
