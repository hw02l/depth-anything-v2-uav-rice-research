from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d


def _sample(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if len(points) <= max_points:
        return points
    rng = np.random.default_rng(seed)
    return points[rng.choice(len(points), max_points, replace=False)]


def _nearest_distances(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    target_cloud = o3d.geometry.PointCloud()
    target_cloud.points = o3d.utility.Vector3dVector(target.astype(np.float64))
    target_tree = o3d.geometry.KDTreeFlann(target_cloud)
    distances = np.empty(len(source), dtype=np.float64)
    for index, point in enumerate(source.astype(np.float64)):
        count, _, squared = target_tree.search_knn_vector_3d(point.tolist(), 1)
        distances[index] = np.sqrt(squared[0]) if count else np.nan
    return distances


def compare_point_clouds(pred_ply: str | Path, gt_ply: str | Path, max_points: int = 50000) -> dict[str, float | int | str]:
    """Compare two clouds already expressed in the same camera coordinate frame."""
    predicted = o3d.io.read_point_cloud(str(pred_ply))
    ground_truth = o3d.io.read_point_cloud(str(gt_ply))
    pred_points = np.asarray(predicted.points, dtype=np.float64)
    gt_points = np.asarray(ground_truth.points, dtype=np.float64)
    pred_points = pred_points[np.all(np.isfinite(pred_points), axis=1)]
    gt_points = gt_points[np.all(np.isfinite(gt_points), axis=1)]
    if len(pred_points) == 0 or len(gt_points) == 0:
        raise ValueError("Both point clouds must contain at least one point")
    pred_sample = _sample(pred_points, max_points, 0)
    gt_sample = _sample(gt_points, max_points, 1)
    pred_to_gt = _nearest_distances(pred_sample, gt_sample)
    gt_to_pred = _nearest_distances(gt_sample, pred_sample)
    valid_pred = pred_to_gt[np.isfinite(pred_to_gt)]
    valid_gt = gt_to_pred[np.isfinite(gt_to_pred)]
    symmetric = np.concatenate([valid_pred, valid_gt])
    return {
        "pred_ply": str(pred_ply),
        "gt_ply": str(gt_ply),
        "pred_points": int(len(pred_points)),
        "gt_points": int(len(gt_points)),
        "sampled_pred_points": int(len(pred_sample)),
        "sampled_gt_points": int(len(gt_sample)),
        "pred_to_gt_mean_m": float(np.mean(valid_pred)),
        "gt_to_pred_mean_m": float(np.mean(valid_gt)),
        "symmetric_chamfer_distance_m": float(np.mean(symmetric)),
        "nearest_neighbor_median_m": float(np.median(symmetric)),
        "nearest_neighbor_p95_m": float(np.percentile(symmetric, 95)),
    }
