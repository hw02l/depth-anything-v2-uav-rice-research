from __future__ import annotations

import numpy as np


METRIC_NAMES = (
    "mae_m",
    "rmse_m",
    "absrel",
    "sqrel",
    "rmse_log",
    "delta_1_25",
    "delta_1_25_2",
    "delta_1_25_3",
)


def depth_metrics(prediction: np.ndarray, ground_truth: np.ndarray, max_depth: float | None = None) -> dict[str, float | int]:
    pred = np.asarray(prediction, dtype=np.float64)
    gt = np.asarray(ground_truth, dtype=np.float64)
    valid = np.isfinite(pred) & np.isfinite(gt) & (gt > 0) & (pred > 0)
    if max_depth is not None:
        valid &= gt <= max_depth
    if not np.any(valid):
        return {"valid_pixels": 0, **{name: float("nan") for name in METRIC_NAMES}}
    pred_valid = pred[valid]
    gt_valid = gt[valid]
    error = pred_valid - gt_valid
    ratio = np.maximum(pred_valid / gt_valid, gt_valid / pred_valid)
    return {
        "valid_pixels": int(valid.sum()),
        "mae_m": float(np.mean(np.abs(error))),
        "rmse_m": float(np.sqrt(np.mean(error**2))),
        "absrel": float(np.mean(np.abs(error) / gt_valid)),
        "sqrel": float(np.mean(error**2 / gt_valid)),
        "rmse_log": float(np.sqrt(np.mean((np.log(pred_valid) - np.log(gt_valid)) ** 2))),
        "delta_1_25": float(np.mean(ratio < 1.25)),
        "delta_1_25_2": float(np.mean(ratio < 1.25**2)),
        "delta_1_25_3": float(np.mean(ratio < 1.25**3)),
    }


def aggregate_depth_metrics(rows: list[dict[str, float | int]]) -> dict[str, float | int]:
    total = sum(int(row["valid_pixels"]) for row in rows)
    if total == 0:
        return {"images": len(rows), "valid_pixels": 0, **{name: float("nan") for name in METRIC_NAMES}}
    # Reconstruct pixel-weighted metrics from per-image sufficient statistics.
    result: dict[str, float | int] = {"images": len(rows), "valid_pixels": total}
    for name in ("mae_m", "absrel", "sqrel", "delta_1_25", "delta_1_25_2", "delta_1_25_3"):
        result[name] = float(sum(float(row[name]) * int(row["valid_pixels"]) for row in rows) / total)
    for name in ("rmse_m", "rmse_log"):
        result[name] = float(np.sqrt(sum(float(row[name]) ** 2 * int(row["valid_pixels"]) for row in rows) / total))
    return result


def summary_statistics(rows: list[dict[str, float | int]]) -> list[dict[str, float | int | str]]:
    """Return mean/median/std/min/max for each per-image metric."""
    output: list[dict[str, float | int | str]] = []
    for name in ("valid_pixels", *METRIC_NAMES):
        values = np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            stats = {key: float("nan") for key in ("mean", "median", "std", "min", "max")}
        else:
            stats = {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        output.append({"metric": name, **stats})
    return output


def depth_bin_metrics(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    bins: list[tuple[float, float]],
    max_depth: float | None = None,
) -> list[dict[str, float | int | str]]:
    """Compute pixel metrics in GT-depth intervals."""
    pred = np.asarray(prediction, dtype=np.float64)
    gt = np.asarray(ground_truth, dtype=np.float64)
    base_valid = np.isfinite(pred) & np.isfinite(gt) & (gt > 0) & (pred > 0)
    if max_depth is not None:
        base_valid &= gt <= max_depth
    rows = []
    for index, (lower, upper) in enumerate(bins):
        in_bin = base_valid & (gt >= lower) & ((gt <= upper) if index == len(bins) - 1 else (gt < upper))
        result = depth_metrics(np.where(in_bin, pred, np.nan), np.where(in_bin, gt, np.nan), max_depth=None)
        rows.append({"depth_min_m": lower, "depth_max_m": upper, "depth_range_m": f"{lower:g}-{upper:g}", **result})
    return rows
