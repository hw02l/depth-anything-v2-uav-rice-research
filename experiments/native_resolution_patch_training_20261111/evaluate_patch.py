from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tifffile
import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from uav_metric_research import multiscale as m
from data_patch import load_pair, rgb_unit_float_to_uint8
from inference import infer_depth, profile_tiles

SCALES = (14, 28, 56, 107, 214)


def _absolute_raw(acc) -> dict[str, Any]:
    result = acc.result()
    result.update({
        "abs_error_sum": acc.abs_error_sum,
        "squared_error_sum": acc.squared_error_sum,
        "abs_relative_sum": acc.abs_relative_sum,
        "prediction_sum": acc.pred_sum,
        "target_sum": acc.target_sum,
        "prediction_squared_sum": acc.pred_sq_sum,
        "target_squared_sum": acc.target_sq_sum,
        "prediction_target_sum": acc.cross_sum,
        "bias_m": (acc.pred_sum - acc.target_sum) / acc.count,
        "full14_abs_delta_error_sum": acc.local_abs_delta_sum,
        "full14_pair_count": acc.local_pair_count,
    })
    return result


def _delta_raw(stats) -> dict[str, Any]:
    result = stats.result()
    values = stats.legacy.total
    n = int(values[0])
    result.update({
        "abs_delta_error_sum": float(values[1]),
        "abs_gt_delta_sum": float(values[2]),
        "abs_prediction_delta_sum": float(values[3]),
        "nonzero_gt_pair_count": int(values[4]),
        "sign_match_count": int(values[5]),
        "squared_delta_error_sum": float(stats.extra[0]),
    })
    return result


def _merge_delta(target, incoming):
    target.merge(incoming)


def _boundary_records(gt: np.ndarray, pred: np.ndarray, coverage: np.ndarray,
                      seam: np.ndarray, image_index: int, method: str) -> list[dict[str, Any]]:
    if method == "P0" or seam is None:
        return []
    mask = m.valid_gt(gt)
    rows = []
    for stride in SCALES:
        stats = defaultdict(m.DeltaStats)
        for axis, y, x, yy, xx in m.sample_pairs(gt.shape, mask, image_index, stride):
            gt_delta = gt[y, x].astype(np.float64) - gt[yy, xx].astype(np.float64)
            pred_delta = pred[y, x].astype(np.float64) - pred[yy, xx].astype(np.float64)
            min_seam = np.minimum(seam[y, x], seam[yy, xx])
            categories = {
                "seam_lt14px": min_seam < 14,
                "seam_14_56px": (min_seam >= 14) & (min_seam < 56),
                "seam_56_107px": (min_seam >= 56) & (min_seam < 107),
                "seam_ge107px": min_seam >= 107,
                "both_nonoverlap": (coverage[y, x] == 1) & (coverage[yy, xx] == 1),
                "touches_overlap": (coverage[y, x] > 1) | (coverage[yy, xx] > 1),
                "both_overlap": (coverage[y, x] > 1) & (coverage[yy, xx] > 1),
            }
            for label, selected in categories.items():
                for delta_group, delta_selected in {
                    "all": selected,
                    "gt_ge20cm": selected & (np.abs(gt_delta) >= 0.2),
                }.items():
                    if delta_selected.any():
                        stats[(label, delta_group)].update(gt_delta[delta_selected], pred_delta[delta_selected])
        for (label, delta_group), value in sorted(stats.items()):
            rows.append({
                "image_index": image_index, "stride_px": stride, "method": method,
                "condition": label, "delta_group": delta_group, **_delta_raw(value),
            })
    return rows


def evaluate_one(model, row: dict[str, str], image_index: int, rgb_root: str | Path,
                 depth_root: str | Path, device: str, method: str,
                 expected_image_pairs: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rgb, gt = load_pair(row, rgb_root, depth_root)
    rgb_u8 = rgb_unit_float_to_uint8(rgb)
    inferred = infer_depth(model, rgb_u8, device, method)
    pred = inferred.prediction
    m.validate_prediction(pred, gt)

    valid = m.valid_gt(gt)
    acc = m.frozen("metrics").MetricAccumulator()
    acc.update(torch.from_numpy(pred), torch.from_numpy(np.where(valid, gt, 0).astype(np.float32)))
    absolute = _absolute_raw(acc)

    sampled, coordinates = m.multiscale(pred, gt, image_index, scales=SCALES)
    expected = [item for item in expected_image_pairs["pairs"] if item["stride"] in SCALES]
    if coordinates != expected:
        raise ValueError(f"Fixed pair-manifest coordinate hash mismatch at validation index {image_index}")
    local_rows = []
    pooled_local = {}
    for stride, groups in sampled.items():
        for label, stats in groups.items():
            raw = _delta_raw(stats)
            local_rows.append({"image_index": image_index, "stride_px": stride,
                               "group": label, **raw})
            pooled_local[(stride, label)] = stats

    boundaries = _boundary_records(gt, pred, inferred.coverage_count,
                                   inferred.seam_distance_px, image_index, method)
    profile = profile_tiles(inferred)
    image_row = {
        "image_index": image_index, "image_relpath": row["image_relpath"],
        "crop": row["crop"], "view": row["view"], "day": row["day"],
        "method": method, **absolute,
        "tile_count": profile["tile_count"],
        "inference_elapsed_seconds": profile["elapsed_seconds"],
        "tile_forward_seconds": profile["tile_seconds"],
        "coverage_min": profile["coverage_min"], "coverage_max": profile["coverage_max"],
        "nonoverlap_fraction": profile["nonoverlap_fraction"],
        "overlap_fraction": profile["overlap_fraction"],
    }
    del rgb, rgb_u8, gt, pred, inferred
    return image_row, local_rows, boundaries, {"sampled": pooled_local, "profile": profile}


def _pool_absolute(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = ("valid_pixels", "abs_error_sum", "squared_error_sum", "abs_relative_sum",
              "prediction_sum", "target_sum", "prediction_squared_sum",
              "target_squared_sum", "prediction_target_sum", "full14_abs_delta_error_sum",
              "full14_pair_count")
    sums = {key: sum(float(row[key]) for row in rows) for key in fields}
    n = int(sums["valid_pixels"])
    p, t = sums["prediction_sum"], sums["target_sum"]
    covariance = sums["prediction_target_sum"] - p * t / n
    pvar = sums["prediction_squared_sum"] - p * p / n
    tvar = sums["target_squared_sum"] - t * t / n
    return {
        **sums, "valid_pixels": n,
        "mae_m": sums["abs_error_sum"] / n,
        "rmse_m": float(np.sqrt(sums["squared_error_sum"] / n)),
        "abs_rel": sums["abs_relative_sum"] / n,
        "bias_m": (p - t) / n,
        "pearson_r": float(covariance / np.sqrt(max(pvar, 0) * max(tvar, 0))) if pvar > 0 and tvar > 0 else None,
        "full14_mae_m": sums["full14_abs_delta_error_sum"] / sums["full14_pair_count"],
    }


def _pool_local(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets = defaultdict(lambda: np.zeros(6, dtype=np.float64))
    sq = defaultdict(float)
    for row in rows:
        key = (row["stride_px"], row["group"])
        vals = buckets[key]
        for index, field in enumerate(("pair_count", "abs_delta_error_sum", "abs_gt_delta_sum",
                                       "abs_prediction_delta_sum", "nonzero_gt_pair_count",
                                       "sign_match_count")):
            vals[index] += float(row.get(field, 0) or 0)
        sq[key] += float(row.get("squared_delta_error_sum", 0) or 0)
    result = []
    for (stride, group), v in sorted(buckets.items()):
        n, err, gt_abs, pred_abs, nonzero, correct = v
        result.append({
            "stride_px": stride, "group": group, "pair_count": int(n),
            "delta_mae_m": err / n if n else None,
            "delta_rmse_m": float(np.sqrt(sq[(stride, group)] / n)) if n else None,
            "amplitude_ratio": pred_abs / gt_abs if gt_abs else None,
            "sign_match_rate": correct / nonzero if nonzero else None,
            "abs_delta_error_sum": err, "abs_gt_delta_sum": gt_abs,
            "abs_prediction_delta_sum": pred_abs,
            "nonzero_gt_pair_count": int(nonzero), "sign_match_count": int(correct),
            "squared_delta_error_sum": sq[(stride, group)],
        })
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to create empty results file: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_combo(model, rows, rgb_root, depth_root, device, method, pair_manifest,
                   output_dir: str | Path, *, group: str, epoch: int | None = None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"evaluation_{group}_{method}.json"
    if result_path.exists():
        previous = json.loads(result_path.read_text(encoding="utf-8"))
        if previous.get("group") == group and previous.get("method") == method and previous.get("complete") is True:
            return previous
        raise FileExistsError(f"Incomplete evaluation output exists and is preserved: {result_path}")
    expected = json.loads(Path(pair_manifest).read_text(encoding="utf-8"))["images"]
    if len(rows) != 128 or len(expected) != 128:
        raise ValueError("Full canonical Validation128 required")
    image_rows, local_rows, boundary_rows = [], [], []
    profiles = []
    for index, row in enumerate(rows):
        per, local, boundary, details = evaluate_one(
            model, row, index, rgb_root, depth_root, device, method, expected[index])
        image_rows.append(per); local_rows.extend(local); boundary_rows.extend(boundary)
        profiles.append({
            "image_index": index, "image_relpath": row["image_relpath"],
            "method": method, **details["profile"],
        })
        if device.startswith("cuda") and index % 8 == 7:
            torch.cuda.empty_cache()
    absolute = _pool_absolute(image_rows)
    pooled_local = _pool_local(local_rows)
    summary = {
        "complete": True, "group": group, "epoch": epoch, "method": method,
        "image_count": len(image_rows), "absolute": absolute, "sampled_local": pooled_local,
        "runtime_seconds": sum(r["inference_elapsed_seconds"] for r in image_rows),
        "runtime_tile_forward_seconds": sum(r["tile_forward_seconds"] for r in image_rows),
        "tile_count": sum(r["tile_count"] for r in image_rows),
        "tile_profiles": profiles,
        "fixed_pair_manifest_sha256": m.sha256(pair_manifest),
    }
    _write_csv(output_dir / f"per_image_absolute_{group}_{method}.csv", image_rows)
    _write_csv(output_dir / f"per_image_local_{group}_{method}.csv", local_rows)
    if boundary_rows:
        _write_csv(output_dir / f"per_image_boundary_{group}_{method}.csv", boundary_rows)
    result_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return summary

