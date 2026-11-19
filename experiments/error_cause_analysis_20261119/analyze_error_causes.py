#!/usr/bin/env python3
"""Reproduce diagnostics from the saved validation sufficient statistics.

This analysis does not load checkpoints or run model inference. It reads the
corrected 2026-11-15 release outputs and prior B/E-series aggregate outputs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REL_NATIVE = Path("experiments/native_resolution_patch_training_20261111/analysis/rgb_input_fix_20261114")
REL_NATIVE_RELEASE = Path("experiments/native_resolution_patch_training_20261111/analysis/corrected_release_20261115")
REL_HUBER = Path("experiments/local_huber_20261105/analysis/final_20261107")
REL_MULTI = Path("experiments/multiscale_tile_inference_20261016")
GROUPS = ("C0", "F1", "F2")
METHODS = ("P0", "P1", "P2")
STRIDES = (14, 28, 56, 107, 214)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: fmt(v) for k, v in row.items()})


def fmt(value: Any) -> Any:
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.12g}"
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def number(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in (None, "") else 0.0


def pool_absolute(rows: list[dict[str, str]]) -> dict[str, float]:
    n = sum(number(r, "valid_pixels") for r in rows)
    if n <= 0:
        raise ValueError("absolute statistics have no valid pixels")
    sum_abs = sum(number(r, "abs_error_sum") for r in rows)
    sum_sqerr = sum(number(r, "squared_error_sum") for r in rows)
    sum_pred = sum(number(r, "prediction_sum") for r in rows)
    sum_gt = sum(number(r, "target_sum") for r in rows)
    sum_pred2 = sum(number(r, "prediction_squared_sum") for r in rows)
    sum_gt2 = sum(number(r, "target_squared_sum") for r in rows)
    sum_pred_gt = sum(number(r, "prediction_target_sum") for r in rows)
    bias = (sum_pred - sum_gt) / n
    mse = sum_sqerr / n
    residual_var = max(0.0, mse - bias * bias)
    pred_mean, gt_mean = sum_pred / n, sum_gt / n
    pred_var = max(0.0, sum_pred2 / n - pred_mean * pred_mean)
    gt_var = max(0.0, sum_gt2 / n - gt_mean * gt_mean)
    cov = sum_pred_gt / n - pred_mean * gt_mean
    slope = cov / gt_var if gt_var > 0 else math.nan
    intercept = pred_mean - slope * gt_mean if math.isfinite(slope) else math.nan
    pearson = cov / math.sqrt(pred_var * gt_var) if pred_var > 0 and gt_var > 0 else math.nan
    affine_var = max(0.0, pred_var - cov * cov / gt_var) if gt_var > 0 else math.nan
    return {
        "image_count": len(rows), "valid_pixels": int(n),
        "mae_m": sum_abs / n, "rmse_m": math.sqrt(mse), "bias_m": bias,
        "residual_variance_m2": residual_var,
        "bias_squared_m2": bias * bias,
        "bias_share_rmse2": bias * bias / mse if mse else math.nan,
        "residual_variance_share_rmse2": residual_var / mse if mse else math.nan,
        "regression_slope_pred_on_gt": slope,
        "regression_intercept_m": intercept,
        "pearson_from_sums": pearson,
        "oracle_offset_rmse_m": math.sqrt(residual_var),
        "oracle_affine_rmse_m": math.sqrt(affine_var) if math.isfinite(affine_var) else math.nan,
        "abs_rel": sum(number(r, "abs_relative_sum") for r in rows) / n,
        "image_mean_rmse_m": statistics.mean(number(r, "rmse_m") for r in rows),
        "image_mean_bias_m": statistics.mean(number(r, "bias_m") for r in rows),
        "image_mean_mae_m": statistics.mean(number(r, "mae_m") for r in rows),
    }


def pool_local(rows: list[dict[str, str]]) -> dict[str, float]:
    pair_n = sum(number(r, "pair_count") for r in rows)
    if pair_n <= 0:
        return {"pair_count": 0, "image_count": 0, "delta_mae_m": math.nan,
                "delta_rmse_m": math.nan, "amplitude_ratio": math.nan,
                "sign_match_rate": math.nan, "image_mean_delta_mae_m": math.nan,
                "image_mean_amplitude_ratio": math.nan}
    gt_sum = sum(number(r, "abs_gt_delta_sum") for r in rows)
    pred_sum = sum(number(r, "abs_prediction_delta_sum") for r in rows)
    nonzero_n = sum(number(r, "nonzero_gt_pair_count") for r in rows)
    matches = sum(number(r, "sign_match_count") for r in rows)
    valid_images = [r for r in rows if number(r, "pair_count") > 0]
    return {
        "pair_count": int(pair_n), "image_count": len(valid_images),
        "delta_mae_m": sum(number(r, "abs_delta_error_sum") for r in rows) / pair_n,
        "delta_rmse_m": math.sqrt(sum(number(r, "squared_delta_error_sum") for r in rows) / pair_n),
        "amplitude_ratio": pred_sum / gt_sum if gt_sum else math.nan,
        "sign_match_rate": matches / nonzero_n if nonzero_n else math.nan,
        "image_mean_delta_mae_m": statistics.mean(number(r, "delta_mae_m") for r in valid_images),
        "image_mean_amplitude_ratio": statistics.mean(number(r, "amplitude_ratio") for r in valid_images),
    }


def validate_image_alignment(all_rows: dict[tuple[str, str], list[dict[str, str]]]) -> list[str]:
    reference = None
    messages: list[str] = []
    for key, rows in sorted(all_rows.items()):
        ids = [int(number(r, "image_index")) for r in rows]
        paths = [r["image_relpath"] for r in rows]
        if len(rows) != 128 or len(set(ids)) != 128:
            raise AssertionError(f"{key}: expected 128 unique image rows; got {len(rows)}")
        signature = list(zip(ids, paths))
        if reference is None:
            reference = signature
        elif signature != reference:
            raise AssertionError(f"image ordering differs at {key}")
    messages.append("nine conditions: 128 unique image IDs and identical ordered image_relpath verified")
    return messages


def paired_bootstrap(
    a_rows: list[dict[str, str]], b_rows: list[dict[str, str]],
    metric: str, estimand: str, reps: int, seed: int,
) -> dict[str, float]:
    """Paired resampling of image IDs; a minus b. Pooled estimands use sufficient sums."""
    amap = {int(number(r, "image_index")): r for r in a_rows}
    bmap = {int(number(r, "image_index")): r for r in b_rows}
    ids = sorted(set(amap) & set(bmap))
    if metric.startswith("local_"):
        ids = [i for i in ids if number(amap[i], "pair_count") > 0 and number(bmap[i], "pair_count") > 0]
    if not ids:
        raise ValueError("no paired image IDs for bootstrap")
    aa, bb = [amap[i] for i in ids], [bmap[i] for i in ids]
    rng = np.random.default_rng(seed)
    # Match the upstream corrected-release bootstrap draw contract exactly.
    samples = rng.integers(0, len(ids), size=(reps, len(ids)), dtype=np.int16)

    def estimate(rows: list[dict[str, str]], idx: np.ndarray) -> np.ndarray:
        if metric == "absolute_rmse_m":
            nums = np.array([number(r, "squared_error_sum") for r in rows])
            dens = np.array([number(r, "valid_pixels") for r in rows])
            if estimand == "pooled": return np.sqrt(nums[idx].sum(axis=1) / dens[idx].sum(axis=1))
            vals = np.array([number(r, "rmse_m") for r in rows])
        elif metric == "absolute_mae_m":
            nums = np.array([number(r, "abs_error_sum") for r in rows])
            dens = np.array([number(r, "valid_pixels") for r in rows])
            if estimand == "pooled": return nums[idx].sum(axis=1) / dens[idx].sum(axis=1)
            vals = np.array([number(r, "mae_m") for r in rows])
        elif metric == "absolute_bias_m":
            nums = np.array([number(r, "prediction_sum") - number(r, "target_sum") for r in rows])
            dens = np.array([number(r, "valid_pixels") for r in rows])
            if estimand == "pooled": return nums[idx].sum(axis=1) / dens[idx].sum(axis=1)
            vals = np.array([number(r, "bias_m") for r in rows])
        elif metric in ("local_gt20_mae_m", "local_all14_mae_m"):
            nums = np.array([number(r, "abs_delta_error_sum") for r in rows])
            dens = np.array([number(r, "pair_count") for r in rows])
            if estimand == "pooled": return nums[idx].sum(axis=1) / dens[idx].sum(axis=1)
            vals = np.array([number(r, "delta_mae_m") for r in rows])
        elif metric == "local_gt20_amplitude_ratio":
            num = np.array([number(r, "abs_prediction_delta_sum") for r in rows])
            den = np.array([number(r, "abs_gt_delta_sum") for r in rows])
            if estimand == "pooled": return num[idx].sum(axis=1) / den[idx].sum(axis=1)
            vals = np.array([number(r, "amplitude_ratio") for r in rows])
        else:
            raise ValueError(f"unknown bootstrap metric: {metric}")
        return vals[idx].mean(axis=1)

    obs_idx = np.arange(len(ids), dtype=int)[None, :]
    observed_a = float(estimate(aa, obs_idx)[0])
    observed_b = float(estimate(bb, obs_idx)[0])
    diffs = estimate(aa, samples) - estimate(bb, samples)
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {"image_count": len(ids), "estimate_a": observed_a, "estimate_b": observed_b,
            "difference_a_minus_b": observed_a - observed_b,
            "percent_change_vs_b": 100.0 * (observed_a - observed_b) / observed_b if observed_b else math.nan,
            "ci95_low_difference": float(lo), "ci95_high_difference": float(hi),
            "prob_a_less_than_b": float(np.mean(diffs < 0.0)),
            "bootstrap_reps": reps, "seed": seed}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)

    native = repo / REL_NATIVE
    native_config_path = repo / "experiments/native_resolution_patch_training_20261111/config.json"
    release = repo / REL_NATIVE_RELEASE
    multi = repo / REL_MULTI
    condition_summary = read_csv(native / "condition_summary.csv")
    expected_metrics = read_csv(release / "metrics_9_conditions.csv")
    abs_files: dict[tuple[str, str], list[dict[str, str]]] = {}
    local_files: dict[tuple[str, str, int, str], list[dict[str, str]]] = {}
    boundary_files: dict[tuple[str, str], list[dict[str, str]]] = {}
    local_per_image_rows: list[dict[str, Any]] = []
    for group in GROUPS:
        for method in METHODS:
            stem = f"{group}_{method}"
            folder = native / "per_condition" / stem
            abs_files[(group, method)] = read_csv(folder / f"per_image_absolute_{stem}.csv")
            local_rows = read_csv(folder / f"per_image_local_{stem}.csv")
            local_per_image_rows.extend(
                {"training_group": group, "inference_method": method, **row}
                for row in local_rows if row["group"] in ("all", "gt_ge20cm")
            )
            for stride in STRIDES:
                for delta_group in ("all", "gt_ge20cm"):
                    local_files[(group, method, stride, delta_group)] = [r for r in local_rows if int(number(r, "stride_px")) == stride and r["group"] == delta_group]
            if method in ("P1", "P2"):
                boundary_files[(group, method)] = read_csv(folder / f"per_image_boundary_{stem}.csv")

    audits = validate_image_alignment(abs_files)
    expected_local_rows = len(GROUPS) * len(METHODS) * 128 * len(STRIDES) * 2
    local_keys = {(r["training_group"], r["inference_method"], int(number(r, "image_index")),
                   int(number(r, "stride_px")), r["group"]) for r in local_per_image_rows}
    if len(local_per_image_rows) != expected_local_rows or len(local_keys) != expected_local_rows:
        raise AssertionError(f"expected {expected_local_rows} unique per-image local rows, found {len(local_keys)}")
    write_csv(out / "per_image_local_diagnostics.csv", local_per_image_rows)
    audits.append(f"per-image local sufficient statistics exported: {expected_local_rows} unique condition/image/distance/group rows")
    expected = {(r["group"], r["method"]): r for r in expected_metrics}
    condition_rows: list[dict[str, Any]] = []
    per_image_rows: list[dict[str, Any]] = []
    for (group, method), rows in sorted(abs_files.items()):
        pooled = pool_absolute(rows)
        ref = expected[(group, method)]
        checks = {k: abs(pooled[k] - float(ref[rv])) for k, rv in (("mae_m", "mae_m"), ("rmse_m", "rmse_m"), ("abs_rel", "abs_rel"), ("bias_m", "bias_m"), ("pearson_from_sums", "pearson_r"))}
        if max(checks.values()) > 1e-8:
            raise AssertionError(f"summary reconciliation failed for {group}-{method}: {checks}")
        audits.append(f"{group}-{method}: pooled stats reconcile to corrected release; max_abs_diff={max(checks.values()):.3g}")
        condition_rows.append({"training_group": group, "inference_method": method, **pooled,
                               "release_rmse_reference_m": float(ref["rmse_m"]),
                               "reconciliation_max_abs_difference": max(checks.values())})
        for r in rows:
            n = number(r, "valid_pixels")
            row_pool = pool_absolute([r])
            per_image_rows.append({"training_group": group, "inference_method": method,
                                   "image_index": int(number(r, "image_index")), "image_relpath": r["image_relpath"],
                                   "crop": r["crop"], "view": r["view"], "valid_pixels": int(n),
                                   "mae_m": number(r, "mae_m"), "rmse_m": number(r, "rmse_m"),
                                   "bias_m": number(r, "bias_m"),
                                   "residual_variance_m2": row_pool["residual_variance_m2"],
                                   "regression_slope_pred_on_gt": row_pool["regression_slope_pred_on_gt"],
                                   "regression_intercept_m": row_pool["regression_intercept_m"],
                                   "oracle_offset_rmse_m": row_pool["oracle_offset_rmse_m"],
                                   "oracle_affine_rmse_m": row_pool["oracle_affine_rmse_m"]})

    local_rows_out: list[dict[str, Any]] = []
    summary_idx = {}
    for r in condition_summary:
        summary_idx[(r["task"], r["training_group"], r["method"], r["scope"], r["crop"], r["view"], r["condition"], r["stride_px"], r["delta_group"], r["metric"], r["estimator"])] = r
    for (group, method, stride, delta_group), rows in sorted(local_files.items()):
        pooled = pool_local(rows)
        if pooled["pair_count"] == 0:
            continue
        for metric, key in (("delta_mae", "delta_mae_m"), ("delta_rmse", "delta_rmse_m"), ("amplitude_ratio", "amplitude_ratio"), ("sign_match_rate", "sign_match_rate")):
            sr = summary_idx.get(("local", group, method, "all", "", "", delta_group, str(stride), "", metric, "pooled"))
            if sr is None:
                raise AssertionError(f"missing summary key for {group}-{method}/{stride}/{delta_group}/{metric}")
            if abs(pooled[key] - float(sr["value"])) > 1e-8:
                raise AssertionError(f"local reconciliation failed {group}-{method}/{stride}/{delta_group}/{metric}: {pooled[key]} vs {sr['value']}")
        local_rows_out.append({"training_group": group, "inference_method": method,
                               "stride_px": stride, "delta_group": delta_group, **pooled})
    audits.append("local per-image sufficient statistics reconcile to condition_summary for all distances/groups")

    subgroup_rows = [r for r in condition_summary if r["scope"] in ("crop", "view", "crop_view") and
                     ((r["task"] == "absolute" and r["metric"] in ("mae", "rmse", "bias", "pearson") and r["estimator"] == "pooled") or
                      (r["task"] == "local" and r["stride_px"] == "14" and r["condition"] == "gt_ge20cm" and r["metric"] in ("delta_mae", "amplitude_ratio", "sign_match_rate") and r["estimator"] == "pooled"))]
    write_csv(out / "absolute_condition_diagnostics.csv", condition_rows)
    write_csv(out / "per_image_absolute_diagnostics.csv", per_image_rows)
    write_csv(out / "local_distance_diagnostics.csv", local_rows_out)
    write_csv(out / "crop_view_stratified_diagnostics.csv", subgroup_rows)

    local_strata_rows = [r for r in condition_summary if r["task"] == "local" and r["stride_px"] == "14" and
                         r["scope"] in ("all", "crop", "view", "crop_view") and
                         r["metric"] in ("delta_mae", "delta_rmse", "amplitude_ratio", "sign_match_rate") and
                         r["estimator"] == "pooled"]
    write_csv(out / "local_stratified_14px_diagnostics.csv", local_strata_rows)

    inventory = read_csv(multi / "image_inventory.csv")
    xy14 = [number(r, "input14_x_px") for r in inventory]
    yy14 = [number(r, "input14_y_px") for r in inventory]
    training_config = json.loads(native_config_path.read_text(encoding="utf-8"))
    groups_cfg = training_config["groups"]
    inference_cfg = training_config["inference"]
    if (groups_cfg["C0"]["global_short_side_px"], groups_cfg["C0"]["crop_px"],
        groups_cfg["F1"]["original_crop_px"], groups_cfg["F1"]["input_px"],
        groups_cfg["F2"]["original_crop_px"], groups_cfg["F2"]["input_px"]) != (518, 518, 1036, 518, 518, 518):
        raise AssertionError("training crop configuration differs from documented scale assumptions")
    if (inference_cfg["P0"]["short_side_px"], inference_cfg["P1"]["tile_original_px"],
        inference_cfg["P1"]["input_px"], inference_cfg["P2"]["tile_original_px"],
        inference_cfg["P2"]["input_px"]) != (518, 1036, 518, 518, 518):
        raise AssertionError("inference scale configuration differs from documented scale assumptions")
    scale_rows = [
        {"stage": "training", "condition": "C0", "model_input_px_for_original_14px": statistics.mean(yy14),
         "interpretation": "full image resized to short side 518; x/y per-image values are in image_inventory.csv"},
        {"stage": "training", "condition": "F1", "model_input_px_for_original_14px": 7.0,
         "interpretation": "1036px crop resized to 518px"},
        {"stage": "training", "condition": "F2", "model_input_px_for_original_14px": 14.0,
         "interpretation": "native 518px crop fed at the same size"},
        {"stage": "inference", "condition": "P0", "model_input_px_for_original_14px": statistics.mean(yy14),
         "interpretation": "full image short-side resize to 518px; average of image_inventory.csv y scale"},
        {"stage": "inference", "condition": "P1", "model_input_px_for_original_14px": 7.0,
         "interpretation": "1036px tile resized to 518px"},
        {"stage": "inference", "condition": "P2", "model_input_px_for_original_14px": 14.0,
         "interpretation": "518px tile fed without scale change"},
    ]
    for r in scale_rows:
        r["input_px_min"] = min(yy14) if r["condition"] in ("C0", "P0") else r["model_input_px_for_original_14px"]
        r["input_px_max"] = max(yy14) if r["condition"] in ("C0", "P0") else r["model_input_px_for_original_14px"]
        r["input_px_mean_x"] = statistics.mean(xy14) if r["condition"] in ("C0", "P0") else r["model_input_px_for_original_14px"]
    write_csv(out / "input_scale_mapping.csv", scale_rows)

    boundary_summary_rows = [r for r in condition_summary if r["task"] == "boundary" and r["scope"] == "all" and
                             r["stride_px"] == "14" and r["delta_group"] == "gt_ge20cm" and
                             r["metric"] in ("delta_mae", "delta_rmse", "amplitude_ratio", "sign_match_rate")]
    write_csv(out / "tile_boundary_diagnostics.csv", boundary_summary_rows)

    comparisons = [
        ("F2_P2_vs_C0_P2", ("F2", "P2"), ("C0", "P2")),
        ("F1_P1_vs_C0_P1", ("F1", "P1"), ("C0", "P1")),
        ("F2_P0_vs_C0_P0", ("F2", "P0"), ("C0", "P0")),
        ("F1_P0_vs_C0_P0", ("F1", "P0"), ("C0", "P0")),
    ]
    for g in GROUPS:
        for m in ("P1", "P2"):
            comparisons.append((f"{g}_{m}_vs_{g}_P0", (g, m), (g, "P0")))
    for m, g in (("P1", "F1"), ("P2", "F2")):
        comparisons.append((f"{g}_{m}_vs_C0_{m}", (g, m), ("C0", m)))
    bootstrap_rows: list[dict[str, Any]] = []
    for label, akey, bkey in comparisons:
        for metric in ("absolute_rmse_m", "absolute_mae_m", "absolute_bias_m"):
            for estimand in ("pooled", "image_mean"):
                res = paired_bootstrap(abs_files[akey], abs_files[bkey], metric, estimand, args.bootstrap_reps, args.seed)
                bootstrap_rows.append({"comparison": label, "condition_a": f"{akey[0]}-{akey[1]}",
                                       "condition_b": f"{bkey[0]}-{bkey[1]}", "metric": metric,
                                       "estimand": estimand, **res})
        for delta_group, metric in (("gt_ge20cm", "local_gt20_mae_m"), ("gt_ge20cm", "local_gt20_amplitude_ratio"), ("all", "local_all14_mae_m")):
            la = local_files[(akey[0], akey[1], 14, delta_group)]
            lb = local_files[(bkey[0], bkey[1], 14, delta_group)]
            for estimand in ("pooled", "image_mean"):
                res = paired_bootstrap(la, lb, metric, estimand, args.bootstrap_reps, args.seed)
                bootstrap_rows.append({"comparison": label, "condition_a": f"{akey[0]}-{akey[1]}",
                                       "condition_b": f"{bkey[0]}-{bkey[1]}", "metric": metric,
                                       "delta_group": delta_group, "estimand": estimand, **res})
    write_csv(out / "paired_image_bootstrap.csv", bootstrap_rows)
    audits.append(f"paired image bootstrap computed with shared image IDs, {args.bootstrap_reps} resamples and seed {args.seed}")

    # Compare matching prespecified paired-bootstrap rows with the corrected
    # release using the same RNG draw dtype, seed, estimands, and percentile CI.
    reference_bootstrap = read_csv(release / "key_paired_comparisons.csv")
    reference_map = {}
    for row in reference_bootstrap:
        key = (row["task"], row["candidate"], row["method"], row["baseline"],
               row["scope"], row["condition"], row["stride_px"], row["metric"], row["estimator"])
        reference_map[key] = row
    metric_map = {
        "absolute_rmse_m": ("absolute", "rmse", "", ""),
        "absolute_mae_m": ("absolute", "mae", "", ""),
        "absolute_bias_m": ("absolute", "bias", "", ""),
        "local_gt20_mae_m": ("local", "delta_mae", "gt_ge20cm", "14"),
        "local_gt20_amplitude_ratio": ("local", "amplitude_ratio", "gt_ge20cm", "14"),
        "local_all14_mae_m": ("local", "delta_mae", "all", "14"),
    }
    reconciled_bootstrap = 0
    for row in bootstrap_rows:
        task, metric_name, condition, stride = metric_map[row["metric"]]
        group_a, method_a = row["condition_a"].split("-")
        group_b, method_b = row["condition_b"].split("-")
        if method_a != method_b:
            continue  # cross-method comparisons are new to this cause analysis
        key = (task, group_a, method_a, group_b, "all", condition, stride,
               metric_name, row["estimand"])
        ref = reference_map.get(key)
        if ref is None:
            continue
        checks = (
            (float(row["estimate_a"]), float(ref["candidate_value"])),
            (float(row["estimate_b"]), float(ref["baseline_value"])),
            (float(row["difference_a_minus_b"]), float(ref["candidate_minus_baseline"])),
            (float(row["ci95_low_difference"]), float(ref["ci95_low"])),
            (float(row["ci95_high_difference"]), float(ref["ci95_high"])),
        )
        if max(abs(a - b) for a, b in checks) > 1e-8:
            raise AssertionError(f"paired bootstrap differs from corrected release: {key}")
        if int(row["bootstrap_reps"]) != int(ref["repeats"]) or int(row["seed"]) != int(ref["seed"]):
            raise AssertionError(f"paired bootstrap settings differ from corrected release: {key}")
        reconciled_bootstrap += 1
    if reconciled_bootstrap == 0:
        raise AssertionError("no paired bootstrap rows matched the corrected release")
    audits.append(f"{reconciled_bootstrap} paired bootstrap rows reconcile to corrected release within 1e-8")

    # Preserve prior B and E results as separately labeled context.
    b_metrics = read_csv(multi / "per_image_metrics.csv")
    b_local = read_csv(multi / "multiscale_metrics.csv")
    b_rows_out: list[dict[str, Any]] = []
    keys = sorted({(r["epoch"], r["method"]) for r in b_metrics})
    for epoch, method in keys:
        rs = [r for r in b_metrics if r["epoch"] == epoch and r["method"] == method]
        n = sum(number(r, "valid_pixels") for r in rs)
        if n <= 0: continue
        pooled_rmse = math.sqrt(sum(number(r, "rmse_m") ** 2 * number(r, "valid_pixels") for r in rs) / n)
        pooled_mae = sum(number(r, "mae_m") * number(r, "valid_pixels") for r in rs) / n
        pooled_absrel = sum(number(r, "abs_rel") * number(r, "valid_pixels") for r in rs) / n
        b_rows_out.append({"source_series": "B-series prior inference-method comparison",
                           "epoch": epoch, "method": method, "image_count": len(rs), "valid_pixels": int(n),
                           "mae_pooled_approx_m": pooled_mae, "rmse_pooled_from_per_image_m": pooled_rmse,
                           "abs_rel_pooled_approx": pooled_absrel,
                           "image_mean_rmse_m": statistics.mean(number(r, "rmse_m") for r in rs),
                           "image_mean_large14_mae_m": statistics.mean(number(r, "large14_mae_m") for r in rs),
                           "image_mean_large14_amplitude_ratio": statistics.mean(number(r, "large14_amplitude_ratio") for r in rs),
                           "image_mean_large14_sign_rate": statistics.mean(number(r, "large14_sign_rate") for r in rs)})
    write_csv(out / "prior_B_series_condition_summary.csv", b_rows_out)
    write_csv(out / "prior_B_series_multiscale_local.csv", [r for r in b_local if r["scope"] == "all" and r["group"] in ("all", "gt_ge20cm")])

    huber = repo / REL_HUBER
    e_rows = []
    for group in ("E0", "E1", "E2", "E3"):
        data = json.loads((huber / "inputs" / f"{group}_summary.json").read_text(encoding="utf-8"))
        absd = data["absolute"]
        local = data["gt_ge20cm_14px"]
        e_rows.append({"source_series": "E-series prior training comparison", "group": group,
                       "epoch": data["epoch"], "checkpoint_sha256": data["checkpoint_sha256"],
                       "image_count": data["image_count"], **absd,
                       "gt_ge20cm_pair_count": local["pair_count"],
                       "gt_ge20cm_local_mae_m": local["local_mae_m"],
                       "gt_ge20cm_amplitude_ratio": local["amplitude_ratio"],
                       "gt_ge20cm_sign_match_rate": local["sign_match_rate"]})
    write_csv(out / "prior_E_series_summary.csv", e_rows)

    # Figures from exact derived tables; figures are static scientific outputs.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    labels = [f"{r['training_group']}–{r['inference_method']}" for r in condition_rows]
    x = np.arange(len(labels))
    bias2 = np.array([r["bias_squared_m2"] for r in condition_rows])
    resvar = np.array([r["residual_variance_m2"] for r in condition_rows])
    fig, ax = plt.subplots(figsize=(10, 4.6), constrained_layout=True)
    ax.bar(x, bias2, color="#d77c36", label="Bias²")
    ax.bar(x, resvar, bottom=bias2, color="#4f78a8", label="Var(Pred−GT)")
    ax.set_xticks(x, labels, rotation=30, ha="right"); ax.set_ylabel("RMSE² contribution (m²)")
    ax.set_title("Absolute error decomposition: RMSE² = Bias² + residual variance")
    ax.legend(frameon=False, ncol=2); fig.savefig(out / "rmse_bias_decomposition.png", dpi=180); fig.savefig(out / "rmse_bias_decomposition.svg"); plt.close(fig)

    rmse_values = np.array([r["rmse_m"] for r in condition_rows])
    fig, ax = plt.subplots(figsize=(10, 4.6), constrained_layout=True)
    colors = ["#4f78a8" if r["inference_method"] == "P0" else
              "#d77c36" if r["inference_method"] == "P1" else "#5a9b72"
              for r in condition_rows]
    ax.bar(x, rmse_values, color=colors)
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_ylabel("Pooled absolute RMSE (m)")
    ax.set_title("Corrected validation absolute RMSE: all nine conditions")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(out / "absolute_rmse_9_conditions.png", dpi=180)
    fig.savefig(out / "absolute_rmse_9_conditions.svg")
    plt.close(fig)

    local_primary = [r for r in local_rows_out if r["stride_px"] == 14 and
                     r["delta_group"] == "gt_ge20cm"]
    local_primary.sort(key=lambda r: (GROUPS.index(r["training_group"]), METHODS.index(r["inference_method"])))
    local_labels = [f"{r['training_group']}–{r['inference_method']}" for r in local_primary]
    lx = np.arange(len(local_primary))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
    axes[0].bar(lx, [r["delta_mae_m"] for r in local_primary], color="#4f78a8")
    axes[0].set_ylabel("MAE (m)"); axes[0].set_title("14px local Δ MAE")
    axes[1].bar(lx, [r["amplitude_ratio"] for r in local_primary], color="#d77c36")
    axes[1].axhline(1.0, color="#333333", linewidth=1, linestyle="--")
    axes[1].set_ylabel("Predicted / GT |Δ| sum"); axes[1].set_title("Amplitude ratio")
    axes[2].bar(lx, [r["sign_match_rate"] for r in local_primary], color="#5a9b72")
    axes[2].axhline(0.5, color="#333333", linewidth=1, linestyle=":")
    axes[2].set_ylim(0, 1); axes[2].set_ylabel("Rate"); axes[2].set_title("Sign agreement")
    for ax in axes:
        ax.set_xticks(lx, local_labels, rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("14px local depth differences (|ΔGT| ≥ 0.20 m)")
    fig.savefig(out / "local_14px_summaries.png", dpi=180)
    fig.savefig(out / "local_14px_summaries.svg")
    plt.close(fig)

    slopes = np.array([r["regression_slope_pred_on_gt"] for r in condition_rows])
    intercepts = np.array([r["regression_intercept_m"] for r in condition_rows])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    axes[0].bar(x, slopes, color="#4f78a8"); axes[0].axhline(1, color="#333333", linewidth=1, linestyle="--")
    axes[0].set_xticks(x, labels, rotation=30, ha="right"); axes[0].set_ylabel("Slope (Pred on GT)"); axes[0].set_title("Metric scale")
    axes[1].bar(x, intercepts, color="#d77c36"); axes[1].axhline(0, color="#333333", linewidth=1, linestyle="--")
    axes[1].set_xticks(x, labels, rotation=30, ha="right"); axes[1].set_ylabel("Intercept (m)"); axes[1].set_title("Metric offset")
    fig.savefig(out / "depth_regression_diagnostics.png", dpi=180); fig.savefig(out / "depth_regression_diagnostics.svg"); plt.close(fig)

    for delta_group in ("all", "gt_ge20cm"):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
        for group, marker in zip(GROUPS, ("o", "s", "^") ):
            rs = [r for r in local_rows_out if r["training_group"] == group and r["delta_group"] == delta_group]
            rs.sort(key=lambda r: r["stride_px"])
            axes[0].plot([r["stride_px"] for r in rs], [r["delta_mae_m"] for r in rs], marker=marker, label=group)
            axes[1].plot([r["stride_px"] for r in rs], [r["amplitude_ratio"] for r in rs], marker=marker, label=group)
        axes[0].set_xscale("log"); axes[0].set_xticks(STRIDES, STRIDES); axes[0].set_xlabel("Original-image pair distance (px)"); axes[0].set_ylabel("Local Δdepth MAE (m)")
        axes[1].set_xscale("log"); axes[1].set_xticks(STRIDES, STRIDES); axes[1].axhline(1, color="#333333", linewidth=1, linestyle="--")
        axes[1].set_xlabel("Original-image pair distance (px)"); axes[1].set_ylabel("Pooled amplitude ratio")
        for ax in axes: ax.grid(axis="y", alpha=0.25)
        axes[0].legend(frameon=False); fig.suptitle(f"Local depth differences: {delta_group.replace('_', ' ')}")
        name = "local_distance_all" if delta_group == "all" else "local_distance_gt20cm"
        fig.savefig(out / f"{name}.png", dpi=180); fig.savefig(out / f"{name}.svg"); plt.close(fig)

    # View, crop, GT-depth and tile-seam diagnostics. These are descriptive
    # strata, not randomized causal contrasts.
    view_rows = [r for r in condition_summary if r["task"] == "absolute" and r["scope"] == "view" and
                 r["metric"] == "rmse" and r["estimator"] == "pooled"]
    view_keys = {(r["training_group"], r["method"], r["view"]): float(r["value"]) for r in view_rows}
    labels_view = [f"{g}-{m}" for g in GROUPS for m in METHODS]
    positions = np.arange(len(labels_view)); width = 0.36
    fig, ax = plt.subplots(figsize=(11, 4.8), constrained_layout=True)
    for j, view in enumerate(("nadir", "oblique")):
        vals = [view_keys.get((g, m, view), np.nan) for g in GROUPS for m in METHODS]
        ax.bar(positions + (j - 0.5) * width, vals, width=width, label=view.title())
    ax.set_yscale("log"); ax.set_xticks(positions, labels_view, rotation=30, ha="right")
    ax.set_ylabel("Pooled absolute RMSE (m; log scale)"); ax.set_title("Absolute RMSE by view (descriptive strata)")
    ax.legend(frameon=False); ax.grid(axis="y", alpha=0.25, which="both")
    fig.savefig(out / "absolute_rmse_by_view.png", dpi=180); fig.savefig(out / "absolute_rmse_by_view.svg"); plt.close(fig)

    local_key_conditions = (("C0", "P2"), ("F1", "P1"), ("F2", "P2"))
    crop_names = ("corn", "oat", "soybean", "wheat")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    for (group, method), marker in zip(local_key_conditions, ("o", "s", "^")):
        crop_vals, view_vals = [], []
        for crop in crop_names:
            rr = next((r for r in subgroup_rows if r["task"] == "local" and r["training_group"] == group and r["method"] == method and
                       r["scope"] == "crop" and r["crop"] == crop and r["condition"] == "gt_ge20cm" and r["stride_px"] == "14" and r["metric"] == "delta_mae"), None)
            crop_vals.append(float(rr["value"]) if rr else np.nan)
        for view in ("nadir", "oblique"):
            rr = next((r for r in subgroup_rows if r["task"] == "local" and r["training_group"] == group and r["method"] == method and
                       r["scope"] == "view" and r["view"] == view and r["condition"] == "gt_ge20cm" and r["stride_px"] == "14" and r["metric"] == "delta_mae"), None)
            view_vals.append(float(rr["value"]) if rr else np.nan)
        axes[0].plot(crop_names, crop_vals, marker=marker, label=f"{group}-{method}")
        axes[1].plot(("Nadir", "Oblique"), view_vals, marker=marker, label=f"{group}-{method}")
    axes[0].set_ylabel("GT≥20 cm, 14px Δdepth MAE (m)"); axes[0].set_title("By crop")
    axes[1].set_title("By view"); axes[1].set_ylabel("GT≥20 cm, 14px Δdepth MAE (m)")
    for ax in axes: ax.grid(axis="y", alpha=0.25); ax.legend(frameon=False)
    fig.savefig(out / "primary_local_mae_by_crop_view.png", dpi=180); fig.savefig(out / "primary_local_mae_by_crop_view.svg"); plt.close(fig)

    depth_bins = ("depth_0_15m", "depth_15_20m", "depth_20_30m", "depth_30_65m")
    fig, ax = plt.subplots(figsize=(8, 4.6), constrained_layout=True)
    for (group, method), marker in zip((("C0", "P0"), ("F1", "P1"), ("F2", "P2")), ("o", "s", "^")):
        vals = []
        for stratum in depth_bins:
            rr = next((r for r in local_strata_rows if r["training_group"] == group and r["method"] == method and
                       r["scope"] == "all" and r["condition"] == stratum and r["metric"] == "delta_mae"), None)
            vals.append(float(rr["value"]) if rr else np.nan)
        ax.plot(("0–15", "15–20", "20–30", "30–65"), vals, marker=marker, label=f"{group}-{method}")
    ax.set_xlabel("GT endpoint depth band (m)"); ax.set_ylabel("14px local Δdepth MAE (m)")
    ax.set_title("14px local error by GT depth band (all sampled pairs)"); ax.grid(axis="y", alpha=0.25); ax.legend(frameon=False)
    fig.savefig(out / "local_error_by_gt_depth.png", dpi=180); fig.savefig(out / "local_error_by_gt_depth.svg"); plt.close(fig)

    delta_bins = ("gt_delta_0_0.05m", "gt_delta_0.05_0.1m", "gt_delta_0.1_0.2m", "gt_delta_0.2_0.5m", "gt_delta_0.5_infm")
    fig, ax = plt.subplots(figsize=(8, 4.6), constrained_layout=True)
    for (group, method), marker in zip((("C0", "P0"), ("F1", "P1"), ("F2", "P2")), ("o", "s", "^")):
        vals = []
        for stratum in delta_bins:
            rr = next((r for r in local_strata_rows if r["training_group"] == group and r["method"] == method and
                       r["scope"] == "all" and r["condition"] == stratum and r["metric"] == "delta_mae"), None)
            vals.append(float(rr["value"]) if rr else np.nan)
        ax.plot(("0–0.05", "0.05–0.1", "0.1–0.2", "0.2–0.5", "≥0.5"), vals, marker=marker, label=f"{group}-{method}")
    ax.set_xlabel("Absolute GT depth difference between endpoints (m)"); ax.set_ylabel("14px Δdepth MAE (m)")
    ax.set_title("14px local error by GT difference bin"); ax.grid(axis="y", alpha=0.25); ax.legend(frameon=False)
    fig.savefig(out / "local_error_by_gt_delta.png", dpi=180); fig.savefig(out / "local_error_by_gt_delta.svg"); plt.close(fig)

    boundary_plot = [r for r in boundary_summary_rows if r["metric"] == "delta_mae" and r["estimator"] == "pooled"]
    boundary_categories = ("both_nonoverlap", "both_overlap", "seam_lt14px", "seam_14_56px", "seam_56_107px", "seam_ge107px")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    for ax, method in zip(axes, ("P1", "P2")):
        for group, marker in zip(GROUPS, ("o", "s", "^")):
            vals = []
            for cat in boundary_categories:
                rr = next((r for r in boundary_plot if r["training_group"] == group and r["method"] == method and r["condition"] == cat), None)
                vals.append(float(rr["value"]) if rr else np.nan)
            ax.plot(boundary_categories, vals, marker=marker, label=group)
        ax.set_title(f"{method}: P1/P2 tile seam strata")
        ax.set_ylabel("GT≥20 cm, 14px pair Δdepth MAE (m)")
        ax.tick_params(axis="x", labelrotation=35); ax.grid(axis="y", alpha=0.25); ax.legend(frameon=False)
    fig.savefig(out / "tile_seam_local_mae.png", dpi=180); fig.savefig(out / "tile_seam_local_mae.svg"); plt.close(fig)

    # Simple audit manifest for all primary source artifacts.
    input_paths = [
        native_config_path, native / "condition_summary.csv", release / "metrics_9_conditions.csv",
        multi / "image_inventory.csv", multi / "per_image_metrics.csv", multi / "multiscale_metrics.csv",
        release / "key_paired_comparisons.csv",
        huber / "inputs" / "final_summary.json",
    ] + [native / "per_condition" / f"{g}_{m}" / f"per_image_absolute_{g}_{m}.csv" for g in GROUPS for m in METHODS]
    input_paths += [native / "per_condition" / f"{g}_{m}" / f"per_image_local_{g}_{m}.csv" for g in GROUPS for m in METHODS]
    input_paths += [huber / "inputs" / f"{g}_summary.json" for g in ("E0", "E1", "E2", "E3")]
    evidence_dir = out.parent / "source_evidence"
    evidence_paths = [evidence_dir / "fresh_reproduction_audit.json", evidence_dir / "checkpoint_hash_audit_verified.csv"]
    hashed = []
    for p in input_paths + evidence_paths:
        if p.exists():
            try:
                relpath = p.relative_to(repo).as_posix()
            except ValueError:
                relpath = "experiments/error_cause_analysis_20261119/source_evidence/" + p.name
            hashed.append({"path": relpath, "sha256": sha256_file(p), "bytes": p.stat().st_size})
        else:
            try:
                relpath = p.relative_to(repo).as_posix()
            except ValueError:
                relpath = "experiments/error_cause_analysis_20261119/source_evidence/" + p.name
            audits.append(f"source evidence missing and not used: {relpath}")
    reproduction_audit = json.loads((evidence_dir / "fresh_reproduction_audit.json").read_text(encoding="utf-8"))
    hash_rows = read_csv(evidence_dir / "checkpoint_hash_audit_verified.csv")
    verified_hashes = {r["group"]: r["verified_checkpoint_sha256"] for r in hash_rows}
    if verified_hashes != reproduction_audit["fresh_run"]["checkpoint_sha256_by_group"]:
        raise AssertionError("verified checkpoint hashes do not match the fresh reproduction audit")
    run_meta = {
        "analysis_id": "error_cause_analysis_20261119",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_mode": "CPU-only aggregation of saved validation sufficient statistics; no model inference or training",
        "validation_images": 128,
        "independent_test319_used": False,
        "bootstrap_reps": args.bootstrap_reps, "bootstrap_seed": args.seed,
        "input_root": "repository checkout (relative paths recorded below)",
        "upstream_corrected_pr6_head": "f8a9530a4c618c72a54abd52eba47eecfc4e5179",
        "checkpoint_source": "PR #6 checkpoint audit and corrected-release evaluation JSON; checkpoints not loaded in this analysis",
        "checkpoint_sha256_verified_by_upstream_audit": verified_hashes,
        "upstream_fresh_reproduction_status": reproduction_audit["status"],
        "upstream_b0_gate_status": reproduction_audit["b0_gate"]["status"],
        "source_file_hashes": hashed,
        "checks": audits,
        "python": platform.python_version(), "platform": platform.platform(),
        "numpy": np.__version__, "matplotlib": matplotlib.__version__,
        "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
    }
    (out / "provenance.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "validation.txt").write_text("PASS\n" + "\n".join(audits) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
