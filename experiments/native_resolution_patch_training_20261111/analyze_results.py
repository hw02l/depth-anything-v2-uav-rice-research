from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _num(row: dict[str, str], key: str) -> float:
    value = row.get(key)
    return float(value) if value not in (None, "", "None") else 0.0


def _absolute_metric(rows: list[dict[str, str]], metric: str) -> np.ndarray:
    values = []
    for row in rows:
        n = max(_num(row, "valid_pixels"), 0)
        err = _num(row, "abs_error_sum")
        sq = _num(row, "squared_error_sum")
        if metric == "mae": value = err / n if n else np.nan
        elif metric == "rmse": value = np.sqrt(sq / n) if n else np.nan
        elif metric == "abs_rel": value = _num(row, "abs_relative_sum") / n if n else np.nan
        elif metric == "bias": value = (_num(row, "prediction_sum") - _num(row, "target_sum")) / n if n else np.nan
        elif metric == "pearson":
            p, t = _num(row, "prediction_sum"), _num(row, "target_sum")
            cov = _num(row, "prediction_target_sum") - p * t / n if n else 0
            pv = _num(row, "prediction_squared_sum") - p * p / n if n else 0
            tv = _num(row, "target_squared_sum") - t * t / n if n else 0
            value = cov / np.sqrt(pv * tv) if pv > 0 and tv > 0 else np.nan
        elif metric == "full14_mae":
            count = _num(row, "full14_pair_count")
            value = _num(row, "full14_abs_delta_error_sum") / count if count else np.nan
        else: raise ValueError(metric)
        values.append(value)
    return np.asarray(values, np.float64)


def _pool_absolute(rows: list[dict[str, str]], metric: str) -> float:
    n = sum(_num(row, "valid_pixels") for row in rows)
    if metric == "full14_mae":
        total = sum(_num(row, "full14_pair_count") for row in rows)
        error = sum(_num(row, "full14_abs_delta_error_sum") for row in rows)
        return error / total
    if metric == "mae":
        return sum(_num(row, "abs_error_sum") for row in rows) / n
    if metric == "rmse":
        return float(np.sqrt(sum(_num(row, "squared_error_sum") for row in rows) / n))
    if metric == "abs_rel":
        return sum(_num(row, "abs_relative_sum") for row in rows) / n
    if metric == "bias":
        return (sum(_num(row, "prediction_sum") - _num(row, "target_sum") for row in rows)) / n
    if metric == "pearson":
        p = sum(_num(row, "prediction_sum") for row in rows)
        t = sum(_num(row, "target_sum") for row in rows)
        ps = sum(_num(row, "prediction_squared_sum") for row in rows)
        ts = sum(_num(row, "target_squared_sum") for row in rows)
        cross = sum(_num(row, "prediction_target_sum") for row in rows)
        covariance = cross - p * t / n
        pv = ps - p * p / n
        tv = ts - t * t / n
        return float(covariance / np.sqrt(pv * tv))
    raise ValueError(metric)


def _local_metric(rows: list[dict[str, str]], metric: str) -> np.ndarray:
    result = []
    for row in rows:
        n = _num(row, "pair_count")
        gt_abs = _num(row, "abs_gt_delta_sum")
        pred_abs = _num(row, "abs_prediction_delta_sum")
        nonzero = _num(row, "nonzero_gt_pair_count")
        if metric == "delta_mae": value = _num(row, "abs_delta_error_sum") / n if n else np.nan
        elif metric == "delta_rmse": value = np.sqrt(_num(row, "squared_delta_error_sum") / n) if n else np.nan
        elif metric == "amplitude_ratio": value = pred_abs / gt_abs if gt_abs else np.nan
        elif metric == "sign_match_rate": value = _num(row, "sign_match_count") / nonzero if nonzero else np.nan
        else: raise ValueError(metric)
        result.append(value)
    return np.asarray(result, np.float64)


def _pool_local(rows: list[dict[str, str]], metric: str) -> float:
    n = sum(_num(row, "pair_count") for row in rows)
    gt_abs = sum(_num(row, "abs_gt_delta_sum") for row in rows)
    pred_abs = sum(_num(row, "abs_prediction_delta_sum") for row in rows)
    nonzero = sum(_num(row, "nonzero_gt_pair_count") for row in rows)
    if metric == "delta_mae":
        return sum(_num(row, "abs_delta_error_sum") for row in rows) / n
    if metric == "delta_rmse":
        return float(np.sqrt(sum(_num(row, "squared_delta_error_sum") for row in rows) / n))
    if metric == "amplitude_ratio":
        return pred_abs / gt_abs
    if metric == "sign_match_rate":
        return sum(_num(row, "sign_match_count") for row in rows) / nonzero
    raise ValueError(metric)


def _paired_delta_bootstrap(base: np.ndarray, candidate: np.ndarray, draws: np.ndarray,
                            *, transform=None) -> tuple[float, float, float]:
    if base.shape != candidate.shape or base.shape[0] != draws.shape[1]:
        raise ValueError("Paired image arrays and bootstrap draws are inconsistent")
    diff = candidate - base
    complete = np.isfinite(diff)
    if not complete.any():
        raise ValueError("No paired image-level metric values")
    point = float(np.nanmean(diff))
    boot_values = np.nanmean(diff[draws], axis=1)
    return point, float(np.nanquantile(boot_values, .025)), float(np.nanquantile(boot_values, .975))


def _pooled_draw_values(rows: list[dict[str, str]], metric: str, draws: np.ndarray, *, local=False) -> np.ndarray:
    if local:
        counts = np.asarray([_num(r, "pair_count") for r in rows])
        if metric in ("delta_mae", "delta_rmse"):
            field = "abs_delta_error_sum" if metric == "delta_mae" else "squared_delta_error_sum"
            total = np.asarray([_num(r, field) for r in rows])[draws].sum(axis=1)
            den = counts[draws].sum(axis=1)
            return total / den if metric == "delta_mae" else np.sqrt(total / den)
        if metric == "amplitude_ratio":
            pred = np.asarray([_num(r, "abs_prediction_delta_sum") for r in rows])[draws].sum(axis=1)
            gt = np.asarray([_num(r, "abs_gt_delta_sum") for r in rows])[draws].sum(axis=1)
            return pred / gt
        if metric == "sign_match_rate":
            correct = np.asarray([_num(r, "sign_match_count") for r in rows])[draws].sum(axis=1)
            n = np.asarray([_num(r, "nonzero_gt_pair_count") for r in rows])[draws].sum(axis=1)
            return correct / n
        raise ValueError(metric)
    counts = np.asarray([_num(r, "valid_pixels") for r in rows])[draws].sum(axis=1)
    if metric == "mae":
        total = np.asarray([_num(r, "abs_error_sum") for r in rows])[draws].sum(axis=1)
        return total / counts
    if metric == "rmse":
        total = np.asarray([_num(r, "squared_error_sum") for r in rows])[draws].sum(axis=1)
        return np.sqrt(total / counts)
    if metric == "abs_rel":
        total = np.asarray([_num(r, "abs_relative_sum") for r in rows])[draws].sum(axis=1)
        return total / counts
    if metric == "bias":
        p = np.asarray([_num(r, "prediction_sum") for r in rows])[draws].sum(axis=1)
        t = np.asarray([_num(r, "target_sum") for r in rows])[draws].sum(axis=1)
        return (p - t) / counts
    if metric == "pearson":
        p = np.asarray([_num(r, "prediction_sum") for r in rows])[draws].sum(axis=1)
        t = np.asarray([_num(r, "target_sum") for r in rows])[draws].sum(axis=1)
        ps = np.asarray([_num(r, "prediction_squared_sum") for r in rows])[draws].sum(axis=1)
        ts = np.asarray([_num(r, "target_squared_sum") for r in rows])[draws].sum(axis=1)
        cross = np.asarray([_num(r, "prediction_target_sum") for r in rows])[draws].sum(axis=1)
        cov, pv, tv = cross - p * t / counts, ps - p * p / counts, ts - t * t / counts
        return cov / np.sqrt(pv * tv)
    if metric == "full14_mae":
        pairs = np.asarray([_num(r, "full14_pair_count") for r in rows])[draws].sum(axis=1)
        errors = np.asarray([_num(r, "full14_abs_delta_error_sum") for r in rows])[draws].sum(axis=1)
        return errors / pairs
    raise ValueError(metric)


def _pooled_bootstrap(base_rows: list[dict[str, str]], candidate_rows: list[dict[str, str]],
                      metric: str, draws: np.ndarray, *, local=False) -> tuple[float, float, float, float, float]:
    evaluator = _pool_local if local else _pool_absolute
    base_point = evaluator(base_rows, metric)
    candidate_point = evaluator(candidate_rows, metric)
    base_values = _pooled_draw_values(base_rows, metric, draws, local=local)
    candidate_values = _pooled_draw_values(candidate_rows, metric, draws, local=local)
    differences = candidate_values - base_values
    return (base_point, candidate_point, candidate_point - base_point,
            float(np.quantile(differences, .025)), float(np.quantile(differences, .975)))


def analyze(eval_root: Path, output_dir: Path, repeats: int = 10000, seed: int = 42):
    output_dir.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, 128, size=(repeats, 128), dtype=np.int16)
    rows_out = []
    contrasts = (("F1", "C0"), ("F2", "C0"))
    absolute_metrics = ("mae", "rmse", "abs_rel", "bias", "pearson", "full14_mae")
    local_metrics = ("delta_mae", "delta_rmse", "amplitude_ratio", "sign_match_rate")
    for method in ("P0", "P1", "P2"):
        cache = {}
        for group in ("C0", "F1", "F2"):
            base_dir = eval_root / f"{group}_{method}"
            abs_rows = _read_csv(base_dir / f"per_image_absolute_{group}_{method}.csv")
            local_rows = _read_csv(base_dir / f"per_image_local_{group}_{method}.csv")
            if len(abs_rows) != 128:
                raise ValueError(f"{group}/{method} absolute table must have 128 rows")
            abs_rows.sort(key=lambda r: int(r["image_index"]))
            if [int(r["image_index"]) for r in abs_rows] != list(range(128)):
                raise ValueError("Image IDs/order differ from canonical Validation128")
            if len({r["image_relpath"] for r in abs_rows}) != 128:
                raise ValueError("Duplicate image IDs")
            primary = [r for r in local_rows if int(r["stride_px"]) == 14 and r["group"] == "gt_ge20cm"]
            primary.sort(key=lambda r: int(r["image_index"]))
            if len(primary) != 128:
                raise ValueError(f"{group}/{method} lacks per-image fixed large-pair 14px rows")
            cache[group] = (abs_rows, primary)
        for candidate, baseline in contrasts:
            base_abs, base_local = cache[baseline]
            cand_abs, cand_local = cache[candidate]
            if [r["image_relpath"] for r in base_abs] != [r["image_relpath"] for r in cand_abs]:
                raise ValueError("Models do not share the same image IDs")
            for metric in absolute_metrics:
                b = _absolute_metric(base_abs, metric)
                c = _absolute_metric(cand_abs, metric)
                point, lo, hi = _paired_delta_bootstrap(b, c, draws)
                pooled_b, pooled_c, pooled_diff, pooled_lo, pooled_hi = _pooled_bootstrap(
                    base_abs, cand_abs, metric, draws, local=False)
                for estimator, estimate, delta, lower, upper in (
                    ("image_mean", float(np.nanmean(b)), point, lo, hi),
                    ("pooled", pooled_b, pooled_diff, pooled_lo, pooled_hi),
                ):
                    candidate_point = float(np.nanmean(c)) if estimator == "image_mean" else pooled_c
                    rows_out.append({
                        "method": method, "candidate": candidate, "baseline": baseline,
                        "metric": metric, "estimator": estimator, "baseline_value": estimate,
                        "candidate_value": candidate_point, "candidate_minus_baseline": delta,
                        "ci95_low": lower, "ci95_high": upper, "repeats": repeats, "seed": seed,
                    })
            for metric in local_metrics:
                b = _local_metric(base_local, metric)
                c = _local_metric(cand_local, metric)
                if metric == "amplitude_ratio":
                    b, c = np.abs(b - 1.0), np.abs(c - 1.0)
                    metric_label = "distance_from_unit_amplitude"
                else:
                    metric_label = metric
                point, lo, hi = _paired_delta_bootstrap(b, c, draws)
                pooled_b, pooled_c, pooled_diff, pooled_lo, pooled_hi = _pooled_bootstrap(
                    base_local, cand_local, metric, draws, local=True)
                if metric == "amplitude_ratio":
                    pooled_diff = abs(pooled_c - 1.0) - abs(pooled_b - 1.0)
                    # The bootstrap CI is recomputed on the transformed quantity.
                    rb = _pooled_draw_values(base_local, metric, draws, local=True)
                    rc = _pooled_draw_values(cand_local, metric, draws, local=True)
                    transformed = np.abs(rc - 1.0) - np.abs(rb - 1.0)
                    pooled_lo, pooled_hi = map(float, np.quantile(transformed, [.025, .975]))
                    pooled_b, pooled_c = abs(pooled_b - 1.0), abs(pooled_c - 1.0)
                for estimator, base_value, candidate_value, delta, lower, upper in (
                    ("image_mean", float(np.nanmean(b)), float(np.nanmean(c)), point, lo, hi),
                    ("pooled", pooled_b, pooled_c, pooled_diff, pooled_lo, pooled_hi),
                ):
                    rows_out.append({
                        "method": method, "candidate": candidate, "baseline": baseline,
                        "metric": metric_label, "estimator": estimator,
                        "baseline_value": base_value, "candidate_value": candidate_value,
                        "candidate_minus_baseline": delta, "ci95_low": lower, "ci95_high": upper,
                        "repeats": repeats, "seed": seed,
                    })
    path = output_dir / "paired_bootstrap.csv"
    fields = list(dict.fromkeys(key for row in rows_out for key in row))
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows_out)
    (output_dir / "bootstrap_metadata.json").write_text(json.dumps({
        "repeats": repeats, "seed": seed, "unit": "image, paired",
        "image_count": 128, "day": "2025/Day047",
        "pooled_and_image_mean_separate": True,
        "independent_test_used": False,
        "scope": "Day047 image-level descriptive uncertainty; no between-day inference",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = analyze(args.eval_root, args.output_dir, args.repeats, args.seed)
    print(json.dumps({"comparisons": len(rows), "output": str(args.output_dir / "paired_bootstrap.csv")}))


if __name__ == "__main__":
    main()
