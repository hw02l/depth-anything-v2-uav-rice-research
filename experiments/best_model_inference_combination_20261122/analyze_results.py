#!/usr/bin/env python3
"""Combine the new B3 inference with four verified historical conditions."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def num(row: dict[str, str], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        if key in row and row[key] not in ("", "NA", "nan", "None"):
            return float(row[key])
    return default


def record(index: int, valid: int, abs_sum: float, sq_sum: float,
           full_n: int, full_sum: float, high_n: int, high_sum: float,
           rel_sum: float | None = None, bias: float | None = None,
           pearson: float | None = None, pred_sum: float | None = None,
           gt_sum: float | None = None, pred_sq: float | None = None,
           gt_sq: float | None = None, pred_gt: float | None = None,
           pooled_pearson: float | None = None) -> dict:
    return {"image_index": index, "valid_pixels": valid, "abs_sum": abs_sum,
            "sq_sum": sq_sum, "full_n": full_n, "full_sum": full_sum,
            "high_n": high_n, "high_sum": high_sum,
            "rel_sum": rel_sum, "bias": bias, "pearson": pearson,
            "pooled_pearson": pooled_pearson,
            "pred_sum": pred_sum, "gt_sum": gt_sum, "pred_sq": pred_sq,
            "gt_sq": gt_sq, "pred_gt": pred_gt,
            "mae": abs_sum / valid if valid else None,
            "rmse": math.sqrt(sq_sum / valid) if valid else None,
            "full_mean": full_sum / full_n if full_n else None,
            "high_mean": high_sum / high_n if high_n else None}


def load_old_b(rows: list[dict[str, str]], method: str, pooled_pearson: float | None) -> list[dict]:
    selected = [r for r in rows if int(r["epoch"]) == 20 and r["method"] == method]
    out = []
    for r in selected:
        n = int(r["valid_pixels"])
        full_n = int(r["local_pair_count"])
        high_n = int(r["large14_pairs"])
        out.append(record(int(r["image_index"]), n,
            num(r, "mae_m") * n, num(r, "rmse_m") ** 2 * n,
            full_n, num(r, "local_depth_delta_mae_m") * full_n,
            high_n, num(r, "large14_mae_m") * high_n,
            rel_sum=num(r, "abs_rel") * n, pearson=num(r, "pearson_r"),
            pooled_pearson=pooled_pearson))
    return out


def load_new(new_dir: Path, group: str) -> list[dict]:
    absolute = [r for r in read_csv(new_dir / "per_image_absolute_sufficient.csv")
                if r["model_id"] == f"{group}-B3"]
    local = [r for r in read_csv(new_dir / "per_image_local_sufficient.csv")
             if r["model_id"] == f"{group}-B3" and int(r["stride_px"]) == 14
             and r["group"] == "gt_ge20cm"]
    high_by = {}
    for r in local:
        i = int(r["image_index"])
        acc = high_by.setdefault(i, [0, 0.0])
        acc[0] += int(r["pair_count"])
        acc[1] += num(r, "absolute_error_sum_m")
    out = []
    for r in absolute:
        i = int(r["image_index"])
        out.append(record(i, int(r["valid_pixel_count"]), num(r, "abs_error_sum_m"),
            num(r, "squared_error_sum_m2"), int(r["full14_pair_count"]),
            num(r, "full14_abs_error_sum_m"), *high_by.get(i, [0, 0.0]),
            rel_sum=num(r, "abs_relative_error_sum"), bias=num(r, "bias_m"),
            pearson=num(r, "pearson_r"), pred_sum=num(r, "prediction_sum_m"),
            gt_sum=num(r, "gt_sum_m"), pred_sq=num(r, "prediction_squared_sum_m2"),
            gt_sq=num(r, "gt_squared_sum_m2"), pred_gt=num(r, "prediction_gt_cross_sum_m2")))
    return sorted(out, key=lambda x: x["image_index"])


def load_e3(rows: list[dict[str, str]]) -> list[dict]:
    selected = [r for r in rows if r["model_id"] == "E3"]
    return [record(int(r["image_index"]), int(r["valid_pixel_count"]),
        num(r, "abs_error_sum_m"), num(r, "squared_error_sum_m2"),
        int(r["full14_pair_count"]), num(r, "full14_abs_error_sum_m"),
        int(r["gt_ge20_pair_count"]), num(r, "gt_ge20_abs_error_sum_m"),
        rel_sum=num(r, "abs_relative_error_sum"), bias=num(r, "bias_m"),
        pearson=num(r, "pearson_r"), pred_sum=num(r, "prediction_sum_m"),
        gt_sum=num(r, "gt_sum_m"), pred_sq=num(r, "prediction_squared_sum_m2"),
        gt_sq=num(r, "gt_squared_sum_m2"), pred_gt=num(r, "prediction_gt_cross_sum_m2")) for r in selected]


def load_c0_p0(absolute_path: Path, local_path: Path) -> list[dict]:
    absolute = read_csv(absolute_path)
    local = [r for r in read_csv(local_path)
             if int(r["stride_px"]) == 14 and r["group"] == "gt_ge20cm"]
    high_by = {}
    for r in local:
        i = int(r["image_index"])
        acc = high_by.setdefault(i, [0, 0.0])
        acc[0] += int(r["pair_count"])
        acc[1] += num(r, "abs_delta_error_sum", "delta_mae_m") * (
            1 if "abs_delta_error_sum" in r and r["abs_delta_error_sum"] else int(r["pair_count"]))
    out = []
    for r in absolute:
        i = int(r["image_index"])
        out.append(record(i, int(r["valid_pixels"]), num(r, "abs_error_sum"),
            num(r, "squared_error_sum"), int(r["full14_pair_count"]),
            num(r, "full14_abs_delta_error_sum"), *high_by.get(i, [0, 0.0]),
            rel_sum=num(r, "abs_relative_sum"), bias=num(r, "bias_m"),
            pearson=num(r, "pearson_r"), pred_sum=num(r, "prediction_sum"),
            gt_sum=num(r, "target_sum"), pred_sq=num(r, "prediction_squared_sum"),
            gt_sq=num(r, "target_squared_sum"), pred_gt=num(r, "prediction_target_sum")))
    return sorted(out, key=lambda x: x["image_index"])


def estimate(records: list[dict], metric: str, estimand: str) -> float:
    if estimand == "pooled":
        if metric == "absolute_mae":
            n = sum(r["valid_pixels"] for r in records)
            return sum(r["abs_sum"] for r in records) / n
        if metric == "absolute_rmse":
            n = sum(r["valid_pixels"] for r in records)
            return math.sqrt(sum(r["sq_sum"] for r in records) / n)
        if metric == "abs_rel":
            vals = [r["rel_sum"] for r in records]
            return sum(vals) / sum(r["valid_pixels"] for r in records) if all(v is not None for v in vals) else float("nan")
        if metric == "bias":
            vals = [r["bias"] for r in records]
            return sum(v * r["valid_pixels"] for r, v in zip(records, vals)) / sum(r["valid_pixels"] for r in records) if all(v is not None for v in vals) else float("nan")
        if metric == "pearson":
            pooled = [r["pooled_pearson"] for r in records]
            if all(v is not None for v in pooled) and all(v == pooled[0] for v in pooled):
                return pooled[0]
            fields = ("pred_sum", "gt_sum", "pred_sq", "gt_sq", "pred_gt")
            if any(any(r[k] is None for k in fields) for r in records):
                return float("nan")
            n = sum(r["valid_pixels"] for r in records)
            ps = sum(r["pred_sum"] for r in records); gs = sum(r["gt_sum"] for r in records)
            cov = sum(r["pred_gt"] for r in records) - ps * gs / n
            vp = sum(r["pred_sq"] for r in records) - ps * ps / n
            vg = sum(r["gt_sq"] for r in records) - gs * gs / n
            return cov / math.sqrt(vp * vg) if vp > 0 and vg > 0 else float("nan")
        prefix = "full" if metric == "full14_mae" else "high"
        n = sum(r[f"{prefix}_n"] for r in records)
        return sum(r[f"{prefix}_sum"] for r in records) / n if n else float("nan")
    key = {"absolute_mae": "mae", "absolute_rmse": "rmse", "abs_rel": "rel_sum",
           "bias": "bias", "pearson": "pearson", "full14_mae": "full_mean",
           "high14_mae": "high_mean"}[metric]
    values = [r[key] for r in records if r[key] is not None]
    if metric == "abs_rel":
        values = [r["rel_sum"] / r["valid_pixels"] for r in records
                  if r["rel_sum"] is not None and r["valid_pixels"]]
    return float(np.mean(values)) if values else float("nan")


def bootstrap_pair(a: list[dict], b: list[dict], metric: str, estimand: str,
                   repeats: int, seed: int) -> dict:
    if len(a) != len(b) or [r["image_index"] for r in a] != [r["image_index"] for r in b]:
        raise ValueError("Paired conditions have different image IDs or ordering")
    point_a, point_b = estimate(a, metric, estimand), estimate(b, metric, estimand)
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, len(a), size=(repeats, len(a)))

    def draws(records: list[dict]) -> np.ndarray:
        if estimand == "pooled":
            if metric == "absolute_rmse":
                numerator = np.asarray([r["sq_sum"] for r in records])[ids].sum(axis=1)
                denominator = np.asarray([r["valid_pixels"] for r in records])[ids].sum(axis=1)
                return np.sqrt(numerator / denominator)
            prefix = "full" if metric == "full14_mae" else "high"
            numerator = np.asarray([r[f"{prefix}_sum"] for r in records])[ids].sum(axis=1)
            denominator = np.asarray([r[f"{prefix}_n"] for r in records])[ids].sum(axis=1)
            return np.divide(numerator, denominator, out=np.full(repeats, np.nan), where=denominator > 0)
        key = {"absolute_rmse": "rmse", "full14_mae": "full_mean",
               "high14_mae": "high_mean"}[metric]
        values = np.asarray([np.nan if r[key] is None else r[key] for r in records])[ids]
        valid = np.isfinite(values)
        counts = valid.sum(axis=1)
        return np.divide(np.nansum(values, axis=1), counts,
                         out=np.full(repeats, np.nan), where=counts > 0)

    av, bv = draws(a), draws(b)
    diffs = av - bv
    relative = np.divide(diffs, bv, out=np.full(repeats, np.nan), where=bv != 0) * 100
    return {"metric": metric, "estimand": estimand, "a": point_a, "b": point_b,
            "difference_a_minus_b": point_a - point_b,
            "relative_change_percent_of_b": (point_a - point_b) / point_b * 100,
            "bootstrap_repeats": repeats, "seed": seed,
            "difference_ci95_low": float(np.nanpercentile(diffs, 2.5)),
            "difference_ci95_high": float(np.nanpercentile(diffs, 97.5)),
            "relative_ci95_low": float(np.nanpercentile(relative, 2.5)),
            "relative_ci95_high": float(np.nanpercentile(relative, 97.5)),
            "fraction_a_better": float(np.mean(diffs < 0))}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True, help="Repository snapshot containing historical CSVs")
    ap.add_argument("--new-dir", type=Path, required=True, help="Completed full_eval output directory")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--repeats", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if json.loads((args.new_dir / "run_complete.json").read_text(encoding="utf-8"))["status"] != "complete":
        raise RuntimeError("New inference run is not complete")
    exp = args.repo / "experiments"
    b_rows = read_csv(exp / "multiscale_tile_inference_20261016" / "per_image_metrics.csv")
    b_summary_rows = read_csv(exp / "multiscale_tile_inference_20261016" / "method_comparison.csv")
    b_pearson = {r["method"]: num(r, "pearson_r") for r in b_summary_rows
                 if int(r["epoch"]) == 20}
    conditions = {
        "epoch20-B0": load_old_b(b_rows, "B0", b_pearson.get("B0")),
        "epoch20-B3": load_old_b(b_rows, "B3", b_pearson.get("B3")),
        "E3-B0": load_e3(read_csv(exp / "local_huber_20261105" / "analysis" / "final_20261107" / "inputs" / "E3_per_image.csv")),
        "C0-P0": load_c0_p0(
            exp / "native_resolution_patch_training_20261111" / "analysis" / "rgb_input_fix_20261114" / "per_condition" / "C0_P0" / "per_image_absolute_C0_P0.csv",
            exp / "native_resolution_patch_training_20261111" / "analysis" / "rgb_input_fix_20261114" / "per_condition" / "C0_P0" / "per_image_local_C0_P0.csv"),
        "C0-B3": load_new(args.new_dir, "C0"), "E3-B3": load_new(args.new_dir, "E3"),
    }
    run_complete = json.loads((args.new_dir / "run_complete.json").read_text(encoding="utf-8"))
    expected_summary = {r["condition"]: r for r in run_complete["summaries"]}
    for name in ("C0-B3", "E3-B3"):
        rows = conditions[name]
        expected = expected_summary[name]
        checks = {
            "absolute_rmse": (estimate(rows, "absolute_rmse", "pooled"), expected["rmse_m"]),
            "full14_mae": (estimate(rows, "full14_mae", "pooled"), expected["full14_local_mae_m"]),
            "high14_mae": (estimate(rows, "high14_mae", "pooled"), expected["gt_ge20_14_local_mae_m"]),
        }
        bad = {metric: pair for metric, pair in checks.items()
               if not math.isclose(pair[0], pair[1], rel_tol=0, abs_tol=1e-12)}
        if bad:
            raise ValueError(f"{name} sufficient-statistic reconciliation failed: {bad}")
    expected_ids = list(range(128))
    for name, rows in conditions.items():
        if [r["image_index"] for r in rows] != expected_ids:
            raise ValueError(f"{name} does not contain the canonical 128 paired image indices")
    summary = []
    for name, rows in conditions.items():
        for metric in ("absolute_mae", "absolute_rmse", "abs_rel", "bias", "pearson", "full14_mae", "high14_mae"):
            summary.append({"condition": name, "metric": metric,
                "pooled": estimate(rows, metric, "pooled"),
                "image_mean": estimate(rows, metric, "image_mean"),
                "valid_pixels": sum(r["valid_pixels"] for r in rows),
                "full14_pairs": sum(r["full_n"] for r in rows),
                "high14_pairs": sum(r["high_n"] for r in rows),
                "images_with_high14_pairs": sum(r["high_n"] > 0 for r in rows)})
    write_csv(args.output / "condition_summary.csv", summary)
    image_rows = []
    for name, rows in conditions.items():
        for r in rows:
            image_rows.append({"condition": name, **r})
    write_csv(args.output / "condition_per_image.csv", image_rows)
    comparisons = [("C0-B3", "C0-P0"), ("E3-B3", "E3-B0"),
        ("C0-B3", "epoch20-B3"), ("E3-B3", "epoch20-B3"), ("C0-B3", "E3-B3")]
    bootstrap = []
    for an, bn in comparisons:
        for metric in ("absolute_rmse", "full14_mae", "high14_mae"):
            for estimand in ("pooled", "image_mean"):
                bootstrap.append({"comparison": f"{an} - {bn}",
                    **bootstrap_pair(conditions[an], conditions[bn], metric, estimand,
                                     args.repeats, args.seed)})
    write_csv(args.output / "paired_bootstrap.csv", bootstrap)
    pareto = []
    pooled_by = {name: {r["metric"]: r["pooled"] for r in summary
                        if r["condition"] == name and r["metric"] in
                        ("absolute_rmse", "full14_mae", "high14_mae")}
                 for name in conditions}
    for name, metrics in pooled_by.items():
        dominated_by = [other for other, values in pooled_by.items() if other != name and
            all(values[m] <= metrics[m] for m in metrics) and
            any(values[m] < metrics[m] for m in metrics)]
        pareto.append({"condition": name, **metrics, "pareto_nondominated": not dominated_by,
                       "dominated_by": ";".join(dominated_by)})
    write_csv(args.output / "pareto.csv", pareto)
    names = list(conditions)
    specs = [("absolute_rmse", "Absolute RMSE (m)"),
             ("full14_mae", "All valid 14 px pair MAE (m)"),
             ("high14_mae", "GT |Δ| ≥ 0.20 m, 14 px pair MAE (m)")]
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for ax, (metric, title) in zip(axes, specs):
            vals = [pooled_by[n][metric] for n in names]
            ax.bar(range(len(names)), vals, color=["#7e8aa2"] * 4 + ["#247ba0", "#70c1b3"])
            ax.set_title(title); ax.set_xticks(range(len(names)), names, rotation=35, ha="right")
            ax.grid(axis="y", alpha=.25)
        fig.tight_layout()
        args.output.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output / "validation_comparison.png", dpi=180)
        fig.savefig(args.output / "validation_comparison.svg")
        plt.close(fig)
    except ImportError:
        (args.output / "PLOT_MISSING.txt").write_text("matplotlib unavailable; numeric CSVs were written.\n", encoding="utf-8")
    print(json.dumps({"conditions": {n: {m: pooled_by[n][m] for m, _ in specs} for n in names},
                      "output": str(args.output), "bootstrap_rows": len(bootstrap)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
