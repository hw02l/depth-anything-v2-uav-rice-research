#!/usr/bin/env python3
"""Finalize paired validation analysis from frozen E0-E3 sufficient statistics.

The script does not train or infer. It validates image/model alignment, checks
re-evaluated pooled metrics against the recorded run summary, then emits
condition tables, paired image-cluster bootstrap intervals, and figures.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

MODELS = ("E0", "E1", "E2", "E3")
PAIRS = (("E3", "E1"), ("E2", "E1"), ("E3", "E0"), ("E2", "E0"))
POOL_FIELDS = {
    "mae_m": ("abs_error_sum_m", "valid_pixel_count", "mean"),
    "rmse_m": ("squared_error_sum_m2", "valid_pixel_count", "rmse"),
    "absrel": ("abs_relative_error_sum", "valid_pixel_count", "mean"),
    "full14_local_mae_m": ("full14_abs_error_sum_m", "full14_pair_count", "mean"),
    "gt_ge20_local_mae_m": ("gt_ge20_abs_error_sum_m", "gt_ge20_pair_count", "mean"),
    "gt_ge20_amplitude_ratio": ("gt_ge20_abs_prediction_sum_m", "gt_ge20_abs_gt_sum_m", "ratio"),
    "gt_ge20_sign_match_rate": ("gt_ge20_sign_match_count", "gt_ge20_nonzero_sign_count", "ratio"),
}
IMAGE_FIELDS = {
    "mae_m": "mae_m", "rmse_m": "rmse_m", "absrel": "abs_rel",
    "full14_local_mae_m": "full14_local_mae_m",
    "gt_ge20_local_mae_m": "gt_ge20_local_mae_m",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def strip_svg_line_padding(path: Path) -> None:
    """Remove renderer-added trailing spaces without changing SVG geometry."""
    raw = path.read_bytes()
    cleaned_lines = []
    for line in raw.splitlines(keepends=True):
        if line.endswith(b"\r\n"):
            body, ending = line[:-2], b"\r\n"
        elif line.endswith(b"\n"):
            body, ending = line[:-1], b"\n"
        else:
            body, ending = line, b""
        cleaned_lines.append(body.rstrip(b" \t") + ending)
    cleaned = b"".join(cleaned_lines)
    if cleaned != raw:
        path.write_bytes(cleaned)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty result: {path}")
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def numeric_rows(rows: list[dict[str, str]], fields: set[str]) -> list[dict[str, Any]]:
    output = []
    for source in rows:
        row: dict[str, Any] = dict(source)
        for key in fields:
            value = row.get(key)
            row[key] = None if value in (None, "") else float(value)
        output.append(row)
    return output


def optional_float(value: str | None) -> float | None:
    """Keep undefined metrics undefined when a frozen stratum has zero pairs."""
    return None if value in (None, "") else float(value)


def aggregate(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    if not rows:
        return {}
    def sm(key: str) -> float:
        return float(sum(float(r[key]) for r in rows))
    n = int(sm("valid_pixel_count"))
    npair = int(sm("full14_pair_count"))
    large_n = int(sm("gt_ge20_pair_count"))
    sum_p, sum_d = sm("prediction_sum_m"), sm("gt_sum_m")
    sum_p2, sum_d2, sum_pd = sm("prediction_squared_sum_m2"), sm("gt_squared_sum_m2"), sm("prediction_gt_cross_sum_m2")
    cov = sum_pd - (sum_p * sum_d / n) if n else 0.0
    vp = sum_p2 - (sum_p * sum_p / n) if n else 0.0
    vd = sum_d2 - (sum_d * sum_d / n) if n else 0.0
    return {
        "valid_pixel_count": n,
        "mae_m": sm("abs_error_sum_m") / n if n else None,
        "rmse_m": math.sqrt(sm("squared_error_sum_m2") / n) if n else None,
        "absrel": sm("abs_relative_error_sum") / n if n else None,
        "bias_m": (sum_p - sum_d) / n if n else None,
        "pearson_r": cov / math.sqrt(vp * vd) if n and vp > 0 and vd > 0 else None,
        "full14_pair_count": npair,
        "full14_local_mae_m": sm("full14_abs_error_sum_m") / npair if npair else None,
        "gt_ge20_pair_count": large_n,
        "gt_ge20_local_mae_m": sm("gt_ge20_abs_error_sum_m") / large_n if large_n else None,
        "gt_ge20_amplitude_ratio": sm("gt_ge20_abs_prediction_sum_m") / sm("gt_ge20_abs_gt_sum_m") if sm("gt_ge20_abs_gt_sum_m") else None,
        "gt_ge20_sign_match_rate": sm("gt_ge20_sign_match_count") / sm("gt_ge20_nonzero_sign_count") if sm("gt_ge20_nonzero_sign_count") else None,
    }


def load_images(path: Path) -> dict[str, list[dict[str, Any]]]:
    rows = read_csv(path)
    required = {
        "model_id", "image_index", "image_relpath", "crop", "view",
        "valid_pixel_count", "abs_error_sum_m", "squared_error_sum_m2", "abs_relative_error_sum",
        "prediction_sum_m", "gt_sum_m", "prediction_squared_sum_m2", "gt_squared_sum_m2",
        "prediction_gt_cross_sum_m2", "full14_pair_count", "full14_abs_error_sum_m",
        "gt_ge20_pair_count", "gt_ge20_abs_error_sum_m", "gt_ge20_abs_gt_sum_m",
        "gt_ge20_abs_prediction_sum_m", "gt_ge20_nonzero_sign_count", "gt_ge20_sign_match_count",
        "mae_m", "rmse_m", "abs_rel", "full14_local_mae_m", "gt_ge20_local_mae_m",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Per-image CSV missing fields: {sorted(required - set(rows[0] if rows else []))}")
    by_model: dict[str, list[dict[str, Any]]] = {m: [] for m in MODELS}
    integer_fields = {"image_index", "valid_pixel_count", "full14_pair_count", "gt_ge20_pair_count", "gt_ge20_nonzero_sign_count", "gt_ge20_sign_match_count"}
    for r in rows:
        model = r["model_id"]
        if model not in by_model:
            raise ValueError(f"Unexpected model id {model}")
        item: dict[str, Any] = dict(r)
        for key in required - {"model_id", "image_relpath", "crop", "view"}:
            if key in integer_fields:
                item[key] = int(float(r[key]))
            else:
                item[key] = None if r[key] in (None, "") else float(r[key])
        by_model[model].append(item)
    for model in MODELS:
        by_model[model].sort(key=lambda r: r["image_index"])
        if len(by_model[model]) != 128 or [r["image_index"] for r in by_model[model]] != list(range(128)):
            raise ValueError(f"{model} must contain the canonical 128 validation rows")
    for model in MODELS[1:]:
        for i, (base, candidate) in enumerate(zip(by_model["E0"], by_model[model])):
            for key in ("image_index", "image_relpath", "crop", "view", "valid_pixel_count", "full14_pair_count", "gt_ge20_pair_count", "gt_ge20_nonzero_sign_count"):
                if base[key] != candidate[key]:
                    raise ValueError(f"Image/GT alignment differs: {model} image {i} field {key}")
    return by_model


def check_pool(by_model: dict[str, list[dict[str, Any]]], summary_path: Path) -> list[dict[str, Any]]:
    expected = json.loads(summary_path.read_text(encoding="utf-8"))["selected"]
    reference = {r["group"]: r for r in expected}
    out = []
    tolerance = 1e-7
    keys = {"rmse_m": "rmse_m", "mae_m": "mae_m", "absrel": "absrel", "full14_local_mae_m": "full14_mae_m", "gt_ge20_local_mae_m": "gt_ge20cm_14_mae_m", "gt_ge20_amplitude_ratio": "amplitude_ratio", "gt_ge20_sign_match_rate": "sign_match_rate"}
    for model in MODELS:
        actual = aggregate(by_model[model]); exp = reference[model]
        for actual_key, expected_key in keys.items():
            a, e = float(actual[actual_key]), float(exp[expected_key])
            passed = abs(a - e) <= tolerance
            out.append({"model": model, "metric": actual_key, "actual": a, "expected": e, "absolute_difference": abs(a-e), "tolerance": tolerance, "passed": passed})
            if not passed:
                raise ValueError(f"Per-image sufficient sums do not reproduce {model}/{actual_key}: {a} vs {e}")
        if int(actual["gt_ge20_pair_count"]) != 1911836:
            raise ValueError(f"Unexpected primary fixed pair count in {model}: {actual['gt_ge20_pair_count']}")
        if int(actual["full14_pair_count"]) != 5320891492:
            raise ValueError(f"Unexpected all-pair 14px count in {model}: {actual['full14_pair_count']}")
    return out


def pool_bootstrap(by_model: dict[str, list[dict[str, Any]]], seed: int, repeats: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, 128, size=(repeats, 128), endpoint=False)
    vectors = {m: {k: np.asarray([r[field] for r in by_model[m]], dtype=np.float64) for k, (field, _, _) in POOL_FIELDS.items()} for m in MODELS}
    # Count and sum fields used by pooled Pearson are carried separately.
    pearson_fields = ("prediction_sum_m", "gt_sum_m", "prediction_squared_sum_m2", "gt_squared_sum_m2", "prediction_gt_cross_sum_m2", "valid_pixel_count")
    pearson = {m: {k: np.asarray([r[k] for r in by_model[m]], dtype=np.float64) for k in pearson_fields} for m in MODELS}
    results = []
    for metric in list(POOL_FIELDS) + ["pearson_r"]:
        vals: dict[str, np.ndarray] = {}
        for m in MODELS:
            if metric in POOL_FIELDS:
                num_key, den_key, op = POOL_FIELDS[metric]
                num = np.take(vectors[m][metric], draws).sum(axis=1)
                den = np.take(np.asarray([r[den_key] for r in by_model[m]], dtype=np.float64), draws).sum(axis=1)
                if op == "rmse":
                    value = np.sqrt(np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0))
                else:
                    value = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
                vals[m] = value
            else:
                sums = {k: np.take(v, draws).sum(axis=1) for k, v in pearson[m].items()}
                n = sums["valid_pixel_count"]
                cov = sums["prediction_gt_cross_sum_m2"] - sums["prediction_sum_m"] * sums["gt_sum_m"] / n
                vp = sums["prediction_squared_sum_m2"] - sums["prediction_sum_m"]**2 / n
                vd = sums["gt_squared_sum_m2"] - sums["gt_sum_m"]**2 / n
                vals[m] = np.divide(cov, np.sqrt(np.maximum(vp, 0) * np.maximum(vd, 0)), out=np.full_like(cov, np.nan), where=(n > 0) & (vp > 0) & (vd > 0))
        for candidate, baseline in PAIRS:
            delta = vals[candidate] - vals[baseline]
            finite = delta[np.isfinite(delta)]
            if len(finite) != repeats:
                raise ValueError(f"Nonfinite pooled bootstrap replicates for {metric} {candidate}-{baseline}")
            point = aggregate(by_model[candidate])[metric] - aggregate(by_model[baseline])[metric]
            results.append({"estimand": "pooled", "metric": metric, "candidate": candidate, "baseline": baseline,
                            "difference_candidate_minus_baseline": point, "ci95_low": float(np.quantile(finite, .025)),
                            "ci95_high": float(np.quantile(finite, .975)), "bootstrap_repeats": repeats,
                            "seed": seed, "images": 128, "images_with_metric": 128,
                            "same_resampled_images_all_models": True,
                            "improved_images_candidate_vs_baseline": int(sum(float(by_model[candidate][i][IMAGE_FIELDS.get(metric, 'mae_m')] or 0) < float(by_model[baseline][i][IMAGE_FIELDS.get(metric, 'mae_m')] or 0) for i in range(128))) if metric in IMAGE_FIELDS else None})
    return results


def image_mean_bootstrap(by_model: dict[str, list[dict[str, Any]]], seed: int, repeats: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, 128, size=(repeats, 128), endpoint=False)
    results = []
    for metric, field in IMAGE_FIELDS.items():
        per = {m: np.asarray([np.nan if r[field] is None else float(r[field]) for r in by_model[m]], dtype=np.float64) for m in MODELS}
        # Counts are GT-derived and must identify the same estimable images across models.
        masks = {m: np.isfinite(per[m]) for m in MODELS}
        if not all(np.array_equal(masks["E0"], masks[m]) for m in MODELS[1:]):
            raise ValueError(f"Image-mean estimable set differs across models for {metric}")
        eligible = masks["E0"]
        if not eligible.any():
            continue
        # Resample only the common complete-image set; this is an image-average estimand,
        # distinct from the pooled sufficient-statistic estimand above.
        ids = np.flatnonzero(eligible)
        idx = rng.integers(0, len(ids), size=(repeats, len(ids)), endpoint=False)
        values = {m: np.take(per[m][ids], idx).mean(axis=1) for m in MODELS}
        for candidate, baseline in PAIRS:
            delta = values[candidate] - values[baseline]
            raw = per[candidate][ids] - per[baseline][ids]
            results.append({"estimand": "image_mean", "metric": metric, "candidate": candidate, "baseline": baseline,
                            "difference_candidate_minus_baseline": float(raw.mean()), "ci95_low": float(np.quantile(delta,.025)),
                            "ci95_high": float(np.quantile(delta,.975)), "bootstrap_repeats": repeats,
                            "seed": seed, "images": 128, "images_with_metric": int(len(ids)),
                            "same_resampled_images_all_models": True,
                            "improved_images_candidate_vs_baseline": int((raw < 0).sum()),
                            "worsened_images_candidate_vs_baseline": int((raw > 0).sum()),
                            "unchanged_images": int((raw == 0).sum())})
    return results


def condition_tables(conditions_path: Path, image_count_path: Path | None, output: Path) -> list[dict[str, Any]]:
    condition_rows = read_csv(conditions_path)
    key = ("scope", "stride_px", "group")
    indexed: dict[tuple[str, int, str], dict[str, dict[str, Any]]] = {}
    for raw in condition_rows:
        model = raw["model"]
        if model not in MODELS:
            raise ValueError(f"Unexpected model in condition CSV: {model}")
        k = (raw["scope"], int(raw["stride_px"]), raw["group"])
        indexed.setdefault(k, {})[model] = {
            "pair_count": int(raw["pair_count"]),
            "delta_mae_m": optional_float(raw["delta_mae_m"]),
            "delta_rmse_m": optional_float(raw["delta_rmse_m"]),
            "amplitude_ratio": optional_float(raw["amplitude_ratio"]),
            "sign_match_rate": optional_float(raw["sign_match_rate"]),
            "mean_abs_gt_delta_m": optional_float(raw["mean_abs_gt_delta_m"]),
            "mean_abs_pred_delta_m": optional_float(raw["mean_abs_pred_delta_m"]),
        }
    counts: dict[tuple[str, int, str], dict[str, int]] = {}
    if image_count_path and image_count_path.exists():
        for row in read_csv(image_count_path):
            counts[(row["scope"], int(row["stride_px"]), row["group"])] = {
                "image_count": int(row["image_count"]),
                "scope_image_count": int(row["scope_image_count"]),
            }
    output_rows = []
    for k in sorted(indexed):
        models = indexed[k]
        if set(models) != set(MODELS):
            raise ValueError(f"Condition lacks model rows: {k} {set(models)}")
        pair_counts = {models[m]["pair_count"] for m in MODELS}
        if len(pair_counts) != 1:
            raise ValueError(f"Pair counts differ across models: {k}")
        count_row = counts.get(k)
        row: dict[str, Any] = {
            "scope": k[0], "stride_px": k[1], "stratum": k[2],
            "image_count": count_row["image_count"] if count_row else None,
            "scope_image_count": count_row["scope_image_count"] if count_row else None,
            "pair_count": pair_counts.pop(),
            "image_count_status": "computed from fixed GT pairs" if count_row else "not recovered from compact aggregate",
        }
        for metric in ("delta_mae_m", "delta_rmse_m", "amplitude_ratio", "sign_match_rate"):
            for m in MODELS:
                row[f"{metric}_{m}"] = models[m][metric]
            for m, ref in (("E1", "E0"), ("E2", "E1"), ("E3", "E1"), ("E2", "E0"), ("E3", "E0")):
                left, right = models[m][metric], models[ref][metric]
                row[f"{metric}_{m}_minus_{ref}"] = left - right if left is not None and right is not None else None
        for m in MODELS:
            row[f"mean_abs_gt_delta_m_{m}"] = models[m]["mean_abs_gt_delta_m"]
            row[f"mean_abs_pred_delta_m_{m}"] = models[m]["mean_abs_pred_delta_m"]
        output_rows.append(row)
    write_csv(output / "condition_comparison_all.csv", output_rows)
    distances = [r for r in output_rows if r["scope"] == "all" and r["stratum"] in ("all", "gt_ge20cm")]
    write_csv(output / "distance_condition_comparison.csv", distances)
    bins = [r for r in output_rows if r["scope"] == "all" and r["stride_px"] == 14 and r["stratum"].startswith("gt_delta_")]
    write_csv(output / "gt_delta_bin_comparison.csv", bins)
    crop_view = [r for r in output_rows if r["stride_px"] == 14 and (r["scope"].startswith("crop=") or r["scope"].startswith("view=")) and r["stratum"] in ("all", "gt_ge20cm")]
    write_csv(output / "crop_view_comparison.csv", crop_view)
    other = [r for r in output_rows if r["scope"] == "all" and r["stride_px"] == 14 and r["stratum"] not in ("all", "gt_ge20cm") and not r["stratum"].startswith("gt_delta_")]
    write_csv(output / "edge_depth_sign_comparison.csv", other)
    return output_rows


def make_figures(output: Path, summary_path: Path, conditions: list[dict[str, Any]], bootstrap: list[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Yu Gothic"
    summary = {r["group"]: r for r in json.loads(summary_path.read_text(encoding="utf-8"))["selected"]}
    colors = {"E0": "#555555", "E1": "#2878B5", "E2": "#F28E2B", "E3": "#C23B22"}
    order = list(MODELS)
    # Main metrics: preserve units and keep pool and fixed-sample definitions explicit.
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6))
    specs = [("rmse_m", "絶対RMSE (m)"), ("full14_mae_m", "14px全有効ペア MAE (m)"), ("gt_ge20cm_14_mae_m", "14px固定標本・GT差≥0.20m MAE (m)")]
    for ax, (field, title) in zip(axes, specs):
        values = [float(summary[m][field]) for m in order]
        bars = ax.bar(order, values, color=[colors[m] for m in order])
        ax.set_title(title); ax.grid(axis="y", alpha=.25)
        ax.bar_label(bars, fmt="%.4f", fontsize=8, padding=2)
    fig.suptitle("E0〜E3 Validation128 比較（B0全体518px）", y=.985)
    fig.text(.5, .015, "絶対指標 n=2,668,757,375 pixels；14px全有効ペア n=5,320,891,492；大差分固定標本 n=1,911,836 pairs", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .07, 1, .93)); fig.savefig(output / "main_metrics_comparison.svg", bbox_inches="tight", pad_inches=.08); fig.savefig(output / "main_metrics_comparison.png", dpi=180, bbox_inches="tight", pad_inches=.08); plt.close(fig)
    # Distance scale table: all sampled pairs and the large-difference group remain separate.
    by_key = {(r["scope"], int(r["stride_px"]), r["stratum"]): r for r in conditions}
    scales = [14, 28, 56, 107, 214]
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for model in ("E0", "E1", "E2", "E3"):
        vals = [by_key[("all", s, "gt_ge20cm")][f"delta_mae_m_{model}"] for s in scales]
        ax.plot(scales, vals, marker="o", label=model, color=colors[model])
    pair_counts = [by_key[("all", s, "gt_ge20cm")]["pair_count"] for s in scales]
    ax.set_xticks(scales, [f"{s}\n(n={n:,})" for s, n in zip(scales, pair_counts)])
    ax.set_xlabel("元画像上の画素間隔 / 固定標本数"); ax.set_ylabel("局所差分MAE (m)"); ax.set_title("GT差分≥0.20m 固定標本の距離別MAE")
    ax.grid(alpha=.25); ax.legend(title="群"); fig.tight_layout(); fig.savefig(output / "distance_large_pair_mae.svg"); fig.savefig(output / "distance_large_pair_mae.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for model in MODELS:
        vals = [by_key[("all", s, "gt_ge20cm")][f"amplitude_ratio_{model}"] for s in scales]
        ax.plot(scales, vals, marker="o", label=model, color=colors[model])
    ax.axhline(1, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(scales, [f"{s}\n(n={n:,})" for s, n in zip(scales, pair_counts)])
    ax.set_xlabel("元画像上の画素間隔 / 固定標本数"); ax.set_ylabel("振幅比 Σ|Δ予測| / Σ|ΔGT|"); ax.set_title("GT差分≥0.20m 固定標本の振幅比")
    ax.grid(alpha=.25); ax.legend(title="群"); fig.tight_layout(); fig.savefig(output / "amplitude_ratio_by_distance.svg"); fig.savefig(output / "amplitude_ratio_by_distance.png", dpi=180); plt.close(fig)
    bins = [r for r in conditions if r["scope"] == "all" and int(r["stride_px"]) == 14 and r["stratum"].startswith("gt_delta_")]
    bins.sort(key=lambda r: float(r["stratum"].split("_")[2]))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(bins)); width = .19
    for j, model in enumerate(MODELS):
        ax.bar(x + (j-1.5)*width, [r[f"delta_mae_m_{model}"] for r in bins], width, label=model, color=colors[model])
    labels = [r["stratum"].replace("gt_delta_", "").replace("m", "") for r in bins]
    ax.set_xticks(x, labels); ax.set_xlabel("GT局所差分の絶対値階級 (m)"); ax.set_ylabel("局所差分MAE (m)"); ax.set_title("14px固定標本・GT差分階級別MAE")
    ax.grid(axis="y", alpha=.25); ax.legend(title="群"); fig.tight_layout(); fig.savefig(output / "gt_delta_bins.svg"); fig.savefig(output / "gt_delta_bins.png", dpi=180); plt.close(fig)
    crop_view = [r for r in conditions if r["stride_px"] == 14 and r["stratum"] == "gt_ge20cm" and (r["scope"].startswith("crop=") or r["scope"].startswith("view="))]
    crop_view.sort(key=lambda r: (0 if r["scope"].startswith("crop=") else 1, r["scope"]))
    fig, ax = plt.subplots(figsize=(11, 5.0))
    x = np.arange(len(crop_view)); width = .19
    for j, model in enumerate(MODELS):
        ax.bar(x + (j-1.5)*width, [r[f"delta_mae_m_{model}"] for r in crop_view], width, label=model, color=colors[model])
    labels = [f"{r['scope'].split('=',1)[1]}\n{r['image_count']}/{r['scope_image_count']} imgs\nn={r['pair_count']:,}" for r in crop_view]
    ax.set_xticks(x, labels); ax.set_xlabel("作物 / 撮影方向（ペアあり画像数 / 対象画像数）"); ax.set_ylabel("14px局所差分MAE (m)")
    ax.set_title("GT差分≥0.20m 固定標本の作物・視点別MAE"); ax.grid(axis="y", alpha=.25); ax.legend(title="群")
    fig.tight_layout(); fig.savefig(output / "crop_view_large_pair_mae.svg"); fig.savefig(output / "crop_view_large_pair_mae.png", dpi=180); plt.close(fig)
    # All three prespecified spatial/depth/sign condition families, with both loss groups
    # contrasted against the same SiLog-only continuation control.
    all14 = {(r["stratum"]): r for r in conditions if r["scope"] == "all" and int(r["stride_px"]) == 14}
    families = [
        ("画像端からの距離", ["border_lt128", "middle_128_512", "interior_ge512"]),
        ("ペア平均GT深度帯", ["depth_0_15m", "depth_15_20m", "depth_20_30m", "depth_30_65m"]),
        ("GT局所差分の符号", ["negative_gt", "zero_gt", "positive_gt"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 6.0))
    for ax, (title, strata) in zip(axes, families):
        subset = [all14[s] for s in strata]
        labels = [f"{r['stratum']}\nn={r['pair_count']:,}; imgs={r['image_count']}/{r['scope_image_count']}" for r in subset]
        y = np.arange(len(subset)); h = .34
        e2 = np.asarray([r["delta_mae_m_E2_minus_E1"] for r in subset], dtype=float)
        e3 = np.asarray([r["delta_mae_m_E3_minus_E1"] for r in subset], dtype=float)
        ax.barh(y - h/2, e2, h, label="E2−E1", color=colors["E2"])
        ax.barh(y + h/2, e3, h, label="E3−E1", color=colors["E3"])
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_yticks(y, labels); ax.invert_yaxis(); ax.set_xlabel("14px 局所MAE差 (m)")
        ax.set_title(title); ax.grid(axis="x", alpha=.25)
    axes[0].legend(title="比較", loc="best")
    fig.suptitle("全条件を表示：E2/E3 − E1（負値は局所MAE改善）")
    fig.tight_layout(); fig.savefig(output / "condition_families_e3_e2_vs_e1.svg"); fig.savefig(output / "condition_families_e3_e2_vs_e1.png", dpi=180); plt.close(fig)
    # Paired pooled bootstrap intervals for the preregistered primary local metric.
    rows = [r for r in bootstrap if r["estimand"] == "pooled" and r["metric"] in ("gt_ge20_local_mae_m", "rmse_m", "full14_local_mae_m")]
    labels = [f"{r['candidate']}−{r['baseline']}  {r['metric']}" for r in rows]
    fig, ax = plt.subplots(figsize=(8.4, max(4.5, len(rows)*.38)))
    y = np.arange(len(rows))
    means = np.asarray([r["difference_candidate_minus_baseline"] for r in rows])
    low = np.asarray([r["ci95_low"] for r in rows]); high = np.asarray([r["ci95_high"] for r in rows])
    ax.errorbar(means, y, xerr=[means-low, high-means], fmt="o", capsize=3, color="#2878B5")
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(y, labels); ax.invert_yaxis(); ax.set_xlabel("candidate − baseline (m); 負値は候補群が低誤差")
    ax.set_title("画像単位paired bootstrap（pool十分統計、95% percentile区間）"); ax.grid(axis="x", alpha=.25)
    fig.tight_layout(); fig.savefig(output / "paired_bootstrap_pooled.svg"); fig.savefig(output / "paired_bootstrap_pooled.png", dpi=180); plt.close(fig)
    # Faceted view keeps unlike units on separate axes while exposing every primary paired contrast.
    key_metrics = ["mae_m", "rmse_m", "absrel", "full14_local_mae_m", "gt_ge20_local_mae_m"]
    metric_labels = {
        "mae_m": "絶対MAE差 (m)", "rmse_m": "絶対RMSE差 (m)", "absrel": "AbsRel差",
        "full14_local_mae_m": "14px全有効ペアMAE差 (m)", "gt_ge20_local_mae_m": "14px固定・GT差≥0.20m MAE差 (m)",
    }
    contrasts = [("E3", "E1"), ("E2", "E1"), ("E3", "E0"), ("E2", "E0")]
    fig, axes = plt.subplots(1, len(key_metrics), figsize=(18, 5.2))
    for ax, metric in zip(axes, key_metrics):
        rows = [r for r in bootstrap if r["estimand"] == "pooled" and r["metric"] == metric]
        by_contrast = {(r["candidate"], r["baseline"]): r for r in rows}
        y = np.arange(len(contrasts))
        means = np.asarray([by_contrast[c]["difference_candidate_minus_baseline"] for c in contrasts])
        low = np.asarray([by_contrast[c]["ci95_low"] for c in contrasts])
        high = np.asarray([by_contrast[c]["ci95_high"] for c in contrasts])
        ax.errorbar(means, y, xerr=[means-low, high-means], fmt="o", capsize=3, color="#2878B5")
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_yticks(y, [f"{a}−{b}" for a, b in contrasts]); ax.invert_yaxis()
        ax.set_xlabel(metric_labels[metric]); ax.set_title(metric.replace("_", " "))
        ax.grid(axis="x", alpha=.25)
    fig.suptitle("画像単位paired bootstrap：主要指標のpool差と95%区間（負値は候補群が低誤差）")
    fig.tight_layout(); fig.savefig(output / "paired_bootstrap_key_metrics.svg"); fig.savefig(output / "paired_bootstrap_key_metrics.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    xs = [float(summary[m]["rmse_m"]) for m in MODELS]
    ys = [float(summary[m]["gt_ge20cm_14_mae_m"]) for m in MODELS]
    for m, xval, yval in zip(MODELS, xs, ys):
        ax.scatter(xval, yval, s=65, label=m, color=colors[m])
        offset = (-8, 5) if m == "E0" else (6, 5)
        align = "right" if m == "E0" else "left"
        ax.annotate(f"{m}\n({xval:.4f}, {yval:.4f})", (xval, yval), xytext=offset, textcoords="offset points", ha=align, fontsize=8)
    ax.set_xlabel("絶対深度RMSE (m；表示範囲をデータ近傍に拡大)")
    ax.set_ylabel("14px固定標本・GT差≥0.20m MAE (m；表示範囲を拡大)")
    ax.set_title("絶対精度と局所差分精度のトレードオフ")
    ax.grid(alpha=.25); ax.legend(title="群")
    fig.text(.5, .015, "絶対評価 n=2,668,757,375 pixels；局所評価 n=1,911,836 fixed pairs。主比較図は0始点の尺度で併記。", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .06, 1, .94)); fig.savefig(output / "rmse_local_tradeoff.svg", bbox_inches="tight", pad_inches=.08); fig.savefig(output / "rmse_local_tradeoff.png", dpi=180, bbox_inches="tight", pad_inches=.08); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-image", type=Path, required=True, help="re-evaluation per_image_primary.csv")
    ap.add_argument("--conditions", type=Path, required=True, help="verified compact_condition_metrics.csv")
    ap.add_argument("--summary", type=Path, required=True, help="local analysis/summary.json")
    ap.add_argument("--image-counts", type=Path, help="optional fixed-GT-pair image-count table")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--repeats", type=int, default=10000)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    images = load_images(args.per_image)
    reconciliation = check_pool(images, args.summary)
    write_csv(args.output / "pool_reconciliation.csv", reconciliation)
    pool = pool_bootstrap(images, args.seed, args.repeats)
    mean = image_mean_bootstrap(images, args.seed, args.repeats)
    write_csv(args.output / "paired_bootstrap_pooled.csv", pool)
    write_csv(args.output / "paired_bootstrap_image_mean.csv", mean)
    bootstrap = pool + mean
    (args.output / "paired_bootstrap.json").write_text(json.dumps({
        "method": "Paired image-cluster bootstrap of Validation128 sufficient statistics",
        "repeats": args.repeats, "seed": args.seed, "confidence": "95% percentile",
        "shared_resampling": "Each replicate samples the same 128 image indices for all E0-E3; all four preregistered contrasts share the same draws.",
        "pool_estimands": "Recompute pooled absolute/local means, RMSE from squared-error sums, amplitude ratio from absolute-delta sums, and sign agreement from integer counts after resampling images.",
        "image_mean_estimands": "Arithmetic mean of per-image metrics over the common set of images with a defined metric; reported separately from pool estimands.",
        "scope_limit": "Day047 only. Images may share flight/day dependence. A single training seed and validation-based checkpoint selection make these intervals conditional and do not establish unseen-day generalization or remove selection uncertainty.",
        "comparisons": bootstrap,
    }, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    conditions = condition_tables(args.conditions, args.image_counts, args.output)
    make_figures(args.output, args.summary, conditions, pool)
    for svg_path in args.output.glob("*.svg"):
        strip_svg_line_padding(svg_path)
    pool_summary = [{"model": model, **aggregate(images[model])} for model in MODELS]
    write_csv(args.output / "pool_metrics_re_evaluated.csv", pool_summary)
    experiment_root = Path(__file__).resolve().parent.parent
    metadata = {
        "analysis_method": "No inference or training; reaggregate frozen final-model per-image additive sufficient statistics and fixed GT-derived condition summaries.",
        "models": list(MODELS), "validation_images": 128,
        "contrasts": [{"candidate": a, "baseline": b} for a, b in PAIRS],
        "bootstrap": {"repeats": args.repeats, "seed": args.seed, "interval": "95% percentile", "unit": "validation image cluster", "same_draws_across_models_per_estimand": True},
        "pool_reconciliation": "Passed at absolute tolerance 1e-7 for seven reported selected-checkpoint metrics; exact full14 and GT>=20 fixed pair counts checked.",
        "inputs": {
            "per_image_csv": {"path": str(args.per_image.resolve().relative_to(experiment_root)), "sha256": sha256(args.per_image)},
            "condition_metrics_csv": {"path": str(args.conditions.resolve().relative_to(experiment_root)), "sha256": sha256(args.conditions)},
            "selection_summary_json": {"path": str(args.summary.resolve().relative_to(experiment_root)), "sha256": sha256(args.summary)},
            "condition_image_counts_csv": ({"path": str(args.image_counts.resolve().relative_to(experiment_root)), "sha256": sha256(args.image_counts)} if args.image_counts and args.image_counts.exists() else None),
        },
        "script_sha256": sha256(Path(__file__).resolve()),
        "limits": ["Day047 is one flight date", "Single training seed", "Checkpoint selection used this validation set", "Intervals do not include model-seed or selection uncertainty"],
    }
    (args.output / "analysis_run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"models": pool_summary, "bootstrap_comparisons": len(bootstrap), "condition_rows": len(conditions), "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
