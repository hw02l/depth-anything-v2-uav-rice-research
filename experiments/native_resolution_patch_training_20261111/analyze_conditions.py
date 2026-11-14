from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

import analyze_results as core


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty result: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def num(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    return float(value) if value not in (None, "", "None") else 0.0


def scopes(abs_rows: list[dict[str, str]]) -> list[tuple[str, str | None, str | None]]:
    crops = sorted({r["crop"] for r in abs_rows})
    views = sorted({r["view"] for r in abs_rows})
    result = [("all", None, None)]
    result.extend(("crop", crop, None) for crop in crops)
    result.extend(("view", None, view) for view in views)
    result.extend(("crop_view", crop, view) for crop in crops for view in views
                  if any(r["crop"] == crop and r["view"] == view for r in abs_rows))
    return result


def in_scope(row: dict[str, Any], image_meta: dict[int, dict[str, str]],
             scope: tuple[str, str | None, str | None]) -> bool:
    kind, crop, view = scope
    meta = row if "crop" in row else image_meta[int(row["image_index"])]
    return ((crop is None or meta["crop"] == crop)
            and (view is None or meta["view"] == view))


def abs_value(rows: list[dict[str, str]], metric: str, estimator: str) -> float:
    if estimator == "pooled":
        return core._pool_absolute(rows, metric)
    values = core._absolute_metric(rows, metric)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else float("nan")


def local_value(rows: list[dict[str, str]], metric: str, estimator: str) -> float:
    if estimator == "pooled":
        return core._pool_local(rows, metric)
    values = core._local_metric(rows, metric)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else float("nan")


def append_summary(out: list[dict[str, Any]], *, task: str, group: str, method: str,
                   scope: tuple[str, str | None, str | None], rows: list[dict[str, str]],
                   metrics: tuple[str, ...], local: bool, condition: str = "",
                   stride: str = "", delta_group: str = "") -> None:
    if not rows:
        return
    image_count = sum(num(r, "pair_count" if local else "valid_pixels") > 0 for r in rows)
    weight_count = (sum(num(r, "pair_count") for r in rows) if local
                    else sum(num(r, "valid_pixels") for r in rows))
    for metric in metrics:
        for estimator in ("pooled", "image_mean"):
            value = local_value(rows, metric, estimator) if local else abs_value(rows, metric, estimator)
            if not np.isfinite(value):
                continue
            out.append({
                "task": task, "training_group": group, "method": method,
                "scope": scope[0], "crop": scope[1] or "", "view": scope[2] or "",
                "condition": condition, "stride_px": stride, "delta_group": delta_group,
                "metric": metric, "estimator": estimator, "value": value,
                "image_count": image_count,
                "pair_count" if local else "valid_pixels": int(weight_count),
            })


def compare_pair(out: list[dict[str, Any]], *, task: str, method: str,
                 candidate: str, baseline: str,
                 scope: tuple[str, str | None, str | None],
                 base_rows: list[dict[str, str]], cand_rows: list[dict[str, str]],
                 metrics: tuple[str, ...], local: bool, condition: str = "",
                 stride: str = "", delta_group: str = "",
                 repeats: int, seed: int) -> None:
    bmap = {int(r["image_index"]): r for r in base_rows}
    cmap = {int(r["image_index"]): r for r in cand_rows}
    shared = sorted(set(bmap) & set(cmap))
    for metric in metrics:
        val = local_value if local else abs_value
        paired = []
        for index in shared:
            b, c = [bmap[index]], [cmap[index]]
            bv, cv = val(b, metric, "image_mean"), val(c, metric, "image_mean")
            if np.isfinite(bv) and np.isfinite(cv):
                paired.append(index)
        if len(paired) < 2:
            continue
        b_rows = [bmap[i] for i in paired]
        c_rows = [cmap[i] for i in paired]
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, len(paired), size=(repeats, len(paired)), dtype=np.int16)
        if local:
            bval = core._local_metric(b_rows, metric)
            cval = core._local_metric(c_rows, metric)
            bpool, cpool, pdiff, plo, phi = core._pooled_bootstrap(
                b_rows, c_rows, metric, draws, local=True)
        else:
            bval = core._absolute_metric(b_rows, metric)
            cval = core._absolute_metric(c_rows, metric)
            bpool, cpool, pdiff, plo, phi = core._pooled_bootstrap(
                b_rows, c_rows, metric, draws, local=False)
        image_delta, ilo, ihi = core._paired_delta_bootstrap(bval, cval, draws)
        for estimator, base, cand, delta, low, high in (
            ("image_mean", float(np.mean(bval)), float(np.mean(cval)), image_delta, ilo, ihi),
            ("pooled", bpool, cpool, pdiff, plo, phi),
        ):
            out.append({
                "task": task, "method": method, "candidate": candidate, "baseline": baseline,
                "scope": scope[0], "crop": scope[1] or "", "view": scope[2] or "",
                "condition": condition, "stride_px": stride, "delta_group": delta_group,
                "metric": metric, "estimator": estimator,
                "baseline_value": base, "candidate_value": cand,
                "candidate_minus_baseline": delta, "ci95_low": low, "ci95_high": high,
                "paired_images": len(paired), "repeats": repeats, "seed": seed,
            })


def analyze(eval_root: Path, output_dir: Path, repeats: int = 10000, seed: int = 42) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=False)
    cache: dict[tuple[str, str], dict[str, Any]] = {}
    for group in ("C0", "F1", "F2"):
        for method in ("P0", "P1", "P2"):
            folder = eval_root / f"{group}_{method}"
            result = json.loads((folder / f"evaluation_{group}_{method}.json").read_text(encoding="utf-8"))
            if result.get("complete") is not True or result.get("image_count") != 128:
                raise ValueError(f"Incomplete evaluation: {group}/{method}")
            abs_rows = read_csv(folder / f"per_image_absolute_{group}_{method}.csv")
            local_rows = read_csv(folder / f"per_image_local_{group}_{method}.csv")
            abs_rows.sort(key=lambda r: int(r["image_index"]))
            if len(abs_rows) != 128 or [int(r["image_index"]) for r in abs_rows] != list(range(128)):
                raise ValueError(f"Invalid image ordering/count: {group}/{method}")
            if len({r["image_relpath"] for r in abs_rows}) != 128:
                raise ValueError(f"Duplicate Validation image: {group}/{method}")
            boundary_path = folder / f"per_image_boundary_{group}_{method}.csv"
            boundary_rows = read_csv(boundary_path) if boundary_path.exists() else []
            meta = {int(r["image_index"]): r for r in abs_rows}
            for row in local_rows + boundary_rows:
                index = int(row["image_index"])
                if index not in meta:
                    raise ValueError(f"Condition row references unknown image {index}")
                row["crop"] = meta[index]["crop"]
                row["view"] = meta[index]["view"]
            cache[(group, method)] = {
                "abs": abs_rows, "local": local_rows, "boundary": boundary_rows,
                "meta": meta, "scopes": scopes(abs_rows),
            }

    summary: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    abs_metrics = ("mae", "rmse", "abs_rel", "bias", "pearson", "full14_mae")
    local_metrics = ("delta_mae", "delta_rmse", "amplitude_ratio", "sign_match_rate")

    for group in ("C0", "F1", "F2"):
        for method in ("P0", "P1", "P2"):
            data = cache[(group, method)]
            for scope in data["scopes"]:
                rows = [r for r in data["abs"] if in_scope(r, data["meta"], scope)]
                append_summary(summary, task="absolute", group=group, method=method,
                               scope=scope, rows=rows, metrics=abs_metrics, local=False)
                for stride in sorted({r["stride_px"] for r in data["local"]}, key=int):
                    for condition in sorted({r["group"] for r in data["local"] if r["stride_px"] == stride}):
                        selected = [r for r in data["local"] if r["stride_px"] == stride
                                    and r["group"] == condition and in_scope(r, data["meta"], scope)]
                        append_summary(summary, task="local", group=group, method=method,
                                       scope=scope, rows=selected, metrics=local_metrics, local=True,
                                       condition=condition, stride=stride)
                    for condition in sorted({r["condition"] for r in data["boundary"]
                                              if r["stride_px"] == stride}):
                        for delta_group in sorted({r["delta_group"] for r in data["boundary"]
                                                   if r["stride_px"] == stride
                                                   and r["condition"] == condition}):
                            selected = [r for r in data["boundary"]
                                        if r["stride_px"] == stride and r["condition"] == condition
                                        and r["delta_group"] == delta_group
                                        and in_scope(r, data["meta"], scope)]
                            append_summary(summary, task="boundary", group=group, method=method,
                                           scope=scope, rows=selected, metrics=local_metrics,
                                           local=True, condition=condition, stride=stride,
                                           delta_group=delta_group)

    for method in ("P0", "P1", "P2"):
        for candidate in ("F1", "F2"):
            baseline_data = cache[("C0", method)]
            candidate_data = cache[(candidate, method)]
            base_primary = {
                int(r["image_index"]): r for r in baseline_data["local"]
                if r["stride_px"] == "14" and r["group"] == "gt_ge20cm"
            }
            candidate_primary = {
                int(r["image_index"]): r for r in candidate_data["local"]
                if r["stride_px"] == "14" and r["group"] == "gt_ge20cm"
            }
            case_rows = []
            for image_index in sorted(set(base_primary) & set(candidate_primary)):
                b, c = base_primary[image_index], candidate_primary[image_index]
                if num(b, "pair_count") <= 0 or num(c, "pair_count") <= 0:
                    continue
                baseline_mae = num(b, "abs_delta_error_sum") / num(b, "pair_count")
                candidate_mae = num(c, "abs_delta_error_sum") / num(c, "pair_count")
                meta = baseline_data["meta"][image_index]
                case_rows.append({
                    "method": method, "candidate": candidate, "baseline": "C0",
                    "image_index": image_index, "image_relpath": meta["image_relpath"],
                    "crop": meta["crop"], "view": meta["view"],
                    "pair_count": int(num(b, "pair_count")),
                    "baseline_delta_mae_m": baseline_mae,
                    "candidate_delta_mae_m": candidate_mae,
                    "candidate_minus_baseline_m": candidate_mae - baseline_mae,
                })
            case_rows.sort(key=lambda r: (r["candidate_minus_baseline_m"], r["image_index"]))
            examples.extend({**r, "case": "largest_improvement"} for r in case_rows[:5])
            examples.extend({**r, "case": "largest_worsening"} for r in case_rows[-5:])
            for scope in baseline_data["scopes"]:
                b = [r for r in baseline_data["abs"] if in_scope(r, baseline_data["meta"], scope)]
                c = [r for r in candidate_data["abs"] if in_scope(r, candidate_data["meta"], scope)]
                compare_pair(comparisons, task="absolute", method=method, candidate=candidate,
                             baseline="C0", scope=scope, base_rows=b, cand_rows=c,
                             metrics=abs_metrics, local=False, repeats=repeats, seed=seed)
                for stride in sorted({r["stride_px"] for r in baseline_data["local"]}, key=int):
                    conditions = sorted({r["group"] for r in baseline_data["local"]
                                         if r["stride_px"] == stride})
                    for condition in conditions:
                        core_condition = condition in {"all", "gt_ge20cm"}
                        stratified_condition = (condition.startswith("gt_delta_")
                                                or condition.startswith("depth_")
                                                or condition in {"border_lt128", "middle_128_512", "interior_ge512"})
                        if scope[0] == "all":
                            if not (core_condition or stratified_condition):
                                continue
                        elif not (stride == "14" and core_condition):
                            continue
                        b = [r for r in baseline_data["local"] if r["stride_px"] == stride
                             and r["group"] == condition and in_scope(r, baseline_data["meta"], scope)]
                        c = [r for r in candidate_data["local"] if r["stride_px"] == stride
                             and r["group"] == condition and in_scope(r, candidate_data["meta"], scope)]
                        compare_pair(comparisons, task="local", method=method, candidate=candidate,
                                     baseline="C0", scope=scope, base_rows=b, cand_rows=c,
                                     metrics=local_metrics, local=True, condition=condition,
                                     stride=stride, repeats=repeats, seed=seed)
                    conditions = sorted({r["condition"] for r in baseline_data["boundary"]
                                         if r["stride_px"] == stride})
                    for condition in conditions:
                        if scope[0] != "all":
                            continue
                        delta_groups = sorted({r["delta_group"] for r in baseline_data["boundary"]
                                               if r["stride_px"] == stride
                                               and r["condition"] == condition})
                        for delta_group in delta_groups:
                            b = [r for r in baseline_data["boundary"] if r["stride_px"] == stride
                                 and r["condition"] == condition and r["delta_group"] == delta_group
                                 and in_scope(r, baseline_data["meta"], scope)]
                            c = [r for r in candidate_data["boundary"] if r["stride_px"] == stride
                                 and r["condition"] == condition and r["delta_group"] == delta_group
                                 and in_scope(r, candidate_data["meta"], scope)]
                            compare_pair(comparisons, task="boundary", method=method,
                                         candidate=candidate, baseline="C0", scope=scope,
                                         base_rows=b, cand_rows=c, metrics=local_metrics,
                                         local=True, condition=condition, stride=stride,
                                         delta_group=delta_group, repeats=repeats, seed=seed)

    summary_path = output_dir / "condition_summary.csv"
    comparison_path = output_dir / "paired_condition_bootstrap.csv"
    examples_path = output_dir / "paired_image_examples.csv"
    write_csv(summary_path, summary)
    write_csv(comparison_path, comparisons)
    write_csv(examples_path, examples)
    metadata = {
        "repeats": repeats, "seed": seed, "unit": "Validation image, paired",
        "image_count": 128, "day": "2025/Day047", "independent_test_used": False,
        "scopes": ["all", "crop", "view", "crop_view"],
        "local_conditions": "GT delta bins, GT depth bins, image-edge distance zones, endpoint zones",
        "boundary_conditions": "tile-seam distance and overlap coverage",
        "estimators": ["pooled", "image_mean"],
        "scope": "Day047 descriptive uncertainty only; no between-day inference",
    }
    metadata_path = output_dir / "condition_analysis_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"summary": summary_path, "bootstrap": comparison_path,
            "examples": examples_path, "metadata": metadata_path}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    paths = analyze(args.eval_root, args.output_dir, args.repeats, args.seed)
    print(json.dumps({name: str(path) for name, path in paths.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()


