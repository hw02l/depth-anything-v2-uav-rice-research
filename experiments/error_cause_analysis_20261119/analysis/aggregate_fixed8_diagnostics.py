#!/usr/bin/env python3
"""Aggregate already-computed fixed-eight diagnostic CSVs without inference.

The per-image/per-axis local metrics are combined using pair counts for pooled
MAE/RMSE. Regression slope, intercept, Pearson r, and amplitude ratio are
reported as image-axis macro summaries because the underlying cross-products
are not present in the exported CSV.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    # Existing diagnostic exports use both snake_case and human-readable
    # headers. Normalize separators once at the I/O boundary.
    return [{key.replace("_", " "): value for key, value in row.items()} for row in rows]


def number(row: dict[str, str], field: str) -> float:
    value = row[field]
    return float(value) if value not in ("", "nan", "NaN", "None") else math.nan


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(v for v in values if math.isfinite(v))
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    fraction = position - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def local_groups(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        common = (row["condition"], row["distance px"], row["subset"])
        groups[(*common, "fixed8", "all")].append(row)
        groups[(*common, "crop", row["crop"])].append(row)
        groups[(*common, "view", row["view"])].append(row)

    result: list[dict[str, object]] = []
    for (condition, distance, subset, group_type, group), members in sorted(groups.items()):
        weighted_n = sum(int(float(row["pair count"])) for row in members)
        if weighted_n == 0:
            continue
        abs_sum = sum(int(float(row["pair count"])) * number(row, "delta mae m") for row in members)
        sq_sum = sum(int(float(row["pair count"])) * number(row, "delta rmse m") ** 2 for row in members)
        sign_sum = sum(int(float(row["pair count"])) * number(row, "sign match rate") for row in members)

        def macro(field: str, statistic: str = "mean") -> float:
            values = [number(row, field) for row in members]
            values = [value for value in values if math.isfinite(value)]
            if not values:
                return math.nan
            return statistics.median(values) if statistic == "median" else statistics.fmean(values)

        result.append({
            "condition": condition,
            "distance_px": int(distance),
            "subset": subset,
            "group_type": group_type,
            "group": group,
            "image_axis_rows": len(members),
            "pair_count": weighted_n,
            "pooled_delta_mae_m": abs_sum / weighted_n,
            "pooled_delta_rmse_m": math.sqrt(sq_sum / weighted_n),
            "image_axis_mean_signed_slope": macro("signed slope"),
            "image_axis_median_signed_slope": macro("signed slope", "median"),
            "image_axis_mean_signed_intercept_m": macro("signed intercept m"),
            "image_axis_mean_pearson": macro("pearson"),
            "image_axis_mean_rms_amplitude_ratio": macro("rms amplitude ratio"),
            "pair_weighted_sign_match_rate": sign_sum / weighted_n,
        })
    return result


def absolute_groups(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        for group_type, group in (("fixed8", "all"), ("crop", row["crop"]), ("view", row["view"])):
            groups[(row["condition"], group_type, group)].append(row)

    result: list[dict[str, object]] = []
    for (condition, group_type, group), members in sorted(groups.items()):
        n = sum(int(float(row["valid pixels"])) for row in members)
        abs_sum = sum(int(float(row["valid pixels"])) * number(row, "mae m") for row in members)
        sq_sum = sum(int(float(row["valid pixels"])) * number(row, "rmse m") ** 2 for row in members)
        bias_sum = sum(int(float(row["valid pixels"])) * number(row, "bias m") for row in members)
        rel_sum = sum(int(float(row["valid pixels"])) * number(row, "absrel") for row in members)
        pearsons = [number(row, "pearson") for row in members if math.isfinite(number(row, "pearson"))]
        result.append({
            "condition": condition,
            "group_type": group_type,
            "group": group,
            "image_count": len(members),
            "valid_pixels": n,
            "pooled_mae_m": abs_sum / n,
            "pooled_rmse_m": math.sqrt(sq_sum / n),
            "pooled_bias_m": bias_sum / n,
            "pooled_absrel": rel_sum / n,
            "image_mean_pearson": statistics.fmean(pearsons) if pearsons else math.nan,
        })
    return result


def tile_groups(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    fields = (
        "overlap signed mean m", "overlap abs mean m", "overlap abs median m",
        "overlap abs p95 m", "pre fusion tile pixel mae m",
        "pre fusion tile pixel rmse m", "post fusion image mae m",
        "post fusion image rmse m",
    )
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["condition"]].append(row)
    result: list[dict[str, object]] = []
    for condition, members in sorted(groups.items()):
        output: dict[str, object] = {"condition": condition, "image_count": len(members)}
        output["overlap_pair_count"] = sum(int(float(row["overlap pair pixel count"])) for row in members)
        output["multi_tile_pixels"] = sum(int(float(row["multi tile pixels"])) for row in members)
        output["fusion_reconstruction_max_abs_m"] = max(number(row, "fusion reconstruction max abs m") for row in members)
        for field in fields:
            values = [number(row, field) for row in members if math.isfinite(number(row, field))]
            output[field.replace(" ", "_")] = statistics.fmean(values) if values else math.nan
            output[field.replace(" ", "_") + "_p95"] = quantile(values, 0.95)
        result.append(output)
    return result


def tile_fit_groups(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    fields = ("tile bias m", "tile gt slope", "tile gt intercept m")
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["condition"]].append(row)
    result: list[dict[str, object]] = []
    for condition, members in sorted(groups.items()):
        output: dict[str, object] = {"condition": condition, "tile_count": len(members)}
        for field in fields:
            values = [number(row, field) for row in members if math.isfinite(number(row, field))]
            name = field.replace(" ", "_")
            output[name + "_mean"] = statistics.fmean(values) if values else math.nan
            output[name + "_median"] = statistics.median(values) if values else math.nan
            output[name + "_p05"] = quantile(values, 0.05)
            output[name + "_p95"] = quantile(values, 0.95)
        result.append(output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = args.input_dir
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    absolute = read_csv(source / "fixed8_absolute_metrics.csv")
    local = read_csv(source / "fixed8_local_signed_delta_metrics.csv")
    tile = read_csv(source / "tile_overlap_metrics.csv")
    tile_fit = read_csv(source / "per_tile_gt_fit.csv")
    integrity = read_csv(source / "fixed8_input_integrity.csv")
    pair_manifest = read_csv(source / "fixed8_pair_sample_manifest.csv")

    expected = {"absolute": 40, "local": 800, "tile": 24, "tile_fit": 2520, "integrity": 8}
    actual = {"absolute": len(absolute), "local": len(local), "tile": len(tile), "tile_fit": len(tile_fit), "integrity": len(integrity)}
    if actual != expected:
        raise SystemExit(f"row-count mismatch: expected={expected}, actual={actual}")

    coordinate_hashes: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for row in pair_manifest:
        key = (row["image index"], row["axis"], row["distance px"])
        coordinate_hashes[key].add(row["coordinate sha256"])
    if len(pair_manifest) != 80 or any(len(values) != 1 for values in coordinate_hashes.values()):
        raise SystemExit("pair manifest is incomplete or coordinates differ across conditions")

    write_csv(output / "fixed8_grouped_local_metrics.csv", local_groups(local))
    write_csv(output / "fixed8_grouped_absolute_metrics.csv", absolute_groups(absolute))
    write_csv(output / "fixed8_grouped_tile_metrics.csv", tile_groups(tile))
    write_csv(output / "fixed8_grouped_tile_gt_fit.csv", tile_fit_groups(tile_fit))
    (output / "fixed8_aggregation_validation.txt").write_text(
        "\n".join([
            "fixed8 absolute rows=40 PASS",
            "fixed8 signed-local image-axis-subset rows=800 PASS",
            "fixed8 tile overlap image rows=24 PASS",
            "per-tile GT-fit rows=2520 PASS",
            "RGB/GT integrity rows=8 PASS",
            "pair manifest rows=80 PASS",
            f"unique image-axis-distance coordinate hashes={len(coordinate_hashes)} PASS",
            "pooled MAE/RMSE use pair-count-weighted sufficient summaries",
            "regression slope/intercept/Pearson and RMS-amplitude ratios are image-axis macro summaries",
            "fixed-eight descriptive sample only; Validation128 population estimates not claimed",
            "training=not run",
            "independent_test319=not used",
        ]) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
