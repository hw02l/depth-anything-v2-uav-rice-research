from __future__ import annotations

"""Reconcile C0 epoch20 P0/B3 with fixed C0 epoch5 and E3/B3 baselines."""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BEST = ROOT / "experiments" / "best_model_inference_combination_20261122"
sys.path.insert(0, str(BEST))
import analyze_results as stats  # noqa: E402


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def number(row: dict[str, str], key: str) -> float:
    value = row.get(key)
    if value in (None, "", "nan", "None"):
        return 0.0
    return float(value)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_ids(name: str, rows: list[dict]) -> None:
    expected = list(range(128))
    actual = [int(row["image_index"]) for row in rows]
    if actual != expected:
        raise ValueError(f"{name} must contain the same ordered Validation128 IDs 0..127")


def load_historical(path: Path) -> dict[str, list[dict]]:
    rows = read_rows(path)
    result: dict[str, list[dict]] = {}
    for condition in ("C0-P0", "C0-B3", "epoch20-B3", "E3-B3"):
        selected = [row for row in rows if row["condition"] == condition]
        records = [stats.record(
            int(row["image_index"]), int(row["valid_pixels"]),
            number(row, "abs_sum"), number(row, "sq_sum"),
            int(row["full_n"]), number(row, "full_sum"),
            int(row["high_n"]), number(row, "high_sum"),
        ) for row in selected]
        result[condition] = sorted(records, key=lambda row: row["image_index"])
        check_ids(condition, result[condition])
    return result


def load_p0(validation_dir: Path) -> list[dict]:
    absolute_path = validation_dir / "per_image_absolute_C0_P0.csv"
    local_path = validation_dir / "per_image_local_C0_P0.csv"
    if not absolute_path.is_file() or not local_path.is_file():
        raise FileNotFoundError(f"P0 sufficient statistics missing under {validation_dir}")
    high_by: dict[int, list[float]] = {}
    for row in read_rows(local_path):
        if int(row["stride_px"]) == 14 and row["group"] == "gt_ge20cm":
            index = int(row["image_index"])
            high_by[index] = [int(row["pair_count"]), number(row, "abs_delta_error_sum")]
    result = []
    for row in read_rows(absolute_path):
        index = int(row["image_index"])
        result.append(stats.record(
            index, int(row["valid_pixels"]), number(row, "abs_error_sum"),
            number(row, "squared_error_sum"), int(row["full14_pair_count"]),
            number(row, "full14_abs_delta_error_sum"), *high_by.get(index, [0, 0.0]),
            rel_sum=number(row, "abs_relative_sum"), bias=number(row, "bias_m"),
            pearson=number(row, "pearson_r"), pred_sum=number(row, "prediction_sum"),
            gt_sum=number(row, "target_sum"), pred_sq=number(row, "prediction_squared_sum"),
            gt_sq=number(row, "target_squared_sum"), pred_gt=number(row, "prediction_target_sum"),
        ))
    result.sort(key=lambda row: row["image_index"])
    check_ids("C0-epoch20-P0", result)
    evaluation_path = validation_dir / "evaluation_C0_P0.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    if (not evaluation.get("complete") or evaluation.get("group") != "C0"
            or int(evaluation.get("epoch", -1)) != 20):
        raise ValueError("P0 evaluation JSON is not the completed C0 epoch20 evaluation")
    for actual, expected, label in (
        (stats.estimate(result, "absolute_rmse", "pooled"), evaluation["absolute"]["rmse_m"], "RMSE"),
        (stats.estimate(result, "full14_mae", "pooled"), evaluation["absolute"]["full14_mae_m"], "full14 MAE"),
    ):
        if abs(actual - float(expected)) > 1e-12:
            raise ValueError(f"P0 sufficient statistics do not reconcile for {label}: {actual} != {expected}")
    expected_high = next(row["delta_mae_m"] for row in evaluation["sampled_local"]
                         if int(row["stride_px"]) == 14 and row["group"] == "gt_ge20cm")
    if abs(stats.estimate(result, "high14_mae", "pooled") - float(expected_high)) > 1e-12:
        raise ValueError("P0 high-delta local sufficient statistics do not reconcile")
    return result


def load_b3(output_dir: Path) -> list[dict]:
    run = json.loads((output_dir / "run_complete.json").read_text(encoding="utf-8"))
    if run.get("status") != "complete" or int(run.get("images", 0)) != 128:
        raise ValueError("C0 epoch20 B3 inference is not complete for Validation128")
    absolute_path = output_dir / "per_image_absolute_sufficient.csv"
    local_path = output_dir / "per_image_local_sufficient.csv"
    high_by: dict[int, list[float]] = {}
    for row in read_rows(local_path):
        if row.get("condition") == "C0-epoch20-B3" and int(row["stride_px"]) == 14 and row["group"] == "gt_ge20cm":
            index = int(row["image_index"])
            value = high_by.setdefault(index, [0, 0.0])
            value[0] += int(row["pair_count"])
            value[1] += number(row, "absolute_error_sum_m")
    result = []
    for row in read_rows(absolute_path):
        if row.get("condition") != "C0-epoch20-B3":
            continue
        index = int(row["image_index"])
        result.append(stats.record(
            index, int(row["valid_pixel_count"]), number(row, "abs_error_sum_m"),
            number(row, "squared_error_sum_m2"), int(row["full14_pair_count"]),
            number(row, "full14_abs_error_sum_m"), *high_by.get(index, [0, 0.0]),
            rel_sum=number(row, "abs_relative_error_sum"), bias=number(row, "bias_m"),
            pearson=number(row, "pearson_r"), pred_sum=number(row, "prediction_sum_m"),
            gt_sum=number(row, "gt_sum_m"), pred_sq=number(row, "prediction_squared_sum_m2"),
            gt_sq=number(row, "gt_squared_sum_m2"), pred_gt=number(row, "prediction_gt_cross_sum_m2"),
        ))
    result.sort(key=lambda row: row["image_index"])
    check_ids("C0-epoch20-B3", result)
    expected = run["summary"]
    for actual, key, label in (
        (stats.estimate(result, "absolute_rmse", "pooled"), "rmse_m", "RMSE"),
        (stats.estimate(result, "full14_mae", "pooled"), "full14_local_mae_m", "full14 MAE"),
        (stats.estimate(result, "high14_mae", "pooled"), "gt_ge20_14_local_mae_m", "high-delta MAE"),
    ):
        if abs(actual - float(expected[key])) > 1e-12:
            raise ValueError(f"B3 sufficient statistics do not reconcile for {label}: {actual} != {expected[key]}")
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args) -> None:
    historical = load_historical(args.historical_sufficient)
    p0 = load_p0(args.epoch20_p0_dir)
    b3 = load_b3(args.epoch20_b3_dir)
    conditions = {
        "C0-epoch5-P0": historical["C0-P0"],
        "C0-epoch5-B3": historical["C0-B3"],
        "C0-epoch20-P0": p0,
        "C0-epoch20-B3": b3,
        "epoch20-B3": historical["epoch20-B3"],
        "E3-epoch5-B3": historical["E3-B3"],
    }
    summaries = []
    for name, records in conditions.items():
        for metric in ("absolute_rmse", "full14_mae", "high14_mae"):
            summaries.append({
                "condition": name, "metric": metric,
                "pooled": stats.estimate(records, metric, "pooled"),
                "image_mean": stats.estimate(records, metric, "image_mean"),
                "valid_pixels": sum(row["valid_pixels"] for row in records),
                "full14_pairs": sum(row["full_n"] for row in records),
                "high14_pairs": sum(row["high_n"] for row in records),
                "images_with_high14_pairs": sum(row["high_n"] > 0 for row in records),
            })
    write_csv(args.output / "condition_summary.csv", summaries)
    per_image = [{"condition": name, **row} for name, records in conditions.items() for row in records]
    write_csv(args.output / "condition_per_image_sufficient.csv", per_image)

    comparisons = [
        ("C0-epoch20-P0", "C0-epoch5-P0"),
        ("C0-epoch20-B3", "C0-epoch5-B3"),
        ("C0-epoch20-B3", "C0-epoch20-P0"),
        ("C0-epoch20-B3", "E3-epoch5-B3"),
    ]
    bootstrap = []
    for a_name, b_name in comparisons:
        for metric in ("absolute_rmse", "full14_mae", "high14_mae"):
            for estimand in ("pooled", "image_mean"):
                bootstrap.append({
                    "comparison": f"{a_name} - {b_name}",
                    **stats.bootstrap_pair(conditions[a_name], conditions[b_name], metric,
                                           estimand, args.repeats, args.seed),
                })
    write_csv(args.output / "paired_bootstrap.csv", bootstrap)
    p0_json = args.epoch20_p0_dir / "evaluation_C0_P0.json"
    b3_provenance_path = args.epoch20_b3_dir / "provenance.json"
    p0_record = json.loads(p0_json.read_text(encoding="utf-8"))
    b3_record = json.loads(b3_provenance_path.read_text(encoding="utf-8"))
    provenance = {
        "script": str(Path(__file__).resolve()),
        "script_sha256": file_sha256(Path(__file__).resolve()),
        "historical_sufficient_path": str(args.historical_sufficient.resolve()),
        "historical_sufficient_sha256": file_sha256(args.historical_sufficient),
        "epoch20_p0_dir": str(args.epoch20_p0_dir.resolve()),
        "epoch20_b3_dir": str(args.epoch20_b3_dir.resolve()),
        "epoch20_p0_checkpoint_sha256": p0_record.get("checkpoint_sha256"),
        "epoch20_p0_evaluation_sha256": file_sha256(p0_json),
        "epoch20_p0_absolute_csv_sha256": file_sha256(args.epoch20_p0_dir / "per_image_absolute_C0_P0.csv"),
        "epoch20_p0_local_csv_sha256": file_sha256(args.epoch20_p0_dir / "per_image_local_C0_P0.csv"),
        "epoch20_b3_checkpoint_sha256": b3_record["checkpoint_state_load_audit"]["checkpoint_sha256"],
        "epoch20_b3_provenance_sha256": file_sha256(b3_provenance_path),
        "epoch20_b3_absolute_csv_sha256": file_sha256(args.epoch20_b3_dir / "per_image_absolute_sufficient.csv"),
        "epoch20_b3_local_csv_sha256": file_sha256(args.epoch20_b3_dir / "per_image_local_sufficient.csv"),
        "bootstrap_repeats": args.repeats,
        "seed": args.seed,
        "independent_test319_used": False,
        "paired_by_image_index": True,
        "pooled_and_image_mean_reported_separately": True,
    }
    (args.output / "analysis_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"conditions": len(conditions), "comparisons": len(comparisons),
                      "bootstrap_rows": len(bootstrap), "output": str(args.output)}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-sufficient", type=Path, required=True)
    parser.add_argument("--epoch20-p0-dir", type=Path, required=True)
    parser.add_argument("--epoch20-b3-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.repeats != 10_000 or args.seed != 42:
        raise ValueError("Pre-registered analysis requires exactly 10,000 repeats and seed 42")
    if args.output.exists():
        raise FileExistsError(f"Output already exists and is preserved: {args.output}")
    run(args)


if __name__ == "__main__":
    main()
