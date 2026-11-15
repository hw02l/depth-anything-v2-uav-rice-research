from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from data_patch import coverage_report, read_manifest

SEED = 42
PILOT_RULE = "one image per crop x view; cycle target day in sorted order; choose minimum SHA256(seed|image_relpath) within the stratum"


def digest(seed: int, relpath: str) -> str:
    return hashlib.sha256(f"{seed}|{relpath}".encode("utf-8")).hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Prepare preregistered pilot manifests and data-only crop coverage.")
    parser.add_argument("--train-manifest", type=Path, default=ROOT / "configs/uav3dcrop_verified/train.csv")
    parser.add_argument("--val-manifest", type=Path, default=ROOT / "configs/uav3dcrop_verified/validation.csv")
    parser.add_argument("--previous-pilot-selection", type=Path,
                        default=ROOT / "experiments/multiscale_tile_inference_20261016/pilot_selection.csv")
    parser.add_argument("--output-dir", type=Path, default=HERE / "pilot")
    args = parser.parse_args()
    train = read_manifest(args.train_manifest)
    val = read_manifest(args.val_manifest)
    if len(train) != 384 or len(val) != 128:
        raise ValueError("Canonical Train384 and Validation128 are required")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)

    days = sorted({row["day"] for row in train})
    groups = sorted({(row["crop"], row["view"]) for row in train})
    selected = []
    for index, (crop, view) in enumerate(groups):
        target_day = days[index % len(days)]
        candidates = [r for r in train if r["crop"] == crop and r["view"] == view and r["day"] == target_day]
        if not candidates:
            candidates = [r for r in train if r["crop"] == crop and r["view"] == view]
        if not candidates:
            raise ValueError(f"Missing train stratum {crop}/{view}")
        selected.append(min(candidates, key=lambda r: digest(SEED, r["image_relpath"])))
    if len(selected) != 8 or len({r["image_relpath"] for r in selected}) != 8:
        raise ValueError("Pilot must contain eight unique crop/view-stratified train images")
    write_csv(out / "pilot_train_manifest.csv", selected)
    (out / "pilot_train_rows.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")

    with args.previous_pilot_selection.open("r", encoding="utf-8-sig", newline="") as stream:
        source_pilot = list(csv.DictReader(stream))
    if len(source_pilot) != 16 or len({r["image_relpath"] for r in source_pilot}) != 16:
        raise ValueError("The existing prespecified Validation pilot must contain 16 unique images")
    selected_paths = [r["image_relpath"] for r in source_pilot]
    val_by_path = {r["image_relpath"]: r for r in val}
    if not set(selected_paths).issubset(val_by_path):
        raise ValueError("Existing pilot selection is not a subset of canonical Validation128")
    pilot_val = [val_by_path[path] for path in selected_paths]
    write_csv(out / "pilot_validation_manifest.csv", pilot_val)
    (out / "pilot_validation_rows.json").write_text(json.dumps(pilot_val, ensure_ascii=False, indent=2), encoding="utf-8")

    coverage_rows = []
    for repeats in (1, 2):
        values = coverage_report(train, shape=(3956, 5280), seed=SEED, epochs=5, repeats=repeats)
        for row in values:
            coverage_rows.append({
                "optimizer_updates_candidate": 480 if repeats == 1 else 960,
                "crops_per_image_per_epoch": repeats,
                **row,
            })
    write_csv(out / "crop_coverage_candidates.csv", coverage_rows)
    summaries = {}
    for updates, repeats in ((480, 1), (960, 2)):
        values = np.asarray([r["unique_coverage_fraction"] for r in coverage_rows
                             if r["optimizer_updates_candidate"] == updates], np.float64)
        summaries[str(updates)] = {
            "images": int(values.size), "median_unique_coverage": float(np.median(values)),
            "p10": float(np.quantile(values, 0.10)), "p90": float(np.quantile(values, 0.90)),
            "min": float(values.min()), "max": float(values.max()),
            "crops_per_image": repeats * 5,
        }
    recommendation = 480 if summaries["480"]["median_unique_coverage"] >= 0.10 else 960
    lock = {
        "status": "data_only_candidate_recommendation; final config lock follows GPU pilot",
        "seed": SEED, "pilot_train_selection_rule": PILOT_RULE,
        "pilot_train_images": [r["image_relpath"] for r in selected],
        "pilot_train_days": sorted({r["day"] for r in selected}),
        "pilot_validation_images": selected_paths,
        "pilot_validation_selection_source": str(args.previous_pilot_selection),
        "budget_rule": "Choose 480 if F2 median unique native-pixel coverage >=10%; otherwise 960; apply equally to all groups.",
        "coverage_candidates": summaries,
        "recommended_common_optimizer_updates": recommendation,
        "shape_basis": [3956, 5280],
        "independent_test_used": False,
    }
    (out / "pilot_and_budget_prereg.json").write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(lock, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
