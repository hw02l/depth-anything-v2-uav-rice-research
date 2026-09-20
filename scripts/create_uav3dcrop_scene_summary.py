from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def mean(rows: list[dict[str, str]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return sum(values) / len(values) if values else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine absolute and local UAV3DCrop metrics by scene")
    parser.add_argument("--results-dir", default="results/metrics/uav3dcrop_all")
    args = parser.parse_args()
    directory = ROOT / args.results_dir
    per_image = read_csv(directory / "uav3dcrop_per_image.csv")
    aligned = read_csv(directory / "uav3dcrop_bias_corrected.csv")
    local = [row for row in read_csv(directory / "local_difference_by_scene.csv") if row["pixel_separation"] == "32"]
    aligned_by_key = {(row["scene"], row["filename"]): row for row in aligned}
    local_by_key = {(row["scene"], row["variant"]): row for row in local}
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in per_image:
        groups[row["scene"]].append(row)

    output = []
    for scene in sorted(groups):
        rows = groups[scene]
        crop = rows[0]["crop"]
        nadir = sum(row["view_type"] == "nadir" for row in rows)
        oblique = sum(row["view_type"] == "oblique" for row in rows)
        aligned_rows = [aligned_by_key[(row["scene"], row["filename"])] for row in rows]
        raw_local = local_by_key[(scene, "metric_raw")]
        relative_local = local_by_key[(scene, "relative_scale_shift")]
        output.append({
            "scene": scene,
            "crop": crop,
            "images": len(rows),
            "nadir_images": nadir,
            "oblique_images": oblique,
            "raw_metric_mae_mean_m": mean(rows, "mae_m"),
            "raw_metric_rmse_mean_m": mean(rows, "rmse_m"),
            "raw_metric_absrel_mean": mean(rows, "absrel"),
            "raw_metric_bias_mean_m": mean(rows, "bias_m"),
            "scale_shift_mae_mean_m": mean(aligned_rows, "scale_shift_mae_m"),
            "relative_aligned_mae_mean_m": mean(aligned_rows, "relative_aligned_mae_m"),
            "local_metric_raw_mae_32_m": raw_local["local_difference_mae_m"],
            "local_metric_raw_pearson_32": raw_local["local_difference_pearson"],
            "local_metric_raw_sign_32": raw_local["local_difference_sign_agreement"],
            "local_relative_mae_32_m": relative_local["local_difference_mae_m"],
            "local_relative_pearson_32": relative_local["local_difference_pearson"],
            "local_relative_sign_32": relative_local["local_difference_sign_agreement"],
        })
    fields = list(output[0].keys())
    path = directory / "uav3dcrop_by_scene.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)
    print(f"[uav3dcrop] wrote {path}")


if __name__ == "__main__":
    main()
