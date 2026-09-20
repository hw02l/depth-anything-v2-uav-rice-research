from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_local_difference_magnitude import fit_scale_shift, regression_metrics, sample_difference_pair  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute local-difference regression slopes by nadir/oblique view")
    parser.add_argument("--selection-csv", default="results/uav3dcrop/all_selected_images.csv")
    parser.add_argument("--output-root", default="outputs/uav3dcrop")
    parser.add_argument("--results-dir", default="results/metrics/uav3dcrop_all")
    parser.add_argument("--pair-samples", type=int, default=20000)
    args = parser.parse_args()
    with (ROOT / args.selection_csv).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rng = np.random.default_rng(20260920)
    grouped: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        stem = Path(row["filename"]).stem
        directory = ROOT / args.output_root / row["scene"].replace("/", "_") / row["view_type"] / stem
        pred = np.load(directory / f"{stem}_metric_depth_m.npy").astype(np.float32)
        relative = np.load(directory / f"{stem}_relative_depth.npy").astype(np.float32)
        gt = tifffile.imread(ROOT / row["local_depth_path"]).astype(np.float32)
        valid = np.isfinite(pred) & (pred > 0) & np.isfinite(relative) & np.isfinite(gt) & (gt > 0)
        bias = float(np.median((pred - gt)[valid]))
        corrected = pred - bias
        scale, shift = fit_scale_shift(pred, gt, valid, rng)
        relative_scale, relative_shift = fit_scale_shift(-relative, gt, valid, rng)
        aligned = scale * pred + shift
        relative_aligned = relative_scale * (-relative) + relative_shift
        d_gt, d_pred = sample_difference_pair(
            gt,
            {"metric_raw": pred, "metric_scale_shift": aligned, "relative_scale_shift": relative_aligned},
            valid,
            32,
            args.pair_samples,
            rng,
        )
        for variant in ("metric_raw", "metric_scale_shift", "relative_scale_shift"):
            grouped[row["view_type"]][variant].append((d_gt, d_pred[variant]))

    output = []
    for view in sorted(grouped):
        for variant in ("metric_raw", "metric_scale_shift", "relative_scale_shift"):
            x = np.concatenate([item[0] for item in grouped[view][variant]])
            y = np.concatenate([item[1] for item in grouped[view][variant]])
            slope, intercept, r2 = regression_metrics(x, y)
            error = y - x
            output.append({
                "view_type": view,
                "variant": variant,
                "pixel_separation": 32,
                "pairs": len(x),
                "local_difference_mae_m": float(np.mean(np.abs(error))),
                "local_difference_rmse_m": float(np.sqrt(np.mean(error**2))),
                "local_difference_pearson": float(np.corrcoef(x, y)[0, 1]),
                "sign_agreement": float(np.mean(np.sign(x[np.abs(x) > 1e-9]) == np.sign(y[np.abs(x) > 1e-9]))),
                "slope": slope,
                "intercept_m": intercept,
                "r2": r2,
            })
    path = ROOT / args.results_dir / "local_difference_view_slope.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0].keys()))
        writer.writeheader()
        writer.writerows(output)
    print(f"[uav3dcrop] wrote {path}")


if __name__ == "__main__":
    main()
