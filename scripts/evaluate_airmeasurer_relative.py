from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import rasterio
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]


def corr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan"), float("nan")
    pearson = float(np.corrcoef(x, y)[0, 1])
    spearman = float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])
    return pearson, spearman


def metrics(gt: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    error = pred - gt
    pearson, spearman = corr(gt, pred)
    sst = float(np.sum((gt - np.mean(gt)) ** 2))
    r2 = float(1.0 - np.sum((gt - pred) ** 2) / sst) if sst > 0 else float("nan")
    return {"pixels": int(len(gt)), "mae_m": float(np.mean(np.abs(error))), "rmse_m": float(np.sqrt(np.mean(error**2))), "bias_m": float(np.mean(error)), "pearson": pearson, "spearman": spearman, "r2": r2}


def pair_coords(height: int, width: int, separation: int, count: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    total_h = height * max(0, width - separation)
    total_v = max(0, height - separation) * width
    h_count = min(count // 2, total_h)
    v_count = min(count - h_count, total_v)
    h = rng.choice(total_h, h_count, replace=False) if h_count else np.empty(0, dtype=np.int64)
    v = rng.choice(total_v, v_count, replace=False) if v_count else np.empty(0, dtype=np.int64)
    y1 = np.concatenate((h // max(1, width - separation), v // width))
    x1 = np.concatenate((h % max(1, width - separation), v % width))
    y2 = np.concatenate((y1[:h_count], y1[h_count:] + separation))
    x2 = np.concatenate((x1[:h_count] + separation, x1[h_count:]))
    return y1, x1, y2, x2


def difference_metrics(gt: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    error = pred - gt
    nonzero = np.abs(gt) > 1e-9
    pearson, spearman = corr(gt, pred)
    slope, intercept = np.linalg.lstsq(np.column_stack((gt, np.ones_like(gt))), pred, rcond=None)[0] if len(gt) > 1 and np.std(gt) > 0 else (float("nan"), float("nan"))
    fitted = slope * gt + intercept
    sst = float(np.sum((pred - np.mean(pred)) ** 2))
    r2 = float(1.0 - np.sum((pred - fitted) ** 2) / sst) if sst > 0 else float("nan")
    return {"pairs": int(len(gt)), "mean_abs_gt_difference_m": float(np.mean(np.abs(gt))), "mae_m": float(np.mean(np.abs(error))), "rmse_m": float(np.sqrt(np.mean(error**2))), "bias_m": float(np.mean(error)), "pearson": pearson, "spearman": spearman, "sign_agreement": float(np.mean(np.sign(gt[nonzero]) == np.sign(pred[nonzero]))) if np.any(nonzero) else float("nan"), "slope": float(slope), "intercept_m": float(intercept), "r2": r2}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DA2 relative geometry against AirMeasurer CHM")
    parser.add_argument("--selection-csv", default="results/airmeasurer/selected_dates.csv")
    parser.add_argument("--inference-dir", default="outputs/airmeasurer/inference")
    parser.add_argument("--chm-dir", default="outputs/airmeasurer/chm")
    parser.add_argument("--results-dir", default="results/metrics/airmeasurer")
    parser.add_argument("--pair-samples", type=int, default=200000)
    parser.add_argument("--pixel-samples", type=int, default=300000)
    args = parser.parse_args()
    with (ROOT / args.selection_csv).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result_dir = ROOT / args.results_dir
    result_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260920)
    pixel_rows = []
    bin_rows = []
    local_rows = []
    magnitude_rows = []
    baseline_rows = []
    manifests = []
    for row in rows:
        date = row["date"]
        relative = np.load(ROOT / args.inference_dir / date / f"{date}_relative_depth.npy").astype(np.float32)
        with rasterio.open(ROOT / args.chm_dir / date / f"{date}_chm_on_orthomosaic_grid.tif") as source:
            chm = source.read(1).astype(np.float32)
            chm[chm <= -9990] = np.nan
        with rasterio.open(ROOT / row["orthomosaic_path"]) as source:
            rgb = source.read([1, 2, 3])
        rgb_mask = np.any(rgb > 0, axis=0)
        valid = np.isfinite(relative) & np.isfinite(chm) & (chm >= 0) & rgb_mask
        indices = np.flatnonzero(valid)
        if len(indices) > args.pixel_samples:
            indices = rng.choice(indices, args.pixel_samples, replace=False)
        gt = chm.flat[indices].astype(np.float64)
        raw = relative.flat[indices].astype(np.float64)
        scale, shift = np.linalg.lstsq(np.column_stack((raw, np.ones_like(raw))), gt, rcond=None)[0]
        aligned = scale * raw + shift
        aligned_map = scale * relative.astype(np.float64) + shift
        raw_pearson, raw_spearman = corr(gt, raw)
        aligned_values = metrics(gt, aligned)
        constant = np.full_like(gt, np.median(gt))
        constant_values = metrics(gt, constant)
        pixel_rows.append({"date": date, "valid_pixels": int(valid.sum()), "sampled_pixels": len(gt), "chm_min_m": float(np.min(gt)), "chm_median_m": float(np.median(gt)), "chm_mean_m": float(np.mean(gt)), "chm_p95_m": float(np.percentile(gt, 95)), "chm_max_m": float(np.max(gt)), "relative_raw_pearson": raw_pearson, "relative_raw_spearman": raw_spearman, "scale": float(scale), "shift_m": float(shift), "aligned_mae_m": aligned_values["mae_m"], "aligned_rmse_m": aligned_values["rmse_m"], "aligned_bias_m": aligned_values["bias_m"], "aligned_r2": aligned_values["r2"]})
        baseline_rows.extend([
            {"date": date, "variant": "constant_chm_median", **constant_values},
            {"date": date, "variant": "relative_scale_shift_aligned", **aligned_values},
        ])
        height_bins = ((0.0, 0.1), (0.1, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0), (1.0, float("inf")))
        for lower, upper in height_bins:
            mask = (gt >= lower) & ((gt <= upper) if np.isinf(upper) else (gt < upper))
            if not np.any(mask):
                continue
            pearson, spearman = corr(gt[mask], aligned[mask])
            bin_rows.append({"date": date, "height_bin_m": f"{lower:.1f}+" if np.isinf(upper) else f"{lower:.1f}-{upper:.1f}", "height_lower_m": lower, "height_upper_m": None if np.isinf(upper) else upper, "pixels": int(mask.sum()), "pearson": pearson, "spearman": spearman, "aligned_mae_m": float(np.mean(np.abs(aligned[mask] - gt[mask]))), "aligned_bias_m": float(np.mean(aligned[mask] - gt[mask]))})
        pair_records = []
        for separation in (8, 16, 32, 64, 128):
            y1, x1, y2, x2 = pair_coords(chm.shape[0], chm.shape[1], separation, args.pair_samples, rng)
            pair_valid = valid[y1, x1] & valid[y2, x2]
            keep = np.flatnonzero(pair_valid)
            if len(keep) > args.pair_samples:
                keep = keep[: args.pair_samples]
            d_gt = chm[y1[keep], x1[keep]].astype(np.float64) - chm[y2[keep], x2[keep]].astype(np.float64)
            d_pred = aligned_map[y1[keep], x1[keep]] - aligned_map[y2[keep], x2[keep]]
            local_rows.append({"date": date, "pixel_separation": separation, **difference_metrics(d_gt, d_pred)})
            pair_records.append((d_gt, d_pred))
        if pair_records:
            all_d_gt = np.concatenate([record[0] for record in pair_records])
            all_d_pred = np.concatenate([record[1] for record in pair_records])
            magnitude_bins = ((0.1, 0.08, 0.12), (0.2, 0.16, 0.24), (0.5, 0.4, 0.6), (1.0, 0.8, 1.2))
            for target, lower, upper in magnitude_bins:
                mask = (np.abs(all_d_gt) >= lower) & (np.abs(all_d_gt) <= upper)
                if np.any(mask):
                    magnitude_rows.append({"date": date, "target_gt_difference_m": target, "lower_abs_gt_difference_m": lower, "upper_abs_gt_difference_m": upper, **difference_metrics(all_d_gt[mask], all_d_pred[mask])})
        manifests.append({"date": date, "valid_pixels": int(valid.sum()), "pixel_sample_count": len(gt), "scale": float(scale), "shift_m": float(shift), "orientation": "raw DA2 inverse-depth-like output used without sign inversion; high canopy is treated as near/highness proxy for this exploratory top-down comparison", "metric_warning": "orthomosaic input; no camera-frame metric depth evaluation"})
        np.save(ROOT / args.inference_dir / date / f"{date}_aligned_height_diagnostic.npy", aligned.astype(np.float32))
    write_rows = lambda name, records: write_csv(result_dir / name, records)
    write_rows("airmeasurer_pixel_metrics.csv", pixel_rows)
    write_rows("airmeasurer_height_bins.csv", bin_rows)
    write_rows("airmeasurer_local_height_difference.csv", local_rows)
    write_rows("airmeasurer_local_height_difference_by_magnitude.csv", magnitude_rows)
    write_rows("airmeasurer_baseline_comparison.csv", baseline_rows)
    plot_status = [{"status": "NOT_AVAILABLE", "reason": "The downloaded small-field shapefile contains four PointZ RTK control points, not plot boundaries or plot IDs; no plot-level CHM/height table was publicly matched to this testing imagery."}]
    write_rows("airmeasurer_plot_metrics.csv", plot_status)
    (result_dir / "airmeasurer_evaluation_manifest.json").write_text(json.dumps({"records": manifests, "chm_height_bins_m": [[0, 0.1], [0.1, 0.2], [0.2, 0.4], [0.4, 0.6], [0.6, 0.8], [0.8, 1.0], [1.0, None]], "raw_relative_not_meter_mae": True, "alignment": "per-date least-squares scale+shift from sampled CHM-valid pixels; diagnostic only", "plot_evaluation": "not performed"}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[airmeasurer] evaluated {len(rows)} date(s); wrote {result_dir}")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else ["status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
