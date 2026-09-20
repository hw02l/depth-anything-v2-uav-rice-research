from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from PIL import Image

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from da2_experiments.io import normalize_depth  # noqa: E402
from da2_experiments.modeling import RELATIVE_MODEL_ID, load_estimator  # noqa: E402


def read_rgb(path: Path) -> np.ndarray:
    with rasterio.open(path) as source:
        rgb = source.read([1, 2, 3])
    return np.moveaxis(rgb, 0, -1).astype(np.uint8)


def tile_predict(rgb: np.ndarray, estimator, tile_size: int, overlap: int) -> np.ndarray:
    height, width = rgb.shape[:2]
    if tile_size <= 0:
        return estimator.predict(Image.fromarray(rgb, mode="RGB"))
    step = max(1, tile_size - overlap)
    result = np.zeros((height, width), dtype=np.float64)
    weights = np.zeros((height, width), dtype=np.float64)
    for y0 in range(0, height, step):
        for x0 in range(0, width, step):
            y1 = min(height, y0 + tile_size)
            x1 = min(width, x0 + tile_size)
            tile = rgb[y0:y1, x0:x1]
            prediction = estimator.predict(Image.fromarray(tile, mode="RGB"))
            tile_h, tile_w = prediction.shape
            wy = np.ones(tile_h, dtype=np.float64) if tile_h <= 2 else np.maximum(np.hanning(tile_h), 0.1)
            wx = np.ones(tile_w, dtype=np.float64) if tile_w <= 2 else np.maximum(np.hanning(tile_w), 0.1)
            window = wy[:, None] * wx[None, :]
            result[y0:y1, x0:x1] += prediction.astype(np.float64) * window
            weights[y0:y1, x0:x1] += window
            print(f"[airmeasurer] relative tile y={y0}:{y1}, x={x0}:{x1}")
            if x1 == width:
                break
        if y1 == height:
            break
    return (result / np.maximum(weights, 1e-12)).astype(np.float32)


def save_overview(rgb: np.ndarray, relative: np.ndarray, chm: np.ndarray | None, path: Path) -> None:
    stride = max(1, int(max(rgb.shape[:2]) / 1000))
    rgb_small = rgb[::stride, ::stride]
    rel_small = relative[::stride, ::stride]
    rel_limits = np.percentile(relative[np.isfinite(relative)], [1, 99])
    panels = [(rgb_small, None, "Orthomosaic RGB"), (rel_small, "turbo", "DA2 relative depth")]
    if chm is not None:
        panels.append((chm[::stride, ::stride], "viridis", "CHM height (m)"))
    figure, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 7), squeeze=False)
    for axis, (image, cmap, title) in zip(axes[0], panels):
        if cmap is None:
            axis.imshow(image)
        elif title.startswith("DA2"):
            plot = axis.imshow(image, cmap=cmap, vmin=float(rel_limits[0]), vmax=float(rel_limits[1]))
            figure.colorbar(plot, ax=axis, fraction=0.046, pad=0.04)
        else:
            valid = np.isfinite(image)
            limits = np.percentile(image[valid], [1, 99]) if np.any(valid) else (0, 1)
            plot = axis.imshow(image, cmap=cmap, vmin=float(limits[0]), vmax=float(limits[1]))
            figure.colorbar(plot, ax=axis, fraction=0.046, pad=0.04)
        axis.set_title(title)
        axis.set_xlabel("orthomosaic pixel x")
        axis.set_ylabel("orthomosaic pixel y")
    figure.suptitle("AirMeasurer 20220505: RGB / relative depth / CHM")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DA2 relative depth on georeferenced AirMeasurer orthomosaics")
    parser.add_argument("--selection-csv", default="results/airmeasurer/selected_dates.csv")
    parser.add_argument("--output-dir", default="outputs/airmeasurer/inference")
    parser.add_argument("--chm-dir", default="outputs/airmeasurer/chm")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=128)
    args = parser.parse_args()
    with (ROOT / args.selection_csv).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    estimator = load_estimator("relative", RELATIVE_MODEL_ID, args.device)
    output_root = ROOT / args.output_dir
    records = []
    for row in rows:
        date = row["date"]
        rgb = read_rgb(ROOT / row["orthomosaic_path"])
        relative = tile_predict(rgb, estimator, args.tile_size, args.overlap)
        date_dir = output_root / date
        date_dir.mkdir(parents=True, exist_ok=True)
        np.save(date_dir / f"{date}_relative_depth.npy", relative.astype(np.float32))
        Image.fromarray(normalize_depth(relative), mode="RGB").save(date_dir / f"{date}_relative_depth.png")
        chm_path = ROOT / args.chm_dir / date / f"{date}_chm_on_orthomosaic_grid.tif"
        chm = None
        if chm_path.exists():
            with rasterio.open(chm_path) as source:
                chm = source.read(1).astype(np.float32)
                chm[chm <= -9990] = np.nan
        overview = date_dir / f"{date}_rgb_relative_chm.png"
        save_overview(rgb, relative, chm, overview)
        metadata = {"date": date, "model_id": RELATIVE_MODEL_ID, "device": str(estimator.device), "input_shape": list(rgb.shape), "output_shape": list(relative.shape), "tile_size": args.tile_size, "overlap": args.overlap, "orientation": "Depth Anything V2 relative output is inverse-depth-like; near is large. No sign inversion was applied when comparing with top-down CHM highness proxy.", "orthomosaic_warning": "The input is an orthomosaic, not a single pinhole camera frame; relative geometry is exploratory and not camera-frame metric depth."}
        (date_dir / "run.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        records.append(metadata)
        print(f"[airmeasurer] completed {date}: {relative.shape}")
    (output_root / "run_manifest.json").write_text(json.dumps({"model_id": RELATIVE_MODEL_ID, "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
