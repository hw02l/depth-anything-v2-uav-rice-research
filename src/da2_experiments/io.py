from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(path: str | Path, limit: int | None = None) -> list[Path]:
    root = Path(path)
    if root.is_file():
        paths = [root]
    else:
        paths = sorted(
            p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(f"No images found under {root}")
    return paths


def read_rgb(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def normalize_depth(depth: np.ndarray, lower: float = 1.0, upper: float = 99.0) -> np.ndarray:
    values = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(values) & (values > 0)
    if not np.any(valid):
        return np.zeros(values.shape, dtype=np.uint8)
    lo, hi = np.percentile(values[valid], [lower, upper])
    if hi <= lo:
        hi = lo + 1e-6
    scaled = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
    # Keep invalid NaN/Inf values out of the integer colormap indices; they are
    # painted black below and are never treated as valid depth.
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=1.0, neginf=0.0)
    # A compact blue-cyan-yellow-red palette, avoiding an optional matplotlib dependency.
    stops = np.array(
        [[12, 35, 120], [37, 152, 200], [245, 233, 80], [190, 25, 25]],
        dtype=np.float32,
    )
    position = scaled * (len(stops) - 1)
    left = np.floor(position).astype(np.int32)
    right = np.minimum(left + 1, len(stops) - 1)
    mix = (position - left)[..., None]
    rgb = stops[left] * (1.0 - mix) + stops[right] * mix
    rgb[~valid] = 0
    return np.clip(rgb, 0, 255).astype(np.uint8)


def save_depth_outputs(depth: np.ndarray, stem: str, outdir: str | Path, suffix: str) -> dict[str, str]:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    raw_path = out / f"{stem}_{suffix}.npy"
    vis_path = out / f"{stem}_{suffix}.png"
    np.save(raw_path, np.asarray(depth, dtype=np.float32))
    Image.fromarray(normalize_depth(depth), mode="RGB").save(vis_path)
    return {"raw": str(raw_path), "visualization": str(vis_path)}


def load_depth(path: str | Path, png_scale: float = 1.0) -> np.ndarray:
    source = Path(path)
    if source.suffix.lower() == ".npy":
        return np.load(source).astype(np.float32)
    with Image.open(source) as image:
        return np.asarray(image).astype(np.float32) / float(png_scale)


def write_json(path: str | Path, payload: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: str | Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
