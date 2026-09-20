from __future__ import annotations

from pathlib import Path
import re

import numpy as np


def read_eth3d_calib(path: str | Path) -> dict[str, float]:
    """Read ETH3D low-res two-view calibration and return cam0 intrinsics."""
    text = Path(path).read_text(encoding="utf-8")
    match = re.search(r"cam0=\[([^;]+);\s*([^;]+);\s*([^\]]+)\]", text)
    if not match:
        raise ValueError(f"Could not find cam0 matrix in {path}")
    row0 = [float(value) for value in match.group(1).split()]
    row1 = [float(value) for value in match.group(2).split()]
    baseline_match = re.search(r"baseline=([0-9.eE+-]+)", text)
    doffs_match = re.search(r"doffs=([0-9.eE+-]+)", text)
    return {
        "fx": row0[0],
        "fy": row1[1],
        "cx": row0[2],
        "cy": row1[2],
        "baseline_mm": float(baseline_match.group(1)) if baseline_match else float("nan"),
        "doffs_px": float(doffs_match.group(1)) if doffs_match else 0.0,
    }


def read_pfm(path: str | Path) -> np.ndarray:
    """Read a floating point PFM image, including ETH3D's little-endian files."""
    with Path(path).open("rb") as handle:
        header = handle.readline().decode("ascii").strip()
        if header not in {"Pf", "PF"}:
            raise ValueError(f"Unsupported PFM header: {header}")
        dimensions = handle.readline().decode("ascii").strip()
        while dimensions.startswith("#"):
            dimensions = handle.readline().decode("ascii").strip()
        width, height = (int(value) for value in dimensions.split())
        scale = float(handle.readline().decode("ascii").strip())
        data = np.fromfile(handle, dtype="<f4" if scale < 0 else ">f4")
    channels = 3 if header == "PF" else 1
    expected = width * height * channels
    if data.size != expected:
        raise ValueError(f"PFM payload has {data.size} values; expected {expected}")
    shape = (height, width, channels) if channels == 3 else (height, width)
    return np.flipud(data.reshape(shape)).astype(np.float32)


def disparity_to_depth_m(
    disparity_px: np.ndarray,
    fx_px: float,
    baseline_mm: float,
    doffs_px: float = 0.0,
) -> np.ndarray:
    """Convert ETH3D rectified disparity to Z depth in metres."""
    disparity = np.asarray(disparity_px, dtype=np.float32)
    depth = np.full(disparity.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(disparity) & (disparity + doffs_px > 0)
    depth[valid] = (fx_px * baseline_mm / (disparity[valid] + doffs_px)) / 1000.0
    return depth
