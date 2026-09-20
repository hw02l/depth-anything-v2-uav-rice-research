from __future__ import annotations

from pathlib import Path

import numpy as np


def depth_to_points(
    depth: np.ndarray,
    rgb: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    stride: int = 1,
    max_depth: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if depth.shape[:2] != rgb.shape[:2]:
        raise ValueError(f"RGB and depth shape mismatch: {rgb.shape} vs {depth.shape}")
    sampled_depth = depth[::stride, ::stride].astype(np.float32)
    sampled_rgb = rgb[::stride, ::stride, :3].astype(np.uint8)
    height, width = sampled_depth.shape
    yy, xx = np.mgrid[0:height, 0:width]
    xx = xx * stride
    yy = yy * stride
    valid = np.isfinite(sampled_depth) & (sampled_depth > 0)
    if max_depth is not None:
        valid &= sampled_depth <= max_depth
    z = sampled_depth[valid]
    x = (xx[valid] - cx) * z / fx
    y = (yy[valid] - cy) * z / fy
    points = np.column_stack((x, y, z)).astype(np.float32)
    colors = sampled_rgb[valid].astype(np.uint8)
    return points, colors


def write_ascii_ply(path: str | Path, points: np.ndarray, colors: np.ndarray) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if len(points) != len(colors):
        raise ValueError("points and colors must have the same length")
    with target.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for point, color in zip(points, colors):
            handle.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {int(color[0])} {int(color[1])} {int(color[2])}\n")
