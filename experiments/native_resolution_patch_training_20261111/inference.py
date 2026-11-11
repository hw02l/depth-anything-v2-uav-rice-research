from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from uav_metric_research import multiscale as m
from data_patch import IMAGENET_MEAN, IMAGENET_STD, round_multiple


@dataclass
class InferenceResult:
    prediction: np.ndarray
    coverage_count: np.ndarray
    seam_distance_px: np.ndarray | None
    boxes: list[tuple[int, int, int, int]]
    elapsed_seconds: float
    tile_seconds: float
    tile_count: int


def _input(rgb: np.ndarray, side: int) -> torch.Tensor:
    h, w = rgb.shape[:2]
    scale = side / min(h, w)
    rh, rw = round_multiple(h * scale), round_multiple(w * scale)
    resized = cv2.resize(rgb.astype(np.float32) / 255.0, (rw, rh), interpolation=cv2.INTER_CUBIC)
    normalized = (resized - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1))).float()


def _predict(model, tensor: torch.Tensor, device: str) -> torch.Tensor:
    output = model(pixel_values=tensor[None].to(device)).predicted_depth
    if output.ndim != 3 or output.shape[0] != 1:
        raise ValueError(f"Unexpected depth output shape: {tuple(output.shape)}")
    return output[0].float()


def _seam_distance(shape: tuple[int, int], boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    h, w = shape
    yy = np.arange(h, dtype=np.float32)[:, None]
    xx = np.arange(w, dtype=np.float32)[None, :]
    distance = np.full((h, w), np.inf, dtype=np.float32)
    y_lines, x_lines = set(), set()
    for top, left, bottom, right in boxes:
        if 0 < top < h: y_lines.add(top)
        if 0 < bottom < h: y_lines.add(bottom)
        if 0 < left < w: x_lines.add(left)
        if 0 < right < w: x_lines.add(right)
    for y in y_lines:
        distance = np.minimum(distance, np.abs(yy - y))
    for x in x_lines:
        distance = np.minimum(distance, np.abs(xx - x))
    return distance


def infer_depth(model, rgb: np.ndarray, device: str, method: str) -> InferenceResult:
    if method not in ("P0", "P1", "P2"):
        raise ValueError(f"Unknown inference method {method}")
    h, w = rgb.shape[:2]
    if method == "P0":
        tensor = m.preprocess(rgb, 518)
        start = time.perf_counter()
        if device.startswith("cuda"): torch.cuda.synchronize()
        with torch.inference_mode():
            native = _predict(model, tensor, device).cpu()
        if device.startswith("cuda"): torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        pred = m.restore(native.numpy(), (h, w))
        count = np.ones((h, w), np.uint16)
        m.validate_prediction(pred, np.ones((h, w), np.float32))
        return InferenceResult(pred, count, None, [(0, 0, h, w)], elapsed, elapsed, 1)

    tile_size = 1036 if method == "P1" else 518
    input_size = 518
    boxes = m.tiles((h, w), tile_size, 0.25)
    if not boxes:
        raise ValueError("No inference tiles")
    total = np.zeros((h, w), np.float32)
    weights = np.zeros((h, w), np.float32)
    coverage = np.zeros((h, w), np.uint16)
    tile_time = 0.0
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    start_all = time.perf_counter()
    with torch.inference_mode():
        for box in boxes:
            top, left, bottom, right = box
            crop = rgb[top:bottom, left:right]
            if crop.shape[:2] != (tile_size, tile_size):
                raise ValueError(f"Unexpected edge tile shape {crop.shape[:2]} for {box}")
            if method == "P1":
                tensor = _input(crop, input_size)
            else:
                normalized = (crop.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
                tensor = torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1))).float()
            if tensor.shape[-2:] != (input_size, input_size):
                raise ValueError(f"Tile preprocessing produced {tuple(tensor.shape[-2:])}")
            start = time.perf_counter()
            prediction = _predict(model, tensor, device)
            if method == "P1":
                prediction = F.interpolate(prediction[None, None], size=(tile_size, tile_size),
                                           mode="bilinear", align_corners=True)[0, 0]
            if device.startswith("cuda"): torch.cuda.synchronize()
            tile_time += time.perf_counter() - start
            prediction = prediction.cpu().numpy()
            if prediction.shape != (tile_size, tile_size) or not np.isfinite(prediction).all() or (prediction <= 0).any() or (prediction > 80.001).any():
                raise ValueError(f"Invalid tile prediction at {box}")
            weight = m.tile_weight((tile_size, tile_size))
            total[top:bottom, left:right] += prediction * weight
            weights[top:bottom, left:right] += weight
            coverage[top:bottom, left:right] += 1
    if device.startswith("cuda"): torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_all
    if (weights <= 0).any() or (coverage == 0).any():
        raise ValueError("Tile grid left pixels uncovered")
    prediction = total / weights
    if not np.isfinite(prediction).all() or (prediction <= 0).any() or (prediction > 80.001).any():
        raise ValueError("Invalid merged prediction")
    return InferenceResult(prediction, coverage, _seam_distance((h, w), boxes),
                           boxes, elapsed, tile_time, len(boxes))


def profile_tiles(result: InferenceResult) -> dict[str, Any]:
    count = result.coverage_count
    return {
        "tile_count": result.tile_count,
        "elapsed_seconds": result.elapsed_seconds,
        "tile_seconds": result.tile_seconds,
        "coverage_min": int(count.min()),
        "coverage_max": int(count.max()),
        "nonoverlap_fraction": float((count == 1).mean()),
        "overlap_fraction": float((count > 1).mean()),
        "boxes_yxyx": [list(box) for box in result.boxes],
    }
