from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset

IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
GROUPS = {"C0": 518, "F1": 1036, "F2": 518}


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"split", "scene", "crop", "day", "view", "image_relpath", "depth_relpath"}
    if not rows or required - set(rows[0]):
        raise ValueError(f"Invalid manifest: {path}")
    return rows


def round_multiple(value: float, multiple: int = 14) -> int:
    return max(multiple, int(round(value / multiple)) * multiple)


def resized_shape(height: int, width: int, short_side: int = 518) -> tuple[int, int]:
    scale = short_side / min(height, width)
    return round_multiple(height * scale), round_multiple(width * scale)


def _draw_rng(seed: int, image_relpath: str, epoch: int, draw: int) -> np.random.Generator:
    payload = f"{seed}\0{image_relpath}\0{epoch}\0{draw}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return np.random.default_rng(value)


def draw_crop(row: dict[str, str], original_hw: tuple[int, int], group: str,
              seed: int, epoch: int, draw: int) -> dict[str, Any]:
    if group not in GROUPS:
        raise ValueError(f"Unknown training group: {group}")
    height, width = map(int, original_hw)
    rng = _draw_rng(seed, row["image_relpath"], epoch, draw)
    uy, ux = rng.random(2)
    flipped = bool(rng.random() < 0.5)
    if group == "C0":
        plane_hw = resized_shape(height, width)
        crop = 518
    else:
        plane_hw = (height, width)
        crop = GROUPS[group]
    ph, pw = plane_hw
    if min(ph, pw) < crop:
        raise ValueError(f"Image {original_hw} is too small for {group} crop {crop}")
    max_y, max_x = ph - crop, pw - crop
    top = min(int(uy * (max_y + 1)), max_y)
    left = min(int(ux * (max_x + 1)), max_x)
    if group == "C0":
        sy, sx = height / ph, width / pw
        footprint = [top * sy, left * sx, (top + crop) * sy, (left + crop) * sx]
        scale_yx = [sy, sx]
        space = "global_resized_image"
    else:
        footprint = [top, left, top + crop, left + crop]
        scale = crop / 518
        scale_yx = [scale, scale]
        space = "original_image"
    return {
        "group": group, "crop_box_yxyx": [top, left, top + crop, left + crop],
        "crop_space": space, "original_hw": [height, width],
        "footprint_original_yxyx_float": footprint, "native_px_per_input_px_yx": scale_yx,
        "input_hw": [518, 518], "flipped_horizontal": flipped,
        "normalized_start_uv": [float(uy), float(ux)],
        "epoch": int(epoch), "draw": int(draw),
    }


def load_pair(row: dict[str, str], rgb_root: str | Path,
              depth_root: str | Path) -> tuple[np.ndarray, np.ndarray]:
    rgb_path = Path(rgb_root) / Path(row["image_relpath"])
    depth_path = Path(depth_root) / Path(row["depth_relpath"])
    bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(rgb_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    depth = np.asarray(tifffile.imread(depth_path), dtype=np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError(f"Depth must be a single 2D plane: {depth_path} {depth.shape}")
    if rgb.shape[:2] != depth.shape:
        raise ValueError(f"RGB/GT mismatch for {row['image_relpath']}: {rgb.shape[:2]} != {depth.shape}")
    return rgb, depth


class NativePatchDataset(Dataset):
    """C0/F1/F2 aligned crop loader; only F1 resizes after its native crop."""
    def __init__(self, rows: list[dict[str, str]], rgb_root: str | Path,
                 depth_root: str | Path, group: str, *, repeats: int = 1,
                 seed: int = 42, training: bool = True):
        if group not in GROUPS or repeats < 1:
            raise ValueError("Invalid group or repeat count")
        self.rows, self.rgb_root, self.depth_root = rows, Path(rgb_root), Path(depth_root)
        self.group, self.repeats, self.seed = group, int(repeats), int(seed)
        self.training = bool(training)
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.rows) * (self.repeats if self.training else 1)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.training:
            row_index, draw = divmod(int(index), self.repeats)
        else:
            row_index, draw = int(index), 0
        row = self.rows[row_index]
        rgb, depth = load_pair(row, self.rgb_root, self.depth_root)
        meta = draw_crop(row, depth.shape, self.group, self.seed, self.epoch, draw)
        top, left, bottom, right = meta["crop_box_yxyx"]
        if self.group == "C0":
            rh, rw = resized_shape(*depth.shape)
            rgb = cv2.resize(rgb, (rw, rh), interpolation=cv2.INTER_CUBIC)
            depth = cv2.resize(depth, (rw, rh), interpolation=cv2.INTER_NEAREST)
            rgb = rgb[top:bottom, left:right]
            depth = depth[top:bottom, left:right]
        else:
            rgb = rgb[top:bottom, left:right]
            depth = depth[top:bottom, left:right]
            if self.group == "F1":
                rgb = cv2.resize(rgb, (518, 518), interpolation=cv2.INTER_CUBIC)
                depth = cv2.resize(depth, (518, 518), interpolation=cv2.INTER_NEAREST)
        if rgb.shape[:2] != (518, 518) or depth.shape != (518, 518):
            raise ValueError(f"Unexpected model crop shapes: {rgb.shape}, {depth.shape}")
        if meta["flipped_horizontal"]:
            rgb, depth = rgb[:, ::-1], depth[:, ::-1]
        normalized = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        valid = np.isfinite(depth) & (depth >= 0.001) & (depth <= 65)
        valid_values = depth[valid]
        meta.update({
            "image_relpath": row["image_relpath"], "scene": row["scene"],
            "day": row["day"], "crop": row["crop"], "view": row["view"],
            "valid_gt_fraction": float(valid.mean()),
            "valid_gt_median_m": float(np.median(valid_values)) if valid_values.size else None,
        })
        return {
            "pixel_values": torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1))).float(),
            "depth": torch.from_numpy(np.ascontiguousarray(depth)).float(),
            "metadata": meta,
        }


def coverage_fraction(shape: tuple[int, int], boxes: list[tuple[int, int, int, int]]) -> float:
    height, width = shape
    covered = np.zeros((height, width), dtype=bool)
    for top, left, bottom, right in boxes:
        if min(top, left) < 0 or bottom > height or right > width or top >= bottom or left >= right:
            raise ValueError(f"Out-of-range crop box: {(top, left, bottom, right)} in {shape}")
        covered[top:bottom, left:right] = True
    return float(covered.mean())


def coverage_report(rows: list[dict[str, str]], *, shape=(3956, 5280),
                    seed=42, epochs=5, repeats=1) -> list[dict[str, Any]]:
    reports = []
    for row in rows:
        boxes = []
        for epoch in range(1, epochs + 1):
            for draw in range(repeats):
                meta = draw_crop(row, shape, "F2", seed, epoch, draw)
                boxes.append(tuple(meta["crop_box_yxyx"]))
        reports.append({
            "image_relpath": row["image_relpath"], "crops": len(boxes),
            "unique_coverage_fraction": coverage_fraction(shape, boxes),
        })
    return reports
