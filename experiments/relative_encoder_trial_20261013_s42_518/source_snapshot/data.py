from __future__ import annotations

import csv
import math
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset

IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Manifest has no rows: {path}")
    required = {"split", "scene", "crop", "day", "view", "image_relpath", "depth_relpath"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Manifest is missing columns {sorted(missing)}: {path}")
    return rows


def worker_seed(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def _round_multiple(value: float, multiple: int = 14) -> int:
    return max(multiple, int(round(value / multiple)) * multiple)


class UAVDepthDataset(Dataset):
    """Paired RGB/float16-TIFF dataset with aligned resize/crop/flip transforms."""

    def __init__(
        self,
        rows: list[dict[str, str]],
        rgb_root: str | Path,
        depth_root: str | Path,
        *,
        image_size: int = 518,
        training: bool,
    ) -> None:
        self.rows = rows
        self.rgb_root = Path(rgb_root)
        self.depth_root = Path(depth_root)
        self.image_size = int(image_size)
        self.training = bool(training)
        if self.image_size < 56 or self.image_size % 14:
            raise ValueError("image_size must be >=56 and divisible by the DAv2 patch size (14)")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image_path = self.rgb_root / Path(row["image_relpath"])
        depth_path = self.depth_root / Path(row["depth_relpath"])
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Could not read RGB image: {image_path}")
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        depth = np.asarray(tifffile.imread(depth_path), dtype=np.float32)
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        if depth.ndim != 2:
            raise ValueError(f"Expected a single-channel depth TIFF, got {depth.shape}: {depth_path}")
        if image.shape[:2] != depth.shape:
            raise ValueError(
                f"RGB/depth dimensions differ: {image_path} {image.shape[:2]} vs "
                f"{depth_path} {depth.shape}"
            )

        original_depth = depth.copy()
        height, width = depth.shape
        scale = self.image_size / min(height, width)
        resized_height = _round_multiple(height * scale)
        resized_width = _round_multiple(width * scale)
        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)
        depth = cv2.resize(depth, (resized_width, resized_height), interpolation=cv2.INTER_NEAREST)

        if self.training:
            top = random.randint(0, resized_height - self.image_size)
            left = random.randint(0, resized_width - self.image_size)
            image = image[top : top + self.image_size, left : left + self.image_size]
            depth = depth[top : top + self.image_size, left : left + self.image_size]
            if random.random() < 0.5:
                image = image[:, ::-1]
                depth = depth[:, ::-1]

        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        pixel_values = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()
        target = torch.from_numpy(np.ascontiguousarray(depth)).float()
        result: dict[str, Any] = {
            "pixel_values": pixel_values,
            "depth": target,
            "scene": row["scene"],
            "view": row["view"],
            "crop": row["crop"],
            "day": row["day"],
            "image_relpath": row["image_relpath"],
        }
        if not self.training:
            result["original_depth"] = torch.from_numpy(np.ascontiguousarray(original_depth)).float()
        return result
