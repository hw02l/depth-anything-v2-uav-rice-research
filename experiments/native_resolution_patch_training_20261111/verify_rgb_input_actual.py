from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from data_patch import (  # noqa: E402
    IMAGENET_MEAN, IMAGENET_STD, NativePatchDataset, load_pair, read_manifest,
    resized_shape, rgb_unit_float_to_uint8,
)
from inference import _input, _native_input, infer_depth  # noqa: E402
from uav_metric_research import multiscale as m  # noqa: E402


class CaptureConstantModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.first = None
        self.last = None

    def forward(self, pixel_values):
        value = pixel_values.detach().cpu()
        if self.first is None:
            self.first = value.clone()
        self.last = value.clone()
        self.calls += 1
        return SimpleNamespace(predicted_depth=torch.full(
            (pixel_values.shape[0], 518, 518), 2.0,
            dtype=pixel_values.dtype, device=pixel_values.device))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def verify_training_rgb_and_gt(row, rgb_root: Path, depth_root: Path) -> dict[str, Any]:
    rgb01, depth = load_pair(row, rgb_root, depth_root)
    results = {}
    for group in ("C0", "F1", "F2"):
        dataset = NativePatchDataset([row], rgb_root, depth_root, group,
                                     repeats=1, seed=42, training=True)
        dataset.set_epoch(1)
        sample = dataset[0]
        meta = sample["metadata"]
        top, left, bottom, right = meta["crop_box_yxyx"]
        if group == "C0":
            rh, rw = resized_shape(*depth.shape)
            rgb = cv2.resize(rgb01, (rw, rh), interpolation=cv2.INTER_CUBIC)
            gt = cv2.resize(depth, (rw, rh), interpolation=cv2.INTER_NEAREST)
            rgb, gt = rgb[top:bottom, left:right], gt[top:bottom, left:right]
        else:
            rgb, gt = rgb01[top:bottom, left:right], depth[top:bottom, left:right]
            if group == "F1":
                rgb = cv2.resize(rgb, (518, 518), interpolation=cv2.INTER_CUBIC)
                gt = cv2.resize(gt, (518, 518), interpolation=cv2.INTER_NEAREST)
        if meta["flipped_horizontal"]:
            rgb, gt = rgb[:, ::-1], gt[:, ::-1]
        expected_pixels = ((rgb - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1).astype(np.float32)
        actual_pixels = sample["pixel_values"].numpy()
        actual_gt = sample["depth"].numpy()
        if not np.array_equal(actual_pixels, expected_pixels):
            raise AssertionError(f"{group}: training RGB normalization/crop mismatch")
        if not np.array_equal(actual_gt, gt, equal_nan=True):
            raise AssertionError(f"{group}: crop RGB/GT coordinates differ")
        results[group] = {
            "passed": True, "sample_rgb_float_range": [float(rgb.min()), float(rgb.max())],
            "pixel_values_shape": list(actual_pixels.shape),
            "pixel_values_sha256": sha256_bytes(actual_pixels.tobytes()),
            "gt_shape": list(actual_gt.shape), "valid_gt_fraction": meta["valid_gt_fraction"],
            "crop_box_yxyx": meta["crop_box_yxyx"],
            "flipped_horizontal": meta["flipped_horizontal"],
        }
    return results


def verify_actual_image(row, rgb_root: Path, depth_root: Path) -> dict[str, Any]:
    image_path = rgb_root / row["image_relpath"]
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(image_path)
    source_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    loaded_rgb01, gt = load_pair(row, rgb_root, depth_root)
    evaluation_rgb = rgb_unit_float_to_uint8(loaded_rgb01)
    if source_rgb.shape != evaluation_rgb.shape:
        raise AssertionError("Image dimensions changed between source and evaluation loader")
    if source_rgb.dtype != evaluation_rgb.dtype:
        raise AssertionError("RGB dtype changed between source and evaluation loader")
    if not np.array_equal(source_rgb, evaluation_rgb):
        raise AssertionError("RGB channel order or source pixel bytes changed")
    if source_rgb.shape[:2] != gt.shape:
        raise AssertionError("RGB and GT geometry do not match")
    if int(evaluation_rgb.max()) <= 1:
        raise AssertionError("Actual image appears black after RGB conversion")

    input_checks = {}
    for method, tile_size in (("P0", None), ("P1", 1036), ("P2", 518)):
        model = CaptureConstantModel().eval()
        result = infer_depth(model, evaluation_rgb, "cpu", method)
        if method == "P0":
            expected_first = m.preprocess(source_rgb, 518)
            expected_last = expected_first
            expected_calls = 1
        else:
            boxes = m.tiles(source_rgb.shape[:2], tile_size, 0.25)
            if not boxes or any((b - y, r - x) != (tile_size, tile_size)
                                for y, x, b, r in boxes):
                raise AssertionError(f"{method}: tile grid has missing or padded boundary tiles")
            y, x, b, r = boxes[0]
            first_crop = source_rgb[y:b, x:r]
            y, x, b, r = boxes[-1]
            last_crop = source_rgb[y:b, x:r]
            expected_first = _input(first_crop, 518) if method == "P1" else _native_input(first_crop)
            expected_last = _input(last_crop, 518) if method == "P1" else _native_input(last_crop)
            expected_calls = len(boxes)
        if model.calls != expected_calls:
            raise AssertionError(f"{method}: expected {expected_calls} forwards, got {model.calls}")
        if not torch.equal(model.first[0], expected_first) or not torch.equal(model.last[0], expected_last):
            raise AssertionError(f"{method}: model input differs from frozen RGB preprocessing")
        if result.prediction.shape != gt.shape or not np.isfinite(result.prediction).all():
            raise AssertionError(f"{method}: invalid restored prediction geometry")
        if not np.all(result.coverage_count > 0):
            raise AssertionError(f"{method}: uncovered output pixels")
        if method != "P0" and result.coverage_count.max() <= 1:
            raise AssertionError(f"{method}: expected overlap was not merged")
        np.testing.assert_allclose(result.prediction, 2.0, rtol=0, atol=1e-5)
        input_checks[method] = {
            "passed": True, "tile_count": result.tile_count, "model_forward_count": model.calls,
            "coverage_min": int(result.coverage_count.min()),
            "coverage_max": int(result.coverage_count.max()),
            "first_input_tensor_sha256": sha256_bytes(model.first.numpy().tobytes()),
            "last_input_tensor_sha256": sha256_bytes(model.last.numpy().tobytes()),
            "coordinate_restore_and_blend_constant_depth_passed": True,
        }

    return {
        "image_relpath": row["image_relpath"],
        "rgb_shape": list(source_rgb.shape), "rgb_dtype": str(source_rgb.dtype),
        "rgb_min": int(source_rgb.min()), "rgb_max": int(source_rgb.max()),
        "rgb_sha256": sha256_bytes(source_rgb.tobytes()),
        "rgb_loader_roundtrip_exact": True,
        "rgb_gt_shape_match": True, "training_input_audit": verify_training_rgb_and_gt(row, rgb_root, depth_root),
        "inference_methods": input_checks,
    }


def main():
    parser = argparse.ArgumentParser(description="Actual UAV3DCrop RGB/P0/P1/P2/training input regression audit.")
    parser.add_argument("--rgb-root", type=Path, required=True)
    parser.add_argument("--depth-root", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--image-index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite RGB audit: {args.output}")
    rows = read_manifest(args.validation_manifest)
    if len(rows) != 128 or not 0 <= args.image_index < len(rows):
        raise ValueError("Expected canonical Validation128 and valid image index")
    result = verify_actual_image(rows[args.image_index], args.rgb_root, args.depth_root)
    result["validation_image_index"] = args.image_index
    result["status"] = "passed_actual_uav_image_rgb_input_regression"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({"status": result["status"], "image": result["image_relpath"],
                      "rgb_shape": result["rgb_shape"], "methods": result["inference_methods"]},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

