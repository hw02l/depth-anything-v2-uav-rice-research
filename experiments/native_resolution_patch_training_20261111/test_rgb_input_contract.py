from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from data_patch import (IMAGENET_MEAN, IMAGENET_STD, load_pair,
                        rgb_unit_float_to_uint8)
from inference import _input, _native_input, infer_depth, validate_rgb_uint8
from uav_metric_research import multiscale as m


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


class RGBInputContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        h, w = 1200, 1500
        yy, xx = np.indices((h, w))
        self.rgb = np.stack((xx % 256, yy % 256, (xx + yy) % 256), axis=-1).astype(np.uint8)
        bgr = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2BGR)
        self.path = self.root / "source.png"
        self.assertTrue(cv2.imwrite(str(self.path), bgr))
        self.row = {"image_relpath": "source.png", "depth_relpath": "unused.tif"}

    def tearDown(self):
        self.temp.cleanup()

    def test_unit_float_roundtrip_recovers_every_rgb_byte(self):
        bgr = cv2.imread(str(self.path), cv2.IMREAD_COLOR)
        raw_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        recovered = rgb_unit_float_to_uint8(raw_rgb.astype(np.float32) / 255.0)
        self.assertEqual(recovered.shape, raw_rgb.shape)
        self.assertEqual(recovered.dtype, np.uint8)
        self.assertTrue(np.array_equal(recovered, raw_rgb))
        self.assertGreater(int(recovered.max()), 1)

    def test_loader_bytes_match_cv2_source_and_training_unit_range(self):
        # Use the saved PNG bytes with a one-pixel synthetic depth file.
        import tifffile
        depth_path = self.root / "unused.tif"
        tifffile.imwrite(depth_path, np.ones(self.rgb.shape[:2], np.float32))
        loaded, _ = load_pair(self.row, self.root, self.root)
        bgr = cv2.imread(str(self.path), cv2.IMREAD_COLOR)
        raw_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.assertEqual(loaded.dtype, np.float32)
        self.assertGreaterEqual(float(loaded.min()), 0.0)
        self.assertLessEqual(float(loaded.max()), 1.0)
        self.assertTrue(np.array_equal(rgb_unit_float_to_uint8(loaded), raw_rgb))

    def test_inference_rejects_float_and_suspicious_black_uint8(self):
        validate_rgb_uint8(self.rgb)
        model = CaptureConstantModel().eval()
        with self.assertRaises(TypeError):
            infer_depth(model, self.rgb.astype(np.float32) / 255.0, "cpu", "P0")
        with self.assertRaises(ValueError):
            infer_depth(model, np.zeros_like(self.rgb), "cpu", "P0")
        with self.assertRaises(ValueError):
            rgb_unit_float_to_uint8(self.rgb.astype(np.float32) * 2.0 / 255.0)

    def test_p0_preprocessing_and_coordinate_restore_match_frozen_b0(self):
        expected = m.preprocess(self.rgb, 518)
        model = CaptureConstantModel().eval()
        result = infer_depth(model, self.rgb, "cpu", "P0")
        self.assertTrue(torch.equal(model.first[0], expected))
        self.assertEqual(result.prediction.shape, self.rgb.shape[:2])
        np.testing.assert_allclose(result.prediction, 2.0, rtol=0, atol=1e-6)

    def test_p1_p2_tile_inputs_preserve_rgb_and_blend_overlaps(self):
        for method, size in (("P1", 1036), ("P2", 518)):
            model = CaptureConstantModel().eval()
            result = infer_depth(model, self.rgb, "cpu", method)
            boxes = m.tiles(self.rgb.shape[:2], size, 0.25)
            self.assertEqual(model.calls, len(boxes))
            self.assertEqual(result.tile_count, len(boxes))
            y0, x0, y1, x1 = boxes[0]
            crop0 = self.rgb[y0:y1, x0:x1]
            expected_first = _input(crop0, 518) if method == "P1" else _native_input(crop0)
            self.assertTrue(torch.equal(model.first[0], expected_first))
            yl, xl, yb, xr = boxes[-1]
            cropl = self.rgb[yl:yb, xl:xr]
            expected_last = _input(cropl, 518) if method == "P1" else _native_input(cropl)
            self.assertTrue(torch.equal(model.last[0], expected_last))
            self.assertTrue(np.all(result.coverage_count > 0))
            np.testing.assert_allclose(result.prediction, 2.0, rtol=0, atol=1e-5)
            self.assertGreater(result.coverage_count.max(), 1)


if __name__ == "__main__":
    unittest.main()

