from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np


SCRIPT = Path(__file__).parents[1] / "make_fixed8_comparison.py"
SPEC = importlib.util.spec_from_file_location("fixed8_visualization", SCRIPT)
fixed8 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(fixed8)


class Fixed8VisualizationTests(unittest.TestCase):
    def test_prediction_cache_requires_fixed_schema_and_shape(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "prediction.npz"
            expected = np.arange(12, dtype=np.float32).reshape(3, 4)
            np.savez_compressed(path, prediction_m=expected)
            self.assertTrue(np.array_equal(fixed8.load_prediction(path, (3, 4)), expected))
            wrong_dtype = Path(temp) / "wrong_dtype.npz"
            np.savez_compressed(wrong_dtype, prediction_m=expected.astype(np.float64))
            with self.assertRaisesRegex(ValueError, "dtype"):
                fixed8.load_prediction(wrong_dtype, (3, 4))
            with self.assertRaisesRegex(ValueError, "shape or values"):
                fixed8.load_prediction(path, (4, 3))


    def test_prediction_cache_rejects_nonfinite_and_extra_arrays(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            nonfinite = temp / "nonfinite.npz"
            np.savez_compressed(nonfinite, prediction_m=np.array([[np.nan]], dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "shape or values"):
                fixed8.load_prediction(nonfinite, (1, 1))
            extra = temp / "extra.npz"
            np.savez_compressed(extra, prediction_m=np.ones((1, 1), dtype=np.float32), rgb=np.zeros((1, 1, 3), dtype=np.uint8))
            with self.assertRaisesRegex(ValueError, "schema"):
                fixed8.load_prediction(extra, (1, 1))


    def test_depth_and_error_figures_use_fixed_shared_ranges(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            rgb = np.zeros((32, 48, 3), dtype=np.uint8)
            gt = np.full((32, 48), 20.0, dtype=np.float32)
            valid = np.ones((32, 48), dtype=bool)
            valid[:, :4] = False
            predictions = {key: gt.copy() for key in fixed8.DEPTH_CONDITIONS}
            predictions["C0-P0"][:, 8:12] += 12.0
            depth_path = temp / "depth.png"
            error_path = temp / "error.png"
            fixed8.render_depth_figure(rgb, gt, predictions, valid, 7, "corn", "nadir", depth_path)
            over_range = fixed8.render_error_figure(gt, predictions, valid, 7, "corn", "nadir", error_path)
            self.assertTrue(depth_path.is_file() and depth_path.stat().st_size > 1000)
            self.assertTrue(error_path.is_file() and error_path.stat().st_size > 1000)
            self.assertEqual(fixed8.DEPTH_MAX_M, 65.0)
            self.assertEqual(fixed8.ERROR_MAX_M, 10.0)
            self.assertEqual(over_range["C0-P0"], 32 * 4)
            self.assertEqual(over_range["C0-B3"], 0)
            self.assertIsNotNone(cv2.imread(str(depth_path)))
            self.assertIsNotNone(cv2.imread(str(error_path)))

    def test_npy_prediction_cache_requires_fixed_shape_dtype_and_finite_values(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            valid = temp / "valid.npy"
            expected = np.arange(12, dtype=np.float32).reshape(3, 4)
            np.save(valid, expected, allow_pickle=False)
            self.assertTrue(np.array_equal(fixed8.load_prediction_npy(valid, (3, 4)), expected))
            wrong_dtype = temp / "wrong_dtype.npy"
            np.save(wrong_dtype, expected.astype(np.float64), allow_pickle=False)
            with self.assertRaisesRegex(ValueError, "shape, dtype, or values"):
                fixed8.load_prediction_npy(wrong_dtype, (3, 4))
            nonfinite = temp / "nonfinite.npy"
            invalid = expected.copy()
            invalid[0, 0] = np.nan
            np.save(nonfinite, invalid, allow_pickle=False)
            with self.assertRaisesRegex(ValueError, "shape, dtype, or values"):
                fixed8.load_prediction_npy(nonfinite, (3, 4))


if __name__ == "__main__":
    unittest.main()
