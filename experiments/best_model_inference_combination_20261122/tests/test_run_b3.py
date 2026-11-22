import hashlib
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
EXP = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(EXP))
from uav_metric_research import multiscale as m
from run_b3 import dense14_stats, fixed_pair_stats, fuse_depth


class B3GeometryTests(unittest.TestCase):
    def test_constant_maps_cancel_tile_offset(self):
        global_pred = np.full((240, 260), 10.0, np.float32)
        tile_pred = np.full((240, 260), 12.0, np.float32)
        fused = fuse_depth(global_pred, tile_pred)
        np.testing.assert_allclose(fused, global_pred, rtol=0, atol=2e-5)

    def test_fusion_preserves_only_tile_high_frequency_residual(self):
        yy, xx = np.mgrid[:400, :430]
        global_pred = np.full((400, 430), 20.0, np.float32)
        detail = (0.3 * np.sin(xx / 4.0) + 0.2 * np.cos(yy / 5.0)).astype(np.float32)
        tile_pred = global_pred + detail
        fused = fuse_depth(global_pred, tile_pred)
        filtered_detail = __import__('cv2').boxFilter(detail, -1, (215, 215), normalize=True,
                    borderType=__import__('cv2').BORDER_REFLECT_101)
        np.testing.assert_allclose(fused, global_pred + detail - filtered_detail, rtol=0, atol=2e-5)

    def test_anchored_tile_grid_covers_original_coordinate_grid(self):
        boxes = m.tiles((3956, 5280), 2000, 0.25)
        self.assertEqual(len(boxes), 12)
        self.assertEqual(boxes[0], (0, 0, 2000, 2000))
        self.assertEqual(boxes[-1], (1956, 3280, 3956, 5280))
        covered = np.zeros((3956, 5280), np.uint8)
        for y, x, bottom, right in boxes:
            covered[y:bottom, x:right] += 1
        self.assertTrue(np.all(covered > 0))

    def test_dense14_statistics_use_both_valid_endpoints(self):
        gt = np.arange(30 * 31, dtype=np.float32).reshape(30, 31) / 100.0 + 1
        pred = gt * 2
        valid = np.ones(gt.shape, dtype=bool)
        valid[0, 14] = False
        result = dense14_stats(pred, gt, valid)
        expected_count = 0
        expected_error_sum = 0.0
        for axis in (0, 1):
            if axis == 0:
                g0, g1, v = gt[:-14, :], gt[14:, :], valid[:-14, :] & valid[14:, :]
                p0, p1 = pred[:-14, :], pred[14:, :]
            else:
                g0, g1, v = gt[:, :-14], gt[:, 14:], valid[:, :-14] & valid[:, 14:]
                p0, p1 = pred[:, :-14], pred[:, 14:]
            err = (p0[v].astype(np.float64) - p1[v].astype(np.float64)) - (g0[v].astype(np.float64) - g1[v].astype(np.float64))
            expected_count += int(v.sum())
            expected_error_sum += float(np.abs(err).sum(dtype=np.float64))
        self.assertEqual(result["full14_pair_count"], expected_count)
        self.assertAlmostEqual(result["full14_abs_error_sum_m"], expected_error_sum, places=6)

    def test_fixed_pair_hash_and_large_delta_statistics(self):
        rng = np.random.default_rng(42)
        gt = rng.uniform(1, 12, size=(300, 320)).astype(np.float32)
        pred = gt * 0.8 + 0.2
        valid = m.valid_gt(gt)
        expected_pairs = []
        for stride in (14, 28, 56, 107, 214):
            for axis, y, x, yy, xx in m.sample_pairs(gt.shape, valid, 0, stride):
                coordinates = np.stack([y, x, yy, xx], axis=1).astype("<i4")
                expected_pairs.append({"stride": stride, "axis": axis, "pairs": int(len(y)),
                    "coordinates_sha256": hashlib.sha256(coordinates.tobytes()).hexdigest()})
        rows, actual = fixed_pair_stats(pred, gt, valid, 0, [{"pairs": expected_pairs}])
        self.assertEqual(actual, expected_pairs)
        high = [r for r in rows if r["stride_px"] == 14 and r["group"] == "gt_ge20cm"]
        self.assertGreater(sum(r["pair_count"] for r in high), 0)
        self.assertTrue(all(np.isfinite(r["absolute_error_sum_m"]) for r in rows))


if __name__ == "__main__":
    unittest.main()
