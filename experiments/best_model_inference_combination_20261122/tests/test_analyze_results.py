import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
from analyze_results import bootstrap_pair, estimate


class ResultAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.a = [
            {"image_index": 0, "valid_pixels": 4, "abs_sum": 4.0, "sq_sum": 4.0,
             "full_n": 2, "full_sum": 0.4, "high_n": 1, "high_sum": 0.3,
             "mae": 1.0, "rmse": 1.0, "full_mean": 0.2, "high_mean": 0.3},
            {"image_index": 1, "valid_pixels": 4, "abs_sum": 8.0, "sq_sum": 16.0,
             "full_n": 2, "full_sum": 0.8, "high_n": 0, "high_sum": 0.0,
             "mae": 2.0, "rmse": 2.0, "full_mean": 0.4, "high_mean": None},
        ]
        self.b = [
            {"image_index": 0, "valid_pixels": 4, "abs_sum": 2.0, "sq_sum": 1.0,
             "full_n": 2, "full_sum": 0.2, "high_n": 1, "high_sum": 0.1,
             "mae": 0.5, "rmse": 0.5, "full_mean": 0.1, "high_mean": 0.1},
            {"image_index": 1, "valid_pixels": 4, "abs_sum": 4.0, "sq_sum": 4.0,
             "full_n": 2, "full_sum": 0.4, "high_n": 0, "high_sum": 0.0,
             "mae": 1.0, "rmse": 1.0, "full_mean": 0.2, "high_mean": None},
        ]

    def test_pooled_and_image_mean_are_distinct_estimands(self):
        self.assertAlmostEqual(estimate(self.a, "full14_mae", "pooled"), 0.3)
        self.assertAlmostEqual(estimate(self.a, "full14_mae", "image_mean"), 0.3)
        self.assertAlmostEqual(estimate(self.a, "absolute_rmse", "pooled"), np.sqrt(20 / 8))
        self.assertAlmostEqual(estimate(self.a, "absolute_rmse", "image_mean"), 1.5)

    def test_undefined_local_image_is_not_zero_filled(self):
        self.assertAlmostEqual(estimate(self.a, "high14_mae", "image_mean"), 0.3)
        self.assertAlmostEqual(estimate(self.a, "high14_mae", "pooled"), 0.3)

    def test_bootstrap_is_paired_and_deterministic(self):
        result1 = bootstrap_pair(self.a, self.b, "absolute_rmse", "pooled", 200, 42)
        result2 = bootstrap_pair(self.a, self.b, "absolute_rmse", "pooled", 200, 42)
        self.assertEqual(result1, result2)
        self.assertAlmostEqual(result1["difference_a_minus_b"], np.sqrt(20 / 8) - np.sqrt(5 / 8))


if __name__ == "__main__":
    unittest.main()
