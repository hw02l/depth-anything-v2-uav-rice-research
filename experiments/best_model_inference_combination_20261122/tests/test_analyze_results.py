import sys
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
from analyze_results import bootstrap_pair, estimate, load_new


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

    def test_new_conditions_do_not_mix_local_rows_between_groups(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            absolute_fields = ["model_id", "image_index", "valid_pixel_count", "abs_error_sum_m",
                "squared_error_sum_m2", "full14_pair_count", "full14_abs_error_sum_m",
                "abs_relative_error_sum", "bias_m", "pearson_r", "prediction_sum_m", "gt_sum_m",
                "prediction_squared_sum_m2", "gt_squared_sum_m2", "prediction_gt_cross_sum_m2"]
            local_fields = ["model_id", "image_index", "stride_px", "group", "pair_count",
                            "absolute_error_sum_m"]
            with (root / "per_image_absolute_sufficient.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=absolute_fields); writer.writeheader()
                for group in ("C0", "E3"):
                    writer.writerow({"model_id": f"{group}-B3", "image_index": 0,
                        "valid_pixel_count": 4, "abs_error_sum_m": 1, "squared_error_sum_m2": 1,
                        "full14_pair_count": 2, "full14_abs_error_sum_m": 0.4,
                        "abs_relative_error_sum": 0.1, "bias_m": 0.0, "pearson_r": 0.5,
                        "prediction_sum_m": 8, "gt_sum_m": 8, "prediction_squared_sum_m2": 16,
                        "gt_squared_sum_m2": 16, "prediction_gt_cross_sum_m2": 16})
            with (root / "per_image_local_sufficient.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=local_fields); writer.writeheader()
                writer.writerow({"model_id": "C0-B3", "image_index": 0, "stride_px": 14,
                    "group": "gt_ge20cm", "pair_count": 1, "absolute_error_sum_m": 0.5})
                writer.writerow({"model_id": "E3-B3", "image_index": 0, "stride_px": 14,
                    "group": "gt_ge20cm", "pair_count": 1, "absolute_error_sum_m": 0.25})
            c0 = load_new(root, "C0")
            e3 = load_new(root, "E3")
            self.assertEqual((c0[0]["high_n"], c0[0]["high_sum"]), (1, 0.5))
            self.assertEqual((e3[0]["high_n"], e3[0]["high_sum"]), (1, 0.25))


if __name__ == "__main__":
    unittest.main()
