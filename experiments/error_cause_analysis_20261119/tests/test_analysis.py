import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze_error_causes as analysis


class AnalysisMathTests(unittest.TestCase):
    def test_rmse_decomposition_and_affine_fit(self):
        rows = [{
            "image_index": "0", "valid_pixels": "2", "abs_error_sum": "5",
            "squared_error_sum": "13", "abs_relative_sum": "0.2",
            "prediction_sum": "8", "target_sum": "3",
            "prediction_squared_sum": "34", "target_squared_sum": "5",
            "prediction_target_sum": "13", "mae_m": "2.5", "rmse_m": str(math.sqrt(6.5)),
            "bias_m": "2.5",
        }]
        got = analysis.pool_absolute(rows)
        self.assertAlmostEqual(got["rmse_m"] ** 2, got["bias_m"] ** 2 + got["residual_variance_m2"])
        self.assertAlmostEqual(got["regression_slope_pred_on_gt"], 2.0)
        self.assertAlmostEqual(got["regression_intercept_m"], 1.0)
        self.assertAlmostEqual(got["pearson_from_sums"], 1.0)
        self.assertAlmostEqual(got["oracle_affine_rmse_m"], 0.0, places=7)

    def test_pool_local_uses_sums_not_mean_of_ratios(self):
        rows = [
            {"pair_count": "2", "abs_delta_error_sum": "0.4", "squared_delta_error_sum": "0.10",
             "abs_gt_delta_sum": "1.0", "abs_prediction_delta_sum": "0.5",
             "nonzero_gt_pair_count": "2", "sign_match_count": "1", "delta_mae_m": "0.2", "amplitude_ratio": "0.5"},
            {"pair_count": "1", "abs_delta_error_sum": "0.3", "squared_delta_error_sum": "0.09",
             "abs_gt_delta_sum": "3.0", "abs_prediction_delta_sum": "2.7",
             "nonzero_gt_pair_count": "1", "sign_match_count": "1", "delta_mae_m": "0.3", "amplitude_ratio": "0.9"},
        ]
        got = analysis.pool_local(rows)
        self.assertAlmostEqual(got["delta_mae_m"], 0.7 / 3)
        self.assertAlmostEqual(got["delta_rmse_m"], math.sqrt(0.19 / 3))
        self.assertAlmostEqual(got["amplitude_ratio"], 3.2 / 4.0)
        self.assertAlmostEqual(got["sign_match_rate"], 2 / 3)

    def test_paired_bootstrap_is_deterministic_and_uses_image_pairing(self):
        a = [
            {"image_index": "0", "valid_pixels": "10", "squared_error_sum": "10", "abs_error_sum": "5", "prediction_sum": "11", "target_sum": "10", "rmse_m": "1", "mae_m": "0.5", "bias_m": "0.1"},
            {"image_index": "1", "valid_pixels": "20", "squared_error_sum": "80", "abs_error_sum": "20", "prediction_sum": "22", "target_sum": "20", "rmse_m": "2", "mae_m": "1", "bias_m": "0.1"},
            {"image_index": "2", "valid_pixels": "30", "squared_error_sum": "270", "abs_error_sum": "45", "prediction_sum": "33", "target_sum": "30", "rmse_m": "3", "mae_m": "1.5", "bias_m": "0.1"},
        ]
        b = [dict(row, squared_error_sum=str(float(row["squared_error_sum"]) * 2), rmse_m=str(float(row["rmse_m"]) * math.sqrt(2))) for row in a]
        first = analysis.paired_bootstrap(a, b, "absolute_rmse_m", "pooled", 256, 42)
        second = analysis.paired_bootstrap(a, b, "absolute_rmse_m", "pooled", 256, 42)
        self.assertEqual(first, second)
        self.assertEqual(first["image_count"], 3)
        self.assertLess(first["difference_a_minus_b"], 0.0)


if __name__ == "__main__":
    unittest.main()
