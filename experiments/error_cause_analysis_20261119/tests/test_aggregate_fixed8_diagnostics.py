import math
import unittest

from experiments.error_cause_analysis_20261119.analysis.aggregate_fixed8_diagnostics import (
    absolute_groups,
    local_groups,
)


class FixedEightAggregationTests(unittest.TestCase):
    def test_local_pooled_errors_use_pair_counts(self):
        base = {
            "condition": "C0-P0",
            "distance px": "14",
            "subset": "all",
            "crop": "corn",
            "view": "nadir",
            "signed slope": "0.5",
            "signed intercept m": "0.1",
            "pearson": "0.8",
            "rms amplitude ratio": "0.6",
            "sign match rate": "0.75",
        }
        rows = [
            {**base, "pair count": "2", "delta mae m": "0.2", "delta rmse m": "0.3"},
            {**base, "pair count": "6", "delta mae m": "0.5", "delta rmse m": "0.4"},
        ]
        summary = next(row for row in local_groups(rows) if row["group_type"] == "fixed8")
        self.assertAlmostEqual(summary["pooled_delta_mae_m"], 0.425)
        self.assertAlmostEqual(summary["pooled_delta_rmse_m"], math.sqrt((2 * 0.3**2 + 6 * 0.4**2) / 8))
        self.assertAlmostEqual(summary["pair_weighted_sign_match_rate"], 0.75)

    def test_absolute_pooled_rmse_is_pixel_weighted(self):
        rows = [
            {"condition": "C0-P0", "crop": "corn", "view": "nadir", "valid pixels": "2", "mae m": "0.2", "rmse m": "0.3", "bias m": "-0.1", "absrel": "0.2", "pearson": "0.8"},
            {"condition": "C0-P0", "crop": "corn", "view": "nadir", "valid pixels": "6", "mae m": "0.5", "rmse m": "0.4", "bias m": "0.1", "absrel": "0.4", "pearson": "0.9"},
        ]
        summary = next(row for row in absolute_groups(rows) if row["group_type"] == "fixed8")
        self.assertAlmostEqual(summary["pooled_mae_m"], 0.425)
        self.assertAlmostEqual(summary["pooled_rmse_m"], math.sqrt((2 * 0.3**2 + 6 * 0.4**2) / 8))
        self.assertAlmostEqual(summary["pooled_bias_m"], 0.05)


if __name__ == "__main__":
    unittest.main()
