from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BEST = ROOT / "experiments" / "best_model_inference_combination_20261122"
sys.path.insert(0, str(BEST))
sys.path.insert(0, str(HERE))

import analyze_extended_results as analysis  # noqa: E402
import analyze_results as stats  # noqa: E402


def make_record(index: int, *, valid: int = 10, abs_sum: float = 2.0,
                sq_sum: float = 1.0, high_n: int = 0,
                high_sum: float = 0.0) -> dict:
    return stats.record(index, valid, abs_sum, sq_sum, 0, 0.0,
                        high_n, high_sum)


class ExtendedAnalysisTests(unittest.TestCase):
    def test_empty_high_delta_images_are_undefined_in_image_mean(self):
        records = [
            make_record(0),
            make_record(1, high_n=2, high_sum=4.0),
            make_record(2, high_n=3, high_sum=12.0),
        ]
        self.assertIsNone(records[0]["high_mean"])
        self.assertAlmostEqual(
            stats.estimate(records, "high14_mae", "image_mean"), 3.0
        )
        self.assertAlmostEqual(
            stats.estimate(records, "high14_mae", "pooled"), 16.0 / 5.0
        )

    def test_bootstrap_requires_same_ordered_image_ids(self):
        a = [make_record(0), make_record(1)]
        b = [make_record(0), make_record(2)]
        with self.assertRaisesRegex(ValueError, "different image IDs"):
            stats.bootstrap_pair(a, b, "absolute_rmse", "pooled", 100, 42)

    def test_bootstrap_is_reproducible_for_paired_images(self):
        a = [make_record(0, sq_sum=4.0), make_record(1, sq_sum=9.0)]
        b = [make_record(0, sq_sum=1.0), make_record(1, sq_sum=4.0)]
        first = stats.bootstrap_pair(a, b, "absolute_rmse", "pooled", 500, 42)
        second = stats.bootstrap_pair(a, b, "absolute_rmse", "pooled", 500, 42)
        self.assertEqual(first, second)
        self.assertEqual(first["bootstrap_repeats"], 500)
        self.assertEqual(first["seed"], 42)

    def test_validation_ids_must_cover_the_fixed_order(self):
        with self.assertRaisesRegex(ValueError, "ordered Validation128 IDs"):
            analysis.check_ids("incomplete", [{"image_index": i} for i in range(127)])


if __name__ == "__main__":
    unittest.main()