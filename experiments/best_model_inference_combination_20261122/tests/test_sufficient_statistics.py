import csv
import math
import unittest
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parents[1] / "results" / "20261122T042859Z-35268" / "analysis_reconciled_d99d8ec"


class SufficientStatisticsTests(unittest.TestCase):
    def test_per_image_rows_reconcile_to_pooled_summary(self):
        with (ROOT / "per_image_sufficient_statistics.csv").open(encoding="utf-8", newline="") as f:
            per_image = list(csv.DictReader(f))
        with (ROOT / "condition_summary.csv").open(encoding="utf-8", newline="") as f:
            summary_rows = list(csv.DictReader(f))
        self.assertEqual(len(per_image), 6 * 128)
        by_condition = defaultdict(list)
        for row in per_image:
            by_condition[row["condition"]].append(row)
        summary = {(r["condition"], r["metric"]): float(r["pooled"]) for r in summary_rows}
        for condition, rows in by_condition.items():
            valid = sum(int(r["valid_pixels"]) for r in rows)
            abs_sum = sum(float(r["abs_sum"]) for r in rows)
            sq_sum = sum(float(r["sq_sum"]) for r in rows)
            full_n = sum(int(r["full_n"]) for r in rows)
            full_sum = sum(float(r["full_sum"]) for r in rows)
            high_n = sum(int(r["high_n"]) for r in rows)
            high_sum = sum(float(r["high_sum"]) for r in rows)
            actual = {
                "absolute_mae": abs_sum / valid,
                "absolute_rmse": math.sqrt(sq_sum / valid),
                "full14_mae": full_sum / full_n,
                "high14_mae": high_sum / high_n,
            }
            for metric, value in actual.items():
                self.assertAlmostEqual(value, summary[(condition, metric)], places=12, msg=f"{condition} {metric}")


if __name__ == "__main__":
    unittest.main()
