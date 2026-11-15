from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_results as analysis


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class BootstrapTests(unittest.TestCase):
    def test_paired_image_and_pooled_bootstrap_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for group, factor in (("C0", 1.0), ("F1", 0.96), ("F2", 0.90)):
                for method in ("P0", "P1", "P2"):
                    run_dir = root / "eval" / f"{group}_{method}"
                    run_dir.mkdir(parents=True)
                    absolute_rows, local_rows = [], []
                    for image_index in range(128):
                        target = 18.0 + (image_index % 7) * 0.1
                        error = factor * (0.4 + (image_index % 5) * 0.01)
                        prediction = target + error
                        absolute_rows.append({
                            "image_index": image_index,
                            "image_relpath": f"image_{image_index:03d}.JPG",
                            "valid_pixels": 100,
                            "abs_error_sum": error * 100,
                            "squared_error_sum": error * error * 100,
                            "abs_relative_sum": error / target * 100,
                            "prediction_sum": prediction * 100,
                            "target_sum": target * 100,
                            "prediction_squared_sum": (prediction * prediction + 0.2) * 100,
                            "target_squared_sum": (target * target + 0.3) * 100,
                            "prediction_target_sum": (prediction * target + 0.1) * 100,
                            "full14_pair_count": 80,
                            "full14_abs_delta_error_sum": error * 80,
                        })
                        delta_error = factor * (0.08 + (image_index % 9) * 0.001)
                        local_rows.append({
                            "image_index": image_index, "stride_px": 14, "group": "gt_ge20cm",
                            "pair_count": 10, "abs_delta_error_sum": delta_error * 10,
                            "squared_delta_error_sum": delta_error * delta_error * 10,
                            "abs_gt_delta_sum": 3.0,
                            "abs_prediction_delta_sum": 3.0 * (0.33 + (1.0 - factor) * 0.5),
                            "nonzero_gt_pair_count": 10, "sign_match_count": 8,
                        })
                    _write_csv(run_dir / f"per_image_absolute_{group}_{method}.csv", absolute_rows)
                    _write_csv(run_dir / f"per_image_local_{group}_{method}.csv", local_rows)

            out1 = root / "bootstrap1"
            out2 = root / "bootstrap2"
            rows1 = analysis.analyze(root / "eval", out1, repeats=128, seed=42)
            rows2 = analysis.analyze(root / "eval", out2, repeats=128, seed=42)
            self.assertEqual(rows1, rows2)
            self.assertEqual(len(rows1), 120)
            delta = analysis._paired_delta_bootstrap(
                np.asarray([0.3, 0.2, 0.1]), np.asarray([0.2, 0.2, 0.05]),
                np.asarray([[0, 1, 2], [2, 2, 0]], dtype=np.int16))
            self.assertAlmostEqual(delta[0], -0.05)
            self.assertEqual((out1 / "bootstrap_metadata.json").is_file(), True)


if __name__ == "__main__":
    unittest.main()
