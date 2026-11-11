from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import tifffile
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from data_patch import NativePatchDataset, coverage_fraction, draw_crop
from evaluate_patch import evaluate_one
from inference import infer_depth
from patch_engine import (capture_rng, cpu_model_state, latest_epoch, load_epoch,
                          save_epoch)
from train_patch import _train_epoch, make_optimizer
from uav_metric_research import multiscale as m


class ConstantModel(torch.nn.Module):
    def forward(self, pixel_values):
        value = pixel_values[:, :1] * 0 + 2.0
        return SimpleNamespace(predicted_depth=value[:, 0])


class ToyTrainModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Conv2d(3, 4, 1)
        self.head = torch.nn.Conv2d(4, 1, 1)

    def forward(self, pixel_values):
        value = torch.nn.functional.softplus(self.head(self.backbone(pixel_values)))[:, 0] + 1.0
        return SimpleNamespace(predicted_depth=value)


def build_synthetic_pair(root: Path, shape=(1200, 1500)):
    h, w = shape
    yy, xx = np.indices((h, w))
    rgb = np.stack((xx % 256, yy % 256, (xx + yy) % 256), axis=-1).astype(np.uint8)
    depth = (8.0 + ((xx // 14 + yy // 14) % 2) * 0.3 + yy * 0.005).astype(np.float32)
    depth[::17, ::19] = 0
    depth[::11, ::13] = np.nan
    depth[::23, ::29] = 70
    image_rel = "2025/Day019_Corn/images/synthetic.png"
    depth_rel = "2025/Day019_Corn_depth/synthetic_depth.tif"
    image_path = root / "rgb" / image_rel
    depth_path = root / "depth" / depth_rel
    image_path.parent.mkdir(parents=True, exist_ok=True)
    depth_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(image_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    tifffile.imwrite(depth_path, depth)
    row = {
        "split": "train", "scene": "2025/Day019_Corn", "day": "2025/Day019",
        "crop": "corn", "view": "nadir", "image_relpath": image_rel,
        "depth_relpath": depth_rel,
    }
    return row, rgb, depth


class NativePatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.row, self.rgb, self.depth = build_synthetic_pair(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def dataset_sample(self, group, epoch=1, draw=0):
        dataset = NativePatchDataset([self.row], self.root / "rgb", self.root / "depth",
                                     group, repeats=draw + 1, seed=42, training=True)
        dataset.set_epoch(epoch)
        return dataset[draw]

    def test_f2_uses_native_aligned_crop_and_preserves_invalid_values(self):
        sample = self.dataset_sample("F2")
        meta = sample["metadata"]
        top, left, bottom, right = meta["crop_box_yxyx"]
        expected = self.depth[top:bottom, left:right]
        if meta["flipped_horizontal"]:
            expected = expected[:, ::-1]
        np.testing.assert_array_equal(sample["depth"].numpy(), expected)
        self.assertEqual(meta["crop_space"], "original_image")
        self.assertEqual(meta["native_px_per_input_px_yx"], [1.0, 1.0])
        self.assertTrue(np.isnan(sample["depth"].numpy()).any())
        self.assertGreater(meta["valid_gt_fraction"], 0.0)

    def test_f1_resizes_only_after_original_crop(self):
        sample = self.dataset_sample("F1")
        meta = sample["metadata"]
        top, left, bottom, right = meta["crop_box_yxyx"]
        cropped = self.depth[top:bottom, left:right]
        expected = cv2.resize(cropped, (518, 518), interpolation=cv2.INTER_NEAREST)
        if meta["flipped_horizontal"]:
            expected = expected[:, ::-1]
        np.testing.assert_array_equal(sample["depth"].numpy(), expected)
        self.assertEqual(meta["crop_space"], "original_image")
        self.assertEqual(meta["native_px_per_input_px_yx"], [2.0, 2.0])

    def test_c0_resizes_whole_pair_before_crop(self):
        sample = self.dataset_sample("C0")
        meta = sample["metadata"]
        h, w = self.depth.shape
        rh, rw = m.frozen("data")._round_multiple(h * 518 / min(h, w)), m.frozen("data")._round_multiple(w * 518 / min(h, w))
        resized = cv2.resize(self.depth, (rw, rh), interpolation=cv2.INTER_NEAREST)
        top, left, bottom, right = meta["crop_box_yxyx"]
        expected = resized[top:bottom, left:right]
        if meta["flipped_horizontal"]:
            expected = expected[:, ::-1]
        np.testing.assert_array_equal(sample["depth"].numpy(), expected)
        self.assertEqual(meta["crop_space"], "global_resized_image")

    def test_corresponding_random_coordinates_match_across_groups(self):
        c0 = draw_crop(self.row, self.depth.shape, "C0", 42, 2, 1)
        f1 = draw_crop(self.row, self.depth.shape, "F1", 42, 2, 1)
        f2 = draw_crop(self.row, self.depth.shape, "F2", 42, 2, 1)
        self.assertEqual(c0["normalized_start_uv"], f1["normalized_start_uv"])
        self.assertEqual(f1["normalized_start_uv"], f2["normalized_start_uv"])
        self.assertEqual(c0["flipped_horizontal"], f1["flipped_horizontal"])
        self.assertEqual(f1["flipped_horizontal"], f2["flipped_horizontal"])
        self.assertNotEqual(f2["normalized_start_uv"], draw_crop(self.row, self.depth.shape, "F2", 42, 3, 0)["normalized_start_uv"])

    def test_coverage_union(self):
        self.assertAlmostEqual(coverage_fraction((100, 100), [(0, 0, 50, 50), (25, 25, 75, 75)]), 0.4375)

    def test_p0_p1_p2_stitching_and_coverage(self):
        rgb = np.zeros((1300, 1500, 3), np.uint8)
        model = ConstantModel().eval()
        for method in ("P0", "P1", "P2"):
            result = infer_depth(model, rgb, "cpu", method)
            self.assertEqual(result.prediction.shape, (1300, 1500))
            self.assertTrue(np.isfinite(result.prediction).all())
            self.assertTrue(np.all(result.coverage_count > 0))
            np.testing.assert_allclose(result.prediction, 2.0, atol=1e-5)
            if method != "P0":
                self.assertIsNotNone(result.seam_distance_px)
                self.assertGreater(result.tile_count, 1)

    def test_fixed_pair_evaluation_and_tile_boundary_strata(self):
        model = ConstantModel().eval()
        expected_prediction = np.full(self.depth.shape, 2.0, np.float32)
        _, pair_rows = m.multiscale(expected_prediction, self.depth, 0)
        expected = {"pairs": pair_rows}
        p0, local, boundary, _ = evaluate_one(
            model, self.row, 0, self.root / "rgb", self.root / "depth", "cpu", "P0", expected)
        self.assertGreater(p0["valid_pixels"], 0)
        valid = m.valid_gt(self.depth); expected_count = int((valid[:-14, :] & valid[14:, :]).sum() + (valid[:, :-14] & valid[:, 14:]).sum()); self.assertEqual(p0["full14_pair_count"], expected_count)
        self.assertTrue(any(r["stride_px"] == 14 and r["group"] == "gt_ge20cm" for r in local))
        p2, _, boundary, _ = evaluate_one(
            model, self.row, 0, self.root / "rgb", self.root / "depth", "cpu", "P2", expected)
        self.assertGreater(p2["tile_count"], 1)
        self.assertTrue(boundary)
        self.assertTrue(all("condition" in row for row in boundary))

    def test_cpu_forward_backward_and_checkpoint_resume(self):
        dataset = NativePatchDataset([self.row], self.root / "rgb", self.root / "depth",
                                     "F2", seed=42, training=True)
        model = ToyTrainModel()
        optimizer = make_optimizer(model)
        scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=1, power=0.9)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        silog = m.frozen("losses").silog_loss
        result = _train_epoch(model, dataset, optimizer, scheduler, scaler, silog,
                              epoch=1, seed=42, device="cpu", amp=False,
                              accumulation=1, max_updates=1)
        self.assertEqual(result["optimizer_updates"], 1)
        self.assertTrue(np.isfinite(result["mean_silog_m"]))
        expected_state = {key: value.clone() for key, value in model.state_dict().items()}
        state = {
            "epoch": 1, "global_step": 1, "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(), "rng": capture_rng(),
            "signature_payload": {"run": "smoke"}, "device_type": "cpu",
        }
        checkpoint_dir = self.root / "run"
        path, digest = save_epoch(checkpoint_dir, state)
        self.assertEqual(path, latest_epoch(checkpoint_dir))
        self.assertEqual(len(digest), 64)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(1)
        model2 = ToyTrainModel()
        optimizer2 = make_optimizer(model2)
        scheduler2 = torch.optim.lr_scheduler.PolynomialLR(optimizer2, total_iters=1, power=0.9)
        scaler2 = torch.amp.GradScaler("cuda", enabled=False)
        restored = load_epoch(path, model2, optimizer2, scheduler2, scaler2,
                              {"run": "smoke"}, "cpu")
        self.assertEqual(restored["epoch"], 1)
        for key, value in expected_state.items():
            torch.testing.assert_close(model2.state_dict()[key], value)


if __name__ == "__main__":
    unittest.main()
