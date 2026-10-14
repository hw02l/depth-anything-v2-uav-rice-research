from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class MetricAccumulator:
    local_stride: int = 14
    count: int = 0
    abs_error_sum: float = 0.0
    squared_error_sum: float = 0.0
    abs_relative_sum: float = 0.0
    pred_sum: float = 0.0
    target_sum: float = 0.0
    pred_sq_sum: float = 0.0
    target_sq_sum: float = 0.0
    cross_sum: float = 0.0
    local_abs_delta_sum: float = 0.0
    local_pair_count: int = 0

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        prediction = prediction.detach().float().cpu().numpy()
        target = target.detach().float().cpu().numpy()
        valid = np.isfinite(prediction) & np.isfinite(target) & (target > 0) & (prediction > 0)
        if not np.any(valid):
            return
        p = prediction[valid].astype(np.float64, copy=False)
        t = target[valid].astype(np.float64, copy=False)
        error = p - t
        self.count += int(p.size)
        self.abs_error_sum += float(np.abs(error).sum())
        self.squared_error_sum += float(np.square(error).sum())
        self.abs_relative_sum += float((np.abs(error) / np.maximum(t, 1e-6)).sum())
        self.pred_sum += float(p.sum())
        self.target_sum += float(t.sum())
        self.pred_sq_sum += float(np.square(p).sum())
        self.target_sq_sum += float(np.square(t).sum())
        self.cross_sum += float((p * t).sum())

        stride = self.local_stride
        for axis in (0, 1):
            if prediction.shape[axis] <= stride:
                continue
            p0 = [slice(None), slice(None)]
            p1 = [slice(None), slice(None)]
            p0[axis] = slice(None, -stride)
            p1[axis] = slice(stride, None)
            p0, p1 = tuple(p0), tuple(p1)
            pair_valid = valid[p0] & valid[p1]
            if not np.any(pair_valid):
                continue
            pred_delta = prediction[p1] - prediction[p0]
            target_delta = target[p1] - target[p0]
            self.local_abs_delta_sum += float(np.abs(pred_delta[pair_valid] - target_delta[pair_valid]).sum())
            self.local_pair_count += int(pair_valid.sum())

    def result(self) -> dict[str, float | int | None]:
        if self.count == 0:
            raise ValueError("No valid depth pixels were available for evaluation")
        covariance = self.cross_sum - self.pred_sum * self.target_sum / self.count
        pred_variance = self.pred_sq_sum - self.pred_sum**2 / self.count
        target_variance = self.target_sq_sum - self.target_sum**2 / self.count
        denominator = np.sqrt(max(pred_variance, 0.0) * max(target_variance, 0.0))
        correlation = float(covariance / denominator) if denominator > 0 else None
        return {
            "valid_pixels": self.count,
            "mae_m": self.abs_error_sum / self.count,
            "rmse_m": float(np.sqrt(self.squared_error_sum / self.count)),
            "abs_rel": self.abs_relative_sum / self.count,
            "local_depth_delta_mae_m": (
                self.local_abs_delta_sum / self.local_pair_count if self.local_pair_count else None
            ),
            "local_pair_count": self.local_pair_count,
            "pearson_r": correlation,
            "local_delta_stride_px": self.local_stride,
        }


class SampledDeltaAccumulator:
    def __init__(self) -> None:
        self.total = [0, 0.0, 0.0, 0.0, 0, 0]
        self.small = [0, 0.0, 0.0, 0.0, 0, 0]
        self.large = [0, 0.0, 0.0, 0.0, 0, 0]

    @staticmethod
    def _add(total, gt_delta, pred_delta):
        if gt_delta.size == 0:
            return
        total[0] += int(gt_delta.size)
        total[1] += float(np.abs(pred_delta - gt_delta).sum())
        total[2] += float(np.abs(gt_delta).sum())
        total[3] += float(np.abs(pred_delta).sum())
        nonzero = np.abs(gt_delta) > 1e-6
        total[4] += int(nonzero.sum())
        total[5] += int(((gt_delta[nonzero] * pred_delta[nonzero]) > 0).sum())

    def update(self, gt_delta, pred_delta) -> None:
        self._add(self.total, gt_delta, pred_delta)
        absolute = np.abs(gt_delta)
        self._add(self.small, gt_delta[absolute < 0.05], pred_delta[absolute < 0.05])
        self._add(self.large, gt_delta[absolute >= 0.20], pred_delta[absolute >= 0.20])

    @staticmethod
    def _result(values):
        n, error_sum, gt_abs_sum, pred_abs_sum, nonzero, sign_correct = values
        return {
            "pair_count": n,
            "delta_mae_m": error_sum / n if n else None,
            "mean_abs_gt_delta_m": gt_abs_sum / n if n else None,
            "mean_abs_pred_delta_m": pred_abs_sum / n if n else None,
            "amplitude_ratio": pred_abs_sum / gt_abs_sum if gt_abs_sum > 0 else None,
            "nonzero_gt_pair_count": nonzero,
            "sign_match_count": sign_correct,
            "sign_match_rate": sign_correct / nonzero if nonzero else None,
        }

    def result(self):
        return {
            "all_sampled_pairs": self._result(self.total),
            "gt_abs_delta_0_5cm": self._result(self.small),
            "gt_abs_delta_ge_20cm": self._result(self.large),
        }


def sampled_local_shape(prediction, target, valid, image_index, crop, view, stride=14, candidates_per_axis=100000):
    prediction = prediction.detach().float().cpu().numpy()
    target = target.detach().float().cpu().numpy()
    valid = valid.detach().cpu().numpy().astype(bool, copy=False)
    height, width = target.shape
    rng = np.random.default_rng(42 + image_index * 10000 + stride)
    accumulators = {}
    candidates = 0
    valid_pairs = 0
    for axis in (0, 1):
        h = height - stride if axis == 0 else height
        w = width - stride if axis == 1 else width
        count = min(candidates_per_axis, h * w)
        flat = rng.choice(h * w, size=count, replace=False)
        y, x = flat // w, flat % w
        y1 = y + (stride if axis == 0 else 0)
        x1 = x + (stride if axis == 1 else 0)
        pair_valid = valid[y, x] & valid[y1, x1]
        pair_valid &= np.isfinite(prediction[y, x]) & np.isfinite(prediction[y1, x1])
        pair_valid &= (prediction[y, x] > 0) & (prediction[y1, x1] > 0)
        y, x, y1, x1 = y[pair_valid], x[pair_valid], y1[pair_valid], x1[pair_valid]
        gt_delta = (target[y1, x1] - target[y, x]).astype(np.float64, copy=False)
        pred_delta = (prediction[y1, x1] - prediction[y, x]).astype(np.float64, copy=False)
        edge = np.minimum.reduce([y, x, height - 1 - y, width - 1 - x, y1, x1, height - 1 - y1, width - 1 - x1])
        selectors = {
            "all": np.ones(edge.shape, dtype=bool),
            "border_0_128px": edge < 128,
            "border_128_512px": (edge >= 128) & (edge < 512),
            "interior_ge_512px": edge >= 512,
            f"view={view}": np.ones(edge.shape, dtype=bool),
            f"crop={crop}": np.ones(edge.shape, dtype=bool),
        }
        for name, select in selectors.items():
            accumulator = accumulators.setdefault(name, SampledDeltaAccumulator())
            accumulator.update(gt_delta[select], pred_delta[select])
        candidates += count
        valid_pairs += int(pair_valid.sum())
    return {
        "method": "100000 candidate pairs per image/axis without replacement; seed=42+image_index*10000+14; shared GT-valid coordinates",
        "candidate_positions": candidates,
        "valid_sampled_pairs": valid_pairs,
        "groups": {name: accumulator.result() for name, accumulator in sorted(accumulators.items())},
    }


@torch.inference_mode()
def evaluate_model(model, dataloader, device: torch.device, max_depth: float) -> dict[str, Any]:
    model.eval()
    accumulator = MetricAccumulator()
    crop_accumulators: dict[str, MetricAccumulator] = {}
    view_accumulators: dict[str, MetricAccumulator] = {}
    crop_view_accumulators: dict[str, MetricAccumulator] = {}
    sampled_rows = []
    for image_index, batch in enumerate(dataloader):
        pixel_values = batch["pixel_values"].to(device)
        output = model(pixel_values=pixel_values).predicted_depth
        target = batch["original_depth"][0]
        prediction = F.interpolate(
            output[:, None], size=target.shape[-2:], mode="bilinear", align_corners=True
        )[0, 0]
        valid = torch.isfinite(target) & (target > 0) & (target <= max_depth)
        masked_target = target.clone()
        masked_target[~valid] = 0
        accumulator.update(prediction, masked_target)
        crop = str(batch["crop"][0])
        crop_accumulators.setdefault(crop, MetricAccumulator()).update(prediction, masked_target)
        view = str(batch["view"][0])
        view_accumulators.setdefault(view, MetricAccumulator()).update(prediction, masked_target)
        crop_view_accumulators.setdefault(f"{crop}|{view}", MetricAccumulator()).update(prediction, masked_target)
        sampled_rows.append(sampled_local_shape(prediction, target, valid, image_index, crop, view))
    result = accumulator.result()
    result["per_crop"] = {crop: values.result() for crop, values in sorted(crop_accumulators.items())}
    result["per_view"] = {view: values.result() for view, values in sorted(view_accumulators.items())}
    result["per_crop_view"] = {group: values.result() for group, values in sorted(crop_view_accumulators.items())}
    sampled_aggregates = {}
    for sampled in sampled_rows:
        for group, value in sampled["groups"].items():
            target_group = sampled_aggregates.setdefault(group, SampledDeltaAccumulator())
            # Merge exact sums by reconstructing the sampled sufficient statistics from each group result.
            for field, key in (("all_sampled_pairs", "total"), ("gt_abs_delta_0_5cm", "small"), ("gt_abs_delta_ge_20cm", "large")):
                row = value[field]
                sums = getattr(target_group, key)
                n = row["pair_count"]
                if n:
                    sums[0] += n
                    sums[1] += row["delta_mae_m"] * n
                    sums[2] += row["mean_abs_gt_delta_m"] * n
                    sums[3] += row["mean_abs_pred_delta_m"] * n
                    sums[4] += row["nonzero_gt_pair_count"]
                    sums[5] += row["sign_match_count"]
    result["sampled_local_shape"] = {
        "method": sampled_rows[0]["method"] if sampled_rows else None,
        "candidate_positions": sum(row["candidate_positions"] for row in sampled_rows),
        "valid_sampled_pairs": sum(row["valid_sampled_pairs"] for row in sampled_rows),
        "groups": {name: values.result() for name, values in sorted(sampled_aggregates.items())},
    }
    return result
