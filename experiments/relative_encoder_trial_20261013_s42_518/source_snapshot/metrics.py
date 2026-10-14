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


@torch.inference_mode()
def evaluate_model(model, dataloader, device: torch.device, max_depth: float) -> dict[str, Any]:
    model.eval()
    accumulator = MetricAccumulator()
    crop_accumulators: dict[str, MetricAccumulator] = {}
    for batch in dataloader:
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
    result = accumulator.result()
    result["per_crop"] = {crop: values.result() for crop, values in sorted(crop_accumulators.items())}
    return result
