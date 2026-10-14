from __future__ import annotations

import torch
from torch import Tensor


def silog_loss(
    prediction: Tensor,
    target: Tensor,
    valid: Tensor,
    *,
    variance_weight: float = 0.5,
    epsilon: float = 1e-6,
) -> Tensor:
    """Scale-invariant logarithmic loss used by the official metric recipe.

    Inputs are depth in metres with shape (B,H,W). Invalid target pixels are
    excluded. Returning a graph-connected zero for an empty mask avoids NaNs.
    """
    valid = valid.bool() & torch.isfinite(target) & torch.isfinite(prediction)
    valid = valid & (target > 0) & (prediction > 0)
    if not torch.any(valid):
        return prediction.sum() * 0.0

    log_difference = torch.log(target[valid].clamp_min(epsilon)) - torch.log(
        prediction[valid].clamp_min(epsilon)
    )
    variance = log_difference.square().mean() - variance_weight * log_difference.mean().square()
    return torch.sqrt(variance.clamp_min(epsilon))
