"""Single-image objective adapter for the experimental training loop."""
import torch
from local_loss import Geometry, multiscale_loss
from sampling import sample_pairs


def objective(prediction, target, valid, silog_loss, *, original_gt, geometry,
              image_id, epoch, local_weight, candidates_per_axis=8192, seed=42):
    """Return SiLog, local loss, and auditable counts without touching global RNG.

    This interface requires batch size one. It accepts an explicit geometry rather
    than inferring it from tensors, and samples even for E1 when explicitly called.
    """
    if prediction.ndim != 3 or prediction.shape[0] != 1:
        raise ValueError('only batch size one is supported')
    if local_weight not in (0.0, 0.05, 0.20):
        raise ValueError('local weight must match a planned group')
    silog = silog_loss(prediction, target, valid)
    if local_weight == 0:
        # E1 is the SiLog-only continuation control. Avoid candidate generation,
        # GT indexing, grid_sample, and any local-loss overhead in this group.
        local = prediction.sum() * 0.0
        counts = {14: 0, 28: 0, 56: 0}
        total = silog
    else:
        pairs = {s: sample_pairs(image_id,epoch,geometry.original_hw,s,candidates_per_axis,seed)
                 for s in (14,28,56)}
        local, counts = multiscale_loss(prediction[:,None], original_gt, pairs, geometry)
        total = silog + local_weight * local
    if not torch.isfinite(total):
        raise FloatingPointError('nonfinite total loss')
    return total, silog, local, counts
