"""Experimental primitives; not yet wired into the training pipeline."""
from dataclasses import dataclass
import torch
import torch.nn.functional as F

@dataclass(frozen=True)
class Geometry:
    original_hw: tuple
    resized_hw: tuple
    crop_yx: tuple
    crop_hw: tuple
    flipped: bool = False

    def map_xy(self, xy):
        """OpenCV half-pixel resize mapping to crop pixel-center coordinates."""
        oh, ow = self.original_hw
        rh, rw = self.resized_hw
        top, left = self.crop_yx
        ch, cw = self.crop_hw
        x = (xy[..., 0] + .5) * (rw / ow) - .5 - left
        y = (xy[..., 1] + .5) * (rh / oh) - .5 - top
        if self.flipped:
            x = (cw - 1) - x
        return torch.stack((x, y), dim=-1)


def local_loss(prediction, gt, pairs_xy, geometry, beta=.1):
    """One scale; pairs [N,2,2] are original integer (x,y) coordinates.

    prediction [1,1,H,W], gt [H0,W0]. Crop pixel centers and output tensor
    share endpoints under align_corners=True. Empty returns differentiable zero.
    """
    if beta <= 0:
        raise ValueError('beta must be positive')
    if prediction.ndim != 4 or prediction.shape[:2] != (1, 1):
        raise ValueError('prediction must have shape [1,1,H,W]')
    # Selection uses the CPU GT, avoiding a full-resolution GPU transfer.
    pairs = pairs_xy.to(device=gt.device, dtype=torch.long)
    oh, ow = geometry.original_hw
    if tuple(gt.shape) != (oh, ow):
        raise ValueError('GT shape mismatch')
    ch, cw = geometry.crop_hw
    if min(ch, cw, *prediction.shape[-2:]) <= 1:
        raise ValueError('spatial dimensions must exceed one')
    in_original = ((pairs[..., 0] >= 0) & (pairs[..., 0] < ow) &
                   (pairs[..., 1] >= 0) & (pairs[..., 1] < oh)).all(dim=1)
    pairs = pairs[in_original]
    mapped = geometry.map_xy(pairs.to(torch.float32))
    valid_crop = ((mapped[..., 0] >= 0) & (mapped[..., 0] <= cw-1) &
                  (mapped[..., 1] >= 0) & (mapped[..., 1] <= ch-1)).all(dim=1)
    pairs, mapped = pairs[valid_crop], mapped[valid_crop]
    values = gt[pairs[..., 1], pairs[..., 0]]
    valid = (torch.isfinite(values) & (values >= 0.001) & (values <= 65)).all(dim=1)
    mapped, values = mapped[valid], values[valid]
    if values.numel() == 0:
        return prediction.sum() * 0, 0
    mapped = mapped.to(device=prediction.device, dtype=torch.float32)
    values = values.to(device=prediction.device, dtype=torch.float32)
    grid = torch.stack((2*mapped[..., 0]/(cw-1)-1,
                        2*mapped[..., 1]/(ch-1)-1), dim=-1)[None]
    sampled = F.grid_sample(prediction.float(), grid, mode='bilinear', align_corners=True)[0, 0]
    diff_gt = values[:, 0] - values[:, 1]
    error = sampled[:, 0] - sampled[:, 1] - diff_gt
    weight = torch.where(diff_gt.abs() >= .2, 3., 1.)
    loss = F.smooth_l1_loss(error, torch.zeros_like(error), beta=beta, reduction='none')
    return (loss * weight).sum() / weight.sum(), values.shape[0]


def multiscale_loss(prediction, gt, pairs_by_stride, geometry, beta=.1):
    """Equal mean over nonempty scales; connected zero if all are empty.

    Returns per-scale valid counts so that empty-scale behavior is auditable.
    """
    losses, counts = [], {}
    for stride in (14, 28, 56):
        value, count = local_loss(prediction, gt, pairs_by_stride[stride], geometry, beta)
        counts[stride] = count
        if count:
            losses.append(value)
    if not losses:
        return prediction.sum() * 0, counts
    return torch.stack(losses).mean(), counts
