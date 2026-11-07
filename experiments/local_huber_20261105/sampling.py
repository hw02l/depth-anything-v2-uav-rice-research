"""Reproducible train-only candidate pairs, independent of model predictions."""
import hashlib
import numpy as np
import torch


def sample_pairs(image_id: str, epoch: int, shape: tuple[int, int], stride: int,
                 candidates_per_axis: int, seed: int = 42) -> torch.Tensor:
    """Sample with replacement uniformly within original-image bounds.

    Separate RNG streams are keyed by image, epoch, stride and axis. Subsequent
    GT and crop filtering is performed by local_loss; no prediction is read.
    """
    h, w = shape
    if epoch < 1 or stride < 1 or candidates_per_axis < 1 or stride >= min(h,w):
        raise ValueError('invalid epoch, shape, stride or candidate count')
    blocks = []
    for axis in (0, 1):
        key = f'{seed}\0{image_id}\0{epoch}\0{stride}\0{axis}'.encode('utf-8')
        axis_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], 'little')
        rng = np.random.default_rng(axis_seed)
        y = rng.integers(0, h - (stride if axis == 0 else 0), candidates_per_axis)
        x = rng.integers(0, w - (stride if axis == 1 else 0), candidates_per_axis)
        yy = y + (stride if axis == 0 else 0)
        xx = x + (stride if axis == 1 else 0)
        blocks.append(np.stack((np.stack((x,y),axis=-1), np.stack((xx,yy),axis=-1)),axis=1))
    return torch.from_numpy(np.concatenate(blocks, axis=0))
