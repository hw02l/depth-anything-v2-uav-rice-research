"""GT-free global/local frequency fusion. No fitted scale, offset, or weights."""
import cv2
import numpy as np


def global_local_fusion(global_depth, tiled_depth, radius=107, alpha=1.0):
    """P = B + alpha * [(Q-L(Q)) - (B-L(B))].

    L is a normalized (2r+1)^2 box filter, BORDER_REFLECT_101, in original
    pixel coordinates. alpha=1 gives low-pass B plus high-pass Q. GT is not
    an input. Constant tile/global depth offsets are removed analytically.
    """
    if global_depth.shape != tiled_depth.shape or global_depth.ndim != 2:
        raise ValueError('Aligned original-resolution depths required')
    if radius < 1 or not 0 <= alpha <= 1:
        raise ValueError('Invalid fusion parameters')
    if not np.isfinite(global_depth).all() or not np.isfinite(tiled_depth).all():
        raise ValueError('Invalid parent predictions')
    b, q = np.asarray(global_depth,np.float32), np.asarray(tiled_depth,np.float32)
    size=(2*radius+1,2*radius+1)
    lb=cv2.boxFilter(b,-1,size,normalize=True,borderType=cv2.BORDER_REFLECT_101)
    lq=cv2.boxFilter(q,-1,size,normalize=True,borderType=cv2.BORDER_REFLECT_101)
    return b + np.float32(alpha)*((q-lq)-(b-lb))
