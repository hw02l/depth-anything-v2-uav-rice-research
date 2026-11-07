"""Training step components, not an executable experiment runner."""
from contextlib import nullcontext
import torch
from local_loss import Geometry
from objective import objective


def step_batch(model, batch, silog_loss, device, epoch, local_weight, *, amp=False, candidates_per_axis=8192):
    """Forward and construct loss for one uncollated train sample.

    Accumulation, scaler, optimizer and persistence are intentionally owned by
    the caller. Full original GT stays on CPU.
    """
    pixels=batch['pixel_values'].unsqueeze(0).to(device)
    target=batch['depth'].unsqueeze(0).to(device)
    valid=torch.isfinite(target)&(target>=0.001)&(target<=65)
    geometry=Geometry(**batch['geometry'])
    context=torch.autocast('cuda',dtype=torch.float16) if amp else nullcontext()
    with context:
        pred=model(pixel_values=pixels).predicted_depth
    # Always run sparse sampling and SiLog in float32, including under AMP.
    return objective(pred.float(),target.float(),valid,silog_loss,
                     original_gt=batch['original_depth'],geometry=geometry,
                     image_id=batch['image_relpath'],epoch=epoch,
                     local_weight=local_weight, candidates_per_axis=candidates_per_axis)
