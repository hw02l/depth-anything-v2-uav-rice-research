from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


RELATIVE_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
METRIC_OUTDOOR_MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"
METRIC_INDOOR_MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"


@dataclass
class DepthEstimator:
    processor: object
    model: object
    device: torch.device
    kind: Literal["relative", "metric"]
    model_id: str

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> np.ndarray:
        """Return a HxW depth array in model units, resized to the RGB image."""
        rgb = image.convert("RGB")
        inputs = self.processor(images=rgb, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        outputs = self.model(**inputs)
        predicted = outputs.predicted_depth
        predicted = F.interpolate(
            predicted.unsqueeze(1),
            size=rgb.size[::-1],
            mode="bicubic",
            align_corners=False,
        )
        return predicted[0, 0].detach().cpu().numpy().astype(np.float32)

    @torch.inference_mode()
    def predict_batch(self, images: list[Image.Image]) -> list[np.ndarray]:
        """Predict a batch of equally sized images and return one HxW array per image."""
        if not images:
            return []
        rgb_images = [image.convert("RGB") for image in images]
        sizes = [image.size[::-1] for image in rgb_images]
        if len(set(sizes)) != 1:
            return [self.predict(image) for image in rgb_images]
        inputs = self.processor(images=rgb_images, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        outputs = self.model(**inputs)
        predicted = F.interpolate(
            outputs.predicted_depth.unsqueeze(1),
            size=sizes[0],
            mode="bicubic",
            align_corners=False,
        )[:, 0]
        return [item.detach().cpu().numpy().astype(np.float32) for item in predicted]


def choose_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_estimator(
    kind: Literal["relative", "metric"],
    model_id: str | None = None,
    device: str = "auto",
) -> DepthEstimator:
    if model_id is None:
        model_id = RELATIVE_MODEL_ID if kind == "relative" else METRIC_OUTDOOR_MODEL_ID

    target = choose_device(device)
    cache_dir = Path(os.environ.get("DA2_HF_CACHE", "work/hf_cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    processor = AutoImageProcessor.from_pretrained(model_id, cache_dir=str(cache_dir))
    model = AutoModelForDepthEstimation.from_pretrained(model_id, cache_dir=str(cache_dir))
    model = model.to(target).eval()
    return DepthEstimator(processor, model, target, kind, model_id)
