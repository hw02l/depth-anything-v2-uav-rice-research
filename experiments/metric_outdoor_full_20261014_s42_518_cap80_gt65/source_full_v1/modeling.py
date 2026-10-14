from __future__ import annotations

import re
from typing import Any
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForDepthEstimation


RELATIVE_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
METRIC_OUTDOOR_MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"
RELATIVE_MODEL_REVISION = "5426e4f0f36572d16453bbda7a8389317b1bef99"
METRIC_OUTDOOR_MODEL_REVISION = "fd2c22027eaf20374204f14099b8341e1925ad39"


def _source_revision(model: torch.nn.Module, source: str) -> str | None:
    revision = getattr(model.config, "_commit_hash", None)
    if revision:
        return str(revision)
    candidate = Path(source).name
    return candidate if re.fullmatch(r"[0-9a-f]{40}", candidate) else None


def build_metric_model(
    initialization: str,
    *,
    max_depth: float,
    relative_model: str = RELATIVE_MODEL_ID,
    metric_model: str = METRIC_OUTDOOR_MODEL_ID,
    local_files_only: bool = True,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Create the HF DAv2-Small metric model through either supported init path.

    ``relative_encoder`` transfers only the ViT backbone from Relative Small;
    the metric neck/head are newly initialized. ``metric_outdoor`` loads the
    full Outdoor Small checkpoint before adapting its output range.
    """
    if max_depth <= 0:
        raise ValueError("max_depth must be positive and expressed in metres")

    if initialization == "relative_encoder":
        source = AutoModelForDepthEstimation.from_pretrained(
            relative_model, local_files_only=local_files_only
        )
        config = AutoConfig.from_pretrained(relative_model, local_files_only=local_files_only)
        config.depth_estimation_type = "metric"
        config.max_depth = float(max_depth)
        model = AutoModelForDepthEstimation.from_config(config)

        target_state = model.state_dict()
        source_state = source.state_dict()
        transferred = {
            key: value
            for key, value in source_state.items()
            if key.startswith("backbone.")
            and key in target_state
            and target_state[key].shape == value.shape
        }
        if not transferred:
            raise RuntimeError(
                "No Relative Small backbone weights matched the Metric model. "
                "Check that both checkpoints use the Hugging Face DAv2-Small architecture."
            )
        incompatible = model.load_state_dict(transferred, strict=False)
        transferred_keys = set(transferred)
        missing_backbone = [
            key for key in incompatible.missing_keys
            if key.startswith("backbone.") and key not in transferred_keys
        ]
        source_revision = _source_revision(source, relative_model)
        del target_state, source_state, source
        if missing_backbone:
            raise RuntimeError(f"Relative encoder transfer was incomplete: {missing_backbone[:5]}")
        source_description = relative_model
        transferred_tensors = len(transferred)
        transferred_parameters = sum(value.numel() for value in transferred.values())
        del transferred

    elif initialization == "metric_outdoor":
        model = AutoModelForDepthEstimation.from_pretrained(
            metric_model, local_files_only=local_files_only
        )
        if getattr(model.config, "depth_estimation_type", None) != "metric":
            raise ValueError("The supplied comparison checkpoint is not configured as metric depth")
        source_description = metric_model
        source_revision = _source_revision(model, metric_model)
        transferred_tensors = 0
        transferred_parameters = 0
    else:
        raise ValueError("initialization must be 'relative_encoder' or 'metric_outdoor'")

    model.config.depth_estimation_type = "metric"
    model.config.max_depth = int(max_depth)
    # The HF head caches this value at construction time.
    model.head.max_depth = float(max_depth)
    model.train()
    return model, {
        "initialization": initialization,
        "source": source_description,
        "source_revision": source_revision,
        "max_depth_m": float(max_depth),
        "transferred_encoder_tensors": transferred_tensors,
        "transferred_encoder_parameters": transferred_parameters,
        "metric_head": "sigmoid * max_depth",
    }
