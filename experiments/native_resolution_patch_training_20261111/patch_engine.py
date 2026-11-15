from __future__ import annotations

import hashlib
import json
import os
import random
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch


def canonical_signature(config: dict[str, Any]) -> str:
    raw = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_torch_save(path: str | Path, value: Any) -> str:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Existing output is preserved: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("xb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return file_sha256(path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def capture_rng() -> dict[str, Any]:
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state: dict[str, Any]) -> None:
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if not required.issubset(state):
        raise ValueError("Checkpoint lacks complete RNG state")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA RNG cannot be restored on CPU")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_epoch(run_dir: str | Path, state: dict[str, Any]) -> tuple[Path, str]:
    path = Path(run_dir) / f"epoch_{int(state['epoch']):04d}.pt"
    state["checkpoint_signature"] = canonical_signature(state["signature_payload"])
    digest = atomic_torch_save(path, state)
    return path, digest


def latest_epoch(run_dir: str | Path) -> Path:
    candidates = sorted(Path(run_dir).glob("epoch_[0-9][0-9][0-9][0-9].pt"))
    if not candidates:
        raise FileNotFoundError(f"No completed epoch checkpoint under {run_dir}")
    return candidates[-1]


def load_epoch(path: str | Path, model, optimizer, scheduler, scaler,
               expected_signature: dict[str, Any], device: str) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    expected = canonical_signature(expected_signature)
    if state.get("checkpoint_signature") != expected:
        raise ValueError("Resume signature mismatch; checkpoint preserved")
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    restore_rng(state["rng"])
    if state.get("device_type") != ("cuda" if device.startswith("cuda") else "cpu"):
        raise ValueError("Resume device type differs")
    return state


def cpu_model_state(model) -> dict[str, torch.Tensor]:
    return {key: value.detach().to(device="cpu").contiguous() for key, value in model.state_dict().items()}
