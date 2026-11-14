from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "local_huber_20261105"))
sys.path.insert(0, str(HERE))
from data_patch import read_manifest
from evaluate_patch import evaluate_combo
from patch_engine import canonical_signature, file_sha256
from train_patch import check_initial_hashes, load_model, verify_manifest

EXPECTED_EPOCH5_SHA256 = {
    "C0": "2fd190727c7a5124c56c10e0a2dd8cb09c6cb62a0f92b34fce97ad394281b35a",
    "F1": "ea1f74c5ff17c0b266152588b5d53f8b24611418f125f2fbf4d9cc7447e49f67",
    "F2": "8124d72a7008c9f3a93dbcce34142d1593063fba2da6a1a58306d55aaf201538",
}


def atomic_json(path: Path, value):
    temporary = path.with_name("." + path.name + "." + str(os.getpid()) + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_grid_csv(path: Path, rows):
    temporary = path.with_name("." + path.name + "." + str(os.getpid()) + ".tmp")
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with temporary.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def run(args):
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Full Validation128 grid evaluation requires CUDA")
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    if config.get("status") != "locked":
        raise RuntimeError("Evaluation requires the locked final config")
    initial_hashes = check_initial_hashes(args.initial_checkpoint)
    rows = read_manifest(args.val_manifest)
    verify_manifest(rows, args.val_manifest, ROOT / "configs/uav3dcrop_verified/validation.csv", 128, "validation")
    pair_hash = file_sha256(args.pair_manifest)
    if args.output_dir.exists():
        raise FileExistsError(f"Fresh RGB-fix output directory required; preserving existing path: {args.output_dir}")
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(exist_ok=False)
    summaries = []
    for group in ("C0", "F1", "F2"):
        checkpoint_path = args.runs_root / group / f"epoch_{config['training']['epochs']:04d}.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        checkpoint_hash = file_sha256(checkpoint_path)
        if checkpoint_hash != EXPECTED_EPOCH5_SHA256[group]:
            raise ValueError(f"{group}: immutable epoch-5 checkpoint SHA256 mismatch")
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if state["epoch"] != config["training"]["epochs"]:
            raise ValueError(f"{group}: primary model must be the fixed final epoch")
        if state.get("checkpoint_signature") != canonical_signature(state["signature_payload"]):
            raise ValueError(f"{group}: checkpoint signature failed")
        if state.get("config", {}).get("group") != group:
            raise ValueError(f"{group}: checkpoint group/config mismatch")
        model, load_audit = load_model(args.initial_checkpoint, args.device)
        target_state = model.state_dict()
        checkpoint_state = state.get("model")
        if not isinstance(checkpoint_state, dict) or set(checkpoint_state) != set(target_state):
            raise ValueError(f"{group}: checkpoint state-dict keys differ")
        for name, tensor in checkpoint_state.items():
            if tensor.shape != target_state[name].shape or tensor.dtype != target_state[name].dtype:
                raise ValueError(f"{group}: checkpoint tensor shape/dtype mismatch: {name}")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{group}: non-finite checkpoint tensor: {name}")
        model.load_state_dict(checkpoint_state, strict=True)
        loaded_state = model.state_dict()
        if any(not torch.equal(tensor, loaded_state[name].detach().cpu())
               for name, tensor in checkpoint_state.items()):
            raise ValueError(f"{group}: loaded model weights differ from immutable checkpoint")
        model.eval()
        for method in ("P0", "P1", "P2"):
            combo_dir = args.output_dir / f"{group}_{method}"
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            summary = evaluate_combo(
                model, rows, args.rgb_root, args.depth_root, args.device, method,
                args.pair_manifest, combo_dir, group=group,
                epoch=config["training"]["epochs"],
            )
            summary["checkpoint_sha256"] = checkpoint_hash
            summary["initial_checkpoint_sha256"] = initial_hashes["model.safetensors"]
            summary["pair_manifest_sha256"] = pair_hash
            summary["strict_load_audit"] = load_audit
            summary["checkpoint_state_load_audit"] = {
                "strict": True, "key_count": len(checkpoint_state),
                "shape_dtype_finite_verified": True, "loaded_values_exact": True,
            }
            summary["rgb_input_contract"] = "load_pair_float32_0_1_roundtrip_rint_to_uint8_0_255_v1"
            summary["evaluation_code_revision"] = "rgb_input_fix_20261114_v1"
            summary["cuda_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
            summary["cuda_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
            atomic_json(combo_dir / f"evaluation_{group}_{method}.json", summary)
            summaries.append({
                "group": group, "method": method, "epoch": summary["epoch"],
                "image_count": summary["image_count"],
                **summary["absolute"],
                "runtime_seconds": summary["runtime_seconds"],
                "runtime_tile_forward_seconds": summary["runtime_tile_forward_seconds"],
                "tile_count": summary["tile_count"],
                "cuda_peak_allocated_bytes": summary["cuda_peak_allocated_bytes"],
                "cuda_peak_reserved_bytes": summary["cuda_peak_reserved_bytes"],
                "checkpoint_sha256": summary["checkpoint_sha256"],
            })
            print(json.dumps({
                "group": group, "method": method, "rmse_m": summary["absolute"]["rmse_m"],
                "full14_mae_m": summary["absolute"]["full14_mae_m"],
                "runtime_seconds": summary["runtime_seconds"],
                "tiles": summary["tile_count"],
            }, ensure_ascii=False), flush=True)
        del model, state
        torch.cuda.empty_cache()
    write_grid_csv(args.output_dir / "grid_summary.csv", summaries)
    atomic_json(args.output_dir / "grid_summary.json", {
        "status": "complete", "groups": ["C0", "F1", "F2"], "methods": ["P0", "P1", "P2"],
        "primary_epoch": config["training"]["epochs"], "image_count": 128,
        "train_manifest_sha256": file_sha256(args.train_manifest),
        "validation_manifest_sha256": file_sha256(args.val_manifest),
        "pair_manifest_sha256": pair_hash, "initial_checkpoint_hashes": initial_hashes,
        "independent_test_used": False,
        "rgb_input_contract": "load_pair_float32_0_1_roundtrip_rint_to_uint8_0_255_v1",
        "evaluation_code_revision": "rgb_input_fix_20261114_v1", "summaries": summaries,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--rgb-root", type=Path, required=True)
    parser.add_argument("--depth-root", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

