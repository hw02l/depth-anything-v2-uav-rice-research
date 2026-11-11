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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for group in ("C0", "F1", "F2"):
        checkpoint_path = args.runs_root / group / f"epoch_{config['training']['epochs']:04d}.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if state["epoch"] != config["training"]["epochs"]:
            raise ValueError(f"{group}: primary model must be the fixed final epoch")
        if state.get("checkpoint_signature") != canonical_signature(state["signature_payload"]):
            raise ValueError(f"{group}: checkpoint signature failed")
        model, load_audit = load_model(args.initial_checkpoint, args.device)
        model.load_state_dict(state["model"], strict=True)
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
            summary["checkpoint_sha256"] = file_sha256(checkpoint_path)
            summary["initial_checkpoint_sha256"] = initial_hashes["model.safetensors"]
            summary["pair_manifest_sha256"] = pair_hash
            summary["strict_load_audit"] = load_audit
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
        "independent_test_used": False, "summaries": summaries,
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
