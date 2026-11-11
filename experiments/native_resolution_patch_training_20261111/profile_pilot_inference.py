from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))
from data_patch import NativePatchDataset, load_pair, read_manifest
from inference import infer_depth, profile_tiles
from patch_engine import file_sha256
from train_patch import check_initial_hashes, load_model


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_name("." + path.name + "." + str(os.getpid()) + ".tmp")
    with temporary.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def under(path: Path, root: Path, label: str) -> None:
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"{label} must be on local GPU work storage: {path}")


def run(args) -> dict:
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Inference pilot requires CUDA")
    for label, path in (("checkpoint", args.initial_checkpoint), ("validation pilot manifest", args.val_manifest),
                        ("fixed validation selection", args.canonical_pilot_manifest),
                        ("training pilot manifest", args.train_pilot_manifest),
                        ("RGB root", args.rgb_root), ("depth root", args.depth_root),
                        ("output", args.output_dir)):
        under(path, args.gpu_work_root, label)
    rows = read_manifest(args.val_manifest)
    if len(rows) != 16 or any(row["split"] != "validation" or row["day"] != "2025/Day047" for row in rows):
        raise ValueError("The fixed 16-image Day047 validation pilot is required")
    canonical_rows = read_manifest(ROOT / "configs/uav3dcrop_verified/validation.csv")
    if len(canonical_rows) != 128 or any(row["split"] != "validation" for row in canonical_rows):
        raise ValueError("Canonical Validation128 manifest is unavailable")
    canonical_by_path = {row["image_relpath"]: row for row in canonical_rows}
    if any(row["image_relpath"] not in canonical_by_path or row != canonical_by_path[row["image_relpath"]]
           for row in rows):
        raise ValueError("Pilot Validation rows differ from canonical Validation128")
    with args.canonical_pilot_manifest.open("r", encoding="utf-8-sig", newline="") as stream:
        fixed = {row["image_relpath"] for row in csv.DictReader(stream)}
    if {row["image_relpath"] for row in rows} != fixed:
        raise ValueError("Validation pilot image IDs differ from the fixed pilot selection")
    check_initial_hashes(args.initial_checkpoint)
    model, load_audit = load_model(args.initial_checkpoint, "cuda")
    model.eval()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    dummy = torch.zeros((1, 3, 518, 518), device="cuda", dtype=torch.float32)
    with torch.inference_mode():
        model(pixel_values=dummy).predicted_depth
    torch.cuda.synchronize()
    del dummy

    read_rows = []
    rgb_images = []
    for index, row in enumerate(rows):
        started = time.perf_counter()
        rgb, gt = load_pair(row, args.rgb_root, args.depth_root)
        elapsed = time.perf_counter() - started
        if rgb.shape[:2] != gt.shape:
            raise ValueError(f"RGB/GT pair changed shape: {row['image_relpath']}")
        read_rows.append({"image_index": index, "image_relpath": row["image_relpath"],
                          "rgb_gt_read_seconds": elapsed, "shape": f"{gt.shape[0]}x{gt.shape[1]}",
                          "research_metric_computed": False})
        rgb_images.append(rgb)
        del gt

    # Separately measure the actual training crop loader, including RGB and TIFF reads.
    train_rows = read_manifest(args.train_pilot_manifest)
    if len(train_rows) != 8 or any(row["split"] != "train" for row in train_rows):
        raise ValueError("The fixed eight-image training pilot is required")
    supply_rows = []
    for group in ("C0", "F1", "F2"):
        dataset = NativePatchDataset(train_rows, args.rgb_root, args.depth_root,
                                     group, repeats=1, seed=42, training=True)
        dataset.set_epoch(1)
        samples = []
        started = time.perf_counter()
        for index in range(len(dataset)):
            sample = dataset[index]
            samples.append({"valid_gt_fraction": sample["metadata"]["valid_gt_fraction"],
                            "image_relpath": sample["metadata"]["image_relpath"]})
            del sample
        elapsed = time.perf_counter() - started
        supply_rows.append({"group": group, "images": len(dataset),
                            "total_seconds": elapsed,
                            "seconds_per_crop": elapsed / len(dataset),
                            "mean_valid_gt_fraction": float(np.mean([s["valid_gt_fraction"] for s in samples]))})

    inference_rows = []
    for method in ("P0", "P1", "P2"):
        for index, (row, rgb) in enumerate(zip(rows, rgb_images)):
            torch.cuda.reset_peak_memory_stats()
            inferred = infer_depth(model, rgb, "cuda", method)
            profile = profile_tiles(inferred)
            inference_rows.append({
                "image_index": index, "image_relpath": row["image_relpath"],
                "method": method, "tile_count": profile["tile_count"],
                "inference_elapsed_seconds": inferred.elapsed_seconds,
                "tile_forward_seconds": inferred.tile_seconds,
                "coverage_min": profile["coverage_min"], "coverage_max": profile["coverage_max"],
                "nonoverlap_fraction": profile["nonoverlap_fraction"],
                "overlap_fraction": profile["overlap_fraction"],
                "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                "prediction_saved": False,
            })
            del inferred
        torch.cuda.empty_cache()

    summaries = []
    for method in ("P0", "P1", "P2"):
        subset = [row for row in inference_rows if row["method"] == method]
        times = [row["inference_elapsed_seconds"] for row in subset]
        summaries.append({
            "method": method, "images": len(subset),
            "total_seconds": float(sum(times)),
            "median_seconds_per_image": float(statistics.median(times)),
            "total_tiles": int(sum(row["tile_count"] for row in subset)),
            "max_cuda_allocated_bytes": int(max(row["cuda_peak_allocated_bytes"] for row in subset)),
            "max_cuda_reserved_bytes": int(max(row["cuda_peak_reserved_bytes"] for row in subset)),
        })
    write_csv(args.output_dir / "validation_pilot_read_times.csv", read_rows)
    write_csv(args.output_dir / "training_crop_supply.csv", supply_rows)
    write_csv(args.output_dir / "inference_profile.csv", inference_rows)
    result = {
        "status": "complete", "device": torch.cuda.get_device_name(0),
        "checkpoint_sha256": file_sha256(args.initial_checkpoint / "model.safetensors"),
        "validation_pilot_images": [row["image_relpath"] for row in rows],
        "train_pilot_images": [row["image_relpath"] for row in train_rows],
        "validation_rgb_gt_read_seconds_total": float(sum(row["rgb_gt_read_seconds"] for row in read_rows)),
        "training_crop_supply": supply_rows, "inference": summaries,
        "strict_load_audit": load_audit, "independent_test_used": False,
        "validation_metrics_used_for_selection": False, "prediction_cache_saved": False,
    }
    (args.output_dir / "pilot_inference_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description="Profile fixed validation reads, train-crop supply, and P0/P1/P2 inference on GPU.")
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--canonical-pilot-manifest", type=Path, required=True)
    parser.add_argument("--train-pilot-manifest", type=Path, required=True)
    parser.add_argument("--rgb-root", type=Path, required=True)
    parser.add_argument("--depth-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-work-root", type=Path, default=Path("/work"))
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
