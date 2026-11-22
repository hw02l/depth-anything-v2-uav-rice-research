from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import subprocess

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
import torch


EXPERIMENT = "best_model_inference_combination_20261122"
DEPTH_CONDITIONS = (
    "epoch20-B0", "epoch20-B3", "C0-P0", "C0-B3", "E3-B0", "E3-B3"
)
DISPLAY_NAMES = {
    "epoch20-B0": "epoch20 / B0",
    "epoch20-B3": "epoch20 / B3",
    "C0-P0": "C0 / P0",
    "C0-B3": "C0 / B3",
    "E3-B0": "E3 / B0",
    "E3-B3": "E3 / B3",
}
DEPTH_MAX_M = 65.0
ERROR_MAX_M = 10.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_prediction(path: Path, shape: tuple[int, int]) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if archive.files != ["prediction_m"]:
            raise ValueError(f"Unexpected prediction archive schema: {path}")
        prediction = np.asarray(archive["prediction_m"], dtype=np.float32)
    if prediction.shape != shape or not np.isfinite(prediction).all():
        raise ValueError(f"Invalid prediction shape or values: {path}")
    return prediction


def load_prediction_npy(path: Path, shape: tuple[int, int]) -> np.ndarray:
    prediction = np.load(path, mmap_mode="r", allow_pickle=False)
    if prediction.shape != shape or prediction.dtype != np.float32 or not np.isfinite(prediction).all():
        raise ValueError(f"Invalid generated prediction shape, dtype, or values: {path}")
    return prediction


def predict_b3_from_global(model, rgb: np.ndarray, global_prediction: np.ndarray, runner, device: str):
    """Use the locked runner's tiling and fusion helpers while reusing its B0 map."""
    merger = runner.m.TileMerger(rgb.shape[:2])
    boxes = runner.m.tiles(rgb.shape[:2], runner.TILE_SIZE, runner.OVERLAP)
    if not boxes:
        raise ValueError("Tiling produced no boxes")
    for y, x, bottom, right in boxes:
        tile_prediction = runner.infer_one(model, rgb[y:bottom, x:right], device)
        merger.add((y, x, bottom, right), tile_prediction)
        del tile_prediction
    tile_blended = merger.result()
    fused = runner.fuse_depth(global_prediction, tile_blended)
    if fused.shape != rgb.shape[:2] or not np.isfinite(fused).all():
        raise FloatingPointError("B3 fusion produced invalid output")
    audit = {
        "tile_count": len(boxes),
        "tile_boxes_yxbr": [list(map(int, box)) for box in boxes],
        "fusion_min_m": float(fused.min()),
        "fusion_max_m": float(fused.max()),
        "fusion_nonfinite_count": int((~np.isfinite(fused)).sum()),
        "fusion_nonpositive_count": int((fused <= 0).sum()),
        "fusion_above_80m_count": int((fused > 80.001).sum()),
    }
    return fused.astype(np.float32, copy=False), audit


def add_depth_panel(axis, image, title, cmap="viridis"):
    shown = np.ma.masked_invalid(np.ma.asarray(image))
    return axis.imshow(shown, cmap=cmap, vmin=0.0, vmax=DEPTH_MAX_M, interpolation="nearest")


def render_depth_figure(rgb, gt, predictions, valid, image_index: int, crop: str, view: str, output: Path):
    fig, axes = plt.subplots(2, 4, figsize=(24, 12), constrained_layout=True)
    axes = axes.ravel()
    axes[0].imshow(rgb, interpolation="nearest")
    axes[0].set_title("RGB")
    gt_shown = np.ma.array(gt, mask=~valid)
    image_artist = add_depth_panel(axes[1], gt_shown, "GT")
    axes[1].set_title("GT")
    for axis, condition in zip(axes[2:], DEPTH_CONDITIONS):
        shown = np.ma.array(predictions[condition], mask=~valid)
        image_artist = add_depth_panel(axis, shown, DISPLAY_NAMES[condition])
        axis.set_title(DISPLAY_NAMES[condition])
    for axis in axes:
        axis.set_axis_off()
    colorbar = fig.colorbar(image_artist, ax=axes[1:], shrink=0.78, pad=0.015)
    colorbar.set_label("Depth (m); common scale 0–65 m")
    fig.suptitle(f"Validation image {image_index:03d} — {crop}, {view} — common depth scale", fontsize=16)
    fig.savefig(output, dpi=110)
    plt.close(fig)


def render_error_figure(gt, predictions, valid, image_index: int, crop: str, view: str, output: Path):
    fig, axes = plt.subplots(2, 3, figsize=(21, 12), constrained_layout=True)
    axes = axes.ravel()
    last_artist = None
    over_range = {}
    for axis, condition in zip(axes, DEPTH_CONDITIONS):
        errors = np.abs(predictions[condition].astype(np.float32) - gt.astype(np.float32))
        over_range[condition] = int(np.count_nonzero(valid & (errors > ERROR_MAX_M)))
        shown = np.ma.array(errors, mask=~valid)
        last_artist = axis.imshow(shown, cmap="magma", vmin=0.0, vmax=ERROR_MAX_M, interpolation="nearest")
        axis.set_title(DISPLAY_NAMES[condition])
        axis.set_axis_off()
    colorbar = fig.colorbar(last_artist, ax=axes.tolist(), shrink=0.78, pad=0.015)
    colorbar.set_label("Absolute depth error (m); common scale 0–10 m, clipped above 10 m")
    fig.suptitle(f"Validation image {image_index:03d} — {crop}, {view} — absolute error", fontsize=16)
    fig.savefig(output, dpi=110)
    plt.close(fig)
    return over_range


def main():
    parser = argparse.ArgumentParser(description="Create private fixed-eight six-condition depth-map figures.")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--base-model-dir", type=Path, required=True)
    parser.add_argument("--c0-checkpoint", type=Path, required=True)
    parser.add_argument("--e3-checkpoint", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--image-inventory", type=Path, required=True)
    parser.add_argument("--rgb-root", type=Path, required=True)
    parser.add_argument("--depth-root", type=Path, required=True)
    parser.add_argument("--visual-indices", type=Path, required=True)
    parser.add_argument("--existing-predictions", type=Path, required=True)
    parser.add_argument("--existing-summary", type=Path, required=True)
    parser.add_argument("--baseline-b3-per-image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    runner_path = repo_root / "experiments" / EXPERIMENT / "run_b3.py"
    sys.path.insert(0, str(runner_path.parent))
    import run_b3 as runner  # pylint: disable=import-outside-toplevel

    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This fixed-image inference requires the audited CUDA device")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {args.output_dir}")
    # The full-validation run peaked below 0.4 GiB reserved. Keep this diagnostic
    # process below 15% of VRAM to leave ample headroom for the authorized user.
    torch.cuda.set_per_process_memory_fraction(0.15)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    inventory_rows = list(csv.DictReader(args.image_inventory.open(encoding="utf-8-sig", newline="")))
    validation_rows = runner.read_validation_rows(args.validation_manifest, inventory_rows)
    with args.visual_indices.open(encoding="utf-8-sig", newline="") as stream:
        indices = sorted({int(row["image_index"]) for row in csv.DictReader(stream)})
    if len(indices) != 8 or any(index < 0 or index >= 128 for index in indices):
        raise ValueError(f"Expected eight fixed Validation128 indices, got {indices}")
    for path, expected in ((args.validation_manifest, runner.VALIDATION_SHA256),
                           (args.image_inventory, runner.INPUT_INVENTORY_SHA256),
                           (args.c0_checkpoint, runner.CHECKPOINT_SHA256["C0"]),
                           (args.e3_checkpoint, runner.CHECKPOINT_SHA256["E3"])):
        if sha256(path) != expected:
            raise ValueError(f"SHA256 mismatch for {path}")

    for path in (args.existing_summary, args.baseline_b3_per_image):
        if not path.is_file():
            raise FileNotFoundError(path)
    with args.existing_summary.open(encoding="utf-8-sig", newline="") as stream:
        summary_rows = list(csv.DictReader(stream))
    existing_rows = {(row["model_id"], int(row["image_index"])): row for row in summary_rows}
    with args.baseline_b3_per_image.open(encoding="utf-8-sig", newline="") as stream:
        baseline_rows = [row for row in csv.DictReader(stream) if row.get("epoch") == "20" and row.get("method") == "B3"]
    baseline_by_index = {int(row["image_index"]): row for row in baseline_rows}
    if len(baseline_by_index) != 128:
        raise ValueError("Expected the frozen 128-image epoch20-B3 baseline rows")

    command = list(sys.argv)
    gpu_before = subprocess_check_gpu()
    if gpu_before["utilization_percent"] > 5 or gpu_before["free_mib"] < 5000:
        raise RuntimeError(f"GPU preflight rejected this diagnostic run: {gpu_before}")

    if sha256(args.pair_manifest) != runner.PAIR_SHA256:
        raise ValueError("Fixed pair manifest SHA256 mismatch")
    pair_images = json.loads(args.pair_manifest.read_text(encoding="utf-8"))["images"]
    args.output_dir.mkdir(parents=True)
    work = args.output_dir / "temporary_predictions"
    work.mkdir()
    load_audits = {}
    fusion_audits = {}
    input_records = {}
    try:
        for group in ("C0", "E3", "E0"):
            model, load_audit = runner.load_model(
                group, args.base_model_dir,
                args.c0_checkpoint if group == "C0" else args.e3_checkpoint if group == "E3" else None,
                args.device,
            )
            load_audits[group] = load_audit
            for image_index in indices:
                row = validation_rows[image_index]
                rgb_path = args.rgb_root / row["image_relpath"]
                gt_path = args.depth_root / row["depth_relpath"]
                bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
                if bgr is None:
                    raise FileNotFoundError(rgb_path)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                gt = np.asarray(tifffile.imread(gt_path), dtype=np.float32)
                if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[:2] != gt.shape:
                    raise ValueError(f"RGB/GT type or coordinate mismatch at image {image_index}")
                if sha256(rgb_path) != inventory_rows[image_index]["rgb_sha256"] or sha256(gt_path) != inventory_rows[image_index]["gt_sha256"]:
                    raise ValueError(f"RGB/GT hash mismatch at image {image_index}")
                if rgb.min() == rgb.max():
                    raise ValueError(f"Constant RGB image at image {image_index}")
                valid = runner.m.valid_gt(gt)
                if not valid.any():
                    raise ValueError(f"No valid GT pixels at image {image_index}")
                global_prediction = runner.infer_one(model, rgb, args.device)
                condition = "C0-P0" if group == "C0" else "E3-B0" if group == "E3" else "epoch20-B0"
                out_path = work / f"{condition.replace('-', '_')}_{image_index:03d}.npy"
                np.save(out_path, global_prediction.astype(np.float32, copy=False), allow_pickle=False)
                if group == "E0":
                    fused, audit = predict_b3_from_global(model, rgb, global_prediction, runner, args.device)
                    fused_path = work / f"epoch20_B3_{image_index:03d}.npy"
                    np.save(fused_path, fused, allow_pickle=False)
                    fusion_audits[str(image_index)] = audit
                    ref = baseline_by_index[image_index]
                    valid_mae = runner.absolute_stats(fused, gt, valid)
                    dense = runner.dense14_stats(fused, gt, valid)
                    local_rows, _ = runner.fixed_pair_stats(fused, gt, valid, image_index, pair_images)
                    high_rows = [x for x in local_rows if x["stride_px"] == 14 and x["group"] == "gt_ge20cm"]
                    pair_count = sum(x["pair_count"] for x in high_rows)
                    high_mae = sum(x["absolute_error_sum_m"] for x in high_rows) / pair_count if pair_count else None
                    for actual, reference, label in ((valid_mae["rmse_m"], float(ref["rmse_m"]), "RMSE"),
                                                     (dense["full14_local_mae_m"], float(ref["local_depth_delta_mae_m"]), "full14"),
                                                     (high_mae, float(ref["large14_mae_m"]), "high14")):
                        if actual is None or abs(actual - reference) > 1e-5:
                            raise ValueError(f"epoch20-B3 image {image_index} failed {label} reproduction: {actual} vs {reference}")
                input_records[str(image_index)] = {
                    "image_relpath": row["image_relpath"], "depth_relpath": row["depth_relpath"],
                    "crop": row["crop"], "view": row["view"],
                    "rgb_sha256": sha256(rgb_path), "gt_sha256": sha256(gt_path),
                    "rgb_dtype": str(rgb.dtype), "rgb_range": [int(rgb.min()), int(rgb.max())],
                    "shape_hw": list(map(int, gt.shape)), "valid_gt_pixels": int(valid.sum()),
                }
                del bgr, rgb, gt, global_prediction
                if (image_index == indices[0] or image_index == indices[-1]):
                    torch.cuda.synchronize()
            del model
            torch.cuda.empty_cache()

        # The smoke gate is the first fixed image across every condition. Check
        # cached B3 arrays against their stored per-image sufficient statistics.
        first_index = indices[0]
        first = validation_rows[first_index]
        bgr = cv2.imread(str(args.rgb_root / first["image_relpath"]), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gt = np.asarray(tifffile.imread(args.depth_root / first["depth_relpath"]), dtype=np.float32)
        valid = runner.m.valid_gt(gt)
        for group, condition in (("C0", "C0-B3"), ("E3", "E3-B3")):
            cache = args.existing_predictions / group / f"prediction_{first_index:03d}.npz"
            pred = load_prediction(cache, gt.shape)
            row = existing_rows.get((condition, first_index))
            if row is None:
                raise ValueError(f"Missing per-image sufficient statistics for {condition} image {first_index}")
            actual = runner.absolute_stats(pred, gt, valid)
            if abs(actual["rmse_m"] - float(row["rmse_m"])) > 1e-5:
                raise ValueError(f"Cached {condition} map does not reproduce its stored per-image RMSE")
        # Load all conditions and render one fixed image only after the smoke gate.
        smoke_predictions = load_all_predictions(first_index, work, args.existing_predictions, gt.shape)
        for condition, pred in smoke_predictions.items():
            if pred.shape != gt.shape or not np.isfinite(pred).all():
                raise ValueError(f"Smoke image has invalid prediction in {condition}")
        del smoke_predictions, bgr, rgb, gt
    finally:
        pass

    import shutil
    per_image_rows = []
    for image_index in indices:
        row = validation_rows[image_index]
        rgb_path = args.rgb_root / row["image_relpath"]
        gt_path = args.depth_root / row["depth_relpath"]
        bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gt = np.asarray(tifffile.imread(gt_path), dtype=np.float32)
        valid = runner.m.valid_gt(gt)
        predictions = load_all_predictions(image_index, work, args.existing_predictions, gt.shape)
        error_over_range = render_error_figure(
            gt, predictions, valid, image_index, row["crop"], row["view"],
            args.output_dir / f"absolute_error_{image_index:03d}.png",
        )
        render_depth_figure(
            rgb, gt, predictions, valid, image_index, row["crop"], row["view"],
            args.output_dir / f"depth_maps_{image_index:03d}.png",
        )
        for condition in DEPTH_CONDITIONS:
            absolute = runner.absolute_stats(predictions[condition], gt, valid)
            dense = runner.dense14_stats(predictions[condition], gt, valid)
            per_image_rows.append({
                "image_index": image_index, "image_relpath": row["image_relpath"],
                "crop": row["crop"], "view": row["view"], "condition": condition,
                "valid_pixel_count": absolute["valid_pixel_count"], "mae_m": absolute["mae_m"],
                "rmse_m": absolute["rmse_m"], "bias_m": absolute["bias_m"],
                "full14_pair_count": dense["full14_pair_count"],
                "full14_local_mae_m": dense["full14_local_mae_m"],
                "error_pixels_clipped_above_10m": error_over_range[condition],
                "prediction_nonpositive_pixels": int(np.count_nonzero(predictions[condition] <= 0)),
                "prediction_above_80m_pixels": int(np.count_nonzero(predictions[condition] > 80.001)),
            })
        del bgr, rgb, gt, predictions
    with (args.output_dir / "fixed8_per_image_metrics.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_image_rows[0]))
        writer.writeheader()
        writer.writerows(per_image_rows)
    shutil.rmtree(work)

    provenance = {
        "experiment": EXPERIMENT,
        "artifact": "private_fixed8_six_condition_visual_diagnostics",
        "utc_finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command_line": command,
        "source_revision": os.environ.get("BEST_COMBO_SOURCE_REVISION"),
        "source_sha256": {
            str(path.relative_to(repo_root)): sha256(path)
            for path in (Path(__file__).resolve(), runner_path, repo_root / "src/uav_metric_research/multiscale.py")
        },
        "checkpoint_sha256": {
            "epoch20": runner.BASE_MODEL_SHA256,
            "C0": runner.CHECKPOINT_SHA256["C0"],
            "E3": runner.CHECKPOINT_SHA256["E3"],
        },
        "validation_sha256": sha256(args.validation_manifest),
        "image_inventory_sha256": sha256(args.image_inventory),
        "visual_indices_sha256": sha256(args.visual_indices),
        "visual_indices": indices,
        "rgb_input_contract": "uint8 RGB HxWx3, range 0..255; frozen preprocessing normalizes once",
        "gt_mask": "finite and 0 < Z <= 65m",
        "condition_definitions": {
            "epoch20-B0": "epoch20 model, short-side 518px global inference",
            "epoch20-B3": "epoch20 model, B3 global-plus-tile residual fusion",
            "C0-P0": "C0 epoch5, short-side 518px global inference",
            "C0-B3": "C0 epoch5, existing B3 full-validation prediction cache",
            "E3-B0": "E3 epoch5, short-side 518px global inference",
            "E3-B3": "E3 epoch5, existing B3 full-validation prediction cache",
        },
        "rendering": {
            "depth_range_m": [0, DEPTH_MAX_M],
            "absolute_error_range_m": [0, ERROR_MAX_M],
            "absolute_error_over_range_counts_recorded": True,
            "invalid_gt_masked_for_all_panels": True,
            "rgb_gt_and_depth_maps_private": True,
        },
        "load_audits": load_audits,
        "epoch20_b3_fixed_image_reproduction_passed": True,
        "fusion_audits": fusion_audits,
        "inputs": input_records,
        "existing_prediction_cache_sha256": {
            f"{group}/{image_index:03d}": sha256(args.existing_predictions / group / f"prediction_{image_index:03d}.npz")
            for group in ("C0", "E3") for image_index in indices
        },
        "gpu_before": gpu_before,
        "gpu_after": subprocess_check_gpu(),
        "training": False,
        "independent_test319": False,
        "figures": [],
    }
    for path in sorted(args.output_dir.glob("*.png")):
        provenance["figures"].append({"path": path.name, "sha256": sha256(path), "bytes": path.stat().st_size})
    provenance["metrics_sha256"] = sha256(args.output_dir / "fixed8_per_image_metrics.csv")
    with (args.output_dir / "provenance.json").open("x", encoding="utf-8") as stream:
        json.dump(provenance, stream, ensure_ascii=False, indent=2, allow_nan=False)
    sum_lines = [f"{sha256(path)}  {path.name}" for path in sorted(args.output_dir.iterdir()) if path.is_file()]
    (args.output_dir / "SHA256SUMS").write_text("\n".join(sum_lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "output_dir": str(args.output_dir),
                      "images": len(indices), "figures": len(provenance["figures"]),
                      "gpu_after": provenance["gpu_after"]}, ensure_ascii=False), flush=True)


def load_all_predictions(image_index: int, work: Path, existing: Path, shape: tuple[int, int]):
    predictions = {
        "epoch20-B0": load_prediction_npy(work / f"epoch20_B0_{image_index:03d}.npy", shape),
        "epoch20-B3": load_prediction_npy(work / f"epoch20_B3_{image_index:03d}.npy", shape),
        "C0-P0": load_prediction_npy(work / f"C0_P0_{image_index:03d}.npy", shape),
        "C0-B3": load_prediction(existing / "C0" / f"prediction_{image_index:03d}.npz", shape),
        "E3-B0": load_prediction_npy(work / f"E3_B0_{image_index:03d}.npy", shape),
        "E3-B3": load_prediction(existing / "E3" / f"prediction_{image_index:03d}.npz", shape),
    }
    return predictions


def subprocess_check_gpu():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.free", "--format=csv,noheader"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    fields = [field.strip() for field in result.split(",")]
    return {
        "name": fields[0],
        "utilization_percent": int(fields[1].split()[0]),
        "used_mib": int(fields[2].split()[0]),
        "free_mib": int(fields[3].split()[0]),
    }


if __name__ == "__main__":
    main()
