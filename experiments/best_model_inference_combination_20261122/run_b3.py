from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np
import tifffile
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForDepthEstimation

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))
from uav_metric_research import multiscale as m  # noqa: E402

BASE_MODEL_SHA256 = "52769e2e95704d0862e708e8772195b861c647dd4812058eaf698a44305c3bcb"
BASE_CONFIG_SHA256 = "3c6ab05881e6a76edbfd33ff393fdfb082582084e9262826bb1ac0fd2b1dac7c"
CHECKPOINT_SHA256 = {
    "C0": "2fd190727c7a5124c56c1e0ea2dd8cb09c6cb62a0f92b34fce97ad394281b35a",
    "E3": "223b75d1f5d81844bc5702565a8291270a8ae951d6e8e7c12be9f1b3ad12e886",
}
PAIR_SHA256 = "bcb56eab03bec4a44f0626da0f4814876fae67774e536e00cd9f1ad16b1f5728"
INPUT_INVENTORY_SHA256 = "59271cd7f39fbd0cf8e5228c222d3825ae2bd11eedec641232072ddce0afa6e5"
VALIDATION_SHA256 = "96c4eab0eb0075ac141ff9b17c22a45734f242f7cc895f8eaf064a4f14cc871e"
# The frozen helpers live in this exact experiment snapshot. Configure the
# imported helper module explicitly because its historical default path uses
# an obsolete experiment directory name absent from the source snapshot.
FROZEN_RUN = REPO / "experiments/metric_outdoor_full_20261014_s42_518_cap80_gt65"
m.RUN = FROZEN_RUN
TILE_SIZE = 2000
OVERLAP = 0.25
INPUT_SIDE = 518
LOWPASS_KERNEL = 215
SCALES = (14, 28, 56, 107, 214)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def audit_checkpoint_identity(payload: dict, expected_group: str) -> dict:
    """Verify the two checkpoint schemas used by the frozen C0 and E3 runs.

    C0 stores its signed run metadata in ``signature_payload`` with a
    ``checkpoint_signature`` field. E3 stores the equivalent metadata in
    ``config`` with a ``signature`` field. The model weights are audited
    separately and loaded with ``strict=True`` below.
    """
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint payload must be a mapping")
    epoch = int(payload.get("epoch", -1))
    if epoch != 5:
        raise ValueError(f"Expected fixed epoch-5 checkpoint, got epoch={epoch}")

    signature_payload = None
    signature = None
    signature_source = None
    for payload_key, signature_key in (
        ("signature_payload", "checkpoint_signature"),
        ("config", "signature"),
    ):
        candidate = payload.get(payload_key)
        stored = payload.get(signature_key)
        if isinstance(candidate, dict) and isinstance(stored, str):
            signature_payload = candidate
            signature = stored
            signature_source = f"{payload_key}/{signature_key}"
            break
    if signature_payload is None or signature is None:
        raise ValueError("Checkpoint has no supported signed run metadata")

    saved_group = signature_payload.get("group")
    if saved_group != expected_group:
        raise ValueError(
            f"Checkpoint signature group mismatch: expected={expected_group!r}, saved={saved_group!r}"
        )
    calculated_signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    if signature != calculated_signature:
        raise ValueError("Checkpoint run signature is internally inconsistent")
    global_step = int(payload.get("global_step", -1))
    if global_step <= 0:
        raise ValueError(f"Checkpoint has invalid global_step={global_step}")
    return {
        "epoch": epoch,
        "group": saved_group,
        "global_step": global_step,
        "run_signature": signature_payload,
        "run_signature_sha256": signature,
        "run_signature_source": signature_source,
    }


def validate_manifest_rows(rows, inventory_rows):
    """Validate canonical manifest rows against the independently hashed inventory."""
    if len(rows) != 128 or len(inventory_rows) != 128:
        raise ValueError("Validation manifest and inventory must each contain 128 rows")
    required = {"split", "day", "crop", "view", "image_relpath", "depth_relpath"}
    if not rows or required - set(rows[0]):
        raise ValueError("Validation manifest is missing required identity columns")
    inventory_required = {"image_index", "crop", "view", "day", "image_relpath", "depth_relpath"}
    if not inventory_rows or inventory_required - set(inventory_rows[0]):
        raise ValueError("Image inventory is missing required identity columns")
    for index, (row, inv) in enumerate(zip(rows, inventory_rows)):
        if row["split"] != "validation" or row["day"] != "2025/Day047":
            raise ValueError(f"Noncanonical split/day in validation manifest at row {index}")
        if int(inv["image_index"]) != index:
            raise ValueError(f"Image inventory order is not canonical at row {index}")
        for key in ("image_relpath", "depth_relpath", "crop", "view", "day"):
            if row[key] != inv[key]:
                raise ValueError(f"Validation/inventory mismatch for {key} at row {index}")
    return rows


def read_validation_rows(manifest_path: Path, inventory_rows):
    if sha256(manifest_path) != VALIDATION_SHA256:
        raise ValueError("Validation manifest SHA256 mismatch")
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return validate_manifest_rows(rows, inventory_rows)


def write_json(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def write_csv(path: Path, rows) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"Refusing to write empty result table: {path}")
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_model(group: str, base: Path, checkpoint: Path | None, device: str):
    base_weights, config_path = base / "model.safetensors", base / "config.json"
    if sha256(base_weights) != BASE_MODEL_SHA256 or sha256(config_path) != BASE_CONFIG_SHA256:
        raise ValueError("Base model/config SHA256 mismatch")
    config = AutoConfig.from_pretrained(str(base), local_files_only=True)
    if config.max_depth != 80 or config.depth_estimation_type != "metric" or config.backbone_config.patch_size != 14:
        raise ValueError("Expected Depth Anything V2 Metric Outdoor Small, 80m cap, patch14")
    model = AutoModelForDepthEstimation.from_config(config)
    base_audit = m.strict_attention_state(model, load_file(str(base_weights), device="cpu"))
    model.requires_grad_(False)
    checkpoint_audit = {"epoch": 20, "group": "E0", "path": str(base_weights), "sha256": BASE_MODEL_SHA256,
                        "strict_load": True, "loaded_values_exact": True, "source": "frozen base safetensors"}
    if group != "E0":
        expected = CHECKPOINT_SHA256[group]
        if checkpoint is None or sha256(checkpoint) != expected:
            raise ValueError(f"{group} checkpoint SHA256 mismatch")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        identity = audit_checkpoint_identity(payload, group)
        epoch = identity["epoch"]
        saved_group = identity["group"]
        state = payload.get("model")
        if not isinstance(state, dict):
            raise ValueError("Checkpoint has no model state_dict")
        target = model.state_dict()
        if set(state) != set(target):
            missing, extra = sorted(set(target) - set(state)), sorted(set(state) - set(target))
            raise ValueError(f"Checkpoint keys differ: missing={missing[:8]}, extra={extra[:8]}")
        for key, tensor in state.items():
            if tensor.shape != target[key].shape or tensor.dtype != target[key].dtype or not torch.isfinite(tensor).all():
                raise ValueError(f"Checkpoint tensor audit failed: {key}")
        model.load_state_dict(state, strict=True)
        if any(not torch.equal(value, model.state_dict()[key]) for key, value in state.items()):
            raise ValueError("Loaded checkpoint tensor values differ")
        checkpoint_audit = {
            **identity,
            "path": str(checkpoint), "sha256": expected, "state_dict_key_count": len(state),
            "strict_load": True, "loaded_values_exact": True,
        }
        del payload, state
    model.to(device).eval()
    return model, {"base_state_load_audit": base_audit, "checkpoint_state_load_audit": checkpoint_audit}


def infer_one(model, rgb: np.ndarray, device: str) -> np.ndarray:
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise TypeError("Inference RGB must be HxWx3 uint8 in RGB order, range 0..255")
    if int(rgb.min()) == int(rgb.max()):
        raise ValueError("RGB image is constant; possible channel/range corruption")
    tensor = m.preprocess(rgb, INPUT_SIDE)
    if tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
        raise ValueError("Preprocessed input must be finite float32")
    with torch.inference_mode():
        native = model(pixel_values=tensor[None].to(device)).predicted_depth[0].float().cpu().numpy()
    pred = m.restore(native, rgb.shape[:2]).astype(np.float32, copy=False)
    if pred.shape != rgb.shape[:2] or not np.isfinite(pred).all():
        raise FloatingPointError("Model output shape/nonfinite audit failed")
    return pred


def fuse_depth(global_pred: np.ndarray, tile_pred: np.ndarray) -> np.ndarray:
    """Apply the locked B3 low/high-frequency residual formula in original pixels."""
    if global_pred.shape != tile_pred.shape or global_pred.ndim != 2:
        raise ValueError("Global and tile predictions must share one HxW coordinate grid")
    global_low = cv2.boxFilter(global_pred, ddepth=-1, ksize=(LOWPASS_KERNEL, LOWPASS_KERNEL),
                               normalize=True, borderType=cv2.BORDER_REFLECT_101)
    tile_low = cv2.boxFilter(tile_pred, ddepth=-1, ksize=(LOWPASS_KERNEL, LOWPASS_KERNEL),
                             normalize=True, borderType=cv2.BORDER_REFLECT_101)
    return (global_low + tile_pred - tile_low).astype(np.float32, copy=False)


def predict_b3(model, rgb: np.ndarray, device: str):
    """Frozen B3: same-model global + original-coordinate 2000px tile residual."""
    global_pred = infer_one(model, rgb, device)
    merger = m.TileMerger(rgb.shape[:2])
    boxes = m.tiles(rgb.shape[:2], TILE_SIZE, OVERLAP)
    if not boxes:
        raise ValueError("Tiling produced no tiles")
    for box in boxes:
        y, x, bottom, right = box
        tile_pred = infer_one(model, rgb[y:bottom, x:right], device)
        merger.add(box, tile_pred)
        del tile_pred
    tile_pred = merger.result()
    pred = fuse_depth(global_pred, tile_pred)
    audit = {
        "tile_count": len(boxes), "tile_boxes_yxbr": [list(map(int, b)) for b in boxes],
        "global_min_m": float(global_pred.min()), "global_max_m": float(global_pred.max()),
        "tile_blended_min_m": float(tile_pred.min()), "tile_blended_max_m": float(tile_pred.max()),
        "fusion_min_m": float(np.nanmin(pred)), "fusion_max_m": float(np.nanmax(pred)),
        "fusion_nonfinite_count": int((~np.isfinite(pred)).sum()),
        "fusion_nonpositive_count": int((pred <= 0).sum()),
        "fusion_above_80m_count": int((pred > 80.001).sum()),
        "formula": "box215_reflect101(global) + hann_blended_tile - box215_reflect101(hann_blended_tile)",
    }
    if audit["fusion_nonfinite_count"]:
        raise FloatingPointError("B3 fusion produced nonfinite predictions; raw output retained in failure record")
    return pred, audit


def _pearson(p: np.ndarray, g: np.ndarray) -> float | None:
    if len(p) < 2:
        return None
    pc, gc = p - p.mean(), g - g.mean()
    den = float(np.sqrt(np.dot(pc, pc) * np.dot(gc, gc)))
    return float(np.dot(pc, gc) / den) if den > 0 else None


def absolute_stats(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray):
    p, g = pred[valid].astype(np.float64), gt[valid].astype(np.float64)
    e = p - g
    n = int(len(p))
    if n == 0:
        raise ValueError("Validation image has no valid GT pixels")
    return {
        "valid_pixel_count": n, "abs_error_sum_m": float(np.abs(e).sum(dtype=np.float64)),
        "squared_error_sum_m2": float(np.square(e).sum(dtype=np.float64)),
        "abs_relative_error_sum": float((np.abs(e) / g).sum(dtype=np.float64)),
        "prediction_sum_m": float(p.sum(dtype=np.float64)), "gt_sum_m": float(g.sum(dtype=np.float64)),
        "prediction_squared_sum_m2": float(np.square(p).sum(dtype=np.float64)),
        "gt_squared_sum_m2": float(np.square(g).sum(dtype=np.float64)),
        "prediction_gt_cross_sum_m2": float(np.multiply(p, g).sum(dtype=np.float64)),
        "mae_m": float(np.abs(e).mean()), "rmse_m": float(np.sqrt(np.square(e).mean())),
        "abs_rel": float((np.abs(e) / g).mean()), "bias_m": float(e.mean()), "pearson_r": _pearson(p, g),
    }


def dense14_stats(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray):
    total_error, total_sq, count = 0.0, 0.0, 0
    stride = 14
    for axis in (0, 1):
        if axis == 0:
            g0, g1 = gt[:-stride, :], gt[stride:, :]
            p0, p1 = pred[:-stride, :], pred[stride:, :]
            v = valid[:-stride, :] & valid[stride:, :]
        else:
            g0, g1 = gt[:, :-stride], gt[:, stride:]
            p0, p1 = pred[:, :-stride], pred[:, stride:]
            v = valid[:, :-stride] & valid[:, stride:]
        d = (p0[v].astype(np.float64) - p1[v].astype(np.float64)) - (g0[v].astype(np.float64) - g1[v].astype(np.float64))
        total_error += float(np.abs(d).sum(dtype=np.float64))
        total_sq += float(np.square(d).sum(dtype=np.float64))
        count += int(v.sum())
    return {"full14_pair_count": count, "full14_abs_error_sum_m": total_error,
            "full14_squared_error_sum_m2": total_sq,
            "full14_local_mae_m": total_error / count if count else None,
            "full14_local_rmse_m": math.sqrt(total_sq / count) if count else None}


def fixed_pair_stats(pred, gt, valid, image_index, expected_manifest):
    rows = []
    expected = [x for x in expected_manifest[image_index]["pairs"] if int(x["stride"]) in SCALES]
    actual_manifest = []
    for stride in SCALES:
        for axis, y, x, yy, xx in m.sample_pairs(gt.shape, valid, image_index, stride):
            coordinates = np.stack([y, x, yy, xx], axis=1).astype("<i4")
            actual_manifest.append({"stride": stride, "axis": axis, "pairs": int(len(y)),
                                    "coordinates_sha256": hashlib.sha256(coordinates.tobytes()).hexdigest()})
            gd = gt[y, x].astype(np.float64) - gt[yy, xx].astype(np.float64)
            pd = pred[y, x].astype(np.float64) - pred[yy, xx].astype(np.float64)
            selectors = {"all": np.ones(len(y), dtype=bool), "gt_ge20cm": np.abs(gd) >= 0.2}
            for name, select in selectors.items():
                g, p = gd[select], pd[select]
                if not len(g):
                    continue
                rows.append({"image_index": image_index, "stride_px": stride, "group": name,
                             "axis": axis, "pair_count": int(len(g)),
                             "absolute_error_sum_m": float(np.abs(p - g).sum(dtype=np.float64)),
                             "squared_error_sum_m2": float(np.square(p - g).sum(dtype=np.float64)),
                             "abs_gt_delta_sum_m": float(np.abs(g).sum(dtype=np.float64)),
                             "abs_prediction_delta_sum_m": float(np.abs(p).sum(dtype=np.float64)),
                             "sign_match_count": int((np.sign(g) == np.sign(p)).sum()),
                             "nonzero_gt_pair_count": int((g != 0).sum())})
    if actual_manifest != expected:
        raise ValueError(f"Fixed pair manifest mismatch at image {image_index}")
    return rows, actual_manifest


def pooled_metrics(absolute_rows, local_rows):
    n = sum(int(r["valid_pixel_count"]) for r in absolute_rows)
    pair14 = sum(int(r["full14_pair_count"]) for r in absolute_rows)
    abs_error = sum(float(r["abs_error_sum_m"]) for r in absolute_rows)
    sq_error = sum(float(r["squared_error_sum_m2"]) for r in absolute_rows)
    full_error = sum(float(r["full14_abs_error_sum_m"]) for r in absolute_rows)
    local = [r for r in local_rows if int(r["stride_px"]) == 14 and r["group"] == "gt_ge20cm"]
    ln = sum(int(r["pair_count"]) for r in local)
    le = sum(float(r["absolute_error_sum_m"]) for r in local)
    pred_abs = sum(float(r["abs_prediction_delta_sum_m"]) for r in local)
    gt_abs = sum(float(r["abs_gt_delta_sum_m"]) for r in local)
    signs = sum(int(r["sign_match_count"]) for r in local)
    nonzero = sum(int(r["nonzero_gt_pair_count"]) for r in local)
    return {
        "images": len(absolute_rows), "valid_pixel_count": n, "mae_m": abs_error / n,
        "rmse_m": math.sqrt(sq_error / n), "full14_pair_count": pair14,
        "full14_local_mae_m": full_error / pair14,
        "gt_ge20_14_pair_count": ln, "gt_ge20_14_local_mae_m": le / ln,
        "gt_ge20_14_local_rmse_m": math.sqrt(sum(float(r["squared_error_sum_m2"]) for r in local) / ln),
        "gt_ge20_14_amplitude_ratio": pred_abs / gt_abs,
        "gt_ge20_14_sign_match_rate": signs / nonzero,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model-dir", type=Path, required=True)
    ap.add_argument("--c0-checkpoint", type=Path, required=True)
    ap.add_argument("--e3-checkpoint", type=Path, required=True)
    ap.add_argument("--validation-manifest", type=Path, required=True)
    ap.add_argument("--pair-manifest", type=Path, required=True)
    ap.add_argument("--image-inventory", type=Path, required=True)
    ap.add_argument("--rgb-root", type=Path, required=True)
    ap.add_argument("--depth-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke-only", action="store_true")
    ap.add_argument("--baseline-b3-per-image", type=Path)
    ap.add_argument("--visual-indices", type=Path)
    args = ap.parse_args()
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This experiment requires the audited CUDA device")
    if args.output.exists():
        raise FileExistsError(f"Refusing to reuse/overwrite output directory: {args.output}")
    if sha256(args.pair_manifest) != PAIR_SHA256:
        raise ValueError("Fixed pair manifest SHA256 mismatch")
    if sha256(args.image_inventory) != INPUT_INVENTORY_SHA256:
        raise ValueError("RGB/GT image inventory SHA256 mismatch")
    inventory_rows = list(csv.DictReader(args.image_inventory.open(encoding="utf-8-sig", newline="")))
    rows = read_validation_rows(args.validation_manifest, inventory_rows)
    frozen_files = [FROZEN_RUN / "source_full_v1" / name for name in ("data.py", "metrics.py")]
    if not all(path.is_file() for path in frozen_files):
        raise FileNotFoundError(f"Required frozen evaluation source missing: {frozen_files}")
    pair_obj = json.loads(args.pair_manifest.read_text(encoding="utf-8"))
    pair_images = pair_obj["images"]
    if len(pair_images) != 128:
        raise ValueError("Pair manifest must contain the fixed 128 validation images")
    if args.baseline_b3_per_image and args.baseline_b3_per_image.exists():
        legacy = list(csv.DictReader(args.baseline_b3_per_image.open(encoding="utf-8-sig", newline="")))
        legacy = [r for r in legacy if r.get("epoch") == "20" and r.get("method") == "B3"]
        if len(legacy) != 128:
            raise ValueError("Expected 128 saved epoch20 B3 image rows")
    else:
        legacy = []
    if args.visual_indices:
        visual_indices = [int(r["image_index"]) for r in csv.DictReader(args.visual_indices.open(encoding="utf-8-sig", newline=""))]
        visual_indices = sorted(set(visual_indices))
        if len(visual_indices) != 8:
            raise ValueError("Visual index file must have exactly eight unique images")
    else:
        visual_indices = []
    if args.smoke_only and not legacy:
        raise ValueError("Smoke gate requires saved epoch20 B3 per-image reference")
    if not args.smoke_only and not (args.output.parent / "smoke_gate.json").exists():
        raise FileNotFoundError("Run and pass smoke gate before full Validation128 inference")

    args.output.mkdir(parents=True, exist_ok=False)

    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    host = subprocess.run(["hostname", "-f"], capture_output=True, text=True, check=True).stdout.strip()
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.total,memory.used,memory.free", "--format=csv,noheader"], capture_output=True, text=True, check=True).stdout.strip()
    # A GPU-side source snapshot may intentionally omit .git metadata. In that
    # case require the caller to pass the audited workstation revision and
    # record the limitation; source-file hashes below still identify bytes.
    try:
        git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
        git_status = subprocess.run(["git", "status", "--short"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
        revision_source = "gpu_source_git_metadata"
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_head = os.environ.get("BEST_COMBO_SOURCE_REVISION")
        if not git_head:
            raise RuntimeError("GPU source has no .git metadata; set BEST_COMBO_SOURCE_REVISION to the audited source revision")
        git_status = "unavailable: source snapshot has no .git metadata; source hashes recorded"
        revision_source = "explicit_audited_snapshot_revision"
    source_hashes = {str(p.relative_to(REPO)): sha256(p) for p in
                     (HERE, REPO / "src/uav_metric_research/multiscale.py", *frozen_files)}
    write_json(args.output / "launch.json", {
        "experiment": "best_model_inference_combination_20261122", "utc_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": host, "gpu_status_pre_run": gpu, "source_revision": git_head, "source_revision_source": revision_source, "git_status": git_status,
        "source_sha256": source_hashes, "command_line": sys.argv, "device": args.device,
        "frozen_helper_root": str(FROZEN_RUN),
        "torch": torch.__version__, "cuda": torch.version.cuda, "numpy": np.__version__, "opencv": cv2.__version__,
        "validation_sha256": sha256(args.validation_manifest), "pair_manifest_sha256": sha256(args.pair_manifest),
        "image_inventory_sha256": sha256(args.image_inventory),
        "rgb_input_contract": "uint8 RGB HxWx3, 0..255; existing frozen preprocessing divides once by 255 then ImageNet normalizes",
        "gt_mask": "finite and 0 < Z <= 65m", "tile_size_px": TILE_SIZE, "input_side_px": INPUT_SIDE,
        "nominal_overlap": OVERLAP, "lowpass": {"kernel_px": LOWPASS_KERNEL, "border": "BORDER_REFLECT_101", "normalize": True},
        "fusion": "L(D_global)+D_tile-L(D_tile)", "visual_indices": visual_indices,
        "training": False, "independent_test319": False,
    })
    indices = [0] if args.smoke_only else list(range(128))
    groups = ("E0", "C0", "E3") if args.smoke_only else ("C0", "E3")
    checkpoints = {"C0": args.c0_checkpoint, "E3": args.e3_checkpoint}
    smoke = {"status": "running", "checks": {}, "images": []}
    absolute_rows, local_rows, runtime_rows = [], [], []
    for group in groups:
        model, load_audit = load_model(group, args.base_model_dir, checkpoints.get(group), args.device)
        if args.smoke_only:
            row = rows[0]
            rgb_path, gt_path = args.rgb_root / row["image_relpath"], args.depth_root / row["depth_relpath"]
            bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(rgb_path)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            gt = np.asarray(tifffile.imread(gt_path), dtype=np.float32)
            if sha256(rgb_path) != inventory_rows[0]["rgb_sha256"] or sha256(gt_path) != inventory_rows[0]["gt_sha256"]:
                raise ValueError("Smoke RGB/GT bytes differ from the locked input inventory")
            if rgb.shape[:2] != gt.shape or rgb.dtype != np.uint8 or not np.array_equal(rgb, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)):
                raise ValueError("RGB/GT alignment or RGB round-trip check failed")
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.monotonic()
            pred, pred_audit = predict_b3(model, rgb, args.device)
            torch.cuda.synchronize()
            seconds = time.monotonic() - t0
            if group == "E0":
                ref = legacy[0]
                valid = m.valid_gt(gt)
                actual_abs = absolute_stats(pred, gt, valid)
                dense = dense14_stats(pred, gt, valid)
                local, _ = fixed_pair_stats(pred, gt, valid, 0, pair_images)
                high = [x for x in local if x["stride_px"] == 14 and x["group"] == "gt_ge20cm"]
                count = sum(x["pair_count"] for x in high)
                err = sum(x["absolute_error_sum_m"] for x in high)
                gate_checks = {
                    "rmse_m": {"actual": actual_abs["rmse_m"], "reference": float(ref["rmse_m"]), "tolerance": 1e-5},
                    "full14_local_mae_m": {"actual": dense["full14_local_mae_m"], "reference": float(ref["local_depth_delta_mae_m"]), "tolerance": 1e-5},
                    "gt_ge20_14_local_mae_m": {"actual": err / count, "reference": float(ref["large14_mae_m"]), "tolerance": 1e-5},
                }
                for value in gate_checks.values():
                    value["passed"] = abs(value["actual"] - value["reference"]) <= value["tolerance"]
                smoke["checks"]["epoch20_B3_saved_row_reproduction"] = {"passed": all(x["passed"] for x in gate_checks.values()), "checks": gate_checks,
                    "note": "single fixed Validation128 image; complete 128-image reference is reused from committed B3 row-level metrics"}
            smoke["images"].append({"group": group, "image_index": 0, "image_relpath": row["image_relpath"],
                "rgb_dtype": str(rgb.dtype), "rgb_range": [int(rgb.min()), int(rgb.max())],
                "rgb_sha256": sha256(rgb_path), "gt_sha256": sha256(gt_path), "prediction_audit": pred_audit,
                "seconds": seconds, "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "load_audit": load_audit})
        else:
            group_dir = args.output / group
            group_dir.mkdir(exist_ok=False)
            for i in indices:
                row = rows[i]
                rgb_path, gt_path = args.rgb_root / row["image_relpath"], args.depth_root / row["depth_relpath"]
                bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
                if bgr is None:
                    raise FileNotFoundError(rgb_path)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                gt = np.asarray(tifffile.imread(gt_path), dtype=np.float32)
                if sha256(rgb_path) != inventory_rows[i]["rgb_sha256"] or sha256(gt_path) != inventory_rows[i]["gt_sha256"]:
                    raise ValueError(f"RGB/GT bytes differ from the locked inventory at image {i}")
                if rgb.shape[:2] != gt.shape:
                    raise ValueError(f"RGB/GT shape mismatch at image {i}")
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                t0 = time.monotonic()
                pred, audit = predict_b3(model, rgb, args.device)
                torch.cuda.synchronize()
                elapsed = time.monotonic() - t0
                valid = m.valid_gt(gt)
                absolute = absolute_stats(pred, gt, valid)
                dense = dense14_stats(pred, gt, valid)
                local, fixed_manifest = fixed_pair_stats(pred, gt, valid, i, pair_images)
                absolute_rows.append({"model_id": group + "-B3", "image_index": i, "image_relpath": row["image_relpath"],
                    "crop": row["crop"], "view": row["view"], "rgb_sha256": sha256(rgb_path), "gt_sha256": sha256(gt_path),
                    **absolute, **dense, "fusion_out_of_range_nonpositive": audit["fusion_nonpositive_count"],
                    "fusion_out_of_range_above80": audit["fusion_above_80m_count"]})
                local_rows.extend({"model_id": group + "-B3", "image_relpath": row["image_relpath"], "crop": row["crop"], "view": row["view"], **x} for x in local)
                runtime_rows.append({"model_id": group + "-B3", "image_index": i, "seconds_global_and_tiles_fusion": elapsed,
                    "tile_count": audit["tile_count"], "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()), **audit})
                if i in visual_indices:
                    with (group_dir / f"prediction_{i:03d}.npz").open("xb") as f:
                        np.savez_compressed(f, prediction_m=pred)
                if (i + 1) % 8 == 0 or i == len(indices) - 1:
                    print(json.dumps({"group": group, "images_done": i + 1, "total": len(indices),
                        "seconds": round(elapsed, 2), "rmse_image_m": absolute["rmse_m"],
                        "fusion_out_of_range_nonpositive": audit["fusion_nonpositive_count"],
                        "fusion_out_of_range_above80": audit["fusion_above_80m_count"]}), flush=True)
                del pred, rgb, gt, bgr
                if (i + 1) % 8 == 0:
                    torch.cuda.empty_cache()
        del model
        torch.cuda.empty_cache()
        if args.smoke_only and group == "E0":
            gate = smoke["checks"].get("epoch20_B3_saved_row_reproduction", {})
            if not gate.get("passed"):
                raise RuntimeError("Epoch20 B3 reproduction smoke gate failed")
    if args.smoke_only:
        smoke.update({"status": "passed", "checkpoint_hashes": {"C0": CHECKPOINT_SHA256["C0"], "E3": CHECKPOINT_SHA256["E3"], "E0": BASE_MODEL_SHA256},
                      "pair_manifest_sha256": PAIR_SHA256})
        write_json(args.output / "smoke_gate.json", smoke)
    else:
        write_csv(args.output / "per_image_absolute_sufficient.csv", absolute_rows)
        write_csv(args.output / "per_image_local_sufficient.csv", local_rows)
        write_csv(args.output / "runtime.csv", runtime_rows)
        summaries = []
        for group in ("C0", "E3"):
            ar = [r for r in absolute_rows if r["model_id"] == group + "-B3"]
            lr = [r for r in local_rows if r["model_id"] == group + "-B3"]
            summaries.append({"condition": group + "-B3", **pooled_metrics(ar, lr)})
        write_csv(args.output / "summary_new_conditions.csv", summaries)
        write_json(args.output / "run_complete.json", {"status": "complete", "groups": list(groups), "images_per_group": 128,
            "new_inference_performed": True, "summaries": summaries, "checkpoint_sha256": CHECKPOINT_SHA256,
            "pair_manifest_sha256": PAIR_SHA256, "out_of_range_values_scored_unmodified": True})
    print("RUN_FINISHED", str(args.output), flush=True)


if __name__ == "__main__":
    main()
