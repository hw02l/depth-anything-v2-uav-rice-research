from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import tifffile

if __package__:
    from .data import read_manifest
    from .runtime_state import preparation_context
else:
    from data import read_manifest
    from runtime_state import preparation_context

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate the training-only metric depth distribution.")
    parser.add_argument("--manifest", type=Path, required=True, help="Training CSV only")
    parser.add_argument("--depth-root", type=Path, required=True)
    parser.add_argument("--max-pixels-per-frame", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "work" / "uav_metric_finetuning" / "gt_distribution.json")
    args = parser.parse_args()
    if args.max_pixels_per_frame < 1:
        parser.error("max-pixels-per-frame must be positive")
    if args.output.exists():
        parser.error(f"Refusing to overwrite existing output: {args.output}")
    rows = read_manifest(args.manifest)
    if any(row["split"] != "train" for row in rows):
        parser.error("GT statistics must be computed from train rows only; validation/test leakage is forbidden.")
    protected = ("2025/Day001_", "2025/Day007_")
    if any(row["scene"].startswith(protected) for row in rows):
        parser.error("Training manifest contains a protected prior-evaluation acquisition day.")

    context_args = dict(manifests={"train": args.manifest}, rgb_root=None,
                        depth_root=args.depth_root, physical_splits=["train"], kind="gt",
                        parameters={"seed": args.seed, "max_pixels_per_frame": args.max_pixels_per_frame})
    before_context = preparation_context(**context_args)
    rng = np.random.default_rng(args.seed)
    samples: list[np.ndarray] = []
    per_scene: dict[str, dict[str, float | int]] = {}
    total_valid = 0
    for row in rows:
        depth_path = args.depth_root / Path(row["depth_relpath"])
        if not depth_path.is_file():
            parser.error(f"Missing training depth file: {depth_path}")
        depth = np.asarray(tifffile.imread(depth_path), dtype=np.float32)
        valid = np.isfinite(depth) & (depth > 0)
        values = depth[valid]
        if values.size == 0:
            parser.error(f"Training frame has no finite positive depth: {depth_path}")
        total_valid += int(values.size)
        take = min(values.size, args.max_pixels_per_frame)
        selected = values if take == values.size else values[rng.choice(values.size, size=take, replace=False)]
        samples.append(selected.astype(np.float32, copy=False))
        bucket = per_scene.setdefault(row["scene"], {"frames": 0, "valid_pixels": 0})
        bucket["frames"] += 1
        bucket["valid_pixels"] += int(values.size)

    sample_values = np.concatenate(samples)
    quantiles = {f"p{q:g}": float(np.percentile(sample_values, q)) for q in (50, 90, 95, 99, 99.5, 99.9, 99.95, 100)}
    suggested = math.ceil((quantiles["p99.9"] * 1.10) / 5.0) * 5.0
    result = {
        "status": "pass",
        "source_manifest": str(args.manifest.resolve()),
        "source_manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "split": "train",
        "seed": args.seed,
        "sampling": "uniform random sample without replacement per TIFF",
        "frames": len(rows),
        "scenes": per_scene,
        "valid_pixels_scanned": total_valid,
        "sampled_valid_pixels": int(sample_values.size),
        "depth_unit": "metres",
        "valid_rule": "finite and > 0; range clipping is not applied before quantiles",
        "quantiles_m": quantiles,
        "suggested_max_depth_m": suggested,
        "suggestion_rule": "ceil(p99.9 * 1.10 / 5) * 5; review alongside histograms before training",
        "note": "This is a train-only estimate. Review the range and units before setting the model output cap.",
    }
    result["reuse_context"] = preparation_context(**context_args)
    if result["reuse_context"] != before_context:
        raise ValueError("GT解析中にデータまたは仕様が変わりました。再解析してください。")
    if args.output.exists():
        parser.error(f"Refusing to overwrite existing output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
