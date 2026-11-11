from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from data_patch import draw_crop, read_manifest


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize deterministic train crop coordinates for both fixed budget candidates.")
    parser.add_argument("--train-manifest", type=Path, default=ROOT / "configs/uav3dcrop_verified/train.csv")
    parser.add_argument("--audit-json", type=Path, default=HERE / "audit/train_validation_pair_dimensions.json")
    parser.add_argument("--output-dir", type=Path, default=HERE / "manifests")
    args = parser.parse_args()
    rows = read_manifest(args.train_manifest)
    if len(rows) != 384 or any(row["split"] != "train" for row in rows):
        raise ValueError("Expected canonical Train384 only")
    audit = json.loads(args.audit_json.read_text(encoding="utf-8"))
    if audit.get("status") != "pass" or audit.get("train_pairs") != 384 or audit.get("rgb_gt_dimension_mismatches") != 0:
        raise ValueError("A passing Train384 RGB/GT dimension audit is required")
    counts = audit.get("shape_counts_by_split", {})
    train_shapes = [(key, count) for key, count in counts.items() if key.startswith("train:")]
    if len(train_shapes) != 1 or train_shapes[0][1] != 384:
        raise ValueError("Training dimensions are not uniform; inspect the audit before manifest creation")
    shape = tuple(map(int, train_shapes[0][0].split(":", 1)[1].split("x")))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summary = {
        "experiment_id": "native_resolution_patch_training_20261111",
        "seed": 42,
        "train_manifest_sha256": sha256(args.train_manifest),
        "audit_json_sha256": sha256(args.audit_json),
        "shape_yx": list(shape),
        "budgets": {},
        "independent_test_used": False,
    }
    for repeats, updates in ((1, 480), (2, 960)):
        output = args.output_dir / f"crop_sampling_candidate_{updates}.csv"
        with output.open("x", encoding="utf-8", newline="") as stream:
            fields = [
                "optimizer_updates_candidate", "group", "epoch", "draw", "train_row_index",
                "image_relpath", "depth_relpath", "scene", "crop", "day", "view",
                "crop_box_yxyx", "crop_space", "footprint_original_yxyx_float",
                "native_px_per_input_px_yx", "normalized_start_uv", "flipped_horizontal",
            ]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            count = 0
            for epoch in range(1, 6):
                for row_index, row in enumerate(rows):
                    for draw in range(repeats):
                        for group in ("C0", "F1", "F2"):
                            metadata = draw_crop(row, shape, group, seed=42, epoch=epoch, draw=draw)
                            writer.writerow({
                                "optimizer_updates_candidate": updates,
                                "group": group,
                                "epoch": epoch,
                                "draw": draw,
                                "train_row_index": row_index,
                                "image_relpath": row["image_relpath"],
                                "depth_relpath": row["depth_relpath"],
                                "scene": row["scene"],
                                "crop": row["crop"],
                                "day": row["day"],
                                "view": row["view"],
                                "crop_box_yxyx": json.dumps(metadata["crop_box_yxyx"]),
                                "crop_space": metadata["crop_space"],
                                "footprint_original_yxyx_float": json.dumps(metadata["footprint_original_yxyx_float"]),
                                "native_px_per_input_px_yx": json.dumps(metadata["native_px_per_input_px_yx"]),
                                "normalized_start_uv": json.dumps(metadata["normalized_start_uv"]),
                                "flipped_horizontal": metadata["flipped_horizontal"],
                            })
                            count += 1
        expected = 384 * 5 * repeats * 3
        if count != expected:
            raise AssertionError(f"Unexpected crop rows for {updates}: {count} != {expected}")
        summary["budgets"][str(updates)] = {
            "crops_per_group": count // 3,
            "crops_per_image": 5 * repeats,
            "csv": output.name,
            "csv_sha256": sha256(output),
        }
    (args.output_dir / "crop_sampling_candidates.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
