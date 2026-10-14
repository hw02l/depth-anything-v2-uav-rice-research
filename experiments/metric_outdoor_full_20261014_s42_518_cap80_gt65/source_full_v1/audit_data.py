from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

import cv2
import numpy as np
import tifffile

if __package__:
    from .data import read_manifest
    from .runtime_state import preparation_context
else:
    from data import read_manifest
    from runtime_state import preparation_context

REPO_ROOT = Path(__file__).resolve().parents[1]
PROTECTED_DAYS = ("2025/Day001", "2025/Day007")


def _safe_path(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"Unsafe relative data path: {relative}")
    resolved_root = root.resolve()
    resolved = (resolved_root / candidate).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ValueError(f"Data path escapes configured root: {relative}")
    return resolved


def _write_report(path: Path | None, report: dict[str, Any]) -> None:
    rendered = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    if path is not None:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing audit report: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
    print(rendered)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit manifests, physical pairs, TIFF pixels, and split isolation.")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--independent-test", type=Path, default=None,
                        help="Optional before final testing; include it to check all manifest splits for leakage.")
    parser.add_argument("--physical-splits", nargs="+", choices=("train", "validation", "independent_test"), default=None,
                        help="Which splits have local file presence/pixel checks. All provided manifests are still checked for leakage.")
    parser.add_argument("--rgb-root", type=Path, required=True)
    parser.add_argument("--depth-root", type=Path, required=True)
    parser.add_argument("--plan-summary", type=Path, default=None,
                        help="Optional metadata-only plan summary to report source-side missing candidates.")
    parser.add_argument("--existing-eval", type=Path, default=REPO_ROOT / "results" / "uav3dcrop" / "all_selected_images.csv")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    manifests: dict[str, list[dict[str, str]]] = {
        "train": read_manifest(args.train),
        "validation": read_manifest(args.validation),
    }
    if args.independent_test is not None:
        manifests["independent_test"] = read_manifest(args.independent_test)
    physical_splits = list(args.physical_splits or manifests.keys())
    unknown_physical = set(physical_splits) - set(manifests)
    if unknown_physical:
        raise ValueError(f"Physical audit requested for missing manifest(s): {sorted(unknown_physical)}")

    scene_sets = {split: {row["scene"] for row in rows} for split, rows in manifests.items()}
    day_sets = {split: {row["day"] for row in rows} for split, rows in manifests.items()}
    for split, rows in manifests.items():
        if any(row["split"] != split for row in rows):
            raise ValueError(f"Manifest split label mismatch for {split}")
        protected = [row["scene"] for row in rows if row["scene"].startswith(PROTECTED_DAYS)]
        if protected:
            raise ValueError(f"{split} contains quarantined historical evaluation scenes: {sorted(set(protected))}")
    split_names = list(manifests)
    leakage = []
    for index, first in enumerate(split_names):
        for second in split_names[index + 1:]:
            shared_scenes = sorted(scene_sets[first] & scene_sets[second])
            shared_days = sorted(day_sets[first] & day_sets[second])
            if shared_scenes or shared_days:
                leakage.append({"first": first, "second": second, "shared_scenes": shared_scenes, "shared_days": shared_days})
    if leakage:
        raise ValueError(f"Scene/day leakage detected: {leakage}")

    prior_eval_scenes: set[str] = set()
    if args.existing_eval.is_file():
        with args.existing_eval.open("r", encoding="utf-8-sig", newline="") as stream:
            prior_eval_scenes = {row["scene"].replace("\\", "/").strip("/") for row in csv.DictReader(stream) if row.get("scene")}
    planned_scenes = set.union(*scene_sets.values()) if scene_sets else set()
    reused_prior_scenes = sorted(prior_eval_scenes & planned_scenes)
    if reused_prior_scenes:
        raise ValueError(f"Existing 52-pair evaluation scene reused: {reused_prior_scenes}")

    report: dict[str, Any] = {
        "status": "pending",
        "manifest_pairs": {split: len(rows) for split, rows in manifests.items()},
        "manifest_sha256": {
            "train": _sha256(args.train),
            "validation": _sha256(args.validation),
            **({"independent_test": _sha256(args.independent_test)} if args.independent_test is not None else {}),
        },
        "physical_splits_audited": physical_splits,
        "distributions": {},
        "split_leakage": "none",
        "prior_evaluation_scenes_checked": len(prior_eval_scenes),
        "prior_evaluation_overlap": "none",
        "metadata_plan_missing_candidates": [],
    }
    if args.plan_summary and args.plan_summary.is_file():
        summary = json.loads(args.plan_summary.read_text(encoding="utf-8"))
        report["metadata_plan_missing_candidates"] = [
            {
                "scene": item["scene"], "split": item["split"],
                "missing_rgb": item.get("missing_rgb_candidates", 0),
                "missing_depth": item.get("missing_depth_candidates", 0),
                "missing_pairs": item.get("missing_pair_candidates", 0),
                "selected_candidates_by_view": item.get("selected_candidates_by_view", {}),
                "verified_pairs_by_view": item.get("selected_pairs_by_view", {}),
            }
            for item in summary.get("scene_metadata", []) if item.get("missing_pair_candidates", 0)
        ]
        report["source_plan_revisions"] = {"rgb": summary.get("rgb_revision"), "depth": summary.get("depth_revision")}

    seen_rgb: dict[Path, str] = {}
    seen_depth: dict[Path, str] = {}
    missing_by_split: dict[str, dict[str, Any]] = {}
    resolved_rows: dict[str, list[tuple[dict[str, str], Path, Path]]] = {}
    for split, rows in manifests.items():
        view_counts: Counter[str] = Counter()
        crop_counts: Counter[str] = Counter()
        day_counts: Counter[str] = Counter()
        scene_counts: Counter[str] = Counter()
        resolved_rows[split] = []
        missing_rgb: list[str] = []
        missing_depth: list[str] = []
        for row in rows:
            image_rel = PurePosixPath(row["image_relpath"])
            depth_rel = PurePosixPath(row["depth_relpath"])
            scene_rel = PurePosixPath(row["scene"])
            expected_depth_parent = scene_rel.name + "_depth"
            if image_rel.parts[:2] != scene_rel.parts or image_rel.parent.name != "images":
                raise ValueError(f"RGB path does not belong to manifest scene: {row['scene']} / {row['image_relpath']}")
            if depth_rel.parts[:2] != (scene_rel.parts[0], expected_depth_parent):
                raise ValueError(f"Depth path does not belong to manifest scene: {row['scene']} / {row['depth_relpath']}")
            if depth_rel.stem != image_rel.stem + "_depth":
                raise ValueError(f"RGB/depth stem mismatch: {row['image_relpath']} / {row['depth_relpath']}")
            rgb_path = _safe_path(args.rgb_root, row["image_relpath"])
            depth_path = _safe_path(args.depth_root, row["depth_relpath"])
            if rgb_path in seen_rgb:
                raise ValueError(f"Duplicate RGB reference across manifest rows: {rgb_path} ({seen_rgb[rgb_path]}, {split})")
            if depth_path in seen_depth:
                raise ValueError(f"Duplicate depth reference across manifest rows: {depth_path} ({seen_depth[depth_path]}, {split})")
            seen_rgb[rgb_path] = split
            seen_depth[depth_path] = split
            if split in physical_splits:
                resolved_rows[split].append((row, rgb_path, depth_path))
                if not rgb_path.is_file():
                    missing_rgb.append(str(rgb_path))
                if not depth_path.is_file():
                    missing_depth.append(str(depth_path))
            view_counts[row["view"]] += 1
            crop_counts[row["crop"]] += 1
            day_counts[row["day"]] += 1
            scene_counts[row["scene"]] += 1
        missing_by_split[split] = {
            "expected_pairs": len(rows),
            "physical_audit": "enabled" if split in physical_splits else "not_requested",
            "missing_rgb_files": len(missing_rgb) if split in physical_splits else None,
            "missing_depth_files": len(missing_depth) if split in physical_splits else None,
            "incomplete_pairs": sum(
                not ((args.rgb_root / Path(row["image_relpath"])).is_file()
                     and (args.depth_root / Path(row["depth_relpath"])).is_file()) for row in rows
            ) if split in physical_splits else None,
            "first_missing_rgb_paths": missing_rgb[:10],
            "first_missing_depth_paths": missing_depth[:10],
        }
        report["distributions"][split] = {
            "crop_counts": dict(crop_counts), "view_counts": dict(view_counts),
            "day_counts": dict(day_counts), "scene_counts": dict(scene_counts),
        }
    report["local_file_presence"] = missing_by_split

    total_missing = sum(
        int(values["missing_rgb_files"] or 0) + int(values["missing_depth_files"] or 0)
        for split, values in missing_by_split.items() if split in physical_splits
    )
    if total_missing:
        report["status"] = "skipped_missing_files"
        report["pixel_audit"] = "skipped: required RGB/depth files are missing from a physically audited split; download only after reviewing the plan"
        _write_report(args.output, report)
        return

    context_args = dict(
        manifests={"train": args.train, "validation": args.validation,
                   **({"independent_test": args.independent_test} if args.independent_test else {})},
        rgb_root=args.rgb_root, depth_root=args.depth_root, physical_splits=physical_splits,
        kind="audit", extra_files={"plan_summary": args.plan_summary, "existing_eval": args.existing_eval},
    )
    before_context = preparation_context(**context_args)

    pair_count = 0
    rgb_shapes: Counter[str] = Counter()
    depth_shapes: Counter[str] = Counter()
    depth_dtypes: Counter[str] = Counter()
    invalid_totals = Counter()
    valid_depth_min = float("inf")
    valid_depth_max = float("-inf")
    valid_depth_pixels = 0
    all_depth_pixels = 0
    for split, rows in resolved_rows.items():
        split_pixel_report = {"pairs_checked": 0, "rgb_shapes": {}, "depth_shapes": {}, "depth_dtypes": {},
                              "depth_pixels": 0, "valid_positive_finite_pixels": 0, "invalid_nonfinite_pixels": 0,
                              "zero_pixels": 0, "negative_pixels": 0, "non_single_channel_tiffs": 0}
        for row, rgb_path, depth_path in rows:
            bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"RGB decode failed: {rgb_path}")
            depth = np.asarray(tifffile.imread(depth_path))
            if depth.ndim == 3 and depth.shape[-1] == 1:
                depth = depth[..., 0]
            if depth.ndim != 2:
                split_pixel_report["non_single_channel_tiffs"] += 1
                raise ValueError(f"Depth TIFF must be single-channel 2D (or HxWx1): {depth_path} {depth.shape}")
            if bgr.shape[:2] != depth.shape:
                raise ValueError(f"RGB/depth pixel dimensions differ: {rgb_path} {bgr.shape[:2]} vs {depth_path} {depth.shape}")
            if not np.issubdtype(depth.dtype, np.floating):
                raise ValueError(f"Depth TIFF must be floating point: {depth_path} ({depth.dtype})")
            finite = np.isfinite(depth)
            positive = finite & (depth > 0)
            zeros = finite & (depth == 0)
            negative = finite & (depth < 0)
            if not np.any(positive):
                raise ValueError(f"Depth TIFF has no finite positive pixels: {depth_path}")
            pair_count += 1
            split_pixel_report["pairs_checked"] += 1
            rgb_shapes[str(tuple(bgr.shape))] = rgb_shapes.get(str(tuple(bgr.shape)), 0) + 1
            depth_shapes[str(tuple(depth.shape))] = depth_shapes.get(str(tuple(depth.shape)), 0) + 1
            depth_dtypes[str(depth.dtype)] += 1
            split_pixel_report["depth_pixels"] += int(depth.size)
            split_pixel_report["valid_positive_finite_pixels"] += int(positive.sum())
            split_pixel_report["invalid_nonfinite_pixels"] += int((~finite).sum())
            split_pixel_report["zero_pixels"] += int(zeros.sum())
            split_pixel_report["negative_pixels"] += int(negative.sum())
            split_pixel_report["rgb_shapes"][str(tuple(bgr.shape))] = split_pixel_report["rgb_shapes"].get(str(tuple(bgr.shape)), 0) + 1
            split_pixel_report["depth_shapes"][str(tuple(depth.shape))] = split_pixel_report["depth_shapes"].get(str(tuple(depth.shape)), 0) + 1
            split_pixel_report["depth_dtypes"][str(depth.dtype)] = split_pixel_report["depth_dtypes"].get(str(depth.dtype), 0) + 1
            valid_values = depth[positive]
            valid_depth_min = min(valid_depth_min, float(valid_values.min()))
            valid_depth_max = max(valid_depth_max, float(valid_values.max()))
            valid_depth_pixels += int(positive.sum())
            all_depth_pixels += int(depth.size)
        report.setdefault("pixel_audit_by_split", {})[split] = split_pixel_report
    report["pixel_audit"] = {
        "status": "pass", "pairs_checked": pair_count,
        "rgb_shapes": dict(rgb_shapes), "depth_shapes": dict(depth_shapes), "depth_dtypes": dict(depth_dtypes),
        "depth_pixels": all_depth_pixels, "valid_positive_finite_pixels": valid_depth_pixels,
        "valid_depth_min_m": valid_depth_min, "valid_depth_max_m": valid_depth_max,
        "invalid_nonfinite_pixels": sum(item["invalid_nonfinite_pixels"] for item in report["pixel_audit_by_split"].values()),
        "zero_pixels": sum(item["zero_pixels"] for item in report["pixel_audit_by_split"].values()),
        "negative_pixels": sum(item["negative_pixels"] for item in report["pixel_audit_by_split"].values()),
    }
    report["status"] = "pass"
    report["reuse_context"] = preparation_context(**context_args)
    if report["reuse_context"] != before_context:
        raise ValueError("監査中にデータまたは仕様が変わりました。再監査してください。")
    report["physical_audit_coverage"] = "all_manifests" if set(physical_splits) == set(manifests) else "selected_splits_only"
    _write_report(args.output, report)


if __name__ == "__main__":
    main()
