from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote

import requests
from huggingface_hub import HfApi

REPO_ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def natural_key(value: str) -> tuple[int, str]:
    match = re.search(r"(\d+)", PurePosixPath(value).stem)
    return (int(match.group(1)) if match else 10**12, value.casefold())


def frame_view(frame: dict[str, Any], threshold: float) -> tuple[str, float]:
    matrix = frame.get("transform_matrix")
    if not matrix or len(matrix) < 3 or len(matrix[2]) < 3:
        raise ValueError(f"Frame lacks transform_matrix: {frame.get('file_path')}")
    verticality = abs(float(matrix[2][2]))
    return ("nadir" if verticality >= threshold else "oblique", verticality)


def evenly_spaced(items: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if len(items) <= count:
        return items
    if count <= 0:
        return []
    # Midpoint sampling is deterministic and spreads samples over the flight.
    indices = [min(len(items) - 1, int((i + 0.5) * len(items) / count)) for i in range(count)]
    return [items[i] for i in indices]


def lookup_paths(api: HfApi, repo: str, revision: str, paths: list[str]) -> dict[str, dict[str, Any]]:
    """Query Hub path metadata in batches; this endpoint never returns file bodies."""
    found: dict[str, dict[str, Any]] = {}
    for start in range(0, len(paths), 400):
        batch = paths[start : start + 400]
        for item in api.get_paths_info(
            repo, batch, repo_type="dataset", revision=revision, token=False
        ):
            item_size = getattr(item, "size", None)
            if item_size is None:
                continue
            lfs = getattr(item, "lfs", None) or {}
            lfs_oid = lfs.get("oid") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
            found[item.path] = {
                "size": item_size,
                "blob_id": getattr(item, "blob_id", None),
                "lfs_oid": lfs_oid,
            }
    return found


def read_existing_eval_scenes(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        return {row["scene"].replace("\\", "/").strip("/") for row in reader if row.get("scene")}


def verify_plan(plan: dict[str, Any], existing_eval_scenes: set[str]) -> None:
    splits = {
        "train": set(plan["train_scenes"]),
        "validation": set(plan["validation_scenes"]),
        "independent_test": set(plan["independent_test_scenes"]),
    }
    labels = list(splits)
    for index, first in enumerate(labels):
        for second in labels[index + 1 :]:
            overlap = splits[first] & splits[second]
            if overlap:
                raise ValueError(f"Scene leakage between {first} and {second}: {sorted(overlap)}")

    protected_days = set(plan["protected_evaluation_days"])
    all_scenes = set.union(*splits.values())
    bad_days = [scene for scene in all_scenes if any(scene.startswith(day + "_") for day in protected_days)]
    if bad_days:
        raise ValueError(f"Planned scenes overlap protected evaluation days: {bad_days}")
    exact_overlap = all_scenes & existing_eval_scenes
    if exact_overlap:
        raise ValueError(f"Planned scenes overlap the existing 52-pair evaluation: {sorted(exact_overlap)}")

    required_crops = set(plan["required_crops"])
    for split_name, scenes in splits.items():
        present = {PurePosixPath(scene).name.split("_", 1)[1].casefold() for scene in scenes}
        if split_name in {"train", "validation"} and not required_crops.issubset(present):
            raise ValueError(f"{split_name} does not cover required crops: {required_crops - present}")
    def day_key(scene: str) -> str:
        parts = scene.split("/")
        return f"{parts[0]}/{parts[1].split('_', 1)[0]}"

    train_days = {day_key(scene) for scene in splits["train"]}
    val_days = {day_key(scene) for scene in splits["validation"]}
    test_days = {day_key(scene) for scene in splits["independent_test"]}
    if train_days & val_days or train_days & test_days or val_days & test_days:
        raise ValueError("Acquisition days must be disjoint across train/validation/test")


def plan_scene_rows(
    session: requests.Session,
    api: HfApi,
    *,
    scene: str,
    split: str,
    rgb_repo: str,
    depth_repo: str,
    rgb_revision: str,
    depth_revision: str,
    max_per_view: int,
    minimum_per_view: int,
    nadir_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scene_path = PurePosixPath(scene)
    year = scene_path.parts[0]
    scene_name = scene_path.name
    transforms_path = scene_path / "transforms.json"
    transforms_url = (
        f"https://huggingface.co/datasets/{rgb_repo}/resolve/{rgb_revision}/"
        f"{quote(str(transforms_path), safe='/')}"
    )
    response = session.get(transforms_url, timeout=90)
    response.raise_for_status()
    transform_payload = response.content
    transforms = response.json()
    frames = transforms.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"No frames in metadata for {scene}")

    by_view: dict[str, list[dict[str, Any]]] = {"nadir": [], "oblique": []}
    for frame in frames:
        file_ref = PurePosixPath(str(frame.get("file_path", "")).replace("\\", "/"))
        filename = file_ref.name
        if not filename:
            continue
        view, verticality = frame_view(frame, nadir_threshold)
        by_view[view].append({"filename": filename, "verticality": verticality})
    for values in by_view.values():
        values.sort(key=lambda row: natural_key(row["filename"]))

    rows: list[dict[str, Any]] = []
    paired_counts: Counter[str] = Counter()
    missing_image = 0
    missing_depth = 0
    missing_pairs = 0
    missing_details: list[dict[str, Any]] = []
    selected_by_view = {
        view: evenly_spaced(by_view[view], max_per_view)
        for view in ("nadir", "oblique")
    }
    candidates: list[dict[str, Any]] = []
    for view, chosen in selected_by_view.items():
        for selected in chosen:
            filename = selected["filename"]
            stem = PurePosixPath(filename).stem
            image_rel = PurePosixPath(year) / scene_name / "images" / filename
            depth_rel = PurePosixPath(year) / f"{scene_name}_depth" / f"{stem}_depth.tif"
            candidates.append({
                "view": view,
                "selected": selected,
                "filename": filename,
                "image_rel": str(image_rel),
                "depth_rel": str(depth_rel),
            })
    image_info = lookup_paths(api, rgb_repo, rgb_revision, [row["image_rel"] for row in candidates])
    depth_info = lookup_paths(api, depth_repo, depth_revision, [row["depth_rel"] for row in candidates])

    for candidate in candidates:
        view = candidate["view"]
        selected = candidate["selected"]
        filename = candidate["filename"]
        image_rel = PurePosixPath(candidate["image_rel"])
        depth_rel = PurePosixPath(candidate["depth_rel"])
        image_meta = image_info.get(candidate["image_rel"])
        depth_meta = depth_info.get(candidate["depth_rel"])
        if image_meta is None:
            missing_image += 1
        if depth_meta is None:
            missing_depth += 1
        if image_meta is None or depth_meta is None:
            missing_pairs += 1
            missing_details.append({
                "view": view,
                "filename": filename,
                "image_relpath": str(image_rel),
                "depth_relpath": str(depth_rel),
                "missing_rgb": image_meta is None,
                "missing_depth": depth_meta is None,
            })
            continue
        crop = scene_name.split("_", 1)[1].casefold()
        day = f"{year}/{scene_name.split('_', 1)[0]}"
        rows.append({
            "split": split,
            "scene": scene,
            "day": day,
            "crop": crop,
            "view": view,
            "verticality": selected["verticality"],
            "filename": filename,
            "image_relpath": str(image_rel),
            "depth_relpath": str(depth_rel),
            "image_url": f"https://huggingface.co/datasets/{rgb_repo}/resolve/{rgb_revision}/{quote(str(image_rel), safe='/')}",
            "depth_url": f"https://huggingface.co/datasets/{depth_repo}/resolve/{depth_revision}/{quote(str(depth_rel), safe='/')}",
            "image_size_bytes": image_meta["size"],
            "depth_size_bytes": depth_meta["size"],
            "image_blob_id": image_meta["blob_id"] or "",
            "depth_blob_id": depth_meta["blob_id"] or "",
            "image_lfs_oid": image_meta["lfs_oid"] or "",
            "depth_lfs_oid": depth_meta["lfs_oid"] or "",
            "source_rgb_revision": rgb_revision,
            "source_depth_revision": depth_revision,
        })
        paired_counts[view] += 1
    for view in ("nadir", "oblique"):
        if paired_counts[view] < minimum_per_view:
            raise ValueError(
                f"{scene} has only {paired_counts[view]} verified {view} pairs; "
                f"minimum required is {minimum_per_view}"
            )
    return rows, {
        "scene": scene,
        "split": split,
        "available_frames_by_view": {key: len(value) for key, value in by_view.items()},
        "selected_pairs_by_view": dict(paired_counts),
        "selected_candidates_by_view": {key: len(value) for key, value in selected_by_view.items()},
        "missing_rgb_candidates": missing_image,
        "missing_depth_candidates": missing_depth,
        "missing_pair_candidates": missing_pairs,
        "missing_candidate_details": missing_details,
        "transforms_sha256": hashlib.sha256(transform_payload).hexdigest(),
        "transforms_size_bytes": len(transform_payload),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing plan file: {path}")
    if not rows:
        raise ValueError(f"Refusing to write an empty manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a metadata-only plan for selected UAV3DCrop pairs.")
    parser.add_argument("--plan", type=Path, default=Path(__file__).parent / "config" / "scene_split_plan.json")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "work" / "uav_metric_finetuning" / "remote_plan")
    parser.add_argument("--existing-eval", type=Path, default=REPO_ROOT / "results" / "uav3dcrop" / "all_selected_images.csv")
    parser.add_argument("--max-per-scene-view", type=int, default=None)
    parser.add_argument("--minimum-per-scene-view", type=int, default=8)
    args = parser.parse_args()
    plan = read_json(args.plan)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"Output directory is not empty; refusing to overwrite: {args.output_dir}")
    existing_eval = read_existing_eval_scenes(args.existing_eval)
    verify_plan(plan, existing_eval)
    max_per_view = args.max_per_scene_view or int(plan["max_frames_per_scene_per_view"])

    api = HfApi()
    rgb_revision = api.dataset_info(plan["rgb_repo"], revision=plan.get("rgb_revision", plan.get("revision", "main"))).sha
    depth_revision = api.dataset_info(plan["depth_repo"], revision=plan.get("depth_revision", plan.get("revision", "main"))).sha
    session = requests.Session()
    session.headers.update({"User-Agent": "uav-metric-depth-research/0.1"})

    split_scenes = {
        "train": plan["train_scenes"],
        "validation": plan["validation_scenes"],
        "independent_test": plan["independent_test_scenes"],
    }
    all_rows: dict[str, list[dict[str, Any]]] = {key: [] for key in split_scenes}
    scene_reports: list[dict[str, Any]] = []
    for split, scenes in split_scenes.items():
        for scene in scenes:
            rows, report = plan_scene_rows(
                session,
                api,
                scene=scene,
                split=split,
                rgb_repo=plan["rgb_repo"],
                depth_repo=plan["depth_repo"],
                rgb_revision=rgb_revision,
                depth_revision=depth_revision,
                max_per_view=max_per_view,
                minimum_per_view=args.minimum_per_scene_view,
                nadir_threshold=float(plan["nadir_threshold"]),
            )
            all_rows[split].extend(rows)
            scene_reports.append(report)

    sizes_known = all(
        row["image_size_bytes"] is not None and row["depth_size_bytes"] is not None
        for rows in all_rows.values() for row in rows
    )
    total_bytes = sum(
        int(row["image_size_bytes"]) + int(row["depth_size_bytes"])
        for rows in all_rows.values() for row in rows
        if row["image_size_bytes"] is not None and row["depth_size_bytes"] is not None
    )
    selected_bytes_by_split = {}
    for split, rows in all_rows.items():
        split_sizes_complete = all(
            row["image_size_bytes"] is not None and row["depth_size_bytes"] is not None
            for row in rows
        )
        split_total = sum(
            int(row["image_size_bytes"]) + int(row["depth_size_bytes"])
            for row in rows
            if row["image_size_bytes"] is not None and row["depth_size_bytes"] is not None
        )
        selected_bytes_by_split[split] = {
            "bytes": split_total if split_sizes_complete else None,
            "gib": split_total / (1024**3) if split_sizes_complete else None,
            "sizes_complete": split_sizes_complete,
        }
    summary: dict[str, Any] = {
        "plan": str(args.plan.resolve()),
        "seed": int(plan["seed"]),
        "rgb_revision": rgb_revision,
        "depth_revision": depth_revision,
        "max_pairs_per_scene_per_view": max_per_view,
        "nadir_rule": f"abs(transform_matrix[2][2]) >= {plan['nadir_threshold']}",
        "selected_pair_counts": {split: len(rows) for split, rows in all_rows.items()},
        "split_distributions": {
            split: {
                "scenes": dict(Counter(row["scene"] for row in rows)),
                "crops": dict(Counter(row["crop"] for row in rows)),
                "days": dict(Counter(row["day"] for row in rows)),
                "views": dict(Counter(row["view"] for row in rows)),
            }
            for split, rows in all_rows.items()
        },
        "total_selected_bytes": total_bytes if sizes_known else None,
        "total_selected_gib": total_bytes / (1024**3) if sizes_known else None,
        "sizes_complete": sizes_known,
        "selected_bytes_by_split": selected_bytes_by_split,
        "scene_metadata": scene_reports,
        "downloaded_rgb_or_depth_frames": 0,
        "message": "This plan queries small transforms.json files and Hugging Face path metadata (file names, sizes, IDs) only. It does not download RGB or depth frames.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in all_rows.items():
        write_csv(args.output_dir / f"{split}.csv", rows)
    summary["plan_sha256"] = hashlib.sha256(
        json.dumps(summary, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    summary_path = args.output_dir / "plan_summary.json"
    if summary_path.exists():
        raise FileExistsError(f"Refusing to overwrite: {summary_path}")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
