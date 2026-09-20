from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_uav3dcrop import (  # noqa: E402
    CAMERA_TYPE,
    DEPTH_REPO,
    HF_RESOLVE,
    RGB_REPO,
    api_tree,
    classify_view,
    download,
    frame_number,
    evenly_select,
)
from da2_experiments.io import write_csv, write_json  # noqa: E402


def prepare_scene(scene_path: str, revision: str, nadir_count: int, oblique_count: int, data_root: Path, depth_root: Path) -> tuple[list[dict], dict]:
    year, scene_name = scene_path.split("/", 1)
    crop = scene_name.split("_", 1)[1].lower()
    date = scene_name.split("_", 1)[0]
    transforms_path = data_root / scene_path / "transforms.json"
    sparse_path = data_root / scene_path / "sparse_pc.ply"
    download(RGB_REPO, f"{scene_path}/transforms.json", revision, transforms_path)
    download(RGB_REPO, f"{scene_path}/sparse_pc.ply", revision, sparse_path)
    transforms = json.loads(transforms_path.read_text(encoding="utf-8-sig"))
    image_prefix = f"{scene_path}/"
    image_entries = {entry["path"][len(image_prefix):]: entry for entry in api_tree(RGB_REPO, f"{scene_path}/images", revision) if entry.get("type") == "file" and entry["path"].startswith(image_prefix)}
    depth_scene = f"{scene_path}_depth"
    depth_entries = {entry["path"]: entry for entry in api_tree(DEPTH_REPO, depth_scene, revision) if entry.get("type") == "file"}
    depth_by_number = {frame_number(path): path for path in depth_entries if path.endswith("_depth.tif")}
    candidates = []
    for frame in transforms["frames"]:
        image_path = frame["file_path"].replace("\\", "/")
        if image_path not in image_entries:
            continue
        number = frame_number(image_path)
        if number not in depth_by_number:
            continue
        view_type, verticality = classify_view(frame)
        candidates.append({"frame": frame, "number": number, "image_repo_path": image_path, "depth_repo_path": depth_by_number[number], "view_type": view_type, "view_verticality_score": verticality})
    candidates.sort(key=lambda item: item["number"])
    selected = evenly_select([item for item in candidates if item["view_type"] == "nadir"], nadir_count)
    selected += evenly_select([item for item in candidates if item["view_type"] == "oblique"], oblique_count)
    selected.sort(key=lambda item: (item["view_type"], item["number"]))
    rows = []
    for item in selected:
        number = item["number"]
        rgb_destination = data_root / scene_path / item["image_repo_path"]
        depth_destination = depth_root / scene_path / f"{number}_depth.tif"
        download(RGB_REPO, f"{scene_path}/{item['image_repo_path']}", revision, rgb_destination)
        download(DEPTH_REPO, item["depth_repo_path"], revision, depth_destination)
        rows.append({
            "scene": scene_path,
            "year": year,
            "date": date,
            "crop": crop,
            "filename": Path(item["image_repo_path"]).name,
            "frame_id": number,
            "camera_type": CAMERA_TYPE,
            "view_type": item["view_type"],
            "nadir_or_oblique": item["view_type"],
            "view_verticality_score": item["view_verticality_score"],
            "width": int(transforms["w"]),
            "height": int(transforms["h"]),
            "corresponding_depth_filename": f"{number}_depth.tif",
            "intrinsics_available": True,
            "extrinsics_available": True,
            "camera_model": transforms.get("camera_model", ""),
            "local_rgb_path": str(rgb_destination).replace("/", "\\"),
            "local_depth_path": str(depth_destination).replace("/", "\\"),
            "source_rgb": f"{HF_RESOLVE}/{RGB_REPO}/resolve/{revision}/{scene_path}/{item['image_repo_path']}?download=true",
            "source_depth": f"{HF_RESOLVE}/{DEPTH_REPO}/resolve/{revision}/{item['depth_repo_path']}?download=true",
            "source_revision": revision,
        })
    manifest = {
        "scene": scene_path,
        "revision": revision,
        "selected_counts": {"nadir": sum(row["view_type"] == "nadir" for row in rows), "oblique": sum(row["view_type"] == "oblique" for row in rows)},
        "available_counts": {"with_rgb_and_depth": len(candidates), "nadir": sum(item["view_type"] == "nadir" for item in candidates), "oblique": sum(item["view_type"] == "oblique" for item in candidates)},
        "transform_metadata": {key: transforms.get(key) for key in ("w", "h", "fl_x", "fl_y", "cx", "cy", "k1", "k2", "k3", "p1", "p2", "camera_model", "scale", "avg_pos", "applied_transform", "ply_file_path")},
        "rows": rows,
    }
    return rows, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Download minimal additional UAV3DCrop scene subsets")
    parser.add_argument("--scenes", nargs="+", default=["2025/Day001_Oat", "2025/Day007_Corn"])
    parser.add_argument("--revision", default="main")
    parser.add_argument("--nadir-count", type=int, default=8)
    parser.add_argument("--oblique-count", type=int, default=8)
    parser.add_argument("--data-root", default="data/uav3dcrop")
    parser.add_argument("--depth-root", default="data/uav3dcrop_depth")
    parser.add_argument("--results-csv", default="results/uav3dcrop/multiscene_selected_images.csv")
    parser.add_argument("--manifest", default="results/uav3dcrop/multiscene_selection_manifest.json")
    args = parser.parse_args()
    all_rows = []
    manifests = []
    for scene in args.scenes:
        rows, manifest = prepare_scene(scene.strip("/"), args.revision, args.nadir_count, args.oblique_count, ROOT / args.data_root, ROOT / args.depth_root)
        all_rows.extend(rows)
        manifests.append(manifest)
        print(f"[uav3dcrop] {scene}: {len(rows)} images ({manifest['selected_counts']['nadir']} nadir, {manifest['selected_counts']['oblique']} oblique)")
    all_rows.sort(key=lambda row: (row["scene"], row["view_type"], int(row["frame_id"])))
    write_csv(ROOT / args.results_csv, all_rows, list(all_rows[0].keys()) if all_rows else [])
    write_json(ROOT / args.manifest, {"revision": args.revision, "scenes": manifests, "total_selected": len(all_rows), "selection": "8 nadir + 8 oblique per additional scene by default"})
    print(f"[uav3dcrop] wrote {len(all_rows)} multi-scene selections")


if __name__ == "__main__":
    main()
