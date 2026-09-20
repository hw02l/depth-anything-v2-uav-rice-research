from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from da2_experiments.io import write_csv, write_json  # noqa: E402


RGB_REPO = "Link-Dev/UAV3DCrop"
DEPTH_REPO = "Link-Dev/UAV3DCrop_depth"
HF_API = "https://huggingface.co/api/datasets"
HF_RESOLVE = "https://huggingface.co/datasets"
CAMERA_TYPE = "DJI Mavic 3M RGB camera"


def api_tree(repo: str, path: str, revision: str) -> list[dict]:
    url = f"{HF_API}/{repo}/tree/{revision}/{path}?recursive=false&expand=false"
    response = requests.get(url, timeout=60, verify=False)
    response.raise_for_status()
    return response.json()


def resolve(repo: str, path: str, revision: str) -> str:
    return f"{HF_RESOLVE}/{repo}/resolve/{revision}/{path}?download=true"


def download(repo: str, path: str, revision: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return
    response = requests.get(resolve(repo, path, revision), timeout=300, verify=False, stream=True)
    response.raise_for_status()
    temporary = destination.with_suffix(destination.suffix + ".part")
    with temporary.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)
    temporary.replace(destination)


def frame_number(path: str) -> int:
    match = re.search(r"(\d+)(?:_depth)?\.(?:JPG|jpg|tif|TIF)$", path)
    if match is None:
        raise ValueError(f"Unexpected image path: {path}")
    return int(match.group(1))


def classify_view(frame: dict) -> tuple[str, float]:
    rotation = np.asarray(frame["transform_matrix"], dtype=np.float64)[:3, :3]
    # transforms.json is a Nerfstudio-compatible camera-to-world matrix.  In
    # this release the optical-axis verticality is |R[2,2]|: approximately 1
    # for the nadir grid and approximately cos(45 degrees) for oblique lines.
    score = float(abs(rotation[2, 2]))
    return ("nadir" if score >= 0.90 else "oblique"), score


def evenly_select(items: list[dict], count: int) -> list[dict]:
    if count <= 0:
        return []
    if len(items) <= count:
        return items
    indices = np.linspace(0, len(items) - 1, count, dtype=int)
    return [items[int(index)] for index in indices]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a minimal UAV3DCrop RGB/depth subset")
    parser.add_argument("--scene", default="2025/Day001_Wheat")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--nadir-count", type=int, default=10)
    parser.add_argument("--oblique-count", type=int, default=10)
    parser.add_argument("--data-root", default="data/uav3dcrop")
    parser.add_argument("--depth-root", default="data/uav3dcrop_depth")
    parser.add_argument("--results-csv", default="results/uav3dcrop/selected_images.csv")
    parser.add_argument("--manifest", default="results/uav3dcrop/selection_manifest.json")
    args = parser.parse_args()
    requests.packages.urllib3.disable_warnings()

    scene_path = args.scene.strip("/")
    scene_parts = scene_path.split("/")
    if len(scene_parts) != 2:
        raise ValueError("--scene must look like 2025/Day001_Wheat")
    year, scene_name = scene_parts
    crop = scene_name.split("_", 1)[1].lower()
    date = scene_name.split("_", 1)[0]

    transforms_path = Path(args.data_root) / scene_path / "transforms.json"
    sparse_path = Path(args.data_root) / scene_path / "sparse_pc.ply"
    download(RGB_REPO, f"{scene_path}/transforms.json", args.revision, transforms_path)
    download(RGB_REPO, f"{scene_path}/sparse_pc.ply", args.revision, sparse_path)
    transforms = json.loads(transforms_path.read_text(encoding="utf-8-sig"))

    image_prefix = f"{scene_path}/"
    image_entries = {
        entry["path"][len(image_prefix):]: entry
        for entry in api_tree(RGB_REPO, f"{scene_path}/images", args.revision)
        if entry.get("type") == "file" and entry["path"].startswith(image_prefix)
    }
    depth_scene = f"{scene_path}_depth"
    depth_entries = {entry["path"]: entry for entry in api_tree(DEPTH_REPO, depth_scene, args.revision) if entry.get("type") == "file"}
    depth_by_number = {frame_number(path): (path, entry) for path, entry in depth_entries.items() if path.endswith("_depth.tif")}

    candidates: list[dict] = []
    missing_depth = []
    for frame in transforms["frames"]:
        image_path = frame["file_path"].replace("\\", "/")
        if image_path not in image_entries:
            continue
        number = frame_number(image_path)
        if number not in depth_by_number:
            missing_depth.append(number)
            continue
        view_type, verticality = classify_view(frame)
        candidates.append({
            "frame": frame,
            "number": number,
            "image_repo_path": image_path,
            "image_entry": image_entries[image_path],
            "depth_repo_path": depth_by_number[number][0],
            "depth_entry": depth_by_number[number][1],
            "view_type": view_type,
            "view_verticality_score": verticality,
        })
    candidates.sort(key=lambda item: item["number"])
    selected = evenly_select([item for item in candidates if item["view_type"] == "nadir"], args.nadir_count)
    selected += evenly_select([item for item in candidates if item["view_type"] == "oblique"], args.oblique_count)
    selected.sort(key=lambda item: (item["view_type"], item["number"]))

    rows = []
    for item in selected:
        number = item["number"]
        rgb_destination = Path(args.data_root) / scene_path / item["image_repo_path"].replace("/", "/")
        depth_destination = Path(args.depth_root) / scene_path / f"{number}_depth.tif"
        download(RGB_REPO, f"{scene_path}/{item['image_repo_path']}", args.revision, rgb_destination)
        download(DEPTH_REPO, item["depth_repo_path"], args.revision, depth_destination)
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
            "source_rgb": f"https://huggingface.co/datasets/{RGB_REPO}/blob/{args.revision}/{scene_path}/{item['image_repo_path']}",
            "source_depth": f"https://huggingface.co/datasets/{DEPTH_REPO}/blob/{args.revision}/{item['depth_repo_path']}",
            "source_revision": args.revision,
        })

    fields = list(rows[0].keys()) if rows else []
    write_csv(ROOT / args.results_csv, rows, fields)
    write_json(ROOT / args.manifest, {
        "rgb_repo": RGB_REPO,
        "depth_repo": DEPTH_REPO,
        "revision_requested": args.revision,
        "canonical_release_note": "Use current default revision; dataset card identifies the canonical JSON/PLY release as 2026-08-15. No legacy normalization is applied.",
        "scene": scene_path,
        "selected_counts": {"nadir": sum(r["view_type"] == "nadir" for r in rows), "oblique": sum(r["view_type"] == "oblique" for r in rows)},
        "available_counts": {"with_rgb_and_depth": len(candidates), "nadir": sum(c["view_type"] == "nadir" for c in candidates), "oblique": sum(c["view_type"] == "oblique" for c in candidates)},
        "missing_depth_frame_ids": sorted(set(missing_depth)),
        "transform_metadata": {key: transforms.get(key) for key in ("w", "h", "fl_x", "fl_y", "cx", "cy", "k1", "k2", "k3", "p1", "p2", "camera_model", "scale", "avg_pos", "applied_transform", "ply_file_path")},
        "view_classification": "nadir if abs(transform_matrix[:3,:3][2,2]) >= 0.90, otherwise oblique; the threshold separates the observed approximately 1.0 nadir and approximately 0.707 oblique orientations",
        "rows": rows,
    })
    print(f"[uav3dcrop] selected {len(rows)} images from {scene_path}: {sum(r['view_type'] == 'nadir' for r in rows)} nadir, {sum(r['view_type'] == 'oblique' for r in rows)} oblique")


if __name__ == "__main__":
    main()
