from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlparse

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]


def safe_target(root: Path, relpath: str) -> Path:
    relative = Path(relpath)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe dataset path: {relpath}")
    resolved_root = root.resolve()
    target = (resolved_root / relative).resolve()
    if not target.is_relative_to(resolved_root):
        raise ValueError(f"Dataset path escapes its root: {relpath}")
    return target


def stream_one(session: requests.Session, url: str, target: Path, expected_size: int,
               expected_sha256: str = "") -> tuple[int, str]:
    if target.exists():
        if target.stat().st_size == expected_size:
            return 0, "already-present-same-size"
        raise FileExistsError(f"Refusing to overwrite an existing file with different size: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".uavmetric.part")
    provenance = temporary.with_name(temporary.name + ".json")
    identity = {"url": url, "expected_size": expected_size, "sha256": expected_sha256}
    if (temporary.exists() and not provenance.is_file()) or (provenance.exists() and json.loads(provenance.read_text(encoding="utf-8")) != identity):
        raise FileExistsError(f"由来の不明な部分ファイルを保護して停止しました: {temporary}。別の保存先を使用するか内容を確認してください。")
    if not provenance.exists():
        provenance.write_text(json.dumps(identity), encoding="utf-8")
    offset = temporary.stat().st_size if temporary.exists() else 0
    if offset > expected_size:
        raise IOError(f"部分ファイルが予定サイズを超えています: {temporary}")
    transferred = 0
    if offset < expected_size:
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        # On interruption keep only our identified partial file. A rerun resumes it.
        with session.get(url, headers=headers, stream=True, allow_redirects=True, timeout=(30, 180)) as response:
            response.raise_for_status()
            if response.status_code == 206:
                match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                if not match or int(match[1]) != offset or int(match[3]) != expected_size:
                    raise IOError("HTTP Range応答が部分ファイルと一致しません。再実行前に確認してください。")
                mode = "ab" if offset else "xb"
            elif response.status_code == 200:
                # Server does not support Range: restart only this identified incomplete file.
                mode = "wb" if temporary.exists() else "xb"
                offset = 0
            else:
                raise IOError(f"Unexpected download HTTP status: {response.status_code}")
            with temporary.open(mode) as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    output.write(chunk)
                    transferred += len(chunk)
    actual_size = temporary.stat().st_size
    if actual_size != expected_size:
        raise IOError(f"Download size mismatch for {url}: expected {expected_size}, got {actual_size}; partial retained for retry")
    digest = hashlib.sha256()
    with temporary.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    if expected_sha256 and digest.hexdigest() != expected_sha256:
        raise IOError(f"SHA256 mismatch; final file was not published: {temporary}")
    if target.exists():
        raise FileExistsError(f"Target appeared during download; preserving it: {target}")
    temporary.replace(target)
    provenance.unlink()
    return transferred, digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch only the pairs explicitly listed in a reviewed plan.")
    parser.add_argument("--plan-dir", type=Path, default=REPO_ROOT / "work" / "uav_metric_finetuning" / "remote_plan")
    parser.add_argument("--rgb-root", type=Path, default=REPO_ROOT / "data" / "uav3dcrop")
    parser.add_argument("--depth-root", type=Path, default=REPO_ROOT / "data" / "uav3dcrop_depth")
    parser.add_argument("--splits", nargs="+", choices=("train", "validation", "independent_test"),
                        default=("train", "validation"),
                        help="Default downloads train/validation only; fetch independent_test separately after model selection.")
    parser.add_argument("--download", action="store_true", help="Explicitly allow the planned file transfer")
    parser.add_argument("--confirm-total-bytes", type=int, default=None,
                        help="Must equal total_selected_bytes in plan_summary.json")
    args = parser.parse_args()
    if not args.download:
        parser.error("No transfer performed. Review plan_summary.json, then rerun with --download and --confirm-total-bytes.")

    summary = json.loads((args.plan_dir / "plan_summary.json").read_text(encoding="utf-8"))
    split_sizes = summary.get("selected_bytes_by_split", {})
    expected_total = None
    if all(split_sizes.get(split, {}).get("bytes") is not None for split in args.splits):
        expected_total = sum(int(split_sizes[split]["bytes"]) for split in args.splits)
    elif set(args.splits) == {"train", "validation", "independent_test"}:
        expected_total = summary.get("total_selected_bytes")
    if expected_total is None:
        parser.error("Plan lacks complete file sizes; refusing a transfer with unknown total size.")
    if args.confirm_total_bytes != expected_total:
        parser.error(f"Confirmation mismatch. Plan total is {expected_total} bytes; no files were downloaded.")

    entries = []
    for split in args.splits:
        manifest = args.plan_dir / f"{split}.csv"
        with manifest.open("r", encoding="utf-8-sig", newline="") as stream:
            entries.extend(csv.DictReader(stream))
    targets: set[Path] = set()
    for row in entries:
        for root, rel_key in ((args.rgb_root, "image_relpath"), (args.depth_root, "depth_relpath")):
            target = safe_target(root, row[rel_key])
            size_key = "image_size_bytes" if rel_key == "image_relpath" else "depth_size_bytes"
            if target.exists() and target.stat().st_size != int(row[size_key]):
                parser.error(f"既存ファイルのサイズが異なるため、通信前に停止します: {target}")
            if target in targets:
                parser.error(f"Duplicate output path in plan: {target}")
            targets.add(target)
        for key, revision_key, repo_key in (("image_url", "rgb_revision", "source_rgb_revision"),
                                            ("depth_url", "depth_revision", "source_depth_revision")):
            revision = summary[revision_key]
            if row.get(repo_key) != revision or f"/resolve/{revision}/" not in row[key]:
                parser.error("Manifestのrevisionが固定計画と一致しません。転送しません。")

    calculated_total = sum(int(row["image_size_bytes"]) + int(row["depth_size_bytes"]) for row in entries)
    if calculated_total != expected_total:
        parser.error("Plan CSV sizes no longer match plan_summary.json; no files were downloaded.")
    print(f"Downloading {len(entries)} curated RGB/depth pairs ({expected_total} bytes, {expected_total / 1024**3:.2f} GiB).")
    session = requests.Session()
    session.headers.update({"User-Agent": "uav-metric-depth-research/0.1"})
    transferred = 0
    for index, row in enumerate(entries, start=1):
        image_target = safe_target(args.rgb_root, row["image_relpath"])
        depth_target = safe_target(args.depth_root, row["depth_relpath"])
        image_bytes, image_status = stream_one(
            session, row["image_url"], image_target, int(row["image_size_bytes"]), row.get("image_lfs_oid", "")
        )
        depth_bytes, depth_status = stream_one(
            session, row["depth_url"], depth_target, int(row["depth_size_bytes"]), row.get("depth_lfs_oid", "")
        )
        transferred += image_bytes + depth_bytes
        print(f"[{index}/{len(entries)}] {row['scene']} {row['filename']} ({image_status}, {depth_status})")
    print(f"Completed. Newly transferred {transferred} bytes; existing same-size files were left unchanged.")


if __name__ == "__main__":
    main()
