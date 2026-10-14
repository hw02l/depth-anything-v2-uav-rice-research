from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

if __package__:
    from .fetch_selected_data import safe_target
else:
    from fetch_selected_data import safe_target


def entries_for(plan_dir: Path, splits: list[str]) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for split in splits:
        with (plan_dir / f"{split}.csv").open("r", encoding="utf-8-sig", newline="") as stream:
            entries.extend(csv.DictReader(stream))
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy already-downloaded selected pairs from persistent storage to local scratch.")
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument("--source-rgb-root", type=Path, required=True)
    parser.add_argument("--source-depth-root", type=Path, required=True)
    parser.add_argument("--target-rgb-root", type=Path, required=True)
    parser.add_argument("--target-depth-root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", choices=("train", "validation", "independent_test"), default=("train", "validation"))
    parser.add_argument("--stage", action="store_true", help="Explicitly allow local copy; no network access is performed")
    args = parser.parse_args()
    if not args.stage:
        parser.error("No files copied. Review the source and local free space, then rerun with --stage.")

    rows = entries_for(args.plan_dir, list(args.splits))
    files: list[tuple[Path, Path, int]] = []
    for row in rows:
        for source_root, target_root, rel_key, size_key in (
            (args.source_rgb_root, args.target_rgb_root, "image_relpath", "image_size_bytes"),
            (args.source_depth_root, args.target_depth_root, "depth_relpath", "depth_size_bytes"),
        ):
            source = safe_target(source_root, row[rel_key])
            target = safe_target(target_root, row[rel_key])
            expected = int(row[size_key])
            if not source.is_file():
                raise FileNotFoundError(f"Persistent source file is missing: {source}")
            if source.stat().st_size != expected:
                raise IOError(f"Persistent source size mismatch: {source} expected={expected}, actual={source.stat().st_size}")
            if target.exists():
                if target.stat().st_size != expected:
                    raise FileExistsError(f"Refusing to overwrite a different local file: {target}")
                continue
            files.append((source, target, expected))

    needed_bytes = sum(size for _, _, size in files)
    target_root = args.target_rgb_root.resolve()
    target_root.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(target_root).free
    # Keep a 10% scratch margin for temporary files and runtime logs.
    required_bytes = int(needed_bytes * 1.1)
    print(f"Pairs listed: {len(rows)}; files to copy: {len(files)}")
    print(f"Copy bytes: {needed_bytes:,} ({needed_bytes / 1024**3:.2f} GiB); free local storage: {free_bytes / 1024**3:.2f} GiB")
    if free_bytes < required_bytes:
        raise OSError(f"Not enough local storage: require at least {required_bytes / 1024**3:.2f} GiB including margin.")

    for index, (source, target, expected) in enumerate(files, start=1):
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".uavmetric.copying")
        if temporary.exists():
            raise FileExistsError(f"Temporary staging file exists; inspect before retry: {temporary}")
        try:
            shutil.copyfile(source, temporary)
            if temporary.stat().st_size != expected:
                raise IOError(f"Staged file size mismatch: {temporary}")
            if target.exists():
                raise FileExistsError(f"Target appeared while copying; preserving it: {target}")
            temporary.replace(target)
        except Exception:
            if temporary.exists():
                temporary.unlink()
            raise
        if index % 25 == 0 or index == len(files):
            print(f"Staged {index}/{len(files)} files")
    print("Staging complete. Persistent source files were retained unchanged.")


if __name__ == "__main__":
    main()
