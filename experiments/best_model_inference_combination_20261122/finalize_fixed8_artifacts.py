"""Refresh final figure provenance and SHA256 inventories after footer repair."""

from __future__ import annotations

import hashlib
import json
import argparse
from datetime import datetime, timezone
from pathlib import Path

import cv2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_inventory(directory: Path, output_name: str) -> None:
    output = directory / output_name
    files = sorted(path for path in directory.rglob("*") if path.is_file() and path != output)
    output.write_text("".join(f"{sha256(path)}  {path.relative_to(directory).as_posix()}\n" for path in files), encoding="utf-8", newline="\n")


def main(root: Path) -> None:
    root = root.resolve()
    fig = root / "figures" / "fixed8_visualizations_20261123"
    repair_script = root / "repair_figure_footers.py"
    review_note = root / "results" / "fixed8_visualization_20261123" / "verification" / "visual_review_ja.md"
    provenance_path = fig / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    image_records = []
    image_sizes = {}
    for record in provenance["figures"]:
        path = fig / record["path"]
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise OSError(f"cannot decode {path}")
        image_sizes[path.name] = [int(image.shape[1]), int(image.shape[0])]
        record["sha256"] = sha256(path)
        record["bytes"] = path.stat().st_size
        image_records.append(record)
    repair_sha = sha256(repair_script)
    provenance["post_generation_qa"] = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "complete",
        "visual_review_note": review_note.relative_to(root).as_posix(),
        "footer_repair": {
            "script": repair_script.relative_to(root).as_posix(),
            "script_sha256": repair_sha,
            "method": "redraw only the bottom 48-pixel footer strip on each composite PNG with ASCII hyphen labels",
            "modified_pixels_scope": "footer strip only; RGB, GT, prediction panels, metric CSV, and prediction arrays unchanged",
            "images_repaired": len(image_records),
            "dimensions_px": image_sizes,
        },
        "final_figure_sha256": {record["path"]: record["sha256"] for record in image_records},
    }
    provenance_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    write_inventory(fig, "SHA256SUMS")
    write_inventory(root, "SHA256SUMS")
    print(f"FIGURES={len(image_records)}")
    print(f"FOOTER_REPAIR_SHA256={repair_sha}")
    print(f"FIGURE_SHA256SUMS_ENTRIES={sum(1 for _ in (fig / 'SHA256SUMS').open(encoding='utf-8'))}")
    print(f"ROOT_SHA256SUMS_ENTRIES={sum(1 for _ in (root / 'SHA256SUMS').open(encoding='utf-8'))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", type=Path, default=Path(__file__).resolve().parent / "experiment")
    main(parser.parse_args().root)
