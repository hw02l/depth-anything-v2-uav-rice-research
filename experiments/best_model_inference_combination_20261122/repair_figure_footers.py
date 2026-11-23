"""Repair only the bottom legend strips on the fixed-eight composite PNGs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2


IMAGE_IDS = ("007", "020", "034", "048", "071", "089", "097", "112")
FOOTER_HEIGHT = 48
BACKGROUND = (32, 32, 32)
FOREGROUND = (245, 245, 245)


def repair_one(path: Path, text: str) -> None:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"cannot read {path}")
    height, width = image.shape[:2]
    if height <= FOOTER_HEIGHT:
        raise ValueError(f"unexpected image dimensions: {path} {image.shape}")
    footer = image[height - FOOTER_HEIGHT :, :]
    footer[:] = BACKGROUND
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2)
    if text_width >= width - 8 or text_height + baseline > FOOTER_HEIGHT - 4:
        raise ValueError(f"footer does not fit: {path} {(text_width, text_height, baseline)} {(width, height)}")
    cv2.putText(footer, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.72, FOREGROUND, 2, cv2.LINE_AA)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"cannot write {path}")


def repair_directory(figure_dir: Path, indices_csv: Path) -> list[Path]:
    repaired: list[Path] = []
    with indices_csv.open(newline="", encoding="utf-8-sig") as handle:
        records = {f"{int(row['image_index']):03d}": row for row in csv.DictReader(handle)}
    if tuple(records) != IMAGE_IDS:
        raise ValueError(f"unexpected fixed image order in {indices_csv}: {tuple(records)}")
    for image_id in IMAGE_IDS:
        depth = figure_dir / f"depth_maps_{image_id}.png"
        error = figure_dir / f"absolute_error_{image_id}.png"
        if not depth.is_file() or not error.is_file():
            raise FileNotFoundError(f"missing fixed-eight image pair for {image_id}")
        crop = records[image_id]["crop"]
        view = records[image_id]["view"]
        repair_one(depth, f"Validation {image_id} | {crop}, {view} | common depth range 0-65 m | invalid GT masked")
        repair_one(error, f"Validation {image_id} | {crop}, {view} | common error range 0-10 m; values above 10 m clipped")
        repaired.extend((depth, error))
    return repaired


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("figure_dir", type=Path)
    parser.add_argument("--indices-csv", type=Path)
    args = parser.parse_args()
    indices_csv = args.indices_csv or args.figure_dir.parents[1] / "fixed8_visualization_indices.csv"
    repaired = repair_directory(args.figure_dir, indices_csv)
    print(f"REPAIRED={len(repaired)}")
    for path in repaired:
        print(path.name)


if __name__ == "__main__":
    main()
