from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import requests
import urllib3
from PIL import ExifTags, Image

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

API_ROOT = "https://api.github.com/repos/LUMS-WIT/infrastructure-free-uav-phenotyping/contents"
RAW_ROOT = "https://raw.githubusercontent.com/LUMS-WIT/infrastructure-free-uav-phenotyping/main"
DEFAULT_DATES = {
    "early": "2025-08-22",
    "middle": "2025-09-12",
    "late": "2025-10-03",
}


def list_pngs(session: requests.Session, date: str) -> list[dict]:
    response = session.get(f"{API_ROOT}/data/{date}", params={"ref": "main"}, verify=False, timeout=60)
    response.raise_for_status()
    return sorted([item for item in response.json() if item["name"].lower().endswith(".png")], key=lambda item: item["name"])


def choose_evenly(items: list[dict], count: int) -> list[dict]:
    if len(items) < count:
        raise ValueError(f"Only {len(items)} files are available; cannot select {count}")
    indices = [round(index * (len(items) - 1) / max(1, count - 1)) for index in range(count)]
    return [items[index] for index in indices]


def exif_summary(image: Image.Image) -> tuple[str, str]:
    exif = image.getexif()
    values = {}
    for key, value in exif.items():
        if key not in {34665, 34853}:
            values[ExifTags.TAGS.get(key, str(key))] = str(value)
    ifd_exif = exif.get_ifd(34665)
    for key, value in ifd_exif.items():
        values[ExifTags.TAGS.get(key, str(key))] = str(value)
    ifd_gps = exif.get_ifd(34853)
    for key, value in ifd_gps.items():
        values[f"GPS_{ExifTags.GPSTAGS.get(key, str(key))}"] = str(value)
    if not values:
        return "none", "No EXIF metadata in published PNG"
    return "; ".join(sorted(values)), json.dumps(values, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a small, documented subset of public rice UAV RGB frames")
    parser.add_argument("--outdir", default="data/rice_uav")
    parser.add_argument("--results-dir", default="results/rice_uav")
    parser.add_argument("--per-stage", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    session = requests.Session()
    outdir = Path(args.outdir)
    results_dir = Path(args.results_dir)
    rows = []
    manifest = {"repository": "https://github.com/LUMS-WIT/infrastructure-free-uav-phenotyping", "dates": {}}
    for stage, date in DEFAULT_DATES.items():
        available = list_pngs(session, date)
        selected = choose_evenly(available, args.per_stage)
        manifest["dates"][date] = {"growth_stage": stage, "available_files": len(available), "selected": [item["name"] for item in selected]}
        date_dir = outdir / date
        date_dir.mkdir(parents=True, exist_ok=True)
        for item in selected:
            destination = date_dir / item["name"]
            if args.force or not destination.exists():
                response = session.get(item["download_url"], verify=False, timeout=180)
                response.raise_for_status()
                destination.write_bytes(response.content)
            with Image.open(destination) as image:
                width, height = image.size
                exif_keys, exif_values = exif_summary(image)
                image_format = image.format or "unknown"
            rows.append({
                "date": date,
                "filename": item["name"],
                "local_path": str(destination),
                "width": width,
                "height": height,
                "format": image_format,
                "available_metadata": exif_keys,
                "exif_values": exif_values,
                "growth_stage": stage,
                "source": item["html_url"],
                "raw_rgb_frame": "true",
                "documented_flight_altitude": "6 m orthomosaic flight / 10 m DEM flight; frame-specific altitude not published",
                "camera_metadata": "Not published in repository; PNG EXIF inspected",
            })
            print(f"[rice] selected {stage} {date}/{item['name']} ({width}x{height})")

    results_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "date", "filename", "local_path", "width", "height", "format", "available_metadata", "exif_values",
        "growth_stage", "source", "raw_rgb_frame", "documented_flight_altitude", "camera_metadata",
    ]
    with (results_dir / "selected_images.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (results_dir / "selection_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[rice] wrote {len(rows)} selected images and {results_dir / 'selected_images.csv'}")


if __name__ == "__main__":
    main()
