from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import laspy
import rasterio

ROOT = Path(__file__).resolve().parents[1]


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit locally available AirMeasurer orthomosaic/LAS pairs")
    parser.add_argument("--data-root", default="data/airmeasurer/small_field")
    parser.add_argument("--results-dir", default="results/airmeasurer")
    args = parser.parse_args()
    data_root = ROOT / args.data_root
    ortho_files = {path.stem: path for path in data_root.rglob("*.tif")}
    las_files = {path.stem: path for path in data_root.rglob("*.las")}
    rows = []
    manifest = {"data_root": str(data_root), "available_pairs": [], "plot_information": "not found; RTK shapefile is point control information"}
    for date in sorted(set(ortho_files) & set(las_files)):
        ortho = ortho_files[date]
        las_path = las_files[date]
        with rasterio.open(ortho) as raster:
            crs = raster.crs.to_string() if raster.crs else None
            raster_info = {
                "width": raster.width,
                "height": raster.height,
                "bands": raster.count,
                "dtypes": list(raster.dtypes),
                "crs": crs,
                "gsd_x_m": raster.res[0],
                "gsd_y_m": raster.res[1],
                "bounds": list(raster.bounds),
                "transform": list(raster.transform),
                "software": raster.tags().get("TIFFTAG_SOFTWARE"),
            }
        las = laspy.read(las_path)
        las_crs = None
        try:
            las_crs = las.header.parse_crs().to_string()
        except Exception:
            las_crs = None
        row = {
            "date": date,
            "orthomosaic_filename": ortho.name,
            "point_cloud_filename": las_path.name,
            "orthomosaic_path": rel(ortho),
            "point_cloud_path": rel(las_path),
            "point_count": len(las.points),
            "raster_width": raster_info["width"],
            "raster_height": raster_info["height"],
            "raster_bands": raster_info["bands"],
            "crs": crs or las_crs or "unknown",
            "las_crs": las_crs or "not parsed",
            "gsd_x_m": raster_info["gsd_x_m"],
            "gsd_y_m": raster_info["gsd_y_m"],
            "available_plot_information": "none; accompanying small_mtps shapefile contains 4 PointZ RTK control points",
            "source": "The-Zhou-Lab/UAV-AirMeasurer V2.0.2 AirMeasurer_testdata_small_field.zip",
        }
        rows.append(row)
        manifest["available_pairs"].append({**row, "raster": raster_info, "las_bounds": [float(v) for v in las.header.mins.tolist() + las.header.maxs.tolist()]})
    if not rows:
        raise FileNotFoundError(f"No matched .tif/.las date pairs under {data_root}")
    output_dir = ROOT / args.results_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "selected_dates.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[airmeasurer] found {len(rows)} matched date pair(s); wrote {output_dir / 'selected_dates.csv'}")


if __name__ == "__main__":
    main()
