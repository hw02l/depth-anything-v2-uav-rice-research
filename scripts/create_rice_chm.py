from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path

import CSF
import laspy
import numpy as np
import open3d as o3d
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[1]


def fill_nearest(array: np.ndarray, valid: np.ndarray) -> np.ndarray:
    if not np.any(valid):
        raise ValueError("Cannot fill a raster with no valid cells")
    indices = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    filled = array[tuple(indices)]
    return filled.astype(np.float32)


def rasterize_stat(x: np.ndarray, y: np.ndarray, z: np.ndarray, left: float, top: float, width: int, height: int, resolution: float, statistic: str) -> np.ndarray:
    columns = np.floor((x - left) / resolution).astype(np.int64)
    rows = np.floor((top - y) / resolution).astype(np.int64)
    inside = (columns >= 0) & (columns < width) & (rows >= 0) & (rows < height) & np.isfinite(z)
    flat = rows[inside] * width + columns[inside]
    values = z[inside].astype(np.float64)
    size = width * height
    if statistic == "max":
        result = np.full(size, -np.inf, dtype=np.float64)
        np.maximum.at(result, flat, values)
        valid = np.isfinite(result) & (result > -np.inf)
    elif statistic == "min":
        result = np.full(size, np.inf, dtype=np.float64)
        np.minimum.at(result, flat, values)
        valid = np.isfinite(result) & (result < np.inf)
    else:
        sums = np.zeros(size, dtype=np.float64)
        counts = np.zeros(size, dtype=np.int32)
        np.add.at(sums, flat, values)
        np.add.at(counts, flat, 1)
        result = np.divide(sums, counts, out=np.full(size, np.nan), where=counts > 0)
        valid = counts > 0
    array = result.reshape(height, width).astype(np.float32)
    return fill_nearest(array, valid.reshape(height, width)), valid.reshape(height, width)


def write_raster(path: Path, array: np.ndarray, transform, crs, nodata: float = -9999.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = np.where(np.isfinite(array), array, nodata).astype(np.float32)
    with rasterio.open(path, "w", driver="GTiff", height=output.shape[0], width=output.shape[1], count=1, dtype="float32", crs=crs, transform=transform, nodata=nodata, compress="deflate") as dst:
        dst.write(output, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate AirMeasurer-style DSM/DEM/CHM from LAS")
    parser.add_argument("--selection-csv", default="results/airmeasurer/selected_dates.csv")
    parser.add_argument("--output-dir", default="outputs/airmeasurer/chm")
    parser.add_argument("--resolution", type=float, default=0.02, help="source CHM raster resolution in meters")
    parser.add_argument("--skip-sor", action="store_true")
    parser.add_argument("--cloth-resolution", type=float, default=0.10)
    args = parser.parse_args()
    with (ROOT / args.selection_csv).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    output_root = ROOT / args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)
    manifests = []
    for row in rows:
        date = row["date"]
        ortho_path = ROOT / row["orthomosaic_path"]
        las_path = ROOT / row["point_cloud_path"]
        with rasterio.open(ortho_path) as raster:
            left, bottom, right, top = raster.bounds
            crs = raster.crs
            ortho_transform = raster.transform
            ortho_width, ortho_height = raster.width, raster.height
        las = laspy.read(las_path)
        xyz = np.vstack((las.x, las.y, las.z)).T.astype(np.float64)
        original_count = len(xyz)
        if args.skip_sor:
            sor_indices = np.arange(original_count, dtype=np.int64)
        else:
            cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
            _, sor_indices_list = cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            sor_indices = np.asarray(sor_indices_list, dtype=np.int64)
            del cloud
        xyz = xyz[sor_indices]
        csf = CSF.CSF()
        csf.params.bSloopSmooth = False
        csf.params.cloth_resolution = args.cloth_resolution
        csf.params.rigidness = 2
        csf.params.class_threshold = 0.5
        csf.params.interations = 500
        csf.setPointCloud(xyz)
        ground_indices = CSF.VecInt()
        non_ground_indices = CSF.VecInt()
        csf.do_filtering(ground_indices, non_ground_indices)
        ground_indices = np.asarray(list(ground_indices), dtype=np.int64)
        ground = xyz[ground_indices]
        resolution = args.resolution
        width = int(np.ceil((right - left) / resolution))
        height = int(np.ceil((top - bottom) / resolution))
        transform = from_origin(left, top, resolution, resolution)
        dsm, dsm_observed = rasterize_stat(xyz[:, 0], xyz[:, 1], xyz[:, 2], left, top, width, height, resolution, "max")
        dem, dem_observed = rasterize_stat(ground[:, 0], ground[:, 1], ground[:, 2], left, top, width, height, resolution, "min")
        chm = np.maximum(dsm - dem, 0.0).astype(np.float32)
        source_valid = dsm_observed & dem_observed & np.isfinite(chm)
        source_chm = np.where(source_valid, chm, np.nan).astype(np.float32)
        date_dir = output_root / date
        date_dir.mkdir(parents=True, exist_ok=True)
        write_raster(date_dir / f"{date}_dsm_{resolution:g}m.tif", dsm, transform, crs)
        write_raster(date_dir / f"{date}_dem_{resolution:g}m.tif", dem, transform, crs)
        write_raster(date_dir / f"{date}_chm_{resolution:g}m.tif", source_chm, transform, crs)
        destination = np.full((ortho_height, ortho_width), -9999.0, dtype=np.float32)
        reproject(source=np.where(np.isfinite(source_chm), source_chm, -9999.0), destination=destination, src_transform=transform, src_crs=crs, src_nodata=-9999.0, dst_transform=ortho_transform, dst_crs=crs, dst_nodata=-9999.0, resampling=Resampling.bilinear)
        ortho_chm = np.where(destination > -9990, destination, np.nan).astype(np.float32)
        write_raster(date_dir / f"{date}_chm_on_orthomosaic_grid.tif", ortho_chm, ortho_transform, crs)
        write_raster(date_dir / f"{date}_valid_mask_on_orthomosaic_grid.tif", np.isfinite(ortho_chm).astype(np.float32), ortho_transform, crs, nodata=0.0)
        filtered = laspy.LasData(las.header)
        filtered.points = las.points[sor_indices]
        filtered.write(date_dir / f"{date}_sor_filtered.las")
        meta = {
            "date": date,
            "source_las": str(las_path),
            "source_orthomosaic": str(ortho_path),
            "source_point_count": original_count,
            "sor_point_count": int(len(xyz)),
            "csf_ground_point_count": int(len(ground)),
            "sor": {"enabled": not args.skip_sor, "nb_neighbors": 20, "std_ratio": 2.0},
            "csf": {"cloth_resolution_m": args.cloth_resolution, "rigidness": 2, "class_threshold_m": 0.5, "iterations": 500, "bSloopSmooth": False},
            "rasterization": {"source_resolution_m": resolution, "statistic_dsm": "max all SOR-filtered points", "statistic_dem": "min CSF ground points", "empty_cell_fill": "nearest observed cell", "chm": "max(DSM-DEM,0)"},
            "orthomosaic_grid": {"width": ortho_width, "height": ortho_height, "gsd_x_m": ortho_transform.a, "gsd_y_m": abs(ortho_transform.e), "resampling": "bilinear from georeferenced CHM grid; not a simple resize"},
            "crs": crs.to_string() if crs else None,
            "official_difference": "AirMeasurer uses SOR, CSF, GCP terrain correction, WhiteboxTools LidarTinGridding, ROI/plot masks, and perspective transform. This implementation uses the official SOR/CSF concepts but custom NumPy rasterization and nearest-cell fill; exact Whitebox output is not claimed.",
        }
        (date_dir / "chm_manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        manifests.append(meta)
        print(f"[airmeasurer] {date}: source={original_count}, SOR={len(xyz)}, CSF ground={len(ground)}, CHM grid={width}x{height}")
        del xyz, ground, dsm, dem, chm, source_chm, ortho_chm, las, csf
        gc.collect()
    (output_root / "run_manifest.json").write_text(json.dumps({"dates": manifests}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
