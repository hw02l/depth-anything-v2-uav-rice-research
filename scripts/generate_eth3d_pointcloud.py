from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from da2_experiments.io import load_depth, read_rgb, save_depth_outputs, write_json  # noqa: E402
from da2_experiments.modeling import METRIC_OUTDOOR_MODEL_ID, load_estimator  # noqa: E402
from da2_experiments.pointcloud import depth_to_points, write_ascii_ply  # noqa: E402


def read_eth3d_calib(path: str | Path) -> dict[str, float]:
    """Read the cam0 matrix from ETH3D's low-res two-view calib.txt."""
    text = Path(path).read_text(encoding="utf-8")
    match = re.search(r"cam0=\[([^;]+);\s*([^;]+);\s*([^\]]+)\]", text)
    if not match:
        raise ValueError(f"Could not find cam0 matrix in {path}")
    row0 = [float(value) for value in match.group(1).split()]
    row1 = [float(value) for value in match.group(2).split()]
    return {"fx": row0[0], "fy": row1[1], "cx": row0[2], "cy": row1[2]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an RGB colored point cloud from ETH3D RGB + depth")
    parser.add_argument("--rgb", required=True)
    parser.add_argument("--depth", help="Depth .npy or image. If omitted, infer with Depth Anything V2 metric model.")
    parser.add_argument("--depth-png-scale", type=float, default=1.0)
    parser.add_argument("--model-id", default=METRIC_OUTDOOR_MODEL_ID)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--outdir", default="outputs/eth3d")
    parser.add_argument("--fx", type=float)
    parser.add_argument("--fy", type=float)
    parser.add_argument("--cx", type=float)
    parser.add_argument("--cy", type=float)
    parser.add_argument("--intrinsics-json", help="JSON with fx, fy, cx, cy")
    parser.add_argument("--calib-txt", help="ETH3D calib.txt; cam0 matrix is read automatically")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--max-depth", type=float)
    args = parser.parse_args()

    rgb_image = read_rgb(args.rgb)
    rgb = np.asarray(rgb_image)
    height, width = rgb.shape[:2]
    intrinsics = {"fx": args.fx, "fy": args.fy, "cx": args.cx, "cy": args.cy}
    if args.calib_txt:
        intrinsics.update(read_eth3d_calib(args.calib_txt))
    if args.intrinsics_json:
        intrinsics.update(json.loads(Path(args.intrinsics_json).read_text(encoding="utf-8")))
    intrinsics["fx"] = intrinsics["fx"] or 0.9 * width
    intrinsics["fy"] = intrinsics["fy"] or 0.9 * width
    intrinsics["cx"] = intrinsics["cx"] if intrinsics["cx"] is not None else width / 2.0
    intrinsics["cy"] = intrinsics["cy"] if intrinsics["cy"] is not None else height / 2.0

    if args.depth:
        depth_path = Path(args.depth)
        depth = load_depth(depth_path, png_scale=args.depth_png_scale)
        depth_source = str(depth_path)
    else:
        estimator = load_estimator("metric", args.model_id, args.device)
        depth = estimator.predict(rgb_image)
        depth_source = args.model_id

    outdir = Path(args.outdir)
    depth_saved = save_depth_outputs(depth, Path(args.rgb).stem, outdir, "depth_for_pointcloud")
    points, colors = depth_to_points(depth, rgb, **intrinsics, stride=max(1, args.stride), max_depth=args.max_depth)
    ply_path = outdir / f"{Path(args.rgb).stem}.ply"
    write_ascii_ply(ply_path, points, colors)
    stats = {
        "rgb": str(args.rgb),
        "depth_source": depth_source,
        "ply": str(ply_path),
        "points": int(len(points)),
        "intrinsics": intrinsics,
        "stride": args.stride,
        "max_depth_m": args.max_depth,
        "depth_min_m": float(np.nanmin(depth)),
        "depth_max_m": float(np.nanmax(depth)),
        "depth_outputs": depth_saved,
    }
    write_json(outdir / f"{Path(args.rgb).stem}_pointcloud.json", stats)
    print(f"[eth3d] wrote {ply_path} with {len(points)} points")


if __name__ == "__main__":
    main()
