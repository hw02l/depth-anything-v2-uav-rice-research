from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from da2_experiments.io import list_images, read_rgb, save_depth_outputs, write_json  # noqa: E402
from da2_experiments.modeling import RELATIVE_MODEL_ID, load_estimator  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Depth Anything V2 relative-depth inference")
    parser.add_argument("--input", required=True, help="Image file or directory")
    parser.add_argument("--outdir", default="outputs/relative")
    parser.add_argument("--model-id", default=RELATIVE_MODEL_ID)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    estimator = load_estimator("relative", args.model_id, args.device)
    records = []
    for path in list_images(args.input, args.limit):
        depth = estimator.predict(read_rgb(path))
        saved = save_depth_outputs(depth, path.stem, args.outdir, "relative_depth")
        records.append({"input": str(path), **saved, "shape": list(depth.shape)})
        print(f"[relative] {path.name} -> {saved['visualization']}")
    write_json(Path(args.outdir) / "run.json", {"kind": "relative", "model_id": args.model_id, "device": str(estimator.device), "records": records})


if __name__ == "__main__":
    main()
