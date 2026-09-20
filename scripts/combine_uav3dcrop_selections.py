from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine existing UAV3DCrop selection manifests")
    parser.add_argument("--inputs", nargs="+", default=["results/uav3dcrop/selected_images.csv", "results/uav3dcrop/multiscene_selected_images.csv"])
    parser.add_argument("--output", default="results/uav3dcrop/all_selected_images.csv")
    args = parser.parse_args()
    rows = []
    seen = set()
    for input_path in args.inputs:
        with (ROOT / input_path).open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                key = (row["scene"], row["filename"])
                if key not in seen:
                    rows.append(row)
                    seen.add(key)
    rows.sort(key=lambda row: (row["scene"], row["view_type"], int(row["frame_id"])))
    destination = ROOT / args.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[uav3dcrop] combined {len(rows)} unique images into {destination}")


if __name__ == "__main__":
    main()
