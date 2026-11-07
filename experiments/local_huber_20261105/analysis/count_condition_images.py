#!/usr/bin/env python3
"""Count contributing images per frozen validation pair stratum from GT only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import tifffile

PAIR_SHA256 = "bcb56eab03bec4a44f0626da0f4814876fae67774e536e00cd9f1ad16b1f5728"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root", type=Path, required=True)
    ap.add_argument("--validation-manifest", type=Path, required=True)
    ap.add_argument("--pair-manifest", type=Path, required=True)
    ap.add_argument("--depth-root", type=Path, required=True)
    ap.add_argument("--condition-reference", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    project = args.project_root.resolve()
    sys.path.insert(0, str(project / "src"))
    from uav_metric_research import multiscale as m

    actual_hash = sha256(args.pair_manifest)
    if actual_hash != PAIR_SHA256:
        raise ValueError(f"Fixed pair manifest hash mismatch: {actual_hash}")
    rows = m.validation_rows(args.validation_manifest)
    pair_obj = json.loads(args.pair_manifest.read_text(encoding="utf-8"))
    compact = list(csv.DictReader(args.condition_reference.open("r", encoding="utf-8", newline="")))
    keys = sorted({(r["scope"], int(r["stride_px"]), r["group"]) for r in compact})
    reference = {(r["scope"], int(r["stride_px"]), r["group"]): int(r["pair_count"]) for r in compact if r["model"] == "E0"}
    sum_counts = {k: 0 for k in keys}
    image_counts = {k: 0 for k in keys}
    scope_total = {"all": len(rows)}
    for row in rows:
        scope_total[f"crop={row['crop'].lower()}"] = scope_total.get(f"crop={row['crop'].lower()}", 0) + 1
        scope_total[f"view={row['view'].lower()}"] = scope_total.get(f"view={row['view'].lower()}", 0) + 1
    for i, row in enumerate(rows):
        gt = np.asarray(tifffile.imread(args.depth_root / row["depth_relpath"]), dtype=np.float32)
        # A constant positive dummy prediction is used only to reuse the frozen GT pair
        # sampler/grouping. Prediction-derived statistics are ignored.
        stats, actual_coords = m.multiscale(np.ones(gt.shape, dtype=np.float32), gt, i)
        expected = [x for x in pair_obj["images"][i]["pairs"] if int(x["stride"]) in stats]
        if actual_coords != expected:
            raise ValueError(f"Fixed pair coordinate hash mismatch: image {i} {row['image_relpath']}")
        crop_scope, view_scope = f"crop={row['crop'].lower()}", f"view={row['view'].lower()}"
        for scope, stride, group in keys:
            if scope != "all" and scope not in (crop_scope, view_scope):
                continue
            value = stats[stride].get(group)
            count = int(value.legacy.total[0]) if value is not None else 0
            k = (scope, stride, group)
            sum_counts[k] += count
            if count > 0:
                image_counts[k] += 1
        if (i + 1) % 16 == 0 or i == len(rows) - 1:
            print(f"GT pair coverage: {i+1}/{len(rows)} images", flush=True)
        del gt, stats
    output_rows = []
    for scope, stride, group in keys:
        k = (scope, stride, group)
        expected_pairs = reference.get(k)
        actual_pairs = sum_counts[k]
        if expected_pairs is None or actual_pairs != expected_pairs:
            raise ValueError(f"Fixed pair count mismatch {k}: {actual_pairs} vs {expected_pairs}")
        output_rows.append({"scope": scope, "scope_image_count": scope_total.get(scope), "stride_px": stride,
                            "group": group, "image_count": image_counts[k], "pair_count": actual_pairs,
                            "images_with_zero_pairs": scope_total.get(scope, 0) - image_counts[k],
                            "pair_manifest_sha256": actual_hash, "verified_against_saved_compact_csv": True})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    with args.output.open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(output_rows[0]))
        writer.writeheader(); writer.writerows(output_rows)
    print(json.dumps({"images": len(rows), "strata": len(output_rows), "pair_manifest_sha256": actual_hash,
                      "output": str(args.output), "sha256": sha256(args.output)}, indent=2))


if __name__ == "__main__":
    main()
