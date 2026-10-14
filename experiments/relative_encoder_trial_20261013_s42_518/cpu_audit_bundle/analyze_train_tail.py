"""Supplement the existing sampled GT report with train-only full pixel tails."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import tifffile
from data import read_manifest
from runtime_state import preparation_context


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--depth-root', type=Path, required=True)
    p.add_argument('--gt-report', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    gt = json.loads(a.gt_report.read_text())
    sha = hashlib.sha256(a.manifest.read_bytes()).hexdigest()
    assert gt['status'] == 'pass' and gt['source_manifest_sha256'] == sha
    rows = read_manifest(a.manifest)
    assert len(rows) == 384 and all(r['split'] == 'train' for r in rows)
    assert not any('Day001_' in r['scene'] or 'Day007_' in r['scene'] for r in rows)
    context_args = dict(manifests={'train': a.manifest}, rgb_root=None,
                        depth_root=a.depth_root, physical_splits=['train'], kind='gt',
                        parameters={'seed': gt['seed'], 'max_pixels_per_frame': 20000})
    before = preparation_context(**context_args)
    cap = float(gt['suggested_max_depth_m'])
    thresholds = sorted(set([cap * .9, cap, cap * 1.1, gt['quantiles_m']['p99.9']]))
    edges = np.array([0, .5, 1, 2, 5, 10, 15, 20, 25, 30, 40, 50,
                      75, 100, 150, 200, 500, 1000, np.inf])
    histogram = np.zeros(len(edges)-1, dtype=np.int64)
    valid_total = pixel_total = nonfinite_total = nonpositive_total = 0
    minimum, maximum = float('inf'), 0.
    above = {str(t): 0 for t in thresholds}
    scenes, frames = {}, []
    for i, r in enumerate(rows, 1):
        depth = np.asarray(tifffile.imread(a.depth_root/r['depth_relpath']))
        finite = np.isfinite(depth)
        values = depth[finite & (depth > 0)].astype(np.float32)
        assert values.size
        lo, hi = float(values.min()), float(values.max())
        minimum, maximum = min(minimum, lo), max(maximum, hi)
        valid_total += int(values.size)
        pixel_total += int(depth.size)
        nonfinite_total += int((~finite).sum())
        nonpositive_total += int((finite & (depth <= 0)).sum())
        histogram += np.histogram(values, bins=edges)[0]
        frame_above = {}
        for t in thresholds:
            count = int(np.count_nonzero(values > t))
            above[str(t)] += count
            frame_above[str(t)] = count
        scene = scenes.setdefault(r['scene'], {'frames':0, 'valid_pixels':0,
                    'min_m':lo, 'max_m':hi, 'above_suggested_cap_pixels':0})
        scene['frames'] += 1
        scene['valid_pixels'] += int(values.size)
        scene['min_m'], scene['max_m'] = min(scene['min_m'],lo), max(scene['max_m'],hi)
        scene['above_suggested_cap_pixels'] += frame_above[str(cap)]
        frames.append({'scene':r['scene'], 'crop':r['crop'], 'view':r['view'],
                       'path':r['depth_relpath'], 'min_m':lo, 'max_m':hi,
                       'valid_pixels':int(values.size), 'above_cap_pixels':frame_above[str(cap)]})
        if i % 32 == 0:
            print(f'Full train pixel scan {i}/{len(rows)}', flush=True)
    after = preparation_context(**context_args)
    assert after == before, 'Data or analysis context changed during scan'
    result = {'status':'pass', 'split':'train', 'frames':len(rows),
              'manifest_sha256':sha, 'gt_report':str(a.gt_report),
              'gt_report_sha256':hashlib.sha256(a.gt_report.read_bytes()).hexdigest(),
              'analysis_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'valid_rule':'finite and >0; no range clipping',
              'full_pixel_min_m':minimum, 'full_pixel_max_m':maximum,
              'total_pixels':pixel_total, 'valid_pixels':valid_total,
              'nonfinite_pixels':nonfinite_total, 'nonpositive_finite_pixels':nonpositive_total,
              'sample_quantiles_m':gt['quantiles_m'], 'suggested_cap_m':cap,
              'above_threshold':{t:{'pixels':n,'fraction_of_valid':n/valid_total}
                                 for t,n in above.items()},
              'histogram':{'edges_m':[float(x) if np.isfinite(x) else 'Infinity' for x in edges],
                           'valid_pixel_counts':histogram.tolist()},
              'per_scene':scenes,'per_frame':frames,'reuse_context':after}
    with a.output.open('x', encoding='utf-8') as f:
        json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ('per_frame','reuse_context')},indent=2))


if __name__ == '__main__':
    main()
