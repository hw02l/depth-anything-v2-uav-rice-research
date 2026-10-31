"""Multiscale diagnostics from immutable B0 caches, after both baseline gates pass."""
import argparse,csv,json,time
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import tifffile
from uav_metric_research import multiscale as m


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',nargs='+',type=Path,required=True)
    p.add_argument('--baseline-gate',type=Path,required=True)
    p.add_argument('--depth-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    gate=json.loads(args.baseline_gate.read_text())
    if not gate['passed'] or sorted(gate['epochs'])!=[17,20]:raise ValueError('Missing baseline gate')
    rows=m.validation_rows(m.ROOT/'configs/uav3dcrop_verified/validation.csv')
    preparation=m.ROOT/'experiments/multiscale_tile_inference_20261016'
    inventory=list(csv.DictReader((preparation/'image_inventory.csv').open()))
    expected={r['image_index']:r['pairs'] for r in json.loads((preparation/'pair_manifest.json').read_text())['images']}
    aggregate,per,pairs={},[],[]
    args.output.mkdir(parents=True,exist_ok=False)
    for run in args.runs:
        launch=json.loads((run/'launch.json').read_text())
        summary=json.loads((run/'summary.json').read_text())
        if summary['method']!=dict(kind='whole',short_side=518) or summary['images']!=128:
            raise ValueError('A requires complete B0 predictions')
        label=f'epoch{summary["epoch"]}'
        for i,row in enumerate(rows):
            pred=m.load_prediction_cache(run/f'prediction_{i:03d}.npy',dict(method=summary['method'],launch_sha256=m.sha256(run/'launch.json')))
            gt=np.asarray(tifffile.imread(args.depth_root/row['depth_relpath']),np.float32)
            if m.sha256(args.depth_root/row['depth_relpath'])!=inventory[i]['gt_sha256']:
                raise ValueError('GT digest mismatch')
            stats,coordinates=m.multiscale(pred,gt,i)
            if coordinates!=expected[i]:raise ValueError('Evaluation pair manifest mismatch')
            pairs.append(dict(model=label,image_index=i,pairs=coordinates))
            for stride,groups in stats.items():
                for group,acc in groups.items():
                    per.append(dict(model=label,image_index=i,crop=row['crop'],view=row['view'],stride_px=stride,group=group,**acc.result()))
                    for scope in ('all','crop='+row['crop'],'view='+row['view'],row['crop']+'|'+row['view']):
                        aggregate.setdefault((label,scope,stride,group),m.DeltaStats()).merge(acc)
            if (i+1)%16==0:print('MULTISCALE_DIAGNOSTIC',label,i+1,'/128',flush=True)
    m.write_csv(args.output/'multiscale_metrics.csv',[dict(model=k[0],scope=k[1],stride_px=k[2],group=k[3],**v.result()) for k,v in sorted(aggregate.items())])
    m.write_csv(args.output/'per_image_metrics.csv',per)
    m.write_json(args.output/'pair_manifest.json',pairs)
    m.write_json(args.output/'provenance.json',dict(baseline_gate_sha256=m.sha256(args.baseline_gate),runs=[str(r) for r in args.runs],
                 weights_updated=False,independent_test_used=False,historical52_used=False))


if __name__=='__main__':main()
