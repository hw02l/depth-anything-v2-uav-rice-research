"""Derive B3 from verified B0/B2 prediction caches; GT is used only afterwards for evaluation."""
import argparse,csv,json,time
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import tifffile
import torch
from uav_metric_research import multiscale as m
from uav_metric_research.fusion import global_local_fusion


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--tiles',type=Path,required=True)
    p.add_argument('--depth-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--radius',type=int,default=107)
    p.add_argument('--alpha',type=float,default=1.0)
    args=p.parse_args()
    torch.set_num_threads(2)
    baseline=json.loads((args.baseline/'launch.json').read_text())
    tiled=json.loads((args.tiles/'launch.json').read_text())
    if baseline['spec']['epoch']!=tiled['spec']['epoch'] or tiled['spec']['method']['kind']!='tile':
        raise ValueError('Aligned same-epoch B0 and B2 required')
    rows=m.validation_rows(m.ROOT/'configs/uav3dcrop_verified/validation.csv')
    indices=tiled['indices']
    args.output.mkdir(parents=True,exist_ok=False)
    method=dict(kind='fusion',parent_tile_method=tiled['spec']['method'],radius_original_px=args.radius,alpha=args.alpha,
                lowpass='normalized box, reflect101')
    launch=dict(phase=tiled['phase'],indices=indices,validation_sha256=tiled['validation_sha256'],
                spec=dict(label='B3',epoch=tiled['spec']['epoch'],method=method),
                source_sha256=m.sha256(m.ROOT/'src/uav_metric_research/fusion.py'),
                parents=dict(baseline_sha256=m.sha256(args.baseline/'launch.json'),tiles_sha256=m.sha256(args.tiles/'launch.json')))
    m.write_json(args.output/'launch.json',launch)
    baseline_timing={int(r['image_index']):r for r in csv.DictReader((args.baseline/'runtime_metrics.csv').open())}
    tile_timing={int(r['image_index']):r for r in csv.DictReader((args.tiles/'runtime_metrics.csv').open())}
    total=m.frozen('metrics').MetricAccumulator()
    per,local,runtimes,grouped,depth_groups=[],[],[],{},{}
    for i in indices:
        bm=json.loads((args.baseline/f'prediction_{i:03d}.json').read_text())
        qm=json.loads((args.tiles/f'prediction_{i:03d}.json').read_text())
        if bm['weights_sha256']!=qm['weights_sha256'] or bm['rgb_sha256']!=qm['rgb_sha256']:
            raise ValueError('B3 parent weights/RGB differ')
        b=m.load_prediction_cache(args.baseline/f'prediction_{i:03d}.npy',dict(method=baseline['spec']['method']))
        q=m.load_prediction_cache(args.tiles/f'prediction_{i:03d}.npy',dict(method=tiled['spec']['method']))
        started=time.monotonic()
        pred=global_local_fusion(b,q,args.radius,args.alpha)
        elapsed=time.monotonic()-started
        row=rows[i]
        # Only after deriving P, read GT for scoring. No GT value enters fusion.
        gt=np.asarray(tifffile.imread(args.depth_root/row['depth_relpath']),np.float32)
        m.validate_prediction(pred,gt)
        path=args.output/f'prediction_{i:03d}.npy'
        with path.open('xb') as stream:np.save(stream,pred,allow_pickle=False)
        m.write_json(path.with_suffix('.json'),dict(weights_sha256=bm['weights_sha256'],rgb_sha256=bm['rgb_sha256'],method=method,
            original_shape=list(gt.shape),prediction_dtype=str(pred.dtype),prediction_sha256=m.sha256(path),
            launch_sha256=m.sha256(args.output/'launch.json'),parents_sha256=[bm['prediction_sha256'],qm['prediction_sha256']]))
        absolute=m.frozen('metrics').MetricAccumulator()
        absolute.update(torch.from_numpy(pred),torch.from_numpy(np.where(m.valid_gt(gt),gt,0)))
        m.merge_depth(total,absolute)
        for scope in ('crop='+row['crop'],'view='+row['view'],row['crop']+'|'+row['view']):
            m.merge_depth(depth_groups.setdefault(scope,m.frozen('metrics').MetricAccumulator()),absolute)
        per.append(dict(image_index=i,image_relpath=row['image_relpath'],crop=row['crop'],view=row['view'],**absolute.result()))
        stats,_=m.multiscale(pred,gt,i)
        for stride,groups in stats.items():
            for group,acc in groups.items():
                local.append(dict(image_index=i,stride_px=stride,group=group,**acc.result()))
                for scope in ('all','crop='+row['crop'],'view='+row['view'],row['crop']+'|'+row['view']):
                    grouped.setdefault((scope,stride,group),m.DeltaStats()).merge(acc)
        rt0,rt2=baseline_timing[i],tile_timing[i]
        runtimes.append(dict(image_index=i,seconds=float(rt0['seconds'])+float(rt2['seconds'])+elapsed,fusion_seconds=elapsed,
            merge_seconds=float(rt2['merge_seconds'])+elapsed,peak_allocated_bytes=max(int(rt0['peak_allocated_bytes']),int(rt2['peak_allocated_bytes'])),
            runtime_kind='sum of measured parent times plus CPU fusion; no separate end-to-end GPU rerun'))
        print('FUSION',tiled['spec']['epoch'],i,elapsed,flush=True)
    m.write_csv(args.output/'per_image_metrics.csv',per)
    m.write_csv(args.output/'multiscale_per_image.csv',local)
    m.write_csv(args.output/'multiscale_metrics.csv',[dict(scope=k[0],stride_px=k[1],group=k[2],**v.result()) for k,v in sorted(grouped.items())])
    m.write_csv(args.output/'group_metrics.csv',[dict(scope=k,**v.result()) for k,v in sorted(depth_groups.items())])
    m.write_csv(args.output/'runtime_metrics.csv',runtimes)
    m.write_json(args.output/'summary.json',dict(images=len(indices),epoch=tiled['spec']['epoch'],method=method,metrics=total.result()))


if __name__=='__main__':main()
