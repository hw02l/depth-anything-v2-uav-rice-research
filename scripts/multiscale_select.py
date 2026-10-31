"""Prespecified pilot-only selection; never reads full-comparison results or test data."""
import argparse,csv,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from uav_metric_research import multiscale as m


def read_csv(path):return list(csv.DictReader(path.open(encoding='utf-8')))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--preparation',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    config=json.loads(args.config.read_text())
    indices={int(r['image_index']) for r in read_csv(args.preparation/'pilot_selection.csv')}
    ranked,evidence,baseline={},[],{}
    for epoch in (17,20):
        rows=[r for r in read_csv(args.results/f'B0_ep{epoch}'/'per_image_metrics.csv') if int(r['image_index']) in indices]
        n=sum(int(r['valid_pixels']) for r in rows)
        baseline[epoch]=(sum(int(r['valid_pixels'])*float(r['rmse_m'])**2 for r in rows)/n)**.5
    for item in config['pilot_candidates_before_results']:
        label=item['label']
        scores,absolute,times,overlap=[],[],[],[]
        for epoch in (17,20):
            run=args.results/f'{label}_ep{epoch}'
            launch=json.loads((run/'launch.json').read_text())
            if set(launch['indices'])!=indices or launch['phase']!='pilot':raise ValueError('Wrong pilot image IDs')
            local=[r for r in read_csv(run/'multiscale_per_image.csv') if r['stride_px']=='14' and r['group']=='gt_ge20cm']
            if {int(r['image_index']) for r in local}!=indices or any(int(r['pair_count'])==0 for r in local):
                raise ValueError('Incomplete/empty pilot cases; no silent exclusions')
            scores.extend(float(r['delta_mae_m']) for r in local)
            summary=json.loads((run/'summary.json').read_text())
            absolute.append(summary['metrics']['rmse_m'])
            runtime=read_csv(run/'runtime_metrics.csv')
            times.extend(float(r['seconds']) for r in runtime)
            overlap.extend(float(r['mean_overlap_disagreement_m']) for r in runtime if r.get('mean_overlap_disagreement_m'))
            evidence.append(dict(path=str((run/'summary.json').resolve()),sha256=m.sha256(run/'summary.json')))
        ranked[label]=dict(label=label,joint_image_mean_large14_mae_m=sum(scores)/len(scores),
                          rmse_ep17_m=absolute[0],rmse_ep20_m=absolute[1],
                          mean_seconds=sum(times)/len(times),mean_overlap_disagreement_m=sum(overlap)/len(overlap) if overlap else None)
    tiles=sorted([v for k,v in ranked.items() if k.startswith('B2_')],
                 key=lambda r:(r['joint_image_mean_large14_mae_m'],(r['rmse_ep17_m']+r['rmse_ep20_m'])/2,r['mean_seconds'],r['label']))
    selected=tiles[0]
    method=next({k:v for k,v in c.items() if k!='label'} for c in config['pilot_candidates_before_results'] if c['label']==selected['label'])
    trigger=(selected['rmse_ep17_m']>1.5*baseline[17] or selected['rmse_ep20_m']>1.5*baseline[20]
             or (selected['mean_overlap_disagreement_m'] or 0)>1)
    m.write_json(args.output,dict(selected_tile_label=selected['label'],selected_tile_method=method,
        unique_pilot_images=len(indices),checkpoint_image_cases=2*len(indices),ranked=ranked,
        baseline_pilot_rmse=baseline,run_B3=trigger,B3=config['B3'],
        selection_rule='mean over 16 images of mean epoch17/20 GT>=20cm 14px MAE; RMSE then time then label break exact ties',
        pilot_selection_sha256=m.sha256(args.preparation/'pilot_selection.csv'),evidence=evidence,
        scope='exploratory Day047 validation selection; checkpoint cases are not independent images'))
    print('PILOT_SELECTION',args.output.read_text(),flush=True)


if __name__=='__main__':main()
