"""Validate baseline evidence and compare completed inference caches on identical IDs."""
import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
from uav_metric_research import multiscale as m


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def gate(args):
    evidence, epochs, manifests = [], [], []
    for run in args.runs:
        launch = read(run/'launch.json')
        summary = read(run/'summary.json')
        reproduction = read(run/'baseline_reproduction.json')
        if launch['phase'] != 'baseline' or summary['images'] != 128 or not reproduction['passed']:
            raise ValueError('Full baseline reproduction did not pass')
        if summary['method'] != dict(kind='whole', short_side=518):
            raise ValueError('Wrong baseline')
        epochs.append(summary['epoch'])
        manifests.append(launch['validation_sha256'])
        path = run/'baseline_reproduction.json'
        evidence.append(dict(path=str(path.resolve()), sha256=m.sha256(path)))
    if sorted(epochs) != [17, 20] or len(set(manifests)) != 1:
        raise ValueError('Need both original epochs on one identical manifest')
    m.write_json(args.output, dict(passed=True, epochs=epochs, validation_sha256=manifests[0], evidence=evidence))


def compare(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import tifffile
    rows = m.validation_rows(args.manifest)
    launches = [read(path/'launch.json') for path in args.runs]
    summaries = [read(path/'summary.json') for path in args.runs]
    if len({r['epoch'] for r in summaries}) != 1 or len({r['validation_sha256'] for r in launches}) != 1:
        raise ValueError('Compare methods at the same epoch and manifest')
    indices = launches[1]['indices']
    if any(r['indices'] != indices for r in launches[1:]) or not set(indices) <= set(launches[0]['indices']):
        raise ValueError('All candidate image IDs must be identical; baseline must include all')
    if summaries[0]['method'] != dict(kind='whole', short_side=518):
        raise ValueError('First run must be B0')
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    per, grouped, absolute, runtime, paired = [], {}, [], [], []
    labels = [launch['spec'].get('label', path.name) for path, launch in zip(args.runs, launches)]
    if len(set(labels)) != len(labels):
        raise ValueError('Method labels must be unique')
    for path, label in zip(args.runs, labels):
        records = list(csv.DictReader((path/'per_image_metrics.csv').open(encoding='utf-8')))
        for record in records:
            if int(record['image_index']) in indices:
                absolute.append(dict(method=label, **record))
        runtime.extend(dict(method=label, **r) for r in csv.DictReader((path/'runtime_metrics.csv').open(encoding='utf-8')) if int(r['image_index']) in indices)
    for i in indices:
        row = rows[i]
        gt = np.asarray(tifffile.imread(args.depth_root/row['depth_relpath']), np.float32)
        predictions = {}
        for path, label, launch in zip(args.runs, labels, launches):
            pred = m.load_prediction_cache(path/f'prediction_{i:03d}.npy',
                      dict(method=launch['spec']['method'], launch_sha256=m.sha256(path/'launch.json')))
            marker = read(path/f'prediction_{i:03d}.json')
            baseline_marker = read(args.runs[0]/f'prediction_{i:03d}.json')
            if marker['weights_sha256'] != baseline_marker['weights_sha256']:
                raise ValueError('Method checkpoints differ')
            inventory = list(csv.DictReader((args.preparation/'image_inventory.csv').open(encoding='utf-8')))[i]
            if marker['rgb_sha256'] != inventory['rgb_sha256'] or m.sha256(args.depth_root/row['depth_relpath']) != inventory['gt_sha256']:
                raise ValueError('Cache/GT input digest mismatch')
            m.validate_prediction(pred, gt)
            predictions[label] = pred
            stats, _ = m.multiscale(pred, gt, i)
            for stride, groups in stats.items():
                for group, accumulator in groups.items():
                    per.append(dict(method=label, image_index=i, stride_px=stride, group=group, **accumulator.result()))
                    grouped.setdefault((label, stride, group), m.DeltaStats()).merge(accumulator)
        # Preselected image IDs: first pilot image in each crop/view, independent of outcome.
        pilot = list(csv.DictReader((args.preparation/'pilot_selection.csv').open(encoding='utf-8')))
        selected = min(int(r['image_index']) for r in pilot if r['crop'] == row['crop'] and r['view'] == row['view'])
        if i == selected:
            mask = m.valid_gt(gt)
            lo, hi = np.quantile(gt[mask], [.01, .99])
            arrays = dict(GT=gt, **predictions)
            fig, axes = plt.subplots(3, len(arrays), figsize=(4*len(arrays), 10), squeeze=False)
            for col, (label, pred) in enumerate(arrays.items()):
                im = axes[0,col].imshow(np.where(mask,pred,np.nan)[::8,::8], vmin=lo, vmax=hi, cmap='viridis')
                fig.colorbar(im, ax=axes[0,col], label='Z-depth (m)')
                err = np.abs(pred-gt)
                im = axes[1,col].imshow(np.where(mask,err,np.nan)[::8,::8], vmin=0, vmax=2, cmap='magma')
                fig.colorbar(im, ax=axes[1,col], label='Absolute error (m), clipped at 2')
                keep = mask[:,14:] & mask[:,:-14]
                delta_err = np.abs((pred[:,14:]-pred[:,:-14])-(gt[:,14:]-gt[:,:-14]))
                im = axes[2,col].imshow(np.where(keep,delta_err,np.nan)[::8,::8], vmin=0, vmax=.5, cmap='magma')
                fig.colorbar(im, ax=axes[2,col], label='14px delta error (m), clipped at 0.5')
                axes[0,col].set_title(label)
                for ax in axes[:,col]: ax.set_axis_off()
            fig.suptitle(f'{row["image_relpath"]} | {row["crop"]}/{row["view"]} | shared scales | Day047 validation')
            fig.tight_layout()
            fig.savefig(out/f'comparison_{i:03d}.png', dpi=130)
            plt.close(fig)
    m.write_csv(out/'per_image_multiscale.csv', per)
    m.write_csv(out/'method_comparison.csv', [dict(method=k[0], stride_px=k[1], group=k[2], images=len(indices), **v.result()) for k,v in sorted(grouped.items())])
    m.write_csv(out/'per_image_metrics.csv', absolute)
    m.write_csv(out/'runtime_metrics.csv', runtime)
    for label in labels[1:]:
        for stride in m.SCALES:
            for group in ('all','gt_ge20cm','border_lt128','interior_ge512'):
                values = {(r['method'],r['image_index']):r['delta_mae_m'] for r in per if r['stride_px'] == stride and r['group'] == group}
                if any(values[(label,i)] is None or values[(labels[0],i)] is None for i in indices):
                    paired.append(dict(method=label, stride_px=stride, group=group, status='empty group; CI not computed'))
                    continue
                diff = [values[(label,i)]-values[(labels[0],i)] for i in indices]
                paired.append(dict(method=label, stride_px=stride, group=group, **m.paired_bootstrap(diff)))
    m.write_json(out/'paired_differences.json', paired)
    m.write_json(out/'provenance.json', dict(runs=[str(p.resolve()) for p in args.runs], indices=indices, epoch=summaries[0]['epoch'],
                 scope='exploratory validation comparison; no independent test or height accuracy claim'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['gate','compare'])
    p.add_argument('--runs', nargs='+', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--manifest', type=Path, default=m.ROOT/'configs/uav3dcrop_verified/validation.csv')
    p.add_argument('--depth-root', type=Path, default=m.ROOT/'data/uav3dcrop_depth')
    p.add_argument('--preparation', type=Path, default=m.ROOT/'experiments/multiscale_tile_inference_20261016')
    args = p.parse_args()
    if len(args.runs) < 2:
        p.error('At least two run directories are needed')
    (gate if args.command == 'gate' else compare)(args)


if __name__ == '__main__':
    main()
