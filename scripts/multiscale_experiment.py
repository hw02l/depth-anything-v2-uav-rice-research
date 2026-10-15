"""Inference-only experiment CLI. Run from the Windows canonical repository copy.

prepare reads only Day047 GT/RGB; baseline/pilot/full require an audited GPU receipt.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
from uav_metric_research import multiscale as m

EXPERIMENT = m.ROOT / 'experiments/multiscale_tile_inference_20261016'


def prepare(args):
    import cv2
    import tifffile
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'manifest.json').exists():
        raise ValueError('Preparation already exists; use a new output directory')
    rows = m.validation_rows(args.manifest)
    inventory, zero_rows, pair_manifest, aggregate = [], [], [], {}
    zero_count, zero_sum = 0, 0.0
    for i, row in enumerate(rows):
        image = Path(args.rgb_root) / row['image_relpath']
        depth = Path(args.depth_root) / row['depth_relpath']
        gt = np.asarray(tifffile.imread(depth), np.float32)
        rgb = cv2.imread(str(image))
        if rgb is None or rgb.shape[:2] != gt.shape:
            raise ValueError('RGB/GT shape mismatch: ' + row['image_relpath'])
        del rgb
        mask = m.valid_gt(gt)
        # GT-only selection, before any predictions. Approximate depth quantiles
        # use deterministic spatial subsampling; valid pixel count uses full GT.
        sampled = gt[::16, ::16]
        sampled = sampled[m.valid_gt(sampled)]
        quantiles = np.quantile(sampled, [.1, .5, .9])
        h, w = gt.shape
        rh, rw = m.resized_shape(h, w)
        inventory.append(dict(image_index=i, image_relpath=row['image_relpath'], depth_relpath=row['depth_relpath'],
                              crop=row['crop'], view=row['view'], day=row['day'], height=h, width=w,
                              resized_height=rh, resized_width=rw, scale_y=rh/h, scale_x=rw/w,
                              input14_y_px=14*rh/h, input14_x_px=14*rw/w,
                              valid_pixels=int(mask.sum()), gt_p10_m=float(quantiles[0]),
                              gt_median_m=float(quantiles[1]), gt_p90_m=float(quantiles[2]),
                              rgb_sha256=m.sha256(image), gt_sha256=m.sha256(depth)))
        # Only GT differences are assessed: constant positive depth gives zero delta.
        stats, pairs = m.multiscale(np.full(gt.shape, 20, np.float32), gt, i)
        for stride, groups in stats.items():
            for group, accumulator in groups.items():
                item = dict(image_index=i, image_relpath=row['image_relpath'], crop=row['crop'], view=row['view'],
                            method='GT_only_zero_delta_reference', stride_px=stride, group=group, **accumulator.result())
                zero_rows.append(item)
                for scope in ('all', 'crop=' + row['crop'], 'view=' + row['view'], row['crop'] + '|' + row['view']):
                    aggregate.setdefault((scope, stride, group), m.DeltaStats()).merge(accumulator)
        pair_manifest.append(dict(image_index=i, shape=list(gt.shape), pairs=pairs))
        # Exact valid horizontal/vertical 14px pairs, bounded row chunks.
        for axis in (0, 1):
            limit = h - 14 if axis == 0 else h
            for top in range(0, limit, 128):
                bottom = min(limit, top + 128)
                a = gt[top:bottom, :] if axis == 0 else gt[top:bottom, :-14]
                b = gt[top+14:bottom+14, :] if axis == 0 else gt[top:bottom, 14:]
                keep = m.valid_gt(a) & m.valid_gt(b)
                zero_count += int(keep.sum())
                zero_sum += float(np.abs(b-a)[keep].sum(dtype=np.float64))
        print(f'GT_PREPARE {i+1}/128 {row["crop"]}/{row["view"]}', flush=True)
    pilot = []
    for key in sorted({(r['crop'], r['view']) for r in inventory}):
        group = sorted([r for r in inventory if (r['crop'], r['view']) == key], key=lambda r: (r['gt_median_m'], r['image_relpath']))
        for fraction in (.25, .75):
            position = min(len(group)-1, int(len(group)*fraction))
            pilot.append(dict(group[position], selection_rule=f'GT median rank floor(n*{fraction}); spatial GT sample step16'))
    if len({r['image_index'] for r in pilot}) != len(pilot):
        raise ValueError('Pilot duplicate IDs; inspect sparse groups')
    m.write_csv(out / 'image_inventory.csv', inventory)
    m.write_csv(out / 'pilot_selection.csv', sorted(pilot, key=lambda r: r['image_index']))
    m.write_csv(out / 'gt_reference_per_image.csv', zero_rows)
    group_rows = []
    for (scope, stride, group), accumulator in sorted(aggregate.items()):
        relevant = [r for r in inventory if scope == 'all' or scope == 'crop='+r['crop'] or scope == 'view='+r['view'] or scope == r['crop']+'|'+r['view']]
        group_rows.append(dict(scope=scope, images=len(relevant), valid_pixels=sum(r['valid_pixels'] for r in relevant),
                               stride_px=stride, group=group, **accumulator.result()))
    m.write_csv(out / 'gt_reference_groups.csv', group_rows)
    m.write_json(out / 'pair_manifest.json', dict(seed_rule='42+original_image_index*10000+stride',
                    candidates_per_axis=100000, numpy_version=np.__version__, coordinate_format='little-endian int32 N x [y,x,yy,xx]', images=pair_manifest))
    m.write_json(out / 'manifest.json', dict(validation_sha256=m.sha256(args.manifest), images=128,
                    pilot_images=len(pilot), independent_test_used=False, historical52_used=False,
                    full14_zero_delta=dict(valid_pairs=zero_count, local_mae_m=zero_sum/zero_count),
                    input_inventory_sha256=m.sha256(out/'image_inventory.csv'),
                    pair_manifest_sha256=m.sha256(out/'pair_manifest.json'),
                    pilot_selection_sha256=m.sha256(out/'pilot_selection.csv')))


def strict_model(checkpoint, epoch):
    import torch
    import transformers
    from safetensors.torch import load_file
    from uav_metric_research.provenance import verify_checkpoint
    path = Path(checkpoint)
    if epoch not in (17, 18, 20):
        raise ValueError('Only the requested epoch17/18/20 checkpoints are allowed')
    verified = verify_checkpoint(path)
    if verified['completed_epoch'] != epoch:
        raise ValueError('Checkpoint epoch differs')
    prior = json.loads((m.RUN / 'gpgpu4_completion_20261015/provenance.json').read_text(encoding='utf-8'))
    key = 'epoch_17_best_rmse' if epoch == 17 else 'epoch_20'
    if epoch in (17, 20):
        for name, digest in prior['checkpoint_hashes'][key].items():
            if m.sha256(path/name) != digest:
                raise ValueError('Checkpoint does not match completion provenance: ' + name)
    # Audit the serialized/runtime key mapping explicitly; never ignore keys.
    if torch.__version__ != '2.11.0+cu126' or transformers.__version__ != '5.18.0':
        raise ValueError('Unaudited runtime: use the recorded GPGPU4 container/private site')
    config = transformers.AutoConfig.from_pretrained(path, local_files_only=True)
    if config.max_depth != 80 or config.depth_estimation_type != 'metric' or config.backbone_config.patch_size != 14:
        raise ValueError('Wrong metric model config')
    model = transformers.AutoModelForDepthEstimation.from_config(config)
    weights = load_file(str(path/'model.safetensors'), device='cpu')
    model.inference_load_audit = m.strict_attention_state(model, weights)
    del weights
    model.requires_grad_(False).eval().cuda()
    if model.head.max_depth != 80:
        raise ValueError('Head cap mismatch')
    return model


def prediction(model, rgb, method):
    import torch
    def infer(image, side):
        tensor = m.preprocess(image, side)[None].cuda()
        with torch.inference_mode():
            native = model(pixel_values=tensor).predicted_depth[0].float().cpu().numpy()
        return m.restore(native, image.shape[:2])
    if method['kind'] == 'whole':
        return infer(rgb, method['short_side']), dict(tiles=1, merge_seconds=0)
    if method['kind'] != 'tile':
        raise ValueError('Unknown inference kind')
    merge = m.TileMerger(rgb.shape[:2])
    merge_seconds, overlaps = 0.0, []
    boxes = m.tiles(rgb.shape[:2], method['tile_size'], method['overlap'])
    for box in boxes:
        y, x, bottom, right = box
        tile = infer(rgb[y:bottom, x:right], method['short_side'])
        start = time.monotonic()
        # GT-free overlap disagreement, before blending (descriptive order-dependent).
        weight = merge.weight[y:bottom, x:right]
        select = weight > 0
        if select.any():
            current = merge.total[y:bottom, x:right][select] / weight[select]
            overlaps.append(float(np.abs(current - tile[select]).mean()))
        merge.add(box, tile)
        merge_seconds += time.monotonic() - start
    start = time.monotonic()
    result = merge.result()
    merge_seconds += time.monotonic() - start
    return result, dict(tiles=len(boxes), merge_seconds=merge_seconds,
                        mean_overlap_disagreement_m=float(np.mean(overlaps)) if overlaps else None)


def run(args):
    import cv2
    import tifffile
    import torch
    spec = json.loads(Path(args.spec).read_text(encoding='utf-8-sig'))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'launch.json').exists():
        raise ValueError('Use a new execution directory; existing results are immutable')
    rows = m.validation_rows(args.manifest)
    receipt = json.loads(Path(args.gpu_receipt).read_text(encoding='utf-8-sig'))
    import socket
    if socket.gethostname().lower() != 'gpgpu4' or not torch.cuda.is_available():
        raise ValueError('Run only on the audited GPGPU4 GPU')
    if receipt.get('status') != 'available' or time.time() - receipt['unix_time'] > 300:
        raise ValueError('A fresh GPU preflight receipt is mandatory')
    local = Path.home() / 'gpu_work'
    if not str(local.resolve()).startswith('/localdata/'):
        raise ValueError('gpu_work is not localdata')
    for path in [out, Path(args.rgb_root), Path(args.depth_root), Path(spec['checkpoint']), Path(args.manifest)]:
        if not path.resolve().is_relative_to(local.resolve()):
            raise ValueError('All execution assets must reside on localdata')
    preparation = Path(args.preparation)
    manifest = json.loads((preparation/'manifest.json').read_text(encoding='utf-8'))
    for name, key in [('image_inventory.csv', 'input_inventory_sha256'), ('pair_manifest.json', 'pair_manifest_sha256'), ('pilot_selection.csv', 'pilot_selection_sha256')]:
        if m.sha256(preparation/name) != manifest[key]:
            raise ValueError('Preparation hash mismatch: ' + name)
    if manifest['validation_sha256'] != m.sha256(args.manifest):
        raise ValueError('Preparation validation manifest mismatch')
    inventory = {int(r['image_index']): r for r in csv.DictReader((preparation/'image_inventory.csv').open(encoding='utf-8'))}
    expected_pairs = {r['image_index']:r['pairs'] for r in json.loads((preparation/'pair_manifest.json').read_text(encoding='utf-8'))['images']}
    indices = list(range(128))
    if args.phase == 'pilot':
        indices = sorted(int(r['image_index']) for r in csv.DictReader((preparation/'pilot_selection.csv').open(encoding='utf-8')))
    if args.phase != 'baseline':
        gate = json.loads(Path(args.baseline_gate).read_text(encoding='utf-8'))
        if not gate.get('passed') or gate.get('validation_sha256') != manifest['validation_sha256']:
            raise ValueError('Both epoch17 and epoch20 full baseline reproductions must pass first')
        if sorted(gate.get('epochs', [])) != [17, 20]:
            raise ValueError('Both epochs are required in the gate')
        for evidence in gate['evidence']:
            if m.sha256(evidence['path']) != evidence['sha256']:
                raise ValueError('Baseline evidence changed')
            proof = json.loads(Path(evidence['path']).read_text(encoding='utf-8'))
            if not proof['passed']:
                raise ValueError('Failed baseline evidence')
    if args.phase == 'full':
        lock = json.loads(Path(args.pilot_lock).read_text(encoding='utf-8'))
        if lock['method'] != spec['method'] or lock['pilot_selection_sha256'] != manifest['pilot_selection_sha256']:
            raise ValueError('Full-validation method was not fixed from the selected pilot')
        if not lock.get('pilot_result_sha256') or not lock.get('selection_rationale'):
            raise ValueError('Pilot lock needs measured results and selection rationale')
        if m.sha256(lock['pilot_result_path']) != lock['pilot_result_sha256']:
            raise ValueError('Pilot selection evidence changed')
        selection = json.loads(Path(lock['pilot_result_path']).read_text(encoding='utf-8'))
        for evidence in selection['evidence']:
            if m.sha256(evidence['path']) != evidence['sha256']:
                raise ValueError('Pilot result evidence changed')
    method = spec['method']
    if args.phase == 'baseline' and method != dict(kind='whole', short_side=518):
        raise ValueError('Baseline must use the exact B0 method')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    import subprocess
    busy = subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    if busy.stdout.strip():
        raise ValueError('GPU compute processes appeared after preflight; stop')
    model = strict_model(spec['checkpoint'], spec['epoch'])
    m.write_json(out/'launch.json', dict(spec=spec, spec_sha256=m.sha256(args.spec), phase=args.phase,
                 validation_sha256=m.sha256(args.manifest), indices=indices, fp32=True,
                 runtime=dict(torch=torch.__version__, numpy=np.__version__), load_audit=model.inference_load_audit,
                 baseline_gate_sha256=m.sha256(args.baseline_gate) if args.phase != 'baseline' else None,
                 pilot_lock_sha256=m.sha256(args.pilot_lock) if args.phase == 'full' else None,
                 source_sha256={str(p.relative_to(m.ROOT)):m.sha256(p) for p in [Path(__file__), m.ROOT/'src/uav_metric_research/multiscale.py']}))
    aggregate = m.frozen('metrics').MetricAccumulator()
    per_images, sample_rows, runtimes, pairs = [], [], [], []
    grouped, depth_groups = {}, {}
    weights_sha = m.sha256(Path(spec['checkpoint'])/'model.safetensors')
    for i in indices:
        row = rows[i]
        image_path = Path(args.rgb_root)/row['image_relpath']
        depth_path = Path(args.depth_root)/row['depth_relpath']
        if m.sha256(image_path) != inventory[i]['rgb_sha256'] or m.sha256(depth_path) != inventory[i]['gt_sha256']:
            raise ValueError('Input digest mismatch')
        rgb = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        gt = np.asarray(tifffile.imread(depth_path), np.float32)
        if rgb.shape[:2] != gt.shape:
            raise ValueError('Original image/GT mismatch')
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        try:
            pred, extra = prediction(model, rgb, method)
            torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError:
            m.write_json(out/'failure.json', dict(reason='OOM; stop without automatic retry', image_index=i))
            raise
        elapsed = time.monotonic() - start
        m.validate_prediction(pred, gt)
        cache = out / f'prediction_{i:03d}.npy'
        with cache.open('xb') as stream:
            np.save(stream, pred, allow_pickle=False)
        m.write_json(cache.with_suffix('.json'), dict(weights_sha256=weights_sha,
                      rgb_sha256=inventory[i]['rgb_sha256'], method=method, original_shape=list(gt.shape),
                      prediction_sha256=m.sha256(cache), prediction_dtype=str(pred.dtype), precision='fp32',
                      interpolation='RGB bicubic; depth bilinear align_corners=True', launch_sha256=m.sha256(out/'launch.json')))
        runtime = dict(image_index=i, seconds=elapsed, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                       peak_reserved_bytes=torch.cuda.max_memory_reserved(), **extra)
        print('RUNTIME', json.dumps(runtime), flush=True)
        if runtime['peak_allocated_bytes'] > 6.5 * 1024**3:
            m.write_json(out/'failure.json', dict(reason='Conservative 6.5GiB allocation guard; stop', image_index=i))
            raise ValueError('GPU memory safety margin exhausted')
        runtimes.append(runtime)
        masked = np.where(m.valid_gt(gt), gt, 0)
        accumulator = m.frozen('metrics').MetricAccumulator()
        accumulator.update(torch.from_numpy(pred), torch.from_numpy(masked))
        m.merge_depth(aggregate, accumulator)
        for scope in ('crop='+row['crop'], 'view='+row['view'], row['crop']+'|'+row['view']):
            m.merge_depth(depth_groups.setdefault(scope, m.frozen('metrics').MetricAccumulator()), accumulator)
        per_images.append(dict(image_index=i, image_relpath=row['image_relpath'], crop=row['crop'], view=row['view'], **accumulator.result()))
        scales = (14,) if args.phase == 'baseline' else m.SCALES
        stats, coordinates = m.multiscale(pred, gt, i, scales=scales)
        if coordinates != [r for r in expected_pairs[i] if r['stride'] in scales]:
            raise ValueError('Evaluation coordinates differ from the fixed pair manifest')
        pairs.append(dict(image_index=i, pairs=coordinates))
        for stride, groups in stats.items():
            for group, accumulator in groups.items():
                sample_rows.append(dict(image_index=i, stride_px=stride, group=group, **accumulator.result()))
                for scope in ('all', 'crop='+row['crop'], 'view='+row['view'], row['crop']+'|'+row['view']):
                    grouped.setdefault((scope,stride,group), m.DeltaStats()).merge(accumulator)
        print('INFERENCE', args.phase, spec['epoch'], i, elapsed, flush=True)
    m.write_csv(out/'per_image_metrics.csv', per_images)
    m.write_csv(out/'multiscale_per_image.csv', sample_rows)
    m.write_csv(out/'runtime_metrics.csv', runtimes)
    m.write_csv(out/'group_metrics.csv', [dict(scope=key, images=sum(1 for i in indices if key in ('crop='+rows[i]['crop'], 'view='+rows[i]['view'], rows[i]['crop']+'|'+rows[i]['view'])), **value.result()) for key,value in sorted(depth_groups.items())])
    m.write_csv(out/'multiscale_metrics.csv', [dict(scope=key[0], stride_px=key[1], group=key[2], **v.result()) for key,v in sorted(grouped.items())])
    m.write_json(out/'pair_manifest.json', pairs)
    result = aggregate.result()
    m.write_json(out/'summary.json', dict(images=len(indices), epoch=spec['epoch'], method=method, metrics=result))
    if args.phase == 'baseline':
        expected_rows = list(csv.DictReader((m.RUN/'gpgpu4_completion_20261015/epoch_metrics_1_20.csv').open(encoding='utf-8-sig')))
        row = next(r for r in expected_rows if int(r['epoch']) == spec['epoch'])
        expected = {k:float(row['validation_'+k]) for k in ('mae_m','rmse_m','abs_rel','pearson_r')}
        expected.update(local_depth_delta_mae_m=float(row['full_14px_local_mae_m']), local_pair_count=int(row['full_14px_valid_pair_count']),
                        valid_pixels=sum(int(v['valid_pixels']) for v in inventory.values()))
        proof = m.check_baseline(result, expected)
        all_pairs = grouped[('all', 14, 'all')].result()
        large = grouped[('all', 14, 'gt_ge20cm')].result()
        sample_checks = dict(
            fixed_sample_pairs=all_pairs['pair_count'] == int(row['fixed_sample_all_pair_count']),
            large_pairs=large['pair_count'] == int(row['fixed_sample_gt_ge20cm_pair_count']),
            large_mae=abs(large['delta_mae_m'] - float(row['fixed_sample_gt_ge20cm_local_mae_m'])) <= .0001,
            large_amplitude=abs(large['amplitude_ratio'] - float(row['fixed_sample_gt_ge20cm_amplitude_ratio'])) <= .001,
            large_sign=abs(large['sign_match_rate'] - float(row['fixed_sample_gt_ge20cm_sign_match_rate'])) <= .001)
        proof.update(sample_checks=sample_checks, sampled_all=all_pairs, sampled_large=large)
        proof['passed'] = proof['passed'] and all(sample_checks.values())
        m.write_json(out/'baseline_reproduction.json', proof)
        if not proof['passed']:
            raise ValueError('Baseline reproduction failed; stop before method comparison')
    del model
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare','run'])
    parser.add_argument('--manifest', type=Path, default=m.ROOT/'configs/uav3dcrop_verified/validation.csv')
    parser.add_argument('--rgb-root', type=Path, default=m.ROOT/'data/uav3dcrop_rgb')
    parser.add_argument('--depth-root', type=Path, default=m.ROOT/'data/uav3dcrop_depth')
    parser.add_argument('--output', type=Path, default=EXPERIMENT)
    parser.add_argument('--spec', type=Path)
    parser.add_argument('--phase', choices=['baseline','pilot','full'], default='baseline')
    parser.add_argument('--preparation', type=Path, default=EXPERIMENT)
    parser.add_argument('--gpu-receipt', type=Path)
    parser.add_argument('--baseline-gate', type=Path)
    parser.add_argument('--pilot-lock', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args)
    else:
        if args.spec is None or args.gpu_receipt is None or (args.phase != 'baseline' and args.baseline_gate is None):
            parser.error('run requires spec, GPU receipt, and (after baseline) combined baseline gate')
        if args.phase == 'full' and args.pilot_lock is None:
            parser.error('full phase requires a pilot lock')
        run(args)


if __name__ == '__main__':
    main()
