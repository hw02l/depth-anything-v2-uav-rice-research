"""Inference-only geometry and evaluation; original pixels, metres, fixed GT mask.

Frozen helpers are imported without modifying their files. No optimizer exists here.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'experiments/metric_outdoor_full_20261014_s42_518_cap80_gt65'
SCALES = (14, 28, 56, 107, 214)


def frozen(name):
    package = '_uav_frozen_full_v1'
    if package not in sys.modules:
        mod = types.ModuleType(package)
        mod.__path__ = [str(RUN / 'source_full_v1')]
        sys.modules[package] = mod
    key = package + '.' + name
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, RUN / 'source_full_v1' / (name + '.py'))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        spec.loader.exec_module(mod)
    return sys.modules[key]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError('Do not emit an empty table as an experiment result')
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validation_rows(path):
    rows = frozen('data').read_manifest(path)
    if len(rows) != 128 or any(r['split'] != 'validation' or r['day'] != '2025/Day047' for r in rows):
        raise ValueError('Only the original 128 Day047 validation rows are allowed')
    canonical = ROOT / 'configs/uav3dcrop_verified/validation.csv'
    if sha256(path) != sha256(canonical):
        raise ValueError('Validation manifest bytes differ from the canonical manifest')
    return rows


def resized_shape(height, width, short_side=518):
    if min(height, width) < 1 or short_side < 56 or short_side % 14:
        raise ValueError('Invalid image or patch size')
    scale = short_side / min(height, width)
    round14 = frozen('data')._round_multiple
    return round14(height * scale), round14(width * scale)


def preprocess(rgb, short_side=518):
    import cv2
    import torch
    data = frozen('data')
    h, w = resized_shape(*rgb.shape[:2], short_side)
    image = cv2.resize(rgb.astype(np.float32) / 255.0, (w, h), interpolation=cv2.INTER_CUBIC)
    image = (image - data.IMAGENET_MEAN) / data.IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()


def restore(native, shape):
    import torch
    import torch.nn.functional as F
    tensor = torch.as_tensor(np.ascontiguousarray(native)).float()
    return F.interpolate(tensor[None, None], size=shape, mode='bilinear', align_corners=True)[0, 0].numpy()


def starts(length, tile_size, overlap):
    if length < 1 or tile_size < 1 or not 0 <= overlap < 1:
        raise ValueError('Invalid tiling parameters')
    tile_size = min(length, tile_size)
    step = max(1, int(round(tile_size * (1 - overlap))))
    result = list(range(0, length - tile_size + 1, step))
    if result[-1] != length - tile_size:
        result.append(length - tile_size)
    return result


def tiles(shape, tile_size, overlap):
    h, w = shape
    return [(y, x, min(y + tile_size, h), min(x + tile_size, w))
            for y in starts(h, tile_size, overlap) for x in starts(w, tile_size, overlap)]


def tile_weight(shape):
    # Positive raised cosine: no uncovered/zero-weight pixels at the outer edge.
    h, w = shape
    return np.outer(np.maximum(np.hanning(h), 1e-3), np.maximum(np.hanning(w), 1e-3)).astype(np.float32)


class TileMerger:
    def __init__(self, shape):
        self.total = np.zeros(shape, np.float32)
        self.weight = np.zeros(shape, np.float32)

    def add(self, box, prediction):
        y, x, bottom, right = box
        shape = (bottom - y, right - x)
        if prediction.shape != shape or min(y, x) < 0 or bottom > self.total.shape[0] or right > self.total.shape[1]:
            raise ValueError('Tile coordinates or prediction shape mismatch')
        if not np.isfinite(prediction).all() or (prediction <= 0).any():
            raise ValueError('Invalid tile prediction')
        weight = tile_weight(shape)
        self.total[y:bottom, x:right] += prediction * weight
        self.weight[y:bottom, x:right] += weight

    def result(self):
        if (self.weight <= 0).any():
            raise ValueError('Uncovered pixels')
        return self.total / self.weight


def valid_gt(gt):
    return np.isfinite(gt) & (gt > 0) & (gt <= 65)


def validate_prediction(pred, gt):
    # Fail the entire image instead of quietly changing the shared GT mask.
    if pred.shape != gt.shape or not np.isfinite(pred).all() or (pred <= 0).any() or (pred > 80.001).any():
        raise ValueError('Prediction shape/range invalid; image must not be silently omitted')


def sample_pairs(shape, valid, image_index, stride, candidates=100000):
    """Exactly the legacy seed/order; coordinates depend only on original GT."""
    h, w = shape
    rng = np.random.default_rng(42 + image_index * 10000 + stride)
    for axis in (0, 1):
        hh, ww = h - (stride if axis == 0 else 0), w - (stride if axis == 1 else 0)
        if min(hh, ww) <= 0:
            continue
        flat = rng.choice(hh * ww, size=min(candidates, hh * ww), replace=False)
        y, x = flat // ww, flat % ww
        yy, xx = y + (stride if axis == 0 else 0), x + (stride if axis == 1 else 0)
        keep = valid[y, x] & valid[yy, xx]
        yield axis, y[keep], x[keep], yy[keep], xx[keep]


def pair_groups(gt, y, x, yy, xx):
    h, w = gt.shape
    # Delta sign follows user convention d(i)-d(j); legacy sign is the reverse
    # for both prediction and GT, leaving MAE/amplitude/sign-match unchanged.
    delta = gt[y, x].astype(np.float64) - gt[yy, xx].astype(np.float64)
    magnitude = np.abs(delta)
    edge0 = np.minimum.reduce([y, x, h - 1 - y, w - 1 - x])
    edge1 = np.minimum.reduce([yy, xx, h - 1 - yy, w - 1 - xx])
    edge = np.minimum(edge0, edge1)
    result = {'all': np.ones(len(y), bool), 'gt_ge20cm': magnitude >= .2,
              'border_lt128': edge < 128, 'middle_128_512': (edge >= 128) & (edge < 512),
              'interior_ge512': edge >= 512, 'positive_gt': delta > 1e-6,
              'negative_gt': delta < -1e-6, 'zero_gt': magnitude <= 1e-6}
    for lo, hi in ((0, .05), (.05, .1), (.1, .2), (.2, .5), (.5, np.inf)):
        result[f'gt_delta_{lo:g}_{hi:g}m'] = (magnitude >= lo) & (magnitude < hi)
    mean_depth = (gt[y, x].astype(float) + gt[yy, xx].astype(float)) / 2
    for lo, hi in ((0, 15), (15, 20), (20, 30), (30, 65)):
        result[f'depth_{lo}_{hi}m'] = (mean_depth >= lo) & ((mean_depth <= hi) if hi == 65 else (mean_depth < hi))
    z0, z1 = np.digitize(edge0, [128, 512]), np.digitize(edge1, [128, 512])
    for a in range(3):
        for b in range(a, 3):
            result[f'endpoint_zones_{a}_{b}'] = (np.minimum(z0, z1) == a) & (np.maximum(z0, z1) == b)
    return delta, result


class DeltaStats:
    def __init__(self):
        self.legacy = frozen('metrics').SampledDeltaAccumulator()
        self.extra = np.zeros(4, np.float64)

    def update(self, gt, pred):
        gt, pred = np.asarray(gt, np.float64), np.asarray(pred, np.float64)
        self.legacy.update(gt, pred)
        diff = np.abs(pred) - np.abs(gt)
        self.extra += [np.square(pred - gt).sum(), (diff > 1e-6).sum(), (diff < -1e-6).sum(), diff.sum()]

    def merge(self, other):
        for name in ('total', 'small', 'large'):
            target, values = getattr(self.legacy, name), getattr(other.legacy, name)
            for i, value in enumerate(values):
                target[i] += value
        self.extra += other.extra

    def result(self):
        result = self.legacy.result()['all_sampled_pairs']
        n = result['pair_count']
        result.update(delta_rmse_m=float(np.sqrt(self.extra[0] / n)) if n else None,
                      zero_delta_mae_m=result['mean_abs_gt_delta_m'],
                      overestimate_fraction=float(self.extra[1] / n) if n else None,
                      underestimate_fraction=float(self.extra[2] / n) if n else None,
                      abs_amplitude_bias_m=float(self.extra[3] / n) if n else None)
        return result


def multiscale(pred, gt, image_index, candidates=100000, scales=SCALES):
    validate_prediction(pred, gt)
    result, pair_manifest = {}, []
    mask = valid_gt(gt)
    for stride in scales:
        stats = {}
        for axis, y, x, yy, xx in sample_pairs(gt.shape, mask, image_index, stride, candidates):
            coordinates = np.stack([y, x, yy, xx], axis=1).astype('<i4')
            pair_manifest.append(dict(stride=stride, axis=axis, pairs=len(y),
                                      coordinates_sha256=hashlib.sha256(coordinates.tobytes()).hexdigest()))
            delta, groups = pair_groups(gt, y, x, yy, xx)
            prediction = pred[y, x].astype(np.float64) - pred[yy, xx].astype(np.float64)
            groups[f'axis_{axis}'] = np.ones(len(y), bool)
            for group, select in groups.items():
                stats.setdefault(group, DeltaStats()).update(delta[select], prediction[select])
        result[stride] = stats
    return result, pair_manifest


def paired_bootstrap(differences, seed=42, repeats=10000):
    differences = np.asarray(differences, np.float64)
    if not len(differences) or not np.isfinite(differences).all():
        raise ValueError('Paired complete finite images required')
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(differences, len(differences), replace=True).mean() for _ in range(repeats)])
    return dict(images=len(differences), mean_difference=float(differences.mean()),
                improved=int((differences < 0).sum()), worsened=int((differences > 0).sum()),
                unchanged=int((differences == 0).sum()), ci95=np.quantile(means, [.025, .975]).tolist(),
                scope='image bootstrap within Day047; correlated flight images; no between-day inference')


def check_baseline(actual, expected):
    checks = {}
    for key, tolerance in [('rmse_m', .002), ('mae_m', .002), ('abs_rel', .0001),
                           ('pearson_r', .0001), ('local_depth_delta_mae_m', .0001)]:
        checks[key] = abs(actual[key] - expected[key]) <= tolerance
    checks['valid_pixels'] = actual['valid_pixels'] == expected['valid_pixels']
    checks['local_pair_count'] = actual['local_pair_count'] == expected['local_pair_count']
    return dict(passed=all(checks.values()), checks=checks, actual=actual, expected=expected)


def merge_depth(target, source):
    if target.local_stride != source.local_stride:
        raise ValueError('Depth accumulator stride differs')
    for field in source.__dataclass_fields__:
        if field != 'local_stride':
            setattr(target, field, getattr(target, field) + getattr(source, field))


def load_prediction_cache(path, expected):
    path = Path(path)
    marker = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    for key, value in expected.items():
        if marker.get(key) != value:
            raise ValueError('Stale cache context: ' + key)
    if sha256(path) != marker['prediction_sha256']:
        raise ValueError('Cache digest mismatch')
    value = np.load(path, allow_pickle=False)
    if list(value.shape) != marker['original_shape'] or str(value.dtype) != marker['prediction_dtype']:
        raise ValueError('Cache shape/dtype mismatch')
    if not np.isfinite(value).all() or (value <= 0).any() or (value > 80.001).any():
        raise ValueError('Invalid cache values')
    return value


def strict_attention_state(model, state):
    """Audit the known 96-name DINOv2 serialization mapping, then strict load.

    No tensor operations, dropped keys, or permissive load. Stored safetensors
    hashes are checked separately before this serialization adapter is called.
    """
    import torch
    target = model.state_dict()
    mapped = dict(state)
    mapping = []
    if set(mapped) != set(target):
        for layer in range(12):
            prefix = f'backbone.encoder.layer.{layer}.attention.'
            for old_name, new_name in [('attention.query','q_proj'), ('attention.key','k_proj'),
                             ('attention.value','v_proj'), ('output.dense','o_proj')]:
                for suffix in ('weight','bias'):
                    old, new_key = prefix+old_name+'.'+suffix, prefix+new_name+'.'+suffix
                    if old not in mapped or new_key in mapped or new_key not in target:
                        raise ValueError('Incomplete or mixed attention key mapping')
                    mapped[new_key] = mapped.pop(old)
                    mapping.append(dict(stored=old, runtime=new_key))
        if len(mapping) != 96 or set(mapped) != set(target):
            raise ValueError('Unapproved state-dict key differences')
    for name, value in mapped.items():
        if value.shape != target[name].shape or value.dtype != target[name].dtype or not torch.isfinite(value).all():
            raise ValueError('State tensor shape/dtype/finite mismatch: ' + name)
    model.load_state_dict(mapped, strict=True)
    loaded = model.state_dict()
    if any(not torch.equal(value, loaded[name]) for name, value in mapped.items()):
        raise ValueError('Loaded tensor bytes differ')
    return dict(mode='identity' if not mapping else 'legacy_serialized_attention_to_runtime',
                renamed_tensors=len(mapping), verified_tensors=len(mapped),
                strict_load=True, loaded_values_exact=True, mapping=mapping)
