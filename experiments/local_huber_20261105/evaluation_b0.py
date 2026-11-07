"""B0 evaluation primitives; full fixed-pair validation integration in progress."""
import json
from pathlib import Path
import sys
import cv2
import numpy as np
import tifffile
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from uav_metric_research import multiscale as m


def evaluate_b0(model, rows, rgb_root, depth_root, device, pair_manifest, *, scales=m.SCALES, indices=None):
    """Pool absolute/full14 and fixed sampled statistics, enforcing pair hashes.

    rows must be the canonical ordered validation manifest, never a selected
    subset; fixed pair hashes protect numpy RNG and input-mask compatibility.
    """
    expected=json.loads(Path(pair_manifest).read_text(encoding='utf-8'))['images']
    if len(rows)!=128 or len(expected)!=128:
        raise ValueError('full canonical validation128 required')
    indices=list(range(128)) if indices is None else list(indices)
    if not indices or len(set(indices))!=len(indices) or any(i<0 or i>=128 for i in indices):
        raise ValueError('invalid validation indices')
    absolute=m.frozen('metrics').MetricAccumulator()
    grouped={}; per=[]
    was_training=model.training; model.eval()
    try:
        for i in indices:
            row=rows[i]
            bgr=cv2.imread(str(Path(rgb_root)/row['image_relpath']),cv2.IMREAD_COLOR)
            if bgr is None: raise FileNotFoundError(row['image_relpath'])
            rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            gt=np.asarray(tifffile.imread(Path(depth_root)/row['depth_relpath']),np.float32)
            if rgb.shape[:2] != gt.shape: raise ValueError('RGB/GT shape mismatch')
            with torch.inference_mode():
                native=model(pixel_values=m.preprocess(rgb,518)[None].to(device)).predicted_depth[0].float().cpu().numpy()
            pred=m.restore(native,gt.shape); m.validate_prediction(pred,gt)
            acc=m.frozen('metrics').MetricAccumulator(); acc.update(torch.from_numpy(pred),torch.from_numpy(np.where(m.valid_gt(gt),gt,0)))
            m.merge_depth(absolute,acc)
            stats,coords=m.multiscale(pred,gt,i,scales=scales)
            if coords != [r for r in expected[i]['pairs'] if r['stride'] in scales]:
                raise ValueError(f'fixed pair manifest mismatch for image {i}')
            for stride,groups in stats.items():
                for group,value in groups.items():
                    for scope in ('all','crop='+row['crop'],'view='+row['view']):
                        grouped.setdefault((scope,stride,group),m.DeltaStats()).merge(value)
            per.append({'image_index':i,'image_relpath':row['image_relpath'],'crop':row['crop'],'view':row['view'],**acc.result()})
            del pred,native,rgb,gt,bgr
        if str(device).startswith('cuda'): torch.cuda.empty_cache()
    finally:
        model.train(was_training)
    return {'absolute':absolute.result(),
            'sampled':[{'scope':k[0],'stride_px':k[1],'group':k[2],**v.result()} for k,v in sorted(grouped.items())],
            'per_image':per}
