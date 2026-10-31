"""Measure the real sequential B0->B2->B3 pipeline on the fixed 16 pilot RGBs."""
import argparse,csv,json,socket,time
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import cv2
import numpy as np
import torch
import multiscale_experiment as cli
from uav_metric_research import multiscale as m
from uav_metric_research.fusion import global_local_fusion


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--fusion-cache',type=Path,required=True)
    p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--rgb-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if socket.gethostname().lower()!='gpgpu4' or not torch.cuda.is_available():raise ValueError('Audited GPGPU4 required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    launch=json.loads((args.baseline/'launch.json').read_text())
    epoch=launch['spec']['epoch']
    selection=json.loads(args.selection.read_text())
    preparation=m.ROOT/'experiments/multiscale_tile_inference_20261016'
    indices=sorted(int(r['image_index']) for r in csv.DictReader((preparation/'pilot_selection.csv').open()))
    inventory=list(csv.DictReader((preparation/'image_inventory.csv').open()))
    rows=m.validation_rows(m.ROOT/'configs/uav3dcrop_verified/validation.csv')
    model=cli.strict_model(launch['spec']['checkpoint'],epoch)
    method=selection['selected_tile_method']
    def rgb(i):
        path=args.rgb_root/rows[i]['image_relpath']
        if m.sha256(path)!=inventory[i]['rgb_sha256']:raise ValueError('RGB hash mismatch')
        return cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB)
    def infer(image):
        b,_=cli.prediction(model,image,dict(kind='whole',short_side=518))
        q,_=cli.prediction(model,image,method)
        return global_local_fusion(b,q,radius=107,alpha=1)
    native_shapes=[]
    hook=model.register_forward_hook(lambda module,inputs,output:native_shapes.append(list(output.predicted_depth.shape)))
    infer(rgb(indices[0]))
    probe,_=cli.prediction(model,rgb(indices[0]),dict(kind='whole',short_side=1036))
    hook.remove()
    b1_expected=m.load_prediction_cache(args.fusion_cache.parent/f'B1_1036_full_ep{epoch}'/f'prediction_{indices[0]:03d}.npy',{})
    b1_difference=float(np.abs(probe-b1_expected).max())
    if b1_difference>1e-5:raise ValueError('B1 shape probe differs from full cache')
    m.write_json(args.output.with_suffix('.shapes.json'),dict(epoch=epoch,native_output_shapes=native_shapes,
        B0_input_shape=list(m.resized_shape(3956,5280,518)),B1_input_shape=list(m.resized_shape(3956,5280,1036)),
        B1_probe_max_abs_difference_from_full_cache_m=b1_difference))
    del probe,b1_expected
    torch.cuda.empty_cache()
    # B0/B1/B2 full-run timings at these IDs were warm; one fixed RGB warm-up
    # makes the 16-image timing comparison consistent. No GT is read here.
    infer(rgb(indices[0]))
    results=[]
    for i in indices:
        image=rgb(i)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        start=time.monotonic()
        pred=infer(image)
        torch.cuda.synchronize()
        elapsed=time.monotonic()-start
        expected=m.load_prediction_cache(args.fusion_cache/f'prediction_{i:03d}.npy',{})
        max_difference=float(np.abs(pred-expected).max())
        if max_difference>1e-5:raise ValueError('B3 sequential execution differs from derived cache')
        results.append(dict(epoch=epoch,image_index=i,seconds=elapsed,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                            peak_reserved_bytes=torch.cuda.max_memory_reserved(),max_abs_difference_from_full_cache_m=max_difference))
        print('B3_REAL_PIPELINE_BENCHMARK',epoch,i,elapsed,max_difference,flush=True)
    m.write_csv(args.output,results)
    del model
    torch.cuda.empty_cache()


if __name__=='__main__':main()
