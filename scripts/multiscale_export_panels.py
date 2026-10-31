"""CPU-only seam diagnostics and color panels for final Windows figure composition.

No matplotlib installation on the GPU host. All scoring uses float32 original
predictions; PNG panels are visualization assets, never metric inputs.
"""
import argparse,csv,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import cv2
import numpy as np
import tifffile
from uav_metric_research import multiscale as m


def panel(path,value,valid,lo,hi,palette,step=1):
    sampled=np.nan_to_num(value[::step,::step],nan=lo,posinf=hi,neginf=lo)
    normalized=np.clip((sampled-lo)/max(hi-lo,1e-9),0,1)
    color=cv2.applyColorMap(np.rint(normalized*255).astype(np.uint8),palette)
    color[~valid[::step,::step]]=(255,255,255)
    if not cv2.imwrite(str(path),color):raise ValueError('Failed PNG write')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',nargs='+',type=Path,required=True)
    p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--depth-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    selection=json.loads(args.selection.read_text())
    tile_method=selection['selected_tile_method']
    rows=m.validation_rows(m.ROOT/'configs/uav3dcrop_verified/validation.csv')
    preparation=m.ROOT/'experiments/multiscale_tile_inference_20261016'
    pilot=list(csv.DictReader((preparation/'pilot_selection.csv').open()))
    inventory=list(csv.DictReader((preparation/'image_inventory.csv').open()))
    representatives={min(int(r['image_index']) for r in pilot if (r['crop'],r['view'])==key)
                     for key in {(r['crop'],r['view']) for r in pilot}}
    launches=[json.loads((run/'launch.json').read_text()) for run in args.runs]
    epoch=launches[0]['spec']['epoch']
    if any(launch['spec']['epoch']!=epoch or launch['indices']!=list(range(128)) for launch in launches):
        raise ValueError('Complete matched full-validation runs required')
    labels=[launch['spec']['label'] for launch in launches]
    all_stats,per,seam_manifest,metadata,consistency={},[],[],[],[]
    for i,row in enumerate(rows):
        gt_path=args.depth_root/row['depth_relpath']
        if m.sha256(gt_path)!=inventory[i]['gt_sha256']:raise ValueError('GT hash mismatch')
        gt=np.asarray(tifffile.imread(gt_path),np.float32)
        valid=m.valid_gt(gt)
        predictions={}
        reference_weights=None
        for run,launch,label in zip(args.runs,launches,labels):
            marker=json.loads((run/f'prediction_{i:03d}.json').read_text())
            if marker['rgb_sha256']!=inventory[i]['rgb_sha256']:raise ValueError('RGB hash mismatch')
            if reference_weights is None:reference_weights=marker['weights_sha256']
            if marker['weights_sha256']!=reference_weights:raise ValueError('Different method weights')
            predictions[label]=m.load_prediction_cache(run/f'prediction_{i:03d}.npy',dict(method=launch['spec']['method'],launch_sha256=m.sha256(run/'launch.json')))
            m.validate_prediction(predictions[label],gt)
        baseline=predictions[labels[0]]
        for label,pred in predictions.items():
            difference=pred-baseline
            consistency.append(dict(epoch=epoch,method=label,image_index=i,crop=row['crop'],view=row['view'],
                mean_difference_from_B0_m=float(difference.mean(dtype=np.float64)),
                mean_abs_difference_from_B0_m=float(np.abs(difference).mean(dtype=np.float64)),
                median_difference_from_B0_m_step16=float(np.median(difference[::16,::16]))))
        image_stats={}
        for axis,y,x,yy,xx in m.sample_pairs(gt.shape,valid,i,14):
            delta=gt[y,x].astype(float)-gt[yy,xx].astype(float)
            groups=m.seam_groups(gt.shape,tile_method,y,x,yy,xx)
            coords=np.stack([y,x,yy,xx],axis=1).astype('<i4')
            import hashlib
            seam_manifest.append(dict(image_index=i,axis=axis,pairs=len(y),coordinates_sha256=hashlib.sha256(coords.tobytes()).hexdigest()))
            for label,pred in predictions.items():
                dp=pred[y,x].astype(float)-pred[yy,xx].astype(float)
                for name,select in groups.items():
                    for suffix,keep in [('',select),('_gt_ge20cm',select&(np.abs(delta)>=.2))]:
                        image_stats.setdefault((label,name+suffix),m.DeltaStats()).update(delta[keep],dp[keep])
        for (label,group),stats in image_stats.items():
            per.append(dict(epoch=epoch,method=label,image_index=i,crop=row['crop'],view=row['view'],group=group,**stats.result()))
            for scope in ('all','crop='+row['crop'],'view='+row['view'],row['crop']+'|'+row['view']):
                all_stats.setdefault((label,scope,group),m.DeltaStats()).merge(stats)
        if i in representatives:
            lo,hi=map(float,np.quantile(gt[valid],[.01,.99]))
            h,w=gt.shape
            boxes=m.tiles(gt.shape,tile_method['tile_size'],tile_method['overlap'])
            boundaries=sorted({box[1] for box in boxes if box[1]>0}|{box[3] for box in boxes if box[3]<w})
            seam=min(boundaries,key=lambda x:abs(x-w/2))
            top=max(0,min(h-512,h//2-256));left=max(0,min(w-526,seam-256))
            roi=np.s_[top:top+512,left:left+512]
            roi_lo,roi_hi=map(float,np.quantile(gt[roi][valid[roi]],[.01,.99]))
            case=dict(epoch=epoch,image_index=i,image_relpath=row['image_relpath'],crop=row['crop'],view=row['view'],
                 shape=list(gt.shape),labels=['GT']+labels,overview_sample_step=8,depth_min_m=lo,depth_max_m=hi,
                 absolute_error_max_m=2,delta_error_max_m=.5,delta_axis='horizontal, 14 original pixels, located at first endpoint',
                 roi=dict(top=top,left=left,height=512,width=512,seam_x=seam,depth_min_m=roi_lo,depth_max_m=roi_hi),
                 tile_boxes=boxes,palette_depth='viridis256',palette_error='magma256')
            for label,pred in dict(GT=gt,**predictions).items():
                stem=f'case_{i:03d}_{label}'
                err=np.abs(pred-gt)
                dv=valid[:,14:]&valid[:,:-14]
                de=np.abs((pred[:,:-14]-pred[:,14:])-(gt[:,:-14]-gt[:,14:]))
                panel(args.output/(stem+'_depth.png'),pred,valid,lo,hi,cv2.COLORMAP_VIRIDIS,8)
                panel(args.output/(stem+'_absolute.png'),err,valid,0,2,cv2.COLORMAP_MAGMA,8)
                panel(args.output/(stem+'_delta.png'),de,dv,0,.5,cv2.COLORMAP_MAGMA,8)
                panel(args.output/(stem+'_roi_depth.png'),pred[roi],valid[roi],roi_lo,roi_hi,cv2.COLORMAP_VIRIDIS)
                panel(args.output/(stem+'_roi_absolute.png'),err[roi],valid[roi],0,2,cv2.COLORMAP_MAGMA)
                panel(args.output/(stem+'_roi_delta.png'),de[roi],dv[roi],0,.5,cv2.COLORMAP_MAGMA)
            metadata.append(case)
        if (i+1)%16==0:print('SEAM_AND_PANELS',epoch,i+1,'/128',flush=True)
    m.write_csv(args.output/'seam_per_image.csv',per)
    m.write_csv(args.output/'seam_metrics.csv',[dict(epoch=epoch,method=k[0],scope=k[1],group=k[2],**s.result()) for k,s in sorted(all_stats.items())])
    m.write_csv(args.output/'global_consistency.csv',consistency)
    m.write_json(args.output/'seam_pair_manifest.json',seam_manifest)
    m.write_json(args.output/'panel_metadata.json',metadata)


if __name__=='__main__':main()
