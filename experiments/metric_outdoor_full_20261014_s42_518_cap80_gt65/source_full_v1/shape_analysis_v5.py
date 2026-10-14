"""Paired validation-only shape diagnostics. No optimizer or downloads.

Absolute metrics are reused from identical-mask full validation evaluations.
Detailed shape metrics use deterministic shared point samples, never claimed
as full-pixel counts. Predictions are cached at native output resolution.
"""
from pathlib import Path
import argparse, csv, hashlib, json, time, math
import numpy as np
import cv2, torch, tifffile
from transformers import AutoModelForDepthEstimation
from data import UAVDepthDataset, read_manifest

def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_new(p,obj):
    with Path(p).open('x',encoding='utf-8') as f: json.dump(obj,f,indent=2,allow_nan=False)

def interp(a,y,x,H,W):
    # Same align_corners=True bilinear coordinates as metrics.evaluate_model.
    yy=y.astype(np.float64)*(a.shape[0]-1)/(H-1)
    xx=x.astype(np.float64)*(a.shape[1]-1)/(W-1)
    y0=np.floor(yy).astype(int); x0=np.floor(xx).astype(int)
    y1=np.minimum(y0+1,a.shape[0]-1); x1=np.minimum(x0+1,a.shape[1]-1)
    wy=yy-y0; wx=xx-x0
    v=(a[y0,x0]*(1-wx)+a[y0,x1]*wx)*(1-wy)+(a[y1,x0]*(1-wx)+a[y1,x1]*wx)*wy
    support=np.isfinite(a[y0,x0]) & np.isfinite(a[y0,x1]) & np.isfinite(a[y1,x0]) & np.isfinite(a[y1,x1])
    return v,support

class Stats:
    def __init__(self): self.a=np.zeros(15,np.float64)
    def update(self,t,p):
        t=np.asarray(t,np.float64); p=np.asarray(p,np.float64); n=len(t)
        if not n: return
        at=np.abs(t); ap=np.abs(p); nz=at>1e-6
        self.a+=np.array([n,np.abs(p-t).sum(),at.sum(),ap.sum(),t.sum(),p.sum(),(t*t).sum(),(p*p).sum(),(t*p).sum(),nz.sum(),((t[nz]*p[nz])>0).sum(),(ap>at+1e-6).sum(),(ap<at-1e-6).sum(),(ap-at).sum(),np.square(p-t).sum()])
    def result(self):
        n,err,zero,absp,sx,sy,sxx,syy,sxy,nz,sign,over,under,bias,se=self.a
        if not n: return {'sampled_pairs':0}
        vx=sxx-sx*sx/n; vy=syy-sy*sy/n; cov=sxy-sx*sy/n
        return {'sampled_pairs':int(n),'delta_mae_m':err/n,'delta_rmse_m':math.sqrt(se/n),'zero_delta_mae_m':zero/n,'mean_abs_gt_delta_m':zero/n,'mean_abs_pred_delta_m':absp/n,'amplitude_ratio':absp/zero if zero else None,'sign_nonzero_gt_pairs':int(nz),'sign_accuracy_nonzero_gt':sign/nz if nz else None,'overestimate_fraction':over/n,'underestimate_fraction':under/n,'abs_amplitude_bias_m':bias/n,'delta_pearson':cov/math.sqrt(vx*vy) if vx>0 and vy>0 else None,'regression_slope_with_intercept':cov/vx if vx>0 else None,'regression_intercept_m':sy/n-(cov/vx)*sx/n if vx>0 else None}

class DepthStats:
    def __init__(self): self.a=np.zeros(9,np.float64)
    def update(self,t,p):
        t=np.asarray(t,np.float64);p=np.asarray(p,np.float64);err=p-t
        self.a+=np.array([len(t),np.abs(err).sum(),np.square(err).sum(),(np.abs(err)/t).sum(),t.sum(),p.sum(),(t*t).sum(),(p*p).sum(),(t*p).sum()])
    def result(self):
        n,ae,se,ar,sx,sy,sxx,syy,sxy=self.a
        if not n:return {'sampled_pixels':0}
        vx=sxx-sx*sx/n;vy=syy-sy*sy/n;cov=sxy-sx*sy/n
        return {'sampled_pixels':int(n),'mae_m':ae/n,'rmse_m':math.sqrt(se/n),'abs_rel':ar/n,'pearson_r':cov/math.sqrt(vx*vy) if vx>0 and vy>0 else None}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--spec',type=Path,required=True)
    ap.add_argument('--phase',choices=['infer','analyze','figures'],required=True)
    args=ap.parse_args(); spec=json.loads(args.spec.read_text()); out=Path(spec['output']); out.mkdir(parents=True,exist_ok=True)
    rows=read_manifest(spec['manifest']); assert len(rows)==128 and all(r['split']=='validation' and r['day']=='2025/Day047' for r in rows)
    assert not any('Day001_' in r['scene'] or 'Day007_' in r['scene'] for r in rows)
    manifest_sha=digest(spec['manifest']); assert manifest_sha==spec['validation_sha256']
    dataset=UAVDepthDataset(rows,spec['rgb_root'],spec['depth_root'],image_size=518,training=False)
    cache=out/'prediction_cache'; cache.mkdir(exist_ok=True)
    if args.phase=='infer':
        assert torch.cuda.is_available(), 'CUDA required for inference phase'
        torch.set_num_threads(2); started=time.monotonic()
        for label,model_path in spec['models'].items():
            fingerprint=digest(Path(model_path)/'model.safetensors')
            model=AutoModelForDepthEstimation.from_pretrained(model_path,local_files_only=True).cuda().eval()
            assert model.head.max_depth==spec['model_caps'][label]
            for i,row in enumerate(rows):
                target=cache/f'{label}_{i:03d}.npy'; marker=target.with_suffix('.json')
                context={'manifest_sha256':manifest_sha,'weights_sha256':fingerprint,'image_size':518,'image_relpath':row['image_relpath'],'model_cap':model.head.max_depth,'amp_inference':False}
                if target.exists():
                    assert marker.exists() and json.loads(marker.read_text())==context, 'Refusing stale prediction cache'
                    continue
                item=dataset[i]
                with torch.inference_mode(): prediction=model(pixel_values=item['pixel_values'][None].cuda()).predicted_depth[0].float().cpu().numpy()
                assert np.isfinite(prediction).all() and (prediction>0).all()
                with target.open('xb') as f: np.save(f,prediction)
                write_new(marker,context)
                if (i+1)%16==0: print('INFER',label,i+1,'/128 elapsed',round(time.monotonic()-started),flush=True)
            del model; torch.cuda.empty_cache()
        print('GPU_INFERENCE_COMPLETE',flush=True); return
    if args.phase=='figures': return figures(spec,rows,cache,out)
    assert not (out/'shape_summary.json').exists(), 'Analysis already exists; reuse it'
    edges=[0,.05,.1,.2,float('inf')]; labels=['0-5cm','5-10cm','10-20cm','>=20cm']; strides=[1,7,14,28,56,107]
    stats={}; image_results=[]; oracle_stats={}; full_zero_n=0; full_zero_sum=0.; quantization=[];depth_stats={}
    def add(model,group,stride,t,p):
        key=f'{model}|{group}|s{stride}'
        stats.setdefault(key,Stats()).update(t,p)
    start=time.monotonic()
    for i,row in enumerate(rows):
        gt=np.asarray(tifffile.imread(Path(spec['depth_root'])/row['depth_relpath']),np.float32); H,W=gt.shape
        valid=np.isfinite(gt)&(gt>0)&(gt<=65)
        # Full-pair zero baseline at stride14, in memory-bounded row chunks.
        for axis in [0,1]:
            s=14
            for top in range(0,H-s if axis==0 else H,128):
                bottom=min(top+128,H-s if axis==0 else H)
                g0=gt[top:bottom,:] if axis==0 else gt[top:bottom,:-s]
                g1=gt[top+s:bottom+s,:] if axis==0 else gt[top:bottom,s:]
                v0=valid[top:bottom,:] if axis==0 else valid[top:bottom,:-s]
                v1=valid[top+s:bottom+s,:] if axis==0 else valid[top:bottom,s:]
                mask=v0&v1; full_zero_n+=int(mask.sum()); full_zero_sum+=float(np.abs(g1-g0)[mask].sum(dtype=np.float64))
        predictions={m:np.load(cache/f'{m}_{i:03d}.npy') for m in spec['models']}
        rng_pixel=np.random.default_rng(420000+i)
        ix=rng_pixel.choice(H*W,size=100000,replace=False); py=ix//W;px=ix%W
        pv=valid[py,px];py=py[pv];px=px[pv];target_pixel=gt[py,px]
        for m,pred in predictions.items():
            predicted_pixel,_=interp(pred,py,px,H,W)
            for group in ['all',f'view:{row["view"]}',f'crop:{row["crop"]}']:
                depth_stats.setdefault(f'{m}|{group}',DepthStats()).update(target_pixel,predicted_pixel)
        low_shape=next(iter(predictions.values())).shape
        low_gt=cv2.resize(gt,(low_shape[1],low_shape[0]),interpolation=cv2.INTER_NEAREST)
        low_gt[(low_gt<=0)|(low_gt>65)]=np.nan
        median=cv2.medianBlur(gt,3)
        interior=cv2.erode(valid.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
        per={m:Stats() for m in predictions}
        q=gt[valid][::max(1,int(valid.sum()/20000))].astype(np.float16)
        quantization.append({'image_index':i,'median_float16_step_m':float(np.median(np.nextafter(q,np.float16(np.inf)).astype(np.float32)-q.astype(np.float32)))})
        for stride in strides:
            rng=np.random.default_rng(42+i*10000+stride)
            for axis in [0,1]:
                hh=H-stride if axis==0 else H; ww=W-stride if axis==1 else W
                # Uniform candidates without replacement, same coordinates for every model.
                ix=rng.choice(hh*ww,size=min(100000,hh*ww),replace=False)
                y=ix//ww; x=ix%ww; y1=y+(stride if axis==0 else 0); x1=x+(stride if axis==1 else 0)
                good=valid[y,x]&valid[y1,x1]; y,x,y1,x1=y[good],x[good],y1[good],x1[good]
                t=(gt[y1,x1]-gt[y,x]).astype(np.float64); mag=np.abs(t); distance=(gt[y,x]+gt[y1,x1])/2
                for m,pred in predictions.items():
                    p0,_=interp(pred,y,x,H,W); p1,_=interp(pred,y1,x1,H,W); p=p1-p0
                    for group in ['all',f'crop:{row["crop"]}',f'view:{row["view"]}',f'axis:{axis}']:
                        add(m,group,stride,t,p)
                    for j,label in enumerate(labels):
                        select=(mag>=edges[j])&(mag<edges[j+1]); add(m,f'gt_delta:{label}',stride,t[select],p[select])
                    for lo,hi in [(0,15),(15,20),(20,30),(30,65)]:
                        select=(distance>=lo)&(distance<hi); add(m,f'depth:{lo}-{hi}m',stride,t[select],p[select])
                    if stride==14:
                        per[m].update(t,p)
                        near_invalid=~(interior[y,x]&interior[y1,x1])
                        median_outlier=np.maximum(np.abs(gt[y,x]-median[y,x]),np.abs(gt[y1,x1]-median[y1,x1]))>.2
                        for group,select in [('gt_support:near_invalid',near_invalid),('gt_support:interior',~near_invalid),('gt_median_outlier:>20cm',median_outlier),('gt_median_outlier:<=20cm',~median_outlier)]:
                            add(m,group,stride,t[select],p[select])
                o0,v0=interp(low_gt,y,x,H,W); o1,v1=interp(low_gt,y1,x1,H,W); ov=v0&v1
                oracle_stats.setdefault(f'GT_resize_roundtrip|s{stride}',Stats()).update(t[ov],(o1-o0)[ov])
                oracle_stats.setdefault(f'GT_median3_control|s{stride}',Stats()).update(t,(median[y1,x1]-median[y,x]).astype(np.float64))
        image_results.append({'index':i,'image_relpath':row['image_relpath'],'crop':row['crop'],'view':row['view'],'models':{m:s.result() for m,s in per.items()},'invalid_fraction':1-float(valid.mean()),'output_shape':list(low_shape)})
        if (i+1)%8==0: print('ANALYZE',i+1,'/128 elapsed',round(time.monotonic()-start),flush=True)
    payload={'analysis_code_sha256':digest(__file__),'manifest_sha256':manifest_sha,'validation_images':128,'gt_max_depth_m':65,'strides_original_pixels':strides,'sample_method':'100000 candidate pairs per image/axis/stride without replacement; seed=42+image_index*10000+stride; shared GT-valid coordinates across models','bin_edges_m':[0,.05,.1,.2,None],'sign_definition':'strict positive product, excluding GT exactly zero (abs<=1e-6); predicted zero is a mismatch','regression_definition':'OLS signed predicted delta vs signed GT delta, with intercept; no calibration of predictions','full_zero_delta_baseline_s14':{'valid_pairs':full_zero_n,'mae_m':full_zero_sum/full_zero_n},'stats':{k:v.result() for k,v in stats.items()},'gt_controls':{k:v.result() for k,v in oracle_stats.items()},'float16_quantization':quantization,'images':image_results,'elapsed_seconds':time.monotonic()-start,'limitations':['GT Z-depth is camera-axis depth, not vertical crop height','No plant/ground segmentation; crop/ground ROI labels not asserted','MVS GT noise controls are diagnostic, not proof of noise-free ground truth','Detailed shape statistics are sampled; full zero baseline and existing full validation MAE separately reported']}
    payload['sampled_absolute_depth_by_group']={k:v.result() for k,v in depth_stats.items()}
    payload['sampled_absolute_method']='100000 uniform candidate pixels per image, seed=420000+image_index; shared GT-valid coordinates; these are sampled, distinct from full validation metrics'
    write_new(out/'shape_summary.json',payload)
    fields=['model','group','stride']+list(Stats().result().keys())
    all_rows=[]
    for key,s in payload['stats'].items():
        m,g,st=key.split('|'); all_rows.append({'model':m,'group':g,'stride':int(st[1:]),**s})
    fields=['model','group','stride']+list(next(r for r in all_rows if r.get('sampled_pairs',0)>0).keys())[3:]
    with (out/'shape_groups.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(all_rows)
    print('SHAPE_ANALYSIS_COMPLETE',out/'shape_summary.json',flush=True)

def figures(spec,rows,cache,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary=json.loads((out/'shape_summary.json').read_text()); images=summary['images']
    ranked=sorted(images,key=lambda r:r['models']['A']['delta_mae_m']-r['models']['B']['delta_mae_m'],reverse=True)
    selected=[]
    for crop in ['corn','oat','soybean','wheat']:
        selected.append(next(r['index'] for r in ranked if r['crop']==crop))
    for view in sorted({r['view'] for r in images}):
        candidate=max((r for r in images if r['view']==view),key=lambda r:r['models']['A']['delta_mae_m'])['index']
        if candidate not in selected: selected.append(candidate)
    best=min(images,key=lambda r:r['models']['A']['delta_mae_m'])['index']
    if best not in selected: selected.append(best)
    figdir=out/'figures'; figdir.mkdir(exist_ok=True); chosen=[]
    for i in selected:
        row=rows[i]; rgb=cv2.cvtColor(cv2.imread(str(Path(spec['rgb_root'])/row['image_relpath'])),cv2.COLOR_BGR2RGB)
        gt=np.asarray(tifffile.imread(Path(spec['depth_root'])/row['depth_relpath']),np.float32); H,W=gt.shape; valid=np.isfinite(gt)&(gt>0)&(gt<=65)
        # Full-resolution interpolation, followed by display-only reduction.
        predictions={}
        for m in spec['models']:
            low=torch.from_numpy(np.load(cache/f'{m}_{i:03d}.npy'))
            predictions[m]=torch.nn.functional.interpolate(low[None,None],size=(H,W),mode='bilinear',align_corners=True)[0,0].numpy()
        fig,ax=plt.subplots(4,4,figsize=(18,15),constrained_layout=True)
        depth_min=float(np.quantile(gt[valid],.01)); depth_max=float(np.quantile(gt[valid],.99))
        # Use common GT-derived scale across ALL prediction panels; clipping labelled.
        ax[0,0].imshow(rgb); ax[0,0].set_title('RGB')
        ax[0,1].imshow(np.where(valid,gt,np.nan),cmap='viridis',vmin=depth_min,vmax=depth_max); ax[0,1].set_title('GT Z-depth (m)')
        ax[0,2].imshow(~valid,cmap='gray'); ax[0,2].set_title('GT invalid mask')
        dg=gt[:,14:]-gt[:,:-14]; dv=valid[:,14:]&valid[:,:-14]
        ax[0,3].imshow(np.where(dv,dg,np.nan),cmap='coolwarm',vmin=-.2,vmax=.2); ax[0,3].set_title('GT signed dx14 (m), clipped ±0.2')
        scores=np.where(dv,np.abs((predictions['A'][:,14:]-predictions['A'][:,:-14])-dg)-np.abs((predictions['B'][:,14:]-predictions['B'][:,:-14])-dg),0)
        # Fixed-grid ROI selection uses largest median error gap to avoid a single bad GT pixel.
        candidates=[(float(np.median(scores[y:y+512,x:x+512])),y,x) for y in range(0,H-512,384) for x in range(0,W-526,384)]
        _,yy,xx=max(candidates); roi=(slice(yy,yy+512),slice(xx,xx+512))
        for j,m in enumerate(['A','B','C'],1):
            p=predictions[m]; error=np.where(valid,np.abs(p-gt),np.nan)
            de=np.where(dv,np.abs((p[:,14:]-p[:,:-14])-dg),np.nan)
            im=ax[j,0].imshow(p,cmap='viridis',vmin=depth_min,vmax=depth_max); ax[j,0].set_title(f'{m} predicted depth; common scale')
            im1=ax[j,1].imshow(error,cmap='magma',vmin=0,vmax=10); ax[j,1].set_title(f'{m} absolute error (m), clipped 10')
            im2=ax[j,2].imshow(de,cmap='magma',vmin=0,vmax=.2); ax[j,2].set_title(f'{m} local dx14 error (m), clipped 0.2')
            ax[j,3].imshow(p[roi],cmap='viridis',vmin=depth_min,vmax=depth_max); ax[j,3].set_title(f'{m} ROI y{yy}:x{xx}, 512px')
        for a in ax.flat: a.axis('off')
        fig.colorbar(im,ax=ax[:,0:1],shrink=.4,label='Z-depth m; GT 1–99% display range')
        fig.colorbar(im1,ax=ax[:,1:2],shrink=.4,label='Absolute error m')
        fig.colorbar(im2,ax=ax[:,2:3],shrink=.4,label='Local delta error m')
        fig.suptitle(f'Validation {i}: {row["crop"]} / {row["view"]} / {Path(row["image_relpath"]).name}; A=relative trained B=outdoor original C=outdoor trained')
        path=figdir/f'comparison_{i:03d}_{row["crop"]}_{row["view"]}.png'; assert not path.exists(); fig.savefig(path,dpi=130); plt.close(fig)
        # Separate zoom with actual RGB/GT, because no plant-ground labels are available.
        f,a=plt.subplots(1,5,figsize=(18,4),constrained_layout=True)
        a[0].imshow(rgb[roi]); a[0].set_title('RGB error-selected ROI')
        for k,(label,depth) in enumerate([('GT',gt),*list(predictions.items())],1): a[k].imshow(depth[roi],cmap='viridis',vmin=depth_min,vmax=depth_max); a[k].set_title(label)
        for z in a:z.axis('off')
        f.savefig(figdir/f'roi_{i:03d}.png',dpi=150); plt.close(f)
        f,a=plt.subplots(1,4,figsize=(18,4),constrained_layout=True)
        for k,(label,p) in enumerate([('GT',gt),*list(predictions.items())]):
            delta=p[:,14:]-p[:,:-14]
            a[k].imshow(np.where(dv,delta,np.nan)[roi],cmap='coolwarm',vmin=-.2,vmax=.2)
            a[k].set_title(f'{label} signed dx14, common ±0.2m');a[k].axis('off')
        f.savefig(figdir/f'signed_delta_roi_{i:03d}.png',dpi=150);plt.close(f)
        f,a=plt.subplots(1,2,figsize=(14,4),constrained_layout=True)
        line_y=yy+256; xx_range=np.arange(xx,xx+512)
        for label,p in [('GT',gt),*list(predictions.items())]:
            z=p[line_y,xx:xx+512].astype(float);z[~valid[line_y,xx:xx+512]]=np.nan
            a[0].plot(xx_range,z,label=label,linewidth=1)
            a[1].plot(xx_range,z-np.nanmedian(z),label=label,linewidth=1)
        a[0].set_title('Raw Z-depth transect (m)');a[1].set_title('Median offset removed ONLY for display (m)')
        for axis in a:axis.legend();axis.set_xlabel('Original-image x pixel');axis.grid(alpha=.2)
        f.savefig(figdir/f'transect_{i:03d}.png',dpi=150);plt.close(f)
        chosen.append({'index':i,'image_relpath':row['image_relpath'],'crop':row['crop'],'view':row['view'],'roi':[yy,xx,512,512],'display_depth_min_m':depth_min,'display_depth_max_m':depth_max,'selection':'worst A minus B local MAE per crop, plus direction worst and one best-local example','figure':str(path)})
        print('FIGURE',path,flush=True)
    write_new(out/'figure_selection.json',chosen)

if __name__=='__main__': main()
