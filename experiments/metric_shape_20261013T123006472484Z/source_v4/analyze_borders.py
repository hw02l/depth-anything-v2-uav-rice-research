from pathlib import Path
import json,sys,time
import numpy as np,tifffile,torch
from shape_analysis_v5 import Stats,interp
from data import read_manifest,UAVDepthDataset
from transformers import AutoModelForDepthEstimation

assert not torch.cuda.is_available()
work=Path(__file__).parent.parent
spec=json.loads((work/'analysis_spec_cpu.json').read_text())
rows=read_manifest(spec['manifest']);out=Path(spec['output']);cache=out/'prediction_cache'
stats={};start=time.monotonic()
for i,row in enumerate(rows):
    gt=np.asarray(tifffile.imread(Path(spec['depth_root'])/row['depth_relpath']),np.float32);H,W=gt.shape
    valid=np.isfinite(gt)&(gt>0)&(gt<=65)
    preds={m:np.load(cache/f'{m}_{i:03d}.npy') for m in spec['models']}
    rng=np.random.default_rng(42+i*10000+14)
    for axis in [0,1]:
        hh=H-14 if axis==0 else H;ww=W-14 if axis==1 else W
        ix=rng.choice(hh*ww,size=100000,replace=False);y=ix//ww;x=ix%ww;y1=y+(14 if axis==0 else 0);x1=x+(14 if axis==1 else 0)
        v=valid[y,x]&valid[y1,x1];y,x,y1,x1=y[v],x[v],y1[v],x1[v]
        t=(gt[y1,x1]-gt[y,x]).astype(float)
        edge=np.minimum.reduce([y,x,H-1-y,W-1-x,y1,x1,H-1-y1,W-1-x1])
        for m,pred in preds.items():
            p0,_=interp(pred,y,x,H,W);p1,_=interp(pred,y1,x1,H,W);p=p1-p0
            for label,mask in [('0-128px',edge<128),('128-512px',(edge>=128)&(edge<512)),('>=512px',edge>=512)]:
                for group,select in [(label,mask),(label+'|GT<5cm',mask&(np.abs(t)<.05)),(label+'|'+row['view'],mask)]:
                    stats.setdefault(m+'|'+group,Stats()).update(t[select],p[select])
    if (i+1)%32==0:print('BORDER_ANALYZE',i+1,'/128',flush=True)
result={'sample_method':'Exactly same seed/coordinates as main stride14 analysis','zones':'minimum distance of either endpoint from original image edge','stats':{k:v.result() for k,v in stats.items()},'elapsed_seconds':time.monotonic()-start}
# One-frame CPU diagnostic: random metric head before trial vs trained head.
old_run=Path(spec['models']['A']).parent
model=AutoModelForDepthEstimation.from_pretrained(old_run/'checkpoint_epoch_0000',local_files_only=True).cpu().eval()
dataset=UAVDepthDataset(rows,spec['rgb_root'],spec['depth_root'],image_size=518,training=False)
item=dataset[7]
with torch.inference_mode():p0=model(pixel_values=item['pixel_values'][None]).predicted_depth[0].float().numpy()
with (out/'initial_A0_frame007.npy').open('xb') as f:np.save(f,p0)
p1=np.load(cache/'A_007.npy');h,w=p0.shape
yy,xx=np.indices((h,w));ed=np.minimum.reduce([yy,xx,h-1-yy,w-1-xx])
result['initial_head_probe']={'validation_image':rows[7]['image_relpath'],'model_cap_m':model.head.max_depth,'native_prediction_shape':list(p0.shape),'before_min_m':float(p0.min()),'before_max_m':float(p0.max()),'before_mean_m':float(p0.mean()),'before_border_17px_mean_m':float(p0[ed<17].mean()),'before_interior_67px_mean_m':float(p0[ed>=67].mean()),'trained_border_17px_mean_m':float(p1[ed<17].mean()),'trained_interior_67px_mean_m':float(p1[ed>=67].mean()),'scope':'single frame descriptive initialization diagnostic; not an architecture causality test'}
with (out/'border_analysis.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
print('BORDER_ANALYSIS_COMPLETE',json.dumps(result['initial_head_probe']),flush=True)
