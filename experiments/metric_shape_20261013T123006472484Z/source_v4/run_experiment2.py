"""Authorized one-epoch experiment only; no HF download or old-file writes."""
from pathlib import Path
import argparse, csv, hashlib, json, os, re, shutil, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor,as_completed
import torch, numpy as np
from transformers import AutoModelForDepthEstimation
from data import UAVDepthDataset, read_manifest
from runtime_state import dataset_status

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def newjson(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--project',type=Path,required=True); ap.add_argument('--work',type=Path,required=True); ap.add_argument('--run-trial',action='store_true'); args=ap.parse_args()
    assert args.run_trial,'Training gate is OFF'
    assert torch.cuda.is_available(),'Free CUDA GPU unavailable; no training'
    project=args.project; work=args.work; source=Path(__file__).parent
    plan=project/'uav_metric_finetuning/config/verified_plan'
    rgb=project/'data/uav3dcrop_rgb'; depth=project/'data/uav3dcrop_depth'
    lr=Path('/content/uav3dcrop/rgb'); ld=Path('/content/uav3dcrop/depth')
    assert sha(plan/'train.csv')=='13ee3432515250eff50eeb5238dfc178c8853239074eb8cbbea2b510272b0602'
    assert sha(plan/'validation.csv')=='96c4eab0eb0075ac141ff9b17c22a45734f242f7cc895f8eaf064a4f14cc871e'
    status=dataset_status(plan,rgb,depth,['train','validation']); assert status['complete'] and status['planned_bytes']==7850330776
    run=project/'runs/metric_outdoor_trial2_20261013_s42_518_cap80_gt65'; assert not run.exists(),'Old run preserved; do not rerun pipeline'
    anchors={}
    for lp in (project/'work/uav_metric_finetuning').glob('download*20261013T090441609126Z.log'):
        for line in lp.read_text().splitlines():
            m=re.match(r'^\[\d+/\d+\] (\S+) (\S+) \(([^,]+), ([^)]+)\)$',line)
            if m:
                scene,filename,ih,dh=m.groups()
                for rel,h in [(f'{scene}/images/{filename}',ih),(f'{scene}_depth/{Path(filename).stem}_depth.tif',dh)]:
                    if re.fullmatch('[0-9a-f]{64}',h):anchors[rel]=h
    rows=[]
    for split in ['train','validation']:
        with (plan/f'{split}.csv').open(encoding='utf-8-sig',newline='') as f:rows.extend(csv.DictReader(f))
    assert len(rows)==512
    files=[]
    for row in rows:
        for old,new,key,sizekey in [(rgb,lr,'image_relpath','image_size_bytes'),(depth,ld,'depth_relpath','depth_size_bytes')]:
            files.append((old/row[key],new/row[key],int(row[sizekey]),anchors.get(row[key])))
    assert shutil.disk_usage('/content').free>sum(n for _,_,n,_ in files)*1.1
    def copy_one(entry):
        src,dst,size,anchor=entry; before=src.stat(); assert before.st_size==size
        dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists():
            assert dst.stat().st_size==size and sha(dst)==(anchor or sha(src)),f'Existing local file mismatch: {dst}'
            return {'path':str(dst),'status':'reused','sha256':anchor or sha(dst)}
        tmp=dst.with_name(dst.name+'.experiment2.copying'); assert not tmp.exists()
        h=hashlib.sha256()
        with src.open('rb') as f,tmp.open('xb') as g:
            for block in iter(lambda:f.read(1024*1024),b''):g.write(block);h.update(block)
        assert tmp.stat().st_size==size and (not anchor or h.hexdigest()==anchor),'Content mismatch; preserved temporary for inspection'
        after=src.stat(); assert before.st_size==after.st_size and before.st_mtime_ns==after.st_mtime_ns
        assert not dst.exists();tmp.replace(dst)
        return {'path':str(dst),'status':'copied','sha256':h.hexdigest(),'cpu_log_anchor':bool(anchor)}
    copied=[]; started=time.monotonic()
    print('STAGE_START 1024 files, four streams; original Drive retained',flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(copy_one,e) for e in files]):
            copied.append(future.result())
            if len(copied)%64==0:print('STAGE',len(copied),'/1024 elapsed',round(time.monotonic()-started),flush=True)
    newjson(work/'ssd_integrity.json',{'status':'pass','files':copied,'elapsed_seconds':time.monotonic()-started,'dataset':dataset_status(plan,lr,ld,['train','validation'])})
    assert dataset_status(plan,lr,ld,['train','validation'])['complete']
    metric=project/'model_cache/models--depth-anything--Depth-Anything-V2-Metric-Outdoor-Small-hf/snapshots/fd2c22027eaf20374204f14099b8341e1925ad39'
    assert (metric/'model.safetensors').exists()
    torch.set_num_threads(2)
    model=AutoModelForDepthEstimation.from_pretrained(metric,local_files_only=True).cuda().eval()
    assert model.head.max_depth==model.config.max_depth==80
    item=UAVDepthDataset(read_manifest(plan/'validation.csv'),lr,ld,image_size=518,training=False)[0]
    with torch.inference_mode():
        x=item['pixel_values'][None].cuda(); y80=model(pixel_values=x).predicted_depth.float()
        model.config.max_depth=65; y_config_only=model(pixel_values=x).predicted_depth.float()
        model.head.max_depth=65; y65=model(pixel_values=x).predicted_depth.float()
    error=float((y65-y80*(65/80)).abs().max().cpu()); config_error=float((y_config_only-y80).abs().max().cpu())
    assert error<1e-4 and config_error==0
    newjson(work/'output_cap_probe.json',{'image_relpath':read_manifest(plan/'validation.csv')[0]['image_relpath'],'max_abs_error_to_65_over_80':error,'config_only_prediction_change_max_abs_m':config_error,'expected_scale':65/80,'prediction_mean_80_m':float(y80.mean().cpu()),'prediction_mean_65_m':float(y65.mean().cpu()),'actual_median_ratio':float((y65/y80).median().cpu()),'forward_source':'DepthAnythingDepthEstimationHead.forward: activation2(predicted_depth) * self.max_depth','model_output_ceiling_m':80,'gt_valid_upper_bound_m':65})
    print('OUTPUT_CAP_PROBE_PASS scale=',65/80,'error=',error,flush=True)
    del model,x,y80,y65,y_config_only,item; torch.cuda.empty_cache()
    preflight=work/'metric_outdoor_preflight.json'
    common=['--initialization','metric_outdoor','--metric-model',str(metric),'--max-depth','65','--model-max-depth','80','--image-size','518','--batch-size','1','--amp']
    subprocess.run([sys.executable,'-u',str(source/'memory_preflight.py'),*common,'--output',str(preflight)],check=True)
    assert json.loads(preflight.read_text())['status']=='pass'
    train_start=time.monotonic()
    subprocess.run([sys.executable,'-u',str(source/'train.py'),*common,'--train-manifest',str(plan/'train.csv'),'--val-manifest',str(plan/'validation.csv'),'--rgb-root',str(lr),'--depth-root',str(ld),'--epochs','1','--run-kind','trial','--grad-accumulation','4','--seed','42','--workers','2','--encoder-lr','5e-6','--save-every-epochs','1','--keep-recent-checkpoints','0','--memory-preflight-report',str(preflight),'--output-dir',str(run)],check=True)
    assert json.loads((run/'run_status.json').read_text())['status']=='completed'
    checkpoints={}
    for cp in run.glob('checkpoint_epoch_*'):
        marker=json.loads((cp/'complete.json').read_text())
        assert all(sha(cp/n)==h for n,h in marker['file_sha256'].items())
        checkpoints[cp.name]=marker
    newjson(work/'trial_complete.json',{'run':str(run),'training_total_seconds_including_startup_save':time.monotonic()-train_start,'checkpoints_verified':checkpoints,'model_output_ceiling_m':80,'gt_mask_max_m':65,'epochs':1,'test_used':False})
    spec={'manifest':str(plan/'validation.csv'),'validation_sha256':sha(plan/'validation.csv'),'rgb_root':str(lr),'depth_root':str(ld),'output':str(work/'analysis'),'models':{'A':str(project/'runs/relative_encoder_trial_20261013_s42_518/checkpoint_epoch_0001'),'B':str(metric),'C':str(run/'checkpoint_epoch_0001')},'model_caps':{'A':65,'B':80,'C':80}}
    newjson(work/'analysis_spec.json',spec)
    subprocess.run([sys.executable,'-u',str(source/'shape_analysis.py'),'--spec',str(work/'analysis_spec.json'),'--phase','infer'],check=True)
    print('ALL_GPU_WORK_COMPLETE. Persisted training and prediction cache. Switch to CPU for analysis.',flush=True)

if __name__=='__main__':main()
