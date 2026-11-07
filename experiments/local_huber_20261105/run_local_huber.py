"""Run the fixed local-Huber continuation groups with audited GPU gates."""
import argparse
import csv
from types import SimpleNamespace
import json
import math
import sys
import time
from pathlib import Path
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'src'))
from uav_metric_research import multiscale as m
from data_local import UAVDepthDataset,read_manifest
from epoch_engine import run_epochs
from initialization import sha256,EPOCH20_MODEL_SHA256,EPOCH20_CONFIG_SHA256
from train_step import step_batch
from evaluation_b0 import evaluate_b0


class SyntheticModel(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.backbone=torch.nn.Conv2d(3,4,1); self.head=torch.nn.Conv2d(4,1,1)
    def forward(self,pixel_values):
        return SimpleNamespace(predicted_depth=torch.nn.functional.softplus(self.head(self.backbone(pixel_values)))[:,0]+1)


def synthetic_run(args):
    # Isolated synthetic run; never use this output as research evidence.
    torch.manual_seed(42); model=SyntheticModel()
    yy,xx=torch.meshgrid(torch.arange(60),torch.arange(60),indexing='ij')
    gt=1.+xx.float()/100+yy.float()/200
    dataset=[{'pixel_values':torch.stack((gt/3,gt/4,gt/5)), 'depth':gt, 'original_depth':gt,
              'geometry':{'original_hw':(60,60),'resized_hw':(60,60),'crop_yx':(0,0),'crop_hw':(60,60),'flipped':False},
              'image_relpath':f'synthetic-{i}'} for i in range(4)]
    weight={'E1':0.,'E2':.05,'E3':.20}[args.group]
    opt=torch.optim.AdamW([{'params':model.backbone.parameters(),'lr':2e-6},{'params':model.head.parameters(),'lr':2e-5}],weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.PolynomialLR(opt,total_iters=5,power=.9)
    scaler=torch.amp.GradScaler('cpu',enabled=False)
    silog=m.frozen('losses').silog_loss
    def step(net,sample,epoch):
        return step_batch(net,sample,silog,'cpu',epoch,weight,candidates_per_axis=8)[0]
    def evaluate(net,epoch):
        with torch.no_grad():
            pred=net(dataset[0]['pixel_values'][None]).predicted_depth[0]
            return {'epoch':epoch,'group':args.group,'image_count':1,'mae_m':float((pred-gt).abs().mean()),'kind':'synthetic_only'}
    config={'seed':42,'epochs':5,'accumulation':4,'group':args.group,'lambda':weight,'mode':'synthetic_only'}
    state=run_epochs(model,dataset,step,evaluate,opt,scheduler,scaler,args.output,config,resume=args.resume,stop_after=args.stop_after)
    rows=[]
    for path in sorted(args.output.glob('epoch_[0-9][0-9][0-9][0-9].pt')):
        ck=torch.load(path,map_location='cpu',weights_only=False); rows.append(ck['validation'])
    for suffix in ('json','csv'):
        path=args.output/('synthetic_metrics.'+suffix); tmp=path.with_suffix(path.suffix+'.tmp')
        with tmp.open('w',encoding='utf-8',newline='') as f:
            if suffix=='json': json.dump(rows,f,indent=2,allow_nan=False)
            else:
                writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        tmp.replace(path)
    print(json.dumps({'mode':'synthetic_only','group':args.group,'completed_epoch':state['epoch'],'global_step':state['global_step']}))




def real_data_smoke(args, train, val):
    """CPU-only toy model over one genuine train and validation image."""
    if args.device != 'cpu' or args.group not in ('E1','E2','E3') or args.amp:
        raise ValueError('real-data smoke is CPU-only, non-AMP, group E1-E3')
    torch.set_num_threads(2); torch.manual_seed(42)
    model=SyntheticModel(); data=UAVDepthDataset(train[:1],args.rgb_root,args.depth_root,image_size=518,training=True)
    weight={'E1':0.,'E2':.05,'E3':.20}[args.group]
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.PolynomialLR(optimizer,total_iters=1,power=.9)
    scaler=torch.amp.GradScaler('cpu',enabled=False)
    pairs=ROOT/'experiments/multiscale_tile_inference_20261016/pair_manifest.json'
    def step(net,sample,epoch):
        return step_batch(net,sample,m.frozen('losses').silog_loss,'cpu',epoch,weight,candidates_per_axis=8192)[0]
    def evaluate(net,epoch):
        return evaluate_b0(net,val,args.rgb_root,args.depth_root,'cpu',pairs,scales=(14,),indices=[0])
    cfg={'group':args.group,'seed':42,'epochs':1,'accumulation':4,'mode':'real_image_toy_model_smoke',
         'train_manifest_sha256':sha256(args.train_manifest),'val_manifest_sha256':sha256(args.val_manifest),'pair_sha256':sha256(pairs)}
    result=run_epochs(model,data,step,evaluate,optimizer,scheduler,scaler,args.output,cfg,resume=args.resume)
    data_out={'mode':'real_image_toy_model_smoke','epoch':result['epoch'],'group':args.group,'validation':result['validation']}
    tmp=args.output/'smoke_validation.json.tmp'; tmp.write_text(json.dumps(data_out,indent=2,allow_nan=False),encoding='utf-8'); tmp.replace(args.output/'smoke_validation.json')
    absolute=result['validation']['absolute']; path=args.output/'smoke_validation.csv'; tmp=path.with_suffix('.csv.tmp')
    with tmp.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['mode','group','epoch',*absolute]); writer.writeheader()
        writer.writerow({'mode':cfg['mode'],'group':args.group,'epoch':result['epoch'],**absolute})
    tmp.replace(path)
    print(json.dumps({'mode':cfg['mode'],'epoch':result['epoch'],'updates':result['global_step']}))
    return result



def export_epoch_metrics(directory, state):
    """Rebuild per-epoch files from durable checkpoints without duplicate rows."""
    directory=Path(directory); epoch=int(state['epoch']); result=state['validation']
    summary={'epoch':epoch,'group':state['config']['group'],'global_step':state['global_step'],
             'updates':state['updates'],'amp_skips':state['amp_skips'],
             'train_loss':state['train_loss'],'train_silog':state.get('train_silog'),
             'train_local':state.get('train_local'),**result['absolute']}
    large=next(r for r in result['sampled'] if r['scope']=='all' and r['stride_px']==14 and r['group']=='gt_ge20cm')
    for key in ('pair_count','delta_mae_m','amplitude_ratio','sign_match_rate'):
        summary['sample_gt_ge20cm_'+key]=large[key]
    data={'summary':summary,'validation':result,'checkpoint':f'epoch_{epoch:04d}.pt'}
    for suffix in ('json','csv'):
        final=directory/f'validation_epoch_{epoch:04d}.{suffix}'
        temp=directory/f'.validation_epoch_{epoch:04d}.{suffix}.tmp'
        with temp.open('w',encoding='utf-8',newline='') as stream:
            if suffix=='json': json.dump(data,stream,ensure_ascii=False,indent=2,allow_nan=False)
            else:
                writer=csv.DictWriter(stream,fieldnames=list(summary)); writer.writeheader(); writer.writerow(summary)
        temp.replace(final)


def _model_from_initial_checkpoint(args):
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForDepthEstimation
    model_config=AutoConfig.from_pretrained(args.checkpoint,local_files_only=True)
    if model_config.max_depth!=80 or model_config.depth_estimation_type!='metric':
        raise ValueError('checkpoint configuration is not metric 80m')
    model=AutoModelForDepthEstimation.from_config(model_config)
    audit=m.strict_attention_state(model,load_file(str(args.checkpoint/'model.safetensors'),device='cpu'))
    model.to(args.device)
    return model,audit


def _atomic_json(path, payload):
    path=Path(path); temp=path.with_name('.'+path.name+'.tmp')
    with temp.open('x',encoding='utf-8') as stream:
        json.dump(payload,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.flush()
    temp.replace(path)


def _flatten_e0(group, validation):
    row={'group':group,'epoch':20,'image_count':len(validation['per_image']),**validation['absolute']}
    for item in validation['sampled']:
        if item['scope']=='all' and item['stride_px']==14 and item['group']=='gt_ge20cm':
            for key,value in item.items():
                if key not in ('scope','stride_px','group'): row['sampled_gt_ge20cm_'+key]=value
    return row


def _check_preflight(path):
    receipt=json.loads(Path(path).read_text(encoding='utf-8'))
    age=time.time()-float(receipt.get('unix_time',0))
    if receipt.get('status')!='available' or receipt.get('hostname','').lower()!='gpgpu4' or not 0<=age<=300:
        raise RuntimeError('GPU receipt missing, unavailable, wrong host, or older than 5 minutes')
    if not str(receipt.get('coordination_note','')).strip():
        raise RuntimeError('GPU receipt has no current coordination note')
    records=receipt.get('commands',[])
    compute=[r for r in records if r.get('command',[])[:1]==['nvidia-smi'] and '--query-compute-apps=pid,process_name,used_memory' in r.get('command',[])]
    if len(compute)!=1 or compute[0].get('returncode')!=0 or compute[0].get('stdout','').strip():
        raise RuntimeError('GPGPU4 has a compute process or the live GPU check is missing')
    if not any(r.get('command',[])==['w'] and r.get('returncode')==0 for r in records):
        raise RuntimeError('GPU receipt lacks a successful live login-session check')
    return {'receipt_sha256':sha256(path),'age_seconds':age,'coordination_note':receipt['coordination_note']}


def _check_e0_proof(path):
    proof_path=Path(path)
    proof=json.loads(proof_path.read_text(encoding='utf-8'))
    companion=json.loads((proof_path.parent/'validation_epoch_0020.json').read_text(encoding='utf-8'))
    if companion.get('initial_model_sha256')!=EPOCH20_MODEL_SHA256 or companion.get('validation',{}).get('absolute')!=proof.get('actual'):
        raise RuntimeError('E0 proof does not match its durable validation JSON')
    if proof.get('passed') is not True or proof.get('initial_model_sha256')!=EPOCH20_MODEL_SHA256:
        raise RuntimeError('E0 baseline proof failed or is for another checkpoint')
    checks=proof.get('checks',{})
    if not checks or not all(value is True for value in checks.values()):
        raise RuntimeError('E0 baseline proof has missing or failed checks')
    return {'proof_sha256':sha256(path),'initial_model_sha256':proof['initial_model_sha256']}


def evaluate_initial(args, config, val, load_audit, model, receipt):
    if args.resume or args.output.exists():
        raise FileExistsError('E0 evaluation requires a new, empty output directory')
    args.output.mkdir(parents=True,exist_ok=False)
    pairs=ROOT/'experiments/multiscale_tile_inference_20261016/pair_manifest.json'
    validation=evaluate_b0(model,val,args.rgb_root,args.depth_root,args.device,pairs)
    actual=validation['absolute']
    large=next(x for x in validation['sampled'] if x['scope']=='all' and x['stride_px']==14 and x['group']=='gt_ge20cm')
    reference=config['e0_reference']
    checks={
        'rmse_m':abs(actual['rmse_m']-reference['rmse_m'])<=reference['tolerance_m'],
        'mae_m':abs(actual['mae_m']-reference['mae_m'])<=reference['tolerance_m'],
        'abs_rel':abs(actual['abs_rel']-reference['abs_rel'])<=reference['tolerance_abs_rel'],
        'pearson_r':abs(actual['pearson_r']-reference['pearson_r'])<=reference['tolerance_pearson'],
        'full14_pair_count':actual['local_pair_count']==reference['full14_pair_count'],
        'full14_mae_m':abs(actual['local_depth_delta_mae_m']-reference['full14_mae_m'])<=reference['tolerance_m'],
        'valid_pixel_count':actual['valid_pixels']==reference['valid_pixel_count'],
        'fixed_large_pair_count':large['pair_count']==reference['fixed_large_pair_count'],
        'fixed_large_mae_m':abs(large['delta_mae_m']-reference['fixed_large_mae_m'])<=reference['tolerance_m'],
        'fixed_large_amplitude_ratio':abs(large['amplitude_ratio']-reference['fixed_large_amplitude_ratio'])<=reference['tolerance_amplitude'],
        'fixed_large_sign_match_rate':abs(large['sign_match_rate']-reference['fixed_large_sign_match_rate'])<=reference['tolerance_rate'],
    }
    proof={'passed':all(checks.values()),'group':'E0','epoch':20,'images':len(val),
           'initial_model_sha256':EPOCH20_MODEL_SHA256,'config_sha256':sha256(args.config),
           'train_manifest_sha256':sha256(args.train_manifest),'validation_manifest_sha256':sha256(args.val_manifest),
           'pair_manifest_sha256':sha256(pairs),'checkpoint_load_audit':load_audit,
           'preflight_receipt':receipt,'checks':checks,'actual':actual,'fixed_gt_ge20cm_14px':large,'reference':reference}
    _atomic_json(args.output/'validation_epoch_0020.json',{'group':'E0','epoch':20,'validation':validation,
                 'initial_model_sha256':EPOCH20_MODEL_SHA256,'checkpoint_load_audit':load_audit,
                 'preflight_receipt':receipt})
    _atomic_json(args.output/'baseline_reproduction.json',proof)
    row=_flatten_e0('E0',validation)
    with (args.output/'validation_epoch_0020.csv.tmp').open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(row)); writer.writeheader(); writer.writerow(row)
    (args.output/'validation_epoch_0020.csv.tmp').replace(args.output/'validation_epoch_0020.csv')
    print(json.dumps({'group':'E0','passed':proof['passed'],'checkpoint_sha256':EPOCH20_MODEL_SHA256,
                      'metrics':actual,'baseline_proof':str(args.output/'baseline_reproduction.json')},ensure_ascii=False))
    if not proof['passed']:
        raise RuntimeError('E0 did not reproduce the fixed B0 reference; E1-E3 must remain blocked')
    return proof


def real_run(args, config, train, val):
    if args.real_data_smoke:
        return real_data_smoke(args, train, val)
    if args.device!='cuda' or not torch.cuda.is_available():
        raise RuntimeError('formal run is restricted to a verified CUDA device')
    if args.group!='E0' and not args.amp:
        raise RuntimeError('formal training requires the preflight-tested AMP mode')
    receipt=_check_preflight(args.preflight_receipt)
    if args.group!='E0':
        proof=_check_e0_proof(args.baseline_proof)
    else:
        proof=None
    model,load_audit=_model_from_initial_checkpoint(args)
    pairs=ROOT/'experiments/multiscale_tile_inference_20261016/pair_manifest.json'
    if args.group=='E0':
        return evaluate_initial(args,config,val,load_audit,model,receipt)
    dataset=UAVDepthDataset(train,args.rgb_root,args.depth_root,image_size=518,training=True)
    encoder=[p for name,p in model.named_parameters() if name.startswith('backbone.')]
    head=[p for name,p in model.named_parameters() if not name.startswith('backbone.')]
    if not encoder or not head: raise ValueError('missing backbone or head parameter group')
    optimizer=torch.optim.AdamW([{'params':encoder,'lr':2e-6},{'params':head,'lr':2e-5}],weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.PolynomialLR(optimizer,total_iters=math.ceil(len(dataset)/4)*5,power=.9)
    scaler=torch.amp.GradScaler('cuda',enabled=True)
    weight={'E1':0.,'E2':.05,'E3':.20}[args.group]
    code_hashes={p.name:sha256(p) for p in sorted(HERE.glob('*.py')) if not p.name.startswith('test_')}
    signature={'group':args.group,'seed':42,'epochs':5,'accumulation':4,'lambda':weight,
       'amp':args.amp,'config_sha256':sha256(args.config),'train_sha256':sha256(args.train_manifest),
       'val_sha256':sha256(args.val_manifest),'pair_sha256':sha256(pairs),'initial_sha256':EPOCH20_MODEL_SHA256,
       'baseline_proof_sha256':proof['proof_sha256'],
       'code_sha256':code_hashes,'torch_version':torch.__version__,'device':args.device,
       'rgb_root':str(args.rgb_root.resolve()),'depth_root':str(args.depth_root.resolve()),'load_audit':load_audit}
    silog=m.frozen('losses').silog_loss
    def step(net,sample,epoch):
        return step_batch(net,sample,silog,args.device,epoch,weight,amp=args.amp)
    def evaluate(net,epoch):
        return evaluate_b0(net,val,args.rgb_root,args.depth_root,args.device,pairs)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    receipt_record=args.output.with_name(args.output.name+f'.preflight.{time.time_ns()}.json')
    _atomic_json(receipt_record,{'receipt':receipt,'baseline_proof':proof,'group':args.group})
    result=run_epochs(model,dataset,step,evaluate,optimizer,scheduler,scaler,args.output,signature,
                      resume=args.resume,stop_after=args.stop_after,
                      on_epoch=lambda state: export_epoch_metrics(args.output,state))
    print(json.dumps({'group':args.group,'completed_epoch':result['epoch'],'global_step':result['global_step'],
                      'updates':result['updates'],'amp_skips':result['amp_skips'],'output':str(args.output)},ensure_ascii=False))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--group',choices=('E0','E1','E2','E3'),required=True)
    p.add_argument('--config',type=Path,default=HERE/'config.json')
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--train-manifest',type=Path,default=ROOT/'configs/uav3dcrop_verified/train.csv')
    p.add_argument('--val-manifest',type=Path,default=ROOT/'configs/uav3dcrop_verified/validation.csv')
    p.add_argument('--rgb-root',type=Path); p.add_argument('--depth-root',type=Path)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    p.add_argument('--resume',action='store_true'); p.add_argument('--amp',action='store_true'); p.add_argument('--dry-run',action='store_true')
    p.add_argument('--synthetic',action='store_true'); p.add_argument('--real-data-smoke',action='store_true'); p.add_argument('--stop-after',type=int)
    p.add_argument('--preflight-receipt',type=Path); p.add_argument('--baseline-proof',type=Path)
    p.add_argument('--gpu-work-root',type=Path,default=Path('/work'))
    args=p.parse_args()
    if args.synthetic:
        if args.group=='E0': p.error('synthetic mode supports E1-E3 only')
        synthetic_run(args); return
    if not all((args.rgb_root,args.depth_root)): p.error('rgb-root and depth-root are required')
    if not args.real_data_smoke and not args.checkpoint: p.error('checkpoint is required')
    config=json.loads(args.config.read_text(encoding='utf-8-sig'))
    train=read_manifest(args.train_manifest); val=m.validation_rows(args.val_manifest)
    if len(train)!=384 or any(r['split']!='train' for r in train): raise ValueError('canonical train384 required')
    if sha256(args.train_manifest)!=sha256(ROOT/'configs/uav3dcrop_verified/train.csv'): raise ValueError('train manifest differs')
    for name,expected in ([] if args.real_data_smoke else [('model.safetensors',EPOCH20_MODEL_SHA256),('config.json',EPOCH20_CONFIG_SHA256)]):
        if sha256(args.checkpoint/name)!=expected: raise ValueError('initial checkpoint mismatch: '+name)
    for row in train+val:
        for root,key in ((args.rgb_root,'image_relpath'),(args.depth_root,'depth_relpath')):
            if not (root/row[key]).is_file(): raise FileNotFoundError(row[key])
    if args.output.exists() and not args.resume: raise FileExistsError(args.output)
    if args.dry_run:
        print(json.dumps({'group':args.group,'train':len(train),'validation':len(val),'checkpoint_hash_verified':not args.real_data_smoke,'dry_run':True}))
        return
    if args.real_data_smoke:
        real_run(args,config,train,val); return
    if not args.preflight_receipt:
        p.error('formal run requires --preflight-receipt from a live GPGPU4 audit')
    gpu_root=args.gpu_work_root.resolve()
    for name,path in [('config',args.config),('checkpoint',args.checkpoint),('train manifest',args.train_manifest),
                      ('validation manifest',args.val_manifest),('RGB root',args.rgb_root),('depth root',args.depth_root),
                      ('preflight receipt',args.preflight_receipt),('output',args.output)]:
        if not path.resolve().is_relative_to(gpu_root): raise ValueError(f'{name} must reside within gpu_work: {path}')
    if args.group=='E0':
        if args.resume or args.baseline_proof: p.error('E0 is a fresh evaluation and does not accept resume/baseline proof')
    elif not args.baseline_proof:
        p.error('E1-E3 require the successful E0 baseline_reproduction.json via --baseline-proof')
    elif not args.baseline_proof.is_file():
        raise FileNotFoundError(args.baseline_proof)
    print(json.dumps({'group':args.group,'train':len(train),'validation':len(val),
                      'checkpoint_hash_verified':True,'formal_gate':'passed input checks; live gate follows'}))
    real_run(args,config,train,val)

if __name__=='__main__': main()
