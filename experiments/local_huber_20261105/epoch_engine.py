"""Epoch-boundary training engine. Experimental; production integration pending."""
import hashlib
import json
import math
import os
import random
import uuid
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader


def signature(config):
    return hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def save_epoch(directory, state):
    directory=Path(directory)
    final=directory / f"epoch_{state['epoch']:04d}.pt"
    if final.exists(): raise FileExistsError(final)
    temporary=directory / ('.'+final.name+'.'+uuid.uuid4().hex+'.tmp')
    with temporary.open('xb') as f:
        torch.save(state,f); f.flush(); os.fsync(f.fileno())
    os.replace(temporary,final)
    return final


def run_epochs(model, dataset, step_fn, evaluate_fn, optimizer, scheduler, scaler,
               directory, config, *, resume=False, stop_after=None, on_epoch=None):
    """Run to epoch boundary; each durable checkpoint contains metrics and RNG.

    step_fn(model, sample, epoch) -> scalar tensor. evaluate_fn(model, epoch)
    returns JSON-serializable metrics. Caller must bind data/code/init hashes into
    config, and is responsible for verifying those hashes before creating model.
    """
    directory=Path(directory); sig=signature(config)
    if resume:
        if not directory.is_dir(): raise FileNotFoundError(directory)
        files=sorted(directory.glob('epoch_[0-9][0-9][0-9][0-9].pt'))
        if not files: raise ValueError('no completed epoch checkpoint')
        state=torch.load(files[-1],map_location='cpu',weights_only=False)
        if state['signature'] != sig: raise ValueError('resume signature mismatch')
        model.load_state_dict(state['model'],strict=True)
        optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler']); scaler.load_state_dict(state['scaler'])
        random.setstate(state['rng_python']); np.random.set_state(state['rng_numpy']); torch.set_rng_state(state['rng_torch'])
        if state['rng_cuda'] is not None: torch.cuda.set_rng_state_all(state['rng_cuda'])
        completed=state['epoch']; global_step=state['global_step']
    else:
        directory.mkdir(parents=True,exist_ok=False)
        random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(config['seed'])
        completed=0; global_step=0
    total=config['epochs']; accumulation=config['accumulation']
    if completed > total or not len(dataset): raise ValueError('invalid epoch or empty dataset')
    for epoch in range(completed+1,total+1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        # epoch seed also controls dataset transforms, independent of prior resume.
        random.seed(config['seed']+epoch-1); np.random.seed(config['seed']+epoch-1)
        generator=torch.Generator().manual_seed(config['seed']+epoch-1)
        loader=DataLoader(dataset,batch_size=None,shuffle=True,num_workers=0,generator=generator)
        steps=0; skipped=0; loss_sum=0.0; silog_sum=0.0; local_sum=0.0; component_count=0
        for i,sample in enumerate(loader):
            remaining=min(accumulation,len(loader)-(i//accumulation)*accumulation)
            value=step_fn(model,sample,epoch)
            if isinstance(value,tuple):
                loss,silog,local,_counts=value
                silog_sum+=float(silog.detach()); local_sum+=float(local.detach()); component_count+=1
            else: loss=value
            if not torch.isfinite(loss): raise FloatingPointError('nonfinite loss')
            scaler.scale(loss/remaining).backward(); loss_sum+=float(loss.detach())
            if (i+1)%accumulation==0 or i+1==len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=not scaler.is_enabled())
                old_scale=scaler.get_scale(); scaler.step(optimizer); scaler.update()
                if scaler.get_scale()>=old_scale:
                    scheduler.step(); steps+=1; global_step+=1
                else: skipped+=1
                optimizer.zero_grad(set_to_none=True)
        model.eval()
        with torch.no_grad(): metrics=evaluate_fn(model,epoch)
        json.dumps(metrics,allow_nan=False)
        state={'epoch':epoch,'global_step':global_step,'signature':sig,'config':config,
          'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),
          'rng_python':random.getstate(),'rng_numpy':np.random.get_state(),'rng_torch':torch.get_rng_state(),
          'rng_cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
          'train_loss':loss_sum/len(loader),'train_silog':silog_sum/component_count if component_count else None,
          'train_local':local_sum/component_count if component_count else None,'updates':steps,'amp_skips':skipped,'validation':metrics}
        save_epoch(directory,state)
        if on_epoch is not None: on_epoch(state)
        if stop_after is not None and epoch>=stop_after: break
    return state
