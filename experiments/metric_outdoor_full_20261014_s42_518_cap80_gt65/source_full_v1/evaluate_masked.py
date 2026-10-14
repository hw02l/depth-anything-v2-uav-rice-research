"""Evaluate validation with a GT mask independent of model output ceiling."""
import argparse,hashlib,json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForDepthEstimation
from data import read_manifest,UAVDepthDataset
from metrics import evaluate_model

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--rgb-root',type=Path,required=True)
    p.add_argument('--depth-root',type=Path,required=True)
    p.add_argument('--gt-max-depth',type=float,required=True)
    p.add_argument('--image-size',type=int,default=518)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() or a.gt_max_depth<=0:p.error('Existing output protected; GT limit must be positive')
    rows=read_manifest(a.manifest)
    if any(r['split']!='validation' for r in rows):p.error('This experiment evaluator accepts validation only')
    model=AutoModelForDepthEstimation.from_pretrained(a.model,local_files_only=True)
    cap=float(model.head.max_depth)
    assert cap==float(model.config.max_depth)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    loader=DataLoader(UAVDepthDataset(rows,a.rgb_root,a.depth_root,image_size=a.image_size,training=False),batch_size=1,num_workers=a.workers,pin_memory=device.type=='cuda')
    result={'manifest_sha256':hashlib.sha256(a.manifest.read_bytes()).hexdigest(),'model':str(a.model),'model_output_cap_m':cap,'gt_mask_max_m':a.gt_max_depth,'split':'validation','pairs':len(rows),'device':str(device),'metrics':evaluate_model(model,loader,device,a.gt_max_depth),'prediction_rescaled':False}
    assert model.head.max_depth==cap
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
