#!/usr/bin/env python3
import argparse,csv,hashlib,json,subprocess,sys,time
from pathlib import Path
import cv2,numpy as np,tifffile,torch
def sha256(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for block in iter(lambda:f.read(4194304),b""): h.update(block)
 return h.hexdigest()
def gpu():
 return subprocess.check_output(["nvidia-smi","--query-gpu=name,utilization.gpu,memory.used,memory.free","--format=csv,noheader"],text=True).strip()
def main():
 ap=argparse.ArgumentParser()
 for k in ("repo-root","base-model-dir","c0-checkpoint","e3-checkpoint","validation-manifest","image-inventory","visual-indices","pair-manifest","rgb-root","depth-root","existing-summary","existing-predictions","epoch20-b0-cache-dir","epoch20-b3-cache-dir","baseline-b3-per-image","cache-audit-json","output"):
  ap.add_argument("--"+k,type=Path,required=True)
 a=ap.parse_args(); repo=a.repo_root.resolve(); exp=repo/"experiments/best_model_inference_combination_20261122"
 sys.path[:0]=[str(repo/"src"),str(exp)]
 import run_b3 as r
 import make_fixed8_comparison as f
 indices=sorted(int(x["image_index"]) for x in csv.DictReader(a.visual_indices.open(encoding="utf-8-sig",newline="")))
 if len(indices)!=8: raise ValueError("not the frozen 8 indices")
 idx=indices[0]
 inventory=list(csv.DictReader(a.image_inventory.open(encoding="utf-8-sig",newline="")))
 rows=r.read_validation_rows(a.validation_manifest,inventory); row=rows[idx]
 rp=a.rgb_root/row["image_relpath"]; gp=a.depth_root/row["depth_relpath"]
 bgr=cv2.imread(str(rp),cv2.IMREAD_COLOR)
 if bgr is None: raise FileNotFoundError(rp)
 rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB); gt=np.asarray(tifffile.imread(gp),dtype=np.float32)
 if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[:2]!=gt.shape or rgb.shape[2]!=3: raise ValueError("RGB/GT mismatch")
 if sha256(rp)!=inventory[idx]["rgb_sha256"] or sha256(gp)!=inventory[idx]["gt_sha256"]: raise ValueError("input SHA mismatch")
 valid=r.m.valid_gt(gt)
 if not valid.any() or rgb.min()==rgb.max(): raise ValueError("invalid mask or constant RGB")
 summary={(x["condition"],int(x["image_index"])):x for x in csv.DictReader(a.existing_summary.open(encoding="utf-8-sig",newline=""))}
 audit=json.load(a.cache_audit_json.open(encoding="utf-8"))
 if not audit.get("passed") or audit.get("cache_maps_checked")!=32 or audit.get("fixed_indices")!=indices: raise ValueError("cache audit mismatch")
 before=gpu(); preds={}; loads={}
 for group,condition,checkpoint in (("C0","C0-P0",a.c0_checkpoint),("E3","E3-B0",a.e3_checkpoint)):
  model,loads[group]=r.load_model(group,a.base_model_dir,checkpoint,"cuda")
  p=r.infer_one(model,rgb,"cuda")
  if p.shape!=gt.shape or not np.isfinite(p).all(): raise ValueError("bad forward "+condition)
  preds[condition]=p; del model; torch.cuda.empty_cache()
 cache={
  "epoch20-B0":a.epoch20_b0_cache_dir/f"prediction_{idx:03d}.npy",
  "epoch20-B3":a.epoch20_b3_cache_dir/f"prediction_{idx:03d}.npy",
  "C0-B3":a.existing_predictions/"C0"/f"prediction_{idx:03d}.npz",
  "E3-B3":a.existing_predictions/"E3"/f"prediction_{idx:03d}.npz"}
 for c,p in cache.items():
  if c.startswith("epoch20-"):
   pred=np.load(p,allow_pickle=False)
   if pred.dtype!=np.float32 or pred.shape!=gt.shape or not np.isfinite(pred).all(): raise ValueError("bad cache "+c)
  else: pred=f.load_prediction(p,gt.shape)
  preds[c]=pred
 checks=[]
 for c in ("epoch20-B0","epoch20-B3","C0-P0","C0-B3","E3-B0","E3-B3"):
  ref=summary.get((c,idx))
  if ref is None: raise ValueError("missing summary "+c)
  rmse=r.absolute_stats(preds[c],gt,valid)["rmse_m"]; delta=abs(rmse-float(ref["rmse"]))
  if delta>1e-5: raise ValueError(f"{c} RMSE mismatch {delta}")
  checks.append({"condition":c,"rmse_m":rmse,"reference_rmse_m":float(ref["rmse"]),"difference_m":delta})
 legacy={int(x["image_index"]):x for x in csv.DictReader(a.baseline_b3_per_image.open(encoding="utf-8-sig",newline=""))}[idx]
 local=r.dense14_stats(preds["epoch20-B3"],gt,valid)["full14_local_mae_m"]
 if abs(local-float(legacy["local_depth_delta_mae_m"]))>1e-5: raise ValueError("B3 full14 legacy mismatch")
 out={"artifact":"fixed8_single_image_smoke_gate","status":"passed","utc_finished":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"image_index":idx,"image_relpath":row["image_relpath"],"depth_relpath":row["depth_relpath"],"rgb_sha256":sha256(rp),"gt_sha256":sha256(gp),"rgb_dtype":str(rgb.dtype),"rgb_range":[int(rgb.min()),int(rgb.max())],"rgb_shape":list(rgb.shape),"gt_shape":list(gt.shape),"valid_gt_pixels":int(valid.sum()),"checkpoint_sha256":{"epoch20":r.BASE_MODEL_SHA256,"C0":r.CHECKPOINT_SHA256["C0"],"E3":r.CHECKPOINT_SHA256["E3"]},"strict_load_audits":loads,"condition_rmse_checks":checks,"epoch20_b3_cached_full14_mae_m":local,"epoch20_b3_legacy_full14_mae_m":float(legacy["local_depth_delta_mae_m"]),"gpu_before":before,"gpu_after":gpu(),"training":False,"independent_test319":False,"command_line":sys.argv}
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open("x",encoding="utf-8") as s: json.dump(out,s,indent=2,ensure_ascii=False); s.write("\n")
 print(json.dumps(out,indent=2,ensure_ascii=False))
if __name__=="__main__": main()
