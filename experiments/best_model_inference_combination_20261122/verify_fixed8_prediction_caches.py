from __future__ import annotations
import argparse, csv, hashlib, json, sys, time
from pathlib import Path
import cv2
import numpy as np
import tifffile

HERE = Path(__file__).resolve()

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def value(row, *keys):
    for key in keys:
        v = row.get(key)
        if v not in (None, ""):
            return float(v)
    return None

def audit(actual, reference, tol=1e-5):
    if actual is None or reference is None:
        return {"actual": actual, "reference": reference, "difference": None, "passed": actual is None and reference is None}
    d = abs(float(actual) - float(reference))
    return {"actual": float(actual), "reference": float(reference), "difference": d, "passed": d <= tol}

def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def main():
    ap=argparse.ArgumentParser()
    for name in ("repo-root","validation-manifest","image-inventory","pair-manifest","visual-indices",
                 "rgb-root","depth-root","existing-predictions","existing-summary",
                 "epoch20-b0-cache-dir","epoch20-b3-cache-dir","baseline-b0-per-image","baseline-b3-per-image","output"):
        ap.add_argument("--"+name, required=True, type=Path)
    a=ap.parse_args()
    sys.path.insert(0,str(a.repo_root/"experiments/best_model_inference_combination_20261122"))
    import run_b3 as runner
    if sha256(a.image_inventory) != runner.INPUT_INVENTORY_SHA256 or sha256(a.pair_manifest) != runner.PAIR_SHA256:
        raise ValueError("Inventory or fixed-pair manifest hash mismatch")
    inventory=read_csv(a.image_inventory)
    validation=runner.read_validation_rows(a.validation_manifest,inventory)
    indices_rows=read_csv(a.visual_indices)
    indices=[int(x["image_index"]) for x in indices_rows]
    if len(indices)!=8 or len(set(indices))!=8:
        raise ValueError(f"Expected eight unique fixed indices, got {indices}")
    for i,row in zip(indices,indices_rows):
        if validation[i]["image_relpath"] != row["image_relpath"]:
            raise ValueError(f"Fixed image identity mismatch at {i}")
    pair_images=json.loads(a.pair_manifest.read_text(encoding="utf-8"))["images"]
    summary=read_csv(a.existing_summary)
    references={(r.get("condition") or r.get("model_id"),int(r["image_index"])):r for r in summary}
    if len(references)<768:
        raise ValueError("Reconciled per-image reference rows are incomplete")
    legacy={
      "epoch20-B0":{int(r["image_index"]):r for r in read_csv(a.baseline_b0_per_image)},
      "epoch20-B3":{int(r["image_index"]):r for r in read_csv(a.baseline_b3_per_image)},
    }
    if any(len(v)!=128 for v in legacy.values()):
        raise ValueError("Historical epoch20 per-image metrics must each contain 128 rows")
    sources={
      "epoch20-B0":a.epoch20_b0_cache_dir,
      "epoch20-B3":a.epoch20_b3_cache_dir,
    }
    entries=[]
    hashes={}
    for i in indices:
        row=validation[i]
        rgb_path=a.rgb_root/row["image_relpath"]
        gt_path=a.depth_root/row["depth_relpath"]
        bgr=cv2.imread(str(rgb_path),cv2.IMREAD_COLOR)
        if bgr is None: raise FileNotFoundError(rgb_path)
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        gt=np.asarray(tifffile.imread(gt_path),dtype=np.float32)
        if rgb.dtype!=np.uint8 or rgb.shape[:2]!=gt.shape or rgb.shape[2]!=3:
            raise ValueError(f"RGB/GT shape or type mismatch at {i}")
        if sha256(rgb_path)!=inventory[i]["rgb_sha256"] or sha256(gt_path)!=inventory[i]["gt_sha256"]:
            raise ValueError(f"RGB/GT hash mismatch at {i}")
        valid=runner.m.valid_gt(gt)
        if not valid.any(): raise ValueError(f"Empty GT mask at {i}")
        for condition in ("epoch20-B0","epoch20-B3","C0-B3","E3-B3"):
            if condition in sources:
                path=sources[condition]/f"prediction_{i:03d}.npy"
                pred=np.load(path,mmap_mode="r",allow_pickle=False)
                if pred.dtype!=np.float32 or pred.shape!=gt.shape or not np.isfinite(pred).all():
                    raise ValueError(f"Invalid fixed-schema NPY cache for {condition}, image {i}")
            else:
                group="C0" if condition=="C0-B3" else "E3"
                path=a.existing_predictions/group/f"prediction_{i:03d}.npz"
                with np.load(path,allow_pickle=False) as z:
                    if z.files!=["prediction_m"]:
                        raise ValueError(f"Invalid NPZ schema for {condition}, image {i}")
                    pred=np.asarray(z["prediction_m"])
                    if pred.dtype!=np.float32 or pred.shape!=gt.shape or not np.isfinite(pred).all():
                        raise ValueError(f"Invalid fixed-schema NPZ cache for {condition}, image {i}")
            hashes[f"{condition}/{i:03d}"]=sha256(path)
            absolute=runner.absolute_stats(pred,gt,valid)
            dense=runner.dense14_stats(pred,gt,valid)
            local,_=runner.fixed_pair_stats(pred,gt,valid,i,pair_images)
            high=[x for x in local if x["stride_px"]==14 and x["group"]=="gt_ge20cm"]
            hn=sum(x["pair_count"] for x in high)
            high_mae=sum(x["absolute_error_sum_m"] for x in high)/hn if hn else None
            ref=references.get((condition,i))
            if ref is None: raise ValueError(f"Missing corrected reference row for {condition}, image {i}")
            checks={
              "reconciled_rmse":audit(absolute["rmse_m"],value(ref,"rmse_m","rmse")),
              "reconciled_full14_mae":audit(dense["full14_local_mae_m"],value(ref,"full14_local_mae_m","full_mean")),
              "reconciled_high14_mae":audit(high_mae,value(ref,"gt_ge20_14_local_mae_m","high_mean")),
            }
            if condition in legacy:
                old=legacy[condition][i]
                checks["historical_rmse"]=audit(absolute["rmse_m"],value(old,"rmse_m"))
                checks["historical_full14_mae"]=audit(dense["full14_local_mae_m"],value(old,"local_depth_delta_mae_m"))
            failures=[k for k,v in checks.items() if not v["passed"]]
            if failures: raise ValueError(f"Metric/cache mismatch {condition} image {i}: {failures}")
            entries.append({"condition":condition,"image_index":i,"valid_pixels":absolute["valid_pixel_count"],
                            "full14_pair_count":dense["full14_pair_count"],"high14_pair_count":hn,"checks":checks})
        del bgr,rgb,gt
    out={
      "artifact":"fixed8_prediction_cache_audit","utc_finished":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
      "cache_maps_checked":len(entries),"fixed_indices":indices,"tolerance_m":1e-5,
      "metrics_reference":"reconciled per-image statistics, plus historical B0/B3 per-image CSVs for epoch20",
      "gt_mask":"finite and 0 < depth <= 65m","rgb_input_contract":"uint8 RGB 0..255 (verified source bytes and inventory hashes)",
      "pair_manifest_sha256":sha256(a.pair_manifest),"validation_sha256":sha256(a.validation_manifest),
      "inventory_sha256":sha256(a.image_inventory),"summary_sha256":sha256(a.existing_summary),
      "cache_sha256":hashes,"entries":entries,"passed":len(entries)==32,
      "gpu_used":False,"training":False,"independent_test319":False,
    }
    if not out["passed"]: raise ValueError("Expected exactly 32 cached maps")
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open("x",encoding="utf-8") as f: json.dump(out,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({"status":"passed","cache_maps_checked":len(entries),"output":str(a.output)}),flush=True)

if __name__=="__main__": main()
