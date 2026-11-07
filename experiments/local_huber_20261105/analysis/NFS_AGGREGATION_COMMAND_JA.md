# NFS保存済みvalidationの追加集計（GPU計算なし）

現在の報告書に残る層別集計とabsoluteおよび全14px局所MAEの画像単位bootstrapを生成する手順である。学習jobやGPUを使わず、NFS backupのJSONを読む。新しい派生物だけをbackup内の別ディレクトリへ書く。既存run、checkpoint、元のSOURCE_SHA256SUMSは変更しない。

GPGPU4へSSH接続後、下記を一度実行し、最後に表示されるCOMPACT_RESULTS_BEGINからCOMPACT_RESULTS_ENDまでを保存する。パスワードは端末にだけ入力する。

~~~bash
set -euo pipefail
B=/home1/eng-std/s0623003/gpu4_research_backup_local_huber_20261105_20261105T081949Z
SIF=/localdata/s0623003/depth-anything-v2-uav/containers/pytorch_2.11.0_cuda12.6_cudnn9_runtime.sif
OUT="$B/derived_analysis_20261105"
test -d "$B/runs/E0" && test -d "$B/runs/E1" && test -d "$B/runs/E2" && test -d "$B/runs/E3"
test ! -e "$OUT"
mkdir "$OUT"
singularity exec --bind /home1:/home1 "$SIF" python - "$B" "$OUT" <<'PY'
import csv, json, math, pathlib, sys, hashlib
import numpy as np
backup=pathlib.Path(sys.argv[1]); out=pathlib.Path(sys.argv[2])
files={
 "E0":backup/"runs/E0/validation_epoch_0020.json",
 "E1":backup/"runs/E1/validation_epoch_0005.json",
 "E2":backup/"runs/E2/validation_epoch_0005.json",
 "E3":backup/"runs/E3/validation_epoch_0005.json",
}
data={}
for group,path in files.items():
    obj=json.loads(path.read_text(encoding="utf-8"))
    data[group]=obj["validation"]
    if len(data[group]["per_image"])!=128:
        raise ValueError(f"{group}: per_image count is not 128")
    if len(data[group]["sampled"])==0:
        raise ValueError(f"{group}: no sampled local results")
condition_rows=[]
for model,v in data.items():
    for row in v["sampled"]:
        condition_rows.append({"model":model,**row})
fields=list(dict.fromkeys(k for row in condition_rows for k in row))
with (out/"all_condition_metrics.csv").open("w",encoding="utf-8",newline="") as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(condition_rows)
strata_prefix=("border_","middle_","interior_","positive_","negative_","zero_","gt_delta_","depth_","endpoint_zones_")
compact=[]
for row in condition_rows:
    scope=row["scope"]; stride=int(row["stride_px"]); category=row["group"]
    include=(scope=="all" and (category in ("all","gt_ge20cm") or (stride==14 and category.startswith(strata_prefix))))
    include=include or (scope.startswith("crop=") and stride==14 and category in ("all","gt_ge20cm"))
    include=include or (scope.startswith("view=") and stride==14 and category in ("all","gt_ge20cm"))
    if include:
        compact.append({k:row.get(k) for k in ("model","scope","stride_px","group","pair_count","delta_mae_m","delta_rmse_m","amplitude_ratio","sign_match_rate","mean_abs_gt_delta_m","mean_abs_pred_delta_m")})
with (out/"compact_condition_metrics.csv").open("w",encoding="utf-8",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(compact[0])); w.writeheader(); w.writerows(compact)
def paired(model, baseline, metric, lower_is_better=True):
    a={int(x["image_index"]):x for x in data[model]["per_image"]}
    b={int(x["image_index"]):x for x in data[baseline]["per_image"]}
    if set(a)!=set(b) or any(a[i]["image_relpath"]!=b[i]["image_relpath"] for i in a):
        raise ValueError(f"Image mismatch: {model} vs {baseline}")
    diffs=[]
    for i in sorted(a):
        av=a[i].get(metric); bv=b[i].get(metric)
        if av is None or bv is None or not math.isfinite(float(av)) or not math.isfinite(float(bv)):
            continue
        diffs.append((float(av)-float(bv)) if lower_is_better else (float(bv)-float(av)))
    if len(diffs)!=len(a):
        return {"candidate":model,"baseline":baseline,"metric":metric,"status":"incomplete_nonfinite_images","valid_pairs":len(diffs),"expected_images":len(a)}
    arr=np.asarray(diffs,dtype=np.float64); rng=np.random.default_rng(42)
    means=np.array([rng.choice(arr,len(arr),replace=True).mean() for _ in range(10000)])
    return {"candidate":model,"baseline":baseline,"metric":metric,"images":len(arr),
            "mean_candidate_minus_baseline":float((arr.mean()) if lower_is_better else (-arr.mean())),
            "mean_improvement_oriented_difference":float(arr.mean()),
            "improved_images":int((arr<0).sum()),"worsened_images":int((arr>0).sum()),
            "unchanged_images":int((arr==0).sum()),"ci95_improvement_oriented":[float(x) for x in np.quantile(means,[.025,.975])],
            "seed":42,"repeats":10000,"scope":"paired image bootstrap within Day047; no unseen-day inference"}
bootstrap=[]
for candidate,baseline in (("E1","E0"),("E2","E0"),("E3","E0"),("E2","E1"),("E3","E1")):
    for metric in ("mae_m","rmse_m","abs_rel","pearson_r","local_depth_delta_mae_m"):
        bootstrap.append(paired(candidate,baseline,metric,metric!="pearson_r"))
payload={"image_mean_paired_bootstrap":bootstrap,
         "gt_ge20cm_local_per_image_bootstrap":{"status":"not_available_from_saved_validation_json","reason":"GT>=0.2m fixed-sample local MAE is pooled by scope/stride/group, not stored per image"},
         "condition_row_count":len(condition_rows),"compact_row_count":len(compact),
         "files":{g:str(p) for g,p in files.items()}}
(out/"paired_image_bootstrap.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
print("COMPACT_RESULTS_BEGIN")
print("CONDITION_ROWS model,scope,stride_px,group,pair_count,delta_mae_m,delta_rmse_m,amplitude_ratio,sign_match_rate")
for row in compact:
    print(",".join("" if row.get(k) is None else str(row[k]) for k in ("model","scope","stride_px","group","pair_count","delta_mae_m","delta_rmse_m","amplitude_ratio","sign_match_rate")))
print("IMAGE_MEAN_PAIRED_BOOTSTRAP_JSON")
print(json.dumps(bootstrap,ensure_ascii=False,separators=(",",":")))
print("GT_GE20CM_LOCAL_PER_IMAGE_BOOTSTRAP=unavailable_from_saved_JSON; rerun final-model evaluation required")
print("DERIVED_FILES")
for p in sorted(out.iterdir()):
    print(f"{p.name} {p.stat().st_size} bytes sha256={hashlib.sha256(p.read_bytes()).hexdigest()}")
print("COMPACT_RESULTS_END")
PY
~~~

この抽出は保存済みper-image absolute指標および全有効14px local_depth_delta_mae_mから画像平均のpaired bootstrapを行う。GT差20cm以上固定標本の局所MAEは画像別に保存されていないため、そのbootstrapにはE0/E1/E2/E3最終モデルの再評価が必要である。上記CSV/JSONの追加ファイルは同backupに置くが、既存のSOURCE_SHA256SUMSは変更しない。

\r\n