# 局所Huber追加学習（実験ID `local_huber_20261105`）

## 条件

固定条件は `config.json` と `PLAN_JA.md` を参照する。E0はepoch20のB0評価、E1はSiLogのみ、E2/E3はSiLog + λ×局所Huber（λ=0.05/0.20）である。E1〜E3は同じepoch20モデル重みから独立に開始する。

## GPU配置

GPGPU4のGPUローカル領域 `/localdata/s0623003` をSingularity内 `/work` にbindし、プロジェクトを `/work/local_huber_project_20261105` に配置する。RGB/GTは既存の `/work/depth-anything-v2-uav/migration_work_20261015/data/uav3dcrop_{rgb,depth}` を読む。checkpointはハッシュ照合済みバックアップからローカル領域へコピーし、コピー後もSHA256を検査する。

```bash
P=/work/local_huber_project_20261105
SIF=/work/depth-anything-v2-uav/containers/pytorch_2.11.0_cuda12.6_cudnn9_runtime.sif
PYTHONPATH=/work/depth-anything-v2-uav/environment/site_py312_v2:$P/src
RGB=/work/depth-anything-v2-uav/migration_work_20261015/data/uav3dcrop_rgb
DEPTH=/work/depth-anything-v2-uav/migration_work_20261015/data/uav3dcrop_depth
TRAIN=$P/configs/uav3dcrop_verified/train.csv
VAL=$P/configs/uav3dcrop_verified/validation.csv
CKPT=$P/checkpoint_epoch_0020
RECEIPT=/work/depth-anything-v2-uav/local_huber_20261105/gpu_audit/receipt_live.json

singularity exec --nv --bind /localdata/s0623003:/work --env PYTHONPATH=$PYTHONPATH $SIF \
  python $P/experiments/local_huber_20261105/run_local_huber.py \
  --group E0 --config $P/experiments/local_huber_20261105/config.json \
  --checkpoint $CKPT --train-manifest $TRAIN --val-manifest $VAL \
  --rgb-root $RGB --depth-root $DEPTH --output $P/runs/E0 \
  --device cuda --gpu-work-root /work --preflight-receipt $RECEIPT
```

E0 must pass all fixed-reference checks before any training group. For each training group, issue a fresh read-only GPU receipt immediately before launch; E1 uses no local pair computation. Example E2 command (E1/E3 change only `--group`, `--output`, and the E0 proof path):

```bash
# Generate a fresh preflight receipt first; do not reuse an expired receipt.
singularity exec --nv --bind /localdata/s0623003:/work --env PYTHONPATH=$PYTHONPATH $SIF \
  python $P/experiments/local_huber_20261105/run_local_huber.py \
  --group E2 --config $P/experiments/local_huber_20261105/config.json \
  --checkpoint $CKPT --train-manifest $TRAIN --val-manifest $VAL \
  --rgb-root $RGB --depth-root $DEPTH --output $P/runs/E2 \
  --device cuda --amp --gpu-work-root /work --preflight-receipt $RECEIPT \
  --baseline-proof $P/runs/E0/baseline_reproduction.json
```

再開は同一の`--output`に`--resume`を付ける。設定、manifest、初期checkpoint、固定ペア、コードが一致しない場合は拒否される。epoch境界checkpointから再開し、未完了epochは直前のcheckpointから再計算する。

`--dry-run` は入力確認のみ、`--real-data-smoke` はCPU上の接続確認、`--synthetic` は合成動作試験である。これらの結果はE0〜E3の研究結果として扱わない。Independent test319は使用しない。

## 実験結果・報告書

- Validation128の全epoch要約は [validation_summary.csv](validation_summary.csv) に保存した。
- 選択epoch比較、選択規則、機械可読集計、provenance、学習曲線を [analysis/](analysis/) に保存した。
- 最終版の結果・統計解析・限界は [report_ja.md](report_ja.md) に記録した。
- 距離別、GT差分階級別、作物・視点別、端距離・深度帯・符号別の詳細値と件数は [analysis/final_20261107/TABLES_JA.md](analysis/final_20261107/TABLES_JA.md) および `condition_comparison_all.csv` にある。
- 画像単位paired bootstrap（10,000回、seed42）のpool推定値と画像平均推定値は `analysis/final_20261107/paired_bootstrap_pooled.csv` と `paired_bootstrap_image_mean.csv` に分けて保存した。
- 取得したper-image十分統計、固定ペアから再生成した条件別画像数、NFSのsource/derived SHA256台帳は `analysis/final_20261107/inputs/` に保存した。追加解析のprovenanceは `analysis/final_20261107/provenance_final.json`、公開用の最終成果・入力のSHA256台帳は `analysis/final_20261107/ANALYSIS_OUTPUT_SHA256SUMS` にある。
- チェックポイント実体はNFS退避先にあり、SHA256と保存先を [analysis/checkpoint_sha256.csv](analysis/checkpoint_sha256.csv) に記録した。巨大checkpointはGitHubへ含めない。
- 既存config内のfixed-pair SHA256記載に63桁の誤記があり、実manifestと不一致である。元configは変更せず、報告書とprovenanceに明記した。

統計の再集計は [report_ja.md](report_ja.md) のCPU再実行手順で行う。固定検証条件に合わない入力は解析コードが停止する。今回の最終分析段階では追加学習を行わず、Independent test319も使用していない。

\r\n