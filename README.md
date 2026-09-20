# Depth Anything V2 UAV RGB 予備実験

UAV RGB画像からのmetric depth、局所的な深度差、3D点群利用可能性を調べるPythonプロジェクトである。研究対象は将来の稲画像だが、公開データセットで予備評価を行う。

## 主な実験

- `Depth-Anything-V2-Small-hf` によるrelative depth
- `Depth-Anything-V2-Metric-Outdoor-Small-hf` による屋外metric depth（max depth 80 m、Virtual KITTI 2でfine-tuningされたcheckpoint）
- KITTI validation subset 50画像の定量評価
- ETH3D low-res two-viewのdisparityからのmetric depth評価と点群比較
- LUMS公開稲UAV RGB 9画像への適用、vegetation mask、canopy-ground depth difference proxy
- UAV3DCropのwheat scene 20画像（nadir 10、oblique 10）によるGT z-depth評価、局所深度差評価、点群生成
- UAV3DCropのWheat/Oat/Corn 3 scene、合計52画像によるGT差分magnitude別評価、zero baseline、scene/crop/view比較
- AirMeasurer公式small-field rice orthomosaic/LAS 1時期によるCHM再生成とrelative geometry探索

## セットアップ

```powershell
python -m venv .venv
& .venv\Scripts\python.exe -m pip install -r requirements.txt
```

モデルは初回実行時にHugging Faceから取得され、既定では `work/hf_cache/` に保存される。CPUでも実行できるが、高解像度UAV画像の推論には時間がかかる。

## 基本推論

```powershell
& .venv\Scripts\python.exe scripts\infer_relative.py --input data\general --outdir outputs\relative --limit 3
& .venv\Scripts\python.exe scripts\infer_metric.py --input data\general --outdir outputs\metric --limit 3
```

## KITTI 50画像評価

データは `data/kitti/val_selection_cropped/image` と `groundtruth_depth` に置く。GT uint16 PNGは256で割ってmeterに変換する。

```powershell
& .venv\Scripts\python.exe scripts\prepare_kitti_subset.py --count 50 --outdir data\kitti_validation_50
& .venv\Scripts\python.exe scripts\evaluate_kitti.py --data-root data\kitti_validation_50 --outdir outputs\kitti --results-dir results\metrics --batch-size 4
```

per-image結果は `results/metrics/kitti_per_image.csv`、summaryは `kitti_summary.csv`、距離帯集計は `kitti_by_depth.csv`、グラフは `outputs/kitti/plots/` に保存される。

## ETH3D評価・点群

```powershell
& .venv\Scripts\python.exe scripts\evaluate_eth3d.py --scene data\eth3d\delivery_area_1l --outdir outputs\eth3d_gt --results-dir results\metrics --stride 2 --max-depth 20
& .venv\Scripts\python.exe scripts\compare_eth3d_pointcloud.py --pred-ply outputs\eth3d_gt\im0_pred.ply --gt-ply outputs\eth3d_gt\im0_gt.ply --out-json outputs\eth3d_gt\im0_pointcloud_comparison_standalone.json
```

ETH3D low-res two-viewのGTはdisparityであり、calibrationから `Z = fx * baseline / (disparity + doffs)` でmetric z-depthへ変換する。座標系や手法の注意点は `docs/eth3d_gt_method.md` を参照する。

## LUMS稲UAV画像

```powershell
& .venv\Scripts\python.exe scripts\prepare_rice_uav.py --per-stage 3
& .venv\Scripts\python.exe scripts\infer_rice_uav.py --selection-csv results\rice_uav\selected_images.csv --output-root outputs\rice_uav
& .venv\Scripts\python.exe scripts\analyze_rice_depth.py
& .venv\Scripts\python.exe scripts\create_vegetation_mask.py --selection-csv results\rice_uav\selected_images.csv --output-root outputs\rice_uav
& .venv\Scripts\python.exe scripts\analyze_canopy_ground.py
& .venv\Scripts\python.exe scripts\create_rice_report_figures.py
```

このデータはraw RGBとDEMのframe-level対応が確認できないため、depth精度のGT評価には使わない。canopy-ground値は `canopy-ground depth difference proxy` と呼ぶ。

## UAV3DCrop GT評価

### データ取得（全量を取得しない）

```powershell
& .venv\Scripts\python.exe scripts\prepare_uav3dcrop.py --scene 2025/Day001_Wheat --nadir-count 10 --oblique-count 10
```

取得したscene、revision、選択画像、view分類は `docs/uav3dcrop_dataset_audit.md`、`docs/uav3dcrop_scene_selection.md`、`results/uav3dcrop/selected_images.csv` に記録される。

### 推論と評価

```powershell
& .venv\Scripts\python.exe scripts\infer_uav3dcrop.py --batch-size 1
& .venv\Scripts\python.exe scripts\evaluate_uav3dcrop.py
& .venv\Scripts\python.exe scripts\analyze_uav3dcrop_local_depth.py
```

`evaluate_uav3dcrop.py` はRGB、GT TIFF、predictionのshapeが一致することを検査し、補間なしでGT z-depth [m]と比較する。出力は以下である。

- `results/metrics/uav3dcrop_per_image.csv`
- `results/metrics/uav3dcrop_summary.csv`
- `results/metrics/uav3dcrop_by_view.csv`
- `results/metrics/uav3dcrop_by_depth.csv`
- `results/metrics/uav3dcrop_bias_corrected.csv`
- `results/metrics/uav3dcrop_local_depth_difference.csv`

raw metric、median global bias補正、scale+shift補正は別の指標である。後二者はgeometry preservationの診断であり、raw metric精度ではない。

relative depthはmeter値を持たないため、公式実装のinverse-depth orientationを確認した上で、GTへのscale+shift整合後だけをgeometry比較に使う。

### 複数sceneとlocal differenceの詳細評価

追加sceneは次で必要最小限だけ取得する。

```powershell
& .venv\Scripts\python.exe scripts\prepare_additional_uav3dcrop_scenes.py `
  --scenes 2025/Day001_Oat 2025/Day007_Corn `
  --nadir-count 8 --oblique-count 8
& .venv\Scripts\python.exe scripts\combine_uav3dcrop_selections.py
```

既存20枚と追加32枚を合わせた52枚の推論結果を評価する。

```powershell
& .venv\Scripts\python.exe scripts\evaluate_uav3dcrop.py `
  --selection-csv results\uav3dcrop\all_selected_images.csv `
  --results-dir results\metrics\uav3dcrop_all `
  --error-dir outputs\uav3dcrop\all_error_analysis
& .venv\Scripts\python.exe scripts\analyze_local_difference_magnitude.py `
  --selection-csv results\uav3dcrop\all_selected_images.csv `
  --results-dir results\metrics\uav3dcrop_all `
  --analysis-dir outputs\uav3dcrop\all_local_analysis `
  --profile-dir outputs\uav3dcrop\all_depth_profiles `
  --pair-samples 20000
```

この解析では、8/16/32/64/128 pxの重複なしpixel pairを同じsupportで比較し、GT difference magnitude別MAE/RMSE、Pearson/Spearman、sign agreement、regression slope、zero baseline、ratio、scene/crop/view別集計を作る。詳細は `docs/local_depth_evaluation_method.md` を参照する。主要CSVは `results/metrics/uav3dcrop_all/`、図は `outputs/uav3dcrop/all_local_analysis/report_figures/` に保存される。

sceneごとのabsolute/local指標を1行にまとめるには次を実行する。

```powershell
& .venv\Scripts\python.exe scripts\create_uav3dcrop_scene_summary.py
```

結果は `results/metrics/uav3dcrop_all/uav3dcrop_by_scene.csv` である。

view別の32 px regression slopeは `scripts/create_uav3dcrop_view_slope.py` で作成し、`results/metrics/uav3dcrop_all/local_difference_view_slope.csv` に保存する。

### 点群とmulti-view fusion

```powershell
& .venv\Scripts\python.exe scripts\create_uav3dcrop_pointcloud.py
& .venv\Scripts\python.exe scripts\fuse_uav3dcrop_pointclouds.py --count 3
& .venv\Scripts\python.exe scripts\create_uav3dcrop_report_figures.py
```

intrinsicsとOPENCV distortionで同一pixelのcamera rayを復元し、GT/pred depthを点群化する。`transforms.json` のcamera-to-worldを使って少数のnadir画像をworld座標へ変換する。raw metric評価ではICP・rigid alignment・scale alignmentを行わない。

PLYは `outputs/uav3dcrop/pointcloud/`、報告用図は `outputs/uav3dcrop/report_figures/`、代表的なerror mapは `outputs/uav3dcrop/error_analysis/`、depth profileは `outputs/uav3dcrop/depth_profiles/` に保存される。

## AirMeasurer rice orthomosaic

AirMeasurer公式V2.0.2 releaseの全testing dataは大容量なので、まずsmall-field assetだけを取得して検証する。今回の実験では `data/airmeasurer/small_field/` に `20220505.tif` と `20220505.las` を置く。取得範囲、CRS、GSD、plot情報の有無は `docs/airmeasurer_dataset_audit.md` を参照する。

```powershell
& .venv\Scripts\python.exe scripts\prepare_airmeasurer.py
& .venv\Scripts\python.exe scripts\create_rice_chm.py
& .venv\Scripts\python.exe scripts\infer_airmeasurer_relative.py --tile-size 1024 --overlap 128
& .venv\Scripts\python.exe scripts\evaluate_airmeasurer_relative.py
& .venv\Scripts\python.exe scripts\create_airmeasurer_report_figures.py
```

生成物は、CHMが `outputs/airmeasurer/chm/`、relative推論が `outputs/airmeasurer/inference/`、報告図が `outputs/airmeasurer/report_figures/`、評価CSVが `results/metrics/airmeasurer/` に保存される。

このデータのRGBはraw UAV frameではなくorthomosaicである。したがって、`Depth-Anything-V2-Metric-Outdoor-Small-hf` の出力をcamera-frame metric depthとして評価してはいけない。AirMeasurerでは `Depth-Anything-V2-Small-hf` のrelative outputをCHMと相関比較し、CHMを使ったscale+shiftはgeometry診断としてのみ扱う。plot boundary・対応height tableがない場合、plot-level H90/H95、held-out calibration、date transferは実施しない。公式処理との差と限界は `docs/airmeasurer_height_method.md`、failureは `docs/airmeasurer_failure_analysis.md` に記録する。

## 重要な解釈上の注意

- metric depthはカメラから表面までのdepthであり、植物高ではない。
- UAV3DCropのreferenceはphotogrammetry MVS z-depthであり、手測りcanopy height GTとは別物である。
- UAV3DCropのheight測定点を今回の公開frameへ対応付ける情報が確認できないため、height精度は評価していない。
- LUMS稲画像にはframe-level GTがないため、UAV3DCropの結果を稲の精度保証へ外挿しない。

## プロジェクト構成

```text
docs/                    監査・計画・進捗報告
scripts/                 推論・評価・図生成
src/da2_experiments/     共通model/io/metric/point cloud処理
data/                    公開データの必要最小限のsubset
results/metrics/         CSVと評価manifest
outputs/                 画像、npy、PLY、報告図
```

## 主要文書

- `docs/metric_model_validation.md`
- `docs/uav3dcrop_dataset_audit.md`
- `docs/uav3dcrop_scene_selection.md`
- `docs/local_depth_evaluation_method.md`
- `docs/uav3dcrop_gt_limitations.md`
- `docs/research_questions.md`
- `docs/progress_report.md`
- `docs/research_questions.md`
