# Depth Anything V2 予備実験計画

## 目的

将来的に UAV で撮影した稲の RGB 画像から高さ推定・3D 復元を行うための予備実験として、公開データセットだけを用いて Depth Anything V2 の相対深度・metric depth・点群化の挙動を確認する。

## 方針

1. 公式 Depth Anything V2 リポジトリのモデル実装と前処理を可能な限りそのまま利用する。
2. 相対深度と metric depth は別 CLI にする。相対深度は `depth_anything_v2.dpt.DepthAnythingV2`、metric depth は公式 `metric_depth/depth_anything_v2/dpt.py` の `max_depth` 付きモデルを使う。
3. データセット固有の処理と評価指標は共通ユーティリティにまとめ、実験スクリプトから再利用する。
4. 全生成物は `outputs/` 配下に保存し、入力データとモデル重みは `data/` と `checkpoints/` に置く。
5. 最初の実行は CPU でも完了する小規模設定（Small encoder、少数画像、低めの入力サイズ）とし、研究用の本実験では encoder・入力サイズ・データ数を増やせる設計にする。

## 実装項目

### 1. 公開一般画像での relative / metric 推論

- `scripts/infer_relative.py`：画像または画像ディレクトリに相対深度を推論し、可視化 PNG と raw `.npy` を保存。
- `scripts/infer_metric.py`：屋外 KITTI 用または屋内 Hypersim 用の公式 metric checkpoint を読み、メートル単位の depth を PNG/NPY で保存。
- `src/da2_experiments/modeling.py`：公式実装のモデル設定・checkpoint 読み込み・デバイス選択を共通化。

### 2. KITTI quantitative evaluation

- `scripts/evaluate_kitti.py`：公式 KITTI depth prediction の `val_selection_cropped/image` と `groundtruth_depth` を読み、推定 depth と GT の有効画素だけで MAE、RMSE、AbsRel を計算。
- KITTI の `uint16` depth は `depth / 256.0` でメートルに変換。
- 画像ごとの指標と全画素集約の指標を JSON/CSV に保存。
- 予測が無い場合は metric model でその場で推論し、既存の prediction directory も利用可能にする。

### 3. ETH3D RGB + depth からの点群生成

- `scripts/generate_eth3d_pointcloud.py`：ETH3D の RGB、推定 depth（または指定した depth map）、カメラ内部パラメータから colored `.ply` を生成。
- `fx, fy, cx, cy` を CLI で指定可能にし、ETH3D の calibration ファイルからの読み取りにも対応する。
- 無効 depth、最大深度、画素間引きを設定できるようにする。
- depth の可視化と点群統計も `outputs/` に保存。

## 出力レイアウト

```text
outputs/
  relative/
  metric/
  kitti/
  eth3d/
  reports/
```

## 検証手順

1. `python -m compileall src scripts` で構文確認。
2. 公開一般画像の少数サンプルで relative / metric 推論を実行。
3. KITTI の少数サンプルで MAE/RMSE/AbsRel が JSON に出ることを確認。
4. ETH3D の少数サンプルで RGB + depth の `.ply` が生成されることを確認。
5. `outputs/README.md` に実行済み成果物と、データセット未取得時の未完了事項を記録。

## 今回の拡張計画と実績

### Metric model監査

- Transformers版のmodel ID、metric/relativeの分離、max depth、単位、KITTI GTのuint16/256変換、invalid maskを確認する。
- 公式Virtual KITTI 2 metric checkpointと同一画像を比較し、実装差を記録する。
- 結果を `docs/metric_model_validation.md` に保存する。

### KITTI複数画像評価

- 公式 `data_depth_selection.zip` をHTTP Rangeで読み、全量を保存せずvalidationから等間隔に50画像を抽出する。
- 画像ごとにMAE、RMSE、AbsRel、SqRel、RMSE log、delta指標を計算する。
- 画像ごとのCSV、mean/median/std/min/max、距離帯別CSV、散布図・誤差図・代表例を生成する。

### ETH3D GT評価

- low-res two-viewのPFM disparityを `calib.txt` からmetric depthへ変換する方法Aを採用する。
- ETH3D屋内シーンには屋内用metricモデルを使い、RGB/GT/pred/errorを比較する。
- 同じim0カメラ座標で予測・GT点群を作り、ICPやscale alignmentなしの双方向nearest-neighbor距離を予備的に計算する。

## 前提・注意

- KITTI の公式評価用画像・GT はサイズが大きいため、初回はユーザーが取得した subset を `data/kitti/` に置き、`--limit` で数枚だけ評価できるようにする。
- ETH3D はデータセットごとに depth の単位と calibration の保持場所が異なるため、入力形式を明示的に CLI で指定する。点群の絶対スケールは metric depth と内部パラメータに依存する。
- relative depth は絶対スケールを持たないため、KITTI の metric 指標には使用しない。
- 公開稲UAV raw RGBとDEM/実測値の対応が確認できない場合は、absolute accuracyを計算しない。

### 公開稲UAV画像への適用

1. GitHubの公開ファイル構成をREADMEと実ファイルの両方で監査する。
2. early/middle/lateから必要最小限のraw RGBを取得し、選択表とEXIFを保存する。
3. 既存のrelative modelとoutdoor metric modelを全選択画像へ適用する。
4. ExG、VARI、HSVを比較し、採用maskを理由とともに記録する。
5. 地面候補が目視で妥当な画像だけ、`canopy-ground depth difference proxy`を算出する。植物高とは呼ばない。
6. depth統計、stage群比較、line profile、failure case、報告図を自動生成する。
7. raw frameとDEM/ground truthの対応が無い限り、DEM比較・稲高精度評価は実施しない。

実行済みの主な成果物は`results/rice_uav/`、`results/metrics/rice_uav_*`、`outputs/rice_uav/`、`docs/rice_dataset_audit.md`、`docs/rice_vegetation_mask.md`、`docs/rice_failure_analysis.md`、`docs/research_questions.md`である。
