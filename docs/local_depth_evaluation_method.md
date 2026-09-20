# UAV3DCrop local depth difference 評価方法

## 目的

UAV RGB画像からの植物高利用を想定し、絶対metric depthの正しさと、同一画像内の局所的な深度差の保存を分けて評価した。単純な Local Difference MAE だけでは、GTの差分がほぼ0のpairを「ほぼ0」と予測するだけでも小さい値になるため、差分分布、zero baseline、相関、符号一致率、回帰傾きを併記する。

## 対象データ

2025/Day001_Wheat 20枚、2025/Day001_Oat 16枚、2025/Day007_Corn 16枚の合計52枚を使用した。各画像はRGBとUAV3DCrop reference depth TIFFが同一の5280×3956 rasterで対応している。GTはcamera-frame z-depth [m]として扱い、GT補間は行っていない。

## pixel pairの定義

pixel separation `s` は、画像上の水平または垂直方向の座標差である。たとえば32 pxでは、水平pair `(y,x)` と `(y,x+32)`、または垂直pair `(y,x)` と `(y+32,x)` を使う。斜め距離や実空間距離ではない。評価した `s` は8, 16, 32, 64, 128 pxである。

各画像・各separationについて、水平・垂直の候補から重複なしで最大20,000 pairを乱数抽出した。GT、metric、relativeの全variantに同じpair集合を使ったため、variant間の比較でsampling差が入らない。以前の実装でvariantごとに独立抽出していた点と、replacement samplingの可能性は今回修正した。

有効pairは、両端点についてGTがfiniteかつGT>0、metric predictionがfiniteかつ>0、relative predictionがfiniteであるpairとした。無効GTは評価から除外し、GTを補間して有効化していない。

## 深度差の符号

すべてのdepth conventionをcamera-frame z-depthに揃え、

```text
Delta d = d(pixel 1) - d(pixel 2)
```

と定義した。正の値はpixel 1の方がcamera z方向に遠いことを意味する。`sign agreement` はGT差分が0でないpairについて `sign(Delta d_pred) == sign(Delta d_GT)` を数え、予測値が0の場合は一致としない。GT差分が完全に0のpairは順序判定がないためsign集計の分母から除外した。

## 比較するprediction

- `metric_raw`: Depth-Anything-V2-Metric-Outdoor-Small-hfのmeter出力
- `metric_bias_corrected`: 各画像の `median(pred-GT)` を全pixelから求め、predictionから引いた診断値
- `metric_scale_shift`: `a*pred+b` を全valid pixelの最小二乗でGTへ合わせた診断値
- `relative_scale_shift`: Depth-Anything-V2-Small-hfのrelative出力を、公式実装のinverse-depth orientationに合わせて符号反転し、GTへscale+shift整合した診断値

global biasはpixel pair差分では理論上相殺されるため、rawとbias-correctedのlocal差分はほぼ同じになる。scale+shiftとrelative scale+shiftはgeometry preservationの診断であり、metric depth精度として報告しない。

## 指標

各separation、GT差分magnitude bin、view、scene、cropについて以下を計算した。

- Local Difference MAE: `mean(abs(Delta d_pred - Delta d_GT))`
- Local Difference RMSE
- local difference bias
- Pearson correlation
- tie-aware Spearman correlation
- sign agreement
- GT差分の平均絶対値と標準偏差

GT差分magnitude binは0–0.02, 0.02–0.05, 0.05–0.10, 0.10–0.20, 0.20–0.50, 0.50–1.00, 1.00 m以上とした。また0.10, 0.20, 0.50, 1.00 mの±20%範囲を抽出し、`Delta d_pred = a Delta d_GT + b` のslope、intercept、R²を求めた。

## zero baseline

`Delta d_pred=0` を全pairに対する単純baselineとした。このbaselineのMAEはGT差分の平均絶対値そのものである。DA2のMAEがzero baselineと同程度または大きい場合、MAEだけを根拠に局所geometryが保存されたとは言わない。

## 実行結果の読み方

今回の52枚では32 pxのGT差分中央値は0.015625 m、平均は0.040656 mであった。したがって「32 pxのMAEが約4 cm」という値は、まず約半数のpairが数cm以下の差しか持たない分布の上で解釈する必要がある。大きな差分に対してはmagnitude別MAE、Pearson/Spearman、sign agreement、slopeを併記する。

## 出力

- `results/metrics/uav3dcrop_all/local_gt_difference_distribution.csv`
- `results/metrics/uav3dcrop_all/local_difference_by_gt_magnitude.csv`
- `results/metrics/uav3dcrop_all/local_difference_baseline_comparison.csv`
- `results/metrics/uav3dcrop_all/local_difference_sign_thresholds.csv`
- `results/metrics/uav3dcrop_all/local_difference_ratio.csv`
- `results/metrics/uav3dcrop_all/local_difference_target_magnitude.csv`
- `results/metrics/uav3dcrop_all/local_difference_by_scene.csv`
- `results/metrics/uav3dcrop_all/local_difference_by_crop.csv`
- `results/metrics/uav3dcrop_all/local_difference_by_view.csv`
- `outputs/uav3dcrop/all_local_analysis/report_figures/`

実装は `scripts/analyze_local_difference_magnitude.py` にまとめ、旧entry point `scripts/analyze_uav3dcrop_local_depth.py` からも呼び出せるようにした。
