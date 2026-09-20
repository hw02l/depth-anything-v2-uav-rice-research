# Metric depth 実装監査

## 結論

現行の Transformers 実装は、KITTI で使う metric depth モデルとして妥当であると判断した。relative depth モデルと metric depth モデルは別の Hugging Face model ID を使っており、KITTI 評価では屋外用 metric モデルだけを使用している。公式実装との同一画像比較でも、出力の単位・値域・空間的な挙動に大きな不整合は見られなかったため、既存コードは維持した。

## 使用モデルと単位

| 用途 | Model ID / checkpoint | 種類 | 学習・想定データ | max depth | 出力単位 |
|---|---|---|---|---:|---|
| relative | `depth-anything/Depth-Anything-V2-Small-hf` | relative depth | 一般単眼深度 | なし | 相対値、mではない |
| KITTI / 屋外 | `depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf` | metric depth | Virtual KITTI 2でfine-tuningされた屋外モデル | 80 | m |
| 屋内用（未使用） | `depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf` | metric depth | Hypersimでfine-tuningされた屋内モデル | 20 | m |
| 公式実装比較用 | `depth_anything_v2_metric_vkitti_vits.pth` | official metric | Virtual KITTI 2 | 80 | m |

公式の metric depth 実装は `DepthAnythingV2(encoder="vits", max_depth=80)` とし、DPT head の出力を `max_depth` 倍している。本プロジェクトのHF設定 `config.json`でも `depth_estimation_type=metric`、`max_depth=80` を確認した。したがって、KITTI用の既定モデルは相対深度値を後処理でメートル化したものではなく、メートルスケールを学習したモデルの出力である。

## 実装の確認項目

- relative と metric は `load_estimator("relative")` / `load_estimator("metric")` で分離されている。
- KITTI 既定値は屋外用 `Depth-Anything-V2-Metric-Outdoor-Small-hf` である。
- KITTI GT は公式の uint16 PNG を `raw / 256.0` として m に変換している。
- prediction は元RGBサイズへ戻した後、GT形状と異なる場合だけ `cv2.INTER_LINEAR` でGTサイズへ合わせる。
- GTの有効画素は有限値かつ `gt > 0`、predictionも有限値かつ `pred > 0` とし、既定の評価範囲 `gt <= 80 m` を適用する。
- invalid depth、0、NaN、無限大は指標から除外する。
- relative depth はメートル指標へ混ぜていない。

Transformers版は内部processorと `predicted_depth` を使い、出力を元RGBサイズへbicubicで戻す。公式実装はOpenCVの前処理と最終bilinear補間（`align_corners=True`）を使うため、画素単位の完全一致を期待する実装ではない。この差は監査で明記した。

## 公式実装との同一画像比較

公式GitHub実装の `metric_depth/depth_anything_v2/dpt.py` と、公式Virtual KITTI 2 checkpoint `depth_anything_v2_metric_vkitti_vits.pth` を用いて、現在のTransformers版と同じ2画像を推論した。値は両方とも m として比較した。

| 画像 | official–Transformers MAE (m) | RMSE (m) | mean relative abs diff | abs diff p95 (m) |
|---|---:|---:|---:|---:|
| KITTI smoke frame | 0.1712 | 0.3622 | 0.00660 | 0.7509 |
| ETH3D `delivery_area_1l/im0.png` | 0.1004 | 0.1332 | 0.00712 | 0.2591 |

比較結果は `work/official_compare.json` に保存済みである。両モデルの最小・最大値も近く、平均相対差は約0.7%であった。これは前処理・補間方法の差を含む実装差として説明可能な範囲であり、モデルの取り違えや単位誤りを示す挙動ではないと判断した。

## 参照した公式情報

- [Depth Anything V2 公式リポジトリ](https://github.com/DepthAnything/Depth-Anything-V2)
- [公式 metric depth README](https://github.com/DepthAnything/Depth-Anything-V2/blob/main/metric_depth/README.md)
- [屋外用 Hugging Face model card](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf)

## 残る注意点

公式評価スクリプトとの完全一致を研究論文の再現性要件にする場合は、公式前処理・補間を完全に移植した比較モードを追加する価値がある。ただし今回の複数画像予備評価では、現在のTransformers版を統一して使用する。
