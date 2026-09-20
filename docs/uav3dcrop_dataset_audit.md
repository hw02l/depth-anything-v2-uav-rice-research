# UAV3DCrop データセット監査

監査日: 2026-09-20

## 参照した公式情報

- [UAV3DCrop project page](https://link-dev.github.io/UAV3DCrop/)
- [UAV3DCrop paper](https://arxiv.org/abs/2608.06404)
- [公式 GitHub repository](https://github.com/Link-dev/UAV3DCrop)
- [Hugging Face RGB dataset](https://huggingface.co/datasets/Link-Dev/UAV3DCrop)
- [Hugging Face depth dataset](https://huggingface.co/datasets/Link-Dev/UAV3DCrop_depth)

READMEだけでなく、Hugging Face APIで現在の `main` revisionのtreeと、選択sceneの実ファイルを確認した。

## 使用したcanonical release

今回の取得対象は、両データセットの現在のdefault `main` である。

| dataset | main target commit | scene |
|---|---|---|
| RGB | `b97cfd8b1b6312ead12d67ed146bd4fa4b917de2` | `2025/Day001_Wheat` |
| depth | `83f544771890035f7ff9883bb7f1eade7cb6906a` | `2025/Day001_Wheat_depth` |

Hugging Faceのcanonical release noteでは、2023年データはlegacy normalized representationからJSON/PLYが変換され、2024/2025年データはmeter-scale local coordinatesであると説明されている。取得した `transforms.json` は `scale=1.0`、`avg_pos="0 0 0"` であったため、legacy normalizationを追加適用していない。`transforms.json` のcamera translationと `sparse_pc.ply` のXYZを、同一のmetric local coordinate frameとして扱った。

## 実ファイル構成

選択sceneには以下が存在した。

- RGB: `images/1.JPG` など908枚、各 `5280×3956` pixel
- reference depth: `2025/Day001_Wheat_depth/1_depth.tif` など908枚、RGBと同じstemで1対1対応
- camera/scene metadata: `transforms.json`
- sparse geometry: `sparse_pc.ply`

今回実際に取得したのは `transforms.json`、`sparse_pc.ply`、RGB 20枚、対応depth TIFF 20枚だけである。scene全体のRGB/depthはダウンロードしていない。

## Camera geometry

`transforms.json` の実値は次の通りである。

| field | value |
|---|---:|
| resolution | 5280 × 3956 |
| `fl_x`, `fl_y` | 3718.3260 px |
| `cx`, `cy` | 2639.0090, 1932.1209 px |
| camera model | `OPENCV` |
| `k1`, `k2`, `k3` | -0.1069135, -0.0025195, -0.0139711 |
| `p1`, `p2` | -0.0002776, 0.0000097 |
| scale / avg_pos | 1.0 / `0 0 0` |

各frameに `transform_matrix` があり、今回の画像にはRGBとextrinsicsの対応がある。view typeはREADMEのファイル名順ではなく、transform matrixのcamera optical-axis方向を確認して分類した。`abs(R[2,2]) >= 0.90` をnadir、それ以外をobliqueとした。今回のsceneでは元々nadir 381枚、oblique 527枚であり、選択は各10枚である。obliqueのverticality scoreは約0.70で、45度撮影の説明と整合する。

## Reference depthの定義・単位

公式論文の評価定義に従い、depth TIFFはphotogrammetryのdense MVSから生成されたcamera-frame z-depthで、光軸方向の距離をmeterで表すものとして扱った。これは視線方向のray distanceではない。

取得した `1_depth.tif` は `float16`、shape `(3956, 5280)` で、値は約14.83–15.63 m、全画素がfiniteであった。他の選択画像では一部0値があり、評価・点群対応ではGT>0かつfiniteの画素だけをvalidとした。GTには補間を適用していない。

RGBとdepthのshapeが一致したため、比較前のresize・crop・orientation変換は不要だった。予測は既存Transformers実装で元RGB rasterへbicubic resizeされるため、その出力と同じraster上で比較した。

## 利用できるデータ

- raw RGB UAV frame
- frame対応のdense MVS z-depth TIFF
- per-scene intrinsicsとOPENCV distortion
- per-frame camera extrinsics
- sparse point cloud
- scene/year/cropのディレクトリ情報
- nadir/obliqueをtransform matrixから再現可能なview geometry

したがって今回、raw metric depth、bias-corrected/scale+shift診断、pixel-corresponding local depth difference、single-view point cloud、少数viewのworld-coordinate fusionを実施できる。

## 利用できない、または今回のファイルだけでは対応できないデータ

論文には2025年のcanopy height評価（31 scenes、210測定点）が記載され、groundは別surveyの50 percentile、canopyは作物に応じた85/90 percentileなどの定義が示されている。しかし、今回確認した公開Hugging Face RGB/depth treeとscene metadataには、個々のframe・pixel/ROI・測定点を結ぶ公開height CSVまたは対応表が見当たらなかった。

したがって、論文にheight benchmarkがあることだけを根拠に、今回の20画像へcanopy-height GTを割り当てない。今回のheight estimation評価は未実施である。

また、reference depthはphotogrammetry由来であり、植物の真の手測り高さそのものではない。したがってdepthの良さを、そのまま植物高の精度と読み替えない。

## 今回実施可能な実験

1. 同一raster上のGT z-depthとDA2 outdoor metric depthの直接評価。
2. nadirとobliqueの分離評価。
3. global median bias補正とscale+shift整合を用いた診断評価。
4. 同一画像内のpixel pairのdepth difference評価。
5. relative modelをmeterへ直接比較せず、scale+shift整合後のgeometry preservation比較。
6. intrinsics/distortionを用いたcamera/world point cloud生成。

## 今回実施しない比較

- canopy height GTとの定量評価: frame/ROI対応が確認できないため未実施。
- ICPやscale alignment後の値をraw metric精度として報告すること: 実施していない。scale+shiftは診断値としてのみ保存。
- sparse PLYとのnearest-neighborを、dense reference surfaceとの厳密な精度値として扱うこと: sparse PLYは位置関係確認用に使用した。

## 監査上の結論

UAV3DCropは、LUMS稲画像では不足していたframe-level depth reference、intrinsics、extrinsicsを同時に提供するため、今回の目的である「absolute metric depth」と「local relative geometry」の分離評価に適している。ただし、作物種は稲ではなく、今回のsceneはwheatである。この結果は稲への直接的な精度保証ではなく、草本作物のUAV RGB条件での予備評価として扱う。

## 追加取得したscene

local geometryの小規模な再現性確認では、同じcanonical `main` revisionから `2025/Day001_Oat`（oat、16枚）と `2025/Day007_Corn`（corn、16枚）を追加した。各sceneはnadir 8枚、oblique 8枚であり、RGB/depth TIFF、`transforms.json`、`sparse_pc.ply`の必要最小限だけを取得した。Wheat 20枚と合わせた52枚の固定selectionは `results/uav3dcrop/all_selected_images.csv`、統合結果は `results/metrics/uav3dcrop_all/` にある。
