# UAV3DCrop failure / error analysis

## 確認済みの事実

- `2025_Day001_Wheat_nadir_1_error_analysis.png` では、GT z-depthはおよそ15 m付近、DA2 raw metric depthはおよそ5 m前後で、画像全体に大きな負のoffsetがある。
- `2025_Day001_Wheat_oblique_849_error_analysis.png` では、GTは画面位置に応じて約15–47 mまで変化する一方、DA2は約4–14 mの範囲である。signed errorは主に負で、距離が大きい領域で絶対誤差も大きい。
- oblique 10枚のraw MAE平均は16.5767 mで、nadir 10枚の9.4131 mより大きい。
- bias-corrected error mapでも、obliqueでは上下・行方向の残差が残る。したがって誤差はglobal offsetだけでは説明できない。
- `depth_profiles/` のnadir 1、oblique 182などでは、GTの緩やかなprofile変化にbias-corrected predictionが部分的に追従する。ただし、profileは任意の画像中心行であり、ground-canopy境界を直接横切るROIとして設計したものではない。

## 推測を含む解釈

### UAV視点・距離分布のdomain shift

UAV3DCropのoblique画像では、同一画像内のGT z-depth分布が広く、作物列が遠近方向に並ぶ。DA2 outdoor checkpointは一般屋外画像ではmetric depthを出すが、この草本作物・高高度・俯瞰/斜視の組合せは学習分布と異なる可能性がある。raw errorがobliqueで大きいことは確認済みだが、原因を学習分布だけに帰属することはまだできない。

### repetitive crop rows

RGBには反復する作物列、裸地、畦、影がある。predictionには列方向の大局的な変化が現れるが、GTの細かな変化をすべて再現しているとはいえない。これはDA2の視覚推論誤差、低解像度内部推論、またはMVS reference側の平滑化のいずれでも起こり得る。

### canopy表面とgroundの混在

今回のUAV3DCrop GTはphotogrammetry MVS z-depthであり、植物高専用のground/canopy annotationではない。したがって、RGBで緑に見える領域のdepthがgroundなのかcanopyなのかを、この比較だけから断定できない。植物高への変換には別途ROIとground planeの定義が必要である。

### image edge / invalid GT

一部oblique GT TIFFには0値があり、valid maskから除外した。これはDA2 failureとは限らず、MVSの観測不足や境界のinvalid表現の可能性があるため、invalid pixelを誤差として扱っていない。

## 原因の切り分け

| 現象 | 現時点の判定 |
|---|---|
| 全体的な負のoffset | DA2 raw metricとGTのglobal calibration差として確認。GT単位の取り違えではない |
| obliqueでの悪化 | 実験事実。viewpoint/domain shift、距離範囲、MVS難度が候補 |
| 細かな作物表面形状の残差 | 実験事実。DA2、MVS、raster化のどれが主因か未分離 |
| ground/canopyの誤分離 | 今回は専用maskを導入していないため、height評価の未解決問題 |
| invalid border | GT valid maskで除外。原因をDA2と断定していない |

## 研究上の含意

raw metric MAEだけを見ると、今回のsceneでDepth Anything V2を植物高へ直接使うことは不適切である。一方、global biasを分離したprofileとlocal depth differenceを調べると、relative/整合後depthが局所geometryをどの程度保持するかを独立に評価できる。これはLUMSのcanopy-ground proxyを検証するための手順として有用だが、wheatからriceへ性能を外挿する根拠にはならない。

## 次の切り分け

1. 同じsceneでground-visible、dense canopy、作物列境界を手動ROIとして分ける。
2. MVS confidenceまたはdepth completenessが公開されている場合は、GT品質の低い領域を別集計する。
3. 稲を含むRGB-depth/pose付きsceneを追加し、作物種とviewpointを分離する。
4. camera geometryとground planeを使い、局所depth差をplant height候補へ変換する。ただし実測GTなしでは精度を主張しない。
