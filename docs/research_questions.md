# 卒業研究で検証するresearch questions

UAV3DCropの結果から、絶対depth精度と局所geometry保存性能を別の問いとして扱う。

## RQ1: UAV俯瞰画像でDepth Anything V2のraw metric depthはどの程度正しいか

KITTIのような道路画像と、植生を含むUAV画像では視点・テクスチャ・距離分布が異なる。稲を含む複数sceneで、nadir/oblique、撮影高度、作物、時期ごとのMAE/RMSE/AbsRelを比較する。

## RQ2: 絶対depthのoffsetやscale誤差があっても、局所depth差は保存されるか

同一画像内のground-canopy、または近接pixel pairについて、raw、global bias補正、scale+shift診断、relative modelを分けて評価する。今回のUAV3DCropで独立に実施した最重要の問いである。

## RQ3: groundが見えない密集期に、植物高のground referenceをどう定義するか

vegetation maskの外側をすべてgroundとみなすことはできない。裸地、水面、畦、影、ground plane推定、隣接飛行画像などを比較し、ground候補の不確かさを高さ推定の誤差として扱う。

## RQ4: relative depthとmetric depthのどちらが稲高推定の中間表現として有効か

metricモデルの絶対値を使う方法と、relative modelをcamera geometry・ground plane・少数のscale anchorでmetric化する方法を比較する。relative出力はmeter値ではないため、直接の絶対depth評価とは区別する。

## RQ5: camera altitude、intrinsics、pose、DEMを組み合わせると局所depthから稲高へ変換できるか

Depth Anything V2のdepthをそのまま植物高と呼ばず、ray geometryとground surfaceを用いて高さへ変換する。raw prediction、bias/scale診断、photogrammetry DEMとの融合を比較する。

## RQ6: 作物種・撮影条件のdomain shiftに対する適応方法は何か

UAV3DCrop wheatで得られた結果を稲へ直接外挿せず、稲を含む公開RGB-depth、研究室データ、または実測付きデータで検証する。必要に応じて、軽量なscale calibration、fine-tuning、segmentation併用の効果を調べる。

現時点ではRQ2がもっとも卒業研究の中心に適している。UAV3DCropではraw absolute errorが大きくても、局所差分という別の評価軸を実測できるためである。

## 今回のlocal difference評価を踏まえた再整理

### RQ1: UAV crop imageryでDepth Anything V2のabsolute metric depthはどの程度正しいか

KITTI、ETH3D、UAV3DCrop、稲UAV画像で、domain、距離、viewpoint、GT定義の違いを明示して比較する。UAV3DCrop 52枚ではraw metricに大きな負biasが残ったため、raw metric評価と補正後の診断評価を混同しない。

### RQ2: absolute biasがあっても、植物高に必要なlocal depth differenceは保存されるか

groundとcanopy、または同一画像内の対応pixel pairで `Delta d` のMAE、RMSE、Pearson、Spearman、sign agreementを測定する。これは絶対depth精度とは独立した中心課題である。

### RQ3: local geometryの評価値はGT difference magnitudeに依存するか

0.1 m、0.2 m、0.5 m、1.0 m付近の差分を別々に評価し、zero baselineとの差、regression slope、R²を調べる。今回の52枚ではslopeが1よりかなり小さく、差分圧縮の可能性が示されたため、単一MAEで結論を出さない。

### RQ4: nadir / oblique、作物種、密度によってlocal geometry保存性能は変わるか

撮影角度を変えたとき、見えているground、葉の遮蔽、投影歪み、MVS reference品質がlocal depth差に与える影響を同一圃場・十分なscene数で調べる。現在のWheat/Oat/Corn 3 sceneは予備的な再現性確認であり、一般化の証拠ではない。

### RQ5: relative depthとmetric depthのどちらがcrop surface geometryに有効か

relativeはmeter値を持たないためscale+shift整合後のgeometry指標だけで比較し、metric rawの絶対精度とは分離する。今回の小規模結果ではrelativeの相関・sign agreementが高かったが、scale anchorやground planeを使ったmetric化が必要である。

### RQ6: ground referenceとcamera geometryを用いてlocal depth差を植物高へ変換できるか

ground pixel候補、ground plane、intrinsics、pose、DEMを組み合わせ、`camera-to-surface depth`を`canopy height`へ変換できるかを検証する。特に密集期にgroundが見えない場合の高さ定義を研究課題とする。手測りheight GTとframe/ROI対応がないデータでは、このRQを定量評価しない。

## 現時点の優先順位

第一優先はRQ2とRQ3である。理由は、今回のlocal MAEがzero baselineと同程度であり、「小さいMAEが意味のある高さ差の再現を示すか」という検証がまだ必要だからである。次に、稲のframe-level depthまたは実測height GTを含むデータでRQ6を検証する。

# AirMeasurer rice canopy height validationを踏まえた再整理

AirMeasurerのsmall-fieldでは、rice orthomosaicと同日LASからCHMを作ることはできたが、plot boundary・plot ID・対応height tableは確認できなかった。したがって、今回のrelative/CHM結果は1時期のpixel-level探索であり、plot-levelの稲高精度を示すものではない。この制約を踏まえ、卒業研究の問いを次のように整理する。

## RQ1: Depth Anything V2 relative depthはrice canopyの高さ順位を保持できるか

orthomosaicではcamera-frame metric depthとして扱わず、CHMまたは対応するcanopy heightとのPearson、Spearman、height-bin相関でrelative geometryを評価する。AirMeasurer 20220505ではPearson 0.2579、Spearman 0.2098であり、単一時期の結果だけでは十分な保持を確認できなかった。

## RQ2: 少数の独立height anchorでrelative scoreをcanopy heightへ校正できるか

CHM/実測heightを使ったscale+shiftを同じplotに適用するだけでなく、未使用plotをtestとする。AirMeasurerで行ったper-pixel scale+shiftは診断であり、uncalibrated単眼推論の精度ではない。今後はheld-out plot MAE/RMSE/R²を主要結果にする。

## RQ3: calibrationはgrowth stageをまたいで利用できるか

early→middle、middle→lateのように時期をまたいだcalibrationを行い、scale/interceptの安定性を検証する。AirMeasurer 20220505だけではこの問いを実行できないため、別dateと対応plotが必要である。

## RQ4: local canopy-height differenceの再現精度はheight差の大きさに依存するか

0.1、0.2、0.5、1.0 m付近の `|Delta h_GT|` を別々に評価し、MAE/RMSE、Pearson/Spearman、sign agreement、regression slopeを併記する。小差分が多いデータではzero/constant baselineも必ず報告する。AirMeasurerでは32 px slope 0.0474だったため、局所差分の圧縮を中心に検証する。

## RQ5: orthomosaicとraw UAV frameでrelative geometryの挙動は異なるか

AirMeasurerのorthomosaic評価と、UAV3DCropのframe-level RGB/MVS評価はGT定義が異なる。両者を同じ誤差値として比較せず、入力表現、camera geometry、GT生成法を揃えたcontrolled experimentでdomain shiftを調べる。

## RQ6: single-image DA2とphotogrammetry CHMをどう組み合わせればrice height estimationが改善するか

DA2 relativeを単独のheight推定器とみなすのではなく、CHM/DEMをground reference、DA2を局所appearance/orderingの補助、plot-level height anchorをcalibrationに使う構成を検証する。ただし、CHM自体のMVS誤差、風、遮蔽、orthomosaic stitchingの影響も独立に評価する。

## 現時点の中心課題

現状で最も重要なのは「relative DA2の値をmeterへ合わせられるか」だけではなく、「未使用のplot・別時期・別圃場でも、意味のあるcanopy height orderとlocal height differenceを保つか」である。AirMeasurerの1時期結果はこの問いの実験設計を示したが、一般的な稲高精度の証拠にはなっていない。
