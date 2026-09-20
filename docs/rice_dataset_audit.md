# 公開稲UAVデータセット監査

## 監査対象

- リポジトリ: [LUMS-WIT/infrastructure-free-uav-phenotyping](https://github.com/LUMS-WIT/infrastructure-free-uav-phenotyping)
- 参照ブランチ: `main`
- 監査日: 2026-09-20
- カメラ: DJI Mini 4 Pro RGB camera とREADMEに記載
- README記載の撮影期間: 2025-08-13〜2025-10-03、13回のweekly flight

READMEだけでなく、GitHub Contents APIとrecursive treeで`data/`以下の実ファイルを確認した。

## 利用可能なデータ

Gitリポジトリ内では次の構成を確認した。

- `data/2025-08-22`、`2025-09-12`、`2025-10-03`など、日付別フォルダ
- 2025-08-13と2025-08-27は`link_to_data.txt`のみで、Google Driveの外部リンクを指す
- その他11日付フォルダには、公開サンプルのPNG RGB frameが各10枚ずつ存在
- 今回ダウンロードしたPNGはすべて3840×2160 pixels
- 今回使用したファイルはraw RGB frameであり、orthomosaicではない
- PNGにはEXIFが残っている。今回の選択画像では、焦点距離24.0、35mm換算24、ISO、露光時間、F値1.7、GPS緯度経度、GPS altitude、ImageUniqueIDを確認した

READMEには、6m flightをorthomosaic用、10m flightをDEM用に使ったこと、4K videoから6フレーム間隔でframeを抽出したこと、13時期の撮影、2 plot（約30m×10m）が記載されている。これらはdataset documentation上の撮影条件であり、各raw frameのframe-specific AGLではない。

## 利用できない、または今回取得していないデータ

recursive Git treeで確認した範囲では、以下のファイルはGitリポジトリ内に存在しなかった。

- orthomosaic本体
- DEM本体
- corrected DEM本体
- DEMのGeoTIFF等の座標系・georeference情報
- RGB frameとorthomosaic/DEMを結ぶカメラpose・frame index対応表
- per-image camera intrinsic matrixまたはdistortion係数
- ground-truth canopy heightの表形式データ
- plot ID、row/plot polygon、各RGB frameの位置対応表

READMEには08-13と08-27のfull datasetを指すGoogle Driveリンクがあるが、今回の実験では外部ストレージ全量を無条件に取得していない。従って、READMEに記載されたorthomosaic、DEM、RANSAC correction、manual canopy-height validationが、この9枚のraw frameにどのように対応するかは確認していない。

## EXIF altitudeの扱い

選択9枚のEXIF GPS altitudeは以下の範囲だった。

- 2025-08-22: 約390.995m
- 2025-09-12: 約307.817〜307.917m
- 2025-10-03: 約259.141〜259.241m

これはGPS記録値であり、地面からの撮影高度（AGL）であることを確認できない。したがって、これをDepth Anything V2のground-depth ground truthやUAV-to-ground距離とは解釈しない。READMEに記載された6m/10mはflight/product acquisitionの説明として扱う。

## RGBとGTの対応可否

今回のGitHub公開ファイルだけでは、次の対応は確認できなかった。

1. raw RGB frameの各pixelをDEMの地理座標へ投影するためのcamera pose
2. raw RGB frameとorthomosaic/DEMのframe-levelまたはtimestamp-level対応
3. canopy heightのground-truth値を各RGB frameへ対応付ける情報

従って、今回の9枚について、Depth Anything V2 metric depthをDEMまたは実測稲高とpixel単位で比較することは実施しない。対応がないままDEMをresizeして比較すると、地理座標・視点・投影の不一致を誤差として数えるためである。

## 今回実施可能な実験

- 公開raw RGB frameへのDepth Anything V2 relative depth適用
- `Depth-Anything-V2-Metric-Outdoor-Small-hf`によるmetric depthの探索的適用
- EXIFとdataset documentationの撮影条件を併記したdepth統計の確認
- RGBのみからの説明可能なvegetation mask比較（ExG、VARI、HSV）
- ground candidateが画像内に存在するかの保守的な目視・appearance-based判定
- ground medianとcanopy depth percentileの差を、`canopy-ground depth difference proxy`として記録
- failure case、depth profile、時期群の分布比較

## 今回実施できない実験

- この9枚に対する絶対metric-depth精度（MAE/RMSEなど）のground-truth評価
- DEMまたはmanual canopy heightに対する稲高精度評価
- 異なる日付の画像を同一位置として扱った縦断的な生育量評価
- RGB frameをorthomosaic/DEMへ厳密に投影した3D比較

## 選択データ

今回は公開GitHubからearly/middle/lateとして各3枚、合計9枚を取得した。

- early: 2025-08-22
- middle: 2025-09-12
- late: 2025-10-03

選択の詳細、URL、解像度、EXIFは`results/rice_uav/selected_images.csv`に保存した。

## 結論

本データセットは、公開raw RGBへDA2を適用し、植生・視点・密集canopyに対する挙動を調べるには利用できる。一方、現状の公開Git treeだけでは、raw frameとDEM/実測稲高の幾何学的対応が不足しているため、今回の主張は「UAV稲画像への探索的適用」と「depth差proxyの検討」に限定する。
