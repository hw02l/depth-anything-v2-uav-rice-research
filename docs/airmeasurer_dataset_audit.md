# AirMeasurer公開データセット監査

## 参照した公式資料

- [The-Zhou-Lab/UAV-AirMeasurer](https://github.com/The-Zhou-Lab/UAV-AirMeasurer)
- [AirMeasurer V2.0.2 release](https://github.com/The-Zhou-Lab/UAV-AirMeasurer/releases/tag/V2.0.2)
- [Sun et al., New Phytologist (2022)](https://nph.onlinelibrary.wiley.com/doi/full/10.1111/nph.18314)
- [公開論文PDF](https://ueaeprints.uea.ac.uk/id/eprint/86781/1/nph_18314_final.pdf)

## 取得したリリースと範囲

公式V2.0.2 releaseのasset構成を確認した。full testing dataは分割zipで、release上では次の2019年8時期のorthomosaic/LASペアが確認できる。

`20190720`, `20190807`, `20190814`, `20190822`, `20190827`, `20190912`, `20190926`, `20191008`

一方、今回の実験では約33 MBの `AirMeasurer_testdata_small_field.zip` だけを取得した。したがって、全11.3 GB相当のtesting dataを取得・処理した実験ではない。

取得したsmall-fieldの構成は次の通りである。

| 種類 | 実ファイル | 確認結果 |
|---|---|---|
| orthomosaic | `2D_Orthomosaic_image/20220505.tif` | 3676 x 3749、4 uint8 bands、GeoTIFF |
| point cloud | `3D_Point_cloud/20220505.las` | LAS 1.2、point format 3、1,190,990 points |
| RTK/GCP情報 | `RTK_Information/small_mtps.{shp,shx,dbf,prj}` | PointZ 4点、`mtp5`–`mtp8` |

同一日を示すファイル名 `20220505` とsmall-field releaseのディレクトリ構成により、今回のorthomosaicとLASは同一データセット内の同日データとして対応付けた。ただし、この対応は画像pixelと点群点の1対1対応を意味しない。

## 画像・座標情報

`20220505.tif` の実測metadataは以下の通りである。

- CRS: EPSG:32650（WGS 84 / UTM zone 50N）
- raster size: width 3676、height 3749
- bands: 4 uint8 bands（RGB + alpha相当）
- pixel size / GSD: 0.00257 m x 0.00257 m
- bounds: x = 706245.23148–706254.67880、y = 3500191.70572–3500201.34065
- software metadata: Pix4Dmapper
- `AREA_OR_POINT=Area`

LASの座標系はLAS VLRおよび座標範囲からEPSG:32650として扱えることを確認した。LASの座標範囲はorthomosaicより広く、LAS全域をそのまま画像pixelへ貼り付けることはできない。そのため本実装では点群から地理座標付きのsource rasterを作り、その後rasterioのreprojection/resamplingでorthomosaicのtransform・shapeへ合わせた。

## 公開されていたもの／確認できなかったもの

### 利用できたもの

- georeferenced rice orthomosaic TIFF
- 同日と判断できるLAS point cloud
- EPSG:32650のCRS
- RTK/GCP用の4個のPointZ shapefile record
- 論文・公式コードに記載されたpoint-cloud処理の方針

### 今回のsmall-field取得物だけでは利用できなかったもの

- plot polygon、plot ID、plot boundary shapefile
- orthomosaicのpixelまたはplotに対応した手測りplant height table
- acquisition metadataとしてのearly/middle/lateラベル
- individual raw UAV frame、camera pose、frame-level intrinsics
- S1/S2のheight値と今回のsmall-field plotを対応付けるキー

RTK shapefileの4点は `mtp5`–`mtp8` というPointZ control pointであり、plot境界やplot単位のheight annotationではない。したがって、今回のデータだけからplot-level height評価を作成していない。

## 実施できる比較

今回実施したのは、同一dateのorthomosaicとLASから作成したCHMを、同じorthomosaic格子上のDepth Anything V2 relative出力と比較する探索実験である。relative出力はmeter値ではないため、raw valueについてPearson/Spearmanを計算し、CHMを使ったscale+shiftは診断目的に限定した。

- pixel-level relative score vs CHM heightの相関
- CHM height binごとの相関・診断MAE
- CHMを使ったscale+shift後のMAE/RMSE/R²
- CHM median constant baselineとの比較
- 同一orthomosaic内のlocal canopy-height difference
- RGB、CHM、relative map、high-error blockの可視化

## 実施できない比較

- Depth Anything V2 metric出力をcamera-frame metric depthとして評価すること。orthomosaicには単一pinhole camera frame、単一camera pose、単一camera intrinsicsがないためである。
- plot-level H90/H95とDA2 scoreの対応評価。plot境界とheight tableがないためである。
- held-out plot評価、5-fold評価、early→middle→late calibration。今回取得物は1 dateかつplot対応なしであるためである。
- AirMeasurer S1/S2のheight値を今回の画像へ結び付けること。公開ファイル間の対応キーを確認できないためである。
- raw RGB frameのcamera-depth精度評価。今回のRGBはorthomosaicである。

## 今回の解釈上の結論

このsmall-fieldは、稲の実画像と同日点群からCHMを再生成し、relative canopy geometryを探索的に調べるには利用できる。一方、これだけでDepth Anything V2の稲高精度、plot-level汎化、時期をまたぐcalibrationを主張することはできない。特に、scale+shiftで得たMAEはCHMを使って後から校正した診断値であり、uncalibrated single-image inferenceの精度ではない。
