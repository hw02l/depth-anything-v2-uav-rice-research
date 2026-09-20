# AirMeasurerのheight/CHM処理と今回の再現

## 公式処理の要約

論文と公式リポジトリの説明から、AirMeasurerの植物高処理は概ね次の流れである。

1. SfMで生成した3D point cloudを入力する。
2. Statistical Outlier Removal（SOR）で外れ点を除去する。
3. Cloth Simulation Filter（CSF）でground-levelとabovegroundの点を分離する。
4. GCP/RTK情報を使い、terrain featureやfield-level slopeの影響を補正する。
5. WhiteboxToolsの `LidarTinGridding` 等で地表側・上部側のraster surfaceを作る。
6. DSMとDEMの差からCHM（canopy height model）を作る。
7. plot maskを作り、作物生育時期に応じてheight percentileを計算する。

論文本文には、grain filling後のplant heightとしてH90th、seedling establishmentのような初期評価ではH95thを使う説明がある。また、plot maskは測定目的に応じて縮小率を変え、field edgeや隣接plotの影響を避ける設計になっている。

なお、論文中のDSM/DEM説明には「ground-level points」と「aboveground points」の命名が通常のGIS用語と逆に読める箇所がある。本プロジェクトでは、計算上の役割を優先し、ground pointsからDEM（ground surface）、全表面点からDSM（upper surface）を作り、`CHM = DSM - DEM` とした。公式の内部出力を完全に再現したと主張せず、この命名差を明記する。

## 今回の実装

実装は `scripts/create_rice_chm.py` で行った。処理内容は以下である。

- Open3DのSOR: `nb_neighbors=20`, `std_ratio=2.0`
- SOR後のpoint cloudを `CSF` Python packageでground / non-ground分類
- source point-cloud grid: 0.02 m
- DSM: SOR後の全点のcell maximum
- DEM: CSFで得たground pointのcell minimum
- empty cell: 観測済みcellからのnearest fill
- CHM: `max(DSM - DEM, 0)`
- orthomosaic gridへの変換: GeoTIFFのCRS、extent、transformを使ったrasterioのbilinear reprojection/resampling
- output unit: meter

今回の実行結果は `20220505` について、入力1,190,990点、SOR後1,131,782点、CSF ground 832,975点、source CHM grid 473 x 482であった。

## 公式処理との差

今回のPython実装は、公式AirMeasurerのGUIやWhiteboxTools処理をそのまま呼び出したものではない。特に以下は簡略化・未再現である。

- GCPを使った公式のfield slope / terrain correction
- 公式WhiteboxTools `LidarTinGridding` のTIN補間
- 公式GUIのfield/plot segmentation
- plot mask縮小率とH90/H95のplot-level集計
- 公式の全時期データに対するquality-control手順

そのため、今回生成したCHMは「公開LASから、公式方針に沿う主要要素を現在の環境で再実装したCHM」であり、AirMeasurer公式GUIから出力した同一CHMではない。実行可能な処理差はmanifestにも保存した。

## 今回の数値評価における位置づけ

CHMはorthomosaic格子へ地理的に再投影した。単純な画像resizeではない。DA2 relative出力はorthomosaic tileへ推論し、overlapを持つtileの重み付き合成で元格子へ戻した。

relative scoreをCHMのmeter heightへscale+shift整合した結果は、CHMを使った診断値である。これは、既知heightを使わずに得た単眼RGBからのheight推定精度ではない。plot boundariesとheight tableがないため、AirMeasurer論文と同じH90/H95のplot-level評価は今回実施していない。
