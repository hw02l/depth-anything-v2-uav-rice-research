# AirMeasurer rice orthomosaicでのfailure analysis

## 確認できた事実

今回の入力は単一のraw UAV frameではなく、Pix4Dmapperで生成されたgeoreferenced orthomosaicである。RGB previewでは、複数のrice plot、裸地または畦に相当する領域、影、画像外の黒領域、回転したfield layoutが確認できた。単一camera poseを持たないため、DA2のmetric出力をcamera-frame z-depthとして解釈していない。

LASから再生成したCHMは、orthomosaic格子上のsampleで次の分布になった。

- min: 0.0000 m
- median: 0.0065 m
- mean: 0.0586 m
- 95 percentile: 0.4028 m
- max: 1.2697 m

DA2 relativeとCHMのsample-level相関はPearson 0.2579、Spearman 0.2098であった。CHMを使ったscale+shift診断ではMAE 0.0791 m、RMSE 0.1450 m、R² 0.0665で、CHM median constant baselineのMAE 0.0563 mよりMAEが良くならなかった。したがって、今回のaligned MAEだけから高さ構造の再現を主張できない。

CHM height bin別では、0.2 m以上のbinで相関がほぼ0または負となり、aligned errorはheight magnitudeとともに大きくなった。これは、少なくとも今回のsmall-field・1時期・簡略CHM条件では、高いcanopy領域の形状を安定して再現できていない可能性を示す。

## failure要因の整理

### Orthomosaic stitching / non-pinhole入力

確認済みの事実として、入力は複数画像から再構成されたorthomosaicである。tile境界や広域の滑らかなrelative gradientが見られるため、単一視点の透視幾何では説明できない出力が含まれる可能性がある。これはDA2だけの失敗と断定できず、DA2が想定する自然画像の透視・局所奥行きとorthomosaic表現のdomain mismatchが主な候補である。

### Canopy closure / narrow leaves

RGBには密集した葉群がある。CHMはSfM/MVS由来のpoint cloudから作られているため、細い葉、反復テクスチャ、遮蔽ではCHM側にも欠測・平滑化・誤対応が起こり得る。今回の点群だけから、どの誤差がDA2由来でどの誤差がMVS由来かを画素単位に完全分離することはできない。ここは推測であり、MVS confidenceや元画像bundleを追加しない限り確定できない。

### Ground / soil / water / shadow

圃場の非植生領域、裸地、影が同じRGB画像内に存在する。単純な黒領域除外は行ったが、soil、水面、暗い葉、影を完全に分離するsegmentationは行っていない。CHMの低い領域が真のgroundとは限らず、CHM median baselineが強く見える理由の一つとして、低height pixelが多数あることが考えられる。ただしこの因果は今回のデータだけでは確定できない。

### Field edge / neighboring plot overlap

small-fieldには複数plotと周辺の裸地が含まれるが、plot boundary shapefileは取得できなかった。そのため、plot edge、隣接plot、通路を分離したplot-level評価は実行していない。pixel-level correlationには、作物以外の領域と境界画素が混在する。

### CHM generation approximation

今回のCHMは公式GUIのWhiteboxTools TIN、GCP terrain correction、公式plot maskを完全には再現していない。SOR・CSF・raster max/min・nearest fillは実際に実行したが、ground surfaceの補間方法とterrain correctionが異なる可能性がある。したがって、CHM高誤差の一部はreference側の再生成差に由来し得る。

## 今回の結論

今回のfailureは、単一の「DA2が植物を見失った」という問題ではない。少なくとも、(1) orthomosaicを単一camera画像として入力したdomain mismatch、(2) plot/ground mask不足、(3) MVS-CHMの品質限界、(4) dense canopyと影・境界の混在、が同時に存在する。次の実験では、plot boundaryと公式CHM、またはraw frameとcamera geometryが揃った稲データを使い、これらを分離する必要がある。
