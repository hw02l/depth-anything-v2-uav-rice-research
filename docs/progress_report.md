# Depth Anything V2 卒業研究予備実験 進捗報告

## 研究目的

将来的にUAV RGB画像から稲の高さ推定・3D復元を行うため、Depth Anything V2のmetric depth性能と、同一画像内の局所的なdepth差を利用できる可能性を予備評価する。

## これまでの公開データ実験

| dataset | 内容 | 実施数 | 主な結果 |
|---|---|---:|---|
| KITTI | metric depth定量評価 | 50画像 | pooled MAE 2.5921 m、RMSE 5.4692 m、AbsRel 0.1282 |
| ETH3D | disparity GTからmetric depth、点群比較 | 1画像 | MAE 1.4367 m、RMSE 1.8926 m、AbsRel 0.2105 |
| LUMS | 公開稲UAV RGBへの適用 | 9画像 | GTがないため精度評価ではなく、depth/mask/proxyの予備観察 |
| UAV3DCrop | raw metric、local difference、scene/crop/view比較 | 52画像、3 scene | photogrammetry z-depth reference；raw absoluteとlocal geometryを分離評価 |

KITTIではGT距離が大きいほど誤差が増えた。LUMSではraw frameとDEMの対応が確認できず、稲の絶対depth精度を評価していない。

# UAV作物画像でのGround Truth Depth評価

## UAV3DCropについて

UAV3DCropの公式RGB/depth datasetから、canonical `main` revisionの `2025/Day001_Wheat` sceneを使用した。RGBとdepthの1対1対応、`transforms.json` のintrinsics/distortion/extrinsics、`sparse_pc.ply` を実ファイルで確認した。depthは論文の定義に従い、photogrammetry MVSから得られたcamera-frame z-depth（光軸方向、meter）として扱った。

データ監査の詳細は `docs/uav3dcrop_dataset_audit.md`、scene選択の詳細は `docs/uav3dcrop_scene_selection.md` に記録した。

## 使用scene

- scene: `2025/Day001_Wheat`
- crop: wheat
- acquisition date: 公開scene名では `Day001` と表記され、calendar dateは今回の公開metadataから確定できなかった
- 解像度: 5280×3956
- 選択画像: 20枚
- view: nadir 10枚、oblique 10枚
- 選択結果: `results/uav3dcrop/selected_images.csv`

view分類はファイル番号の推測ではなく、`transforms.json` のrotationから `abs(R[2,2]) >= 0.90` をnadirとした。

## Camera geometry

intrinsicsは `fl_x=fl_y=3718.3260`、`cx=2639.0090`、`cy=1932.1209`、camera modelはOPENCVである。distortion係数も `transforms.json` から読み込み、point cloudでは `cv2.undistortPoints` を用いた。RGB、GT、predictionは全て同じ5280×3956 rasterで、評価時にGTを補間していない。

## Absolute metric depth評価

使用モデルは既存実験と同じ `Depth-Anything-V2-Metric-Outdoor-Small-hf`、max depthは80 mである。raw predictionとGT z-depthを同一valid pixel上で比較した。

| 指標 | 20画像 pooled |
|---|---:|
| MAE | 12.9915 m |
| RMSE | 14.1955 m |
| AbsRel | 0.6669 |
| Bias（prediction−GT） | -12.9915 m |
| valid pixels | 417,202,973 |

raw metricはこのwheat UAV sceneではGTより一貫して小さい値を出した。これはcheckpointの出力単位が不明という意味ではなく、出力はmeter単位のmetricモデルだが、学習分布とUAV作物画像のdomain/viewpoint差により絶対値が合わなかったと解釈する。UAV3DCropではnadirとobliqueでGT距離分布も異なるため、view間の単純順位付けはしない。

## Nadir / oblique比較

以下は各画像の指標をviewごとに平均した値である。

| view | images | MAE (m) | RMSE (m) | AbsRel | Bias (m) |
|---|---:|---:|---:|---:|---:|
| nadir | 10 | 9.4131 | 9.4220 | 0.6318 | -9.4131 |
| oblique | 10 | 16.5767 | 17.6872 | 0.7020 | -16.5767 |

obliqueの方が悪い結果となった。ただしscene数1・各10枚であり、作物や飛行条件全体へ統計的に一般化しない。

## Global bias

各画像のmedian biasを予測から引く診断を実施した。画像ごとの平均は corrected MAE 2.5776 m、corrected RMSE 3.4065 mで、rawより改善した。ただしこれは各画像のGTを使って補正量を求める診断であり、推論時に利用可能なraw metric精度ではない。

## Scale+shift alignment

GTに対する最小二乗scale+shiftを画像ごとに推定した。これはmetric性能ではなくgeometry preservationの診断である。画像ごとの平均は次の通りである。

| variant | MAE (m) | RMSE (m) | AbsRel |
|---|---:|---:|---:|
| raw metric | 12.9949 | 13.5546 | 0.6669 |
| median-bias corrected | 2.5776 | 3.4065 | - |
| metric scale+shift | 1.3804 | 1.8162 | 0.0601 |
| relative scale+shift | 1.0922 | 1.3528 | 0.0486 |

scale+shift後の値をraw metricの絶対精度として報告してはいけない。relativeモデルはmeter scaleを持たないため、relativeの値も整合後の形状比較に限る。

## Local depth difference評価

同一画像内で同じ行のpixel pairをランダムサンプリングし、GTとpredictionのdepth差を比較した。以下は20画像の画像平均である。

| variant | separation | Difference MAE (m) | RMSE (m) | Pearson r | sign agreement |
|---|---:|---:|---:|---:|---:|
| metric raw | 8 px | 0.0187 | 0.0392 | 0.0530 | 0.5301 |
| metric raw | 32 px | 0.0508 | 0.0940 | 0.1245 | 0.5440 |
| metric raw | 128 px | 0.1096 | 0.1991 | 0.2294 | 0.5718 |
| relative scale+shift | 8 px | 0.0152 | 0.0351 | 0.1927 | 0.5602 |
| relative scale+shift | 32 px | 0.0404 | 0.0713 | 0.3303 | 0.5765 |
| relative scale+shift | 128 px | 0.0787 | 0.1258 | 0.4540 | 0.6175 |

pixel separationが増えると差分誤差は増える一方、relative scale+shiftでは相関とsign agreementが上がった。これは、絶対depthの大きなoffsetと、局所形状の再現性を分離して評価する必要があることを示す。ただしpairは同一画像内からのサンプリングであり、植物高GTそのものではない。

## Relative depthとの比較

公式実装のrelative outputはmetric depthではなく、affine-invariant inverse-depth系の出力であるため、raw値をmeter GTへ直接比較していない。大きい値が近い方向であることを確認し、z-depthと向きが合うように符号を反転した上でscale+shift整合した。今回のsceneでは、relativeの整合後局所差分はmetric rawより相関が高く、局所形状の予備指標として有望だった。ただし、これはGTを利用した診断であり、絶対距離を復元できたことを意味しない。

## Point cloud評価

同一pixel、同一intrinsics、同一distortion補正rayでGT/pred点を生成した。raw metricの3D Euclidean errorは各画像の平均として以下であった。

| 指標 | 20画像の画像平均 |
|---|---:|
| mean 3D error | 14.7723 m |
| median 3D error | 13.2846 m |
| RMSE | 15.5370 m |
| 95 percentile | 23.6131 m |

nadirのmean 3D errorは10.6543 m、obliqueは18.8903 mであった。PLYは4画像（各view 2画像）についてsingle-view camera/world座標で出力した。さらにnadir 3画像を`transforms.json`のcamera-to-worldで融合し、GT/predのworld-coordinate PLYとoverlayを作成した。

ICP、rigid alignment、scale alignmentはraw metric評価に適用していない。predicted point cloudとGT point cloudの大きなworld Z差は、raw depth biasが点群へ反映されたものとして可視化される。

## LUMS稲実験との関係

LUMSではframe-level GTがなく、canopy-ground depth difference proxyを直接検証できなかった。UAV3DCrop wheatでは、raw絶対depthが大きく外れていても、同一画像内の局所差分を別指標として計測でき、relative整合出力では差分形状の相関が改善する傾向を確認した。この結果はLUMS proxyを検証する方法論を与えるが、作物種・scene・camera条件が異なるため、稲でも同じ精度になるとは結論づけない。

## Canopy height GT

UAV3DCrop論文にはcanopy-height recoveryの評価があるが、今回取得した公開RGB/depth/transforms/sparse PLYだけでは、height測定点・ROIと今回のframeを対応付ける公開ファイルを確認できなかった。そのためheight MAE/RMSE/correlationは計算していない。推定depthを植物高と呼ばず、今後はframe/ROI対応付きheight annotationを取得できた場合だけ実施する。

## 分かったこと

- UAV3DCropの公開ファイルだけで、RGBとdense reference z-depth、intrinsics、extrinsicsを同一pixelで対応付けられた。
- このwheat sceneでは、現在のoutdoor metric checkpointのraw絶対depthは大きな負のbiasを持った。
- oblique viewはnadirよりraw評価・point errorが悪かった。
- global offsetを補正した診断では誤差が減るが、補正量はGT依存である。
- relative scale+shiftは、局所depth差の相関・順序一致を調べる補助指標として使える可能性がある。
- 同一pixel rayとcamera poseを使ったsingle-view点群と、3画像のworld-coordinate fusionを実装できた。

## 分からないこと

- 稲で同じlocal depth difference性能が得られるか。
- groundとcanopyのどのROI定義が植物高と対応するか。
- photogrammetry MVS depthの誤差が今回のreference側にどの程度含まれるか。
- camera altitude、姿勢、intrinsics、DEMを利用した補正でraw metricを実用的に改善できるか。

## 卒業研究への示唆

植物高推定では、raw metric depthをそのまま稲高へ変換するのではなく、(1) camera geometryからground planeを推定し、(2) vegetation/canopy ROIを定義し、(3) ground-to-canopyの局所差分を評価し、(4)実測または対応付きDEM/height GTで検証する必要がある。次段階では、稲を含むframe-level RGB-depthまたはcamera pose付きデータを用い、同一画像の局所差分をcrop heightへ接続する。

# 局所深度差の詳細評価

## なぜMAEだけでは不十分なのか

今回の目的は、絶対depthのbiasが残っていても、植物高に相当する局所的な深度差の大きさと方向が保存されるかを調べることである。しかし、GT差分がほぼ0のpairを常に0と予測するだけでもLocal Difference MAEは小さくなる。そこで、zero baseline、Pearson/Spearman、sign agreement、回帰slope、GT差分magnitude別評価を追加した。

## GT depth differenceの分布

追加sceneを含む52画像（Wheat 20、Oat 16、Corn 16）では、32 px pairのGT `|Delta d|` は平均0.040656 m、中央値0.015625 m、標準偏差0.131434 mであった。25 percentileはCSVに記録した通り0 m、95 percentileは0.15625 mであり、0.02 m未満のpairが多数を占める。この分布が、数cmのMAEを解釈する際の重要な前提である。

## Height difference別評価

32 pxのmagnitude binでは、metric rawのLocal Difference MAEは0–0.02 mで0.02299 m、0.10–0.20 mで0.14389 m、0.20–0.50 mで0.29280 m、0.50–1.00 mで0.63113 m、1.00 m以上で1.84857 mであった。relative scale+shiftではそれぞれ0.01221, 0.12577, 0.25928, 0.55228, 1.73288 mであった。差分の大きさが増えるほど誤差も増え、raw metricの相関は各binで低かった。relativeは大きい差分でmetric rawよりPearson/Spearmanとsign agreementが高かったが、これはGTへscale+shift整合した診断値である。

## Zero baselineとの比較

32 pxについて、zero baselineのMAEは0.040656 m、metric rawは0.053259 m、relative scale+shiftは0.040874 mであった。したがって全pairをまとめたMAEだけでは、metric rawはzero baselineより良くなく、relativeも改善は0.000未満の非常に小さい差である。128 pxではzero 0.0860709 m、metric raw 0.119323 m、relative scale+shift 0.083601 mであった。

## Sign agreement

32 pxでのsign agreementは、`|Delta d_GT| >= 0.02/0.05/0.10/0.20/0.50 m` に対して、metric rawが0.5529/0.5879/0.6184/0.6345/0.6424、relative scale+shiftが0.6133/0.6893/0.7481/0.8002/0.8415であった。差分が大きいsubsetではrelativeの順序一致率は上がるが、metric rawでは0.64程度に留まった。zero baselineのsign agreementは定義上0とした。

## Regression slope

GT差分の絶対値が0.10, 0.20, 0.50, 1.00 mの±20%範囲で、metric rawの回帰slopeは0.1220, 0.1551, 0.1071, 0.1077、relative scale+shiftは0.1779, 0.2110, 0.3064, 0.2491であった。いずれも1よりかなり小さく、今回のsampleでは局所差分を強く圧縮している可能性がある。相関もmetric rawでは低く、relativeでも1 m付近のPearsonは0.581、R²は0.338であり、完全な再現とは言えない。

## 複数sceneでの再現性

32 pxのmetric rawは、Wheat MAE 0.05102/Pearson 0.0795/sign 0.5433、Oat 0.07087/0.0962/0.5215、Corn 0.03844/0.0840/0.5104であった。relative scale+shiftはWheat 0.04028/0.3296/0.5758、Oat 0.05510/0.3294/0.5606、Corn 0.02739/0.2648/0.5435であった（順にMAE/Pearson/sign）。scene数と画像数が少ないため、作物一般の優劣とは解釈しない。

## Nadir / Obliqueの影響

52枚をview別にまとめた32 pxのmetric rawは、nadir MAE 0.03528/Pearson 0.1098/sign 0.5264、oblique MAE 0.07124/Pearson 0.0816/sign 0.5309であった。relative scale+shiftはnadir 0.02495/0.2482/0.5440、oblique 0.05680/0.2822/0.5817であった。obliqueはMAEが大きかったが、これはview別のGT分布やscene構成も影響し得るため、撮影角度だけの因果効果とは断定しない。

32 pxの回帰slopeはmetric rawでnadir 0.0637、oblique 0.0493、relative scale+shiftでnadir 0.0469、oblique 0.1545であった。これはviewごとのgeometry保持の差を示す診断値だが、scene構成とGT分布が混在しているため、撮影角度単独の効果ではない。

## Relative / Metric比較

raw metricは絶対値を出すが、今回のUAV3DCropではpooled MAE 13.5787 m、RMSE 14.7136 m、AbsRel 0.6570、Bias -13.5787 mで大きな負biasがあった。relativeはmeterとして直接評価できないためscale+shift整合した。32 pxではrelativeの局所相関とsign agreementがmetric rawより高かったが、これは診断上のgeometry比較であり、relativeがmetric depthとして正しいことを意味しない。

## LUMS稲画像への示唆

LUMSで得たcanopy-ground depth difference proxy（early -0.0641～0.0955 m、middle -0.4297～1.6496 m）は、対応GTのない仮説的proxyである。今回の他作物MVS referenceでは0.1 m付近のmetric raw MAEが0.1147 m、slope 0.122、Pearson 0.073、0.2 m付近ではMAE 0.2048 m、slope 0.155であった。一方relative scale+shiftでも0.1 m付近MAE 0.0922 m、slope 0.178、0.2 m付近MAE 0.1755 m、slope 0.211である。したがってLUMSの数十cm以上のproxyを植物高と読むには、少なくとも対応GTとROI定義を追加し、差分方向・圧縮を検証する必要がある。作物種が異なるため、UAV3DCropの値を稲へ直接転用しない。

## 現段階で言えること

- 小さいLocal Difference MAEだけでは局所geometry再現の証拠にならない。
- 今回の52画像ではGT差分の多くが小さく、zero baselineが強い。
- metric rawのabsolute depthは大きなglobal biasを持ち、local differenceのslopeと相関も低かった。
- relative scale+shiftは今回のdiagnostic local geometry指標ではmetric rawより相関・sign agreementが高い傾向を示した。
- 0.1–1.0 m級の差分では予測差分がGTよりかなり小さく、差分圧縮が示唆された。

## 言えないこと

- 5 cmのLocal Difference MAEを、5 cm精度の稲高推定とは言えない。
- Wheat/Oat/Cornで良かった結果を稲へ外挿できない。
- MVS z-depthを手測りcanopy height GTとみなせない。
- 少数sceneのnadir/oblique差から一般的な撮影角度効果を確定できない。

## 次の研究課題

地面とcanopyの対応ROIを持つ稲画像を用意し、camera geometry・ground plane・vegetation mask・height GTを同時に定義する必要がある。その上で、局所差分のmagnitude別slope、sign agreement、relative/metric、nadir/obliqueを同一圃場・同一時期で評価する。MVS referenceを使う場合はconfidenceと風・遮蔽failureを併記する。

# Rice canopy height validation using AirMeasurer data

## Dataset

AirMeasurer公式V2.0.2 releaseのsmall-field testing dataから、`20220505.tif` と同名の `20220505.las` を取得した。全testing dataは約11.3 GBとされるが、今回取得したのは約33 MBのsmall-field archiveだけである。orthomosaicは3676 x 3749、4 band、EPSG:32650、GSD 0.00257 mであった。LASは1,190,990点である。RTK shapefileには4個のPointZ control pointがあるが、plot polygonやplot IDではない。

実ファイルの監査は `docs/airmeasurer_dataset_audit.md` にまとめた。今回の取得物にはearly/middle/lateの対応ラベル、plot-level実測height、raw frameのcamera poseは確認できなかったため、1時期の探索実験として扱う。

## Point cloud / CHM generation

公式論文に記載されたSOR、CSF、GCP/terrain correction、WhiteboxToolsによるsurface生成、CHM差分、H90/H95の方針を確認した。今回の再実装ではOpen3D SOR、Python CSF、0.02 m gridのDSM/DEM rasterization、`CHM = DSM - DEM`、GeoTIFF transformを用いたorthomosaic gridへのbilinear reprojectionを実行した。入力1,190,990点からSOR後1,131,782点、CSF ground 832,975点を得た。source CHM gridは473 x 482で、出力単位はmである。

ただし、WhiteboxToolsのTIN、GCPによるfield slope補正、公式plot mask、H90/H95のplot-level処理は完全再現していない。したがって、今回のCHMは公式方針に沿った再実装であり、公式GUIの完全一致出力ではない。詳細は `docs/airmeasurer_height_method.md` に記録した。

## Why metric depth cannot be directly evaluated on orthomosaic

orthomosaicは複数のUAV画像から再構成された地図画像であり、単一のpinhole camera frame、camera pose、intrinsicsを持たない。したがって、Depth Anything V2 Metricの値を「カメラから表面までのmetric z-depth」として評価していない。今回の主要実験は `Depth-Anything-V2-Small-hf` のrelative outputとCHM heightのgeometry比較である。

## Relative depth evaluation

1024 x 1024、overlap 128のtileでrelative推論し、overlap重み付き合成でorthomosaic格子へ戻した。公式仕様に従いrelative outputのnear-large orientationを保持し、CHMの高いcanopyとの比較で恣意的な反転は行っていない。raw relativeとCHMの相関は、Pearson 0.2579、Spearman 0.2098であった。これらはmeter精度ではなく、CHMとの順序・形状の一致度を表す探索指標である。

## Pixel-level correlation

CHM-validかつ非黒RGBの4,958,593 pixelを評価対象とし、相関計算用には300,000 pixelをsampleした。CHMのsample heightはmedian 0.0065 m、mean 0.0586 m、95 percentile 0.4028 m、最大1.2697 mであった。

CHMを使ってrelative scoreをheightへscale+shift整合した診断では、scale 0.07898、shift -0.05926 m、MAE 0.0791 m、RMSE 0.1450 m、R² 0.0665となった。これはGT/CHMを使った後処理結果であり、uncalibrated single-image height精度ではない。

## Height calibration

CHM medianを全pixelのconstant baselineとした場合、MAE 0.0563 m、RMSE 0.1588 mであった。per-date scale+shift DA2のMAE 0.0791 mはbaselineより良くなかったが、RMSEは0.1450 mでbaselineの0.1588 mを下回った。相関が低く、height binの誤差がheight増加とともに大きくなるため、「calibrationにより稲高が推定できた」とは結論できない。

plot boundaryと対応height表がなかったため、plot-level H90/H95、70/30 held-out plot、5-fold、時期をまたぐcalibrationは実行していない。`airmeasurer_plot_metrics.csv` には未実施理由を保存した。

## Height-bin behavior

CHM height bin別のDA2/CHM相関は、0.0–0.1 mでPearson 0.1561、0.1–0.2 mで0.0164、0.2–0.4 mで0.0191、0.4–0.6 mで-0.0127、0.6–0.8 mで0.0242、0.8–1.0 mで0.0917、1.0 m以上で-0.4864であった。0.2 m以上のbinはpixel数が少ないものもあり、一般化はできないが、今回のsampleでは高いcanopy heightの順序保持が弱い。

## Cross-date evaluation

今回実際に処理したのは20220505の1時期だけである。したがって、生育時期による性能差、earlyでfitしてmiddle/lateへ適用するcalibration、date間のscale安定性は判定できない。

## Local height difference

camera z-depthではなく、CHMのcanopy height差をlocal height differenceとして8/16/32/64/128 pixelで評価した。結果は以下の通りである。

| separation | pairs | mean |Δh_GT| (m) | Local Difference MAE (m) | RMSE (m) | Pearson | Spearman | sign agreement | slope |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 px | 67,055 | 0.0268 | 0.0269 | 0.0751 | 0.2012 | 0.1450 | 0.5308 | 0.0144 |
| 16 px | 66,781 | 0.0385 | 0.0383 | 0.1028 | 0.3225 | 0.1968 | 0.5425 | 0.0270 |
| 32 px | 64,569 | 0.0514 | 0.0508 | 0.1305 | 0.4448 | 0.2733 | 0.5542 | 0.0474 |
| 64 px | 61,016 | 0.0654 | 0.0642 | 0.1546 | 0.5249 | 0.3599 | 0.5786 | 0.0687 |
| 128 px | 54,143 | 0.0815 | 0.0806 | 0.1776 | 0.5267 | 0.4305 | 0.5956 | 0.0805 |

32 pxでは相関は0.4448まで得られたが、slopeは0.0474で1から大きく離れている。したがって、差分が大きく圧縮されている可能性があり、MAE 5.08 cmを稲高精度と解釈してはいけない。これはCHM-MVS側の誤差と、orthomosaic/DA2のdomain mismatchの両方を含む診断結果である。

GT local height differenceの大きさを±20%相当のbinで抽出した結果は次の通りである。これは全pixelのheight binではなく、同一画像内のpair差分に対する結果である。

| target `|Δh_GT|` | pairs | mean `|Δh_GT|` (m) | MAE (m) | RMSE (m) | Pearson | sign agreement | slope |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 m (0.08–0.12) | 9,664 | 0.0986 | 0.0890 | 0.0926 | 0.3852 | 0.6834 | 0.0985 |
| 0.2 m (0.16–0.24) | 9,019 | 0.1963 | 0.1827 | 0.1856 | 0.4826 | 0.7411 | 0.0698 |
| 0.5 m (0.4–0.6) | 6,155 | 0.4909 | 0.4637 | 0.4680 | 0.6956 | 0.8881 | 0.0554 |
| 1.0 m (0.8–1.2) | 1,246 | 0.8773 | 0.8301 | 0.8348 | 0.8241 | 0.9615 | 0.0536 |

大きい差分ほどsign agreementは上がったが、slopeは0.05–0.10程度で、予測差分の振幅はGT差分より大きく圧縮されている。従って「方向の一致」と「差分量の正確な再現」は分けて解釈する必要がある。この追加CSVは `results/metrics/airmeasurer/airmeasurer_local_height_difference_by_magnitude.csv` である。

## Failure analysis

主な候補は、orthomosaic stitching、密集canopy、細葉・遮蔽、影・裸地・水面、field edge、plot boundary欠如、簡略CHM生成である。今回のデータだけではDA2側とMVS-CHM側の誤差を完全分離できない。RGB/CHM/relative/aligned/errorの代表図と、確認済み事実・推測の区別は `docs/airmeasurer_failure_analysis.md` にまとめた。

## Relation to LUMS

LUMS riceではground referenceがないため、canopy-ground depth differenceはproxyに留まっていた。今回初めてrice imageryと同日LASからCHMを作ったが、1時期・plot対応なし・公式CHM完全再現ではないため、LUMS proxyを直接検証したわけではない。ただし、CHMを使った今回のrelative相関が弱く、height difference slopeも小さいことから、LUMSの -0.0641～0.0955 m（early）や -0.4297～1.6496 m（middle）を植物高と呼ぶには、対応するground/canopy GTとgeometry確認が必要だと再確認した。

## Relation to UAV3DCrop

UAV3DCropはRGB、camera intrinsics/extrinsics、dense MVS z-depthがframe単位で対応していたのに対し、AirMeasurerはorthomosaicとCHMの比較である。両者の数値を同じmetricとして順位付けしてはいけない。一方、UAV3DCropで見られた「metric rawのdomain bias」と「relativeのgeometry診断を分離する必要性」はAirMeasurerにも整合する。AirMeasurerではrelativeを使っても、今回の1時期ではheight orderingの証拠は弱かった。

## Current findings

- A: orthomosaicへmetric modelをcamera-frame depthとして直接適用する方針は不適切である。
- B: relative outputをCHMと比較することは可能だが、今回の相関は弱く、局所height順序の確立には不十分である。
- C: 既知CHMでscale+shiftすれば診断的にはmeterへ変換できるが、constant baselineをMAEで上回らず、held-out/cross-dateの汎化も未検証である。
- したがって、現段階で最も研究価値があるのは、relative出力をground plane・plot/vegetation ROI・少数の独立height anchorと組み合わせ、未使用plot・別時期で検証する設計である。

## Limitations

1. small-fieldの1 dateのみ。
2. plot boundary、plot ID、height tableがない。
3. CHMは公式Whitebox/GCP/plot pipelineの完全再現ではない。
4. orthomosaic入力であり、raw frame camera geometryを持たない。
5. MVS/point-cloudの細葉・風・遮蔽誤差を除外できない。

## Next steps

次は、(i) plot境界と対応height表を持つAirMeasurer full testing dataの必要最小限subset、または(ii) raw rice UAV frameとcamera pose/ground referenceが対応する公開データを取得する。そこでplot-level held-out calibration、early→late transfer、CHM confidence別評価、raw frameでのcamera-ground-canopy height変換を行う。研究室データを使う段階では、実測稲高を明示的に分離して評価する。
