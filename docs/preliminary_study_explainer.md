# Depth Anything V2予備実験を理解するための解説レポート

この文書は、専門外の人にも実験の意味が分かるように、既存の予備実験を順番に説明するものである。学術論文が「他人に正確に説明する文書」であるのに対して、こちらは「自分が何を確かめ、何がまだ分からないかを理解する文書」である。数値は既存のCSVとJSONから引用しており、実験を追加していない。

## 1. この研究で最終的にやりたいこと

最終的にやりたいことは、UAVで上空から撮影した稲のRGB画像を使って、稲の高さや圃場の3D形状を推定することである。理想的な流れは次の通りである。

```text
UAV RGB画像
    ↓
Depth Anything V2
    ↓
画像中の深度情報
    ↓
地面と稲冠を分離
    ↓
地面から稲冠までの高さを推定
    ↓
稲圃場の3D復元
```

ここで一番大切なのは、Depth Anything V2が直接「稲の高さ」を出すわけではないことである。モデルが出すのは、基本的にはカメラから表面までの距離、または画像内の前後関係である。稲高を出すには、地面をどこに置くか、カメラがどこにありどちらを向いているか、scaleをどう合わせるかが必要である。

## 2. そもそもdepthとは何か

画像の各画素に「その場所がどのくらい遠いか」を割り当てたものがdepth mapである。例えば、カメラの前に机と壁がある場合、机は5 m、壁は10 mのように距離を持つ。depth mapは、普通のRGB画像の赤・緑・青の代わりに、距離の値を画素ごとに持つ画像である。

ただし、画像だけから距離を完全に知るのは難しい。同じ大きさに見える物体が、近くの小さい物体なのか、遠くの大きい物体なのかは、単眼画像だけでは一意に決まらない。学習モデルは、物体の見え方、テクスチャ、影、透視図法などから距離を推測している。

## 3. Relative depthとMetric depth

relative depthは「AはBより近い」という順序を表す深度である。例えば、モデル出力がA=8、B=3なら、Aの方が近いと解釈できる。しかし8が8 mとは限らない。単位がなく、画像ごとにscaleが変わる。

metric depthは「Aはカメラから5.2 m」のように、メートル単位の距離を出すことを目標とする。今回のプロジェクトでは、relativeに `Depth-Anything-V2-Small-hf`、屋外metricに `Depth-Anything-V2-Metric-Outdoor-Small-hf`、屋内metricに `Depth-Anything-V2-Metric-Indoor-Small-hf`を使った。

metric modelだからどの画像でも正しいわけではない。屋外道路画像で学習したモデルを、上空から見た作物画像へ入れると、距離の基準がずれることがある。これがdomain shiftである。

## 4. なぜいきなり稲で評価しなかったのか

予測が正しいかどうかを知るには、正解値、つまりground truth（GT）が必要である。稲画像を見てdepth mapが自然に見えても、それだけでは正しいとは言えない。

そこで、最初にGTが使える一般データセットで、次に3D評価ができるデータセットで確認した。その後、GTがない実際の稲画像へ進み、最後に作物画像とGT depthが対応するUAV3DCrop、稲orthomosaicと点群があるAirMeasurerを使った。

研究の流れは次のようになっている。

1. 一般画像でDA2を実行できるか確認する。
2. KITTIでmetric depthの基本性能を見る。
3. ETH3Dでdepthから3D点群を作る。
4. LUMSの実稲画像へ適用し、失敗の仕方を見る。
5. UAV3DCropで作物画像とGT depthを直接比較する。
6. Wheat、Oat、Cornへ広げ、local geometry評価を追加する。
7. AirMeasurerのrice orthomosaicとCHMで、稲の高さ構造を探索する。

## 5. KITTIで何を確認したか

KITTIは車から撮影した道路画像であり、レーザで測ったdepth GTを持つ。一般的な屋外画像に対して、metric modelがどの程度動くかを見るために50画像を使用した。

最終的な全valid pixelの結果は、MAE 2.5921 m、RMSE 5.4692 m、AbsRel 0.1282である。MAEは誤差の絶対値の平均、RMSEは大きな誤差を強く反映する値である。

距離別には、0–10 mでMAE 0.7002 m、10–20 mで1.7930 m、20–40 mで4.7977 m、40–60 mで13.7114 m、60–80 mで20.1499 mであった。遠くなるほど誤差が大きくなった。

この実験から分かるのは、「このmetric modelは少なくともKITTIのような屋外道路画像ではdepthを出せる」ということである。分からないのは、「同じ精度で稲を測れるか」である。道路と稲圃場は見え方も距離分布も異なるからである。

![KITTIのGTと予測のscatter](../outputs/kitti/plots/gt_vs_pred_scatter.png)

## 6. ETH3Dで何を確認したか

ETH3Dでは、左右のステレオ画像のずれであるdisparityを使って、正解depthを作った。近い物体ほど左右画像の位置ずれが大きく、遠い物体ほど小さくなる。

簡単な式は次の通りである。

```text
depth = 焦点距離 × stereo baseline ÷ disparity
```

実際には補正値と単位変換を含めた式を使った。ETH3Dでは屋内metric modelを使い、1画像でMAE 1.4367 m、RMSE 1.8926 m、AbsRel 0.2105であった。

depth mapから3D点群を作るときは、各画素をカメラから出る光線に置く。画素座標を `(u,v)`、焦点距離を `fx,fy`、主点を `cx,cy`、depthを`Z`とすると、概念的には、

```text
X = (u - cx) / fx × Z
Y = (v - cy) / fy × Z
Z = depth
```

となる。RGBを点の色に使えば、colored point cloudになる。

ETH3Dでは同じカメラ座標で予測点群とGT点群を作った。pred→GT平均距離は0.6751 m、GT→pred平均距離は1.1517 m、対称平均は0.9134 m、95 percentileは3.5264 mであった。これにより、depthから点群を作る処理自体を確認できた。ただし1画像中心であり、稲の3D復元性能を示すものではない。

## 7. LUMS稲UAVで何が分かったか

LUMSのデータは実際の稲圃場をUAVで撮影したRGB画像である。early、middle、lateから各3画像、計9画像を使った。稲画像へDA2を適用すると、relative depthとmetric depthの画像を出力できた。

metric depthの画像ごとの中央値はおよそ5.3–6.8 mであり、公開説明の約6 mという飛行高度と同じオーダーだった。しかし、これは正しさの証明ではない。GPSの記録高度は地面からの高さとは限らず、各pixelのdepthが撮影高度そのものでもないからである。

vegetation maskは、RGBの色から稲らしい画素を分ける処理である。ExG、VARI、HSVを比較し、今回はVARIを採用した。だが黄化した葉、土、乾いた草、水面、影は色だけで完全に分けられない。

ground depthとcanopy depthの差を取ると、earlyでは約-0.0641〜0.0955 m、middleでは約-0.4297〜1.6496 mとなった。ただしこれはcanopy-ground depth difference proxyであり、稲高ではない。地面候補が本当に地面か、カメラ姿勢がどうか、groundとcanopyが同じ幾何条件かを確認できないからである。

LUMSで重要だったのは、綺麗なdepth mapが出ても「正しい」とは限らないことを実際に確認したことである。

## 8. UAV3DCropが重要だった理由

UAV3DCropでは、RGB画像と同じframeに対応するMVS depth TIFFがある。さらにcamera intrinsicsとextrinsicsもある。つまり、初めて「UAV作物RGB＋対応GT depth」を同じpixelで比較できた。

使用画像はWheat 20、Oat 16、Corn 16の計52画像である。nadirが26、obliqueが26である。GTはphotogrammetryで作ったcamera-frame z-depthであり、稲の手測りheightではない。

このdatasetを使ったことで、metric depthの絶対誤差だけでなく、同じ画像の2点間のdepth差も調べられた。これは、稲高のように「地面と稲冠の差」が重要な研究に近い評価である。

## 9. Metric depthの大きな失敗

UAV3DCrop 52画像のraw metric結果は、MAE約13.5787 m、RMSE約14.7136 m、AbsRel約0.6570、Bias約-13.5787 mであった。

Biasが-13.6 mというのは、予測がGTより大幅に小さい方向へずれているという意味である。例えばGTが15 mなのに、予測が1–2 m付近になると、ほとんどの画素で約-13 mの誤差になる。

これはmetric modelが「メートルを出す」と設計されていても、UAV crop画像のdomainでは距離基準が合わないことを示す。原因候補は、道路画像と作物画像の違い、上空・斜めからの視点、葉の反復パターン、canopyの遮蔽である。ただし、今回の結果だけで「DA2はUAVで使えない」とは言えない。評価したsceneとmodelが限定されているからである。

## 10. 「でもlocal MAEは5 cmだった」の罠

ここが最も重要である。UAV3DCropの32 px pairでは、relative alignedのLocal Difference MAEが約0.0409 m、つまり約4.1 cmだった。これだけ見ると、4 cm精度で高さが測れたように見える。しかし、その解釈は間違いである。

32 px pairのGT差分の中央値は0.015625 m、約1.56 cmである。つまり多くのpairは、そもそも本当の差がほとんど0である。

全部0と予測するzero baselineを考える。

```text
GT差分: 0 cm, 0 cm, 2 cm, 4 cm
予測:   0 cm, 0 cm, 0 cm, 0 cm
```

この場合の誤差は0、0、2、4 cmで、平均は1.5 cmである。差が小さいpairが多ければ、何も予測していないのにMAEは小さくなる。

実際のUAV3DCrop 32 px結果では、zero baseline MAEは0.040656 m、metric rawは0.053259 m、relative alignedは0.040874 mだった。relative alignedはzero baselineをほとんど改善していない。このため「MAE 4 cm」はgeometry preservationの十分な証拠ではない。

必要なのは、MAEだけでなく、次の指標を見ることである。

- Pearson/Spearman: GT差分が増えたとき、予測差分も増えるか
- sign agreement: どちらが近いかの順序が合うか
- regression slope: 差分の大きさをどの程度再現するか
- GT difference magnitude: 10 cmや50 cmなど意味のある差を含むか

## 11. 相関とslope

相関は「一緒に増減するか」を見る。例えば、本当の差が1 cm、2 cm、3 cmと増えるとき、予測が0.5 cm、1 cm、1.5 cmでも相関は高くなり得る。増減の順番は合っているからである。

slopeは「本当の変化量をどのくらいの大きさで再現したか」を見る。理想は1である。GT差分が50 cmのとき予測差分も50 cmならslopeは1に近い。GTが50 cmなのに予測が3 cm程度なら、slopeは非常に小さくなる。

UAV3DCropの0.5 m差分では、relative alignedのPearsonは0.5458、sign agreementは0.8369だったが、slopeは0.3064であった。方向は比較的合っていても、振幅を小さく出している。

## 12. Sign agreement

sign agreementは、2点のどちらが近いかを当てた割合である。例えば、GTで `A-B > 0` なら、Aの方が遠い。予測でも正なら順序が一致する。

これは「順位」を見る指標であり、「差が何cmあるか」は見ない。AirMeasurerの1.0 m差分binではsign agreement 0.9615と高かったが、slopeは0.0536であった。つまり大きい差の向きは合っていても、差分量は大きく圧縮されている。この2つは別の性能である。

## 13. 実際の稲で調べたAirMeasurer

AirMeasurerでは、rice orthomosaicとpoint cloudを使った。用語を整理する。

- orthomosaic: 複数の写真をつなぎ、地図のように正射化した画像
- point cloud: 3D空間中の大量の点
- DSM: 植物や地表を含む上側の表面
- DEM: 地面の表面
- CHM: 植物の高さに相当する表面差

概念的には、

```text
植物の上端  ───── DSM
                 │
                 │ CHM = DSM - DEM
                 │
地面       ───── DEM
```

である。今回の実装ではLASへSORとCSFを適用し、ground側からDEM、表面側からDSMを近似し、CHMを作った。CRSはEPSG:32650、GSDは0.00257 mで、同じGeoTIFF transformを使ってCHMをorthomosaic pixelへ合わせた。

ただし、今回のCHMはAirMeasurer公式GUIの完全再現ではない。WhiteboxToolsのTIN、GCPによるterrain補正、plot mask、H90/H95の集計は完全には再現していない。また、orthomosaicは通常の一枚のcamera imageではないので、ここでmetric depthをcameraからの距離として評価してはいけない。

## 14. AirMeasurerの結果

AirMeasurer 20220505では、DA2 relativeとCHMのPearsonが0.2579、Spearmanが0.2098だった。CHMを使ってrelative出力をscale+shiftすると、MAE 0.0791 m、RMSE 0.1450 m、R² 0.0665だった。

constant baselineは、すべてのpixelをCHMの中央値で埋める予測である。baselineのMAEは0.0563 m、RMSEは0.1588 mだった。DA2はRMSEでは少し良かったが、MAEではbaselineより悪かった。

これは「DA2が全く情報を持たない」と断定する結果ではない。local height differenceの1.0 m binでは、Pearson 0.8241、sign agreement 0.9615となったからである。しかし、1時期・1圃場であり、slopeは0.0536だった。したがって、今回の結果は「DA2が稲高を測れる証明」ではなく、「相関と差分圧縮を同時に調べる必要があることを示した結果」である。

![AirMeasurerのRGB、CHM、DA2 relative](../outputs/airmeasurer/report_figures/figure_2_rgb_chm_relative.png)

## 15. 今回の結果を一言で言うと

「Depth Anything V2は、稲の高低関係を部分的に捉えている可能性はあるが、その出力値をそのまま稲高として使える段階ではない」

となる。特に、absolute metric depthはUAV作物画像で大きなbiasが生じた。relative depthは順序を見る用途では可能性があるが、差分の大きさを圧縮するため、そのまま数十cmの稲高へ変換できない。

## 16. 何が問題なのか

### Domain shift

モデルが学んだ画像の種類と、実際に入力する画像の種類が違う問題である。道路、室内、UAVの作物では、視点・物体・距離・テクスチャが違う。

### Canopyとthin leaves

稲冠は細い葉が大量に重なり、奥の地面が見えにくい。葉の形は似ており、反復テクスチャになりやすい。単眼画像から奥行きを決める手がかりが少ない。

### Ground visibility

植物高には地面基準が必要だが、密集期には地面がほとんど見えない。画像の非植生pixelを全部groundとみなすと、水面、影、畦、黄化葉を誤る。

### Orthomosaic

orthomosaicは写真を地図化した画像であり、1枚のカメラが見た透視画像ではない。画像の場所によって異なる撮影画像が使われるため、通常のcamera-frame depthの意味が失われる。

### MVS error

AirMeasurerやUAV3DCropのGTにも、風、遮蔽、細葉、反復模様による誤差が含まれる可能性がある。予測だけでなくGT側も完全ではない。

### Camera geometryとscale

camera intrinsicsは焦点距離や主点、extrinsicsはカメラの位置・姿勢である。これらがあれば画素を3D rayに変換でき、ground planeと組み合わせて高さを計算しやすくなる。relative depthだけではscaleがないため、既知heightやDEMが必要である。

## 17. 卒業研究では何をすればよいか

次の実験では、できれば以下を同じ画像に揃える必要がある。

- raw UAV frame
- camera intrinsics
- camera pose（extrinsics）
- AGLまたは地面基準
- ground planeまたはDEM
- canopy ROI
- 実測rice height

まず地面の位置を決め、camera rayとground surfaceからground depthを得る。次に稲冠の代表値、例えば上位percentileを決め、両者の差を高さ候補にする。その後、既知heightを使うcalibrationをtrain plotとtest plotに分けて評価する。同じplotで校正と評価をすると、過学習した数字になるからである。

relative depth＋外部scale、photogrammetry CHM＋DA2、単眼推定だけの3方式を比較する価値がある。特に、別時期でも同じcalibrationが使えるかを検証する必要がある。

## 18. 用語集

- **monocular depth estimation**: 1枚のRGB画像から深度を推定する方法。
- **relative depth**: meterではなく、画像内の前後関係を表す深度。
- **metric depth**: meterなど絶対単位を持つことを目標とする深度。
- **depth map**: 各画素にdepth値を持つ画像。
- **point cloud**: 3D座標を持つ点の集合。
- **SfM**: 複数画像からカメラ姿勢と疎な3D構造を推定する方法。
- **MVS**: 複数視点画像から密な3D表面を推定する方法。
- **DEM**: 地面の高さを表すモデル。
- **DSM**: 植物や建物を含む上側表面のモデル。
- **CHM**: DSMからDEMを引いた canopy height model。
- **orthomosaic**: 複数写真をつないだ地図状の正射画像。
- **camera intrinsics**: 焦点距離、主点、歪みなどカメラ内部の情報。
- **camera extrinsics**: カメラの位置と姿勢。
- **MAE**: 誤差の絶対値の平均。
- **RMSE**: 二乗誤差の平均の平方根。大きな誤差に敏感である。
- **AbsRel**: GTに対する相対的な絶対誤差。
- **Bias**: 予測誤差の平均。正負の偏りを表す。
- **Pearson correlation**: 2つの値が線形に一緒に増減する度合い。
- **Spearman correlation**: 値そのものではなく順位が一緒に増減する度合い。
- **regression slope**: 予測差分がGT差分の何倍の大きさかを表す傾き。
- **baseline**: 比較の基準となる単純な予測。今回はzeroまたはconstantを使った。
- **domain shift**: 学習時と評価時の画像domainが違うこと。
- **calibration**: 既知の基準値を使い、モデル出力を別の単位へ合わせること。

## よくある誤解

### 誤解1: Depth mapが綺麗だから正しい

可視化は値を色へ変換しているだけであり、正しさを保証しない。GTと数値比較が必要である。

### 誤解2: MAEが5 cmだから高さを5 cm精度で測れる

GT差分がほぼ0のpairが多いと、全部0予測でもMAEが小さくなる。zero baseline、correlation、slopeを見る必要がある。

### 誤解3: Relative depthはmeter単位である

relative outputにmeterの意味はない。scale+shiftを使った場合も、GT依存の診断値である。

### 誤解4: Metric depthならどんな画像でもmeter単位で正しい

metric checkpointでも、道路で学習したモデルを作物へ使えばdomain shiftが起こる。UAV3DCropで大きなBiasが確認された。

### 誤解5: Correlationが高ければ絶対値も正しい

予測が本当の変化の半分でも、増減の順番が合えば相関は高くなり得る。slopeとMAEを併記する必要がある。

### 誤解6: Scale+shift後のMAEが良ければRGBだけでその精度が出る

scaleとshiftをGTから求めているなら、そのGTを使わない推論の精度ではない。train/testやheld-out plotを分ける必要がある。

### 誤解7: Orthomosaicを通常のcamera imageと同じように扱える

orthomosaicは地図画像であり、単一camera poseの透視画像ではない。metric camera-frame depthとして評価してはいけない。

## まとめ

本予備実験で、DA2を実際の稲画像へ適用すること自体はできた。しかし、絶対metric depthは作物UAV domainで大きくずれ、relative depthも差分振幅を圧縮した。したがって、現在の最も妥当な研究方針は、DA2の出力をそのまま稲高と呼ぶことではなく、camera geometry、ground reference、外部scale、実測heightを組み合わせて、局所geometryと高さ推定を別々に検証することである。
