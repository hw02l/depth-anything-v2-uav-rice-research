# UAV3DCrop reference depthの限界

## 参照した資料

- [UAV3DCrop project page](https://link-dev.github.io/UAV3DCrop/)
- [UAV3DCrop論文](https://arxiv.org/abs/2608.06404)
- [UAV3DCrop GitHub](https://github.com/Link-dev/UAV3DCrop)
- [UAV3DCrop RGB dataset](https://huggingface.co/datasets/Link-Dev/UAV3DCrop)
- [UAV3DCrop depth dataset](https://huggingface.co/datasets/Link-Dev/UAV3DCrop_depth)

## 確認できた事実

UAV3DCropの公開sceneには、raw UAV RGB frame、frame対応のdepth TIFF、sceneごとの `transforms.json`、`sparse_pc.ply` がある。今回取得した3 sceneでは、RGBとdepth TIFFは同じstemで対応し、shapeは5280×3956で一致した。depthは論文・dataset実装の定義に合わせ、photogrammetryによるdense MVS referenceのcamera-frame z-depth [m]として扱った。これはcamera rayに沿ったEuclidean distanceではない。

`transforms.json` にはOPENCV intrinsics、distortion係数、各frameのposeがある。現在のcanonical `main` releaseでは、今回確認した2025 sceneの `scale=1.0`、`avg_pos="0 0 0"` であり、legacy normalizationを追加していない。RGB、GT depth、predictionは同じraster上で評価し、GTのresize/interpolationは行っていない。

論文では、UAV3DCropのdepth referenceがSfM・bundle adjustmentを含むphotogrammetry/MVS reconstructionから得られること、またcanopy heightの別評価でground/canopyのpercentile定義と手測り点が使われることが説明されている。しかし、今回取得した公開treeだけでは、canopy-height手測り点を各RGB frameのpixel/ROIへ結び付ける対応表を確認できなかった。

## MVS referenceに固有の制約

以下はMVS/植物表面で一般に問題になり得る点であり、今回の各pixelの誤差を直接確認したという意味ではない。

- 細い葉や葉先は、複数viewで十分な対応特徴が得られず、depthが欠落・平滑化・誤対応になる可能性がある。
- 風による葉の移動や撮影時刻差があると、複数画像を同一静止物体として扱うMVS仮定が破れる可能性がある。
- 稲・麦などの反復的な葉模様は、誤った対応や過度な平滑化につながる可能性がある。
- 密なcanopyでは、見えていない地面や下位葉のdepthはMVS referenceでも直接観測されない。
- 影、反射、低テクスチャ、水分を含む地面では、MVSの品質が低下する可能性がある。
- GT TIFFがdenseであっても、それは手測りした植物高さではなく、reconstructed surfaceのz-depthである。

上記のうち「UAV3DCropがMVS/photogrammetry referenceを使う」「height benchmarkは別のground/canopy定義を用いる」は公式資料で確認した事実である。「細葉・風・反復textureが個々のpairに与えた誤差量」は今回のデータから分離して測定していないため、技術的な可能性として記述する。

## 今回の評価で避けた誤解

1. reference depthを植物高GTと呼ばない。
2. scale+shift alignment後の誤差をraw metric精度と呼ばない。
3. `sparse_pc.ply`との近傍距離をdense depthの正解として扱わない。
4. 相関が低いままlocal MAEだけを植物高精度の証拠にしない。
5. 1枚のcamera-frame point cloudをposeなしに別viewのpoint cloudと比較しない。

## 今回の研究での扱い

reference depthは、同一pixelでのabsolute z-depth、local depth difference、camera intrinsicsを使ったsingle-view point cloudの比較には利用できる。一方、植物高そのものを検証するには、地面reference、canopy定義、frame/ROI対応を含むheight GTが必要である。したがって今回の結論は「公開MVS z-depthに対するDA2の挙動」であり、稲高の実測精度ではない。

## 今後必要な検証

- MVS referenceのconfidence/validity情報が公開されていれば、低信頼pixelを分けて再評価する。
- 手測りground/canopy heightとframe/ROIを対応付けたデータでheight評価を行う。
- 風・反復texture・遮蔽を含むfailure caseを人手確認し、GT側の誤差とDA2側の誤差を分ける。
- 多視点poseを用いたGT/predicted surface整合を、raw metric評価と分離して評価する。
