# ETH3D ground truth 評価方法

## 選択

今回は **A: low-res two-view の ground-truth disparity と calibration から metric depth を復元する方法** を採用した。

ETH3D low-res two-view の `im0.png`、`disp0GT.pfm`、`calib.txt` は同じrectified stereo画像に対応している。`calib.txt` の `cam0` と `baseline` を使えば、各画素の disparity をカメラ前方のZ深度へ直接変換できる。小規模な1シーンでRGB、depth、calibration、点群を同一座標系で扱えるため、今回の「既存成果物を拡張して検証する」という目的に適している。

Bのhigh-res multi-view training dataは、より密な3D geometryを得られる可能性がある一方、データ構成・カメラ姿勢・深度の対応付けが増え、今回の予備実験に必要な最小構成を越える。将来、複数視点の復元や表面精度を主題にする段階で追加する。

## disparity から metric depth への変換

ETH3Dのrectified stereoで、次式を用いた。

```text
Z[m] = fx[pixel] * baseline[mm] / (disparity[pixel] + doffs[pixel]) / 1000
```

今回の `delivery_area_1l` では、`fx=541.764`、`fy=541.764`、`cx=553.869`、`cy=232.396`、`baseline=59.9101 mm`、`doffs=0` である。PFMの非有限値、0以下のdisparity、変換後の非有限値は無効画素として除外した。

実装は `src/da2_experiments/eth3d.py` の `read_pfm`、`read_eth3d_calib`、`disparity_to_depth_m` にまとめ、`scripts/evaluate_eth3d.py` から使う。

## depth評価

ETH3Dの `delivery_area_1l` は屋内シーンであるため、Depth Anything V2の屋内metric model `depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf` を使う。モデルの最大深度20 mに合わせて `gt <= 20 m` を評価対象とし、同じ `im0` サイズへ戻した出力をETH3Dから変換したZ深度と比較する。MAE、RMSE、AbsRel、SqRel、RMSE log、delta指標を保存する。比較図はRGB、GT depth、predicted depth、absolute errorの4枚構成とした。

## 点群比較の座標系判断

予測点群とGT点群は、同じ `im0` RGB画像、同じETH3D `cam0` intrinsics、同じ画素位置から、同じZ深度の式で生成する。そのため今回の点群比較では、rigid transformation、scale alignment、ICPを適用していない。両者を別カメラ・別時刻から復元していないため、単純な座標系の不一致をICPで隠す必要がない。

同じ画素の無効値パターンや点群の間引きの違いがあるため、実装上はOpen3DのKD-treeによる双方向nearest-neighbor距離を計算する。これは厳密な表面Chamfer評価ではなく、同一カメラ座標系での予備的なgeometry差である。`pred_to_gt_mean_m`、`gt_to_pred_mean_m`、対称平均最近傍距離、中央値、95 percentileを保存する。

`scripts/compare_eth3d_pointcloud.py --visualize` を使うと、予測（青）とGT（赤）の点群をOpen3Dで表示できる。GUIのない環境ではJSON計算だけを実行する。

## データ範囲上の限界

low-res two-viewのdisparity GTはstereo評価用であり、全ての遮蔽領域・細部表面を完全に覆うground-truth meshではない。したがって本結果は「同一画像のmetric depthおよびカメラ座標点群の予備検証」と解釈し、植物の3D表面復元性能へ直接一般化しない。

参照: [ETH3D datasets](https://www.eth3d.net/datasets)、[ETH3D two-view evaluation](https://github.com/ETH3D/two-view-evaluation)
