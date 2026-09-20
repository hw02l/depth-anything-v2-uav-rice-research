# UAV3DCrop scene選択

## 選択scene

`2025/Day001_Wheat` を選択した。

理由は次の通りである。

- wheatは稲と同じく細葉・草本系の作物で、UAV俯瞰画像の視覚条件が今回の研究目的に近い。
- 2025年sceneであり、論文が説明するcanopy-height測定年と同じ年だが、後述の通り個別height GTの対応表は公開ファイルから確認できない。
- RGB、depth TIFF、transforms.json、sparse_pc.plyの対応が揃っている。
- 1 scene内にnadirとobliqueの両方があり、viewpoint差を同一作物・同一local frameで調べられる。

corn/soybean/oatも候補だったが、まずはscene内のRGB/depth対応とcamera geometryを優先し、全量取得を避けて1 sceneに限定した。

## 取得範囲

| 区分 | 選択枚数 | frame filename |
|---|---:|---|
| nadir | 10 | 1, 43, 85, 127, 169, 447, 489, 531, 573, 616 |
| oblique | 10 | 182, 240, 298, 357, 415, 674, 732, 791, 849, 908 |
| 合計 | 20 | `results/uav3dcrop/selected_images.csv` に記録 |

実データは各RGB JPGと同名stemのdepth TIFFで取得した。画像はすべて5280×3956である。選択理由・view分類・source revision・対応depth pathは `results/uav3dcrop/selection_manifest.json` と `selected_images.csv` に固定的に記録した。

## view分類

READMEのファイル番号を単純にnadir/obliqueへ割り当てず、`transforms.json` のcamera-to-world rotationを使用した。camera optical-axisの鉛直性スコアを `abs(R[2,2])` とし、0.90以上をnadir、それ未満をobliqueとした。scene全908 frameではnadir 381、oblique 527であり、選択20枚は各viewの先頭から等間隔に近い候補を採用した。

## 再現コマンド

```powershell
.venv\Scripts\python.exe scripts\prepare_uav3dcrop.py `
  --scene 2025/Day001_Wheat `
  --nadir-count 10 `
  --oblique-count 10
```

推論は次である。

```powershell
.venv\Scripts\python.exe scripts\infer_uav3dcrop.py --batch-size 1
```

使用モデルは既存プロジェクトと同じ `Depth-Anything-V2-Small-hf` と `Depth-Anything-V2-Metric-Outdoor-Small-hf` であり、metric modelのmax depth設定は80 mである。

## 追加scene（local geometry再現性確認）

local depth differenceの再現性を確認するため、同じcanonical `main` releaseから次の2 sceneを追加した。

| scene | crop | available frame | available view | selected |
|---|---|---:|---|---:|
| `2025/Day001_Oat` | oat | 700 | 203 nadir / 497 oblique | 8 + 8 = 16 |
| `2025/Day007_Corn` | corn | 1210 | 237 nadir / 973 oblique | 8 + 8 = 16 |

Wheatと形状の近い草本作物としてoatを、比較対象としてcornを選んだ。これは作物種の一般性能を決めるためではなく、同一実験コードでscene/cropの違いがどの程度現れるかを見るための小規模診断である。各sceneの選択結果、RGB/depth URL、view分類、intrinsics/extrinsics availabilityは `results/uav3dcrop/all_selected_images.csv` に記録した。

追加sceneの取得コマンドは次の通りである。

```powershell
& .venv\Scripts\python.exe scripts\prepare_additional_uav3dcrop_scenes.py `
  --scenes 2025/Day001_Oat 2025/Day007_Corn `
  --nadir-count 8 --oblique-count 8
& .venv\Scripts\python.exe scripts\combine_uav3dcrop_selections.py
```

全52枚の推論済みデータに対する統合評価は `results/metrics/uav3dcrop_all/` に保存した。追加sceneは全量取得しておらず、各sceneでRGB 16枚、対応depth TIFF 16枚、`transforms.json`、`sparse_pc.ply`だけを取得した。
