# 元解像度クロップ追加学習：事前実験計画

| 項目 | 値 |
|---|---|
| 実験ID | `native_resolution_patch_training_20261111` |
| ブランチ | `research/native-resolution-patch-training-20261111` |
| 基点 | PR #5 head `research/local-huber-20261105`、着手時commit `4369cf2` |
| 状態 | 実装とCPU統合確認済み。GPU preflight/pilot待ち。正式GPU学習前に設定とsource hashを固定する。 |

## 1. 目的と仮説

従来の短辺518pxへの全体縮小が、元画像14pxスケールの局所Z-depth差を過小推定させているかを検証する。全群の損失をSiLogに固定し、入力領域の違いを単独要因として比較する。GTは写真測量由来のカメラ座標系Z-depthであり、鉛直草丈を示さない。

- H1：元解像度518pxクロップ学習は、同一推論方式の従来群より14px局所差MAEを改善する。
- H2：1036pxを518pxへ縮小する中間クロップも改善しうる。
- H3：GT差20cm以上の14px振幅比が従来の約0.33から1に近づく。
- H4：改善が14/28/56pxで107/214pxより顕著となる可能性がある。
- H5：小クロップの全体文脈不足により絶対深度RMSEが悪化する可能性がある。
結果を見て仮説や成功基準を変更しない。

## 2. 正本・分割・保全

正本は C:\Users\81902\Documents\Codex\2026-09-20\depth-anything-v2-python-uav-rgb。開始時の未追跡ファイルは変更しない。固定source_full_v1、過去の実験、checkpoint、生RGB/GTを編集・移動・上書きしない。2026-11-11の固定source検証はpass（81 source files、10 signature files、train384、validation128）。GPU側checkpointは正式学習前に実物のSHA256を照合する。

Train384はDay019/028/040、Validation128はDay047とする。Independent test319は設計、pilot、checkpoint選択、評価に一切使わない。正本のtrain/validation manifestを使いSHA256を保存する。学習GTマスクはfiniteかつ0.001<=depth<=65m、評価はfiniteかつ0<depth<=65m。モデル出力上限80m。

## 3. 学習群・前処理

全群は入力518×518、SiLog、batch 1、gradient accumulation 4とする。

| 群 | RGB/GTの切り出しと変換 | 元14pxの入力換算 |
|---|---|---:|
| C0 | 全画像を短辺518pxへ縮小し、寸法を14の倍数へ丸める。RGBはOpenCV bicubic、GTはnearest。縮小後に518 crop | 約1.8px |
| F1 | 元画像座標で1036×1036 crop後、そのcropだけを518へ縮小。RGB bicubic、GT nearest | 約7px |
| F2 | 元画像座標で518×518 crop。リサイズなし | 14px |

読み込み時にRGB/GTの元寸法一致を確認する。F1/F2は元画像全体を事前縮小しない。cropは一様抽出し、予測やGT差分に依存させない。反転確率0.5をRGB/GTに同時適用する。crop座標・倍率・補間・flipを記録する。画像順、crop RNG、flip乱数は群名から独立し、同じimage/drawでは3群で対応するseedを使う。

## 4. データ供給量・予算ロック

seed 42、全画像を均等に使い、各epochの画像順は固定seedでshuffleする。全群共通の予算候補を次に限る。

- 480 optimizer updates：5 epoch、各画像1 crop/epoch（1920 crops/group、5 crops/image）。
- 960 optimizer updates：5 epoch、各画像2 crops/epoch（3840 crops/group、10 crops/image）。

CPU上で固定seedのcrop座標を生成し、F2で各画像のunique pixel coverageを計算する。480 updateと960 updateそれぞれのF2 unique coverageをCPU上で事前計算する。coverage中央値が10%未満なら960を全群に適用し、10%以上なら480とする。この規則を先に固定し、最終的なbudget値はGPU pilot後にconfigへ記録する。GPU pilotのvalidation性能は予算や方式選択に使わない。画像別coverage分布とcrop数を保存する。候補外の変更が必要なら正式学習を開始せず、変更理由を記録して再計画する。

GPU pilotは固定少数画像で実モデルload、CUDA FP16 forward/backward、GradScaler、VRAM、data throughput、checkpoint save/resumeを各群で確認する。既存multiscale実験の事前固定pilot画像でP0/P1/P2の時間、coverage、境界処理を測る。pilot重みは破棄する。正式条件とsource/config hashのlock前に学習しない。

## 5. 初期重み・最適化

C0/F1/F2はすべてepoch20の同一重みから独立に開始する。model.safetensors SHA256：52769e2e95704d0862e708e8772195b861c647dd4812058eaf698a44305c3bcb。config.json SHA256：3c6ab05881e6a76edbfd33ff393fdfb082582084e9262826bb1ac0fd2b1dac7c。GPU実物と照合後、strict=Trueでモデル重みのみロードする。optimizer/schedulerは群ごとに新規作成する。

AdamW（weight_decay=0.01、encoder LR=2e-6、neck/head LR=2e-5）、PolynomialLR（power=0.9）、CUDA FP16 autocast + GradScaler。unscale後にgradient norm 1.0でclipし、AMP skip時はschedulerを進めない。lossはSiLogのみ。epoch末にatomic checkpointを作り、model/optimizer/scheduler/scaler/RNGを保存する。hash/signature不一致のresumeを拒否する。

## 6. 推論方式

すべての学習群を以下で評価する。

| 方式 | 推論入力 |
|---|---|
| P0 | 全体を短辺518pxへ縮小する既存B0 |
| P1 | 元画像の1036×1036 tilesを518へ縮小 |
| P2 | 元画像から518×518を等倍切り出し |

P1/P2の重複率は25%、strideはround(tile×0.75)。最終tileは遠端にanchorし、paddingなしでcoverageを保証する。実tile boxとpixel coverage countを保存する。既存multiscale.pyに合わせ、max(hann,1e-3)の正Hann窓を使った正規化加重平均とする。GTによるscale/offset、画像別補正、係数調整は禁止する。

## 7. 評価・統計

Validation128を共通順序、GT mask、image ID、座標で評価する。主要結果は事前固定した最終epoch。checkpoint選択にはvalidation bestを使わず、各epochのP0値は推移記録に用いる。正式なcross-inferenceは最終epochの全9組合せ。

絶対指標：MAE、RMSE、AbsRel、Bias、Pearson。局所指標：元画像座標で14/28/56/107/214pxの水平・垂直pair MAE/RMSE、振幅比sum|pred delta|/sum|GT delta|、符号一致率、GT差階級。14px全有効pair pool MAEと、固定pair manifest由来の標本（各画像・軸・距離100000候補、seed=42+image_index*10000+stride）のGT差20cm以上MAEを区別する。既存pair_manifest.jsonの座標hashと一致しなければ停止する。

作物、撮影方向、GT深度帯、GT差階級、画像端距離を層別する。P1/P2ではtile境界からの距離とoverlap/non-overlap領域を加える。画像単位paired bootstrapは10000回、seed42、全モデル共通draw。pooled推定値と画像平均を分け、Day047内の不確実性として報告する。

## 8. 成功基準・融合

主要比較はF2-P2対C0-P2。GT差20cm以上・14px固定pair MAEを5%以上低減することを事前目標とする。絶対RMSEを同時に示し、実用性の目安は同方式C0の1.05倍以下とする。RMSEが閾値を超えた場合、局所改善と実用上のtrade-offを分けて報告する。振幅比のみ増加して誤差や他指標が改善しない場合は成功としない。総合点を後付けで作らない。

第一段階は全9組合せの評価までとする。局所改善が認められ、同方式C0比RMSEが1.05を超えた場合に限り、B3型fusionを別実験として検討する。

## 9. GPU・保存運用

GPGPU4を優先する。開始前に他者session/process、VRAM、disk、Singularity、~/gpu_workの実体をread-onlyで確認し、他者利用中は開始しない。データ読み込みと実行中checkpointは~/gpu_work（実体 /localdata/<user>）を使う。過去SIF・初期checkpointは実体、所有権、hashを確認してから使う。OOM等で条件を自動変更しない。checkpointと必要結果は許可されたNFSへ退避しSHA256照合後、自身のGPU/container/SSH資源を解放する。

GitHubにはcode、config、manifest、CSV/JSON、図、日本語報告、checkpoint SHA256/provenanceを含める。大容量checkpoint、生RGB/GT、prediction cache、secretは含めない。

## 10. 限界

ValidationはDay047単日由来、学習seedは1種類であり、未知撮影日への汎化を示さない。写真測量GTの微細境界誤差・平滑化が高解像度学習に与える影響を議論する。Z-depthを草丈精度と呼ばない。結果は探索的評価と明記する。
