# 実験計画：既存モデルとB3型融合推論の組合せ

## 目的

既存のC0 epoch5モデル、E3 epoch5モデルをそれぞれ同一モデル内で全体推論とタイル推論に使い、固定済みB3型融合を適用する。再学習は行わず、Validation128で絶対深度と局所深度差を比較する。

## 固定した条件

- T1：C0 epoch5による全体518px推論と元画像2000pxタイル（518px入力）のB3融合
- T2：選択済みE3 epoch5による同じB3融合
- 比較対象：epoch20-B0、epoch20-B3、E3-B0、C0-P0
- GT mask：有限かつ0 < Z <= 65m
- 分割：2025/Day047のValidation128のみ。Independent test319は使用しない
- 固定pair：既存manifest SHA256 `bcb56eab03bec4a44f0626da0f4814876fae67774e536e00cd9f1ad16b1f5728`
- B3：`L(D_global)+D_tile-L(D_tile)`、正のHann融合、215×215 box filter、`BORDER_REFLECT_101`
- FP32、TF32無効。深度のclipping・GTを使う補正・係数調整はしない

## 再現ゲートと実行順

1. checkpoint/config/manifestのSHA256を検証する。
2. 1枚のValidation画像でRGB値域、strict load、C0・E3 forwardを確認する。
3. 同じ画像のepoch20-B3を再推論し、保存済み画像別行のRMSE・全14px MAE・大差分14px MAEを照合する。許容誤差は各1e-5mとする。
4. ゲート通過後にC0-B3とE3-B3を各128画像で実行する。
5. 既存4条件は同じ画像別十分統計から再利用し、6条件を対応画像単位で比較する。

ゲートが失敗した場合は正式な全件推論へ進まず、差異を記録する。

## 評価・統計

絶対MAE/RMSE/AbsRel/Bias/Pearson、全有効14px固定ペアMAE、GT差20cm以上14px固定ペアMAE/RMSE、振幅比、符号一致率を扱う。距離別局所統計も14/28/56/107/214pxで保存する。pool推定量と画像平均推定量を分け、画像単位paired bootstrapを10,000回、seed 42で計算する。ペアがない画像の局所指標は未定義とし0に置き換えない。

## 視覚化標本

B-seriesで事前固定された16-image pilot selectionから、manifest順に各作物×視点層の最初の1画像を抽出した8画像を使う。対象は `fixed8_visualization_indices.csv` に固定し、結果を見て差し替えない。

## 保存・保護

推論・中間データはGPGPU4の`~/gpu_work`（`/localdata`上）へ置く。checkpointはNFSから専用の新規作業ディレクトリへコピーし、コピー元とSHA256照合する。既存checkpoint、過去run、NFSバックアップ、canonical checkoutの変更・削除は禁止する。新規成果を既存PR #6/#7の成果物と混ぜず、PR #7を基点に独立ブランチとPRを作る。
