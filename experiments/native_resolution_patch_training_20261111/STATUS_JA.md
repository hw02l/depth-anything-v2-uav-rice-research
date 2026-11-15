
# 訂正評価ステータス（2026-11-15）

## 完了済み

- `evaluate_patch.py`、`evaluate_all.py`、`inference.py`のRGB値域を修正し、0〜255のRGB画像を一度だけ0〜1へ変換してImageNet正規化する仕様に統一した。
- RGB完全一致、P0/P1/P2の前処理、入力dtype・値域を確認する回帰テストを追加した。GPGPU4で単一画像のGPU forward smoke testを実施した。
- 初期checkpoint SHA256 `52769e2e95704d0862e708e8772195b861c647dd4812058eaf698a44305c3bcb` のepoch20 B0をValidation128で再評価し、過去基準を許容誤差 `1e-5` 以内で再現した。
- C0/F1/F2の既存epoch5 checkpointを再利用し、Validation128の全9条件を修正版コードで再推論した。再学習はしていない。
- 訂正結果は `experiments/native_resolution_patch_training_20261111/analysis/rgb_input_fix_20261114/` に分離し、集計CSV、paired bootstrap、画像別例、B0ゲート、GPU smoke結果、provenanceを保存した。
- F2-P2対C0-P2の主要比較（GT差20cm以上、元画像14px固定ペア）では局所MAEが `0.1169906817m` から `0.1023338303m` へ低下し、相対改善は `12.53%` であった。10,000反復、seed 42、128画像のpaired bootstrapを実施した。
- 2026-11-14初回評価はRGB float32 `[0,1]` をuint8へ直接変換し、画素情報を失っていたため、`invalid_pre_fix` として無効化した。旧ファイルは削除・上書きしていない。Independent test319は未使用である。

## 未完了

- 訂正済み報告書・進捗記録・必要成果物を既存ブランチ `research/native-resolution-patch-training-20261111` へcommitし、PR #6へpushする。
- 訂正内容を反映したPR #6本文を保存し、表示を確認する。
- 訂正済みファイルを研究室NFSの専用RGB修正バックアップへ追加し、コピー後のSHA256を再検証する。

GPUでの追加推論・学習は停止済みである。既存checkpoint、元データ、修正前の評価記録は変更していない。
