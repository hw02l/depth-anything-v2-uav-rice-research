# 作業状況（最終）

| 項目 | 値 |
|---|---|
| 最終更新 | 2026-11-14 |
| 実験ID | `native_resolution_patch_training_20261111` |
| branch | `research/native-resolution-patch-training-20261111` |
| PR base | `research/local-huber-20261105`（PR #5のhead） |
| 状態 | C0/F1/F2の各5 epoch学習、固定epoch5のValidation128全9条件評価、解析、報告、NFSバックアップが完了。 |

## 完了内容

- C0/F1/F2を共通のepoch20初期重みから各5 epoch学習した。C0は960 successful updates、F1/F2は各958（AMP skipによる2回差）。追加学習は行っていない。
- 固定epoch5 checkpointを用いて、全3学習群×P0/P1/P2の9組合せをValidation128で評価した。Independent test319は一度も読み込んでいない。
- 主比較F2-P2対C0-P2の14px・GT差20cm以上のpair MAEは0.447917→0.447938m（差+0.000020m、画像単位bootstrap 95% CI [-0.000054,+0.000094]）。事前登録した5%改善基準を満たさなかった。
- F2-P2 RMSEはC0-P2より0.923m低い一方、F2自体も7.542mである。全条件RMSE 7.542〜8.651mは過去の同一初期epoch20評価0.794239mと大きく食い違う。初期重みSHAとstrict loadは一致したが、差の原因は未解決であり、実用性や解像度変更の因果効果を確証できない。
- 報告書、集計CSV、3枚のSVG図を本実験ディレクトリに掲載した。完全な画像別評価・bootstrap・条件別結果・学習ログ・provenanceはNFSバックアップに含めた。
- 条件別解析では有効pair数0の層を未定義値として扱う修正を加え、全条件を別出力先へ再実行した。図の生成はmatplotlib依存を外し、標準ライブラリだけでSVGを作成した。
- 最終epoch5 checkpoint SHA256はC0 `2fd190727c7a5124c56c10e0a2dd8cb09c6cb62a0f92b34fce97ad394281b35a`、F1 `ea1f74c5ff17c0b266152588b5d53f8b24611418f125f2fbf4d9cc7447e49f67`、F2 `8124d72a7008c9f3a93dbcce34142d1593063fba2da6a1a58306d55aaf201538`。
- NFS最終バックアップ：`/home1/eng-std/s0623003/gpu4_research_backup_native_resolution_patch_training_20261114_final`。サイズ4.3 GiB、218ファイル。SHA256SUMS 217件すべて検証PASS。保管したブランチsnapshotはcommit `b24ecfeaebf076cad3df4fe3f3cd8f6265e6d8`。epoch5 checkpoint 3件のコピー後SHAも実行元と一致した。raw RGB/GTおよび予測画像cacheは含まない。
- 最終確認でGPGPU4のGPU utilization 0%、メモリ15 MiB使用 / 8098 MiB空き。ユーザー所有の学習・評価プロセスは残っていない。

## 成果物

- `report_ja.md`：手法、結果、bootstrap、条件別結果、重大な再現性留保、成功基準判定。
- `analysis/final_20261114/`：summary、primary comparisons、training history、P0 history、尺度別MAE、条件ハイライト、画像例、checkpoint SHA、SVG図3点。
- `analyze_conditions.py`：空層を安全に扱う修正済み。
- `plot_results.py`：外部描画ライブラリ不要のSVG生成。

## 再現性・運用上の確認

- 初期model SHA256: `52769e2e95704d0862e708e8772195b861c647dd4812058eaf698a44305c3bcb`
- 初期config SHA256: `3c6ab05881e6a76edbfd33ff393fdfb082582084e9262826bb1ac0fd2b1dac7c`
- 学習manifestはTrain384、評価はValidation128。test319は未使用。
- GPU作業はCodex右側の接続済みGPGPU4端末から実施し、研究室のGPU利用ルールに従った。学習・評価後にGPUを解放した。
- 再学習、test評価、GTを使った予測補正、fusionは実施していない。
