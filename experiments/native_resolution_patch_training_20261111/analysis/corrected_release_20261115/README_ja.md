# 訂正後評価の再集計（2026-11-15）

**状態：暫定。** B0基準再現はPASSし、RGB不具合を直した全9条件の集計とbootstrap出力がある。ただしC0/F1/F2すべての評価checkpoint SHA256が指定値と一致せず、評価JSON provenanceにも不一致がある。以下の9条件図表を指定checkpointの正式結果として扱わない。

## 主要値

- 事前登録主比較F2-P2対C0-P2、GT差20cm以上・元画像14px固定pair MAE：pooled 0.415271 → 0.384910m、7.31%低減、95% CI差 [-0.037702, -0.022933]m、126画像。
- 画像平均推定：0.370191 → 0.355550m、3.95%低減。5%基準の点推定に届かない。
- 全有効14px pair MAE（副次）：0.116935 → 0.102251m、12.56%低減。これはGT差20cm以上の主指標とは異なる。
- F2-P2 RMSE 3.084m、C0-P2 RMSE 8.235m。F2-P2は同じP2方式のC0より低いが、C0-P0 RMSE 0.744mより約4.14倍大きい。

## ファイル

- metrics_9_conditions.csv: 9条件の全体絶対/局所指標。
- within_model_inference_method.csv: 学習群内のP0/P1/P2 RMSEと全14px局所MAEの差（記述統計）。
- key_paired_comparisons.csv: 主比較、全有効pair副次比較、F2-P2 RMSE paired bootstrap。
- checkpoint_hash_audit.csv: 指定された3 checkpoint hashと評価出力記録の照合。全3群不一致。
- figure1_absolute_rmse.svg: 9条件の絶対RMSE。
- figure2_primary_fixed_pairs.svg: 推論法ごとのGT差20cm以上局所MAE。
- figure3_primary_vs_all_pairs.svg: P2主指標と全有効pair副次指標。
- provenance_audit.md: 評価由来、manifest再照合、NFS状態の留保。
- SHA256SUMS: 本ディレクトリの成果物manifest。
- prepare_release.py: raw訂正結果からCSV、SVG、provenance、hash manifestを再生成するスクリプト。

元の9条件per-image出力、条件別集計、完全bootstrapは隣接する ../rgb_input_fix_20261114/ にある。修正前の無効結果は ../final_20261114/ に監査用として残す。

再生成コマンド（repository rootから）：

~~~powershell
.\\.venv\\Scripts\\python.exe experiments\\native_resolution_patch_training_20261111\\analysis\\corrected_release_20261115\\prepare_release.py
~~~

この再集計は保存済みCSV/JSONを読む処理のみであり、学習・GPU推論を行わない。
