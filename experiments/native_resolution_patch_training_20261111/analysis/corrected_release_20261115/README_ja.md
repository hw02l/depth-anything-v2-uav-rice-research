# 訂正後評価の最終集計（2026-11-15）

**状態：fresh再推論・十分統計照合済み。** 修正版コードでValidation128の9条件を再評価し、既存epoch5 checkpoint、B0基準、RGB契約、state_dict strict loadを確認した。fresh結果は従来の訂正後出力と科学指標・bootstrapまで一致した。再学習は行っていない。

## 主要結果

- 事前登録主比較はF2-P2対C0-P2のGT差20cm以上・元画像14px固定pair MAEである。pooled推定は0.415271mから0.384910mへ7.31%低下し、差の画像単位paired bootstrap 95% CIは[-0.037702, -0.022933]mであった（126画像、10,000回、seed 42）。
- 画像平均推定は0.370191mから0.355550mへ3.95%低下し、5%基準の点推定には届かない。pooledと画像平均の違いを保ったまま報告する。
- 全有効14px pair MAEは副次指標で、pooled 0.116935mから0.102251mへ12.56%低下した。これはGT差20cm以上の主要指標ではない。
- F2-P2絶対RMSEは3.084m、C0-P2は8.235mである。F2-P2はC0-P2より小さいが、C0-P0の0.744mより約4.14倍大きい。局所指標の差を全体深度精度の改善と解釈しない。
- 学習crop解像度と推論タイル解像度の効果は交互作用を示す。F2はP0で主要局所MAEが悪化し、P1の差は区間が0を含み、P2では低下した。解像度単独の一様な効果とは結論しない。

## 公開ファイル

- `metrics_9_conditions.csv`: 9条件の絶対指標、全14px局所MAE、tile数、checkpoint SHA256。再実行間で変動するruntime列は含めない。
- `key_paired_comparisons.csv`: 主要比較、全有効pair副次比較、RMSE、距離別、作物別・視点別paired bootstrap。
- `within_model_inference_method.csv`: 各学習群内のP0/P1/P2絶対RMSEと全14px局所MAEの記述差。
- `checkpoint_hash_audit.csv`: 依頼時に転記された誤記、実checkpoint、旧評価JSON記録の照合。
- `figure1_absolute_rmse.svg`, `figure2_primary_fixed_pairs.svg`, `figure3_primary_vs_all_pairs.svg`: 確認済み9条件結果の図。
- `fresh_reproduction_audit.json`: fresh runのrevision、コードSHA256、checkpoint SHA256、B0 gate、旧出力との比較、NFS保存記録。
- `provenance_audit.md`: 実行由来と履歴provenanceの留保。
- `prepare_release.py`: 集計CSV・図表・provenance・SHA256SUMSの再生成スクリプト。過去の評価JSONにprovenanceを後付けしない。
- `SHA256SUMS`: 公開用成果物のSHA256。

修正前に誤ったRGB値域で実施した結果は `analysis/final_20261114/` に監査履歴として残し、正式値として扱わない。旧訂正後JSONにprovenanceがない理由として、利用可能な旧 `evaluate_patch.py` の完了JSON cache経路が、既存出力をメタデータ更新なしで返すことを確認した。ただし当時の完全な実行コマンドとdirty source revisionが残っていないため、その経路が実際に当該JSONへ使われたとは断定できない。欠落項目を過去に遡って補わず、修正コードによるfresh runを独立した根拠として記録した。

fresh raw runはNFS `/home1/eng-std/s0623003/gpu4_research_backup_native_resolution_patch_training_20261115_checkpoint_audit_reproduced/` に退避し、94項目のmanifestをコピー後に照合した。学習済みcheckpoint、RGB/GT原画像、prediction cacheはこの公開releaseに含めない。

再生成コマンド（repository rootから）：

~~~powershell
.\.venv\Scripts\python.exe experiments\native_resolution_patch_training_20261111\analysis\corrected_release_20261115\prepare_release.py
~~~

スクリプトは保存済み訂正CSVと `fresh_reproduction_audit.json` を検証して集計物を作る。学習・GPU推論は行わない。
