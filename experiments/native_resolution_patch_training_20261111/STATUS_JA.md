# 訂正評価・checkpoint監査の最終状態

最終更新：2026-11-15<br>
branch：research/native-resolution-patch-training-20261111<br>
PR：[#6](https://github.com/hw02l/depth-anything-v2-uav-rice-research/pull/6)<br>
状態：**checkpoint同一性、B0基準再現、修正版コードによる全9条件再推論、paired bootstrap、旧訂正結果との照合、NFS保存とSHA256検証を完了した。**

## 確認済み

- C0/F1/F2 epoch5 checkpoint実体は旧NFSコピー、epoch5記録、内部signatureと照合済み。実ファイルのSHA256は旧評価JSONの記録とも一致する。依頼時に示された3つのSHA256は文字列の転記誤りであり、実物や学習結果が変わった証拠はない。
- checkpointはepoch 5、C0 global step 960、F1/F2 global step 958、各287-key state_dictである。key/shape/dtype/有限性とstrict load後の値一致を確認した。
- 既存epoch20 B0のValidation128/P0再現は8指標すべて許容誤差1e-5以内でPASSした。
- 修正済みRGB入力契約を使うfresh runは9/9条件完了。各評価JSONにcheckpoint SHA256、global step、RGB契約、evaluation code revision、checkpoint state load auditが記録された。
- fresh runと過去の訂正済み出力を比較し、condition summary CSVと10,000回・seed42 bootstrap CSVがbyte-identicalであることを確認した。画像別絶対十分統計は全9条件で一致（runtime列を除く）、局所統計も全9条件、tile境界統計もP1/P2の6条件で一致した。
- NFSの既存最終バックアップは217項目、従来の11月15日バックアップは68項目と43項目の各manifestが全件PASSである。fresh監査スナップショットは `/home1/eng-std/s0623003/gpu4_research_backup_native_resolution_patch_training_20261115_checkpoint_audit_reproduced/` に94項目・約438 MiBで保存し、コピー後に全SHA256を検証した。
- GPU評価終了後に自身の推論プロセスが残っていないことを確認した。最終確認時はGPU使用率0%、compute processなし、メモリ使用約705 MiBであった。
- 再学習はしていない。Independent test319は使用していない。旧評価JSONへprovenanceを後付けしていない。checkpoint、原画像、GT、prediction cacheも変更していない。

## 確定した主比較

- 主指標：GT差20cm以上・元画像14px固定pair MAE、F2-P2 vs C0-P2。
- Pooled：0.415271mから0.384910mへ7.31%低下。差 -0.030361m、画像単位paired bootstrap 95% CI [-0.037702, -0.022933]m、126画像。
- 画像平均：0.370191mから0.355550mへ3.95%低下し、5%基準の点推定に届かない。
- F2-P2 RMSEは3.084m。C0-P2の8.235mより低いが、C0-P0の0.744mより約4.14倍大きい。
- 学習解像度と推論解像度の交互作用が大きく、元解像度クロップ学習が一様に有効とは結論しない。単一seed、同一撮影日Validation128、epoch5固定選択、探索的サブグループ解析の限界は残る。

## 公開成果物と再現情報

- 報告書：[report_ja.md](report_ja.md)
- 集計・図表・provenance：[analysis/corrected_release_20261115/](analysis/corrected_release_20261115/)
- fresh再推論の詳細な機械可読監査：[fresh_reproduction_audit.json](analysis/corrected_release_20261115/fresh_reproduction_audit.json)
- fresh runのGPU出力、ログ、コードスナップショット、bootstrap一式は上記NFSバックアップに保存した。
- 修正前の無効評価は `analysis/final_20261114/` に監査履歴として残し、正しい結果とは区別する。

## 公開状態

最終報告・監査成果物は既存PR #6のブランチ `research/native-resolution-patch-training-20261111` に反映済みである。PR #3/#4/#5は変更していない。公開対象からcheckpoint、RGB/GT原画像、prediction cache、認証情報を除外した。
