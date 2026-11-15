# RGB入力訂正評価・文書整合性ステータス

最終更新：2026-11-15<br>
実験ID：native_resolution_patch_training_20261111<br>
branch：research/native-resolution-patch-training-20261111<br>
PR：[#6](https://github.com/hw02l/depth-anything-v2-uav-rice-research/pull/6)<br>
現状態：**B0再現とRGB回帰は確認済み。9条件の修正後出力は存在するが、C0/F1/F2すべてのepoch5 checkpoint SHA256と評価provenanceが一致せず、正式な研究結果としての確定は保留。**

## 完了した作業

- RGB入力不具合を修正した。推論入口はuint8 [0,255] RGBを要求し、load_pair由来のfloat32 [0,1]は値域を検証して元画素へ戻してから推論へ渡す。
- test_rgb_input_contract.pyとverify_rgb_input_actual.pyのrepository root計算を修正した。前者をCPUで実行し、5件すべてpassした。
- GPGPU4の実画像P0 smoke testは2025/Day047_Corn/images/7.JPGでpassした。3956×5280のRGB pixel完全一致、0〜255値域、有限な元サイズdepth出力を記録した。P1/P2のタイル・blend契約はCPU回帰テストで検査した。
- 既存epoch20 B0をValidation128・P0で再評価し、MAE、RMSE、AbsRel、Pearson、全14px局所MAE、GT差20cm以上の14px局所MAE、振幅比、符号一致率の8指標を許容誤差1e-5以内で再現した。
- 既存C0/F1/F2 epoch5 checkpointを使った9条件の訂正後評価、画像別十分統計、距離別/条件別集計、10,000回seed42 paired bootstrapの出力がある。今回の文書作業では再学習・GPU再評価をしていない。
- analysis/rgb_input_fix_20261114/SHA256SUMSの43ファイルをローカル抽出後に検証し、全件一致した。これは当該結果パッケージの転送・内容整合性を示すが、checkpoint個体性を証明しない。
- 修正前の結果CSV・JSON・図・ログは削除・上書きしていない。これらはinvalid_pre_fixとして扱い、訂正後の研究結果として参照しない。
- PR #6はstate open、base `research/local-huber-20261105`であり、本訂正一式を既存head branch `research/native-resolution-patch-training-20261111`へ追加した。PR #3/#4/#5は変更していない。

## 正式確定を止めている不一致

### epoch5 checkpoint SHA256

| 群 | 指定・旧監査記録 | 訂正後評価出力 | 状態 |
|---|---|---|---|
| C0 | 2fd190727c7a5124c56c10e0a2dd8cb09c6cb62a0f92b34fce97ad394281b35a | 2fd190727c7a5124c56c1e0ea2dd8cb09c6cb62a0f92b34fce97ad394281b35a | 不一致 |
| F1 | ea1f74c5ff17c0b266152588b5d53f8b24611418f125f2fbf4d9cc7447e49f67 | ae1f74c5ff17cb0266152588b5d53fb824611418712152fbf4d9cc7447e49f67 | 不一致 |
| F2 | 8124d72a7008c9f3a93dbcce34142d1593063fba2da6a1a58306d55aaf201538 | 8124d72a7008c9f3a93dbce934142d1593065fba2da61a51838e055aaf201538 | 不一致 |

現行evaluate_all.pyは指定hashと異なるcheckpointなら停止する設計である。また、保存済み9条件JSONには現行sourceが書くはずのrgb_input_contract、evaluation_code_revision、checkpoint_state_load_auditが存在しない。strict_load_auditは初期重みのload監査であり、epoch5ファイルの一致を証明しない。原因を説明できる実体checksumまたは当時の完全な実行receiptはまだ照合できていない。

次の再開時はGPUを使わず、既に接続したGPGPU4 shellで読み取り専用に以下を実行する。

```bash
cd "$HOME/gpu_work/native_resolution_patch_training_20261111" || exit
sha256sum runs/formal/C0/epoch_0005.pt runs/formal/F1/epoch_0005.pt runs/formal/F2/epoch_0005.pt
BACKUP=/home1/eng-std/s0623003/gpu4_research_backup_native_resolution_patch_training_20261115_rgb_input_fix
find "$BACKUP" -maxdepth 3 -type f -name SHA256SUMS -print
```

指定hashと一致しない場合は、ファイル名だけで置き換えた扱いにせず、NFSおよび元run実体を照合する。必要があれば同一性が判明したcheckpointで評価を新規出力先へやり直す。既存結果・checkpointは上書きしない。

## NFS保存状態

現PR本文には、上記専用NFS保存先へ70ファイルをコピーしSHA256照合passと記録されている。一方、PR head上の従来STATUSにはNFS保存と照合が未完了と記録されていた。この2記録は矛盾している。現在のローカル環境からNFS実体やcopy receiptを独立して読めていないため、本STATUSでは「完了」と断定せず、**既報あり・独立再照合待ち**とする。以前の大容量バックアップをこの訂正成果の検証済みバックアップと混同しない。

## 公開・作業状態

- PR #6本文を更新し、12.53%を主要指標と誤読させる旧記載を訂正した。正しい主要指標、pooled/画像平均の区別、RMSEとの対比、全checkpoint SHA256不一致、provenanceおよびNFSの未解決点を記載した。
- 正しい主要GT差20cm以上・14px固定pair pooled値はC0-P2 0.415271m、F2-P2 0.384910m、低減7.31%。画像平均では0.370191mから0.355550m、低減3.95%。全有効pairの副次値は0.116935mから0.102251m、低減12.56%である。
- F2-P2 RMSEは3.084mでC0-P2の8.235mより低いが、C0-P0の0.744mより大幅に高い。
- 本訂正の報告書、STATUS、README、回帰テスト、再集計CSV、図表、provenanceを対象ファイル限定のcommitとして同じbranchへ追加し、PR #6で確認できる状態にした。
- 訂正後結果一式のNFS再照合、C0/F1/F2 epoch5 hashの説明、評価JSON provenanceの由来解決は未完了であり、正式最終判定は保留。
- 追加の学習/GPU評価は開始していない。直近の評価provenanceではGPGPU4が0% utilization/15MiB使用と記録されているが、今ターンのfresh preflightではない。

## 文書・成果物

- 最終報告：[report_ja.md](report_ja.md)
- 訂正後の全条件原データ：[analysis/rgb_input_fix_20261114/](analysis/rgb_input_fix_20261114/)
- 旧監査成果（invalid_pre_fix）：[analysis/final_20261114/](analysis/final_20261114/)
- 今回の再集計版：[analysis/corrected_release_20261115/](analysis/corrected_release_20261115/)
- root README、報告書、PR本文は「12.53%が主要指標」という誤記を訂正し、9条件の数値がcheckpoint/provenance確認まで暫定であることを明示した。
- `analysis/corrected_release_20261115/` の再集計スクリプトは終了コード0で完了し、raw manifest 43項目に失敗0、指定hash不一致3件を報告した。出力release manifest 10項目とSVG XML 3点を検証した。Markdown内のローカルリンク34件はすべて存在した。
- RGB回帰テスト5件は別の既存実行環境でpassしている。一方、このターンで使用したWindows PythonにはOpenCV (`cv2`) がなく、同じテストを再実行できなかった。依存関係の追加はしていない。
- 最終差分は `git diff --cached --check` で問題なし。再集計スクリプトと2つのテストスクリプトはPython構文検査を通過した。
