# C0追加15 epoch（09-20研究の継続）

C0の既存epoch5モデルからepoch6〜20を追加し、学習期間延長の影響を検証する実験である。新規Codexプロジェクトではなく、09-20正本リポジトリ内の一実験として管理する。

- 計画：PLAN_JA.md
- 固定条件：config.json
- 現在の進捗：STATUS_JA.md
- 結果と限界：report_ja.md

C0 epoch5 checkpointのSHA256はconfig.json記載値と照合し、modelをstrict loadする。AdamW moment、GradScaler、RNGを復元し、既存5 epoch schedulerは継続せず、追加15 epoch専用PolynomialLRを指定LRで設定する。学習率はValidation結果を見て変更しない。

Validation128のみを使用し、Independent test319は使わない。初期checkpointおよび既存研究出力を上書きしない。

## checkpoint整理

容量確保のため、既存NFSバックアップのmanifestと実ファイルが一致することを確認した旧C0/F1/F2・E1/E2/E3のepoch checkpoint 28件と、C0/E3 epoch5の完全重複コピー2件をGPUローカルから整理した。C0 epoch5とE3 epoch5の正本は残している。B系列checkpoint、Metric Outdoor初期モデル、NFSバックアップ、既存評価結果は変更していない。GPU上の監査表は`~/gpu_work/c0_extended_training_20261126/audit/checkpoint_cleanup_manifest.tsv`、削除記録は同階層の`checkpoint_cleanup_receipt.txt`にある。

## 検証状態

継続学習・GPU smoke test・最終B3評価スクリプトと単体テストを追加した。Python構文検査は通過済みで、PyTorch依存テスト、実モデルstrict load、GPU smoke testはこれから実施する。GPGPU4のユーザー領域はcheckpoint整理後に45GiB、`/localdata`全体の空きは107GiBである。GPU利用者から共有利用の許可を得たため、開始直前のpreflightで空きVRAM5GiB以上、使用率5%以下、既存compute PID不変、実験領域50GiB未満の見込みを確認し、まず1バッチのsmoke testを実行する。別ユーザーのプロセスには触れない。
