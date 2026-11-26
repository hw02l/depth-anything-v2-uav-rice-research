# C0追加15 epoch（09-20研究の継続）

C0の既存epoch5モデルからepoch6〜20を追加し、学習期間延長の影響を検証する実験である。新規Codexプロジェクトではなく、09-20正本リポジトリ内の一実験として管理する。

- 計画：PLAN_JA.md
- 固定条件：config.json
- 現在の進捗：STATUS_JA.md
- 結果と限界：report_ja.md

C0 epoch5 checkpointのSHA256はconfig.json記載値と照合し、modelをstrict loadする。AdamW moment、GradScaler、RNGを復元し、既存5 epoch schedulerは継続せず、追加15 epoch専用PolynomialLRを指定LRで設定する。学習率はValidation結果を見て変更しない。

Validation128のみを使用し、Independent test319は使わない。初期checkpointおよび既存研究出力を上書きしない。学習中は約8回のsuccessful optimizer updateごとに、epoch内サンプル数、更新数、AMP skip数、経過時間、累積global stepを`progress.json`へ原子的に記録する。epoch終了時はValidation中・checkpoint保存済みの状態も記録する。

## checkpoint整理

容量確保のため、既存NFSバックアップのmanifestと実ファイルが一致することを確認した旧C0/F1/F2・E1/E2/E3のepoch checkpoint 28件と、C0/E3 epoch5の完全重複コピー2件をGPUローカルから整理した。C0 epoch5とE3 epoch5の正本は残している。B系列checkpoint、Metric Outdoor初期モデル、NFSバックアップ、既存評価結果は変更していない。GPU上の監査表は`~/gpu_work/c0_extended_training_20261126/audit/checkpoint_cleanup_manifest.tsv`、削除記録は同階層の`checkpoint_cleanup_receipt.txt`にある。

## 検証状態

継続学習・GPU smoke test・最終B3評価スクリプトと単体テストを追加した。単体テスト、厳密な実モデル読み込み、GPU上の一更新smoke testは成功済みである。C0 epoch5 checkpointのSHA256は実行前後で一致した。固定pair manifestは`experiments/multiscale_tile_inference_20261016/pair_manifest.json`を使用する。GPGPU4のユーザー領域はcheckpoint整理後に約45GiB、`/localdata`全体の空きは約107GiBである。共有利用許可を得ているが、正式学習開始直前のfresh preflightで使用率5%以下、空きVRAM5GiB以上、既存compute PID不変、追加checkpointを含む実験領域50GiB以内を再確認する。実行中は他ユーザーのPID集合とVRAM使用量を監視し、許可済みPIDの使用量が基準+256MiBを超える、またはGPU空きVRAMが1GiBを下回れば自分の処理を停止する。別ユーザーのプロセスには触れない。正式学習の初回起動試行は固定manifestのパス誤りで入力検証前に終了しており、GPU学習は始まっていない。パスを修正し、正本manifestのSHA256と128画像数を確認した。progress.jsonは学習中に約8回のsuccessful updateごとに更新される。
