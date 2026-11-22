# STATUS

- [x] PR #7最新HEADを確認し、同HEADをbaseにした独立worktree/branchを作成した。
- [x] GPU4へ接続し、GPU機種・利用率・既存compute process・空きストレージを確認した。他ユーザーのprocessには触れていない。
- [x] C0 epoch5とE3 epoch5の保存先・記録SHA256を特定した。
- [x] Validation128、固定pair manifest、訂正済みRGB入力仕様、B3定義と既存評価CSVの所在を確認した。
- [ ] checkpointをGPU4ローカルへ新規コピーし、コピー後SHA256を照合する。
- [ ] 1画像でRGB・strict load・epoch20-B3再現ゲートを実行する。
- [ ] C0-B3/E3-B3をValidation128で推論する。
- [ ] 既存4条件を含む6条件の集計、paired bootstrap、図表を作成する。
- [ ] 実測報告、provenance、SHA256、NFSバックアップ検証を完成する。
- [ ] 新規branchをpushし、PR #7をbaseとする新規PRを作成・検証する。

現時点ではGPU推論・学習とも未実施。数値結論は未確定である。
