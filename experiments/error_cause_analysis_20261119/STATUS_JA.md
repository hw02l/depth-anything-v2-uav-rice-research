# 作業状態

更新日：2026年11月19日（GPU3追加診断を反映）

## 完了

- 既存の訂正済みC0/F1/F2 × P0/P1/P2 Validation128画像別十分統計を読み取り、各条件128画像のID・順序を照合した。
- 全9条件のpooled絶対MAE/RMSE/AbsRel/Bias/Pearsonを訂正済みrelease CSVと比較し、最大差 `5e-15` 未満で一致した。
- 全距離・全局所条件の画像別十分統計をcondition summaryと照合した。
- 絶対誤差分解、回帰、局所距離、作物・撮影方向、GT深度・GT差、画像端、タイル境界を再集計した。
- 画像ID単位のpaired bootstrapを10,000回、seed 42で再計算した。pooled推定量と画像単純平均を別々に出力した。
- 画像別局所十分統計を条件・距離・ペア群ごとに集約して出力し、重複のない11,520行を確認した。
- 訂正済みreleaseの対応付きbootstrap CSVと24個の共通比較を `1e-8` 未満で照合した。
- B0–B3 epoch20およびE0–E3保存済み結果を、今回の9条件と分けて統合した。
- PR #6のB0再現監査とcheckpointハッシュ監査を証拠として保存し、実チェックポイント3件のハッシュが訂正後評価記録と一致することを確認した。
- 固定選定済みValidation128画像から8枚を用い、RGB・GTだけのローカル確認図を作成し、原画像・GTファイルSHA256を既存inventoryと照合した。
- GPU3のRTX 2080で固定8画像・5条件（C0-P0、C0-P2、F1-P1、F2-P0、F2-P2）の推論と可視化を実施した。Validation128全体の再推論・新規学習は行っていない。
- GPU3のv3出力で絶対指標40行、符号付き局所差800行、tile重複24行、tile別GT回帰2,520行、RGB/GT integrity 8行、共有pair manifest 80座標セットを検算した。
- GPU3上のCPU集約検証で、固定8画像の14px `|ΔGT|>=0.20m` は条件あたり23,978ペア。共有座標hashの一致と融合再構成最大差0.03125mを確認した。
- GPU3の実checkpoint3件のSHA256を照合し、修正済みPR #6評価記録と一致することを確認した。40件すべての推論メタデータでepoch 5、state_dict 287キー、strict load、missing/unexpected/shape mismatch 0件を確認した。global stepはC0 960、F1/F2 958である。
- 40件の出力でRGB完全一致と有限予測を確認し、すべてNVIDIA RTX 2080で生成されたことを確認した。checkpoint別の内部signatureは `source_evidence/gpu3_checkpoint_load_audit.csv` に記録した。
- GPU3の固定8成果、1画像forward記録、実行ログ、ソース、推論・集約スクリプトを研究室のprivate NFSへ退避した。保存先は151ファイル・31MBである。149項目の外側SHA256SUMSがPASSし、コピー元／先のdiffもPASSした。公開記録ではNFSの実ディレクトリを伏せた。
- source snapshotの5ファイルSHA256は実ファイルと一致した。完全なGit revisionと完全な実行argvが記録されていない理由を `analysis/gpu3_provenance_reconciliation.json` に明記した。
- 固定8画像の丸め済み集約CSV2件と、RGB/GTを含まない要約図（PNG/SVG各2件）を作成した。図表はCSVと生成コードから再現でき、SHA256SUMSに含めた。
- 正本Python 3.12環境で既存解析と固定8集約コードの単体テスト5件がpassした。
- 新規学習とIndependent test319の使用はない。

## 制限・未実施

- source snapshotに `.git` が含まれないため、GPU3推論のGit commit revisionを特定できない。保存済みprovenanceとログにも完全な実行コマンドラインがない。5つのソースSHA256とcheckpoint・ロード・画像入力記録は照合済みであるが、commit単位のsource同一性と完全なargv再現性は未確認である。
- マップ・断面・差分・tile図はprivate NFSへ保全したが、Windows作業コピー上での目視確認は未実施である。図の形状については断定せず、記載した数値と検算結果の範囲に限って解釈する。
- 保存済み十分統計に符号付きの全ペア `ΔGT`・`ΔPred` とその積和がなく、符号付き散布図や差分回帰傾きを厳密に復元できない。
- 符号付き差分図は固定8画像に限る。Validation128全体の符号付き差分回帰・全条件tile重複診断ではない。
- 既存絶対誤差十分統計にGT深度帯ごとの二乗誤差和がないため、絶対RMSEのGT深度別層別値は算出していない。GT深度帯別の局所差MAEは算出した。
- タイル境界・重複群の値は領域構成が異なる記述統計で、同一GTペアの対応付き因果比較ではない。
- bootstrap区間は同じ撮影日からの128画像と選択済みcheckpointに条件付けられ、撮影日間汎化やモデル選択不確実性を表さない。

## GPU

GPGPU3で固定8画像の推論を実施した。最終確認時はGPU利用率0%、VRAM使用量15 MiB/8192 MiBで、compute applicationと自身のPython/Singularityプロセスは残っていなかった。GPU資源は解放済みである。追加図の任意プレビュー時、GPU3側PythonでOpenCVがnumpyを読み込めず、SSHセッションがrelay shellへ戻った。推論処理は既に終了しており、再接続や追加GPU処理は不要である。

## 公開

- PR #7はopen・未mergeであり、baseはPR #6のブランチである。今回のcheckpoint・provenance・NFS監査、固定8集計表・公開図、訂正済み報告書を既存の `research/error-cause-analysis-20261119` ブランチへ追加した。PR #3〜#6は変更していない。
- 配布用 `SHA256SUMS` は61ファイルを収録し、全項目の照合を完了した。公開リリース準備スクリプトのgzip往復検証もpassした。GPU3のprivate raw出力は含まない。
- 集約・解析の単体テストは5件すべてpassした。公開図2組（PNG/SVG）は集約値から再生成し、RGB/GT画像を含まないことを確認した。内部GPU/NFS保存先は公開ファイルから伏せた。
- RGB/GT原画像を含む予測図、checkpoint、prediction cacheはGitHubへアップロードしない。
