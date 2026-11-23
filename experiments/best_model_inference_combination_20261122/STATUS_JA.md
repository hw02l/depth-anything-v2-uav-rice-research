# 実験状態

## 完了

- PR #7の最新系を基点とする独立ブランチ `research/best-model-inference-combination-20261122` を作成した。作業開始時HEADは `d99d8ecd4efcfa2863f34eaa130b87033088c96a` であり、現在の作業は同じブランチ上の追加commitとして管理している。
- C0 epoch5とE3 epoch5のcheckpointを確認し、記録済みSHA256と照合した。Strict state load、evalモード、RGB uint8 0–255入力をsmoke runで確認した。
- epoch20-B3の固定代表画像1枚について、RMSE、全14px局所MAE、大差分14px局所MAEを既存per-image値と照合した。各差は1e-5m未満で再現ゲートを通過した。
- RTX 1080上でC0-B3とE3-B3を各Validation128画像、新規推論した。新規学習、optimizer更新、Independent test319は実施していない。
- 4つの既存条件を検証済みper-image統計から再利用し、6条件のpooled/image-mean集計、画像単位paired bootstrap（10,000回、seed 42）、Pareto判定を再計算した。
- 初回解析出力 `analysis_final_d99d8ec` は旧analyzerの条件間行混入が判明したため無効扱いとし、削除せず監査記録に残した。修正analyzer `d99d8ec` で再集計し、推論時の十分統計との整合確認を通過した。
- 集計CSV、bootstrap、Pareto表、推論時launch/run-complete記録をローカルへ保全した。NFSバックアップ `/home1/eng-std/s0623003/gpu4_research_backup_best_model_inference_combination_20261122_1420` は39成果ファイル・約1.2MBで、`sha256sum -c SHA256SUMS` の照合が通過した。
- summary/bootstrap/paretoの3 CSVはNFSのSHA256SUMS記録と実ハッシュが一致した。画像別十分統計は768行で、絶対MAE/RMSE・全14px MAE・高差分14px MAEのpooled再集計値がsummaryと1e-12以内で一致した。
- 解析・融合コードの回帰テスト15件をGPU4の既存Singularity環境と依存site-packagesで実行し、すべて通過した。図表SVGと十分統計のローカル検査2件も通過した。ローカルWindows PythonにはNumPyとpytestがないため、数値依存テストはGPU環境内で実行した。
- 公開用worktree内のSHA256SUMSは最終成果物を列挙し、全項目の再照合が通過した。
- GPU4の最終確認ではRTX 1080使用率0%、他ユーザーPID 3894が約2030MiBを使用していた。自分の推論・解析プロセスは終了しており、他ユーザーのプロセスは操作していない。
- ブランチ `research/best-model-inference-combination-20261122` をcommit `5020da95644b36083d543f3750365aa4e17e5899` までpushした。PR #8はPR #7のheadをbaseとするopen・未mergeのPRであり、直近確認時点でmergeableであった。
- 固定8画像の図表生成に先立つ回帰テスト3件を含め、GPU4の既存Singularity環境で22件すべて通過した。単一画像smokeでは6条件の予測RMSEが既存画像別統計と一致し、B3局所指標差は許容誤差内であった。
- 事前固定8画像について深度図8枚・絶対誤差図8枚を生成し、48行の画像別CSV、provenance、SHA256SUMSを保存した。元の出力と実験ディレクトリ内複製のSHA256を照合済みである。
- 生成図は共通深度範囲0–65m、共通絶対誤差範囲0–10mで作成し、表示上限超過数をCSVへ記録した。検証対象は8画像に限り、Validation128全体の再推論は行っていない。
- GPU4ではRTX 1080を使用した。学習・optimizer更新・Independent test319は実施していない。

## 成果物の正本

- 正式集計：`results/20261122T042859Z-35268/analysis_reconciled_d99d8ec/`
- 推論時記録：`results/20261122T042859Z-35268/full_eval/`
- 修正版解析コード：`analyze_results.py`
- 推論コード：`run_b3.py`
- 集計図：`figures/metric_tradeoffs.svg`
- 詳細な方法・結果・限界：`report_ja.md`

## 残る制約

- 固定8画像の目視確認は未完了である。図は作成済みだが、WindowsからGPU4のプレビューサーバーに接続できなかったため、GitHub公開後に全16枚を確認して記録を更新する。

- 固定8画像は視覚確認用の部分標本であり、図からValidation128全体や別撮影日への一般化を結論できない。独立test319は未使用である。
- epoch20-B0/B3の一部符号付きBiasは過去の十分統計に存在せず、比較表では未提供とした。
- 全指標を含む元の画像別CSV完全版は既存の検証済みNFSバックアップに保全した。今回の公開CSVは固定8画像分であり、過去の正式評価表と区別する。
- NFS追補バックアップ、今回の追加commit・PR本文更新は、完了後の実検証結果を追記する。

## 重要な結論

C0-B3はC0-P0に対してpooled高差分14px MAEを2.46%改善したが、全有効14px MAEは7.17%悪化した。RMSE差は−0.000123mでbootstrap 95% CIが0をまたぐ。E3-B3はE3-B0より高差分14px MAEを3.16%改善し、六条件中の点推定最良値0.319867mを示したが、全有効14px MAEは8.74%悪化し、RMSEも点推定でわずかに悪化した。単一の総合最良条件はない。
