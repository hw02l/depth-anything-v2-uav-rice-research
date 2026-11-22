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
- 追加追補として、固定8画像の6条件深度図・絶対誤差図を作る `make_fixed8_comparison.py` と3件の回帰テストを追加し、commit `05b305f`、`5020da9` をPRブランチへpushした。スクリプトの構文検査は通過したが、追加した3件の実行テストと図表生成は未実施である。
- GPU4の端末が利用できなくなったため、接続中のGPU3の状態を確認した。GPU3はRTX 2080、使用率0%、メモリ使用15MiBであったが、今回の実験ディレクトリ、データ、checkpoint、SingularityイメージはGPU3上の所定領域に存在しなかった。GPU3のローカルディスク空きは30GB（使用率94%）である。追加推論は開始していない。

## 成果物の正本

- 正式集計：`results/20261122T042859Z-35268/analysis_reconciled_d99d8ec/`
- 推論時記録：`results/20261122T042859Z-35268/full_eval/`
- 修正版解析コード：`analyze_results.py`
- 推論コード：`run_b3.py`
- 集計図：`figures/metric_tradeoffs.svg`
- 詳細な方法・結果・限界：`report_ja.md`

## 未完了・制約

- 六条件すべての予測深度マップ・絶対誤差図は作成していない。GPU4のローカルにある既存cacheはC0-B3/E3-B3に限られ、他4条件の固定8画像予測が不足している。生成スクリプトは追加したが、実行環境が利用できず、追加推論も行っていない。新しい回帰テストも未実行であり、現時点では構文検査のみ通過している。したがって視覚的な形状解釈は報告書に含めていない。
- 旧epoch20-B0/B3の符号付きBiasは過去の十分統計に存在しないため、比較表では未提供とした。
- 全指標を含む元の画像別CSVはNFS保管である。公開worktreeには画像単位の対応付きbootstrapに必要な十分統計を含めた。
- この成果の公開範囲はPR #8で確認できる。PR #3〜#7は変更・mergeしていない。

## 重要な結論

C0-B3はC0-P0に対してpooled高差分14px MAEを2.46%改善したが、全有効14px MAEは7.17%悪化した。RMSE差は−0.000123mでbootstrap 95% CIが0をまたぐ。E3-B3はE3-B0より高差分14px MAEを3.16%改善し、六条件中の点推定最良値0.319867mを示したが、全有効14px MAEは8.74%悪化し、RMSEも点推定でわずかに悪化した。単一の総合最良条件はない。
