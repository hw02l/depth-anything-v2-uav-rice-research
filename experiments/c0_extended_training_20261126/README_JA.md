# C0追加15 epoch実験

GitHubでの成果公開: [PR #9](https://github.com/hw02l/depth-anything-v2-uav-rice-research/pull/9)（open・未merge）。

C0のepoch5 checkpointからepoch6〜20を追加学習し、学習期間の延長をValidation128で評価した実験である。C0以外は再学習せず、Independent test319も使用していない。

## 主な結果

- C0 epoch20-P0: pooled RMSE 0.723565m、全14px局所MAE 0.058528m、高差分14px局所MAE 0.332422m。
- C0 epoch20-B3: pooled RMSE 0.724078m、全14px局所MAE 0.062684m、高差分14px局所MAE 0.323614m。
- epoch5-P0からepoch20-P0では3指標とも小幅に低下した。画像単位bootstrapでも改善方向の95%区間であった。
- B3は高差分局所MAEを下げる一方、全14px局所MAEを上げた。E3 epoch5-B3は高差分局所MAE 0.319867mで引き続き最良である。

## 成果物

- 実験計画: [PLAN_JA.md](PLAN_JA.md)
- 進捗・保全状況: [STATUS_JA.md](STATUS_JA.md)
- 詳細報告: [report_ja.md](report_ja.md)
- 固定設定: [config.json](config.json)
- pooled / image mean集計: [condition_summary.csv](results/final_20261127/condition_summary.csv)
- 6条件比較図: [condition_summary.svg](results/final_20261127/figures/condition_summary.svg)
- 対応付きbootstrap 24行（4比較×3指標×2推定量）: [paired_bootstrap.csv](results/final_20261127/analysis/paired_bootstrap.csv)
- bootstrap・解析provenance: [analysis_provenance.json](results/final_20261127/analysis/analysis_provenance.json)
- epoch別曲線の元データ: [learning_curve_data.csv](results/final_20261127/learning_curves/learning_curve_data.csv)
- 学習曲線8図: [learning_curves/](results/final_20261127/learning_curves/)
- GPU環境での直接再実行ログ（16+12+1件、exit 0）: [unittest_direct_rerun_20261127.log](results/final_20261127/logs/unittest_direct_rerun_20261127.log)
- 初回テスト実行記録: [unittest_final_20261127.log](results/final_20261127/logs/unittest_final_20261127.log)
- NFSバックアップの検証記録: [SHA256SUMS.verify.final.log](results/final_20261127/SHA256SUMS.verify.final.log)
- ローカル成果物ハッシュ: [SHA256SUMS.local.txt](results/final_20261127/SHA256SUMS.local.txt)
- provenance・ハッシュ照合に関する補足: [provenance_reconciliation_ja.md](results/final_20261127/provenance_reconciliation_ja.md)

NFSバックアップは110ファイルを保持し、うち106ファイルをSHA256SUMSで照合した。NFSからローカルへコピーした統計・provenance・学習曲線・テストログの12ファイルも、NFSのSHA256SUMS.sourceとの照合にすべて成功した。学習曲線8図と対応CSVはローカルにも保存している。

GitHubとこの成果物ディレクトリにはcheckpoint本体、生RGB/GT画像、予測cacheを含めない。epoch別checkpoint、詳細な画像別十分統計、予測出力はGPGPU4の実行領域およびNFSバックアップに保管した。

## 再現

継続学習コードは[continue_c0.py](continue_c0.py)である。epoch20のB3評価と解析には[evaluate_final_b3.py](evaluate_final_b3.py)、[analyze_extended_results.py](analyze_extended_results.py)、[plot_learning_curves.py](plot_learning_curves.py)を用いた。設定、入力、checkpoint、スクリプトの識別情報とハッシュはconfig.json、results/final_20261127/analysis/analysis_provenance.json、NFS側の実行記録で照合した。ただしepoch20-P0評価JSONにはcheckpoint hashがなく、analysis provenanceの同項目もnullであるため、P0評価とcheckpointの直接の暗号学的結び付けは残る制約である。後から原JSONへ値を追記していない。

この結果は単一seed・Validation128での探索的比較であり、独立test319への汎化を示すものではない。学習率はepoch6で追加フェーズ用に再設定しているため、同一の学習率スケジュールを20 epoch連続適用した結果ではない。
