# 既存最良モデルとB3融合推論の組合せ検証

C0 epoch5、E3 epoch5の同一checkpoint内で全体推論と2000pxタイル推論を融合するB3方式を、Validation128で比較した推論専用実験である。学習・Independent test319は行っていない。

## 主な結果

- C0-B3: RMSE 0.743945m、全14px局所MAE 0.063323m、高差分14px局所MAE 0.326687m。
- E3-B3: RMSE 0.768533m、全14px局所MAE 0.063721m、高差分14px局所MAE 0.319867m。
- C0-B3はC0-P0に比べ、高差分14px局所MAEが2.46%低下した一方、全14px局所MAEは7.17%上昇した。RMSE差は統計的に明確でない。
- E3-B3はE3-B0より高差分14px局所MAEが3.16%低下したが、全14px局所MAEは8.74%上昇し、RMSEはわずかに高くなった。
- 6条件で指標の優劣が分かれ、単一の総合最良条件はない。検証を繰り返し用いたValidation128上の探索結果であり、独立テストへの汎化は示さない。
- 初回に生成した解析出力は条件間の局所統計混在が判明したため無効とし、監査記録を残した。修正analyzerで十分統計を再照合した `analysis_reconciled_d99d8ec` だけを正本とする。

詳細な方法、統計比較、再現条件と限界は [report_ja.md](report_ja.md) を参照。

GitHubの積み重ねPRは [PR #8](https://github.com/hw02l/depth-anything-v2-uav-rice-research/pull/8) である。baseはPR #7のブランチである。

## 成果物

- `results/20261122T042859Z-35268/analysis_reconciled_d99d8ec/condition_summary.csv`: 6条件のpooled・画像平均指標
- `results/20261122T042859Z-35268/analysis_reconciled_d99d8ec/per_image_sufficient_statistics.csv`: 6条件×128画像のbootstrap用十分統計（768行）。pooled再集計との一致を検証済み。
- `results/20261122T042859Z-35268/analysis_reconciled_d99d8ec/paired_bootstrap.csv`: 10,000回のpaired bootstrap
- `results/20261122T042859Z-35268/analysis_reconciled_d99d8ec/pareto.csv`: 3指標のPareto判定
- 全指標を含む元の画像別CSVと十分統計の完全版は検証済みNFSバックアップに保全してある。
- `results/20261122T042859Z-35268/full_eval/`: 新規推論のlaunchおよび完了記録
- `figures/metric_tradeoffs.svg`: 6条件の集約指標比較図
- `provenance.json` と `SHA256SUMS`: 実行条件とファイルハッシュ
- `STATUS_JA.md`: 作業状態と未完了項目

## 固定8画像の視覚比較

事前固定したimage index 7、20、34、48、71、89、97、112の各画像について、6条件の予測深度図8枚と絶対誤差図8枚を作成した。深度図は共通0–65m、絶対誤差図は共通0–10mで表示し、10m超は図示時のみクリップした。超過件数・割合を含む48行の画像別CSV、生成provenance、SHA256SUMSも同じディレクトリに置いた。

図ファイルは [depth_maps_007.png](figures/fixed8_visualizations_20261123/depth_maps_007.png)、[depth_maps_020.png](figures/fixed8_visualizations_20261123/depth_maps_020.png)、[depth_maps_034.png](figures/fixed8_visualizations_20261123/depth_maps_034.png)、[depth_maps_048.png](figures/fixed8_visualizations_20261123/depth_maps_048.png)、[depth_maps_071.png](figures/fixed8_visualizations_20261123/depth_maps_071.png)、[depth_maps_089.png](figures/fixed8_visualizations_20261123/depth_maps_089.png)、[depth_maps_097.png](figures/fixed8_visualizations_20261123/depth_maps_097.png)、[depth_maps_112.png](figures/fixed8_visualizations_20261123/depth_maps_112.png)と、[absolute_error_007.png](figures/fixed8_visualizations_20261123/absolute_error_007.png)、[absolute_error_020.png](figures/fixed8_visualizations_20261123/absolute_error_020.png)、[absolute_error_034.png](figures/fixed8_visualizations_20261123/absolute_error_034.png)、[absolute_error_048.png](figures/fixed8_visualizations_20261123/absolute_error_048.png)、[absolute_error_071.png](figures/fixed8_visualizations_20261123/absolute_error_071.png)、[absolute_error_089.png](figures/fixed8_visualizations_20261123/absolute_error_089.png)、[absolute_error_097.png](figures/fixed8_visualizations_20261123/absolute_error_097.png)、[absolute_error_112.png](figures/fixed8_visualizations_20261123/absolute_error_112.png)である。

複合図にRGB・GT表示を含むことについて、ユーザーから明示的に掲載許可を得た。生のRGB/GT、checkpoint、prediction cacheは掲載しない。固定8画像は記述的な視覚診断であり、Validation128の数値評価を置き換えず、全標本への因果的結論を支えない。詳細は [report_ja.md](report_ja.md) と [STATUS_JA.md](STATUS_JA.md) を参照。
