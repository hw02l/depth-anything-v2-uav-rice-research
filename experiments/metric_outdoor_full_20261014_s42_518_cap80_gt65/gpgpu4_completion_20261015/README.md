# GPGPU4 epoch 20 完了記録

このフォルダーは、Metric Outdoor Smallの20 epoch計画について、epoch 5 checkpointからGPGPU4で継続した結果をまとめたものである。epoch 1〜5の原記録は保持し、このフォルダーに全20 epochの集計を新しい版として保存した。

- `report_ja.md`: 日本語の結果報告
- `epoch_metrics_1_20.csv` / `.json`: 全epochの主要指標と詳細記録
- `group_metrics_1_20.csv`: 作物別・撮影方向別の全epoch集計
- `figures/`: 学習曲線5点
- `run_status_gpgpu4.json`: GPGPU4継続runの完了状態（epoch20/20）
- `provenance.json`: 元ログとチェックポイントのSHA256、完了監査情報

RGB・深度GT、全予測キャッシュ、checkpoint本体はGit管理対象に含めない。epoch 17（validation RMSE最良）およびepoch 20（最終epoch）の完全checkpointは、研究室のユーザー専用NFSバックアップ領域に保存してハッシュ照合済みである。GPGPU4の計算用localdataにはepoch 5〜20のcheckpointを保持している。既存のepoch 5 checkpointとDrive上の原本は変更していない。

この評価はtrain 384組・validation 128組のみを使い、independent test 319組および過去52組は使用していない。GTはカメラ座標系Z-depthであり、植物の垂直高さを実証する評価ではない。
