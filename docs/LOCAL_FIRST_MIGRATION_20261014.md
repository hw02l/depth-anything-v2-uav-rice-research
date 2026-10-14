# ローカル統合の記録（2026-10-14）

移行先は9月20日の既存UAV研究Gitリポジトリである。9月22日のDA-2Kプロジェクトは無変更。既存main履歴を保持し、第2回の既存draft PRの3コミットを継承する専用branch `research/local-first-20261014` で作業した。既存の未追跡スライド2件と準備コードを削除していない。

## 取り込み

- 第1回：実行版source 25ファイル、CPU監査・GT統計・ログ、run metadata、checkpoint metadata、Notebook。
- 第2回：最終ZIPの84ファイル。SHA256 `92d3e194a860b0a075036b00c92fcb3318d9e8dca633c37bf2632e83dcf9d8f1` とCRCが一致。source_v4 28ファイル、比較図29枚、詳細CSV/JSONと報告書を保持。
- 第3回：source_full_v1 28ファイル、epoch1〜5 CSV/JSON、PNG5枚、実行ログ、環境・再開署名・同期receipt、checkpoint0〜5の軽量metadata。
- 最新論文：日本語フォント修正版PDF、画像版互換PDF、TeX各章、ベクトル図、BibTeX、根拠一覧、build log。図表と数値を再生成していない。
- 122ファイルをDriveからrawで取得し、サイズとSHA256を記録。コピーした289ファイルをSHA256で照合。取り込み一覧はimport_manifest.json。

## 異なる版の扱い

既存の未追跡uav_metric_finetuningと第1回実行版で異なるファイルは0件。既存コードを上書きせず、実行版をexperiments内へ別保存した。詳細はversion_comparisons.json。

第3回の正式run署名10項目とtrain/validation manifestは一致。初期source_buildのmodeling.py hashは後続修正前である。ログとrun_metadataに整合する修正後ソースを保存し、過去markerを書き換えていない。

論文添付epoch CSVはDrive原本より3 bytes大きく、UTF-8 BOMが1個余分に付いていた（原本1個・添付2個）。先頭BOMを全て除去したテキスト一致はTrue。両方を元の別パスに保持し、論文フォルダーへ正規CSVをsource_epoch_metrics_canonical.csvとして別名追加した。元ファイルを上書きしていない。

## 大容量と公開

checkpointのmodel/trainer合計は2,377,317,172 bytesで、Driveに保持。データ7,850,330,776 bytes、model cache、全予測cacheも移行コピーしない。原本の削除・権限変更・新しいGPU実行はない。今回大容量本体のSHAを再計算したとは主張せず、complete.jsonの記録値と再確認した存在・サイズを区別している。

画像・深度の公式CC BY 4.0表示を確認し、比較図に出典・改変表示を付けた。学習済み重みを公開していない。Notebookの出力と個人metadataを除去し、実行履歴原本はGit対象外workに保持した。

固定ソースのGit改行変換を無効化した。研究の正本local→GitHub/bundle→Colab、結果Colab→Drive→localという方向をWORKFLOW_JA.mdに定めた。
