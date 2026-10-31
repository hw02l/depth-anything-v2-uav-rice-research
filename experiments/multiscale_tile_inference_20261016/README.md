# 多スケール診断・高解像度／タイル推論

20epoch学習済みDepth Anything V2 Metric Outdoor Smallを固定し、Day047 validation 128画像で比較した探索的実験です。追加学習・重み更新は行わず、独立test319組と過去52組は未使用です。[PR #4](https://github.com/hw02l/depth-anything-v2-uav-rice-research/pull/4)にコード・結果・報告をまとめました。GTは写真測量由来のカメラ座標系Z-depth（m）であり、世界座標の鉛直草丈ではありません。

## 主な実測結果（epoch 20）

|方式|絶対RMSE (m)|14px全ペア局所MAE (m)|GT差20cm以上局所MAE (m)|振幅比|符号一致率|
|---|---:|---:|---:|---:|---:|
|B0 全体518px|0.7942|0.05964|0.33618|0.3327|82.48%|
|B1 全体1036px|2.4203|0.06842|0.34448|0.3862|81.91%|
|B2 2000pxタイル・公称25%重複|3.8536|0.06983|0.34082|0.4663|83.19%|
|B3 全体低周波＋タイル高周波|0.7940|0.06406|0.32801|0.4014|84.63%|

単純な高解像度化・タイル化は14px全ペアMAEと絶対RMSEを悪化させました。GTを使わないB3は大差分MAEを下げ、絶対RMSEを保ちましたが、14px全ペアMAEと画像端誤差は悪化し、振幅比も1未満です。解像度だけで局所形状の過小推定を解消する仮説は支持されません。B3は本validation内で選んだ探索的方式で、独立test上の性能を意味しません。

## 成果物

- report_ja.md: 方法・結果・全図・群別・限界を含む日本語報告。
- config.json, manifest.json, pair_manifest.json, pilot_selection.csv: 設定、入力ハッシュ、共通評価ペア、予測前に固定したpilot選択。
- baseline_reproduction.json: epoch17/18/20の基準再現。
- multiscale_metrics.csv, method_comparison.csv, per_image_metrics.csv, group_metrics.csv, runtime_metrics.csv: 指標と画像／群／実行時間。
- paired_differences.csv, image_variability.csv, per_image_multiscale.csv.gz: 対応付き画像差、ばらつき、尺度別画像結果。
- provenance.json, run_status.json, results_manifest.json: 来歴、実行・未完了cleanup状態、公開ファイルSHA256。
- figures/: 同じ尺度で作成した集計図と8枚の代表画像・境界拡大図。図はCC BY 4.0のデータから作成した加工図で、RGB/GT原本は再配布していません。

## データセットと図の出典

データ: [UAV3DCrop RGB](https://huggingface.co/datasets/Link-Dev/UAV3DCrop) および [UAV3DCrop Depth](https://huggingface.co/datasets/Link-Dev/UAV3DCrop_depth)。各配布元のdataset cardはCC BY 4.0と記載しており、[ライセンス本文](https://creativecommons.org/licenses/by/4.0/)に従って出典を表示しています。図はvalidation画像・深度を切り出し、色付け、予測・誤差比較した加工物です。RGB/GT原本やcheckpointは本リポジトリに再配布していません。

引用: Zhou et al., “UAV3DCrop: Benchmarking 3D Reconstruction in Repeated Multi-Angle UAV Crop Surveys,” arXiv:2608.06404 (2026), https://arxiv.org/abs/2608.06404.

## 再現と保護

CPUテスト（全28件）: `& '.\.venv\Scripts\python.exe' -m unittest discover -s tests -v`。環境・所要時間・結果は `logs/cpu_tests_20261031.txt` に記録しました。多尺度評価・図表生成コードは scripts/multiscale_analyze.py と scripts/multiscale_write_report.py、推論コードは scripts/multiscale_experiment.py、scripts/multiscale_diagnose.py、scripts/multiscale_fuse.py にあります。元のrun/checkpoint・固定source・分割manifestは変更していません。画像、GT、checkpoint、全画素予測cache、認証情報はGit対象外です。

## 遠隔後片付けの状態

2026-10-31の読み取り専用監査（GPGPU4、job `009_recovery_inventory`、監査スクリプトSHA256=`78a078bdd5ff44e59f6072e50c71f258cb86096f7bb03a99057f3685077a4fe4`）は `status=pass` で完了しました。中断していたcleanup PID 26516は存在せず、実験自身のGPU・推論プロセスとSingularity instanceもありません。NFS二次バックアップのcleanup receipt・結果アーカイブ・実行ソースのSHA256はすべてmanifestと一致しました。予測cacheは保持対象72件がすべて正常、整理対象1272件はすべて消去済みで、不整合はありません。整理対象の一時ファイル106,276,678,656 bytesが削除され、学習checkpointと重みは変更されていません。PowerShellはidle timeout後にローカルpromptへ戻り、SSH接続の終了を確認しました。終了時にstdin待機中のdaemon threadによるPythonのfatal interpreter-shutdown errorが出ましたが、`session_closed`後に発生し、SSHは切断され端末はPowerShell promptへ戻りました。監査記録は `logs/cleanup_state_audit_20261031.json`、詳細は `report_ja.md` を参照してください。
