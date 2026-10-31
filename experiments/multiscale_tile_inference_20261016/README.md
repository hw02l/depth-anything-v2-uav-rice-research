# 多スケール診断・高解像度／タイル推論

20epoch学習済みDepth Anything V2 Metric Outdoor Smallを固定し、Day047 validation 128画像で比較した探索的実験です。追加学習・重み更新は行わず、独立test319組と過去52組は未使用です。 [PR #4](https://github.com/hw02l/depth-anything-v2-uav-rice-research/pull/4)を作成し、GPGPU4 cleanup監査未確認のためdraftにしています。GTは写真測量由来のカメラ座標系Z-depth（m）であり、世界座標の鉛直草丈ではありません。

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

CPUテスト: python -m unittest discover -s tests -p "test_multiscale.py" -v。多尺度評価・図表生成コードは scripts/multiscale_analyze.py と scripts/multiscale_write_report.py、推論コードは scripts/multiscale_experiment.py、scripts/multiscale_diagnose.py、scripts/multiscale_fuse.py にあります。元のrun/checkpoint・固定source・分割manifestは変更していません。画像、GT、checkpoint、全画素予測cache、認証情報はGit対象外です。

## 遠隔後片付けの状態

2026-10-31 09:22:57 JSTの事前確認時点ではGPU compute process・ログイン利用者・Singularity instanceはありませんでした。結果転送後に始めた予測cache整理はSSHのConnection resetにより完了通知前に切断されました。その後の監査接続はDNSパスワード待ちで停止し、認証も監査コマンドも実行していません。NFS二次バックアップ、予測cache保持状態、cleanup receipt、切断後のGPU・コンテナ・SSH終了は未確認です。実験結果の再推論は不要で、後片付けの完了と誤って記録していません。詳細は run_status.json と report_ja.md を参照してください。
