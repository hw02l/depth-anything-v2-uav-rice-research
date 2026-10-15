# UAV農作物画像のDepth Anything V2適応研究

Depth Anything V2 SmallをUAV農作物画像へ適応し、絶対深度と局所深度構造を評価する研究である。将来の稲の高さ推定に向けた基礎検討であり、現在のGTは**カメラ座標系のZ-depth**である。垂直草丈そのものではない。

**現在の確定状態：第3回runは20/20 epochまで完了。** epoch 1〜5はColab Tesla T4、epoch 6〜20は研究室GPGPU4（GTX 1080）で同一runから継続した。validation RMSE最良はepoch 17（0.77696 m）、最終epoch 20は0.79424 m。独立test 319組と過去52組は未使用。稲の垂直高さ推定精度は未検証である。

通常の編集・分析・図表・論文・Git作業はWindowsのこのローカルプロジェクトを正本とする。Driveは大容量データ、checkpoint、Colab受け渡しとバックアップに使い、GPU実験は明示的に許可されたColabまたは研究室GPGPU4で実施する。[運用ルール](docs/WORKFLOW_JA.md)を参照。

## モデルと主な結果

共通validation 128画像（Day047）、GT有効上限65m。局所14px MAEは元解像度の全有効ペアによる集計である。

|モデル|初期化・学習|出力上限|MAE m|RMSE m|AbsRel|Pearson|局所14px MAE m|
|---|---|---:|---:|---:|---:|---:|---:|
|A|Relative Small encoder＋新Metric head、1 epoch|65m|2.4897|3.9063|0.12120|0.8712|0.10545|
|B|未追加学習のMetric Outdoor Small|80m|13.2730|14.5340|0.65896|0.7848|0.06561|
|C|Metric Outdoorから1 epoch追加学習|80m|1.1324|1.7490|0.05208|0.9773|0.06800|
|D-5|元のMetric Outdoorから独立した20 epoch計画、epoch 5|80m|0.7373|1.0094|0.03694|0.9919|0.06148|
|D-17|同一run、validation RMSE最良checkpoint|80m|0.5529|0.7770|0.02789|0.9945|0.05980|
|D-20|同一run、最終epoch|80m|0.5735|0.7942|0.02919|0.9944|0.05964|

DはCのcheckpointからの継続ではない。CとDではscheduler総期間等も異なるため、差をepoch数だけの効果とは断定しない。

|D epoch|validation RMSE m|AbsRel|局所14px MAE m（全ペア）|GT差≥20cm MAE m（固定標本）|振幅比（固定標本）|符号一致率（固定標本）|
|---:|---:|---:|---:|---:|---:|---:|
|5|1.0094|0.03694|0.06148|0.34090|0.3304|81.02%|
|10|1.0219|0.03392|0.06030|0.34050|0.3187|81.78%|
|15|0.8579|0.03070|0.06000|0.33772|0.3366|82.44%|
|17（RMSE最良）|0.7770|0.02789|0.05980|0.33642|0.3327|82.37%|
|18（局所14px最良）|0.8007|0.02935|0.05957|0.33703|0.3306|82.42%|
|20（最終）|0.7942|0.02919|0.05964|0.33618|0.3327|82.48%|

GT差≥20cmの振幅比はepoch 5でも約0.33で、大きいZ-depth差の過小推定が残る。ゼロ差分基準の局所MAEは0.06421mであり、総局所MAEだけで形状を評価できない。Aの局所誤差増分は画像端128px未満へ強く集中し、512px以上の内部では相対順位が変わった。詳細指標は固定標本と全ペアを区別して読む。

## データと学習条件

UAV3DCropのCorn/Oat/Wheat/Soybeanを用いる。train 384組はDay019/028/040、validation 128組はDay047。512組・1024ファイル・7,850,330,776 bytes。分割は撮影日単位で固定し、[manifest](configs/uav3dcrop_verified/)を保持する。`independent_test.csv`は選択計画の記録のみで、実データ取得・評価を意味しない。

入力518、batch 1、gradient accumulation 4、AMP、seed 42、AdamW（weight decay 0.01）、encoder LR 5e-6／neck-head LR 5e-5、SiLog。DはPolynomialLR(power=0.9)、総1920予定optimizer stepsを維持する。AMP skip時にschedulerを進めない。出力係数80mとGT mask上限65mは別設定である。

epoch 1〜5は無料Colab Tesla T4、epoch 6〜20は研究室GPGPU4のGTX 1080で実行した。後半では移行用の厳格な互換adapterを用い、元の`source_full_v1/`とepoch 5原本は変更していない。GPU・ソフトウェア環境の差はepoch間比較の留意点である。再開署名検証は迂回していない。

## 構成と再現

```text
src/da2_experiments/        従来の評価・幾何処理
src/uav_metric_research/    今後のローカル作業用の検証・受け渡し補助
uav_metric_finetuning/      既存の学習準備コード（移行前の内容を保存）
experiments/               第1〜3回の固定コード・結果・ログ・run metadata
configs/uav3dcrop_verified/ 固定分割manifest
notebooks/                 出力・個人用metadataを除いた実行Notebook
reports/                   各回日本語報告書と既存研究資料
papers/                    最新30ページ論文・LaTeX・図・参考文献
figures/                   主要図への索引・出典
checksums/                 実験時ソースのSHA256一覧
scripts/                   従来実行コードとローカル受け渡しCLI
docs/                      運用・移行記録・大容量資産の所在
data/, work/               ローカルのみ。Git対象外
```

最初にCPUで以下を実行する（モデル・データの取得やGPU実行なし）。

```powershell
.\.venv\Scripts\python.exe scripts/research_workspace.py verify
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_research_workspace.py" -v
.\.venv\Scripts\python.exe -m unittest discover -s uav_metric_finetuning/tests -v
.\.venv\Scripts\python.exe scripts/research_workspace.py bundle --output work/colab_bundles/trial3_resume_code.zip
```

第3回runはepoch 20で完了しており、同じrunを再開しない。元の`source_full_v1/`、Colab時代のepoch 5 `run_status.json`、checkpointは監査用原本として無改変で保持し、GPGPU4継続後の状態は[別版のrun_status_gpgpu4.json](experiments/metric_outdoor_full_20261014_s42_518_cap80_gt65/gpgpu4_completion_20261015/run_status_gpgpu4.json)と[完了報告](experiments/metric_outdoor_full_20261014_s42_518_cap80_gt65/gpgpu4_completion_20261015/report_ja.md)に記録した。追加の方法比較は新しいrun IDで行う。

## 報告書・図

- [第1回報告](reports/trial1_execution_report_ja.md)、[第2回原報告](reports/trial2_report_ja.md)、[第2回の後続ローカル検証版](reports/trial2_report_verified_20261014.md)、[第3回報告](reports/trial3_report_ja.md)
- [最新論文PDF](papers/interim_finetuning_report_20261014/paper_ja.pdf)、[LaTeX](papers/interim_finetuning_report_20261014/paper_ja.tex)、[根拠一覧](papers/interim_finetuning_report_20261014/evidence_manifest.csv)、[後から確認できた環境情報の補記](papers/interim_finetuning_report_20261014/MIGRATION_ADDENDUM_JA.md)
- [第2回の比較図29枚](experiments/metric_shape_20261013T123006472484Z/analysis/figures/)、[第3回epoch 1〜5の原CSV/JSONと曲線](experiments/metric_outdoor_full_20261014_s42_518_cap80_gt65/results/)、[GPGPU4継続のepoch 1〜20報告・CSV/JSON・5曲線](experiments/metric_outdoor_full_20261014_s42_518_cap80_gt65/gpgpu4_completion_20261015/)、[図の索引](figures/README.md)
- [従来研究README](docs/archive/README_before_local_migration_20261014.md)には既存のKITTI、ETH3D、稲画像等の説明を保持している。

![第3回20epoch validation RMSE](experiments/metric_outdoor_full_20261014_s42_518_cap80_gt65/gpgpu4_completion_20261015/figures/curve_validation_rmse.svg)

![第3回20epoch局所深度差MAE](experiments/metric_outdoor_full_20261014_s42_518_cap80_gt65/gpgpu4_completion_20261015/figures/curve_local14_mae.svg)

![GT差20cm以上の振幅比](experiments/metric_outdoor_full_20261014_s42_518_cap80_gt65/gpgpu4_completion_20261015/figures/curve_large_delta_amplitude.svg)

## 保存・公開方針と課題

RGB/GT、学習重み、optimizer state、全予測cacheはGitへ含めない。Drive原本は残し、[REMOTE_ASSETS.json](docs/REMOTE_ASSETS.json)にID・サイズ・記録済みSHA256を保存する。`checkpoint_metadata/`は軽量な記録だけで、ローカルの完全checkpointではない。画像版互換PDFと元実行Notebookはローカルには保持するがGit対象外。公開Notebookは出力・個人metadataを削除し、固定コードは`.gitattributes`で改行変換を無効化した。

20 epoch計画は完了した。今後は独立testをモデルと評価手順を固定した後に一度だけ使用し、複数seed、局所差分損失、高解像度tile、画像端歪み対策、未知撮影日の汎化、稲のMetric Depth／垂直草丈GTとの比較を別実験として検討する。D-17（RMSE最良）とD-20（大局所差MAE最良）を目的に応じて保持し、単一指標だけでcheckpointを選ばない。

データ出典：UAV3DCrop／UAV3DCrop Depth, Link-Dev（[RGB](https://huggingface.co/datasets/Link-Dev/UAV3DCrop)、[Depth](https://huggingface.co/datasets/Link-Dev/UAV3DCrop_depth)、CC BY 4.0）。図はこれらのRGB/深度を加工した研究用比較図である。[出典・改変表示](figures/ATTRIBUTION.md)を参照。
