# 多スケール診断・高解像度タイル推論

20epoch完了コミット `3de99b0f35e9083c2c74426a57c44014c6b032c2` に依存する、重み固定の新実験。PR baseは `research/gpgpu4-migration-20261015`（未マージPR #3）。PR #2はmainへのローカル統合、PR #3はPR #2のheadへの20epoch完了記録であり、既存PRを変更しない。

固定source_full_v1、既存checkpoint、過去run、分割manifestを変更しない。独立test319組・過去52組は使用しない。GTはカメラ座標系Z-depthであり、稲の垂直草丈ではない。

## ファイル

- `config.json`: 推論前に定めた候補・指標・中止基準。
- `manifest.json`, `image_inventory.csv`: Day047の128画像、サイズ、入力SHA256、有効画素数。
- `pilot_selection.csv`: 作物×方向8群から各2枚、GT深度中央値の四分位順位で固定した16枚。
- `pair_manifest.json`: 全モデルで共有する5距離×2方向の有効ペア座標SHA256。座標生成は旧方式と同じseed、順序、GT mask。
- `gt_reference_*.csv`: **GTのみ**の実測とゼロ差分基準。学習済みモデルの推論結果ではない。
- `figures/gt_reference_*.png`, `.svg`: GTのみの尺度・作物・方向・位置別図。
- `code_history/baseline_004/`: baseline実行時の新コードのバイト保存。既存研究の固定ソースとは別で、監査用。以後の最適化は数値定義を変えない。

## 再現の入口

正本の仮想環境で `python -m unittest discover -s tests -p test_multiscale.py -v` を実行する。

`python scripts/multiscale_experiment.py prepare --output work/new_gt_preparation` は既存のvalidation128枚だけを読み、GT基準・入力ハッシュ・pilotと評価標本を新規出力する。既存出力は上書きしない。

GPGPU4では、自己所有localdataの独立した計算用repoにコードを配置する。`scripts/multiscale_gpu_preflight.py` で現在のGPU空き、w/who、ローカル領域、コンテナを確認し、Singularity `exec --nv --cleanenv` 内で `scripts/multiscale_experiment.py run --phase baseline` を実行する。`--help` の各パスは存在確認済みのlocaldata資産に置換する。SIFとprivate siteは既存運用記録のSHA256を照合する。ホストへのパッケージ導入・sudoは不要。

epoch17/20の両方が再現してから `scripts/multiscale_compare.py gate --runs <B0_ep17> <B0_ep20> --output <gate.json>` で証拠ハッシュ付きgateを作る。`scripts/multiscale_diagnose.py` がB0予測cacheを再利用して5距離を診断する。pilotは `run --phase pilot`、全128枚はpilotの実測と選択理由を保存したlockを用いて `run --phase full` とする。予測cacheは重み、RGB、方式、座標、実行ソースと内容SHA256を検証し、分析時に再利用する。

モデルはcheckpointのcomplete markerと全ファイルSHA256を照合し、configから構築する。既知のattention名前対応96件だけを監査し、287テンソルをstrict=Trueでロードして全値の一致を確認する。optimizerはロードせず、eval、inference_mode、FP32、requires_grad=Falseを使う。

RGB/GT、重み、全画素cache、認証情報、私的SSHログはGitに含めない。元データはLink-Dev [UAV3DCrop](https://huggingface.co/datasets/Link-Dev/UAV3DCrop)／[UAV3DCrop Depth](https://huggingface.co/datasets/Link-Dev/UAV3DCrop_depth)、[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。図はGTの統計・色付け等を加えた加工物。公開cardの許諾とZ-depthの定義を2026-10-16に確認した。
