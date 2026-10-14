# UAV3DCrop 向け Depth Anything V2 Small Metric Depth

このフォルダーは、従来の推論・評価コードから分離したファインチューニング環境です。既存の研究データ、評価結果、モデル重みは変更していません。Colabノートブック、CPU確認用コード、データ取得計画、GPU学習コードをここにまとめています。

## 実装した2つの初期化方式

| 学習方式 | 初期重み | 学習対象 |
|---|---|---|
| `relative_encoder` | Depth Anything V2 Relative Small | Relative SmallのViT backboneを転送し、Metric用neck/headを新規初期化。その後、モデル全体を学習 |
| `metric_outdoor` | Depth Anything V2 Metric Outdoor Small | Metric Outdoor Small全体を初期値として追加学習 |

どちらもHF Transformers版Depth Anything V2 Smallを用い、出力を`sigmoid × max_depth_m`としてメートル単位で扱います。損失は公式Metric Depth実装を参考にしたSiLogです。学習率はbackbone `5e-6`、neck/head `5e-5`、AdamW、gradient accumulationを使います。局所深度差損失は今回追加していません。評価指標はMAE、RMSE、AbsRel、縦横14 pixel間の局所深度差MAE、Pearson相関係数です。

モデル重みも固定revisionを使います。Relative Smallは`5426e4f0f36572d16453bbda7a8389317b1bef99`、Metric Outdoor Smallは`fd2c22027eaf20374204f14099b8341e1925ad39`です。

参照先: [公式 Depth Anything V2 Metric Depth実装](https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth)、[公式学習コード](https://github.com/DepthAnything/Depth-Anything-V2/blob/main/metric_depth/train.py)、[Transformers Depth Anything仕様](https://huggingface.co/docs/transformers/model_doc/depth_anything)。

## 固定済みデータ分割・取得計画

`config/scene_split_plan.json`の分割を維持し、RGB/depthデータはrevisionを固定しました。

| split | 撮影日・対象 | 確認済みRGB/GTペア | RGB+GT転送量 |
|---|---|---:|---:|
| train | Day019、Day028、Day040。Corn/Oat/Wheat/soybean | 384 | 5.354 GiB |
| validation | Day047。4作物 | 128 | 1.958 GiB |
| independent test | Day052の4作物、Day069/077/083のCorn・soybean | 319 | 4.949 GiB |
| 合計 | testは最終評価まで隔離 | 831 | 12.260 GiB |

RGB revision: `90226a6f9c0d7afea9e2bb863f385c2a2010a3d1`。深度revision: `3d3afe42f84bafc36c80633bbe39c059b455c895`。計画は各シーン・撮影方向ごとに最大16枚を飛行経路上で等間隔に選びます。Nadir判定は`abs(transform_matrix[2][2]) >= 0.9`です。trainは撮影日ごと128組、作物ごと96組、nadir/oblique各192組です。validationは作物各32組、撮影方向各64組です。testはCorn 128、Oat 32、Wheat 31、soybean 128、撮影方向はnadir 160/oblique 159です。

metadata照会で、train 384組とvalidation 128組に欠損はありませんでした。testでは`2025/Day052_Wheat/images/1383.JPG`のRGBはありますが、対応する`2025/Day052_Wheat_depth/1383_depth.tif`が見つからず、候補320組のうち319組が成立しています。この欠損はtestだけで、学習・検証に影響しません。Day019/028/040、Day047、test各撮影日のシーンはsplit間で重複せず、以前の52組評価とDay001/Day007にも重複しないことを確認しました。作物と撮影方向の件数は同梱の`config/verified_plan/plan_summary.json`に記録しています。

同梱計画はファイル名・サイズ・Hub object IDと小さな`transforms.json`だけを照会して生成し、RGB/depth本体は取得していません。計画更新は別の出力先を指定してください。画像本体の取得はノートブックの`RUN_DOWNLOAD`と確認byte数の両方を設定した時だけです。

## 容量と保存先

**転送量の訂正:** 前回報告した7,850,370,776 bytesは合計の計算誤りでした。検証済みCSVのtrain 5,748,288,339 bytes + validation 2,102,042,437 bytes = **7,850,330,776 bytes** が正しい値です。対象512組、各ファイル、revision、分割計画は変更していません。取得確認欄には下記の正しい合計を入力してください。旧値は安全ゲートで拒否されます。

train+validationをDriveへ保存する場合は7,850,330,776 byte（7.311 GiB）です。10%の余裕を加え、Driveに少なくとも約8.05 GiBを確保してください。DriveからColabの`/content`にもstageする場合、Driveとローカルに各約7.31 GiBが必要で、両方あわせてモデル・checkpoint用の空きを除いて少なくとも約16.1 GiBを見込んでください。Colabの`/content`はruntime終了で消えますが、Drive上の元データは保持されます。独立testを取得するとさらに4.95 GiB必要です。

Colabのディスク表示がDriveの空き容量を正確に示さない場合があります。Google Driveの空き容量とColabの`/content`空き容量の両方を目視してください。`stage_selected_data.py`はコピー前にローカル空き容量を検査し、既存の同サイズファイルを再利用し、異なるサイズのファイルを上書きしません。

## Colab初心者向けの実行手順

Notebookは見出し番号 **0 → 1 → 2 → …** の順に必要なセルを1つずつ実行します。コードセル左側の再生ボタンを押すと、そのセルだけ動きます。「すべて実行」は不要です。データ取得、モデル取得、監査、GT解析、stage、GPU事前計測、学習、独立testの実行フラグはすべて初期値Falseです。

### 1. Windows PCからGoogle Driveへアップロード

エクスプローラーのアドレス欄で、この研究プロジェクトの `uav_metric_finetuning` フォルダーを開いてください。現在のPCでは `C:/Users/81902/Documents/Codex/2026-09-20/depth-anything-v2-python-uav-rgb/uav_metric_finetuning` です。中にNotebook、README、Pythonファイル、`config/verified_plan`があることを確認します。

ブラウザーで [Google Drive](https://drive.google.com/) を開き、Google AI Plusを契約したGoogleアカウントを選びます。「マイドライブ」→「新規」→「新しいフォルダ」で `DepthAnythingV2-UAV` を作成し、その中で「新規」→「フォルダのアップロード」からWindowsの `uav_metric_finetuning` を選びます。アップロード完了まで待ってください。巨大な `data`、`.venv`、既存評価結果をアップロードする必要はありません。

すでに同名のコードフォルダーをDriveへ置いている場合は、以前のコードを別のバックアップ名へ変更してから最新フォルダーを追加してください。既存の `data`、`runs`、`model_cache`、`work` は保持します。Notebookの既定パスは `/content/drive/MyDrive/DepthAnythingV2-UAV` です。別名のプロジェクトを使う場合は最初のセルの `PROJECT_DIR` だけ実際の場所へ合わせてください。

### 2. Colab Notebookを開く

Driveの `uav_metric_finetuning/colab_train.ipynb` を右クリックし、「アプリで開く」→「Google Colaboratory」を選びます。見つからない場合は [Colab](https://colab.research.google.com/) の「ファイル」→「ノートブックを開く」のDriveタブから選ぶか、「ノートブックをアップロード」でローカルのipynbを選びます。コード一式は上のDriveパスに配置しておく必要があります。Notebookの変更はDriveへ保存してください。

### 3. CPUランタイムで準備する

「ランタイム」→「ランタイムのタイプを変更」でハードウェアアクセラレータをCPUまたはNoneにします。最初の **0. Drive接続・状態復元** コードセルを実行し、Google認証とDrive接続を許可します。成功すると「初期化完了」と、監査・GT解析の再利用可否が表示されます。初回の「未完了」は正常です。

続いて **1. CPU: 環境確認** を実行します。Python、PyTorch、TransformersとCUDAの状態を表示します。CPUではCUDA Falseでも問題ありません。不足モジュールが表示された場合のみ、同じセルで `INSTALL_EXTRA_DEPS=True` にして再実行します。requirementsはTransformers 4.56.2を指定し、Torchを含めていません。既に入っているTransformersが異なる場合も、今回検証した版へ合わせるかをここで判断してください。GPU切替後には新runtimeで依存関係をもう一度確認します。

### 4. DriveとColabローカルの空きを確認する

Drive左側の「保存容量」で、使用量と上限を確認してください。train+validationは **7,850,330,776 bytes（7.311 GiB）** なので、データだけでも余裕を含め約8.05 GiB必要です。モデルとcheckpoint分も別に確保してください。checkpointは既定で全エポック分を保持するため、本学習ではさらに保存容量が必要です。

Notebook最初のセルに出るDriveの空き表示はマウント実装に依存するため、Drive画面の容量を優先します。Colabの `/content` にstageする場合はローカルにも約8.05 GiB必要です。Driveとローカルの空きは別々の容量です。

### 5. train+validationの512組を取得する

**2. CPU: 分割計画と転送量** を実行し、train 384、validation 128、test 319と固定commitを確認します。 `RUN_METADATA_PLAN=False` のまま同梱計画を使えば、新しいメタデータ照会はしません。

**3. CPU: データ取得ゲート** で次の3箇所を設定して実行します。

```python
RUN_DOWNLOAD = True
SPLITS_TO_DOWNLOAD = ['train', 'validation']
CONFIRM_PLAN_TOTAL_BYTES = 7850330776
```

転送前に計画総量、取得済み数、新規取得量が表示されます。独立testは対象に含めません。サイズが異なる既存ファイルやrevision不一致があれば通信前に停止します。新規取得ファイルは部分ファイルへ保存し、サイズと利用可能なLFS SHA256を確認してから公開します。

### 6. ダウンロード完了・中断後の再実行を確認する

成功時はスクリプトの `Completed` に続き、Notebookに **「取得完了: 512組、1024ファイルの存在とサイズを確認」** と表示されます。終了後は `RUN_DOWNLOAD=False` に戻します。このセルはFalseでも取得状況を表示するので、再ダウンロードなしで確認できます。

通信切断時は同じ取得セルを再実行してください。完成済みの同サイズファイルはスキップし、由来を記録した部分ファイルはHTTP Range対応サーバーなら続きから取得します。Range非対応なら中断した1ファイルだけを先頭から取り直します。由来不明の部分ファイル、異なるサイズの完成済みファイル、整合しないRange応答やSHA256は自動で上書きせず停止します。GPUへ切り替えてもDriveのデータは残ります。

### 7. データ監査を実行する

**5. CPU: 対応・画素・漏洩監査** で `RUN_DATA_AUDIT=True` にします。 `FORCE_REAUDIT=False` のままで構いません。監査はDriveの永続データを読み、RGB/深度の対応、サイズ、画素寸法、GT型、0/負値/NaN/Inf、撮影日・シーン・過去評価との漏洩を検査します。

成功するとJSONの `status: pass` と `pass: True` を表示し、Driveの `work/uav_metric_finetuning/data_audit_日時.json` に保存します。未取得の場合は `skipped_missing_files` と不足数を表示し、画像本体の監査をスキップします。その場合は取得セルへ戻ってください。

正常な監査がすでにあれば、manifest SHA256、保存先、ファイルサイズ・更新日時、監査仕様と関連コード、ライブラリ版が一致した場合だけ再利用し、画素を再読込しません。旧仕様のレポートや条件が変わったレポートは拒否して再監査を案内します。意図的に再監査したい場合のみ `FORCE_REAUDIT=True` にします。再利用時はサイズ・更新日時までを検査し、全画像の内容SHA256を毎回計算する方式ではありません。

### 8. train GT分布を解析しmax_depthと設定を保存する

監査pass後、**6. CPU: train-only GT深度分布** で `RUN_GT_ANALYSIS=True` にして初回解析します。成功時にGT分位点と `max_depth候補` を表示し、Driveに `gt_distribution_train_日時.json` を保存します。正常レポートはtrain manifest、深度保存先、ファイルサイズ・更新日時、seed、1枚当たりサンプル数、解析コードと版を照合して再利用します。val/testのGTは読みません。

候補はtrainのp99.9に10%の余裕を加え5m単位へ切り上げた値です。深度の単位、分位点と外れ値をレビューし、上限を超えるGTが学習・評価から除外される点を確認して決めます。80m等の固定値を自動設定しません。

確認後、**7. CPU: 学習条件** の `OVERRIDES` と `SAVE_SETTINGS` を編集します。レビュー済みの候補を採用する場合の例は以下です。異なる上限を選んだ場合は式を選んだ数値へ置き換えます。

```python
OVERRIDES = {'MAX_DEPTH_M': GT_REPORT['suggested_max_depth_m']}
SAVE_SETTINGS = True
```

「設定保存完了」とDriveの `session_settings.json` が表示されれば、GPU切替後も復元できます。初期値はrelative_encoder、trial、1epoch、518、batch1、accum4、AMP、seed42です。比較方式は `OVERRIDES={'TRAINING_MODES':['metric_outdoor'],'RUN_LABEL':'trial_metric_s42_v1'}` で明示し、別run名を使ってください。

### 9. GPUランタイムへ切り替える

データ・監査・GT解析・設定保存を終えてから「ランタイム」→「ランタイムのタイプを変更」→利用可能なGPUを選び、保存します。切替で変数と `/content` のデータは失われるため、次の初期化が必要です。GPU種類は利用可能性に依存し、特定機種の確保や518pxの成功を保証しません。

モデル重みの初回取得はGPU切替前のCPUで済ませるとGPU待ち時間を減らせます。下の手順11をCPUで先に行っても構いません。

### 10. 切替後に再実行するセル

GPU切替後は **0. Drive接続・状態復元 → 1. 環境確認 → 2. 分割計画表示 → 7. 学習条件確認 → 8. モデルcache確認** の順に実行します。設定確認セルの `OVERRIDES={}`、`SAVE_SETTINGS=False` なら保存済み設定をそのまま表示します。

最初のセルは監査とGTレポートを軽量検査で復元し、再利用可否を日本語で表示します。再利用可能ならデータ取得、画素監査、GTスキャンを繰り返す必要はありません。再利用できない理由が表示された場合は、そのデータ取得・監査・GT解析セルだけへ戻ってください。

必要なら **4. ローカルSSDへのコピー** で `RUN_STAGE_TO_LOCAL=True` にします。セッションごとにDriveからローカルへコピーし、完成済みの同サイズコピーは再利用します。ネットワークからデータを再取得しません。監査対象は永続Driveデータで、学習時にローカルコピーの存在・サイズも確認します。速度とコピー待ち時間を比較して選んでください。

初期化前に後続セルを実行すると、NameErrorではなく「最初のDrive接続・状態復元セルを実行してください」と表示されます。初期化が途中で失敗した場合も後続処理を許可しません。

### 11. モデル重みの取得・cache確認

**8. CPU: モデルキャッシュ確認・取得ゲート** を最初は `ALLOW_MODEL_DOWNLOAD=False` のまま実行します。成功時は **「モデルcache確認成功」** と固定snapshotの保存先が表示されます。見つからない場合のみ、取得を許可するときに `ALLOW_MODEL_DOWNLOAD=True` としてこのセルだけ実行します。Relative SmallとMetric Outdoor Smallは選択方式に必要なものだけ取得し、Driveの `model_cache` を再利用します。

取得後はFalseに戻します。False時は `local_files_only=True` でネットワークへ接続せず、以後のGPU事前計測と学習も同じローカルsnapshotのみを使います。失敗時にはモデル・例外とcache未取得の案内が出ます。

### 12. GPUメモリ事前計測

**9. GPU: CUDA・メモリ事前計測** で `RUN_GPU_PREFLIGHT=True` にして実行します。必要条件はCUDA、再利用検査済み監査・GT解析、設定済みmax_depth、選択モデルのcacheです。合成1stepのforward/backward/AdamW stepを実行し、GPU名、総VRAM、開始時空きVRAM、peak allocated/reserved、成功状態と `training_feasible` をDriveの `preflight` に記録します。

`status: pass`、`training_feasible: true` が次へ進む条件です。実測の追加メモリに15%と512MiBの安全余裕を考慮します。合成正方形stepの判定なので、実データや縦横比が違うvalidationではOOMになる可能性があり、本学習中も監視します。OOMや安全余裕不足なら学習を開始しません。392/280pxを検討する場合は設定を手動変更・保存し、事前計測をやり直してください。

GPU切替後には以前の `PREFLIGHT_REPORTS` 変数を復元しません。新しいGPUランタイムでこの短い計測を実行し、そのセッションの計測結果を使って学習します。

### 13. 1epochの試験学習を開始する

設定セルで `RUN_KIND='trial'`、`EPOCHS=1`、新しい試験 `RUN_LABEL` を確認します。 **10. GPU: 試験学習・本学習** で `RUN_TRAINING=True` として実行します。事前計測と同一の方式・モデルrevision・解像度・batch・AMP・max_depthを照合し、学習直前の空きVRAMも検査します。既存runは `RESUME_TRAINING=True` と同一設定の場合だけ利用できます。

train384組とvalidation128組を使い、testは読みません。完了時にepoch1のSiLog、validation各指標、時間とVRAM、`status: completed`、最良validation RMSEを表示します。OOM時は `stopped_cuda_oom` と最後の完全なcheckpointを残して停止し、後続方式へ自動的に進みません。

### 14. 結果・checkpoint・複数epochの時間概算を確認する

**10b. 試験結果・checkpoint・時間概算** を実行します。これはCPUでも確認でき、学習は始まりません。SiLog、validation MAE/RMSE/AbsRel/局所深度差MAE/Pearson、1epochの秒数、実測peak allocated/reserved、run状態とcheckpoint保存先を表示します。 `ESTIMATE_EPOCHS=20` は時間の概算だけで、試験checkpointから20epochへ再開する設定ではありません。

Driveの `runs/<方式>_<RUN_LABEL>/` に `run_metadata.json`、`epoch_metrics.jsonl`、`run_status.json`、`latest_checkpoint.json`、`best_checkpoint.json` が保存されます。checkpointの各フォルダーには `config.json`、`model.safetensors`、`trainer_state.pt`、`complete.json` があります。初期epoch0と各完全epochを既定で全部保持します。内容SHA256を確認して復元し、同じepochのcheckpointを上書きしません。保存途中の `.tmp` は再開元に使いません。

### 15. 終了後の再開と、本学習を別runで始める方法

中断した本学習の再開では、初期化して同じ保存設定を読み、設定セルで `OVERRIDES={'RESUME_TRAINING':True}`、`SAVE_SETTINGS=True` として保存します。総epochs、方式、解像度、batch、accum、max_depth、seed、run名を変えません。依存・モデルcacheを確認し、必要ならstageし、新GPU事前計測後に学習セルを許可します。最終の完全checkpointからモデル、AdamW、scheduler、AMP scalerを復元し、未完了epochは先頭からやり直します。

**試験1epochが完了したら、本学習は新しいrunです。** 試験結果を確認後、設定セルで例えば次を保存します。

```python
OVERRIDES = {
    'RUN_KIND': 'full',
    'EPOCHS': 20,
    'RUN_LABEL': 'full_s42_v1',
    'RESUME_TRAINING': False,
}
SAVE_SETTINGS = True
```

この新runは選択した元checkpointから学習を開始します。試験runのoptimizer/schedulerを流用しません。1→20epochの強制resumeはなく、異なる総epochs・設定・manifest・コードを誤って再開しようとすると拒否します。本学習開始前に検証結果、時間見積もり、Driveのcheckpoint保存容量、残りCompute Unitsを確認してください。

### 16. Compute Unitsを確認し、作業後にランタイムを終了する

Colab右上のRAM/ディスク表示またはリソース表示を開くか、「ランタイム」→「リソースを表示 / View resources」で残りCompute Unitsと表示される消費率を確認します。表示位置・翻訳が異なる場合は「設定」→「Subscription」でも契約アカウントを確認してください。見つからない値やGoogle AI Plusの付与量をこのコードから推測しません。

試験前後の表示値を自分で記録してください。Notebookの時間概算はCCUの確定値ではありません。GPUが不要になったら、Driveへの保存を確認し「ランタイム」→「接続を解除してランタイムを削除 / Disconnect and delete runtime」で計算環境を終了します。タブを閉じただけで計算環境が直ちに止まったと判断しないでください。Googleの [Colab FAQ](https://research.google.com/colaboratory/faq.html) も参照できます。

### 独立testは最終モデル確定後だけ

初期状態では `RUN_INDEPENDENT_TEST=False` で、取得対象にもtestを含めません。方式と設定・checkpointをvalidationだけで確定してから、test319組の転送量を別途確認して取得します。最終評価セルはtestを含む物理監査pass後に評価します。test結果を見て調整した場合は、新たな未使用シーンで独立評価してください。

## 学習設定、再開、保存、再現性

GPU事前計測では有限なloss/勾配とAdamW状態の生成を確認し、AMPの数値異常で更新が省略された場合を成功扱いしません。学習中にAMPがoptimizer更新を省略した場合、schedulerもその更新では進めません。

学習CLIの既定値もepoch 1、518、batch 1、gradient accumulation 4、AMP有効、seed 42です。SiLogは有限かつ正のGT値で、`max_depth`以下の画素だけに適用されます。最終batchを含む勾配蓄積の端数も補正します。validationではMAE/RMSE/AbsRel/局所深度差MAE/Pearson rを全体および作物別に計算します。

run metadataに方式、初期checkpoint revision、設定、Python/PyTorch/Transformers等の版、GPU、manifest SHA256、コードSHA256、事前計測値を保存します。seed 42とepochごとのseedでデータ順・augmentationを固定し、再開はepoch境界で行います。bit単位再現はGPU/driver/cuDNNの差により保証しません。checkpointは既存runを上書きせず、新しいrun名に保存します。試験と本学習は別runで、resume設定が一致しない場合は拒否されます。既定では全checkpointを保持します。

## CPUとGPUの利用可能範囲

2026-10-13に確認した開発PCはAMD Ryzen 5 5625U（12論理CPU）、RAM 16,483,872,768 byte（約15.35 GiB）、AMD Radeon内蔵GPU、CUDAなし、PyTorch 2.14.0+cpu、Transformers 4.56.2でした。C:空きは307,109,732,352 byte（約286 GiB）です。このPCではGPU preflightや本格学習を行いません。

`cpu_smoke.py`はローカルcache済みの2種類のcheckpointを使い、人工画像1枚・深度1枚・56×56入力で方式A/B各々のforward/backward、finite gradient、optimizer step、Hugging Face safetensors save/reloadを確認します。UAV3DCrop実データを使わず、重みのネット取得も行いません。CPU smokeは学習品質や518解像度のVRAM容量を示しません。

## 既知の制約と今後

- metadata計画上、train+validationは512組・7.311 GiBです。まだ画像本体はダウンロードしていません。GT分布は本体取得後に計算するため、`max_depth`は未確定です。
- testの319組はtrain/validationとは別撮影日です。作物構成がCorn 128、Oat 32、Wheat 31、soybean 128と偏るため、全体平均だけでなく作物別指標を報告します。
- UAV3DCropの今回選定した作物はCorn/Oat/Wheat/soybeanです。稲の高さ推定へ応用するには、稲のUAV画像と距離・高さが校正されたGTを別途収集し、未使用の撮影日・圃場による独立評価を計画する必要があります。
- 独立test 319組も過去の52組も、学習やモデル選択には使いません。最終的な研究評価ではDay001/Day007以外の、これまで未使用の撮影シーンによる独立testを追加する計画です。
