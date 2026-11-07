# 引き継ぎ状況

正本は2026-09-20の研究リポジトリである。2026-11-05に元branchのHEAD 49bdd188および未追跡ファイルを確認し、新branch research/local-huber-20261105を作成した。既存の未追跡ファイル、固定source、過去runは変更していない。先行cloneから計画と試作・試作テストをコピーした。

計画は暫定である。学習・評価は未実施。損失部品は学習ループに接続されておらず、サンプル数・seed、best選択規則、座標と実際のOpenCV変換の照合、各群の学習・評価・再開と指定されたCPUテスト、実データの事前検証が未完了である。GPU接続は行っていない。

追加作業：元画像の画素ペア候補生成部品を試作した。画像ID・epoch・距離・軸ごとに独立した固定seedを使い、モデル予測を参照しない。候補数や実行条件は未固定であり、本学習には未使用。OpenCVの画素中心、crop、左右反転を線形合成画像で照合し、候補の再現性・境界とともに新規CPUテスト計7件が成功した。実画像や実際の学習データ処理との統合検証は未実施。

## 2026-11-05 続行分
PR #3/#4はopen、リモートの対象branchは基点49bdd188のまま。固定source検証は81ファイルについてpass。既存CPU tests 28件、新規CPU tests 11件が成功した。新規data_local.pyは既存data.pyの派生であり、実trainの1組で2 seedのRGB・縮小GTが固定版と完全一致した。実GTを使った疎な局所損失は3尺度で有効ペアが存在し、損失と出力に対する勾配が有限かつ非ゼロであることをCPUで確認した。これは学習済みモデルを通した勾配検証ではない。元解像度GTはCPUに残し、必要な標本値のみを予測デバイスに移すよう修正した。試作objectiveでE1のlambda=0が固定SiLogと厳密一致する合成試験が成功した。

既存の20 epoch学習のマスクはGT>=0.001mであり、今回指定のGT>0mと相違するため計画に明記した。config_draft.jsonの標本・checkpoint選択案は学習前の検証待ちであり、確定版ではない。実画像における画素対応の詳細検証、データローダーからobjectiveへの統合学習ループ、checkpointの実物の照合・厳密ロード、評価接続、再開処理、GPU前検証、E0評価、E1–E3学習、詳細分析、報告・PRは未実施。ローカルのwork/experiments/data配下ではmodel.safetensorsが見つからなかった。epoch20の保存先と実物はGPU接続後に照合する。VPN/SSHはまだ依頼しておらず、GPUは使用していない。本学習を開始してはならない。

## 2026-11-05 最新の続行指示への対応
学習時GTをSiLog・局所ペアとも0.001<=d<=65mに変更。評価時の旧B0コードは0<d<=65mであることを確認し計画へ明記。閾値境界テスト成功。GT全384画像の0<d<0.001mの集計は完了前に中断し、件数は未確定。

記録によるepoch20の保存先はGPGPU4ローカル計算runと `~/gpu4_research_backup_metric_outdoor_epoch20_20261015` のNFSバックアップ。model.safetensors SHA256は52769e2e95704d0862e708e8772195b861c647dd4812058eaf698a44305c3bcb、config.json SHA256は3c6ab05881e6a76edbfd33ff393fdfb082582084e9262826bb1ac0fd2b1dac7c。現在の存在・所有権・実ハッシュは未確認。探索とロードを分けること。

train_step.pyは単一batchのforwardとobjective接続だけで、run全体を実行するコードではない。合成モデルで1 optimizer stepと有限勾配を確認。initialization.pyのモデルのみ厳密ロードとハッシュ検査は合成checkpointで成功（実checkpoint未検証）。新規テスト13件成功後、閾値テスト単独を追加して成功。保存・再開統合、評価、実モデル勾配、GPU preflight、計画確定は未完了。GPU接続・実験・公開は未実施。本学習禁止。

## 全epochエンジンの試作（続行時）
`epoch_engine.py` にepoch境界の保存・再開、gradient accumulation、optimizer/scheduler/scaler状態、乱数状態、設定署名、原子的checkpoint保存、各epochの評価コールバックを試作した。`test_epoch_engine.py` で合成モデルの3 epoch連続実行と1 epochで中断後再開が同一となり、設定差による再開拒否を確認。新規試験15件成功。これは実験CLIや完全なE1/E2/E3ドライバーではない。

次の作業：1. データ・初期重み・コードの実ハッシュを設定署名に含める起動CLI、`train_step.py`、本エンジンを統合し、E1/E2/E3各群の条件・順序・実効更新数・AMP skip・安全な失敗ログを検証する。2. `scripts/multiscale_experiment.py` と `src/uav_metric_research/multiscale.py` のB0処理・固定ペアmanifestを再利用してepoch毎の絶対・全14px・固定標本評価と単独CLIを作る。既存strict_modelは過去epoch17/18/20と旧runtimeに限定し、attention state mappingを監査するため新checkpointにそのまま使えない。既存の `initialization.py` の直接strict loadが実runtimeで対応可能か要検証。3. 5 epoch、各群、評価CSV/JSONと復帰を統合試験する。4. 設定とpreflightを確定してからVPN/SSHを依頼する。5. 実checkpointのSHA256とロード、実モデル勾配・E0再現が成功するまで本学習しない。

GPU、実験、push/PRは未実施。epoch_engineの状態はCPU合成試作であり、実験の実行許可ではない。`error_if_nonfinite=True` は非有限勾配を停止するためAMP skip動作の実環境適合検証が必要。評価・CPU統合試験・GPU接続を進める前に解決する。

## 実験CLI・評価の続行作業
未完了。`evaluation_b0.py` に既存multiscale前処理、全14px累積、固定ペアhash照合、距離・作物・視点別の固定標本累積を組み合わせた評価関数を追加した。構文確認のみで、実画像全128組と既存値の再現、CSV/JSON出力、epochエンジンへの接続は未検証。実験に使用しない。

`run_local_huber.py` に引数とmanifest/初期checkpoint hash/入力ファイルの検査を設けたが、学習・評価への接続は未実装であり、dry-run以外は明示的に停止する。現在は `--help` のみ実行確認。未発見の実checkpointを必要とするdry-runも未実施。完成CLIではない。

`epoch_engine.py` の非有限勾配の扱いを修正し、CPU GradScalerでInf発生時にoptimizer・schedulerを進めないことをテストした。CUDA実機のAMPと実験全体の統合は未検証。新規CPUテスト16件成功。既存の固定source・run・他の未追跡物は変更していない。

次の具体的作業：`run_local_huber.py` の停止箇所を、実モデル構築と記録済みhash検証後の `m.strict_attention_state`、AdamW/PolynomialLR/GradScaler、新データセット、`step_batch`、`run_epochs`、`evaluate_b0` に接続する。DataLoaderやresume署名にconfig・全コード・train/val manifest・チェックポイント・評価ペアmanifestのhashを含める。各epochの集計CSV/JSONと選択規則、OOM/AMP安全処理を実装。CPU CLI合成A–Dと実データE、resume比較、実評価入出力を通し、GPU前提preflightを実装してから接続を依頼する。現時点で本学習禁止。

再開確認コマンド（正本から）：`.venv/Scripts/python.exe -m unittest discover -s experiments/local_huber_20261105 -v` と `.venv/Scripts/python.exe experiments/local_huber_20261105/run_local_huber.py --help`。GPU接続、E0–E3実行、PRは未実施。

## 今回のCLI統合状況
`run_local_huber.py` に `--synthetic` 専用経路を追加し、既存 `train_step.step_batch` → `epoch_engine.run_epochs` → 合成評価callback → checkpoint → 合成CSV/JSONを接続した。E1/E2/E3各5 epoch、各5更新が完走し、CSV/JSONに各5行を確認。E2は5 epoch連続と2 epochで停止後再開した実行で最終model stateとschedulerが完全一致し、E3設定での再開を拒否した。テストは合成のみ、正式なB0評価ではない。`train_step.py` に標本候補数の引数を追加した（正式既定8192、合成試験8）。

`run_local_huber.real_run` の試作に、AutoConfig/AutoModel生成、既存の厳密attention mapping、AdamW/PolynomialLR/GradScaler、dataset、step、evaluate_b0、epoch engineの接続を記述した。ただし未検証で `main()` から呼び出さない。本番停止は維持。実checkpoint・実モデル・実B0評価は未実施。未接続・未完成は具体的に以下：`main()`の最終RuntimeErrorの後にあるべき `real_run(args,config,train,val)` 呼び出し、安全なGPU preflight、epoch毎の正式CSV/JSONの原子的出力と再開時再構築、E0単独評価、best checkpoint選択、loss各成分・GPU時間・VRAM記録、standalone評価、モデルを通す実データforward/backward、評価固定manifest・実画像128件の検証、実checkpoint key mapping、CLI正式実行／resumeの統合試験。設定hashは検査用であり候補数・標本manifest等の実行時厳密検証は不足。`evaluation_b0` はまだ実行検証していない。

合成CLI再実行例（新規空きパス指定）：`.venv/Scripts/python.exe experiments/local_huber_20261105/run_local_huber.py --synthetic --group E2 --output <新規パス>`。2 epoch後に止めるには `--stop-after 2`、同じパスへ `--resume`。合成checkpoint・指標は実験結果ではない。VPN/SSH依頼・GPU・E0–E3実験・公開は未実施。本番の停止を解除しないこと。

## 実データ経路とGPU接続引き継ぎ
`run_local_huber.py --real-data-smoke` は `main`→`real_run`→`real_data_smoke`→実train画像・試験用小型モデルの1更新→実validation画像1枚の `evaluate_b0` callback→checkpoint/JSON/CSVを実行する。実画像ではB0の14px固定ペア座標hashがmanifestと一致し、validation有効画素20,887,680、全14px有効ペア41,646,056、14px GT差20cm以上固定標本11,532組。値は試験モデルによるもので研究結果でない。`--resume` は既存完了epochを再更新せず戻った。実画像評価のcheckpointから `export_epoch_metrics` のCSV/JSONを2回生成し重複なし。ランダム初期化Depth Anything V2はCPU 56x56でforward/backward成功、有限勾配を持つparameter tensor 282個。epoch20重みは使っていない。新規16件・既存28件CPUテスト成功。存在しない初期checkpointは正式経路で拒否。

`epoch_engine.run_epochs` にSiLog/局所成分の記録と完了checkpoint後のon_epoch callbackを追加。`real_run` はexport callbackを使用するが正式実行は引き続き未完成のため無条件に停止する。未実装：正式preflightの安全証拠、`main`から本番real_runへの分岐、正式設定と結果選択、CSV/JSONの途中失敗時の再構築、全128評価、実checkpoint実ロード、E0 CLI、GPU検証。CLIの `--real-data-smoke --group E2 --device cpu --rgb-root data/uav3dcrop_rgb --depth-root data/uav3dcrop_depth --output <新しいパス>` が再実行方法。smoke用成果はwork/local_huber_real_data_smoke_20261105_aに保存し既存成果を変更していない。

利用者へGPGPU4接続を依頼済み。利用者はs0623003でSSHを開始しパスワードを自分で入力すると表明。Computer UseからPowerShellの起動を試みたが製品ポリシーで拒否されたため、利用者にターミナルを開き `ssh s0623003@192.168.33.121` を実行してログイン済み画面を表示するよう依頼中。これを回避しない。GPU占有状況・リモートcheckpointは未確認、学習は開始していない。

追記：利用者はGPGPU4へのSSHログイン成功画面を提示したが、Computer Useの `listWindows` にはChatGPT・Brave・エクスプローラーのみが表示され、接続済みSSH端末を操作できない。ユーザーに読み取り専用のGPU・ログイン・ストレージ・Singularity確認コマンドの実行と出力提示を依頼中。GPU使用状況は未確認、学習未開始。


## 2026-11-05 中断後の続行

### ローカル正本での変更
- `objective.py`を変更し、E1（lambda=0）は局所候補抽出、GT indexing、grid_sampleを実行しないようにした。E1のlocal componentはgraph-connected zero、尺度別pair countは0として記録する。
- `run_local_huber.py`へE0の単独B0評価と固定参照値との比較、E0の評価JSON/CSV・baseline proof出力を追加した。E1-E3正式実行は、CUDA、AMP（学習群）、5分以内GPGPU4 preflight、初期checkpoint hash、GPU localdata配下の全入力/出力、成功E0 proofを要求するfail-closed経路にした。E0 proofは対応するvalidation JSONの実測値と一致しなければ拒否する。E1-E3のresume署名には一時的なGPU receiptを含めず、別ファイルへ記録する。
- `config.json`を固定版として作成し、既知E0参照、manifest/pair SHA256、480更新、optimizer/scheduler/AMP、選択規則、runtime情報を記録。`README_JA.md`にGPUコマンドを記載。計画書へ、前回の暫定記述を置き換える事前条件追記を追加。
- 既存ブランチは`research/local-huber-20261105`、実験ファイルはuntrackedのまま。コミット、push、PRはまだ行っていない。既存のuntracked報告書や過去runは保持。

### CPU検証
正本のvenvで`.venv/Scripts/python.exe -m unittest discover -s experiments/local_huber_20261105 -v`を実行し16件すべて成功。`run_local_huber.py --help`と`config.json`のJSON読み込みも成功。E1-only経路の変更後もSiLog完全一致、E2勾配、座標・データ・epoch save/resume、CPU AMP skip関連テストを含む。GPU/実checkpointを要求するE0新規評価とE1-E3本学習は含まない。

### GPGPU4で確認した事実
- `s0623003`でGPGPU4へ接続。GTX1080 8GB、Debian12、Singularity CE4.2.2。監査時はnvidia-smi 15MiB表示用途のみ、compute query空、w/whoでは自身のみ、Singularity instanceなし。`~/gpu_work`実体は`/localdata/s0623003`、467GiB中146GiB空き。
- epoch20 NFSバックアップ`~/gpu4_research_backup_metric_outdoor_epoch20_20261015/checkpoint_epoch_0020`は本人所有で、model.safetensors 95MiB・mode600。model/config両方の実SHA256が記録値一致。学習用localdataコピーはまだ作っていない。
- SIF SHA256照合済み。container torch2.11.0+cu126、transformers5.18.0、numpy2.1.3。既存Train384/Validation128、RGB/GT、pair manifestがあり、最初のtrain RGBも確認。
- 実checkpoint strict load、実Day019画像1枚、518 crop、元GT疎ペアのFP32/AMP forward-backward・optimizer 1step試験成功。E2相当でSiLog=0.00894023、Llocal=0.02352084、勾配全有限、step非skip、max allocated 817,264,128 bytes、所要4.38s。これはpreflightであり学習結果ではない。
- 既存`results/B0_ep20/summary.json`と`baseline_reproduction.json`を読取確認。128枚、B0全体518px、RMSE=0.7942392449109983m、14px全ペアMAE=0.05963898850565961m、全有効ペア5,320,891,492、固定標本25,552,938、大差分1,911,836、固定大差分MAE=0.3361806171089982m、振幅比0.3326918588874222、符号一致率0.8248495163811121で`passed=true`。これは既存成果の再利用であり、今回CLIによるE0新規再評価ではない。
- 読取専用GPU preflight receiptを`/localdata/s0623003/depth-anything-v2-uav/local_huber_20261105/gpu_audit/receipt_live.json`に作成。GPU学習は未開始。

### 次の具体的作業
1. 実験コードを`/localdata/s0623003/local_huber_project_20261105`等の新規専用rootへ配置し、既存repository_v1やrunは上書きしない。現在用意済み最小bundleにはconfigと実験runtime Python modules、正本のmultiscale.pyを含む。source_full_v1（NFSではなくGPU localdata側に既存データあり）、manifest、pair manifestは旧資材のSHA一致を確認後、新rootへ複製する。
2. model/configを検証済みNFS checkpointから新rootへコピーし、コピー後SHA256を再確認する。
3. 正本のmultiscale.pyとstrict mappingで、実checkpointを再ロードし、real-data CPU smokeでなくCUDA E0 full validation128をCLIから実行。`baseline_reproduction.json`の全checkがpassすることを確認する。
4. E0合格後、各群前にpreflightを再実行。E1/E2/E3を1群ずつ5epoch実行。AMP skip、更新数、validation、checkpointを確認し、共有利用者が現れた場合は停止してepoch checkpointから再開する。
5. その後に詳細評価・統計・図表・報告書・SHA backup・GPU解放・SSH終了・GitHub PRを行う。

## 2026-11-05 GPGPU4実行の続行記録

- E0はCUDA上で実験CLIからValidation128全画像をB0条件で再評価し、基準値と一致して合格した。valid pixels=2,668,757,375、MAE=0.5735048960m、RMSE=0.7942392449m、AbsRel=0.02918563832、Pearson=0.9943680377、14px全有効ペアMAE=0.05963898851m、14px有効ペア数=5,320,891,492。`runs/E0/baseline_reproduction.json`に記録。
- E1を2026-11-05 12:09頃(JST)にGPGPU4で開始。初期epoch20 checkpoint SHA256検証済み。CUDA/AMP、設定・データ確認の正式ゲート通過後の実行である。出力先は新規`/localdata/s0623003/local_huber_project_20261105/runs/E1`。
- 12:17:46 JSTの読み取り専用確認では、Singularity parent PID 10626、Python PID 10644が経過08:35で稼働中。GTX1080使用VRAM1143MiB、GPU使用率100%、compute queryは当該Pythonのみ、w/whoは自身のSSHセッションのみ。E1ログには起動時gate JSONのみ、runs/E1にepoch checkpoint等のファイルはまだ確認できず。これは実行中スナップショットであり、完了・epoch数は未確認。
- E1が完了するまでE2/E3を開始しない。次は同じ読み取り専用確認でログ、runs/E1成果物、PID、GPU占有を確認する。E1が異常終了していた場合はログと既存checkpointを保全し、原因を確定してから再開する。
- Codexのターミナル連携は読み取り専用であり、Computer Use規則上Windows/Codexターミナルへのコマンド入力自動化は行わない。GPGPU4の確認出力は利用者が接続済みSSHシェルで実行して提供した。

### E1監視の接続制約

12:17:46 JSTの利用者実行による読み取り専用確認ではE1(PID 10626/10644)が経過08:35で稼働、GTX1080使用率100%、VRAM1143MiB、ログはformal gateの初期JSONのみ、runs/E1にファイル出力なしだった。これはその時点のスナップショットであり、現在の完了状態は未確認である。Codexアプリのターミナル連携は読み取り専用である。Windows側から公開鍵認証のみ・BatchModeで新規SSHの読み取り確認を試したが、banner exchange timeoutで接続できず、認証情報入力は行っていない。E1を再起動・再実行してはならず、GPGPU4接続後に既存PID/ログ/checkpointを最初に確認すること。E1の出力先は`/localdata/s0623003/local_huber_project_20261105/runs/E1`、ログは`/localdata/s0623003/local_huber_project_20261105/gpu_audit/E1_20261105.log`。

### 2026-11-05 続行時の再接続試行

利用者の続行指示を受け、公開鍵認証のみ・BatchModeで踏み台DNS(192.168.33.205)経由のGPGPU4読み取り確認を試行したが、DNSホストのSSH 22番ポートがconnection refusedとなった。mips2経由の前回試行もbanner exchange timeoutで失敗している。パスワード入力・取得は行っていない。E1の最新確認済み状態は12:17:46 JSTの稼働中スナップショットであり、その後の終了・完了・checkpointは未確認。再接続後はE1の既存runを確認してから続ける。E1を重複起動しない。

### E1 epoch 1 checkpoint確認

12:30:51 JSTの利用者実行による読み取り専用確認で、`runs/E1/epoch_0001.pt`（297,438,120 bytes）、`validation_epoch_0001.csv`（552 bytes）、`validation_epoch_0001.json`（675,078 bytes）が12:27 JSTに作成されていることを確認。E1 Python PID 10644 / Singularity PID 10626は経過21:39で稼働継続。GPU瞬間使用率は0%、VRAM 1211MiB、compute queryは当該Pythonのみ。これはepoch1完了後の状態であり、E1全5epochは未完了。validation JSONの指標はまだ読み出していない。E1終了後にepoch1〜5全件を確認する。

### E1 epoch 2確認

12:49:22 JSTの確認でE1はepoch 2/5まで完了し、`epoch_0002.pt`（297,438,184 bytes）、`validation_epoch_0002.csv`（555 bytes）、`validation_epoch_0002.json`（675,121 bytes）が12:44 JSTに保存されていた。Python PID 10644 / Singularity PID 10626は経過40:10で稼働中。GPUはGTX1080、使用率100%、VRAM1211MiB、compute queryは当該Pythonのみ。GPU利用者は自身のSSHセッションのみ。学習指標は未集計。

### E1 epoch 3確認

13:03:52 JSTの1分間隔監視で、`epoch_0003.pt`が13:02 JSTに保存済みであることを確認。E1 Python PID 10644 / Singularity PID 10626は経過54:41で稼働継続、GTX1080使用率100%、VRAM1211MiB。保存済みは3/5 epochである。

### E1 epoch 4確認

13:19:53 JSTの1分間隔監視で、`epoch_0004.pt`が13:19 JSTに保存された。E1は4/5 epoch、PID 10644/10626で稼働継続、GTX1080使用率100%、VRAM1193MiB。残り1 epoch。validation指標の内容確認はE1終了後に行う。

### E1全5epoch完了

2026-11-05 13:37:53 JST、実行ログの完了JSONは`completed_epoch=5`、`global_step=480`、`updates=96`、`amp_skips=0`、group=E1。epoch_0001〜0005のcheckpointおよびvalidation CSV/JSON各5件を確認。Singularity/PythonのE1 PIDは終了し、GPUは15MiB/0%、compute-app queryは空、w/whoは自身のSSHのみ。E1学習は完了した。validation JSONの数値集計、checkpoint SHA256/バックアップは未実施。次はE1のepoch別指標を集計し、GPGPU4の新しい5分以内read-only receiptを作成してからE2を開始する。E2/E3は未開始。

### E1 validation集計とE2起動記録

E1 epoch 1〜5のvalidation JSONから14px指標を集計した。epoch 5: RMSE=0.76783767m、MAE=0.54963614m、AbsRel=0.02785830、14px全有効ペアMAE=0.05950074m、大差分(>=0.2m)固定14px MAE=0.33495933m、振幅比=0.33725415、符号一致率=0.82747422。E0比ではRMSEは改善したが、大差分MAE改善は約0.36%で事前目標5%未達、全14px MAE改善も小さい。E1はSiLogのみなので局所損失固有の改善とは扱わない。各epoch値は接続済みGPGPU4端末の集計出力に基づく。

13:50:30 JST、新しいpreflightはavailable（hostname GPGPU4、compute-app query空、w/whoはs0623003のみ、Singularity instanceなし、空き144.5GiB）。E2をPID 13580、ログ`/localdata/s0623003/local_huber_project_20261105/gpu_audit/E2_20261105.log`でnohup起動した直後にSSH接続が閉じた。E2が継続中かは未確認であり、再起動禁止。再接続後にPID、ログ、`runs/E2`、GPU状態を読み取り確認する。

### E2 epoch 1確認

14:08:16 JSTの監視で、`runs/E2/epoch_0001.pt`（297,438,120 bytes）が14:08 JSTに作成済み。E2 Python PID 13601 / Singularity PID 13580は経過17:46で稼働継続。GTX1080使用率100%、VRAM1193MiB、compute queryはE2 Pythonのみ。validation JSON/CSVの最終一覧は完了後に確認する。

### E2 epoch 2確認

14:26:17 JSTの監視で、`runs/E2/epoch_0002.pt`（297,438,184 bytes）が14:25 JSTに作成済み。E2 Python PID 13601 / Singularity PID 13580は経過35:46で稼働継続。GTX1080使用率100%、VRAM1211MiB、compute queryはE2 Pythonのみ。E2は2/5 epoch完了。

### 2026-11-05 14:43 JST — E2進捗
- E2（λ=0.05）のepoch 3/5 checkpointをGPGPU4の/localdata/s0623003/local_huber_project_20261105/runs/E2/epoch_0003.ptに保存したことを端末監視で確認した。
- 学習プロセスは継続中。epoch 1～3のcheckpointが存在する。現時点の監視ではGTX 1080上の計算プロセスはE2のPythonプロセスのみである。
- epoch別指標の集計、checkpoint SHA256、E3開始前のfresh preflightはE2完了後に実施する。

### 2026-11-05 15:00 JST — E2進捗
- E2のepoch 4/5 checkpoint /localdata/s0623003/local_huber_project_20261105/runs/E2/epoch_0004.ptを15:00 JSTに保存したことを監視表示で確認した。
- E2 Python PID 13601は継続中。GPU計算プロセスはE2のみ。epoch 5完了後、ログとvalidation結果を確認する。

### 2026-11-05 15:19 JST — E2完了
- E2（SiLog + 0.05 × Local Huber）は5/5 epochを完了した。実行ログ上のglobal_step=480、optimizer updates=96、AMP skips=0である。
- epoch 1～5のcheckpointとvalidation_epoch_0001～0005.csv/jsonの存在を端末監視で確認した。runは/localdata/s0623003/local_huber_project_20261105/runs/E2、ログは/localdata/s0623003/local_huber_project_20261105/gpu_audit/E2_20261105.logである。
- 15:19 JSTに学習PIDは終了し、GTX1080の使用量15MiB、compute-app queryは空でありGPU解放を確認した。
- 全epochのvalidation指標とcheckpoint SHA256の集計、E3直前preflightは次の確認で行う。

### 2026-11-05 15:34 JST — E3開始
- E2の完了確認後、E2のepoch 1～5 validation指標とcheckpoint SHA256を端末に出力した。数値はSTATUSへ反映する前に、端末の折返し表示を避けた再抽出で照合する。
- E3（λ=0.20）はfresh preflight通過後の15:33:50 JSTに起動した。receipt: /localdata/s0623003/local_huber_project_20261105/gpu_audit/receipt_E3_20261105T063350Z.json。
- preflightではGPGPU4/s0623003、compute-apps空、whoは本SSHセッション1件、Singularity instanceなし、空きVRAM8177MiB、localdata空き143.1GiBを確認した。
- 15:36 JSTの監視ではE3 Singularity PID 16969、Python PID 16993が稼働し、GTX1080使用量1143MiB、GPU使用率100%、compute queryはE3 Pythonのみである。

### 2026-11-05 15:51 JST — E3進捗
- E3（λ=0.20）のepoch 1/5 checkpoint /localdata/s0623003/local_huber_project_20261105/runs/E3/epoch_0001.ptを保存した。対応するvalidation CSV/JSONも同時刻に作成された。
- E3 Singularity PID 16969 / Python PID 16993が継続中。GTX1080使用量1211MiB、計算プロセスはE3のみ。

### 2026-11-05 16:09 JST — E3進捗
- E3のepoch 2/5 checkpoint /localdata/s0623003/local_huber_project_20261105/runs/E3/epoch_0002.ptおよび対応validation結果を保存した。
- Python PID 16993が継続中。GTX1080の計算プロセスはE3のみ。

### 2026-11-05 16:26 JST — E3進捗
- E3のepoch 3/5 checkpoint /localdata/s0623003/local_huber_project_20261105/runs/E3/epoch_0003.ptおよび対応validation結果を16:26 JSTに確認した。
- E3学習プロセスは継続中。GPU計算プロセスはE3のみ。

### 2026-11-05 16:44 JST — E3進捗
- E3のepoch 4/5 checkpoint /localdata/s0623003/local_huber_project_20261105/runs/E3/epoch_0004.ptと対応validation結果を確認した。
- E3 Python PID 16993が継続し、GTX1080計算プロセスはE3のみ。次はepoch 5完了と終了ログを確認する。

### 2026-11-05 17:02 JST — E3完了
- E3は5/5 epoch完了、global_step=480、optimizer updates=96、AMP skips=0、終了コード0である。
- epoch 1～5のmodel checkpointとvalidation CSV/JSONを確認した。ログ末尾でcheckpoint SHA256を出力済みだが、最終バックアップ手順でも再計算する。
- 完了後GTX1080は15MiB使用、compute-app queryは空でありGPU解放を確認した。
- 次工程はE1～E3の指標再抽出、実験成果・設定・auditのNFSバックアップとSHA256照合である。

### 2026-11-05 17:20 JST — NFSバックアップ検証
- NFS homeは/home1/eng-std/s0623003で、findmntの最終mount typeはnfs4（autofs配下）、空きは5,695,841,280 KiBであった。
- E0基準結果、E1～E3全epochのcheckpoint・validation CSV/JSON、gpu_audit、実験コード・設定・テスト、train/validation manifestを、/home1/eng-std/s0623003/gpu4_research_backup_local_huber_20261105_20261105T081949Zへ複製した。runsは4.2GiB。
- SOURCE_SHA256SUMSによるコピー元照合が成功し、E1_E3_validation_summary.csv/jsonのSHA256確認も通過した。コマンドはBACKUP_VERIFIEDを出力して正常終了した。
- GPU計算プロセスは空のままであり、使用量15MiBを確認した。

### 2026-11-05 17:30 JST — E0～E3 validation集計
CSV: validation_summary.csv（E0 epoch20とE1～E3各5 epoch、validation128）
E1/E2/E3はすべて独立したepoch20初期重みから開始し、各5 epoch・96 optimizer updates・AMP skips 0で完了した。

| 群/epoch | RMSE (m) | MAE (m) | AbsRel | 全14px MAE (m) | GT差≥20cm 14px MAE (m) | 振幅比 | 符号一致 |
|---|---:|---:|---:|---:|---:|---:|---:|
| E0/20 | 0.794239 | 0.573505 | 0.029186 | 0.059639 | 0.336181 | 0.332692 | 82.485% |
| E1/5 | 0.767838 | 0.549636 | 0.027858 | 0.059501 | 0.334959 | 0.337254 | 82.747% |
| E2/5 | 0.771474 | 0.552091 | 0.027817 | 0.059064 | 0.332659 | 0.335832 | 83.066% |
| E3/5 | 0.767921 | 0.552444 | 0.028016 | 0.058598 | 0.330311 | 0.336196 | 83.506% |

固定選択規則では、E1・E2・E3ともepoch 5が条件を満たす最小のGT差≥20cm局所MAEである。E0比でE3はRMSEが3.31%低く、全14px MAEが1.75%低く、大差分14px MAEが1.75%低い。絶対RMSEの5%許容内であるが、事前成功目標の大差分局所MAE5%改善には届かない。E2の大差分MAE改善はE0比1.05%、E1比0.69%。E3はE1比1.39%改善、RMSEはE1比0.01%高い。振幅比はE3でも0.336であり約1/3への過小推定が残る。符号一致率はE0比1.02 percentage points改善した。
H1の事前目標は達成せず、H2も振幅再現の実質的改善を確認できない。H3（絶対RMSEの重大悪化なし）は点推定上達成した。H4についてはE1比の局所MAEに小幅な改善があるが、局所指標の画像単位対応bootstrap CIは未計算のため、局所損失固有の効果を統計的に確立したとは扱わない。評価JSONのper_imageは絶対指標および全有効14pxのlocal_depth_delta_mae_m/local_pair_countを含む。GT差20cm以上固定標本の局所MAEは画像別内訳を含まない。作物・視点別は既存集計に含むが、端距離・GT深度帯・画像別局所差分の詳細分析は未完了である。

### 2026-11-05 17:40 JST — 主結果報告と解析資産
- 正本に report_ja.md を作成し、実験方法、E0〜E3、事前仮説の支持状況、NFS保全、限界、再現情報を記録した。
- analysis/summarize_results.py から事前選択規則を再計算し、E1/E2/E3全群でepoch 5を選択。selected_comparison.csv、summary.json、5指標のepoch曲線 learning_curves.svg を生成した。
- E3 epoch5はE0比RMSE -3.31%、全14px MAE -1.75%、大差分14px MAE -1.75%。E1比ではRMSE +0.01%、全14px MAE -1.52%、大差分MAE -1.39%。H1（5%大差分MAE改善）は未達。振幅比は0.333から0.336で約1/3の過小推定が残る。
- 画像単位bootstrap、crop/view/端距離/GT深度/差分階級の公開用集計、画像比較は未作成。NFS上のJSONに集計要素は残るが、SSH/VPN切断後のため未抽出であり、推定値を作らず未完了と記録した。
- config.jsonのmanifest_sha256.fixed_pairsは63文字で、実manifestのSHA256と不一致であることを検出。実行コードは固定manifestの実SHAをsignatureへ記録し、評価は座標列の一致を検証している。元configは改変せず、provenance issueとして記録した。
- GitHub PR #3/#4は2026-11-05時点でopenかつ未merge。今回branchはPR #4のhead commitを基点としており、新規PRのbaseは research/multiscale-tile-inference-20261016 となる。
- GPU/SSH接続はユーザーによるexit後に閉じたことをCodex端末履歴で確認。学習完了後のGPUはcompute-appなし、15MiB使用の状態を最後に確認済み。

- NFSから全尺度・層別JSONとabsolute/全14px局所MAEの画像平均paired bootstrapを1回のCPU処理で抽出する手順を analysis/NFS_AGGREGATION_COMMAND_JA.md に作成した。利用者側のVPN/SSH再接続が必要だが、GPUは使用しない。全14px局所MAEの画像平均bootstrapは既存JSONから可能。GT差20cm以上固定標本の画像別値はなく、この主要指標のbootstrapにはfinal E0/E1/E2/E3の再評価が要る。




### 2026-11-05 続行引き継ぎ状態
- E0再現、E1/E2/E3 5epoch学習とNFS checkpoint backup/hash照合は完了。GPU compute jobは終了し、Codex端末履歴でGPGPU4→DNS→踏み台SSH切断を確認した。
- 正本に主結果報告、16行epoch summary、選択規則計算器、5指標SVG、provenance、NFS抽出手順を保存した。比較表ではE1/E2/E3すべてepoch5選択。集計スクリプトは再実行済み、SVG XMLとJSON parseを確認した。
- 現ブランチはresearch/local-huber-20261105、HEAD=49bdd188d9caa01cf9bacdf02f7aebec65537fcb。実験ディレクトリを含む変更は未commit、未push。PRは未作成。PR #3/#4はopen/unmergedで、予定baseはresearch/multiscale-tile-inference-20261016。
- 他の未追跡docs、preparation、presentationファイルには触れていない。commit時はexperiments/local_huber_20261105のみを選択する。
- 残作業はNFS JSONから条件別表・absolute/full14 image-mean bootstrapを取得すること（手順: analysis/NFS_AGGREGATION_COMMAND_JA.md）。GT差20cm以上14pxの画像別bootstrapと画像比較は既存JSONに画像別サンプルがないため、必要ならfinal E0/E1/E2/E3の追加評価が必要。詳細統計が完成するまで報告書・PRを完了扱いにしない。
- 次のユーザー側操作は、必要時にVPN/SSHでGPGPU4へ再接続し、NFS集計手順を1回実行してCOMPACT_RESULTSを返すこと。これはCPU処理でGPU利用を伴わない。

### 2026-11-07 — 最終再評価・統計解析・報告書更新
- NFSの最終E0〜E3結果bundleを取得し、転送archive SHA256 `4c9dc18d2ccf34a133ea5c7241d5dd682480c8958989d23176f94fb2604c0f4e` を検証した。bundle内の派生成果15ファイルは `DERIVED_RESULTS_SHA256SUMS` と照合済み。
- E0〜E3最終選択checkpointを、既存B0全体518px・Validation128・共通mask/fixed-pair条件で再評価した。512件の画像×群レコードを保存し、pool絶対指標7項目、full14有効pair数、GT差≥20cm固定pair数が既存run結果と許容差1e−7以内/完全一致であることを確認した。E0は既存B0基準値と一致。Independent test319は使用していない。
- 画像単位paired bootstrapを10,000回、seed42で実施した。各反復で全モデル共通の画像drawを使い、pool指標は画像別誤差和・SSE・有効数・ペア数から再計算した。画像ごとの指標平均bootstrapは別の推定量として保存した。主要GT差≥20cm・14px局所MAEはE3−E1 = −0.004648m、95%区間[−0.005098, −0.004205]m。絶対RMSEのE3−E1区間は0を含む。
- 画素距離別（14/28/56/107/214px）、GT差分階級、作物・視点、画像端距離、GT平均深度帯、差分符号の条件別表と画像数・pair数を作成した。固定pair由来43層のうち有効pairがある42層を報告し、pair数0のendpoint-zone層は指標を欠損のまま保持した。
- `analysis/finalize_analysis.py` と `analysis/build_analysis_tables.py` を再実行し、pool reconciliation、CSV/JSON出力、図表生成を確認した。main figureの下端ラベルを修正し、画像を目視確認した。`analysis/write_final_ledgers.py` は4 checkpoint記録、15派生入力、48解析成果物のSHA256を検証・出力する。
- `report_ja.md` を最終解析に合わせて更新し、`README_JA.md`、checkpoint hash記録、provenance、条件別Markdown表を整備した。E0初期checkpointのhashはE0 summary/評価signatureに記録される一方、取得済みrun backup `SOURCE_SHA256SUMS` にcheckpoint本体の行がないことを明記し、E1〜E3のmanifest照合と区別した。元の `config.json` は変更していない。
- この最終解析中の再学習・追加GPU計算はない。既知の最終GPU再評価後の状態はcompute-appなし、15MiB使用であり、このターンでは接続済みSSHの出力を読み取ったのみでGPUジョブを起動していない。
- 残作業は、公開対象のファイルを `experiments/local_huber_20261105` 内から個別選択して `git diff --check`、commit、remote branch/base最新状態の確認、push、新規PR作成を行うこと。根ディレクトリに残る他の未追跡ファイルはステージしない。

### 2026-11-07 — GitHub公開完了
- 最終解析一式をcommit `81cae91682df67cf438fa736a12e75a62d751f3a` として保存し、`research/local-huber-20261105` にpushした。対象は実験ディレクトリ内85ファイルのみである。
- PR #5を作成した: https://github.com/hw02l/depth-anything-v2-uav-rice-research/pull/5 。baseは `research/multiscale-tile-inference-20261016`、headは `research/local-huber-20261105`。GitHub再確認時点でopen、未merge、mergeableである。比較実験のPR #3/#4は変更していない。
- 公開前に解析成果物48件のSHA256を全件照合し、解析スクリプトのSHA256とmetadataの一致、`git -c core.whitespace=cr-at-eol diff --cached --check` の成功を確認した。
- 未追跡の別資料および `analysis/nfs_inputs/` の取得用archiveはステージしていない。RGB/GT原本、巨大checkpoint、prediction cache、認証情報はPRに含めていない。
