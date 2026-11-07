# 局所深度差損失による追加学習：事前実験計画

状態：事前条件を固定してE0再評価・E1〜E3各5 epochの学習・Validation128評価を完了した。2026-11-07に条件別解析、画像単位paired bootstrap、pool再評価照合、報告書と図表を完了し、commit・pushおよびPR #5作成まで完了した。PR #5は research/multiscale-tile-inference-20261016 をbaseとするopenなPRである。実験IDは local_huber_20261105。既存runと固定sourceは変更しない。PR #3/#4は確認時点でopenかつ未mergeであり、変更していない。

## 問いと事前基準
SiLog に元画像座標の局所差 Smooth L1 を加えたとき、追加学習量を揃えた SiLog のみと比べ、局所差再現が改善し、絶対精度を維持できるかを検証する。E0 比で 14px・GT差0.2m以上の固定標本 MAE を5%以上低減、振幅比を1に近づけ、絶対 RMSE と14px全有効ペア MAE の悪化を各5%以内、符号一致率の大幅悪化を避けることを目標とする。E1との比較も必須。単一指標の後付けによる選択をしない。最終epochと validation 選択 checkpoint は区別し、選択規則は学習前に固定する必要がある。

## 群と学習
E0：第3回 epoch20 重み、追加学習なし。E1：同一重みから SiLog のみ5 epoch。E2：SiLog + 0.05 L_local。E3：SiLog + 0.20 L_local。各群は独立にモデル重みのみを厳密ロード、新しい AdamW と PolynomialLR(power 0.9) を用い optimizer/scheduler state は継承しない。train384、Day047 validation128、test319は使用しない。518 crop、batch1、accumulation4、seed42、encoder LR 2e-6、head LR 2e-5、weight decay 0.01、上限80m、学習GT 0.001<=d<=65m（評価は既存B0に従う）。各群で順序・augmentation・更新回数を揃え、AMP skip と scheduler・実時間を記録。5 epoch全checkpointおよび再開用状態を独立保存する。

## 局所損失
元解像度の水平・垂直14/28/56pxペアを予測に依存しない固定乱数で採取し、両端が有効GTかつ crop 内にあるものだけ使う。入力の実際の resize/crop/flip を追跡し OpenCV の画素中心と align_corners=True の復元規約を区別し、微分可能な bilinear grid_sample で予測値を得る。各尺度で e=(p_i-p_j)-(d_i-d_j) に PyTorch Smooth L1 beta=0.1m を適用（出力単位 m）、|GT差|<0.2mの重み1、>=0.2mの重み3、重み和で正規化、尺度平均。尺度に有効ペアがない場合の扱いは実装とテストで明示し、全無効は graph-connected zero とする。各尺度の標本数と seed 規則は事前検証後、本学習前に固定・記録する。各群の学習順序・augmentationに損失標本用乱数が影響しないよう独立乱数を使う。

学習前に座標変換、crop境界、無効値、定数・完全予測・平行移動・符号反転、再現性、勾配、SiLog互換、checkpoint厳密ロードをCPU検証する。実データで SiLog と局所損失の値および各勾配ノルム、λを掛けた寄与、有限性、VRAM、速度をGPUで測定し、明らかな破綻があれば本学習を行わない。代替方式への変更は実験開始前に説明する。

## 評価と解析
E0–E3はB0全体518px推論のみ、共通GTマスク・画像ID・固定ペアを用いる。14px水平垂直の全有効ペア MAE と固定標本を別集計。既存固定標本 seed は 42+original_image_index*10000+stride、各軸候補100000、座標 hash と 14px件数（全有効 5,320,891,492、固定 25,552,938、固定GT差>=0.2m 1,911,836）を照合する。14/28/56/107/214pxで MAE、RMSE、Pearson、Spearman、符号一致、傾き、Σ|予測差|/Σ|GT差|、GT差階級別MAE、ゼロ予測 baseline を記録。絶対MAE/RMSE/AbsRel/Bias/Pearsonを併記。作物、視点、画像端距離、GT深度、符号、距離で層別。画像単位の対応差について固定seedの10000回 bootstrap 95%区間を求め、pool指標と画像平均を区別し、同一撮影日由来の依存性を明記する。B3は異なる推論方式の参考値に限る。

## 運用と停止条件
VPN・SSHは利用者が接続。GPGPU4優先、共有利用者・GPU・ストレージ・Singularityを読み取り専用で確認し、他者稼働時は使わない。データと実行中出力は ~/gpu_work の実体 /localdata/ユーザー名 に配置。チェックサム・所有権・容量を検証して既存物を再利用。1群ずつ実施、OOM等で条件を自動変更しない。checkpointを許可されたNFSへ退避し SHA256照合後、自分のプロセスのみ停止し解放・SSH終了を確認。生データ、予測キャッシュ、checkpoint、認証情報はGitに含めない。PRは成果と検証完了後に作成する。

## 未確定・実行前に解決
既存の実際の変換順序と変換パラメータ追跡、訓練用サンプル数、validation best の選択規則、epoch20 checkpoint所在・SHA256、実行環境、GPU所有状況、計算予算と検証結果は未確認。計画はこれらを検証し、実行前に具体化する。ここに結果を推測して記入しない。

## 正本の訂正・引き継ぎ
2026-11-05に利用者が指定した正本は `C:\Users\81902\Documents\Codex\2026-09-20\depth-anything-v2-python-uav-rgb` である。2026-09-22の独立cloneは先行作業の一時的な置き場であり、以後の成果の正本ではない。正本の既存未追跡ファイルは保持し、新規branch `research/local-huber-20261105` を 49bdd188 から作成した。元epoch20の記録済みmodel.safetensors SHA256は `52769e2e95704d0862e708e8772195b861c647dd4812058eaf698a44305c3bcb` である。この行は接続前の引継ぎ時点の記録であり、実験開始前にGPGPU4でcheckpoint本体・SHA256の一致を確認した。

## 実行前に固定する候補仕様（事前試験待ち）
設定案は `config_draft.json` に保存した。各距離・各軸8192候補を復元可能な独立seedで復元抽出し重複を許す。三尺度は有効な尺度を等平均し、有効尺度が一つもない場合は勾配経路を保つゼロを返し、件数を記録する。GPU事前試験前の候補案であり、検証後に設定ファイルのハッシュを確定してから学習する。

best選択は、絶対RMSEおよび全14px MAEがE0の1.05倍以下のepochから、14pxのGT差0.2m以上の固定標本MAEを最小にする。同MAEが最小値から0.000001m以内なら絶対RMSEが低いepoch、なお同値なら早いepochとする。条件を満たすepochがなければ選択なし。最終epochは別途必ず報告する。

既存 `source_full_v1/train.py` のSiLog用マスクは `target >= 0.001` だが、本実験の指定は `target > 0` である。利用者の2026-11-05続行指示により、学習時は既存runと同じ `0.001<=depth<=65m` に統一し、局所ペアの両端にも同じ条件を適用する。評価マスクは既存B0の評価実装に従う。この記述はcheckpoint実測前の段階であり、後続のGPGPU4事前検証で実物とのSHA256一致を確認した。

学習時のマスク下限を0.001mに変更する指示は、以前の「0超」の暫定案に優先する。評価側の既存B0実装は有限かつ0<depth<=65mである。


## 2026-11-05 事前条件の確定追記（上記の暫定記述を置き換える）

学習候補は各尺度14/28/56px、水平・垂直各8192候補/画像/epoch/尺度とし、復元抽出・重複許可とする。seed=42に画像相対パス・epoch・尺度・軸をSHA256由来で混合するため、E2/E3は同じ標本を使い、E1は局所標本生成自体を行わない。両端GTは有限かつ0.001<=d<=65m、両端は変換後の518crop内でなければ除外する。予測は元画像の画素中心をOpenCVのhalf-pixel resize対応でcrop座標へ移し、align_corners=Trueのgrid_sampleで疎に読む。beta=0.1m、GT差<0.2mの重み1、>=0.2mの重み3、重み和で正規化、非空の尺度を等平均する。空尺度は平均対象外、全尺度空は勾配接続ゼロである。

各群はseed42、384枚、518crop、batch1、累積4、AdamW(weight_decay=0.01; encoder LR=2e-6; head/neck LR=2e-5)、新規PolynomialLR(power=0.9)、5epoch、96 optimizer updates/epoch（計480）とする。各群ともepoch20のmodel weightsのみ厳密ロードし、optimizer/scheduler/scalerは新規作成する。学習時GTは0.001<=d<=65m、既存B0評価は有限かつ0<d<=65mであり、差を維持する。CUDA FP16+GradScalerを学習に用い、unscale後norm1.0でclipし、AMP skip時はschedulerを進めない。validationはFP32で全128枚、各epoch実施する。

E0評価の参照値は従来B0_ep20の固定成果である。absolute RMSE=0.7942392449109983m、MAE=0.5735048960437379m、AbsRel=0.0291856383240263、Pearson=0.9943680377017142、14px全ペアMAE=0.05963898850565961m（5,320,891,492ペア）、有効画素2,668,757,375、固定14px大差分標本1,911,836ペア、MAE=0.3361806171089982m、振幅比=0.3326918588874222、符号一致率=0.8248495163811121である。既存 `baseline_reproduction.json` は `passed=true`。新CLIでも同じreferenceを許容差内で再現するまでE1-E3を開始しない。各epochのselectionはE0比RMSEおよび14px全ペアMAEが各1.05以下の中から固定大差分標本MAE最小、1e-6m以内の同値ならRMSE、次に早いepochとする。該当なしはbestなしとし、E1比較を独立に報告する。

GPGPU4はDebian12/GTX1080 8GB、Singularity CE 4.2.2、SIF SHA256 `94a2c940d4a7e4c191b5de4df6eff0607b9d06f18e25be3004525ada518e292c`、container torch 2.11.0+cu126/transformers 5.18.0である。epoch20 NFSバックアップは本人所有・mode600で実物とmodel/configの記録済みSHA256一致を確認した。コンテナ実験前のGPU試験は実データ1枚でSiLog=0.00894023、Llocal=0.02352084、E2総損失有限、勾配有限、optimizer step非skip、max allocated約0.76GiBであった。これは1step preflightであり学習結果ではない。

GPGPU4の11月5日読み取り専用preflightでは、利用者自身のSSHのみ、CUDA compute processなし、Singularity instanceなし、`~/gpu_work`実体`/localdata/s0623003`、空き146GiBを確認した。receiptは `/localdata/s0623003/depth-anything-v2-uav/local_huber_20261105/gpu_audit/receipt_live.json` に作成した。GPU使用時は毎群5分以内のreceiptを要求し、保存先は同じlocaldata内のみとする。既存B0_ep20の全128画像reference再現済みと新CLI E0再評価は区別する。E0新評価、実験コードのGPU用独立配置、E1-E3学習は未実施である。

## 実験後の状態訂正（2026-11-05）

本節は冒頭の「計画・学習未実施」および「未確定・実行前に解決」節の進捗記述を置き換える。学習条件そのものは変更しない。

- GPGPU4でepoch20 checkpointの実体・記録済SHA256を確認し、厳密ロードを実施した。E0 B0再評価は記録済み基準値の許容差内で成功した。
- E1/E2/E3はそれぞれepoch20モデル重みから独立に開始し、各5 epoch、480 global steps、96 optimizer updates/epoch、AMP skip 0で完了した。全epochにValidation128を実施した。
- 群ごとのcheckpointとvalidation JSON/CSVはNFSへ退避し、SOURCE_SHA256SUMSによるコピー照合が成功した。バックアップは /home1/eng-std/s0623003/gpu4_research_backup_local_huber_20261105_20261105T081949Z にある（パス両端の空白を除く）。
- 主結果は `report_ja.md` と `validation_summary.csv` に記録した。E1/E2/E3の事前best規則選択は各epoch 5である。
- 詳細条件別の公開表・図、paired bootstrap、画像例、PRは未完了である。設定のfixed-pair SHA欄は実ファイルのSHA256と一致しないため、元設定を変更せずprovenanceの問題として記録した。

## 最終解析状態（2026-11-07）

本節は前節の「詳細条件別解析・paired bootstrap・PRは未完了」を更新する。保存済みの最終E0〜E3評価を既存B0条件で再照合し、条件別表・図、10,000回の対応付き画像bootstrap、日本語報告書の最終版を作成した。分析は学習・推論を追加せず、検証済みper-image十分統計と固定GT由来層別集計から実施した。Independent test319は未使用である。主指標のE3−E1差は−0.004648m、95% percentile区間[−0.005098,−0.004205]mだが、事前5%改善目標には届かず、振幅比も約0.336にとどまる。図表・入力・派生結果のSHA256は `analysis/final_20261107/ANALYSIS_OUTPUT_SHA256SUMS` に記録した。元 `config.json` のfixed-pair SHA256欄は修正せず、実manifestと設定記載誤りを別途provenanceに記録した。GitHubではPR #3/#4のopen状態を維持し、本branchから `research/multiscale-tile-inference-20261016` をbaseとする新規PRを作成する。

\r\n
