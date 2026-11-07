# 局所深度差Huber損失による追加学習実験報告

実験ID: local_huber_20261105<br>
実行日: 2026-11-05<br>
ブランチ: research/local-huber-20261105<br>
基準コードcommit: 49bdd188d9caa01cf9bacdf02f7aebec65537fcb

## 1. 目的

Depth Anything V2 Metric Outdoor Smallの第3回学習epoch 20モデルは、絶対深度を良好に推定する一方、14px離れた画素間の大きな深度差を過小推定していた。本実験では、B0全体518px推論を固定し、追加学習でSiLogだけを用いる条件と、SiLogに元画像座標の局所深度差Huber損失を加える条件を比較した。

主要な問いは、局所損失によって大きな局所差分のMAEが低下し、その改善が単に5 epoch継続学習した効果を上回るかである。

## 2. 仮説と事前基準

- H1: GT差20cm以上の14px固定標本MAEをE0より5%以上低減する。
- H2: 振幅比 Σ|ΔPred| / Σ|ΔGT| を1に近づけ、過小推定を緩和する。
- H3: 絶対RMSEの悪化をE0比5%以内に抑える。
- H4: E1（SiLogのみ）との比較で、局所損失に固有の改善を確認する。
- H5: 効果は作物、撮影方向、画像端、GT深度、画素間隔により異なる。

チェックポイント選択規則は学習前に固定した。各群で、絶対RMSEおよび全14px局所MAEがE0の1.05倍以内であるepochを候補とし、その中でGT差20cm以上の14px固定標本MAEが最低のepochを選ぶ。同値（1e-6m以内）の場合はRMSEが小さい方、次に早いepochを選ぶ。

## 3. データと評価分割

UAV3DCropのTrain384（Day019、Day028、Day040）で追加学習し、Day047のValidation128で各epochを評価した。Independent test319画像は、途中確認、モデル選択、追加実験のいずれにも使用していない。

学習GTは 0.001 <= depth <= 65m、評価GTは既存B0実装に合わせて 0 < depth <= 65m とした。モデル出力上限は80mである。518pxの全体推論を全群に適用した。

## 4. 損失関数

局所ペア (i,j) について、

Δd_ij = d_i - d_j, Δp_ij = p_i - p_j, e_ij = Δp_ij - Δd_ij

とし、局所Huber損失を次の重み付き平均とした。

L_local = Σ_ij w_ij SmoothL1_beta=0.1m(e_ij) / Σ_ij w_ij

GT差が20cm未満の重みを1、20cm以上を3とした。元画像の水平・垂直14/28/56pxペアを用い、各距離・軸あたり8192候補を固定乱数から抽出し、E2/E3で同じ標本を用いた。GT端点が学習マスク外、または変換後crop外なら除外した。予測は元画像座標から学習crop座標へ写像し、微分可能なbilinear samplingで取得した。

総損失は L_total = L_SiLog + λ L_local であり、E1/E2/E3のλはそれぞれ0、0.05、0.20である。

## 5. 学習条件と実行環境

| 項目 | 条件 |
|---|---|
| 初期モデル | 既存epoch 20、Metric Outdoor Small |
| 群 | E1: SiLog、E2: SiLog+0.05 Local Huber、E3: SiLog+0.20 Local Huber |
| 初期化 | 各群ともepoch 20のモデル重みから独立。旧optimizer/schedulerは引き継がない |
| 更新量 | 各群5 epoch、各epoch 96 optimizer updates、合計480 |
| Optimizer | AdamW、weight decay 0.01 |
| 学習率 | encoder 2e-6、neck/head 2e-5 |
| Scheduler | 群ごとに新規PolynomialLR、power=0.9 |
| Batch / accumulation | 1 / 4 |
| Seed | 42 |
| AMP | CUDA FP16 + GradScaler。unscale後にnorm 1.0でclip |
| AMP skip | 全群0回。各群の実更新96/epoch。scheduler skipなし |
| GPU | GPGPU4、GTX 1080 8GB、Debian 12、Singularity CE 4.2.2 |
| Container | PyTorch 2.11.0+cu126、Transformers 5.18.0、NumPy 2.1.3 |

初期重みのSHA256は 52769e2e95704d0862e708e8772195b861c647dd4812058eaf698a44305c3bcb、config SHA256は 3c6ab05881e6a76edbfd33ff393fdfb082582084e9262826bb1ac0fd2b1dac7c であり、GPGPU4上の実体を照合してからロードした。E0の再評価は既存B0参照値に一致し、正式学習ゲートを通過した。

## 6. 評価指標と標本

絶対深度はMAE、RMSE、AbsRel、Bias、Pearsonを評価した。評価マスクは全モデル共通の `0 < depth <= 65 m` である。局所差は元画像座標で14/28/56/107/214 pxを評価し、全14 px有効ペアpoolと固定標本を分けた。絶対深度の有効画素は2,668,757,375画素、全画像を走査した14 pxの水平・垂直有効ペアは5,320,891,492組である。別の固定標本は14 px全階級で25,552,938組、うちGT差20 cm以上は1,911,836組（有効ペアを持つ画像126/128）である。モデル間で画像・GTマスク・固定標本を共有した。

NFSから取得した最終選択checkpointのper-image十分統計を検証し、絶対MAE/RMSE/AbsRel、全14 px局所MAE、主要なGT差20 cm以上14 px固定標本について、画像別誤差和・二乗誤差和・ペア数が保存されていることを確認した。主要指標の画像別評価は同じ固定ペアによる再評価データから復元した。画素間隔・GT差階級・作物・視点・画像端・GT深度・差分符号の全比較値と標本数は [条件別評価詳細表](analysis/final_20261107/TABLES_JA.md) および [condition_comparison_all.csv](analysis/final_20261107/condition_comparison_all.csv) に示す。局所Spearman相関と回帰傾きは保存済み十分統計から復元できないため、推定値を補わず未報告とした。

## 7. E0〜E3の主結果

全行はValidation128画像のB0全体518px評価である。E1〜E3は事前選択規則によりepoch 5が選択された。

| 群/epoch | MAE (m) | RMSE (m) | AbsRel | Bias (m) | Pearson | 全14px MAE (m) | GT差≥20cm 14px MAE (m) | 振幅比 | 符号一致率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E0 / 20 | 0.573505 | 0.794239 | 0.029186 | 0.088099 | 0.994368 | 0.059639 | 0.336181 | 0.332692 | 82.485% |
| E1 / 5 | 0.549636 | 0.767838 | 0.027858 | 0.057692 | 0.994699 | 0.059501 | 0.334959 | 0.337254 | 82.747% |
| E2 / 5 | 0.552091 | 0.771474 | 0.027817 | 0.045059 | 0.994667 | 0.059064 | 0.332659 | 0.335832 | 83.066% |
| E3 / 5 | 0.552444 | 0.767921 | 0.028016 | 0.060884 | 0.994712 | 0.058598 | 0.330311 | 0.336196 | 83.506% |

絶対MAE/RMSEの分母は共通有効画素2,668,757,375である。全14 px MAEは全有効ペア5,320,891,492組を走査したpool値、GT差≥20 cm局所値・振幅比・符号一致率は固定標本1,911,836組のpool値である。epoch別推移は [learning_curves.svg](analysis/learning_curves.svg)、全epochの機械可読データは [validation_summary.csv](validation_summary.csv)、選択epochと相対変化は [selected_comparison.csv](analysis/selected_comparison.csv) を参照する。

## 8. 結果の解釈

E1はE0に比べてRMSEを3.32%低下させた。5 epochの継続学習自体が絶対深度を改善しており、E0だけとの比較では局所損失の効果を分離できない。

主要なGT差≥20 cm・14 px固定標本MAEはE3がE1より0.004648 m（1.39%）、E2がE1より0.002301 m（0.69%）低かった。全14 px局所MAEのE3−E1差は−0.000903 m（1.52%）であった。E3のRMSEはE1より0.000084 m（0.011%）高く、点推定では実質的に同程度である。一方、絶対MAEはE3がE1より0.002808 m高い。局所誤差の低下と絶対深度指標の間に小さなトレードオフがある。

E0に対するE3の主要局所MAE改善は1.75%で、事前基準の5%に届かなかった。GT差≥20 cm・14 pxの振幅比はE0の0.332692からE3の0.336196へ上がったものの、対照のE1は0.337254であり、E3はE1より1に近くない。大きなGT差の振幅はなお約3分の1しか再現されない。従ってH2は局所損失固有の効果として支持されない。符号一致率はE1からE3で0.76 percentage points上昇したが、振幅不足の解消を意味しない。

画像単位paired bootstrap（10,000回、seed 42）のpooled主要指標では、E3−E1のGT差≥20 cm局所MAE差は−0.004648 m、95% percentile区間は[−0.005098, −0.004205] mであり、区間は0を含まない。画像ごとのMAEを等重みした別推定量でも、E3−E1は−0.003997 m [−0.004523, −0.003465] mであった。主要固定標本を持つ126画像中124画像でE3がE1より低誤差、2画像で高誤差であった。Day047内のこの画像再標本化のもとでは小幅な局所MAE差を支持するが、validation選択や同日画像依存を含む不確実性は区間に反映されない。従ってH4は「Day047の選択済みcheckpointに条件づけた小幅な差」として支持されるにとどまり、未知日付への一般化や損失の普遍的優位性を示すものではない。

H1の5%成功目標は未達である。H3の絶対RMSE悪化5%以内はE0比の点推定および画像bootstrap区間で満たした。E3−E1のRMSE差の95%区間は[−0.012727, 0.012215] mで0を含み、E1に対する重大なRMSE悪化は確認されなかったが、絶対MAEやAbsRelを含むすべての指標が改善したわけではない。

## 9. B0/B3との位置づけ

既存B0 epoch20は本実験のE0と同一推論条件であり、再評価値を照合した。既存B3はRMSE 0.7940m、全14px MAE 0.06406m、大差分MAE 0.32801m、振幅比0.4014である。B3は異なる推論方式（低周波・高周波融合）を使用しているため、損失関数の効果を切り分ける直接対照ではない。参考値としてのみ扱う。

## 10. 学習安定性・成果保全

E1/E2/E3は各5 epochを完了した。各群のglobal stepは480、optimizer update数は96/epoch、AMP skipは0であった。群間で初期重み、seed、データ分割、学習量、最適化条件を統一し、E1からE2/E3への重み継承は行っていない。

epoch checkpointとvalidation CSV/JSONを含む約4.2GiBのrun成果を、次の本人所有NFS領域へ退避し、コピー元とのSHA256照合を行った。

/home1/eng-std/s0623003/gpu4_research_backup_local_huber_20261105_20261105T081949Z

バックアップ内のSOURCE_SHA256SUMSに全ファイルの照合記録がある。巨大checkpointと生RGB/GTはGitHubへ含めない。

## 11. 統計的不確実性と限界

pool差のbootstrapでは画像ごとの誤差和・二乗誤差和・有効画素数・ペア数を共通画像drawで再標本化し、各反復でMAE、RMSE（二乗誤差和から再計算）、AbsRel、振幅比、符号一致率を再計算した。画像ごとの指標の単純平均を推定するbootstrapも別に計算し、pool推定量と混同していない。4つの事前指定比較（E3−E1、E2−E1、E3−E0、E2−E0）と13指標を記録した。全比較は [paired_bootstrap_pooled.csv](analysis/final_20261107/paired_bootstrap_pooled.csv)、画像平均は [paired_bootstrap_image_mean.csv](analysis/final_20261107/paired_bootstrap_image_mean.csv) に保存した。

| Pool指標 | E3−E1 (95%区間) | E2−E1 (95%区間) | E3−E0 (95%区間) | E2−E0 (95%区間) |
|---|---|---|---|---|
| GT差≥20cm・14px局所MAE (m) | −0.004648 [−0.005098, −0.004205] | −0.002301 [−0.002574, −0.002029] | −0.005870 [−0.006399, −0.005334] | −0.003522 [−0.003892, −0.003156] |
| 全14px局所MAE (m) | −0.000903 [−0.001046, −0.000762] | −0.000437 [−0.000523, −0.000353] | −0.001041 [−0.001186, −0.000899] | −0.000575 [−0.000664, −0.000490] |
| 絶対RMSE (m) | +0.000084 [−0.012727, +0.012215] | +0.003636 [−0.008366, +0.017218] | −0.026318 [−0.042386, −0.010381] | −0.022765 [−0.041550, −0.003664] |
| 絶対MAE (m) | +0.002808 [−0.005131, +0.010973] | +0.002455 [−0.006572, +0.012772] | −0.021061 [−0.034903, −0.007468] | −0.021414 [−0.037816, −0.004994] |
| AbsRel | +0.000158 [−0.000209, +0.000522] | −0.000042 [−0.000469, +0.000415] | −0.001169 [−0.001939, −0.000421] | −0.001369 [−0.002263, −0.000495] |
| GT差≥20cm・14px振幅比 | −0.001058 [−0.001919, −0.000104] | −0.001422 [−0.001876, −0.000915] | +0.003504 [+0.002475, +0.004585] | +0.003140 [+0.002462, +0.003848] |

局所Huber群の差分MAE区間は0を下回る一方、振幅比のE3−E1区間は負であり、E3の比率はE1よりわずかに1から離れる。絶対RMSE・MAEのE3−E1区間は0を含む。これはDay047の128画像を標本単位とした条件付き区間であり、撮影日単位の独立反復ではない。また学習seedは1種類であり、epoch/checkpoint選択に同じvalidationを使ったため選択不確実性も含まれない。13指標・4対比を調べた結果は多重比較を補正していない。主要評価指標は事前指定であるが、他の指標の区間を確証的検定と解釈してはならない。

保存済み結果から復元できる範囲では、14/28/56/107/214 pxの局所MAE・RMSE・振幅比・符号一致率、GT差分階級、作物・視点、画像端距離、GT深度帯、GT差分符号の条件別表を作成した。局所Spearman相関・回帰傾き、および各条件別の画像bootstrapは集計要素が不足するため算出していない。代表画像の深度マップ比較は事後的な画像選択を避けるため作成していない。改善・悪化例を画像選択で示す代わりに、全条件表と画像ごとの差分件数を公開した。

validationはDay047の単一撮影日に由来する。同日の画像間依存を考慮すると、画像単位bootstrapを行った場合も未知撮影日への汎化を示すものではない。Independent test319は未使用である。

## 12. 再現性・provenance

| 対象 | SHA256 / 識別情報 |
|---|---|
| 初期 model.safetensors | 52769e2e95704d0862e708e8772195b861c647dd4812058eaf698a44305c3bcb |
| 初期 config.json | 3c6ab05881e6a76edbfd33ff393fdfb082582084e9262826bb1ac0fd2b1dac7c |
| 実験設定 config.json | 1692ebf048a99b5624c69330b23ad67954aac7ab047e024c83da54a72b308127 |
| train manifest | 13ee3432515250eff50eeb5238dfc178c8853239074eb8cbbea2b510272b0602 |
| validation manifest | 96c4eab0eb0075ac141ff9b17c22a45734f242f7cc895f8eaf064a4f14cc871e |
| fixed pair manifest（実ファイル） | bcb56eab03bec4a44f0626da0f4814876fae67774e536e00cd9f1ad16b1f5728 |
| E1 epoch 5 checkpoint | d912edb89b4d42cfeabef5051d97e756210019414116f60e0961084525f3a19c |
| E2 epoch 5 checkpoint | b6875d59fd4ee32bbacce32664c493c885645acec906b074f2f615f27ef892b4 |
| E3 epoch 5 checkpoint | 223b75d1f5d81844bc5702565a8291270a8ae951d6e8e7c12be9f1b3ad12e886 |
| 基準コードcommit | 49bdd188d9caa01cf9bacdf02f7aebec65537fcb |
| 実験ID | local_huber_20261105 |
| GPU実行root | /localdata/s0623003/local_huber_project_20261105 |
| NFS backup | /home1/eng-std/s0623003/gpu4_research_backup_local_huber_20261105_20261105T081949Z |

E1〜E3のepoch 5 checkpoint SHA256は同backupの `SOURCE_SHA256SUMS` と一致する。E0初期checkpointの記録済みSHA256は、保存済みE0 summaryと評価signatureに含まれるが、今回取得したrun backupの `SOURCE_SHA256SUMS` には初期checkpoint本体の項目がない。この差を保ったまま、[checkpoint_sha256.csv](analysis/checkpoint_sha256.csv) にE0の証拠元とE1〜E3のmanifest照合結果を区別して記録した。

設定ファイルの manifest_sha256.fixed_pairs 欄は63文字であり、実ファイルのSHA256（64文字）と一致しない。実行CLIは固定pair manifestをコードで指定し、その実ファイルSHAをrun signatureへ保存し、各画像の固定座標列との一致を評価時に検証している。したがって、実験のペア座標がずれていたとは確認されていないが、設定内の記載誤りはprovenance上の不備である。結果を再現する場合は、設定欄の誤記を修正した別revisionを作る前に、バックアップ内のrun signatureとmanifestを照合する必要がある。既存configは結果後に書き換えていない。

再現コマンドと前処理、resume方法はREADME_JA.mdに記載した。実行環境・詳細なepoch成果・固定標本評価JSONは上記NFSバックアップに保存している。

## 13. 追加解析と成果物

NFSから取得した全15成果ファイルは `DERIVED_RESULTS_SHA256SUMS` と元の `SOURCE_SHA256SUMS` に照合した。最終E0〜E3のB0再評価は、既存固定条件を用い、pool指標が既存run要約と絶対許容差1e−7以内、主要固定ペア数が完全一致することを確認した。E0は従来B0基準値と一致した。検証記録は [pool_reconciliation.csv](analysis/final_20261107/pool_reconciliation.csv)、[analysis_run_metadata.json](analysis/final_20261107/analysis_run_metadata.json)、および `analysis/final_20261107/inputs/` にある検証済み要約・画像別十分統計に保存した。

| 成果物 | 内容 |
|---|---|
| [TABLES_JA.md](analysis/final_20261107/TABLES_JA.md) | 画素間隔・差分階級・作物・視点・端距離・深度・符号の詳細表 |
| [condition_comparison_all.csv](analysis/final_20261107/condition_comparison_all.csv) | 43条件層のE0〜E3 MAE/RMSE/振幅比/符号一致率、画像数・ペア数 |
| [paired_bootstrap_pooled.csv](analysis/final_20261107/paired_bootstrap_pooled.csv) | pool指標の対応付き画像bootstrap 10,000回 |
| [paired_bootstrap_image_mean.csv](analysis/final_20261107/paired_bootstrap_image_mean.csv) | 画像平均指標の対応付きbootstrap。pool推定量とは別集計 |
| [provenance_final.json](analysis/final_20261107/provenance_final.json) | 入力・checkpoint・分割・bootstrap・既知のprovenance差分 |
| [ANALYSIS_OUTPUT_SHA256SUMS](analysis/final_20261107/ANALYSIS_OUTPUT_SHA256SUMS) | 最終解析成果物と公開用入力一式のSHA256 |
| [paired_bootstrap_key_metrics.png](analysis/final_20261107/paired_bootstrap_key_metrics.png) | 絶対・局所主要指標のpool差と95%区間 |
| [main_metrics_comparison.png](analysis/final_20261107/main_metrics_comparison.png) | 絶対RMSE、全14px局所MAE、主要大差分MAE |
| [distance_large_pair_mae.png](analysis/final_20261107/distance_large_pair_mae.png) | 大差分標本の距離別局所MAE |
| [amplitude_ratio_by_distance.png](analysis/final_20261107/amplitude_ratio_by_distance.png) | 大差分標本の距離別振幅比 |
| [gt_delta_bins.png](analysis/final_20261107/gt_delta_bins.png) | GT差分階級別MAE |
| [crop_view_large_pair_mae.png](analysis/final_20261107/crop_view_large_pair_mae.png) | 作物・視点別の大差分MAE |
| [condition_families_e3_e2_vs_e1.png](analysis/final_20261107/condition_families_e3_e2_vs_e1.png) | 端距離・深度・符号別にE1からの差を全表示 |
| [rmse_local_tradeoff.png](analysis/final_20261107/rmse_local_tradeoff.png) | 絶対RMSEと局所MAEのトレードオフ |

分析の再実行は、検証済み入力を使い、学習・推論を行わずCPU上で以下のように実行する。

```powershell
$repo = 'C:\Users\81902\Documents\Codex\2026-09-20\depth-anything-v2-python-uav-rgb'
$exp = Join-Path $repo 'experiments\local_huber_20261105'
& (Join-Path $repo '.venv\Scripts\python.exe') (Join-Path $exp 'analysis\finalize_analysis.py') `
  --per-image (Join-Path $exp 'analysis\final_20261107\inputs\per_image_primary.csv') `
  --conditions (Join-Path $exp 'analysis\final_20261107\inputs\compact_condition_metrics.csv') `
  --summary (Join-Path $exp 'analysis\summary.json') `
  --image-counts (Join-Path $exp 'analysis\final_20261107\inputs\condition_image_counts.csv') `
  --output (Join-Path $exp 'analysis\final_20261107') --seed 42 --repeats 10000
& (Join-Path $repo '.venv\Scripts\python.exe') (Join-Path $exp 'analysis\build_analysis_tables.py')
```

Independent test319は使用していない。局所深度差の小幅な改善は観察されたが、5%成功目標と振幅再現の改善は達成していない。本実験データはカメラZ深度であり、地表と作物頂部の対応づけや実測稲高を含まないため、深度誤差を稲高誤差へ直接読み替えることはできない。次段階では別撮影日・複数学習seedの再検証、GT差分の計測不確かさを含む感度評価、地表基準を伴う作物高の独立測定が必要である。これらは今後の提案であり、本実験では未実施である。

## 14. データ・モデルの出典

- UAV3DCropデータセット: Zhou et al., “UAV3DCrop: Benchmarking 3D Reconstruction in Repeated Multi-Angle UAV Crop Surveys,” 2026, [arXiv:2608.06404](https://arxiv.org/abs/2608.06404). 公開プロジェクトおよびデータアクセスは[UAV3DCrop project page](https://link-dev.github.io/UAV3DCrop/)を参照する。本実験で使用した画像分割とmanifestのSHA256は第12節のprovenanceに記録した。
- Depth Anything V2: Yang et al., “Depth Anything V2,” 2024, [arXiv:2406.09414](https://arxiv.org/abs/2406.09414). Metric Outdoor Smallのモデル構成・公式実装・pretrained metric model informationは[official Metric Depth repository](https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth)を参照する。本実験の初期checkpointは研究室で保管していたepoch 20 fine-tuned重みである。

\r\n