# 状態

09-20正本を引き継いだC0追加学習を実施中である。正式実行は2026-11-26 22:28 JSTに開始した。2026-11-27 04:56 JST時点でepoch 6〜16の学習・Validation128・checkpoint保存が確認できており、epoch 17の開始記録もあるが、その後の実行状態はSSH切断後に未確認である。FortiClientの再接続を開始し、Microsoft SAML認証画面でユーザーのサインイン待ちである。新しいプロジェクトは作成していない。

## 既知の確認結果

- PR #8の最新headは4e87f047ddf9bd43144fdc9d9754e3abbf28b716。
- GPGPU4へのSSHは一度切断された。FortiClientのVPNは未接続であり、同じCodex端末から再確認するためMicrosoft SAML認証を待っている。
- C0 epoch5実体のSHA256は指定値と一致した。
- checkpoint内部にepoch 5、global_step 960、model 287 keys、optimizer 2 groups、scheduler/scaler/RNG/history/signatureが存在する。モデルtensorはfinite。現時点のoptimizer LRは0であり、scheduler再設定が必要である。
- C0/F1/F2 epoch1〜5とE1/E2/E3 epoch1〜5のNFSバックアップ各15件を既存SHA256 manifestで照合し、全件一致した。GPUローカルの同一30ファイルも各manifestのhashと一致した。
- GPUローカルから旧checkpoint 28件と、best-model実験フォルダーにあるC0/E3 epoch5の完全重複2件を削除した。削除対象のパス・SHA256・サイズ、NFS側の照合ログを`~/gpu_work/c0_extended_training_20261126/audit/`へ記録した。
- 保持した学習checkpointはC0 epoch5（継続開始用）とE3 epoch5（比較用）である。B系列、元Metric Outdoorモデル、NFSバックアップ、評価結果、学習ログには触れていない。
- 削除後に`du -sh /localdata/s0623003`で再計測した使用量は45GiBである。削除前の約53GiBから約8GiB減少した。`df -h /localdata`ではパーティション全体の空きが107GiBである。`~/gpu_work`はローカル領域へのシンボリックリンクであるため、容量はリンク先`/localdata/s0623003`で確認した。
- ユーザーからGPU利用者の共有利用許可取得済みとの連絡を受けた。21:42 JSTのGPGPU4確認では別ユーザーのPython process（PID 3894、約2,030MiB）、GPU使用率0%、空きVRAM6,066MiBであった。processには触れず、開始直前のfresh preflightでPID集合不変・空きVRAM5GiB以上・使用率5%以下を再確認する。実行中はPID集合維持、許可済みprocessのメモリ増加+256MiB以内、空きVRAM1GiB以上を監視し、条件外なら自分の処理を停止する。1バッチsmoke test後に正式学習へ進む。

## 進捗

- [x] PR #8の最新状態を確認
- [x] 正本作業ツリーを保護
- [x] C0 epoch5 SHA256と内部状態を確認
- [x] 旧checkpointのNFS・ローカルSHA256照合とGPUローカル整理
- [x] 継続学習スクリプトと単体テストの追加、構文検査
- [x] 継続学習ロジックの既存単体テスト11件（すべて成功）
- [x] epoch内進捗を`progress.json`へ定期的に記録する処理を追加
- [x] 進捗記録を含む更新後の12件単体テスト（GPU用コンテナ内で全件成功）
- [x] epoch5 checkpoint strict load・GPU 1更新smoke test
- [x] optimizer状態を含む中断・再開一致テスト（小型単体ケース）
- [x] 開始直前preflightとユーザー領域50GiB予算の確認
- [ ] epoch6〜20学習と各epoch Validation（epoch6〜16完了、epoch17以降の実行状態はSSH再接続待ち）
- [ ] epoch20 B3、bootstrap、図表
- [ ] NFS退避とハッシュ照合
- [ ] PR作成・公開

正式学習の最初の起動試行（2026-11-26 22:17 JST）は、設定に記載した固定ペアmanifestのパスが誤っていたため、初期入力検証で終了した。GPU forward・学習更新・Validation・checkpoint保存はいずれも始まっていない。出力runディレクトリは作成されず、epoch5 checkpointと他ユーザーのGPU processにも変更はない。固定manifestの正しい正本は`experiments/multiscale_tile_inference_20261016/pair_manifest.json`であり、SHA256は`bcb56eab03bec4a44f0626da0f4814876fae67774e536e00cd9f1ad16b1f5728`、画像数は128である。設定をこのパスへ訂正し、GPU側ではGit追跡ファイルから同一内容を小容量の入力領域へ取り出してハッシュ一致を確認してから、fresh preflightで再起動する。

共有利用許可はユーザーの報告に基づき記録済みである。正式学習は進行中である。完了していないepoch、最終B3、bootstrap、NFS退避、PR公開は完了扱いにしない。NFS退避先は変更しておらず、新規実験の成果バックアップは未実施である。epoch5 checkpointの実測サイズ約284MiBを基準に、epoch5保護コピーとepoch6〜20の計16個は約4.4GiB、評価統計等の予備を加えてユーザー領域50GiB以内に収める計画である。

単体テスト後の最初のGPU smoke testは、計算開始前のGPU安全ガードで停止した。preflightの使用率0%をPythonのtruthinessで欠損値と誤判定し、100%として扱う不具合である。GPU forwardやoptimizer更新は開始されておらず、他ユーザーの処理・元checkpointに変更はない。判定ロジックと回帰テストを追加した後、fresh preflightで再試行する。

GPU安全ガードの修正後、次のsmoke testは既存SiLog実装のimport時に停止した。GPU用sparse cloneが固定ソースの`source_full_v1/losses.py`を含んでいなかったためである。GPU演算に入る前のimport失敗であり、共有GPUやcheckpointへの影響はない。必要な固定ソースをsparse checkoutへ追加し、依存コードのSHA256もprovenance対象に加えた。

`--containall`付きコンテナではPID namespaceが分離され、コンテナ内の`os.getpid()`と
vidia-smi`が表示するホストPIDが一致しないことを確認した。そこで正式実行とsmoke testには`--contain --cleanenv`を用いる。PID namespaceなしでも`/localdata`だけをbindし、コンテナ内外のPID一致を実測した。共有GPU監視は自分のGPU processを除外し、既存利用者PIDを追跡できる。

最終smoke test（2026-11-26 22:13 JST）は成功した。epoch5 checkpoint（SHA256 `2fd190727c7a5124c56c1e0ea2dd8cb09c6cb62a0f92b34fce97ad394281b35a`）をstrict loadし、model 287 keys、missing/unexpected/shape mismatch各0を確認した。epoch/global stepは5/960、optimizer state・GradScaler・RNGが有効であることを監査した。Train384から4 cropを処理し、FP16 optimizer updateを1回成功、AMP skip 0、有限なSiLog損失を確認した。これは一時モデル上のsmoke updateであり、重みを保存していない。元checkpointのSHA256は実行後も一致し、GPUは使用率0%、空きVRAM6,066MiB、既存PID 3894の約2,030MiBへ戻った。ピークGPU割当量は981,260,288 bytes、実行時間は約11.7秒である。監査JSONとlogはGPUローカルの`audit/smoke_test_20261126T2213.{json,log}`に保存済みである。

## 正式実行の進捗（2026-11-27 04:41 JST）

- 実行先はGPGPU4、ローカル出力は`/localdata/s0623003/c0_extended_training_20261126/run_epoch6_20_20261126T222819/`である。実行コードrevisionは`e33ec02f772f4bb584403c7dbc4d736c9e2fd66a`で、起動ログは`audit/formal_training_20261126T222819.log`にある。
- 2026-11-26 22:17 JSTの起動試行は固定pair manifestのパス検証で終了した。正式runは修正後の別出力先で22:28 JSTに開始しており、最初の試行の部分成果を再利用していない。
- epoch 6〜8は各768 crop、192 successful optimizer updates、AMP skip 0で完了した。global stepは1152、1344、1536の順で進んだ。epoch 8の学習SiLogは0.0109799051である。

| Epoch | Validation RMSE (m) | 全14px local MAE (m) | 高差分14px local MAE (m) | Checkpoint SHA256 |
|---:|---:|---:|---:|---|
| 6 | 0.807531 | 0.059194 | 0.334683 | `115700ba50ffd3c19af5a6adfedb444a468123f37cf40fe28fe05cb5ebe101ba` |
| 7 | 0.856295 | 0.058810 | 0.335134 | `3f895674e9ddbaf0848c1a8a884f10143c24c0e80830361160c5d35cbfcb089e` |
| 8 | 0.761091 | 0.058860 | 0.334211 | `8c937678b8d4880c29c4d2bd2a326481e45ff8e3aa42a85ab88f4aa46f1b6d56` |
| 9 | 0.735352 | 0.058825 | 0.334096 | `cb0239910076ca033bfc08913edc92a4650f0e7ee60d37072c8fba0cc9c9f244` |
| 10 | 0.762720 | 0.059195 | 0.332658 | `3102f0f601dabd4bf7a2e6136f9cc19925b0fcdc13eea2a7388963335d200ccc` |
| 11 | 0.773296 | 0.058918 | 0.332759 | `03eb02805a1cfd6268137acde80747ed91eeeb4d0ce0db0b8e68187bdfb01123` |
| 12 | 0.743721 | 0.058404 | 0.333518 | `1c8ddec06fe2953b5373a8fe6ccc166c485740c1ec4f67a0d265eb217583cd306` |
| 13 | 0.729365 | 0.058695 | 0.333306 | `fb2caee9fe5839d963d83079f39e79266a177f3dc9bbbd35c364eeb841205101` |
| 14 | 0.726086 | 0.058755 | 0.333166 | `3d16570686abd64d95c932b86e45b4a3377bfc7ea65cfa16e9638f4947aee197` |
| 15 | 0.747917 | 0.058687 | 0.332473 | `24a9f5abe2808f30a0d820a8112a9508a720b58efd2c77fe0d566a964e520d23` |

- epoch 9のValidation128ではRMSE 0.735352 m、全14px局所MAE 0.058825 m、高差分14px局所MAE 0.334096 mであった。epoch5-P0基準（RMSE 0.744069 m、全14px 0.059084 m、高差分14px 0.334913 m）に対し、3指標はいずれも小幅に低下した。大差分局所MAEの改善は0.001 m未満であり、epoch20完了前に結論を確定しない。
- epoch 9の正式評価JSON `validation/epoch_0009_attempt_1795707216235661325/evaluation_C0_P0.json` には局所指標が保存されている。コンソール要約の`validation_high_delta14_mae_m`だけはgroup名の照合不一致で`null`となるが、評価本体の`sampled_local`にある`gt_ge20cm`行は1,911,836 pair、MAE 0.3340955349 m、RMSE 0.5672403714 m、振幅比0.3380612、符号一致率0.8300545である。epoch9 JSON、実checkpointのSHA256、当該評価JSONを照合した。これはログ表示欄だけの問題であり、保存済み評価統計は有効である。ソースの表示欄は正式run完了後に修正し、実行時コードとの差をprovenanceへ記録する。
- epoch 10の正式評価ではRMSE 0.762720 m、全14px局所MAE 0.059195 m、高差分14px局所MAE 0.332658 m（1,911,836 pair）である。epoch 9よりRMSEと全14px指標が悪化し、高差分指標は改善した。epoch 5基準との比較でも指標間の差があり、epoch20を主評価として固定した方針を保つ。
- epoch 11の正式評価ではRMSE 0.773296 m、全14px局所MAE 0.058918 m、高差分14px局所MAE 0.332759 m（1,911,836 pair）であった。epoch 10よりRMSE・高差分MAEが悪化し、全14px MAEは小幅に改善した。評価JSONの値とcheckpoint SHA256を実ファイルで照合済みである。ログ要約欄の高差分MAEはgroup名の不一致で`null`となるが、詳細評価JSONの数値は有効である。
- epoch 12はglobal step 2304で学習を終え、successful update 192回、AMP skip 0回、学習時間1713.95秒であった。Validation128のJSON・画像別CSVとepoch12 checkpointが保存された。評価JSONからRMSE 0.7437205292 m、全14px局所MAE 0.0584038945 m、高差分14px局所MAE 0.3335183408 m（1,911,836 pair）を読み、checkpoint実体のSHA256 `1c8ddec06fe2953b5373a8fe6ccc166c485740c1ec4f67a0d265eb217583cd306`をcheckpoint metadataと照合した。高差分局所RMSEは0.5655663697 m、振幅比0.3381884、符号一致率0.8328460である。epoch12の値はepoch9以来のRMSE改善水準だが、epoch20固定評価前に結論を出さない。epoch 13は完了し、次の段落に詳細を記録した。
- epoch 6〜13は各768 crop、192 successful optimizer updates、AMP skip 0回で完了した。epoch 13の学習時間は1717.25秒である。Validation・checkpoint保存を含めた所要時間は概ね32分/epochである。epoch 13のValidation128はRMSE 0.7293646946 m、全14px局所MAE 0.0586949952 m、高差分14px局所MAE 0.3333063521 m（1,911,836 pair）であった。高差分局所RMSEは0.5660370014 m、振幅比0.3372884、符号一致率0.8331494である。epoch13 checkpoint SHA256 `fb2caee9fe5839d963d83079f39e79266a177f3dc9bbbd35c364eeb841205101`は実ファイルとcheckpoint recordで一致した。epoch13はepoch12よりRMSEと高差分局所MAEが小さく、全14px MAEはやや大きい。epoch20固定評価前に結論を確定しない。epoch14は768 crop、192 successful optimizer update、global step 2688、AMP skip 0回、学習時間1713.63秒で完了した。Train SiLogは0.00910825498である。Validation128（`validation/epoch_0014_attempt_1795717026/evaluation_C0_P0.json`）のMAEは0.5269526850m、RMSE 0.7260863319m、AbsRel 0.0268732309、Bias 0.0458612079m、Pearson 0.9953570620であった。全14px局所MAEは0.0587550138m、高差分14pxは1,911,836 pair、MAE 0.3331655611m、RMSE 0.5654180806m、振幅比0.3385156681、符号一致率0.8336808178である。epoch14 checkpoint SHA256 `3d16570686abd64d95c932b86e45b4a3377bfc7ea65cfa16e9638f4947aee197`は実ファイルとepoch JSON recordで一致した。epoch15は768 crop、192 successful optimizer update、global step 2880、AMP skip 0回、学習時間1711.46秒で完了した。Train SiLogは0.00905594552である。Validation128（`validation/epoch_0015_attempt_17957183457984835/evaluation_C0_P0.json`）のMAEは0.5305756275m、RMSE 0.7479169634m、AbsRel 0.0271026021、Bias 0.0859766013m、Pearson 0.9950118369であった。全14px局所MAEは0.0586872653m、高差分14pxは1,911,836 pair、MAE 0.3324728613m、RMSE 0.5651623459m、振幅比0.3417438006、符号一致率0.8327387140である。epoch15 checkpoint SHA256 `24a9f5abe2808f30a0d820a8112a9508a720b58efd2c77fe0d566a964e520d23`は実ファイルとepoch JSON recordで一致した。epoch16の学習は768 crop、192 successful optimizer update、global step 3072、AMP skip 0回、学習時間1,715.10秒で完了した。Validation128の`epoch_0016.json`は生成され、MAE 0.5257774775 m、RMSE 0.7256868971 m、AbsRel 0.0268346654、Bias 0.0939900239 m、Pearson 0.9953526815、全14px局所MAE 0.0585208952 m、大差分14px（1,911,836ペア）MAE 0.3326851547 m、RMSE 0.5655578199 m、振幅比0.3398288765、符号一致率0.8332419726である。記録上epoch17の開始も確認した。epoch16 checkpointは`epoch_0016.pt`が存在しJSON内にcheckpoint SHA256欄があるが、SHA256実計算を行う前にSSHが切断されたため、実体との照合は未完了である。FortiClientのVPN再接続にはMicrosoft SAML認証が必要である。認証ウィンドウは時間切れで閉じ、04:41 JST時点でVPNは未接続である。epoch17以降の進捗は未確認である。

## 解析コードの追加検証

- 画像単位paired bootstrap、固定画像ID照合、高差分ペアがない画像の未定義処理を確認する4件の回帰テストを`test_analyze_extended_results.py`に追加した。
- Windows標準Pythonにはnumpyがなかったため、作業用の隔離venvにnumpy 2.4.6を導入してテストを実行し、4件すべて成功した。解析・作図・テストスクリプト3件の`py_compile`と`git diff --check`も成功した。
- 実験用Singularity内でのテスト実行は未確認であり、GPU再接続後に実行する。
