# 状態

実験条件と継続学習コードを準備中である。正式な追加学習はまだ開始していない。09-20正本を引き継いでおり、新しいプロジェクトは作成していない。

## 既知の確認結果

- PR #8の最新headは4e87f047ddf9bd43144fdc9d9754e3abbf28b716。
- GPGPU4の既存Codex端末は接続済み。
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
- [ ] 進捗記録を含む更新後の12件単体テスト
- [x] epoch5 checkpoint strict load・GPU 1更新smoke test
- [x] optimizer状態を含む中断・再開一致テスト（小型単体ケース）
- [ ] 開始直前preflightとユーザー領域50GiB予算の確認
- [ ] epoch6〜20学習と各epoch Validation
- [ ] epoch20 B3、bootstrap、図表
- [ ] NFS退避とハッシュ照合
- [ ] PR作成・公開

正式学習の最初の起動試行（2026-11-26 22:17 JST）は、設定に記載した固定ペアmanifestのパスが誤っていたため、初期入力検証で終了した。GPU forward・学習更新・Validation・checkpoint保存はいずれも始まっていない。出力runディレクトリは作成されず、epoch5 checkpointと他ユーザーのGPU processにも変更はない。固定manifestの正しい正本は`experiments/multiscale_tile_inference_20261016/pair_manifest.json`であり、SHA256は`bcb56eab03bec4a44f0626da0f4814876fae67774e536e00cd9f1ad16b1f5728`、画像数は128である。設定をこのパスへ訂正し、GPU側ではGit追跡ファイルから同一内容を小容量の入力領域へ取り出してハッシュ一致を確認してから、fresh preflightで再起動する。

共有利用許可はユーザーの報告に基づき記録済みである。正式学習はまだ開始していない。実行していない検証・結果は完了扱いにしない。NFS退避先は変更しておらず、新規実験の成果バックアップは未実施である。epoch5 checkpointの実測サイズ約284MiBを基準に、epoch5保護コピーとepoch6〜20の計16個は約4.4GiB、評価統計等の予備を加えてユーザー領域50GiB以内に収める計画である。直前の容量計測で上限を超える予測になれば開始しない。

単体テスト後の最初のGPU smoke testは、計算開始前のGPU安全ガードで停止した。preflightの使用率0%をPythonのtruthinessで欠損値と誤判定し、100%として扱う不具合である。GPU forwardやoptimizer更新は開始されておらず、他ユーザーの処理・元checkpointに変更はない。判定ロジックと回帰テストを追加した後、fresh preflightで再試行する。

GPU安全ガードの修正後、次のsmoke testは既存SiLog実装のimport時に停止した。GPU用sparse cloneが固定ソースの`source_full_v1/losses.py`を含んでいなかったためである。GPU演算に入る前のimport失敗であり、共有GPUやcheckpointへの影響はない。必要な固定ソースをsparse checkoutへ追加し、依存コードのSHA256もprovenance対象に加えた。

`--containall`付きコンテナではPID namespaceが分離され、コンテナ内の`os.getpid()`と`nvidia-smi`が表示するホストPIDが一致しないことを確認した。そこで正式実行とsmoke testには`--contain --cleanenv`を用いる。PID namespaceなしでも`/localdata`だけをbindし、コンテナ内外のPID一致を実測した。共有GPU監視は自分のGPU processを除外し、既存利用者PIDを追跡できる。

最終smoke test（2026-11-26 22:13 JST）は成功した。epoch5 checkpoint（SHA256 `2fd190727c7a5124c56c1e0ea2dd8cb09c6cb62a0f92b34fce97ad394281b35a`）をstrict loadし、model 287 keys、missing/unexpected/shape mismatch各0を確認した。epoch/global stepは5/960、optimizer state・GradScaler・RNGが有効であることを監査した。Train384から4 cropを処理し、FP16 optimizer updateを1回成功、AMP skip 0、有限なSiLog損失を確認した。これは一時モデル上のsmoke updateであり、重みを保存していない。元checkpointのSHA256は実行後も一致し、GPUは使用率0%、空きVRAM6,066MiB、既存PID 3894の約2,030MiBへ戻った。ピークGPU割当量は981,260,288 bytes、実行時間は約11.7秒である。監査JSONとlogはGPUローカルの`audit/smoke_test_20261126T2213.{json,log}`に保存済みである。
