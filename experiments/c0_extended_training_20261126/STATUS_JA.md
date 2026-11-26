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
- ユーザーからGPU利用者の共有利用許可取得済みとの連絡を受けた。21:42 JSTのGPGPU4確認では別ユーザーのPython process（PID 3894、約2,030MiB）、GPU使用率0%、空きVRAM6,066MiBであった。processには触れず、開始直前のfresh preflightでPID集合不変・空きVRAM5GiB以上・使用率5%以下を再確認し、1バッチsmoke test後に正式学習へ進む。

## 進捗

- [x] PR #8の最新状態を確認
- [x] 正本作業ツリーを保護
- [x] C0 epoch5 SHA256と内部状態を確認
- [x] 旧checkpointのNFS・ローカルSHA256照合とGPUローカル整理
- [x] 継続学習スクリプトと単体テストの追加、構文検査
- [ ] PyTorch strict model load・resumeテスト・GPU smoke test
- [ ] 開始直前preflightとユーザー領域50GiB予算の確認
- [ ] epoch6〜20学習と各epoch Validation
- [ ] epoch20 B3、bootstrap、図表
- [ ] NFS退避とハッシュ照合
- [ ] PR作成・公開

共有利用許可は記録済みであるが、smoke testと正式学習はまだ開始していない。実行していない検証・結果は完了扱いにしない。NFS退避先は変更しておらず、新規実験の成果バックアップは未実施である。epoch5 checkpointの実測サイズ約284MiBを基準に、epoch5保護コピーとepoch6〜20の計16個は約4.4GiB、評価統計等の予備を加えてユーザー領域50GiB以内に収める計画である。直前の容量計測で上限を超える予測になれば開始しない。
