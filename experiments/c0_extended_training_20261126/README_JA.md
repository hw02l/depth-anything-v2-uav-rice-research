# C0追加15 epoch（09-20研究の継続）

C0の既存epoch5モデルからepoch6〜20を追加し、学習期間延長の影響を検証する実験である。新規Codexプロジェクトではなく、09-20正本リポジトリ内の一実験として管理する。

- 計画：PLAN_JA.md
- 固定条件：config.json
- 現在の進捗：STATUS_JA.md
- 結果と限界：report_ja.md
- epoch 1〜20の学習曲線表・図の生成：plot_learning_curves.py
- epoch5/20のP0/B3条件間bootstrap集計：analyze_extended_results.py

C0 epoch5 checkpointのSHA256はconfig.json記載値と照合し、modelをstrict loadする。AdamW moment、GradScaler、RNGを復元し、既存5 epoch schedulerは継続せず、追加15 epoch専用PolynomialLRを指定LRで設定する。学習率はValidation結果を見て変更しない。

Validation128のみを使用し、Independent test319は使わない。初期checkpointおよび既存研究出力を上書きしない。学習中は約8回のsuccessful optimizer updateごとに、epoch内サンプル数、更新数、AMP skip数、経過時間、累積global stepを`progress.json`へ原子的に記録する。epoch終了時はValidation中・checkpoint保存済みの状態も記録する。

## checkpoint整理

容量確保のため、既存NFSバックアップのmanifestと実ファイルが一致することを確認した旧C0/F1/F2・E1/E2/E3のepoch checkpoint 28件と、C0/E3 epoch5の完全重複コピー2件をGPUローカルから整理した。C0 epoch5とE3 epoch5の正本は残している。B系列checkpoint、Metric Outdoor初期モデル、NFSバックアップ、既存評価結果は変更していない。GPU上の監査表は`~/gpu_work/c0_extended_training_20261126/audit/checkpoint_cleanup_manifest.tsv`、削除記録は同階層の`checkpoint_cleanup_receipt.txt`にある。

## 検証状態

継続学習、GPU smoke test、最終B3評価、統計解析、作図スクリプトと単体テストを追加した。継続学習の12件の単体テスト、厳密なepoch5重み・optimizer/scaler/RNG復元、GPU上の一更新smoke testは成功済みである。追加した解析回帰テスト4件はWindows隔離venv（numpy 2.4.6）で全件成功し、関連スクリプトの構文検査と差分空白検査も成功した。解析テストのSingularity内実行は未確認である。C0 epoch5 checkpointのSHA256は正式学習開始前後で一致し、固定pair manifestは`experiments/multiscale_tile_inference_20261016/pair_manifest.json`である。GPGPU4ユーザー領域は整理後約45GiB、`/localdata`全体の空きは約107GiBであった。他ユーザーのプロセスには触れていない。初回起動試行は固定manifestのパス誤りで入力検証前に終了したが、修正後に正式runを2026-11-26 22:28 JSTに開始した。epoch6〜16は各768 crop、192 successful update、AMP skip 0で完了し、epoch16 P0のRMSEは0.725687m、全14px局所MAEは0.058521m、高差分14px局所MAEは0.332685mである。epoch17の開始を確認した後にSSH接続がリセットした。FortiClientはMicrosoft SAML認証待ちで、現在VPN未接続のため、epoch17以降の進捗とepoch16 checkpointの実SHA256は未確認である。

`plot_learning_curves.py`は保存済みのepoch5 P0評価、C0 epoch1〜5の学習履歴、継続runのepoch JSONを結合し、CSVとSVGの曲線を出力する。保存されていないepoch1〜4のValidation指標は補間せず欠損のまま扱う。epoch6で学習率スケジュールを再設定した境界を図に表示する。合成epochデータによる動作確認で20行のCSVと8図の生成を確認し、SVGをXMLとして読み取れることも確認した。実測結果の図表生成はepoch20までの成果をローカルへ保全した後に実施する。

`analyze_extended_results.py`はepoch20-P0、epoch20-B3を、固定されたepoch5-P0、epoch5-B3、epoch20-B3（初期Metric Outdoor）、E3-epoch5-B3と画像IDで対応付ける。絶対RMSE、全14px局所MAE、高差分14px局所MAEについてpooledと画像平均を分け、画像単位bootstrapを10,000回、seed 42で計算する。評価JSON/run_complete.jsonと十分統計から再計算した値の一致を検査する。
