# 実験計画：C0追加15 epoch

## 目的

09-20正本の研究を継続し、既存C0 epoch5 checkpointからepoch6〜20の15 epochを追加する。新しいCodexプロジェクトや独立したリポジトリは作らず、既存リポジトリ内の実験として記録する。

## 固定した仮説

- H1：追加学習で絶対深度RMSEがepoch5-P0より低下する。
- H2：追加学習で局所深度差MAEが改善する。
- H3：epoch6〜20の途中でValidation性能が停滞または悪化する可能性がある。
- H4：epoch20-B3がepoch5-B3を上回る可能性がある。

学習量は15 epochに固定する。Validation結果を理由に学習率、損失、crop条件を変更しない。主判定checkpointはepoch20である。

## 初期checkpointと再開方針

C0 epoch5の実体SHA256は実験開始前に再計算する。指定値は
2fd190727c7a5124c56c1e0ea2dd8cb09c6cb62a0f92b34fce97ad394281b35a
である。epoch=5、global_step=960、model key数287を検査し、modelはstrict loadする。optimizer moment、GradScaler、RNGはcheckpointから復元する。既存schedulerは5 epoch計画の終端で学習率が0になっているため、stateを継続しない。AdamW momentを保ったままLRだけをencoder 1e-6、neck/head 1e-5へ設定し、successful optimizer update数2880のPolynomialLR(power=0.9)を新規作成する。

## 学習条件

C0の既存方式を維持する。Train384、1画像2crop/epoch、入力518x518、元画像を短辺518pxへbicubic縮小してから518x518 cropを取得し、GTはnearestで縮小する。SiLogのみ、AdamW、batch 1、gradient accumulation 4、seed 42、FP16 GradScaler、GT有効範囲0.001〜65m、出力上限80mとする。AMP skipではoptimizerとschedulerを進めない。

期待量は768 crop、192 update機会/epoch、追加11,520 crop、2,880 update機会である。実際のsuccessful update数とskip数を別に記録する。

## GPU利用ゲート

GPGPU4を第一候補にする。学習開始直前にwho、w、nvidia-smi、df、gpu_work実体と使用量、Singularity版を確認する。GPU利用者との共有利用許可をユーザーから得たため、別ユーザーのcompute processが最大1件、GPU使用率5%以下、空きVRAM5GiB以上の場合に限り、許可記録と直前preflightを保存してsmoke testを行う。process集合が実行中に変わった場合は停止する。別ユーザーのprocessを停止・変更しない。GPGPU3へ独断で切り替えない。

2026-11-26 21:42 JSTの確認では、GPGPU4に別ユーザーのPID 3894（`/mnt/.venv/bin/python`、約2,030MiB）が存在し、GPU使用率0%、空きVRAM6,066MiBであった。ユーザーから当該GPU利用者の許可取得済みとの連絡があり、共有利用条件を満たす場合に限って開始する。実行中は許可済みPID集合の不変、当該processのVRAM増加が基準+256MiB以内、全GPU空きVRAM1GiB以上を定期確認し、外れれば自分の処理を停止する。

### checkpoint保管方針

GPUローカルの容量確保のため、既存NFSバックアップの実ファイルとSHA256 manifestが一致することを確認したうえで、旧C0/F1/F2・E1/E2/E3の不要なepoch checkpoint 28件と、完全一致するC0/E3 epoch5重複2件をGPUローカルから整理した。今回の継続学習に必要なC0 epoch5、最終比較に必要なE3 epoch5は正本だけを残す。B系列checkpoint、元Metric Outdoor checkpoint、NFSバックアップ、学習ログ、評価結果は削除対象外である。整理対象のパス・hash・サイズと検証ログを`audit/checkpoint_cleanup_manifest.tsv`および関連ログに記録する。学習中に生成するepoch6〜20 checkpointは毎epoch保存し、NFSへ検証退避する。

## 評価

epoch6〜20は各epoch終了後にValidation128をP0評価する。maskはfinite and 0 < depth <= 65m、固定pair manifestを使用する。RMSE、MAE、AbsRel、Bias、Pearson、全14px局所MAE、高差分14px局所MAE/RMSE、振幅比、符号一致率を保存する。epoch20では既存B3実装を固定条件のまま評価する。

独立test319は使わない。画像単位paired bootstrapは10,000回、seed42とし、pooledと画像平均を区別する。

## 成功条件・停止条件

学習はepoch20まで行う。非有限loss/gradient、繰り返すOOM、checkpoint破損、disk safety gate失敗、GPU他ユーザー利用のpreflight blockがあれば安全に停止し、最後の正常checkpointと理由を記録する。最良epochを主結果として選び直さない。
