# C0追加15 epoch実験報告

## 実験状態

本報告書は09-20正本リポジトリ内の継続実験を記録する。新規学習ではなく、C0 epoch5 checkpointから15 epochを追加する計画である。学習開始条件がそろい、実測が完了するまでは数値結果を確定しない。

## 学習前に確認した事実

GPGPU4のC0 epoch5 checkpointはSHA256
2fd190727c7a5124c56c1e0ea2dd8cb09c6cb62a0f92b34fce97ad394281b35a
と一致した。checkpointにはepoch 5、global_step 960、model state 287 keysとoptimizer、scheduler、scaler、RNG、history、signatureが含まれ、model tensorはfiniteであった。保存LRはscheduler終端で0であり、そのままの再開では学習が進まないため、optimizer moment等を保持しつつ、追加フェーズ用LRとPolynomialLRを設定する。

checkpoint整理後、`du -sh /localdata/s0623003`でユーザー領域全体を再計測した結果は45GiBである。削除前の約53GiBから約8GiB減少した。`df -h /localdata`ではパーティション全体に107GiBの空きがある。2026-11-26 21:42 JSTのGPU4確認では使用率0%、空きVRAM6,066MiBであり、別ユーザーのPython process（PID 3894）が約2,030MiBを保持していた。ユーザーからGPU利用者の共有利用許可取得済みとの連絡を受けたため、直前preflightでPID集合不変・空きVRAM5GiB以上・使用率5%以下を満たす場合に限り、1バッチsmoke test後に学習を始める。別ユーザーのプロセスは停止・変更しない。追加checkpointはepoch5保護コピーを含め16個で約4.4GiBと見積もり、評価統計等を含めユーザー領域50GiB以内に収まることを開始前に確認する。

## 評価基準

既存C0 epoch5-P0基準は絶対RMSE 0.744069m、全14px局所MAE 0.059084m、高差分14px局所MAE 0.334913mである。epoch5-B3基準はそれぞれ0.743945m、0.063323m、0.326687mである。epoch20を主checkpointとしてValidation128で評価し、学習曲線中の中間値から最良条件を選び直さない。

## 結果

未実施。GPU preflight・訓練・Validationが完了するまで結果欄を更新しない。
