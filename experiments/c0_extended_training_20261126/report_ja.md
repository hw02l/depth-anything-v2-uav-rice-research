# C0追加15 epoch実験報告（進行中）

## 実験状態

本報告書は09-20正本リポジトリ内の継続実験を記録する。新規モデル学習ではなく、C0 epoch5 checkpointから15 epoch（epoch6〜20）を追加する実験である。主評価checkpointは事前固定したepoch20であり、中間epochの数値から最終結論を決めない。2026-11-27 04:56 JST時点でepoch6〜16の学習・Validation128・checkpoint保存が確認でき、epoch17の開始記録もある。その後SSH接続がリセットされ、epoch17以降の状態とcheckpoint実体のSHA256照合は未確認である。FortiClientのVPNは未接続で、再接続後に必要なMicrosoft SAML認証画面が表示されており、ユーザーのサインイン待ちである。以下は検証済みepochの中間値であり、最終的な統計比較・B3評価を含まない。

## 学習前に確認した事実

GPGPU4のC0 epoch5 checkpointはSHA256
2fd190727c7a5124c56c1e0ea2dd8cb09c6cb62a0f92b34fce97ad394281b35a
と一致した。checkpointにはepoch 5、global_step 960、model state 287 keysとoptimizer、scheduler、scaler、RNG、history、signatureが含まれ、model tensorはfiniteであった。保存LRはscheduler終端で0であり、そのままの再開では学習が進まないため、optimizer moment等を保持しつつ、追加フェーズ用LRとPolynomialLRを設定する。

checkpoint整理後、`du -sh /localdata/s0623003`でユーザー領域全体を再計測した結果は45GiBである。削除前の約53GiBから約8GiB減少した。`df -h /localdata`ではパーティション全体に107GiBの空きがある。2026-11-26 21:42 JSTのGPU4確認では使用率0%、空きVRAM6,066MiBであり、別ユーザーのPython process（PID 3894）が約2,030MiBを保持していた。ユーザーからGPU利用者の共有利用許可取得済みとの連絡を受けたため、直前preflightでPID集合不変・空きVRAM5GiB以上・使用率5%以下を満たす場合に限り、1バッチsmoke test後に学習を始める。別ユーザーのプロセスは停止・変更しない。追加checkpointはepoch5保護コピーを含め16個で約4.4GiBと見積もり、評価統計等を含めユーザー領域50GiB以内に収まることを開始前に確認する。

## 評価基準

既存C0 epoch5-P0基準は絶対RMSE 0.744069m、全14px局所MAE 0.059084m、高差分14px局所MAE 0.334913mである。epoch5-B3基準はそれぞれ0.743945m、0.063323m、0.326687mである。epoch20を主checkpointとしてValidation128で評価し、学習曲線中の中間値から最良条件を選び直さない。

## 中間結果

GPGPU4上で2026-11-26 22:28 JSTに正式実行を開始した。epoch6〜16は各768 crop、successful optimizer update 192回、AMP skip 0回で完了し、各epochのValidation128 P0評価とcheckpoint保存が行われた。epoch16の学習時間は1715.10秒（約28分35秒）である。学習とValidation・checkpoint処理を合わせた所要時間は、先行epochの実測から約32分/epochである。他ユーザーのプロセスには触れていない。epoch16完了後のSSH接続リセットとVPN再認証により、epoch17以降の実行状態は確認待ちである。

| Epoch | Validation RMSE (m) | 全14px局所MAE (m) | GT差20cm以上・14px局所MAE (m) | 状態 |
|---:|---:|---:|---:|---|
| 6 | 0.807531 | 0.059194 | 0.334683 | 完了 |
| 7 | 0.856295 | 0.058810 | 0.335134 | 完了 |
| 8 | 0.761091 | 0.058860 | 0.334211 | 完了 |
| 9 | 0.735352 | 0.058825 | 0.334096 | 完了 |
| 10 | 0.762720 | 0.059195 | 0.332658 | 完了 |
| 11 | 0.773296 | 0.058918 | 0.332759 | 完了 |
| 12 | 0.743721 | 0.058404 | 0.333518 | 完了 |
| 13 | 0.729365 | 0.058695 | 0.333306 | 完了 |
| 14 | 0.726086 | 0.058755 | 0.333166 | 完了 |
| 15 | 0.747917 | 0.058687 | 0.332473 | 完了 |
| 16 | 0.725687 | 0.058521 | 0.332685 | 完了 |
| 17〜20 | 未測定 | 未測定 | 未測定 | 未確認・未完了 |

epoch6〜16では、RMSEは単調に改善せず、局所指標も指標間で異なる動きを示している。epoch16はRMSE 0.725687 m、全14px MAE 0.058521 m、高差分14px MAE 0.332685 mである。epoch5-P0基準より3指標とも小さいが、高差分局所MAEの差は約0.00223 mにとどまり、最終結論ではない。epoch20のP0およびB3評価、epoch5とのpaired bootstrapを完了してから仮説を判定する。各epochのcheckpoint SHA256、更新回数、AMP skip、詳細指標は[STATUS_JA.md](STATUS_JA.md)とGPU上の実行provenanceで追跡している。

## 最終結果

未確定。epoch20のP0評価と固定条件B3評価、画像単位paired bootstrap（10,000回、seed 42）、図表、バックアップのSHA256照合が完了した後に確定する。Independent test319は使用しない。
