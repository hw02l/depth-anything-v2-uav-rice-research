# 状態

**実験・保全完了、GitHub公開作業中**：C0 epoch5からepoch6〜20の追加学習、Validation128評価、epoch20 P0/B3比較、paired bootstrap、回帰テスト、NFSバックアップ、ローカル保存を完了した。Independent test319は使用していない。C0以外のモデルは再学習していない。

## 完了項目

- [x] C0 epoch5 checkpointをSHA256・内部状態で監査し、元ファイルを変更せず保護した。
- [x] AdamW moment、GradScaler、RNGを復元し、追加フェーズ用に学習率スケジュールを再設定した。
- [x] epoch6〜20を完了した。追加11,520 crop、2,880 successful optimizer update、AMP skip 0回、global_step 960→3,840である。
- [x] epoch6〜20の各checkpointとValidation128 P0結果を保存した。
- [x] epoch20 checkpointをP0とB3で評価し、主要指標と十分統計を保存した。
- [x] 画像単位paired bootstrapを10,000回、seed 42で実施し、pooledとimage meanを別々に保存した。
- [x] GPU環境でテストを直接再実行し、16件・12件・1件の各suiteがすべて成功、総合終了コード0を確認した。初回ログのshell wrapperに終了コード集約エラーがあったため、修正版の直接実行ログも保存した。
- [x] GPGPU4の実行成果をNFSの新規バックアップへ保存し、106ファイルのSHA256照合に成功した。
- [x] paired bootstrap、provenance、epoch別曲線データ、SVG 8図、テストログをNFSからWindows正本へコピーし、12ファイルすべてSHA256SUMS.sourceと一致することを確認した。
- [x] 6条件比較CSV・SVG、ローカル成果物SHA256一覧、provenance照合メモを作成した。
- [ ] 最終差分・公開対象を確認し、既存の実験ブランチへcommit・pushして、PR #8をbaseとする新規PRを作成する。

## 正式runと主要成果物

- GPU：GPGPU4、GeForce GTX 1080 8GB
- 実行開始：2026-11-26 22:28 JST
- 実行root：/localdata/s0623003/c0_extended_training_20261126/run_epoch6_20_20261126T222819/
- epoch5 checkpoint SHA256：2fd190727c7a5124c56c1e0a2dd8cb09c6cb62a0f92b34fce97ad394281b35a
- epoch20 checkpoint SHA256：ce7068a619c41775f0d42553510954fc15eb1be15a5cc269f5a485303b2e5fa4
- epoch20 global_step：3,840
- NFSバックアップ：/home1/eng-std/s0623003/gpu4_research_backup_c0_extended_epoch20_20261127/
- Windows正本内成果：results/final_20261127/
- NFSバックアップ：110ファイル、SHA256照合済み106ファイル
- ローカル転送ファイル：12件すべてNFS SHA256 manifestと一致
- ブランチ：research/c0-extended-training-epoch20-20261126
- 現在のHEAD：cffb44eb8d935376df8a9d09d0cecd4881e7694a

## 結果の要点

epoch20-P0のValidation128 RMSEは0.723565mであり、epoch5-P0の0.744069mより2.76%低かった。全14px局所MAEは0.059084mから0.058528mへ0.94%、GT差20cm以上の14px局所MAEは0.334913mから0.332422mへ0.74%低下した。

epoch20-B3はRMSE 0.724078m、全14px局所MAE 0.062684m、高差分14px局所MAE 0.323614mである。P0より高差分局所誤差は小さい一方、全14px局所MAEは大きい。E3 epoch5-B3は高差分局所MAE 0.319867mであり、この指標では引き続き最良である。指標別の比較とbootstrap区間はreport_ja.mdおよびresults/final_20261127/analysis/paired_bootstrap.csvに記録した。

## 保全・公開上の注意

epoch5およびepoch20を含むcheckpoint本体、原RGB/GT画像、予測cacheはNFSバックアップに保管し、GitHubには含めない。NFS側の検証記録と、ローカル成果物に対するSHA256SUMS.local.txtを保存した。epoch20の実SHA256はNFSのcheckpoint hash記録とバックアップmanifestで照合し、旧報告書の転記誤りをprovenance_reconciliation_ja.mdに記録した。元の実行provenanceファイルは改変せず保持した。なお、epoch20-P0評価JSON自体にはcheckpoint SHA256がなく、元provenanceの同項目もnullである。この未解決の結び付け制約を報告書に明記し、後付け修正は行っていない。

GPU上の当実験プロセスは終了し、終了確認時のGPU使用率は0%であった。他ユーザーのプロセスは停止・変更していない。既存の作図HTTPサーバーは当実験で起動したものと確認できないため停止していない。SSH端末は既存の接続状態を維持しており、別端末を追加していない。

## 限界

単一seed、同一撮影日のValidation128による探索的比較である。epoch6で学習率を追加フェーズ用に再設定しており、同じ学習率スケジュールを20 epoch連続適用した学習ではない。epoch17でRMSE最小を観測したが、主評価は事前固定したepoch20である。独立test319は未使用で、独立データへの汎化は未検証である。
