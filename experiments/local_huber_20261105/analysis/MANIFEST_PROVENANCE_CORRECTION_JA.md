# 固定ペアmanifestのprovenance訂正記録

## 訂正内容

実験設定ファイル `config.json` の `manifest_sha256.fixed_pairs` は16進文字列が63文字であり、SHA256表記として不完全である。この設定ファイルは実験後に書き換えず、実行時の原本を保全する。

固定ペアmanifestの実ファイル `experiments/multiscale_tile_inference_20261016/pair_manifest.json` のSHA256は、GPGPU4上の実ファイルとNFSバックアップ上の記録を照合した結果、次の64文字である。

`bcb56eab03bec4a44f0626da0f4814876fae67774e536e00cd9f1ad16b1f5728`

検証コードは実ファイルのSHA256を実行時signatureに記録し、各画像の固定座標列を再生成した座標列と照合している。最終モデル再評価でも同じmanifest hashを検査し、各画像の14px pair coordinate listの一致を検証する。したがって、今回の訂正対象は設定ファイル内のprovenance文字列である。評価に使った固定座標の変更や、誤記を事後修正した設定ファイルへの置換は行っていない。

## 第三者による確認

NFS上の検証済みbackupは次の場所にある。

`/home1/eng-std/s0623003/gpu4_research_backup_local_huber_20261105_20261105T081949Z`

同backupの `SOURCE_SHA256SUMS` とmanifest実体のSHA256を比較する。設定ファイル、manifest、run signatureを併せて確認すれば、実験時に使われた座標標本を検証できる。原本 `config.json` のSHA256は `1692ebf048a99b5624c69330b23ad67954aac7ab047e024c83da54a72b308127` であり、当初の63文字表記を含むファイルとして保存されている。

## 制限

原本設定内のhash文字列の誤りはprovenance品質上の不備であり、ここで作る訂正記録はその事実を隠さない。manifest hashだけでモデル出力の正しさは保証されないため、B0前処理・画像順序・GTマスク・座標標本・保存済みvalidationとの数値照合を別に記録する。
