# Driveに保持する大容量資産と復元

正本ルートは`MyDrive/DepthAnythingV2-UAV/`。この移行ではデータ・モデル・checkpoint・全予測を再ダウンロードしていない。Drive側を削除・上書きしていない。

|資産|正確な保存先|扱い|
|---|---|---|
|RGB|`data/uav3dcrop_rgb/`|train/validationの既取得512組を再利用。全実データはGit対象外|
|Z-depth GT|`data/uav3dcrop_depth/`|RGBと合わせて1024ファイル、7,850,330,776 bytes|
|A checkpoint|`runs/relative_encoder_trial_20261013_s42_518/checkpoint_epoch_0000/`〜`0001/`|完全ファイルはDriveのみ|
|C checkpoint|`runs/metric_outdoor_trial2_20261013_s42_518_cap80_gt65/checkpoint_epoch_0000/`〜`0001/`|同上|
|D checkpoint|`runs/metric_outdoor_full_20261014_s42_518_cap80_gt65/checkpoint_epoch_0000/`〜`0005/`|同上。5/20epoch、15残|
|事前学習モデル|`model_cache/models--depth-anything--Depth-Anything-V2-Metric-Outdoor-Small-hf/snapshots/fd2c22027eaf20374204f14099b8341e1925ad39/`|署名に元のColab絶対パスとrevisionが含まれる|
|A/B/C予測|`experiments/metric_shape_20261013T123006472484Z/analysis/prediction_cache/`|384予測。SHA一覧は同実験のprediction_cache_integrity.json|

DriveのファイルID・サイズ・記録済みhashは`REMOTE_ASSETS.json`を参照する。checkpoint hashはDriveの`complete.json`をそのまま取り込んだ**記録値**であり、今回大容量本体を取得して再計算した値ではない。存在とサイズはDriveフォルダー一覧で再確認した。全データの内容は既存`ssd_integrity.json`／監査JSONに依拠し、今回1024ファイルを再走査したわけではない。

復元する場合は認証済みDriveの正確なフォルダーから3ファイル `model.safetensors`、`trainer_state.pt`、`config.json`と`complete.json`をまとめてコピーする。モデルだけではoptimizer/scheduler/scaler/RNGを再開できない。コピー後はcomplete.jsonの3つのSHA256を全て計算・比較する。`scripts/research_workspace.py verify --checkpoint <コピー先>`でも確認できる。元のDrive checkpointは保持する。

第3回epoch5：model SHA256 `c80887515ffdf4ed1708e64a4f85ce6aee20b2550cea7b1b39d9e4d50da3ebf0`、trainer state `b8be075012d158237a2a158d127d23279228b71b9c9e24827eb44d7748f9eaf1`。ローカル`checkpoint_metadata/`は軽量metadata専用で、完全checkpointとして扱わない。

Colab上の元パスは`/content/drive/MyDrive/DepthAnythingV2-UAV/`である。RGB/GTを`/content/uav3dcrop/rgb`と`depth`へ置く必要がある場合、既存Driveデータから不足ファイルだけコピーし、manifest/size/hashを確認する。Hugging Faceから512組を再取得しない。

初期`source_build.json`は型修正前のmodeling.py hashを含む。後続`preflight_config_type_fix.json`と正式`run_metadata.json`を根拠とする。今回取得したsource_full_v1は正式run署名10項目・train/validation manifestと一致した。source_build自体を書き換えて辻褄を合わせない。
