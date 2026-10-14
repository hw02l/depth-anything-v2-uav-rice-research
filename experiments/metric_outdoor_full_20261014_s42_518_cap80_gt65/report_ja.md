# 第3回 Metric Outdoor Small 学習（5/20 epoch）

20epoch総期間のPolynomialLRを維持し、epoch 5で安全停止した。初期重みはDrive上の元Metric Outdoor Small。モデル出力上限80m、GT有効上限65m。Independent testと旧52組は未使用。

|epoch|train SiLog|val MAE (m)|val RMSE (m)|AbsRel|Pearson|14px全ペア局所MAE (m)|20cm+標本局所MAE (m)|振幅比|符号一致|時間 (s)|peak allocated/reserved (GiB)|AMP skip|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|1|0.106219|1.0791|1.6299|0.05088|0.9805|0.06537|0.35276|0.3046|0.7828|1085.5|0.877/1.248|1|
|2|0.052883|1.7002|1.9953|0.08907|0.9884|0.06339|0.34543|0.3263|0.8024|1044.1|0.877/1.248|0|
|3|0.035958|1.0492|1.3334|0.05726|0.9894|0.06299|0.34708|0.3228|0.8082|1097.8|0.877/1.248|0|
|4|0.030246|0.7744|1.0562|0.03913|0.9901|0.06237|0.34495|0.3428|0.8052|1024.4|0.877/1.248|0|
|5|0.027943|0.7373|1.0094|0.03694|0.9919|0.06148|0.34090|0.3304|0.8102|1020.9|0.877/1.248|0|

## checkpoint選択

- RMSE最良: epoch 5（1.0094 m）, `best_checkpoint.json`.
- 20cm以上の局所差MAE最良: epoch 5（0.34090 m）, `best_shape_checkpoint.json`.
- 20cm以上の標本は前回と同じseed・候補座標ルールで抽出し、全ペア指標と分けて保存した。

## 作物・方向・画像端

各epochの `epoch_metrics.json` に作物別・撮影方向別の全有効画素指標と、画像端0–128px・内部512px以上の固定標本局所指標を記録した。主評価は全画面のまま。

## 第2回Model Cとの参照

保存済み第2回Model C: RMSE 1.7490m、AbsRel 0.05208、14px局所差MAE 0.06800m、20cm+振幅比0.327、符号一致率0.7795。第3回の初期モデルは元の未追加学習Metric Outdoor Smallから開始し、testは使っていない。

## 保存物

- epoch CSV: `/content/drive/MyDrive/DepthAnythingV2-UAV/experiments/metric_outdoor_full_20261014_s42_518_cap80_gt65/results/epoch_metrics.csv`
- epoch JSON: `/content/drive/MyDrive/DepthAnythingV2-UAV/experiments/metric_outdoor_full_20261014_s42_518_cap80_gt65/results/epoch_metrics.json`
- 学習曲線: results/validation_rmse.png, validation_abs_rel.png, local14_mae.png, gt_ge20cm_amplitude_ratio.png, gt_ge20cm_sign_match.png
- checkpoint: `/content/drive/MyDrive/DepthAnythingV2-UAV/runs/metric_outdoor_full_20261014_s42_518_cap80_gt65`（epoch 0–5のcomplete markerとSHA256を検証）
- 再開: 同じrun、epochs=20、同じコード・manifest・学習設定を保ち、`--resume --stop-after-epoch 10`。scheduler期間を5へ変更しない。
- 次の15epochは自動開始していない。無料GPUを再割当でき、validationの絶対・局所形状の推移が継続価値を示す場合に、同一runから再開する。

生成時刻: 2026-10-14T00:22:20.728961+00:00
