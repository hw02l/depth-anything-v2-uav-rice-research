# Colab実行報告（2026-10-13）

## 再開時の確認
前回のローカルSSDコピーは正常終了（終了コード0）。学習は補助セルの文字列一致検査で停止し、未開始だった。停止原因は subprocess.run(cmd,check=True) の空白差。補助セルのみ修正し、研究コード・分割計画は変更していない。
DriveとSSDの両方で512組・1024ファイル、7,850,330,776 bytes、欠損0・サイズ不一致0。SSD全1024ファイルをSHA256で確認。1023ファイルはCPU取得時ログ、残り1ファイルはDrive原本とのSHA256比較で一致。再取得・再コピー・画素再監査は実施していない。

## 保存先と準備
MyDrive/DepthAnythingV2-UAV/uav_metric_finetuning にコード・Notebook・config/verified_planを配置済み。RGBは data/uav3dcrop_rgb、GTは data/uav3dcrop_depth。作業レポートは work/uav_metric_finetuning。既存成果物は保持した。
train384（Day019/028/040）、validation128（Day047）。独立test319は未取得・未使用。Day001/007と旧52組のシーンは隔離。
CPU監査pass：RGB/GT同解像度3956×5280、GT float16、NaN/Inf/負値0。無効GTは0。全512組の有効画素10,671,514,368、無効ゼロ22,977,792。
GPU切替時のDrive更新時刻丸め77件はCPU取得ログのSHA256との一致で再検証し、旧レポートを保持して派生監査・GTレポートを作成。監査コード・仕様・manifest・保存先・サイズ・環境の一致を確認して再利用した。
trainのGTのみ解析：p99.9=52.90625m、全有効画素最大62.15625m。60m超15,424画素（0.0001927%）。全train GTを包含するためmax_depth=65mを選択。validation/testによる上限選択なし。

## GPUと試験条件
無料Tesla T4、総VRAM14.563GiB。CUDA/PyTorch GPU利用可。
固定revisionのRelative SmallとMetric Outdoor Smallのcacheを再利用。今回relative_encoder方式（エンコーダ転用、新Metric head）、AdamW、SiLog、518×518、batch1、蓄積4、AMP、seed42、1epoch、max_depth65m。
事前計測pass：Forward/Backward・AdamW step成功、数値検査pass。allocated624,851,456B、reserved815,792,128B、合成step22.656秒。実データ全体の成功をこの事前計測だけで保証していない。

## 実測結果
全モデルを同じvalidation128組・元解像度で評価。有効画素2,668,757,375、局所差ペア5,320,891,492で一致。局所差は14pixel間隔。深度はm、AbsRelと相関は無次元。

|指標|Relative encoder＋新head（学習前）|1epoch学習後|既存Metric Outdoor Small|
|---|---:|---:|---:|
|MAE m|13.8251|2.4897|13.2730|
|RMSE m|14.7163|3.9063|14.5340|
|AbsRel|0.8324|0.1212|0.6590|
|局所深度差MAE m|0.06437|0.10545|0.06561|
|Pearson r|-0.05599|0.87117|0.78475|

学習前はRelativeモデルの直接的なMetric評価ではなく、転用encoder＋新Metric headのcheckpoint_epoch_0000。既存Metric Outdoorは元の上限80mを維持（65mへ変更せず）、追加学習なし。有効画素は全モデルで同じ。
SiLog Loss=0.29133354。学習＋validation=774.469秒（12分54秒、checkpoint保存・起動は別）。実測VRAM peak allocated943,323,648B（0.879GiB）、reserved1,340,080,128B（1.248GiB）。学習正常終了コード0、completed1、remaining0。
作物別RMSE：corn4.9581m、oat3.4187m、soybean3.6021m、wheat3.4340m。
平均深度誤差・相関は既存モデルより改善。しかし局所深度差は悪化し、独立評価は未実施。稲のGT・高さ評価も未実施。この結果で稲の高さ精度の改善を主張できない。

## checkpointと資源管理
runs/relative_encoder_trial_20261013_s42_518/checkpoint_epoch_0000 および checkpoint_epoch_0001 をDriveへ保存。complete.jsonの全保存ファイルSHA256が一致。モデル読み込みによる学習前評価も成功。optimizer/scheduler/scaler状態も保存され、完全性は確認済み。今回のGPU中断再開は実施していない。
約20分07秒のSSDコピーは前回に完了。データ取得は前回約11分26秒。今回の追加処理は試験学習＋学習前評価＋既存モデル評価。既存モデル評価は約9分39秒の完了確認時経過（監視間隔を含む）。
20epoch単純概算258.16分（約4時間18分、起動・保存除外）。無料GPUの継続時間・同GPU割当は保証されない。Compute Units表示は利用可能ユニットなしで、実消費量の数値を取得できない。追加購入・課金・プラン変更なし。
GPUを使う処理はすべて終了。Drive保存確認後にランタイムを終了する。終了確認は最終メッセージを参照。

## 次の作業
本学習は新runで開始する。試験1epochから20epochへの強制resumeは行わない。本学習中のresumeは同じ総epoch数・設定・コード・データ・パッケージ条件を保持する。
518pxのメモリは今回T4で十分だが、次回GPU割当では再計測する。GT65m、SiLog、seed42を維持し、まずmetric_outdoor初期化方式の比較実験と局所形状・稲/地面のGT評価を計画する。損失の無断変更なし。独立testはモデル選択完了後まで保持する。
セッション終了でローカルSSDは失われる。次回はDrive原本から必要時のみSSDへコピーし、HFからの再取得は不要。0.初期化→環境→設定/監査再利用→モデルcache→新GPU事前計測の順で復元する。

実行Notebook：https://colab.research.google.com/drive/1_kykQhNEVXorSIVAsl_dKYxqpJZDcbmk
詳細JSON：work/uav_metric_finetuning/execution_summary_20261013T113119407440Z.json
