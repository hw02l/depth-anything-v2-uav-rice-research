# データ・図の出典と改変

UAV3DCrop dataset / UAV3DCrop Depth dataset, Link-Dev。

- RGB・camera情報：https://huggingface.co/datasets/Link-Dev/UAV3DCrop
- 深度GT：https://huggingface.co/datasets/Link-Dev/UAV3DCrop_depth
- 公式project：https://link-dev.github.io/UAV3DCrop/
- 許諾表示：両dataset cardのCC BY 4.0を2026-10-14に確認した。
- ライセンス：https://creativecommons.org/licenses/by/4.0/

本repoの比較図は元RGBと写真測量由来Z-depthに、推定depthの色付け、リサイズ、ROI、断面、統計等を追加・合成したものである。元データそのものの再配布ではない。著者表示とライセンスリンクをこのファイルとREADMEに保持する。

Depth Anything V2 Smallのコード・モデルは各公式配布元の許諾に従う。重みおよび学習済みcheckpointを本repoへ再配布しない。本repo独自コード全体へ新しいライセンスを一括付与する変更はしていない。
