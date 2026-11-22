# 既存最良モデルとB3型融合推論の組合せ

本実験は、既存C0・E3 checkpointを固定した推論専用の比較である。学習、optimizer更新、Independent test319の利用は行わない。正式評価は、訂正済みRGB前処理を用いたValidation128に限定する。

## 条件

- **C0-B3**：C0 epoch5 checkpointを全体518px推論と2000pxタイル推論の両方に使用し、従来B3式で融合する。
- **E3-B3**：既存の選択済みE3 epoch5 checkpointを両方の推論に使用し、同じ融合を適用する。
- 比較する既存条件はepoch20-B0、epoch20-B3、E3-B0、C0-P0である。画像別の検証済み結果を再利用する。

B3は `L(D_global) + D_tile - L(D_tile)` である。`L`は元画像座標の215×215正規化box filter、境界は`BORDER_REFLECT_101`である。2000×2000元画像タイルを518×518入力で推論し、25%公称重複、端部anchoring、正のHann重みを用いる。

## 進行状況

実験計画、固定設定、実行コード、回帰テストを準備中である。GPU推論は1枚のRGB/strict-load/B3再現ゲートを通過した後に限り開始する。実測値が揃うまで結果と結論は未確定である。

詳細は [PLAN_JA.md](PLAN_JA.md) と [STATUS_JA.md](STATUS_JA.md) を参照する。
