# 実行成果物

このディレクトリには、実行後に次の成果物が保存されます。

- `relative/`：relative depth の可視化 PNG、raw NPY、`run.json`
- `metric/`：metric depth（メートル）の可視化 PNG、raw NPY、`run.json`
- `kitti/`：KITTI 50画像のprediction、metrics JSON、plots、best/median/worst代表例
- `eth3d/`：既存のRGB色付きASCII PLY、depth可視化、点群統計JSON
- `eth3d_gt/`：ETH3D low-res two-view GT depth、predicted/GT PLY、depth比較図、最近傍距離JSON

CSV指標はプロジェクトルートの `results/metrics/` に保存されます。

データセット本体、チェックポイント、キャッシュはサイズが大きいため Git 管理対象外です。
