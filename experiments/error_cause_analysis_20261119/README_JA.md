# 深度誤差要因分析（2026-11-19）

このディレクトリは、Depth Anything V2によるUAV農作物深度推定で観測された絶対深度誤差と局所深度差の過小推定を、保存済みの検証統計から分析する独立成果物である。学習と推論は実施していない。

公開PR：[PR #7](https://github.com/hw02l/depth-anything-v2-uav-rice-research/pull/7)。PR #6のブランチをbaseとしている。

## 主な結果

- C0-P0の絶対RMSEは0.7441 mであり、9条件中で最小である。
- P1/P2ではモデル・推論方式の組合せによりRMSE、Bias、Pred対GTの回帰傾きが大きく変化する。F1-P1は2.6166 m、F2-P2は3.0840 mで、C0-P0よりかなり大きい。
- `|ΔGT| >= 0.20 m` の元画像14px局所MAEは全ての9条件で振幅比1未満となり、局所変化の過小推定が残る。
- F2-P2対C0-P2の主要比較ではpooled局所MAEが7.31%改善するが、画像平均改善は3.96%である。C0-P0との絶対RMSE比較ではF2-P2が約4.14倍となる。
- 解像度一致は一部の比較で改善と関連するが、それだけで絶対深度誤差や局所差圧縮を解消しない。単一seed・同一撮影日・モデル選択の限界がある。

詳細な仮説判定と制約は [`report_ja.md`](report_ja.md) を参照する。

## ディレクトリ

- `analysis/`：再集計CSV、図表、検算記録、解析provenance。
- `analysis/per_image_absolute_diagnostics.csv` と `analysis/per_image_local_diagnostics.csv.gz`：bootstrap・層別に利用した画像別十分統計。局所統計は公開時にgzip圧縮してある。元のCSVはローカル解析ディレクトリに保持している。
- `analysis/local_stratified_14px_diagnostics.csv.gz`：14px局所差の層別集計。大容量のため公開版はgzip形式である。
- `source_evidence/`：PR #6上の検証済みB0再現監査とepoch5 checkpoint監査の控え。
- `tests/`：十分統計の集計・bootstrapに対する単体テスト。
- `figures/`：固定代表画像のRGB・GT確認図。実画像を含むためローカル保管のみとし、GitHubには公開しない。
- `analyze_error_causes.py`：CSV/JSONから結果を再生成するCPU解析コード。
- `prepare_public_release.py` と `SHA256SUMS`：gzip公開コピーと配布対象ファイルのハッシュ一覧を再生成・検証する。
- `make_fixed_rgb_gt_figure.py`：既存の固定代表画像をRGB・GTだけで描くコード。推論図ではない。

公開版のgzip CSVは一般的な `gzip -d` で展開できる。再解析で出力される非圧縮CSVはローカル解析ディレクトリに保持している。

## 再現

正本の2026-09-20リポジトリと、その `.venv` にあるPython 3.12環境を使用する。

```powershell
python analyze_error_causes.py `
  --repo-root "<canonical repository root>" `
  --output analysis `
  --bootstrap-reps 10000 `
  --seed 42

python -m unittest discover -s tests -v
```

RGB・GT確認図は次のコマンドでローカル生成できる。

```powershell
python make_fixed_rgb_gt_figure.py `
  --repo-root "<canonical repository root>" `
  --output-dir figures
```

解析の詳細は [`analysis/provenance.json`](analysis/provenance.json)、判定可能な検算は [`analysis/validation.txt`](analysis/validation.txt) に記録した。
