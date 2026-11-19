# 深度誤差要因分析（2026-11-19）

このディレクトリは、Depth Anything V2によるUAV農作物深度推定で観測された絶対深度誤差と局所深度差の過小推定を分析する成果物である。初回分析は保存済みValidation128十分統計によるCPU解析である。追加診断では固定8画像・5条件だけをGPU3で推論し、予測マップ、符号付き局所差、tile重複を調べた。新規学習、Validation128全件の再推論、Independent test319の使用はない。

公開PR：[PR #7](https://github.com/hw02l/depth-anything-v2-uav-rice-research/pull/7)。PR #6のブランチをbaseとしている。

## 主な結果

- C0-P0の絶対RMSEは0.7441 mであり、9条件中で最小である。
- P1/P2ではモデル・推論方式の組合せによりRMSE、Bias、Pred対GTの回帰傾きが大きく変化する。F1-P1は2.6166 m、F2-P2は3.0840 mで、C0-P0よりかなり大きい。
- `|ΔGT| >= 0.20 m` の元画像14px局所MAEは全ての9条件で振幅比1未満となり、局所変化の過小推定が残る。
- F2-P2対C0-P2の主要比較ではpooled局所MAEが7.31%改善するが、画像平均改善は3.96%である。C0-P0との絶対RMSE比較ではF2-P2が約4.14倍となる。
- 解像度一致は一部の比較で改善と関連するが、それだけで絶対深度誤差や局所差圧縮を解消しない。単一seed・同一撮影日・モデル選択の限界がある。
- 固定8画像の14px・GT差20cm以上では、F2-P2の局所MAEはC0-P2より6.3%低い一方、C0-P0より10.7%高い。これはValidation128の正式値ではない。
- 固定8画像のtile重複域における隣接tile予測差平均はC0-P2 0.365m、F1-P1 0.925m、F2-P2 0.911mであった。融合前後で評価支持域が異なるため、この差を融合効果とは解釈しない。

詳細な仮説判定と制約は [`report_ja.md`](report_ja.md) を参照する。

## ディレクトリ

- `analysis/`：Validation128再集計CSV、図表、検算記録、解析provenance。GPU3固定8の全出力・ログ・ソースは研究室のprivate NFSへ退避し、外側SHA256SUMS 149項目の照合を完了した。公開記録ではNFSの実ディレクトリを伏せる。RGB/GTを含む図は公開しない。
- `analysis/per_image_absolute_diagnostics.csv` と `analysis/per_image_local_diagnostics.csv.gz`：bootstrap・層別に利用した画像別十分統計。局所統計は公開時にgzip圧縮してある。元のCSVはローカル解析ディレクトリに保持している。
- `analysis/local_stratified_14px_diagnostics.csv.gz`：14px局所差の層別集計。大容量のため公開版はgzip形式である。
- `source_evidence/`：PR #6上の検証済みB0再現監査とepoch5 checkpoint監査の控え。
- `tests/`：十分統計の集計・bootstrapに対する単体テスト。
- `figures/`：固定代表画像のRGB・GT確認図。実画像を含むためローカル保管のみとし、GitHubには公開しない。
- `analyze_error_causes.py`：CSV/JSONから結果を再生成するCPU解析コード。
- `prepare_public_release.py` と `SHA256SUMS`：gzip公開コピーと配布対象ファイルのハッシュ一覧を再生成・検証する。
- `make_fixed_rgb_gt_figure.py`：既存の固定代表画像をRGB・GTだけで描くコード。推論図ではない。
- `analysis/fixed8_gpu3_transfer_status.json`：GPU3追加診断、checkpoint・strict load監査、NFS保全結果、残るprovenance限界を記録する。
- `analysis/gpu3_fixed8_grouped_summary.csv` と `analysis/gpu3_fixed8_tile_summary.csv`：固定8画像の公開可能な丸め済み集約表。元のRGB/GT画像と予測マップを含まない。
- `analysis/gpu3_fixed8_diagnostics_summary.png/.svg` と `analysis/gpu3_fixed8_tile_disagreement_summary.png/.svg`：上記集約値から再生成する公開用のRGB/GT非含有図。
- `analysis/gpu3_provenance_reconciliation.json`、`analysis/gpu3_nfs_backup_verification.json`、`source_evidence/gpu3_checkpoint_load_audit.csv`：checkpoint実体・内部ロード・ソースハッシュ・NFS退避の照合記録。

GPU3追加診断の固定8結果は記述的な補助標本であり、Validation128正式評価の値と混同しない。チェックポイント実体のSHA256・epoch・global step・内部signature・strict loadを40件の推論記録から照合した。完全なGit source revisionと実行時の完全なargvは保存されておらず、その制約は `analysis/gpu3_provenance_reconciliation.json` に明記した。図表はprivate NFSに退避し、公開図からはRGB/GTを除外する。詳細は [`STATUS_JA.md`](STATUS_JA.md) と `report_ja.md` の8.6–8.7節を参照する。

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
