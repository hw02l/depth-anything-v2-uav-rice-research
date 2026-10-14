# 第1〜3回実験の中間論文パッケージ

## 日本語フォント修正版

日本語フォントをYu Mincho / Meiryoの静的TrueTypeへ変更し、字体データとUnicode対応表をPDFへ埋め込んだ。XeLaTeXで再ビルドする場合はWindowsの `C:/Windows/Fonts/` に `yumin.ttf`, `yumindb.ttf`, `meiryo.ttc`, `meiryob.ttc` が必要である。フォントファイル自体は同梱しない。

`paper_ja_fontfixed.pdf` は文字検索可能な修正版である。`paper_ja_compat.pdf` は同じ30ページを180dpiの画像として保持した互換版で、日本語フォントの表示機能に依存しない。互換版では本文テキスト検索・選択とリンク操作はできない。

- `paper_ja.pdf`: 確認済み第1〜3回の結果による日本語論文
- `paper_ja.tex`: XeLaTeXで編集できる主原稿
- `chapters/`, `figures/`: 各章とTikZ/PGFPlotsのベクトル図ソース
- `refs.bib`: 本文8件の参考文献に対応するBibTeX形式一覧（主原稿は再コンパイルしやすい `thebibliography` を使用）
- `source_epoch_metrics.csv`: 第3回Drive上のepoch metrics CSVの複製。元ファイルは変更していない
- `evidence_manifest.csv`: 実験根拠ファイルのDrive ID・場所・利用目的
- `reproducibility_appendix.csv`: 条件と記録状態の機械可読一覧
- `report_package.zip`: ソース、図、PDFと付録一覧のアーカイブ

## 再コンパイル

TeX LiveのXeLaTeX環境でパッケージ直下から次を2回実行する。

```powershell
xelatex -interaction=nonstopmode -halt-on-error paper_ja.tex
xelatex -interaction=nonstopmode -halt-on-error paper_ja.tex
```

本文の図は保存済みCSV・JSON・報告値からベクトルとして再描画した。第2回の元比較図29枚は転載せず、根拠一覧のDriveリンクから参照できる。independent testは未使用。著者名・所属は未提供のため、表紙のプレースホルダーを提出時に置き換えること。
