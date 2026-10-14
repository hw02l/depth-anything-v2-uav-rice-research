# 移行時に確認できた補記（原稿・実験結果を変更しない）

2026-10-14のDrive原本取り込みにより、第3回run_metadata.jsonのversionsを直接確認できた。原稿の再現性付録に「未記録」と記載した環境は、Python 3.13.15、torch 2.11.0+cu130、transformers 5.18.0、NumPy 2.1.3、OpenCV 4.14.0、tifffile 2026.9.20、huggingface_hub 1.33.0、requests 2.32.4、safetensors 0.8.0である。正確な元文字列は../../experiments/metric_outdoor_full_20261014_s42_518_cap80_gt65/run_metadata/run_metadata.jsonを参照する。

source_build.jsonは後続のmax_depth整数型修正前のmodeling.py hashを含む。正式run_metadataのresume_signatureには修正後のhashが記録され、取り込んだ固定ソースと一致する。初期markerを改変していない。局所標本件数や同一性について原稿の保守的記述を変更する解析は今回行っていない。

最新PDFはTrueType日本語フォント・Unicode対応表を埋め込んだ修正版である。画像版互換PDFもローカルに保持する。原稿、参考文献、数値・図の内容は今回の移行によって変更していない。

元の論文添付source_epoch_metrics.csvにはUTF-8 BOMが2個あり、Drive原本は1個であった。全先頭BOMを除けば内容は同一である。元の添付と記載hashは維持し、原本の正規コピーをsource_epoch_metrics_canonical.csvとして別名追加した。
