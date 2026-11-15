# invalid_pre_fix：2026-11-14 初回評価の監査用保存先

このフォルダーのCSV・図表は、RGB値域不具合を含む初回評価をそのまま保存した監査記録である。研究結果として引用・比較・結論付けに使用してはならない。ファイルは履歴を保つため変更していない。

初回評価では、load_pair()が返すfloat32 [0,1] RGBをuint8へ直接変換し、推論入力の画素値の多くが0になっていた。そのため9条件すべて、そこから生成したsummary_metrics.csv、primary_comparisons.csv、validation_p0_history.csv、local_scale_p2.csv、condition_highlights.csv、example_images.csv、checkpoint_sha256.txt、およびfigures/の図を無効として扱う。RMSE 7.542〜8.651m、負のPearson、約0.448mの大差分局所MAE、約50%の符号一致率は修正前の無効値である。

訂正後のB0再現、RGB回帰検証、全9条件の出力、paired bootstrap、provenance監査は[訂正後報告](../../report_ja.md)および[訂正評価一式](../rgb_input_fix_20261114/)を参照する。なお、C0/F1/F2すべてのepoch5 checkpoint SHA256と評価JSON provenanceに未解決の不一致があるため、訂正後の9条件値も現時点では暫定である。
