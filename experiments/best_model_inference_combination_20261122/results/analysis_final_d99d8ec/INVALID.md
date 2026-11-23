# 旧集計出力の監査記録

`analysis_final_d99d8ec` は正式結果ではない。初回に使用したanalyzerに、条件間の画像別局所統計行を誤って混合する処理があったと判明した。旧出力を消去せずNFSに保持し、SHA256とともに監査用として区別した。

正式集計は、修正analyzer commit `d99d8ecd4efcfa2863f34eaa130b87033088c96a` に対応する `analysis_reconciled_d99d8ec` である。新規条件C0-B3/E3-B3の推論run内十分統計との再照合を通過した。旧出力の数値を報告書の結果に再利用していない。
