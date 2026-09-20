# 稲UAV画像のvegetation mask検討

## 目的

RGB画像だけから、稲冠とそれ以外を分けるための説明可能な初期maskを作成した。今回のmaskはground-truth segmentationではなく、canopy-ground depth difference proxyの候補領域を作るための探索的な前処理である。

## 比較した方法

`scripts/create_vegetation_mask.py`で、各画素に対して次の3方法を比較した。

1. **Excess Green (ExG)**: `2G - R - B > 20`
2. **VARI**: `(G - R) / (G + R - B) > 0.05`（分母が小さい場合は除外）
3. **HSV**: OpenCV HSVでH=20〜100、S≥45、V≥30

比較結果は`results/rice_uav/vegetation_mask_method_comparison.csv`に保存した。各画像について、3方式のvegetation pixel fractionも記録している。

## 選択した方法

今回の採用方法は**VARI**である。理由は、画像ごとの明るさ・露出差の影響をExGの固定RGB差より受けにくく、明らかな背景・土壌を比較的保守的に除外できたためである。選択理由と閾値は`results/rice_uav/vegetation_mask_selection.json`にも保存した。

実際の観察では、earlyの一部ではExG/HSVが乾いた草や土の色を多く拾い、VARIの方が稲のまとまりを保った。middleでは3方式ともcanopyを概ね覆ったが、VARIは画像端・背景をより多く除外した。

## 失敗と限界

- lateの2025-10-03では黄化・褐色化した葉がVARIで非植生側に回りやすい
- dry grass、土、水面、反射、影は色特徴だけでは完全に分離できない
- VARIの閾値0.05は本データの目視に基づく初期値で、アノテーションによる最適化ではない
- 公開データにpixel-level vegetation ground truthを確認できなかったため、precision/recall/F1は報告しない
- maskの誤りはDA2の深度推定誤差とは別の要因であり、両者を混同しない

## 出力

各画像について、ExG、VARI、HSVのmask、採用したVARI mask、方式比較図を`outputs/rice_uav/<stage>/<date>/<frame>/`へ保存した。報告用の一覧図は`outputs/rice_uav/report_figures/figure_3_vegetation_mask.png`である。

## 今後

次段階では、少数画像でよいので手動pixel annotationを作成し、閾値を定量評価する。また、黄化葉を含むlate画像では、HSVの単独閾値ではなく、色・texture・局所連結性を組み合わせたmaskを比較する。ただしground referenceが見えない問題はmaskだけでは解決しない。
