# 稲UAV画像で観察したfailure case

## 調査範囲

2025-08-22、2025-09-12、2025-10-03から各3枚を処理し、RGB、relative depth、outdoor metric depth、VARI mask、ground-candidate overlayを目視比較した。ここで「確認済み」は画像・生成物から直接確認した事項、「推測」は原因候補である。

## 1. lateの高密度・黄化canopy

**確認済みの事実**

- 2025-10-03のframe132などでは画像の大部分が密集した稲葉で覆われる
- relative depthは葉の輪郭や局所的な面の差を示すが、地面を明瞭に分離したとは言えない
- metric depthは画面内で連続的に変化するが、地面候補maskの赤領域が黄色・褐色の葉の上にも広がる
- VARIは黄化葉を非植生側に取りこぼし、ground heuristicがその領域を地面候補と誤認し得る
- そのため、late 3枚は`NO_GROUND_REFERENCE`としてdepth difference proxyを数値化しなかった

**原因の整理**

- 主な直接原因はRGB vegetation maskのdomain shift（緑でない葉を除外）と、地面が見えないこと
- DA2が葉の奥にある地面を推定できるかは、公開データにdepth ground truthがないため判定できない
- 仮説として、細い葉・遮蔽・繰り返しテクスチャは単眼depthの不確実性を大きくする可能性がある

## 2. earlyの土・乾いた草・影

**確認済みの事実**

- early画像には稲列間の露出土壌や明るい背景が見える画像がある
- ExG/HSVは乾いた草や土をvegetation側に含める場合があり、VARIは比較的保守的だった
- 地面候補が広い画像でも、候補depthの分布が一様とは限らない
- earlyのdepth difference median proxyは、3枚で約-0.064〜0.095mだった

**解釈**

この値は小さい局所差を示す可能性があるが、地面候補が純粋な土壌である保証も、同一平面である保証もない。従って稲高とは呼ばない。

## 3. middleの地面可視性

**確認済みの事実**

- middleの3枚では、画像内の隙間や画像端に裸地候補が残るものがある
- proxyは約-0.430〜1.650mと画像間でばらついた
- frame1108の候補fractionは約0.039で、候補が少ない

**推測**

候補が地面の一部だけを表す、視点・斜面・遮蔽が違う、またはvegetation maskが異なるなど、複数要因が考えられる。GTがないため、depthモデル単独の誤差とは断定できない。

## 4. カメラgeometryとprofile

今回のprofileは各画像の中央水平線を機械的に採用した。これは説明用のline profileであり、groundからcanopyへ移る物理的な断面を保証しない。

- nadir姿勢が保証されていない
- 地面が平坦とは限らない
- camera intrinsics、pose、AGLがframeごとに揃っていない
- occlusionにより同一line上でgroundとcanopyを対応できない可能性がある

このためprofileに変化があっても「稲高を再現した」とは言えない。

## 5. 問題の切り分け

| 問題 | 主な候補 | 現段階の判断 |
|---|---|---|
| 黄色い葉がground候補になる | vegetation mask | 確認済み。VARIの色依存による失敗 |
| groundが見えない | scene geometry / canopy density | 確認済み。lateではproxyを中止 |
| depthがsmoothで絶対精度不明 | DA2 / domain shift | GTがないため未判定 |
| proxyが画像間で不安定 | mask・ground選択・視点・モデル | 複合要因と推測 |
| 画像間の時期差が生育差か不明 | frame位置・pose対応不足 | 確認済み。longitudinal比較不可 |

## 結論

今回の重要なfailureは「深度推定が失敗した」と単純化することではなく、地面基準を得られない密集canopyと、RGBだけのvegetation/ground分離の不確実性である。公開GTがないため、DA2の絶対精度とmaskの誤りを分離できない。代表図は`outputs/rice_uav/report_figures/figure_7_failure_case.png`である。
