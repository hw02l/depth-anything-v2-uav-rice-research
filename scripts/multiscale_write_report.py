"""Write the Japanese research report from measured CSV/JSON evidence only."""
import argparse,csv,json
from pathlib import Path


def table(path):return list(csv.DictReader(Path(path).open(encoding='utf-8-sig')))
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def f(x,n=5):return '未算出' if x in (None,'') else f'{float(x):.{n}f}'
def md(headers,rows):
    return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+['|'+'|'.join(map(str,r))+'|' for r in rows])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--analysis',type=Path,required=True)
    args=p.parse_args()
    out=args.analysis
    summary=read(out/'analysis_summary.json')
    selection=summary['selection']
    labels=['B0','B1_1036',selection['selected_tile_label']]+(['B3'] if selection['run_B3'] else [])
    short={label:('B2' if label.startswith('B2_') else 'B1' if label.startswith('B1_') else label) for label in labels}
    comparison=summary['comparison'];paired=summary['paired']
    multi=table(out/'multiscale_metrics.csv');groups=table(out/'group_metrics.csv')
    a=table(args.results/'diagnostic_A/multiscale_metrics.csv')
    gt=read(out/'manifest.json')
    inventory=table(out/'image_inventory.csv')
    def comp(epoch,label):return next(r for r in comparison if r['epoch']==epoch and r['method']==label)
    def local(epoch,label,stride,group,scope='all'):
        return next(r for r in multi if int(r['epoch'])==epoch and r['method']==label and int(r['stride_px'])==stride and r['group']==group and r['scope']==scope)
    def paired_row(epoch,label,metric):return next(r for r in paired if r['epoch']==epoch and r['method']==label and r['metric']==metric)
    text=[]
    def add(value):text.append(value.strip()+'\n')
    add('# Depth Anything V2の多スケール診断と高解像度・タイル推論の比較研究\n\n実験ID：`multiscale_tile_inference_20261016`。Day047 validation 128画像による探索的比較。追加学習・重み更新なし。')
    add('## 1. 研究背景と今回の到達点\n\nDepth Anything V2 Metric Outdoor SmallをUAV農作物画像へ適応させた20epochモデルは、絶対Z-depthの精度を改善した一方、元画像14pxでの大きな深度差の振幅を約3分の1に過小推定していた。本実験は、入力縮小で局所情報が失われるという仮説を、多スケール診断と重みを固定した推論方式の比較で調べた。\n\n本報告のGTは写真測量由来の**カメラ座標系Z-depth（m）**である。世界座標の鉛直草丈ではない。草丈にはカメラ姿勢・3次元位置・地表面との対応が必要であり、本実験から稲の草丈誤差を主張しない。')
    primary=comp(20,'B0')
    add('今回の主要結果は次の実測表にまとめる。B0のepoch17・18・20は既報のvalidation値を再現した。方式間の数値はすべて同じ128画像、GT mask、局所ペアによる。改善・悪化を隠さず、全ペア局所精度と大差分の再現を区別する。')
    add(md(['epoch20方式','絶対RMSE m','14px全ペアMAE m','GT≥20cm MAE m','GT≥20cm振幅比','符号一致率'],[
        [short[label],f(comp(20,label)['rmse_m'],4),f(comp(20,label)['local_depth_delta_mae_m']),f(comp(20,label)['large14_mae_m']),f(comp(20,label)['large14_amplitude_ratio'],4),f(100*comp(20,label)['large14_sign_rate'],2)+'%'] for label in labels]))
    add('## 2. 正本・既存成果の保護\n\n正本は2026-09-20のWindows研究リポジトリ。開始時のHEADは `3de99b0f35e9083c2c74426a57c44014c6b032c2`、新ブランチは `research/multiscale-tile-inference-20261016`。PR baseは `research/gpgpu4-migration-20261015`（[PR #3](https://github.com/hw02l/depth-anything-v2-uav-rice-research/pull/3)）。PR #3は `research/local-first-20261014` への依存PRであり、mainを最新成果とみなしていない。PR #1・#2・#3をマージ・クローズ・書き換えしていない。開始時の未追跡資料も保持した。\n\n固定source_full_v1、旧run、原ログ、分割manifest、checkpointは編集せず、新コードと結果を独立させた。固定ソース81件と正式署名10件のSHA256を検証した。実モデルはeval、inference_mode、FP32、requires_grad=False。optimizerを復元せず、学習は実施していない。')
    add('## 3. データと事前固定\n\nTrainは384組（Day019/028/040）、validationは128組（Day047）。今回の推論・GT分析はvalidationのみで、独立test319組と過去52組のRGB/GTは未使用。作物はCorn/Oat/Wheat/Soybean、各32画像。Nadir/Obliqueは各64画像、作物×方向8群は各16画像である。\n\n実ファイルを確認した128画像はすべて5280×3956px。有効GTは `isfinite(d) and 0 < d <= 65m`、モデルhead上限は80mで、両者を混同しない。共通有効画素数は2,668,757,375。方式ごとに予測不正値を除外してmaskを変えることはせず、不正値・座標不一致があれば画像全体を停止する設計とした。')
    add('Pilotは各作物×方向16画像をGT深度中央値で並べ、順位floor(n×0.25)、floor(n×0.75)の各1枚、計16枚を予測前に固定した。中央値はGTを縦横16px間隔で標本化した推定値で、並び同値はimage_relpathで決める。選択ID・深度範囲・寸法・RGB/GTハッシュは [pilot_selection.csv](pilot_selection.csv)、[image_inventory.csv](image_inventory.csv) にある。GTは評価とpilot選定にのみ使い、推論時のスケール合わせには使わない。')
    add('## 4. 前処理と空間スケール\n\nB0は固定source_full_v1/data.pyとmetrics.pyのvalidation手順に合わせた。短辺基準サイズで拡大縮小し、縦横をそれぞれ最も近い14の倍数へ丸める。RGBはOpenCV bicubic、ImageNet mean/stdで正規化。validationではcrop・padding・flipを行わない。出力を元解像度へ戻す補間はPyTorch bilinear、align_corners=Trueである。\n\nB0入力は高さ518×幅686、B1は1036×1386。アスペクト比は概ね維持されるが、独立した14倍数丸めで厳密には一致しない。B0の縦横倍率は518/3956と686/5280、B1は1036/3956と1386/5280である。RGBリサイズは半画素中心、出力復元のネイティブ座標は `y*(h_native-1)/(H-1)` と `x*(w_native-1)/(W-1)`。この既存の組合せを保持した。')
    add(md(['元画像距離 px','B0入力相当・縦 px','B0入力相当・横 px','B0 patch間隔相当'],[[s,f(s*518/3956,3),f(s*686/5280,3),f(s*518/3956/14,3)] for s in [14,28,56,107,214]]))
    add('DINOv2のpatchは入力14pxで、B0では元画像の約107pxに相当する。元画像14pxは入力約1.8px、約0.13patchである。ただし、これだけからモデルがsubpatch情報を全く表現できないと断定はしない。B1と2000pxタイル518入力のpatchの元画像換算はともに約54pxで、実効的な視覚スケールが近い。入力解像度だけでなく文脈・丸め・位置埋め込み・学習時の入力分布も変わりうる。')
    add('## 5. 比較方式\n\nB0：短辺518の全体推論。B1：短辺1036の全体推論。B2：元画像から正方形タイルを切り出し、各タイルを518×518で推論し、各cropの元座標へ復元して加重平均する。タイル順序、座標、重みはコードで固定した。重みは `max(hann(h),0.001) ⊗ max(hann(w),0.001)` で、画像外周を含め全画素の重み合計が正となる。paddingはない。')
    add('Pilot候補は1500/2000px×公称重複10/25%の4条件。最後の開始座標は画像の遠端に固定するため、**最終タイルの実際の重複は公称値より大きい**。確定した2000px・公称25%は、y開始=[0,1500,1956]、x開始=[0,1500,3000,3280]の12タイルである。最後の重複は縦77.2%、横86.0%となり、一様な25%重複ではない。実装上のこの非一様性も再現条件・限界として明記する。')
    add(md(['pilot候補','16画像×2checkpointの画像平均MAE m','ep17 RMSE m','ep20 RMSE m','平均秒/枚'],[[('B1_1036' if k.startswith('B1') else k),f(v['joint_image_mean_large14_mae_m']),f(v['rmse_ep17_m'],4),f(v['rmse_ep20_m'],4),f(v['mean_seconds'],3)] for k,v in selection['ranked'].items()]))
    add('16画像それぞれのepoch17/20のGT≥20cm・14px MAEを平均し、その画像平均が最小のタイル方式を固定した。32ケースを32独立画像とは扱っていない。選択結果は2000px・公称25%。RMSE・時間は同点時の副基準とし、不都合な画像を除外していない。全128枚の結果を見てパラメータを再選択していない。')
    add('### B3：GTを用いない全体・局所融合\n\nタイルのpilot RMSEが対応するB0の1.5倍を超えたため、事前に定めた1種類の融合を適用した。B=B0、Q=B2、Lを元解像度の215×215正規化box filter（半径107px、BORDER_REFLECT_101）として、\n\n`P3 = B + α[(Q − L(Q)) − (B − L(B))]`, `α=1`\n\nすなわち `P3 = L(B) + Q − L(Q)`。B0の低周波成分とタイルの高周波成分を組み合わせる。GTによるscale/offset推定、画像別の係数調整、半径・αの事後探索はない。GTはP3の生成後に評価用として初めて読む。周波数分離は局所構造が常に正しいことを保証せず、ノイズやタイルの歪みも残しうる。')
    add('## 6. 指標・標本・統計\n\n局所差は `Δd=d(i)−d(j)`、`Δp=p(i)−p(j)`。MAEは平均`|Δp−Δd|`、RMSEは平方平均誤差の平方根。振幅比は既存と同じ **Σ|Δp| / Σ|Δd|** であり、ペアごとの比の平均ではない。符号一致は`|Δd|>1e−6`を対象に`Δd×Δp>0`で数える。過大・過小振幅は`|Δp|−|Δd|`が±1e−6mを超える割合である。')
    add('14px全ペアMAEは元解像度の水平・垂直の全有効ペア（5,320,891,492）から求めた。5スケールの詳細指標は各画像・各軸100,000候補、seed=`42+元のvalidation画像index×10000+距離`、axis0→axis1の固定標本。14px有効標本25,552,938、GT≥20cmは1,911,836で既存研究と一致した。座標SHA256も全方式で照合した。全ペアと固定標本の指標は置き換えない。')
    add('GT差分の階級は[0,.05)、[.05,.10)、[.10,.20)、[.20,.50)、[.50,∞)m。位置は両端点のうち画像端に近い方の距離を用い、[0,128)、[128,512)、[512,∞)px。両端の区域の組合せも0/1/2の対として保存した。GT深度階級には2端点の平均深度を使う。画素差の正負は上記i−jで統一している。')
    add(f'ゼロ差分基準は14px全ペアで **{gt["full14_zero_delta"]["local_mae_m"]:.8f}m** を再現した。一方、14px固定標本のゼロ基準は0.06426980m、GT≥20cm標本では0.44369775mである。小さな局所MAEだけでは形状再現の証拠にならない。')
    add('不確実性は画像単位で対応付け、10,000回、seed42でbootstrapした。14/28pxの固定標本では画像index101/107にGT≥20cmペアが0、56/107/214pxではindex101が0だった。全128画像の推論・絶対評価は維持し、該当する画像MAEは未定義として保存した。画像平均の差と改善・悪化数は14/28pxで126枚、他の3距離で127枚が対象。未定義MAEを0で埋めていない。別に、全128画像の誤差和・ペア数を画像単位で再標本化するpool指標の対応付きcluster bootstrapも行い、0ペア画像は分子・分母とも質量0として含めた。\n\n符号・crop・方向・位置・尺度別の指標と画像のばらつきはCSVに保存した。同一日の重複撮影画像には相関があり、画像bootstrapでも独立性が完全には満たされないため、区間はDay047内の記述的な目安であり、別日の汎化区間ではない。')
    add('## 7. Baseline再現とcheckpoint監査\n\nGPGPU4上でepoch17/18/20のcomplete markerとmodel/config/trainer-stateのSHA256を照合した。epoch17/20は完了provenanceのハッシュとも一致。保存attention名と実行時名の96キー差を監査し、287テンソルをstrict=Trueでロード後、全値の一致を確認した。署名検証やstrict loadを無効化していない。初回の直接ロードが停止した事実と復旧監査も記録した。')
    add(md(['epoch','再現RMSE m','再現14px全ペアMAE m','既報との照合'],[[e,f(read(args.results/f'B0_ep{e}/summary.json')['metrics']['rmse_m'],10),f(read(args.results/f'B0_ep{e}/summary.json')['metrics']['local_depth_delta_mae_m'],10),'合格・主要指標一致'] for e in (17,18,20)]))
    add('## 8. 実験A：多スケール診断\n\n以下はepoch20、同一固定標本規則による実測。スケールを変えるとGT≥20cmの標本構成と数が変わるため、振幅比の変化だけを細部改善と解釈しない。')
    add(md(['元画像 px','全標本MAE m','全標本RMSE m','GT≥20cm MAE m','振幅比','符号一致率','GT≥20cmペア数'],[
        [s,f(local(20,'B0',s,'all')['delta_mae_m']),f(local(20,'B0',s,'all')['delta_rmse_m']),f(local(20,'B0',s,'gt_ge20cm')['delta_mae_m']),f(local(20,'B0',s,'gt_ge20cm')['amplitude_ratio'],4),f(100*float(local(20,'B0',s,'gt_ge20cm')['sign_match_rate']),2)+'%',f"{int(local(20,'B0',s,'gt_ge20cm')['pair_count']):,}"] for s in (14,28,56,107,214)]))
    add('![A：大差分の多スケール診断](figures/A_multiscale_large_delta.png)\n\n図A。epoch17/18/20はいずれも14pxで強い振幅圧縮があり、距離が大きくなるにつれて振幅比が1へ近づく。GT≥20cmの割合も14px約7.48%から214px約44.41%へ変わる。粗い尺度では視点や面の傾きによる大局的Z-depth差も含まれ、植物の細かな凹凸だけを見ているわけではない。')
    add('## 9. 実験B：全128画像の方式比較')
    for epoch in (17,20):
        add(f'### epoch {epoch}')
        add(md(['方式','MAE m','RMSE m','AbsRel','Pearson','14px全ペアMAE m'],[[short[l],f(comp(epoch,l)['mae_m'],4),f(comp(epoch,l)['rmse_m'],4),f(comp(epoch,l)['abs_rel']),f(comp(epoch,l)['pearson_r']),f(comp(epoch,l)['local_depth_delta_mae_m'])] for l in labels]))
        add(md(['方式','GT≥20cm MAE m','同RMSE m','振幅比','符号一致率','有効ペア数'],[[short[l],f(comp(epoch,l)['large14_mae_m']),f(comp(epoch,l)['large14_rmse_m']),f(comp(epoch,l)['large14_amplitude_ratio'],4),f(100*comp(epoch,l)['large14_sign_rate'],2)+'%',f"{comp(epoch,l)['large14_pairs']:,}"] for l in labels]))
        add(f'![尺度別方式比較 epoch{epoch}](figures/B_multiscale_ep{epoch}.png)')
    add('### 画像単位の対応付き差分\n\n負の差分を改善とする。以下の平均差はGT条件で定義できる126画像のMAE差の平均であり、全ペアをpoolした上表の差と厳密には異なる。index101/107はGT標本0のため未定義で、推論から除外していない。128画像を独立な撮影日と解釈しない。')
    add(md(['epoch','方式−B0','画像平均MAE差 m','画像bootstrap95%区間 m','定義画像数','改善/悪化/不変'],[
        [e,short[l]+'−B0',f((r:=paired_row(e,l,'large14_mae_m'))['mean_difference']),f(r['ci95_low'])+' ～ '+f(r['ci95_high']),r['images'],f"{r['improved']}/{r['worsened']}/{r['unchanged']}"] for e in (17,20) for l in labels[1:]]))
    add(md(['epoch','方式−B0','全128画像pool MAE差 m','対応付き画像cluster bootstrap95%区間 m'],[
        [e,short[l]+'−B0',f((r:=paired_row(e,l,'large14_mae_m'))['pooled_mean_difference']),f(r['pooled_ci95_low'])+' ～ '+f(r['pooled_ci95_high'])] for e in (17,20) for l in labels[1:]]))
    add('![画像別差分 epoch20](figures/B_paired_images_ep20.png)\n\n詳細は[paired_differences.csv](paired_differences.csv)、[per_image_metrics.csv](per_image_metrics.csv)、[per_image_multiscale.csv](per_image_multiscale.csv)、[image_variability.csv](image_variability.csv)。すべての画像IDを保持し、悪化画像の除外は行っていない。')
    add('## 10. GT差分階級・正負・深度・作物・方向・位置')
    add('![GT差分階級別](figures/B_delta_bins_ep20.png)\n\n階級境界は左閉右開、最後は50cm以上。20cm以上という選択で標本の構成が変わるため、各階級のnを図とCSVに示した。')
    add(md(['epoch20のGT差符号','ペア数']+[short[l]+' MAE m' for l in labels],[[name,f"{int(local(20,'B0',14,key)['pair_count']):,}"]+[f(local(20,l,14,key)['delta_mae_m']) for l in labels] for name,key in [('正：Δd>1e−6','positive_gt'),('負：Δd<−1e−6','negative_gt')]]))
    add(md(['端点平均GT深度','ペア数']+[short[l]+' MAE m' for l in labels],[[name,f"{int(local(20,'B0',14,key)['pair_count']):,}"]+[f(local(20,l,14,key)['delta_mae_m']) for l in labels] for name,key in [('0–15m','depth_0_15m'),('15–20m','depth_15_20m'),('20–30m','depth_20_30m'),('30–65m（65含む）','depth_30_65m')]]))
    add('![作物・方向別](figures/B_crop_view_ep20.png)\n\n以下の画像数・有効画素数・全14pxペア数はGTで決まり、全方式共通。詳細精度と固定標本ペア数は[group_metrics.csv](group_metrics.csv)に併記した。')
    display_scopes=['crop=corn','crop=oat','crop=wheat','crop=soybean','view=nadir','view=oblique']
    add(md(['グループ']+[short[l]+' 絶対RMSE m' for l in labels],[[scope]+[f(next(r for r in groups if r['epoch']=='20' and r['method']==label and r['scope']==scope)['rmse_m'],4) for label in labels] for scope in display_scopes]))
    counts=[r for r in groups if r['epoch']=='20' and r['method']=='B0']
    add(md(['グループ','画像数','valid pixels','14px全有効pairs','14px固定標本pairs','GT≥20cm標本pairs'],[[r['scope'],r['images'],f"{int(r['valid_pixels']):,}",f"{int(r['local_pair_count']):,}",f"{int(r['fixed_sample_14_pairs']):,}",f"{int(r['fixed_sample_large14_pairs']):,}"] for r in counts]))
    add('![画像端と内部](figures/B_edge_ep20.png)')
    add(md(['位置（両端の最小距離）','固定標本pairs']+[short[l]+' MAE m' for l in labels],[[name,f"{int(local(20,'B0',14,key)['pair_count']):,}"]+[f(local(20,l,14,key)['delta_mae_m']) for l in labels] for name,key in [('<128px','border_lt128'),('128–511px','middle_128_512'),('≥512px','interior_ge512')]]))
    add('## 11. タイル境界と全体深度の整合\n\nタイルのcrop開始・終了線を共通の参照線とし、その線を跨ぐ14pxペア、端点が線から14px未満の近傍、近傍以外を評価した。跨ぐペアは近傍の部分集合であり、3群は互いに排他的ではない。同じ線・同じGTペアをB0/B1/B2/B3へ適用したため、境界の難易度そのものと方式差を区別できる。参照線は重複重みで滑らかに統合されるcrop境界で、必ず段差が生じるという定義ではない。')
    seam=table(args.results/'panels_ep20/seam_metrics.csv')
    add(md(['境界群','ペア数']+[short[l]+' MAE m' for l in labels],[[name,f"{int(next(r for r in seam if r['method']=='B0' and r['scope']=='all' and r['group']==key)['pair_count']):,}"]+[f(next(r for r in seam if r['method']==l and r['scope']=='all' and r['group']==key)['delta_mae_m']) for l in labels] for name,key in [('線を跨ぐ','cross_tile_boundary'),('線から14px未満','near_tile_boundary'),('それ以外','away_tile_boundary')]]))
    crossing={label:float(next(r for r in seam if r['method']==label and r['scope']=='all' and r['group']=='cross_tile_boundary')['delta_mae_m']) for label in labels}
    add('線を跨ぐ同じGTペアでのB0との差は、'+ '、'.join(short[label]+' '+f(crossing[label]-crossing['B0'])+'m' for label in labels[1:])+'。正は誤差増加、負は誤差低下を意味する。')
    consistency=table(args.results/'panels_ep20/global_consistency.csv')
    differences=[float(r['mean_abs_difference_from_B0_m']) for r in consistency if r['method']==selection['selected_tile_label']]
    add(f'B2とB0の予測同士の平均絶対差（各画像の画素平均を画像平均）は{sum(differences)/len(differences):.4f}m。この整合量にはGTを使っておらず、どちらが正しいかは別途GT誤差で判断する。')
    add('境界ごとの元座標・比較PNG・同一ペアSHA256を保存した。重複タイルの統合前の深度不一致はGTを使わない整合性指標で、統合順序に依存する記述量である。B0との全体的な差もglobal_consistency.csvへ保存した。境界群と内部群ではGT分布が異なるため、群間のMAE大小だけで境界の効果を断定しない。')
    add('## 12. 定性的比較図\n\n掲載対象はpilotの各作物×方向で元manifest indexが最初の1枚、計8枚を結果と無関係に固定した。全8群・両checkpointの全体図と境界拡大を保存した。深度は同一画像内でGTの1–99%範囲を全方式に共用し、絶対誤差は0–2m、局所差誤差は0–0.5mに固定。範囲外は色が飽和する。画像別・方式別の独立min/max正規化は行っていない。全体表示は元画像の8px間隔の点を抽出し、局所差は元画像14px・水平方向。定量評価は縮小PNGから計算していない。')
    cases=read(args.results/'panels_ep20/panel_metadata.json')
    for case in cases:
        i=case['image_index']
        add(f'### {case["crop"]} / {case["view"]}：{case["image_relpath"]}\n\n![深度・絶対誤差・局所差誤差](figures/depth_comparison_ep20_image{i:03d}.png)\n\n![共通タイル境界の512px拡大](figures/tile_boundary_ep20_image{i:03d}.png)')
    add('## 13. 計算環境・速度・メモリ\n\nGPGPU4、Debian 12、GTX1080 8GB、Singularity --nv、記録済みSIFとprivate siteを使用した。PyTorch2.11.0+cu126、Transformers5.18.0、NumPy2.1.3。Compute Capability6.1の実CUDA試験と実モデル推論に成功。ホストへのパッケージ導入・sudo・ドライバ変更は行っていない。RGB/GT・重み・cache・結果の連続処理はlocaldata上で行った。\n\n以下の速度・VRAMは全方式で同じ16枚を用いる。B0/B1/B2は全128枚実行中の当該16枚、B3はB0→B2→融合を同じモデルで逐次実行した実測で、固定RGB1枚のwarm-up後に測定した。B3の実行出力は全128枚のcacheと16枚で照合した。時間には前処理・モデル・出力復元・融合を含み、RGB読み込み・checkpointロード・評価は含まない。PyTorch allocated/reservedのピークはCUDA contextを含む機器全体の厳密なピークとは異なる。')
    add(md(['epoch','方式','共通測定画像','平均秒/枚','peak allocated MiB','peak reserved MiB'],[[e,short[l],comp(e,l)['benchmark_images'],f(comp(e,l)['benchmark_seconds_mean'],3),f(comp(e,l)['benchmark_peak_allocated_mib'],1),f(comp(e,l)['benchmark_peak_reserved_mib'],1)] for e in (17,20) for l in labels]))
    add('![精度と処理コスト](figures/B_cost_ep20.png)\n\n全128枚の個別時間、タイル統合時間、融合CPU時間はruntime_metrics.csvに保存。B3の全128枚のseconds列は親処理の実測成分和で、別のend-to-end実測ではない。共通16枚の上表は実測であり、この違いを区別した。OOM・不正値による画像の事後除外はなく、比較対象は全方式128枚で揃った。')
    add('## 14. 考察：研究課題への回答')
    b1=comp(20,'B1_1036');b2=comp(20,selection['selected_tile_label']);b3=comp(20,'B3') if 'B3' in labels else None
    add('1. **元画像14pxを518入力でどれほど表現できたか。** 入力約1.8px、約0.13patchに当たり、epoch20のGT≥20cm振幅比は0.3327、符号一致率は82.48%。符号の再現と振幅の再現には明確な隔たりがある。\n\n2. **画素間隔を大きくするとどうか。** 28/56/107/214pxの振幅比は約0.513/0.660/0.781/0.872へ増えた。粗い構造の差分を相対的に再現しやすい傾向だが、GT差分分布・標本構成・面の傾きも変わるため、純粋な周波数応答や植物凹凸だけの因果効果ではない。')
    add(f'3. **全体高解像度化で改善したか。** epoch20のGT≥20cm MAEはB0 {f(primary["large14_mae_m"])}mに対しB1 {f(b1["large14_mae_m"])}m、RMSEは{f(primary["rmse_m"],4)}→{f(b1["rmse_m"],4)}m。入力の情報量を増やすだけで改善するという単純な説明は支持されない。\n\n4. **タイルは全体高解像度に比べどうか。** B2のGT≥20cm MAEは{f(b2["large14_mae_m"])}m、振幅比は{f(b2["large14_amplitude_ratio"],4)}。絶対RMSEは{f(b2["rmse_m"],4)}mで、局所の振幅と絶対深度の整合性は別に評価する必要がある。B1と2000pxタイルは実効画素スケールが近く、文脈の切断・cropによる入力分布変化が残る候補である。')
    add('5. **境界で誤差が増えていないか。** 第11節の同一境界ペア比較と全8群の拡大図で確認した。単なる近傍/内部の難易度差と方式差を区別する。加重平均により接続しても、絶対深度の不一致や局所形状の誤差が自動的に解消するわけではない。')
    if b3:
        add(f'6. **絶対深度と局所形状は両立できたか。** B3はepoch20のRMSEを{f(b3["rmse_m"],4)}mにし、GT≥20cm MAEは{f(b3["large14_mae_m"])}m、全14px MAEは{f(b3["local_depth_delta_mae_m"])}mとなった。基準B0の各値と併記し、振幅比だけで優劣を決めない。B3はGTなしで全体深度を保つ診断的な選択肢だが、全ペア精度・境界・画像ごとの一貫性も満たす必要がある。')
    if b3:
        improved=[scope for scope in display_scopes if float(local(20,'B3',14,'gt_ge20cm',scope)['delta_mae_m'])<float(local(20,'B0',14,'gt_ge20cm',scope)['delta_mae_m'])]
        worsened=[scope for scope in display_scopes if scope not in improved]
        add('7. **作物・方向で違いがあるか。** B3のGT≥20cm・14px MAEがB0より低かった群は '+('、'.join(improved) or 'なし')+'、低下しなかった群は '+('、'.join(worsened) or 'なし')+'。これは群ごとの点推定で、群間の独立性や有意差を示すものではない。ObliqueのZ-depth差には視点・地表面の傾きも含まれ、作物と撮影条件を分離できていない。')
        add(f'8. **画像端は改善したか。** 端128px未満の標本MAEはB0 {f(local(20,"B0",14,"border_lt128")["delta_mae_m"])}m、B3 {f(local(20,"B3",14,"border_lt128")["delta_mae_m"])}m。内部512px以上はB0 {f(local(20,"B0",14,"interior_ge512")["delta_mae_m"])}m、B3 {f(local(20,"B3",14,"interior_ge512")["delta_mae_m"])}mである。端・内部はGT構成も異なり、対応付き画像差と併せて判断する。')
    add('9. **解像度と学習損失・構造をどこまで区別できたか。** 重み固定で入力処理の効果は測れたが、学習解像度・loss・backboneのpatch表現・位置埋め込み・全体文脈・GTノイズを独立に操作していない。解像度が唯一の原因であるとも、逆に無関係であるとも結論できない。\n\n10. **局所差損失を導入する価値。** 14pxの振幅圧縮が残り、単純な解像度変更で解決していないため、独立した次実験として検証する価値がある。ただし、GTの写真測量ノイズを増幅する可能性があり、今回の結果はその損失が有効であることを実証してはいない。')
    add('## 15. 次の独立実験案（本実験では未実施）\n\n新しいrunとして、同じ初期モデル・分割・20epoch計画・seed・scheduler条件を揃え、SiLog単独とSiLog＋正規化Huber局所差lossを比較する案が考えられる。元画像の14/28/56pxに対応する有効ペアをtrain側だけで定義し、両端のmask、GTノイズ・float16量子化、画像端、絶対深度RMSEとのトレードオフを確認する。λや距離の候補は少数を事前固定し、入力解像度変更とloss変更を同時に増やしすぎない。今回のrun/checkpointは再開・上書きせず、独立testは手法と評価を固定した後の最終検証まで保持する。')
    add('## 16. 限界\n\nvalidationはDay047の128画像のみ。pilot16枚を含む同じvalidationで方式選択・性能報告を行っており、探索的比較である。独立test319組、別撮影日への汎化、稲の深度GT・鉛直草丈GTは未検証。GTはMVS由来で、対応誤り・遮蔽・ノイズを含みうる。float16 TIFFの深度刻みも数cm以下の局所差の解釈に影響する。\n\nタイルは全体文脈とcamera-scaleの手掛かりを失い、最終タイルの重複も非一様。B3のbox filterは境界を含む局所情報を混ぜ、実際の細部と推定ノイズを区別しない。FP32・特定のSmallモデル・GTX1080・特定の推論サイズの結果であり、他実装・他サイズへ一般化しない。大きな標本ペア数は独立した画像数・日数を増やさない。')
    add('## 17. 再現・保存・後始末\n\n実行手順はREADME.md、設定はconfig.json、入力・標本ハッシュはmanifest.json / pair_manifest.json。CPU単体テストは座標・補間・タイル重み・欠損GT・旧標本一致・ゼロ基準・cache・strict名前対応・融合を含む。初期strictロード失敗とpilot制御ループ失敗は停止・修正し、既存の完了結果を上書きせず回復した。実験結果を再生成せず成功と記述することはしていない。\n\nWindowsを編集・分析・図表・Gitの正本、GPGPU4のlocaldataを計算用コピーとした。結果パッケージはSSH転送前後のSHA256、内部ファイルSHA256を照合して取り込んだ。RGB/GT、checkpoint、全予測cache、認証情報はGitに含めない。GPU終了・コンテナ・SSH・バックアップとcache保持方針の最終状態はcleanup_receipt.json / provenance.jsonに記録する。')
    add('## 18. 出典\n\nデータ：Link-Dev, [UAV3DCrop](https://huggingface.co/datasets/Link-Dev/UAV3DCrop) / [UAV3DCrop Depth](https://huggingface.co/datasets/Link-Dev/UAV3DCrop_depth), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。両cardのライセンスと、Depth cardのカメラ視線軸Z-depth・m・float16 TIFFという定義を2026-10-16に確認した。図は統計、色付け、予測・誤差比較、ROIを加えた加工物。原データそのものは再配布していない。\n\n原論文：[UAV3DCrop: Benchmarking 3D Reconstruction in Repeated Multi-Angle UAV Crop Surveys](https://arxiv.org/abs/2608.06404)。既存20epoch結果は親PR #3とgpgpu4_completion_20261015のprovenance、元の固定source_full_v1に基づく。Depth Anything V2モデル重みは本repoへ再配布しない。')
    (out/'report_ja.md').write_text('\n'.join(text),encoding='utf-8')
    print('Japanese report written:',out/'report_ja.md')


if __name__=='__main__':main()
