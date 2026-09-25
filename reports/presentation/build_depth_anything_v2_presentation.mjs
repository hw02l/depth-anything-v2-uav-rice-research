import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { Presentation, PresentationFile } from '@oai/artifact-tool';

const PROJECT = String.raw`C:\Users\81902\Documents\Codex\2026-09-20\depth-anything-v2-python-uav-rgb`;
const TASK = String.raw`C:\Users\81902\Documents\Codex\2026-09-25\powerpoint-c-users-81902-documents-codex`;
const BUILD = path.join(PROJECT, '.codex-presentation-build');
const FINAL = path.join(PROJECT, 'reports', 'presentation', 'depth_anything_v2_rice_research.pptx');
const VALIDATED_FINAL = path.join(PROJECT, 'reports', 'presentation', 'depth_anything_v2_rice_research_reviewed_v3.pptx');
const SKILL_DIR = String.raw`C:\Users\81902\.codex\plugins\cache\openai-primary-runtime\presentations\26.904.11930\skills\Presentations`;
const PYTHON = String.raw`C:\Users\81902\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`;
process.env.RUNTIME_NODE = String.raw`C:\Users\81902\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe`;
process.env.RUNTIME_NODE_MODULES = String.raw`C:\Users\81902\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\node_modules`;
const FONT = 'Yu Gothic';
const C = {
  navy: '#142A43', blue: '#2D6F9F', sky: '#DCEAF4', teal: '#168C88', tealPale: '#E3F3F1',
  orange: '#D98839', orangePale: '#F8EBDD', red: '#B54747', redPale: '#F7E8E8',
  ink: '#253443', gray: '#62717F', mid: '#A9B6C1', line: '#D9E1E7', pale: '#F4F7F9', white: '#FFFFFF',
};
await fs.mkdir(BUILD, { recursive: true });
await fs.mkdir(path.dirname(FINAL), { recursive: true });
const presentation = Presentation.create({ slideSize: { width: 1280, height: 720 } });

function addShape(slide, geom, x, y, w, h, fill = C.white, lineFill = 'none', lineWidth = 0, name) {
  return slide.shapes.add({
    geometry: geom, name,
    position: { left: x, top: y, width: w, height: h },
    fill,
    line: { style: 'solid', fill: lineFill, width: lineWidth },
  });
}
function text(slide, value, x, y, w, h, opts = {}) {
  const sh = addShape(slide, 'textbox', x, y, w, h, opts.fill ?? 'none', opts.line ?? 'none', opts.lineWidth ?? 0, opts.name);
  sh.text = value;
  sh.text.style = {
    typeface: opts.font ?? FONT,
    fontSize: opts.size ?? 24,
    color: opts.color ?? C.ink,
    bold: opts.bold ?? false,
    alignment: opts.align ?? 'left',
    verticalAlignment: opts.valign ?? 'top',
    autoFit: opts.autoFit ?? 'shrinkText',
    wrap: 'square',
    lineSpacing: opts.lineSpacing ?? 1.0,
    insets: { top: opts.pad ?? 0, right: opts.pad ?? 0, bottom: opts.pad ?? 0, left: opts.pad ?? 0 },
  };
  return sh;
}
function rect(slide, x, y, w, h, fill = C.white, lineFill = C.line, lineWidth = 1, radius = false, name) {
  return addShape(slide, radius ? 'roundRect' : 'rect', x, y, w, h, fill, lineFill, lineWidth, name);
}
function line(slide, x1, y1, x2, y2, color = C.line, width = 1.5) {
  return slide.shapes.add({ geometry: 'line', position: { left: x1, top: y1, width: x2 - x1, height: y2 - y1 }, fill: 'none', line: { style: 'solid', fill: color, width } });
}
function arrow(slide, x, y, w = 52, h = 30, color = C.blue) {
  return addShape(slide, 'rightArrow', x, y, w, h, color, 'none', 0);
}
function label(slide, str, x, y, w, fill = C.sky, color = C.navy) {
  rect(slide, x, y, w, 28, fill, 'none', 0, true);
  text(slide, str, x + 8, y + 3, w - 16, 22, { size: 16, color, bold: true, align: 'center', valign: 'middle' });
}
async function image(slide, rel, x, y, w, h, alt, fit = 'contain') {
  const bytes = new Uint8Array(await fs.readFile(path.join(PROJECT, rel)));
  return slide.images.add({ blob: bytes, contentType: 'image/png', alt, fit, position: { left: x, top: y, width: w, height: h } });
}
function base(title, no, source = '') {
  const slide = presentation.slides.add();
  slide.background.fill = C.white;
  rect(slide, 0, 0, 14, 720, C.navy, 'none', 0);
  text(slide, title, 58, 30, 1165, 60, { size: 40, color: C.navy, bold: true, valign: 'middle' });
  rect(slide, 60, 100, 1160, 3, C.blue, 'none', 0);
  rect(slide, 60, 680, 1160, 1.5, C.line, 'none', 0);
  text(slide, source, 62, 687, 1080, 22, { size: 13, color: C.gray, valign: 'middle' });
  text(slide, String(no).padStart(2, '0'), 1160, 685, 58, 24, { size: 16, color: C.gray, bold: true, align: 'right', valign: 'middle' });
  return slide;
}
function notes(slide, body, refs = []) {
  const urls = refs.length ? `\n\n一次資料:\n${refs.map(x => `- ${x}`).join('\n')}` : '';
  slide.speakerNotes.textFrame.setText(`${body}${urls}`);
}
function smallMetric(slide, x, y, w, h, title, value, unit, tint = C.pale, valueColor = C.navy) {
  rect(slide, x, y, w, h, tint, 'none', 0, true);
  const compact = h < 100;
  text(slide, title, x + 12, y + 6, w - 24, 22, { size: compact ? 15 : 16, color: C.gray, bold: true });
  text(slide, value, x + 12, y + 29, w - 24, 38, { size: compact ? 26 : 30, color: valueColor, bold: true, valign: 'middle' });
  if (unit) text(slide, unit, x + 12, y + h - 19, w - 24, 15, { size: compact ? 11 : 13, color: C.gray, valign: 'middle' });
}
function cell(slide, value, x, y, w, h, opts = {}) {
  rect(slide, x, y, w, h, opts.fill ?? C.white, opts.border ?? C.line, opts.borderWidth ?? 1, false);
  text(slide, value, x + 8, y + 3, w - 16, h - 6, { size: opts.size ?? 18, color: opts.color ?? C.ink, bold: opts.bold ?? false, align: opts.align ?? 'left', valign: 'middle' });
}
function bulletList(slide, items, x, y, w, lineH, opts = {}) {
  items.forEach((item, i) => {
    addShape(slide, 'ellipse', x, y + i * lineH + 8, 9, 9, opts.dot ?? C.blue, 'none', 0);
    text(slide, item, x + 22, y + i * lineH, w - 22, lineH - 2, { size: opts.size ?? 21, color: opts.color ?? C.ink, valign: 'middle' });
  });
}
function flowNode(slide, x, y, w, h, title, subtitle = '', fill = C.pale, accent = C.blue) {
  rect(slide, x, y, w, h, fill, C.line, 1, true);
  rect(slide, x, y, 7, h, accent, 'none', 0);
  text(slide, title, x + 17, y + 10, w - 28, subtitle ? 30 : h - 20, { size: 22, bold: true, color: C.navy, valign: subtitle ? 'middle' : 'middle' });
  if (subtitle) text(slide, subtitle, x + 17, y + 44, w - 28, h - 48, { size: 16, color: C.gray, valign: 'top' });
}

// 1. Cover
{
  const s = presentation.slides.add(); s.background.fill = C.white;
  rect(s, 0, 0, 18, 720, C.navy, 'none', 0);
  rect(s, 72, 92, 94, 8, C.teal, 'none', 0);
  text(s, 'Depth Anything V2を用いた\nUAV稲画像からの高さ推定・3D復元に向けた\n予備的検討', 72, 145, 1090, 220, { size: 46, bold: true, color: C.navy, lineSpacing: 1.0 });
  text(s, '一般環境評価から実稲UAV画像・CHM比較まで', 76, 385, 1020, 44, { size: 27, color: C.blue, bold: true, valign: 'middle' });
  text(s, '卒業研究 発表資料', 76, 515, 400, 36, { size: 21, color: C.gray });
  text(s, '発表者：＿＿＿＿＿＿＿＿　　所属：＿＿＿＿＿＿＿＿', 76, 565, 680, 34, { size: 19, color: C.gray });
  // editable visual motif
  flowNode(s, 820, 470, 112, 76, 'UAV', 'RGB', C.sky, C.blue); arrow(s, 945, 495, 44, 28, C.teal);
  flowNode(s, 998, 470, 112, 76, 'DA2', 'depth', C.tealPale, C.teal); arrow(s, 1120, 495, 42, 28, C.orange);
  flowNode(s, 1168, 470, 84, 76, '稲高', '', C.orangePale, C.orange);
  text(s, '2026', 1120, 630, 112, 24, { size: 17, color: C.gray, align: 'right' });
  notes(s, 'この発表では、Depth Anything V2の深度出力が稲高計測にそのまま使えるかを段階的に確認した予備実験を紹介します。');
}

// 2. Background / target
{
  const s = base('研究背景と最終目標', 2, '本研究の位置づけ：画像から深度を得ることと、地面基準の作物高を測ることは異なる。');
  text(s, '最終目標', 78, 137, 270, 36, { size: 24, bold: true, color: C.blue });
  text(s, 'UAV RGB画像から\n稲の高さと3D形状を推定する', 78, 177, 530, 108, { size: 34, bold: true, color: C.navy, valign: 'middle' });
  rect(s, 78, 316, 520, 126, C.orangePale, 'none', 0, true);
  text(s, 'カメラからの距離（depth）\n≠ 地面からの稲高', 104, 333, 470, 87, { size: 30, bold: true, color: C.red, align: 'center', valign: 'middle' });
  text(s, '地面基準・撮影幾何・実測スケールが必要', 90, 462, 500, 32, { size: 22, color: C.ink, align: 'center', bold: true });
  flowNode(s, 695, 153, 440, 76, 'UAV RGB', '稲群落を撮影', C.sky, C.blue);
  arrow(s, 872, 238, 58, 32, C.blue);
  flowNode(s, 695, 288, 440, 76, 'Depth Anything V2', '画素ごとの深度表現', C.tealPale, C.teal);
  arrow(s, 872, 373, 58, 32, C.blue);
  flowNode(s, 695, 423, 440, 76, '幾何・スケール補正', 'camera pose / ground plane / 実測高', C.orangePale, C.orange);
  arrow(s, 872, 508, 58, 32, C.blue);
  flowNode(s, 695, 558, 440, 74, '稲高・3D復元', '研究全体の到達目標', C.pale, C.navy);
  notes(s, '深度推定の出力は、画素のカメラ距離または相対的な順序です。稲高へ変換するには地面の位置、カメラの姿勢と尺度が必要です。');
}

// 3. What DA2 outputs
{
  const s = base('Depth Anything V2が出力する深度', 3, '[1] Yang et al., Depth Anything V2, NeurIPS 2024.');
  rect(s, 80, 138, 1120, 82, C.pale, 'none', 0, true);
  text(s, 'RGB画像', 104, 159, 196, 38, { size: 25, bold: true, align: 'center', valign: 'middle' });
  arrow(s, 320, 163, 58, 28, C.blue);
  text(s, 'Depth Anything V2', 395, 154, 290, 46, { size: 26, bold: true, align: 'center', valign: 'middle', color: C.navy });
  arrow(s, 703, 163, 58, 28, C.blue);
  text(s, '深度マップ', 783, 159, 220, 38, { size: 25, bold: true, align: 'center', valign: 'middle' });
  // metric / relative
  rect(s, 82, 257, 526, 210, C.sky, 'none', 0, true);
  label(s, 'Metric depth', 106, 277, 160, C.white, C.blue);
  text(s, '距離をメートル単位で出力', 106, 321, 438, 40, { size: 25, bold: true, color: C.navy });
  bulletList(s, ['モデル・学習domainの尺度に依存', 'カメラ距離であり、稲高そのものではない'], 112, 375, 455, 38, { size: 19, dot: C.blue });
  rect(s, 672, 257, 526, 210, C.tealPale, 'none', 0, true);
  label(s, 'Relative depth', 696, 277, 170, C.white, C.teal);
  text(s, '画面内の相対的な前後関係', 696, 321, 450, 40, { size: 25, bold: true, color: C.navy });
  bulletList(s, ['絶対尺度・メートル単位を持たない', '局所的な高低順序が残るかを別途評価'], 702, 375, 455, 38, { size: 19, dot: C.teal });
  rect(s, 82, 505, 1116, 106, C.orangePale, 'none', 0, true);
  text(s, 'この予備検討の問い', 110, 520, 230, 30, { size: 20, color: C.orange, bold: true });
  text(s, '作物domainで出力が保たれるか？　　局所差分は高さ差を表すか？', 110, 554, 1040, 38, { size: 25, bold: true, color: C.navy, align: 'center' });
  notes(s, 'DA2にはmetric depthとrelative depthの出力があります。相対深度をGTに合わせてscale+shiftした値は診断用途であり、未知データでの自動校正を意味しません。', ['https://arxiv.org/abs/2406.09414', 'https://github.com/DepthAnything/Depth-Anything-V2']);
}

// 4. Study sequence
{
  const s = base('予備実験の流れ：次の問いへ進む', 4, '比較対象とGTの条件が異なるため、各datasetの値を単純なランキングには用いない。');
  const xs = [80, 370, 660, 950];
  const items = [
    ['一般環境', 'KITTI\nmetric depth', C.sky, C.blue],
    ['GT depth', 'ETH3D\nRGB–depth対応', C.tealPale, C.teal],
    ['3D化', 'camera intrinsics\n→ point cloud', C.pale, C.navy],
    ['稲UAV画像', 'LUMS\n適用可能性', C.orangePale, C.orange],
    ['UAV作物＋GT', 'UAV3DCrop\n52 images', C.sky, C.blue],
    ['実際の稲＋CHM', 'AirMeasurer\nrelative depth比較', C.tealPale, C.teal],
    ['可能性と限界', '高さ推定の\n条件を整理', C.orangePale, C.orange],
  ];
  const coords = [[80,160],[370,160],[660,160],[950,160],[950,385],[660,385],[370,385]];
  items.forEach((d,i)=>flowNode(s,coords[i][0],coords[i][1],230,106,d[0],d[1],d[2],d[3]));
  arrow(s, 320, 196, 42, 24); arrow(s, 610, 196, 42, 24); arrow(s, 900, 196, 42, 24);
  // downward and serpentine links
  addShape(s, 'downArrow', 1050, 282, 34, 68, C.blue, 'none', 0);
  arrow(s, 900, 421, 42, 24, C.blue); arrow(s, 610, 421, 42, 24, C.blue); arrow(s, 320, 421, 42, 24, C.blue);
  rect(s, 80, 550, 1075, 66, C.pale, 'none', 0, true);
  text(s, 'ストーリー：適用できる？ → 深度を評価できる？ → 3D化できる？ → 稲高と対応する？', 102, 565, 1030, 36, { size: 22, bold: true, color: C.navy, align: 'center', valign: 'middle' });
  notes(s, '一般道路から始めて評価基盤を確認し、次に点群化、稲画像への適用、作物UAVのGT深度、実稲のCHMへ進みました。各段階で別の研究上の問いを確認しています。');
}

// 5. KITTI
{
  const s = base('KITTI：距離が伸びるほど誤差が拡大', 5, '[2] KITTI benchmark; 50 images, pooled valid pixels.');
  await image(s, 'outputs/kitti/representatives/kitti_median_example.png', 76, 132, 715, 178, 'KITTI RGB・GT・予測・誤差の代表例');
  await image(s, 'outputs/kitti/plots/error_by_depth.png', 76, 332, 715, 230, 'KITTIのGT距離帯別誤差グラフ');
  text(s, '距離帯別 MAE (m)', 832, 134, 365, 32, { size: 20, bold: true, color: C.navy });
  const bands = [['0–10 m',0.7002],['10–20 m',1.7930],['20–40 m',4.7977],['40–60 m',13.7114],['60–80 m',20.1499]];
  const max = 20.1499;
  bands.forEach((b,i)=>{
    const yy=180+i*48;
    text(s,b[0],832,yy,92,24,{size:17,color:C.ink,valign:'middle'});
    rect(s,925,yy+3,Math.max(6,195*b[1]/max),20,i===4?C.orange:C.blue,'none',0,true);
    text(s,b[1].toFixed(2),1127,yy,76,24,{size:18,bold:true,color:i===4?C.orange:C.navy,align:'right',valign:'middle'});
  });
  smallMetric(s, 830, 445, 115, 108, 'MAE', '2.592', 'm');
  smallMetric(s, 957, 445, 115, 108, 'RMSE', '5.469', 'm');
  smallMetric(s, 1084, 445, 115, 108, 'AbsRel', '0.128', '', C.pale);
  rect(s, 831, 572, 368, 56, C.orangePale, 'none', 0, true);
  text(s, '50画像：遠距離帯で誤差が大きく増加', 850, 583, 331, 34, { size: 19, color: C.red, bold: true, align: 'center', valign: 'middle' });
  notes(s, '50画像・有効画素をpoolして集計。GT距離帯MAEは0–10 m 0.7002、10–20 m 1.7930、20–40 m 4.7977、40–60 m 13.7114、60–80 m 20.1499 m。屋外道路domainでの確認で、作物domainの性能を意味しません。', ['https://www.cvlibs.net/datasets/kitti/', 'https://www.cvlibs.net/publications/Geiger2012CVPR.pdf']);
}

// 6. ETH3D point cloud
{
  const s = base('ETH3D：深度から点群を再構成', 6, '[3] ETH3D; one indoor image for this preliminary pipeline check.');
  await image(s, 'outputs/eth3d_gt/im0_depth_comparison.png', 72, 138, 760, 190, 'ETH3DのRGB・GT depth・予測depth・誤差');
  text(s, 'RGB + Depth + Camera Intrinsics', 95, 348, 685, 32, { size: 22, bold: true, color: C.navy, align: 'center' });
  // process at right
  flowNode(s, 875, 150, 310, 80, 'RGB画像', 'rectified camera view', C.sky, C.blue);
  arrow(s, 1008, 237, 48, 25, C.blue);
  flowNode(s, 875, 274, 310, 80, 'Depth × Intrinsics', '各pixelをXYZへ逆投影', C.tealPale, C.teal);
  arrow(s, 1008, 361, 48, 25, C.blue);
  flowNode(s, 875, 398, 310, 80, '予測 / GT point cloud', 'PLYを同一camera座標に生成', C.pale, C.navy);
  arrow(s, 1008, 485, 48, 25, C.blue);
  flowNode(s, 875, 522, 310, 80, 'Chamfer-like mean NN', '対称平均 0.913 m', C.orangePale, C.orange);
  smallMetric(s, 79, 416, 223, 116, 'GT depth MAE', '1.437', 'm');
  smallMetric(s, 319, 416, 223, 116, 'GT depth RMSE', '1.893', 'm');
  smallMetric(s, 559, 416, 223, 116, '有効pixel', '343,531', '1 image');
  text(s, '深度→3Dの処理経路を確認。稲画像の精度評価ではない。', 90, 562, 680, 42, { size: 21, bold: true, color: C.gray, align: 'center', valign: 'middle' });
  notes(s, 'ETH3Dの1画像でdisparity由来GT depthと予測を比較し、intrinsicsを使った点群を生成しました。予測115,395点・GT 86,105点からそれぞれ50,000点sample。ICP等のalignmentなしで、予測→GT NN平均0.6751m、GT→予測1.1517m、対称平均0.9134m。', ['https://eth3d.ethz.ch/', 'https://openaccess.thecvf.com/content_cvpr_2017/papers/Schops_A_Multi-View_Stereo_CVPR_2017_paper.pdf']);
}

// 7. LUMS rice
{
  const s = base('LUMS：稲UAV画像への適用', 7, '[4] LUMS rice UAV dataset; 9 images (early / middle / late, n=3 each).');
  await image(s, 'outputs/rice_uav/report_figures/figure_2_rgb_relative_metric.png', 65, 132, 700, 475, 'early・middle・lateのRGB、relative depth、metric depth');
  text(s, '確認できたこと', 795, 145, 375, 30, { size: 23, bold: true, color: C.teal });
  bulletList(s, ['稲画像でもdepth mapを生成', 'Metric中央値は約5.26–6.82 m', '撮影高度に近い値でも精度証明ではない'], 800, 185, 374, 44, { size: 18, dot: C.teal });
  rect(s, 792, 345, 405, 245, C.orangePale, 'none', 0, true);
  text(s, '評価上の限界', 814, 360, 350, 30, { size: 22, color: C.orange, bold: true });
  bulletList(s, ['dense canopy / ground visibility', '黄化・褐色葉、water、shadow', 'vegetation mask・camera pose不足', 'frame-level GT不足'], 817, 402, 350, 39, { size: 17, dot: C.orange });
  text(s, 'canopy–ground depth差は proxy。稲高として扱わない。', 86, 623, 1090, 35, { size: 22, color: C.red, bold: true, align: 'center', valign: 'middle' });
  notes(s, 'Early/middle/lateを各3枚、合計9枚に適用。canopy-ground depth differenceは地面・canopyの対応やframe-level GTがないためproxyにとどまり、稲高とは呼びません。Lateの候補地面値は最終methodでNO_GROUND_REFERENCEとして除外されています。', ['https://github.com/LUMS-WIT/infrastructure-free-uav-phenotyping', 'https://doi.org/10.1016/j.compag.2026.111677']);
}

// 8. UAV3DCrop
{
  const s = base('UAV3DCrop：作物UAVのGT depth評価', 8, '[5] UAV3DCrop: photogrammetry MVS camera-frame z-depth; not hand-measured rice height.');
  await image(s, 'outputs/uav3dcrop/report_figures/figure_2_rgb_gt_pred.png', 72, 142, 645, 390, 'Wheat代表sceneのRGB、GT depth、metric prediction');
  label(s, '代表例：Wheat（図版はWheat subset）', 130, 535, 470, C.orangePale, C.orange);
  text(s, '評価セット：52画像', 762, 137, 411, 34, { size: 24, bold: true, color: C.navy });
  const rows = [
    ['Crop','Wheat 20','Oat 16','Corn 16'],
    ['View','Nadir 26','Oblique 26','Total 52'],
  ];
  rows.forEach((r,ri)=>r.forEach((v,ci)=>cell(s,v,762+ci*108,184+ri*43,108,42,{fill:ci===0?C.sky:C.white,size:16,bold:ci===0,align:'center'})));
  smallMetric(s, 762, 302, 125, 112, 'MAE', '13.579', 'm', C.redPale, C.red);
  smallMetric(s, 897, 302, 125, 112, 'RMSE', '14.714', 'm', C.pale, C.navy);
  smallMetric(s, 1032, 302, 145, 112, 'AbsRel', '0.657', '', C.pale, C.navy);
  rect(s, 760, 432, 420, 84, C.redPale, 'none', 0, true);
  text(s, 'Bias −13.579 m\nraw metric depthの大きなdomain bias', 783, 441, 370, 66, { size: 21, bold: true, color: C.red, align: 'center', valign: 'middle' });
  text(s, 'KITTIとの順位付けではなく、撮影視点・作物domainで性能が大きく変化した結果。', 761, 542, 430, 66, { size: 18, bold: true, color: C.ink, align: 'center', valign: 'middle' });
  notes(s, '全52画像はWheat20 + Oat16 + Corn16、nadir26 + oblique26のcanonical uav3dcrop_all集計。MAE/RMSE/AbsRel/Biasはpooled。提示したRGB/GT/pred図は既存のWheat-only図なので代表例と明記。KITTIとの直接ランキングはしません。', ['https://link-dev.github.io/UAV3DCrop/', 'https://arxiv.org/abs/2608.06404', 'https://github.com/Link-dev/UAV3DCrop']);
}

// 9. Local difference trap
{
  const s = base('Local Depth Differenceの落とし穴', 9, 'UAV3DCrop all, 52 images; pair-pooled 32 px pairs, n=1,040,000.');
  await image(s, 'outputs/uav3dcrop/all_local_analysis/report_figures/figure_4_zero_baseline_comparison.png', 66, 145, 564, 355, '32px local difference MAEとzero baselineの比較');
  text(s, '32 px pair（m）', 672, 143, 475, 30, { size: 21, bold: true, color: C.navy });
  const heads = ['指標','値']; heads.forEach((v,i)=>cell(s,v,672+i*235,182,235,39,{fill:C.navy,color:C.white,size:17,bold:true,align:i? 'right':'left'}));
  const data = [
    ['|GT差分| 中央値','0.015625'],
    ['zero baseline MAE','0.040656'],
    ['metric raw MAE','0.053259'],
    ['relative aligned MAE','0.040874'],
  ];
  data.forEach((r,i)=>{
    cell(s,r[0],672,221+i*48,235,48,{size:17,fill:i===1?C.tealPale:C.white,bold:i===1});
    cell(s,r[1],907,221+i*48,235,48,{size:19,align:'right',fill:i===1?C.tealPale:C.white,bold:true,color:i===1?C.teal:C.ink});
  });
  rect(s, 78, 535, 1120, 92, C.redPale, 'none', 0, true);
  text(s, 'Local MAEが約4～5 cmでも、高さを4～5 cm精度で測定できることを意味しない', 104, 552, 1070, 55, { size: 27, color: C.red, bold: true, align: 'center', valign: 'middle' });
  notes(s, '|GT差分|中央値0.015625 m、mean absolute GT difference = zero-baseline MAE = 0.0406557 m。metric raw MAEは0.0532587、relative scale+shiftは0.0408736 mでzeroより0.000218m悪い。relative alignmentはGTを使う診断であり、実運用で利用できる校正ではありません。Local differenceはcrop GTのcamera-frame Z差。', ['https://arxiv.org/abs/2608.06404']);
}

// 10. Difference magnitude by target
{
  const s = base('高さ差が大きいほど順序は合うが、振幅は縮む', 10, 'UAV3DCrop relative scale+shift; target bins ±20%, pooled across 8–128 px separations.');
  text(s, 'GT差分target', 96, 145, 220, 30, { size: 19, bold: true, color: C.gray });
  text(s, 'MAE (m)', 355, 145, 155, 30, { size: 19, bold: true, color: C.gray, align: 'center' });
  text(s, 'Sign agreement', 557, 145, 210, 30, { size: 19, bold: true, color: C.gray, align: 'center' });
  text(s, '回帰 slope（理想=1）', 863, 145, 265, 30, { size: 19, bold: true, color: C.gray, align: 'center' });
  const d = [
    ['0.1 m','0.0922','0.652','0.178',0.178],
    ['0.2 m','0.1755','0.736','0.211',0.211],
    ['0.5 m','0.3720','0.837','0.306',0.306],
    ['1.0 m','0.7516','0.883','0.249',0.249],
  ];
  d.forEach((r,i)=>{
    const yy=193+i*88;
    rect(s,77,yy,1125,72,i%2?C.white:C.pale,C.line,1,true);
    text(s,r[0],97,yy+16,190,38,{size:25,bold:true,color:C.navy,valign:'middle'});
    text(s,r[1],353,yy+17,158,38,{size:23,bold:true,color:C.ink,align:'center',valign:'middle'});
    const sign=Number(r[2]); rect(s,565,yy+22,180,22,C.line,'none',0,true); rect(s,565,yy+22,180*sign,22,C.teal,'none',0,true);
    text(s,r[2],756,yy+16,90,34,{size:21,bold:true,color:C.teal,align:'right',valign:'middle'});
    rect(s,884,yy+22,176,22,C.line,'none',0,true); rect(s,884,yy+22,176*r[4],22,C.orange,'none',0,true);
    text(s,r[3],1071,yy+16,100,34,{size:21,bold:true,color:C.orange,align:'right',valign:'middle'});
  });
  rect(s, 80, 566, 1120, 64, C.orangePale, 'none', 0, true);
  text(s, '大きな差ほど符号は合いやすい。一方、slope 0.18–0.31は差分振幅の強い圧縮を示す。', 101, 578, 1080, 38, { size: 22, color: C.red, bold: true, align: 'center', valign: 'middle' });
  notes(s, 'Rows use the report target-magnitude definition: |GT Δ| within ±20% around 0.1, 0.2, 0.5, and 1.0m, pooled over all tested separations (not only 32px). The GT is photogrammetry camera Z depth, not measured plant height. Sign agreement rises from .652 to .883; regression slopes remain .178–.306.', ['https://arxiv.org/abs/2608.06404']);
}

// 11. AirMeasurer
{
  const s = base('AirMeasurer：実稲画像とCHMの比較', 11, '[6] AirMeasurer rice field, 2022-05-05; relative DA2 vs approximate CHM from orthomosaic + LAS.');
  await image(s, 'outputs/airmeasurer/report_figures/figure_2_rgb_chm_relative.png', 77, 136, 1120, 304, 'AirMeasurer稲のRGB orthomosaic、CHM、relative depth');
  // native process line
  const yy=462;
  flowNode(s, 100, yy, 248, 66, 'Orthomosaic + LAS', '', C.sky, C.blue);
  arrow(s, 365, yy+19, 48, 27, C.blue);
  flowNode(s, 423, yy, 210, 66, 'DSM / DEM', '', C.pale, C.navy);
  arrow(s, 648, yy+19, 48, 27, C.blue);
  flowNode(s, 704, yy, 170, 66, 'CHM', '', C.tealPale, C.teal);
  arrow(s, 889, yy+19, 48, 27, C.blue);
  flowNode(s, 945, yy, 252, 66, 'DA2 relative depth', '', C.orangePale, C.orange);
  smallMetric(s, 138, 550, 210, 86, 'Pearson', '0.258', '', C.pale);
  smallMetric(s, 370, 550, 210, 86, 'Spearman', '0.210', '', C.pale);
  smallMetric(s, 602, 550, 260, 86, 'DA2 aligned MAE', '0.0791', 'm (RMSE 0.1450 m)', C.pale);
  smallMetric(s, 884, 550, 260, 86, 'constant baseline MAE', '0.0563', 'm (RMSE 0.1588 m)', C.orangePale, C.orange);
  text(s, 'pixel-level MAEではconstant baselineを上回らず（RMSEはDA2が低い）', 110, 640, 1060, 28, { size: 18, color: C.red, bold: true, align: 'center', valign: 'middle' });
  notes(s, '実際の稲の1日分（20220505）を使ったpixel comparison。sample 300,000、valid 4,958,593。relativeとCHMのPearson .257874 / Spearman .209845。GT-assisted scale+shift aligned MAE/RMSE .079086/.144965m。CHM sample median constant baseline MAE/RMSE .056350/.158830m。DA2はMAEでconstant baselineを上回らず、RMSEでは下回る。CHMは簡略化したSOR+CSF/rasterizationによる近似で公式AirMeasurer GUIと同一ではない。入力はorthomosaicで単一raw camera frameではありません。', ['https://doi.org/10.1111/nph.18314', 'https://github.com/The-Zhou-Lab/UAV-AirMeasurer/releases/tag/V2.0.2']);
}

// 12. AirMeasurer local
{
  const s = base('AirMeasurer：高低順序と高さ差の振幅', 12, 'Target bins are ±20% pooled across separations; GT is local CHM difference.');
  await image(s, 'outputs/airmeasurer/report_figures/figure_7_local_height_difference.png', 76, 148, 500, 438, 'DA2 local relative depth difference vs CHM difference');
  text(s, 'GT差分target', 614, 150, 140, 30, { size: 17, bold: true, color: C.gray });
  text(s, 'Pearson', 762, 150, 105, 30, { size: 17, bold: true, color: C.gray, align: 'center' });
  text(s, 'Sign', 880, 150, 86, 30, { size: 17, bold: true, color: C.gray, align: 'center' });
  text(s, 'slope', 986, 150, 94, 30, { size: 17, bold: true, color: C.gray, align: 'center' });
  const rows=[['0.1 m','.385','.683','.099'],['0.2 m','.483','.741','.070'],['0.5 m','.696','.888','.055'],['1.0 m','.824','.962','.054']];
  rows.forEach((r,i)=>{
    const y=193+i*65;
    cell(s,r[0],613,y,142,58,{size:18,bold:true,fill:i%2?C.white:C.pale});
    cell(s,r[1],755,y,112,58,{size:19,align:'center',bold:true,fill:i%2?C.white:C.pale,color:C.blue});
    cell(s,r[2],867,y,110,58,{size:19,align:'center',bold:true,fill:i%2?C.white:C.pale,color:C.teal});
    cell(s,r[3],977,y,112,58,{size:19,align:'center',bold:true,fill:i%2?C.white:C.pale,color:C.orange});
  });
  rect(s, 614, 474, 493, 102, C.orangePale, 'none', 0, true);
  text(s, '差が大きいほど相関・符号一致は上昇。\nslopeは約0.05–0.10で、振幅を強く圧縮。', 635, 486, 450, 76, { size: 21, color: C.red, bold: true, align: 'center', valign: 'middle' });
  text(s, '32 px: MAE 0.0508 m / sign 0.554 / slope 0.0474', 614, 590, 493, 28, { size: 17, color: C.gray, bold: true, align: 'center', valign: 'middle' });
  notes(s, 'Air local 32px all-pair summary: n=64,569, MAE .0507964m, RMSE .1305396, Pearson .444826, Spearman .273283, sign .554222, slope .047368. Table magnitude bins pool across all separations using target ±20%. Magnitude trends are local relative-depth differences compared with CHM; they do not validate absolute rice height.');
}

// 13. Synthesis
{
  const s = base('全実験から分かったこと', 13, 'Evidence spans road, indoor stereo, crop UAV, rice images and CHM; each supports a different claim.');
  const colY=154, colH=397;
  rect(s, 78, colY, 345, colH, C.tealPale, 'none', 0, true);
  label(s, '確認できたこと', 104, colY+18, 205, C.white, C.teal);
  bulletList(s, ['稲UAV画像にも適用可能', 'Metric depth / relative depthを生成', '局所的な高低順序を部分保持', '大きな差ほどsignが合いやすい'], 105, colY+70, 292, 64, { size: 20, dot: C.teal });
  rect(s, 467, colY, 345, colH, C.orangePale, 'none', 0, true);
  label(s, '性能上の課題', 493, colY+18, 184, C.white, C.orange);
  bulletList(s, ['一般環境とcrop domainに差', 'raw metric depthにdomain bias', 'local differenceの振幅圧縮', 'AirMeasurer MAEでconstantに届かず'], 494, colY+70, 292, 64, { size: 19, dot: C.orange });
  rect(s, 856, colY, 345, colH, C.redPale, 'none', 0, true);
  label(s, '未確認のこと', 882, colY+18, 176, C.white, C.red);
  bulletList(s, ['実測稲高との十分な対応', 'raw frameでのGT付き評価', 'plot-level汎化', 'cross-date calibration'], 883, colY+70, 292, 64, { size: 20, dot: C.red });
  rect(s, 100, 585, 1080, 62, C.navy, 'none', 0, true);
  text(s, '深度マップの生成可能性 ≠ 高精度な稲高計測', 122, 597, 1035, 37, { size: 26, bold: true, color: C.white, align: 'center', valign: 'middle' });
  notes(s, '結論では出力の生成可能性、順序の部分保持、振幅の校正可能性、実測稲高との対応を分けて述べます。現在のデータから絶対稲高精度は確認できていません。');
}

// 14. Current conclusion
{
  const s = base('現時点での結論', 14, 'この段階で主張するのは適用可能性と課題の整理まで。');
  rect(s, 86, 145, 1107, 110, C.navy, 'none', 0, true);
  text(s, 'DA2の出力をそのまま稲高として使用しない', 120, 171, 1040, 58, { size: 34, bold: true, color: C.white, align: 'center', valign: 'middle' });
  rect(s, 90, 292, 510, 286, C.tealPale, 'none', 0, true);
  text(s, '確認できた', 117, 312, 410, 34, { size: 24, color: C.teal, bold: true });
  bulletList(s, ['稲画像への推論適用は可能', 'UAV crop domainではraw metricにbias', 'relative depthは局所順序を部分保持'], 119, 367, 448, 55, { size: 20, dot: C.teal });
  rect(s, 680, 292, 510, 286, C.orangePale, 'none', 0, true);
  text(s, 'まだ確認できていない', 707, 312, 440, 34, { size: 24, color: C.orange, bold: true });
  bulletList(s, ['高精度な稲高・実測高との対応', 'raw frame単位のGT付き評価', 'plot-level・別日への汎化'], 709, 367, 448, 55, { size: 20, dot: C.orange });
  text(s, '必要条件：camera geometry ＋ ground plane ＋ 外部scale ＋ 実測高さ', 120, 612, 1040, 38, { size: 23, bold: true, color: C.navy, align: 'center', valign: 'middle' });
  notes(s, 'Height estimation needs camera geometry, a ground reference, scale information, and measured heights. Current results do not establish sufficient alignment to ground truth rice height.');
}

// 15. Future workflow + references
{
  const s = base('今後の研究フロー', 15, '研究室で取得するUAVデータで、独立した稲高評価と校正を行う。');
  const y1=141, y2=270;
  label(s, '入力・校正用データ', 84, 108, 192, C.sky, C.blue);
  flowNode(s, 77, y1, 225, 75, 'UAV RGB', 'intrinsics', C.sky, C.blue);
  arrow(s, 310, y1+24, 46, 27);
  flowNode(s, 365, y1, 264, 75, 'Camera Pose / AGL', 'ground plane', C.pale, C.navy);
  arrow(s, 637, y1+24, 46, 27);
  flowNode(s, 692, y1, 233, 75, '実測稲高', 'plot / date reference', C.orangePale, C.orange);
  arrow(s, 934, y1+24, 46, 27);
  flowNode(s, 989, y1, 204, 75, 'Depth Anything V2', '', C.tealPale, C.teal);
  // second row reversed conceptually from input to output
  flowNode(s, 989, y2, 204, 75, 'DA2 depth', '', C.tealPale, C.teal);
  addShape(s, 'downArrow', 1073, 226, 34, 34, C.teal, 'none', 0);
  addShape(s, 'leftArrow', 934, y2+24, 46, 27, C.orange, 'none', 0);
  flowNode(s, 692, y2, 233, 75, 'Calibration', 'geometry correction', C.pale, C.navy);
  addShape(s, 'leftArrow', 637, y2+24, 46, 27, C.blue, 'none', 0);
  flowNode(s, 365, y2, 264, 75, '稲高推定', 'plot / date evaluation', C.orangePale, C.orange);
  addShape(s, 'leftArrow', 310, y2+24, 46, 27, C.blue, 'none', 0);
  flowNode(s, 77, y2, 225, 75, '3D復元', '稲群落 point cloud', C.sky, C.blue);
  text(s, '独立評価：plot-level / cross-date / 実測高hold-out', 82, 370, 1115, 33, { size: 21, bold: true, color: C.navy, align: 'center', valign: 'middle' });
  rect(s, 80, 429, 1118, 2, C.line, 'none', 0);
  text(s, '一次資料', 83, 444, 180, 28, { size: 19, bold: true, color: C.blue });
  const refsLeft = [
    '[1] Depth Anything V2 — Yang et al., NeurIPS 2024',
    '[2] KITTI — Geiger et al., CVPR 2012',
    '[3] ETH3D — Schöps et al., CVPR 2017',
  ];
  const refsRight = [
    '[4] LUMS rice UAV — Chaudhry et al., 2026',
    '[5] UAV3DCrop — Zhou et al., 2026',
    '[6] AirMeasurer — Sun et al., New Phytologist, 2022',
  ];
  refsLeft.forEach((r,i)=>text(s,r,92,480+i*42,520,32,{size:16,color:C.ink,valign:'middle'}));
  refsRight.forEach((r,i)=>text(s,r,662,480+i*42,526,32,{size:16,color:C.ink,valign:'middle'}));
  notes(s, '今後は、intrinsics、camera pose/AGL、地面基準、plotごとの実測稲高を同期取得します。DA2深度をgeometry-awareに補正し、校正用とは独立したplot/dateのhold-outで評価します。', [
    'https://arxiv.org/abs/2406.09414',
    'https://www.cvlibs.net/publications/Geiger2012CVPR.pdf',
    'https://openaccess.thecvf.com/content_cvpr_2017/papers/Schops_A_Multi-View_Stereo_CVPR_2017_paper.pdf',
    'https://github.com/LUMS-WIT/infrastructure-free-uav-phenotyping',
    'https://arxiv.org/abs/2608.06404',
    'https://doi.org/10.1111/nph.18314',
  ]);
}

// Persist a candidate, render every slide as PNG for visual QA, then finalize.
const candidate = path.join(BUILD, 'candidate.pptx');
await (await PresentationFile.exportPptx(presentation)).save(candidate);
const previewDir = path.join(BUILD, 'preview');
await fs.mkdir(previewDir, { recursive: true });
for (let i = 0; i < presentation.slides.items.length; i++) {
  const slide = presentation.slides.items[i];
  const png = await presentation.export({ slide, format: 'png', scale: 1 });
  await fs.writeFile(path.join(previewDir, `slide-${String(i+1).padStart(2,'0')}.png`), new Uint8Array(await png.arrayBuffer()));
}
const { finalizePresentation } = await import(pathToFileURL(path.join(SKILL_DIR, 'container_tools/artifact_tool_utils.mjs')).href);
const result = await finalizePresentation({
  explicitTotalSlideCount: 15,
  requiredNativeTableOwnerSlides: [],
  requiredNativeChartOwnerSlides: [],
  workspaceDir: PROJECT,
  candidatePath: candidate,
  finalPath: VALIDATED_FINAL,
  pythonExecutable: PYTHON,
  integrityValidatorPath: path.join(SKILL_DIR, 'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath: path.join(SKILL_DIR, 'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs: ['--expected-slide-size-emu', '12192000,6858000', '--validate-bullet-geometry', '--validate-heading-fit'],
  fontPolicy: { basis: 'design', families: [FONT] },
  verifyArtifactToolImport: true,
  receiptPath: path.join(BUILD, 'depth_anything_v2_rice_research_v3.validation.json'),
});
await fs.copyFile(VALIDATED_FINAL, FINAL);
console.log(JSON.stringify({ candidate, validatedFinal: VALIDATED_FINAL, deliveredFinal: FINAL, slideCount: presentation.slides.items.length, result }, null, 2));
