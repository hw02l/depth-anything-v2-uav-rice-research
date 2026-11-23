#!/usr/bin/env python3
"""Render a dependency-free SVG comparison from the frozen condition summary."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

COLORS = {
    "epoch20-B0": "#94a3b8",
    "epoch20-B3": "#6366f1",
    "E3-B0": "#d97706",
    "C0-P0": "#0f766e",
    "C0-B3": "#2563eb",
    "E3-B3": "#dc2626",
}
ORDER = ["epoch20-B0", "epoch20-B3", "E3-B0", "C0-P0", "C0-B3", "E3-B3"]
PLOTS = [
    ("絶対RMSE (m)", "absolute_rmse", 0.72, 0.81, [0.72, 0.75, 0.78, 0.81]),
    ("全有効14px MAE (m)", "full14_mae", 0.055, 0.066, [0.055, 0.060, 0.065]),
    ("GT差≥0.20m・14px MAE (m)", "high14_mae", 0.30, 0.35, [0.30, 0.32, 0.34, 0.35]),
]

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    values: dict[str, dict[str, float]] = {name: {} for name in ORDER}
    with args.summary.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["condition"] in values:
                values[row["condition"]][row["metric"]] = float(row["pooled"])
    missing = [(c, metric) for c in ORDER for _, metric, *_ in PLOTS if metric not in values[c]]
    if missing:
        raise ValueError(f"Missing pooled metrics: {missing}")

    width, height = 1820, 650
    panel_lefts = [330, 840, 1350]
    plot_width = 360
    row_y = [150 + 56 * i for i in range(len(ORDER))]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:"Yu Gothic UI","Noto Sans JP",sans-serif;fill:#172033}.title{font-size:27px;font-weight:700}.panel{font-size:19px;font-weight:700}.small{font-size:13px;fill:#475569}.label{font-size:16px}.value{font-size:14px;font-weight:600}.tick{font-size:12px;fill:#475569}</style>',
        '<text x="28" y="42" class="title">Validation128：6条件のpooled指標</text>',
        '<text x="28" y="70" class="small">点推定。各パネルの横軸範囲は異なる（読み取りやすさのため一部を切り詰めて表示）。</text>',
    ]
    for i, condition in enumerate(ORDER):
        y = row_y[i]
        parts.append(f'<circle cx="38" cy="{y-5}" r="7" fill="{COLORS[condition]}"/>')
        parts.append(f'<text x="58" y="{y}" class="label">{condition}</text>')
    for panel_idx, (title, metric, lo, hi, ticks) in enumerate(PLOTS):
        left = panel_lefts[panel_idx]
        parts.append(f'<text x="{left}" y="104" class="panel">{title}</text>')
        for tick in ticks:
            x = left + (tick - lo) / (hi - lo) * plot_width
            parts.append(f'<line x1="{x:.1f}" y1="122" x2="{x:.1f}" y2="490" stroke="#e2e8f0" stroke-width="1"/>')
            parts.append(f'<text x="{x:.1f}" y="520" text-anchor="middle" class="tick">{tick:.3f}</text>')
        parts.append(f'<line x1="{left}" y1="490" x2="{left+plot_width}" y2="490" stroke="#64748b" stroke-width="1.5"/>')
        for i, condition in enumerate(ORDER):
            y = row_y[i] - 5
            value = values[condition][metric]
            x = left + (value - lo) / (hi - lo) * plot_width
            parts.append(f'<line x1="{left}" y1="{y}" x2="{x:.1f}" y2="{y}" stroke="{COLORS[condition]}" stroke-opacity="0.32" stroke-width="2"/>')
            parts.append(f'<circle cx="{x:.1f}" cy="{y}" r="6.5" fill="{COLORS[condition]}"/>')
            anchor = "end" if x > left + plot_width - 48 else "start"
            tx = x - 10 if anchor == "end" else x + 10
            parts.append(f'<text x="{tx:.1f}" y="{y+5}" text-anchor="{anchor}" class="value">{value:.6f}</text>')
    parts.extend([
        '<text x="28" y="585" class="small">低い値が良い。全14px MAEと高差分14px MAEは異なるpair集合である。信頼区間はpaired_bootstrap.csvを参照。</text>',
        '<text x="28" y="610" class="small">対象：2025/Day047 Validation128。Independent test319は未使用。条件定義とprovenanceはreport_ja.md / launch.jsonを参照。</text>',
        '</svg>',
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(parts) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
