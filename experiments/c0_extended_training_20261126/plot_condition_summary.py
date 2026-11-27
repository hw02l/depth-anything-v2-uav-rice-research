#!/usr/bin/env python3
"""Render the fixed six-condition Validation128 metric summary as an SVG."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

ORDER = [
    "C0-epoch5-P0",
    "C0-epoch5-B3",
    "C0-epoch20-P0",
    "C0-epoch20-B3",
    "initial-model-epoch20-B3",
    "E3-epoch5-B3",
]
COLORS = ["#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628", "#e41a1c"]
SPECS = [
    ("absolute_rmse", "Absolute depth RMSE", "m"),
    ("full14_mae", "All valid 14 px pair MAE", "m"),
    ("high14_mae", "GT |ΔZ| ≥ 0.20 m, 14 px pair MAE", "m"),
]


def render(rows: list[dict[str, str]], destination: Path) -> None:
    values = {(r["condition"], r["metric"]): float(r["pooled"]) for r in rows}
    width, height = 1500, 570
    panel_w, panel_gap = 460, 35
    left0, top, bottom = 50, 95, 390
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="Arial, sans-serif" fill="#202124">',
        '<text x="750" y="38" text-anchor="middle" font-size="22" font-weight="bold">Validation128 pooled metrics</text>',
        '<text x="750" y="64" text-anchor="middle" font-size="14">Fixed six-condition comparison; common sample and mask</text>',
    ]
    for panel, (metric, title, unit) in enumerate(SPECS):
        x0 = left0 + panel * (panel_w + panel_gap)
        vals = [values[(condition, metric)] for condition in ORDER]
        lo, hi = min(vals), max(vals)
        pad = max((hi - lo) * 0.22, 0.0002)
        lo, hi = lo - pad, hi + pad
        plot_l, plot_r = x0 + 58, x0 + panel_w - 14
        plot_t, plot_b = top, bottom
        def sy(v: float) -> float:
            return plot_t + (hi - v) / (hi - lo) * (plot_b - plot_t)
        parts.append(f'<text x="{x0+panel_w/2}" y="88" text-anchor="middle" font-size="17" font-weight="bold">{title}</text>')
        for tick in range(5):
            value = lo + (hi - lo) * tick / 4
            y = sy(value)
            parts.append(f'<line x1="{plot_l}" y1="{y:.1f}" x2="{plot_r}" y2="{y:.1f}" stroke="#e2e5e9"/>')
            parts.append(f'<text x="{plot_l-8}" y="{y+4:.1f}" text-anchor="end" font-size="11">{value:.4f}</text>')
        parts.append(f'<line x1="{plot_l}" y1="{plot_t}" x2="{plot_l}" y2="{plot_b}" stroke="#555"/>')
        parts.append(f'<line x1="{plot_l}" y1="{plot_b}" x2="{plot_r}" y2="{plot_b}" stroke="#555"/>')
        step = (plot_r - plot_l) / len(ORDER)
        for i, condition in enumerate(ORDER):
            x = plot_l + step * (i + 0.5)
            y = sy(values[(condition, metric)])
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{COLORS[i]}" stroke="white" stroke-width="1.5"/>')
            parts.append(f'<text x="{x:.1f}" y="{y-11:.1f}" text-anchor="middle" font-size="11">{values[(condition, metric)]:.6f}</text>')
            parts.append(f'<text x="{x:.1f}" y="{plot_b+18}" text-anchor="end" font-size="10" transform="rotate(-42 {x:.1f} {plot_b+18})">{condition}</text>')
        parts.append(f'<text x="{x0+panel_w/2}" y="{height-18}" text-anchor="middle" font-size="12">Pooled error ({unit}; lower is better)</text>')
    parts.extend(["</g>", "</svg>"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.input.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    expected = {(condition, metric) for condition in ORDER for metric, _, _ in SPECS}
    actual = {(row["condition"], row["metric"]) for row in rows}
    if actual != expected:
        raise ValueError(f"Condition/metric coverage mismatch: missing={expected-actual}, extra={actual-expected}")
    render(rows, args.output)


if __name__ == "__main__":
    main()