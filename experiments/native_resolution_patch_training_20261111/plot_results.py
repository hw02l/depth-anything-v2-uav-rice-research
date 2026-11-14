from __future__ import annotations

import argparse
import csv
from html import escape
from pathlib import Path


COLORS = {"C0": "#356A9A", "F1": "#E58B2A", "F2": "#4B9363"}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def svg_start(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,"Noto Sans JP",sans-serif;fill:#222} .muted{fill:#666}</style>',
        f'<text x="{width/2:.1f}" y="30" text-anchor="middle" font-size="20" font-weight="bold">{escape(title)}</text>',
    ]


def add_text(out: list[str], x: float, y: float, value: object, size: int = 12,
             anchor: str = "middle", color: str = "#222", weight: str = "normal") -> None:
    out.append(f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-size="{size}" fill="{color}" font-weight="{weight}">{escape(str(value))}</text>')


def add_line(out: list[str], x1: float, y1: float, x2: float, y2: float,
             color: str = "#aaa", width: float = 1, dash: str = "") -> None:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    out.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width}"{dash_attr}/>')


def add_rect(out: list[str], x: float, y: float, width: float, height: float,
             color: str, stroke: str = "none") -> None:
    out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" fill="{color}" stroke="{stroke}"/>')


def finish(out: list[str], path: Path) -> None:
    out.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def draw_bar_panel(out: list[str], rows: list[dict[str, str]], x0: float, y0: float,
                   width: float, height: float, field: str, title: str,
                   maximum: float, decimals: int) -> None:
    left, right, top, bottom = x0 + 54, x0 + width - 12, y0 + 52, y0 + height - 74
    add_text(out, (left + right) / 2, y0 + 26, title, 15, weight="bold")
    plot_h = bottom - top
    for tick in range(6):
        value = maximum * tick / 5
        y = bottom - plot_h * tick / 5
        add_line(out, left, y, right, y, "#e4e7ea", 1)
        add_text(out, left - 8, y + 4, f"{value:.{decimals}f}", 10, anchor="end", color="#555")
    add_line(out, left, top, left, bottom, "#555")
    add_line(out, left, bottom, right, bottom, "#555")
    step = (right - left) / len(rows)
    bar_w = step * 0.60
    for i, row in enumerate(rows):
        value = float(row[field])
        x = left + i * step + (step - bar_w) / 2
        bar_h = plot_h * value / maximum
        add_rect(out, x, bottom - bar_h, bar_w, bar_h, COLORS[row["group"]], "#333")
        add_text(out, x + bar_w / 2, bottom - bar_h - 5, f"{value:.{decimals}f}", 9)
        add_text(out, left + (i + 0.5) * step, bottom + 18,
                 f'{row["group"]}-{row["method"]}', 9)
    add_text(out, x0 + 12, (top + bottom) / 2, "m", 11, anchor="start", color="#555")


def make_grid(summary_path: Path, output: Path) -> None:
    rows = read_rows(summary_path)
    expected = {(g, m) for g in ("C0", "F1", "F2") for m in ("P0", "P1", "P2")}
    if {(r["group"], r["method"]) for r in rows} != expected:
        raise ValueError("summary CSV must contain the complete C0/F1/F2 × P0/P1/P2 grid")
    rows.sort(key=lambda r: (("C0", "F1", "F2").index(r["group"]),
                             ("P0", "P1", "P2").index(r["method"])))
    out = svg_start(1260, 620, "Validation128 · 固定epoch 5 · 3×3評価")
    draw_bar_panel(out, rows, 18, 48, 610, 510, "rmse_m", "絶対深度 RMSE", 10.0, 1)
    draw_bar_panel(out, rows, 632, 48, 610, 510, "full14_mae_m", "14px局所全pair MAE", 0.10, 3)
    for i, group in enumerate(("C0", "F1", "F2")):
        x = 475 + i * 120
        add_rect(out, x, 592, 16, 14, COLORS[group])
        add_text(out, x + 23, 604, group, 12, anchor="start")
    finish(out, output)


def draw_effect_panel(out: list[str], rows: list[dict[str, str]], x0: float, y0: float,
                      width: float, height: float, metric: str, title: str,
                      low: float, high: float, digits: int) -> None:
    subset = [r for r in rows if r["metric"] == metric and r["method"] == "P2"
              and r["baseline"] == "C0" and r["estimator"] == "pooled"]
    subset.sort(key=lambda r: ("F1", "F2").index(r["candidate"]))
    if {r["candidate"] for r in subset} != {"F1", "F2"}:
        raise ValueError(f"missing P2 paired effects for {metric}")
    left, right, top, bottom = x0 + 102, x0 + width - 22, y0 + 54, y0 + height - 62
    add_text(out, (left + right) / 2, y0 + 27, title, 14, weight="bold")
    map_x = lambda value: left + (value - low) / (high - low) * (right - left)
    for tick in range(5):
        value = low + (high - low) * tick / 4
        x = map_x(value)
        add_line(out, x, top, x, bottom, "#e4e7ea", 1)
        add_text(out, x, bottom + 22, f"{value:.{digits}f}", 10)
    zero_x = map_x(0.0)
    add_line(out, zero_x, top, zero_x, bottom, "#555", 1.5, "5 4")
    positions = [top + (bottom - top) * 0.33, top + (bottom - top) * 0.72]
    for row, y in zip(subset, positions):
        point = float(row["candidate_minus_baseline"])
        lo, hi = float(row["ci95_low"]), float(row["ci95_high"])
        add_text(out, left - 12, y + 4, row["candidate"], 13, anchor="end", weight="bold")
        add_line(out, map_x(lo), y, map_x(hi), y, "#315f83", 2)
        add_line(out, map_x(lo), y - 6, map_x(lo), y + 6, "#315f83", 2)
        add_line(out, map_x(hi), y - 6, map_x(hi), y + 6, "#315f83", 2)
        out.append(f'<circle cx="{map_x(point):.1f}" cy="{y:.1f}" r="5" fill="#e58b2a" stroke="#222"/>')
        add_text(out, right, y - 12, f"Δ={point:+.4f} [{lo:+.4f}, {hi:+.4f}]", 10, anchor="end")
    add_text(out, (left + right) / 2, bottom + 45, "候補 − C0（画像paired bootstrap 95% CI）", 11)


def make_effects(bootstrap_path: Path, output: Path) -> None:
    rows = read_rows(bootstrap_path)
    out = svg_start(1100, 520, "P2 · F1/F2 − C0 · 対応付き画像bootstrap")
    draw_effect_panel(out, rows, 10, 45, 535, 420, "delta_mae",
                      "主要局所MAE（m）・負値が改善", -0.0002, 0.0016, 4)
    draw_effect_panel(out, rows, 555, 45, 535, 420, "rmse",
                      "絶対深度RMSE差（m）・負値が低下", -1.30, 0.10, 2)
    add_text(out, 550, 500, "10,000 resamples · seed 42 · Validation128 / Day047", 11, color="#555")
    finish(out, output)


def make_scale(scale_path: Path, output: Path) -> None:
    rows = read_rows(scale_path)
    rows.sort(key=lambda r: int(r["stride_px"]))
    width, height = 900, 560
    out = svg_start(width, height, "P2 · 14/28/56/107/214px 局所全pair MAE")
    left, right, top, bottom = 88, 865, 80, 450
    ymax = 0.75
    for tick in range(6):
        value = ymax * tick / 5
        y = bottom - (bottom - top) * tick / 5
        add_line(out, left, y, right, y, "#e4e7ea")
        add_text(out, left - 10, y + 4, f"{value:.2f}", 10, anchor="end", color="#555")
    add_line(out, left, top, left, bottom, "#555")
    add_line(out, left, bottom, right, bottom, "#555")
    xs = [left + i * (right - left) / max(1, len(rows) - 1) for i in range(len(rows))]
    for x, row in zip(xs, rows):
        add_text(out, x, bottom + 22, row["stride_px"], 11)
    for group, field in (("C0", "C0_mae_m"), ("F1", "F1_mae_m"), ("F2", "F2_mae_m")):
        coords = []
        for x, row in zip(xs, rows):
            y = bottom - (bottom - top) * float(row[field]) / ymax
            coords.append((x, y))
        for (x1, y1), (x2, y2) in zip(coords, coords[1:]):
            add_line(out, x1, y1, x2, y2, COLORS[group], 2)
        for (x, y), row in zip(coords, rows):
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{COLORS[group]}" stroke="#fff" stroke-width="1.5"/>')
            add_text(out, x, y - 10, f'{float(row[field]):.3f}', 9, color=COLORS[group])
    for i, group in enumerate(("C0", "F1", "F2")):
        x = 365 + i * 110
        add_line(out, x, 510, x + 22, 510, COLORS[group], 3)
        add_text(out, x + 30, 514, group, 12, anchor="start")
    add_text(out, 22, 250, "MAE (m)", 11, anchor="start", color="#555")
    finish(out, output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate dependency-free SVG result figures.")
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--bootstrap-csv", type=Path, required=True)
    parser.add_argument("--local-scale-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    make_grid(args.summary_csv, args.output_dir / "figure1_final_metrics.svg")
    make_effects(args.bootstrap_csv, args.output_dir / "figure2_primary_effects.svg")
    make_scale(args.local_scale_csv, args.output_dir / "figure3_local_scale.svg")
    print(args.output_dir)
    for path in sorted(args.output_dir.glob("*.svg")):
        print(f"{path.name}: {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
