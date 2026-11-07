#!/usr/bin/env python3
"""Build Japanese report tables from frozen local-huber condition aggregates."""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "final_20261107"
MODELS = ("E0", "E1", "E2", "E3")


def load(name: str) -> list[dict[str, str]]:
    with (ROOT / name).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def num(value: str | None, digits: int = 6) -> str:
    if value in (None, ""):
        return "—"
    return f"{float(value):.{digits}f}"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
        *["| " + " | ".join(row) + " |" for row in rows],
    ])


def main() -> None:
    distance = load("distance_condition_comparison.csv")
    distance_rows = []
    for r in sorted(distance, key=lambda x: (int(x["stride_px"]), 0 if x["stratum"] == "all" else 1)):
        target = "全局所差分" if r["stratum"] == "all" else "GT差≥0.20 m"
        for model in MODELS:
            distance_rows.append([
                r["stride_px"], target, f"{int(r['pair_count']):,}",
                f"{int(r['image_count'])}/{int(r['scope_image_count'])}", model,
                num(r[f"delta_mae_m_{model}"], 6),
                num(r[f"delta_rmse_m_{model}"], 6),
                num(r[f"amplitude_ratio_{model}"], 4),
                num(r[f"sign_match_rate_{model}"], 4),
            ])

    bins = load("gt_delta_bin_comparison.csv")
    bin_label = {
        "gt_delta_0_0.05m": "0–0.05 m",
        "gt_delta_0.05_0.1m": "0.05–0.10 m",
        "gt_delta_0.1_0.2m": "0.10–0.20 m",
        "gt_delta_0.2_0.5m": "0.20–0.50 m",
        "gt_delta_0.5_infm": "≥0.50 m",
    }
    bins.sort(key=lambda r: list(bin_label).index(r["stratum"]))
    bin_rows = [[
        bin_label[r["stratum"]], f"{int(r['pair_count']):,}",
        f"{int(r['image_count'])}/{int(r['scope_image_count'])}",
        *[num(r[f"delta_mae_m_{m}"], 6) for m in MODELS],
    ] for r in bins]

    cv = load("crop_view_comparison.csv")
    cv_rows = []
    for r in sorted(cv, key=lambda x: (0 if x["scope"].startswith("crop=") else 1, x["scope"], 0 if x["stratum"] == "all" else 1)):
        category = "全ペア" if r["stratum"] == "all" else "GT差≥0.20 m"
        cv_rows.append([
            r["scope"].split("=", 1)[1], category,
            f"{int(r['pair_count']):,}", f"{int(r['image_count'])}/{int(r['scope_image_count'])}",
            *[num(r[f"delta_mae_m_{m}"], 6) for m in MODELS],
        ])

    other = load("edge_depth_sign_comparison.csv")
    include = {
        "border_lt128", "middle_128_512", "interior_ge512",
        "depth_0_15m", "depth_15_20m", "depth_20_30m", "depth_30_65m",
        "negative_gt", "zero_gt", "positive_gt",
    }
    labels = {
        "border_lt128": "画像端 <128 px",
        "middle_128_512": "画像端 128–512 px",
        "interior_ge512": "画像端 ≥512 px",
        "depth_0_15m": "ペア平均GT深度 0–15 m",
        "depth_15_20m": "ペア平均GT深度 15–20 m",
        "depth_20_30m": "ペア平均GT深度 20–30 m",
        "depth_30_65m": "ペア平均GT深度 30–65 m",
        "negative_gt": "GT差分 <0",
        "zero_gt": "GT差分 =0",
        "positive_gt": "GT差分 >0",
    }
    order = list(labels)
    other_rows = []
    for r in sorted((x for x in other if x["stratum"] in include), key=lambda x: order.index(x["stratum"])):
        other_rows.append([
            labels[r["stratum"]], f"{int(r['pair_count']):,}",
            f"{int(r['image_count'])}/{int(r['scope_image_count'])}",
            *[num(r[f"delta_mae_m_{m}"], 6) for m in MODELS],
        ])

    text = [
        "# 条件別評価の詳細表",
        "",
        "全表はValidation128画像のB0全体518px予測、元画像座標上の凍結固定ペアを使用した集計である。`画像数`は当該階級で有効なペアを持つ画像数/対象範囲内の画像数を示す。モデル間で同じペアを使用し、空階級は欠損としている。MAE/RMSEの単位はm、振幅比・符号一致率は無次元比率である。",
        "",
        "## 画素間隔別",
        "",
        "MAE、RMSE、振幅比、符号一致率を、全固定標本とGT差20cm以上の標本に分けて示す。ペア数は画像pool値である。",
        "",
        md_table(["間隔(px)", "標本", "ペア数", "画像数", "群", "局所MAE", "局所RMSE", "振幅比", "符号一致率"], distance_rows),
        "",
        "## 14px GT差分階級別",
        "",
        md_table(["GT差の絶対値階級", "ペア数", "画像数", *[f"{m} MAE (m)" for m in MODELS]], bin_rows),
        "",
        "## 作物・撮影方向別（14px）",
        "",
        "`全ペア`とGT差20cm以上を分けて示す。",
        "",
        md_table(["条件", "標本", "ペア数", "画像数", *[f"{m} MAE (m)" for m in MODELS]], cv_rows),
        "",
        "## 画像端・GT深度・GT差分符号別（14px全標本）",
        "",
        md_table(["条件", "ペア数", "画像数", *[f"{m} MAE (m)" for m in MODELS]], other_rows),
        "",
        "追加の固定endpoint-zone strataを含む全43行は `condition_comparison_all.csv` に記録する。endpoint_zones_0_2 は全モデル共通で有効ペア数0であり、指標は未定義のまま保持した。",
        "",
    ]
    (ROOT / "TABLES_JA.md").write_text("\n".join(text), encoding="utf-8")


if __name__ == "__main__":
    main()
