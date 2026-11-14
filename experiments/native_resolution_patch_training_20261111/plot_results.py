from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {"C0": "#4C78A8", "F1": "#F58518", "F2": "#54A24B"}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def make_grid(summary_path: Path, output: Path) -> None:
    data = rows(summary_path)
    expected = {(group, method) for group in ("C0", "F1", "F2")
                for method in ("P0", "P1", "P2")}
    actual = {(r["group"], r["method"]) for r in data}
    if actual != expected:
        raise ValueError(f"Grid has missing/extra combinations: {actual ^ expected}")
    data.sort(key=lambda r: (("C0", "F1", "F2").index(r["group"]),
                             ("P0", "P1", "P2").index(r["method"])))
    labels = [f'{r["group"]}-{r["method"]}' for r in data]
    x = np.arange(len(data))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)
    plots = (
        ("rmse_m", "Absolute depth RMSE (m)"),
        ("full14_mae_m", "14 px local-pair MAE (m), all valid pairs"),
    )
    for axis, (field, title) in zip(axes, plots):
        values = [float(r[field]) for r in data]
        axis.bar(x, values, color=[COLORS[r["group"]] for r in data], edgecolor="#333333", linewidth=.5)
        axis.set_title(title)
        axis.set_xticks(x, labels, rotation=45, ha="right")
        axis.grid(axis="y", alpha=.25)
        axis.set_axisbelow(True)
        axis.set_ylabel("metres")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[g], label=g) for g in COLORS]
    axes[0].legend(handles=handles, title="Training group", frameon=False)
    fig.suptitle("Validation128: epoch-5 models × inference methods", fontsize=13)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_effects(bootstrap_path: Path, output: Path) -> None:
    data = rows(bootstrap_path)
    chosen = [r for r in data if r["method"] == "P2" and r["baseline"] == "C0"
              and r["estimator"] == "pooled"
              and ((r["metric"] == "delta_mae")
                   or (r["metric"] == "rmse"))]
    metrics = [("delta_mae", "Primary: 14 px, GT |Δ| ≥ 0.20 m pair MAE", "m"),
               ("rmse", "Absolute depth RMSE", "m")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    for axis, (metric, title, unit) in zip(axes, metrics):
        subset = [r for r in chosen if r["metric"] == metric]
        subset.sort(key=lambda r: ("F1", "F2").index(r["candidate"]))
        if {r["candidate"] for r in subset} != {"F1", "F2"}:
            raise ValueError(f"Missing paired contrast for {metric}/P2")
        pos = np.arange(len(subset))
        point = np.asarray([float(r["candidate_minus_baseline"]) for r in subset])
        low = np.asarray([float(r["ci95_low"]) for r in subset])
        high = np.asarray([float(r["ci95_high"]) for r in subset])
        axis.errorbar(point, pos, xerr=np.vstack((point - low, high - point)),
                      fmt="o", color="#4C78A8", capsize=4)
        axis.axvline(0, color="#555555", linewidth=1, linestyle="--")
        axis.set_yticks(pos, [r["candidate"] for r in subset])
        axis.set_title(title)
        axis.set_xlabel(f"candidate − C0 ({unit}); 95% paired image-bootstrap CI")
        axis.grid(axis="x", alpha=.25)
        axis.set_axisbelow(True)
    fig.suptitle("P2 inference: paired effects on Validation128", fontsize=13)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--bootstrap-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    make_grid(args.eval_root / "grid_summary.csv", args.output_dir / "figure1_grid_performance.png")
    make_effects(args.bootstrap_csv, args.output_dir / "figure2_paired_effects.png")
    print(args.output_dir)


if __name__ == "__main__":
    main()


