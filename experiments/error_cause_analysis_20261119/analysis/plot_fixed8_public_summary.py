"""Create public-safe summary charts from fixed-eight aggregate CSVs."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


HERE = Path(__file__).resolve().parent


def save_public_figure(fig: plt.Figure, stem: str) -> None:
    svg_path = HERE / f"{stem}.svg"
    png_path = HERE / f"{stem}.png"
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=120, bbox_inches="tight")
    with Image.open(png_path) as rendered:
        rendered.convert("RGB").quantize(
            colors=128, method=Image.Quantize.FASTOCTREE
        ).save(png_path, optimize=True)


def read_rows(filename: str) -> list[dict[str, str]]:
    with (HERE / filename).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def plot_metrics() -> None:
    rows = read_rows("gpu3_fixed8_grouped_summary.csv")
    labels = [row["condition"] for row in rows]
    rmse = [float(row["absolute_rmse_m"]) for row in rows]
    local = [float(row["local14_gt20_mae_m"]) for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for ax, values, title, ylabel in (
        (axes[0], rmse, "Absolute depth error", "RMSE (m)"),
        (axes[1], local, "Local depth-difference error", "MAE (m)"),
    ):
        bars = ax.bar(labels, values, color="#5277a3")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", rotation=25)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)

    fig.suptitle(
        "GPU3 fixed-eight diagnostics (descriptive; not a Validation128 estimate)",
        fontsize=11,
    )
    save_public_figure(fig, "gpu3_fixed8_diagnostics_summary")
    plt.close(fig)


def plot_tile_disagreement() -> None:
    rows = read_rows("gpu3_fixed8_tile_summary.csv")
    labels = [row["condition"] for row in rows]
    means = [float(row["overlap_neighbor_abs_delta_mean_m"]) for row in rows]
    p95 = [float(row["overlap_neighbor_abs_delta_p95_m"]) for row in rows]

    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    x = list(range(len(labels)))
    width = 0.36
    b1 = ax.bar([i - width / 2 for i in x], means, width, label="Mean")
    b2 = ax.bar([i + width / 2 for i in x], p95, width, label="95th percentile")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Absolute difference between adjacent tile predictions (m)")
    ax.set_title("Tile-overlap disagreement on eight fixed images")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    ax.bar_label(b1, fmt="%.3f", padding=2, fontsize=8)
    ax.bar_label(b2, fmt="%.3f", padding=2, fontsize=8)
    ax.text(
        0.01,
        -0.24,
        "Descriptive only. Tile overlap and fused full-image metrics have different support.",
        transform=ax.transAxes,
        fontsize=8,
    )
    save_public_figure(fig, "gpu3_fixed8_tile_disagreement_summary")
    plt.close(fig)


def main() -> None:
    plot_metrics()
    plot_tile_disagreement()
    print("PASS: generated two aggregate-only, RGB/GT-free figures")


if __name__ == "__main__":
    main()
