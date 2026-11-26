from __future__ import annotations

"""Create fixed C0 epoch 1-20 learning-curve tables and plots.

Historical validation values before epoch 5 are intentionally left missing when
no per-epoch result is present; the script does not interpolate them.
"""

import argparse
import csv
import html
import json
import math
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def load_records(historical_training: Path, epoch5_evaluation: Path,
                 run_dir: Path) -> list[dict]:
    train_rows = [row for row in read_rows(historical_training)
                  if row.get("group") == "C0"]
    if sorted(int(row["epoch"]) for row in train_rows) != [1, 2, 3, 4, 5]:
        raise ValueError("Historical C0 training history must contain epochs 1-5 exactly")

    epoch5 = json.loads(epoch5_evaluation.read_text(encoding="utf-8"))
    if not epoch5.get("complete") or epoch5.get("group") != "C0" or int(epoch5.get("epoch", -1)) != 5:
        raise ValueError("The baseline evaluation must be the completed C0 epoch5 result")
    base_abs = epoch5["absolute"]
    base_local = [row for row in epoch5["sampled_local"]
                  if int(row["stride_px"]) == 14 and row["group"] == "gt_ge20cm"]
    if len(base_local) != 1:
        raise ValueError("C0 epoch5 evaluation lacks the unique high-delta 14px metric")

    records: dict[int, dict] = {}
    for row in train_rows:
        epoch = int(row["epoch"])
        records[epoch] = {
            "epoch": epoch,
            "train_silog_m": float(row["mean_silog_m"]),
            "optimizer_updates": int(row["attempted_updates"]),
            "amp_skips": int(row["amp_skips"]),
            "validation_rmse_m": None,
            "full14_mae_m": None,
            "high_delta14_mae_m": None,
            "amplitude_ratio": None,
            "sign_match_rate": None,
            "encoder_lr": None,
            "neck_head_lr": None,
            "validation_source": "not_available_in_preserved_epoch1_5_summary",
        }

    for epoch in range(6, 21):
        records[epoch] = {
            "epoch": epoch, "train_silog_m": None, "optimizer_updates": None,
            "amp_skips": None, "validation_rmse_m": None, "full14_mae_m": None,
            "high_delta14_mae_m": None, "amplitude_ratio": None,
            "sign_match_rate": None, "encoder_lr": None, "neck_head_lr": None,
            "validation_source": "awaiting_epoch_record",
        }

    records[5].update({
        "validation_rmse_m": float(base_abs["rmse_m"]),
        "full14_mae_m": float(base_abs["full14_mae_m"]),
        "high_delta14_mae_m": float(base_local[0]["delta_mae_m"]),
        "amplitude_ratio": float(base_local[0]["amplitude_ratio"]),
        "sign_match_rate": float(base_local[0]["sign_match_rate"]),
        "validation_source": "fixed_rgb_input_corrected_epoch5_C0_P0",
    })

    for epoch in range(6, 21):
        path = run_dir / f"epoch_{epoch:04d}.json"
        if not path.is_file():
            continue
        row = json.loads(path.read_text(encoding="utf-8"))
        if int(row.get("epoch", -1)) != epoch or row.get("group") != "C0":
            raise ValueError(f"Unexpected or mismatched epoch record: {path}")
        validation = row["validation"]
        absolute = validation["absolute"]
        high = [item for item in validation["sampled_local"]
                if int(item["stride_px"]) == 14 and item["group"] == "gt_ge20cm"]
        if len(high) != 1:
            raise ValueError(f"Epoch {epoch} lacks the unique high-delta 14px metric")
        records[epoch] = {
            "epoch": epoch,
            "train_silog_m": float(row["mean_silog_m"]),
            "optimizer_updates": int(row["optimizer_updates"]),
            "amp_skips": int(row["amp_skips"]),
            "validation_rmse_m": float(absolute["rmse_m"]),
            "full14_mae_m": float(absolute["full14_mae_m"]),
            "high_delta14_mae_m": float(high[0]["delta_mae_m"]),
            "amplitude_ratio": float(high[0]["amplitude_ratio"]),
            "sign_match_rate": float(high[0]["sign_match_rate"]),
            "encoder_lr": float(row["scheduler_lr_encoder"]),
            "neck_head_lr": float(row["scheduler_lr_neck_head"]),
            "validation_source": path.name,
        }
    return [records[i] for i in range(1, 21)]


def plot_one(records: list[dict], field: str, title: str, ylabel: str,
             output_dir: Path, restart_marker: bool = False) -> None:
    width, height = 960, 540
    left, right, top, bottom = 92, 35, 56, 72
    plot_w, plot_h = width - left - right, height - top - bottom
    points = [(float(row["epoch"]), float(row[field])) for row in records
              if row[field] is not None and math.isfinite(float(row[field]))]
    ymin = min((value for _, value in points), default=0.0)
    ymax = max((value for _, value in points), default=1.0)
    if ymax <= ymin:
        pad = max(abs(ymin) * 0.1, 0.1)
    else:
        pad = (ymax - ymin) * 0.08
    ymin, ymax = ymin - pad, ymax + pad

    def sx(epoch: float) -> float:
        return left + (epoch - 1.0) / 19.0 * plot_w

    def sy(value: float) -> float:
        return top + (ymax - value) / (ymax - ymin) * plot_h

    esc = lambda value: html.escape(str(value), quote=True)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<g font-family="Arial, sans-serif" fill="#202124">',
        f'<text x="{width/2:.1f}" y="30" text-anchor="middle" font-size="20" font-weight="600">{esc(title)}</text>',
    ]
    for i in range(5):
        value = ymax - i * (ymax - ymin) / 4
        y = sy(value)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" stroke="#d9dde2" stroke-width="1"/>')
        parts.append(f'<text x="{left-12}" y="{y+4:.2f}" text-anchor="end" font-size="12">{value:.4g}</text>')
    for epoch in range(1, 21):
        x = sx(float(epoch))
        parts.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{height-bottom}" stroke="#eef0f2" stroke-width="1"/>')
        parts.append(f'<text x="{x:.2f}" y="{height-bottom+23}" text-anchor="middle" font-size="11">{epoch}</text>')
    parts.extend([
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#4b5563"/>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#4b5563"/>',
        f'<text x="{left+plot_w/2:.1f}" y="{height-20}" text-anchor="middle" font-size="14">Cumulative C0 epoch</text>',
        f'<text x="22" y="{top+plot_h/2:.1f}" text-anchor="middle" font-size="14" transform="rotate(-90 22 {top+plot_h/2:.1f})">{esc(ylabel)}</text>',
    ])
    if restart_marker:
        x = sx(5.5)
        parts.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{height-bottom}" stroke="#bd5a36" stroke-width="1.5" stroke-dasharray="6 5"/>')
        parts.append(f'<text x="{x+8:.2f}" y="{top+15}" font-size="11" fill="#8a3f27">LR reset at epoch 6</text>')
    path_data: list[str] = []
    previous = None
    points_by_epoch = {int(epoch): value for epoch, value in points}
    for row in records:
        epoch = int(row["epoch"])
        if row[field] is None or not math.isfinite(float(row[field])):
            if path_data:
                parts.append(f'<polyline points="{" ".join(path_data)}" fill="none" stroke="#2468a2" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>')
                path_data = []
            previous = None
            continue
        point = f'{sx(epoch):.2f},{sy(float(row[field])):.2f}'
        path_data.append(point)
    if path_data:
        parts.append(f'<polyline points="{" ".join(path_data)}" fill="none" stroke="#2468a2" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>')
    for epoch, value in points_by_epoch.items():
        parts.append(f'<circle cx="{sx(epoch):.2f}" cy="{sy(value):.2f}" r="4.2" fill="#2468a2" stroke="#ffffff" stroke-width="1.2"/>')
    parts.extend(['</g>', '</svg>'])
    (output_dir / f"{field}.svg").write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-training", type=Path, required=True)
    parser.add_argument("--epoch5-evaluation", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(args.historical_training, args.epoch5_evaluation, args.run_dir)
    if [row["epoch"] for row in records] != list(range(1, 21)):
        raise RuntimeError("Learning curve must have one row for every cumulative epoch 1-20")
    with (args.output_dir / "learning_curve_data.csv").open(
            "x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    specs = [
        ("train_silog_m", "Training SiLog", "Mean SiLog loss (m)", False),
        ("validation_rmse_m", "Validation absolute depth RMSE", "RMSE (m)", False),
        ("full14_mae_m", "All valid 14 px pair MAE", "MAE (m)", False),
        ("high_delta14_mae_m", "GT |ΔZ| ≥ 0.20 m, 14 px pair MAE", "MAE (m)", False),
        ("amplitude_ratio", "GT |ΔZ| ≥ 0.20 m, 14 px amplitude ratio", "Amplitude ratio", False),
        ("sign_match_rate", "GT |ΔZ| ≥ 0.20 m, 14 px sign match", "Sign match rate", False),
        ("encoder_lr", "Encoder learning rate", "Learning rate", True),
        ("neck_head_lr", "Neck/head learning rate", "Learning rate", True),
    ]
    for field, title, ylabel, marker in specs:
        plot_one(records, field, title, ylabel, args.output_dir, marker)
    print(json.dumps({"epochs": len(records), "observed_validation_epochs": [
        r["epoch"] for r in records if r["validation_rmse_m"] is not None],
        "output_dir": str(args.output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
