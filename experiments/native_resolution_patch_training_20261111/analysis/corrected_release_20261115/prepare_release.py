from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from xml.sax.saxutils import escape

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[1]
RAW = EXP / "analysis" / "rgb_input_fix_20261114"
OUT = HERE
FRESH_AUDIT = OUT / "fresh_reproduction_audit.json"

EXPECTED = {
    "C0": "2fd190727c7a5124c56c1e0ea2dd8cb09c6cb62a0f92b34fce97ad394281b35a",
    "F1": "ae1f74c5ff17cb0266152588b5d53fb824611418712152fbf4d9cc7447e49f67",
    "F2": "8124d72a7008c9f3a93dbce934142d1593065fba2da61a51838e055aaf201538",
}
REFERENCE_LITERAL_SHA256 = {
    "C0": "2fd190727c7a5124c56c10e0a2dd8cb09c6cb62a0f92b34fce97ad394281b35a",
    "F1": "ea1f74c5ff17c0b266152588b5d53f8b24611418f125f2fbf4d9cc7447e49f67",
    "F2": "8124d72a7008c9f3a93dbcce34142d1593063fba2da6a1a58306d55aaf201538",
}
COLORS = {"C0": "#4C78A8", "F1": "#F58518", "F2": "#54A24B"}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_rows(path: Path, rows: list[dict[str, str]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows for {path.name}")
    keys = fields or list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_raw_manifest() -> tuple[int, list[str]]:
    manifest = RAW / "SHA256SUMS"
    failures: list[str] = []
    count = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(None, 1)
        relative = relative.lstrip("*")
        path = EXP.parents[1] / Path(relative.replace("/", "\\"))
        count += 1
        if not path.is_file() or sha256(path) != expected:
            failures.append(relative)
    return count, failures


def select_one(rows: list[dict[str, str]], **criteria: str) -> dict[str, str]:
    found = [row for row in rows if all(row.get(key, "") == value for key, value in criteria.items())]
    if len(found) != 1:
        raise ValueError(f"Expected one row for {criteria}, got {len(found)}")
    return found[0]


def svg_chart(title: str, subtitle: str, labels: list[str], series: list[tuple[str, list[float]]],
              maximum: float, ylabel: str, filename: str) -> None:
    width, height = 1120, 650
    left, right, top, bottom = 100, 60, 110, 115
    plot_w, plot_h = width - left - right, height - top - bottom
    group_w = plot_w / len(labels)
    bar_count = len(series)
    bar_w = min(54, group_w * 0.68 / bar_count)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="36" font-family="Arial,sans-serif" font-size="25" font-weight="700">{escape(title)}</text>',
        f'<text x="{left}" y="66" font-family="Arial,sans-serif" font-size="15" fill="#555">{escape(subtitle)}</text>',
        f'<text x="{left}" y="92" font-family="Arial,sans-serif" font-size="13" fill="#1b5e20">fresh再推論と科学指標・十分統計の照合済み</text>',
    ]
    for tick in range(6):
        value = maximum * tick / 5
        y = top + plot_h - (value / maximum) * plot_h
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#ddd"/>')
        parts.append(f'<text x="{left-12}" y="{y+5:.1f}" text-anchor="end" font-family="Arial,sans-serif" font-size="12" fill="#444">{value:.3g}</text>')
    for i, label in enumerate(labels):
        center = left + group_w * (i + 0.5)
        for j, (name, values) in enumerate(series):
            value = values[i]
            x = center + (j - (bar_count - 1) / 2) * bar_w - bar_w / 2
            bar_h = max(0, value / maximum * plot_h)
            y = top + plot_h - bar_h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w-3:.1f}" height="{bar_h:.1f}" fill="{COLORS[name]}"/>')
            parts.append(f'<text x="{x + (bar_w-3)/2:.1f}" y="{max(top+13,y-5):.1f}" text-anchor="middle" font-family="Arial,sans-serif" font-size="10" fill="#333">{value:.3f}</text>')
        parts.append(f'<text x="{center:.1f}" y="{top+plot_h+28}" text-anchor="middle" font-family="Arial,sans-serif" font-size="13" fill="#333">{escape(label)}</text>')
    parts.append(f'<text x="24" y="{top+plot_h/2:.1f}" transform="rotate(-90 24 {top+plot_h/2:.1f})" text-anchor="middle" font-family="Arial,sans-serif" font-size="14" fill="#333">{escape(ylabel)}</text>')
    legend_y = height - 30
    legend_x = left
    for name, _ in series:
        parts.append(f'<rect x="{legend_x}" y="{legend_y-12}" width="14" height="14" fill="{COLORS[name]}"/>')
        parts.append(f'<text x="{legend_x+20}" y="{legend_y}" font-family="Arial,sans-serif" font-size="13" fill="#333">{name}</text>')
        legend_x += 90
    parts.append("</svg>")
    (OUT / filename).write_text("\n".join(parts) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    grid = read_rows(RAW / "per_condition" / "grid_summary.csv")
    if len(grid) != 9 or {(r["group"], r["method"]) for r in grid} != {
        (group, method) for group in ("C0", "F1", "F2") for method in ("P0", "P1", "P2")
    }:
        raise ValueError("Expected exactly the C0/F1/F2 × P0/P1/P2 grid")
    metric_fields = [
        "group", "method", "epoch", "image_count", "valid_pixels", "mae_m", "rmse_m",
        "abs_rel", "bias_m", "pearson_r", "full14_mae_m",
        "tile_count", "checkpoint_sha256",
    ]
    write_rows(OUT / "metrics_9_conditions.csv", grid, metric_fields)

    bootstrap = read_rows(RAW / "paired_condition_bootstrap.csv")
    primary = [
        r for r in bootstrap
        if r["task"] == "local" and r["scope"] == "all"
        and r["condition"] == "gt_ge20cm" and r["stride_px"] == "14"
        and r["metric"] == "delta_mae" and not r["crop"] and not r["view"]
    ]
    if len(primary) != 12:
        raise ValueError(f"Expected 12 primary comparisons, got {len(primary)}")
    secondary = [
        r for r in bootstrap
        if r["task"] == "local" and r["scope"] == "all"
        and r["condition"] == "all" and r["stride_px"] == "14"
        and r["metric"] == "delta_mae" and r["method"] == "P2"
        and r["candidate"] == "F2" and not r["crop"] and not r["view"]
    ]
    absolute_comparisons = [
        r for r in bootstrap
        if r["task"] == "absolute" and r["scope"] == "all"
        and r["candidate"] in ("F1", "F2")
        and r["metric"] in ("rmse", "full14_mae")
        and r["estimator"] == "pooled"
    ]
    amplitude_sign = [
        r for r in bootstrap
        if r["task"] == "local" and r["scope"] == "all"
        and r["method"] == "P2" and r["candidate"] == "F2"
        and r["condition"] == "gt_ge20cm" and r["stride_px"] == "14"
        and r["metric"] in ("amplitude_ratio", "sign_match_rate")
        and r["estimator"] == "pooled" and not r["crop"] and not r["view"]
    ]
    distance_effects = [
        r for r in bootstrap
        if r["task"] == "local" and r["scope"] == "all"
        and r["method"] == "P2" and r["candidate"] == "F2"
        and r["condition"] == "all" and r["metric"] == "delta_mae"
        and r["estimator"] == "pooled" and r["stride_px"] in {"14", "28", "56", "107", "214"}
        and not r["crop"] and not r["view"]
    ]
    crop_view_effects = [
        r for r in bootstrap
        if r["task"] == "local" and r["scope"] in ("crop", "view")
        and r["method"] == "P2" and r["candidate"] == "F2"
        and r["condition"] == "gt_ge20cm" and r["stride_px"] == "14"
        and r["metric"] == "delta_mae" and r["estimator"] == "pooled"
    ]
    key_rows = primary + secondary + absolute_comparisons + amplitude_sign + distance_effects + crop_view_effects
    write_rows(OUT / "key_paired_comparisons.csv", key_rows)

    within_method_rows = []
    by_condition = {(row["group"], row["method"]): row for row in grid}
    for group in ("C0", "F1", "F2"):
        reference = by_condition[(group, "P0")]
        for method in ("P0", "P1", "P2"):
            row = by_condition[(group, method)]
            within_method_rows.append({
                "group": group,
                "method": method,
                "rmse_m": row["rmse_m"],
                "rmse_delta_vs_P0_m": str(float(row["rmse_m"]) - float(reference["rmse_m"])),
                "full14_mae_m": row["full14_mae_m"],
                "full14_mae_delta_vs_P0_m": str(float(row["full14_mae_m"]) - float(reference["full14_mae_m"])),
            })
    write_rows(OUT / "within_model_inference_method.csv", within_method_rows)

    expected_checkpoints = []
    for group, expected in EXPECTED.items():
        recorded_values = sorted({r["checkpoint_sha256"] for r in grid if r["group"] == group})
        if len(recorded_values) != 1:
            raise ValueError(f"Inconsistent checkpoint hashes inside {group}")
        recorded = recorded_values[0]
        if recorded != expected:
            raise ValueError(f"Historical evaluation checkpoint hash differs from verified {group} hash")
        expected_checkpoints.append({
            "group": group,
            "prior_reference_literal_sha256": REFERENCE_LITERAL_SHA256[group],
            "verified_checkpoint_sha256": expected,
            "historical_evaluation_recorded_sha256": recorded,
            "checkpoint_matches_historical_record": str(expected == recorded).lower(),
        })
    write_rows(OUT / "checkpoint_hash_audit.csv", expected_checkpoints)

    absolute_rmse = {f'{r["group"]}-{r["method"]}': float(r["rmse_m"]) for r in grid}
    svg_chart(
        "Absolute depth RMSE by training and inference condition",
        "Validation128, GT mask 0 < depth <= 65 m; independently reproduced",
        ["P0", "P1", "P2"],
        [(group, [absolute_rmse[f"{group}-{method}"] for method in ("P0", "P1", "P2")])
         for group in ("C0", "F1", "F2")],
        9.0, "RMSE (m)", "figure1_absolute_rmse.svg",
    )

    conditions = read_rows(RAW / "condition_summary.csv")
    primary_values: dict[str, dict[str, float]] = {group: {} for group in ("C0", "F1", "F2")}
    all_pair_values: dict[str, dict[str, float]] = {group: {} for group in ("C0", "F1", "F2")}
    for group in ("C0", "F1", "F2"):
        for method in ("P0", "P1", "P2"):
            row = select_one(
                conditions, task="local", training_group=group, method=method, scope="all",
                crop="", view="", condition="gt_ge20cm", stride_px="14",
                delta_group="", metric="delta_mae", estimator="pooled",
            )
            primary_values[group][method] = float(row["value"])
            row = select_one(
                conditions, task="local", training_group=group, method=method, scope="all",
                crop="", view="", condition="all", stride_px="14",
                delta_group="", metric="delta_mae", estimator="pooled",
            )
            all_pair_values[group][method] = float(row["value"])

    svg_chart(
        "Primary local MAE: GT delta ≥ 0.20 m, 14 px fixed pairs",
        "Pooled estimate; C0/F1/F2 training compared within each inference method",
        ["P0", "P1", "P2"],
        [(group, [primary_values[group][method] for method in ("P0", "P1", "P2")])
         for group in ("C0", "F1", "F2")],
        0.5, "Fixed-pair MAE (m)", "figure2_primary_fixed_pairs.svg",
    )
    svg_chart(
        "P2 local MAE: fixed large-delta primary and all-pair secondary",
        "GT delta ≥ 0.20 m primary vs all valid 14 px pairs secondary",
        ["GT ≥ 0.20 m", "All valid pairs"],
        [
            ("C0", [primary_values["C0"]["P2"], all_pair_values["C0"]["P2"]]),
            ("F2", [primary_values["F2"]["P2"], all_pair_values["F2"]["P2"]]),
        ],
        0.5, "Local MAE (m)", "figure3_primary_vs_all_pairs.svg",
    )

    raw_count, failures = verify_raw_manifest()
    if failures:
        raise ValueError("Raw result SHA256 verification failed: " + ", ".join(failures))
    if not FRESH_AUDIT.is_file():
        raise FileNotFoundError(f"Fresh reproduction audit is required: {FRESH_AUDIT}")
    fresh_audit = json.loads(FRESH_AUDIT.read_text(encoding="utf-8"))
    fresh_run = fresh_audit["fresh_run"]
    fresh_comparison = fresh_audit["fresh_vs_historical_corrected_outputs"]
    fresh_nfs = fresh_audit["nfs_archive"]
    if fresh_audit.get("status") != "fresh_reproduction_verified":
        raise ValueError("Fresh reproduction audit is not marked verified")
    if fresh_run.get("completion_status") != "complete" or fresh_run.get("evaluation_json_count") != 9:
        raise ValueError("Fresh reproduction audit does not confirm all nine evaluations")
    if fresh_run.get("checkpoint_sha256_by_group") != EXPECTED:
        raise ValueError("Fresh checkpoint hashes do not match the audited checkpoint hashes")
    if not fresh_run.get("checkpoint_state_load_audit_present_in_all_json"):
        raise ValueError("Fresh evaluation JSONs lack checkpoint state-load audit")
    if fresh_run.get("rgb_input_contract") != "load_pair_float32_0_1_roundtrip_rint_to_uint8_0_255_v1":
        raise ValueError("Unexpected RGB input contract in fresh audit")
    if fresh_comparison.get("condition_summary_csv") != "byte-identical":
        raise ValueError("Fresh condition summary comparison is not byte-identical")
    if fresh_comparison.get("paired_condition_bootstrap_csv") != "byte-identical":
        raise ValueError("Fresh paired bootstrap comparison is not byte-identical")
    if not fresh_nfs.get("sha256_manifest_generated_and_readback_verified"):
        raise ValueError("Fresh audit snapshot NFS manifest was not verified")
    f2_json = json.loads((RAW / "per_condition" / "F2_P2" / "evaluation_F2_P2.json").read_text(encoding="utf-8"))
    missing_provenance = [
        key for key in ("rgb_input_contract", "evaluation_code_revision", "checkpoint_state_load_audit")
        if key not in f2_json
    ]
    provenance = f"""# Provenance audit

- Historical corrected-result bundle: {RAW.relative_to(EXP).as_posix()}; its internal SHA256 manifest has {raw_count} entries and {len(failures)} failures. Its evaluation JSON lacks {", ".join(missing_provenance) if missing_provenance else "no required provenance keys"}; these historical JSON files remain unchanged and provenance has not been added retroactively.
- Fresh run: status `{fresh_run["completion_status"]}`, {fresh_run["evaluation_json_count"]} evaluation JSONs, {fresh_run["validation_images_per_condition"]} validation images per condition, cache-free unique output `{fresh_run["evaluation_output"]}`.
- Evaluation code revision `{fresh_run["evaluation_code_revision"]}`, SHA256 `{fresh_run["evaluation_code_sha256"]}`; RGB contract `{fresh_run["rgb_input_contract"]}`. All nine JSONs contain checkpoint hashes and state-load audits.
- Verified epoch5 checkpoint SHA256 values match old JSON records, epoch5 training records, old NFS copies, canonical signatures, and run signatures. The three literals supplied in the request were transcription errors; see checkpoint_hash_audit.csv.
- B0 gate: PASS; reference epoch20 checkpoint SHA256 `52769e2e95704d0862e708e8772195b861c647dd4812058eaf698a44305c3bcb`; all 8 metrics were within 1e-5.
- Fresh versus historical corrected outputs: condition summary and paired bootstrap are byte-identical; per-image absolute sufficient statistics match in all nine conditions excluding runtime-only fields; per-image local statistics match all nine; tile-boundary statistics match all six P1/P2 conditions. Bootstrap used 10,000 paired image resamples and seed 42.
- NFS fresh audit snapshot `{fresh_nfs["path"]}` contains {fresh_nfs["file_count_in_sha256sums"]} SHA256 entries ({fresh_nfs["reported_size"]}); copy readback verification passed. It excludes checkpoint binaries and RGB/GT source data.
- No checkpoint, source image, ground-truth image, or prediction cache was modified or included in this release directory; no retraining was performed and Independent test319 was not used.
"""
    (OUT / "provenance_audit.md").write_text(provenance, encoding="utf-8", newline="\n")

    files = sorted(
        p for p in OUT.rglob("*")
        if p.is_file() and p.name != "SHA256SUMS"
        and "__pycache__" not in p.parts and p.suffix != ".pyc"
    )
    lines = [f"{sha256(path)}  {path.relative_to(OUT).as_posix()}" for path in files]
    (OUT / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({
        "status": "fresh_reproduction_verified",
        "nine_conditions": len(grid),
        "fresh_evaluation_jsons": fresh_run["evaluation_json_count"],
        "fresh_code_sha256": fresh_run["evaluation_code_sha256"],
        "primary_bootstrap_rows": len(primary),
        "raw_manifest_entries": raw_count,
        "raw_manifest_failures": len(failures),
        "checkpoint_hash_mismatches": sum(row["checkpoint_matches_historical_record"] == "false" for row in expected_checkpoints),
        "historical_json_missing_provenance_fields": missing_provenance,
        "fresh_summary_and_bootstrap_byte_identical": True,
        "fresh_nfs_manifest_entries": fresh_nfs["file_count_in_sha256sums"],
        "outputs": [p.name for p in files],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
