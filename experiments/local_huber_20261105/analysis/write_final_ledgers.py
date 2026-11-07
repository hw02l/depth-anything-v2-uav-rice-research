#!/usr/bin/env python3
"""Verify source manifests and write final provenance/checksum ledgers."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ANALYSIS = Path(__file__).resolve().parent
EXPERIMENT = ANALYSIS.parent
FINAL = ANALYSIS / "final_20261107"
INPUTS = FINAL / "inputs"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def manifest_entries(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relpath = line.split(None, 1)
        entries[relpath.strip()] = digest.lower()
    return entries


def main() -> None:
    source_ledger = INPUTS / "SOURCE_SHA256SUMS"
    derived_ledger = INPUTS / "DERIVED_RESULTS_SHA256SUMS"
    source = manifest_entries(source_ledger)
    derived = manifest_entries(derived_ledger)
    checkpoints = list(csv.DictReader((ANALYSIS / "checkpoint_sha256.csv").open(encoding="utf-8-sig", newline="")))
    for row in checkpoints:
        name = f"runs/{row['model']}/epoch_{int(row['epoch']):04d}.pt" if row["model"] != "E0" else "checkpoint_epoch_0020/model.safetensors"
        if row["model"] == "E0":
            e0_summary = json.loads((INPUTS / "E0_summary.json").read_text(encoding="utf-8"))
            if e0_summary.get("checkpoint_sha256", "").lower() != row["sha256"].lower():
                raise ValueError("E0 checkpoint checksum does not match the saved E0 evaluation record")
        elif source.get(name) != row["sha256"].lower():
            raise ValueError(f"Checkpoint checksum does not match source ledger: {name}")
    for name, digest in derived.items():
        path = INPUTS / name
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"Derived result checksum mismatch: {name}")

    metadata = json.loads((FINAL / "analysis_run_metadata.json").read_text(encoding="utf-8"))
    provenance = {
        "experiment_id": "local_huber_20261105",
        "analysis_date": "2026-11-07",
        "branch": "research/local-huber-20261105",
        "baseline_code_commit": "49bdd188d9caa01cf9bacdf02f7aebec65537fcb",
        "data_split": {"train": "UAV3DCrop Train384; Day019/028/040", "validation": "Day047; 128 images", "independent_test_319_used": False},
        "comparison": {"models": ["E0", "E1", "E2", "E3"], "selected_epoch": {"E0": 20, "E1": 5, "E2": 5, "E3": 5}, "inference": "B0 full-frame 518px", "local_pair_manifest_sha256_actual": "bcb56eab03bec4a44f0626da0f4814876fae67774e536e00cd9f1ad16b1f5728"},
        "known_config_issue": "Original config.json fixed_pairs SHA256 string has 63 characters. It was preserved unchanged; the actual 64-character manifest digest above was used in the runtime signature and coordinate checks.",
        "initial_checkpoint": {"sha256": checkpoints[0]["sha256"], "storage": checkpoints[0]["verified_backup_path"], "evidence": "Recorded in the E0 evaluation signature and E0 summary. The checkpoint itself is not listed in the bundled run-backup SOURCE_SHA256SUMS."},
        "final_checkpoints": {r["model"]: {"epoch": int(r["epoch"]), "sha256": r["sha256"], "storage": r["verified_backup_path"]} for r in checkpoints[1:]},
        "nfs_backup_root": "/home1/eng-std/s0623003/gpu4_research_backup_local_huber_20261105_20261105T081949Z",
        "nfs_source_sha256sums_sha256": sha256(source_ledger),
        "nfs_derived_results_sha256sums_sha256": sha256(derived_ledger),
        "analysis": {"method": "No training during final analysis; the four final checkpoints were re-evaluated on the frozen validation set, then additive sufficient statistics were reaggregated locally.", "bootstrap": metadata["bootstrap"], "contrasts": metadata["contrasts"], "pool_reconciliation": metadata["pool_reconciliation"], "input_hashes": {k: v["sha256"] for k, v in metadata["inputs"].items()}},
        "checkpoint_files_committed_to_git": False,
        "raw_rgb_gt_or_prediction_cache_committed_to_git": False,
    }
    (FINAL / "provenance_final.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    entries = []
    for path in sorted(p for p in FINAL.rglob("*") if p.is_file() and p.name != "ANALYSIS_OUTPUT_SHA256SUMS"):
        rel = path.relative_to(FINAL).as_posix()
        entries.append(f"{sha256(path)}  {rel}")
    (FINAL / "ANALYSIS_OUTPUT_SHA256SUMS").write_text("\n".join(entries) + "\n", encoding="utf-8")
    print(json.dumps({"verified_source_files": len(derived), "checkpoint_records": len(checkpoints), "analysis_output_files": len(entries), "source_ledger_sha256": sha256(source_ledger)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
