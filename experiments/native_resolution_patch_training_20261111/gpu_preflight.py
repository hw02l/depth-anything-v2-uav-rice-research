from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time


def run(args: list[str], timeout: int = 20) -> dict:
    try:
        result = subprocess.run(args, check=False, text=True, capture_output=True, timeout=timeout)
        return {"returncode": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        return {"returncode": None, "stdout": "", "stderr": f"{type(error).__name__}: {error}"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only GPGPU4 eligibility receipt; does not create directories or launch workloads.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    hostname = run(["hostname", "-f"])
    who = run(["who"])
    users = [line for line in who["stdout"].splitlines() if line.strip()]
    nvidia = run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,utilization.gpu", "--format=csv,noheader,nounits"])
    processes = run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits"])
    work_path = Path.home() / "gpu_work"
    real_work = run(["realpath", str(work_path)])
    df = run(["df", "-PB1", str(work_path)])
    du = run(["du", "-sb", str(work_path)]) if work_path.exists() else {"returncode": 1, "stdout": "", "stderr": "~/gpu_work does not exist"}
    singularity = run(["singularity", "--version"])
    w = run(["w", "-h"])

    gpu_rows = [line.strip() for line in nvidia["stdout"].splitlines() if line.strip()]
    compute_rows = [line.strip() for line in processes["stdout"].splitlines() if line.strip() and "No running processes found" not in line]
    fields = gpu_rows[0].split(",") if len(gpu_rows) == 1 else []
    free_bytes = int(fields[2]) * 1024**2 if len(fields) == 4 and fields[2].strip().isdigit() else 0
    df_rows = [line.split() for line in df["stdout"].splitlines()[1:] if line.split()]
    disk_free_bytes = int(df_rows[0][3]) if len(df_rows) == 1 and df_rows[0][3].isdigit() else 0
    real_root = real_work["stdout"]
    version_match = re.search(r"\b(\d+\.\d+\.\d+)\b", singularity["stdout"])
    singularity_version = version_match.group(1) if version_match else singularity["stdout"].strip()
    expected_host = hostname["stdout"].split(".")[0].lower() == "gpgpu4"
    local_root = real_root.startswith("/localdata/")
    correct_singularity = singularity_version == "4.2.2"
    checks = {
        "hostname_is_gpgpu4": expected_host,
        "exactly_one_gpu_visible": len(gpu_rows) == 1,
        "at_least_1gib_vram_free": free_bytes >= 1024**3,
        "no_gpu_compute_processes": len(compute_rows) == 0,
        "gpu_work_exists": work_path.exists(),
        "gpu_work_is_localdata": local_root,
        "at_least_1gib_disk_free": disk_free_bytes >= 1024**3,
        "singularity_4_2_2": correct_singularity,
    }
    receipt = {
        "status": "available" if all(checks.values()) else "blocked",
        "unix_time": time.time(),
        "hostname": hostname["stdout"],
        "gpu_name": fields[0].strip() if fields else None,
        "gpu_total_bytes": int(fields[1]) * 1024**2 if len(fields) == 4 and fields[1].strip().isdigit() else 0,
        "gpu_free_bytes": free_bytes,
        "gpu_utilization_percent": int(fields[3]) if len(fields) == 4 and fields[3].strip().isdigit() else None,
        "compute_processes": compute_rows,
        "active_login_session_count": len(users),
        "active_login_sessions_redacted": ["<session>" for _ in users],
        "w_session_count": len([line for line in w["stdout"].splitlines() if line.strip()]),
        "work_root": str(work_path),
        "work_root_realpath": real_root,
        "work_root_size_bytes": int(du["stdout"].split()[0]) if du["returncode"] == 0 and du["stdout"].split() else None,
        "disk_free_bytes": disk_free_bytes,
        "singularity_version": singularity_version,
        "checks": checks,
        "commands": {
            "nvidia_smi": nvidia,
            "compute_process_query": processes,
            "df": df,
            "du": du,
            "singularity": singularity,
            "w": {"returncode": w["returncode"], "stderr": w["stderr"]},
        },
        "read_only": True,
    }
    payload = json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name("." + args.output.name + ".tmp")
        temporary.write_text(payload + "\n", encoding="utf-8")
        os.replace(temporary, args.output)
    print(payload, flush=True)
    if receipt["status"] != "available":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
