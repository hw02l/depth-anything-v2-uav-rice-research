"""Read-only GPGPU4 preflight; writes a new localdata receipt, never starts GPU work."""
import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sif', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--coordination-note', required=True, help='Actual current lab usage coordination; no assumption from yesterday')
    args = p.parse_args()
    local = (Path.home()/'gpu_work').resolve()
    if socket.gethostname().lower() != 'gpgpu4' or not str(local).startswith('/localdata/'):
        raise RuntimeError('Wrong host or non-local storage')
    if not args.output.resolve().is_relative_to(local) or not args.sif.resolve().is_relative_to(local) or not args.sif.is_file():
        raise RuntimeError('Receipt and verified SIF must reside in localdata')
    commands = [['hostname'], ['nvidia-smi'], ['w'], ['who'], ['readlink','-f',str(Path.home()/'gpu_work')],
                ['df','-h',str(local)], ['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits']]
    records = []
    for command in commands:
        proc = subprocess.run(command, text=True, capture_output=True, timeout=30)
        records.append(dict(command=command, returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr))
    singularity = shutil.which('singularity')
    apptainer = shutil.which('apptainer')
    tool = singularity or apptainer
    if tool:
        proc = subprocess.run([tool,'instance','list'], text=True, capture_output=True, timeout=30)
        records.append(dict(command=[tool,'instance','list'], returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr))
    available = bool(tool) and all(r['returncode'] == 0 for r in records) and not records[6]['stdout'].strip()
    result = dict(status='available' if available else 'unavailable', unix_time=time.time(),
                  hostname=socket.gethostname(), uid=os.getuid(), commands=records,
                  singularity=singularity, apptainer=apptainer, sif=str(args.sif),
                  coordination_note=args.coordination_note,
                  note='Read w/who/nvidia-smi output too; no claim that zero utilization proves an unused GPU')
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result))
    if not available:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
