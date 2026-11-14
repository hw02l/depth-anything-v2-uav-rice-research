from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from train_patch import check_gpu_receipt


class GpuReceiptTests(unittest.TestCase):
    def test_fqdn_for_gpgpu4_is_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "receipt.json"
            path.write_text(json.dumps({
                "status": "available", "unix_time": time.time(),
                "hostname": "GPGPU4.mips.lab", "gpu_name": "NVIDIA GeForce GTX 1080",
                "gpu_free_bytes": 8 * 1024**3, "compute_processes": [],
                "work_root_realpath": "/localdata/s0623003", "singularity_version": "4.2.2",
            }), encoding="utf-8")
            result = check_gpu_receipt(path)
            self.assertEqual(result["gpu"], "NVIDIA GeForce GTX 1080")
            self.assertEqual(len(result["receipt_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
