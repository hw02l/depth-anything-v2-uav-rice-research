from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(HERE.parents[1] / "experiments" / "native_resolution_patch_training_20261111"))

import continue_c0 as continuation  # noqa: E402
from patch_engine import latest_epoch, save_epoch  # noqa: E402


def make_optimizer(first: torch.nn.Parameter, second: torch.nn.Parameter):
    return torch.optim.AdamW(
        [{"params": [first], "lr": 2e-6}, {"params": [second], "lr": 2e-5}],
        weight_decay=0.01,
    )


def update(first, second, optimizer, scheduler):
    optimizer.zero_grad(set_to_none=True)
    loss = (first - 2.0).square().sum() + (second + 3.0).square().sum()
    loss.backward()
    optimizer.step()
    scheduler.step()


class ContinuationSchedulerTests(unittest.TestCase):
    def test_zero_gpu_utilization_is_not_treated_as_missing(self):
        self.assertEqual(
            continuation._shared_preflight_utilization_percent(
                {"gpu_utilization_percent": 0}
            ),
            0,
        )

    def test_missing_gpu_utilization_fails_closed(self):
        self.assertEqual(
            continuation._shared_preflight_utilization_percent({}),
            100,
        )

    def test_nonzero_gpu_utilization_is_preserved(self):
        self.assertEqual(
            continuation._shared_preflight_utilization_percent(
                {"gpu_utilization_percent": 6}
            ),
            6,
        )

    def test_initial_learning_rates_and_decay(self):
        first = torch.nn.Parameter(torch.tensor([1.0]))
        second = torch.nn.Parameter(torch.tensor([-1.0]))
        optimizer = make_optimizer(first, second)
        scheduler = continuation.make_continuation_scheduler(optimizer, total_updates=10)
        self.assertEqual(tuple(group["lr"] for group in optimizer.param_groups), continuation.GROUP_LRS)
        for _ in range(5):
            update(first, second, optimizer, scheduler)
        self.assertLess(optimizer.param_groups[0]["lr"], continuation.GROUP_LRS[0])
        self.assertLess(optimizer.param_groups[1]["lr"], continuation.GROUP_LRS[1])
        for _ in range(5):
            update(first, second, optimizer, scheduler)
        self.assertLess(optimizer.param_groups[0]["lr"], 1e-12)
        self.assertLess(optimizer.param_groups[1]["lr"], 1e-12)

    def test_resetting_lr_preserves_adamw_moments(self):
        first = torch.nn.Parameter(torch.tensor([1.0]))
        second = torch.nn.Parameter(torch.tensor([-1.0]))
        optimizer = make_optimizer(first, second)
        scheduler = continuation.make_continuation_scheduler(optimizer, total_updates=10)
        update(first, second, optimizer, scheduler)
        moments_before = [
            optimizer.state[param]["exp_avg"].clone()
            for param in (first, second)
        ]
        continuation.reset_group_lrs(optimizer)
        moments_after = [optimizer.state[param]["exp_avg"] for param in (first, second)]
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(moments_before, moments_after, strict=True)))

    def test_interrupted_resume_matches_continuous_updates(self):
        continuous_first = torch.nn.Parameter(torch.tensor([1.0]))
        continuous_second = torch.nn.Parameter(torch.tensor([-1.0]))
        continuous_optimizer = make_optimizer(continuous_first, continuous_second)
        continuous_scheduler = continuation.make_continuation_scheduler(continuous_optimizer, total_updates=20)
        for _ in range(20):
            update(continuous_first, continuous_second, continuous_optimizer, continuous_scheduler)

        split_first = torch.nn.Parameter(torch.tensor([1.0]))
        split_second = torch.nn.Parameter(torch.tensor([-1.0]))
        split_optimizer = make_optimizer(split_first, split_second)
        split_scheduler = continuation.make_continuation_scheduler(split_optimizer, total_updates=20)
        for _ in range(8):
            update(split_first, split_second, split_optimizer, split_scheduler)
        saved_parameters = (split_first.detach().clone(), split_second.detach().clone())
        optimizer_state = split_optimizer.state_dict()
        scheduler_state = split_scheduler.state_dict()

        resumed_first = torch.nn.Parameter(saved_parameters[0].clone())
        resumed_second = torch.nn.Parameter(saved_parameters[1].clone())
        resumed_optimizer = make_optimizer(resumed_first, resumed_second)
        resumed_scheduler = continuation.make_continuation_scheduler(resumed_optimizer, total_updates=20)
        resumed_optimizer.load_state_dict(optimizer_state)
        resumed_scheduler.load_state_dict(scheduler_state)
        for _ in range(12):
            update(resumed_first, resumed_second, resumed_optimizer, resumed_scheduler)

        self.assertTrue(torch.equal(continuous_first, resumed_first))
        self.assertTrue(torch.equal(continuous_second, resumed_second))
        self.assertEqual(continuous_scheduler.state_dict(), resumed_scheduler.state_dict())

    def test_checkpoint_write_is_atomic_and_discoverable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            state = {
                "epoch": 6,
                "global_step": 1152,
                "model": {"weight": torch.tensor([1.0])},
                "signature_payload": {"test": "fixed"},
            }
            path, digest = save_epoch(root, state)
            self.assertEqual(path, root / "epoch_0006.pt")
            self.assertEqual(latest_epoch(root), path)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
            self.assertEqual(torch.load(path, map_location="cpu", weights_only=False)["epoch"], 6)
            self.assertEqual(list(root.glob("*.tmp")), [])

    @patch("continue_c0.subprocess.run", return_value=SimpleNamespace(stdout="6000\n"))
    @patch("continue_c0._gpu_compute_processes", return_value={3894: 2030})
    def test_shared_gpu_monitor_accepts_unchanged_authorized_process(self, _processes, _run):
        continuation.assert_no_new_gpu_compute_processes({3894}, {3894: 2030})

    @patch("continue_c0.subprocess.run", return_value=SimpleNamespace(stdout="6000\n"))
    @patch("continue_c0._gpu_compute_processes", return_value={3894: 2300})
    def test_shared_gpu_monitor_stops_when_other_memory_grows(self, _processes, _run):
        with self.assertRaisesRegex(RuntimeError, "memory changed"):
            continuation.assert_no_new_gpu_compute_processes({3894}, {3894: 2030})

    @patch("continue_c0.subprocess.run", return_value=SimpleNamespace(stdout="6000\n"))
    @patch("continue_c0._gpu_compute_processes", return_value={3894: 2030, 4000: 512})
    def test_shared_gpu_monitor_stops_when_new_compute_process_appears(self, _processes, _run):
        with self.assertRaisesRegex(RuntimeError, "process set changed"):
            continuation.assert_no_new_gpu_compute_processes({3894}, {3894: 2030})

    @patch("continue_c0.subprocess.run", return_value=SimpleNamespace(stdout="800\n"))
    @patch("continue_c0._gpu_compute_processes", return_value={3894: 2030})
    def test_shared_gpu_monitor_stops_below_emergency_free_vram(self, _processes, _run):
        with self.assertRaisesRegex(RuntimeError, "1GiB emergency"):
            continuation.assert_no_new_gpu_compute_processes({3894}, {3894: 2030})


if __name__ == "__main__":
    unittest.main()
