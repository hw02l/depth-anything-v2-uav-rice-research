# GPGPU4 cleanup connection incident (2026-10-31)

- 09:22:57 JST: read-only preflight on GPGPU4. nvidia-smi --query-compute-apps returned no compute applications; w and who listed no interactive users; Singularity reported no running instances. The GTX 1080 showed 15 MiB for Xorg and gnome-shell display processes. This snapshot was before cleanup and is not a current post-cleanup check.
- After the result archive transfer was verified locally, cleanup job 009_multiscale_cleanup started. The client recorded PID 26516 and remote script SHA256 4494a82c5cf9faee9750d478201a8f74e188e220d7a6633d37588d92ac3b8691.
- The SSH client then reported Connection reset before emitting job_finished. No cleanup receipt or post-cleanup status was received, so the remote cleanup outcome, secondary NFS backup, cache retention, and process/container/session state remain unverified.
- 10:08:44 JST on 2026-10-31: an authenticated session ran the read-only audit job `009_recovery_inventory` (PID 26847; script SHA256 `78a078bdd5ff44e59f6072e50c71f258cb86096f7bb03a99057f3685077a4fe4`). The job exited 0 with `status=pass`; it did not rerun inference or re-query general user/GPU status.
- The audit found cleanup PID 26516 absent, no experiment-owned GPU/inference process matches, and no Singularity instances. The NFS backup `gpu4_inference_backup_multiscale_20261016_bbc7829e60b6` had matching completion hashes for `cleanup_receipt.json`, `results_package_20261016.tar.gz`, and `runtime_source_20261016.tar.gz`. The original 594,550,777-byte training backup was verified.
- Prediction-cache inventory: 72 retained arrays present and valid; 1,272 scheduled temporary arrays pruned, with none still present or invalid. The reported pruned temporary data totaled 106,276,678,656 bytes. The cleanup receipt records checkpoint hashes and `weights_updated=false`.
- The local PowerShell client remained open until its 150-second idle timeout. It emitted a `session_closed` event, printed `Connection to mips2.mips.cis.iwate-u.ac.jp closed.`, and returned to the local prompt. After that, Python emitted `Fatal Python error: _enter_buffered_busy` because a daemon stdin reader held a lock during interpreter shutdown. The SSH disconnect itself completed, the terminal prompt returned, and no remote job remained active.
- Inference had already completed and was not rerun.

No credentials are recorded here.
