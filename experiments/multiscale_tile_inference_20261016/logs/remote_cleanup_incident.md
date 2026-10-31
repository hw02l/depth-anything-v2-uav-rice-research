# GPGPU4 cleanup connection incident (2026-10-31)

- 09:22:57 JST: read-only preflight on GPGPU4. nvidia-smi --query-compute-apps returned no compute applications; w and who listed no interactive users; Singularity reported no running instances. The GTX 1080 showed 15 MiB for Xorg and gnome-shell display processes. This snapshot was before cleanup and is not a current post-cleanup check.
- After the result archive transfer was verified locally, cleanup job 009_multiscale_cleanup started. The client recorded PID 26516 and remote script SHA256 4494a82c5cf9faee9750d478201a8f74e188e220d7a6633d37588d92ac3b8691.
- The SSH client then reported Connection reset before emitting job_finished. No cleanup receipt or post-cleanup status was received, so the remote cleanup outcome, secondary NFS backup, cache retention, and process/container/session state remain unverified.
- On 2026-10-31, subsequent read-only audit attempts reached the DNS password prompt. They were stopped before authentication; no audit command reached GPGPU4. This turn did not repeat the GPU or interactive-user check.
- Inference had already completed and was not rerun.

No credentials are recorded here.
