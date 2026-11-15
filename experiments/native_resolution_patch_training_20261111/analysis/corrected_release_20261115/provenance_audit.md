# Provenance audit

- Corrected result bundle: analysis/rgb_input_fix_20261114
- Its internal SHA256 manifest was rechecked: 43 entries, 0 failures.
- B0 gate: PASS; reference epoch20 checkpoint SHA256 52769e2e95704d0862e708e8772195b861c647dd4812058eaf698a44305c3bcb; 8 metrics within 1e-5.
- Epoch5 file identity: All three recorded hashes differ from their expected values. See checkpoint_hash_audit.csv.
- In the F2-P2 evaluation JSON, expected provenance keys absent from the current evaluator output schema: rgb_input_contract, evaluation_code_revision, checkpoint_state_load_audit.
- The stored evaluation command names the current repository evaluator, but the checkpoint hashes and output JSON fields do not reconcile with that source. The nine-condition metrics therefore remain provisional.
- NFS state: existing PR text reports a 70-file SHA256-verified backup; branch STATUS previously says backup pending. No NFS receipt or fresh readback is present in this local audit bundle, so the archive state remains independently unverified.
- No checkpoint, source image, ground-truth image, or prediction cache was modified or copied into this release directory.
