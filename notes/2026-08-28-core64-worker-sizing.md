# CORE-64 worker sizing

## Classification

- Size: medium
- Risk: high, because the task changes a calibration projection under a preservation lock and must keep the held-out MTP value unread.
- Expected work: derive and preregister the EP72 per-rank shape correction, implement the field-addressed calibration-only analysis, publish its audit artifacts, update the CORE registry literally, and run the required verification.

## Intended paths

- `docs/modules/core.md`
- `notes/2026-08-28-core64-worker-sizing.md`
- A CORE-64 analysis or evidence path selected after repository inspection
- Mechanical task-progress and open-count cells in the applicable README files, if required by the registry update

## Constraints

- Do not access the held-out MTP value.
- Reject whole-file record streams by construction.
- Preserve all prior records and scored artifacts byte-for-byte.
- Keep bulk output outside the repository under `<bulk-root>/core64/`.
- Do not enter the parallel-owned deployment, `p2loggopsim`, `codex/deploy_p2_*`, or `nvcompare2` lanes.
