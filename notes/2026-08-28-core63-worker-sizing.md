# CORE-63 worker sizing

## Scope

CORE-63 tests the decode expert-residency mechanism in the
`examples/deployment_curve_v1/` study lineage and registers the work in the
core calibration registry. It does not modify `simllm/deploy/`,
`worktrees/p2loggopsim/`, or any `codex/deploy_p2_*` branch.

## Frozen expectation

Before any retained record or scored comparison is read, the expected signed
direction is that replacing tensor-parallel-one full-model routed-expert work
with EP72 per-rank resident-expert work reduces the MoE portion of each decode
layer step. Attention, MLA, and the shared expert remain per-rank-local under
data-parallel attention. Therefore the corrected decode step must decrease and
the standard-decode calibration-only throughput prediction must increase.

Uniform routed-expert assignment is the declared assumption. The exact scale
will be derived only from disclosed architecture fields: batch per node,
top-k routing, 288 routed-expert slots, and 72 expert-parallel ranks. No fitted
constant is permitted. Decode communication is absent from the current
compute-only pricing, so overlap is a registered follow-on rather than a term
in this mechanism.

## Expected implementation size

- One field-addressed retained-record reader with an append-only access ledger.
- One expectations artifact pair frozen before result access.
- One residency calculation and publication entry point.
- One calibration-only result pair plus preservation and hash evidence.
- Focused tests and mechanical registry/task-progress updates.

Expected production and study change: roughly 500 to 900 lines across 8 to 14
files, excluding generated bulk evidence under the untracked wave-run root.

## Commit and verification plan

1. Commit this sizing note, the expectations-only freeze, the reader, and the
   open CORE-63 registry entry before accessing retained values.
2. Run the residency calculation through the committed reader, publish only
   the standard-decode calibration movement, and keep the MTP held-out value
   unread.
3. Close CORE-63 only when the published artifacts literally support closure.
4. For every commit, verify end-of-line attributes, run Ruff, and run the full
   pytest suite directly with the worktree Python 3.10 virtual environment.

All tracked paths in this note are repository-relative so the tracked-file
scanner remains portable.
