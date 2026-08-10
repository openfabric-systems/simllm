# Simulated coordinator integration-review regressions

## Chronology and status

These checks were specified after implementation commit `3ed8d0e`, the first
component and live results, and the integrator's four-lens review. They are
post-specified regression checks, not public pre-registration and not additions
to the seven frozen scored instances in `expectations.md`.

This file precedes the corrective implementation. The original expectations
commit `29221e4` and its files remain unchanged. Before this review contract is
committed, both existing `--check-only` commands must pass without running the
target behavior or writing results.

## COMP-15 servable payload domain

The read-only COMP-15 validator at
`simllm/compute/nccl_stack.py:529-550` accepts a positive payload `P` only when:

1. `P` divides evenly over `world_size * channel_count * warps_per_channel`.
2. The resulting per-lane byte share divides evenly into `chunk_bytes`.
3. At least one chunk exists in every lane.

The coordinator must publish this restriction instead of implying that every
nonzero tensor shape is servable by the current lower-stack surrogate.
VLLM-20 owns removal of this compatibility restriction when native
operation-specific stack entries exist.

A multi-rank zero-byte call cannot enter COMP-15 because that validator
requires a positive payload. It must still emit one upper coordinator event
with literal stack disposition `zero_payload_bypass`, zero nested stack events,
and its normal semantic `CollectiveWork`.

An unservable nonzero call remains an explicit `ValueError`. It emits no upper
event and consumes no operation identifier. In the fixed four-rank test, an
invalid 10-byte call followed by a valid 4,096-byte call must give the valid
event both sequence zero and operation id `tp:all_reduce:0`. These exact
statuses and identifiers are author-defined fatal structural guards, not
scored behavioral evidence.

## DP return consumption

The pinned vLLM v0.26.0 runner consumes `num_tokens_across_dp` at
`vllm/v1/worker/gpu_model_runner.py:3948-3965`: it indexes the returned vector
at the local DP rank and uses that value as `num_tokens_padded`. The producing
helper places padded token counts in row one at
`vllm/v1/worker/dp_utils.py:36-54`, then returns that row at
`vllm/v1/worker/dp_utils.py:77-89`.

The skeleton must consume its simulated coordinator return through the same
logical projection. The resulting local value is written to the optional
`StepRecord.num_tokens_after_padding` field. The record-construction site in
`executor.py` remains untouched.

For the already-observed two-step live Granite workload, the post-specified
regression literal is `(4, 1)`: four prompt tokens on step zero and one decode
token on step one. The live smoke must assert those serialized field values.
Changing the coordinator return's local-rank padding value must change the
record field, which proves that the runner does not discard the return.

This exact `(4, 1)` assertion is outcome-aware and unscored. The original
engine-reachability relation remains the scored external-runtime evidence.

## Decision relevance

If padding consumption cannot remain inside the copied runner while preserving
the shared `StepRecord` contract and avoiding the parallel-owned executor
construction site, the current communicator seam is too shallow. That result
would require redesign before the GPU-present VLLM-13 half adopts it.

## Check-only commands

```bash
.venv/bin/python examples/vllm_group_coordinator_v1/run_study.py --check-only
/data3/yifeng/simllm-dev/venv-vllm/bin/python \
  examples/vllm_group_coordinator_v1/live_smoke.py --check-only
```
