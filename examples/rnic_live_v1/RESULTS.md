# Live RNIC composition: Tier A results

## Claim scope

This report claims TIER A of the frozen two-tier gate and nothing more.
Per the frozen expectations, until Tier B passes, the native composition
is component and step-sink evidence, BACK-8 remains open, and no
`CompletionEvent`, `ExecutionResult`, TTFT or TPOT claim is made here.
Packet-level first and last packet issue timestamps remain
`not_applicable` at the ABI-v1 flow-extent port; the packetized event
vocabulary is BACK-25 and BACK-26 scope, and HTSIM-9 therefore remains
open as an ABI-v1 checkpoint.

## Freeze provenance

The expectations were frozen before any implementation at commit
`65b5609`, amended pre-run at `facb26d` and `947399c`, with the final
lineage recorded at `d5d98a2`. The frozen file was never edited after
its freeze; the Tier A harness and its port-factory seam were landed and
separately frozen by the preparation work
([tier_a_harness_expectations.md](tier_a_harness_expectations.md),
fake-port precedent in
[tier_a_harness_results.md](tier_a_harness_results.md)). The Tier B run
expectations are frozen in [tier_b_expectations.md](tier_b_expectations.md)
with the review supplement, awaiting execution.

## The composed binary

The htsim side landed on the backend main through its PR 11: the
combined session implements `AtlahsFlowRuntime`, the SimLLM native RNIC
library is compiled from the pinned SimLLM checkout behind the
`HTSIM_ENABLE_SIMLLM_RNIC` build option, opaque tokens only cross the
boundary, structural mode constructs no legacy ledger (exclusivity
counters wired to observations), backpressure is relayed from htsim
rather than fabricated (a review round removed a fabricated-drop path
and added runtime-bound multi-flow tests plus a composed eight-rank GOAL
smoke with three overlapping flows), and completion projections carry
stable directed-pair transport identities. The submodule pin in this
change points at that backend main merge commit, which also carries the
full former addon-branch lineage, so the frozen bypass reference
`8c3f8b2` is reachable from the pinned history.

## Tier A evidence

Scored and fatal families, from the frozen acceptance harness driven
against the composed producer (`--factory htsim`), independently rebuilt
and rerun twice by the integrator from fresh build trees:

- Exact oracle rows: 8 of 8.
- D-additivity over payload by rate: 4 of 4 (the 1,000 ps native
  doorbell shift is additive at both link rates).
- Inverse-rate serialization: 4 of 4 (the wire term halves exactly when
  the rate doubles).
- Two-WQE FIFO: 4 of 4 (the frozen four-equation timeline, W1 queue wait
  exactly L(R), no bypass of W0).
- Fatal unscored invariant families, all holding: authority exclusivity,
  token conservation, controlled drop producing the modeled error
  completion and never a success CQE, FIFO ordering, terminal atomicity,
  quiescence, and wrapper-bypass sensitivity (the bypass mutant is
  rejected by the same D predicate that accepts the composition).
- Full backend test suite: 363 of 363 native tests on the composed
  build; the legacy `htsim_uec` one-flow scenario is byte-identical to
  the protected reference `8c3f8b2` (stdout, stderr, `logout.dat`,
  `idmap.txt`, `flowsInfo.csv`).

Step-sink replay half of Tier A: the complete pre-registered m4 study
(sections A through E, 36 rows: fluid closed forms, packetized point
forms, the tp=8 replay, and both recorded frontend replays) reran with
`SIMLLM_HTSIM_RNIC` pointing at the composed binary. Every row
reproduced with zero FAIL verdicts and every residual exactly 0 ps, so
the composed binary is a drop-in producer for the accepted step-sink
path.

## What Tier A passage does not claim

The DeviceRuntime-to-TTFT/TPOT chain (Tier B) has not been executed
against the composed binary; its expectations are frozen and its
producer invocation contract is pinned, so the run requires no further
agreement. BACK-8 remains open for live metric reachability. The
per-WQE packet-issue timeline remains unclaimed at ABI-v1.

Postscript, 2026-08-11: BACK-25 and BACK-26 later closed at the vocabulary and
relay boundary; see [the ABI-v2 packet study](../rnic_packet_v2/RESULTS.md) and
its registered producer residuals.
