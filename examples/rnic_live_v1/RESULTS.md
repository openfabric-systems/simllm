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

## Tier B live reachability

Tier B passed on 2026-08-11 and makes the first network-affected TTFT and TPOT
claim in this repository. The claim is limited to the frozen isolated
`rnic_live_v1` fixture: one request, one prefill graph, two decode graphs,
single-WQE payloads of 4 KiB and 1 MiB, the two-WQE 4 KiB FIFO, link rates of
200 and 400 Gbit/s, and native doorbell service of 0 and 1,000 ps. It does not
generalize these exact relations to congestion, headers, propagation, control
frames, packetized issue timing, or arbitrary application graphs.

The original Tier B expectations were frozen at `fc3836d`, and the review
supplement and machine-readable schema were frozen at `067cbfb`. Their final
pre-run SHA-256 values remained:

- `0058fbe9a2fe3892f739af736ce2523250e1274d4731caaef64d00497482460c`
  for the two-tier expectations;
- `be540c2b30aa300a6f92e31ca3f8bd724ca0c9e8aba6ed94240444c33693603d`
  for the original Tier B relations;
- `99c8206e9f434f671807cdcf5f060ce596794213e75da64647a93624f3f31ab2`
  for the review supplement; and
- `ef27bf3ff455fe6144337573d752bd671e8b7b6730e4915b7fdcf4f57ac7fe3d`
  for the machine-readable schema.

Implementation `8a95e3b` connected immutable composed observations to the
runtime's structural transaction and completion projection. The first
result-producing attempt stopped before publishing raw observations or a
result ledger because the protected DCQCN reference rejected a 400 Gbit/s
override against a tracked 100 Gbit/s topology. Review commit `42222d7`
corrected only that producer fixture by deriving the same 32-node topology at
400 Gbit/s for both reference and candidate runs. The successful rerun used
the unchanged registered command and frozen producer argument vector.

The composed build source resolved to the pinned htsim gitlink
`edb28c3015c173b4251abc5858c587df325e1ebc` and source tree
`238f6deab7a98efffe51ec619dd1352cbe4bb2e0`. The SimLLM native sources came
from frozen base `fc282efc91573638de7dcfae2befee1cf022011b`, tree
`ad5ed557a541c26ad061f47ac7d3a12deb223da6`. Bypass reference binaries were
rebuilt from the frozen `8c3f8b231a6a9311ffc1e7969a003dcba724b50d`
reference, while candidate bypass binaries came from the pinned main commit
with structural RNIC composition disabled.

The independent checker reported these genuine-risk fractions:

- single-WQE D additivity: 4/4;
- single-WQE inverse-rate serialization: 4/4;
- live StepResult, TTFT and TPOT forms: 8/8;
- single-WQE seven-component rows: 8/8;
- two-WQE FIFO contention: 4/4; and
- bypass artifact identity: 4/4.

All 8 single-WQE and 4 FIFO exact rows passed. Increasing D added exactly
1,000 ps to each single-WQE step latency, TTFT and decode TPOT. Doubling R
halved the wire term while retaining D as an additive constant. The FIFO W1
queue wait was exactly 163,840 ps at 200 Gbit/s and 81,920 ps at 400 Gbit/s in
every step. The objectively selected doorbell projection was `nic_owner`:
doorbell wait was zero, doorbell service was D, single-WQE request attribution
was `queue_ps = 0` and `nic_ps = D + L`, and FIFO attribution was
`queue_ps = L` and `nic_ps = D + L`. Every request's seven components summed
exactly to its live latency. Additive visit totals remained separate.

The four retained bypass profiles were `rnic-nn-fluid`, `rnic-nn`, `rnic-cn`
and `dcqcn`. Completion CSV bytes, canonical completion rows and JCT, replayed
StepResult tuples, and TTFT/TPOT summaries were byte-identical between the
protected reference and pinned candidate for every profile. All twelve fatal
unscored invariant families and all five checker-sensitivity controls held.
The raw observations contained no expected values or producer verdicts.

The published raw-observation and result SHA-256 values are
`acaca5c57134848a314a92d223c283a7dc63f1c3ef964f65f7dea75487d6dfa1` and
`3755bf5c2b37e9c30f90f97e3d6920841c70d052ff9164434d26c4f56773f0ed`.
CORE-15 and BACK-8 close on this live metric evidence.

Packet-level first-packet and last-packet issue timestamps remain outside this
claim. ABI-v1 exposes network acceptance and whole-flow terminal timestamps,
which are not substitutes for packet issue. BACK-25 owns the versioned packet
attempt and TX/RX observation vocabulary, BACK-26 owns transport-control
events, and HTSIM-9 remains open until their packet-issue evidence populates
the native timeline.
