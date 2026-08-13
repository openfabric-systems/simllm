# vLLM observed overlap v1 expectations

This document freezes a fresh TRAF-13 qualification before its harness, its
control arm, or any measured value exists. The repository source at this
boundary is commit `4e1be35af5327c27db53ed002dc420e1de6f613b`.

The earlier qualification in
[`examples/vllm_observed_schedule_v1`](../vllm_observed_schedule_v1/RESULTS.md)
is recorded void and is not edited, rescored, or reinterpreted here. Its
findings are treated as prior information only, and every quantitative figure
it reported predates the landed TRAF-25 token-ownership correction, so no
number is carried over.

## Amendment record

The control arm's edge scheme was amended once, after the first freeze and
before any arm was executed. The originally frozen scheme gated microbatch
one's pre-dispatch on microbatch zero's combine of the same layer. That is
unconstructible: both shared logical queues contribute implicit FIFO edges in
submission order, and microbatch one's pre-dispatch is submitted before
microbatch zero's combine, so the graph validator correctly rejects the result
as a cycle. The replacement below imposes the only total order those FIFO
chains admit and was confirmed constructible on one step.

At the moment of the amendment no latency, TTFT, TPOT, overlap, structure or
terminal value had been observed in any arm. The two facts observed were that
the graph is legal and that the control arm shows zero cross-microbatch
temporal overlap where the observed arm shows 8,256 overlapping visit pairs on
that step. Both are unscored construction facts. Every scored band below is
unchanged from the first freeze.

## What the void run got wrong, and what this study changes

The void run compared two arms: the serial compatibility lowering and the
observed vLLM schedule. It froze `ttft_exact_single_batch`, a guard asserting
that on a step where dual batch overlap (DBO) is configuration-forced off the
two arms must produce identical TTFT. That guard failed. The premise it
encoded, that the two arms differ only by DBO, was false: the arms also differ
by the open TRAF-9 whole-layer MoE ordering approximation and by the observed
arm's terminal logits and request-visibility frontier, neither of which is
overlap.

Because that premise was also required to attribute the measured decode
reductions to DBO, the whole run became uninterpretable as a score.

This study therefore does not compare overlap against the serial arm. It adds
a third arm that holds structure fixed and removes only concurrency.

## The three arms

Every arm consumes the same live vLLM observation stream, the same routed
supply, the same compute provider, the same requests, and the same completion
policy.

1. `serial`: observations are withheld, so `DeviceRuntimeStepSink` delegates
   to `SerialStepLowerer`. This is the compatibility arm and the explicit off
   path.
2. `control`: the observed operation tuple with cross-microbatch serialization
   edges added. Every operation identity, tuple position, rank, logical queue,
   work object, nominal duration, correlation, and completion endpoint is
   byte-identical to the `observed` arm. Only whole-operation `depends_on`
   edges are added.
3. `observed`: the observed operation tuple exactly as the live producer
   emitted it.

The `control` arm is a concurrency control, not a claim about any vLLM
configuration.

Both shared logical queues, the per-rank compute queue and the single EP
communication queue, already contribute implicit whole-operation FIFO edges in
submission order. The control arm therefore cannot invent an arbitrary order:
the only total order it can impose is the one those two FIFO chains interleave
into. Per layer that order is pre-dispatch zero, pre-dispatch one, dispatch
zero, experts zero, dispatch one, experts one, combine zero, combine one, and
then the next layer. Program order and both FIFO chains are subsequences of
it.

The edges that are not already implied, for every layer `L` in 0 through 23
and every EP rank `r`, are:

- `ubatch-0:layer-L:ep-dispatch` waits on the whole of
  `ubatch-1:layer-L:rank-r:pre-dispatch`;
- `ubatch-1:layer-L:ep-dispatch` waits on the whole of
  `ubatch-0:layer-L:rank-r:experts`;
- `ubatch-0:layer-L:ep-combine` waits on the whole of
  `ubatch-1:layer-L:rank-r:experts`;
- for `L` below 23, `ubatch-0:layer-(L+1):rank-r:pre-dispatch` waits on the
  whole of `ubatch-1:layer-L:ep-combine`.

That is 24 edges per layer for the first three rules and 8 per layer boundary
for the fourth, so exactly `24 * 24 + 23 * 8 = 760` added edges on each DBO
step and exactly zero on each single-batch step. Under that total order no
operation of one microbatch runs while any operation of the other is in
service, and because the audited source order already alternates compute and
communication inside one microbatch, the control arm realizes no
compute-communication concurrency at all.

## The decomposition this makes possible

Write `T_a` for the arm-`a` latency of one step, all three measured on the
same placement.

```text
serial minus observed = structure + overlap
structure             = T_serial  - T_control
overlap               = T_control - T_observed
```

`overlap` is the DBO effect with structure held exactly fixed. `structure` is
the TRAF-9 plus terminal-frontier difference with concurrency held exactly
fixed at zero in both terms. The identity is algebra over three measured
latencies and is a by-construction check, not evidence.

`structure` is further split by the one timestamp that separates the two named
causes. Let `moe_end_a` be the completion of the last collective operation in
arm `a` and let `terminal_a = T_a - moe_end_a`. The serial arm's last
operation is layer 23's `ep-combine`, so `terminal_serial` is exactly zero.
Then:

```text
structure = (moe_end_serial - moe_end_control) - terminal_control
```

The first term is the layer-ordering (TRAF-9) contribution and the second is
the terminal-frontier contribution. Both are measured, so the residual the
void run could not attribute is attributed here.

## Frozen inputs

The inputs are the existing three-request Granite greedy replay, unchanged:

- capture schema `simllm-preplay-trace-v1`, 120 rows, SHA-256
  `5f55fccc265ee5519430a0f73d6631e49aa547ec07fcb81034e9fc2b4d9fead6`;
- replay-run SHA-256
  `b4d38a09011caf6de159c22133264d62a2727063496953f4337b17d79cfde93e`;
- routed-experts SHA-256
  `24e986e989e21f1bfe7e758d4470928c82c3bbaec06072a839743b9b17d7cf5f`;
- model `ibm-granite/granite-3.0-1b-a400m-instruct`, revision
  `ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445`;
- 24 MoE layers, hidden width 1,024, 32 experts, top-k 8, BF16 activations,
  four resident experts per EP rank;
- requests `r0`, `r1` and `r2`, with prompt lengths 22, 12 and 20.

The live cell fixes TP 1, DP and EP 8, PP 1, `enable_expert_parallel=True`,
`all2all_backend=deepep_high_throughput`, `enable_dbo=True`, decode threshold
2 and prefill threshold 512. The replay has 32 nonempty scheduler steps: step
0 is the 54-token prefill, steps 1 through 23 are the DBO decode steps that
carry `r0`'s 23 TPOT intervals, and steps 24 through 31 are one-request decode
steps for which the frozen configuration forces DBO off. So 23 steps take
control edges and 9 do not.

The live producer is driven by the existing
[`live_driver.py`](../vllm_observed_schedule_v1/live_driver.py), reused
unchanged so the producer under qualification is the landed one, not a copy.

## Pinned source audit

The vLLM v0.26.0 source read for the earlier freeze is unchanged and is
re-verified by SHA-256 at run time. The mechanism claims and line-level
reading are in
[the earlier pinned-source audit](../vllm_observed_schedule_v1/expectations.md#pinned-source-audit);
they are not restated here because restating them would create a second
authority that could drift. The nine identities a run must observe are:

| Source file | SHA-256 |
|---|---|
| `vllm/model_executor/models/granitemoe.py` | `b60e452c3f28b25aa104c88869daa25c06a7fb6ed45bd34e908fa6a8395efda1` |
| `vllm/config/parallel.py` | `a6581c267ab265e24905d2f5caa514482c28359f71380c6f894ceab25aa22541` |
| `vllm/v1/worker/gpu_model_runner.py` | `81b7627fbe81f7aaa2f77b4bf085faa353c69d03662ebfe369536a9773bb70d0` |
| `vllm/v1/worker/dp_utils.py` | `2ba84bbf92a25e756576918bfb215c1fb387b006899885d811bdb2f774e843a9` |
| `vllm/v1/worker/ubatch_utils.py` | `0b727aaa1c7072152e25f684ddc2fc9790c430eddd862e610c97a8e3e9febdc4` |
| `vllm/v1/worker/gpu_ubatch_wrapper.py` | `4eae50c929f3ba873072c13291c7140be3dd00d4a5b623170dff44754519c021` |
| `vllm/v1/worker/ubatching.py` | `40391241c564feb5f16c77898ae6ae152ed6e71a4682e2a406387785d8de02d7` |
| `vllm/model_executor/layers/fused_moe/modular_kernel.py` | `f78ae626babfd69f3c6ba37eef9c8f5186f28cd9064f566e341ca0c9e0fdb9b9` |
| `vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_ht.py` | `465cdf1d6cee91b2ee8c2e43abbea6e8408976e3048c10f44c089f34b415bc60` |

The evidence is authored against official vLLM v0.26.0 commit
`568afb3a13806beb53bb2e6bd518269357b237c0`. Authored-against and observed
identities stay separate provenance fields. No check may require a live
checkout, package or submodule pin to equal the authored-against commit.

## The available effect, computed before the run

The realizable overlap is bounded above by the communication a step actually
contains. That bound is arithmetic over frozen inputs and is computed here,
before any arm exists, rather than hoped for afterwards.

The coarse device runtime charges an `all-to-allv` at the maximum per-source
egress divided by that source's link rate. Ingress is not charged, so the
combine incast costs one source's return leg, not the sum. Under the landed
TRAF-25 token-ownership correction one `StepRecord` has one engine source, so
dispatch egress leaves rank 0 alone and combine collapses to a single remote
rank's return. Total routed bytes and the communication term therefore move by
very different factors from the pre-TRAF-25 tables, and no earlier figure is
reused.

Summing the maximum per-source egress over every collective invocation of a
DBO decode step, with the token split vLLM applies, and averaging over `r0`'s
23 TPOT steps gives 15,071,232 bytes over 23 steps, that is 655,270.96 bytes
per step. At the two placement rates:

| Placement | Link rate | Communication ceiling per decode step |
|---|---:|---:|
| `single-node` | 3,600,000,000,000 bit/s | 1,456,158 ps |
| `cross-node` | 400,000,000,000 bit/s | 13,105,420 ps |

The per-step compute term is about 99,400,000 ps and is nearly constant across
these step sizes because decode is weight-read bound. So on a single node the
entire communication of a decode step is about 1.5 percent of its compute, and
no scheduling policy can recover more than that. Cross-node it is about 13
percent. The honest expectation is therefore a small single-node effect and a
roughly ninefold larger cross-node effect, and this study registers that from
arithmetic rather than from hope. If an effect is near zero on a placement,
that is a finding about where overlap matters, not a failure of the producer.

A second arithmetic prediction follows from the same reasoning. In the
steady-state two-microbatch pipeline the compute resource is saturated and
every collective except the final one hides behind the other microbatch's
compute, so the realized overlap should approach the ceiling from below and
fall short of it by roughly one combine service.

## Frozen behavioral relations

All raw per-step latencies for all six cells are collected first. Every scored
relation below is evaluated from those raw rows before any exact identity,
physical ceiling or other fatal guard is interpreted, so no fatal oracle can
entail a scored relation.

`r0` TPOT is the exact mean of its 23 decode intervals. `r0` TTFT is the
completion of the single-batch prefill step.

### B1. The overlap effect sits inside its arithmetic band

Two instances, one per placement. `overlap = TPOT_control - TPOT_observed`
for `r0`.

| Placement | Inclusive band | Fraction of the ceiling |
|---|---:|---|
| `single-node` | 1,092,119 through 1,456,158 ps | 0.75 through 1.00 |
| `cross-node` | 9,829,065 through 13,105,420 ps | 0.75 through 1.00 |

The upper end is the physical ceiling above. The lower end is a real gate: a
lowering that reached the runtime but failed to realize the source-legal
concurrency would produce an overlap near zero and miss both bands.

### B2. The overlap effect scales with the link rate

One instance. `overlap_cross / overlap_single` must lie in the inclusive band
7.5 through 10.5. The expectation is 9.0, the exact ratio of the two link
rates, because overlap is bounded by and closely tracks a term that is pure
serialization.

### B3. The terminal frontier does not scale with the link rate

One instance. `terminal_observed`, averaged over `r0`'s 23 decode steps, is
the time from the last collective completion to step completion in the
observed arm. Its cross-node to single-node ratio must lie in the inclusive
band 0.95 through 1.05, with an expectation of 1.0, because the terminal
frontier is per-rank logits compute plus a zero-duration visibility endpoint
and moves no bytes.

B2 and B3 together are the discriminator this task exists to produce: the same
two placements must move one term by about nine and the other not at all. A
terminal fan-in charged to the fabric would fail B3, and an overlap effect that
was really a compute artifact would fail B2.

### B4. The corrected single-batch TTFT relation

Two instances, one per placement. On the 54-token prefill step DBO is
configuration-forced off, so the control arm adds no edge and the overlap term
is exactly zero. The frozen expectation is the corrected form of the guard
that voided the earlier run:

- `TTFT_serial` is strictly greater than `TTFT_observed`; and
- the entire difference is structural.

The void run asserted equality here. Asserting strict inequality is a genuine
test, because equality remains the outcome if the refuted premise were right,
and the sign could a priori have gone the other way.

Six scored instances in total: two for B1, one for B2, one for B3 and two for
B4.

## Fatal unscored guards

A violation of any guard below voids the run. Fatal means void, not a lost
point, and these are never reported as a pass fraction.

Control-arm construction:

1. `control_differs_only_by_edges`: on every step and both placements, the
   control and observed operation tuples agree on operation identity, tuple
   position, rank, logical queue, work object, correlation, `not_before_ps`,
   priority and completion endpoints, and the control `depends_on` set is a
   superset of the observed one.
2. `control_edge_counts`: exactly 760 added edges on each of the 23 DBO steps,
   exactly 0 on each of the 9 single-batch steps, so 17,480 in total.
3. `control_has_no_cross_microbatch_concurrency`: in the control arm, no queue
   visit belonging to microbatch zero overlaps in realized time with any queue
   visit belonging to microbatch one, on any step.
4. `overlap_zero_on_single_batch_steps`: the per-step control and observed
   latencies are equal on step 0 and on steps 24 through 31.
5. `decomposition_identity`: `serial minus observed` equals
   `structure + overlap` exactly, and `structure` equals the layer-ordering
   term minus the terminal term exactly, on every step and both placements.

Physical bounds:

6. `overlap_within_ceiling`: the measured overlap does not exceed 1.05 times
   the arithmetic ceiling on either placement. Exceeding a serialization
   ceiling is proof of a defect somewhere.
7. `decode_tpot_bounds`: every arm's `r0` TPOT lies between 69,206,016 ps, the
   553,648,128-byte weight and LM-head read at 8 TB/s, and 150,000,000 ps, the
   compute term plus a fully serialized cross-node communication term.
8. `prefill_ttft_bounds`: every arm's `r0` TTFT lies between 69,206,016 ps and
   500,000,000 ps. The cross-node prefill communication ceiling alone is
   304,988,160 ps for 15,249,408 bytes of summed peak per-source egress.

Producer fidelity, unchanged in meaning from the earlier freeze:

9. `producer_every_nonempty_step`: all 32 nonempty translated steps emit
   observations.
10. `producer_layer_and_site_inventory`: every step covers layers 0 through 23
    and exactly 48 unique `(layer, site)` values, one dispatch and one combine
    per layer, with 96 invocations on DBO steps and 48 on single-batch steps.
11. `producer_request_partitions`: microbatch request slices are disjoint, in
    source order, and partition the scheduled requests.
12. `producer_completion_frontiers`: one request-correlated completion
    endpoint per active microbatch.
13. `producer_compute_conservation`: per-rank compute in the observed tuple
    equals the serial arm's represented compute on the same step.
14. `producer_no_overlap_knob`: the producer source contains no overlap
    fraction, percentage or amount.

Preservation through the metric chain:

15. `schedule_fields_survive_lowering`: in the observed and control arms, every
    tuple position, queue, dependency of both scopes, gate, priority,
    correlation and completion identity survives traffic binding into the
    lowered graph.
16. `request_pair_rebinding_exact`: every per-step request-pair routed table
    is identical across all three arms and both placements, and completion
    reduction returns the original `r0`, `r1` and `r2` identities while those
    requests are active.
17. `live_cross_node_identity`: the independent cross-node replay executed
    inside the live driver matches the harness graph, execution-result and
    `StepResult` identities on every step.

The explicit off path:

18. `serial_direct_all_steps`: with observations withheld, the sink's graph
    JSON, execution result and `StepResult` equal an independently constructed
    direct `SerialStepLowerer` run on every step and both placements.
19. `serial_fixture`: the accepted fixed serial fixture still renders 4,127
    bytes of canonical graph JSON with SHA-256
    `aa3c836fe559973a7bf0940384c2e8a84e6af84e0fbd2c02d3b89774ee0c8e2d` and
    1,880 bytes of legacy direct diagnostic GOAL with SHA-256
    `7087db6780f7e34f5a559a6505eeccc15d984c7b478cd8f0bc5838053825d4b6`.

Guards 1 through 5 and 9 through 19 are conservation identities, construction
facts or fixed hashes and never enter a behavioral denominator. Guards 6
through 8 are physical bounds. None of them pins a scored value: the scored
bands are evaluated first, and the only guard that touches a scored quantity,
guard 6, is strictly looser than B1's upper end.

## What a void looks like

If any guard above fails, the run is reported void with findings, the evidence
is retained, and TRAF-13 stays open. In particular, refuting the arithmetic
ceiling, refuting the control arm's zero-concurrency property, or finding that
the added edges changed something other than dependencies are all legitimate
and valuable outcomes, and all of them are void rather than closure.

No scored band is survivable-by-exception. No fatal guard is nominated as
survivable. If a scored relation fails, the run remains interpretable, the
failure is reported as a failure, and the undemonstrated clause takes a new
task ID.

## Reproduction

Configure the machine-local roots, then run from the repository root. The
selected output directory must not already exist.

```bash
.venv/bin/python examples/vllm_observed_overlap_v1/run_study.py \
  --capture "${SIMLLM_MOE_E2E_ROOT:?configure SIMLLM_MOE_E2E_ROOT}/capture/granite-greedy.jsonl" \
  --replay-run "${SIMLLM_MOE_E2E_ROOT:?configure SIMLLM_MOE_E2E_ROOT}/replay-400g-nvlink/run.json" \
  --routed-experts "${SIMLLM_MOE_E2E_ROOT:?configure SIMLLM_MOE_E2E_ROOT}/replay-400g-nvlink/routed-experts.json" \
  --vllm-source "${SIMLLM_VLLM_SOURCE:?configure SIMLLM_VLLM_SOURCE}" \
  --vllm-python "${SIMLLM_VLLM_PYTHON:?configure SIMLLM_VLLM_PYTHON}" \
  --output-dir "${SIMLLM_OVERLAP_RUN_ROOT:?configure SIMLLM_OVERLAP_RUN_ROOT}/qualification"
```

Adding `--check-only` repeats the complete artifact-free registry validation:
it imports no target module, creates no output directory and writes no
artifact.
