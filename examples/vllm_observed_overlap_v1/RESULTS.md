# vLLM observed overlap v1 results

TRAF-13 is closed. A real eight-rank vLLM v0.26.0 replay emitted
`ExecutionObservations` for all 32 nonempty scheduler steps, and a third,
structure-matched control arm separated the dual batch overlap (DBO) effect
from the TRAF-9 layer-ordering and terminal-frontier differences instead of
assuming they were absent. All 19 fatal unscored guards passed. Four of six
scored instances passed.

The headline is a measurement, not an assumption:

| Term | Single-node NVLink | Cross-node 400 Gbit/s |
|---|---:|---:|
| Overlap, `control - observed` | 1,450,472.652 ps | 13,051,993.043 ps |
| Structure, `serial - control` | -903.913 ps | -7,123.478 ps |
| of which layer ordering (TRAF-9) | +17,973,426.304 ps | +17,967,206.739 ps |
| of which terminal frontier | -17,974,330.217 ps | -17,974,330.217 ps |
| Observed `r0` TPOT | 99,481,763.174 ps | 99,529,252.174 ps |

Overlap removes 1.437 percent of the control arm's TPOT on one node and
11.593 percent across nodes. The two structural terms are each about 18
microseconds and cancel to under 0.007 percent of TPOT, because the producer
moves the LM-head compute out of the 24 layers into a terminal logits operation
while conserving total compute. The void run could not separate these; this run
measures all three.

## Chronology and provenance

Expectations were frozen in commit `c52d07129052affcd33f8b46926237da97e21a81`
before any harness existed, and amended once in commit
`a8bb18e380fa640719468fedf94684add5b94322`. The amendment replaced an
unconstructible control-arm edge scheme, is recorded in
[the expectations](expectations.md#amendment-record), and changed no scored
band. At that moment no latency, TTFT, TPOT, overlap, structure or terminal
value had been observed in any arm. The registered command passed with
`--check-only` before the freeze, imported no target module, created no output
directory and wrote no artifact.

Two debug runs preceded the registered run and are disclosed rather than
hidden. The first exposed three harness defects and its scored outcome was
seen before they were fixed:

1. the serial fixture built the plan-attached graph instead of the accepted
   absent-plan wire form, so it compared 19,664 bytes against the frozen
   4,127-byte identity;
2. the edge-free graph identity included `released_at_ps`, which legitimately
   differs between arms because each arm owns its own `VirtualClock`;
3. the critical-path diagnostic read
   `RuntimeReport.realized_critical_path_segments` as
   `(operation_id, duration)` when it is `(operation_id, participant_rank)`.

All three are harness defects, all three were fixed to measure what the freeze
already named, and no frozen literal, band, guard or expectation was changed
after any number was observed. The second debug run passed every guard and was
followed by the registered run reported here.

The registered run observed repository commit
`a8bb18e380fa640719468fedf94684add5b94322` with the harness present in the
working tree; the harness landed unchanged in commit
`0bbc27062a148c1abc3363434cf7257448a91aaf`. The
evidence is authored against official vLLM v0.26.0 commit
`568afb3a13806beb53bb2e6bd518269357b237c0`; the run independently observed
version `0.26.0` and all nine frozen source-file hashes. Authored-against and
observed identities remain separate provenance fields with no equality
assumption. The 120-row capture was observed at its frozen SHA-256
`5f55fccc265ee5519430a0f73d6631e49aa547ec07fcb81034e9fc2b4d9fead6`.

The external result is at
`$SIMLLM_OVERLAP_RUN_ROOT/qualification/results.json` with SHA-256
`48695beea629b467e85798b5e9d5038311177186eadbc3bd37ee6fdba989470e`. The live
observation stream has SHA-256
`9f6dcc426e75baea2efa19ce53a6f3e0b3909c8482965dd8869b87053a52772c`.
Reproduce with the command in
[the expectations](expectations.md#reproduction).

## Evidence classes

The classes answer different questions and are never added together.

| Evidence class | Count | Outcome |
|---|---:|---|
| Physical placement configurations | 2 | Single-node NVLink and one rank per node at 400 Gbit/s |
| Raw runtime cells | 6 | Serial, control and observed on both placements, 32 steps each |
| Behavioral instances | 6 | 4 passed, 2 failed (both are the same refuted TTFT direction) |
| Fatal unscored guards | 19 | All passed |
| Live framework ranks | 8 | All exited successfully |
| Live nonempty scheduler steps | 32 | 32 emitted observations |
| Control edges added | 17,480 | 760 on each of 23 DBO steps, 0 on each of 9 single-batch steps |
| Schedule preservation checks | 128 | 32 steps times two arms times two placements, all passed |
| Direct serial per-step comparisons | 64 | All graph, execution-result and `StepResult` identities passed |
| Fixed serial compatibility fixture | 1 | Both frozen byte identities passed |
| Full Python regression | 1 | 1,307 passed, 7 skipped |
| Repository lint | 1 | Passed |

## The control arm, and why it is the point

The void run compared the serial compatibility lowering against the observed
schedule and froze a guard asserting that on a step with DBO forced off the two
must be identical. They were not, so the run was void. The premise that the
arms differ only by DBO was false, and that premise was also what attributed
the decode reductions to DBO.

This study adds a `control` arm: the observed operation tuple with only
cross-microbatch serialization edges added. Everything else is held identical
by construction and checked. Both shared logical queues already impose FIFO in
submission order, so the control cannot invent an order; it imposes the single
total order those FIFO chains interleave into. Per layer that is pre-dispatch
zero, pre-dispatch one, dispatch zero, experts zero, dispatch one, experts one,
combine zero, combine one.

Under that order the control realizes no compute-communication concurrency at
all, which is measured rather than asserted: on every step and both placements
no queue visit of one microbatch overlaps in realized time with any queue visit
of the other, so the count of intersecting busy intervals between the two
microbatches is zero on all 32 steps. The observed arm has 95 such intersecting
intervals on every DBO step.

The originally frozen edge direction, gating microbatch one's pre-dispatch on
microbatch zero's combine, is unconstructible for exactly this reason: it runs
against the queue FIFO order and closes a cycle. That is now a repository
regression test.

## Decomposition

Write `T_a` for the arm-`a` latency of one step and `moe_end_a` for its last
collective completion. The serial arm's last operation is layer 23's combine,
so its terminal term is exactly zero on every step, which the run confirmed.

```text
serial minus observed = structure + overlap
structure = (moe_end_serial - moe_end_control) - terminal_control
```

Averaged over `r0`'s 23 decode intervals:

| Placement | Serial | Control | Observed | Structure | Overlap |
|---|---:|---:|---:|---:|---:|
| `single-node` | 100,931,331.913 | 100,932,235.826 | 99,481,763.174 | -903.913 | 1,450,472.652 |
| `cross-node` | 112,574,121.739 | 112,581,245.217 | 99,529,252.174 | -7,123.478 | 13,051,993.043 |

Both identities held exactly on every step and both placements.

The structural residual is fully explained and is neither TRAF-9 nor the
terminal frontier. Both arms carry exactly the same per-rank compute, so the
whole residual is communication accounting. Splitting each collective into two
microbatch collectives raises the summed maximum per-source egress whenever the
peak source differs between the two microbatch tables. Over `r0`'s 23 steps the
frozen routed table gives 8,192 extra bytes, that is 356.174 bytes per step:

| Placement | Predicted split penalty | Measured structure | Residual |
|---|---:|---:|---:|
| `single-node` | 791.5 ps | 903.913 ps | 112.4 ps |
| `cross-node` | 7,123.5 ps | 7,123.478 ps | 0.02 ps |

The cross-node prediction is exact to two parts in a million. The single-node
residual of 112.4 ps is per-extent rounding: the runtime rounds each extent's
serialization up, the 96-collective control has more extents than the
48-collective serial arm, and at 400 Gbit/s every 2,048-byte extent is a whole
number of picoseconds so there is nothing to round. The measured
critical-path communication confirms this directly: the control arm minus the
serial arm is 903.913 ps single-node and 7,123.478 ps cross-node, which equals
the structural residual exactly.

## Scored relations

Raw per-step latencies for all six cells were collected first, then the
decomposition, then the scored relations, then the producer inventory, the
fixture and the fatal guards. No fatal guard pins a scored value.

### B1. The overlap effect sits inside its arithmetic band, 2 of 2 passed

| Placement | Overlap | Frozen band | Ceiling | Fraction of ceiling |
|---|---:|---:|---:|---:|
| `single-node` | 1,450,472.652 ps | 1,092,119 to 1,456,158 ps | 1,456,158 ps | 0.996096 |
| `cross-node` | 13,051,993.043 ps | 9,829,065 to 13,105,420 ps | 13,105,420 ps | 0.995923 |

The ceilings were computed from the frozen routed table before the harness
existed. The realized overlap approaches each ceiling from below and falls
short of it by roughly one combine service, which is what the freeze predicted
from the steady-state pipeline argument.

### B2. Overlap scales with the link rate, 1 of 1 passed

The cross-node to single-node ratio is 8.998441, against the exact link-rate
ratio of 9.0 and a frozen band of 7.5 to 10.5. The 0.017 percent shortfall is
the same per-extent rounding described above.

### B3. The terminal frontier does not scale with the link rate, 1 of 1 passed

The mean terminal term is 17,974,330.217 ps on both placements, so the ratio is
exactly 1.000000 against a frozen band of 0.95 to 1.05. The terminal frontier
is per-rank logits compute plus a zero-duration visibility endpoint and moves
no bytes.

B2 and B3 are the discriminator this task existed to produce. The same pair of
placements moves one term by 8.998 and the other by exactly 1.000, so the
overlap effect cannot be a compute artifact and the terminal frontier cannot be
a fabric cost.

### B4. The corrected single-batch TTFT relation, 0 of 2 passed

| Placement | Serial TTFT | Observed TTFT | Difference | Frozen expectation |
|---|---:|---:|---:|---|
| `single-node` | 133,223,654 ps | 133,223,654 ps | 0 | strictly positive; failed |
| `cross-node` | 404,324,160 ps | 404,324,160 ps | 0 | strictly positive; failed |

This is a refutation of a frozen expectation and it is reported as a failure,
not reinterpreted. The freeze predicted the serial arm would remain strictly
slower on the single-batch prefill, as it was in the void run. At this
repository state the two arms are exactly equal on both placements.

The measured reason is available because of the decomposition. On a
single-batch step there is no microbatch split, so the communication accounting
is identical in both arms, and the only remaining difference is where the
LM-head compute sits: inside the 24 layers in the serial arm, in a terminal
logits operation in the observed arm. The serial arm's MoE phase ends
17,975,818 ps later and the observed arm's terminal term is 17,975,818 ps, so
they cancel exactly. The structure of a single-batch step is exactly zero, not
merely small.

Two consequences worth stating plainly. First, the void run's
`ttft_exact_single_batch` guard would pass at this repository state, but that
does not unvoid it: its numbers were measured against the pre-TRAF-25
source-multiplied routed table and a different collective service model, and no
figure from it is portable. Second, a frozen expectation was wrong in both
qualifications now, in opposite directions, which is the argument for measuring
the residual rather than predicting its sign.

## Physical sanity

Three independent checks were made before the mechanism findings were
interpreted.

Compute and memory physics. The 553,648,128 resident weight and LM-head bytes
at 8 TB/s give a 69,206,016 ps floor. The 0.7-efficiency roofline predicts
99,488,866.130 ps of compute per decode step, and the represented compute the
graphs actually carry is 99,475,826.087 ps, 13,040 ps lower because the serial
lowerer truncates each layer boundary to whole GOAL nanoseconds. Both arms
carry that same represented value, which is why the whole structural residual
is communication. Observed TPOT was 99,481,763.174 ps single-node, that is the
represented compute plus 5,937.087 ps of exposed communication.

Network and serialization physics. The mean summed maximum per-source egress
of a decode step is 654,914.783 bytes as 48 whole collectives and
655,270.957 bytes as 96 microbatch collectives. At the two rates those are
1,455,366.2 and 1,456,157.7 ps single-node, and 13,098,295.7 and
13,105,419.1 ps cross-node. Measured critical-path communication was
1,455,505.826 and 1,456,409.739 ps single-node, and 13,098,295.652 and
13,105,419.130 ps cross-node. The cross-node pair is exact. Overlap removed
1,450,472.652 and 13,051,993.043 ps of that communication from the realized
critical path, which equals the measured TPOT reduction to the picosecond.

Prefill and end-to-end plausibility. The 54-token prefill carries 15,249,408
bytes of summed peak per-source egress, so its cross-node serialization floor
is 304,988,160 ps. It is a single-batch step, so nothing overlaps and TTFT
should be compute plus communication: 99,340,800 + 304,988,160 = 404,328,960 ps
against a measured 404,324,160 ps, 0.0012 percent below the sum through
per-extent rounding. Single-node the same arithmetic gives 133,228,373 against
a measured 133,223,654 ps. Observed TPOT implies about 10,052 and 10,047 tokens
per second for `r0`. That is an ideal-model figure for a 400M-active-parameter
model with an analytic roofline and no absolute calibration; NVIDIA's
[Blackwell inference report](https://developer.nvidia.com/blog/blackwell-breaks-the-1000-tps-user-barrier-with-metas-llama-4-maverick/)
reports more than 1,000 tokens per second per user for a much larger 400B
Llama 4 Maverick deployment on eight B200 GPUs, so the figure is not
physically impossible for a model three orders of magnitude smaller in active
parameters. Absolute accuracy is not claimed. TRAF-11 owns the flat NVLink
rate, TRAF-14 owns immutable physical collective expansion and TRAF-23 owns
measured completion frontiers.

## Where overlap matters

The freeze registered the size of the available effect from arithmetic before
any arm ran, and the run confirmed it. On a single node a decode step contains
about 1.456 microseconds of communication against about 99.5 microseconds of
compute, so even a perfect schedule can recover at most 1.44 percent of TPOT
there. The realized 1.437 percent is 99.61 percent of that ceiling: the
mechanism is working essentially perfectly and the effect is still small,
because there is almost nothing to hide.

Cross-node the same step carries 13.105 microseconds of communication and the
realized overlap is 11.593 percent of the control arm's TPOT. This is the
finding: DBO is a cross-node mechanism at this model size. Reporting a
single-node overlap number as evidence that a schedule producer matters would
be reporting the 1.4 percent that the physics allows, not the mechanism. The ninefold ratio between the
placements is the same ninefold ratio between the link rates, so the effect is
set by the link, not by the schedule.

## Preservation through the metric chain

Every captured order and dependency fact survived. The 128 schedule
preservation checks, 32 steps for each of the observed and control arms on both
placements, confirmed that traffic binding preserves tuple position, rank,
logical queue, both dependency scopes, `not_before_ps`, priority, correlation
and completion identities.

The control arm differs from the observed arm by dependency edges alone. The
whole-graph identity with every edge removed is equal between the two arms on
every step and both placements, and the per-operation field comparison agrees
on identity, position, rank, queue, work object, correlation, gate, priority
and placement epoch, with the observed `depends_on` set a subset of the
control's.

All six cells produced the same 15,000 request-pair rows and the same
54,218,752 directed bytes per cell, and completion reduction returned the
original `r0`, `r1` and `r2` identities while those requests were active. These
byte figures are post-TRAF-25 and are guarded independently by the VLLM-24
conservation check in `simllm.traffic.routed_conservation`, which both
`lower_step_observations` and `render_step_goal` run on the full-step routed
plan.

The live cross-node replay executed inside the vLLM driver matched the
harness's graph identity, execution-result identity and `StepResult` on all 32
steps.

## Producer inventory

All eight vLLM ranks ran the frozen Granite model revision through the actual
v0.26.0 scheduler and `SimWorker` model-runner seam. Rank zero was the sole
schedule, sink and virtual-clock authority.

| Step shape | Steps | Collective invocations | Unique `(layer, site)` values | Completion endpoints |
|---|---:|---:|---:|---:|
| Single batch | 9 | 48 | 48 | 1 |
| Two-way DBO | 23 | 96 | 48 | 2 |

Every step covered layers 0 through 23 with exactly one dispatch and one
combine site per layer. Microbatch request slices were disjoint, in source
order, and partitioned the scheduled requests. Per-rank compute in the observed
tuple equalled the serial arm's represented compute on every step. The producer
source contains no overlap fraction, percentage or amount, and derives no edge
from the compatibility graph.

## Exact serial off path

With observations withheld, `DeviceRuntimeStepSink` delegated to
`SerialStepLowerer`. All 64 per-step direct comparisons of graph JSON,
execution result and `StepResult` passed on both placements, and 32 GOAL
renderings were recorded per placement. The accepted fixture is unchanged:

| Artifact | Bytes | SHA-256 | Outcome |
|---|---:|---|---|
| Canonical absent-plan execution graph JSON plus LF | 4,127 | `aa3c836fe559973a7bf0940384c2e8a84e6af84e0fbd2c02d3b89774ee0c8e2d` | Passed |
| Legacy direct diagnostic GOAL | 1,880 | `7087db6780f7e34f5a559a6505eeccc15d984c7b478cd8f0bc5838053825d4b6` | Passed |

The plan-attached default graph differs from that fixture by its
`CollectivePlan` alone, which the run also checked.

## Entailment and evidence accounting

The six raw mode workers completed before the decomposition, and the
decomposition completed before the scored relations. The producer inventory,
the fixture and all 19 fatal guards were interpreted afterwards.

No fatal guard pins a scored value. The only guard that touches a scored
quantity is `overlap_within_ceiling`, whose bound is 1.05 times the arithmetic
ceiling and is therefore strictly looser than B1's upper end of 1.00 times it.
The `decomposition_identity` guard is algebra over three measured latencies and
is by construction. `overlap_zero_on_single_batch_steps` and
`control_edge_counts` are construction facts. Source hashes, byte conservation,
serial identities and the absent overlap knob are fixed-value or
configuration-forced guards. None of them increases a behavioral denominator.

Each scored instance could fail after reaching execution. An overlap outside
either band would miss B1. A lowering that ignored the observation tuple would
produce zero overlap and miss B1 and B2. A terminal frontier charged to the
fabric would miss B3. B4 did fail, which is itself the demonstration that these
were live tests rather than restatements of the model.

| Frozen behavioral family | Instances | Passed |
|---|---:|---:|
| B1 overlap band | 2 | 2 |
| B2 overlap rate ratio | 1 | 1 |
| B3 terminal rate invariance | 1 | 1 |
| B4 single-batch TTFT direction | 2 | 0 |
| Total | 6 | 4 |

## TRAF-13 evidence map

> "qualify at least one real framework schedule producer through
> `ObservedStepLowerer`."

The live vLLM v0.26.0 producer passed observations through
`DeviceRuntimeStepSink` and `ObservedStepLowerer` on all 32 nonempty steps, on
both placements, in both the observed and control arms.

> "A future expectations-only qualification must distinguish DBO from the
> TRAF-9 and terminal-frontier differences"

Done by construction and by measurement. The structure-matched control arm
isolates overlap at 1,450,472.652 and 13,051,993.043 ps, while the
layer-ordering term is +17,973,426.304 and +17,967,206.739 ps and the terminal
term is -17,974,330.217 ps on both placements. The residual after those two
cancel is -903.913 and -7,123.478 ps and is separately explained by the
microbatch split's byte accounting to two parts in a million cross-node.

> "preserve every captured order and dependency fact through traffic binding,
> `DeviceRuntime`, `CompletionEvent`, `StepResult`, TTFT and TPOT"

128 schedule preservation checks, 192 per-step request-pair identities across
six cells, request-attributed `StepResult`, and `r0` TTFT and TPOT in every
cell.

> "and show a registered live-metric relation."

B1 on both placements and B2 across them, all on `r0` TPOT, the repository's
signature metric. B3 is the accompanying invariance.

> "Disabling the producer must select the serial lowerer and preserve every
> accepted serial graph, GOAL byte, timestamp and completion order exactly."

64 per-step direct serial comparisons and both accepted fixture identities
passed.

> "Routed-byte acceptance depends on TRAF-25 and VLLM-24, both of which have
> landed."

Both are landed and the run consumed them: the routed tables are post-TRAF-25
and the VLLM-24 conservation guard runs inside the lowering path on every step.

Every registered clause is demonstrated, so TRAF-13 closes and **zero new task
IDs are registered**. The two failed scored instances are one refuted
directional expectation of this study's own, not a registered clause left
undemonstrated: no TRAF-13 clause asks for a signed single-batch TTFT
difference. TRAF-29, TRAF-30 and VLLM-25 remain unused.

## Deliberate limits, recorded as prose rather than as new IDs

- The control arm is a concurrency control, not a vLLM configuration. It says
  what the observed schedule's concurrency is worth; it does not predict what
  vLLM would do with DBO disabled, which would emit one microbatch and 48
  collectives rather than 96.
- The overlap measured here is the coarse device runtime's realization of the
  source-legal concurrency. A packet-level overlap claim is not made.
- `deep_ep` was not installed, so completion semantics below the audited vLLM
  wrapper remain inferred from wrapper behavior rather than directly
  source-backed. VLLM-23 owns parallel shapes outside the frozen TP1, PP1,
  uniform-decode DBO boundary.
- TRAF-9 keeps the serial whole-layer MoE ordering approximation. This run
  measures its effect at +17.97 microseconds on the MoE phase of a decode step,
  exactly offset by the terminal term, which is useful information for whoever
  closes it but does not close it.
- TRAF-11 owns the flat NVLink rate, TRAF-14 owns immutable physical collective
  expansion, TRAF-23 owns measured per-rank completion frontiers, and VLLM-12
  owns general device-schedule templates.

## Contradiction sweep

`README.md` and `docs/architecture.md` contain no TRAF-13 state.
`docs/README_PRO.md` line 294 has one prose hit in the fidelity-level table:
the dependency row still reads "landed, TRAF-7; live producer is TRAF-13",
which is stale once TRAF-13 closes. The generated progress block and module
open counts are mechanically reconciled for ledger CI; that prose row is left
for the integrator. The void run's expectations and results retain their
original chronology and are not rewritten, rescored or unvoided.

VLLM-22 in `docs/modules/adapters-vllm.md` registers clauses that this run's
evidence also covers, but VLLM-22 is not in this change's assignment and is
left open for its owner rather than closed here.

## Verification evidence

The registered command passed with `--check-only` and created no artifact.
Full-tree Ruff passed and the Python suite passed 1,307 tests with 7 skips,
including three new regression tests for the control arm. No C++ source, native
executable or submodule pin changed.
