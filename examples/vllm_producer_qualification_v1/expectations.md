# vLLM producer qualification v1 expectations

This document freezes the VLLM-22 qualification before its harness, its
producer-disabled live arm, its CI byte lock, or any result-producing run
exists. The repository source at this boundary is commit
`6b7200b300f10f7b46a16921487c9b58ca32c987`.

The landed
[`vllm_observed_overlap_v1`](../vllm_observed_overlap_v1/RESULTS.md) study is
the foundation for this qualification. Its measurements are prior information,
not results of this study. Its three-arm construction is reused rather than
copied: `serial` withholds observations, `control` adds only the frozen
cross-microbatch serialization edges, and `observed` consumes the live
producer tuple unchanged.

## Registered closure scope

VLLM-22 currently requires:

> qualify the real vLLM `ExecutionObservations` producer for the frozen
> Granite boundary.

> preserve exact submission order, logical streams, dependencies, request
> correlation and completion frontiers, and reach TTFT and TPOT through the
> supported metric chain.

> name the wrapper or measured mechanism that makes each concurrency legal and
> derive no edge from an overlap percentage or compatibility schedule.

> Completion reduction must return the original request identities;
> per-request routed-byte acceptance depends on TRAF-25 and VLLM-24, both of
> which have landed.

> With the producer absent, preserve the legacy sink call, serial graph and
> GOAL bytes, timestamps and completion order exactly.

Every quoted clause must be mapped to evidence in the result report. If the
run is valid but a clause is not demonstrated, the clause moves to one of
VLLM-26, VLLM-27 or VLLM-28 with a category, priority and difficulty. No ID is
registered speculatively. If any fatal guard fails, the run is void and
VLLM-22 remains open as one task rather than being partially closed.

## Frozen inputs and provenance

The input is the unchanged three-request Granite greedy replay used by the
landed overlap study:

- capture schema `simllm-preplay-trace-v1`, 120 rows, SHA-256
  `5f55fccc265ee5519430a0f73d6631e49aa547ec07fcb81034e9fc2b4d9fead6`;
- replay-run SHA-256
  `b4d38a09011caf6de159c22133264d62a2727063496953f4337b17d79cfde93e`;
- routed-experts SHA-256
  `24e986e989e21f1bfe7e758d4470928c82c3bbaec06072a839743b9b17d7cf5f`;
- model `ibm-granite/granite-3.0-1b-a400m-instruct`, revision
  `ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445`;
- 24 MoE layers, hidden width 1,024, 32 experts, top-k 8, BF16 activations
  and four resident experts per EP rank;
- original request identities `r0`, `r1` and `r2`, with prompt lengths 22,
  12 and 20.

The live cell fixes TP 1, DP and EP 8, PP 1,
`enable_expert_parallel=True`,
`all2all_backend=deepep_high_throughput`, `enable_dbo=True`, decode threshold
2 and prefill threshold 512. There are 32 nonempty steps. Step 0 is the
54-token single-batch prefill, steps 1 through 23 carry `r0`'s TPOT intervals
and use two microbatches, and steps 24 through 31 are single-request decode
steps for which DBO is off.

The evidence is authored against official vLLM v0.26.0 commit
`568afb3a13806beb53bb2e6bd518269357b237c0`. A run records version 0.26.0 and
the nine source-file identities already frozen by the landed study. The
authored-against identity and observed source identities remain separate. No
check requires a live checkout, installed package or submodule pin to equal
the authored-against commit.

## Source audit and concurrency authority

The line-level source audit was performed before this freeze against the
same vLLM v0.26.0 files whose hashes the run verifies. The relevant legal
concurrency mechanisms are named here:

- `vllm/v1/worker/gpu_ubatch_wrapper.py:305-341` launches the two model
  forwards in cooperative threads, starts microbatch zero first, joins both
  and restores result order. Lines 441-490 bind the common compute stream and
  per-microbatch metadata or take the single-batch bypass.
- `vllm/v1/worker/ubatching.py:77-147` records and waits on compute and
  communication stream events. Those event waits plus FIFO order on the
  shared compute and communication streams are the concurrency authority.
- `vllm/model_executor/layers/fused_moe/modular_kernel.py:1162-1229` performs
  asynchronous dispatch preparation, registers the receive hook, yields and
  waits before expert compute. Lines 1312-1378 provide the corresponding
  combine path.
- `vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_ht.py:106-192`
  records the preceding compute event, yields for dispatch, launches on the
  communication stream and waits before returning to compute. Lines 347-410
  provide the combine-side event and stream switch.
- `vllm/v1/worker/gpu_model_runner.py:4394-4429` computes logits only after
  the model forward, so request visibility follows the final combines and
  logits work.

The audited wrapper is the evidence for shared-stream FIFO and event waits.
`deep_ep` itself is not installed, so rank-local behavior beneath that wrapper
remains an explicit inference and is not promoted to a direct DeepEP semantic
claim.

The SimLLM producer may encode only those source mechanisms. Its source must
contain no overlap fraction, percentage, duration discount, fitted
concurrency constant or random choice. It must not import or inspect
`SerialStepLowerer`, a compatibility `ExecutionGraph`, or a compatibility
schedule to create an edge. Unsupported shapes retain their existing explicit
refusals.

## Arms and live-chain boundary

Four views are used, but only three are timing arms over the enabled producer:

1. `serial`: observations from the enabled live replay are withheld, selecting
   `SerialStepLowerer` through the supported sink off path.
2. `control`: the observed tuple receives only the landed 760
   cross-microbatch serialization edges on each DBO step. Operation identity,
   tuple position, rank, queue, work, correlation, gate, priority and
   completion endpoints remain identical to `observed`.
3. `observed`: the enabled producer tuple is consumed unchanged.
4. `disabled-live`: a second real vLLM replay selects
   `SIMLLM_VLLM_OBSERVED_SCHEDULE=off` and supplies a one-argument legacy
   `StepRecord` sink. That sink delegates to the accepted serial mechanism and
   records the exact call and artifacts. It is compared against `serial`; it
   is not a fourth scored timing policy.

The supported chain under qualification is:

```text
real vLLM scheduler and SimWorker model-forward boundary
  -> ExecutionObservations
  -> ObservedStepLowerer and routed traffic binding
  -> CoarseDeviceRuntime
  -> CompletionEvent and CompletionReducer
  -> StepResult
  -> original-request TTFT and TPOT
```

The `control` arm distinguishes DBO concurrency from the TRAF-9 whole-layer
ordering and terminal-frontier differences. No serial-versus-observed
single-batch equality is assumed as an attribution premise. On a single-batch
step `control` adds no edge, so `control` and `observed` must be exactly equal.
The structure term `serial - control` is measured separately.

## Frozen flagship relations

All raw per-step latencies and request metrics are collected before any exact
identity, source inventory, physical bound or fatal guard is interpreted.
`r0` TPOT is the exact mean of its 23 decode intervals.

For placement `p`, define the producer-enabled per-request reduction as:

```text
reduction_p = (TPOT_serial,p - TPOT_observed,p) / TPOT_serial,p
```

The landed structure-matched measurement found 1.437 percent overlap on the
single-node placement and 11.593 percent on the cross-node placement. The
new bands deliberately surround those established points while leaving room
for the complete live replay and new disabled arm to expose a regression.

### B1. Enabled producer changes per-request TPOT in band

Two genuine-risk instances, one per placement:

| Placement | Required direction | Inclusive relative-reduction band |
|---|---|---:|
| `single-node` | `TPOT_observed < TPOT_serial` | 1.25 through 1.65 percent |
| `cross-node` | `TPOT_observed < TPOT_serial` | 10.5 through 12.5 percent |

The relation is against the serial arm as required. The control decomposition
is a fatal attribution premise below, not a second score over nearly the same
quantity.

### B2. The enabled effect follows the bandwidth change

One genuine-risk instance. The ratio of the absolute TPOT reductions,
`(serial - observed)_cross / (serial - observed)_single`, must lie in the
inclusive band 7.5 through 10.5. The expected center is 9.0, the exact ratio
between 3.6 Tbit/s and 400 Gbit/s. A compute artifact or an ignored observation
tuple can fail this relation even if one B1 cell happens to land in band.

The frozen behavioral denominator is three genuine-risk instances. No exact
disabled-path identity contributes to it.

## Fatal unscored guards

A violation of any guard in this section voids the run for closure. Fatal
means void, never a lost point or a fractional score. No guard is designated
survivable.

### Arm equivalence and attribution

1. `control_differs_only_by_edges`: control and observed preserve the same
   ordered operation tuple and completion endpoints; only whole-operation
   dependency supersets are allowed.
2. `control_edge_counts`: exactly 760 edges are added on each of 23 DBO
   steps and zero on all nine single-batch steps.
3. `control_has_no_cross_microbatch_concurrency`: no realized busy interval
   from one microbatch intersects one from the other in control.
4. `control_observed_single_batch_identity`: on step 0 and steps 24 through
   31, control and observed graph, execution, timestamp, completion order,
   `StepResult` and request metric bytes are identical.
5. `decomposition_identity`: `serial - observed` equals
   `(serial - control) + (control - observed)` exactly on every step.
6. `structure_is_bounded`: over the 23 TPOT intervals, the absolute
   `serial - control` term is at most 0.01 percent of serial TPOT on each
   placement. This prevents the old arm-equivalence mistake from hiding
   inside the flagship relation. It is a prerequisite for attribution and
   therefore fatal unscored.

### Producer fidelity and preservation

7. `producer_every_nonempty_step`: all 32 nonempty translated steps emit
   observations.
8. `producer_exact_submission_order`: every source operation ID follows the
   audited layer, microbatch, phase, rank and terminal ordering grammar,
   including the source-ordered request partition.
9. `producer_layer_and_site_inventory`: every step covers ordered layers 0
   through 23 and 48 unique dispatch/combine sites, with 96 collective
   invocations on DBO steps and 48 on single-batch steps.
10. `producer_request_partitions`: microbatch request slices are disjoint,
    preserve source order and partition the scheduled requests.
11. `producer_completion_frontiers`: there is exactly one request-correlated
    completion endpoint per active microbatch, after both final combines and
    every rank's logits operation.
12. `producer_no_synthetic_concurrency`: source inspection finds no forbidden
    overlap knob, random choice, compatibility lowerer import or compatibility
    graph dependency.
13. `schedule_fields_survive_lowering`: tuple position, rank, logical stream,
    both dependency scopes, gate, priority, correlation and completion
    identity survive traffic binding in observed and control.
14. `request_pair_rebinding_exact`: the post-TRAF-25 per-request routed tables
    are identical across arms and placements and the VLLM-24 independent
    conservation check remains live.
15. `original_request_identities`: completion reduction returns only the
    original `r0`, `r1` and `r2` identities, returns each while it is active,
    and introduces no replacement or synthetic request identity.
16. `live_cross_node_identity`: the execution inside the enabled live driver
    matches the independent harness graph, execution and `StepResult` on all
    32 steps.

### Exact producer-disabled path

17. `disabled_live_step_records`: the real producer-off replay makes exactly
    one one-argument legacy sink call per translated step and its canonical
    `StepRecord` bytes equal the serial reference.
18. `disabled_live_artifacts`: on every step, serial graph JSON, graph-derived
    GOAL, legacy diagnostic GOAL, execution result, completion timestamps,
    completion order, `StepResult` and request metrics are byte-identical to
    the serial reference.
19. `serial_fixture`: the accepted fixed fixture still renders 4,127 bytes of
    canonical graph JSON with SHA-256
    `aa3c836fe559973a7bf0940384c2e8a84e6af84e0fbd2c02d3b89774ee0c8e2d`
    and 1,880 bytes of legacy diagnostic GOAL with SHA-256
    `7087db6780f7e34f5a559a6505eeccc15d984c7b478cd8f0bc5838053825d4b6`.
20. `pytest_disabled_byte_lock`: a tracked pytest that needs neither vLLM nor
    a `third_party` checkout compares the one-argument legacy call and all
    accepted serial artifact classes with the repository's standard exact
    artifact comparator. It includes one-byte negative controls so the lock
    is capable of failing.

Guards 1 through 5 and 7 through 20 are construction, conservation, exact
identity or change-set facts and are unscored. Guard 6 is a measured premise
needed to interpret B1 as a producer-concurrency result and remains unscored
for that reason. None pins either B1 reduction to its band or pins the B2
ratio. Behavioral relations are evaluated first against raw observations.

## Physical sanity frozen before measurement

Every reported modeled time is interpreted only after these bounds, written
before the run, are applied.

Compute and memory floor. The 553,648,128 resident weight and LM-head bytes at
8 TB/s require at least 69,206,016 ps per decode step. No TPOT arm may fall
below that floor.

Compute and network ceiling. The 0.7 roofline is about 99.5 microseconds. Even
a fully serialized cross-node decode communication term is below 50
microseconds for the frozen post-TRAF-25 step table, so 150,000,000 ps is the
inclusive TPOT ceiling for every arm.

Network serialization bounds. The 15,071,232 bytes of summed peak-source
egress over the 23 TPOT steps give mean communication ceilings of 1,456,158 ps
at 3.6 Tbit/s and 13,105,420 ps at 400 Gbit/s. An overlap reduction cannot be
negative and cannot exceed 1.05 times the corresponding ceiling without a
model, harness or interpretation defect. The expected effect scales by nine
with link rate; B2 checks that paired quantity.

Prefill bound. The same compute floor applies to TTFT. The 15,249,408-byte
cross-node prefill communication term is 304,988,160 ps, so 500,000,000 ps is
the inclusive TTFT ceiling for every arm.

End-to-end plausibility. A roughly 100 microsecond modeled TPOT implies about
10,000 tokens per second for this 400M-active-parameter model. That is an ideal
analytic result, not a real-deployment prediction. The accepted mission-study
budget puts real decode at 1.1 to 4.5 ms against 0.205 ms simulated, roughly
5x to 22x slower.

This qualification changes no modeled duration, host cost, collective floor,
GPU envelope or compute calibration. The fixed per-step term remains 0 ps,
the mission packet-level collective floor remains 2.000 microseconds per
collective, and compute remains the B100 roofline at a flat 0.7 derate. The
composed 5x to 22x optimism budget is therefore frozen to remain unchanged
before and after this work.

## Evidence classes and entailment

Evidence classes remain separate:

- two physical configurations;
- six enabled-producer raw timing cells;
- one disabled live replay;
- three genuine-risk behavioral instances;
- one fatal-unscored guard set;
- source and producer inventories;
- exact disabled-path artifact comparisons;
- focused pytest and full repository test invocations.

Counts from different classes are never added. By-construction checks cannot
increase the behavioral denominator. The scored relations are read directly
from raw request metrics before any fatal guard. A producer that is ignored
can produce zero reduction and fail B1; a schedule that creates a
placement-invariant compute artifact can fail B2. The later exact identities
do not entail either outcome.

## Freeze integrity and post-run changes

The result report must list every commit after the first measured run. For
each commit it must state whether the change fixes a harness or implementation
defect or changes modeled behavior. Any modeled-behavior change records the
measurement before and after. If modeled behavior is changed after a failed
number is observed, the evidence is either labeled post-specified or refrozen
and rerun as a disclosed second attempt.

## Reproduction and pre-freeze dry run

Configure the machine-local roots, then run from the repository root. The
selected output directory must not already exist.

```bash
.venv/bin/python examples/vllm_producer_qualification_v1/run_study.py \
  --capture "${SIMLLM_MOE_E2E_ROOT:?configure SIMLLM_MOE_E2E_ROOT}/capture/granite-greedy.jsonl" \
  --replay-run "${SIMLLM_MOE_E2E_ROOT:?configure SIMLLM_MOE_E2E_ROOT}/replay-400g-nvlink/run.json" \
  --routed-experts "${SIMLLM_MOE_E2E_ROOT:?configure SIMLLM_MOE_E2E_ROOT}/replay-400g-nvlink/routed-experts.json" \
  --vllm-source "${SIMLLM_VLLM_SOURCE:?configure SIMLLM_VLLM_SOURCE}" \
  --vllm-python "${SIMLLM_VLLM_PYTHON:?configure SIMLLM_VLLM_PYTHON}" \
  --output-dir "${SIMLLM_VLLM22_RUN_ROOT:?configure SIMLLM_VLLM22_RUN_ROOT}/qualification"
```

Before this expectations commit, that exact command is run with
`--check-only`. Check-only validates the frozen input identities, vLLM source
hashes and version, bands, physical arithmetic, evidence count and output-path
precondition. It imports no SimLLM target module, creates no output directory
and writes no artifact.
