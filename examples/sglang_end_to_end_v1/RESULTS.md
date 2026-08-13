# SGLang end to end v1: results against pre-registered expectations

Run of 2026-08-13. Frozen by expectations-only commit **`8907c53`**, which
landed before `simllm/adapters/sglang/pump.py` existed and before the measuring
half of `run_study.py` existed. Nothing in
[expectations.md](expectations.md) was edited after the run.

**Verdict: the run is not void. All 11 fatal guards held. Scored exact
relations 5 of 5. Scored behavioral relations 4 of 4. The two classes are kept
separate and are never summed.**

## Outcome in one paragraph

A real SGLang `Scheduler` at pinned commit `8f2a3ad`, with its real
`RadixCache`, `ReqToTokenPool` and `TokenToKVPoolAllocator`, made every
batching decision in the same operating-system process that installed the step
sink. Each of its scheduler steps drove a packet-level `htsim_rnic` run whose
makespan advanced the engine's own `VirtualClock`, and the MoE bytes on that
fabric came from SGLang's own post-selection expert ids rather than from
uniform routing. Per-request TTFT and TPOT came back out with an exactly
conserved seven-component attribution, checked against an independent
standard-library recomputation. The loop is demonstrably closed: making the
fabric slower changed how many steps the framework took and which requests
shared them, from 26 steps at 400 Gbit/s to 24 at 200 and 21 at 100, with every
request still served exactly 12 tokens. No SGLang run in this repository had
previously driven `htsim_rnic` at all.

## What ran

| stage | selection |
|---|---|
| framework | SGLang at `8f2a3ad6d7d68c58ae65b61a75bb2115449addca`, verified against the source tree's git HEAD |
| driver | `SglangSchedulerPump`, one unrolled `event_loop_normal` body per step, in the process that called `install()` and `configure(step_sink=...)` |
| model | `ibm-granite/granite-3.0-1b-a400m-instruct` at `ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445`, CPU, float32, `tp_size=1`, chunked prefill disabled |
| admission | `RequestAdmissionGate` in `ARRIVAL_GATED` mode on the worker's `VirtualClock`, arrivals 1 ms apart |
| routing | SGLang v2 framework trace, provenance `sglang` / `observed-dispatch`, projected by `project_framework_routing` into `RoutedMoeSupply` |
| fabric | `HtsimStepSink` on profile `rnic-nn-fluid`, executed by `htsim_rnic` |
| compute | `RooflineProvider(0.7)` on `GPU_ENVELOPES["b100"]`, `HostInitiationModel.ideal()` |
| metrics | `HtsimRequestMetricReducer`, the same reducer the vLLM end-to-end study uses |

Scale, per cell:

| cell | scheduler steps | `htsim_rnic` runs | routed bytes | peak rank egress | wall |
|---|---|---|---|---|---|
| `ep8-400g` | 26 | 1,248 | 35,696,640 | 17,848,320 | 438 s |
| `ep8-200g` | 24 | 1,152 | 35,696,640 | 17,848,320 | 412 s |
| `ep8-100g` | 21 | 1,008 | 35,696,640 | 17,848,320 | 368 s |
| `ep4-400g` | 24 | 1,152 | 20,668,416 | 10,334,208 | 394 s |
| `control-nosink` | 16 | 0 | 0 | 0 | 4 s |

Total 4,560 `htsim_rnic` invocations, 48 per simulated step, and 54 MB of
retained artifacts outside the repository. Disclosure: the five cells were
executed concurrently as five independent child processes rather than through
`run_study`'s sequential loop. Each cell owns its own work directory, its own
scheduler and its own deterministic backend, so only the reported `wall`
column is affected by the concurrency; no simulated quantity is.

Each simulated step lowers to 72 executed artifacts: 24 per-layer compute
artifacts with zero fabric service, and 48 MoE dispatch and combine artifacts
that each cost one `htsim_rnic` process. `attribute_step` charges the first
group to `kernel_ps` and the second to `collective_ps`, which is why the two
numbers below reconcile exactly with the provider's own compute service.

## Fatal guards

All 11 held, so no behavioral fraction is invalidated. They are unscored by
construction and are not added to any total.

| guard | result |
|---|---|
| **G1** provenance | trace header `framework=sglang`, `routing_source=observed-dispatch`, model revision `ffec3c35...`, observed source `8f2a3ad6...` equal to the source tree HEAD |
| **G2** same-process closed loop | in all four simulated cells `latest_worker().step_sink is sink`, worker class `SimTpModelWorker`, tree cache class `RadixCache`, and scheduler steps == step records == sink locality outcomes == sink network outcomes (26, 24, 21, 24) |
| **G3** no mid-prompt extend, no retraction | 0 mid-prompt extend rows and 0 retractions across all 5 cells |
| **G4** conservation | 192 reduced intervals, 0 makespan failures, 0 interval failures, 0 inactive-component violations, 0 artifact-partition failures, every TTFT strictly positive |
| **G5** captured routing | all 95 simulated steps report mode `captured`, epoch 0, quiescent, 0 NVLink bytes |
| **G6** declared output length | all 20 request-cell rows finished with reason `length` after exactly 12 tokens, and the reducer counted exactly 12 |
| **G7** expert-parallel byte monotonicity | 20,668,416 bytes at width 4 against 35,696,640 at width 8, ratio 0.5790 |
| **S1** first prefill routed bytes | 3,756,032 in `[786,432, 5,505,024]` |
| **S2** serialization floor | first prefill fabric service 171.12 us, above the 15.73 us serialization floor |
| **S3** compute service | 98.93 to 99.26 us, inside `[60, 200]` us |
| **S4** prefill makespan | 268.9 to 271.8 us, inside `[80, 2000]` us |
| **S5** decode makespan | 203.96 to 224.11 us, inside `[100, 600]` us |

## Scored exact relations: 5 of 5

> **E1** ... the study recomputes, using the Python standard library and the
> declared arrivals only, the request's first-token time, last-token time,
> token count, TTFT, TPOT as an exact `Fraction`, and its TTFT and decode
> attributions ... Every one of those values equals the value
> `HtsimRequestMetricReducer` published, exactly.

**Passes.** 16 request-cell comparisons (4 simulated cells x 4 requests), 0
failures. Each comparison covers 6 scalar values and 6 attribution components
over the request's 12 token intervals, so the 16 rows summarize 192 intervals.

> **E2** For every simulated step, every layer, every phase, every scheduled
> request and every directed rank pair, the byte count recomputed straight from
> the raw v2 trace with the standard library equals
> `MoeAllToAll.request_pair_payload_bytes` exactly.

**Passes.** 40,596 per-request directed-pair rows, 0 mismatched operations. The
per-layer and per-request byte tables are in `summary.json`. In `ep8-400g` the
24 layer totals run from 1,343,488 to 1,630,208 bytes, a 21 percent spread that
is expert popularity per layer and nothing else, and the four requests carry
4,318, 4,342, 4,360 and 4,410 hidden vectors, summing to the cell's 35,696,640
bytes with no remainder.

> **E3** ... the sum over requests of the E2 table equals the sends actually
> present in the GOAL artifact `htsim_rnic` consumed, pair by pair, read from
> the files under the cell work directory.

**Passes.** 24,136 directed-pair rows read back out of the executed `.goal`
files, 0 mismatches.

> **E4** Every send in every executed dispatch artifact has `source == 0`, and
> every send in every executed combine artifact has `destination == 0`.

**Passes.** 0 ownership violations. This is the direct guard against the
historical 8x defect in which every rank replicated the scheduled tokens while
every per-request table stayed self-consistent.

> **E5** For every request, phase, token index and each of the 24 layers, the
> ordered top-8 expert ids in the `RoutedExperts` projection equal the ordered
> top-8 expert ids in the raw trace.

**Passes.** 12,960 `(request, phase, token, layer)` cells compared, 12,960
equal.

## Scored behavioral relations: 4 of 4

> **B1** ... `S(100G, i) - S(200G, i) == 2 * (S(200G, i) - S(400G, i))` to
> within `max(1000 ps, 1e-6 * S(100G, i))`.

**Passes, exactly.** 384 artifact comparisons over the 8 scheduling
compositions present in all three bandwidth cells. Worst relative residual
**0.0**: not one artifact deviated by a picosecond.

> **B2** For every step whose scheduling composition appears in both cells of a
> halving pair, the slower cell's step makespan is strictly greater than the
> faster cell's and strictly less than twice it.

**Passes.**

| halving | matched compositions | makespan ratio |
|---|---|---|
| 400G to 200G | 15 | 1.0442 to 1.2827 |
| 200G to 100G | 8 | 1.0846 to 1.4408 |

No violations. Registering this on step makespans only, and explicitly not on
TTFT or TPOT, was the right call: the vLLM end-to-end study's equivalent clause
failed on a per-request TPOT whose intervals changed composition across cells,
and the same reallocation is visible here (p0's TPOT rises 1.649x from 400G to
100G while p3's rises only 1.450x, because the two requests' intervals absorb
different amounts of other requests' prefill work).

> **B3** ... `steps(ep8-400g) >= steps(ep8-200g) >= steps(ep8-100g)` with at
> least one strict inequality, while every request still receives exactly 12
> output tokens in all three cells.

**Passes, and this is the relation the study exists for.**

```
steps(400G) = 26  >  steps(200G) = 24  >  steps(100G) = 21
```

Both inequalities are strict, all three cells produced pairwise distinct step
sequences, and all 12 request-cell rows in the ladder served exactly 12 tokens.
The mechanism is the one registered before the run: the sink's makespan is what
advances the engine's `VirtualClock`, a slower fabric therefore covers more
virtual time per step, the arrival gate releases later requests after fewer
steps, and the framework merges them into larger batches. The sink-free control
cell makes the point from the other side: it never reads a link rate at all,
its steps are settled by the worker's own roofline estimate of 953 us, and it
completes the identical workload in 16 steps.

This is the observation that separates a live closed loop from JSONL replay. A
replayed record set has one fixed step sequence by construction; here the
sequence is an output of the fabric.

> **B4** ... the ratio `compute_service(ep4) / compute_service(ep8)` lies in
> `[1.45, 1.65]`. First-principles prediction: **1.544**.

**Passes.** Measured **1.54448** (153.018 us against 99.074 us mean per-step
compute service). The prediction derived from resident weight bytes before the
run was 1.54390. The measurement is 0.04 percent from it.

## Physical sanity, checked against first principles rather than against the model

Every headline number was bounded before it was read. Three independent
framings, per the physical-sanity rule.

**Compute and memory physics.** Counted by hand before the run, per-rank
resident weights at width 8 are 555,227,136 bytes including the 24 router
matrices and 553,654,272 without them. At the B100 envelope's `8.0e12` bytes/s
that is a weight-read floor of 69.40 us, which no step can beat, and 99.15 us
after the provider's 0.7 derate. Measured: 98.928 to 99.264 us. Inverting the
minimum through the derate gives 553,996,800 bytes of implied weight traffic,
which sits between the two hand counts, 0.22 percent under the with-router
count and 0.06 percent over the without-router count. At width 4 the same
inversion gives 855,993,600 against hand counts of 857,217,024 and 855,644,160.
The compute model's resident weight bytes therefore agree with first principles
to better than 0.1 percent at both widths, and the whole residual is whether
router and norm weights are counted. The 0.34 percent spread inside a cell
tracks the number of co-scheduled requests and their KV context, not the token
count: the four 8-token prefill steps sit at the 98.928 us minimum, and the
three-request decode steps sit at the 99.264 us maximum. The path is weight-read
bound at this batch size, exactly as the freeze asserted, which is why B4's
ratio reproduces the weight-byte ratio rather than an activation ratio.

**Network and serialization physics.** Bytes over link rate is a floor no flow
can beat. The first prefill step moves 3,756,032 directed bytes, which is
4.776 distinct remote destinations per token, layer and phase, inside the
`[1, 7]` range the owner rule allows and near the 4.931 a uniform selection
would give. Rank 0 is the bottleneck endpoint of every artifact, so its share
is 3,756,032 / 48 = 78,251 bytes per artifact, i.e. 1.565 us at 20 ps per byte,
i.e. 75.12 us of serialization over the 48 artifacts. The measured fabric
service of that step is 171.121 us. The difference, 96.00 us, is exactly
48 x 2.00 us. A post-specified diagnostic confirms the constant independently:
because service is affine in the reciprocal of the rate, `2 * S(400G) - S(200G)`
cancels the byte term per artifact with no averaging, and over 720 artifacts it
returns 2,000,000 ps at minimum and 2,000,001 ps at maximum. The ideal-network
model charges exactly 2 us of propagation per collective plus the bottleneck
rank's bytes at the declared rate, with the implied bottleneck load running
from 6,144 to 98,304 bytes per artifact.

**End-to-end plausibility against the real system.** The implied per-request
decode rate `1 / TPOT` runs from 2,314 to 4,652 tokens per second. A real
deployment of a 400M-active-parameter MoE on one accelerator decodes a single
request at roughly `10^2` tokens per second, because the fixed host cost per
decode step dominates at this model size. The simulated rate is therefore
optimistic by roughly one order of magnitude, which is what the frozen error
budget predicted and which is attributable almost entirely to
`HostInitiationModel.ideal()` charging zero per-step host cost. A simulated
rate near the realistic value would have been evidence of a compensating error,
not of accuracy.

## Per-request metrics

`ep8-400g`, times in microseconds, attributions in the seven-component form
(only `queue`, `kernel` and `collective` are populated; the other four are
configuration-forced zeros and are guarded by G4):

| request | arrival | TTFT | TPOT | TTFT queue / kernel / collective | decode queue / kernel / collective |
|---|---|---|---|---|---|
| p0 | 0 ms | 270.05 | 262.04 | 0.0 / 98.9 / 171.1 | 542.6 / 1089.7 / 1250.2 |
| p1 | 1 ms | 358.38 | 268.70 | 86.6 / 98.9 / 172.8 | 539.7 / 1090.8 / 1325.2 |
| p2 | 2 ms | 483.60 | 244.41 | 212.8 / 98.9 / 171.9 | 268.9 / 1091.1 / 1328.5 |
| p3 | 3 ms | 421.38 | 214.95 | 152.5 / 98.9 / 170.0 | 0.0 / 1090.5 / 1274.0 |

Every row conserves: the three TTFT components sum to the TTFT, and the three
decode components sum to `TPOT x 11`. The queue column is the interesting one.
p0 waits for nothing because it is alone when it arrives; p2 carries 212.8 us
of queue before its first token because two other requests are already
prefilling; p3 carries none in decode because it is the last request admitted
and never waits behind another's prefill after that.

At 100 Gbit/s the same requests report TTFT 495.41 to 727.89 us and TPOT 311.68
to 432.12 us, with the collective share of each interval growing and the kernel
share unchanged, which is the signature of a network-side change.

## Evidence classes, kept separate

Counts from different classes are not summed.

- **Fatal-unscored guards:** 11 declared, 11 held, run not void.
- **Scored exact relations:** **5 of 5**. E1, E2, E3, E4, E5.
- **Scored behavioral relations:** **4 of 4**. B1, B2, B3, B4.
- **Reported, not scored:** the `ep4` / `ep8` byte ratio, the control cell, the
  scale table, the fluid-fit diagnostic and the error budget.

### Genuine-risk analysis

The entailment answers frozen before the run hold up, and the freeze already
removed three candidate relations that could not fail. Restating the honest
fraction: **8 of the 9 scored relations are genuine risk.** E5 is the
exception and was declared narrow risk in the freeze, not after it: a
projection that dropped tokens aborts in `RoutedMoeSupply` rather than failing
E5, and a projection that changed an expert id to one with a different owner
fails E2, so what E5 alone catches is a permutation inside a layer's top-8
tuple that preserves the owner set. It is reported as a passing relation and
its narrowness is stated rather than hidden.

The three relations removed from the scored set before the run, and why:

1. *Per-request TTFT and TPOT conserve exactly against the seven-component
   attribution.* Entailed: `HtsimRequestMetricReducer.consume` and
   `RequestLatencyTotals.__post_init__` **raise** when it does not hold, so it
   can abort the run but cannot be observed as a failing relation. Kept as
   fatal guard G4, where it held over 192 intervals.
2. *Every TTFT is strictly positive and is measured from the declared arrival.*
   Entailed: the reducer raises when a request is scheduled before its declared
   arrival, the study hands the same arrival mapping to the gate and to the
   reducer, and every simulated step has a strictly positive makespan. Kept as
   part of G4.
3. *Narrowing the expert-parallel world strictly decreases routed bytes.* The
   weak inequality is forced by `owner_4(e) = owner_8(e) % 4`. Kept as G7 and
   reported: 0.5790.

E1 is genuine risk because the independent side is a standard-library
recomputation from the per-step rows and the declared arrivals, so a reducer
that carried a pending attribution to the wrong request or started the first
interval at the wrong endpoint would conserve internally and still disagree.
E3 is not entailed by E2 (in-memory table against executed artifact bytes), E4
is not entailed by either (a replication defect keeps every per-request table
self-consistent), and B3 is falsifiable in both directions: an unfed sink or a
gate on a different clock would have made all three cells take the same number
of steps.

## What this study supports, and what it does not

**Supported:** that a real SGLang `Scheduler`, `RadixCache` and token pools
made every batching decision while only the forward pass was simulated; that
each scheduler step drove a packet-level `htsim_rnic` run whose makespan
advanced the engine's clock; per-request TTFT and TPOT with exact conservation
and an independent recomputation; MoE bytes derived from SGLang's own
post-selection expert ids rather than uniform routing; and relative sweeps in
which exactly one parameter moves.

**Not supported, explicitly:**

- **No oracle token replay.** There is no replay source on this adapter
  (PLAY-7). The worker emits one constant fabricated token id, so the served
  token sequence is not any oracle's and the prefix-cache behavior is that of a
  degenerate token stream. No clause reads served token identities.
- **No validation against silicon**, and no absolute TTFT or TPOT claim. SGL-4
  owns the silicon comparison.
- **SGLang's own communicator collectives are not metric-live.** The mirror is
  zero time (SGL-13, SGL-15) and was left disabled. Every microsecond of
  `collective_ps` above comes from the shared lowerer's MoE all-to-alls.
- **No observed dependency schedule.** There is no SGLang
  `ExecutionObservations` producer (SGL-10, SGL-17), so the lowering is serial
  and the 72 artifacts of a step execute one after another.
- **Wall-clock client timings and simulated metrics stay separate types.** The
  `wall` column above is process time on this machine and is never reported as
  a latency.

The error budget frozen in `expectations.md` is unchanged by the run. Its
dominant term, the zero fixed per-step host cost, is quantified above by the
decode-rate comparison.

## Closure

> **SGL-8:** a live closed-loop run with `HtsimStepSink` installed via
> `configure(step_sink=...)` on the CPU-engine smoke path, mirroring the vLLM
> tp=8 run of examples/m4 (the M4 slice covered this adapter by JSONL replay
> only).

Mapped clause by clause:

- *"a live closed-loop run"*: 95 simulated scheduler steps across four cells,
  each settled by a `StepResult` the sink produced from an executed
  `htsim_rnic` run, each advancing the worker's `VirtualClock`. B3 shows the
  feedback changed the framework's own decisions, which is what distinguishes
  this from the M4 replay.
- *"with `HtsimStepSink` installed via `configure(step_sink=...)`"*: the driver
  calls `configure(step_sink=sink)` and G2 asserts
  `latest_worker().step_sink is sink` for the exact object, in the same process,
  with step records, sink locality outcomes and sink network outcomes all equal
  in count.
- *"on the CPU-engine smoke path"*: `device="cpu"`, the real `Scheduler`, the
  real `RadixCache`, the fabricated `ReqToTokenPool` and
  `TokenToKVPoolAllocator`, and `SimTpModelWorker`. One deliberate difference
  is disclosed: the scheduler is constructed in this process by
  `build_in_process_scheduler` rather than by `sglang.Engine`'s `mp.Process`
  launcher. That is not a shortcut, it is the only way the clause can be
  satisfied at all, because `configure` is process local and cannot reach a
  child process. The stack below the launcher is identical, and SGLang carries
  its own in-tree precedent for constructing the scheduler this way in
  `srt/ray/scheduler_actor.py`.
- *"mirroring the vLLM tp=8 run of examples/m4"*: the same `HtsimStepSink` on
  the same `rnic-nn-fluid` profile over an eight-rank fabric group. A second
  disclosed difference: that group is expert-parallel, not tensor-parallel.
  `tp_ranks = (0,)` here, following the vLLM end-to-end study rather than M4,
  because this study's traffic authority is SGLang's own captured MoE routing
  and SGLang itself ran at `tp_size = 1`, so a declared tensor-parallel shard
  would describe a geometry the framework did not execute. A reviewer who reads
  "tp=8" as a required topology rather than as a pointer to which M4 run is
  being mirrored should leave SGL-8 open; nothing else in this report depends
  on that reading.
- *"(the M4 slice covered this adapter by JSONL replay only)"*: closed. The
  adapter has now driven `htsim_rnic` live.

**Registered IDs: none.** Every registered acceptance clause of SGL-8 is
demonstrated above. Nothing this run failed to demonstrate belongs to SGL-8,
and the adjacent gaps it exposes are already owned: `num_sampled` at the worker
seam by SGL-12 (this study guards it with G3 rather than fixing it), the
runtime `CompletionEvent` projection by SGL-13, the observed schedule by SGL-10
and SGL-17, token replay by PLAY-7, and the cross-cell TPOT bound by PLAY-15.
No new ID was created for any of them, and none is required.

SGL-16 is **not** closed here. The routing authority this run consumed is the
SGL-16 framework-layer-id trace, so the live-reachability precondition that
task recorded is met, but this study registered no SGL-16 acceptance clause and
a demo cannot close a precision task by association.

## Library change

`simllm/adapters/sglang/pump.py`, the in-process scheduler pump.
`build_in_process_scheduler` constructs the pinned `Scheduler` in the calling
process, `SglangSchedulerPump.step()` unrolls one `event_loop_normal` body, and
`SchedulerOutputCollector` replaces the detokenizer socket so completions,
finish reasons, radix hits and retraction counts are observable in process. The
module imports without SGLang and without torch, the pump only calls duck-typed
scheduler methods, and its ordering contract, its refusal of a duplicate or
blank request identity, its double-completion guard and its exact off path
(`attach_output_collector=False` mutates nothing) are tested against a stub
scheduler in `tests/test_adapters_sglang_pump.py`. No existing behavior
changed: `simllm/adapters/sglang/worker.py` is untouched by this work.

## Reproduction

```
python examples/sglang_end_to_end_v1/run_study.py \
    --cache-dir <hugging-face-cache-root> \
    --sglang-python <interpreter-that-owns-sglang-and-torch> \
    --sglang-source <sglang-source-checkout-at-the-pinned-commit> \
    --routing-trace <strict-v2-sglang-framework-trace> \
    --htsim-rnic <path-to-htsim_rnic> \
    --run-dir <writable-output-directory>
```

`--check-only` validates every frozen input and writes nothing. The parent
interpreter never imports SGLang or torch. Backends are deterministic, so every
number above reproduces exactly.
