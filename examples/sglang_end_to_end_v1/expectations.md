# SGLang end to end v1 expectations

This document freezes the acceptance contract for the first live closed-loop
SGLang run before the mechanism it needs exists and before any result-producing
run. It is the pre-run record required by the validation discipline: every
relation, bound, direction and band below is written down here first, the
harness is exercised with `--check-only` before this commit lands, and the
result report cites this freeze commit.

## What this study claims, and what it refuses to claim

The vLLM adapter already drives the simulated executor end to end and publishes
per-request TTFT and TPOT ([examples/end_to_end_replay_v1](../end_to_end_replay_v1/RESULTS.md)).
The SGLang adapter does not. Everything downstream of the worker is shared and
framework neutral, so the gap is not in the lowerer, the traffic renderer, the
sink or the reducer. The gap is the process boundary: SGLang runs its
`Scheduler` inside an `mp.Process` and loads plugins in `run_scheduler_process`,
so `simllm.adapters.sglang.worker.configure(step_sink=...)` in a parent process
never reaches the worker. `configure` is process local and says so in its own
docstring. No live sink has ever been installed on this adapter.

This study closes that gap with an in-process scheduler pump: the driver
constructs `sglang.srt.managers.scheduler.Scheduler` in the same process that
called `install()` and `configure(...)`, then drives it one step at a time by
unrolling the body of `event_loop_normal`, exactly as the vLLM study interleaves
`llm.llm_engine.step()` with its arrival gate. SGLang ships in-tree precedent
for constructing the scheduler this way in `srt/ray/scheduler_actor.py`.

The claim is split into three separately registered parts so a weak part cannot
ride on a strong one:

1. the loop is genuinely closed and runs in one process,
2. the fabric bytes are SGLang's own post-selection expert identities, checked
   against an independent recomputation, and
3. per-request TTFT and TPOT come out with a conserved seven-component
   attribution and respond to the fabric in a registered direction.

This study does **not** claim absolute accuracy against real hardware. It also
refuses, explicitly, the following, which are named here so no reader has to
infer them from silence:

- **No oracle token replay.** There is no replay source on this adapter
  (PLAY-7 owns it). The worker emits one constant fabricated token id, so the
  served token sequence is not the oracle's and the prefix-cache behavior is
  that of a degenerate token stream. Served token identities are therefore not
  compared against anything, and no clause below reads them.
- **No validation against silicon.**
- **No metric-live SGLang communicator.** The communicator mirror is zero time
  (SGL-13, SGL-15). Any collective time reported here comes from the shared
  lowerer's MoE all-to-alls, not from SGLang's own collectives, and the
  communicator flag is left unset.
- **No observed dependency schedule.** There is no SGLang
  `ExecutionObservations` producer (SGL-10, SGL-17), so the lowering is serial.
- **No absolute latency accuracy of any kind.** See the error budget.

## Frozen configuration

### SGLang

- SGLang pinned commit `8f2a3ad6d7d68c58ae65b61a75bb2115449addca`, the commit
  the adapter is authored against. The harness reads the observed source tree's
  git HEAD and refuses to run if it differs.
- device `cpu`, dtype `float32`, `tp_size = 1`, `page_size = 1`,
  `context_length = 512`, `max_total_tokens = 4096`,
  `max_running_requests = 8`, `random_seed = 173`, `disable_cuda_graph = True`.
- `disable_overlap_schedule = True`: the pump unrolls `event_loop_normal`, and
  the overlap loop is out of scope for this adapter (SGL-6).
- `chunked_prefill_size = -1`. At the pinned commit this resolves to
  `self.chunked_prefill_size = None` in `Scheduler.init_chunked_prefill`, which
  makes `PrefillAdder.rem_chunk_tokens` `None` and every prefill whole
  sequence. This is load bearing, see the trap section below.
- the simulated worker is installed by `simllm.adapters.sglang.install()`,
  which applies the same `REPLACE` hook on `Scheduler.init_tp_model_worker`
  that the entry-point plugin applies. `SIMLLM_SGLANG_ORACLE_CAPTURE` is never
  set: capture and simulate are separate phases and the two gates are mutually
  exclusive.
- the model is `ibm-granite/granite-3.0-1b-a400m-instruct` at revision
  `ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445`, loaded offline from a Hugging
  Face cache root supplied on the command line. No path is hardcoded.

### Requests and arrivals

Four requests, the leading prefix of the `preempt` cell of the SGL-16 capture,
so their routing rows already exist and no new capture is required:

| request | prompt token ids | prompt tokens | max_new_tokens |
|---|---|---|---|
| `p0` | 1000..1007 | 8 | 12 |
| `p1` | 1100..1107 | 8 | 12 |
| `p2` | 1200..1207 | 8 | 12 |
| `p3` | 1300..1307 | 8 | 12 |

Prompt token ids are `1000 + 100 * index + step` for `step` in `0..7`, which is
the capture's own `_pressure_prompt` rule. Declared arrivals are
`index * 1_000_000_000` ps, i.e. one millisecond apart in declaration order.
The same arrival mapping is handed to the `RequestAdmissionGate` in
`ARRIVAL_GATED` mode, bound to the worker's `VirtualClock`, and to the
`HtsimRequestMetricReducer`. `join_preplay_arrivals` is not used: it is a v1
projection and the routing authority here is a v2 framework trace. Arrivals are
appended to the `RequestBookkeeper` directly as framework-request objects.

### Routing authority

The MoE traffic comes from SGLang's own captured post-selection expert ids. The
harness consumes a strict v2 framework trace supplied on the command line and
projects it with `project_framework_routing`, which copies the captured tuples
without reordering. The trace must declare `framework = "sglang"`,
`routing_source = "observed-dispatch"`, the frozen model id and revision, 32
experts, top-k 8 and MoE layer indices `0..23`. A vLLM Transformers capture
would run and would produce numbers, and those numbers would be another
framework's realization attributed to an SGLang run, so the provenance check is
a fatal guard rather than a convenience.

The reference cell contributes exactly 8 prefill dispatch rows and 19 decode
dispatch rows per request, because the last generated token is never fed
forward. A 12-token output therefore forwards 8 prefill tokens and 11 decode
tokens, both strictly inside the captured extent.

### Model geometry

`ModelDims` is the per-rank geometry of the declared deployment:

| field | value |
|---|---|
| `num_layers` | 24 |
| `hidden_size` | 1024 |
| `intermediate_size` | 512 |
| `num_heads` | 16 |
| `num_kv_heads` | 8 |
| `head_size` | 64 |
| `vocab_size` | 49155 |
| `dtype_bytes` | 2 |
| `num_experts` | 32 |
| `top_k` | 8 |
| `moe_intermediate_size` | 512 |
| `local_num_experts` | `32 // ep_world` |

The routed hidden vector is `hidden_size * dtype_bytes = 2048` bytes.

### Placement, ownership and traffic rules

These are the same rules the vLLM end-to-end study froze, reused deliberately
so the two adapters are comparable:

- expert owner rule `owner(layer, expert) = expert % ep_world`, one immutable
  `ExpertPlacementSnapshot` at `placement_epoch = 0` for the whole run;
- `RoutedMoeSupply.engine_rank = 0`. Rank 0 is the only rank that dispatches
  scheduled tokens; the other EP ranks own experts and carry no scheduled
  tokens;
- dispatch dedup: for one token at one layer the destination set is the set of
  owners of its top-8 experts; each distinct **remote** destination receives
  exactly one 2048-byte vector, and a destination equal to `engine_rank` stays
  local and contributes zero fabric bytes;
- combine is the exact transpose of dispatch after owner-side pre-reduction;
- no placement manifest is supplied, so the run is at the all-remote
  compatibility level: `nvlink_directed_bytes == 0` and every MoE byte crosses
  the simulated fabric;
- `tp_ranks = (0,)`, so there are no tensor-parallel all-reduces. The only
  fabric traffic is captured MoE dispatch and combine, and the eight-rank group
  in this study is the expert-parallel group. SGLang itself ran at `tp_size = 1`
  and a declared tensor-parallel shard would contradict the geometry the
  framework actually executed.

### Network and compute

- backend profile `rnic-nn-fluid`, the ideal-network baseline, executed by
  `htsim_rnic` through `HtsimStepSink`. Each of the 24 layers contributes one
  dispatch artifact and one combine artifact, so a step that moves tokens
  executes 48 GOAL artifacts;
- compute provider `RooflineProvider(0.7)` against `GPU_ENVELOPES["b100"]`;
  `HostInitiationModel.ideal()`, so a modeled step charges zero fixed per-step
  cost;
- `SIMLLM_SGLANG_MODE = virtual`, so the worker returns immediately and reports
  simulation-native timings. Wall-clock client timings and simulated metrics
  are separate types and are never reported as one another.

### Cells

Two parameters vary, as the validation discipline requires: the link rate and
the expert-parallel width.

| cell | ep_world | link rate |
|---|---|---|
| `ep8-400g` | 8 | 400 Gbit/s |
| `ep8-200g` | 8 | 200 Gbit/s |
| `ep8-100g` | 8 | 100 Gbit/s |
| `ep4-400g` | 4 | 400 Gbit/s |

A fifth configuration, `control-nosink`, runs the identical pump and the
identical requests with **no** step sink installed, so the worker settles every
step on its own roofline estimate and no backend runs. It costs no `htsim_rnic`
invocation. It exists to make the closed-loop relation legible: with no sink,
link rate is not read at all, so the step sequence cannot depend on it.

## Registered acceptance clauses

Clauses marked *fatal-unscored* are conservation identities,
configuration-forced values or by-construction consequences. Violating one
voids the run rather than costing a point, and they never enter a scored
denominator. A violated fatal guard is reported as void with findings, never as
a fraction.

### Fatal guards

- **G1 provenance.** The routing trace header declares `framework = "sglang"`,
  `routing_source = "observed-dispatch"`, `schema = "simllm-preplay-trace-v2"`,
  the frozen model id and revision, `expert_count = 32`, `top_k = 8` and
  `moe_layer_indices = 0..23`, and the observed SGLang source tree's git HEAD
  equals the pinned commit.
- **G2 same-process closed loop.** In every simulated cell,
  `latest_worker().step_sink is sink` for the exact sink object the driver
  constructed, the driver's process id equals the process id recorded by the
  worker at construction, and the number of pump steps that ran a batch equals
  `len(worker.step_records)` equals `len(worker.step_results)` equals
  `len(sink.locality_outcomes)` equals `len(sink.outcomes)`.
- **G3 no mid-prompt extend and no retraction.** Every `PREFILL` row of every
  `StepRecord` has `context_length` equal to the request's declared prompt
  length, so no scheduled row is a mid-prompt chunk, and every completion
  reports `retraction_count == 0`. This is the guard on the absent
  `num_sampled` field: when `num_sampled` is `None`, `sampled_request_ids`
  treats every scheduled row as sampling, which is exactly true under this
  guard and silently false without it.
- **G4 conservation identities.** For every simulated step,
  `completed_at_ps == virtual_time_ps + step_latency_ps` and
  `sum(composed_phase_service_ps) == step_latency_ps`; for every reduced
  interval, the seven components sum to the interval's elapsed time; and
  `kv_ps`, `dma_ps`, `nic_ps` and `control_ps` are exactly zero, because this
  configuration models none of them.
- **G5 captured routing.** Every simulated step reports routing mode
  `captured`, placement epoch 0, backend quiescence, and zero NVLink directed
  bytes.
- **G6 declared output length.** Every request in every cell finishes with
  reason `length` after exactly 12 output tokens, and the reducer counts
  exactly 12 tokens for it. Forced by `max_new_tokens` and by G3.
- **G7 expert-parallel byte monotonicity.** Total routed MoE bytes at
  `ep_world = 4` are not greater than at `ep_world = 8` for the identical token
  set. This is by construction, because `owner_4(e) = owner_8(e) % 4` maps each
  token's destination set onto a set that is no larger and drops the
  destination that becomes local. Only the strict version could fail, and with
  4 requests, 19 forwarded tokens each and 24 layers it cannot realistically
  fail, so the ratio is reported and never scored.
- **S1 to S5**, the physical bounds below.

### Scored exact relations

- **E1 independent TTFT and TPOT.** For every request in every simulated cell,
  the study recomputes, using the Python standard library and the declared
  arrivals only, the request's first-token time, last-token time, token count,
  TTFT, TPOT as an exact `Fraction`, and its TTFT and decode attributions from
  the per-step artifact service lists. Every one of those values equals the
  value `HtsimRequestMetricReducer` published, exactly.
- **E2 independent per-request directed bytes.** For every simulated step,
  every layer, every phase, every scheduled request and every directed rank
  pair, the byte count recomputed straight from the raw v2 trace with the
  standard library equals `MoeAllToAll.request_pair_payload_bytes` exactly.
  Nothing from `simllm.traffic` produces the expected side. The report gives
  the per-layer and per-request tables, not only their sum.
- **E3 executed GOAL artifacts.** For every simulated step, layer and phase,
  the sum over requests of the E2 table equals the sends actually present in
  the GOAL artifact `htsim_rnic` consumed, pair by pair, read from the files
  under the cell work directory rather than from an in-memory render.
- **E4 ownership.** Every send in every executed dispatch artifact has
  `source == 0`, and every send in every executed combine artifact has
  `destination == 0`. No peer rank sources a scheduled token.
- **E5 projection fidelity.** For every request, phase, token index and each of
  the 24 layers, the ordered top-8 expert ids in the `RoutedExperts` projection
  equal the ordered top-8 expert ids in the raw trace.

### Scored behavioral relations

- **B1 bandwidth linearity.** Fabric service is affine in the reciprocal of the
  link rate with a bandwidth-independent constant. For every artifact index `i`
  of every step whose scheduling composition is present in all three `ep8`
  cells, with `S(bw, i)` the artifact's fabric service in ps,

  ```
  S(100G, i) - S(200G, i) == 2 * (S(200G, i) - S(400G, i))
  ```

  to within `max(1000 ps, 1e-6 * S(100G, i))`.
- **B2 halving response on matched compositions.** For every step whose
  scheduling composition appears in both cells of a halving pair
  (400G to 200G, and 200G to 100G), the slower cell's step makespan is strictly
  greater than the faster cell's and strictly less than twice it, because the
  propagation term does not scale with the link rate. This relation is
  registered on step makespans only. It is deliberately **not** registered on
  TTFT or TPOT: the vLLM end-to-end study found that a slower fabric changes
  which requests share a step, so a per-request interval can change composition
  across cells and a serialization bound does not apply to it (PLAY-15).
- **B3 the loop is closed.** The number of scheduler steps the framework takes
  is monotonically non-increasing in the link rate across the ladder,

  ```
  steps(ep8-400g) >= steps(ep8-200g) >= steps(ep8-100g)
  ```

  with **at least one strict inequality**, while every request still receives
  exactly 12 output tokens in all three cells. The mechanism: the simulated
  makespan of each step is what advances the engine's `VirtualClock`, a slower
  fabric therefore covers more virtual time per step, the arrival gate releases
  later requests after fewer steps, and the framework batches them together.
  If the fabric timing did not feed back into SGLang's batching decisions the
  three cells would produce identical step sequences and the chain would have
  no strict inequality. This is the flagship relation of this study.
- **B4 expert-parallel compute response.** Narrowing the expert-parallel world
  from 8 to 4 strictly increases the mean per-step compute service, and the
  ratio `compute_service(ep4) / compute_service(ep8)` lies in `[1.45, 1.65]`.
  First-principles prediction, derived below: **1.544**.

### Reported, never scored

- the routed-byte ratio between `ep4` and `ep8` (G7 is the fatal half),
- the `control-nosink` step sequence and its comparison against the simulated
  cells,
- wall time, step counts, `htsim_rnic` invocation counts and stored bytes,
- the error budget, which is a stated position rather than a measurement.

## The pre-freeze entailment question, answered per relation

The question, asked of every scored relation before freezing it: *given the
fatal guards and identity checks already registered, and given how the fixture
is constructed, can this relation fail?*

- **E1: yes.** The reducer's own interval conservation cannot fail as a scored
  relation, because `HtsimRequestMetricReducer.consume` and
  `RequestLatencyTotals.__post_init__` **raise** when it does not hold. That
  naive form is entailed and is registered as fatal guard G4 instead. What E1
  scores is different: a standard-library recomputation of TTFT, TPOT and the
  two attributions from the per-step rows and the declared arrivals, compared
  against the reducer's published values. A reducer that carried a pending
  attribution to the wrong request, that started the first interval at the
  wrong endpoint, or that counted a co-scheduled non-sampling row as a token
  would conserve internally and still disagree with the independent value.
- **E2: yes.** The expected side is produced by re-reading the trace JSONL with
  `json` and applying the frozen owner and dedup rules by hand. A defect in
  token slicing, in dedup, or in ownership shows up as a mismatch.
- **E3: yes, and it is not entailed by E2.** E2 compares an independent
  recomputation against an in-memory library table; E3 compares it against the
  bytes written into the artifacts the backend actually consumed. A renderer
  defect separates them.
- **E4: yes, and it is not entailed by E2 or E3.** A replication defect that
  made every rank source the scheduled tokens keeps every per-request table
  self-consistent while adding peer-sourced sends. That is exactly how an
  earlier 8x routed-byte defect in this repository survived an aggregate check.
- **E5: yes, but narrowly.** A projection that dropped tokens would abort the
  run in `RoutedMoeSupply` rather than fail this relation, and a projection
  that changed an expert id to one with a different owner would fail E2. What
  E5 alone can catch is a permutation inside a layer's top-8 tuple that
  preserves the owner set. This is declared here as narrow risk rather than
  presented as an independent test of the whole path.
- **B1: yes.** Any bandwidth-dependent term the model does not claim, or any
  per-flow control term that scales with the rate, breaks the three-point
  identity. The test needs no knowledge of the propagation constant.
- **B2: yes.** The upper half fails if any bandwidth-independent term is
  missing, the lower half fails if the fabric service is insensitive to the
  rate. Restricting it to matched compositions is what makes the serialization
  argument applicable at all.
- **B3: yes, and this is the relation the study exists for.** If the sink were
  not installed, or if the worker ignored the `StepResult` and settled on its
  own estimate, or if the arrival gate were bound to a different clock, the
  three cells would take the same number of steps and the chain would hold with
  no strict inequality, which this clause counts as a failure. It can also fail
  in the other direction, if the admission points happen to coincide.
- **B4: yes.** The band is 6.5 percent wide around a first-principles
  prediction. It fails if the compute model is not weight-read bound at this
  batch size, if `local_num_experts` is not honored, or if the embedding and
  head weights are accounted differently than the derivation assumes.

Two candidate relations were **removed from the scored set** after asking this
question, rather than kept as padding:

1. *Per-request TTFT and TPOT conserve exactly against the seven-component
   attribution.* Entailed: the library raises instead of reporting a failure.
   Registered as fatal guard G4.
2. *Every TTFT is strictly positive and is measured from the declared arrival.*
   Entailed: `HtsimRequestMetricReducer.consume` raises when a request is
   scheduled before its declared arrival, the study hands the same arrival
   mapping to the gate and to the reducer, and every simulated step has a
   strictly positive makespan, so `first_token_at_ps > arrived_at_ps` follows
   by construction. It is checked and it is fatal, and it is not scored.
3. *Narrowing the expert-parallel world strictly decreases routed bytes.* The
   weak inequality is forced by the owner rule, as G7 records. Only strictness
   could fail and it cannot realistically fail on this token set.

## Physical sanity bounds, stated before any digit is read

Each bound is a floor and a ceiling from first principles. A value inside a
range is not proof of correctness; a value outside one is proof of a defect in
the model, the harness or the reading.

Per-rank resident weights at expert-parallel width `W`, at 2 bytes per element:

```
per layer   = attention 3,145,728 + experts (32 / W) * 1,572,864 + router 32,768
embed/head  = 49,155 * 1,024 = 50,334,720 (tied, counted once)
W = 8:  24 * 9,469,952  + 50,334,720 = 277,613,568 params = 555,227,136 bytes
W = 4:  24 * 15,761,408 + 50,334,720 = 428,608,512 params = 857,217,024 bytes
```

At the B100 envelope's `8.0e12` bytes/s the weight-read floor of any step is
`69.4 us` at `W = 8` and `107.2 us` at `W = 4`; with the provider's 0.7 derate
the expected per-step compute service is about `99.2 us` and `153.1 us`. Their
ratio, `857,217,024 / 555,227,136 = 1.5439`, is B4's prediction. The active
FLOP term is far smaller: an 8-token prefill activates about `377e6` parameters
and costs about `6.0e9` FLOP, which is microseconds at any B100-class peak, so
every step in this study is weight-read bound and the ratio is not diluted by a
compute-bound step.

At `W = 8` rank 0 owns 4 of the 32 experts, so a token's 8 distinct selected
experts put at least 1 and at most 7 distinct **remote** destinations on the
wire, each taking one 2048-byte vector per layer per phase.

- **S1** The first prefill step dispatches exactly the 8 prompt tokens of the
  first admitted request. Its total routed MoE bytes lie in
  `[8 * 24 * 2 * 1 * 2048, 8 * 24 * 2 * 7 * 2048]`, i.e.
  `[786,432, 5,505,024]` bytes.
- **S2** At 400 Gbit/s the link moves `5.0e10` bytes/s, i.e. 20 ps per byte.
  The first prefill step's serialization floor is therefore
  `48 * 8 * 1 * 2048 * 20 ps = 15.7 us` and its ceiling
  `48 * 8 * 7 * 2048 * 20 ps = 110.1 us`. Bytes over link rate is a floor no
  flow can beat.
- **S3** Per-step compute service at `W = 8` lies in `[60 us, 200 us]`, around
  the `69.4 us` weight-read floor and the `99.2 us` derated expectation.
- **S4** A prefill step makespan at 400 Gbit/s lies in `[80 us, 2000 us]`. The
  floor is S2's floor plus S3's floor. The ceiling is deliberately loose,
  because the ideal-network model carries an additive per-artifact propagation
  term and 48 artifacts execute serially.
- **S5** A decode step makespan at 400 Gbit/s lies in `[100 us, 600 us]`. With
  at most 4 co-scheduled decode tokens the serialization term is at most
  `48 * 4 * 7 * 2048 * 20 ps = 55.1 us`, so a decode step is compute and
  propagation dominated and must sit far closer to S3 than a prefill step does.
- **S6, reported not bounded.** The implied per-request decode rate `1 / TPOT`
  is reported. Sanity against the real system rather than against the
  simulator: a 400M-active-parameter MoE served on one real accelerator decodes
  a single request at roughly `10^2` tokens per second, because the fixed host
  cost per decode step dominates at this model size. A simulated rate near
  `10^3` to `10^4` is the expected consequence of charging zero fixed per-step
  cost, and a simulated rate that landed near the realistic value would be
  evidence of a compensating error rather than of accuracy.

## Error budget, stated quantitatively

The compute path is an uncalibrated bootstrap and COMP-1 owns real calibration.
Magnitudes are relative to a decode step whose simulated makespan is of order
`0.25 ms`.

1. **Zero fixed per-step cost.** `HostInitiationModel.ideal()` charges nothing
   for kernel launch, scheduler bookkeeping, sampling or the framework's own
   Python work. On a real engine those cost of order `0.2 ms` to `2 ms` per
   decode step, so this single omission is expected to be comparable to or
   larger than the entire simulated step.
2. **Uncalibrated compute.** A flat 0.7 derate against peak; real achieved
   bandwidth for these shapes is typically 0.6 to 0.85 of peak, so the compute
   term carries roughly plus or minus 30 percent before architecture
   calibration.
3. **Envelope.** `GPU_ENVELOPES["b100"]` at `8.0e12` bytes/s. An H100 at
   `3.35e12` bytes/s would make the same step's compute about 2.4 times longer.
4. **Ideal network.** `rnic-nn-fluid` has no congestion control, no switch
   queueing and no incast. Direction: optimistic under load.
5. **Traffic coverage.** Only captured MoE dispatch and combine cross the
   fabric. No tensor-parallel all-reduces, no pipeline stages, no KV movement,
   no weight traffic. SGLang's own communicator is zero time, so a run with it
   enabled would understate communication by exactly the whole of it; it stays
   disabled here. Direction: understates network time.
6. **Degenerate token stream.** The fabricated constant token id means the
   prefix cache sees a stream no real workload produces, and the routing
   attributed to each forwarded token comes from a separate capture rather than
   from the tokens this run served. Direction: not bounded by this study.
7. **No end-to-end calibration.** No part of this chain has been compared
   against a measured SGLang serving trace of this model on this hardware.

Therefore, stated before the run:

- **Supported:** identity and conservation claims; per-request and per-token
  byte attribution against an independent recomputation; relative sweeps in
  which exactly one parameter moves; the shape of the decomposition; and the
  demonstration that the simulated fabric's timing changes SGLang's own
  batching decisions.
- **Not supported:** any absolute TTFT or TPOT prediction for real hardware;
  SLO attainment; goodput; comparison against another simulator's or a real
  deployment's absolute numbers; any claim about a GPU other than the modeled
  B100 envelope; any claim about served token identities.

## Traps this freeze addresses explicitly

- **`num_sampled` is not populated at the SGLang worker seam** (SGL-12). When
  it is `None`, `sampled_request_ids` counts every scheduled row as producing a
  token, so a mid-prompt chunked-prefill row would be scored as a token and
  TTFT would land one step early with nothing raising. This study disables
  chunked prefill and asserts, per row, that no scheduled prefill row is a
  mid-prompt extend (G3). The absent field itself is not fixed here; SGL-12
  keeps it.
- **Replaying JSONL offline is not a closed loop.** Records written under the
  roofline fallback can be re-fed to the sink afterwards and produce
  arithmetically self-consistent numbers, which is what `examples/m4` did. G2
  proves same-process identity and B3 proves the feedback changed the
  framework's decisions.
- **Another framework's routing.** G1 refuses any trace whose provenance is not
  `sglang`.
- **Capture and simulate are separate phases.** The oracle-capture gate and the
  simulated-worker gate are mutually exclusive and raise if set together. This
  study only simulates.

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

`--check-only` validates every frozen input and produces no artifacts. The
parent interpreter never imports SGLang or torch: each cell runs in a child
process under `--sglang-python` with `PYTHONPATH` pointing at this worktree,
the same parent and child split `examples/sglang_layer_id_v1` uses.
`SIMLLM_TXT2BIN` selects the GOAL text-to-binary converter when it is not
discoverable from the build tree, and `SIMLLM_HTSIM_RNIC` is set by the parent
from `--htsim-rnic`.
