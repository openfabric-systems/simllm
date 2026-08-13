# End to end replay v1 expectations

This document freezes the acceptance contract for the whole-mission study
before the harness is implemented and before any result-producing run. It is
the pre-run record required by the validation discipline: every relation,
bound, direction and normalization below is written down here first, and the
result report cites this freeze commit.

## What this study claims, and what it refuses to claim

SimLLM's stated purpose is to simulate the end to end of LLM serving with a
per-request replay. Roughly forty studies exist in this repository and each is
scoped to one component or one mechanism. None of them makes the whole claim.
The one time the whole chain ran, it ran as an unregistered exploration and
three defects were found in it afterwards. This study exists to establish, under
the evidence discipline, what the repository can truthfully say about its own
mission.

The claim is deliberately split into four separately registered parts so a weak
part cannot ride on a strong one:

1. per-request replay correctness,
2. per-token routing fidelity, reported per layer and per request,
3. end-to-end conservation of TTFT and TPOT, and
4. scale.

This study does **not** claim absolute accuracy against real hardware, and it
does not claim that any simulated millisecond corresponds to a measured
millisecond. The quantitative error budget in the last section says why, names
each unmodeled mechanism with its rough magnitude, and states which comparisons
the record therefore supports.

## Frozen configuration

### Oracle capture

The oracle is a pinned CPU capture through `simllm.preplay`:

- model `ibm-granite/granite-3.0-1b-a400m-instruct`, revision
  `ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445`, loaded offline from the cache
  directory supplied on the command line;
- runner `TransformersCpuRunner`, device CPU, dtype `float32`;
- sampling `SamplingConfig.greedy()`, i.e. mode `greedy`, no seed;
- prompt format `chat`;
- trace schema `simllm-preplay-trace-v1`.

The capture records, for every forwarded token, the ordered top-8 expert ids at
each of the 24 MoE layers. The last generated token of a request is never fed
forward, so a request with `n` output tokens contributes exactly `n - 1` decode
forward rows and `len(input_token_ids)` prefill forward rows.

### Model geometry

`ModelDims` is the per-rank sharded geometry of that model:

| field | value |
|---|---|
| `num_layers` | 24 |
| `hidden_size` | 1024 |
| `intermediate_size` | 512 |
| `num_heads` | 16 |
| `num_kv_heads` | 8 |
| `head_size` | 64 |
| `vocab_size` | 49152 |
| `dtype_bytes` | 2 |
| `num_experts` | 32 |
| `top_k` | 8 |
| `moe_intermediate_size` | 512 |
| `local_num_experts` | `32 // ep_world` |

The routed hidden vector is therefore `hidden_size * dtype_bytes = 2048` bytes.

### Placement, ownership and traffic rules

- expert owner rule: `owner(layer, expert) = expert % ep_world`, one immutable
  `ExpertPlacementSnapshot` at `placement_epoch = 0` for the whole run;
- `RoutedMoeSupply.engine_rank = 0`. Rank 0 is the only rank that dispatches
  scheduled tokens; the other EP ranks own experts and carry no scheduled
  tokens;
- dispatch dedup rule: for one token at one layer, the set of destination ranks
  is the set of owners of its top-8 experts. Each distinct **remote**
  destination receives exactly one 2048-byte vector. A destination equal to
  `engine_rank` stays local and contributes zero fabric bytes;
- combine is the exact transpose of dispatch after owner-side pre-reduction;
- no placement manifest is supplied, so the run takes the all-remote
  compatibility level: `nvlink_directed_bytes == 0` and every MoE byte crosses
  the simulated fabric;
- `tp_ranks = (0,)`, so there are no tensor-parallel all-reduces. The only
  fabric traffic in this study is captured MoE dispatch and combine.

### Network and compute

- backend profile `rnic-nn-fluid`, the ideal-network baseline, executed by
  `htsim_rnic` through `HtsimStepSink`. Each of the 24 layers contributes one
  dispatch artifact and one combine artifact, so a step that moves tokens
  executes 48 GOAL artifacts;
- compute provider `RooflineProvider()`, i.e. `efficiency = 0.7`, against the
  default `GPU_ENVELOPES["b100"]` envelope at `8.0e12` bytes/s of memory
  bandwidth. `HostInitiationModel` keeps `initiation_delay_ps = 0` and the
  `ideal` profile, so a modeled step charges zero fixed per-step cost;
- arrival gating is on: `RequestAdmissionGate` in `ARRIVAL_GATED` mode, bound to
  the adapter's single `VirtualClock`, releases a request to
  `LLMEngine.add_request` only once the virtual clock has reached its declared
  arrival. Arrivals are `index * 1_000_000_000` ps, i.e. one millisecond apart
  in declaration order;
- replay pinning is on: `SimExecutorConfig(mode="virtual", replay_run_path=...)`
  binds every served token to the oracle sequence under the request's own
  scheduler identity, added with `llm_engine.add_request(<oracle id>, ...)`.

### Cells

Two parameters vary, as the validation discipline requires. Case A is the small
case; case B is the registered expansion, not an afterthought.

| cell | case | requests | ep_world | link rate |
|---|---|---|---|---|
| `a-ep8-400g` | A | 3 | 8 | 400 Gbit/s |
| `a-ep8-200g` | A | 3 | 8 | 200 Gbit/s |
| `a-ep8-100g` | A | 3 | 8 | 100 Gbit/s |
| `a-ep4-400g` | A | 3 | 4 | 400 Gbit/s |
| `b-ep8-400g` | B | 12 | 8 | 400 Gbit/s |

## Registered acceptance clauses

Each clause is numbered, and the result report quotes each one and maps it to
evidence. A clause that the run does not demonstrate stays open and takes a new
ID from the range PLAY-15, PLAY-16, CORE-56. Clauses marked *fatal-unscored* are
conservation identities or configuration-forced zeros: violating one voids the
run rather than costing a point, and they never enter a scored denominator.

### Claim 1: per-request replay correctness

- **C1.1** For every request in every cell, the served token id sequence equals
  the oracle `output_token_ids` exactly, element by element. A single differing
  element is a failure, not a rounding difference.
- **C1.2** For every request in every cell, the framework finish reason agrees
  with the oracle stop reason under this normalization, frozen here:
  oracle `length-cap` requires `finish_reason == "length"`; oracle `eos`
  requires `finish_reason == "stop"` and `stop_reason is None`; oracle
  `stop-string` requires `finish_reason == "stop"` and `stop_reason is not
  None`.
- **C1.3** Request identity is conserved end to end: `add_request` returns the
  oracle id for every request, the set of scheduler ids observed in every
  `StepRecord` is a subset of the oracle id set, and the replay snapshot's
  completed-request set equals the oracle request set exactly.

### Claim 2: per-token routing fidelity, per layer and per request

An aggregate check is what hid the earlier 8x routed-byte defect, so every
clause here is evaluated and reported per layer and per request. The evidence is
an independent recomputation: the capture JSONL is re-read with the Python
standard library only, and the frozen owner and dedup rules above are applied by
hand. Nothing in `simllm.traffic` participates in producing the expected side.

- **C2.1** For every request, every phase, every token index and every one of
  the 24 layers, the ordered top-8 expert ids in the study's routed-experts
  projection equal the ordered top-8 expert ids in the raw capture. The number
  of compared `(request, phase, token, layer)` cells is reported.
- **C2.2** For every simulated step, every layer, every phase, every scheduled
  request and every directed rank pair, the independently recomputed byte count
  equals `MoeAllToAll.request_pair_payload_bytes` exactly. The report gives the
  per-layer and per-request tables, not only their sum.
- **C2.3** For every simulated step and every layer and phase, the sum over
  requests of the C2.2 table equals the sends actually present in the GOAL
  artifact that `htsim_rnic` consumed, pair by pair. The comparison reads the
  artifacts written under the cell's work directory, not an in-memory render.
- **C2.4** Every send in every executed dispatch artifact has `source == 0`, and
  every send in every executed combine artifact has `destination == 0`. No peer
  rank sources a scheduled token. This is the direct guard against the 8x
  defect, which arose because all eight ranks replicated the scheduled tokens.
- **C2.5** *(fatal-unscored)* For every simulated step, the routing mode
  reported by the backend is `captured`, the placement epoch is 0, and the
  backend reports quiescence.

### Claim 3: end-to-end conservation

The seven-component `LatencyAttribution` already exists in `simllm.core` and is
used unchanged. The packet-level step sink publishes a whole-step makespan and
no per-request endpoint, so, exactly as in `per_request_fidelity_v1`, no rule
divides a step makespan among co-scheduled requests: every request co-scheduled
in a step is charged that step's whole service, and the decomposition is of the
request's own elapsed interval.

- **C3.1** *(fatal-unscored)* For every simulated step,
  `sum(composed_phase_service_ps) == step_latency_ps`, and
  `completed_at_ps == virtual_time_ps + step_latency_ps`.
- **C3.2** For every request and every emitted token, the interval attribution
  conserves exactly: `attribution.total_ps == completed_at_ps - previous
  accounted point`. Every component is a nonnegative integer.
- **C3.3** *(fatal-unscored)* In this all-remote, zero-fixed-cost configuration
  the components `kv_ps`, `dma_ps`, `nic_ps` and `control_ps` are exactly zero
  for every interval, and only `queue_ps`, `kernel_ps` and `collective_ps` are
  populated.
- **C3.4** *(fatal-unscored)* Each executed artifact contributes to exactly one
  component. An artifact with nonzero fabric service contributes its composed
  service to `collective_ps`; an artifact with zero fabric service contributes
  its composed service to `kernel_ps`. No artifact contributes twice and none is
  skipped, so there is no unattributed remainder.
- **C3.5** For every request, `ttft_ps == first_token_completed_at_ps -
  arrival_ps`, and that value equals the sum of the seven components of the
  first interval's attribution, with no remainder.
- **C3.6** For every request with at least two output tokens,
  `tpot_ps * (token_count - 1)` equals the sum of the components of all
  post-first-token intervals exactly, evaluated as a `Fraction`.
- **C3.7** `ttft_ps > 0` for every request in every cell. The earlier
  exploratory run produced negative TTFT for two of three requests because it
  admitted every request at `t = 0` while declaring staggered arrivals; the
  arrival gate is the fix and this clause is its acceptance test.
- **C3.8** For every request, the sum of `kernel_ps` over all its intervals
  equals the sum of `compute_service_ps` over the steps that request was
  scheduled in. This ties the attributed compute back to the provider's own
  published per-step number rather than to a derived quantity.

### Claim 4: scale

- **C4.1** Case B, at four times the request count of case A, completes the same
  chain with C1, C2 and C3 holding in full.
- **C4.2** The report states, for both cases, wall time, scheduler steps,
  `htsim_rnic` invocations, total routed bytes and peak per-rank egress, and
  names the mechanism that sets the practical limit together with the case size
  at which it stops being practical.

### Sweep relations

- **C5.1** Fabric service is affine in the reciprocal of the link rate with a
  bandwidth-independent constant. For every artifact index `i` of every step
  present in all three bandwidth cells of case A, with `S(bw, i)` the artifact's
  fabric service in ps,

  ```
  S(100G, i) - S(200G, i) == 2 * (S(200G, i) - S(400G, i))
  ```

  to within `max(1000 ps, 1e-6 * S(100G, i))`. This is a three-point linearity
  test, so it needs no knowledge of the model's propagation constant or of its
  per-byte rate, and a failure localizes to a bandwidth-dependent term the model
  does not claim.
- **C5.2** Halving the link rate strictly increases every step's makespan and
  strictly increases every request's TTFT and TPOT, and the increase is strictly
  less than a factor of two, because the propagation term does not scale.
- **C5.3** Narrowing the expert-parallel world from 8 to 4 strictly decreases
  total routed MoE bytes for the identical token set, because
  `owner_4(e) = owner_8(e) % 4` maps the destination set of every token onto a
  set that is no larger and drops the destination that becomes local.
- **C5.4** Narrowing the expert-parallel world from 8 to 4 strictly increases
  the per-step compute service, because each rank then holds 8 resident experts
  instead of 4. The ratio `compute_service(ep4) / compute_service(ep8)` is
  expected in `[1.35, 1.75]`: the per-layer resident weight count moves from
  `3,145,728 + 4 * 3 * 1024 * 512 = 9,437,184` to
  `3,145,728 + 8 * 3 * 1024 * 512 = 15,728,640` parameters, a ratio of 1.667,
  diluted by the unsharded embedding and head weights.

## Physical sanity bounds, stated before any digit is read

Each bound is a floor and a ceiling from first principles. A measured value
inside a range is not proof of correctness; a value outside one is proof of a
defect in the model, the harness or the reading, and is treated as a defect to
find rather than a result to publish.

Let `T` be the number of tokens the engine rank dispatches in a step. At
`ep_world = 8`, rank 0 owns 4 of the 32 experts, so a token's 8 selected experts
put at least `ceil((8 - 4) / 4) = 1` and at most 7 distinct **remote**
destinations on the wire. Each destination takes one 2048-byte vector, per
layer, per phase.

- **S1** Case A step 0 dispatches all 54 prompt tokens. Its total routed MoE
  bytes lie in `[54 * 24 * 2 * 1 * 2048, 54 * 24 * 2 * 7 * 2048]`, i.e.
  `[5,308,416, 37,158,912]` bytes. Under a uniform-selection model the expected
  count of distinct remote destinations per token-layer is
  `8 * (1 - C(28,8)/C(32,8)) * 7/8 = 4.931`, giving about 26.2 MB, so a value
  near either endpoint indicates strongly skewed expert popularity and is worth
  reporting either way.
- **S2** Rank 0's egress is the bottleneck of a dispatch artifact and its
  ingress is the bottleneck of a combine artifact. At 400 Gbit/s the link moves
  `5.0e10` bytes/s, i.e. 20 ps per byte. A prefill step of case A therefore has
  a serialization floor of `48 * 54 * 1 * 2048 * 20 ps = 106.2 us` and a
  serialization ceiling of `48 * 54 * 7 * 2048 * 20 ps = 743.4 us`.
- **S3** Per-rank resident weights at `ep_world = 8` are about 554 MB: 24 layers
  of `3,145,728` attention parameters and `6,291,456` expert parameters at 2
  bytes each, plus about 100 MB of embedding and head weights. At the B100
  envelope's `8.0e12` bytes/s the weight-read floor of any decode step is
  `69.2 us`, and with the provider's 0.7 derate the expected per-step compute
  service is about `98.9 us`. The per-step compute service must lie in
  `[60 us, 200 us]` at `ep_world = 8`.
- **S4** A case A prefill step makespan at 400 Gbit/s lies in
  `[0.27 ms, 3.0 ms]`. The floor is S2's floor plus S3's floor. The ceiling is
  deliberately three times the naive `S2 ceiling + S3 ceiling` of about
  `0.94 ms`, because the fluid model may carry an additive per-flow control
  term; a measured value above `1.0 ms` is a signal to isolate that term and
  report it rather than a failure.
- **S5** A case A decode step makespan at 400 Gbit/s lies in
  `[0.15 ms, 0.60 ms]`. With at most 3 co-scheduled decode tokens the
  serialization term is at most `48 * 3 * 7 * 2048 * 20 ps = 41.3 us`, so a
  decode step is compute- and propagation-dominated and must be far closer to
  S3's compute floor than a prefill step is.
- **S6** Per-step compute service is weight-read bound, so it must vary by less
  than 10 percent across decode steps of different token counts within one cell.
  If it moves proportionally with token count, the step is not weight-bound and
  the roofline classification is wrong.
- **S7** The implied per-request decode rate `1 / TPOT` must lie in
  `[1000, 20000]` tokens per second. This band is a defect detector, not a
  plausibility claim: see the error budget below, which states that the true
  value for a comparable real deployment is expected to be far lower and that a
  simulated value inside the realistic range would itself indicate an error.

## Error budget, stated quantitatively

The compute path is an uncalibrated bootstrap on the target architecture. COMP-1
is open and owns real calibration. Each unmodeled mechanism below is named with
its rough magnitude relative to a case A decode step whose simulated makespan is
of order `0.2 ms`.

1. **Zero fixed per-step cost.** `compute/host.py` keeps
   `initiation_delay_ps = 0` on the `ideal` profile, so kernel launch, scheduler
   bookkeeping, sampling and the framework's own Python work are charged
   nothing. On a real engine these cost of order `0.2 ms` to `2 ms` per decode
   step. This single omission is expected to be comparable to or larger than the
   entire simulated step.
2. **Uncalibrated compute.** The roofline derates peak by a flat 0.7. Real
   achieved memory bandwidth for these kernel shapes is typically 0.6 to 0.85 of
   peak, so the compute term carries roughly plus or minus 30 percent even before
   architecture calibration. Expected magnitude: about 30 us on a 99 us term.
3. **Envelope confusion.** The default envelope is `GPU_ENVELOPES["b100"]` at
   `8.0e12` bytes/s. A caller who omits `gpu=` gets B100, not the H100 a reader
   may assume. An H100 at `3.35e12` bytes/s would make the same step's compute
   about 2.4 times longer. Any reader who assumes H100 is reading a number that
   is optimistic by that factor.
4. **Ideal network.** `rnic-nn-fluid` has no congestion control, no switch
   queueing and no incast. The repository's own acceptance bar puts a physical
   profile within 2x of this baseline on short flows, so the network term is
   expected to be optimistic by up to that factor under load.
5. **Traffic coverage.** Only captured MoE dispatch and combine cross the
   fabric. There are no tensor-parallel all-reduces (`tp_ranks = (0,)`), no
   pipeline stages, no KV-cache movement and no weight or parameter traffic.
   Direction: understates network time.
6. **Expert compute granularity.** The roofline reads all resident expert
   weights every step regardless of which experts the batch actually hits, and
   models no per-expert capacity factor, padding or grouped-GEMM tail. Direction
   is ambiguous and the magnitude is not bounded by this study.
7. **No end-to-end calibration.** No part of this chain has been compared
   against a measured serving trace of this model on this hardware.

The consequence, stated before the run: the simulated TPOT is expected to be
optimistic relative to a real deployment of the same model by roughly one order
of magnitude, plausibly between 3x and 30x, dominated by item 1. A simulated
TPOT that landed within 2x of a plausible real value would be evidence of a
compensating error, not of accuracy.

Therefore:

- **Supported by this study:** conservation and identity claims; per-request and
  per-token byte attribution; relative sweeps in which exactly one parameter
  moves and everything else is held fixed, including the bandwidth ladder and
  the expert-parallel width change; the shape of the decomposition, i.e. which
  component dominates and how that share moves with a parameter.
- **Not supported by this study:** any absolute TTFT or TPOT prediction for real
  hardware; SLO attainment against a real service-level objective; absolute
  goodput; any comparison against a different simulator's or a real
  deployment's absolute numbers; any claim about a GPU other than the modeled
  B100 envelope.

## Evidence classes

Counts from different classes are never summed into one headline total.

- **Fatal-unscored guards:** C2.5, C3.1, C3.3, C3.4, and the S1 to S7 physical
  bounds. A single violation voids the run for the purpose of closing anything,
  and the run is reported as void with findings.
- **Scored exact-oracle relations:** C1.1, C1.2, C1.3, C2.1, C2.2, C2.3, C2.4,
  C3.2, C3.5, C3.6, C3.7, C3.8, C4.1. Thirteen relations, each evaluated over
  every request, layer and step of every cell it applies to.
- **Scored behavioral relations:** C5.1, C5.2, C5.3, C5.4. Four relations.
- **Reported, not scored:** C4.2, and the error budget, which is a stated
  position rather than a measurement.

Entailment: C2.3 is not entailed by C2.2, because C2.2 compares an independent
recomputation against an in-memory library table while C2.3 compares that
recomputation against the bytes actually written to the artifacts the backend
consumed. C2.4 is not entailed by C2.2 or C2.3 either, because a uniform
replication defect would keep every per-request table self-consistent while
adding peer-sourced sends; it is scored on the executed artifacts for exactly
that reason. C3.5 is not entailed by C3.2: C3.2 checks each interval in
isolation, C3.5 checks that the first interval's endpoints are the declared
arrival and the first token. C3.6 is not entailed by C3.2 for the same reason.
C3.8 is not entailed by C3.1 or C3.4, because those conserve the makespan
without tying the compute share to the provider's separately published number.

## Pre-freeze source audit

The audited worktree state is the parent of this commit. The audit found:

- `simllm/backends/step_sink.py` publishes `StepResult(step_index,
  step_latency_ps, completed_at_ps)` and nothing else. The packet-level sink is
  a one-argument legacy sink and produces no `request_metrics`, so the
  seven-component attribution is not reachable from it today.
- `simllm/backends/device_step_sink.py` does reach `CompletionReducer` and
  therefore `RequestMetric`, but it executes on the analytic
  `CoarseDeviceRuntime` and never invokes `htsim_rnic`.
- `simllm/core/completion.py` owns the interval logic this study needs:
  a scheduler gap charged to `queue_ps`, a carried pending attribution for
  co-scheduled requests that do not sample this step, and a hard conservation
  check on every sampled interval.
- Consequently the minimum library change is a reducer that turns the
  packet-level sink's already public per-step outcomes into the existing
  `RequestMetric` and `LatencyAttribution` types. That reducer is what this
  study adds; it introduces no second timing authority, because every value it
  reads is published by the one sink that ran the backend.
- `simllm.core.step.AdditiveVisitTotals` is a work sum over `QueueVisit`
  records. The packet-level sink produces no `QueueVisit`, so the additive work
  sum is not reachable on this path. No clause above depends on it, and the
  result report records the gap in prose rather than claiming it.

## Reproduction

```
python examples/end_to_end_replay_v1/run_study.py \
    --cache-dir <hugging-face-cache-root> \
    --htsim-rnic <path-to-htsim_rnic> \
    --run-dir <writable-output-directory>
```

`--check-only` validates every frozen input and produces no artifacts. Capture
and replay require the environment that carries torch, transformers and vLLM
0.26.0, with `PYTHONPATH` pointing at this worktree; `SIMLLM_TXT2BIN` selects
the GOAL text-to-binary converter when it is not discoverable from the build
tree.
