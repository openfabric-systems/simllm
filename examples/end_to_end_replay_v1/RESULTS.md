# End to end replay v1 results

The whole mission claim, run under the evidence discipline for the first time.
Real requests enter a real vLLM scheduler at declared arrivals, the simulated
executor serves them from a pinned CPU oracle, the captured MoE routing drives
every dispatch and combine onto a packet-level fabric timed by `htsim_rnic`,
and per-request TTFT and TPOT come back out with a partition that conserves
exactly.

Expectations were frozen in commit `0e3d38b`, before the harness, the library
change and every run below existed. Every clause quoted here is quoted from
that commit's `expectations.md`.

## Outcome in one paragraph

The run is **not void**: all ten fatal guards held. **Thirteen of thirteen**
scored exact-oracle relations passed and **three of four** scored behavioral
relations passed. Claims 1, 2, 3 and 4 hold. The one failure is inside claim
5's sweep relations: C5.2's "strictly less than a factor of two" bound holds for
every step makespan and every TTFT on both halvings of the bandwidth ladder, and
holds for every TPOT on the first halving, but a three-token request moves its
TPOT by 2.628 on the second halving because the slower fabric changes which
requests share its steps. That undemonstrated part is registered as **PLAY-15**
and is the only new ID this study creates.

## What ran

| item | value |
|---|---|
| oracle capture | 12 Granite requests, greedy, CPU, SHA-256 `ef570a67fd8bbbb6a8d73b8ad9f73171d3eaaf9a51efc04f676bd9f25c8988fe` |
| cells | 5 (three bandwidths, two expert-parallel widths, two case sizes) |
| simulated steps | 220 |
| executed GOAL artifacts | 15,840 (72 per step: 24 compute, 48 collective) |
| `htsim_rnic` invocations | 10,560 |
| request-token intervals reduced | 471 |
| wall time | 711 s, 678 s, 600 s, 680 s and 1,084 s per cell, run concurrently |

The capture was produced twice from the same pinned snapshot and both runs
wrote the same SHA-256, so the oracle side of the chain is byte-reproducible.

## Fatal guards

Fatal means void, not a lost point, so these are never reported as a fraction.
All ten held.

| guard | what it asserts | observed |
|---|---|---|
| C2.5 | every simulated step reports captured routing at epoch 0 and backend quiescence | 220 of 220 steps |
| C3.1 | artifact services conserve the step makespan and completion equals release plus makespan | 0 failures |
| C3.3 | `kv_ps`, `dma_ps`, `nic_ps` and `control_ps` are exactly zero in this configuration | 0 violations over 471 intervals |
| C3.4 | each executed artifact contributes to exactly one component | 0 failures over 15,840 artifacts |
| S1 | case A first prefill step routed bytes inside `[5,308,416, 37,158,912]` | 8,859,648 |
| S3 | per-step compute service inside `[60 us, 200 us]` | 99.024 us to 99.504 us |
| S4 | case A prefill step makespan inside `[0.27 ms, 3.0 ms]` | 0.347 ms, 0.372 ms, 0.376 ms |
| S5 | case A decode step makespan inside `[0.15 ms, 0.60 ms]` | 0.204 ms to 0.215 ms |
| S6 | per-step compute varies by under 10 percent across decode steps | 0.48 percent |
| S7 | implied decode rate inside `[1000, 20000]` tokens per second | 1,634 to 4,786 |

## Claim 1: per-request replay correctness

> **C1.1** For every request in every cell, the served token id sequence equals
> the oracle `output_token_ids` exactly, element by element.

**Passes.** 24 request instances across five cells, zero divergences.

> **C1.2** For every request in every cell, the framework finish reason agrees
> with the oracle stop reason under this normalization, frozen here: oracle
> `length-cap` requires `finish_reason == "length"`; oracle `eos` requires
> `finish_reason == "stop"` and `stop_reason is None`; oracle `stop-string`
> requires `finish_reason == "stop"` and `stop_reason is not None`.

**Passes.** 24 of 24 agree. Coverage limitation, stated plainly: the capture
produced only `length-cap` and `eos`, so the `stop-string` branch of the frozen
table is untested here. It is exercised in `preplay_validation_v1`.

> **C1.3** Request identity is conserved end to end.

**Passes.** In all five cells `add_request` returned the oracle id, every
scheduler id in every `StepRecord` is an oracle id, and the replay snapshot's
completed set equals the oracle request set.

## Claim 2: per-token routing fidelity, per layer and per request

The expected side of every relation below is an independent recomputation. The
capture JSONL is re-read with the Python standard library only, and the frozen
owner rule `expert % ep_world` and the frozen dedup rule are applied by hand.
Nothing in `simllm.traffic` contributes to it.

> **C2.1** For every request, every phase, every token index and every one of
> the 24 layers, the ordered top-8 expert ids in the study's routed-experts
> projection equal the ordered top-8 expert ids in the raw capture.

**Passes.** 20,976 `(request, phase, token, layer)` cells compared, 20,976
equal.

> **C2.2** For every simulated step, every layer, every phase, every scheduled
> request and every directed rank pair, the independently recomputed byte count
> equals `MoeAllToAll.request_pair_payload_bytes` exactly.

**Passes.** 104,580 per-request directed-pair rows, zero mismatched operations.
Reported per layer and per request rather than only in aggregate, as the freeze
required. For `a-ep8-400g` the per-layer totals run from 1,843,200 bytes (layer
5) to 2,297,856 bytes (layer 20), a 25 percent spread that an aggregate check
would have hidden entirely, and the per-request totals are r00 19,607,552, r01
8,028,160 and r02 23,166,976 bytes.

> **C2.3** For every simulated step and every layer and phase, the sum over
> requests of the C2.2 table equals the sends actually present in the GOAL
> artifact that `htsim_rnic` consumed, pair by pair.

**Passes.** 55,738 directed-pair rows read back out of the artifacts on disk,
zero mismatches. This is read from the files the backend consumed, not from an
in-memory render.

> **C2.4** Every send in every executed dispatch artifact has `source == 0`, and
> every send in every executed combine artifact has `destination == 0`.

**Passes.** Zero violations. This is the direct guard against the earlier 8x
routed-byte defect, in which all eight ranks replicated the scheduled tokens.
The arithmetic of that defect is now visible in the record: `a-ep8-400g` moved
50,802,688 total routed bytes with a peak per-rank egress of 25,401,344 bytes at
rank 0 and 3.3 to 3.9 MB at each peer. A replicating engine would multiply the
total by eight while leaving the peak per-rank egress, and therefore the
makespan, almost unchanged, which is exactly why an aggregate makespan check
could not see it and a per-request ownership check can.

## Claim 3: end-to-end conservation

> **C3.2** For every request and every emitted token, the interval attribution
> conserves exactly.

**Passes.** 471 intervals, zero conservation failures.

> **C3.5** For every request, `ttft_ps == first_token_completed_at_ps -
> arrival_ps`, and that value equals the sum of the seven components of the
> first interval's attribution, with no remainder.

**Passes.** 24 of 24.

> **C3.6** For every request with at least two output tokens,
> `tpot_ps * (token_count - 1)` equals the sum of the components of all
> post-first-token intervals exactly, evaluated as a `Fraction`.

**Passes.** 24 of 24, exact in rational arithmetic.

> **C3.7** `ttft_ps > 0` for every request in every cell.

**Passes.** 24 of 24 strictly positive. The earlier unregistered run reported
negative TTFT for two of three requests because it declared staggered arrivals
and then admitted every request at `t = 0`. With `RequestAdmissionGate` in
`ARRIVAL_GATED` mode on the adapter's clock, TTFT is queue plus service by
construction.

> **C3.8** For every request, the sum of `kernel_ps` over all its intervals
> equals the sum of `compute_service_ps` over the steps that request was
> scheduled in.

**Passes.** Zero disagreements.

The per-request decomposition, `a-ep8-400g`, all values in picoseconds:

| request | arrival | TTFT | queue | kernel | collective | TPOT | tokens |
|---|---|---|---|---|---|---|---|
| r00 | 0 | 372,217,008 | 0 | 99,024,000 | 273,193,008 | 224,797,046.26 | 24 |
| r01 | 1,000,000,000 | 536,566,546 | 189,740,578 | 99,192,000 | 247,633,968 | 213,741,964 | 3 |
| r02 | 2,000,000,000 | 544,628,780 | 168,485,372 | 99,264,000 | 276,879,408 | 208,925,745.55 | 32 |

Every row sums exactly: 0 + 99,024,000 + 273,193,008 = 372,217,008. The queue
column is real framework queueing, not padding: r01 and r02 arrive while r00 is
already decoding and wait 0.19 ms and 0.17 ms for the step boundary.

## Claim 4: scale

> **C4.1** Case B, at four times the request count of case A, completes the same
> chain with C1, C2 and C3 holding in full.

**Passes.** 12 requests, all served token sequences exact, all finish reasons
agreeing, zero conservation failures over 235 intervals, zero routing
mismatches over 55,194 per-request rows and 19,294 GOAL rows, zero ownership
violations.

> **C4.2** The report states, for both cases, wall time, scheduler steps,
> `htsim_rnic` invocations, total routed bytes and peak per-rank egress, and
> names the mechanism that sets the practical limit together with the case size
> at which it stops being practical.

| | case A (3 requests) | case B (12 requests) |
|---|---|---|
| scheduler steps | 41 | 64 |
| `htsim_rnic` invocations | 1,968 | 3,072 |
| total routed bytes | 50,802,688 | 208,007,168 |
| peak per-rank egress | 25,401,344 | 104,003,584 |
| wall time | 711 s | 1,084 s |
| stored artifacts | 24 MB | 38 MB |

Four times the requests cost 1.53 times the wall time. The mechanism is one
`htsim_rnic` process per collective artifact per step: 48 processes per step,
each about 0.36 s including GOAL text-to-binary conversion and process startup,
which is essentially all of the wall time. Cost is therefore
`steps x 48 x per-process cost`, and it is dominated by step count, not by
request count: adding requests widens each step's token set, which changes the
bytes inside an artifact but not the number of artifacts.

Step count grows with two things: the longest output length, and the spread of
declared arrivals divided by the step time. At the frozen one-millisecond
spacing the run must advance through 11 ms of virtual time before the last
request is even admitted, which alone costs about 45 steps at 0.2 ms each.

Where it stops being practical: at this per-artifact cost, a 64-request case at
the same arrival spacing would need roughly 300 steps and about 1.5 hours per
cell, and a 512-token output length would need roughly 520 steps and about
2.5 hours. The practical ceiling for a five-cell sweep on one machine is therefore
around 100 steps, i.e. tens of requests with tens of output tokens. Two things
would move it: `HtsimPersistentStepSink` already exists to run prepared steps
concurrently, and the per-step process count would drop from 48 to 1 if a
backend run could preserve state across ordered artifacts, which BACK-38 owns.
Neither is in this study's scope and neither is registered here, because no
clause claimed them.

## Claim 5: sweep relations

> **C5.1** Fabric service is affine in the reciprocal of the link rate with a
> bandwidth-independent constant: `S(100G, i) - S(200G, i) == 2 * (S(200G, i) -
> S(400G, i))` to within `max(1000 ps, 1e-6 * S(100G, i))`.

**Passes, exactly.** 672 artifact comparisons over 14 scheduling compositions
present in all three bandwidth cells. Worst relative residual: **0.0**. Not
one artifact deviated by a single picosecond.

A post-specified diagnostic pins the constant this proves must exist. Because
service is affine, `2 * S(400G) - S(200G)` cancels the byte term exactly, per
artifact and with no averaging. Over 912 artifacts it returns 2,000,000 ps at
minimum and 2,000,001 ps at maximum, mean 2,000,000.32. The ideal-network model
charges exactly 2 us of propagation per collective plus the bottleneck rank's
bytes at the declared rate, with the implied bottleneck load running from 6,144
to 200,704 bytes per artifact. This number matters for the error budget below.

> **C5.2** Halving the link rate strictly increases every step's makespan and
> strictly increases every request's TTFT and TPOT, and the increase is strictly
> less than a factor of two, because the propagation term does not scale.

**Fails, on one of six per-request TPOT comparisons.**

The clause names no particular pair, and the frozen cell table carries three
rates, so both halvings are in scope. Disclosure of a harness correction: the
first implementation scored only the 400 to 200 halving. Reviewing it against
the frozen wording after the cells had run, the check was widened to both
halvings, which is stricter, and the widened check is what fails. The narrower
check would have passed. The record shows the widened one.

| halving | matched steps | makespan ratio | violations |
|---|---|---|---|
| 400G to 200G | 19 | 1.0441 to 1.4760 | none |
| 200G to 100G | 14 | 1.0845 to 1.6450 | r01 TPOT ratio 2.6281 |

Every step makespan and every TTFT obeys the bound on both halvings. TTFT
ratios run 1.2826 to 1.4760 and 1.3614 to 1.6450. Five of six TPOT ratios obey
it. The sixth does not, and the cause is in the step trace rather than in the
network model:

```
a-ep8-200g  step 4  0.497 ms  [r00 decode, r01 prefill 15]
a-ep8-200g  step 5  0.234 ms  [r00 decode, r01 decode]
a-ep8-200g  step 6  0.232 ms  [r00 decode, r01 decode]

a-ep8-100g  step 2  0.800 ms  [r00 decode, r01 prefill 15]
a-ep8-100g  step 3  0.272 ms  [r00 decode, r01 decode]
a-ep8-100g  step 4  0.952 ms  [r00 decode, r01 decode, r02 prefill 18]
```

r01 emits three tokens, so its TPOT is the mean of exactly two intervals. At
100 Gbit/s the slower fabric delays r00's decode enough that r02's admission and
its 18-token prefill fall between r01's second and third tokens. One of r01's
two intervals therefore changes from a two-token decode step to a twenty-token
prefill step. The 2.628 is a co-scheduling reallocation, not a serialization
effect, and the "less than two" argument is a serialization argument that does
not apply to a metric whose intervals can change composition.

This is not a defect in the model. It is a defect in the registered relation:
the clause asserted a serialization bound over a quantity that a closed loop
does not hold composition-fixed. That undemonstrated part is **PLAY-15**.

> **C5.3** Narrowing the expert-parallel world from 8 to 4 strictly decreases
> total routed MoE bytes for the identical token set.

**Passes.** 50,802,688 bytes at width 8, 29,306,880 at width 4, a ratio of
0.5769. The uniform-selection prediction written down before the run was 0.566.

> **C5.4** Narrowing the expert-parallel world from 8 to 4 strictly increases
> the per-step compute service, with the ratio expected in `[1.35, 1.75]`.

**Passes.** 153,186,600 ps against 99,250,537 ps, a ratio of **1.5434**, inside
the frozen band whose midpoint prediction was 1.667 diluted by the unsharded
embedding and head weights.

The two together are the interesting part: narrowing the expert-parallel world
cuts routed bytes by 42 percent and raises compute by 54 percent, and on this
workload compute wins. Decode step makespan moves from 0.204 to 0.254 ms and
r00's TPOT from 224.8 to 266.5 us. This is the kind of comparison the study
supports, and it is a comparison no single-component study in this repository
could have made.

## Evidence classes, kept separate

Counts from different classes are not summed.

- **Fatal-unscored guards:** 10 declared, 10 held, run not void.
- **Scored exact-oracle relations:** **13 of 13**. C1.1, C1.2, C1.3, C2.1, C2.2,
  C2.3, C2.4, C3.2, C3.5, C3.6, C3.7, C3.8, C4.1.
- **Scored behavioral relations:** **3 of 4**. C5.1, C5.3, C5.4 pass; C5.2 fails.
- **Reported, not scored:** C4.2, the error budget, and the three post-specified
  diagnostics below.

### Genuine-risk analysis

Every scored relation above is genuine risk, and the entailment analysis frozen
before the run holds up:

- C2.3 is not entailed by C2.2. C2.2 compares the independent recomputation
  against an in-memory library table; C2.3 compares it against the bytes written
  into the artifacts the backend consumed. A renderer defect would separate
  them.
- C2.4 is not entailed by C2.2 or C2.3. A uniform replication defect keeps every
  per-request table self-consistent while adding peer-sourced sends, which is
  precisely how the earlier 8x defect survived an aggregate check. C2.4 is
  scored on the executed artifacts.
- C3.5 and C3.6 are not entailed by C3.2. C3.2 checks each interval in
  isolation; C3.5 and C3.6 check that the chain of intervals starts at the
  declared arrival and ends at the right token.
- C3.8 is not entailed by C3.1 or C3.4. Those conserve the makespan without tying
  the compute share to the provider's separately published number.

C3.3 and C3.4 are configuration-forced and by-construction respectively, so they
are fatal-unscored and were classified that way before the run, not after.

### Post-specified diagnostics

These were not frozen. They are reported and never scored.

1. **S1 recomputed for the tokens actually scheduled.** The frozen S1 literal was
   derived from a 54-token first prefill step, taken from the earlier
   exploratory request set. Arrival gating admits one request at a time, so the
   realized first prefill step carried only r00's 19 prompt tokens, and this
   study's own capture has 52 case A prompt tokens rather than 54. Applying the
   same first-principles rule to 19 tokens gives `[1,867,776, 13,074,432]`
   bytes, and the measured 8,859,648 sits inside that too. The mean number of
   distinct remote destinations per token-layer is **4.743**, against the
   uniform-selection prediction of 4.931 written down before the run. The frozen
   guard held, but it held for a token count it did not describe, and saying so
   is the point of stating bounds before reading digits.
2. **Ideal-network fixed term.** Exactly 2,000,000 ps per collective artifact,
   extracted per artifact from two rates with no averaging, as described under
   C5.1.
3. **Schedule divergence under a closed loop.** The same three requests with the
   same declared arrivals produce 41, 39 and 36 scheduler steps at 400, 200 and
   100 Gbit/s, and only 19 and 14 scheduling compositions are shared with the
   400 Gbit/s cell. The simulated network speed feeds back into the framework's
   own batching decisions. This is the correct behavior for a closed-loop
   simulator, and it is why every cross-cell relation here is evaluated over
   matched compositions rather than over matched step ordinals.

## Error budget

The frozen budget named seven unmodeled mechanisms. The run makes three of them
quantitative. A case A decode step at 400 Gbit/s costs **0.205 ms** simulated:
0.099 ms of modeled compute and 0.106 ms of modeled network, the latter being 48
collectives of 2.000 us propagation plus their serialization.

| # | mechanism | modeled | plausible real | effect on a decode step |
|---|---|---|---|---|
| 1 | fixed per-step cost | 0 ps (`initiation_delay_ps = 0`, profile `ideal`) | 0.3 to 3 ms of launch, scheduling and sampling | +0.3 to +3 ms, i.e. 1.5x to 15x the whole simulated step |
| 2 | collective latency floor | 2.000 us per collective, measured exactly above | 15 to 30 us for a small NCCL all-to-all over 8 peers | 48 collectives, so 0.72 to 1.44 ms against a modeled 0.106 ms, 7x to 14x |
| 3 | compute calibration | roofline at a flat 0.7 derate, COMP-1 open | 0.6 to 0.85 of peak for these shapes | plus or minus 30 percent on 0.099 ms |
| 4 | GPU envelope | `GPU_ENVELOPES["b100"]`, 8.0e12 bytes/s, taken by default when `gpu=` is omitted | an H100 reader assumes 3.35e12 bytes/s | compute understated 2.4x for that reader |
| 5 | ideal network profile | `rnic-nn-fluid`, no congestion control, no switch queueing, no incast | repository bar puts a physical profile within 2x on short flows | up to 2x on the network term under load |
| 6 | traffic coverage | captured MoE dispatch and combine only | plus tensor-parallel all-reduces, KV movement, weight traffic | understates network, magnitude not bounded here |
| 7 | expert compute granularity | all resident expert weights read every step, no capacity factor or grouped-GEMM tail | per-expert kernels with padding | ambiguous direction, not bounded here |

Composing items 1 to 3: a real decode step for this model at this parallelism
would plausibly cost 1.1 to 4.5 ms against the simulated 0.205 ms, so the
simulation is optimistic by roughly **5x to 22x** on per-request decode latency.
The frozen budget predicted "roughly one order of magnitude, plausibly between
3x and 30x" before the run, and the measured composition lands inside it. The
implied 3,292 to 4,786 tokens per second per request that the study reports is
therefore a modeled quantity and not a prediction of anything a real deployment
would produce.

### What this study supports

- Conservation and identity claims: token sequences, stop conditions, request
  identity, per-request and per-token byte attribution, and the exact
  decomposition of TTFT and TPOT.
- Relative sweeps in which exactly one parameter moves: the bandwidth ladder and
  the expert-parallel width change, evaluated over matched scheduling
  compositions.
- The shape of the decomposition: which component dominates a request's latency
  and how that share moves with a parameter. For r00 at 400 Gbit/s the fabric
  carries 73 percent of TTFT; at 100 Gbit/s it carries 89 percent.
- The mechanism claim: that a per-request replay through a real scheduler, a
  simulated executor and a packet-level network is coherent end to end, with
  nothing lost, duplicated or unattributed.

### What this study does not support

- Any absolute TTFT or TPOT prediction for real hardware. The error budget above
  gives the reason and the magnitude.
- SLO attainment against a real service-level objective, or absolute goodput.
- Any comparison against another simulator's or a real deployment's absolute
  numbers.
- Any claim about a GPU other than the modeled B100 envelope. A reader who
  assumes H100 is reading numbers that are optimistic by a further 2.4x on the
  compute term.
- Per-request time attribution finer than a step. The packet-level sink
  publishes one whole-step makespan and no per-request endpoint, so every
  co-scheduled request is charged the whole step. That is stated, not hidden,
  and it is the same boundary `per_request_fidelity_v1` drew.

## Library change

`simllm.backends.step_attribution` is the minimum change that made the chain
runnable end to end. Before it, the packet-level sink published a whole-step
makespan and nothing else, while the only reducer that produces the
seven-component attribution ran on the analytic device runtime. No study could
carry a per-request latency claim through the packet-level path.

The new reducer is a read-only projection over values that sink already
published. Each executed artifact contributes its composed service to exactly
one component, collective when it ran on the fabric and kernel when it did not,
so the partition is complete and disjoint by construction. It refuses a run
whose locality reports NVLink bytes, because a mixed artifact's composed service
is a maximum over two resources and there is no evidence for picking one.

One gap is recorded here rather than claimed: `AdditiveVisitTotals` is a work
sum over `QueueVisit` records, and the packet-level sink produces no
`QueueVisit`. The reducer leaves that field empty rather than fabricating it
from wall-clock services. No frozen clause depended on it, so no ID is
registered for it, and the freeze said so before the run.

## Registered IDs

**One**, and only because a registered acceptance clause went undemonstrated:

- **PLAY-15**, for the part of C5.2 the run did not demonstrate: the "strictly
  less than a factor of two" bound over per-request TPOT when the link rate
  changes which requests share a step.

Nothing else is registered. The persistent-sink and state-preserving-artifact
paths that would raise the practical scale ceiling, and the reachability of the
seven-component attribution from inside the sink rather than from a study-driven
reducer, are recorded in this report and in the module doc as narrative, because
no clause claimed either.

## Contradiction sweep

Hits are reported here rather than edited, since this study did not register a
clause about any of them.

1. `README.md:17` reads "SimLLM predicts the serving performance (TTFT, TPOT,
   goodput, SLO attainment) of large LLM deployments **before you buy or
   reserve the hardware**". This study is the first registered evidence about
   that sentence, and it does not carry it. What the chain demonstrably does is
   conserve and decompose per-request TTFT and TPOT and rank configurations
   against each other. Absolute prediction is 5x to 22x optimistic on decode
   latency by the budget above, and goodput and SLO attainment are not
   demonstrated at all here. The sentence states an aspiration in the present
   tense.
2. `README.md:19` reads "with a packet-level network underneath rather than a
   `bytes / bandwidth` estimate". True of the repository's capability, but this
   study ran `rnic-nn-fluid`, the explicit ideal baseline, whose service was
   measured here to be exactly 2.000 us of propagation plus bytes at the
   declared rate. A reader could take the sentence to mean every run is
   congestion-aware; this one deliberately was not, and the ideal profile is
   what makes C5.1 exact.
3. Neither `README.md` nor `docs/README_PRO.md` says that
   `GPU_ENVELOPES["b100"]` is what a caller gets when `gpu=` is omitted. Both
   name the 8 by 8 B100 reference configuration, and `docs/README_PRO.md:190`
   correctly labels the A100 and H100 parameters as bootstrap seeds, but the
   silent default is the specific hazard this study's error budget item 4
   quantifies at 2.4x.
4. `docs/architecture.md:520` lists "per-request TTFT/TPOT/queueing delay" among
   the sim-native metrics. That is now true of the packet-level path only
   through `simllm.backends.step_attribution`; before this change the
   packet-level sink published a whole-step makespan and no per-request row.

## Reproduction

```
python examples/end_to_end_replay_v1/run_study.py \
    --cache-dir <hugging-face-cache-root> \
    --htsim-rnic <path-to-htsim_rnic> \
    --run-dir <writable-output-directory>
```

`--check-only` validates every frozen input and produces no artifacts. Capture
and replay need the environment carrying torch, transformers and vLLM 0.26.0,
with `PYTHONPATH` at this worktree. The driver runs the capture and then each
cell as an isolated child stage; the recorded run executed the five cells
concurrently through `--internal cell:<name>` and then `--internal summarize`,
which produces the identical per-cell artifacts because each cell is
independent and deterministic.
