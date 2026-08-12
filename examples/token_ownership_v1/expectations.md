# Token ownership v1 expectations

This is the expectations-only record for TRAF-25. It freezes the token
population, source-ownership decision, byte relations, native profile sweep
and blast-radius audit before implementation or any result-producing run.

## Decision and claim boundary

One `StepRecord` remains one engine scheduler step. Every scheduled token in a
captured routed supply belongs to one declared `engine_rank` in the EP group.
The other EP ranks hold zero scheduled tokens in this projected step. They
still own experts and return combine traffic, but they do not source copies of
the captured tokens.

This is an isolated one-engine projection of a realizable DP times EP
deployment state in which the peer engines are idle for the modeled step. It
is not a full-group throughput model and does not claim that peer engines are
normally idle. TRAF-26 owns full-group population: it must supply captured or
sampled peer workloads and route each peer independently. Replaying this
engine's routing table on every peer is not acceptable because it makes all
peers target the same hot experts in the same order and manufactures
correlated incast.

The captured path will require its source declaration. The source-multiplying
behavior is a correctness defect, not a compatibility mode, so there is no
switch that preserves it. The analytic no-supply path uses the first EP rank
as the documented single engine and emits only that source's uniform remote
share. Dense, zero-token and EP-width-one paths remain explicit empty bypasses.

The decision-relevant relation separates population work from critical-rank
work. At EP width 8, total directed bytes must fall by about the population
factor, while peak per-rank step egress must fall by only about two. If peak
egress also fell by eight, a simple byte division would be hiding the source
shape. If total bytes did not fall by about eight, captured tokens would still
be duplicated.

## Pre-freeze source audit

The evidence was authored against SimLLM commit
`cede92930a469bd0be2f2c588866885c9e0e3618`. The observed htsim gitlink was
`fc4400e4ca619223481536632074045cb6af2756`; this is provenance only and is
not an equality requirement on a future live pin.

- `simllm/core/step.py:109-111` defines `total_new_tokens` as the sum over one
  record's scheduled requests.
- `simllm/traffic/step_comm.py:1-5` says the module renders the collective
  traffic of that engine step.
- `simllm/traffic/step_comm.py:515-543` currently nests every scheduled token
  under every EP source. Lines 628-634 make the same population mistake in
  the uniform path by expanding one per-pair scalar over all ordered pairs.
- `simllm/traffic/step_comm.py:385-443` selects exactly the captured token
  slices named by the record. The selected count is required to equal
  `record.total_new_tokens`.
- `simllm/traffic/step_comm.py:204-211` sizes each TP activation from
  `record.total_new_tokens`, so TP already describes one engine's token
  population.

The vLLM 0.26.0 source audit supplied with TRAF-25 establishes the external
ownership rule:

- `vllm/distributed/parallel_state.py:1896-1903` and
  `vllm/model_executor/layers/fused_moe/config.py:1119,1204-1237` form EP from
  DP, PCP and TP dimensions.
- `vllm/v1/engine/core.py:153,1253-1299` gives each DP rank a separate
  `EngineCore` and scheduler.
- `vllm/model_executor/layers/fused_moe/all2all.py:115-127` identifies the
  local activation row count with the local group rank.
- `vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_ht.py:131-171`
  derives dispatch layout from local top-k rows.
- `vllm/model_executor/layers/fused_moe/all2all_utils.py:139-156` emits no
  all-to-all when tokens are TP-replicated rather than DP-owned.

The external Granite inputs are supplied through `SIMLLM_MOE_E2E_ROOT`. They
are observed inputs, not tracked paths or live-pin literals:

- `capture/granite-greedy.jsonl`: 120 LF-terminated rows, SHA-256
  `5f55fccc265ee5519430a0f73d6631e49aa547ec07fcb81034e9fc2b4d9fead6`;
- `replay-400g/steps.jsonl`: SHA-256
  `824cd9557293328bb42b593ac893b6a067302e545b087c9219195ccb8031d755`;
- `replay-400g/routed-experts.json`: SHA-256
  `24e986e989e21f1bfe7e758d4470928c82c3bbaec06072a839743b9b17d7cf5f`;
- the archived defective step-0 GOAL: 334,432 bytes, 2,688 sends and
  207,499,264 directed bytes, SHA-256
  `08a0403af66ff8a9d6b18f93afd15ae0bc925cc85555acf8a0593438a3d7bc92`.

The backend rules reused by the native sweep were audited in the accepted M1
and M5 studies. `rnic-nn-fluid` has 2,000,000 ps propagation and full-rate
serialization of 20 ps/byte at 400 Gbit/s. `rnic-nn` adds packetization and
store-and-forward service, so the packet result must not beat the fluid result
for the identical GOAL and topology.

## Fixed workload and source convention

Granite prefill step 0 schedules 54 tokens across requests `r0`, `r1` and
`r2`. Model geometry is 24 MoE layers, hidden width 1,024, dtype width two,
32 experts and top-k eight. Expert owner is `expert_id % W`. The concrete
single-engine source is rank 0 for every EP width in the sweep.

For token `t`, layer `l`, home rank `h = 0` and expert-owner function `o`,
the independent byte projection is:

```text
destinations(t,l) = {o(e) for e in selected_experts(t,l)}
dispatch_hops(t,l) = # {d in destinations(t,l) where d != h}
dispatch_bytes(h,d,t,l) = 2048 * indicator(d != h and d in destinations(t,l))
combine_bytes(d,h,t,l) = dispatch_bytes(h,d,t,l)
```

Destination deduplication occurs before bytes are counted. Gate weights do
not affect this activation-vector projection.

## Frozen EP-width byte sweep

The legacy columns are raw observations of the source-multiplying renderer.
The corrected columns were calculated from the captured expert IDs and the
declared rank-0 source before implementation. Peak egress means the largest
sum of bytes sourced by one rank across all 48 dispatch and combine phases.

| EP width | Legacy total bytes | Corrected total bytes | Legacy peak-rank egress | Corrected peak-rank egress |
|---:|---:|---:|---:|---:|
| 2 | 10,612,736 | 5,304,320 | 5,306,368 | 2,652,160 |
| 4 | 58,773,504 | 14,594,048 | 14,792,704 | 7,297,024 |
| 8 | 207,499,264 | 25,563,136 | 27,060,224 | 12,781,568 |

At width 8 the corrected 25,563,136 bytes are 12,482 remote vector hops.
The independent upper bound is
`54 * 8 * 24 * 2 = 20,736` dispatch-plus-combine hops. The defective renderer
emits 101,318 hops and violates that bound.

The often-quoted `207,499,264 / 8 = 25,937,408` bytes is the arithmetic mean
over all eight possible home-rank choices. It is not the byte count of a
concrete engine rank because each home choice omits a different set of local
destinations. Rank 0 is the declared deployment choice, so 25,563,136 is the
frozen concrete oracle. The token population changes by exactly eight; the
remote-byte ratio is 8.117 because locality is source-dependent.

The width-8 corrected per-request totals across dispatch and combine are:

| Request | Scheduled tokens | Corrected bytes |
|---|---:|---:|
| `r0` | 22 | 10,403,840 |
| `r1` | 12 | 5,701,632 |
| `r2` | 20 | 9,457,664 |

These values sum to 25,563,136. Each request may be attributed only to rank 0
on dispatch and to the exact transpose on combine.

### TRAF-B1: raw population response

Before exact totals, conservation or hop bounds are checked, compare raw
legacy and corrected total bytes. The legacy-to-corrected ratio must be in
`[1.95, 2.05]`, `[3.9, 4.2]` and `[7.9, 8.3]` for widths 2, 4 and 8,
respectively. The ratio must increase strictly with width. These are three
scored instances.

### TRAF-B2: raw critical-rank response

Before exact egress or per-request oracles, compare raw peak-rank step egress.
The legacy-to-corrected ratio must be in `[1.95, 2.05]`, `[1.95, 2.15]` and
`[1.95, 2.30]` for widths 2, 4 and 8. At width 8 it must be less than one
third of the total-byte ratio. These are three scored instances. This family
is the primary guard against implementing a blind division by EP width.

## Native makespan sweep

The result-producing study uses both `rnic-nn-fluid` and packet-level
`rnic-nn`, with an all-remote eight-rank topology. It records the archived
defective GOAL and the corrected production renderer separately, using the
same backend binaries, link rate and topology. It also sweeps corrected fluid
cells over EP widths 2, 4 and 8 and link rates 200 and 400 Gbit/s. Represented
compute is fixed at 99,360,000 ps, split over 24 layers.

For the corrected star-shaped traffic, full-rate serialization plus one
2,000,000 ps propagation per phase gives the preregistered fluid points:

| EP width | Corrected JCT at 200 Gbit/s, ps | Corrected JCT at 400 Gbit/s, ps |
|---:|---:|---:|
| 2 | 407,532,800 | 301,446,400 |
| 4 | 779,121,920 | 487,240,960 |
| 8 | 1,217,885,440 | 706,622,720 |

Whole-ps backend rounding may add at most one picosecond per positive fluid
flow. The acceptance band for each point is therefore the table value through
that value plus its positive-flow count. Halving bandwidth must increase only
the serialization term; at width 8 the signed increase is 511,262,720 ps,
subject to the same rounding envelope.

The previously published width-8 400 Gbit/s fluid step was 974,838,253 ps.
The corrected result must be lower, in `[680,000,000, 830,000,000]` ps, and
its ratio to that published result must be in `[0.69, 0.85]`. The same native
binary rerun of the archived defective GOAL is diagnostic provenance, not a
compatibility requirement.

For packet-level `rnic-nn` at width 8 and 400 Gbit/s, the corrected makespan
must be below the archived defective-GOAL rerun and have ratio `[0.60, 0.85]`.
It must be at least the corrected fluid makespan and no more than
850,000,000 ps. This is one scored packet instance.

### TRAF-B3: live makespan response

The width-8 fluid 200 and 400 Gbit/s cells and the width-8 packet 400 Gbit/s
cell are three scored instances. Raw JCTs and signed changes are evaluated
before exact byte tables, exact fluid points, provenance hashes or quiescence
checks. Controlled one-step TTFT equals the step JCT. A second equal decode
step supplies TPOT reachability, but fixed-step algebra entails equality with
its JCT and therefore TTFT and TPOT do not add scored instances.

## Physical sanity bounds

At width 8 and 400 Gbit/s, corrected peak egress gives the mandatory floor:

```text
12,781,568 bytes * 8 / 400e9 = 255,631,360 ps
```

No measured makespan may be below that floor. A conservative packet ceiling
serializes all 12,482 remote vectors, charges 2,000,000 ps propagation to each
one separately, and adds 99,360,000 ps compute:

```text
25,563,136 * 8 / 400e9 + 12,482 * 2,000,000 + 99,360,000
    = 25,574,622,720 ps
```

The fluid point must also satisfy the much tighter work-conserving star bound
of 706,622,720 ps plus its rounding envelope. At 200 Gbit/s the serialization
term must double while compute and propagation remain fixed. These network,
compute and bandwidth-scaling views are separate physical-sanity angles.

## Exact and structural guards

After scoring the raw relations, the study and focused tests require:

- exact totals and peak egress for every EP-width row above;
- exact width-8 per-request bytes and a sum equal to the aggregate;
- per-layer source conservation against an independent walk of captured
  token destinations;
- renderer hops no greater than `total_new_tokens * top_k * num_layers * 2`;
- exact agreement between renderer hops and the independent routed-token
  projection used by the compute-side workload description;
- one declared source rank on every dispatch row and exact combine transpose;
- TP activation payload still based on the same 54-token population;
- dense, drain and EP-width-one bypasses remain empty;
- malformed or out-of-group engine ranks fail before rendering;
- backend quiescence, exact profile identity and separate observed versus
  authored-against provenance.

These checks are fatal-unscored. They are conservation, by-construction,
configuration or exact-oracle evidence and never increase the behavioral
denominator.

## Blast-radius refreeze

Historical expectations that freeze source-multiplied physical bytes are
withdrawn as acceptance oracles. Their original files remain chronological
records. This study refreezes their affected claim surfaces before the fix:

- `framework_oracle_v1`: changed-byte comparisons remain valid because each
  request already declares one `source_rank`; its zero changed-byte outcomes
  are rerun only if the expensive external framework environment remains
  available.
- `nvlink_locality_v1`: all routed total, locality split, flow-count, GOAL and
  JCT literals must be recomputed from single-engine traffic. Locality
  partition and node-span directions remain the accepted relations.
- `per_request_fidelity_v1`: synthetic and Granite physical tables, request
  totals, permutation errors, GOAL size and digest are refrozen from one
  declared source. Aggregate-preserving permutation rejection remains the
  accepted relation.
- `preplay_validation_v1`: the two-rank emitted pair table now has one
  dispatch direction and its combine transpose. The independent raw-trace
  oracle must use the declared source. Two-rank fluid JCT may remain equal
  because the removed reverse flow used an independent port, but this is an
  observation to verify, not a preserved byte oracle.
- `routed_supply_v1`: captured and uniform two-rank traffic now has one
  dispatch direction and its transpose. Physical hashes and flow counts are
  invalid. Exact routed fluid JCT is expected to remain unchanged because its
  critical pair is unchanged; this must be rerun.
- `routing_lifetime_v1`: legacy-object and packed-arena projections must
  remain mutually identical after both declare the same engine rank. Equality
  to the archived defective 334,432-byte GOAL is removed, and the corrected
  GOAL is accepted only as a labelled post-specified artifact.

`RESULTS.md` must list every published numeric value that changes with old and
new values. It must say plainly that the old source-multiplied values were
wrong. Unavailable expensive reruns are reported as deliberate omissions and
never counted as passes.

## Evidence classes and entailment

The scored headline has three genuine-risk families and nine instances:
TRAF-B1 has three width cells, TRAF-B2 has three width cells, and TRAF-B3 has
three native cells. Each is read from raw renderer or backend observations
before an exact oracle can entail it. A competent implementation could retain
the source loop, divide bytes without changing the source graph, choose a
different hidden home convention, lose request attribution, or produce packet
timing outside the registered response.

Exact byte rows, per-request sums, hop conservation, transpose, source
identity, strict input hashes, quiescence, dense and drain bypasses, run
configuration, artifact digests, unit tests and native executable status are
separate fatal-unscored evidence classes.

## Registered command and dry run

The result-producing command is:

```text
.venv/bin/python examples/token_ownership_v1/run_study.py \
  --source-root "$SIMLLM_MOE_E2E_ROOT" \
  --out "$SIMLLM_TOKEN_OWNERSHIP_RUN_ROOT"
```

Before the expectations commit, this exact CLI is run with `--check-only`.
Check-only parses the production arguments and validates only frozen literal
shapes and arithmetic. It imports no SimLLM implementation, reads no external
input, invokes no native executable and creates no artifact.
