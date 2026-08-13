# Dispatch sequence timing requalification expectations

Date: 2026-08-13

This is an expectations-only freeze for TRAF-22. It is a **fresh
qualification**, not a repair of an earlier one. It was written before the
sequenced timing was requalified, before the held-out fixture was rendered or
executed, and before any new measurement existed.

## Why a fresh qualification is possible now

The 2026-08-12 ownership refreeze under `examples/dispatch_sequence_v1` is
**void**. Its record stands unmodified. Its fatal floor added the dispatch and
combine endpoint loads as if they were two globally serial link loads. Single
home rank traffic is not that: dispatch leaves the engine rank while combine
returns to it, and the modeled port is full duplex, so the two loads occupy
opposite directions of the same endpoint and overlap.

CORE-41 landed the endpoint charge and recorded the consequence. For the
synthetic fixture the home endpoint carries 16,384 bytes of egress and 16,384
bytes of ingress, so the correct single-endpoint payload floor is
`max(egress, ingress) * 8 / rate`: **655,360 ps at 200 Gbit/s** and **327,680 ps
at 400 Gbit/s**, exactly half the summed floors the void freeze used.

This freeze adopts that endpoint arithmetic, replaces the void run's byte-only
behavioral surrogates with bounds derived from the rendered endpoint loads and
the packet backend's full-envelope calendar, and adds a held-out routing shape
and payload that has never been rendered or executed.

Nothing here unvoids, rescores or reinterprets the earlier run. Its raw
observations remain retained findings of a void run and are never used as a
pass in this study.

## Chronology this freeze commits to

| Event | Requirement |
|---|---|
| Void run record | unmodified, retained as chronology |
| This expectations-only freeze | precedes the requalification runner, the held-out fixture and every new measurement |
| Pairwise frontier documentation alignment | doc-only, lands before the first run, changes no rendered byte |
| First result-producing run | after both of the above |

`simllm/traffic/patterns.py` documents `pairwise_all_to_allv`'s source-only
frontier as the rank's last send while the implementation retains its first
send. `simllm/traffic/collective_plan.py` already documents the implemented
rule. TRAF-22 requires the two statements to agree before requalification. The
alignment is a documentation correction, so the Granite aggregate GOAL digest
below is the fatal evidence that no rendered byte moved.

## Frozen physical model and derivations

All native cells use zero compute in the synthetic arms, zero propagation, and
one endpoint serializer per direction per rank. The packet profile uses a
4,096-byte maximum wire packet with a 64-byte data header, so one packet
carries at most 4,032 payload bytes and reserves a full 4,096-byte calendar
envelope.

For a rendered message set, define per endpoint `e`:

```text
load(e)      = max(egress_bytes(e), ingress_bytes(e))
floor_ps     = ceil(max_e load(e) * 8 * 1e12 / rate_bps)
envelope(m)  = 4096 * ceil(payload(m) / 4032)     for the packet profile
envelope(m)  = payload(m)                          for the fluid profile
ceiling_ps   = 2 * sum_m envelope(m) * 8 * 1e12 / rate_bps + 1000 * message_count
```

`floor_ps` is the full-duplex single-endpoint floor no schedule can beat.
`ceiling_ps` serializes every envelope at a source serializer and again at a
destination serializer and adds one nanosecond of GOAL quantization per
message, so no schedule can exceed it. Both are computed by the runner from
the actually rendered messages, not from a hand-copied literal.

## Frozen fixtures

### Primary fixture (retained corrected shape)

Four ranks `(0, 1, 2, 3)`, engine rank 0, one MoE layer, four prefill tokens,
`top_k = 2`, hidden size 1,024 and two dtype bytes, so one routed vector is
2,048 bytes. The observed route table is `((3, 1), (2, 1), (3, 2), (1, 3))`.

| Quantity | Value |
|---|---:|
| Dispatch ordered pairs | `(0,1,6144)`, `(0,2,4096)`, `(0,3,6144)` |
| Combine ordered pairs | exact transpose |
| Total directed bytes | 32,768 |
| Remote vector hops | 16 |
| Independent hop ceiling `tokens*top_k*layers*phases` | 16 |
| Aggregate messages | 6 |
| Per-expert-group messages | 6 |
| Per-token messages | 16 |
| Home endpoint load | 16,384 bytes |
| Floor at 200 Gbit/s | 655,360 ps |
| Floor at 400 Gbit/s | 327,680 ps |

### Held-out fixture (declared here for the first time)

This routing shape and this payload have never been rendered, compiled or
executed by any study in this repository. Five ranks are not used; the group
stays `(0, 1, 2, 3)` with engine rank 0 so the comparison isolates shape and
payload. Hidden size is 512 with two dtype bytes, so one routed vector is
1,024 bytes. Six prefill tokens with `top_k = 2` use the route table
`((1, 2), (1, 3), (2, 3), (1, 2), (3, 1), (2, 1))`.

| Quantity | Value |
|---|---:|
| Dispatch ordered pairs | `(0,1,5120)`, `(0,2,4096)`, `(0,3,3072)` |
| Combine ordered pairs | exact transpose |
| Total directed bytes | 24,576 |
| Remote vector hops | 24 |
| Independent hop ceiling `tokens*top_k*layers*phases` | 24 |
| Aggregate messages | 6 |
| Per-expert-group messages | 6 |
| Per-token messages | 24 |
| Home endpoint load | 12,288 bytes |
| Floor at 200 Gbit/s | 491,520 ps |
| Floor at 400 Gbit/s | 245,760 ps |

### Granite scale point

The corrected 54-token, 24-layer, EP-width-eight Granite step from the
2026-08-12 replay, with per-layer calc 4,139 ns on the engine rank. Accepted
TRAF-25 arithmetic supplies the pre-run scale oracles:

| Grouping | Messages | Directed bytes |
|---|---:|---:|
| Aggregate | 336 | 25,563,136 |
| Per-expert-group | 1,008 | 25,563,136 |
| Per-token | 12,482 | 25,563,136 |

Peak per-rank egress is 12,781,568 bytes and the engine rank's ingress is the
same, so the full-duplex endpoint floor is 255,631,360 ps at 400 Gbit/s and
511,262,720 ps at 200 Gbit/s. Represented compute is `24 * 4,139 ns =
99,336,000 ps` and is rate independent because the rendered chain places calc
serially before each layer's collectives on the engine rank. The step floors
are therefore 354,967,360 ps at 400 Gbit/s and 610,598,720 ps at 200 Gbit/s.

The independent Granite hop ceiling is `54 * 8 * 24 * 2 = 20,736`.

## Scored behavioral relations

Scored relations are evaluated from raw native completions **before** any exact
sequence, pair, request, hop, quiescence, floor or ceiling guard runs. No fatal
guard contributes to a behavioral numerator or denominator. Five families and
thirty-four instances are registered.

**R1, packet granularity cost from the envelope calendar (8 instances).** The
per-token rendering splits the same bytes into more messages, and the packet
profile reserves a full envelope per packet, so the home endpoint's egress
calendar grows by
`excess = envelope_bytes(per-token dispatch) - envelope_bytes(comparator dispatch)`.
For each fixture, rate and comparator in `(per-expert-group, aggregate)`:

```text
packet(per-token) - packet(comparator) in [excess*8/rate, 4*excess*8/rate]
```

The lower edge charges the excess once at the home egress serializer. The upper
edge allows the excess at the home egress serializer, at the home ingress
serializer, and twice more for peer-side envelope growth and quantization. For
the primary fixture `excess` is 8,192 bytes, giving `[327,680, 1,310,720]` ps at
200 Gbit/s and `[163,840, 655,360]` ps at 400 Gbit/s. For the held-out fixture
`excess` is 28,672 bytes, giving `[1,146,880, 4,587,520]` ps at 200 Gbit/s and
`[573,440, 2,293,760]` ps at 400 Gbit/s. The runner recomputes `excess` from the
rendered messages; the literals above are the frozen expected values.

**R2, inverse-rate scaling of synthetic completions (12 instances).** Every
term in the synthetic arms is serialization. Compute is zero, propagation is
zero and there is no fixed per-flow latency, so halving the rate must double
every completion. GOAL quantizes to whole nanoseconds and at most six
quantized events sit on any of these critical paths, so for each fixture,
grouping and profile:

```text
abs(completion(200 Gbit/s) - 2 * completion(400 Gbit/s)) <= 6,000 ps
```

**R3, fluid granularity direction (4 instances).** On the fluid profile there is
no envelope quantum, but equal-sized per-token flows reach the peer receive
frontiers together and remove the short-return overlap that unequal aggregate
flows enjoy. For each fixture and rate:

```text
fluid(per-token) > fluid(aggregate)
```

No magnitude is registered for this family because the fair-share manifold does
not admit a first-principles magnitude.

**R4, fluid grouping insensitivity for equal message sets (4 instances).** The
aggregate and per-expert-group renderings of both fixtures carry the identical
ordered-pair table with the identical message count and differ only in issue
order within one source. On a fluid profile with no envelope quantum, for each
fixture and rate:

```text
abs(fluid(per-expert-group) - fluid(aggregate)) <= 2,000 ps
```

**R5, Granite rate scaling (6 instances).** This is the check the void run
registered but never executed. Represented compute is rate independent and
every other term is serialization, so for each Granite grouping and profile:

```text
(completion(200 Gbit/s) - 99,336,000) / (completion(400 Gbit/s) - 99,336,000)
    in [1.95, 2.05]
```

## Fatal guards, all unscored

A single violation makes the run void with findings and closes nothing. No
fatal guard is survivable in this study and none may be reported as a fraction.

- **F1 floor.** Every native cell is at or above its fixture's `floor_ps`.
- **F2 ceiling.** Every native cell is at or below its cell's `ceiling_ps`.
- **F3 pair equality.** For both fixtures and both sequenced groupings, the
  ordered-pair byte table aggregated by `(layer, phase, source, destination)`
  equals the aggregate renderer's table exactly, in the plan and in the
  rendered GOAL messages.
- **F4 request equality.** Per-request ordered-pair totals match exactly.
- **F5 hop ceiling.** Remote vector hops never exceed
  `tokens * top_k * layers * phases`, for both fixtures and for Granite.
- **F6 ownership.** Every dispatch message sources `engine_rank` and every
  combine message returns to `engine_rank`, at every grouping.
- **F7 conservation.** All three groupings emit identical total directed bytes,
  and combine is the exact transpose of dispatch with unchanged ordinals.
- **F8 quiescence.** Every native manifest reports `physical_quiescence`
  verified.
- **F9 input identity.** The Granite routing and step inputs match the
  authored-against digests
  `24e986e989e21f1bfe7e758d4470928c82c3bbaec06072a839743b9b17d7cf5f` and
  `824cd9557293328bb42b593ac893b6a067302e545b087c9219195ccb8031d755`.
- **F10 aggregate-default regression.** The Granite aggregate GOAL is 47,399
  bytes with SHA-256
  `6bb83366a3936bcf1e435cab008bb55c2966777528a9e8d885dd44d47f5a4943`, carries
  336 messages and 25,563,136 directed bytes, and its 400 Gbit/s completions
  reproduce the retained values 503,658,600 ps on the packet profile and
  489,235,306 ps on the fluid profile. This is the clause that the frontier
  documentation alignment moved no rendered byte and no accepted timing.
- **F11 cost limits.** Render plus compile within 30 seconds, peak traced
  Python memory within 1 GiB, GOAL text within 64 MiB, and each backend run
  within 60 seconds, at every Granite grouping.
- **F12 Granite scale oracles.** Message counts 336, 1,008 and 12,482, directed
  bytes 25,563,136 at every grouping, and peak per-rank egress 12,781,568.

Executable, gitlink and revision digests are recorded as observations. No
equality between an observed submodule pin and a frozen literal is asserted.

## Physical sanity statement before any digit is read

Primary fixture, 200 Gbit/s: no cell can finish before 655,360 ps, and no cell
can exceed 5,258,880 ps. At 400 Gbit/s the same bounds are 327,680 ps and
2,637,440 ps. Held-out fixture, 200 Gbit/s: no cell can finish before 491,520 ps
and none can exceed 7,888,320 ps; at 400 Gbit/s, 245,760 ps and 3,956,160 ps.
Granite, 400 Gbit/s: no cell can finish before 354,967,360 ps; at 200 Gbit/s,
before 610,598,720 ps. A cell outside its own bounds is a defect in the model,
the harness or the reading, and it voids the run.

Sitting inside a bound is not proof of correctness, which is why R2 and R5 check
the quantity that must scale with the primary one.

## Claim boundaries

The strict v2 trace observes the order framework dispatch returned. It does not
observe the order a fused kernel, NCCL or an RNIC posted bytes to the wire.
PLAY-14 retains that residual and this study does not upgrade it.

The Granite v1 tuple order is a Transformers reconstruction and is scale and
cost evidence only.

All cells are all-remote native profiles, so the analytic intra-node locality
service is not exercised here.

TRAF-26 retains real multi-engine population. Copying one engine's routing
table onto peer sources remains forbidden.

## Fatal and failed outcomes

A violated fatal guard voids the whole run: no behavioral fraction is
published, and TRAF-22 stays open with findings. If every fatal guard passes and
one or more scored relations miss, the run is failed, reports its scored
fraction, and TRAF-22 stays open. A missed scored band is never converted into a
fatal guard and is never refrozen after observation.

## Registered command and dry run

```bash
.venv/bin/python examples/dispatch_sequence_v2/run_study.py \
  --out "$SIMLLM_WAVE10_RUN_ROOT/dispatch_sequence_v2" \
  --granite-root "$SIMLLM_GRANITE_REPLAY_ROOT" \
  --htsim-rnic "$SIMLLM_HTSIM_RNIC" \
  --txt2bin "$SIMLLM_TXT2BIN"
```

Before this expectations-only commit the complete command is run with
`--check-only`. Check-only parses the CLI and validates only the frozen
registries and arithmetic above. It imports no SimLLM module, reads no external
artifact, creates no output directory, invokes no native tool and writes
nothing.
