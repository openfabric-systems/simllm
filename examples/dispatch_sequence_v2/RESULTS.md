# Dispatch sequence timing requalification results

## Outcome

The requalification **passed**. Every fatal guard held and all 34 registered
scored instances in all 5 families passed. TRAF-22 closes on this run.

This is a fresh qualification on a corrected floor, not a rescoring of the
2026-08-12 run under `examples/dispatch_sequence_v1`. That run remains void
and its record is unmodified.

| Evidence class | Result |
|---|---|
| Fatal guards (unscored) | 6 of 6 groups held; no violation |
| Scored behavioral relations | 34 of 34 instances, 5 of 5 families |
| Exact fixture and Granite oracles | all held |
| Physical floors and ceilings | every one of 36 cells inside its own bounds |

## Chronology

| Event | Commit |
|---|---|
| Void ownership refreeze (unchanged) | `82d3ab45ea47c811fa6db0d91ac8122e255fd62b` |
| Void run record (unchanged) | `62e08f7` |
| This expectations-only freeze | `68d8a8d` |
| Pairwise frontier documentation alignment | `2f61990` |
| Harness fix: read the manifest as a list | `5d98f44` |
| Observed revision of the completed run | `5d98f443ee813be7649302a76e058302edbd3442` |

The first complete attempt executed every backend cell and then raised
`AttributeError` while assembling its fatal-guard block, because the quiescence
row treated `HtsimRnicResult.manifest` as a mapping when it is a list of raw
manifest lines. No summary was written and no measured value was read before
the fix. The fix touched no frozen registry, bound, band, guard or evidence
count. The completed run then used a separate output directory.

The registered command passed `--check-only` before the expectations commit and
produced no artifacts.

| Provenance field | Observed value |
|---|---|
| htsim gitlink | `fc4400e4ca619223481536632074045cb6af2756` |
| `htsim_rnic` SHA-256 | `cfb5014a663791f7619fe33309114a74e82878de860c14fc8a723713501f027d` |
| `txt2bin` SHA-256 | `f3745f34ad86febe9c9eebef10aee5fae00b8865cb29943344fb75b0f142495b` |
| Routing input SHA-256 | matches the authored-against digest |
| Step input SHA-256 | matches the authored-against digest |

Executable, gitlink and revision digests are observations, not requirements on
a future checkout.

## What the corrected floor changed

The void run's floor charged the dispatch and the combine endpoint loads as two
globally serial link loads. The modeled port is full duplex, so the correct
single-endpoint payload floor is `max(egress, ingress) * 8 / rate`. For the
primary fixture the home endpoint carries 16,384 bytes in each direction, so
the floors are 655,360 ps at 200 Gbit/s and 327,680 ps at 400 Gbit/s, exactly
half the summed floors the void freeze used. The held-out fixture's home
endpoint carries 12,288 bytes in each direction, giving 491,520 ps and
245,760 ps.

Every one of the 24 synthetic cells and 12 Granite cells sits inside its own
floor and ceiling. The lowest synthetic cell is the held-out aggregate fluid
value of 431,082 ps at 400 Gbit/s against a 245,760 ps floor, and the highest
is the held-out per-token packet value of 3,650,000 ps against a 7,888,320 ps
ceiling.

## Raw synthetic observations

Zero compute, zero propagation, 4,096-byte maximum wire packet and 64-byte data
header on the packet profile. Every native manifest reported physical
quiescence verified.

| Fixture | Renderer | Profile | Gbit/s | Flows | Min FCT ps | Max FCT ps | Completion ps |
|---|---|---|---:|---:|---:|---:|---:|
| Primary | Aggregate | Fluid | 200 | 6 | 163,840 | 655,361 | 1,147,881 |
| Primary | Aggregate | Fluid | 400 | 6 | 81,920 | 327,681 | 574,441 |
| Primary | Aggregate | Packet | 200 | 6 | 491,520 | 1,070,080 | 1,890,280 |
| Primary | Aggregate | Packet | 400 | 6 | 245,760 | 535,040 | 945,640 |
| Primary | Expert group | Fluid | 200 | 6 | 163,840 | 655,361 | 1,149,000 |
| Primary | Expert group | Fluid | 400 | 6 | 81,920 | 327,681 | 576,000 |
| Primary | Expert group | Packet | 200 | 6 | 654,360 | 988,160 | 1,809,000 |
| Primary | Expert group | Packet | 400 | 6 | 326,680 | 494,080 | 905,000 |
| Primary | Per token | Fluid | 200 | 16 | 655,360 | 655,360 | 1,313,000 |
| Primary | Per token | Fluid | 400 | 16 | 327,680 | 327,680 | 658,000 |
| Primary | Per token | Packet | 200 | 16 | 248,320 | 1,395,200 | 2,544,000 |
| Primary | Per token | Packet | 400 | 16 | 124,160 | 697,600 | 1,273,000 |
| Held out | Aggregate | Fluid | 200 | 6 | 184,321 | 491,521 | 861,162 |
| Held out | Aggregate | Fluid | 400 | 6 | 92,161 | 245,761 | 431,082 |
| Held out | Aggregate | Packet | 200 | 6 | 327,680 | 824,320 | 1,480,680 |
| Held out | Aggregate | Packet | 400 | 6 | 163,840 | 412,160 | 740,840 |
| Held out | Expert group | Fluid | 200 | 6 | 184,321 | 491,521 | 863,000 |
| Held out | Expert group | Fluid | 400 | 6 | 92,161 | 245,761 | 433,000 |
| Held out | Expert group | Packet | 200 | 6 | 326,680 | 824,320 | 1,481,000 |
| Held out | Expert group | Packet | 400 | 6 | 162,840 | 412,160 | 741,000 |
| Held out | Per token | Fluid | 200 | 24 | 491,521 | 491,521 | 986,000 |
| Held out | Per token | Fluid | 400 | 24 | 245,761 | 245,761 | 494,000 |
| Held out | Per token | Packet | 200 | 24 | 207,360 | 2,009,600 | 3,650,000 |
| Held out | Per token | Packet | 400 | 24 | 103,680 | 1,004,800 | 1,826,000 |

Every primary-fixture cell reproduces the void run's retained raw value
exactly, at both rates and on both profiles. The two runs used the same htsim
binary digest, so this is a determinism check rather than independent
evidence, but it does establish that the frontier documentation alignment and
the collective-plan lowering default moved no rendered byte and no timing.

## Scored relations

**R1, packet granularity cost from the envelope calendar: 8 of 8.** The band is
`[excess*8/rate, 4*excess*8/rate]` where `excess` is the growth of the home
endpoint's dispatch envelope calendar when the same bytes are split per token.

| Fixture | Gbit/s | Comparison | Raw delta ps | Band ps |
|---|---:|---|---:|---|
| Primary | 200 | per token minus expert group | 735,000 | [327,680, 1,310,720] |
| Primary | 200 | per token minus aggregate | 653,720 | [327,680, 1,310,720] |
| Primary | 400 | per token minus expert group | 368,000 | [163,840, 655,360] |
| Primary | 400 | per token minus aggregate | 327,360 | [163,840, 655,360] |
| Held out | 200 | per token minus expert group | 2,169,000 | [1,146,880, 4,587,520] |
| Held out | 200 | per token minus aggregate | 2,169,320 | [1,146,880, 4,587,520] |
| Held out | 400 | per token minus expert group | 1,085,000 | [573,440, 2,293,760] |
| Held out | 400 | per token minus aggregate | 1,085,160 | [573,440, 2,293,760] |

The held-out fixture is the load-bearing part of this family. Its route table,
its 1,024-byte vector and its 28,672-byte envelope excess were declared in the
freeze and had never been rendered or executed, and its deltas landed inside a
band derived only from the calendar arithmetic.

**R2, inverse-rate scaling of synthetic completions: 12 of 12.** The largest
absolute error was 3,000 ps against the registered 6,000 ps allowance, on a
smallest magnitude of 431,082 ps. Six of the twelve cells were within 1,002 ps.

**R3, fluid granularity direction: 4 of 4.** Per-token fluid exceeded aggregate
fluid by 165,119 and 83,559 ps on the primary fixture and by 124,838 and
62,918 ps on the held-out fixture.

**R4, fluid insensitivity for equal message sets: 4 of 4.** Aggregate and
per-expert-group carry the identical ordered-pair table and differ only in
issue order within one source. Deltas were 1,119 and 1,559 ps on the primary
fixture and 1,838 and 1,918 ps on the held-out fixture, against the registered
2,000 ps bound. The held-out cells passed with 4 percent of the bound to
spare, so this family was close to failing and is not a formality.

**R5, Granite rate scaling: 6 of 6.** This is the check the void run registered
and never executed. Represented compute is `24 * 4,139 ns = 99,336,000 ps` and
is rate independent.

| Grouping | Profile | 200 Gbit/s network ps | 400 Gbit/s network ps | Ratio |
|---|---|---:|---:|---:|
| Aggregate | Packet | 809,083,960 | 404,322,600 | 2.0011 |
| Aggregate | Fluid | 779,798,570 | 389,899,306 | 2.0000 |
| Expert group | Packet | 1,032,571,000 | 516,297,000 | 2.0000 |
| Expert group | Fluid | 928,529,000 | 464,288,000 | 1.9999 |
| Per token | Packet | 1,992,707,000 | 996,365,000 | 2.0000 |
| Per token | Fluid | 1,022,573,000 | 511,310,000 | 1.9999 |

## Fatal guards, all held

- Full-duplex floor and calendar ceiling on all 36 cells.
- Exact ordered-pair equality between both sequenced groupings and the
  aggregate renderer, per `(layer, phase, source, destination)`, on both
  fixtures.
- Exact per-request equality on both fixtures.
- Hop ceilings: 16 of 16 on the primary fixture, 24 of 24 on the held-out
  fixture, 12,482 of 20,736 at Granite scale.
- Engine ownership: every dispatch sourced `engine_rank`, every combine
  returned to it.
- Conservation: all three groupings emitted identical directed bytes, 32,768
  and 24,576 on the two fixtures and 25,563,136 at Granite scale, and combine
  was the exact transpose of dispatch with unchanged routing ordinals.
- Physical quiescence verified in every one of the 36 native manifests.
- Input identity: both Granite inputs matched their authored-against digests.
- Aggregate-default regression: the Granite aggregate GOAL is 47,399 bytes with
  SHA-256 `6bb83366a3936bcf1e435cab008bb55c2966777528a9e8d885dd44d47f5a4943`,
  336 messages, 25,563,136 directed bytes, and its 400 Gbit/s completions
  reproduced the retained 503,658,600 ps packet and 489,235,306 ps fluid
  values exactly.
- Cost limits: every grouping stayed inside 30 seconds of render plus compile,
  1 GiB of traced memory, 64 MiB of GOAL text and 60 seconds per backend run.

## Granite scale and cost

| Grouping | Messages | Render plus compile s | Peak MiB | GOAL bytes | Slowest backend s |
|---|---:|---:|---:|---:|---:|
| Aggregate | 336 | 0.625 | 1.50 | 47,399 | 0.08 |
| Expert group | 1,008 | 1.050 | 3.31 | 185,452 | 0.30 |
| Per token | 12,482 | 16.718 | 17.17 | 2,243,956 | 58.02 |

The per-token 200 Gbit/s packet cell took 58.02 seconds against the 60-second
limit. That is the practical boundary of this scale point on this machine and
it is a measured finding, not a reason to move the limit.

| Grouping | Packet 200 ps | Packet 400 ps | Fluid 200 ps | Fluid 400 ps |
|---|---:|---:|---:|---:|
| Aggregate | 908,419,960 | 503,658,600 | 879,134,570 | 489,235,306 |
| Expert group | 1,131,907,000 | 615,633,000 | 1,027,865,000 | 563,624,000 |
| Per token | 2,092,043,000 | 1,095,701,000 | 1,121,909,000 | 610,646,000 |

## Three independent physical reviews

**Serialization and endpoint physics.** Peak per-rank egress at Granite scale
is 12,781,568 bytes and the engine rank's ingress is the same, so the
full-duplex endpoint floor is 255,631,360 ps at 400 Gbit/s. Adding the
99,336,000 ps compute gives a 354,967,360 ps step floor, and the fastest
observed Granite cell is 489,235,306 ps, 38 percent above it. The synthetic
floors behave the same way.

**Fair-share manifold, independent closed form.** The fluid profile shares an
endpoint max-min, so an aggregate step should complete at
`(first dispatch completion) + (total ingress bytes / rate)` whenever the home
ingress never idles after the first return starts. On the primary fixture that
predicts `491,520 + 655,360 = 1,146,880` ps at 200 Gbit/s against an observed
1,147,881, and on the held-out fixture `368,640 + 491,520 = 860,160` ps against
an observed 861,162. Both residuals are the same 1,001 to 1,002 ps of GOAL
nanosecond quantization, and the closed form was derived from the fixture
shapes rather than fitted. This is a post-specified cross-check, not a
registered oracle.

**End-to-end plausibility.** A 54-token Granite prefill at EP width eight,
all-remote on 400 Gbit/s, completes in 503.7 microseconds of which 99.3
microseconds is compute. That is a network-dominated step, which is the
expected shape for an expert-parallel group spread across a fabric with no
NVLink locality, and it is the deliberate what-if this study measures. A real
eight-GPU node would serve most of that traffic over NVLink, so these values
are not a prediction of a single-node deployment. TRAF-11 owns the NVLink
calibration that would make the intra-node comparison meaningful.

## Claim boundaries and residuals

The strict v2 trace observes the order framework dispatch returned. It does not
observe the order a fused kernel, NCCL or an RNIC posted bytes to the wire.
PLAY-14 retains that residual and this run does not upgrade it.

The Granite v1 tuple order is a Transformers reconstruction and remains scale
and cost evidence only.

All cells are all-remote native profiles, so the analytic intra-node locality
service is not exercised. TRAF-11 owns its calibration.

TRAF-26 retains real multi-engine population. Copying one engine's routing
table onto peer sources remains forbidden.

The pairwise source-only frontier is now documented as what it implements, the
rank's first send. Moving it to the last send would change accepted rendered
timing and is a separate model decision that needs its own freeze. It is
recorded here and in the docstring rather than as a new task, because no
registered acceptance clause claimed it.

## Registered acceptance clauses

Each clause of the TRAF-22 registry entry, quoted, with its evidence.

- *"Acceptance requires all fatal guards to pass"*: **demonstrated**. Six guard
  groups, no violation.
- *"exact sequenced-to-aggregate per-pair and per-request equality"*:
  **demonstrated** on both fixtures for both sequenced groupings.
- *"the independent EP-width-eight hop ceiling"*: **demonstrated**. 12,482 hops
  against the independent `54 * 8 * 24 * 2 = 20,736` ceiling.
- *"raw packet and fluid relations inside their preregistered bounds"*:
  **demonstrated**. R1, R3 and R4 evaluated from raw completions before any
  fatal oracle, 16 of 16 instances.
- *"the missing 200/400 Gbit/s Granite scaling check"*: **demonstrated**. R5,
  6 of 6, every ratio within 0.06 percent of two.
- *"unchanged aggregate-default bytes and timing"*: **demonstrated**. GOAL
  identity plus both retained 400 Gbit/s completions reproduced exactly.
- *"retention of the 30-second render-plus-compile, 1 GiB memory, 64 MiB GOAL
  and 60-second backend limits"*: **demonstrated**, with the per-token
  200 Gbit/s packet run at 58.02 seconds noted as the practical boundary.
- *"Replace those surrogates with bounds proved from the rendered endpoint
  dependency frontiers and the packet backend's full-envelope calendar"*:
  **demonstrated**. Both bounds are computed from the actually rendered
  messages, and the R1 band is the envelope-calendar excess.
- *"Align the documented source-only pairwise frontier with its
  implementation before requalification"*: **demonstrated** in commit
  `2f61990`, before the first run.
- *"Preserve the void chronology, then freeze and score at least one held-out
  payload or routing shape before its first run"*: **demonstrated**. The void
  record is unmodified and the held-out fixture is scored in R1 through R4.

Zero new IDs were registered. Every registered clause was demonstrated, so the
residual discipline required none.

## Verification and contradiction sweep

`ruff check .` passed and the full suite reported 1,213 passed with 7
environment-dependent skips.

The post-result sweep found the same three integrator-owned map omissions the
void run reported, unchanged and still outside this branch's scope:

- `README.md` does not distinguish aggregate-default traffic from the explicit
  sequence granularity.
- `docs/README_PRO.md` has no message-granularity fidelity row or dispatch
  sequence study entry.
- `docs/architecture.md` does not describe engine-local framework-return order
  at the explicit sequence level.

No statement in those three files contradicts this result; they are silent
about the sequence level rather than wrong about it.
