# Dispatch message sequence v1 results

## Corrected ownership outcome

The corrected ownership run is **void**. It therefore has no behavioral pass
fraction and closes no timing qualification. The run reached all twelve
synthetic native cells and every corrected Granite cost cell, but four fluid
cells violated a frozen fatal floor. The emitted traffic remained physically
plausible; the run refuted the floor's assumption that dispatch and combine
are two globally serial link loads.

All exact ownership, sequence, ordered-pair, request, byte, transpose, hop,
input-identity and quiescence guards passed. Those facts remain valid
structural findings, but a fatal guard is a precondition for interpreting the
scored relations. The raw packet and fluid predicates are retained below
without a numerator, denominator or family score.

The implementation now has one routed-contribution authority. It reads
`RoutedMoeSupply.engine_rank` once, expands the captured token routes once, and
returns ordered per-token dispatch contributions. The aggregate renderer
folds those contributions into request and pair tables. The sequenced
renderer groups the same contributions and transposes them for combine.
Ownership can no longer drift between the two renderers through duplicated
source traversal.

## Chronology and provenance

The original expectations and result remain unmodified chronology. TRAF-25
then corrected the source-multiplicity defect on main. The ownership refreeze
superseded only the invalid source-multiplied exact and physical literals; it
kept the original behavioral bands unchanged.

| Event | Commit |
|---|---|
| Original expectations-only freeze | `7efd71e7e54fc6faecde17c5faebab9430a2e847` |
| Original sequence implementation | `04eeb5ca50b4625f87f7872aa4fd204cf629bae4` |
| Original bounded Granite attempt | `bd752cd8cbe3b6948c165a2e548b0c09e7cea9f2` |
| Historical pre-TRAF-25 result | `fc42b4377bc5cba26d7138663a2597e7009b34b9` |
| TRAF-25 merge | `14d8447b838e651f8321ffb0588ea02219e26e9a` |
| Main merged into this branch | `f2e0581` |
| Corrected ownership expectations-only freeze | `82d3ab45ea47c811fa6db0d91ac8122e255fd62b` |
| Shared ownership implementation and observed revision | `50ea211f978229481af8ea372b3db7fc2c954701` |

The registered command first passed `--check-only` before the shared
ownership implementation. Check-only read no source artifacts, invoked no
native tool and created no output. The result-producing command wrote to the
new external directory
`$SIMLLM_WAVE6_RUN_ROOT/dispatch_sequence_v1-ownership-refreeze` and exited
nonzero only after writing the void result.

| Provenance field | Observed value |
|---|---|
| htsim gitlink | `fc4400e4ca619223481536632074045cb6af2756` |
| `htsim_rnic` SHA-256 | `cfb5014a663791f7619fe33309114a74e82878de860c14fc8a723713501f027d` |
| `txt2bin` SHA-256 | `f3745f34ad86febe9c9eebef10aee5fae00b8865cb29943344fb75b0f142495b` |
| Routing input SHA-256 | `24e986e989e21f1bfe7e758d4470928c82c3bbaec06072a839743b9b17d7cf5f` |
| Step input SHA-256 | `824cd9557293328bb42b593ac893b6a067302e545b087c9219195ccb8031d755` |
| Raw observation SHA-256 | `1c3b6681f3b4b917a70ad9111bd36452eb3bb5967afebf693d1a98e5690d2678` |
| Summary SHA-256 | `267165bcb2770e1dd0c5c1f9b1454acaaa4d881a86ba73b2d1d03732e3184aa0` |

Executable and gitlink hashes are observations, not frozen requirements on a
future checkout.

## Ownership and conservation evidence

The corrected strict v2 fixture has one engine, rank 0. Its source-local
framework-returned dispatch destinations are:

```text
source 0: 3, 1, 2, 1, 3, 2, 1, 3
source 1:
source 2:
source 3:
```

The aggregate, expert-group and per-token levels all projected the same
ordered-pair and per-request totals:

| Quantity | Aggregate | Expert group | Per token |
|---|---:|---:|---:|
| Messages, dispatch plus combine | 6 | 6 | 16 |
| Directed bytes | 32,768 | 32,768 | 32,768 |
| Remote vector hops | 16 | 16 | 16 |

The independent synthetic ceiling is `4 * 2 * 1 * 2 = 16` hops. Every
dispatch message sourced rank 0, every combine message returned to rank 0,
and combine preserved the exact routing ordinals while transposing the
directed rows.

The new EP-width-eight regression is deliberately capable of rejecting the
old rule. For both grouping modes it aggregates actual planned and rendered
messages by ordered rank pair, compares every layer and phase with the
aggregate renderer, checks dispatch source and combine destination against
`engine_rank`, and applies an independent token/top-k/layer/phase ceiling.
The corrected fixture emits 42 vector hops under a ceiling of 48. The old
`for source in ranks` loop would emit 336 hops, add non-engine source pairs and
violate both the pair comparison and the independent ceiling. It cannot pass
the guard even if both renderers were later changed together.

The Granite scale projection provides a second independent discrimination.
The old sequence loop emitted 101,318 hops and 207,499,264 bytes, above the
20,736-hop ceiling. The corrected authority emits 12,482 hops and 25,563,136
bytes, and the generated aggregate GOAL is 47,399 bytes with SHA-256
`6bb83366a3936bcf1e435cab008bb55c2966777528a9e8d885dd44d47f5a4943`.
The archived 334,432-byte GOAL and its SHA-256
`08a0403af66ff8a9d6b18f93afd15ae0bc925cc85555acf8a0593438a3d7bc92`
remain input provenance only.

## Fatal physical finding

The frozen synthetic floor added 16,384 bytes of dispatch and 16,384 bytes of
combine as two globally serial phases. Four observed cells were below that
floor:

| Cell | Observed ps | Frozen floor ps | Shortfall ps |
|---|---:|---:|---:|
| Aggregate fluid, 200 Gbit/s | 1,147,881 | 1,310,720 | 162,839 |
| Aggregate fluid, 400 Gbit/s | 574,441 | 655,360 | 80,919 |
| Expert-group fluid, 200 Gbit/s | 1,149,000 | 1,310,720 | 161,720 |
| Expert-group fluid, 400 Gbit/s | 576,000 | 655,360 | 79,360 |

This is a refuted fatal premise, so the run stays void. It is not converted
into a lost point or repaired after observation.

The post-specified causal audit explains the digits exactly. The GOAL graph
uses participant-local phase frontiers. At 200 Gbit/s, rank 2's 4,096-byte
dispatch completes at 492,521 ps and its 4,096-byte combine return then runs
to 656,361 ps. That return overlaps the tail of the two 6,144-byte dispatches,
which also complete at 656,361 ps. Their returns complete at 1,147,881 ps.
Thus the observed aggregate value is:

```text
1,310,720 - 163,840 + 1,001 = 1,147,881 ps
```

At 400 Gbit/s the same relation is:

```text
655,360 - 81,920 + 1,001 = 574,441 ps
```

The native fluid profile has independent source-uplink and
destination-downlink constraints, so this opposite-direction overlap is
intentional. The harness also takes the maximum of all flow completions and
the GOAL completion; it did not discard a late event. Per-token happened to
satisfy the old floor because its equal 2,048-byte dispatches reach the peer
frontiers together. That accidental synchronization does not validate a
global phase barrier.

The safe first-principles payload floor is one critical endpoint load,
`max(home egress, home ingress) * 8 / rate`: 655,360 ps at 200 Gbit/s and
327,680 ps at 400 Gbit/s. Every measured cell exceeds those post-specified
diagnostic floors. A stronger fixture-specific relation must be derived from
the emitted dependency graph and backend resource contract before a future
run, and a value learned from this fixture cannot be presented as its
pre-registered oracle.

## Raw synthetic observations

The fixture used zero compute, zero propagation and endpoint rates of 200 and
400 Gbit/s. Packet cells used a 4,096-byte maximum wire packet, a 64-byte data
header and full-envelope calendar reservation. Every native manifest reported
`physical_quiescence=verified`. The external completion CSVs retain every raw
flow; this table reports the observed range and step completion.

| Renderer | Profile | Gbit/s | Flows | Min FCT ps | Max FCT ps | Completion ps |
|---|---|---:|---:|---:|---:|---:|
| Aggregate | Fluid | 200 | 6 | 163,840 | 655,361 | 1,147,881 |
| Aggregate | Fluid | 400 | 6 | 81,920 | 327,681 | 574,441 |
| Aggregate | Packet | 200 | 6 | 491,520 | 1,070,080 | 1,890,280 |
| Aggregate | Packet | 400 | 6 | 245,760 | 535,040 | 945,640 |
| Expert group | Fluid | 200 | 6 | 163,840 | 655,361 | 1,149,000 |
| Expert group | Fluid | 400 | 6 | 81,920 | 327,681 | 576,000 |
| Expert group | Packet | 200 | 6 | 654,360 | 988,160 | 1,809,000 |
| Expert group | Packet | 400 | 6 | 326,680 | 494,080 | 905,000 |
| Per token | Fluid | 200 | 16 | 655,360 | 655,360 | 1,313,000 |
| Per token | Fluid | 400 | 16 | 327,680 | 327,680 | 658,000 |
| Per token | Packet | 200 | 16 | 248,320 | 1,395,200 | 2,544,000 |
| Per token | Packet | 400 | 16 | 124,160 | 697,600 | 1,273,000 |

## Retained unscored behavioral findings

The runner evaluated the ten behavioral predicates from raw native
completions before applying fatal oracles. Their truth values are retained,
but the behavioral score is withheld because the physical guard voided the
run.

All packet directions stayed positive, but all four magnitudes were outside
the unchanged frozen bands:

| Comparison | 200 Gbit/s delta ps | Frozen band ps | 400 Gbit/s delta ps | Frozen band ps |
|---|---:|---:|---:|---:|
| Per token minus expert group | 735,000 | [15,360, 61,440] | 368,000 | [7,680, 30,720] |
| Per token minus aggregate | 653,720 | [15,360, 61,440] | 327,360 | [7,680, 30,720] |

Both inverse-rate predicates were true: each 200 Gbit/s delta differed from
twice its 400 Gbit/s counterpart by 1,000 ps against the frozen 2,000 ps
tolerance. This supports a serialization mechanism, but cannot supply a score
for a void run.

The fluid comparisons also missed their unchanged 1,000 ps absolute bound:

| Comparison | 200 Gbit/s delta ps | 400 Gbit/s delta ps |
|---|---:|---:|
| Expert group minus aggregate | 1,119 | 1,559 |
| Per token minus aggregate | 165,119 | 83,559 |

The large per-token fluid movement follows the same dependency finding. Its
equal-sized dispatch messages synchronize the peer frontiers and remove the
short-return overlap available to aggregate and expert-group traffic. This is
a post-specified explanation, not a replacement expectation.

## Corrected Granite cost and physical sanity

The Granite v1 order remains a Transformers reconstruction. It is scale and
cost evidence, not an observation of framework, kernel, NCCL or wire issue
order.

The corrected 54-token, 24-layer point is practical under every retained cost
limit:

| Grouping | Messages | Plan s | Render s | Compile s | Render plus compile s | Peak MiB | GOAL bytes | Packet backend s | Fluid backend s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Aggregate | 336 | 0.271 | 0.309 | 0.172 | 0.480 | 1.51 | 47,399 | 0.082 | 0.059 |
| Expert group | 1,008 | 0.382 | 0.791 | 0.157 | 0.948 | 3.31 | 185,452 | 0.294 | 0.274 |
| Per token | 12,482 | 0.365 | 16.708 | 0.312 | 17.020 | 17.17 | 2,243,956 | 32.890 | 24.592 |

Each level carries exactly 25,563,136 directed bytes. The limits are 30
seconds for render plus compile, 1 GiB traced peak memory, 64 MiB of GOAL text
and 60 seconds per backend. No larger corrected size was measured, so this
study found no corrected impractical-size boundary.

The historical source-multiplied measurement remains useful cost chronology,
but not a current workload:

| Grouping | Historical messages | Historical bytes | Plan s | Render s | Compile s | Peak MiB | GOAL bytes | Practical |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Aggregate | 2,688 | 207,499,264 | 0.461 | 1.020 | 0.196 | 10.4 | 334,432 | yes |
| Expert group | 8,064 | 207,499,264 | 2.579 | 6.134 | 0.203 | 22.7 | 1,446,770 | yes |
| Per token | 101,318 | 207,499,264 | 1.856 | greater than 60 | not reached | 45.8 during attempt | not reached | no |

The ownership fix changed total group bytes by 8.117 times but peak per-rank
egress by only 2.117 times, from 27,060,224 to 12,781,568 bytes. The latter is
the relevant bandwidth-bound scale. The corrected 400 Gbit/s peak-egress term
is 255,631,360 ps. Adding the fixed `24 * 4,139 ns = 99,336,000 ps` compute
gives a 354,967,360 ps context floor. Every corrected 400 Gbit/s Granite cell
lies above it:

| Grouping | Packet completion ps | Fluid completion ps |
|---|---:|---:|
| Aggregate | 503,658,600 | 489,235,306 |
| Expert group | 615,633,000 | 563,624,000 |
| Per token | 1,095,701,000 | 610,646,000 |

Archived-to-corrected aggregate completion changed by 1.890 times for packet
and 1.909 times for fluid. After removing fixed compute, the network-bearing
ratios are about 2.11 and 2.14, close to the 2.117 peak-egress change and far
from the 8.117 total-byte change. Expert-group completion changed by 1.392 and
1.410 times while its message sequence also changed. No observed completion
scaled with the physically irrelevant total group-byte ratio.

The registered 200 Gbit/s Granite context floor is 610,598,720 ps, but the
corrected runner executed Granite only at 400 Gbit/s. The required 200/400
Granite scaling comparison is therefore unqualified and remains in TRAF-22.

## Claim boundaries and residuals

TRAF-21's sequence interface remains complete, but corrected timing
qualification stays open under TRAF-22. TRAF-22 is upgraded to P0 because a
fatal validation gate voided this run. It owns dependency-aware physical
bounds, the missed packet and fluid relations, a held-out qualification, the
missing 200 Gbit/s Granite point and retention of the corrected cost limits.

The physical audit also found a non-causal source-frontier discrepancy:
`pairwise_all_to_allv` documents the last source send as its source-only
frontier, while its compatibility implementation retains the first send.
The first send was tied for the long dispatch tail in this fixture, and peer
receive frontiers already explain the observed overlap, so the discrepancy
did not cause the fatal miss. TRAF-22 records it for correction before timing
is requalified. Changing it after this run to obtain a different number would
invalidate the evidence.

PLAY-14 retains the issue-order residual. A strict v2 trace observes what
framework dispatch returned; it does not observe the order a fused kernel,
NCCL or an RNIC used to post bytes. This result does not upgrade that claim.

CORE-41 owns the separate analytic locality defect that omits destination
ingress. These all-remote native cells do not call that analytic service, so
CORE-41 does not alter the values above. Any future analytic intra-node value
remains pending CORE-41.

TRAF-26 retains real multi-engine population. Copying one engine's routing
table to peer sources remains forbidden. A globally serial MoE phase barrier
would also be a new, separately frozen model decision; TRAF-9 already records
the broader whole-layer MoE ordering approximation.

## Verification and contradiction sweep

After this report update, `ruff check .` passed and the full Python suite
reported 1,057 passed with seven environment-dependent skips. A focused
routed-traffic, step-communication, preplay-routing and study dry-run set
reported 52 passed. The earlier routed-only set reported 28 passed, and the
broader implementation set reported 128 passed with two skips.

The owning traffic and preplay module docs now state the single-engine
authority, void timing result and retained ordering boundary. The required
post-result sweep found three integrator-owned omissions that remain outside
this branch's scope:

- `README.md` does not distinguish aggregate-default traffic from the explicit
  sequence granularity.
- `docs/README_PRO.md` has no message-granularity fidelity row or dispatch
  sequence study entry.
- `docs/architecture.md` does not describe engine-local framework-return order
  at the explicit sequence level.

Those map omissions do not change the module interface or evidence, and this
branch does not edit them.
