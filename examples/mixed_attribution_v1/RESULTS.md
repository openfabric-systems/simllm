# Mixed attribution v1: results against pre-registered expectations

Run of 2026-08-14. Frozen by expectations-only commit
**`4e7bcb9a356f9660b7f708c6e9f5d53735e264a3`**, which landed before the
behavior, before the harness and before any run. Nothing in
[expectations.md](expectations.md) was edited after the run.

**Verdict: the run is not void. All 8 fatal guards held. The scored exact
relation passed, 1 of 1. The scored behavioral relations passed, 4 of 4. The
two scored classes are kept separate from each other and from the guards, and
no count is added across them.**

## Outcome in one paragraph

Per-request TTFT and TPOT now come out of a placement that mixes intra-node
NVLink service with cross-node fabric service, which the reducer refused
outright before this change. In the flagship cell a single step carries 24
NVLink-owned artifacts, 24 fabric-owned artifacts and 24 compute artifacts,
and their components add to the request's realized TTFT exactly, with no
remainder and no unnamed term. The masked NVLink service that the fabric hid
inside the same step, 110,000 ps of it, is reported under its own name and is
absent from every latency total. Halving the NVLink rate moved the TTFT by
exactly 120,000 ps, the doubling of the NVLink-owned service alone, while the
fabric component stayed identical to the picosecond and no artifact changed
owner. The all-remote path is byte-identical to what the previous code
computed, checked against the exact input shape that code saw.

## Chronology and provenance

The freeze landed first, then the implementation, then the harness, then one
result-producing run. No run was discarded, repeated or replaced, and no
frozen literal was edited.

| stage | commit |
|---|---|
| expectations only | `4e7bcb9a356f9660b7f708c6e9f5d53735e264a3` |
| implementation and tests | `1571f84156755eb8c5a1ef5681f9cebe6e21d2cd` |
| study harness | `dbeb5fd7e385c1e5b003c2a8af156c87b3d7409d` |

| provenance field | value |
|---|---|
| SimLLM revision observed by the run | `dbeb5fd7e385c1e5b003c2a8af156c87b3d7409d` |
| expectations SHA-256 | `906ac794c3a2567bb8f07f57f4a1756e1001219a071dd9e581293b028899d1a6` |
| htsim gitlink observed by the run | `fc4400e4ca619223481536632074045cb6af2756` |
| `htsim_rnic` SHA-256 | `32035c778e40e9b11dd32d081350a36a92872855a97dc4b5f217c634420c0816` |
| `txt2bin` SHA-256 | `f3745f34ad86febe9c9eebef10aee5fae00b8865cb29943344fb75b0f142495b` |
| captured trace SHA-256 | `36334f3aaa767c46d5f9c8498e02f6c2805a46e5000a57aea2747e17dd5d1341` |
| runtime | Python 3.12.12 on Linux x86-64 |

Two probes ran against the unmodified base before the freeze and are declared
in it: a structural probe that reported only how many phases of each step are
local-only, fabric-only or mixed, and a feasibility and timing probe of a
configuration that is not a cell of this study. Both are run-configuration
evidence. The structural probe's phase counts are reproduced exactly by the
measured artifact tables below, which is itself a consistency check.

## What ran

Five cells over one fixed step-record set: three prefill tokens of the tracked
Granite capture, one token per step, replayed through `HtsimStepSink` and
`HtsimRequestMetricReducer` at 400 Gbit/s on `rnic-nn-fluid`. Every cell
executes 72 artifacts per step, 24 compute and 48 collective.

| cell | hosts | expert layout | NVLink bytes/s | backend runs | wall |
|---|---|---|---:|---:|---:|
| `all-remote-450` | `ABCD` | uniform | 450,000,000,000 | 144 | 43 s |
| `all-local-450` | `AAAA` | uniform | 450,000,000,000 | 0 | 1 s |
| `all-local-225` | `AAAA` | uniform | 225,000,000,000 | 0 | 1 s |
| `mixed-450` | `AABB` | node-local-even | 450,000,000,000 | 116 | 45 s |
| `mixed-225` | `AABB` | node-local-even | 225,000,000,000 | 116 | 25 s |

376 `htsim_rnic` invocations in total and 4.6 MB of retained artifacts, all
outside the repository under the run root named by
`SIMLLM_MIXED_ATTRIBUTION_RUN_ROOT`.

## Measured per-request results

| cell | TTFT ps | TPOT ps | kernel ps | NVLink ps | fabric ps | masked NVLink ps |
|---|---:|---:|---:|---:|---:|---:|
| `all-local-450` | 656,000 | 656,000 | 24,000 | 632,000 | 0 | 0 |
| `all-local-225` | 1,278,000 | 1,278,000 | 24,000 | 1,254,000 | 0 | 0 |
| `mixed-450` | 50,028,160 | 95,106,000 | 24,000 | 120,000 | 49,884,160 | 110,000 |
| `mixed-225` | 50,148,160 | 95,116,000 | 24,000 | 240,000 | 49,884,160 | 220,000 |
| `all-remote-450` | 101,512,678 | 101,512,678 | 24,000 | 0 | 101,488,678 | 0 |

The component columns are the TTFT partition. Every row's kernel plus NVLink
plus fabric equals its TTFT exactly, and the masked column is outside that sum
by construction.

Ownership by artifact, first step of each cell:

| cell | compute | NVLink owned | fabric owned | co-critical |
|---|---:|---:|---:|---:|
| `all-local-450` | 24 | 48 | 0 | 0 |
| `mixed-450` | 24 | 24 | 24 | 0 |
| `all-remote-450` | 24 | 0 | 48 | 0 |

TPOT exceeds TTFT in the mixed cells because the node-local expert layout is
built from the capture's first prefill token: step 0 has 24 fully intra-node
phases while steps 1 and 2 have 2 each, which the decode NVLink component
reports as exactly 20,000 ps, i.e. 2 phases times 5,000 ps times 2 steps. That
is a property of the declared fixture and is not a claim about decode behavior.

## Physical sanity before precision

Three independent framings, each stated against the bounds frozen before any
digit was read.

**Network and serialization physics.** A 2,048-byte message cannot cross a
400 Gbit/s link faster than 40.96 ns. The measured fabric services take exactly
two values: 2,081,920 ps for a phase with two remote destinations and
2,122,881 ps for a phase with three. Subtracting the model's 2.000 us
propagation term leaves 81,920 ps and 122,881 ps, i.e. exactly two and three
message serializations, the second one picosecond above 3 x 40,960 because the
fluid manifold floors the shared-source rate at `floor(400e9 / 3)` and then
ceilings the completion to a whole picosecond. The all-remote step is
`38 x 2,122,881 + 10 x 2,081,920 = 101,488,678` ps, which is the measured value
to the picosecond and sits inside the frozen interval
`[96,000,000, 110,000,000]` ps.

**Interconnect physics.** One endpoint's NVLink service is
`ceil(peak_bytes * 1e9 / rate)` whole nanoseconds. The measured all-local
services take exactly two values, 14,000 ps for the 38 phases whose source
egress is 3 x 2,048 bytes and 10,000 ps for the 10 phases at 2 x 2,048 bytes,
reproducing `ceil(13.6533) = 14` and `ceil(9.1022) = 10`. Their sum,
632,000 ps, lies inside the frozen `[240,000, 672,000]` ps interval. At half
the rate the same phases price at 28,000 and 19,000 ps, sum 1,254,000 ps,
inside the frozen `[1,216,000, 1,264,000]` ps bracket and below twice the fast
value exactly because `ceil(2x) <= 2 ceil(x)`.

**The covariate that had to move with it.** Halving the NVLink rate moved the
all-local step by 622,000 ps, which is exactly the NVLink component's own
increase, and moved the mixed cell by 120,000 ps, which is exactly the
doubling of its 24 NVLink-owned phases. The mixed cell's fabric component did
not move by one picosecond, which is the covariate that must not move: NVLink
bandwidth cannot change fabric service. A masked-service leak would have shown
up here as an extra 110,000 ps.

**End-to-end plausibility, and what this study does not claim.** These are
microsecond-scale steps for a single token because compute is a declared
24,000 ps fixture, roughly four thousand times below the near 99 us a B100
roofline prices for this geometry. The 2.000 us propagation term is itself
15.06x below the calibrated width-8 collective intercept that wave 14 is
landing elsewhere. The absolute microseconds here are therefore a floor on
what a calibrated model reports, and this study claims only that the
components are the right ones, that they are owned by the right resource, and
that they add up. It makes no claim about the realistic ratio of kernel to
collective time, and none about the deployment behavior of any real system.

## Fatal guards

All 8 held, so no scored fraction is invalidated. They are unscored by
construction and are not added to any total.

| guard | result |
|---|---|
| **G1** all-remote byte identity | 3 of 3 steps compared, 0 step differences and 0 totals differences between the enriched replay and the same steps with the per-artifact projection stripped |
| **G2** conservation | 20 checks, 0 failures: every step and every request interval totals its own realized elapsed time in both partitions |
| **G3** roll-up agreement | 15 checks, 0 failures: the medium components roll up to the coarse partition in every step |
| **G4** inactive components | 15 intervals, 0 violations of `kv_ps`, `dma_ps`, `nic_ps` and `control_ps` |
| **G5** backend health | 25 checks, 0 failures: every cell reduced 3 captured, epoch 0, quiescent steps with a positive TTFT |
| **G6** projection agreement | 15 steps, 0 failures: all five per-artifact tuples have length 72 and the NVLink-medium local services sum to the published `nvlink_service_ps` |
| **G7** all-remote has no NVLink component | owned and masked NVLink are 0 in all 3 steps |
| **G8** all-local runs no backend | 0 runs, 0 fabric bytes and 0 fabric service in all 6 all-local steps |

## Scored exact relation: 1 of 1

> **E1** For every cell, step and reduced interval, a recomputation using only
> the Python standard library, the published per-artifact tuples and the
> declared arrivals reproduces the reducer's owner for every artifact, its
> seven medium components, its coarse attribution, TTFT, TPOT as an exact
> `Fraction`, and both per-request partitions, exactly.

**Passes.** 20 comparisons, 0 mismatches: 15 step tables of 72 artifacts each
plus 5 per-request rows. As the freeze states, this shares the per-artifact
inputs with the reducer, so it tests owner selection, masking, the
pending-interval carry, the queue-gap charge and the accumulation into TTFT
and decode partitions, and does not test the sink's composition.

## Scored behavioral relations: 4 of 4

> **F1** In both all-local cells, for every step, the fabric component is
> exactly zero, the kernel component is exactly 24,000 ps, the NVLink
> component lies in its frozen interval, and the step latency equals kernel
> plus NVLink exactly.

**Passes.** 6 instances, 0 failures. 632,000 ps inside `[240,000, 672,000]` at
450 GB/s and 1,254,000 ps inside `[480,000, 1,344,000]` at 225 GB/s, with
`24,000 + 632,000 = 656,000` and `24,000 + 1,254,000 = 1,278,000` exact.

> **F2** For `mixed-450`, in the TTFT interval, the NVLink component is exactly
> 120,000 ps, the fabric component is strictly positive, the kernel component
> is exactly 24,000 ps, the three plus the zero queue gap total the request's
> TTFT, and the step reports at least 24 NVLink-owned and at least 24
> fabric-owned artifacts.

**Passes.** NVLink 120,000 ps, i.e. 24 fully intra-node phases at
`ceil(2048 * 1e9 / 450e9) = 5` ns each; fabric 49,884,160 ps; kernel 24,000 ps;
total 50,028,160 ps, equal to the published TTFT; 24 NVLink-owned and 24
fabric-owned artifacts in the same step. This is the relation BACK-43 exists
for: on the previous code the interval could not be produced at all, and under
any step-level ownership rule the NVLink component would be zero.

> **F3** `mixed-225` and `mixed-450` differ by exactly 120,000 ps in TTFT and
> in the NVLink component, their fabric components are byte-identical, and no
> artifact changes owner. In both all-local cells the NVLink component at
> 225 GB/s lies in `[2 n450 - 48,000, 2 n450]` ps and the latency difference
> equals the NVLink component difference exactly.

**Passes.** 4 instances, 0 failures. The mixed TTFT delta is 120,000 ps, the
mixed NVLink delta is 120,000 ps, the fabric component is 49,884,160 ps in
both cells, and the owner table is identical artifact for artifact. The
masked NVLink service doubled too, 110,000 to 220,000 ps, and did not enter
any total, which is exactly what this relation was registered to catch.

> **F4** Across the three 450 GB/s cells the TTFT order is all-local, mixed,
> all-remote; the fabric component is strictly increasing in that order with
> an exact zero in the first cell; and the all-remote step service lies in
> `[96,000,000, 110,000,000]` ps.

**Passes.** TTFT 656,000 then 50,028,160 then 101,512,678 ps; fabric component
0 then 49,884,160 then 101,488,678 ps; all-remote step service 101,512,678 ps
inside the frozen interval.

## Findings

**F-1. Ownership is decided by the fabric's fixed per-phase cost, not by
bandwidth.** NVLink is 9x faster than the fabric per byte at these rates, but
the measured per-phase ratio is 152x, because a fabric phase pays 2.000 us of
propagation before it moves a byte. Every artifact that carried even one
fabric segment was fabric-owned in every cell, and every NVLink-owned artifact
was a phase with no fabric segment at all. The frozen crossing analysis said an
artifact would need roughly 150 to 400 times more local bytes, or an NVLink
rate that many times lower, to change owner; the measurement confirms the
regime rather than probing the boundary. The practical consequence for
placement studies is that intra-node placement pays off by removing phases
from the fabric, not by making a mixed phase cheaper.

**F-2. Masked service is large enough to matter and must stay out of the
totals.** In `mixed-450` the fabric hid 110,000 ps of NVLink service, which is
92 percent as large as the whole NVLink-owned component of the same step. Had
it been added to the partition, TTFT would have been overstated by that amount
and the bandwidth relation F3 would have reported 240,000 ps of movement
instead of 120,000 ps. The separate name and the absent total are what keep
this from becoming a silent error.

**F-3. The canonical mixed configuration is blocked elsewhere.** Tensor
parallelism inside a node with expert parallelism across nodes is the
configuration a reader would expect to produce NVLink-owned collectives, and
it cannot be planned today: the graph projection refuses
`tp_ranks=(0, 1)` with `ep_ranks=(0, 1, 2, 3)` with "graph cannot be
represented by ordered GOAL artifacts". This study therefore produced its
NVLink-owned artifacts from fully intra-node MoE phases under a declared
per-layer expert layout. The limitation is registered as BACK-44.

## Disclosures

- The compute term is a declared 24,000 ps fixture, not a device model, and
  the NVLink rate is the same uncalibrated 450 GB/s one-direction surrogate
  TRAF-10 registered. Neither is a calibration and neither is claimed to be.
- The node-local-even expert layout was chosen so that a single step carries
  both fully intra-node and fabric-crossing phases. It is a declared fixture,
  disclosed in the freeze before the run, and the study says so rather than
  presenting it as a natural placement.
- The `co_critical_ps` component was exactly zero in every measured cell, so
  it carries unit evidence only. The ownership comparison also inherits
  CORE-48's missing cross-node destination-ingress serializer, which
  under-charges a converging combine and biases a near-boundary artifact
  toward NVLink ownership. Both are registered as BACK-45.
- E1's recomputation shares the sink's per-artifact inputs, so it qualifies
  the reducer, not the composition that produced those inputs. The composition
  is qualified separately by the closed forms in the physical sanity section.
