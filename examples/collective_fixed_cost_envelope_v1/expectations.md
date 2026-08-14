# Frozen expectations: per-collective fixed-cost envelope with named arms

This is an expectations-only freeze. It lands before any implementation of the
named fixed-cost arms, before the cross-node provisional profile exists, and
before the first run of the study it describes. Nothing below is derived from a
measurement of the mechanism under study.

## Working-tree status at freeze time

The worktree is on branch `codex/traf36_collective_envelope`, and `git status
--porcelain` is empty at `e18b9b0102808e9b8e0f276c2b82c51ed8c5b51d` ("Merge
pull request #80 from openfabric-systems/codex/comp32_composed_budget"), which
is also the merge base with `main`. The only files this commit adds are
`examples/collective_fixed_cost_envelope_v1/expectations.md` and
`examples/collective_fixed_cost_envelope_v1/expectations.json`. No profile, no
envelope object, no configuration surface, no test and no study runner exists
yet.

## What this study is about

The per-collective fixed cost is the part of a semantic collective's service
that is not endpoint serialization. Today the repository makes a silent choice
between exactly two values for it, and the choice is worth more than the whole
rest of the step:

- The default charges no fixed cost at all beyond what the fluid backend
  already prices, which is one 2.000 us propagation delay per collective.
- The single selectable profile charges a DGX B200 intra-node NVLink ring
  ALL-REDUCE intercept, 30.128029 us at participant width 8, unchanged, on top
  of that same fabric transport.

Nothing owns the interval between them, and the interval decides the answer to
questions this repository reports. This study turns the choice into an
explicitly bracketed envelope with named arms, adds a provenance-labeled
provisional cross-node profile, and measures the bracket.

## Verified premises, with file and line

Every premise in the task brief was checked against the working tree at the
freeze commit before this document was written.

1. `simllm/traffic/collective_latency.py:183` to `:194` defines
   `B200_NCCL_2_27_LOCAL_PROFILE` with `profile_id="b200-nccl-2.27-local-v1"`,
   `bandwidth_bytes_per_second=70_027_079_100`, `participant_latency_ps` equal
   to `((2, 10_722_112), (4, 15_745_167), (8, 30_128_029))`,
   `source_payload_bytes_min=8`, `source_payload_bytes_max=262_144` and
   `propagation_reference_ps=2_000_000`. Confirmed exactly as stated.
2. `simllm/traffic/collective_latency.py:120` to `:131`
   (`CollectiveLatencyProfile.base_latency_ps`) raises `ValueError` for any
   participant count that is not a row of the table. Widths outside the table
   fail closed. Confirmed.
3. `simllm/backends/step_sink.py:697` to `:723` is the only place the step path
   consults the profile. It resolves the profile, computes the critical
   endpoint load, takes `base_latency_ps(participant_count)`, and charges it
   once per semantic collective, or zero when the collective carries no bytes.
   With no profile the loop is skipped entirely and every base latency is zero.
   Confirmed.
4. `simllm/backends/step_sink.py:243` to `:250` refuses a resolved profile
   unless `profile == "rnic-nn-fluid"`, so the propagation reference stays
   explicit. Confirmed.
5. `simllm/backends/step_sink.py:1151` to `:1159` is the composition:
   `composed_service = base_latency + max(local_service, fabric_service)` per
   artifact, and `makespan = represented_compute + sum(composed_service)`.
   Confirmed. This is the closed form the whole study rests on.
6. TRAF-32 was declared as a contingent residual ID by the composed-step-budget
   freeze at `examples/composed_step_budget_v1/expectations.md:184` ("the
   finding is that the calibrated envelope is too narrow for the mission
   workload, the composed measurement is void, and the defect takes TRAF-32")
   and again at `:381` ("a collective floor defect takes TRAF-32"), and is
   carried in `examples/composed_step_budget_v1/expectations.json:177` as
   `"collective_floor_defect": "TRAF-32"`. That study then reported it unused at
   `examples/composed_step_budget_v1/RESULTS.md:389`, and the earlier floor
   study said the same at `examples/collective_latency_floor_v1/RESULTS.md:464`.
   A search of every file under `docs/modules/` finds no TRAF-32 entry, and
   `docs/task-ledger.json` lists it in neither `closed` nor
   `retired_never_landed`. TRAF-32 is therefore currently registered nowhere.
   Confirmed.
7. The 2.000 us per-collective propagation term is a measured property of the
   fluid backend, not an assumption of this study:
   `docs/modules/traffic.md:475` to `:478` records that over 31 matched decode
   compositions `2 * fabric(400G) - fabric(200G)` returned 96,000,006 to
   96,000,048 ps across 48 collectives, i.e. 2.000 us each plus backend
   quantization. Confirmed.
8. `docs/modules/traffic.md:483` to `:486` records that the width-8 endpoint
   envelope ceiling of 458,752 bytes was reached to within 17 percent by a
   34-token prefill step, so a larger case is rejected at planning time.
   Confirmed.

### One premise of the brief is refined rather than confirmed

The brief describes the 2.000 us propagation constant as understating the
width-8 intercept by 15.06x and calls it "the only width-dependent term". The
first half is confirmed: 30,128,029 / 2,000,000 = 15.064. The second half is
backwards as written. The propagation term is the one term that is *not*
width-dependent: `simllm/backends/step_sink.py:1151` charges one fabric
transport per artifact and each artifact is one collective, so propagation
contributes 48 x 2.000 us regardless of participant width. The width-dependent
term is the intercept table itself, which is exactly why its absence removes
all width dependence from the fixed cost and lets endpoint serialization,
which falls as width rises, decide the width ordering on its own. The
conclusion the brief draws is right; the sentence naming the term is not.

## Recomputation of the two audit numbers

The brief asks for an independent recomputation before freezing, as evidence
that the mechanism is understood. Both reproduce.

- 48 collectives per step (24 layers x 2 all-to-alls) at width 8 add
  48 x 30,128,029 = 1,446,145,392 ps. A decode step reported at 212.71 us
  therefore becomes 212.71 + 1446.145392 = 1658.855392 us, i.e. 1658.86 us.
- At width 4 the same 48 collectives add 48 x 15,745,167 = 755,768,016 ps. An
  ep4 decode step of 259.23 us (which is 1.2187 x 212.71 us) becomes
  1015.00 us, and 1015.00 / 1658.86 = 0.61186, i.e. 0.6119.

Both audit figures are arithmetic on the intercept table, and both are
reproduced here without reading any new measurement.

## Structural inputs read from the existing lowerer

These are inputs, not results. They are deterministic properties of
`simllm/backends/step_lowerer.py` and `simllm/traffic` at the freeze commit for
the fixture defined below, and they were read at freeze time so that the
predictions can be closed form. They do not involve the mechanism under study,
which does not exist yet.

| cell | collectives | width | endpoint bytes each | provider compute ps |
|---|---|---|---|---|
| ep4 prefill | 48 | 4 | 98,304 | 152,862,720 |
| ep4 decode | 48 | 4 | 12,288 | 152,871,497 |
| ep8 prefill | 48 | 8 | 114,688 | 98,935,954 |
| ep8 decode | 48 | 8 | 14,336 | 98,944,731 |

Every collective is a pairwise ALL-TO-ALLV over the expert-parallel group. The
tensor-parallel world has size 1, so no ALL-REDUCE is emitted.

### Disclosed feasibility probe

One measurement was taken before this freeze, and it is disclosed rather than
hidden. A single ep8 decode step at 400 Gbit/s with `context_length=65` and no
profile was run through the unmodified sink to confirm the study is affordable
and that htsim is reachable. It returned `step_latency_ps=209,194,608` with
`compute_estimate_ps=99,436,251` and 336 flows in 22 seconds of wall time. That
record has a different context length from the fixture frozen below, so it is
not one of the frozen cells, but it is close enough to the ep8 decode cell that
it must be treated as prior information: the napkin prediction for that probe
is 99,436,251 + 48 x 2,286,720 = 209,198,811 ps, which the probe undershot by
4,203 ps, or 0.002 percent. The tolerance chosen for S1 below is 25 times wider
than that observed deviation, and it was chosen after seeing it. This is stated
plainly so the tolerance is not read as an independent prediction.

## The fixture

Deterministic step records constructed in the study runner, no live framework,
no adapter and no captured trace. Model dimensions match
`examples/collective_latency_floor_v1` exactly so the constants are comparable
with the accepted floor study: 24 layers, hidden size 1024, intermediate size
512, 16 heads, 8 KV heads, head size 64, vocabulary 49,152, 2 dtype bytes, 32
experts, top-k 8, MoE intermediate size 512, and `local_num_experts = 32 /
ep_world` so total expert count is held fixed as the expert-parallel width
changes.

Two records per cell:

- step 0, PREFILL, 8 new tokens, context length 8
- step 1, DECODE, 1 new token, context length 9

No placement manifest, so every segment is classified all-remote and reaches
the fabric. `tp_ranks=(0,)`, `ep_ranks=tuple(range(ep_world))`,
`profile="rnic-nn-fluid"`, `RooflineProvider(efficiency=0.7)`, `b100` GPU
envelope, ideal host model.

## The sweep

Three parameters vary. Sixteen cells, thirty-two simulated steps.

- Fixed-cost arm: `off`, `floor`, `local`, `cross` (defined below).
- Link bandwidth: 400 Gbit/s and 200 Gbit/s.
- Expert-parallel width: 4 and 8.

## The named arms

The arm is a selection on a named envelope object. An envelope names two
profiles, its `lower` and `upper` arm, plus the `off` arm that charges no fixed
cost at all. Two envelopes are shipped, and the study draws its four arm labels
from them:

| study arm | envelope | arm | charged intercept |
|---|---|---|---|
| `off` | `intra-node-fixed-cost-v1` | `off` | none, today's default |
| `floor` | `intra-node-fixed-cost-v1` | `lower` | 0 ps surcharge, claimed fixed cost is the modeled 2.000 us propagation |
| `local` | `cross-node-fixed-cost-provisional-v1` | `lower` | `b200-nccl-2.27-local-v1` |
| `cross` | `cross-node-fixed-cost-provisional-v1` | `upper` | `b200-nccl-2.27-cross-node-provisional-v1` |

The `upper` arm of `intra-node-fixed-cost-v1` is the same profile object as the
`lower` arm of `cross-node-fixed-cost-provisional-v1`, so it is run once, under
the label `local`. That coincidence is deliberate and is the seam between the
two brackets: the intra-node capture is the ceiling of what a collective that
stays on NVLink can cost, and it is simultaneously the floor of what the same
collective costs once its ring steps cross the fabric, because a fabric hop
cannot be cheaper than the NVLink hop it replaces.

### Why the `floor` arm charges zero and still means 2.000 us

The implemented composition at `simllm/backends/step_sink.py:1151` adds the
intercept on top of the backend transport. The intercept is therefore a
surcharge, not a total. The lower bound on the total per-collective fixed cost
is one propagation delay, and the fluid backend already charges exactly that
(premise 7). The honest surcharge for a lower-bound arm is consequently zero,
and the arm's claim is carried by the profile's `propagation_reference_ps`
field, which already exists for this purpose. The `floor` arm therefore states
"the per-collective fixed cost is 2.000 us and nothing more", which is what the
default silently assumes, and it makes that assumption a named, reportable
claim instead of an absence.

### The surcharge over-counts one propagation, by construction

Because the intercept is a surcharge on a transport that already contains one
2.000 us propagation delay, the realized per-collective fixed cost of an active
arm is `intercept + 2.000 us`, while the source capture's intercept was a
complete intra-node figure that contained an NVLink hop, not a fabric
propagation. Every active arm therefore over-counts by at most 2.000 us per
collective: 6.6 percent at width 8, 12.7 percent at width 4, 18.7 percent at
width 2 relative to the `local` arm. This is a property of the accepted
behavior at the freeze commit, not something this study introduces, and this
study does not change it. It is registered as a residual rather than silently
corrected, because correcting it would move an accepted baseline.

## The provisional cross-node profile

`b200-nccl-2.27-cross-node-provisional-v1` is labeled
**provisional-transferred**, never calibrated. No cross-node measurement was
taken. Every number below is a stated transfer from evidence already committed
to this repository, and every width carries a band. Widths outside the table
fail closed exactly as premise 2 requires.

### The transfer, in one sentence

A ring collective over W ranks takes `2 (W - 1)` ring steps; the source capture
performed all of them over NVLink inside one DGX B200; in the reference
configuration an 8-wide expert-parallel group is spread one rank per node, so
every one of those ring steps becomes a fabric hop; the transfer replaces the
per-ring-step NVLink cost implied by the source table with a per-ring-step
fabric cost anchored in this repository's own documents.

### Step one: the NVLink ring-step cost implied by the source table

Using only the two extreme published widths,
`(30,128,029 - 10,722,112) / (14 - 2) = 1,617,159.75` ps, rounded up to
`NVLINK_RING_STEP_PS = 1,617,160` ps. An ordinary least-squares fit of
`I(W) = a + b * 2(W - 1)` over all three published points gives
`b = 1,642,973.75` ps and `a = 6,816,628.5` ps. The two slopes differ by
1.6 percent, far inside the band below, so the simpler two-point slope is used
and the least-squares value is recorded as the robustness check. Provenance:
`simllm/traffic/collective_latency.py:186` to `:190`, which in turn cites the
nccl-tests issue 333 capture recorded by
`examples/collective_latency_floor_v1/RESULTS.md`.

### Step two: the fabric ring-step cost, with a band

| anchor | value | source |
|---|---|---|
| low | 2,000,000 ps | this repository's own measured fluid one-way propagation, `docs/modules/traffic.md:475` to `:478`; a fabric hop cannot be faster than propagation |
| point | 3,000,000 ps | propagation 2,000,000 ps plus 1,000,000 ps, one half of Kalia et al. ATC'16's commodity RDMA round-trip anchor of about 2 us, `docs/papers/msg-size-vs-bandwidth.md:80` to `:82` and the A1 row at `:101` |
| high | 5,000,000 ps | propagation 2,000,000 ps plus 3,000,000 ps, the upper p50 of UCCL's Table 2 ACK turnaround of 2 to 3 us, `docs/papers/msg-size-vs-bandwidth.md:33` to `:34` |

### Step three: the table

`cross(W) = I(W) + 2 (W - 1) * (fabric_ring_step - 1,617,160)`.

| width | ring steps | low ps | point ps | high ps |
|---|---|---|---|---|
| 2 | 2 | 11,487,792 | 13,487,792 | 17,487,792 |
| 4 | 6 | 18,042,207 | 24,042,207 | 36,042,207 |
| 8 | 14 | 35,487,789 | 49,487,789 | 77,487,789 |

The profile's charged intercept is the point column; the band is machine
readable on the profile so no consumer can read the point value without the
uncertainty attached.

### Where this transfer is weakest, stated before the run

- It keeps the source's ring algorithm. NCCL selects a tree or CollNet
  algorithm for small messages across nodes, whose depth grows as log W rather
  than as `2(W - 1)`, so the point estimate is biased high at width 8. The
  low band edge, 35.49 us at width 8, is the edge that sits inside the range
  published nccl-tests cross-node small-message all-reduce results usually
  occupy; the point estimate at 49.49 us sits above it.
- It keeps the source's operation. The capture is a ring ALL-REDUCE and the
  workload is a pairwise ALL-TO-ALLV. That mismatch is inherited from
  `b200-nccl-2.27-local-v1` and is not introduced here, but it is now a
  first-class contributor to the bracket width and is registered as a residual.
- It keeps the source's generation and stack. B200 with NCCL 2.27; a different
  NIC generation or a persistent-kernel dispatch such as DeepEP would move the
  fixed cost substantially, in the direction of the low band edge or below it.

## Napkin bounds, written before any measured digit is read

Per the physical-sanity rule, a floor and a ceiling from first principles for
the ep8 decode cell at 400 Gbit/s, before any frozen cell is run:

- **Floor.** Each of the 48 collectives must move 14,336 endpoint bytes at
  50 GB/s, which is 286,720 ps, and must cross the fabric once, which the
  backend prices at 2,000,000 ps. No collective can beat 2,286,720 ps, so the
  network term cannot be below 109,762,560 ps, and the step cannot be below
  compute plus that, 208,707,291 ps, in the `off` arm.
- **Ceiling.** With nothing overlapping, the step is compute plus the sum of
  48 serialized collective services. In the `cross` arm each collective adds
  49,487,789 ps of fixed cost, so the step cannot exceed
  98,944,731 + 48 x (2,286,720 + 49,487,789) = 2,584,121,163 ps. Anything above
  that is a defect in the model, the harness or the reading.
- **Covariate.** Halving the link rate doubles only the serialization term.
  At ep8 decode that term is 286,720 of 2,286,720 ps per collective, i.e.
  12.5 percent of the network term and 6.6 percent of the step, so halving
  bandwidth must move the `off` step by roughly 6.6 percent and not by 2x. The
  ep8 prefill cell has 2,293,760 of 4,293,760 ps in serialization, i.e. 53
  percent of the network term, so the same halving must move it much further,
  by roughly 36 percent. If prefill and decode respond by the same factor, the
  model is not doing what it claims regardless of how well any single number
  matches.
- **System plausibility.** 208.7 us per decode step for a 24-layer,
  1024-hidden toy MoE is fast but not absurd for a fluid null network. The
  `cross` arm's 2.584 ms for the same step implies about 387 steps per second
  for a model this small, which is implausibly slow against published MoE
  serving behavior, and the 2.375 ms of pure fixed cost inside it is 92 percent
  of the step. That is the honest reading of an upper-bound arm built from a
  ring transfer, and it is why the arm is published as a bracket edge rather
  than as a recommended operating point.

## Predicted values

All predictions are `compute + 48 * (2,000,000 + ceil(endpoint_bytes * 1e12 /
link_bytes_per_second)) + 48 * intercept(arm, width)`, evaluated exactly. Every
division is exact at both link rates, so no rounding enters.

| cell | off ps | floor ps | local ps | cross ps |
|---|---|---|---|---|
| ep4 prefill 400G | 343,234,560 | 343,234,560 | 1,099,002,576 | 1,497,260,496 |
| ep4 decode 400G | 260,667,977 | 260,667,977 | 1,016,435,993 | 1,414,693,913 |
| ep8 prefill 400G | 305,036,434 | 305,036,434 | 1,751,181,826 | 2,680,450,306 |
| ep8 decode 400G | 208,707,291 | 208,707,291 | 1,654,852,683 | 2,584,121,163 |
| ep4 prefill 200G | 437,606,400 | 437,606,400 | 1,193,374,416 | 1,591,632,336 |
| ep4 decode 200G | 272,464,457 | 272,464,457 | 1,028,232,473 | 1,426,490,393 |
| ep8 prefill 200G | 415,136,914 | 415,136,914 | 1,861,282,306 | 2,790,550,786 |
| ep8 decode 200G | 222,469,851 | 222,469,851 | 1,668,615,243 | 2,597,883,723 |

## Fatal guards, void and never scored

A violated fatal guard voids the run for closure purposes. These are never
reported as a fraction and never enter any denominator.

- **G1 default identity.** A sink configured with no envelope and no profile
  produces exactly the `off` cell: identical `StepResult` values, identical
  GOAL artifact SHA-256 digests, identical flow counts, and an empty
  `collective_timing_outcomes` list. The default path is untouched.
- **G2 floor identity.** The `floor` arm reproduces the `off` arm exactly:
  identical `step_latency_ps`, identical `completed_at_ps`, identical GOAL
  artifact digests. It differs only by publishing a collective timing record
  whose `propagation_reference_ps` is 2,000,000, whose envelope id is
  `intra-node-fixed-cost-v1`, whose arm is `lower`, and whose every base
  latency is 0.
- **G3 all-remote placement.** Every cell reports
  `nvlink_directed_bytes == 0` and `nvlink_service_ps == 0`. This is the
  precondition that makes G2 and the exact-oracle rows meaningful, because the
  arms also select the profile's endpoint bandwidth and that only matters when
  NVLink-local bytes exist.
- **G4 collective structure.** Every cell reports exactly 48 semantic
  collectives, every one a pairwise ALL-TO-ALLV whose participant count equals
  the cell's expert-parallel width, and whose critical endpoint load equals the
  frozen structural value in the table above.
- **G5 one base charge per collective.** For every active arm, the sum of
  `collective_base_latency_ps` equals `48 * intercept(arm, width)` and no
  semantic collective receives more than one base charge.
- **G6 envelope well-formedness.** Every shipped envelope has
  `lower(W) < upper(W)` strictly at every supported width, both arms share the
  same endpoint bandwidth and the same source payload interval so the bracket
  isolates the fixed cost, both arms carry provenance, and every width of every
  shipped profile is band-anchored.
- **G7 unanchored width fails closed.** A participant width absent from a
  profile's table raises before any GOAL artifact or backend process exists,
  for the new profiles exactly as for the existing one.
- **G8 backend health.** Every simulated step reports backend quiescence and
  routing mode `uniform`.
- **G9 endpoint envelope.** Every collective's critical endpoint load lies
  inside the active profile's envelope for its width. The width-4 ceiling is
  393,216 bytes and the width-8 ceiling is 458,752 bytes; the frozen fixture's
  largest load is 114,688 bytes, so the fixture is deliberately far from the
  ceiling that premise 8 says a 34-token prefill nearly reaches.

No guard in this list is declared survivable. Any violation voids the run.

## Exact-oracle rows, entailed and therefore unscored

Given G1 to G9, the implementation at `simllm/backends/step_sink.py:1151` to
`:1159` reduces to

```
latency(arm) = compute + sum_over_collectives(fabric) + 48 * intercept(arm, W)
```

with `fabric` independent of the arm, because the arm changes only the base
latency and the endpoint bandwidth, and G3 forces the endpoint bandwidth to be
inert. The following rows therefore cannot fail once the guards hold. They are
checked, and a violation is fatal, but they are not scored and they are not
counted anywhere near the behavioral denominator.

- **E1 exact additivity.** `latency(arm) - latency(off)` equals
  `48 * intercept(arm, W)` exactly, to the picosecond, for every cell.
- **E2 arm ordering.** `off == floor < local < cross` in every cell.
- **E3 compression direction.** The 200G-to-400G ratio is weakly decreasing
  across `off`, `floor`, `local`, `cross` in every cell. Adding an equal
  positive constant to the numerator and the denominator of a ratio greater
  than one moves it toward one; this is algebra, not evidence.
- **E4 flip direction.** Once the `off` ep4-to-ep8 ratio exceeds one and
  `intercept(4) < intercept(8)`, some active arm must produce a ratio below the
  `off` ratio. Also algebra.

**The pre-freeze entailment answer, stated plainly.** With a closed-form
manifold underneath, nearly everything this study could report is entailed once
the guards hold. Scoring E1 to E4 would inflate a pass count with arithmetic. So
they are not scored. What is genuinely at risk is only this: whether the fluid
backend and the byte model actually produce the absolute values predicted from
bytes, link rate and the documented 2.000 us propagation, and whether the
ratios built from them land where predicted. Those are predictions about a
simulator this study does not control, made before it was run, and they can be
refuted. The scored set is three families, and it is small on purpose.

## Scored behavioral relations

Three families. For each, the entailment question is answered: given the guards
above, can this relation fail?

### S1, the napkin bound on the absolute values

Every one of the 32 measured step latencies lies within **0.05 percent** of its
predicted value in the table above.

*Can it fail?* Yes. The prediction uses only endpoint bytes, link rate and the
documented propagation constant. The fluid backend could add per-flow framing,
MTU quantization, per-artifact setup or a different propagation at 200 Gbit/s,
any of which would move a step outside a 0.05 percent band. The disclosed probe
deviated by 0.002 percent on one nearby cell, which is why 0.05 percent is
chosen rather than 0.5 percent, but one cell is not eight and the probe used no
arm at all.

### S2, the width ordering across arms

The ep4-to-ep8 decode-step ratio, at each link rate, lies within **0.02
percent** of these predicted values:

| link | off | floor | local | cross |
|---|---|---|---|---|
| 400G | 1.248964 | 1.248964 | 0.614215 | 0.547456 |
| 200G | 1.224725 | 1.224725 | 0.616219 | 0.549097 |

The reported conclusion, which follows from these rows and is not separately
scored, is that the arm envelope of this ratio is [0.547456, 1.248964] at
400 Gbit/s and [0.549097, 1.224725] at 200 Gbit/s, and that both brackets
contain 1. The sign of the expert-parallel width ordering is therefore not
determined by the available evidence.

*Can it fail?* Yes, and not only through S1. The band is 0.02 percent, which is
tighter than the 0.05 percent S1 allows each latency separately, precisely so
that this family is not entailed by S1: two latencies can each sit inside their
own 0.05 percent band while their ratio leaves a 0.02 percent band, whenever the
two deviations point in opposite directions. Independently, the `off` ratio
could come out below 1, in which case there is no ordering to invert and the
headline claim is refuted outright.

### S3, the bandwidth-sensitivity compression

The 200G-to-400G step ratio lies within **0.02 percent** of these predicted
values:

| cell | off | floor | local | cross |
|---|---|---|---|---|
| ep4 decode | 1.045255 | 1.045255 | 1.011606 | 1.008339 |
| ep4 prefill | 1.274949 | 1.274949 | 1.085870 | 1.063030 |
| ep8 decode | 1.065942 | 1.065942 | 1.008316 | 1.005326 |
| ep8 prefill | 1.360942 | 1.360942 | 1.062872 | 1.041075 |

*Can it fail?* Yes, for the same reason as S2, and additionally because the
prefill rows encode the covariate check from the napkin section. The prediction
is that prefill responds to halving the link rate roughly five times as strongly
as decode in the `off` arm, because prefill puts 53 percent of its network term
in serialization against decode's 12.5 percent. If prefill and decode respond
alike, the model is not pricing serialization the way it claims, and this family
fails whatever S1 does.

## Evidence classes, never summed

Four classes are reported separately and their counts are never added:

1. Run configurations: 16 cells, 32 simulated steps.
2. Fatal guards: G1 to G9, reported as held or violated, never as a fraction.
3. Exact-oracle rows: E1 to E4, entailed, checked, unscored.
4. Scored behavioral relations: S1, S2, S3. The headline is the genuine-risk
   fraction over these three families only.

## Registered residual IDs

Allocated range for this work is TRAF-36 to TRAF-39, plus the reconciliation of
TRAF-32. COMP-33 is not used. The following registrations are decided before
the run so that the outcome cannot influence which IDs appear:

- **TRAF-32** is registered in `docs/modules/traffic.md` under its originally
  declared meaning, with an explicit disclosure that it is being registered late
  and by a different change from the one that declared it. Its live clause is
  the envelope-width clause of
  `examples/composed_step_budget_v1/expectations.md:184`; the double-charge
  clause of `:381` is closed by construction, because
  `StepCollectiveTimingOutcome` already raises when a semantic collective
  receives more than one base charge, and the mapping is recorded in the entry.
- **TRAF-36** carries the real cross-node per-collective fixed-cost measurement
  that would replace the provisional-transferred profile, with its identifying
  observable named.
- **TRAF-37** carries the one-propagation over-count described above.
- **TRAF-38** carries arm selection outside the all-remote fluid path, where
  the arm also changes the endpoint bandwidth and the bracket stops isolating
  the fixed cost.
- **TRAF-39** carries the ALL-REDUCE to ALL-TO-ALLV operation-shape transfer,
  which is now a first-class contributor to the bracket width.

No ID is registered for an adjacent improvement or for future work beyond these.

## Closure scope

This study closes nothing. It registers TRAF-32 rather than closing it. If a
fatal guard is violated the run is reported as void with findings and every
listed ID stays open.

## Chronology commitment

This document and `expectations.json` are the only files in their commit. The
implementation, the study runner, the tests and the results land in later
commits, and the git log will show that order. If any prediction here turns out
to be wrong, the prediction stays as written and the deviation is explained in
the results; nothing here is edited after a measurement.
