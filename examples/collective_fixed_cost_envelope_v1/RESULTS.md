# Results: the per-collective fixed-cost envelope

Frozen expectations: [expectations.md](expectations.md) and
[expectations.json](expectations.json), committed as `9c568b5` before any arm,
any profile and any run existed. The implementation landed at `8395617` and the
runner at `d18bb7a`. The git log shows that order, so the pre-registration is
genuine rather than reconstructed.

## Verdict

**Attempt 1 is void. Attempt 2 passed 3 of 3 scored families with all nine
fatal guards held.**

- **Attempt 1** (`93978d1`, run root `envelope-v1-final`): void. Fatal guard G4
  was violated. Every scored family passed and every other guard held, but a
  violated fatal guard voids the run, so attempt 1 closes nothing and its
  numbers are reported as findings only.
- **Attempt 2** (`22490db`): all nine fatal guards held, all four exact-oracle
  rows held, and 3 of 3 scored families passed.

The scored fraction is **3 of 3 genuine-risk families**. The fatal guards are
reported as held or violated and never as a fraction. The 32 frozen simulated
steps, the 8 guard simulated steps, the 4 exact-oracle rows and the 3 scored
families are four separate evidence classes and their counts are not added.

## Why attempt 1 was void, and what changed

G4 checked that the fixture emits exactly 48 pairwise ALL-TO-ALLV collectives
per step at the cell's participant width and the frozen endpoint load. The
guard compared the observed collective-kind inventory, which the runner built
as a sorted list of `(collective, algorithm_hint)` tuples, against the frozen
list of two-item lists. In Python a tuple never equals a list, so the guard
could not hold for any data at all. Every fact it was checking was in fact
correct in attempt 1: 48 collectives, kind `all-to-allv` with hint `pairwise`,
participant counts exactly `[4]` and `[8]`, endpoint loads exactly 98,304 /
12,288 / 114,688 / 14,336 bytes, and provider compute exactly 152,862,720 /
152,871,497 / 98,935,954 / 98,944,731 ps, all equal to the frozen structural
inputs.

Attempt 2 changed the runner in exactly two places, neither of which touches a
modeled behavior, a cell, a prediction, a tolerance or any other oracle:

1. `_structural_inventory` now emits each collective kind as a list rather than
   a tuple, so the G4 comparison compares like with like.
2. A `results_summary` projection was added so a trackable document can carry
   the verdict and the measured latencies without the bulk per-cell artifact
   manifests.

Every raw measurement was preserved. Attempt 2 reproduced all 32 frozen step
latencies and all 8 guard step latencies from attempt 1 to the picosecond, and
every GOAL artifact digest in all 20 cells is byte-identical between the two
attempts. That is the check that the harness repair changed nothing measurable.

## Chronology, disclosed in full

- `9c568b5` freezes expectations. Its commit contains only `expectations.md`
  and `expectations.json`.
- `8395617` adds the envelope, the provenance record, the two new profiles, the
  sink configuration surface and the unit tests.
- `d18bb7a` adds the study runner.
- `93978d1` adds the E3 and E4 checks that the freeze registered and the first
  runner draft had omitted. The first launch was stopped after 7 of its 21
  cells so that those two rows could be added; it produced no result document,
  and none of its per-cell numbers were read before the change was made.
- `22490db` repairs the G4 comparison after attempt 1 measured it as violated.
  This is a post-observation harness change and is labeled as one. It changes
  no modeled behavior and no prediction.
- `2d9f730` removes the backend binary's filesystem location from the trackable
  results projection, after the attempt-2 run had already been recorded. It is
  a second post-observation change to the reporting layer only: it drops one
  machine-specific string, touches no measurement, no oracle and no tolerance,
  and the tracked summary was regenerated from attempt 2's unchanged raw
  document.
- The attempt-2 run record reports `git_status_clean: false`. The two dirty
  paths at the moment the environment was captured were `docs/modules/traffic.md`
  and this results document, both being drafted while the cells executed.
  Neither is read by the study: the runner consumes only `expectations.json` and
  the `simllm` package, and both were committed at `22490db` before the run
  started.
- One measurement was taken **before** the freeze and is disclosed in
  [expectations.md](expectations.md): a single ep8 decode step at 400 Gbit/s
  with `context_length=65` and no arm, which returned 209,194,608 ps against a
  209,198,811 ps napkin. It is not one of the frozen cells. The S1 tolerance was
  chosen after seeing that 0.002 percent deviation, and the expectations
  document says so.

## Napkin bounds against the measurement, three independent angles

The bounds below were written into the freeze before any frozen cell ran.

### Angle one: compute and memory physics

A decode step cannot beat its own weight traffic. Per rank the fixture holds,
per layer, 6,291,456 bytes of attention projections plus 3,145,728 bytes per
resident expert, plus a 100,663,296-byte LM head:

| cell | resident weight bytes | floor at 0.7 x 8 TB/s | provider estimate | above floor |
|---|---:|---:|---:|---:|
| ep8 | 553,648,128 | 98.866 us | 98.936 to 98.945 us | 0.07 to 0.08 percent |
| ep4 | 855,638,016 | 152.793 us | 152.863 to 152.871 us | 0.05 percent |

The compute term is memory-bandwidth bound and sits a tenth of a percent above
the physical floor at both widths. This also explains the width ordering the
`off` arm produces without any appeal to the network: at ep4 each rank holds
eight experts instead of four, so it moves 1.545 times the weight bytes, while
its critical endpoint load is only 0.857 times as large. Compute favors the
wider expert-parallel group; the per-collective fixed cost favors the narrower
one. That opposition is the whole subject of this study.

### Angle two: network and serialization physics

A collective cannot beat its own endpoint serialization plus one propagation
delay. The fluid backend returned exactly one picosecond above that bound at
every one of the eight (width, phase, link rate) points:

| cell | endpoint bytes | link | closed form ps | measured ps | delta |
|---|---:|---|---:|---:|---:|
| ep4 prefill | 98,304 | 400G | 3,966,080 | 3,966,081 | +1 |
| ep4 decode | 12,288 | 400G | 2,245,760 | 2,245,761 | +1 |
| ep8 prefill | 114,688 | 400G | 4,293,760 | 4,293,761 | +1 |
| ep8 decode | 14,336 | 400G | 2,286,720 | 2,286,721 | +1 |
| ep4 prefill | 98,304 | 200G | 5,932,160 | 5,932,161 | +1 |
| ep4 decode | 12,288 | 200G | 2,491,520 | 2,491,521 | +1 |
| ep8 prefill | 114,688 | 200G | 6,587,520 | 6,587,521 | +1 |
| ep8 decode | 14,336 | 200G | 2,573,440 | 2,573,441 | +1 |

The one picosecond is the backend's whole-picosecond ceiling on its own fluid
solution, and it is constant rather than proportional, which is exactly what a
rounding term should look like.

### Angle three: end-to-end system plausibility

The `off` arm puts an ep8 decode step at 208.7 us, which is 4,792 decode steps
per second for a single request against a 553.6 MB per-rank weight footprint.
No real B100-class deployment reaches that, and the reason is not in this
study: the SGLang chain currently prices the per-step host cost at zero, and
small-kernel inefficiency is not modeled either. The `off` arm is optimistic by
construction and this study does not repair that.

The `cross` arm puts the same step at 2.584 ms, which is 387 steps per second,
91.9 percent of it pure surcharge. That is the pessimistic edge and it should
be read as one: the transfer retains the source capture's ring algorithm where
NCCL selects a tree for small cross-node messages, and it retains NCCL's
per-launch cost where a persistent-kernel MoE dispatch avoids most of it. The
honest reading is that a real deployment sits inside the bracket, probably
nearer the `local` arm than the `cross` arm, and that nothing in this
repository says where.

## The residual against the closed form, fully explained

Every one of the 32 frozen step latencies is below its frozen prediction, by
between 0.0003 and 0.008 percent. Two named mechanisms account for the residual
exactly, with no remainder:

1. The fluid backend's plus-one-picosecond ceiling, worth `+48` ps per step.
2. The sink charges GOAL calc units quantized to whole nanoseconds per layer,
   not the provider's picosecond estimate. At ep8 that is `24 x 4,122 ns =
   98,928,000 ps` against the provider's 98,936,000 to 98,945,000 ps; at ep4 it
   is `24 x 6,369 ns = 152,856,000 ps` against 152,863,000 to 152,871,000 ps.
   The freeze predicted from the provider estimate, which is the term the
   lowerer reports, and the sink charges the rendered term.

Reconstructing every step as

```
step = compute_service_ps + 48 * fabric_service_ps + 48 * surcharge(arm, width)
```

reproduces all 32 frozen step latencies and all 8 guard step latencies to the
picosecond, with zero mismatches. The deviation from the freeze is therefore
not noise; it is one rounding term and one quantization term, both identified.

## Scored behavioral relations

### S1, the napkin bound on the absolute values

Tolerance 0.05 percent, 32 rows, **passed**. The largest relative error was
7.99e-05 at `ep8-decode-400g-off`, a factor of 6.3 inside the band. The errors
are systematically larger on the `off` and `floor` arms than on `local` and
`cross`, which is the expected signature of a fixed absolute quantization
residual measured against a growing denominator.

### S2, the width ordering across arms

Tolerance 0.02 percent, 8 rows, **passed**. Largest relative error 2.10e-05.

| link | off | floor | local | cross |
|---|---:|---:|---:|---:|
| 400G measured | 1.248990 | 1.248990 | 0.614212 | 0.547454 |
| 400G predicted | 1.248964 | 1.248964 | 0.614215 | 0.547456 |
| 200G measured | 1.224748 | 1.224748 | 0.616216 | 0.549095 |
| 200G predicted | 1.224725 | 1.224725 | 0.616219 | 0.549097 |

**The headline.** The `arm_ratio_envelope` of the ep4-to-ep8 decode-step ratio
is [0.547454, 1.248990] at 400 Gbit/s and [0.549095, 1.224748] at 200 Gbit/s.
Both brackets contain 1, with a multiplicative width of 2.28 and 2.23. The sign
of the expert-parallel width ordering is not determined by the evidence this
repository holds. A study that reports "ep8 is 1.25x faster than ep4" and a
study that reports "ep4 is 1.83x faster than ep8" are both reachable from the
same fixture by moving one constant that no measurement pins down.

### S3, the bandwidth-sensitivity compression

Tolerance 0.02 percent, 16 rows, **passed**. Largest relative error 6.92e-06.

| cell | off | floor | local | cross |
|---|---:|---:|---:|---:|
| ep4 decode | 1.045257 | 1.045257 | 1.011606 | 1.008339 |
| ep4 prefill | 1.274954 | 1.274954 | 1.085871 | 1.063030 |
| ep8 decode | 1.065947 | 1.065947 | 1.008317 | 1.005326 |
| ep8 prefill | 1.360951 | 1.360951 | 1.062872 | 1.041075 |

Every measured value is within 7e-06 of its frozen prediction. The compression
is large: at ep8 decode a 2x bandwidth cut moves the step by 6.59 percent under
`off` and by 0.53 percent under `cross`, a factor of 12.4. A study that
concludes "this workload is insensitive to link bandwidth" is reporting the arm
it chose, not the workload.

**The covariate check the freeze demanded.** Prefill must respond far more
strongly than decode because far more of its network term is serialization. At
ep8 the serialization share of one collective is 53.42 percent for prefill
against 12.54 percent for decode, which puts serialization at 36.1 percent and
6.59 percent of the whole step. The measured `off` responses to halving the
link rate were 36.095 percent and 6.5947 percent, each equal to its own
serialization share of the step to five significant figures, and both were
frozen in advance as 1.360942 and 1.065942. Prefill responded 5.47 times as
strongly as decode rather than alike, which is what the model has to do if it
prices serialization the way it claims.

## Fatal guards

Never reported as a fraction. All nine held in attempt 2.

| guard | what it asserted | attempt 1 | attempt 2 |
|---|---|---|---|
| G1 | the plain default equals the `off` arm in step results and GOAL digests, with no timing record | held | held |
| G2 | the `floor` arm equals `off` to the picosecond and to the GOAL digest, differing only by publishing the claim | held | held |
| G3 | every cell reports zero NVLink bytes and zero NVLink service | held | held |
| G4 | 48 pairwise ALL-TO-ALLV collectives at the cell width with the frozen endpoint load | **violated by the oracle's own type comparison** | held |
| G5 | the base charges sum to `48 x surcharge` with at most one per semantic collective | held | held |
| G6 | both shipped envelopes bracket strictly and isolate the fixed cost | held | held |
| G7 | an unanchored participant width raises before any artifact or backend process | held | held |
| G8 | every simulated step is quiescent with routing mode `uniform` | held | held |
| G9 | every endpoint load lies inside the active profile's envelope for its width | held | held |

G2 is the guard that makes the `floor` arm meaningful: it is byte-identical to
the default in this all-remote placement, so the arm publishes a claim rather
than changing a number. G7 used a 16-wide expert-parallel group, which no
profile anchors; the sink raised before writing a single GOAL artifact or
completion file and published no outcome.

## Exact-oracle rows, entailed and unscored

All four held. They are checked because a violation would be fatal, and they
are not scored because they cannot fail once the guards hold.

- **E1** exact additivity: `latency(arm) - latency(off)` equals
  `48 x surcharge(arm, width)` to the picosecond in all 16 cells.
- **E2** arm ordering: `off == floor < local < cross` in every cell.
- **E3** compression direction: the 200G-to-400G ratio is weakly decreasing
  across `off`, `floor`, `local`, `cross` in every cell.
- **E4** flip direction: with the `off` ratio above 1 and `surcharge(4) <
  surcharge(8)`, an active arm produced a ratio below the `off` ratio at both
  link rates.

**The entailment answer, restated against the measurement.** The freeze said
that with a closed-form manifold underneath, nearly everything this study could
report is entailed once the guards hold, and that is exactly what happened: E1
to E4 are arithmetic on a formula the guards pin down, and reporting them as
passes would have inflated a headline with algebra. What was genuinely at risk
was whether the fluid backend and the byte model produce the predicted absolute
values, and whether the ratios built from them land where predicted. Both
turned out to be true, but only after a plus-one-picosecond term and a
nanosecond quantization term that the freeze did not anticipate; those two
consumed most of the S1 band and would have consumed all of a band ten times
tighter. The ratio bands were deliberately set tighter than S1 permits per
latency, so that a ratio could leave its band while both latencies stayed
inside theirs; that did not happen, because the two residual mechanisms are
common-mode and cancel in a quotient.

## What the bracket says about the wave's question

The wave audit found that the SGLang chain charges zero per-collective fixed
cost while calibrated machinery for a nonzero one exists. This study replaces
that binary choice with a measured interval.

Per collective at participant width 8, the surcharge bracket is
[0, 49,487,789] ps and the realized fixed cost bracket, propagation included,
is [2,000,000, 51,487,789] ps, a factor of 25.7. Across a 48-collective step
that is [96,000,048, 2,471,413,920] ps of fixed cost, against a network
transport of 109,762,608 ps and a compute term of 98,928,000 ps at ep8 decode
and 400 Gbit/s. The fixed cost is between 46.0 percent and 95.6 percent of the
step. It is the dominant term in every arm, including the default, and the
default is the only arm that does not say so out loud.

| envelope | width | lower surcharge ps | upper surcharge ps | provisional band at the upper arm |
|---|---:|---:|---:|---|
| intra-node-fixed-cost-v1 | 2 | 0 | 10,722,112 | [10,461,112, 10,983,112] |
| intra-node-fixed-cost-v1 | 4 | 0 | 15,745,167 | [15,398,167, 16,092,167] |
| intra-node-fixed-cost-v1 | 8 | 0 | 30,128,029 | [30,048,029, 30,208,029] |
| cross-node-fixed-cost-provisional-v1 | 2 | 10,722,112 | 13,487,792 | [11,487,792, 17,487,792] |
| cross-node-fixed-cost-provisional-v1 | 4 | 15,745,167 | 24,042,207 | [18,042,207, 36,042,207] |
| cross-node-fixed-cost-provisional-v1 | 8 | 30,128,029 | 49,487,789 | [35,487,789, 77,487,789] |

The cross-node upper arm is labeled **provisional-transferred** and never
calibrated. It carries no cross-node measurement. Its construction, its three
anchors and the three ways it is weakest are stated in
[expectations.md](expectations.md) and reproduced by
`run_study.py --check-only`.

## Limits, stated plainly

- The bracket isolates the fixed cost only in an all-remote placement. An arm
  also selects its profile's endpoint bandwidth, and the sink accepts an arm
  only on `rnic-nn-fluid`. G3 confirms the study stayed inside that regime;
  TRAF-38 owns leaving it.
- Every active arm charges its surcharge on top of a transport that already
  contains one propagation delay, so the realized fixed cost over-counts a
  complete source capture by up to 2.000 us per collective: 6.6 percent at
  width 8, 12.7 percent at width 4, 18.7 percent at width 2. This study did not
  change that, because changing it would move an accepted baseline. TRAF-37
  owns it.
- Every shipped profile applies a ring ALL-REDUCE intercept to pairwise
  ALL-TO-ALLV, which is the only collective the fixture emits. TRAF-39 owns
  that transfer.
- The fixture is deliberately far from the endpoint-byte ceiling, at most
  114,688 bytes against ceilings of 393,216 and 458,752. A mission-scale
  prefill is not, and is rejected at planning time. TRAF-32 owns it.
- The toy geometry is 24 layers at hidden size 1024. The mechanism is exactly
  per-collective and the arithmetic is closed form, so the conclusions carry to
  any layer count, but the absolute microsecond figures do not describe a real
  model.

## Registered residual IDs

Decided in the freeze before the run, so the outcome could not choose them.

- **TRAF-32** is registered in `docs/modules/traffic.md` under its originally
  declared meaning with an explicit late-registration disclosure. Its
  double-charge clause is closed by construction and the mapping is recorded in
  the entry; the envelope-width clause stays live.
- **TRAF-36** carries the real cross-node measurement with its identifying
  observable.
- **TRAF-37** carries the one-propagation over-count.
- **TRAF-38** carries arm selection outside the all-remote fluid path.
- **TRAF-39** carries the ALL-REDUCE to ALL-TO-ALLV operation-shape transfer.

## Closure scope

This study closes nothing. It registers TRAF-32 rather than closing it, and it
registers four new residuals. The measured artifact of record is
[results-summary.json](results-summary.json); the full per-cell document with
artifact manifests is bulk run output and stays outside the repository under
the directory named by `SIMLLM_FIXED_COST_ENVELOPE_RUN_ROOT`.
