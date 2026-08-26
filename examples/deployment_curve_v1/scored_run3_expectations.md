# CORE-54 third scored flagship expectations

Status: **EXPECTATIONS_ONLY**. This freeze precedes the third scored runner,
the third fit, every third-run held-out read, every third-run result, and the
third publication figure. The machine-readable authority is
`scored_run3_expectations.json`.

## Inherited execution and disclosure rulings

The second run's configurations are unchanged. Prefill and decode remain
separate experiments: four eight-GPU H100 nodes at EP32 for prefill and nine
eight-GPU H100 nodes at EP72 for decode. The 13-node, 104-rank simultaneous
render remains a structural comparator and is not called the 96-GPU
disclosure. The PLACE-5 16-prefill plus 40-decode deployment remains
second-legend what-if context.

The live execution remains one eight-rank prefill engine plus one eight-rank
decode engine under the serialized parent clock. Per-node observables retain
the exact factors four and nine. CORE-58 stable identity is unchanged. The
offered-load grids, integer-picosecond interarrivals, SGL-38 remote-KV
projection, exact candidate keys, packet handoff, and all runtime limits are
inherited byte-for-byte from the second run.

The simulated-MTP row remains `BLOCKED` on COMP-72. Its numeric disclosure is
not read or imputed. The standard-decode calibration miss remains
unattenuated: it is owned by the registered 61-over-4 depth-extrapolation
residual under policy rule five, not by benchmark bias.

## Physics and overlap-exposure envelope

Let `C` be the frozen per-prompt compute service and `P` the clean COMP-75
packet service. The physics-only form is the second run's perfect-overlap
floor:

`T0(C, P) = max(C, P)`.

TRAF-67 cleanly reproduces the TRAF-66 two-child ceiling:

`T2(C, P) = max(C, P) + min(C, P) / 2`.

The third run therefore admits exactly one new fitted constant, exposed
fraction `f` in the closed interval `[0, 1/2]`, and uses

`Tf(C, P) = max(C, P) + f * min(C, P)`.

This is exactly the calibration-clean bracket between the two derived forms.
The floor is the clean COMP-75 reproduction, the ceiling is the clean TRAF-67
repetition, and no envelope edge is inferred from an anchor. The initial point
is `f = 0`. The fit may move it only inside the bracket and may read only the
1K prefill and standard-decode calibration anchors.

The inherited intra-node collective surcharge remains in the tunable list,
with `[0, 30,128,029]` ps as its closed envelope. Its successor paths have
zero applications, so its objective is flat and the smaller-value tie rule
selects zero.

## Derived communication refinements

The EP32 layout derives seven same-node destination peers and 24 fabric peers
from four nodes with eight ranks each, excluding the source rank. COMP-75
already prices those partitions separately: same-node bytes use its analytic
NVLink endpoint serializer, fabric bytes use `rnic-nn`, and each phase takes
their maximum. The third run explicitly retains this zero-anchor-input
refinement and does not fit it.

The merged three-module packet candidate is not substituted. Its profile is
explicitly A100 NVLink3 candidate evidence, while the disclosure target is
H100. Applying it here would introduce an unsupported cross-architecture
mechanism rather than a derived refinement. The second run's H100-scoped
analytic locality path therefore stays frozen.

## Attenuation factor and derivation

Exactly one benchmark-bias factor is admitted. It corrects the disclosure's
stated in-distribution expert-balance condition against simLLM's uniform
top-k destination-rank incidence with same-destination deduplication.

There are 256 logical experts, eight experts on each of 32 ranks, and top-k
eight routing without replacement. For one named rank,

`p = [C(256, 8) - C(248, 8)] / C(256, 8)`

`  = 939691952959 / 4138017124000`.

The expected number of unique destination ranks is `32p`, or
`939691952959 / 129313035125 = 7.266799917353`. The balanced benchmark
reference has top-k eight destination opportunities. The frozen attenuation
factor is therefore

`a = E[unique destination ranks] / 8`

`  = 939691952959 / 1034504281000`

`  = 0.908349989669110`.

It multiplies the physics-plus-boundary throughput for all three prefill
anchors. One factor touches three anchors, satisfying the policy's
non-vacuity rule. The derivation uses zero anchor numeric input.

The uncertainty comes from the exact covariance of destination-hit
indicators. For two named ranks,

`q = 1 - 2 C(248, 8)/C(256, 8) + C(240, 8)/C(256, 8)`

`  = 528807826297 / 11379547091000`.

Thus

`Var(U) = 32 p(1-p) + 32*31 (q-p^2)`

`       = 102764354724981604704634 / 183940471585634321421875`.

Across 16,384 routed tokens, the standard error of `U/8` is
0.000729932406642. The frozen two-standard-error interval, rounded outward at
15 decimal places, is `[0.906890124855826, 0.909809854482394]`.

The exact-length packing candidate is not admitted. Both the disclosure and
this configured study already use exact input lengths packed to 16,384 tokens
per rank. No relative correction remains, and there is no independently
published or mechanically derived per-request overhead magnitude. Looking at
the anchors to manufacture one is forbidden.

## Three pre-fit prediction layers

These are predictions, not comparisons. No 2K or 4K published throughput is
present in this freeze. `Physics` propagates the COMP-75 integer-rounding
record. `Boundary` additionally propagates the complete `f` envelope.
`Attenuated` additionally propagates the factor interval.

| Anchor | Physics band | Physics plus boundary band | Physics plus boundary plus attenuation band |
|---|---:|---:|---:|
| SGLang EP32 prefill, 1K | [57,332.324492, 57,332.324550] | [44,164.630548, 57,332.324550] | [40,052.467312, 52,161.513856] |
| SGLang EP32 prefill, 2K | [57,332.324492, 57,332.324550] | [43,744.208139, 57,332.324550] | [39,671.190380, 52,161.513856] |
| SGLang EP32 prefill, 4K | [57,332.324492, 57,332.324550] | [42,504.142847, 57,332.324550] | [38,546.587414, 52,161.513856] |
| SGLang EP72 standard decode | [8,949.759685, 8,949.759685] | [8,949.759685, 8,949.759685] | [8,949.759685, 8,949.759685] |
| SGLang simulated-MTP decode | BLOCKED | BLOCKED | BLOCKED |

The pre-fit point is the perfect-overlap floor in the first two layers. The
attenuated point is 52,077.816412 tokens/s/node for each prefill row before
fit. Boundary bands are deliberately broad because they carry the complete
clean physical bracket, not a post-fit confidence interval. The COMP-74
distribution contribution remains the explicit zero-width
`insufficient-replays` placeholder and makes no zero-variance claim.

## Fit, one-shot score, and figure

The fit reads only `sglang_prefill_1k` and `sglang_decode_standard`. It solves
the prefill point equation for `f`, clamps the result to `[0, 1/2]`, applies
the smaller-value tie rule, and selects zero surcharge on the flat successor
path. The serialized, content-addressed fit freezes before any held-out access.

The scorer may then read the 2K and 4K prefill anchor values exactly once. It
publishes signed errors in all three layers and decides the 5 percent verdict
only from the attenuated point errors, as required by the policy. Bands do not
convert a point miss into a pass. The MTP row remains unread and blocked.

The two-column publication figure uses throughput increasing rightward and
inverse per-token delay increasing upward. Panel a carries the ordered curves
and propagated record, constant, boundary, and attenuation bands. Panel b
places physics-only, boundary-fit unattenuated, and attenuated prefill
projections against the published anchors. The figure includes the DeepSeek
H800 profile and PLACE-5 16-prefill plus 40-decode what-if as a second legend,
states that the scored scope is prefill under the declared benchmark-bias
model, shows the unattenuated comparison, retains the MTP blocker and decode
calibration miss, and has no watermark.

Every artifact in the JSON preservation-lock class must remain byte-identical.
A guard violation voids the run instead of becoming a lost or amended score.
