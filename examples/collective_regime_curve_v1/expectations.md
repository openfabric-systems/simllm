# Collective regime curve v1 expectations

## Freeze scope and chronology

This is the expectations-only record for TRAF-43, replacing the single-slope
collective serializer with a regime-aware form. It is committed before the
implementation, before any error is computed against the measured sweeps, and
before any test asserting an accuracy bound exists.

**Chronology, stated plainly.** The measurements this form is validated against
already exist: the
[A100 hardware envelope](../a100_hardware_envelope_v1/RESULTS.md) and the
[GH200 hardware envelope](../gh200_hardware_envelope_v1/RESULTS.md) were both
measured and published before this file. The accuracy check below is therefore
a **post-specified regression check**, not a pre-registered prediction, and it
must never be described as the latter. Two things are genuinely frozen ahead of
the evidence they will be judged on, and only these may be called that:

- the 15 percent acceptance bar, which was registered in TRAF-43 at commit
  `321113c` when no regime-aware form existed and no candidate had been fitted;
- the anchor rule and the held-out split in this file, which are chosen on
  structural grounds below and fixed before any interpolation error is
  computed.

If the held-out error exceeds the bar, that is reported as a failure of this
candidate form and TRAF-43 stays open. The anchor rule is not to be retuned
after seeing the errors; a second candidate needs a second freeze.

## What is wrong today

`CollectiveLatencyProfile.endpoint_serialization_ps` charges
`endpoint_bytes / bandwidth_bytes_per_second` at every payload, one slope
everywhere. Both hardware studies measured what that costs. Anchoring the
intercept at the measured 8-byte floor and the slope at the 1 GiB algorithm
bandwidth, the resulting two-parameter model is exact at both anchors and
optimistic at every payload between them:

| Machine and width | Worst signed error | At payload |
|---|---:|---:|
| A100 width 2 | -50.8 percent | 1 MiB |
| A100 width 4 | -45.8 percent | 2 MiB |
| GH200 width 2 | -48.1 percent | 1 MiB |
| GH200 width 4 | -32.7 percent | 2 MiB |

The cause is identified and is the same on both machines: bus bandwidth is
still climbing across that window, reaching half its asymptote only at 2.45 and
4 MiB at width 2, and 8.24 and 8 MiB at width 4.

## The candidate form

A payload-indexed bandwidth curve attached to the existing width-indexed
latency table. The existing profile already documents why its latency table is
a table rather than a law: the capture identifies widths but not an
interpolation law. The same reasoning applies here with more force, because the
mechanism behind the ramp is NCCL selecting protocols and channel counts in
discrete steps, so a smooth closed form would be inventing a law the hardware
does not follow.

Frozen design:

- A new frozen `CollectiveBandwidthCurve` carries `curve_id`, an increasing
  tuple of `(endpoint_bytes, bytes_per_second)` anchors, and a provenance
  record. At least two anchors are required and endpoint bytes must strictly
  increase.
- The curve stores **serialization** bandwidth, meaning endpoint bytes divided
  by the measured time with the width's base latency already removed. Storing
  total algorithm bandwidth instead would double-count the latency floor when
  the caller adds `base_latency_ps` on top.
- Interpolation is geometric on both axes: between two anchors the logarithm of
  bandwidth is linear in the logarithm of endpoint bytes. Below the first
  anchor the first anchor's bandwidth is used; above the last, the last. The
  quantity spans more than three decades, so a geometric law is the natural one
  and a linear one would be dominated by the largest anchor.
- `CollectiveLatencyProfile` gains an optional `bandwidth_curves` field, a
  tuple of `(participant_count, curve)`. A width with no curve keeps today's
  behavior exactly.
- `endpoint_serialization_ps` keeps its upward `_ceil_div` rounding so
  timestamps stay integral and monotone.

## Frozen anchor rule and held-out split

Anchors sit at the collective payloads that bound the three regimes both
studies observed, plus two points spacing the ramp roughly evenly in log space:

| Anchor | Collective payload | Regime it pins |
|---|---:|---|
| 1 | 8 KiB | top of the flat-latency regime |
| 2 | 256 KiB | start of the ramp |
| 3 | 4 MiB | ramp, near the half-bandwidth payload |
| 4 | 64 MiB | ramp, approaching the asymptote |
| 5 | 1 GiB | flat-bandwidth regime |

Five anchors for 22 measured payloads. Endpoint bytes for an all-reduce of
payload `S` at width `n` are `2(n-1)S/n`, the nccl-tests bus convention, so the
anchors are placed at those endpoint byte counts.

The held-out set is every measured all-reduce payload from 1 KiB to 512 MiB
inclusive that is not an anchor: 1, 2, 4 KiB, 16, 32, 64, 128 KiB, 512 KiB,
1, 2, 8, 16, 32 MiB, 128, 256, 512 MiB. That is 16 held-out payloads per curve
and four curves, A100 and GH200 at widths 2 and 4.

The 8-byte payload is excluded from the held-out set. It is the anchor for the
latency floor itself, its serialization term is nanoseconds against a
microsecond floor, and a relative error on a near-zero term is not meaningful.

## Acceptance

- **E-1** For each of the four curves, the worst signed relative error of
  modeled total time against measured total time over the held-out payloads is
  at most 15 percent in magnitude. This is the bar TRAF-43 registered.
- **E-2** At every anchor payload the modeled total time reproduces the
  measured time to within the upward rounding of one picosecond, by
  construction.
- **E-3** The modeled error is not systematically one-signed over the held-out
  set the way the single-slope model was: at least one held-out payload per
  curve has positive error and at least one has negative error.
- **E-4** Modeled time is strictly increasing in endpoint bytes for every
  curve, so a larger collective never completes sooner.
- **E-5** The worst held-out error of this form is at least three times smaller
  in magnitude than the worst error of the single-slope form on the same
  payload set and curve.

## Bypass, and what must not move

- **B-1** A profile with no `bandwidth_curves` returns exactly the picosecond
  values it returns today from `endpoint_serialization_ps`,
  `total_service_ps` and every method that reads them, for every width and
  endpoint byte count in the existing tests.
- **B-2** The three shipped profiles, `b200-nccl-2.27-local-v1`,
  `collective-fixed-cost-floor-v1` and
  `b200-nccl-2.27-cross-node-provisional-v1`, carry no curve in this change, so
  every accepted artifact and every accepted timestamp stays byte-identical.
- **B-3** `CollectiveFixedCostEnvelope` continues to require its arms to share
  `bandwidth_bytes_per_second`, and gains the matching requirement that arms
  agree on whether a curve is present.

B-1 and B-2 are configuration-forced identity assertions. They are fatal when
violated and unscored: a violated bypass voids the change rather than costing a
point.

## Scoring

The scored denominator is 5, the expectations E-1 through E-5, evaluated
independently for four curves and reported per curve as well as in total.
Bypass guards B-1 through B-3 are unscored and fatal.

## What this change does not do

- It ships no new calibrated profile and selects no curve by default. Landing
  the measured A100 and GH200 curves as selectable arms is TRAF-44.
- It does not touch the width-indexed latency table, the propagation reference,
  or the cross-node transfer.
- It changes no reported TTFT or TPOT, because nothing selects a curve.
- It does not claim the interpolation law is physical. The law is declared,
  the anchors are measured, and the held-out error is what is defended.
