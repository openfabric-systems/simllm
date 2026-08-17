# Collective regime curve v1 results

The reviewed state is `CANDIDATE REFUTED, 16 of 20`. The frozen five-anchor
rule clears the 15 percent bar at participant width 4 on both machines and
misses it at width 2 on both, at the same payload, for the same identified
reason. TRAF-43 stays open.

The mechanism this study built is sound and is landed: the bypass is exact,
anchors reproduce to within a picosecond, and modeled time is strictly
increasing everywhere. What failed is the anchor rule, and the freeze forbids
retuning it after seeing the errors. A second candidate needs a second freeze.

## Chronology

Both hardware sweeps were measured and published before this model existed, so
the accuracy check here is a **post-specified regression check**, not a
pre-registered prediction, and it is not described as one anywhere in this
record. Two things were frozen ahead of the evidence they are judged on:

- the 15 percent bar, registered in TRAF-43 at commit `321113c`, when no
  regime-aware form existed and no candidate had been fitted;
- the anchor rule and held-out split, frozen at commit `2b969e3` before any
  interpolation error was computed.

The [expectations](expectations.md) also fixed in advance what to do with a
failure, which is what made the failure reportable rather than tempting: the
anchor rule is not retuned, the candidate is refuted, and the task stays open.

## What was measured against

The A100 and GH200 all-reduce sweeps at widths 2 and 4, four curves, each with
five anchor payloads and 16 held-out payloads from 1 KiB to 512 MiB.

## Outcome per curve

| Curve | Worst held-out error | At payload | Single slope worst | Improvement | Verdict |
|---|---:|---:|---:|---:|---|
| A100 width 2 | -27.76 percent | 1 MiB | -50.8 percent | 1.83x | fails E-1, E-5 |
| A100 width 4 | -11.86 percent | 1 KiB | -45.8 percent | 3.86x | passes |
| GH200 width 2 | -27.35 percent | 1 MiB | -48.1 percent | 1.76x | fails E-1, E-5 |
| GH200 width 4 | -9.95 percent | 1 KiB | -32.7 percent | 3.29x | passes |

E-2, E-3 and E-4 pass on all four curves: anchors reproduce to better than
1e-9 relative, held-out errors carry both signs rather than the single sign the
incumbent model showed, and modeled time is strictly increasing in endpoint
bytes.

## Why width 2 fails

The frozen form interpolates serialization bandwidth geometrically between
anchors, which can only produce a monotone curve between two anchors. The
measured serialization bandwidth is not monotone.

Removing each width's latency floor and recomputing endpoint bytes over the
residual time gives the serialization bandwidth the model is trying to
represent. It rises, then dips, then rises again:

| Payload | A100 w2 | A100 w4 | GH200 w2 | GH200 w4 |
|---:|---:|---:|---:|---:|
| 128 KiB | 36.57 | 41.29 | 55.58 | 74.11 |
| 256 KiB | 33.91 | 76.04 | 57.41 | 125.39 |
| 512 KiB | 32.61 | 77.58 | 52.90 | 141.00 |
| 1 MiB | **27.09** | 82.58 | **45.05** | 186.96 |
| 2 MiB | 38.10 | **82.47** | 66.75 | **175.04** |
| 4 MiB | 50.13 | 99.58 | 82.78 | 197.87 |

Bandwidth in GB/s; the bold entries are local minima.

At width 2 the dip is 26 percent below the local peak on the A100 and 22
percent on the GH200, and both bottom out at exactly 1 MiB. The frozen anchors
sit at 256 KiB and 4 MiB, on either side of it, so any interpolation between
them passes over the dip and predicts a faster collective than the hardware
delivers. That is precisely the -27 percent seen, and it is the same defect
on both machines. At width 4 the dip is shallow, under 7 percent, and lands at
2 MiB where the 4 MiB anchor is close enough that interpolation absorbs it.

The dip itself is the more interesting result. It is reproducible across two
NVLink generations, two link counts and two host architectures, it moves from
1 MiB at width 2 to 2 MiB at width 4, and it has the shape of a protocol
transition: a switch to an algorithm with a higher asymptote but a larger
pipeline fill costs bandwidth right at the crossover. This study did not
instrument NCCL's protocol selection, so that mechanism is a hypothesis, not a
finding. What is a finding is that intra-node ring serialization bandwidth is
**not monotone in payload**, and any interpolating model with anchors sparser
than the transition will miss it in exactly the region where
tensor-parallel activation exchanges live.

## What landed anyway

`CollectiveBandwidthCurve` and the optional `bandwidth_curves` field on
`CollectiveLatencyProfile` are landed, because the mechanism is correct and is
the substrate any second candidate needs. It is inert:

- no shipped profile carries a curve, so `b200-nccl-2.27-local-v1`,
  `collective-fixed-cost-floor-v1` and the cross-node provisional profile are
  byte-identical in behavior;
- a width with no curve charges exactly the flat slope it always charged, which
  is asserted directly against the shipped profile across its whole endpoint
  byte envelope;
- `CollectiveFixedCostEnvelope` now refuses arms that disagree about curves,
  the same way it already refuses arms that disagree about the flat slope.

Nothing selects a curve, so no reported TTFT or TPOT moves.

## What the next candidate needs

- Anchors placed at the observed transition rather than on a log-spaced grid.
  The transition is at 1 MiB at width 2 and 2 MiB at width 4 on both machines,
  so a rule that adds the local-minimum payload as an anchor is the obvious
  next candidate, and it must be frozen before its error is computed.
- A held-out split that keeps at least one payload inside the dip, otherwise
  the new anchors trivially reproduce the region they were placed in and the
  check stops measuring generalization.
- If a protocol-transition anchor rule needs NCCL's own selection thresholds,
  those are observable from `NCCL_DEBUG=INFO` at comm init and would make the
  anchor rule mechanistic rather than empirical.
