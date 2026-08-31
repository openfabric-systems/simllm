# Matched-seam frontier figure addendum

This append-only addendum refines the figure contract of
`expectations.md` on maintainer direction received after that freeze. It
adds no scored family, widens no band, and changes no predicate. The
scored families S, R, F, M, D and W stand exactly as frozen.

## The directed figure

Three curves in the external tool's published visual grammar, log-log,
`tokens/s/user` on x, `tokens/s/gpu` on y, better direction up and to
the right:

1. Their Pareto curve, from the tracked external tables.
2. Our curve with network contention fully off (the ideal rung).
3. Our curve with network contention on (the packet rung).

Curves 1 and 2 are the matched-seam demonstration: same measured
database, same composition target, contention priced by neither side.
They are expected to coincide, and Family R and Family F are what decide
whether they actually do.

Curve 3 is the mechanism. The gap between curves 2 and 3 is annotated
with an arrow labelled as the term their model class does not price.

## What the annotation may and may not say

The arrow's label must name the mechanism precisely. The contention-off
arm is not faster than light: the ideal rung charges propagation latency
L. What it omits is receiver-side serialization under fan-in. It allows
several senders to deliver into one receiver at full rate at the same
time, which exceeds the receiver's ingress bandwidth. That is a
bandwidth-limit violation, and it is measurable: the frozen mechanism
envelope from `frontier_ladder_v1` and `loggopsim_acceptance_v1` puts
eight-into-one incast at 7.678 to 8.110 times the ideal rung, while
contention-free point-to-point legs differ by only 1.000 to 1.020.

The label therefore states the unpriced term and the measured envelope.
It does not say "faster than light", and it does not assign blame beyond
the modelling fact that this class of planner prices no contention.

## Honesty conditions on the arrow

The arrow is conditional on measured separation, not on the expectation
of it:

- If Family M2 holds (at least one candidate at a packet-to-ideal step
  quotient of 1.02 or more), the arrow is drawn at the candidate with the
  largest measured separation and labelled with that measured quotient,
  the workload it belongs to, and the evidence class of both arms.
- If Family M2 is refuted (no candidate reaches 1.02), the arrow is NOT
  drawn. The figure instead states the measured maximum separation
  plainly and says that at this workload the mechanism does not
  materially move the frontier. The eight-times envelope stays a
  fan-in-regime statement about other schedules and is not transplanted
  onto this figure.
- If curves 1 and 2 do not coincide within the frozen Family F bracket,
  the figure says so on its face. An arrow attributing a gap to
  unmodelled contention is only legible when the contention-off arms
  already agree, so a Family F miss is disclosed in the caption rather
  than drawn over.

## Caption requirements

One sentence stating that both arms price from the same imported
measured database, so curve differences are composition and mechanism
rather than kernel timing. The evidence class of every series named.
The regime scope stated: which rungs and fan-in degrees the envelope
covers, and that the annotated separation is this workload's measured
value, not a general claim.
