# TRAF-69 NVLink flow dynamics: frozen expectations

This is the expectations-only authority for the NV4 flow-dynamics study
directed on 2026-08-27. It is committed before release-aware flow scheduling,
before the study runner, before the first TRAF-69 simulated run and before any
result-dependent edit. The machine-readable authority is
[`expectations.json`](expectations.json). A miss leaves its band unchanged and
publishes a refutation.

The first expectations commit stated the divergence phase as packet admission
plus one link cadence. A preimplementation review corrected it to the literal
phase-3 identity below before any target-code edit or TRAF-69 simulated run.
Both commits remain in history; this amended file is the final pre-run
authority.

## Physical story and evidence boundary

One A100 source turns each flow extent into packets. Four direct NVLink3 links
carry one ordered pair. The direct NV4 mesh has no switch queue. The receiver
then drains arriving wire bytes and delivers each extent in sequence. Flows on
one ordered pair share the same packet grants. Independent incast senders have
independent ordered pairs but meet at one receiver ingress.

TRAF-70 measured two effective plateaus: TX endpoint egress is
160,795,737,454 byte/s and RX ingress is 207,101,921,876 byte/s. It also
confirmed request and response direction, extent-sequence reassembly and
per-extent delivery. Those five fields are measured evidence. The direct-mesh
pass-through switch is structural evidence, not a measurement.

The score's eleven unchanged internals comprise ten declared candidates plus
the structural switch. The ten candidates are maximum packet payload, header,
links per peer, per-link rate, bond policy, credit unit, credits per
destination, RX buffer, credit-return latency, queue scope and the candidate
values represented by that complete scored catalog. Every rate, transition and
CDF consumes some of them, so every figure must say so. No result from this
study promotes them.

The scored packet is 256 payload bytes plus a 16-byte header, or 272 wire
bytes. The candidate serializers therefore take:

- endpoint admission: `ceil(272e12 / 160795737454) = 1,692 ps`;
- one 25 GB/s link: `ceil(272e12 / 25000000000) = 10,880 ps`;
- measured RX ingress: `ceil(272e12 / 207101921876) = 1,314 ps`.

Four candidate links provide 100 GB/s of raw ordered-pair service, below the
measured TX plateau. Packet overhead makes the payload ceiling
`100 * 256 / 272 = 94.1176470588 GB/s`. No simulated payload rate may exceed
the ceiling belonging to its topology.

## Flow scheduler contract

The new path is explicitly selected for this study. The existing default path
remains the identity off mode and must retain its canonical bytes and every
timestamp exactly.

The selected path schedules active extents round-robin at maximum-packet
boundaries. A newly released flow joins after any grant already made at the
same timestamp. Credits are consumed by ordered-pair packet visit rather than
by an extent-local sequence number. Packets from independent sources are
ordered at the receiver by upstream arrival time with stable topology order
for ties. Each extent's sequence stays strictly increasing. These rules define
the analytical transition identities below; they are not fitted after a run.

## Overall three-flow schedule

Flows A, B and C use source 0 and receiver 1. They release at 0, 11,141,120 and
22,282,240 ps. Their byte targets are 4 MiB, 2 MiB and 1 MiB, respectively.
The targets reverse the join order, so A must finish last. The schedule uses
696,320 ps bins, exactly 64 candidate maximum-packet link serializations.
Each point is the payload delivered inside one fixed bin divided by the bin
width. There is no rolling average, interpolation or smoothing.

The center of the one-, two- and three-flow steady bands is the pair payload
ceiling divided by the active count. The half-width is exactly one 256-byte
packet per 696,320 ps bin, 0.3676470588 GB/s. The figure must shade these
frozen quantization bands and mark both joins and every completion.

## Exact 1-to-2 convergence identity

The incumbent sends 1 MiB. The 256 KiB joiner releases at the incumbent's
packet-256 grant, 696,320 ps after the incumbent release. The incumbent owns
that boundary grant. The joiner becomes the next round-robin selection.

The joiner has a free credit. With 256 credits, the slot it consumes recycled
far earlier than this join. The declared 200,000 ps credit-return constant is
therefore checked and contributes exactly zero. The first competing payload
becomes receiver-visible after:

`T_open = C_wait + A_packet + S_link + S_rx`

`T_open = 0 + 1,692 + 10,880 + 1,314 = 13,886 ps`.

This is the NVLink analogue of the reference slide's control plus two-delay
identity: every causal term is named, the inactive credit term remains visible,
and the observed value must equal 13,886 ps with zero tolerance. The incumbent
rate moves from 94.1176470588 to 47.0588235294 GB/s. The displayed transition
uses raw 10,880 ps bins with no smoothing; rate scoring uses the separately
frozen quantization band rather than reading pixels.

## Exact 2-to-1 divergence identity

Two flows start together. The remaining flow targets 1 MiB and the departing
flow targets 64 KiB. Time-to-target begins at the departing flow's final
receiver delivery. It ends when the receiver observes the fifth subsequent
remaining-flow delivery, which closes the first complete four-link solo
cadence.

The credit slot is already free. RX serialization is common to the two
receiver-visible anchors and cancels. The identity is:

The departing 64 KiB flow contains 256 packets. Its last packet is phase 3 of
the four-link cycle. The fifth remaining-flow delivery closes at the same
phase in the second following link cadence. The phase-aware identity is:

`T_target = C_wait + 2*S_link - 3*A_packet + delta(S_rx)`

`T_target = 0 + 21,760 - 5,076 + 0 = 16,684 ps`.

The observed value must equal 16,684 ps with zero tolerance. The remaining
flow moves from 47.0588235294 back to 94.1176470588 GB/s. The plot again uses
raw 10,880 ps bins and no smoothing.

## Flow-completion-time CDF freeze

The byte ladder is 256 B, 1 KiB, 4 KiB, 16 KiB, 64 KiB, 256 KiB and 512 KiB.
The first rung is one maximum payload packet. The final rung is the scored
profile's composed-validation extent, the largest extent this study claims the
published envelope supports.

The frozen seeds are 1103, 1907, 2801, 3691, 4513, 5381, 6271, 7159 and 8053.
There are nine seeds. Each seed produces 12 waves per sender at each degree and
size. The nominal inter-wave interval is three quarters of the independently
derived wave service. Seeded release jitter spans plus or minus one 10,880 ps
candidate link serialization. This creates scheduling jitter without adding a
stochastic hardware term to the scored profile.

For each seed, the empirical CDF is the fraction of that seed's flows whose
flow completion time is at most x. The common x grid is the sorted union of
observed values for one degree and size. The solid line is the pointwise
arithmetic mean across nine seed CDFs. The shaded band is the pointwise minimum
through maximum across the same nine CDFs. The seed count and min-max
definition must appear on the figure.

Each of the 21 degree-by-size rungs receives four checks: monotonic CDF,
terminal value exactly one, mean seed p50 in its frozen band and mean seed p95
in its frozen band. The numeric bands are in `expectations.json`. Their lower
edge is the independent single-flow serialization floor. Their upper edge is
a deliberately conservative finite-wave drain bound. A failure at any rung is
reported beside that rung and never hidden in an aggregate pass count.

## Incast degrees and ceilings

The physical maximum is three sources into one receiver on a four-GPU node.
The schedule flow at every degree is 512 KiB per sender. Raw receiver-goodput
bins are 696,320 ps with no smoothing.

Payload ceilings retain the candidate 256/272 packet-efficiency factor:

| Degree | Raw ceiling | Payload ceiling | Binding module |
|---:|---:|---:|---|
| 1 | 100.000000 GB/s | 94.117647 GB/s | candidate ordered-pair links |
| 2 | 200.000000 GB/s | 188.235294 GB/s | two candidate ordered pairs |
| 3 | 207.101922 GB/s | 194.919456 GB/s | measured RX ingress |

Degrees one and two cannot be called receiver-limited. Degree three is the
first and only physical incast ceiling in this topology. Each simulated
aggregate gets its own measured-versus-ceiling row and may exceed neither its
raw nor payload ceiling.

The published 281.65 GB/s row is one sender fanning out to three receivers. It
is not an incast receiver ceiling. The scored TX plateau plus candidate packet
efficiency predicts 151.337165 GB/s for that separate topology, so the frozen
10 percent check expects a `REFUTED` verdict. The miss must remain visible and
must not be used to raise the incast ceiling.

## Figure contract

The presentation follows the existing rnic-cn join and exit grammar: stacked
time-aligned rate panels, event markers, fair-share guides, raw step traces and
short timing verdicts in the panel title or annotation. The publication set is:

1. one three-panel overall, convergence and divergence figure;
2. one seven-panel FCT ladder figure;
3. one two-panel schedule and FCT figure for each incast degree.

Every figure is written as PDF and PNG with POSIX-rendered relative paths.
Every figure states the nine-seed min-max definition where applicable, says
that rate bins are raw and unsmoothed, and carries the mixed-evidence disclosure
for measured, candidate and structural fields. The final raster renders are
inspected for clipping, overlap, hidden bands, unreadable legends and lines
stuck to panel borders.

## Preservation and verdict

The study inherits all 43 artifacts in the prior flagship preservation class
and adds the deployment-frontier publication plus the complete TRAF-70 scored
profile and score publication, for 60 byte-locked artifacts total. No prior
runner is invoked and no prior record or figure is rewritten.

Any failed fatal guard voids the study. Both transition identities must match
exactly. Steady rates, all 21 CDF rungs and all three incast ceilings are scored
against their frozen bands. A nonfatal miss is a published refutation. TRAF-69
closes only when every panel and rung has a verdict, both formats render,
preservation holds, all fatal guards pass and the result is published without
widening a band.
