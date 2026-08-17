# Merlin ss-dragonfly fabric calibration v1 expectations (TRAF-51)

## Freeze scope and chronology

This is the expectations-only record for the wave-19 calibration
comparison registered as TRAF-51: the hosted htsim Slingshot-class
fabric (`htsim_ss_dragonfly`, submodule pin `89b7a5a`) against the
byte-locked Merlin capture dataset of
[merlin_fabric_flow_capture_v1](../merlin_fabric_flow_capture_v1/RESULTS.md)
(manifest SHA-256
`a6b7e61e294d87d76ce69ee7042e15c2eade99bbc8789e296377615d2bd4af88`),
on the solo-stream, incast and join families, with the measured socket
host-stack floor separated from fabric serialization. It is committed
before any simulation cell of this study has run on the declared Merlin
instance.

Chronology, disclosed exactly:

- The measured side of every comparison in this file is already
  published and was read before this freeze: the capture dataset is
  byte-locked in this repository and its RESULTS are public. Nothing on
  the measured side is blind. What is blind at this commit is every
  simulated quantity: no `htsim_ss_dragonfly` invocation on the
  declared Merlin instance has happened.
- Because the measured side is known, this freeze computes, discloses
  and tables the composed-versus-measured arithmetic in advance. Rows
  whose verdict is already determined by that arithmetic, conditional
  only on the simulation behaving as its own scored rows require, are
  labeled conditional rows and are counted in a separate denominator
  from the genuinely blind simulation rows. They are calibration
  consistency checks, not blind predictions, and the RESULTS must
  repeat that label.
- Pre-freeze work that already happened, all unscored: the submodule
  pin bump and build (448 of 448 backend tests pass), the backend
  sanity determinism precondition (four sanity arms of
  `experiments/ss_dragonfly_sanity` rerun on this build, each twice:
  byte-identical between repeats and byte-identical to the archived
  wave-18 scored CSVs for `incast_1_minimal`, `incast_2_adaptive`,
  `incast_8_adaptive`, `join_adaptive`), and one harness feasibility
  probe of the 9038-byte framing flags on the backend study topology
  `p2a2h1g3_200g.topo` (10 ms, degree 1; clean exit, delivered equals
  injected, payload arithmetic exact). The probe did not touch the
  Merlin instance. Its role is the same as the capture study's
  discovery: it makes the frozen invocation grammar known-runnable, and
  it is not a scored measurement.
- The steady-window measured anchors and targets quoted below were
  computed from the tracked dataset series before this freeze, using
  the capture study's own frozen stage-steady definition (final 20
  seconds of a stage, 1-second bins). Where the dataset's published
  stats files carry the same quantity, the recomputation agrees exactly
  (all six stage steady rates match `stats/*_stats.json` to full
  precision). The analyzer must re-derive all of them from the tracked
  bytes at scoring time.

No job in this study touches the Merlin cluster (its ssh is down and
nothing here needs it), no framework runs, and no TTFT or TPOT is
reported. This study can therefore close no task whose acceptance names
an end-to-end metric; it addresses TRAF-51's calibration clause only.

## The composition rule, stated before any simulated number is read

The captured per-chunk completion series contain two stages the model
must keep separate: the endpoint host-stack service (kernel socket
transport, NCCL proxy, user-space loop) that the fabric simulator does
not model, and fabric serialization plus delivery that it does. The
capture study measured that most of a chunk's life is the endpoint
stack (solo 8 MiB chunk service p50 4.7 to 4.8 times the wire floor),
and TRAF-51 requires the stack charged at the endpoint, never absorbed
into link parameters. The rule below is frozen before any simulated
number exists; the simulation supplies its fabric terms, and the
composed quantities are then mechanical.

Definitions, with B = 8,388,608 bytes (the capture's frozen chunk):

- `R_fab(f; F)`: the per-flow steady payload goodput the fabric
  instance delivers for the port mapping of family F, measured from
  the `htsim_ss_dragonfly` cell that mirrors that mapping, driven open
  loop at line rate (an upper-stress drive: the real sources offer at
  most a fifth of line rate, so a fabric that shows no cross-flow
  interference at line rate cannot show it at the captured loads).
- `T_fab_rate = B / C_p`: the fabric chunk service time at line rate,
  where C_p is the per-port payload ceiling under the derived framing
  (napkin values below; the simulation must confirm both).
- `T_fab_lat`: the fabric delivery time of one 8 MiB chunk injected at
  line rate into an idle fabric, first bit to last byte (hand value
  below; the simulation must confirm it).
- `H_rate(P)` for a source-destination pair P: the measured solo
  stage-steady rate of that pair, from the pair's solo anchor cell.
  This is a measured calibration input, not a fitted parameter.
- `H_time(P) = p50_solo(P) - T_fab_lat`: the endpoint host-stack floor
  per chunk, the measured solo steady-window per-chunk completion p50
  minus the modeled fabric chunk delivery time. This subtraction is the
  separation TRAF-51 demands: the endpoint floor is an explicit,
  separate, reported term.

Composed quantities per family F and flow f with pair P(f):

- Steady rate (pipelined stages; the binding stage sets the rate):
  `R_comp(f; F) = min(H_rate(P(f)), R_fab(f; F))`.
- Steady per-chunk completion p50 (valid only while the stack is the
  binding stage): `p50_comp(f; F) = H_time(P(f)) + T_fab_lat(f; F)`,
  where `T_fab_lat(f; F)` is the family's fabric chunk delivery time
  for that flow. For every captured family in this study the port
  mapping gives each flow its own source port and its own destination
  port, so the frozen fabric-side prediction is
  `T_fab_lat(f; F) = T_fab_lat(solo)` and the composed p50 of a flow
  equals its pair's solo anchor p50. The simulation's role in those
  rows is to confirm the premise (no interference at the mapping), and
  that cancellation is disclosed here rather than discovered later.
- Aggregates are sums of composed per-flow rates over active flows;
  Jain fairness is computed from the composed per-flow rates. Both are
  therefore derived rows, not independent scored rows (see the
  entailment section).
- Convergence: the composed model contains exactly two settling
  mechanisms, fabric settling (bounded by the scored fabric rows) and
  nothing on the endpoint side, because H is a static floor. The
  composed settling time for any family is therefore the fabric's own,
  and the composed model is structurally incapable of producing a
  multi-second endpoint transient. This is declared now: the measured
  119-second i2 convergence transient is outside the expressiveness of
  this composition rule, is pre-attributed (evidence below) to endpoint
  stack dynamics, and no band is registered on it.

Validity condition of the rule, checked before composed rows are read:
`T_fab_rate < B / H_rate(P)` and `T_fab_lat < p50_solo(P)` for every
scored pair, i.e. the fabric is never the binding stage at captured
loads. If a simulated fabric term violates this, the composition
switches meaning (the fabric would bind) and every composed row is
re-derived under the min rule with that stated; the latency rows would
then be unavailable (an open queue at saturation has no additive p50)
and reported as such.

## The declared topology instance

[merlin_a100_singleswitch_v1.topo](merlin_a100_singleswitch_v1.topo)
carries the full per-parameter provenance in its own comments, which
are part of this freeze. Summary: single-switch degenerate dragonfly
p = 20, a = 1, h = 0, g = 1 (all twenty 200G Cassini ports of the five
A100 nodes on one Rosetta-class switch), which is exactly what
five-node socket-level discovery can determine and no more; the
per-port rate (200 Gbit/s, measured), the wire framing (9038-byte wire
packet with 8948-byte payload, derived from the measured MTU 9000 and
the measured 0.64 percent sender-side counter overhead through the
standard TCP/IPv4-with-timestamps/Ethernet framing model), propagation
and pipeline latencies (declared, with the published Rosetta 350 ns
pipeline; insensitivity at chunk scale stated in the file), buffer and
seed (declared defaults). Host numbering: host = node * 4 + port over
gpu101..gpu105.

The mapping from captured flows to instance ports, from the dataset's
sender tx counters and NCCL interface lines: every captured flow rides
exactly one source port (measured per cell), the two i2/j2x flows ride
distinct source nodes and land on distinct destination hsn devices of
gpu102 (the reviewed capture record's reading of `nccl_interfaces.txt`,
with the rx counters consistent), so every scored family maps to flows
with pairwise distinct source ports and pairwise distinct destination
ports on one switch. A single flow's two NCCL channels may split its
destination-side arrival across two ports (seen in s1); the sim maps
each flow to one destination port, which is the conservative choice for
contention and changes nothing for solo cells.

At this shape the fabric's progressive adaptive routing is structurally
unreachable (every delivery is same-router, route class Undecided), and
the congestion quantizer and advertisement knobs are unreachable with
it; the calibration claim of this study accordingly cannot and does not
extend to multi-switch routing behavior.

## Napkin bounds, before any simulated value is read

| Quantity | Derivation | Value |
|---|---|---:|
| wire serialization per packet, 200G | 9038 B * 8 / 200e9 | 361,520 ps |
| per-port payload ceiling C_p | 200e9 * 8948 / 9038 | 198.0084 Gbit/s = 24.7511 GB/s |
| packets per 8 MiB chunk | ceil(8,388,608 / 8,948) | 938 |
| T_fab_rate | B / C_p | 338.919 us |
| T_fab_lat, first chunk, idle fabric | 937 * 361,520 + (361,520 + 300,000 + 350,000 + 361,520 + 300,000) | 340,417,280 ps = 340.417 us |
| same, 4160/4096 framing | 2047 * 166,400 + 1,282,800 | 341,903,600 ps |
| same, 100G variant | 937 * 723,040 + 2,396,080 | 679,884,560 ps |
| C_p at 4160/4096 framing | 200e9 * 4096 / 4160 | 196.9231 Gbit/s = 24.6154 GB/s |
| framing-shift rate ratio | 24.6154 / 24.7511 | 0.99452 |
| 100-us bin packet quantization | ceil(100,000,000 / 361,520) + 1 packets | at most 2,487,544 payload B per bin, 1.0050 * C_p |
| shared-buffer fill at 2C into C (echo) | 4 MiB / 25 GB/s wire | 167.8 us |
| worst buffer drain | 4 MiB / 25 GB/s | 167.8 us, far under the 50 ms drain window |
| separation inequality, tightest pair | T_fab_lat / p50_solo(gpu105 to gpu102) | 340.417 / 1596.855 = 21.3 percent, stack floor 3.69x the fabric term |

No simulated rate may exceed C_p at its framing in any full 100-us bin
beyond the packet quantization bound; a violation is a conservation
defect and fatal, never hardware truth.

## Measured anchors and targets (published bytes, recomputed at scoring)

Stage-steady definition throughout: final 20 seconds of a stage,
1-second bins, destination clock; steady-window p50 means the p50 of
per-chunk completion deltas whose completions fall in that window.

Anchors (calibration inputs; their cells become unscorable anchors and
appear in no scored row of this study):

| Anchor | Source cell | Steady rate GB/s | Steady-window p50 us | H_time us | Endpoint share |
|---|---|---:|---:|---:|---:|
| A-1 gpu105 to gpu102 | s1-stream, [280, 300) s | 4.9824 | 1596.855 | 1256.438 | 78.7 percent |
| A-2 gpu103 to gpu102 | j2x-join stage 0, [40, 60) s | 3.8193 | 2198.644 | 1858.227 | 84.5 percent |

A-2 comes from the capture's post-specified, unscored j2x cell. Using
it as a calibration input is legitimate (it is byte-locked data whose
guards all hold) and is disclosed: no scored relation of the capture
study reads it, and this study uses it as the only solo evidence for
the gpu103 source pair. Discovery burst rates are barred as anchors:
the capture study itself established that burst and sustained rates
differ by tens of percent with pair-dependent sign, and withdrew the
generalization. This freeze does not resurrect it.

Comparison targets (steady windows [160, 180) s of their cells):

| Target | Cell, flow | Steady rate GB/s | Steady-window p50 us |
|---|---|---:|---:|
| i2 flow 0 (gpu103) | i2-incast | 3.6553 | 2239.049 |
| i2 flow 1 (gpu105) | i2-incast | 4.8931 | 1600.271 |
| j2x stage-1 flow 0 (gpu103) | j2x-join | 4.0249 | 2135.626 |
| j2x stage-1 flow 1 (gpu105) | j2x-join | 4.8486 | 1722.941 |

## Cells and the run matrix

Binary: `htsim_ss_dragonfly` built from the pinned submodule commit
`89b7a5a`, SHA-256
`5075021a6af762e914d782a1c69c1633d9b084f767ac82d9b6931bd33f69f787`,
recorded per run and re-verified before scoring. Every cell runs twice
with identical arguments (repeat determinism guard). Outputs stay in
the bulk run directory outside Git; a packaging commit locks the small
derived summaries.

Common flags: `-routing adaptive -wire_bytes 9038 -header_bytes 90`
(framing per the declared instance; the 4160-framing cell overrides
exactly these two), topology
`merlin_a100_singleswitch_v1.topo` unless named otherwise. The
harness's incast pattern assigns sources as consecutive hosts after the
receiver; at this single-switch shape every ordered host pair is
isomorphic (identical links, one switch, routing degenerate), so the
harness's pair stands for any captured pair, and the symmetry cell
turns that argument into a checked relation rather than an assumption.

| Cell | Command core | Duration, bins | Role |
|---|---|---|---|
| cal-solo-a | `-pattern incast -receiver 5 -degree 1 -duration_ps 200000000000 -bin_ps 1000000` | 200 ms, 1 us | the fabric term (EX-1, BE-1, BE-2, BE-3) |
| cal-solo-b | same, `-receiver 17` | 200 ms, 1 us | port symmetry (ST-1) |
| cal-solo-4160 | as cal-solo-a with `-wire_bytes 4160 -header_bytes 64` | 200 ms, 1 us | framing shift (EX-3, BE-6) |
| cal-solo-100g | as cal-solo-a on `merlin_a100_singleswitch_v1_100g.topo` | 200 ms, 1 us | link-rate scaling (EX-2, BE-7) |
| cal-echo-2 | `-pattern incast -receiver 5 -degree 2 -duration_ps 100000000000 -bin_ps 100000000` | 100 ms, 100 us | overload positive control (BE-4) |
| cal-echo-4 | same, `-degree 4` | 100 ms, 100 us | overload positive control (BE-5) |

Duration scaling rule, frozen: simulated windows are scaled from the
captured 180-to-300-second windows down to 200 ms (solo) and 100 ms
(echo) under the rule that every cell holds at least 500 chunk
completions per active-and-served flow and at least 1000 steady
100-us bins. Cost of the scaling, stated: a 200 ms window cannot
exhibit any model dynamics slower than tens of milliseconds. The
fabric model's slowest mechanism at this shape is the shared-buffer
fill and drain at 167.8 us; there is no mechanism with a longer
timescale to truncate, so the scaling costs fidelity only against
phenomena the model does not contain, which is exactly the endpoint
dynamics the composition rule already declares out of scope.

The join family needs no join-pattern simulation cell: the harness's
join pattern is structurally inapplicable at a single-switch instance
(it requires degree strictly below the router count, which is 1, and
its per-router source walk collapses onto the receiver), and under the
distinct-port mapping a staggered join is the superposition of solo
flows, so its fabric term is the solo term. This is declared here, the
code-level reason recorded, and the multi-receiver harness gap is
registered as a backend follow-up at closure.

The echo cells deliberately drive the one regime the backend sanity
studies already analyzed as a harness artifact (admission-phase capture
under synchronized open-loop overload at a shared egress). They are
positive controls connecting this build and instance to that evidence,
and they are never compared against the captured incast cells: the
captured i2 is a distinct-port family at a fifth of line rate per
stack, not a shared-egress line-rate overload. Re-measuring the known
artifact as a calibration miss is exactly the failure mode this
paragraph exists to prevent.

## Fatal guards, void and never scored

Any violation voids the affected run for the purpose of closing
anything; guards are never reported as a fraction. No survivable guard
is declared in this study.

- FG-1 binary and pin identity: the binary's SHA-256 equals the value
  above and the submodule HEAD is `89b7a5a` at run time; the topology
  file SHA-256s are recorded and match the freeze-committed files.
- FG-2 backend sanity determinism precondition: the four rerun sanity
  arms are byte-identical between repeats on this build (already
  satisfied pre-freeze; evidence retained in the bulk directory and
  re-checked into the run manifest).
- FG-3 per-cell determinism: both repeats of every cell produce
  byte-identical CSV output.
- FG-4 conservation: (a) no destination port's aggregated full 100-us
  bin exceeds the packet quantization bound at its framing (napkin
  table); (b) in every solo cell, delivered packets equal injected
  packets and dropped equals zero; (c) delivered payload bytes equal
  delivered packets times the payload size exactly.
- FG-5 execution sanity: harness exit status zero and the harness's
  own quiescence validation passes in every run.
- FG-6 dataset integrity: the capture dataset manifest hashes to
  `a6b7e61e...` and every dataset file the analyzer reads re-verifies
  against its manifest entry at scoring time; the repository's dataset
  lock test is green at the scoring commit.

## Scored simulation relations (blind at this freeze)

The entailment question is answered per relation: given the fatal
guards, can this relation fail? Every row below can; none is implied by
a guard.

Exact-oracle rows (hand-derived values, stated above, never copied from
a run):

- EX-1 cal-solo-a: the 1-us bin in which cumulative delivered payload
  first reaches B ends at exactly 341,000,000 ps (hand delivery time
  340,417,280 ps). Can fail: any extra or missing per-packet or
  per-hop charge in the fabric's store-and-forward arithmetic moves the
  crossing bin.
- EX-2 cal-solo-100g: the same crossing bin ends at exactly
  680,000,000 ps (hand value 679,884,560 ps): serialization terms
  double at half rate, propagation and pipeline do not. Can fail: any
  latency mis-attributed to a rate-dependent or rate-independent term
  breaks exactly this row.
- EX-3 cal-solo-4160: the same crossing bin ends at exactly
  342,000,000 ps (hand value 341,903,600 ps). Can fail: per-packet
  overheads that do not scale with the packet count would leave the
  crossing where EX-1 put it.

Behavioral rows:

- BE-1 cal-solo-a steady rate: delivered payload over [50, 150) ms
  divided by 0.1 s lies in [0.99, 1.0005] * C_p. Can fail: pacing,
  drop, or accounting defects.
- BE-2 cal-solo-a chunk delta: the p50 of successive chunk-boundary
  crossing deltas (1-us bin ends) lies in [335, 345] us. Can fail:
  any queueing the model wrongly inserts on an uncontended path.
- BE-3 settling: in every solo cell, every aggregated 100-us bin with
  index 1 through 1999 holds within 1 percent of that cell's C_p. The
  fabric contributes no transient beyond one bin. Can fail: a warmup,
  oscillation, or periodic stall in the model would show directly.
- BE-4 cal-echo-2 positive control, all three clauses: mean aggregate
  over the final 50 bins in [0.90, 1.005] * C_p; exactly one flow
  starved (every one of its final 50 bins below 0.02 * C_p); dropped
  packets nonzero. Can fail: if the single-switch VoQ admission shares
  fairly under synchronized overload, the starvation clause fails and
  the artifact boundary would be topology-dependent, a real finding.
- BE-5 cal-echo-4 positive control: mean aggregate over the final 50
  bins in [0.90, 1.005] * C_p; at least one flow starved as above; the
  largest flow holds at least 0.5 * C_p over the final 50 bins;
  dropped packets nonzero. Can fail: as BE-4.
- BE-6 framing shift: the cal-solo-4160 steady rate over the
  cal-solo-a steady rate lies in [0.9935, 0.9955] (predicted 0.99452).
  Can fail: any framing-independent rate term breaks the ratio.
- BE-7 rate scaling: the cal-solo-100g steady rate lies in
  [0.99, 1.0005] * (C_p / 2). Can fail: as EX-2, in the rate domain.

Structural row:

- ST-1 port symmetry: the (bin, delivered payload) sequences of
  cal-solo-a and cal-solo-b are identical (columns other than the
  host identities byte-equal). Can fail: any per-port asymmetry in the
  instance or model.

Simulation denominator: 3 exact + 7 behavioral + 1 structural = 11
rows, reported per class and never summed with the conditional rows.

## Conditional composed rows (arithmetic disclosed at freeze)

Frozen bands: composed over measured within [0.85, 1.15] for steady
rates (motivated by the capture's own same-pair cross-cell
repeatability: the gpu103 pair moved 4.5 to 10.1 percent between its
anchor and its two-flow cells), within [1/1.2, 1.2] for steady-window
p50s (rates plus distribution-shape drift), [0.90, 1.10] for the
join-unharmed ratio, and 1 s for composed settling. Direction is
deliberately unsigned except CO-5's registered meaning (the
established flow must not lose materially at a join).

Each row's entailment status is stated: the measured and anchor values
are published bytes, so given the simulation rows above (which fix
R_fab = C_p per flow and T_fab_lat at its hand value), every CO verdict
below is determined now and disclosed now. They can change after this
freeze only through the simulation side (a failed EX/BE/ST row changes
R_fab or T_fab and re-derives every composed value under the min
rule). The RESULTS must report them as exactly this: consistency
confirmations under a frozen rule, not blind predictions.

| Row | Quantity | Composed | Measured | Ratio | Frozen verdict given sim rows |
|---|---|---:|---:|---:|---|
| CO-1 | i2 flow 0 steady rate | 3.8193 | 3.6553 | 1.0449 | pass |
| CO-2 | i2 flow 1 steady rate | 4.9824 | 4.8931 | 1.0183 | pass |
| CO-3 | j2x stage-1 flow 0 steady rate | 3.8193 | 4.0249 | 0.9489 | pass |
| CO-4 | j2x stage-1 flow 1 steady rate | 4.9824 | 4.8486 | 1.0276 | pass |
| CO-5 | join-unharmed, flow 0 stage 1 over stage 0 | 1.0000 | 1.0538 | | pass (measured in [0.90, 1.10]) |
| CO-6 | i2 flow 0 steady-window p50 | 2198.644 us | 2239.049 us | 0.9820 | pass |
| CO-7 | i2 flow 1 steady-window p50 | 1596.855 us | 1600.271 us | 0.9979 | pass |
| CO-8 | j2x stage-1 flow 0 steady-window p50 | 2198.644 us | 2135.626 us | 1.0295 | pass |
| CO-9 | j2x stage-1 flow 1 steady-window p50 | 1596.855 us | 1722.941 us | 0.9268 | pass |
| CO-10 | staggered-join composed settling | fabric settling from BE-3 | 0.0 s | | pass iff BE-3 holds |

Conditional denominator: 10 rows over about 9 independent measured
quantities (CO-5 reads CO-3's measurement against a different
reference; the four p50 rows share distributions with the four rate
rows but test a different functional of them; CO-10's measured side is
the single j2x convergence value).

## Derived, excluded and refuted-by-construction quantities (unscored)

- Aggregates and Jain fairness are functions of the scored per-flow
  rows and are reported, never scored: if CO-1 and CO-2 pass their 15
  percent bands, the i2 aggregate is inside 15 percent and the Jain
  difference is inside 0.018 by arithmetic, so scoring them would
  inflate the denominator with unlosable rows. Freeze-computed values,
  reported for the record: i2 aggregate composed 8.8017 versus
  measured 8.5484 (ratio 1.0296), composed Jain 0.98284 versus
  0.97947; j2x stage-1 aggregate composed 8.8017 versus 8.8735
  (0.9919), composed Jain 0.98284 versus 0.99146; the TRAF-51
  headline ratio, i2 aggregate over solo: composed 1.7666 versus
  measured 1.7157 on the uniform stage definition (published 1.7129 on
  the capture freeze's mixed definitions, both readings recorded
  there).
- The anchor rows (s1 rate and p50, j2x stage-0 rate and p50) are
  circular under this composition by construction and appear in no
  scored set.
- The i2 simultaneous-start convergence transient, measured 119 s, is
  the pre-declared out-of-expressiveness row: the composed model
  cannot produce it (static endpoint floor), the fabric will be shown
  by BE-3 to contribute none of it or BE-3 fails, and the attribution
  to endpoint stack dynamics rests on the capture's own controlled
  contrast (the same pairs on the same stacks settle in 0.0 s when the
  join is staggered onto pre-established connections, j2x, and take
  119 s under simultaneous socket starts, i2). This row is reported as
  a scope refutation of the composition rule, not as a fabric miss,
  and it seeds the endpoint-dynamics residual task at closure.
- The mixed A100-plus-GH200 cells are excluded from the scored
  calibration, with the reason frozen: (a) the 2.775x direction
  asymmetry is endpoint-stack-owned by the capture's own evidence (the
  same two endpoints and the same fabric path carry 2.8 times more one
  way than the other, and the slow leg is the Grace-sourced one), and
  the declared fabric instance is direction-symmetric per link, so a
  fabric-only model has no term that could produce it except by
  absorbing the stack into the fabric, which the separation
  requirement forbids; (b) no independent sustained solo anchor exists
  for the mx pairs (the mx cells are the only sustained measurements
  of themselves, so every composed row would be circular, and burst
  anchors are barred above); (c) the GH200 nodes are outside the
  declared five-node instance, and extending the instance would add
  ports the scored families never touch. TRAF-51's mixed-pair clause
  offers exactly this branch (modeled as an endpoint-stack term or
  declared out of scope); this study takes the out-of-scope branch and
  hands the endpoint-term evidence to the residual task: the derived
  endpoint floors under the frozen separation are 2164.5 us (A100 to
  GH200) and 6515.0 us (GH200 to A100) per chunk against their
  measured p50s of 2504.9 and 6855.4 us, reported as
  derived-not-scored.

## Failure semantics and the diagnosis tree, pre-declared

- Any fatal guard: the affected runs are void, nothing closes, the
  evidence is retained and reported.
- An EX row fails alone: fabric latency-accounting defect
  (store-and-forward arithmetic); backend follow-up; no calibration
  language moves.
- BE-1, BE-2 or BE-7 fails: fabric rate or pacing defect; same
  handling.
- BE-6 fails with EX rows passing: framing-dependent accounting
  defect; additionally re-examine the derived framing parameter (a
  topology-instance defect is the alternative branch; if the framing
  parameter is at fault the instance is corrected and the whole run
  matrix is void for closure and rerun under a new freeze).
- ST-1 fails: topology-instance defect (per-port asymmetry); void for
  closure, fix, re-freeze.
- BE-4 or BE-5 fails while solo rows pass: the artifact boundary
  differs from the backend sanity analysis on this shape; the echo
  finding is reported and the backend follow-up records it; the
  calibration rows are unaffected (echoes gate nothing).
- BE-3 fails: the fabric owns a transient; CO-10 re-derives against
  it, and the pre-attribution of the 119 s transient to the endpoint
  is withdrawn for re-analysis (this is the one branch that could
  move the convergence attribution).
- A CO row fails through the simulation side: handled by the branch
  that moved it. No CO row can fail any other way; if scoring finds
  one that did, the freeze itself was wrong and that is reported as a
  freeze defect, not patched.
- All simulation rows pass and every CO row confirms: the calibration
  claim upgrades exactly as far as the evidence carries, stated in the
  closure rule below.

## TRAF-51 closure rule, frozen

TRAF-51's registered clause includes, among the behaviors a calibrated
instance must reproduce, the 119-second convergence transient under
simultaneous starts. This study's frozen composition rule cannot
reproduce that transient by construction, and this is known now, before
any run. Therefore, frozen in advance: this study cannot fully close
TRAF-51 under a genuine reading of its clause. The best achievable
outcome is an honest partial:

- If every fatal guard holds, all 11 simulation rows pass, and all 10
  conditional rows confirm: the hosted-pending-calibration language is
  updated to name exactly what is calibrated: steady-state solo,
  two-source incast and staggered-join behavior at the captured
  distinct-port mappings and captured loads, on the declared
  single-switch Merlin instance, with the endpoint host-stack floor as
  an explicit separate measured term; the endpoint dynamics (the 119 s
  transient, the source-identity rate asymmetry, burst-versus-sustained
  variability, the mixed-pair asymmetry) are declared un-modeled and
  registered as a residual precision task. TRAF-51's entry is
  rewritten to the narrowed remainder and stays open. The rnic-ss
  endpoint claim does not move: these cells exercise the
  `htsim_ss_dragonfly` fabric harness, not the rnic-ss endpoint.
- If simulation rows fail: refutation-with-findings per the diagnosis
  tree; no language upgrade; TRAF-51 stays open unchanged plus the
  findings.
- Residual registrations at closure use TRAF-53 and up here and
  HTSIM-29 and up for backend-repo follow-ups (the rate-controlled or
  closed-loop source gap, the multi-receiver pattern gap, and the
  backend design-note status wording), registered in this repository's
  module docs without touching the backend repo.

## The model action, frozen

No simllm profile, envelope, arm or reported metric changes in this
study. The deliverables are: the declared topology instance files, the
runner and analyzer, the bulk simulation artifacts with a tracked
locked summary (packaging commit with its manifest lock test and
`.gitattributes` rules in the same change), RESULTS.md with the
per-quantity measured, simulated, composed, residual and verdict table,
and the registry edits the closure rule above authorizes.
