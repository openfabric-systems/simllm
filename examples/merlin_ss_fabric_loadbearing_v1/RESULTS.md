# Merlin ss-dragonfly load-bearing recalibration v1 results (TRAF-51)

The reviewed state is **CLEAN: no fatal guard fired, all 8 scored rows
pass (2 exact, 5 behavioral, 1 structural), all 3 consistency rows hold
as recorded, and the registered two-configuration discrimination
statement is demonstrated at the composed level**. What that
establishes, stated precisely: with the fabric genuinely load-bearing,
the hosted Slingshot-class instance reproduces the captured x4
shared-egress family's steady aggregate to within 4.3 percent from
per-stack rate-derived endpoint floors (simulated composed 10.6276
against measured 11.0952 GB/s, ratio 0.9579 inside the frozen
[0.90, 1.001] band), with the sharing waits genuinely simulated: the
zero-wait composed value equals the measured aggregate by construction,
so the entire 4.2 percent residual is the simulator's own shared-egress
queueing, and the disclosed fluid napkin predicted it to within 0.4
percent (0.954 against 0.958). The same composed cell run on the 4 MiB
buffer configuration faults by the registered closed-loop drop
signature while the 32 MiB configuration completes clean in band, the
capture-shaped control cell is byte-identical across all three buffer
configurations, and the saturating shared-egress arm separates the pair
inside the frozen bands, so two fabric configurations the wave-19
evidence class could not distinguish produce opposite registered
verdicts exactly where the fabric carries load. The p50-derived static
floor overshoots the measured aggregate by the registered 12.7 percent
(signed band [1.05, 1.21]), which refutes p50-static endpoint floors
for skewed shared-port families and hands that evidence to TRAF-53.
TRAF-51's narrowed clause is met for exactly these clauses and no
further: the 119-second transient, the tranche-2 families and
multi-switch routing stay open, no claim is made about Merlin's
physical buffer sizing, and the rnic-ss endpoint claim does not move.

## Freeze integrity and chronology

One clock throughout (this workstation, commits in CEST, run manifest
in UTC).

| Step | Identity | Note |
|---|---|---|
| pin and build | submodule `1dcbfec`, binary SHA-256 `66241691...` | pre-freeze; 468/468 ctest at the pin, including the load-harness fixtures |
| freeze | commit `179cdc9` (13:29:38) | expectations only: bands, guards, instances, napkin script, diagnosis tree, closure rule |
| harness | commit `f40582c` (13:38:00) | runner, analyzer, harness self-check tests; no cell had run |
| runs | 11:48:16 to 11:49:35 UTC | all 11 cells twice, one runner invocation, strictly sequential |
| analyzer fix | commit `fc48ec0` (13:52:21) | disclosed below; crashed before any verdict, no comparison logic changed |
| packaging | this commit | tracked `results/` with manifest lock test and registry edits |

Chronology disclosures, exactly as the freeze stated them: the
measured side of every comparison (the byte-locked x4 dataset bytes,
the archived wave-20 artifacts and their hashes) was published and
read before the freeze; the genuinely blind quantities were the
load-bearing simulated outcomes, and the reproduction rows EX-1 and
EX-2 are exactly what they claim to be: reproduction at the pin, not
blind prediction. The fluid napkin model was committed with the freeze
and its outputs quoted there before any cell ran. One correction after
the runs: the analyzer's first invocation crashed on a key collision
(the load manifest's `flows=` count token overwrote the per-flow
record list) before evaluating any guard or row; commit `fc48ec0`
renamed the internal key and changed no band, no comparison and no
registered value. The runs themselves were never touched, and every
verdict below was produced by the fixed analyzer over the unchanged
run bytes.

## Fatal guards, all held

- FG-1 identity: binary SHA-256 equals the frozen value, submodule
  HEAD `1dcbfec36a33753bf978cf6323bade1a6645fe4f`, all three instance
  hashes match the tracked bytes (machine-enforced by the analyzer and
  the lock test), dataset manifest `67f898a0...` verified with every
  consumed file re-hashed.
- FG-2 repeat determinism: all 11 cells byte-identical across their
  two repeats (bins, chunks, stdout, stderr, exit), including the
  fault arm (2/2 with identical stderr).
- FG-3 conservation: no per-flow 100-us bin above 2,487,544 B, no
  cell aggregate bin above its distinct-destination-port bound,
  injected = delivered + dropped and payload arithmetic exact in
  every clean cell.
- FG-4 execution: every clean cell exited 0 with quiescent fabric and
  zero closed-loop drops; the x4r-v1 fault arm took its pre-declared
  survivable branch (its fault IS the scored BE-3 observable).
- FG-5 chunk integrity: chunk CSV rows equal manifest counts,
  completions strictly increasing, 1,028 to 2,413 completions per
  flow in the scored windows (frozen minimum 500).
- FG-6 frozen-input integrity: the per-flow [160, 180) s counts
  {4,014, 7,704, 7,682, 7,053}, the series p50s and both think tables
  re-derived from the tracked dataset bytes equal the frozen values
  exactly.
- FG-7 seam echo: every manifest flow line echoes its declared
  think_ps, offered_bps and start_ps exactly, in every cell.

## Scored rows, 8 of 8 pass

| Row | Class | Registered | Observed | Verdict |
|---|---|---|---|---|
| EX-1 | exact | sat-v1 reproduces the wave-20 discriminate-A record: bins SHA `af1d1e04...`, masked stdout `4bdf0f07...`, injected 17,980 (8,990 per flow), delivered 14,295, payload 127,911,660 B, dropped 3,685, first_drop 560,255,151 ps | all equal, byte-exact | PASS |
| EX-2 | exact | ctrl-v1 reproduces the wave-20 control-A record: bins `7eb66e67...`, chunks `1072cd8f...`, masked stdout `4a7ce046...` | all equal, byte-exact | PASS |
| BE-1 | behavioral | x4 rate-arm composed aggregate over measured in [0.90, 1.001]; fluid point 0.954 | 10.6276 GB/s, ratio 0.95786 | PASS |
| BE-2 | behavioral | x4 p50-arm ratio in [1.05, 1.21], signed above 1.05; fluid point 1.127 | 12.5067 GB/s, ratio 1.12721 | PASS |
| BE-3 | behavioral | the composed x4 cell on the 4 MiB instance faults: exit 2 with the registered closed-loop drop message | exits 2/2, message present | PASS |
| BE-4 | behavioral | sat-b32 first_drop_ps in [4,474,000,000, 4,476,000,000]; point 4,474,938,884 | 4,474,404,382 | PASS |
| BE-5 | behavioral | sat-b32 dropped in [430, 442]; point 436 | 437 | PASS |
| ST-1 | structural | x4r-b32 and x4r-b64 byte-identical (bins, chunks; stdout masked) | identical | PASS |

Denominators per the freeze: 2 exact, 5 behavioral, 1 structural,
never summed with consistency rows. Every row's could-fail statement
and simulator coupling is in the freeze; none was entailed by a guard.

## The composed table: measured, simulated-composed, residual

The composition happened inside the simulator (endpoint floors as
declared think times, waits simulated), so the composed quantities ARE
the simulation outputs; this is the wave-19 lesson executed, and the
coupling column of that study's conditional table has no analog here
because every residual below is simulator-owned.

| Quantity | Measured | Simulated composed | Residual | Status |
|---|---:|---:|---:|---|
| x4 aggregate steady (rate arm) | 11.0952 GB/s | 10.6276 GB/s | -4.21 percent | scored, BE-1 PASS |
| x4 flow 0 rate | 1.6836 | 1.6335 | -2.98 percent | derived, unscored (floors carry the asymmetry) |
| x4 flow 1 rate | 3.2313 | 3.0916 | -4.32 percent | derived |
| x4 flow 2 rate | 3.2221 | 3.0824 | -4.33 percent | derived |
| x4 flow 3 rate | 2.9582 | 2.8201 | -4.67 percent | derived |
| x4 Jain | 0.9496 | 0.9513 | +0.0017 | derived |
| x4 aggregate (p50-arm static floor) | 11.0952 | 12.5067 | +12.72 percent | scored, BE-2 PASS: the registered overshoot |

The implied mean sharing wait per chunk is 117 to 153 us against the
339-us burst (per-flow cycle inflation 3.1 to 4.9 percent), which is
what the shared egress at 44.8 percent utilization adds under
round-robin VoQ grants. The disclosed fluid napkin predicted per-flow
window counts of {1069, 2012, 2006, 1852} against the simulator's
{1071, 2027, 2021, 1849} (D-3 residuals: +0.0040 rate arm, +0.0004 p50
arm), so the flow-level water-filling model captured the packet-level
sharing dynamics almost exactly.

## The discrimination outcome, as registered

- Control leg: ctrl-v1, ctrl-b32 and ctrl-b64 byte-identical (CN-1),
  and the control bytes equal the wave-20 archived record (EX-2). At
  capture-shaped load the buffer pair is invisible.
- Composed leg: the identical composed x4 closed-loop cell completes
  clean and in band on 32 MiB (FG-4 plus BE-1) and faults by the
  registered signature on 4 MiB (BE-3): opposite registered verdicts.
  The 4 MiB fault also confirms the freeze's napkin: a two-burst
  overlap beyond 167.8 us must overfill 4 MiB, and the fluid model
  predicted the first crossing at 21.4 ms of the 6-s window.
- Saturating leg: first drops at 560,255,151 ps (v1, EX-1) against
  4,474,404,382 ps (x4buf32, BE-4), drop counts 3,685 against 437
  (BE-5), difference 3,248 inside the derived [3,243, 3,255] band
  (D-1), orderings as registered (D-2).

The registered statement therefore holds: two fabric configurations
that the capture-shaped evidence class cannot separate produce
different registered verdicts exactly where the fabric is
load-bearing, inside one pre-registered study. Explicit non-claim,
repeated from the freeze: nothing here says which buffer the Merlin
switch has; the registered abstraction has no loss recovery and the
real transport does.

## Consistency rows (recorded, unscored) and derived rows

- CN-1 holds: control cell byte-identical across all three buffer
  configurations.
- CN-2 holds: the staggered-join mirror's cadences equal the solo
  formulas exactly over 137 and 126 chunks (injected 246,694,
  dropped 0): the model-side join-unharmed premise restated, not
  tested, per the freeze's structural reason.
- CN-3 holds: the paced mirror at the measured offered rates delivers
  exactly what it injects, per flow, with dropped 0.
- D-1 3,248 inside [3,243, 3,255]; D-2 ordering holds; D-3 fluid
  residuals +0.0040 (rate) and +0.0004 (p50); D-4 sim Jain 0.9513 and
  per-flow rates as tabled above.

## Physical sanity review

Three independent framings, per the local rules.

**Network and serialization physics.** The pinned build reproduces the
wave-20 evaluation of record byte for byte on the same parameter block
(EX-1, EX-2), the 32 MiB first-drop lands 534.5 ns early against a
3.91-ms linear-fill extrapolation (0.014 percent of the extrapolated
interval, within the pacing quantization), and the drop-count
difference equals the buffer delta over the wire packet size (3,248
against 29,360,128 / 9,038 = 3,248.5). No bin anywhere exceeded the
packet quantization bound.

**Queueing physics.** At 44.8 percent offered utilization of one
shared egress, the simulated mean wait per 339-us chunk burst is 117
to 153 us. The M/D/1-shaped ballpark rho * S / (2 (1 - rho)) gives
137.6 us, and the disclosed fluid model's prediction of the composed
aggregate was within 0.4 percent of the packet-level result, so the
sharing residual behaves like genuine burst queueing, not like an
artifact of the harness. The p50-arm overshoot (12.7 percent) sits
between the zero-wait ceiling (20.5 percent) and the rate-arm level,
exactly where its higher 54.0 percent utilization puts it.

**End-to-end plausibility.** The composed model reproduces the
capture's shared-port headline (aggregate scaling with stack count:
10.63 simulated against 11.10 measured through one port), preserves
the per-flow NUMA ordering and fairness (Jain 0.9513 against 0.9496),
and the 4 MiB fault is consistent with reality: the real transport
carried this exact load through one port with TCP loss recovery
absorbing any drops, and a lossless abstraction needs the buffer
headroom the x4buf32 instance declares. The measured aggregate stays
inside every physical ceiling (44.8 percent of one port), and no
simulated quantity beats its zero-wait bound.

## What the fabric is now calibrated for, exactly

Added by this study to the wave-19 statement, on the declared
single-switch Merlin instance at the pinned load harness: the
shared-egress x4 family's steady aggregate at its captured load (44.8
percent of one port) is reproduced within the frozen band with
per-stack rate-derived endpoint floors as declared think times and the
sharing waits genuinely simulated; fabric-configuration discrimination
at the composed level is demonstrated (the registered
indistinguishable-then-separated triple); and the p50-static endpoint
floor is refuted for skewed shared-port families (a static think time
cannot reproduce mean and median of the captured x4 cadences at once;
the 12.7 percent registered overshoot is the measurement of that gap,
TRAF-53 evidence). Everything else keeps exactly the wave-19 wording:
solo, distinct-port incast and staggered-join families at captured
loads, instance arithmetic under three parameterizations, endpoint
floor separate.

Not established, stated plainly: endpoint-stack dynamics (the 119-s
transient, source-identity asymmetry, burst-versus-sustained
variability, the mixed-pair asymmetry; TRAF-53); the tranche-2
families (i3, j3, i4, GH; TRAF-52, with this study's frozen
late-arrival path ready for any shared-egress group among them);
multi-switch adaptive routing (structurally unreachable at the
declared shape); the true source-shared x4 mapping (the capture shares
the SOURCE port; the harness's pairwise-distinct dispatch forces the
destination-shared abstraction; HTSIM-32 and HTSIM-33 register the
backend gaps); Merlin's physical buffer sizing (no fabric-side
measurement exists and the loss-recovery abstraction differs); and the
rnic-ss endpoint, which no cell here exercised.

## Late arrivals: the frozen tranche-2 path

Unchanged from the freeze and mechanically ready: when the other
orchestrator session lands i3, j3, i4 or the GH family into the
byte-locked dataset, the frozen mapping rule sorts each cell's flows
by their recorded port evidence; distinct-port families are
consistency-only under this model and feed no scored row here, and
any shared-egress group feeds R-LATE-AGG through
`analyze_recalibration.py --derive-late-cell` (the frozen formulas;
the x4 self-check reproduces this study's think table byte-exactly,
enforced by `tests/test_merlin_ss_fabric_loadbearing_harness.py`),
run by `run_cells.py --late`, scored by `--score-late` against the
frozen [0.90, 1.001] band with no code change and no band moved. GH
cells fail closed unless their floors derive from the GH cells
themselves (the E-A-7 per-width rule).

## Residual registrations

- HTSIM-32 (Completeness; P2; M): flow-identity-keyed delivery
  dispatch so several flows can share one (source, destination) host
  pair; enables the true source-shared x4 mapping.
- HTSIM-33 (Completeness; P2; S): the host injection queue depth as a
  topology parameter (hardcoded 64 wire packets), needed before any
  source-side sharing study can be expressed.
- TRAF-51: stays open, entry narrowed to the un-met remainder.
- TRAF-53: gains the p50-floor refutation evidence (BE-2) and the
  mean-versus-median skew numbers.

## Reproduction

```bash
./scripts/build_htsim.sh <build-dir> --test
python examples/merlin_ss_fabric_loadbearing_v1/run_cells.py \
    --binary <build-dir>/datacenter/htsim_ss_dragonfly \
    --out-root <bulk-dir> --submodule-dir third_party/htsim
python examples/merlin_ss_fabric_loadbearing_v1/analyze_recalibration.py \
    --run-root <bulk-dir> --out examples/merlin_ss_fabric_loadbearing_v1/results
```

Determinism makes the reproduction exact: identical binaries produce
byte-identical CSVs, and the tracked `results/` summaries are locked
by `tests/test_merlin_ss_fabric_loadbearing_results.py` against the
frozen binary hash, submodule pin, dataset manifest hash and the
tracked topology instances.
