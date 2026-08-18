# Merlin ss-dragonfly load-bearing recalibration v1 results (TRAF-51)

The reviewed state is **CLEAN: no fatal guard fired, all 8 scored rows
pass (2 exact, 5 behavioral, 1 structural), all 3 consistency rows hold
as recorded, and the registered two-configuration discrimination
statement is demonstrated at the composed level**. What that
establishes, stated precisely: with the fabric genuinely load-bearing,
the hosted Slingshot-class instance reproduces the captured x4
shared-egress family's steady aggregate to within 4.21 percent from
per-stack rate-derived endpoint floors (simulated composed 10.6276
against measured 11.0952 GB/s, ratio 0.9579 inside the frozen
[0.90, 1.001] band), with the sharing waits genuinely simulated: the
zero-wait composed value equals the measured aggregate by construction,
so the entire 4.21 percent residual is the simulator's own
shared-egress queueing, and the disclosed fluid napkin predicted it to
within 0.4 percent (0.954 against 0.958). The band's discriminating
power is coarse by construction and travels with every citation of
that tight residual (review correction 1): [0.90, 1.001] separates
fluid-like sharing from a chunk-serializing egress (which would land
near 0.74) and from starvation, but tolerates up to roughly 2.5 times
the observed sharing wait (doubling every simulated wait would still
pass at 0.9192), so the 4.21 percent residual is a reported
observation, not a validated tolerance, and what the PASS validates is
the sharing-mechanism class. The same composed cell run on the 4 MiB
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
  think_ps, offered_bps and start_ps exactly, in every cell that
  produced a manifest (10 of 11). The fault arm writes no manifest, so
  its seam declarations are verified at the argv level only: its
  recorded argument list is byte-identical to the clean x4 rate arms'
  (review correction 4).

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
arm), so the flow-level water-filling model predicted the aggregate
ratios and per-flow window counts almost exactly; the agreement claim
is limited to those recorded quantities (review correction 2), since
no within-run wait traces or fault instants exist to compare
trajectories against.

## The discrimination outcome, as registered

- Control leg: ctrl-v1, ctrl-b32 and ctrl-b64 byte-identical (CN-1),
  and the control bytes equal the wave-20 archived record (EX-2). At
  capture-shaped load the buffer pair is invisible.
- Composed leg: the identical composed x4 closed-loop cell completes
  clean and in band on 32 MiB (FG-4 plus BE-1) and faults by the
  registered signature on 4 MiB (BE-3): opposite registered verdicts.
  Scope of BE-3's risk, corrected by the review (correction 3): the
  genuine risk was the fault SIGNATURE (the registered exit and
  message, i.e. the harness's drop detection and error path behaving
  as registered) and the clean-on-32-MiB contrast, not the occurrence
  itself, because the freeze's own beat arithmetic makes a full burst
  overlap, hence a 4 MiB overfill, near-entailed within about 0.77 to
  0.91 s of the 6-s window unless the closed loop phase-locks, a
  mechanism the freeze never names. No simulated fault instant was
  recorded (the faulting harness writes no manifest, bins or chunks);
  the reviewer's wall-clock inference, recorded as the best estimate
  with its indirectness labeled, notes the per-packet wall cost is
  stable across the six clean load cells and places the fault arm's
  stall near the 0.91-s flows-1-and-2 cadence beat, not at the fluid
  trajectory's own 21.4-ms first crossing.
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
interval), and the drop-count difference equals the buffer delta over
the wire packet size (3,248 against 29,360,128 / 9,038 = 3,248.5). Two
quantization scales for that first-drop miss, stated honestly (review
correction 5): the freeze's rationale named the 278-ns combined
arrival gap, and the miss is 1.92 times that scale (0.96 times the
556-ns per-flow gap), so a band matched to the freeze's own stated
rationale would have FAILED while the registered 2-us band passed; the
linear-fill model is validated at the 0.014 percent level and its
sub-microsecond correction term is unmodeled. No bin anywhere exceeded
the packet quantization bound.

**Queueing physics.** At 44.8 percent offered utilization of one
shared egress, the simulated mean wait per 339-us chunk burst is 117
to 153 us. The M/D/1-shaped ballpark rho * S / (2 (1 - rho)) gives
137.6 us, corroboration-grade only (review correction 6): closed-loop
near-periodic arrivals are not the Poisson family, and agreement at
the 15-percent level carries no inference. The load-bearing
consistency evidence is the disclosed fluid model, whose
composed-aggregate prediction landed within 0.4 percent of the
packet-level result, so the sharing residual is consistent with burst
queueing at this utilization. The p50-arm overshoot (12.7 percent)
sits between the zero-wait ceiling (20.5 percent) and the rate-arm
level, exactly where its higher 54.0 percent utilization puts it.

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
sharing waits genuinely simulated (band power as stated in the
headline: the sharing-mechanism class is validated, not a
4-percent-level tolerance); fabric-configuration discrimination
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
frozen [0.90, 1.001] band with no code change and no band moved. Two
hardenings from the review round (corrections 7 and 8), both
future-only with the tracked results proven byte-identical: the
scoring path now applies the applicable fatal-guard set (run identity
with absence failing closed, repeat determinism, clean execution,
conservation, seam echo, the 500-completion floor) before any band
verdict and reports VOID on a violation, and the derivation
machine-checks the shared-egress classification against the recorded
port evidence (a dominant sender hsn port carrying the group payload,
or a dominant destination hsn device), failing closed when readable
evidence refutes it; where a cell's evidence shape cannot be evaluated
the descriptor's `mapping_check` says `not-machine-checkable` with the
reason, and any late score published from such a descriptor must
repeat that label. GH cells fail closed unless their floors derive
from the GH cells themselves (the E-A-7 per-width rule).

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

## Post-specified corrections (adversarial evidence review)

Recorded after the review of the published record, which otherwise
verified the study end to end (freeze immutable with the napkin
reproducing every disclosed value, chronology to the second, the
`fc48ec0` fix confirmed logic-neutral, the scored form regenerating
byte-for-byte, and an independent fresh-binary rerun reproducing every
row with FG-1 correctly voiding it as not the run of record). The
frozen files (`expectations.md`, both new `.topo` instances,
`napkin_x4_fluid.py`, the locked `results/` bytes) are byte-identical
to their commits; no band, verdict or registered value changed.

1. **BE-1's discriminating power.** The [0.90, 1.001] band tolerates
   up to roughly 2.5 times the observed sharing wait (doubling every
   simulated wait still passes at 0.9192, failure only near 2.5x);
   what it separates is fluid-like sharing from a chunk-serializing
   egress (about 0.74) or starvation. The freeze disclosed the
   coarseness; this file and the registry citations of the 4.21
   percent residual now carry the power statement so the tight number
   is never read as a validated tolerance.
2. **The fluid model's agreement claim.** "Captured the packet-level
   sharing dynamics almost exactly" overstated; the agreement is the
   composed aggregate ratios and per-flow window counts, the only
   recorded quantities, and the wording now says so.
3. **BE-3's risk scope.** The fault occurrence was near-entailed: the
   freeze's own beat arithmetic forces a full burst overlap, hence a
   4 MiB overfill, within about 0.77 to 0.91 s of the window unless
   the closed loop phase-locks, a mechanism the freeze never names.
   The genuine risk was the fault signature and the clean-on-32-MiB
   contrast. The earlier "confirms the freeze's napkin 21.4 ms
   crossing" sentence is withdrawn: no fault instant was recorded (the
   faulting harness writes no manifest, bins or chunks), and the
   reviewer's wall-clock inference (per-packet wall cost stable across
   the six clean load cells) placing the stall near the 0.91-s beat is
   recorded as the best estimate, explicitly indirect.
4. **FG-7's quantifier.** "In every cell" over-stated: the echo is
   verifiable only in the 10 manifest-producing cells; the fault arm's
   seam declarations are verified at the argv level (its recorded
   argument list is byte-identical to the clean x4 rate arms').
5. **BE-4's quantization scales.** The 534,502 ps early arrival is
   1.92 times the 278-ns combined arrival gap the freeze's rationale
   named (0.96 times the per-flow gap); a band matched to the stated
   rationale would have failed while the registered 2-us band passed.
   Recorded per the corrections convention; the registered band and
   verdict stand.
6. **The M/D/1 comparison's grade.** Corroboration only: closed-loop
   near-periodic arrivals are not the Poisson family, and 15-percent
   agreement carries no inference; "behaves like genuine burst
   queueing, not an artifact" is softened to consistency wording.
7. **Late-path fail-closed hardening (future-only).** The analyzer
   defaulted a missing submodule pin to the frozen value (absence
   passed) and `--score-late` applied no fatal guards; both are fixed
   (absence fails closed; the applicable guard set gates every late
   band verdict with VOID on violation), covered by new harness tests,
   and the re-run analyzer reproduces the tracked `results/` bytes
   identically.
8. **Late-path mapping check (future-only).** The shared-egress
   classification previously trusted the operator's `--shared-flows`
   input; the derivation now machine-checks it against the recorded
   port counters (the x4 self-check verifies as shared sender port
   hsn2 and dominant destination device hsn3), fails closed on
   refuting evidence, and labels unevaluable evidence
   `not-machine-checkable`; byte-identity of the tracked results
   re-proven after the change.
9. **Numeric and chronology nits.** The headline's "within 4.3
   percent" and the registry's "4.2 percent" are unified to the
   recomputed 4.21 percent; and the `.gitattributes` eol rules landed
   with the freeze commit rather than with the packaging commit the
   freeze's deliverable list named (the safe direction: the rules
   preceded every tracked artifact they cover).

- Correction 10 (post-published, fold-in compatibility, 2026-08-18
  evening): the FG-1 dataset clause originally hashed the LIVING
  capture MANIFEST.json against the frozen digest, which voided every
  rerun the moment the capture study's committed fold-in protocol
  advanced the dataset (the tranche-2 landing did exactly that, and
  the same protocol delivered the x4 cell this study consumed). The
  clause now verifies the freeze's actual claim: it accepts the living
  manifest only while it still is the frozen version, and otherwise
  requires the preserved byte-exact snapshot at
  dataset/manifest_versions/ under the frozen digest, failing closed
  on absence or self-hash mismatch, with the consumed files verified
  against the frozen entries either way. The guard's recorded basis is
  unchanged (the frozen digest), the analyzer rerun on the
  post-fold-in dataset reproduces the tracked summary byte for byte,
  and the companion CI test performs the matching snapshot
  verification. This mirrors, one level deeper, the disclosed test
  repoint the tranche-2 landing made for the same reason.

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
