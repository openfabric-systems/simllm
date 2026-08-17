# Merlin ss-dragonfly fabric calibration v1 results (TRAF-51)

The reviewed state is **PARTIAL as pre-declared, clean by evidence: no
fatal guard fired, all 11 simulation rows pass (3 exact oracles to the
bin, 7 behavioral, 1 structural), and all 10 conditional composed rows
confirm the frozen composition rule against the byte-locked capture
dataset**. The hosted ss-dragonfly fabric is now calibrated for
steady-state solo, two-source incast and staggered-join behavior at the
captured distinct-port mappings and captured loads, on the declared
single-switch Merlin instance, with the endpoint host-stack floor an
explicit separate measured term. The calibration is partial for exactly
the reason the freeze declared before any run: the frozen composition
rule (fabric plus a static endpoint floor) is structurally incapable of
reproducing the captured 119-second simultaneous-start convergence
transient, so TRAF-51's registered clause is not fully met, TRAF-51
stays open narrowed, and the endpoint-dynamics remainder is registered
as TRAF-53. No mixed-architecture quantity is scored (out-of-scope
branch of TRAF-51's mixed clause, reason frozen), and the rnic-ss
endpoint claim does not move: every cell here drives the
`htsim_ss_dragonfly` fabric harness, not the rnic-ss endpoint.

## Freeze integrity and chronology

| Step | Identity | Note |
|---|---|---|
| pin and registry | commit `ab3a6bc` | submodule to backend merge `89b7a5a`; build 448/448 tests; sanity determinism precondition run pre-freeze |
| freeze | commit `7f7550e` | expectations only: composition rule, declared instance, bands, guards, diagnosis tree, closure rule |
| harness | commit `86a385b` | `run_cells.py`, `analyze_calibration.py`; no cell had run |
| runs | 2026-08-17 22:18:39 to 22:18:51 UTC | all 6 cells twice plus 4 sanity arms twice, single invocation of the runner |
| packaging | commit `32451b4` | tracked `results/` with manifest lock test `tests/test_merlin_ss_fabric_calibration_results.py` |

Chronology disclosures, exactly as the freeze stated them: the measured
side of every comparison was published and byte-locked before this study
began, so nothing measured was blind; the freeze computed and disclosed
the composed-versus-measured arithmetic in advance, and the conditional
rows below are calibration consistency confirmations under a frozen
rule, never blind predictions. The genuinely blind quantities were the
simulation outputs, and the freeze hand-derived their exact oracles
before the first run. One pre-freeze harness feasibility probe of the
9038-byte framing ran on the backend study topology only (disclosed in
the freeze); no Merlin-instance invocation preceded the freeze commit.

## Fatal guards, all held

- FG-1 binary and pin identity: `htsim_ss_dragonfly` SHA-256
  `5075021a...` (frozen value), submodule HEAD `89b7a5a8...`, topology
  file hashes recorded in the locked run manifest.
- FG-2 backend sanity determinism: the four rerun sanity arms
  byte-identical between repeats on this build, and additionally
  byte-identical to the archived wave-18 scored CSVs
  (`incast_1_minimal`, `incast_2_adaptive`, `incast_8_adaptive`,
  `join_adaptive`), so this build reproduces the backend study's
  artifacts exactly.
- FG-3 per-cell determinism: all 6 cells byte-identical across repeats.
- FG-4 conservation: worst 100-us bin 2,478,596 bytes against the
  277-plus-1 packet bound 2,487,544; delivered equals injected with
  zero drops in every solo cell (553,220 of 553,220 packets in
  cal-solo-a); payload arithmetic exact in every cell.
- FG-5 execution sanity: exit 0 and harness quiescence validation in
  all 20 invocations.
- FG-6 dataset integrity: capture manifest SHA-256 `a6b7e61e...`
  verified, all 10 consumed dataset files hash-verified, and the six
  re-derived stage-steady rates agree with the published `stats/*.json`
  values exactly.

## Simulation rows, 11 of 11 pass

| Row | Class | Quantity | Expected | Observed |
|---|---|---|---|---|
| EX-1 | exact | solo first-chunk crossing bin end | 341,000,000 ps (hand 340,417,280) | 341,000,000 ps |
| EX-2 | exact | 100G variant, same | 680,000,000 ps (hand 679,884,560) | 680,000,000 ps |
| EX-3 | exact | 4160-framing, same | 342,000,000 ps (hand 341,903,600) | 342,000,000 ps |
| BE-1 | behavioral | solo steady rate over C_p | [0.99, 1.0005] | 1.000000 |
| BE-2 | behavioral | solo chunk-delta p50 | [335, 345] us | 339.0 us |
| BE-3 | behavioral | solo settling | every 100-us bin 1..1999 within 1 percent of C_p | holds in all four solo cells |
| BE-4 | behavioral | echo-2 admission-capture signature | aggregate in band, exactly one starved flow, drops > 0 | 1.000037 C_p, flow 1 starved, 276,146 drops |
| BE-5 | behavioral | echo-4 admission-capture signature | aggregate in band, >= 1 starved, top >= 0.5 C_p, drops > 0 | 1.000037 C_p, flows 1, 2, 3 starved, flow 0 at C_p, 829,366 drops |
| BE-6 | behavioral | framing-shift rate ratio | [0.9935, 0.9955], predicted 0.99452 | 0.994517 |
| BE-7 | behavioral | 100G steady rate over C_p/2 | [0.99, 1.0005] | 1.000000 |
| ST-1 | structural | port symmetry | identical delivered series | identical |

The three exact oracles verify the fabric's store-and-forward
arithmetic in three independent directions: absolute (EX-1), rate
scaling with propagation and pipeline held fixed (EX-2, serialization
terms exactly doubled), and framing scaling (EX-3, per-packet count
changed 938 to 2048). BE-6's ratio matches the pure framing-efficiency
prediction to 3 parts in a million. The echo rows are positive
controls, not calibration: they demonstrate on the Merlin instance the
same deterministic admission-phase capture the backend sanity studies
analyzed on the multi-hop study fabric (junior flows starve, the
aggregate holds at exactly C_p, drops accumulate), confirming that
regime is a property of synchronized open-loop overload at a shared
egress, which no captured cell is in.

## The calibration table: measured, simulated, composed, residual, verdict

Fabric terms entering the composition, simulated on the declared
instance: R_fab = 24.7511 GB/s per flow (line-rate delivery, no
cross-flow term at the captured distinct-port mappings), T_fab_lat =
340.417 us per 8 MiB chunk (EX-1-confirmed hand value), T_fab_rate =
338.919 us. Validity condition confirmed: the endpoint stack is the
binding stage for every scored pair (its floor is 3.69x to 5.46x the
fabric term). Endpoint floors, the explicit separated term:
H_time(gpu105 to gpu102) = 1596.855 - 340.417 = 1256.438 us per chunk
(78.7 percent of chunk life), H_time(gpu103 to gpu102) = 2198.644 -
340.417 = 1858.227 us (84.5 percent).

Anchors (calibration inputs, unscorable by construction, listed for
the record):

| Quantity | Measured | Role |
|---|---:|---|
| s1 solo steady rate (gpu105 to gpu102) | 4.9824 GB/s | H_rate anchor |
| s1 solo steady-window p50 | 1596.855 us (published whole-window p50 1598.428) | H_time anchor |
| j2x stage-0 steady rate (gpu103 to gpu102) | 3.8193 GB/s | H_rate anchor, post-specified cell, disclosed |
| j2x stage-0 steady-window p50 | 2198.644 us | H_time anchor |

Scored conditional rows (bands frozen; composed values disclosed at
freeze; every verdict landed as the freeze computed):

| Row | Quantity | Measured | Simulated fabric term | Composed | Residual (composed over measured) | Verdict |
|---|---|---:|---:|---:|---:|---|
| CO-1 | i2 flow 0 steady rate | 3.6553 GB/s | R_fab 24.7511 GB/s, not binding | 3.8193 GB/s | +4.49 percent | pass |
| CO-2 | i2 flow 1 steady rate | 4.8931 GB/s | same | 4.9824 GB/s | +1.83 percent | pass |
| CO-3 | j2x stage-1 flow 0 steady rate | 4.0249 GB/s | same | 3.8193 GB/s | -5.11 percent | pass |
| CO-4 | j2x stage-1 flow 1 steady rate | 4.8486 GB/s | same | 4.9824 GB/s | +2.76 percent | pass |
| CO-5 | join-unharmed, flow 0 stage 1 over stage 0 | 1.0538 | no cross-flow fabric term | 1.0000 | measured 5.38 percent above composed, inside [0.90, 1.10] | pass |
| CO-6 | i2 flow 0 p50 | 2239.049 us | T_fab_lat 340.417 us | 2198.644 us | -1.80 percent | pass |
| CO-7 | i2 flow 1 p50 | 1600.271 us | same | 1596.855 us | -0.21 percent | pass |
| CO-8 | j2x stage-1 flow 0 p50 | 2135.626 us | same | 2198.644 us | +2.95 percent | pass |
| CO-9 | j2x stage-1 flow 1 p50 | 1722.941 us | same | 1596.855 us | -7.32 percent | pass |
| CO-10 | staggered-join settling | 0.0 s | fabric settles within one 100-us bin (BE-3) | no term above one bin | both effectively zero | pass |

Derived quantities (functions of scored rows, reported and never
scored, per the freeze's entailment analysis):

| Quantity | Measured | Composed |
|---|---:|---:|
| i2 aggregate steady | 8.5484 GB/s | 8.8017 GB/s (ratio 1.0296) |
| i2 aggregate over solo (the TRAF-51 1.71x behavior) | 1.7157 stage-definition (published 1.7129) | 1.7666 |
| i2 Jain | 0.9795 | 0.9828 |
| j2x stage-1 aggregate | 8.8735 GB/s | 8.8017 GB/s (ratio 0.9919) |
| j2x stage-1 Jain | 0.9915 | 0.9828 |

The refuted-by-construction row, reported exactly as frozen:

| Quantity | Measured | Composed | Verdict |
|---|---:|---|---|
| i2 simultaneous-start convergence | 119 s | not producible by the frozen rule (static endpoint floor, fabric settles in one bin) | out of expressiveness; scope refutation of the composition rule, not a fabric miss; registered TRAF-53 |

The attribution rests on the capture's own controlled contrast (the
same pairs and stacks settle in 0.0 s when the join is staggered onto
pre-established connections) plus BE-3 (the fabric contributes no
settling timescale beyond one 100-us bin), which was the one frozen
branch that could have withdrawn the attribution and did not.

Mixed-architecture cells: excluded from scoring on the frozen
out-of-scope branch (endpoint-stack-owned asymmetry, no independent
sustained solo anchor so every composed row would be circular, GH nodes
outside the declared instance). Reported derived-not-scored under the
same separation: endpoint floors 2164.5 us (A100 to GH200, measured p50
2504.9) and 6515.0 us (GH200 to A100, measured p50 6855.4) per chunk,
handed to TRAF-53 as its mixed-pair starting numbers. The 2.775x
asymmetry itself was scored by the capture study (E-M-4) and gains no
new evidence here.

## Physical sanity review

Three independent framings, per the local rules.

**Network and serialization physics.** The simulated solo steady rate
sits exactly on the arithmetic payload ceiling (24,751,062,800 against
24,751,050,674 bytes/s, a 5e-7 quantization surplus inside the frozen
packet bound), the first-chunk time is the hand-derived
serialization-plus-pipeline sum to the bin, and halving the link rate
moved exactly the serialization terms (EX-2). No bin anywhere exceeded
the packet quantization bound.

**Composition physics.** The measured chunk life decomposes as 78.7 to
84.5 percent endpoint stack and 15.5 to 21.3 percent fabric at 200G
framing, which restates the capture's 4.8x host-stack-floor finding
through this study's separation (1596.855 / 340.417 = 4.69 at the
tightest pair). Every composed steady rate then lands within 7.3
percent of its measured counterpart using solo anchors only, that is,
the captured multi-flow families behave as independent per-pair stacks
over a non-interfering fabric to within the capture's own same-pair
repeatability (the gpu103 pair itself moved 4.5 to 10.1 percent between
its capture cells).

**End-to-end plausibility.** The composed model reproduces the
capture's headline sharing behaviors in the steady state: the aggregate
nearly doubles with a second source stack (1.77 composed against 1.72
measured), fairness stays near 1 (0.983 against 0.979 and 0.991), and
an established flow loses nothing at a staggered join (1.000 against
1.054). What it cannot reproduce, and was frozen as unreproducible, is
the 119-second settling of simultaneously started socket flows, a
dynamics term that lives in the endpoint stack the composition
deliberately holds static.

## What the fabric is now calibrated for, exactly

On the declared single-switch Merlin instance
(`merlin_a100_singleswitch_v1.topo`, every parameter carrying measured,
derived or declared-with-reason provenance), driving
`htsim_ss_dragonfly` at the pinned merge:

- Calibrated: steady-state per-flow and aggregate goodput, per-chunk
  completion p50, join-unharmed behavior and near-instant staggered-join
  settling for the captured solo, two-source-incast and two-flow-join
  families at their captured port mappings (pairwise distinct source
  and destination ports) and captured loads (each stack at most a
  fifth of a port), under the frozen composition rule with the
  measured per-pair endpoint floor as a separate term.
- Not calibrated, stated plainly: endpoint-stack dynamics (the 119 s
  simultaneous-start transient, source-identity rate asymmetry,
  burst-versus-sustained variability, the mixed-pair direction
  asymmetry); shared-port and higher-degree incast families (i3, j3,
  i4, x4 are uncaptured, TRAF-52, and the open-loop harness cannot
  express host-stack-bound offered load or multiple receivers,
  HTSIM-29 and HTSIM-30); multi-switch dragonfly routing behavior
  (structurally unreachable at the single-switch shape that five-node
  discovery determines); and the rnic-ss endpoint, which no cell here
  exercised.

## Residual registrations

- TRAF-53 (Precision; P2; L): the endpoint host-stack dynamics term.
- HTSIM-29 (Completeness; P2; M): rate-controlled or closed-loop
  greedy sources for `htsim_ss_dragonfly`.
- HTSIM-30 (Completeness; P2; S): multi-receiver patterns in the
  harness.
- HTSIM-31 (Completeness; P2; S): the backend design note's status
  wording update to name this calibration scope.
- TRAF-51: stays open, entry narrowed to the un-met remainder.

## Reproduction

```bash
./scripts/build_htsim.sh <build-dir> --test
python examples/merlin_ss_fabric_calibration_v1/run_cells.py \
    --binary <build-dir>/datacenter/htsim_ss_dragonfly \
    --sanity-topo third_party/htsim/htsim/sim/datacenter/topologies/ss_dragonfly/p2a2h1g3_200g.topo \
    --submodule-dir third_party/htsim --out-root <bulk-dir>
python examples/merlin_ss_fabric_calibration_v1/analyze_calibration.py \
    --dataset-root examples/merlin_fabric_flow_capture_v1/dataset \
    --run-root <bulk-dir> --out <results-dir>
```

Determinism makes the reproduction exact: identical binaries produce
byte-identical CSVs, and the tracked `results/` summaries are locked by
`tests/test_merlin_ss_fabric_calibration_results.py` against the frozen
binary hash, submodule pin and dataset manifest hash.
