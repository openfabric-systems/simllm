# M5 MoE slice: results against pre-registered expectations

Runs of 2026-08-04. Backend binary: `htsim_rnic` from a machine-local
HTSIM-rnic-private build pinned at `c03e1f2`, the same build the M4 studies
used; the resolved historical build path is intentionally omitted. Current
reproductions select the build root with `SIMLLM_HTSIM_BUILD`. GOAL
conversion used the prebuilt `txt2bin`. All sections ran in one `run_m5.py`
invocation
(22 rows in `summary.csv`, with raw GOALs and completion CSVs beside it).
Those artifacts remain outside Git in the machine-local directory used for
the historical run; its resolved historical path is intentionally omitted. New runs
default to `${SIMLLM_DATA_ROOT}/m5`. The backends are deterministic, so every
number below reproduces exactly. Checks A1-A6, B1-B6 and C-q1/q2/q3
are the registered predictions of [expectations.md](expectations.md)
(kept frozen; deviations are disclosed here, never edited there). The
harness compares every measurement against the frozen constants, never
against runtime-derived values, and additionally checks the runtime
roofline estimate and calc split against the frozen inputs per cell
(the M4 post-audit pattern).

Verdict: **every registered check passes. All 6 A cells and all 6 B cells
reproduce their closed forms with 0 ps residual (with
`estimate_matches_frozen` and `calc_matches_frozen` true on every B
cell), and all three qualitative directions hold.** Passing check A
closes the pairwise-a2av part of TRAF-4 on the fluid profile.

Provenance disclosure (registration honesty): before the freeze, a
4-point development probe (per-pair 2048 and 3072 B, W in {2, 4, 8}, plus
the `test_sink_moe_end_to_end_matches_closed_form` unit test) exposed a
+1 ps deviation from the naive `(W-1) * S * 20 + P` law when W - 1 does
not divide 400e9. That probe motivated reading the backend source and
registering the floor/ceil quantization form
(`T_share(S, n) = ceil(S * 8e12 / floor(400e9 / n)) + P`, from
`rnic_max_min_allocator.cpp` and `rnic_fluid_manifold.cpp`) in
expectations.md. The probe used sizes outside the registered grid; none
of the registered A/B cells were executed before the freeze, and the
mechanism is source-derived, not fitted. This mirrors the M1 probe
convention (M1 P1/P2) and is disclosed rather than hidden inside the
registration.

## Check A: standalone pairwise all-to-allv on rnic-nn-fluid

Registered: JCT = ceil(S * 8e12 / floor(400e9 / (W-1))) + 2,000,000 ps,
exact.

| Check | S bytes/pair | W | predicted ps | measured ps | residual | match |
|---|---|---|---|---|---|---|
| A1 | 65,536 | 2 | 3,310,720 | 3,310,720 | 0 | yes |
| A2 | 65,536 | 4 | 5,932,161 | 5,932,161 | 0 | yes |
| A3 | 65,536 | 8 | 11,175,041 | 11,175,041 | 0 | yes |
| A4 | 16,777,216 | 2 | 337,544,320 | 337,544,320 | 0 | yes |
| A5 | 16,777,216 | 4 | 1,008,632,961 | 1,008,632,961 | 0 | yes |
| A6 | 16,777,216 | 8 | 2,350,810,241 | 2,350,810,241 | 0 | yes |

All pass, including the registered +1 ps whole-bps-floor and
whole-ps-ceil quantization at W in {4, 8}. This validates the fluid
pairwise all-to-allv closed form end to end and closes the pairwise-a2av
part of TRAF-4 (binomial broadcast remains).

## Check B: MoE step makespan on rnic-nn-fluid

Registered: makespan = L * c * 1000 + 2L * JCT_a2av(S_pair, W), exact,
with the frozen roofline inputs E and c (granite-3.0-1b-a400m-like dims,
EP-only step through `HtsimStepSink` with `ep_ranks`, TP world of 1).

| Check | shape | EP | E matches frozen | c matches frozen | predicted ps | measured ps | residual | match |
|---|---|---|---|---|---|---|---|---|
| B1 | decode8x2048 | 2 | yes (404,450,742) | yes (16,852) | 563,362,560 | 563,362,560 | 0 | yes |
| B2 | decode8x2048 | 4 | yes (296,597,211) | yes (12,358) | 486,963,888 | 486,963,888 | 0 | yes |
| B3 | decode8x2048 | 8 | yes (242,670,445) | yes (10,111) | 448,764,528 | 448,764,528 | 0 | yes |
| B4 | prefill2048 | 2 | yes (1,390,831,206) | yes (57,951) | 17,592,951,360 | 17,592,951,360 | 0 | yes |
| B5 | prefill2048 | 4 | yes (1,390,831,206) | yes (57,951) | 25,646,015,088 | 25,646,015,088 | 0 | yes |
| B6 | prefill2048 | 8 | yes (1,390,831,206) | yes (57,951) | 29,672,546,928 | 29,672,546,928 | 0 | yes |

All pass. The registered structure (decode memory-bound with E shrinking
as resident experts shrink, prefill compute-bound with E invariant in W)
showed up exactly as frozen.

## Check C: registered qualitative directions

- C-q1 (a2av time grows with EP width at fixed total payload): the
  network component of the B makespan (measured minus the frozen
  L * c * 1000) is strictly increasing in W for both shapes. decode:
  158,914,560 / 190,371,888 / 206,100,528 ps for EP = 2/4/8; prefill:
  16,202,127,360 / 24,255,191,088 / 28,281,722,928 ps. Pass.
- C-q2 (fluid vs nn ordering): the check-A grid on `rnic-nn` dominates
  fluid on every cell. nn JCTs: 3,414,400 / 6,076,800 / 11,401,600 ps at
  64 KiB per pair and 342,870,400 / 1,024,444,800 / 2,387,593,600 ps at
  16 MiB per pair (W = 2/4/8). Pass on all 6 cells.
- C-q3 (makespan monotonicity, implied by the frozen B table): decode
  strictly decreasing in W (563.4 / 487.0 / 448.8 us), prefill strictly
  increasing (17.59 / 25.65 / 29.67 ms). Pass on both shapes.

Unregistered observation (report only, no pre-registration weight): every
measured nn a2av JCT equals ((W-1) * Npkt + 1) * 83,200 + 2,000,000 ps
exactly, with Npkt = S / 4096 the full packets per flow: the packetized
egress serializes all (W-1) concurrent flows' packets back to back and
the single store-and-forward slot of the M1 C3 intercept appears once.
All 6 cells match this form to 0 ps. It was not registered (M1 C7 saw
final-packet pipelining under contention, so a point form was withheld),
and registering plus validating it belongs with the TRAF-4 remainder.

## Artifacts

The historical external artifact directory, whose resolved historical path is
intentionally omitted, contains `summary.csv` (22 rows), per-run GOAL
text/binary and completion CSVs (`a2av-*` for checks A and C-q2, `moe-*`
for check B). New runs default to `${SIMLLM_DATA_ROOT}/m5`.

## Audit note (2026-08-04, folded after the independent verify pass)

The mathematician audit rederived every A and B cell digit by digit in
integer arithmetic, reran A3, A5, B2 and B6 fresh (0 ps residual), and
confirmed the harness compares only frozen literals. Its findings, all
folded: C-q1 and C-q3 are arithmetic consequences of the frozen B table
once B passes at 0 ps (only C-q2 carries independent content; A1/A4
likewise reduce to the M1 single-flow law, so check A's new content is
the W = 4 and 8 cells); the harness now exits nonzero on any failed
verdict and refuses to claim direction checks under a partial
--sections run; the largest MoE fidelity gap (one hidden-vector copy
per token-expert assignment instead of per-token-per-rank dedupe, up to
top_k / W byte inflation) is now a named sub-approximation under
TRAF-2; the attention-pairs half-pair convention (n * prior + n * n / 2,
about 1e-4 of prefill flops) is a documented model convention the
frozen E values inherit, not an M5 change; and two provider docstring
imprecisions were corrected. The float-envelope note (family-sum
equality above 2^53 flops) is COMP-8; current scales sit far below it.
