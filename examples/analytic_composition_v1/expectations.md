# Analytic composition v1 expectations

These expectations freeze wave P-2 of the deployment planning ladder: the
`analytic-composition` network level, an in-process fabric substitute that
prices the sink's supported collectives from the landed collective-latency
profiles, curves, locality split and registration ledger under the COMP-75
maximum composition, with zero subprocesses on its own arms. They are
committed before any implementation of that level exists. Every existing
network level, artifact and diagnostic stays byte-identical; the analytic
level is a new selection, never a change to an old one.

Scoring-fidelity clause, binding on the runner (the P0-1 lesson): every
scored family names, in its emitted record, the mechanism by which it could
have failed, and no scored predicate may re-check an invariant that a
constructor or parser already enforces on every object that can exist. A
family whose predicate is discovered to be constructor-entailed is reported
unscored with a finding, never as a pass.

## Frozen synthetic profile P*

All synthetic cells use one frozen profile, stated inline so the oracles are
self-contained integer arithmetic:

- Base latency K(2) = 10,000 ps and K(4) = 20,000 ps; K applies once per
  collective when the critical endpoint byte count is positive.
- Flat bandwidth B = 1,000,000,000 bytes per second.
- Propagation reference P = 2,000,000 ps, charged once per fabric-bearing
  phase.
- A width-2 effective-bandwidth curve with exact anchors
  (4 bytes, 1,000,000,000 B/s) and (8 bytes, 2,000,000,000 B/s), selected by
  full semantic endpoint bytes; widths without a curve use the flat B.
- Registration: one channel at 20,000,000 ps, charged once per first-use
  semantic identity, zero on reuse.
- Ceiling division throughout: (numerator + denominator - 1) // denominator;
  local phases quantize to whole nanoseconds (1,000 ps steps), fabric terms
  to whole picoseconds. No floating point participates in any oracle.

The composition per collective: registration + base + the sum over phases of
max(local term, fabric term); per step: T = max(C, sum of collective
prices), the COMP-75 operator, with C the lowered compute service.

## Fatal guards (violation voids the run)

- FG-1 analytic arms are process-free: subprocess.Popen and os.posix_spawn
  are intercepted around every analytic-arm evaluation, and the native
  txt2bin and htsim entry points are additionally patched to raise on those
  arms. One firing voids the run. The fluid comparison arm of family E8 and
  the wall-time family W run outside this window and are expected to spawn
  exactly their counted processes.
- FG-2 byte identity of everything that exists today: the repository's
  locked identity tests pass unchanged, including the precision-surface
  agreeing-surface artifact, command and authority locks, the collective
  latency legacy and none identity paths, the fixed-cost envelope off arm,
  the registration disabled paths, and the StepRecord wire-format lock. The
  existing fluid-versus-composed-native refusal diagnostic stays a byte-
  exact literal. Machine-checkable: the pre-existing test list is enumerated
  in the run record and every test passes on the implementation commit.
- FG-3 refusal matrix: selecting analytic-composition with composed-native
  RNIC hardware is refused; selecting it with no resolvable collective
  profile (none, legacy, or an envelope off arm) is refused; a topology
  file, a dependency cross-check, or a nondefault link speed under the
  analytic level is refused; every refusal happens before any output
  artifact exists.
- FG-4 strict schemas: any new record the study emits round-trips strictly.
- FG-5 chronology: verified by a shallow-clone-safe CI test and by the
  integrator, with the runner's record naming that venue; this freeze
  commit contains only this file, and RESULTS.md cites its hash.

No fatal guard is declared survivable.

## Family E: exact oracles (scored)

E1 width-2 all-remote ring, source payload 8 bytes: chunk 4, rounds 2,
endpoint bytes 8 select the 2 GB/s curve anchor; price exactly
10,000 + 2 x (2,000,000 + 2,000) = 4,014,000 ps.

E2 width-4 mixed ring, source payload 16, placement [a,a,b,b]: rounds 6,
local term 4,000 ps, fabric term 2,004,000 ps; price exactly
20,000 + 6 x 2,004,000 = 12,044,000 ps.

E3 width-4 uniform mixed all-to-allv, per-pair 4 bytes, placement
[a,a,b,b]: local 4,000 ps, fabric 2,000,000 + 8,000; price exactly
20,000 + 2,008,000 = 2,028,000 ps.

E4 width-4 sparse all-remote all-to-allv with source-major pairs
(0 to 2: 3 bytes), (1 to 2: 5 bytes), (3 to 2: 7 bytes): the critical
endpoint is rank 2 ingress at 15 bytes; price exactly
20,000 + 2,000,000 + 15,000 = 2,035,000 ps.

E5 curve discriminator, width-2 all-remote pairwise, 8 bytes: with the
curve, 10,000 + 2,000,000 + 4,000 = 2,014,000 ps; the flat-bandwidth
reading would be 2,018,000 ps; the frozen difference is exactly 4,000 ps
and the run must demonstrate the curve reading.

E6 registration and reuse: the first use of the E1 site prices exactly
20,000,000 + 4,014,000 = 24,014,000 ps; an identical reuse prices exactly
4,014,000 ps.

E7 COMP-75 maximum: two E1 collectives without registration give
Q = 8,028,000 ps; with compute 9,000,000 ps the step is exactly
9,000,000 ps (communication hidden), and with compute 8,000,000 ps the
step is exactly 8,028,000 ps (compute hidden). Both cells are evaluated.

E8 analytic versus fluid discrimination: the E1 record priced through the
existing fluid subprocess path with the same profile surcharges gives
exactly 10,000 + 2 x (80 + 2,000,000) = 4,010,160 ps; the analytic level
gives 4,014,000 ps; the frozen difference is exactly 3,840 ps and its
mechanism is the serialization rate (the analytic level uses the selected
2 GB/s curve anchor, the fluid transport its native 400 Gbit/s manifold).
Both arms run for real; the fluid arm spawns exactly one txt2bin and one
htsim process.

E9 live chain: one request whose two steps each execute two E1 sites with
compute 8,000,000 ps: step 0 prices exactly
2 x (20,000,000 + 4,014,000) = 48,028,000 ps so TTFT is exactly
48,028,000 ps; step 1 prices exactly 8,028,000 ps so the request completes
at 56,056,000 ps and TPOT is exactly 8,028,000 ps. The cell is expressed
as StepRecord inputs, StepResult outputs and the reducer's request totals
through the standard metric chain.

## Family W: wall time and process counts (scored)

On the E8 record, after one unscored warm-up per arm: the analytic arm's
subprocess count is exactly zero (fatal via FG-1); the fluid arm's count
is exactly two per invocation (scored exact); over five paired repetitions,
ten times the median analytic wall seconds is at most the median fluid
wall seconds (scored; machine disclosed; the historical 7.252 s per
invocation calibration figure is context, never a required outcome).

## Direction and structure (scored unless noted)

D1: for each E-cell with a fabric phase, halving the profile bandwidth
never decreases the price, and doubling it never increases the price.
D2: GOAL-rank padding invariance (fatal-unscored): changing only the
rendered rank padding leaves every analytic price identical.
D3: registration identity (fatal-unscored): the analytic level charges
registration through the existing ledger; a second identical site within
one step charges zero on reuse exactly as the ledger's landed tests state.

## Closure

This study validates the analytic-composition level's arithmetic,
selection surface, refusals, zero-subprocess property, discrimination
against the fluid path, and live reachability to TTFT and TPOT. It makes
no packet-behavior claim, no statistical claim (TRAF-19), no LogGOPSim
claim (TRAF-20), and no calibration claim for registration (TRAF-56) or
transferred profiles. Scored families are E1 to E9, W, and D1, reported in
their classes and never summed with fatal rows.
