# Deployment scan v1 expectations

These expectations freeze the first wave of the deployment planning mode:
the installed candidate schema, capacity estimator and frontier driver
(`simllm/deploy/`), validated as one wave-level study. They are committed
before any implementation of that module exists, per the repository
validation discipline. The estimator is a registered model class distinct
from simulation: every number it produces carries an estimator stamp and a
point class, and nothing in this study claims simulation or silicon
accuracy. The study is backend-free by construction; proving that property
is a fatal guard, not a footnote.

## Frozen inputs

All inputs are tracked at the base commit of this branch and pinned by
SHA-256. A hash mismatch at run time is fatal.

- `examples/deployment_frontier_v1/expectations.json`
  `54295c81cebe36ee32d12b8ab1432c9fc060094ddf98403152b0d619cc37438f`
  (the CORE-62 frozen deployments, batch sweep, GPU envelopes, network
  inputs and byte-partition rules; this study reuses them verbatim).
- `offline/calibration/deployment-projections/ee154ed5f07c104269df9cf60d8730b8c6dced0ccf619fb7ff7146ec2ddfd5a2.json`
  `ee154ed5f07c104269df9cf60d8730b8c6dced0ccf619fb7ff7146ec2ddfd5a2`
  (the DeepSeek-V3 model inventory the kernel work derives from).
- `examples/deployment_frontier_v1/result.json`
  `f2f216068bf5ba914853c62a2ee965ede0ebfc0a6f29e3d11cfa5f45eac359ad`
  (the published CORE-62 run of record; its per-point integers are the
  compatibility oracles below, and its fabric and intra excess terms are the
  SIM-DERIVED inputs of family C2).

Synthetic cells (families E and W) use study-declared inputs only, stated
inline below, so their oracles are self-contained hand arithmetic and do not
depend on shipped envelope literals.

## Grid

- Compatibility grid: the three frozen CORE-62 configurations
  (`b100-one-node-intra`, `h100-two-node-serialized`,
  `h100-nine-node-incast`) at batch per GPU 1, 2, 4, 8, 16, 32 (18 points).
- Bandwidth grid: the same 18 points re-priced at inter-node nominal rates
  200e9 and 100e9 bits per second (36 points).
- SLA thresholds: TPOT targets 4,000,000,000 ps and 8,500,000,000 ps applied
  to the compatibility grid.
- Synthetic cells: defined inline in families E1 to E6 and W1c.

## Fatal guards (violation voids the run)

- FG-1 zero subprocess: process creation is intercepted (a monkeypatched
  `subprocess.Popen` and `os.posix_spawn` that raise) around every scan and
  estimator call in the study; one interception firing voids the run.
- FG-2 stamp: every emitted point carries a strict
  `simllm-deployment-estimate-v1` stamp; point class is ESTIMATE everywhere
  except family C2, whose points are SIMULATED because their excess terms are
  consumed from the pinned record.
- FG-3 evidence: every priced term carries an evidence class from
  {MEASURED, ROOFLINE, DECLARED, SIM-DERIVED} and a nonempty source; in this
  study the kernel floor is ROOFLINE, link floors and handoff are DECLARED,
  batch-surface service is MEASURED (synthetic points declared inline are
  labeled DECLARED), and C2 excess terms are SIM-DERIVED.
- FG-4 strict schemas: candidate, estimate stamp and frontier record round
  trip strictly (unknown fields rejected, schema tags checked first).
- FG-5 input pinning: the three frozen input hashes above verify before any
  family is evaluated.
- FG-6 chronology: RESULTS.md cites this file's commit hash, and that commit
  contains no implementation of `simllm/deploy/`.

No fatal guard is declared survivable.

## Family C1: analytic reproduction (exact, 18 cells, scored)

The installed estimator, given the frozen CORE-62 inputs, reproduces the
published `analytical_step_ps` of every point of the run of record exactly
(0 ps), reading the oracle integers from the pinned `result.json`. Spot
literals, hand-copied here so the family cannot drift with the record:
`b100-one-node-intra` batch 1 is 3,448,398,380 ps and batch 32 is
4,257,218,560 ps; `h100-two-node-serialized` batch 1 is 8,234,981,205 ps and
batch 32 is 9,535,537,623 ps. This family carries genuine risk: it requires
the new module to reproduce the kernel-work derivation, byte partition,
floor-division floors and max-composition of the study-local code from the
frozen inputs alone.

## Family C2: simulated reproduction (exact, 18 cells, scored)

Consuming the pinned record's fabric and intra-node excess terms as
SIM-DERIVED inputs, the estimator's telescoping composition reproduces the
published `simulated_step_ps` of every point exactly (0 ps). Spot literal:
`b100-one-node-intra` batch 32 is 4,523,298,348 ps, the only point of the
grid where simulated differs from analytic. Points of this family carry
point class SIMULATED.

## Family C3: coordinate transform (exact, 18 cells, scored thin)

For every C2 point, the frontier record's coordinates equal the exact
reduced fractions x = 10^12 / step_ps and
y = batch_per_gpu x 10^12 / step_ps, matching the run of record's
`simulated_operating_point`. Disclosed: conditional on C2, this family
checks only the transform and record plumbing, so it is scored but thin;
it is kept scored because the transform is new installed code.

## Family E: synthetic exact oracles (scored)

Study-declared envelope for E1: peak 8e15 flops per second, HBM 8e12 bytes
per second, efficiency 1.0.

- E1 roofline: a kernel of 8e12 flops and 1e10 HBM bytes prices to exactly
  1,250,000,000 ps (memory bound).
- E2 fabric floor: a largest single flow of 5e8 bytes at 400e9 bits per
  second prices to exactly 10,000,000,000 ps under the floor-division form.
- E3 intra floor: 9e8 logical bytes at 450e9 bytes per second prices to
  exactly 2,000,000,000 ps.
- E4 surface interpolation: DECLARED batch points (2, 200,000,000 ps) and
  (8, 800,000,000 ps) interpolate at batch 4 to exactly 400,000,000 ps.
- E5 queue: with the E4 surface, output tokens 4, max batch 8, one decode
  engine and cell size 64 requests: capacity is exactly 2,500 requests per
  second; offered 500 gives occupancy 2 and zero overload wait; offered
  2,000 gives occupancy 7 and zero overload wait; offered 4,000 gives
  occupancy 8 and overload wait exactly 4,725,000,000 ps.
- E6 rate match: at 100 requests per second, prefill service 5e10 ps per
  request requires exactly 5 prefill engines; a decode surface with
  (2, 100,000,000 ps) and (8, 400,000,000 ps), output tokens 4 and max
  batch 8 gives per-engine capacity exactly 5,000 requests per second and
  requires exactly 1 decode engine.

## Family B1: bandwidth composition (exact, 36 cells, scored)

For each compatibility point re-priced at inter-node rate R in {200e9,
100e9}: the analytic step equals
max(kernel_floor_ps, floor(max_flow_bytes x 8 x 10^12 / R),
intra_floor_ps), where kernel_floor_ps, max_flow_bytes and intra_floor_ps
are the run's own emitted components for that point. The family is anchored
by C1: if C1 fails, B1 is not interpretable and is reported unevaluated.
The floor is recomputed at R with the same floor-division form, never
scaled by multiplication, because floor division does not commute with
doubling.

Direction (scored, one instance): for every point, the 100e9 analytic step
is greater than or equal to the 200e9 step, which is greater than or equal
to the 400e9 step, with strict increase wherever the fabric floor is the
binding term at the lower rate.

## Family S1: SLA membership (exact sets, scored)

Applying TPOT targets to the compatibility grid's simulated steps:

- Target 4,000,000,000 ps admits exactly the five points
  `b100-one-node-intra` batches 1, 2, 4, 8, 16.
- Target 8,500,000,000 ps admits exactly the six `b100-one-node-intra`
  points plus batches 1, 2, 4 of both h100 configurations (12 points).

## Family P1: Pareto front (exact set, scored)

The Pareto front of the 18-point compatibility grid (maximize both axes,
dominance with at least one strict inequality, ties kept) is exactly the
six `b100-one-node-intra` points priced by their simulated steps. Hand
derivation, frozen: within the b100 configuration x is strictly decreasing
and y strictly increasing in batch, so its six points are mutually
non-dominated; the b100 batch-32 point (x = 10^12/4,523,298,348,
y = 32 x 10^12/4,523,298,348) exceeds every h100 point in both axes (the
largest h100 x is 10^12/8,234,981,205 and the largest h100 y is
32 x 10^12/9,535,537,623), so every h100 point is dominated; the two h100
configurations coincide point for point and both fall out.

## Family W1: wall time (scored, generous by construction)

- W1a: the complete study scan (all families, at least 64 priced points)
  completes in at most 10 seconds of wall time measured around the scan and
  estimator calls only, single process.
- W1c: a synthetic throughput grid of 1,000 generated candidates (schema
  variants of the E1 candidate) at 6 batch values each, 6,000 points,
  scans in at most 60 seconds single process.
- W1b (reported, unscored): points per second for both scans, with machine
  disclosure.

The bands are deliberately generous; the scored claim is the order of
magnitude of the backend-free path, not a tight ceiling. Wall time is the
only nondeterministic quantity in the study and both bands are declared
robust to machine noise at these widths.

## Expected directions (scored unless noted)

- D1: per-request speed x is nonincreasing in batch within every
  configuration.
- D2: per-GPU throughput y is nondecreasing in batch within every
  configuration.
- D3: tightening either SLA target never grows the admitted set
  (evaluated on the two frozen thresholds plus the trivial infinite
  target).
- D4 (structural, fatal-unscored): every feasibility-rejected candidate
  carries a stable reason code and no points; a candidate with
  pipeline_parallel greater than 1 is rejected with
  `pipeline-parallel-unpriced`.

## Closure

This study validates the wave spine: schema, estimator, frontier driver and
their composition, the zero-subprocess property, and exact compatibility
with the CORE-62 run of record. It does not validate absolute accuracy of
any price (the decode calibration gap against the published DeepSeek anchor
remains open under CORE-54 and successors), does not price parallel widths
beyond the three frozen configurations, and does not touch PrecisionConfig.
Scored families are C1, C2, C3, E1 to E6, B1, S1, P1, W1a, W1c, D1 to D3,
reported in their classes and never summed with fatal or entailed rows.
