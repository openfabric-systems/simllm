# Collective floor calibration expectations

These expectations freeze the TRAF-76 aggregate slice: calibrating the
packet arm's intra-node collective completion against the imported
measured H200 NCCL table, so a zero-fan-in collective no longer prices
as bare link serialization. They are committed before any
implementation exists. The design basis is the executed read-only probe
of 2026-08-31; its inventory numbers are restated here as frozen facts.

The registry acceptance text of TRAF-76 is binding but not fully
satisfiable from this evidence: the table is completion-latency truth
with no packet-mechanism fields, so this study calibrates the aggregate
completion authority and leaves packet-mechanism calibration (credits,
geometry, switch behavior, arbitration) explicitly open in the narrowed
entry. Closing TRAF-76 outright is NOT a permitted outcome of this
study.

## Frozen calibration truth

The tracked artifact e432db694195110aa39c1e1eccf1accda012e69ef68e9521
0d049809bb93f015 (aiconfigurator 0.11.0, h200_sxm, NCCL database
2.26.2, row version 2.29.2): 1,008 raw rows, 504 unique coordinates
under first-row-wins, ranks 2, 4 and 8, four operations, half and int8,
21 message sizes per curve. The half-precision all-gather and
reduce-scatter families relevant to the packet arm hold 126 unique
cells. Raw rows carry MEASURED-EXTERNAL.

## Frozen fit protocol

- Training half: at each rank and operation, the cells at even size
  indices for all-gather and at odd size indices for reduce-scatter
  (index 0 = 256 bytes ascending). Holdout half: the complementary
  indices. Exactly 63 training and 63 holdout cells. No holdout cell
  influences any fitted value; the ignored duplicate rows are used for
  nothing.
- Model per operation, rank and regime: one aggregate completion floor
  plus one effective-bandwidth slope,
  T_ps = floor_ps + bytes * slope_ps_per_byte. The regime boundaries
  (at most three regimes per curve) are chosen from training cells
  only and frozen in the study configuration in a commit that precedes
  the fit implementation. No separately named launch, synchronization,
  per-rank or algorithm-selection term exists anywhere: the probe
  showed the data cannot discriminate them (rank-8 single-slope R2
  0.070 to 0.209), and naming them would be invented evidence.
- The message-size axis is pinned to BYTES with resolver evidence
  cited in the study configuration before any fit runs. If the
  evidence shows elements instead, the freeze value here is wrong and
  the run is VOID with that finding published; the axis is never
  silently reinterpreted.

## Fatal guards

- FG-1 no invented terms: the calibrated authority consists of exactly
  the fitted floors, slopes and frozen regime boundaries. Any other
  constant reaching a calibrated value voids the run.
- FG-2 no double counting: in the calibrated path, the sink's
  registration charge, semantic collective base surcharge and host
  launch model are each either proven disjoint from the fitted floor
  (with the disjointness argument written down) or disabled for the
  calibrated collective; a test constructs a case where double
  charging would show and proves it does not.
- FG-3 evidence classes: fitted values carry calibrated with the full
  source identity (artifact hash, operation, rank, dtype, regime,
  training-cell list); applied outside their fitted operation, dtype,
  rank set or size range they carry transferred-at-use and a test
  proves the downgrade fires; nothing fitted is ever served as
  MEASURED or MEASURED-EXTERNAL.
- FG-4 exact bypass: with the calibration disabled, phase and step
  timestamps, local and fabric segment tuples, application and wire
  byte counts, completion order, backend invocation order and
  random-generator state are IDENTICAL to the pre-wave path, proven by
  a byte-level comparison test on a pinned scenario. The bypass branch
  occurs before any calibrated construction.
- FG-5 A100 fence: nothing from the A100 candidate profile (packet
  geometry, credits, link counts, rates, buffers, arbitration) enters
  any H200-calibrated value; a scan of the calibrated authority's
  inputs is the guard. NVLink module consumption anywhere in this
  study pins policy and parameter set explicitly.
- FG-6 determinism: the complete scored record reproduces byte for
  byte across two full evaluations in fresh processes, wall time
  excluded by name.
- FG-7 chronology.

## Family H: held-out reproduction (scored)

For all 63 held-out cells, the calibrated authority's completion for
that operation, rank, dtype and message size reproduces the measured
cell within 10 percent relative error. The registry bar is the larger
of 10 percent and two GPU cycles; at these microsecond latencies the
10 percent side dominates and is the numeric bar, with the two-cycle
equivalent reported informationally against a sourced H200 clock named
in the record. Scored per cell, 63 cells. The before error (the
current bare-serialization path against the same cells) is published
beside it; the expected direction is that calibration reduces median
error by at least an order of magnitude, and that expectation is
falsifiable by the published numbers.

## Family B: bypass identity (scored)

The FG-4 comparison scenario scores as its own family row: identical
is pass, any divergence is fail with the first divergent field named.

## Family D8: the zero-fan-in cell repriced (scored)

The MiniMax dense EP-8 phase pair (172,032 bytes per endpoint per
phase, all intra-node), repriced through the calibrated authority,
lands within [0.90, 1.10] of the external arm's 1.92205 ms over 65
layers. The current path prices 0.04979 ms (quotient 0.02590); the
calibrated quotient is expected inside the band because both sides now
derive from the same measured table family, and a miss is published as
a refutation with the diverging regime named. EP-32 and EP-128 local
components are repriced and PUBLISHED with their fabric composition
unchanged, unscored, because their cross-node legs remain TRAF-77
territory; presenting them as calibrated end-to-end cells would repeat
the confound this wave exists to remove.

## Family M: metric-chain demonstration (scored)

One supported-chain run (StepRecord through HtsimStepSink to
HtsimRequestMetricReducer, the nccl_registration_v1 template) executes
twice, calibration off and on, everything else identical. Scored: the
off arm reproduces the pre-wave TTFT and TPOT exactly; the on arm
moves both in the expected direction (larger, because a real
completion floor replaces bare serialization) by an amount consistent
with the calibrated floors, with the arithmetic shown in the record.

## Family W: wall time (scored, generous)

Conversion, fits, all scored families and the record complete in at
most 600 s in one process, machine disclosed.

## Closure

A full pass lands the aggregate completion authority and its seam,
proves the bypass exact, and demonstrates the metric-chain effect.
TRAF-76 then NARROWS to the packet-mechanism remainder (credits,
geometry, switch behavior, arbitration, nonzero-fan-in integration)
quoting the held-out numbers; it does not close. TRAF-77 is untouched
by this study. Scored families are H, B, D8, M and W, in their
classes, never summed.
