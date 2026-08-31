# Matched-seam frontier expectations, second freeze

The first freeze's FG-1 was mis-specified and the run published under it
is void. This second freeze corrects the specification and defines a
fresh scored run. Nothing here widens a band that fired: the first
freeze's bands are carried forward unchanged, and the void is published
rather than repaired by relabelling.

## Why the first run is void

FG-1 required that "no roofline term, no declared efficiency and no
fitted constant appears anywhere in the scored arm." Two things make
that unsatisfiable for any study that prices from the imported database:

- The external resolver is speed-of-light normalized by construction. It
  divides a measured neighbour latency by that neighbour's analytical
  roofline, blends in utilization space, and reconstructs with the query
  roofline. The scored TP4 batch-64 cell alone evaluates that roofline
  240 times across 15 GEMM shapes.
- The external composition applies its own published empirical factors:
  prefill latency 1.1, decode latency 1.08, prefill rate matching 0.92,
  decode rate matching 0.9, and an autoscale heuristic of 1.8. Their
  source documents the first two as tuning factors for predictions that
  are too optimistic and the rate factors as silicon-calibrated.

Reproducing their pricing requires adopting their composition semantics,
including both of the above. A guard forbidding them forbids the study's
own purpose. That is an authoring error in the first freeze, and the run
made under it is void, not passing.

## The corrected contract

The claim this study can honestly test is narrower and still worth
testing: **our composition adds no timing model of its own.** Every
duration comes from their database through their documented composition
semantics, and any adjustment we apply is one of theirs, adopted
verbatim, declared, and attributed to its source location.

- FG-1a no SimLLM-side timing model: no roofline, declared efficiency,
  fitted constant or curve of our own authorship reaches a scored value.
  The deploy estimator's own RooflineProvider must be provably bypassed.
- FG-1b declared external adjustments: every external factor applied is
  listed in a tracked table with its value, its exact source file and
  line in the pinned installation, and the external documentation of
  what it compensates. An applied factor missing from the table voids
  the run. The table is frozen with this document.
- FG-1c disclosed sensitivity: for each applied factor, the run
  publishes the Family R quotients recomputed with that factor removed,
  so a reader sees exactly how much of the agreement each one carries.
  This is published evidence, not a scored family.
- FG-2 through FG-5 carry forward unchanged from the first freeze.
- FG-6 corrected: determinism compares the complete scored record byte
  for byte across two full evaluations in fresh processes, not parsed
  dictionaries from a single evaluation. Wall time is excluded by name.

## Families

Families S, R, F, M, D and W carry forward with their frozen bands
unchanged, including the [0.98, 1.02] decode band and the [0.75, 1.35]
frontier bracket. Two additions:

- Family R gains a disclosure row per applied external factor, carrying
  the recomputed quotient range without that factor. Unscored.
- Family F gains a boundary-proximity row: every external row whose
  exact local x lies within one published rounding unit of its published
  x is listed, not only the row that failed. The first run's lens found
  rows 1, 2, 6, 7 and 9 all cross that boundary while only row 9 became
  visible.

## Family M, corrected scope

The first run set the ideal arm's network service to exactly zero, so
the measured 1.042715399805 prices full network transfer against no
network charge. It does not isolate receiver-side fan-in serialization.
Two consequences, both binding:

- The published quantity is renamed to what it measures: the ratio of a
  packet-priced network to an unpriced network at this workload.
- If a receiver-serialization-specific number is wanted, it requires a
  third arm whose ideal rung charges LogGOPSim's L, o, g and G terms
  rather than zero. That arm is optional in this wave; if it is not run,
  the study says so and makes no isolated-mechanism claim.

## The figure, corrected

The arrow label must state what was measured: the network cost their
planner class does not price at all, at this workload's measured ratio,
with our unpriced-network arm named as charging zero network service.
The receiver-side fan-in wording is removed unless the third arm above
is actually run. The fan-in envelope from the earlier waves stays named
as a different schedule regime, as already required.

## Closure

A full pass under this freeze establishes that our composition adds no
timing model of its own and reproduces their published serving numbers
from their database under their declared semantics, with the
contribution of each of their empirical factors disclosed. It does not
claim zero fitted constants, because theirs are not zero, and it does
not isolate a single network mechanism unless the third arm runs.
