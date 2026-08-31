# Collective floor calibration, third freeze: D8 coordinate mapping

The adversarial review of attempt-0002 found that Family D8 queried the fitted
authority at the wrong coordinate. This third freeze records the integrator's
accepted coordinate interpretation before the repaired implementation and
attempt-0003. It is a post-specified correction, not a public pre-registration.
The first two freezes and `study_config.json` remain immutable.

## The operation-buffer coordinate

The calibrated authority's byte axis is the external table's operation-buffer
coordinate converted to bytes. The MiniMax external arm constructs its NCCL
query as `tokens_per_rank * hidden_size * width` elements. At expert
parallelism 8 that query is 98,304 half-precision elements. Half precision is
two bytes per element, so the semantically matched authority query is:

`98,304 elements * 2 bytes/element = 196,608 bytes`

This operation-buffer coordinate is not the 172,032 physical bytes emitted by
each endpoint in the simulated ring. It is also not the 344,064-byte query in
attempt-0002, which incorrectly treated those already physical endpoint bytes
as source elements and multiplied them by two.

## Corrected Family D8 scoring

Family D8 keeps the original external total, quotient band, layer count, and
all other frozen arithmetic. The band remains `[0.90, 1.10]`.

- The literal 172,032-byte physical-endpoint reading is published as an
  unscored diagnostic. Its expected 65-layer phase-pair total is
  2.060523530 ms and its quotient against 1.922050 ms is 1.072044707.
- The semantically matched 196,608-byte operation-buffer reading is scored.
  Its expected 65-layer phase-pair total is 2.131828400 ms and its quotient is
  1.109143050.
- The matched quotient is above the unchanged 1.10 ceiling. D8 is therefore
  expected to publish as `REFUTED`, with the coordinate arithmetic and both
  readings shown.

This correction deliberately withdraws attempt-0002's D8 pass. A repaired run
must not score either the 172,032-byte physical-endpoint reading or the doubled
344,064-byte query. Family D8 remains a scored refutation and does not void the
other evidence classes when every fatal guard holds.

## Carry-forward rules

All fatal guards and the H, B, M, and W family definitions carry forward
unchanged except for repairs required to make their already frozen claims
literal. In particular, the Family H band does not move, Family B must compare
against the pre-wave implementation, and TRAF-76 remains open after the run.
