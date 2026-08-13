# SGLang worker seam v1 results

Nonvoid. No fatal guard was violated, so the scored numbers mean what the
freeze says they mean.

**Scored, genuine risk: 82 of 82 exact-oracle rows.** Separately, 106 of 106
entailed conformance rows held, and none of the seven fatal guards was
violated. These counts belong to different evidence classes and are never
summed.

## Chronology

- Expectations frozen by `6981d34`, which contains `expectations.md`,
  `expectations.json` and the artifact-free `--check-only` gate only. That
  gate was run before the commit and re-derived every frozen literal from the
  model geometry and row shapes.
- The implementation landed in `6af1186`, after the freeze and before any
  measurement.
- The measuring harness and this report landed after the first and only
  measured run. Nothing in `expectations.md` or `expectations.json` was
  edited after `6981d34`: no modeled behavior was changed after a
  measurement, and no measurement was taken before the freeze.

## What was measured

One stub SGLang batch stream per cell, replayed through the adapter's own
seam and the shared metric chain, twice:

```
stub ScheduleBatch -> observe_schedule_batch -> SglStepTranslator
  -> StepRecord -> ObservedStepLowerer -> ExecutionGraph
  -> CoarseDeviceRuntime -> CompletionReducer -> StepResult -> TTFT, TPOT
```

The stubs carry SGLang's own attribute names and leave the sampled-row
decision entirely to the adapter: the chunked request keeps a positive
``inflight_middle_chunks`` on every chunk before its last one, exactly as the
scheduler leaves it, and nothing in the harness declares which rows sample.

## Physical sanity, against the bounds frozen before the run

- Floor: 15,009,316,864 ps, the weight and LM-head bytes over the fast
  effective bandwidth. The smallest measured step is 15,061,728,000 ps, which
  sits 0.35 percent above the floor, and the largest fast-bandwidth step is
  15,167,232,000 ps against the frozen 15,167,258,624 ps ceiling. Every step
  is inside the band, and the two ends differ by exactly the KV bytes of the
  scheduled contexts.
- Against the real system: the decode step is 15,140,512,000 ps, i.e. 66
  tokens per second for an 8B model at an effective 1.0 TB/s. A real 8B bf16
  decode reads about 15 GB per token, so an H100 at 60 to 80 percent of
  3.35 TB/s lands at 130 to 200 tokens per second. The fixture is slower in
  the same proportion as its effective bandwidth is lower, which is the right
  order of magnitude.
- The paired 700-token prefill step needed 9.87 ms of compute against 15.10 ms
  of weight streaming, so the roofline priced it memory-bound. That is the
  known coarseness of a `max()` roofline for a large prefill; it is identical
  in both arms and cannot touch the attribution being measured.

## The defect, in reported metrics

Every value below is the frozen prediction and the measurement, which agreed
exactly.

| cell | compatibility TTFT (ps) | fixed TTFT (ps) | error (ps) | compat tokens |
|---|---:|---:|---:|---:|
| solo-2chunk-fast | 15,087,936,000 | 30,228,320,000 | 15,140,384,000 | 3 |
| solo-3chunk-fast | 15,061,728,000 | 45,303,168,000 | 30,241,440,000 | 4 |
| mixed-2chunk-fast | 15,114,304,000 | 30,281,152,000 | 15,166,848,000 | 3 |
| mixed-3chunk-fast | 15,088,064,000 | 45,382,560,000 | 30,294,496,000 | 4 |
| paired-2chunk-fast | 15,101,056,000 | 30,241,440,000 | 15,140,384,000 | 3 |
| paired-3chunk-fast | 15,074,848,000 | 45,316,288,000 | 30,241,440,000 | 4 |
| solo-3chunk-slow | 30,123,488,000 | 90,606,368,000 | 60,482,880,000 | 4 |

With the fields absent the chunked request's TTFT is the completion of the
first extend step in all twelve cells: 49.9 percent of the true value with two
chunks and 33.2 percent with three. The same absence makes the reducer count
one token per scheduled step, so the request reports 3 or 4 tokens instead of
2, and its TPOT becomes an average over intervals that generated nothing: in
`solo-3chunk-fast` the compatibility arm reports 45,381,952,000/3 ps against
the true 15,140,512,000 ps.

The control rows moved in neither arm, which is what separates "count the
right rows" from "count fewer rows": the MIXED decode companion `D` and the
prefill `S` that completes its whole prompt in one step report the same first
metric at the same picosecond in both arms, in all eight cells that carry one.

Halving the memory bandwidth doubled every TTFT to within the 32,000 ps
per-step quantization, in both arms. That relation is entailed by the per-step
exact rows and is reported as conformance, not as scored evidence.

## Fatal guards

All held; none is reported as a fraction.

- G1, both arms price every step identically: held in all twelve cells. The
  fixture is memory-bound, so the sampled count never reaches the selected
  roof.
- G2, every step memory-bound, and G3, every step at least 2 ps from a
  quantization boundary: checked before the run by `--check-only`; the
  smallest observed margins are a factor of 1.530 and 1,536 ps.
- G4, no compatibility record carries `num_sampled` or `sampled_request_ids`
  and every fixed record carries both: held in all twelve cells. The
  byte-level baseline is enforced separately by
  `tests/test_sglang_communicator.py`, against the tracked LF fixture
  `tests/fixtures/sglang/communicator_flag_identity.jsonl`, which is
  unchanged from before this branch.
- G5, no fixed record carries a partial count without its identity list: held
  by construction; the translator emits both fields or neither.
- G6, the chain raised nothing: the run completed with the reducer's
  conservation, graph-structure and completion-projection validation active on
  every step.
- G7, each cell consumed its frozen number of steps from time zero in both
  arms: held in all twelve cells.

## Evidence classes

- Exact-oracle rows, scored: 82 of 82. E1 (42 step latencies) 42 of 42, E2
  (24 request TTFT values) 24 of 24, E3 (16 control TTFT values) 16 of 16.
  The rows within one arm are correlated: they test one rule under twelve
  compositions, not twelve independent mechanisms.
- Entailed conformance findings, unscored: 106 of 106 (sampling-step tuples,
  token counts, TPOT, the TTFT error and the bandwidth ratios).
- Fatal guards: seven predicates, all held, never scored.
- Structural evidence, separate: the import-free adapter tests in
  `tests/test_adapters_sglang.py` and the byte-lock test in
  `tests/test_sglang_communicator.py`.

## What this run does not demonstrate

- No live SGLang scheduler ran. The rule for which rows consume a generated
  token is transcribed from the pinned source and exercised against stubs
  carrying its attribute names. Observed agreement with a live scheduler is a
  separate claim, and it is not made here.
- The pre-play replay token source is not on this metric chain. It has
  import-free tests and its fabricated-token off path, and PLAY-7's in-process
  live smoke is not part of this branch.
- The vLLM adapter, the shared lowerer, the sink and the reducer were used
  unchanged and were not modified by this branch.
