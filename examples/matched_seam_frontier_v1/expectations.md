# Matched-seam frontier expectations

These expectations freeze the maintainer-directed overlay: the external
planning tool's Pareto curve drawn in its own published visual grammar,
with our curve added as a second series. They are committed before any
implementation of the binding, the scan or the figure exists.

The premise this wave tests is the one the imported database made
possible. Both arms now price from the same measured operation database,
bit-identically at the pass seam (parity study `external_db_parity_v1`,
merged at main 8add51c). Any frontier difference that survives is
therefore composition mechanism, not kernel timing. That is the claim,
and these families are written so it can fail.

## Frozen external givens

From the tracked, hash-pinned tables in
`examples/frontier_comparison_v1/external/` (aiconfigurator 0.11.0,
h200_sxm, trtllm 1.3.0rc10, Qwen3-32B-FP8, ISL 4000, OSL 500, prefix
500):

- Their published axis identities, verified before design: `tokens/s/user`
  equals `1000 / tpot`, and `tokens/s/gpu` equals output `tokens/s`
  divided by `num_total_gpus`.
- The reference disagg row: concurrency 192, prefill tp4 with 5 workers,
  decode tp4 with batch 64 and 3 workers, 32 GPUs total, ttft 196.423 ms,
  tpot 9.179 ms, 602.586 tokens/s/GPU, 108.944 tokens/s/user.
- Their disagg ttft is 196.423 ms on all ten disagg Pareto rows,
  independent of concurrency.
- Ten disagg Pareto rows and twenty-five agg Pareto rows are the external
  curve; they are display and comparison data, never our result.

## Fatal guards

- FG-1 shared timing base: every duration our scored arm consumes comes
  from the imported database through the external pass model. No roofline
  term, no declared efficiency and no fitted constant appears anywhere in
  the scored arm. A single roofline-sourced term voids the run.
- FG-2 evidence class: every consumed duration carries MEASURED-EXTERNAL
  with the frozen slice identity, end to end into the frontier record.
  Nothing is served or recorded as MEASURED.
- FG-3 external immutability: the tracked external tables and the parity
  study's artifacts are byte-identical before and after this run.
- FG-4 chronology: this expectations commit precedes the binding, the
  scan, the figure and the first comparison run.
- FG-5 no naive TTFT match: the run refuses to score any family that
  equates their `ttft` with an isolated prefill service duration. Their
  ttft enters only the decomposition of Family D, never a matched-point
  claim. This guard encodes the DEPLOY-12 finding.
- FG-6 determinism: every scored quantity is reproduced in a second fresh
  process, bit-equal.

## Family S: seam identity carried into the frontier (exact, scored)

The binding must deliver the database's own numbers unaltered. For a
frozen list of at least six decode configurations and three prefill
configurations spanning the tensor-parallel widths their Pareto rows use,
the duration the frontier consumes is bit-equal to the external pass
model's value for the same configuration, compared as IEEE-754 hex. The
expected side is computed from the pinned external sdk and frozen in the
study configuration in a commit that precedes the binding, so no scored
cell can be satisfied by the binding reproducing its own output.

## Family R: reproducing their published decode step (scored)

This is the sharp test of the matched seam at serving level. Their decode
step is a composition over the same database we now hold. Reproducing it
from their declared configuration is therefore predicted, not hoped for.

For each of the ten disagg Pareto rows, our composition over the imported
database at that row's declared decode configuration (its tensor-parallel
width, batch size, sequence length and prefix) yields a decode step whose
quotient against their published `tpot` lies in **[0.98, 1.02]**, with the
ideal network seam and no queueing term. The prediction is that these
land near 1.000 because both sides are the same arithmetic over the same
measured rows.

A row outside the band is published as a refutation of this expectation
with its residual decomposed into named terms, and the band is not
widened. Scoring is per row, ten rows, never summed with other families.

## Family F: the frontier overlay (scored)

Our frontier is scanned over the candidate grid their sweep covers, using
only imported timings, and reduced by the same weak-dominance Pareto rule
already used by `simllm/deploy`. Scored:

- F1 axis identity: our published curve uses their axis definitions
  exactly (`tokens/s/user` as `1000 / step_ms`, `tokens/s/gpu` as output
  tokens per second divided by GPUs). One frozen worked cell demonstrates
  each axis from its components.
- F2 bracket: at each of their ten disagg Pareto `tokens/s/user` values,
  our curve's `tokens/s/gpu` lies within a factor of **[0.75, 1.35]** of
  theirs. The asymmetry is declared in advance and is the expected sign:
  our composition prices terms theirs omits, so our curve should sit at or
  below theirs more often than above.
- F3 monotonicity: our Pareto curve is non-increasing in `tokens/s/gpu`
  as `tokens/s/user` rises, and contains at least eight distinct points.
- F4 no endpoint degeneracy: no single point of our curve answers more
  than three of their ten rows. This encodes the X3b finding from
  `frontier_comparison_v1`.

## Family M: mechanism isolation (scored, our precision claim)

On the identical timing base, toggling only our network seam moves the
frontier by an amount their model class cannot express, because it prices
no contention at all.

- M1: the ideal and packet rungs are run over the same candidates with
  every other input identical, and the per-point step quotients are
  published. The frozen expectation is that the packet rung is never
  faster: every quotient is greater than or equal to 1.000000.
- M2: at least one candidate shows a packet-to-ideal step quotient of at
  least 1.02, demonstrating that the seam is not inert at this workload.
  If no candidate reaches it, that is published as a refutation with the
  measured maximum stated, and the claim that the mechanism matters at
  this workload is withdrawn for this workload rather than restated.

## Family D: their TTFT decomposed (unscored, published)

Their disagg ttft is constant at 196.423 ms across concurrency, so it is
not queueing-loaded in their disagg model. This family publishes its
decomposition against the imported database: the prefill pass at their
declared prefill configuration, and the residual between that pass and
196.423 ms, attributed to named terms or left explicitly unattributed.
Nothing here is scored, and no matched-point claim is made. This is the
work that retires DEPLOY-12 by replacing the conflated premise with a
decomposition.

## Family W: wall time (scored, generous)

The complete scan, both seam rungs, all scored families and the figure
complete in at most 600 s in one process, machine disclosed.

## The figure

One publication-grade figure in their published visual grammar: log-log
axes, `tokens/s/user` on x, `tokens/s/gpu` on y, the better direction
annotated up and to the right. Series: their agg Pareto curve, their
disagg Pareto curve, and our curve as its own legend entry, plus our
packet-rung curve where Family M shows separation. The caption states the
shared timing base in one sentence, names the evidence class of every
series, and states plainly that curve differences are composition
mechanism because the timing base is identical. Every plotted number is
traceable to the record.

## Closure

A full pass demonstrates that at the matched seam our serving-level
numbers reproduce theirs, and that the residual frontier differences are
named mechanisms we can toggle. It does not validate either stack against
hardware, does not import another system or version, and closes no
calibration task. Scored families are S, R, F, M and W, in their classes,
never summed with each other or with fatal rows.
