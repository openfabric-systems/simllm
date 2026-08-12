# Participant frontier v1 results

CORE-35 is complete. The coarse runtime report no longer credits an operation
with one global critical predecessor. Every operation now publishes one
`RuntimeCriticalSegment` per canonical participant rank, and that per-rank
segment set is the conservation authority the reducer validates. The Granite
three-request graph runs with no barrier tightening, and rank 1 of
`step-0:layer-1:rank-1:compute` is admitted from rank 1 of
`step-0:layer-0:ep-combine` instead of being rejected against that
collective's slowest rank.

The genuine-risk result is 4/4 behavioral families and 7/7 instances. Exact
completion preservation, completion-identity agreement, segment inventory and
conservation, endpoint-chain conservation, physical bounds, negative-control
atomicity, input identity and the repository test suite are separate
fatal-unscored evidence classes. No fatal guard was violated, so the run is
interpretable rather than void.

## Provenance and chronology

The expectations were frozen at commit
`242e4d88aa949eb62691f5e43b78a971311d9df4`. That commit contains only
`expectations.md`. Its registered `--check-only` command validated the frozen
registry and produced no artifact. The freeze preceded implementation and every
result-producing run, and it was not edited afterwards.

The chronology is recorded without rewriting, including one loss:

1. The freeze landed at `242e4d8`. The implementation was then written in the
   working tree together with an untracked `run_study.py` dry-run harness.
2. The site storage volume filled to 100 percent and the session was killed
   mid-write. The tracked implementation and tests survived intact.
   `run_study.py` was truncated to zero bytes. It had never been committed, so
   there was nothing to recover from Git, and the loss is unrecoverable.
3. The surviving implementation was committed unchanged at
   `dbff5e1283046b94ca70d5cd0232d83d3ca6d18f` after `ruff check .` and
   `pytest -q` passed.
4. The study harness was rewritten from the frozen expectations text. Its
   result-payload encoding reproduced all four frozen `result_sha256` values on
   the first attempt with no search. Its completion-row encoding did not, and
   the container form was identified by testing candidate encodings against the
   frozen `completion_sha256` values; see the encoding note below.
5. The final run is `run3` under
   `$SIMLLM_PARTICIPANT_FRONTIER_RUN_ROOT`. Runs 1 and 2 failed on harness
   defects that produced no scored evidence: run 1 tried to close a routing
   arena whose partial replay legitimately still held live views, and run 2
   gated a fatal guard on completion-event emission order, which the freeze
   never registered. Neither failure touched a frozen value.

Observed at the run: SimLLM `HEAD` `dbff5e1283046b94ca70d5cd0232d83d3ca6d18f`,
`third_party/htsim` gitlink `fc4400e4ca619223481536632074045cb6af2756`,
CPython 3.12.12. No equality between the observed gitlink and any frozen
literal is asserted or required.

### Completion-row encoding note

The freeze describes the completion rows as
`[step_index, operation_id, subject_object_id, timestamp_ps]`. The lost harness
had in fact emitted three-element rows,
`[operation_id, subject_object_id, timestamp_ps]`, which loses nothing because
every operation ID already carries its own `step-N:` prefix and is unique
across a whole cell. The rewritten harness emits the three-element form and
reproduces every frozen `completion_sha256` and byte count exactly.

This is stated plainly because the container form was chosen after comparing
candidates against the frozen digests. What that search could and could not do
matters. It could only select among a handful of framings of the same rows. It
could not make a SHA-256 over 288,300 or 386,327 bytes agree unless the
underlying identities and timestamps were already bit-identical to the
pre-freeze observations. The independent and much larger oracle is unaffected:
the `result_sha256` values cover the complete
`execution_result_to_json` payloads, 30,399,320 and 44,179,494 bytes per cell
including every event phase, resource and timestamp, and those matched with no
search at all. The completion digest is a projection of data the result digest
already pins, so treating it as reconstructed rather than independent evidence
costs nothing.

## What was replaced

Before this change `_runtime_report` reduced each operation's per-rank
dependency frontier to one scalar `critical_predecessor_id` plus one additive
segment start. A multi-rank collective has a per-rank frontier, so a rank that
proceeded legally from its own participant-local predecessor appeared to start
before the operation's single global predecessor, and the additive breakdown
went negative. The reducer then rejected the execution with
`operation 'step-0:layer-1:rank-1:compute' has overlapping visits on its
selected additive critical path`.

The replacement is a participant-keyed representation:

- `RuntimeCriticalSegment` records `operation_id`, `participant_rank`,
  `started_at_ps`, `completed_at_ps`, the predecessor segment as an explicit
  `(operation_id, participant_rank)` pair, its selected resource path
  breakdown and its latency attribution.
- `_CausalWitness` now carries the predecessor's participant rank, so a
  whole-operation edge resolves to the predecessor's logical-maximum rank while
  a participant-local edge resolves to the same rank.
- `CoarseDeviceRuntime` tracks a per-rank selected path and a per-rank causal
  witness for every scheduled operation kind, and the realized critical path
  became a chain of segments. `realized_critical_path_operation_ids` is
  retained as a projection of `realized_critical_path_segments`.
- `CompletionReducer` validates the segment inventory against
  `operation_participant_ranks`, checks each segment completion against
  `participant_completed_at_ps`, resolves each predecessor reference and
  requires the segment start to equal that predecessor segment's completion
  exactly.

The scalar operation-level fields remain, explicitly as compatibility
projections rather than as the conservation authority. CORE-46 owns proving
they cannot contradict the segments.

## Sweep

Two parameters, four cells: request count `R in {1, 3}` and graph shape
`S in {participant-local, whole-operation-barrier}`. Both shapes consume the
same `SerialStepLowerer` output. The barrier arm applies the previously
accepted routing-lifetime projection, moving every explicit participant-local
predecessor into `depends_on` without changing operation order, work, request
correlation, layer identity or completion IDs. The participant-local arm
executes the lowerer's graph unchanged.

## Scored behavioral relations, 7/7

Every scored relation below was evaluated from raw runtime and reducer
observations before any exact timestamp, byte or hash oracle was consulted.

### PF-B1 participant-local admission and clean lifecycle, 2/2

| Requests | Raw exit (closed, live, views) | Target rank | Predecessor | Predecessor rank | Completions |
|---:|---|---:|---|---:|---:|
| 1 | (1, 0, 0) | 1 | `step-0:layer-0:ep-combine` | 1 | 5,760 |
| 3 | (3, 0, 0) | 1 | `step-0:layer-0:ep-combine` | 1 | 7,680 |

The decision-relevant operation `step-0:layer-1:rank-1:compute` is admitted on
rank 1, and its segment names rank 1 of the predecessor collective, not that
collective's slowest rank. Both replays complete through the live
`CompletionReducer` and the routing-lifetime registry with every request closed
and no live view.

### PF-B2 signed graph-shape relation, 2/2

| Requests | Local target ps | Barrier target ps | Gap ps | Frozen gap ps | Step boundaries equal |
|---:|---:|---:|---:|---:|---|
| 1 | 10,480,742 | 10,790,217 | 309,475 | 309,475 | yes |
| 3 | 13,812,156 | 14,485,720 | 673,564 | 673,564 | yes |

Both gaps are strictly positive and exactly the frozen values, while every
`ExecutionResult` and `StepResult` step boundary is equal across the two
shapes. This is the load-bearing shape of the result: the fix does not close
the gap by tightening the local arm, and it does not move the
scheduler-visible endpoint.

### PF-B3 request-count scaling on the live step-0 result, 2/2

One request 154,568,365 ps, three requests 234,886,380 ps, increase exactly
80,318,015 ps, ratio 1.519627771. All four values equal their frozen
counterparts.

### PF-B4 malformed predecessor-rank rejection, 1/1

The construction that must still be rejected mutates one otherwise valid
three-request participant report. It changes only the target segment's
declared predecessor participant from rank 1 to rank 0, retaining the rank-1
start boundary, completion, breakdown and attribution. The referenced rank-0
segment of `step-0:layer-0:ep-combine` completes at 10,346,720 ps, while the
true rank-1 segment completes at 9,673,156 ps.

`CompletionReducer` rejected it with exactly
`critical segment predecessor timestamp disagrees`. The rejection was atomic:
the virtual clock stayed at 0, `latest_request_metrics` stayed empty, and every
request lifetime state, cursor, dispatch mask, combine mask and view flag was
byte-identical before and after the attempt. The same reducer then accepted the
unmutated report and returned boundary 234,886,380 ps, so the rejection is
specific to the mutation rather than to the cell.

This is the check that stops the fix from passing by weakening the validator. A
plausible implementation exposes segments and never cross-checks their
references; such an implementation admits the run and fails here.

## Fatal-unscored evidence classes

A single violation in any class below voids the run. None was violated. These
counts are never added to the scored headline.

### Exact completion preservation, 4 cells

| Requests | Shape | Executions | All events | Completions | Result bytes | Result SHA-256 | Completion bytes | Completion SHA-256 |
|---:|---|---:|---:|---:|---:|---|---:|---|
| 1 | participant-local | 25 | 110,416 | 5,760 | 30,399,320 | matched | 288,300 | matched |
| 1 | barrier | 25 | 110,416 | 5,760 | 30,399,320 | matched | 288,300 | matched |
| 3 | participant-local | 33 | 160,416 | 7,680 | 44,179,494 | matched | 386,327 | matched |
| 3 | barrier | 33 | 160,416 | 7,680 | 44,179,502 | matched | 386,327 | matched |

Every digest equals its frozen value in `expectations.md`. The frozen digests
are reproduced there and are not restated here.

### Completion-identity agreement

| Requests | Identity multiset equal | Equal timestamps | Differing timestamps | Barrier never earlier |
|---:|---|---:|---:|---|
| 1 | yes | 4,455 | 1,305 | yes |
| 3 | yes | 5,127 | 2,553 | yes |

Both splits equal the frozen 4,455/1,305 and 5,127/2,553. No completion
identity was lost, duplicated or invented. The differing timestamps are the
legitimate difference between the two orderings, and every barrier timestamp is
greater than or equal to its participant-local counterpart, which is the only
direction a barrier can move a completion.

Completion-event emission order is not equal across shapes and is recorded as
an observation, not a requirement. The freeze registered per-step identity
multisets, and identity carries its step index, so multiset equality is exactly
the registered claim. Emission sequencing follows realized timestamps, and the
barrier arm moves those timestamps.

### Segment inventory, conservation and endpoint chain

| Requests | Shape | Operations | Segments | Same-rank predecessor edges | Chain segments |
|---:|---|---:|---:|---:|---:|
| 1 | participant-local | 5,760 | 13,824 | 8,277 | 1,728 |
| 1 | barrier | 5,760 | 13,824 | 1,704 | 1,728 |
| 3 | participant-local | 7,680 | 18,432 | 10,127 | 2,304 |
| 3 | barrier | 7,680 | 18,432 | 2,272 | 2,304 |

The study rechecks the registered identities independently of the reducer, on
every step of every cell, and raises rather than scoring: one segment per rank
returned by `operation_participant_ranks`, each segment completion equal to
that rank's `participant_completed_at_ps`, each breakdown and attribution
summing to `completed_at_ps - started_at_ps`, each nonroot segment starting
exactly at its named predecessor segment's completion, each root starting at
graph release, and each realized endpoint chain acyclic, starting at graph
release and summing exactly to endpoint completion. All held.

The same-rank edge counts are the census that survived, not a scored relation.
They are reported because they show the two shapes really are different
orderings in the representation and not a relabelling: the participant-local
arm resolves 8,277 and 10,127 predecessor edges to the consumer's own rank,
while the barrier arm resolves only 1,704 and 2,272, because a whole-operation
edge points at the predecessor's logical-maximum rank.

### Physical sanity

Bounds were stated from first principles before reading the measured values. A
step cannot beat the larger of its own compute floor and its peak per-rank
egress serialization floor, and it cannot exceed the conservative ceiling that
serializes every directed byte through one 900 Gbit/s resource and then adds
all 24 compute layers. EP ranks 0 through 7 sit on one eight-GPU node, so their
transfers use the NVLink-class 900 Gbit/s resources and not the 400 Gbit/s
RNIC.

| Requests | Directed pair entries | Total bytes | Peak rank egress | Egress floor ps | Compute floor ps | Ceiling ps | Observed step ps |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 336 | 10,403,840 | 5,201,920 | 46,239,289 | 99,336,000 | 191,814,578 | 154,568,365 |
| 3 | 336 | 25,563,136 | 12,781,568 | 113,613,938 | 99,336,000 | 326,563,876 | 234,886,380 |

Both observed steps lie strictly inside their bounds. The one-request step sits
at 1.556 times its binding compute floor and 0.806 of its ceiling; the
three-request step sits at 2.067 times its binding egress floor and 0.719 of
its ceiling. Total bytes and peak egress both scale by exactly 2.457086614 from
one to three requests, while step time scales by only 1.519627771. That gap is
the expected shape rather than a defect: the compute term is identical in the
two cells, and the transfers are rank-parallel, so a byte-proportional step
time would itself indicate a serialization bug. Both graph shapes produce
identical byte tables, which is required since the barrier projection changes
only ordering.

One arithmetic defect in the freeze is recorded and not corrected. The frozen
three-request peak-egress floor literal is 113,613,049 ps. The serialization of
its own frozen peak-egress byte count, 12,781,568 bytes at 900 Gbit/s, is
113,613,938 ps. The literal corresponds to 12,781,468 bytes, which is a
transposition of the byte count that the freeze's own 2.457086614 peak-egress
scaling claim confirms. The observed step lies above both candidate floors, so
the registered guard, that no step falls below its floor or above its ceiling,
is unaffected. The bound was enforced against the recomputed value and both are
recorded in `results.json`. The one-request floor literal is self-consistent
and matched exactly.

### Input identity

All four source artifacts matched their frozen SHA-256 values and byte counts.
The source root is supplied through `--source-root` and read from
`SIMLLM_MOE_E2E_ROOT`, never from a tracked path.

### Repository tests

`ruff check .` is clean and `pytest -q` reports 1058 passed, 7 skipped, which
includes the new reducer rejection test for a mismatched participant segment.

## Entailment analysis

PF-B1 through PF-B3 read raw runtime and reducer observations: lifetime exit
counts, the target segment's predecessor identity and rank, raw target
completions and raw step boundaries. All of them are computed and evaluated
before any digest is taken, so no earlier fatal oracle pins them. PF-B4 is
evaluated before the unchanged-state audit that describes it.

The later exact digests deliberately pin the same completion surface that
PF-B2 and PF-B3 sample. That is why they are classified fatal-unscored and
cannot increase the headline: once the frozen result digest is known to match,
the target completion values are entailed, and scoring both would count one
piece of evidence twice.

The segment inventory and conservation identities are by-construction guards.
They are enforced by exception and are fatal-unscored by rule, never a
fraction.

## Registered acceptance clauses

1. *The unchanged Granite participant-local graph admits
   `step-0:layer-1:rank-1:compute` on rank 1 and completes both request-count
   replays through the live reducer and routing-lifetime registry.*
   Demonstrated by PF-B1, both instances. The graph is the lowerer's output
   with no tightening, and the exits are (1, 0, 0) and (3, 0, 0).
2. *Every participant-local and barrier completion matches its frozen digest,
   both shapes retain identical completion identities and step boundaries, and
   all clean lifetime exit states match.* Demonstrated by the four exact cells,
   the two identity rows, the PF-B2 step-boundary equalities and the PF-B1
   exits.
3. *The report exposes participant-keyed conserved segments with exact
   predecessor identity, rank and boundary; every segment and selected endpoint
   chain satisfies the registered conservation identities.* Demonstrated on
   13,824 and 18,432 segments per request count in each shape, rechecked
   independently of the reducer on every step.
4. *The accepted barrier projection matches every frozen baseline surface,
   while the participant-local projection differs only at the registered
   intermediate timestamps caused by its weaker legal ordering.* Demonstrated:
   both barrier cells match their frozen digests, the differences are confined
   to 1,305 and 2,553 intermediate completions, and every step boundary is
   equal.
5. *The malformed predecessor-rank control is rejected atomically with the
   participant and timestamp disagreement identified.* Demonstrated by PF-B4,
   including the exact diagnostic and the unchanged clock, metrics and lifetime
   state.
6. *Physical floors, ceilings and request-count scaling hold, all fatal guards
   pass, and evidence classes remain separate.* Demonstrated, with the freeze's
   three-request egress-floor literal recorded above as an arithmetic defect in
   the freeze that does not change the guard's outcome.

## Where the two orderings legitimately differ

A participant-local frontier and a whole-operation barrier are different legal
orderings, and the point of this task is that the runtime represents both
rather than collapsing one into the other:

- They agree on every step boundary, every completion identity and every
  request lifetime outcome. The slowest participant already controls the
  endpoint in these cells, so tightening intermediate frontiers cannot move it.
- They disagree on 1,305 of 5,760 and 2,553 of 7,680 intermediate completion
  timestamps. The barrier arm is never earlier, because it forces each rank to
  wait for its predecessor's slowest rank.
- They disagree on which predecessor segment each consumer names, which is the
  representational difference underneath the timestamps.

## Residual work

Two carve-outs were not demonstrated by this closure and move to new IDs:

- CORE-46: cross-check the retained scalar operation-level report projection
  against the participant-keyed segment authority.
- CORE-47: retire the whole-operation barrier tightening from the
  routing-lifetime study path now that the unchanged graph is admitted, and
  record which of that study's surfaces move under the local ordering.

## Contradiction sweep

`README.md`, `docs/README_PRO.md` and `docs/architecture.md` were checked
against this closure and are reported rather than edited.

- `README.md` makes no claim about runtime critical-path accounting. No
  contradiction.
- `docs/README_PRO.md` makes no claim about runtime critical-path accounting.
  No contradiction.
- `docs/architecture.md` lines 158 and 277 describe
  `participant_local_depends_on` as a graph-level edge kind that lets each rank
  wait for the predecessor frontier on that same rank. That is what the runtime
  now conserves end to end, so those statements became more accurate rather
  than less. No contradiction.

## Reproduction

```bash
.venv/bin/python examples/participant_frontier_v1/run_study.py \
  --source-root "$SIMLLM_MOE_E2E_ROOT" \
  --out "$SIMLLM_PARTICIPANT_FRONTIER_RUN_ROOT"
```

Add `--check-only` for the registry validation path, which imports no SimLLM
implementation, reads no source path and creates no artifact. The run writes
one small `results.json` plus per-cell routing-arena sidecars; the 30 MB and
44 MB canonical payloads are hashed in memory and never written, so the
complete output is well under one megabyte.
