# Routing-lifetime rerun on the unchanged graph, expectations

This is the expectations-only record for CORE-47. It freezes what the rerun
must reproduce, what is allowed to move, and the causal rule that must account
for every moved value. Nothing here is an implementation, and no value below
was measured by this task.

## Why the rerun exists

`examples/routing_lifetime_v1/run_study.py` executes
`_runtime_report_compatible_graph`, which promotes every participant-local
dependency edge the lowerer emits into a whole-operation barrier. It was added
for one reason, recorded in its own docstring: the coarse runtime's additive
operation report rejected a rank that started legally from its own
participant-local predecessor, with
`operation 'step-0:layer-1:rank-1:compute' has overlapping visits on its
selected additive critical path`.

CORE-35 removed that limitation. `RuntimeCriticalSegment` is now keyed by
participant, and the unchanged graph is admitted. The routing-lifetime study's
accepted evidence was therefore produced under a stricter ordering than the
lowerer actually emits, and the barrier must become an explicit comparator
rather than the executed path.

## What changes and what does not

The executed path becomes `lowerer.lower(record)` with no projection. The
barrier transformation is retained, unchanged in body, and applied only to a
second comparator arm run beside the executed one. Every registry, every input
digest, every memory cell and every suppression target keeps its accepted
definition. The study gains one surface it did not previously persist: the
per-execution completion-timestamp inventory, without which "which values
moved" cannot be stated.

## Independently pinned expectations

CORE-35's closure measured both graph shapes on cells built from the same
capture, the same `SerialStepLowerer` configuration
(`tp_ranks=(0,)`, `ep_ranks=0..7`, fixed provider at 24 * 4,139,000 ps), the
same 25-record one-request selection and the same 32-record plus drain
three-request selection. Its published values are the oracle this rerun is
registered against, and they were produced by a different study before CORE-47
existed:

| Cell | Executions | Completion events | Final step boundary ps |
|---|---:|---:|---:|
| One request (`r0`) | 25 | 5,760 | 154,568,365 |
| Three requests | 33 | 7,680 | 234,886,380 |

| Cell | Executed (participant-local) target ps | Barrier target ps | Gap ps |
|---|---:|---:|---:|
| One request | 10,480,742 | 10,790,217 | 309,475 |
| Three requests | 13,812,156 | 14,485,720 | 673,564 |

The target is `step-0:layer-1:rank-1:compute` at participant rank 1, the exact
operation whose admission the barrier existed to avoid.

## Scored behavioral relations

Every relation is evaluated from raw registry and runtime observations before
any digest, audit or conservation guard runs. The fatal guards below constrain
input identity and arena legality; none of them constrains a completion
timestamp or an exit count.

### LIFE-C1, lifecycle exits retained on the executed graph, 2 instances, originally scored

Raw `(closed, live, views)` at exit is `(1, 0, 0)` for the one-request cell and
`(3, 0, 0)` for the three-request cell, read before `audit_closed()`. This is
the first clause of the acceptance: retiring the barrier must not cost a
lifecycle exit.

### LIFE-C2, suppression diagnostics retained on the executed graph, 2 instances, originally scored

With the `r0` dispatch layer-7 end flag suppressed, and separately the `r2`
combine layer-19 end flag, the executed graph must still produce exactly one
raw subjectless final-token completion, leave the record in `FINISH_FLAGGED`
with its view live, raise from `audit_closed()` with a diagnostic naming the
request, the phase and the missing model layer, and raise `BufferError` from
`arena.close()`.

### LIFE-C3, scheduler-visible boundaries unchanged, 2 instances

The full per-step boundary vector of the executed arm equals the barrier arm's
vector element by element in both cells, and the final boundary is
154,568,365 ps and 234,886,380 ps respectively. A scheduler-visible boundary is
the step's `execution.completed_at_ps` and the reduced
`StepResult.completed_at_ps`. Both must agree across arms for every execution,
not only at the end.

### LIFE-C4, moved intermediate values, 4 instances

Registered exactly, from CORE-35's independent measurement:

- One request: exactly 1,305 of 5,760 completion timestamps differ between the
  two arms, and 4,455 agree.
- Three requests: exactly 2,553 of 7,680 differ, and 5,127 agree.
- One request: the decision target completes at 10,480,742 ps executed and
  10,790,217 ps under the barrier, a gap of 309,475 ps.
- Three requests: 13,812,156 ps and 14,485,720 ps, a gap of 673,564 ps.

A moved intermediate value is expected here and is not a regression. The
registered direction is that the barrier is never earlier.

### LIFE-C5, causal attribution of every moved value, 2 instances

One instance per cell. Every single moved completion timestamp must satisfy all
of:

1. the barrier value is strictly greater than the executed value, so no moved
   value is earlier under the barrier;
2. the moved operation carries at least one participant-local dependency edge
   in the lowered graph, so an operation with no local frontier cannot move;
3. that operation's predecessor has at least one participant whose completion
   is strictly earlier than the predecessor's whole-operation completion, which
   is the only thing the barrier removes.

The count of moved values lacking any of the three must be zero, and the count
of unmoved values that are earlier under the barrier must be zero. This is what
"state each moved intermediate value with its cause" means operationally: the
per-value inventory is written to the run artifact, and the rule above is
checked against every entry rather than asserted over a sample.

## Fatal-unscored guards

1. **Input identity.** The five recorded source artifacts keep their frozen
   sizes and digests, and the capture keeps its 120 lines.
2. **Completion identity multiset.** The two arms produce the same execution
   count, the same total event count, and the same
   `(step_index, operation_id, subject_object_id)` completion identities in the
   same order. Only timestamps may differ.
3. **Arena and lifecycle rejections.** Every accepted rejection of the original
   study still fires: malformed, truncated, overlapping and wider-than-uint8
   arenas, premature view release, cursor overflow, skipped transitions and any
   nonclosed end-of-run record.
4. **Suppression view retention.** `arena.close()` raises while a view is live.
5. **Uint8 layout.** Payload length is `tokens * 24 * 8` in both cells.
6. **Traffic identity.** The 32-step compatibility-versus-arena traffic sweep
   remains byte-identical. It never applied the barrier and cannot move.
7. **Executed arm is the unchanged graph.** Every executed graph carries at
   least one participant-local dependency edge, and the comparator arm carries
   none. A rerun that silently kept the barrier on the executed path fails
   here.

## Physical sanity

- Floor: no completion can precede the data it depends on, so removing a
  barrier can only make an intermediate completion earlier or leave it equal.
  A moved value that is later without the barrier is proof of a defect.
- Ceiling: the step boundary is set by the slowest participant, which the
  barrier cannot change, so no step boundary may move at all. If one moves, the
  ordering change escaped the intermediate frontier and reached the endpoint.
- Scale check: the three-request cell must move strictly more values than the
  one-request cell, because it carries more concurrent participants over more
  steps, and 2,553 out of 7,680 is a larger fraction than 1,305 out of 5,760.

## Registered acceptance clauses for CORE-47

1. The study is rerun on the unchanged graph, and the barrier arm is retained
   as an explicit comparator rather than as the executed path.
2. Every lifecycle exit, suppression diagnostic and scheduler-visible boundary
   is retained.
3. Each moved intermediate value is stated with its cause.

## Production commands

`SIMLLM_WAVE10_RUN_ROOT` names this branch's external run root and
`SIMLLM_MOE_E2E_ROOT` the recorded capture tree.

```bash
.venv/bin/python examples/routing_lifetime_v1/run_study.py \
  --out "$SIMLLM_WAVE10_RUN_ROOT/routing_lifetime_v1-rerun" \
  --source-root "$SIMLLM_MOE_E2E_ROOT"
```

The dry run registered with this freeze is the same command with
`--check-only`, which validates the frozen registry including the comparator
literals above, imports nothing under study, reads neither path and writes no
artifact.

## Amendment after a void run

The first execution of this rerun, retained at `routing_lifetime-dev1`, is
**void**: its `completion_identity_multiset` fatal guard was violated. Under
the validation discipline a violated fatal guard voids the run rather than
costing a point, so nothing was closed on it and its evidence is kept. Two
defects in the section above caused it, both of them defects in this freeze
rather than properties of the system, and both demonstrable without reading a
result.

**1. The cross-arm ordering clause contradicts this freeze's own expectation.**
Fatal guard 2 required the two arms to emit their completion identities "in the
same order". LIFE-C4 simultaneously registers that 1,305 and 2,553 completion
timestamps move. A completion stream ordered by time cannot both carry moved
timestamps and preserve its emission order, so the two registered statements
cannot both hold, and the ordering clause is the one that is wrong: reordering
is the observable consequence of the movement this task exists to record, not a
precondition for it. The corrected guard requires the identity multiset to be
equal, duplicate free, and of equal length in both arms, and requires each arm
to be internally deterministic. It says nothing about the cross-arm order.

**2. The registered boundary literals name the wrong surface.** CORE-35's
`step_completed_at_ps` field is written inside its decision-step branch, so
154,568,365 ps and 234,886,380 ps are the boundaries of **step 0**, not of the
last step. LIFE-C3 registered them as the final boundary. The corrected
registration keeps both literals and attaches them to the step-0 boundary,
which is the surface CORE-35 actually published. The substantive half of
LIFE-C3, that the full per-step boundary vector is identical between the two
arms, is unchanged.

Nothing else is amended. The LIFE-C4 and LIFE-C5 literals, their direction and
their causal rule stay exactly as registered before the void run.

## Post-run evidence-accounting correction

Integrator review after the accepted run found that LIFE-C1 and LIFE-C2 are
duplicate scoring surfaces. LIFE-C1 projects the same `clean_one` and
`clean_three` objects and pass predicate already scored by LIFE-B1. LIFE-C2
reuses the exact `suppression_rows` list and pass predicate already scored by
LIFE-B2. The runner retains both comparator views as unscored duplicate
records, but they add no family or instance.

The final scored population is therefore the six unique families MEM-B1,
LIFE-B1, LIFE-B2, LIFE-C3, LIFE-C4 and LIFE-C5, with 14 instances. The earlier
five-family comparator registry above is retained as chronology, not as the
final evidence classification.
