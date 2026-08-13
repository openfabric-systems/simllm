# Scalar projection v1 expectations

This expectations-only record freezes CORE-46 before its implementation and
before its study produces a result. CORE-35 made `RuntimeCriticalSegment` the
conservation authority of the coarse runtime report but left
`critical_predecessor_id`, the operation-level breakdown and attribution, and
the realized critical-path projection as unjoined compatibility fields. The
one-authority rule requires a read-only projection to be joined by stable
identity and checked for loss, duplication and timestamp disagreement. Nothing
asserts today that the scalar fields are derivable from the segments.

## Pre-freeze source audit and disclosed diagnostics

The evidence is authored against SimLLM revision
`b529f2953d1c5ac2a44f4de79fef1b0d7ee00da5`. The relevant producer locations at
that revision are:

- `simllm/core/runtime.py:1365-1406` builds one `_CausalWitness` per incoming
  edge, carrying the predecessor operation, the predecessor participant rank
  and that participant's completion, which for a whole-operation edge is the
  predecessor's logical completion;
- `simllm/core/runtime.py:2786-2846` reduces those witnesses to the scalar
  `causal_predecessor_id`, `causal_predecessor_completed_at_ps` and
  `critical_predecessor_id`, where the last is present only when the causal
  boundary equals the predecessor's whole-operation logical completion;
- `simllm/core/runtime.py:2652-2716` computes the operation-level breakdown
  over `segment_start_ps` and the operation's logical completion;
- `simllm/core/completion.py:246-296` already joins the segments to each other
  and to `realized_critical_path_segments`, and already checks that
  `realized_critical_path_operation_ids` is exactly the operation projection of
  that segment chain, but treats the remaining scalar fields as opaque.

Two read-only diagnostics ran before this freeze and are disclosed here rather
than presented as results:

1. A probe wrapped the reducer's validation and measured the candidate
   derivation on every report the existing test suite reduces: 474 reports and
   877 operation records, of which 230 carry a causal predecessor, 228 an
   additive one and 1 an asynchronous scheduler boundary earlier than its
   segment maximum. Every clause below held on all of them. This is why the
   clauses are stated as exact laws rather than as bounds, and it means the
   synthetic shapes already covered by the test suite are not novel evidence.
2. A probe built the out-of-order fixture below and submitted six hand-built
   scalar contradictions to the current reducer. All six were accepted, which
   is the concrete form of the gap this task closes.

Neither diagnostic ran the Granite cells' derivation, which stays unobserved
until the registered run. One further diagnostic confirmed that reusing the
accepted `participant_frontier_v1` cell construction reproduces its published
`result_sha256` for the one-request participant-local cell; that preservation
value is an already accepted public number and is registered below as a fatal
guard, never as scored evidence.

## The exact derivation

For every operation record `R` in a `RuntimeReport`, with `S` the segments of
`R`, `graph.released_at_ps` the release, and `by_id` the report's own operation
index, the following are required. The join key is the stable pair
`(operation_id, participant_rank)`.

1. **Physical completion is the segment maximum.**
   `R.physical_completed_at_ps == max(s.completed_at_ps for s in S)`.
2. **The scheduler-visible completion is a real participant completion.**
   `R.completed_at_ps` is the completion of one of `R`'s own segments. It may
   be earlier than the maximum, which is how an asynchronous control operation
   releases the framework before its physical work retires; it may never be a
   timestamp the segment authority does not carry.
3. **The causal boundary is present or absent as a pair.**
   `R.causal_predecessor_id is None` exactly when
   `R.causal_predecessor_completed_at_ps is None`, and a present predecessor
   names an operation the report carries.
4. **The causal boundary is a participant completion of the named
   predecessor,** and `R`'s own segments corroborate it: some segment of the
   named predecessor completes exactly at the boundary, and some segment of `R`
   names that predecessor and starts exactly at the boundary.
5. **The additive predecessor is exactly the whole-operation case.**
   `R.critical_predecessor_id` equals `R.causal_predecessor_id` when the
   boundary equals that predecessor's scheduler-visible completion, and is
   `None` otherwise, including when there is no causal predecessor.
6. **The scalar breakdown spans exactly its own segment.**
   `R.breakdown.operation_latency_ps == R.completed_at_ps - start`, where
   `start` is the causal boundary when the additive predecessor is present and
   the graph release otherwise. The attribution total already has to equal the
   breakdown, so this pins both.

Clauses 1 through 6 are read-only. They may not change a timestamp, a digest,
a completion identity, a request metric or a random draw.

## Registered fixtures

**Granite cells.** The four cells accepted by
[participant_frontier_v1](../participant_frontier_v1/RESULTS.md): one and three
requests, each in the participant-local and the whole-operation barrier shape.
They are built by reusing that study's cell construction rather than by a
reimplementation, because the preservation claim is about the accepted
construction and a private copy could only prove that it agrees with itself.
The inputs are addressed through `SIMLLM_MOE_E2E_ROOT` and are required to
match their accepted SHA-256 values:

| Input | SHA-256 |
|---|---|
| `capture/granite-greedy.jsonl` | `5f55fccc265ee5519430a0f73d6631e49aa547ec07fcb81034e9fc2b4d9fead6` |
| `replay-400g/run.json` | `b4d38a09011caf6de159c22133264d62a2727063496953f4337b17d79cfde93e` |
| `replay-400g/steps.jsonl` | `824cd9557293328bb42b593ac893b6a067302e545b087c9219195ccb8031d755` |
| `replay-400g/routed-experts.json` | `24e986e989e21f1bfe7e758d4470928c82c3bbaec06072a839743b9b17d7cf5f` |

**Out-of-order collective fixture.** One graph over semantic ranks 0, 8 and 16
whose pairwise `all-to-allv` carries `(8 -> 0, 1,000,000 bytes)` and
`(16 -> 8, 1 byte)`, so the highest-numbered participant finishes first and the
collective's ranks do not finish in rank order. Three successors read that
frontier: `early` on rank 16 and `late` on rank 0 through participant-local
edges, and `barrier` on rank 8 through a whole-operation edge. Their nominal
compute durations are 10, 5 and 1 ps.

The accepted coarse profile serializes at 20 ps per byte with no fixed
overhead, which is the arithmetic the published
`tests/test_dependency_authority_core.py` fixture pins with its 1-byte and
1,000,000-byte transfers. The registered table follows from it:

| Object | Value ps |
|---|---:|
| `collective` participant 16 | 20 |
| `collective` participants 0 and 8 | 20,000,000 |
| `collective` scheduler-visible completion | 20,000,000 |
| `early` causal boundary, participant-local | 20 |
| `early` completion | 30 |
| `late` causal boundary, whole operation | 20,000,000 |
| `late` completion | 20,000,005 |
| `barrier` causal boundary, whole operation | 20,000,000 |
| `barrier` completion | 20,000,001 |
| `ExecutionResult` boundary | 20,000,005 |

`early` must carry no additive predecessor while `late` and `barrier` must
carry `collective`, which is the discrimination clause 5 exists for: all three
successors name the same causal predecessor and only the boundary comparison
separates them. `early`'s scalar breakdown must span 30 ps while its segment
spans 10 ps, because a participant-local boundary is not additive.

## Scored behavioral families

All scored relations are evaluated on the raw `RuntimeReport` before the
reducer is invoked on that report, so the validator being added cannot entail
any of them.

### SP-B1: the derivation holds on live fixtures, 5 instances

For each of the four Granite cells and the out-of-order fixture, every clause
holds for every operation record of every step, measured by an independent
recomputation in the harness before `CompletionReducer.reduce` sees the report.
One instance per fixture.

### SP-B2: the projection check rejects a contradiction, 6 instances

Six hand-built reports, each a single-field mutation of the accepted
out-of-order report, must be rejected by the reducer, while the unmutated
report of the same cell is accepted and returns the registered 20,000,005 ps
boundary. All six are accepted by the validator as it stands today, which is
recorded above.

| ID | Mutation | Clause it must violate |
|---|---|---|
| M1 | `early.critical_predecessor_id = "collective"` | 5 |
| M2 | `late.critical_predecessor_id = None` | 5 |
| M3 | `early.causal_predecessor_completed_at_ps = 17` | 4 |
| M4 | `late.causal_predecessor_id = "barrier"` | 4 |
| M5 | `collective.physical_completed_at_ps + 7` | 1 |
| M6 | `early` breakdown and attribution both shortened by 5 ps | 6 |

A rejection must leave the reducer's state untouched: the virtual clock, the
latest request metrics and every request lifetime field must be identical
before and after the refused attempt.

### SP-B3: the shapes are distinguished, 2 instances

A participant-local Granite cell must contain a strictly positive count of
operation records whose causal predecessor is present and whose additive
predecessor is absent, i.e. records admitted from a rank-local frontier. The
matching barrier cell must contain exactly zero such records, because every
edge in that shape is a whole-operation edge. One instance per request count.
If a participant-local cell reported zero, the derivation would be untested
exactly where the two authorities can disagree.

The scored headline is 3 families over 13 instances. Nothing else is added to
it.

## Fatal and unscored evidence

A violation in any class below voids the run. None of these counts enters the
behavioral denominator.

- The four source artifacts match their accepted SHA-256 values.
- Every Granite cell reproduces the accepted `result_bytes`, `result_sha256`,
  `completion_bytes`, `completion_sha256`, execution count, event count and
  completion count frozen in
  [participant_frontier_v1/expectations.md](../participant_frontier_v1/expectations.md).
  This is the preservation claim: adding a validator must change no accepted
  timestamp, digest or completion identity.
- The out-of-order fixture reproduces every value in the registered table.
- The reducer accepts every unmutated report of every cell.
- The realized critical-path operation projection stays exactly the operation
  projection of the realized segment chain in every cell.
- No completion identity is lost, duplicated or invented in any cell.

## Physical sanity before the exact comparison

This task changes no modeled duration, so the sanity check is that the
timestamps it reads still obey their own physics.

- Floor for the out-of-order collective: 1,000,000 bytes at 20 ps per byte
  cannot complete before 20,000,000 ps, and 1 byte cannot complete before
  20 ps. Both registered values sit exactly on their floors because the
  fixture has no other contention.
- A successor cannot start before the data it waits for arrives: `early`
  cannot complete before 20 + 10 ps and `late` cannot complete before
  20,000,000 + 5 ps. Both registered completions sit exactly on those floors.
- Ceiling for the whole graph: if every participant had waited for the slowest
  rank, `early` would complete at 20,000,010 ps instead of 30 ps. The
  registered value is 666,667 times smaller, which is the entire point of the
  participant-local frontier and the reason the scalar projection must not
  quietly report a barrier.
- The Granite cells keep their accepted 4,139,000 ps per-layer compute, so
  every registered target completion stays exactly one layer above its
  boundary, and the barrier shape is never earlier than the participant-local
  shape.

## Registered acceptance clauses

1. The exact derivation above is identified, implemented in the reducer, and
   holds on the four Granite cells and the out-of-order fixture.
2. All six registered contradictions are rejected, each unmutated report is
   accepted, and no refused attempt mutates reducer or lifetime state.
3. Every accepted timestamp, digest and completion identity is preserved
   exactly.
4. The participant-local and barrier shapes are distinguished by the registered
   frontier counts.
5. Any registered clause that the run does not demonstrate moves to a new task
   ID from the range allocated to this branch.

## Registered command and check-only dry run

```bash
.venv/bin/python examples/scalar_projection_v1/run_study.py \
  --source-root "$SIMLLM_MOE_E2E_ROOT" \
  --out "$SIMLLM_SCALAR_PROJECTION_RUN_ROOT"
```

Before this expectations-only commit the same command is run with
`--check-only`. That path parses both paths and validates only the frozen
literal shapes and arithmetic. It imports no SimLLM module, reads neither input
path, invokes no native binary and creates no output artifact.

The result report records the SimLLM revision the run observes as provenance,
separately from the revision this evidence was authored against, and asserts no
equality between either of them and a live submodule pin.
