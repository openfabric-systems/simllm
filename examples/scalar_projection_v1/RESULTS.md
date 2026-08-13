# Scalar projection v1 results

CORE-46 asked whether the scalar compatibility fields of the coarse runtime
report are derivable from the participant-keyed critical segments that own
conservation. They are, the derivation is now enforced by the completion
reducer, and the study passes **3/3 genuine-risk families over 13
parameterized instances** with every fatal guard holding.

The gap was real rather than theoretical. Before this change the validator
accepted all six of the registered single-field contradictions, including a
report that claims an additive whole-operation predecessor while its own
segments show a rank-local boundary.

## Chronology and provenance

| Provenance field | Revision or value |
|---|---|
| Expectations-only commit | `5d3e5ab8bb6b65eeb2378453f5e51ec6b858b4ac` |
| Evidence authored against | `b529f2953d1c5ac2a44f4de79fef1b0d7ee00da5` |
| SimLLM revision observed by the run | `e739dd0c2b2b9482353150d73cb30b19255c0475` |
| `summary.json` SHA-256 | `26c22103186d10b2ada2334d16a924d8ace76e89ef8c9a80e5bafe45822245dd` |
| Runtime | Python 3.12.12 on Linux x86-64 |

[The expectations](expectations.md) landed before the implementation and before
the first result-producing run, after the registered command was exercised with
`--check-only`. That path imported no SimLLM module, read no input path and
wrote nothing. The two read-only pre-freeze diagnostics are disclosed in the
freeze itself: one measured the candidate derivation over the existing test
suite, and one confirmed that the six contradictions passed the validator as it
then stood. Neither observed the Granite cells' derivation.

Two runs exist. The first was executed at the freeze commit with the
implementation still uncommitted and produced identical values; the reported
run was executed after the implementation commit so that the recorded revision
is exactly the tree that ran. No frozen literal was changed after either.

The observed revision is provenance and is not asserted equal to any submodule
pin.

## The exact derivation

For an operation record `R` with segments `S`, joined on the stable
`(operation_id, participant_rank)` key:

1. `R.physical_completed_at_ps` is the maximum segment completion.
2. `R.completed_at_ps` is the completion of one of `R`'s own segments. It may
   be earlier than the maximum, which is how an asynchronous control operation
   releases the framework before its physical work retires.
3. `R.causal_predecessor_id` and `R.causal_predecessor_completed_at_ps` are
   present or absent together, and a present predecessor is in the report.
4. The boundary is a participant completion of the named predecessor, and one
   of `R`'s own segments names that predecessor and starts exactly there.
5. `R.critical_predecessor_id` is the causal predecessor exactly when the
   boundary equals that predecessor's scheduler-visible completion, and `None`
   otherwise.
6. `R.breakdown.operation_latency_ps` spans from the additive boundary, or from
   graph release when there is none, to `R.completed_at_ps`.

`realized_critical_path_operation_ids` was already the checked operation
projection of `realized_critical_path_segments` and needed no new rule; the
study rechecks it independently in every cell.

## Physical sanity before the exact comparison

The out-of-order fixture's floors were stated before its values were read. A
1,000,000-byte transfer at the accepted 20 ps per byte cannot complete before
20,000,000 ps and a 1-byte transfer cannot complete before 20 ps; both landed
exactly on their floors because the fixture has no other contention. A
successor cannot precede the data it waits for, so `early` cannot complete
before 30 ps and `late` cannot complete before 20,000,005 ps; both landed
exactly there. The ceiling that a barrier would impose is 20,000,010 ps for
`early`, i.e. 666,667 times its realized 30 ps, which is the size of the error
a scalar projection that quietly reports a barrier would introduce.

## Registered fixture values

| Object | Registered ps | Observed ps |
|---|---:|---:|
| `collective` participant 16 | 20 | 20 |
| `collective` participants 0 and 8 | 20,000,000 | 20,000,000 |
| `collective` scheduler-visible completion | 20,000,000 | 20,000,000 |
| `early` causal boundary, participant-local | 20 | 20 |
| `early` completion | 30 | 30 |
| `late` causal boundary, whole operation | 20,000,000 | 20,000,000 |
| `late` completion | 20,000,005 | 20,000,005 |
| `barrier` causal boundary, whole operation | 20,000,000 | 20,000,000 |
| `barrier` completion | 20,000,001 | 20,000,001 |
| `ExecutionResult` boundary | 20,000,005 | 20,000,005 |

All three successors name `collective` as their causal predecessor. Only
`early` carries no additive predecessor, and only the boundary comparison
separates them, which is the discrimination clause 5 exists for. `early`'s
scalar breakdown spans 30 ps while its segment spans 10 ps: a participant-local
boundary is not additive, so the scalar charges the whole interval from graph
release as external dependency. The realized chain was
`("collective", "late")` over segments `(("collective", 0), ("late", 0))`.

## Scored behavioral evidence

| Family | Instances | Result | Raw relation |
|---|---:|---|---|
| SP-B1 derivation on live fixtures | 5 | 5/5 pass | Every clause held for every operation record of every step, recomputed in the harness before the reducer saw the report. |
| SP-B2 contradiction rejection | 6 | 6/6 pass | Each single-field contradiction was rejected, the clock stayed at 0 and no request metric was committed. |
| SP-B3 shape discrimination | 2 | 2/2 pass | Participant-local cells carry 1,305 and 2,553 rank-local frontier records; their barrier twins carry exactly zero. |

The scored headline is **3/3 families over 13 instances**.

### Derivation coverage

| Cell | Operation records | With a causal predecessor | Additive | Rank-local frontier |
|---|---:|---:|---:|---:|
| 1 request, participant-local | 5,760 | 5,568 | 4,263 | 1,305 |
| 1 request, barrier | 5,760 | 5,568 | 5,568 | 0 |
| 3 requests, participant-local | 7,680 | 7,424 | 4,871 | 2,553 |
| 3 requests, barrier | 7,680 | 7,424 | 7,424 | 0 |
| out-of-order fixture | 4 | 3 | 2 | 1 |

The 1,305 and 2,553 rank-local frontier records are exactly the counts of
intermediate timestamps that CORE-35 reported as differing between the two
graph shapes. That agreement was not registered in advance and is recorded as
an observation: the records whose scalar projection is non-additive are the
same records whose realized timing the barrier moves, which is the expected
correspondence if the projection is reading the same frontier the runtime
scheduled.

### Contradiction table

| ID | Mutation | Accepted before | Rejection after |
|---|---|---|---|
| M1 | `early.critical_predecessor_id = "collective"` | yes | additive predecessor disagrees with the critical segment that admitted it |
| M2 | `late.critical_predecessor_id = None` | yes | additive predecessor disagrees with the critical segment that admitted it |
| M3 | `early.causal_predecessor_completed_at_ps = 17` | yes | causal boundary is not a participant completion of `collective` |
| M4 | `late.causal_predecessor_id = "barrier"` | yes | causal boundary is not a participant completion of `barrier` |
| M5 | `collective.physical_completed_at_ps + 7` | yes | physical completion disagrees with its critical segments |
| M6 | `early` breakdown and attribution shortened by 5 ps | yes | critical-path breakdown does not span its own projected segment |

M1 is the contradiction CORE-46 names: a report whose scalar predecessor
contradicts its segments. Each rejection left the virtual clock at 0 with no
committed request metric, and the same reducer then accepted the unmutated
report and returned the registered 20,000,005 ps boundary, so the rejections
are specific to the mutation rather than to the fixture.

### Entailment check

SP-B1 and SP-B3 are recomputed from the raw `RuntimeReport` before
`CompletionReducer.reduce` is called on that same report, so the validator
under test cannot pin them. SP-B2 reads only rejection outcomes. Every digest,
preservation and exact-timestamp oracle is evaluated after all three families.
The pre-freeze diagnostic over the existing test suite reduces the novelty of
the derivation on the synthetic shapes that suite already covers, and that is
disclosed in the freeze; the Granite cells and the out-of-order fixture were
not observed before the freeze.

## Fatal and unscored evidence

All guards held. A violation in any of them would have voided the run rather
than costing a point, and none is reported as a fraction.

- All four source artifacts matched their accepted SHA-256 values and byte
  counts.
- Every Granite cell reproduced its accepted `result_bytes`, `result_sha256`,
  `completion_bytes`, `completion_sha256`, execution count and completion
  count: 25 executions and 5,760 completions per one-request cell, 33 and 7,680
  per three-request cell. Adding the projection check changed no accepted
  timestamp, digest or completion identity.
- The out-of-order fixture reproduced every registered value, its additive map
  and its realized chain.
- No completion identity was duplicated in any cell.
- The reducer accepted every unmutated report of every cell.

The preservation evidence is strong precisely because the cells are built by
reusing the accepted `participant_frontier_v1` construction rather than a
private copy: a copy could only have shown that it agrees with itself.

## CORE-46 closure map

| Registered clause | Evidence and disposition |
|---|---|
| "check the retained scalar operation-level report projection against the participant-keyed segment authority" | Six clauses derived and enforced in `CompletionReducer`; SP-B1 passed 5/5 on live fixtures. |
| "left `critical_predecessor_id`, the operation-level breakdown and attribution, and `realized_critical_path_operation_ids` as unjoined compatibility fields" | The first two are now joined by clauses 4 through 6; the third was already the checked projection of the realized segment chain and is rechecked independently in every cell. |
| "joined by stable identity and checked for loss, duplication and timestamp disagreement" | The join key is `(operation_id, participant_rank)`. Loss and duplication were already covered by the CORE-35 inventory checks; timestamp disagreement is what clauses 1, 2, 4 and 6 add. |
| "Identify the exact derivation" | Stated above and in the freeze, with the producer locations it was read from. |
| "then require it on the Granite participant-local and barrier cells" | All four accepted cells, 26,880 operation records in total, zero derivation errors. |
| "plus a fixture whose collective ranks finish out of order" | The registered fixture completes its highest-numbered participant 999,980 ps before the other two and drives an additive and a participant-local successor from the same causal predecessor. |
| "Acceptance must reject a hand-built report whose scalar predecessor contradicts its segments" | M1 and M2 rejected, with the unmutated report accepted by the same reducer; four further contradictions rejected. |
| "must preserve every accepted timestamp, digest and completion identity exactly" | All four accepted result and completion digests, byte counts, execution counts and completion counts reproduced. |

CORE-46 closes. Every registered clause is demonstrated, so no new task ID was
required.

## Observations that are not registered clauses

These belong in the record but claim nothing and open no task.

- The rank-local frontier count equals CORE-35's differing-timestamp count in
  both request counts, as noted above.
- The asynchronous control shape is the only one in the repository where the
  scheduler-visible completion is strictly earlier than the physical maximum.
  It appears once in the test suite and in none of the Granite cells, so clause
  2 is exercised by a unit fixture rather than by this study's cells.
- Clause 4 requires the operation's own segment table to corroborate the causal
  boundary. That held on all 26,884 records seen here, and it is the clause a
  future producer change is most likely to strain, because the scalar witness
  and the per-participant witnesses are selected by two separate reductions in
  `simllm/core/runtime.py`.

## Contradiction sweep

The required sweep after closure found no statement in `README.md`,
`docs/README_PRO.md` or `docs/architecture.md` that contradicts this result.
The one-authority paragraph in `docs/architecture.md` already requires backend
rows and diagnostic records to be read-only projections joined by stable
identity and checked for loss, duplication and timestamp disagreement, which is
what this change implements for the coarse report's scalar fields.
