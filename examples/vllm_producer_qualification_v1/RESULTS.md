# VLLM observation producer qualification results

VLLM-22 is complete. The closure run was VALID: all three pre-registered
genuine-risk instances passed and no fatal guard was violated. A real
eight-rank vLLM v0.26.0 replay emitted observations for all 32 nonempty
steps, those observations reached original-request TTFT and TPOT through the
supported metric chain, and a second real replay proved the producer-off
legacy path byte-identical to an independent serial reference.

The expectations document records repository source boundary
`6b7200b300f10f7b46a16921487c9b58ca32c987`. It was then committed without
implementation or results in the expectations-only commit
`6459c3c469bb4bbddb47fbfdd1f6ed5ba8b86949` before implementation and before
the first measured run. The registered check-only command passed before that
commit, imported no SimLLM target module and wrote no artifact. The closure
run observed repository commit
`fc4283d2ebed2297d60afbbc963b0a8abea48110` and wrote a result with SHA-256
`43953e585d72dab5bc2aa0b5cfd47488045a49212f86027977ee1d6aa0d533f7`.

## Frozen physical bounds

These bounds were registered before the first measurement. Decode TPOT must
remain between the 69,206,016 ps resident-weight floor and 150,000,000 ps
roofline-plus-network ceiling. The enabled effect cannot exceed 1,456,158 ps
on one node or 13,105,420 ps across nodes. Prefill TTFT has the same compute
floor and a 500,000,000 ps ceiling. Moving from 3.6 Tbit/s to 400 Gbit/s should
scale a serialization-driven effect by about nine. A roughly 100 microsecond
analytic decode would imply an implausible roughly 10,000 tokens per second,
so the mission study's 5x to 22x optimism budget remains the real-system
plausibility reference.

## Outcome

The three genuine-risk instances read raw per-request metrics before any
fatal guard:

| Relation | Measured | Frozen inclusive band | Result |
|---|---:|---:|---|
| B1 single-node producer reduction | 1.436193 percent | 1.25 to 1.65 percent | pass |
| B1 cross-node producer reduction | 11.587805 percent | 10.5 to 12.5 percent | pass |
| B2 cross/single absolute reduction ratio | 8.999138 | 7.5 to 10.5 | pass |

The serial and observed TPOT values were 100,931,331.913 ps and
99,481,763.174 ps on one node, then 112,574,121.739 ps and 99,529,252.174 ps
across nodes. The corresponding absolute reductions were 1,449,568.739 ps
and 13,044,869.565 ps. The enabled producer therefore moved TPOT in the
registered direction and bandwidth-scaled band.

The fatal guard set is reported as a single result: no guard was violated.
It is not reported as a score or added to the behavioral denominator. The
set covered arm attribution, all 32 producer schedules, sequential metric
history, source-derived legality, exact disabled artifacts, live rank
completion, the fixed serial fixture and the tracked pytest byte lock.

## Physical sanity against the frozen bounds

Compute and memory floor: 553,648,128 resident weight and LM-head bytes at
8 TB/s require at least 69,206,016 ps per decode. Every measured TPOT was
between 99,481,763.174 ps and 112,581,245.217 ps, above that floor.

Compute and network ceiling: the 0.7 roofline is about 99.5 microseconds and
the frozen cross-node serialized decode term is below 50 microseconds, so no
TPOT may exceed 150,000,000 ps. Every TPOT was below that ceiling. The
observed overlap terms were 1,450,472.652 ps and 13,051,993.043 ps, below the
pre-registered 1,456,158 ps and 13,105,420 ps communication ceilings.

Prefill floor and ceiling: prefill cannot beat its 69,206,016 ps compute
floor, while the 15,249,408-byte cross-node term gives a 500,000,000 ps
inclusive TTFT ceiling. Measured TTFT was 133,223,654 ps on one node and
404,324,160 ps across nodes for serial, control and observed arms.

Scaling check: moving from 3.6 Tbit/s NVLink to 400 Gbit/s RNIC increases the
absolute TPOT reduction by 8.999138 times, matching the expected ninefold
rate ratio. End-to-end plausibility remains poor by design: roughly 100
microseconds implies about 10,000 tokens per second for this
400M-active-parameter model, so these analytic values are not real-deployment
predictions.

This work moves none of the mission error-budget terms. Fixed host cost stays
0 ps, the packet-level mission collective floor stays 2.000 microseconds per
collective, and compute stays on the B100 roofline at a flat 0.7 derate. The
composed deployment gap remains 5x to 22x optimistic before and after this
qualification.

## Producer and metric-chain evidence

Each of the 32 enabled steps satisfied exact submission-order,
dependency-grammar, correlation-grammar, logical-rank-and-stream,
completion-frontier and original-identity checks. Every step covered ordered
layers 0 through 23 and 48 unique semantic dispatch/combine sites. The 23 DBO
steps carried 96 collective invocations each; the nine single-batch steps
carried 48 each.

The dependency oracle independently derived both dependency scopes from the
frozen layer, phase, rank and microbatch grammar. The correlation oracle
independently split each source-ordered scheduler request list and required
the exact batch, layer and microbatch fields on every operation. The rank and
queue oracle likewise derived both fields from operation identity, and its
regression test changes rank plus queue together to prove the check can fail.

Every logits operation waited on all final combines. Every request-visible
completion endpoint waited on all eight logits operations, partitioned the
scheduled request list without loss or duplication and reduced to the
original `r0`, `r1` and `r2` identities. The enabled cross-node live driver
matched the independent harness graph, execution result and `StepResult` on
all 32 steps. Routed request-pair tables remained exact across arms and
placements, exercising the landed TRAF-25 and VLLM-24 path.

The legal concurrency mechanisms were the named vLLM v0.26.0 cooperative DBO
wrapper threads, shared compute and communication streams, and modular-MoE
DeepEP event waits cited in the frozen expectations. Static inspection found
no overlap fraction, percentage, duration discount, random choice,
compatibility lowerer import or compatibility graph dependency. The exact
dependency grammar and compute-conservation checks constrain the result
independently of that source scan. Rank-local behavior beneath the audited
wrapper remains an inference because `deep_ep` itself was not installed.

The supported chain exercised by the result was:

```text
real vLLM scheduler and SimWorker model-forward boundary
  -> ExecutionObservations
  -> ObservedStepLowerer and routed traffic binding
  -> CoarseDeviceRuntime
  -> CompletionEvent and CompletionReducer
  -> StepResult
  -> original-request TTFT and TPOT
```

On all nine single-batch steps, control and observed were compared after the
same sequential prefix had populated independent `CompletionReducer`
histories. Both complete graphs, execution events, timestamps, completion
order, `StepResult` values and request metrics were exact. Steps 24 through
31 carried historical TPOT rather than isolated first-observation TTFT, which
closes the arm-equivalence failure mode that motivated this task.

## Exact producer-off evidence

The disabled real replay selected `SIMLLM_VLLM_OBSERVED_SCHEDULE=off` and
called a one-argument legacy sink exactly once on every translated step. An
independent `SerialStepLowerer`, `CoarseDeviceRuntime` and
`CompletionReducer` replay constructed the reference from the enabled
records. The repository-standard `BypassArtifacts` comparator found every
input and behavioral class equal:

- legacy diagnostic GOAL bytes;
- graph-derived GOAL bytes;
- canonical `StepRecord` bytes;
- serial graph JSON;
- execution events, timestamps and completion order;
- `StepResult` and request-metric bytes;
- profile, seed and canonical parameters.

The permanent pytest calls the actual one-argument wrapper without vLLM or a
`third_party` checkout, uses the same standard comparator and mutates each of
the seven byte fields by one byte to prove the lock can fail. The closure run
executed that test as a fatal guard and it passed. The fixed fixture also
retained 4,127 graph bytes at SHA-256
`aa3c836fe559973a7bf0940384c2e8a84e6af84e0fbd2c02d3b89774ee0c8e2d`
and 1,880 GOAL bytes at SHA-256
`7087db6780f7e34f5a559a6505eeccc15d984c7b478cd8f0bc5838053825d4b6`.

## Evidence classes and entailment

Evidence classes remain separate: three genuine-risk behavioral instances,
one fatal-unscored guard set, six raw timing cells, two live replays, producer
inventories, exact artifact comparisons, the fixed fixture and test
executables. No conservation, identity, by-construction or test count is
added to the behavioral denominator.

B1 can fail if the enabled observation tuple is ignored, produces the wrong
effect or changes TPOT outside either frozen band. B2 can fail if the effect
does not respond to the ninefold bandwidth change. Both are evaluated from
raw metrics before the exact and fatal checks, so no earlier oracle entails
their outcome. The disabled identity and every structural preservation fact
are fatal-unscored.

## Freeze integrity and run chronology

The expectations file is byte-unchanged from commit
`6459c3c469bb4bbddb47fbfdd1f6ed5ba8b86949`. No commit after the first
measured run changed modeled behavior.

| Attempt | Observed commit | Result SHA-256 | Status | B1 single/cross; B2 |
|---|---|---|---|---|
| one | `74e9e694e5d64dfd38fdc08f8714986d06838560` | `4c61e9b970d04639aa5cb78318b67f3dffdcf79a6923398e0e413b3bf3e1a04f` | VOID | 1.436193%, 11.587805%; 8.999138 |
| two | `b9c6f6b800efd26340db7292880f22c55bdda4db` | `7c323f7232d9f78ed52027b47837cadcc1ea420794e5d92a2572b26b30a942d1` | superseded | 1.436193%, 11.587805%; 8.999138 |
| three | `fc4283d2ebed2297d60afbbc963b0a8abea48110` | `43953e585d72dab5bc2aa0b5cfd47488045a49212f86027977ee1d6aa0d533f7` | VALID | 1.436193%, 11.587805%; 8.999138 |

The closure documentation and ledger change after attempt three changes no
executable or modeled behavior.

- Commit `74e9e694e5d64dfd38fdc08f8714986d06838560` implemented the harness before
  any measurement.
- Attempt one at that commit was VOID. It measured the same B1 values and B2
  ratio reported above, but declared failures in
  `control_observed_single_batch_identity`, `disabled_live_artifacts` and
  `disabled_live_step_records`. The raw result is retained with SHA-256
  `4c61e9b970d04639aa5cb78318b67f3dffdcf79a6923398e0e413b3bf3e1a04f`.
- Commit `b9c6f6b800efd26340db7292880f22c55bdda4db` fixed checker defects. It
  replaced a GOAL identity comparison that included metadata on only one
  side, compared same-input rather than offset cumulative arm clocks, used
  the standard byte comparator, added independent graph grammar and made the
  actual wrapper pytest fatal. It changed no modeled behavior. Attempt two
  printed VALID and retained identical B1 and B2 measurements, but review
  found that its isolated single-batch comparison lacked prior reducer
  history and that rank plus queue were not independently pinned. It is
  retained but superseded, with result SHA-256
  `7c323f7232d9f78ed52027b47837cadcc1ea420794e5d92a2572b26b30a942d1`.
- Commit `fc4283d2ebed2297d60afbbc963b0a8abea48110` fixed those checker defects.
  It replayed identical sequential prefixes into both reducers, derived rank
  and queue from the frozen grammar and strengthened import-alias inspection.
  It changed no modeled behavior. Attempt three retained every B1 and B2
  measurement exactly, then passed the complete guard set.

There is therefore no before/after modeled-behavior measurement to disclose:
all three runs produced identical serial and observed TPOT, absolute
reductions, relative reductions, TTFT and B2 ratio. The later commits only
changed what evidence was recorded and which false-closure paths the checker
could reject.

## Closure scope

The VLLM-22 acceptance clauses map as follows:

> "distinguish DBO from the TRAF-9 and terminal-frontier differences"

The structure-matched control adds only cross-microbatch serialization edges.
The exact decomposition measured structure at -903.913 ps single-node and
-7,123.478 ps cross-node, separately from 1,450,472.652 ps and
13,051,993.043 ps of DBO overlap. The structure term stayed within the frozen
0.01 percent bound.

> "preserve exact submission order, logical streams, dependencies, request
> correlation and completion frontiers"

All six independently derived inventories passed on all 32 steps; negative
tests prove dependency, correlation and coupled rank/queue defects are
detectable.

> "reach TTFT and TPOT through the supported metric chain"

The live scheduler-to-`StepResult` chain returned the TTFT and TPOT values
reported above for serial, control and observed arms on both placements.

> "name the wrapper or measured mechanism that makes each concurrency legal
> and derive no edge from an overlap percentage or compatibility schedule"

The frozen source audit names the DBO cooperative wrapper, shared streams and
DeepEP event waits. Source and exact grammar checks found no prohibited knob
or compatibility-derived edge.

> "Completion reduction must return the original request identities;
> per-request routed-byte acceptance depends on TRAF-25 and VLLM-24"

Every live and independent result used only `r0`, `r1` and `r2`, with active
request order preserved, and the post-TRAF-25 request-pair tables agreed
exactly across all arms and placements under the VLLM-24 conservation path.

> "With the producer absent, preserve the legacy sink call, serial graph and
> GOAL bytes, timestamps and completion order exactly."

The real disabled replay and permanent pytest prove every listed artifact
class through the standard comparator, including mutation controls.

Every registered clause is demonstrated. No residual task IDs were newly
registered; VLLM-26, VLLM-27 and VLLM-28 remain unused. The wrapper-below-
DeepEP inference and the unchanged absolute-timescale error budget are study
limitations, not undemonstrated VLLM-22 acceptance clauses.
