# Arbitrated co-runnable order v1 results

Registered production run of the expectations frozen in
[expectations.md](expectations.md). Both scored families pass in full and every
fatal guard holds, so the run is not void.

| Quantity | Value |
|---|---|
| genuine-risk families | 2 |
| genuine-risk instances | 8 |
| genuine-risk predicates | 18 of 18 pass |
| fatal guards | 44 of 44 hold, run not void |
| new IDs registered | 0 |
| expectations-only commit | `9d89d513baec9785093e8d95671051c78447379a` |
| post-freeze amendment commit | `5309d49744d61077b36f136a05ced037bfd3566d` |
| implementation commit | `6fde6c54a64a7fa9d9c2ed7c20f7f550a4bcbef3` |
| revision observed by the run | `47aba5fc539fb087eb194a265f1de06475f2fd7e` |
| commit the evidence was authored against | `aeb40ac95cdd8163942297335948c94df0376e04` |

Reproduce with:

```text
.venv/bin/python examples/arbitrated_order_v1/run_study.py --out "$SIMLLM_ARBITRATED_ORDER_RUN_ROOT"
```

## What was wrong

`CoarseDeviceRuntime._select_ready_operation` consulted the arbitration policy,
and `_compute_group` then rebuilt the co-runnable compute set as
`tuple(sorted(candidates, key=operation_index.__getitem__))`, which is
`ExecutionGraph` tuple order. `_schedule_compute_group` handed that tuple to
`SmSchedulerModel.estimate_concurrent`, whose replay orders issue candidates by
task index, so the order decides which task wins contended cycle-zero issue
resources. A class-aware policy could therefore win a grant while the compute
service still replayed graph order, and the submission-order issue term COMP-12
registered would have followed an order the runtime no longer chose. Under the
identity policy the two orders coincide, which is why every accepted study
passed over it.

The fix derives the group order from the same seam that selected the first
operation: the runtime grants one member at a time and offers the members not
yet granted in the deterministic baseline order. Membership is computed before
arbitration and is unchanged, so the policy decides only the order. Under
identity every grant is the smallest remaining baseline sequence, which is
exactly the previous graph order.

## Chronology

The expectations were frozen at `9d89d51`, before any implementation existed
and before any live case had been executed, with the registered CLI passing
`--check-only` beforehand. Amendment 1 was appended at `5309d49` after the
harness was debugged against the implemented behavior and before the registered
run, and it is disclosed below rather than folded into the freeze. The
implementation landed at `6fde6c5`, the runner and its regressions at `47aba5f`,
and the registered run observed `47aba5f` from a clean worktree.

Harness debugging runs were executed between `5309d49` and `47aba5f`, writing
only into a scratch directory outside the repository. They exercised the same
frozen literals; nothing frozen was edited to match an observation.

## Family A, the arbitrated order reaches the compute service

Fixture `F2` is `(memory, network)` on rank 0 of one request, three steps, at
the synthetic 1 GHz two-SM profile. Every row below held identically in all
three steps of its case.

| Instance | policy | labels `(memory, network)` | issue, lanes | ordered tuple | step JCT | identity baseline |
|---|---|---|---|---|---:|---:|
| A1 | strict priority | `(2, 1)` | 4, 4 | `(network, memory)` | 328,000 ps | 329,000 ps |
| A2 | strict priority | `(1, 2)` | 4, 4 | `(memory, network)` | 329,000 ps | 329,000 ps |
| A3 | strict priority | `(2, 1)` | 8, 8 | `(network, memory)` | 328,000 ps | 328,000 ps |
| A4 | weighted round robin `{1: 2, 2: 1}` | `(2, 1)` | 4, 4 | `(network, memory)` | 328,000 ps | 329,000 ps |
| A5 | weighted round robin `{1: 2, 2: 1}` | `(1, 2)` | 4, 4 | `(memory, network)` | 329,000 ps | 329,000 ps |

Each instance contributes three predicates: the ordered tuple, the step JCT and
the policy-caused move against the identity baseline at the same issue
currencies. All 15 pass.

A1 is the discriminating instance: the ordered tuple and the step JCT both move
against identity at the same issue currencies, and the move is exactly one
cycle, 1,000 ps. A2 and A5 are label controls and reproduce the identity value,
so the policies follow labels rather than reversing the graph. A3 is the
mechanism control: the ordered tuple still follows the policy while the
policy-caused JCT move is exactly 0 ps once both per-SM issue currencies are
widened, which is the registered COMP-12 behavior of the issue term rather than
an arbitrary policy artifact.

TTFT of step 0 and the TPOT of each of the two decode steps equal the step JCT
in every case, so the arbitration decision reaches the reported metric chain.
That equality is entailed by repeated equal steps and is unscored reachability
evidence, not a scored result.

## Family B, class-aware policies order by their own contract

Fixture `F3` is `(memory, network, compute)` at issue budget 4 and lanes 4, so
the identity order is that same tuple.

| Instance | policy | labels `(memory, network, compute)` | step 0 | step 1 | step 2 |
|---|---|---|---|---|---|
| B1 | strict priority | `(2, 1, 1)` | `(network, compute, memory)` | `(network, compute, memory)` | `(network, compute, memory)` |
| B2 | weighted round robin `{1: 2, 2: 1}` | `(2, 1, 1)` | `(network, memory, compute)` | `(network, compute, memory)` | `(network, memory, compute)` |
| B3 | strict priority | `(3, 2, 1)` | `(compute, network, memory)` | `(compute, network, memory)` | `(compute, network, memory)` |

All three pass exactly. B1 and B3 are constant because strict priority is
stateless and is a total order over any candidate set. B2 alternates with period
two because weighted round robin carries credits across grants, and that
alternation is what separates it from strict priority: with identical labels the
two policies disagree on step 0 and step 2. The recorded grants show the
mechanism directly. In step 0 the class-1 credits are spent by the ready-seam
grant and the first group grant, so class 2 takes the next grant and `memory`
lands second; in step 1 the round has already turned over, class 1 still holds a
credit at the third grant, and `compute` takes it instead.

## Falsification against the pre-fix code

The frozen families were replayed against the pre-fix `_compute_group`, which
sorted the group by `operation_index`. Six of the eight instances fail there:

| Instance | pre-fix ordered tuple | pre-fix step JCT | verdict |
|---|---|---:|---|
| A1 | `(memory, network)` | 329,000 ps | fails order and JCT |
| A2 | `(memory, network)` | 329,000 ps | passes, label control |
| A3 | `(memory, network)` | 328,000 ps | fails order |
| A4 | `(memory, network)` | 329,000 ps | fails order and JCT |
| A5 | `(memory, network)` | 329,000 ps | passes, label control |
| B1, B2, B3 | `(memory, network, compute)` | not evaluated | fail order |

The two passing instances are exactly the two the freeze named as label
controls, whose registered order is the graph order. This is a diagnostic
replay, not a scored family: it is reported so the falsifiability of the
registered rows is visible rather than asserted.

## Physical sanity before precision

Bounds were stated in the freeze before any value was read.

| Case | floor | ceiling | measured | location |
|---|---:|---:|---|---|
| `F2` concurrent | 328 cycles | 460 cycles | 328 or 329 cycles | at the floor or one cycle above |
| `F3` concurrent | 328 cycles | 467 cycles | 328 or 329 cycles | at the floor or one cycle above |
| `F2` dependent | 460 cycles | 460 cycles | 460 cycles | at both bounds |

Nothing landed outside an interval. The floors are the longest isolated control,
which the measured isolated durations reproduce exactly at 132, 328 and 7 cycles;
the ceilings serialize the isolated controls completely, which the dependent case
reaches exactly because a dependency removes the overlap.

Second angle, the arithmetic of the moved term. The component replay of the same
ordered tuples identifies where the cycle goes. Every task admits at cycle 0 in
this fixture, so residency is not the mechanism, and the per-task completion
cycles are:

| ordered tuple | issue, lanes | `memory` completes | `network` completes | makespan |
|---|---|---:|---:|---:|
| `(memory, network)` | 4, 4 | 132 | 329 | 329 |
| `(network, memory)` | 4, 4 | 133 | 328 | 328 |
| `(memory, network)` | 8, 8 | 132 | 328 | 328 |
| `(network, memory)` | 8, 8 | 132 | 328 | 328 |

The task submitted second pays exactly one cycle while the first keeps its
isolated duration, and the payment disappears when both issue currencies are
widened. The makespan therefore moves only when the second-submitted task is
also the longer one. That rule was not registered, and it predicts every
unregistered `F3` step latency in this run: 329,000 ps whenever `network` is not
granted first (identity and B3) and 328,000 ps whenever it is (B1 and B2). An
independent regularity that the registered mechanism predicts is stronger
evidence than the registered rows alone, and it is reported here as an
observation rather than promoted to a claim.

Third angle, end-to-end plausibility. These are 328 to 460 nanosecond synthetic
kernels on a two-SM fixture, and one cycle at 1 GHz is one nanosecond. No served
model decodes in 329 ns, and nothing here is offered as a deployment number. The
live path matters because the policy decision reaches `StepResult`, TTFT and
TPOT at all, not because these durations resemble hardware. The production
compute context is untouched by this change.

## Fatal guards

All 44 hold. They cover, in their own evidence class and never in a behavioral
fraction:

- the exact fixture, launch inventory and the 132, 328 and 7 cycle isolated
  controls;
- the three physical intervals above;
- the frozen grant model of `1 + N` grants per step, each offering exactly the
  members not yet granted, with every grant inside its offer;
- rejection of a grant outside the offered set before any state mutates,
  checked at the group seam rather than only at the ready seam;
- policy-invariant co-runnable membership on both fixtures, with no loss and no
  duplication, so arbitration changes order and nothing else;
- behavioral identity of the omitted policy, the explicit
  `IdentityArbitrationPolicy`, `StrictPriorityArbitrationPolicy(class_aware=False)`
  and `WeightedRoundRobinArbitrationPolicy(class_aware=False)` across all three
  registered labelings on both fixtures;
- the literal identity ordered tuples `(memory, network)` and
  `(memory, network, compute)`, pinned as values;
- the order-invariant scalar compatibility path at 328,000 ps, where the grant
  log shows the policy did reorder the group while the scalar service, which
  does not read order, produced the identical baseline;
- the dependent fixture, where strict priority favoring the dependent operation
  moves nothing: each grant offers one candidate, the service receives
  `(memory,)` then `(network,)`, and the JCT stays 460,000 ps;
- `CompletionEvent.QUEUED` at resource eligibility and `STARTED` at the resource
  grant, one logical completion event per operation, and `StepResult` and
  request-metric conservation of the runtime completion;
- live per-operation completions equal to the group start plus the component
  replay completion cycle of the same ordered tuple;
- a clean worktree, with output written only beneath the run directory.

## Amendment 1, disclosed

The frozen byte-identity guard required two identity settings to produce
byte-identical runtime reports under a class-label permutation. That is
unsatisfiable by any implementation: `RuntimeOperationRecord.class_label` echoes
`ExecutionOperation.priority` and `RuntimeReport.class_service_bytes` attributes
the same bytes per class label, so a report that repeats its own input cannot be
identical when the input is permuted. The repository's own accepted identity
regression already projected both fields out, so the freeze asserted something
stricter than any accepted study had ever claimed.

The guard was narrowed to behavior by projecting out those two passive echoes,
and strengthened by additionally requiring the total service bytes to be
conserved across every identity setting and labeling. The correction was
committed before the registered run with the original freeze left untouched. It
is a post-freeze correction of a mis-specified guard, not part of the original
registration, and no scored family depends on it.

## Evidence classes and entailment

Two genuine-risk families, eight instances, 18 predicates. Every predicate was
evaluated from raw ordered tuples and raw `StepResult` values before any
identity, conservation, inventory or artifact guard ran, so no earlier fatal
oracle pins a scored quantity. Families A and B read the same seam through
different projections and are not statistically independent; they stay separate
because A alone cannot show that a policy follows its own contract rather than
any reordering, and B alone cannot show that the order reaches a reported
metric.

The 132, 328 and 7 cycle controls and the one-cycle issue term are COMP-12 and
task-mix measurements reused as fixed inputs, with full prior knowledge
disclosed. This study did not rediscover them; it registered which of them the
runtime must produce under which policy. Run configurations, raw observations,
physical intervals, identity checks, grant models, inventories, queue-visit
semantics, completion events, focused tests and repository gates are separate
classes and are never added into one total.

## Closure

CORE-49 closes. Its registered clauses map as follows. "Derive the group order
from the same policy decision that selected the first operation" is the repeated
grant at `_arbitrated_order`. "Use the concurrent makespan and per-task
admission cycles as the identifying observables" is the component replay table
above, which reports admission cycles of 0 for every task and identifies the
moved cycle in the per-task completion cycles instead, since this fixture has no
residency pressure to make admission the discriminating observable. "Identity
arbitration and class-label permutation must preserve every accepted timestamp,
wait, byte count and completion order exactly" is the behavioral identity guard
across four identity settings and three labelings, pinned additionally by
`tests/test_device_runtime.py` as literal ordered tuples rather than asserted.

CORE-10 closes. "Start with strict priority and weighted round robin over legal
ready candidates" is `StrictPriorityArbitrationPolicy` and
`WeightedRoundRobinArbitrationPolicy`, each demonstrated by a family-B instance
and by family A. "Keep per-SQ ordering and protocol forward-progress rules
outside the policy" is the dependent-fixture guard plus the legality guard: the
policy is only ever offered candidates that mandatory filters already admitted,
and a grant outside the offer is refused. "Every policy has an explicit identity
setting whose class-label permutation leaves the accepted baseline
byte-identical" is `class_aware=False` on both policies, covered by the identity
guard and by the runtime regressions.

CORE-10's registered text also sequences the work "only after CORE-8 establishes
the policy seam and exact identity baseline". CORE-8 remains open and nothing
here claims any part of it. The specific condition CORE-8 states for enabling a
non-identity policy, that the existing PCIe, WQ and GPU studies remain
byte-identical under identity arbitration, holds: those studies' tracked
regressions are unchanged and green, and the policy seam is only reachable when
a caller passes a class-aware policy. The rest of CORE-8, the cross-layer
projection and the shared golden fixtures in both languages, is untouched.

Zero new IDs were registered. Every registered acceptance clause of both tasks
is demonstrated above, and the wave rule reserves a new ID for a registered
clause a run did not demonstrate rather than for an adjacent idea.

## What this run does not claim

- No closed form for a three-task concurrent makespan. The `F3` step latencies
  are raw observations, and the second-submitted-task rule that predicts them is
  reported as an observation from four component rows, not registered.
- No production reachability. No production step path selects the concurrent
  kernel service today, which is COMP-25's scope, so the live evidence here runs
  through the study's own fixture rather than through a served model.
- No claim that these two policies are the right ones for any deployment. They
  are the two CORE-10 named, and a third policy would need its own registered
  relation before it belongs in the runtime.
- No silicon claim. The 1 GHz two-SM profile is a mechanism fixture.
