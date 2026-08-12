# Arbitrated co-runnable order v1 expectations

This is the expectations-only record for CORE-49 and CORE-10. It freezes what
the coarse device runtime must hand to the concurrent compute service once the
co-runnable group order is derived from the arbitration policy instead of from
`ExecutionGraph` tuple order, and it freezes the first two non-identity
policies that make that difference observable.

## Claim boundary and chronology

`CoarseDeviceRuntime._select_ready_operation` already consults the arbitration
policy. `CoarseDeviceRuntime._compute_group` then rebuilds the co-runnable set
as `tuple(sorted(candidates, key=operation_index.__getitem__))`, which is graph
order, and never consults the policy. Under the identity policy the two orders
coincide, so the defect is invisible to every accepted study.

This registration therefore claims exactly two things and nothing more:

1. the ordered tuple the compute service receives follows the arbitration
   grants, and that change reaches a reported step metric;
2. strict priority and weighted round robin exist as class-aware policies with
   an explicit identity setting each.

It does not change the `ArbitrationPolicy` protocol, does not add a scheduling
law, does not touch which operations co-run, and makes no silicon claim. The
synthetic 1 GHz fixture below is a mechanism fixture, replicated from the
already published COMP-12 registration, never a B100, H100 or Turing anchor.

The evidence is authored against SimLLM commit
`aeb40ac95cdd8163942297335948c94df0376e04`. Before this freeze no
implementation of the behavior existed and no live case had been executed. The
component rows quoted from COMP-12 were measured and published earlier, in
`examples/gpu_task_mix/RESULTS.md` and `examples/mixed_makespan_v1/RESULTS.md`;
they are prior knowledge, disclosed here, and this study reuses them as fixed
inputs rather than rediscovering them.

## Pre-freeze source audit

- `simllm/core/runtime.py`, `_select_ready_operation`: builds
  `ArbitrationCandidate` records from the ready set, calls `policy.select`
  once, and rejects a grant outside the legal set. This is the seam that
  already works.
- `simllm/core/runtime.py`, `_compute_group`: collects the co-runnable set on
  one rank and returns it sorted by `operation_index`. This is the defect.
- `simllm/core/runtime.py`, `_schedule_compute_group`: builds one `GpuTask`
  per member in the order it receives and passes that tuple to
  `SmSchedulerModel.estimate_concurrent`. The order therefore leaves the
  runtime through exactly one call.
- `simllm/compute/gpu_model.py`, `estimate_concurrent`: replays the tasks in
  the supplied order; issue candidates are ordered by task index, block and
  warp, so the first task in the tuple wins contended cycle-zero issue
  resources.
- `simllm/core/runtime.py`, `_schedule_launches`: the default profile has
  `launch_service_ps = 0`, so every operation of one graph with no
  `not_before_ps` becomes eligible at the graph release timestamp.
- `simllm/core/execution.py`: `ExecutionOperation.priority` is the class label
  the seam reads. Its numeric direction is undefined today; this registration
  defines a smaller label as the more favored class and says so in the policy
  contract rather than assuming a convention.

The audit identifies no measured basis for any closed form outside the exact
fixtures and sweeps below.

## Frozen policy contract

Three policies participate. All three receive candidates in the resource's
deterministic baseline order and must return one member of the tuple.

`IdentityArbitrationPolicy` (already accepted, unchanged): returns the
candidate with the smallest `baseline_sequence` and ignores every class label.

`StrictPriorityArbitrationPolicy(class_aware=True)`: returns the candidate
minimizing `(class_label, baseline_sequence)`. It is stateless, so repeated
grants over a shrinking candidate set reproduce one total order. With
`class_aware=False` it ignores class labels and returns the smallest
`baseline_sequence`, which is exactly the identity policy.

`WeightedRoundRobinArbitrationPolicy(weights, default_weight=1,
class_aware=True)`: keeps one integer credit counter per class label, carried
across grants and across executions of the same policy object. A class not yet
seen starts with its weight. One grant is served as follows:

1. `present` is the ascending list of distinct class labels in the candidates;
2. `eligible` is the members of `present` with a remaining credit above zero;
3. if `eligible` is empty, every class in `present` is refilled to its weight
   and `eligible` becomes `present`;
4. the winning class is the smallest label in `eligible`, its credit is spent,
   and the grant is the smallest `baseline_sequence` inside that class.

With `class_aware=False` it ignores labels and credits and returns the smallest
`baseline_sequence`, which is exactly the identity policy.

## Frozen runtime grant model

The runtime grants one operation at a time. Per scheduling iteration it issues
exactly one ready-seam grant, and when the winner is compute work it then
issues exactly one grant per member of the co-runnable set, each time offering
the members not yet granted, in baseline order. A step whose graph is one
co-runnable group of `N` compute operations therefore produces `1 + N` grants.

This is the interpretive precondition of every weighted-round-robin sequence
below, because that policy carries credits across grants. If the observed grant
count or the observed candidate sets differ from this model, the run is void.

## Frozen synthetic fixture

The 1 GHz two-SM mechanism profile of COMP-12 is restated here so this study
owns its own inventory. One cycle is exactly 1,000 ps.

| Property | Frozen value |
|---|---:|
| SMs | 2 |
| scheduler issue budget per SM | swept, 4 or 8 instructions per cycle |
| load/store lanes per SM | swept, 4 or 8 |
| ALU lanes per SM | 4 |
| HBM service | 64 bytes per cycle plus 100 cycles return latency |
| NVLink egress service | 16 bytes per cycle plus 200 cycles return latency |
| shared memory per SM | 65,536 bytes, unused by every launch here |
| task shape | 8 CTAs, one warp per CTA, four instructions per warp |

Three launches, with the isolated durations already published by COMP-12:

| Launch | Instruction stream | Isolated cycles |
|---|---|---:|
| `memory` | 4 HBM loads of 64 bytes per warp | 132 |
| `network` | 4 NVLink stores of 64 bytes per warp | 328 |
| `compute` | 4 ALU instructions per warp | 7 |

Two live fixtures, both on rank 0 of one request, all operations independent
and released together, three steps per case with a private virtual clock
starting at 5,000 ps:

- fixture `F2`, graph order `(memory, network)`;
- fixture `F3`, graph order `(memory, network, compute)`.

For these fixed-step fixtures TTFT and each decode TPOT equal the step JCT.
That equality demonstrates the supported metric chain but is entailed by
repeated equal steps, so it is unscored reachability evidence.

## Family A, the arbitrated order reaches the compute service

Evaluated on fixture `F2` from the raw ordered tuple the compute service
receives and the raw `StepResult.step_latency_ps`, identically in all three
steps of each case. The 328 and 329 cycle values are COMP-12 measurements
reused as fixed inputs: a concurrent makespan is the longest isolated control
plus a one-cycle issue delay charged to whichever task is submitted first when
both per-SM issue currencies are narrow.

| Instance | policy | class labels `(memory, network)` | issue budget | lanes | ordered tuple | step JCT |
|---|---|---|---:|---:|---|---:|
| A1 | strict priority | `(2, 1)` | 4 | 4 | `(network, memory)` | 328,000 ps |
| A2 | strict priority | `(1, 2)` | 4 | 4 | `(memory, network)` | 329,000 ps |
| A3 | strict priority | `(2, 1)` | 8 | 8 | `(network, memory)` | 328,000 ps |
| A4 | weighted round robin, weights `{1: 2, 2: 1}` | `(2, 1)` | 4 | 4 | `(network, memory)` | 328,000 ps |
| A5 | weighted round robin, weights `{1: 2, 2: 1}` | `(1, 2)` | 4 | 4 | `(memory, network)` | 329,000 ps |

The identity baseline of the same fixture is `(memory, network)` with 329,000
ps at issue budget 4 and lanes 4, and `(memory, network)` with 328,000 ps at
issue budget 8 and lanes 8.

A1 is the discriminating instance: it is the only one whose ordered tuple and
step JCT both differ from the identity baseline at the same issue currencies.
A2 and A5 are label controls, since a policy that merely reversed the graph
order would produce 328,000 ps there. A3 is the mechanism control: the ordered
tuple still follows the policy, and the policy-caused JCT difference against
the identity baseline is exactly 0 ps once both issue currencies are widened,
which is the registered COMP-12 behavior of the issue-delay term rather than an
arbitrary policy artifact. A4 shows the same reordering under a second policy.

Under the present defect A1, A3, A4 fail on the ordered tuple and A1, A4 also
fail on the step JCT, so the family is falsifiable by the code it is meant to
fix.

## Family B, class-aware policies order by their own contract

Evaluated on fixture `F3` at issue budget 4 and lanes 4, from the raw ordered
tuple the compute service receives in each of the three steps. Graph order is
`(memory, network, compute)`, so the identity order is that same tuple.

| Instance | policy | class labels `(memory, network, compute)` | step 0 | step 1 | step 2 |
|---|---|---|---|---|---|
| B1 | strict priority | `(2, 1, 1)` | `(network, compute, memory)` | `(network, compute, memory)` | `(network, compute, memory)` |
| B2 | weighted round robin, weights `{1: 2, 2: 1}` | `(2, 1, 1)` | `(network, memory, compute)` | `(network, compute, memory)` | `(network, memory, compute)` |
| B3 | strict priority | `(3, 2, 1)` | `(compute, network, memory)` | `(compute, network, memory)` | `(compute, network, memory)` |

B1 and B3 are constant across steps because strict priority is stateless. B2
alternates with period two because weighted round robin carries credits across
grants, and that alternation is the observable that separates it from strict
priority: with the same labels the two policies disagree on step 0 and step 2.
B2 is derived from the frozen grant model, four grants per step, and is stated
here before any implementation exists.

Fixture `F3` step JCTs are raw observations, not predictions. No closed form for
a three-task concurrent makespan is registered by this study or by any accepted
one.

## Physical sanity before precision comparison

Bounds are stated from first principles before any measured value is read.

| Case | first-principles floor | conservative ceiling | expected location |
|---|---:|---:|---:|
| `F2` concurrent | `max(132, 328) = 328` cycles | `132 + 328 = 460` cycles | 328 or 329 |
| `F3` concurrent | `max(132, 328, 7) = 328` cycles | `132 + 328 + 7 = 467` cycles | unregistered, inside the interval |
| `F2` with a dependency | `132 + 328 = 460` cycles | `460` cycles | 460, at both bounds |

The floors follow from work that cannot complete before its longest isolated
control, and from the fact that no reordering can make a task finish before its
own service. The ceilings serialize the isolated controls completely. A
reordering policy can therefore move a `F2` makespan by at most 132 cycles and a
`F3` makespan by at most 139 cycles; a larger policy-caused move is a defect
regardless of which order produced it. Landing outside an interval voids the run
before any exact relation is interpreted.

Second angle, the arithmetic of the moved term. The registered move is one cycle
at 1 GHz, which is 1,000 ps, and it must vanish exactly when both per-SM issue
currencies are doubled. If widening the currencies changed the move to anything
other than zero, the term would not be the issue-order term COMP-12 measured.

Third angle, end-to-end plausibility. These are nanosecond-scale synthetic
kernels on a two-SM fixture. The step JCT of 329,000 ps is 329 ns and is a
mechanism number, not a deployment metric: no real decode step of a served model
completes in 329 ns, and nothing here is offered as a performance claim. The
value of the live path is that the policy decision reaches `StepResult`, TTFT and
TPOT at all, not that these particular durations resemble hardware.

## Evidence classes and entailment

The scored headline contains exactly two genuine-risk families:

1. family A, the arbitrated order reaches the compute service, five
   parameterized instances;
2. family B, class-aware policies order by their own contract, three
   parameterized instances.

That is two families and eight instances. Each predicate is evaluated from raw
ordered tuples and raw `StepResult` values before any identity, conservation,
inventory or artifact guard runs, so no earlier fatal oracle pins a scored
quantity. Families A and B are not statistically independent: both read the
ordered tuple of the same seam, A through a timing projection and B through a
multi-class order. They stay separate because A alone cannot show that a policy
follows its own contract rather than any reordering, and B alone cannot show
that the order reaches a reported metric.

The 328 and 329 cycle inputs are COMP-12 measurements. Family A does not
rediscover them; it registers which of them the runtime must now produce under
which policy. Full prior knowledge of those two values is disclosed.

Run configurations, raw observations, physical intervals, identity checks,
inventories, grant models, queue-visit semantics, completion events, focused
tests and repository gates are separate evidence classes. Configuration-forced
equalities, conservation identities, repeated-step TTFT and TPOT equality,
clean-worktree state and deterministic serialization are fatal or
by-construction guards and never enter the behavioral denominator.

Any fatal-guard failure voids the complete run. A void run retains its raw
findings, reports no behavioral fraction, and closes nothing.

## Frozen fatal guards

After the raw relations are evaluated, the runner must require:

- the exact synthetic profile, launch inventory and isolated controls above,
  with `memory` at 132 cycles, `network` at 328 cycles and `compute` at 7
  cycles;
- all physical intervals above;
- the frozen grant model: `1 + N` grants per step, the first offering every
  ready operation and each later one offering exactly the members not yet
  granted, in baseline order;
- every grant is a member of the candidate tuple it was offered, and a policy
  returning a foreign candidate is rejected before any state mutates;
- the co-runnable membership is identical under every policy on both fixtures,
  so only the order changes, with no loss and no duplication;
- the omitted policy, an explicit `IdentityArbitrationPolicy`,
  `StrictPriorityArbitrationPolicy(class_aware=False)` and
  `WeightedRoundRobinArbitrationPolicy(class_aware=False)` produce byte-identical
  ordered tuples, step latencies, TTFT, TPOT, execution results, runtime
  reports, visit counts, wait sums, byte counts, random draw counts and
  completion orders on both fixtures;
- the same holds under class-label permutation for every identity setting, so
  labels move nothing when the class-aware setting is off;
- the identity ordered tuples are literally `(memory, network)` and
  `(memory, network, compute)`, pinned as values rather than as a comparison
  against another identity run;
- the scalar compatibility path, which supplies nominal durations of 132,000 ps
  and 328,000 ps and no kernel service, stays order invariant at 328,000 ps
  under both the identity baseline and strict priority favoring `network`;
- with a dependency from `memory` to `network` and strict priority favoring
  `network`, the dependent operation does not move: each grant offers exactly
  one candidate, the compute service receives `(memory,)` then `(network,)`,
  and the step JCT is 460,000 ps under both the identity baseline and strict
  priority, which is the mandatory-ordering-before-arbitration requirement;
- `CompletionEvent.QUEUED` equals resource eligibility and
  `CompletionEvent.STARTED` equals the resource grant, and each operation has
  exactly one logical completion event at its reported completion;
- each live `StepResult` and request metric conserves the runtime completion;
- every live per-operation completion equals the group start plus the component
  `estimate_concurrent` completion cycle of the same ordered tuple, so the live
  projection and the component observables agree, and the per-task admission
  cycles of that replay are recorded as the identifying observable CORE-49 asks
  for;
- the registered production run starts from a clean worktree and writes only
  beneath its explicit output directory.

## Registered command and pre-freeze dry run

The result-producing command is:

```text
.venv/bin/python examples/arbitrated_order_v1/run_study.py --out "$SIMLLM_ARBITRATED_ORDER_RUN_ROOT"
```

Before this expectations commit that exact CLI is run with `--check-only`.
Check-only parses the production arguments and validates only the frozen literal
registry and its internal arithmetic. It imports no SimLLM implementation, reads
no input or output path, invokes no native binary and creates no artifact. The
result records the final expectations-only commit and the observed SimLLM
revision separately, with no equality assumed between them.

## Closure scope

CORE-49 closes only if family A passes in full, every fatal guard passes, the
ordered tuple reaching the compute service is shown to follow the grants, the
identity path is pinned byte-identical by a test rather than asserted in prose,
and the live projection reaches `StepResult`, TTFT and TPOT.

CORE-10 closes only if family B passes in full, both named policies exist over
legal ready candidates, each carries an explicit identity setting whose
class-label permutation leaves the accepted baseline byte-identical, and the
mandatory-ordering guard shows that dependency and protocol order stay outside
the policy. CORE-10's registered text also sequences the work after CORE-8. That
clause is read as CORE-8's own condition for enabling a non-identity policy,
namely that the existing PCIe, WQ and GPU studies remain byte-identical under
identity arbitration; the run must demonstrate that condition and must state
plainly that the rest of CORE-8 stays open and is not claimed here.

Neither task closes on the other's evidence. If family A passes and family B
does not, CORE-49 closes and CORE-10 stays open, and the reverse is not
available because family B is meaningless if the order never reaches the
service.

Any registered acceptance clause this run does not demonstrate moves to CORE-54
or CORE-55 with the owning module and category tag before the closing entry is
removed. A clause that passes needs no new ID, and an adjacent improvement that
no registered clause claimed is recorded in prose instead of as a task.
