# Corrected mixed-makespan v1 results

## Outcome

The registered run is **not void**. Every one of the 124 frozen fatal guards
passed, and all 11 genuine-risk instances across the four registered families
passed:

| Scored family | Instances | Passed |
|---|---:|---:|
| G1 component issue-delay matrix | 5 | 5 |
| G2 component residency and backfill form | 3 | 3 |
| live CORE-4 issue-order projection | 2 | 2 |
| live CORE-4 residency projection | 1 | 1 |

COMP-12 closes. Two undemonstrated residuals move to COMP-24 and COMP-25, and
one defect this closure exposed in CORE-4 moves to CORE-49; all three are
registered below. COMP-17 was declared out of scope in the freeze and stays
open unchanged.

The four families are separate evidence, never one headline total. The 124
fatal guards are a void condition, not a score, and are never reported as a
fraction. The 13 raw-observation and run-configuration rows assert nothing and
carry status `REPORTED` rather than a pass or fail.

## Chronology and provenance

| Event | Commit |
|---|---|
| Source task-mix study that measured G1 and G2 | `0d9e2337eab6d5e49c112f3fbccb7d5e70a44f7f` |
| Branch point the evidence was authored against | `76223875557a552deb5aa2c2c529a07f000135ba` |
| Expectations-only freeze, final pre-run state | `3d079077ae91699a14c180eaba0e534bca7a7e91` |
| Implementation, and the revision the run observed | `63f83ea2f3ecd3e8a77ede97c45cfece2a46fea4` |

The freeze precedes the implementation and the first production run, so the
registered relations are pre-registered. The freeze also discloses that a
read-only exploratory probe had already observed 329,000, 328,000 and 243,000
ps for the three principal live rows before the freeze was written. This is
therefore a pre-registered replication with fully disclosed prior knowledge,
not an unseen prediction, and the result claims nothing stronger.

The registered CLI passed `--check-only` before the freeze commit. Check-only
validates the frozen literal registry and its arithmetic, imports no SimLLM
implementation, reads no input, and writes nothing. The result-producing
command wrote only beneath `$SIMLLM_MIXED_MAKESPAN_RUN_ROOT`. The run started
from a clean worktree and the worktree was still clean after the study wrote
its artifacts; both halves of that guard are recorded in `summary.json`.

## Physical sanity before the exact comparison

Bounds first, digits second. Every synthetic row below is on the frozen
two-SM 1 GHz mechanism fixture, where one cycle is exactly 1,000 ps. That
fixture is not a B100, H100 or Turing calibration and no silicon claim is made
from it.

| Case | floor, longest isolated control | ceiling, fully serialized | measured | where it sits |
|---|---:|---:|---:|---|
| G1 baseline, memory then network | 328 | 460 | 329 | 1 cycle above the floor |
| G2 half-shared | 229 | 243 | 243 | exactly at the serialized ceiling |
| G2 zero-shared | 132 | 139 | 133 | 1 cycle above the floor |

No concurrent replay can finish before its longest isolated control, and full
serialization of the two controls is the conservative upper bound. All three
measurements are inside their interval. The frozen fatal guard is interval
membership; the exact location inside the interval is what the scored families
assert, so the guard does not pin the scored value.

The second angle is the production compute context, which is deliberately kept
apart from the synthetic fixture. Default callers select
`GPU_ENVELOPES["b100"]`, whose HBM envelope is 8.0e12 bytes/s. For the
motivating Granite decode fixture the roofline moves 556,449,792 bytes:

| Bound | Value |
|---|---:|
| B100 at 100 percent bandwidth, an absolute hardware floor | 69.556224 us |
| B100 at the configured 0.7 achievable bandwidth | 99.366034 us |
| `RooflineProvider` integer estimate against B100 | 99.366034 us |
| H100 at 100 percent bandwidth, for contrast only | 166.104415 us |

The modeled 99.366034 us sits exactly at the configured floor, which is
1.429 times the hardware roof, i.e. exactly `1 / 0.7`. Labeling that number
H100 would be invalid: 556,449,792 bytes alone need about 166.104 us at
H100's 3.35e12 bytes/s peak, so an H100 attribution would place the step
1.67 times below its own memory traffic. There is no honest finite ceiling
here without a measured minimum sustained bandwidth or timeout, and the result
says so rather than inventing one.

The third angle is end-to-end plausibility, which requires disclosing what the
roofline omits. `HostInitiationModel` defaults to `initiation_delay_ps=0` with
profile `ideal`, so a modeled step charges no kernel launch, no host
scheduling and no device scheduling. Any absolute-fidelity gap against a real
decode step therefore starts with that omission, which COMP-2 already owns.
It is not evidence for or against either form registered here.

## G1: the issue-order form

Registered form, for the exact frozen pair:

```text
T_mixed = max(isolated durations) + delta_issue
```

where `delta_issue` follows the ordered co-runnable tuple the caller
submitted. Measured isolated controls: memory 132 cycles, network 328 cycles.

| ordered tuple | issue budget | load/store lanes | `delta_issue` | mixed cycles |
|---|---:|---:|---:|---:|
| memory, network | 4 | 4 | 1 | 329 |
| memory, network | 8 | 4 | 1 | 329 |
| memory, network | 4 | 8 | 1 | 329 |
| memory, network | 8 | 8 | 0 | 328 |
| network, memory | 4 | 4 | 0 | 328 |

Five of five instances match the frozen matrix exactly.

The per-task completions identify the cause more sharply than the makespan
does, and they are worth quoting because they are symmetric:

| ordered tuple | memory completes | network completes |
|---|---:|---:|
| memory, network | 132 | 329 |
| network, memory | 133 | 328 |

The delay does not appear and disappear with the order. It is always one
cycle, and it always lands on whichever task lost the cycle-zero issue
resources. Submitting memory first costs the critical NVLink store its first
cycle and the makespan moves; submitting network first pushes the same one
cycle onto the memory task, which finishes at 133 instead of 132 and is not
on the critical path. That is the behavior of a shared issue path, and it is
not what a `max(isolated)` model or a task-kind heuristic would produce.

The counterfactual sweep identifies which resource binds. Doubling the per-SM
scheduler budget alone leaves 329; doubling the load/store issue width alone
leaves 329; doubling both recovers 328. Neither resource alone is the cause:
whichever is scarcer binds, so the correct statement is a submission-order
delay set by the binding per-SM issue currency, not "the scheduler" or "the
lanes".

`delta_issue` as a subtraction is a definition, not a prediction, and the
implementation says so in `MixedMakespanForm.issue_delay_cycles`. The genuine
risk in this family lives in the five parameterized instances: the order
sensitivity, the two single-resource counterfactuals and the joint one.

## G2: the residency form

Registered form, for a single residency-gated task:

```text
T_mixed = admitted_cycle(gated task) + isolated duration(gated task)
```

With each CTA claiming 32,768 bytes, exactly half an SM's shared memory, an
SM holds two CTAs in total across both tasks, so the 8-CTA compute launch
occupies the four slots across two SMs and the memory task cannot admit until
it finishes.

| relation instance | expected | measured |
|---|---:|---:|
| half-shared tasks serialize, `14 + 229` | 243 | 243 |
| memory admits when constrained compute finishes | 14 | 14 |
| zero-shared control restores backfill plus the issue delay | `(133, 1)` | `(133, 1)` |

Three of three instances pass. The admission equality is part of the form and
not decoration: a 243-cycle makespan on its own would not identify residency
as the cause, because 243 is also what a fully serialized pair would give for
any other reason. The measured regime is `residency-serialized` with `memory`
as the gated task, its residency delay is 14 cycles, and
`admitted + isolated = 14 + 229 = 243` reproduces the makespan.

The zero-shared control removes the residency currency while keeping the same
instruction streams. Isolated durations fall to 7 and 132, backfill happens,
and the makespan is 133, i.e. the maximum plus the same independently
registered one-cycle G1 term. Shared memory is a residency currency, and a
co-scheduled kernel is free only while the SM has room for it.

## Live CORE-4 and the metric chain

Each live case places both kernel launches on rank 0 in separate logical
queues, correlates both operations to one request, and runs three identical
steps on a private virtual clock: one prefill and two decode steps.
`CoarseDeviceRuntime` receives the same `SmSchedulerModel` and the same
`KernelLaunch` records as the component replay, forms the co-runnable group in
`ExecutionGraph` tuple order, and passes that ordered tuple to
`estimate_concurrent`. Its task completion cycles project through GPU queue
visits, `CompletionEvent`, `ExecutionResult`, `RuntimeReport`,
`CompletionReducer`, `StepResult`, TTFT and TPOT.

Scored from raw `StepResult.step_latency_ps`:

| issue budget | load/store lanes | memory-first minus network-first JCT | measured |
|---:|---:|---:|---:|
| 4 | 4 | +1,000 ps | +1,000 ps |
| 8 | 8 | 0 ps | 0 ps |

```text
JCT_half_shared - JCT_zero_shared = 243,000 - 133,000 = +110,000 ps  (measured +110,000 ps)
```

Three of three, plus the residency instance, pass. A difference cannot see a
constant offset added to both sides, so the absolute step latencies are in the
record as raw observations. Every one of the ten live cases measured the same
value on all three of its steps, and those values are exactly the component
cycles at 1,000 ps per cycle:

| live case | step JCT, all three steps |
|---|---:|
| memory-first, sched 4, lanes 4 | 329,000 ps |
| network-first, sched 4, lanes 4 | 328,000 ps |
| memory-first, sched 8, lanes 8 | 328,000 ps |
| network-first, sched 8, lanes 8 | 328,000 ps |
| half-shared residency | 243,000 ps |
| zero-shared control | 133,000 ps |
| explicit identity policy, memory-first | 329,000 ps |
| permuted priority labels, memory-first | 329,000 ps |
| scalar compatibility, either order | 328,000 ps |

There is no constant offset. The runtime adds nothing to the component result
for this fixture, which is expected: this graph contains only compute work,
and CORE-4 charges no fixed per-operation cost on that path.

TTFT and each decode TPOT equal the step JCT in every case. That demonstrates
the supported metric chain end to end, but for repeated equal steps it is
algebraically entailed, so it is reachability evidence and stays unscored.

## Evidence classes and the entailment analysis

Four families and 11 instances are genuine risk. Each predicate is evaluated
from raw isolated durations, concurrent makespans, admission cycles or
`StepResult` latencies before any fixture, identity, event, conservation or
artifact guard runs.

The freeze registered the physical guard as an interval, and the first draft
of the runner over-implemented it as equality to the interval's expected
location. That would have fatally pinned 329, 243 and 133, the exact values
the scored families assert, converting genuine risk into bookkeeping. The
landed runner asserts only membership plus the interval's own endpoints. For
the same reason the fixture-inventory guard fixes the configuration (SM count,
issue widths, shared memory, clock, HBM and NVLink service, and each launch's
blocks, warps, instructions, opcodes, transaction bytes and shared bytes) and
not the measured isolated durations, which are reported raw instead.

Two guards the freeze did not register were also dropped from the guard class
for the same reason and are reported as raw observations: the measured regime
labels and the `admitted + isolated = makespan` identity. Adding an
unregistered fatal guard could void a run for a reason the freeze never
declared, and the residency identity in particular would have entailed the
243-cycle scored row given the admission row and the isolated control.

The component and live families are not statistically independent. The live
families deliberately re-measure the same scheduler through a different
interface, so their value is integration evidence, not a second sample of the
mechanism. They stay separate rows and their counts are never added to the
component counts or to any fatal or test count.

Fatal and by-construction guards, all passing and all unscored: fixture
configuration and per-launch inventory (6), physical interval membership (3),
nonnegative component timestamps (1), task-kind relabelling moving labels only
(1), per-task instruction and byte conservation (7), identity-policy and
priority-permutation preservation (2), scalar compatibility order invariance
(1), live graph order and identity with no loss or duplication (30),
queue-visit contract with `QUEUED` at eligibility, `STARTED` at the grant and
one logical completion event per operation (30), live metric conservation
(30), B100 envelope and roofline arithmetic (2), run hygiene (1), and
repeated-step TTFT/TPOT reachability (10). Total 124.

## Closure scope

The registered COMP-12 acceptance text, clause by clause:

| Clause | Evidence | Status |
|---|---|---|
| "register the corrected mixed-makespan forms measured by the task-mix study" | `MixedMakespanForm` and `decompose_mixed_makespan`; both forms written down in the compute module doc; the freeze registered them before implementation | met |
| "a concurrent makespan is `max(isolated durations)` plus a submission-order issue delay" | G1 component family, 5 of 5, plus the symmetric per-task completions and the two single-resource counterfactuals | met |
| "tasks whose CTAs exhaust an SM's shared memory serialize on residency instead of backfilling" | G2 component family, 3 of 3, including the admission equality and the zero-shared backfill control | met |
| "Both need a pre-registered form of their own" | expectations-only commit `3d07907` precedes implementation `63f83ea` and the first run | met |
| "including how the issue-order delay should behave once CORE-4 owns submission policy" | documented, and demonstrated live for the omitted and explicit identity policies plus priority-label permutation | met for the identity policy only |
| the same clause, for a class-aware policy | not demonstrated; the runtime cannot satisfy it today | moved to CORE-49 |

The last row is a defect this closure exposed rather than a scope choice.
`_select_ready_operation` consults the arbitration policy, but `_compute_group`
then rebuilds the co-runnable group as `sorted(candidates, key=operation_index)`,
i.e. `ExecutionGraph` tuple order, and never consults the policy. Under the
identity policy the two orders coincide, which is exactly why the registered
G1 term could be observed live against the graph order. A CORE-10 class-aware
policy would reorder its selection while the compute service still received
graph order, and the registered form would then follow an order the runtime no
longer chose. CORE-49 owns it.

Two further residuals are registered rather than claimed:

- **COMP-24**: the forms were measured on one fixture with one gated task.
  `decompose_mixed_makespan` refuses a replay in which more than one task
  waited for residency, and no measured row covers other shared-memory
  fractions, register or warp pressure, launch shapes or instruction mixes.
- **COMP-25**: the concurrent kernel service is live-reachable and this study
  drove it to TTFT and TPOT, but no production study or step sink selects it.
  Production steps still take the scalar `nominal_duration_ps` path, whose
  concurrent makespan is the independent-resource maximum and carries neither
  registered form. That off path is a registered fatal guard here (328,000 ps,
  order-invariant) and is honest about claiming nothing.

COMP-17 is untouched. The freeze placed it outside this run because the
tracked Turing table holds aggregate family, shape and dtype cells rather than
per-layer identities or measured per-layer durations, and COMP-6 still owns
the per-invocation shapes that `estimate_layers` would need. Presenting Turing
numbers as a B100 or H100 layer anchor would be invalid, so no layer estimates
were populated and COMP-17 remains open unchanged.

## Contradiction sweep

`README.md` and `docs/README_PRO.md` contain no statement contradicted by this
closure.

`docs/architecture.md` line 561 states that in `estimate_concurrent` "a later
task backfills capacity an earlier one cannot use", with no residency
qualification. Finding G2 shows backfill happens only while the SM still has
room: with each CTA claiming half the shared memory, the later task does not
backfill at all and the two tasks serialize. The sentence is incomplete rather
than wrong, and it is reported here rather than edited.

## Claim boundaries

- The synthetic 1 GHz fixture is a deterministic mechanism fixture. Nothing
  here calibrates B100, H100 or Turing silicon, and the tracked Turing
  calibration remains method evidence only.
- Both forms are the measured behavior of the exact frozen fixtures. Neither
  extrapolates to other shared-memory fractions, launch shapes, instruction
  mixes or GPU architectures. COMP-24 owns that extension.
- The live families demonstrate that CORE-4 preserves the ordered tuple the
  forms depend on and projects the result to TTFT and TPOT. They do not show
  that any production configuration selects the service. COMP-25 owns that.
- The 99.366034 us B100 roofline number is context for the motivating capture,
  not a result of this study. It omits kernel launch and scheduling entirely,
  which COMP-2 owns.

## Reproduce

```bash
python examples/mixed_makespan_v1/run_study.py --check-only
python examples/mixed_makespan_v1/run_study.py --out "$SIMLLM_MIXED_MAKESPAN_RUN_ROOT"
```

Set the run root in local configuration. The study writes `rows.csv`,
`raw_observations.json` and `summary.json` beneath that directory, about 60 KB
in total, and exits nonzero if any fatal guard or scored relation fails. The
focused regressions are in `tests/test_mixed_makespan.py`.
