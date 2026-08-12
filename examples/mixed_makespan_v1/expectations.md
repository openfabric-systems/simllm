# Corrected mixed-makespan v1 expectations

This is the expectations-only record for COMP-12. It freezes a replication of
the corrected G1 and G2 forms from the earlier GPU task-mix study, then carries
those forms through the live CORE-4 runtime and request-metric chain. The
earlier study discovered the forms before this registration. This study tests
whether the current implementation reproduces them; it does not present the
known rows as a new GPU discovery or as silicon calibration.

## Claim boundary and chronology

The trace-driven SM scheduler already contains the issue and residency
mechanisms. This change registers their measured forms and demonstrates that
CORE-4 preserves the order on which the G1 term depends. It does not add a
second mixed-task estimator, fit a broader scheduling law, or change the
`ComputeProvider` interface.

The source evidence was measured and published in
`examples/gpu_task_mix/RESULTS.md` at SimLLM commit
`0d9e2337eab6d5e49c112f3fbccb7d5e70a44f7f`. The present evidence is authored
against SimLLM commit `76223875557a552deb5aa2c2c529a07f000135ba`.

Before this freeze, a read-only exploratory probe imported the current runtime
and observed live values of 329,000 ps, 328,000 ps and 243,000 ps for the three
principal rows. It wrote no artifact and changed no file. Those observations
are prior exploratory evidence, not scored results. The registered production
run below is therefore a pre-registered replication with fully disclosed prior
knowledge, not an unseen prediction.

## Pre-freeze source and measurement audit

The corrected form is taken from measured rows, not from an assumed model of
how GPUs normally schedule:

- `examples/gpu_task_mix/RESULTS.md:159-184` reports the five G1 controls. A
  memory-first pair is one cycle slower than network-first; widening only the
  scheduler budget or only the load/store lanes retains the delay; widening
  both removes it.
- `examples/gpu_task_mix/RESULTS.md:186-221` reports the G2 controls. With each
  CTA claiming half an SM's shared memory, the constrained isolated durations
  are 14 and 229 cycles, the second task admits at cycle 14, and the concurrent
  result is their 243-cycle sum. Removing shared-memory demand restores
  backfill and produces 133 cycles from isolated durations 7 and 132 plus the
  one-cycle G1 term.
- `examples/gpu_task_mix/diagnostics.csv:2-11` retains the exact measured
  controls independently of the prose.
- `simllm/compute/gpu_model.py:942-1000` accounts for blocks, warps, threads,
  registers and shared memory at admission. This is the runtime mechanism that
  can make the second task wait for residency.
- `simllm/compute/gpu_model.py:1010-1078` orders issue candidates by task index,
  block and warp, then applies both the scheduler budget and pipeline-lane
  availability. This is the runtime mechanism that produces the G1 delay.
- `simllm/core/execution.py:244-276` preserves observed operation order and
  declares graph tuple order to be FIFO submission order within a logical
  queue.
- `simllm/core/runtime.py:1300-1419` selects a legal ready operation, forms a
  co-runnable compute group in graph order, and passes that ordered tuple to
  `SmSchedulerModel.estimate_concurrent`.

The audit identifies no measured basis for a closed form outside the exact
fixture and sweep below. In particular, this registration does not extrapolate
to arbitrary shared-memory fractions, launch shapes, instruction mixes or GPU
architectures.

## Frozen synthetic fixture

The component and live runtime rows use the earlier study's synthetic 1 GHz
profile. One cycle is exactly 1,000 ps. This profile is a deterministic
mechanism fixture, not a B100, H100 or Turing calibration.

| Property | Frozen value |
|---|---:|
| SMs | 2 |
| scheduler issue budget per SM | swept, 4 or 8 instructions per cycle |
| load/store lanes per SM | swept, 4 or 8 |
| HBM service | 64 bytes per cycle plus 100 cycles return latency |
| NVLink egress service | 16 bytes per cycle plus 200 cycles return latency |
| shared memory per SM | 65,536 bytes |
| task shape | 8 CTAs, one warp per CTA, four instructions per warp |

The memory task issues 32 HBM loads of 64 bytes. Its unconstrained isolated
duration is

```text
T_memory = 32 * ceil(64 / 64) + 100 = 132 cycles.
```

The network task issues 32 NVLink stores of 64 bytes. Its isolated duration is

```text
T_network = 32 * ceil(64 / 16) + 200 = 328 cycles.
```

## G1 registered issue-order form

For the exact task pair above, define the observed issue delay without
inventing a new scheduler:

```text
delta_issue = T_mixed - max(T_memory_isolated, T_network_isolated)
T_mixed = max(T_memory_isolated, T_network_isolated) + delta_issue.
```

The frozen replication matrix is:

| ordered task tuple | issue budget | load/store lanes | `delta_issue` | mixed cycles |
|---|---:|---:|---:|---:|
| memory, network | 4 | 4 | 1 | 329 |
| memory, network | 8 | 4 | 1 | 329 |
| memory, network | 4 | 8 | 1 | 329 |
| memory, network | 8 | 8 | 0 | 328 |
| network, memory | 4 | 4 | 0 | 328 |

The delay follows the actual ordered co-runnable tuple supplied by CORE-4. It
must not be reconstructed from `GpuTaskKind`, priority labels or a canonical
memory-before-network sort. When the graph supplies memory first under the
baseline identity policy, memory consumes the binding cycle-zero issue
resources and shifts the critical first NVLink store by one cycle. When the
graph supplies network first, the critical store wins those resources and the
delay is zero. Widening only one issue resource leaves the other binding;
widening both removes the delay.

Tasks separated by a dependency or eligibility time are not instances of this
concurrent form and receive no synthetic G1 addition. Future non-identity
policy work under CORE-10 must likewise pass the policy-selected legal order
to the compute service, after which the same measured form follows that actual
order. This study claims only the currently supported omitted and explicit
identity policies.

## G2 registered residency form

The constrained pair gives each CTA 32,768 bytes, exactly half the SM's shared
memory. Each SM can therefore hold two CTAs total across both tasks. The first
8-CTA task fills the four available slots across two SMs, including its later
wave, so the second task cannot admit until the first finishes.

The frozen constrained form is:

```text
T_compute_isolated = 14 cycles
T_memory_isolated = 229 cycles
A_memory_mixed = T_compute_isolated = 14 cycles
T_mixed = T_compute_isolated + T_memory_isolated = 243 cycles.
```

The admission equality is required alongside the sum. A 243-cycle makespan by
itself would not identify residency as the cause.

The zero-shared-memory control freezes the same launch instruction streams with
the residency currency removed:

```text
T_compute_isolated = 7 cycles
T_memory_isolated = 132 cycles
T_mixed = max(7, 132) + 1 = 133 cycles.
```

This control must restore backfill while retaining the independently registered
one-cycle G1 term.

## Live CORE-4 and metric projection

Each live case places the two kernel launches on rank 0 in separate logical
queues, correlates both operations to one request, and uses the ordered graph
tuple as the submission order. `CoarseDeviceRuntime` is supplied the same
`SmSchedulerModel` and launch records as the component replay. Its task
completion cycles must project through GPU queue visits, `CompletionEvent`,
`ExecutionResult`, `RuntimeReport`, `CompletionReducer`, `StepResult`, TTFT and
TPOT.

The live issue-order relation is evaluated from raw `StepResult.step_latency_ps`
values:

| issue budget | load/store lanes | memory-first minus network-first JCT |
|---:|---:|---:|
| 4 | 4 | +1,000 ps |
| 8 | 8 | 0 ps |

The live residency relation is also evaluated from raw step latencies:

```text
JCT_half_shared - JCT_zero_shared = 243,000 - 133,000 = +110,000 ps.
```

Every live configuration runs three identical steps with a private virtual
clock: one prefill followed by two decode steps. For this fixed-step fixture,
TTFT and each decode TPOT equal the step JCT. That equality demonstrates the
supported metric chain but is algebraically entailed by repeated equal steps,
so it is unscored reachability evidence.

The scalar compatibility path omits `kernel_services` and supplies nominal
durations of 132,000 ps and 328,000 ps. Both graph orders must retain the
historical independent-resource maximum of 328,000 ps, with the same TTFT and
TPOT. This off path does not claim the corrected mixed mechanism.

Omitted arbitration and explicit `IdentityArbitrationPolicy` must preserve all
timing and completion observations. Permuting priority labels under identity
arbitration must also preserve them. These are mandatory identity guards and
remain unscored.

## Physical sanity before precision comparison

The synthetic rows are bounded before any exact form is checked:

| Case | first-principles floor | conservative ceiling | expected location |
|---|---:|---:|---:|
| G1 baseline | `max(132, 328) = 328` cycles | `132 + 328 = 460` cycles | 329, one cycle above floor |
| G2 half shared | `max(14, 229) = 229` cycles | `14 + 229 = 243` cycles | 243, at serialized ceiling |
| G2 zero shared | `max(7, 132) = 132` cycles | `7 + 132 = 139` cycles | 133, one cycle above floor |

The lower bounds follow from work that cannot complete before its longer
isolated control. The ceilings serialize the two isolated controls completely.
Landing outside either interval voids the run before the exact relation is
interpreted.

The production compute context is kept separate from this synthetic profile.
Default callers select `GPU_ENVELOPES["b100"]`, whose HBM envelope is 8.0e12
bytes/s (`simllm/compute/transformer.py:35-42` and
`simllm/backends/step_sink.py:129-138`). For the motivating Granite decode
fixture, total roofline memory traffic is 556,449,792 bytes. The 100 percent
B100 bandwidth floor is 69.556224 us. At the configured 0.7 achievable
bandwidth, the read floor is

```text
556,449,792 / (8.0e12 * 0.7) = 99.366034285... us.
```

The integer roofline estimate is 99.366034 us, exactly at that configured
floor and 1.429 times the hardware-roof floor. A finite physical ceiling cannot
be derived without a measured minimum sustained bandwidth or timeout, so the
honest first-principles ceiling is unbounded. The result must say so instead of
inventing a tight upper limit. Labeling 99.366034 us as H100 would be invalid:
556,449,792 bytes alone need about 166.104 us at H100's 3.35e12 bytes/s peak.

`HostInitiationModel` defaults to `initiation_delay_ps=0` and profile `ideal`
(`simllm/compute/host.py:33-42`). The roofline result therefore omits kernel
launch, host scheduling and device scheduling overhead. That absolute-fidelity
gap belongs to calibration work, not to either mixed-makespan form registered
here.

Three independent sanity angles are retained in the result: synthetic
serialization and issue bounds, B100 memory traffic and bandwidth scaling, and
end-to-end plausibility after disclosing the zero launch overhead. The tracked
Turing calibration remains method evidence only and is never presented as a
B100 or H100 anchor.

## Evidence classes and entailment

The scored headline contains exactly four genuine-risk families:

1. G1 component issue-delay matrix, five parameterized instances;
2. G2 component residency and backfill form, three parameterized instances;
3. live CORE-4 issue-order projection, two parameterized instances;
4. live CORE-4 residency projection, one parameterized instance.

This gives four families and 11 instances. Each predicate is evaluated from
raw isolated, concurrent, admission or `StepResult` observations before any
exact fixture, identity, event, conservation or artifact guard. No earlier
fatal exact oracle pins a scored quantity. The component and live families are
not statistically independent: the latter deliberately tests projection of the
same scheduler through a different interface. They remain separate mechanism
and integration evidence, and no row count is added to a fatal or test count.

Run configurations, raw observations, physical intervals, identity checks,
exact configuration inventories, task counters, queue-visit semantics,
completion events, metric reachability, focused tests and repository gates are
separate evidence classes. Configuration-forced zeros, counter conservation,
repeated-step TTFT/TPOT equality, clean-worktree state and deterministic output
serialization are fatal or by-construction guards. They never increase the
behavioral denominator.

Any fatal-guard failure voids the complete run. A void run retains raw findings
but reports no behavioral fraction and cannot close COMP-12.

## Frozen fatal guards

After raw relations are evaluated, the runner must require:

- the exact synthetic profile and launch inventory above;
- all three physical intervals above;
- nonnegative component admission and completion timestamps;
- the expected task identities and CORE-4 graph order with no loss or
  duplication;
- `GpuTaskKind` relabeling changes attribution labels only;
- omitted and explicit identity policies preserve every timing and completion
  observation;
- priority-label permutation under identity preserves timing and completion;
- the scalar compatibility path is order-invariant at 328,000 ps;
- `CompletionEvent.QUEUED` equals resource eligibility and
  `CompletionEvent.STARTED` equals the resource grant;
- each operation has one logical completion event at its reported completion;
- component instruction and byte totals conserve exactly;
- each live `StepResult` and request metric conserves the runtime completion;
- the B100 envelope is named `b100` with 8.0e12 bytes/s and the roofline
  arithmetic above remains exact;
- the registered production run starts from a clean worktree and writes only
  beneath its explicit output directory.

## Registered command and pre-freeze dry run

The result-producing command is:

```text
.venv/bin/python examples/mixed_makespan_v1/run_study.py --out "$SIMLLM_MIXED_MAKESPAN_RUN_ROOT"
```

Before this expectations commit, that exact CLI is run with `--check-only`.
Check-only parses the production arguments and validates only the frozen
literal registry and arithmetic. It imports no SimLLM implementation, reads no
input or output path, invokes no native binary and creates no artifact. The
result records the final expectations-only commit and the observed SimLLM
revision separately.

## Closure scope

COMP-12 closes only if all four genuine-risk families pass, every fatal guard
passes, the live metric chain is present, and the registered forms and CORE-4
order rule are documented without a broader silicon claim. Any undiscovered
acceptance gap moves to COMP-24, COMP-25 or CORE-49 with the owning module and
required category tag before COMP-12 is removed.

COMP-17 is outside this run. The tracked Turing table contains aggregate
family, shape and dtype cells rather than per-layer identities or measured
per-layer durations, and COMP-6 still owns the required per-invocation shapes.
Consequently this change will not populate calibrated-provider layer estimates
or claim a B100 layer anchor. COMP-17 remains open unchanged unless qualifying
per-layer evidence becomes available before implementation begins.
