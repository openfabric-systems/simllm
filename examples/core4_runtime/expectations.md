# CORE-4 coarse device runtime expectations

## Freeze and provenance

This file is the expectations-only record for the first coarse
`DeviceRuntime`. It is written before any CORE-4 implementation file, test,
study runner, or measured result exists in this worktree. The four experiment
families were already frozen in `docs/modules/core.md`: dependency overlap at
commit `ea3961b`, then the GPU-affine RNIC, tail-attribution, and identity-policy
forms at commit `37357cc`. This refinement fixes concrete workloads, exact
units, evidence classes, and decision consequences without weakening those
older relations.

The freeze-stage command registered for this file is:

```bash
.venv/bin/python -c "from pathlib import Path; p=Path('examples/core4_runtime/expectations.md'); s=p.read_text(); assert all(x in s for x in ('max(C, D)', '8 * B / R', 'CORE-5', 'Decision consequence')); assert not Path('examples/core4_runtime/run_study.py').exists()"
```

It must be run from the repository root before the freeze commit. It is a
parse-only audit of the frozen relation anchors and the absence of a study
implementation. It creates no result or generated artifact.

## Source audit before freeze

No scored expectation below mirrors vLLM, an external NCCL release, or a GPU
or RNIC hardware specification. Rates and service times are ideal fixture
inputs, not claims about silicon. The fixed eight-GPU profile comes from the
repository contract in `docs/modules/core.md`, not from an external hardware
source.

Two fatal structural comparisons do use repository-native frozen referents:

- Ring expansion uses `2 * (W - 1)` rounds, one `payload_bytes // W` chunk per
  rank and round, and consecutive round tags, as frozen at base commit
  `6aa3a76` in `simllm/traffic/patterns.py:67-100` and
  `simllm/traffic/execution_goal.py:146-194,239-253`.
- Pairwise all-to-all expansion emits one uniform ordered-pair send and uses
  the operation tag frozen at base commit `6aa3a76` in
  `simllm/traffic/patterns.py:103-127` and
  `simllm/traffic/execution_goal.py:193-194,254-274`.

These exact expansion and tag comparisons are fatal and unscored. They guard
operation identity through NCCL command, semantic WQE, completion event, and
rendered GOAL tag. They are not behavioral evidence. Dispatch into the
concurrent kernel service is likewise a structural integration guard against
the frozen `SmSchedulerModel.estimate_concurrent` entry point at
`simllm/compute/gpu_model.py:789-831`; an author-defined call sequence is not
scored.

## Fixed ideal profile

All times are integer picoseconds. Every graph is released at
`T0 = 5,000 ps`, which detects a hardcoded zero origin. Unless a cell says
otherwise, the profile has:

- eight GPUs per node and one GPU-affine RNIC or QP per GPU;
- zero launch, protocol, propagation, and completion-delivery overhead;
- one identity arbitration policy, with omission selecting the same baseline;
- one independent directional copy engine per GPU, priced by the
  `simllm.compute` copy service;
- 400 Gbit/s per RNIC, except in the registered rate sweep;
- a timing-neutral bypass WQE authority named `AtlahsWqeLedger`;
- no random mechanism, so the random-draw count is exactly zero.

The device runtime must validate an `ExecutionGraph`, preserve its operation
identities, and return an `ExecutionResult`. Define graph JCT as

```text
JCT = ExecutionResult.completed_at_ps - ExecutionGraph.released_at_ps.
```

`ExecutionResult.quiesced_at_ps` is the last physical completion. It may
exceed framework-visible completion when control delivery is asynchronous.

## A. Dependency versus legal overlap

Each graph contains one compute operation of duration `C` and one DMA
operation of duration `D` on different logical queues and independent ideal
resources. The compute operation declares zero HBM demand in this family, so
the shared HBM arbiter is intentionally inactive. Sweep:

- `(C, D) = (10,000,000, 40,000,000) ps`;
- `(C, D) = (80,000,000, 40,000,000) ps`;
- no dependency, or DMA depends on compute completion.

The exact graph-to-result relations are:

```text
JCT(no edge) = max(C, D)
JCT(edge) = C + D
delta(edge minus no edge) = min(C, D) > 0
completed_at_ps = T0 + JCT
```

The four JCTs are therefore `40,000,000`, `50,000,000`, `80,000,000`, and
`120,000,000 ps`, with signed edge penalties of `10,000,000` and
`40,000,000 ps`. Every value must match exactly.

Decision consequence: if either no-edge cell is additive instead of the
registered maximum, the design decision to retain distinct, concurrently
progressing GPU and copy-engine lanes is rejected. CORE-4 must then remain
open while the runtime resource decomposition is replaced. If an edge cell
does not add exactly the smaller demand, dependency release rather than
resource separation is the rejected design.

This is the required end-to-end metric-changing relation for CORE-4. It ends
at `ExecutionResult` JCT. CORE-5 is the named successor that reduces this
event stream through `StepResult` into TTFT and TPOT. CORE-4 makes no claim to
that later reduction.

### Shared-HBM guard

A separate fatal, unscored guard repeats the no-edge cells after changing
only the compute HBM demand from zero to nonzero. The first coarse arbiter is
exclusive at whole-operation granularity, so the expected JCT is `C + D`.
This is a structural check that kernel and DMA work share one HBM authority;
it does not add to family A's behavioral denominator.

## B. Eight GPU-affine RNICs

Use `B = 1,048,576 bytes`. Each active source GPU `g` in node 0 submits two
same-QP fabric control messages, in FIFO order, to GPU `g` in node 1. The
two-message phase makes each source serialize exactly `2 * B` bytes while
different source RNICs overlap. Sweep active GPU count `N` in `{1, 8}` and
per-port rate `R` in `{200, 400}` Gbit/s.

For one WQE:

```text
S(B, R) = 8,000 * B / R ps.
```

For the registered two-WQE phase:

```text
JCT(N, R) = 2 * S(B, R) = 16,000 * B / R ps,
useful throughput = N * 2 * B * 8 / JCT = N * R,
JCT(8, R) = JCT(1, R),
JCT(N, 200) = 2 * JCT(N, 400).
```

Exact JCT is `83,886,080 ps` at 200 Gbit/s and `41,943,040 ps` at
400 Gbit/s, independent of `N`. Each QP must complete its first WQE before its
second at both rates. GPU `g` must select only RNIC `g` in its node. A global
single-RNIC cursor would multiply the `N=8` makespan by eight and fail this
family.

The older module wording states the one-WQE phase as `8 * B / R` seconds.
This study concatenates two such phases on every QP to make FIFO ordering
observable; each individual phase and each WQE retains the original exact
formula.

Decision consequence: if JCT grows with `N`, the fixed one-RNIC-per-GPU
mapping is not an independent-rail model and must be replaced before it can
serve as the first profile. If doubling `R` does not halve JCT, the ideal
serialization law is rejected and no physical-RNIC calibration may use this
runtime baseline.

## C. Tail attribution and asynchronous control

The control workload contains one independent `10,000,000 ps` compute anchor
and, on each of eight GPUs, the two `B`-byte control messages from family B at
400 Gbit/s. Sweep `ControlMode` over synchronous and asynchronous and permute
the operation priority or class label between `3` and `9` under identity.

For synchronous control, delivery is part of logical completion:

```text
JCT(sync) = 2 * S(B, 400) = 41,943,040 ps.
```

For asynchronous control, logical completion occurs after the zero-cost
control-queue handoff while physical WQEs continue:

```text
JCT(async) = 10,000,000 ps,
quiescence(async) - T0 = 41,943,040 ps.
```

Thus changing asynchronous to synchronous has the signed JCT effect
`+31,943,040 ps` in both class-label cells. The compute anchor and every
dependency-independent timestamp remain identical. Only the control
operation's completion boundary and work reachable from that boundary may
move.

For every logically completed root operation, measure latency from its first
resource eligibility to logical completion. Its selected critical path must
partition exactly into launch-queue wait and service, device-queue wait,
selected service, and downstream completion delivery. The integer sum equals
that operation latency exactly. External dependency delay, when exercised by
unit tests, is reported separately and is never relabeled as queue wait.

Every queue visit obeys:

```text
queue_wait_ps = started_at_ps - eligible_at_ps
service_ps = finished_at_ps - started_at_ps
visibility_ps = completed_at_ps - finished_at_ps
```

All intervals are nonnegative. `CompletionEvent.QUEUED` equals eligibility and
`CompletionEvent.STARTED` equals grant. Graph completion equals the latest
required logical completion, while quiescence equals the latest physical
completion.

The second WQE on each of eight RNICs waits exactly `S(B, 400)`, so the
additive visit-wait total is `8 * S(B, 400) = 167,772,160 ps`. This exceeds
the `41,943,040 ps` synchronous graph JCT by exactly four times. It must be
reported as additive work and must not enter the critical-path identity.

Decision consequence: if the additive visit sum is used as JCT delay, the
single-reduction design is rejected and the diagnostics must expose distinct
work-sum and critical-path types before CORE-5 can consume them.

## D. Identity arbitration is the bypass baseline

Use one deterministic mixed graph containing compute, DMA, synchronous
control, and a two-rank ring collective. Run the following four policy cells:

1. arbitration argument omitted, labels in baseline order;
2. explicit identity policy, labels in baseline order;
3. arbitration argument omitted, labels permuted;
4. explicit identity policy, labels permuted.

Mandatory dependencies, stream FIFO, channel FIFO, QP FIFO, and directional
engine legality are applied before the policy seam. Canonical outcome bytes
contain event order, event timestamps, queue-visit timestamps, wait totals,
service-byte counters, WQE identities and ordering, GOAL tags, random-draw
count, JCT, and quiescence. They exclude the input class-label echo itself.
All four canonical byte strings must be identical. This is the
bypass-preserves-baseline check for the CORE-8 policy seam.

Decision consequence: any difference rejects identity as the feature-off
policy. No non-identity policy from CORE-10 may be admitted until omitted and
explicit identity are exact aliases under label permutation.

## Fatal structural and authority guards

The following fail the study but never increase a scored denominator:

- exactly eight GPU slots and eight GPU-affine RNIC slots exist per node;
- a rank maps to `(node=rank // 8, gpu=rank % 8)`;
- bypass constructs and mutates exactly one `AtlahsWqeLedger`, while structural
  mode constructs none and delegates every WQE lifecycle through the native
  session seam reserved for HTSIM-9;
- bypass and structural authorities cannot both be supplied;
- every collective creates one NCCL command projection and every cross-node
  semantic send creates exactly one WQE projection with one SQ, RQ, CQ, and
  QP/link-pair compatibility context;
- WQE completion-event identity, timestamp, byte count, operation identity,
  and rendered GOAL tag agree with the authoritative WQE record;
- intra-node traffic selects an NVLink-class resource and creates no RNIC WQE;
- a mapped trace kernel dispatch calls the `simllm.compute` concurrent service
  once with the complete simultaneously legal task set;
- a DMA descriptor is priced only by the selected
  `simllm.compute.CopyEngineServiceModel` and cannot use an unsupported
  direction;
- bookkeeping events are the same immutable event objects returned in the
  `ExecutionResult`, with no loss or duplication;
- all event timestamps are monotonic in the returned stream and no event
  exceeds quiescence.

Exact call or event sequences are structural unless their individual elements
are tied to the frozen repository referents in the source audit. Configuration
echoes, zero random draws, zero ideal overheads, inactive HBM fields, and
conservation identities are also unscored.

## Evidence accounting

The result report will keep these classes separate:

- run configurations;
- exact-oracle JCT rows;
- scored behavioral relation families and parameterized instances;
- fatal structural and authority guards;
- unit-test executables;
- the existing full repository regression suite.

The scored families are A's dependency effect, B's rail independence and rate
scaling, C's synchronous completion effect and additive-wait separation, and
D's identity equivalence. A competent implementation can plausibly fail each:
accidental graph serialization can fail A, a node-global NIC cursor can fail
B, mixing physical quiescence with logical completion or visit sums can fail
C, and consulting priority in the off path can fail D. `RESULTS.md` must state
the observed genuine-risk fraction per family and explain why each counted
relation was not guaranteed merely by constructing the fixture.

## Integration-review amendment before corrective implementation

This appendix records the corrections requested by the integration review
after the first CORE-4 implementation. It is intentionally later than the
original expectations-only commit `d43cddb`, which remains unchanged in Git
history, and must precede every corrective implementation edit and corrected
study run. The original family-C JCT, quiescence, synchronous penalty, and
additive-wait literals remain unchanged.

The registered amendment dry run is:

```bash
.venv/bin/python examples/core4_runtime/run_study.py --check-only
```

It must be executed before the amendment commit. It parses the registered
study inputs and produces no measured result or generated artifact. These
review corrections mirror no external runtime, NCCL release, or hardware
specification, so there is no new external-source referent to audit.

### Transactional structural authority

Structural mode stages semantic submissions in a transaction supplied by the
sole `NativeRnicSession`. Scheduling and every projection check complete
before that transaction commits. In the regression fixture, the second
submission returns an invalid projection after one valid submission has been
staged. The execution must fail, the session's committed submission list and
sequence counter must remain byte-identical to their pre-execution values,
and retrying the same graph with the fault removed must commit exactly two
WQEs with sequences `[1, 2]`. A result of `[2, 3]`, duplicate records, or any
committed record after the failed attempt is fatal and unscored.

This remains component evidence until CORE-15 connects the transaction seam
to the HTSIM-9 composed native session and changes an `ExecutionGraph`
completion time.

### Realized critical-path clipping

Add two dependency-chain cells with framework-launch service `L` in
`{10, 20} ps`. At release zero, operation A is a `100 ps` compute. A zero-cost
filler on another GPU is second in graph order and has `not_before_ps =
100 + L`. Operation B is third in graph order, depends on A, and is a `20 ps`
compute. Only B is a required completion. All three launches use the same
node framework-launch FIFO.

The filler occupies the launch service immediately after A completes. B's
launch was submitted at release, but only the final `L ps` of its much larger
recorded queue-wait interval occurs after A's completion and lies on the
realized dependency path. The exact rows are:

| L (ps) | JCT (ps) | realized critical-path queue (ps) |
|---:|---:|---:|
| 10 | 150 | 10 |
| 20 | 180 | 20 |

For each cell, summing `operation_latency_ps` over the reported A to B
critical chain must equal graph JCT exactly. Each operation breakdown must
also conserve its own clipped chain segment. Launch wait or service completed
before the preceding chain boundary contributes zero. The uncorrected queue
totals of `120` and `140 ps` are additive visit work, not critical-path wait.

The existing family-C fixed cells additionally require
`critical_path_queue_ps = 0` for asynchronous control and
`critical_path_queue_ps = 20,971,520` for synchronous control, for both class
labels. These four values are scored because an implementation can plausibly
select physical quiescence or sum all eight RNIC waits instead.

Decision consequence: if clipping cannot make both chain conservation and
the exact queue literals hold, `critical_path_queue_ps` cannot feed CORE-5.
The chain-summed representation must then be replaced by an interval-union or
explicit predecessor-edge representation before TTFT or TPOT attribution.

### Destination arrivals and fail-closed inputs

An asynchronous control send from rank 0 to rank 8 publishes rank 0 at local
handoff and rank 8 at transfer completion. A rank-8 operation with a
participant-local dependency starts no earlier than that destination
timestamp. A peer-to-peer DMA from `gpu:0` to `gpu:1` likewise publishes both
rank arrivals, and a rank-1 participant-local successor starts no earlier than
DMA completion. Source-only publication is fatal and unscored.

Until CORE-3 supplies byte-carrying KV semantics, a READ or WRITE with
`byte_count > 0` must fail during preflight with an error naming CORE-3. Zero
byte lifecycle observations remain timing-neutral. Ring all-reduce accepts
only a positive payload evenly divisible among its ranks; zero, sub-rank, and
non-divisible payloads fail preflight without mutating runtime or RNIC state.
CORE-16 owns exact remainder chunking and wider control-tag allocation.

At most 1,024 destinations fit one control operation's reserved GOAL-tag
block. The block begins at `goal_base_tag + 1,000,000 + operation_index *
1,024`; collective tags must stay below the `goal_base_tag + 1,000,000`
boundary. Either overflow fails preflight before authority mutation.

### Replaceable arbitration policy

Any object satisfying `ArbitrationPolicy` is accepted. A recording policy
must observe the complete legal ready set and may choose a member of that set;
choosing any other record fails before authority commit. The shipped identity
policy continues to preserve the byte-exact class-permutation baseline.
Mandatory launch and logical-queue FIFO, protocol ordering, deterministic
copy-engine routing, all-member co-runnable compute dispatch, and native RNIC
legality remain policy-free and must be documented as such because they are
not optional arbitration points.
