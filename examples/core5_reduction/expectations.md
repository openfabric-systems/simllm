# CORE-5 completion reduction expectations

## Freeze status and chronology

This file is the expectations-only record for CORE-5. It precedes the
completion reducer, the scheduler-result attribution types, task tests, every
result-producing run of this study, and `RESULTS.md`. The result report must
cite the final expectations-only commit and state the actual implementation
and run chronology.

The study will exercise the already-landed `CoarseDeviceRuntime`; it will not
create a second lifecycle or timing authority. The runtime's immutable event
stream, operation records and queue visits are inputs to one scheduler-facing
reduction. A reducer failure must leave its request history and `VirtualClock`
unchanged.

## Source audit before freeze

No scored expectation below mirrors a vLLM or SGLang implementation, an
external NCCL release, or a hardware timing specification. Fixture rates and
service times are ideal inputs. The source surfaces that define this study
were audited at base commit `fc282ef` before this freeze:

- `QueueVisit` fixes submitted, eligible, started, finished and completed
  timestamps and the three interval differences at
  `simllm/core/runtime.py:193-243`.
- The landed operation reduction clips every visit against the realized
  predecessor boundary, checks each operation segment, and checks graph-chain
  conservation at `simllm/core/runtime.py:1913-2055`. This reduction is the
  input being classified, not an independently advanced timing model.
- The runtime emits submitted, queued, started, progress and completed phases,
  including WQE subject identity, at `simllm/core/runtime.py:1813-1911`.
- `ExecutionGraph.completion_operation_ids` selects the framework boundary,
  while `ExecutionResult` keeps framework completion distinct from physical
  quiescence at `simllm/core/execution.py:233-250,279-293`.
- The existing scheduler boundary has only step index, step latency and
  completion time at `simllm/core/step.py:88-96`; the clock's monotonic
  absorption operation is `VirtualClock.advance_to` at
  `simllm/core/clock.py:42-46`.
- The existing replay convention takes the first completed step as TTFT and
  the mean of later token intervals as TPOT at
  `examples/m4/run_m4.py:225-244`.
- vLLM's translated step knows the per-request token-production mask, but
  `StepRecord` currently retains only its count at
  `simllm/adapters/vllm/executor.py:420-474`. The exact mixed-batch identity
  limitation is therefore recorded explicitly below rather than guessed.

These are repository-native contract referents. Exact event membership and
call order are fatal structural guards, not scored behavior. The scored
relations are closed-form latency changes that a plausible incorrect
reduction can fail.

## Scheduler result and request metric contract

`StepResult.step_latency_ps` is exactly

```text
ExecutionResult.completed_at_ps - ExecutionGraph.released_at_ps.
```

The result carries the request metrics completed in that scheduler step. For
one request, the first sampled boundary defines

```text
TTFT = first_token_completed_at - first_observed_virtual_time.
```

Every later sampled boundary contributes one inter-token interval. TPOT is
the exact rational mean of all completed inter-token intervals, with no
rounding in the core record. A request delayed across a non-sampling prefill
chunk or an intervening scheduler step retains that elapsed time in its next
reported interval.

Each reported token interval has exactly these critical-path components:

- `queue_ps`: realized queue waits, causal eligibility gaps, and scheduler
  boundary wait that lies inside this request interval;
- `kv_ps`: selected KV lifecycle or KV-resource service;
- `kernel_ps`: selected GPU kernel service and visibility;
- `dma_ps`: selected copy-engine or DMA service and visibility;
- `collective_ps`: selected NCCL-channel or NVLink service and visibility;
- `nic_ps`: selected RNIC, send, receive or completion service and visibility;
- `control_ps`: selected host-launch or local control service and visibility.

The fatal conservation identity for every reported request interval is

```text
latency_ps = queue_ps + kv_ps + kernel_ps + dma_ps
           + collective_ps + nic_ps + control_ps.
```

Only operation segments on the request endpoint's realized critical
predecessor chain enter this identity. Shared operations may project into more
than one request, but their authority and timestamps remain the runtime's.

Additive visit accounting is a different type with `queue_wait_ps`,
`service_ps`, `visibility_ps` and `visit_count`. It sums all relevant physical
visits, including overlapping visits and noncritical branches. Neither an
additive field nor its sum may enter the conservation identity above.
Graph-wide additive totals cover every runtime visit once. A per-request
projection covers visits on operations correlated with that request; shared
work may therefore appear in more than one request projection and request
projections must not be summed as graph work.

Legacy `StepRecord` readers remain valid. An optional exact
`sampled_request_ids` field may identify a subset and must agree with
`num_sampled`. If the field is absent, zero samples and all-scheduled samples
are unambiguous. An absent legacy `num_sampled` deliberately retains the
documented approximation that every scheduled request samples. A partial
exact count without identities must fail rather than assign TTFT to an
arbitrary prefill request. CORE-17 will carry exact sampled-request identity
from framework adapters for that mixed partial-sampling case.

## Fixed two-request fixture

All values are integer picoseconds. Start the clock and the first graph at
`T0 = 7,000 ps`. Each cell runs three otherwise identical scheduler steps:
one prefill step followed by two decode steps. Both `request-0` and
`request-1` are explicitly sampled in every step. Each request uses separate
GPU, NVLink and GPU-affine RNIC resources, so the two request chains overlap
exactly and finish together.

For each request, use these ideal demands:

- zero-cost KV observation `K = 0 ps`;
- kernel service `C = 20,000 ps`;
- DMA service `M = 10,000 ps`;
- two-rank intra-node ring service `A = 8,000 ps`;
- local control service `H = 1,000 ps`;
- two 4 KiB synchronous control sends on the same RNIC.

The ring has 100 bytes at 100 Gbit/s. Two rounds each serialize a 50-byte
chunk, so one participant's realized ring path is exactly `A = 8,000 ps`.
The RNIC term at link rate R in Gbit/s is

```text
N(R) = 4096 * 8 * 1000 / R ps.
```

Therefore `N(200) = 163,840 ps` and `N(400) = 81,920 ps`.
The two control operations become legal together. The first occupies the
RNIC while the second overlaps its local `H` service, so the selected second
control segment is exactly `queue=N(R)`, `control=H`, `nic=N(R)`.

Sweep dependency shape over:

- parallel: kernel and DMA are independent, then the collective waits for
  both;
- serial: DMA waits for the kernel, then the collective waits for DMA.

Sweep RNIC rate R over `{200, 400}` Gbit/s. The exact one-step request and
graph latency is

```text
J(parallel, R) = max(C, M) + A + H + 2 * N(R)
J(serial, R)   = C + M + A + H + 2 * N(R).
```

The exact oracle is:

| Shape | R (Gbit/s) | J (ps) | queue | KV | kernel | DMA | collective | NIC | control |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| parallel | 200 | 356,680 | 163,840 | 0 | 20,000 | 0 | 8,000 | 163,840 | 1,000 |
| serial | 200 | 366,680 | 163,840 | 0 | 20,000 | 10,000 | 8,000 | 163,840 | 1,000 |
| parallel | 400 | 192,840 | 81,920 | 0 | 20,000 | 0 | 8,000 | 81,920 | 1,000 |
| serial | 400 | 202,840 | 81,920 | 0 | 20,000 | 10,000 | 8,000 | 81,920 | 1,000 |

For both requests and every cell:

```text
J(serial, R) - J(parallel, R) = +10,000 ps
J(shape, 200) - J(shape, 400) = +163,840 ps
StepResult.completed_at_ps(step i) = T0 + (i + 1) * J
TTFT = J
each inter-token interval = J
TPOT after either decode = J exactly
final VirtualClock.now_ps = T0 + 3 * J.
```

The signed directions and bands are exact, so every residual band is 0 ps.
If the dependency penalty is absent, the reducer lost a realized predecessor
segment. If the rate delta is not exact, it either dropped or double-counted a
NIC tail. Either failure rejects this reduction as a TTFT or TPOT source.

## Additive totals for the fixed fixture

Each request owns 21 visits in one step. Its additive visit totals are

```text
queue_wait_ps = N(R)
service_ps = C + 2 * M + 2 * A + 2 * H + 2 * N(R)
visibility_ps = 0.
```

The graph-wide totals are exactly twice those values and contain 42 visits.
They are unchanged by dependency shape. The per-request additive total is
`549,520 ps` at 200 Gbit/s and `303,760 ps` at 400 Gbit/s, both larger than
the corresponding realized request latency. The graph-wide additive queue
wait is `2 * N(R)`, while either realized request's queue component is only
`N(R)`. These nonzero inequalities are required guards against adding work
totals into TTFT or TPOT.

## Event stream and asynchronous progress

For every runtime visit, queued time equals eligibility and started time
equals the grant. Submitted, queued, started and progress events are streamed
in nondecreasing timestamp order. Every operation has one logical completed
event, and every WQE has a subject-specific completed event. The callback
stream and `ExecutionResult.events` must contain the same immutable event
objects in the same order. Event sequence equality is fatal and unscored.

Use a `P = 10,000 ps` required compute anchor and a background duration
`G = 20,971,520 ps` at 400 Gbit/s for two progress checks:

1. Required asynchronous control completes logically at local handoff while
   its WQE drains to G. Changing only that control to synchronous moves the
   required boundary from P to G, an exact signed change of
   `+20,961,520 ps`.
2. A background collective excluded from `completion_operation_ids` drains to
   G while the anchor releases the framework at P. Adding the collective to
   the required set moves the boundary from P to G by the same exact amount.

In either asynchronous cell, `StepResult.step_latency_ps = P`,
`VirtualClock.now_ps = T0 + P`, and
`ExecutionResult.quiesced_at_ps = T0 + G`. Events and additive visits after
the framework boundary remain observable, but no future timestamp may advance
the scheduler clock. In the synchronous cells, both result completion and
clock advance to `T0 + G`.

Decision consequence: if asynchronous quiescence advances the scheduler
clock, control or collective overlap is not representable and the reduction
cannot be used by a closed-loop scheduler. If synchronous completion stops at
P, required work was incorrectly treated as background.

## Fatal unscored guards

The following fail the study without increasing a behavioral denominator:

- record, graph, execution result and runtime report agree on step and
  execution identity;
- graph release equals both `StepRecord.virtual_time_ps` and the pre-reduction
  clock value;
- every scheduled request has a correlated required endpoint;
- every request critical chain is acyclic and its operation segments conserve
  endpoint latency before cross-step accumulation;
- all attribution and additive fields are nonnegative;
- explicit sampled IDs are unique, scheduled, and agree with `num_sampled`;
- a failed identity, event, conservation or sampling check changes neither
  request history nor clock;
- legacy records without the new optional field still load and round-trip;
- the execution and completion event schema identifiers remain v1;
- author-defined event order, zero KV cost in this CORE-3-off fixture, and
  exact configuration echoes remain unscored.

## Evidence accounting and genuine risk

The result report keeps run configurations, exact-oracle rows, scored
behavioral relations, fatal invariants, unit tests and the repository gates in
separate sections. Scored families are:

- dependency-shape penalty, one instance per RNIC rate;
- inverse-rate tail change, one instance per dependency shape;
- per-request TTFT, TPOT and component conservation, one instance per request
  and cell;
- additive-work separation, one instance per rate and shape;
- asynchronous control and collective boundary movement, one instance each.

Conservation itself is fatal and unscored, but the exact component rows are
scored because the same J can be produced with a lost DMA branch, a duplicated
NIC tail or an additive wait substituted for a selected wait. `RESULTS.md`
must report the observed genuine-risk fraction for each family and explain
why every numerator member could fail independently of fixture construction.

## Registered command and pre-freeze dry run

The result-producing command is:

```bash
.venv/bin/python examples/core5_reduction/run_study.py \
  --out "$SIMLLM_CORE5_RUN_ROOT"
```

Before the expectations commit, the same command must be run in its
non-result-producing mode:

```bash
.venv/bin/python examples/core5_reduction/run_study.py \
  --out "$SIMLLM_CORE5_RUN_ROOT" \
  --check-only
```

`SIMLLM_CORE5_RUN_ROOT` is configured in the gitignored local environment to
the branch's required wave-3 bulk-output directory. `--check-only` parses the
complete CLI, validates every frozen literal and source-audit input, prints a
registry confirmation by design, and creates no directory, result or measured
artifact. At freeze time the untracked command harness may contain only these
frozen literals and check-only validation. It must not import the not-yet-
implemented reduction API.
