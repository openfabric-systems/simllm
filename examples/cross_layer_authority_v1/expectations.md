# Cross-layer authority v1 expectations

This expectations-only record freezes the first enforced cross-layer authority
for CORE-8, before its implementation and before any run of its study.

`docs/modules/core.md` already declares the rule: the request bookkeeper,
backend rows and completion stream are projections of the runtime authority,
and `CompletionEvent.QUEUED` projects eligibility while `CompletionEvent.STARTED`
projects the resource grant. Nothing enforces it. `CompletionReducer` today
checks exactly one property of the event stream, that each operation carries a
unique logical `COMPLETED` event at the operation record's completion
(`simllm/core/completion.py:413-425`); every other event is accepted on trust.
The bookkeeping ledger is never joined to the runtime report at all: the
reducer reads only framework request arrivals from it
(`simllm/core/completion.py:172-179`).

This freeze registers the exact projection that closes those two gaps and the
contradictions that must be rejected once it exists.

## Duplicated-quantity inventory this freeze acts on

The full inventory belongs in the result report. The three entries this study
brings under one authority are:

1. **Queue-visit timestamps.** `QueueVisit` (`simllm/core/runtime.py:319-372`)
   and the four phase events rendered from it
   (`simllm/core/runtime.py:2713-2733`) are two representations of the same
   five-point visit. Unchecked.
2. **WQE lifecycle facts.** `WqeLifecycleProjection`
   (`simllm/core/runtime.py:518-553`), the WQE `QueueVisit`s derived from it
   (`simllm/core/runtime.py:2603-2652`), the five subject-keyed events
   (`simllm/core/runtime.py:2735-2793`) and the bookkeeping
   `CreatedObjectRecord` with its `bytes`, `goal_tag`, rank and sequence
   metadata (`simllm/core/runtime.py:3328-3350`) are four representations of
   one WQE. `simllm/core/bookkeeping.py:488-514` and `:564-583` check only WQE
   structure, never the copied bytes, tags or timestamps.
3. **Operation-level completion bytes and the class byte reduction.**
   `_work_completed_bytes` (`simllm/core/runtime.py:2682-2697`) feeds the
   logical `COMPLETED` event's `completed_bytes`
   (`simllm/core/runtime.py:2807`), and `class_service_bytes`
   (`simllm/core/runtime.py:3092-3096`) is a second reduction over the same
   graph work. Neither is reconciled by any consumer.

## Pre-freeze source audit and disclosed diagnostics

The evidence is authored against SimLLM revision
`4e1be35af5327c27db53ed002dc420e1de6f613b`. Three read-only diagnostics ran
before this freeze and are disclosed here rather than presented as results.

1. A probe wrapped `CoarseDeviceRuntime.execute` and measured every candidate
   clause below on every triple the existing test suite produces: 719 reports,
   1,663 operations, 8,989 queue visits, 2,777 WQEs and 38,540 completion
   events. Every clause held on all of them, with zero violations. This is why
   the clauses are stated as exact laws rather than as bounds, and it means the
   shapes the suite already covers are not novel evidence.
2. A second probe wrapped `CompletionReducer.reduce` instead, so it also saw
   the 593 hand-built graph, result and report triples the tests feed the
   reducer directly rather than through the coarse runtime. Every clause held
   on all 593. Enforcing the clauses at the reducer therefore changes no
   accepted run.
3. A probe built the fixture below and submitted the thirteen registered
   contradictions to the checking surfaces that exist today. All thirteen were
   accepted, which is the concrete form of the gap this task closes.

One earlier guess was refuted by diagnostic 1 and is recorded because it
changed a clause: `class_service_bytes` was assumed to reduce WQE payload bytes
by class, which failed on 116 reports. It reduces control-work bytes only, and
clause A6 below states the rule the source actually implements.

## The exact projection

Let `report` be the `RuntimeReport` and `result` the `ExecutionResult` of one
execution of `graph`. Write `visits` for `report.visits` and `wqes` for
`report.wqes`. The join key of the event projection is the stable tuple
`(operation_id, subject_object_id, phase, resource)`; the join key of the
bookkeeping projection is `wqe_id`.

### Authority 1: the completion-event stream projects the visit authority

Define the expected event multiset `E(graph, report)` as the union of:

1. **Subjectless visits.** For every `v` in `visits` with
   `v.subject_object_id is None`, exactly four events on
   `(v.operation_id, None, phase, v.resource)`:
   `SUBMITTED` at `v.submitted_at_ps` with no byte count, `QUEUED` at
   `v.eligible_at_ps` with no byte count, `STARTED` at `v.started_at_ps` with
   no byte count, and `PROGRESS` at `v.finished_at_ps` carrying exactly
   `v.service_bytes`.
2. **WQE objects.** For every `w` in `wqes`, exactly five events on
   `(w.operation_id, w.wqe_id, phase, resource)`: `SUBMITTED` at
   `w.submitted_at_ps` on `NIC_SEND_QUEUE w.sq_id`; `QUEUED` at
   `w.network_eligible_at_ps` when present and `w.eligible_at_ps` otherwise;
   `STARTED` at `w.network_started_at_ps` when present and `w.started_at_ps`
   otherwise; `PROGRESS` at `w.network_finished_at_ps` when present and
   `w.finished_at_ps` otherwise, carrying `w.payload_bytes`; all three on
   `NIC w.rnic_id`; and `COMPLETED` at `w.completed_at_ps` on
   `COMPLETION_QUEUE w.cq_id`, carrying `w.payload_bytes`.
3. **Operation logical completions.** For every operation of `graph`, exactly
   one event on `(operation_id, None, COMPLETED)` at the operation record's
   `completed_at_ps`, carrying the graph's declared semantic work bytes:
   `hbm_bytes` for compute, `byte_count` for DMA and KV-cache work, the sparse
   pair total or the scalar payload for a collective, `payload_bytes` for
   control work, and no byte count for any other payload. Its resource must be
   the resource of one of that operation's own visits whose `completed_at_ps`
   equals the event timestamp.

Then:

- **A1 exactness.** `result.events` as a multiset equals `E(graph, report)`.
  An expected tuple with no event is loss; an event tuple appearing more often
  than expected is duplication; an event whose only difference is its
  timestamp is timestamp disagreement. All three are rejections.
- **A5 subject closure.** The set of `subject_object_id` values carried by
  `visits` equals `{w.wqe_id for w in wqes}`, so no runtime object may appear
  in the visit ledger without a WQE authority record and no WQE may be absent
  from it.
- **A6 class reduction.** `report.class_service_bytes` equals the per-priority
  sum of `payload_bytes * max(1, len(destination_ranks))` over the graph's
  control-work operations, sorted by class label.

Clause 3 fixes an ownership question the inventory raises: the operation-level
`completed_bytes` is a projection of the graph's declared semantic payload, and
the `PROGRESS` and WQE byte counts are projections of physical service. For a
ring all-reduce they differ by construction. They keep distinct names, are
never summed, and each is checked against its own authority.

### Authority 2: the request bookkeeper projects the same runtime authority

For a ledger the runtime appended for `graph`:

- **B1 WQE objects.** For every `w` in `wqes` there is exactly one
  `NETWORK_WQE` `CreatedObjectRecord` with `object_id == w.wqe_id`,
  `native_id == w.native_wqe_id`, `created_at_ps == w.submitted_at_ps`, and
  metadata `bytes == w.payload_bytes`, `goal_tag == w.goal_tag`,
  `source_rank`, `destination_rank`, `sq_post_sequence`, `cq_post_sequence`,
  `channel == w.channel_id` and `graph_operation_id == w.operation_id` all
  equal to the projection's fields.
- **B1b no invention.** No `NETWORK_WQE` object scoped to this execution exists
  that `wqes` does not carry.
- **B2 event closure.** The multiset of ledger `CompletionEvent` facts whose
  `execution_id` is this execution equals the multiset of `result.events`.
- **B3 completion stage.** Exactly one `COMPLETION`/`COMPLETED` `StageRecord`
  is scoped to this execution, at `result.completed_at_ps`.

Every clause is read-only. None may change a timestamp, a digest, a completion
identity, a request metric or a random draw.

## Registered fixture

One graph, `cross-layer-authority`, released at 0 on the coarse runtime with
`launch_service_ps=1000`, `nccl_channel_service_ps=5000`,
`control_service_ps=2000`, `completion_delivery_ps=700` and the default
400 Gbit/s RNIC rate. All five operations are on rank 0 and correlate one
request:

| Operation | Logical queue | Work |
|---|---|---|
| `compute-a` | `cuda:0:compute` | compute, 100,000 ps, 1,024 HBM bytes |
| `xfer` | `cuda:0:nccl` | all-to-allv, pair (0 -> 8, 4,096 bytes) |
| `xfer-fifo` | `cuda:0:nccl` | all-to-allv, pair (0 -> 8, 2,048 bytes) |
| `xfer-rival` | `cuda:0:nccl-b` | all-to-allv, pair (0 -> 8, 8,192 bytes) |
| `ctrl` | `cuda:0:ctrl` | asynchronous control to rank 8, 128 bytes |

The shape is chosen so the projection is tested where the two authorities can
actually disagree rather than where they agree trivially:

- `xfer-fifo` shares a logical queue with `xfer`, so its NCCL-channel visit has
  `submitted_at_ps` strictly earlier than `eligible_at_ps`. A `QUEUED` event
  rendered at submission instead of eligibility is a different number only on a
  visit of this shape.
- `xfer-rival` sits on its own logical queue and contends for the same NCCL
  channel, so its visit has a strictly positive queue wait. Eligibility and the
  grant are different numbers only on a visit of this shape.
- rank 0 and rank 8 are on different nodes, so all four transfers become WQEs
  on one RNIC and exercise the subject-keyed half of the projection.
- `ctrl` is the only control work, so `class_service_bytes` is nonempty.

## Registered contradictions

Each mutation below violates exactly one registered clause and must be
rejected. All thirteen are accepted today; diagnostic 3 recorded that.

| ID | Mutation | Clause |
|---|---|---|
| C1 | `QUEUED` of the gated visit moved to its `submitted_at_ps` | A1 |
| C2 | one `SUBMITTED` event dropped | A1 loss |
| C3 | one `STARTED` event duplicated | A1 duplication |
| C4 | one `PROGRESS` byte count raised by one | A1 |
| C5 | a WQE `COMPLETED` timestamp moved one ps earlier | A1 |
| C6 | an event copied onto a subject object no authority carries | A1, A5 |
| C7 | eligibility and grant swapped on the contended visit | A1 |
| C8 | `class_service_bytes` replaced by an invented pair | A6 |
| D1 | WQE object `created_at_ps` moved one ps earlier | B1 |
| D2 | WQE object `bytes` metadata raised by one | B1 |
| D3 | one report WQE removed from the ledger | B1 |
| D4 | a completion event added to the ledger only | B2 |
| D5 | the completion stage timestamp moved one ps later | B3 |

C1 and C7 are the discriminating pair. Under a visit with zero queue wait and
immediate eligibility, submission, eligibility and grant are the same number
and all three renderings agree; the fixture exists so they do not.

A rejection must leave consumer state untouched: the virtual clock, the latest
request metrics and the ledger must be identical before and after each refused
attempt.

## Scored behavioral families

Diagnostics 1 and 2 are disclosed above and are not scored. The scored headline
is three families over eighteen instances.

### CLA-B1: the registered contradictions are rejected, 13 instances

Each of C1 through C8 and D1 through D5 must be rejected by the enforcing
consumer with an error naming the violated join, while the unmutated evidence
of the same fixture is accepted.

### CLA-B2: the projection survives a two-parameter sweep, 4 instances

The fixture is run at RNIC rate 400 and 200 Gbit/s crossed with
`completion_delivery_ps` 700 and 1,500. Every cell must satisfy A1, A5, A6, B1,
B1b, B2 and B3, and must reproduce the registered JCT exactly.

The four WQEs carry 4,096 + 2,048 + 8,192 + 128 = 14,464 bytes, i.e. 115,712
bits, and share one RNIC. The launch queue serializes five operations at
1,000 ps each, so `xfer` reaches its NCCL channel at 2,000 ps and its NIC at
7,000 ps. From that grant the RNIC is saturated with no idle gap, so

`JCT = 7,000 + 115,712 * 10^12 / rate + completion_delivery_ps`.

| Cell | Rate Gbit/s | Delivery ps | JCT ps |
|---|---:|---:|---:|
| baseline | 400 | 700 | 296,980 |
| slow link | 200 | 700 | 586,260 |
| slow delivery | 400 | 1,500 | 297,780 |
| both | 200 | 1,500 | 587,060 |

The baseline cell was observed while designing the fixture and is disclosed as
a preservation baseline, not as a scored prediction. The other three cells are
predicted from the arithmetic above and are the scored instances, together with
the two relations they imply:

- halving the rate adds exactly 289,280 ps at both delivery values, because the
  serialization term scales as 1/bandwidth;
- adding 800 ps of delivery adds exactly 800 ps at both rates, because
  visibility is a constant term that does not scale with bandwidth.

`result.completed_at_ps`, `result.quiesced_at_ps`, `StepResult.step_latency_ps`
and the single request's TTFT are all equal to the cell's JCT, because the
last WQE to complete is the graph's latest logical completion and the request
is scheduled at virtual time 0.

### CLA-B3: the shapes are distinguished, 1 instance

The fixture must contain a strictly positive count of subjectless visits with
`eligible_at_ps > submitted_at_ps` and a strictly positive count with
`started_at_ps > eligible_at_ps`, and those two counts must be reported. If
either were zero, C1 or C7 would be a no-op mutation and its rejection would
prove nothing.

## Fatal and unscored evidence

A violation in any class below voids the run. None of these counts enters the
behavioral denominator.

- The whole existing test suite passes, unchanged, with the enforcement live.
  This is the preservation claim: adding a consumer-side projection check must
  change no accepted timestamp, digest or completion identity.
- The unmutated evidence of every cell is accepted by the reducer and by the
  ledger projection.
- No refused attempt mutates the virtual clock, the latest request metrics or
  the ledger.
- The runtime remains the only producer of events; the enforcing check creates
  no event, visit, WQE or ledger fact.
- `ruff check .` passes.

## Physical sanity before the exact comparison

This task changes no modeled duration, so the sanity check is that the numbers
it reads still obey their own physics.

- Floor: 14,464 bytes on one 400 Gbit/s RNIC cannot be serialized in less than
  289,280 ps, and nothing can complete before the 7,000 ps at which the first
  WQE is granted. The registered 296,980 ps sits exactly on that floor plus the
  700 ps visibility term, which is what a saturated single-RNIC train must do.
- Floor at 200 Gbit/s: 578,560 ps of serialization, so 586,260 ps is again
  exactly the floor plus visibility.
- Ceiling: if no transfer overlapped anything, each of the four would also pay
  its own launch and channel service serially, giving at most
  7,000 + 289,280 + 3 * 5,000 + 700 = 311,980 ps at 400 Gbit/s. The registered
  value is below that ceiling, which is the overlap the runtime claims.
- The scaling check: halving the link rate must move a serialization-bound term
  by close to two. Here the serialization term moves by exactly two and the
  fixed 7,700 ps does not move at all, so the JCT ratio is 1.974, not 2. A
  measured ratio of exactly 2 would mean the fixed terms had been scaled too,
  and a ratio near 1 would mean the transfers were not serialization bound.
- System plausibility: 14 KB across a 400 Gbit/s NIC in about 0.3 us is the
  right order for a single small collective on one rail. A value in
  nanoseconds would imply a rate above the link, and a value in milliseconds
  would imply about 40 Mbit/s.

## Registered acceptance clauses

1. The exact projection above is implemented as one shared derivation used by
   both the producer and the consumer, and enforced by the consumer: the
   completion-event clauses in `CompletionReducer`, the bookkeeping clauses on
   the ledger the runtime appends.
2. All thirteen registered contradictions are rejected, each unmutated cell is
   accepted, and no refused attempt mutates consumer state.
3. Every accepted timestamp, digest and completion identity is preserved
   exactly, and the existing test suite passes unchanged.
4. The registered JCT table and both scaling relations hold exactly.
5. The duplicated-quantity inventory is recorded, including the entries this
   study does not resolve.
6. Any registered CORE-8 acceptance clause this run does not demonstrate stays
   on CORE-8, and any clause this run registers and does not demonstrate moves
   to a new task ID from the range allocated to this branch.

## Registered command and check-only dry run

```bash
.venv/bin/python examples/cross_layer_authority_v1/run_study.py \
  --out "$SIMLLM_CROSS_LAYER_AUTHORITY_RUN_ROOT"
```

Before this expectations-only commit the same command is run with
`--check-only`. That path validates only the frozen literal table and its
arithmetic. It imports no SimLLM module, reads no input path, invokes no native
binary and creates no output artifact.

The result report records the SimLLM revision the run observes as provenance,
separately from the revision this evidence was authored against, and asserts no
equality between either of them and a live submodule pin.

## Scope this freeze does not cover

CORE-8 also requires the native side to extract a protocol-neutral reservation
timeline and finite-capacity resource from the PCIe implementation, and shared
golden fixtures proving isolated zero wait, one external contention wait
without triangular self-charging, predecessor service excluded from queue wait,
finite-capacity release and overflow rollback with identical reductions in both
languages. Those clauses are frozen in
[queue_contract_v1/expectations.md](../queue_contract_v1/expectations.md) and
are not demonstrated by this study. They stay on CORE-8.
