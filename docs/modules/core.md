# simllm.core

Framework-agnostic heart of the simulator: virtual time, scheduler-step
records, the device-level execution graph and completion feedback. The core
defines contracts and orchestration. Framework policy stays in adapters;
compute calibration, traffic expansion and physical backends stay in their
own modules.

## Interface

### Scheduler boundary

- `StepRecord`: what a framework scheduler decided to run in one engine step
  (per-request phase, new tokens, cached tokens, preemptions and finishes),
  plus optional exact `num_sampled` and `num_tokens_after_padding` counts. The
  latter is the physical model-input token count after framework padding and
  must not replace the logical scheduled-token fields. Absent optional counts
  keep legacy v1 records valid and select the consumer's documented
  approximation. Optional `sampled_request_ids` identifies an exact sampled
  subset and must agree with `num_sampled`; legacy readers remain valid when
  it is absent.
- `StepResult`: the scheduler-facing result (step latency and completion time
  on the virtual clock), sampled per-request `RequestMetric` rows and a
  graph-wide `AdditiveVisitTotals` work sum. Each request row carries exact
  rational TPOT plus a conserved `LatencyAttribution` over queue, KV, kernel,
  DMA, collective, NIC and control owners. Additive visit work is a different
  type and never enters that identity.
- `RequestPhase`, `ScheduledRequest`: the per-request vocabulary.
- Closed-loop wire schemas: `atlahs-closed-loop-step-v1` is the JSON form of
  `StepRecord`; `simllm-step-result-v2` is the strict full `StepResult` form.
  It preserves every request metric, exact rational TPOT, conserved latency
  attribution and separately typed additive visit totals. The earlier
  `atlahs-closed-loop-result-v1` name had no accepted payload and is rejected
  explicitly rather than upgraded from invented fields.
  `simllm-step-result-v3` preserves that metric payload field for field and
  adds one required `run_provenance_ref` object with exactly `schema` equal to
  `simllm-run-provenance-v2` and `sha256` equal to that canonical record's
  lowercase hexadecimal SHA-256. The v2 provenance writer preserves the core
  v1-family byte convention: compact UTF-8 JSON followed by exactly one LF.
  The reference hashes the complete serialized record including that terminal
  LF; it does not use the calibration record writer's no-newline convention.
  The live publisher atomically publishes the
  canonical provenance record beside the v3 result and verifies the reference
  before any callback or terminal frame. A path that emits no live provenance
  keeps using strict v2 and reproduces its bytes exactly; an empty, dangling or
  fabricated v3 reference is invalid.
- Record serialization both ways: `step_record_to_json`,
  `step_records_to_json`, `write_step_records`, `step_record_from_json` and
  `step_records_from_jsonl`, plus `step_result_to_json` and
  `step_result_from_json` for full results.

### Disaggregated serving boundary

- `DisaggregatedSession` composes one prefill pool, one explicit
  `KvHandoffEvent` and one decode pool on a shared `VirtualClock`. The two
  framework schedulers remain the only batching authorities. The session
  carries one stable request identity across pool-local scheduler identities,
  and it reports TTFT from session admission through the first decode token
  and TPOT only from decode-pool cadence.
- `KvHandoffEvent` is the sole handoff timing authority. Its declared-constant
  arm derives bytes from model KV geometry and prompt context, advances the
  shared clock once, and publishes a loss-checked read-only record. Its packet
  policy charges PCIe submission, renders the same bytes through GOAL and the
  packet backend, and completes at the last required arrival. TRAF-62 and
  TRAF-64 retain the PLACE-5-dependent target-topology qualification.

### Execution and completion boundary

`simllm.core` publishes four versioned contracts:

- `simllm-execution-graph-v1`: an immutable `ExecutionGraph` containing an
  ordered tuple of `ExecutionOperation` nodes. Tuple order is FIFO submission
  order within each `logical_queue`; `depends_on` supplies whole-operation
  cross-queue edges. The separate `participant_local_depends_on` tuple lets
  each rank of a distributed command wait for the predecessor frontier on that
  same rank. A node may carry both edge kinds, so a local arrival cannot weaken
  an independent whole-operation barrier. Operations in different queues with
  no path between them are legally allowed to overlap.
  `completion_operation_ids`
  names the logical boundary that releases the framework; an empty tuple means
  every operation, while an explicit subset permits background asynchronous
  work to remain in the stateful runtime.
  `effective_dependency_edges` is the one typed expansion of that ordering:
  it combines explicit and same-queue FIFO predecessors, retains
  whole-operation versus participant-local scope and records the applicable
  rank for each local edge. Graph validation, `CoarseDeviceRuntime` and traffic
  projections consume this same immutable edge inventory.
- `simllm-completion-event-v1`: a timestamped `CompletionEvent` at submitted,
  queued, started, progress or completed phase, optionally attributed to one
  concrete `ResourceRef`. `subject_object_id` identifies the WQE or other
  created object when one operation expands into several runtime objects.
- `simllm-execution-result-v1`: an `ExecutionResult` containing the graph
  completion time and its ordered completion-event stream. A separate optional
  quiescence time distinguishes framework-visible completion from all physical
  work draining.
- `CompletionReducer`: the stateful read-only projection from one validated
  `ExecutionResult` plus its `RuntimeReport` to `StepResult`. It follows the
  runtime's realized predecessor chain for each correlated request endpoint,
  accumulates TTFT across non-sampling prefill chunks, computes exact TPOT,
  advances `VirtualClock` only to framework completion and retains later
  asynchronous events as physical evidence. Its optional bookkeeping input
  seeds each request's first-token history at the framework-request creation
  timestamp; omitting it selects the exact legacy timing origin.
- `simllm-request-bookkeeping-v1`: an append-only `BookkeepingLedger` of
  request-stage transitions, created-object records and the same
  `CompletionEvent` objects returned by the runtime. `RequestBookkeeper`
  assigns sequence numbers, validates lineage, registers a graph without
  mutating it, and queries by request, execution or object ancestry.
  `framework_request_arrivals` validates the complete ledger and returns one
  ordered arrival projection per framework request from
  `CreatedObjectRecord.created_at_ps`.

Created objects use typed portable identities and opaque owner-native handles:
framework request and vRAM allocation, execution operation, NCCL command,
SQ/RQ/CQ and network WQE. The current v1 compatibility ledger also records
either a DCQCN QP or an `rnic-cn` directed L2 link pair. In that compatibility
shape, every WQE names exactly one SQ, RQ and CQ. Physical compatibility WQEs
name exactly one QP or link pair;
topology-free null profiles instead record `transport_kind=none`. BACK-11 will
give every full-RNIC policy a hardware QP and retain CC/link-pair identity as a
separate policy context. Packet objects are intentionally absent. Framework
pointers remain strings owned by the adapter; the core records them but never
interprets or dereferences them. Reusable vRAM, queue, QP and link-pair records
retain their creation scope but may be referenced by later steps and
executions. Each use carries its own scope. Causal object lineage may narrow a
batched request set, but cannot introduce a request absent from its causal
parents.

### Registered device resource projection

The registered-device extension is a new strict wire family. It does not widen
the accepted vocabulary of `simllm-completion-event-v1`,
`simllm-execution-result-v1` or `simllm-request-bookkeeping-v1`:

- `simllm-completion-event-v2` preserves every v1 event field and adds one
  closed `registered-device` resource-reference variant. That variant carries
  `registry_sha256`, `device_kind_id`, `device_instance_id`, `axis_id`,
  `resource_instance_id` and `latency_owner`.
- `simllm-execution-result-v2` contains only v2 completion events and preserves
  the v1 completion-boundary and physical-quiescence meanings.
- `simllm-request-bookkeeping-v2` contains the same v2 completion events and
  preserves the v1 append, lineage and query meanings.

`registry_sha256` is the canonical identity of the validated device resource
registry. `device_kind_id` and `axis_id` must resolve in that exact registry.
`device_instance_id` distinguishes concrete devices of the same kind, while
`resource_instance_id` distinguishes resources on that device, e.g. an HBM
service, copy engine, channel or peer port. Their tuple is unique within one
run. Core owns a
closed `LatencyOwner` vocabulary whose wire values are exactly `queue_ps`,
`kv_ps`, `kernel_ps`, `dma_ps`, `collective_ps`, `nic_ps` and `control_ps`, the
seven fields already conserved by `LatencyAttribution`. The registered
reference must name one of them. Unknown registries, unknown device kinds,
blank device or resource instances and absent or unknown latency owners are rejected before a
completion or bookkeeping record is published.

The in-memory surface is the closed union
`ResourceReference = ResourceRef | RegisteredDeviceResourceRef`.
`ResourceRef` retains its enum-valued `ResourceKind` and identifier exactly.
`RegisteredDeviceResourceRef` carries only `registry_sha256`,
`device_kind_id`, `device_instance_id`, `axis_id`, `resource_instance_id` and
`LatencyOwner`. Authoritative `QueueVisit`,
v2 completion and v2 bookkeeping projections accept that union; v1 records
accept only `ResourceRef`. A bare axis ID, base unit or capacity string can
never masquerade as either reference variant.

The v2 JSON union is exact. A legacy resource retains the v1 object bytes
`{"kind": <ResourceKind wire value>, "resource_id": <string>}`. A registered
resource is the strict object `{"kind": "registered-device",
"registry_sha256": ..., "device_kind_id": ..., "device_instance_id": ...,
"axis_id": ..., "resource_instance_id": ..., "latency_owner": ...}` with no
other member. `registered-device` is a union discriminator, not a new
`ResourceKind` value. The strict v1 reader rejects it, and the strict v2 reader
rejects missing or unknown members and every other discriminator.

The resource registry remains compute-owned. Core validates its supplied
identity, registered kind and latency projection, but does not interpret
service-demand axes, capacities or interaction laws. A device resource becomes
a queue visit only when its authoritative runtime emits the visit boundaries;
an internal service axis is never promoted into a core queue merely because it
is registered. Strict v1 readers and canonical v1 bytes remain unchanged, and
each strict reader rejects a payload from the other wire version.

### Precision selection and run provenance

`PrecisionConfig` names one level for each of the eight seams in the fidelity
matrix: `workload`, `request_outcome`, `framework`, `compute`, `dependency`,
`locality`, `network` and `rnic_hardware`. It is strict, so every field is
required, unknown fields and values are rejected, and a string is never
coerced to a nearby level. Construction validates the combination and refuses
an incompatible one with a diagnostic that names both seams and both escapes:
`composed-native` RNIC hardware cannot run on the `rnic-nn-fluid` closed form,
because the fluid path is the explicit nonstructural bypass anchor.
`PrecisionConfig.compatibility()` is the byte-locked baseline level at every
seam.

`RunProvenance` binds a resolved configuration to one source artifact's schema
and hash under `simllm-run-provenance-v1`, so a published result can be read
back with the precision that produced it. It serializes canonically, hashes
the precision payload itself, and rejects a missing or unknown field, a
malformed source hash, an unsupported schema, an invalid combination and a
mismatched precision digest. It is deliberately outside the bypass-identity
contract, so stamping a run moves no accepted byte.

The operational selectors are unchanged and stay authoritative. A provider
object, profile string, placement-manifest presence, observation supply,
authority mode and adapter environment spelling still select the mechanism;
the surface only resolves which level each one names and refuses an explicit
disagreement. Two entry points keep that honest:

- `check_precision_selection` reports only the seams a component observes and
  validates those against an explicit configuration. It never invents a level,
  so a component with a partial view is neither credited with nor refused on a
  seam it does not own. A structural `CoarseDeviceRuntime` selects
  `composed-native` and selects no network level at all.
- `resolve_precision_config` composes a complete run configuration, filling
  any unobserved seam with its compatibility level and validating the whole.
  Its caller is asserting a whole run, so an incoherent completion is refused
  rather than degraded.

`compute_level_for_provider` reads the `precision_compute_level` attribute a
`ComputeProvider` declares. A caller-defined provider that declares nothing
resolves to `None`: the surface records that the spelling constrains nothing
here rather than putting a guessed level into a stamp.

`simllm-run-provenance-v2` preserves every v1 source and precision field and
adds exactly top-level `instance_graph_sha256`,
`resolved_device_binding_closure_sha256` and canonical `device_models` members.
For a compact-device execution its inherited `source_schema` is exactly
`simllm-execution-graph-v1` and inherited `source_sha256` equals
`instance_graph_sha256`; two different source identities reject the record.
The closure hash names the graph-total
`simllm-resolved-device-binding-closure-v1` record, not one different closure
per device. Each device-model entry has exactly `device_instance_id`,
`device_model_id`, `device_model_sha256`, `acceptance_status`, `target_basis`
and `operating_envelope_sha256`. Acceptance status has the closed wire values
`candidate` and `validated`; target basis has the closed wire values
`target-silicon` and `architecture-derived`. Provenance copies both unchanged
from the selected model and rejects `architecture-derived` with any status
other than `candidate`. Entries are unique and ordered by device instance ID.
Exactly one model is selected per device instance, but a heterogeneous tuple
across device instances is legal and remains part of the result identity. The
resolved operation set, optional collective set and their dispatch context
must name exactly these device-instance/model-SHA pairs and the provenance
graph; a cross-graph, cross-context or cross-model splice rejects publication.
The
selected model and operating-envelope records are reachable in the result's
content-addressed artifact closure and are digest-verified before publication.
Candidate and architecture-derived selections remain explicit facts; recording
either never promotes it to validated target-silicon evidence.

The live result carries provenance by content identity, not by copying a
mutable selection object into request metrics. `simllm-step-result-v3` adds the
single `run_provenance_ref` described at the scheduler boundary, while every
TTFT, TPOT and attribution field keeps the v2 meaning. A selected compact
device model requires a total resolved-device-binding-closure digest and a v3
result. The
compatibility path keeps strict `simllm-run-provenance-v1` available as a
separate record and keeps `simllm-step-result-v2` byte-identical.

An in-process publisher adopts the canonical provenance object and v3 result in
one prepared result batch before callbacks. An out-of-process session sends the
canonical provenance record in the same transaction before its terminal v3
frame; the receiver recomputes the digest and refuses to publish either object
on mismatch or absence. A content hash never stands alone as the only way to
retrieve result provenance.

### Authority, queue visits and arbitration

An enabled execution profile has one mutable authority for each object. The
`ExecutionGraph` is authoritative for semantic work, logical release
constraints and dependency identity. The `DeviceRuntime` owns orchestration:
realized operation eligibility, operation-level start and completion, and
selection only for resources it directly implements. It delegates
provider-owned physical objects. When the structural RNIC path is enabled, the
selected native `WqeRecord` owns the WQE lifecycle and every calibrated
per-WQE start stage.
The request bookkeeper, backend result rows and completion stream are
projections of that record, not independent WQE state machines; the
`AtlahsWqeLedger` is not constructed in this mode. A run may instead explicitly
select `AtlahsWqeLedger` as its sole timing-neutral bypass authority. A run
never enables both mutable authorities, and every
projection must conserve identity, cardinality and timestamps available at its
boundary. `CompletionReducer` owns only request metric history; it does not
schedule, progress or complete a runtime object.
In a graph-authoritative sink run, GOAL operations, messages, dependency
provenance and backend rows are checked read-only projections of the graph and
runtime authorities. They may reject an unrepresentable graph before
execution, but they may not reconstruct or weaken its ordering. A standalone
direct-GOAL run instead selects the independently rendered ATLAHS schedule as
its ordering authority. Setting
`HtsimStepSinkConfig.dependency_cross_check="atlahs-goal"` does not change the
serial sink's selection: the `ExecutionGraph` projection still determines its
result, while the direct-GOAL execution contributes only a diagnostic report.
The report records ordering-scope, raw phase-frontier and completion-time
differences; it never averages, overrides or silently prefers one mechanism's
timestamps. The switch selects a diagnostic, never a fidelity level: seam
levels are named by `PrecisionConfig` and recorded by `RunProvenance`.

The target contract requires all contended resources to use one queue-visit
meaning even when Python and C++ use different mechanisms:

| Point | Meaning |
|---|---|
| `submitted_at` | Work entered its logical queue. |
| `eligible_at` | Causal, ordering and other gates external to this resource are satisfied. |
| `started_at` | Arbitration granted the resource and non-preemptive service began. |
| `finished_at` | This resource was released. |
| `completed_at` | The result became visible to the downstream consumer. |

For one visit, queue wait is exactly `started_at - eligible_at`, service is
exactly `finished_at - started_at`, and downstream response or visibility is
`completed_at - finished_at`. `CompletionEvent.QUEUED` projects eligibility,
not submission or an arbitrary simulator callback time;
`CompletionEvent.STARTED` projects the resource grant. An internal mechanism
may retain finer visits than the public event stream, but it must reduce them
with these meanings and pass the shared conformance fixtures.

`simllm/core/authority.py` is where those projections stop being a declaration
and become a check. The `RuntimeReport` is the authority: its queue visits own
visit timing, its WQE projections own WQE lifecycle timing, its operation
records own operation completion, and the `ExecutionGraph` owns the semantic
bytes an operation declares. The completion stream must equal, as a multiset
joined on `(operation_id, subject_object_id, phase, resource)`, the four phase
events of every subjectless visit, the five lifecycle events of every WQE and
one logical completion per operation. The bookkeeping ledger must carry one
network-WQE object per reported WQE with that WQE's creation time, bytes, tag,
ranks, sequences and channel, exactly the result's completion events for the
execution, and exactly one completion stage at the result boundary. Loss,
duplication and timestamp disagreement are separated in the rejection message.
`CompletionReducer` enforces the event half, which puts the check on the
consumer boundary every device runtime passes through; the runtime enforces the
ledger half inside its staged append, whose staged copy is discarded on
failure, so a disagreeing ledger aborts before the caller's bookkeeper, the
runtime state or the WQE authority is mutated.

Two byte quantities stay deliberately distinct there. An operation's logical
completion event carries the graph's declared semantic payload, while
`PROGRESS` and WQE events carry the bytes a resource actually served. A ring
all-reduce declares one payload and physically moves several times that, so the
two are never summed and each is checked against its own owner.

Two reductions remain deliberately separate. `sum_visit_wait_ps` sums work
waiting across resource visits and may exceed elapsed time when visits overlap.
`critical_path_queue_ps` includes only waits on the realized dependency and
resource critical path. Only the latter can participate in an additive TTFT,
TPOT or JCT decomposition. GPU wall-idle classifications, PCIe transaction
wait sums, WQ stage waits and last-completion makespans retain distinct names;
no caller may compare or add them merely because each has units of time.
Critical-path accounting is keyed by participant, not by operation. Each
operation publishes one `RuntimeCriticalSegment` for every rank returned by
`operation_participant_ranks`, and that per-rank segment set is the
conservation authority. A segment records its rank's causal boundary,
completion, selected resource path breakdown, latency attribution and its
predecessor as an explicit `(operation_id, participant_rank)` pair: a
participant-local edge names the same rank, a whole-operation edge names the
predecessor's logical-maximum rank, and a root segment names none and starts at
graph release. A scalar predecessor per operation cannot express this, because
a multi-rank collective has a per-rank frontier and a rank proceeding legally
from its own predecessor would appear to overlap the operation's one global
predecessor.

Each segment begins at its named predecessor segment's completion, exactly.
Queue, service and visibility intervals that completed before that boundary
contribute zero; intervals crossing it contribute only their remaining tail.
Each segment's breakdown and attribution must each sum to
`completed_at_ps - started_at_ps`, and the realized endpoint chain
`realized_critical_path_segments` must be acyclic, start at graph release and
sum to endpoint completion exactly.
`realized_critical_path_operation_ids` and the operation-level
`critical_predecessor_id`, breakdown and attribution remain as explicit
compatibility projections of that authority; they may not replace, relax or
contradict it. CORE-46 closed that gap: the reducer now derives each scalar
field from the segments and rejects any disagreement. The physical completion
is the segment maximum, the scheduler-visible completion is one of the
operation's own participant completions, the causal boundary is a participant
completion of the named predecessor that one of the operation's own segments
starts at, the additive `critical_predecessor_id` is present exactly when that
boundary is the predecessor's scheduler-visible completion, and the
operation-level breakdown spans exactly the interval that projection implies.
An asynchronous operation may release the framework at a participant completion
earlier than its physical maximum; it may not report a timestamp the segments
do not carry.

Every optional class or priority scheduler must sit behind a replaceable
policy. Mandatory protocol legality and ordering constrain the ready set before
that policy is called. The required identity policy ignores class and priority
labels and returns the first request in the resource's deterministic baseline
order.
Selecting identity is the feature-off path: it must preserve timestamps,
waits, byte counts, random draws and completion order exactly, including when
class labels are permuted. A non-identity policy may reorder only legal ready
requests. For example, it cannot violate SQ ordering or PCIe forward-progress
rules.

The coarse runtime calls this seam at two points, and both grant one operation
at a time. It selects among simultaneously legal ready graph operations, and
once the winner is compute work it then orders the co-runnable set by repeated
grants, offering the members not yet granted in the deterministic baseline
order. Membership of that set is computed before arbitration and is never
changed by it: the policy decides only the order in which the members reach
`SmSchedulerModel.estimate_concurrent`, which replays them in the order it
receives and therefore charges the submission-order issue term registered by
COMP-12 to whichever task the policy submitted second. Under identity every
grant is the smallest remaining baseline sequence, so the ordered tuple is the
`ExecutionGraph` tuple order the seam produced before the policy reached it.

Framework-launch FIFO, logical-queue FIFO, dependency release, NCCL round order
and WQ protocol order are mandatory rather than optional arbitration.
Compatible copy engines use deterministic earliest-availability routing after
the operation has won the ready seam; native RNIC arbitration remains owned by
its sole session. These policy-free points do not inspect class labels.

Three policies ship. `IdentityArbitrationPolicy` is the feature-off path.
`StrictPriorityArbitrationPolicy` grants the smallest class label and falls
back to baseline order, so it is stateless and a total order over any candidate
set. `WeightedRoundRobinArbitrationPolicy` gives each class label its weight of
grants per round and carries the credits between grants, so a favored class
wins more often without starving the others; a new round begins when every
class present has spent its credits. Both carry an explicit `class_aware=False`
identity setting that reproduces the identity policy grant for grant, which is
what keeps a class-label permutation a no-op on the accepted baseline. Any
other object conforming to `ArbitrationPolicy` may replace them.
[The arbitrated-order study](../../examples/arbitrated_order_v1/RESULTS.md)
closed CORE-49 and CORE-10 on this seam: it passes 8 genuine-risk instances
across two families with all 44 fatal guards holding, moves a live step JCT,
TTFT and TPOT by exactly the one registered issue cycle under a reordering
policy, and pins the identity ordered tuples as literal values.

The pre-implementation queue conformance expectations were first frozen in
[examples/queue_contract_v1](../../examples/queue_contract_v1/expectations.md)
at commit `65b5609`; commit `facb26d` clarified identity-policy scope, and
commit `947399c` records the final pre-run state.

The graph carries five typed work payloads rather than one unstructured
dictionary:

| Payload | Semantic owner | Runtime lowering |
|---|---|---|
| `ComputeWork` | Model runner plus `ComputeProvider` | launch queue, CUDA stream, GPU work queue, hardware scheduler and HBM |
| `KvCacheWork` | Real framework KV/prefix-cache manager | `KvLifecycleLedger` accounting in preflight, then the rank's HBM queue for a byte-carrying action; refused before authority mutation when no pool is declared |
| `DmaWork` | Data-mover planner | DMA descriptor, directional copy engine, source/destination memory queues |
| `CollectiveWork` | Framework/NCCL observation plus traffic planner | NCCL channels, DMA/HBM work, WQEs and network flows |
| `ControlWork` | Framework or runtime controller | synchronous or asynchronous labeled local/fabric control path; priority only under an opted-in policy |

`CollectiveWork.payload_bytes` is algorithm-relative, and consumers must
branch on `(collective, algorithm_hint)`: a ring `all-reduce` carries the
full reduced payload, while a `pairwise` `all-to-allv` carries the bytes each
rank sends to each other rank (one uniform ordered-pair share; the serial
lowerer and the serial GOAL renderer agree on this decoding). The optional
`pair_payload_bytes` table is the variable-size form for pairwise all-to-allv:
each source-major entry is `(source_rank, destination_rank, bytes)`, omitted
pairs carry zero bytes, and a nonempty table requires the scalar to be zero.
The field is omitted from JSON when empty, so old
`simllm-execution-graph-v1` scalar payloads retain their exact bytes and
meaning. The strict reader rejects duplicate, self, out-of-group, nonpositive,
unsorted and scalar-plus-table entries. Both the serial GOAL renderer and the
coarse runtime consume the same table when each declared rank sends or
receives. The runtime also accepts a valid table with an uncovered rank; the
diagnostic serial renderer rejects that case because it cannot emit the
rank's collective-completion frontier.
An optional `request_pair_payload_bytes` tuple partitions that sparse table by
stable request identity. Its request-major entries are
`(request_id, source_rank, destination_rank, bytes)`, and strict validation
requires their per-pair sums to reproduce the aggregate table exactly and each
identity to appear in the operation correlation. The field is read-only
metadata: graph JSON retains it, structured GOAL messages retain it, and GOAL
text does not emit it. Direct and graph renderers fail closed if the structured
message projection disagrees with the corresponding authority.
The combined captured-routing study populated that table from real Granite
assignments, carried it through the step graph and GOAL, and changed live
fluid JCT by every frozen exact relation. It also retained the old v1 scalar
wire and GOAL hashes, closing CORE-6; see
[the routed supply results](../../examples/routed_supply_v1/RESULTS.md).
The coarse ring path currently requires a positive payload evenly divisible
by its rank count, so every round sends an exact integer chunk and never
fabricates a byte. CORE-16 owns remainder chunking. Control sends reserve
1,024 tags per operation beginning 1,000,000 above the collective base; more
than 1,024 destinations or a collective allocation reaching that boundary
fails preflight before authority mutation.

`ExecutionLowerer` and `DeviceRuntime` remain narrow protocols.
`CoarseDeviceRuntime` is the first additive implementation. Its
`CoarseDeviceProfile` fixes the initial eight-GPU/eight-RNIC mapping, while
`RuntimeReport`, `RuntimeOperationRecord` and `QueueVisit` expose diagnostics
without widening `ExecutionResult` or changing the serial baselines.
`SerialStepLowerer` implements the diagnostic compatibility schedule and
`render_serial_execution_graph_goal` replays its supported subset using only
the JSON-round-tripped graph. The lowerer places distributed sequencing in
`participant_local_depends_on`; the renderer rejects operation-scoped
cross-rank barriers instead of weakening them. The backend driver's GOAL
completion summary participates in schedule JCT even when earlier WQE rows
exist, so compute-only TP=1 graphs and graphs with trailing compute are both
covered. A runtime can receive an optional central bookkeeper; the coarse
resource scheduler is implemented alongside these baselines. Current adapters continue to use the existing
`StepRecord -> StepResult | None` sink until their observation producers land.

Nothing in this package may import vLLM or SGLang.

## Mental model and ownership

```text
workload
  -> framework scheduler and KV control plane
  -> StepRecord plus adapter observations
       \-> request stages + opaque vRAM refs -> RequestBookkeeper
  -> ExecutionLowerer
       compute estimates + KV events + stream/dependency order
  -> ExecutionGraph v1
  -> DeviceRuntime
       launch/CUDA queues -> GPU scheduler + HBM
                          -> copy engines + DMA
                          -> NCCL channels -> semantic RNIC submission
                          -> one GPU-affine RNIC session per GPU (WQ/CQ + hardware)
                                                       -> network backend
  -> CompletionEvent v1
       \-> operation + WQE subject -> RequestBookkeeper
  -> StepResult, virtual clock and TTFT/TPOT attribution
  -> framework scheduler
```

The ownership rule for overlap is strict:

1. The framework adapter records program order, streams, events, barriers and
   whether a control or collective launch is synchronous or asynchronous.
2. The lowerer expresses that knowledge as queues and dependency edges. It
   never inserts an empirical overlap percentage.
3. The device runtime applies queueing and physical resource arbitration
   through an explicit policy. Identity arbitration preserves each resource's
   existing deterministic baseline after legality. Overlap occurs only when
   operations are ready together and their selected resources permit
   concurrent service.
4. Compute, collective and network models supply service demand and resource
   behavior. They do not rewrite framework ordering.

The first runtime profile is intentionally fixed: eight GPUs per node, one
logical WQE submission queue or QP per GPU, and one GPU-affine 400G physical
RNIC per GPU. Cross-node WQEs contend within their selected RNIC, while the
eight rail endpoints can transmit concurrently. Intra-node traffic uses an
NVLink-class resource. General fabric discovery and arbitrary GPU-to-NIC
mapping are not prerequisites for this profile.

GPU fidelity is split at the operation boundary. `simllm.compute` owns the
service of one kernel, an explicitly supplied set of concurrent kernel tasks,
or one isolated copy descriptor. Its trace-driven model admits CTAs under
register, warp, thread, block and shared-memory limits, assigns them to SMs,
issues ready warps through dependency scoreboards, and services their shared
HBM and NVLink demand. The isolated copy model supplies descriptor setup and
directional bandwidth service. These mechanisms can be calibrated or replaced
without changing `ExecutionGraph`.

CORE-4 owns everything between semantic operations: launch and CUDA-stream
FIFO order, event dependencies, selection of the kernel set passed to
concurrent compute service, selection and queueing of copy engines,
simultaneous kernel and copy execution, kernel-versus-DMA HBM arbitration,
graph-level NCCL expansion, GPU-affine RNIC selection and semantic submission,
and completion-event/projection plumbing. In bypass mode it delegates SQ/RQ/CQ
and WQE state to the sole timing-neutral `AtlahsWqeLedger` authority. In
structural mode it delegates WQE lifecycle, WQ/CQ state, NIC arbitration and
completion to the native RNIC session delivered by BACK-8, with wider WQ/CQ
objects and pipeline arbitration remaining under BACK-9 and BACK-12. It must
call the compute service model rather than grow a second SM or SASS model in
`simllm.core`. The compute slice therefore does not claim whole-task execution
timing or compute/copy overlap.

## Pre-registered runtime sanity experiments

These expectations are recorded before CORE-4 implements scheduling. CORE-2
does not claim to produce these resource-contention measurements.

1. **Dependency versus legal overlap.** Release one compute operation of C
   picoseconds and one DMA operation of D picoseconds on independent logical
   queues with ideal independent resources. With no edge, makespan must be
   `max(C, D)`; adding a dependency must make it `C + D`. Sweep both the
   dependency setting and two demand pairs, `(C, D) = (10 us, 40 us)` and
   `(80 us, 40 us)`. Every result must match exactly in the ideal profile.
2. **Eight GPU-affine RNICs.** Each active GPU submits one aligned WQE of B
   bytes from its own FIFO/QP to its own rail RNIC, with no propagation or
   protocol overhead in the ideal profile. Sweep active GPUs N in `{1, 8}`
   and per-port rate R in `{200, 400}` Gbit/s. The phase makespan must be
   `8 * B / R` seconds independent of N, aggregate useful throughput must be
   `N * R`, and doubling R halves makespan exactly. Per-GPU FIFO order must
   remain stable under both rates.
3. **Tail attribution conservation.** For every completed operation, the sum
   of critical-path time attributed to launch queue, device queue, service and
   completion delivery must equal its end-to-end latency exactly. Separately
   report the sum over all queue visits, which may exceed latency when visits
   overlap and therefore must not enter that identity. No interval may be
   negative, and graph completion must equal the latest required completion
   event. Sweep synchronous versus asynchronous control delivery and two
   control-class labels under identity. Class labels must move nothing; only
   dependency-reachable work may move. Priority-caused movement belongs to the
   class-aware policies, never to identity.
4. **Identity arbitration is the exact off path.** Run each shared resource
   with omitted class arbitration and with the explicit identity policy, then
   permute class labels without changing arrivals or service demand. Event
   order, every timestamp, all wait and byte counters, random draws and final
   JCT must remain byte-identical. A separately enabled priority policy may
   change only the order of simultaneously legal ready requests.

## Status

Step records and the virtual clock (`VirtualClock`: heap-ordered events,
monotonic picosecond time and deterministic tie-breaking) are implemented and
tested; CORE-1 closed with M1. The step-record JSON readers landed with the
M4 first slice, which also exercised the step schema for real: recorded M2/M3
smoke JSONLs load, round-trip and replay through `HtsimStepSink`.

BRIDGE-1 is complete for the pinned-binary prepared-replay scope.
`HtsimPersistentStepSink` retains a local worker pool across batches, prepares
isolated diagnostic runs concurrently, publishes only a complete batch and
serves exact records in order. The
[frozen study](../../examples/bridge_persistent_v1/RESULTS.md) matched all 34
prepared-versus-diagnostic pairs in each of the result, outcome, GOAL text,
GOAL binary and completion-CSV evidence classes. All four scored live wall-time
instances passed with 3.36x to 5.43x speedup, and all six cells reported
physical quiescence. It remains the finite known-replay acceleration and does
not become the online stateful client.

CORE-24 is complete. `simllm-step-result-v2` now round-trips empty, prefill,
decode and mixed results through real JSON bytes, including `Fraction(1, 3)`
without float conversion. The strict reader rejects missing and unknown
fields, invalid conservation and the payload-less legacy schema name. The
paired persistent-session study records the exact wire digests and closure
evidence. HTSIM-18 is also delivered in the paired backend commit. BRIDGE-2
remains the graph-level client, lifecycle translation, ledger-cursor and
transactional-publication layer above these two foundations.

BRIDGE-3 is complete. Native simulator invocations now pass through one owned
child boundary. Linux uses a handshake launcher that arms a parent-death
signal before replacing itself with the simulator, with a unique process group
and signal-and-reap cleanup for catchable shutdown. Other POSIX platforms use
the process-group and catchable-signal path, but make no claim for uncatchable
termination or host failure. Windows assigns the blocked launcher to a
per-invocation Job Object with kill-on-close before releasing it. Primitive or
assignment failure rejects the launch, and unsupported platforms do not fall
back to an unowned process.

The [frozen child-lifetime study](../../examples/bridge_lifecycle_v1/RESULTS.md)
sent real `SIGTERM` signals to diagnostic and prepared owners. The unsafe
negative rows retained exactly 1, 2 and 4 orphaned children, while every
managed row retained zero after bounded polling, for 3/3 genuine-risk scored
instances. A separate pinned `htsim_rnic` run also retained zero. The unchanged
BRIDGE-1 checker preserved 34/34 pairs in each result, outcome, GOAL text, GOAL
binary and completion-CSV class, plus 4/4 latency streams and 6/6 quiescence
cells. Timeout, repeated cleanup, repeated close and unrelated-process controls
complete the registered lifecycle scope.

CORE-2 is complete. Graph structural and payload validation includes implicit
FIFO edges, strict JSON readers and writers cover all five work kinds, and the
serial compatibility lowerer retains per-layer request correlation, queues,
dependencies and collective semantics. The independent central ledger and its
JSON form retain opaque framework objects and WQE-level runtime lineage without
making the graph mutable. Its lineage rules distinguish causal parents from
reusable resource references, allow batched request scopes to split, and reject
request identities not supplied by a causal parent.
`RequestBookkeeper` supports sequential use only: append and extend do not
re-audit committed history, while snapshot validation and wire loads retain
the full reference validator.

CORE-7 is complete. `RequestBookkeeper.append` and `extend` validate only new
facts against private object, subject-timestamp and terminal-WQE indexes.
Atomic batches use a copy-on-write state overlay, so failed validation changes
neither the ledger nor its indexes. The complete validator remains the
complete-scan authority for initial immutable ledgers and wire loads. The
[incremental validation study](../../examples/core7_incremental/RESULTS.md)
matched the full validator across all seeded valid and invalid families. A
quadrupling from 1,000 to 4,000 and from 4,000 to 16,000 facts took at most
4.27x on the incremental path, while the reproduced former path grew at least
16.09x from 1,000 to 4,000 facts.

The pre-registered
[CORE-2 lowering study](../../examples/core2_lowering/RESULTS.md) compared the
legacy sink, graph-only JSON replay and a frozen closed form over TP width
`{2, 4}` and link rate `{200, 400}` Gbit/s. All four JCTs matched to 0 ps,
full completion rows were identical, and the MoE sentinel matched
25,811,524 ps with 48 flows. The HTSIM WQE identity layer was then checked
against the same frozen values; its post/dispatch and immediate-CQ accounting
adds no packet timing behavior. The core ledger invariants and backend CSV
rows are separately validated surfaces. `CoarseDeviceRuntime` now supplies
their concrete graph-operation/tag/WQE correlation.

Actual framework observation producers remain VLLM-11/12 and SGL-9/10.
BACK-9 and BACK-12 own the remaining structural RNIC objects and arbitration
depth.

Explicit KV state semantics now exist. `simllm/core/kv.py` holds the sole
mutable KV authority: `KvLifecycleLedger` replays the observed vocabulary
against a `KvPoolSpec` of fixed capacity, keeping each block in exactly one of
`FREE`, `LIVE` or `RECLAIMABLE`, and refusing a stream that breaks allocation,
ownership, reference-count, capacity or byte conservation. Every rule is
derived from the pinned vLLM 0.26.0 sources with file and line citations in the
module docstring, including the three that a plausible account of paged
attention gets wrong: eviction is lazy, so a cached block may only be
reallocated after an explicit `EVICT`; a request frees in reverse order, so a
shared prefix outlives its creator; and a block becomes reusable only once it
is full, which is what separates a voluntary `FREE` from an involuntary
`EVICT`. `BIND_PREFIX` is the reuse decision and `TOUCH` the reference-count
mechanism, split so a hit is never counted twice.

`CoarseDeviceRuntime` consumes that accounting during preflight, before any
resource is scheduled, on a clone that is adopted only when the whole graph is
legal. A byte-carrying action is then served from the rank's HBM queue at the
profile's `hbm_rate_bps` and attributed to `kv_ps`, so KV bytes reach
`RequestMetric`, TTFT and TPOT. Both off paths are exact: with no pool declared
a byte-carrying read or write is still refused before authority mutation, and
with a pool declared a zero-byte observation preserves every timestamp and
completion event.

The [KV lifecycle study](../../examples/kv_cache_strategies/RESULTS.md)
demonstrates the live relation. Shrinking the pool from 64 to 32 blocks raises
the replayed request's TTFT by exactly 2.0000x, a preemption raises TPOT by
exactly 2,090,000,000 ps at both HBM rates, and capacities above the
constraint threshold leave TTFT bit-identical. All 16 pre-registered scored
instances and all four post-specified family B regression checks pass, all 17
entailed relations hold and none of the 56 fatal guards is violated. CORE-3
stays open for the case matrix, the remaining sweep axes, the remaining
reporting surface and the `SWAP`, `TRANSFER` and `RECOMPUTE` lowering gap.

CORE-5 is complete for the supported core path. `CompletionReducer` consumes
the required graph boundary and the runtime's corrected critical-path
segments, validates the streamed event projection, returns `StepResult`, and
advances the scheduler clock without confusing physical quiescence with
framework completion. Per-request TTFT and exact rational TPOT retain queue,
KV, kernel, DMA, collective, NIC and control components. Graph and request
visit-work totals remain separately typed. Zero-sample prefill work accumulates
into the later first-token interval, and a zero sampled count remains empty
for decode rows too. Exact partial sampling requires explicit request
identities and fails closed when a count alone is ambiguous. The reducer
consumes each execution ID once, including zero-latency results, and the v1
reader treats an explicit null sampled-identity field as absent.

CORE-31 is complete. When supplied a bookkeeping snapshot, the reducer now
starts each request at its exact framework-request creation timestamp and
attributes arrival-to-first-release time to the request critical-path queue.
Scheduling before arrival rejects without advancing the clock or metric
history, while omitting bookkeeping retains legacy results. The joined live
vLLM study matched queue plus service to TTFT exactly in all 12 request rows;
see [the CORE-31 results](../../examples/arrival_admission_v1/RESULTS.md).

CORE-4 is complete for the coordinated first coarse bypass profile and the
frozen Tier B structural fixture.
`CoarseDeviceRuntime` implements host-launch and CUDA-stream order, dependency
release, co-runnable non-preemptive kernel dispatch into `simllm.compute`,
directional copy-engine queues, coarse shared HBM arbitration, NCCL channels,
NVLink-class intra-node service, GPU-affine cross-node semantic submission,
synchronous/asynchronous control completion and completion/bookkeeping
projection. `AtlahsWqeLedger` is the sole live bypass authority. Structural
mode constructs no ledger and stages submissions through an isolated
`NativeRnicTransaction`. The composed adapter consumes immutable native
observations transactionally; an abort leaves both runtime state and adapter
session counters unchanged. The composed C++ session remains the sole mutable
WQE lifecycle authority.

The [CORE-4 runtime study](../../examples/core4_runtime/RESULTS.md) cites the
older module expectations, the original expectations-only commit `d43cddb`,
and the integration-review amendment `67cabda`. Across 18 configurations it
passed 22/22 exact-oracle rows, 23/23 scored relations and 18/18 fatal
structural guards. Independent compute and DMA
matched `max(C, D)`, a dependency matched `C + D`, eight affine RNICs retained
one-port makespan while aggregate throughput reached `8R`, additive visit wait
was exactly four times wall JCT without entering the critical path, dependency
chain launch intervals were clipped at the realized predecessor boundary, and
omitted/explicit identity remained canonical-byte identical under class-label
permutation. Remaining coarse approximations and completeness gaps are
registered as CORE-11 through CORE-14 and CORE-16 rather than being claimed as
calibrated behavior.

The [TRAF-7 overlap study](../../examples/compute_comm_overlap_v1/RESULTS.md)
now drives adapter-authored logical queues and dependencies through that
runtime and the live completion reducer. Across two compute-to-communication
ratios, independent work matched `max(C, D)`, serial work matched `C + D`, and
the registered two-stage pipeline landed strictly between them at its exact
closed form. TTFT and TPOT moved by the same signed amount. The first added
resource family isolated the existing per-rank NCCL-channel cursor: changing
only shared to split channel identity reduced JCT by exactly 999 ps. This is
coarse executor evidence, not a claim that NCCL GPU kernels, HBM demand or
copy/GPUDirect service are calibrated; CORE-26, CORE-27 and COMP-22 retain
those gaps.

TRAF-12 reconciled the graph and GOAL dependency paths for the demonstrated
serial step-sink scope. `CoarseDeviceRuntime` now consumes the canonical edge
inventory, gates each supported participant on its exact predecessor frontier
and carries the causal witness selected by the realized completion path through
`RuntimeReport`. The same frozen Granite graph projects to checked backend
artifacts. The corrected rerun retained 144 operations, 423 effective edges,
72 causal artifacts, 47 required distributed FIFO boundaries and 376 other
serialized edges. Its 20,392-byte direct diagnostic produced 144 flows and
completed in 150,838,767 ps and 205,653,487 ps at 1,024 and 2,048 vector
bytes. The graph-authoritative path completed in 155,702,768 ps and
215,381,488 ps, for positive graph-minus-direct changes of 4,864,001 ps and
9,728,001 ps. Unsupported early completion and asynchronous
destination-local control shapes fail closed under CORE-29 and CORE-30; see
[the dependency authority results](../../examples/dependency_authority_v1/RESULTS.md).
The selectable ATLAHS cross-check preserves that sole authority while keeping
the independently constructed direct-GOAL schedule executable. The all-remote
comparator inspects all 423 canonical effective edges and reports 94
structural differences: 47 whole-operation logical-queue FIFO differences and
47 participant-local syntactic-frontier differences. Raw timing remains
evaluated on the 47 whole-operation boundaries, with 32/47 unequal, early
gaps. These are diagnostic findings, not values folded into `ExecutionResult`
or `StepResult`, so the authority conclusion is unchanged. Cross-check
disabled preserves the accepted graph artifacts, timestamps and results
exactly. Local-NVLink comparison rejects at preflight; TRAF-16 owns its
frontier precision. CORE-41 closed the ingress gap and refroze the two
single-node `AAAA` cells from 4,538,000 ps and 9,047,000 ps of service to
6,652,000 ps and 13,286,000 ps, carrying JCT to 6,676,000 ps and 13,310,000 ps;
every `AABB` and `ABCD` row is unchanged. CORE-42 then requalified
[nvlink_locality_v1](../../examples/nvlink_locality_v1/RESULTS.md) against those
refrozen cells: 3/3 genuine-risk families and 8/8 instances pass, both services
sit inside their exact serialization floor and their 48,000 ps
whole-nanosecond ceiling, and the refrozen all-local instances are classified
as genuine risk narrowed to the charge rule, the phase split and the rounding,
because the conserved local byte total already pins their magnitude inside that
window and the star fixture cannot falsify the full-duplex ruling.
The repository-wide fidelity
selector is `PrecisionConfig`; the cross-check switch is a diagnostic and
names no seam level.

The pre-registered
[CORE-5 reduction study](../../examples/core5_reduction/RESULTS.md) drove two
requests through three `ExecutionGraph -> CoarseDeviceRuntime ->
ExecutionResult -> StepResult` steps across dependency shape and 200/400
Gbit/s RNIC rate. All four exact JCT rows, 18 scored behavioral instances and
60 live in-harness structural predicates passed. Two expected validator
rejections and two compatibility accepts are reported as separate unscored
evidence classes. The serial dependency penalty was exactly
10,000 ps at both rates; the 200-to-400 Gbit/s delta was exactly 163,840 ps in
both shapes. Every request component row summed to TTFT or TPOT exactly while
the 21-visit request work sum exceeded wall latency. Asynchronous control and
collective cells advanced the scheduler by 10,000 ps while their physical
quiescence remained 20,971,520 ps. The separately frozen
[Tier B expectations](../../examples/rnic_live_v1/tier_b_expectations.md)
and their
[review supplement](../../examples/rnic_live_v1/tier_b_review_supplement.md)
pinned the raw producer schema, four bypass profiles, both objectively selected
doorbell-owner mappings, and the two-WQE live FIFO relation. The registered
Tier B run then passed all six scored families: D additivity 4/4, inverse-rate
serialization 4/4, live StepResult/TTFT/TPOT forms 8/8, seven-component rows
8/8, FIFO contention 4/4, and bypass artifact identity 4/4. The selected
`nic_owner` mapping put D and network service on NIC attribution while W1's
wait of exactly L stayed in queue attribution. All fatal invariants and
checker-sensitivity controls held. This demonstrated CORE-15's structural
path from a graph to changed completion and live request metrics, its
sole-authority projection, and its explicit bypass artifact guard. Transaction
rollback is separate unit-test evidence in `tests/test_composed_rnic.py`: an
adapter failure consumes neither native observations nor runtime state.

Tier B itself stopped short of one fixed contended graph through both bypass
and composed native authority. The subsequent
[RNIC authority comparison](../../examples/rnic_authority_v1/RESULTS.md)
closed that residual as CORE-21. One canonical graph and `StepRecord` traversed
the timing-neutral ledger and composed native session through the deployed
reducer. Structural minus bypass was exactly +1,000 ps for JCT, prefill TTFT
and decode TPOT at both rates, passing 6/6 signed instances. The independent
rate family passed 12/12, and every live cell recorded a 0/0 native transaction
abort followed by one committed two-WQE retry. The bypass bundle matched
through the repository `BypassArtifacts` comparator after the isolated
link-disabled build. The result ledger quotes and maps every registered
CORE-21 clause; no residual remains.

CORE-28 is complete. Sparse pairwise `CollectiveWork` may carry an optional
request-major partition that sums exactly to the aggregate physical pair table.
Strict validation rejects malformed, unknown, duplicate, noncanonical or
aggregate-inconsistent ownership before rendering. The partition survives the
execution-graph JSON round trip and graph-only GOAL renderer, where a second
fail-closed comparison checks the structured message projection. Empty
attribution remains absent from the wire form, and adding attribution changes
no physical GOAL operation. Six placement and request-count cells plus the
real Granite prefill matched every frozen identity; see
[the CORE-28 results](../../examples/per_request_fidelity_v1/RESULTS.md).

CORE-34 is complete. `RequestRoutingLifetime` is the one mutable request
identity from join through close: it carries opaque join provenance, arrival,
an arena extent, a monotonic unique-token cursor, delayed scheduler finish and
two layer masks. `CompletionReducer` optionally advances its registry only
after graph, result and runtime-report validation. Only subjectless logical
pairwise dispatch/combine completions for the request's final captured token
set end bits; WQE-subject events do not. CLOSED requires full masks, exact
captured coverage and the scheduler finish before the view release callback
runs. The real Granite study closed one and three requests with zero live views
and failed closed for suppressed dispatch layer 7 and combine layer 19; see
[the routing lifetime results](../../examples/routing_lifetime_v1/RESULTS.md).

CORE-35 is complete. Scalar predecessor accounting is gone from the coarse
report. Each operation publishes one conserved critical segment per canonical
participant, and `CompletionReducer` validates the inventory, every segment
completion, every predecessor reference and the realized endpoint chain. The
Granite replay now runs with no barrier tightening: rank 1 of
`step-0:layer-1:rank-1:compute` is admitted from rank 1 of
`step-0:layer-0:ep-combine` rather than rejected against that collective's
slowest rank. Both graph shapes reproduced their frozen result and completion
digests over 25 and 33 executions, agreed on every step boundary and every
completion identity, and differed on exactly 1,305 of 5,760 and 2,553 of 7,680
intermediate timestamps, never with the barrier earlier. A report that declares
rank 0 as the predecessor while keeping the rank-1 boundary is still rejected
atomically, so the graph is not admitted by a weaker check; see
[the participant frontier results](../../examples/participant_frontier_v1/RESULTS.md).

CORE-46 is complete. The scalar fields CORE-35 left as unjoined compatibility
projections are now derived from those same segments and rejected on
disagreement. The six-clause derivation held on all four accepted Granite cells
in both graph shapes, 26,880 operation records with zero errors, and on a
fixture whose collective ranks finish out of rank order and whose successors
split into an additive and a participant-local boundary from one causal
predecessor. Six single-field contradictions that the previous validator
accepted are now rejected atomically, and all four accepted result and
completion digests, execution counts and completion counts are unchanged. The
rank-local frontier records number 1,305 and 2,553, exactly the intermediate
timestamps CORE-35 found moving between the two shapes; see
[the scalar projection results](../../examples/scalar_projection_v1/RESULTS.md).

CORE-8 is partly demonstrated and stays open. Its cross-layer projection half
is now enforced: the completion stream and the request bookkeeper are joined to
the runtime report by stable identity and rejected on loss, duplication or
timestamp disagreement. The clauses held unchanged on 719 runtime reports, 593
reducer inputs and 38,540 completion events before the check existed, so
enforcing them moved no accepted timestamp, digest or completion identity.
Thirteen hand-built contradictions that both consumers accepted are now
rejected, including a `QUEUED` event rendered at logical submission and an
eligibility/grant swap, which a zero-wait visit could never separate. The
registered sweep reproduced its four closed-form job completion times exactly,
with the serialization term scaling as 1/bandwidth and the visibility term
constant. The cross-language half of CORE-8, the native reservation timeline
and the shared golden fixtures frozen in
[queue_contract_v1](../../examples/queue_contract_v1/expectations.md), is not
demonstrated and keeps the task open; see
[the cross-layer authority results](../../examples/cross_layer_authority_v1/RESULTS.md).

CORE-43 is complete. The analytic intra-node endpoint charge and the
`rnic-nn-fluid` manifold were run on the same Granite capture traffic at EP
width eight, all-local and all-remote, over all 48 phases of all 32 recorded
steps at matched rates of 20 and 40 picoseconds per byte. They agree at every
one of 3,072 phase instances inside the preregistered band, with the fluid
manifold exceeding bytes over rate by 0 or 1 picosecond and never more, against
a registered ceiling of one picosecond per directed segment. On the prefill step
at 400 Gbit/s the analytic charge is 511,290,000 ps against 511,262,768 ps of
realized fluid serialization, and the whole 27,232 ps difference is the analytic
model's declared whole-nanosecond GOAL calc quantum. Live step latencies
compose both serializers correctly: the all-remote arm exceeds the all-local one
by the 48 fixed propagation delays in every step at both rates, and the
all-remote path is bit-identical when the analytic bandwidth is changed under
it. The capture-scale effect of the CORE-41 correction on this traffic is a
factor of 1.510 on live TTFT; see
[the endpoint fabric cross-check results](../../examples/endpoint_fabric_crosscheck_v1/RESULTS.md).

CORE-47 is complete. The routing-lifetime study executes the lowerer's graph
unchanged, and the whole-operation barrier is retained beside it as an explicit
comparator that never selects a reported value. Every lifecycle exit,
suppression diagnostic and state trace is retained, and all 58 scheduler-visible
step boundaries agree between the two arms, including the two step-0 boundaries
CORE-35 published. The moved intermediate values reproduced CORE-35's counts
exactly, 1,305 of 5,760 and 2,553 of 7,680, and every one of the 3,858 of them
is the completion of a compute operation admitted from an `ep-combine` frontier
whose participants finished at different times, always later under the barrier
and never earlier. PLAY-13 and CORE-34 were accepted under the barrier
configuration, and that qualification is now discharged; see
[the routing lifetime results](../../examples/routing_lifetime_v1/RESULTS.md).

The first CORE-51 disaggregated-session slice is live through the pinned vLLM
0.27.1 scheduler seam. One eight-rank prefill engine and one eight-rank decode
engine share one virtual clock; the scheduler-side KV connector gates producer
completion and consumer admission while one core handoff event owns the
declared 100 or 200 microsecond transfer. The frozen study passed all four
exact decomposition rows with 0 ps residual and all six behavioral relations.
It also rendered the 16-prefill plus 40-decode target as 448 rank, GPU and NIC
records without running those engines. The live 56-engine target, accepted
lookup pricing, physical packet-handoff topology and validated concurrent
throughput-delay shape remain CORE-52, CORE-53, TRAF-62, TRAF-64, PLACE-5,
VLLM-35 and VLLM-39. The bounded packet handoff itself is live and moves TTFT
by the exact signed difference from the constant arm at 0 ps residual without
moving decode TPOT; see
[the disaggregated-session results](../../examples/pd_session_v1/RESULTS.md)
and [the packet-handoff results](../../examples/pd_session_fabric_handoff_v1/RESULTS.md).
The concurrent extension now conserves 144 independent request lifecycles at
0 ps maximum decomposition residual and exposes stock-scheduler batches in
both roles across all three small pool ratios. Its exact curve records are
live, but the frozen delay direction is refuted in all six curves, so VLLM-35
stays open through VLLM-39; see
[the concurrent-session results](../../examples/pd_session_concurrent_v1/RESULTS.md).
The corresponding SGLang session also conserves 144 request lifecycles and 576
decode tokens with 0 ps maximum decomposition residual, and its packet handoff
moves TTFT by the exact signed -76,918,400 ps difference without moving TPOT.
Its throughput is nondecreasing in 4 of 6 frozen curves, so SGL-33 remains open
through SGL-36. The scored CORE-54 freeze resolves the allocation finding
before its first run: the four-node EP32 prefill and nine-node EP72 decode
disclosures are separate experiments on the 12-node cluster. Their
simultaneous 104-rank render remains a structural comparator and is not called
the 96-GPU system; joint deployments appear only as declared what-if context.
This closes CORE-57 while preserving the original structural render; see
[the scored deployment result](../../examples/deployment_curve_v1/RESULTS.md).

The [second scored CORE-54 run](../../examples/deployment_curve_v1/RESULTS_RUN2.md)
prices prefill through the clean COMP-75 composition and enables SGL-38's
default-off remote-KV decode projection. All four exact candidate keys select
live, so CORE-56 is complete. The independently reproduced COMP-75 authority,
not the preserved void CORE-60 record, supplies the destination deduplication,
FP8 dispatch bytes and max-like overlap used by the run, so CORE-60 is also
complete without promoting that void record. The inherited surcharge fits to
0 ps with zero applications per step. Both priced held-out prefill rows miss
the frozen 5 percent bar: 2K is 5.113992 percent high and 4K is 13.976233
percent high. The shared communication term dominates all three compute rows
and flattens their point capacities at 57,332.324550 tokens per second per
node. Standard decode binds the exact EP72 candidate key but predicts
8,949.759685 tokens per second per node, 59.834128 percent below the published
calibration value. No decode-side mechanism or in-run adjustment is claimed.
CORE-54 remains open on the refuted prefill score, the COMP-72 MTP cell,
COMP-76 decode repetition, CORE-61 depth validity and COMP-74 distributions.

The [third scored CORE-54 run](../../examples/deployment_curve_v1/RESULTS_RUN3.md)
fits the calibration-clean overlap-exposure fraction to its perfect-overlap
floor and applies the independently frozen expert-balance attenuation factor.
The two scorable held-out prefill anchors pass the 5 percent bar at -4.519707
percent for 2K and +3.530310 percent for 4K under the declared benchmark-bias
model. Their unattenuated errors remain published at +5.113992 percent and
+13.976233 percent. MTP remains blocked without numeric access and the
standard-decode calibration miss remains unattenuated at -59.834128 percent.
This scoped prefill pass does not close CORE-54 or any registered residual.

The CORE-53 session binding is live as an explicit content-addressed candidate
path through the existing compute provider chain. The retained candidate row
is selected twice at the exact prior-KV-16 decode shape, its status and partial
coverage remain in request provenance, and every miss delegates to an explicit
roofline comparator. The first acceptance run is void: both record-absent arms
reproduce every accepted KV byte count and timestamp, but their complete
request-result bytes differ in the two vLLM-owned random pool-local request
identifiers. CORE-53 therefore stays open on CORE-58 and COMP-73; see
[the session kernel-cycle result](../../examples/pd_session_kernel_cycle_v1/RESULTS.md).

The frozen CORE-62 roofline replay passes its exact accounting gate at
all 18 points: the residual after the inter-node then intra-node terms is 0 ps
everywhere, kernel simulation is off, and the 43-artifact preservation class
is byte-identical. The new log-log line-and-dot contract, the y-only horizontal
anchor and both figure formats render literally, so CORE-62 is complete. The
TRAF-68 study is still a refutation of its registered bottleneck direction
because the nine-node htsim incast service remains below the H100 roofline over
batch per GPU 1 through 32. The complete gate and refuted map remain published in
[the analytical frontier result](../../examples/deployment_frontier_v1/RESULTS.md).
TRAF-69 and COMP-77 remain reserved because the unexplained residual is zero.

## Open tasks

### Precision

- CORE-53 (Precision; P1; M): replace the first disaggregated session slice's
  roofline bootstrap with the accepted COMP-64 kernel-cycle lookup record.
  Bind each prefill and decode step to the lookup key and provenance already
  owned by the compute provider chain, then rerun the frozen prompt-length and
  handoff-cost grid. Acceptance requires exact lookup-record selection, the
  signed TTFT and TPOT movement predicted from the selected rows, and an
  explicit roofline comparator that preserves this slice's accepted bytes and
  timestamps when the lookup record is absent. This task depends on COMP-64
  and does not create another pricing model. The candidate binding and exact
  signed movement are implemented, but the frozen acceptance run is void on
  complete request-result byte identity and the record covers only one decode
  shape. CORE-58 owns the identity-proof boundary and COMP-73 owns the missing
  target-record coverage; this task stays open on both.

- CORE-58 (Precision; P1; S): repair the CORE-53 record-absent identity
  acceptance boundary after its frozen run was voided solely by the fresh
  random suffixes in vLLM's prefill and decode pool-local request identifiers.
  Before rerunning, freeze either a canonical comparison projection that
  retains every pricing-relevant and client-visible field, KV byte count and
  timestamp while excluding only those two opaque identifiers, or a stable
  identifier mechanism that does not change the accepted off path. Acceptance
  requires two independent native sessions to match byte for byte on the
  frozen boundary, an exact diagnostic proving that the unprojected results
  differ only in the declared opaque fields, and unchanged accepted
  `pd_session_v1` compact cells. The void run remains void and cannot be
  retrospectively rescored.

- CORE-61 (Precision; P1; M): validate whether the measured four-layer EP72
  standard-decode basis may be extrapolated linearly to DeepSeek-V3's declared
  61-layer depth. Freeze at least two measured depths and the exact batch-32,
  remote-KV-2000 shape before comparison; retain the candidate key, per-layer
  work inventory and every fixed component separately. Acceptance requires a
  held-out depth prediction within 5 percent of its measured service and a
  signed residual ledger that distinguishes depth scaling from the finite
  compute/communication overlap owned by TRAF-66. The published 22,282
  tokens-per-second calibration value is comparison evidence only and may not
  tune the depth rule. COMP-76 owns independent repetition of the four-layer
  basis, while COMP-72 owns the missing MTP cell.
  The [CORE-61 local derivation](../../examples/deployment_curve_v1/core61_depth_result.md)
  separates the retained step into 489 ps fixed plus 1,875,679,511 ps
  repeatable across four layers. Its corrected 61-layer declaration is
  28.604113032 ms, only 6.96825 ns below the 28.604120000 ms linear rule, so
  the fixed-component hypothesis has the expected sign but a materially null
  magnitude. The result remains a `DECLARED` derivation from one `MEASURED`
  service decomposition with `DISCLOSED` component attribution. CORE-61 stays
  open with a frozen 3.751359511 ms prediction for the held-out eight-layer,
  batch-32, remote-KV-2000 cell. That measurement joins COMP-72's resumable
  Merlin remainder; COMP-76 is unchanged and CORE-63 remains reserved.

- CORE-48 (Precision; P1; M): give the cross-node coarse RNIC path a
  destination-ingress serializer. Semantic sends serialize per source RNIC and
  nothing at the receiver, so an all-remote many-to-one combine completes at
  the maximum single-source egress rather than at a contended arrival. The
  TRAF-14 qualification could therefore only report its converging four-rank
  combine as structural evidence: 100 ps at 400 Gbit/s and 200 ps at
  200 Gbit/s are the largest single extent, not a physical oracle. Identify the
  correction from explicit per-endpoint byte ledgers, sweep payload and fan-in
  across dispatch-star, combine-star and symmetric all-remote fixtures, and
  require exact byte conservation, the preregistered ingress-bound increase and
  its live TTFT and TPOT effect, while symmetric and single-source cases keep
  their accepted timestamps. Scope boundary: CORE-41 owns the analytic
  intra-node routed service and must preserve all-remote timestamps exactly.
- CORE-8 (Precision; P1; L): establish the cross-layer authority and
  queue-visit contract above before residual-driven calibration. Define one
  loss-checked projection from each authoritative runtime object into
  `CompletionEvent` and `RequestBookkeeper`. Use an existing closed
  `ResourceKind` when it faithfully names the projected outer resource, and
  consume CORE-50's strict registered-device reference only for an
  otherwise-unrepresentable outer resource owned by a compact device service,
  without interpreting its compute-owned registry or converting an internal
  service axis into a queue visit. Keep language-specific mechanisms
  behind the same contract: the native side extracts a protocol-neutral exact
  reservation
  timeline and finite-capacity resource from the PCIe implementation, while
  Python uses a reference serial and capacity resource for GPU and runtime
  queues. Mandatory protocol rules stay in their owning adapters. Shared
  golden fixtures must prove isolated zero wait, one external contention wait
  without triangular self-charging, predecessor service excluded from queue
  wait, finite-capacity release, overflow rollback and identical reductions in
  both languages. Existing PCIe, WQ and GPU studies must remain byte-identical
  under identity arbitration before any non-identity class policy is enabled.
- CORE-11 (Precision; P1; L): replace CORE-4's whole-operation exclusive HBM
  reservation with calibrated kernel-versus-DMA shared-bandwidth service. The
  current surrogate serializes an entire HBM-using kernel against an entire
  copy descriptor, even when measured issue and transfer phases could share
  bandwidth. Identify service from simultaneous kernel/copy traces with HBM
  byte and throughput counters. Acceptance must freeze a compute-duration,
  copy-size and bandwidth sweep, change graph JCT into the measured bands, and
  retain the zero-HBM independent-lane baseline and identity path exactly.
- CORE-12 (Precision; P1; M): admit a kernel that becomes legal while a
  concurrent kernel batch is already active. The first coarse runtime freezes
  the co-runnable set at dispatch and waits until batch completion before a
  later arrival can enter. This task is the sole owner of transactional
  later-arrival admission into an active device engine; COMP-25's complete-tuple
  batch service does not close it. Identify admission and completion offsets
  from a reproducible multi-stream trace. Acceptance must vary arrival offset
  and residency pressure, match the observed overlap bands, preserve the
  simultaneous-arrival and single-kernel baselines exactly, and prove that
  resolution, feasibility, dispatch, advance or prepare failure aborts without
  callbacks or mutation of live engine, arbitration, runtime or bookkeeping
  state. Successful adoption follows one fixed infallible order after every
  participant prepares: device engine, arbitration policy, runtime state, one
  bookkeeping batch, then callbacks. Consume the compute-owned transaction's
  pure `admissible`, timestamped mutating `dispatch_granted`,
  `peek_next_event_ps`, typed `advance`, compute-owned `release_held`, read-only
  `accounting`, and `prepare`/`commit`/`abort` capabilities;
  drain equal-time events to a finite fixed point. The device-engine transaction
  is quiescent when runtime requests its physical closure; this does not require
  unrelated network or background work to drain and does not change the
  separate framework-completion or optional physical-quiescence boundary.
- CORE-13 (Precision; P1; L): replace the flat per-endpoint intra-node
  NVLink-class serializer with calibrated compute-owned NCCL/NVLink service.
  The current surrogate uses payload bytes and one configured rate; it does
  not replay the network kernel, HBM reads or link/topology selection. Use
  captured NCCL kernel traces plus NVLink byte/rate observations. Acceptance
  must vary payload, participant count and competing kernel demand, change
  end-to-end graph JCT into the measured bands, and retain the explicit
  cross-node RNIC path exactly. The
  [A100 hardware envelope](../../examples/a100_hardware_envelope_v1/RESULTS.md)
  now supplies the first-party observations this task names, on a 4-GPU
  A100-SXM4-80GB `NV4` mesh under NCCL 2.31.2. Copy-engine peer transfers hold
  94.0 percent of the 100 GB/s per-pair wire rate uniformly across all twelve
  ordered pairs, and a device fan-out reaches 281.65 GB/s, 2.995 times one
  pair, so the three link groups of a GPU compose without interference. Ring
  all-reduce holds 72.8 percent of per-GPU egress at width 2 and 71.0 percent
  at width 4, giving 72.77 and 212.89 GB/s of bus bandwidth: widening the
  collective multiplies bus bandwidth by 2.925 because a two-rank ring reaches
  only one pair's four links while a four-rank ring set reaches all twelve.
  One configured flat rate therefore cannot represent this node, and the
  current 450 GB/s `DEFAULT_NVLINK_BANDWIDTH_BYTES_PER_SECOND` is 1.598 times
  the measured per-GPU egress. Under a concurrent BF16 GEMM the collective
  grows 1.481 times and the GEMM 1.161 times, with a makespan 0.839 of the
  serial sum, which is the competing-kernel evidence the acceptance requires.
  A second architecture now separates what transfers from what does not. The
  [GH200 hardware envelope](../../examples/gh200_hardware_envelope_v1/RESULTS.md)
  measured a 4-GPU GH200 `NV6` mesh with the identical sweep. Ring efficiency
  against a GPU's own link ceiling is 74.9 percent there against 71.0 percent
  on the A100, and the width-2 to width-4 bus bandwidth scaling is 2.926
  against 2.925, so both are far more stable across a link generation than any
  rate is. The rates are not: per-pair copy-engine efficiency falls from 94.0
  percent on NVLink3 to 88.8 percent on NVLink4, and per-GPU egress rises from
  281.65 to 398.71 GB/s against nameplates of 300 and 450 GB/s. The 450 GB/s
  flat surrogate is exactly the Hopper per-GPU NVLink4 payload nameplate and
  exactly 1.5 times the Ampere one, so it is a machine identity rather than a
  portable intra-node rate. A calibrated
  service should therefore carry a per-architecture link ceiling and a shared
  ring efficiency, not one bandwidth. The task stays open because no runtime
  composition landed and no graph JCT moved.
- CORE-26 (Precision; P1; L): replace the cross-node collective path's current
  independent GPU-versus-RNIC surrogate with one runtime composition of the
  GPU-resident NCCL task and the existing WQE/NIC authority. Consume the
  resource demands calibrated by COMP-22 and compose with CORE-13 and COMP-11
  rather than adding a second SM, HBM or NVLink scheduler. Version 1 consumes
  the exact `CollectiveDeviceRankFrontier` barrier: one resolved resident stage
  per plan rank, device grant releasing that rank's copied entry actions, and
  a compute-owned residency lease released at the maximum of internal device
  work finish and the copied traffic-terminal frontier. The stage charges only
  GPU-resident SM/HBM demand; existing traffic remains sole chunk and port
  timing authority and supplies its terminal timestamp read-only. The
  composite stage visit finishes and completes at lease release, while the
  internal work-finish-to-release interval is occupancy evidence rather than an
  additive latency term. The live path consumes CORE-12's incremental
  external-frontier transaction: device work finish is an explicit service
  event, traffic supplies a read-only terminal timestamp, and the compute
  transaction alone returns the final fact and releases the reservation.
  Offline capture and fitting do not wait for CORE-12. The
  semantic collective emits one completion at the maximum across ranks; no
  stage emits another graph completion. Reject a multi-stage rank until capture
  identifies its stream dependencies. Sweep payload,
  participant count, channel count and compute-neighbor pressure across the
  crossover; require TTFT and TPOT to enter the measured overlap bands and
  reconcile every GPU and network byte exactly. Zero GPU demand and disabled
  composition must preserve every accepted TRAF-7 timestamp and artifact byte.
- CORE-27 (Precision; P1; L): add only the data-mover resources that COMP-22
  observes on the cross-node NCCL path, including copy-engine or GPUDirect DMA
  visits when present, plus their shared-HBM interaction and downstream
  visibility. The current surrogate charges no such visit. Identify eligibility,
  grant, release and consumer-visible completion from a reproducible concurrent
  capture; vary transfer size, direction and competing copy pressure and match
  held-out queue wait and JCT within the declared measurement band. An observed
  no-copy path must stay explicitly zero, and disabling this mechanism must
  preserve the CORE-26 baseline exactly.

### Completeness

- CORE-3 (Completeness; P1; L): widen the KV lifecycle case matrix, sweep and
  reporting surface. The accounting itself has landed: `KvLifecycleLedger`
  consumes all thirteen vocabulary members before resource contention, enforces
  the allocation, ownership, reference-count, capacity and byte-conservation
  invariants, and reaches TTFT and TPOT through the HBM queue with both off
  paths preserved exactly; see
  [the KV lifecycle study](../../examples/kv_cache_strategies/RESULTS.md),
  which registered and demonstrated no reuse, repeated system prefixes,
  capacity pressure, eviction and preemption/recompute over a six-level
  capacity sweep at two HBM rates. What remains, all still under this ID
  because its own entry already registered it: the SGLang cases, and the vLLM
  cases for competing prefix pools, multi-turn sessions, chunked prefill,
  mixed contexts and bursts; sweeps over block size, arrival rate, length,
  sharing and concurrency; and reporting for fragmentation, eviction age, a
  preemption counter, capacity wait and TTFT/TPOT tails. The observation
  streams must come from the adapter capture halves VLLM-11 and SGL-9 rather
  than from the study's own vLLM-policy fixture. Acceptance keeps the frozen
  cells of the landed study bit-identical. One further lowering divergence
  remains under this ID: `SWAP` and `TRANSFER` are byte-carrying and currently
  served only from the rank's HBM queue by `_schedule_kv_traffic`, while
  `RECOMPUTE` is accounted as tokens and is not lowered at all. Complete the
  architecture contract by lowering swap and remote movement to DMA plus
  network work, and recompute to compute plus a KV write. Acceptance must
  preserve the zero-byte and absent-action paths exactly and carry declared
  bytes and tokens through the runtime report and TTFT/TPOT projection without
  loss or duplication.
- CORE-9 (Completeness; P1; M): replace the bookkeeping-v1 WQE compatibility
  shape with a versioned structural projection while retaining a strict v1
  reader. A send WQE names one local SQ and send CQ, a receive WQE names one RQ
  or SRQ and receive CQ, and RX matching is a later relation. Represent
  transport retirement, SQ reclamation, optional CQE visibility and CQ poll as
  distinct facts so an unsignaled success never fabricates a CQ completion.
  Preserve the native session, endpoint, WQ kind, WQ identity and post
  sequence as the conservation key; WR ID, GOAL flow ID and backend tokens are
  explicit aliases only. Acceptance covers send, RQ receive, SRQ receive,
  one-sided no-receive and unsignaled no-CQE records plus v1 round-trip
  compatibility.
- CORE-14 (Completeness; P2; L): generalize `CoarseDeviceProfile` beyond the
  fixed eight GPUs, eight affine RNICs and arithmetic rank mapping. Consume the
  repository placement and fabric manifests through the existing schemas,
  reject incomplete mappings before runtime state mutates, and keep the fixed
  profile as an explicit off path. Enabling manifest discovery with the
  equivalent eight-by-eight mapping must preserve every accepted CORE-4 event,
  WQE, byte count and timestamp exactly.
- CORE-16 (Completeness; P2; M): replace CORE-4's fail-closed collective and
  control expansion limits with exact ring remainder chunking and a collision-
  free tag allocator wider than 1,024 control destinations. The present off
  path rejects zero, sub-rank and non-divisible ring payloads, control fanout
  above 1,024, and collective tags reaching the control range before any
  authority mutates. Acceptance must cover remainder byte conservation and
  tag uniqueness across adjacent collective and control operations while the
  current divisible-payload and bounded-fanout baselines remain byte-identical.
- CORE-17 (Completeness; P1; M): populate
  `StepRecord.sampled_request_ids` from vLLM and SGLang whenever an exact
  `num_sampled` is a strict subset of the scheduled batch. The current CORE-5
  reducer accepts explicit identities, infers the decode-only subset when that
  is uniquely determined, preserves the absent-count all-scheduled legacy
  approximation, and rejects an ambiguous partial prefill subset instead of
  assigning TTFT arbitrarily. Acceptance must use a mixed chunked-prefill,
  completed-prefill and decode batch, match the framework's actual token
  production mask request by request, and preserve zero-sample, all-sample and
  legacy wire behavior exactly.
- CORE-29 (Completeness; P2; M): support explicit early and background
  graph completion boundaries separately from full backend quiescence. The
  checked GOAL projector currently rejects any completion set other than the
  full terminal-operation set before writing an artifact. Acceptance must
  preserve the selected framework completion while later physical work drains,
  reject a boundary that cannot be represented exactly, and keep the omitted
  and full-terminal baselines byte- and timing-identical.
- CORE-30 (Completeness; P2; M): realize participant-local readiness for
  asynchronous `ControlWork` destination ranks. `CoarseDeviceRuntime` currently
  rejects this unsupported graph shape before scheduling, while synchronous and
  supported asynchronous control paths remain accepted. Acceptance must gate
  each destination on its exact local predecessor frontier, conserve the
  selected causal timestamp through reporting and completion reduction, and
  preserve every accepted control byte and timestamp when the path is absent.
- CORE-32 (Completeness; P2; L): model optional framework or server admission
  control after arrival eligibility, including rejection, rate limits,
  concurrency caps and policy-driven deferral, without duplicating the
  framework scheduler. The disabled policy must preserve the arrival-gated
  baseline exactly, and policy queue time must remain distinct from arrival
  gating and scheduler queue time.
- BRIDGE-2 (Completeness; P1; L): implement the online stateful co-simulator
  client using the delivered strict full `StepResult` codec and an HTSIM
  persistent flow session extended with HTSIM-28. BRIDGE-2 is blocked behind
  HTSIM-28 because the delivered HTSIM-18 protocol cannot express the exact
  completion-to-dependent-injection boundary; see
  [the protocol audit](../../examples/congestion_chain_v1/RESULTS.md). The
  backend foundation retains one event list, topology,
  native RNIC authority and transport policy across flow injections; its
  frozen study demonstrated byte-identical stateless-equivalent latencies,
  discriminating retained queue state and lower wall time. The remaining
  client must lower live `ExecutionGraph` dependencies into flow injections
  and inclusive virtual-time horizons, translate the returned native
  lifecycle projections into canonical `CompletionEvent` values, append the
  exact object, stage and completion facts at the supplied bookkeeping cursor,
  construct `ExecutionResult`, reduce the full `StepResult`, and publish only
  after all identities, cursors, timestamps and quiescence evidence validate.
  A proposed
  `simllm-cosim-session-v1` uses the same length-prefixed canonical JSON frame
  rule as the backend flow session. Handshake frames select the exact backend
  session and authority. Each input frame carries a contiguous sequence,
  canonical `ExecutionGraph`, source `StepRecord` and starting bookkeeping
  cursor.
  Output event frames carry canonical `CompletionEvent` values and the exact
  append batch of object, stage and completion facts. A compact-device session
  emits the canonical `simllm-run-provenance-v2` record in a provenance frame
  before the terminal frame. The terminal carries `ExecutionResult`, strict
  `simllm-step-result-v2` on the compatibility path or
  `simllm-step-result-v3` plus its verified provenance reference when a compact
  device model is selected, ending ledger cursor and physical quiescence
  separately from framework completion. Reject loss,
  duplication, cursor disagreement, graph/event identity disagreement and
  timestamp regression before publishing a result. The explicit diagnostic
  and BRIDGE-1 prepared modes remain the identity off paths and must preserve
  every accepted byte and timestamp when the online session is disabled.
- CORE-44 (Completeness; P2; M): route the workload and framework seam
  spellings through `PrecisionConfig`. The other six seams resolve from a
  spelling some component can observe, but there is no combined workload
  selector anywhere, and the framework level is chosen by which entry point a
  deployment starts rather than by any record. A run therefore cannot derive
  those two levels from its own components and must name them explicitly. Add
  an observable selector for each, resolve it through
  `check_precision_selection`, and keep every current spelling byte-identical.
- CORE-45 (Completeness; P1; M): emit device-model provenance from a live run.
  `simllm-run-provenance-v1` round-trips and the precision surface study stamps
  its own result, but no sink or backend run writes one. Add the strict
  `simllm-run-provenance-v2` device-model selection tuple and the required
  provenance reference in `simllm-step-result-v3`. Bind every selected model
  ID and hash, acceptance status, target basis, operating envelope and total
  resolved-device-binding-closure digest to the source graph and complete precision
  configuration before publishing TTFT or TPOT. Reject an incomplete model
  selection, more than one model for one device instance, missing closure
  digest or provenance hash disagreement. Heterogeneous device instances may
  select different models; aggregators group compatible provenance or reject
  cross-envelope aggregation rather than erasing that tuple. The strict v1
  provenance reader, strict v2 step-result
  reader and every accepted v1/v2 byte remain unchanged; the explicit path
  with no live provenance continues to emit v2 rather than an empty v3 record.
- CORE-50 (Completeness; P1; M): extend completion and bookkeeping resource
  references with the strict registered-device wire variant defined above.
  Add the closed in-memory `ResourceReference` union without widening
  `ResourceKind`.
  Keep core-owned `ResourceKind` and `LatencyOwner` vocabularies closed;
  validate a supplied registry SHA-256, registered device kind and axis,
  concrete device and resource instances, and required latency owner without parsing the compute-owned demand
  or capacity schema. Emit the new variant only through
  `simllm-completion-event-v2`, `simllm-execution-result-v2` and
  `simllm-request-bookkeeping-v2`; reject an unknown registry or kind before
  publishing any projection. Shared fixtures must prove loss-free CORE-8
  projection, exact latency attribution, rejection of service-axis strings as
  queue resources and unchanged strict v1 readers and canonical bytes.
- CORE-51 (Completeness; P1; L): run the disaggregated serving session toward
  the 40 decode plus 16 prefill node target: separate prefill-pool and
  decode-pool instances of a real frontend over simulated GPUs in one driven
  session, each pool's role declared per instance, joined by the
  prefill-to-decode KV handoff priced through TRAF-61 (a declared constant
  transfer is the explicit first arm), placed by PLACE-4's manifests, and
  reduced to per-request TTFT and TPOT through the existing runtime chain
  with no second authority. Audit the pinned frameworks' own disaggregation
  seams (the vLLM KV-connector surface first) and prefer the real seam where
  reachable, with a driver-level join as the disclosed fallback. Compute
  prices from the kernel-cycle lookup record or its roofline bootstrap;
  intra-node collectives ride the declared constant arm; inter-node work
  charges the declared PCIe submission constant and then the packet
  simulator. First slice runs one prefill node plus one decode node of eight
  simulated GPUs each; the full 448-rank target scales through the same
  session with a stated engine-count feasibility bound. The first slice is
  delivered through the real vLLM scheduler-side KV connector and a shared
  virtual clock, with the declared-constant handoff, role-aware manifests and
  one-plus-one live run. This umbrella remains open on CORE-52, CORE-53,
  TRAF-62, TRAF-64, PLACE-5 and VLLM-35; those residuals own the live 448-rank
  scale, lookup pricing, topology-qualified packet handoff, physical target
  topology and validated concurrent curve respectively.
- CORE-54 (Completeness; P1; L): reproduce the public DeepSeek-V3 deployment
  curve inside the simulator, evidence first. Freeze the published anchors
  from [the deployment disclosures](../papers/deepseek-deployment-disclosures.md)
  before any run, then drive the disaggregated session as the disclosure's
  separate 4-node EP32 prefill and 9-node EP72 decode experiments, with 8 GPUs
  per node and one-shot requests at the disclosed input lengths, sweeping
  offered load so each configuration traces a curve of aggregated output
  throughput against per-token request delay, plotted with the upper-right
  corner optimal. Compute prices from the DeepSeek per-rank lookup
  projection on the Hopper-anchored campaign tables; intra-node collectives
  and PCIe submission ride the declared constants under the dossier's
  calibration policy (fitted on a declared anchor subset inside physically
  justified envelopes, scored on the held-out anchors); the fabric and the
  prefill-to-decode KV transfer ride the packet simulator through TRAF-61.
  Acceptance: the simulated curve within 5 percent of every held-out
  published anchor, error bars propagated from the calibrated component
  uncertainties, and a second legend carrying DeepSeek's own H800
  production profile and declared what-if configurations including the
  16-prefill plus 40-decode target. Depends on COMP-67's column, CORE-52
  and CORE-53's concurrent scaled session, SGL-33's SGLang-side session,
  TRAF-61 and the GH200-anchored campaign tables; a bar this task cannot
  meet is reported as a refutation with findings, never absorbed by
  loosening the frozen anchors. The first scored run is honestly REFUTED:
  priced held-out prefill errors are 69.20% at 2K and 63.35% at 4K, and the MTP
  anchor is BLOCKED on COMP-72's absent cell. The figure, propagated component
  bands, separate-experiment allocation and second legend are published. The
  second scored run applies the clean COMP-75 composition and selects all four
  exact candidate keys, but still REFUTES the priced prefill scope at 5.113992
  percent for 2K and 13.976233 percent for 4K. MTP remains BLOCKED and the
  disclosed decode calibration row misses 59.834128 percent low. The third
  scored run fits the calibration-clean overlap exposure to `f = 0` and passes
  both scorable held-out prefill anchors under the frozen benchmark-bias model:
  2K is 4.519707 percent low and 4K is 3.530310 percent high. The corresponding
  unattenuated errors remain 5.113992 percent and 13.976233 percent high. MTP
  was still BLOCKED without numeric access and the decode calibration residual
  remained unattenuated. The fourth scored run removes that pricing blocker
  and scores the final numeric held-out anchor once. Its frozen unattenuated
  EP72 MTP prediction is 8,253.338082 tokens per second per node against
  17,373 published, or 52.493305 percent low, so physics-only,
  physics-plus-boundary and physics-plus-boundary-plus-attenuation all REFUTE
  the 5 percent bar. No admissible decode factor exists. The run-3 2K and 4K
  PASS rows remain byte-identical and unrescored, making the combined
  three-anchor verdict REFUTED. The deployment-frontier figure stays
  byte-locked because its v2 contract has no MTP marker slot. CORE-54 stays
  open on COMP-76's decode calibration reproduction, COMP-74 distribution
  propagation now partially unlocked by two retained observations, COMP-78's
  Granite campaign arm, CORE-61 depth linearity, SGL-36 and TRAF-64.
- CORE-52 (Completeness; P1; L): run the live 16-prefill plus 40-decode target
  through the same disaggregated session with 448 simulated workers. Retain
  every engine simultaneously, route requests through every declared pool
  instance, and report measured resident memory, construction time, request
  throughput and virtual-time conservation per added engine. The one-plus-one
  session and manifest-only 448-rank render are the explicit smaller off path
  and must remain byte-identical. If the integration host cannot retain all 56
  engines, keep this task open and report the measured stopping point rather
  than extrapolating a pass. Depends on VLLM-35 and PLACE-5.
