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
  approximation.
- `StepResult`: the scheduler-facing result (step latency and completion time
  on the virtual clock).
- `RequestPhase`, `ScheduledRequest`: the per-request vocabulary.
- Closed-loop wire schemas: `atlahs-closed-loop-step-v1` and
  `atlahs-closed-loop-result-v1` are the JSON forms of `StepRecord` and
  `StepResult` exchanged with the simulator per scheduler step.
- Record serialization both ways: `step_record_to_json`,
  `step_records_to_json`, `write_step_records`, `step_record_from_json` and
  `step_records_from_jsonl`.

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
- `simllm-completion-event-v1`: a timestamped `CompletionEvent` at submitted,
  queued, started, progress or completed phase, optionally attributed to one
  concrete `ResourceRef`. `subject_object_id` identifies the WQE or other
  created object when one operation expands into several runtime objects.
- `simllm-execution-result-v1`: an `ExecutionResult` containing the graph
  completion time and its ordered completion-event stream. A separate optional
  quiescence time distinguishes framework-visible completion from all physical
  work draining.
- `simllm-request-bookkeeping-v1`: an append-only `BookkeepingLedger` of
  request-stage transitions, created-object records and the same
  `CompletionEvent` objects returned by the runtime. `RequestBookkeeper`
  assigns sequence numbers, validates lineage, registers a graph without
  mutating it, and queries by request, execution or object ancestry.

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
`AtlahsWqeLedger` is not constructed in this mode. Until the structural path is
live, a run may explicitly select `AtlahsWqeLedger` as its sole timing-neutral
bypass authority. A run never enables both mutable authorities, and every
projection must conserve identity, cardinality and timestamps available at its
boundary.

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

Two reductions remain deliberately separate. `sum_visit_wait_ps` sums work
waiting across resource visits and may exceed elapsed time when visits overlap.
`critical_path_queue_ps` includes only waits on the realized dependency and
resource critical path. Only the latter can participate in an additive TTFT,
TPOT or JCT decomposition. GPU wall-idle classifications, PCIe transaction
wait sums, WQ stage waits and last-completion makespans retain distinct names;
no caller may compare or add them merely because each has units of time.

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

The pre-implementation queue conformance expectations were first frozen in
[examples/queue_contract_v1](../../examples/queue_contract_v1/expectations.md)
at commit `65b5609`; commit `facb26d` clarified identity-policy scope, and
commit `947399c` records the final pre-run state.

The graph carries five typed work payloads rather than one unstructured
dictionary:

| Payload | Semantic owner | Runtime lowering |
|---|---|---|
| `ComputeWork` | Model runner plus `ComputeProvider` | launch queue, CUDA stream, GPU work queue, hardware scheduler and HBM |
| `KvCacheWork` | Real framework KV/prefix-cache manager | logical state transition; READ/WRITE becomes HBM work, SWAP/TRANSFER becomes DMA, RECOMPUTE becomes compute plus WRITE |
| `DmaWork` | Data-mover planner | DMA descriptor, directional copy engine, source/destination memory queues |
| `CollectiveWork` | Framework/NCCL observation plus traffic planner | NCCL channels, DMA/HBM work, WQEs and network flows |
| `ControlWork` | Framework or runtime controller | synchronous or asynchronous labeled local/fabric control path; priority only under an opted-in policy |

`CollectiveWork.payload_bytes` is algorithm-relative, and consumers must
branch on `(collective, algorithm_hint)`: a ring `all-reduce` carries the
full reduced payload, while a `pairwise` `all-to-allv` carries the bytes each
rank sends to each other rank (one uniform ordered-pair share; the serial
lowerer and the serial GOAL renderer agree on this decoding). An all-to-allv
with per-pair size variation, e.g. captured routed-expert dispatch, is not
representable by the single scalar; CORE-6 owns the contract extension.

`ExecutionLowerer` and `DeviceRuntime` remain narrow protocols.
`SerialStepLowerer` implements the diagnostic compatibility schedule and
`render_serial_execution_graph_goal` replays its supported subset using only
the JSON-round-tripped graph. The lowerer places distributed sequencing in
`participant_local_depends_on`; the renderer rejects operation-scoped
cross-rank barriers instead of weakening them. The backend driver's GOAL
completion summary participates in schedule JCT even when earlier WQE rows
exist, so compute-only TP=1 graphs and graphs with trailing compute are both
covered. A runtime can receive an optional central bookkeeper; the coarse
resource scheduler itself remains CORE-4. Current adapters continue to use the existing
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
completion to the native RNIC session owned by BACK-8, BACK-9 and BACK-12. It
must call the compute service model rather than grow a second SM or SASS model
in `simllm.core`. The compute slice therefore does not claim whole-task
execution timing or compute/copy overlap.

## Status

Step records and the virtual clock (`VirtualClock`: heap-ordered events,
monotonic picosecond time and deterministic tie-breaking) are implemented and
tested; CORE-1 closed with M1. The step-record JSON readers landed with the
M4 first slice, which also exercised the step schema for real: recorded M2/M3
smoke JSONLs load, round-trip and replay through `HtsimStepSink`.

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
rows are separately validated surfaces. CORE-4 owns their concrete
graph-operation/tag/WQE correlation.

Actual framework observation producers remain VLLM-11/12 and SGL-9/10.
Explicit KV state semantics remain CORE-3. Non-RNIC device-resource arbitration
and projection of NCCL, SQ/RQ/CQ, WQE and transport correlation records remain
CORE-4; BACK-8, BACK-9 and BACK-12 own structural RNIC objects and arbitration.
Completion reduction and tail attribution remain CORE-5.

The trace-driven GPU service slice establishes the intra-kernel scheduler,
SM, residency and HBM mechanisms plus isolated copy-descriptor service in
`simllm.compute`. Its synthetic study validates those component laws only.
It does not close CORE-4, because no inter-operation resource runtime or
whole-graph overlap policy is added by that slice.

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
   dependency-reachable work may move. CORE-10 owns priority-caused movement.
4. **Identity arbitration is the exact off path.** Run each shared resource
   with omitted class arbitration and with the explicit identity policy, then
   permute class labels without changing arrivals or service demand. Event
   order, every timestamp, all wait and byte counters, random draws and final
   JCT must remain byte-identical. A separately enabled priority policy may
   change only the order of simultaneously legal ready requests.

## Open tasks

- CORE-3: implement explicit KV lifecycle accounting before resource
  contention. Consume adapter observations for RESERVE, ALLOCATE,
  BIND_PREFIX, TOUCH, READ, WRITE, RETAIN/RELEASE, EVICT, FREE, SWAP,
  TRANSFER and RECOMPUTE. Enforce allocation, ownership, reference-count and
  byte-conservation invariants. Add `examples/kv_cache_strategies/` with
  pre-registered vLLM and SGLang cases: no reuse, repeated system prefixes,
  competing prefix pools, multi-turn sessions, chunked prefill, capacity
  pressure, eviction, preemption/recompute, mixed contexts and bursts. Sweep
  capacity, block size, arrival rate, length, sharing and concurrency; report
  live/reserved/reclaimable bytes, fragmentation, hits, eviction reason and
  age, reads/writes, transfers, recompute, preemption, capacity wait and
  TTFT/TPOT tails. Adapter capture halves are VLLM-11 and SGL-9.
- CORE-4 (Completeness; P1; L): implement the first coarse `DeviceRuntime`:
  framework launch FIFO,
  per-CUDA-stream FIFO and event dependencies, non-preemptive GPU work queue
  and selection of the co-runnable task set dispatched into the
  `simllm.compute` kernel service model, shared kernel-versus-DMA HBM
  arbitration, directional copy-engine selection and queueing for explicit
  DMA descriptors, NCCL channel queues, GPU-affine RNIC selection and semantic
  submission, completion-event/projection plumbing and a control queue whose
  class is accounting-only under identity. Start with the fixed eight-GPU,
  eight-RNIC node above. Expose the CORE-8 policy seam and use only identity;
  CORE-10 owns non-identity policies. In bypass mode delegate SQ/RQ/CQ and WQE
  state to the sole `AtlahsWqeLedger` authority. In structural mode delegate
  WQE lifecycle, WQ/CQ state, RNIC arbitration and completion to the
  BACK-8/BACK-9/BACK-12 native session. Do not duplicate the SASS scheduler,
  SM-residency or isolated
  copy-service mechanisms owned by `simllm.compute`. Append concrete NCCL
  command, SQ/RQ/CQ, WQE and QP/link-pair
  projections to `RequestBookkeeper`; the timing-neutral immediate-CQ and
  identity-only RQ behavior remains a compatibility path until BACK-9 supplies
  structural WQ/CQ service through the SimLLM RNIC extension. The native path
  must be live-reachable from an `ExecutionGraph` to a changed completion time;
  its standalone probes do not close this task. Preserve
  graph operation identity through collective expansion and rendered GOAL tags
  so backend WQE rows correlate without inference. Run and defend all
  pre-registered experiments above.
- CORE-5 (Completeness; P1; L): implement completion feedback and tail
  attribution. Stream queue,
  start, progress and completion events, reduce the required completion
  boundary to `StepResult`, advance `VirtualClock`, and export per-request
  TTFT/TPOT plus queue-, KV-, kernel-, DMA-, collective-, NIC- and control-
  attributed components. Export additive visit totals separately from the
  realized critical-path decomposition; only the latter must conserve
  end-to-end latency. Support synchronous waits and asynchronous control or
  collective progress. Preserve v1 readers if the queue-visit projection needs
  a versioned event extension.
- CORE-6: represent variable per-pair all-to-allv sizes in the graph
  contract. `CollectiveWork.payload_bytes` carries one uniform ordered-pair
  share for `pairwise` all-to-allv, so a captured, non-uniform dispatch
  (routed experts under real gating) cannot be expressed. Decide between an
  optional per-pair size table on the collective payload and a schema bump;
  the uniform scalar form must stay readable either way. Coordinate with the
  TRAF-2 capture half so traffic expansion and the renderer consume the same
  representation.
- CORE-8 (Precision; P1; L): establish the cross-layer authority and
  queue-visit contract above before residual-driven calibration. Define one
  loss-checked projection from each authoritative runtime object into
  `CompletionEvent` and `RequestBookkeeper`; use a versioned bookkeeping or
  completion-event extension only where v1 cannot represent that projection
  without ambiguity. Keep language-specific mechanisms behind the same
  contract: the native side extracts a protocol-neutral exact reservation
  timeline and finite-capacity resource from the PCIe implementation, while
  Python uses a reference serial and capacity resource for GPU and runtime
  queues. Mandatory protocol rules stay in their owning adapters. Shared
  golden fixtures must prove isolated zero wait, one external contention wait
  without triangular self-charging, predecessor service excluded from queue
  wait, finite-capacity release, overflow rollback and identical reductions in
  both languages. Existing PCIe, WQ and GPU studies must remain byte-identical
  under identity arbitration before any non-identity class policy is enabled.
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
- CORE-10 (Completeness; P2; M): add non-identity arbitration policies only
  after CORE-8 establishes the policy seam and exact identity baseline. Start
  with strict priority and weighted round robin over legal ready candidates;
  keep per-SQ ordering and protocol forward-progress rules outside the policy.
  Every policy has an explicit identity setting whose class-label permutation
  leaves the accepted baseline byte-identical.
- BRIDGE-1 (inherited from the folded bridge module): persistent co-simulator
  process for closed loop, replacing per-step subprocess spawns. Its
  incremental flow-injection transport should carry `ExecutionGraph` and
  `CompletionEvent` and bookkeeping facts once CORE-5 lands. The M4 diagnostic
  mode currently pays
  about 8 seconds of process/parse overhead per live tp=8 step.
