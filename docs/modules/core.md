# simllm.core

Framework-agnostic heart of the simulator: virtual time, scheduler-step
records, the device-level execution graph and completion feedback. The core
defines contracts and orchestration. Framework policy stays in adapters;
compute calibration, traffic expansion and physical backends stay in their
own modules.

## Interface

### Scheduler boundary

- `StepRecord`: what a framework scheduler decided to run in one engine step
  (per-request phase, new tokens, cached tokens, preemptions and finishes).
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
SQ/RQ/CQ, network WQE, and either a DCQCN QP or an `rnic-cn` directed L2 link
pair. A WQE must name exactly one SQ, RQ and CQ. Physical WQEs name exactly one
QP or link pair; topology-free null profiles instead record
`transport_kind=none`. Packet objects are intentionally absent. Framework
pointers remain strings owned by the adapter; the core records them but never
interprets or dereferences them. Reusable vRAM, queue, QP and link-pair records
retain their creation scope but may be referenced by later steps and
executions. Each use carries its own scope. Causal object lineage may narrow a
batched request set, but cannot introduce a request absent from its causal
parents.

The graph carries five typed work payloads rather than one unstructured
dictionary:

| Payload | Semantic owner | Runtime lowering |
|---|---|---|
| `ComputeWork` | Model runner plus `ComputeProvider` | launch queue, CUDA stream, GPU work queue, hardware scheduler and HBM |
| `KvCacheWork` | Real framework KV/prefix-cache manager | logical state transition; READ/WRITE becomes HBM work, SWAP/TRANSFER becomes DMA, RECOMPUTE becomes compute plus WRITE |
| `DmaWork` | Data-mover planner | DMA descriptor, directional copy engine, source/destination memory queues |
| `CollectiveWork` | Framework/NCCL observation plus traffic planner | NCCL channels, DMA/HBM work, WQEs and network flows |
| `ControlWork` | Framework or runtime controller | synchronous or asynchronous high-priority local/fabric control path |

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
                          -> NCCL channels -> per-GPU/QP WQE queues
                          -> one shared NIC per node -> network backend
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
3. The device runtime applies FIFO queueing and physical resource arbitration.
   Overlap occurs only when operations are ready together and their selected
   resources permit concurrent service.
4. Compute, collective and network models supply service demand and resource
   behavior. They do not rewrite framework ordering.

The first runtime profile is intentionally fixed: eight GPUs per node, one
logical WQE submission queue or QP per GPU, and one shared 400G physical NIC
arbiter and serializer. All cross-node WQEs contend for that NIC. Intra-node
traffic uses an NVLink-class resource. General fabric discovery and arbitrary
GPU-to-NIC mapping are not prerequisites for this profile.

The first GPU model stops at coarse, replaceable mechanisms: non-preemptive
kernel service, a calibrated compute duration, shared HBM demand and a small
number of copy engines. Warp issue, SM residency and block-level cycle
scheduling remain later refinements behind `DeviceRuntime`.

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
Explicit KV state semantics remain CORE-3; physical queue arbitration and
creation of NCCL, SQ/RQ/CQ, WQE and transport records remain CORE-4; completion
reduction and tail attribution remain CORE-5.

## Pre-registered runtime sanity experiments

These expectations are recorded before CORE-4 implements scheduling. CORE-2
does not claim to produce these resource-contention measurements.

1. **Dependency versus legal overlap.** Release one compute operation of C
   picoseconds and one DMA operation of D picoseconds on independent logical
   queues with ideal independent resources. With no edge, makespan must be
   `max(C, D)`; adding a dependency must make it `C + D`. Sweep both the
   dependency setting and two demand pairs, `(C, D) = (10 us, 40 us)` and
   `(80 us, 40 us)`. Every result must match exactly in the ideal profile.
2. **Eight producers sharing one NIC.** Each active GPU submits one aligned
   WQE of B bytes from its own FIFO/QP to the same node NIC, with no propagation
   or protocol overhead in the ideal profile. Sweep active GPUs N in `{1, 8}`
   and link rate R in `{200, 400}` Gbit/s. The phase makespan must be
   `8 * N * B / R` seconds; doubling R halves it exactly and changing N from 1
   to 8 multiplies it by eight. Per-GPU FIFO order must remain stable under
   both rates.
3. **Tail attribution conservation.** For every completed operation, the sum
   of time attributed to launch queue, device queue, service and completion
   delivery must equal its end-to-end latency exactly. No interval may be
   negative, and graph completion must equal the latest required completion
   event. Sweep synchronous versus asynchronous control delivery and control
   priority at two levels; only dependency-reachable work may move.

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
- CORE-4: implement the first coarse `DeviceRuntime`: framework launch FIFO,
  per-CUDA-stream FIFO and event dependencies, non-preemptive GPU work queue
  and hardware service, shared HBM queue, directional copy engines, explicit
  DMA descriptors, NCCL channel queues, per-GPU/QP WQE queues, one shared NIC
  arbiter/serializer, completion queue and high-priority control queue. Start
  with the fixed eight-GPU, one-NIC node above. Keep every resource policy
  replaceable so a later GPU scheduler can add SM/block detail without a new
  graph contract. Append concrete NCCL command, SQ/RQ/CQ, WQE and QP/link-pair
  records to `RequestBookkeeper`; CQ is initially consumed immediately at the
  WQE completion timestamp and RQ stays an identity-only placeholder. Preserve
  graph operation identity through collective expansion and rendered GOAL tags
  so backend WQE rows correlate without inference. Run and defend all
  pre-registered experiments above.
- CORE-5: implement completion feedback and tail attribution. Stream queue,
  start, progress and completion events, reduce the required completion
  boundary to `StepResult`, advance `VirtualClock`, and export per-request
  TTFT/TPOT plus queue-, KV-, kernel-, DMA-, collective-, NIC- and control-
  attributed components. Support synchronous waits and asynchronous control
  or collective progress without changing the event schema.
- CORE-6: represent variable per-pair all-to-allv sizes in the graph
  contract. `CollectiveWork.payload_bytes` carries one uniform ordered-pair
  share for `pairwise` all-to-allv, so a captured, non-uniform dispatch
  (routed experts under real gating) cannot be expressed. Decide between an
  optional per-pair size table on the collective payload and a schema bump;
  the uniform scalar form must stay readable either way. Coordinate with the
  TRAF-2 capture half so traffic expansion and the renderer consume the same
  representation.
- CORE-7: make `RequestBookkeeper.append` and `extend` validation
  incremental. Every append currently revalidates the entire candidate
  ledger, so N single-fact appends cost quadratic work; CORE-4 streams
  per-WQE events for 64-rank runs and needs amortized constant-time appends
  with unchanged invariants. The full-ledger validator remains the reference
  implementation for snapshots and wire loads.
- BRIDGE-1 (inherited from the folded bridge module): persistent co-simulator
  process for closed loop, replacing per-step subprocess spawns. Its
  incremental flow-injection transport should carry `ExecutionGraph` and
  `CompletionEvent` and bookkeeping facts once CORE-5 lands. The M4 diagnostic
  mode currently pays
  about 8 seconds of process/parse overhead per live tp=8 step.
