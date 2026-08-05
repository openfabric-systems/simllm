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

`simllm.core.execution` reserves three versioned contracts:

- `simllm-execution-graph-v1`: an immutable `ExecutionGraph` containing an
  ordered tuple of `ExecutionOperation` nodes. Tuple order is FIFO submission
  order within each `logical_queue`; `depends_on` supplies cross-queue
  start-after-completion edges. Operations in different queues with no path
  between them are legally allowed to overlap. `completion_operation_ids`
  names the logical boundary that releases the framework; an empty tuple means
  every operation, while an explicit subset permits background asynchronous
  work to remain in the stateful runtime.
- `simllm-completion-event-v1`: a timestamped `CompletionEvent` at submitted,
  queued, started, progress or completed phase, optionally attributed to one
  concrete `ResourceRef`.
- `simllm-execution-result-v1`: an `ExecutionResult` containing the graph
  completion time and its ordered completion-event stream. A separate optional
  quiescence time distinguishes framework-visible completion from all physical
  work draining.

The graph carries five typed work payloads rather than one unstructured
dictionary:

| Payload | Semantic owner | Runtime lowering |
|---|---|---|
| `ComputeWork` | Model runner plus `ComputeProvider` | launch queue, CUDA stream, GPU work queue, hardware scheduler and HBM |
| `KvCacheWork` | Real framework KV/prefix-cache manager | logical state transition; READ/WRITE becomes HBM work, SWAP/TRANSFER becomes DMA, RECOMPUTE becomes compute plus WRITE |
| `DmaWork` | Data-mover planner | DMA descriptor, directional copy engine, source/destination memory queues |
| `CollectiveWork` | Framework/NCCL observation plus traffic planner | NCCL channels, DMA/HBM work, WQEs and network flows |
| `ControlWork` | Framework or runtime controller | synchronous or asynchronous high-priority local/fabric control path |

`ExecutionLowerer` and `DeviceRuntime` are protocols only. No lowerer,
resource scheduler or serializer is installed yet. Current adapters continue
to use the existing `StepRecord -> StepResult | None` sink unchanged. This
keeps the new boundary inert until CORE-2 through CORE-5 are validated.

Nothing in this package may import vLLM or SGLang.

## Mental model and ownership

```text
workload
  -> framework scheduler and KV control plane
  -> StepRecord plus adapter observations
  -> ExecutionLowerer
       compute estimates + KV events + stream/dependency order
  -> ExecutionGraph v1
  -> DeviceRuntime
       launch/CUDA queues -> GPU scheduler + HBM
                          -> copy engines + DMA
                          -> NCCL channels -> per-GPU/QP WQE queues
                          -> one shared NIC per node -> network backend
  -> CompletionEvent v1
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

The execution slice currently contains only passive typed records, reserved
schema names and the `ExecutionLowerer`/`DeviceRuntime` protocols. It changes
no current timing path. This is deliberate: the SASS pipeline, explicit KV
lifecycle, resource runtime and overlap arrive in that dependency order.

## Pre-registered runtime sanity experiments

These expectations are recorded before CORE-4 implements scheduling. The
contract-only slice cannot produce the measurements yet.

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

- CORE-2: implement `ExecutionLowerer`, graph validation and JSON round trips.
  Lower one `StepRecord` plus adapter observations into per-layer compute,
  KV, DMA, collective and control nodes while preserving framework stream
  order and dependencies. The current `HtsimStepSink` remains the diagnostic
  compatibility path until graph replay reproduces its serial closed forms.
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
  graph contract. Run and defend all pre-registered experiments above.
- CORE-5: implement completion feedback and tail attribution. Stream queue,
  start, progress and completion events, reduce the required completion
  boundary to `StepResult`, advance `VirtualClock`, and export per-request
  TTFT/TPOT plus queue-, KV-, kernel-, DMA-, collective-, NIC- and control-
  attributed components. Support synchronous waits and asynchronous control
  or collective progress without changing the event schema.
- BRIDGE-1 (inherited from the folded bridge module): persistent co-simulator
  process for closed loop, replacing per-step subprocess spawns. Its
  incremental flow-injection transport should carry `ExecutionGraph` and
  `CompletionEvent` once CORE-2/5 land. The M4 diagnostic mode currently pays
  about 8 seconds of process/parse overhead per live tp=8 step.
