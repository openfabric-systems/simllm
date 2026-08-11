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
- Record serialization both ways: `step_record_to_json`,
  `step_records_to_json`, `write_step_records`, `step_record_from_json` and
  `step_records_from_jsonl`, plus `step_result_to_json` and
  `step_result_from_json` for full results.

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
- `CompletionReducer`: the stateful read-only projection from one validated
  `ExecutionResult` plus its `RuntimeReport` to `StepResult`. It follows the
  runtime's realized predecessor chain for each correlated request endpoint,
  accumulates TTFT across non-sampling prefill chunks, computes exact TPOT,
  advances `VirtualClock` only to framework completion and retains later
  asynchronous events as physical evidence.
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
`AtlahsWqeLedger` is not constructed in this mode. A run may instead explicitly
select `AtlahsWqeLedger` as its sole timing-neutral bypass authority. A run
never enables both mutable authorities, and every
projection must conserve identity, cardinality and timestamps available at its
boundary. `CompletionReducer` owns only request metric history; it does not
schedule, progress or complete a runtime object.

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
For a dependency chain, each operation segment begins at its realized
predecessor completion. Queue, service and visibility intervals that completed
before that boundary contribute zero; intervals crossing it contribute only
their remaining tail. Summed segment latency must equal graph JCT exactly.

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

The coarse runtime calls this seam only to select among simultaneously legal
ready graph operations. Framework-launch FIFO, logical-queue FIFO, dependency
release, NCCL round order and WQ protocol order are mandatory rather than
optional arbitration. A co-runnable compute set is dispatched as one set, so
there is no losing candidate to prioritize within that call. Compatible copy
engines use deterministic earliest-availability routing after the operation
has won the ready seam; native RNIC arbitration remains owned by its sole
session. These policy-free points do not inspect class labels. Any object
conforming to `ArbitrationPolicy` may replace identity at the ready seam, while
CORE-10 owns the first supported non-identity behavior.

The pre-implementation queue conformance expectations were first frozen in
[examples/queue_contract_v1](../../examples/queue_contract_v1/expectations.md)
at commit `65b5609`; commit `facb26d` clarified identity-policy scope, and
commit `947399c` records the final pre-run state.

The graph carries five typed work payloads rather than one unstructured
dictionary:

| Payload | Semantic owner | Runtime lowering |
|---|---|---|
| `ComputeWork` | Model runner plus `ComputeProvider` | launch queue, CUDA stream, GPU work queue, hardware scheduler and HBM |
| `KvCacheWork` | Real framework KV/prefix-cache manager | zero-byte lifecycle observation only in CORE-4; byte-carrying READ/WRITE fails preflight until CORE-3 supplies HBM lowering |
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
Explicit KV state semantics remain CORE-3. BACK-9 and BACK-12 own the remaining
structural RNIC objects and arbitration depth.

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

- CORE-3 (Completeness; P1; L): implement explicit KV lifecycle accounting before resource
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
  TTFT/TPOT tails. Adapter capture halves are VLLM-11 and SGL-9. Until this
  lands, CORE-4 accepts zero-byte lifecycle observations but rejects every
  byte-carrying READ or WRITE during preflight rather than reporting silent
  zero-cost HBM work. Acceptance must enable those same fixtures through the
  HBM service and preserve the explicit zero-byte path exactly.
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
  later arrival can enter. Identify admission and completion offsets from a
  reproducible multi-stream trace. Acceptance must vary arrival offset and
  residency pressure, match the observed overlap bands, and preserve the
  simultaneous-arrival and single-kernel baselines exactly.
- CORE-13 (Precision; P1; L): replace the flat per-source intra-node
  NVLink-class serializer with calibrated compute-owned NCCL/NVLink service.
  The current surrogate uses payload bytes and one configured rate; it does
  not replay the network kernel, HBM reads or link/topology selection. Use
  captured NCCL kernel traces plus NVLink byte/rate observations. Acceptance
  must vary payload, participant count and competing kernel demand, change
  end-to-end graph JCT into the measured bands, and retain the explicit
  cross-node RNIC path exactly.
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
- BRIDGE-2 (Completeness; P1; L): implement the online stateful co-simulator
  client above the delivered HTSIM persistent flow session and strict full
  `StepResult` codec. The backend foundation retains one event list, topology,
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
  append batch of object, stage and completion facts; the terminal frame
  carries `ExecutionResult`, `simllm-step-result-v2`, ending ledger cursor and
  physical quiescence separately from framework completion. Reject loss,
  duplication, cursor disagreement, graph/event identity disagreement and
  timestamp regression before publishing a result. The explicit diagnostic
  and BRIDGE-1 prepared modes remain the identity off paths and must preserve
  every accepted byte and timestamp when the online session is disabled.
