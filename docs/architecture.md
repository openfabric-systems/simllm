# SimLLM Architecture

SimLLM couples the *real* request scheduler of a serving framework with a
simulated GPU executor and a packet-level network backend. This document
describes the components, the exact integration seams in vLLM and SGLang, and
the trace format that connects the core to the network simulator.

## Components

```
   Workload Generator ──► Framework Frontend ──► SimLLM Core ──► Network Backend
```

### Workload generator (`simllm/workload/`)

Requests are produced by a queueing model: an arrival process (Poisson,
bursty/MMPP, or trace replay) plus prompt/output length distributions. Prompts
are synthetic token-ID sequences with *controllable shared-prefix structure*
(system-prompt pools, multi-turn sessions). This matters because both
frameworks match prefixes on actual token IDs; prefix-cache hit rates must be
emergent from the workload, not assumed.

### Framework frontend (adapters, `simllm/adapters/`)

The framework's scheduler, batching policy and KV/prefix-cache accounting run
unmodified; only model execution is replaced. Both adapters reduce to the same
contract: per scheduler step, a **step record** (which requests ran, how many
prefill/decode tokens each, how many tokens were served from cache) goes to
the core, and a **step result** (simulated step latency, flow completions)
comes back.

**vLLM (v1 engine, pinned to v0.26.0):** no fork needed. The engine's step
loop is `EngineCore.step()`, i.e. `Scheduler.schedule()` producing a
`SchedulerOutput`, then `Executor.execute_model(scheduler_output)` returning
a `ModelRunnerOutput`, then `Scheduler.update_from_output()`. The executor
class is pluggable: `--distributed-executor-backend` accepts a dotted import
path, resolved in `Executor.get_class()` (`vllm/v1/executor/abstract.py`).
SimLLM ships `simllm.adapters.vllm.SimExecutor`, which

- serves the init-time RPCs with model-derived values (the current RPC list
  and every version-specific behavior live in
  [modules/adapters-vllm.md](modules/adapters-vllm.md), the true source for
  the adapter); the simulated vRAM size is pinned via
  `CacheConfig.num_gpu_blocks_override`;
- fabricates `ModelRunnerOutput(req_ids, req_id_to_index, sampled_token_ids)`
  per step and attaches simulated timing;
- exports the placement manifest (below) from the workers via
  `collective_rpc`, for both simulated and real capture runs.

The vLLM v1 KV-cache manager, block pool, prefix-block hashing and preemption
logic are pure CPU-side bookkeeping inside the scheduler process, so they run
for real under the simulated executor.

**SGLang (main, pinned by commit):** the seam is the TP worker. SimLLM
implements `SimTpModelWorker(TpModelWorker)` whose
`forward_batch_generation(batch)` returns a `GenerationBatchResult` with
fabricated `next_token_ids` and simulated timing; it is installed at the
scheduler's worker-construction point (`Scheduler.init_tp_model_worker`,
the same seam SGLang uses to swap in its MLX worker) through SGLang's
plugin framework (a `sglang.srt.plugins` entry point applying a `REPLACE`
hook, no fork; the pinned commit and every version-specific behavior live
in [modules/adapters-sglang.md](modules/adapters-sglang.md), the true
source for the adapter). The first iteration runs with
`--disable-overlap-schedule`. RadixCache prefix matching, eviction, and the
token/request pool accounting are scheduler-side index bookkeeping and stay
real, so radix hit rates and vRAM pressure respond to the workload exactly as
in production.

### Placement and the mapper (`simllm/placement/`)

Serving frameworks are *topology-light*, not topology-agnostic. Each layer of
the stack knows a different slice of the truth:

| Layer | What it knows |
|---|---|
| Model-parallel framework | Logical ranks, TP/PP/DP/EP groups, how weights/experts are partitioned |
| Executor / launcher | Which global rank runs on which host and local GPU |
| NCCL / cluster runtime | PCIe/NVLink/NIC topology, selected communication path and collective algorithm |
| Network simulator | Switches, links, routing, queueing, congestion-control behavior |

SimLLM joins two independent descriptions:

**Placement manifest** (`simllm-placement-manifest-v1`), per global rank:
host, local rank, GPU UUID / PCI bus ID, group memberships with the *actual*
global rank lists (export `GroupCoordinator.ranks` at runtime rather than
recomputing from the `((dp*PP+pp)*PCP+pcp)*TP+tp` formula: external DP,
elastic behavior or implementation changes break derived layouts), the
pipeline layer range `[start, end)` taken from the model (partitions are not
guaranteed equal), and per-MoE-layer local global expert IDs. A GPU generally
owns a *product* of shards simultaneously, e.g. a PP layer range, a TP slice,
a DP replica index and a set of local experts at the same time, so the
manifest records all coordinates rather than a single "shard id". Manifests
are either **declared** (what-if placements for simulation) or **extracted**
from a live run; in vLLM, one `collective_rpc` over the workers (via a
callable or a worker extension class, still no fork), recording the framework
version since this is an internal surface. Both modes share one schema, which
is what makes simulated and real deployments directly comparable.

With dynamic expert load balancing (EPLB), expert ownership changes at
runtime: each re-placement bumps a monotonically increasing
`placement_epoch`, the physical/logical expert maps are snapshotted per
epoch, and every traffic record references the epoch it was routed under.
With redundant experts, the logical expert ID alone does not identify the
destination rank; the dispatcher's selected physical expert (or destination
rank) is recorded.

**Fabric topology manifest** (`simllm-fabric-topology-v1`): nodes, GPUs,
PCIe/NVLink links, NICs, GPU-to-NIC affinity, switches, links, bandwidths,
delays, queue configuration. Intra-node structure can be taken from NCCL's
detected topology (`NCCL_TOPO_DUMP_FILE`); the switch-level graph always
comes from a cluster inventory or the simulator topology config. NCCL's
local discovery is not a description of the routed network.

The **mapper** resolves every rank in a communication event to a physical
endpoint (rank to node to GPU to NIC) and assigns GOAL ranks. GOAL rank
assignment mirrors the htsim RNIC drivers' `-goal_rank_mapping` option:
`gpu-rank` (one GOAL rank per GPU) or `unique-nic` (one per NIC; intra-node
transfers stay off the fabric).

General fabric discovery is not on the critical path for the first execution
runtime. That profile fixes each node at eight GPUs and one GPU-affine 400G
RNIC per GPU: one logical WQE submission queue or QP per GPU feeds its rail
endpoint, while intra-node traffic uses an NVLink-class resource. This retains
the queue structure needed for head-of-line blocking, fairness and control
priority without first solving arbitrary GPU-to-NIC inventory.

### Execution and resource boundary (`simllm/core/execution.py`)

`StepRecord` deliberately stays small: it records the scheduler's semantic
batch decision, not a guessed GPU timeline. A framework-independent
`ExecutionLowerer` combines that record with adapter observations and emits
`simllm-execution-graph-v1`:

```text
framework scheduler and KV control plane
  -> StepRecord + KV events + captured stream/event schedule
       \-> request stages + opaque framework objects -> RequestBookkeeper
  -> ExecutionLowerer + ComputeProvider
  -> ExecutionGraph
       ComputeWork | KvCacheWork | DmaWork | CollectiveWork | ControlWork
  -> DeviceRuntime
       host launch queue
       logical CUDA streams and event dependencies
       GPU work queue -> isolated kernel scheduler/SM/HBM service
       directional copy engines and DMA descriptors
       NCCL channels -> semantic RNIC submission
       one GPU-affine RNIC session per GPU (WQ/CQ and hardware)
       completion/event plumbing and control queue (class label only under identity)
  -> CompletionEvent stream
       \-> operation + created-object subject -> RequestBookkeeper
  -> StepResult + virtual-time metrics + next framework decision
```

The graph records FIFO submission order within each logical queue and two
explicit start-after-completion edge kinds across queues. `depends_on` waits
for the whole predecessor operation. `participant_local_depends_on` lets a
distributed command's rank wait for the predecessor frontier on that same
rank. One target may carry both kinds without conflating them. Two nodes on
different queues with no dependency are legally concurrent, but that is not a
promise that they overlap in time. The `DeviceRuntime` maps ready work onto
physical queues and resources; overlap emerges from resource availability and
contention. The framework never supplies an overlap percentage, and compute,
traffic or network providers never rewrite framework ordering.

The target cross-layer queue contract is a canonical `QueueVisit` accounting
record, not one universal queue implementation. GPU schedulers, PCIe rational
calendars,
capacity pools and RNIC work queues keep their mechanism-specific state, but
each visit exposes the same boundary: subject and resource identity, accounting
class, deterministic tie-break sequence, `submitted_at_ps`, `eligible_at_ps`,
`started_at_ps`, `finished_at_ps` and `completed_at_ps` when a later observer
exists. `eligible_at_ps` includes required same-subject
predecessors, so an object's earlier fragments are service dependencies rather
than fictitious contention. Queue wait is exactly
`started_at_ps - eligible_at_ps`; service is exactly
`finished_at_ps - started_at_ps`. A credit, dependency or
downstream visibility gate is either its own visit or an explicitly named
nonqueue interval. It is never folded into another resource's wait. End-to-end
attribution follows the graph's realized critical path; it does not add nested
or parallel visit durations. CORE-8 owns the shared semantics, CORE-4 emits
runtime visits, and each mechanism must reconcile its local counters to them.

Every contended resource must also expose an explicit arbitration-policy seam.
The identity policy preserves the mechanism's existing deterministic order,
adds no policy delay and ignores accounting class for scheduling. This is the
mandatory off path. An enabled policy may use class or priority to choose among
eligible visits, but it cannot rewrite readiness or service demand. Disabling
it must reproduce the identity visits and completion times exactly. Thus class
remains useful attribution in the baseline and becomes a scheduling dimension
only through an opted-in, testable policy.
The queue contract expectations were first frozen before implementation in
[examples/queue_contract_v1](../examples/queue_contract_v1/expectations.md) at
commit `65b5609`; commit `facb26d` clarified identity-policy scope, and commit
`947399c` records the final pre-run state.

`completion_operation_ids` separates the framework-visible boundary from
physical quiescence. An empty tuple means all graph operations must complete.
An explicit subset lets asynchronous DMA, collective or control work remain in
the stateful runtime when the next framework step is released. The result
records both the boundary time and, when reached, the physical-quiescence time;
later events retain their original execution and operation IDs.

Typed payloads preserve replaceable fidelity boundaries:

- `ComputeWork` carries kernel identity, shape, flops, HBM demand and the
  selected provider's nominal service estimate. A measured table, calibrated
  SASS table or trace-driven GPU service model can price the same node.
- `KvCacheWork` carries explicit allocation, prefix, access, retention,
  eviction, movement and recompute decisions made by the real framework.
  Physical KV reads/writes lower to HBM operations; swap and remote movement
  lower to DMA plus network work; recompute lowers to compute plus a KV write.
- `DmaWork` is a data-mover descriptor independent of a particular copy
  engine. The runtime selects and contends engines.
- `CollectiveWork` is semantic. The traffic/NCCL planner selects algorithms,
  chunks and channels before WQEs reach the network backend.
- `ControlWork` records local or in-band messages and their synchronous or
  asynchronous semantics. Control priority is a runtime policy and remains
  visible in completion events.

`simllm-completion-event-v1` records submitted, queued, started, progress and
completed timestamps with the selected resource and an optional created-object
subject. This is the accounting
surface for queue delay, compute service, HBM wait, DMA/copy delay, collective
progress, WQE/NIC delay and control completion. `simllm-execution-result-v1`
reduces a graph to its completion boundary while retaining those events for
tail-latency attribution.

`simllm-request-bookkeeping-v1` is the independent, append-only runtime fact
stream. It keeps the graph immutable while correlating request stages and
owner-created objects: opaque framework vRAM allocations, execution
operations, NCCL commands, SQ/RQ/CQ and WQEs. It is a public projection, not a
second WQE lifecycle implementation. Packet identities and packet lifecycle
remain backend-private.

One run has one mutable WQE authority. Today the live htsim path uses its
timing-neutral `AtlahsWqeLedger`, while the structural C++ `WorkQueue` is
reachable only from native tests and probes. In the target structural mode the
SimLLM native RNIC session alone allocates WQ and WQE identities, changes
occupancy and records lifecycle timestamps. The htsim ledger is then neither
constructed nor mutated; bookkeeping facts and result rows are immutable
projections of native records. The explicit hardware-bypass mode does the
opposite: it constructs no structural RNIC state and labels the compatibility
ledger as the authority for that run. Structural and bypass records are never
merged or reconciled by choosing timestamps after simulation.

The stable WQE key is session and endpoint plus WQ kind, WQ identity and post
sequence. WR IDs, GOAL flow IDs, htsim tokens and local vector indices are
correlations only. Each logical network extent has a stable index, while every
transmission or retry has its own attempt index and opaque token; a dropped
attempt does not terminate the logical extent if reliability retries it. A
send WQE belongs to its local SQ and send CQ; a remote RQ
or SRQ owns a separately posted receive WQE and is linked only when RX matching
occurs. The current bookkeeping-v1 `rq_id` and immediate-CQ behavior remain
compatibility projections until BACK-9 supplies the structural schema. BACK-11
gives every full-RNIC policy a hardware QP while keeping its CC or link-pair
identity separate.
Reusable vRAM, queue, QP and link-pair objects retain the scope in which they
were created, while each later use records its own step, execution, operation
and request scope. Causal lineage may narrow a batched request set and rejects
new request IDs that do not occur in any causal parent; reusable resource
references do not impose their original transient scope on later users.

### Core (`simllm/core/`, `simllm/compute/`, `simllm/traffic/`, `simllm/goal/`)

- **Virtual clock**: orders request arrivals and step completions.
- **Execution contracts** (`simllm/core/execution.py`): the passive,
  versioned graph and completion records above, plus `ExecutionLowerer` and
  `DeviceRuntime` protocols. CORE-2 supplies strict graph/result JSON,
  serial diagnostic lowering and graph-only replay. CORE-3 through CORE-5
  implement KV lifecycle, resources and feedback in order.
  Dependency semantics are explicit per edge: `depends_on` waits for complete
  predecessor operations, while `participant_local_depends_on` lets
  distributed collective ranks arrive after their own predecessor frontiers.
  The serial GOAL renderer rejects cross-rank operation barriers that it
  cannot encode.
- **Central bookkeeping** (`simllm/core/bookkeeping.py`): validated frozen
  stage, object-lineage and completion facts behind one mutable append seam,
  with strict JSON round trips and request/execution/object queries.
- **Compute-time providers** (`simllm/compute/`): the duration of every
  GOAL `calc` node comes from a pluggable `ComputeProvider`:
  `ProfileTableProvider` (measured (kernel, config, GPU) duration tables
  from real captures), `RooflineProvider` (analytical
  `max(flops/peak, bytes/bw)`, classifying each kernel as compute- or
  memory-bound from its configuration alone), and a trace-driven service
  model behind the same boundary. The service model represents SASS
  dependencies, warp issue, CTA admission and SM residency, HBM latency and
  bandwidth, plus isolated copy-descriptor service. It is intentionally an
  extensible mechanism model, not an analytical roofline in disguise.
  Catalog replay happens once while constructing a provider or artifact, and
  validated artifacts compile into O(1) profile tables. Detailed Accel-Sim
  replay still runs offline for configurations nobody measured. The serving
  step loop invokes neither cycle simulator. Every estimate carries provenance
  and uncertainty so bootstrap parameters cannot be mistaken for silicon
  validation.
- **Host initiation model** (`simllm/compute/host.py`): the data-parallel
  handoff chain (receive data plus a small start packet, compute, hand data
  over, write a small packet releasing the next rank) is exactly GOAL's
  `recv`/`calc`/`send` chain with `requires` edges. The doorbell packet
  itself is modeled *in-band* as a small control-class message on the fabric
  (the RNIC endpoint models already carry ~64 B control packets),
  so it competes for wire time and sees network-side jitter. The host path
  before the wire (CPU proxy vs GPU-initiated networking, PCIe, RNIC
  doorbell-to-wire) defaults to **zero delay and zero jitter**: those
  effects are roughly constant per operation, orthogonal to fabric behavior,
  and folding them in by default would confound network attribution. A
  single per-endpoint `initiation_delay_ps` constant exists for launch-path
  studies (e.g. sub-microsecond GPU-initiated vs multi-microsecond
  CPU-proxy) where small-message all-to-all makes launch overhead comparable
  to transfer time.
- **Traffic model**: consumes three inputs: a *collective trace*
  (`simllm-collective-trace-v1`, one JSONL record per op: step, layer, op,
  group type, group global ranks, send counts, element bytes, hidden size,
  placement epoch, release time), the placement manifest, and the fabric
  manifest. For
  MoE, the static map `expert_owners[layer][global_expert_id]` (a list of
  ranks) turns routed tokens into all-to-allv destinations (per placement
  epoch). Semantic collectives are expanded into the algorithm actually
  used, e.g. ring, tree, pairwise all-to-allv, or a custom
  collective-network schedule, as chunked send/recv chains. Covers TP
  collectives per layer, MoE dispatch/combine (optionally driven by captured
  per-token expert routings), PP activations, and KV-cache transfers
  (PD-disaggregation, cache-miss re-prefill). For simulating communication
  patterns, group memberships plus activation shapes suffice; exact TP
  weight-storage intervals (packed QKV, gate/up packing, quantization
  padding) are deliberately out of scope.
- **GOAL emitter**: renders the step DAG as a GOAL trace (below).

### Network backend (`simllm/backends/`, `third_party/`)

The target full packet path has two independent model axes. SimLLM owns a C++
RNIC hardware extension under `simllm/backends/rnic/`: RDMA WQ/CQ, QP/QPC,
MMIO/PCIe/DMA and TX/RX hardware. htsim owns selectable transport/CC policies
and the packet fabric. A versioned C++ adapter passes opaque packet or flow
tokens and feedback events between them; no QP, queue, context or DMA object
crosses that boundary. BACK-8 and HTSIM-9 will link the SimLLM static library
into the directly invoked htsim binaries and present the composition through
the existing `AtlahsFlowRuntime` interface. There is no Python callback in the
packet event loop.

That composition is not live today. The wheel carries the CMake files and C++
sources but builds no Python extension, and the current `htsim_rnic` binaries
do not link `simllm::rnic`. The implemented standalone C++17 SQ/CQ,
`PcieFabric` and opaque flow-level `NetworkPort` are therefore reached by
native tests and probe studies only. Their timing cannot yet change packet
FCT, `ExecutionResult`, `StepResult` or TTFT/TPOT. The descriptor carries GOAL
flow/tag identity and a separate policy-context token, while completion uses a
network-owned token. It does not equate flow acceptance or delivery with
first/last packet issue. The standalone slice is validated in
[examples/rnic_wq_v1](../examples/rnic_wq_v1/RESULTS.md); its live wrapper and
packet-level adapter remain BACK-8 and HTSIM-9. The detailed evidence and
calibration plan is
[papers/rnic-hardware-calibration.md](papers/rnic-hardware-calibration.md).

The reachability contract is one timing path. `ExecutionGraph` enters the
CORE-4 `DeviceRuntime`, which invokes the composed htsim binary and returns one
`ExecutionResult`; its completion boundary becomes `StepResult`, advances the
virtual clock and therefore changes TTFT/TPOT. The current `HtsimStepSink`
already maps an htsim makespan into `StepResult`, but it still invokes the
uncomposed binary and no concrete graph `DeviceRuntime` exists. Structural
mode must start the inner htsim network operation only when the native WQE is
eligible and must release the outer GOAL operation only when native completion
delivery, e.g. CQ polling, permits it. The resulting terminal completion is
consumed once. Python must never compute `native delay + htsim FCT`, and the
composed binary must never retain the timing-neutral ledger beside the native
WorkQueue. Bypass mode uses the old path alone and must preserve its accepted
completion times exactly.
The composition expectations were first frozen before implementation in
[examples/rnic_live_v1](../examples/rnic_live_v1/expectations.md) at commit
`65b5609`; commit `facb26d` clarified retry identity, and commit `947399c`
records the final pre-run drain and audit wording.

The GOAL trace is executed by a discrete-event simulator:

- **htsim** (packet-level): `htsim_uec -goal <bin> -topo <topo>` executes the
  GOAL schedule over a Clos topology with full transport behavior, and
  `htsim_rnic` (currently pinned on the append-only
  `2026_08_05/simllm-addon` branch) runs the RNIC
  policy profiles: the packetized no-CC baseline `rnic-nn`, explicit-hardware
  bypass baseline `rnic-nn-fluid` and explicit-rate endpoint `rnic-cn`. The
  GOAL-driven `htsim_dcqcn_atlahs` comparator is also available. Only the
  Slingshot-like profile `rnic-ss` remains a profile-wiring follow-up
  (HTSIM-1 in [modules/backends.md](modules/backends.md)); the factory rejects
  it with an explicit error.
- **LogGOPSim** (flow-level, fast): same GOAL input, LogGOP cost model;
  useful for quick sweeps before packet-level runs.

Completion times flow back to the core, which advances the virtual clock and,
in closed-loop mode, releases the frontend's next scheduling step.

## GOAL trace format

GOAL (Group Operation Assembly Language) is the dependency-graph schedule
format consumed by LogGOPSim and htsim (via `txt2bin`):

```
num_ranks 2
rank 0 {
  l0: calc 5000
  l1: send 8192b to 1 tag 0
  l1 requires l0
}
rank 1 {
  l2: recv 8192b from 0 tag 0
}
```

- `calc <cost>` is local compute; `send/recv <size>b to/from <peer> tag <t>`
  are point-to-point transfers. Optional `cpu <c>` / `nic <n>` clauses pin
  ops to resources.
- `a requires b`: `a` starts after `b` *finishes*. `a irequires b`: `a`
  starts after `b` *starts*.

Collectives are decomposed into send/recv chains by the emitter, so the
network simulator sees the real chunked traffic pattern rather than an
abstract collective op.

## Coupling modes

1. **Offline (open-loop).** The frontend runs to completion under the sim
   executor (fast, no network in the loop); every step is recorded; one GOAL
   trace is emitted and simulated once. Cheap, deterministic, but network
   congestion cannot influence batch composition.
2. **Closed-loop.** Each scheduler step (or window of steps) is simulated
   before the next one is released; the network's completion time advances the
   virtual clock the scheduler sees. The step/result exchange uses versioned
   JSON manifests (`atlahs-closed-loop-step-v1` / `atlahs-closed-loop-result-v1`,
   see `simllm/core`). Per-step subprocess invocation is the diagnostic
   mode; a persistent co-simulator process is planned for scale.

## Timing and metrics

An instantly-returning simulated executor breaks metric *meaning*, not metric
plumbing: serving frameworks compute TTFT/TPOT from wall-clock timestamps
spanning two processes. SimLLM handles this in two ways:

- **Paced mode**: the sim executor delays completion by the simulated step
  latency; every stock metric works unchanged (sim runs in scaled real time).
- **Virtual mode**: the executor returns immediately and SimLLM reports
  sim-native metrics (per-request TTFT/TPOT/queueing delay on the virtual
  clock) through its own metrics pipeline, bypassing the framework's
  wall-clock histograms.

## Validation

Every simulated configuration should carry provenance (backend profile,
topology, calibration profile, seeds) and be checked against real captures
where available: single-node runs for compute-model calibration, multi-node
NCCL traces (via the ATLAHS capture pipeline) for network-model validation,
with error bounds reported per metric.

Accuracy validation advances only after calibrated compute and the resource
runtime are available. Use identical framework commit, model, parallelism,
request trace, seed and warm-up policy in simulation and silicon. Progress
through single-GPU compute, eight-GPU intra-node, two-node rail-RNIC,
offered-load sweeps, KV pressure, chunked prefill/preemption or retraction,
then mixed and bursty workloads. Report p50, p90, p99 and p99.9 TTFT/TPOT,
with residuals attributed to request queues, KV state, kernel service, HBM,
DMA, collectives, WQEs/NIC, flow completions and control delivery. Fit only
the early calibration cases; later cases remain held out. The largest
attributed held-out residual selects the next fidelity improvement.

### GPU service-model evidence boundary

The first trace-driven GPU slice validates mechanisms before claiming device
accuracy. Exact synthetic fixtures cover partial CTA waves, scheduler width,
dependency chains, occupancy minima, HBM latency plus serialization, and
copy-descriptor setup plus bandwidth. The frozen study is
[examples/gpu_service_model](../examples/gpu_service_model/expectations.md).
It varies at least two parameters in every component sweep and requires zero
cycle residual for the closed forms.

The GPU service primitive can also schedule several explicitly supplied
kernels at once. `estimate_concurrent`
replays compute, memory and NCCL network tasks together: they share SM
residency, per-SM issue budgets, pipelines, the HBM cursor and a per-GPU
NVLink egress cursor, and a later task backfills capacity an earlier one
cannot use. A collective is therefore an ordinary kernel here, built by
`simllm.compute.nccl` as the per-GPU egress half of a ring all-reduce, so
it contends for the same GPU as the work it overlaps with rather than
being priced alone. The intra-node NVLink path deliberately stays inside
this model instead of reaching the fabric backend, which is the split
TRAF-10 owns; only inter-node segments become GOAL traffic. The frozen
study is [examples/gpu_task_mix](../examples/gpu_task_mix/expectations.md),
and COMP-11 owns peer topology, ingress service and reduction lanes. The
primitive does not select runnable graph operations; CORE-4 still owns that
DeviceRuntime policy and dispatch.

Open public evidence seeds the A100 SXM 80 GB and H100 SXM 80 GB profiles:

- NVIDIA's [Ampere tuning guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html)
  and [Hopper tuning guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)
  supply documented occupancy and shared-memory limits.
- NVIDIA's [A100 data sheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/a100-80gb-datasheet-update-nvidia-us-1521051-r2-web.pdf)
  and [H100 product specifications](https://www.nvidia.com/en-us/data-center/h100/)
  supply SKU-level peak memory and arithmetic envelopes.
- The open [Ampere microbenchmark study](https://arxiv.org/abs/2208.11174),
  [Hopper/H800 study](https://arxiv.org/abs/2402.13499), and later
  [A100/H800 study](https://arxiv.org/abs/2501.12084) provide timing context
  where NVIDIA does not publish a contract. The numeric memory priors are
  transferred from the last paper. H800 PCIe timing is not H100 SXM timing,
  and the A100 test device is not assumed to match the target SKU exactly.
- The [Accel-Sim ISCA 2020 paper](https://doi.org/10.1109/ISCA45697.2020.00047)
  and [framework repository](https://github.com/accel-sim/accel-sim-framework)
  define the external SASS trace-replay and correlation path used by COMP-1.

These sources do not establish production-kernel timing accuracy. Public
measurements can initialize a parameter with explicit uncertainty, but exact
framework kernel identity, copy-engine topology, cache state, launch mode and
silicon durations must come from a capture ledger.
`simllm-gpu-model-artifact-v2` binds a target-architecture calibration and a
structured capture envelope to trace, kernel and copy identities, stream
order, numeric observed core/memory clocks, simulated cycles, optional
measured samples, calibration split, uncertainty and replay counters. Its
strict loader reruns deterministic estimates and rejects target, clock,
identity or sample-summary drift. Bulk raw traces stay outside Git under
`/data3/yifeng/`. COMP-1,
COMP-5, COMP-6 and advanced instruction/cache semantics in COMP-10 remain
open, as does the inter-operation `DeviceRuntime` in CORE-4.
