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
runtime. That profile fixes each node at eight GPUs and one shared 400G NIC:
one logical WQE submission queue or QP per GPU feeds one physical NIC arbiter
and serializer, while intra-node traffic uses an NVLink-class resource. This
retains the queue structure needed for head-of-line blocking, fairness and
control priority without first solving arbitrary GPU-to-NIC inventory.

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
       GPU work queue, coarse hardware scheduler and HBM queue
       directional copy engines and DMA descriptors
       NCCL channels and per-GPU/QP WQE queues
       one shared NIC arbiter and serializer per node
       completion and high-priority control queues
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

`completion_operation_ids` separates the framework-visible boundary from
physical quiescence. An empty tuple means all graph operations must complete.
An explicit subset lets asynchronous DMA, collective or control work remain in
the stateful runtime when the next framework step is released. The result
records both the boundary time and, when reached, the physical-quiescence time;
later events retain their original execution and operation IDs.

Typed payloads preserve replaceable fidelity boundaries:

- `ComputeWork` carries kernel identity, shape, flops, HBM demand and the
  selected provider's nominal service estimate. A measured table, calibrated
  SASS table or future GPU scheduler can price the same node.
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
operations, NCCL commands, SQ/RQ/CQ, WQEs and either DCQCN QPs or `rnic-cn`
directed L2 link pairs. WQE completion is the lowest public network unit.
Packet identities and packet lifecycle remain backend-private. The initial CQ
policy consumes completion immediately at the WQE timestamp. RQ is an
identity-only placeholder in this contract. Every WQE names one SQ, RQ and CQ;
physical profiles also name one DCQCN QP or `rnic-cn` directed link pair,
while null profiles explicitly record transport kind `none`.
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
  memory-bound from its configuration alone), and, planned, an offline
  SASS-level provider (Accel-Sim / GPGPU-Sim). Cycle-accurate GPU simulation
  is orders of magnitude too slow to sit inside the step loop, so it runs
  offline to *populate profile tables* for configurations nobody measured;
  the step loop always reads tables or analytical estimates. Every estimate
  carries an uncertainty so results can report error bounds honestly.
- **Host initiation model** (`simllm/compute/host.py`): the data-parallel
  handoff chain (receive data plus a small start packet, compute, hand data
  over, write a small packet releasing the next rank) is exactly GOAL's
  `recv`/`calc`/`send` chain with `requires` edges. The doorbell packet
  itself is modeled *in-band* as a small high-priority control message on
  the fabric (the RNIC endpoint models already carry ~64 B control packets),
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

The GOAL trace is executed by a discrete-event simulator:

- **htsim** (packet-level): `htsim_uec -goal <bin> -topo <topo>` executes the
  GOAL schedule over a Clos topology with full transport behavior, and
  `htsim_rnic` (on the submodule's `main` since 2026-08-03) runs the RNIC
  fidelity profiles: the null-network baselines `rnic-nn`/`rnic-nn-fluid`
  and the explicit-rate collective-network endpoint `rnic-cn`. The
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
through single-GPU compute, eight-GPU intra-node, two-node shared-NIC,
offered-load sweeps, KV pressure, chunked prefill/preemption or retraction,
then mixed and bursty workloads. Report p50, p90, p99 and p99.9 TTFT/TPOT,
with residuals attributed to request queues, KV state, kernel service, HBM,
DMA, collectives, WQEs/NIC, flow completions and control delivery. Fit only
the early calibration cases; later cases remain held out. The largest
attributed held-out residual selects the next fidelity improvement.
