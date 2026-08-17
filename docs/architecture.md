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
unmodified; the executor seam replaces model execution alone, while the
flagged VLLM-13/VLLM-16 model-runner seam additionally replaces the collective
layer with `SimGroupCoordinator` and the simulated NCCL stack. Both adapters
reduce to the same contract: per scheduler step, a **step record** (which
requests ran, how many prefill/decode tokens each, how many tokens were served
from cache) goes to the core, and a **step result** (simulated step latency,
flow completions) comes back.

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

One run has one mutable WQE authority. Today the composed native C++ RNIC
session is the sole mutable WQE authority on the live htsim path, and the
timing-neutral `AtlahsWqeLedger` remains the sole authority only in explicit
hardware-bypass mode. In structural mode, live for the frozen isolated
`rnic_live_v1` fixture, the SimLLM native RNIC session alone allocates WQ and
WQE identities, changes occupancy and records lifecycle timestamps. Structural
mode constructs no htsim ledger at all, so bookkeeping facts and result rows
are immutable projections of native records. The explicit hardware-bypass mode
does the opposite: it constructs no structural RNIC state and labels the
compatibility ledger as the authority for that run. Structural and bypass
records are never merged or reconciled by choosing timestamps after simulation.

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
  `DeviceRuntime` protocols. CORE-2, CORE-4 and CORE-5 are complete for their
  supported scopes: strict graph/result JSON with serial diagnostic lowering
  and graph-only replay, the coarse `DeviceRuntime` including the frozen
  Tier B structural fixture, and completion reduction with seven-component
  attribution. Explicit KV lifecycle accounting and its HBM metric path are
  implemented. CORE-3 still owns the remaining case matrix, sweeps, reporting
  surface and the `SWAP`, `TRANSFER` and `RECOMPUTE` lowering gap.
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
  single per-endpoint `initiation_delay_ps` constant remains the analytical
  fallback for launch-path studies (e.g. sub-microsecond GPU-initiated vs
  multi-microsecond CPU-proxy) where small-message all-to-all makes launch
  overhead comparable to transfer time. BACK-27 adds the structural
  alternative: the CPU-proxy and GPU-initiated producers submit timed tasks
  to the concurrent compute service, which measured +20 and +23 cycle
  submission delays at saturated occupancy while staying disabled and
  byte-identical by default.
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
MMIO/PCIe/DMA, the virtual host-memory registry with its QPC translation
asymmetry, the host-CPU, CPU-proxy and GPU-initiated submission profiles with
their independently named requester and sole CQ consumer, and TX/RX hardware.
htsim owns selectable transport/CC policies and the packet fabric. A versioned
C++ adapter passes opaque flow tokens and, at ABI v2, session-unique
packet-attempt identity, committed TX/RX boundaries, typed drop evidence and
transport-control events between them; no QP, queue, context or DMA object
crosses that boundary. BACK-18 landed the modular construction entry point,
and the pinned backend main links the SimLLM static library into the directly
invoked `htsim_rnic` binary through the `AtlahsFlowRuntime` composition, which
passed both frozen live-composition tiers at ABI v1 and now also carries the
ABI-v2 packet-attempt and transport-control vocabulary. There is no Python
callback in the packet event loop.

The composition is live for the frozen isolated `rnic_live_v1` fixture.
Tier A exercised the directly invoked composed binary and its flow-level FCT
and JCT evidence. Tier B consumed immutable native observations through
`ExecutionGraph`, `CoarseDeviceRuntime`, `CompletionEvent`,
`ExecutionResult` and `StepResult`, so native doorbell and link-rate changes
reached TTFT and TPOT by the frozen relations. BACK-8 and the demonstrated
CORE-15 live-seam clauses closed on that evidence. The CORE-21
same-contended-graph bypass-versus-composed comparison with its signed JCT,
TTFT and TPOT change, and the BACK-31 executable-level unlinked-native
negative control, landed after Tier B, which ran neither.

The ABI-v1 descriptor carries GOAL flow and tag identity plus a separate
policy-context token, while completion uses a network-owned token. It does not
equate flow acceptance or delivery with first-packet or last-packet issue.
BACK-25 and BACK-26 closed on 2026-08-11 at the versioned vocabulary and relay
boundary, so NetworkPort ABI v2 now carries session-unique packet-attempt
identity, explicit TX start and finish, RX arrival, attempt terminals, typed
drop evidence, ECN/CNP, effective rate updates, PFC and link-state forms, with
ABI v1 kept as the exact default compatibility path. HTSIM-9 closed on
2026-08-11, when a composed run of the Tier B class carried that ABI-v2
packet-issue evidence through `ExecutionGraph` to `CompletionEvent`,
`StepResult`, TTFT and TPOT; the dynamic-link producer, the physical control
producers and the partial-final-packet cell landed under HTSIM-15, HTSIM-16
and BACK-34. The standalone
slice is validated in
[examples/rnic_wq_v1](../examples/rnic_wq_v1/RESULTS.md), the live gate in
[examples/rnic_live_v1](../examples/rnic_live_v1/RESULTS.md), and the ABI-v2
packet-attempt and transport-control vocabulary in
[examples/rnic_packet_v2](../examples/rnic_packet_v2/RESULTS.md). The
detailed evidence and calibration plan is
[papers/rnic-hardware-calibration.md](papers/rnic-hardware-calibration.md).

The reachability contract is one timing path. `ExecutionGraph` enters the
CORE-4 `DeviceRuntime`, which projects one composed native completion into an
`ExecutionResult`; its completion boundary becomes `StepResult`, advances the
virtual clock and therefore changes TTFT/TPOT. Tier B implements that contract
for its fixed cells without adding a second probe or compatibility-ledger
delay. Structural mode starts network service only when the native WQE is
eligible and releases the outer operation at native completion visibility.
The terminal completion is consumed once. Python must never compute
`native delay + htsim FCT`, and the composed binary must never retain the
timing-neutral ledger beside the native WorkQueue. Bypass mode uses the old
authority alone and preserves its accepted completion artifacts. General
same-graph authority comparison landed under CORE-21, and packet-level
completion detail closed with the HTSIM-9 Tier C run.
The composition expectations were first frozen before implementation in
[examples/rnic_live_v1](../examples/rnic_live_v1/expectations.md) at commit
`65b5609`; commit `facb26d` clarified retry identity, and commit `947399c`
records the final pre-run drain and audit wording.

The GOAL trace is executed by a discrete-event simulator:

- **htsim** (packet-level): `htsim_uec -goal <bin> -topo <topo>` executes the
  GOAL schedule over a Clos topology with full transport behavior, and
  `htsim_rnic`, pinned on backend main (`4885c64`) with the WQE bookkeeping
  and the composed SimLLM RNIC wrapper behind `HTSIM_ENABLE_SIMLLM_RNIC`,
  runs the RNIC policy profiles: the packetized
  no-CC baseline `rnic-nn`, explicit-hardware bypass baseline `rnic-nn-fluid`
  and explicit-rate endpoint `rnic-cn`. The
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
   before the next one is released; one composed native completion projects
   through `CoarseDeviceRuntime` into `ExecutionResult` and `StepResult`,
   whose boundary advances the virtual clock the scheduler sees. Structural
   native authority and explicit hardware bypass are mutually exclusive within
   a run. The step manifest crosses the boundary as versioned JSON
   (`atlahs-closed-loop-step-v1`, see `simllm/core`), while the result returns
   as an in-process `StepResult`: the `atlahs-closed-loop-result-v1` name has
   no reader or writer and predates CORE-5 attribution, so CORE-24 owns the
   strict full-`StepResult` wire codec that replaces it. Per-step subprocess
   invocation remains the default diagnostic mode; BRIDGE-1 landed the opt-in
   prepared-replay `HtsimPersistentStepSink`, which reuses a persistent worker
   pool, matched all 34 prepared-versus-diagnostic pairs exactly and reported
   3.36x to 5.43x speedup in its four scored wall-time instances, while the
   online stateful co-simulator session remains BRIDGE-2, CORE-24 and
   HTSIM-18, and child-process lifetime binding remains BRIDGE-3.


## Devices, ports and packets

Coupling mode and precision level are per-run selections. This section is the
modeling principle underneath both: what a device is, and what moves between
devices. The full statement, with the port taxonomy and its measured ceilings,
the producer taxonomy and its sources, the mapping from landed assets to model
roles, the calibration doctrine and the tasks that close the remaining gaps, is
[design/packet-device-model.md](design/packet-device-model.md).

1. **A device is typed ports plus a service model.** A port carries protocol
   identity, direction, ceiling and the provenance of that ceiling; the service
   model decides when a packet leaves and when it lands. An NVIDIA GPU has PCIe
   ports and NVLink ports, an AMD ROCm GPU has PCIe ports and xGMI ports, a
   Grace Hopper superchip replaces the GPU's host-side PCIe port with
   NVLink-C2C, and an RNIC has PCIe ports and wire ports. These are one object
   with different parameters, not four modeling techniques. A disabled port
   keeps its interface with parameters inert or explicitly rejected, exactly as
   a disabled RNIC module does.
2. **Software stacks are the packet producers.** NCCL and RCCL decide how many
   bytes cross which port in what order: a collective becomes ring steps, ring
   steps become chunks, and a chunk becomes either a peer store on a GPU link
   or a descriptor, doorbell, DMA and wire packets through the NIC. The
   dynamically loaded net-plugin ABI those stacks share (`isend`, `irecv`,
   `test`) is the seam where a producer meets a device.
3. **A ceiling belongs to a port on an architecture; stack efficiency largely
   transfers.** Per-GPU NVLink egress moves by exactly 1.5 times from the A100
   `NV4` mesh to the GH200 `NV6` mesh, while ring all-reduce efficiency against
   that ceiling moves from 71.0 to 74.9 percent, 3.9 percentage points across a
   link generation. Calibration therefore carries the ceiling per architecture
   and reuses the efficiency, and a port with no measured or declared profile
   fails closed rather than borrowing another architecture's number.
4. **Packetizing a leg is a precision level, so it keeps a byte-identical off
   path.** The analytic closed forms this repository validated, the fluid
   fabric serializer and the flat intra-node rate among them, stay selectable
   and stay exact, and the contract in the next section applies to every
   packetized leg added beside them.

Today the NIC is modeled this way and the GPU is not: `RnicDevice` composes a
work-queue core with an optional PCIe fabric, an optional host-memory registry
and either an injected `NetworkPort` or an owned inert one, while the GPU's
intra-node link is one flat per-GPU egress cursor with no port object, no peer
identity and no packet. COMP-34, COMP-35, BACK-46, BACK-47, BACK-48 and TRAF-45
own that difference.

## Precision levels and their contract

Coupling mode says how much of the loop is closed. Precision level says
how much detail each seam spends time on, and the two are independent.
The seam matrix, the owning tasks and the current state live in the
[developer guide](README_PRO.md#fidelity-levels-and-switches); this
section states the contract every level must satisfy.

1. **Semantics are level-invariant.** A level may change a duration or
   its variance. It may not change token identity, output length, stop
   reason, scheduler decisions, collective participants or payload
   shape. Any level that would change those is a different model, not a
   different precision, and must be registered as such.
2. **Every seam names a compatibility level.** That level's accepted
   artifacts stay byte-identical as other levels are added, which is what
   makes the rest of the ladder safe to extend.
3. **Deterministic and calibrated levels are labeled, never mixed
   silently.** A deterministic level returns one value per input. A
   calibrated level returns a draw from a distribution fitted offline
   against captured evidence, and must carry the fit provenance, the
   calibration envelope it is valid within, and the seed that reproduces
   the draw. A result produced at a calibrated level is reported with its
   distributional claim, not as a point estimate.
4. **The run records its configuration.** A result is only interpretable
   with the precision that produced it, so the selected level of every
   seam belongs in the run provenance. CORE-36 owns making that one
   validated surface rather than the current per-seam mixture of provider
   objects, profile strings, manifests, build options and environment
   variables.
5. **Cross-checking is an explicit mode, not an accident.** Where two
   paths can execute the same schedule, as the ATLAHS GOAL path and the
   runtime's own dependency realization both can, exactly one is the
   authority for a given run and the other may be selected as a
   cross-check whose disagreements are reported. TRAF-12 owns that
   reconciliation.


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
and the CPU pre-play oracle ([modules/preplay.md](modules/preplay.md)) for
each request's true output length, stop reason and expert routing, with error
bounds reported per metric.

Accuracy validation advances only after calibrated compute; the coarse
resource runtime landed with CORE-4, so its registered approximations and
their calibration, not its existence, now gate this stage. Use identical
framework commit, model, parallelism, request trace, seed and warm-up policy
in simulation and silicon. Progress
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
copy-descriptor setup plus bandwidth. The post-specified regression study is
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
TRAF-10 owns; only inter-node segments become GOAL traffic. The post-specified
regression study is [examples/gpu_task_mix](../examples/gpu_task_mix/expectations.md),
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
open. The inter-operation `DeviceRuntime` in CORE-4 is complete for the
coordinated first coarse bypass profile and the frozen Tier B structural
fixture, with its residual approximations registered as CORE-11 through
CORE-14, CORE-16 and CORE-21; CORE-21 has since closed.
