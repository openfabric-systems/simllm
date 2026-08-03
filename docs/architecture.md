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
frameworks match prefixes on actual token IDs — prefix-cache hit rates must be
emergent from the workload, not assumed.

### Framework frontend (adapters, `simllm/adapters/`)

The framework's scheduler, batching policy and KV/prefix-cache accounting run
unmodified; only model execution is replaced. Both adapters reduce to the same
contract: per scheduler step, a **step record** (which requests ran, how many
prefill/decode tokens each, how many tokens were served from cache) goes to
the core, and a **step result** (simulated step latency, flow completions)
comes back.

**vLLM (v1 engine, ≥ 0.14):** no fork needed. The engine's step loop is
`EngineCore.step()` → `Scheduler.schedule()` → `SchedulerOutput` →
`Executor.execute_model(scheduler_output)` → `ModelRunnerOutput` →
`Scheduler.update_from_output()`. The executor class is pluggable:
`--distributed-executor-backend` accepts a dotted import path, resolved in
`Executor.get_class()` (`vllm/v1/executor/abstract.py`). SimLLM ships
`simllm.adapters.vllm.SimExecutor`, which

- serves the init-time RPCs (`get_kv_cache_spec`, `determine_available_memory`,
  `initialize_from_config`, `compile_or_warm_up_model`, `initialize_cache`,
  `get_supported_tasks`) with model-derived values — the simulated vRAM size is
  pinned via `CacheConfig.num_gpu_blocks_override`;
- fabricates `ModelRunnerOutput(req_ids, req_id_to_index, sampled_token_ids)`
  per step and attaches simulated timing;
- exports the placement manifest (below) from the workers via
  `collective_rpc`, for both simulated and real capture runs.

The vLLM v1 KV-cache manager, block pool, prefix-block hashing and preemption
logic are pure CPU-side bookkeeping inside the scheduler process, so they run
for real under the simulated executor.

**SGLang:** the seam is the TP worker. SimLLM implements
`SimTpModelWorker(BaseTpWorker)` whose `forward_batch_generation(batch)`
returns a `GenerationBatchResult` with fabricated `next_token_ids` and
simulated timing; it is selected at the scheduler's worker-construction point
(`Scheduler.init_tp_model_worker`), the same seam SGLang already uses to swap
in platform-specific workers. First iteration runs with
`--disable-overlap-schedule`. RadixCache prefix matching, eviction, and the
token/request pool accounting are scheduler-side index bookkeeping and stay
real — so radix hit rates and vRAM pressure respond to the workload exactly as
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

**Placement manifest** (`simllm-placement-manifest-v1`) — per global rank:
host, local rank, GPU UUID / PCI bus ID, group memberships with the *actual*
global rank lists (export `GroupCoordinator.ranks` at runtime rather than
recomputing from the `((dp·PP+pp)·PCP+pcp)·TP+tp` formula — external DP,
elastic behavior or implementation changes break derived layouts), the
pipeline layer range `[start, end)` taken from the model (partitions are not
guaranteed equal), and per-MoE-layer local global expert IDs. A GPU generally
owns a *product* of shards simultaneously — a PP layer range × a TP slice ×
a DP replica index × a set of local experts — so the manifest records all
coordinates, not a single "shard id". Manifests are either **declared**
(what-if placements for simulation) or **extracted** from a live run — in
vLLM, one `collective_rpc` over the workers (via a callable or a worker
extension class, still no fork), recording the framework version since this
is an internal surface. Both modes share one schema, which is what makes
simulated and real deployments directly comparable.

With dynamic expert load balancing (EPLB), expert ownership changes at
runtime: each re-placement bumps a monotonically increasing
`placement_epoch`, the physical↔logical expert maps are snapshotted per
epoch, and every traffic record references the epoch it was routed under.
With redundant experts, the logical expert ID alone does not identify the
destination rank — the dispatcher's selected physical expert (or destination
rank) is recorded.

**Fabric topology manifest** (`simllm-fabric-topology-v1`) — nodes, GPUs,
PCIe/NVLink links, NICs, GPU→NIC affinity, switches, links, bandwidths,
delays, queue configuration. Intra-node structure can be taken from NCCL's
detected topology (`NCCL_TOPO_DUMP_FILE`); the switch-level graph always
comes from a cluster inventory or the simulator topology config — NCCL's
local discovery is not a description of the routed network.

The **mapper** resolves every rank in a communication event to a physical
endpoint (`rank → node → GPU → NIC`) and assigns GOAL ranks. GOAL rank
assignment mirrors the htsim RNIC drivers' `-goal_rank_mapping` option:
`gpu-rank` (one GOAL rank per GPU) or `unique-nic` (one per NIC; intra-node
transfers stay off the fabric).

### Core (`simllm/core/`, `simllm/traffic/`, `simllm/goal/`)

- **Virtual clock** — orders request arrivals and step completions.
- **Compute-cost model** — calibrated per-(GPU, model) profiles mapping
  (prefill tokens, decode batch size) to kernel time, with uncertainty bounds.
- **Traffic model** — consumes three inputs: a *collective trace*
  (`simllm-collective-trace-v1`, one JSONL record per op: step, layer, op,
  group type, group global ranks, send counts, element bytes, placement
  epoch, release time), the placement manifest, and the fabric manifest. For
  MoE, the static map `expert_owners[layer][global_expert_id] → ranks` turns
  routed tokens into all-to-allv destinations (per placement epoch). Semantic
  collectives are expanded into the algorithm actually used — ring, tree,
  pairwise all-to-allv, or a custom collective-network schedule — as chunked
  send/recv chains. Covers TP collectives per layer, MoE dispatch/combine
  (optionally driven by captured per-token expert routings), PP activations,
  and KV-cache transfers (PD-disaggregation, cache-miss re-prefill). For
  simulating communication patterns, group memberships + activation shapes
  suffice; exact TP weight-storage intervals (packed QKV, gate/up packing,
  quantization padding) are deliberately out of scope.
- **GOAL emitter** — renders the step DAG as a GOAL trace (below).

### Network backend (`simllm/backends/`, `third_party/`)

The GOAL trace is executed by a discrete-event simulator:

- **htsim** (packet-level): `htsim_uec -goal <bin> -topo <topo>` executes the
  GOAL schedule over a Clos topology with full transport behavior. The RNIC
  model series (null-network baselines `rnic-nn`/`rnic-nn-fluid`, the
  explicit-rate collective-network endpoint `rnic-cn`, DCQCN over a VoQ
  traffic-manager switch, and a Slingshot-like profile `rnic-ss`) provides
  fidelity profiles; CLI wiring for the RNIC profiles is being upstreamed to
  the htsim submodule's `main`.
- **LogGOPSim** (flow-level, fast): same GOAL input, LogGOP cost model —
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

- `calc <cost>` — local compute; `send/recv <size>b to/from <peer> tag <t>`
  — point-to-point transfers. Optional `cpu <c>` / `nic <n>` clauses pin ops
  to resources.
- `a requires b` — `a` starts after `b` *finishes*; `a irequires b` — `a`
  starts after `b` *starts*.

Collectives are decomposed into send/recv chains by the emitter, so the
network simulator sees the real chunked traffic pattern, not an abstract
collective op.

## Coupling modes

1. **Offline (open-loop).** The frontend runs to completion under the sim
   executor (fast, no network in the loop); every step is recorded; one GOAL
   trace is emitted and simulated once. Cheap, deterministic, but network
   congestion cannot influence batch composition.
2. **Closed-loop.** Each scheduler step (or window of steps) is simulated
   before the next one is released; the network's completion time advances the
   virtual clock the scheduler sees. The step/result exchange uses versioned
   JSON manifests (`atlahs-closed-loop-step-v1` / `atlahs-closed-loop-result-v1`,
   see `simllm/bridge/`). Per-step subprocess invocation is the diagnostic
   mode; a persistent co-simulator process is planned for scale.

## Timing and metrics

An instantly-returning simulated executor breaks metric *meaning*, not metric
plumbing: serving frameworks compute TTFT/TPOT from wall-clock timestamps
spanning two processes. SimLLM handles this in two ways:

- **Paced mode** — the sim executor delays completion by the simulated step
  latency; every stock metric works unchanged (sim runs in scaled real time).
- **Virtual mode** — the executor returns immediately and SimLLM reports
  sim-native metrics (per-request TTFT/TPOT/queueing delay on the virtual
  clock) through its own metrics pipeline, bypassing the framework's
  wall-clock histograms.

## Validation

Every simulated configuration should carry provenance (backend profile,
topology, calibration profile, seeds) and be checked against real captures
where available: single-node runs for compute-model calibration, multi-node
NCCL traces (via the ATLAHS capture pipeline) for network-model validation,
with error bounds reported per metric.
