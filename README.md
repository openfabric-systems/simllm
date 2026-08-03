<h1 align="center">SimLLM</h1>

<h3 align="center">
Network-faithful simulation of LLM serving and training deployments
</h3>

<p align="center">
| <a href="#about"><b>About</b></a> | <a href="#architecture"><b>Architecture</b></a> | <a href="#getting-started"><b>Getting Started</b></a> | <a href="#demo"><b>Demo</b></a> | <a href="#tutorials"><b>Tutorials</b></a> | <a href="#roadmap"><b>Roadmap</b></a> | <a href="#open-task-registry"><b>Open Tasks</b></a> | <a href="#contributing"><b>Contributing</b></a> |
</p>

## About

SimLLM predicts the serving performance (TTFT, TPOT, goodput, SLO attainment)
of large LLM deployments **before you buy or reserve the hardware**, with a
packet-level network underneath rather than a `bytes / bandwidth` estimate.

Modern serving stacks (vLLM, SGLang) treat deployment as a GPU scheduling
problem. At 4+ nodes, and especially at 64+, the network stops being a
constant: incast, queue buildup, congestion-control oscillation and
head-of-line blocking reshape TTFT/TPOT distributions in ways no analytic
model captures. SimLLM couples the **real frontend scheduler** of your
serving framework with a discrete-event, packet-level network backend
(ATLAHS + htsim) so that scheduling, KV/prefix-cache management, and the
fabric feed back on each other.

**Key ideas**

- **Frontends are real, GPUs are simulated.** The framework's own scheduler,
  batching policy, and prefix/KV-cache accounting run unmodified (they are
  CPU-side bookkeeping); only model execution is replaced by a calibrated
  compute-cost model plus simulated network time.
- **Framework-agnostic core, thin adapters.** vLLM and SGLang differ exactly
  where it matters: request scheduling, RadixCache vs block-hash prefix
  caching, vRAM accounting. Each framework plugs in behind one common
  scheduler-step interface:
  - vLLM: a `SimExecutor` selected with the existing
    `--distributed-executor-backend simllm.adapters.vllm.SimExecutor` flag
    (no fork required).
  - SGLang: a `SimTpModelWorker` selected at the scheduler's worker-class
    seam.
- **Workload as a queueing model.** Arrival processes (Poisson, bursty, trace
  replay) and prompt/output length distributions, with controllable shared-
  prefix structure, drive the frontend, so prefix-hit probability, cache-miss
  re-prefill traffic, and vRAM pressure are emergent rather than assumed.
- **Network with fidelity profiles.** GOAL dependency graphs (compute / send /
  recv) are executed by htsim at packet granularity: RoCEv2-style DCQCN,
  explicit-rate collective-network endpoints, Slingshot-like adaptive routing
  (planned), over Clos topologies with detailed switch models (VoQ traffic
  manager, request/grant input-buffered).
- **Pluggable compute-time fidelity.** The duration of every simulated
  compute region comes from a `ComputeProvider`: measured profile tables
  from real captures, an analytical roofline model (classifying each kernel
  as compute- or memory-bound from its configuration alone), or, planned,
  offline SASS-level cycle simulation (Accel-Sim / GPGPU-Sim) used to
  populate profile tables for configurations nobody measured. Cycle-accurate
  GPU simulation never sits inside the step loop; the loop always reads
  tables or analytical estimates.

SimLLM is developed in the open. It is initiated and sponsored by
**OpenFabric** (we design network hardware to enable distributed AI
workloads for the AGI era) and welcomes community contributions.

## Architecture

```
   Workload Generator ──► Framework Frontend ──► SimLLM Core ──► Network Backend
   arrivals, lengths,     vLLM  : SimExecutor    virtual clock,   ATLAHS GOAL →
   prefix structure       SGLang: SimTpWorker    compute model,   htsim (packet-level
   (queueing model)       real scheduler +       DAG → GOAL       RNIC + switch models)
                          KV/prefix cache        emission              │
        ▲                                                              │
        └───────────── completion times feed back (closed loop) ◄──────┘
```

Two coupling modes:

1. **Offline (open-loop):** run the frontend fast with the sim executor,
   record every scheduler step, emit one GOAL trace, simulate once, report.
2. **Closed-loop:** the network's completion times advance the virtual clock
   that the frontend schedules against, so congestion changes batching, and
   batching changes traffic.

A key design point: serving frameworks are **topology-light, not
topology-agnostic**. vLLM knows logical ranks, TP/PP/DP/EP groups and a
rank-to-worker placement, but not the switch-level graph; NCCL knows the
intra-node PCIe/NVLink/NIC topology; only the network simulator knows links,
routing and queueing. SimLLM therefore joins two independent descriptions,
a **placement manifest** (rank to node to GPU to shard to groups to local
experts) and a **fabric topology manifest** (GPU to PCIe/NVLink to NIC to
switch to links), via the mapper, and every communication event is resolved
through both.

Host-side effects are deliberately out of the default model: the inter-rank
doorbell (the small packet that releases the next rank) is simulated
in-band as a high-priority control message on the fabric, while the host
launch path (CPU proxy vs GPU-initiated networking, PCIe, RNIC doorbell)
defaults to zero delay and zero jitter. A single per-endpoint
initiation-delay constant exists for launch-path studies.

### Modules

| Module | Purpose |
|---|---|
| `simllm/core` | Virtual clock, scheduler-step records |
| `simllm/workload` | Arrival processes, length distributions, shared-prefix structure |
| `simllm/compute` | Pluggable compute-time providers (measured profile tables, analytical roofline, offline SASS-level simulation) + host initiation model (doorbell launch path) |
| `simllm/placement` | **The mapper**: placement manifest (rank to node/GPU/shard/groups, expert ownership, EPLB epochs), fabric topology manifest, rank-to-endpoint/GOAL-rank resolution |
| `simllm/traffic` | Semantic collectives to physical flows: TP/PP/DP collectives, MoE all-to-allv from expert owners, KV-cache transfers |
| `simllm/goal` | GOAL dependency-graph trace emission |
| `simllm/backends` | htsim (packet-level) and LogGOPSim (flow-level) invocation + result parsing |
| `simllm/adapters/vllm` | `SimExecutor` (pluggable, no fork) + shard/placement manifest exporter |
| `simllm/adapters/sglang` | `SimTpModelWorker` + placement exporter |
| `simllm/bridge` | Closed-loop step/result manifest schemas |

See [docs/architecture.md](docs/architecture.md) for the full design,
including the exact integration seams in vLLM and SGLang, the manifest
schemas and the GOAL trace format.

## Getting Started

> ⚠️ Early stage: the scaffold, package layout and backends are in place;
> the first runnable end-to-end pipeline lands with milestone M1 below.

```bash
git clone https://github.com/yifeng-ethz/simllm.git
cd simllm

# Backends (~250 MB). Do NOT use --recursive: ATLAHS carries large nested
# application submodules that are not needed for simulation.
git submodule update --init third_party/atlahs third_party/htsim

pip install -e .

# Build the packet-level network simulator
cmake -S third_party/htsim/htsim/sim -B build/htsim -DCMAKE_BUILD_TYPE=Release
cmake --build build/htsim --parallel
```

Pinned backends:

| Submodule | Repo | Ref |
|---|---|---|
| `third_party/atlahs` | [ATLAHS-rnic-private](https://github.com/yifeng-ethz/ATLAHS-rnic-private) | `codex/rnic-rewrite` (GOAL toolchain + RNIC launcher) |
| `third_party/htsim` | [HTSIM-rnic-private](https://github.com/yifeng-ethz/HTSIM-rnic-private) | `main` (UEC htsim + RNIC model libraries) |

## Demo

`demo/` will contain a full integration walk-through: workload definition,
vLLM frontend with `SimExecutor`, GOAL emission, htsim packet simulation,
and a TTFT/TPOT report with bottleneck attribution.

## Tutorials

- Using parts of the library standalone (workload generation, GOAL emission,
  htsim invocation): `docs/tutorials/`
- Writing a frontend adapter for a new framework: `docs/tutorials/`

## Roadmap

- [x] M0: repo scaffold, backend submodules, CI
- [ ] M1: standalone core: workload gen to GOAL to htsim to metrics (no frontend)
- [ ] M2: vLLM adapter (`SimExecutor`, offline mode)
- [ ] M3: SGLang adapter (`SimTpModelWorker`; RadixCache-aware KV traffic)
- [ ] M4: closed loop + calibration/validation against real captures
- [ ] M5: RNIC fidelity profiles wired end to end (explicit-rate CN,
  Slingshot-like); MoE expert-parallel traffic from real routing captures;
  training workloads; SASS-level (Accel-Sim/GPGPU-Sim) offline profile
  generation
- [ ] M6: PD-disaggregation and KV-transfer traffic modeling

## Open task registry

Work that is intentionally deferred, so nothing silently disappears. Items
are added here when a decision defers them and removed when they land.

| Area | Open item | Target |
|---|---|---|
| htsim | `rnic-ss` (Slingshot-like) profile wiring; the runtime factory rejects it until the Slingshot PR lands | backend [5/5] follow-up |
| htsim | Goodput/state/queue trace CLI flags for `rnic-cn` (need trace hooks in the reviewed runtime first) | backend follow-up |
| htsim | GOAL-driven DCQCN profile (`htsim_dcqcn_atlahs`) | backend follow-up |
| htsim | GOAL parser hardening + `htsim_goal_txt2bin` tool | backend follow-up |
| ATLAHS | Correct the vendored-fallback wording (vendored htsim cannot satisfy the resolver); pin a known-good HTSIM commit | backend follow-up |
| simllm | Fabric topology schema (`simllm-fabric-topology-v1`) + NIC selection in the mapper | M4 |
| simllm | `unique-nic` GOAL-rank mapping (depends on the fabric manifest) | M4 |
| simllm | Persistent co-simulator process for closed loop (per-step subprocess is the diagnostic mode) | M4 |
| simllm | Calibrated host-initiation profiles (GPU-initiated vs CPU-proxy constants) | M4/M5 |
| simllm | SASS-level offline profile generation (Accel-Sim / GPGPU-Sim table generator) | M5 |
| simllm | `ProfileTableProvider` interpolation across uncovered configs | M5 |
| simllm | EPLB epoch-snapshot wiring in collective-trace records | M5 |

## Contributing

We welcome contributions of every size; see [CONTRIBUTING.md](CONTRIBUTING.md).
Good first areas: workload generators, compute-cost calibration profiles for
new GPUs, topology configs, metrics/plotting.

## License

SimLLM is licensed under [Apache-2.0](LICENSE). The backend submodules keep
their own permissive licenses (htsim: BSD 2-Clause, UEC/UCL/UPB/Broadcom;
ATLAHS: MIT, SPCL).

## Design & References

- RFC: [Network backend simulation for vLLM](https://discuss.vllm.ai/t/network-backend-simulation-for-vllm/2812)
- ATLAHS: application-centric network simulator toolchain (SPCL, SC'25)
- htsim: packet-level datacenter network simulator (UCL / Broadcom / UEC)
- LogGOPSim: GOAL-driven LogGOPS simulator (Hoefler et al.)
