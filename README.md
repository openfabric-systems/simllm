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
   prefix structure       SGLang: SimTpModel-    compute model,   htsim (packet-level
                                  Worker
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

Each module has its own doc as the source of truth for design, current
status and numbered open tasks; the README stays a map.

| Module | Purpose | Doc |
|---|---|---|
| `simllm/core` | Virtual clock, scheduler-step records, closed-loop wire schemas | [core](docs/modules/core.md) |
| `simllm/workload` | Arrival processes, length distributions, shared-prefix structure | [workload](docs/modules/workload.md) |
| `simllm/compute` | Pluggable compute-time providers + host initiation model | [compute](docs/modules/compute.md) |
| `simllm/placement` | **The mapper**: placement + fabric manifests, rank-to-endpoint/GOAL-rank resolution | [placement](docs/modules/placement.md) |
| `simllm/traffic` | Semantic collectives to physical flows | [traffic](docs/modules/traffic.md) |
| `simllm/goal` | GOAL dependency-graph trace emission | [goal](docs/modules/goal.md) |
| `simllm/backends` | htsim / LogGOPSim invocation + result parsing, submodule pins | [backends](docs/modules/backends.md) |
| `simllm/adapters/vllm` | `SimExecutor` (pluggable, no fork) + placement exporter | [adapters-vllm](docs/modules/adapters-vllm.md) |
| `simllm/adapters/sglang` | `SimTpModelWorker` + placement exporter | [adapters-sglang](docs/modules/adapters-sglang.md) |

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

Pinned backends (details in [docs/modules/backends.md](docs/modules/backends.md)):

| Submodule | Repo | Ref |
|---|---|---|
| `third_party/atlahs` | [ATLAHS-rnic-private](https://github.com/yifeng-ethz/ATLAHS-rnic-private) | `main` (GOAL toolchain + validated RNIC launcher) |
| `third_party/htsim` | [HTSIM-rnic-private](https://github.com/yifeng-ethz/HTSIM-rnic-private) | `main` (UEC htsim + `htsim_rnic`: rnic-nn, rnic-nn-fluid, rnic-cn) |

## Demo

`demo/` will contain a full integration walk-through: workload definition,
vLLM frontend with `SimExecutor`, GOAL emission, htsim packet simulation,
and a TTFT/TPOT report with bottleneck attribution.

## Tutorials

Planned, will live in `docs/tutorials/`: using parts of the library
standalone (workload generation, GOAL emission, htsim invocation), and
writing a frontend adapter for a new framework. Until then, each module doc
above is the reference.

## Roadmap

- [x] M0: repo scaffold, backend submodules, CI, per-module docs. Landed
  ahead of schedule on the backend side: `htsim_rnic` with the `rnic-nn`,
  `rnic-nn-fluid` and `rnic-cn` fidelity profiles and the validated ATLAHS
  launcher are merged and pinned (2026-08-03), so packet-level RNIC runs
  work from a fresh clone today.
- [ ] M1 (next): standalone core: workload gen to GOAL to `htsim_rnic` to
  metrics, no frontend; first JCT sanity studies sweeping bandwidth and
  parallelism.
- [ ] M2: vLLM adapter (`SimExecutor`, offline mode) + placement-manifest
  exporter.
- [ ] M3: SGLang adapter (`SimTpModelWorker`; RadixCache-aware prefix-hit
  and vRAM studies).
- [ ] M4: closed loop (persistent co-simulator), fabric manifest + NIC
  selection, calibration/validation against real captures.
- [ ] M5: Slingshot-like `rnic-ss` profile end to end; MoE expert-parallel
  traffic from real routing captures; training workloads; SASS-level
  (Accel-Sim/GPGPU-Sim) offline profile generation.
- [ ] M6: PD-disaggregation and KV-transfer traffic modeling.

## Open task registry

Open tasks are tracked in each module's doc with stable numbered IDs
(`PREFIX-N`, e.g. `PLACE-1`, `HTSIM-2`); an ID is added in the change that
defers the work and closed by the change that completes it, never renumbered.
This section is only the index:

- [core](docs/modules/core.md): CORE-*, plus BRIDGE-* inherited from the
  folded bridge module
- [workload](docs/modules/workload.md): WORK-*
- [compute](docs/modules/compute.md): COMP-*
- [placement](docs/modules/placement.md): PLACE-*
- [traffic](docs/modules/traffic.md): TRAF-*
- [goal](docs/modules/goal.md): GOAL-*
- [backends](docs/modules/backends.md): BACK-*, plus backend-repo
  follow-ups HTSIM-* and ATLAHS-*
- [adapters-vllm](docs/modules/adapters-vllm.md): VLLM-*
- [adapters-sglang](docs/modules/adapters-sglang.md): SGL-*

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
