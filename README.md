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
  - SGLang: a `SimTpModelWorker` installed through SGLang's own plugin
    framework (an entry point applying a replace hook at the scheduler's
    worker-construction seam; no fork required, inert unless explicitly
    enabled).
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

# Run the M1 sanity studies: probes + bandwidth/parallelism sweeps +
# pipeline-parallel TTFT/TPOT on the default 8-node x 8-GPU 400G Clos
python examples/m1/run_m1.py --out runs/m1
```

Pinned backends (details in [docs/modules/backends.md](docs/modules/backends.md)):

| Submodule | Repo | Ref |
|---|---|---|
| `third_party/atlahs` | [ATLAHS-rnic-private](https://github.com/yifeng-ethz/ATLAHS-rnic-private) | `main` (GOAL toolchain + validated RNIC launcher) |
| `third_party/htsim` | [HTSIM-rnic-private](https://github.com/yifeng-ethz/HTSIM-rnic-private) | `main` (UEC htsim + `htsim_rnic`: rnic-nn, rnic-nn-fluid, rnic-cn) |

## Demo

[examples/m4](examples/m4/) is the current flagship demo: the closed loop.
A real vLLM v0.26.0 engine runs in-process at `tensor_parallel_size=8`
under the `SimExecutor` (no GPUs touched), and every scheduler step's
tensor-parallel traffic is executed by `htsim_rnic` at packet granularity
before the step's completion time advances the scheduler's clock. All 36
pre-registered checks pass, the fluid-profile closed forms to a residual
of 0 ps ([expectations](examples/m4/expectations.md),
[results](examples/m4/RESULTS.md)).

[examples/m1](examples/m1/) remains the standalone-core demo: workload to
GOAL to `htsim_rnic` to TTFT/TPOT with per-flow FCT as the debug layer,
validated the same way ([results](examples/m1/RESULTS.md)). The
[examples/cn_ladder](examples/cn_ladder/) study compares the fidelity
profiles (rnic-cn vs DCQCN under incast and all-to-all) and carries the
definitive comparator figures.

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
- [x] M1: standalone core: workload gen to GOAL to `htsim_rnic` to metrics,
  no frontend. Virtual clock, length distributions, collective patterns
  (scatter/gather, ring allreduce, all-to-allv, binomial tree), `txt2bin`
  conversion, direct `htsim_rnic` invocation with FCT parsing and
  nn-normalized FCT. Validated by pre-registered sanity studies sweeping
  bandwidth and parallelism, then independently audited: 15 of 18 checks
  pass (ten runs reproduce their closed forms with zero picosecond
  residual); the three misses are traced to mis-registered expectations,
  each with a closed ledger ([examples/m1](examples/m1/RESULTS.md)).
- [x] M2: vLLM adapter, pinned to v0.26.0. `SimExecutor` services the full
  init and step RPC surface, translates every scheduler step into a step
  record (streamed JSONL, schema-tagged), refuses what fabricated tokens
  would silently corrupt (speculative decoding, structured output), and
  the `PlacementExporter` extracts placement manifests from real runs.
  Independently audited (19 findings folded) and validated end to end
  against a live engine (2026-08-04). Remaining halves are numbered tasks
  in [adapters-vllm](docs/modules/adapters-vllm.md).
- [x] M3: SGLang adapter, pinned to a fresh main-branch commit. SGLang now
  ships a first-class plugin framework, so `SimTpModelWorker` installs
  with no fork through an entry point (inert unless
  `SIMLLM_SGLANG_ENABLE=1`), fabricates CPU-resident pools so RadixCache
  and retraction stay real, and passed a live CPU-engine smoke on the
  pinned commit (2026-08-04). Remaining halves in
  [adapters-sglang](docs/modules/adapters-sglang.md).
- [ ] M4 (in progress): closed loop validating M2/M3. The loop itself
  landed and is validated: `HtsimStepSink` runs each step's TP traffic
  through `htsim_rnic` (diagnostic per-step mode), a live vLLM tp=8 run
  closed the loop with 0 ps residual against pre-registered closed forms
  ([examples/m4](examples/m4/RESULTS.md)). Remaining: the persistent
  co-simulator (BRIDGE-1), fabric manifest + NIC selection (PLACE-1/2),
  and calibration against real captures.
- [ ] M5 (in progress): all-to-all traffic studies (MoE expert-parallel)
  with the focus on SASS-level (Accel-Sim/GPGPU-Sim) offline calibration
  of the compute model (the calibration plan is recorded in
  [compute](docs/modules/compute.md)); training workloads. Slingshot is
  out of scope for simllm (the `rnic-ss` profile remains a backend-repo
  follow-up only).
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
