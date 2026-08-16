<p align="center">
  <img src="resources/logo/openfabric-logo-mark.png" width="80" alt="OpenFabric">
</p>

<h1 align="center">SimLLM</h1>

<h3 align="center">
Network-faithful simulation of LLM serving and training deployments
</h3>

<p align="center">
| <a href="#about"><b>About</b></a> | <a href="#architecture"><b>Architecture</b></a> | <a href="#getting-started"><b>Getting Started</b></a> | <a href="#demo"><b>Demo</b></a> | <a href="#models"><b>Models</b></a> | <a href="#modules"><b>Modules</b></a> | <a href="#development"><b>Development</b></a> | <a href="#contributing"><b>Contributing</b></a> | <a href="docs/README_PRO.md"><b>Pro Guide</b></a> |
</p>

<p align="center">
<img src="resources/figures/simllm-overview.png" width="58%" alt="SimLLM pipeline: a workload model feeds the real vLLM or SGLang scheduler, which drives a simulated GPU executor and a packet-level htsim network, producing TTFT, TPOT and goodput on a virtual clock, with step completions closing the loop back to the scheduler">
</p>

## About

SimLLM predicts the serving performance (TTFT, TPOT, goodput, SLO
attainment) of large LLM deployments **before you buy or reserve the
hardware**, with a packet-level network underneath rather than a
`bytes / bandwidth` estimate.

At 4+ nodes, and especially at 64+, the network stops being a constant:
incast, queue buildup, congestion-control oscillation and head-of-line
blocking reshape TTFT/TPOT distributions in ways no analytic model
captures. SimLLM couples the **real frontend scheduler** of your serving
framework with a simulated GPU executor and a discrete-event,
packet-level network backend (ATLAHS + htsim), so scheduling,
KV/prefix-cache management and the fabric feed back on each other.

Three ideas carry the design:

- **Frontends are real, GPUs are simulated.** The framework's own
  scheduler, batching policy and KV/prefix-cache accounting run
  unmodified (they are CPU-side bookkeeping); only model execution is
  replaced by a calibrated compute-cost model plus simulated network
  time.
- **No forks.** vLLM plugs in through its existing
  `--distributed-executor-backend` flag; SGLang through its own plugin
  framework (inert unless explicitly enabled). Each framework sits
  behind one common scheduler-step interface.
- **The network is simulated at packet granularity.** Every scheduler
  step's collective traffic (tensor-parallel allreduce, MoE all-to-all,
  pipeline activations) runs through htsim RNIC models over a Clos
  fabric, and the simulated completion times can feed back into the
  scheduler's clock.

SimLLM is developed in the open. It is initiated and sponsored by
**OpenFabric** (we design network hardware to enable distributed AI
workloads for the AGI era) and welcomes community contributions.

### Supported by

SimLLM's architecture, its models and its ongoing maintenance are
developed with the support of ETH Zurich, the Swiss National
Supercomputing Centre (CSCS), the Scalable Parallel Computing
Laboratory (SPCL) at ETH Zurich, SLAC National Accelerator Laboratory,
Stanford University and the National Energy Research Scientific
Computing Center (NERSC) at Lawrence Berkeley National Laboratory.

<p align="center"><a href="https://ethz.ch"><img src="resources/figures/supporters/tile-ethz.png" height="64" alt="ETH Zurich"></a><a href="https://www.cscs.ch"><img src="resources/figures/supporters/tile-cscs.png" height="64" alt="Swiss National Supercomputing Centre (CSCS)"></a><a href="https://spcl.inf.ethz.ch"><img src="resources/figures/supporters/tile-spcl.png" height="64" alt="Scalable Parallel Computing Laboratory (SPCL)"></a><a href="https://www6.slac.stanford.edu"><img src="resources/figures/supporters/tile-slac.png" height="64" alt="SLAC National Accelerator Laboratory"></a><a href="https://www.stanford.edu"><img src="resources/figures/supporters/tile-stanford.png" height="64" alt="Stanford University"></a><a href="https://www.nersc.gov"><img src="resources/figures/supporters/tile-nersc.png" height="64" alt="NERSC"></a></p>

## Architecture

```
workload model (arrivals, prompt/output lengths, shared prefixes)
        |
        v
real framework scheduler (vLLM / SGLang, unmodified)
        |    scheduler step: which requests, which tokens, cache hits
        v
simulated GPU executor (roofline, profile tables, or trace-driven service)
        |    per-step collectives: TP allreduce, MoE all-to-all, PP
        v
packet-level network simulator (htsim RNIC models on a Clos fabric)
        |
        v
TTFT / TPOT / goodput on a virtual clock
```

Two coupling modes:

1. **Offline (open-loop):** run the frontend fast, record every
   scheduler step, emit one GOAL dependency trace, simulate once,
   report.
2. **Closed-loop:** each step's simulated completion time advances the
   virtual clock the scheduler sees, so congestion changes batching and
   batching changes traffic.

Under the hood, every scheduler step is lowered to a versioned execution
graph, and the GPU itself is modeled rather than assumed. Its service-model
primitive can schedule explicitly submitted compute, memory and NCCL network
kernels together so each finds its own limit, and the coarse DeviceRuntime
(CORE-4) now lowers graph operations into that primitive for the first
coordinated bypass profile and the frozen Tier B fixture. The full
design, including the exact vLLM/SGLang integration
seams, the manifest schemas and the GOAL trace format, is in
[docs/architecture.md](docs/architecture.md). The developer map
(module status, contracts, open tasks, development process) is in
[docs/README_PRO.md](docs/README_PRO.md).

## Getting Started

```bash
git clone https://github.com/openfabric-systems/simllm.git
cd simllm

# Backends (~250 MB). Do NOT use --recursive: ATLAHS carries large nested
# application submodules that are not needed for simulation.
git submodule update --init third_party/atlahs third_party/htsim

pip install -e .
```

Build HTSIM with a C++17 toolchain. On Linux, use GCC or Clang with
CMake 3.16 or newer:

```bash
./scripts/build_htsim.sh build/htsim --test
```

This builds the default `htsim_rnic`. The composed binary that links the
SimLLM RNIC library into htsim, which the `rnic_live_v1` study uses, is built
behind the `HTSIM_ENABLE_SIMLLM_RNIC` option described in
[docs/modules/backends.md](docs/modules/backends.md).

On Windows, install Visual Studio 2022 Build Tools with the **Desktop
development with C++** workload and CMake, then run in PowerShell:

```powershell
.\scripts\build_htsim.ps1 -BuildDirectory build\htsim -RunTests
```

The Windows build uses MSVC and places executables in CMake's
configuration directories, for example
`build/htsim/datacenter/Release/htsim_rnic.exe`. SimLLM automatically
discovers both that layout and the single-configuration Linux layout.
The `SIMLLM_HTSIM_RNIC`, `SIMLLM_HTSIM_DCQCN`, and `SIMLLM_TXT2BIN`
environment variables can override discovery.

The vLLM and SGLang adapters and traffic models are Python code and do
not need a Windows-specific fork; on Windows they invoke the native
HTSIM `.exe` files through the same backend interface.

Run a sanity study after building:

```bash
# Run the M1 sanity studies: probes + bandwidth/parallelism sweeps +
# pipeline-parallel TTFT/TPOT on the default 8-node x 8-GPU 400G Clos
python examples/m1/run_m1.py --out runs/m1
```

Pinned backends (details in [docs/modules/backends.md](docs/modules/backends.md)):

| Submodule | Repo | Ref |
|---|---|---|
| `third_party/atlahs` | [ATLAHS-rnic-private](https://github.com/yifeng-ethz/ATLAHS-rnic-private) | `main` (GOAL toolchain + validated RNIC launcher) |
| `third_party/htsim` | [HTSIM-rnic-private](https://github.com/yifeng-ethz/HTSIM-rnic-private) | `main` (UEC htsim + `htsim_rnic`: rnic-nn, rnic-nn-fluid, rnic-cn; WQE bookkeeping; the composed SimLLM RNIC wrapper behind the `HTSIM_ENABLE_SIMLLM_RNIC` build option) |

## Demo

Every study under [examples/](examples/) is open to users and carries an
`expectations.md`, and every study that has run also carries a run script, an
audited `RESULTS.md` and explicit evidence provenance. Most new and extended
studies freeze expectations in their own commit before implementation and
execution; the ones that could not are labeled post-specified in their own
results. Start with these:

| Study | Question | Headline result |
|---|---|---|
| [rnic_live_v1](examples/rnic_live_v1/RESULTS.md) | **Does the native RNIC model change end-to-end serving metrics?** | Tier A runs the composed htsim binary with the SimLLM RNIC library linked in, and Tier B carries the native completion through the execution graph, runtime and `StepResult` into the first TTFT and TPOT claim affected by that chain; all 8 single-WQE and 4 FIFO rows exact, scoped to one frozen isolated fixture (one request, 4 KiB and 1 MiB payloads, 200 and 400 Gbit/s, doorbell service 0 and 1,000 ps) |
| [m4](examples/m4/RESULTS.md) | Does the closed loop work end to end? | A live vLLM engine at `tensor_parallel_size=8` runs under the `SimExecutor` with `htsim_rnic` inside the step loop; all 36 pre-registered checks pass, fluid closed forms to 0 ps |
| [dcqcn_micro](examples/dcqcn_micro/RESULTS.md) | **NIC calibration: message size and incast.** How does goodput scale with message size, and is incast bandwidth shared fairly? | Model tracks the real-NIC (UCCL) message-size anchors at saturation but undershoots at 64 to 256 KB (0.79x at 64 KB): WQ completion is BACK-9 and PCIe calibration is BACK-16 atop the landed BACK-10 fabric; persistent post-CNP DCQCN state is HTSIM-5; incast fair share is near-ideal (Jain 0.993 to 1.000 across fan-in 2 to 20) |
| [dcqcn_vs_cn](examples/dcqcn_vs_cn/RESULTS.md) | When does DCQCN collapse, and when does it honestly win? | Buffer-exceeding incast collapses DCQCN by 2 to 3 orders of magnitude (32x64 KiB: p99 slowdown 1161x vs rnic-cn 1.60); buffer-absorbed incast is a registered DCQCN win (1.07 vs 1.68) |
| [cn_ladder](examples/cn_ladder/RESULTS.md) | Does the explicit-rate `rnic-cn` endpoint meet its acceptance bar? | 46 of 49 incast ladder cells within the 20% target of the ideal baseline; under a lossy all-to-all, DCQCN p99 slowdown is 1902x vs rnic-cn 19.3x (lossless, deterministic) |
| [breakdown](examples/breakdown/RESULTS.md) | Where does request time actually go? | Network share of request time rises from 52% (TP=2) to 89% (TP=8) at 400G, 96% at 100G |
| [m1](examples/m1/RESULTS.md) | Standalone core: workload to GOAL to htsim to metrics | Ten runs reproduce their closed forms with 0 ps residual |
| [m5](examples/m5/RESULTS.md) | MoE expert-parallel all-to-all | Pairwise all-to-allv closed forms exact to 0 ps across size and width |
| [routed_supply_v1](examples/routed_supply_v1/RESULTS.md) | Does real expert routing change the all-to-all? | Captured granite routing replaces the uniform per-pair share and moves fluid JCT by about -59 percent at 200 Gbit/s and -48 percent at 400 Gbit/s; TRAF-25 later corrected both captured and uniform traffic from 96 to 48 positive flows per step while this study's JCTs and bandwidth deltas remained unchanged |
| [gpu_task_mix](examples/gpu_task_mix/RESULTS.md) | **GPU scheduling: what limits a compute, a memory and an NCCL kernel?** | Compute scales with SMs, memory is pinned to the HBM cursor and gains nothing from more SMs, and a double-buffered ring egress kernel falls from 6.1 times its NVLink bound at one warp per channel to within 2.4 percent at eight warps |
| [core2_lowering](examples/core2_lowering/RESULTS.md) | Execution lowering and WQE bookkeeping | Legacy path, graph-only replay and frozen closed form agree to 0 ps on all rows; flow and WQE ledgers field-identical |
| [rnic_wq_v1](examples/rnic_wq_v1/RESULTS.md) | **RNIC queueing: what do doorbell batches, signaling and network credits change?** | All 11 native C++ sweep cells match their closed forms exactly; batching cuts 32 doorbells to 2, signaling cuts 32 CQEs to 2 independently, and four network credits cut JCT from 16,110 ps to 4,140 ps |
| [rnic_pcie_v1](examples/rnic_pcie_v1/RESULTS.md) | **RNIC PCIe: how do transactions, finite resources and analytical path penalties change completion time?** | All 35 deterministic row oracles and 18 behavioral predicate instances pass; corrected link-queue accounting leaves every JCT unchanged, and ready posted traffic passes a blocked non-posted request in the frozen gap case |

The message-size calibration curve and the definitive comparator figure:

<p align="center">
<img src="examples/dcqcn_micro/plots/msg_size_vs_goodput.png" width="46%" alt="Goodput vs message size against real-NIC anchors">
<img src="examples/cn_ladder/plots/a2a16_lossy_fct_cdf_seeded.png" width="46%" alt="Seeded FCT CDF, DCQCN vs rnic-cn under a lossy all-to-all">
</p>

The NIC calibration anchors themselves (UCCL, DCQCN, HPCC, Kalia et al.
measurements distilled into parameter sets) are recorded in
[docs/papers/msg-size-vs-bandwidth.md](docs/papers/msg-size-vs-bandwidth.md).
The hardware/CC boundary, mlx5 hook, CX-7 evidence rules and full boundary
campaign are in
[docs/papers/rnic-hardware-calibration.md](docs/papers/rnic-hardware-calibration.md).

## Models

What SimLLM models today and what is planned next. Each task ID links to
the module doc that owns it.

### Network and NIC

RNIC hardware and congestion control are separate model axes. A full-RNIC
comparison holds the hardware profile fixed and swaps only the htsim
transport/CC policy. The analytical fluid baseline keeps an explicit hardware
bypass so closed-form validation remains available.
The native WQ and PCIe slices now feed a composed flow-level path for the
frozen isolated `rnic_live_v1` fixture, over a NetworkPort ABI whose v1 flow
form stays the exact default and whose v2 form adds packet-attempt and
transport-control events. Tier B projects the composed native completion
through the execution graph, runtime and `StepResult` into the first TTFT and
TPOT numbers in this repository that the native RNIC chain affects. That claim
stays inside the frozen fixture: one request, single-WQE 4 KiB and 1 MiB
payloads, a two-WQE FIFO, 200 and 400 Gbit/s, and native doorbell service of 0
and 1,000 ps. It says nothing yet about congestion, packet issue timing,
multi-request contention or arbitrary graphs. BACK-8 and the demonstrated
CORE-15 live-seam clauses closed on that evidence. CORE-21 retains the
same-contended-graph bypass-versus-composed comparison and BACK-31 retains the
executable-level unlinked-native negative control. BACK-25 and BACK-26 closed
with the ABI v2 packet-attempt and transport-control vocabulary and an htsim
relay that emits committed TX and RX boundaries, so HTSIM-9 now needs only one
composed run that carries packet-issue evidence through the live metric chain.
The RNIC device itself is now built from modules behind one construction
entry point: the work-queue core plus optional DMA (PCIe), QPC (connection
and context) and network transport modules
([backends](docs/modules/backends.md)). Disabling a module keeps the same
interface, with its parameters inert or explicitly rejected, so one entry
point serves everything from a bare work queue to the full device.

#### RNIC hardware

| Model | Status | What it is |
|---|---|---|
| RDMA Work Queue | partial, first native slice live in the frozen composed fixture ([study](examples/rnic_live_v1/RESULTS.md), [BACK-9](docs/modules/backends.md)) | SimLLM C++ now models one finite SQ/CQ pair, WR-prefix posting, doorbell batches, ordered retirement, signaling, polling, owner wrap, network backpressure and controlled queue failures; RQ/SRQ, shared CQs and mlx5 encoding remain planned |
| PCIe, MMIO and DMA | available, deterministic transaction slice ([study](examples/rnic_pcie_v1/RESULTS.md), [BACK-16/17](docs/modules/backends.md)) | Shared host-store, MWr/MRd/CplD scheduling with finite credits, tags and buffers, class ledgers and analytical path profiles; measured calibration and optional BlueFlame, ATS/ATC and MSI-X remain planned |
| QP, QPC and context memory | host memory available, QP lifecycle planned ([BACK-11](docs/modules/backends.md)) | BACK-19 landed the tracked virtual host memory ([study](examples/rnic_hostmem_v1/RESULTS.md)), so QPC, rings, doorbell records and data regions are explicit allocations and a QPC fetch skips the per-access translation that data buffers take; QP pairing and state transitions plus MTT/MPT and WQE-cache residency in a measured device-cache and host-ICM hierarchy remain planned |
| Host and GPU submission | producer shapes and GPU task coupling available; GPU CQ consumption remains open ([BACK-37](docs/modules/backends.md)) | Who submits work and consumes completions: a host CPU driver ringing doorbells, a CPU proxy fed from GPU queues, or GPU-initiated rings (the GPU posts its own network work) with a GPU-owned completion queue |
| TX/RX hardware pipelines | planned ([BACK-12](docs/modules/backends.md)) | Packetization, schedulers, port buffers, ACK/NAK/RNR/retry, CQE completion, PFC gates and location-specific fault injection |
| CX-7 observable state | planned ([BACK-13/14/15](docs/modules/backends.md)) | Versioned driver-visible registers, counters and traces, verbs capture/replay, and Collie-seeded boundary calibration; undocumented internals stay explicit calibrated abstractions |

#### Congestion-control and transport policies

| Model | Status | What it is |
|---|---|---|
| `rnic-nn` | available through the composed flow adapter at the default ABI v1 flow form; a composed run carrying the landed ABI v2 packet-issue events through the metric chain is still open ([HTSIM-9](docs/modules/backends.md)) | Packetized no-CC policy and the reference for normalized FCT; full-RNIC runs use the same hardware path as physical policies |
| `rnic-nn-fluid` | available | Continuous fluid policy with deterministic closed forms and the explicit hardware-bypass 0 ps anchor |
| `rnic-cn` | available through the composed flow adapter at the default ABI v1 flow form; a composed run carrying the landed ABI v2 packet-issue events through the metric chain is still open ([HTSIM-6/9](docs/modules/backends.md)) | Explicit-rate policy with deterministic reservations, packet spraying and resequencing, lossless without PFC |
| DCQCN | available through the composed flow adapter at the default ABI v1 flow form; a composed run carrying the landed ABI v2 packet-issue events through the metric chain is still open ([HTSIM-5/9](docs/modules/backends.md)) | RoCEv2 comparator with per-QP CNP state, rate reduction/recovery and ECN plus optional PFC; DCQCN calibration lands before programmable CC |
| LogGOPSim flow level | planned ([BACK-2](docs/modules/backends.md)) | Fast flow-level sweeps before packet-level runs |

Fabrics are two-tier Clos topologies with detailed switch models (VoQ
traffic manager, request/grant input-buffered). The default reference
configuration is 8 nodes x 8 B100 GPUs, one 400G NIC per GPU; intra-node
traffic rides an NVLink-class path and stays off the fabric.
Slingshot-style adaptive routing is out of simllm scope.

### GPU compute

| Model | Status | What it is |
|---|---|---|
| Roofline | available | Analytical `max(flops/peak, bytes/bandwidth)` per kernel family, dense and MoE geometry, per-GPU envelopes |
| Profile tables | available | Measured (kernel, config, GPU) duration tables; versioned artifact with mandatory provenance and interpolation |
| Trace-driven GPU service | available, bootstrap | Isolated-kernel CTA/SM/warp scheduling, dependency scoreboards, occupancy and HBM service, plus isolated copy descriptors; [22 post-specified exact-oracle rows](examples/gpu_service_model/RESULTS.md) match to zero cycles, A100/H100 seed timing is not yet silicon-validated |
| Concurrent GPU tasks | available, service primitive | Explicitly submitted compute, memory and NCCL network kernels share SM residency, issue budgets, the HBM cursor and the NVLink egress cursor; the coarse DeviceRuntime (CORE-4) dispatches into this primitive today for the first coordinated bypass profile ([task-mix study](examples/gpu_task_mix/RESULTS.md), [runtime study](examples/core4_runtime/RESULTS.md)) |
| NVLink egress + NCCL ring | available, first cut | One flat per-GPU egress serializer and the per-GPU egress kernel of a ring all-reduce; peer topology, ingress and reduction lanes are planned ([COMP-11](docs/modules/compute.md)) |
| NCCL stack skeleton | available, zero-time component stream | Mirrored NCCL names and causal boundaries as `simllm.compute.nccl`: proxy-op enqueue, GPU send FIFO with head/tail credits, verbs posting, doorbell and CQE poll; the receive leg is absent and the event stream is not yet projected onto the live TTFT/TPOT chain ([study](examples/nccl_stack_v1/RESULTS.md)) |
| SASS offline calibration | planned ([COMP-1/5](docs/modules/compute.md)) | Accel-Sim trace-driven replay populates the tables offline for configurations nobody measured; a cycle simulator never sits inside the step loop |
| Service-time distributions | planned ([COMP-9](docs/modules/compute.md)) | Beyond-mean service times for honest p99+ tails |

### Framework (scheduling and KV cache)

| Framework | Real (runs unmodified) | Simulated | Doc |
|---|---|---|---|
| vLLM, pinned v0.26.0 | v1 scheduler, KV-cache manager, block pool, prefix hashing, preemption | Model execution, sampled tokens, step latency, and the `GroupCoordinator` communicator calls that feed the NCCL stack skeleton | [adapters-vllm](docs/modules/adapters-vllm.md) |
| SGLang, pinned main commit | RadixCache prefix matching, eviction, token/request pool accounting, retraction | Forward results and timing, and the matching communicator calls on the same shared base | [adapters-sglang](docs/modules/adapters-sglang.md) |

Planned on this axis: explicit KV-lifecycle capture
([CORE-3](docs/modules/core.md), [VLLM-11](docs/modules/adapters-vllm.md),
[SGL-9](docs/modules/adapters-sglang.md)), device-schedule capture
([VLLM-12](docs/modules/adapters-vllm.md),
[SGL-10](docs/modules/adapters-sglang.md)), and PD-disaggregation /
KV-transfer traffic (M6). The vLLM model-runner seam, where the SGLang
adapter already sits, is partly there: the flagged skeleton worker and runner
are already live and GPU-invisible in process
([vllm_skeleton_v1](examples/vllm_skeleton_v1/RESULTS.md)), and only the
GPU-present half remains ([VLLM-13](docs/modules/adapters-vllm.md)).

The offline CPU pre-play oracle is no longer planned: capture, arrival join,
vLLM replay and routed-expert supply are live, so a real granite MoE run fixes
each request's output length, stop reason and expert routing, and a replay run
reproduces those completions at the oracle lengths with the captured routing
driving the all-to-all ([preplay](docs/modules/preplay.md)). The one open half
is the independent CPU comparison against a second framework build (PLAY-5),
which is blocked because the installed CUDA vLLM build does not export the CPU
memory operator.

## Modules

Each module has its own doc as the source of truth for design, current
status and numbered open tasks; the README stays a map.

| Module | Purpose | Doc |
|---|---|---|
| `simllm/core` | Virtual clock, scheduler-step records, execution graphs, central bookkeeping, completion contracts | [core](docs/modules/core.md) |
| `simllm/workload` | Arrival processes, length distributions, deterministic generation requests, shared-prefix structure | [workload](docs/modules/workload.md) |
| `simllm/compute` | Pluggable compute-time providers, the GPU service model and its concurrent task primitive, host initiation, and the NCCL stack skeleton | [compute](docs/modules/compute.md) |
| `simllm/placement` | **The mapper**: placement + fabric manifests, rank-to-endpoint/GOAL-rank resolution | [placement](docs/modules/placement.md) |
| `simllm/traffic` | Semantic collectives to physical flows | [traffic](docs/modules/traffic.md) |
| `simllm/goal` | GOAL dependency-graph trace emission | [goal](docs/modules/goal.md) |
| `simllm/preplay` | Offline CPU inference oracle: capture, arrival join, vLLM replay and routed-expert supply, with the independent CPU comparison still open | [preplay](docs/modules/preplay.md) |
| `simllm/backends` | htsim / LogGOPSim invocation + result parsing, submodule pins | [backends](docs/modules/backends.md) |
| `simllm/adapters/vllm` | `SimExecutor` (pluggable, no fork), placement exporter, the simulated communicator and the flagged model-runner skeleton | [adapters-vllm](docs/modules/adapters-vllm.md) |
| `simllm/adapters/sglang` | `SimTpModelWorker`, single-GPU MoE geometry, open-loop generate driver and the simulated communicator | [adapters-sglang](docs/modules/adapters-sglang.md) |

## Development

SimLLM is built in validated stages; every stage ships with studies whose
numbers are defended in the open, pre-registered wherever the expectations
could be frozen before implementation and labeled post-specified where they
could not (see
[docs/README_PRO.md](docs/README_PRO.md) for the process and the stage-by-stage
fidelity plan).

- [x] M0: repo scaffold, backend submodules, CI, per-module docs
- [x] M1: standalone core (workload to GOAL to `htsim_rnic` to metrics)
- [x] M2: vLLM adapter, pinned to v0.26.0, no fork
- [x] M3: SGLang adapter, plugin entry point, no fork
- [ ] M4 (in progress): the closed loop and the execution/resource
      runtime. The composed native RNIC path already drives TTFT and TPOT
      for the frozen isolated `rnic_live_v1` fixture; what remains is the
      online stateful session, KV lifecycle, the same-contended-graph
      comparison and calibration against real captures
- [ ] M5 (in progress): MoE all-to-all studies with captured expert
      routing driving the traffic, plus offline calibration. The pre-play
      oracle's capture, join, replay and routed-supply halves are
      validated; SASS compute calibration and the independent CPU
      comparison half of the oracle remain
- [ ] M6: PD-disaggregation and KV-transfer traffic modeling
- [ ] M7 (in progress): the rest of the RNIC module set. The composition
      entry point, the PCIe/DMA slice, virtual host memory, the three
      submission shapes including GPU-initiated rings and the flagged
      vLLM model-runner skeleton have landed; QP lifecycle, the TX/RX
      pipelines, the GPU-owned completion queue and the model-runner seam
      under a real GPU worker remain

Everything deeper lives in the developer guide
[docs/README_PRO.md](docs/README_PRO.md): the open task registry, the full
roadmap and milestone detail, the execution-fidelity order, and the
development workflow (pre-registered studies, audited results, numbered
deferrals).

## Contributing

We welcome contributions of every size; see [CONTRIBUTING.md](CONTRIBUTING.md).
Good first areas: workload generators, compute-cost calibration profiles
for new GPUs, topology configs, metrics/plotting.

## License

SimLLM is licensed under [Apache-2.0](LICENSE). The backend submodules
keep their own permissive licenses (htsim: BSD 2-Clause,
UEC/UCL/UPB/Broadcom; ATLAHS: MIT, SPCL).

## Design & References

- RFC: [Network backend simulation for vLLM](https://discuss.vllm.ai/t/network-backend-simulation-for-vllm/2812)
- ATLAHS: application-centric network simulator toolchain (SPCL, SC'25)
- htsim: packet-level datacenter network simulator (UCL / Broadcom / UEC)
- LogGOPSim: GOAL-driven LogGOPS simulator (Hoefler et al.)
