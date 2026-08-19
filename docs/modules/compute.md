# simllm.compute

Pluggable compute-time providers plus the host initiation model. The core
needs one number per GOAL `calc` node: how long a rank computes before it
hands data over. Providers answer at different fidelity/cost points. A compact
trace-driven service model replays captured kernels when a provider or
calibration artifact is built. Its online provider path is a cached lookup;
external Accel-Sim replay also remains offline. Neither simulator runs once
per serving step.

## Interface

- `KernelSpec`: fused work plus its stable shape key. A fused transformer step
  also carries the exact `family_kernels` projection used to apportion work;
  ordinary kernels leave it empty.
- `ComputeProvider.estimate(kernel: KernelSpec, gpu: GpuSpec) -> DurationEstimate`
- `ComputeProvider.estimate_layers(kernel, gpu, num_layers)`: optional ordered
  layer estimates for the same fused kernel. The default returns `None` and
  preserves scalar callers exactly. An implemented breakdown must contain one
  nonnegative duration per layer and sum to `estimate()` exactly; consumers
  validate both invariants before using it.
- `ProfileTableProvider`: measured (kernel name, config, GPU) duration
  tables from real captures or offline SASS simulation. Exact-match
  lookups return the entry; a miss interpolates log-linearly along one
  numeric config axis between the nearest bracketing entries of the same
  kernel and GPU, with the uncertainty inflated to
  `max(0.15, neighbors')` (interpolation never claims tighter error than
  its inputs). Queries outside the covered range, or differing on more
  than one axis (COMP-4), raise `KeyError`. Tables round-trip through a
  versioned JSON artifact (schema `simllm-profile-table-v1`) with
  mandatory provenance (`source` e.g. "accel-sim" or "capture",
  simulator/capture `version`, `gpu`, caller-supplied `created` date; the
  library never reads the clock). `enable_family_sum=True` is an explicit
  opt-in that sums a fused kernel's declared family projections and propagates
  conservative uncertainty. The default ignores that projection, retains the
  historical miss behavior and serializes the same table byte for byte.
- `ComputeCalibrationArtifact`: strict
  `simllm-compute-calibration-v1` capture record. It binds GPU, driver, CUDA,
  profiler, source, binary, static-SASS and capture-manifest identities to an
  immutable train or held-out split, launch metadata and every raw duration
  sample. Its compiler emits the existing profile-table schema using train
  medians and held-out error to set family uncertainty.
- `RooflineProvider`: analytical `max(flops/peak, bytes/bw)` with an
  efficiency derate; classifies compute- vs memory-bound from the kernel
  configuration alone. `enable_layer_breakdown=True` apportions the fused
  duration using family work on the selected roof. Repeated transformer
  families divide evenly and the complete LM-head family belongs to the last
  layer. Cumulative integer boundaries guarantee that the nonnegative layer
  durations sum to the scalar estimate exactly. The default is disabled and
  retains the scalar compatibility path byte for byte.
- `TraceCalibratedGpuProvider`: validates and replays its exact trace catalog
  once at construction, then serves O(1) cached estimates behind the existing
  `ComputeProvider` interface. `gpu_model_artifact_to_profile_table` compiles
  validated replays into the smaller immutable online table artifact.
- `ModelDims`: per-rank transformer geometry, dense or MoE. MoE fields
  (`num_experts`, `top_k`, `moe_intermediate_size`, `local_num_experts`)
  default to the dense model; when declared, per-token MLP flops count
  `top_k` experts and weight bytes count only the experts resident on
  this rank under expert parallelism (`local_num_experts`; 0 means all).
- `step_kernel`: one engine step as a single fused kernel (what the
  adapters price today).
- `step_kernels`: the same step split into named kernel families
  (`attn_gemm`, `attn_score`, `mlp_gemm`, `lm_head`, `kv_read`), each
  carrying its shape key (`new_tokens`, `kv_tokens`, `sampled` as
  applicable). Family flops and bytes sum to the fused kernel exactly
  (unit-tested invariant; weights counted once, in the family that
  streams them). This is the COMP-1 groundwork: offline SASS runs
  populate per-family profile tables, and the step loop sums per-family
  estimates instead of pricing one opaque blob.
- `GpuDeviceConfig` and `GpuDevice`: the versioned GPU composition entry point.
  A device is an architecture profile plus typed `GpuPortConfig` ports, each
  carrying protocol, role, direction, declared capabilities, an optional
  declared ceiling and the provenance of the ceiling it ends up with.
  `default_gpu_device_config` derives the port set an architecture's own
  mechanisms already imply. With no declared ceiling, `GpuDevice.architecture`
  is the input object itself, so `sm_scheduler_model()` and
  `copy_engine_service()` reproduce every accepted artifact exactly.
  `GpuPortProtocol` names PCIe and NVLink-C2C on the host link and NVLink,
  PCIe, xGMI and UALink on the peer link. Naming a protocol is not supporting
  it: xGMI and UALink have no first-party measurement and no declared profile
  here, so a port claiming either is rejected during configuration with a
  diagnostic naming COMP-35, which owns vendor instantiation for both.
- `HostInitiationModel`: the exact-zero `ideal` profile, legacy additive
  constants, and two device-bound fixed-step launch-throughput profiles.
  `turing_cuda_graph(N)` and `turing_eager_host(N)` compose provider service
  `C` as `max(C, N * g)`, because host launch demand can overlap device
  service. Each calibrated estimate retains its raw provider duration, launch
  floor, empirical bounds and exposed host contribution. The named Turing
  profiles accept only `GpuSpec.name="gtx1660-ti-sm75"`; a B100 or H100
  selection fails during configuration instead of borrowing the constant.

Every estimate carries an honest uncertainty so results can report error
bounds.

## Kernel-time determinism

This is the model's kernel-time semantics, and it is a contract, not an
implementation detail. It follows the maintainer ruling of 2026-08-18.

**A compute kernel's service time is a deterministic constant with no tail.**
It is a pure function of exactly four inputs:

1. the **kernel family** (`attn_gemm`, `attn_score`, `mlp_gemm`, `lm_head`,
   `kv_read`, or the fused `llm_step` that projects onto them),
2. the **phase**, prefill or decode,
3. the **token and shape inputs**, i.e. `new_tokens`, `kv_tokens`, `sampled`
   and the per-rank `ModelDims` geometry, and
4. the **architecture profile**, i.e. the `GpuSpec` envelope or the
   `GpuArchitectureProfile` the mechanistic replay is calibrated to.

Nothing else may enter. The same four inputs give the same picoseconds on every
rank, in every worker, through either frontend adapter, on every repeat, and in
every process. No provider draws a random number, reads a wall clock or reads
the environment, and no pricing entry point accepts a rank, a worker id or an
adapter identity.

**Rank and runner independence is a statement about the function, not about the
shape.** Two ranks may legitimately carry different shape inputs and therefore
different constants. Uneven expert parallelism is the case already in the
repository: vLLM spreads global experts over the expert-parallel world and gives
the low ranks the remainder, so 30 experts over 8 ranks leaves ranks 0 to 5 with
four resident experts and ranks 6 and 7 with three. Those ranks stream different
weight bytes and their decode steps cost different amounts. That is an input
difference, and the contract is unaffected by it. What the contract forbids is a
provider whose answer depends on who asked.

**Memory-bound kernels are pinned to the HBM bound.** In the roofline provider
a memory-bound estimate is exactly `bytes_moved / (mem_bandwidth * efficiency)`
with no compute term leaking in, and it reports `bound="memory"`. In the
mechanistic `SmSchedulerModel` a kernel whose limiter is the flat per-GPU HBM
cursor takes exactly its cursor occupancy plus the profile's fixed HBM return
latency, and adding SMs changes nothing.

**CUDA-graph launch and eager launch differ only in the host launch cost.** The
COMP-2 profiles already distinguish `turing-cuda-graph` from
`turing-eager-host`, and both compose as `max(C, N * g)` over an unchanged
provider service `C`. The launch class never reaches kernel service time. The
`ideal` host profile contributes exactly zero, so a study with no host profile
selected is reading kernel service time and nothing else.

> Qualification pending. First-party A100 measurement in the
> [graph launch study](../../examples/a100_graph_launch_v1/RESULTS.md) finds a
> device-side per-kernel cost that is 1.415 to 1.506 microseconds larger in
> eager mode than in a graph, of which a null kernel accounts for 1.080. The
> measurement does not determine whether the residual is kernel service time,
> which would qualify this clause, or a device front-end gap outside kernel
> service, which would leave the clause intact. The maintainer's ruling between
> those two readings is pending and COMP-48 owns whatever qualification it
> calls for. The clause above is unchanged and remains in force meanwhile.

**There is no per-kernel tail, and the rationale is that tails are emergent.**
Reported TTFT and TPOT distributions have wide tails in real deployments, and
this model produces them from the network, from batching decisions and from
queueing at contended resources, which is where they physically come from.
Attributing a tail to per-kernel stochasticity would double count: the same
spread would appear once in the kernel constant and again in the queueing that
constant feeds. It would also be unfalsifiable at the metric, because a p99
TTFT can be reproduced by an arbitrary mix of kernel noise and queue noise. So
kernel service time carries a mean-valued constant with an honest uncertainty
for error bounds, and every tail claim is owned by the network, batching and
queueing chain (COMP-9).

**Collective work is the one declared exception**, and it is owned by the
traffic and collective side rather than here. Its destiny is a packetized path
over the GPU's NVLink, xGMI or UALink ports; until then collectives complete
through the deterministic ATLAHS and htsim chain with no-tail constant
completion.

Enforcement is in `tests/test_kernel_determinism.py`, which locks all four
clauses with a mutation control for each, against the fixtures and exact
constants pre-registered by the
[kernel determinism study](../../examples/kernel_determinism_v1/RESULTS.md).

Two limits of that enforcement, so the locks are not read as stronger than they
are. First, the runner-independence evidence is asymmetric: the vLLM executor's
own pricing method is invoked, while the SGLang worker is not importable without
SGLang installed, so its half drives SGLang's own geometry reader into the same
shared call its `_settle` makes rather than invoking `_settle` itself. Second,
the "no random source, wall clock or environment read" check is a static fence
over statically resolvable references: import statements and their aliases,
`from` imports at any relative level, dotted attribute uses resolved through the
alias map, and run-time imports with a constant name, with a computed import
name rejected outright. It cannot see a source reached through a name it cannot
resolve statically, such as a callable passed in as an argument. It is a fence
against introducing one, not a proof that none can exist.

## Fixed per-step host profiles

The fixed-step calibration is scoped to an NVIDIA GeForce GTX 1660 Ti
(`gtx1660-ti-sm75`, compute capability 7.5) on an AMD Ryzen 9 3950X host with
driver 550.90.07 and CUDA 12.4.99. It installs two explicit launch classes:

| Profile | Launch class | Point (ps/launch) | Sample-limited empirical range (ps/launch) |
|---|---|---:|---:|
| `turing-cuda-graph` | `cuda-graph-node` | 809,306 | 624,665 to 809,306 |
| `turing-eager-host` | `eager-host-bound` | 2,364,255 | 2,327,730 to 2,544,074 |

The empirical range is the minimum and maximum of five observations, not a
confidence interval. GPU UUID, host CPU, driver, CUDA version, launch class,
source study and uncertainty kind travel with each profile. The profile point
is a sensitivity constant for this measured Turing device and host only. It
is not a H100 or B100 calibration. Scheduler, sampler and Python-side costs
outside the measured launch classes remain unknown.

The serial step lowerer is the one timing authority. For provider service
`C`, launch count `N` and per-launch point `g`, it computes
`F = max(C, N * g)`. Since GOAL represents whole nanoseconds, calibrated
service is the smallest enclosure `Q = ceil(F / 1,000) * 1,000` ps. The
packet-level sink selects and exposes that same model, while coordinator
dispatch validates that the adapter and sink share it and does not add the
term again. A nonideal profile is rejected on a fallback that has no
host-model-aware timing sink. The default in `SerialStepLowererConfig` and
`HtsimStepSinkConfig` remains `HostInitiationModel.ideal()`, which contributes
exactly zero. Legacy explicit scalar constants retain their historical
additive behavior.

## Trace-driven GPU service boundary

The first SASS service slice models one isolated kernel at a time. Its input
contains stable implementation and trace identities, launch grid and CTA
resource use, plus explicit CTA trace classes. Each class binds a per-warp
instruction stream and dependencies to exact linear block IDs, so edge or
data-dependent CTAs need not be cloned from a representative block. The model
has four replaceable mechanisms:

1. **CTA admission and assignment.** Resident CTAs are limited by the minimum
   of SM block, warp, thread, per-warp register allocation, static and total
   shared-memory capacities. Per-block thread and per-thread register limits
   are checked separately. CTAs are assigned deterministically to SMs as
   capacity becomes available. A launch that cannot admit one CTA fails
   instead of returning a precise-looking duration.
2. **Warp scheduling and SM service.** Ready warps issue through a declared
   number of schedulers and per-cycle issue width. Dependency scoreboards
   preserve RAW and WAW producer ordering. Instruction classes map to
   replaceable latency, initiation interval and execution-port parameters, so
   later calibration can improve tensor, scalar, special-function and memory
   behavior independently. Warp selection is an explicit calibration choice;
   v1 provides deterministic loose round-robin and greedy-then-oldest policies.
   The bootstrap profiles use loose round-robin without claiming NVIDIA's
   undisclosed subpartition policy. The current model handles synchronous
   normalized per-warp instructions only. Barriers, `cp.async`, TMA, warpgroup
   async issue/commit/wait, cooperative launches and thread-block clusters
   fail closed under COMP-10.
3. **HBM service.** Global-memory instructions create explicit byte demand.
   The first slice separates logical lane-request bytes from physical
   transacted bytes, then applies a fixed return latency plus sustained service
   bandwidth to the latter. It reports requested, transacted and serviced
   bytes plus request-instruction count. One flat GPU-wide cursor serializes
   HBM demand across every kernel passed to `estimate_concurrent`, which is the
   first explicit cross-kernel contention mechanism. An input trace may label
   L1, L2 or shared-memory service and receive an explicit fixed latency, but
   v1 does not predict cache hits, partitions or bank conflicts. Those deeper
   mechanisms remain unsupported under COMP-10, not hidden efficiency factors.
   CORE-4 decides which graph operations are released together and arbitrates
   kernel traffic against explicit DMA; the compute model prices the kernel set
   it is given.
4. **Copy service.** A copy descriptor declares direction, endpoints and
   bytes. Isolated service is setup time plus byte serialization in the copy
   engine's own declared clock domain and directional bandwidth. API launch
   delay, engine selection, queue waiting, simultaneous copies, compute/copy
   overlap and shared-HBM arbitration belong to CORE-4. This is external
   device DMA service. In-kernel async copy and TMA are not approximated as
   external DMA.
5. **NVLink egress service.** A store may name the `nvlink` memory space,
   which serializes on one per-GPU egress cursor with its own latency and
   bandwidth, exactly as HBM stores serialize on the HBM cursor. This is
   the intra-node path that keeps NVLink traffic off the fabric backend
   (TRAF-10). It is one flat same-generation egress serializer: peer
   topology, per-link routing, ingress service and reduction lanes are
   absent under COMP-31. A calibration without an `nvlink` profile rejects
   NVLink instructions rather than pricing them as HBM.

### Concurrent task scheduling

`estimate_concurrent` replays several `GpuTask` records on one GPU.
A task is a kernel launch plus a `GpuTaskKind` label (compute, memory or
network) used only for attribution: the replay prices every task by its
instructions and the resources it touches, never by its label. Tasks
share SM residency, per-SM issue budgets, pipelines, the HBM cursor and
the NVLink cursor, and CTAs of a later task backfill capacity an earlier
task cannot use. The result carries the makespan plus per-task admitted
and completion cycles, issued instructions and byte counts.

Each task now also carries logical submission and eligibility cycles. The
concurrent service admits no CTA before eligibility, includes a newly eligible
task in the same deterministic replay as resident kernels, and projects both
input cycles into its per-task estimate. Default zero cycles preserve every
accepted replay. Idle time before the first eligible task advances virtual
time but is not misreported as dependency, pipeline or completion drain.

`simllm.compute.rnic` uses this timed service for optional RNIC submission
production. CPU-proxy mode submits a light GPU descriptor-store and
publication task. GPU-initiated mode submits a WQE-store, doorbell-record
store and publication task. Both use the network task class and contend with
surrounding kernels for SM residency, issue and HBM service. The surrounding
NCCL egress task retains its NVLink cursor and can be delayed through the
shared issue path. Compute completion is resolved against the caller's
submission deadline, then projected into the native RNIC record as an
immutable link. Coupling is disabled by default, and host-CPU mode never
constructs a task or invokes the scheduler.

Replay order is a declared input to this coupling. `RnicProducerCoupling`
passes caller-supplied concurrent tasks first in caller order, followed by
non-host producer tasks in request order. The deterministic baseline scheduler
uses task index to break admission and issue ties. Producer-last order lets the
frozen residency-saturated background claim the full SM before the producer,
which creates the registered +20 and +23 cycle submission delays. Reversing
that order admits the producer first and does not preserve those rows. The
COMP-13 concurrent artifact must therefore serialize and validate the exact
task order rather than reconstruct it from task kind or identity.

NCCL collectives enter through `simllm.compute.nccl`, which builds the
per-GPU egress kernel of a ring all-reduce: `2 * (W - 1) * P / W` bytes
per GPU, chunked across channel CTAs and their warps, each chunk loaded
from HBM and stored to NVLink. This makes a collective a schedulable
kernel like any other, so it contends with compute and memory work
instead of being priced in isolation. Proxy operations, ingress and
multi-ring topologies are COMP-31.

The [task-mix study](../../examples/gpu_task_mix/RESULTS.md) measures
what limits each kind: compute scales with SMs and with the pipeline
initiation interval, memory is pinned to the HBM cursor and gains nothing
from more SMs, and a double-buffered ring egress kernel falls from 6.1
times its own egress bound with one warp per channel to within 2.4 percent
at eight warps. At that point a ring-first run hides a 132-cycle memory task
under its NVLink drain while conserving all HBM and NVLink counters.
The study also ledgers two registration misses that name real shared
resources: concurrent tasks contend for the issue path, and SM residency is
itself contended, so a co-scheduled kernel is free only while the SM has room
for it.

The result reports total cycles and picoseconds together with occupancy,
instruction issue, HBM demand and per-SM counters. Scheduler pressure counts
wall cycles in which an SM exhausts its dispatch budget. Dependency idle and
pipeline idle count whole-SM idle wall cycles; final instruction or memory
completion is reported separately as completion drain. These counters are
model observables, not aliases for Nsight's per-warp stall metrics.
Deterministic replay of the same artifact must be bit-identical. Unknown
opcodes, missing trace identity, impossible residency, unsupported
cooperative or cluster launches, and incompatible copy directions fail
loudly. The model does not infer a SASS stream from the five aggregate
`step_kernels` accounting families; exact per-invocation records remain
COMP-6.

This boundary is deliberately below the online `ComputeProvider` lookup and
above a full device runtime. Provider construction can replay a catalog once,
or an offline run can populate `simllm-profile-table-v1`. `ExecutionGraph` keeps
CUDA streams and dependencies. CORE-4 composes service calls, selects physical
engines, arbitrates resources and determines inter-operation overlap. Neither
package duplicates the other's scheduler.

### Registered mixed-makespan forms

A concurrent makespan is not the maximum of the isolated durations. The
task-mix study measured two reasons, and `decompose_mixed_makespan` names the
terms of a replay that already happened so a study or regression can compare
them. It is a read-only projection of one `GpuConcurrentEstimate` against the
single-task controls of the same architecture, never a second estimator.
`MixedMakespanForm` reports the regime, both physical bounds, the issue delay
and the residency decomposition.

The G1 issue-order form. When every task admits its first CTAs at its own
eligibility cycle, the tasks overlap and the makespan is

```text
T_mixed = max(isolated durations) + delta_issue,
```

where `delta_issue` follows the actual ordered tuple the caller submitted.
For the frozen 8-CTA memory and NVLink egress pair, memory-first measures 329
cycles against a 328-cycle egress control and network-first measures 328. The
delay survives widening the per-SM scheduler budget alone and widening the
load/store issue width alone; only widening both together removes it, so the
binding resource is whichever per-SM issue currency is scarcer. This is not a
label rule: `GpuTaskKind`, priority and a canonical memory-before-network sort
are all irrelevant, and reconstructing the order from any of them would
reproduce the number for the wrong reason.

The G2 residency form. When an SM's shared memory cannot hold both tasks'
CTAs, the second task does not backfill; it waits and then pays its whole
isolated duration:

```text
T_mixed = admitted_cycle(gated task) + isolated duration(gated task).
```

With each CTA claiming half an SM's shared memory the isolated controls are 14
and 229 cycles, the memory task admits at cycle 14 exactly when the compute
task finishes, and the makespan is their 243-cycle sum. Removing the shared
memory demand restores backfill: isolated 7 and 132, makespan 133, i.e. the
maximum plus the same one-cycle G1 term. The admission equality is part of the
form, because a 243-cycle makespan on its own would not identify residency as
the cause.

Submission order is therefore an input CORE-4 owns, not a property the compute
service may infer. `CoarseDeviceRuntime` fixes the membership of a co-runnable
compute group, orders it by repeated arbitration grants, and passes that
ordered tuple to `estimate_concurrent`, so the measured G1 term follows the
order the runtime actually chose. Under the identity policy every grant is the
deterministic baseline sequence, which is `ExecutionGraph` tuple order, and
permuting priority labels changes nothing. A class-aware policy reorders only
legal ready candidates, and
[the arbitrated-order study](../../examples/arbitrated_order_v1/RESULTS.md)
measured the same one-cycle G1 term following that reordered tuple through the
live metric chain.

Both forms are the behavior of the exact frozen fixtures, replicated by
[the mixed-makespan study](../../examples/mixed_makespan_v1/RESULTS.md)
through the component scheduler and through the live CORE-4 metric chain.
Neither extrapolates to other shared-memory fractions, launch shapes,
instruction mixes or GPU architectures, and the synthetic 1 GHz profile is a
mechanism fixture rather than any silicon calibration.

## Seed profiles and calibration ledger

`GpuArchitectureProfile` contains structural limits. Its swappable
`GpuCalibrationProfile` is explicitly bound to one target architecture profile
and contains the target core and optional memory clock, instruction/pipeline
timing, memory timing and bandwidth, warp selection, copy-engine timing,
provenance and uncertainty. The provenance GPU may identify a transferred
evidence source, e.g. H800 timing used as an H100 prior, without changing the
target identity. Recalibration therefore leaves architecture and trace
identity unchanged, and attaching an A100 calibration to an H100 structure
fails at construction.

The A100 SXM 80 GB and H100 SXM 80 GB profiles are bootstrap artifacts. Their
documented occupancy limits and SKU peaks come from NVIDIA's
[Ampere tuning guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html),
[Hopper tuning guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html),
[A100 data sheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/a100-80gb-datasheet-update-nvidia-us-1521051-r2-web.pdf)
and [H100 specifications](https://www.nvidia.com/en-us/data-center/h100/).
Instruction and memory context comes from the open
[Ampere study](https://arxiv.org/abs/2208.11174),
[Hopper/H800 study](https://arxiv.org/abs/2402.13499), and the later
[A100/H800 microbenchmark study](https://arxiv.org/abs/2501.12084). The
numeric memory-latency priors are transferred from the last study. Its Hopper
device is H800 PCIe, not H100 SXM, and its A100 measurements are not treated as
an exact A100 SXM 80 GB match. The profile provenance says so and assigns 50
percent relative uncertainty. A public peak is a capacity constraint, not a
claim that an arbitrary kernel reaches it. Public documentation does not
expose a complete copy-engine timing or selection contract, so the seed
profiles intentionally contain no copy engines until capture supplies them.

The production path uses the
[Accel-Sim paper](https://doi.org/10.1109/ISCA45697.2020.00047) and
[Accel-Sim framework](https://github.com/accel-sim/accel-sim-framework) for
external SASS replay and counter correlation. Every capture/calibration
run must eventually close this production ledger before COMP-1 can close:

| Component | Required evidence |
|---|---|
| Run envelope | framework and commit, model, exact GPU SKU and UUID, driver, CUDA, libraries, dtype/quantization, eager or graph mode, numeric observed core/memory clocks, lock policy and warm-up policy |
| Kernel identity | binary and function hash, semantic operation, launch order, stream, grid/block dimensions, registers, static/dynamic shared memory, cooperative/cluster flags |
| SASS and scheduler | tracer/version, trace hash, warp and CTA identities, instruction classes and dependencies, elapsed cycles, eligible/active warps, issue utilization and stall reasons |
| Memory | requested and transacted bytes, cache hit/miss counters, HBM throughput, latency probes, cache-state protocol and memory-clock state |
| Copy | API kind, direction and endpoints, bytes, stream/event order, reported device engine capabilities, setup samples, sustained bandwidth and concurrent-copy experiment |
| Fit | immutable train/held-out split, raw samples, sample count, fitted parameters, residuals by component, uncertainty and creation date |

The v2 artifact enforces the capture environment, model/GPU identity,
framework/tool/library versions, clock and warm-up policy, numeric observed
core and memory clocks, hashes, semantic attributes, launch resources,
CTA/warp traces, stream order, requested and transacted bytes, copy
direction/endpoints, raw duration samples, deterministic replay, split and
residual. A captured artifact must use the calibration's exact core and target
memory clocks; seed calibrations without a numeric memory-clock target cannot
claim captured measurements. It does not yet encode profiler cache counters,
per-warp eligible/active samples, 3D launch coordinates or concurrent-copy
experiments. Those production ledger fields land with COMP-1 and COMP-10 in a
new schema version before either task closes. Bulk counter exports remain
content-addressed outside Git.

The ledger keeps structural facts separate from fitted timing parameters. A
future capture can replace instruction latencies, throughput corrections,
cache/HBM behavior and copy parameters without changing the trace, service or
provider interfaces.

### Artifact boundary

`simllm-gpu-model-artifact-v2` is the versioned interchange record for this
model. It complements `simllm-profile-table-v1`: the GPU-model artifact keeps
one replay auditable, while the profile table is the compact online lookup
surface produced after calibration. The reader promotes v1 artifacts by
renaming the clarified per-SM completion counter and filling the absent NVLink
profile and counters with `null` and zero; writers always emit v2. A GPU-model
artifact retains:

- the architecture-profile identity, exact SKU, structural limits, fitted
  parameter set, source links and declared uncertainty;
- capture envelope and calibration provenance, including framework,
  toolchain, tracer/simulator versions, observed core/memory clocks and
  creation date;
- SASS trace identity, kernel binary/function identity, semantic catalog key,
  launch shape and resource declaration;
- simulated cycles and picoseconds, replay counters, occupancy, issued
  instructions, per-SM idle/pressure/drain counters, and requested,
  transacted and serviced HBM bytes;
- explicit copy transfers and service replays with direction, endpoints,
  selected engine, independent clock domain and stream order;
- measured duration samples, sample count and summary statistics when silicon
  measurements exist; absent measurements remain explicitly absent rather
  than being synthesized from the model;
- immutable train or held-out split and the fitted residual/uncertainty when
  the artifact participates in calibration.

The strict loader normalizes hash spellings, rejects duplicate semantic keys
and stream orders, recomputes sample summaries, checks capture split isolation,
and reruns every deterministic kernel/copy estimate before accepting it.
Changing an identity, source, fit or split produces a new artifact. Small
synthetic fixtures may live with tests and studies. Raw production SASS traces,
profiler exports and bulk replay outputs live under the external root
configured by `SIMLLM_DATA_ROOT`, never in
Git; the public artifact records their content hashes and provenance.

## COMP-1: offline SASS calibration plan

Strictly offline; the step loop never invokes a cycle-level simulator.

- Use a hybrid measured plus SASS pipeline. Raw cycle-simulator output is
  never treated as silicon truth. Pin a support envelope for every table:
  framework and commit, model, GPU architecture, CUDA/toolchain, dtype and
  quantization, eager or CUDA-graph mode, kernel implementation, tensor
  parallel width, batch/new-token/context shapes, KV dtype and MoE shape.
  Unsupported combinations miss loudly rather than borrowing a precise-
  looking number.
- Capture the exact production run first. Nsight/CUPTI metadata records
  kernel identity, launch order, streams, shapes and silicon durations;
  NVBit supplies the SASS traces required by Accel-Sim. Key table entries
  by kernel binary/hash plus the semantic shape, not by a family label alone,
  so a framework or compiler kernel change invalidates the correct entries.
- Build one replayable microbenchmark per captured kernel implementation.
  It must reproduce launch parameters, tensor layout, dtype, workspace,
  stream/graph mode and relevant cache state. Sweep the captured shape axes,
  not synthetic square GEMMs that the real framework never launches.
- Replay traces offline with a pinned Accel-Sim/GPGPU-Sim configuration on
  hardware that the simulator supports. Fit and report calibration residuals
  against silicon using train shapes, then evaluate held-out shapes. Launch
  overhead, host delay and queueing are measured separately from kernel
  service, so the SASS table cannot hide a missing runtime queue.
- Populate `simllm-gpu-model-artifact-v2` with capture hash, kernel hash, GPU,
  tool versions, shape, measured samples, simulated cycles, calibrated
  duration, uncertainty, calibration split and creation date. Derive the
  compact `simllm-profile-table-v1` lookup entry from that record and retain
  the model artifact's identity in table provenance. Both artifacts are
  immutable; changing an identity field produces a new record.
- Initial acceptance bars, to be tightened from evidence: 100 percent kernel
  identity coverage for the supported run; the stability bar below; held-out
  per-kernel median absolute percentage error below 10 percent and p95 below
  20 percent; per-phase median below 5 percent and p95 below 10 percent;
  compute-only step error below 5 percent. Every miss is reported, never
  averaged away.
- Stability bar, environment-scoped. In a **controlled** environment, defined
  as a non-display device with locked application clocks and exclusive compute
  access, the bar is the original one: measured coefficient of variation below
  2 percent over every sample of a cell. That remains the bar the production
  target-architecture capture must meet. In an environment explicitly declared
  as a **shared display GPU without clock control**, a cell is stable when its
  excursion-trimmed coefficient of variation is below 2 percent, its excursion
  fraction is below 10 percent of the cell's samples, and its maximum excursion
  ratio is below 1.35, where an excursion is a sample above 1.05 times the cell
  median. Every cell additionally reports its all-sample coefficient of
  variation and its full excursion census; no sample is ever discarded from the
  artifact. The
  [fidelity study](../../examples/compute_fidelity_v1/RESULTS.md) froze this
  form before evaluating it, and measured why the second form is the one that
  identifies kernel service-time stability on a display GPU: across the tracked
  Turing capture, 7 samples out of 2,050 exceed the excursion threshold, one in
  each of 7 cells, and the three cells that failed the all-sample bar have
  trimmed coefficients of variation of 0.172, 0.212 and 0.842 percent. A fresh
  4,000-launch probe of the worst of those cells attributes 93.4 percent of its
  excursions to longer block residency at an unchanged 1,869 MHz effective SM
  clock, and the remainder to clock-state drops to 76.9 percent of that clock.
  Neither is kernel service-time variation, and neither is removable without
  the administrator action COMP-5 requires.
- Fixed per-step cost. Kernel service time is not step time. A modeled step is
  exactly the sum of its kernel service: `RooflineProvider` returns 0 ps for a
  zero-work kernel and uses a homogeneous roofline formula above that, with its
  public integer-picosecond result subject to rounding,
  `ProfileTableProvider` returns a measured kernel duration, and
  `HostInitiationModel` is a per-send network initiation delay rather than a
  per-kernel launch cost, so nothing in this package prices kernel launch,
  scheduling or sampling. The fidelity study bounds what that omission is worth
  for a 24-layer top-8 MoE decode step: 440 to 567 device-visible launches in
  eager mode, at a Turing-measured 630 ns per CUDA-graph node, 1,603 ns of
  device-side inter-kernel gap, or 2,332 ns per host-bound eager launch, which
  leaves an omitted excess of 1.79 to 12.31 times the whole modeled decode
  compute of that step. The launch count is a property of the model geometry
  and the framework rather than of the GPU, but the constant itself is Turing
  evidence and does not transfer. At 440 launches, the omitted excess remains
  at least one modeled compute only above 451.7 ns per launch and disappears at
  or below 225.8 ns. Calibrating the production constant on
  the target architecture belongs to this task's "launch overhead, host delay
  and queueing are measured separately from kernel service" clause, and no knob
  is added to the step path until it is measured.
- Simulator starting point: Accel-Sim v1.3.0 in SASS trace-driven mode over
  GPGPU-Sim 4.x with a compatible NVBit tracer. Tool versions remain pinned
  per artifact because modern framework kernels and GPU architectures may
  require newer support than this starting point provides.
- B100: no validated Accel-Sim config exists. The per-family
  efficiency-versus-shape surface measured on the validated GPU is
  transferred onto the B100 roofline roofs (peak flops and HBM bandwidth
  of the declared envelope), replacing the flat 0.7 derate, with
  explicitly inflated uncertainty in every transferred entry. The first
  real B100 capture recalibrates by table swap; nothing in the step loop
  changes.
- Hard dependency (COMP-5): local CUDA 12.4 and CUPTI activity timing work on
  the GTX 1660 Ti with driver 550.90.07. Nsight Compute attaches but returns
  `ERR_NVGPUCTRPERM` because the loaded driver has
  `RmProfilingAdminOnly: 1`; no performance counters are collected. The
  display GPU also produced isolated timing outliers above the original
  all-sample stability ceiling, and the fidelity study identified their two
  mechanisms: blocks resident longer because the desktop shares the SM, and
  discrete drops of the effective SM clock. Both need permissions this project
  does not have, so the controlled-environment form of the stability bar cannot
  be met here at all. Production closure needs counter permission, a non-display
  device with lockable clocks, and allocation on the exact target architecture
  with a compatible dynamic SASS and Accel-Sim path. The
  [A100 environment qualification](../../examples/a100_environment_qualification_v1/RESULTS.md)
  now proves that one Merlin A100 allocation supports CUDA activity, basic
  performance counters, static SASS and exact environment provenance. It does
  not yet prove controlled-cell stability, dynamic tracing, Accel-Sim replay
  or a production kernel.

## GPU device composition and typed ports

The NIC has been a device with typed ports since BACK-18. The GPU now is too:
`GpuDeviceConfig` composes an architecture profile with typed ports over the
two link mechanisms that already exist, and adds nothing to their timing. The
design statement is
[the packet-device model](../design/packet-device-model.md); the validated slice
is [gpu_device_ports_v1](../../examples/gpu_device_ports_v1/RESULTS.md).

A port carries protocol (`pcie`, `nvlink_c2c`, `nvlink`, `xgmi`), role (host
link or peer link), direction (ingress, egress or bidirectional, relative to the
GPU), declared capabilities, and a ceiling with the provenance of that ceiling.
The mechanism behind a capability stays authoritative: `copy_engine_transfer`
names the per-direction `CopyDirectionProfile` entries of one `CopyEngineProfile`
and `peer_store_egress` names the flat `NvlinkProfile` egress cursor.

Four rules make the port layer safe to add under a byte-identical off path.

1. **Reading a ceiling is not declaring one.** A port with no declared ceiling
   reads its ceiling out of the mechanism and reports
   `calibration_derived` provenance. A device whose ports declare no ceiling
   returns the input architecture object itself, so every accepted timestamp,
   counter and byte count is reproduced by object identity rather than by
   equality. A declared ceiling replaces the mechanism parameter for the
   directions that one port carries, and only those; the derived architecture is
   renamed (`<profile>+<port>@<value>bpc`) so no artifact can claim the base
   profile identity while carrying a rescoped parameter.
2. **A disabled port is a declaration that is absent, not a mechanism that is
   off.** Disabling a port never rescopes the copy engine or the egress cursor.
   The port keeps its interface and is still reported with `not_applicable`
   applicability, its own parameters are inert, and every request made of it is
   rejected with a diagnostic naming it. A disabled port carrying a declared
   ceiling is itself a configuration error.
3. **One mechanism has one port authority.** Two enabled ports may not claim the
   same copy direction of the same engine, and two may not claim the one
   per-GPU egress cursor.
4. **Anything without a mechanism fails closed at configuration time.** A
   peer-store port on a calibration with no `nvlink` profile, a copy direction
   the engine does not declare, an unknown engine, a `device_to_device` copy
   (which stays inside one GPU and crosses no port), an xGMI port (COMP-35 owns
   vendor instantiation), a transport-control capability such as ECN marking
   (BACK-48 owns making the ABI v2 packet vocabulary reachable from a non-wire
   port), and a single bidirectional port over two disagreeing mechanism
   ceilings are all rejected during configuration rather than at first use. The
   last of those is why the measured Grace C2C asymmetry, 419.93 GB/s inbound
   against 169.96 GB/s outbound, has to be declared as two ports instead of one
   averaged rate.

The ports declare and negotiate; they do not emit packets. Carrying an extent
and attempt identity in the ABI v2 vocabulary across a non-wire port is BACK-48
with COMP-40 as its compute-side half, and attaching measured per-port ceilings
to a shipped profile is COMP-41.

## Status

The kernel-time determinism contract above is stated publicly and enforced. The
pre-registered
[kernel determinism study](../../examples/kernel_determinism_v1/RESULTS.md) is
nonvoid: all 23 fatal guards held, all 3 controls discriminated, and all 8
frozen scored instances passed with a zero residual, with 5 derived rows and 8
raw observations reported separately and never added in. It fixes the exact
prefill and decode constants of its own fixture, shows the memory-bound pin on
both the roofline and the SM-scheduler paths (including that the pin does not
notice SM count), and shows the vLLM and SGLang readers pricing one step to the
identical picosecond. Its findings are that the contract constrains the pricing
function and not the per-rank shape assignment (an uneven expert split is an
input difference, not a violation), that COMP-9's original per-kernel
distribution scope is refuted rather than unfinished, and that the two adapter
readers store two optional dtype fields differently while resolving them
identically, which is COMP-42. The study makes no silicon claim, prices no
collective, and validates no tail: locating the tail is COMP-9, which is open.
A static import and attribute-reference audit found no random source, wall clock
or environment read reachable by a statically resolvable name anywhere in
`simllm/compute`, so the guard that forbids them is a fence rather than a fix.
Review widened that audit after showing its first form could be stepped around
by a bare `numpy` import with a `numpy.random` use, by
`importlib.import_module` or `__import__`, or by a relative import; its residual
blind spot, a source reached through a name that cannot be resolved statically,
is stated with the contract above rather than left implied.

Both providers, the transformer step model (fused and family-decomposed),
the host model, and the trace-driven GPU service are implemented and
tested. The service covers isolated-kernel replay, copy descriptors, the
NVLink egress cursor, concurrent multi-task scheduling
(`estimate_concurrent`) and the NCCL ring-collective builder. The
[service-model study](../../examples/gpu_service_model/RESULTS.md) validates
22 post-specified exact-oracle rows to zero-cycle residual, and the
[task-mix study](../../examples/gpu_task_mix/RESULTS.md) reports 36 passing
exact-oracle rows and 6 passing behavioral relation families over 17
instances. Its 21 structural invariants are unscored, and its two superseded
registration misses remain visible as the chronology behind findings G1 and
G2. Those two findings now have registered forms: the
[mixed-makespan study](../../examples/mixed_makespan_v1/RESULTS.md) replicates
them through the component scheduler and through the live CORE-4 metric chain,
passing 11 genuine-risk instances across four families with all 124 fatal
guards holding. Its residuals are COMP-24 (the forms cover one fixture and one
residency-gated task), COMP-25 (no production step path selects the concurrent
kernel service) and CORE-49, which closed with
[the arbitrated-order study](../../examples/arbitrated_order_v1/RESULTS.md):
the co-runnable group is now ordered by repeated arbitration grants rather than
by graph order. The built-in
A100/H100 profiles are unvalidated bootstrap seeds and do not establish
production accuracy: their pipeline initiation intervals are derived from
published per-SM unit counts, not measured.

The GPU device composition entry point with typed PCIe and NVLink ports is
landed and closes COMP-34. The
[device-port study](../../examples/gpu_device_ports_v1/RESULTS.md) passes 11 of
11 scored instances across four families with all 54 fatal guards holding: a
declared host-link ceiling moves the job completion time of a `DmaWork`
descriptor through the live CORE-4 chain by the exact registered amount, a
declared peer-link ceiling moves the NVLink egress term of the accepted task-mix
cells onto values that study already published, the override never leaves the
direction its port carries, and every accepted `gpu_task_mix`,
`gpu_service_model` and `mixed_makespan_v1` artifact reproduces byte for byte
through the composed device with default ports, locked by
`tests/test_gpu_device_ports.py` with a mutation control per artifact. Four
further identity-path cells are retained as an unscored baseline register, which
is the correction the study's own correction section records against its first
publication of 15 of 15. Its
residuals are COMP-40 (the ports declare capabilities but emit no packet event)
and COMP-41 (no shipped profile carries a measured per-port ceiling). Finding F1
of that study is a constraint on later registrations: halving the egress ceiling
of the accepted ring cell added the full serialization delta with nothing hidden
by overlap, because at eight warps per channel the kernel is already within 101
cycles of its own egress bound. Finding F3 is a constraint on how a freeze is
written: entailment has to be checked per parameterized instance, because a
relation can be unlosable in some of its cells and genuinely at risk in others.

The
[A100 environment qualification](../../examples/a100_environment_qualification_v1/RESULTS.md)
is `QUALIFIED` at SimLLM commit
`3c829c660ec6d48a627447632ee99bd40f001784`. One nonexclusive Merlin
allocation exposed exactly one A100 SXM4 80 GB, stable disabled MIG state, no
foreign process, a nonempty Nsight Systems CUDA trace, numeric Nsight Compute
basic counters and static `sm_80` SASS. This establishes the capability gate
for an A100 production study. It populates no profile table, transfers no
duration to H100 or B100, and leaves dynamic NVBit capture, Accel-Sim
compatibility and registered-cell stability unproven.

The [Turing calibration study](../../examples/compute_calibration_v1/RESULTS.md)
lands the first real activity-timing pipeline and populated table. On the
available GTX 1660 Ti it captured 50 family, dtype and shape cells with 2,050
target samples. Held-out calibrated median and p95 error were 0.674 percent
and 1.773 percent versus 17.782 percent and 25.069 percent for the roofline
bootstrap. The frozen study is nevertheless an overall failure: isolated
high-duration samples put 3 of 50 final cells above the 2 percent coefficient
of variation ceiling, and the preceding post-fix capture missed 2 of 50.
These Turing numbers validate the method and do not transfer to Hopper.

The [fidelity study](../../examples/compute_fidelity_v1/RESULTS.md) is void with
findings because frozen fatal guard XFER-G4's exact proportionality predicate
failed by a 1 ps integer-quantization residual. Its behavioral pass fraction is
therefore uninterpretable. The measurement layer still changes what is known
about the earlier stability failure and the modeled step. Re-reading the same
immutable capture shows the ceiling was failed by 7 samples out of 2,050, one
in each of 7 cells, while the worst excursion-trimmed coefficient of variation
anywhere in the capture is 1.054 percent. A 4,000-launch device probe that
records each block's own cycle span and residency alongside its wall duration
attributes 93.4 percent of a fresh excursion population to longer block
residency at an unchanged 1,869 MHz effective SM clock and the remainder to
clock-state drops to 76.9 percent of that clock, so the tail is the display GPU
rather than the kernel. The stability bar above is refrozen accordingly, with
the original all-sample form retained unchanged for the controlled environment
the production capture must use. The same study measures a fixed per-step cost
whose omitted excess is 1.79 to 12.31 times the whole modeled decode compute of
a 24-layer top-8 MoE step. It registers no new task ID: COMP-1 and COMP-5 both
stay open and keep every clause they registered.

The two A100 calibration studies of the same campaign are void beside it, and
neither closes anything. The
[A100 kernel constants study](../../examples/a100_kernel_constants_v1/RESULTS.md)
violated three stability guards across two runs and deliberately withholds the
`simllm-profile-table-v1` artifact it was built to produce, because a table
from a void run is one a provider would load without noticing. Its retained
evidence is a measured 1818.21 GB/s HBM roof, per-family roofline efficiency
spanning 0.125 to 0.951 where the surrogate is a flat 0.7, captured MoE expert
cells at 5.17 to 12.20 times their own memory roof, a bimodal 1275 and 1410 MHz
SM clock that moves compute constants by the clock ratio and leaves
memory-limited ones still, and a 2.34 microsecond device cost for one CUDA
event placed between two launches. The
[A100 graph launch study](../../examples/a100_graph_launch_v1/RESULTS.md)
violated one dispersion guard and installs neither of the two
`HostInitiationModel` profiles it measured. Its retained evidence separates
host submission, 1,629,633 ps per eager launch against a flat 1.6 microseconds
per graph replay at any chain length, from a device-side per-kernel cost that
is 1.415 to 1.506 microseconds larger in eager mode than in a graph. That last
number is why the CUDA-graph clause of the determinism contract above now
carries a pending-ruling qualification note and why COMP-48 exists. Neither
study registers a closure; between them they register COMP-43, COMP-44,
COMP-45, COMP-46, COMP-47 and COMP-48.

The [fixed host-step study](../../examples/host_step_cost_v1/RESULTS.md)
re-established that measurement under a corrected freeze before installing
anything. Corrected calibration attempt three was nonvoid and accepted: all
3 genuinely risky relations plus 1 post-specified replication passed (CAL-1,
whose band was widened after the attempt-two miss at 809,068 ps), and all six
fatal guards held. It measured 809,306 ps per CUDA-graph node and 2,364,255 ps
per host-bound eager launch on the declared Turing device, with the empirical
ranges and provenance recorded above. The live `a-ep8-200g` holdout is a
nonvoid end-to-end conformance and reach demonstration with a genuine-risk
denominator of zero and 12 retained entailed rows. Across graph versus eager
launch and 440 versus 567 launches, decode multipliers were 2.2011, 2.6813,
5.3978 and 6.8006; TPOT multipliers were 2.2019, 2.6825, 5.4008 and 6.8047.
Those values show that the installed cost reaches TTFT and TPOT, not that its
magnitude was independently predicted.

The ideal compatibility guard is separate, fatal and unscored. A fresh
five-cell `end_to_end_replay_v1` replay was nonvoid, retained all 13 of that
study's exact-oracle relations, and reproduced its aggregate canonical digest
plus every `steps.jsonl` byte stream. The first calibrated live attempt was
void because repeated per-layer integer floors underrepresented
`max(C, N * g)` by 6,640 to 20,502 ps. The corrected second attempt verifies
the exact whole-nanosecond enclosure, but its magnitude rows are unscored
because fatal exact-row oracles entail them. The held-out third attempt, not
that regression, supplies live conformance and reach evidence but no magnitude
score.

For the mission error budget, item 1 moves from zero to a measured launch
floor only in the device-bound Turing sensitivity. Correlating that launch
term in the simulated and plausible-real expressions leaves a point residual
optimism range of 1.424953 to 3.891039 times; propagating the sample-limited
empirical endpoints gives 1.396964 to 4.508550 times. These ranges assume all
unmeasured scheduler, sampler and Python costs are zero and sit beside, rather
than replace, the mission's generic 5 to 22 times budget. The reference B100
host cost is unknown, so no absolute B100 composed optimism range is supported.
The fixed 99,024,000 ps input is B100-derived. Its 554,631,168 bytes need
1,925,802,667 ps on the Turing device's 288 GB/s roof and 2,751,146,667 ps at
the 0.7 derate, above all four launch floors, so the hybrid rows are not a
device-consistent Turing step prediction. The reported rows use
`network + max(C, N * g)`.

The [composed step budget study](../../examples/composed_step_budget_v1/RESULTS.md)
settled the composition by measurement instead of arithmetic. Running the
mission chain with the host profile and the TRAF-11 collective floor both
enabled shows that the merged code computes
`max(C, N * g) + collective floor + raw fabric`: the launch demand overlaps
provider compute and nothing else. The alternative `max(C + network, N * g)`
reading, which would have given 1.650672 ms for every profile, appears in none
of the study's 93 decode-step observations. Attempt one of that study is void
because one of its own fatal predicates compared a raw provider value against a
quantized literal; attempt two held all ten guards, passed 3 of 3 scored
families, and reproduced attempt one's raw values exactly. A case A decode step
at 400 Gbit/s measures 1.916754 ms at CUDA graph with 440 launches and
2.901192 ms at eager host with 567, against 0.204527 ms with both features
disabled, which the same run reproduced byte for byte. Composition is exact:
over 31 matched decode compositions the two host profiles separate by exactly
984,438,000 ps, the difference of their quantized launch demands, in every
pair. The composition is consistent with the launch count's own registered
exclusion of collective launches, so it is not a defect and no task ID was
registered for it. What the study makes plain is that the modeled compute
contributes zero exposed picoseconds once a calibrated host profile is
selected, because the launch floor masks every provider estimate below it, and
that 94.03 to 96.05 percent of the composed step is transferred constants.

The M5 first slice landed the COMP-1 groundwork: `step_kernels`, the
`simllm-profile-table-v1` artifact with provenance, and 1D log-linear
interpolation (closing COMP-3; the multi-axis extension is COMP-4). The
production SASS pipeline itself (above) has not run yet. Nsight Systems
activity timing works locally, while Nsight Compute counters fail with
`ERR_NVGPUCTRPERM`, the display GPU misses the frozen stability guard and
TU116 cannot supply target-architecture evidence. COMP-5 records those exact
hardware requirements. Therefore COMP-1, COMP-5 and COMP-6 remain open. MoE
geometry
landed with the same slice and is exercised by the examples/m5 studies
together with the MoE traffic mapping
([traffic](traffic.md), [M5 results](../../examples/m5/RESULTS.md)).

COMP-16 is complete. The roofline provider now supplies an explicit opt-in
layer breakdown from the fused step's exact family projection. The
[latent-knob study](../../examples/step_sink_latent_knobs/RESULTS.md) sweeps
two layer counts and two TP widths on the live fluid step sink: every enabled
row moves first-token latency later by the frozen 1,000 ps with zero residual,
while the default path retains the historical GOAL SHA-256 exactly. Profile
table and trace-calibrated layer estimates still require COMP-6 and remain
open as COMP-17.

The COMP-15 first slice is implemented in `simllm.compute.nccl_stack`. Its
function identities were audited against NVIDIA NCCL release `v2.30.7-1`,
commit `73cf112295c33aee2b895f329f592f2a9b4b0f97`. It adds name-mirrored
`ncclCommInitRank` and `ncclAllReduce` entry points and a planner with the same
explicit `2 * (world_size - 1)` ring-step decomposition and strict lane
divisibility as `simllm.compute.nccl`. The send connector follows NCCL's
head/tail convention: the GPU publishes ready state and advances `tail`, while
the CPU proxy advances `head` only after a separately produced network
completion is observed. `ncclProxySaveOp` queues operations before kernel
launch, independent proxy progression permits FIFO occupancy above one, and a
doorbell separates verbs posting from the fake external completion source.

The intra-node route stays inside `ncclKernelMain`, `runRing` and `genericOp`.
The inter-node route traverses the GPU send FIFO, CPU proxy, `ncclNet.isend`,
verbs post, doorbell, external CQE, `ncclNet.test`, CQ poll and head-credit
return. The receive leg is explicitly absent from this slice. Every call,
proactive signal store, and successful poll observation emits a strict
`simllm-nccl-stack-event-v1` record from one caller-supplied `VirtualClock`.
The [NCCL stack skeleton study](../../examples/nccl_stack_v1/RESULTS.md)
reports 5 of 5 passing behavioral relation families over all 35 instances and
10 of 10 fatal unscored structural invariants. This zero-time component stream
is not yet projected onto the live TTFT/TPOT metric chain.

One boundary of that skeleton now carries an opt-in gate. `ncclNetRegMr`
mirrors the net plugin's `regMr` together with the channel FIFO establishment
that follows it, and a communicator built with `require_buffer_registration`
refuses a collective whose destination buffer is not registered on every
channel. The seam declares the registration's one-time cost and, as everywhere
else in this module, never advances the caller's clock. The cost model, the
identity and re-registration rules, and the ledger that spends that cost on the
live metric chain are traffic-owned and are stated in
[the interim collective completion and registration contract](traffic.md#collective-completion-and-registration-the-interim-contract).
Only two claims there rest on the ABI, that a registration entry point exists
and that one seam serves NCCL and RCCL; the one-time charging rule, the
per-buffer identity scope, the channel factor and the three re-registration
events are declared model choices. This gate keeps its own registered-buffer
state, which carries no generation and which the live chain never consults, so
the seam and the traffic-owned ledger are two states that agree by convention
until TRAF-58 unifies them. A communicator that does not ask for the gate emits
exactly the events it emitted before the gate existed. BACK-47 still owns the
device-facing packet emission contract at this same seam.

The same [collective latency floor study](../../examples/collective_latency_floor_v1/RESULTS.md)
closes COMP-11, with its undemonstrated mechanism clauses moved exactly to
COMP-31. The selectable profile replaces the flat local endpoint rate, adds
one participant-indexed base latency at the semantic collective boundary and
reports that base separately from raw fabric transport and the 2.000
microsecond propagation reference. The one-charge and exact identity guards
show that local and fabric projections do not advance or price the same
collective twice, including in a two-node mixed-placement collective with
simultaneous positive local and fabric service. The study does not demonstrate
peer topology, per-link routing, receiving-HBM interaction, reduction lanes or
proxy operations.

### NCCL stack name audit

SimLLM mirrors names and causal boundaries only. It copies no NCCL source.
Every event function is either an audited NCCL symbol or has a `simllm` prefix
and an explicit reason:

| Mirrored event name | NCCL source and symbol, or SimLLM reason |
|---|---|
| `ncclCommInitRank` | `src/init.cc`, `ncclCommInitRank` |
| `ncclBuildRings` | `src/graph/rings.cc`, `ncclBuildRings` |
| `initChannel` | `src/channel.cc`, `initChannel` |
| `ncclAllReduce` | `src/collectives.cc`, `ncclAllReduce` |
| `ncclEnqueueCheck` | `src/enqueue.cc`, `ncclEnqueueCheck` |
| `scheduleCollTasksToPlan` | `src/enqueue.cc`, `scheduleCollTasksToPlan` |
| `calcCollChunking` | `src/enqueue.cc`, `calcCollChunking` |
| `ncclProxySaveOp` | `src/proxy.cc`, `ncclProxySaveOp`; upload call in `src/enqueue.cc` |
| `ncclLaunchKernel` | `src/enqueue.cc`, `ncclLaunchKernel` |
| `ncclKernelMain` | `src/device/common.h`, `ncclKernelMain` |
| `runRing` | `src/device/all_reduce.h`, `runRing` |
| `waitPeer` | `src/device/prims_simple.h`, `waitPeer` |
| `genericOp` | `src/device/prims_simple.h`, `genericOp` |
| `postPeer` | `src/device/prims_simple.h`, `postPeer` |
| `ncclProxyProgress` | `src/proxy.cc`, `ncclProxyProgress` |
| `sendProxyProgress` | `src/transport/net.cc`, `sendProxyProgress` |
| `ncclNet.isend` | `src/include/plugin/net/net_v12.h`, `isend` member; `ncclIbIsend` in `src/transport/net_ib/p2p.cc` is the audited IB implementation |
| `ncclNet.test` | `src/include/plugin/net/net_v12.h`, `test` member; called by `sendProxyProgress` in `src/transport/net.cc` |
| `ncclNet.regMr` | `src/include/plugin/net/net_v12.h`, `regMr` member, in the same audited NCCL `v2.30.7-1` release as the `isend` and `test` rows above; the entry NCCL calls so an RDMA NIC can prepare a buffer. The published `ncclNet_v6` form of the same member, and its RCCL equivalent, are quoted in [the AMD GPU fabric note](../papers/amd-gpu-fabric.md) |
| `simllmChannelBufferRegistered` | simllm-invented: the one-time (communicator, channel, buffer) registration boundary, where the mirrored seam declares a cost that the traffic-owned ledger spends |
| `wrap_ibv_post_send` | `src/include/ibvwrap.h`, `wrap_ibv_post_send`; called by `ncclIbIsend` in `src/transport/net_ib/p2p.cc` |
| `wrap_ibv_poll_cq` | `src/include/ibvwrap.h`, `wrap_ibv_poll_cq`; called by `ncclIbTest` in `src/transport/net_ib/p2p.cc` |
| `simllmRnicRingDoorbell` | simllm-invented: exposes the RNIC notification hidden inside the verbs provider's post operation |
| `simllmNetworkComplete` | simllm-invented: deterministic external completion injection until a native RNIC session supplies CQEs |
| `simllmKernelComplete` | simllm-invented: stack-internal kernel-completion observation until runtime projection lands |

## Open tasks

### Precision

- COMP-1 (Precision; P1; L): complete production compute calibration.
  The Turing method anchor lands activity capture, immutable raw samples,
  train-only table compilation, interpolation and the provider seam, but its
  numbers are synthetic TU116 evidence. Its final run passed held-out
  calibrated median and p95 error at 0.674 percent and 1.773 percent versus
  17.782 percent and 25.069 percent for the flat 0.7 roofline surrogate.
  Stability is no longer the reason this task is open: the fidelity study
  showed the 3-of-50 miss came from 7 samples in 2,050 against a worst trimmed
  coefficient of variation of 1.054 percent, and refroze the bar in the
  environment-scoped form above. Two things now block it. First, no
  target-architecture evidence exists: replace the active A100/H100 bootstrap
  and the flat 0.7 roofline surrogate only after capturing exact production
  framework kernels on the target architecture, collecting the full activity,
  counter and dynamic-SASS ledger, calibrating pinned Accel-Sim replay, and
  validating immutable held-out kernels. Second, the fixed-step seam now has
  calibrated Turing CUDA-graph and eager-host profiles, but no H100 or B100
  constant. The fidelity study's omitted excess of 1.79 to 12.31 times the
  modeled decode compute therefore remains unbounded on the production target,
  so the compute-only step error clause is unreachable until launch overhead,
  host delay and queueing are measured on that exact architecture. Do not
  transfer the Turing launch constants in the meantime. Acceptance remains
  the environment-scoped stability bar with the controlled form required for the
  production capture, held-out kernel median error below 10 percent and p95
  below 20 percent, per-phase median below 5 percent and p95 below 10 percent,
  and compute-only step error below 5 percent. The roofline and calibration-off
  paths must retain accepted artifacts and timestamps byte for byte. The
  [A100 hardware envelope](../../examples/a100_hardware_envelope_v1/RESULTS.md)
  narrows the second blocker without removing it. On one A100-SXM4-80GB with
  clocks observed at 1410 MHz through every timed block, the launch constants
  are 1.806 us for a pipelined eager launch, 6.069 us for a synchronized
  roundtrip and 0.791 us for a CUDA-graph replay node, so the graph path costs
  0.44 of the eager path on the target architecture rather than on Turing.
  The same lane bounds the flat 0.7 roofline surrogate directly: BF16 GEMM
  reaches 302.22 TFLOP/s at 16384 cubed, which is 96.9 percent of the 311.87
  TFLOP/s clock-derived peak, while HBM read reaches 86.8 percent of the
  2,039.04 GB/s memory-clock-derived peak, and the memory-to-compute crossover
  for `N` = `K` = 8192 measures at `M` = 256 against an ideal 158.9. A single
  0.7 efficiency constant cannot span 0.47 percent of peak at `M` = 1 and 96.9
  percent at 16384 cubed. The
  [GH200 hardware envelope](../../examples/gh200_hardware_envelope_v1/RESULTS.md)
  adds the Hopper constants from the identical sweep: 1.304 us pipelined, 6.126
  us synchronized roundtrip and 0.589 us per CUDA-graph replay node, with the
  roofline crossover at `M` = 512 against an ideal 284.6 and 918.66 TFLOP/s at
  16384 cubed. Two architecture pairs now bound the transfer question the task
  asks. The host-issue constants move with the host, not the GPU: the aarch64
  Grace launches 28 percent faster than the x86 EPYC while the synchronized
  roundtrip is unchanged within one percent, so a launch constant may not be
  carried across hosts even at fixed GPU generation. Both are microbenchmark
  evidence only: no production framework kernel, no dynamic SASS, no Accel-Sim
  calibration and no held-out kernel matrix, so COMP-1 stays open on its first
  blocker.
  The [A100 kernel constants study](../../examples/a100_kernel_constants_v1/RESULTS.md)
  is reviewed `VOID` and therefore closes nothing, but its retained evidence
  narrows the surrogate question further. Its measured HBM roof is 1818.21 GB/s,
  89.17 percent of nameplate, and it publishes clock-conditioned constants
  because application clock control is denied on that allocation. Three of its
  findings bear directly on this task. The flat 0.7 roofline derate is wrong in
  opposite directions for different shapes: measured roofline efficiency spans
  0.315 to 0.763 on the granite QKV family and 0.820 to 0.951 on an
  8192-squared synthetic family, so no single constant covers both. Captured
  MoE expert GEMMs at the granite population's expert loads run 5.17 to 12.20
  times their own memory roof over all 18 captured cells, because at those loads
  the kernel is bound by a fixed per-kernel cost rather than by bandwidth or
  arithmetic; COMP-43 owns that term and COMP-7's entry carries the same trap
  for the per-rank load work. And the operand layout, not the shape, produced a factor 2.9 swing
  between neighbouring token counts in the study's first run, which is why any
  future table must record the layout its constants were measured under.
  The [A100 graph launch study](../../examples/a100_graph_launch_v1/RESULTS.md)
  is also reviewed `VOID` and installs nothing. It retains evidence on part of
  this task's second blocker, the launch and host-delay terms, on the target
  architecture and host; the queueing term that blocker also names is not
  measured by it, so the blocker is narrowed and not removed. As retained
  evidence, the eager per-launch host cost on this A100 and EPYC pair is
  1,629,633 ps, 31.07 percent below the Turing `eager-host-bound` point, which
  alongside the already recorded Grace against EPYC difference is evidence that
  a launch constant tracks the host and driver rather than the GPU generation.
  Two host pairs are not a proof of that rule, and the rule stays a hypothesis
  the next host measurement can refute. CUDA
  graph replay costs the host 1.6 microseconds regardless of chain length, so
  at 256 nodes the host pays 6.5 nanoseconds per enqueued kernel, a factor 251
  below eager. And the study refutes the contract's CUDA-graph clause at the
  device level: a real kernel's period is 1.42 to 1.51 microseconds larger in
  eager mode than in a graph, of which a null kernel accounts for 1.08, so the
  residual 0.34 to 0.43 microseconds is launch-mode-conditioned per-kernel cost
  whose split between service time and front-end gap this driver will not
  report.
- COMP-5 (Precision; P1; L): provide the production capture
  environment required by COMP-1. The local GTX 1660 Ti still cannot qualify:
  Nsight Compute returns `ERR_NVGPUCTRPERM`, and display sharing produces the
  residency and clock-state excursions measured by the fidelity study. The
  [A100 environment qualification](../../examples/a100_environment_qualification_v1/RESULTS.md)
  removes the corresponding basic-capability uncertainty for one Merlin A100.
  Job `195283` produced a nonempty activity trace, numeric basic counters,
  exact tool and GPU provenance, matching disabled MIG and allowed-clock policy
  immediately before and after profiling, no foreign process and static
  `sm_80` SASS. All three probe executions agreed on device identity and
  checksum. This evidence is A100-scoped and must reject H100 or B100 use.
  The task remains open because the qualification intentionally omitted
  production SGLang kernels, dynamic NVBit tracing, Accel-Sim compatibility,
  controlled-clock evidence and the registered-cell stability sweep. The next
  expectations-only production study must exercise those mechanisms, retain
  the exact calibration-off path, and keep every registered cell below the
  controlled-environment stability ceiling before COMP-1 may consume an A100
  profile. Nsight Systems warned that device-side CUDA-event completion tracing
  can add overhead or false cross-stream dependencies. The production freeze
  must explicitly set `--cuda-event-trace=false`, or defend a frozen
  alternative, before interpreting multi-stream dependency evidence. The
  [A100 hardware envelope](../../examples/a100_hardware_envelope_v1/RESULTS.md)
  adds one measured constraint the production freeze must respect: every
  event-bracketed kernel in that study carried about 6 microseconds of fixed
  cost, matching its own 6.069 microsecond launch roundtrip, which silently
  destroyed the L2-residency signature it had predicted at an 8 MiB working
  set. Any registered cell whose kernel is shorter than roughly 60 microseconds
  measures the launch path as much as the kernel, so the production capture
  must amortize inside the timed region or declare a minimum kernel duration.
  Clocks were not locked there either, so the controlled-clock requirement is
  untouched. The
  [A100 kernel constants study](../../examples/a100_kernel_constants_v1/RESULTS.md)
  establishes why: on that allocation `nvidia-smi --lock-gpu-clocks` and
  `nvidia-smi -ac` are both refused with "The current user does not have
  permission to change clocks". The refusal was observed on three allocations
  of this account on `a100-hourly`, on nodes `gpu101` and `gpu105`, so the
  controlled-environment form of the stability bar cannot be met on the
  allocations this project has obtained, without an administrator action it
  does not have. Whether another account or another partition would be refused
  is not established by that evidence. The study substituted a
  clock-conditioned form, publishing constants per SM clock state over
  clock-stationary batches, and that substitute itself failed on 16 of 97
  scored cells, which is evidence about the environment rather than about the
  kernels. It also measured two facts a production capture must respect: the
  SM clock under load is bimodal at 1275 and 1410 MHz with a 283 to 432
  millisecond transition, so a cell that spans the boost boundary mixes two
  constants; and one `cudaEventRecord` placed between two consecutive launches
  costs 2.34 microseconds of device time, so per-kernel event instrumentation
  is not a free observation of a short kernel.
- COMP-7 (Precision; P1; M): MoE compute assumes perfectly balanced routing:
  every rank computes `top_k` experts' flops for its own tokens and streams all
  resident experts once. Consume the landed `simllm-routed-experts-v1`
  projection through `RoutedMoeSupply`, using the same selected placement
  epoch as traffic, to drive per-rank effective expert load and hot-expert
  imbalance. Pricing trap, from first-party A100 measurement: at the captured
  granite expert loads the roofline is not the binding term. The
  [A100 kernel constants study](../../examples/a100_kernel_constants_v1/RESULTS.md)
  measured all 18 captured expert cells at 5.17 to 12.20 times their own memory
  roof, because a load of 1 to 54 rows sits far below the 218 and 277 row
  roofline knees of those two shapes and the kernel is bound by a fixed
  per-kernel cost instead. That evidence is from a void run and closes nothing,
  but any work here that makes per-rank expert load more precise is refining an
  input to a term whose magnitude is wrong by 5 to 12 times, so COMP-43 should
  land alongside it rather than after it.
- COMP-9 (Precision; P1; L): locate and validate latency-tail fidelity in the
  network, batching and queueing chain, which is where the standing kernel-time
  determinism decision (maintainer, 2026-08-18) puts every tail. This task
  previously promised a measured or fitted service-time distribution on
  `DurationEstimate` and the profile artifacts so CORE-5 could claim
  kernel-level p99 and p99.9 accuracy. That scope is refuted for compute. A
  kernel's service time is a deterministic constant with no tail, so a
  per-kernel distribution would double count spread that the queueing it feeds
  already produces, and a reported p99 TTFT could then be reproduced by an
  arbitrary mix of kernel noise and queue noise, which makes the attribution
  unfalsifiable at the metric. `DurationEstimate` keeps one nominal value plus
  an honest uncertainty, and that uncertainty stays an error bound on a
  constant, never a sampling distribution.
  The surrogate now being replaced is the repository's silence about where a
  reported tail comes from: p50 through p99.9 TTFT and TPOT are named as
  milestone deliverables in `docs/architecture.md`, `adapters-vllm.md` and
  `adapters-sglang.md` with no statement of which mechanism owns each
  percentile. The identifying observables are per-visit queue waits under the
  one queue-visit contract (`submitted_at`, `eligible_at`, `started_at`,
  `finished_at`, `completed_at`), per-flow FCT from the packet-level backend,
  and batch composition per step. Acceptance: each reported TTFT and TPOT
  percentile is attributed to network, batching or queueing terms selected on
  the realized critical path with no additive mixing of wait reductions; a
  held-out workload's tail is predicted within a declared band; and removing all
  fabric contention and all batching collapses the distribution onto the
  deterministic constant this module guarantees. The deterministic compute path
  stays byte-identical throughout.
- COMP-31 (Precision; P1; L): complete the mechanism detail retained from
  COMP-11 after the calibrated endpoint serializer and semantic collective
  floor landed. The active selectable model still projects local traffic onto
  one endpoint serializer and folds unresolved stack work into a
  participant-indexed base. Add peer topology and per-link routing, ingress
  service and receiving-HBM interaction, priced reduction lanes and proxy
  operations. Identify those terms from pinned B200 per-link traffic, HBM
  counters, reduction-kernel timing and proxy timestamps over payload and
  participant sweeps with held-out cells. Require exact byte and work
  conservation, one timing authority for every term, no local/fabric double
  count, and held-out phase-completion error no larger than 10 percent or
  1 microsecond, whichever is larger. Report the reduced-form profile's
  before error and preserve the exact `legacy` and all-remote identity paths.
- COMP-17 (Precision; P1; M): after COMP-6 supplies per-invocation captured
  shapes, populate `estimate_layers` for `ProfileTableProvider` and
  `TraceCalibratedGpuProvider`. The current surrogate is the step sink's even
  split whenever these calibrated providers are selected. Use a real model's
  measured per-layer profile, or a published layer-heterogeneity reference,
  as the fidelity anchor and calibration target. Acceptance requires the
  modeled normalized layer-to-layer shape to match that anchor within its
  declared measurement uncertainty. Use measured per-layer kernel durations
  as the identifying observable, reconcile their integer sum to the existing
  fused estimate exactly, and require every rendered cumulative boundary to
  remain within the declared capture uncertainty. The explicit no-breakdown
  path must retain the accepted GOAL bytes and TTFT exactly.
- COMP-21 (Precision; P1; L): calibrate the active optional RNIC producer
  task shapes that currently use a synthetic normalized trace. The v1
  surrogate charges one 64-byte descriptor store plus publication for a CPU
  proxy, or one 64-byte WQE store, one 4-byte doorbell-record store and
  publication for GPU initiation, with one CTA, one warp and minimal
  residency. Calibration must resolve the current GPU-initiated overlap: the
  producer task charges that 4-byte doorbell-record update before effective
  submission, then the native path charges the same physical update as a
  `DoorbellRecord` host store starting at submission. Assign its service to
  one timing authority and retain only the ordering projection at the other
  boundary. Capture GPU descriptor publication and mapped-UAR submission on
  the selected production GPU while sweeping batch sizes 1, 4 and 16 and
  idle, half-resident and residency-saturated neighbors. Use task admission,
  producer completion and RNIC-visible doorbell time as the identifying
  observables. Replace the trace and profile entries only when an independent
  validation capture predicts completion and queue wait within the larger of
  two GPU cycles or 10 percent in every cell. Report the synthetic
  before-versus-calibrated after error for every cell. The disabled coupling
  and host-CPU paths must retain every accepted timestamp and artifact byte.
- COMP-22 (Precision; P1; L): calibrate the GPU resource demand of the active
  cross-node collective path before CORE-26 and CORE-27 replace TRAF-7's
  independent-resource surrogate. Capture pinned NCCL collectives across
  payload, participant and channel-count sweeps, alone and beside compute- and
  HBM-bound kernels. Use kernel residency, channel occupancy, SM issue, HBM
  read/write traffic, network ingress/egress and any copy-engine or GPUDirect
  activity as identifying observables. Record an explicit zero for resources
  absent from the measured path. Replace the synthetic demands only when an
  independent holdout predicts task completion and queue wait within the larger
  of two GPU cycles or 10 percent in every cell, and report the surrogate's
  before-versus-calibrated error. The calibration-off path must preserve every
  accepted TRAF-7 timestamp and artifact byte.
- COMP-23 (Precision; P2; L): record the calibrated per-kernel duration spread
  as capture evidence beside the mean-valued table. The landed profile table and
  trace-calibrated service model return one value per input, which cannot
  express the run-to-run spread that clock, cache and scheduling variation
  produce on real silicon, and a calibration that reports only a median cannot
  say how well identified its own constant is. The Turing method anchor supplies
  41 raw samples per family, dtype and shape cell and demonstrates why the
  record must retain outliers rather than only a mean. Those synthetic TU116
  samples validate the artifact shape but do not calibrate production kernels.
  Fit the spread per production kernel family after COMP-1 and COMP-5 provide
  the target capture, carry the fit provenance and calibration envelope, and
  validate held-out quantiles against raw silicon samples. Report the
  deterministic point-table error before the distributional result.
  Scope constraint from the standing kernel-time determinism decision
  (maintainer, 2026-08-18): the fitted spread is calibration evidence about how
  well the constant is identified and about capture-environment stability, and
  it feeds the estimate's honest uncertainty. It is not a sampling source. No
  provider may draw from it to price a kernel, no seed enters a service path,
  and a reported latency tail is owned by COMP-9's chain rather than by this
  entry. The deterministic providers remain exact compatibility levels and their
  accepted artifacts stay byte-identical.
- COMP-24 (Precision; P1; M): extend the registered mixed-makespan forms
  beyond the single frozen fixture they were measured on. COMP-12 registered
  one issue-order pair and one residency-gated pair, so
  `decompose_mixed_makespan` refuses a replay in which more than one task
  waited for residency, and no measured row covers other shared-memory
  fractions, register or warp pressure, launch shapes or instruction mixes.
  The surrogate being replaced is the assumption that the two-task rows
  generalize. Use isolated controls, admission cycles and concurrent
  makespans as the identifying observables, sweep the residency currencies
  independently so the binding one is identified rather than assumed, and
  require the extended form to predict each held-out cell exactly on the
  synthetic fixture before any silicon claim. The registered two-task rows
  must stay exact.
- COMP-25 (Precision; P1; M): connect the concurrent kernel service to a
  production step path. The trace-driven SM scheduler is reachable through
  `CoarseDeviceRuntime(kernel_services=...)` and COMP-12 demonstrated the
  chain to `StepResult`, TTFT and TPOT, but no production study or step sink
  selects it. Every reported production step therefore takes the scalar
  `ComputeWork.nominal_duration_ps` path, whose concurrent makespan is the
  independent-resource maximum and carries neither registered form. Supply
  the per-operation `KernelLaunch` records a production step needs (COMP-6
  owns the per-invocation shapes), select the service explicitly, and report
  the before-versus-after TTFT and TPOT change on one accepted study. The
  explicit scalar off path must keep every accepted baseline timestamp
  exactly.
- COMP-28 (Precision; P2; L): After COMP-21 supplies device-bound structural
  captures for CPU-proxy and GPU-initiated network submission, fit and
  validate their scalar host-initiation projections for the analytical
  fallback used only while structural submission is disabled. Carry GPU,
  host, RNIC and submission-class provenance plus predeclared capture
  uncertainty; held-out ready-to-RNIC-visible latency must remain within that
  uncertainty. The ideal zero-cost profile remains the exact compatibility
  path.
- COMP-41 (Precision; P2; M): attach measured per-port ceilings to a shipped
  architecture profile. COMP-34 landed ports that carry a ceiling with its
  provenance, but every ceiling reachable today is either read out of a
  synthetic study calibration (`calibration_derived`) or declared by a study
  (`model_configuration`); no shipped profile carries a `first_party_measured`
  port ceiling, and the A100 and GH200 seed profiles declare no copy engine and
  no NVLink profile at all, so they compose to a device with no ports. The
  surrogate being replaced is the absence of a port ceiling on any shipped
  profile. The identifying observables are the measured cells already published
  by
  [a100_hardware_envelope_v1](../../examples/a100_hardware_envelope_v1/RESULTS.md)
  and
  [gh200_hardware_envelope_v1](../../examples/gh200_hardware_envelope_v1/RESULTS.md):
  26.78 GB/s host to device and 26.19 GB/s device to host on PCIe generation 4
  by 16, 419.93 GB/s inbound against 169.96 GB/s outbound on Grace C2C, 94.00 to
  94.07 GB/s per NVLink3 ordered pair with 281.65 GB/s of per-GPU egress, and
  133.24 to 133.27 GB/s per NVLink4 pair with 398.71 GB/s of egress.
  Acceptance: each shipped ceiling carries its envelope study as provenance and
  its own validity window, the asymmetric host link is expressed as two ports
  rather than one averaged rate, a request for an architecture with no measured
  ceiling is rejected rather than borrowing another architecture's number, and
  every accepted artifact stays byte-identical. This is P2 while no study
  selects a measured port ceiling and becomes P1 when one does.
- COMP-43 (Precision; P1; M): price the fixed per-kernel cost that neither
  compute provider carries. The surrogate being replaced is the absence of any
  floor: `RooflineProvider` returns `max(flops/peak, bytes/bandwidth)` and
  `ProfileTableProvider` returns a table entry, so a kernel whose work is
  smaller than the device's own per-kernel cost is priced below what the device
  can do. The identifying observables are first-party and already measured on
  the target architecture by the
  [A100 kernel constants study](../../examples/a100_kernel_constants_v1/RESULTS.md):
  the uninstrumented back-to-back period of an empty kernel is 1.904
  microseconds, and the captured granite MoE expert GEMMs at their captured
  expert loads measure 4.725 to 9.227 microseconds against memory roofs of
  0.578 to 1.275 microseconds, a factor of 5.17 to 12.20 over all 18 cells.
  Acceptance: a per-kernel floor whose value is measured on the architecture it
  is applied to and refuses an architecture it was not measured on, an explicit
  off path that reproduces every accepted artifact and timestamp byte for byte,
  and a
  reported before and after on the decode step of the granite fixture with the
  omitted excess bounded rather than estimated. The evidence this task consumes
  comes from a void run, so a non-void measurement (COMP-45) is a prerequisite
  for the calibrated value even though the mechanism can land first.
- COMP-45 (Precision; P1; M): produce a non-void A100 kernel-constant run. The
  [A100 kernel constants study](../../examples/a100_kernel_constants_v1/RESULTS.md)
  is void twice on its stability preconditions, so its constants close nothing
  and its profile table is deliberately withheld. Two causes are identified and
  neither is a kernel property. Application clock control is denied on the
  Merlin A100 partition, so a cell that spans the 1275 to 1410 MHz boost
  boundary mixes two constants; and one `cudaEventRecord` between two
  consecutive launches costs 2.34 microseconds of device time, so a
  per-repetition chain does not measure a short kernel. The surrogate being
  replaced is the flat 0.7 roofline derate on `a100`. Acceptance: a protocol
  whose stability precondition is achievable without clock control, stated and
  frozen before the run; every scored cell inside its own frozen dispersion
  ceiling; and a `simllm-profile-table-v1` artifact loadable by
  `ProfileTableProvider` whose held-out interpolation error meets COMP-1's
  registered median 10 percent and p95 20 percent bars. The void run already
  reaches 0.70 percent median and 18.53 percent p95 on its held-out shapes, so
  the bars are reachable; what is missing is a run whose guards hold.
  Boundary. COMP-45 owns producing the artifact; COMP-1 is its consumer and
  keeps the surrogate claim, so COMP-45 does not restate it. The registry
  answers the consumption question ONE way, stated here: COMP-5 remains the
  gate, and COMP-1 may consume an A100 profile only once COMP-5's
  environment-scoped stability bar is met on the cells that profile contains.
  COMP-45 is the work that makes that possible on an allocation without clock
  control; it does not bypass COMP-5's gate, and closing COMP-45 does not by
  itself license consumption.
- COMP-47 (Precision; P1; L): reach a non-void A100 graph-launch run and
  install the two host profiles it produces. The
  [A100 graph launch study](../../examples/a100_graph_launch_v1/RESULTS.md) is
  reviewed `VOID`: fatal guard `GG7` was violated, so its behavioral score is
  uninterpretable and no fraction of it is a result. Fourteen of its 15 scored
  expectations passed and one failed, and that 14 is not a score; it is written
  down only so a reader can see which relations survived. `GG7` bounds the
  block-mean dispersion of every reported period at 4 percent, which a chain of
  one to eight kernels cannot meet against the device's 1024 ns event quantum.
  This is L rather than S because closing it needs a fresh allocation on the
  target hardware and a re-frozen protocol, which is hardware evidence.
- COMP-48 (Precision; P1; M): resolve the CUDA-graph clause of the kernel-time
  determinism contract against first-party measurement, once the maintainer has
  ruled. The
  [A100 graph launch study](../../examples/a100_graph_launch_v1/RESULTS.md)
  measured a device-side per-kernel cost that is 1.415 to 1.506 microseconds
  larger in eager mode than under CUDA-graph replay, roughly constant across
  kernels whose own periods span 8.9 to 89.6 microseconds, of which a null
  kernel accounts for 1.080 microseconds. Two readings fit that observation
  equally well and this entry commits to neither. Under the first, the residual
  0.335 to 0.426 microseconds is kernel service time, the constant is
  launch-mode conditioned, and the contract clause needs qualifying. Under the
  second, the residual is a device front-end gap that sits outside kernel
  service time, the clause is intact as written, and what is missing is a
  front-end term the model does not yet carry. The surrogate being replaced is
  the absence of any first-party evidence on the question, which is why the
  clause carried no qualification before. The identifying observable that
  separates the two is per-kernel timing inside a captured graph, which the
  driver on this allocation refuses, returning `invalid argument` for an event
  recorded during stream capture; a different mechanism, for example Nsight
  Systems kernel activity rows over a replayed graph, is required. Acceptance:
  the maintainer's ruling is recorded; the contract text is qualified or
  confirmed to match it; a measurement separating service time from front-end
  gap on the target architecture is published; and the deterministic compute
  path stays byte-identical whichever reading wins. This task does not
  pre-empt the ruling and must not be closed by asserting one of the two
  readings without it. The surrogate being replaced is the absence of any A100 entry
  in `HostInitiationModel`, whose calibrated profiles today accept only
  `gtx1660-ti-sm75`. Acceptance: a freeze whose dispersion guard is scoped to
  the periods the study actually publishes, stated before the run; a run whose
  guards hold; and `a100-epyc-eager-host` and `a100-epyc-cuda-graph` installed
  with the same fail-closed device check the Turing profiles carry, rejecting
  every key except `a100`. The measured values are already published as
  retained evidence: 1,629,633 ps per eager launch over an empirical 1,625,986
  to 1,927,260 ps, and 1,647,674 ps per graph replay independent of chain
  length. Installing them from the void run is refused on purpose.

### Completeness

- COMP-13 (Completeness; P1; M): extend `simllm-gpu-model-artifact-v2` with a
  narrow concurrent replay record for `GpuTask` inputs and
  `GpuConcurrentEstimate` outputs,
  including task order, per-task submission/eligibility,
  admission/completion, requested and
  transacted HBM/NVLink bytes, request counts and deterministic replay
  validation. Until that record lands, concurrent demo CSVs are reviewed
  evidence but are not GPU-model artifacts.
- COMP-14 (Completeness; P2; L): add optional NCCL algorithm builders for
  tree all-reduce, all-to-all, reduce-scatter and all-gather behind an
  explicit algorithm selection. The ring builder remains the identity
  baseline: selecting or omitting the default ring path must preserve every
  accepted ring timestamp, counter and task order exactly.
- COMP-15 (Completeness; P1; L): model the NCCL software stack with the real
  stack's functional names and interfaces, trimmed to the main path. The
  audited zero-time first slice is landed: communicator and ring setup,
  explicit ring-step chunk planning, GPU send-FIFO tail publication,
  `ncclProxySaveOp` queueing, independent CPU proxy progression,
  `ncclNet.isend`, verbs post, RNIC doorbell, external CQE production, CQ poll,
  proxy head-credit return, and distinct intra-node and inter-node call loops
  all emit strict events on one caller-owned clock. The
  [study](../../examples/nccl_stack_v1/RESULTS.md) freezes and validates the
  exact call sequences and planner relations.
  Remaining work is to replace deliberate zero-time boundaries and
  metadata-only movement with calibrated service mechanisms connected to the
  existing GPU, PCIe, native RNIC and fabric authorities; add the
  GPU-initiated leg; project selected events through the supported runtime and
  metric chain; and land the VLLM-14 and SGL-11 adapter callers. Receive-leg
  progression must wire `recvProxyProgress`, `ncclNet.irecv`, `ncclIbIrecv`,
  `wrap_ibv_post_recv`, receive completion through `ncclNet.test` and
  `wrap_ibv_poll_cq`, receive-connector tail publication, and GPU `waitPeer`
  plus `postPeer` head-credit return. These additions must retain one timing
  authority and the explicit bypass behavior.
  Intra-node collectives must compose with the NVLink-class egress model and
  stay off the fabric. Inter-node transfer and receive completion must project
  through CORE-4 and CORE-5 to `CompletionEvent`, `StepResult`, TTFT and TPOT.
  Boundary against BACK-47: this task owns the stack's own calibrated service,
  its receive leg and its metric projection, while BACK-47 owns the
  device-facing packet-emission contract at the plugin ABI seam. Neither may
  claim the other's half.
  Add the BACK-20 GPU-initiated leg behind the same upper interface while
  preserving the CPU-host proxy path as the default identity baseline. The
  VLLM-14 and SGL-11 simulated communicators remain the adapter callers that
  must connect to this stack. Function and event identities must remain stable
  so later captures, timing calibration and adapter traces align with this
  first slice.
- COMP-35 (Completeness; P2; M): instantiate vendor peer ports, so an AMD ROCm
  GPU and a UALink pod can be expressed at all. Once COMP-34 lands port objects,
  a vendor instantiation names the peer port xGMI or UALink rather than NVLink,
  names the collective producer RCCL rather than NCCL on the AMD arm, and
  supplies the envelope slots those names need. Neither protocol has a
  first-party measurement in this repository and the only figures available for
  either are vendor or consortium nameplate, so the instantiation must fail
  closed exactly as a calibrated B100 or H100 host-cost request already does,
  rejecting during configuration instead of borrowing an NVLink ceiling or an
  NVLink efficiency. UALink is the sharper case of the same rule: the UALink 200G
  1.0 specification states a 200 GT/s per-lane data rate carried at a 212.5 GT/s
  signalling rate, so taking the headline figure as a payload ceiling repeats
  exactly the NVLink4 signalling-versus-payload error the port taxonomy already
  records. Acceptance: a declared xGMI or UALink profile carrying its own
  provenance and validity window is required before any cell on that protocol
  runs; an undeclared or unmeasured request is rejected with a diagnostic naming
  the missing profile and the port it belongs to; and every accepted NVIDIA cell
  stays byte-identical. This is P2 while no AMD or UALink study exists and
  becomes P1 when one opts in. COMP-34 landed the port objects and made the xGMI
  protocol nameable; the kernel-time determinism contract added UALink beside it
  on the peer-link role, and both are rejected at configuration time with a
  diagnostic naming this task, so what remains is a declared ceiling per
  protocol with its own provenance and validity window, plus the RCCL producer
  naming.
- COMP-42 (Completeness; P2; S): normalize how the two adapter geometry readers
  spell the optional dtype widths on `ModelDims`. The vLLM reader resolves
  `weight_dtype_bytes` and `kv_dtype_bytes` from the quantization and cache
  configs and stores explicit floats; the SGLang reader stores `None` and lets
  `ModelDims` fall back to the activation width. Both resolve to the same number
  through `weight_element_bytes` and `kv_element_bytes`, so no reported
  picosecond moves today, which is measured by
  [kernel_determinism_v1](../../examples/kernel_determinism_v1/RESULTS.md)
  and pinned by `tests/test_kernel_determinism.py`. The unavailable path is a
  consumer that compares or hashes `ModelDims` itself: two adapters describing
  one identical rank would disagree, which is the failure mode BACK-50 already
  records for the effective-hardware snapshot. Give SGLang the same quantization
  and cache-dtype resolution vLLM has, or make both store the resolved width,
  and keep the explicit unresolved path testable. The off path is the current
  behavior: every accepted artifact and every priced step must stay
  byte-identical, and the pinning test must be updated in the same change rather
  than deleted.
- COMP-40 (Completeness; P2; M): the landed GPU ports declare capabilities but
  emit no packet event, so an intra-node leg still cannot report an extent, an
  attempt, a TX boundary or an arrival in the same language a wire port uses.
  The three transport-control capabilities (ECN marking, priority flow control,
  congestion notification) exist today only to be rejected by name, and the
  rejection diagnostic points at BACK-48. Boundary against BACK-48: that task
  owns making the ABI v2 vocabulary reachable from a non-wire port at all, while
  this one owns binding the GPU host and peer ports to it, including which
  capabilities a GPU port may then honestly advertise. Acceptance: an intra-node
  transfer emits session-unique extent and attempt identity through a GPU port,
  loss, duplication and double-charged bytes are detectable from those events,
  and the no-emission path preserves every accepted timestamp, counter and
  artifact byte exactly. This is P2 while no study consumes port events and
  becomes P1 when TRAF-45 packetizes the intra-node leg.
- COMP-44 (Completeness; P2; S): let a calibrated host profile carry a fixed
  per-invocation cost beside its per-launch constant. `HostInitiationModel`'s
  calibrated form has exactly one term, `point_ps_per_launch`, composed as
  `max(C, N * g)`, which is the right shape for eager launching and the wrong
  shape for CUDA graph replay. The
  [A100 graph launch study](../../examples/a100_graph_launch_v1/RESULTS.md)
  measured the graph host cost at 1.574 to 1.686 microseconds per replay across
  chain lengths 1 to 256, a fitted per-node slope of 0.000297 microseconds at
  an R-squared of 0.516, so there is no per-launch constant to fit: the cost is
  a fixed per-replay term plus a per-node term indistinguishable from zero.
  Expressing it as a per-launch constant makes the published point depend on
  the chain length, which is why that study's `a100-epyc-cuda-graph` point is
  declared scoped to a reference chain length rather than universal. The
  surrogate being replaced is that scoping. Acceptance: a calibrated profile
  may declare a fixed per-invocation term and a per-launch term, the
  composition states which one a launch class uses, the exact `ideal` zero
  profile and both Turing profiles reproduce every accepted artifact and
  timestamp byte for byte, and a graph profile built from a fixed term is
  independent of the launch count it is asked about. This is P2 while no study
  selects an A100 host profile and becomes P1 when COMP-47 installs one.
- COMP-46 (Completeness; P2; M): supply a production-grade decode attention
  microbenchmark. The decode lane of the
  [A100 kernel constants study](../../examples/a100_kernel_constants_v1/RESULTS.md)
  reached 5.5 to 13.3 percent of the measured HBM roof even after its warps
  carried four independent online-softmax accumulators, and its time grew by
  3.02 between batch 64 and batch 256 where the KV bytes grew by 4, so it is
  still gaining efficiency with occupancy rather than sitting on the roof. Its
  constants therefore describe that microbenchmark and not a paged or flash
  decoding kernel, and the study says so. The surrogate being replaced is that
  lane's own kernel. The path this adds is a second, selectable decode kernel
  in the study harness; the existing kernel stays reachable and is the explicit
  off path, so the current lane's constants remain reproducible byte for byte
  after the new kernel lands. Acceptance: a decode kernel whose achieved KV
  bandwidth reaches a frozen fraction of the measured roof over the whole batch
  and cache-length grid, with the fraction stated before the run; a published
  comparison against the current kernel on the identical grid; and the off path
  reproducing the published constants of this study exactly. This is P2 while
  no study consumes a decode attention constant and becomes P1 when one does.
- COMP-49 (Completeness; P1; M): reify the xPU's inter-subsystem
  communication as a streaming crossbar behind the common interface. The
  README states the device as pluggable subsystems, the hardware scheduler,
  HBM, the copy engines, the PCIe host port and the scale-up ports,
  communicating over one common interface. Today the service model couples
  them through direct cursor and budget references with no reified
  interconnect object. Add a crossbar of point-to-point streaming lanes: a
  subsystem pushes work descriptors down a lane and consumes them from the
  far end, with no shared bus, since the model deliberately has no NoC on
  the GPU; the crossbar is contention-free by design and every crossing
  emits an observability event. The default composition must preserve every
  accepted baseline byte for byte. BACK-53 owns the RNIC counterpart, a
  NoC-like signal-slot bus whose contention is a registered future upgrade.

### Uncategorized

- COMP-4: multi-axis interpolation in `ProfileTableProvider`. The landed
  rule interpolates along one config axis with every other axis pinned to
  covered values; a query differing on two or more axes raises `KeyError`
  instead of attempting a multilinear fit.
- COMP-6: per-layer kernel shapes. `step_kernels` aggregates each family
  over all layers of the step; SASS tables index per-invocation shapes,
  so the mapping needs a per-layer (per-invocation) split before tables
  can be keyed the way the tracer sees kernels.
- COMP-8: the fused-vs-family sum invariant test compares in float; above
  2 to the 53rd flops (a 32k-token prefill chunk on a 100B-class dense
  rank) ULP effects could mask a real mismatch even though the integer
  identity is exact. Assert the sums in the integer domain when such
  shapes enter scope (audit note, examples/m5/RESULTS.md).
- COMP-10: extend trace replay beyond synchronous normalized per-warp
  instructions. Add subpartition-aware scheduler ownership, barriers,
  `cp.async`, Hopper TMA and warpgroup async issue/commit/wait semantics, plus
  calibrated cache partitions, bank conflicts and hit/miss behavior. Until
  each mechanism lands with capture evidence, its opcode or launch form must
  fail closed rather than borrow a scalar latency.
