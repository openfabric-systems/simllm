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
- `HostInitiationModel`: constant per-operation delay between "ready" and
  "on the wire" (default 0, profile-labeled). The doorbell packet itself is
  modeled in-band on the fabric; host/PCIe/RNIC launch effects default to
  zero delay and zero jitter so network attribution stays clean.

Every estimate carries an honest uncertainty so results can report error
bounds.

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
   absent under COMP-11. A calibration without an `nvlink` profile rejects
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
multi-ring topologies are COMP-11.

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
  with a compatible dynamic SASS and Accel-Sim path.

## Status

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
  validating immutable held-out kernels. Second, the step model has no fixed
  per-step cost at all, and the fidelity study measures an omitted excess of
  1.79 to 12.31 times the whole modeled decode compute of a 24-layer top-8 MoE
  step, so the compute-only step error clause is unreachable until launch
  overhead, host delay and queueing are measured on the target architecture and
  given a seam.
  Do not add an uncalibrated launch constant in the meantime. Acceptance remains
  the environment-scoped stability bar with the controlled form required for the
  production capture, held-out kernel median error below 10 percent and p95
  below 20 percent, per-phase median below 5 percent and p95 below 10 percent,
  and compute-only step error below 5 percent. The roofline and calibration-off
  paths must retain accepted artifacts and timestamps byte for byte.
- COMP-5 (Precision; P1; L): provide the production capture
  environment required by COMP-1. On 2026-08-12 the local GTX 1660 Ti with
  driver 550.90.07 successfully produced CUPTI activity timing through Nsight
  Systems. Nsight Compute attached but returned `ERR_NVGPUCTRPERM`, profiled
  no kernels and reported that this user lacks performance-counter permission;
  the loaded driver exposes `RmProfilingAdminOnly: 1`. The local requirement
  is an administrator disabling that restriction or granting the documented
  profiling capability, followed by a successful counter probe. The display
  GPU also produced isolated samples above the 2 percent per-cell variation
  ceiling in two consecutive post-fix captures, and the fidelity study measured
  the two mechanisms behind them: 93.4 percent of a fresh 4,000-launch excursion
  population is blocks staying resident longer at an unchanged 1,869 MHz
  effective SM clock, i.e. the desktop sharing the SM, and the rest is the
  effective clock dropping to 76.9 percent of that value. Locking clocks and
  freeing the device from the display are both administrator actions, so the
  controlled-environment stability form cannot be met on this host at all, no
  matter how the capture is disciplined. Production closure therefore
  needs a stable non-display or exclusive capture environment, controlled
  clocks, and allocation on the exact A100, H100 or B100 target with compatible
  dynamic NVBit tracing and Accel-Sim support. Acceptance is a nonempty
  activity trace, successful required-counter probe, exact tool and GPU
  provenance, and every registered cell below the controlled-environment
  stability ceiling.
- COMP-7 (Precision; P1; M): MoE compute assumes perfectly balanced routing:
  every rank computes `top_k` experts' flops for its own tokens and streams all
  resident experts once. Consume the landed `simllm-routed-experts-v1`
  projection through `RoutedMoeSupply`, using the same selected placement
  epoch as traffic, to drive per-rank effective expert load and hot-expert
  imbalance.
- COMP-11 (Precision; P1; L): deepen the active NVLink and NCCL ring model.
  The v1 egress path is one
  flat per-GPU serializer and the ring builder emits only the egress half
  of an all-reduce. Add peer topology and per-link routing, ingress
  service and its interaction with the receiving GPU's HBM, reduction
  lanes so a collective's arithmetic is priced, and proxy operations.
  Calibrate the egress latency and bandwidth from real
  captures rather than the current synthetic profiles, and reconcile the
  intra-node split with TRAF-10 so one collective is never counted both
  here and on the fabric backend.
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
- COMP-23 (Precision; P2; L): add a calibrated per-kernel latency
  distribution provider beside the mean-valued table. The landed profile
  table and trace-calibrated service model return one value per input, which
  cannot express the run-to-run spread that clock, cache and scheduling
  variation produce on real silicon. The Turing method anchor now supplies
  41 raw samples per family, dtype and shape cell and demonstrates why the
  distribution must retain outliers rather than only a mean. Those synthetic
  TU116 samples validate the artifact shape but do not calibrate production
  kernels. Fit a distribution per production kernel family after COMP-1 and
  COMP-5 provide the target capture, carry the fit provenance, calibration
  envelope and seed, and validate held-out quantiles against raw silicon
  samples. Report the deterministic point-table error before the
  distributional result. The deterministic providers remain exact
  compatibility levels and their accepted artifacts stay byte-identical.
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
  Add the BACK-20 GPU-initiated leg behind the same upper interface while
  preserving the CPU-host proxy path as the default identity baseline. The
  VLLM-14 and SGL-11 simulated communicators remain the adapter callers that
  must connect to this stack. Function and event identities must remain stable
  so later captures, timing calibration and adapter traces align with this
  first slice.

### Uncategorized

- COMP-2: calibrated host-initiation profiles (GPU-initiated vs CPU-proxy
  constants) for launch-path sensitivity studies.
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
- COMP-9: extend `DurationEstimate` and profile artifacts from one nominal
  value plus uncertainty to a measured or fitted service-time distribution
  with declared sample count and quantiles. CORE-5 needs this before claiming
  kernel-level p99 or p99.9 tail accuracy; deterministic means remain valid
  for closed-form sanity studies.
- COMP-10: extend trace replay beyond synchronous normalized per-warp
  instructions. Add subpartition-aware scheduler ownership, barriers,
  `cp.async`, Hopper TMA and warpgroup async issue/commit/wait semantics, plus
  calibrated cache partitions, bank conflicts and hit/miss behavior. Until
  each mechanism lands with capture evidence, its opcode or launch form must
  fail closed rather than borrow a scalar latency.
