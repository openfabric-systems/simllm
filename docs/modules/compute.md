# simllm.compute

Pluggable compute-time providers plus the host initiation model. The core
needs one number per GOAL `calc` node: how long a rank computes before it
hands data over. Providers answer at different fidelity/cost points. A compact
trace-driven service model replays captured kernels when a provider or
calibration artifact is built. Its online provider path is a cached lookup;
external Accel-Sim replay also remains offline. Neither simulator runs once
per serving step.

## Interface

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
  library never reads the clock).
- `RooflineProvider`: analytical `max(flops/peak, bytes/bw)` with an
  efficiency derate; classifies compute- vs memory-bound from the kernel
  configuration alone.
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
profiler exports and bulk replay outputs live under `/data3/yifeng/`, never in
Git; the public artifact records their content hashes and provenance.

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
G2 (COMP-12). The built-in
A100/H100 profiles are unvalidated bootstrap seeds and do not establish
production accuracy: their pipeline initiation intervals are derived from
published per-SM unit counts, not measured.

The M5 first slice landed the COMP-1 groundwork: `step_kernels`, the
`simllm-profile-table-v1` artifact with provenance, and 1D log-linear
interpolation (closing COMP-3; the multi-axis extension is COMP-4). The
production SASS pipeline itself (below) has not run yet; it is blocked on
trace-capture hardware
(COMP-5). Therefore COMP-1, COMP-5 and COMP-6 remain open. MoE geometry
landed with the same slice and is exercised by the examples/m5 studies
together with the MoE traffic mapping
([traffic](traffic.md), [M5 results](../../examples/m5/RESULTS.md)).

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
  identity coverage for the supported run; measured coefficient of variation
  below 2 percent for controlled microbenchmarks; held-out per-kernel median
  absolute percentage error below 10 percent and p95 below 20 percent;
  per-phase median below 5 percent and p95 below 10 percent; compute-only
  step error below 5 percent. Every miss is reported, never averaged away.
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
- Hard dependency (COMP-5): trace capture needs a working modern GPU.
  The local GTX 1660 Ti's driver is currently too old for the CUDA
  toolchain (confirmed during the M3 smoke: torch cu130 reports CUDA
  unavailable), so a driver fix or cluster A100/H100 hours are required
  before any trace lands.

## Open tasks

- COMP-1 (Precision; P1; L): run the offline SASS pipeline above and ship the
  first populated per-family table (groundwork landed with M5: `step_kernels`,
  the table artifact and interpolation; the trace-driven service-model
  mechanisms and bootstrap profiles are also landed, but production capture,
  calibration and a populated table remain blocked on COMP-5).
- COMP-2: calibrated host-initiation profiles (GPU-initiated vs CPU-proxy
  constants) for launch-path sensitivity studies.
- COMP-4: multi-axis interpolation in `ProfileTableProvider`. The landed
  rule interpolates along one config axis with every other axis pinned to
  covered values; a query differing on two or more axes raises `KeyError`
  instead of attempting a multilinear fit.
- COMP-5: trace-capture hardware for COMP-1 (local driver fix for the
  GTX 1660 Ti, or cluster A100/H100 allocation). Sub-task of COMP-1.
- COMP-6: per-layer kernel shapes. `step_kernels` aggregates each family
  over all layers of the step; SASS tables index per-invocation shapes,
  so the mapping needs a per-layer (per-invocation) split before tables
  can be keyed the way the tracer sees kernels.
- COMP-7: MoE compute assumes perfectly balanced routing: every rank
  computes `top_k` experts' flops for its own tokens and streams all
  resident experts once. Routed-experts captures (TRAF-2 second half)
  would drive per-rank effective expert load and hot-expert imbalance.
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
- COMP-12 (Precision; P1; M): register the corrected mixed-makespan forms
  measured by the [task-mix study](../../examples/gpu_task_mix/RESULTS.md).
  Findings G1
  and G2 there show that a concurrent makespan is
  `max(isolated durations)` plus a submission-order issue delay, and that
  tasks whose CTAs exhaust an SM's shared memory serialize on residency
  instead of backfilling. Both need a pre-registered form of their own,
  including how the issue-order delay should behave once CORE-4 owns
  submission policy.
- COMP-13 (Completeness; P1; M): extend `simllm-gpu-model-artifact-v2` with a
  narrow concurrent replay record for `GpuTask` inputs and
  `GpuConcurrentEstimate` outputs,
  including task order, per-task admission/completion, requested and
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
- COMP-16 (Precision; P1; M): populate `ComputeProvider.estimate_layers` in
  the live providers so the step sink can replace its current even per-layer
  split with real layer durations. Implement the roofline breakdown first,
  with nonnegative layer estimates whose exact sum equals the fused estimate;
  add profile-table and trace-calibrated breakdowns after COMP-6 supplies the
  per-layer kernel shapes seen by captures. Sweep layer count and TP width and
  require the rendered cumulative-nanosecond calc values to match the provider
  breakdown under the registered truncation rule. Providers without the
  breakdown must retain the byte-identical even-split fallback.
