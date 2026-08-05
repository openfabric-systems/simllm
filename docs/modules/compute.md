# simllm.compute

Pluggable compute-time providers plus the host initiation model. The core
needs one number per GOAL `calc` node: how long a rank computes before it
hands data over. Providers answer at different fidelity/cost points; the
step loop always reads tables or analytical estimates, never a cycle-level
simulator.

## Interface

- `ComputeProvider.estimate(kernel: KernelSpec, gpu: GpuSpec) -> DurationEstimate`
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

## Status

Both providers, the transformer step model (fused and family-decomposed)
and the host model are implemented and tested. The M5 first slice landed
the COMP-1 groundwork: `step_kernels`, the `simllm-profile-table-v1`
artifact with provenance, and 1D log-linear interpolation (closing
COMP-3; the multi-axis extension is COMP-4). The offline SASS pipeline
itself (below) has not run yet; it is blocked on trace-capture hardware
(COMP-5). MoE geometry landed with the same slice and is exercised by the
examples/m5 studies together with the MoE traffic mapping
([traffic](traffic.md), examples/m5/RESULTS.md).

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
- Populate `simllm-profile-table-v1` with capture hash, kernel hash, GPU,
  tool versions, shape, measured samples, simulated cycles, calibrated
  duration, uncertainty, calibration split and creation date. Tables are
  immutable artifacts; changing any identity field produces a new entry.
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

- COMP-1: run the offline SASS pipeline above and ship the first
  populated per-family table (groundwork landed with M5: `step_kernels`,
  the table artifact, interpolation; blocked on COMP-5 for capture).
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
