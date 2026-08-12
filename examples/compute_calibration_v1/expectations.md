# Turing compute calibration v1 expectations

## Freeze scope and chronology

This is the expectations-only record for the first COMP-1 silicon anchor. It
freezes the capture matrix, physical bounds, held-out accuracy distribution,
provider opt-in relation and exact compatibility treatment before any timed
kernel capture, calibration implementation or result-producing run.

The evidence is deliberately bounded to a GTX 1660 Ti method anchor. It can
validate the capture pipeline, provenance record, table compiler,
interpolation and `ComputeProvider` seam. Its timings are not A100, H100,
B100 or Hopper timings and must never populate a profile for one of those
devices.

## Pre-freeze hardware and source audit

The evidence was authored against SimLLM commit
`cede92930a469bd0be2f2c588866885c9e0e3618`. No timed CUDA kernel workload was
run before this freeze. Read-only tool and device diagnostics established the
following inputs, which are context rather than scored outcomes:

- `nvidia-smi` reports an NVIDIA GeForce GTX 1660 Ti, compute capability 7.5,
  driver 550.90.07 and 6 GiB of device memory. CUDA 12.4 sees the device.
- The installed tools are Nsight Compute 2024.1.0.0, Nsight Systems
  2023.4.4.54 and CUDA compiler 12.4.99. The repository environment does not
  contain PyTorch, so the anchor uses a framework-neutral CUDA benchmark.
- Querying Nsight Compute metrics stops with `ERR_NVGPUCTRPERM`. The loaded
  driver exposes `RmProfilingAdminOnly: 1`, and this user has no administrative
  counter capability. NVIDIA documents that setting as the state in which
  non-admin users cannot access GPU performance counters, while activity
  tracing remains available. The exact unmet local requirement is an
  administrator setting `NVreg_RestrictProfilingToAdminUsers=0`, or granting
  the documented profiling capability, followed by a successful counter
  probe. See the
  [NVIDIA counter-permission guide](https://developer.nvidia.com/nvidia-development-tools-solutions-ERR_NVGPUCTRPERM-permission-issue-performance-counters).
- Nsight Compute 2024.1 lists Turing TU1xx and GH100 as supported
  architectures in `gpu-support.html:146-155`. Support does not make their
  metrics or instructions interchangeable.
- The installed Nsight Systems report definition
  `cuda_gpu_trace.py:38-61,89-121` defines per-kernel start, duration, launch
  resources, device, context, stream and name from CUPTI activity rows. The
  importer must consume that report's CSV surface rather than duplicate its
  private SQLite query.
- `simllm/compute/provider.py:58-82` is the required provider boundary.
  `simllm/compute/provider.py:256-362` already defines the measured-table and
  one-axis log-linear interpolation semantics. The calibration extends that
  provider through an explicit family-sum opt-in instead of adding a parallel
  table implementation.
- `simllm/compute/transformer.py:268-310` defines the five semantic family
  names and their stable configuration axes. Those names, not the benchmark's
  C++ function names, are the table keys.

The official
[GTX 16-series specification](https://www.nvidia.com/en-us/geforce/graphics-cards/compare/?section=compare-specs)
states 1,536 CUDA cores, 1,500 MHz base clock, a 192-bit memory interface and
no Tensor Cores for the GTX 1660 Ti. The 288 GB/s memory roof is the 192-bit
interface at the device-reported 6,001 MHz double-data-rate clock. The
[Turing tuning guide](https://docs.nvidia.com/cuda/archive/12.2.2/turing-tuning-guide/index.html)
defines compute capability 7.5 and the Turing scheduling model. The
[Hopper tuning guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)
documents Hopper-only mechanisms including the Tensor Memory Accelerator,
thread-block clusters and distributed shared memory.

The transferable claims are the activity-timing capture method, immutable
train/held-out split, raw-sample distribution, binary and static-SASS hashes,
table format, interpolation and provider opt-in. The numeric latencies,
throughput, scheduler behavior, cache behavior, instruction mix and SASS do
not transfer. In particular, this TU116 device has no Tensor Cores and cannot
exercise Hopper FP8 tensor paths, TMA, warpgroup instructions or cluster
launches. Static `sm_75` disassembly is an identity artifact, not the dynamic
NVBit trace required by Accel-Sim and not `sm_90` SASS.

## Frozen capture and split

The benchmark has the five repository family labels `attn_gemm`,
`attn_score`, `mlp_gemm`, `lm_head` and `kv_read`. Each label maps to one
named CUDA kernel implementation. The sweep varies:

- semantic shape `S` over train values `{1, 4, 16}` and immutable held-out
  values `{2, 8}`;
- dtype over `{fp32, fp64}`.

Each shape unit represents 262,144 work items. Every cell receives 10 warm-up
launches before capture and 41 measured launches. A 32 MiB cache-flush kernel
runs before each measured target launch and is excluded by its distinct
identity. The measured target uses 256-thread blocks. This gives 50 captured
cells, 30 train cells, 20 held-out cells and 2,050 measured target durations.

The semantic key uses the repository's native family axes: `new_tokens` for
the two GEMM families, `kv_tokens` with `new_tokens=1` for `attn_score`,
`sampled` for `lm_head`, and `kv_tokens` for `kv_read`. Dtype is represented
by a distinct exact GPU profile identity on the same observed physical
device. Held-out queries therefore differ from covered train rows on exactly
one positive numeric axis and exercise the existing log-linear interpolation
rule.

The capture must record the report digest, benchmark source and binary
digests, static-SASS digest, exact GPU model and UUID, driver, CUDA and
profiler versions, compute capability, dtype, warm-up and cache policy,
observed endpoint core and memory clocks, kernel and launch identity, stream,
grid, block, registers, shared memory, every raw duration sample, split and
creation date. Unsupported or missing fields reject instead of acquiring a
default. The compact profile table contains train medians only and cites the
full calibration artifact digest.

Nsight Systems activity timing is the required capture path for this anchor.
If it cannot attach, export a nonempty CUDA GPU trace and account for exactly
2,050 target rows, no measured table may be published. The implementation and
synthetic-input tests may still land, while COMP-1 remains open and COMP-5 is
rewritten with the exact failure. Nsight Compute's already observed counter
permission failure is not allowed to invalidate a successful activity-timing
capture or to become a fabricated counter value.

## Physical sanity before accuracy

Physical guards are evaluated for every raw target duration before any
accuracy band or exact artifact oracle.

The lower time bound is

```text
floor = max(source-level FLOPs / declared dtype peak,
            compulsory input bytes / 288 GB/s)
```

The cache flush and a target working set larger than Turing L2 make the input
read compulsory for this controlled benchmark. Output bytes are excluded from
the floor because a store may become kernel-visible while still resident in
cache. A duration below the floor is a measurement, work-accounting or cache
protocol defect, not a fast result.

The conservative upper time bound adds, rather than overlaps, serialization
on one arithmetic lane at the 1.5 GHz base clock and one 32-bit GDDR6 channel
at 48 GB/s:

```text
ceiling = source-level operations / 1.5 Gop/s
          + total logical bytes / 48 GB/s
```

The full grid exposes much more parallel service than that bound assumes. A
duration above it is treated as a scheduling, clock, benchmark or accounting
defect. The result must state both bounds and the measured position for every
cell. It must also report the nearest floor ratio and nearest ceiling ratio.

Within each family and dtype, raw median duration must increase strictly from
shape 1 through 2, 4, 8 and 16. Each adjacent fourfold train-shape increase
must have a duration ratio in `[1, 8]`. At every family and shape, fp64 must
not be faster than fp32. These are signed physical trends, not exact roofline
oracles.

## Held-out decision relation

For each held-out cell, silicon truth is the median of its 41 raw Nsight
Systems durations. The calibrated prediction comes only from the two
bracketing train medians for the same family and dtype. The roofline bootstrap
uses the same declared source-level FLOPs and compulsory plus output bytes,
the dtype-specific GTX 1660 Ti peak, 288 GB/s and the repository default 0.7
efficiency.

Absolute percentage error is computed per held-out cell before any exact
identity, row-count, table-roundtrip or digest check. Across all 20 held-out
cells, the calibrated error distribution must have median at most 10 percent
and p95 at most 20 percent. The roofline bootstrap must miss at least one of
those same bands. The calibrated median and p95 must each be strictly below
the corresponding roofline statistic. Every cell remains visible, and each
cell's 41-sample coefficient of variation must be below 2 percent.

This is one genuine-risk family over 20 raw held-out instances. A profiler can
successfully return rows while interpolation is wrong, one family is badly
fit, or the roofline is already inside the band. No earlier fatal oracle pins
the measured median or either predictor.

## Provider and compatibility relations

`ProfileTableProvider` receives an explicit family-sum opt-in. Disabled is the
identity behavior: a fused `llm_step` miss remains the same `KeyError`, exact
family lookups and interpolation retain their accepted values, and saving the
same legacy table remains byte-identical. These are fatal unscored
compatibility guards.

With family sum enabled, one all-train fused query and one held-out fused
query must equal the integer sum of their five independent family estimates.
The enabled path must report a conservative combined uncertainty and must
fail if any family is unsupported. These are two scored component instances:
the sum is evaluated from raw child predictions before a later exact
roundtrip check, so a malformed reducer can fail after every child lookup
succeeds.

This study does not claim that the synthetic Turing microkernels are the live
Granite kernels or that their sum is a production TTFT or TPOT. A supported
production capture and a live `ExecutionGraph` through `CompletionEvent`,
`StepResult`, TTFT and TPOT remain separate closure clauses if they are not
demonstrated.

## Evidence classes and entailment

The scored headline keeps four families separate:

1. held-out accuracy distribution, 20 raw cells;
2. train-shape scaling, 20 adjacent fourfold ratios;
3. fp64 slowdown, 25 family and shape pairs;
4. enabled family-sum response, 2 fused queries.

Accuracy statistics, monotonicity and dtype signs are derived from raw
activity durations before exact capture inventories and artifact checks. The
family-sum response is computed from provider observations before its exact
sum oracle. Each relation can fail in a run that reaches it and is not
entailed by a prior fatal check.

Run configuration, exact capture row counts, kernel names, launch geometry,
sample counts, hashes, schema validation, table roundtrip, physical floors and
ceilings, absence of held-out rows from the fitted table, source-authored work
counts, static SASS identity, counter-permission diagnostics and check-only
behavior are fatal unscored evidence. They are structural, by construction or
change-set guards and never increase a behavioral denominator. Native tool
smokes, focused Python tests and the full Python suite are reported as their
own evidence classes.

## Registered command and pre-freeze dry run

The result-producing command is:

```text
.venv/bin/python examples/compute_calibration_v1/run_study.py \
  --cuda-root "${SIMLLM_CUDA_ROOT:?configure SIMLLM_CUDA_ROOT}" \
  --out "${SIMLLM_WAVE6_RUN_ROOT:?configure SIMLLM_WAVE6_RUN_ROOT}/codex/comp1_compute_calibration/compute-calibration-v1"
```

Before the expectations commit, the same command is run with `--check-only`.
The untracked dry-run harness parses the complete production CLI, validates
only the frozen matrix, formulas, denominator counts and output placement,
imports no SimLLM implementation, invokes no CUDA or profiler tool and creates
no artifact or output directory.
