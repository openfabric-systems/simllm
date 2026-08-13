# Compute fidelity v1: frozen expectations and corrected projections

This freeze precedes the implementation of `run_study.py`, the CUDA probe
`tools/compute_capture/gpu_fixed_cost_probe.cu`, and every measurement this
study reports. It targets the three questions that decide whether SimLLM's
serving numbers can be defended, because after the network corrections a
decode step is compute bound: roughly 99 microseconds of modeled compute
against a few microseconds of communication.

The behavioral registry remains exactly the freeze in commit `62c088e`. After
the run, the token-ownership study retracted the prefill projection anchor and
the mission study landed a real decode-step band. This copy corrects those two
non-scored projection inputs to 706,622,768 ps and 204,000,000 to 215,000,000
ps. The corrections are post-specified and do not repair or change any frozen
guard, especially XFER-G4.

The study has three parts, kept deliberately separate so a stall in one does
not invalidate the others:

- **VAR** re-reads the immutable Turing capture already tracked in
  `examples/compute_calibration_v1/calibration.json` and adds one new targeted
  device probe, to decide what the 2 percent coefficient-of-variation ceiling
  that kept COMP-1 open actually measured.
- **XFER** states, and where possible machine-checks, what a Turing anchor
  transfers to a production envelope and what it does not.
- **FIX** bounds the fixed per-step cost that the modeled compute path omits,
  from a Turing launch-overhead capture and from the eager-mode launch count
  of a 24-layer top-8 MoE decode step.

## What was already observed before this freeze

Honesty about chronology is a hard rule here, so the prior observations that
could inform a "prediction" are listed first. Anything derived from them is
labeled post-specified, not pre-registered.

1. `examples/compute_calibration_v1/RESULTS.md` publishes the three cells that
   failed the 2 percent all-sample coefficient-of-variation guard, with their
   minimum, median, maximum and CV: `attn_gemm` FP32 shape 8 (99.585 /
   99.809 / 115.392 us, 2.395 percent), `lm_head` FP32 shape 4 (50.784 /
   50.912 / 58.657 us, 2.343 percent) and `attn_score` FP64 shape 1 (26.657 /
   27.073 / 31.168 us, 2.432 percent).
2. The same document publishes that a preceding capture missed 2 of 50 cells.
3. The remaining 47 cells' raw samples have not been read. No quartile,
   trimmed or excursion statistic has been computed for any of the 50 cells,
   including the three above.
4. The corrected projection inputs are the token-ownership study's live prefill
   step 0 makespan of 706,622,768 ps and the mission study's published S5
   decode-step band of 204,000,000 to 215,000,000 ps. They landed after this
   freeze and are post-specified projection evidence. The roughly 99.4 us of
   modeled per-step compute was already published.
5. The active weight-byte figure 556,449,792 and the default envelope
   `GPU_ENVELOPES["b100"]` at 8.0e12 bytes/s with a 0.7 derate are published
   repository facts.

Therefore: every VAR relation is evaluated on the 47 cells whose samples were
never observed and is genuine-risk on that denominator. The three published
cells are evaluated too and reported separately as post-specified
confirmations. Every PROBE and FIX relation is evaluated on data that does not
exist yet and is pre-registered without qualification.

## Physical sanity, stated before any measurement

Every number below is named against its envelope. The repository default is
`GPU_ENVELOPES["b100"]` at `mem_bandwidth = 8.0e12` bytes/s; a caller that
omits `gpu=` gets B100, not H100.

For the motivating decode step at 556,449,792 active weight bytes:

| bound | envelope | value |
|---|---|---|
| weight-read floor at 100 percent bandwidth | b100, 8.0e12 B/s | 69.556224 us |
| modeled value with the flat 0.7 derate | b100, 8.0e12 B/s | 99.36603428571428 us |
| weight-read floor at 100 percent bandwidth | h100, 3.35e12 B/s | 166.10441552238808 us |
| weight-read floor at 100 percent bandwidth | a100, 2.039e12 B/s | 272.9032820009808 us |

Two ratios follow arithmetically and are frozen as exact identities, not as
predictions: the modeled 99.366 us sits at exactly `1/0.7 = 1.4285714...`
times the B100 hundred-percent floor, and exactly `1/1.6716417910447763`
times the H100 hundred-percent floor. A step that is faster than the H100
floor is not an H100 step; the number is a B100 number and must be reported
that way.

For the Turing probe device (GTX 1660 Ti, TU116): no kernel can beat its own
memory traffic at the device's roughly 288 GB/s peak, and no launch can
complete faster than the runtime can enqueue it. The launch probe's floor is
therefore zero and its ceiling is the measured serialized launch-plus-sync
latency; a per-launch cost outside `[0, that latency]` is a defect in the
harness, not a result.

## Part VAR: what the 2 percent ceiling measured

### Definitions, frozen

For a capture cell with samples `d_1..d_n` and median `m`:

- **ratio** `r_i = d_i / m`.
- **excursion**: a sample with `r_i > 1.05`. The 1.05 threshold is frozen here
  and is not tuned after seeing any cell.
- **all-sample CV**: the existing `KernelCaptureCell.coefficient_of_variation`,
  population standard deviation over mean. This is the statistic the failed
  COMP-1 guard used and it is reported unchanged for every cell.
- **trimmed CV**: the same statistic computed over the samples with
  `r_i <= 1.05`.
- **quartile relative spread**: `(p75 - p25) / m` using nearest-rank
  percentiles. Reported descriptively, not scored.
- **excursion fraction**: excursion count over sample count.
- **maximum excursion ratio**: `max(d_i) / m`.

### The hypothesis being tested

The 2 percent all-sample CV bar was written as a proxy for "the median this
table stores is a trustworthy estimate of kernel service time". The hypothesis
frozen here is that on a shared display GPU the all-sample CV is the wrong
estimator for that proxy, because the sample distribution is a tight core plus
a sparse, bounded right tail produced by the environment rather than by the
kernel. If that is true, the core dispersion passes comfortably while the
all-sample CV fails on one or two samples in 41.

The competing hypothesis, which the data may support instead, is that kernel
service time on this device is genuinely dispersed at the 2 percent level. In
that case the trimmed CV also fails and the ceiling was correctly specified
and correctly failed. Both outcomes are reportable; the second one leaves
COMP-1 open with nothing gained beyond a cleaner statement.

### Scored relations

Evaluated over the 47 never-observed cells. The denominator is fixed before
the run and does not depend on any measured value.

- **VAR-S1**, 47 instances: every cell's trimmed CV is below 2.0 percent.
  Independent failure mode: if the kernel's own service time is dispersed, the
  central mass spreads and this fails regardless of the tail.
- **VAR-S3**, 47 instances: every cell's maximum excursion ratio is at most
  1.35. Independent failure mode: an unbounded or heavy tail (thermal
  throttling, memory-clock collapse, a stuck sample) would exceed this even
  when the core is tight, and it is a statement about the tail rather than
  about the core, so VAR-S1 does not entail it.
- **VAR-S2**, 1 aggregate instance: over all 2,050 samples of all 50 cells,
  the excursion fraction is below 5 percent. Independent failure mode: a
  frequent rather than sparse disturbance.

### Entailment analysis

The fatal guards below assert inventory, artifact identity and reproduction of
already-published summary values. None of them constrains a trimmed CV, an
excursion ratio or an excursion fraction, because the published summaries
report minimum, median, maximum and all-sample CV only. Every scored VAR
relation is evaluated from raw samples before any artifact-identity guard
runs. Therefore no earlier fatal oracle pins a scored VAR result.

VAR-S1 and VAR-S3 are correlated in the sense that both derive from the same
sample vector, but they are not entailed by one another: S1 can pass with an
arbitrarily large single outlier and S3 can pass with a uniformly dispersed
core. VAR-S2 is a separate aggregate and is reported as one instance rather
than 50 so it cannot inflate the denominator.

### The new device probe

`gpu_fixed_cost_probe --stability` replays the exact failing cell's kernel
shape (family `attn_gemm`, FP32, shape 8, 2,097,152 work items, 256 threads
per block, 8,192 blocks, one 32 MiB L2 flush between samples) for 4,000
launches, timed with CUDA events, and additionally records each block's own
`clock64()` span. Per launch it reports the minimum, maximum and mean block
cycle count alongside the wall duration.

This separates two physically different causes that produce the same wall-time
excursion:

- **clock-state cause**: cycle count stays flat while wall duration rises, so
  the effective SM clock `cycles / duration` fell.
- **residency cause**: cycle count rises with wall duration, so the block was
  resident for longer because something else shared or preempted the SM.

Scored relations, 3 instances:

- **PROBE-1**, 1 instance: at least one of the 4,000 launches exceeds 1.05
  times the run median. This is a prediction that the ceiling is *not* met on
  this device even with a dedicated probe, and it fails if the device turns out
  to be clean, which would be the more convenient result.
- **PROBE-2**, 1 instance: the excursion launches are attributed to exactly one
  of the two causes by the frozen rule below, for at least 80 percent of the
  excursion launches. Attribution rule, frozen: an excursion launch is
  clock-caused when its cycle ratio (launch mean block cycles over the run
  median mean block cycles) is within 1.00 plus or minus 0.05, and
  residency-caused when its cycle ratio is within 5 percent of its duration
  ratio and its duration ratio exceeds 1.05. A launch matching neither, or
  both, is unattributed.
- **PROBE-3**, 1 instance: the excursion fraction over the 4,000 launches is
  strictly between 0 and 5 percent, i.e. the disturbance is sparse rather than
  continuous.

### The proposed refreeze, stated before it is evaluated

If and only if VAR-S1, VAR-S2 and VAR-S3 all pass and PROBE-2 attributes the
excursions to an identified environmental cause, this study proposes the
following replacement for the COMP-1 stability clause. The replacement is
written here, before evaluation, so that it cannot be shaped by the result.

> A capture cell is stable when its excursion-trimmed coefficient of variation
> is below 2 percent, its excursion fraction is below 10 percent of the cell's
> samples, and its maximum excursion ratio is below 1.35, where an excursion is
> a sample above 1.05 times the cell median. Every cell additionally reports
> its all-sample coefficient of variation and its full excursion census; no
> sample is ever discarded from the artifact.
>
> This form applies only to a capture environment explicitly declared as a
> shared display GPU without clock control. The original all-sample 2 percent
> ceiling remains the bar for a controlled environment, defined as a
> non-display device with locked application clocks and exclusive compute
> access, and remains the bar that the production target-architecture capture
> must meet.

The refreeze narrows nothing about production acceptance. It states which
statistic identifies kernel service-time stability when the environment
contributes a sparse bounded tail that the median-valued table is by
construction insensitive to, and it labels that environment explicitly so a
production capture cannot silently inherit the weaker form.

If any of VAR-S1, VAR-S2, VAR-S3 or PROBE-2 fails, no refreeze is proposed,
the original ceiling stands unmodified, and COMP-1 stays open on the original
clause.

## Part XFER: what a Turing anchor transfers

The claim frozen here, to be stated in the module doc and checked where it is
mechanically checkable:

**Transfers**: the capture pipeline (per-cell immediate warmup, per-cell
profiler capture ranges, ordered report collation, `cuda_gpu_trace` CSV
parsing), the `simllm-compute-calibration-v1` artifact schema and its strict
provenance, the immutable train and held-out split, the table compiler that
turns train medians into `simllm-profile-table-v1` with held-out error setting
family uncertainty, the one-axis log-linear interpolation rule, and the
`ProfileTableProvider` seam into `ComputeProvider.estimate`.

**Does not transfer**: every duration, every derived uncertainty, the
per-family efficiency-versus-shape surface, the measured separation between the
calibrated table and the roofline bootstrap, and the physical-bound constants
used in that study, which are TU116 numbers.

Fatal unscored structural guards, 5 instances, evaluated as by-construction
checks and never added to a behavioral denominator:

- **XFER-G1**: the tracked `profile_table.json` declares the Turing device as
  its GPU identity and contains no entry whose GPU key appears in
  `GPU_ENVELOPES`.
- **XFER-G2**: a `ProfileTableProvider` built from that table raises `KeyError`
  for a query naming `b100`, i.e. the seam fails closed rather than borrowing
  the Turing number for a production envelope.
- **XFER-G3**: `HostInitiationModel()` defaults to `initiation_delay_ps == 0`
  and `profile == "ideal"`.
- **XFER-G4**: `RooflineProvider` carries no fixed additive per-step term:
  doubling a kernel's flops and bytes doubles the returned duration exactly, so
  the estimate is exactly proportional and contains no launch, scheduling or
  sampling constant.
- **XFER-G5**: the four envelope floor values and the two ratios in the
  physical-sanity table above reproduce to within 1e-9 relative error from
  `GPU_ENVELOPES`.

## Part FIX: bounding the omitted fixed per-step cost

### Launch count of a 24-layer top-8 MoE decode step, eager mode

Enumerated statically from vLLM 0.26.0 sources for the pinned granite MoE
geometry (24 layers, 32 experts, top-8, the geometry recorded in the
end-to-end capture). Each row is a device-visible kernel launch in the eager
path, i.e. with `torch.compile` and CUDA-graph capture both off. The count is
frozen as a bracket because several rows depend on options this study does not
pin.

Per decoder layer, from `vllm/model_executor/models/granitemoe.py`
`GraniteMoeDecoderLayer.forward`, `GraniteMoeAttention.forward`,
`GraniteMoeMoE.forward` and
`vllm/model_executor/layers/fused_moe/fused_moe.py` `fused_experts_impl`:

| op | minimum | maximum |
|---|---:|---:|
| input RMSNorm | 1 | 1 |
| qkv projection GEMM | 1 | 1 |
| rotary embedding | 1 | 1 |
| KV cache write | 1 | 1 |
| attention | 1 | 2 |
| output projection GEMM | 1 | 1 |
| residual scale and add | 2 | 2 |
| post-attention RMSNorm | 1 | 1 |
| router gate GEMM | 1 | 1 |
| top-k softmax routing | 1 | 1 |
| routing renormalization | 0 | 2 |
| expert assignment and block alignment | 1 | 2 |
| intermediate buffer zeroing | 0 | 1 |
| fused MoE first GEMM | 1 | 1 |
| SiLU and multiply activation | 1 | 1 |
| fused MoE second GEMM | 1 | 1 |
| MoE top-k reduction | 1 | 1 |
| residual scale and add | 2 | 2 |
| **per layer** | **18** | **23** |

Per step, outside the 24 layers:

| op | minimum | maximum |
|---|---:|---:|
| input and position host-to-device copies | 2 | 5 |
| token embedding | 1 | 1 |
| embedding multiplier | 1 | 1 |
| final RMSNorm | 1 | 1 |
| LM head GEMM | 1 | 1 |
| greedy sampling and logits bookkeeping | 2 | 6 |
| **per step** | **8** | **15** |

Frozen bracket: `N_min = 24 * 18 + 8 = 440` and
`N_max = 24 * 23 + 15 = 567` device-visible launches per eager decode step.
Tensor-parallel all-reduces are excluded because the reference configuration
places the collective on the fabric backend, which the network model already
prices.

### Launch-cost probe

`gpu_fixed_cost_probe --launch` measures, on the Turing device:

- `empty_stream_cpu_enqueue_ns`: host wall time of N `cudaLaunchKernel` calls
  for an empty kernel with no synchronization, divided by N.
- `empty_stream_pipelined_ns`: the same N launches plus one final
  synchronization, divided by N. This is the launch-bound throughput of an
  eager stream.
- `empty_stream_serialized_ns`: N iterations of launch plus
  `cudaDeviceSynchronize`, divided by N. This is the per-launch latency
  ceiling.
- `empty_graph_ns`: 512 empty launches captured into one CUDA graph, replayed
  200 times, divided by the node count. This is what a CUDA-graph deployment
  pays instead.
- `long_isolated_ns` and `long_backtoback_ns`: a 262,144-item `attn_gemm` FP32
  kernel timed isolated and back to back, whose difference is the device-side
  inter-kernel gap.

### The bound

Modeled step time today is exactly the sum of kernel service time; the
structural guard XFER-G4 establishes that no fixed term exists anywhere in the
compute path. Real step time obeys

```text
T_step >= max( sum kernel service, N * host launch cost ) + exposed device gap
```

so the omitted amount for a decode step whose modeled compute is `C` is

```text
omitted(N, g) = max(0, N * g - C)
```

evaluated at the eager host-bound cost and at the graph-replay cost, over the
frozen launch bracket. `C` is the published 99.36603428571428 us B100 value.

Scored relations, 4 instances:

- **FIX-1**, 1 instance: `empty_stream_cpu_enqueue_ns` lies in
  [300, 20000] ns. A value outside that band means the probe measured
  something other than a launch.
- **FIX-2**, 1 instance: `long_backtoback_ns > long_isolated_ns`, i.e. the
  device-side inter-kernel gap is strictly positive, so even a perfectly
  pipelined host leaves a fixed cost the model omits.
- **FIX-3**, 1 instance: `empty_graph_ns < empty_stream_pipelined_ns`, i.e.
  graph replay is cheaper per launch than eager stream launching. This can
  fail: if the two are equal the eager path was never host-bound and the whole
  upper bound collapses.
- **FIX-4**, 1 instance: the eager upper bound `N_max * empty_stream_pipelined`
  exceeds the modeled 99.36603428571428 us decode compute, i.e. the omitted
  cost is not a rounding error on the mission's binding constraint.

The TTFT and decode-step statements are projections, not scored relations. The
corrected live prefill step 0 makespan of 706,622,768 ps and the mission S5
decode-step band of 204,000,000 to 215,000,000 ps gain only the omitted excess
once. This study reports the resulting bracket in eager and graph-replay form
without claiming either is the production value. The Turing launch costs are a
Turing measurement on this host's CPU; they anchor an order of magnitude, they
are not a B100 launch model, and the study says so.

## Fatal unscored guards

A single violation voids the run for closure purposes. These are structural,
compatibility or by-construction facts and never enter a behavioral
denominator, and they are never reported as a fraction.

| id | guard |
|---|---|
| G1 | `calibration.json` loads through the strict artifact reader with exactly 50 cells and 2,050 total durations |
| G2 | recomputed all-sample CV for the three published failing cells matches the published percentages within 1e-3 absolute percentage points |
| G3 | the recomputed all-sample CV is at or above 2 percent in exactly 3 of the 50 cells |
| G4 | the 47-cell genuine-risk denominator equals 50 minus the 3 published cells |
| G5 | the probe device model and compute capability match the tracked capture provenance |
| G6 | the probe's median duration is within 25 percent of the tracked `attn_gemm` FP32 shape-8 cell median, so the instrumented twin measures the same kernel |
| G7 | every probe launch records a positive duration and a positive block cycle count |
| XFER-G1 | tracked profile table declares the Turing GPU and no `GPU_ENVELOPES` key |
| XFER-G2 | `ProfileTableProvider` raises `KeyError` for a `b100` query |
| XFER-G3 | `HostInitiationModel()` defaults to zero delay and profile `ideal` |
| XFER-G4 | `RooflineProvider` duration is exactly proportional, with no additive constant |
| XFER-G5 | envelope floors and ratios reproduce to 1e-9 relative |

## Inventory

| family | instances |
|---|---:|
| VAR-S1 trimmed CV below 2 percent | 47 |
| VAR-S3 maximum excursion ratio at most 1.35 | 47 |
| VAR-S2 aggregate excursion fraction below 5 percent | 1 |
| PROBE-1..3 excursion reproduction, attribution and sparsity | 3 |
| FIX-1..4 launch cost bracket and bound | 4 |
| **total genuine-risk instances** | **102** |

Fatal unscored guards: 12. Post-specified confirmations on the three published
cells: 3, reported separately and never added to the 102.

## What closure this study can and cannot support

COMP-1 cannot close here. Its registered clauses require production framework
kernels on the target architecture, a dynamic SASS ledger and pinned Accel-Sim
replay, none of which this study attempts. The most this study can do for
COMP-1 is retire the stability clause as the reason it is open, replace it with
an explicitly environment-scoped form, and leave the production clauses
untouched and open.

COMP-5 cannot close here either. The counter permission requirement is an
administrator action on the loaded driver's `RmProfilingAdminOnly` parameter
and target-architecture allocation is not available. This study can only
sharpen what remains.

A fabricated table is the one unacceptable outcome. No table is produced by
this study, and no Turing number is transferred onto a production envelope.
