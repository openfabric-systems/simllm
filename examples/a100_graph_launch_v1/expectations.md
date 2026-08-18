# A100 graph launch v1 expectations

## Freeze scope and chronology

This is the expectations-only record for stage 2 of the A100 compute
calibration. It is committed before the stage-2 harness exists, before any
timed chain of this study runs, and before any result-producing Slurm job is
submitted. No number produced by this study may be written back into this file.

Stage 2 decomposes launch cost into three separated quantities and tests one
falsifier of the kernel-time determinism ruling. The ruling says CUDA-graph
launch and eager launch differ only in the host launch cost and never in
kernel service time. This study is built so that claim can fail visibly.

## What stage 1 supplied, and its status

[Stage 1](../a100_kernel_constants_v1/RESULTS.md) is reviewed `VOID`: it
violated three fatal guards, so its behavioral score is uninterpretable and it
closed nothing. Its measured magnitudes are retained evidence and are used
here in exactly two ways, both disclosed:

1. as the source of the band widths frozen below, and
2. as an external cross-check on stage 2's own numbers, never as an anchor.

Stage 2 measures its own standalone baseline in the same job, in the same
process, on the same allocation, so its falsifier does not depend on a void
run's constants. That is a deliberate design choice made because stage 1 was
void, and it makes the falsifier stronger rather than weaker.

The stage-1 magnitudes this freeze derives bands from:

| Stage-1 measurement | Value |
|---|---:|
| uninstrumented back-to-back empty-kernel period, eager | 1.904 us |
| per-boundary cost of one `cudaEventRecord` between launches | 2.336 us (stride sweep), 2.352 us (event-only chain) |
| back-to-back period, `gemm_G1_m64` (N 2048, K 1024, M 64) | 9.072 us |
| back-to-back period, `gemm_G2_m1024` (N 1024, K 1024, M 1024) | 18.816 us |
| back-to-back period, `gemm_G4_m1` (N 8192, K 8192, M 1) | 89.648 us |
| batch-mean coefficient of variation, worst scored cell | 3.315 percent |
| SM clock states under load | 1275 MHz and 1410 MHz, ratio 1.10588 |
| measured HBM roof | 1818.21 GB/s |

Two independently accepted prior measurements are also used as comparison
points and are named here so their role is fixed before the run: the
[A100 hardware envelope](../a100_hardware_envelope_v1/RESULTS.md) measured
1.806 us per pipelined eager launch and 0.791 us per CUDA-graph replay node on
this hardware class, and the accepted COMP-2 profiles carry the Turing points
809,306 ps per `cuda-graph-node` and 2,364,255 ps per `eager-host-bound`
launch.

## Definitions, stated rather than implied

**Back-to-back period `P_mode(k)`.** The device-side time per kernel of a long
chain of kernel `k` issued in mode `mode`, measured with CUDA events around
the whole chain and no events between kernels. It is the kernel's service time
plus whatever per-kernel device-side gap that mode carries. It is NOT the
kernel's service time alone, and this freeze never treats it as such.

**Host submission cost.** The host wall time of the launch loop alone,
measured with a monotonic host clock immediately around the loop, taken before
any synchronization and containing none. The device is still executing when
the second host timestamp is taken, by construction, so the host interval
never contains device time and the device events never contain host launch
time. That is what "start and stop grace excluded by construction" means here.

**Null kernel.** An empty `__global__` function with grid 1 and block 1. Its
service time is not zero but it is the smallest kernel this device can run, so
`P_mode(nop)` is an upper bound on that mode's per-kernel device-side gap.

**Differenced kernel service time.** `S_mode(k) = P_mode(k) - P_mode(nop)`.
Subtracting the null kernel in the same mode removes that mode's per-kernel
gap and leaves the kernel's marginal cost. This is the quantity the ruling
claims is mode-invariant, and F1 below is the direct test of that claim.

**Clock-stationary block.** As in stage 1: the NVML SM clock read immediately
before and immediately after a timed block are equal, the memory clock reads
exactly 1593 MHz on both sides, and the NVML throttle-reason word is zero on
both sides. Every reported number comes from clock-stationary blocks of one
clock state, and the state is published with the number.

## Frozen substrate

| Item | Frozen value |
|---|---|
| cluster and partition | `gmerlin7`, `a100-hourly`, account `merlin` |
| allocation | 1 node, 1 task, 1 `nvidia_a100-sxm4-80gb`, 8 CPUs, 64 GiB, 1 hour wall |
| CUDA toolchain | `cuda/12.2.2`, `nvcc` 12.2 V12.2.140, `-arch=sm_80`, `-O3 -std=c++17` |
| GEMM library and dtype | cuBLAS from `cuda/12.2.2`, `cublasGemmEx`, BF16 in and out, FP32 accumulate |
| GEMM layout | `Out[N,M] = W[N,K] * X[K,M]`, leading dimensions N, K, N, the stage-1 repaired layout |
| clocks | not locked, not settable, recorded per block and conditioned on |
| chain lengths `K` | 1, 2, 4, 8, 16, 32, 64, 128, 256 |
| graph replays `M` per cell | 64 |
| repetitions per timed block | 12, each preceded by one untimed priming launch or replay |
| host clock | `std::chrono::steady_clock` |

## The kernel set

Five kernels, all compute-only and all drawn from the stage-1 set. Their
stage-1 back-to-back periods are quoted above.

| Tag | Kernel | Role |
|---|---|---|
| `nop` | empty kernel, grid 1 block 1 | the per-kernel gap probe |
| `g1` | `gemm` N 2048, K 1024, M 64 | granite QKV at a small token count, about 9 us |
| `g2` | `gemm` N 1024, K 1024, M 1024 | granite output projection at a large token count, about 19 us |
| `g4` | `gemm` N 8192, K 8192, M 1 | weight-streaming decode GEMM, about 90 us |
| `mix` | the repeating cycle `g1`, `g2`, `nop`, `g4` | the mixed chain |

## Quantity 1: the ruling's falsifier

- **F1** For each of `g1`, `g2` and `g4` at `K` = 256, the differenced service
  time is mode-invariant: `S_graph(k) / S_eager(k)` lies in [0.95, 1.05]. The
  band is 5 percent because stage 1's worst scored batch-mean coefficient of
  variation on this allocation was 3.315 percent and its arm-to-arm
  reproducibility of a memory-limited constant was within 0.3 percent; 5
  percent is above the observed dispersion and far below any plausible
  mechanism difference.
- **F2** For `g4`, whose 90 us period is well above the 60 microsecond
  threshold at which fixed costs stop dominating, the raw ratio
  `P_graph(g4) / P_eager(g4)` lies in [0.97, 1.03]. A long kernel's period
  should not care how it was launched.
- **F3** For the null kernel the two modes must differ, and in a signed
  direction: `P_eager(nop) / P_graph(nop)` is at least 1.5. Graph replay
  removes the per-launch driver path, so the residual per-kernel device period
  must be smaller in a graph.

If F1 fails, the constant is launch-mode conditioned and this study says so in
those words. It will not be folded into an averaged "graph and eager agree"
statement, and the kernel-time determinism contract in
`docs/modules/compute.md` will be reported as refuted on its CUDA-graph
clause. F2 and F3 are separate risks: F2 can hold while F1 fails if the gap
difference happens to be small next to a 90 us kernel, and F3 constrains the
gap difference itself rather than the kernel.

## Quantity 2: host submission cost

Host submission is timed with host counters around the launch loop only. In
eager mode the loop issues `K` kernel launches. In graph mode the loop issues
`M` = 64 graph replays of an already instantiated `K`-node graph.

- **H1** Eager host submission time is linear in `K`: an ordinary least
  squares fit of the host loop time against `K` over `K` in [8, 256] returns
  R-squared of at least 0.99.
- **H2** The fitted eager per-launch host slope lies in [0.8, 4.0] us. The
  band brackets the A100 hardware envelope's 1.806 us pipelined launch and the
  Turing 2.364 us eager point with margin on both sides.
- **H3** Graph replay host submission cost per replay is flat in `K`: the
  value at `K` = 256 divided by the value at `K` = 1 lies in [0.5, 2.0].
- **H4** The fitted graph per-node host slope, the slope of host cost per
  replay against `K`, is at most 0.10 times the eager per-launch slope.
- **H5** At `K` = 256 the graph host submission cost per enqueued kernel is at
  least 20 times smaller than the eager per-launch host cost. This is the
  signed amortization the campaign brief asks for.
- **H6** The measured A100 and EPYC eager per-launch host cost differs from
  the Turing `eager-host-bound` point of 2,364,255 ps by more than 10 percent,
  and the measured value is the smaller of the two. A launch constant measured
  on one host and GPU pair does not transfer to another.

Entailment: H4 and H5 are not the same statement. H4 constrains a slope over
the whole sweep; H5 constrains one ratio at the largest `K`, and a curve with
a near-zero slope but a large intercept could satisfy H4 while failing H5 at
small `K`. H1 does not entail H2, which is a magnitude. H3 does not entail H4:
a cost that doubles across the sweep still passes H3 while failing H4.

### Output profiles

Two `HostInitiationModel` profiles are installed from the measurement, using
the existing calibrated-profile machinery unchanged:

- `a100-epyc-eager-host`, launch class `eager-host-bound`, point value the
  fitted eager per-launch host slope, empirical range the minimum and maximum
  per-launch host cost over `K` in [8, 256].
- `a100-epyc-cuda-graph`, launch class `cuda-graph-node`, point value the
  graph host submission cost per enqueued kernel at the frozen reference chain
  length `K_ref` = 64, empirical range the minimum and maximum of that
  quantity over `K` in [8, 256].

The reference chain length is frozen here because H3 predicts the graph host
cost is flat in `K`, which makes a per-kernel graph constant depend on `K` by
construction. If H3 holds, the installed `a100-epyc-cuda-graph` point is a
K-scoped sensitivity constant and the report will say so in those words,
because `HostInitiationModel`'s `max(C, N * g)` composition assumes a
per-launch constant and that assumption is wrong for graph replay. The fixed
per-replay cost is published beside it and is NOT installed, because the
calibrated profile form has nowhere to carry it. A task is registered for that
gap rather than a knob being invented for it.

Both profiles reject every GPU key except `a100`, exactly as the Turing
profiles reject every key except `gtx1660-ti-sm75`. This study closes no COMP-2 clause.
COMP-2 is already closed; what this supplies is the A100 leg that its
fail-closed device check currently refuses, and nothing beyond that.

## Quantity 3: the device inter-kernel gap in a graph, reserved

This constant is the seed for a future device front-end model that treats
kernel launch as port traffic. It is measured, recorded with provenance, and
wired to nothing.

- **D1** `P_graph(nop)` lies in [0.3, 1.5] us. The band brackets the A100
  hardware envelope's 0.791 us graph replay node with wide margin.
- **D2** `P_graph(nop)` is roughly constant in `K`: its value at `K` = 256
  divided by its value at `K` = 16 lies in [0.8, 1.25].
- **D3** `P_eager(nop)` lies in [1.5, 3.0] us, bracketing stage 1's measured
  1.904 us and the A100 hardware envelope's 1.806 us.
- **D4** `P_graph(nop)` is at most 0.6 times the fitted eager per-launch host
  slope. The device front end is cheaper than the host launch path it replaces.

Entailment: D4 is not entailed by D1 and H2 together, because their bands
overlap in a region where the ratio exceeds 0.6.

## Mixed chains

- **M1** The graph makespan of one `mix` cycle equals the sum of the graph
  makespans of its four members, measured separately at the same `K`, within 5
  percent.
- **M2** The same additivity holds in eager mode within 8 percent. The wider
  band reflects the host path's larger dispersion.

## Fatal guards

A violated fatal guard voids the run. No guard in this study is declared
survivable.

- **GG1** The job sees exactly one `NVIDIA A100-SXM4-80GB`, MIG disabled, ECC
  enabled, with no foreign compute process at the start or the end, and both
  the harness and `nvidia-smi` report the same GPU UUID.
- **GG2** Every reported number comes from clock-stationary blocks of a single
  clock state, with at least 8 such blocks, and no reported number averages
  across clock states.
- **GG3** Every CUDA graph instantiates successfully and reports exactly the
  node count its chain length implies, read back from the graph object rather
  than assumed.
- **GG4** Every device makespan is measured by CUDA events recorded inside the
  measured stream, and every host submission cost is measured by a host clock
  pair that contains no synchronization call and no event record. The harness
  records, per cell, that the host interval closed before the first
  synchronization.
- **GG5** No measured period is negative, and no GEMM period falls below its
  own two-sided roofline floor computed at the measured stage-1 HBM roof of
  1,818,210,000,000 B/s and the clock-derived FLOP peak of the block's own
  clock state.
- **GG6** Every GEMM in the set returns the arithmetically correct result on a
  fixed element sample, checked once per kernel outside every timed region.
- **GG7** The batch-mean coefficient of variation of every reported period is
  at most 4 percent. The ceiling is derived from stage 1, whose worst scored
  batch-mean coefficient of variation on this uncontrolled-clock allocation was
  3.315 percent; a 2 percent ceiling is not achievable here and pretending
  otherwise would void every run by construction.
- **GG8** The eager and graph arms of a given kernel and chain length observe
  the same clock state, so a mode comparison is never a clock comparison.

## Scoring

The scored behavioral denominator is 15: three in F, six in H, four in D and
two in M. They are one evidence class: measured device and host times checked
against magnitudes and relations frozen before the run. Fatal guards GG1
through GG8 are unscored and never enter that denominator.

Recorded but unscored: the full `K` sweep of every quantity, the per-kernel
in-graph times measured with inner event pairs and their instrumentation
correction, the graph instantiation cost, the clock telemetry, and the
comparison of stage 2's own standalone periods against stage 1's retained
values.

## What this study will not claim

- It does not close COMP-1. It measures launch and host cost, which is COMP-1's
  second blocker, but it captures no production framework kernel and calibrates
  no SASS replay, so the first blocker is untouched and the compute-only step
  error clause stays unreachable.
- It does not reopen or re-close COMP-2. COMP-2 is closed. This supplies the
  A100 leg its device check refuses today.
- It wires nothing to the device inter-kernel gap. That constant is recorded
  with provenance and consumed by no code path.
- It measures one A100 SXM4 80 GB on one AMD EPYC host. The stage-1 and GH200
  evidence already shows host-issue constants move with the host, so nothing
  here transfers to another host even at fixed GPU generation.
