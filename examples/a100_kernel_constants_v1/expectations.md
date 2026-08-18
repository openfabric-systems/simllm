# A100 kernel constants v1 expectations

## Freeze scope and chronology

This is the expectations-only record for the first silicon measurement of the
per-kernel service constants that the kernel-time determinism contract
(`docs/modules/compute.md`) declares to exist. It is committed before the
measurement harness exists, before any timed kernel of this study runs, and
before any result-producing Slurm job is submitted. No number produced by this
study may be written back into this file.

The contract says a compute kernel's service time is a deterministic constant
with no tail, keyed by kernel family, phase, token and shape inputs, and the
architecture profile. This study measures those constants on one A100 SXM4
80 GB and publishes them as per-family efficiency surfaces over shape. It
measures no framework kernel, replays no SASS, and reports no TTFT or TPOT.

Stage 2 of the same campaign (`examples/a100_graph_launch_v1`) decomposes
launch cost and tests the falsifier that in-graph execution equals the
standalone constant measured here. Stage 2 carries its own separate freeze.

## Pre-freeze discovery facts

An unscored discovery job ran before this freeze. It timed nothing that is
scored here, published no constant, and exists to make the clock policy, the
warm-state definition and the warmup discard measured rather than assumed.
Slurm job `195960` on `gpu101` recorded the following. These are frozen inputs,
not outcomes.

| Observation | Value |
|---|---|
| device | `NVIDIA A100-SXM4-80GB`, 108 SM, 40 MiB L2, ECC enabled, MIG disabled |
| driver and toolkit | 565.57.01, driver API 12.7, `cuda/12.2.2` (`nvcc` 12.2 V12.2.140) |
| application clock lock | **denied**: `nvidia-smi --lock-gpu-clocks=1410,1410` returns "The current user does not have permission to change clocks for GPU 00000000:C1:00.0", exit 4 |
| application clock set | **denied**: `nvidia-smi -ac 1593,1410` returns the same denial, exit 4 |
| supported memory clocks | exactly one, 1593 MHz |
| default applications clock | graphics 1275 MHz, memory 1593 MHz |
| maximum clocks | SM 1410 MHz, memory 1593 MHz |
| observed SM clock states | 210 MHz idle, 1275 MHz under load before boost, 1410 MHz boosted; no other value persisted, and `nvidia-smi dmon` at 1 Hz independently saw only 210, 1275 and 1410 apart from two single-sample transients |
| observed memory clock | 1593 MHz in every sample of both the in-process NVML series and the independent `dmon` series |
| throttle reasons | zero in every NVML sample, through 300 W and 44 C |
| boost latency | 283 ms to 432 ms of sustained load from a 1275 MHz start |
| CUDA event quantum | every observed `cudaEventElapsedTime` value on this device is an exact integer multiple of 1024 ns |
| first-launch cost | the first launch of a `cublasGemmEx` shape costs 1.6 to 314 times its steady value; the second launch is already within 10 percent |
| reps to steady, fixed clock | at most 8 repetitions to reach and hold 5 percent of the steady median in every discovery cell whose clock did not change |
| clock sensitivity | a compute-limited cell measured 4,110,340 ns at 1275 MHz and 3,723,260 ns at 1410 MHz, a ratio of 1.1040 against the clock ratio 1.10588; memory-limited cells measured the same value in both states within one 1024 ns quantum |
| warm versus rotated | at 64 MiB working set the warm and rotated medians were identical to the quantum, both for a 3-buffer triad and a 1-buffer scale |

Clock locking being denied is a permission denial and is reported as one. It
is the reason this study publishes CLOCK-CONDITIONED constants instead of
locked-clock constants, and the reason clock stationarity is a fatal guard
rather than a footnote.

## Definitions, stated rather than implied

**Warm state.** The steady state of `R` consecutive back-to-back launches of
the identical kernel with identical arguments on identical buffers, in one
stream, with no host synchronization between launches, after discarding the
first `K` repetitions. The contract's constant is DEFINED as this warm
steady-state value. Every scored constant in this study is a warm-state value.

**Rotated state.** Identical in every respect except that the kernel's buffers
are cycled over a pool whose distinct footprint exceeds the 40 MiB L2 by at
least a factor 8, so no repetition can reuse the previous repetition's cache
residency. The rotated value and the warm-versus-rotated delta are measured
and reported UNSCORED, as the definition note for the warm-state choice.

**Clock-stationary batch.** A timed batch whose NVML SM clock reading taken
immediately before it and immediately after it are equal to each other, whose
memory clock reads exactly 1593 MHz on both sides, and whose NVML throttle
reason word is zero on both sides. The common SM clock value is the batch's
clock state.

**Clock-conditioned constant.** For kernel cell `c` and clock state `s`, the
mean of the batch means over the clock-stationary batches of `c` in state `s`.
A cell has a constant in state `s` only when it has at least 8 stationary
batches in that state. Constants are never averaged across clock states.

**Timer quantum.** `q` = 1024 ns, the measured `cudaEventElapsedTime`
granularity on this device. Every batch is sized so that `q` is at most 0.5
percent of the batch's elapsed time, which is what makes a batch mean a
sub-quantum estimate of the per-launch constant.

## Frozen substrate

| Item | Frozen value |
|---|---|
| cluster and partition | `gmerlin7`, `a100-hourly`, account `merlin` |
| allocation | 1 node, 1 task, 1 `nvidia_a100-sxm4-80gb`, 8 CPUs, 64 GiB, 1 hour wall |
| CUDA toolchain | `cuda/12.2.2`, `nvcc` 12.2 V12.2.140, `-arch=sm_80`, `-O3 -std=c++17` |
| GEMM library | cuBLAS from `cuda/12.2.2`, `cublasGemmEx` and `cublasGemmStridedBatchedEx` |
| GEMM dtype | BF16 input and output, FP32 accumulation, `CUBLAS_COMPUTE_32F`, `CUBLAS_GEMM_DEFAULT` |
| streaming kernels | grid-stride `float4`, grid 864 blocks of 256 threads (108 SM x 8 blocks) |
| timing | CUDA events on the measured stream; host clock used only where the freeze says so |
| clocks | not locked, not settable; recorded per batch and conditioned on |
| bandwidth unit | decimal, 1 GB/s is 1,000,000,000 B/s |
| warmup discard `K` | 20 repetitions per cell, at least twice the largest observed reps-to-steady and covering the first-launch cost |
| batches per cell | 12, of which the first 2 may be discarded only if not clock-stationary |
| batch size `G` | the smallest power of two for which `G` times the cell's own first-batch elapsed time is at least 200 microseconds, capped at 256 |
| per-repetition series | one additional event-per-repetition chain of `R` = 64 launches per cell, for the warmup and stationarity diagnostic |

## Nameplate constants and derived roofs

| Constant | Derivation | Value |
|---|---|---:|
| HBM nameplate ceiling | 1593 MHz x 2 x 5120 bits / 8 | 2,039.04 GB/s |
| BF16 dense tensor peak at 1410 MHz | 108 SM x 1410 MHz x 2048 FLOP/cycle | 311.869 TFLOP/s |
| BF16 dense tensor peak at 1275 MHz | 108 SM x 1275 MHz x 2048 FLOP/cycle | 282.010 TFLOP/s |
| clock ratio | 1410 / 1275 | 1.10588 |
| machine balance at nameplate | 311.869e12 / 2.03904e12 | 152.949 FLOP/B |
| L2 capacity | reported by the device | 41,943,040 B |

No measured rate may exceed the matching ceiling. A rate above a ceiling is
evidence of a harness defect, never of hardware.

`R_hbm` is the MEASURED HBM roof produced by lane 1 and is defined as the
maximum achieved bytes-per-second over every lane-1 cell at 256 MiB and above
in the boosted arm. Every memory-bound expectation in lanes 2 to 5 is stated
against `R_hbm`, never against nameplate. The nameplate value is used only as
a ceiling and to state the pre-run knee prediction.

For a GEMM with `C[M,N] = A[M,K] * B[K,N]` in BF16 the frozen work model is

```text
flops(M, N, K)  = 2 * M * N * K
bytes(M, N, K)  = 2 * (M*K + K*N + M*N)
t_flop(M, N, K) = flops / P(clock)
t_mem(M, N, K)  = bytes / R_hbm
t_roof          = max(t_flop, t_mem)
```

and the roofline knee in `M`, the row count where the two roofs cross, is

```text
M*(N, K) = P * K * N / (N * K * R - P * (K + N)).
```

At nameplate `R` and 1410 MHz `P` that gives the pre-run knee predictions in
the lane-2 table below. The asymptotic large-`N`, large-`K` limit of `M*` is
the machine balance itself, 152.9 rows, which is the "about 156 rows" the
campaign brief signs.

## Arms

Two arms differ only in what precedes and surrounds the measurement, never in
the kernel or the shape.

- **BOOSTED.** Three seconds of sustained 8192-cubed GEMM immediately precede
  the arm and the arm keeps the device continuously busy, so the SM clock is
  expected at 1410 MHz throughout.
- **BASE.** Each selected cell is preceded by 3 seconds of idle and its
  batches are recorded from the first one, so the leading batches are expected
  at 1275 MHz before the boost engages. Only the stationary 1275 MHz batches
  form the BASE constant.

The BASE arm covers a fixed subset: the four lane-1 cells at 1024 MiB, six
lane-2 cells (`M` in {1, 64, 256, 1024} at N=K=8192 and `M` in {32, 512} at
N=2048, K=1024), two lane-3 cells, two lane-4 cells and two lane-5 cells.

## Lane 1, measured HBM roof

Kernels `read` (reduces `S` bytes), `write` (fills `S` bytes), `copy` (reads
`S`, writes `S`, moves `2S`) and `triad` (reads `2S`, writes `S`, moves `3S`),
at `S` in {256, 512, 1024, 2048} MiB. This lane runs FIRST and its output
defines `R_hbm`.

Floor and ceiling stated before reading: no rate can exceed the 2,039.04 GB/s
nameplate ceiling, and the accepted
[A100 hardware envelope](../a100_hardware_envelope_v1/RESULTS.md) already
measured 1,770.5 GB/s read and 1,672.4 GB/s copy at 4 GiB on this hardware
class, so the bands below are informed by that accepted result and are
therefore low-risk confirmations rather than blind predictions. They are
scored, and their being low-risk is disclosed here rather than discovered
later.

- **E-1-1** At `S` = 2048 MiB the `read` rate is in [1700, 1937] GB/s.
- **E-1-2** At `S` = 2048 MiB the `write` rate is in [1700, 1937] GB/s.
- **E-1-3** At `S` = 2048 MiB the `copy` rate is in [1600, 1937] GB/s.
- **E-1-4** At `S` = 2048 MiB the `triad` rate is in [1600, 1937] GB/s.
- **E-1-5** The four rates at 1024 MiB and the four at 2048 MiB agree pairwise
  within 3 percent, so `R_hbm` is a size-independent roof over the scored
  range.
- **E-1-6** `R_hbm` lies in [1700, 1937] GB/s, that is 83.4 to 95.0 percent of
  nameplate.

The genuinely risky lane-1 relation is E-1-7, which is not entailed by any of
the above because the clock-invariance of a memory-limited kernel is a claim
about the mechanism, not about a magnitude:

- **E-1-7** For every lane-1 cell measured in both arms, the ratio of the
  1275 MHz constant to the 1410 MHz constant is in [0.98, 1.02]. HBM service
  is pinned to the memory clock, which never changes, so the SM clock must not
  move a memory-limited constant.

## Lane 2, dense GEMM family over a knee-anchored grid

Five shape families, each an `M` sweep at fixed `(N, K)`. G1 to G3 are the
granite-3.0-1b-a400m per-rank shapes used across the accepted studies
(`hidden_size` 1024, 16 query heads and 8 key/value heads of head size 64, MoE
intermediate size 512). G4 and G5 are the larger synthetic set.

| Family | Role | N | K | `M*` at nameplate |
|---|---|---:|---:|---:|
| G1 | granite QKV projection | 2048 | 1024 | 197.11 |
| G2 | granite output projection and expert gate/up | 1024 | 1024 | 218.10 |
| G3 | granite expert down projection | 1024 | 512 | 277.13 |
| G4 | synthetic 70B-class attention projection | 8192 | 8192 | 158.88 |
| G5 | synthetic mid-size projection | 4096 | 4096 | 165.29 |

The **grid** for each family is the sorted deduplicated union of
{1, 2, 4, 8, 16, 32, 64}, `round(M* * f)` for `f` in
{0.25, 0.40, 0.55, 0.70, 0.85, 1.00, 1.15, 1.30, 1.50, 1.80, 2.20, 3.00, 4.00},
and {1024, 2048, 4096, 8192}. The grid is deliberately dense between
`0.25 M*` and `4 M*` because that is where the transition lives.

The **held-out** set for each family is `round(M* * f)` for `f` in
{0.32, 0.62, 0.92, 1.07, 1.22, 1.65, 2.60}, minus any value already in the
grid. Held-out shapes sit INSIDE the suspect region on purpose: the TRAF-43
and TRAF-44 lesson is that a log-spaced grid with an excellent fit still
misses a transition that happens between its knots.

Fatal floor F5 covers the physical impossibility. The scored expectations are:

- **E-2-1** For every family, at `M` = 1 the measured time is within
  [0.9, 2.5] times `t_mem`, that is the small-`M` end is memory-limited and
  within a factor 2.5 of the measured HBM roof.
- **E-2-2** For every family, the measured time is nondecreasing in `M` up to
  a tolerance of one timer quantum plus 2 percent, over the whole grid.
- **E-2-3** For every family, the roofline efficiency
  `eff(M) = max(t_flop, t_mem) / t_measured` is monotone nondecreasing in `M`
  over the grid from `M` = 1 to `M` = `4 M*`, to a tolerance of 0.02.
- **E-2-4** For every family, the measured knee, defined as the smallest grid
  `M` whose measured time exceeds 1.5 times the median measured time over
  `M` in {1, 2, 4, 8}, lies in [`0.7 M*_meas`, `4 M*_meas`], where `M*_meas`
  recomputes the knee formula with the measured `R_hbm` and the arm's clock
  peak.
- **E-2-5** For every family, at `M` = `4 M*` rounded to the grid, the
  achieved rate is at least 45 percent of `P(clock)`.
- **E-2-6** At `M` = `N` = `K` = 8192 the achieved rate is at least 85 percent
  of `P(clock)` in the boosted arm.
- **E-2-7** For every family the efficiency at `M` = 1 is below 0.62 and the
  efficiency at the largest grid `M` is above 0.62, so no single flat
  efficiency constant can describe the family. This is the direct measurement
  of the surrogate `RooflineProvider(efficiency=0.7)` being replaced.
- **E-2-8** Every held-out shape is predicted by log-linear interpolation of
  the neighbouring grid entries, exactly the rule `ProfileTableProvider`
  applies, within 12 percent absolute percentage error for `M` outside
  [`0.5 M*`, `2 M*`] and within 25 percent inside it.
- **E-2-9** Across every held-out shape of every family, the median absolute
  percentage error of the interpolated surface is below 10 percent and the
  p95 is below 20 percent, which are COMP-1's registered held-out bars.
- **E-2-10** For the lane-2 cells measured in both arms, the ratio of the
  1275 MHz constant to the 1410 MHz constant is in [1.06, 1.13] whenever the
  cell's measured roofline regime is compute (`t_flop > t_mem`), and in
  [0.98, 1.02] whenever it is memory. The interval [1.06, 1.13] brackets the
  clock ratio 1.10588 and excludes 1.
- **E-2-11** The roofline efficiency of a cell is the same in both arms to
  within 0.03 absolute, for every cell measured in both. This is the claim
  that makes a clock-conditioned constant transferable: the CONSTANT depends
  on the clock, the EFFICIENCY SURFACE does not.

Entailment: E-2-10 and E-2-11 are not equivalent. E-2-10 constrains the ratio
of two constants; E-2-11 constrains the ratio of each constant to its own
clock-dependent roof, and for a memory-limited cell the roof does not move
with the clock at all, so E-2-11 is a different statement there. E-2-3 does
not entail E-2-4: a monotone efficiency curve can put its knee anywhere.
E-2-9 aggregates the same residuals E-2-8 tests per cell, so E-2-9 is scored
and E-2-8 is scored, but they are declared as one family of two relations
rather than two independent risks, and neither is counted twice.

## Lane 3, attention prefill and decode

**Prefill.** The score and value passes as batched GEMMs, exactly the
`attn_score` family of `step_kernels`. Per head: `QK^T` is
`M` = `N` = `S`, `K` = `D`; `PV` is `M` = `S`, `N` = `D`, `K` = `S`. Batch
count is the head count. Two geometries, granite (16 heads, `D` = 64) and
synthetic (64 heads, `D` = 128), at `S` in {128, 256, 512, 1024, 2048, 4096}.

- **E-3-1** For both geometries the prefill time grows superlinearly in `S`:
  the time ratio between `S` = 4096 and `S` = 2048 is in [3.2, 4.4],
  bracketing the quadratic 4.
- **E-3-2** At `S` = 4096 the synthetic geometry reaches at least 40 percent
  of `P(clock)` on the combined score and value flops.

**Decode.** One kernel per `(batch, query head)` streaming the key and value
caches with an online-softmax accumulation, BF16 cache, granite geometry (16
query heads, 8 key/value heads, `D` = 64). Batch `B` in {1, 4, 16, 64, 256}
and cache length `L` in {128, 512, 2048, 8192}, full cross product.
KV bytes are `2 * B * H_kv * L * D * 2` and are the compulsory traffic.

- **E-3-3** For every decode cell with KV bytes at least 160 MiB, the achieved
  KV bandwidth `kv_bytes / t` is in [0.55, 1.00] times `R_hbm`.
- **E-3-4** Decode time is linear in `L` at fixed `B` for `B` >= 16: for each
  such `B` the ratio between consecutive `L` values, which quadruple, is in
  [3.4, 4.6].
- **E-3-5** Decode time is linear in `B` at fixed `L` = 8192 for `B` >= 16:
  the ratio between `B` = 256 and `B` = 64 is in [3.4, 4.6].
- **E-3-6** Every decode cell is memory-limited under the frozen work model,
  that is `t_mem > t_flop` at the measured `R_hbm`, for the whole grid.

Entailment: E-3-6 is very nearly forced by the geometry, since decode
attention performs 4 flops per KV byte read while the machine balance is above
150, so it is declared a GUARD (G6) and NOT scored. E-3-4 and E-3-5 are
separate risks because a per-`(batch, head)` kernel can saturate in `B`
without saturating in `L`.

## Lane 4, MoE expert GEMM at captured expert loads

The captured population is the granite cell of the accepted
[token ownership study](../token_ownership_v1/RESULTS.md): prefill step 0, 54
scheduled tokens, 24 MoE layers, top-8 routing over 32 experts. Balanced
routing puts 54 times 8 divided by 32, that is 13.5 rows, on each expert. The
frozen expert-load grid spans the reachable range from one token to one
expert taking every token: `M_e` in {1, 2, 4, 7, 11, 14, 18, 27, 54}. Shapes
are the granite expert gate/up (`N` = 1024, `K` = 1024, family G2) and expert
down (`N` = 1024, `K` = 512, family G3).

The campaign brief signs the expectation that the A100 knee sits near
peak-over-bandwidth, about 156 rows in FP16, so every captured expert load is
far below it and every captured cell scores memory-bound.

- **E-4-1** Every cell of the frozen expert-load grid is memory-limited under
  the frozen work model at the measured `R_hbm`, that is `t_mem > t_flop`.
- **E-4-2** Over the expert-load grid at fixed shape, the measured time varies
  by at most a factor 1.6 between `M_e` = 1 and `M_e` = 54, while the flops
  grow by a factor 54. A memory-limited plateau, not a compute ramp.
- **E-4-3** At `M_e` = 54 the achieved rate is below 25 percent of `P(clock)`
  for both shapes.
- **E-4-4** The measured time at `M_e` = 14, the grid point nearest the
  balanced 13.5-row captured load, is within [1.0, 3.0] times `t_mem` for both
  shapes.

Entailment: E-4-1 is a claim about where the captured load sits relative to
the measured knee. It is NOT forced by construction, because `M*` is computed
from the measured `R_hbm`, which lane 1 has not yet produced at freeze time,
and a sufficiently low `R_hbm` would move the knee below 54. It is therefore
scored. E-4-3 is close to entailed by E-4-1 but not identical: a memory-bound
cell can still reach a high flop rate if the shape has enough reuse, so the
25 percent bound is a separate magnitude claim.

## Lane 5, elementwise and normalization

Kernels `scale` (read and write one buffer), `add` (read two, write one) and
`rmsnorm` (two passes over the activation plus one weight vector read; bytes
counted as two activation reads, one activation write and one weight read),
at buffer sizes {4, 16, 64, 256} MiB, each in the warm and rotated variant.

- **E-5-1** At 256 MiB every lane-5 kernel achieves at least 80 percent of
  `R_hbm` on its own compulsory byte count.
- **E-5-2** At 64 MiB every lane-5 kernel achieves at least 80 percent of
  `R_hbm`.
- **E-5-3** For every lane-5 cell, the warm and rotated constants agree within
  6 percent at 64 MiB and above. The discovery pass saw them agree to the
  timer quantum, so this is a low-risk confirmation and is disclosed as one.
- **E-5-4** At 4 MiB, whose 4 MiB working set is inside the 40 MiB L2, the
  warm `scale` constant is NOT more than 1.15 times faster than its own
  compulsory-bytes-over-`R_hbm` time. The a100 hardware envelope predicted an
  L2 signature at a comparable size and was refuted by its own fixed cost;
  this study predicts the same refutation for the opposite reason, that a
  small streaming kernel on this device does not convert L2 residency into
  bandwidth.

## Efficiency surfaces, the published product

For every measured cell the study publishes, per clock state:

```text
eff_roofline = max(t_flop, t_mem) / t_measured
eff_compute  = t_flop / t_measured
eff_memory   = t_mem  / t_measured
```

with `t_mem` taken against the measured `R_hbm` and `t_flop` against the
arm's `P(clock)`. The per-family surface is the ordered set of
`(shape, eff_roofline, uncertainty)` triples, where the uncertainty is the
larger of the cell's own batch-mean relative spread and the family's held-out
p95 interpolation error. The compact `simllm-profile-table-v1` artifact
carries the boosted-arm constants with that uncertainty, keyed by family name,
shape config and gpu name `a100`, and is loadable by `ProfileTableProvider`
without modification.

## Fatal guards

A violated fatal guard voids the run. The behavioral score becomes
uninterpretable, the study is reported void with findings, and every task it
touches stays open. These are never reported as a fraction. No guard in this
study is declared survivable.

- **G1** The job sees exactly one `NVIDIA A100-SXM4-80GB`, MIG disabled, ECC
  enabled, and no foreign compute process occupies it at the start or the end.
- **G2** Every scored constant is computed only from clock-stationary batches
  of a single clock state, and every scored cell has at least 8 such batches
  in the state it is reported for. A cell that does not is VOID for scoring
  and is reported as void, not as a failure.
- **G3** The memory clock reads exactly 1593 MHz and the NVML throttle reason
  word reads zero on both sides of every scored batch.
- **G4** No measured rate exceeds its nameplate ceiling: 2,039.04 GB/s for any
  lane-1 cell, and `P(clock)` for any GEMM cell at that batch's clock state.
- **G5** No measured kernel time falls below its own compulsory-traffic floor
  `max(t_flop, (distinct_bytes - L2_bytes) / R_hbm)` where `distinct_bytes` is
  the distinct bytes the kernel touches in one repetition and `L2_bytes` is
  41,943,040. Crediting the full L2 makes this floor exact rather than
  approximately true for cache-resident cells.
- **G6** Every lane-3 decode cell and every lane-4 expert cell satisfies
  `t_mem > t_flop` under the frozen work model. This is unscored because the
  geometry very nearly forces it; E-4-1 makes the scored version of the claim
  only for lane 4, where the captured load is the point.
- **G7** Every reported cell used exactly the frozen warmup discard `K`, the
  frozen batch count and a batch size satisfying the frozen 200 microsecond
  rule, and every timed batch is bracketed by CUDA events on the measured
  stream.
- **G8** Every GEMM cell returns a numerically correct result on a fixed
  element sample, checked once per cell outside the timed region against a
  reference computed from the known constant operands.
- **G9** Every scored cell's per-repetition coefficient of variation, over the
  clock-stationary part of its 64-repetition diagnostic chain, is at most
  `0.02 + q / (sqrt(12) * median)`, the sum of a 2 percent service ceiling and
  the standard deviation of uniform quantization at the measured 1024 ns
  quantum.
- **G10** Every scored cell's batch-mean coefficient of variation within its
  reported clock state is at most 0.02.
- **G11** The batch-mean constant and the per-repetition mean of the same cell
  agree within 3 percent, so the event-per-repetition instrumentation used by
  the diagnostic chain and by stage 2 does not itself move the constant.
- **G12** Both lane-1 arms and both lane-2 arms observe the same GPU UUID, and
  the study records the UUID, driver, toolkit, source digest and binary digest
  of the exact executed harness.

G9 and G10 are the frozen per-cell stability ceilings the campaign brief asks
for. They are fatal rather than scored because they assert the precondition
under which a "deterministic constant with no tail" is the right description
of the cell, not a predicted magnitude. A cell that violates them is void, and
its void status is reported plainly.

## Scoring

The scored behavioral denominator is 31:

| Lane | Scored expectations | Count |
|---|---|---:|
| 1, HBM roof | E-1-1 to E-1-7 | 7 |
| 2, dense GEMM | E-2-1 to E-2-11 | 11 |
| 3, attention | E-3-1 to E-3-5 | 5 |
| 4, MoE expert GEMM | E-4-1 to E-4-4 | 4 |
| 5, elementwise | E-5-1 to E-5-4 | 4 |

They are one evidence class: measured silicon kernel times checked against
magnitudes and relations frozen before the run. Every expectation that
quantifies over cells passes only if it holds for every cell in its scope, and
a partial pass is reported as a fail with the failing cells named.

Fatal guards G1 to G12 are unscored and never enter that denominator. E-3-6
was moved into G6 and is not counted twice. The raw per-repetition series, the
rotated-state constants, the warm-versus-rotated deltas, the clock telemetry
and the BASE-arm coverage are recorded and reported as structural facts
without entering any denominator.

Low-risk disclosure: E-1-1 to E-1-4 and E-5-3 are informed by already accepted
first-party measurements on this hardware class and by the discovery pass, and
are confirmations rather than blind predictions. They stay in the denominator
because they are stated before this run and can still fail, but a reader should
weight them accordingly. The genuinely risky relations are E-1-7, E-2-3,
E-2-4, E-2-7, E-2-8, E-2-9, E-2-10, E-2-11, E-3-1, E-3-3, E-3-4, E-3-5, E-4-1,
E-4-2, E-4-4 and E-5-4.

## What this study will not claim

- It does not close COMP-1. It captures no production framework kernel,
  replays no SASS, calibrates no Accel-Sim configuration and validates no
  framework-kernel held-out matrix. It supplies the target-architecture
  efficiency surface that COMP-1's first blocker names, and nothing more.
- It does not close COMP-5. Clocks are not locked, because locking them is
  denied on this allocation. The controlled-environment form of the stability
  bar therefore cannot be met here, and this study substitutes a
  clock-conditioned form and says so.
- It does not measure launch cost. Stage 2 does.
- It measures one A100 SXM4 80 GB on one node. Nothing here transfers to
  H100, GH200, B100 or B200, and nothing here describes a multi-GPU or
  cross-node path.
- Its constants are microbenchmark constants for standalone kernels. Whether a
  kernel executing inside a CUDA graph carries the same constant is the
  stage-2 falsifier, and until stage 2 reports, no constant here may be
  assumed launch-mode independent.
