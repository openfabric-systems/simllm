# GH200 hardware envelope v1 expectations

## Freeze scope and chronology

This is the expectations-only record for the Grace Hopper measurement, the
second architecture in the hardware envelope series. It is committed before
the ported harness is committed, before any timed GH200 kernel, collective or
transfer runs, and before any result-producing GH200 job is submitted. No
GH200 number may be written back into this file.

The sweep is deliberately identical to the
[A100 hardware envelope](../a100_hardware_envelope_v1/expectations.md): the
same payload sizes, the same GEMM shapes, the same iteration counts, the same
collectives at the same widths. Only the nameplate floors change, and they are
derived here from the GH200 inventory before the run. Holding the sweep fixed
is the point, because the comparison between the two architectures is only
meaningful if the instrument did not move.

**This freeze is informed by A100 results and says so.** Three A100
expectations were refuted, and repeating them unchanged would be dishonest
theatre. Each is replaced by a corrected form stated below, and one of them,
the mid-range optimism of the two-parameter collective model, becomes the
central forward prediction this study exists to test on a different NVLink
generation. Prior evidence legitimately shapes a new prediction; the rule this
must not violate is using *this study's own* results, and none exist.

## Registry motivation

The A100 study left CORE-13 with first-party intra-node evidence on exactly
one architecture and one link topology. TRAF-43 was registered on the strength
of a finding from a single machine: that a single-slope latency and bandwidth
model is optimistic by up to 50.8 percent across the payload decade where bus
bandwidth is still climbing. A finding from one node is a hypothesis. If the
same shape appears on an NVLink4 mesh with a different link count, a different
channel count and a different reduction engine, then TRAF-43 is an
architecture-independent defect in the model form rather than an A100 quirk,
and its priority is settled.

The A100 study also found `DEFAULT_NVLINK_BANDWIDTH_BYTES_PER_SECOND` of
450 GB/s to be 1.598 times the measured per-GPU egress. On this node the
egress ceiling is 478.125 GB/s, so the same constant should be roughly correct
here. A surrogate that is wrong by 1.6 times on one machine and right on
another is not a bandwidth: it is a machine identity, and that is worth
recording as a scored prediction rather than an aside.

## Pre-freeze facts

Slurm job `195463` recorded the node inventory before this freeze. It timed
nothing.

| Property | Observed value |
|---|---|
| GPU | 4 x NVIDIA GH200 120GB, compute capability 9.0 |
| driver | 590.48.01, driver API 13010, MIG disabled, ECC enabled |
| persistence and compute mode | persistence enabled, compute mode Default |
| SM count and clock | 132 SMs, 1980 MHz maximum SM clock |
| memory | 102,005,473,280 B per GPU, 6144-bit bus, 2619 MHz, 60 MiB L2 |
| power | 900 W limit per superchip module |
| NVLink topology | `NV6` between every ordered pair, 18 links x 26.5625 GB/s |
| host link | NVLink-C2C, not PCIe; GPU memory exposed as NUMA nodes 4, 12, 20, 28 |
| host | 4 Grace sockets, aarch64, 288 CPUs, 4 NUMA domains, one GPU each |
| toolchain | `cuda/12.8.1` and `cuda/12.9.1` on this architecture; no site NCCL |
| peer access | enabled for all 12 ordered pairs, performance rank 1, native atomics |

Two differences from the A100 node change what the harness may assume. The
architecture is aarch64, so the staged NCCL must be the aarch64 build, and
`cuda/12.2.2` does not exist here. The host link is NVLink-C2C rather than
PCIe generation 4, so lane A4 measures a fundamentally different path.

## Frozen substrate

| Item | Frozen value |
|---|---|
| cluster and partition | `gmerlin7`, `gh-hourly`, account `merlin` |
| CUDA toolchain | `cuda/12.9.1`, `nvcc` release 12.9 V12.9.86, `-arch=sm_90` |
| NCCL | `nvidia-nccl-cu12` 2.31.2 aarch64, `+cuda12.9` |
| GEMM library | cuBLAS from `cuda/12.9.1`, `cublasGemmEx` |
| harness | the A100 lanes of commit `8509a2c` with architecture dependence removed |
| timing | CUDA events on the measured stream, host wall clock only for launch cost |
| bandwidth unit | decimal, 1 GB/s is 1,000,000,000 B/s |

The harness changes are frozen here, before they are committed, so the diff is
part of the record rather than a later explanation. There are four: the HBM and
tensor-core peaks are derived from device properties plus a table of BF16 dense
FLOP per SM per cycle, 2048 on Ampere and 4096 on Hopper, and are emitted in
the result; the launch grid is eight blocks per SM instead of a hardcoded 864;
a missing peer link is recorded and skipped rather than fatal; and the study
name changes. No timing path, iteration count, event placement or sweep value
changes. On an A100 the derivation reproduces the 2,039.04 GB/s and
311.87 TFLOP/s the A100 study hardcoded.

Clocks are left at the site default and observed around every timed block.

## Frozen allocation envelope

| Resource | Single-card lane | Multi-card lane |
|---|---:|---:|
| nodes and tasks | 1 node, 1 task | 1 node, 1 task |
| GPUs | 1 `gh200` | 4 `gh200` |
| CPUs | 16 | 64 |
| host memory | 64 GiB | 256 GiB |
| wall limit | 30 minutes | 50 minutes |
| expected runtime | 5 to 20 minutes | 10 to 40 minutes |

The `gh` partitions were fully allocated at freeze time, so both lanes are
expected to queue. Neither requests exclusive nodes, arrays, requeue or network
installation, and no computation runs on a login node.

## Nameplate constants and derived floors

| Constant | Derivation | Value |
|---|---|---:|
| HBM peak bandwidth | 2619 MHz x 2 x 6144 bits / 8 | 4,022.78 GB/s |
| BF16 dense tensor peak | 132 SM x 1980 MHz x 4096 FLOP/cycle | 1,070.53 TFLOP/s |
| machine balance, BF16 | 1070.53e12 / 4022.78e9 | 266.12 FLOP/B |
| NVLink per ordered pair | 6 bonded links x 26.5625 GB/s | 159.375 GB/s per direction |
| NVLink per GPU egress | 18 links x 26.5625 GB/s | 478.125 GB/s per direction |
| NVLink-C2C host link | GH200 specification | 450 GB/s per direction |

The tensor peak is derived from the clock this device reports, 1980 MHz, not
from a vendor reference clock. NVIDIA quotes 989.4 TFLOP/s dense BF16 for an
H100 SXM at its own reference clock; the number above is higher because the
reported boost clock here is higher. Efficiency against a clock-derived peak
is only meaningful if the clock held, so this study also scores efficiency
against the peak recomputed at the SM clock actually observed during the
measured block.

No measured rate may exceed the matching ceiling. A rate above its ceiling is
evidence of a harness defect and is fatal.

## Lane A, single card

The sweep is identical to the A100 lane A: HBM sizes 1 MiB through 4096 MiB in
powers of two with three warmup and ten timed iterations; the decode GEMM sweep
at `N` = `K` = 8192 over `M` in {1 ... 8192}; the square sweep at 1024 through
16384; the three launch measurements at 200,000 pipelined launches, 2,000
roundtrips and a 1,000-node graph replayed 200 times; and a 256 MiB pinned
host transfer in each direction.

### A1 HBM bandwidth

- **E-A1-1** At 4096 MiB the read bandwidth lies in [2800, 4022.78] GB/s.
- **E-A1-2** At 4096 MiB the write bandwidth lies in [2000, 4022.78] GB/s.
- **E-A1-3** At 4096 MiB the copy bandwidth lies in [2400, 4022.78] GB/s.
- **E-A1-4** The copy values at 1024, 2048 and 4096 MiB agree within 5 percent
  of their median.
- **E-A1-5** *(corrected form of the refuted A100 E-A1-5)* The small end of the
  sweep is launch-floor dominated, not an L2 measurement. The measured times at
  1 MiB and 2 MiB agree within 25 percent of each other for the read kernel,
  despite a factor two in payload, and both lie within a factor two of the
  `roundtrip_us` this same lane measures in A3. The A100 study predicted an L2
  signature at 8 MiB and was refuted because a roughly 6 microsecond fixed cost
  swamped a 3 microsecond transfer; the corrected claim is about that floor
  existing, which is what the data actually showed.

### A2 GEMM roofline crossover

Two-sided floors use the derived peaks above:

```
bytes(M, N, K) = 2 * (M*K + K*N + M*N)
t_mem  = bytes / 4.02278e12
t_flop = 2*M*N*K / 1.07053e15
```

At `M` = 1, `N` = `K` = 8192 the memory floor is 33.37 us and the compute floor
is 0.125 us, so the point is memory-bound by a factor 266. Setting the two
equal puts the ideal crossover at `M` = 284.6, against 158.9 on the A100: a
faster tensor core relative to memory pushes the knee to a wider batch.

- **E-A2-1** *(corrected form of the refuted A100 E-A2-1)* At `M` = 1 the
  achieved rate is below 5 TFLOP/s. The A100 version wrongly extended this to
  the whole plateau; in a memory-bound plateau the time is flat, so the rate
  necessarily grows with `M`, and the corrected claim is the single-point one.
- **E-A2-2** For `M` <= 64 the measured times form a plateau whose maximum
  divided by minimum is at most 1.8. The bound is looser than the A100's 1.6
  because the same fixed launch floor is a larger share of a 33 us memory floor
  than of a 66 us one.
- **E-A2-3** For `M` <= 64 every measured time lies in [`t_mem`, 3 x `t_mem`].
- **E-A2-4** The measured crossover, the smallest `M` whose time exceeds 1.5
  times the plateau median, lies in [256, 2048], bracketing the ideal 284.6 and
  above the A100's measured 256.
- **E-A2-5** For `M` >= 2048 the achieved rate is at least 450 TFLOP/s.
- **E-A2-6** Between `M` = 4096 and `M` = 8192 the time ratio lies in
  [1.7, 2.3].
- **E-A2-7** At the square 8192 point, the achieved rate divided by the peak
  recomputed at the SM clock observed during that block lies in [0.75, 1.00].
  This is the throttle-robust form of the A100's E-A2-7.

### A3 kernel launch cost

- **E-A3-1** `pipelined_period_us` lies in [0.5, 10.0].
- **E-A3-2** `roundtrip_us` lies in [3.0, 40.0].
- **E-A3-3** `roundtrip_us` is greater than `pipelined_period_us`.
- **E-A3-4** `graph_period_us` is less than `pipelined_period_us`.
- **E-A3-5** `pipelined_period_us` is within a factor 2.5 of the A100's
  measured 1.806 us in either direction. The launch path is driver and host
  work, and the host here is an aarch64 Grace rather than an x86 EPYC, so the
  question is whether that changes the constant materially.

### A4 host link over NVLink-C2C

This is the lane where the two architectures should differ most. The A100
measured 26.78 and 26.19 GB/s over PCIe generation 4 by 16.

- **E-A4-1** Host-to-device bandwidth lies in [150, 450] GB/s.
- **E-A4-2** Device-to-host bandwidth lies in [150, 450] GB/s.
- **E-A4-3** Both directions exceed the A100's PCIe measurements by at least a
  factor 5.

## Lane B, four cards over NVLink

### B1 peer bandwidth matrix

- **E-B1-1** Every unidirectional ordered-pair bandwidth lies in
  [120, 159.375] GB/s.
- **E-B1-2** The 12 unidirectional values agree within 10 percent of their
  median.
- **E-B1-3** The bidirectional sum for the pair 0 and 1 lies in
  [200, 318.75] GB/s.
- **E-B1-4** The device 0 fan-out aggregate lies in [330, 478.125] GB/s.
- **E-B1-5** The fan-out aggregate exceeds the single-pair value by at least a
  factor 2.4.

### B2 NCCL collectives

Same operations, widths, sizes and iteration counts as the A100 lane.

- **E-B2-1** At 1 GiB the all-reduce bus bandwidth lies in [90, 159.375] GB/s
  at width 2.
- **E-B2-2** At 1 GiB the all-reduce bus bandwidth lies in [200, 478.125] GB/s
  at width 4.
- **E-B2-3** The width-4 bus bandwidth at 1 GiB exceeds the width-2 value by at
  least a factor 1.4, for the same structural reason as on the A100: a two-rank
  ring reaches one pair's six links while a four-rank ring set reaches all
  eighteen.
- **E-B2-4** At 1 GiB and fixed width, the bus bandwidths of all-reduce,
  all-gather and reduce-scatter agree within 30 percent of their mean.
- **E-B2-5** The measured 8 B all-reduce time lies in [2, 40] us at both widths
  and is larger at width 4 than at width 2.
- **E-B2-6** The all-reduce time ratio between 1 GiB and 512 MiB lies in
  [1.8, 2.2] at both widths.
- **E-B2-7** At 256 MiB and width 4, the all-reduce time lies between 0.7 and
  1.3 times the sum of the reduce-scatter and all-gather times.
- **E-B2-8** *(the forward prediction this study exists for)* The A100
  mid-range optimism reproduces here. Anchoring `alpha` at the measured 8 B
  time and `beta` at the 1 GiB algorithm bandwidth, the two-parameter model has
  a worst signed error more negative than -20 percent at both widths, and that
  worst point falls at a payload between 256 KiB and 32 MiB inclusive.
- **E-B2-9** The half-bandwidth payload, where bus bandwidth first reaches half
  its 1 GiB value, lies in [1, 32] MiB at both widths and is larger at width 4
  than at width 2, as it was on the A100 at 2.45 and 8.24 MiB.
- **E-B2-10** *(the falsifier for E-B2-8)* A single ordinary least squares fit
  of `t = alpha + S / beta` over 1 MiB to 1 GiB again returns an `alpha` more
  than three times the measured 8 B all-reduce time at width 2, while reporting
  an R-squared of at least 0.99. If this fails while E-B2-8 passes, the A100
  intercept artifact was a fitting-window accident rather than a property of
  the model form.

### B3 collective under compute contention

At width 4 and 256 MiB against a BF16 8192-cubed GEMM loop, 20 collective and
8 GEMM iterations, as on the A100.

- **E-B3-1** The collective grows by at least 1.02 times and the GEMM by at
  least 1.02 times.
- **E-B3-2** The collective grows by at most 3.0 times and the GEMM by at most
  2.0 times.
- **E-B3-3** The makespan is at least the larger of the two alone totals and at
  most 0.95 times their sum.

## Cross-architecture comparison

- **E-C-1** The measured device-0 fan-out egress here is within a factor 1.5 of
  `DEFAULT_NVLINK_BANDWIDTH_BYTES_PER_SECOND`, currently 450 GB/s, in either
  direction. The same constant was 1.598 times the A100 measurement, so it
  encodes a Hopper-class machine and is not a portable intra-node rate.
- **E-C-2** The width-4 all-reduce bus bandwidth at 1 GiB exceeds the A100's
  measured 212.89 GB/s by at least a factor 1.3.
- **E-C-3** The ratio of measured width-4 bus bandwidth to the per-GPU egress
  ceiling lies within 15 percentage points of the A100's 71.0 percent. Ring
  efficiency should be a property of the NCCL kernel rather than of the link
  generation.
- **E-C-4** The 8 B all-reduce latency floor at width 4 lies within a factor 2
  of the A100's 12.95 us in either direction. Small-message collective latency
  is dominated by kernel launch and synchronization rather than by wire rate,
  so a 1.6 times faster link should not move it much.

## Fatal guards

- **F1** The single-card lane sees exactly one GH200 and the multi-card lane
  exactly four, every one with MIG disabled and ECC enabled.
- **F2** No measured rate exceeds its nameplate ceiling: 4,022.78 GB/s for any
  HBM-resident point at or above 256 MiB, 159.375 GB/s per ordered NVLink pair
  direction, 318.75 GB/s for a bidirectional pair sum, 478.125 GB/s for
  per-GPU egress and for any width-4 bus bandwidth, 159.375 GB/s for any
  width-2 bus bandwidth, and 450 GB/s on the host link.
- **F3** No measured GEMM time falls below its own two-sided floor.
- **F4** Every timed NCCL all-reduce returns the arithmetically correct sum on
  every rank at every size.
- **F5** No foreign compute process occupies an allocated GPU at the start of
  either lane.
- **F6** Every reported point carries exactly the frozen warmup and timed
  iteration counts, and every timed block is bracketed by CUDA events on the
  measured stream.
- **F7** Both lanes record the SM clock, memory clock, power draw and
  temperature immediately before and immediately after every timed block.
- **F8** The harness reports a BF16 FLOP-per-SM-per-cycle constant for compute
  capability 9.0 rather than refusing the architecture, and the emitted derived
  peaks equal the values tabulated above.

## Scoring

The scored behavioral denominator is 42: five in A1, seven in A2, five in A3,
three in A4, five in B1, ten in B2, three in B3 and four in C. Lane A carries
20, lane B carries 18, and the four comparison expectations read lane B values
and previously published A100 values only. Fatal guards F1 through F8 are
unscored.

Two expectations are corrected restatements of refuted A100 ones and are marked
in place: E-A1-5 and E-A2-1. E-A2-7 is a throttle-robust reformulation of an
A100 expectation that passed, changed because a 900 W superchip is more likely
to clock down than a 500 W board. E-B2-8 and E-B2-10 carry the refuted A100
E-B2-6 forward as a paired prediction and falsifier. All are scored as ordinary
expectations here, and the A100 record keeps its own failures unchanged.

## What this study does not claim

- It does not close COMP-1, COMP-5, CORE-13, TRAF-43 or TRAF-44. It adds a
  second architecture to the evidence each of them needs.
- It runs no framework, loads no model and reports no TTFT or TPOT.
- It measures one node of one GH200 variant, the 120GB HBM3e part in a 4-GPU
  `NV6` mesh without NVSwitch. It does not describe an H100 PCIe or SXM board,
  a GH200 96GB HBM3 part, an NVL32 rack, or any cross-node path.
