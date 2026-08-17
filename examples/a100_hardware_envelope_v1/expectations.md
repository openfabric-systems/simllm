# A100 hardware envelope v1 expectations

## Freeze scope and chronology

This is the expectations-only record for the first first-party measurement of
the Merlin A100 node's compute and NVLink envelope. It is committed before the
measurement harness exists, before any timed kernel, collective or transfer
runs in this study, and before any result-producing Slurm job is submitted. No
number produced by this study may be written back into this file.

The study has two lanes and they run as two separate jobs, in this order:

1. a single-card lane on one allocated A100, which measures HBM bandwidth,
   dense GEMM throughput across the roofline crossover, kernel launch cost and
   the host PCIe link;
2. a multi-card lane on four A100s inside one node, which measures the NVLink
   peer bandwidth matrix and the NCCL collective latency and bandwidth model at
   participant widths two and four, including one compute-contention cell.

The single-card lane is a substrate for the compute model. The multi-card lane
is a substrate for the intra-node collective model. Neither lane runs a
framework, a model or a serving engine, so neither lane can close a task whose
acceptance names framework kernels or an end-to-end metric.

## Registry motivation

Three open tasks currently price intra-node GPU behavior from evidence that was
not measured on hardware this project can reach.

CORE-13 replaces the flat per-endpoint intra-node NVLink-class serializer with
a calibrated compute-owned NCCL and NVLink service. Its acceptance names
captured NCCL traces plus NVLink byte and rate observations, varied payload and
varied participant count. No such first-party capture exists.

The collective fixed-cost envelope in `simllm/traffic/collective_latency.py`
carries `b200-nccl-2.27-local-v1`, whose intercepts and 70,027,079,100 B/s
bandwidth term come from a published third-party DGX B200 nccl-tests capture.
That is the only calibrated intra-node collective profile in the repository and
it describes hardware the project cannot observe.

COMP-1 and COMP-5 need target-architecture compute evidence. The
[A100 environment qualification](../a100_environment_qualification_v1/RESULTS.md)
established that one Merlin A100 supports profiling, but it deliberately
measured no bandwidth, no throughput and no launch constant. COMP-1 explicitly
forbids transferring the Turing launch constants and states that the fixed-step
launch, host-delay and queueing terms are unmeasured on the production target.

This study supplies the hardware envelope those tasks consume. It does not
consume it itself.

## Pre-freeze facts

These were observed by inventory-only and capability-only jobs before this
freeze. They are context and frozen inputs, not scored outcomes. Neither job
timed a kernel, a transfer or a collective.

Slurm job `195456` on `gpu105` recorded the node inventory:

| Property | Observed value |
|---|---|
| GPU | 4 x NVIDIA A100-SXM4-80GB, compute capability 8.0 |
| driver | 565.57.01, driver API 12070, MIG disabled, ECC enabled |
| persistence and compute mode | persistence enabled, compute mode Default |
| SM count and clock | 108 SMs, 1410 MHz maximum SM clock |
| memory | 85,097,971,712 B per GPU, 5120-bit bus, 1593 MHz, 40 MiB L2 |
| power | 500 W limit on every GPU |
| NVLink topology | `NV4` between every ordered pair, 12 links x 25 GB/s per GPU |
| host link | PCIe generation 4, width 16x, current and maximum |
| host | AMD EPYC 7713, 128 CPUs, 4 NUMA domains, one GPU affine to each |
| toolchain | `cuda/12.2.2`, `cuda/12.8.1`, `cuda/12.9.1`; no site NCCL |
| peer access | enabled for all 12 ordered pairs, performance rank 0, native atomics |

Slurm job `195459` recorded NCCL capability. Both staged builds initialized a
communicator and returned a correct all-reduce at widths two and four:
`nvidia-nccl-cu12` 2.31.2 built against CUDA 12.9, and 2.21.5 built against
CUDA 12.4. On the four-device communicator NCCL reported that NVLS multicast is
unavailable on every device and built at least ten rings, so the selected
algorithm family on this node is ring, not NVLS.

## Frozen substrate

| Item | Frozen value |
|---|---|
| cluster and partition | `gmerlin7`, `a100-hourly`, account `merlin` |
| CUDA toolchain | `cuda/12.2.2`, `nvcc` release 12.2 V12.2.140, `-arch=sm_80` |
| NCCL | `nvidia-nccl-cu12` 2.31.2, reported version 23102, `+cuda12.9` |
| GEMM library | cuBLAS from `cuda/12.2.2`, `cublasGemmEx` |
| timing | CUDA events on the measured stream, host wall clock only for launch cost |
| bandwidth unit | decimal, 1 GB/s is 1,000,000,000 B/s |

Clocks are left at the site default. This study observes the SM clock, memory
clock and power draw around every timed block and reports them. It does not
lock application clocks, so it does not supply the controlled-clock evidence
COMP-5 still requires.

## Frozen allocation envelope

| Resource | Single-card lane | Multi-card lane |
|---|---:|---:|
| nodes and tasks | 1 node, 1 task | 1 node, 1 task |
| GPUs | 1 `nvidia_a100-sxm4-80gb` | 4 `nvidia_a100-sxm4-80gb` |
| CPUs | 8 | 32 |
| host memory | 64 GiB | 256 GiB |
| wall limit | 30 minutes | 50 minutes |
| expected runtime | 8 to 20 minutes | 15 to 40 minutes |
| device memory ceiling | 40 GiB | 12 GiB per GPU |

Neither lane requests exclusive nodes, uses a job array, requeues, installs
anything over the network, or performs computation on a login node. All staged
inputs were placed before submission.

## Nameplate constants and derived floors

Every constant below follows from the frozen inventory. They are the bounds the
measured values are checked against, and they are stated before any measurement.

| Constant | Derivation | Value |
|---|---|---:|
| HBM peak bandwidth | 1593 MHz x 2 x 5120 bits / 8 | 2,039.04 GB/s |
| BF16 dense tensor peak | 108 SM x 1410 MHz x 2048 FLOP/cycle | 311.87 TFLOP/s |
| TF32 dense tensor peak | half the BF16 rate | 155.93 TFLOP/s |
| machine balance, BF16 | 311.87e12 / 2,039.04e9 | 152.95 FLOP/B |
| NVLink per ordered pair | 4 bonded links x 25 GB/s | 100 GB/s per direction |
| NVLink per GPU egress | 12 links x 25 GB/s | 300 GB/s per direction |
| PCIe host link | generation 4, width 16 | 31.5 GB/s per direction |

No measured rate may exceed the matching ceiling. A rate above its ceiling is
evidence of a harness defect, never of hardware, and is fatal.

## Lane A, single card

### A1 HBM bandwidth

Three grid-stride kernels over a device buffer of `S` bytes: a read kernel that
reduces the buffer, a write kernel that fills it, and a copy kernel that reads
one buffer and writes another. Frozen sizes are `S` in
{1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096} MiB. Three warmup
iterations then ten timed iterations per point; the reported value is the
median. The copy kernel moves `2S` bytes and its bandwidth is `2S / t`.

- **E-A1-1** At `S` = 4096 MiB the read bandwidth lies in [1400, 2039.04] GB/s.
- **E-A1-2** At `S` = 4096 MiB the write bandwidth lies in [1000, 2039.04] GB/s.
- **E-A1-3** At `S` = 4096 MiB the copy bandwidth lies in [1200, 2039.04] GB/s.
- **E-A1-4** Copy bandwidth is flat above the L2 working set: the values at
  1024, 2048 and 4096 MiB agree within 5 percent of their median.
- **E-A1-5** The 8 MiB copy point, whose 16 MiB working set fits the 40 MiB L2,
  exceeds the 4096 MiB copy point by at least a factor 1.2 and exceeds the HBM
  ceiling of 2,039.04 GB/s. Exceeding the HBM ceiling is expected here and is
  not a fatal violation, because this point is not HBM-resident.

### A2 GEMM roofline crossover

BF16 inputs and outputs with FP32 accumulation through `cublasGemmEx`. Two
frozen sweeps:

- decode-shaped: `N` = `K` = 8192, `M` in
  {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192};
- square: `M` = `N` = `K` in {1024, 2048, 4096, 8192, 16384}.

Five warmup and twenty timed iterations per point, except `M` = `N` = `K` =
16384 which uses two warmup and five timed iterations. Every point carries its
own two-sided floor:

```
bytes(M, N, K) = 2 * (M*K + K*N + M*N)
t_mem = bytes / 2.03904e12
t_flop = 2*M*N*K / 311.87e12
t_floor = max(t_mem, t_flop)
```

At `M` = 1, `N` = `K` = 8192 that gives 134,250,496 B, a memory floor of
65.84 us and a compute floor of 0.43 us, so the point is memory-bound by a
factor 153. Setting `t_mem` equal to `t_flop` at `N` = `K` = 8192 puts the
ideal roofline crossover at `M` = 158.9.

The requirement that no measured time falls below its own `t_floor` is fatal
guard F3, not a scored expectation, because it asserts a physical
impossibility rather than a predicted magnitude.

- **E-A2-1** For `M` <= 64 the achieved rate stays below 5 TFLOP/s, which is
  1.6 percent of the BF16 peak.
- **E-A2-2** For `M` <= 64 the measured times form a plateau: the maximum
  divided by the minimum over that range is at most 1.6.
- **E-A2-3** For `M` <= 64 every measured time lies in [`t_mem`, 3 x `t_mem`].
- **E-A2-4** The measured crossover, the smallest `M` whose time exceeds 1.5
  times the plateau median, lies in [128, 1024], bracketing the ideal 158.9.
- **E-A2-5** For `M` >= 2048 the achieved rate is at least 150 TFLOP/s, which
  is 48 percent of the BF16 peak.
- **E-A2-6** Between `M` = 4096 and `M` = 8192 the time ratio lies in
  [1.7, 2.3], the linear scaling of a compute-bound regime.
- **E-A2-7** The square 8192 point achieves between 180 and 300 TFLOP/s.

### A3 kernel launch cost

An empty kernel is used for every launch measurement.

- `pipelined_period_us`: 200,000 launches into one stream followed by one
  synchronization, divided by 200,000.
- `roundtrip_us`: 2,000 repetitions of one launch followed by one stream
  synchronization, divided by 2,000.
- `graph_period_us`: one CUDA graph holding 1,000 empty kernels, replayed 200
  times with one synchronization per replay, divided by 200,000.

- **E-A3-1** `pipelined_period_us` lies in [0.5, 10.0].
- **E-A3-2** `roundtrip_us` lies in [3.0, 40.0].
- **E-A3-3** `roundtrip_us` is greater than `pipelined_period_us`.
- **E-A3-4** `graph_period_us` is less than `pipelined_period_us`, because
  graph replay removes the per-launch driver path.

### A4 host link

A 256 MiB pinned host buffer transferred to and from the device, three warmup
and ten timed iterations, median reported.

- **E-A4-1** Host-to-device bandwidth lies in [15, 31.5] GB/s.
- **E-A4-2** Device-to-host bandwidth lies in [15, 31.5] GB/s.

## Lane B, multi card over NVLink

### B1 peer bandwidth matrix

`cudaMemcpyPeerAsync` over 1 GiB device buffers, three warmup and ten timed
iterations per cell, median reported.

- unidirectional: all 12 ordered pairs, one at a time;
- bidirectional: devices 0 and 1 copying to each other concurrently on separate
  streams, reported as the sum of both directions;
- fan-out: device 0 copying to devices 1, 2 and 3 concurrently on separate
  streams, reported as the aggregate egress of device 0.

- **E-B1-1** Every unidirectional ordered-pair bandwidth lies in [75, 100] GB/s.
  Above 100 GB/s is fatal.
- **E-B1-2** The 12 unidirectional values agree within 10 percent of their
  median, as the topology is a symmetric mesh with identical `NV4` bonding.
- **E-B1-3** The bidirectional sum for the pair 0 and 1 lies in [150, 200] GB/s.
  Above 200 GB/s is fatal.
- **E-B1-4** The fan-out aggregate egress of device 0 lies in [225, 300] GB/s.
  Above 300 GB/s is fatal.
- **E-B1-5** The fan-out aggregate exceeds the single-pair unidirectional value
  by at least a factor 2.4, because a fan-out reaches all 12 links while a
  single pair reaches four.

### B2 NCCL collectives

One process, one communicator per width, one stream and one host thread per
device, built with `ncclCommInitAll`. Widths are 2, using devices 0 and 1, and
4, using devices 0 through 3. Operations are all-reduce, all-gather,
reduce-scatter and broadcast. Element type is 32-bit float and the reduction is
sum.

Frozen sizes `S` are 8 B and every power of two from 1 KiB to 1 GiB, which is
22 points per operation and width. `S` is the nccl-tests convention: the total
buffer size, so an all-gather of `S` sends `S/n` per rank and receives `S`, and
a reduce-scatter of `S` sends `S` and receives `S/n`. Five warmup and twenty
timed iterations per point, reduced to ten timed iterations above 64 MiB. The
timed block is bracketed by CUDA events on every rank's stream and the reported
time is the maximum across ranks.

Bus bandwidth uses the nccl-tests factors: `2(n-1)/n` for all-reduce,
`(n-1)/n` for all-gather and reduce-scatter, and 1 for broadcast, applied to
the algorithm bandwidth `S / t`.

The requirement that no bus bandwidth exceeds its physical ceiling, 100 GB/s at
width 2 and 300 GB/s at width 4, is fatal guard F2 rather than a scored
expectation.

- **E-B2-1** At `S` = 1 GiB the all-reduce bus bandwidth lies in [70, 100] GB/s
  at width 2.
- **E-B2-2** At `S` = 1 GiB the all-reduce bus bandwidth lies in [130, 300] GB/s
  at width 4.
- **E-B2-3** The width-4 all-reduce bus bandwidth at 1 GiB exceeds the width-2
  value by at least a factor 1.4. Widening the collective raises bus bandwidth
  on this node because a two-rank ring reaches only the four links of one pair
  while a four-rank ring set reaches all twelve links of each GPU.
- **E-B2-4** At `S` = 1 GiB and fixed width, the bus bandwidths of all-reduce,
  all-gather and reduce-scatter agree within 30 percent of their mean. Bus
  bandwidth is a property of the fabric, not of the collective.
- **E-B2-5** An ordinary least squares fit of `t = alpha + S / beta` over `S` in
  [1 MiB, 1 GiB], 11 points, achieves R-squared of at least 0.99 for all-reduce
  at both widths.
- **E-B2-6** The fitted `alpha` for all-reduce lies in [2, 20] us at width 2 and
  in [3, 40] us at width 4, and the width-4 value is the larger of the two.
- **E-B2-7** The measured 8 B all-reduce time lies in [2, 40] us at both widths
  and is larger at width 4 than at width 2.
- **E-B2-8** The all-reduce time ratio between `S` = 1 GiB and `S` = 512 MiB
  lies in [1.8, 2.2] at both widths.
- **E-B2-9** At `S` = 256 MiB and width 4, the all-reduce time lies between
  0.7 and 1.3 times the sum of the reduce-scatter and all-gather times at the
  same size and width.

### B3 collective under compute contention

At width 4 and `S` = 256 MiB the all-reduce is measured alone as `t0`. A BF16
8192-cubed GEMM loop is measured alone on each device as `g0`. Both are then
issued concurrently on separate streams on every device and measured as `t1`,
`g1` and the combined makespan `m`.

- **E-B3-1** `t1` is at least 1.02 times `t0` and `g1` is at least 1.02 times
  `g0`. The collective and the GEMM share SMs and HBM, so neither runs free.
- **E-B3-2** `t1` is at most 3.0 times `t0` and `g1` is at most 2.0 times `g0`.
- **E-B3-3** The makespan `m` is at least `max(t0, g0)` and at most
  0.95 times `t0 + g0`, so real overlap exists and no work disappears.

## Comparison against the surrogates this study exists to replace

These compare a measured A100 value against a constant currently in the
repository. They are predictions about the size and sign of the error the
surrogate carries on this hardware.

- **E-C-1** The measured device-0 fan-out aggregate egress is smaller than
  `DEFAULT_NVLINK_BANDWIDTH_BYTES_PER_SECOND`, currently 450 GB/s, by at least
  a factor 1.5. The flat intra-node rate overstates this node.
- **E-C-2** Fitting `t = alpha + S / beta` over the same window the B200 profile
  used, `S` in [8 B, 256 KiB], the width-4 all-reduce `beta` on this node lies
  in [2, 70.03] GB/s, at or below the 70.027079100 GB/s of
  `b200-nccl-2.27-local-v1`.
- **E-C-3** The width-4 all-reduce `alpha` measured here is greater than
  15.745167 us, the width-4 intercept of `b200-nccl-2.27-local-v1`, because
  A100 NVLink3 rings are longer in time than B200 NVLink5 rings.

## Fatal guards

A violated fatal guard voids the run. The behavioral score becomes
uninterpretable, the study is reported as void with findings, and every task it
touches stays open. These are never reported as a fraction.

- **F1** The single-card lane sees exactly one A100-SXM4-80GB and the
  multi-card lane sees exactly four, every one with MIG disabled and ECC
  enabled, matching the frozen inventory identity.
- **F2** No measured rate exceeds its nameplate ceiling: 2,039.04 GB/s for any
  HBM-resident point at or above 256 MiB, 100 GB/s per ordered NVLink pair
  direction, 200 GB/s for a bidirectional pair sum, 300 GB/s for per-GPU
  egress and for any width-4 bus bandwidth, 100 GB/s for any width-2 bus
  bandwidth, and 31.5 GB/s on the host link.
- **F3** No measured GEMM time falls below its own `t_floor`.
- **F4** Every timed NCCL all-reduce returns the arithmetically correct sum on
  every rank at every size, checked on a fixed element sample.
- **F5** No foreign compute process occupies an allocated GPU at the start of
  either lane.
- **F6** Every reported point carries exactly the frozen warmup and timed
  iteration counts, and every timed block is bracketed by CUDA events on the
  measured stream.
- **F7** Both lanes record the SM clock, memory clock, power draw and
  temperature immediately before and immediately after every timed block, and
  no lane reports a point whose surrounding clock observations are missing.

## Scoring

The scored behavioral denominator is 38: five in A1, seven in A2, four in A3,
two in A4, five in B1, nine in B2, three in B3 and three in C. They are counted
in one total because they are one evidence class: measured hardware rates
checked against magnitudes and relations frozen before the run. Fatal guards F1
through F7 are unscored and never enter that denominator, and the two
physical-impossibility assertions that were moved into F2 and F3 are not
counted twice.

The two lanes are scored separately as well as together, because they run as
two jobs and one lane can void without voiding the other. Lane A carries 18
expectations. Lane B carries 17, plus the three comparison expectations in C,
which read only lane B values and repository constants.

Structural facts recorded but not scored are the observed clocks, the observed
NCCL ring and channel counts, the topology restatement, and the retained raw
sample files.

## What this study does not claim

- It does not close COMP-1. No production framework kernel is captured, no
  Accel-Sim replay is calibrated, no dynamic SASS is traced and no held-out
  kernel matrix is validated.
- It does not close COMP-5. Clocks are not locked, so the controlled-cell
  stability sweep is not performed.
- It does not close SGL-24. No SGLang model step runs and no device-visible
  launch count is observed.
- It does not close CORE-13 or COMP-31. It supplies the NVLink byte and rate
  observations and the collective latency and bandwidth fit those tasks name,
  but it lands no runtime composition and changes no reported TTFT or TPOT.
- It measures no cross-node path. Every number here is intra-node.
- It measures one node. Values are A100-SXM4-80GB scoped on a 4-GPU `NV4` mesh
  without NVSwitch and must not be transferred to H100, B100 or B200, nor to an
  8-GPU NVSwitch baseboard.
