# A100 hardware envelope v1 results

The reviewed study state is `VALID, 35 of 38`. Every fatal guard held in both
lanes. Three scored expectations were refuted, and all three are corrected
below with their cause. Two are specification errors in the freeze. The third,
E-B2-6, is a substantive finding about the shape of the collective model this
project uses, and it is the most useful result in the study.

This is hardware envelope evidence. It runs no framework, loads no model and
reports no TTFT or TPOT. It closes no task on its own.

## Freeze integrity and chronology

The expectations-only commit `52ea841` preceded the harness commit `8509a2c`,
which preceded both measurement submissions. The
[expectations](expectations.md) were byte-identical through both runs at
SHA-256
`f2e0d65870502201d4ef51ca8a7d59ecae82d3708fa5ea2b6a0ffee370351f65`.
No measured value was written back into that freeze.

Two jobs preceded the freeze and are inventory and capability evidence only.
Neither timed anything. Job `195456` recorded the node inventory and job
`195459` recorded that both staged NCCL builds initialize and return a correct
all-reduce on this driver. Their observations are quoted in the freeze as
pre-freeze facts.

| Artifact | SHA-256 |
|---|---|
| `lane_a_single_card.cu` as submitted | `52d2ee74b87c0cf041c1ffe402acae1b28cec6e07fc735d3bdde0fb5c4245225` |
| `lane_b_multi_card.cu` as submitted | `0880dcf7950eee6e50706fc64bf04fd20aabae16ef8d259f415aa928befff854` |
| staged `libnccl.so.2` | `dba12e429fe11268b895d0531ba96a7f679f35227d5b1ec77c5febbcd02281bd` |
| `lane_a_result.json` | `921cb279a3aa9a5a89956aa571485d53fbcbd75e052c4c694a7284871773f8d0` |
| `lane_b_result.json` | `faccbefe305e6d635875daa4638f3ad83e630be9547f9246b899a404049a38d6` |

One post-run repair was made, to the scoring script only. Its first version
tried to read a bus bandwidth from the contention cell, which carries none, and
raised before printing any score. The fix skips rows without a bandwidth field.
It changed no measured value, no bound and no expectation. Both raw lane
results predate it and are unchanged.

## Runs

| Lane | Job | Node | GPUs | Elapsed | State |
|---|---|---|---:|---:|---|
| A, single card | `195460` | `gpu105` | 1 | 00:00:17 | complete |
| B, four cards | `195461` | `gpu105` | 4 | 00:00:22 | complete |

Both ran on `a100-hourly` with the frozen allocation. Neither exceeded its wall
limit, and no work ran on a login node beyond staging and a short compile.

Clocks held at the maximum through every compute block: SM 1410 MHz and memory
1593 MHz on every GEMM point, with power rising to 323 W of the 500 W limit and
temperature to 37 C at the largest GEMM. Nothing throttled, so comparing
achieved throughput against the 311.87 TFLOP/s peak derived at 1410 MHz is
valid. The HBM sweep sampled 1275 MHz on the SM clock between kernel batches;
memory clock stayed at 1593 MHz throughout, and the HBM ceiling depends only on
the memory clock.

## Lane A, single card

### HBM bandwidth

| Size | Read GB/s | Write GB/s | Copy GB/s | Copy vs 2039.04 peak |
|---:|---:|---:|---:|---:|
| 8 MiB | 1024.0 | 910.2 | 1638.4 | 80.4 percent |
| 64 MiB | 1489.5 | 1618.2 | 1506.6 | 73.9 percent |
| 256 MiB | 1685.8 | 1719.0 | 1625.7 | 79.7 percent |
| 1 GiB | 1747.6 | 1747.6 | 1680.4 | 82.4 percent |
| 4 GiB | 1770.5 | 1791.3 | 1672.4 | 82.0 percent |

At 4 GiB the read reaches 86.8 percent of nameplate and the write 87.8 percent.
Copy flatness across 1, 2 and 4 GiB is 0.48 percent. Nothing exceeded the
ceiling at any size at or above 256 MiB.

Points below about 64 MiB are not bandwidth measurements. Every one of them
carries the same fixed cost: a 1 MiB read took 6.14 us and a 2 MiB read took
6.14 us, which is the launch roundtrip this same lane measures at 6.07 us. The
small end of this sweep measures the launch path, not the memory system.

### GEMM roofline crossover

BF16 with FP32 accumulation, `N` = `K` = 8192.

| M | Time ms | TFLOP/s | Memory floor ms | Time over floor |
|---:|---:|---:|---:|---:|
| 1 | 0.0922 | 1.46 | 0.0658 | 1.40 |
| 8 | 0.0963 | 11.16 | 0.0660 | 1.46 |
| 64 | 0.1126 | 76.26 | 0.0669 | 1.68 |
| 128 | 0.1147 | 149.80 | 0.0679 | 1.69 |
| 256 | 0.1567 | 219.31 | 0.0699 | 2.24 |
| 1024 | 0.5274 | 260.62 | 0.0823 | 6.41 |
| 8192 | 3.7079 | 296.53 | 0.1975 | 18.78 |

The memory-bound plateau is real and tight: over `M` from 1 to 64 the time
varies by a factor 1.300 while the work grows by a factor 64. The measured
crossover, the first `M` whose time exceeds 1.5 times the plateau median, is
256. The ideal crossover from the machine balance of 152.95 FLOP/B is 158.9,
and the measured value must sit above it because small-`M` kernels do not reach
the peak FLOP rate. Direction and magnitude are both correct.

The square sweep tops out at 302.22 TFLOP/s at 16384 cubed, which is 96.9
percent of the 311.87 TFLOP/s peak. No GEMM ran below its own two-sided floor.

### Kernel launch cost

| Quantity | Value |
|---|---:|
| pipelined launch period, 200,000 empty kernels | 1.806 us |
| launch and synchronize roundtrip, 2,000 repetitions | 6.069 us |
| CUDA graph replay period, 1,000 nodes x 200 replays | 0.791 us |

Graph replay costs 0.44 times a pipelined eager launch and 0.13 times a
synchronized roundtrip. This is the first A100 measurement of the constants
COMP-1 currently has only for Turing, and it forbids nothing: it supplies them.

### Host link

Pinned 256 MiB transfers reached 26.78 GB/s to the device and 26.19 GB/s from
it, which is 85.0 and 83.1 percent of the PCIe generation 4 by 16 nameplate.

## Lane B, four cards over NVLink

### Peer bandwidth matrix

All twelve ordered pairs are `NV4`, four bonded NVLink3 links, ceiling 100 GB/s
per direction.

| Pattern | Measured | Ceiling | Efficiency |
|---|---:|---:|---:|
| single ordered pair, all twelve | 94.00 to 94.07 GB/s | 100 | 94.0 percent |
| pair 0 and 1, both directions | 186.44 GB/s | 200 | 93.2 percent |
| device 0 fan-out to 1, 2 and 3 | 281.65 GB/s | 300 | 93.9 percent |

The twelve ordered pairs agree within 0.04 percent of their median. The fan-out
is 2.995 times one pair, so the three link groups of a GPU compose with no
measurable interference. A copy-engine transfer on this fabric delivers 94
percent of the wire rate, uniformly.

### NCCL collective envelope

NCCL 2.31.2 built 8 channels on the width-2 communicator and 24 on the width-4
one, which is two channels per physical NVLink in both cases. Every connection
was `via P2P/direct`; no proxy or copy-engine hop appears. NVLS multicast is
unavailable on this board, so the algorithm family is ring and tree, not NVLS.

| Quantity | Width 2 | Width 4 |
|---|---:|---:|
| asymptotic all-reduce bus bandwidth at 1 GiB | 72.77 GB/s | 212.89 GB/s |
| asymptotic all-reduce algorithm bandwidth | 72.77 GB/s | 141.93 GB/s |
| efficiency against per-GPU egress ceiling | 72.8 percent | 71.0 percent |
| 8 B all-reduce time | 9.11 us | 12.95 us |
| mean time over 8 B to 8 KiB | 9.55 us | 14.49 us |
| half-bandwidth payload | 2.45 MiB | 8.24 MiB |
| payload reaching 90 percent of asymptote | 128 MiB | 128 MiB |

Widening from two ranks to four multiplies bus bandwidth by 2.925. That is not
a paradox and it is the central structural fact of this node: a two-rank ring
can only use the four links of one pair, while a four-rank ring set reaches all
twelve links of every GPU. On a mesh, participant count buys links.

Bus bandwidth at 1 GiB is close to a property of the fabric rather than of the
collective, but not exactly. At width 4 all-reduce, all-gather and
reduce-scatter agree within 5.16 percent; at width 2 they spread by 12.18
percent. Broadcast sits above all three, at 88.58 GB/s at width 2 and 228.77
GB/s at width 4, because it moves bytes without reducing them. The roughly 18
percent gap between broadcast and all-reduce is the reduction arithmetic and
the extra HBM traffic it needs.

### Collective under compute contention

At width 4 and 256 MiB, with a BF16 8192-cubed GEMM loop on every rank:

| Quantity | Alone | Concurrent | Ratio |
|---|---:|---:|---:|
| all-reduce, per iteration | 1944.4 us | 2880.1 us | 1.481 |
| GEMM, per iteration | 3715.8 us | 4314.0 us | 1.161 |
| makespan for 20 collectives and 8 GEMMs | 68.62 ms serial | 57.60 ms | 0.839 |

Both sides slow down and neither is free. Running them together saves 16.1
percent against running them in sequence. The collective pays the larger share,
which is what a ring implemented as an SM-resident kernel should do when it has
to share the scheduler with a tensor-core kernel.

## The three refuted expectations

### E-A1-5, the L2 signature, refuted by harness resolution

The freeze predicted that an 8 MiB copy, whose 16 MiB working set fits the
40 MiB L2, would exceed the HBM ceiling and beat the 4 GiB point by at least
1.2 times. It measured 1638.4 GB/s against 1672.4 GB/s, a ratio of 0.98.

The cause is in this study's own lane A data. Every event-bracketed kernel here
carries about 6 us of fixed cost, which lane A measures independently as a
6.069 us launch roundtrip. An 8 MiB copy moves 16.8 MB, which at a plausible
L2 rate near 5 TB/s is 3.4 us of work sitting under a 6 us floor. The
measurement cannot see L2 at this size. The freeze should have picked a size
where the transfer time exceeds the launch floor by an order of magnitude, or
should have amortized many copies inside one kernel. This is a specification
error and it says nothing about the hardware.

### E-A2-1, the plateau rate bound, refuted by an arithmetic error

The freeze predicted that the achieved rate stays below 5 TFLOP/s for every
`M` at or below 64. It reaches 76.26 TFLOP/s at `M` = 64.

The prediction contradicts the neighbouring prediction that the freeze got
right. In a memory-bound plateau the time is flat, so the achieved rate must
grow linearly with `M`. The defensible form of this expectation is the `M` = 1
point alone, where the measured 1.46 TFLOP/s is 0.47 percent of peak, and the
plateau claim belongs in E-A2-2 and E-A2-3, both of which passed. This is a
specification error.

### E-B2-6, the fitted latency intercept, refuted by the model shape

This one is not a bookkeeping mistake. The freeze fitted `t = alpha + S / beta`
over 1 MiB to 1 GiB and predicted `alpha` in [2, 20] us at width 2 and [3, 40]
us at width 4, with width 4 larger. The fit returned 87.36 us and 66.15 us,
both far outside their bands and in the wrong order, at an R-squared of 0.99974
and 0.99987.

The high R-squared is not evidence that the model fits. The three largest sizes
span 87 percent of the x-range and dominate an ordinary least squares fit, so
the line reproduces the bandwidth-dominated tail and dumps everything the tail
cannot explain into the intercept. Over that window the bus bandwidth is still
climbing: it passes half of its asymptote only at 2.45 MiB at width 2 and 8.24
MiB at width 4, and reaches 90 percent of it only at 128 MiB. Fitting one line
across a region where the slope changes by a factor three produces an intercept
that is an artifact, not a latency.

The true latency floor is measured directly and it passed its own expectation:
9.11 us at width 2 and 12.95 us at width 4, in the right order, with a flat
plateau of ratio 1.135 and 1.166 up to 8 KiB.

The consequence is quantitative. Anchoring `alpha` at the measured 8 B floor
and `beta` at the 1 GiB algorithm bandwidth, the two-parameter model is exact at
both anchors and wrong in between, always in the same direction:

| Payload | Width 2 signed error | Width 4 signed error |
|---:|---:|---:|
| 16 KiB | -19.7 percent | -20.0 percent |
| 256 KiB | -24.5 percent | -18.4 percent |
| 1 MiB | -50.8 percent | -36.5 percent |
| 2 MiB | -40.9 percent | -45.8 percent |
| 8 MiB | -19.9 percent | -39.7 percent |
| 128 MiB | -8.1 percent | -6.8 percent |

The model is optimistic everywhere between the anchors, by up to a factor two.
A collective priced this way completes too early, and the error peaks exactly
in the 1 to 8 MiB band that tensor-parallel activation exchanges occupy at
moderate batch sizes.

## Physical sanity review

Three independent framings, as the local rules require.

**Network and serialization physics.** Every measured rate sits under its
nameplate ceiling and the efficiencies are internally consistent. Copy-engine
peer transfers deliver 94.0 percent of wire rate; the ring collective delivers
72.8 and 71.0 percent of per-GPU egress at widths 2 and 4; broadcast, which
omits the reduction, recovers to 88.6 and 76.3 percent. Fan-out scales at
2.995 times a single pair against a structural bound of exactly 3. The channel
count NCCL chose, two per physical link at both widths, explains why bus
bandwidth tracks link count rather than participant count directly.

**Compute and memory physics.** HBM read reaches 86.8 percent of the
memory-clock-derived ceiling and the largest GEMM reaches 96.9 percent of the
clock-derived FLOP ceiling, with clocks observed at maximum on both sides of
every timed block. The `M` = 1 GEMM moves its 134.25 MB in 92.2 us, an
effective 1456 GB/s, which is 71 percent of nameplate for a shape with no reuse
on one operand. The measured crossover at 256 lies above the ideal 158.9, the
only direction that is physically possible.

**End-to-end plausibility against real deployments.** Take a 70B-class model at
tensor-parallel width 4 on this node. Per-GPU BF16 resident weight is about
35 GB, so weight streaming alone floors a decode step at 35e9 / 1770e9, which
is 19.8 ms. Published A100 tensor-parallel serving lands in that neighbourhood,
so the memory number is not fantasy. The collective side: one BF16 activation
all-reduce for hidden size 8192 is 16 KiB, measured here at 16.33 us at width
4, so 80 layers with two collectives each is 2.61 ms per decoded token, about
13 percent of the step. At that size the bus bandwidth is 1.50 GB/s, one half
of one percent of the 300 GB/s egress ceiling. Tensor-parallel decode is a
latency problem on this fabric and not a bandwidth problem, which is exactly
why a per-width intercept belongs in the collective model and why the
bandwidth term barely matters there.

## Calibration this study delivers

For an A100-SXM4-80GB 4-GPU `NV4` mesh under NCCL 2.31.2, ring family, no NVLS:

| Parameter | Width 2 | Width 4 |
|---|---:|---:|
| per-collective latency floor, 8 B all-reduce | 9,113,600 ps | 12,953,600 ps |
| mean floor over 8 B to 8 KiB | 9,553,920 ps | 14,489,600 ps |
| asymptotic all-reduce algorithm bandwidth | 72,774,312,725 B/s | 141,927,693,992 B/s |
| validity of the flat-latency regime | up to 8 KiB | up to 8 KiB |
| validity of the flat-bandwidth regime | from 128 MiB | from 128 MiB |
| per-GPU NVLink egress, copy engine | 281.65 GB/s | 281.65 GB/s |

Against the surrogates this replaces:

- `DEFAULT_NVLINK_BANDWIDTH_BYTES_PER_SECOND` of 450 GB/s is 1.598 times the
  measured 281.65 GB/s per-GPU egress on this node, and 2.11 times the width-4
  all-reduce bus bandwidth. The flat intra-node rate is optimistic here.
- `b200-nccl-2.27-local-v1` carries a 70.027 GB/s bandwidth term fitted over
  8 B to 256 KiB. Fitting this node's width-4 all-reduce over the identical
  window returns 68.10 GB/s. The two numbers nearly coincide, and the
  coincidence is the warning: over that window the achieved algorithm bandwidth
  at 256 KiB is only 14.46 GB/s, so 68 to 70 GB/s is a local slope in the
  latency-dominated regime and not a fabric bandwidth on either machine. Using
  it to price a large collective on A100 would overstate the time by a factor
  2.03 at width 4, since 141.93 divided by 70.027 is 2.027.
- The B200 profile's width-4 intercept of 15.745 us brackets this node's
  measured 12.95 to 14.49 us floor closely, which is more agreement than the
  bandwidth term shows. Intercepts transfer between NVLink generations better
  than slopes do.

## What stays open

- CORE-13 stays open. This study supplies the NVLink byte and rate observations
  and the varied payload and participant sweep its acceptance names, but it
  lands no runtime composition and moves no graph JCT.
- COMP-1 and COMP-5 stay open. No framework kernel was captured, no Accel-Sim
  replay calibrated, no dynamic SASS traced and no clock locked. The launch
  constants and the roofline crossover are now measured on the target
  architecture, which removes one of the two stated blockers on COMP-1's
  fixed-step seam but not the production capture.
- SGL-24 stays open and is untouched. No SGLang step ran.
- Everything here is intra-node and A100-scoped on a 4-GPU mesh without
  NVSwitch. None of it transfers to H100, B100, B200 or an 8-GPU NVSwitch
  baseboard, and none of it describes a cross-node path.
