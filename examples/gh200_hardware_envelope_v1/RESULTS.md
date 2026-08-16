# GH200 hardware envelope v1 results

The reviewed study state is `VALID, 42 of 42`. Every fatal guard held in both
lanes and no scored expectation was refuted, including the forward prediction
E-B2-8 and its falsifier E-B2-10.

The headline is not the Grace Hopper numbers themselves. It is that the A100
study's one substantive finding, that a single-slope collective model is
optimistic across the payload decade where bus bandwidth is still climbing,
reproduces on a different NVLink generation with a different link count, a
different channel count and a different host architecture. TRAF-43 was
registered on evidence from one machine. It now has two.

This is hardware envelope evidence. It runs no framework, loads no model and
reports no TTFT or TPOT. It closes no task on its own.

## Freeze integrity and chronology

The expectations-only commit `1d03b7a` preceded the harness port `d0ca575`,
which preceded both submissions. The [expectations](expectations.md) stayed
byte-identical through both runs at SHA-256
`3136fc383b46e9ef0fcfc8dcd9828b8cd8d41651cbff3f3cfe10baa836cae8b0`.

The freeze declares openly that it is informed by A100 results. Two A100
expectations that were refuted are restated in corrected form and marked in
place, E-A1-5 and E-A2-1, rather than repeated unchanged. The A100 record
keeps its own failures. No GH200 result existed when this freeze was written,
and none was written back into it.

Inventory job `195463` recorded the node before the freeze and timed nothing.

| Artifact | SHA-256 |
|---|---|
| `lane_a_single_card.cu` as submitted | `3a60147e63b4949de2eaca8e63e1e3b918fb17e3000bfe71b83e39ea746a900c` |
| `lane_b_multi_card.cu` as submitted | `1f81b19924248c904423b6e7b5a4a81867403980ae8a5f29c122104399116f4a` |
| staged aarch64 `libnccl.so.2` | `1dbd9a78c092f7b20e597793ca21622644ba1d1baba8a82292605e808f276dd9` |
| `lane_a_result.json` | `91d84ae54135e0b41afdc1861d0b5abb77f03750f85fb79cebf37ced80d9aebc` |
| `lane_b_result.json` | `0423192907d7abd532f16d202d6459953da6b2d8672ea73ee9c791918f5a97e7` |

No post-run repair of any kind was needed. Both lanes ran once and scored.

## Runs

| Lane | Job | Node | GPUs | Elapsed | State |
|---|---|---|---:|---:|---|
| A, single card | `195467` | `gpu003` | 1 | 00:00:12 | complete |
| B, four cards | `195471` | `gpu003` | 4 | 00:00:21 | complete |

Both ran on `gh-hourly` with the frozen allocation, after queueing roughly an
hour each because the partition was saturated. The harness derived the
architecture constants correctly on first contact with Hopper: 4096 BF16 FLOP
per SM per cycle, 4,022.78 GB/s of HBM and 1,070.53 TFLOP/s of tensor peak,
matching the freeze table to within the 0.1 percent tolerance of guard F8.

## Lane A, single card

### HBM bandwidth

| Size | Read GB/s | Write GB/s | Copy GB/s | Copy vs 4022.78 peak |
|---:|---:|---:|---:|---:|
| 4 MiB | 1032.1 | 1032.1 | 1956.3 | 48.6 percent |
| 16 MiB | 2970.5 | 2399.5 | 4185.9 | 104.1 percent |
| 64 MiB | 2970.5 | 2951.7 | 3123.1 | 77.6 percent |
| 256 MiB | 3477.9 | 3488.0 | 3295.1 | 81.9 percent |
| 1 GiB | 3709.1 | 3704.2 | 3409.7 | 84.8 percent |
| 4 GiB | 3757.8 | 3747.3 | 3451.1 | 85.8 percent |

Read reaches 93.4 percent of nameplate and write 93.2 percent, against the
A100's 86.8 and 87.8 percent. Copy flatness across 1, 2 and 4 GiB is 1.01
percent.

The 16 MiB copy exceeds the HBM ceiling at 104.1 percent because its 32 MiB
working set fits the 60 MiB L2. This is the residency signature the A100 study
predicted and failed to observe, and it appears here for the reason that study
identified: GH200 moves the same bytes faster, so a 16 MiB copy clears the
fixed launch floor that swamped an 8 MiB copy on the A100. Guard F2 scopes the
HBM ceiling to points at or above 256 MiB precisely so this is not a violation.

The corrected E-A1-5 confirms that floor directly. Reads of 1 MiB and 2 MiB
took 3.90 and 3.87 microseconds, a ratio of 1.008 against a factor two in
payload, both sitting at 0.63 times the 6.13 microsecond launch roundtrip this
same lane measures.

### GEMM roofline crossover

| M | Time ms | TFLOP/s | Memory floor ms | Time over floor |
|---:|---:|---:|---:|---:|
| 1 | 0.0445 | 3.02 | 0.0334 | 1.33 |
| 64 | 0.0455 | 188.64 | 0.0339 | 1.34 |
| 256 | 0.0532 | 646.44 | 0.0354 | 1.50 |
| 512 | 0.0865 | 794.63 | 0.0375 | 2.30 |
| 8192 | 1.2471 | 881.63 | 0.1001 | 12.46 |

The plateau over `M` from 1 to 64 varies by a factor 1.282 while the work grows
by 64. The measured crossover is 512, above the ideal 284.6 that the 266.12
FLOP per byte machine balance predicts, in the only direction physics allows.
The A100 pair was 256 measured against 158.9 ideal, so both machines land one
sweep point above their own ideal.

The square sweep reaches 918.66 TFLOP/s at 16384 cubed. Against the peak
recomputed at the SM clock observed there, that is 84.9 percent at 8192 and
89.9 percent at 16384.

Clocks held at 1980 MHz through every decode point and the 8192 square, and
dropped to 1890 MHz only at 16384 cubed. The 918.66 TFLOP/s measured there
implies a sustained clock of at least 1699 MHz on its own, so the clock
observation and the throughput are mutually consistent. Reported power peaked
at 230 W of the 900 W superchip limit; that sample is taken after the timed
block, so it is a lower bound on draw during the block rather than a
measurement of it.

### Kernel launch cost

| Quantity | GH200 | A100 | Ratio |
|---|---:|---:|---:|
| pipelined launch period | 1.304 us | 1.806 us | 0.72 |
| launch and synchronize roundtrip | 6.126 us | 6.069 us | 1.01 |
| CUDA graph replay period | 0.589 us | 0.791 us | 0.74 |

The aarch64 Grace host issues launches faster than the x86 EPYC did, by 28
percent on the pipelined path and 26 percent on graph replay. The synchronized
roundtrip is unchanged within one percent, which suggests that path is bounded
by device and driver round-trip rather than by host issue.

### Host link over NVLink-C2C

| Direction | GH200 C2C | A100 PCIe Gen4 x16 | Gain |
|---|---:|---:|---:|
| host to device | 419.93 GB/s | 26.78 GB/s | 15.68x |
| device to host | 169.96 GB/s | 26.19 GB/s | 6.49x |

Both directions passed their bands, but the interesting number is the one no
expectation asked for: the link is asymmetric by a factor 2.47, at 93.3 percent
of the 450 GB/s specification inbound and 37.8 percent outbound. The A100's
PCIe link was symmetric to within 2 percent. Any model that carries one
bidirectional host-link rate is wrong on Grace Hopper in one direction, and
which direction matters depends on whether the workload stages weights in or
reads results out.

## Lane B, four cards over NVLink

### Peer bandwidth matrix

| Pattern | Measured | Ceiling | Efficiency |
|---|---:|---:|---:|
| single ordered pair, all twelve | 133.24 to 133.27 GB/s | 159.375 | 83.6 percent |
| pair 0 and 1, both directions | 264.80 GB/s | 318.75 | 83.1 percent |
| device 0 fan-out to 1, 2 and 3 | 398.71 GB/s | 478.125 | 83.4 percent |

The twelve ordered pairs agree within 0.02 percent of their median and the
fan-out is 2.992 times one pair, so the six-link groups compose as cleanly as
the A100's four-link groups did.

The efficiency is not the same, though. NVLink4 delivers 83.6 percent of wire
rate to a copy engine where NVLink3 delivered 94.0, and the 10-point loss is
uniform across all three patterns, so it is a property of the link generation
rather than of one measurement.

### NCCL collective envelope

| Quantity | Width 2 | Width 4 |
|---|---:|---:|
| asymptotic all-reduce bus bandwidth at 1 GiB | 115.15 GB/s | 336.94 GB/s |
| asymptotic all-reduce algorithm bandwidth | 115.15 GB/s | 224.62 GB/s |
| efficiency against its own link ceiling | 72.3 percent | 70.5 percent |
| 8 B all-reduce time | 6.22 us | 8.46 us |
| mean time over 8 B to 8 KiB | 6.36 us | 9.45 us |
| half-bandwidth payload | 4 MiB | 8 MiB |

Widening from two ranks to four multiplies bus bandwidth by 2.926, against
2.925 on the A100. Two different link counts, six per pair against four, and
the width scaling is identical to three decimal places, because in both cases
a four-rank ring set reaches every link of every GPU while a two-rank ring
reaches only one pair's.

At 1 GiB the collectives agree within 4.96 percent at width 4 and 11.85 percent
at width 2. Broadcast at width 4 reaches 338.27 GB/s against all-reduce's
336.94, a gap of 0.4 percent, where on the A100 the same gap was 7.4 percent.
The reduction lane costs almost nothing here at full width.

### Collective under compute contention

| Quantity | Alone | Concurrent | Ratio |
|---|---:|---:|---:|
| all-reduce, per iteration | 1233 us | 1363 us | 1.106 |
| GEMM, per iteration | 1278 us | 1542 us | 1.206 |
| makespan for 20 collectives and 8 GEMMs | 34.88 ms serial | 27.27 ms | 0.782 |

Overlap saves 21.8 percent here against 16.1 percent on the A100, and the
burden is distributed differently: on the A100 the collective paid more than
the GEMM, here the GEMM pays more.

## The forward prediction, and what it decides

E-B2-8 predicted before the run that the A100 mid-range optimism would
reproduce. Anchoring `alpha` at the measured 8 B time and `beta` at the 1 GiB
algorithm bandwidth, the two-parameter model is exact at both anchors and
optimistic at every payload between them, on both machines:

| Payload | GH200 width 2 | GH200 width 4 | A100 width 2 | A100 width 4 |
|---:|---:|---:|---:|---:|
| 256 KiB | -21.2 percent | -17.0 percent | -24.5 percent | -18.4 percent |
| 1 MiB | -48.1 percent | -22.2 percent | -50.8 percent | -36.5 percent |
| 2 MiB | -35.1 percent | -32.7 percent | -40.9 percent | -45.8 percent |
| 8 MiB | -18.4 percent | -24.6 percent | -19.9 percent | -39.7 percent |
| 32 MiB | -12.4 percent | -12.5 percent | -12.7 percent | -18.3 percent |
| 512 MiB | -2.5 percent | -1.0 percent | -2.3 percent | -1.7 percent |

Every entry is negative. The worst point is -48.1 percent at 1 MiB and width 2
here, against -50.8 percent at the same payload and width on the A100.

E-B2-10 was the falsifier: if the A100's absurd fitted intercept had been an
accident of that machine rather than of the model form, a wide-window fit here
would behave. It did not. Fitting over 1 MiB to 1 GiB puts `alpha` at 56.44
microseconds against a measured 6.22 microsecond floor, a factor 9.07, at an
R-squared of 0.99968. The A100 numbers were 87.36 against 9.11, a factor 9.59,
at 0.99974. The artifact is the same size on both machines, and a
near-perfect R-squared accompanies it both times.

This is now an architecture-independent defect in the model form. TRAF-43 does
not need a third machine.

## Physical sanity review

**Network and serialization physics.** Every rate sits under its ceiling. The
copy-engine efficiency of 83.6 percent is uniform across twelve ordered pairs,
the bidirectional pair and the three-way fan-out, and the fan-out composes at
2.992 times a single pair against a structural bound of exactly 3. Ring
all-reduce reaches 70.5 percent of per-GPU egress at width 4 against the A100's
71.0 percent, a difference of half a percentage point across a link generation.
Two framings of that agreement are both worth keeping: measured against the
wire, the NCCL ring is a near-constant fraction of egress on both machines;
measured against what a copy engine actually achieves on the same fabric, the
ring gets 84.3 percent here and 75.6 percent on the A100, so the Hopper ring
recovers more of what its link can really deliver.

**Compute and memory physics.** HBM read reaches 93.4 percent of a nameplate
derived from the reported 2619 MHz memory clock and 6144-bit bus, and the
largest GEMM reaches 85.8 percent of the peak derived from the reported 1980
MHz SM clock and 4096 BF16 FLOP per SM per cycle. The measured 918.66 TFLOP/s
independently implies at least 1699 MHz sustained, which is consistent with the
1890 MHz sampled after that block. The `M` = 1 GEMM moves 134.25 MB in 44.5
microseconds, an effective 3017 GB/s or 75 percent of nameplate for a shape
with no reuse on one operand, against 71 percent on the A100.

**End-to-end plausibility.** A 70B-class model at tensor-parallel width 4 puts
about 35 GB of BF16 weights on each GPU, floored at 35e9 / 3757.8e9, which is
9.3 milliseconds of streaming per decode step, roughly half the A100's 19.8
milliseconds and in the right neighbourhood for published Hopper serving. One
BF16 activation all-reduce at hidden size 8192 is 16 KiB, measured at 10.48
microseconds at width 4, so 80 layers with two collectives each is 1.68
milliseconds per token, 15 percent of the step against 13 percent on the A100.
At that payload the bus bandwidth is 2.35 GB/s, half of one percent of the
478 GB/s egress ceiling. Tensor-parallel decode is a latency problem on this
fabric too, and a 1.59 times faster link bought only a 1.53 times faster small
collective, while the 1.38 times faster kernel launch tracks it closely. That
is the direct evidence that the small-message floor is launch-bound rather than
wire-bound.

## Calibration this study delivers

For a GH200 120GB 4-GPU `NV6` mesh under NCCL 2.31.2, ring family, no NVLS:

| Parameter | Width 2 | Width 4 |
|---|---:|---:|
| per-collective latency floor, 8 B all-reduce | 6,220,800 ps | 8,457,600 ps |
| mean floor over 8 B to 8 KiB | 6,362,240 ps | 9,447,680 ps |
| asymptotic all-reduce algorithm bandwidth | 115,151,100,868 B/s | 224,623,611,127 B/s |
| validity of the flat-latency regime | up to 8 KiB | up to 8 KiB |
| validity of the flat-bandwidth regime | from 128 MiB | from 128 MiB |
| per-GPU NVLink egress, copy engine | 398.71 GB/s | 398.71 GB/s |

Against the surrogate: `DEFAULT_NVLINK_BANDWIDTH_BYTES_PER_SECOND` of 450 GB/s
is 1.129 times the measured per-GPU egress here and 1.598 times the A100's.
The constant is close to correct on Hopper and wrong by 60 percent on Ampere,
which settles what it is: a Hopper-class machine identity rather than a
portable intra-node rate.

## What stays open

- TRAF-43 is confirmed on two architectures and stays open until the
  regime-aware form lands. It no longer needs more evidence to justify it.
- TRAF-44 should now cover two first-party profiles rather than one.
- CORE-13 stays open. Two nodes of NVLink byte and rate observations exist, but
  no runtime composition landed and no graph JCT moved.
- COMP-1 and COMP-5 stay open. Launch constants and the roofline crossover are
  now measured on two target architectures; the production framework capture,
  Accel-Sim calibration, dynamic SASS and clock locking are not.
- Everything here is intra-node. This is one GH200 120GB variant on a 4-GPU
  `NV6` mesh without NVSwitch, and it does not describe an H100 SXM or PCIe
  board, a GH200 96GB HBM3 part, an NVL32 rack, or any cross-node path.
