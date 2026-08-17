# Cross-node collective envelope v1 results

The reviewed state is `PARTIAL, 11 of 11 evaluated relations pass, 7 of the 18
frozen relations never got a measurement`. Every fatal guard held. The study
produces the repository's first first-party cross-node collective numbers and
lands them as a calibrated, provenance-labeled profile. **TRAF-36 does not
close**, for reasons the freeze stated before any number existed.

Two results are worth more than the score. The shipped intercept that carries
70.6 to 95.3 percent of the composed SGLang deployment step is a factor 3.74
too small on the one cross-node cell that could be measured. And the regime
break that the intra-node studies could only hypothesize is now a **measured
mechanism**: NCCL's own log names the protocol per call, and the break is the
LL to SIMPLE switch.

## Freeze integrity and chronology

The expectations-only commit `0348861` preceded the harness commit `13a9407`,
which preceded every run. The [expectations](expectations.md) were unchanged
through the measurement at SHA-256
`fc04782d63991edc121d724ad99e0f9ef90befcbfd1ec336c332a1fa3513adf8`. No measured
value was written back into that freeze.

| Step | Identity | Note |
|---|---|---|
| discovery job | `195640` | took no timing measurement of any kind |
| freeze commit | `0348861` | expectations only |
| harness written and staged | | |
| width-2 and width-8 jobs submitted | `195648`, `195649` | both still `PENDING` |
| harness commit | `13a9407` | both jobs still `PENDING` |
| width-2 job ran | `195648` | after the harness commit |

**One chronology disclosure.** The harness commit lands after the jobs were
submitted, not before, though it precedes every run. The staged bytes and the
committed bytes are identical: `crossnode_lane.cu` is
`2150e8567597ea42edc8c6e212139211c3877c3c9ac15512e437629fe4085192` and
`run_lane.sh` is
`18a5453afc1c21d233a2f4f941010b1f758c5e3f2a782ebd8de2cda2f5292de5` in both
places. This is a bookkeeping ordering rather than a validity problem, because
the freeze precedes both, but it is recorded rather than smoothed over.

`measure_w4x2.sbatch`, committed at `025f8c2`, is a **post-specified** cell
added after the freeze. It is labeled post-specified in the file itself and no
relation is scored against it.

## The discovery that reframes the study

Job `195640`, two nodes, 23 seconds, no timing taken. It found that the fabric
is categorically not the one the shipped cross-node envelope targets:

| Property | Observed |
|---|---|
| host NICs per node | 4 x `Cray Inc Cassini 1 [Slingshot 200Gb]` |
| InfiniBand | none; the sysfs InfiniBand class directory is empty |
| NCCL net plugin | `libnccl-net.so` not found; `NET/IB : No device found` |
| selected transport | `Using network Socket` |
| GPUDirect RDMA | disabled on every interface; the ring reports `GDR 0` |

So every cross-node byte is staged through host memory and pushed through the
kernel TCP stack. The ports are 25.0 GB/s each; the stack is the constraint.

## Runs

| Cell | Job | Nodes x GPUs | Elapsed | State |
|---|---|---|---:|---|
| `w2-default` and `w2-fournic` | `195648` | 2 x 1 on `gpu102`, `gpu105` | 00:00:27 | complete |
| `w8-default` | `195649` | 2 x 4 | | **queued, never ran** |
| `w4x2` post-specified | `195654` | 2 x 2 | | **queued, never ran** |
| `w4-default` optional | | 4 x 1 | | **never submitted** |

The width-8 allocation needs four free GPUs on each of two nodes, i.e. two
whole nodes drained. Two of the five A100 nodes sat under the `psicourse01`
reservation, `gpu103` carried a seven-day job with over a day left and `gpu102`
a one-day job, so the scheduler estimated the allocation a full day out.
Resubmitting to `a100-daily` is provably worse there, because that partition
offers only `gpu102` and `gpu103` and `gpu103` is the node blocked for a day.
The frozen optional four-node width-4 cell was never submitted: it needs four
nodes and the reservation leaves three.

**Nothing about the missing cells is estimated, extrapolated or inferred.**
Their relations are reported as unevaluated, never as passed and never as
failed.

## Fatal guards

Every guard held, in both interface arms.

- **G1 fabric identity.** Both nodes reported exactly four Cassini ports and
  zero InfiniBand devices, and both arms logged `Using network Socket` with
  `GDR 0`. The runner fails the job on any other value, so this is enforced and
  not merely observed.
- **G2 clock and timer sanity.** Every one of the 132 reported times is finite
  and strictly positive.
- **G3 byte and value conservation.** Zero mismatching probes across every
  timed cell, as an equality and not a tolerance.
- **G4 exclusive use.** No foreign compute process on either allocated GPU.
- **G5 no rate above its ceiling.** The highest measured rate is 7.91 GB/s
  against the 100.0 GB/s four-port ceiling.
- **G6 declared rank placement.** Two rank directories, two distinct GPU UUIDs,
  two tasks.

Guards are never reported as a fraction. All six held, so the scored numbers
mean what they claim.

## Scored relations, 11 of 11 evaluated

| Relation | Measured | Verdict |
|---|---|---|
| E-P-1 8 B point-to-point in [10, 200] us | 18.790 us | pass |
| E-P-2 128 MiB point-to-point in [1.0, 15.0] GB/s | 3.301 GB/s, 13.2 percent of port | pass |
| E-P-3 point-to-point time increasing, 64 KiB to 128 MiB | increasing | pass, see disclosure |
| E-A-1 width-2 8 B all-reduce in [15, 150] us and at least 1.5x intra-node | 40.141 us, 4.404x | pass |
| E-A-4 measured over shipped cross-node provisional in [1.2, 12] | 2.976x, underestimate as predicted | pass |
| E-A-5 width-2 128 MiB bus bandwidth in [0.5, 15.0] GB/s | 1.610 GB/s, 2.2 percent of intra-node | pass |
| E-T-1 all-to-allv and all-reduce floors within a factor 2 | 1.023x | pass |
| E-N-1 four-port over one-port point-to-point in [1.2, 4.0] | 2.396x | pass |
| E-N-2 8 B floor moves at most 25 percent across arms | 5.2 percent | pass |
| E-C-1 serialization bandwidth monotone, 256 KiB to 4 MiB | monotone within 5 percent | pass |
| E-M-2 measured over shipped local width-2 intercept in [1.5, 12.0] | 3.744x, underestimate as predicted | pass |

Unevaluated for want of a measurement: E-A-2, E-A-3, E-A-6, E-A-7, E-T-2,
E-T-3, E-M-1. All seven need the width-8 cell.

### Two corrections to the freeze's own bookkeeping

Neither changes any bound; both are the freeze getting its arithmetic about
itself wrong.

**The independence count.** The freeze says "18, of which 15 are independent
measured quantities". The real count is 13. The 18 relations read these
quantities: the width-2 default point-to-point 8 B time (E-P-1, E-N-2), its
128 MiB algorithm bandwidth (E-P-2, E-N-1), its time shape (E-P-3), the width-2
all-reduce 8 B time (E-A-1, E-A-2, E-A-4, E-T-1, E-M-2), the width-8 all-reduce
8 B time (E-A-2, E-A-3, E-A-7, E-T-2, E-M-1), the width-2 and width-8 128 MiB
bus bandwidths (E-A-5, E-A-6), the width-2 and width-8 all-to-allv 8 B times
(E-T-1, E-T-2), the width-8 all-to-allv 1 MiB time (E-T-3), the four-port
128 MiB bandwidth and 8 B time (E-N-1, E-N-2), and the width-2 serialization
shape (E-C-1). Two of the thirteen are shapes over many payloads rather than
scalars.

**E-A-5 is one claim wearing two.** Its second conjunct, that the width-2 bus
bandwidth is at most 25 percent of the intra-node 72.77 GB/s, is entailed by
its first: the band's upper edge of 15.0 GB/s is already 20.6 percent of 72.77.
The conjunct could not have failed independently.

### Disclosure on E-P-3

The freeze puts E-P-3 under "Block P, point-to-point ramp at width 2" and gives
the four-port arm its own block, so E-P-3 is scored on the default arm, which
is also the arm the shipped profile is built from. On that arm it passes. **On
the four-port arm the same statement is false**: point-to-point completion
falls from 164.56 us at 768 KiB to 152.42 us at 1 MiB. That is recorded as a
reported-not-scored row rather than left in prose, and the freeze should have
named the arm.

## The measured mechanism behind the regime break

This is the most useful result in the study.

The [collective regime curve](../collective_regime_curve_v1/RESULTS.md) found a
non-monotone serialization bandwidth on two intra-node NVLink machines, said
"the dip has the shape of a protocol transition but this study did not
instrument NCCL's selection, so that mechanism is a hypothesis", and asked a
future study to read NCCL's own thresholds from `NCCL_DEBUG=INFO`. This study
did that. NCCL logs the algorithm and protocol per call, and the log is
unambiguous:

| Payload | Algorithm and protocol |
|---:|---|
| 4 B to 786,432 B | `RING` / `LL` |
| 1,048,576 B to 134,217,728 B | `RING` / `SIMPLE` |

The switch lands between 768 KiB and 1 MiB, and that is exactly where the
measurement breaks:

| Payload | All-reduce time, default arm | Serialization bandwidth |
|---:|---:|---:|
| 524,288 | 973.36 us | 0.5618 GB/s |
| 786,432 | 1456.49 us | 0.5553 GB/s |
| 1,048,576 | **822.84 us** | **1.3397 GB/s** |
| 1,572,864 | 1172.68 us | 1.3888 GB/s |

**Completion time falls by a third while the payload grows by a third.** The
serialization bandwidth jumps by a factor 2.413.

The mechanism explains the size. LL carries a 4-byte flag for every 4 bytes of
payload, so it puts twice the bytes on the wire. On NVLink, which has bandwidth
to spare at these sizes, that doubling is nearly free and the transition shows
up as the shallow dip the intra-node studies measured. On a kernel socket
transport that is bandwidth-starved, the doubling is the dominant cost, and
removing it very nearly doubles the throughput. The measured 2.413 against a
predicted 2.0 leaves about 20 percent for SIMPLE's better pipelining.

So the intra-node dip and this cross-node step are the same mechanism with
opposite signs, and the mechanism is now named rather than hypothesized. Any
interpolating model whose anchors straddle this boundary is wrong on both
sides, and the boundary is observable at communicator init without any fitting.

E-C-1 passed as written, and its frozen prediction was that the cross-node
socket path does not reproduce the intra-node dip. It does not: there is no dip
in 256 KiB to 4 MiB. But the reason is not the one the freeze reasoned from.
The freeze expected the socket path to have no protocol transition at all in
that window; in fact it has a violent one, pointing the other way. **The
prediction was right and its stated reason was wrong**, which is worth more
than a quiet pass.

## Port ceiling against stack efficiency

The brief asked whether the ceiling(port) times efficiency(stack) doctrine
gains or loses cross-node support. It loses a specific piece of it: efficiency
is not one scalar per stack.

| Quantity at 128 MiB | One port, default | Four ports | Gain |
|---|---:|---:|---:|
| point-to-point algorithm bandwidth | 3.301 GB/s | 7.908 GB/s | 2.396x |
| all-reduce bus bandwidth | 1.610 GB/s | 6.466 GB/s | 4.017x |
| all-to-allv per-rank egress | 1.395 GB/s | 5.818 GB/s | 4.171x |

Against the discovered wire ceiling, the default arm delivers 13.2 percent of
one 25.0 GB/s Cassini port on a point-to-point transfer. The four-port arm
delivers 7.9 percent of the 100.0 GB/s that four ports offer. **Adding ports
lowered the efficiency**, which is what a stack-limited rather than
port-limited system does.

The gain is not one number. A single point-to-point pair gains 2.40x from four
ports while the two collectives gain 4.02x and 4.17x, because a collective
spreads across channels and a single pair does not. A model that multiplies one
port ceiling by one stack efficiency cannot express that, and the operation, not
just the stack, has to be an input.

Latency is unmoved by ports, as E-N-2 predicted: the 8-byte point-to-point floor
is 18.79 us on one port and 17.82 us on four, 5.2 percent apart.

## How badly the shipped intercept misprices this cell

The composed SGLang deployment study charges the `lower` arm of
`cross-node-fixed-cost-provisional-v1` in every cross-node cell, which is the
DGX B200 **intra-node** NVLink intercept applied unchanged across a fabric.

| Constant | Value | Measured width-2 cross-node floor | Factor |
|---|---:|---:|---:|
| `b200-nccl-2.27-local-v1` width 2, the `lower` arm | 10.722 us | 40.141 us | **3.744x too small** |
| `b200-nccl-2.27-cross-node-provisional-v1` width 2 | 13.488 us | 40.141 us | **2.976x too small** |
| that profile's own declared upper band edge, width 2 | 17.488 us | 40.141 us | 2.295x too small |

The provisional profile's **entire declared uncertainty band** sits below the
measurement, by more than a factor two at its pessimistic edge. Both signed
predictions in the freeze were correct and both bands held.

The construction can be inspected directly, because a width-2 ring all-reduce
is exactly `2(W-1)` = 2 steps. Measured per-step cost is 40.141 / 2 = 20.07 us.
The repository prices a fabric ring step at 2.000 us on the lower edge, 3.000 us
at the point estimate and 5.000 us at the upper edge. The measured step on this
transport is **4.0 times the pessimistic edge**.

That decomposition also settles one clause of TRAF-36 at this width. NCCL's log
names `RING`, and 2 x the measured one-way point-to-point time of 18.79 us is
37.58 us against a measured all-reduce floor of 40.141 us, a 6.8 percent
residual. The `2(W-1)` ring-step model this repository assumes is **confirmed at
width 2 cross-node**, on the algorithm NCCL actually selected rather than an
assumed one.

## Physical sanity review

Three independent framings, as the local rules require.

**Network and serialization physics.** Every rate sits under its ceiling: the
best measured 7.91 GB/s is 7.9 percent of the four-port 100.0 GB/s, and the
single-port 3.30 GB/s is 13.2 percent of 25.0 GB/s. The all-reduce and the
point-to-point are consistent with each other by construction rather than by
luck: a width-2 ring all-reduce moves each byte twice, and 3.301 / 2 = 1.651
GB/s against a measured all-reduce bus bandwidth of 1.610 GB/s, 2.5 percent
apart. The latency side agrees the same way, with 2 x 18.79 = 37.58 us against a
measured 40.14 us.

**Host and transport physics.** GPUDirect RDMA is off, so each byte crosses the
GPU's PCIe link into host memory before the NIC sees it. That link measures
26.19 GB/s outbound on this node type, and the port offers 25.0 GB/s, so
neither is within a factor five of the measured 3.30 GB/s. The kernel socket
stack is the binding constraint, which is what the four-port result confirms
independently: quadrupling the ports bought 2.4x on a pair, so the ports were
never the limit.

**End-to-end plausibility.** Take the eight-wide expert-parallel reference
geometry the composed study prices, 48 cross-node all-to-allv collectives per
decode step. At the measured width-2 floor of 40.141 us, the fixed cost alone is
1.927 ms per step, against the 1.446 ms that study charges from the transferred
constant. The direction and the size are both plausible for a socket transport,
and both are far above the 0.205 ms modeled decode step the collective-floor
study started from. A cross-node MoE decode on this fabric would be entirely
collective-bound, which is the correct qualitative conclusion for a deployment
with no RDMA.

## The curve, and why it misses its bar

The freeze committed to a four-anchor rule and to reporting the worst held-out
error against the 15 percent bar TRAF-43 registered, without claiming that
task's closure. Both obligations are met and the bar is missed.

| Form | Worst signed held-out error | At payload |
|---|---:|---:|
| the frozen four-anchor curve | -25.53 percent | 512 B |
| the single slope that ships today | -62.43 percent | 524,288 B |

The curve is 2.45 times better than the flat slope and still misses. The cause
is a **specification error in this freeze**, not a fact about the hardware. The
anchor rule takes "the smallest measured endpoint load" as an anchor, and near
the floor the residual is a difference between two nearly equal measured times:
at 64 B the residual is 0.46 us, so its reciprocal is not a bandwidth, it is
noise. The rule anchored on that noise. The freeze forbids retuning the rule
after seeing the error, so it is not retuned, the curve ships as frozen, and the
defect is recorded for the next candidate.

The rule did get the interesting part right: it placed two anchors at 786,432
and 1,048,576, straddling the protocol boundary, which is why the model
reproduces the step rather than averaging across it.

## Calibration this study delivers

For two A100-SXM4-80GB nodes under NCCL 2.31.2, ring family, kernel socket
transport over Cray Cassini Slingshot 200Gb ports, GPUDirect RDMA disabled:

| Parameter | Width 2, one rank per node |
|---|---:|
| per-collective latency floor, 8 B all-reduce, back-to-back | 40,140,799 ps |
| the same floor, isolated method | 55,808,000 ps |
| 8 B pairwise all-to-allv floor | 41,062,400 ps |
| 8 B one-way point-to-point | 18,790,400 ps |
| asymptotic all-reduce bus bandwidth at 128 MiB | 1,610,316,109 B/s |
| point-to-point bandwidth at 128 MiB, one port | 3,300,613,857 B/s |
| point-to-point bandwidth at 128 MiB, four ports | 7,907,908,384 B/s |
| measured fabric ring-step cost | 20,070,399 ps |
| LL to SIMPLE protocol boundary | 1,048,576 B |

`a100-nccl-2.31-cross-node-socket-v1` carries the width-2 row, its band, and a
four-anchor bandwidth curve, with evidence class `calibrated`. It supports
width 2 and **fails closed at every other width**, including the widths the
shipped B200 profiles support, because inventing a width from a neighbouring
one would be inventing a measurement. Nothing selects it, no envelope contains
it, and no reported TTFT or TPOT moves.

## What this anchors and what it cannot

Anchors, on a named machine and a named stack:

- the first measured cross-node collective completion in this repository, at
  one rank per node, for both ring all-reduce and pairwise all-to-allv;
- the port ceiling of a Cassini Slingshot 200Gb link and the fraction of it a
  kernel socket NCCL delivers, separately, so the two are not confounded;
- the operation dependence of the port gain, which no single stack-efficiency
  scalar can express;
- the LL to SIMPLE protocol boundary as a measured mechanism with a payload
  attached;
- the `2(W-1)` ring-step decomposition at width 2, on the algorithm NCCL
  actually chose.

Cannot anchor:

- anything about a 400 Gbit/s RDMA fabric with GPUDirect RDMA, which is what
  the reference configuration assumes and what the shipped cross-node envelope
  targets. The transport is a first-order term, and the measured per-step cost
  of 20.07 us against an RDMA anchor of 2 to 5 us is the size of that term;
- any width above 2. The width-8 allocation never scheduled and the four-node
  width-4 cell was unschedulable, so no mixed NVLink and fabric ring was
  measured at all;
- width 8 at one rank per node, which needs eight nodes and is physically
  impossible on this five-node cluster.

## What stays open

- **TRAF-36 stays open.** Its clause names widths 2, 4 and 8 at one rank per
  node over a 400 Gbit/s fabric. This study satisfies the width-2 clause on a
  fabric that is not the one named, settles the algorithm question at width 2
  by reading NCCL's own selection, and reports the transferred profile's before
  error at the one width it reached. It cannot satisfy the rest here.
- **TRAF-48** is registered for the RDMA capture the envelope actually needs.
- **TRAF-49** is registered because a measured-width-only profile cannot join a
  fixed-cost envelope at all: both arms must support identical widths, so the
  first measured cross-node profile cannot be bracketed against the transferred
  one it exists to check.
- **TRAF-50** is registered for the missing width-8 cell. Jobs `195649`
  (width 8, two nodes by four GPUs) and `195654` (the post-specified width-4
  mixed ring, two nodes by two GPUs) were still queued when this record was
  written, with every A100 on the cluster allocated to other users. Both were
  submitted from the harness committed here, so if either lands its result JSON
  drops into `measurements/` and `score_expectations.py` scores it against the
  same freeze with no code change: pass the new file as `--w8-default`, or read
  the width-4 file as post-specified evidence that no relation is scored
  against. Neither may be folded into the profile without a fresh freeze, the
  width-4 cell because it is post-specified and the width-8 cell because its
  relations were frozen and must be scored rather than fitted.
- TRAF-43 stays open and is untouched. This study reports against its bar for
  comparability and claims nothing about it, because that task is about the
  intra-node NVLink serializer and this is a different transport. What this
  study contributes to it is the mechanism, not a candidate.
