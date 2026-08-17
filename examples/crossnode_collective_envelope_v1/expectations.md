# Cross-node collective envelope v1 expectations

## Freeze scope and chronology

This is the expectations-only record for the first first-party **cross-node**
collective measurement in this repository. It is committed before the
measurement harness exists, before any timed collective, transfer or kernel
runs in this study, and before any result-producing Slurm job is submitted. No
number produced by this study may be written back into this file.

Every cross-node number the repository currently ships is transferred from an
intra-node capture of hardware this project cannot reach.
`b200-nccl-2.27-cross-node-provisional-v1` takes the DGX B200 intra-node
NVLink all-reduce intercepts of `b200-nccl-2.27-local-v1` and swaps each of
`2(W-1)` NVLink ring steps for a fabric step. Nothing in it was measured
across any fabric. The composed SGLang deployment study found that in its ten
surcharge-bearing cells, 70.6 to 95.3 percent of the upper-median step is that
one transferred constant. This study measures the thing.

The study has three jobs and they run in this order:

1. a width-2 job on 2 nodes with 1 GPU each, which measures the point-to-point
   send and receive ramp, ring all-reduce and pairwise all-to-allv, under two
   declared network-interface arms;
2. a width-8 job on 2 nodes with 4 GPUs each, which measures ring all-reduce
   and pairwise all-to-allv on a ring whose hops are part NVLink and part
   fabric;
3. an optional width-4 job on 4 nodes with 1 GPU each, declared below as a
   best-effort extension whose absence is not a failure.

No job runs a framework, loads a model or reports TTFT or TPOT. This study
therefore cannot close any task whose acceptance names an end-to-end metric.

## Registry motivation and what this study can and cannot close

TRAF-36 asks for "the completion time of a small-payload collective, at
participant widths 2, 4 and 8 with one rank per node, over a 400 Gbit/s
fabric". Two clauses of that sentence cannot be satisfied on this cluster, and
they are stated here before any measurement so that no closure claim can be
retrofitted:

- **Width 8 with one rank per node needs 8 nodes.** Merlin has five A100
  nodes. This is a physical impossibility, not a queue problem.
- **The fabric is not 400 Gbit/s RDMA.** Phase 1 discovery, recorded below,
  found four Cray Cassini 1 Slingshot 200Gb NICs per node, no InfiniBand
  device of any kind, and NCCL 2.31.2 falling back to its kernel-socket
  transport with GPUDirect RDMA disabled on every interface.

So TRAF-36 stays open by construction. What this study does produce is the
repository's first measured cross-node collective numbers, on a named port and
a named stack, with the port ceiling and the stack efficiency separated. That
is the evidence the ceiling-times-efficiency doctrine needs on the cross-node
side, and it is the first check of any kind on the transferred constant.

## Phase 1 discovery, unscored

Slurm job `195640` on `gmerlin7`, partition `a100-hourly`, nodes `gpu102` and
`gpu103`, elapsed 23 seconds. It took **no timing measurement of any kind**.
Its probe built a two-rank cross-node communicator, checked one all-reduce for
numerical correctness and completed one send and receive pair. Everything below
is an observation, is unscored, and is a frozen input to the expectations that
follow.

| Property | Observed value |
|---|---|
| nodes | `gpu102`, `gpu103`, one A100-SXM4-80GB visible per task |
| driver | 565.57.01, PCIe generation 4 by 16 |
| host NICs per node | 4 x `Cray Inc Cassini 1 [Slingshot 200Gb]`, plus one 1 GbE management port |
| InfiniBand | the sysfs InfiniBand class directory is empty, `ibstat` and `ibv_devinfo` absent |
| kernel modules | `ib_core` and `gdrdrv` loaded, both with zero users |
| interfaces seen by NCCL | `nmn0` at 0.125 GB/s, `hsn0` to `hsn3` at 25.0 GB/s each |
| NCCL net plugin | `libnccl-net.so` not found; `NET/IB : No device found`; `Failed to initialize NET plugin IB` |
| selected transport | `Using network Socket`, i.e. `NET/Socket` |
| GPUDirect RDMA | `Disabled` for all five HCAs; `use ring PXN 0 GDR 0` |
| ring and channels | `Ring 00 : 1 -> 0 -> 1`, `Ring 01`, 2 coll channels |
| chosen device | `gpu102` used `NET/Socket/3`, `gpu103` used `NET/Socket/1`, in both cases the `hsn` port that is PHB-affine to that node's GPU |
| NCCL internal estimate | the tuner logged `.algo = RING, .proto = LL, timeUs = 16.941334` for the probe's 4 KiB all-reduce |

Two disclosures about that table:

- The last row is NCCL's own analytic estimate, not a measurement. It was
  visible before this freeze, so the reader should discount the width-2
  latency expectation below accordingly. The band in E-A-1 was set from the
  physical reasoning stated with it and not from that number, and the number
  sits near the bottom edge of that band rather than in its middle.
- `libfabric` modules (1.22.0, 2.2.0-oss, 2.4.0-oss, 2.5.1-oss) exist on this
  cluster and `cxi0` is present, so a Slingshot-native `NET/OFI` path is
  conceivable with an `aws-ofi-nccl` plugin that is not installed. This study
  measures the stack as deployed, without installing anything.

## Napkin bounds, stated before any measured value is read

### Floors

| Floor | Derivation | Value |
|---|---|---:|
| cross-node width-2 all-reduce completion | cannot beat the first-party intra-node width-2 floor, because the ring replaces an NVLink hop with a NIC, PCIe, wire and switch hop | > 9.11 us |
| cross-node width-8 all-reduce completion | cannot beat the first-party intra-node width-4 floor, the strongest first-party intra-node anchor that exists on this node type | > 12.95 us |
| any single event-bracketed block | the launch-and-synchronize roundtrip this node measures | > 6.07 us |
| serialization of `S` endpoint bytes | `S` over the 25.0 GB/s Slingshot port rate | `S / 25.0e9` s |

### Ceilings

| Ceiling | Derivation | Value |
|---|---|---:|
| one Cassini port, one direction | 200 Gbit/s decimal | 25.0 GB/s |
| four Cassini ports, one direction | 4 x 25.0 | 100.0 GB/s |
| device to host over PCIe 4 by 16 | measured on this node type by the A100 hardware envelope | 26.19 GB/s |
| host to device over PCIe 4 by 16 | measured on this node type | 26.78 GB/s |

Because GPUDirect RDMA is disabled, every cross-node byte is staged through
host memory, so the end-to-end path at width 2 is bounded by
`min(25.0, 26.19, 26.78)`, i.e. by the wire at 25.0 GB/s. The two PCIe legs
sit on different hosts, so they do not compose into a 13 GB/s bound. The wire
and the PCIe link are within 5 percent of each other here, which means neither
is expected to be the binding constraint: the kernel socket stack is.

No measured rate may exceed the matching ceiling. A rate above its ceiling is
evidence of a harness defect, never of hardware, and is fatal.

## Frozen substrate

| Item | Frozen value |
|---|---|
| cluster, partition, account | `gmerlin7`, `a100-hourly`, account `merlin` |
| CUDA toolchain | `cuda/12.2.2`, `nvcc` release 12.2 V12.2.140, `-arch=sm_80` |
| NCCL | `nvidia-nccl-cu12` 2.31.2, reported version 23102, `+cuda12.9` |
| bootstrap | `ncclUniqueId` written by rank 0 to the shared filesystem and polled by the others; ranks from `SLURM_PROCID`, `SLURM_NTASKS`, `SLURM_LOCALID`; no MPI |
| timing | CUDA events on the measured stream, per rank |
| reported value | the maximum over ranks, reduced by one `ncclAllReduce` with `ncclMax` after every timed block has completed |
| bandwidth unit | decimal, 1 GB/s is 1,000,000,000 B/s |
| element type | `float` (4 bytes), sum reduction |

Clocks are left at the site default and observed before and after each job.
No environment variable tunes NCCL in the default arm.

## Frozen allocation envelope

| Resource | Width-2 job | Width-8 job | Optional width-4 job |
|---|---:|---:|---:|
| nodes | 2 | 2 | 4 |
| tasks per node | 1 | 4 | 1 |
| GPUs per node | 1 | 4 | 1 |
| CPUs per task | 16 | 8 | 16 |
| host memory per node | 64 GiB | 256 GiB | 64 GiB |
| wall limit | 30 minutes | 45 minutes | 30 minutes |
| device memory ceiling | 8 GiB per GPU | 8 GiB per GPU | 8 GiB per GPU |

The width-8 job requests 8 GPUs, which is exactly the `gpu_hourly` QOS cap of
`gres/gpu=8` per user, so only one such job can run at a time. No job requests
exclusive nodes, uses a job array, requeues, installs anything over the
network, or performs computation on a login node. If the 45-minute limit turns
out to be insufficient the final sweep moves to `a100-daily` with the same
harness and the same grid, and the partition change is recorded.

## Frozen sweep

### Payload grid

All 22 payloads, in bytes, used identically for the all-reduce payload, the
point-to-point message size and the per-pair payload of the all-to-allv:

```
8            64           512          4096         16384        65536
131072       196608       262144       393216       524288       786432
1048576      1572864      2097152      3145728      4194304
8388608      16777216     33554432     67108864     134217728
```

The grid is dense from 128 KiB to 4 MiB because that is where the intra-node
study found the serialization bandwidth dip, it carries six latency-floor
points from 8 B to 64 KiB, and it reaches 128 MiB so the saturation regime is
sampled beyond the 64 MiB the brief requires.

### Iterations

Five warmup iterations, then a barrier, then an event-bracketed loop of `N`
back-to-back iterations divided by `N`, with `N` = 20 at or below 8 MiB, 10
from 16 MiB to 64 MiB, and 5 at 128 MiB. This is the same method the
intra-node A100 and GH200 lanes used and the same method `nccl-tests` uses, so
the numbers are comparable to the intra-node floors this study compares against
and to the `nccl-tests` capture the shipped B200 profile came from.

For the six payloads at or below 64 KiB an additional **isolated** reading is
taken: each iteration is bracketed by its own events with a stream
synchronize and a rank barrier between iterations, median over 20 samples.
That number is a diagnostic only. It is reported and never used as a profile
input, because it includes the 6.07 us launch-and-synchronize roundtrip that
the back-to-back method amortizes.

### Cells

| Cell | Nodes x GPUs | Width | Operations | Interface arm |
|---|---|---:|---|---|
| `w2-default` | 2 x 1 | 2 | p2p ramp, all-reduce, all-to-allv | NCCL default selection |
| `w2-fournic` | 2 x 1 | 2 | p2p ramp, all-reduce, all-to-allv | `NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3`, `NCCL_SOCKET_NTHREADS=4`, `NCCL_NSOCKS_PERTHREAD=2` |
| `w8-default` | 2 x 4 | 8 | all-reduce, all-to-allv | NCCL default selection |
| `w4-default` | 4 x 1 | 4 | all-reduce, all-to-allv | NCCL default selection |

`w4-default` is the **optional best-effort cell**. Four A100 nodes are needed
and two of the five are held by the `psicourse01` reservation until
2026-08-19. If the allocation is not granted inside the study window the cell
is absent, its relations are recorded as unevaluated rather than failed, and
nothing else in this freeze changes. No relation below depends on it.

The point-to-point ramp runs only at width 2, where a send and receive pair is
exactly one cross-node hop with nothing else in it.

## Fatal guards, void and never scored

A violated fatal guard voids the run for the purpose of closing anything. The
behavioral score below becomes uninterpretable, because each guard asserts a
precondition under which the scored numbers mean what they claim. Guards are
never reported as a fraction.

- **G1 fabric identity.** Every job records the per-node NIC inventory, the
  contents of the sysfs InfiniBand class directory, the NCCL transport line,
  and the GDR status. All must match the phase 1 discovery: four Cassini
  Slingshot 200Gb ports per node, no InfiniBand device, `Using network
  Socket`, GDR disabled.
- **G2 clock and timer sanity.** SM and memory clocks are recorded before and
  after each job. Every reported time is strictly positive, and every reported
  time is finite.
- **G3 byte and value conservation.** Every timed all-reduce of all-ones over
  `W` ranks returns exactly `W` at three probed elements, an equality and not
  a tolerance because a float32 sum of small integers is exact. Every timed
  all-to-allv returns, at three probed elements per source, exactly the value
  the source rank wrote.
- **G4 exclusive use of the allocated GPUs.** No foreign compute process
  occupies an allocated GPU at job start.
- **G5 no rate above its ceiling.** No cross-node algorithm or bus bandwidth
  exceeds the port ceiling of the ports in play: 25.0 GB/s for a single
  Cassini port, 100.0 GB/s for four.
- **G6 declared rank placement.** Each job records `SLURM_PROCID`,
  `SLURM_LOCALID` and the hostname of every rank, and the realized placement
  matches the cell's declared nodes-times-GPUs shape exactly.

### Survivability

G1 is the only guard with a declared survivable branch, and only in one
direction. If a measurement job finds that NCCL selected `NET/OFI` or
`NET/IB` instead of `NET/Socket`, or found an InfiniBand device, then the
fabric is **better** than the one this freeze characterizes. In that case the
run is void for every relation below, because the relations were written
against a kernel-socket stack, and the measurement is nonetheless retained and
reported in full as a different and more capable configuration. Every other
guard is void with no survivable branch.

## Scored relations

The entailment question was asked of every candidate relation: given the guards
already registered, can this relation fail? Relations that cannot fail are
listed in the retained-entailed section and are not scored.

### Block P, point-to-point ramp at width 2

- **E-P-1** The 8-byte cross-node send and receive completion, back-to-back
  method, lies in [10, 200] us. Reasoning: it must exceed the 9.11 us
  intra-node all-reduce floor and a kernel socket hop with a proxy-thread
  wakeup on each side should not reach a fifth of a millisecond.
- **E-P-2** The point-to-point algorithm bandwidth at 128 MiB lies in
  [1.0, 15.0] GB/s, i.e. NCCL's kernel socket transport delivers between 4 and
  60 percent of the 25.0 GB/s port rate. This is the stack-efficiency claim and
  it is deliberately not the same statement as G5.
- **E-P-3** Point-to-point completion time is strictly increasing over the
  payloads from 64 KiB to 128 MiB inclusive.

### Block A, ring all-reduce

- **E-A-1** The width-2 cross-node 8-byte all-reduce floor lies in [15, 150] us
  and is at least 1.5 times the first-party intra-node width-2 floor of
  9.1136 us. Reasoning: the cross-node ring replaces both NVLink hops with
  host-staged kernel socket hops, and each such hop costs a syscall pair plus a
  proxy-thread wakeup rather than a direct peer store.
- **E-A-2** The width-8 cross-node 8-byte all-reduce floor is strictly greater
  than the width-2 cross-node 8-byte floor.
- **E-A-3** The width-8 cross-node 8-byte all-reduce floor lies in [20, 250] us.
- **E-A-4** The width-2 cross-node 8-byte floor exceeds the shipped
  `b200-nccl-2.27-cross-node-provisional-v1` width-2 intercept of 13.487792 us,
  with a measured-over-shipped ratio in [1.2, 12]. Signed prediction: the
  shipped number is an underestimate, because it prices each fabric step at
  3.000 us on the strength of an RDMA round-trip anchor and this fabric step is
  a kernel socket hop.
- **E-A-5** The width-2 cross-node asymptotic all-reduce bus bandwidth at
  128 MiB lies in [0.5, 15.0] GB/s and is at most 25 percent of the first-party
  intra-node width-2 value of 72.77 GB/s.
- **E-A-6** The width-8 cross-node asymptotic bus bandwidth at 128 MiB exceeds
  the width-2 value by a factor in [1.5, 8.0]. Reasoning: at width 8 each
  node's four GPUs are each PHB-affine to a different Cassini port, so more
  than one port can carry ring traffic, while at width 2 exactly one does.

**E-A-7 is registered as a two-sided band with no predicted sign, and the
reason is stated before the run.** The width-8 cross-node 8-byte all-reduce
floor lies in [20, 250] us relative to the shipped
`b200-nccl-2.27-cross-node-provisional-v1` width-8 intercept of 49.487789 us,
and this study reports the signed error either way. The sign is genuinely not
predictable here, because two errors in the shipped construction push in
opposite directions at width 8: it charges 14 fabric steps where a 2-node
8-rank ring crosses the fabric only twice, which inflates it, and it prices
each step at an RDMA cost where this stack pays a socket cost, which deflates
it. Registering a sign here would be a coin flip dressed as a prediction. The
band is scored; the sign is reported, not scored.

### Block T, pairwise all-to-allv

- **E-T-1** At width 2 the 8-byte all-to-allv floor and the 8-byte all-reduce
  floor agree within a factor 2 in either direction. This is the first
  cross-node test of the operation-shape transfer that every shipped profile
  performs and that TRAF-39 owns.
- **E-T-2** At width 8 the 8-byte all-to-allv floor exceeds the 8-byte
  all-reduce floor by a factor in [1.0, 6.0]. Reasoning: an all-to-allv at
  width 8 posts 7 sends and 7 receives per rank with no reduction tree to
  collapse them.
- **E-T-3** At width 8 and a 1 MiB per-pair payload, the aggregate cross-node
  rate, taken as the 4 MiB per rank that leaves each node's four ranks for the
  other node divided by the measured time, lies in [1.0, 40.0] GB/s.

### Block N, port count against stack efficiency, width 2

- **E-N-1** Forcing all four `hsn` ports raises the width-2 point-to-point
  algorithm bandwidth at 128 MiB over the default single-port selection by a
  factor in [1.2, 4.0]. This is the direct test of the
  ceiling-times-efficiency doctrine on the cross-node side: if the port is the
  binding constraint, four ports buy close to four times; if the socket stack
  is, they buy close to one.
- **E-N-2** The 8-byte point-to-point floor changes by no more than 25 percent
  between the two arms. Ports buy bandwidth, not latency.

### Block C, regime shape

- **E-C-1** At width 2 the measured all-reduce serialization bandwidth, defined
  as endpoint bytes divided by the measured time with the width's own 8-byte
  base latency removed, is monotone non-decreasing over the payloads from
  256 KiB to 4 MiB inclusive, allowing a 5 percent tolerance on each
  consecutive pair. Signed prediction: **monotone**, i.e. the cross-node socket
  path does **not** reproduce the intra-node dip. Reasoning: the intra-node dip
  was hypothesized to be an NVLink protocol transition, and the socket
  transport's chunking is fixed by its own buffer size rather than by an NVLink
  protocol switch. A refutation here would make the dip a property of
  collectives rather than of NVLink, which is the more interesting outcome.

### Block M, how badly the shipped intercept misprices the composed cells

- **E-M-1** The measured width-8 cross-node 8-byte all-reduce floor exceeds
  30.128029 us, the `b200-nccl-2.27-local-v1` width-8 intercept that the
  `lower` arm of `cross-node-fixed-cost-provisional-v1` charges and that the
  composed SGLang deployment study charges in every cross-node cell, with a
  measured-over-shipped ratio in [1.2, 8.0]. Signed prediction: the shipped
  number is an underestimate. Reasoning: an intra-node NVLink floor cannot
  exceed a host-staged kernel-socket cross-node floor on comparable silicon.
- **E-M-2** The measured width-2 cross-node 8-byte all-reduce floor exceeds
  10.722112 us, the `b200-nccl-2.27-local-v1` width-2 intercept, with a ratio
  in [1.5, 12.0]. Signed prediction: underestimate, same reasoning.

### Scored denominator

Eighteen relations are scored: E-P-1 to E-P-3, E-A-1 to E-A-7, E-T-1 to E-T-3,
E-N-1, E-N-2, E-C-1, E-M-1, E-M-2. That is 3 + 7 + 3 + 2 + 1 + 2 = 18. Two of
those, E-A-4 and E-M-2, are the same measured quantity compared against two
different shipped constants, and E-A-3 and the band half of E-A-7 are the same
interval on the same quantity. They are retained as separate rows because they
answer separate questions about separate shipped numbers, and the RESULTS
record must say so rather than presenting 18 as 18 independent facts. **The
scored denominator is 18, of which 15 are independent measured quantities.**

The optional `w4-default` cell adds no scored relation.

## Retained as entailed, not scored

- Every cross-node rate sits at or below its port ceiling. G5 asserts this as
  a fatal guard, so a scored copy would double count.
- All-reduce returns the world size. G3 asserts it.
- Cross-node completion exceeds the intra-node floor at the matching width.
  The interval floors of E-A-1 and E-A-3 already imply it and the napkin
  section states it as a bound rather than a prediction.
- The realized rank placement matches the declared cell shape. G6 asserts it.

## What is done with a failure

Frozen in advance, so that a miss is reportable rather than tempting:

- A refuted band is reported as refuted, with the measured value, the cause,
  and a statement of whether the miss is a specification error in this freeze
  or a fact about the hardware. The band is not widened after the fact.
- E-C-1 is the one relation whose refutation is more valuable than its
  confirmation, and either outcome is reported in the same words.
- No anchor rule, bar or interval in this file is retuned after any error is
  computed.
- If a fatal guard is violated, the run is reported as void with findings, its
  measurements are retained, and TRAF-36 stays open regardless of what the
  numbers look like.

## The model action this study will take, frozen before the measurement

A new profile is added to `simllm/traffic/collective_latency.py`, named for the
hardware and the stack it was measured on, carrying a
`CollectiveLatencyProvenance` record with evidence class `calibrated`, and:

- a `participant_latency_ps` table containing **only** the widths this study
  actually measured with a realized cross-node ring, so every unmeasured width
  fails closed through the existing mechanism;
- a per-width `CollectiveBandwidthCurve` whose anchors follow this frozen
  rule, applied per width: the smallest measured endpoint load; the payload at
  which the measured serialization bandwidth first reaches 50 percent of its
  maximum over the sweep; the payload of the local minimum of the measured
  serialization bandwidth inside the 128 KiB to 8 MiB window if one exists, and
  otherwise the geometric midpoint of that window; and the largest measured
  endpoint load;
- a held-out set consisting of every measured payload that is not an anchor,
  with the worst signed held-out error reported against the 15 percent bar
  TRAF-43 registered. That bar is quoted for comparability. This study does not
  claim TRAF-43's closure, because TRAF-43 is about the intra-node NVLink
  serializer and this is a different transport;
- an uncertainty band per width taken from the spread of the measured samples
  the profile's point value is drawn from, so the profile cannot be read
  without its uncertainty;
- no change to any existing profile, envelope or arm. The `off` arm, the
  existing intra-node arm and every accepted artifact stay byte-identical, and
  a test asserts it.

The profile is selectable and selected by nothing. No reported TTFT or TPOT
moves in this change.
