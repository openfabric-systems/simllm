# Merlin fabric flow capture v1 expectations (freeze 1: A100 and mixed families)

## Freeze scope and chronology

This is the expectations-only record for the wave-18 long-running NCCL flow
capture on the Merlin cluster: the calibration reference dataset a
Slingshot-class fabric model in htsim will be validated against in the next
wave. It is committed before the capture harness exists and before any scored
run. No number produced by a scored run of this study may be written back
into this file.

This study makes no fabric-model claims and closes no fabric task. It
produces a frozen per-chunk completion-time dataset with descriptive
statistics, plus the topology and transport evidence for a Merlin-matched
fabric instance. The maintainer's design constraint is binding: flows are
LONG RUNNING so CPU tracer jitter is minimized, and the jitter bound is
measured before the chunk size is frozen. That measurement exists and is
cited below.

Freeze staging, declared in advance:

- **Freeze 1 (this file)**: the A100-family cells and the mixed
  A100-plus-GH200 cells, whose chunk-jitter evidence is measured (discovery
  jobs 195692 and 195710).
- **Freeze 2 (a later, separately committed expectations-only file)**: the
  GH200-to-GH200 cells. Their jitter ladder cannot run before the
  psicourse02 reservation on gpu[001-002] lifts at 2026-08-19 08:00, and the
  chunk size for that family is not frozen until that ladder is measured.
  Freeze 2 will carry its own guard list and its own scored denominator;
  nothing in this file scores a GH-to-GH cell, and no GH-to-GH scored run
  happens before freeze 2 is committed.

No job in this study runs a framework, loads a model, or reports TTFT or
TPOT, so this study cannot close any task whose acceptance names an
end-to-end metric.

## Discovery, unscored, frozen input

Discovery jobs, all on cluster `gmerlin7`, account `merlin`, and all
explicitly unscored:

| Job | Cell | Outcome |
|---|---|---|
| 195692 | a100-2n battery, pair probe, jitter ladder on gpu[101-102] | complete |
| 195693 | gh-1n battery, first attempt | failed: the job script pointed the aarch64 link at the x86_64 NCCL wheel; fixed and resubmitted |
| 195699 | gh-1n battery retry on gpu003 | complete |
| 195694 | a100-4n all-pair matrix, pinned to gpu101,102,103,105 | still pending at freeze time |
| 195695 | gh-2n, first attempt | cancelled before start (same wheel defect as 195693) |
| 195700 | gh-2n battery, pair probe, jitter ladder | still pending at freeze time (reservation) |
| 195696 | het a100 plus gh200 probe, first attempt | allocation GRANTED and ran on gpu105 plus gpu003; NCCL bootstrap timed out on the management network (evidence below); cancelled |
| 195704 | het probe with bootstrap pinned to hsn | complete: cross-architecture communicator works |
| 195710 | het dual-direction jitter ladders and matrix on gpu104 plus gpu003 | complete |

Discovery artifacts and their SHA-256, frozen as inputs to this file:

| Artifact | SHA-256 |
|---|---|
| `disco_lane.cu` (the unscored discovery probe, staged and run) | `7fc49ddd2bee8de6371f1446557c99220aa8085687a9e80cda4a898a1131fcc7` |
| 195692 `disco_jitter.json` | `bfd12a539dc474d5cb7061953441b1170bfffb4c8f8cfa3b9eb72c75b6d0c5b2` |
| 195692 `disco_matrix.json` | `922f88dd4fb03a307c439fe23e6cdb8f906f1fec0a0d93661ffacfca4e34831f` |
| 195704 `disco_het_matrix.json` | `098fdbe4ea429159b506277a6c178c41d4cff7baeeced9a0c38f542a27221dae` |
| 195710 `disco_het_jitter-a100dest.json` | `572e56866fa420d77df8cf8a129c0a9bf91559f8e5ce832febbef2dda65d1aae` |
| 195710 `disco_het_jitter-ghdest.json` | `ce4d6aea77e3035f2a74eae072550bc1bf4bfe899998419887b1cb71bb3be5b3` |
| 195710 `disco_het_matrix.json` | `7d31802724cfd854f1000114600ee13fb028838c7f0941c5ad30027a6bd370e1` |

### Fabric identity

Every observation below matches the wave-16 discovery where they overlap.

- A100 nodes gpu[101-105]: 4 x Cray Cassini 1 `SS11 200Gb 2P` per node
  (`cxi_stat` names them cxi0 to cxi3), driver `cxi_ss1`, one port per NUMA
  domain, interfaces hsn0 to hsn3 at MTU 9000, speed 200,000 Mbit each,
  addresses in 172.30.136.0/22. No InfiniBand device. NCCL 2.31.2 selects
  `NET/Socket` with GDR disabled on every interface.
- GH200 nodes gpu[001-003]: the identical port inventory (4 x Cassini
  SS11 200Gb, cxi_ss1, MTU 9000), with hsn addresses in the same
  172.30.136.0/22. One Slingshot fabric spans both architectures.
- The two architectures' management networks are NOT mutually routable
  (10.100.28.0/22 on A100 nodes against 10.100.36.x on GH nodes). NCCL's
  default bootstrap interface choice therefore times out on a mixed
  communicator (job 195696, connection timeout evidence retained). Pinning
  `NCCL_SOCKET_IFNAME=hsn` moves bootstrap and data onto the shared fabric
  subnet and the cross-architecture communicator then works (job 195704).
- Heterogeneous Slurm jobs spanning `a100-hourly` and `gh-hourly` are
  granted and run (jobs 195696, 195704, 195710). The 195710 A100 component
  ran on gpu104 while the psicourse01 FLEX reservation was active, so FLEX
  reservations do not exclude hourly jobs from those nodes.
- Interface byte counters: `tx_bytes` on the cxi_ss1 driver is consistent
  with payload plus MTU-9000 header overhead (41.0 GB counted for 40.75 GB
  of payload sent in the 195692 jitter ladder, 0.6 percent overhead).
  `rx_bytes` grossly undercounts bulk receive (4.09 GB counted for the same
  40.75 GB delivered and probe-verified). Port accounting in this study
  therefore uses the sender-side `tx_bytes` only, and the receive port is
  identified by its ACK egress and by NCCL's own interface selection line.
- Port usage: a single pair flow uses exactly one hsn port per direction
  (the PHB-affine port of each side's GPU), confirmed by per-port tx deltas.
- NCCL socket transport opened 6 to 8 established TCP connections against a
  baseline of about 627 on the node during the two-rank jitter flow, with 2
  data channels per direction (`via NET/Socket/<dev>` plus `Shared` rows).
- Clocks: chrony synchronized, stratum 3, on every probed node.

### Node-pair latency evidence so far

8-byte NCCL ping-pong RTT (back-to-back over 100 iterations) and isolated
one-way recv-side median (20 samples):

| Pair | 8 B RTT | 8 B one-way | 16 MiB one-way |
|---|---:|---:|---:|
| gpu101 to gpu102 (A100, job 195692) | 79.68 us | 25.60 us | 4389.89 us (3.82 GB/s) |
| gpu105 to gpu003 (A100 to GH200, job 195704) | 80.41 us | 23.23 us | 5252.80 us (3.19 GB/s) |
| gpu104 to gpu003 (A100 to GH200, job 195710) | 72.46 us | 18.98 us | 3922.11 us (4.28 GB/s) |

OS-level `ping` across every hsn interface pairing of gpu101 and gpu102
shows 0.030 to 0.150 ms RTT with 0 percent loss on all 16 combinations. The
full allocatable-pair matrix is owned by the still-pending job 195694 and by
the freeze-2 discovery; whatever lands is recorded in RESULTS as unscored
discovery, and pairs never allocated are recorded as such.

### The jitter bound, measured before the chunk size was frozen

The chunk-size decision rule was stated in the discovery probe's header
before it first ran: the capture chunk is the smallest ladder size whose
tracer contribution stays under 1 percent of the median per-chunk service
time. The tracer floor block (kernel launch, stream synchronize, 4-byte
device-to-host probe readback, `clock_gettime`, no network) measures the
tracer's own cost and spread; the ladder blocks measure realized per-chunk
service time with the identical loop the capture harness will use.

Tracer floor, 1000 repetitions per host:

| Host | p5 | p50 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|
| gpu101 (A100, EPYC host) | 13.33 us | 14.07 us | 15.31 us | 15.86 us | 497.77 us |
| gpu104 (A100, EPYC host) | 12.69 us | 13.16 us | 15.03 us | 15.57 us | 1374.10 us |
| gpu003 (GH200, Grace host) | 12.86 us | 13.31 us | 14.08 us | 14.75 us | 24.35 us |

Per-chunk service time at the 8 MiB ladder point, 200 repetitions:

| Flow | p50 chunk service | tracer p50 over service p50 | tracer spread (p95 minus p5) over service p50 |
|---|---:|---:|---:|
| A100 to A100 (gpu102 to gpu101) | 2216.00 us | 0.63 percent | 0.09 percent |
| GH200 to A100 (gpu003 to gpu104) | 5505.00 us | 0.24 percent | 0.04 percent |
| A100 to GH200 (gpu104 to gpu003) | 1932.00 us | 0.69 percent | 0.06 percent |

**The frozen chunk size is 8 MiB (8,388,608 bytes) for every cell in this
freeze.** 4 MiB fails the rule on the A100-to-GH200 direction (13.31 over
963 is 1.4 percent) and on A100-to-A100 (14.07 over 1111 is 1.27 percent);
8 MiB passes on every measured direction with margin, and contention can
only lengthen chunk service, which loosens the bound further. The remaining
per-chunk spread at 8 MiB (p95 minus p5 of 419 us A100-to-A100, 581 us
GH-to-A100, 301 us A100-to-GH) is transport variability, not tracer noise:
the no-network floor's spread is under 2.7 us on every host. That transport
variability is the signal this dataset exists to record.

One disclosure: the single worst tracer-floor outlier observed is 1374 us
(1 of 1000 on gpu104). At 8 MiB that is 62 percent of one chunk. Such
outliers are rare (about 0.1 percent of samples), are visible in the raw
series as isolated spikes, and are why the descriptive statistics below
report percentiles rather than means alone.

### Solo-flow and mixed-flow rates observed in discovery

| Flow | p50 goodput across the 4 to 128 MiB ladder |
|---|---:|
| A100 to A100 | 3.71 to 3.89 GB/s |
| GH200 to A100 | 1.43 to 1.54 GB/s |
| A100 to GH200 | 4.16 to 4.36 GB/s |

The mixed-pair direction asymmetry (the Grace-sourced direction is about
2.8 times slower than the Grace-sunk direction) was not predicted and is
frozen here as a discovery observation the scored bands below must bracket.

## Napkin bounds, stated before any scored value is read

### Floors

| Floor | Derivation | Value |
|---|---|---:|
| per-chunk completion, 8 MiB | chunk bytes over one 25.0 GB/s Cassini port | 335.5 us |
| 8-byte one-way completion | a kernel-socket hop with a proxy wakeup on each side cannot beat single-digit microseconds | > 5 us |
| per-chunk tracer contribution | measured floor above | about 14 us |
| any flow's chunk sequence | chunk k+1 completes after chunk k on the same flow, by construction of a single stream | ordering, not a time |

### Ceilings

| Ceiling | Derivation | Value |
|---|---|---:|
| any single flow | one Cassini port, one direction | 25.0 GB/s |
| A100 destination aggregate | GPUs stage through host memory, so the host-to-device PCIe 4 x16 leg bounds an A100 sink; measured on this node type | 26.78 GB/s |
| GH200 destination aggregate | four Cassini ports; the Grace-Hopper C2C link is far above them | 100.0 GB/s |
| A100 source-node aggregate (4 GPUs) | device-to-host PCIe legs are per GPU, four ports | 100.0 GB/s |

No measured rate may exceed the matching ceiling. A rate above its ceiling
is evidence of a harness defect, never of hardware, and is fatal (G5).

## Frozen substrate

| Item | Frozen value |
|---|---|
| cluster, account | `gmerlin7`, account `merlin` |
| A100 partition and toolchain | `a100-hourly`, `cuda/12.2.2`, `-arch=sm_80`, x86_64 |
| GH200 partition and toolchain | `gh-hourly`, `cuda/12.9.1`, `-arch=sm_90`, aarch64 |
| NCCL | `nvidia-nccl-cu12` 2.31.2 (reported 23102), the staged x86_64 and aarch64 wheels whose `libnccl.so.2` hashes are recorded per job |
| bootstrap | `ncclUniqueId` through a shared-filesystem file, ranks from Slurm or explicit overrides; no MPI |
| interface selection | NCCL default on A100-only cells; `NCCL_SOCKET_IFNAME=hsn` on mixed cells (required, see discovery) |
| chunk size | 8 MiB (8,388,608 bytes), frozen from the measured jitter bound |
| per-chunk loop | post `ncclRecv` (destination) or probe-write kernel plus `ncclSend` (source); `cudaStreamSynchronize`; 4-byte device-to-host probe readback and sequence check (destination); `clock_gettime(CLOCK_MONOTONIC_RAW)` |
| per-chunk timestamp | taken after the probe check, so the tracer cost inside each chunk is exactly the measured floor |
| flow plumbing | one two-rank NCCL communicator per flow (destination plus one source), one CUDA stream and one host thread per flow at the destination, so flows progress independently and contend only in the stack and on the fabric |
| flow warm connection | one 8-byte exchange per flow communicator before the window, so a join is a data-plane arrival on an established connection, not a TCP handshake |
| chunk identity | the source writes `float(chunk_index)` into elements 0, mid and last before each send; the destination checks element 0 on every chunk on arrival and all three on the final data chunk (double-buffered receive so the sentinel cannot overwrite it) |
| termination | after the window closes the source sends one full-size sentinel chunk with probe value -1; sentinel bytes count toward no goodput and no series row |
| common epoch | all ranks exit a world-communicator barrier together; T0 is each rank's `CLOCK_MONOTONIC_RAW` at barrier exit, with `CLOCK_REALTIME` recorded beside it; the destination clock is the authoritative series clock |
| element type | `float` (4 bytes) |
| bandwidth unit | decimal, 1 GB/s is 1,000,000,000 B/s |
| goodput series | 1-second bins over destination per-chunk completions; a chunk's bytes land in the bin its completion timestamp falls in |
| steady value of a stage | mean goodput over the final 20 seconds of that stage |

## Frozen cells

All A100 cells: `a100-hourly`, one task per node, 16 CPUs per task, 64 GiB
host memory per node, one GPU per task, 30-minute wall limit. The x4 cell
requests 4 GPUs and 4 tasks per node and 256 GiB per node. The destination
is the sole rank on its node in every cell except x4 (where it is the only
active rank on its node). Sources are greedy: each sends chunk k+1
immediately after chunk k completes locally.

| Cell | Nodes x GPUs | Flows (source ranks to destination) | Join offsets (s) | Window (s) | Status |
|---|---|---|---|---:|---|
| `s1-stream` | 2 x 1 | 1 | 0 | 300 | required |
| `i2-incast` | 3 x 1 | 2 | 0, 0 | 180 | required |
| `i3-incast` | 4 x 1 | 3 | 0, 0, 0 | 180 | required |
| `j3-join` | 4 x 1 | 3 | 0, 60, 120 | 240 | required |
| `i4-incast` | 5 x 1 | 4 | 0, 0, 0, 0 | 180 | best effort: needs all five A100 nodes |
| `x4-incast` | 2 x 4 | 4 (all four GPUs of one node) to 1 | 0, 0, 0, 0 | 180 | best effort: needs two drained nodes |
| `mx-pair` | het 1 + 1 | phase runs below | | | required if a het allocation lands in the campaign window |

`mx-pair` phases, in order, in one het job: (1) the three-payload pair
matrix probe (8 B, 128 KiB, 16 MiB); (2) a 60-second stream with the
destination on the A100 side (GH sources the flow); (3) a 60-second stream
with the destination on the GH side. Every phase pins
`NCCL_SOCKET_IFNAME=hsn`.

Queue-degradation rule, frozen: cells are independent jobs; a cell that
never schedules inside the campaign window leaves its relations unevaluated
(never failed, never passed), with its submission id and reproduction path
recorded. No cell substitutes for another, and nothing is extrapolated
across cells. `j3-join` follows the dcqcn_micro join-study shape: flows
join one at a time on a fixed cadence against an established flow, and the
per-chunk series must be dense enough to reconstruct goodput versus time
around each join; at 8 MiB chunks and the rates observed in discovery, each
60-second stage holds thousands of chunks per active flow.

Every capture job re-runs the tracer floor block (1000 repetitions) on
every rank before the window opens, so the jitter bound is re-verified in
the scored run itself rather than transferred from discovery.

## Fatal guards, void and never scored

A violated fatal guard voids the run for the purpose of closing anything;
the behavioral score becomes uninterpretable. Guards are never reported as
a fraction. Evidence for G1, G2, G4 and G6 is collected per rank before and
after each job, exactly as the wave-16 runner did.

- **G1 fabric identity.** Every node reports exactly four Cassini ports and
  zero InfiniBand devices; every communicator logs `Using network Socket`
  and GDR 0. Survivable branch, one direction only: if NCCL selects a
  better transport (`NET/OFI`, `NET/IB`, or GDR active), the run is void
  for every relation below but is retained and reported in full as a more
  capable configuration.
- **G2 clock and timer sanity.** Every per-chunk completion time is finite
  and strictly positive; every flow's destination timestamp sequence is
  strictly increasing; SM and memory clocks recorded before and after.
- **G3 sequence and value conservation.** Every arriving chunk's element-0
  probe equals its expected sequence index, an equality on the float; the
  final data chunk matches on all three probes; each flow's destination
  chunk count equals its source chunk count; exactly one sentinel arrives
  per flow. Any mismatch, gap, duplicate or missing sentinel is fatal.
- **G4 exclusive use.** No foreign compute process on any allocated GPU at
  job start.
- **G5 no rate above its ceiling.** No per-flow goodput above 25.0 GB/s; no
  destination aggregate above 26.78 GB/s into an A100 sink or 100.0 GB/s
  into a GH200 sink, in any 1-second bin outside the first and last bin of
  a stage boundary.
- **G6 declared placement.** Realized ranks, hosts and GPU UUIDs match the
  cell's declared shape; as many distinct GPU UUIDs as active ranks.

## Scored relations

The entailment question was asked of every relation: given the guards, can
this relation fail? Every relation below can. Relations on cells that never
run are unevaluated. Bands were set with the unscored discovery numbers on
the table, and that is stated openly rather than presented as blind
prediction; the scored runs are separate, longer, and on
whichever nodes the scheduler grants.

### Block T, tracer discipline

- **E-T-1** In every captured cell and every flow arm, the in-job tracer
  floor p50 is below 1.0 percent of that cell's realized median per-chunk
  service time, and the floor's p95 minus p5 is below 0.3 percent of it.
  This is the maintainer's design constraint verified live in the scored
  run. Evaluated over all cells that run; fails if any violates.

### Block S, sustained solo stream (cell s1-stream)

- **E-S-1** Steady goodput (mean over the window excluding the first 10
  seconds) lies in [2.0, 5.5] GB/s.
- **E-S-2** Per-chunk completion p95 over p50, same region, is at most 1.5.

### Block I, incast ladder

- **E-I-1** i2-incast aggregate steady goodput lies in [1.8, 9.0] GB/s.
- **E-I-2** i2-incast aggregate steady over s1-stream steady lies in
  [0.9, 2.2]. Two-sided on purpose: a second source-side stack can raise
  the aggregate a destination can absorb, and incast collapse would drop
  it; discovery measured neither.
- **E-I-3** i3-incast aggregate steady goodput lies in [1.8, 12.0] GB/s.
- **E-I-4** i3-incast aggregate steady over s1-stream steady lies in
  [0.9, 3.2].
- **E-I-5** Jain fairness over per-flow steady goodputs is at least 0.6 in
  every incast cell that runs (i2, i3, and i4 or x4 if they run).
- **E-I-6** (best effort) i4-incast aggregate steady lies in [1.8, 14.0]
  GB/s.
- **E-I-7** (best effort) x4-incast aggregate steady lies in [1.8, 12.0]
  GB/s.

### Block J, step-wise joins (cell j3-join)

Stages: stage 0 is [0, 60) with flow 0 alone, stage 1 is [60, 120) with
flows 0 and 1, stage 2 is [120, 240) with all three flows.

- **E-J-1** Flow 0's stage-0 steady goodput lies within [0.7, 1.3] times
  the s1-stream steady goodput. Cross-cell reproducibility of the solo
  rate.
- **E-J-2** Flow 0's stage-1 steady goodput is at most 1.05 times its
  stage-0 steady goodput. A saturating flow cannot gain from a competitor;
  more than 5 percent gain is a defect.
- **E-J-3** Flow 0's stage-2 steady goodput is at most 1.05 times its
  stage-1 steady goodput.
- **E-J-4** The smaller of the stage-1 and stage-2 aggregate steady
  goodputs is at least 0.7 times the stage-0 steady goodput: joining flows
  must not collapse the aggregate by more than 30 percent.

### Block M, mixed pair (cell mx-pair)

- **E-M-1** The GH-to-A100 stream steady goodput lies in [0.7, 3.5] GB/s.
- **E-M-2** The A100-to-GH stream steady goodput lies in [2.0, 8.0] GB/s.
- **E-M-3** The mixed 8-byte one-way isolated median lies in [8, 200] us.
- **E-M-4** Signed prediction from the discovery asymmetry: the A100-to-GH
  steady goodput is at least 1.3 times the GH-to-A100 steady goodput.

### Scored denominator

E-T-1; E-S-1, E-S-2; E-I-1 to E-I-7; E-J-1 to E-J-4; E-M-1 to E-M-4 is
1 + 2 + 7 + 4 + 4 = 18. The denominator is **18**, and
the independent measured quantities are fewer, about 13: the s1 steady
value appears in E-S-1, E-I-2, E-I-4 and E-J-1; the i2 and i3 aggregates
each appear twice; the mixed direction goodputs appear in E-M-1, E-M-2 and
E-M-4. The RESULTS record must say so rather than present 18 independent
facts.

## Retained as entailed or definitional, not scored

- Every rate at or below its ceiling: G5 asserts it.
- Chunk sequences gapless and counts conserved: G3 asserts it.
- Per-flow series strictly ordered in time: G2 asserts it.
- The join cadence realized within tens of microseconds of the declared
  offsets: definitional given the barrier-epoch construction, and reported
  as a number rather than scored.

## Descriptive deliverables, reported and never scored

- The per-chunk completion series per cell, per flow, on the destination
  clock, as the byte-locked dataset the next wave consumes.
- Goodput-versus-time reconstructions (1-second bins) around each join.
- Convergence interval after each join, under this frozen definition: for
  join event at T_j opening stage k, the convergence time C_j is the
  smallest tau such that for every t in [T_j plus tau, end of stage k),
  every active flow's 1-second-bin goodput lies within 25 percent of that
  flow's stage-k steady value. If no such tau exists the stage is reported
  "not converged within stage", which is a legitimate outcome of an
  oscillating transport. Both the tolerance (25 percent) and the bin width
  (1 second) are frozen here.
- Per-stage per-flow distributions: mean, sd, p5, p25, p50, p75, p95, p99
  of per-chunk completion time, plus chunk counts and byte totals.
- Jain fairness per stage.
- Sender-side per-port tx-byte deltas per cell, and the NCCL interface
  selection lines, as the port-usage record.
- Established-socket counts sampled at 1 Hz during every window.
- The tracer floor per rank per job.
- The node-pair latency matrix from whatever discovery pairs land by
  publication time, labeled unscored.

## What is done with a failure

Frozen in advance:

- A refuted band is reported as refuted, with the measured value and a
  statement of whether the miss is a specification error in this freeze or
  a fact about the machine. Bands are not widened after the fact.
- E-M-4 is the one signed relation; if the asymmetry reverses or vanishes
  in the scored window, that is reported as a refutation of the discovery
  observation's stability, which is itself information about the fabric's
  time variability.
- A violated fatal guard voids the run; the evidence is retained; the
  dataset files of a void run are still published but marked void and
  excluded from the calibration reference list.
- No band, tolerance, bin width or steady-window length in this file is
  retuned after any scored number is seen.

## The model action this study will take, frozen before any scored run

- No simllm profile, envelope, arm or reported metric changes in this
  study. The dataset is published as files plus descriptive statistics
  only.
- Tracked in the repository: the harness sources, the job scripts, the
  per-cell summary JSONs with descriptive statistics, and the per-chunk
  series (gzipped CSV) for every cell whose compressed series is small;
  any series too large for the repository stays on the bulk stores with
  its SHA-256 recorded in RESULTS.md and in a tracked manifest. RESULTS.md
  names exactly which files are the wave-19 calibration reference and
  gives their SHA-256.
- TRAF-51 is registered for the wave-19 calibration comparison: the htsim
  Slingshot-class fabric instance versus this dataset, on the incast and
  join families, with the socket host-stack floor separated from fabric
  serialization so the fabric model is not blamed for the stack. Further
  deferrals discovered during the campaign get TRAF-52 and up.
- The GH-to-GH family arrives only through freeze 2, as declared above.
