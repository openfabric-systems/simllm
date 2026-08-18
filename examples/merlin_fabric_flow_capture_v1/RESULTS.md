# Merlin fabric flow capture v1 results

The reviewed state is **COMPLETE for the A100 families: 18 of 18 frozen
relations evaluated, 16 pass and 2 fail honestly (E-J-2 and E-J-3, the
join cell's established-flow bars, classified below under the freeze's
failure policy), every fatal guard held in every captured cell, and only
the GH200-to-GH200 family still gated on its freeze-2 jitter ladder**.
The study publishes the byte-locked reference dataset the wave-19
Slingshot calibration comparison consumes: 1,305,427 verified 8 MiB
chunks across the scored cells (1,457,959 including the post-specified
join cell), each with a destination-clock completion timestamp, over one
solo stream, a full incast degree ladder (degrees 1 through 4 plus the
two-node four-GPU-source shape), the step-wise join cell, one
mixed-architecture pair in both directions, and one post-specified
two-flow join. The incast ladder's aggregate is non-monotone in degree
(4.99, 8.55, 10.12, 8.35 GB/s at degrees 1 to 4), peaking at degree 3
with perfect fairness throughout. This study makes no fabric-model
claims and closes no fabric task; TRAF-51 registers the comparison and
TRAF-52 the families still queued.

## Freeze integrity and chronology

The expectations-only commit `cc276c2` froze the cells, guards, bands, the
8 MiB chunk (from the measured jitter bound), and the convergence and
steady-window definitions, at 15:55:28 on the workstation clock. The
harness commit `d4a2e8f` carries 16:04:24 on the same clock; Slurm's
accounting records every scored submission at 16:05:04 to 16:05:05 on the
cluster clock (chrony, stratum 3), with the first run starting 16:05:10.
Read across those clocks the submissions follow the harness commit by
about forty seconds, but the two timestamps live on different machines, so
the ordering claim is only as strong as their agreement, and an earlier
draft of the working ledger wrote "~16:02" for the submissions from
memory, which the accounting record corrects. The claim that depends on no
clock is byte identity: `fabric_flow_lane.cu` has SHA-256
`145b3e9a0b3b8d34d8df1e0f5636ac8512e5580cf60b27b6a2b00f22c7f603da` in the
`d4a2e8f` tree, in the Merlin stage, and in every captured job's in-run
`source.sha256`, now packaged per cell in the dataset, so the bytes that
ran are the committed bytes for every captured cell regardless of clock
skew. That byte identity, not the timestamp ordering, is the guarantee
this record stands on, and it is the same guarantee the wave-16 record
relied on for its own submission-before-commit disclosure.

| Step | Identity | Note |
|---|---|---|
| discovery jobs | 195692, 195693/195699, 195696, 195704, 195710 | unscored; probe `disco_lane.cu` staged, run, and committed post-freeze byte-identical (SHA-256 `7fc49ddd...` in every job's `source.sha256`) |
| freeze 1 | commit `cc276c2` | expectations only |
| harness | commit `d4a2e8f` | lane, job bodies, sbatch files, analyzer; `fabric_flow_lane.cu` SHA-256 `145b3e9a...` matches every captured job's in-run `source.sha256` |
| analyzer het fix and packager | commit `77becdb` | analysis tooling |
| G6 distinct-GPU enforcement | commit `9dcdffa` | staged before any A100 cell ran |
| scored submissions | jobs 195728 to 195735 | 2026-08-17 ~16:02 |
| post-specified j2x cell | commit `5892587`, job 195764 | added after freeze 1, labeled unscored in the file itself |

Three disclosures. First, the discovery probe necessarily ran before the
freeze (its outputs are the freeze's cited inputs); it is unscored, its
outputs' hashes are pinned in the freeze, and the committed source matches
every run copy exactly. Second, the analyzer initially misread the
heterogeneous job's transport evidence layout and flagged a false G1
violation on the mixed cell; the fix (`77becdb`) changed only where the
analyzer reads evidence that was already recorded and enforced in-job.
Third, the analyzer briefly included the post-specified j2x cell in scored
E-T-1's evaluation set; the fix restricting scored relations to frozen
cells changed no verdict (j2x satisfied the criterion too) and landed
before this record was assembled.

## Discovery (unscored)

The freeze carries the full discovery record; headlines, plus what landed
after the freeze:

- One Slingshot fabric spans both architectures: every A100 node
  (gpu[101-105]) and every GH200 node (gpu[001-003]) carries 4 x Cray
  Cassini 1 `SS11 200Gb 2P` (driver `cxi_ss1`, one port per NUMA domain,
  MTU 9000), with every hsn address in 172.30.136.0/22.
- The two architectures' management networks are not mutually routable, so
  NCCL's default bootstrap interface choice times out on a mixed
  communicator (job 195696, evidence retained). Pinning
  `NCCL_SOCKET_IFNAME=hsn` moves bootstrap and data onto the shared fabric
  subnet, and job 195704 then formed the first working cross-architecture
  NCCL communicator here (x86_64 A100 rank plus aarch64 GH200 rank, both
  NCCL 2.31.2).
- Heterogeneous Slurm jobs spanning `a100-hourly` and `gh-hourly` are
  granted and run. The psicourse01 FLEX reservation admits hourly jobs on
  its nodes (jobs 195710 and 195732 ran on gpu104) and floats across nodes
  (REPLACE flag: it sat on gpu[102,104] in the morning and gpu[101,104] in
  the evening), which is why multi-node allocations stay pinned behind it
  even while single-node jobs slip through.
- `rx_bytes` on cxi_ss1 undercounts bulk receive by roughly a factor 10 on
  both architectures; `tx_bytes` matches payload plus MTU-9000 header
  overhead to a fraction of a percent (1506.19 GB counted for 1496.6 GB of
  payload in the s1 cell, 0.64 percent). All port accounting here is
  sender-side.
- A single pair flow rides exactly one hsn port on the sender in every
  observed case; the destination's two NCCL channels may split the return
  path across two ports (s1). NCCL's socket transport added 6 to 8
  established TCP connections per flow against the node baseline.
- Tracer floors (launch, synchronize, 4-byte readback, clock): 13.2 to
  14.5 us p50 on EPYC hosts with rare millisecond outliers (worst 1374 us,
  about 0.1 percent of samples), 13.3 us p50 with a 24 us maximum on the
  Grace host.
- The chunk-size rule (tracer p50 under 1 percent of median chunk service)
  rejected 4 MiB and froze 8 MiB on the measured evidence of jobs 195692
  and 195710.
- The hsn `ping` matrix between gpu101 and gpu102 (all 16 interface
  pairings) showed 0 percent loss everywhere with RTT averages 0.042 to
  0.150 ms.

Node-pair latency evidence (8-byte NCCL ping-pong RTT back-to-back;
isolated one-way recv-side median; 16 MiB one-way):

| Pair | 8 B RTT | 8 B one-way | 16 MiB one-way |
|---|---:|---:|---:|
| gpu101 to gpu102 (A100-A100, 195692) | 79.68 us | 25.60 us | 4389.89 us |
| gpu105 to gpu003 (A100-GH200, 195704) | 80.41 us | 23.23 us | 5252.80 us |
| gpu104 to gpu003 (A100-GH200, 195710) | 72.46 us | 18.98 us | 3922.11 us |
| gpu104 to gpu003 (mx capture matrix, 195732) | 71.75 us | 24.26 us | 4232.74 us |

The last row is not discovery: it is phase 1 of the frozen mx-pair cell,
and its 8-byte one-way value is the one scored under E-M-3; it appears in
this table once, for comparability with the discovery pairs above it.

The all-pair matrix jobs both landed 2026-08-18 evening (195694 pinned to
gpu101, gpu102, gpu103, gpu105 in 42 seconds; 195737 unpinned, drawing
the same four nodes, in 44 seconds), giving two independent samples of
the full six-pair 8-byte matrix, now packaged under `dataset/discovery/`
beside the frozen first two-node sample (whose `disco_matrix.json` copies
in bit-identical to its freeze-pinned digest `922f88dd...`):

| Pair | RTT 195694 / 195737 (us) | one-way 195694 / 195737 (us) |
|---|---|---|
| gpu101-gpu102 | 82.24 / 86.08 | 22.53 / 22.53 |
| gpu101-gpu103 | 79.96 / 87.47 | 22.53 / 30.72 |
| gpu101-gpu105 | 80.17 / 76.52 | 20.48 / 21.50 |
| gpu102-gpu103 | 82.17 / 79.19 | 63.49 / 60.42 |
| gpu102-gpu105 | 74.85 / 77.09 | 35.84 / 40.96 |
| gpu103-gpu105 | 83.73 / 74.90 | 22.53 / 23.55 |

Back-to-back RTTs are uniform across all pairs (74.9 to 87.5 us, no
pair structure), while the isolated one-way medians show a reproducible
elevation on exactly the two pairs involving gpu102 as a receiver-side
endpoint (60 to 63 us on gpu102-gpu103 and 36 to 41 us on gpu102-gpu105,
in both samples), consistent with resident load on that node rather
than fabric distance. gpu104 pairs were never sampled by these jobs
(the pinned set predates the FLEX discovery) and are reported as never
sampled. All of this is unscored discovery evidence.

## Cells and runs

| Cell | Job | Nodes x GPUs | Elapsed | State |
|---|---|---|---:|---|
| `mx-pair` (matrix, gh2a, a2gh) | 195732 | het 1 A100 (gpu104) + 1 GH200 (gpu003) | 00:02:18 | complete, rc 0 |
| `s1-stream` | 195728 | 2 x 1 (gpu102 dest, gpu105 source) | 00:05:14 | complete, rc 0 |
| `i2-incast` | 195729 | 3 x 1 (gpu102 dest; gpu103, gpu105 sources) | 00:03:14 | complete, rc 0 |
| `j2x-join` post-specified | 195764 | 3 x 1 (gpu102 dest; gpu103, gpu105 sources) | 00:03:12 | complete, rc 0 |
| `i3-incast` | 195730 | 4 x 1 (gpu101 dest; gpu102, gpu104, gpu105 sources) | 00:03:14 | complete, rc 0; landed 2026-08-18 evening by backfill, folded in by the disclosed tranche-2 packaging |
| `j3-join` | 195731 | 4 x 1 (gpu101 dest; gpu102, gpu104, gpu105 sources) | 00:04:13 | complete, rc 0; landed 2026-08-18 evening, folded in with two honest relation failures recorded below |
| `i4-incast` best effort | 195734 | 5 x 1 (gpu101 dest; all four other A100 nodes source) | 00:03:12 | complete, rc 0; landed 2026-08-18 evening, every A100 node in one cell |
| `x4-incast` best effort | 195735 | 2 x 4 (gpu102 sources, gpu105 dest) | 00:03:17 | complete, rc 0; landed overnight after the first publication and was folded in by the disclosed follow-up packaging |

At first publication the blocking mechanism was structural: the psicourse
reservations plus the floating FLEX reservation left too few simultaneous
free nodes for a four-or-five-node allocation. On the evening of
2026-08-18 the scheduler backfilled all three remaining A100 cells (and
both pair-matrix discovery jobs) onto windows that included the
FLEX-reserved nodes, hours before the reservation lift. Each landed cell
was collected, digest-verified byte-for-byte against the Merlin originals
(231 files across the five late trees, zero mismatches; the j3 pull
landed within seconds of job end and was re-verified specifically to rule
out a partial flush), and scored by the byte-unchanged frozen analyzer:
the reproduction path worked exactly as this record promised, three more
times.

The GH200-to-GH200 family is not in this freeze at all: as declared, it
arrives only under a freeze-2 expectations commit gated on its own jitter
ladder (job 195700, queued; schedulable no earlier than 2026-08-19 08:00).

## Fatal guards

Every guard held in every captured cell; guards are never a fraction.
Enforcement differed by cell family and is stated exactly rather than
summarized. In the homogeneous cells (s1, i2, j2x) the job body enforced
G1 (Cassini port count, InfiniBand absence, Socket transport, GDR state),
G4 and G6 before the window and failed the job on any violation, and the
lane enforced G3 in its exit code. In the heterogeneous mx job the in-job
enforcement covered the transport half of G1 (Socket selected, GDR 0,
asserted before exit) and G3; the port-inventory, foreign-process and
placement evidence was recorded per side but not asserted in-job. The
analyzer re-derives all seven verdicts (G1 in both halves, G2, G3, G4, G5,
G6) for every cell from the packaged evidence alone, and all hold for all
five captured cells; the mx cell's G4 and G6 therefore rest on
recorded-then-rechecked evidence rather than in-job assertion, which is
worth stating because four of the ten passing relations read that cell.

- **G1 fabric identity.** Four Cassini ports and zero InfiniBand devices
  per node; `Using network Socket` and GDR 0 in every communicator log;
  enforced in-job for the homogeneous cells, transport half enforced
  in-job for mx, both halves re-derived for every cell by the analyzer
  from the packaged evidence.
- **G2 clock and timer sanity.** Every one of the 1,457,959 recorded
  completion timestamps is finite, positive, and strictly increasing per
  flow.
- **G3 sequence and value conservation.** Every arriving chunk's sequence
  probe matched its expected index on arrival (zero mismatches across all
  cells); destination and source chunk counts equal per flow; exactly one
  sentinel per flow; final-chunk three-probe and fill-constant checks
  exact.
- **G4 exclusive use.** No foreign compute process on any allocated GPU.
- **G5 ceilings.** No per-flow 1-second bin above 25.0 GB/s; no
  destination aggregate bin above 26.78 GB/s (A100 sink) or 100.0 GB/s
  (GH200 sink).
- **G6 declared placement.** Rank directories equal tasks; distinct GPU
  UUIDs at least the task count; realized hosts recorded per rank. One
  analyzer defect disclosed: the first dataset-mode re-derivation formula
  required as many hosts as active ranks, which mis-generalizes to the
  eight-rank two-node x4 shape and false-flagged it when that cell landed;
  the formula now checks hosts against the cell's declared node count. The
  in-job enforcement had already held (8 rank directories, 8 distinct GPU
  UUIDs), no previously published cell's verdict or stats bytes changed,
  and the corrected formula still fails on any genuinely wrong placement.

## Scored relations, 18 of 18 evaluated: 16 pass, 2 fail

| Relation | Band | Measured | Verdict |
|---|---|---|---|
| E-T-1 tracer discipline, every frozen captured cell | floor p50 under 1.0 percent and spread under 0.3 percent of median chunk service | all eight frozen cells within bounds (worst rank 0.90 percent p50, on s1) | pass |
| E-S-1 s1 steady goodput | [2.0, 5.5] GB/s | 4.991 GB/s | pass |
| E-S-2 s1 per-chunk p95 over p50 | at most 1.5 | 1.322 | pass |
| E-I-1 i2 aggregate steady | [1.8, 9.0] GB/s | 8.548 GB/s | pass, 95 percent of the upper edge |
| E-I-2 i2 aggregate over s1 steady | [0.9, 2.2] | 1.713 | pass |
| E-I-3 i3 aggregate steady | [1.8, 12.0] GB/s | 10.115 GB/s | pass |
| E-I-4 i3 aggregate over s1 | [0.9, 3.2] | 2.027 | pass |
| E-I-5 Jain in every incast cell that runs | at least 0.6 | i2: 0.9795, i3: 0.9995, i4: 1.0000, x4: 0.9496 | pass |
| E-I-6 i4 aggregate steady | [1.8, 14.0] GB/s | 8.354 GB/s | pass |
| E-I-7 x4 aggregate steady | [1.8, 12.0] GB/s | 11.095 GB/s | pass, 92 percent of the upper edge |
| E-J-1 j3 flow-0 stage-0 over s1 | [0.7, 1.3] | 0.8307 | pass |
| E-J-2 j3 flow-0 stage-1 over stage-0 | at most 1.05 | 1.0729 | **FAIL** |
| E-J-3 j3 flow-0 stage-2 over stage-1 | at most 1.05 | 1.0894 | **FAIL** |
| E-J-4 j3 aggregate floor after joins | at least 0.7 x stage 0 | 1.8634 | pass |
| E-M-1 gh-to-a100 steady | [0.7, 3.5] GB/s | 1.202 GB/s | pass |
| E-M-2 a100-to-gh steady | [2.0, 8.0] GB/s | 3.336 GB/s | pass |
| E-M-3 mixed 8 B one-way | [8, 200] us | 24.256 us | pass |
| E-M-4 signed: a100-to-gh at least 1.3 x gh-to-a100 | at least 1.3 | 2.775 | pass |

The freeze's independence disclosure stands: the 18 relations read about
13 independent quantities, all now evaluated. (The previous revision's
section header still said 10 evaluated while its own table showed 11, a
merge-era slip corrected here.) Three qualifications on the passing set.
E-T-1 and E-I-5 are quantified over "every cell that runs"; with every
A100 cell captured they cover eight frozen cells and four incast cells
respectively, both passing, and this freeze's scope is now closed (the
GH family arrives only under freeze 2 with its own denominator). E-M-3's band [8, 200] us was wide
enough that failure was implausible short of a harness defect: its lower
edge is 1.6 times this study's own stated floor and its upper edge 8
times the largest observation in hand at freeze time, so its pass carries
little evidential weight and is counted with that caveat. E-I-2 divides
an aggregate defined on the final-20-second stage window by the s1 steady
defined on the whole window excluding the first 10 seconds, as the freeze
wrote them; with both sides on the final-20-second definition the ratio
is 1.7157 against the published 1.7129, and the verdict is unchanged
under either reading.

**The two failures, classified under the freeze's failure policy.** E-J-2
measured 1.0729 and E-J-3 measured 1.0894 against the frozen bar of at
most 1.05: the established flow's steady goodput rose about 7 then 9
percent across the two joins instead of holding or falling. Both bands
are refuted as written and neither is widened. The classification the
policy requires, made now with the evidence on the table: **a
specification error in this freeze, on two counts**. First, the premise
("a saturating flow cannot gain from a competitor") presumed a shared
bottleneck, and the campaign's own scored cells establish the machine is
source-stack-bound (i2, i3, i4 and x4 all scale aggregate with stack
count), so a competitor need not take anything from the established flow
and the premise does not describe this machine. Second, the bar compares
two 20-second steady windows of a rate process whose documented
variability exceeds the 5 percent headroom: the solo stream's 1-second
bins span 4.245 to 5.562 GB/s, and flow 0's stage-0 window caught a 3.6
to 3.7 GB/s excursion in the final seconds before the first join (its
earlier stage-0 bins sat near 4.4), depressing the denominator. The
age-ramp alternative was checked and refuted: the s1 control shows a
60-second-old flow already at steady rate (5.026 GB/s at age 40 to 60
against 5.085 at 100 to 120), so the rise is not TCP ramp. The machine
fact worth keeping: the established flow ran fastest with two
competitors present (4.846 GB/s stage-2 steady against 4.146 solo
steady), so the harm the relations guarded against, competitors stealing
throughput from an established flow, did not occur in any direction;
what failed is the freeze's assumption that a socket flow's rate is
stationary enough for a 5 percent bar on 20-second windows. The
post-specified j2x cell flagged exactly this before j3 ran (unscored
ratio 1.054), and the pre-run revision of this paragraph, preserved in
history, recorded that warning without pre-committing the
classification.

## Descriptive statistics (the dataset)

### s1-stream (job 195728, gpu105 to gpu102, 300 s)

178,411 chunks, 1,496.6 GB. Steady goodput excluding the first 10 s:
4.991 GB/s (20.0 percent of one port). Per-chunk completion deltas (us):
mean 1681.5, sd 309.8, p5 1475.8, p50 1598.4, p95 2112.6, p99 2860.0, max
21008.1. Every 1-second bin between 4.245 and 5.562 GB/s. Tracer floors
14.337 (dest) and 13.615 us (source) p50. Source tx: 1506.19 GB on one hsn
port; destination ACK egress split across two ports matching NCCL's two
channels.

This pair's 300-second rate sits 28 to 34 percent above the discovery
ladder's 200-repetition bursts (3.72 to 3.89 GB/s) and 51 percent above
wave-16's single 128 MiB transfer (3.30 GB/s), but those burst anchors
were measured on other node pairs, and this campaign's own controls
refuse the obvious generalization: the per-source spread measured in i2
(33.9 percent between gpu105 at 4.893 and gpu103 at 3.655 GB/s) is the
same size as the claimed gain and s1 used the fast source; the
gpu103-sourced 60-second solo stage of the post-specified j2x cell lands
at 3.819 GB/s, inside the burst range; and the same-pair mixed cells
moved the opposite way, 16 to 24 percent below their burst anchors
(1.202 against 1.43 to 1.54, and 3.336 against 4.16 to 4.36 GB/s). So no
sustained-versus-burst direction is established; what stands is that
burst and sustained rates on this stack differ by tens of percent with
pair-dependent sign, and a calibration consumer must anchor per
source-destination pair and per duration rather than transfer a scalar.

### i2-incast (job 195729, gpu103 and gpu105 into gpu102, 180 s)

| Flow | Source | Chunks | Bytes | delta p50 us | delta p95 us | steady GB/s |
|---|---|---:|---:|---:|---:|---:|
| 0 | gpu103 | 76,697 | 643.4 GB | 2260.8 | 3214.8 | 3.655 |
| 1 | gpu105 | 102,635 | 861.0 GB | 1603.4 | 2443.9 | 4.893 |

Aggregate steady 8.548 GB/s, 1.71 times the solo stream: **a second
source-side stack nearly doubles what one pushes**, because each flow
rides its own source stack, its own wire path and its own destination port
(NCCL device lines show the two flows on different destination hsn
devices). Jain 0.9795 over the steady window. Convergence under the frozen
definition took 119 seconds: the flows' 1-second bins kept excursions
beyond 25 percent of their final steady values for most of the window's
first two thirds, so the slow settling of socket flows into their sharing
allocation is itself a measured, reportable transient. The per-source
asymmetry (gpu105 sustains 4.89 GB/s where gpu103 sustains 3.66) reappears
identically in the post-specified join cell, so source identity, not flow
count alone, shapes per-flow rates on this stack.

### i3-incast (job 195730, gpu102, gpu104 and gpu105 into gpu101, 180 s)

| Flow | Source | Chunks | delta p50 us | delta p95 us | steady GB/s |
|---|---|---:|---:|---:|---:|
| 0 | gpu102 | 73,316 | 2492 | 2578 | 3.425 |
| 1 | gpu104 | 70,101 | 2496 | 2616 | 3.262 |
| 2 | gpu105 | 73,537 | 2493 | 2749 | 3.429 |

Aggregate steady 10.115 GB/s (2.027 times s1), Jain 0.9995, convergence
43 seconds. Each source rode one hsn port (614.3, 586.5 and 620.1 GB tx
respectively, header-exact). Three near-identical per-flow rates on a
cell whose destination differs from i2's: at degree 3 the flows share
tightly and fairly.

### i4-incast (job 195734, every other A100 node into gpu101, 180 s)

| Flow | Source | Chunks | delta p50 us | delta p95 us | steady GB/s |
|---|---|---:|---:|---:|---:|
| 0 | gpu102 | 45,874 | 3998 | 4184 | 2.094 |
| 1 | gpu103 | 45,846 | 3961 | 4769 | 2.093 |
| 2 | gpu104 | 45,358 | 4007 | 4183 | 2.073 |
| 3 | gpu105 | 45,817 | 3998 | 4323 | 2.094 |

Aggregate steady 8.354 GB/s at Jain 1.0000 (per-flow rates within 1
percent of each other), convergence 151 seconds. **The incast ladder is
non-monotone in degree**: 4.99, 8.55, 10.12, 8.35 GB/s at degrees 1
through 4. Degree 4 falls 17 percent below degree 3 while fairness goes
exactly flat, and settling slows (43 seconds at degree 3, 151 at degree
4): the destination side begins to bind somewhere between three and four
source stacks, and it binds fairly. The ceiling analysis says which side
it is not: 8.35 GB/s sits at 31 percent of the destination's PCIe bound
and 33 percent of one port, so neither wire nor PCIe saturates; the
onset is in the host stack. This is a descriptive observation for the
calibration consumer, not a model claim.

### j3-join (job 195731, three staged joins into gpu101, 240 s)

Flow 0 (gpu102) established at T0; flow 1 (gpu104) joins at 60 s; flow 2
(gpu105) joins at 120 s.

| Stage | Active | Aggregate steady GB/s | Flow 0 | Flow 1 | Flow 2 | Jain | Convergence |
|---|---|---:|---:|---:|---:|---:|---:|
| 0: [0, 60) | 0 | 4.146 | 4.146 | | | 1.000 | 0 s |
| 1: [60, 120) | 0, 1 | 7.725 | 4.448 | 3.277 | | 0.978 | 1 s |
| 2: [120, 240) | 0, 1, 2 | 13.347 | 4.846 | 3.540 | 4.961 | 0.980 | 0 s |

275,285 chunks. Every join settled within one second under the frozen
convergence definition, reproducing at degree 3 what the post-specified
j2x cell showed at degree 2: staggered joins on pre-established
connections are near-instantaneous and non-disruptive, in sharp contrast
to the simultaneous-start settling of the incast cells (43 to 151
seconds). The stage-2 aggregate (13.347 GB/s) is the campaign's largest,
and E-J-4 passed at 1.863 against its 0.7 floor: joins grew the
aggregate rather than collapsing it. The two failed established-flow
bars are classified above.

### x4-incast (job 195735, four gpu102 GPUs into one gpu105 GPU, 180 s)

| Flow | Source GPU | Chunks | delta p50 us | delta p95 us | steady GB/s |
|---|---|---:|---:|---:|---:|
| 0 | gpu102 device 0 | 32,363 | 5139 | 8973 | 1.684 |
| 1 | gpu102 device 1 | 69,793 | 2124 | 4949 | 3.231 |
| 2 | gpu102 device 2 | 70,364 | 2176 | 4973 | 3.222 |
| 3 | gpu102 device 3 | 67,545 | 2135 | 3785 | 2.958 |

Aggregate steady 11.095 GB/s, the largest of the campaign, from four
same-node source stacks whose combined traffic left over **one** hsn port
(2014.25 GB tx on gpu102's hsn2 against 2013.6 GB of payload; the port ran
at 44 percent of its 25.0 GB/s rate). Four stacks on one shared port
out-push two stacks on two ports (i2 at 8.55) and one stack (s1 at 4.99),
so aggregate goodput scales with the number of host stacks, not with port
count, which is the sharpest statement of the stack-bound regime this
dataset makes. Jain 0.9496: flow 0, whose GPU sits furthest from the
selected port, sustains 1.68 GB/s against 3.0 to 3.2 for the other three,
so GPU-to-port NUMA distance shapes per-flow rates within one node.
Convergence under the frozen definition was not reached within the window
for at least one flow (reported "not converged within stage"), the flows'
final-20-second steady values being the reference. This cell landed after
the first publication; its relations were scored by the unchanged frozen
analyzer and folded in through the disclosed re-packaging.

### mx-pair (job 195732, gpu104 plus gpu003, 60 s per direction)

| Direction | Chunks | Steady GB/s | delta p50 us | delta p95 us | delta max us |
|---|---:|---:|---:|---:|---:|
| GH200 to A100 (`mx-gh2a`) | 8,627 | 1.202 | 6855.4 | 7796.6 | 8646.5 |
| A100 to GH200 (`mx-a2gh`) | 23,858 | 3.336 | 2504.9 | 2574.4 | 19716.0 |

The direction asymmetry (2.77x) reproduces the discovery observation in a
longer window and passes its signed relation. The A100-sourced direction
sits near the A100-to-A100 burst rate while the GH200-sourced direction is
the slow leg: the sender-side host stack is the binding term, consistent
with wave-16's port-versus-stack separation. Each direction rode one
sender hsn port (72.82 and 201.35 GB tx, exact to the header model).

### j2x-join (job 195764, post-specified, unscored, 180 s)

Flow 0 (gpu103) established at T0; flow 1 (gpu105) joins at 60 s.

| Stage | Active | Aggregate steady GB/s | Flow 0 steady | Flow 1 steady | Jain | Convergence |
|---|---|---:|---:|---:|---:|---:|
| 0: [0, 60) | flow 0 | 3.819 | 3.819 | | 1.000 | 0 s |
| 1: [60, 180) | 0 and 1 | 8.873 | 4.025 | 4.849 | 0.991 | 0 s |

The established flow lost nothing when the competitor joined (its bins
around the join sit flat at 3.7 to 4.1 GB/s), and the aggregate roughly
doubled within the join's first 1-second bin. On this source-stack-bound
transport, a data-plane join on a pre-established connection is
near-instantaneous and non-disruptive at degree 2, in sharp contrast to
the 119-second settling the simultaneous-start incast cell showed: **how
flows arrive, not only how many there are, shapes the transient**. This
cell is post-specified: no scored relation reads it.

## Physical sanity review

Three independent framings, per the local rules.

**Network and serialization physics.** Every rate is far under its
ceiling: s1 at 4.99 GB/s is 20 percent of one 25.0 GB/s port; the i2
aggregate at 8.55 GB/s is 32 percent of the destination's PCIe H2D bound;
the x4 aggregate at 11.10 GB/s is 44 percent of its single shared source
port and 41 percent of the destination PCIe bound; the mixed directions
are 4.8 and 13.3 percent of a port. Chunk arithmetic is self-consistent
everywhere: 8 MiB over 4.991 GB/s is 1681 us against a measured mean
delta of 1681.5 us. Sender tx counters conserve payload within the
MTU-9000 header model in every completed cell (0.6 percent on the
one-flow cells; 0.03 percent recorded on x4, where the counter sits
slightly under payload plus nominal headers and is reported as recorded).

**Host and transport physics.** The tracer contributes 13.2 to 14.5 us p50
per chunk, 0.20 to 0.90 percent of realized chunk service in every
captured cell, so the series is transport signal, not tracer noise. The
wire floor for an 8 MiB chunk is 335.5 us; measured chunk service sits 4.8
to 20.4 times above it: the kernel socket stack, not the port, binds every
path, and adding a second source-side stack (i2) nearly doubles the
aggregate while a single stack cannot, which is the same conclusion
wave-16 reached from its four-port arm by an independent route.

**End-to-end plausibility.** Every solo and shared rate sits between the
socket-stack readings wave-16 anchored and the port rate, and the i2
aggregate (8.55 GB/s) sits below twice the best single source (2 x 4.99)
and above what the weaker source's solo rate alone would give, matching
per-source stacks with mild interference. Sustained-versus-burst
differences are real but pair-dependent in sign (above the cross-pair
anchors for s1, below the same-pair anchors for both mixed directions), so
they are reported as observations, not as a law. The mixed asymmetry's
slow leg is the Grace sender stack, not the fabric: the same two endpoints
carry 2.8 times more the other way.

## The calibration reference and its byte locks

The wave-19 calibration comparison consumes exactly the files under
`dataset/`, whose per-file SHA-256 and byte sizes are pinned in
[dataset/MANIFEST.json](dataset/MANIFEST.json), itself locked at SHA-256
`780fec5dac34a21199685a6333120e67fe39007c58b8cc53eb09b521a76f6e18` and
enforced in CI by `tests/test_merlin_fabric_flow_dataset.py`, which
verifies every tracked file's hash and size, rejects unmanifested files,
and carries a mutation-sensitive negative control.

Three disclosed re-packagings, each with the manifest transition stated.
First, `dd45890c...` (commit `ab8776d`) to `a6b7e61e...` (review fix
round): exactly five files changed, all under `stats/` (relative job
identity instead of a machine-local absolute path, a `hosts` field naming
both het components, seven re-derived guard keys), 41 evidence files
added, nothing removed. Second, `a6b7e61e...` to `67f898a0...` (the x4
cell landing): only `stats/relations.json` changed (E-I-7, E-T-1 and
E-I-5 updated by the unchanged frozen analyzer), 40 x4 files added
(series, summary, rank metadata, guard evidence, stats), nothing removed.
Third, `67f898a0...` to `780fec5d...` (the tranche-2 landing of
2026-08-18 evening): only `stats/relations.json` changed among existing
files, and 100 files were added: the i3-incast, i4-incast and j3-join
cells (22, 26 and 22 tracked files each, plus their stats and their 10
bulk-side source series), 16 discovery files (the two four-node
pair-matrix samples, 195694 and 195737, added beside the frozen
two-node first sample 195692, each in its own directory, nothing
overwritten), and one manifest-version snapshot. Every staged input was
digest-verified against the Merlin originals before packaging (231
files, zero mismatches). The snapshot,
`manifest_versions/67f898a0...json`, preserves the tranche-1 manifest
byte-exact (its own SHA-256 equals its name): the
merlin_ss_fabric_loadbearing_v1 study froze that digest as its run-time
dataset identity, and its repository test previously pinned the living
manifest to it, which would have forbidden this dataset from ever
growing through its own committed fold-in protocol. That test now
verifies the frozen claim exactly: the consumed manifest version exists
byte-exact in-tree and every file that study's frozen analyzer consumes
still hashes to that version's entries. The loadbearing analyzer itself
is untouched, its consumed files are byte-unchanged, and this
accommodation is flagged for that study's owner to ratify.
Across both transitions **every previously published series and metadata
byte is unchanged**, verifiable by comparing the manifests across the
three commits.

- **Repo-tracked reference (204 manifest-listed files, plus the
  manifest itself):** per-cell destination-side per-chunk series
  (`*_dest.csv.gz`, deterministic gzip), cell summaries, per-rank metadata
  (hosts, roles, T0 clocks, tracer floors), the mixed-pair matrix probe,
  the analyzer's statistics and relation verdicts (`stats/`), and the
  guard-evidence set the verdicts re-derive from: per-rank
  `guards_before/after` files, per-side `side.txt` for the mx cells,
  `transport_summary.txt`, NCCL interface-selection lines, per-port
  sender-side tx deltas, established-socket samples for the homogeneous
  cells (the mx job ran no 1 Hz sampler, a disclosed gap), and each job's
  in-run `source.sha256`. Running
  `analyze_capture.py --dataset-root dataset/` re-derives all seven guard
  verdicts and all 18 relation rows from the tracked tree alone.
- **Bulk-side mirror (21 files, hashed in the manifest under
  `bulk/`):** the twenty-one source-side series (`*_src.csv.gz`), one per
  flow;
  nothing else is hashed under `bulk/`. They live on the site storage root
  (see `docs/architecture.md`) and, as raw CSVs, in the Merlin capture
  tree under the per-job directories named in this record.
- The evidence files are verbatim run captures and keep the site paths
  they were recorded with, byte-identical to the raw tree for integrity;
  the generated `stats/` files carry no absolute path. The per-rank
  metadata records each het component's own `SLURM_JOB_NODELIST` (gpu104
  on the A100 side, gpu003 on the GH side), which is that component's
  truth under Slurm's het-job semantics; the stats `hosts` field carries
  the full pair.
- The series schema is one row per chunk:
  `cell,flow,side,chunk_idx,chunk_bytes,t_ns_since_t0`, with T0 defined as
  the world-barrier exit on the recording rank and both `CLOCK_REALTIME`
  and `CLOCK_MONOTONIC_RAW` epochs recorded per rank in the rank metadata.

Anything a void run would have produced is absent by construction: no
captured cell was void.

## What stays open

- **TRAF-51** registers the wave-19 calibration comparison: an htsim
  Slingshot-class fabric instance versus this dataset on the incast and
  join families, with the socket host-stack floor separated from fabric
  serialization so the fabric model is not fitted to host-stack behavior.
- **TRAF-52** now covers only the GH200-to-GH200 family (freeze 2, gated
  on the gh-2n jitter ladder, job 195700, still pending). Every frozen
  A100 cell has been captured, scored against the unchanged bands and
  folded in: x4 first, then i3, i4 and j3 in the tranche-2 landing, each
  through the same packaging protocol with its manifest transition
  disclosed and no band moved.
- The wave-16 leftover job 195649 remains queued and belongs to that
  study's reproduction path, untouched by this one.
