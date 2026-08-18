# Merlin fabric flow capture v1 results

The reviewed state is **PARTIAL by queue, clean by evidence: 11 of 18
frozen relations evaluated, all 11 pass, every fatal guard held in every
captured cell, and 7 relations unevaluated because the psicourse
reservations pin the four-and-five-node A100 allocations until 2026-08-19
08:00**. The study publishes the byte-locked reference dataset the wave-19
Slingshot calibration comparison will consume: 630,293 verified 8 MiB
chunks across the scored cells (782,825 including the post-specified join
cell), each with a destination-clock completion timestamp, over one solo
stream, two incast cells (three-node degree 2 and the two-node
four-GPU-source shape), one mixed-architecture pair in both directions,
and one post-specified two-flow join. This study makes no fabric-model
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

The all-pair matrix jobs (195694 pinned, 195737 unpinned, four nodes each)
never scheduled inside the window and remain queued; the pairs above are
what the discovery reached. Pairs never allocated are reported as never
allocated, not interpolated.

## Cells and runs

| Cell | Job | Nodes x GPUs | Elapsed | State |
|---|---|---|---:|---|
| `mx-pair` (matrix, gh2a, a2gh) | 195732 | het 1 A100 (gpu104) + 1 GH200 (gpu003) | 00:02:18 | complete, rc 0 |
| `s1-stream` | 195728 | 2 x 1 (gpu102 dest, gpu105 source) | 00:05:14 | complete, rc 0 |
| `i2-incast` | 195729 | 3 x 1 (gpu102 dest; gpu103, gpu105 sources) | 00:03:14 | complete, rc 0 |
| `j2x-join` post-specified | 195764 | 3 x 1 (gpu102 dest; gpu103, gpu105 sources) | 00:03:12 | complete, rc 0 |
| `i3-incast` | 195730 | 4 x 1 | | **queued, never ran** (estimated 2026-08-19 09:10) |
| `j3-join` | 195731 | 4 x 1 | | **queued, never ran** (estimated 2026-08-19 09:40) |
| `i4-incast` best effort | 195734 | 5 x 1 | | **queued, never ran** (estimated 2026-08-19 08:00) |
| `x4-incast` best effort | 195735 | 2 x 4 (gpu102 sources, gpu105 dest) | 00:03:17 | complete, rc 0; landed overnight after the first publication and was folded in by the disclosed follow-up packaging |

The blocking mechanism is structural, not luck: the two psicourse
reservations hold two A100 nodes and two GH200 nodes until 2026-08-19
08:00, the FLEX reservation floats onto whichever A100 nodes go idle, and
a four-or-five-node 1-GPU allocation needs more simultaneous free nodes
than the three that remain. Reproduction is mechanical: the queued jobs
were submitted from the committed harness; when any of them lands, its
outputs drop into the same capture tree and `analyze_capture.py` scores it
against the same frozen bands with no code change. **Nothing about the
missing cells is estimated, extrapolated or inferred**; their relations
are unevaluated, never passed and never failed.

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
- **G2 clock and timer sanity.** Every one of the 542,760 recorded
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

## Scored relations, 10 of 18 evaluated

| Relation | Band | Measured | Verdict |
|---|---|---|---|
| E-T-1 tracer discipline, every frozen captured cell | floor p50 under 1.0 percent and spread under 0.3 percent of median chunk service | s1 0.90/0.06, i2 0.64/0.09, x4 0.63/0.09 (worst rank), mx-gh2a 0.20/0.03, mx-a2gh 0.55/0.02 percent | pass |
| E-S-1 s1 steady goodput | [2.0, 5.5] GB/s | 4.991 GB/s | pass |
| E-S-2 s1 per-chunk p95 over p50 | at most 1.5 | 1.322 | pass |
| E-I-1 i2 aggregate steady | [1.8, 9.0] GB/s | 8.548 GB/s | pass, 95 percent of the upper edge |
| E-I-2 i2 aggregate over s1 steady | [0.9, 2.2] | 1.713 | pass |
| E-I-3 i3 aggregate steady | [1.8, 12.0] GB/s | | unevaluated |
| E-I-4 i3 aggregate over s1 | [0.9, 3.2] | | unevaluated |
| E-I-5 Jain in every incast cell that runs | at least 0.6 | i2: 0.9795, x4: 0.9496 | pass |
| E-I-6 i4 aggregate steady | [1.8, 14.0] GB/s | | unevaluated |
| E-I-7 x4 aggregate steady | [1.8, 12.0] GB/s | 11.095 GB/s | pass, 92 percent of the upper edge |
| E-J-1 j3 flow-0 stage-0 over s1 | [0.7, 1.3] | | unevaluated |
| E-J-2 j3 flow-0 stage-1 over stage-0 | at most 1.05 | | unevaluated |
| E-J-3 j3 flow-0 stage-2 over stage-1 | at most 1.05 | | unevaluated |
| E-J-4 j3 aggregate floor after joins | at least 0.7 x stage 0 | | unevaluated |
| E-M-1 gh-to-a100 steady | [0.7, 3.5] GB/s | 1.202 GB/s | pass |
| E-M-2 a100-to-gh steady | [2.0, 8.0] GB/s | 3.336 GB/s | pass |
| E-M-3 mixed 8 B one-way | [8, 200] us | 24.256 us | pass |
| E-M-4 signed: a100-to-gh at least 1.3 x gh-to-a100 | at least 1.3 | 2.775 | pass |

The freeze's independence disclosure stands: the 18 relations read about
13 independent quantities, and the 11 evaluated ones read about 9. Three
qualifications on the passing set. E-T-1 and E-I-5 are quantified over
"every cell that runs", so their passes are provisional: E-T-1 currently
covers five frozen cells and E-I-5 two incast cells (both re-evaluated
when x4 landed, both still passing), and they are re-evaluated again if
any still-queued cell lands. E-M-3's band [8, 200] us was wide
enough that failure was implausible short of a harness defect: its lower
edge is 1.6 times this study's own stated floor and its upper edge 8
times the largest observation in hand at freeze time, so its pass carries
little evidential weight and is counted with that caveat. E-I-2 divides
an aggregate defined on the final-20-second stage window by the s1 steady
defined on the whole window excluding the first 10 seconds, as the freeze
wrote them; with both sides on the final-20-second definition the ratio
is 1.7157 against the published 1.7129, and the verdict is unchanged
under either reading.

**A warning the post-specified cell raises about E-J-2.** In j2x (unscored)
the established flow's post-join steady was 1.054 times its solo steady,
just past the 1.05 edge E-J-2 freezes for j3. The relation's reasoning ("a
saturating flow cannot gain from a competitor") assumed a shared
bottleneck, and the post-specified observation is source-stack-bound
behavior in which the established flow need not lose anything. If j3 lands
and E-J-2 misses, the freeze's failure policy applies as written: the miss
is reported as refuted with a statement of whether it is a specification
error of the freeze or a fact about the machine, weighing this evidence,
and the classification is made then, not pre-committed from a
post-specified measurement now. Either way the band is not widened.
Recorded before j3 has run so the option is not invented after a miss.

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
`67f898a04dd0e6787d4d50d6139f99f3c507963c6c937a9505434cf2d9dca002` and
enforced in CI by `tests/test_merlin_fabric_flow_dataset.py`, which
verifies every tracked file's hash and size, rejects unmanifested files,
and carries a mutation-sensitive negative control.

Two disclosed re-packagings, each with the manifest transition stated.
First, `dd45890c...` (commit `ab8776d`) to `a6b7e61e...` (review fix
round): exactly five files changed, all under `stats/` (relative job
identity instead of a machine-local absolute path, a `hosts` field naming
both het components, seven re-derived guard keys), 41 evidence files
added, nothing removed. Second, `a6b7e61e...` to `67f898a0...` (the x4
cell landing): only `stats/relations.json` changed (E-I-7, E-T-1 and
E-I-5 updated by the unchanged frozen analyzer), 40 x4 files added
(series, summary, rank metadata, guard evidence, stats), nothing removed.
Across both transitions **every previously published series and metadata
byte is unchanged**, verifiable by comparing the manifests across the
three commits.

- **Repo-tracked reference (114 manifest-listed files, 6.87 MB, plus the
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
- **Bulk-side mirror (11 files, hashed in the manifest under
  `bulk/`):** the eleven source-side series (`*_src.csv.gz`), one per flow;
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
- **TRAF-52** registers the families this window could not run: the
  GH200-to-GH200 cells (freeze 2, gated on the gh-2n jitter ladder, job
  195700) and the still-queued A100 cells i3/j3/i4 (jobs 195730, 195731,
  195734), all reproducible from the committed harness, scoreable against
  the frozen bands with no code change, and foldable into the dataset
  only through a follow-up packaging commit that moves no band. The x4
  cell (195735) demonstrated the protocol: it landed after the first
  publication and was scored and folded in exactly that way.
- The wave-16 leftover job 195649 remains queued and belongs to that
  study's reproduction path, untouched by this one.
