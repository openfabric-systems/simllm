# Merlin ss-dragonfly load-bearing recalibration v1 expectations (TRAF-51)

## Freeze scope and chronology

This is the expectations-only record for the wave-21 load-bearing
recalibration of the hosted Slingshot-class fabric, registered under
TRAF-51's narrowed clause: rerun the calibration comparison with the
fabric genuinely load-bearing, using the pinned load harness (htsim
submodule `1dcbfec36a33753bf978cf6323bade1a6645fe4f`: paced sources at
declared sub-line-rate offered load, closed-loop sources with the
declared endpoint think-time seam, explicit distinct-destination-port
multi-flow cells) against the byte-locked Merlin capture dataset of
[merlin_fabric_flow_capture_v1](../merlin_fabric_flow_capture_v1/RESULTS.md)
(manifest SHA-256
`67f898a04dd0e6787d4d50d6139f99f3c507963c6c937a9505434cf2d9dca002`,
the post-x4 state). It is committed before any `htsim_ss_dragonfly`
load cell of this study has run.

The wave-19 lesson this freeze encodes, stated up front: that study's
conditional rows were invariant to every simulation outcome (the fabric
latency term cancelled algebraically, and the min() coupling had 4.97x
to 6.48x headroom), so its evidence validated the composition rule and
the instance arithmetic, never the fabric under load. Here the
composition happens INSIDE the simulator: the measured endpoint floors
enter as closed-loop think times, the fabric's sharing behavior sets
the waits, and the composed quantity is the simulator's own output. For
every composed or conditional row this freeze states what the simulator
could do, within the registered guards, to change the verdict; a row
where the answer is nothing is labeled consistency-only and unscored.

Chronology, disclosed exactly, on one clock (this workstation's clock,
as recorded in the commits and the run manifest):

- The measured side of every comparison is published and was read
  before this freeze: the capture dataset is byte-locked in this
  repository, and the wave-20 discrimination experiment's archived run
  artifacts (bulk directory of the backend study) were read and hashed.
  Nothing measured is blind. The genuinely blind quantities are the
  load-bearing simulated outcomes: the x4 closed-loop waits and
  aggregate, the 4 MiB fault behavior, the cross-buffer byte
  identities on the new instances, and the 32 MiB saturating-arm drop
  timing. The two reproduction rows (EX-1, EX-2) have known expected
  values from the wave-20 record; their risk is reproduction at the
  pin with this build and these instance files, stated as exactly
  that, not as blind prediction.
- Pre-freeze work, all unscored: the htsim build at the pin
  (`scripts/build_htsim.sh`, Release, tests on; 468 of 468 ctest cases
  pass, which includes the load-harness fixtures F7 to F12 and the
  frozen legacy byte locks), the binary SHA-256 recorded below, the
  dataset re-derivations quoted below (chunk counts and p50s from the
  tracked series bytes, verified against the published stats to full
  precision), and the fluid napkin model committed beside this file
  (`napkin_x4_fluid.py`), whose outputs are quoted below. No
  invocation of `htsim_ss_dragonfly` outside the pinned ctest suite
  has happened, and no cell of this study has run in any form.
- The wave-19 and wave-20 frozen artifacts cited here (anchor floors,
  archived CSV hashes, registered values) are published byte-locked
  records; using them as inputs and reproduction targets is disclosed,
  and none of them is re-scored as new evidence except through the
  explicitly registered reproduction rows.

No job of this study touches the Merlin cluster, no framework runs,
and no TTFT or TPOT is reported. The rnic-ss endpoint claim does not
move: every cell drives the `htsim_ss_dragonfly` fabric harness.

## The load-bearing composition, stated before any simulated number

The wave-19 composition rule separated each captured pair's chunk life
into a measured endpoint host-stack floor and a fabric term, composed
outside the simulator. This study feeds the same separation through
the load harness's think-time seam: a closed-loop source injects one
8 MiB chunk greedily at line rate, waits for the chunk's delivery plus
a declared think time, and repeats. With think time Z and idle-fabric
chunk completion F = 340,417,280 ps, the uncontended cadence is
exactly F + Z (backend fixtures F9 and F10). Under sharing, the
simulator inserts its own waits W, and the realized cadence is
F + W + Z with W entirely simulator-owned. That W is the fabric term
genuinely carrying risk in this study.

Two think-time derivations are frozen for the x4 family, and the pair
is the diagnosis instrument:

- Rate-derived floors (the scored reproduction arm):
  `Z_i = round(20e12 / n_i) - F`, where n_i is flow i's measured
  chunk-completion count in the capture's final-20-second stage
  window. The zero-wait composed aggregate then equals the measured
  aggregate by construction, so the registered band bounds the
  simulated sharing inflation and nothing else. This entailment is
  disclosed: the row scores the fabric's sharing behavior, not the
  endpoint floors.
- p50-derived floors (the skew-diagnosis arm): `Z'_i = p50_i - F`
  with p50_i the flow's measured successive-delta median, the same
  functional the wave-19 anchors froze. The measured x4 delta
  distributions are right-skewed (mean over median 1.082 to 1.248),
  so the zero-wait composed aggregate overshoots the measured
  aggregate by 20.5 percent; the registered signed row freezes that
  overshoot as the prediction. No static think time can reproduce
  both the mean and the median of a skewed cadence; the two arms
  bracket that structural gap, and the gap is TRAF-53 evidence, not a
  fabric miss.

Per-width and per-pair constant discipline (the E-A-7 finding: a
transferred constant that is within 2.6 percent at width 8 by error
cancellation is 2.976x off at width 2): every think time in this study
is declared per flow with provenance from that flow's own measured
cell, in the tables below. Nothing is transferred across pairs,
stacks, widths, or families. A cell whose per-flow floors lack
measured provenance fails closed and is not run.

## The declared instances

Three committed instances, identical in every parameter except the
shared buffer; the analyzer re-hashes all three against the tracked
bytes at scoring time:

| Instance | File | Shared buffer |
|---|---|---|
| v1 | `../merlin_ss_fabric_calibration_v1/merlin_a100_singleswitch_v1.topo` (SHA-256 `c0a0050b074385b7daa8b9fa83619dfd49ec1bcbf21fd82447b6029d6cff8f18`) | 4,194,304 B |
| x4buf32 | [merlin_a100_singleswitch_v1_x4buf32.topo](merlin_a100_singleswitch_v1_x4buf32.topo) | 33,554,432 B |
| x4buf64 | [merlin_a100_singleswitch_v1_x4buf64.topo](merlin_a100_singleswitch_v1_x4buf64.topo) | 67,108,864 B |

v1's parameter block is byte-identical to the wave-20 experiment's
configuration A (`merlin_shape_buffer4mib.topo`), including the
routing seed, which is why exact reproduction rows against the
archived wave-20 artifacts are registerable at all. The x4buf32
instance exists because the closed-loop abstraction has no loss
recovery, so its validity requires no drops: four closed-loop flows
through one shared egress hold at most one 938-packet chunk in flight
each, bounding worst-case shared-buffer occupancy by
3 * 938 * 9038 = 25,432,932 bytes plus packet granularity, and the
4 MiB default cannot absorb even a single two-burst overlap beyond
167.8 us. The declared 32 MiB clears the bound by more than 8 MiB and
cannot tune any scored number (no reachable occupancy touches it),
which ST-1 checks against the 64 MiB negative control. The real
transport carried the captured x4 traffic with TCP loss recovery
absorbing any drops invisibly; no statement here is about the buffer
the Merlin switch actually has.

## The x4 mapping, disclosed

The captured x4 cell is four same-node source stacks (gpu102's four
GPUs) whose combined traffic left one hsn port (gpu102 hsn2, host 6 in
the instance numbering) and arrived at one gpu105 GPU through one hsn
port (gpu105 hsn3, host 19): one shared 25 GB/s source egress, one
shared 25 GB/s destination ingress. The harness cannot express four
flows on one (source, destination) host pair: the delivery dispatch
routes by that pair and rejects duplicates, and a one-source-host,
four-destination-host mirror would share the 64-packet host injection
queue, whose overflow is a fatal closed-loop error within tens of
microseconds of any burst overlap. The frozen mirror therefore places
the four stacks on four distinct model source hosts (gpu102's four
ports, hosts 4 to 7) sending to ONE shared model destination host
(host 19), so the single shared 25 GB/s stage is the switch-to-host-19
egress. What this preserves: one 25 GB/s stage carries the whole
aggregate, chunk bursts interleave at that stage under round-robin
VoQ grants, and per-stack cadences ride their own think times. What it
moves: the sharing point sits at the destination egress where the
capture shared the source port, so burst overlap materializes as
switch-buffer occupancy (up to the 25.4 MB bound) where the real host
serialized bursts into the port under TCP backpressure. This is a
declared abstraction with its cost stated; the inexpressible
source-shared shape is registered as a backend follow-up at closure.

## Napkin bounds, before any simulated value is read

| Quantity | Derivation | Value |
|---|---|---:|
| shared-egress payload ceiling | C_p = 200e9 * 8948 / 9038 bits per second | 24.7511 GB/s |
| measured x4 aggregate over C_p | 11.0951923712 / 24.7511 | 44.8 percent |
| chunk wire bytes | 938 * 9038 | 8,477,644 B |
| burst arrival window | 938 * 361,520 ps | 339.106 us |
| zero-wait composed aggregate, rate arm | sum B / (F + Z_i) | 11.09519237 GB/s (ratio 1.0000000) |
| zero-wait composed aggregate, p50 arm | sum B / (F + Z'_i) | 13.36716204 GB/s (ratio 1.20477) |
| offered egress utilization, rate arm | sum ARR / cycle_i | 44.8 percent |
| offered egress utilization, p50 arm | sum ARR / p50_i | 54.0 percent |
| pathological chunk-serializing floor | every cycle inflated by 3 full bursts (+1017.3 us) | aggregate 8.21 GB/s, ratio 0.740 |
| worst-case closed-loop occupancy | 3 chunks wire | 25,432,932 B |
| two-burst overlap that fills 4 MiB | 4,194,304 B at 25 wire B/ns | 167.8 us |
| saturating-arm net fill | 2 * 16.25 - 25 wire B/ns | 7.5 B/ns |
| saturating-arm buffer-delta drop shift | 29,360,128 B / 7.5 B/ns | 3,914,683.7 ns |
| saturating-arm drop rate after fill | 7.5 / 9038 packets per ns | 0.00082984 pkt/ns |
| 100-us bin packet quantization per destination port | (ceil(1e8 / 361,520) + 1) * 8948 | 2,487,544 B |

The fluid napkin model ([napkin_x4_fluid.py](napkin_x4_fluid.py),
committed with this freeze; water-filling equal-share service over the
round-robin grant loop, exact rational arithmetic) refines the
expectation inside those bounds. Its disclosed outputs:

| Fluid prediction | Rate arm | p50 arm |
|---|---:|---:|
| aggregate over measured, 32 MiB, window [0.5, 6.0) s | 0.95387 | 1.12680 |
| chunks per flow in window | 1069, 2012, 2006, 1852 | 1028, 2408, 2362, 2399 |
| max shared-buffer occupancy | 20,563,717 B | 23,932,860 B |
| first capacity crossing on 4 MiB | 21.398 ms | 5.319 ms |

The fluid model predicts the packet-level simulator's behavior only
approximately (packet-granular grants, discrete completions, and phase
sensitivity diverge from fluid); the registered bands are set around
the fluid points with room for that divergence, and the simulator is
scored against the bands, never against the fluid script. The
registered relation is stated before the run in the
harsh-third-party-mathematician form: sharing inflation at 44.8
percent utilization should shave a few percent (fluid says 4.6), an
order-of-magnitude larger shave means the sharing mechanism is wrong,
and an aggregate above the zero-wait ceiling is an accounting defect,
never hardware truth.

## Measured anchors and inputs (published bytes, re-derived at scoring)

x4 per-flow inputs, re-derived at freeze time from the tracked series
(`dataset/x4-incast/x4-incast_flow{0..3}_dest.csv.gz`) and equal to
the published `stats/x4-incast_stats.json` fields to full precision.
The analyzer must re-derive all of them from the tracked bytes at
scoring time (FG-6):

| Flow | Source stack | n_i, completions in [160, 180) s | Steady rate GB/s (n_i B / 20) | delta p50 us | mean over p50 |
|---|---|---:|---:|---:|---:|
| 0 | gpu102 GPU 0 | 4,014 | 1.6835936256 | 5,138.805 | 1.0824 |
| 1 | gpu102 GPU 1 | 7,704 | 3.2312918016 | 2,123.632 | 1.2143 |
| 2 | gpu102 GPU 2 | 7,682 | 3.2220643328 | 2,175.749 | 1.1758 |
| 3 | gpu102 GPU 3 | 7,053 | 2.9582426112 | 2,134.978 | 1.2482 |

Measured aggregate: 26,453 * 8,388,608 / 20 = 11,095,192,371.2 B/s
(11.0951923712 GB/s). Measured Jain 0.9496 (derived, unscored here).

Frozen think tables (integers, picoseconds), computed by the frozen
formulas from the table above; the harness's derivation self-check
must reproduce them byte-exactly:

| Flow | Rate arm Z_i = round(20e12 / n_i) - F | p50 arm Z'_i = p50_i - F |
|---|---:|---:|
| 0 | 4,642,143,756 | 4,798,387,720 |
| 1 | 2,255,636,718 | 1,783,214,720 |
| 2 | 2,263,071,395 | 1,835,331,720 |
| 3 | 2,495,255,483 | 1,794,560,720 |

Control-cell floors (capture-shaped i2 mirror), taken verbatim from
the wave-19 freeze through the wave-20 control cell, provenance
per pair: gpu103 to gpu102 anchor floor 1,858,227,000 ps; gpu105 to
gpu102 anchor floor 1,256,438,000 ps. Width-2 pair constants used only
at their own pairs and loads.

Archived wave-20 evaluation-of-record artifacts (bulk directory of the
backend experiment), hashed at freeze time, reproduction targets for
EX-1 and EX-2:

| Artifact | SHA-256 |
|---|---|
| `discriminate_A_r1.csv` (bins) | `af1d1e043685227e502411a7f9835290055f98bdda2157b1268d14d8917099f5` |
| `control_A_r1.csv` (bins) | `7eb66e67969286438bb25290e469a15dff5cc945a6866f17d907fdd64331cb4e` |
| `control_A_r1.chunks.csv` | `1072cd8f86f589fdd21f15c1df284c8786e00dfcb96c27abda7e4cd75c6ebc96` |

Registered manifest values of those artifacts: discriminate A injected
17,980 (8,990 per flow), delivered 14,295, delivered payload
127,911,660 B, dropped 3,685, first_drop_ps 560,255,151; control A
injected = delivered = 203,546, dropped 0, chunks 91 and 126,
first_drop none. The wave-20 drop-rate napkin reproduces the archived
count exactly ((5e9 - 560,255,151) ps * 0.00082984 pkt/ns = 3,684.3,
observed 3,685), which is why the same model prices the 32 MiB arm
below.

## Cells and the run matrix

Binary: `htsim_ss_dragonfly` built from the pinned submodule
`1dcbfec36a33753bf978cf6323bade1a6645fe4f`, SHA-256
`662416918457542c4fcab18aebd99f43f00cd80b5518ed78947a65e65620ce92`,
recorded per run and re-verified before scoring. Every invocation runs
twice with identical arguments (repeat determinism guard), all runs
strictly sequential. Common flags:
`-pattern explicit -routing adaptive -wire_bytes 9038 -header_bytes 90
-bin_ps 100000000`. Bulk outputs stay outside Git; a packaging commit
locks the derived summaries.

| Cell | Instance | Source mode | Flow declarations | Duration | Role |
|---|---|---|---|---|---|
| ctrl-v1 | v1 | closed, chunk 8 MiB | `src=8,dst=5,think_ps=1858227000`; `src=16,dst=6,think_ps=1256438000` | 200 ms | EX-2 reproduction; control leg |
| ctrl-b32 | x4buf32 | same | same | 200 ms | CN-1 control identity |
| ctrl-b64 | x4buf64 | same | same | 200 ms | CN-1 control identity |
| mj2x-v1 | v1 | closed, chunk 8 MiB | flow 0 as ctrl flow 0; flow 1 `src=16,dst=6,think_ps=1256438000,start_ps=100000000000` | 300 ms | CN-2 staggered-join mirror |
| x4p-v1 | v1 | paced | `src=4,dst=19,offered_bps=13468749005`; `src=5,dst=19,offered_bps=25850334413`; `src=6,dst=19,offered_bps=25776514662`; `src=7,dst=19,offered_bps=23665940890` | 1 s | CN-3 paced mirror at measured loads |
| x4r-b32 | x4buf32 | closed, chunk 8 MiB, chunk_out | `src=4,dst=19,think_ps=4642143756`; `src=5,dst=19,think_ps=2255636718,start_ps=400000000`; `src=6,dst=19,think_ps=2263071395,start_ps=800000000`; `src=7,dst=19,think_ps=2495255483,start_ps=1200000000` | 6 s | BE-1 the load-bearing x4 aggregate |
| x4r-b64 | x4buf64 | same | same | 6 s | ST-1 insensitivity pair |
| x4r-v1 | v1 | same | same | 6 s | BE-3 expected fault (registered) |
| x4p50-b32 | x4buf32 | closed, chunk 8 MiB, chunk_out | same sources with `think_ps` 4798387720, 1783214720, 1835331720, 1794560720 and the same starts | 6 s | BE-2 skew-diagnosis arm |
| sat-v1 | v1 | paced, `-offered_bps 130000000000` | `src=8,dst=5`; `src=16,dst=5,start_ps=278092` | 5 s | EX-1 reproduction; load-bearing proof |
| sat-b32 | x4buf32 | same | same | 5 s | BE-4, BE-5 buffer-shifted drop timing |

Host mapping (host = node * 4 + port over gpu101..gpu105): hosts 4 to
7 are gpu102's four ports (the four x4 source stacks, one model port
per stack per the mapping disclosure), host 19 is gpu105 hsn3 (the
captured x4 destination port), hosts 8 and 16 are gpu103 port 0 and
gpu105 port 0, hosts 5 and 6 are gpu102 ports 1 and 2 (the control
mirror's distinct destination ports).

Duration scaling rule, frozen: the scored closed-loop window
[0.5 s, 6.0 s) holds at least 500 chunk completions per flow (fluid
predicts 1069 to 2412) and 55,000 100-us bins, and covers at least six
periods of the slowest pairwise cadence beat (flows 1 and 2, beat
about 0.91 s). The 400-us start stagger prevents only the artificial
simultaneous first-burst pileup; every later overlap is the dynamics
under test. The 200-ms and 300-ms consistency cells reuse the wave-20
control duration; the 1-s paced mirror is bounded by CSV size, not
dynamics (pacing is memoryless). Cost of scaling, stated: a 6-s window
cannot show model dynamics slower than about a second; the model's
slowest mechanism at this shape is the cadence beat structure itself,
sampled at least six times, and the endpoint dynamics the composition
holds static are out of scope by construction (TRAF-53).

The stdout comparison rule, registered up front (wave-20 correction 1
lesson): stdout comparisons tokenize on whitespace and compare token
sequences with the `topology=` token masked; bins CSV and chunk CSV
comparisons are literal byte identity. Exit status and stderr are
recorded per run and compared literally for repeat determinism.

## Fatal guards, void and never scored

Any violation voids the affected runs for the purpose of closing
anything; guards are never reported as a fraction. One survivable
exception is named below, exactly as the validation discipline
requires.

- FG-1 identity: the binary SHA-256 equals the frozen value, the
  submodule HEAD is `1dcbfec36a33753bf978cf6323bade1a6645fe4f` at run
  time, the three instance files hash to their tracked bytes (the
  analyzer re-hashes them; machine-enforced, the wave-19 correction 4
  lesson), the dataset manifest hashes to `67f898a0...`, and every
  dataset file the analyzer reads re-verifies against its manifest
  entry.
- FG-2 repeat determinism: both repeats of every invocation are
  byte-identical in bins CSV, chunk CSV where present, stdout, stderr,
  and exit status.
- FG-3 conservation: per flow, no 100-us bin's delivered payload
  exceeds 2,487,544 B; per cell, no bin's aggregate exceeds 2,487,544
  B times the cell's number of distinct destination ports (ctrl and
  mj2x cells 2, all others 1); manifest injected = delivered + dropped;
  delivered payload = delivered packets * 8,948 exactly.
- FG-4 execution: every cell except x4r-v1 exits 0 with the harness's
  quiescence validation passing and, for closed-loop cells, zero
  drops. Survivable exception, declared now: x4r-v1's registered
  EXPECTED outcome is the closed-loop drop fault (scored row BE-3);
  its nonzero exit is not a guard violation, and every other row of
  the study remains interpretable when it faults because no other row
  reads that run. If x4r-v1 instead exits 0, that is not a guard
  violation either: BE-3 simply fails as a scored row.
- FG-5 chunk integrity: in every clean closed-loop cell, chunk CSV row
  counts equal the manifest `chunks_completed` per flow, completions
  are strictly increasing per flow, and the scored x4 arms hold at
  least 500 completions per flow inside the scored window.
- FG-6 frozen-input integrity: the analyzer re-derives from the
  tracked series bytes the per-flow [160, 180) s counts (must equal
  4,014, 7,704, 7,682, 7,053), the whole-series delta p50s (must
  equal 5,138.805, 2,123.632, 2,175.749, 2,134.978 us), cross-checks
  the stats file, and recomputes both think tables by the frozen
  formulas (must equal the frozen integers).
- FG-7 seam echo: every manifest flow line echoes the declared
  think_ps, offered_bps, and start_ps of its flow exactly. This is
  the think-time-seam integrity check the diagnosis tree consults.

## Scored simulation relations

The entailment question is answered per relation: given the fatal
guards, can this relation fail, and what could the simulator do within
the guards to change it? Every row below can fail and none is implied
by a guard. Denominators: 2 exact, 5 behavioral, 1 structural,
reported per class and never summed with consistency rows.

Exact rows (registered integer and byte equalities):

- EX-1 sat-v1 reproduces the wave-20 discriminate-A evaluation of
  record: bins CSV byte-equal to the archived hash `af1d1e04...`, and
  stdout token-equal under the registered mask, which entails injected
  17,980 (8,990 per flow), delivered 14,295, dropped 3,685,
  first_drop_ps 560,255,151. Can fail: any dynamical drift between
  the pinned merge and the wave-20 branch content, any build
  sensitivity, any v1-instance divergence. Simulator coupling: total;
  every number is simulated under load with drops.
- EX-2 ctrl-v1 reproduces the wave-20 control-A evaluation of record:
  bins CSV byte-equal to `7eb66e67...`, chunk CSV byte-equal to
  `1072cd8f...`, stdout token-equal under the mask (entailing
  injected = delivered = 203,546, dropped 0, chunks 91 and 126). Can
  fail: as EX-1. Simulator coupling: total. This row doubles as the
  capture-shaped control leg of the discrimination statement.

Behavioral rows (bands and signs frozen; fluid points disclosed
above):

- BE-1 the load-bearing x4 aggregate, rate arm: on x4r-b32, chunk
  completions in [5e11, 6e12) ps times 8,388,608 B over 5.5 s,
  divided by the measured 11,095,192,371.2 B/s, lies in
  [0.90, 1.001]. Fluid point 0.954. The upper edge is structural
  (cycles cannot beat their zero-wait values; only window-edge
  quantization of at most 0.06 percent can push above 1.0), so the
  genuine risk is one-sided at the lower edge and is exactly the
  simulated sharing inflation; this asymmetry is disclosed rather
  than double-counted. Can fail: sharing waits above 10 percent of
  cycle time (the fluid predicts 4.6; a chunk-serializing egress
  gives 26 percent, ratio 0.74; starvation gives less). Simulator
  coupling: the waits are wholly simulator-owned (VoQ round-robin,
  grant cycle, burst interleave, buffer dynamics).
- BE-2 the skew-diagnosis arm: on x4p50-b32, the same aggregate ratio
  lies in [1.05, 1.21], direction registered: ABOVE 1.05. Zero-wait
  ceiling 1.20477 plus edge quantization caps the top (the upper edge
  is structural); fluid point 1.127. A pass means the p50-static
  floor overshoots the measured aggregate as predicted (no static
  floor reproduces a skewed cadence's mean and median at once,
  TRAF-53 evidence) while the fabric's sharing stays non-pathological
  at 54 percent utilization. Can fail: sharing inflation above 13
  percent pushes below 1.05 (the fabric branch); above 1.21 is
  structurally impossible within guards and would be an accounting
  defect. Simulator coupling: as BE-1.
- BE-3 the composed-level discrimination row: x4r-v1 (the identical
  composed x4 cell on the 4 MiB instance) terminates with exit
  status 2 and stderr containing the harness's registered closed-loop
  drop message ("closed-loop cell dropped packets"). The fluid model
  predicts the capacity crossing at 21.4 ms of the 6-s window; the
  packet-level instant is not registered, only the fault. Can fail:
  if packet-level phase repulsion keeps every overlap below the
  167.8-us fill threshold for the whole window, the run exits 0 and
  this row FAILS, which would be a genuine finding about the sharing
  dynamics and would withdraw the composed-level discrimination
  claim. Simulator coupling: total. Paired with FG-4 on x4r-b32 (the
  same cell, clean and in-band on 32 MiB), a pass makes the two
  configurations produce OPPOSITE registered verdicts at the composed
  level; the signed expectation is frozen here.
- BE-4 sat-b32 first drop: first_drop_ps lies in
  [4,474,000,000, 4,476,000,000]. Point prediction, disclosed:
  560,255,151 + 29,360,128 / 7.5 B per ns = 4,474,938,884 ps, exact
  threshold-difference cancellation plus packet quantization of order
  the 278-ns combined arrival gap. Can fail: any occupancy accounting
  or fill nonlinearity above 4 MiB. Simulator coupling: total.
- BE-5 sat-b32 drop count: dropped lies in [430, 442]. Point
  prediction 436 (drop rate 0.00082984 pkt/ns over the
  525,061,116 ps between predicted first drop and injection end; the
  same model reproduces wave-20's 3,685 exactly). Can fail: as BE-4.
  Simulator coupling: total.

Structural row:

- ST-1 buffer insensitivity above the closed-loop bound: x4r-b32 and
  x4r-b64 produce byte-identical bins CSV and chunk CSV, and stdout
  token-equal under the mask. Can fail: any in-flight accounting
  above the 25,432,932 B bound (more than one chunk in flight per
  flow, a loop or accounting defect) or any buffer-coupled dynamics
  below the drop threshold. Simulator coupling: total. A pass proves
  the declared 32 MiB cannot have tuned any scored number.

## Consistency-only rows, labeled and unscored

For each row the simulator-could-do answer is NOTHING within the
registered guards, for the stated structural reason; they are recorded
in RESULTS with observed values and never counted in any denominator.

- CN-1 control identity across buffers: ctrl-v1, ctrl-b32, ctrl-b64
  byte-identical (bins, chunks; stdout under the mask). At
  capture-shaped distinct-port load the shared buffer is never
  touched (occupancy stays at the in-service packet scale), so no
  buffer value can change a byte; this is the control leg of the
  discrimination statement, already demonstrated by wave-20 CT-4 on
  the same dynamics, and re-scoring a known-invariant outcome would
  inflate the denominator.
- CN-2 staggered-join mirror: mj2x-v1's flow 0 completion series
  equals its solo formula (340,417,280 + c * 2,198,644,280 ps, 137
  completions) and flow 1 equals its start-shifted solo formula (126
  completions); injected = delivered = 246,694, dropped 0. Distinct
  source and destination ports share no dynamical stage in this
  model, so cross-flow displacement is structurally zero (fixtures
  F10 and F11 pin the mechanism; wave-20 CT-3 pinned it at 91 and 126
  chunks). This restates the model-side join-unharmed premise; it
  cannot test it.
- CN-3 paced mirror at measured loads: x4p-v1 delivers exactly what
  it injects (dropped 0) with per-flow payload equal to the ceiling
  schedule's packet count times 8,948. Four smooth sub-line-rate
  streams at 44.8 percent aggregate cannot accumulate shared-buffer
  occupancy beyond packet scale, and the pacing schedule is fixture
  arithmetic (F7, F8), so delivered-equals-offered is entailed by the
  conservation guards given the mechanism.
- CN-4 BE-1's and BE-2's upper edges, as disclosed in their rows.
- CN-5 per-flow x4 composed rates and Jain: the per-flow think
  inputs already carry the measured NUMA asymmetry, so per-flow
  ordering and a near-measured Jain are entailed by the inputs plus
  small waits; reported derived-not-scored (sim Jain, per-flow
  ratios), because scoring them would re-score the inputs.
- CN-6 saturation-band and share properties of the sat cells: every
  steady-bin property of sat-v1 is entailed by EX-1's byte equality,
  and sat-b32's by BE-4/BE-5 plus conservation; the wave-20 DP-5 and
  DP-6 analogs are therefore not re-registered.
- CN-7 the wave-19 distinct-port composed families (s1, i2, j2x
  steady rates and p50s at measured loads): under this model the
  distinct-port mirrors add zero wait by construction, so every
  wave-19 conditional row would re-confirm identically; they are not
  re-run and not re-scored. The load-bearing content of this study
  lives where flows share a stage.

## Derived rows (reported, never scored)

- D-1 dropped(sat-v1) - dropped(sat-b32) in [3,243, 3,255], point
  3,248.5 (entailed by EX-1 plus BE-5; the buffer delta over the wire
  packet size, 29,360,128 / 9,038 = 3,248.5, end effects cancelling
  under the identical injection schedule).
- D-2 first_drop ordering sat-v1 before sat-b32 (entailed by EX-1
  plus BE-4's disjoint band).
- D-3 the fluid residual: |observed BE-1 ratio - 0.954| and
  |observed BE-2 ratio - 1.127|, reported for the record as the
  fluid-versus-packet gap, unscored.
- D-4 sim x4 Jain and per-flow rates (CN-5's observed values), and
  the sim convergence descriptivies of the x4 arms (settling of
  per-flow 1-s rates), reported for TRAF-53 context, unscored; the
  119-s capture transient stays out of scored scope (TRAF-53 owns
  endpoint dynamics).

## The registered two-configuration discrimination statement

Configurations v1 (4 MiB) and x4buf32 (32 MiB), identical in every
other parameter, on the SAME composed cells:

- Control leg: at capture-shaped load the pair is indistinguishable
  (CN-1 identity; EX-2 ties the control bytes to the wave-20 record).
- Composed leg: the identical composed x4 closed-loop cell yields
  OPPOSITE registered verdicts: v1 faults by the registered
  closed-loop drop signature (BE-3) while x4buf32 completes clean
  with the aggregate in the registered band (FG-4 plus BE-1).
- Saturating leg: the same paced shared-egress cell yields
  first-drop instants and drop counts in disjoint registered bands
  (EX-1 versus BE-4 and BE-5).

If those rows pass, the demonstration holds at the composed level:
two fabric configurations the capture-shaped evidence class cannot
separate produce different verdicts exactly where the fabric is
load-bearing, inside the pre-registered study. Signed expectations
frozen here; nothing claims which buffer the Merlin fabric has (the
registered abstraction has no loss recovery, the real transport does).

## Failure semantics and the diagnosis tree, pre-declared

- Any fatal guard (outside the named FG-4 exception): the affected
  runs are void, nothing closes, evidence retained and reported.
- EX-1 or EX-2 fails: content or instance drift against the wave-20
  record; the study is void for closure until the drift is diagnosed
  and reported (every other row is still evaluated and reported).
- BE-1 fails low AND BE-2 below 1.05: the fabric-sharing branch: the
  model's shared-egress waits exceed the capture-consistent envelope;
  refutation with findings, backend follow-up registered, no
  shared-port calibration language.
- BE-1 fails low AND BE-2 in band: cadence-dependent sharing anomaly;
  finding reported, no upgrade; the seam question is settled by FG-7
  (echo right and cadence wrong means fabric; echo wrong means void,
  a harness defect, per the seam branch).
- BE-1 in band AND BE-2 below 1.05: a sharp congestion knee between
  44.8 and 54.0 percent utilization; the rate-arm claim stands scoped
  to its load, the knee is reported as a fabric-model finding.
- BE-1 above 1.001 or BE-2 above 1.21: structural ceiling violated;
  accounting or freeze defect; void for closure, investigate, report.
- BE-3 fails (x4r-v1 exits 0): the composed-level discrimination
  claim is withdrawn; the phase-repulsion finding is reported with
  the run's occupancy evidence; the saturating-leg discrimination
  (EX-1, BE-4, BE-5) stands on its own if it passed; TRAF-51 keeps
  its discrimination clause open.
- BE-4 or BE-5 fails: the linear-fill model is wrong above 4 MiB;
  fabric buffer-dynamics finding; the composed-leg discrimination
  (BE-3) stands on its own if it passed.
- ST-1 fails: in-flight accounting defect; both x4 closed-loop arms
  void for closure; fix, re-freeze, rerun.
- A consistency row (CN-*) fails: the freeze's structural reasoning
  was wrong, which is itself reportable as a freeze defect; the run
  is void for closure and the defect is diagnosed, never patched
  silently.

## Late arrivals: the tranche-2 extension path, frozen

Tranche 2 (i3-incast, j3-join, i4-incast, and the GH family under the
capture study's own freeze-2) lands after the 2026-08-19 08:00
reservation lift and is collected by the other orchestrator session
into the byte-locked dataset by the capture study's published
protocol. This study freezes, now, exactly how each late cell feeds
the frozen relations, scored by the frozen analyzer with no code
change; the landing of a late cell moves no band registered here.

- Mapping rule, frozen: a late cell's flows whose recorded sender hsn
  ports and destination hsn devices are pairwise distinct form a
  distinct-port family: its mirror is consistency-only under this
  model (CN-7's structural reason) and feeds NO scored simulation
  relation of this study; its measured relations remain the capture
  study's. Any group of two or more flows sharing one recorded
  destination hsn device or one sender hsn port is a shared-egress
  group and feeds the frozen late relation below.
- R-LATE-AGG(cell, group), the frozen late relation: a closed-loop
  cell on the x4buf32 instance with one model source host per stack,
  one shared model destination host, think_i = round(20e12 / n_i) - F
  from the group's final-20-second completion counts, starts
  staggered 400 us apart, duration 6e12 ps; the simulated group
  aggregate over [5e11, 6e12) against the measured group aggregate
  sum(n_i) * B / 20 lies in [0.90, 1.001], the same band and meaning
  as BE-1. Expected tranche-2 outcomes under the rule: i3 and i4
  mirror i2's distinct-destination precedent and are expected
  consistency-only unless their captured mapping shows sharing; j3 is
  the same on its final all-active stage; the x4 family is already
  captured and scored here.
- Fail-closed clauses, frozen: a late cell without published per-flow
  series, final-20-second counts, and port-mapping evidence cannot
  feed R-LATE-AGG and is reported unevaluated. A GH cell additionally
  fails closed unless its floors derive from that GH cell itself
  (per-pair, per-width provenance; the E-A-7 rule; no A100-to-GH
  transfer). A cell whose frozen chunk size differs from 8,388,608 B
  fails closed unless N, ARR, F are re-derived by the frozen formulas
  (N = ceil(chunk / 8948), ARR = N * 361,520 ps,
  F = (N - 1) * 361,520 + 1,673,040 ps) AND the closed-loop occupancy
  bound (group size - 1) * N * 9,038 B stays below 33,554,432 B;
  otherwise the family is registered as needing a new instance under
  a new freeze.
- Mechanism, frozen at the harness commit: the analyzer exposes the
  descriptor derivation (`--derive-late-cell`) implementing exactly
  these formulas against the then-current byte-locked dataset, and
  the runner accepts the emitted descriptor; the x4 rate arm is the
  self-check: its derivation must reproduce this freeze's think table
  byte-exactly, enforced as a harness test. Late scoring lands as a
  follow-up packaging with the frozen analyzer, chronology recorded,
  no code change, no band moved.

## TRAF-51 closure rule, frozen

This study cannot fully close TRAF-51 under a genuine reading: the
119-second simultaneous-start transient needs the TRAF-53 endpoint
dynamics this composition holds static, multi-switch routing is
structurally unreachable at the declared shape, and the i3, j3, i4,
GH captures are tranche 2. Frozen in advance, the best achievable
outcome is a further honest narrowing:

- If every guard holds and all 8 scored rows pass: TRAF-51's entry is
  rewritten to state, additionally: the shared-egress x4 family's
  steady aggregate is reproduced at the composed level within the
  registered band on the declared instance, with per-stack
  rate-derived endpoint floors as declared think times and the
  sharing waits genuinely simulated; fabric-configuration
  discrimination at the composed level is demonstrated
  (capture-shaped control identical, load-bearing verdicts opposite,
  saturating-arm separations in band); and the p50-static floor is
  registered as refuted for skewed shared-port families with the
  evidence handed to TRAF-53. The claim language upgrades ONLY that
  far: solo, distinct-port incast and join families keep exactly the
  wave-19 wording, no claim about Merlin's physical buffer sizing is
  made, and the rnic-ss endpoint claim stays untouched. TRAF-51
  stays open for the remainder (endpoint dynamics, tranche-2
  families, multi-switch routing).
- If rows fail: refutation with findings per the diagnosis tree; no
  language upgrade; TRAF-51 stays open unchanged plus the findings.
- Residual registrations at closure use TRAF-61 and up here and
  HTSIM-32 and up for backend-repo follow-ups (candidates identified
  now: the same-pair multi-flow dispatch gap that makes the true
  source-shared x4 mapping inexpressible, and the hardcoded 64-packet
  host injection queue), registered in this repository's module docs
  without touching the backend repo.

## The model action, frozen

No simllm profile, envelope, arm or reported metric changes in this
study. The deliverables are: this freeze with the two new instance
files and the napkin script, the runner and analyzer, the bulk run
artifacts outside Git with a tracked locked summary (packaging commit
with manifest lock test, mutation-sensitive negative control, and
`.gitattributes` eol rules in the same change), RESULTS.md with the
per-quantity measured, simulated, composed, residual and verdict
table, and the registry edits the closure rule above authorizes.
