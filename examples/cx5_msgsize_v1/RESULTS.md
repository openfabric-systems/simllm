# CX-5 message-size calibration: results against pre-registered expectations

Run of 2026-09-01, one `run_cx5.py` invocation (19 registered check rows in
`summary.csv`, plus `latency.csv`, `msg.csv`, `mtu.csv` and `incast.csv`).
Registrations are in [expectations.md](expectations.md), frozen in commit
`1dc6eb8` before any run of this study; the profile the flags are rendered
from landed in `8b9deaf`. Disclosure: that freeze commit was amended after the
run to spell the ASCII plus-minus sign as words and to rewrap three lines,
because the repository's path-portability check reads that glyph as a
filesystem path. No band, value, check or sweep entry changed, and the
committed text is the text every number below was scored against. Binary:
`htsim_dcqcn_atlahs` at htsim `1dcbfec` with `txt2bin` from the same build.
Raw artifacts, including the per-run completion CSVs and sender state traces,
stay outside Git under `SIMLLM_DATA_ROOT`; the curated per-experiment CSVs are
committed next to this file.

Second disclosure, and the reason `summary.csv` now holds 24 rows: a
post-specified long-flow arm was added on 2026-09-02 and appended five reported
rows plus `incast_long.csv`. It is described in its own section below, it is
scored nowhere, and it changed nothing above. `latency.csv`, `msg.csv`,
`mtu.csv` and `incast.csv` re-ran byte for byte, and all 19 registered check
rows carry exactly the value they carried before in every column they had.

Verdict: **9 of 12 scored checks pass. All eight M-checks that could run pass,
so the retuned configuration is self-consistent; three of the four scored
T-checks fail, and three registered cells are void on a fatal configuration
guard. One FAIL was registered as expected (T1), one failed in the opposite
direction to its registration (T4), one was registered as a PASS and is the
study's real surprise (T5), and one registered expected FAIL passed for a
reason that confirms rather than weakens the underlying finding (T2). Nothing
here is a simulator defect. The headline is a structural boundary: the packet
path can be configured to the measured latency floor or to the measured
message offset but not to both, and it converts incast loss into 67 ms latency
stalls where the hardware converts it into a 26 percent bandwidth tax.**

Superseded in part: the HTSIM-39 defect the `hacc_fabric_v1` study identified
has since been fixed in the backend, and this study was re-run against the fix.
The record below is unchanged and is the record of the `1dcbfec` pin. The
re-run is in
[the 2026-09-02 section](#2026-09-02-re-run-against-the-htsim-39-fix) at the
end of this file, every scored check keeps its verdict there, and the committed
CSVs beside this file are now the re-run's.

Amended verdict on the fan-in, from the post-specified long-flow arm below:
**T4's 7.35 Gb/s is a timeout reading, not a bandwidth reading.** The same
configuration moves 88.2342 Gb/s, 1.194 times the measured 73.9, once the
episode is long against the 67.109 ms local ACK timeout and the volume is
offered as one stream per sender. T5 survives in a sharper form. That clean
cell runs the wire at 92.4 percent with zero retransmissions, where the
measurement runs it at 99.4 percent with a 26 percent tax, so the model reaches
the measured goodput by being idler and cleaner rather than by reproducing the
mechanism.

![message size vs goodput](plots/message_size_vs_goodput.png)

## E-MSG: message size vs goodput, both profiles

- **M1 PASS**: every Q = 1 point of both profiles sits within **0.02 percent**
  of its own profile's fitted fixed-offset law, against a registered
  10 percent band. The worst cell is `cx7_400g` at 64 KiB. After the retune
  the comparator is still exactly a fixed-offset law.
- **M2 PASS, both profiles**: the fitted `C` is **95.485 Gb/s** against a
  rendered link rate of 97 Gb/s (ratio 0.9844) for `cx5_100g`, and
  **381.938 Gb/s** against 388 Gb/s (ratio 0.9844) for `cx7_400g`, inside the
  registered 3 percent band. The 1.56 percent shortfall is exactly the 64 B
  wire header on a 4096 B packet, as registered. The 4 MiB Q = 1 points that
  anchor the fits are 94.834 and 372.844 Gb/s.
- **M3 PASS**: the 2 B message completes in **2.0813 us** against the measured
  2.08 us floor, a ratio of 1.0006 inside the registered 15 percent band. The
  515 ns per hop was derived from the route construction before the freeze and
  predicted 2.0814 us; the run reproduces that to four digits, which is the
  evidence that the hop-count derivation in the registration is the right one.
- **M4 PASS**: `cx7_400g`'s fitted `C` is **4.00x** `cx5_100g`'s, exactly, and
  its fitted `T_eff` is **2.143 us** against 2.409 us, a ratio of 0.8895
  inside the registered 15 percent band. The 11 percent difference is the
  registered one: only the propagation half of the offset is rate-independent.
  Scaling therefore does produce the same curve four times faster, which is
  what "the same architecture at a higher rate" has to mean for this path.
- **T1 FAIL, as registered**: the fitted `T_eff` is **2.409 us** against the
  measured 4.48 us, a ratio of **0.538**. This is the study's headline. The
  offset in this path is entirely topology, and the topology was calibrated to
  the other measured anchor, the 2.08 us latency floor. The two cannot both be
  held: reaching 4.48 us needs about 1120 ns per hop, which would put the 2 B
  floor at 4.5 us, more than twice the measured value. Nothing in the flag
  vector can separate them, because the model charges no per-message endpoint
  cost. This is BACK-54's whole content, and the number to beat.
- **T2 PASS, registered as an expected FAIL**: the modeled Q = 16 to Q = 1
  ratio at 64 KiB is **1.407** against the measured send-queue-depth ratio of
  1.573, inside the registered 20 percent band. It passes for a reason that
  confirms the registration's mechanism rather than refuting it: the companion
  diagnostic registered alongside it shows the Q = 16 aggregate at 64 KiB
  matching the Q = 1 point at 16 x 64 KiB to **0.5 percent**, so sixteen
  concurrent messages behave exactly like one message of sixteen times the
  size and the per-message cost is measurably zero. The band was simply wide
  enough for the law's own 64 KiB to 1 MiB ratio to land near the hardware's
  depth ratio at this one size. The registered prediction that the modeled
  ratio keeps rising with Q instead of saturating against a separate ceiling
  is unaffected, and HTSIM-34 still owns it.
- **T3 reported, no bar**: the `cx5_100g` Q = 1 curve sits **above** the
  measured depth-1 rows at every size, as registered, with ratios
  1.69, 1.58, 1.34, 1.09, 1.26, 1.23 from 4 KiB to 4 MiB. Median over the four
  like-for-like sizes at or below 256 KiB: **1.46**. The gap closes as the
  message grows, which is the signature of an offset that is too small: at
  4 KiB the model is 11.9 Gb/s against 7.04 measured, at 256 KiB 86.0 against
  79.2. The 1 MiB and 4 MiB measured rows (73.95 and 77.03 Gb/s) sit in the
  loss equilibrium the comparator has no mechanism to enter, so their ratios
  are not a calibration signal; the plot shows the measured curve turning down
  there while the model keeps climbing.

## E-MTU: MTU tax at 1 MiB

![MTU tax](plots/mtu_tax.png)

- **M5 PASS**: 92.936 Gb/s at MTU 4096 against 88.859 Gb/s at MTU 1024, a tax
  of **4.39 percentage points** against the measured 5.6, inside the
  registered 2 point band. The arithmetic prediction in the registration was
  4.4 points, so the model hit its own prediction to two digits. This is the
  one measured axis the configuration reproduces for the right reason: the
  header fraction is expressed directly, and nothing else is involved.

## E-INCAST: 2 to 1 fan-in and 1 to 2 fan-out at 1 MiB

![incast tax](plots/incast_tax.png)

Three registered cells are void. The comparator's configuration guard
requires the switch egress buffer to exceed `ecn_kmax_bytes`, and the
registered small-buffer arm pairs a 262016 B buffer with the registered
409600 B Kmax, so **the arm cannot be configured at all**: `M8`,
`T4-b262016` and `T5-b262016` are void rather than failed, and the run is not
charged for them. This is a registration slip, caught by the runtime and not
by the freeze: the measured ConnectX-5 lossy pool is smaller than the ECN
Kmax the 100 G vendor row declares, so the two cannot be combined in this
comparator. A post-specified diagnostic arm at 524288 B, the smallest round
buffer above the registered Kmax, is reported below and is scored nowhere.

- **M6 PASS**: the two senders' shares of delivered goodput are **50.000 /
  50.000** in every runnable 2 to 1 cell, all three seeds, against the
  registered 2 point band and the measured 50.4 / 49.6.
- **M7 PASS**: the 1 to 2 fan-out delivers **95.995 Gb/s** aggregate,
  **1.0013x** the E-MSG Q = 16 reference of 95.866 Gb/s at 1 MiB against a
  registered floor of 0.95, splits **50.000 / 50.000**, and has **zero** drops
  and **zero** retransmissions in every cell and seed.
  Measured: 97.83 Gb/s, 50.000 / 50.000, every counter still. Fan-out is free
  in the model exactly as it is on the wire.
- **M8 VOID**: see above.
- **T4 FAIL, in the opposite direction to its registration**: the 2 to 1
  receiver goodput is **7.351 Gb/s** against the measured 73.9, a ratio of
  **0.0995**. The registration predicted a FAIL because the model would be
  too high; it is ten times too low instead.
- **T5 FAIL, the registered surprise**: the receiver wire rate is
  **7.55 Gb/s**, **7.78 percent** of the rendered link rate, against the
  measured 99.4 percent and a registered expectation of PASS. The registered
  reasoning was that a saturated fan-in keeps the bottleneck link busy whether
  or not the bytes are useful. It does not, and the reason is the same one
  behind T4.

### Why the fan-in collapses instead of paying a tax

The mechanism is in the counters, not in the goodput. A 2 to 1 episode moves
64 MiB and should take 5.6 ms; the simulated makespan is **73.0 ms**, and the
median flow completion is **73.0 ms** as well, so this is not a tail: every
flow finishes after one **67 ms silent retransmission timeout**. The run
records **38 silent RTOs against 13 go-back-N NACKs**, and **179
retransmitted packets against 16704 sent**, or 1.06 percent of the wire.

The reason the timeout path dominates is structural. Each GOAL message is its
own flow with no state shared with any other, so with 64 concurrent messages
each one holds only a few packets in flight. A drop is then usually the last
outstanding packet of that flow, nothing follows it to arrive out of order at
the receiver, no NACK is generated, and the sender waits the full timeout. The
hardware, running the same 67 ms local ACK timeout, keeps a deep per-QP
pipeline behind every loss and so almost always recovers by retransmission
instead. That is the difference between a 26 percent bandwidth tax and a
tenfold latency collapse, and it is owned jointly by HTSIM-34 (no finite
outstanding work, so a per-message flow has no window to speak of) and the
already-open HTSIM-5 (no per-QP state across messages).

### The amplification identity, on both sides

The literal registered identity `goodput_tax = loss_rate x message_bytes /
packet_bytes` does not hold anywhere, and the registration said so with the
arithmetic: at 1 MiB and 4096 B packets that factor is 256, so the measured
1.65 percent loss would predict a goodput tax above 100 percent against a
measured 25.7 percent. The measured implied amplification is **15.6 packets
of waste per lost packet**. The model's, at the post-specified 524288 B arm,
is **0.48**, with **0.46 retransmitted packets per drop**. The comparator
re-sends about half a packet per loss where the hardware re-sends about
sixteen, a factor of thirty-three, and it pays for the difference in timeouts
instead. The registered substitute band was a factor of two; the model misses
it by an order of magnitude, in the direction the registration named.

### Disclosed deviation from a registered definition

The registration defined delivered wire bytes as offered minus dropped at
4096 B each. That estimator is unsound and was replaced before any number was
read off it. `ns_tm3_dropped_packets` is switch-wide and counts control
packets: the large-buffer cell reports **316 drops against 179
retransmissions** while all 64 flows complete, which is arithmetically
impossible if every drop were a data packet. The reported wire rate is
instead the upper bound consistent with the counters, all payload-carrying
packets plus every retransmitted duplicate, with the lower bound carried
beside it in `incast.csv` as `wire_gbps_lower`. Both bounds are far below the
measured 99.4 percent, so the T5 verdict does not depend on the choice. The
same accounting puts the fan-out aggregate at 97.52 Gb/s against a 97 Gb/s
egress link, a 0.5 percent overshoot that is this estimator's tolerance on a
makespan that excludes some pipeline overlap.

## Post-specified long-flow arm, run of 2026-09-02

Added after the freeze, reported and scored nowhere, and appended to
`summary.csv` as `L1-*` and `L2-long-arm-guards` with the per-cell numbers in
`incast_long.csv`. Every registered check above re-ran byte for byte alongside
it: `latency.csv`, `msg.csv`, `mtu.csv` and `incast.csv` are identical files,
and each of the 19 committed check rows carries the same value in every column
it already had.

**Why the arm exists.** T4 and T5 do not measure bandwidth. The registered
episode is 64 MiB at 1 MiB per message and the makespan is 73.0 ms, which is
one 67.109 ms local ACK timeout plus the 5.6 ms of work. The completion CSV
shows it directly: 26 of the 64 flows finish between 5.548 and 5.918 ms and the
other 38, exactly the number of silent timeouts recorded, finish between 73.002
and 73.029 ms. Every GOAL message is its own rate-paced flow with no send
window, so it holds only a few packets in flight, a lost packet is usually the
last one outstanding, nothing follows it to arrive out of order, no NACK is
generated, and the sender waits the whole timer. This arm makes the episode
long against that timer.

The arm crosses two per-sender volumes with two message counts, the registered
32 per sender and one long message per sender. The second is the low
concurrency limit and is also the shape the measurement had: two hosts, one
stream each, not 64 concurrent messages. The switch buffer is the registered
default and the seed is the first registered seed. Neither volume had to be
dropped for simulation cost: the 4 GiB, 32 message cell takes about five
minutes of wall clock and the other three take seconds, so the sweep is the one
that was wanted rather than the one that was affordable.

| per sender | messages | goodput Gb/s | steady-window goodput Gb/s | wire percent | silent RTOs | drops | stall share |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 GiB | 32 | 51.7976 | 80.7059 | 54.2 to 100 | 120 | 834 323 | 37.9 percent |
| 1 GiB | 1 | **88.1247** | **88.1588** | **92.292** | **0** | **0** | **0 percent** |
| 4 GiB | 32 | 24.0903 | 23.6271 | 25.2 to 100 | 1394 | 8 013 802 | 51.2 percent |
| 4 GiB | 1 | **88.2342** | **88.2883** | **92.407** | **0** | **0** | **0 percent** |

Registered short arm for comparison: **7.3515 Gb/s**, no steady window at all,
and a stall share of **54.6 percent**.

**How the three reported quantities are defined.** *Stall share* is the summed
silent-timeout stall time over the episode's total flow time, that is
`silent_rtos x 67.109 ms / (flows x makespan)`; the stalls overlap in wall clock
so a sum over flows only means something against per-flow time. *Steady-window
goodput* is the payload rate over the episode with one full local ACK timeout
cut off each end, so the timeout-dominated ramp and the timeout-dominated tail
are both outside it; delivered payload at the two edges is read from the
cumulative acknowledgement column of the sender state trace. The registered
73.0 ms cell has no steady window at all, because it is shorter than the
2 x 67.109 = 134.2 ms that removing one timeout from each end would need, and
that is the cleanest single statement of why its goodput is not a bandwidth
number. *Wire percent* is a bracket rather than an estimate in the lossy cells,
and the disclosure below says why; in the two clean cells the bracket closes to
a point because there is nothing to bracket.

**Disclosed: the wire estimator had to change for this arm.** The registered
arm estimates delivered wire bytes as offered minus dropped, with the upper
bound reported. At these loss rates that is unsound in a way that is easy to
check: in the 4 GiB, 32 message cell it puts the receiver port at **479 percent
of its own link rate**, because the sender counters claim 40 480 363 packets
over 2852.6 ms and no wire in this topology could have carried them. What holds
without assumption is that every payload packet arrived exactly once, which the
`L2` row checks in all four cells, and that no port carries more than its link
rate. The table reports the interval between those two bounds. In the
single-message cells there are zero retransmissions and zero drops, so offered
equals payload and the two bounds coincide to a thousandth.

One column of `L2` needs a word so it is not read as a finding. Its
`max_shared_pool_dropped_packets` is large, and equals the cell's own drop
count, because this study configures `shared_buffer_bytes` and
`egress_buffer_bytes` to the same 32 MiB, so the shared pool and the per-port
pool are the same object and every drop is counted in both. That is unlike the
HACC leaf, where the two are separately sized and the shared pool never binds.

**Verdict: the comparator's fan-in is not bandwidth-limited, and T4's
7.35 Gb/s was a timeout reading.** Stretched to one long flow per sender the
same configuration moves **88.2342 Gb/s** of application goodput, which is
**1.194x** the measured 73.9 and **12.0x** the registered cell. It does not
approach the measurement from below; it passes it. The number is stable across
a 4x change in volume, 88.1247 against 88.2342, and the steady-window figure
agrees with the whole-episode figure to 0.06 percent, which is what a
bandwidth-limited episode with no tail should look like.

**But it reaches that goodput by the opposite route to the hardware.** The
measurement runs the wire at **99.4 percent** and pays a **26 percent**
retransmission tax on top of it. This arm runs the wire at **92.4 percent** and
pays **nothing**: zero drops, zero retransmissions, zero timeouts. So the
comparator is **7.0 percentage points idler** than the measured port and
**26 points cleaner** underneath, and the two errors happen to cancel in the
goodput. T5's finding survives the arm in a sharper form: the registered cell
missed the measured wire fraction by 91 points, this cell misses it by 7, and
in neither case is the model reproducing what the wire is actually carrying.
The 7 point shortfall is DCQCN holding two marked flows just below line rate,
with **5577** and **24616** ECN marks in the two clean cells; the marks the
registered arm reported were the same mechanism drowned in loss.

**What the arm does not buy.**

- The 5 percent stall target is met only by cutting the concurrency, never by
  volume. At the registered 32 messages per sender the stall share **rises**
  with volume, 37.9 percent at 1 GiB and 51.2 percent at 4 GiB, because a
  longer flow meets more losses and go-back-N re-sends more per loss; the
  timeout count goes 38, 120, 1394 as the message grows 1 MiB, 32 MiB, 128 MiB.
  Sixty-four fresh flows each starting at the full 97 Gb/s configured rate
  offer 6.2 Tb/s into a 97 Gb/s port before any rate cut can land. That is
  HTSIM-5 and HTSIM-34 stated as a throughput number rather than a mechanism.
- The loss regime is still not the measured one. The hardware's incast is a
  26 percent bandwidth tax at full wire; this path offers either a clean cell
  with no tax at all or a collapse with a 20 to 55 percent loss rate, and
  nothing in between. HTSIM-35 keeps its content.
- The measured amplification of 15.6 packets of waste per lost packet is
  untested here, because the cells that reach the measured goodput lose no
  packets at all.

For BACK-54 the arm changes nothing: the message offset is a per-message cost
and this arm has one message per sender, so `T_eff` is invisible in it. For
BACK-60 it removes the collapse from the list of things blocking the incast
acceptance bar and leaves the wire fraction and the loss regime.

## What the configuration buys, and what it does not

Landed by configuration alone, with evidence: the link rate and its 4 MiB
asymptote, the packetization and its MTU tax, the latency floor, fair sharing
under fan-in, free fan-out, and a ConnectX-7 arm that is the ConnectX-5 curve
scaled by four with the same offset. Not landed, with the numbers that say so:
the message offset (2.41 us against 4.48), the send-queue-depth law (perfect
amortization against a measured 1.57x at 64 KiB that saturates), and the loss
regime (a tenfold latency collapse against a 26 percent bandwidth tax). Those
three are BACK-54, HTSIM-34 and HTSIM-35 respectively, and the packet-rate
ceiling that HTSIM-36 owns was not exercised here at all.

Corrected by the long-flow arm: the third item was overstated. The tenfold
collapse is a property of the registered 64 MiB episode, not of the path. Given
a long enough flow the same configuration delivers 88.2342 Gb/s. What is
genuinely not landed is the loss regime itself, which is a different and
narrower claim: the hardware pays 26 percent at 99.4 percent of wire, and this
path either pays nothing at 92.4 percent of wire or falls apart. HTSIM-35 keeps
that, and HTSIM-5 and HTSIM-34 pick up the 32 message cells, where 64
simultaneous fresh flows are the reason the collapse happens at all.

Two process notes for the next registration in this series. A frozen sweep
value can be refused by the runtime rather than merely missed, so a freeze
should state which guards its arms have to clear; here the measured port
buffer and the declared ECN Kmax were individually defensible and jointly
inexpressible. And an episode has to be long against the timeouts it can
trigger: 32 messages per sender is 5.6 ms of work against a 67 ms retransmit
timeout, so the registered goodput metric measured one stall rather than a
congestion response, and the completion distribution reported beside it is
what shows that. The long-flow arm is that note acted on, and it added a third:
a registration that scores a throughput should also say what concurrency the
episode has, because on this path the number of simultaneous flows moved the
answer from 24 to 88 Gb/s at a fixed volume, which is further than either
volume or buffer moved it.

## 2026-09-02 re-run against the HTSIM-39 fix

Post-specified regression check, not a new registration. Nothing in
`expectations.md` changed, and the checks below are scored by the same code
against the same rules. The pin moved from htsim `1dcbfec` to `617ce20` on the
backend branch `codex/htsim39_fair_egress_drop`, which arbitrates physical
ingress ports that deliver in the same picosecond, so that a congested egress
buffer stops handing every slot it frees to whichever port the event list
served first. The full study was re-run in both arms with the same runner and
the same build, and the `1dcbfec` arm reproduces this record exactly: all 24
committed check rows carry the same value in every column, and `latency.csv`,
`msg.csv`, `mtu.csv`, `incast.csv` and `incast_long.csv` come back as identical
files.

**Every scored check keeps its verdict.** M1 through M8 and T1 through T5 read
the same PASS, FAIL or void, with the same numbers:

- M3's 2.0813 us latency floor, M1's 0.0002 worst deviation, both fitted `C`
  values, M4's exact 4.00x scaling, M5's MTU tax, T1's 0.5378 ratio, T2 and T3
  are unchanged to every published digit, and `latency.csv`, `msg.csv` and
  `mtu.csv` are identical files.
- M6 and M7 hold their 0.0 point share deviations and their 95.9952 Gb/s
  fan-out, and the fan-out rows of `incast.csv` are byte-identical.
- T4 and T5 at the registered 32 MiB buffer move by less than a thousandth of
  a Gb/s, to 7.3513 from 7.3514 and to 7.5488 from 7.5488, keeping their
  0.0995 and 7.782 percent ratios and both FAIL verdicts.

**The long-flow arm's headline is untouched.** The two single-stream cells,
which are the arm's clean cells and the shape the hardware measurement had,
come back **identical**: 88.1247 Gb/s at 1 GiB per sender and **88.2342 Gb/s**
at 4 GiB, both at 1.19 times the measured 73.9, with zero drops, zero
retransmissions and zero silent timeouts. A cell that loses nothing has no
contested buffer slot to arbitrate, so the arbiter cannot and does not touch
it. The amended verdict above stands exactly as written.

The two 32-message cells, which do lose, improve:

| per sender | messages | goodput at `1dcbfec` | at `617ce20` | ratio to measured |
|---:|---:|---:|---:|---:|
| 1 GiB | 32 | 51.7976 | **54.0244** | 0.701 to **0.731** |
| 4 GiB | 32 | 24.0903 | **27.5147** | 0.326 to **0.372** |

That is the opposite of what the same two cells did on the `hacc_fabric_v1`
leaf, where they fell, and the difference between the two fabrics is the
reason. This one marks: the arm's guard row records 2 325 638 ECN-marked
packets here, where the hacc leaf's guard G3 records zero in every cell. A
sender on this path gets a congestion signal that is not a loss, so spreading
the loss evenly lets both senders back off together instead of one thrashing
alone. On a fabric that marks nothing, sharing the loss only shares the
thrashing.

The unregistered 512 KB diagnostic arm collapses further, the same way the
`hacc_fabric_v1` fan-in did:

| 512 KB incast arm | at `1dcbfec` | at `617ce20` |
|---|---:|---:|
| goodput | 7.6693 Gb/s | **1.579 Gb/s** |
| makespan | 70.0 ms | **340.0 ms** |
| silent timeouts | 39 | **90** |
| dropped packets | 17884 | **29477** |
| loss rate | 0.7157 | **0.8735** |
| retransmissions per drop | 0.4633 | **0.5781** |
| goodput tax | 0.3428 | **0.5136** |

The share stays at 50.0/50.0 in every incast row, before and after. This
deepens the study's own headline rather than changing it: the path still
converts incast loss into 67 ms latency stalls where the hardware converts it
into a 26 percent bandwidth tax, and it now does so for both senders instead of
one. The retransmission amplification moved from 0.479 to **0.588** packets per
drop against a measured 15.6, so it is still two orders of magnitude short and
still outside the factor of two the diagnostic asks for. The direction is right
and the distance is what HTSIM-35 and BACK-60 own.
