# HACC fabric model: results against pre-registered expectations

Run of 2026-09-02, one `run_hacc_fabric.py` invocation (13 registered check
rows in `summary.csv`, plus `latency.csv`, `msg.csv`, `buffer.csv` and
`incast.csv`).
Registrations are in [expectations.md](expectations.md), frozen in commit
`2a49e8c` before any run of this study; the fabric profile the flags and the
topology are rendered from landed in `d1892ff`. Binary: `htsim_dcqcn_atlahs`
at htsim `1dcbfec` with `txt2bin` from the same build. Raw artifacts, the
per-run completion CSVs and the sender state traces, stay outside Git under
`SIMLLM_DATA_ROOT`; the curated per-experiment CSVs are committed next to this
file.

Disclosure: the study was run twice. The first invocation scored every check
exactly as reported below; the second added three reported-only columns to
`buffer.csv` that attribute the loss to individual senders, because the first
run's headline finding could not be read off the registered columns. No band,
check, definition or sweep entry changed, and every scored number is identical
across the two invocations, which is a second determinism observation beyond
guard G1.

Second disclosure, and the reason `summary.csv` now holds 18 rows: a
post-specified long-flow arm was added later on 2026-09-02 and appended five
reported rows plus `incast_long.csv`. It is described in its own section below,
it is scored nowhere, and it changed nothing above. `latency.csv`, `msg.csv`,
`buffer.csv` and `incast.csv` re-ran byte for byte, and all 13 registered check
rows carry exactly the value they carried before in every column they had.

Superseded in part: the HTSIM-39 defect this study identified has since been
fixed in the backend, and everything below was re-run against the fix. The
record below is unchanged and is the record of the `1dcbfec` pin. The re-run,
with a before-and-after column on every number that moved, is in
[the 2026-09-02 section](#2026-09-02-re-run-against-the-htsim-39-fix) at the
end of this file, and the committed CSVs beside this file are now the re-run's.
Third disclosure, and the reason `summary.csv` now holds 19 rows: the re-run
adds one reported row and five reported columns from a switch-side per-ingress
drop counter the fix also adds, and nothing scored reads them.

Verdict: **4 of 5 scored checks pass and all four fatal guards hold. The
fabric profile lands by configuration: the 2.08 us latency floor, the rendered
100 G link, fair sharing under fan-in and a drop-only switch that marked zero
packets in any of the 18 cells. The one failure is the
buffer identity, and it fails for a reason worth more than the check: the
modeled egress buffer does not share its loss. Two symmetric senders into one
full port put 100 percent of the retransmissions on one of them while the
other finished untouched, where the measurement split the loss evenly to
within 0.5 percent across eight streams. The starved sender then receives
nothing at all after its first hole, so no out-of-order packet ever reaches
its receiver, no NACK is ever generated, and the sender learns of the loss
2.3 ms later when the queue finally drains. That is a new mechanism gap, it is
registered as HTSIM-39, and it is the reason a first-drop timing instrument
cannot work on this path today.**

Amended verdict on the fan-in, from the post-specified long-flow arm below:
**B1's 7.5613 Gb/s is a timeout reading, not a bandwidth reading.** The same
configuration moves 76.8671 Gb/s once the episode is long against the 67.109 ms
local ACK timeout, inside the 74 to 78 Gb/s band acceptance bar A1 names and
10.2 times the registered number. What the arm does not deliver is the wire
half of A1 or the measured loss regime, so the bar stays blocked and BACK-60
keeps it.

## E-LAT: the latency anchor

- **M1 PASS**: the 2 B message completes in **2.0808 us** against the measured
  2.08 us floor, a ratio of 1.0004 inside the registered 15 percent band. The
  registration predicted 2.081 us from `4 x 515 ns` of propagation plus
  20.8 ns of store-and-forward serialization at 100 Gb/s, and the run
  reproduces that to four digits. The cx5 study derived the same 515 ns per
  pipe at 97 Gb/s and measured 2.0813 us there, so the hop-count derivation is
  now confirmed at two link rates, which is what makes it a geometry result
  rather than a fitted constant.

## E-MSG: single-flow goodput

- **M2 PASS**: the 4 MiB single-flow goodput is **97.7488 Gb/s** against the
  rendered 100 Gb/s link, **2.251 percentage points** low, inside the
  registered 3 percent band. The registration predicted 97.7 to 97.8 Gb/s and
  named the systematic in advance: 1.56 points of it is the 64 B wire header
  on a 4096 B packet and the rest is the finite topology offset. The margin is
  thin by construction and was registered as thin.
- **M2-buffer reported**: the two buffer arms are identical to **0 ps** at
  every size, as registered. A single flow into an idle port never queues, so
  the egress buffer is unreachable and cannot change the result. The measured
  curve is **11.98, 67.84, 95.74, 97.75 Gb/s** at 4 KiB, 64 KiB, 1 MiB and
  4 MiB.

## E-BUF: the buffer identity

- **M3a FAIL**: the buffer estimate is three to ten times the configured
  buffer in all four cells, against a registered 20 percent band.

  | senders | buffer | predicted `t_nack` | measured `t_nack` | estimate over configured |
  |---:|---:|---:|---:|---:|
  | 2 | 5.2 MB | 833 us | **3145.208 us** | 3.78x |
  | 2 | 2.6 MB | 417 us | **2937.131 us** | 7.06x |
  | 3 | 5.2 MB | 625 us | **3145.192 us** | 5.04x |
  | 3 | 2.6 MB | 313 us | **2937.115 us** | 9.41x |

  The measured first NACK depends on the buffer and not at all on the sender
  count, where the identity says it should depend on both.

  The estimator is not simply lagging. Its two terms separate cleanly in the
  data and one of them is exactly right. The difference between the 5.2 MB and
  2.6 MB cells is **208.077 us** at both sender counts, against a predicted
  `(5.2 - 2.6) MB / 100 Gb/s = 208.0 us`: the registered "the packet that
  exposes the gap crosses a full buffer" term is confirmed to four digits.
  What is wrong is the other term. Subtracting the drain time leaves a
  constant **2729.2 us** in all four cells, and one sender's 8323 wire packets
  take `8323 x 4096 x 8 / 100 Gb/s = 2727.5 us` to transmit. The first NACK
  arrives one path latency after the sender has finished sending its entire
  message, whatever the buffer is and whatever the excess is.

- **M3a-loss-split, reported, and this is the finding**: the retransmissions
  per sender are **0 and 7054** with two senders at 5.2 MB, **0 and 7689** at
  2.6 MB, and **0, 7689 and 13943** with three senders. One sender is clean in
  every cell, and the worst carries **100 percent** of the loss in the
  two-sender cells and **64 to 66 percent** in the three-sender ones.

  The mechanism is in the switch, not in the endpoints. The ns-tm3 egress
  buffer drops whatever arrives while it is full, and the senders are
  open-loop rate-paced at identical rates with deterministic timing, so the
  same source wins the race for every slot the drain frees. In the two-sender
  cells the starved flow loses **every** packet from its first hole to its
  last: 7053 dropped out of the 7054 it sent after packet 1269, with the one
  survivor being the final packet, which only got in because both senders had
  by then stopped. Its receiver therefore never sees an out-of-order packet,
  because it sees no packet at all, so it never sends the NACK the sink is
  written to send on the first gap. The sender sits open-loop and unaware
  until that last packet is delivered when the queue drains, 2.3 ms after the
  first drop.

  Everything else about the cell follows from this, and the buffer itself
  behaves exactly as registered. The starved sender's cumulative
  acknowledgement stops at packet **1269**, so packet **1270** was the first
  one dropped, and that packet leaves the sender at
  `1270 x 4096 x 8 / 100 Gb/s = 416.4 us` against the registered
  `B / excess = 416 us`. The buffer identity holds in the simulator to three
  digits; it is only unobservable through any sender-side signal. And there is
  exactly **one** loss-rate cut per starved sender in the whole episode, at the
  very end, so the congestion response never engages while congestion is
  happening.

  This is registered as **HTSIM-39** (Precision; P1; M). The comparison that
  makes it a defect rather than a modeling choice is on the measured side:
  `data/p6/buffer.csv` records eight concurrent streams through a full
  tail-drop switch losing 186466, 186514, 186528, 186721, 185845, 185868,
  186362 and 186102 packets, a spread of **0.5 percent**.

- **M3b VOID, as registered**: the steady-state half of the identity needs an
  open-loop paced sender and this path has only a DCQCN closed-loop source.
  The reported numbers confirm the registration's reasoning and add to it: the
  configured rate does fall to **50 Gb/s** in every cell, and the loss
  fractions are **29.8, 31.6, 46.4 and 48.2 percent** of offered packets, far
  above any excess-rate share, because the loss is concentrated on the starved
  sender rather than spread across the offered excess.

## E-INCAST: fan-in behavior and the comparator baseline

- **M4 PASS**: the two senders' shares of per-sender throughput are within
  **0.632 percentage points** of 50/50 in the worst cell, against a registered
  2 point band, and the four cells read 49.71/50.29, 49.90/50.10, 49.37/50.63
  and 49.38/50.62. Fairness holds here and not in E-BUF for a reason worth
  stating: each of the 32 messages per sender is its own flow with its own
  loss and its own timeout, so the starvation that a single long flow suffers
  is re-drawn 64 times and averages out. Fair sharing at this granularity is
  therefore a weaker statement than it looks, and E-BUF is where the sharing
  question is actually answered.
- **B1 PASS**: the 2 to 1 receiver goodput at 1 MiB is **7.5613 Gb/s**, below
  the registered 20 Gb/s collapse ceiling and within 3 percent of the cx5
  study's **7.351 Gb/s** on the same packet path with a switch buffer six
  times larger. The collapse signature is the registered one: makespan
  **71.0 ms** for 64 MiB of work that should take 5.4 ms, **32 silent
  retransmission timeouts** against **15** go-back-N NACKs, and 7083 drops.
  Shrinking the buffer from 32 MiB to the measured 5.2 MB changes the depth of
  the loss and not its regime, which is what the registration predicted. The
  measured hardware moves 74 to 78 Gb/s through the same pattern; the distance
  between 7.56 and that number is what acceptance bar A1 is for.

## Post-specified long-flow arm, run of 2026-09-02

Added after the freeze, reported and scored nowhere, and appended to
`summary.csv` as `L1-*` and `L2-long-arm-guards` with the per-cell numbers in
`incast_long.csv`. Every registered check above re-ran byte for byte alongside
it: `latency.csv`, `msg.csv`, `buffer.csv` and `incast.csv` are identical files,
and each of the 13 committed check rows carries the same value in every column
it already had. The arm is deliberately kept out of the `G` guard set so that a
post-specified cell cannot move a scored cell count; its own guard numbers are
carried per cell instead and summarised in `L2`.

**Why the arm exists.** B1's 7.5613 Gb/s is not a bandwidth measurement. The
episode is 64 MiB at 1 MiB per message and the makespan is 71.0 ms, which is
one 67.109 ms local ACK timeout plus the 5.4 ms of work. Every GOAL message is
its own rate-paced flow with no send window, so it holds only a few packets in
flight; a lost packet is usually the last one outstanding, nothing follows it
to arrive out of order, no NACK is generated, and the sender waits the whole
timer. The registered cell therefore measures that timer. This arm makes the
episode long against it.

The arm crosses two per-sender volumes with two message counts, the registered
32 per sender and one long message per sender. The second is the low
concurrency limit and is also the shape the measurement had: two hosts, one
stream each, not 64 concurrent messages. The switch buffer is the measured
5.2 MB and the seed is the study seed. Neither volume had to be dropped for
simulation cost: every cell here finishes in under 30 seconds of wall clock, so
the sweep is the one that was wanted rather than the one that was affordable.

| per sender | messages | goodput Gb/s | steady-window goodput Gb/s | wire percent | silent RTOs | stall share |
|---:|---:|---:|---:|---:|---:|---:|
| 1 GiB | 32 | **78.6885** | **96.1353** | 79.9 to 100 | 48 | 23.1 percent |
| 1 GiB | 1 | 47.6906 | 38.4079 | 48.4 to 100 | 7 | 65.2 percent |
| 4 GiB | 32 | 65.6501 | 67.3905 | 66.7 to 100 | 193 | 19.3 percent |
| 4 GiB | 1 | **76.8671** | **73.0570** | 78.1 to 100 | 10 | 37.5 percent |

Registered short arm for comparison: 7.5613 Gb/s, no steady window at all, and
a stall share of 47.3 percent.

**How the three reported quantities are defined.** *Stall share* is the summed
silent-timeout stall time over the episode's total flow time, that is
`silent_rtos x 67.109 ms / (flows x makespan)`; the stalls overlap in wall clock
so a sum over flows only means something against per-flow time. *Steady-window
goodput* is the payload rate over the episode with one full local ACK timeout
cut off each end, so the timeout-dominated ramp and the timeout-dominated tail
are both outside it; delivered payload at the two edges is read from the
cumulative acknowledgement column of the sender state trace. The registered
71.0 ms cell has no steady window at all, because it is shorter than the
2 x 67.109 = 134.2 ms that removing one timeout from each end would need, and
that is the cleanest single statement of why its goodput is not a bandwidth
number. *Wire percent* is a bracket rather than an estimate, and the disclosure
below says why.

**Disclosed: the wire estimator had to change for this arm.** The registered
studies estimate delivered wire bytes as offered minus dropped. At these loss
rates that is unsound in a way that is easy to check: in the 4 GiB, 32 message
cell the sender counters claim 8 304 371 packets over 1046.8 ms, which is
260 Gb/s across two 100 Gb/s sender links, so the counters cannot be describing
packets that reached a wire. What holds without assumption is that every
payload packet arrived exactly once, because every flow completed and the `L2`
row records byte conservation in all four cells, and that no port carries more
than its link rate. The table therefore reports the interval between those two
bounds. The lower bound is the useful half: it says at least 78.1 percent of
the receiver port carried payload in the best cell.

**Verdict: the comparator's fan-in is not bandwidth-limited, and B1's 7.56 Gb/s
was a timeout reading.** Stretch the episode and the same configuration moves
**76.8671 Gb/s** at 4 GiB per sender in one stream, and **78.6885 Gb/s** at
1 GiB per sender in 32 messages. Both sit inside the 74 to 78 Gb/s band that
acceptance bar A1 names, against 7.5613 Gb/s registered, a factor of **10.2**.
The ratio to the 73.9 Gb/s anchor the cx5 study reports is **1.04** and **1.06**
respectively, so the comparator does not merely approach the measurement, it
crosses it.

**What the arm does not buy.** Three things, and they are the reason this is a
reported arm and not a claim that A1 is met.

- The wire fraction cannot be pinned. The measurement is 99.4 percent of wire
  with a 26 percent retransmission tax underneath it; this arm can only say the
  wire is between 78.1 and 100 percent, and cannot say how much of it is
  useful. A1 asks for both halves and only one is legible here.
- The stall share never reaches the 5 percent target, at any volume or message
  count tried. It falls from 47.3 percent registered to 19.3 percent at best,
  and *rises* to 65.2 percent in the single-stream 1 GiB cell. On this leaf that
  is expected rather than surprising: the `L2` row records zero ECN marks in all
  four cells here, as guard G3 does across the registered 18, so a sender on
  this leaf gets no congestion signal at all and learns of trouble only by
  losing packets. The loss fractions of 12.8 to 35.2 percent in this
  arm are the same open-loop behaviour E-BUF found, at a larger scale.
- The two message counts rank in the opposite order to the cx5 study's, and the
  reason is the same missing ECN. On the cx5 path a single stream per sender is
  the clean cell; here a single stream is the *worst* cell at 1 GiB, because two
  unmarked open-loop flows thrash go-back-N against each other, and cutting the
  volume into 32 messages spreads the loss and helps. Concurrency is not a
  nuisance parameter on this path, it is interacting with HTSIM-39.

For BACK-60 the arm moves the four acceptance bars in one specific way: A1's
goodput half is no longer blocked on the collapse, only on the wire accounting,
while A2, A3 and A4 are untouched because they need mechanisms this path still
does not have. HTSIM-39 is unaffected and HTSIM-5 keeps its content, since the
32 message cells still open 64 unrelated flows.

## Fatal guards

- **G1 PASS**: the repeated 2 sender, 5.2 MB cell matches on every scored
  quantity, with no mismatched field. The whole study repeated identically
  across two invocations as well.
- **G2 PASS**: all 18 cells conserve bytes. Every flow completed and the
  delivered payload equals the offered payload exactly in every cell.
- **G3 PASS**: **zero** ECN-marked packets across all 18 cells. The measured
  drop-only switch is realised, not approximated: the threshold pair sits two
  bytes below the tail-drop limit at one part per million and nothing was ever
  marked, including in the cells that dropped 23217 packets.
- **G4 PASS**: **zero** shared-pool drops across all 18 cells. Every drop is a
  per-port egress-domain drop, so the buffer numbers are about the measured
  per-port pool and the shared pool never bound.
- **G5 reported**: **zero** PFC pause frames, as expected with `-pfc off`.

## Full-chain acceptance bars: BLOCKED

None of these is scorable on the current packet path. They are restated here
with the work each waits on, unchanged from the registration.

| Bar | Target | Blocked on |
|---|---|---|
| **A1** incast goodput | 74 to 78 Gb/s within 15 percent at 99.3 Gb/s of wire | its wire clause. The goodput clause is no longer the obstacle: the post-specified long-flow arm reaches 76.8671 Gb/s, inside the band. What cannot be shown is the 99.3 Gb/s of wire, because the sender counters over-count at these loss rates and the receiver port can only be bracketed between 78.1 and 100 percent. HTSIM-39 and the golden-model transport work still stand behind the registered B1 cell's 7.5613 Gb/s |
| **A2** lone-flow ingress loss | 0.18 percent within 30 percent, in bursts of 50 to 100 packets | the golden-model receive-path work (a responder ingress meter); the path drops only at switch queues, and every drop in this study is a switch drop |
| **A3** DCQCN recovery | 95 percent of the pre-congestion rate in 447 ms within 25 percent | the golden-model rate-control work. Unmeasurable here for a second reason as well: the rate cut fires once, at the end of the episode, so there is no recovery to time |
| **A4** CNP rate | 283 per second per congested queue pair within 30 percent | HTSIM-38, the endpoint-side congestion-notification hook. With G3 at zero marks the current path can originate no notification at all |

BACK-60 owns all four and BACK-61 owns the queue-depth calibration the
campaign could not run.

## What the configuration buys, and what it does not

Landed by configuration alone, with evidence: the one-leaf geometry and its
2.08 us latency floor at a second link rate, the 100 G port rate and its
4 MiB asymptote, the per-port tail-drop buffer as the only place packets are
lost, a switch that marks nothing, a switch that neither sends nor honours
pause, and byte conservation and determinism throughout.

Not landed, with the numbers that say so: the loss cannot be observed from a
sender, because the switch starves one flow completely instead of sharing the
drop (0 against 7054 retransmissions, HTSIM-39); the congestion response never
engages during congestion, because the only notification a starved flow can
receive arrives after it has stopped sending (one rate cut per episode); and
the fan-in's registered goodput is a tenth of the measurement, unchanged from
the cx5 study and for the same reason.

Corrected by the long-flow arm: that last item is a statement about the
registered 64 MiB episode and not about the path. Lengthen the episode and the
fan-in reaches 76.8671 Gb/s, so the model is not short of bandwidth here. The
part that survives is the loss regime, which the arm makes worse rather than
better: 12.8 to 35.2 percent of offered packets are dropped and no cell reaches
the 5 percent stall target, because the switch marks nothing and the senders
learn about congestion only by losing.

One process note for the next registration in this series. The freeze chose
its instrument carefully, derived its lag from the route construction, and
predicted four distinct times from one identity. It was still the wrong
instrument, because it assumed the loss would be shared. A registration that
depends on a signal reaching a sender should state what has to be true of the
loss pattern for that signal to exist at all, and register a switch-side
counter beside it; the drop counters in this study were the only reason the
mechanism could be identified after the fact.

A second process note, from the long-flow arm. A registered episode has to be
long against the timers it can trigger, or its throughput metric measures a
timer. B1's 64 MiB is 5.4 ms of work against a 67.109 ms retransmit timeout,
which is not a ratio any bandwidth number survives, and the registration did
not state one. A freeze that scores a rate should say how long the episode is
in units of the longest timer the path can arm, and should register the
completion distribution beside the makespan so the two can be told apart
without a second study.

## 2026-09-02 re-run against the HTSIM-39 fix

Post-specified regression check, not a new registration. Every frozen
expectation, band, definition and sweep entry in `expectations.md` is
untouched, and the checks below are scored by the same code against the same
rules. What changed is the backend: the pin moved from htsim `1dcbfec` to
`617ce20` on the backend branch `codex/htsim39_fair_egress_drop`, which
arbitrates physical ingress ports that deliver in the same picosecond. One
reported check row and five reported columns were added, read from a
switch-side per-ingress drop counter the same backend change adds; nothing
scored reads them.

Both arms were run with the same runner and the same build, the only
difference being whether the arbiter is in the path. The `1dcbfec` arm
reproduces the record above exactly: all 18 committed check rows carry the
same value in every column they had, and `latency.csv`, `msg.csv`,
`buffer.csv`, `incast.csv` and `incast_long.csv` come back identical on every
column they had. That is what makes the comparison a comparison rather than
two runs.

### The mechanism, measured

Two senders paced at the same rate reach the switch at exactly the same
picosecond, and the switch pipeline behind the ingress ports took them one at
a time in a stable order, so the same port went first on every round. A
congested egress frees exactly one packet of room per packet time, so that
port took every freed slot. In the 2 sender 5.2 MB cell, **8323** instants had
an arrival from both ports at the identical picosecond, and **all 7053** drops
fell at an instant where the other port was admitted at that same picosecond.
Not most of them: all of them. The losing port's queue drained to zero and the
winning port held the whole 5.2 MB buffer alone.

So it is neither a per-flow reservation, of which there is none, nor a
receiver-side out-of-order discard, since the starved receiver saw no packet at
all. It is a fixed serialization order applied to a tie.

### E-BUF, the buffer identity: M3a now PASSES

| senders | buffer | predicted `t_nack` | at `1dcbfec` | at `617ce20` | estimate over configured |
|---:|---:|---:|---:|---:|---|
| 2 | 5.2 MB | 833 us | 3145.208 us | **834.705 us** | 3.7803 to **1.0033** |
| 2 | 2.6 MB | 417 us | 2937.131 us | **418.552 us** | 7.0604 to **1.0061** |
| 3 | 5.2 MB | 625 us | 3145.192 us | **626.956 us** | 5.0404 to **1.0047** |
| 3 | 2.6 MB | 313 us | 2937.115 us | **314.677 us** | 9.4138 to **1.0086** |

The worst deviation is **0.86 percent** against the registered 20 percent
band, where it was 841 percent. The measured time now depends on both the
buffer and the sender count, as the identity says it must, and the four times
predicted before any run are each met to better than one percent. That
confirms the earlier record's own diagnosis rather than overturning it: the
buffer was always right, and only the signal that reads it was unreachable.

### E-BUF, the loss split

The HTSIM-39 acceptance clause is about **equal-rate** sources. That is the
window from the start of a cell to the moment the first loss notification
crosses the switch. Before it no source has reacted, so the offered loads are
still identical and admission is the only thing that can make the loss
unequal. After it the sources are not equal-rate any more, because go-back-N
re-offers everything behind a hole and the rate cuts land at different times.

The band applied here is therefore **10 percent maximum deviation from an even
split, over the equal-rate window, across the senders' physical ingress
ports**. That is the registered band read literally. It is scored from a
switch-side counter rather than from retransmissions, because go-back-N
amplifies a gap by however long the sender took to notice it, so a
retransmission count measures the transport as much as the buffer.

| senders | buffer | equal-rate split at `1dcbfec` | at `617ce20` | deviation |
|---:|---:|---|---|---:|
| 2 | 5.2 MB | 0 and 7053 | **638 and 637** | 0.078 % |
| 2 | 2.6 MB | 0 and 7688 | **320 and 320** | 0.000 % |
| 3 | 5.2 MB | 7688, 7687 and 0 | **851, 850 and 850** | 0.078 % |
| 3 | 2.6 MB | 8005, 8005 and 0 | **427, 427 and 426** | 0.156 % |

The worst is **0.156 percent**, inside the 10 percent band and tighter than
the 0.5 percent spread the measured switch shows across eight concurrent
streams in `data/p6/buffer.csv`. Every source now loses something:
`clean_sources` is **0** in all four cells where it was **1** in all four, and
the retransmission split follows, reading **10591 and 9368** at 2 senders and
5.2 MB where it read 0 and 7054.

Over the whole run the split is still uneven, at worst 77.8 percent from an
even split, and that number belongs to the endpoints. Once the first gap is
signalled the senders diverge, one re-offering under go-back-N while the
other's rate cut lands elsewhere, and a shared FIFO correctly charges more
loss to whoever offers more. That residue is BACK-58 and BACK-60, not the
admission decision.

### E-INCAST: the fan-in was never fair either

M4 passed at `1dcbfec` with a 0.632 point worst deviation, and the earlier
record warned that fair sharing at that granularity was a weaker statement
than it looked. The switch-side counter says how much weaker. The equal-rate
loss split across the two senders' ports, in the four fan-in cells, was
**834 and 22**, **1465 and 19**, **7068 and 15** and **7702 and 13**, which is
94.9 to **99.7 percent** from an even split. M4 was passing while one sender
took 99.7 percent of the loss. At `617ce20` the same four cells read
**428 and 428**, **742 and 742**, **3542 and 3541** and **3857 and 3858**, a
worst deviation of **0.014 percent**.

### The long-flow arm: the A1 cells hold, the concurrent cells do not

| per sender | messages | goodput at `1dcbfec` | at `617ce20` | equal-rate loss split before | after |
|---:|---:|---:|---:|---|---|
| 1 GiB | 32 | 78.6885 | **40.7898** | 265053 and 15 | 133170 and 133168 |
| 1 GiB | 1 | 47.6906 | **47.3296** | 84 and 207277 | 638 and 637 |
| 4 GiB | 32 | 65.6501 | **15.9372** | 1063966 and 14 | 532624 and 532624 |
| 4 GiB | 1 | **76.8671** | **76.6298** | 84 and 207277 | 638 and 637 |

The arm's own headline survives intact. The 4 GiB single-stream cell, the one
the amended verdict above is built on and the one whose shape matches the
hardware measurement, reads **76.6298 Gb/s** against 76.8671, a ratio to the
measured 73.9 of **1.0369** against 1.0402, still inside the 74 to 78 Gb/s
band acceptance bar A1 names. The 1 GiB single-stream cell is likewise
unmoved, at 47.3296 against 47.6906. Both single-stream cells were losing
84 against 207277 in the equal-rate window and now lose 638 against 637, so
the fairness changed completely while the throughput did not.

The two 32-message cells fall a long way, and the reason is in the same table.
Their previous numbers were produced by the defect. In the 1 GiB cell one
sender took **265053** drops in the equal-rate window and the other took
**15**, so the aggregate was carried by a sender that was never losing
anything and ran near line rate while its neighbour thrashed. Once both
senders lose equally, both thrash, and the aggregate reads what two senders
sharing an unmarked drop-tail port actually achieve. The 4 GiB cell is the
same story at four times the volume, 1063966 against 14 becoming 532624 each.

The earlier arm's third bullet predicted exactly this: it said concurrency was
not a nuisance parameter on this path but was interacting with HTSIM-39. It
was, and this is the size of the interaction. What that costs is the arm's
claim that both of its best cells sit inside the A1 band; only the
single-stream one does now, and that is the cell whose shape the measurement
had.

### What got worse, and why that is the honest reading

- **M4 flips from PASS to FAIL**, at **15.546** percentage points against a
  registered 2 point band. Three of its four cells improved, to 50.0/50.0,
  49.994/50.006 and 50.0/50.0 from 49.706/50.294, 49.896/50.104 and
  49.379/50.621. The fourth, 1 MiB into the 5.2 MB buffer, reads
  34.454/65.546 because one sender's last message waited one more silent
  retransmission timeout than the other's. That timeout is **67.108 ms**
  against 5.4 ms of useful work, so this metric snaps to multiples of it and
  is measuring the timer, not the fabric.
- **B1 still PASSES and the collapse deepens**, to **3.9187 Gb/s** from
  7.5613, with a **137.0 ms** makespan against 71.0 ms, **43** silent timeouts
  against 32 and **33** go-back-N NACKs against 15. With the loss shared, both
  senders now stall where one used to sail through untouched.

Neither is a new fabric defect, and neither is bandwidth. Both are the
endpoint transport showing through a fabric that no longer hides it by
starving one flow, on a leaf that marks nothing so a sender learns of
congestion only by losing. Both are already owned: A1 is blocked on this same
go-back-N source and BACK-60 owns the sender state it needs.

### Unchanged, byte for byte

M1 is **2.0808 us**, M2 is **97.7488 Gb/s**, the M2 message-size curve is
**11.9773, 67.8441, 95.7382 and 97.7488 Gb/s**, M2-buffer is still 0 ps at
every size, and G1 through G5 hold exactly as before: no mismatched field on
the repeated cell, byte conservation in all 18 cells and in all four long-arm
cells, zero ECN marks, zero shared-pool drops and zero pause frames.
`latency.csv` and `msg.csv` are identical files. That is the registration's
third clause, that a single-source run preserves every accepted result byte
for byte, met literally.

### Verdict of the re-run

**4 of 5 scored checks pass, as before, but not the same 4.** M3a passes and
M4 fails, where M3a failed and M4 passed. All four fatal guards hold. The
buffer identity is landed to better than one percent at four points. The loss
is shared to within 0.156 percent among equal-rate sources, against a 10
percent band and a 0.5 percent measurement. The long-flow arm's single-stream
cells, including its 76.87 Gb/s headline, are unmoved. What the fix exposes is
that the fan-in numbers on this path have been flattered by starvation twice
over: once in B1, and once in the concurrent long-flow cells whose aggregate
was carried by a sender that never lost a packet.
