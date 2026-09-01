# HACC fabric model: results against pre-registered expectations

Run of 2026-09-02, one `run_hacc_fabric.py` invocation (13 check rows in
`summary.csv`, plus `latency.csv`, `msg.csv`, `buffer.csv` and `incast.csv`).
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
| **A1** incast goodput | 74 to 78 Gb/s within 15 percent at 99.3 Gb/s of wire | the golden-model transport work (per-queue-pair sender state across messages), and now also HTSIM-39, since a starved flow cannot reach a fair share whatever the endpoint does. Today's number is B1's 7.5613 Gb/s |
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
the fan-in still collapses to a tenth of the measured goodput, unchanged from
the cx5 study and for the same reason.

One process note for the next registration in this series. The freeze chose
its instrument carefully, derived its lag from the route construction, and
predicted four distinct times from one identity. It was still the wrong
instrument, because it assumed the loss would be shared. A registration that
depends on a signal reaching a sender should state what has to be true of the
loss pattern for that signal to exist at all, and register a switch-side
counter beside it; the drop counters in this study were the only reason the
mechanism could be identified after the fact.
