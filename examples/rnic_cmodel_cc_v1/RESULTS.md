# RNIC golden C model slice D results

Run on 2026-09-02 against the expectations frozen in
[expectations.md](expectations.md), committed as
`3dd63a9fbc96d3c87f4a6e1285be276db31197f4` before the notification point, the
reaction point and the tail-drop egress queue existed and before any number was
produced by them.

**Verdict: 55 of 78 scored checks pass and every fatal guard holds, so the run
is scored rather than voided. The notification point is validated outright: on
a fabric that marks nothing, the receiving endpoint raises 272 notifications
per second per congested queue pair against a measured 283, and exactly zero
for a lone flow paced below saturation. The collapse the receive slice left
open is gone: the same two senders that paid a 92.92 percent tax with no
reaction point pay 3.17 percent with one, and they split the receiver 49.94 to
50.06 with the wire 98.50 percent full. The twenty-three misses are three
findings, not twenty-three. First, an established flow takes 293 ms to give up
30 percent of its rate when a competitor arrives, against a measured 3 to 39,
because one alpha has to be both the depth of a cut and the height of the
operating point. Second, the incast tax comes out at 3.17 percent against a
measured 21 to 27, because a notification point that lives on the receiver's
own ingress meter settles the loop at that meter's drain rate, and the value
slice C fitted for it sits below the switch's egress rate, so the switch buffer
never fills and there is no loss to amplify. Third, `packet_seq_err` counts
recovery episodes rather than loss bursts, which inverts the counter's meaning
under fan-in. Two post-specified sweeps, run after those verdicts were
recorded, show what each of the first two would take: a sixteen times larger
alpha gain brings the cut to 37 ms without moving the split, and a meter drain
rate at the value the incast measurement itself implies brings the tax to 24.39
percent at a goodput of 75.40 Gb/s against a measured 73.89.**

## Method

The study drives the `extern "C"` facade, not the C++ classes. Behind it each
sender queue pair packetizes, paces, keeps go-back-N state and holds one
reaction point; the receiver runs the ingress meter, the receive processor and
the notification point; and the probe owns the fabric. A sender of four queue
pairs is four endpoint objects sharing one 100 Gb/s uplink, each paced at a
quarter of it, because per-NIC arbitration across the queue pairs of one
endpoint is BACK-56's remaining clause and this study does not pretend to have
it. The switch is one tail-drop egress queue per receiver, draining at the port
rate, holding the measured 5.2 MB, with no marking, no pause and no
notification of any kind: the receiver is the only thing on the path that can
notice congestion, which is the whole point of the block.

Every steady quantity is read off the second half of its cell, so the startup
transient is excluded, exactly as slice C reads its equilibria. The dynamics
cell samples each host's own transmit byte count at 1 kHz and smooths it with a
five-sample boxcar, which is the instrument the campaign used.

Reproduce from the repository root:

```bash
python examples/rnic_cmodel_cc_v1/run_cc_study.py
```

Per-cell rows are in [curves.csv](curves.csv) and one row per registered check
is in [summary.csv](summary.csv). Sample traces and replay traces are written
to a scratch directory and are not tracked.

## Fitted latent parameters

| parameter | fitted | how |
|---|---:|---|
| `np_cnp_threshold_bytes` | 131008 | half the ingress buffer, selected from a three-value grid as the cell whose notification rate lands closest to the measured 283 per second per congested queue pair |
| `cnp_min_interval_ps` | 3.53e9 | the same fit. It corrects the vendor default the profile carried, 50 us, by a factor of seventy |
| `dcqcn_alpha_init_ppm` | 500000 | selected from a three-value grid by the transient, jointly with the additive step |
| `dcqcn_rate_increase_step_bps` | 27500000 per 1e9 ps | the same fit, per queue pair, so a sender of four recovers at 0.11 Gb/s per ms |
| `dcqcn_alpha_gain_ppm` | 3906 | not separated by any registered cell; held at the vendor default and declared |
| `dcqcn_alpha_update_ps` | 50e6 | held at the value the loop needs to hold its measured operating point, and corrected from the profile's 1 us. See the closed form below |

The alpha cadence is not free. With the limiter at one notification per 3.53 ms
and alpha decaying once per update interval, the steady alpha after `n` decays
between notifications is `g / (1 - (1 - g)^(n+1))`, which for a small gain is
about `1 / (n + 1)`. The equilibrium the loop then holds is
`R = 2 s / (c alpha)`, with `s` the additive increase per second and `c` the
notification rate. At the measured `c` of 283 per second and the fitted step,
that identity puts one queue pair at 11 to 13 Gb/s, which is the fair share of
a 100 Gb/s port between eight of them. The model measures 12.1 Gb/s per queue
pair with alpha at 0.013, so both sides of the identity agree to within 8
percent, which is what makes the notification rate a result rather than a
setting.

## Checks

| check | rows | verdict |
|---|---:|---|
| `np_rate`, within 30 percent of 283 per congested queue pair | 1 | PASS at 272.0 |
| `np_grid_direction`, non-increasing in the limiter interval | 3 | FAIL, 3 rows |
| `lone_quiet`, no notification below saturation | 4 | PASS, exactly zero everywhere |
| `lone_rate_intact`, delivered within 2 percent of offered | 4 | PASS, exactly equal |
| `rp_cut`, at least 30 percent inside 3 to 39 ms | 1 | FAIL at 293 ms |
| `rp_fair_time`, fair share inside 5 ms to 2.3 s | 1 | PASS at 5 ms |
| `rp_recovery`, 95 percent of the pre rate inside 337 to 557 ms | 1 | PASS at 526 ms |
| `rp_slope`, within 25 percent of 0.1 Gb/s per ms | 1 | FAIL at 0.0721 |
| `rp_steady`, split and wire under contention | 2 | PASS at 51.45 percent and 101.40 percent of the effective wire |
| `rp_persistent`, the rate crosses a completion unchanged | 1 | PASS, zero breaks |
| `incast_wire`, at least 97 percent utilization | 1 | PASS at 98.50 |
| `incast_goodput`, goodput and tax | 2 | FAIL at 94.06 Gb/s and a 3.17 percent tax |
| `incast_fair`, 50 plus or minus 2 points | 1 | PASS at 50.69 |
| `incast_seq_err_bursts`, packets lost over `packet_seq_err` at least 2 | 8 | FAIL, 0.15 to 0.86 |
| `incast_direction_size`, tax rising with message size | 4 | FAIL, 4 rows |
| `incast_direction_senders`, tax rising with sender count | 4 | PASS |
| `incast_buffer_direction`, tax not smaller at 2.6 MB | 4 | FAIL, 4 rows |
| `fanout_rate`, 97.8 plus or minus 3 points | 4 | PASS at 98.62 |
| `fanout_split`, equal to within 0.5 points | 4 | PASS at exactly 50.000 |
| `fanout_clean`, no drop, no discard, no notification | 4 | PASS, 0/0/0 everywhere |
| `identity_off`, byte-identical to the slice-C code path | 6 | PASS, zero differing fields |
| `identity_counters`, the three notification counters zero | 6 | PASS |
| `identity_off`, the collapse survives with the block off | 1 | PASS at a 92.92 percent tax |
| `cnp_ignored_zero`, `rp_cnp_ignored` exactly zero | 1 | PASS over 53 cells |
| `inert_marking`, the marking counter zero while notifications flow | 1 | PASS over 60 cells |
| `cnp_ledger`, sender and receiver notification counts agree | 1 | PASS over 53 cells |
| `fatal_guard` | 7 | PASS, every guard held |

### The notification point is validated

This is the block the P6 campaign asked for and the one the model reproduces
without a caveat. On a fabric that never marks, the receiving endpoint notices
its own ingress congestion and raises 272 notifications per second per
congested queue pair against a measured 283, a residual of 3.9 percent inside a
30 percent band. Below saturation it raises none at all: at 80 and 90 Gb/s of
paced offer, at one queue pair and at four, `np_cnp_sent` is exactly zero and
the delivered rate equals the offered rate exactly. The marking counter stays
at zero in all sixty cells while notifications flow, which is ANOM-07 and
ANOM-16 in one row, and `rp_cnp_ignored` is zero in all fifty-three controlled
cells because the reaction point acts on every notification it is handed.

The one miss here is `np_grid_direction`, and it answers the question the
expectations posed rather than failing it. The registered claim was that the
notification rate is non-increasing in the limiter interval. It is not: at
every threshold the rate rises slightly from 2.0 to 3.53 ms (263 to 264, 267 to
272, 266 to 270) and then falls hard at 6.0 ms (to 168). The reason is that the
rate is a product of two terms, the limiter's cap and the fraction of arrivals
that observe a meter above the threshold, and the interval moves them in
opposite directions: a slower limiter cuts less, which keeps the meter fuller,
which raises the duty cycle. At 6.0 ms the loop loses control entirely and the
cell collapses. So the measured 283 is a duty cycle times a cap, not a reading
of the cap, and it is emergent in the sense the anomaly table claims.

### The collapse is gone

Slice C recorded two senders into one responder collapsing to a 98.8 percent
tax with no reaction point, against a measured 26.9. With the reaction point
the same cell holds the wire 98.50 percent full, splits it 50.69 to 49.31 and
pays a 3.17 percent tax; with the block switched off on the same fabric it pays
92.92 percent. The mechanism is doing what it was built to do. What it does not
yet reproduce is the size of the tax, which is the next finding.

### The loop settles at the meter's drain rate, so the loss locus is wrong

The reaction point holds the aggregate at 97.14 Gb/s of receiver wire, which is
the ingress meter's fitted drain rate and 98.50 percent of the effective wire.
That is below the switch's 100 Gb/s egress rate, so the 5.2 MB buffer drains as
fast as it fills and there is almost nothing left to lose: the primary cell
drops 7803 packets at the switch and discards 10522 at the receiver's PHY out
of 1172792 issued, and the resulting tax is 3.17 percent against a measured 21
to 27.

The campaign's own receiver did not behave that way. Under fan-in it absorbed
99.39 Gb/s of wire while discarding almost nothing at its PHY, and the 1.65
percent of packets that went missing went missing at the switch. So the incast
measurement is a second anchor for the drain rate, and it wants a value near
the link rate while the lone-flow anchors slice C fitted against want one below
it. One value cannot serve both. This is the same shape as the bidirectional
incompatibility already registered against this meter, and it points at the
same missing stall mechanism (ANOM-17). BACK-57 carries it.

### One alpha cannot be both the cut and the operating point

The reaction point reproduces four of the six things the measured transient
says. The split under contention is 51.45 percent against 50, the receiver's
wire stays full, the recovery after the competitor leaves takes 526 ms against
a measured 447 plus or minus 110, and the rate crosses a work-request boundary
without moving in any of the 53 controlled cells, which is the persistence
clause the task was named for.

It does not reproduce the cut. An established flow needs 293 ms to give up 30
percent when a competitor arrives, against a measured 3 to 39 ms, and every one
of the nine cells of the registered grid is between 293 and 393 ms. The cause
is arithmetic, not tuning. Alpha's initial value only affects a queue pair's
very first notification; by the time a competitor arrives, alpha has settled at
the value the closed form above fixes, about 0.013, so each notification cuts
0.65 percent and a 30 percent cut needs about fifty of them at one per 3.53 ms.
Raising alpha would fix the cut and break the operating point, because the same
alpha divides into the equilibrium identity. Real DCQCN separates the two with
a target rate and a fast-recovery leg; the campaign measured a recovery that is
linear in time, which is what hid that half from the design. Adding it back is
BACK-58's remaining clause.

The fitted slope misses by 4 percent for a related reason: the fit selects the
cell by the cut, fair-share and recovery times, and the cell it selects has a
recovery leg whose first tens of milliseconds are still clearing the backlog
the overlap left behind, so the least-squares line through it reads 0.0721
Gb/s per ms where the reaction point's own step is 0.11.

### The sequence-error counter counts the wrong thing

The campaign measured `packet_seq_err` moving 73 times less often than packets
were lost, because silicon raises one per loss burst. The model raises one per
recovery episode, and a replayed packet that is itself out of sequence opens a
fresh episode, so the ratio of packets lost to sequence errors runs from 0.15
to 0.86 across the eight incast cells instead of above 2. In the primary cell
that is 23932 sequence errors for 18325 packets lost. The counter is inverted
relative to silicon for any tool that reads it, which is exactly the failure
mode the NIC-named facade exists to avoid. BACK-62 registers it.

### The tax does not rise with message size, and a smaller buffer costs less

Two of the three registered direction claims miss. The tax rises with sender
count in all four cells, as registered. It does not rise with message size: at
the 5.2 MB buffer and two senders it is 57.03 percent at 64 KiB and 3.17
percent at 1 MiB, and the same inversion appears at every buffer and sender
count. And the 2.6 MB buffer costs less than the 5.2 MB one in all four
comparisons rather than more, which the expectations registered in advance as
the weakest of the three claims.

Both have the same explanation and it is the replay window. Go-back-N replays
from the lost sequence number across everything the queue pair has in flight,
and how much that is depends on the standing queueing delay in front of the
receiver, not on the message size. A deeper buffer means a longer standing
delay, hence a longer replay, hence more waste per loss. A smaller buffer loses
earlier but replays less, and the second effect wins. The message-size result
is the same mechanism seen from the other end: the 1 MiB cells settle into the
clean equilibrium above and the 64 KiB cells do not, because at 64 KiB the same
byte rate is sixteen times the message rate and the loop spends the cell in the
loss regime. Neither is a band worth moving; both name the mechanism.

## Post-specified sweeps

Neither sweep is registered in `expectations.md`. Both were run after the
verdicts above were recorded and neither restates one. They are reported as
observations and are scored as nothing.

### The meter's drain rate under fan-in

Three drain rates by three alpha starts, on the primary incast cell:

| drain (Gb/s) | alpha 0.25 | alpha 0.35 | alpha 0.50 |
|---|---|---|---|
| 96.6 (the slice-C fit) | tax 90.37, goodput 9.63 | tax 1.66, goodput 95.03 | tax 3.17, goodput 94.06 |
| 98.6 | tax 88.18, goodput 11.82 | tax 1.57, goodput 97.06 | tax 1.65, goodput 97.01 |
| 99.4 (what the incast measured) | tax 85.00, goodput 15.00 | **tax 24.39, goodput 75.40** | tax 11.04, goodput 88.57 |

The cell in bold meets every incast bar the expectations registered: the tax is
24.39 percent inside 21 to 27, the application goodput is 75.40 Gb/s against a
measured 73.89 and 2.0 percent from it, the receiver's wire carries 99.72 Gb/s,
and the notification rate is 275 per second per congested queue pair. Its
steady split over the second half of the cell is 49.73 percent; the split the
check reads, which is over the whole cell including the transient, is 52.12 and
so is 0.12 points outside the two-point band. The reading is therefore that the
model can produce the measured bandwidth tax, that the parameter it needs is
the ingress meter's drain rate rather than anything in the congestion-control
block, and that the value it needs is the one the incast measurement itself
implies.

At alpha 0.25 every drain rate collapses, which is the other side of the same
tension: too shallow an opening cut and the loop never catches the transient.

### The alpha cadence against the cut time

Three gains by three update intervals, on the dynamics cell, over a shortened
window that shows the cut and not the recovery:

| gain | update 50 us | update 100 us | update 200 us |
|---|---|---|---|
| 3906 (1/256) | cut 293 ms, split 49.9 | cut 178 ms, split 49.9 | cut 103 ms, split 49.8 |
| 15625 (1/64) | cut 210 ms, split 51.0 | cut 113 ms, split 49.2 | cut 68 ms, split 49.8 |
| 62500 (1/16) | cut 49 ms, split 50.4 | cut 44 ms, split 47.4 | **cut 37 ms, split 49.7** |

The cut time falls monotonically in both directions and reaches 37 ms, inside
the registered 3 to 39 ms window, at the largest gain and the longest update
interval, with the split still fair at 49.7 percent. So the transient the
campaign measured is reachable, and reaching it costs a steady alpha about four
times larger than the one the notification rate wants. That is the quantity
BACK-58's remaining clause has to decouple. The recovery column is empty in
every cell because this sweep's tail is 200 ms and the recovery takes about
450; the registered dynamics cell above is where recovery is measured.

## Fatal guards

Every guard held, so the run is scored.

| guard | status |
|---|---|
| deterministic replay identity, row and both traces | held |
| byte conservation across the endpoint and the queue | held |
| no completion before delivery | held |
| counter monotonicity | held |
| pacing integrity, zero late releases | held |
| the reaction point's rate never leaves its interval | held |
| egress-queue conservation | held |
| identity off | held |

Identity off is the strongest of them. With the congestion-control byte clear,
all six cells of sweep (f) are byte-identical to the same cell run on the
slice-C code path, field for field, and the committed slice-C study artifacts
regenerate byte for byte after this slice landed.

## What this study did not close

It does not model the internal loopback-versus-wire arbiter, so ANOM-05 stays
unreproduced and registered under BACK-57. It does not explain the ingress
stall bursts of ANOM-17, and the drain-rate conflict above is a second symptom
of the same gap. It cannot confirm any DCQCN register against silicon, because
the parameter block is not readable on the campaign hosts, so every value it
fits is opaque and labelled so. It has no per-NIC arbitration across the queue
pairs of one sender: four queue pairs are four endpoint objects each paced at a
quarter of the port, which is what a fair arbiter would give them at saturation
and nothing more, and BACK-56 owns the difference. And it reproduces the honest
notification ledger, sender handled equals receiver sent in all fifty-three
controlled cells, where the campaign measured senders handling 2.24 times what
the receiver reported sending; that discrepancy stays recorded and unexplained
under ANOM-19.
