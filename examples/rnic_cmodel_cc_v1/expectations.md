# RNIC golden C model slice D expectations

This document is frozen before the congestion-control block exists and before
any number is produced by it. It registers the sweeps, the closed forms, the
bands and the fatal guards for slice D of the RNIC golden C model: the DCQCN
notification point at the receiving endpoint, the per-queue-pair reaction point
at the requester, and the tail-drop egress queue the test fabric needs before
either of them can be exercised (BACK-58), together with the control-event half
of the C facade (BACK-55). No result appears here; results go in `RESULTS.md`
and cite this file's commit hash.

It lives in its own directory rather than beside the slice-C study, because
that study's `expectations.md`, `curves.csv` and `summary.csv` are a closed
record of a run that has already been scored and must stay byte-identical.

## Scope

Slice D is the congestion-control loop and nothing else. It closes the two
clauses the slice-C study left open that depend on a reaction point: the
incast, where a go-back-N requester with no rate control answers loss by
raising its own offered load and collapses, and the counter pair that a real
congestion-controlled run moves. It does not close the internal
loopback-versus-wire arbiter (ANOM-05) or the ingress stall-burst structure
(ANOM-17); both stay registered, ANOM-05 under BACK-58 and ANOM-17 under
BACK-57, and no cell here is allowed to claim either.

The study drives the `extern "C"` facade, not the C++ classes, so the entry
points under test are the ones an RTL testbench uses.

## Model configuration

Endpoints of the `cx5_100g` profile joined by a fabric the probe owns. Each
sender runs one reliable-connection queue pair per endpoint object, so a sender
of four queue pairs is four endpoint objects sharing one sender uplink. That is
a stated limitation, not an oversight: per-NIC arbitration across several queue
pairs of one endpoint is BACK-56's remaining clause, and modelling four queue
pairs as four endpoints gives the reaction point the per-queue-pair state the
measurement resolves without pretending the missing arbiter exists.

The fabric is the measured HACC leaf, as
[`simllm/backends/fabric_profile.py`](../../simllm/backends/fabric_profile.py)
carries it:

| fabric constant | value | source |
|---|---:|---|
| port rate | 100e9 | measured leaf |
| per-port egress buffer, tail drop | 5 200 000 bytes | measured leaf |
| pipe latency | 515 000 ps | measured leaf |
| pipes per path | 4 | measured leaf |
| ECN marking | none | measured leaf, zero marks in 670 M packets |
| PFC and pause propagation | off | measured leaf |

Each sender has its own uplink into the switch at the port rate. The switch has
one egress queue toward the receiver: it drains at the port rate, holds at most
the egress buffer, and a packet that does not fit is dropped there and then.
There is no marking, no pause and no notification of any kind from the switch,
which is what forces the notification point to sit at the endpoint.

Frozen hardware constants, all already in the profile:

| constant | value |
|---|---:|
| `link_bps` | 100e9 |
| `goodput_bps` | 97.1e9 |
| `mtu_bytes` | 4096 |
| `wire_header_bytes` | 64 |
| effective wire rate | 98.617190e9 |
| `rx_ingress_bytes` | 262016 |
| `rx_drain_bps` | 96.6e9 |
| `recovery` | go-back-N |
| `rto_ps` | 67108864000 |
| `ecn_stamp` | ECT(0) |

Frozen study constants, chosen here so they cannot be chosen after the fact:

1. The incast messages are reliable-connection WRITEs of the swept size, four
   queue pairs per sender, send-queue depth 1024, MTU 4096.
2. The dynamics cell is one incumbent sender of four queue pairs running alone,
   a second sender of four queue pairs starting at a declared instant and
   stopping at a second declared instant, and the incumbent running on. Rates
   are sampled at 1 kHz on each sender's own transmit byte count and smoothed
   with a five-sample boxcar, which is the instrument the campaign used.
3. Loss in the incast cells comes from the egress queue and from the receiver's
   own ingress meter. The injected-loss knob of slice C stays, is set to zero in
   every slice-D cell, and keeps working for the slice-C checks.
4. A congested queue pair is one whose sender is offering into the contended
   egress queue during the measured window. The notification rate is reported
   per congested queue pair, which is the campaign's own denominator.

## Latent parameters

These are `calibrated-opaque`: the study fits them over a declared candidate
grid and reports the fitted value with the band the checks leave around it. No
value is stated here.

| parameter | meaning | candidate grid |
|---|---|---|
| `np_cnp_threshold_bytes` | ingress occupancy above which an arriving packet is treated as having observed congestion | one quarter, one half and three quarters of `rx_ingress_bytes` |
| `cnp_min_interval_ps` | shortest gap between two notifications for one queue pair | 2.0e9, 3.53e9, 6.0e9 |
| `dcqcn_alpha_init_ppm` | alpha a queue pair starts at | 250000, 350000, 500000 |
| `dcqcn_alpha_gain_ppm` | the gain g of the alpha recursion | 3906, 15625, 62500 |
| `dcqcn_alpha_update_ps` | the interval alpha decays over without a notification | 50e6, 100e6, 200e6 |
| `dcqcn_rate_increase_step_bps` per `dcqcn_rate_increase_interval_ps` | the additive increase, per queue pair | 22.5e6, 25.0e6, 27.5e6 bits per 1e9 ps |

The profile's declared vendor defaults for the four DCQCN registers are the
starting point and are expected to be corrected: the campaign measured a
reaction two to twenty times slower and a recovery about four and a half times
slower than those defaults imply, and the corrected values are what this study
fits. The rate floor is taken from the profile and is not fitted.

## Closed forms

**Notification rate.** With the meter above the threshold a fraction `d` of the
arrival instants of one queue pair and the per-queue-pair limiter at `T`, the
notification rate is `min(d * arrival rate, 1 / T)`. Under sustained fan-in `d`
approaches one and the limiter binds, so the measured 283 per second per
congested queue pair is a reading of `1 / T` whenever the meter is pinned and a
reading of the duty cycle whenever it is not. Which of the two it is, is a
result and is reported as such.

**Reaction point.** On a notification,

```
alpha <- (1 - g) * alpha + g
rate  <- max(rate * (1 - alpha / 2), rate floor)
```

and with no notification inside one update interval, `alpha <- (1 - g) alpha`.
Every increase interval, `rate <- min(rate + step, ceiling)`. The steady alpha
after `n` update intervals between two notifications is
`g / (1 - (1 - g)^(n+1))`, which for a small `g` is about `1 / (n + 1)`.

**Equilibrium.** A queue pair holding a rate `R` under a notification rate `c`
with an additive increase `s` per second satisfies `c * (alpha / 2) * R = s`.
This is the identity the fit has to satisfy at the measured operating point,
and the study reports both sides.

**Recovery time.** From a fair share `F` to a fraction `f` of a ceiling `C`,
purely additive recovery takes `(f * C - F) / s`.

**Incast tax.** With the campaign's own definition,

```
tax = 1 - application goodput at the receiver / receiver wire rate
```

and the amplification factor is the tax divided by the loss rate on the path.

## Sweeps

Every sweep varies at least two parameters.

### (a) Notification-point calibration, 9 cells

`np_cnp_threshold_bytes` (three values) times `cnp_min_interval_ps` (three
values), on the primary incast cell: two senders, four queue pairs each, 1 MiB
reliable-connection WRITE, the 5.2 MB egress queue.

1. `np_rate` (1 row, the selected cell). The receiver's `np_cnp_sent` per
   second divided by the number of congested queue pairs is within 30 percent
   of 283, that is inside 198.1 to 367.9.
2. `np_grid_direction` (2 rows). Across the grid the notification rate is
   non-increasing in `cnp_min_interval_ps` at a fixed threshold, and
   non-increasing in the threshold at a fixed interval. Both are monotone
   claims about the mechanism, not about a value.

### (b) Lone flow, 4 cells

Offered rate {80, 90} Gb/s of payload times queue-pair count {1, 4}, one sender
into one receiver, no contention.

3. `lone_quiet` (4 rows). `np_cnp_sent` is exactly zero in every cell: below
   saturation the receiver's meter never reaches the threshold, so a fabric
   with no congestion generates no notifications at all.
4. `lone_rate_intact` (4 rows). The delivered goodput is within 2 percent of
   the offered rate, so the notification point costs nothing when it is quiet.

A lone flow **at** saturation is reported as an observation rather than a
check, and is registered here as an expected disagreement: the ingress meter
that reproduces ANOM-03 sits near full under a saturated lone flow, so the
model will raise notifications there while the silicon raised only about 38 per
second. That gap is the same missing stall structure ANOM-17 names, it belongs
to the meter and not to the notification point, and BACK-57 already owns it.

### (c) Reaction-point dynamics, 9 cells

`dcqcn_alpha_init_ppm` (three values) times the additive increase (three
values), on the dynamics cell: an incumbent sender of four queue pairs, a
competitor of four queue pairs starting at 1.0 s of modelled time and stopping
at 2.0 s, the incumbent running to 2.8 s.

5. `rp_cut` (1 row, the selected cell). The incumbent's rate falls at least 30
   percent below its pre-competitor rate between 3 and 39 ms after the
   competitor starts.
6. `rp_fair_time` (1 row). The incumbent reaches 50 plus or minus 5 percent of
   the receiver's delivered rate between 5 ms and 2.3 s after the competitor
   starts.
7. `rp_recovery` (1 row). After the competitor stops, the incumbent returns to
   at least 95 percent of its pre-competitor rate within 447 plus or minus 110
   ms, that is inside 337 to 557 ms.
8. `rp_slope` (1 row). The fitted additive increase, expressed at the sender
   level over its four queue pairs, is within 25 percent of 0.1 Gb/s per ms.
9. `rp_steady` (2 rows). During the overlap the two senders split the receiver
   within 50 plus or minus 2 percentage points and the receiver's wire is at
   least 97 percent utilised.
10. `rp_persistent` (1 row). The reaction point's rate is unchanged across a
    work-request boundary: the rate a queue pair holds at the completion of one
    message is the rate it starts the next with, to the bit. This is the
    per-queue-pair persistence clause and is checked on the model's own state,
    not on a derived rate.

### (d) Incast acceptance, 8 cells plus one primary

Egress buffer {5.2 MB, 2.6 MB} times message size {64 KiB, 1 MiB} times sender
count {2, 3}, four queue pairs per sender, reliable-connection WRITE. The
primary cell is 5.2 MB, 1 MiB, two senders, which is the measured one.

11. `incast_wire` (1 row, primary). The receiver's wire utilization is at least
    97 percent of the link rate.
12. `incast_goodput` (1 row, primary). The application goodput at the receiver
    is within 15 percent of the measured 73.89 Gb/s, that is inside 62.8 to
    85.0 Gb/s, and the tax is inside 21 to 27 percent of the wire. Both halves
    must hold: the first is the campaign's own number with its band, the second
    is the registered tax window, and the second is the tighter one.
13. `incast_fair` (1 row, primary). The per-sender share of the receiver's
    delivered bytes is 50 plus or minus 2 percentage points.
14. `incast_seq_err_bursts` (8 rows). The requester's `packet_seq_err` is
    strictly below the number of packets lost on the path in every cell, and
    the ratio of packets lost to `packet_seq_err` is at least 2. The counter
    counts loss bursts, not packets (ANOM-19).
15. `incast_direction_size` (4 rows). At a fixed buffer and sender count the tax
    is strictly larger at 1 MiB than at 64 KiB.
16. `incast_direction_senders` (4 rows). At a fixed buffer and message size the
    tax is strictly larger with three senders than with two.
17. `incast_buffer_direction` (4 rows). At a fixed message size and sender
    count the tax is not smaller with the 2.6 MB buffer than with the 5.2 MB
    one. Registered as the weakest of the three directions: a smaller buffer
    loses earlier but also queues less, and if the model shows no dependence
    the mechanism is named rather than the band moved.

The amplification factor is reported for every cell and is not banded, because
the campaign's own closed form does not reproduce its own measured 16.3 and is
a shape rather than a predictor.

### (e) Fan-out control, 4 cells

One sender into two receivers, message size {64 KiB, 1 MiB} times queue pairs
per receiver {2, 4}. This is the control that says the pain is receiver side.

18. `fanout_rate` (4 rows). The aggregate delivered rate is 97.8 plus or minus
    3 percent of the link rate.
19. `fanout_split` (4 rows). The two receivers get equal byte counts, to within
    0.5 percentage points.
20. `fanout_clean` (4 rows). Zero drops anywhere: no egress-queue drop, no
    `rx_discards_phy` on either receiver, and `np_cnp_sent` zero on both.

### (f) Identity off, 6 cells

Congestion control disabled, over message size {64 KiB, 1 MiB} times sender
count {1, 2, 3}, everything else as in sweep (d) at the 5.2 MB buffer.

21. `identity_off` (6 rows). Every field of the probe's row is byte-identical to
    the same cell run on the slice-C code path with the same inputs, including
    the collapse the slice-C study recorded for two senders into one responder.
    The point of this check is that turning the block off restores the previous
    model exactly, collapse included.
22. `identity_counters` (6 rows). With congestion control off, `np_cnp_sent`,
    `rp_cnp_handled` and `rp_cnp_ignored` are all exactly zero.

### (g) Counter ledger, every cell of sweeps (a) through (f)

23. `cnp_ignored_zero`. `rp_cnp_ignored` is exactly zero everywhere. The
    reaction point handles every notification it is given; a notification it
    could not attribute would be counted here and there are none.
24. `inert_marking`. `np_ecn_marked_roce_packets` is exactly zero everywhere,
    while `np_cnp_sent` is nonzero in the congested cells. That pair is
    ANOM-07 and it is what a detection tool reads.
25. `cnp_ledger`. In every cell the senders' summed `rp_cnp_handled` equals the
    receiver's `np_cnp_sent` exactly. The campaign measured senders handling
    2.24 times the notifications the receiver reported sending and could not
    explain it (ANOM-19); the model reproduces the honest ledger and the
    difference is reported as an open counter-semantics gap, not as a modelled
    effect.

## Fatal guards

A fatal guard voids the run. It is never reported as a fraction of passing
checks.

- **Deterministic replay identity.** The designated replay cell (`cx5_100g`,
  1 MiB, two senders of four queue pairs, the 5.2 MB egress queue, congestion
  control on) is run twice in one process and both the CSV row and both facade
  transaction traces must be byte-identical.
- **Byte conservation.** In every cell the payload bytes the requesters
  packetized equal the payload bytes delivered plus the payload bytes discarded
  plus the payload bytes still in flight, and every completed message carries
  exactly its offered byte count.
- **No completion before delivery.** No completion becomes visible for a
  message before every packet of that message has been delivered to the
  responder's receive processor, and no message completes twice.
- **Counter monotonicity.** Every NIC-named counter is non-decreasing across
  the run on every endpoint.
- **Pacing integrity.** Zero late releases in every cell.
- **Rate bounds.** The reaction point's rate never leaves the closed interval
  between the profile's rate floor and the effective wire rate, in any cell, at
  any instant the probe samples.
- **Egress-queue conservation.** In every cell the bytes the egress queue
  admitted equal the bytes it forwarded plus the bytes it holds, and the bytes
  it dropped equal the bytes offered minus the bytes admitted.
- **Identity off.** With congestion control disabled the probe reproduces the
  slice-C rows byte for byte, and every existing native test passes unchanged.

## What this study cannot show

It cannot show the internal loopback-versus-wire arbiter, which needs a
loopback datapath this slice does not add, so ANOM-05 stays unreproduced and
registered. It cannot explain the ingress stall bursts of ANOM-17: the meter
still reproduces the average lone-flow loss without its structure, and the
notification rate under a saturated lone flow inherits that error. It cannot
confirm the DCQCN register values against silicon, because the parameter block
is not readable on the campaign hosts; every value it fits is opaque and is
labelled so. It cannot settle the counter discrepancy where senders handle 2.24
times the notifications the receiver reports sending, only report that the
model's own ledger closes and the silicon's does not. And it cannot give the
per-NIC arbitration across the four queue pairs of one sender, because each
queue pair is its own endpoint object here; that is BACK-56's clause and the
study says so wherever it matters.
