# RNIC golden C model slice C expectations

This document is frozen before the receive pipeline exists and before any
number is produced by it. It registers the sweeps, the closed forms, the bands
and the fatal guards for slice C of the RNIC golden C model: the ingress meter,
the receive processor and the requester transport (BACK-57), the NIC-named
counter facade and the facade's receive entry point (BACK-55). No result
appears here; results go in `RESULTS.md` and cite this file's commit hash.

It lives in its own directory rather than beside the slice-B study, because
slice B's `expectations.md`, `curves.csv` and `summary.csv` are a closed record
of a run that has already been scored and must stay byte-identical.

## Scope

Slice C is the receive half of the endpoint plus the requester's transport
state. There is still no rate control, no DCQCN reaction point, no internal
arbiter and no loopback path, so an ECN mark and a CNP remain refused at the
facade and the loopback rows of the anomaly table stay untouched. The study
drives the `extern "C"` facade, not the C++ classes, so the entry points under
test are the ones an RTL testbench uses.

## Model configuration

Two endpoints of the `cx5_100g` profile, a requester and a responder, joined by
a wire the probe owns. The requester runs one RC send queue on one QP with the
transmit pipeline enabled (network ABI v2), and the responder runs the receive
pipeline. Data packets leave the requester through `rnic_cm_tx_next` and enter
the responder through `rnic_cm_rx_packet`; the ACK and NAK packets the
responder generates leave it through `rnic_cm_tx_next` and enter the requester
through `rnic_cm_rx_packet`. Nothing else crosses.

Frozen hardware constants, all from the mlx5 campaign and already in the
profile:

| constant | value |
|---|---:|
| `link_bps` | 100e9 |
| `goodput_bps` (C) | 97.1e9 |
| `mtu_bytes` | 4096 |
| `wire_header_bytes` | 64 |
| effective wire rate `eff` | 98.617190e9 |
| `t_eff_ps` | 4480000 |
| `wire_round_trip_floor_ps` | 2100000 |
| `recovery` | go-back-N |
| `rto_ps` | 67108864000 |

Frozen study constants, chosen here so they cannot be chosen after the fact:

1. A burst is **128 messages**, matching the burst size the campaign's engine
   used for every gap-swept point. The inter-burst gap is the swept parameter.
   The burst byte count therefore scales with the message size, which is why
   the measured drain window is expected to scale with it too.
2. The fabric between two senders and one receiver is one egress queue of
   **65536 bytes** at the receiver's port, declared, plus the profile's
   round-trip floor. It is a fabric constant, not an endpoint one: the
   endpoint cannot see it and cannot fit it. It is what supplies the standing
   queueing delay that sets the go-back-N replay depth in an incast, which a
   bare 2.1 us round trip cannot supply on its own.
3. Injected loss is applied by the wire, never by an endpoint. It carries
   `DropLocation::Fabric` and `DropEvidenceProvenance::Controlled` so a reader
   can always separate an injected loss from a modelled one.

## Latent parameters

These are `calibrated-opaque`: the study fits them and reports the fitted value
with the band the checks leave around it. No value is stated here.

| parameter | meaning | plausible range |
|---|---|---|
| `rx_drain_bps` | the rate the ingress meter drains wire bits at | 90e9 to 99e9 |
| `rx_ingress_bytes` | the finite receive buffer at the port | 64 KiB to 1 MiB |
| `rx_pps_per_qp_ud` | per-QP UD receive packet-rate cap | 2.9e6 to 3.2e6 |
| `rx_pps_per_qp_rc` | per-QP RC receive packet-rate cap | 2.8e6 to 3.2e6, or unset |
| `rx_pps_per_nic` | per-NIC receive packet-rate cap | 9.0e6 to 10.5e6 |

The fit is constrained, not free. `rx_drain_bps` is pinned from two independent
directions at once: the loss it implies under saturation sets the equilibrium
goodput through the go-back-N efficiency below, and the same deficit sets how
long a gap has to be for the buffer to drain between bursts.
`rx_ingress_bytes` only has to be large enough that a single burst does not
overflow it at the largest gap that is clean.

## Closed forms

**Go-back-N efficiency.** With per-packet loss probability `p` and `R` packets
in flight over one recovery round trip, the responder discards every packet
after the lost one until the replay arrives, so

```
eta = (1 - p) / (1 + R * p)        goodput = C * eta
R   = (round trip + standing queue delay) * send rate / wire bits per packet
```

For one saturated RC QP with no fabric queue, `R = 2.1 us * 98.617e9 / 33280 =
6.2`, plus one packet of detection, so `R = 7.2`.

**Ingress meter.** Arrival at wire rate `S`, drain at `D < S`. The buffer grows
at `S - D` while a burst is on and drains at `D` while it is off, so a burst of
`B_w` wire bits leaves `p * B_w` bits of backlog with `p = 1 - D / S`, and the
gap that clears it is `G = p * B_w / D`. A gap shorter than `G` accumulates
backlog burst over burst until the buffer overflows, which is why a
sufficiently short gap is indistinguishable from no gap at all.

**Loss equilibrium.** Under continuous offer the meter accepts `D` and discards
the rest, so `p = 1 - D / S` and the equilibrium goodput is `C * (1 - p) /
(1 + 7.2 p)`. The registered 78 to 92 Gb/s window therefore corresponds to
`p` between 0.68 and 2.91 percent, i.e. `D` between 95.74e9 and 97.94e9.

**Incast tax.** With the campaign's own definition,

```
tax = 1 - goodput at the receiver / (sum of the senders' wire transmit rates)
amplification = tax / injected loss rate
```

## Sweeps

Every sweep varies at least two parameters.

### (a) Gap sweep, 12 cells

Message size {8 KiB, 64 KiB, 1 MiB} times inter-burst gap {0, 4, 100, 368} us,
one RC QP, send-queue depth 1024, MTU 4096, through the two-endpoint wire at
100 G. Gap 0 is a continuous offer, not a burst train.

Checks:

1. `gap_discards` (categorical, 12 rows). The responder's `rx_discards_phy` is
   nonzero at gap 0 for every size. At 8 KiB it is zero at 4, 100 and 368 us.
   At 64 KiB it is nonzero at 4 us and zero at 100 and 368 us. At 1 MiB the
   registered claim is only the monotone one: the smallest clean gap is
   strictly larger than it is at 64 KiB, because the threshold scales with the
   burst's byte count and the burst is 128 messages at every size.
2. `paced_goodput` (2 rows). In-burst goodput at the smallest clean gap is
   within 15 percent of 92.2 Gb/s at 8 KiB (gap 4 us) and of 97.3 Gb/s at
   64 KiB (gap 100 us).
3. `saturated_goodput` (2 rows). Gap-0 goodput at 8 KiB and 64 KiB is inside
   the 78 to 92 Gb/s window.

### (b) Depth ratio, 4 cells

Send-queue depth {1, 1024} times message size {8 KiB, 64 KiB} at gap 0 with the
receive pipeline enabled.

4. `depth_ratio_measured` (2 rows). The depth-1024 over depth-1 goodput ratio
   is within 20 percent of the measured 5.9 at 8 KiB and 1.57 at 64 KiB. This
   is the band slice B missed by design; closing it is the point of this slice.
5. `depth1_unchanged` (2 rows). The depth-1 goodput with the receive pipeline
   enabled is within 1 percent of the slice-B depth-1 law value, because at
   depth 1 the arrival rate is far below any receive ceiling and the receive
   pipeline must not cost anything there.

### (c) UD receive cap, 8 cells

Offered packet rate {2, 3, 4, 5.85} Mpps times UD QP count {1, 4}, 2 KiB
messages so each is one packet, driven straight into the responder's
`rnic_cm_rx_packet` at the offered rate. The offered rate is an input here, not
a transmit-side result, because the measured 5.85 Mpps is above the profile's
own single-QP transmit message rate.

6. `ud_cap` (2 rows). At offered 4 and 5.85 Mpps on one QP, the delivered rate
   is within 10 percent of 3.07 Mpps.
7. `ud_passthrough` (2 rows). At offered 2 and 3 Mpps on one QP, delivered
   equals offered exactly.
8. `ud_silent` (4 rows, one per offered rate at one QP). Discarded equals
   offered minus delivered exactly, `rx_discards_phy` equals the discarded
   count exactly, and the transport is silent: `out_of_sequence`,
   `packet_seq_err` and `roce_adp_retrans` are all zero and no NAK is emitted.
9. `ud_aggregate` (1 row). At 4 QPs offered 5.85 Mpps each, delivered is within
   10 percent of 9.65 Mpps, so a per-NIC ceiling binds below four times the
   per-QP one.

### (d) Incast, 6 cells plus one primary

Two requesters into one responder over a shared fabric egress queue, RC WRITE,
four QPs per sender. Injected Bernoulli loss {0.5, 1.65, 5} percent on data
packets times message size {64 KiB, 1 MiB}. The primary cell is 1.65 percent at
1 MiB, which is the measured one.

10. `incast_wire` (1 row, primary). Wire utilization at the receiver is at
    least 97 percent of the link rate.
11. `incast_tax` (1 row, primary). The goodput tax is within 25 percent of the
    measured 26.9 percent, i.e. inside 20.2 to 33.6 percent.
12. `incast_fair` (1 row, primary). The per-sender wire share is 50 plus or
    minus 2 percentage points.
13. `incast_direction` (2 rows). The tax is strictly increasing in the injected
    loss rate at both message sizes, and strictly increasing in the message
    size at a fixed loss rate. The second half is registered as the weaker of
    the two: go-back-N replays from the lost sequence number over a window the
    fabric queue sets, not over the message, so if the model shows no message
    size dependence the mechanism is named rather than the band moved.

The amplification factor is reported for every cell, not banded, because the
campaign's own closed form (`loss times message bytes over packet bytes`) does
not reproduce its own measured 16.3 and is a shape rather than a predictor.

### (e) Bidirectional against unidirectional, 3 cells

One unidirectional flow offered at the measured 93.4 Gb/s of payload, and a
bidirectional pair offered at the measured 91.8 Gb/s of payload per direction.
The offered rate is an input, because the mechanism that makes a duplex NIC
slower per direction is the shared in-NIC budget, which is BACK-58's internal
arbiter and not this slice.

14. `duplex_clean` (2 rows). At 91.8 Gb/s of payload per direction, both
    responders report `rx_discards_phy` zero and both requesters report
    `packet_seq_err` and `roce_adp_retrans` zero.
15. `simplex_dirty` (1 row). At 93.4 Gb/s of payload, the responder reports
    `rx_discards_phy` nonzero.

**Check 15 is registered in advance as an expected miss.** The measured pair
implies a responder discard threshold between 93.23 and 94.86 Gb/s of wire.
The gap sweep and the loss equilibrium of check 3 together require a drain rate
between 95.74e9 and 97.94e9 wire bits per second. The two are incompatible, so
one ingress meter with one drain rate cannot make 93.4 dirty and keep the
equilibrium inside its window; it will report 93.4 clean. Reconciling them
needs a second limiter in the receive path, and the candidate is the per-QP
receive packet-rate ceiling that the internal arbiter (BACK-58) also feeds. If
this row passes instead, the fit landed outside the range the equilibrium
implies and the failure is in check 3, not here.

### (f) Counter ledger, 6 cells

The gap-0 and gap-4 cells of sweep (a) at all three sizes.

16. `ledger_identity` (6 rows). The requester's `packet_seq_err` and the
    responder's `out_of_sequence` agree within 10 percent of each other and
    together bracket the number of loss events (injected plus ingress) within
    10 percent. A packet discarded at the PHY produces no signal itself; the
    signal is the next in-order packet arriving out of sequence, so the two
    counters are one loss event each and track each other, exactly as the
    campaign measured them tracking 1 to 1.
17. `inert_marking` (12 rows, every cell of sweep (a)).
    `np_ecn_marked_roce_packets` is exactly zero everywhere, as silicon
    reports it.
18. `firmware_variant` (2 rows). On the `fw_16_31` variant
    `local_ack_timeout_err` counts every timeout-driven recovery; on
    `fw_16_32` it stays zero for the same stimulus, and every other counter is
    identical between the two.

### (g) Fatal guards

A fatal guard voids the run. It is never reported as a fraction of passing
checks.

- **Deterministic replay identity.** The designated replay cell (`cx5_100g`,
  64 KiB, gap 4 us, depth 1024, 1.0 percent deterministic 1-in-100 injected
  loss) is run twice in one process and both the CSV row and both facade
  transaction traces must be byte-identical. Deterministic loss rather than
  Bernoulli, so the guard tests the model and not the generator.
- **Byte conservation.** In every cell the payload bytes the requester
  packetized equal the payload bytes delivered plus the payload bytes
  discarded plus the payload bytes still in flight, and every completed message
  carries exactly its offered byte count.
- **No completion before delivery.** No CQE becomes visible for a message
  before every packet of that message has been delivered to the responder's
  receive processor, and no message completes twice.
- **Counter monotonicity.** Every NIC-named counter is non-decreasing across
  the run on both endpoints.
- **Pacing integrity.** Zero late releases in every cell.
- **Identity off.** With the receive pipeline disabled the probe reproduces the
  slice-B rows byte for byte, and every existing native test passes unchanged.

## What this study cannot show

It cannot show congestion response, ECN marking, CNP generation or handling,
the in-NIC arbiter, the loopback path or PFC, all of which are BACK-58. It
cannot independently confirm the drain rate or the buffer size: they are fitted
here from four measured facts (two drain-window thresholds, one equilibrium
window and one paced goodput pair) and are opaque parameters of the silicon,
not readable ones. It cannot show why a duplex NIC is slower per direction than
a simplex one, only what the ingress meter does once it is. And it cannot
settle the incast amplification factor from first principles: the replay depth
is set by the queueing delay in front of the responder, which is a fabric
property this slice supplies as a declared constant.
