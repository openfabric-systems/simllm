# CX-5 message-size calibration: pre-registered expectations

Written and frozen before any simulation run of this study. Purpose: fold the
measured ConnectX-5 Ex 100 GbE campaign into the existing RoCEv2 DCQCN packet
path as a configuration, measure exactly how far that gets, and leave the
residual as registered evidence for the mechanism work. A ConnectX-7 400 G arm
runs the identical sweep with every rate-carrying field scaled by four, so the
study also tests whether "same architecture at a higher rate" is expressible
today.

Checks are target-anchored in the format of
[examples/dcqcn_micro/expectations.md](../dcqcn_micro/expectations.md):
M-checks describe what the current comparator is expected to do given its own
mechanics, T-checks describe the measured hardware behavior. A T-check FAIL
with an M-check PASS means the comparator is self-consistent but uncalibrated,
which is the scope of the profile's gap ledger and of the registry tasks it
points at. Several T-checks are registered as expected FAILs below; a
registered expected FAIL that passes is as much a finding as one that fails.

Measured anchors come from the ConnectX-5 campaign records: the depth-1
two-parameter refit `T_eff = 4.48 us, C = 97.1 Gb/s` (residuals at or below
5.1 percent), the 2 B RDMA WRITE latency floor 2.08 us, the MTU-1024 tax
5.6 percent, and the 2 to 1 incast triple (wire 99.4 Gb/s, application goodput
73.9 Gb/s, fabric loss 1.65 percent of packets).

## Frozen configuration

### Topology

Both profiles run on a three-node two-tier Clos, the smallest the runtime
accepts that still expresses a 2 to 1 fan-in. The constraint chain read out of
the backend's fat-tree loader is: pod size must equal (tier 0 down-radix) x
(tier 1 down-radix); at oversubscription 1 the tier 0 up-radix must equal its
down-radix; and the comparator additionally requires exactly two tiers, a node
count equal to the GOAL rank count, and one equal-rate link on every Clos edge.
`Nodes 3, Podsize 3, tier 0 down-radix 3, up-radix 3, tier 1 down-radix 1`
satisfies all of them. All three hosts therefore hang off one leaf, and the
three spine switches carry no traffic. That is deliberate: the measured fabric
is a single switch hop, and a same-leaf pair reproduces it exactly.

Hop count. The comparator builds the forward route as host queue, host-to-leaf
pipe, leaf ingress, leaf-to-host queue, host-to-leaf pipe, sink; the reverse
route for the ACK is the mirror image; and a RoCE flow completes when the ACK
for its last packet reaches the source. A same-leaf message therefore crosses
**four pipes** on the round trip and pays two store-and-forward serializations
in each direction. With per-hop latency L and link rate R, the 2 B floor is

```
FCT(2 B) = 4 L + 2 x (66 B) x 8 / R + 2 x (64 B) x 8 / R
```

At R = 97 Gb/s the serialization terms sum to 21.4 ns, so **L = 515 ns** gives
2081 ns, inside the registered 2.08 us plus or minus 15 percent band. The
same arithmetic applied to the current 400 G cross-leaf pair (eight pipes at
1000 ns, four 4096 B store-and-forward stages at 400 G, four ACK stages) gives
8333 ns, which reproduces the published 8.3 us offset of that configuration
and is the cross-check that this derivation is the right one.

Both topology files use per-hop latency 515 ns and switch latency 0 on both
tiers. The only difference between them is the link rate.

### Frozen flag vector

Rendered from the `cx5_100g` profile. Every value is stated here so the run is
reproducible from this file alone.

| Flag | cx5_100g | cx7_400g | Why |
|---|---|---|---|
| `-link_bps` | 97000000000 | 388000000000 | the measured goodput asymptote C = 97.1 Gb/s, and 4 x that. The loader parses `Downlink_speed_Gbps` as a whole number, so 97.1 is not expressible; 97 is the nearest legal value, a 0.10 percent deviation, and the comparator rejects any topology whose link rate differs from this flag |
| `-max_wire_packet_bytes` | 4096 | 4096 | measured RoCE active MTU 4096 |
| `-data_header_bytes` | 64 | 64 | the backend's own per-packet wire header; RoCEv2 headers plus FCS are about 58 B |
| `-pfc` | off | off | measured PFC off on all eight priorities, one 262016 B lossy pool, zero headroom |
| `-recovery` | gbn | gbn | the measured loss signature is retransmit amplification, not selective repair |
| `-loss_rate_cut` | on | on | measured ECN-first, loss-second congestion response |
| `-silent_rto_us` | 67108 | 67108 | the campaign ran a local ACK timeout of 14, i.e. 4.096 us x 2^14 = 67.109 ms |
| `-ecn_kmin_bytes` | 102400 | 409600 | the 100 G D2 vendor row of [msg-size-vs-bandwidth.md](../../docs/papers/msg-size-vs-bandwidth.md), read in the KiB units the binary's own defaults use (65536 = 64 KiB). The endpoint thresholds are unreadable on the measured deployment, so these are declared, not measured. Scaling by four reproduces that document's D3 400 G row exactly |
| `-ecn_kmax_bytes` | 409600 | 1638400 | as above |
| `-ecn_pmax_ppm` | 250000 | 250000 | the binary default, retained: the D2 row does not state Pmax and the deployment cannot be read |
| `-shared_buffer_bytes` | 33554432 | 33554432 | the binary default, except in the E-INCAST buffer sweep below |
| `-egress_buffer_bytes` | 33554432 | 33554432 | as above |
| `-seed`, `-ecn_seed` | 1 | 1 | except in the E-INCAST seed sweep |

Switch buffer sizes are a fabric property, not a NIC property, so they are not
carried by the profile and are swept explicitly instead.

### Fields the flag vector cannot carry

The profile's gap ledger names six fields with no CLI target: the separate wire
link rate (the comparator has one rate, not a wire rate and a goodput rate),
the endpoint initiation cost `t_eff_ps`, the send-queue depth, the per-QP and
per-NIC packet-rate ceilings, and the responder ingress meter. Those six are
exactly the fields the registry tasks cover, and they are why several T-checks
below are registered as expected FAILs.

## E-MSG: message size vs goodput, both profiles

Sizes `{4 KiB, 16 KiB, 64 KiB, 256 KiB, 1 MiB, 4 MiB}` x Q in `{1, 16}`
independent concurrent messages x profile in `{cx5_100g, cx7_400g}`, one
sender to one receiver on the same leaf, third rank idle. Twenty-four cells.
Goodput is `Q x S / JCT`. There is no contention on this path and no ECMP
choice to make (one path), so the cells are deterministic and one seed is used.

Fits are two-parameter least squares of `B = S / (T_eff + S/C)` over the six
Q = 1 points of one profile.

- **M1** (law self-consistency): every Q = 1 point of both profiles lies within
  **10 percent** of the fitted fixed-offset law of its own profile. This checks
  that the comparator still is a fixed-offset law after the retune, not that
  the offset is right.
- **M2** (C tracks the configured rate): the fitted `C` is within **3 percent**
  of that profile's rendered link rate (97 Gb/s and 388 Gb/s). The expected
  systematic is the 64 B wire header, worth 1.56 percent, so the fit should
  land just below the flag. The 4 MiB Q = 1 point, which anchors the fit, is
  reported next to it.
- **M3** (latency anchor): a 2 B Q = 1 message on `cx5_100g` completes within
  **15 percent** of 2.08 us. This is the topology calibration check, and it is
  the reason the per-hop latency is 515 ns.
- **M4** (scaled profile is the same curve): `cx7_400g`'s fitted `C` is
  **4.00 x** `cx5_100g`'s within **3 percent**, and its fitted `T_eff` is
  within **15 percent** of `cx5_100g`'s. The band on `T_eff` is 15 rather than
  3 percent for a stated reason: only the propagation part of the offset is
  rate-independent, while the store-and-forward part falls by four, so the two
  offsets are predicted to differ by about 11 percent.
- **T1** (T_eff against hardware), **registered as an expected FAIL**: the
  `cx5_100g` fitted `T_eff` is within **15 percent** of the measured 4.48 us.
  It will not be. The offset in this path is entirely topology, and the hops
  were calibrated to the measured 2.08 us latency floor, which the same
  hardware also exhibits. A model with no endpoint initiation cost cannot
  satisfy both anchors at once: hitting 4.48 us would require about 1120 ns per
  hop and would put the 2 B floor at 4.5 us, more than twice the measured
  value. The registered prediction is a fitted `T_eff` near 2.4 us, about
  46 percent below the target, and the deciding number is which of the two
  hardware anchors the configuration is able to hold.
- **T2** (depth and Q amortization), **registered as an expected FAIL**: the
  ratio of the Q = 16 aggregate goodput to the Q = 1 goodput at 64 KiB matches
  the measured send-queue-depth ratio at the same size within **20 percent**.
  Measured, depth 1 to depth 16 at 64 KiB is 49.6 to 78.1 Gb/s, a ratio of
  1.57, and the hardware saturates against a separate ceiling. The comparator
  has no window and no send queue: Q independent messages behave exactly like
  one message of Q x S, so the modeled ratio is the law's own ratio between
  64 KiB and 1 MiB, about 1.75 to 1.9 at 100 G, and it keeps rising with Q
  instead of saturating. Reported statistic: the modeled ratio, the measured
  ratio, and the Q = 16 aggregate against the Q = 1 point at 16 x 64 KiB, which
  should agree to within a percent and is the direct evidence of zero per-message
  cost.
- **T3** (measured curve, no bar, reported): the `cx5_100g` Q = 1 curve is
  reported against the measured depth-1 WRITE rows at the same six sizes (7.04,
  21.95, 49.63, 79.23, 73.95, 77.03 Gb/s, medians of the two directions). No
  band is registered for this comparison because the measured rows at 1 MiB and
  above sit in the loss equilibrium the comparator cannot enter, so only the
  four sizes at or below 256 KiB are a like-for-like comparison. Registered
  direction: the model is expected to sit **above** the measurement at every
  size, because its offset is smaller.

## E-MTU: MTU tax at 1 MiB, cx5_100g only

`-max_wire_packet_bytes` in `{1024, 4096}` at 1 MiB, Q = 1, `-data_header_bytes`
64 in both arms. Two cells.

- **M5**: the goodput tax `1 - B(1024) / B(4096)` is **5.6 plus or minus 2
  percentage points**, the measured MTU-1024 tax (73.53 against 77.94 Gb/s
  at 1 MiB). The comparator expresses this axis directly through the header
  fraction, so the arithmetic prediction is 4.4 percent, inside the band. This
  is the one axis the configuration is expected to get right for the right
  reason.

## E-INCAST: 2 to 1 fan-in and 1 to 2 fan-out at 1 MiB, cx5_100g only

Pattern in `{2 to 1, 1 to 2}` x switch buffer in `{33554432, 262016}` bytes x
seed in `{1, 2, 3}`. Twelve cells. Each sender posts 32 independent 1 MiB
messages, so an episode moves 32 MiB per sender and lasts long enough for the
DCQCN timers (50 us CNP interval, 55 us update period) to act. In the 2 to 1
pattern ranks 0 and 2 send to rank 1; in the 1 to 2 pattern rank 0 sends to
ranks 1 and 2. Results are reported as the median over the three seeds with the
minimum and maximum alongside.

The 262016 B arm exists because the large-buffer arm cannot drop: 64 MiB
offered into a 32 MiB switch buffer at a 2 to 1 fan-in never overflows, so the
loss mechanism the measurement is about would be unobservable. 262016 B is the
measured single lossy pool of the ConnectX-5 port buffer, used here as the
nearest available analog for a switch egress pool; the measured fabric's own
switch buffers were not observable, so this arm is an analog, not a
measurement.

Definitions used in the checks: per-flow goodput is `S / FCT`; the offered
packet count per sender is `new_packets_sent + rtx_packets_sent` from the state
trace; `loss_rate` is `ns_tm3_dropped_packets` over the offered packet count;
receiver wire rate is the delivered wire bytes (offered minus dropped, at
4096 B each) over the makespan; and `goodput_tax` is
`1 - goodput / receiver wire rate`.

- **M6** (fair share): the two senders' shares of the delivered goodput are
  **50/50 plus or minus 2 percentage points** in every 2 to 1 cell, both
  buffers, all three seeds. Measured 50.4/49.6.
- **M7** (fan-out clean): in every 1 to 2 cell the aggregate goodput is at
  least **95 percent** of the E-MSG Q = 16 aggregate at 1 MiB, the split is
  **50/50 plus or minus 2 percentage points**, and both the dropped-packet
  count and the retransmitted-packet count are **exactly zero**. Measured
  97.83 Gb/s, split 50.000/50.000, every counter still.
- **M8** (go-back-N amplification identity), **registered as an expected FAIL
  with a stated substitute**: in the 262016 B arm,
  `goodput_tax ~= loss_rate x message_bytes / packet_bytes` within
  **25 percent**. It will not hold, and the measurement does not hold it
  either. At 1 MiB and 4096 B packets that factor is 256, so 1.65 percent loss
  would predict a goodput tax above 100 percent; the measured tax is 25.7
  percent, an implied amplification of **15.6 packets per loss**, not 256. The
  comparator's own mechanics predict the same kind of number for the same kind
  of reason: its NACK handler rewinds the sender to the last cumulatively
  acknowledged packet, which is a bandwidth-delay-product-scale rewind, not a
  rewind to the start of the message. The registered substitute, and the number
  the check reports, is the implied amplification
  `A = goodput_tax / loss_rate` in packets, compared against the measured
  **15.6 packets within a factor of 2**. Both halves are recorded so the
  literal identity's failure is on the record for the measured side as well as
  the modeled side.
- **T4** (incast goodput): the 2 to 1 receiver goodput is within **15 percent**
  of the measured 73.9 Gb/s. Registered honest expectation: **FAIL on the
  33554432 B arm** (no loss is possible there, so the model should return
  roughly the full 95.5 Gb/s payload asymptote, about 29 percent high), and
  **FAIL on the 262016 B arm as well, high**, because the model's rewind is
  bandwidth-delay-product-scale while the hardware's amplification is larger.
  If the 262016 B arm lands inside 15 percent, that is a registered surprise
  and the buffer analog, not the mechanism, is what earned it.
- **T5** (incast wire): the 2 to 1 receiver wire rate is within **2 percentage
  points** of **99.4 percent** of the profile's rendered link rate, the
  measured wire utilization. Registered expectation: **PASS on both arms**. A
  saturated fan-in keeps the bottleneck link busy whether or not the bytes it
  carries are useful, and this check is what separates "incast is a tax" from
  "incast is a collapse".

## Verdict rule

Every check listed above gets an explicit PASS or FAIL in RESULTS.md with its
measured number, including the ones registered as expected FAILs. A T-check
FAIL with its M-checks passing is read as self-consistent but uncalibrated and
is routed to the registry task that owns the missing field. This file is never
edited after the first run.
