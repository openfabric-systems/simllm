# HACC fabric model: pre-registered expectations

Written and frozen before any simulation run of this study. Purpose: carry the
measured HACC leaf fabric into SimLLM as a profile, render it onto the existing
RoCEv2 DCQCN packet path, measure exactly what that path can hold, and freeze
the full-chain acceptance bars that only the golden-model endpoint work can
clear.

Checks follow the format of
[examples/cx5_msgsize_v1/expectations.md](../cx5_msgsize_v1/expectations.md):
M-checks describe what the configured comparator is expected to do given its
own mechanics, and the acceptance bars in the last section describe measured
hardware behavior that is registered now and scored later. Every check below
gets an explicit PASS, FAIL, VOID or BLOCKED in RESULTS.md with its number,
including the ones registered as expected failures. This file is never edited
after the first run.

## The measured fabric constants

Source records, all from the mlx5 campaign and all evidence class `inferred`:
`RESULTS-p6-fabric.md` and its freeze `expectations-p6-fabric.md`,
`FINDINGS-cx5.md` section D, and the raw tables under `data/p6/`
(`buffer.csv`, `ecn_fit.txt`, `dcqcn_summary.csv`, `loneflow.csv`,
`udcap.csv`, `latency_matrix.csv`, `uplink.csv`). No switch counter was
readable from the endpoints, so every constant is bracketed by endpoint
counters and per-packet logging rather than read off the device. That is what
`inferred` means here, and it is why no field of this profile carries
`documented`.

| Field | Value | Evidence for it |
|---|---|---|
| `switch_count` | 1 | one non-blocking leaf: all six host pairs fall in one latency class, range 0.13 us over 12 directions (`latency_matrix.csv`) |
| `host_ports` | 4 | the four hosts that were measured; a directed ring of four ran 391.94 Gb/s aggregate with every port at 97.98 (`uplink.csv`) |
| `port_bps` | 100000000000 | 100000 Mb/s reported on every port, and no uplink is reachable from these hosts |
| `pipe_latency_ps` | 515000 | 2.08 us 2 B WRITE floor over four pipe traversals; the same 515 ns the cx5 study derived from the route construction |
| `pipes_per_path` | 4 | host queue, host to leaf, leaf ingress, leaf to host on the forward path and its mirror on the ACK path |
| `egress_buffer_bytes` | 5200000 | `t_drop x excess` = 5.39 / 5.04 / 5.05 MB at excess 4.76 / 9.74 / 19.68 Gb/s, spread 6.8 percent over 12 runs; independent drain-tail estimate 5.76 MB (`buffer.csv`) |
| `ecn` | `"none"` | 0 CE-marked packets in 670 M packets over 14 runs, at DSCP 0 and DSCP 26, with every packet ECT(0) and the buffer full and dropping 6.36 percent. Kmin, Kmax and Pmax are undefined, not small (`ecn_fit.txt`) |
| `pfc_enabled` | false | PFC off on all eight priorities in hardware |
| `pause_honoured` | false | hosts emit 802.3x pause under load, about 760 frames per run; `rx_global_pause` is 0 on every host over the whole node lifetime, so the switch has never paused a host |
| `ecmp_paths` | 1 | 16 fresh 5-tuples unimodal, range 0.10 us; no path diversity is reachable from these hosts |

Scope, stated because the profile does not carry it: the cluster-wide Clos, its
uplinks, its ECMP and whatever marking policy it runs are all out of reach from
these four hosts and are not described by this profile.

## Frozen configuration

### Topology

Four nodes on one leaf, the smallest geometry that keeps every pair one hop
apart while still expressing a 3 to 1 fan-in. The constraint chain the backend
fat-tree loader and the DCQCN runtime impose is: node count a multiple of pod
size; pod size a multiple of the tier 0 down radix; tier 0 down radix divided
by tier 0 up radix equal to the oversubscription ratio, which is 1; the tier 0
up radix equal to the aggregation switches per pod; exactly two tiers; the node
count equal to the GOAL rank count; and one equal-rate link on every Clos edge.

`Nodes 4, Podsize 4, tier 0 Radix_Down 4 Radix_Up 4, tier 1 Radix_Down 1`
satisfies all of them. All four hosts hang off one leaf and the four spine
switches carry no traffic, which is deliberate: the measured fabric is a single
switch hop and a same-leaf pair reproduces it exactly.

Per-hop latency is the profile's 515 ns on both tiers, switch latency 0. The
2 B round trip then costs

```
FCT(2 B) = 4 x 515 ns + 2 x (66 B) x 8 / R + 2 x (64 B) x 8 / R
```

which at R = 100 Gb/s is `2060 + 10.56 + 10.24 = 2080.8 ns`. The cx5 study
derived the same 515 ns at 97 Gb/s and measured 2081.3 ns, so the hop count is
already validated and this study inherits it rather than refitting it.

### Rate

`-link_bps` renders the fabric's `port_bps`, 100 Gb/s, not the NIC profile's
97.1 Gb/s goodput asymptote. The fabric owns the link. Two reasons this is the
right half to render: the buffer identity below is only arithmetic if the drain
rate is the port rate, and the NIC's separate goodput asymptote is exactly the
`link_bps` gap field the NIC profile already registers against the endpoint
calibration task. Rendering the fabric rate does not close that gap; it puts it
where it belongs.

The loader parses `Downlink_speed_Gbps` as a whole number and the runtime
rejects a topology whose rate differs from `-link_bps`, so 100 is written in
both places and no rounding is involved at this rate.

### How "no marking" is realised, exactly

The measured switch does not mark. The runtime has no switch for that, and the
two obvious spellings are both refused by its fatal configuration guard
(`dcqcn_atlahs_runtime.cpp`, `validate_config`):

- `-ecn_pmax_ppm 0` is refused: `ecn_pmax_ppm == 0` is in the guard.
- Kmin and Kmax above the buffer are refused:
  `ecn_kmax_bytes >= ns_tm3_egress_buffer_bytes` is in the same guard.

The legal chain is `0 <= Kmin < Kmax < egress buffer` with
`0 < Pmax <= 1000000`, and the marking predicate returns false whenever the
egress occupancy is at or below Kmin. So "no marking" is realised by putting
the whole marking band out of reach at the lowest legal probability:

| buffer B | `-ecn_kmin_bytes` | `-ecn_kmax_bytes` | `-ecn_pmax_ppm` |
|---|---|---|---|
| 5200000 | 5199998 | 5199999 | 1 |
| 2600000 | 2599998 | 2599999 | 1 |

Marking then needs the egress port to hold at least `B - 1` bytes, that is the
buffer full to within two bytes of its tail-drop threshold, and even there the
per-packet probability is 1e-6. This is not zero by construction, so it is
checked rather than assumed: guard G3 below requires `ecn_marked_packets` to be
exactly 0 in every cell. The residual, that the path has no drop-only switch
mode and no endpoint that can generate a notification without one, is
registered as HTSIM-38.

### Frozen flag vector

Rendered from the `cx5_100g` NIC profile and the `hacc_leaf_4x100g` fabric
profile. Every value is stated so the run is reproducible from this file alone.

| Flag | Value | Owner and why |
|---|---|---|
| `-link_bps` | 100000000000 | fabric `port_bps`; the topology carries `Downlink_speed_Gbps 100` |
| `-max_wire_packet_bytes` | 4096 | NIC, measured RoCE active MTU |
| `-data_header_bytes` | 64 | NIC, the backend's own per-packet wire header |
| `-pfc` | off | fabric and NIC agree: PFC off in hardware, and the switch ignores the pause the hosts emit, so drop on overflow is the measured behavior and the flag reproduces it exactly |
| `-recovery` | gbn | NIC, the measured loss signature is go-back-N with burst-collapsed sequence errors |
| `-loss_rate_cut` | on | NIC, the measured congestion response is a rate cut on loss |
| `-silent_rto_us` | 67108 | NIC, local ACK timeout 14 |
| `-ecn_kmin_bytes` | B minus 2 | fabric `ecn = "none"`, see above |
| `-ecn_kmax_bytes` | B minus 1 | as above |
| `-ecn_pmax_ppm` | 1 | as above |
| `-egress_buffer_bytes` | 5200000, or 2600000 in the sensitivity arm | fabric `egress_buffer_bytes`, per-port tail drop |
| `-shared_buffer_bytes` | 4 x the egress buffer | the switch-wide pool was not observable; setting it to the sum of the four measured per-port pools makes the per-port limit the binding one, which guard G4 checks |
| `-seed`, `-ecn_seed` | 1 | one measured path and no marking, so neither seed can change an outcome. This is why the study runs one seed where the cx5 study ran three |

### Fields the flag vector cannot carry

With `ecn = "none"` the only fabric field that no flag carries honestly is the
drop-only semantics itself: the runtime can put the threshold out of reach but
cannot turn marking off. That single gap is HTSIM-38's content. Everything else
lands: `port_bps` and `pipe_latency_ps` through the topology,
`egress_buffer_bytes` through the egress pool, `pfc_enabled` and
`pause_honoured` jointly through `-pfc off` (a switch that ignores pause and a
model that never sends it have the same drop-on-overflow behavior), and
`switch_count`, `host_ports`, `pipes_per_path` and `ecmp_paths` through the
topology geometry.

## E-LAT: the latency anchor

2 B message, one sender to one receiver, third and fourth ranks idle. One cell.

- **M1** (2 B floor): the message completes within **15 percent** of the
  measured **2.08 us**. Registered prediction: 2.081 us, from the arithmetic
  above. Expected PASS. A FAIL here means the geometry or the per-hop latency
  did not carry over from 97 to 100 Gb/s, and every other number in the study
  would be suspect.

## E-MSG: single-flow goodput

Message size in `{4 KiB, 64 KiB, 1 MiB, 4 MiB}` x egress buffer in
`{5200000, 2600000}` bytes. Eight cells. One sender to one receiver, one
message at a time.

- **M2** (rate tracks the rendered link): the 4 MiB single-flow goodput is
  within **3 percent** of the rendered link rate, 100 Gb/s, in the
  5200000 B arm. Registered prediction: **97.7 to 97.8 Gb/s**, that is 2.2 to
  2.3 percentage points low. The systematic is named in advance: the 64 B wire
  header on a 4096 B packet is worth 1.56 percent and the finite topology
  offset accounts for the rest. Expected PASS, with little margin, which is
  stated so a pass is not read as a tight fit.
- **M2-buffer** (reported, no bar): the eight cells are reported as a curve,
  and the two buffer arms of each size are expected to be **identical to the
  picosecond**, because a single flow into an idle port never queues. A
  difference would mean the buffer configuration reaches something it should
  not.

## E-BUF: the buffer identity

Sender count in `{2, 3}` x egress buffer in `{5200000, 2600000}` bytes. Four
cells. Every sender posts one 32 MiB message to the same receiver, so each
sender is one long-lived flow rather than a stream of short ones, and all
senders start together.

The measured estimator (P6c) paced two senders to a fixed excess over the port
rate and read `B = t_drop x excess`. **No sub-line-rate paced source is
expressible on this path**: the RoCE source is open-loop rate-paced with no
window and no configurable rate cap, its only rate input is `-link_bps`, and a
GOAL `calc` chain would pace by fragmenting the stream into short per-message
flows, whose drops have no successor to expose them and therefore land on the
67 ms silent RTO instead of a NACK (the cx5 stage-1 finding). The registered
110 Gb/s operating point and its 4.16 ms first drop are therefore **not
runnable as such**. Sender count is the knob that is available: two senders
offer 200 Gb/s into a 100 Gb/s port, three offer 300, so the excess is
100 or 200 Gb/s.

First-drop instrument. The final manifest carries totals, not timestamps, so
the first drop is read from the sender state trace as the earliest `gbn-nack`
row. That row lags the drop by the queueing of the packet that exposes the
gap, which enters a full buffer and therefore waits `B / R_link`, plus about
one 1.05 us round trip:

```
t_nack = B / excess + B / R_link + RTT
B_hat  = t_nack x excess x R_link / (R_link + excess)
```

- **M3a** (buffer identity, scored): `B_hat` is within **20 percent** of the
  configured egress buffer in all four cells. Registered predictions for
  `t_nack`: **833 us** at 2 senders and 5.2 MB, **625 us** at 3 senders and
  5.2 MB, **417 us** at 2 senders and 2.6 MB, **313 us** at 3 senders and
  2.6 MB. Four distinct times from one identity over two swept parameters.
  Expected PASS. If no `gbn-nack` row exists in a cell, that cell is VOID and
  the reason is reported, because the estimator then has no input.
- **M3b** (steady-state loss equals the excess rate), **registered as VOID by
  construction**: the measured claim that loss in steady overflow equals the
  excess rate to better than 0.04 Gb/s is a property of an **open-loop paced**
  sender. On this path the senders are DCQCN closed-loop with
  `-loss_rate_cut on`, so they back off after the first loss and the excess
  collapses toward the fair share. The check cannot be scored and is recorded
  as void with its reason. The observed loss fraction, drop counts and the
  sender rate trajectory are reported next to it anyway, because a run that
  did not back off would itself be a finding.

## E-INCAST: fan-in behavior and the comparator baseline

Message size in `{262144, 1048576}` bytes x egress buffer in
`{5200000, 2600000}` bytes. Four cells. Two senders to one receiver, 32
independent messages per sender.

- **M4** (fair share): the two senders' shares are **50 plus or minus 2
  percentage points** in every cell. The share is measured as per-sender
  throughput, that sender's payload bytes over that sender's own flow
  completion span, normalized across the two senders. It is deliberately not
  the share of delivered bytes: with equal message counts that is 50/50 by
  construction and measures nothing. Measured hardware: 50.18 against
  50.13 Gb/s in steady state, that is 50.02 / 49.98.
- **B1** (comparator baseline): the 2 to 1 receiver goodput at 1 MiB is
  registered as **collapsed**, with the value **bounded above by 20 Gb/s**.
  This is not a prediction about the hardware; it is the cx5 stage-1 finding
  carried forward. That study measured **7.351 Gb/s** on the same packet path
  against a measured 73.9, because each GOAL message is its own flow, a drop
  is usually the last outstanding packet of that flow, no NACK follows and the
  sender waits a full 67 ms silent timeout. The HACC fabric's 5.2 MB buffer is
  six times smaller than the cx5 study's 32 MiB default, so more cells drop
  and the collapse should if anything deepen. A value **above** 20 Gb/s would
  be the surprise, and would mean the smaller buffer changed the loss regime
  rather than only its depth.

## Fatal guards

A guard failure voids the run; the study reports it and stops claiming
anything about the affected cells.

- **G1 determinism**: the 2 sender, 5200000 B cell of E-BUF is run twice with
  an identical flag vector. Job completion time, dropped-packet count,
  retransmitted-packet count and the first `gbn-nack` timestamp must be
  **exactly equal**.
- **G2 byte conservation**: in every cell every flow completes
  (`completed_flows` equals the flow count) and the sum of the completion CSV's
  payload bytes equals the offered payload bytes **exactly**.
- **G3 no marking**: `ecn_marked_packets` is **exactly 0** in every cell. This
  is what makes the threshold placement above a realisation of the measured
  drop-only switch rather than an approximation of one.
- **G4 the per-port pool binds**: `ns_tm3_shared_pool_dropped_packets` is
  **exactly 0** in every cell, so every drop is a per-port egress-domain drop
  and the buffer identity is about the measured per-port pool.

Also recorded, not a guard: `dcqcn_pfc_pause_frames` is expected to be 0
throughout, since PFC is off.

## Full-chain acceptance bars, registered now and BLOCKED

These are the measured endpoint behaviors that this fabric profile is meant to
reproduce once the endpoint exists. None of them is scorable on the current
packet path, and each is blocked on golden-model work that is not merged. They
are cross-referenced by description rather than by task number, because those
numbers live on an unmerged branch.

| Bar | Target | Blocked on |
|---|---|---|
| **A1** incast goodput | 2 to 1 RC receiver goodput **74 to 78 Gb/s plus or minus 15 percent** while the wire carries **99.3 Gb/s** | the golden-model transport work, which gives a sender per-QP state that survives across messages so a loss is repaired by retransmission rather than by a timeout |
| **A2** lone-flow ingress loss | **0.18 percent plus or minus 30 percent** above about 94 Gb/s, in bursts of **50 to 100 packets** | the golden-model receive-path work, which gives the responder an ingress meter that can discard; today the packet path drops only at switch queues |
| **A3** DCQCN recovery | return to at least 95 percent of the pre-congestion rate in **447 ms plus or minus 25 percent**, with additive increase near 0.1 Gb/s per ms | the golden-model rate-control work, which owns the DCQCN timer constants and the increase shape |
| **A4** CNP rate | **283 per second per congested queue pair, plus or minus 30 percent**, generated at the receiving NIC | HTSIM-38, the endpoint-side congestion-notification hook, since the measured fabric never marks and the current path can only originate a notification from a switch mark |

A1 is the one this study measures a placeholder for: B1 above is the same
configuration on the current path, and the distance between B1 and A1 is the
size of the endpoint gap.

## Registered tasks

- **BACK-60** (Precision; P1; M): the HACC fabric profile and its full-chain
  validation, carrying bars A1 to A4.
- **BACK-61** (Precision; P2; S): synchronised-clock queue-depth calibration,
  the follow-up P6 could not run because its delay column mixed two
  unsynchronised sender clocks.
- **HTSIM-38** (Completeness; P2; M): an endpoint-side congestion-notification
  hook in the DCQCN runtime, plus an explicit drop-only switch mode, so a
  fabric that never marks is expressible as itself.

## Verdict rule

Every check above gets an explicit PASS, FAIL, VOID or BLOCKED in RESULTS.md
with its measured number. A guard failure voids the run. Blocked bars are
listed with the work they wait on and are not scored.
