# RNIC golden C model: mlx5-class endpoint

## Summary

The RNIC golden C model is a deterministic, transaction-level C++17 model of
an mlx5-class RDMA NIC endpoint (ConnectX-5 at 100 GbE calibrated, ConnectX-7
at 400 GbE declared as the same architecture at higher rate). It is one
module with three consumers and one timing authority:

1. Inside SimLLM it is the native RNIC hardware path under
   `simllm/backends/rnic/`, the sole mutable WQE and packet authority in a
   structural run.
2. Toward htsim it dispatches packet attempts and consumes fabric events
   through the existing `NetworkPort` ABI (version 2, packet-attempt scope),
   so links, switch queues, ECN marking, drops and PFC stay in htsim and the
   endpoint stays here.
3. Toward RTL it is the golden reference: a C-linkage facade with plain
   fixed-width structs, no exceptions across the boundary, a caller-owned
   clock and a replayable transaction trace, so a UVM testbench drives the
   same stimulus through DPI-C and compares timestamps and counters.

Every mechanism in the model is either measured on real silicon, inferred
from reviewed driver source, or declared and labelled as such, following the
evidence-class contract in `docs/papers/rnic-hardware-calibration.md`.
Measured performance anomalies are carried in one explicit table
(`## Anomaly table`) rather than scattered as special cases, and each row
names the mechanism that reproduces it or states that it is injected.

## Context

The native queue core already models one finite SQ and CQ per QP with a
thirteen-stage per-WQE timeline, doorbell batching, serialized WQE fetch,
QPC lookup, scheduler admission, CQE write, a transaction-level PCIe fabric
and a versioned `NetworkPort` seam. It stops at admission: one flow extent
per WQE leaves through ABI version 1, there is no packetization, no
outstanding-work window, no receive side, no transport state, no rate
control, and no counters that correspond to what a real NIC exposes. The
campaign records in the `hacc-fpga-llm` repository (`report/mlx5-campaign/`,
in particular `FINDINGS-cx5.md`) measured exactly those missing pieces on
ConnectX-5 and are the calibration source for this model.

The golden model therefore extends the queue core rather than replacing it.
The existing stages, records, evidence events and the one-authority contract
remain; the model adds the blocks below behind the same public surface.

## Interface

### Placement

- Source: `simllm/backends/rnic/` (CMake target `simllm::rnic`, the
  dependency-free native gate with warnings as errors). New headers live
  beside the existing ones under `include/simllm/rnic/`; nothing under
  `third_party/` changes.
- Composition: `RnicDevice` gains the receive and transport blocks and an
  `RnicHwProfile`; the caller keeps the clock (`progress(now_ps)`), delivers
  external events before `progress` at the same timestamp, and polls the CQ
  afterwards, exactly as today.
- Network seam: the packetizer submits one `NetworkTxDescriptor` per packet
  attempt (ABI version 2) and consumes `PacketTxStarted`, `PacketTxFinished`,
  `PacketRxArrived`, `EcnMarked`, `CnpReceived`, `Dropped`, `PfcPaused` and
  `PfcResumed` events. Version 1 flow-extent ports keep working with the
  packetizer bypassed, which preserves every accepted version 1 baseline.
- Loopback: a send whose destination is the local endpoint is admitted, does
  not leave through the port, and traverses the receive pipeline through the
  internal arbiter.

### C facade for RTL and UVM

`include/simllm/rnic/rnic_cmodel_c.h` exposes `extern "C"` entry points over
plain structs with fixed-width integers and picosecond timestamps:

| entry | role |
|---|---|
| `rnic_cm_create(const rnic_cm_profile*, const rnic_cm_config*)` | construct one endpoint from a profile and queue configuration |
| `rnic_cm_post(handle, const rnic_cm_wqe*, now_ps)` | post one work request (opcode, bytes, SGE count, destination, signaled) |
| `rnic_cm_doorbell(handle, now_ps)` | publish the accepted prefix as one doorbell batch |
| `rnic_cm_rx_packet(handle, const rnic_cm_packet*, now_ps)` | deliver one wire packet (data, ack, nak, cnp, pause) to the receive side |
| `rnic_cm_event(handle, const rnic_cm_event*, now_ps)` | deliver one fabric event (marking, drop, pause, rate) |
| `rnic_cm_progress(handle, now_ps)` | advance the endpoint to the caller's time |
| `rnic_cm_next_event_ps(handle)` | next internally scheduled time, so a testbench can step exactly |
| `rnic_cm_poll(handle, rnic_cm_cqe*, max, now_ps)` | poll completions |
| `rnic_cm_tx_next(handle, rnic_cm_packet*, max)` | drain packets the endpoint has emitted since the last call |
| `rnic_cm_counters(handle, rnic_cm_counters*)` | read the observable-state facade |
| `rnic_cm_trace(handle, path)` | write the transaction trace (stimulus and observed timeline) for replay |
| `rnic_cm_destroy(handle)` | release |

The facade is a thin wrapper over the C++ classes; it holds no state of its
own. Determinism is a contract: the same stimulus sequence and profile
produce byte-identical traces, so a trace recorded from SimLLM is the
expected-result file for the RTL testbench, and a divergence localizes to
the first differing timestamp or counter.

### Profile

`RnicHwProfile` carries every hardware parameter with its evidence class.

| field group | fields | ConnectX-5 (measured) | ConnectX-7 (declared) |
|---|---|---|---|
| link | `link_bps`, `goodput_bps`, `mtu_bytes`, `wire_header_bytes` | 100e9, 97.1e9, 4096, 64 | 400e9, 388.4e9, 4096, 64 |
| initiation | five work-queue service stages summing to `t_eff_ps` | 4.48 us lumped (`calibrated-opaque`; the split across stages is an assumption until the WQE publication campaign lands) | same |
| outstanding work | `sq_depth`, `max_inflight_bytes`, `max_inflight_packets` | 1024 default queue, inflight bounded by the queue | same |
| packet rate | `tx_pps_per_qp`, `tx_pps_per_nic`, `rx_pps_per_qp_rc`, `rx_pps_per_qp_ud`, `rx_pps_per_nic` | per-QP UD receive 5.51e6 (`calibrated-opaque`, P6 wire; a post-specified correction of the 3.07e6 the measurement engine produced), per-NIC 9.65e6 retained but unconfirmed, host-bound TX 3.87e6 single QP | scaled by 4 |
| ingress | `rx_ingress_bytes`, `rx_drain_bps`, `internal_budget_bps`, `loopback_priority` | meter sized so a saturated deep pipeline settles at the measured 78 to 92 Gb/s equilibrium and clears within the measured drain window; internal budget 197e9; wire wins | scaled by 4 |
| transport | `recovery` (go-back-N), `selective_repeat_window`, `rto_ps`, `ack_coalescing` | go-back-N, 0, 67 ms (`qp_timeout` 14) | same |
| congestion control | `dcqcn_enabled`, `ecn_stamp` (ECT(0) forced), `cnp_min_interval_ps`, DCQCN alpha, timer, byte counter, rate step, `np_cnp_threshold_bytes`, alpha start and gain, additive step and interval, rate floor | enabled at firmware default, ECT(0); the notification threshold, the limiter interval and the alpha and increase parameters are fitted by the slice-D study over declared candidate grids, because the vendor register block is not readable on the campaign hosts and the measured reaction is two to twenty times slower than its defaults | thresholds scaled by 4 |
| flow control | `pfc_enabled`, `global_pause_tx`, `pause_propagates` | false, true, false | same |
| counters | `firmware_counter_variant` | `fw_16_32` or `fw_16_31` (semantics differ for `local_ack_timeout_err`) | `fw_16_32` |

Evidence classes: `documented`, `driver-inferred`, `calibrated-opaque`,
`declared`. A field is `declared` when its value was derived by scaling
rather than measured; every ConnectX-7 field that differs from ConnectX-5 is
`declared`. The profile hashes into the effective-hardware record so a policy
comparison cannot silently change hardware.

## Blocks

| block | responsibility | measured behaviour it carries |
|---|---|---|
| queue core (existing) | SQ, CQ, doorbell batches, stage services, PCIe, evidence | fixed-offset initiation cost T_eff; doorbell batching |
| packetizer | segments a WQE into MTU-sized packets with header bytes and PSNs; emits one attempt per packet through the port | MTU 1024 tax of 5.6 percent; go-back-N amplification proportional to message bytes over packet bytes |
| outstanding-work window | bounds WQEs and bytes in flight per QP; ACK-clocked release | throughput versus queue depth (5.9x at 8 KiB between depth 1 and 1024) |
| transmit pacer | bits per second and packets per second ceilings per QP and per NIC, shared across QPs | small-message regime; multi-QP recovery of the ceiling |
| ingress meter | finite receive buffer at the port drained at a service rate; overflow discards at the PHY with no transport signal | loss equilibrium under saturation; drain window; bidirectional cleaner than unidirectional; the 0.18 percent lone-flow floor above about 94 Gb/s |
| receive processor | per-QP receive packet-rate cap; RC responder PSN check with ACK and NAK; UD delivery with silent drop beyond the cap | one UD QP absorbing 5.51 Mpps on the wire with silent discard beyond it; out_of_sequence at the responder |
| requester transport | PSN and ACK tracking; go-back-N on NAK or timeout; retransmit counters | packet_seq_err, roce_adp_retrans, local_ack_timeout_err by firmware variant |
| rate control | DCQCN notification point at the endpoint's own ingress meter and reaction point with per-QP state that persists across WQEs; ECT(0) stamping | 283 CNP per second per congested QP with no switch mark anywhere; rate cut in 3 to 39 ms and recovery in 447 ms |
| fabric egress queue (test fabric) | one switch egress port per receiver: a finite tail-drop buffer draining at the port rate, with no marking, no pause and no notification | the measured leaf's 5.2 MB per-port buffer, and the silence around it that forces the endpoint to notice congestion itself |
| internal arbiter | one processing budget shared by loopback ingress and wire ingress, wire priority | in-NIC cap near 197 Gb/s, loopback starves to about half |
| observable state | counters named as the real NIC exposes them, with the inert marking counter reproduced as inert | counter semantics that detection tools rely on |
| anomaly table | the explicit list below, with mechanism kind and test id | everything a reviewer needs to know is reproduced on purpose |

### What the P6 fabric campaign changed in two of these blocks

The P6 campaign probed the fabric from the endpoints after slice C's
expectations were frozen, so everything in this subsection is a
post-specified correction rather than a registered prediction.

**Rate control: the notification point is the endpoint, not the switch.**
The earlier design assumed the ordinary RoCEv2 arrangement, a switch RED mark
that the receiver turns into a CNP. On the measured fabric the switch never
marks: zero congestion-experienced packets in 670 M, at two DSCP classes,
with the egress buffer full and dropping 6.36 percent, every packet arriving
markable. Kmin, Kmax and Pmax are undefined there; the switch tail-drops.
What does generate notifications is the receiving NIC itself, from its own
ingress congestion: the receiver's `np_cnp_sent` rises from 38 to 2262 per
second exactly during reliable fan-in. So the model's notification point sits
at the endpoint's own ingress meter, and a fabric that never marks is the
normal case rather than a degenerate one. The measured dynamics are this
block's acceptance data:

| quantity | measured |
|---|---|
| CNP rate per congested QP | 283 per second, that is one per 3.54 ms |
| rate cut of at least 30 percent | after 3, 16 and 39 ms in three repeats |
| fair share reached | after 5 ms, 1.8 s and 2.3 s |
| recovery to at least 95 percent | 447 plus or minus 10 ms |
| additive increase | about 0.107 falling to 0.093 Gb/s per ms |
| steady two-flow split | 50.18 and 50.13 Gb/s on a 99.30 Gb/s wire |

The reaction is 2 to 20 times slower than the vendor-default reading in the
profile suggests and the recovery is about 4.5 times slower, so a study that
needs the transient must carry these numbers rather than the defaults.

**Ingress meter: the discard ledger does not close.** ANOM-03 is registered
as emergent from this block, and the slice-C study does reproduce the
equilibrium with one fitted drain rate. It does not explain it: the campaign's
own counters show far too few PHY discards at that operating point to account
for a 20 percent goodput deficit by loss alone, and P6 sharpened rather than
closed the gap. A lone flow above about 94 Gb/s loses 0.18 percent of its
packets at the receiver's PHY, and it loses them in bursts of about 73 packets
lasting about 94 us, path-independent over 32 fresh 5-tuples, with the event
count falling 12.5 times under at least 10 Gb/s of reverse traffic while the
burst length does not move. That signature is ANOM-17, it is not a full-buffer
overflow (`out_of_buffer` never moves in any run), and reproducing it is the
next mechanism this block needs. Until then the meter's drain rate is a fitted
stand-in for a stall process, and BACK-57 carries the clause.

## Anomaly table

Kinds: `emergent` (falls out of a modelled mechanism and is validated),
`injected` (applied by rule from the table because the mechanism is not
public), `fabric` (a property of the switch or link, reproduced by htsim, not
by the endpoint), `counter` (a facade behaviour, not a datapath effect),
`tool` (an artifact of the instrument that measured it rather than a property
of the silicon, kept for the record and reproduced by nothing).

ANOM-01 was re-attributed and ANOM-16 through ANOM-19 were added after the
P6 fabric campaign, which ran after slice C's expectations were frozen. All
five are post-specified corrections and say so.

| id | anomaly | trigger | effect and magnitude | kind | evidence |
|---|---|---|---|---|---|
| ANOM-01 | single UD QP receive cap | one UD QP receiving above 3.07 Mpps through the measurement engine | re-attributed after slice C froze: the 3.07 Mpps knee and the 47.5 percent silent loss were the engine's receive path, not the NIC. On the wire one UD QP absorbs 5.51 Mpps of 2 KiB with only the 0.17 to 0.19 percent ingress floor, and four QPs are slightly worse | tool (the measurement engine's receive path, not the silicon) | P3 seed 1, re-attributed by P6 |
| ANOM-02 | two-SGE SEND sequence-error storm | RC SEND with two gather entries at 512 B each, 32 QPs | packet_seq_err of 68 k to 400 k per 30 s, 1-SGE control zero, goodput within 3 percent | injected (per-packet drop rule keyed on SGE count and size) | P3 seed 5/6 |
| ANOM-03 | saturated deep-pipeline loss equilibrium | one RC QP, queue depth 1024, no inter-burst gap, above about 92 Gb/s | responder PHY discards plus go-back-N tail; goodput settles at 78 to 92 Gb/s | emergent (ingress meter) | P2, P3, P4 |
| ANOM-04 | drain window | inter-burst gap of at least 4 us at 8 KiB, 4 to 100 us at 64 KiB | discards go to zero; goodput follows bytes over (burst plus gap) within 0.1 percent; a 4 us gap raises 8 KiB goodput 13.8 percent | emergent (ingress meter) | P4 |
| ANOM-05 | in-NIC loopback starvation | loopback and wire ingress together above 197 Gb/s | wire keeps 99 percent, loopback drops to 51 percent, no PCIe stall | emergent (internal arbiter) | P3 seed 13 |
| ANOM-06 | ECT(0) stamping | any RoCEv2 transmit | ECN bits forced to ECT(0) regardless of requested ToS; DSCP honoured | injected (rate control stamp) | P5b |
| ANOM-07 | inert marking counter | any traffic | np_ecn_marked_roce_packets stays zero while CNPs are generated | counter | P3, P5a, P5b |
| ANOM-08 | one-hop pause | receiver overload | receiver emits global pause, no peer ever receives it | fabric | P3, audit |
| ANOM-09 | firmware counter variant | retransmission | fw 16.31 counts local_ack_timeout_err, fw 16.32 counts zero | counter | P5a |
| ANOM-10 | bidirectional cleaner than unidirectional | full duplex at 91.8 Gb/s per direction versus one direction at 93.4 | duplex counter-clean, unidirectional dirty | emergent (ingress meter) | P4 |
| ANOM-11 | incast amplification | N senders into one receiver, 1 MiB messages | wire full, goodput tax equals loss rate times a go-back-N amplification factor (16x at 1.65 percent loss over 1 MiB messages, a 26.9 percent tax) | emergent (packetizer plus go-back-N) with fabric loss from htsim | P5a |
| ANOM-12 | UD one-over-N delivery | N UD senders into one receiver | delivery exactly 1/N, no endpoint counter moves | fabric | P5a |
| ANOM-13 | MTU tax | MTU 1024 versus 4096 | 5.6 percent goodput | emergent (packetizer) | P2 |
| ANOM-14 | host-bound message rate | small messages, single process | 3.87 Mpps single QP at 1 KiB, 16.7 Mmsg/s per sender at 512 B | emergent (transmit pacer) | P2, P5a |
| ANOM-15 | memory-region and gather insensitivity | 12 000 MRs, 1024 MRs of 64 KiB, gathers except ANOM-02 | no throughput effect | emergent by absence (no MR cache modelled; documented) | P3 seeds 7/8 |
| ANOM-16 | NIC-generated congestion notification | RC fan-in on a fabric whose switch never marks | the receiver signals its own ingress congestion: np_cnp_sent rises from 38 to 2262 per second during the fan-in, which is 283 CNP per second per congested QP, or one per 3.54 ms | emergent (rate control, notification point at the endpoint) | P6 |
| ANOM-17 | ingress stall bursts | one lone RC flow above about 94 Gb/s | 0.18 percent of packets lost at the receiver's PHY in bursts of about 73 packets (about 94 us), path-independent over 32 fresh 5-tuples; at least 10 Gb/s of reverse traffic cuts the event count 12.5x without changing the burst length, and receiver CPU load does nothing | emergent (ingress meter, mechanism not yet modelled) | P6 |
| ANOM-18 | slow DCQCN dynamics | a DCQCN reaction point under 2 to 1 RC fan-in | rate cut of at least 30 percent after 3 to 39 ms, fair share after 5 ms to 2.3 s, recovery to at least 95 percent in 447 plus or minus 10 ms, additive increase about 0.1 Gb/s per ms | emergent (rate control, reaction point) | P6 |
| ANOM-19 | counter semantics under loss | any loss on the path | packet_seq_err counts loss bursts, 73x fewer than packets lost; rx_discards_phy counts NIC-ingress loss exactly; switch loss appears in neither; out_of_buffer never moves; senders handle 2.24x the CNPs the receiver reports sending, which is reproducible and unexplained | counter | P6 |

The table is data: a `constexpr` array in `rnic_anomaly_table.h`, a generated
Markdown projection checked by a test against this document, and one native
test per row that asserts the effect within its registered band.

## Validation

Golden vectors are the campaign CSVs: depth-1 and depth-1024 message-size
curves, the gap sweep, the multi-QP ceiling, the MTU pair, the bidirectional
pair, and the incast rows. Each check is registered in an expectations file
before the run and reports as one `summary.csv` row with a `within` band.
Fatal guards (deterministic replay identity, one-authority conservation,
counter monotonicity) void a run rather than costing a point.

| measured input | model output | band |
|---|---|---|
| depth-1 goodput versus size | B = S / (T_eff + S / C) at every size | 15 percent |
| depth ratio at 8 KiB and 64 KiB | goodput(depth 1024) over goodput(depth 1) | 20 percent on the ratio |
| gap sweep at 8 KiB and 64 KiB | discards reach zero across the threshold; goodput matches the duty model | categorical plus 15 percent |
| single QP saturated | equilibrium goodput | 78 to 92 Gb/s window |
| multi-QP at 4 KiB | aggregate reaches the goodput ceiling | 3 percent |
| MTU 1024 versus 4096 | goodput ratio | 2 percentage points around 5.6 |
| 2 to 1 incast at 1 MiB | tax equals loss times the amplification factor; fair share | 25 percent on the identity, 2 points on the split |
| notification rate under fan-in | `np_cnp_sent` per second per congested QP on a fabric that never marks | 30 percent around 283 |
| notification rate below saturation | `np_cnp_sent` for a lone paced flow | exactly zero |
| DCQCN transient | cut, fair-share and recovery times, and the additive slope | the measured ranges; 25 percent on the slope |
| 1 to 2 fan-out | delivered fraction, split, drops | 3 points, 0.5 points, exactly zero |
| single UD QP above the cap | delivered rate and discard counter | 10 percent |
| loopback plus wire | wire share and loopback share | 5 percent |
| fabric under the endpoint | the merged [`FabricProfile`](../modules/backends.md) carrier and its measured leaf: one non-blocking switch, four 100 G ports, 515 ns pipes, a 5.2 MB tail-drop egress buffer per port, and no marking, no PFC and no propagated pause. A study that needs the endpoint under load configures the drop-only switch and lets the endpoint generate its own notifications | topology and buffer as declared; loss rate within the study's own band |
| deterministic replay | trace equality across two runs | exact |

## Status

The queue core, PCIe fabric, host memory and submission models exist and are
validated. The profile, the anomaly table, the C facade and the packetizer,
outstanding-work window and pacer are landed and validated, and so are the
ingress meter, the receive processor and the go-back-N requester transport.
Rate control is landed behind an opt-in whose off path is the unchanged
slice-C code: the notification point at the receiver's own ingress meter, the
per-queue-pair reaction point in front of the transmit pacer, and the tail-drop
egress queue the test fabric needs before either can be exercised. The internal
arbiter is not landed and has moved to BACK-57, beside the ingress meter whose
service budget it shares. The remaining blocks and the
clauses the receive and congestion-control slices did not close are registered
as open tasks in `docs/modules/backends.md`; the ConnectX-5 profile study in
`examples/cx5_msgsize_v1` establishes the measured curves the blocks are
validated against.

Rate control closes the incast collapse the receive slice left open, and it
opens one clause of its own. Because the notification point is the receiver's
own ingress meter, the control loop settles wherever that meter's drain rate
is. Slice C fitted that drain from the lone-flow anchors, and at the fitted
value the loop settles below the switch's egress rate, so the switch buffer
never fills, there is no loss to amplify and the incast tax comes out near zero
instead of at the measured 26.9 percent. The incast measurement is a second
anchor for the same latent and it wants a higher value: the campaign's receiver
absorbed 99.39 Gb/s under fan-in while discarding almost nothing at its PHY. One
drain rate cannot satisfy both anchors, which is the same shape as the
bidirectional incompatibility already registered against this meter, and it
points at the same missing stall mechanism (ANOM-17).

Two rows of the anomaly table carry a correction the receive slice earned,
and both remain open. ANOM-03 attributes the saturated loss equilibrium to
the ingress meter alone; the campaign's own P4 counters show only a few
thousand PHY discards over a 30 s window at that point, which is far too few
to explain a 20 percent goodput deficit by loss, so the meter reproduces the
equilibrium at a fitted drain rate rather than explaining it. P6 narrowed the
missing part to the stall-burst signature now registered as ANOM-17. ANOM-11
is listed as emergent from the packetizer plus go-back-N with fabric loss;
the slice-C study shows that without the reaction point the same mechanism
collapses rather than settling at the measured tax, so the row depends on
rate control as well. Neither of those two attributions is rewritten in the
table: the table is data with a generated projection and a byte-comparison
test, and an attribution is edited when the task that owns it lands, not when
a study reads it.

The P6 fabric campaign ran after slice C's expectations were frozen and
corrected five things in this document, one parameter and four table rows.
The unreliable per-QP receive ceiling moved from 3.07e6 to the 5.51e6 measured
on the wire, because the earlier
number was the measurement engine's receive path (ANOM-01, re-attributed and
kept). Congestion notification is generated by the receiving NIC and not by a
switch mark (ANOM-16), the DCQCN transient is far slower than the vendor
defaults imply (ANOM-18), the lone-flow ingress floor has a burst structure
no block reproduces yet (ANOM-17), and three counters mean something other
than their names suggest (ANOM-19). Every one of those is post-specified: it
is labelled as such here, in the anomaly table, in the registry entries and
in the slice-C result record, and none of them was allowed to edit a frozen
expectation or restate a frozen verdict.
