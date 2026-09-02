# RNIC golden model anomaly table

This file is generated from the `constexpr` table in
`simllm/backends/rnic/include/simllm/rnic/rnic_anomaly_table.h` and a
native test compares it byte for byte. Edit the table, not this file.
It is the projection of the anomaly table in
[the golden-model design](rnic-cmodel.md), which carries the same
rows and states how each one is reproduced.

Kinds: `emergent` falls out of a modelled mechanism and is validated,
`injected` is applied by rule because the mechanism is not public,
`fabric` is a property of the switch or link reproduced by the packet
simulator rather than by the endpoint, `counter` is a facade
behaviour with no datapath effect, and `tool` is an artifact of the
instrument that measured it rather than a property of the silicon,
kept for the record and reproduced by nothing.

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

Every row is registered against a model block or is explicitly the
fabric's. A row whose kind is `emergent` must be reproduced by the
named mechanism inside its registered band before that block can be
called validated; a row whose kind is `injected` is reproduced by
rule and is honest about it.
