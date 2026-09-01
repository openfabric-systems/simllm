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
simulator rather than by the endpoint, and `counter` is a facade
behaviour with no datapath effect.

| id | anomaly | trigger | effect and magnitude | kind | evidence |
|---|---|---|---|---|---|
| ANOM-01 | single UD QP receive cap | one UD QP receiving above 3.07 Mpps | excess dropped at the PHY, no sender-visible signal, 47.5 percent at 5.85 Mpps offered | emergent (receive processor) | P3 seed 1 |
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

Every row is registered against a model block or is explicitly the
fabric's. A row whose kind is `emergent` must be reproduced by the
named mechanism inside its registered band before that block can be
called validated; a row whose kind is `injected` is reproduced by
rule and is honest about it.
