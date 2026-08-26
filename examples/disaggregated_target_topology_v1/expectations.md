# Disaggregated target topology expectations

Date: 2026-08-26

This is the expectations-only freeze for PLACE-5. It precedes the fixed
physical topology implementation, its study harness, generated placement,
fabric or Group Operation Assembly Language (GOAL) artifacts, and every
result-producing run. The fixed topology is a declared target, not a measured
cluster inventory or a calibrated transport.

## Question

Can the accepted disaggregated placement builder express the complete
448-rank target as one physical `simllm-fabric-topology-v1` graph, carry every
GPU-affine network interface controller (NIC) through a switch port with an
explicit link rate and propagation delay, and prove that the same endpoints
are reachable from rendered GOAL messages without changing a placement record
when physical rendering is disabled?

## Frozen source and compatibility identity

The implementation starts from commit `4f7a316926ecd55fb00d376e5aae1bcfc01c1929`.
The JSON registry records the pre-change SHA-256 identities of the PLACE-4
builder, the fabric manifest module, its focused test, and the reference
64-endpoint topology file.

The compatibility authority is the UTF-8 JSON emitted by
`PlacementManifest.save`, including its terminal line feed. Before PLACE-5,
the one-prefill plus one-decode record is exactly 15,772 bytes with SHA-256
`019f818e02252407e560b37415da12151c8f6ca0ff01bcb7ff8aabfead47f286`.
The 16-prefill plus 40-decode record is exactly 2,639,042 bytes with SHA-256
`48029d871293762007ab33082d59a7b5a4efb22583394e718c97e733717fd709`.

The new builder has an explicit physical-rendering switch. With that switch
disabled, both placement records must retain those byte lengths and digests.
Enabled and disabled construction must also emit byte-identical placement
records to each other. This identity is a fatal, unscored guard.

## Declared fabric

The graph extends the repository's `examples/m1/topologies/clos_64_400g.topo`
conventions without claiming that a real 448-rank installation was measured:

- two switch tiers, leaf then spine;
- eight GPU-affine NIC endpoints per leaf switch;
- eight spine uplinks per leaf, one to every spine;
- 400,000,000,000 bit/s on endpoint and leaf-to-spine links;
- 1,000,000 ps propagation on every link;
- zero switch-latency term in the manifest;
- deterministic dense switch, port and link identities.

For `R` ranks, the graph has `ceil(R / 8)` leaves, eight spines, `R` endpoint
links and eight leaf-to-spine links per leaf. The one-plus-one cell therefore
has two leaves, eight spines, 32 links and 48 switch ports. The target cell has
56 leaves, eight spines, 896 links and 1,344 switch ports. Every NIC has one
and only one endpoint link to the leaf selected by `global_rank // 8`. Every
leaf has one and only one link to every spine. Link endpoints, switch ports,
link identities, GPU identities and NIC identities are unique.

The graph is undirected for reachability. A deterministic path chooses the
lowest numbered spine for a cross-leaf pair. Two NICs on one leaf have a
two-link path and 2,000,000 ps declared propagation. NICs on different leaves
have a four-link path and 4,000,000 ps declared propagation. Every positive
link rate and delay comes from the one graph authority, not from a second GOAL
or study-side table.

## Frozen conservation cells

The two cells come from the same builder and differ only in pool scale:

| Cell | Prefill nodes | Decode nodes | Ranks | GPUs | NICs | Prefill ranks | Decode ranks |
|---|---:|---:|---:|---:|---:|---:|---:|
| one-plus-one | 1 | 1 | 16 | 16 | 16 | 8 | 8 |
| target | 16 | 40 | 448 | 448 | 448 | 128 | 320 |

For each cell, the rank set, GPU rank set, NIC affinity rank set and GOAL rank
set must all be the exact dense interval from zero through `R - 1`. Every
logical node has eight GPUs and eight NICs, and its manifest role agrees with
all eight ranks. Loss, duplication, a foreign identity or a role disagreement
is fatal and voids the run.

## GOAL reachability witness

Each cell renders one explicit 4,096-byte message per rank. Rank `r` sends to
`(r + 8) mod R`. This permutation makes every rank appear exactly once as a
source and once as a destination, crosses a leaf boundary for every message,
and produces 16 messages for one-plus-one and 448 for the target. The rendered
GOAL declares exactly `R` ranks.

Every structured GOAL message must resolve through the placement rank, its
GPU, its affine NIC, the NIC endpoint link, one leaf-to-spine link, one
spine-to-leaf link and the destination endpoint link. The path must have four
distinct links, 4,000,000 ps aggregate propagation and 400,000,000,000 bit/s
as its minimum rate. The source and destination sets, payload bytes, message
count and resolved endpoint sets are exact structural oracles.

This witness validates that the topology authority is usable by GOAL
rendering. It does not claim backend packet timing. TRAF-64 owns the target
key-value handoff through GOAL, packet service, last-arrival completion and
the live time-to-first-token (TTFT) and time-per-output-token (TPOT) metric
chain.

## Physical sanity before graph observations

A 4,096-byte message cannot serialize onto a 400 Gbit/s link in less than
81,920 ps. A cross-leaf path cannot propagate in less than four times
1,000,000 ps. Before reading the graph observation, the no-queue physical
interval for one witness message is therefore:

```text
cut-through floor = 4,000,000 + 81,920 = 4,081,920 ps
store-and-forward ceiling = 4,000,000 + 4 * 81,920 = 4,327,680 ps
```

The study reports the graph's propagation and bottleneck rate beside this
interval. It does not report a modeled completion timestamp because no packet
runtime owns these messages in PLACE-5. A future measured or simulated value
outside this interval under the same no-queue assumptions would prove a
defect, while landing inside would not prove calibration.

Scaling from 16 to 448 ranks must multiply endpoint links by 28, leaf switches
by 28, GOAL messages by 28 and leaf-to-spine links by 28. It must leave the
per-message path length, propagation, bottleneck rate and placement bytes per
rank mechanism unchanged. The independent axes are deployment scale,
cross-leaf path selection and the enabled versus disabled physical-rendering
mode.

## Evidence accounting and closure

This study has no scored behavioral denominator. Topology cardinality,
identity, connectivity, path arithmetic, GOAL coverage, serialization bounds,
round-trip wire identity and disabled-path placement identity are fatal,
unscored exact or structural guards. A violated guard makes the result void
and PLACE-5 remains open.

PLACE-5 closes only if both cells conserve every rank, GPU, NIC and role; the
target graph is complete and round-trips through
`simllm-fabric-topology-v1`; every NIC and every rendered GOAL endpoint is
reachable; all declared rates and delays match this freeze; and the disabled
placement digests remain exact. If closure is earned, PLACE-4's remaining
physical-location clause becomes literal and TRAF-62's PLACE-5 dependency
note is updated in the same documentation change.

The result does not close TRAF-62 or TRAF-64, execute the target packet
handoff, calibrate a fabric, discover live inventory, implement general
`unique-nic` mapping, change a README, or modify either backend repository.
PLACE-1 and PLACE-2 retain the general inventory and mapping work.
