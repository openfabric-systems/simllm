# Unique-NIC mapping expectations

Date: 2026-09-13

This is the expectations-only freeze for PLACE-2, the `unique-nic` GOAL-rank
mapping, together with the mapper's fabric-backed NIC selection that PLACE-1
still owes. It precedes the mapper change, the renderer and sink changes,
the shared-NIC declared builder, their tests, the study harness and every
result-producing run.

## Question

Can the mapper assign one GOAL rank per network interface controller (NIC)
from the fabric manifest's GPU-to-NIC affinity, so that several GPUs behind
one NIC share one fabric endpoint and intra-node traffic stays off the
fabric, with the default `gpu-rank` path byte identical to today, and with
the fluid closed form telling exactly how much a shared endpoint costs?

## Frozen source and compatibility identity

The implementation starts from commit
`26704c1c18e3c5503cb717cdd31391b82d3983a0`. The JSON registry records the
pre-change SHA-256 identities of the placement package entry, the manifest,
mapper and peer topology modules, the traffic step renderer and locality
classifier, and the step sink.

The compatibility authority is the UTF-8 JSON of the manifest writers plus
the m5 check-B makespans. Under the default mapping nothing may move:

| Record | Builder call | Bytes | SHA-256 |
|---|---|---:|---|
| worked example | `declared_manifest(tp=4, pp=2, dp=2)` | 10,832 | `2e46eadcaccd83de1778ea98c368585b452b83067489b9a2df064e25a14d857e` |
| m4 tensor group | `declared_manifest(tp=8)` | 5,698 | `3812aef241d93f2af8d86da0f6bbb606dae9c43c768ceb3efb8d649dc3bacfd6` |
| rail pipeline | `declared_pipeline_placement(8)` | 52,310 | `8d38cf4b6990bfd75fc90dcfe994180c3c1150adce297b6f71983fd8c9c877db` |
| m5 expert world | `declared_manifest(tp=1, dp=8)` | 5,698 | `0894fae1687217d88466b5692034a8d1863fb8c658afdfde2c317d1c412f60a5` |
| width tail | `declared_manifest(tp=64, nodes=8, gpus_per_node=8)` | 102,050 | `bed8d26c72feaf117f48b651ec18b4fb0caf39df2e47866a550f74252972aecd` |
| one-plus-one fabric | `disaggregated_manifests(prefill_nodes=1, decode_nodes=1).fabric` | 21,379 | `36527abf437eb875ce8db097b301bd1c25789f98fd38271378d65f6416361367` |
| one-plus-one fabric, rendering off | same with `render_physical_topology=False` | 8,204 | `f3950a9593817adbba344fa1d91a752b5d44a0d066b0f29163a5add448b84841` |

and the six m5 check-B makespans on `rnic-nn-fluid` (`decode8x2048` at
`W` 2, 4, 8: 563,362,560; 486,963,888; 448,764,528 ps; `prefill2048`:
17,592,951,360; 25,646,015,088; 29,672,546,928 ps) reproduce exactly through
`HtsimStepSink` with the default mapping. These are fatal, unscored guards.

## Declared interface

```text
RankMapper(placement, mode="gpu-rank" | "unique-nic", fabric=None | FabricTopologyManifest)
    .goal_rank(rank)       # unique-nic: the index of rank's NIC in fabric order
    .num_goal_ranks()      # unique-nic: the fabric's NIC count
    .nic_of(rank)          # the affine nic_id, from the fabric; requires a fabric in either mode
    .is_intra_node(a, b)   # unchanged, placement authority
declared_shared_nic_fabric(placement, *, gpus_per_nic) -> FabricTopologyManifest
HtsimStepSinkConfig(..., goal_rank_mapping="gpu-rank" | "unique-nic", fabric_manifest=None | FabricTopologyManifest)
```

- Fabric order is the order of `fabric.nodes` and, inside a node, the order
  of `node.nics`. `unique-nic` requires a fabric whose GPU set equals the
  placement's rank set, whose every GPU names one NIC on its own node, and
  whose `goal_rank_mapping` field is `"unique-nic"`;
  `FabricTopologyManifest.validate()` accepts both spellings and refuses
  anything else. With one NIC per GPU in GPU order the mapping is the
  identity, and every artifact is byte identical to `gpu-rank`.
- `declared_shared_nic_fabric` gives every node of a declared placement
  `gpus_per_node // gpus_per_nic` NICs, `nic_id` `"<host>:nic-<j>"`, with GPU
  `local_rank` `l` affine to NIC `l // gpus_per_nic`, no physical rendering.
  `gpus_per_nic` must divide the node width; one NIC per GPU reproduces the
  identity mapping.
- Renderer rule for collapsed pairs. Segments of one phase share one tag
  today and are distinct only by their semantic `(source, destination)`.
  When two or more segments of a phase land on the same GOAL endpoint pair,
  the renderer gives them distinct tags by a fixed rule that is the identity
  when no pair collapses: with `M` the largest number of segments sharing
  one endpoint pair in the step, every rendered tag is `tag * M + j` where
  `j` is the segment's index among its endpoint pair's segments ordered by
  `(source_rank, destination_rank)`. `M = 1` gives `tag * 1 + 0 = tag`, which
  keeps `gpu-rank` rendering byte identical. The step plan carries a
  projection table from `(goal source, goal destination, rendered tag)` to
  the semantic segment; every backend completion row joins to exactly one
  segment and a duplicate key is fatal.
- Sink rule. `goal_rank_mapping="unique-nic"` requires `fabric_manifest`,
  runs on the null-network profiles, and is refused with a diagnostic naming
  the seam when combined with `peer_packet`, `flow_session`,
  `dependency_cross_check`, a physical `topology` file or `num_goal_ranks`
  below the NIC count. Those combinations are registered, not modeled. The
  coarse device runtime keeps its fixed eight-RNIC profile and is untouched.
- The locality split is unchanged: local segments never reach the fabric,
  so two GPUs behind one NIC exchanging data stay on NVLink exactly as under
  `gpu-rank`.

## Frozen cells

Cell U1, identity: the one-plus-one disaggregated deployment (16 GPUs, 16
NICs, NIC order equal to GPU order), with `ep_ranks` the 16 ranks, dims the
m5 granite geometry at 16 experts per rank width (32 experts), one prefill
and one decode m5 step, on `rnic-nn-fluid` at 400 Gbit/s. `unique-nic` and
`gpu-rank` produce byte-identical GOAL text, equal `StepResult` rows and
equal outcomes; `num_goal_ranks()` is 16 and `goal_rank(r) == r`.

Cell U2, permutation: a declared placement of two nodes of four GPUs
(`declared_manifest(tp=1, dp=8, gpus_per_node=4)`) with a fabric whose NICs
are ordered in reverse of the GPU order on each node (NIC `j` affine to
local rank `3 - j`), one NIC per GPU. `goal_rank` is `4 * node + (3 - local)`,
every fabric segment's endpoints are permuted, and because the fluid null
network is endpoint symmetric, every `StepResult` row and every joined
per-segment completion time equals the `gpu-rank` run exactly, while the
GOAL text differs only in rank numbers. Every completion row joins back to
its semantic pair through the projection table.

Cell U3, shared endpoints: two nodes of eight GPUs (`declared_manifest(tp=1,
dp=16)`) with `declared_shared_nic_fabric(placement, gpus_per_nic=g)` for
`g` in 1, 2, 4, and a pairwise all-to-allv of `S` bytes on every remote
ordered GPU pair for `S` in 65,536 and 1,048,576, rendered through the
sink's expert-parallel path on `rnic-nn-fluid` at 400 Gbit/s. Under
`gpu-rank` each fabric endpoint carries 8 concurrent flows of `S` bytes;
under `unique-nic` with `g` GPUs per NIC each endpoint carries `8 g` flows
and the GOAL declares `16 / g` ranks. The fluid closed form gives every
flow's completion time as `flows_per_endpoint * S * 20 ps/byte +
2,000,000 ps`, so the frozen relations are, for every `S`:

| g | GOAL ranks | flows per endpoint | per-flow FCT minus 2,000,000 ps |
|---:|---:|---:|---|
| 1 | 16 | 8 | `8 * S * 20 ps` |
| 2 | 8 | 16 | `16 * S * 20 ps`, exactly twice the `g = 1` value |
| 4 | 4 | 32 | `32 * S * 20 ps`, exactly four times the `g = 1` value |

and the phase makespan obeys the same ratios. The `g = 1` row must equal the
`gpu-rank` run byte for byte. Fabric bytes are identical across `g` (the
collapse moves no byte), and NVLink bytes are identical across `g` and across
modes (the locality split is untouched). Every rendered tag is unique per
endpoint pair and the projection table joins every completion row exactly
once.

Cell U4, refusals: `unique-nic` without a fabric; a fabric missing a NIC for
one GPU; a fabric whose GPU set differs from the placement; `gpus_per_nic`
that does not divide the node width; `unique-nic` with `peer_packet`, with a
physical `topology` file, with `flow_session`, and with
`dependency_cross_check`; `num_goal_ranks` below the NIC count; and a
projection table with a duplicate key. Each is refused before any workdir,
GOAL artifact or backend process exists.

Cell U5, wire identity: the shared-NIC fabric round-trips through `save` and
`load`, `goal_rank_mapping` survives the round trip in both spellings, and
the seven digests above plus the six m5 makespans hold under the default.

## Physical sanity before observation

A flow of `S` bytes cannot complete faster than its own serialization,
`S * 20 ps` at 400 Gbit/s, plus one 2,000,000 ps propagation. Sharing an
endpoint among `n` equal flows that start together cannot finish any of
them before `n * S * 20 ps + 2,000,000 ps` under fair sharing, which is the
fluid rule the m1 scatter cells validated to 0 ps. The U3 table is that rule
applied to `8 g` flows; a ratio other than exactly `g` between the `g` and
`g = 1` rows is a defect in the collapse, the tags or the join, never a
network finding.

## Evidence accounting and closure

U3's twelve per-flow relation instances (two payloads, three `g`, FCT and
makespan) are the scored behavioral family, and U1's and U2's identities are
scored exact-oracle families. U5 and the projection-table joins are
structural exact guards, U4 is a rejection control family, and the seven
digests plus six makespans are fatal by-construction identities. Counts from
these classes are never added; a violated fatal guard voids the run.

PLACE-2 closes if every cell holds. PLACE-1's mapper clause ("general NIC
selection in the mapper") becomes literal for the declared shared-NIC shape
and its registry text narrows accordingly; the combinations the sink
refuses are registered as a new task. `docs/architecture.md`'s `unique-nic`
sentence becomes literal and gains its citation, and the placement doc's
mapper bullet states the fabric-backed selection.
