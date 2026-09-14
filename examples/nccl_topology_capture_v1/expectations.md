# NCCL topology capture expectations

Date: 2026-09-13

This is the expectations-only freeze for the second PLACE-1 slice: sourcing
one node's intra-node structure from an NCCL topology dump. It precedes the
reader, the captured-node fabric join, their tests, the study harness and
every result-producing run. The dump and the inventory it is checked against
were captured before this freeze and are tracked under
`tests/fixtures/nccl_topology/perlmutter_a100_nid001056/` with their digests
in that directory's `PROVENANCE.md`; they are input evidence, not results.

## Question

Can the placement module read the intra-node structure of a real GPU node
from NCCL's own topology dump (`NCCL_TOPO_DUMP_FILE`), join it to the
repository's fabric and peer-topology schemas, and drive the live peer packet
path with it, so that a captured node substitutes exactly for a hand-declared
twin, without changing one byte of any existing manifest?

## Frozen source and compatibility identity

The implementation starts from commit
`26704c1c18e3c5503cb717cdd31391b82d3983a0`. The JSON registry records the
pre-change SHA-256 identities of the placement package entry, the manifest,
mapper, peer topology and DGX modules, the traffic step renderer and locality
classifier, the step sink and the peer step module, and the five fixture files.

The compatibility authority is the UTF-8 JSON emitted by the manifest writers
and the canonical JSON of a peer fabric. Eight reference artifacts must keep
their pre-change byte lengths and digests:

| Record | Builder call | Bytes | SHA-256 |
|---|---|---:|---|
| worked example | `declared_manifest(tp=4, pp=2, dp=2)` | 10,832 | `2e46eadcaccd83de1778ea98c368585b452b83067489b9a2df064e25a14d857e` |
| m4 tensor group | `declared_manifest(tp=8)` | 5,698 | `3812aef241d93f2af8d86da0f6bbb606dae9c43c768ceb3efb8d649dc3bacfd6` |
| rail pipeline | `declared_pipeline_placement(8)` | 52,310 | `8d38cf4b6990bfd75fc90dcfe994180c3c1150adce297b6f71983fd8c9c877db` |
| m5 expert world | `declared_manifest(tp=1, dp=8)` | 5,698 | `0894fae1687217d88466b5692034a8d1863fb8c658afdfde2c317d1c412f60a5` |
| width tail | `declared_manifest(tp=64, nodes=8, gpus_per_node=8)` | 102,050 | `bed8d26c72feaf117f48b651ec18b4fb0caf39df2e47866a550f74252972aecd` |
| one-plus-one fabric | `disaggregated_manifests(prefill_nodes=1, decode_nodes=1).fabric` | 21,379 | `36527abf437eb875ce8db097b301bd1c25789f98fd38271378d65f6416361367` |
| one-plus-one fabric, rendering off | same with `render_physical_topology=False` | 8,204 | `f3950a9593817adbba344fa1d91a752b5d44a0d066b0f29163a5add448b84841` |
| DGX A100 peer fabric | `dgx_peer_fabric("a100", node_id="node-0", ranks=(0..7), propagation_delay_ps=1000, switch_input_buffer_bytes=65536)`, `asdict` as sorted compact JSON | 84,797 | `47d66b66acd5a477f757ee30400256d39569df5a5fef5aea1655a937fc9fa219` |

These identities are fatal, unscored guards. The slice adds a reader and a
join; it adds no field to any serialized schema, so nothing built without the
reader may change.

## Captured node, stated before the reader exists

The fixture is one NERSC Perlmutter GPU node, `nid001056`, captured on
2026-09-13 in Slurm job 58271200 with NCCL 2.29.7 under PyTorch 2.13.0. The
facts below are read from the fixture by hand and are what the reader must
reproduce:

- Four `<cpu>` elements, one per NUMA node, in file order `numaid` 3, 0, 1,
  2, all with the same `host_hash`, `arch="x86_64"`, `vendor="AuthenticAMD"`.
- Eight `<pci>` elements, all with `link_speed="16.0 GT/s PCIe"` and
  `link_width="16"`: four GPUs (`vendor="0x10de"`, `device="0x20b0"`, class
  `0x030200`) and four NICs (`vendor="0x17db"`, `device="0x0501"`, class
  `0x020000`).
- GPUs `dev` 0, 1, 2, 3 at bus ids `0000:03:00.0`, `0000:41:00.0`,
  `0000:82:00.0`, `0000:c1:00.0` under NUMA nodes 3, 2, 1, 0, each with
  `sm="80"`, `rank` equal to `dev`, `gdr="1"`, and exactly three `<nvlink>`
  children with `count="4"` and `tclass="0x030200"`, one per other GPU.
- NICs `cxi3`, `cxi0`, `cxi1`, `cxi2` (file order) at bus ids `0000:01:00.0`,
  `0000:c2:00.0`, `0000:81:00.0`, `0000:42:00.0` under NUMA nodes 3, 0, 1, 2,
  each with `speed="200000"` (megabit per second, so 200 Gbit/s), `port="1"`,
  `gdr="1"`, `maxconn="128"`.
- The GPU-to-NIC affinity by shared NUMA node is therefore
  `{0: cxi3, 1: cxi2, 2: cxi1, 3: cxi0}`: every GPU has exactly one NIC on
  its own NUMA node, and NIC order is the reverse of GPU order.
- `nvidia-smi topo -m` in the inventory shows `NV4` for all twelve ordered
  pairs and NUMA affinity 3, 2, 1, 0 for GPUs 0 through 3; `nvidia-smi
  nvlink -s` shows twelve links at 25 GB/s per GPU, so 3 pairs times 4 links
  times 25 GB/s equals the 300 GB/s per-GPU egress the packet-device model
  states for NVLink3.

## Declared interface

```text
simllm.placement.nccl_topology
    NcclTopologyDump.load(path) -> NcclTopologyDump      # strict parse of <system version="1">
    captured_fabric_node(dump, *, node_id, pool_role, global_rank_by_gpu_dev,
                         nvlink_link_rate_bps=200_000_000_000,
                         nvlink_propagation_delay_ps, ...)
        -> (FabricNodePlacement, PeerFabric)
```

The parse keeps, per `<cpu>`, its `numaid`, `host_hash`, `affinity`, `arch`
and `vendor`; per `<pci>`, its `busid`, `class`, `vendor`, `device`,
`link_speed` and `link_width`; per `<gpu>`, `dev`, `sm`, `rank`, `gdr` and
its `<nvlink>` rows (`target`, `count`, `tclass`); per `<nic>`/`<net>`,
`name`, `dev`, `speed`, `port`, `guid`, `maxconn`, `gdr`. Required attributes
must be present, `version` must be `"1"`, and an element the schema does not
name is refused. Reading is fail-closed: nothing is guessed from a missing
attribute.

The join produces, through the existing schemas only:

- `FabricNodePlacement(node_id, pool_role, gpus, nics)` with one
  `GpuFabricPlacement` per GPU in ascending `dev` order (`global_rank` from the
  caller's map, `gpu_id` the PCI bus id, `node_id`, `pcie_location`
  `numa-<numaid>/pci-<busid>`, `nic_id` `<node_id>:<net name>`), and one
  `NicFabricPlacement` per NIC in ascending net-name order (`nic_id`,
  `node_id`, `fabric_location` `numa-<numaid>/pci-<busid>`,
  `affine_gpu_rank`, no switch, port or link binding because no physical
  fabric is rendered).
- Affinity rule: a GPU's affine NIC is the one NIC under the same `<cpu>`
  element. Zero or more than one NIC under that element is refused, and the
  refusal names the GPU; the general rule for those shapes is registered, not
  modeled.
- `PeerFabric(domain_id="<node_id>:nv4", node_id, protocol="nvlink")` as a
  direct mesh: for each unordered GPU pair `(a, b)` with `count` links, `count`
  physical links `"<domain>:link-<a>-<b>:lane-<k>"` between GPU ports
  `"<domain>:gpu-<a>:to-<b>:lane-<k>"` and `"<domain>:gpu-<b>:to-<a>:lane-<k>"`
  at the declared per-link rate and propagation delay, and one `PeerRoute`
  per ordered pair whose paths are the `count` single-link paths in lane
  order. The `<nvlink>` count must agree in both directions and every GPU
  pair must have at least one link; a partial mesh is refused and registered.
  The mesh's `evidence_class` stays `"declared"`, because its timing inputs
  are declared; the wiring provenance is the fixture and the fabric manifest's
  `source="extracted"`.

## Frozen cells

Cell N1, parse: the fixture parses to 4 CPUs, 8 PCI devices, 4 GPUs, 4 NICs,
12 NVLink rows, with every literal in the captured-node section above exact.

Cell N2, join: with `node_id="nid001056"`, `pool_role="serving"` and the
identity rank map, the node carries GPUs with `global_rank` 0 through 3 at
the four bus ids, `nic_id` `nid001056:cxi3`, `nid001056:cxi2`,
`nid001056:cxi1`, `nid001056:cxi0` respectively, and NICs in order `cxi0`,
`cxi1`, `cxi2`, `cxi3` with `affine_gpu_rank` 3, 2, 1, 0.

Cell N3, mesh: 24 physical links, 48 GPU ports, 12 routes, 48 single-link
paths, every link at 200,000,000,000 bit/s and the declared propagation delay;
`paths_between(a, b)` returns four `ResolvedPeerPath` rows with distinct input
resources for every ordered pair, and `FabricTopologyManifest(nodes=[node],
peer_fabrics=(mesh,), source="extracted").validate()` passes.

Cell N4, inventory cross-check: the pair-link count the reader derives (4)
equals the `NV4` matrix parsed from the inventory's `nvidia-smi topo -m`
block for all twelve ordered pairs; the reader's NUMA per GPU equals the
matrix's NUMA affinity column; the dump's vendor and device ids equal the
`lspci` lines for the four GA100 and four Cassini devices; and twelve links
at 25 GB/s per GPU in `nvidia-smi nvlink -s` equal 3 pairs times 4 links.

Cell N5, live substitution: `declared_manifest(tp=4, gpus_per_node=4,
hostname_pattern="nid001056")` plus the captured fabric drive `HtsimStepSink`
on `rnic-nn-fluid` with `tp_ranks=(0, 1, 2, 3)`, a dense tensor-parallel
step (one prefill step and two decode steps from an existing dense step-record
helper in the repository), a fixed 37,000 ps compute provider, the placement
manifest, and `peer_packet=PeerPacketConfig(fabric, ((domain, profile),))`
with the pass-through direct-mesh profile the live peer packet study uses
(25 GB/s per link). The same run with a hand-built twin fabric written from
the literals of cells N2 and N3, never from the reader, must produce equal
`StepResult` rows, equal outcomes and equal fabric `to_dict()` documents. In
both runs every directed segment is local: fabric bytes are exactly zero, no
fabric backend is invoked, and NVLink bytes are positive. Two frozen
relations then hold on the captured run: the packet phase service of every
phase is at least its peak endpoint bytes divided by 100 GB/s (four links at
25 GB/s), converted to picoseconds and rounded up, and doubling the declared
per-link rate to 50 GB/s does not increase any phase service or the step
latency.

Cell N6, refusals: `version="2"`; a `<gpu>` without `rank`; an NVLink count
of 2 in one direction against 4 in the other; a GPU whose `<cpu>` holds two
NICs; a GPU whose `<cpu>` holds no NIC; a duplicate `busid`; an `<nvlink
target>` naming an unknown bus id; a `<net>` without `speed`; and a non-integer
`link_width` are each refused before any schema object is built.

Cell N7, wire identity: the captured fabric manifest round-trips through
`save` and `load` to an equal object and `to_dict()` carries the peer
inventory.

## Physical sanity before observation

One decode step's tensor-parallel exchange on this mesh cannot serialize a
rank's egress faster than 100 GB/s per destination pair and 300 GB/s per
GPU. The N5 floor states the pair bound; the study reports where the measured
service sits above it. The fluid null-network profile carries no fabric
bytes here, so no fabric closed form applies. A captured run that lands on a
different `StepResult` than its literal twin is a defect in the reader or the
join, never a calibration finding.

## Evidence accounting and closure

N5's twin identity and its two relations are the only scored rows (one exact
identity family and two behavioral relation instances). N1 through N4 and N7
are structural exact guards, N6 is a rejection control family, and the eight
digests are fatal by-construction identities. Counts from these classes are
never added. A violated fatal guard voids the run and PLACE-1 stays as it is.

This slice does not close PLACE-1. If every cell is exact, PLACE-1's registry
text narrows to what remains: general NIC selection when GPUs share a NIC or
a NUMA node holds several NICs, rendering the captured NIC inventory into a
physical fabric with its 200 Gbit/s rates, partial meshes, and the mapper's
`unique-nic` projection (PLACE-2, frozen separately). The residual
affinity shapes are registered as a new task. `docs/architecture.md`'s
NCCL-dump sentence gains its owning-task citation in the same change, and the
stale `select_nic` mention in the mapper docstring is corrected.
