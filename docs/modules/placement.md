# simllm.placement

The mapper: where logical ranks physically live. Serving frameworks are
topology-light (they know ranks and groups, not the fabric), so SimLLM joins
two independent descriptions and resolves every communication event through
both.

## Interface

- Placement manifest (`simllm-placement-manifest-v1`): per global rank the
  host, local rank, GPU UUID / PCI bus ID, group memberships with the actual
  global-rank lists, pipeline layer range, per-MoE-layer local expert IDs
  and the EPLB `placement_epoch`. `PlacementManifest` loads/saves JSON and
  answers `by_rank` / `group_ranks`. Manifests are either declared (what-if)
  or extracted from a live run; both share one schema.
- Fabric topology manifest (`simllm-fabric-topology-v1`): GPU to PCIe/NVLink
  to NIC to switch to link graph. Schema name pinned in `manifest.py`;
  contents land with M4.
- `RankMapper`: rank to GOAL-rank assignment mirroring the htsim drivers'
  `-goal_rank_mapping` (`gpu-rank` implemented; `unique-nic` needs the
  fabric manifest), plus `is_intra_node`.

## Status

Manifest round-trip and the gpu-rank mapper are implemented and tested
(including the DP=2 x PP=2 x TP=4 worked example). Fabric manifest and NIC
selection are design-only.

## Open tasks

- PLACE-1: fabric topology schema contents and NIC selection in the mapper
  (milestone M4), sourcing intra-node structure from NCCL topology dumps.
- PLACE-2: `unique-nic` GOAL-rank mapping (depends on PLACE-1).
- PLACE-3: the extraction path, i.e. the worker-side manifest exporter
  invoked over `collective_rpc` (ships with the vLLM adapter, see
  [adapters-vllm](adapters-vllm.md)).
