"""Placement manifest: global rank → node → GPU → shard → process groups.

A manifest can be **declared** (a what-if placement for a simulated
deployment) or **extracted** from a live run (each worker exports its own
entry; in vLLM this is one ``collective_rpc`` over the workers). Both produce
the same schema, which is what makes simulated and real deployments directly
comparable.

Extraction rules that matter for correctness:

- Export the *actual* group memberships (e.g. vLLM ``GroupCoordinator.ranks``)
  rather than recomputing them from a rank formula — external DP, elastic
  scaling or implementation changes silently break derived layouts.
- Use GPU UUID or PCI bus ID as the stable cross-system GPU identifier.
- Record the framework version/commit in the manifest; the extraction surface
  is internal API.
- With dynamic expert load balancing (EPLB), expert ownership changes at
  runtime: every re-placement bumps ``placement_epoch`` and traffic events
  reference the epoch they were routed under.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

PLACEMENT_SCHEMA = "simllm-placement-manifest-v1"


@dataclass
class GroupMembership:
    """One rank's view of one process group (tp/pp/dp/ep/pcp/...)."""

    rank_in_group: int
    global_ranks: list[int]


@dataclass
class RankPlacement:
    """Physical and logical placement of one global rank."""

    global_rank: int
    hostname: str
    local_rank: int
    gpu_uuid: str | None = None
    pci_bus_id: str | None = None
    #: group name ("tp", "pp", "dp", "ep", "pcp", ...) → membership
    groups: dict[str, GroupMembership] = field(default_factory=dict)
    #: pipeline layer ownership as [start, end) — take the model's actual
    #: range, partitions are not guaranteed equal
    pipeline_layer_range: tuple[int, int] | None = None
    #: MoE layer id → global expert ids owned by this rank
    local_expert_ids: dict[int, list[int]] = field(default_factory=dict)
    #: expert-placement epoch these expert assignments belong to
    placement_epoch: int = 0


@dataclass
class PlacementManifest:
    ranks: list[RankPlacement]
    #: "declared" (what-if) or "extracted" (from a live run)
    source: str = "declared"
    framework: str | None = None
    framework_version: str | None = None
    schema: str = PLACEMENT_SCHEMA

    def by_rank(self, global_rank: int) -> RankPlacement:
        for r in self.ranks:
            if r.global_rank == global_rank:
                return r
        raise KeyError(f"global rank {global_rank} not in manifest")

    def group_ranks(self, global_rank: int, group: str) -> list[int]:
        """Global ranks of ``group`` ("tp", "ep", ...) as seen by a member."""
        return self.by_rank(global_rank).groups[group].global_ranks

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")
        return path

    @classmethod
    def load(cls, path: str | Path) -> PlacementManifest:
        raw = json.loads(Path(path).read_text())
        if raw.get("schema") != PLACEMENT_SCHEMA:
            raise ValueError(f"unsupported schema: {raw.get('schema')!r}")
        ranks = []
        for r in raw["ranks"]:
            groups = {
                name: GroupMembership(**g) for name, g in r.pop("groups", {}).items()
            }
            layer_range = r.pop("pipeline_layer_range", None)
            expert_ids = {
                int(layer): ids for layer, ids in r.pop("local_expert_ids", {}).items()
            }
            ranks.append(
                RankPlacement(
                    **r,
                    groups=groups,
                    pipeline_layer_range=tuple(layer_range) if layer_range else None,
                    local_expert_ids=expert_ids,
                )
            )
        return cls(
            ranks=ranks,
            source=raw.get("source", "declared"),
            framework=raw.get("framework"),
            framework_version=raw.get("framework_version"),
        )
