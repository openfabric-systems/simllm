"""Placement manifest: global rank → node → GPU → shard → process groups.

A manifest can be **declared** (a what-if placement for a simulated
deployment) or **extracted** from a live run (each worker exports its own
entry; in vLLM this is one ``collective_rpc`` over the workers). Both produce
the same schema, which is what makes simulated and real deployments directly
comparable.

Extraction rules that matter for correctness:

- Export the *actual* group memberships (e.g. vLLM ``GroupCoordinator.ranks``)
  rather than recomputing them from a rank formula: external DP, elastic
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
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

PLACEMENT_SCHEMA = "simllm-placement-manifest-v1"

#: Fabric topology manifest: the physical graph under the ranks (nodes, GPUs,
#: PCIe/NVLink links, NICs, GPU-to-NIC affinity, switches, links, bandwidths,
#: delays, queue configuration). Intra-node structure can come from NCCL's
#: detected topology (NCCL_TOPO_DUMP_FILE); the switch-level graph always
#: comes from a cluster inventory or the simulator topology config. Concrete
#: contents land with the M4 mapper work (PLACE-1); the schema name is pinned
#: here so every producer and consumer agrees early.
FABRIC_SCHEMA = "simllm-fabric-topology-v1"


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
    #: pipeline layer ownership as [start, end); take the model's actual
    #: range, partitions are not guaranteed equal
    pipeline_layer_range: tuple[int, int] | None = None
    #: MoE layer id → global expert ids owned by this rank
    local_expert_ids: dict[int, list[int]] = field(default_factory=dict)
    #: expert-placement epoch these expert assignments belong to
    placement_epoch: int = 0
    #: disaggregated serving pool role; absent for ordinary placements
    pool_role: str | None = None


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
        raw = asdict(self)
        for rank in raw["ranks"]:
            if rank["pool_role"] is None:
                del rank["pool_role"]
        path.write_text(json.dumps(raw, indent=2) + "\n")
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


@dataclass(frozen=True)
class GpuFabricPlacement:
    """One simulated GPU's concrete node, PCIe, and NIC attachment."""

    global_rank: int
    gpu_id: str
    node_id: str
    pcie_location: str
    nic_id: str


@dataclass(frozen=True)
class NicFabricPlacement:
    """One GPU-affine NIC pinned to a switch-facing fabric location."""

    nic_id: str
    node_id: str
    fabric_location: str
    affine_gpu_rank: int
    switch_id: str | None = None
    switch_port_id: str | None = None
    link_id: str | None = None


@dataclass(frozen=True)
class FabricSwitchPort:
    """One uniquely identified port owned by a fabric switch."""

    port_id: str
    switch_id: str


@dataclass(frozen=True)
class FabricSwitchPlacement:
    """One switch and its physical ports in the declared fabric graph."""

    switch_id: str
    tier: int
    ports: tuple[FabricSwitchPort, ...]


@dataclass(frozen=True)
class FabricLink:
    """One undirected physical link between NIC or switch-port endpoints."""

    link_id: str
    endpoint_a: str
    endpoint_b: str
    link_rate_bps: int
    propagation_delay_ps: int


@dataclass(frozen=True)
class FabricNodePlacement:
    """The concrete GPU and NIC inventory of one serving node."""

    node_id: str
    pool_role: str
    gpus: tuple[GpuFabricPlacement, ...]
    nics: tuple[NicFabricPlacement, ...]


@dataclass
class FabricTopologyManifest:
    """Concrete disaggregated inventory using the pinned fabric schema."""

    nodes: list[FabricNodePlacement]
    goal_rank_mapping: str = "gpu-rank"
    source: str = "declared"
    schema: str = FABRIC_SCHEMA
    physical_rendering_enabled: bool = False
    topology_name: str | None = None
    evidence_class: str | None = None
    switch_latency_ps: int | None = None
    switches: tuple[FabricSwitchPlacement, ...] = ()
    links: tuple[FabricLink, ...] = ()

    def by_rank(self, global_rank: int) -> GpuFabricPlacement:
        for node in self.nodes:
            for gpu in node.gpus:
                if gpu.global_rank == global_rank:
                    return gpu
        raise KeyError(f"global rank {global_rank} not in fabric manifest")

    def by_nic(self, nic_id: str) -> NicFabricPlacement:
        for node in self.nodes:
            for nic in node.nics:
                if nic.nic_id == nic_id:
                    return nic
        raise KeyError(f"NIC {nic_id!r} not in fabric manifest")

    def by_link(self, link_id: str) -> FabricLink:
        for link in self.links:
            if link.link_id == link_id:
                return link
        raise KeyError(f"link {link_id!r} not in fabric manifest")

    def _physical_indexes(
        self,
    ) -> tuple[
        dict[str, NicFabricPlacement],
        dict[str, FabricSwitchPort],
        dict[str, FabricLink],
    ]:
        if not self.physical_rendering_enabled:
            raise ValueError("physical topology rendering is disabled")
        nic_by_id = {
            nic.nic_id: nic for node in self.nodes for nic in node.nics
        }
        port_by_id = {
            port.port_id: port for switch in self.switches for port in switch.ports
        }
        link_by_id = {link.link_id: link for link in self.links}
        return nic_by_id, port_by_id, link_by_id

    def validate(self) -> None:
        """Fail closed unless the optional physical graph is self-consistent."""

        if self.schema != FABRIC_SCHEMA:
            raise ValueError(f"unsupported schema: {self.schema!r}")
        if self.goal_rank_mapping != "gpu-rank":
            raise ValueError("fixed fabric topology requires gpu-rank mapping")
        if type(self.physical_rendering_enabled) is not bool:
            raise TypeError("physical_rendering_enabled must be a boolean")

        nics = tuple(nic for node in self.nodes for nic in node.nics)
        nic_ids = [nic.nic_id for nic in nics]
        if len(nic_ids) != len(set(nic_ids)):
            raise ValueError("fabric NIC identities must be unique")

        if not self.physical_rendering_enabled:
            if self.switches or self.links:
                raise ValueError("disabled physical rendering cannot retain a graph")
            if any(
                value is not None
                for value in (
                    self.topology_name,
                    self.evidence_class,
                    self.switch_latency_ps,
                )
            ):
                raise ValueError("disabled physical rendering cannot retain metadata")
            if any(
                nic.switch_id is not None
                or nic.switch_port_id is not None
                or nic.link_id is not None
                for nic in nics
            ):
                raise ValueError("disabled physical rendering cannot retain NIC links")
            return

        for name, value in (
            ("topology_name", self.topology_name),
            ("evidence_class", self.evidence_class),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonblank string")
        if (
            isinstance(self.switch_latency_ps, bool)
            or type(self.switch_latency_ps) is not int
            or self.switch_latency_ps < 0
        ):
            raise ValueError("switch_latency_ps must be a nonnegative integer")
        if not self.switches or not self.links:
            raise ValueError("enabled physical rendering requires switches and links")

        switch_ids = [switch.switch_id for switch in self.switches]
        if len(switch_ids) != len(set(switch_ids)):
            raise ValueError("fabric switch identities must be unique")
        switch_id_set = set(switch_ids)
        ports = tuple(port for switch in self.switches for port in switch.ports)
        port_ids = [port.port_id for port in ports]
        if len(port_ids) != len(set(port_ids)):
            raise ValueError("fabric switch-port identities must be unique")
        for switch in self.switches:
            if isinstance(switch.tier, bool) or type(switch.tier) is not int:
                raise TypeError("fabric switch tier must be an integer")
            if switch.tier not in (0, 1):
                raise ValueError("fixed fabric switch tier must be zero or one")
            if not switch.ports:
                raise ValueError("every fabric switch must own at least one port")
            if any(port.switch_id != switch.switch_id for port in switch.ports):
                raise ValueError("fabric port owner disagrees with its switch")
        if any(port.switch_id not in switch_id_set for port in ports):
            raise ValueError("fabric port names an unknown switch")

        link_ids = [link.link_id for link in self.links]
        if len(link_ids) != len(set(link_ids)):
            raise ValueError("fabric link identities must be unique")
        known_endpoints = set(nic_ids) | set(port_ids)
        endpoint_degree = {endpoint: 0 for endpoint in known_endpoints}
        for link in self.links:
            if link.endpoint_a == link.endpoint_b:
                raise ValueError("fabric link endpoints must be distinct")
            if link.endpoint_a not in known_endpoints or link.endpoint_b not in known_endpoints:
                raise ValueError("fabric link names an unknown endpoint")
            for name, value in (
                ("link_rate_bps", link.link_rate_bps),
                ("propagation_delay_ps", link.propagation_delay_ps),
            ):
                if isinstance(value, bool) or type(value) is not int or value < 1:
                    raise ValueError(f"fabric {name} must be a positive integer")
            endpoint_degree[link.endpoint_a] += 1
            endpoint_degree[link.endpoint_b] += 1
        if any(degree != 1 for degree in endpoint_degree.values()):
            raise ValueError("every NIC and switch port must terminate one link")

        nic_by_id, port_by_id, link_by_id = self._physical_indexes()
        for nic in nics:
            if nic.switch_id not in switch_id_set:
                raise ValueError("fabric NIC names an unknown switch")
            port = port_by_id.get(nic.switch_port_id or "")
            if port is None or port.switch_id != nic.switch_id:
                raise ValueError("fabric NIC switch-port binding is inconsistent")
            if nic.fabric_location != port.port_id:
                raise ValueError("fabric NIC location disagrees with its switch port")
            link = link_by_id.get(nic.link_id or "")
            if link is None or {link.endpoint_a, link.endpoint_b} != {
                nic.nic_id,
                port.port_id,
            }:
                raise ValueError("fabric NIC endpoint-link binding is inconsistent")

        adjacency = self._adjacency()
        first_nic = next(iter(nic_by_id), None)
        if first_nic is None:
            raise ValueError("physical fabric must contain at least one NIC")
        reached = {first_nic}
        pending = deque([first_nic])
        while pending:
            endpoint = pending.popleft()
            for peer, _link in adjacency[endpoint]:
                if peer not in reached:
                    reached.add(peer)
                    pending.append(peer)
        if not set(nic_by_id).issubset(reached):
            raise ValueError("physical fabric does not reach every NIC endpoint")

    def _adjacency(self) -> dict[str, list[tuple[str, FabricLink | None]]]:
        nic_by_id, port_by_id, _link_by_id = self._physical_indexes()
        switch_ids = {switch.switch_id for switch in self.switches}
        adjacency: dict[str, list[tuple[str, FabricLink | None]]] = {
            endpoint: []
            for endpoint in set(nic_by_id) | set(port_by_id) | switch_ids
        }
        for link in self.links:
            adjacency[link.endpoint_a].append((link.endpoint_b, link))
            adjacency[link.endpoint_b].append((link.endpoint_a, link))
        for port in port_by_id.values():
            adjacency[port.port_id].append((port.switch_id, None))
            adjacency[port.switch_id].append((port.port_id, None))
        for peers in adjacency.values():
            peers.sort(
                key=lambda item: (
                    "" if item[1] is None else item[1].link_id,
                    item[0],
                )
            )
        return adjacency

    def path_between_ranks(
        self, source_rank: int, destination_rank: int
    ) -> tuple[FabricLink, ...]:
        """Return the deterministic shortest physical path between two ranks."""

        if source_rank == destination_rank:
            self.by_rank(source_rank)
            return ()
        source_nic = self.by_rank(source_rank).nic_id
        destination_nic = self.by_rank(destination_rank).nic_id
        adjacency = self._adjacency()
        if source_nic not in adjacency or destination_nic not in adjacency:
            raise ValueError("rank NIC is absent from the physical topology")

        predecessor: dict[str, tuple[str, FabricLink | None] | None] = {
            source_nic: None
        }
        pending = deque([source_nic])
        while pending and destination_nic not in predecessor:
            endpoint = pending.popleft()
            for peer, link in adjacency[endpoint]:
                if peer not in predecessor:
                    predecessor[peer] = (endpoint, link)
                    pending.append(peer)
        if destination_nic not in predecessor:
            raise ValueError(
                f"no physical path from rank {source_rank} to {destination_rank}"
            )

        path: list[FabricLink] = []
        endpoint = destination_nic
        while endpoint != source_nic:
            previous = predecessor[endpoint]
            assert previous is not None
            endpoint, link = previous
            if link is not None:
                path.append(link)
        path.reverse()
        return tuple(path)

    def resolve_goal_paths(self, trace: object) -> tuple[tuple[FabricLink, ...], ...]:
        """Resolve each structured GOAL message through this physical graph."""

        from simllm.goal import GoalTrace

        if not isinstance(trace, GoalTrace):
            raise TypeError("trace must be a GoalTrace")
        fabric_ranks = sorted(
            gpu.global_rank for node in self.nodes for gpu in node.gpus
        )
        if fabric_ranks != list(range(len(fabric_ranks))):
            raise ValueError("fabric GPU ranks must be dense for GOAL projection")
        if trace.num_ranks != len(fabric_ranks):
            raise ValueError("GOAL and fabric rank cardinalities disagree")
        return tuple(
            self.path_between_ranks(message.source_rank, message.destination_rank)
            for message in trace.messages
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")
        return path

    @classmethod
    def load(cls, path: str | Path) -> FabricTopologyManifest:
        raw = json.loads(Path(path).read_text())
        if raw.get("schema") != FABRIC_SCHEMA:
            raise ValueError(f"unsupported schema: {raw.get('schema')!r}")
        nodes = []
        for node in raw["nodes"]:
            nodes.append(
                FabricNodePlacement(
                    node_id=node["node_id"],
                    pool_role=node["pool_role"],
                    gpus=tuple(
                        GpuFabricPlacement(**gpu) for gpu in node.get("gpus", ())
                    ),
                    nics=tuple(
                        NicFabricPlacement(**nic) for nic in node.get("nics", ())
                    ),
                )
            )
        switches = tuple(
            FabricSwitchPlacement(
                switch_id=switch["switch_id"],
                tier=switch["tier"],
                ports=tuple(
                    FabricSwitchPort(**port) for port in switch.get("ports", ())
                ),
            )
            for switch in raw.get("switches", ())
        )
        links = tuple(FabricLink(**link) for link in raw.get("links", ()))
        return cls(
            nodes=nodes,
            goal_rank_mapping=raw.get("goal_rank_mapping", "gpu-rank"),
            source=raw.get("source", "declared"),
            physical_rendering_enabled=raw.get(
                "physical_rendering_enabled", False
            ),
            topology_name=raw.get("topology_name"),
            evidence_class=raw.get("evidence_class"),
            switch_latency_ps=raw.get("switch_latency_ps"),
            switches=switches,
            links=links,
        )
