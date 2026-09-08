"""Physical GPU attachments inside the existing fabric manifest.

Links retain FabricLink's full-duplex endpoint contract. A route selects one
or more explicitly declared physical paths; a rank pair never creates a link.
The initial switched domain has one store-and-forward switch per path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .manifest import FabricLink

if TYPE_CHECKING:
    from .manifest import FabricTopologyManifest


def _text(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonblank")


def _integer(value: object, name: str, *, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class PeerPortPlacement:
    """One physical port owned by exactly one GPU or one peer switch."""

    port_id: str
    gpu_rank: int | None = None
    switch_id: str | None = None

    def __post_init__(self) -> None:
        _text(self.port_id, "peer port_id")
        if (self.gpu_rank is None) == (self.switch_id is None):
            raise ValueError("peer port requires exactly one GPU or switch owner")
        if self.gpu_rank is not None:
            _integer(self.gpu_rank, "peer GPU rank")
        if self.switch_id is not None:
            _text(self.switch_id, "peer switch_id")


@dataclass(frozen=True)
class PeerRoute:
    """Alternative explicit physical paths between two semantic GPU ranks."""

    source_rank: int
    destination_rank: int
    paths: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        _integer(self.source_rank, "peer source rank")
        _integer(self.destination_rank, "peer destination rank")
        if self.source_rank == self.destination_rank:
            raise ValueError("peer route cannot be a self-pair")
        if not isinstance(self.paths, (list, tuple)) or any(
            not isinstance(path, (list, tuple)) for path in self.paths
        ):
            raise TypeError("peer paths must be an array of link arrays")
        object.__setattr__(self, "paths", tuple(tuple(path) for path in self.paths))
        if not self.paths or len(set(self.paths)) != len(self.paths):
            raise ValueError("peer route requires unique physical paths")
        for path in self.paths:
            if len(path) not in (1, 2) or len(set(path)) != len(path):
                raise ValueError("peer path requires one direct or two switched links")
            for link_id in path:
                _text(link_id, "peer route link identity")


@dataclass(frozen=True)
class ResolvedPeerPath:
    """An immutable directional projection of the manifest's physical links."""

    source_port_id: str
    destination_port_id: str
    input_link: FabricLink
    output_link: FabricLink | None = None
    switch_id: str | None = None
    switch_input_port_id: str | None = None
    switch_output_port_id: str | None = None

    @property
    def input_resource(self) -> tuple[str, str]:
        return self.input_link.link_id, self.source_port_id

    @property
    def output_resource(self) -> tuple[str, str] | None:
        if self.output_link is None or self.switch_output_port_id is None:
            return None
        return self.output_link.link_id, self.switch_output_port_id


@dataclass(frozen=True)
class PeerFabric:
    """One node's declared direct or switched GPU attachment domain."""

    domain_id: str
    node_id: str
    ports: tuple[PeerPortPlacement, ...]
    links: tuple[FabricLink, ...]
    routes: tuple[PeerRoute, ...]
    protocol: str = "nvlink"
    evidence_class: str = "declared"
    switch_input_buffer_bytes: int = 65536

    def __post_init__(self) -> None:
        _text(self.domain_id, "peer domain_id")
        _text(self.node_id, "peer node_id")
        if self.protocol not in ("nvlink", "xgmi", "ualink"):
            raise ValueError("unknown peer topology protocol")
        if self.evidence_class != "declared":
            raise ValueError("peer topology requires explicit declared evidence")
        _integer(self.switch_input_buffer_bytes, "switch input buffer bytes", minimum=1)
        for name in ("ports", "links", "routes"):
            if not isinstance(getattr(self, name), (list, tuple)):
                raise TypeError(f"peer {name} must be an array")
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if not self.ports or not self.links or not self.routes:
            raise ValueError("peer domain requires physical ports, links and routes")
        if not all(isinstance(port, PeerPortPlacement) for port in self.ports):
            raise TypeError("peer ports require PeerPortPlacement")
        if not all(isinstance(link, FabricLink) for link in self.links):
            raise TypeError("peer links require FabricLink")
        if not all(isinstance(route, PeerRoute) for route in self.routes):
            raise TypeError("peer routes require PeerRoute")
        ports = {port.port_id: port for port in self.ports}
        links = {link.link_id: link for link in self.links}
        if len(ports) != len(self.ports) or len(links) != len(self.links):
            raise ValueError("peer port and link inventories must be unique")
        if set(ports) & set(links):
            raise ValueError("peer port and link identities cannot alias")
        attached: set[str] = set()
        for link in self.links:
            _text(link.link_id, "peer link_id")
            _integer(link.link_rate_bps, "peer link rate", minimum=1)
            _integer(link.propagation_delay_ps, "peer propagation")
            endpoints = (link.endpoint_a, link.endpoint_b)
            if endpoints[0] == endpoints[1] or any(endpoint not in ports for endpoint in endpoints):
                raise ValueError("peer link endpoints must be distinct declared ports")
            if any(endpoint in attached for endpoint in endpoints):
                raise ValueError("one physical peer port cannot have two attachments")
            attached.update(endpoints)
        if attached != set(ports):
            raise ValueError("peer topology contains an unattached physical port")
        pairs = [(route.source_rank, route.destination_rank) for route in self.routes]
        if len(set(pairs)) != len(pairs):
            raise ValueError("peer rank pair has two route authorities")
        if len({len(path) for route in self.routes for path in route.paths}) != 1:
            raise ValueError("one peer domain must use one direct or switched route kind")
        for route in self.routes:
            self.paths_between(route.source_rank, route.destination_rank)

    @property
    def switched(self) -> bool:
        return len(self.routes[0].paths[0]) == 2

    def validate(self, manifest: FabricTopologyManifest) -> None:
        nodes = [node for node in manifest.nodes if node.node_id == self.node_id]
        if len(nodes) != 1:
            raise ValueError("peer domain must name exactly one fabric node")
        ranks = {gpu.global_rank for gpu in nodes[0].gpus}
        all_ranks = [gpu.global_rank for node in manifest.nodes for gpu in node.gpus]
        if len(set(all_ranks)) != len(all_ranks):
            raise ValueError("fabric inventory has duplicate GPU rank ownership")
        if any(gpu.node_id != self.node_id for gpu in nodes[0].gpus):
            raise ValueError("GPU attachment disagrees with its containing node")
        owners = {port.gpu_rank for port in self.ports if port.gpu_rank is not None}
        if not owners <= ranks:
            raise ValueError("peer port is outside its declared physical node")

    def paths_between(self, source: int, destination: int) -> tuple[ResolvedPeerPath, ...]:
        route = next((row for row in self.routes
                      if (row.source_rank, row.destination_rank) == (source, destination)), None)
        if route is None:
            raise ValueError(f"peer domain has no declared route {source} to {destination}")
        ports = {port.port_id: port for port in self.ports}
        links = {link.link_id: link for link in self.links}
        result = []
        for path in route.paths:
            if any(link_id not in links for link_id in path):
                raise ValueError("peer route names an unknown physical link")
            first = links[path[0]]
            source_ports = [port for port in (first.endpoint_a, first.endpoint_b)
                            if ports[port].gpu_rank == source]
            if len(source_ports) != 1:
                raise ValueError("peer route first link is not attached to its source GPU")
            source_port = source_ports[0]
            other = first.endpoint_b if first.endpoint_a == source_port else first.endpoint_a
            if len(path) == 1:
                if ports[other].gpu_rank != destination:
                    raise ValueError("direct peer link does not reach its destination")
                result.append(ResolvedPeerPath(source_port, other, first))
                continue
            second = links[path[1]]
            destination_ports = [port for port in (second.endpoint_a, second.endpoint_b)
                                 if ports[port].gpu_rank == destination]
            if len(destination_ports) != 1:
                raise ValueError("peer route last link is not attached to its destination GPU")
            destination_port = destination_ports[0]
            output = second.endpoint_b if second.endpoint_a == destination_port else second.endpoint_a
            switch = ports[other].switch_id
            if switch is None or ports[output].switch_id != switch or other == output:
                raise ValueError("switched peer route must cross two ports of one switch")
            result.append(ResolvedPeerPath(source_port, destination_port, first, second,
                                           switch, other, output))
        return tuple(result)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PeerFabric:
        if not isinstance(value, dict) or set(value) != {
            "domain_id", "node_id", "ports", "links", "routes", "protocol",
            "evidence_class", "switch_input_buffer_bytes",
        }:
            raise ValueError("peer fabric object has missing or unknown fields")
        return cls(
            domain_id=value["domain_id"], node_id=value["node_id"],
            ports=tuple(PeerPortPlacement(**row) for row in value["ports"]),
            links=tuple(FabricLink(**row) for row in value["links"]),
            routes=tuple(PeerRoute(**row) for row in value["routes"]),
            protocol=value["protocol"], evidence_class=value["evidence_class"],
            switch_input_buffer_bytes=value["switch_input_buffer_bytes"],
        )
