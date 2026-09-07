"""Project the fixed declared Clos into htsim's contiguous leaf endpoint slots.

The topology file describes a regular switch graph; GOAL endpoint numbering
encodes its NIC attachments. The semantic execution graph remains unchanged.
The returned graph and completion rows are read-only identity projections.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from simllm.backends.htsim_rnic import FlowCompletion
from simllm.core import CollectiveWork, ComputeWork, ExecutionGraph
from simllm.core.execution_io import validate_execution_graph
from simllm.placement import FabricTopologyManifest
from simllm.traffic.collective_plan import plan_execution_graph_collectives


@dataclass(frozen=True)
class HtsimClosProjection:
    """A validated topology text plus a bijection of semantic ranks to slots."""

    topology_text: str
    endpoint_by_rank: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.endpoint_by_rank) != 64 or any(
            type(rank) is not int for rank in self.endpoint_by_rank
        ) or set(self.endpoint_by_rank) != set(range(64)):
            raise ValueError("endpoint projection must be a bijection of 64 integer ranks")

    def project_graph(self, graph: ExecutionGraph) -> ExecutionGraph:
        """Map supported compute and collective participants, preserving edges."""
        validate_execution_graph(graph)
        mapping = self.endpoint_by_rank
        operations = []
        for operation in graph.operations:
            if operation.rank >= len(mapping):
                raise ValueError("graph rank is outside the fabric")
            work = operation.work
            if isinstance(work, CollectiveWork):
                if any(rank >= len(mapping) for rank in work.ranks):
                    raise ValueError("collective rank is outside the fabric")
                work = replace(
                    work,
                    ranks=tuple(mapping[rank] for rank in work.ranks),
                    pair_payload_bytes=tuple(sorted(
                        (mapping[source], mapping[destination], size)
                        for source, destination, size in work.pair_payload_bytes
                    )),
                    request_pair_payload_bytes=tuple(sorted(
                        (request, mapping[source], mapping[destination], size)
                        for request, source, destination, size in work.request_pair_payload_bytes
                    )),
                )
            elif not isinstance(work, ComputeWork):
                raise TypeError("Clos projection supports only compute and collective work")
            operations.append(replace(operation, rank=mapping[operation.rank], work=work))
        if mapping == tuple(range(64)):
            return graph
        if graph.collective_plans and graph.collective_plans != (
            plan_execution_graph_collectives(replace(graph, collective_plans=())).collective_plans
        ):
            raise ValueError("endpoint projection requires canonical traffic-owned plans")
        projected = replace(graph, operations=tuple(operations), collective_plans=())
        validate_execution_graph(projected)
        if graph.collective_plans:
            projected = plan_execution_graph_collectives(projected)
        return projected

    def semantic_flow(self, flow: FlowCompletion) -> FlowCompletion:
        """Undo endpoint numbering without altering any backend timing or identity."""
        if any(type(rank) is not int or not 0 <= rank < 64 for rank in (flow.source, flow.destination)):
            raise ValueError("completion endpoint is outside the fabric")
        inverse = {endpoint: rank for rank, endpoint in enumerate(self.endpoint_by_rank)}
        return replace(flow, source=inverse[flow.source], destination=inverse[flow.destination])


def project_declared_clos(fabric: FabricTopologyManifest) -> HtsimClosProjection:
    """Validate and render the fixed eight-leaf, eight-spine physical graph.

    Unsupported topology or nonuniform timing fails before the backend runs.
    Link timing is read from the manifest, not from the fabric variant name.
    The htsim null-network profile bypasses this switch graph; only a physical
    network profile uses the topology text to time traversed links.
    """
    if not isinstance(fabric, FabricTopologyManifest):
        raise TypeError("fabric must be a FabricTopologyManifest")
    fabric.validate()
    leaves = sorted((switch for switch in fabric.switches if switch.tier == 0),
                    key=lambda switch: switch.switch_id)
    spines = sorted((switch for switch in fabric.switches if switch.tier == 1),
                    key=lambda switch: switch.switch_id)
    if len(leaves) != 8 or len(spines) != 8 or len(fabric.switches) != 16:
        raise ValueError("only the fixed eight-leaf, eight-spine Clos is supported")
    nics = {nic.nic_id: nic for node in fabric.nodes for nic in node.nics}
    ranks = {gpu.global_rank: gpu for node in fabric.nodes for gpu in node.gpus}
    if len(nics) != 64 or set(ranks) != set(range(64)) or len(fabric.links) != 128:
        raise ValueError("the fixed Clos requires 64 affine endpoints and 128 links")
    timing = {(link.link_rate_bps, link.propagation_delay_ps) for link in fabric.links}
    if len(timing) != 1:
        raise ValueError("the fixed Clos renderer requires uniform link timing")
    rate, delay = next(iter(timing))
    latency = fabric.switch_latency_ps
    if latency is None or rate % 1_000_000_000 or delay % 1000 or latency % 1000:
        raise ValueError("htsim topology needs whole Gbit/s and nanosecond timing")
    ports = {port.port_id: switch.switch_id for switch in fabric.switches for port in switch.ports}
    leaf_ids = {switch.switch_id for switch in leaves}
    spine_ids = {switch.switch_id for switch in spines}
    spine_pairs = set()
    endpoint_links = 0
    for link in fabric.links:
        if link.endpoint_a in nics or link.endpoint_b in nics:
            endpoint_links += 1
            continue
        pair = (ports[link.endpoint_a], ports[link.endpoint_b])
        if pair[0] in spine_ids:
            pair = pair[::-1]
        if pair[0] not in leaf_ids or pair[1] not in spine_ids or pair in spine_pairs:
            raise ValueError("fabric links must connect each leaf to each spine exactly once")
        spine_pairs.add(pair)
    if endpoint_links != 64 or len(spine_pairs) != 64:
        raise ValueError("fixed Clos endpoint or spine wiring is incomplete")
    slots_by_nic = {}
    for leaf_index, leaf in enumerate(leaves):
        attached = sorted((nic for nic in nics.values() if nic.switch_id == leaf.switch_id),
                          key=lambda nic: nic.switch_port_id)
        if len(attached) != 8:
            raise ValueError("each leaf must have exactly eight NIC endpoints")
        for port_index, nic in enumerate(attached):
            slots_by_nic[nic.nic_id] = 8 * leaf_index + port_index
    if len(slots_by_nic) != 64:
        raise ValueError("all NIC endpoints must attach directly to leaves")
    endpoints = tuple(slots_by_nic[ranks[rank].nic_id] for rank in range(64))
    text = (
        "Nodes 64\nTiers 2\nPodsize 64\n\n"
        f"Tier 0\nDownlink_speed_Gbps {rate // 1_000_000_000}\n"
        f"Radix_Down 8\nRadix_Up 8\nDownlink_Latency_ns {delay // 1000}\n"
        f"Switch_Latency_ns {latency // 1000}\n\n"
        f"Tier 1\nDownlink_speed_Gbps {rate // 1_000_000_000}\n"
        f"Radix_Down 8\nDownlink_Latency_ns {delay // 1000}\n"
        f"Switch_Latency_ns {latency // 1000}\n"
    )
    return HtsimClosProjection(text, endpoints)
