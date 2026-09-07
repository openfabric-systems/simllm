from __future__ import annotations

import json
from dataclasses import asdict, replace

import pytest

from simllm.backends.htsim_rnic import FlowCompletion
from simllm.backends.rail_topology import HtsimClosProjection, project_declared_clos
from simllm.core import CollectiveWork, ExecutionGraph, ExecutionOperation, OperationCorrelation
from simllm.placement import (
    FabricTopologyManifest,
    declared_manifest,
    declared_pipeline_placement,
    declared_rail_fabric,
    disaggregated_manifests,
)
from simllm.traffic import plan_execution_graph_collectives, render_serial_execution_graph_goal


@pytest.mark.parametrize("width", [1, 2, 4, 8])
def test_reference_group_placement(width):
    placement = declared_pipeline_placement(width)
    assert placement == declared_manifest(tp=8, pp=width, dp=8 // width, nodes=8)
    assert len(placement.ranks) == 64
    for nic in range(8):
        assert placement.group_ranks(nic, "pp") == [8 * stage + nic for stage in range(width)]


@pytest.mark.parametrize("variant", ["rail", "node-local"])
def test_manifest_paths_and_wire_round_trip(tmp_path, variant):
    placement = declared_pipeline_placement(8)
    before = json.dumps(asdict(placement), sort_keys=True)
    fabric = declared_rail_fabric(placement, variant=variant)
    fabric.validate()
    assert len(fabric.nodes) == 8
    assert len(fabric.links) == 128
    assert len(fabric.switches) == 16
    assert {link.link_rate_bps for link in fabric.links} == {400_000_000_000}
    assert {link.propagation_delay_ps for link in fabric.links} == {1_000_000}
    assert fabric.switch_latency_ps == 0
    assert json.dumps(asdict(placement), sort_keys=True) == before
    for rank in range(64):
        nic = fabric.by_nic(fabric.by_rank(rank).nic_id)
        leaf = rank % 8 if variant == "rail" else rank // 8
        assert nic.switch_id == f"leaf-{leaf:03d}"
        assert nic.affine_gpu_rank == rank
    for source in range(64):
        for destination in range(source + 1, 64):
            shared_leaf = (source % 8 == destination % 8 if variant == "rail"
                           else source // 8 == destination // 8)
            path = fabric.path_between_ranks(source, destination)
            assert len(path) == (2 if shared_leaf else 4)
    path = fabric.save(tmp_path / "fabric.json")
    loaded = FabricTopologyManifest.load(path)
    assert loaded == fabric
    assert loaded.save(tmp_path / "again.json").read_bytes() == path.read_bytes()


def test_spine_wiring_identical_across_variants():
    placement = declared_pipeline_placement(4)
    rail = declared_rail_fabric(placement, variant="rail")
    local = declared_rail_fabric(placement, variant="node-local")
    assert rail.switches == local.switches
    spine_links = lambda fabric: tuple(link for link in fabric.links
                                      if link.link_id.startswith("fabric-link"))
    assert spine_links(rail) == spine_links(local)
    assert project_declared_clos(rail).topology_text == project_declared_clos(local).topology_text


def test_existing_builder_bytes_unchanged_by_new_builder(tmp_path):
    before = disaggregated_manifests(prefill_nodes=4, decode_nodes=4)
    p = before.placement.save(tmp_path / "before-placement.json").read_bytes()
    f = before.fabric.save(tmp_path / "before-fabric.json").read_bytes()
    declared_rail_fabric(declared_pipeline_placement(), variant="rail")
    after = disaggregated_manifests(prefill_nodes=4, decode_nodes=4)
    assert after.placement.save(tmp_path / "after-placement.json").read_bytes() == p
    assert after.fabric.save(tmp_path / "after-fabric.json").read_bytes() == f


def test_legacy_disabled_fabric_bytes_remain_readable(tmp_path):
    fabric = disaggregated_manifests(prefill_nodes=1, decode_nodes=1,
                                    render_physical_topology=False).fabric
    raw = asdict(fabric)
    for key in ("physical_rendering_enabled", "topology_name", "evidence_class",
                "switch_latency_ps", "switches", "links"):
        raw.pop(key)
    for node in raw["nodes"]:
        for nic in node["nics"]:
            for key in ("switch_id", "switch_port_id", "link_id"):
                nic.pop(key)
    path = tmp_path / "old.json"
    path.write_text(json.dumps(raw))
    loaded = FabricTopologyManifest.load(path)
    assert loaded == fabric


@pytest.mark.parametrize("variant", ["rail", "node-local"])
def test_endpoint_projection_and_inverse_preserve_messages(variant):
    mapping = project_declared_clos(declared_rail_fabric(declared_pipeline_placement(), variant=variant))
    work = CollectiveWork("all-to-allv", (0, 8), 0, "pairwise", "pp-forward-0",
                          ((0, 8, 65536),), (("request", 0, 8, 65536),))
    graph = plan_execution_graph_collectives(ExecutionGraph(
        "step-0", 0, 0, (ExecutionOperation("pp", 0, "pp", work,
                                                  correlation=OperationCorrelation(request_ids=("request",))),), ("pp",),
    ))
    projected = mapping.project_graph(graph)
    trace = render_serial_execution_graph_goal(projected, num_goal_ranks=64)
    assert len(trace.messages) == 1
    message = trace.messages[0]
    assert message.source_rank == 0
    assert message.destination_rank == (1 if variant == "rail" else 8)
    assert message.request_payload_bytes == (("request", 65536),)
    completion = FlowCompletion("rnic-nn", 9, message.source_rank, message.destination_rank,
                                message.tag, 65536, 1000000, 5000000, 4000000)
    semantic = mapping.semantic_flow(completion)
    assert (semantic.source, semantic.destination) == (0, 8)
    assert replace(semantic, source=completion.source, destination=completion.destination) == completion
    if variant == "node-local":
        assert projected is graph
        assert render_serial_execution_graph_goal(projected).render() == (
            render_serial_execution_graph_goal(graph).render()
        )


@pytest.mark.parametrize("width", [0, 3, True, 2.0, 16])
def test_bad_pipeline_width(width):
    with pytest.raises(ValueError):
        declared_pipeline_placement(width)


def test_bad_placement_and_variant():
    placement = declared_pipeline_placement()
    with pytest.raises(ValueError, match="variant"):
        declared_rail_fabric(placement, variant="unknown")
    with pytest.raises(ValueError, match="64"):
        declared_rail_fabric(declared_manifest(), variant="rail")
    with pytest.raises(ValueError, match="declared"):
        declared_rail_fabric(replace(placement, source="extracted"), variant="rail")
    ranks = [*placement.ranks]
    ranks[0] = replace(ranks[0], hostname="moved")
    with pytest.raises(ValueError, match="eight-rank block"):
        declared_rail_fabric(replace(placement, ranks=ranks), variant="rail")
    ranks[0] = replace(placement.ranks[0], local_rank=7)
    with pytest.raises(ValueError, match="indices"):
        declared_rail_fabric(replace(placement, ranks=ranks), variant="rail")


def test_renderer_refuses_changed_physical_contract():
    fabric = declared_rail_fabric(declared_pipeline_placement(), variant="rail")
    links = (replace(fabric.links[0], link_rate_bps=200_000_000_000), *fabric.links[1:])
    with pytest.raises(ValueError, match="uniform"):
        project_declared_clos(replace(fabric, links=links))
    with pytest.raises(ValueError, match="whole"):
        project_declared_clos(replace(fabric, switch_latency_ps=1))
    with pytest.raises(ValueError, match="eight-leaf"):
        project_declared_clos(disaggregated_manifests(prefill_nodes=1, decode_nodes=1).fabric)


def test_projection_rejects_nonbijective_endpoints():
    with pytest.raises(ValueError, match="bijection"):
        HtsimClosProjection("", (0,) * 64)
    with pytest.raises(ValueError, match="bijection"):
        HtsimClosProjection("", (False, *range(1, 64)))
