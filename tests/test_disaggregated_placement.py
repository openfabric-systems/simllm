import hashlib
import json
from dataclasses import replace

import pytest

from simllm.goal import GoalTrace
from simllm.placement import (
    DECLARED_CLOS_LINK_PROPAGATION_DELAY_PS,
    DECLARED_CLOS_LINK_RATE_BPS,
    FabricTopologyManifest,
    PlacementManifest,
    RankMapper,
    disaggregated_manifests,
)
from simllm.traffic import ordered_pairwise_messages


def test_one_plus_one_uses_the_same_concrete_builder():
    manifests = disaggregated_manifests(
        prefill_nodes=1,
        decode_nodes=1,
        gpus_per_node=8,
        framework="vllm",
        framework_version="0.27.1",
    )

    assert len(manifests.placement.ranks) == 16
    assert len(manifests.fabric.nodes) == 2
    assert [rank.pool_role for rank in manifests.placement.ranks[:8]] == [
        "prefill"
    ] * 8
    assert [rank.pool_role for rank in manifests.placement.ranks[8:]] == [
        "decode"
    ] * 8
    assert manifests.placement.group_ranks(0, "tp") == list(range(8))
    assert manifests.placement.group_ranks(8, "tp") == list(range(8, 16))
    assert manifests.placement.group_ranks(0, "dp") == [0]
    assert manifests.placement.group_ranks(8, "dp") == [8]
    assert manifests.fabric.by_rank(15).nic_id == "sim-nic-0015"
    nic15 = manifests.fabric.by_nic("sim-nic-0015")
    assert nic15.affine_gpu_rank == 15
    assert nic15.switch_id == "leaf-001"
    assert nic15.switch_port_id == "leaf-001/endpoint-07"
    assert nic15.link_id == "endpoint-link-0015"
    assert manifests.fabric.physical_rendering_enabled
    assert len(manifests.fabric.switches) == 10
    assert len(manifests.fabric.links) == 32
    assert sum(len(switch.ports) for switch in manifests.fabric.switches) == 48
    assert RankMapper(manifests.placement).goal_rank(15) == 15


def test_target_builder_renders_448_role_pinned_ranks():
    manifests = disaggregated_manifests(
        prefill_nodes=16,
        decode_nodes=40,
        gpus_per_node=8,
    )
    ranks = manifests.placement.ranks

    assert len(ranks) == 448
    assert len(manifests.fabric.nodes) == 56
    assert [rank.global_rank for rank in ranks] == list(range(448))
    assert {rank.pool_role for rank in ranks[:128]} == {"prefill"}
    assert {rank.pool_role for rank in ranks[128:]} == {"decode"}
    assert manifests.placement.group_ranks(127, "tp") == list(range(120, 128))
    assert manifests.placement.group_ranks(128, "tp") == list(range(128, 136))
    assert manifests.placement.group_ranks(3, "dp") == list(range(3, 128, 8))
    assert manifests.placement.group_ranks(131, "dp") == list(
        range(131, 448, 8)
    )
    assert set(manifests.placement.group_ranks(0, "pool")) == set(range(128))
    assert set(manifests.placement.group_ranks(128, "pool")) == set(
        range(128, 448)
    )
    assert all(len(node.gpus) == 8 for node in manifests.fabric.nodes)
    assert all(len(node.nics) == 8 for node in manifests.fabric.nodes)
    assert all(
        nic.fabric_location
        for node in manifests.fabric.nodes
        for nic in node.nics
    )
    assert manifests.fabric.physical_rendering_enabled
    assert len([switch for switch in manifests.fabric.switches if switch.tier == 0]) == 56
    assert len([switch for switch in manifests.fabric.switches if switch.tier == 1]) == 8
    assert len(manifests.fabric.links) == 896
    assert sum(len(switch.ports) for switch in manifests.fabric.switches) == 1_344
    assert {
        link.link_rate_bps for link in manifests.fabric.links
    } == {DECLARED_CLOS_LINK_RATE_BPS}
    assert {
        link.propagation_delay_ps for link in manifests.fabric.links
    } == {DECLARED_CLOS_LINK_PROPAGATION_DELAY_PS}
    mapper = RankMapper(manifests.placement)
    assert [mapper.goal_rank(rank) for rank in range(448)] == list(range(448))
    for destination in range(1, 448):
        assert manifests.fabric.path_between_ranks(0, destination)


def test_disaggregated_manifests_round_trip(tmp_path):
    manifests = disaggregated_manifests(prefill_nodes=1, decode_nodes=1)
    placement_path = manifests.placement.save(tmp_path / "placement.json")
    fabric_path = manifests.fabric.save(tmp_path / "fabric.json")

    placement = PlacementManifest.load(placement_path)
    fabric = FabricTopologyManifest.load(fabric_path)

    assert placement.by_rank(0).pool_role == "prefill"
    assert placement.by_rank(8).pool_role == "decode"
    assert fabric.by_rank(0).node_id == "prefill-node-0"
    assert fabric.by_rank(8).node_id == "decode-node-0"
    assert json.loads(placement_path.read_text())["schema"] == (
        "simllm-placement-manifest-v1"
    )
    assert json.loads(fabric_path.read_text())["schema"] == (
        "simllm-fabric-topology-v1"
    )
    assert fabric == manifests.fabric
    fabric.validate()


@pytest.mark.parametrize(
    ("prefill_nodes", "decode_nodes", "rank_count"),
    [(1, 1, 16), (16, 40, 448)],
)
def test_declared_topology_resolves_every_rendered_goal_endpoint(
    prefill_nodes, decode_nodes, rank_count
):
    manifests = disaggregated_manifests(
        prefill_nodes=prefill_nodes,
        decode_nodes=decode_nodes,
    )
    trace = GoalTrace(rank_count)
    ordered_pairwise_messages(
        trace,
        ranks=list(range(rank_count)),
        messages=[
            (
                f"endpoint-{source}",
                source,
                (source + 8) % rank_count,
                4_096,
            )
            for source in range(rank_count)
        ],
        tag=5_005,
        operation_id="place-5:goal-reachability",
    )

    paths = manifests.fabric.resolve_goal_paths(trace)

    assert trace.render().startswith(f"num_ranks {rank_count}\n")
    assert len(trace.messages) == rank_count
    assert {message.source_rank for message in trace.messages} == set(
        range(rank_count)
    )
    assert {message.destination_rank for message in trace.messages} == set(
        range(rank_count)
    )
    assert len(paths) == rank_count
    for path in paths:
        assert len(path) == 4
        assert sum(link.propagation_delay_ps for link in path) == 4_000_000
        assert min(link.link_rate_bps for link in path) == 400_000_000_000
        assert "spine-000" in path[1].link_id
        assert "spine-000" in path[2].link_id


@pytest.mark.parametrize(
    ("prefill_nodes", "decode_nodes", "expected_bytes", "expected_sha256"),
    [
        (
            1,
            1,
            15_772,
            "019f818e02252407e560b37415da12151c8f6ca0ff01bcb7ff8aabfead47f286",
        ),
        (
            16,
            40,
            2_639_042,
            "48029d871293762007ab33082d59a7b5a4efb22583394e718c97e733717fd709",
        ),
    ],
)
def test_disabled_physical_rendering_preserves_placement_bytes(
    tmp_path,
    prefill_nodes,
    decode_nodes,
    expected_bytes,
    expected_sha256,
):
    enabled = disaggregated_manifests(
        prefill_nodes=prefill_nodes,
        decode_nodes=decode_nodes,
        render_physical_topology=True,
    )
    disabled = disaggregated_manifests(
        prefill_nodes=prefill_nodes,
        decode_nodes=decode_nodes,
        render_physical_topology=False,
    )
    enabled_path = enabled.placement.save(tmp_path / "enabled.json")
    disabled_path = disabled.placement.save(tmp_path / "disabled.json")
    enabled_bytes = enabled_path.read_bytes()
    disabled_bytes = disabled_path.read_bytes()

    assert enabled_bytes == disabled_bytes
    assert len(disabled_bytes) == expected_bytes
    assert hashlib.sha256(disabled_bytes).hexdigest() == expected_sha256
    assert not disabled.fabric.physical_rendering_enabled
    assert disabled.fabric.switches == ()
    assert disabled.fabric.links == ()
    disabled.fabric.validate()


def test_physical_topology_rejects_a_nonpositive_link_rate():
    manifests = disaggregated_manifests(prefill_nodes=1, decode_nodes=1)
    manifests.fabric.links = (
        replace(manifests.fabric.links[0], link_rate_bps=0),
        *manifests.fabric.links[1:],
    )

    with pytest.raises(ValueError, match="link_rate_bps"):
        manifests.fabric.validate()


def test_ordinary_manifest_omits_the_new_optional_role_byte_for_byte(tmp_path):
    manifest = PlacementManifest.load(
        PlacementManifest(
            ranks=[],
            framework="test",
        ).save(tmp_path / "first.json")
    )
    second = manifest.save(tmp_path / "second.json")

    assert "pool_role" not in second.read_text()


@pytest.mark.parametrize(
    ("name", "value"),
    [("prefill_nodes", 0), ("decode_nodes", 0), ("gpus_per_node", 0)],
)
def test_disaggregated_builder_rejects_empty_pools(name, value):
    kwargs = {"prefill_nodes": 1, "decode_nodes": 1, "gpus_per_node": 8}
    kwargs[name] = value
    with pytest.raises(ValueError, match=name):
        disaggregated_manifests(**kwargs)


def test_fixed_physical_topology_rejects_a_nonreference_node_width():
    with pytest.raises(ValueError, match="exactly eight"):
        disaggregated_manifests(
            prefill_nodes=1,
            decode_nodes=1,
            gpus_per_node=4,
        )

    compatibility = disaggregated_manifests(
        prefill_nodes=1,
        decode_nodes=1,
        gpus_per_node=4,
        render_physical_topology=False,
    )
    assert len(compatibility.placement.ranks) == 8
