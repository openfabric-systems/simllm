import json

import pytest

from simllm.placement import (
    FabricTopologyManifest,
    PlacementManifest,
    RankMapper,
    disaggregated_manifests,
)


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
    assert manifests.fabric.by_nic("sim-nic-0015").affine_gpu_rank == 15
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
    mapper = RankMapper(manifests.placement)
    assert [mapper.goal_rank(rank) for rank in range(448)] == list(range(448))


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
