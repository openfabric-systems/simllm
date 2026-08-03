import pytest

from simllm.placement import GroupMembership, PlacementManifest, RankMapper, RankPlacement


def two_node_manifest() -> PlacementManifest:
    # DP=2 x PP=2 x TP=4, ranks 0-7 on node-a, 8-15 on node-b; rank 9 as in
    # the worked example: tp=[8..11], pp=[9,13], dp=[1,9], ep=stage-0 flatten.
    ranks = []
    for r in range(16):
        host = "node-a" if r < 8 else "node-b"
        ranks.append(RankPlacement(global_rank=r, hostname=host, local_rank=r % 8))
    ranks[9].groups = {
        "tp": GroupMembership(1, [8, 9, 10, 11]),
        "pp": GroupMembership(0, [9, 13]),
        "dp": GroupMembership(1, [1, 9]),
        "ep": GroupMembership(5, [0, 1, 2, 3, 8, 9, 10, 11]),
    }
    ranks[9].local_expert_ids = {17: [5, 13, 21, 29]}
    return PlacementManifest(ranks=ranks, framework="vllm", framework_version="0.14.0")


def test_manifest_roundtrip(tmp_path):
    m = two_node_manifest()
    path = m.save(tmp_path / "placement.json")
    loaded = PlacementManifest.load(path)
    r9 = loaded.by_rank(9)
    assert r9.hostname == "node-b"
    assert loaded.group_ranks(9, "ep") == [0, 1, 2, 3, 8, 9, 10, 11]
    assert r9.groups["tp"].rank_in_group == 1
    assert r9.local_expert_ids[17] == [5, 13, 21, 29]


def test_mapper_gpu_rank():
    mapper = RankMapper(two_node_manifest())
    assert mapper.goal_rank(9) == 9
    assert mapper.num_goal_ranks() == 16
    assert mapper.is_intra_node(8, 9)          # TP stays on node-b
    assert not mapper.is_intra_node(1, 9)      # DP crosses nodes
    with pytest.raises(KeyError):
        mapper.goal_rank(99)


def test_mapper_rejects_unfinished_mode():
    with pytest.raises(NotImplementedError):
        RankMapper(two_node_manifest(), mode="unique-nic")
