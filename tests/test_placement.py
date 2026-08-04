import pytest

from simllm.placement import (
    GroupMembership,
    PlacementManifest,
    RankMapper,
    RankPlacement,
    declared_manifest,
)


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


def test_declared_manifest_worked_example():
    # DP=2 x PP=2 x TP=4, global_rank = (dp*PP + pp)*TP + tp: the same
    # 16-rank worked example as two_node_manifest above.
    m = declared_manifest(tp=4, pp=2, dp=2)
    assert m.source == "declared"
    assert len(m.ranks) == 16
    assert [r.global_rank for r in m.ranks] == list(range(16))

    # rank 9 = (dp=1, pp=0, tp=1)
    r9 = m.by_rank(9)
    assert m.group_ranks(9, "tp") == [8, 9, 10, 11]
    assert m.group_ranks(9, "pp") == [9, 13]
    assert m.group_ranks(9, "dp") == [1, 9]
    assert r9.groups["tp"].rank_in_group == 1
    assert r9.groups["pp"].rank_in_group == 0
    assert r9.groups["dp"].rank_in_group == 1

    # rank 0 and the last rank, exact lists
    assert m.group_ranks(0, "tp") == [0, 1, 2, 3]
    assert m.group_ranks(0, "pp") == [0, 4]
    assert m.group_ranks(0, "dp") == [0, 8]
    assert m.group_ranks(15, "tp") == [12, 13, 14, 15]
    assert m.group_ranks(15, "pp") == [11, 15]
    assert m.group_ranks(15, "dp") == [7, 15]

    # default node fill: 8 GPUs per node, ranks 0-7 on node-0, 8-15 on node-1
    assert r9.hostname == "node-1"
    assert r9.local_rank == 1
    assert m.by_rank(7).hostname == "node-0"


def test_declared_manifest_round_trips_and_maps(tmp_path):
    m = declared_manifest(tp=8, pp=1, dp=1, hostname_pattern="host{}")
    loaded = PlacementManifest.load(m.save(tmp_path / "declared.json"))
    assert loaded.source == "declared"
    assert loaded.group_ranks(3, "tp") == list(range(8))
    mapper = RankMapper(loaded)
    assert mapper.num_goal_ranks() == 8
    assert mapper.is_intra_node(0, 7)
    assert loaded.by_rank(0).hostname == "host0"


def test_declared_manifest_rejects_bad_shapes():
    with pytest.raises(ValueError, match="tp"):
        declared_manifest(tp=0)
    with pytest.raises(ValueError, match="does not fit"):
        declared_manifest(tp=8, pp=2, dp=1, nodes=1, gpus_per_node=8)
    # explicit larger node count is fine, world still fills from node 0
    m = declared_manifest(tp=4, nodes=4, gpus_per_node=4)
    assert m.by_rank(3).hostname == "node-0"
