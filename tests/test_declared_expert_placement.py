"""PLACE-3: declared expert-parallel groups and per-layer expert ownership.

Every constant here is read from the frozen registration in
``examples/declared_expert_placement_v1``; nothing is derived from a run.
"""

import hashlib
import json
from dataclasses import fields
from itertools import pairwise
from pathlib import Path

import pytest

from simllm.placement import (
    DeclaredExpertLayout,
    PlacementManifest,
    RankPlacement,
    declared_local_expert_ids,
    declared_manifest,
    declared_pipeline_partition,
    declared_pipeline_placement,
)
from simllm.traffic.routed_moe import ExpertPlacementSnapshot

ROOT = Path(__file__).resolve().parents[1]
STUDY_DIR = ROOT / "examples" / "declared_expert_placement_v1"
EXPECTATIONS_PATH = STUDY_DIR / "expectations.json"
AMENDMENT_PATH = STUDY_DIR / "expectations-amendment-2026-09-13.json"

#: Cell C1: the worked example, 48 layers all carrying an MoE block.
C1_LAYOUT = {
    "num_layers": 48,
    "moe_layers": tuple(range(48)),
    "num_experts": 32,
}
#: Cell C3: a DeepSeek-class pipeline, MoE on layers 3 through 60.
C3_LAYOUT = {
    "num_layers": 61,
    "moe_layers": tuple(range(3, 61)),
    "num_experts": 256,
}
#: Cell C8, from the 2026-09-13 amendment: 30 experts over eight EP ranks.
C8_LAYOUT = {
    "num_layers": 24,
    "moe_layers": tuple(range(24)),
    "num_experts": 30,
}
C8_COUNTS = [4, 4, 4, 4, 4, 4, 3, 3]
C8_LINEAR_OWNERS = {
    0: [0, 1, 2, 3],
    1: [4, 5, 6, 7],
    2: [8, 9, 10, 11],
    3: [12, 13, 14, 15],
    4: [16, 17, 18, 19],
    5: [20, 21, 22, 23],
    6: [24, 25, 26],
    7: [27, 28, 29],
}
C8_ROUND_ROBIN_OWNERS = {
    0: [0, 8, 16, 24],
    5: [5, 13, 21, 29],
    6: [6, 14, 22],
    7: [7, 15, 23],
}


def _expectations() -> dict:
    return json.loads(EXPECTATIONS_PATH.read_text(encoding="utf-8"))


def _amendment() -> dict:
    return json.loads(AMENDMENT_PATH.read_text(encoding="utf-8"))


def c1_manifest(strategy: str = "linear") -> PlacementManifest:
    return declared_manifest(
        tp=4,
        pp=2,
        dp=2,
        experts=DeclaredExpertLayout(**C1_LAYOUT, placement_strategy=strategy),
    )


def c3_manifest() -> PlacementManifest:
    return declared_manifest(
        tp=8,
        pp=4,
        dp=2,
        nodes=8,
        gpus_per_node=8,
        experts=DeclaredExpertLayout(**C3_LAYOUT),
    )


def c8_manifest(strategy: str = "linear") -> PlacementManifest:
    return declared_manifest(
        tp=1,
        dp=8,
        experts=DeclaredExpertLayout(**C8_LAYOUT, placement_strategy=strategy),
    )


# Cell C1: the worked example


@pytest.mark.parametrize(
    ("strategy", "rank_9_experts", "rank_15_experts"),
    [
        ("linear", [20, 21, 22, 23], [28, 29, 30, 31]),
        ("round_robin", [5, 13, 21, 29], [7, 15, 23, 31]),
    ],
)
def test_c1_worked_example_ep_group_range_and_ownership(
    strategy, rank_9_experts, rank_15_experts
):
    manifest = c1_manifest(strategy)

    assert len(manifest.ranks) == 16
    # rank 9 = (dp=1, pp=0, tp=1), so ep_rank = dp * TP + tp = 5.
    rank_9 = manifest.by_rank(9)
    assert manifest.group_ranks(9, "ep") == [0, 1, 2, 3, 8, 9, 10, 11]
    assert rank_9.groups["ep"].rank_in_group == 5
    assert rank_9.pipeline_layer_range == (0, 24)
    assert sorted(rank_9.local_expert_ids) == list(range(24))
    assert all(ids == rank_9_experts for ids in rank_9.local_expert_ids.values())

    # rank 15 = (dp=1, pp=1, tp=3), so ep_rank = 7 on the second stage.
    rank_15 = manifest.by_rank(15)
    assert manifest.group_ranks(15, "ep") == [4, 5, 6, 7, 12, 13, 14, 15]
    assert rank_15.groups["ep"].rank_in_group == 7
    assert rank_15.pipeline_layer_range == (24, 48)
    assert sorted(rank_15.local_expert_ids) == list(range(24, 48))
    assert all(ids == rank_15_experts for ids in rank_15.local_expert_ids.values())


@pytest.mark.parametrize("strategy", ["linear", "round_robin"])
def test_c1_eight_owners_partition_every_moe_layer(strategy):
    manifest = c1_manifest(strategy)

    for stage_ranks, stage_range in (
        ([0, 1, 2, 3, 8, 9, 10, 11], range(24)),
        ([4, 5, 6, 7, 12, 13, 14, 15], range(24, 48)),
    ):
        assert len(stage_ranks) == 8
        for layer in stage_range:
            owned = [
                expert
                for rank in stage_ranks
                for expert in manifest.by_rank(rank).local_expert_ids[layer]
            ]
            assert sorted(owned) == list(range(32))
        # a stage owns no layer of the other stage
        for rank in stage_ranks:
            assert set(manifest.by_rank(rank).local_expert_ids) == set(stage_range)


def test_c1_places_ep_after_tp_pp_and_dp_and_stamps_the_epoch():
    manifest = declared_manifest(
        tp=4,
        pp=2,
        dp=2,
        experts=DeclaredExpertLayout(**C1_LAYOUT, placement_epoch=7),
    )

    for rank in manifest.ranks:
        assert list(rank.groups) == ["tp", "pp", "dp", "ep"]
        assert rank.placement_epoch == 7


# Cell C3: a DeepSeek-class pipeline


def test_c3_deepseek_class_groups_stages_and_ownership():
    manifest = c3_manifest()

    assert len(manifest.ranks) == 64
    ep_groups = {tuple(rank.groups["ep"].global_ranks) for rank in manifest.ranks}
    assert len(ep_groups) == 4
    for pipeline_index in range(4):
        expected_group = [
            (dp * 4 + pipeline_index) * 8 + tp for dp in range(2) for tp in range(8)
        ]
        assert tuple(expected_group) in ep_groups
        for dp in range(2):
            for tp in range(8):
                rank = manifest.by_rank((dp * 4 + pipeline_index) * 8 + tp)
                assert rank.groups["ep"].global_ranks == expected_group
                assert rank.groups["ep"].rank_in_group == dp * 8 + tp

    stage_ranges = [(0, 15), (15, 30), (30, 46), (46, 61)]
    moe_layers_per_stage = [12, 15, 16, 15]
    for pipeline_index, (stage_range, moe_count) in enumerate(
        zip(stage_ranges, moe_layers_per_stage)
    ):
        for dp in range(2):
            for tp in range(8):
                rank = manifest.by_rank((dp * 4 + pipeline_index) * 8 + tp)
                assert rank.pipeline_layer_range == stage_range
                assert len(rank.local_expert_ids) == moe_count
                ep_rank = dp * 8 + tp
                expected = list(range(16 * ep_rank, 16 * ep_rank + 16))
                assert all(
                    ids == expected for ids in rank.local_expert_ids.values()
                )


def test_c3_ownership_entries_are_conserved_over_the_whole_world():
    manifest = c3_manifest()

    entries = [
        (layer, expert, rank.global_rank)
        for rank in manifest.ranks
        for layer, experts in rank.local_expert_ids.items()
        for expert in experts
    ]

    assert len(entries) == 58 * 256 == 14_848
    keys = [(layer, expert) for layer, expert, _rank in entries]
    assert len(set(keys)) == len(keys)
    assert set(keys) == {
        (layer, expert) for layer in range(3, 61) for expert in range(256)
    }


# Cell C4: partition arithmetic


@pytest.mark.parametrize(
    ("num_layers", "pp", "intervals"),
    [
        (48, 2, ((0, 24), (24, 48))),
        (61, 4, ((0, 15), (15, 30), (30, 46), (46, 61))),
        (
            61,
            8,
            (
                (0, 7),
                (7, 14),
                (14, 22),
                (22, 30),
                (30, 38),
                (38, 46),
                (46, 54),
                (54, 61),
            ),
        ),
        (30, 4, ((0, 7), (7, 15), (15, 23), (23, 30))),
        (5, 2, ((0, 3), (3, 5))),
    ],
)
def test_c4_pipeline_partition_rows(num_layers, pp, intervals):
    assert declared_pipeline_partition(num_layers, pp) == intervals
    # the partition covers the layer axis exactly once, with no gap
    assert intervals[0][0] == 0
    assert intervals[-1][1] == num_layers
    assert all(left[1] == right[0] for left, right in pairwise(intervals))


# Cell C5: refusals


def test_c5_refuses_a_moe_layer_outside_the_model():
    with pytest.raises(ValueError, match="moe_layers"):
        DeclaredExpertLayout(
            num_layers=48, moe_layers=tuple(range(49)), num_experts=32
        )


def test_c5_refuses_a_zero_layer_pipeline_stage():
    layout = DeclaredExpertLayout(num_layers=1, moe_layers=(0,), num_experts=32)
    with pytest.raises(ValueError, match="num_layers"):
        declared_manifest(tp=1, pp=2, dp=32, experts=layout)


def test_c5_refuses_an_unknown_placement_strategy():
    with pytest.raises(ValueError, match="placement_strategy"):
        DeclaredExpertLayout(
            num_layers=48,
            moe_layers=tuple(range(48)),
            num_experts=32,
            placement_strategy="hash",
        )


def test_c5_refuses_duplicate_or_unsorted_moe_layers():
    with pytest.raises(ValueError, match="moe_layers"):
        DeclaredExpertLayout(num_layers=48, moe_layers=(0, 1, 1), num_experts=32)
    with pytest.raises(ValueError, match="moe_layers"):
        DeclaredExpertLayout(num_layers=48, moe_layers=(2, 1), num_experts=32)


def test_c5_refuses_an_empty_moe_layer_list():
    with pytest.raises(ValueError, match="moe_layers"):
        DeclaredExpertLayout(num_layers=48, moe_layers=(), num_experts=32)


def test_c5_refuses_a_negative_placement_epoch():
    with pytest.raises(ValueError, match="placement_epoch"):
        DeclaredExpertLayout(
            num_layers=48,
            moe_layers=tuple(range(48)),
            num_experts=32,
            placement_epoch=-1,
        )


def test_c5_refuses_a_boolean_expert_count():
    with pytest.raises(ValueError, match="num_experts"):
        DeclaredExpertLayout(
            num_layers=48, moe_layers=tuple(range(48)), num_experts=True
        )


@pytest.mark.parametrize("num_experts", [0, -1])
def test_refuses_an_expert_count_below_one(num_experts):
    with pytest.raises(ValueError, match="num_experts"):
        DeclaredExpertLayout(
            num_layers=48, moe_layers=tuple(range(48)), num_experts=num_experts
        )
    with pytest.raises(ValueError, match="num_experts"):
        declared_local_expert_ids(num_experts, 8, 0)


def test_refuses_a_negative_moe_layer():
    with pytest.raises(ValueError, match="moe_layers"):
        DeclaredExpertLayout(num_layers=48, moe_layers=(-1, 0), num_experts=32)


# Cell C6: wire identity


@pytest.mark.parametrize("build", [c1_manifest, c3_manifest])
def test_c6_round_trip_preserves_the_declared_layout(tmp_path, build):
    manifest = build()
    path = manifest.save(tmp_path / "declared.json")

    loaded = PlacementManifest.load(path)

    assert loaded == manifest
    assert loaded.save(tmp_path / "again.json").read_bytes() == path.read_bytes()


@pytest.mark.parametrize("build", [c1_manifest, c3_manifest])
def test_c6_serializes_ep_after_dp_with_ascending_expert_layers(tmp_path, build):
    manifest = build()
    path = manifest.save(tmp_path / "declared.json")

    raw = json.loads(path.read_text(encoding="utf-8"))

    for rank in raw["ranks"]:
        assert list(rank["groups"]) == ["tp", "pp", "dp", "ep"]
        layers = [int(layer) for layer in rank["local_expert_ids"]]
        assert layers == sorted(layers)
        assert rank["pipeline_layer_range"] == list(
            manifest.by_rank(rank["global_rank"]).pipeline_layer_range
        )


# Fatal, unscored compatibility digests: the option-absent path is byte locked


def _reference_manifests() -> dict:
    return {
        "worked_example_tp4_pp2_dp2": lambda: declared_manifest(tp=4, pp=2, dp=2),
        "m4_tp8": lambda: declared_manifest(tp=8),
        "rail_pp8": lambda: declared_pipeline_placement(8),
        "m5_tp1_dp8": lambda: declared_manifest(tp=1, dp=8),
        "width_tp64": lambda: declared_manifest(tp=64, nodes=8, gpus_per_node=8),
    }


@pytest.mark.parametrize("record", sorted(_reference_manifests()))
def test_expert_free_manifests_keep_their_frozen_bytes_and_digest(tmp_path, record):
    frozen = _expectations()["baseline"]["placement_records"][record]

    path = _reference_manifests()[record]().save(tmp_path / f"{record}.json")
    payload = path.read_bytes()

    assert len(payload) == frozen["bytes"]
    assert hashlib.sha256(payload).hexdigest() == frozen["sha256"]


def test_declared_manifest_without_experts_adds_no_expert_field():
    manifest = declared_manifest(tp=1, dp=8)

    for rank in manifest.ranks:
        assert list(rank.groups) == ["tp", "pp", "dp"]
        assert rank.pipeline_layer_range is None
        assert rank.local_expert_ids == {}
        assert rank.placement_epoch == 0


def _expert_twins() -> dict:
    return {
        "c1": (
            lambda: declared_manifest(
                tp=4,
                pp=2,
                dp=2,
                experts=DeclaredExpertLayout(**C1_LAYOUT, placement_epoch=7),
            ),
            lambda: declared_manifest(tp=4, pp=2, dp=2),
        ),
        "c3": (
            c3_manifest,
            lambda: declared_manifest(tp=8, pp=4, dp=2, nodes=8, gpus_per_node=8),
        ),
        "c8": (c8_manifest, lambda: declared_manifest(tp=1, dp=8)),
    }


#: The only rank fields an expert layout may change, besides ``groups["ep"]``.
EXPERT_RANK_FIELDS = ("pipeline_layer_range", "local_expert_ids", "placement_epoch")


@pytest.mark.parametrize("cell", sorted(_expert_twins()))
def test_expert_manifest_differs_from_its_expert_free_twin_only_in_expert_fields(
    cell,
):
    build_with, build_without = _expert_twins()[cell]
    with_experts, without = build_with(), build_without()

    for manifest_field in fields(PlacementManifest):
        if manifest_field.name != "ranks":
            assert getattr(with_experts, manifest_field.name) == getattr(
                without, manifest_field.name
            )
    for expert_rank, free_rank in zip(with_experts.ranks, without.ranks, strict=True):
        for rank_field in fields(RankPlacement):
            name = rank_field.name
            if name == "groups":
                groups = dict(expert_rank.groups)
                assert "ep" in groups and "ep" not in free_rank.groups
                del groups["ep"]
                assert list(groups) == list(free_rank.groups)
                assert groups == free_rank.groups
            elif name not in EXPERT_RANK_FIELDS:
                assert getattr(expert_rank, name) == getattr(free_rank, name)
        assert free_rank.pipeline_layer_range is None
        assert free_rank.local_expert_ids == {}
        assert free_rank.placement_epoch == 0
        assert expert_rank.pipeline_layer_range is not None
        assert expert_rank.local_expert_ids


# Cell C2 conservation: the snapshot projection needs no backend


@pytest.mark.parametrize("world", [2, 4, 8])
def test_c2_snapshot_conserves_every_layer_expert_pair(world):
    manifest = declared_manifest(
        tp=1,
        dp=world,
        experts=DeclaredExpertLayout(
            num_layers=24, moe_layers=tuple(range(24)), num_experts=32
        ),
    )
    ep_ranks = manifest.group_ranks(0, "ep")
    assert ep_ranks == list(range(world))

    snapshot = ExpertPlacementSnapshot.from_manifest(manifest, ep_ranks)
    owners = snapshot.owner_map()

    assert snapshot.placement_epoch == 0
    assert len(snapshot.expert_owners) == 24 * 32 == 768
    assert len(owners) == 768
    experts_per_rank = 32 // world
    assert owners == {
        (layer, expert): expert // experts_per_rank
        for layer in range(24)
        for expert in range(32)
    }


# Cell C8: the remainder layout of the 2026-09-13 amendment


@pytest.mark.parametrize(
    ("strategy", "owners"),
    [("linear", C8_LINEAR_OWNERS), ("round_robin", C8_ROUND_ROBIN_OWNERS)],
)
def test_c8_remainder_layout_counts_and_named_owners(strategy, owners):
    manifest = c8_manifest(strategy)

    assert len(manifest.ranks) == 8
    assert manifest.group_ranks(0, "ep") == list(range(8))
    for rank in manifest.ranks:
        assert rank.pipeline_layer_range == (0, 24)
        assert sorted(rank.local_expert_ids) == list(range(24))
        rows = {tuple(ids) for ids in rank.local_expert_ids.values()}
        assert len(rows) == 1
        assert len(next(iter(rows))) == C8_COUNTS[rank.global_rank]
    for global_rank, expected in owners.items():
        placement = manifest.by_rank(global_rank)
        assert all(ids == expected for ids in placement.local_expert_ids.values())
        assert list(declared_local_expert_ids(30, 8, global_rank, strategy)) == expected


@pytest.mark.parametrize("strategy", ["linear", "round_robin"])
def test_c8_every_expert_is_owned_once_per_layer(strategy):
    manifest = c8_manifest(strategy)

    for layer in range(24):
        owned = [
            expert for rank in manifest.ranks for expert in rank.local_expert_ids[layer]
        ]
        assert sorted(owned) == list(range(30))


@pytest.mark.parametrize("strategy", ["linear", "round_robin"])
def test_c8_snapshot_carries_720_owner_entries(strategy):
    manifest = c8_manifest(strategy)
    ep_ranks = manifest.group_ranks(0, "ep")

    snapshot = ExpertPlacementSnapshot.from_manifest(manifest, ep_ranks)
    owners = snapshot.owner_map()

    assert len(snapshot.expert_owners) == 24 * 30 == 720
    assert set(owners) == {(layer, expert) for layer in range(24) for expert in range(30)}
    assert all(
        expert in manifest.by_rank(rank).local_expert_ids[layer]
        for (layer, expert), rank in owners.items()
    )


@pytest.mark.parametrize("strategy", ["linear", "round_robin"])
def test_remainder_rule_partitions_every_small_expert_count_and_ep_size(strategy):
    for ep_size in range(1, 10):
        for num_experts in range(1, 41):
            if strategy == "round_robin" and num_experts < ep_size - 1:
                # the framework's round-robin map fails for ranks above the expert count
                for ep_rank in range(ep_size):
                    with pytest.raises(ValueError, match="num_experts"):
                        declared_local_expert_ids(num_experts, ep_size, ep_rank, strategy)
                continue
            base, remainder = divmod(num_experts, ep_size)
            owned = [
                declared_local_expert_ids(num_experts, ep_size, ep_rank, strategy)
                for ep_rank in range(ep_size)
            ]
            assert [len(ids) for ids in owned] == [
                base + 1 if ep_rank < remainder else base for ep_rank in range(ep_size)
            ]
            assert sorted(expert for ids in owned for expert in ids) == list(
                range(num_experts)
            )
            assert all(list(ids) == sorted(ids) for ids in owned)


def test_round_robin_refuses_one_expert_over_three_ranks_and_linear_leaves_rank_2_empty():
    layout = {"num_layers": 2, "moe_layers": (0, 1), "num_experts": 1}
    with pytest.raises(ValueError, match="num_experts"):
        declared_local_expert_ids(1, 3, 0, "round_robin")
    with pytest.raises(ValueError, match="num_experts"):
        declared_manifest(
            tp=1,
            dp=3,
            experts=DeclaredExpertLayout(**layout, placement_strategy="round_robin"),
        )

    manifest = declared_manifest(tp=1, dp=3, experts=DeclaredExpertLayout(**layout))

    assert declared_local_expert_ids(1, 3, 0) == (0,)
    assert declared_local_expert_ids(1, 3, 2) == ()
    assert manifest.by_rank(2).local_expert_ids == {0: [], 1: []}
    snapshot = ExpertPlacementSnapshot.from_manifest(manifest, manifest.group_ranks(0, "ep"))
    assert snapshot.owner_map() == {(0, 0): 0, (1, 0): 0}


def test_round_robin_accepts_a_rank_at_the_expert_count_owning_nothing():
    # vLLM builds arange(2, 2, 3) as an empty map for rank 2, so this layout runs
    owned = [declared_local_expert_ids(2, 3, rank, "round_robin") for rank in range(3)]

    assert owned == [(0,), (1,), ()]


# The frozen registration itself


def test_freeze_pins_the_schema_and_framework_semantics():
    frozen = _expectations()

    assert frozen["schema"] == "simllm-declared-expert-placement-expectations-v1"
    assert frozen["task"] == "PLACE-3"
    assert frozen["framework"] == {
        "name": "vllm",
        "version": "0.27.1",
        "rank_formula": "(dp * PP + pp) * TP + tp",
        "ep_group": (
            "per pipeline index: (dp * PP + pp) * TP + tp for dp in range(DP) "
            "for tp in range(TP)"
        ),
        "ep_rank_in_group": "dp * TP + tp",
        "ep_size": "DP * TP",
        "expert_count_rule": "num_experts % (DP * TP) == 0",
        "placement_strategies": ["linear", "round_robin"],
        "pipeline_partition": (
            "base = L // PP; remainder adds one layer to stages -2, -3, ... in order"
        ),
    }
    assert frozen["interface"]["group_key_order"] == ["tp", "pp", "dp", "ep"]
    assert frozen["evidence"]["scored_exact_oracle_rows"] == 6
    assert frozen["evidence"]["fatal_compatibility_digests"] == 5


def test_freeze_pins_the_c2_makespan_literals():
    cell = _expectations()["cells"]["c2_m5_identity"]

    assert cell["profile"] == "rnic-nn-fluid"
    assert cell["linkspeed_bps"] == 400_000_000_000
    assert cell["worlds"] == [2, 4, 8]
    assert cell["frozen_makespan_ps"] == {
        "decode8x2048": {"2": 563_362_560, "4": 486_963_888, "8": 448_764_528},
        "prefill2048": {
            "2": 17_592_951_360,
            "4": 25_646_015_088,
            "8": 29_672_546_928,
        },
    }
    assert cell["snapshot_owner_entries"] == 768
    assert cell["decode_w8_compute_floor_ps"] == 242_664_000
    # the frozen floor is 24 layers of the frozen 10,111 ns compute gate
    assert cell["decode_w8_compute_floor_ps"] == 24 * 10_111 * 1_000
    assert cell["frozen_makespan_ps"]["decode8x2048"]["8"] > (
        cell["decode_w8_compute_floor_ps"]
    )


def test_freeze_agrees_with_the_m5_record_it_reuses():
    from examples.m5.run_m5 import FROZEN_B_MAKESPAN_PS

    frozen = _expectations()["cells"]["c2_m5_identity"]["frozen_makespan_ps"]

    assert {
        (shape, int(world)): makespan
        for shape, rows in frozen.items()
        for world, makespan in rows.items()
    } == FROZEN_B_MAKESPAN_PS


def test_amendment_pins_the_refuted_rule_and_the_c8_literals():
    amendment = _amendment()

    assert set(amendment) == {"schema", "date", "chronology", "refuted", "c8"}
    assert amendment["schema"] == "simllm-declared-expert-placement-amendment-v1"
    assert amendment["date"] == "2026-09-13"
    assert amendment["chronology"] == (
        "post-specified to a review finding, frozen before the corrected builder "
        "and rerun"
    )
    assert amendment["refuted"] == ["divisibility refusal"]
    assert amendment["c8"] == {
        "tp": 1,
        "pp": 1,
        "dp": 8,
        "num_layers": 24,
        "moe_layers": "0..23",
        "num_experts": 30,
        "counts": C8_COUNTS,
        "linear": {str(rank): ids for rank, ids in C8_LINEAR_OWNERS.items()},
        "round_robin": {str(rank): ids for rank, ids in C8_ROUND_ROBIN_OWNERS.items()},
        "snapshot_owner_entries": 720,
    }
    # the original freeze stays in place, including the rule it refutes
    assert _expectations()["framework"]["expert_count_rule"] == (
        "num_experts % (DP * TP) == 0"
    )
