"""PLACE-13: the declared SGLang rank layout, EP group and expert ownership.

Every constant here is read from the frozen registration in
``examples/sglang_declared_layout_v1``; nothing is derived from a run. The
cells are the freeze's S1 through S10, with S10 carried only as far as the
manifest and the ownership snapshot, because its timed half needs the backend
and lives in the study harness.
"""

import hashlib
import json
from inspect import Parameter, signature
from itertools import pairwise
from pathlib import Path

import pytest

from simllm.placement import (
    DeclaredExpertLayout,
    PlacementManifest,
    declared_manifest,
    declared_pipeline_partition,
    declared_pipeline_placement,
    declared_sglang_manifest,
    declared_sglang_pipeline_partition,
)
from simllm.traffic.routed_moe import ExpertPlacementSnapshot

ROOT = Path(__file__).resolve().parents[1]
STUDY_DIR = ROOT / "examples" / "sglang_declared_layout_v1"
EXPECTATIONS_PATH = STUDY_DIR / "expectations.json"

#: Cells S1, S2 and S3 share one layout: 48 layers all carrying an MoE block.
S1_LAYOUT = {
    "num_layers": 48,
    "moe_layers": tuple(range(48)),
    "num_experts": 32,
}
#: Cell S4: a DeepSeek-class pipeline, MoE on layers 3 through 60.
S4_LAYOUT = {
    "num_layers": 61,
    "moe_layers": tuple(range(3, 61)),
    "num_experts": 64,
}
#: Cell S10: the m5 granite geometry, 24 layers all MoE, 32 global experts.
S10_LAYOUT = {
    "num_layers": 24,
    "moe_layers": tuple(range(24)),
    "num_experts": 32,
}
#: The pinned SGLang commit, passed through as ``framework_version``.
PINNED_COMMIT = "bfeae4e79a8dc4600e006f1a5fbc85321a01c1a3"


def _expectations() -> dict:
    return json.loads(EXPECTATIONS_PATH.read_text(encoding="utf-8"))


def s1_manifest() -> PlacementManifest:
    return declared_sglang_manifest(
        tp=8, pp=1, ep_size=8, experts=DeclaredExpertLayout(**S1_LAYOUT)
    )


def s2_manifest() -> PlacementManifest:
    return declared_sglang_manifest(
        tp=8, pp=1, ep_size=2, experts=DeclaredExpertLayout(**S1_LAYOUT)
    )


def s3_manifest() -> PlacementManifest:
    return declared_sglang_manifest(
        tp=8,
        pp=1,
        ep_size=2,
        moe_dp_size=4,
        experts=DeclaredExpertLayout(**S1_LAYOUT),
    )


def s4_manifest() -> PlacementManifest:
    return declared_sglang_manifest(
        tp=4, pp=4, ep_size=4, experts=DeclaredExpertLayout(**S4_LAYOUT)
    )


def s10_manifest(world: int) -> PlacementManifest:
    return declared_sglang_manifest(
        tp=world, pp=1, ep_size=world, experts=DeclaredExpertLayout(**S10_LAYOUT)
    )


# Cell S1: the one geometry where both frameworks agree


def test_s1_coincidence_named_literals():
    manifest = s1_manifest()

    assert len(manifest.ranks) == 8
    assert manifest.framework == "sglang"
    for rank in manifest.ranks:
        assert rank.groups["ep"].global_ranks == [0, 1, 2, 3, 4, 5, 6, 7]
        assert rank.groups["ep"].rank_in_group == rank.global_rank
        assert rank.groups["dp"].global_ranks == [rank.global_rank]
        assert rank.groups["dp"].rank_in_group == 0

    rank_5 = manifest.by_rank(5)
    assert rank_5.pipeline_layer_range == (0, 48)
    assert sorted(rank_5.local_expert_ids) == list(range(48))
    assert all(ids == [20, 21, 22, 23] for ids in rank_5.local_expert_ids.values())
    assert rank_5.groups["moe_tp"].global_ranks == [5]
    assert rank_5.groups["moe_dp"].global_ranks == [5]


def test_s1_equals_the_vllm_twin_outside_framework_and_the_two_moe_groups():
    frozen = _expectations()["cells"]["s1_coincidence"]
    sglang = s1_manifest()
    vllm = declared_manifest(tp=8, experts=DeclaredExpertLayout(**S1_LAYOUT))

    for declared, twin in zip(sglang.ranks, vllm.ranks, strict=True):
        assert declared.global_rank == twin.global_rank
        assert declared.hostname == twin.hostname
        assert declared.local_rank == twin.local_rank
        for group in ("tp", "pp", "dp", "ep"):
            assert declared.groups[group] == twin.groups[group]
        assert declared.pipeline_layer_range == twin.pipeline_layer_range
        assert declared.local_expert_ids == twin.local_expert_ids
        assert declared.placement_epoch == twin.placement_epoch
    assert frozen["fields_differing_from_vllm_twin"] == ["framework", "moe_tp", "moe_dp"]
    assert sglang.framework == "sglang"
    assert vllm.framework is None
    assert set(sglang.by_rank(0).groups) - set(vllm.by_rank(0).groups) == {
        "moe_tp",
        "moe_dp",
    }


# Cell S2: expert sharding, ep_size 2 over a tensor group of eight


def test_s2_expert_sharding_named_ranks():
    manifest = s2_manifest()

    rank_5 = manifest.by_rank(5)
    assert rank_5.groups["ep"].global_ranks == [1, 5]
    assert rank_5.groups["ep"].rank_in_group == 1
    assert rank_5.groups["moe_tp"].global_ranks == [4, 5, 6, 7]
    assert rank_5.groups["moe_tp"].rank_in_group == 1
    assert rank_5.groups["moe_dp"].global_ranks == [5]
    assert all(
        ids == list(range(16, 32)) for ids in rank_5.local_expert_ids.values()
    )

    rank_0 = manifest.by_rank(0)
    assert rank_0.groups["ep"].global_ranks == [0, 4]
    assert rank_0.groups["ep"].rank_in_group == 0
    assert all(ids == list(range(16)) for ids in rank_0.local_expert_ids.values())


def test_s2_groups_partition_the_experts_and_share_them_across_moe_tp():
    manifest = s2_manifest()

    groups = sorted({tuple(rank.groups["ep"].global_ranks) for rank in manifest.ranks})
    assert groups == [(0, 4), (1, 5), (2, 6), (3, 7)]
    for group in groups:
        for layer in range(48):
            owned = [
                expert
                for rank in group
                for expert in manifest.by_rank(rank).local_expert_ids[layer]
            ]
            assert sorted(owned) == list(range(32))
    for block in ((0, 1, 2, 3), (4, 5, 6, 7)):
        sets = {
            tuple(manifest.by_rank(rank).local_expert_ids[0]) for rank in block
        }
        assert len(sets) == 1
        assert {
            tuple(manifest.by_rank(rank).groups["moe_tp"].global_ranks)
            for rank in block
        } == {block}


# Cell S3: MoE data parallelism, moe_tp collapses to one


def test_s3_moe_data_parallel_named_ranks_and_groups():
    manifest = s3_manifest()

    assert sorted(
        {tuple(rank.groups["ep"].global_ranks) for rank in manifest.ranks}
    ) == [(0, 1), (2, 3), (4, 5), (6, 7)]

    rank_5 = manifest.by_rank(5)
    assert rank_5.groups["ep"].global_ranks == [4, 5]
    assert rank_5.groups["ep"].rank_in_group == 1
    assert rank_5.groups["moe_tp"].global_ranks == [5]
    assert rank_5.groups["moe_dp"].global_ranks == [1, 3, 5, 7]
    assert rank_5.groups["moe_dp"].rank_in_group == 2
    assert all(ids == list(range(16, 32)) for ids in rank_5.local_expert_ids.values())

    rank_6 = manifest.by_rank(6)
    assert rank_6.groups["ep"].global_ranks == [6, 7]
    assert rank_6.groups["ep"].rank_in_group == 0
    assert rank_6.groups["moe_dp"].global_ranks == [0, 2, 4, 6]
    assert rank_6.groups["moe_dp"].rank_in_group == 3
    assert all(ids == list(range(16)) for ids in rank_6.local_expert_ids.values())


# Cell S4: a four-stage pipeline over sixteen ranks


def test_s4_pipeline_stages_groups_and_ownership():
    manifest = s4_manifest()

    assert len(manifest.ranks) == 16
    stage_ranges = [(0, 15), (15, 30), (30, 45), (45, 61)]
    moe_layers_per_stage = [12, 15, 15, 16]
    for stage, (interval, moe_count) in enumerate(
        zip(stage_ranges, moe_layers_per_stage, strict=True)
    ):
        for tp_index in range(4):
            rank = manifest.by_rank(stage * 4 + tp_index)
            assert rank.pipeline_layer_range == interval
            assert len(rank.local_expert_ids) == moe_count
            assert rank.groups["ep"].global_ranks == [
                stage * 4 + t for t in range(4)
            ]
            assert rank.groups["ep"].rank_in_group == tp_index

    rank_13 = manifest.by_rank(13)
    assert rank_13.groups["pp"].rank_in_group == 3
    assert rank_13.groups["tp"].rank_in_group == 1
    assert rank_13.groups["ep"].global_ranks == [12, 13, 14, 15]
    assert rank_13.groups["ep"].rank_in_group == 1
    assert rank_13.pipeline_layer_range == (45, 61)
    assert sorted(rank_13.local_expert_ids) == list(range(45, 61))
    assert all(
        ids == list(range(16, 32)) for ids in rank_13.local_expert_ids.values()
    )


def test_s4_ownership_entries_are_conserved_over_the_whole_world():
    manifest = s4_manifest()

    entries = [
        (layer, expert, rank.global_rank)
        for rank in manifest.ranks
        for layer, experts in rank.local_expert_ids.items()
        for expert in experts
    ]

    assert len(entries) == 58 * 64 == 3_712
    keys = [(layer, expert) for layer, expert, _rank in entries]
    assert len(set(keys)) == len(keys)
    assert set(keys) == {
        (layer, expert) for layer in range(3, 61) for expert in range(64)
    }


# Cell S5: partition arithmetic (the harness runs the framework oracle)


#: The frozen `(num_layers, pp)` rows, keyed as the freeze spells them.
S5_KEYS = tuple(
    key for key in _expectations()["cells"]["s5_partitions"] if "/" in key
)


def test_s5_carries_every_frozen_partition_row():
    assert len(S5_KEYS) == 6


@pytest.mark.parametrize("key", S5_KEYS)
def test_s5_pipeline_partition_rows(key):
    frozen = _expectations()["cells"]["s5_partitions"][key]
    num_layers, pp = (int(part) for part in key.split("/"))

    intervals = declared_sglang_pipeline_partition(num_layers, pp)

    assert [list(interval) for interval in intervals] == frozen
    # the partition covers the layer axis exactly once, with no gap
    assert intervals[0][0] == 0
    assert intervals[-1][1] == num_layers
    assert all(left[1] == right[0] for left, right in pairwise(intervals))


def test_s5_hands_the_remainder_to_the_last_stages():
    for num_layers, pp in ((61, 4), (5, 2), (30, 4)):
        intervals = declared_sglang_pipeline_partition(num_layers, pp)
        base, remainder = divmod(num_layers, pp)
        sizes = [end - start for start, end in intervals]
        assert sizes == [base] * (pp - remainder) + [base + 1] * remainder


# Cell S6: refusals, each raised before any rank is built


def _refusal_cases() -> dict:
    valid_layout = dict(S1_LAYOUT)
    return {
        "round_robin under the sglang builder": lambda: declared_sglang_manifest(
            tp=8,
            ep_size=8,
            experts=DeclaredExpertLayout(
                **valid_layout, placement_strategy="round_robin"
            ),
        ),
        "num_experts 30 at ep_size 8": lambda: declared_sglang_manifest(
            tp=8,
            ep_size=8,
            experts=DeclaredExpertLayout(
                num_layers=48, moe_layers=tuple(range(48)), num_experts=30
            ),
        ),
        "tp 8 ep_size 3": lambda: declared_sglang_manifest(tp=8, ep_size=3),
        "tp 8 ep_size 16": lambda: declared_sglang_manifest(tp=8, ep_size=16),
        "tp 8 ep_size 2 moe_dp_size 2": lambda: declared_sglang_manifest(
            tp=8, ep_size=2, moe_dp_size=2
        ),
        "tp 8 moe_dp_size 2 pp 2": lambda: declared_sglang_manifest(
            tp=8, moe_dp_size=2, pp=2
        ),
        "ep_size 0": lambda: declared_sglang_manifest(tp=8, ep_size=0),
        "moe_dp_size 0": lambda: declared_sglang_manifest(tp=8, moe_dp_size=0),
        "num_layers 1 at pp 2": lambda: declared_sglang_manifest(
            tp=1,
            pp=2,
            experts=DeclaredExpertLayout(
                num_layers=1, moe_layers=(0,), num_experts=1
            ),
        ),
        "tp 4 pp 3 nodes 2": lambda: declared_sglang_manifest(tp=4, pp=3, nodes=2),
        "tp 4 pp 1 nodes 3": lambda: declared_sglang_manifest(tp=4, pp=1, nodes=3),
        "tp 2 pp 1 nodes 1 gpus_per_node 1": lambda: declared_sglang_manifest(
            tp=2, pp=1, nodes=1, gpus_per_node=1
        ),
    }


def test_s6_covers_exactly_the_frozen_rejection_control_family():
    assert sorted(_refusal_cases()) == sorted(
        _expectations()["cells"]["s6_refusals"]
    )


@pytest.mark.parametrize("case", sorted(_refusal_cases()))
def test_s6_refuses_each_frozen_layout(case):
    with pytest.raises(ValueError):
        _refusal_cases()[case]()


@pytest.mark.parametrize(
    "field", ["tp", "pp", "ep_size", "moe_dp_size", "nodes", "gpus_per_node"]
)
def test_s6_refuses_a_boolean_width(field):
    with pytest.raises(ValueError, match=field):
        declared_sglang_manifest(**{field: True})


# Cell S7: wire identity


@pytest.mark.parametrize("build", [s2_manifest, s4_manifest])
def test_s7_round_trip_preserves_the_declared_layout(tmp_path, build):
    manifest = build()
    path = manifest.save(tmp_path / "declared.json")

    loaded = PlacementManifest.load(path)

    assert loaded == manifest
    assert loaded.save(tmp_path / "again.json").read_bytes() == path.read_bytes()


@pytest.mark.parametrize("build", [s2_manifest, s4_manifest])
def test_s7_serializes_the_six_groups_in_order_with_ascending_layers(tmp_path, build):
    frozen = _expectations()["interface"]["group_key_order"]
    manifest = build()

    path = manifest.save(tmp_path / "declared.json")
    raw = json.loads(path.read_text(encoding="utf-8"))

    assert raw["framework"] == "sglang"
    assert raw["framework_version"] is None
    assert raw["source"] == "declared"
    for rank in raw["ranks"]:
        assert list(rank["groups"]) == frozen
        layers = [int(layer) for layer in rank["local_expert_ids"]]
        assert layers == sorted(layers)


def test_s7_passes_the_pinned_commit_through_as_the_framework_version():
    frozen = _expectations()["cells"]["s7_round_trip"]

    manifest = declared_sglang_manifest(
        tp=8,
        ep_size=2,
        framework_version=PINNED_COMMIT,
        experts=DeclaredExpertLayout(**S1_LAYOUT),
    )

    assert frozen["framework_version_passthrough"] == PINNED_COMMIT
    assert frozen["framework_version_default"] is None
    assert manifest.framework_version == PINNED_COMMIT
    assert s2_manifest().framework_version is None


# Cell S8: node placement


def test_s8_node_placement_rows():
    for tp, pp in ((8, 2), (16, 1)):
        manifest = declared_sglang_manifest(tp=tp, pp=pp, nodes=2)
        for rank in manifest.ranks:
            assert rank.hostname == f"node-{rank.global_rank // 8}"
            assert rank.local_rank == rank.global_rank % 8

    narrow = declared_sglang_manifest(tp=4, pp=1, nodes=2)
    assert [(rank.hostname, rank.local_rank) for rank in narrow.ranks] == [
        ("node-0", 0),
        ("node-0", 1),
        ("node-1", 0),
        ("node-1", 1),
    ]


def test_s8_default_node_count_is_the_fewest_that_fit():
    default = declared_sglang_manifest(tp=8, pp=2)

    assert sorted({rank.hostname for rank in default.ranks}) == ["node-0", "node-1"]
    explicit = declared_sglang_manifest(tp=8, pp=2, nodes=2)
    assert [rank.hostname for rank in default.ranks] == [
        rank.hostname for rank in explicit.ranks
    ]


# Cell S9: the second builder is not the first one relabeled


def test_s9_diverges_from_the_vllm_builder_on_the_partition_and_the_ep_group():
    frozen = _expectations()["cells"]["s9_divergence"]

    vllm_partition = [list(i) for i in declared_pipeline_partition(61, 4)]
    sglang_partition = [list(i) for i in declared_sglang_pipeline_partition(61, 4)]
    assert vllm_partition == frozen["partition_61_4"]["vllm"]
    assert sglang_partition == frozen["partition_61_4"]["sglang"]
    differing = [
        stage
        for stage in range(4)
        if vllm_partition[stage] != sglang_partition[stage]
    ]
    assert differing == frozen["partition_61_4"]["differing_stages"]

    sglang = declared_sglang_manifest(
        tp=8, ep_size=2, experts=DeclaredExpertLayout(**S1_LAYOUT)
    )
    vllm = declared_manifest(tp=8, experts=DeclaredExpertLayout(**S1_LAYOUT))
    assert sglang.group_ranks(5, "ep") == frozen["ep_group_rank_5_tp8_ep2"]["sglang"]
    assert vllm.group_ranks(5, "ep") == frozen["ep_group_rank_5_tp8_ep2"]["vllm"]


# Cell S10: the manifest half of the m5 identity, no backend


@pytest.mark.parametrize("world", [2, 4, 8])
def test_s10_ep_group_ownership_and_snapshot_conservation(world):
    frozen = _expectations()["cells"]["s10_m5_identity"]
    manifest = s10_manifest(world)

    ep_ranks = manifest.group_ranks(0, "ep")
    assert ep_ranks == list(range(world))
    experts_per_rank = 32 // world
    for rank in manifest.ranks:
        expected = list(
            range(
                rank.global_rank * experts_per_rank,
                (rank.global_rank + 1) * experts_per_rank,
            )
        )
        assert all(ids == expected for ids in rank.local_expert_ids.values())

    snapshot = ExpertPlacementSnapshot.from_manifest(manifest, ep_ranks)
    owners = snapshot.owner_map()

    assert snapshot.placement_epoch == 0
    assert len(snapshot.expert_owners) == frozen["snapshot_owner_entries"] == 768
    assert owners == {
        (layer, expert): expert // experts_per_rank
        for layer in range(24)
        for expert in range(32)
    }


# Fatal, unscored compatibility digests: the vLLM builder is byte locked


def _reference_manifests() -> dict:
    return {
        "worked_example_tp4_pp2_dp2": lambda: declared_manifest(tp=4, pp=2, dp=2),
        "m4_tp8": lambda: declared_manifest(tp=8),
        "rail_pp8": lambda: declared_pipeline_placement(8),
        "m5_tp1_dp8": lambda: declared_manifest(tp=1, dp=8),
        "width_tp64": lambda: declared_manifest(tp=64, nodes=8, gpus_per_node=8),
    }


@pytest.mark.parametrize("record", sorted(_reference_manifests()))
def test_the_vllm_reference_manifests_keep_their_frozen_bytes_and_digest(
    tmp_path, record
):
    frozen = _expectations()["baseline"]["placement_records"][record]

    path = _reference_manifests()[record]().save(tmp_path / f"{record}.json")
    payload = path.read_bytes()

    assert len(payload) == frozen["bytes"]
    assert hashlib.sha256(payload).hexdigest() == frozen["sha256"]


def test_the_sglang_builder_without_experts_carries_only_the_three_base_groups():
    frozen = _expectations()["interface"]["group_key_order_without_experts"]

    manifest = declared_sglang_manifest(tp=4, pp=2)

    assert manifest.framework == "sglang"
    for rank in manifest.ranks:
        assert list(rank.groups) == frozen == ["tp", "pp", "dp"]
        assert rank.pipeline_layer_range is None
        assert rank.local_expert_ids == {}
        assert rank.placement_epoch == 0


# The frozen registration itself


def test_freeze_pins_the_sglang_framework_semantics_and_interface():
    frozen = _expectations()

    assert frozen["schema"] == "simllm-sglang-declared-layout-expectations-v1"
    assert frozen["task"] == "PLACE-13"
    framework = frozen["framework"]
    assert framework["name"] == "sglang"
    assert framework["commit"] == PINNED_COMMIT
    assert framework["installed_version"] == "0.5.6.post3.dev9406+gbfeae4e79"
    assert framework["version_suffix"] == "gbfeae4e79"
    assert framework["world_size"] == "tp * pp"
    assert framework["rank_formula"] == "tp * pp_rank + tp_rank"
    assert framework["dp_in_rank_space"] is False
    assert framework["placement_strategies"] == ["linear"]
    assert framework["expert_count_rule"] == "num_experts % ep_size == 0"
    interface = frozen["interface"]
    assert interface["builder"] == "declared_sglang_manifest"
    assert interface["partition_function"] == "declared_sglang_pipeline_partition"
    assert interface["framework_field"] == "sglang"
    assert interface["group_key_order"] == [
        "tp",
        "pp",
        "dp",
        "ep",
        "moe_tp",
        "moe_dp",
    ]
    assert frozen["evidence"]["scored"] == ["s10_m5_identity"]
    assert frozen["baseline"]["place3_study_check_reproduces"] is True


def test_freeze_pins_the_s10_makespans_and_agrees_with_the_m5_record():
    from examples.m5.run_m5 import FROZEN_B_MAKESPAN_PS

    cell = _expectations()["cells"]["s10_m5_identity"]

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
    assert {
        (shape, int(world)): makespan
        for shape, rows in cell["frozen_makespan_ps"].items()
        for world, makespan in rows.items()
    } == FROZEN_B_MAKESPAN_PS
    # the frozen floor is 24 layers of the frozen 10,111 ns compute gate
    assert cell["decode_w8_compute_floor_ps"] == 24 * 10_111 * 1_000 == 242_664_000
    assert cell["frozen_makespan_ps"]["decode8x2048"]["8"] > (
        cell["decode_w8_compute_floor_ps"]
    )


def test_freeze_pins_the_builder_keywords_the_implementation_accepts():
    frozen = _expectations()["interface"]["builder_keywords"]

    parameters = signature(declared_sglang_manifest).parameters

    assert list(parameters) == frozen
    assert all(
        parameter.kind is Parameter.KEYWORD_ONLY for parameter in parameters.values()
    )
    assert declared_sglang_manifest(
        tp=2, ep_size=2, hostname_pattern="host-{}"
    ).by_rank(0).hostname == "host-0"
