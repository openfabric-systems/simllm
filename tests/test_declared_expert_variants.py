"""PLACE-7 and PLACE-11: expert parallelism off, and the expert map exceptions.

Every literal here is read from the frozen registration in
``examples/declared_expert_variants_v1``, including its two amendments of
2026-09-15; nothing is derived from a run. One test carries one frozen cell,
V1 through V13, plus the five compatibility digests that guard the accepted
output.
"""

import hashlib
import json
from pathlib import Path

import pytest

from simllm.placement import (
    DeclaredExpertLayout,
    DeclaredExpertMapExceptions,
    PlacementManifest,
    declared_manifest,
    declared_pipeline_placement,
    declared_resolved_placement_strategy,
    declared_sglang_manifest,
)
from simllm.traffic.routed_moe import ExpertPlacementSnapshot

ROOT = Path(__file__).resolve().parents[1]
STUDY_DIR = ROOT / "examples" / "declared_expert_variants_v1"
EXPECTATIONS_PATH = STUDY_DIR / "expectations.json"
BASELINE_AMENDMENT_PATH = STUDY_DIR / "expectations-amendment-2026-09-15.json"
V13_AMENDMENT_PATH = STUDY_DIR / "expectations-amendment-2026-09-15b.json"

#: Cells V1, V3, V6 and V10: 48 layers, all carrying a routed MoE block.
L48 = {"num_layers": 48, "moe_layers": tuple(range(48)), "num_experts": 32}
#: Cells V4, V7, V8, V11 and V13: the m5 granite geometry.
L24 = {"num_layers": 24, "moe_layers": tuple(range(24)), "num_experts": 32}
#: The remainder layout the PLACE-3 amendment froze, reused as a control.
L30 = {"num_layers": 24, "moe_layers": tuple(range(24)), "num_experts": 30}
#: Cell V5: fewer experts than the round-robin arange can serve.
L4 = {"num_layers": 24, "moe_layers": tuple(range(24)), "num_experts": 4}

#: The four framework conditions that overrule a declared round_robin.
FRAMEWORK_CONDITIONS = (
    "single_expert_group",
    "redundant_experts",
    "eplb",
    "all2all_without_round_robin",
)
#: The two model-scoped conditions, which only ever add a refusal.
MODEL_CONDITIONS = (
    "model_uniform_expert_blocks",
    "model_refuses_tp_above_experts",
)


def _expectations() -> dict:
    return json.loads(EXPECTATIONS_PATH.read_text(encoding="utf-8"))


def _baseline_amendment() -> dict:
    return json.loads(BASELINE_AMENDMENT_PATH.read_text(encoding="utf-8"))


def _v13_amendment() -> dict:
    return json.loads(V13_AMENDMENT_PATH.read_text(encoding="utf-8"))


def _digest(manifest: PlacementManifest, tmp_path: Path, name: str) -> tuple[int, str]:
    path = tmp_path / f"{name}.json"
    manifest.save(path)
    blob = path.read_bytes()
    return len(blob), hashlib.sha256(blob).hexdigest()


def v1_manifest() -> PlacementManifest:
    return declared_manifest(
        tp=4,
        pp=2,
        dp=2,
        experts=DeclaredExpertLayout(**L48, expert_parallel=False),
    )


def v7_manifest(exceptions: DeclaredExpertMapExceptions | None) -> PlacementManifest:
    return declared_manifest(
        tp=1,
        dp=8,
        experts=DeclaredExpertLayout(
            **L24, placement_strategy="round_robin", exceptions=exceptions
        ),
    )


def v10_manifest() -> PlacementManifest:
    return declared_manifest(
        tp=4,
        pp=2,
        dp=2,
        experts=DeclaredExpertLayout(
            **L48, exceptions=DeclaredExpertMapExceptions(eplb=True)
        ),
    )


def v11_manifest(layout: dict) -> PlacementManifest:
    return declared_manifest(
        tp=1,
        dp=8,
        experts=DeclaredExpertLayout(
            **layout,
            expert_parallel=False,
            exceptions=DeclaredExpertMapExceptions(eplb=True),
        ),
    )


def _owner_counts(manifest: PlacementManifest) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    for placement in manifest.ranks:
        for layer, experts in placement.local_expert_ids.items():
            for expert in experts:
                counts[(layer, expert)] = counts.get((layer, expert), 0) + 1
    return counts


# --- the fatal compatibility identity -------------------------------------


def test_reference_manifests_keep_their_digests(tmp_path: Path) -> None:
    """The five accepted manifests are byte identical to the freeze."""

    frozen = _expectations()["baseline"]["placement_records"]
    builders = {
        "worked_example_tp4_pp2_dp2": lambda: declared_manifest(tp=4, pp=2, dp=2),
        "m4_tp8": lambda: declared_manifest(tp=8),
        "rail_pp8": lambda: declared_pipeline_placement(8),
        "m5_tp1_dp8": lambda: declared_manifest(tp=1, dp=8),
        "width_tp64": lambda: declared_manifest(tp=64, nodes=8, gpus_per_node=8),
    }
    assert set(builders) == set(frozen)
    for name, builder in builders.items():
        size, sha = _digest(builder(), tmp_path, name)
        assert (size, sha) == (frozen[name]["bytes"], frozen[name]["sha256"]), name


#: The baseline files this slice changes, so their recorded digests are
#: pre-change values by design and no longer describe the tree. Everything
#: else in the baseline must still match, which is what makes the record a
#: statement about what the slice touched rather than a list of hashes.
BASELINE_FILES_THIS_SLICE_CHANGES = {
    # The two selections and their exports.
    "simllm/placement/declared.py",
    "simllm/placement/__init__.py",
    # The registry: two entries close, one registers, the interface and status
    # sections gain the selections.
    "docs/modules/placement.md",
    # Repointed at its merged commit hashes so its own --check runs on main.
    "examples/sglang_declared_layout_v1/run_study.py",
}


def test_baseline_amendment_records_this_tree() -> None:
    """Every baseline file the slice does not touch still matches its digest."""

    amendment = _baseline_amendment()
    checked = 0
    for name, sha in amendment["baseline"]["files"].items():
        if name in BASELINE_FILES_THIS_SLICE_CHANGES:
            continue
        blob = (ROOT / name).read_bytes()
        assert hashlib.sha256(blob).hexdigest() == sha, name
        checked += 1
    assert checked == len(amendment["baseline"]["files"]) - len(
        BASELINE_FILES_THIS_SLICE_CHANGES
    )
    # Every name declared as changed must really be in the baseline, so a
    # stale entry here cannot silently excuse a file the record never named.
    assert BASELINE_FILES_THIS_SLICE_CHANGES <= set(amendment["baseline"]["files"])


# --- PLACE-7 ---------------------------------------------------------------


def test_v1_expert_parallel_off_worked_example() -> None:
    """Cell V1: the EP group is unchanged and every rank owns every expert."""

    frozen = _expectations()["cells"]["v1_ep_disabled_worked_example"]
    manifest = v1_manifest()
    every_expert = list(range(32))

    rank9 = manifest.by_rank(9)
    assert rank9.groups["ep"].global_ranks == frozen["rank_9"]["ep_group"]
    assert rank9.groups["ep"].rank_in_group == frozen["rank_9"]["rank_in_group"]
    assert list(rank9.pipeline_layer_range) == frozen["rank_9"]["layer_range"]
    assert sorted(rank9.local_expert_ids) == list(range(24))
    assert all(ids == every_expert for ids in rank9.local_expert_ids.values())

    rank15 = manifest.by_rank(15)
    assert rank15.groups["ep"].global_ranks == frozen["rank_15"]["ep_group"]
    assert rank15.groups["ep"].rank_in_group == frozen["rank_15"]["rank_in_group"]
    assert list(rank15.pipeline_layer_range) == frozen["rank_15"]["layer_range"]
    assert sorted(rank15.local_expert_ids) == list(range(24, 48))
    assert all(ids == every_expert for ids in rank15.local_expert_ids.values())

    # The expert-parallel-on twin is the accepted PLACE-3 cell C1.
    twin = declared_manifest(tp=4, pp=2, dp=2, experts=DeclaredExpertLayout(**L48))
    assert twin.by_rank(9).local_expert_ids[0] == frozen["rank_9"][
        "owned_with_expert_parallel_true"
    ]
    assert twin.by_rank(15).local_expert_ids[24] == frozen["rank_15"][
        "owned_with_expert_parallel_true"
    ]

    counts = _owner_counts(manifest)
    assert sum(counts.values()) == frozen["owner_entries_total"]
    assert set(counts.values()) == {frozen["owners_per_layer_expert_pair"]}


def test_v2_off_path_defaults_are_the_accepted_behavior() -> None:
    """Cell V2: both new fields default to what the tree did before."""

    frozen = _expectations()["interface"]["layout_field_defaults"]
    layout = DeclaredExpertLayout(**L48)
    assert layout.expert_parallel is frozen["expert_parallel"]
    assert layout.exceptions is frozen["exceptions"]
    assert DeclaredExpertMapExceptions() == DeclaredExpertMapExceptions(
        **{name: False for name in FRAMEWORK_CONDITIONS + MODEL_CONDITIONS}
    )


def test_v3_group_inventory_is_the_only_new_field() -> None:
    """Cell V3: the all-owner manifest differs from the expert-free one only
    in the fields the freeze names."""

    frozen = _expectations()["cells"]["v3_group_inventory_gap"]
    bare = declared_manifest(tp=4, pp=2, dp=2)
    manifest = v1_manifest()
    differing: set[str] = set()
    for plain, variant in zip(bare.ranks, manifest.ranks, strict=True):
        assert plain.global_rank == variant.global_rank
        assert plain.hostname == variant.hostname
        assert plain.local_rank == variant.local_rank
        for key in ("tp", "pp", "dp"):
            assert plain.groups[key] == variant.groups[key]
        differing |= set(variant.groups) - set(plain.groups)
        if plain.pipeline_layer_range != variant.pipeline_layer_range:
            differing.add("pipeline_layer_range")
        if plain.local_expert_ids != variant.local_expert_ids:
            differing.add("local_expert_ids")
        if plain.placement_epoch != variant.placement_epoch:
            differing.add("placement_epoch")
    # The declared epoch is zero here, so it is equal rather than differing.
    assert differing == set(frozen["differing_fields"]) - {"placement_epoch"}
    assert set(frozen["equal_fields"]) == {
        "global_rank",
        "hostname",
        "local_rank",
        "tp",
        "pp",
        "dp",
    }


def test_v4_degenerate_world_is_byte_identical(tmp_path: Path) -> None:
    """Cell V4: at a flattened width of one the flag cannot change anything."""

    frozen = _expectations()["cells"]["v4_degenerate_identity"]
    on = declared_manifest(experts=DeclaredExpertLayout(**L24))
    off = declared_manifest(experts=DeclaredExpertLayout(**L24, expert_parallel=False))
    assert _digest(on, tmp_path, "on") == _digest(off, tmp_path, "off")
    assert frozen["byte_identical_across_expert_parallel"] is True
    assert off.ranks[0].groups["ep"].global_ranks == frozen["ep_group"]
    assert off.ranks[0].groups["ep"].rank_in_group == frozen["rank_in_group"]
    assert off.ranks[0].local_expert_ids[0] == list(range(32))


def test_v5_round_robin_arange_follows_the_resolved_strategy() -> None:
    """Cell V5: a layout refused today builds once the fallback forces
    linear, because the framework never reaches the failing arange."""

    frozen = _expectations()["cells"]["v5_round_robin_arange"]
    with pytest.raises(ValueError) as refusal:
        declared_manifest(
            tp=1,
            dp=8,
            experts=DeclaredExpertLayout(**L4, placement_strategy="round_robin"),
        )
    assert str(refusal.value) == frozen["refused_with_expert_parallel_true"]

    layout = DeclaredExpertLayout(
        **L4, placement_strategy="round_robin", expert_parallel=False
    )
    assert declared_resolved_placement_strategy(layout, 8) == frozen[
        "resolved_strategy"
    ]
    manifest = declared_manifest(tp=1, dp=8, experts=layout)
    for placement in manifest.ranks:
        assert placement.local_expert_ids[0] == frozen["owned_every_rank"]


def test_v6_sglang_refuses_the_flag_and_spells_it_ep_size_one() -> None:
    """Cell V6: SGLang states the same deployment as ep_size=1, and its EP
    group is a singleton where the vLLM one spans the stage."""

    frozen = _expectations()["cells"]["v6_sglang"]
    with pytest.raises(ValueError, match="expert_parallel"):
        declared_sglang_manifest(
            tp=8,
            ep_size=8,
            experts=DeclaredExpertLayout(**L48, expert_parallel=False),
        )

    manifest = declared_sglang_manifest(
        tp=8, ep_size=1, experts=DeclaredExpertLayout(**L48)
    )
    rank5 = manifest.by_rank(5)
    assert rank5.groups["ep"].global_ranks == [5]
    assert rank5.groups["ep"].rank_in_group == frozen["contrast_rank_in_group"]
    assert rank5.local_expert_ids[0] == list(range(32))
    assert rank5.groups["moe_tp"].global_ranks == frozen["contrast_moe_tp_group"]

    # The divergence the freeze names: vLLM keeps the whole stage in the group.
    vllm_twin = declared_manifest(
        tp=8, experts=DeclaredExpertLayout(**L48, expert_parallel=False)
    )
    assert vllm_twin.by_rank(5).groups["ep"].global_ranks == list(range(8))


# --- PLACE-11 --------------------------------------------------------------


@pytest.mark.parametrize("condition", FRAMEWORK_CONDITIONS)
def test_v7_each_framework_condition_forces_linear(condition: str) -> None:
    """Cell V7: any one framework condition downgrades round_robin."""

    frozen = _expectations()["cells"]["v7_fallback_rows"]
    exceptions = DeclaredExpertMapExceptions(**{condition: True})
    manifest = v7_manifest(exceptions)
    assert manifest.by_rank(0).local_expert_ids[0] == frozen["resolved"]["rank_0"]
    assert manifest.by_rank(5).local_expert_ids[0] == frozen["resolved"]["rank_5"]
    layout = DeclaredExpertLayout(
        **L24, placement_strategy="round_robin", exceptions=exceptions
    )
    assert declared_resolved_placement_strategy(layout, 8) == frozen[
        "resolved_strategy"
    ]


def test_v7_absent_and_all_false_keep_round_robin() -> None:
    """Cell V7: the two inert declarations keep the caller's map."""

    frozen = _expectations()["cells"]["v7_fallback_rows"]
    for exceptions in (None, DeclaredExpertMapExceptions()):
        manifest = v7_manifest(exceptions)
        assert manifest.by_rank(0).local_expert_ids[0] == frozen["unresolved"][
            "rank_0"
        ]
        assert manifest.by_rank(5).local_expert_ids[0] == frozen["unresolved"][
            "rank_5"
        ]
        layout = DeclaredExpertLayout(
            **L24, placement_strategy="round_robin", exceptions=exceptions
        )
        assert declared_resolved_placement_strategy(layout, 8) == "round_robin"

    every = DeclaredExpertMapExceptions(**{name: True for name in FRAMEWORK_CONDITIONS})
    assert v7_manifest(every).by_rank(5).local_expert_ids[0] == frozen["resolved"][
        "rank_5"
    ]


def test_v8_exceptions_never_touch_a_declared_linear() -> None:
    """Cell V8: the conditions are inert outside a declared round_robin."""

    frozen = _expectations()["cells"]["v8_linear_untouched"]
    every = DeclaredExpertMapExceptions(
        **{name: True for name in FRAMEWORK_CONDITIONS + MODEL_CONDITIONS}
    )
    layout = DeclaredExpertLayout(**L24, exceptions=every)
    manifest = declared_manifest(tp=1, dp=8, experts=layout)
    assert manifest.by_rank(5).local_expert_ids[0] == frozen["all_six_booleans_true_rank_5"]
    assert declared_resolved_placement_strategy(layout, 8) == frozen[
        "resolved_strategy"
    ]
    plain = declared_manifest(tp=1, dp=8, experts=DeclaredExpertLayout(**L24))
    assert manifest.by_rank(5).local_expert_ids == plain.by_rank(5).local_expert_ids


def test_v9_refusals() -> None:
    """Cell V9: every frozen refusal raises a ValueError naming the field."""

    frozen = set(_expectations()["cells"]["v9_refusals"])
    assert len(frozen) == 8

    with pytest.raises(ValueError, match="eplb"):
        declared_manifest(
            tp=1,
            dp=8,
            experts=DeclaredExpertLayout(
                **L30, exceptions=DeclaredExpertMapExceptions(eplb=True)
            ),
        )
    with pytest.raises(ValueError, match="model_uniform_expert_blocks"):
        declared_manifest(
            tp=1,
            dp=8,
            experts=DeclaredExpertLayout(
                **L30,
                exceptions=DeclaredExpertMapExceptions(
                    model_uniform_expert_blocks=True
                ),
            ),
        )
    with pytest.raises(ValueError, match="model_refuses_tp_above_experts"):
        declared_manifest(
            tp=8,
            experts=DeclaredExpertLayout(
                **L4,
                exceptions=DeclaredExpertMapExceptions(
                    model_refuses_tp_above_experts=True
                ),
            ),
        )
    with pytest.raises(ValueError, match="expert_parallel must be a bool"):
        DeclaredExpertLayout(**L24, expert_parallel=1)
    with pytest.raises(ValueError, match="single_expert_group must be a bool"):
        DeclaredExpertMapExceptions(single_expert_group=1)
    with pytest.raises(ValueError, match="exceptions must be a"):
        DeclaredExpertLayout(**L24, exceptions="round_robin")
    with pytest.raises(ValueError, match="expert_parallel"):
        declared_sglang_manifest(
            tp=8,
            ep_size=8,
            experts=DeclaredExpertLayout(**L48, expert_parallel=False),
        )
    with pytest.raises(ValueError, match="exceptions must be None"):
        declared_sglang_manifest(
            tp=8,
            ep_size=8,
            experts=DeclaredExpertLayout(
                **L48, exceptions=DeclaredExpertMapExceptions(eplb=True)
            ),
        )


def test_v9_build_controls() -> None:
    """Cell V9: the matching controls must build and must not change."""

    frozen = _expectations()["cells"]["v9_build_controls"]
    remainder = frozen["remainder_without_eplb"]
    manifest = declared_manifest(tp=1, dp=8, experts=DeclaredExpertLayout(**L30))
    counts = [len(placement.local_expert_ids[0]) for placement in manifest.ranks]
    assert counts == remainder["counts"]
    assert manifest.by_rank(6).local_expert_ids[0] == remainder["linear_rank_6"]
    assert manifest.by_rank(7).local_expert_ids[0] == remainder["linear_rank_7"]

    narrow = frozen["tp_below_expert_count"]
    built = declared_manifest(
        tp=narrow["tp"],
        experts=DeclaredExpertLayout(
            **L24,
            exceptions=DeclaredExpertMapExceptions(
                model_refuses_tp_above_experts=narrow["model_refuses_tp_above_experts"]
            ),
        ),
    )
    plain = declared_manifest(tp=narrow["tp"], experts=DeclaredExpertLayout(**L24))
    assert built.by_rank(0).local_expert_ids == plain.by_rank(0).local_expert_ids


def test_v10_eplb_group_mirrors_the_ep_group() -> None:
    """Cell V10: EPLB adds one membership with the ep group's own ranks."""

    frozen = _expectations()["cells"]["v10_eplb_group"]
    manifest = v10_manifest()
    rank9 = manifest.by_rank(9)
    assert rank9.groups["eplb"].global_ranks == frozen["rank_9_eplb_group"]
    assert rank9.groups["eplb"].rank_in_group == frozen["rank_9_rank_in_group"]
    assert list(rank9.groups) == frozen["group_keys"]
    for placement in manifest.ranks:
        assert placement.groups["eplb"] == placement.groups["ep"]
    assert "eplb" not in v1_manifest().by_rank(9).groups


def test_v11_the_two_selections_composed() -> None:
    """Cell V11: with expert parallelism off the EPLB refusal cannot fire."""

    frozen = _expectations()["cells"]["v11_composed"]
    layout = DeclaredExpertLayout(
        **L24,
        placement_strategy="round_robin",
        expert_parallel=False,
        exceptions=DeclaredExpertMapExceptions(eplb=True),
    )
    assert declared_resolved_placement_strategy(layout, 8) == frozen["at_32_experts"][
        "resolved_strategy"
    ]
    manifest = v11_manifest(L24)
    assert manifest.by_rank(3).local_expert_ids[0] == list(range(32))
    assert manifest.by_rank(3).groups["eplb"].global_ranks == frozen["at_32_experts"][
        "eplb_group"
    ]

    # The framework's EPLB divisibility ValueError is guarded by use_ep.
    assert frozen["at_30_experts"]["refused"] is False
    remainder = v11_manifest(L30)
    assert remainder.by_rank(7).local_expert_ids[0] == list(range(30))


def test_v12_wire_identity(tmp_path: Path) -> None:
    """Cell V12: both manifests survive a save and load round trip."""

    frozen = _expectations()["cells"]["v12_wire_identity"]
    for name, manifest in (("v1", v1_manifest()), ("v10", v10_manifest())):
        path = tmp_path / f"{name}.json"
        manifest.save(path)
        assert PlacementManifest.load(path) == manifest
        assert manifest.source == frozen["source"]
        for placement in manifest.ranks:
            keys = list(placement.groups)
            assert keys == [key for key in frozen["group_key_order"] if key in keys]
            assert list(placement.local_expert_ids) == sorted(placement.local_expert_ids)


def test_v13_the_snapshot_refuses_replicated_ownership() -> None:
    """Cell V13, as corrected on 2026-09-15: the projection fails closed."""

    frozen = _v13_amendment()["v13"]
    manifest = v11_manifest(L24)
    counts = _owner_counts(manifest)
    assert sum(counts.values()) == frozen["ownership_entries"]
    assert len(counts) == frozen["distinct_layer_expert_pairs"]

    with pytest.raises(ValueError) as refusal:
        ExpertPlacementSnapshot.from_manifest(manifest, tuple(range(8)))
    assert str(refusal.value) == frozen["snapshot_over_full_ep_group"]["error"]

    sliced = ExpertPlacementSnapshot.from_manifest(
        manifest, tuple(frozen["snapshot_over_single_rank_slice"]["ep_ranks"])
    )
    assert len(sliced.expert_owners) == frozen["snapshot_over_single_rank_slice"][
        "entries"
    ]

    twin = declared_manifest(tp=1, dp=8, experts=DeclaredExpertLayout(**L24))
    on = ExpertPlacementSnapshot.from_manifest(twin, tuple(range(8)))
    assert len(on.expert_owners) == frozen["snapshot_over_expert_parallel_on_twin"][
        "entries"
    ]
