"""Run the frozen PLACE-3 declared expert placement qualification.

The study answers one question: can `declared_manifest` emit the
expert-parallel (EP) group, the pipeline layer range and the per-layer expert
ownership that the pinned vLLM 0.27.1 creates for the same DP x PP x TP
layout, without changing one byte of a manifest that does not ask for
experts? Cells C1, C3, C4, C6 and C8 are structural exact guards over the
declared builder, C5 is a rejection control family, and the five compatibility
digests are fatal by-construction identities. Cell C8 and the withdrawal of
C5's divisibility refusal come from the 2026-09-13 amendment of the freeze:
thirty experts over eight EP ranks follow the framework's remainder rule and
are recorded structurally, with no backend.

Cell C2 is the only timed and only scored cell. For each m5 step shape and
each expert-parallel width W in 2, 4 and 8 it builds a declared manifest,
reads its EP group out of the manifest, and drives `HtsimStepSink` on
`rnic-nn-fluid` at 400 Gbit/s with the frozen m5 mixture-of-experts (MoE)
geometry. The same step is then run a second time with the hand-typed EP rank
list the m5 record used. Both runs must reproduce the frozen m5 check-B
makespan exactly and must emit byte identical Group Operation Assembly
Language (GOAL) text, which is what makes "read the EP group from the
manifest" a substitution rather than a new model.

The run needs the backend binaries (`SIMLLM_HTSIM_RNIC`, `SIMLLM_TXT2BIN`).
Bulk artifacts go under the output root, which defaults to the study name
inside `SIMLLM_DATA_ROOT`; nothing machine specific reaches `results.json`.

Usage:

    python examples/declared_expert_placement_v1/run_study.py
    python examples/declared_expert_placement_v1/run_study.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

STUDY_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = STUDY_DIR.parents[1]
EXPECTATIONS_PATH = STUDY_DIR / "expectations.json"
AMENDMENT_PATH = STUDY_DIR / "expectations-amendment-2026-09-13.json"
RESULTS_PATH = STUDY_DIR / "results.json"
STUDY_NAME = "declared_expert_placement_v1"
EXPECTATIONS_COMMIT = "821969ae529eed76e56e9058addf1373516a2e1e"
AMENDMENT_COMMIT = "121098c3c8e0951e2a6b76d1b668baf431b8bf37"
RESULT_SCHEMA = "simllm-declared-expert-placement-result-v1"
DATA_ROOT_ENV = "SIMLLM_DATA_ROOT"

#: Cell C1: the worked example, 48 layers all carrying a routed MoE block.
C1_LAYOUT = {
    "num_layers": 48,
    "moe_layers": tuple(range(48)),
    "num_experts": 32,
}
#: Cell C2: the m5 granite geometry, 24 layers all MoE, 32 global experts.
C2_LAYOUT = {
    "num_layers": 24,
    "moe_layers": tuple(range(24)),
    "num_experts": 32,
}
#: Cell C3: a DeepSeek-class pipeline, MoE on layers 3 through 60.
C3_LAYOUT = {
    "num_layers": 61,
    "moe_layers": tuple(range(3, 61)),
    "num_experts": 256,
}
#: Cell C4: the frozen pipeline partition rows, as (num_layers, pp) keys.
C4_SHAPES = ((48, 2), (61, 4), (61, 8), (30, 4), (5, 2))
#: Cell C2 expert-parallel widths.
C2_WORLDS = (2, 4, 8)
#: Cell C8: the amendment's remainder layout, 24 MoE layers and 30 experts.
C8_LAYOUT = {
    "num_layers": 24,
    "moe_layers": tuple(range(24)),
    "num_experts": 30,
}
#: The frozen C5 control that each refuted claim of the amendment withdraws.
WITHDRAWN_C5_REFUSALS = {"divisibility refusal": "num_experts 30 at ep_size 8"}
#: Structural guard cells the amendment adds to the frozen evidence classes.
AMENDMENT_STRUCTURAL_CELLS = ("c8_remainder_layout",)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _require_expectations_ancestor() -> None:
    """Refuse to run unless the freeze and its amendment precede this code."""

    for label, commit in (
        ("expectations", EXPECTATIONS_COMMIT),
        ("amendment", AMENDMENT_COMMIT),
    ):
        completed = subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=False,
        )
        if completed.returncode != 0:
            raise SystemExit(f"the PLACE-3 {label} commit is not an ancestor")


def _default_output_root() -> Path | None:
    from simllm._local_config import path_from_env

    data_root = path_from_env(DATA_ROOT_ENV)
    return None if data_root is None else data_root / STUDY_NAME


def _c1_manifest(strategy: str = "linear"):
    from simllm.placement import DeclaredExpertLayout, declared_manifest

    return declared_manifest(
        tp=4,
        pp=2,
        dp=2,
        experts=DeclaredExpertLayout(**C1_LAYOUT, placement_strategy=strategy),
    )


def _c3_manifest():
    from simllm.placement import DeclaredExpertLayout, declared_manifest

    return declared_manifest(
        tp=8,
        pp=4,
        dp=2,
        nodes=8,
        gpus_per_node=8,
        experts=DeclaredExpertLayout(**C3_LAYOUT),
    )


def _c2_manifest(world: int):
    from simllm.placement import DeclaredExpertLayout, declared_manifest

    return declared_manifest(tp=1, dp=world, experts=DeclaredExpertLayout(**C2_LAYOUT))


def _c8_manifest(strategy: str = "linear"):
    from simllm.placement import DeclaredExpertLayout, declared_manifest

    return declared_manifest(
        tp=1,
        dp=8,
        experts=DeclaredExpertLayout(**C8_LAYOUT, placement_strategy=strategy),
    )


def run_c1() -> dict[str, Any]:
    """Cell C1: the worked example, both placement strategies."""

    observed: dict[str, Any] = {}
    for strategy in ("linear", "round_robin"):
        manifest = _c1_manifest(strategy)
        rows: dict[str, Any] = {}
        for global_rank in (9, 15):
            placement = manifest.by_rank(global_rank)
            owned = sorted(placement.local_expert_ids)
            rows[str(global_rank)] = {
                "ep_group": list(placement.groups["ep"].global_ranks),
                "expert_ids": list(placement.local_expert_ids[owned[0]]),
                "layer_range": list(placement.pipeline_layer_range),
                "local_expert_layers": [owned[0], owned[-1]],
                "one_expert_row_per_layer": len(
                    {tuple(ids) for ids in placement.local_expert_ids.values()}
                )
                == 1,
                "rank_in_group": placement.groups["ep"].rank_in_group,
            }
        stage_partitions = []
        for stage_ranks in (
            [0, 1, 2, 3, 8, 9, 10, 11],
            [4, 5, 6, 7, 12, 13, 14, 15],
        ):
            layers = sorted(manifest.by_rank(stage_ranks[0]).local_expert_ids)
            stage_partitions.append(
                all(
                    sorted(
                        expert
                        for rank in stage_ranks
                        for expert in manifest.by_rank(rank).local_expert_ids[layer]
                    )
                    == list(range(32))
                    for layer in layers
                )
            )
        observed[strategy] = {
            "group_key_order": list(manifest.by_rank(0).groups),
            "owners_partition_every_moe_layer": all(stage_partitions),
            "rank_count": len(manifest.ranks),
            "ranks": rows,
        }
    return observed


def run_c2_structure(world: int) -> dict[str, Any]:
    """Cell C2, the manifest half: EP group and snapshot conservation."""

    from simllm.traffic.routed_moe import ExpertPlacementSnapshot

    manifest = _c2_manifest(world)
    ep_ranks = manifest.group_ranks(0, "ep")
    snapshot = ExpertPlacementSnapshot.from_manifest(manifest, ep_ranks)
    owners = snapshot.owner_map()
    experts_per_rank = 32 // world
    return {
        "ep_ranks": list(ep_ranks),
        "every_pair_owned_once": len(owners) == len(snapshot.expert_owners),
        "owner_entries": len(snapshot.expert_owners),
        "owner_rule_holds": owners
        == {
            (layer, expert): expert // experts_per_rank
            for layer in range(24)
            for expert in range(32)
        },
        "placement_epoch": snapshot.placement_epoch,
    }


def _goal_text_index(workdir: Path) -> dict[str, str]:
    """Digest every rendered GOAL text artifact of one sink workdir."""

    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(workdir.rglob("*.goal"))
    }


def _run_step(workdir: Path, shape: str, world: int, ep_ranks) -> int:
    from examples.m5.run_m5 import STEP_SHAPES, G, moe_dims
    from simllm.backends import HtsimStepSink, HtsimStepSinkConfig

    sink = HtsimStepSink(
        HtsimStepSinkConfig(
            profile="rnic-nn-fluid",
            tp_ranks=(0,),
            dims=moe_dims(world),
            workdir=workdir,
            ep_ranks=tuple(ep_ranks),
            linkspeed_bps=400 * G,
        )
    )
    result = sink(STEP_SHAPES[shape]())
    if result is None:
        raise SystemExit(f"{shape} at W={world} produced no collective work")
    return result.step_latency_ps


def run_c2_live(output_root: Path, shape: str, world: int) -> dict[str, Any]:
    """Cell C2, the timed half: the manifest EP group drives the same step."""

    from examples.m5.run_m5 import FROZEN_B_MAKESPAN_PS

    manifest = _c2_manifest(world)
    manifest_ranks = tuple(manifest.group_ranks(0, "ep"))
    hand_typed_ranks = tuple(range(world))

    manifest_dir = output_root / f"c2-{shape}-w{world}-manifest"
    hand_typed_dir = output_root / f"c2-{shape}-w{world}-hand-typed"
    manifest_makespan = _run_step(manifest_dir, shape, world, manifest_ranks)
    hand_typed_makespan = _run_step(hand_typed_dir, shape, world, hand_typed_ranks)

    manifest_goal = _goal_text_index(manifest_dir)
    hand_typed_goal = _goal_text_index(hand_typed_dir)
    frozen = FROZEN_B_MAKESPAN_PS[(shape, world)]
    structure = run_c2_structure(world)
    return {
        "ep_ranks_from_manifest": list(manifest_ranks),
        "ep_ranks_hand_typed": list(hand_typed_ranks),
        "expected_makespan_ps": frozen,
        "goal_artifact_count": len(manifest_goal),
        "goal_text_byte_identical": (
            bool(manifest_goal) and manifest_goal == hand_typed_goal
        ),
        "hand_typed_makespan_ps": hand_typed_makespan,
        "makespan_ps": manifest_makespan,
        "makespan_matches_frozen": manifest_makespan == frozen,
        "shape": shape,
        "snapshot_owner_entries": structure["owner_entries"],
        "snapshot_owner_rule_holds": structure["owner_rule_holds"],
        "world": world,
    }


def run_c3() -> dict[str, Any]:
    """Cell C3: a 64-rank DeepSeek-class pipeline with four EP groups."""

    manifest = _c3_manifest()
    ep_groups = sorted(
        {tuple(rank.groups["ep"].global_ranks) for rank in manifest.ranks}
    )
    expected_groups = sorted(
        tuple(
            (dp * 4 + pipeline_index) * 8 + tp for dp in range(2) for tp in range(8)
        )
        for pipeline_index in range(4)
    )
    stage_ranges = sorted({rank.pipeline_layer_range for rank in manifest.ranks})
    moe_layers_per_stage = [
        len(manifest.by_rank(pipeline_index * 8).local_expert_ids)
        for pipeline_index in range(4)
    ]
    entries = [
        (layer, expert, rank.global_rank)
        for rank in manifest.ranks
        for layer, experts in rank.local_expert_ids.items()
        for expert in experts
    ]
    keys = {(layer, expert) for layer, expert, _rank in entries}
    block_rule = all(
        manifest.by_rank((dp * 4 + pipeline_index) * 8 + tp).local_expert_ids[layer]
        == list(range(16 * (dp * 8 + tp), 16 * (dp * 8 + tp) + 16))
        for pipeline_index in range(4)
        for dp in range(2)
        for tp in range(8)
        for layer in manifest.by_rank(
            (dp * 4 + pipeline_index) * 8 + tp
        ).local_expert_ids
    )
    rank_in_group_rule = all(
        manifest.by_rank((dp * 4 + pipeline_index) * 8 + tp).groups["ep"].rank_in_group
        == dp * 8 + tp
        for pipeline_index in range(4)
        for dp in range(2)
        for tp in range(8)
    )
    return {
        "contiguous_block_rule_holds": block_rule,
        "ep_group_count": len(ep_groups),
        "ep_groups_match_formula": ep_groups == expected_groups,
        "every_pair_owned_once": len(keys) == len(entries),
        "moe_layers_per_stage": moe_layers_per_stage,
        "owner_entries_total": len(entries),
        "rank_count": len(manifest.ranks),
        "rank_in_group_rule_holds": rank_in_group_rule,
        "stage_ranges": [list(interval) for interval in stage_ranges],
    }


def run_c4() -> dict[str, Any]:
    """Cell C4: the pipeline partition rows, with no expert in the loop."""

    from simllm.placement import declared_pipeline_partition

    rows: dict[str, Any] = {}
    for num_layers, pp in C4_SHAPES:
        intervals = declared_pipeline_partition(num_layers, pp)
        rows[f"{num_layers}/{pp}"] = [list(interval) for interval in intervals]
    return rows


def run_c5() -> dict[str, Any]:
    """Cell C5: every frozen refusal still in force, each a ValueError."""

    from simllm.placement import DeclaredExpertLayout, declared_manifest

    def refuses(build) -> bool:
        try:
            build()
        except ValueError:
            return True
        except Exception:  # noqa: BLE001 - any other type is a wrong refusal
            return False
        return False

    valid = dict(C1_LAYOUT)
    return {
        "boolean num_experts": refuses(
            lambda: DeclaredExpertLayout(**{**valid, "num_experts": True})
        ),
        "duplicate moe layers": refuses(
            lambda: DeclaredExpertLayout(**{**valid, "moe_layers": (0, 1, 1)})
        ),
        "empty moe layers": refuses(
            lambda: DeclaredExpertLayout(**{**valid, "moe_layers": ()})
        ),
        "moe layer 48 with 48 layers": refuses(
            lambda: DeclaredExpertLayout(**{**valid, "moe_layers": tuple(range(49))})
        ),
        "num_layers 1 with pp 2": refuses(
            lambda: declared_manifest(
                tp=1,
                pp=2,
                dp=32,
                experts=DeclaredExpertLayout(
                    num_layers=1, moe_layers=(0,), num_experts=32
                ),
            )
        ),
        "placement_epoch -1": refuses(
            lambda: DeclaredExpertLayout(**{**valid, "placement_epoch": -1})
        ),
        "unknown strategy": refuses(
            lambda: DeclaredExpertLayout(**{**valid, "placement_strategy": "hash"})
        ),
    }


def run_c6(output_root: Path) -> dict[str, Any]:
    """Cell C6: wire identity of the two declared expert manifests."""

    from simllm.placement import PlacementManifest

    cell_root = output_root / "c6-round-trip"
    cell_root.mkdir(parents=True, exist_ok=True)
    rows: dict[str, Any] = {}
    for label, build in (
        ("c1_worked_example", _c1_manifest),
        ("c3_deepseek_class", _c3_manifest),
    ):
        manifest = build()
        first = manifest.save(cell_root / f"{label}.json")
        loaded = PlacementManifest.load(first)
        second = loaded.save(cell_root / f"{label}-round-trip.json")
        raw = json.loads(first.read_text(encoding="utf-8"))
        rows[label] = {
            "ascending_expert_layer_keys": all(
                [int(layer) for layer in rank["local_expert_ids"]]
                == sorted(int(layer) for layer in rank["local_expert_ids"])
                for rank in raw["ranks"]
            ),
            "byte_identical": first.read_bytes() == second.read_bytes(),
            "ep_follows_dp": all(
                list(rank["groups"]) == ["tp", "pp", "dp", "ep"]
                for rank in raw["ranks"]
            ),
            "loaded_equals_built": loaded == manifest,
        }
    return rows


def run_c8() -> dict[str, Any]:
    """Cell C8: the amendment's remainder layout, recorded with no backend."""

    from simllm.traffic.routed_moe import ExpertPlacementSnapshot

    observed: dict[str, Any] = {}
    for strategy in ("linear", "round_robin"):
        manifest = _c8_manifest(strategy)
        ep_ranks = manifest.group_ranks(0, "ep")
        layers = sorted(manifest.by_rank(ep_ranks[0]).local_expert_ids)
        snapshot = ExpertPlacementSnapshot.from_manifest(manifest, ep_ranks)
        owners = snapshot.owner_map()
        every_pair = {
            (layer, expert)
            for layer in layers
            for expert in range(C8_LAYOUT["num_experts"])
        }
        observed[strategy] = {
            "counts": [
                len(manifest.by_rank(rank).local_expert_ids[layers[0]])
                for rank in ep_ranks
            ],
            "ep_ranks": list(ep_ranks),
            "every_pair_owned_once": (
                len(owners) == len(snapshot.expert_owners) and set(owners) == every_pair
            ),
            "moe_layer_count": len(layers),
            "one_expert_row_per_layer": all(
                sorted(manifest.by_rank(rank).local_expert_ids) == layers
                and len(
                    {
                        tuple(ids)
                        for ids in manifest.by_rank(rank).local_expert_ids.values()
                    }
                )
                == 1
                for rank in ep_ranks
            ),
            "owners": {
                str(rank): list(manifest.by_rank(rank).local_expert_ids[layers[0]])
                for rank in ep_ranks
            },
            "snapshot_owner_entries": len(snapshot.expert_owners),
        }
    return observed


def run_compatibility_digests(output_root: Path) -> dict[str, Any]:
    """The fatal identity: the option-absent builder output is byte locked."""

    from simllm.placement import declared_manifest, declared_pipeline_placement

    cell_root = output_root / "compatibility-digests"
    cell_root.mkdir(parents=True, exist_ok=True)
    builders = {
        "m4_tp8": lambda: declared_manifest(tp=8),
        "m5_tp1_dp8": lambda: declared_manifest(tp=1, dp=8),
        "rail_pp8": lambda: declared_pipeline_placement(8),
        "width_tp64": lambda: declared_manifest(tp=64, nodes=8, gpus_per_node=8),
        "worked_example_tp4_pp2_dp2": lambda: declared_manifest(tp=4, pp=2, dp=2),
    }
    rows: dict[str, Any] = {}
    for label, build in builders.items():
        payload = build().save(cell_root / f"{label}.json").read_bytes()
        rows[label] = {
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    return rows


def analyze_observation(
    observation: dict[str, Any], frozen: dict[str, Any], amendment: dict[str, Any]
) -> dict[str, Any]:
    """Apply every frozen and amended guard, then score only the C2 oracle rows."""

    findings: list[str] = []
    cells = observation["cells"]
    frozen_cells = frozen["cells"]

    if observation.get("expectations_commit") != EXPECTATIONS_COMMIT:
        findings.append("expectations commit identity")
    if observation.get("amendment_commit") != AMENDMENT_COMMIT:
        findings.append("amendment commit identity")

    # C1
    frozen_c1 = frozen_cells["c1_worked_example"]
    for strategy in ("linear", "round_robin"):
        observed = cells["c1_worked_example"][strategy]
        if observed["rank_count"] != 16:
            findings.append(f"c1 {strategy}: rank count")
        if observed["group_key_order"] != frozen["interface"]["group_key_order"]:
            findings.append(f"c1 {strategy}: group key order")
        if observed["owners_partition_every_moe_layer"] is not True:
            findings.append(f"c1 {strategy}: expert partition of a MoE layer")
        for global_rank, expected in (
            ("9", frozen_c1["rank_9"]),
            ("15", frozen_c1["rank_15"]),
        ):
            row = observed["ranks"][global_rank]
            if row["ep_group"] != expected["ep_group"]:
                findings.append(f"c1 {strategy}: rank {global_rank} EP group")
            if row["rank_in_group"] != expected["rank_in_group"]:
                findings.append(f"c1 {strategy}: rank {global_rank} rank in group")
            if row["layer_range"] != expected["layer_range"]:
                findings.append(f"c1 {strategy}: rank {global_rank} layer range")
            if row["expert_ids"] != expected[strategy]:
                findings.append(f"c1 {strategy}: rank {global_rank} expert ids")
            if row["one_expert_row_per_layer"] is not True:
                findings.append(f"c1 {strategy}: rank {global_rank} per-layer identity")
        rank_9 = observed["ranks"]["9"]
        if rank_9["local_expert_layers"] != [0, 23]:
            findings.append(f"c1 {strategy}: rank 9 MoE layer span")

    # C2, the scored oracle rows
    frozen_c2 = frozen_cells["c2_m5_identity"]
    scored_rows = 0
    scored_matches = 0
    for row in cells["c2_m5_identity"]:
        label = f"c2 {row['shape']} W={row['world']}"
        expected = frozen_c2["frozen_makespan_ps"][row["shape"]][str(row["world"])]
        scored_rows += 1
        if row["makespan_ps"] == expected:
            scored_matches += 1
        else:
            findings.append(f"{label}: makespan {row['makespan_ps']} != {expected}")
        if row["hand_typed_makespan_ps"] != expected:
            findings.append(f"{label}: hand-typed makespan")
        if row["ep_ranks_from_manifest"] != row["ep_ranks_hand_typed"]:
            findings.append(f"{label}: manifest EP group disagrees with the m5 list")
        if row["goal_text_byte_identical"] is not True:
            findings.append(f"{label}: GOAL text identity")
        if row["snapshot_owner_entries"] != frozen_c2["snapshot_owner_entries"]:
            findings.append(f"{label}: snapshot owner entries")
        if row["snapshot_owner_rule_holds"] is not True:
            findings.append(f"{label}: snapshot owner rule")
    if scored_rows != frozen["evidence"]["scored_exact_oracle_rows"]:
        findings.append("c2: scored oracle row count")
    decode_w8 = frozen_c2["frozen_makespan_ps"]["decode8x2048"]["8"]
    if decode_w8 <= frozen_c2["decode_w8_compute_floor_ps"]:
        findings.append("c2: decode W=8 sits below its own compute floor")

    # C3
    frozen_c3 = frozen_cells["c3_deepseek_class"]
    c3 = cells["c3_deepseek_class"]
    if c3["rank_count"] != frozen_c3["ranks"]:
        findings.append("c3: rank count")
    if c3["ep_group_count"] != frozen_c3["ep_groups"]:
        findings.append("c3: EP group count")
    if c3["ep_groups_match_formula"] is not True:
        findings.append("c3: EP group membership formula")
    if c3["rank_in_group_rule_holds"] is not True:
        findings.append("c3: EP rank in group")
    if c3["stage_ranges"] != frozen_c3["stage_ranges"]:
        findings.append("c3: stage ranges")
    if c3["moe_layers_per_stage"] != frozen_c3["moe_layers_per_stage"]:
        findings.append("c3: MoE layers per stage")
    if c3["owner_entries_total"] != frozen_c3["owner_entries_total"]:
        findings.append("c3: ownership entry count")
    if c3["every_pair_owned_once"] is not True:
        findings.append("c3: ownership conservation")
    if c3["contiguous_block_rule_holds"] is not True:
        findings.append("c3: contiguous expert block")

    # C4
    for key, expected in frozen_cells["c4_partitions"].items():
        if cells["c4_partitions"].get(key) != expected:
            findings.append(f"c4: partition {key}")

    # C5, less the controls the amendment withdraws
    withdrawn = set()
    for claim in amendment["refuted"]:
        if claim in WITHDRAWN_C5_REFUSALS:
            withdrawn.add(WITHDRAWN_C5_REFUSALS[claim])
        else:
            findings.append(f"amendment: refuted claim {claim!r} names no control")
    if not withdrawn <= set(frozen_cells["c5_refusals"]):
        findings.append("amendment: a withdrawn control is not a frozen C5 refusal")
    in_force = [
        refusal for refusal in frozen_cells["c5_refusals"] if refusal not in withdrawn
    ]
    if sorted(cells["c5_refusals"]) != sorted(in_force):
        findings.append("c5: rejection control family membership")
    for refusal in in_force:
        if cells["c5_refusals"].get(refusal) is not True:
            findings.append(f"c5: {refusal} was not refused")

    # C6
    for label in frozen_cells["c6_round_trip"]:
        row = cells["c6_round_trip"][label]
        for guard in (
            "ascending_expert_layer_keys",
            "byte_identical",
            "ep_follows_dp",
            "loaded_equals_built",
        ):
            if row.get(guard) is not True:
                findings.append(f"c6 {label}: {guard}")

    # C8, the amendment's remainder layout
    frozen_c8 = amendment["c8"]
    for strategy in ("linear", "round_robin"):
        observed = cells["c8_remainder_layout"][strategy]
        if observed["ep_ranks"] != list(range(frozen_c8["tp"] * frozen_c8["dp"])):
            findings.append(f"c8 {strategy}: EP group")
        if observed["counts"] != frozen_c8["counts"]:
            findings.append(f"c8 {strategy}: per-rank expert counts")
        for rank, expected in frozen_c8[strategy].items():
            if observed["owners"].get(rank) != expected:
                findings.append(f"c8 {strategy}: rank {rank} expert ids")
        if observed["moe_layer_count"] != frozen_c8["num_layers"]:
            findings.append(f"c8 {strategy}: MoE layer count")
        if observed["one_expert_row_per_layer"] is not True:
            findings.append(f"c8 {strategy}: per-layer identity")
        if observed["every_pair_owned_once"] is not True:
            findings.append(f"c8 {strategy}: ownership conservation")
        if observed["snapshot_owner_entries"] != frozen_c8["snapshot_owner_entries"]:
            findings.append(f"c8 {strategy}: snapshot owner entries")

    # The fatal compatibility digests
    frozen_records = frozen["baseline"]["placement_records"]
    for label, expected in frozen_records.items():
        row = cells["compatibility_digests"].get(label, {})
        if row.get("bytes") != expected["bytes"]:
            findings.append(f"digest {label}: bytes")
        if row.get("sha256") != expected["sha256"]:
            findings.append(f"digest {label}: sha256")

    return {
        "evidence": {
            "fatal_compatibility_digests": len(frozen_records),
            "rejection_controls": len(cells["c5_refusals"]),
            "scored_exact_oracle_rows": scored_rows,
            "scored_exact_oracle_rows_matched": scored_matches,
            "structural_guard_cells": (
                len(frozen["evidence"]["structural_guard_cells"])
                + len(AMENDMENT_STRUCTURAL_CELLS)
            ),
        },
        "findings": findings,
        "status": "PASS" if not findings else "VOID",
    }


def run_study(output_root: Path) -> dict[str, Any]:
    """Execute every cell and return the complete tracked summary."""

    from examples.m5.run_m5 import STEP_SHAPES

    frozen = json.loads(EXPECTATIONS_PATH.read_text(encoding="utf-8"))
    amendment = json.loads(AMENDMENT_PATH.read_text(encoding="utf-8"))
    cells = {
        "c1_worked_example": run_c1(),
        "c2_m5_identity": [
            run_c2_live(output_root, shape, world)
            for shape in STEP_SHAPES
            for world in C2_WORLDS
        ],
        "c3_deepseek_class": run_c3(),
        "c4_partitions": run_c4(),
        "c5_refusals": run_c5(),
        "c6_round_trip": run_c6(output_root),
        "c8_remainder_layout": run_c8(),
        "compatibility_digests": run_compatibility_digests(output_root),
    }
    observation = {
        "amendment_commit": AMENDMENT_COMMIT,
        "cells": cells,
        "expectations_commit": EXPECTATIONS_COMMIT,
    }
    analysis = analyze_observation(observation, frozen, amendment)
    return {
        "amendment_commit": AMENDMENT_COMMIT,
        "cells": cells,
        "evidence": analysis["evidence"],
        "expectations_commit": EXPECTATIONS_COMMIT,
        "findings": analysis["findings"],
        "implementation_commit": _git_output("rev-parse", "HEAD"),
        "schema": RESULT_SCHEMA,
        "status": analysis["status"],
    }


def comparable(summary: dict[str, Any]) -> dict[str, Any]:
    """Drop the one key that legitimately moves between reproducing runs."""

    return {
        key: value for key, value in summary.items() if key != "implementation_commit"
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the PLACE-3 qualification")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_default_output_root(),
        help=(
            "external artifact directory, fresh for each run; defaults to the "
            f"study name inside {DATA_ROOT_ENV}"
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare this run against the tracked results.json instead of writing it",
    )
    args = parser.parse_args()
    if args.output_root is None:
        raise SystemExit(f"provide --output-root or set {DATA_ROOT_ENV}")
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit("output root must be absent or empty")
    output_root.mkdir(parents=True, exist_ok=True)
    _require_expectations_ancestor()

    summary = run_study(output_root)
    _write_json(output_root / "results.json", summary)
    print(
        json.dumps(
            {
                "evidence": summary["evidence"],
                "findings": summary["findings"],
                "status": summary["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    for row in summary["cells"]["c2_m5_identity"]:
        print(
            f"C2 {row['shape']} W={row['world']} makespan_ps={row['makespan_ps']} "
            f"expected_ps={row['expected_makespan_ps']} "
            f"match={row['makespan_matches_frozen']} "
            f"goal_identical={row['goal_text_byte_identical']}"
        )
    for strategy, row in summary["cells"]["c8_remainder_layout"].items():
        print(
            f"C8 {strategy} counts={row['counts']} "
            f"snapshot_owner_entries={row['snapshot_owner_entries']} "
            f"every_pair_owned_once={row['every_pair_owned_once']}"
        )

    if args.check:
        tracked = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        if comparable(summary) != comparable(tracked):
            raise SystemExit("this run disagrees with the tracked results.json")
        print("check: this run reproduces the tracked results.json")
    else:
        _write_json(RESULTS_PATH, summary)

    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
