"""Run the frozen PLACE-13 SGLang declared layout qualification.

The study answers one question: can a second declared builder emit the rank
layout, the pipeline layer range, the expert-parallel (EP) group and the two
mixture-of-experts (MoE) side groups that the pinned SGLang commit creates for
one `tp x pp` world with `--ep-size` and `--moe-dp-size`, so that a study reads
its EP group and its expert owners from an SGLang-declared manifest exactly as
it reads them from a vLLM-declared one, without changing one byte of any
manifest the vLLM builder emits today?

Cells S1, S2, S3, S4, S5, S7, S8 and S9 are structural exact guards over the
declared builder, S6 is a rejection control family, and the five compatibility
digests together with the PLACE-3 study check are fatal by-construction
identities. Cell S5 additionally runs the installed SGLang package's own
`get_pp_indices` as an executable oracle; that arm is required for closure, and
a run without it is incomplete rather than failed.

Cell S10 is the only timed and only scored cell. For each m5 step shape and
each expert-parallel width W in 2, 4 and 8 it builds `tp=W, pp=1, ep_size=W`,
reads the EP group out of the manifest, and drives `HtsimStepSink` on
`rnic-nn-fluid` at 400 Gbit/s with the frozen m5 MoE geometry. The same step is
then run a second time with the hand-typed EP rank list the m5 record used.
Both runs must reproduce the frozen m5 check-B makespan exactly and must emit
byte identical Group Operation Assembly Language (GOAL) text, which is what
makes "read the EP group from the manifest" a substitution rather than a new
model.

The run needs the backend binaries (`SIMLLM_HTSIM_RNIC`, `SIMLLM_TXT2BIN`) and,
for the S5 oracle arm, an interpreter with SGLang installed (`SIMLLM_SGLANG_ENV`
or `--sglang-python`). Bulk artifacts go under the output root, which defaults
to the study name inside `SIMLLM_DATA_ROOT`; nothing machine specific reaches
`results.json`.

Usage:

    python examples/sglang_declared_layout_v1/run_study.py
    python examples/sglang_declared_layout_v1/run_study.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

STUDY_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = STUDY_DIR.parents[1]
EXPECTATIONS_PATH = STUDY_DIR / "expectations.json"
RESULTS_PATH = STUDY_DIR / "results.json"
STUDY_NAME = "sglang_declared_layout_v1"
EXPECTATIONS_COMMIT = "1eef03f5823c00703a78119773fd805bfa3e64f1"
RESULT_SCHEMA = "simllm-sglang-declared-layout-result-v1"
DATA_ROOT_ENV = "SIMLLM_DATA_ROOT"
SGLANG_ENV = "SIMLLM_SGLANG_ENV"
#: The PLACE-3 harness whose tracked results must still reproduce.
PLACE3_STUDY = Path("examples") / "declared_expert_placement_v1" / "run_study.py"

# Run this checkout, not whichever one the environment happens to have
# installed: the study compares a builder against a freeze that lives in the
# same tree, so both halves must come from that tree.
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

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
#: Cell S10 expert-parallel widths.
S10_WORLDS = (2, 4, 8)
#: The pinned SGLang commit, passed through as ``framework_version`` in S7.
PINNED_COMMIT = "bfeae4e79a8dc4600e006f1a5fbc85321a01c1a3"
#: The environment override the freeze deliberately does not model; it is
#: removed from the oracle interpreter's environment so a stray export cannot
#: silently change the rows the framework reports.
PARTITION_OVERRIDE = "SGLANG_PP_LAYER_PARTITION"

ORACLE_SCRIPT = """
import json
import sys

import sglang
from sglang.srt.distributed.utils import get_pp_indices

shapes = json.loads(sys.argv[2])
rows = {}
for key, (num_layers, pp) in shapes.items():
    rows[key] = [list(get_pp_indices(num_layers, stage, pp)) for stage in range(pp)]
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump({"version": sglang.__version__, "rows": rows}, handle)
"""


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
    """Refuse to run unless the freeze precedes this code."""

    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", EXPECTATIONS_COMMIT, "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit("the PLACE-13 expectations commit is not an ancestor")


def _default_output_root() -> Path | None:
    from simllm._local_config import path_from_env

    data_root = path_from_env(DATA_ROOT_ENV)
    return None if data_root is None else data_root / STUDY_NAME


def _default_sglang_python() -> Path | None:
    from simllm._local_config import path_from_env

    environment = path_from_env(SGLANG_ENV)
    return None if environment is None else environment / "bin" / "python"


def _layout(fields: dict[str, Any], **overrides: Any):
    from simllm.placement import DeclaredExpertLayout

    return DeclaredExpertLayout(**fields, **overrides)


def _s1_manifest():
    from simllm.placement import declared_sglang_manifest

    return declared_sglang_manifest(
        tp=8, pp=1, ep_size=8, experts=_layout(S1_LAYOUT)
    )


def _s2_manifest():
    from simllm.placement import declared_sglang_manifest

    return declared_sglang_manifest(
        tp=8, pp=1, ep_size=2, experts=_layout(S1_LAYOUT)
    )


def _s3_manifest():
    from simllm.placement import declared_sglang_manifest

    return declared_sglang_manifest(
        tp=8, pp=1, ep_size=2, moe_dp_size=4, experts=_layout(S1_LAYOUT)
    )


def _s4_manifest():
    from simllm.placement import declared_sglang_manifest

    return declared_sglang_manifest(
        tp=4, pp=4, ep_size=4, experts=_layout(S4_LAYOUT)
    )


def _s10_manifest(world: int):
    from simllm.placement import declared_sglang_manifest

    return declared_sglang_manifest(
        tp=world, pp=1, ep_size=world, experts=_layout(S10_LAYOUT)
    )


def _rank_row(manifest, global_rank: int, layer: int) -> dict[str, Any]:
    """Project the fields every frozen per-rank row names."""

    placement = manifest.by_rank(global_rank)
    return {
        "ep_group": list(placement.groups["ep"].global_ranks),
        "expert_ids": list(placement.local_expert_ids[layer]),
        "layer_range": list(placement.pipeline_layer_range),
        "moe_dp_group": list(placement.groups["moe_dp"].global_ranks),
        "moe_dp_rank": placement.groups["moe_dp"].rank_in_group,
        "moe_tp_group": list(placement.groups["moe_tp"].global_ranks),
        "moe_tp_rank": placement.groups["moe_tp"].rank_in_group,
        "one_expert_row_per_layer": len(
            {tuple(ids) for ids in placement.local_expert_ids.values()}
        )
        == 1,
        "rank_in_group": placement.groups["ep"].rank_in_group,
    }


def _ep_groups(manifest) -> list[list[int]]:
    return [
        list(group)
        for group in sorted(
            {tuple(rank.groups["ep"].global_ranks) for rank in manifest.ranks}
        )
    ]


def run_s1() -> dict[str, Any]:
    """Cell S1: the coincidence, where both frameworks agree rank for rank."""

    from simllm.placement import declared_manifest

    manifest = _s1_manifest()
    twin = declared_manifest(tp=8, experts=_layout(S1_LAYOUT))
    shared_fields = True
    for declared, other in zip(manifest.ranks, twin.ranks, strict=True):
        shared_fields = shared_fields and (
            declared.global_rank == other.global_rank
            and declared.hostname == other.hostname
            and declared.local_rank == other.local_rank
            and all(
                declared.groups[name] == other.groups[name]
                for name in ("tp", "pp", "dp", "ep")
            )
            and declared.pipeline_layer_range == other.pipeline_layer_range
            and declared.local_expert_ids == other.local_expert_ids
            and declared.placement_epoch == other.placement_epoch
        )
    return {
        "dp_is_a_singleton": all(
            rank.groups["dp"].global_ranks == [rank.global_rank]
            and rank.groups["dp"].rank_in_group == 0
            for rank in manifest.ranks
        ),
        "ep_group": list(manifest.group_ranks(0, "ep")),
        "ep_rank_equals_tp_rank": all(
            rank.groups["ep"].rank_in_group == rank.global_rank
            for rank in manifest.ranks
        ),
        "extra_groups_over_the_twin": sorted(
            set(manifest.by_rank(0).groups) - set(twin.by_rank(0).groups)
        ),
        "framework": manifest.framework,
        "group_key_order": list(manifest.by_rank(0).groups),
        "rank_5": _rank_row(manifest, 5, 0),
        "rank_count": len(manifest.ranks),
        "twin_framework": twin.framework,
        "vllm_twin_fields_equal": shared_fields,
    }


def run_s2() -> dict[str, Any]:
    """Cell S2: expert sharding, four EP groups of two over eight ranks."""

    manifest = _s2_manifest()
    partitions = []
    for group in _ep_groups(manifest):
        partitions.append(
            all(
                sorted(
                    expert
                    for rank in group
                    for expert in manifest.by_rank(rank).local_expert_ids[layer]
                )
                == list(range(32))
                for layer in range(48)
            )
        )
    identical_sets = all(
        len(
            {
                tuple(manifest.by_rank(rank).local_expert_ids[0])
                for rank in manifest.by_rank(member).groups["moe_tp"].global_ranks
            }
        )
        == 1
        for member in range(8)
    )
    return {
        "ep_groups": _ep_groups(manifest),
        "moe_tp_members_own_identical_sets": identical_sets,
        "owners_partition_every_ep_group": all(partitions),
        "rank_0": _rank_row(manifest, 0, 0),
        "rank_5": _rank_row(manifest, 5, 0),
    }


def run_s3() -> dict[str, Any]:
    """Cell S3: MoE data parallelism, where the MoE tensor width collapses."""

    manifest = _s3_manifest()
    return {
        "ep_groups": _ep_groups(manifest),
        "rank_5": _rank_row(manifest, 5, 0),
        "rank_6": _rank_row(manifest, 6, 0),
    }


def run_s4() -> dict[str, Any]:
    """Cell S4: a four-stage pipeline over sixteen ranks."""

    manifest = _s4_manifest()
    entries = [
        (layer, expert, rank.global_rank)
        for rank in manifest.ranks
        for layer, experts in rank.local_expert_ids.items()
        for expert in experts
    ]
    keys = {(layer, expert) for layer, expert, _rank in entries}
    return {
        "every_pair_owned_once": len(keys) == len(entries),
        "moe_layers_per_stage": [
            len(manifest.by_rank(stage * 4).local_expert_ids) for stage in range(4)
        ],
        "owner_entries_total": len(entries),
        "rank_13": _rank_row(manifest, 13, 45),
        "rank_count": len(manifest.ranks),
        "stage_ranges": [
            list(interval)
            for interval in sorted({rank.pipeline_layer_range for rank in manifest.ranks})
        ],
    }


def _frozen_partition_shapes(frozen: dict[str, Any]) -> dict[str, tuple[int, int]]:
    return {
        key: tuple(int(part) for part in key.split("/"))
        for key in frozen["cells"]["s5_partitions"]
        if "/" in key
    }


def run_s5(
    output_root: Path, frozen: dict[str, Any], sglang_python: Path | None
) -> dict[str, Any]:
    """Cell S5: partition arithmetic, with the framework itself as oracle.

    The oracle arm imports the installed SGLang package in its own interpreter
    and asks ``get_pp_indices`` for every frozen row. The environment override
    the freeze does not model is removed from that interpreter's environment,
    so a stray export cannot silently change what the framework reports.
    """

    from simllm.placement import declared_sglang_pipeline_partition

    shapes = _frozen_partition_shapes(frozen)
    rows = {
        key: [list(interval) for interval in declared_sglang_pipeline_partition(*shape)]
        for key, shape in shapes.items()
    }
    oracle: dict[str, Any] = {
        "agrees": None,
        "ran": False,
        "rows": None,
        "version": None,
        "version_suffix_matches": None,
    }
    if sglang_python is not None and Path(sglang_python).exists():
        environment = dict(os.environ)
        environment.pop(PARTITION_OVERRIDE, None)
        # The oracle is the framework's own answer; keep this repository off
        # that interpreter's import path so nothing here can reach it.
        environment.pop("PYTHONPATH", None)
        cell_root = output_root / "s5-oracle"
        cell_root.mkdir(parents=True, exist_ok=True)
        payload = cell_root / "get_pp_indices.json"
        try:
            subprocess.run(
                [
                    str(sglang_python),
                    "-c",
                    ORACLE_SCRIPT,
                    str(payload),
                    json.dumps({key: list(shape) for key, shape in shapes.items()}),
                ],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            )
            reported = json.loads(payload.read_text(encoding="utf-8"))
        except (OSError, subprocess.CalledProcessError, json.JSONDecodeError):
            reported = None
        if reported is not None:
            suffix = frozen["cells"]["s5_partitions"]["oracle_version_suffix"]
            oracle = {
                "agrees": reported["rows"] == rows,
                "ran": True,
                "rows": reported["rows"],
                "version": reported["version"],
                "version_suffix_matches": str(reported["version"]).endswith(suffix),
            }
    return {"oracle": oracle, "rows": rows}


class _RankBuilt(Exception):
    """Signal that the builder reached rank construction."""


class _SentinelPattern(str):
    """A hostname pattern that reports the moment a rank is built.

    Every rank takes its hostname from ``hostname_pattern.format(...)``, so a
    pattern whose ``format`` raises turns "no rank was built" from a claim
    about the code into something the run observes. A refusal that fires late
    raises this instead of its ``ValueError`` and is recorded as a failure.
    """

    def format(self, *args: object, **kwargs: object) -> str:
        raise _RankBuilt


def _observe_refusal(build) -> dict[str, Any]:
    """Record whether a layout was refused, and whether it was refused early."""

    try:
        build()
    except _RankBuilt:
        return {"no_rank_was_built": False, "refused": False}
    except ValueError:
        return {"no_rank_was_built": True, "refused": True}
    except Exception:  # noqa: BLE001 - any other type is a wrong refusal
        return {"no_rank_was_built": True, "refused": False}
    return {"no_rank_was_built": True, "refused": False}


def run_s6() -> dict[str, Any]:
    """Cell S6: every frozen refusal, each a ValueError before any rank.

    Each control is built with the sentinel hostname pattern, and one accepted
    layout is built with it as well: without that control the sentinel could
    be silently inert and every row would pass for the wrong reason.
    """

    from simllm.placement import DeclaredExpertLayout, declared_sglang_manifest

    sentinel = _SentinelPattern("node-{}")

    def refuses(build) -> dict[str, Any]:
        return _observe_refusal(build)

    controls = {
        "round_robin under the sglang builder": refuses(
            lambda: declared_sglang_manifest(
                tp=8,
                ep_size=8,
                hostname_pattern=sentinel,
                experts=_layout(S1_LAYOUT, placement_strategy="round_robin"),
            )
        ),
        "num_experts 30 at ep_size 8": refuses(
            lambda: declared_sglang_manifest(
                tp=8,
                ep_size=8,
                hostname_pattern=sentinel,
                experts=DeclaredExpertLayout(
                    num_layers=48, moe_layers=tuple(range(48)), num_experts=30
                ),
            )
        ),
        "tp 8 ep_size 3": refuses(
            lambda: declared_sglang_manifest(
                tp=8, ep_size=3, hostname_pattern=sentinel
            )
        ),
        "tp 8 ep_size 16": refuses(
            lambda: declared_sglang_manifest(
                tp=8, ep_size=16, hostname_pattern=sentinel
            )
        ),
        "tp 8 ep_size 2 moe_dp_size 2": refuses(
            lambda: declared_sglang_manifest(
                tp=8, ep_size=2, moe_dp_size=2, hostname_pattern=sentinel
            )
        ),
        "tp 8 moe_dp_size 2 pp 2": refuses(
            lambda: declared_sglang_manifest(
                tp=8, moe_dp_size=2, pp=2, hostname_pattern=sentinel
            )
        ),
        "ep_size 0": refuses(
            lambda: declared_sglang_manifest(
                tp=8, ep_size=0, hostname_pattern=sentinel
            )
        ),
        "moe_dp_size 0": refuses(
            lambda: declared_sglang_manifest(
                tp=8, moe_dp_size=0, hostname_pattern=sentinel
            )
        ),
        "num_layers 1 at pp 2": refuses(
            lambda: declared_sglang_manifest(
                tp=1,
                pp=2,
                hostname_pattern=sentinel,
                experts=DeclaredExpertLayout(
                    num_layers=1, moe_layers=(0,), num_experts=1
                ),
            )
        ),
        "tp 4 pp 3 nodes 2": refuses(
            lambda: declared_sglang_manifest(
                tp=4, pp=3, nodes=2, hostname_pattern=sentinel
            )
        ),
        "tp 4 pp 1 nodes 3": refuses(
            lambda: declared_sglang_manifest(
                tp=4, pp=1, nodes=3, hostname_pattern=sentinel
            )
        ),
        "tp 2 pp 1 nodes 1 gpus_per_node 1": refuses(
            lambda: declared_sglang_manifest(
                tp=2, pp=1, nodes=1, gpus_per_node=1, hostname_pattern=sentinel
            )
        ),
    }
    accepted = _observe_refusal(
        lambda: declared_sglang_manifest(tp=2, pp=1, hostname_pattern=sentinel)
    )
    return {
        "controls": controls,
        "sentinel_detects_a_built_rank": accepted["no_rank_was_built"] is False,
    }


def run_s7(output_root: Path) -> dict[str, Any]:
    """Cell S7: wire identity of the two declared SGLang expert manifests."""

    from simllm.placement import PlacementManifest, declared_sglang_manifest

    cell_root = output_root / "s7-round-trip"
    cell_root.mkdir(parents=True, exist_ok=True)
    rows: dict[str, Any] = {}
    for label, build in (
        ("s2_expert_sharding", _s2_manifest),
        ("s4_pipeline", _s4_manifest),
    ):
        manifest = build()
        first = manifest.save(cell_root / f"{label}.json")
        loaded = PlacementManifest.load(first)
        second = loaded.save(cell_root / f"{label}-round-trip.json")
        raw = json.loads(first.read_text(encoding="utf-8"))
        key_orders = {tuple(rank["groups"]) for rank in raw["ranks"]}
        rows[label] = {
            "ascending_expert_layer_keys": all(
                [int(layer) for layer in rank["local_expert_ids"]]
                == sorted(int(layer) for layer in rank["local_expert_ids"])
                for rank in raw["ranks"]
            ),
            "byte_identical": first.read_bytes() == second.read_bytes(),
            "framework": raw["framework"],
            "framework_version": raw["framework_version"],
            "group_key_order": list(min(key_orders)),
            "loaded_equals_built": loaded == manifest,
            "one_group_key_order": len(key_orders) == 1,
        }
    passed = declared_sglang_manifest(
        tp=8,
        ep_size=2,
        framework_version=PINNED_COMMIT,
        experts=_layout(S1_LAYOUT),
    )
    rows["framework_version_passthrough"] = passed.framework_version
    return rows


def run_s8() -> dict[str, Any]:
    """Cell S8: how ranks fill nodes, explicitly and by default."""

    from simllm.placement import declared_sglang_manifest

    def rows(manifest) -> list[list[Any]]:
        return [[rank.hostname, rank.local_rank] for rank in manifest.ranks]

    return {
        "tp16_pp1_nodes2": rows(declared_sglang_manifest(tp=16, pp=1, nodes=2)),
        "tp4_pp1_nodes2": rows(declared_sglang_manifest(tp=4, pp=1, nodes=2)),
        "tp8_pp2_default_nodes": len(
            {rank.hostname for rank in declared_sglang_manifest(tp=8, pp=2).ranks}
        ),
        "tp8_pp2_default_rows": rows(declared_sglang_manifest(tp=8, pp=2)),
        "tp8_pp2_nodes2": rows(declared_sglang_manifest(tp=8, pp=2, nodes=2)),
    }


def run_s9() -> dict[str, Any]:
    """Cell S9: the second builder is not the first one relabeled."""

    from simllm.placement import (
        declared_manifest,
        declared_pipeline_partition,
        declared_sglang_pipeline_partition,
    )

    vllm_partition = [list(i) for i in declared_pipeline_partition(61, 4)]
    sglang_partition = [list(i) for i in declared_sglang_pipeline_partition(61, 4)]
    sglang_ep = list(_s2_manifest().group_ranks(5, "ep"))
    vllm_ep = list(
        declared_manifest(tp=8, experts=_layout(S1_LAYOUT)).group_ranks(5, "ep")
    )
    return {
        "differing_stages": [
            stage
            for stage in range(4)
            if vllm_partition[stage] != sglang_partition[stage]
        ],
        "ep_group_rank_5_sglang": sglang_ep,
        "ep_group_rank_5_vllm": vllm_ep,
        "partition_61_4_sglang": sglang_partition,
        "partition_61_4_vllm": vllm_partition,
    }


def run_s10_structure(world: int) -> dict[str, Any]:
    """Cell S10, the manifest half: EP group and snapshot conservation."""

    from simllm.traffic.routed_moe import ExpertPlacementSnapshot

    manifest = _s10_manifest(world)
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


def run_s10_live(output_root: Path, shape: str, world: int) -> dict[str, Any]:
    """Cell S10, the timed half: the SGLang EP group drives the same step."""

    from examples.m5.run_m5 import FROZEN_B_MAKESPAN_PS

    manifest = _s10_manifest(world)
    manifest_ranks = tuple(manifest.group_ranks(0, "ep"))
    hand_typed_ranks = tuple(range(world))

    manifest_dir = output_root / f"s10-{shape}-w{world}-manifest"
    hand_typed_dir = output_root / f"s10-{shape}-w{world}-hand-typed"
    manifest_makespan = _run_step(manifest_dir, shape, world, manifest_ranks)
    hand_typed_makespan = _run_step(hand_typed_dir, shape, world, hand_typed_ranks)

    manifest_goal = _goal_text_index(manifest_dir)
    hand_typed_goal = _goal_text_index(hand_typed_dir)
    frozen = FROZEN_B_MAKESPAN_PS[(shape, world)]
    structure = run_s10_structure(world)
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


def run_compatibility_digests(output_root: Path) -> dict[str, Any]:
    """The fatal identity: the vLLM builder's output is byte locked."""

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


def run_place3_study_check(output_root: Path) -> dict[str, Any]:
    """The second fatal identity: the PLACE-3 study still reproduces."""

    cell_root = output_root / "place3-study-check"
    cell_root.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(REPOSITORY_ROOT) if not existing else f"{REPOSITORY_ROOT}{os.pathsep}{existing}"
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(PLACE3_STUDY),
            "--check",
            "--output-root",
            str(cell_root / "run"),
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    return {
        "reproduces": completed.returncode == 0
        and "check: this run reproduces" in completed.stdout,
        "study": PLACE3_STUDY.as_posix(),
    }


def analyze_observation(
    observation: dict[str, Any], frozen: dict[str, Any]
) -> dict[str, Any]:
    """Apply every frozen guard, then score only the six S10 oracle rows."""

    findings: list[str] = []
    cells = observation["cells"]
    frozen_cells = frozen["cells"]
    interface = frozen["interface"]

    if observation.get("expectations_commit") != EXPECTATIONS_COMMIT:
        findings.append("expectations commit identity")

    # S1, the coincidence
    frozen_s1 = frozen_cells["s1_coincidence"]
    s1 = cells["s1_coincidence"]
    if s1["rank_count"] != frozen_s1["tp"] * frozen_s1["pp"]:
        findings.append("s1: rank count")
    if s1["group_key_order"] != interface["group_key_order"]:
        findings.append("s1: group key order")
    if s1["framework"] != interface["framework_field"]:
        findings.append("s1: framework field")
    if s1["ep_group"] != frozen_s1["ep_group"]:
        findings.append("s1: EP group")
    if s1["ep_rank_equals_tp_rank"] is not True:
        findings.append("s1: EP rank equals the tensor rank")
    if s1["dp_is_a_singleton"] is not True:
        findings.append("s1: dp membership is not a singleton")
    frozen_rank_5 = frozen_s1["rank_5"]
    row = s1["rank_5"]
    if row["rank_in_group"] != frozen_rank_5["rank_in_group"]:
        findings.append("s1: rank 5 index in the EP group")
    if row["layer_range"] != frozen_rank_5["layer_range"]:
        findings.append("s1: rank 5 layer range")
    if row["expert_ids"] != frozen_rank_5["expert_ids"]:
        findings.append("s1: rank 5 expert ids")
    if row["moe_tp_group"] != frozen_rank_5["moe_tp_group"]:
        findings.append("s1: rank 5 MoE tensor group")
    if row["moe_dp_group"] != frozen_rank_5["moe_dp_group"]:
        findings.append("s1: rank 5 MoE data-parallel group")
    if row["one_expert_row_per_layer"] is not True:
        findings.append("s1: rank 5 per-layer identity")
    if s1["vllm_twin_fields_equal"] is not True:
        findings.append("s1: the frozen fields disagree with the vLLM twin")
    if s1["extra_groups_over_the_twin"] != sorted(("moe_tp", "moe_dp")):
        findings.append("s1: the extra memberships over the vLLM twin")
    if s1["twin_framework"] == s1["framework"]:
        findings.append("s1: the twin carries the same framework field")

    # S2, expert sharding
    frozen_s2 = frozen_cells["s2_expert_sharding"]
    s2 = cells["s2_expert_sharding"]
    if s2["ep_groups"] != frozen_s2["ep_groups"]:
        findings.append("s2: EP group membership")
    if s2["owners_partition_every_ep_group"] is not True:
        findings.append("s2: the owners of an EP group do not partition the experts")
    if s2["moe_tp_members_own_identical_sets"] is not True:
        findings.append("s2: a MoE tensor group owns disagreeing expert sets")
    expected_rank_5 = frozen_s2["rank_5"]
    row = s2["rank_5"]
    if row["ep_group"] != expected_rank_5["ep_group"]:
        findings.append("s2: rank 5 EP group")
    if row["rank_in_group"] != expected_rank_5["rank_in_group"]:
        findings.append("s2: rank 5 index in the EP group")
    if row["moe_tp_group"] != expected_rank_5["moe_tp_group"]:
        findings.append("s2: rank 5 MoE tensor group")
    if row["moe_tp_rank"] != expected_rank_5["moe_tp_rank"]:
        findings.append("s2: rank 5 index in the MoE tensor group")
    if row["moe_dp_group"] != expected_rank_5["moe_dp_group"]:
        findings.append("s2: rank 5 MoE data-parallel group")
    if row["expert_ids"] != list(range(16, 32)):
        findings.append("s2: rank 5 expert ids")
    row = s2["rank_0"]
    if row["ep_group"] != frozen_s2["rank_0"]["ep_group"]:
        findings.append("s2: rank 0 EP group")
    if row["rank_in_group"] != frozen_s2["rank_0"]["rank_in_group"]:
        findings.append("s2: rank 0 index in the EP group")
    if row["expert_ids"] != list(range(16)):
        findings.append("s2: rank 0 expert ids")

    # S3, MoE data parallelism
    frozen_s3 = frozen_cells["s3_moe_data_parallel"]
    s3 = cells["s3_moe_data_parallel"]
    if s3["ep_groups"] != frozen_s3["ep_groups"]:
        findings.append("s3: EP group membership")
    for label, expected_experts in (("rank_5", (16, 32)), ("rank_6", (0, 16))):
        expected = frozen_s3[label]
        row = s3[label]
        if row["ep_group"] != expected["ep_group"]:
            findings.append(f"s3: {label} EP group")
        if row["rank_in_group"] != expected["rank_in_group"]:
            findings.append(f"s3: {label} index in the EP group")
        if row["moe_dp_group"] != expected["moe_dp_group"]:
            findings.append(f"s3: {label} MoE data-parallel group")
        if row["moe_dp_rank"] != expected["moe_dp_rank"]:
            findings.append(f"s3: {label} index in the MoE data-parallel group")
        if "moe_tp_group" in expected and row["moe_tp_group"] != expected["moe_tp_group"]:
            findings.append(f"s3: {label} MoE tensor group")
        if row["expert_ids"] != list(range(*expected_experts)):
            findings.append(f"s3: {label} expert ids")

    # S4, the pipeline
    frozen_s4 = frozen_cells["s4_pipeline"]
    s4 = cells["s4_pipeline"]
    if s4["rank_count"] != frozen_s4["ranks"]:
        findings.append("s4: rank count")
    if s4["stage_ranges"] != frozen_s4["stage_ranges"]:
        findings.append("s4: stage ranges")
    if s4["moe_layers_per_stage"] != frozen_s4["moe_layers_per_stage"]:
        findings.append("s4: MoE layers per stage")
    if s4["owner_entries_total"] != frozen_s4["owner_entries_total"]:
        findings.append("s4: ownership entry count")
    if s4["every_pair_owned_once"] is not True:
        findings.append("s4: ownership conservation")
    expected = frozen_s4["rank_13"]
    row = s4["rank_13"]
    if row["ep_group"] != expected["ep_group"]:
        findings.append("s4: rank 13 EP group")
    if row["rank_in_group"] != expected["rank_in_group"]:
        findings.append("s4: rank 13 index in the EP group")
    if row["layer_range"] != expected["layer_range"]:
        findings.append("s4: rank 13 layer range")
    if row["expert_ids"] != list(range(16, 32)):
        findings.append("s4: rank 13 expert ids")
    if row["one_expert_row_per_layer"] is not True:
        findings.append("s4: rank 13 per-layer identity")

    # S5, the partition rows and the framework oracle
    frozen_s5 = frozen_cells["s5_partitions"]
    s5 = cells["s5_partitions"]
    for key, expected in frozen_s5.items():
        if "/" not in key:
            continue
        if s5["rows"].get(key) != expected:
            findings.append(f"s5: partition {key}")
    oracle = s5["oracle"]
    oracle_ran = oracle["ran"] is True
    if oracle_ran:
        if oracle["agrees"] is not True:
            findings.append("s5: the framework oracle disagrees with the builder")
        if oracle["version_suffix_matches"] is not True:
            findings.append("s5: the oracle interpreter is not the pinned commit")

    # S6, the rejection control family
    frozen_s6 = frozen_cells["s6_refusals"]
    controls = cells["s6_refusals"]["controls"]
    if sorted(controls) != sorted(frozen_s6):
        findings.append("s6: rejection control family membership")
    if cells["s6_refusals"]["sentinel_detects_a_built_rank"] is not True:
        findings.append("s6: the sentinel hostname pattern never fired")
    for refusal in frozen_s6:
        row = controls.get(refusal, {})
        if row.get("refused") is not True:
            findings.append(f"s6: {refusal} was not refused")
        if row.get("no_rank_was_built") is not True:
            findings.append(f"s6: {refusal} built a rank before refusing")

    # S7, wire identity
    frozen_s7 = frozen_cells["s7_round_trip"]
    for label in ("s2_expert_sharding", "s4_pipeline"):
        row = cells["s7_round_trip"][label]
        for guard in (
            "ascending_expert_layer_keys",
            "byte_identical",
            "loaded_equals_built",
            "one_group_key_order",
        ):
            if row.get(guard) is not True:
                findings.append(f"s7 {label}: {guard}")
        if list(row["group_key_order"]) != frozen_s7["group_key_order"]:
            findings.append(f"s7 {label}: group key order")
        if row["framework"] != frozen_s7["framework"]:
            findings.append(f"s7 {label}: framework field")
        if row["framework_version"] != frozen_s7["framework_version_default"]:
            findings.append(f"s7 {label}: default framework version")
    if cells["s7_round_trip"]["framework_version_passthrough"] != (
        frozen_s7["framework_version_passthrough"]
    ):
        findings.append("s7: framework version passthrough")

    # S8, node placement
    frozen_s8 = frozen_cells["s8_node_placement"]
    s8 = cells["s8_node_placement"]
    wide = [[f"node-{rank // 8}", rank % 8] for rank in range(16)]
    for label in ("tp8_pp2_nodes2", "tp16_pp1_nodes2"):
        if s8[label] != wide:
            findings.append(f"s8: {label} placement")
    if s8["tp4_pp1_nodes2"] != [
        ["node-0", 0],
        ["node-0", 1],
        ["node-1", 0],
        ["node-1", 1],
    ]:
        findings.append("s8: tp4_pp1_nodes2 placement")
    if s8["tp8_pp2_default_nodes"] != frozen_s8["tp8_pp2_default_nodes"]:
        findings.append("s8: default node count")
    if s8["tp8_pp2_default_rows"] != s8["tp8_pp2_nodes2"]:
        findings.append("s8: the default node count does not match the explicit one")

    # S9, divergence from the vLLM builder
    frozen_s9 = frozen_cells["s9_divergence"]
    s9 = cells["s9_divergence"]
    if s9["partition_61_4_vllm"] != frozen_s9["partition_61_4"]["vllm"]:
        findings.append("s9: the vLLM partition row")
    if s9["partition_61_4_sglang"] != frozen_s9["partition_61_4"]["sglang"]:
        findings.append("s9: the SGLang partition row")
    if s9["differing_stages"] != frozen_s9["partition_61_4"]["differing_stages"]:
        findings.append("s9: the stages the two partitions disagree on")
    if s9["ep_group_rank_5_vllm"] != frozen_s9["ep_group_rank_5_tp8_ep2"]["vllm"]:
        findings.append("s9: the vLLM EP group of rank 5")
    if s9["ep_group_rank_5_sglang"] != frozen_s9["ep_group_rank_5_tp8_ep2"]["sglang"]:
        findings.append("s9: the SGLang EP group of rank 5")

    # S10, the scored oracle rows
    frozen_s10 = frozen_cells["s10_m5_identity"]
    scored_rows = 0
    scored_matches = 0
    for row in cells["s10_m5_identity"]:
        label = f"s10 {row['shape']} W={row['world']}"
        expected = frozen_s10["frozen_makespan_ps"][row["shape"]][str(row["world"])]
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
        if row["snapshot_owner_entries"] != frozen_s10["snapshot_owner_entries"]:
            findings.append(f"{label}: snapshot owner entries")
        if row["snapshot_owner_rule_holds"] is not True:
            findings.append(f"{label}: snapshot owner rule")
    if scored_rows != len(S10_WORLDS) * 2:
        findings.append("s10: scored oracle row count")
    # The physical floor is checked against what this run measured, not
    # against the literal the freeze already carries: the frozen number
    # cannot fail its own frozen bound, so comparing the two would be a guard
    # that no defect can trip.
    floor = frozen_s10["decode_w8_compute_floor_ps"]
    decode_w8_rows = [
        row
        for row in cells["s10_m5_identity"]
        if row["shape"] == "decode8x2048" and row["world"] == 8
    ]
    if len(decode_w8_rows) != 1:
        findings.append("s10: the decode W=8 row the compute floor bounds is absent")
        observed_floor_row: dict[str, Any] = {
            "compute_floor_ps": floor,
            "observed_makespan_ps": None,
            "sits_above_the_floor": None,
        }
    else:
        observed = decode_w8_rows[0]["makespan_ps"]
        observed_floor_row = {
            "compute_floor_ps": floor,
            "observed_makespan_ps": observed,
            "sits_above_the_floor": observed > floor,
        }
        if observed <= floor:
            findings.append(
                f"s10: the measured decode W=8 makespan {observed} sits at or "
                f"below its compute floor {floor}"
            )

    # The fatal identities
    frozen_records = frozen["baseline"]["placement_records"]
    for label, expected in frozen_records.items():
        row = cells["compatibility_digests"].get(label, {})
        if row.get("bytes") != expected["bytes"]:
            findings.append(f"digest {label}: bytes")
        if row.get("sha256") != expected["sha256"]:
            findings.append(f"digest {label}: sha256")
    if cells["place3_study_check"]["reproduces"] is not (
        frozen["baseline"]["place3_study_check_reproduces"]
    ):
        findings.append("the PLACE-3 study check did not reproduce")

    if findings:
        status = "VOID"
    elif not oracle_ran:
        status = "INCOMPLETE"
    else:
        status = "PASS"
    return {
        "evidence": {
            "fatal_compatibility_digests": len(frozen_records),
            "fatal_place3_study_check": 1,
            "rejection_controls": len(cells["s6_refusals"]["controls"]),
            "s5_oracle_arm_ran": oracle_ran,
            "scored_exact_oracle_rows": scored_rows,
            "scored_exact_oracle_rows_matched": scored_matches,
            "structural_guard_cells": len(frozen["evidence"]["structural_exact"]),
        },
        "findings": findings,
        "physical_sanity": {"s10_decode_w8_compute_floor": observed_floor_row},
        "status": status,
    }


def run_study(output_root: Path, sglang_python: Path | None) -> dict[str, Any]:
    """Execute every cell and return the complete tracked summary."""

    from examples.m5.run_m5 import STEP_SHAPES

    frozen = json.loads(EXPECTATIONS_PATH.read_text(encoding="utf-8"))
    cells = {
        "compatibility_digests": run_compatibility_digests(output_root),
        "place3_study_check": run_place3_study_check(output_root),
        "s1_coincidence": run_s1(),
        "s2_expert_sharding": run_s2(),
        "s3_moe_data_parallel": run_s3(),
        "s4_pipeline": run_s4(),
        "s5_partitions": run_s5(output_root, frozen, sglang_python),
        "s6_refusals": run_s6(),
        "s7_round_trip": run_s7(output_root),
        "s8_node_placement": run_s8(),
        "s9_divergence": run_s9(),
        "s10_m5_identity": [
            run_s10_live(output_root, shape, world)
            for shape in STEP_SHAPES
            for world in S10_WORLDS
        ],
    }
    observation = {"cells": cells, "expectations_commit": EXPECTATIONS_COMMIT}
    analysis = analyze_observation(observation, frozen)
    return {
        "cells": cells,
        "evidence": analysis["evidence"],
        "expectations_commit": EXPECTATIONS_COMMIT,
        "findings": analysis["findings"],
        "implementation_commit": _git_output("rev-parse", "HEAD"),
        "physical_sanity": analysis["physical_sanity"],
        "schema": RESULT_SCHEMA,
        "status": analysis["status"],
    }


def comparable(summary: dict[str, Any]) -> dict[str, Any]:
    """Drop the one key that legitimately moves between reproducing runs."""

    return {
        key: value for key, value in summary.items() if key != "implementation_commit"
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the PLACE-13 qualification")
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
        "--sglang-python",
        type=Path,
        default=_default_sglang_python(),
        help=(
            "interpreter with the pinned SGLang installed, for the S5 oracle "
            f"arm; defaults to bin/python inside {SGLANG_ENV}"
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

    summary = run_study(output_root, args.sglang_python)
    _write_json(output_root / "results.json", summary)
    print(
        json.dumps(
            {
                "evidence": summary["evidence"],
                "findings": summary["findings"],
                "physical_sanity": summary["physical_sanity"],
                "status": summary["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    for row in summary["cells"]["s10_m5_identity"]:
        print(
            f"S10 {row['shape']} W={row['world']} makespan_ps={row['makespan_ps']} "
            f"expected_ps={row['expected_makespan_ps']} "
            f"match={row['makespan_matches_frozen']} "
            f"goal_identical={row['goal_text_byte_identical']}"
        )
    oracle = summary["cells"]["s5_partitions"]["oracle"]
    print(
        f"S5 oracle ran={oracle['ran']} version={oracle['version']} "
        f"agrees={oracle['agrees']} suffix={oracle['version_suffix_matches']}"
    )
    print(
        "PLACE-3 study check reproduces="
        f"{summary['cells']['place3_study_check']['reproduces']}"
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
