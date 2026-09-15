"""Run the frozen PLACE-7 and PLACE-11 declared expert variant qualification.

The study answers one question: can `DeclaredExpertLayout` state the two
mixture-of-experts (MoE) deployments the pinned vLLM 0.27.1 really runs and the
declared builder could not describe, a MoE model with expert parallelism
switched off and a model whose framework-side conditions downgrade a declared
`round_robin` map to `linear`, without changing one byte of any manifest that
does not ask for either selection?

Every cell is structural. Cells V1, V3, V4, V5, V6, V7, V8, V10, V11 and V12
are exact guards over the two builders, V9 is a rejection control family with
its matching build controls, and V13 is a recorded finding about the consumers
rather than a claim about this slice. V2, the five compatibility digests and
the two study checks are fatal by-construction identities.

The scored denominator is zero, deliberately and as frozen. The m5 identity
that PLACE-3 and PLACE-13 both score cannot move under either selection, since
it declares `linear` at 32 experts over a world that divides it with no
exception; and the enabled PLACE-7 path is not live-reachable, because the
vLLM step schedule refuses a replicated expert geometry and the routed expert
snapshot refuses to project one. The live evidence that the selections changed
nothing reachable is therefore the two study `--check` runs this harness drives
as subprocesses, which are fatal here.

Cell V13 follows the 2026-09-15 amendment: the snapshot refuses replicated
ownership outright rather than collapsing it, which is what the freeze got
wrong and what the amendment replaced before this harness existed.

The two identity cells need the backend binaries (`SIMLLM_HTSIM_RNIC`,
`SIMLLM_TXT2BIN`) that the PLACE-3 and PLACE-13 studies drive, and the PLACE-13
check additionally wants an interpreter with SGLang installed
(`SIMLLM_SGLANG_ENV`). Bulk artifacts go under the output root, which defaults
to the study name inside `SIMLLM_DATA_ROOT`; nothing machine specific reaches
`results.json`.

Usage:

    python examples/declared_expert_variants_v1/run_study.py
    python examples/declared_expert_variants_v1/run_study.py --check
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
BASELINE_AMENDMENT_PATH = STUDY_DIR / "expectations-amendment-2026-09-15.json"
V13_AMENDMENT_PATH = STUDY_DIR / "expectations-amendment-2026-09-15b.json"
RESULTS_PATH = STUDY_DIR / "results.json"
STUDY_NAME = "declared_expert_variants_v1"
EXPECTATIONS_COMMIT = "841dfb380180608f6c18b68ed5ae1e585d9be689"
BASELINE_AMENDMENT_COMMIT = "e08bed05e13c72a62fd183eba7798f86eb86e0f6"
V13_AMENDMENT_COMMIT = "5ffe5606c303fccd56b14ff13284cd22ae11543d"
RESULT_SCHEMA = "simllm-declared-expert-variants-result-v1"
DATA_ROOT_ENV = "SIMLLM_DATA_ROOT"
#: The two harnesses whose tracked results must still reproduce.
PLACE3_STUDY = Path("examples") / "declared_expert_placement_v1" / "run_study.py"
PLACE13_STUDY = Path("examples") / "sglang_declared_layout_v1" / "run_study.py"

# Run this checkout, not whichever one the environment happens to have
# installed: the study compares a builder against a freeze that lives in the
# same tree, so both halves must come from that tree.
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

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
MODEL_CONDITIONS = ("model_uniform_expert_blocks", "model_refuses_tp_above_experts")


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
    """Refuse to run unless the freeze and both amendments precede this code."""

    for label, commit in (
        ("expectations", EXPECTATIONS_COMMIT),
        ("baseline amendment", BASELINE_AMENDMENT_COMMIT),
        ("cell V13 amendment", V13_AMENDMENT_COMMIT),
    ):
        completed = subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=False,
        )
        if completed.returncode != 0:
            raise SystemExit(f"the {label} commit is not an ancestor")


def _default_output_root() -> Path | None:
    from simllm._local_config import path_from_env

    data_root = path_from_env(DATA_ROOT_ENV)
    return None if data_root is None else data_root / STUDY_NAME


def _layout(fields: dict[str, Any], **overrides: Any):
    from simllm.placement import DeclaredExpertLayout

    return DeclaredExpertLayout(**fields, **overrides)


def _exceptions(**fields: bool):
    from simllm.placement import DeclaredExpertMapExceptions

    return DeclaredExpertMapExceptions(**fields)


def _membership(placement, key: str) -> dict[str, Any]:
    group = placement.groups[key]
    return {"rank_in_group": group.rank_in_group, "ranks": list(group.global_ranks)}


def _owner_counts(manifest) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    for placement in manifest.ranks:
        for layer, experts in placement.local_expert_ids.items():
            for expert in experts:
                counts[(layer, expert)] = counts.get((layer, expert), 0) + 1
    return counts


def _refusal(thunk) -> str | None:
    """Return the ValueError message a frozen refusal raises, or None."""

    try:
        thunk()
    except ValueError as error:
        return str(error)
    return None


# --- the cells -------------------------------------------------------------


def run_v1() -> dict[str, Any]:
    """Cell V1: the EP group is unchanged and every rank owns every expert."""

    from simllm.placement import declared_manifest

    manifest = declared_manifest(
        tp=4, pp=2, dp=2, experts=_layout(L48, expert_parallel=False)
    )
    twin = declared_manifest(tp=4, pp=2, dp=2, experts=_layout(L48))
    counts = _owner_counts(manifest)
    rows = {}
    for rank in (9, 15):
        placement = manifest.by_rank(rank)
        rows[str(rank)] = {
            "ep": _membership(placement, "ep"),
            "layer_range": list(placement.pipeline_layer_range),
            "layers": sorted(placement.local_expert_ids),
            "owned": placement.local_expert_ids[placement.pipeline_layer_range[0]],
            "owned_with_expert_parallel_true": twin.by_rank(rank).local_expert_ids[
                twin.by_rank(rank).pipeline_layer_range[0]
            ],
        }
    return {
        "owner_entries_total": sum(counts.values()),
        "owners_per_pair": sorted({count for count in counts.values()}),
        "ranks": rows,
    }


def run_v2() -> dict[str, Any]:
    """Cell V2: both new fields default to the accepted behavior."""

    layout = _layout(L48)
    return {
        "exceptions_default_is_none": layout.exceptions is None,
        "expert_parallel_default": layout.expert_parallel,
        "exceptions_all_false_by_default": _exceptions() == _exceptions(
            **{name: False for name in FRAMEWORK_CONDITIONS + MODEL_CONDITIONS}
        ),
    }


def run_v3() -> dict[str, Any]:
    """Cell V3: the group-inventory gap PLACE-7 exists to close."""

    from simllm.placement import declared_manifest

    plain = declared_manifest(tp=4, pp=2, dp=2)
    variant = declared_manifest(
        tp=4, pp=2, dp=2, experts=_layout(L48, expert_parallel=False)
    )
    equal = {"global_rank", "hostname", "local_rank", "tp", "pp", "dp"}
    differing: set[str] = set()
    for one, other in zip(plain.ranks, variant.ranks, strict=True):
        if one.global_rank != other.global_rank:
            differing.add("global_rank")
        if one.hostname != other.hostname:
            differing.add("hostname")
        if one.local_rank != other.local_rank:
            differing.add("local_rank")
        for key in ("tp", "pp", "dp"):
            if one.groups[key] != other.groups[key]:
                differing.add(key)
        differing |= set(other.groups) - set(one.groups)
        if one.pipeline_layer_range != other.pipeline_layer_range:
            differing.add("pipeline_layer_range")
        if one.local_expert_ids != other.local_expert_ids:
            differing.add("local_expert_ids")
        if one.placement_epoch != other.placement_epoch:
            differing.add("placement_epoch")
    return {
        "base_fields_equal": not (differing & equal),
        "differing": sorted(differing),
    }


def run_v4(output_root: Path) -> dict[str, Any]:
    """Cell V4: at a flattened width of one the flag cannot change anything."""

    from simllm.placement import declared_manifest

    cell_root = output_root / "v4-degenerate"
    cell_root.mkdir(parents=True, exist_ok=True)
    digests = {}
    for label, layout in (
        ("expert_parallel_true", _layout(L24)),
        ("expert_parallel_false", _layout(L24, expert_parallel=False)),
    ):
        payload = declared_manifest(experts=layout).save(
            cell_root / f"{label}.json"
        ).read_bytes()
        digests[label] = hashlib.sha256(payload).hexdigest()
    off = declared_manifest(experts=_layout(L24, expert_parallel=False))
    return {
        "byte_identical": digests["expert_parallel_true"]
        == digests["expert_parallel_false"],
        "ep": _membership(off.ranks[0], "ep"),
        "owned": off.ranks[0].local_expert_ids[0],
    }


def run_v5() -> dict[str, Any]:
    """Cell V5: the arange refusal follows the resolved strategy."""

    from simllm.placement import declared_manifest, declared_resolved_placement_strategy

    refused = _refusal(
        lambda: declared_manifest(
            tp=1, dp=8, experts=_layout(L4, placement_strategy="round_robin")
        )
    )
    layout = _layout(L4, placement_strategy="round_robin", expert_parallel=False)
    manifest = declared_manifest(tp=1, dp=8, experts=layout)
    # One row per distinct ownership list: with the flag off every rank must
    # own the same full range, so the set has exactly one member.
    rows = {tuple(p.local_expert_ids[0]) for p in manifest.ranks}
    return {
        "refused_with_expert_parallel_true": refused,
        "resolved_strategy": declared_resolved_placement_strategy(layout, 8),
        "owned_every_rank": list(min(rows)),
        "distinct_ownership_rows": len(rows),
    }


def run_v6() -> dict[str, Any]:
    """Cell V6: SGLang spells the same deployment as ep_size=1."""

    from simllm.placement import declared_manifest, declared_sglang_manifest

    refused = _refusal(
        lambda: declared_sglang_manifest(
            tp=8, ep_size=8, experts=_layout(L48, expert_parallel=False)
        )
    )
    contrast = declared_sglang_manifest(tp=8, ep_size=1, experts=_layout(L48))
    rank5 = contrast.by_rank(5)
    vllm_twin = declared_manifest(tp=8, experts=_layout(L48, expert_parallel=False))
    return {
        "refusal": refused,
        "contrast_ep": _membership(rank5, "ep"),
        "contrast_moe_tp": _membership(rank5, "moe_tp"),
        "contrast_owned": rank5.local_expert_ids[0],
        "vllm_ep_ranks": list(vllm_twin.by_rank(5).groups["ep"].global_ranks),
    }


def run_v7() -> dict[str, Any]:
    """Cell V7: each framework condition downgrades a declared round_robin."""

    from simllm.placement import declared_manifest, declared_resolved_placement_strategy

    def rows(exceptions) -> dict[str, Any]:
        layout = _layout(L24, placement_strategy="round_robin", exceptions=exceptions)
        manifest = declared_manifest(tp=1, dp=8, experts=layout)
        return {
            "rank_0": manifest.by_rank(0).local_expert_ids[0],
            "rank_5": manifest.by_rank(5).local_expert_ids[0],
            "resolved": declared_resolved_placement_strategy(layout, 8),
        }

    declarations = {"none": rows(None), "all_false": rows(_exceptions())}
    for condition in FRAMEWORK_CONDITIONS:
        declarations[condition] = rows(_exceptions(**{condition: True}))
    declarations["all_four"] = rows(
        _exceptions(**{name: True for name in FRAMEWORK_CONDITIONS})
    )
    return declarations


def run_v8() -> dict[str, Any]:
    """Cell V8: the conditions are inert outside a declared round_robin."""

    from simllm.placement import declared_manifest, declared_resolved_placement_strategy

    every = _exceptions(
        **{name: True for name in FRAMEWORK_CONDITIONS + MODEL_CONDITIONS}
    )
    layout = _layout(L24, exceptions=every)
    manifest = declared_manifest(tp=1, dp=8, experts=layout)
    plain = declared_manifest(tp=1, dp=8, experts=_layout(L24))
    return {
        "rank_5": manifest.by_rank(5).local_expert_ids[0],
        "resolved": declared_resolved_placement_strategy(layout, 8),
        "matches_the_plain_layout": manifest.by_rank(5).local_expert_ids
        == plain.by_rank(5).local_expert_ids,
    }


def run_v9() -> dict[str, Any]:
    """Cell V9: the rejection control family and its build controls."""

    from simllm.placement import (
        DeclaredExpertLayout,
        DeclaredExpertMapExceptions,
        declared_manifest,
        declared_sglang_manifest,
    )

    controls = {
        "eplb remainder": lambda: declared_manifest(
            tp=1, dp=8, experts=_layout(L30, exceptions=_exceptions(eplb=True))
        ),
        "model_uniform_expert_blocks remainder": lambda: declared_manifest(
            tp=1,
            dp=8,
            experts=_layout(
                L30, exceptions=_exceptions(model_uniform_expert_blocks=True)
            ),
        ),
        "model_refuses_tp_above_experts": lambda: declared_manifest(
            tp=8,
            experts=_layout(
                L4, exceptions=_exceptions(model_refuses_tp_above_experts=True)
            ),
        ),
        "expert_parallel spelled 1": lambda: DeclaredExpertLayout(
            **L24, expert_parallel=1
        ),
        "single_expert_group spelled 1": lambda: DeclaredExpertMapExceptions(
            single_expert_group=1
        ),
        "exceptions is a string": lambda: DeclaredExpertLayout(
            **L24, exceptions="round_robin"
        ),
        "sglang expert_parallel False": lambda: declared_sglang_manifest(
            tp=8, ep_size=8, experts=_layout(L48, expert_parallel=False)
        ),
        "sglang exceptions": lambda: declared_sglang_manifest(
            tp=8, ep_size=8, experts=_layout(L48, exceptions=_exceptions(eplb=True))
        ),
    }
    refusals = {label: _refusal(thunk) for label, thunk in controls.items()}

    remainder = declared_manifest(tp=1, dp=8, experts=_layout(L30))
    narrow = declared_manifest(
        tp=4,
        experts=_layout(
            L24, exceptions=_exceptions(model_refuses_tp_above_experts=True)
        ),
    )
    plain_narrow = declared_manifest(tp=4, experts=_layout(L24))
    return {
        "refusals": refusals,
        "all_refused": all(message is not None for message in refusals.values()),
        "build_controls": {
            "remainder_counts": [
                len(p.local_expert_ids[0]) for p in remainder.ranks
            ],
            "remainder_rank_6": remainder.by_rank(6).local_expert_ids[0],
            "remainder_rank_7": remainder.by_rank(7).local_expert_ids[0],
            "tp_below_expert_count_unchanged": narrow.by_rank(0).local_expert_ids
            == plain_narrow.by_rank(0).local_expert_ids,
        },
    }


def run_v10() -> dict[str, Any]:
    """Cell V10: EPLB adds one membership with the ep group's own ranks."""

    from simllm.placement import declared_manifest

    manifest = declared_manifest(
        tp=4, pp=2, dp=2, experts=_layout(L48, exceptions=_exceptions(eplb=True))
    )
    without = declared_manifest(tp=4, pp=2, dp=2, experts=_layout(L48))
    return {
        "rank_9_eplb": _membership(manifest.by_rank(9), "eplb"),
        "group_keys": list(manifest.by_rank(9).groups),
        "every_rank_mirrors_ep": all(
            p.groups["eplb"] == p.groups["ep"] for p in manifest.ranks
        ),
        "absent_without_eplb": "eplb" not in without.by_rank(9).groups,
    }


def run_v11() -> dict[str, Any]:
    """Cell V11: with expert parallelism off the EPLB refusal cannot fire."""

    from simllm.placement import declared_manifest, declared_resolved_placement_strategy

    layout = _layout(
        L24,
        placement_strategy="round_robin",
        expert_parallel=False,
        exceptions=_exceptions(eplb=True),
    )
    at32 = declared_manifest(
        tp=1,
        dp=8,
        experts=_layout(L24, expert_parallel=False, exceptions=_exceptions(eplb=True)),
    )
    refused30 = _refusal(
        lambda: declared_manifest(
            tp=1,
            dp=8,
            experts=_layout(
                L30, expert_parallel=False, exceptions=_exceptions(eplb=True)
            ),
        )
    )
    at30 = declared_manifest(
        tp=1,
        dp=8,
        experts=_layout(L30, expert_parallel=False, exceptions=_exceptions(eplb=True)),
    )
    return {
        "resolved": declared_resolved_placement_strategy(layout, 8),
        "owned_at_32": at32.by_rank(3).local_expert_ids[0],
        "eplb_at_32": _membership(at32.by_rank(3), "eplb"),
        "refused_at_30": refused30,
        "owned_at_30": at30.by_rank(7).local_expert_ids[0],
    }


def run_v12(output_root: Path) -> dict[str, Any]:
    """Cell V12: both manifests survive a save and load round trip."""

    from simllm.placement import PlacementManifest, declared_manifest

    cell_root = output_root / "v12-round-trip"
    cell_root.mkdir(parents=True, exist_ok=True)
    manifests = {
        "v1": declared_manifest(
            tp=4, pp=2, dp=2, experts=_layout(L48, expert_parallel=False)
        ),
        "v10": declared_manifest(
            tp=4, pp=2, dp=2, experts=_layout(L48, exceptions=_exceptions(eplb=True))
        ),
    }
    rows = {}
    for label, manifest in manifests.items():
        path = manifest.save(cell_root / f"{label}.json")
        rows[label] = {
            "round_trips": PlacementManifest.load(path) == manifest,
            "source": manifest.source,
            # Lists, not tuples: this row is compared against the tracked
            # JSON on --check, and JSON has no tuple to compare against.
            "group_key_orders": [
                list(order) for order in sorted({tuple(p.groups) for p in manifest.ranks})
            ],
            "layer_keys_ascending": all(
                list(p.local_expert_ids) == sorted(p.local_expert_ids)
                for p in manifest.ranks
            ),
        }
    return rows


def run_v13() -> dict[str, Any]:
    """Cell V13, as corrected on 2026-09-15: the projection fails closed."""

    from simllm.placement import declared_manifest
    from simllm.traffic.routed_moe import ExpertPlacementSnapshot

    manifest = declared_manifest(
        tp=1,
        dp=8,
        experts=_layout(L24, expert_parallel=False, exceptions=_exceptions(eplb=True)),
    )
    counts = _owner_counts(manifest)
    refusal = _refusal(
        lambda: ExpertPlacementSnapshot.from_manifest(manifest, tuple(range(8)))
    )
    sliced = ExpertPlacementSnapshot.from_manifest(manifest, (3,))
    twin = declared_manifest(tp=1, dp=8, experts=_layout(L24))
    on = ExpertPlacementSnapshot.from_manifest(twin, tuple(range(8)))
    return {
        "ownership_entries": sum(counts.values()),
        "distinct_pairs": len(counts),
        "snapshot_refusal": refusal,
        "single_rank_slice_entries": len(sliced.expert_owners),
        "expert_parallel_on_twin_entries": len(on.expert_owners),
    }


# --- the fatal identities --------------------------------------------------


def run_compatibility_digests(output_root: Path) -> dict[str, Any]:
    """The first fatal identity: the accepted output is byte locked."""

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


def run_study_check(output_root: Path, study: Path, label: str) -> dict[str, Any]:
    """Drive one tracked study's own --check into a fresh subdirectory."""

    cell_root = output_root / label
    cell_root.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(REPOSITORY_ROOT)
        if not existing
        else f"{REPOSITORY_ROOT}{os.pathsep}{existing}"
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(study),
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
        "study": study.as_posix(),
    }


# --- analysis --------------------------------------------------------------


def analyze_observation(
    observation: dict[str, Any], frozen: dict[str, Any], v13_frozen: dict[str, Any]
) -> dict[str, Any]:
    """Apply every frozen guard. Nothing here is scored: the denominator is
    zero by design, so the status is exactness plus the fatal identities."""

    findings: list[str] = []
    cells = observation["cells"]
    frozen_cells = frozen["cells"]

    if observation.get("expectations_commit") != EXPECTATIONS_COMMIT:
        findings.append("expectations commit identity")

    # V1
    fv1 = frozen_cells["v1_ep_disabled_worked_example"]
    v1 = cells["v1_ep_disabled_worked_example"]
    every_expert = list(range(fv1["num_experts"]))
    for rank, frozen_rank in (("9", fv1["rank_9"]), ("15", fv1["rank_15"])):
        row = v1["ranks"][rank]
        if row["ep"]["ranks"] != frozen_rank["ep_group"]:
            findings.append(f"v1: rank {rank} ep group")
        if row["ep"]["rank_in_group"] != frozen_rank["rank_in_group"]:
            findings.append(f"v1: rank {rank} rank_in_group")
        if row["layer_range"] != frozen_rank["layer_range"]:
            findings.append(f"v1: rank {rank} layer range")
        if row["owned"] != every_expert:
            findings.append(f"v1: rank {rank} does not own every expert")
        if row["owned_with_expert_parallel_true"] != frozen_rank[
            "owned_with_expert_parallel_true"
        ]:
            findings.append(f"v1: rank {rank} expert-parallel-on twin")
    if v1["owner_entries_total"] != fv1["owner_entries_total"]:
        findings.append("v1: ownership entry conservation")
    if v1["owners_per_pair"] != [fv1["owners_per_layer_expert_pair"]]:
        findings.append("v1: owners per layer-expert pair")

    # V2, a fatal identity
    v2 = cells["v2_off_path_identity"]
    defaults = frozen["interface"]["layout_field_defaults"]
    if v2["expert_parallel_default"] is not defaults["expert_parallel"]:
        findings.append("v2: expert_parallel default")
    if not v2["exceptions_default_is_none"]:
        findings.append("v2: exceptions default")
    if not v2["exceptions_all_false_by_default"]:
        findings.append("v2: exceptions field defaults")

    # V3
    fv3 = frozen_cells["v3_group_inventory_gap"]
    v3 = cells["v3_group_inventory_gap"]
    if not v3["base_fields_equal"]:
        findings.append("v3: a base field moved")
    # The declared epoch is zero in this cell, so it is equal rather than
    # differing; every other named field must differ.
    if v3["differing"] != sorted(set(fv3["differing_fields"]) - {"placement_epoch"}):
        findings.append("v3: differing fields")

    # V4
    fv4 = frozen_cells["v4_degenerate_identity"]
    v4 = cells["v4_degenerate_identity"]
    if v4["byte_identical"] is not fv4["byte_identical_across_expert_parallel"]:
        findings.append("v4: byte identity across the flag")
    if v4["ep"]["ranks"] != fv4["ep_group"]:
        findings.append("v4: ep group")
    if v4["ep"]["rank_in_group"] != fv4["rank_in_group"]:
        findings.append("v4: rank_in_group")
    if v4["owned"] != every_expert:
        findings.append("v4: ownership")

    # V5
    fv5 = frozen_cells["v5_round_robin_arange"]
    v5 = cells["v5_round_robin_arange"]
    if v5["refused_with_expert_parallel_true"] != fv5[
        "refused_with_expert_parallel_true"
    ]:
        findings.append("v5: the arange refusal message")
    if v5["resolved_strategy"] != fv5["resolved_strategy"]:
        findings.append("v5: resolved strategy")
    if list(v5["owned_every_rank"]) != fv5["owned_every_rank"]:
        findings.append("v5: ownership")
    if v5["distinct_ownership_rows"] != 1:
        findings.append("v5: ranks disagree on ownership")

    # V6
    fv6 = frozen_cells["v6_sglang"]
    v6 = cells["v6_sglang"]
    if v6["refusal"] is None or "expert_parallel" not in v6["refusal"]:
        findings.append("v6: the SGLang refusal does not name the field")
    if v6["contrast_ep"]["rank_in_group"] != fv6["contrast_rank_in_group"]:
        findings.append("v6: contrast rank_in_group")
    if len(v6["contrast_ep"]["ranks"]) != 1:
        findings.append("v6: the SGLang ep group is not a singleton")
    if v6["contrast_moe_tp"]["ranks"] != fv6["contrast_moe_tp_group"]:
        findings.append("v6: contrast moe_tp group")
    if v6["contrast_owned"] != every_expert:
        findings.append("v6: contrast ownership")
    if v6["vllm_ep_ranks"] != list(range(8)):
        findings.append("v6: the vLLM ep group is not the whole stage")

    # V7
    fv7 = frozen_cells["v7_fallback_rows"]
    v7 = cells["v7_fallback_rows"]
    for label in ("none", "all_false"):
        row = v7[label]
        if row["rank_0"] != fv7["unresolved"]["rank_0"]:
            findings.append(f"v7: {label} rank 0")
        if row["rank_5"] != fv7["unresolved"]["rank_5"]:
            findings.append(f"v7: {label} rank 5")
        if row["resolved"] != "round_robin":
            findings.append(f"v7: {label} resolved strategy")
    for label in (*FRAMEWORK_CONDITIONS, "all_four"):
        row = v7[label]
        if row["rank_0"] != fv7["resolved"]["rank_0"]:
            findings.append(f"v7: {label} rank 0")
        if row["rank_5"] != fv7["resolved"]["rank_5"]:
            findings.append(f"v7: {label} rank 5")
        if row["resolved"] != fv7["resolved_strategy"]:
            findings.append(f"v7: {label} resolved strategy")

    # V8
    fv8 = frozen_cells["v8_linear_untouched"]
    v8 = cells["v8_linear_untouched"]
    if v8["rank_5"] != fv8["all_six_booleans_true_rank_5"]:
        findings.append("v8: rank 5")
    if v8["resolved"] != fv8["resolved_strategy"]:
        findings.append("v8: resolved strategy")
    if not v8["matches_the_plain_layout"]:
        findings.append("v8: the exceptions moved a declared linear")

    # V9
    fv9 = frozen_cells["v9_build_controls"]
    v9 = cells["v9_refusals"]
    if len(v9["refusals"]) != len(frozen_cells["v9_refusals"]):
        findings.append("v9: refusal count")
    if not v9["all_refused"]:
        findings.append("v9: a frozen refusal did not raise")
    remainder = fv9["remainder_without_eplb"]
    controls = v9["build_controls"]
    if controls["remainder_counts"] != remainder["counts"]:
        findings.append("v9: remainder control counts")
    if controls["remainder_rank_6"] != remainder["linear_rank_6"]:
        findings.append("v9: remainder control rank 6")
    if controls["remainder_rank_7"] != remainder["linear_rank_7"]:
        findings.append("v9: remainder control rank 7")
    if not controls["tp_below_expert_count_unchanged"]:
        findings.append("v9: the narrow-tp control changed its ownership")

    # V10
    fv10 = frozen_cells["v10_eplb_group"]
    v10 = cells["v10_eplb_group"]
    if v10["rank_9_eplb"]["ranks"] != fv10["rank_9_eplb_group"]:
        findings.append("v10: eplb ranks")
    if v10["rank_9_eplb"]["rank_in_group"] != fv10["rank_9_rank_in_group"]:
        findings.append("v10: eplb rank_in_group")
    if v10["group_keys"] != fv10["group_keys"]:
        findings.append("v10: group key order")
    if not v10["every_rank_mirrors_ep"]:
        findings.append("v10: a rank's eplb group differs from its ep group")
    if v10["absent_without_eplb"] is not fv10["absent_when_eplb_false"]:
        findings.append("v10: the eplb key appears without the condition")

    # V11
    fv11 = frozen_cells["v11_composed"]
    v11 = cells["v11_composed"]
    if v11["resolved"] != fv11["at_32_experts"]["resolved_strategy"]:
        findings.append("v11: resolved strategy")
    if v11["owned_at_32"] != every_expert:
        findings.append("v11: ownership at 32 experts")
    if v11["eplb_at_32"]["ranks"] != fv11["at_32_experts"]["eplb_group"]:
        findings.append("v11: eplb group")
    if (v11["refused_at_30"] is not None) is not fv11["at_30_experts"]["refused"]:
        findings.append("v11: the EPLB refusal fired with expert parallelism off")
    if v11["owned_at_30"] != list(range(30)):
        findings.append("v11: ownership at 30 experts")

    # V12
    fv12 = frozen_cells["v12_wire_identity"]
    v12 = cells["v12_wire_identity"]
    for label in fv12["manifests"]:
        row = v12[label]
        if not row["round_trips"]:
            findings.append(f"v12: {label} round trip")
        if row["source"] != fv12["source"]:
            findings.append(f"v12: {label} source")
        if not row["layer_keys_ascending"]:
            findings.append(f"v12: {label} layer key order")
        for order in row["group_key_orders"]:
            expected = [key for key in fv12["group_key_order"] if key in order]
            if list(order) != expected:
                findings.append(f"v12: {label} group key order")

    # V13, a recorded finding rather than a guard on this slice
    v13 = cells["v13_consumer_finding"]
    if v13["ownership_entries"] != v13_frozen["ownership_entries"]:
        findings.append("v13: ownership entries")
    if v13["distinct_pairs"] != v13_frozen["distinct_layer_expert_pairs"]:
        findings.append("v13: distinct pairs")
    if v13["snapshot_refusal"] != v13_frozen["snapshot_over_full_ep_group"]["error"]:
        findings.append("v13: the snapshot refusal message")
    if v13["single_rank_slice_entries"] != v13_frozen[
        "snapshot_over_single_rank_slice"
    ]["entries"]:
        findings.append("v13: single-rank slice entries")
    if v13["expert_parallel_on_twin_entries"] != v13_frozen[
        "snapshot_over_expert_parallel_on_twin"
    ]["entries"]:
        findings.append("v13: expert-parallel-on twin entries")

    # The fatal identities
    frozen_records = frozen["baseline"]["placement_records"]
    for label, expected in frozen_records.items():
        row = cells["compatibility_digests"].get(label, {})
        if row.get("bytes") != expected["bytes"]:
            findings.append(f"digest {label}: bytes")
        if row.get("sha256") != expected["sha256"]:
            findings.append(f"digest {label}: sha256")
    for label, key in (
        ("place3_study_check", "place3_study_check_reproduces"),
        ("place13_study_check", "place13_study_check_reproduces"),
    ):
        if cells[label]["reproduces"] is not frozen["baseline"][key]:
            findings.append(f"the {label} did not reproduce")

    return {
        "evidence": {
            "fatal_compatibility_digests": len(frozen_records),
            "fatal_off_path_defaults": 1,
            "fatal_study_checks": 2,
            "recorded_findings": 1,
            "rejection_controls": len(cells["v9_refusals"]["refusals"]),
            "scored_exact_oracle_rows": 0,
            "structural_guard_cells": len(frozen["evidence"]["structural_exact"]),
        },
        "findings": findings,
        "physical_sanity": {
            "v1_ownership_entries": cells["v1_ep_disabled_worked_example"][
                "owner_entries_total"
            ],
            "v1_conservation_rule": "num_experts times the stage MoE layer count times DP x TP, summed over stages",
            "v13_ownership_entries": v13["ownership_entries"],
            "v13_distinct_pairs": v13["distinct_pairs"],
        },
        "status": "VOID" if findings else "PASS",
    }


def run_study(output_root: Path) -> dict[str, Any]:
    """Execute every cell and return the complete tracked summary."""

    frozen = json.loads(EXPECTATIONS_PATH.read_text(encoding="utf-8"))
    v13_frozen = json.loads(V13_AMENDMENT_PATH.read_text(encoding="utf-8"))["v13"]
    cells = {
        "compatibility_digests": run_compatibility_digests(output_root),
        "place3_study_check": run_study_check(
            output_root, PLACE3_STUDY, "place3-study-check"
        ),
        "place13_study_check": run_study_check(
            output_root, PLACE13_STUDY, "place13-study-check"
        ),
        "v1_ep_disabled_worked_example": run_v1(),
        "v2_off_path_identity": run_v2(),
        "v3_group_inventory_gap": run_v3(),
        "v4_degenerate_identity": run_v4(output_root),
        "v5_round_robin_arange": run_v5(),
        "v6_sglang": run_v6(),
        "v7_fallback_rows": run_v7(),
        "v8_linear_untouched": run_v8(),
        "v9_refusals": run_v9(),
        "v10_eplb_group": run_v10(),
        "v11_composed": run_v11(),
        "v12_wire_identity": run_v12(output_root),
        "v13_consumer_finding": run_v13(),
    }
    observation = {"cells": cells, "expectations_commit": EXPECTATIONS_COMMIT}
    analysis = analyze_observation(observation, frozen, v13_frozen)
    return {
        "amendment_commits": {
            "baseline": BASELINE_AMENDMENT_COMMIT,
            "cell_v13": V13_AMENDMENT_COMMIT,
        },
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
    parser = argparse.ArgumentParser(
        description="Run the PLACE-7 and PLACE-11 qualification"
    )
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
                "physical_sanity": summary["physical_sanity"],
                "status": summary["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    for label in ("place3_study_check", "place13_study_check"):
        print(f"{label} reproduces={summary['cells'][label]['reproduces']}")

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
