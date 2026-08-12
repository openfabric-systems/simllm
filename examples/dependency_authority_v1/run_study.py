"""Run the frozen TRAF-12 dependency-authority study and TRAF-27 refreeze."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, replace
from itertools import pairwise
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

EVIDENCE_AUTHORED_AGAINST = "dcbef8682b1d74fb059a95d5b8b6f0c4ae07c9eb"
REFREEZE_EVIDENCE_AUTHORED_AGAINST = (
    "14d8447b838e651f8321ffb0588ea02219e26e9a"
)
VECTOR_BYTES = (1_024, 2_048)
PLACEMENTS = ("AAAA", "AABB", "ABCD")
PHASE_COUNT = 48
ADJACENT_TRANSITIONS = 47
GRAPH_ARTIFACT_COUNT = 72
GRAPH_ARTIFACT_OPERATION_COUNTS = (4, 1, 1) * 24
PRE_OWNERSHIP_DIRECT_JCT_PS = {
    1_024: 156_569_755,
    2_048: 217_222_486,
}
DIRECT_JCT_PS = {
    1_024: 150_838_767,
    2_048: 205_653_487,
}
EXPECTED_JCT_BANDS = {
    (1_024, "AAAA"): (6_676_000, 6_676_000),
    (1_024, "AABB"): (136_246_720, 136_246_720),
    (1_024, "ABCD"): (155_702_720, 155_702_864),
    (2_048, "AAAA"): (13_310_000, 13_310_000),
    (2_048, "AABB"): (176_469_440, 176_469_440),
    (2_048, "ABCD"): (215_381_440, 215_381_584),
}
#: represented compute the all-local cells carry outside NVLink service
ALL_LOCAL_COMPUTE_PS = 24_000
EXPECTED_SIGNED_CHANGE_BANDS = {
    1_024: (4_863_953, 4_864_097),
    2_048: (9_727_953, 9_728_097),
}
DIRECT_GOAL_ORACLES = {
    1_024: (
        20_392,
        "917961edf996753223857d64010fc61e4f6b08672f18dcadf42c70d60ee36c4a",
    ),
    2_048: (
        20_392,
        "16ee686eda4634886b117788b3893c893f5e12ea819736e0afdbdf63bab0e826",
    ),
}
FROZEN_GRAPH_CENSUS = {
    "operations": 144,
    "effective_edges": 423,
    "whole_operation_edges": 139,
    "participant_local_edges": 284,
    "explicit_participant_local_references": 212,
    "implicit_fifo_edges": 139,
    "distributed_fifo_edges": 47,
    "artifacts": 72,
    "boundaries": 47,
    "serialized_edges": 376,
    "serialized_participant_explicit_edges": 284,
    "serialized_whole_fifo_edges": 92,
    "backend_artifacts": 48,
    "positive_flows": 144,
}
TRACE_SHA256 = "36334f3aaa767c46d5f9c8498e02f6c2805a46e5000a57aea2747e17dd5d1341"
EXPECTATIONS_COMMIT = "d39dfdc2951e147187446e27c46d9ed3f1a6816a"
CROSS_CHECK_EXPECTATIONS_COMMIT = "69a7ada2ec192b3d7eec81b53529a5662371e3b1"
REFREEZE_EXPECTATIONS_COMMIT = "bf6780f21c3029b3dbc06c1ea1868c1eeb03ec97"
HTSIM_COMPILER_EVIDENCE_AUTHORED_AGAINST = (
    "034e2419f061f872ece400b7280319290c7589d9"
)
EXPECTED_BEHAVIORAL_FAMILIES = 2
EXPECTED_BEHAVIORAL_INSTANCES = 3
CROSS_CHECK_MODE = "atlahs-goal"
CROSS_CHECK_COMPLETION_TOLERANCE_PS = 0
EXPECTED_CROSS_CHECK_FINDINGS = {
    1_024: {
        "ordering_scope_differences": 47,
        "full_ordering_edge_count": 423,
        "full_ordering_scope_differences": 94,
        "participant_local_scope_differences": 47,
        "whole_fifo_scope_differences": 47,
        "negative_phase_frontiers": 32,
        "first_phase_frontier_gap_ps": -81_920,
        "minimum_phase_frontier_gap_ps": -716_800,
        "direct_completion_ps": 150_838_767,
        "signed_completion_difference_band_ps": (-4_864_097, -4_863_953),
    },
    2_048: {
        "ordering_scope_differences": 47,
        "full_ordering_edge_count": 423,
        "full_ordering_scope_differences": 94,
        "participant_local_scope_differences": 47,
        "whole_fifo_scope_differences": 47,
        "negative_phase_frontiers": 32,
        "first_phase_frontier_gap_ps": -163_840,
        "minimum_phase_frontier_gap_ps": -1_433_600,
        "direct_completion_ps": 205_653_487,
        "signed_completion_difference_band_ps": (-9_728_097, -9_727_953),
    },
}
EXPECTED_CELLS = {
    (1_024, "AAAA"): (2_983_936, 0, 2_983_936, 6_652_000),
    (1_024, "AABB"): (2_983_936, 2_011_136, 972_800, 2_194_000),
    (1_024, "ABCD"): (2_983_936, 2_983_936, 0, 0),
    (2_048, "AAAA"): (5_967_872, 0, 5_967_872, 13_286_000),
    (2_048, "AABB"): (5_967_872, 4_022_272, 1_945_600, 4_358_000),
    (2_048, "ABCD"): (5_967_872, 5_967_872, 0, 0),
}
#: cells refrozen by CORE-41 from the old maximum-source-egress surrogate
#: 4,538,000 ps and 9,047,000 ps to the maximum-endpoint-load charge
ENDPOINT_REFROZEN_CELLS = ((1_024, "AAAA"), (2_048, "AAAA"))
SUPERSEDED_SOURCE_EGRESS_SERVICE_PS = {
    (1_024, "AAAA"): 4_538_000,
    (2_048, "AAAA"): 9_047_000,
}
PHYSICAL_SANITY_BOUNDS_PS = {
    1_024: {
        "peak_egress_serialization_floor": 29_839_360,
        "phase_chain_floor": 155_702_720,
        "conservative_ceiling": 347_702_720,
    },
    2_048: {
        "peak_egress_serialization_floor": 59_678_720,
        "phase_chain_floor": 215_381_440,
        "conservative_ceiling": 407_381_440,
    },
}
SOURCE_ARTIFACTS = {
    "capture": (
        "capture/granite-greedy.jsonl",
        "5f55fccc265ee5519430a0f73d6631e49aa547ec07fcb81034e9fc2b4d9fead6",
    ),
    "steps": (
        "replay-400g/steps.jsonl",
        "824cd9557293328bb42b593ac893b6a067302e545b087c9219195ccb8031d755",
    ),
    "routing": (
        "replay-400g/routed-experts.json",
        "24e986e989e21f1bfe7e758d4470928c82c3bbaec06072a839743b9b17d7cf5f",
    ),
}


def _check_frozen_registry() -> None:
    expected_keys = {
        (vector_bytes, placement)
        for vector_bytes in VECTOR_BYTES
        for placement in PLACEMENTS
    }
    if set(EXPECTED_JCT_BANDS) != expected_keys:
        raise AssertionError("JCT grid is incomplete")
    if set(EXPECTED_SIGNED_CHANGE_BANDS) != set(VECTOR_BYTES):
        raise AssertionError("signed-change grid is incomplete")
    if set(DIRECT_GOAL_ORACLES) != set(VECTOR_BYTES):
        raise AssertionError("direct GOAL grid is incomplete")
    if set(EXPECTED_CELLS) != expected_keys:
        raise AssertionError("exact locality grid is incomplete")
    if set(EXPECTED_CROSS_CHECK_FINDINGS) != set(VECTOR_BYTES):
        raise AssertionError("cross-check finding grid is incomplete")
    if CROSS_CHECK_MODE != "atlahs-goal":
        raise AssertionError("cross-check selection literal drifted")
    if CROSS_CHECK_COMPLETION_TOLERANCE_PS != 0:
        raise AssertionError("cross-check completion tolerance drifted")
    if PHASE_COUNT - 1 != ADJACENT_TRANSITIONS:
        raise AssertionError("adjacent-transition count drifted")
    if FROZEN_GRAPH_CENSUS["distributed_fifo_edges"] != ADJACENT_TRANSITIONS:
        raise AssertionError("distributed FIFO census drifted")
    if FROZEN_GRAPH_CENSUS["artifacts"] != GRAPH_ARTIFACT_COUNT:
        raise AssertionError("graph artifact census drifted")
    if FROZEN_GRAPH_CENSUS["boundaries"] != ADJACENT_TRANSITIONS:
        raise AssertionError("artifact boundary census drifted")
    if FROZEN_GRAPH_CENSUS["positive_flows"] != 3 * PHASE_COUNT:
        raise AssertionError("positive-flow arithmetic drifted")
    if set(ENDPOINT_REFROZEN_CELLS) - expected_keys:
        raise AssertionError("refrozen locality cell is outside the sweep")
    if set(SUPERSEDED_SOURCE_EGRESS_SERVICE_PS) != set(ENDPOINT_REFROZEN_CELLS):
        raise AssertionError("superseded service grid is incomplete")
    for key in ENDPOINT_REFROZEN_CELLS:
        service_ps = EXPECTED_CELLS[key][3]
        if service_ps <= SUPERSEDED_SOURCE_EGRESS_SERVICE_PS[key]:
            raise AssertionError("the endpoint charge must exceed its surrogate")
        low, high = EXPECTED_JCT_BANDS[key]
        if low != high or low - service_ps != ALL_LOCAL_COMPUTE_PS:
            raise AssertionError("all-local compute term drifted")
    if any(len(digest) != 64 for _, digest in SOURCE_ARTIFACTS.values()):
        raise AssertionError("external source provenance is malformed")
    for vector_bytes in VECTOR_BYTES:
        low, high = EXPECTED_JCT_BANDS[(vector_bytes, "ABCD")]
        delta_low, delta_high = EXPECTED_SIGNED_CHANGE_BANDS[vector_bytes]
        if (low - DIRECT_JCT_PS[vector_bytes], high - DIRECT_JCT_PS[vector_bytes]) != (
            delta_low,
            delta_high,
        ):
            raise AssertionError("signed-change arithmetic drifted")
        if delta_low <= 0 or delta_high < delta_low:
            raise AssertionError("signed-change direction or band is invalid")
        ordered = [
            EXPECTED_JCT_BANDS[(vector_bytes, placement)][0]
            for placement in PLACEMENTS
        ]
        if not ordered[0] < ordered[1] < ordered[2]:
            raise AssertionError("node-span JCT direction is not increasing")
        size, digest = DIRECT_GOAL_ORACLES[vector_bytes]
        if size <= 0 or len(digest) != 64:
            raise AssertionError("direct GOAL oracle is malformed")
        if DIRECT_JCT_PS[vector_bytes] >= PRE_OWNERSHIP_DIRECT_JCT_PS[vector_bytes]:
            raise AssertionError("direct JCT correction lost its predicted direction")
        total_bytes = EXPECTED_CELLS[(vector_bytes, "ABCD")][0]
        bounds = PHYSICAL_SANITY_BOUNDS_PS[vector_bytes]
        expected_phase_floor = 24_000 + PHASE_COUNT * 2_000_000 + 20 * total_bytes
        expected_ceiling = 24_000 + 144 * 2_000_000 + 20 * total_bytes
        if bounds["phase_chain_floor"] != expected_phase_floor:
            raise AssertionError("phase-chain physical floor arithmetic drifted")
        if bounds["conservative_ceiling"] != expected_ceiling:
            raise AssertionError("physical ceiling arithmetic drifted")
        if low != bounds["phase_chain_floor"] or high - low != 144:
            raise AssertionError("graph JCT band no longer follows the frozen physics")
        if not (
            bounds["peak_egress_serialization_floor"]
            <= DIRECT_JCT_PS[vector_bytes]
            <= bounds["conservative_ceiling"]
            and bounds["phase_chain_floor"] <= low <= high
            <= bounds["conservative_ceiling"]
        ):
            raise AssertionError("JCT prediction is outside its physical bounds")
        finding = EXPECTED_CROSS_CHECK_FINDINGS[vector_bytes]
        if finding["ordering_scope_differences"] != ADJACENT_TRANSITIONS:
            raise AssertionError("cross-check ordering census drifted")
        if finding["full_ordering_edge_count"] != FROZEN_GRAPH_CENSUS[
            "effective_edges"
        ]:
            raise AssertionError("cross-check edge census drifted")
        if finding["full_ordering_scope_differences"] != (
            finding["participant_local_scope_differences"]
            + finding["whole_fifo_scope_differences"]
        ):
            raise AssertionError("cross-check disagreement classes do not sum")
        if finding["whole_fifo_scope_differences"] != ADJACENT_TRANSITIONS:
            raise AssertionError("whole-FIFO cross-check census drifted")
        if finding["participant_local_scope_differences"] != ADJACENT_TRANSITIONS:
            raise AssertionError("participant-local cross-check census drifted")
        if not 0 < finding["negative_phase_frontiers"] < ADJACENT_TRANSITIONS:
            raise AssertionError("cross-check frontier census drifted")
        if finding["first_phase_frontier_gap_ps"] >= 0:
            raise AssertionError("first cross-check frontier gap lost its sign")
        if finding["minimum_phase_frontier_gap_ps"] > finding[
            "first_phase_frontier_gap_ps"
        ]:
            raise AssertionError("minimum cross-check frontier gap is invalid")
        if finding["direct_completion_ps"] != DIRECT_JCT_PS[vector_bytes]:
            raise AssertionError("direct cross-check completion drifted")
        signed_low, signed_high = finding[
            "signed_completion_difference_band_ps"
        ]
        original_low, original_high = EXPECTED_SIGNED_CHANGE_BANDS[vector_bytes]
        if (signed_low, signed_high) != (-original_high, -original_low):
            raise AssertionError("cross-check signed band arithmetic drifted")


def check_only(args: argparse.Namespace) -> None:
    _check_frozen_registry()
    print(
        f"check-only source_root={args.source_root} out={args.out}; "
        "validated frozen literals and produced no artifacts"
    )


def _git_revision(*args: str) -> str:
    return subprocess.run(
        ("git", "rev-parse", *args),
        check=True,
        capture_output=True,
        text=True,
        cwd=REPOSITORY_ROOT,
    ).stdout.strip()


def _require_clean_worktree() -> None:
    status = subprocess.run(
        ("git", "status", "--porcelain", "--untracked-files=all"),
        check=True,
        capture_output=True,
        text=True,
        cwd=REPOSITORY_ROOT,
    ).stdout
    if status:
        raise RuntimeError(
            "production evidence requires a clean worktree so the recorded "
            "SimLLM revision identifies the executed source"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_source_artifacts(source_root: Path) -> dict[str, object]:
    observations: dict[str, object] = {}
    for name, (relative, expected_digest) in SOURCE_ARTIFACTS.items():
        path = source_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"required source artifact is missing: {relative}")
        observed_digest = _sha256(path)
        if observed_digest != expected_digest:
            raise AssertionError(f"source artifact changed: {relative}")
        observations[name] = {
            "relative_path": relative,
            "bytes": path.stat().st_size,
            "sha256": observed_digest,
        }
    return observations


def _run_cell(
    out: Path,
    *,
    vector_bytes: int,
    placement: str,
    supply,
) -> tuple[dict[str, object], object]:
    from examples.nvlink_locality_v1.run_study import (
        NVLINK_BYTES_PER_SECOND,
        _dims,
        _physical_manifest,
        _record,
    )
    from examples.nvlink_locality_v1.run_study import (
        PLACEMENTS as HOSTS,
    )
    from simllm.backends import HtsimStepSink, HtsimStepSinkConfig
    from simllm.compute import ComputeProvider, DurationEstimate

    class FixedProvider(ComputeProvider):
        def estimate(self, kernel, gpu):
            return DurationEstimate(duration_ps=24_000, bound="declared-fixed")

    sink = HtsimStepSink(
        HtsimStepSinkConfig(
            profile="rnic-nn-fluid",
            tp_ranks=(0,),
            dims=_dims(vector_bytes),
            workdir=out / f"vector-{vector_bytes}" / placement,
            ep_ranks=(0, 1, 2, 3),
            linkspeed_bps=400_000_000_000,
            provider=FixedProvider(),
            routed_moe_supply=supply,
            placement_manifest=_physical_manifest(HOSTS[placement]),
            nvlink_bandwidth_bytes_per_second=NVLINK_BYTES_PER_SECOND,
            num_goal_ranks=4,
        )
    )
    virtual_time_ps = 0
    results = []
    for step_index in range(3):
        result = sink(_record(step_index, virtual_time_ps))
        if result is None:
            raise AssertionError("frozen collective did not reach StepResult")
        results.append(result)
        virtual_time_ps = result.completed_at_ps
    latencies = tuple(result.step_latency_ps for result in results)
    return (
        {
            "vector_bytes": vector_bytes,
            "placement": placement,
            "jct_ps": latencies[0],
            "ttft_ps": latencies[0],
            "tpot_ps": sum(latencies[1:]) // len(latencies[1:]),
            "replay_latencies_ps": list(latencies),
            "network": asdict(sink.outcomes[0]),
            "locality": asdict(sink.locality_outcomes[0]),
        },
        sink,
    )


def _run_cross_check(
    out: Path,
    *,
    vector_bytes: int,
    supply,
) -> tuple[object, object, Path]:
    from examples.nvlink_locality_v1.run_study import (
        PLACEMENTS as HOSTS,
    )
    from examples.nvlink_locality_v1.run_study import (
        _dims,
        _physical_manifest,
        _record,
    )
    from simllm.backends import HtsimStepSink, HtsimStepSinkConfig
    from simllm.compute import ComputeProvider, DurationEstimate

    class FixedProvider(ComputeProvider):
        def estimate(self, kernel, gpu):
            return DurationEstimate(duration_ps=24_000, bound="declared-fixed")

    workdir = out / "cross-check" / f"vector-{vector_bytes}"
    sink = HtsimStepSink(
        HtsimStepSinkConfig(
            profile="rnic-nn-fluid",
            tp_ranks=(0,),
            dims=_dims(vector_bytes),
            workdir=workdir,
            ep_ranks=(0, 1, 2, 3),
            linkspeed_bps=400_000_000_000,
            provider=FixedProvider(),
            routed_moe_supply=supply,
            placement_manifest=_physical_manifest(HOSTS["ABCD"]),
            num_goal_ranks=4,
            dependency_cross_check=CROSS_CHECK_MODE,
            dependency_cross_check_tolerance_ps=(
                CROSS_CHECK_COMPLETION_TOLERANCE_PS
            ),
        )
    )
    result = sink(_record(0, 0))
    if result is None:
        raise AssertionError("cross-check collective did not reach StepResult")
    if len(sink.dependency_cross_check_reports) != 1:
        raise AssertionError("one selected cross-check must produce one report")
    return result, sink.dependency_cross_check_reports[0], workdir


def _raw_tag_gaps(workdir: Path, locality: dict[str, object]) -> dict[str, object]:
    from simllm.backends import parse_completion_csv

    services = locality["composed_phase_service_ps"]
    absolute_rows = []
    artifact_offset_ps = 0
    for artifact_index, service_ps in enumerate(services):
        csv_path = workdir / (
            f"step-000000.artifact-{artifact_index:04d}.rnic-nn-fluid.csv"
        )
        if csv_path.exists():
            for flow in parse_completion_csv(csv_path):
                absolute_rows.append(
                    (
                        flow.tag,
                        artifact_offset_ps + flow.start_time_ps,
                        artifact_offset_ps + flow.completion_time_ps,
                    )
                )
        artifact_offset_ps += service_ps
    by_tag: dict[int, list[tuple[int, int]]] = {}
    for tag, started_at_ps, completed_at_ps in absolute_rows:
        by_tag.setdefault(tag, []).append((started_at_ps, completed_at_ps))
    tag_rows = [
        {
            "tag": tag,
            "minimum_start_ps": min(start for start, _ in rows),
            "maximum_completion_ps": max(completion for _, completion in rows),
        }
        for tag, rows in sorted(by_tag.items())
    ]
    gaps = [
        current["minimum_start_ps"] - previous["maximum_completion_ps"]
        for previous, current in pairwise(tag_rows)
    ]
    return {
        "tag_rows": tag_rows,
        "adjacent_gaps_ps": gaps,
        "early_transitions": sum(gap < 0 for gap in gaps),
        "minimum_gap_ps": min(gaps),
        "passed": len(gaps) == ADJACENT_TRANSITIONS and all(gap >= 0 for gap in gaps),
    }


def _negative_control() -> dict[str, object]:
    from simllm.core import CollectiveWork, ExecutionGraph, ExecutionOperation
    from simllm.traffic import (
        project_execution_graph_goal,
        verify_execution_goal_projection,
    )

    operations = tuple(
        ExecutionOperation(
            operation_id,
            0,
            "shared-nccl",
            CollectiveWork("all-reduce", (0, 1), 8, "ring"),
        )
        for operation_id in ("first", "second")
    )
    graph = ExecutionGraph("negative-control", 0, 0, operations, ("second",))
    projection = project_execution_graph_goal(graph)
    valid_accepted = True
    try:
        verify_execution_goal_projection(graph, projection)
    except (TypeError, ValueError):
        valid_accepted = False
    if len(projection.boundaries) != 1:
        raise AssertionError("negative control needs exactly one boundary")
    removed = projection.boundaries[0]
    if removed.edge.origin != "logical-queue-fifo":
        raise AssertionError("negative control did not select the FIFO boundary")
    perturbed = replace(projection, boundaries=())
    mutation_rejected = False
    rejection = None
    try:
        verify_execution_goal_projection(graph, perturbed)
    except ValueError as exc:
        rejection = str(exc)
        mutation_rejected = (
            "edge mismatch" in rejection
            and "missing=" in rejection
            and removed.edge.predecessor_id in rejection
            and removed.edge.operation_id in rejection
        )
    return {
        "valid_projection_accepted": valid_accepted,
        "mutation": "remove one nonredundant FIFO artifact boundary",
        "mutation_rejected": mutation_rejected,
        "rejection": rejection,
        "removed_edge": asdict(removed.edge),
        "passed": mutation_rejected,
    }


def _direct_goal_oracle(vector_bytes: int, supply) -> dict[str, object]:
    from examples.nvlink_locality_v1.run_study import _dims, _record
    from simllm.traffic import render_step_goal

    payload = render_step_goal(
        _record(0, 0),
        _dims(vector_bytes),
        (0,),
        per_layer_calc_ns=1,
        ep_ranks=(0, 1, 2, 3),
        routed_supply=supply,
        num_goal_ranks=4,
    ).render().encode()
    return {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _active_goal_manifest(workdir: Path) -> dict[str, object]:
    rows = []
    aggregate = hashlib.sha256()
    for path in sorted(workdir.glob("step-000000*.goal")):
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        rows.append({"name": path.name, "bytes": len(payload), "sha256": digest})
        aggregate.update(path.name.encode())
        aggregate.update(b"\0")
        aggregate.update(payload)
    return {
        "artifact_count": len(rows),
        "aggregate_sha256": aggregate.hexdigest(),
        "artifacts": rows,
    }


def _cross_check_finding(
    *,
    vector_bytes: int,
    authority_jct_ps: int,
    authority_flow_count: int,
    authority_edge_count: int,
    authority_workdir: Path,
    cross_check_result,
    report,
) -> dict[str, object]:
    expected = EXPECTED_CROSS_CHECK_FINDINGS[vector_bytes]
    direct_bytes, direct_sha256 = DIRECT_GOAL_ORACLES[vector_bytes]
    phase_comparisons = report.phase_frontier_comparisons
    evaluated_cross_gaps = tuple(
        comparison.cross_check_gap_ps
        for comparison in phase_comparisons
        if comparison.evaluated and comparison.cross_check_gap_ps is not None
    )
    negative_cross_gaps = tuple(
        gap_ps for gap_ps in evaluated_cross_gaps if gap_ps < 0
    )
    authority_manifest = _active_goal_manifest(authority_workdir)
    authority_artifacts = authority_manifest["artifacts"]
    expected_authority_names = tuple(
        artifact["name"] for artifact in authority_artifacts
    )
    expected_authority_sha256 = tuple(
        artifact["sha256"] for artifact in authority_artifacts
    )
    expected_authority_bytes = tuple(
        artifact["bytes"] for artifact in authority_artifacts
    )
    signed_low, signed_high = expected[
        "signed_completion_difference_band_ps"
    ]
    checks = {
        "explicit_mechanisms": (
            report.authority_mechanism == "execution-graph-projection"
            and report.cross_check_mechanism == "atlahs-independent-goal"
        ),
        "complete_ordering_inventory": (
            len(report.ordering_comparisons) == authority_edge_count
            and report.ordering_edge_count == authority_edge_count
        ),
        "registered_boundary_ordering_differences": (
            report.frontier_boundary_count == ADJACENT_TRANSITIONS
            and report.boundary_ordering_disagreement_count
            == expected["ordering_scope_differences"]
        ),
        "registered_full_ordering_census": (
            report.ordering_edge_count
            == expected["full_ordering_edge_count"]
            and report.ordering_disagreement_count
            == expected["full_ordering_scope_differences"]
            and report.ordering_disagreement_classes
            == (
                (
                    "participant-local",
                    "explicit",
                    expected["participant_local_scope_differences"],
                ),
                (
                    "whole-operation",
                    "logical-queue-fifo",
                    expected["whole_fifo_scope_differences"],
                ),
            )
        ),
        "complete_phase_frontier_inventory": (
            len(phase_comparisons) == ADJACENT_TRANSITIONS
            and len(evaluated_cross_gaps) == ADJACENT_TRANSITIONS
        ),
        "registered_negative_phase_frontiers": (
            report.phase_frontier_disagreement_count
            == expected["negative_phase_frontiers"]
            and len(negative_cross_gaps)
            == expected["negative_phase_frontiers"]
        ),
        "registered_first_phase_frontier_gap": (
            evaluated_cross_gaps[0]
            == expected["first_phase_frontier_gap_ps"]
        ),
        "registered_minimum_phase_frontier_gap": (
            min(evaluated_cross_gaps)
            == expected["minimum_phase_frontier_gap_ps"]
        ),
        "graph_authority_result_preserved": (
            cross_check_result.step_latency_ps == authority_jct_ps
            and report.authority_completion_ps == authority_jct_ps
        ),
        "registered_direct_completion": (
            report.cross_check_completion_ps
            == expected["direct_completion_ps"]
        ),
        "registered_signed_completion_difference": (
            signed_low
            <= report.signed_completion_difference_ps
            <= signed_high
        ),
        "registered_zero_tolerance_disagreement": (
            report.completion_tolerance_ps
            == CROSS_CHECK_COMPLETION_TOLERANCE_PS
            and report.completion_disagreement
            and report.has_disagreement
        ),
        "registered_direct_artifact_preserved": (
            report.cross_check_artifact_bytes == direct_bytes
            and report.cross_check_artifact_sha256 == direct_sha256
        ),
        "authority_artifacts_preserved": (
            report.authority_artifact_names == expected_authority_names
            and report.authority_artifact_sha256
            == expected_authority_sha256
            and report.authority_artifact_bytes == expected_authority_bytes
        ),
        "both_runs_quiescent": (
            report.authority_quiescent and report.cross_check_quiescent
        ),
        "complete_flow_inventories": (
            report.authority_flow_count == authority_flow_count
            and report.cross_check_flow_count == authority_flow_count
            and authority_flow_count > 0
        ),
    }
    return {
        "vector_bytes": vector_bytes,
        "registered": expected,
        "observed": {
            "ordering_scope_differences": (
                report.boundary_ordering_disagreement_count
            ),
            "full_ordering_edge_count": report.ordering_edge_count,
            "full_ordering_scope_differences": (
                report.ordering_disagreement_count
            ),
            "full_ordering_disagreement_classes": (
                report.ordering_disagreement_classes
            ),
            "negative_phase_frontiers": len(negative_cross_gaps),
            "first_phase_frontier_gap_ps": evaluated_cross_gaps[0],
            "minimum_phase_frontier_gap_ps": min(evaluated_cross_gaps),
            "direct_completion_ps": report.cross_check_completion_ps,
            "signed_completion_difference_ps": (
                report.signed_completion_difference_ps
            ),
            "authority_completion_ps": report.authority_completion_ps,
            "completion_tolerance_ps": report.completion_tolerance_ps,
        },
        "checks": checks,
        "passed": all(checks.values()),
        "report": asdict(report),
    }


def run_study(source_root: Path, out: Path) -> None:
    from examples.nvlink_locality_v1.run_study import (
        NVLINK_BYTES_PER_SECOND,
        _dims,
        _record,
        _routed_projection,
        _routed_supply,
        _trace_path,
    )
    from examples.nvlink_locality_v1.run_study import (
        PLACEMENTS as HOSTS,
    )
    from simllm.backends import (
        HtsimStepSink,
        HtsimStepSinkConfig,
        SerialStepLowerer,
        SerialStepLowererConfig,
        find_htsim_rnic,
    )
    from simllm.compute import ComputeProvider, DurationEstimate
    from simllm.core import effective_dependency_edges, operation_participant_ranks
    from simllm.goal import find_txt2bin
    from simllm.traffic import (
        plan_execution_graph_locality,
        project_execution_graph_goal,
        validate_execution_graph_locality_projection,
        verify_execution_goal_projection,
    )

    class FixedProvider(ComputeProvider):
        def estimate(self, kernel, gpu):
            return DurationEstimate(duration_ps=24_000, bound="declared-fixed")

    _check_frozen_registry()
    _require_clean_worktree()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing output {out}")
    source_observations = _validate_source_artifacts(source_root)
    trace_path = _trace_path()
    if _sha256(trace_path) != TRACE_SHA256:
        raise AssertionError("tracked Granite trace does not match its frozen digest")
    txt2bin = find_txt2bin()
    htsim = find_htsim_rnic()
    if txt2bin is None or htsim is None:
        raise RuntimeError("native binaries are required for the production study")
    out.mkdir(parents=True)
    supply = _routed_supply(_routed_projection(trace_path))

    cells: dict[tuple[int, str], dict[str, object]] = {}
    sinks = {}
    for vector_bytes in VECTOR_BYTES:
        for placement in PLACEMENTS:
            cell, sink = _run_cell(
                out,
                vector_bytes=vector_bytes,
                placement=placement,
                supply=supply,
            )
            cells[(vector_bytes, placement)] = cell
            sinks[(vector_bytes, placement)] = sink

    # Execute both mechanisms first, but do not apply any registered direct
    # completion, signed-band, artifact, or comparator oracle yet.
    raw_cross_checks = {}
    for vector_bytes in VECTOR_BYTES:
        raw_cross_checks[vector_bytes] = _run_cross_check(
            out,
            vector_bytes=vector_bytes,
            supply=supply,
        )

    signed_instances = []
    causal_instances = []
    for vector_bytes in VECTOR_BYTES:
        graph_observed = cells[(vector_bytes, "ABCD")]["jct_ps"]
        _, raw_report, _ = raw_cross_checks[vector_bytes]
        direct_observed = raw_report.cross_check_completion_ps
        delta = (
            graph_observed - direct_observed
            if direct_observed is not None
            else None
        )
        delta_low, delta_high = EXPECTED_SIGNED_CHANGE_BANDS[vector_bytes]
        signed_instances.append(
            {
                "vector_bytes": vector_bytes,
                "registered_direct_jct_ps": DIRECT_JCT_PS[vector_bytes],
                "observed_direct_jct_ps": direct_observed,
                "observed_graph_authority_jct_ps": graph_observed,
                "signed_change_ps": delta,
                "expected_signed_change_band_ps": [delta_low, delta_high],
                "passed": (
                    delta is not None and delta_low <= delta <= delta_high
                ),
            }
        )
        workdir = out / f"vector-{vector_bytes}" / "ABCD"
        causal = _raw_tag_gaps(
            workdir,
            cells[(vector_bytes, "ABCD")]["locality"],
        )
        causal["vector_bytes"] = vector_bytes
        causal_instances.append(causal)
    negative_control = _negative_control()
    behavioral = {
        "TRAF-B1-signed-jct": {
            "instances": signed_instances,
            "passed": all(row["passed"] for row in signed_instances),
        },
        "TRAF-B3-negative-control": {
            "instances": [negative_control],
            "passed": negative_control["passed"],
        },
    }

    # The decision-relevant relations above now contain both raw completion
    # observations. Only after scoring them do the exact and bounded
    # cross-check guards inspect those values.
    cross_check_findings = []
    for vector_bytes in VECTOR_BYTES:
        cross_check_result, report, _ = raw_cross_checks[vector_bytes]
        authority_cell = cells[(vector_bytes, "ABCD")]
        cross_check_findings.append(
            _cross_check_finding(
                vector_bytes=vector_bytes,
                authority_jct_ps=authority_cell["jct_ps"],
                authority_flow_count=authority_cell["network"]["num_flows"],
                authority_edge_count=authority_cell["locality"][
                    "effective_dependency_edge_count"
                ],
                authority_workdir=(
                    out / f"vector-{vector_bytes}" / "ABCD"
                ),
                cross_check_result=cross_check_result,
                report=report,
            )
        )

    physical_sanity = []
    for finding in cross_check_findings:
        vector_bytes = finding["vector_bytes"]
        bounds = PHYSICAL_SANITY_BOUNDS_PS[vector_bytes]
        graph_jct_ps = cells[(vector_bytes, "ABCD")]["jct_ps"]
        direct_jct_ps = finding["observed"]["direct_completion_ps"]
        physical_sanity.append(
            {
                "vector_bytes": vector_bytes,
                "peak_egress_serialization_floor_ps": bounds[
                    "peak_egress_serialization_floor"
                ],
                "graph_phase_chain_floor_ps": bounds["phase_chain_floor"],
                "conservative_ceiling_ps": bounds["conservative_ceiling"],
                "observed_direct_jct_ps": direct_jct_ps,
                "observed_graph_jct_ps": graph_jct_ps,
                "passed": (
                    bounds["peak_egress_serialization_floor"]
                    <= direct_jct_ps
                    <= bounds["conservative_ceiling"]
                    and bounds["phase_chain_floor"]
                    <= graph_jct_ps
                    <= bounds["conservative_ceiling"]
                ),
            }
        )

    exact_cells = []
    for key, cell in cells.items():
        vector_bytes, placement = key
        locality = cell["locality"]
        expected = EXPECTED_CELLS[key]
        low, high = EXPECTED_JCT_BANDS[key]
        observed_tuple = (
            locality["total_directed_bytes"],
            locality["fabric_directed_bytes"],
            locality["nvlink_directed_bytes"],
            locality["nvlink_service_ps"],
        )
        exact_cells.append(
            {
                "vector_bytes": vector_bytes,
                "placement": placement,
                "precision_status": (
                    "accepted-core41-refreeze"
                    if key in ENDPOINT_REFROZEN_CELLS
                    else "accepted"
                ),
                "superseded_source_egress_service_ps": (
                    SUPERSEDED_SOURCE_EGRESS_SERVICE_PS.get(key)
                ),
                "observed_locality": list(observed_tuple),
                "expected_locality": list(expected),
                "observed_jct_ps": cell["jct_ps"],
                "expected_jct_band_ps": [low, high],
                "passed": observed_tuple == expected
                and low <= cell["jct_ps"] <= high
                and cell["ttft_ps"] == cell["jct_ps"]
                and cell["tpot_ps"] == cell["jct_ps"]
                and cell["replay_latencies_ps"] == [cell["jct_ps"]] * 3,
            }
        )

    structural = []
    for vector_bytes in VECTOR_BYTES:
        lowerer = SerialStepLowerer(
            SerialStepLowererConfig(
                _dims(vector_bytes),
                (0,),
                ep_ranks=(0, 1, 2, 3),
                provider=FixedProvider(),
                routed_moe_supply=supply,
            )
        )
        graph = lowerer.lower(_record(0, 0))
        mapper = sinks[(vector_bytes, "ABCD")]._rank_mapper
        locality_plan = plan_execution_graph_locality(
            graph,
            rank_mapper=mapper,
        )
        validate_execution_graph_locality_projection(
            graph,
            locality_plan,
            rank_mapper=mapper,
        )
        goal_projection = project_execution_graph_goal(graph, num_goal_ranks=4)
        verify_execution_goal_projection(graph, goal_projection)
        edges = effective_dependency_edges(graph)
        positive_pair_count = sum(
            len(phase.phase.segments) for phase in locality_plan.phases
        )
        operation_by_id = {
            operation.operation_id: operation for operation in graph.operations
        }
        explicit_local_references = sum(
            len(operation.participant_local_depends_on)
            for operation in graph.operations
        )
        implicit_fifo_edges = sum(
            edge.origin.value == "logical-queue-fifo" for edge in edges
        )
        distributed_fifo_edges = sum(
            edge.origin.value == "logical-queue-fifo"
            and (
                len(
                    operation_participant_ranks(
                        operation_by_id[edge.predecessor_id]
                    )
                )
                > 1
                or len(
                    operation_participant_ranks(
                        operation_by_id[edge.operation_id]
                    )
                )
                > 1
            )
            for edge in edges
        )
        direct_oracle = _direct_goal_oracle(vector_bytes, supply)
        direct_size, direct_digest = DIRECT_GOAL_ORACLES[vector_bytes]
        active_manifest = _active_goal_manifest(
            out / f"vector-{vector_bytes}" / "ABCD"
        )
        boundary_whole_fifo = sum(
            boundary.edge.scope == "whole-operation"
            and boundary.edge.origin == "logical-queue-fifo"
            for boundary in goal_projection.boundaries
        )
        serialized_participant_explicit = sum(
            edge.scope == "participant-local" and edge.origin == "explicit"
            for edge in goal_projection.serialized_edges
        )
        serialized_whole_fifo = sum(
            edge.scope == "whole-operation"
            and edge.origin == "logical-queue-fifo"
            for edge in goal_projection.serialized_edges
        )
        outcome_locality = cells[(vector_bytes, "ABCD")]["locality"]
        structural.append(
            {
                "vector_bytes": vector_bytes,
                "operation_count": len(graph.operations),
                "effective_edge_count": len(edges),
                "positive_pair_count": positive_pair_count,
                "whole_operation_edges": sum(
                    edge.scope.value == "whole-operation" for edge in edges
                ),
                "participant_local_edges": sum(
                    edge.scope.value == "participant-local" for edge in edges
                ),
                "explicit_participant_local_references": explicit_local_references,
                "implicit_fifo_edges": implicit_fifo_edges,
                "distributed_fifo_edges": distributed_fifo_edges,
                "graph_artifact_count": len(goal_projection.artifacts),
                "graph_artifact_operation_counts": [
                    len(artifact.operation_ids)
                    for artifact in goal_projection.artifacts
                ],
                "artifact_boundary_edge_count": len(goal_projection.boundaries),
                "serialized_edge_count": len(goal_projection.serialized_edges),
                "boundary_whole_fifo_edges": boundary_whole_fifo,
                "serialized_participant_explicit_edges": (
                    serialized_participant_explicit
                ),
                "serialized_whole_fifo_edges": serialized_whole_fifo,
                "direct_goal": direct_oracle,
                "direct_goal_passed": direct_oracle
                == {"bytes": direct_size, "sha256": direct_digest},
                "active_goal_manifest": active_manifest,
                "ordering_authority": outcome_locality["ordering_authority"],
                "passed": len(graph.operations) == FROZEN_GRAPH_CENSUS["operations"]
                and positive_pair_count == FROZEN_GRAPH_CENSUS["positive_flows"]
                and explicit_local_references
                == FROZEN_GRAPH_CENSUS["explicit_participant_local_references"]
                and implicit_fifo_edges
                == FROZEN_GRAPH_CENSUS["implicit_fifo_edges"]
                and distributed_fifo_edges
                == FROZEN_GRAPH_CENSUS["distributed_fifo_edges"]
                and len(edges) == FROZEN_GRAPH_CENSUS["effective_edges"]
                and sum(edge.scope.value == "whole-operation" for edge in edges)
                == FROZEN_GRAPH_CENSUS["whole_operation_edges"]
                and sum(edge.scope.value == "participant-local" for edge in edges)
                == FROZEN_GRAPH_CENSUS["participant_local_edges"]
                and len(goal_projection.artifacts)
                == FROZEN_GRAPH_CENSUS["artifacts"]
                and [
                    len(artifact.operation_ids)
                    for artifact in goal_projection.artifacts
                ]
                == list(GRAPH_ARTIFACT_OPERATION_COUNTS)
                and len(goal_projection.boundaries)
                == FROZEN_GRAPH_CENSUS["boundaries"]
                and len(goal_projection.serialized_edges)
                == FROZEN_GRAPH_CENSUS["serialized_edges"]
                and boundary_whole_fifo
                == FROZEN_GRAPH_CENSUS["distributed_fifo_edges"]
                and serialized_participant_explicit
                == FROZEN_GRAPH_CENSUS[
                    "serialized_participant_explicit_edges"
                ]
                and serialized_whole_fifo
                == FROZEN_GRAPH_CENSUS["serialized_whole_fifo_edges"]
                and active_manifest["artifact_count"]
                == FROZEN_GRAPH_CENSUS["backend_artifacts"]
                and outcome_locality["effective_dependency_edge_count"]
                == len(edges)
                and outcome_locality["graph_artifact_count"]
                == len(goal_projection.artifacts)
                and outcome_locality["boundary_edge_count"]
                == len(goal_projection.boundaries)
                and outcome_locality["serialized_edge_count"]
                == len(goal_projection.serialized_edges)
                and direct_oracle
                == {"bytes": direct_size, "sha256": direct_digest}
                and outcome_locality["ordering_authority"] == "execution-graph",
            }
        )

    identity = []
    for vector_bytes in VECTOR_BYTES:
        explicit_dir = out / f"vector-{vector_bytes}" / "ABCD"
        omitted_dir = out / f"vector-{vector_bytes}" / "omitted-plan"
        omitted = HtsimStepSink(
            HtsimStepSinkConfig(
                profile="rnic-nn-fluid",
                tp_ranks=(0,),
                dims=_dims(vector_bytes),
                workdir=omitted_dir,
                ep_ranks=(0, 1, 2, 3),
                provider=FixedProvider(),
                routed_moe_supply=supply,
                num_goal_ranks=4,
            )
        )
        plan = omitted._plan_step(_record(0, 0))
        if plan is None:
            raise AssertionError("omitted all-remote plan unexpectedly empty")
        explicit_payloads = tuple(
            path.read_bytes() for path in sorted(explicit_dir.glob("step-000000*.goal"))
        )
        omitted_payloads = tuple(
            path.read_bytes() for path in sorted(omitted_dir.glob("step-000000*.goal"))
        )
        identity.append(
            {
                "vector_bytes": vector_bytes,
                "explicit_artifact_count": len(explicit_payloads),
                "omitted_artifact_count": len(omitted_payloads),
                "omitted_graph_artifact_count": plan.graph_artifact_count,
                "omitted_edge_count": plan.effective_dependency_edge_count,
                "passed": explicit_payloads == omitted_payloads
                and len(explicit_payloads) == PHASE_COUNT
                and plan.graph_artifact_count
                == FROZEN_GRAPH_CENSUS["artifacts"],
            }
        )

    fatal_passed = (
        all(row["passed"] for row in exact_cells)
        and all(row["passed"] for row in structural)
        and all(row["passed"] for row in identity)
        and all(row["passed"] for row in physical_sanity)
        and all(row["passed"] for row in causal_instances)
        and all(row["passed"] for row in cross_check_findings)
        and negative_control["valid_projection_accepted"]
        and all(
            cell["network"]["quiescent"]
            for cell in cells.values()
        )
    )
    passed_families = sum(row["passed"] for row in behavioral.values())
    passed_instances = sum(
        instance["passed"]
        for family in behavioral.values()
        for instance in family["instances"]
    )
    total_instances = sum(
        len(family["instances"]) for family in behavioral.values()
    )
    if len(behavioral) != EXPECTED_BEHAVIORAL_FAMILIES:
        raise AssertionError("behavioral family denominator drifted")
    if total_instances != EXPECTED_BEHAVIORAL_INSTANCES:
        raise AssertionError("behavioral instance denominator drifted")
    summary = {
        "schema": "simllm-dependency-authority-study-v1",
        "provenance": {
            "expectations_commit": EXPECTATIONS_COMMIT,
            "cross_check_expectations_commit": CROSS_CHECK_EXPECTATIONS_COMMIT,
            "refreeze_expectations_commit": REFREEZE_EXPECTATIONS_COMMIT,
            "simllm_evidence_authored_against": EVIDENCE_AUTHORED_AGAINST,
            "refreeze_evidence_authored_against": (
                REFREEZE_EVIDENCE_AUTHORED_AGAINST
            ),
            "htsim_compiler_evidence_authored_against": (
                HTSIM_COMPILER_EVIDENCE_AUTHORED_AGAINST
            ),
            "observed_simllm_revision": _git_revision("HEAD"),
            "observed_htsim_gitlink": _git_revision("HEAD:third_party/htsim"),
            "txt2bin_sha256": _sha256(txt2bin),
            "htsim_rnic_sha256": _sha256(htsim),
            "python": sys.version,
            "platform": platform.platform(),
            "trace_sha256": _sha256(trace_path),
            "external_source_observations": source_observations,
        },
        "configuration": {
            "vector_bytes": list(VECTOR_BYTES),
            "placements": {name: list(HOSTS[name]) for name in PLACEMENTS},
            "linkspeed_bps": 400_000_000_000,
            "profile": "rnic-nn-fluid",
            "num_layers": 24,
            "represented_compute_ps": 24_000,
            "tp_ranks": [0],
            "ep_ranks": [0, 1, 2, 3],
            "num_goal_ranks": 4,
            "nvlink_bandwidth_bytes_per_second": NVLINK_BYTES_PER_SECOND,
            "trace_sha256": TRACE_SHA256,
            "dependency_workload": "tracked-granite-length-cap",
            "external_source_role": "provenance-only",
            "replays_per_cell": 3,
            "cross_check_mode": CROSS_CHECK_MODE,
            "cross_check_replays_per_vector": 1,
            "cross_check_completion_tolerance_ps": (
                CROSS_CHECK_COMPLETION_TOLERANCE_PS
            ),
        },
        "cells": [cells[key] for key in sorted(cells)],
        "behavioral": behavioral,
        "behavioral_score": {
            "passed_families": passed_families,
            "total_families": len(behavioral),
            "passed_instances": passed_instances,
            "total_instances": total_instances,
        },
        "exact_oracle_rows": exact_cells,
        "structural_invariants": structural,
        "all_remote_identity": identity,
        "physical_sanity": physical_sanity,
        "cross_check_findings": cross_check_findings,
        "fatal_unscored": {
            "passed": fatal_passed,
            "composed_causal_guards": causal_instances,
            "valid_negative_control_projection_accepted": negative_control[
                "valid_projection_accepted"
            ],
            "note": (
                "Exact cells, projection inventories, direct hashes, authority "
                "labels, composed causal gaps, identity and quiescence are "
                "fatal and unscored. Cross-check completeness and agreement "
                "with the frozen disagreement findings are also fatal and "
                "unscored; the disagreements themselves are diagnostic "
                "findings, not API failures."
            ),
        },
        "entailment": (
            "Signed JCT was evaluated from live StepResult values and mutation "
            "rejection from the specifically diagnosed checker outcome before "
            "any cross-check equality or signed-band guard and before exact "
            "cells, inventories or hashes. The frozen causal-gap family "
            "was reclassified fatal-unscored because ordered artifact offsets "
            "make nonnegative cross-artifact gaps true by construction."
        ),
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if (
        passed_families != len(behavioral)
        or passed_instances != total_instances
        or not fatal_passed
    ):
        raise RuntimeError("TRAF-27 refreeze failed its frozen acceptance bar")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.check_only:
        check_only(args)
        return
    run_study(args.source_root, args.out)


if __name__ == "__main__":
    main()
