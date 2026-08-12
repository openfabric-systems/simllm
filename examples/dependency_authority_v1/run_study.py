"""Run the frozen TRAF-12 dependency-authority study after implementation."""

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
VECTOR_BYTES = (1_024, 2_048)
PLACEMENTS = ("AAAA", "AABB", "ABCD")
PHASE_COUNT = 48
ADJACENT_TRANSITIONS = 47
GRAPH_ARTIFACT_COUNT = 72
GRAPH_ARTIFACT_OPERATION_COUNTS = (4, 1, 1) * 24
LEGACY_JCT_PS = {
    1_024: 156_569_755,
    2_048: 217_222_486,
}
EXPECTED_JCT_BANDS = {
    (1_024, "AAAA"): (7_121_000, 7_121_000),
    (1_024, "AABB"): (139_195_840, 139_195_840),
    (1_024, "ABCD"): (160_781_760, 160_781_808),
    (2_048, "AAAA"): (14_180_000, 14_180_000),
    (2_048, "AABB"): (182_367_680, 182_367_680),
    (2_048, "ABCD"): (225_539_520, 225_539_568),
}
EXPECTED_SIGNED_CHANGE_BANDS = {
    1_024: (4_212_005, 4_212_053),
    2_048: (8_317_034, 8_317_082),
}
LEGACY_GOAL_ORACLES = {
    1_024: (
        72_819,
        "0417832c8788a0477d48b414cf2d8456b87215abd1d0193ba46fb8db46185d8a",
    ),
    2_048: (
        72_819,
        "bcd72e63546d03efaddd48c16e160457d1e28f19795036d1f871788d78cf5a02",
    ),
}
FROZEN_GRAPH_CENSUS = {
    "operations": 144,
    "explicit_participant_local_references": 212,
    "implicit_fifo_edges": 139,
    "distributed_fifo_edges": 47,
}
TRACE_SHA256 = "36334f3aaa767c46d5f9c8498e02f6c2805a46e5000a57aea2747e17dd5d1341"
EXPECTATIONS_COMMIT = "d39dfdc2951e147187446e27c46d9ed3f1a6816a"
CROSS_CHECK_EXPECTATIONS_COMMIT = "69a7ada2ec192b3d7eec81b53529a5662371e3b1"
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
        "negative_phase_frontiers": 46,
        "first_phase_frontier_gap_ps": -368_640,
        "minimum_phase_frontier_gap_ps": -1_413_120,
        "direct_completion_ps": 156_569_755,
        "signed_completion_difference_band_ps": (-4_212_053, -4_212_005),
    },
    2_048: {
        "ordering_scope_differences": 47,
        "negative_phase_frontiers": 46,
        "first_phase_frontier_gap_ps": -737_280,
        "minimum_phase_frontier_gap_ps": -3_675_091,
        "direct_completion_ps": 217_222_486,
        "signed_completion_difference_band_ps": (-8_317_082, -8_317_034),
    },
}
EXPECTED_CELLS = {
    (1_024, "AAAA"): (11_870_208, 0, 11_870_208, 7_097_000),
    (1_024, "AABB"): (11_870_208, 7_913_472, 3_956_736, 2_442_000),
    (1_024, "ABCD"): (11_870_208, 11_870_208, 0, 0),
    (2_048, "AAAA"): (23_740_416, 0, 23_740_416, 14_156_000),
    (2_048, "AABB"): (23_740_416, 15_826_944, 7_913_472, 4_838_000),
    (2_048, "ABCD"): (23_740_416, 23_740_416, 0, 0),
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
    if set(LEGACY_GOAL_ORACLES) != set(VECTOR_BYTES):
        raise AssertionError("legacy GOAL grid is incomplete")
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
    for vector_bytes in VECTOR_BYTES:
        low, high = EXPECTED_JCT_BANDS[(vector_bytes, "ABCD")]
        delta_low, delta_high = EXPECTED_SIGNED_CHANGE_BANDS[vector_bytes]
        if (low - LEGACY_JCT_PS[vector_bytes], high - LEGACY_JCT_PS[vector_bytes]) != (
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
        size, digest = LEGACY_GOAL_ORACLES[vector_bytes]
        if size <= 0 or len(digest) != 64:
            raise AssertionError("legacy GOAL oracle is malformed")
        finding = EXPECTED_CROSS_CHECK_FINDINGS[vector_bytes]
        if finding["ordering_scope_differences"] != ADJACENT_TRANSITIONS:
            raise AssertionError("cross-check ordering census drifted")
        if finding["negative_phase_frontiers"] != ADJACENT_TRANSITIONS - 1:
            raise AssertionError("cross-check frontier census drifted")
        if finding["first_phase_frontier_gap_ps"] >= 0:
            raise AssertionError("first cross-check frontier gap lost its sign")
        if finding["minimum_phase_frontier_gap_ps"] > finding[
            "first_phase_frontier_gap_ps"
        ]:
            raise AssertionError("minimum cross-check frontier gap is invalid")
        if finding["direct_completion_ps"] != LEGACY_JCT_PS[vector_bytes]:
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
        f"check-only out={args.out}; validated frozen literals and produced no artifacts"
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


def _legacy_goal_oracle(vector_bytes: int, supply) -> dict[str, object]:
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
    legacy_bytes, legacy_sha256 = LEGACY_GOAL_ORACLES[vector_bytes]
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
        "post_specified_full_ordering_census": (
            report.ordering_edge_count == 423
            and report.ordering_disagreement_count == 235
            and report.ordering_disagreement_classes
            == (
                ("participant-local", "explicit", 188),
                ("whole-operation", "logical-queue-fifo", 47),
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
        "legacy_direct_artifact_preserved": (
            report.cross_check_artifact_bytes == legacy_bytes
            and report.cross_check_artifact_sha256 == legacy_sha256
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


def run_study(out: Path) -> None:
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

    signed_instances = []
    causal_instances = []
    for vector_bytes in VECTOR_BYTES:
        observed = cells[(vector_bytes, "ABCD")]["jct_ps"]
        delta = observed - LEGACY_JCT_PS[vector_bytes]
        delta_low, delta_high = EXPECTED_SIGNED_CHANGE_BANDS[vector_bytes]
        signed_instances.append(
            {
                "vector_bytes": vector_bytes,
                "historical_jct_ps": LEGACY_JCT_PS[vector_bytes],
                "reconciled_jct_ps": observed,
                "signed_change_ps": delta,
                "expected_signed_change_band_ps": [delta_low, delta_high],
                "passed": delta_low <= delta <= delta_high,
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

    # Score decision-relevant relations before cross-check guards that inspect
    # the same completion values or registered signed-change bands.
    cross_check_findings = []
    for vector_bytes in VECTOR_BYTES:
        cross_check_result, report, _ = _run_cross_check(
            out,
            vector_bytes=vector_bytes,
            supply=supply,
        )
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
        legacy_oracle = _legacy_goal_oracle(vector_bytes, supply)
        legacy_size, legacy_digest = LEGACY_GOAL_ORACLES[vector_bytes]
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
                "legacy_direct_goal": legacy_oracle,
                "legacy_direct_goal_passed": legacy_oracle
                == {"bytes": legacy_size, "sha256": legacy_digest},
                "active_goal_manifest": active_manifest,
                "ordering_authority": outcome_locality["ordering_authority"],
                "passed": len(graph.operations) == FROZEN_GRAPH_CENSUS["operations"]
                and explicit_local_references
                == FROZEN_GRAPH_CENSUS["explicit_participant_local_references"]
                and implicit_fifo_edges
                == FROZEN_GRAPH_CENSUS["implicit_fifo_edges"]
                and distributed_fifo_edges
                == FROZEN_GRAPH_CENSUS["distributed_fifo_edges"]
                and len(edges) == 423
                and len(goal_projection.artifacts) == GRAPH_ARTIFACT_COUNT
                and [
                    len(artifact.operation_ids)
                    for artifact in goal_projection.artifacts
                ]
                == list(GRAPH_ARTIFACT_OPERATION_COUNTS)
                and len(goal_projection.boundaries) == ADJACENT_TRANSITIONS
                and len(goal_projection.serialized_edges) == 376
                and boundary_whole_fifo == ADJACENT_TRANSITIONS
                and serialized_participant_explicit == 284
                and serialized_whole_fifo == 92
                and active_manifest["artifact_count"] == PHASE_COUNT
                and outcome_locality["effective_dependency_edge_count"]
                == len(edges)
                and outcome_locality["graph_artifact_count"]
                == len(goal_projection.artifacts)
                and outcome_locality["boundary_edge_count"]
                == len(goal_projection.boundaries)
                and outcome_locality["serialized_edge_count"]
                == len(goal_projection.serialized_edges)
                and legacy_oracle
                == {"bytes": legacy_size, "sha256": legacy_digest}
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
                and plan.graph_artifact_count == GRAPH_ARTIFACT_COUNT,
            }
        )

    fatal_passed = (
        all(row["passed"] for row in exact_cells)
        and all(row["passed"] for row in structural)
        and all(row["passed"] for row in identity)
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
            "simllm_evidence_authored_against": EVIDENCE_AUTHORED_AGAINST,
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
        "cross_check_findings": cross_check_findings,
        "fatal_unscored": {
            "passed": fatal_passed,
            "composed_causal_guards": causal_instances,
            "valid_negative_control_projection_accepted": negative_control[
                "valid_projection_accepted"
            ],
            "note": (
                "Exact cells, projection inventories, legacy hashes, authority "
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
        raise RuntimeError("TRAF-12 study failed its frozen acceptance bar")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.check_only:
        check_only(args)
        return
    run_study(args.out)


if __name__ == "__main__":
    main()
