"""Run the frozen CORE-46 scalar-projection study.

The scalar compatibility fields of the coarse runtime report are checked
against the participant-keyed critical segments that own conservation. Every
scored relation is recomputed here, independently of the reducer, and read
before the reducer validates the same report.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

EXPECTATIONS_COMMIT = "5d3e5ab8bb6b65eeb2378453f5e51ec6b858b4ac"
EVIDENCE_AUTHORED_AGAINST = "b529f2953d1c5ac2a44f4de79fef1b0d7ee00da5"

#: the accepted participant_frontier_v1 harness, reused so that the
#: preservation claim is about the accepted construction rather than a copy
FRONTIER_STUDY = "examples/participant_frontier_v1/run_study.py"

SHAPES = ("participant-local", "barrier")
REQUEST_COUNTS = (1, 3)

#: frozen in examples/participant_frontier_v1/expectations.md; nothing here is
#: recomputed from a run
PRESERVED_CELLS = {
    (1, "participant-local"): {
        "executions": 25,
        "completions": 5_760,
        "result_bytes": 30_399_320,
        "result_sha256": (
            "00cff9f56b550a166548e9c44e98d4dffe26c8102eb17b7a1fcdeda6e863fb94"
        ),
        "completion_bytes": 288_300,
        "completion_sha256": (
            "73b7415729185e9b4481561da8e6caff23487b68bb6a962f401bfe7052beb8b4"
        ),
    },
    (1, "barrier"): {
        "executions": 25,
        "completions": 5_760,
        "result_bytes": 30_399_320,
        "result_sha256": (
            "38cb6503f5475f2acd8071771c09119ddfa7ae4dc7af875169612b9375347420"
        ),
        "completion_bytes": 288_300,
        "completion_sha256": (
            "6f70c590af674dea6f9f24860e16fda3cf1f9a20eda4869ec0a34b027cb637af"
        ),
    },
    (3, "participant-local"): {
        "executions": 33,
        "completions": 7_680,
        "result_bytes": 44_179_494,
        "result_sha256": (
            "f58841e7747ae08fb41355e48a1aba30fdf9b12bb3b2e68642241550cf36115f"
        ),
        "completion_bytes": 386_327,
        "completion_sha256": (
            "1fcaf34da306efac867c27d45d0e2d0ae8975c7c692a34cafbf650b68adec6c7"
        ),
    },
    (3, "barrier"): {
        "executions": 33,
        "completions": 7_680,
        "result_bytes": 44_179_502,
        "result_sha256": (
            "66668afa531ab34054d2e4a3b3dc476d539600cc945ceb6188b87cbeb233f1a5"
        ),
        "completion_bytes": 386_327,
        "completion_sha256": (
            "dd2356365d657d9c0c1e4056b1677bf184d14060bac0827ac8c34cbbbb18125e"
        ),
    },
}

OUT_OF_ORDER_EXPECTATIONS = {
    "collective_participant_16_ps": 20,
    "collective_participant_0_ps": 20_000_000,
    "collective_participant_8_ps": 20_000_000,
    "collective_completed_ps": 20_000_000,
    "early_boundary_ps": 20,
    "early_completed_ps": 30,
    "late_boundary_ps": 20_000_000,
    "late_completed_ps": 20_000_005,
    "barrier_boundary_ps": 20_000_000,
    "barrier_completed_ps": 20_000_001,
    "result_boundary_ps": 20_000_005,
}
OUT_OF_ORDER_ADDITIVE = {
    "early": None,
    "late": "collective",
    "barrier": "collective",
}
NEGATIVE_CONTROLS = ("M1", "M2", "M3", "M4", "M5", "M6")
SERIALIZATION_PS_PER_BYTE = 20


def _check_frozen_registry() -> None:
    if set(PRESERVED_CELLS) != {
        (count, shape) for count in REQUEST_COUNTS for shape in SHAPES
    }:
        raise AssertionError("the preserved sweep must be two counts by two shapes")
    digests = set()
    for key, cell in PRESERVED_CELLS.items():
        for name in ("result_sha256", "completion_sha256"):
            if len(cell[name]) != 64:
                raise AssertionError(f"cell {key} has a malformed {name}")
        digests.add(cell["result_sha256"])
        if cell["result_bytes"] <= 0 or cell["completion_bytes"] <= 0:
            raise AssertionError(f"cell {key} has a nonpositive frozen size")
        if cell["executions"] <= 0 or cell["completions"] <= 0:
            raise AssertionError(f"cell {key} has a nonpositive frozen count")
    if len(digests) != len(PRESERVED_CELLS):
        raise AssertionError("the four cells must carry four distinct result digests")
    expectations = OUT_OF_ORDER_EXPECTATIONS
    if expectations["collective_participant_16_ps"] != SERIALIZATION_PS_PER_BYTE:
        raise AssertionError("the one-byte transfer must sit on its serialization floor")
    if expectations["collective_participant_0_ps"] != (
        1_000_000 * SERIALIZATION_PS_PER_BYTE
    ):
        raise AssertionError("the bulk transfer must sit on its serialization floor")
    if expectations["collective_participant_16_ps"] >= (
        expectations["collective_participant_0_ps"]
    ):
        raise AssertionError("the fixture must finish its ranks out of rank order")
    if expectations["early_completed_ps"] - expectations["early_boundary_ps"] != 10:
        raise AssertionError("the early successor must span its own compute only")
    if expectations["late_completed_ps"] - expectations["late_boundary_ps"] != 5:
        raise AssertionError("the late successor must span its own compute only")
    if expectations["barrier_completed_ps"] - expectations["barrier_boundary_ps"] != 1:
        raise AssertionError("the barrier successor must span its own compute only")
    if expectations["result_boundary_ps"] != expectations["late_completed_ps"]:
        raise AssertionError("the graph boundary must be the last required completion")
    if OUT_OF_ORDER_ADDITIVE["early"] is not None:
        raise AssertionError("a participant-local boundary is never additive")
    if set(NEGATIVE_CONTROLS) != {"M1", "M2", "M3", "M4", "M5", "M6"}:
        raise AssertionError("the registered negative-control set changed")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_revision(*args: str) -> str:
    return subprocess.run(
        ("git", "rev-parse", *args),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _frontier_module() -> Any:
    path = Path(__file__).resolve().parents[2] / FRONTIER_STUDY
    spec = importlib.util.spec_from_file_location("participant_frontier_v1", path)
    if spec is None or spec.loader is None:
        raise AssertionError("could not load the accepted frontier harness")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _derivation_errors(graph: Any, report: Any) -> list[str]:
    """Recompute the frozen derivation from the segments, independently.

    This runs before the reducer validates the same report, so the validator
    under test cannot entail the result.
    """

    errors: list[str] = []
    by_id = {record.operation_id: record for record in report.operations}
    for record in report.operations:
        name = record.operation_id
        segments = record.critical_segments
        if not segments:
            errors.append(f"{name}: no critical segment")
            continue
        completions = [segment.completed_at_ps for segment in segments]
        if record.physical_completed_at_ps != max(completions):
            errors.append(f"{name}: physical completion is not the segment maximum")
        if record.completed_at_ps not in completions:
            errors.append(f"{name}: completion is not a participant segment completion")
        predecessor_id = record.causal_predecessor_id
        boundary_ps = record.causal_predecessor_completed_at_ps
        if (predecessor_id is None) != (boundary_ps is None):
            errors.append(f"{name}: causal identity and boundary disagree on presence")
            continue
        if predecessor_id is None:
            if record.critical_predecessor_id is not None:
                errors.append(f"{name}: additive predecessor without a causal one")
            start_ps = graph.released_at_ps
        else:
            predecessor = by_id.get(predecessor_id)
            if predecessor is None:
                errors.append(f"{name}: causal predecessor is not in the report")
                continue
            predecessor_completions = {
                segment.completed_at_ps for segment in predecessor.critical_segments
            }
            if boundary_ps not in predecessor_completions:
                errors.append(
                    f"{name}: boundary is not a participant completion of "
                    f"{predecessor_id}"
                )
            if not any(
                segment.predecessor_operation_id == predecessor_id
                and segment.started_at_ps == boundary_ps
                for segment in segments
            ):
                errors.append(f"{name}: no own segment corroborates the boundary")
            expected = (
                predecessor_id
                if boundary_ps == predecessor.completed_at_ps
                else None
            )
            if record.critical_predecessor_id != expected:
                errors.append(
                    f"{name}: additive predecessor is "
                    f"{record.critical_predecessor_id!r}, derived {expected!r}"
                )
            start_ps = (
                boundary_ps
                if record.critical_predecessor_id is not None
                else graph.released_at_ps
            )
        if record.breakdown.operation_latency_ps != record.completed_at_ps - start_ps:
            errors.append(f"{name}: scalar breakdown does not span its own segment")
    chain = tuple(key[0] for key in report.realized_critical_path_segments)
    if chain != report.realized_critical_path_operation_ids:
        errors.append("realized critical-path projections disagree")
    return errors


def _local_frontier_records(report: Any) -> int:
    return sum(
        1
        for record in report.operations
        if record.causal_predecessor_id is not None
        and record.critical_predecessor_id is None
    )


def _run_granite_cell(
    frontier: Any,
    arguments: argparse.Namespace,
    external_run: Any,
    steps: tuple[Any, ...],
    request_count: int,
    shape: str,
) -> dict[str, Any]:
    from simllm.core import (
        CoarseDeviceProfile,
        CoarseDeviceRuntime,
        CompletionReducer,
        EventPhase,
        StepRecord,
        VirtualClock,
        execution_result_to_json,
    )

    directory = arguments.out / f"{shape}-requests-{request_count}"
    _joined, _arena, lifetimes, supply = frontier._open_cell(
        arguments,
        directory,
        external_run,
        steps,
        request_count,
    )
    lowerer = frontier._lowerer(supply)
    clock = VirtualClock(0)
    reducer = CompletionReducer(clock, lifetimes=lifetimes)
    runtime = CoarseDeviceRuntime(CoarseDeviceProfile())
    wanted = {f"r{index}" for index in range(request_count)}
    selected = steps[:25] if request_count == 1 else steps

    def drain_record() -> Any:
        return StepRecord(
            step_index=32,
            virtual_time_ps=clock.now_ps,
            finished_request_ids=["r2"],
            num_sampled=0,
            sampled_request_ids=[],
        )

    pending: list[Any] = [
        lambda source=source: frontier._filtered_record(source, wanted, clock.now_ps)
        for source in selected
    ]
    if request_count != 1:
        pending.append(drain_record)

    payloads: list[Any] = []
    completion_rows: list[list[Any]] = []
    identities: set[tuple[int, str, str | None]] = set()
    derivation_errors: list[str] = []
    operation_records = 0
    causal_records = 0
    additive_records = 0
    local_frontier_records = 0
    early_boundary_records = 0
    for make_record in pending:
        record = make_record()
        graph = lowerer.lower(record)
        if shape == "barrier":
            graph = frontier._barrier_projection(graph)
        execution = runtime.execute(graph)
        report = runtime.last_report
        assert report is not None

        # Scored measurement first: the reducer has not seen this report yet.
        derivation_errors.extend(_derivation_errors(graph, report))
        operation_records += len(report.operations)
        causal_records += sum(
            1
            for item in report.operations
            if item.causal_predecessor_id is not None
        )
        additive_records += sum(
            1
            for item in report.operations
            if item.critical_predecessor_id is not None
        )
        local_frontier_records += _local_frontier_records(report)
        early_boundary_records += sum(
            1
            for item in report.operations
            if item.completed_at_ps
            != max(
                segment.completed_at_ps for segment in item.critical_segments
            )
        )

        reducer.reduce(record, graph, execution, report)
        payloads.append(execution_result_to_json(execution))
        for event in execution.events:
            if event.phase is not EventPhase.COMPLETED:
                continue
            completion_rows.append(
                [event.operation_id, event.subject_object_id, event.timestamp_ps]
            )
            identity = (record.step_index, event.operation_id, event.subject_object_id)
            if identity in identities:
                raise AssertionError("duplicate completion identity in one cell")
            identities.add(identity)

    result_bytes = (frontier._compact(payloads) + "\n").encode()
    completion_bytes = (frontier._compact(completion_rows) + "\n").encode()
    return {
        "request_count": request_count,
        "shape": shape,
        "executions": len(payloads),
        "completions": len(completion_rows),
        "result_bytes": len(result_bytes),
        "result_sha256": _sha256(result_bytes),
        "completion_bytes": len(completion_bytes),
        "completion_sha256": _sha256(completion_bytes),
        "operation_records": operation_records,
        "causal_records": causal_records,
        "additive_records": additive_records,
        "local_frontier_records": local_frontier_records,
        "early_scheduler_boundary_records": early_boundary_records,
        "derivation_errors": derivation_errors,
    }


def _out_of_order_graph() -> Any:
    from simllm.core import (
        CollectiveWork,
        ComputeWork,
        ExecutionGraph,
        ExecutionOperation,
        OperationCorrelation,
    )

    correlation = OperationCorrelation(request_ids=("request",))
    return ExecutionGraph(
        execution_id="scalar-projection-out-of-order",
        step_index=0,
        released_at_ps=0,
        operations=(
            ExecutionOperation(
                "collective",
                0,
                "collective",
                CollectiveWork(
                    "all-to-allv",
                    (0, 8, 16),
                    0,
                    "pairwise",
                    pair_payload_bytes=((8, 0, 1_000_000), (16, 8, 1)),
                ),
                correlation=correlation,
            ),
            ExecutionOperation(
                "early",
                16,
                "compute",
                ComputeWork("early", nominal_duration_ps=10),
                participant_local_depends_on=("collective",),
                correlation=correlation,
            ),
            ExecutionOperation(
                "late",
                0,
                "compute",
                ComputeWork("late", nominal_duration_ps=5),
                participant_local_depends_on=("collective",),
                correlation=correlation,
            ),
            ExecutionOperation(
                "barrier",
                8,
                "compute",
                ComputeWork("barrier", nominal_duration_ps=1),
                depends_on=("collective",),
                correlation=correlation,
            ),
        ),
        completion_operation_ids=("early", "late", "barrier"),
    )


def _out_of_order_record() -> Any:
    from simllm.core import RequestPhase, ScheduledRequest, StepRecord

    return StepRecord(
        step_index=0,
        virtual_time_ps=0,
        scheduled=[ScheduledRequest("request", RequestPhase.PREFILL, 1)],
        num_sampled=1,
        sampled_request_ids=["request"],
    )


def _mutations() -> dict[str, Any]:
    def scalar(operation_id: str, **fields: Any) -> Any:
        def transform(record: Any) -> Any:
            if record.operation_id != operation_id:
                return record
            return replace(record, **fields)

        return transform

    def shorten(record: Any) -> Any:
        if record.operation_id != "early":
            return record
        breakdown = record.breakdown
        attribution = record.attribution
        return replace(
            record,
            breakdown=replace(
                breakdown,
                external_dependency_ps=breakdown.external_dependency_ps - 5,
                operation_latency_ps=breakdown.operation_latency_ps - 5,
            ),
            attribution=replace(attribution, queue_ps=attribution.queue_ps - 5),
        )

    return {
        "M1": scalar("early", critical_predecessor_id="collective"),
        "M2": scalar("late", critical_predecessor_id=None),
        "M3": scalar("early", causal_predecessor_completed_at_ps=17),
        "M4": scalar("late", causal_predecessor_id="barrier"),
        "M5": scalar("collective", physical_completed_at_ps=20_000_007),
        "M6": shorten,
    }


def _run_out_of_order(out: Path) -> dict[str, Any]:
    from simllm.core import (
        CoarseDeviceProfile,
        CoarseDeviceRuntime,
        CompletionReducer,
        VirtualClock,
    )

    graph = _out_of_order_graph()
    runtime = CoarseDeviceRuntime(CoarseDeviceProfile())
    execution = runtime.execute(graph)
    report = runtime.last_report
    assert report is not None

    # Scored measurement first, before the reducer validates this report.
    derivation_errors = _derivation_errors(graph, report)

    by_id = {record.operation_id: record for record in report.operations}
    collective = by_id["collective"]
    participants = dict(collective.participant_completed_at_ps)
    observed = {
        "collective_participant_16_ps": participants[16],
        "collective_participant_0_ps": participants[0],
        "collective_participant_8_ps": participants[8],
        "collective_completed_ps": collective.completed_at_ps,
        "early_boundary_ps": by_id["early"].causal_predecessor_completed_at_ps,
        "early_completed_ps": by_id["early"].completed_at_ps,
        "late_boundary_ps": by_id["late"].causal_predecessor_completed_at_ps,
        "late_completed_ps": by_id["late"].completed_at_ps,
        "barrier_boundary_ps": by_id["barrier"].causal_predecessor_completed_at_ps,
        "barrier_completed_ps": by_id["barrier"].completed_at_ps,
        "result_boundary_ps": execution.completed_at_ps,
    }
    additive = {
        name: by_id[name].critical_predecessor_id
        for name in ("early", "late", "barrier")
    }

    controls = []
    for name, transform in _mutations().items():
        mutated = replace(
            report,
            operations=tuple(transform(record) for record in report.operations),
        )
        clock = VirtualClock(0)
        reducer = CompletionReducer(clock)
        rejected = False
        message = ""
        try:
            reducer.reduce(_out_of_order_record(), graph, execution, mutated)
        except (ValueError, TypeError, RuntimeError) as error:
            rejected = True
            message = f"{type(error).__name__}: {error}"
        controls.append(
            {
                "control": name,
                "rejected": rejected,
                "message": message,
                "clock_unmoved": clock.now_ps == 0,
                "no_metrics_committed": not reducer.latest_request_metrics,
                "passed": rejected
                and clock.now_ps == 0
                and not reducer.latest_request_metrics,
            }
        )

    clock = VirtualClock(0)
    accepted = CompletionReducer(clock).reduce(
        _out_of_order_record(),
        graph,
        execution,
        report,
    )
    (out / "out-of-order").mkdir(parents=True, exist_ok=False)
    return {
        "derivation_errors": derivation_errors,
        "observed": observed,
        "expected": dict(OUT_OF_ORDER_EXPECTATIONS),
        "additive_predecessors": additive,
        "expected_additive_predecessors": dict(OUT_OF_ORDER_ADDITIVE),
        "early_scalar_latency_ps": by_id["early"].breakdown.operation_latency_ps,
        "early_segment_latency_ps": (
            by_id["early"].critical_segments[0].breakdown.operation_latency_ps
        ),
        "realized_chain": list(report.realized_critical_path_operation_ids),
        "realized_segments": [
            list(key) for key in report.realized_critical_path_segments
        ],
        "accepted_boundary_ps": accepted.completed_at_ps,
        "negative_controls": controls,
    }


def _behavioral(
    cells: dict[tuple[int, str], dict[str, Any]],
    out_of_order: dict[str, Any],
) -> dict[str, Any]:
    b1 = []
    for key in sorted(cells):
        cell = cells[key]
        b1.append(
            {
                "fixture": f"granite-{key[1]}-requests-{key[0]}",
                "operation_records": cell["operation_records"],
                "derivation_errors": cell["derivation_errors"][:5],
                "error_count": len(cell["derivation_errors"]),
                "passed": not cell["derivation_errors"],
            }
        )
    b1.append(
        {
            "fixture": "out-of-order-collective",
            "operation_records": 4,
            "derivation_errors": out_of_order["derivation_errors"][:5],
            "error_count": len(out_of_order["derivation_errors"]),
            "passed": not out_of_order["derivation_errors"],
        }
    )

    b2 = [dict(row) for row in out_of_order["negative_controls"]]

    b3 = []
    for request_count in REQUEST_COUNTS:
        local = cells[(request_count, "participant-local")]
        barrier = cells[(request_count, "barrier")]
        b3.append(
            {
                "request_count": request_count,
                "local_frontier_records": local["local_frontier_records"],
                "barrier_frontier_records": barrier["local_frontier_records"],
                "passed": local["local_frontier_records"] > 0
                and barrier["local_frontier_records"] == 0,
            }
        )

    return {
        "SP-B1": {"instances": b1, "passed": all(row["passed"] for row in b1)},
        "SP-B2": {"instances": b2, "passed": all(row["passed"] for row in b2)},
        "SP-B3": {"instances": b3, "passed": all(row["passed"] for row in b3)},
    }


def _fatal_checks(
    cells: dict[tuple[int, str], dict[str, Any]],
    out_of_order: dict[str, Any],
    inputs: dict[str, Any],
) -> dict[str, Any]:
    preservation = []
    for key in sorted(cells):
        cell = cells[key]
        frozen = PRESERVED_CELLS[key]
        observed = {name: cell[name] for name in frozen}
        preservation.append(
            {
                "request_count": key[0],
                "shape": key[1],
                "observed": observed,
                "expected": dict(frozen),
                "passed": observed == dict(frozen),
            }
        )
    exact_out_of_order = {
        "observed": out_of_order["observed"],
        "expected": out_of_order["expected"],
        "additive_observed": out_of_order["additive_predecessors"],
        "additive_expected": out_of_order["expected_additive_predecessors"],
        "accepted_boundary_ps": out_of_order["accepted_boundary_ps"],
        "early_scalar_latency_ps": out_of_order["early_scalar_latency_ps"],
        "early_segment_latency_ps": out_of_order["early_segment_latency_ps"],
        "realized_chain": out_of_order["realized_chain"],
    }
    exact_out_of_order["passed"] = (
        out_of_order["observed"] == out_of_order["expected"]
        and out_of_order["additive_predecessors"]
        == out_of_order["expected_additive_predecessors"]
        and out_of_order["accepted_boundary_ps"]
        == OUT_OF_ORDER_EXPECTATIONS["result_boundary_ps"]
        and out_of_order["early_scalar_latency_ps"] == 30
        and out_of_order["early_segment_latency_ps"] == 10
        and out_of_order["realized_chain"] == ["collective", "late"]
    )
    checks = {
        "source_inputs": inputs,
        "digest_preservation": preservation,
        "out_of_order_exact": exact_out_of_order,
    }
    checks["passed"] = (
        all(row["passed"] for row in preservation) and exact_out_of_order["passed"]
    )
    return checks


def check_only(arguments: argparse.Namespace) -> None:
    _check_frozen_registry()
    print(
        f"check-only source-root={arguments.source_root} out={arguments.out}; "
        "validated the frozen CORE-46 registry and produced no artifacts"
    )


def run_study(arguments: argparse.Namespace) -> dict[str, Any]:
    from simllm.core import step_records_from_jsonl
    from simllm.preplay import read_preplay_replay_run

    _check_frozen_registry()
    frontier = _frontier_module()
    inputs = frontier._validate_inputs(arguments)
    arguments.out.mkdir(parents=True, exist_ok=False)
    external_run = read_preplay_replay_run(
        arguments.source_root / frontier.SOURCE_ARTIFACTS["run"][0]
    )
    steps = step_records_from_jsonl(
        arguments.source_root / frontier.SOURCE_ARTIFACTS["steps"][0]
    )
    if len(steps) != 32:
        raise AssertionError("expected 32 recorded scheduler steps")

    cells: dict[tuple[int, str], dict[str, Any]] = {}
    for request_count in REQUEST_COUNTS:
        for shape in SHAPES:
            cells[(request_count, shape)] = _run_granite_cell(
                frontier,
                arguments,
                external_run,
                steps,
                request_count,
                shape,
            )
    out_of_order = _run_out_of_order(arguments.out)

    behavioral = _behavioral(cells, out_of_order)
    fatal = _fatal_checks(cells, out_of_order, inputs)
    summary = {
        "schema": "simllm-scalar-projection-study-v1",
        "provenance": {
            "expectations_commit": EXPECTATIONS_COMMIT,
            "evidence_authored_against": EVIDENCE_AUTHORED_AGAINST,
            "observed_simllm_revision": _git_revision("HEAD"),
            "python": sys.version,
            "platform": platform.platform(),
        },
        "cells": {f"{key[0]}:{key[1]}": value for key, value in cells.items()},
        "out_of_order": out_of_order,
        "behavioral": behavioral,
        "behavioral_score": {
            "passed_families": sum(
                bool(family["passed"]) for family in behavioral.values()
            ),
            "total_families": 3,
            "passed_instances": sum(
                sum(bool(instance["passed"]) for instance in family["instances"])
                for family in behavioral.values()
            ),
            "total_instances": 13,
        },
        "fatal_unscored": fatal,
        "entailment": (
            "SP-B1 and SP-B3 recompute the derivation and the frontier counts "
            "from the raw RuntimeReport before CompletionReducer validates that "
            "same report, so the validator under test entails neither. SP-B2 "
            "reads only rejection outcomes, and every digest, preservation and "
            "exact-timestamp oracle is evaluated after all three families."
        ),
    }
    (arguments.out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not all(family["passed"] for family in behavioral.values()):
        raise AssertionError("CORE-46 study failed its frozen behavioral bar")
    if not fatal["passed"]:
        raise AssertionError("CORE-46 study violated a fatal guard")
    return summary


def main() -> None:
    arguments = _parse_args()
    if arguments.check_only:
        check_only(arguments)
        return
    summary = run_study(arguments)
    print(json.dumps(summary["behavioral_score"], sort_keys=True))


if __name__ == "__main__":
    main()
