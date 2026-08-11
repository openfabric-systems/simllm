"""Independently validate and score the frozen Tier B observation schema."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTATIONS = Path(__file__).with_name("tier_b_review_expectations.json")
T0 = 7_000
ATTRIBUTION_KEYS = {
    "queue_ps",
    "kv_ps",
    "kernel_ps",
    "dma_ps",
    "collective_ps",
    "nic_ps",
    "control_ps",
}


class TierBAcceptanceError(RuntimeError):
    """Raw observations violate a frozen Tier B relation or invariant."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TierBAcceptanceError(message)


def _object(value: Any, name: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{name} must be an object")
    return value


def _array(value: Any, name: str) -> list[Any]:
    _require(isinstance(value, list), f"{name} must be an array")
    return value


def _integer(value: Any, name: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool),
        f"{name} must be an integer",
    )
    return value


def _text(value: Any, name: str) -> str:
    _require(isinstance(value, str) and bool(value.strip()), f"{name} must be text")
    return value


def _exact_keys(value: dict[str, Any], expected: list[str] | set[str], name: str) -> None:
    expected_set = set(expected)
    actual = set(value)
    _require(
        actual == expected_set,
        f"{name} keys differ: missing={sorted(expected_set - actual)}, "
        f"unexpected={sorted(actual - expected_set)}",
    )


def _load_json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TierBAcceptanceError(f"cannot read {name} {path}: {error}") from error
    return _object(value, name)


def _expected_service(payload: int, rate: int) -> int:
    numerator = payload * 8 * 1000
    _require(rate > 0 and numerator % rate == 0, "wire service is not exact")
    return numerator // rate


def _fraction(value: Any, name: str) -> tuple[int, int] | None:
    if value is None:
        return None
    row = _object(value, name)
    _exact_keys(row, {"numerator", "denominator"}, name)
    numerator = _integer(row["numerator"], f"{name}.numerator")
    denominator = _integer(row["denominator"], f"{name}.denominator")
    _require(denominator > 0, f"{name} denominator must be positive")
    return numerator, denominator


def _attribution(value: Any, name: str) -> dict[str, int]:
    row = _object(value, name)
    _exact_keys(row, ATTRIBUTION_KEYS, name)
    result = {key: _integer(row[key], f"{name}.{key}") for key in row}
    _require(all(item >= 0 for item in result.values()), f"{name} is negative")
    return result


def _additive(value: Any, expectations: dict[str, Any], name: str) -> dict[str, int]:
    row = _object(value, name)
    _exact_keys(row, expectations["raw_additive_visit_keys"], name)
    result = {key: _integer(row[key], f"{name}.{key}") for key in row}
    _require(all(item >= 0 for item in result.values()), f"{name} is negative")
    return result


@dataclass(frozen=True)
class StepView:
    raw: dict[str, Any]
    release_ps: int
    completed_at_ps: int
    latency_ps: int
    metric: dict[str, Any]
    report: dict[str, Any]
    wqes: tuple[dict[str, Any], ...]
    visits: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class CellView:
    raw: dict[str, Any]
    payload_bytes: int
    rate_gbps: int
    doorbell_ps: int
    wqe_count: int
    steps: tuple[StepView, ...]
    summary: dict[str, Any]


def _validate_event(
    value: Any,
    index: int,
    execution_id: str,
    operation_ids: set[str],
    expectations: dict[str, Any],
    name: str,
) -> dict[str, Any]:
    row = _object(value, name)
    _exact_keys(row, expectations["raw_event_keys"], name)
    _require(row["event_index"] == index, f"{name} event index drifted")
    _require(
        row["schema"] == "simllm-completion-event-v1",
        f"{name} event schema drifted",
    )
    _require(row["execution_id"] == execution_id, f"{name} execution ID drifted")
    _require(row["operation_id"] in operation_ids, f"{name} operation is unknown")
    _require(
        row["phase"] in {"submitted", "queued", "started", "progress", "completed"},
        f"{name} phase is unsupported",
    )
    _integer(row["timestamp_ps"], f"{name}.timestamp_ps")
    if row["resource_kind"] is not None:
        _text(row["resource_kind"], f"{name}.resource_kind")
        _text(row["resource_id"], f"{name}.resource_id")
    else:
        _require(row["resource_id"] is None, f"{name} has a partial resource")
    if row["completed_bytes"] is not None:
        _integer(row["completed_bytes"], f"{name}.completed_bytes")
    if row["subject_object_id"] is not None:
        _text(row["subject_object_id"], f"{name}.subject_object_id")
    return row


def _validate_visit(
    value: Any,
    execution_id: str,
    operation_ids: set[str],
    expectations: dict[str, Any],
    name: str,
) -> dict[str, Any]:
    row = _object(value, name)
    _exact_keys(row, expectations["raw_visit_keys"], name)
    _require(row["execution_id"] == execution_id, f"{name} execution ID drifted")
    _require(row["operation_id"] in operation_ids, f"{name} operation is unknown")
    _text(row["stage"], f"{name}.stage")
    _text(row["resource_kind"], f"{name}.resource_kind")
    _text(row["resource_id"], f"{name}.resource_id")
    if row["subject_object_id"] is not None:
        _text(row["subject_object_id"], f"{name}.subject_object_id")
    timestamps = [
        _integer(row[key], f"{name}.{key}")
        for key in (
            "submitted_at_ps",
            "eligible_at_ps",
            "started_at_ps",
            "finished_at_ps",
            "completed_at_ps",
        )
    ]
    _require(timestamps == sorted(timestamps), f"{name} timestamps are not monotonic")
    _require(
        _integer(row["service_bytes"], f"{name}.service_bytes") >= 0,
        f"{name} has negative service bytes",
    )
    return row


def _validate_wqe(
    value: Any,
    ordinal: int,
    operation_ids: set[str],
    payload: int,
    expectations: dict[str, Any],
    name: str,
) -> dict[str, Any]:
    row = _object(value, name)
    _exact_keys(row, expectations["raw_wqe_keys"], name)
    _require(row["ordinal"] == ordinal, f"{name} ordinal drifted")
    _text(row["wqe_id"], f"{name}.wqe_id")
    _require(row["operation_id"] in operation_ids, f"{name} operation is unknown")
    timestamps = [
        _integer(row[key], f"{name}.{key}")
        for key in (
            "submitted_at_ps",
            "doorbell_started_at_ps",
            "doorbell_completed_at_ps",
            "network_eligible_at_ps",
            "network_started_at_ps",
            "network_finished_at_ps",
            "completed_at_ps",
        )
    ]
    _require(timestamps == sorted(timestamps), f"{name} timeline is not monotonic")
    _require(row["payload_bytes"] == payload, f"{name} payload drifted")
    _require(
        _integer(row["sq_post_sequence"], f"{name}.sq_post_sequence") > 0
        and _integer(row["cq_post_sequence"], f"{name}.cq_post_sequence") > 0,
        f"{name} has a nonpositive queue sequence",
    )
    return row


def _validate_step(
    value: Any,
    *,
    cell_name: str,
    step_index: int,
    expected_release: int,
    payload: int,
    wqe_count: int,
    authority: str,
    expectations: dict[str, Any],
) -> StepView:
    name = f"{cell_name}.steps[{step_index}]"
    step = _object(value, name)
    _exact_keys(step, expectations["raw_step_keys"], name)
    _require(step["step_index"] == step_index, f"{name} index drifted")

    graph = _object(step["execution_graph"], f"{name}.execution_graph")
    _exact_keys(graph, expectations["raw_graph_keys"], f"{name}.execution_graph")
    execution_id = _text(graph["execution_id"], f"{name}.execution_id")
    _require(graph["step_index"] == step_index, f"{name} graph index drifted")
    release = _integer(graph["released_at_ps"], f"{name}.released_at_ps")
    _require(release == expected_release, f"{name} release recurrence drifted")
    operation_ids_raw = _array(graph["operation_ids"], f"{name}.operation_ids")
    operation_ids = {_text(item, f"{name}.operation_id") for item in operation_ids_raw}
    _require(
        len(operation_ids_raw) == len(operation_ids) == wqe_count,
        f"{name} operation inventory drifted",
    )
    completion_ids = _array(
        graph["completion_operation_ids"],
        f"{name}.completion_operation_ids",
    )
    _require(
        completion_ids == [operation_ids_raw[-1]],
        f"{name} required endpoint is not W1 or the sole WQE",
    )
    request_ids = _array(graph["request_ids"], f"{name}.request_ids")
    _require(len(request_ids) == 1, f"{name} must carry one request")
    request_id = _text(request_ids[0], f"{name}.request_id")

    events_raw = _array(step["completion_events"], f"{name}.completion_events")
    events = [
        _validate_event(
            event,
            index,
            execution_id,
            operation_ids,
            expectations,
            f"{name}.completion_events[{index}]",
        )
        for index, event in enumerate(events_raw)
    ]
    full_indices = list(range(len(events)))
    _require(
        step["callback_event_indices"] == full_indices,
        f"{name} callback objects were not reused in full order",
    )
    _require(
        {event["phase"] for event in events}
        == {"submitted", "queued", "started", "progress", "completed"},
        f"{name} completion-event lifecycle is incomplete",
    )
    event_times = [event["timestamp_ps"] for event in events]
    _require(event_times == sorted(event_times), f"{name} events are not time ordered")

    result = _object(step["execution_result"], f"{name}.execution_result")
    _exact_keys(
        result,
        expectations["raw_execution_result_keys"],
        f"{name}.execution_result",
    )
    _require(
        result["schema"] == "simllm-execution-result-v1"
        and result["execution_id"] == execution_id
        and result["event_indices"] == full_indices,
        f"{name} ExecutionResult projection drifted",
    )
    completed = _integer(result["completed_at_ps"], f"{name}.completed_at_ps")
    quiesced = _integer(result["quiesced_at_ps"], f"{name}.quiesced_at_ps")
    _require(completed == quiesced, f"{name} selected physical quiescence as a tail")

    report = _object(step["runtime_report"], f"{name}.runtime_report")
    _exact_keys(
        report,
        expectations["raw_runtime_report_keys"],
        f"{name}.runtime_report",
    )
    _require(
        report["execution_id"] == execution_id and report["authority"] == authority,
        f"{name} runtime authority drifted",
    )
    operation_rows = _array(report["operations"], f"{name}.operations")
    _require(len(operation_rows) == wqe_count, f"{name} runtime lost an operation")
    operations: dict[str, dict[str, Any]] = {}
    for index, raw_operation in enumerate(operation_rows):
        operation_name = f"{name}.operations[{index}]"
        operation = _object(raw_operation, operation_name)
        _exact_keys(
            operation,
            expectations["raw_runtime_operation_keys"],
            operation_name,
        )
        operation_id = _text(operation["operation_id"], f"{operation_name}.id")
        _require(operation_id in operation_ids, f"{operation_name} ID is unknown")
        _require(operation_id not in operations, f"{operation_name} repeats an ID")
        operation_completed = _integer(
            operation["completed_at_ps"],
            f"{operation_name}.completed_at_ps",
        )
        _require(
            operation["critical_predecessor_id"] is None,
            f"{operation_name} gained a graph predecessor",
        )
        attribution = _attribution(operation["attribution"], f"{operation_name}.attribution")
        _require(
            all(
                attribution[key] == 0
                for key in ("kv_ps", "kernel_ps", "dma_ps", "collective_ps", "control_ps")
            ),
            f"{operation_name} has an active zero-service component",
        )
        _require(
            sum(attribution.values()) == operation_completed - release,
            f"{operation_name} seven-component row does not conserve",
        )
        operations[operation_id] = operation
    _require(set(operations) == operation_ids, f"{name} operation report is incomplete")
    _require(
        operations[completion_ids[0]]["completed_at_ps"] == completed,
        f"{name} required operation and result boundary disagree",
    )

    visits_raw = _array(report["visits"], f"{name}.visits")
    visits = tuple(
        _validate_visit(
            visit,
            execution_id,
            operation_ids,
            expectations,
            f"{name}.visits[{index}]",
        )
        for index, visit in enumerate(visits_raw)
    )
    visit_wait = sum(visit["started_at_ps"] - visit["eligible_at_ps"] for visit in visits)
    _require(
        report["sum_visit_wait_ps"] == visit_wait,
        f"{name} additive visit wait drifted",
    )
    _require(
        _integer(report["critical_path_queue_ps"], f"{name}.critical_path_queue_ps")
        >= 0,
        f"{name} critical queue tail is negative",
    )
    realized = _array(
        report["realized_critical_path_operation_ids"],
        f"{name}.realized_critical_path_operation_ids",
    )
    _require(realized == completion_ids, f"{name} realized endpoint drifted")

    wqe_rows = _array(report["wqes"], f"{name}.wqes")
    _require(len(wqe_rows) == wqe_count, f"{name} WQE cardinality drifted")
    wqes = tuple(
        _validate_wqe(
            wqe,
            ordinal,
            operation_ids,
            payload,
            expectations,
            f"{name}.wqes[{ordinal}]",
        )
        for ordinal, wqe in enumerate(wqe_rows)
    )
    _require(
        len({wqe["wqe_id"] for wqe in wqes}) == wqe_count,
        f"{name} WQE identities are not unique",
    )
    _require(
        [wqe["operation_id"] for wqe in wqes] == operation_ids_raw,
        f"{name} WQE order disagrees with graph order",
    )

    for wqe in wqes:
        subject_events = [
            event
            for event in events
            if event["subject_object_id"] == wqe["wqe_id"]
        ]
        expected_subject_times = {
            "submitted": wqe["submitted_at_ps"],
            "queued": wqe["network_eligible_at_ps"],
            "started": wqe["network_started_at_ps"],
            "progress": wqe["network_finished_at_ps"],
            "completed": wqe["completed_at_ps"],
        }
        _require(
            len(subject_events) == len(expected_subject_times),
            f"{name} WQE lifecycle event count drifted",
        )
        for phase, timestamp in expected_subject_times.items():
            matching = [event for event in subject_events if event["phase"] == phase]
            _require(
                len(matching) == 1 and matching[0]["timestamp_ps"] == timestamp,
                f"{name} WQE {phase} projection disagrees with native evidence",
            )
            expected_bytes = payload if phase in {"progress", "completed"} else None
            _require(
                matching[0]["completed_bytes"] == expected_bytes,
                f"{name} WQE {phase} byte projection drifted",
            )
    for operation_id, operation in operations.items():
        logical = [
            event
            for event in events
            if event["operation_id"] == operation_id
            and event["subject_object_id"] is None
            and event["phase"] == "completed"
        ]
        _require(
            len(logical) == 1
            and logical[0]["timestamp_ps"] == operation["completed_at_ps"],
            f"{name} operation completion projection drifted",
        )

    step_result = _object(step["step_result"], f"{name}.step_result")
    _exact_keys(
        step_result,
        expectations["raw_step_result_keys"],
        f"{name}.step_result",
    )
    latency = _integer(step_result["step_latency_ps"], f"{name}.step_latency_ps")
    _require(
        step_result["step_index"] == step_index
        and step_result["completed_at_ps"] == completed
        and latency == completed - release,
        f"{name} StepResult boundary drifted",
    )
    metrics = _array(step_result["request_metrics"], f"{name}.request_metrics")
    _require(len(metrics) == 1, f"{name} lost the request metric")
    metric = _object(metrics[0], f"{name}.request_metrics[0]")
    _exact_keys(metric, expectations["raw_request_metric_keys"], f"{name}.metric")
    _require(
        metric["request_id"] == request_id
        and metric["phase"] == ("prefill" if step_index == 0 else "decode")
        and metric["token_index"] == step_index + 1
        and metric["completed_at_ps"] == completed
        and metric["latency_ps"] == latency,
        f"{name} request metric identity or boundary drifted",
    )
    metric_attribution = _attribution(metric["attribution"], f"{name}.metric.attribution")
    _require(
        all(
            metric_attribution[key] == 0
            for key in ("kv_ps", "kernel_ps", "dma_ps", "collective_ps", "control_ps")
        ),
        f"{name} request metric has an active zero-service component",
    )
    _require(
        sum(metric_attribution.values()) == latency,
        f"{name} request attribution does not conserve",
    )
    metric_additive = _additive(
        metric["additive_visit_totals"],
        expectations,
        f"{name}.metric.additive_visit_totals",
    )
    graph_additive = _additive(
        step_result["additive_visit_totals"],
        expectations,
        f"{name}.step_result.additive_visit_totals",
    )
    computed_additive = {
        "queue_wait_ps": visit_wait,
        "service_ps": sum(
            visit["finished_at_ps"] - visit["started_at_ps"] for visit in visits
        ),
        "visibility_ps": sum(
            visit["completed_at_ps"] - visit["finished_at_ps"] for visit in visits
        ),
        "visit_count": len(visits),
    }
    _require(
        metric_additive == graph_additive == computed_additive,
        f"{name} additive visit totals drifted",
    )
    return StepView(step, release, completed, latency, metric, report, wqes, visits)


def _validate_cell(
    value: Any,
    *,
    wqe_count: int,
    expectations: dict[str, Any],
    name: str,
) -> CellView:
    cell = _object(value, name)
    _exact_keys(cell, expectations["raw_structural_cell_keys"], name)
    payload = _integer(cell["payload_bytes"], f"{name}.payload_bytes")
    rate = _integer(cell["link_rate_gbps"], f"{name}.link_rate_gbps")
    doorbell = _integer(cell["doorbell_service_ps"], f"{name}.doorbell_service_ps")
    _require(
        cell["hardware_mode"] == "structural"
        and cell["authority"] == "SimllmNativeRnicSession"
        and cell["simllm_profile"] == "coarse-zero-service",
        f"{name} structural configuration drifted",
    )
    raw_steps = _array(cell["steps"], f"{name}.steps")
    _require(len(raw_steps) == 3, f"{name} must contain three request steps")
    steps = []
    release = T0
    for index, raw_step in enumerate(raw_steps):
        step = _validate_step(
            raw_step,
            cell_name=name,
            step_index=index,
            expected_release=release,
            payload=payload,
            wqe_count=wqe_count,
            authority=cell["authority"],
            expectations=expectations,
        )
        steps.append(step)
        release = step.completed_at_ps

    summary = _object(cell["request_summary"], f"{name}.request_summary")
    _exact_keys(summary, expectations["raw_request_summary_keys"], f"{name}.summary")
    request_id = steps[0].metric["request_id"]
    completions = [step.completed_at_ps for step in steps]
    _require(
        summary["request_id"] == request_id
        and summary["token_completion_times_ps"] == completions
        and summary["ttft_ps"] == steps[-1].metric["ttft_ps"]
        and _fraction(summary["tpot_ps"], f"{name}.summary.tpot_ps")
        == _fraction(steps[-1].metric["tpot_ps"], f"{name}.metric.tpot_ps"),
        f"{name} request summary drifted",
    )
    return CellView(cell, payload, rate, doorbell, wqe_count, tuple(steps), summary)


def _doorbell_owner(step: StepView, doorbell: int, name: str) -> str:
    owners = set()
    for wqe in step.wqes:
        visits = [
            visit
            for visit in step.visits
            if visit["stage"] == "native_doorbell"
            and visit["subject_object_id"] == wqe["wqe_id"]
        ]
        _require(len(visits) == 1, f"{name} has no unique native doorbell visit")
        visit = visits[0]
        wait = visit["started_at_ps"] - visit["eligible_at_ps"]
        service = visit["finished_at_ps"] - visit["started_at_ps"]
        queue_owner = (
            visit["resource_kind"] == "nic_send_queue"
            and wait == doorbell
            and service == 0
        )
        nic_owner = (
            visit["resource_kind"] == "nic"
            and wait == 0
            and service == doorbell
        )
        _require(queue_owner != nic_owner, f"{name} has no unique doorbell owner")
        owners.add("queue_owner" if queue_owner else "nic_owner")
        _require(
            visit["started_at_ps"] == wqe["doorbell_started_at_ps"]
            and visit["finished_at_ps"] == wqe["doorbell_completed_at_ps"],
            f"{name} doorbell visit disagrees with its WQE projection",
        )
        network_visits = [
            candidate
            for candidate in step.visits
            if candidate["stage"] == "native_network"
            and candidate["subject_object_id"] == wqe["wqe_id"]
        ]
        _require(len(network_visits) == 1, f"{name} has no unique native network visit")
        network = network_visits[0]
        _require(
            network["eligible_at_ps"] == wqe["network_eligible_at_ps"]
            and network["started_at_ps"] == wqe["network_started_at_ps"]
            and network["finished_at_ps"] == wqe["network_finished_at_ps"]
            and network["completed_at_ps"] == wqe["completed_at_ps"],
            f"{name} network visit disagrees with its WQE projection",
        )
        _require(
            visit["service_bytes"] == 0
            and network["service_bytes"] == wqe["payload_bytes"],
            f"{name} native visit byte accounting drifted",
        )
    _require(len(owners) == 1, f"{name} mixes doorbell owners")
    return owners.pop()


def _expected_components(owner: str, doorbell: int, network: int, fifo: bool) -> dict[str, int]:
    if owner == "queue_owner":
        queue_ps = doorbell + (network if fifo else 0)
        nic_ps = network
    else:
        queue_ps = network if fifo else 0
        nic_ps = doorbell + network
    return {
        "queue_ps": queue_ps,
        "kv_ps": 0,
        "kernel_ps": 0,
        "dma_ps": 0,
        "collective_ps": 0,
        "nic_ps": nic_ps,
        "control_ps": 0,
    }


def _check_metric_form(cell: CellView) -> None:
    service = _expected_service(cell.payload_bytes, cell.rate_gbps)
    jct = cell.doorbell_ps + service
    for index, step in enumerate(cell.steps):
        _require(step.latency_ps == jct, "single-WQE step latency missed J")
        _require(
            step.completed_at_ps == T0 + (index + 1) * jct,
            "single-WQE absolute completion missed the recurrence",
        )
        _require(step.metric["ttft_ps"] == jct, "single-WQE TTFT missed J")
        expected_tpot = None if index == 0 else (jct, 1)
        _require(
            _fraction(step.metric["tpot_ps"], "single metric TPOT") == expected_tpot,
            "single-WQE TPOT missed J",
        )
    _require(
        cell.summary["ttft_ps"] == jct
        and _fraction(cell.summary["tpot_ps"], "single summary TPOT") == (jct, 1),
        "single-WQE request summary missed J",
    )


def _check_component_row(cell: CellView, owner: str) -> None:
    service = _expected_service(cell.payload_bytes, cell.rate_gbps)
    expected = _expected_components(owner, cell.doorbell_ps, service, False)
    for step in cell.steps:
        _require(
            step.metric["attribution"] == expected,
            "single-WQE seven-component row missed the selected owner mapping",
        )
        _require(
            step.report["critical_path_queue_ps"] == expected["queue_ps"],
            "single-WQE critical queue tail disagrees with attribution",
        )


def _check_d_pair(low: CellView, high: CellView) -> None:
    _require(low.doorbell_ps == 0 and high.doorbell_ps == 1000, "D pair inputs drifted")
    for index, (low_step, high_step) in enumerate(zip(low.steps, high.steps, strict=True)):
        _require(high_step.latency_ps - low_step.latency_ps == 1000, "D did not add to J")
        _require(
            high_step.completed_at_ps - low_step.completed_at_ps == (index + 1) * 1000,
            "D did not follow the chained absolute-completion recurrence",
        )
        _require(
            high_step.metric["ttft_ps"] - low_step.metric["ttft_ps"] == 1000,
            "D did not add to TTFT",
        )
        if index:
            high_tpot = _fraction(high_step.metric["tpot_ps"], "high TPOT")
            low_tpot = _fraction(low_step.metric["tpot_ps"], "low TPOT")
            _require(
                high_tpot is not None
                and low_tpot is not None
                and high_tpot[0] * low_tpot[1] - low_tpot[0] * high_tpot[1]
                == 1000 * high_tpot[1] * low_tpot[1],
                "D did not add to TPOT",
            )


def _check_inverse_rate(slow: CellView, fast: CellView) -> None:
    _require(slow.rate_gbps == 200 and fast.rate_gbps == 400, "rate pair drifted")
    slow_service = _expected_service(slow.payload_bytes, slow.rate_gbps)
    fast_service = _expected_service(fast.payload_bytes, fast.rate_gbps)
    _require(slow_service == 2 * fast_service, "wire service did not halve")
    expected_residual = 0 if slow.doorbell_ps == 0 else -1000
    _require(
        slow.steps[0].latency_ps - 2 * fast.steps[0].latency_ps == expected_residual,
        "inverse-rate total did not retain the additive doorbell term",
    )


def _check_fifo(cell: CellView, owner: str) -> None:
    service = _expected_service(cell.payload_bytes, cell.rate_gbps)
    jct = cell.doorbell_ps + 2 * service
    expected_components = _expected_components(owner, cell.doorbell_ps, service, True)
    for index, step in enumerate(cell.steps):
        release = step.release_ps
        w0, w1 = step.wqes
        _require(
            w0["doorbell_completed_at_ps"] == release + cell.doorbell_ps
            and w1["doorbell_completed_at_ps"] == release + cell.doorbell_ps
            and w0["network_started_at_ps"] == release + cell.doorbell_ps
            and w0["completed_at_ps"] == release + cell.doorbell_ps + service
            and w1["network_started_at_ps"] == release + cell.doorbell_ps + service
            and w1["completed_at_ps"] == release + cell.doorbell_ps + 2 * service,
            "FIFO native timeline missed a frozen boundary",
        )
        _require(
            w1["network_started_at_ps"] - w1["network_eligible_at_ps"] == service,
            "FIFO W1 queue wait is not L",
        )
        _require(
            w0["completed_at_ps"] < w1["completed_at_ps"]
            and w0["sq_post_sequence"] < w1["sq_post_sequence"]
            and w0["cq_post_sequence"] < w1["cq_post_sequence"],
            "FIFO W0 to W1 completion order drifted",
        )
        _require(
            step.latency_ps == jct
            and step.completed_at_ps == T0 + (index + 1) * jct
            and step.metric["ttft_ps"] == jct,
            "FIFO live metric boundary missed J_fifo",
        )
        expected_tpot = None if index == 0 else (jct, 1)
        _require(
            _fraction(step.metric["tpot_ps"], "FIFO TPOT") == expected_tpot,
            "FIFO TPOT missed J_fifo",
        )
        _require(
            step.metric["attribution"] == expected_components,
            "FIFO component row missed the selected owner mapping",
        )


def _hex(value: Any, name: str) -> str:
    _require(isinstance(value, str), f"{name} must be hexadecimal text")
    _require(len(value) % 2 == 0, f"{name} must have even length")
    _require(value == value.lower(), f"{name} must be lower case")
    try:
        bytes.fromhex(value)
    except ValueError as error:
        raise TierBAcceptanceError(f"{name} is not hexadecimal") from error
    return value


def _validate_bypass(
    value: Any,
    expectations: dict[str, Any],
    name: str,
) -> dict[str, Any]:
    row = _object(value, name)
    _exact_keys(row, expectations["raw_bypass_keys"], name)
    _text(row["profile"], f"{name}.profile")
    _require(
        row["hardware_mode"] == "bypass" and row["authority"] == "AtlahsWqeLedger",
        f"{name} bypass authority drifted",
    )
    inputs = _object(row["inputs"], f"{name}.inputs")
    _exact_keys(inputs, expectations["raw_bypass_input_keys"], f"{name}.inputs")
    _hex(inputs["goal_text_hex"], f"{name}.goal_text_hex")
    _hex(inputs["goal_binary_hex"], f"{name}.goal_binary_hex")
    _hex(inputs["topology_hex"], f"{name}.topology_hex")
    _integer(inputs["seed"], f"{name}.seed")
    argv = _array(inputs["baseline_argv"], f"{name}.baseline_argv")
    _require(argv and all(isinstance(item, str) for item in argv), f"{name} argv drifted")
    for side in ("reference_artifacts", "candidate_artifacts"):
        artifacts = _object(row[side], f"{name}.{side}")
        _exact_keys(
            artifacts,
            expectations["raw_bypass_artifact_keys"],
            f"{name}.{side}",
        )
        _hex(artifacts["completion_csv_hex"], f"{name}.{side}.completion_csv_hex")
        _array(
            artifacts["canonical_completion_rows"],
            f"{name}.{side}.canonical_completion_rows",
        )
        _array(artifacts["step_result_tuples"], f"{name}.{side}.step_result_tuples")
        _array(
            artifacts["replay_request_summary"],
            f"{name}.{side}.replay_request_summary",
        )
    return row


def _check_bypass_identity(row: dict[str, Any]) -> None:
    _require(
        row["reference_artifacts"] == row["candidate_artifacts"],
        f"bypass artifacts changed for {row['profile']}",
    )


@dataclass
class _Family:
    expected: int
    passed: int = 0
    misses: list[dict[str, str]] | None = None

    def __post_init__(self) -> None:
        if self.misses is None:
            self.misses = []

    def score(self, instance: str, check: Callable[[], None]) -> None:
        try:
            check()
        except TierBAcceptanceError as error:
            assert self.misses is not None
            self.misses.append({"instance": instance, "reason": str(error)})
        else:
            self.passed += 1

    def as_dict(self) -> dict[str, Any]:
        assert self.misses is not None
        return {
            "expected_instances": self.expected,
            "passed_instances": self.passed,
            "misses": self.misses,
            "genuine_risk_fraction": f"{self.passed}/{self.expected}",
        }


def _expect_rejection(name: str, check: Callable[[], None]) -> bool:
    try:
        check()
    except TierBAcceptanceError:
        return True
    raise TierBAcceptanceError(f"negative control {name} was accepted")


def check_observations(
    observations: dict[str, Any],
    expectations: dict[str, Any],
) -> dict[str, Any]:
    _exact_keys(observations, expectations["raw_top_keys"], "raw observations")
    _require(
        observations["schema"] == expectations["observation_schema"]
        and observations["factory"] == expectations["factory"]
        and observations["simllm_base_commit"] == expectations["simllm_base_commit"],
        "Tier B top-level identity drifted",
    )

    raw_single = _array(observations["structural_single_wqe"], "single-WQE rows")
    single: dict[tuple[int, int, int], CellView] = {}
    for index, raw in enumerate(raw_single):
        view = _validate_cell(
            raw,
            wqe_count=1,
            expectations=expectations,
            name=f"structural_single_wqe[{index}]",
        )
        key = (view.payload_bytes, view.rate_gbps, view.doorbell_ps)
        _require(key not in single, f"single-WQE repeats {key}")
        single[key] = view
    expected_single = {
        (payload, rate, doorbell)
        for payload in expectations["single_wqe"]["payload_bytes"]
        for rate in expectations["single_wqe"]["link_rate_gbps"]
        for doorbell in expectations["single_wqe"]["doorbell_service_ps"]
    }
    _require(set(single) == expected_single, "single-WQE grid is incomplete")

    raw_fifo = _array(observations["structural_fifo"], "FIFO rows")
    fifo: dict[tuple[int, int], CellView] = {}
    for index, raw in enumerate(raw_fifo):
        view = _validate_cell(
            raw,
            wqe_count=2,
            expectations=expectations,
            name=f"structural_fifo[{index}]",
        )
        _require(
            view.payload_bytes == expectations["fifo"]["payload_bytes"],
            "FIFO payload drifted",
        )
        key = (view.rate_gbps, view.doorbell_ps)
        _require(key not in fifo, f"FIFO repeats {key}")
        fifo[key] = view
    expected_fifo = {
        (rate, doorbell)
        for rate in expectations["fifo"]["link_rate_gbps"]
        for doorbell in expectations["fifo"]["doorbell_service_ps"]
    }
    _require(set(fifo) == expected_fifo, "FIFO grid is incomplete")

    observed_owners = {
        _doorbell_owner(step, cell.doorbell_ps, "doorbell owner")
        for cell in (*single.values(), *fifo.values())
        if cell.doorbell_ps != 0
        for step in cell.steps
    }
    _require(len(observed_owners) == 1, "nonzero-D rows mix doorbell owners")
    owner = observed_owners.pop()
    _require(owner in expectations["doorbell_owner_mappings"], "owner is not frozen")
    for cell in (*single.values(), *fifo.values()):
        for step in cell.steps:
            _require(
                _doorbell_owner(step, cell.doorbell_ps, "doorbell owner") == owner,
                "zero-D row did not inherit the selected owner",
            )

    counts = expectations["behavioral_family_instances"]
    families = {name: _Family(expected) for name, expected in counts.items()}
    for payload in expectations["single_wqe"]["payload_bytes"]:
        for rate in expectations["single_wqe"]["link_rate_gbps"]:
            families["single_wqe_d_additivity"].score(
                f"payload={payload},rate={rate}",
                lambda payload=payload, rate=rate: _check_d_pair(
                    single[(payload, rate, 0)],
                    single[(payload, rate, 1000)],
                ),
            )
        for doorbell in expectations["single_wqe"]["doorbell_service_ps"]:
            families["single_wqe_inverse_rate"].score(
                f"payload={payload},doorbell={doorbell}",
                lambda payload=payload, doorbell=doorbell: _check_inverse_rate(
                    single[(payload, 200, doorbell)],
                    single[(payload, 400, doorbell)],
                ),
            )
    for key, cell in sorted(single.items()):
        families["single_wqe_metric_forms"].score(
            str(key), lambda cell=cell: _check_metric_form(cell)
        )
        families["single_wqe_component_rows"].score(
            str(key), lambda cell=cell: _check_component_row(cell, owner)
        )
    for key, cell in sorted(fifo.items()):
        families["two_wqe_fifo"].score(
            str(key), lambda cell=cell: _check_fifo(cell, owner)
        )

    raw_bypass = _array(observations["bypass"], "bypass rows")
    bypass: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(raw_bypass):
        row = _validate_bypass(raw, expectations, f"bypass[{index}]")
        profile = row["profile"]
        _require(profile not in bypass, f"bypass repeats {profile}")
        bypass[profile] = row
    _require(
        set(bypass) == set(expectations["retained_bypass_profiles"]),
        "bypass profile inventory drifted",
    )
    for profile in expectations["retained_bypass_profiles"]:
        families["bypass_artifact_identity"].score(
            profile,
            lambda profile=profile: _check_bypass_identity(bypass[profile]),
        )

    mutant_d = replace(
        single[(4096, 400, 1000)],
        steps=single[(4096, 400, 0)].steps,
    )
    owner_step = single[(4096, 400, 1000)].steps[0]
    owner_visits = list(copy.deepcopy(owner_step.visits))
    owner_index = next(
        index
        for index, visit in enumerate(owner_visits)
        if visit["stage"] == "native_doorbell"
    )
    owner_visits[owner_index]["resource_kind"] = "control_queue"
    mutant_owner_step = replace(owner_step, visits=tuple(owner_visits))
    changed_bypass = copy.deepcopy(bypass[expectations["retained_bypass_profiles"][0]])
    changed_bypass["candidate_artifacts"]["completion_csv_hex"] += "00"
    fifo_cell = fifo[(400, 0)]
    fifo_step = fifo_cell.steps[0]
    fifo_wqes = list(copy.deepcopy(fifo_step.wqes))
    fifo_wqes[1]["network_eligible_at_ps"] = fifo_wqes[1]["network_started_at_ps"]
    mutant_fifo_step = replace(fifo_step, wqes=tuple(fifo_wqes))
    mutant_fifo = replace(
        fifo_cell,
        steps=(mutant_fifo_step, *fifo_cell.steps[1:]),
    )
    callback_indices = list(
        single[(4096, 400, 0)].steps[0].raw["callback_event_indices"]
    )
    callback_indices.pop()
    expected_callback_indices = list(
        range(len(single[(4096, 400, 0)].steps[0].raw["completion_events"]))
    )
    negative_controls = {
        "wrapper_bypass_d_sensitivity": _expect_rejection(
            "wrapper_bypass_d_sensitivity",
            lambda: _check_d_pair(single[(4096, 400, 0)], mutant_d),
        ),
        "event_object_reuse_sensitivity": _expect_rejection(
            "event_object_reuse_sensitivity",
            lambda: _require(
                callback_indices == expected_callback_indices,
                "callback index mutant rejected",
            ),
        ),
        "doorbell_owner_sensitivity": _expect_rejection(
            "doorbell_owner_sensitivity",
            lambda: _doorbell_owner(
                mutant_owner_step,
                1000,
                "doorbell owner mutant",
            ),
        ),
        "fifo_wait_sensitivity": _expect_rejection(
            "fifo_wait_sensitivity",
            lambda: _check_fifo(mutant_fifo, owner),
        ),
        "bypass_byte_sensitivity": _expect_rejection(
            "bypass_byte_sensitivity",
            lambda: _check_bypass_identity(changed_bypass),
        ),
    }

    family_rows = {name: family.as_dict() for name, family in families.items()}
    _require(
        {name: row["expected_instances"] for name, row in family_rows.items()}
        == counts,
        "behavioral family denominators drifted",
    )
    all_scored_passed = all(
        row["passed_instances"] == row["expected_instances"]
        for row in family_rows.values()
    )
    return {
        "schema": "simllm-rnic-tier-b-results-v1",
        "factory": observations["factory"],
        "simllm_base_commit": observations["simllm_base_commit"],
        "passed": all_scored_passed,
        "doorbell_owner": owner,
        "behavioral_families": family_rows,
        "exact_oracle_rows": {
            "structural_single_wqe": len(single),
            "structural_fifo": len(fifo),
        },
        "fatal_unscored_invariants": {
            "authority_exclusivity": True,
            "schema_compatibility": True,
            "callback_execution_result_object_reuse": True,
            "event_timestamp_order": True,
            "one_completion_boundary": True,
            "request_component_conservation": True,
            "additive_visit_separation": True,
            "clock_monotonicity": True,
            "native_timeline_projection": True,
            "fifo_completion_order": True,
            "inactive_components_zero": True,
            "bypass_input_guards": True,
        },
        "negative_controls": negative_controls,
    }


def _producer_command(producer: Path, expectations: Path, observations: Path) -> list[str]:
    return [
        str(producer),
        "--factory",
        "htsim",
        "--expectations",
        str(expectations),
        "--observations",
        str(observations),
    ]


def run_acceptance(out: Path, producer: Path) -> dict[str, Any]:
    out = out.resolve(strict=False)
    producer = producer.resolve(strict=True)
    expectations_path = EXPECTATIONS.resolve(strict=True)
    _require(out.is_absolute(), "Tier B output path must be absolute")
    _require(producer.is_relative_to(out), "Tier B producer must reside under output")
    observations_path = out / "raw_observations.json"
    results_path = out / "results.json"
    _require(
        not observations_path.exists()
        and not results_path.exists()
        and not Path(f"{observations_path}.tmp").exists()
        and not Path(f"{results_path}.tmp").exists(),
        "Tier B output already contains observations, results, or a partial file",
    )
    expectations = _load_json(expectations_path, "Tier B expectations")
    command = _producer_command(producer, expectations_path, observations_path)
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    _require(observations_path.is_file(), "producer published no raw observations")
    observations = _load_json(observations_path, "Tier B observations")
    report = check_observations(observations, expectations)
    report["expectations_sha256"] = hashlib.sha256(
        expectations_path.read_bytes()
    ).hexdigest()
    report["producer_argument_names"] = expectations["producer_argument_names"]
    temporary = Path(f"{results_path}.tmp")
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    try:
        temporary.write_text(serialized, encoding="utf-8", newline="\n")
        os.replace(temporary, results_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    for name, family in report["behavioral_families"].items():
        print(f"{name} genuine-risk fraction: {family['genuine_risk_fraction']}")
    if not report["passed"]:
        raise TierBAcceptanceError("Tier B has one or more scored relation misses")
    return report


__all__ = ["TierBAcceptanceError", "check_observations", "run_acceptance"]
