"""Score Tier C packet relations before running any entailing exact oracle."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path
from typing import Any

from examples.rnic_live_v1.tier_b_acceptance import (
    check_observations as check_tier_b_observations,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTATIONS = Path(__file__).with_name("tier_c_expectations.json")
TIER_B_EXPECTATIONS = Path(__file__).with_name("tier_b_review_expectations.json")
FREEZE_COMMIT = "2bd61cdfe7b6d545c05ea17db6894bb50eb14735"
T0 = 7_000


class TierCAcceptanceError(RuntimeError):
    """Raw observations violate a frozen Tier C relation or invariant."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TierCAcceptanceError(message)


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


def _exact_keys(value: dict[str, Any], expected: list[str], name: str) -> None:
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
        raise TierCAcceptanceError(f"cannot read {name} {path}: {error}") from error
    return _object(value, name)


def _cell_key(cell: dict[str, Any]) -> tuple[int, int, int]:
    return (
        _integer(cell.get("payload_bytes"), "cell.payload_bytes"),
        _integer(cell.get("link_rate_gbps"), "cell.link_rate_gbps"),
        _integer(cell.get("doorbell_service_ps"), "cell.doorbell_service_ps"),
    )


def _step(cell: dict[str, Any], index: int) -> dict[str, Any]:
    steps = _array(cell.get("steps"), "cell.steps")
    _require(len(steps) == 3, "Tier C cell must contain three request steps")
    return _object(steps[index], f"cell.steps[{index}]")


def _release(step: dict[str, Any]) -> int:
    graph = _object(step.get("execution_graph"), "step.execution_graph")
    return _integer(graph.get("released_at_ps"), "step.released_at_ps")


def _wqes(step: dict[str, Any]) -> list[dict[str, Any]]:
    report = _object(step.get("runtime_report"), "step.runtime_report")
    return [
        _object(value, f"step.wqes[{index}]")
        for index, value in enumerate(_array(report.get("wqes"), "step.wqes"))
    ]


def _metric(step: dict[str, Any]) -> dict[str, Any]:
    result = _object(step.get("step_result"), "step.step_result")
    metrics = _array(result.get("request_metrics"), "step.request_metrics")
    _require(len(metrics) == 1, "Tier C step must contain one request metric")
    return _object(metrics[0], "step.request_metrics[0]")


def _tpot(value: Any, name: str) -> Fraction | None:
    if value is None:
        return None
    row = _object(value, name)
    _require(set(row) == {"numerator", "denominator"}, f"{name} shape drifted")
    numerator = _integer(row["numerator"], f"{name}.numerator")
    denominator = _integer(row["denominator"], f"{name}.denominator")
    _require(denominator > 0, f"{name} denominator must be positive")
    return Fraction(numerator, denominator)


def _subject_started(step: dict[str, Any], wqe_id: str) -> int:
    events = _array(step.get("completion_events"), "step.completion_events")
    matches = [
        _object(event, "completion event")
        for event in events
        if isinstance(event, dict)
        and event.get("subject_object_id") == wqe_id
        and event.get("phase") == "started"
    ]
    _require(len(matches) == 1, "Tier C WQE has no unique STARTED event")
    return _integer(matches[0].get("timestamp_ps"), "STARTED timestamp")


def _validate_cell_surface(
    cell: dict[str, Any],
    expectations: dict[str, Any],
    expected_wqes: int,
    name: str,
) -> None:
    _exact_keys(cell, expectations["raw_structural_cell_keys"], name)
    _cell_key(cell)
    _require(
        cell.get("hardware_mode") == "structural"
        and cell.get("authority") == "SimllmNativeRnicSession"
        and cell.get("simllm_profile") == "coarse-zero-service",
        f"{name} structural identity drifted",
    )
    steps = _array(cell.get("steps"), f"{name}.steps")
    _require(len(steps) == 3, f"{name} step count drifted")
    release = T0
    for index, raw_step in enumerate(steps):
        step = _object(raw_step, f"{name}.steps[{index}]")
        _require(_release(step) == release, f"{name} release recurrence drifted")
        rows = _wqes(step)
        _require(len(rows) == expected_wqes, f"{name} WQE count drifted")
        for ordinal, wqe in enumerate(rows):
            _exact_keys(
                wqe,
                expectations["raw_packet_wqe_keys"],
                f"{name}.steps[{index}].wqes[{ordinal}]",
            )
            _require(wqe.get("ordinal") == ordinal, f"{name} WQE ordinal drifted")
            for field in (
                "submitted_at_ps",
                "doorbell_started_at_ps",
                "doorbell_completed_at_ps",
                "network_eligible_at_ps",
                "network_accepted_at_ps",
                "network_started_at_ps",
                "first_packet_at_ps",
                "last_packet_at_ps",
                "network_finished_at_ps",
                "completed_at_ps",
                "payload_bytes",
            ):
                _integer(wqe.get(field), f"{name}.{field}")
            starts = _array(
                wqe.get("packet_tx_started_at_ps"),
                f"{name}.packet_tx_started_at_ps",
            )
            for packet_index, value in enumerate(starts):
                _integer(value, f"{name}.packet_tx_started_at_ps[{packet_index}]")
            _subject_started(step, str(wqe.get("wqe_id")))
        step_result = _object(step.get("step_result"), f"{name}.step_result")
        release = _integer(
            step_result.get("completed_at_ps"),
            f"{name}.step_result.completed_at_ps",
        )
        _integer(step_result.get("step_latency_ps"), f"{name}.step_latency_ps")
        metric = _metric(step)
        _integer(metric.get("ttft_ps"), f"{name}.ttft_ps")
        _tpot(metric.get("tpot_ps"), f"{name}.tpot_ps")


def _prepare_cells(
    observations: dict[str, Any], expectations: dict[str, Any]
) -> tuple[
    dict[tuple[int, int, int], dict[str, Any]],
    dict[tuple[int, int], dict[str, Any]],
]:
    _exact_keys(observations, expectations["raw_observation_keys"], "observations")
    _require(
        observations.get("schema") == expectations["observation_schema"]
        and observations.get("factory") == "htsim"
        and observations.get("network_abi_version") == 2,
        "Tier C observation identity drifted",
    )
    single_rows = [
        _object(value, f"single[{index}]")
        for index, value in enumerate(
            _array(observations.get("structural_single_wqe"), "single rows")
        )
    ]
    fifo_rows = [
        _object(value, f"fifo[{index}]")
        for index, value in enumerate(
            _array(observations.get("structural_fifo"), "FIFO rows")
        )
    ]
    single: dict[tuple[int, int, int], dict[str, Any]] = {}
    for index, cell in enumerate(single_rows):
        _validate_cell_surface(cell, expectations, 1, f"single[{index}]")
        key = _cell_key(cell)
        _require(key not in single, f"single cell repeats {key}")
        single[key] = cell
    fifo: dict[tuple[int, int], dict[str, Any]] = {}
    for index, cell in enumerate(fifo_rows):
        _validate_cell_surface(cell, expectations, 2, f"fifo[{index}]")
        payload, rate, doorbell = _cell_key(cell)
        _require(payload == 4096, "Tier C FIFO payload drifted")
        key = (rate, doorbell)
        _require(key not in fifo, f"FIFO cell repeats {key}")
        fifo[key] = cell
    expected_single = {
        (payload, rate, doorbell)
        for payload in expectations["single_wqe"]["payload_bytes"]
        for rate in expectations["single_wqe"]["link_rate_gbps"]
        for doorbell in expectations["single_wqe"]["doorbell_service_ps"]
    }
    expected_fifo = {
        (rate, doorbell)
        for rate in expectations["fifo"]["link_rate_gbps"]
        for doorbell in expectations["fifo"]["doorbell_service_ps"]
    }
    _require(set(single) == expected_single, "Tier C single-WQE matrix drifted")
    _require(set(fifo) == expected_fifo, "Tier C FIFO matrix drifted")
    return single, fifo


def _instance(name: str, observed: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    _require(observed == expected, f"Tier C scored instance {name} failed")
    return {
        "name": name,
        "observed": observed,
        "expected": expected,
        "passed": True,
        "genuine_risk": True,
    }


def _family(name: str, instances: list[dict[str, Any]], expected: int) -> dict[str, Any]:
    _require(len(instances) == expected, f"Tier C {name} denominator drifted")
    return {
        "passed_instances": len(instances),
        "expected_instances": expected,
        "genuine_risk_fraction": f"{len(instances)}/{expected}",
        "instances": instances,
    }


def _score_doorbell(
    single: dict[tuple[int, int, int], dict[str, Any]],
    expectations: dict[str, Any],
) -> dict[str, Any]:
    instances = []
    for relation in expectations["doorbell_live_relations"]:
        payload = relation["payload_bytes"]
        rate = relation["link_rate_gbps"]
        low = single[(payload, rate, 0)]
        high = single[(payload, rate, 1000)]
        observed: dict[str, Any] = {
            "first_packet_offset_delta_ps": [],
            "last_packet_offset_delta_ps": [],
            "completion_event_started_offset_delta_ps": [],
            "step_latency_delta_ps": [],
            "ttft_delta_ps": [],
            "tpot_delta_ps": [],
            "absolute_completion_delta_ps": [],
        }
        for index in range(3):
            low_step = _step(low, index)
            high_step = _step(high, index)
            low_release = _release(low_step)
            high_release = _release(high_step)
            low_wqe = _wqes(low_step)[0]
            high_wqe = _wqes(high_step)[0]
            observed["first_packet_offset_delta_ps"].append(
                high_wqe["first_packet_at_ps"]
                - high_release
                - (low_wqe["first_packet_at_ps"] - low_release)
            )
            observed["last_packet_offset_delta_ps"].append(
                high_wqe["last_packet_at_ps"]
                - high_release
                - (low_wqe["last_packet_at_ps"] - low_release)
            )
            observed["completion_event_started_offset_delta_ps"].append(
                _subject_started(high_step, high_wqe["wqe_id"])
                - high_release
                - (_subject_started(low_step, low_wqe["wqe_id"]) - low_release)
            )
            low_result = _object(low_step["step_result"], "low step result")
            high_result = _object(high_step["step_result"], "high step result")
            observed["step_latency_delta_ps"].append(
                high_result["step_latency_ps"] - low_result["step_latency_ps"]
            )
            observed["absolute_completion_delta_ps"].append(
                high_result["completed_at_ps"] - low_result["completed_at_ps"]
            )
            low_metric = _metric(low_step)
            high_metric = _metric(high_step)
            observed["ttft_delta_ps"].append(
                high_metric["ttft_ps"] - low_metric["ttft_ps"]
            )
            if index:
                low_tpot = _tpot(low_metric["tpot_ps"], "low TPOT")
                high_tpot = _tpot(high_metric["tpot_ps"], "high TPOT")
                assert low_tpot is not None
                assert high_tpot is not None
                observed["tpot_delta_ps"].append(int(high_tpot - low_tpot))
        packet_delta = relation["packet_offset_delta_ps"]
        live_delta = relation["live_metric_delta_ps"]
        expected = {
            "first_packet_offset_delta_ps": [packet_delta] * 3,
            "last_packet_offset_delta_ps": [packet_delta] * 3,
            "completion_event_started_offset_delta_ps": [packet_delta] * 3,
            "step_latency_delta_ps": [live_delta] * 3,
            "ttft_delta_ps": [live_delta] * 3,
            "tpot_delta_ps": [live_delta] * 2,
            "absolute_completion_delta_ps": relation[
                "absolute_completion_delta_ps"
            ],
        }
        instances.append(_instance(f"payload={payload},rate={rate}", observed, expected))
    return _family(
        "doorbell_packet_to_live_chain",
        instances,
        expectations["behavioral_family_instances"][
            "doorbell_packet_to_live_chain"
        ],
    )


def _score_link_rate(
    single: dict[tuple[int, int, int], dict[str, Any]],
    expectations: dict[str, Any],
) -> dict[str, Any]:
    instances = []
    for relation in expectations["link_rate_live_relations"]:
        payload = relation["payload_bytes"]
        doorbell = relation["doorbell_service_ps"]
        slow = single[(payload, 200, doorbell)]
        fast = single[(payload, 400, doorbell)]
        observed: dict[str, Any] = {
            "first_packet_offset_delta_ps": [],
            "last_packet_offset_delta_ps": [],
            "completion_event_started_offset_delta_ps": [],
            "step_latency_delta_ps": [],
            "ttft_delta_ps": [],
            "tpot_delta_ps": [],
        }
        for index in range(3):
            slow_step = _step(slow, index)
            fast_step = _step(fast, index)
            slow_release = _release(slow_step)
            fast_release = _release(fast_step)
            slow_wqe = _wqes(slow_step)[0]
            fast_wqe = _wqes(fast_step)[0]
            observed["first_packet_offset_delta_ps"].append(
                slow_wqe["first_packet_at_ps"]
                - slow_release
                - (fast_wqe["first_packet_at_ps"] - fast_release)
            )
            observed["last_packet_offset_delta_ps"].append(
                slow_wqe["last_packet_at_ps"]
                - slow_release
                - (fast_wqe["last_packet_at_ps"] - fast_release)
            )
            observed["completion_event_started_offset_delta_ps"].append(
                _subject_started(slow_step, slow_wqe["wqe_id"])
                - slow_release
                - (_subject_started(fast_step, fast_wqe["wqe_id"]) - fast_release)
            )
            slow_result = _object(slow_step["step_result"], "slow step result")
            fast_result = _object(fast_step["step_result"], "fast step result")
            observed["step_latency_delta_ps"].append(
                slow_result["step_latency_ps"] - fast_result["step_latency_ps"]
            )
            slow_metric = _metric(slow_step)
            fast_metric = _metric(fast_step)
            observed["ttft_delta_ps"].append(
                slow_metric["ttft_ps"] - fast_metric["ttft_ps"]
            )
            if index:
                slow_tpot = _tpot(slow_metric["tpot_ps"], "slow TPOT")
                fast_tpot = _tpot(fast_metric["tpot_ps"], "fast TPOT")
                assert slow_tpot is not None
                assert fast_tpot is not None
                observed["tpot_delta_ps"].append(int(slow_tpot - fast_tpot))
        first_delta = relation["first_packet_offset_delta_ps"]
        last_delta = relation["last_packet_offset_delta_ps"]
        live_delta = relation["live_metric_delta_ps"]
        expected = {
            "first_packet_offset_delta_ps": [first_delta] * 3,
            "last_packet_offset_delta_ps": [last_delta] * 3,
            "completion_event_started_offset_delta_ps": [first_delta] * 3,
            "step_latency_delta_ps": [live_delta] * 3,
            "ttft_delta_ps": [live_delta] * 3,
            "tpot_delta_ps": [live_delta] * 2,
        }
        instances.append(
            _instance(f"payload={payload},doorbell={doorbell}", observed, expected)
        )
    return _family(
        "link_rate_packet_to_live_chain",
        instances,
        expectations["behavioral_family_instances"][
            "link_rate_packet_to_live_chain"
        ],
    )


def _packet_origin_status(
    single: dict[tuple[int, int, int], dict[str, Any]],
    fifo: dict[tuple[int, int], dict[str, Any]],
    packet_quantum: int,
) -> dict[str, bool]:
    origin = True
    event_projection = True
    acceptance_separation = True
    for cell in (*single.values(), *fifo.values()):
        payload = _integer(cell["payload_bytes"], "payload")
        expected_packets = (payload - 1) // packet_quantum + 1
        for index in range(3):
            step = _step(cell, index)
            for wqe in _wqes(step):
                starts = [
                    _integer(value, "packet TX start")
                    for value in _array(
                        wqe["packet_tx_started_at_ps"], "packet TX starts"
                    )
                ]
                origin = origin and (
                    len(starts) == expected_packets
                    and starts == sorted(starts)
                    and wqe["first_packet_at_ps"] == min(starts, default=-1)
                    and wqe["last_packet_at_ps"] == max(starts, default=-1)
                )
                event_projection = event_projection and (
                    wqe["network_started_at_ps"] == wqe["first_packet_at_ps"]
                    and _subject_started(step, wqe["wqe_id"])
                    == wqe["first_packet_at_ps"]
                )
                acceptance_separation = acceptance_separation and (
                    wqe["network_eligible_at_ps"]
                    <= wqe["network_accepted_at_ps"]
                    <= wqe["first_packet_at_ps"]
                    <= wqe["last_packet_at_ps"]
                    < wqe["network_finished_at_ps"]
                )
                if payload == 1_048_576:
                    acceptance_separation = acceptance_separation and (
                        wqe["last_packet_at_ps"]
                        > wqe["network_accepted_at_ps"]
                    )
    _require(origin, "Tier C packet fields did not derive from explicit TX starts")
    _require(event_projection, "Tier C CompletionEvent STARTED lost packet issue")
    _require(
        acceptance_separation,
        "Tier C packet issue collapsed onto acceptance or terminal time",
    )
    return {
        "explicit_tx_start_origin": origin,
        "completion_event_started_projection": event_projection,
        "acceptance_and_terminal_separation": acceptance_separation,
    }


def _packet_exact_oracle(
    single: dict[tuple[int, int, int], dict[str, Any]],
    fifo: dict[tuple[int, int], dict[str, Any]],
    packet_quantum: int,
) -> dict[str, int]:
    for (payload, rate, doorbell), cell in single.items():
        service = payload * 8 * 1000 // rate
        packet_service = packet_quantum * 8 * 1000 // rate
        packet_count = payload // packet_quantum
        for index in range(3):
            step = _step(cell, index)
            release = _release(step)
            wqe = _wqes(step)[0]
            first = release + doorbell
            expected_starts = [
                first + packet_index * packet_service
                for packet_index in range(packet_count)
            ]
            expected = {
                "network_accepted_at_ps": first,
                "first_packet_at_ps": first,
                "last_packet_at_ps": expected_starts[-1],
                "packet_tx_started_at_ps": expected_starts,
                "network_finished_at_ps": first + service,
                "completed_at_ps": first + service,
            }
            observed = {name: wqe[name] for name in expected}
            _require(
                observed == expected,
                f"Tier C single-WQE packet oracle failed for "
                f"{(payload, rate, doorbell)} step {index}",
            )

    payload = 4096
    for (rate, doorbell), cell in fifo.items():
        service = payload * 8 * 1000 // rate
        for index in range(3):
            step = _step(cell, index)
            release = _release(step)
            for ordinal, wqe in enumerate(_wqes(step)):
                first = release + doorbell + ordinal * service
                expected = {
                    "network_accepted_at_ps": first,
                    "first_packet_at_ps": first,
                    "last_packet_at_ps": first,
                    "packet_tx_started_at_ps": [first],
                    "network_finished_at_ps": first + service,
                    "completed_at_ps": first + service,
                }
                observed = {name: wqe[name] for name in expected}
                _require(
                    observed == expected,
                    f"Tier C FIFO packet oracle failed for "
                    f"{(rate, doorbell)} step {index} WQE {ordinal}",
                )
    return {
        "structural_single_wqe": len(single),
        "structural_fifo": len(fifo),
    }


def _tier_b_projection(
    observations: dict[str, Any], v1_observations: dict[str, Any]
) -> dict[str, Any]:
    projected = copy.deepcopy(observations)
    projected["schema"] = "simllm-rnic-tier-b-observations-v1"
    projected.pop("network_abi_version")
    projected["bypass"] = copy.deepcopy(v1_observations.get("bypass"))
    packet_fields = {
        "network_accepted_at_ps",
        "first_packet_at_ps",
        "last_packet_at_ps",
        "packet_tx_started_at_ps",
    }
    for cell in [
        *projected["structural_single_wqe"],
        *projected["structural_fifo"],
    ]:
        for step in cell["steps"]:
            for wqe in step["runtime_report"]["wqes"]:
                for field in packet_fields:
                    wqe.pop(field)
    return projected


def _set_packet_timeline(
    step: dict[str, Any],
    wqe: dict[str, Any],
    first: int,
    last: int,
) -> None:
    count = max(1, len(wqe["packet_tx_started_at_ps"]))
    wqe["first_packet_at_ps"] = first
    wqe["last_packet_at_ps"] = last
    wqe["packet_tx_started_at_ps"] = [first] * (count - 1) + [last]
    wqe["network_started_at_ps"] = first
    for event in step["completion_events"]:
        if (
            event["subject_object_id"] == wqe["wqe_id"]
            and event["phase"] == "started"
        ):
            event["timestamp_ps"] = first
    for visit in step["runtime_report"]["visits"]:
        if (
            visit["subject_object_id"] == wqe["wqe_id"]
            and visit["stage"] == "native_network"
        ):
            visit["started_at_ps"] = first


def _mutate_acceptance(observations: dict[str, Any]) -> None:
    for cell in [
        *observations["structural_single_wqe"],
        *observations["structural_fifo"],
    ]:
        for step in cell["steps"]:
            for wqe in step["runtime_report"]["wqes"]:
                accepted = wqe["network_accepted_at_ps"]
                _set_packet_timeline(step, wqe, accepted, accepted)


def _mutate_constant(observations: dict[str, Any]) -> None:
    for cell in [
        *observations["structural_single_wqe"],
        *observations["structural_fifo"],
    ]:
        for step in cell["steps"]:
            constant = _release(step) + 12_345
            for wqe in step["runtime_report"]["wqes"]:
                _set_packet_timeline(step, wqe, constant, constant)


def _mutate_missing_tx(observations: dict[str, Any]) -> None:
    for cell in [
        *observations["structural_single_wqe"],
        *observations["structural_fifo"],
    ]:
        for step in cell["steps"]:
            for wqe in step["runtime_report"]["wqes"]:
                wqe["packet_tx_started_at_ps"] = []


def _evaluate(
    observations: dict[str, Any],
    expectations: dict[str, Any],
    v1_observations: dict[str, Any],
    *,
    include_negative_controls: bool,
    bypass_binary_hashes: Mapping[str, tuple[str, str]] | None,
) -> dict[str, Any]:
    single, fifo = _prepare_cells(observations, expectations)

    # Amendment 1 ordering: these cross-cell families consume raw observations
    # before any inherited or per-cell exact oracle can pin the same effects.
    families = {
        "doorbell_packet_to_live_chain": _score_doorbell(single, expectations),
        "link_rate_packet_to_live_chain": _score_link_rate(single, expectations),
    }

    packet_status = _packet_origin_status(
        single,
        fifo,
        expectations["packet_wire_quantum_bytes"],
    )
    exact_oracle_rows = _packet_exact_oracle(
        single,
        fifo,
        expectations["packet_wire_quantum_bytes"],
    )
    _require(
        exact_oracle_rows == expectations["exact_oracle_rows"],
        "Tier C exact packet oracle row counts drifted",
    )
    tier_b_expectations = _load_json(TIER_B_EXPECTATIONS, "Tier B expectations")
    inherited = check_tier_b_observations(
        _tier_b_projection(observations, v1_observations),
        tier_b_expectations,
        bypass_binary_hashes=bypass_binary_hashes,
    )
    _require(inherited["passed"] is True, "inherited Tier B live chain failed")

    negative_controls: dict[str, bool] = {}
    if include_negative_controls:
        mutators = {
            "acceptance_surrogate": _mutate_acceptance,
            "producer_constant": _mutate_constant,
            "missing_tx_start": _mutate_missing_tx,
        }
        for name in expectations["negative_controls"]:
            mutant = copy.deepcopy(observations)
            mutators[name](mutant)
            rejected = False
            try:
                _evaluate(
                    mutant,
                    expectations,
                    v1_observations,
                    include_negative_controls=False,
                    bypass_binary_hashes=bypass_binary_hashes,
                )
            except (TierCAcceptanceError, ValueError, TypeError):
                rejected = True
            _require(rejected, f"Tier C checker accepted {name} mutant")
            negative_controls[name] = rejected

    return {
        "schema": expectations["results_schema"],
        "passed": True,
        "expectation_commit": FREEZE_COMMIT,
        "behavioral_families": families,
        "genuine_risk": {
            name: {
                "plausible_failures": family["passed_instances"],
                "relations": family["expected_instances"],
                "evaluation": "raw_observations_before_exact_oracles",
            }
            for name, family in families.items()
        },
        "entailment_analysis": {
            "scored_evaluation": "raw_observations_before_exact_oracles",
            "origin_guard_scope": "within_cell_explicit_tx_start_equality_only",
            "origin_guard_entails_scored_relations": False,
            "inherited_tier_b_checker_order": "after_tier_c_scored_families",
            "exact_packet_oracle_order": "after_tier_c_scored_families",
        },
        "exact_oracle_rows": exact_oracle_rows,
        "fatal_unscored_invariants": {
            **packet_status,
            "packet_closed_forms": True,
            "inherited_tier_b_live_chain": inherited["passed"],
            "inherited_tier_b_fatal_invariants": all(
                inherited["fatal_unscored_invariants"].values()
            ),
        },
        "negative_controls": negative_controls,
        "inherited_tier_b": inherited,
    }


def check_observations(
    observations: dict[str, Any],
    expectations: dict[str, Any],
    v1_observations: dict[str, Any],
    *,
    bypass_binary_hashes: Mapping[str, tuple[str, str]] | None = None,
) -> dict[str, Any]:
    return _evaluate(
        observations,
        expectations,
        v1_observations,
        include_negative_controls=True,
        bypass_binary_hashes=bypass_binary_hashes,
    )


def _producer_command(
    producer: Path, expectations: Path, observations: Path
) -> list[str]:
    return [
        str(producer),
        "--factory",
        "htsim",
        "--expectations",
        str(expectations),
        "--observations",
        str(observations),
    ]


def run_acceptance(
    out: Path,
    producer: Path,
    v1_observations_path: Path,
) -> dict[str, Any]:
    out = out.resolve(strict=False)
    producer = producer.resolve(strict=True)
    v1_observations_path = v1_observations_path.resolve(strict=True)
    expectations_path = EXPECTATIONS.resolve(strict=True)
    _require(out.is_absolute(), "Tier C output path must be absolute")
    _require(producer.is_relative_to(out), "Tier C producer must reside under output")
    observations_path = out / "raw_observations.json"
    results_path = out / "results.json"
    _require(
        not observations_path.exists()
        and not results_path.exists()
        and not Path(f"{observations_path}.tmp").exists()
        and not Path(f"{results_path}.tmp").exists(),
        "Tier C output already contains observations or results",
    )
    command = _producer_command(producer, expectations_path, observations_path)
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    observations = _load_json(observations_path, "Tier C observations")
    expectations = _load_json(expectations_path, "Tier C expectations")
    v1_observations = _load_json(v1_observations_path, "ABI-v1 Tier B observations")
    report = check_observations(observations, expectations, v1_observations)
    report["expectations_sha256"] = hashlib.sha256(
        expectations_path.read_bytes()
    ).hexdigest()
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
    return report


__all__ = ["TierCAcceptanceError", "check_observations", "run_acceptance"]
