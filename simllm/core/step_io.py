"""Strict JSON codec for the complete :class:`StepResult` contract."""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Any, cast

from simllm.core._wire import (
    _array,
    _enum_value,
    _fail,
    _fields,
    _integer,
    _object,
    _string,
)
from simllm.core.step import (
    LEGACY_RESULT_SCHEMA,
    RESULT_SCHEMA,
    AdditiveVisitTotals,
    LatencyAttribution,
    RequestMetric,
    RequestPhase,
    StepResult,
)

_ATTRIBUTION_FIELDS = (
    "queue_ps",
    "kv_ps",
    "kernel_ps",
    "dma_ps",
    "collective_ps",
    "nic_ps",
    "control_ps",
)
_ADDITIVE_FIELDS = (
    "queue_wait_ps",
    "service_ps",
    "visibility_ps",
    "visit_count",
)
_METRIC_FIELDS = {
    "request_id",
    "phase",
    "token_index",
    "completed_at_ps",
    "latency_ps",
    "ttft_ps",
    "tpot_ps",
    "attribution",
    "additive_visit_totals",
}
_RESULT_FIELDS = {
    "schema",
    "step_index",
    "step_latency_ps",
    "completed_at_ps",
    "request_metrics",
    "additive_visit_totals",
}


def _attribution_to_json(value: LatencyAttribution) -> dict[str, int]:
    if not isinstance(value, LatencyAttribution):
        _fail("result.request_metrics.attribution", "expected LatencyAttribution")
    return {name: cast(int, getattr(value, name)) for name in _ATTRIBUTION_FIELDS}


def _attribution_from_json(value: Any, path: str) -> LatencyAttribution:
    payload = _object(value, path)
    fields = set(_ATTRIBUTION_FIELDS)
    _fields(payload, path, required=fields)
    return LatencyAttribution(
        **{
            name: _integer(payload[name], f"{path}.{name}", nonnegative=True)
            for name in _ATTRIBUTION_FIELDS
        }
    )


def _additive_to_json(value: AdditiveVisitTotals) -> dict[str, int]:
    if not isinstance(value, AdditiveVisitTotals):
        _fail(
            "result.additive_visit_totals",
            "expected AdditiveVisitTotals",
        )
    return {name: cast(int, getattr(value, name)) for name in _ADDITIVE_FIELDS}


def _additive_from_json(value: Any, path: str) -> AdditiveVisitTotals:
    payload = _object(value, path)
    fields = set(_ADDITIVE_FIELDS)
    _fields(payload, path, required=fields)
    return AdditiveVisitTotals(
        **{
            name: _integer(payload[name], f"{path}.{name}", nonnegative=True)
            for name in _ADDITIVE_FIELDS
        }
    )


def _fraction_to_json(value: Fraction | None) -> dict[str, int] | None:
    if value is None:
        return None
    if not isinstance(value, Fraction):
        _fail("result.request_metrics.tpot_ps", "expected Fraction or null")
    if value < 0:
        _fail("result.request_metrics.tpot_ps", "must be nonnegative")
    return {"numerator": value.numerator, "denominator": value.denominator}


def _fraction_from_json(value: Any, path: str) -> Fraction | None:
    if value is None:
        return None
    payload = _object(value, path)
    _fields(payload, path, required={"numerator", "denominator"})
    numerator = _integer(payload["numerator"], f"{path}.numerator", nonnegative=True)
    denominator = _integer(payload["denominator"], f"{path}.denominator", minimum=1)
    if math.gcd(numerator, denominator) != 1:
        _fail(path, "fraction must be reduced")
    if numerator == 0 and denominator != 1:
        _fail(path, "zero must use denominator 1")
    return Fraction(numerator, denominator)


def _metric_to_json(metric: RequestMetric, path: str) -> dict[str, Any]:
    if not isinstance(metric, RequestMetric):
        _fail(path, "expected RequestMetric")
    if not isinstance(metric.phase, RequestPhase):
        _fail(f"{path}.phase", "expected RequestPhase")
    return {
        "request_id": metric.request_id,
        "phase": metric.phase.value,
        "token_index": metric.token_index,
        "completed_at_ps": metric.completed_at_ps,
        "latency_ps": metric.latency_ps,
        "ttft_ps": metric.ttft_ps,
        "tpot_ps": _fraction_to_json(metric.tpot_ps),
        "attribution": _attribution_to_json(metric.attribution),
        "additive_visit_totals": _additive_to_json(
            metric.additive_visit_totals
        ),
    }


def _metric_from_json(value: Any, path: str) -> RequestMetric:
    payload = _object(value, path)
    _fields(payload, path, required=_METRIC_FIELDS)
    return RequestMetric(
        request_id=_string(payload["request_id"], f"{path}.request_id"),
        phase=_enum_value(RequestPhase, payload["phase"], f"{path}.phase"),
        token_index=_integer(payload["token_index"], f"{path}.token_index", minimum=1),
        completed_at_ps=_integer(
            payload["completed_at_ps"],
            f"{path}.completed_at_ps",
            nonnegative=True,
        ),
        latency_ps=_integer(
            payload["latency_ps"], f"{path}.latency_ps", nonnegative=True
        ),
        ttft_ps=_integer(payload["ttft_ps"], f"{path}.ttft_ps", nonnegative=True),
        tpot_ps=_fraction_from_json(payload["tpot_ps"], f"{path}.tpot_ps"),
        attribution=_attribution_from_json(
            payload["attribution"], f"{path}.attribution"
        ),
        additive_visit_totals=_additive_from_json(
            payload["additive_visit_totals"],
            f"{path}.additive_visit_totals",
        ),
    )


def _validate_step_result(result: StepResult) -> None:
    if not isinstance(result, StepResult):
        _fail("result", "expected StepResult")
    for name in ("step_index", "step_latency_ps", "completed_at_ps"):
        _integer(getattr(result, name), f"result.{name}", nonnegative=True)
    if not isinstance(result.request_metrics, tuple):
        _fail("result.request_metrics", "in-memory contract requires a tuple")
    request_ids: list[str] = []
    for index, metric in enumerate(result.request_metrics):
        path = f"result.request_metrics[{index}]"
        _metric_to_json(metric, path)
        if metric.completed_at_ps > result.completed_at_ps:
            _fail(path, "completion exceeds result.completed_at_ps")
        request_ids.append(metric.request_id)
    if len(request_ids) != len(set(request_ids)):
        _fail("result.request_metrics", "contains duplicate request IDs")
    if result.additive_visit_totals is not None:
        _additive_to_json(result.additive_visit_totals)


def step_result_to_json(result: StepResult) -> dict[str, Any]:
    """Return the canonical JSON-ready form of one full step result."""

    _validate_step_result(result)
    return {
        "schema": RESULT_SCHEMA,
        "step_index": result.step_index,
        "step_latency_ps": result.step_latency_ps,
        "completed_at_ps": result.completed_at_ps,
        "request_metrics": [
            _metric_to_json(metric, f"result.request_metrics[{index}]")
            for index, metric in enumerate(result.request_metrics)
        ],
        "additive_visit_totals": (
            None
            if result.additive_visit_totals is None
            else _additive_to_json(result.additive_visit_totals)
        ),
    }


def step_result_from_json(value: Any) -> StepResult:
    """Parse and validate one canonical full step-result payload."""

    payload = _object(value, "result")
    if "schema" not in payload:
        _fail("result", "missing fields ['schema']")
    schema = _string(payload["schema"], "result.schema")
    if schema == LEGACY_RESULT_SCHEMA:
        _fail(
            "result.schema",
            "unsupported legacy schema 'atlahs-closed-loop-result-v1'; "
            "no accepted legacy payload exists to upgrade",
        )
    if schema != RESULT_SCHEMA:
        _fail(
            "result.schema",
            f"unsupported schema {schema!r}; expected {RESULT_SCHEMA!r}",
        )
    _fields(payload, "result", required=_RESULT_FIELDS)
    metrics = tuple(
        _metric_from_json(entry, f"result.request_metrics[{index}]")
        for index, entry in enumerate(
            _array(payload["request_metrics"], "result.request_metrics")
        )
    )
    additive_value = payload["additive_visit_totals"]
    additive = (
        None
        if additive_value is None
        else _additive_from_json(additive_value, "result.additive_visit_totals")
    )
    result = StepResult(
        step_index=_integer(
            payload["step_index"], "result.step_index", nonnegative=True
        ),
        step_latency_ps=_integer(
            payload["step_latency_ps"],
            "result.step_latency_ps",
            nonnegative=True,
        ),
        completed_at_ps=_integer(
            payload["completed_at_ps"],
            "result.completed_at_ps",
            nonnegative=True,
        ),
        request_metrics=metrics,
        additive_visit_totals=additive,
    )
    _validate_step_result(result)
    return result


__all__ = ["step_result_from_json", "step_result_to_json"]
