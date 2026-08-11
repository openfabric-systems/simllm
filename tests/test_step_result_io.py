import copy
import json
from fractions import Fraction

import pytest

from simllm.core import (
    LEGACY_RESULT_SCHEMA,
    RESULT_SCHEMA,
    AdditiveVisitTotals,
    LatencyAttribution,
    RequestMetric,
    RequestPhase,
    StepResult,
    step_result_from_json,
    step_result_to_json,
)


def _additive(multiplier: int = 1) -> AdditiveVisitTotals:
    return AdditiveVisitTotals(
        queue_wait_ps=2 * multiplier,
        service_ps=3 * multiplier,
        visibility_ps=5 * multiplier,
        visit_count=multiplier,
    )


def _prefill() -> RequestMetric:
    return RequestMetric(
        request_id="prefill",
        phase=RequestPhase.PREFILL,
        token_index=1,
        completed_at_ps=107,
        latency_ps=7,
        ttft_ps=7,
        tpot_ps=None,
        attribution=LatencyAttribution(kernel_ps=7),
        additive_visit_totals=_additive(),
    )


def _decode() -> RequestMetric:
    return RequestMetric(
        request_id="decode",
        phase=RequestPhase.DECODE,
        token_index=3,
        completed_at_ps=109,
        latency_ps=9,
        ttft_ps=5,
        tpot_ps=Fraction(1, 3),
        attribution=LatencyAttribution(queue_ps=2, nic_ps=7),
        additive_visit_totals=_additive(),
    )


@pytest.mark.parametrize(
    "result",
    [
        StepResult(0, 0, 100),
        StepResult(1, 7, 107, (_prefill(),), _additive()),
        StepResult(2, 9, 109, (_decode(),), _additive()),
        StepResult(
            3,
            9,
            109,
            (_prefill(), _decode()),
            _additive(2),
        ),
    ],
    ids=("empty", "prefill", "decode", "mixed"),
)
def test_full_step_result_round_trips_through_real_json(result):
    payload = step_result_to_json(result)
    wire = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    assert step_result_from_json(json.loads(wire)) == result


def test_nonterminating_tpot_uses_exact_reduced_rational():
    payload = step_result_to_json(StepResult(2, 9, 109, (_decode(),)))
    assert payload["request_metrics"][0]["tpot_ps"] == {
        "numerator": 1,
        "denominator": 3,
    }
    assert "0.333" not in json.dumps(payload)


def test_canonical_shape_carries_separate_attribution_and_visit_totals():
    payload = step_result_to_json(
        StepResult(3, 9, 109, (_prefill(), _decode()), _additive(2))
    )
    assert payload["schema"] == RESULT_SCHEMA
    assert set(payload) == {
        "schema",
        "step_index",
        "step_latency_ps",
        "completed_at_ps",
        "request_metrics",
        "additive_visit_totals",
    }
    metric = payload["request_metrics"][1]
    assert set(metric["attribution"]) == {
        "queue_ps",
        "kv_ps",
        "kernel_ps",
        "dma_ps",
        "collective_ps",
        "nic_ps",
        "control_ps",
    }
    assert set(metric["additive_visit_totals"]) == {
        "queue_wait_ps",
        "service_ps",
        "visibility_ps",
        "visit_count",
    }


def _decode_payload():
    return step_result_to_json(StepResult(2, 9, 109, (_decode(),), _additive()))


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda value: value.update(extra=1), "unknown fields"),
        (lambda value: value.pop("completed_at_ps"), "missing fields"),
        (lambda value: value.update(step_index=True), "expected an integer"),
        (
            lambda value: value["request_metrics"][0]["attribution"].update(
                nic_ps=8
            ),
            "does not conserve",
        ),
        (
            lambda value: value["request_metrics"][0]["tpot_ps"].update(
                denominator=0
            ),
            "at least 1",
        ),
        (
            lambda value: value["request_metrics"][0]["tpot_ps"].update(
                numerator=2, denominator=6
            ),
            "must be reduced",
        ),
        (
            lambda value: value["request_metrics"].append(
                copy.deepcopy(value["request_metrics"][0])
            ),
            "at most one row per request",
        ),
    ],
)
def test_strict_reader_rejects_invalid_full_results(mutate, match):
    payload = _decode_payload()
    mutate(payload)
    with pytest.raises((TypeError, ValueError), match=match):
        step_result_from_json(payload)


def test_legacy_name_is_not_misrepresented_as_an_accepted_payload():
    payload = _decode_payload()
    payload["schema"] = LEGACY_RESULT_SCHEMA
    with pytest.raises(ValueError, match="no accepted legacy payload exists"):
        step_result_from_json(payload)


def test_writer_revalidates_mutable_step_result():
    result = StepResult(0, 0, 100)
    result.step_index = True
    with pytest.raises(ValueError, match="expected an integer"):
        step_result_to_json(result)
