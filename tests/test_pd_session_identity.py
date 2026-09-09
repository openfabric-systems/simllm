"""Discrimination and nonmutation checks for complete session comparisons."""

from copy import deepcopy
from dataclasses import replace

import pytest

from examples.pd_session_identity_v1.run_study import (
    EXCLUDED,
    corruption_controls,
    difference_paths,
    exact_json_bytes,
    project_serialized,
    state_snapshot,
    timeline_checks,
)
from simllm.adapters.vllm.pd_session import VllmPdRequestResult
from simllm.core import DisaggregatedRequestTimeline, KvHandoffEvent
from simllm.core.step import RequestPhase, ScheduledRequest, StepRecord


@pytest.fixture
def request_result():
    handoff = KvHandoffEvent("client", 393216, 110, 110, 110, 130, 130, "declared-constant")
    timeline = DisaggregatedRequestTimeline(
        "client", 10, 20, 110, handoff, 135, (175, 205, 245, 280)
    )
    return VllmPdRequestResult(
        timeline,
        "prefill-0",
        "decode-0",
        "prefill:opaque:a",
        "decode:opaque:a",
        512,
        (1, 2, 3, 4),
        {"nested": {"kept": [1, 2]}},
        (),
        (),
    )


def test_complete_projection_changes_only_the_two_root_fields(request_result):
    raw = request_result.to_json()
    before = exact_json_bytes(raw)
    other = replace(
        request_result,
        prefill_internal_request_id="prefill:new",
        decode_internal_request_id="decode:new",
    )
    assert set(difference_paths(raw, other.to_json())) == {(name,) for name in EXCLUDED}
    assert exact_json_bytes(request_result.to_comparison_json()) == exact_json_bytes(
        other.to_comparison_json()
    )
    assert exact_json_bytes(request_result.to_json()) == before
    projected = request_result.to_comparison_json()
    assert projected["request"] == {key: value for key, value in raw.items() if key not in EXCLUDED}
    projected["request"]["kv_transfer_params"]["nested"]["kept"].append(99)
    assert exact_json_bytes(request_result.to_json()) == before


def test_every_preserved_scalar_and_future_member_discriminates(request_result):
    controls = corruption_controls(request_result.to_json())
    assert controls and all(row["discriminated"] for row in controls)
    paths = [row["path"] for row in controls]
    assert ["handoff", "kv_bytes"] in paths
    assert ["decode_token_completed_at_ps", 0] in paths
    assert ["future-member"] in paths
    assert ["nested-opaque-name"] in paths


def test_state_snapshot_observes_same_count_record_mutation(request_result):
    record = StepRecord(
        0, 0, [ScheduledRequest("client", RequestPhase.PREFILL, 8, context_length=8)]
    )
    result = replace(request_result, prefill_records=(record,))
    raw = exact_json_bytes(result.to_json())
    before = exact_json_bytes(state_snapshot(result))
    record.virtual_time_ps = 1
    assert exact_json_bytes(result.to_json()) == raw
    assert exact_json_bytes(state_snapshot(result)) != before


def test_nested_corruption_control_rejects_recursive_id_erasure(request_result, monkeypatch):
    original = VllmPdRequestResult.to_comparison_json

    def faulty_projection(self):
        projected = original(self)
        nested = projected["request"].get("future_metadata")
        if isinstance(nested, dict):
            nested.pop(EXCLUDED[0], None)
        return projected

    monkeypatch.setattr(VllmPdRequestResult, "to_comparison_json", faulty_projection)
    controls = corruption_controls(request_result.to_json())
    nested = next(row for row in controls if row["path"] == ["nested-opaque-name"])
    assert not nested["discriminated"]


@pytest.mark.parametrize("field", EXCLUDED)
@pytest.mark.parametrize("bad", [None, "", " ", 1, True])
def test_invalid_opaque_ids_cannot_be_erased(request_result, field, bad):
    with pytest.raises(ValueError, match=field):
        replace(request_result, **{field: bad}).to_comparison_json()


def test_missing_opaque_id_rejects_before_projection(request_result):
    for name in EXCLUDED:
        raw = request_result.to_json()
        del raw[name]
        before = deepcopy(raw)
        with pytest.raises(ValueError, match=name):
            project_serialized(raw)
        assert raw == before


def test_nested_opaque_name_and_new_pricing_fields_are_preserved(request_result):
    raw = request_result.to_json()
    raw["future_metadata"] = {EXCLUDED[0]: "nested-id", "implementation": "new"}
    raw["compute_pricing"] = {"record_sha256": "a" * 64, "float_metadata": 0.5}
    projected = project_serialized(raw)
    assert projected["request"]["future_metadata"] == raw["future_metadata"]
    assert projected["request"]["compute_pricing"] == raw["compute_pricing"]
    changed = deepcopy(raw)
    changed["compute_pricing"]["record_sha256"] = "b" * 64
    assert exact_json_bytes(projected) != exact_json_bytes(project_serialized(changed))


def test_comparison_bytes_preserve_unicode_and_json_number_types(request_result):
    raw = request_result.to_json()
    first = {**raw, "future": "\u00e9"}
    second = {**raw, "future": "e\u0301"}
    assert exact_json_bytes(project_serialized(first)) != exact_json_bytes(
        project_serialized(second)
    )
    for first, second in ((True, 1), (1, 1.0), (0.0, -0.0)):
        assert exact_json_bytes({"value": first}) != exact_json_bytes({"value": second})
    assert difference_paths({"value": True}, {"value": 1}) == [("value",)]
    assert difference_paths({"value": 1}, {"value": 1.0}) == [("value",)]
    assert difference_paths({"value": 0.0}, {"value": -0.0}) == [("value",)]


def test_timeline_guard_joins_each_component_and_derived_metric(request_result):
    timeline = request_result.timeline
    row = {
        "request_result": request_result.to_json(),
        "label": "client",
        "handoff_ps": 20,
        "kv_bytes": 393216,
        "prefill_service_ps": timeline.prefill_service_ps,
        "decode_first_token_service_ps": timeline.decode_first_token_service_ps,
        "ttft_ps": timeline.ttft_ps,
        "tpot_ps": timeline.tpot_ps,
        "decomposition_total_ps": timeline.decomposition_total_ps,
    }
    assert timeline_checks(row)
    changed = deepcopy(row)
    changed["request_result"]["decomposition"]["prefill_queue_ps"] += 1
    assert not timeline_checks(changed)
    changed = deepcopy(row)
    changed["request_result"]["handoff"]["eligible_at_ps"] = 999
    assert not timeline_checks(changed)
    changed = deepcopy(row)
    changed["request_result"]["decode_token_completed_at_ps"] = []
    assert not timeline_checks(changed)


@pytest.mark.parametrize(
    "value", [{1: "not-a-json-key"}, {"nonfinite": float("nan")}, {"tuple": (1, 2)}]
)
def test_comparison_encoding_rejects_non_json_inputs(value):
    with pytest.raises((ValueError, TypeError)):
        exact_json_bytes(value)
