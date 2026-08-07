import json
from pathlib import Path

import pytest

from simllm.core import (
    STEP_SCHEMA,
    RequestPhase,
    ScheduledRequest,
    StepRecord,
    StepResult,
    step_record_from_json,
    step_record_to_json,
    step_records_from_jsonl,
    write_step_records,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
M2_SMOKE_JSONL = _REPO_ROOT / "examples" / "m4" / "fixtures" / "vllm-m2-steps.jsonl"
M3_SMOKE_JSONL = _REPO_ROOT / "examples" / "m4" / "fixtures" / "sglang-m3-steps.jsonl"


def test_step_record_totals():
    step = StepRecord(
        step_index=3,
        virtual_time_ps=1_000_000,
        scheduled=[
            ScheduledRequest("a", RequestPhase.PREFILL, num_new_tokens=512, num_cached_tokens=128),
            ScheduledRequest("b", RequestPhase.DECODE, num_new_tokens=1, context_length=901),
        ],
        finished_request_ids=["c"],
    )
    assert step.total_new_tokens == 513
    assert step.scheduled[0].phase is RequestPhase.PREFILL
    assert step.scheduled[0].num_cached_tokens == 128
    assert step.preempted_request_ids == []


def test_step_result_carries_virtual_time():
    result = StepResult(step_index=3, step_latency_ps=42_000_000, completed_at_ps=1_042_000_000)
    assert result.completed_at_ps - result.step_latency_ps == 1_000_000_000


def test_step_record_json_round_trip():
    record = StepRecord(
        step_index=7,
        virtual_time_ps=123_456_789,
        scheduled=[
            ScheduledRequest("a", RequestPhase.PREFILL, num_new_tokens=512,
                             num_cached_tokens=128, context_length=640),
            ScheduledRequest("b", RequestPhase.DECODE, num_new_tokens=1, context_length=901),
        ],
        preempted_request_ids=["p"],
        finished_request_ids=["c", "d"],
    )
    payload = step_record_to_json(record)
    assert payload["schema"] == STEP_SCHEMA
    back = step_record_from_json(payload)
    assert back == record
    assert back.scheduled[0].phase is RequestPhase.PREFILL


def test_step_record_from_json_rejects_unknown_schema():
    record = StepRecord(step_index=0, virtual_time_ps=0)
    payload = step_record_to_json(record)
    payload["schema"] = "atlahs-closed-loop-step-v2"
    with pytest.raises(ValueError, match="schema"):
        step_record_from_json(payload)
    with pytest.raises(ValueError, match="schema"):
        step_record_from_json({"step_index": 0, "virtual_time_ps": 0})


def test_step_records_jsonl_round_trip(tmp_path):
    records = [
        StepRecord(step_index=0, virtual_time_ps=0, scheduled=[
            ScheduledRequest("a", RequestPhase.PREFILL, num_new_tokens=4, context_length=4)]),
        StepRecord(step_index=1, virtual_time_ps=1_000, scheduled=[
            ScheduledRequest("a", RequestPhase.DECODE, num_new_tokens=1, context_length=5)],
            finished_request_ids=["z"]),
    ]
    path = write_step_records(records, tmp_path / "steps.jsonl")
    assert step_records_from_jsonl(path) == records


def test_step_records_jsonl_names_bad_line(tmp_path):
    path = tmp_path / "bad.jsonl"
    good = step_record_to_json(StepRecord(step_index=0, virtual_time_ps=0))
    path.write_text(json.dumps(good) + "\n" + '{"schema": "nope"}' + "\n")
    with pytest.raises(ValueError, match=":2:"):
        step_records_from_jsonl(path)


@pytest.mark.parametrize(
    "path,expected_records",
    [(M2_SMOKE_JSONL, 8), (M3_SMOKE_JSONL, 9)],
    ids=["vllm-m2", "sglang-m3"],
)
def test_real_smoke_jsonl_loads(path, expected_records):
    records = step_records_from_jsonl(path)
    assert len(records) == expected_records
    times = [r.virtual_time_ps for r in records]
    assert times == sorted(times)
    assert all(r.scheduled for r in records)
    assert all(r.total_new_tokens >= 1 for r in records)
