from simllm.core import RequestPhase, ScheduledRequest, StepRecord, StepResult


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
