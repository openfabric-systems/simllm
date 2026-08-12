"""Per-request routing lifetime state and fail-closed completion tests."""

from __future__ import annotations

from dataclasses import replace

import pytest

from simllm.core import (
    CollectiveWork,
    CompletionEvent,
    EventPhase,
    ExecutionGraph,
    ExecutionOperation,
    ExecutionResult,
    JoinProvenance,
    OperationCorrelation,
    RequestLifecycleError,
    RequestLifecycleState,
    RequestLifetimeRegistry,
    RequestPhase,
    RoutingViewDescriptor,
    ScheduledRequest,
    StepRecord,
)


def _registry(
    released: list[str],
    *,
    token_count: int = 2,
    prompt_token_count: int = 1,
) -> RequestLifetimeRegistry:
    registry = RequestLifetimeRegistry((0, 2))
    registry.register(
        request_id="request",
        provenance=JoinProvenance(
            run_schema="run-v1",
            trace_schema="trace-v1",
            trace_sha256="a" * 64,
        ),
        arrived_at_ps=7,
        view=RoutingViewDescriptor(
            arena_id="arena",
            token_offset=5,
            token_count=token_count,
            prompt_token_count=prompt_token_count,
            release_callback=lambda: released.append("request"),
        ),
    )
    return registry


def _record(
    step: int,
    *,
    context: int | None = None,
    new_tokens: int = 1,
    cached_tokens: int = 0,
    finished: bool = False,
) -> StepRecord:
    scheduled = []
    if context is not None:
        scheduled.append(
            ScheduledRequest(
                "request",
                RequestPhase.PREFILL if step == 0 else RequestPhase.DECODE,
                new_tokens,
                num_cached_tokens=cached_tokens,
                context_length=context,
            )
        )
    return StepRecord(
        step_index=step,
        virtual_time_ps=step * 10,
        scheduled=scheduled,
        finished_request_ids=["request"] if finished else [],
        num_sampled=len(scheduled),
        sampled_request_ids=["request"] if scheduled else [],
    )


def _execution(
    record: StepRecord,
    *,
    logical_completions: bool = True,
) -> tuple[ExecutionGraph, ExecutionResult]:
    operations = []
    events = []
    for layer in (0, 2):
        for phase in ("dispatch", "combine"):
            operation_id = f"step-{record.step_index}:{phase}:{layer}"
            operations.append(
                ExecutionOperation(
                    operation_id=operation_id,
                    rank=0,
                    logical_queue="collective",
                    work=CollectiveWork(
                        collective="all-to-allv",
                        ranks=(0, 1),
                        payload_bytes=0,
                        algorithm_hint="pairwise",
                        channel_hint=phase,
                        pair_payload_bytes=((0, 1, 1),),
                    ),
                    correlation=OperationCorrelation(
                        request_ids=("request",),
                        layer=layer,
                    ),
                )
            )
            events.append(
                CompletionEvent(
                    execution_id=f"step-{record.step_index}",
                    operation_id=operation_id,
                    phase=EventPhase.COMPLETED,
                    timestamp_ps=record.virtual_time_ps + 1,
                    subject_object_id=(None if logical_completions else "wqe"),
                )
            )
    graph = ExecutionGraph(
        execution_id=f"step-{record.step_index}",
        step_index=record.step_index,
        released_at_ps=record.virtual_time_ps,
        operations=tuple(operations),
    )
    return graph, ExecutionResult(
        execution_id=graph.execution_id,
        completed_at_ps=record.virtual_time_ps + 1,
        events=tuple(events),
    )


def _empty_execution(record: StepRecord) -> tuple[ExecutionGraph, ExecutionResult]:
    graph = ExecutionGraph(
        execution_id=f"step-{record.step_index}",
        step_index=record.step_index,
        released_at_ps=record.virtual_time_ps,
    )
    return graph, ExecutionResult(
        execution_id=graph.execution_id,
        completed_at_ps=record.virtual_time_ps,
    )


def test_delayed_scheduler_finish_closes_only_after_both_phase_masks() -> None:
    released: list[str] = []
    registry = _registry(released)

    first = _record(0, context=1)
    graph, result = _empty_execution(first)
    registry.consume_step(first, graph, result)
    lifetime = registry.by_request_id("request")
    assert lifetime.state is RequestLifecycleState.EXECUTING
    assert lifetime.consumption_cursor == 1
    assert lifetime.dispatch_end_mask == 0

    final = _record(1, context=2)
    graph, result = _execution(final)
    registry.consume_step(final, graph, result)
    lifetime = registry.by_request_id("request")
    assert lifetime.execution_drained
    assert lifetime.state is RequestLifecycleState.EXECUTING
    assert registry.live_view_count == 1

    drain = _record(2, context=None, finished=True)
    graph, result = _empty_execution(drain)
    registry.consume_step(drain, graph, result)
    lifetime = registry.by_request_id("request")
    assert lifetime.state is RequestLifecycleState.CLOSED
    assert lifetime.view_released
    assert released == ["request"]
    assert registry.closed_request_count == 1
    assert registry.live_request_count == 0
    assert registry.live_view_count == 0
    registry.audit_closed()


def test_suppressed_bit_fails_closed_with_request_phase_and_layer() -> None:
    class SuppressedRegistry(RequestLifetimeRegistry):
        def _mark_end_flag(self, lifetime, phase, layer):
            if phase == "combine" and layer == 2:
                return
            super()._mark_end_flag(lifetime, phase, layer)

    released: list[str] = []
    base = _registry(released, token_count=1)
    registry = SuppressedRegistry((0, 2))
    lifetime = base.by_request_id("request")
    registry.register(
        request_id=lifetime.request_id,
        provenance=lifetime.provenance,
        arrived_at_ps=lifetime.arrived_at_ps,
        view=lifetime.view,
    )
    record = _record(0, context=1, finished=True)
    graph, result = _execution(record)
    registry.consume_step(record, graph, result)

    raw = registry.by_request_id("request")
    assert raw.state is RequestLifecycleState.FINISH_FLAGGED
    assert raw.missing_layers("dispatch") == ()
    assert raw.missing_layers("combine") == (2,)
    assert registry.live_view_count == 1
    with pytest.raises(RequestLifecycleError) as caught:
        registry.audit_closed()
    diagnostic = str(caught.value)
    assert "request" in diagnostic
    assert "combine" in diagnostic
    assert "2" in diagnostic
    assert released == []


def test_subject_wqe_completion_is_not_a_request_end_flag() -> None:
    released: list[str] = []
    registry = _registry(released, token_count=1)
    record = _record(0, context=1, finished=True)
    graph, result = _execution(record, logical_completions=False)
    registry.consume_step(record, graph, result)

    lifetime = registry.by_request_id("request")
    assert lifetime.dispatch_end_mask == 0
    assert lifetime.combine_end_mask == 0
    with pytest.raises(RequestLifecycleError, match="dispatch missing layers"):
        registry.audit_closed()


def test_cached_admission_and_idempotent_recompute_advance_unique_coverage() -> None:
    released: list[str] = []
    registry = _registry(released, token_count=5, prompt_token_count=3)
    cached = _record(0, context=3, new_tokens=1, cached_tokens=2)
    graph, result = _empty_execution(cached)
    registry.consume_step(cached, graph, result)
    assert registry.by_request_id("request").consumption_cursor == 3

    replay = replace(cached, step_index=1, virtual_time_ps=10)
    graph, result = _empty_execution(replay)
    registry.consume_step(replay, graph, result)
    assert registry.by_request_id("request").consumption_cursor == 3

    overflow = _record(2, context=6)
    graph, result = _empty_execution(overflow)
    with pytest.raises(RequestLifecycleError, match="cursor overflow"):
        registry.consume_step(overflow, graph, result)
    assert registry.by_request_id("request").consumption_cursor == 3


def test_premature_release_and_unknown_finish_are_atomic_fatal_errors() -> None:
    released: list[str] = []
    registry = _registry(released)
    with pytest.raises(RequestLifecycleError, match="before CLOSED"):
        registry.by_request_id("request").release_view()

    record = StepRecord(
        step_index=0,
        virtual_time_ps=0,
        finished_request_ids=["unknown"],
        num_sampled=0,
        sampled_request_ids=[],
    )
    graph, result = _empty_execution(record)
    with pytest.raises(RequestLifecycleError, match="absent"):
        registry.consume_step(record, graph, result)
    assert registry.by_request_id("request").state is RequestLifecycleState.JOINED
    assert released == []
