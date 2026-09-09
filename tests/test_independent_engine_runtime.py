"""Independent engine causality, receipt integrity and failure containment."""

from copy import deepcopy
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import pytest

from simllm.core.clock import VirtualClock
from simllm.core.engine_steps import EngineStepRuntime, PublicationBinding
from simllm.core.execution import EventPhase, ResourceKind
from simllm.core.step import RequestPhase, ScheduledRequest, StepRecord, StepResult
from simllm.core.value_snapshot import value_snapshot


def step(runtime, engine, duration, *, index=0, publish=None, guard=lambda: None):
    now = runtime.clock.now_ps
    record = StepRecord(index, now, [ScheduledRequest(engine + ":request", RequestPhase.DECODE, 1)])
    result = StepResult(index, duration, now + duration)
    receipt = runtime.submit(engine, record, result, guard=guard, publish=publish or (lambda result: None))
    return receipt, record, result


def test_short_later_step_completes_before_long_step_without_early_visibility():
    clock, output = VirtualClock(), []
    runtime = EngineStepRuntime(clock)
    long, _, _ = step(runtime, "long", 10_000, publish=lambda result: output.append(("long", clock.now_ps)))
    assert not runtime.visits and not runtime.results and output == []
    assert [event.phase for event in runtime.events] == [EventPhase.SUBMITTED, EventPhase.QUEUED, EventPhase.STARTED]
    runtime.advance_to(1000)
    short, _, _ = step(runtime, "short", 1000, publish=lambda result: output.append(("short", clock.now_ps)))
    runtime.advance_to(1999)
    assert runtime.complete_due() == () and output == []
    runtime.advance_to(2000)
    assert runtime.complete_due() == (short,)
    assert output == [("short", 2000)] and runtime.pending_for("long") is long
    runtime.advance_to(10_000)
    assert runtime.complete_due() == (long,)
    assert output == [("short", 2000), ("long", 10_000)]
    assert sum(visit.service_ps for visit in runtime.visits) == 11_000 > clock.now_ps
    assert all(visit.queue_wait_ps == visit.visibility_ps == 0 for visit in runtime.visits)
    assert all(visit.resource.kind is ResourceKind.GPU_WORK_QUEUE for visit in runtime.visits)
    projection = runtime.results
    projection[0][1].completed_at_ps = 0
    assert runtime.results[0][1].completed_at_ps == 2000
    runtime.close()


def test_tied_completions_retire_in_insertion_order_before_reuse():
    runtime, output = EngineStepRuntime(VirtualClock()), []
    first, _, _ = step(runtime, "a", 4000, publish=lambda result: output.append("a"))
    runtime.advance_to(1000)
    second, _, _ = step(runtime, "b", 3000, publish=lambda result: output.append("b"))
    runtime.advance_to(4000)
    assert runtime.complete_due() == (first, second) and output == ["a", "b"]
    third, _, _ = step(runtime, "a", 1000, index=1, publish=lambda result: output.append("c"))
    runtime.advance_to(5000)
    assert runtime.complete_due() == (third,) and output == ["a", "b", "c"]


def test_finished_only_drain_completes_once_without_time_advance():
    runtime, output = EngineStepRuntime(VirtualClock(3000)), []
    record = StepRecord(0, 3000, finished_request_ids=["done"])
    receipt = runtime.submit("engine", record, StepResult(0, 0, 3000), guard=lambda: None,
                             publish=lambda result: output.append(result.completed_at_ps))
    assert runtime.complete_due() == (receipt,) and output == [3000]
    assert runtime.clock.now_ps == 3000 and runtime.visits[0].service_ps == 0
    with pytest.raises(RuntimeError, match="already consumed"):
        runtime.complete(receipt)
    assert output == [3000] and runtime.failure is not None


@pytest.mark.parametrize("duration,finished", [(0, []), (1, ["done"])])
def test_phantom_or_nonzero_drain_cannot_create_a_visit(duration, finished):
    runtime = EngineStepRuntime(VirtualClock())
    record = StepRecord(0, 0, finished_request_ids=finished)
    with pytest.raises(ValueError, match="drain"):
        runtime.submit("engine", record, StepResult(0, duration, duration),
                       guard=lambda: None, publish=lambda result: None)
    assert runtime.failure and not runtime.events and not len(runtime.clock)


@pytest.mark.parametrize("mutation", ["input", "price", "same-engine", "foreign", "early", "skip", "external-clock", "external-event"])
def test_pending_work_rejects_mutation_and_poisoned_run_cannot_retry(mutation):
    runtime, output = EngineStepRuntime(VirtualClock()), []
    receipt, record, result = step(runtime, "a", 1000, publish=lambda result: output.append(result))
    with pytest.raises((RuntimeError, ValueError)):
        if mutation == "input":
            record.scheduled[0].context_length += 1
            runtime.advance_to(1000)
            runtime.complete(receipt)
        elif mutation == "price":
            result.step_latency_ps += 1
            result.completed_at_ps += 1
            runtime.advance_to(1000)
            runtime.complete(receipt)
        elif mutation == "same-engine":
            step(runtime, "a", 2000, index=1)
        elif mutation == "foreign":
            runtime.advance_to(1000)
            runtime.complete(deepcopy(receipt))
        elif mutation == "early":
            runtime.complete(receipt)
        elif mutation == "skip":
            runtime.advance_to(1001)
        elif mutation == "external-clock":
            runtime.clock.advance_to(500)
            runtime.advance_to(1000)
        else:
            runtime.clock.schedule(1000, object())
            runtime.advance_to(1000)
    assert runtime.failure and output == [] and not runtime.visits
    with pytest.raises(RuntimeError, match="poisoned"):
        runtime.complete_due()


def test_due_order_cannot_be_chosen_by_the_caller():
    runtime = EngineStepRuntime(VirtualClock())
    step(runtime, "a", 1000)
    other, _, _ = step(runtime, "b", 1000)
    runtime.advance_to(1000)
    with pytest.raises(RuntimeError, match="due order"):
        runtime.complete(other)


def test_second_authority_and_pending_reset_are_rejected():
    clock = VirtualClock()
    runtime = EngineStepRuntime(clock)
    with pytest.raises(ValueError, match="already has"):
        EngineStepRuntime(clock)
    step(runtime, "a", 1000)
    with pytest.raises(RuntimeError, match="CORE-69"):
        runtime.close()
    assert len(clock) == 1 and runtime.pending_for("a") is not None


def test_callback_failure_is_not_retried_or_reported_as_completion():
    runtime, calls = EngineStepRuntime(VirtualClock()), []

    def fail(result):
        calls.append(result.completed_at_ps)
        raise RuntimeError("native update failed")

    step(runtime, "a", 1000, publish=fail)
    runtime.advance_to(1000)
    with pytest.raises(RuntimeError, match="native update failed"):
        runtime.complete_due()
    assert calls == [1000] and runtime.visits == () and runtime.results == ()
    assert all(event.phase is not EventPhase.COMPLETED for event in runtime.events)
    with pytest.raises(RuntimeError, match="poisoned"):
        runtime.complete_due()
    assert calls == [1000]


def test_guard_must_preserve_input_value_before_reserving_clock_work():
    runtime = EngineStepRuntime(VirtualClock())
    record = StepRecord(0, 0, [ScheduledRequest("a", RequestPhase.DECODE, 1)])
    result = StepResult(0, 1000, 1000)

    def mutate():
        result.step_latency_ps += 1

    with pytest.raises(ValueError, match="guard changed"):
        runtime.submit("a", record, result, guard=mutate, publish=lambda result: None)
    assert not len(runtime.clock) and not runtime.events


def test_typed_snapshot_includes_compare_false_fields_and_nested_provider_values():
    @dataclass
    class Config:
        selection: str
        resolved: dict = field(compare=False)

    config = Config("selected", {"efficiency": 0.7, "scales": [1, Fraction(1, 2)], "path": Path("profiles")})
    original = deepcopy(config)
    snapshot = value_snapshot(config)
    config.resolved["efficiency"] = 0.8
    assert config == original and value_snapshot(config) != snapshot
    assert value_snapshot(1) != value_snapshot(True)
    assert value_snapshot([1]) != value_snapshot((1,))
    with pytest.raises(ValueError, match="finite"):
        value_snapshot(float("nan"))
    recursive = []
    recursive.append(recursive)
    with pytest.raises(ValueError, match="cycle"):
        value_snapshot(recursive)
    with pytest.raises(TypeError, match="unsupported"):
        value_snapshot(object())


def test_frozen_authority_name_and_due_before_submission():
    runtime = EngineStepRuntime(VirtualClock())
    assert runtime.authority == "simllm-independent-engine-completion-v1"
    step(runtime, "a", 1000)
    runtime.advance_to(1000)
    before = runtime.events
    with pytest.raises(ValueError, match="due completions must retire"):
        step(runtime, "b", 1000)
    assert runtime.failure and runtime.events == before


@pytest.mark.parametrize("member,value", [("step_index", 99), ("engine_id", "other"),
                                        ("sequence", 9), ("submitted_at_ps", 1), ("completed_at_ps", 1001)])
def test_same_receipt_object_cannot_change_its_authoritative_values(member, value):
    runtime, output = EngineStepRuntime(VirtualClock()), []
    receipt, _, _ = step(runtime, "a", 1000, publish=lambda result: output.append(result))
    runtime.advance_to(1000)
    object.__setattr__(receipt, member, value)
    with pytest.raises(RuntimeError, match="receipt values changed"):
        runtime.complete(receipt)
    assert runtime.failure and output == [] and runtime.results == ()
    assert [event.operation_id for event in runtime.events] == ["step-0"] * 3


def test_publication_binding_is_copied_privately_and_must_be_consumed():
    runtime = EngineStepRuntime(VirtualClock())
    record = StepRecord(0, 0, [ScheduledRequest("a", RequestPhase.DECODE, 1)])
    result = StepResult(0, 1000, 1000)
    owner, payload = object(), {"data": 1}
    binding = PublicationBinding(owner, payload, lambda item: item)
    runtime.submit("a", record, result, guard=lambda: None, publish=lambda result: None,
                   publications=(binding,))
    object.__setattr__(binding, "payload", {"data": 2})
    runtime.advance_to(1000)
    with pytest.raises(ValueError, match="omitted a bound publication"):
        runtime.complete_due()
    assert runtime.failure and runtime.results == ()


def test_unsupported_receipt_value_poisoning_prevents_repaired_retry():
    runtime = EngineStepRuntime(VirtualClock())
    receipt, _, _ = step(runtime, "a", 1000)
    object.__setattr__(receipt, "sequence", object())
    with pytest.raises(TypeError, match="unsupported deferred value"):
        runtime.advance_to(1000)
    assert runtime.failure and runtime.clock.now_ps == 0
    object.__setattr__(receipt, "sequence", 0)
    with pytest.raises(RuntimeError, match="poisoned"):
        runtime.advance_to(1000)
    assert runtime.results == ()
