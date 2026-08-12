from dataclasses import replace

import pytest

from simllm.backends.dependency_cross_check import (
    complete_dependency_cross_check,
    plan_dependency_cross_check,
)
from simllm.core import CollectiveWork, ExecutionGraph, ExecutionOperation
from simllm.goal import GoalGraphEdge, GoalMessage, GoalTrace


def _graph() -> ExecutionGraph:
    first = ExecutionOperation(
        "first",
        0,
        "network",
        CollectiveWork("all-to-allv", (0, 1), 8, "pairwise"),
    )
    second = ExecutionOperation(
        "second",
        0,
        "network",
        CollectiveWork("all-to-allv", (0, 1), 8, "pairwise"),
    )
    return ExecutionGraph("cross-check", 0, 0, (first, second), ("second",))


def _direct_trace() -> GoalTrace:
    trace = GoalTrace(2)
    first_send = trace.rank(0).send(8, 1, tag=10, operation_id="first")
    first_recv = trace.rank(1).recv(8, 0, tag=10, operation_id="first")
    trace.record_message(
        GoalMessage("first", 0, 1, 8, 10, first_send, first_recv)
    )
    second_send = trace.rank(0).send(8, 1, tag=20, operation_id="second")
    second_recv = trace.rank(1).recv(8, 0, tag=20, operation_id="second")
    trace.record_message(
        GoalMessage("second", 0, 1, 8, 20, second_send, second_recv)
    )
    trace.rank(0).requires(second_send, first_send)
    trace.rank(1).requires(second_recv, first_recv)
    return trace


def _edge() -> GoalGraphEdge:
    return GoalGraphEdge(
        "first",
        "second",
        "whole-operation",
        "logical-queue-fifo",
    )


def _plan():
    graph = _graph()
    trace = _direct_trace()
    return plan_dependency_cross_check(
        graph,
        trace,
        (_edge(),),
        trace.messages,
    )


def _complete(plan, **overrides):
    arguments = {
        "authority_rows": ((10, 0, 100), (20, 100, 180)),
        "cross_check_rows": ((10, 0, 100), (20, 90, 170)),
        "authority_completion_ps": 200,
        "cross_check_completion_ps": 180,
        "tolerance_ps": 0,
        "authority_artifact_names": ("first.goal", "second.goal"),
        "authority_artifact_sha256": ("a" * 64, "b" * 64),
        "authority_artifact_bytes": (100, 200),
        "cross_check_artifact_name": "direct.goal",
        "cross_check_artifact_sha256": "c" * 64,
        "cross_check_artifact_bytes": 250,
        "authority_quiescent": True,
        "cross_check_quiescent": True,
        "authority_flow_count": 2,
        "cross_check_flow_count": 2,
    }
    arguments.update(overrides)
    return complete_dependency_cross_check(plan, **arguments)


def test_plan_reports_missing_cross_rank_requires_as_a_finding():
    plan = _plan()

    assert plan.execution_id == "cross-check"
    assert plan.step_index == 0
    assert plan.operation_ids == ("first", "second")
    assert plan.expected_message_count == 2
    assert len(plan.ordering_comparisons) == 1
    comparison = plan.ordering_comparisons[0]
    assert comparison.predecessor_ranks == (0, 1)
    assert comparison.target_ranks == (0, 1)
    assert comparison.missing_predecessor_ranks_by_target == (
        (0, (1,)),
        (1, (0,)),
    )
    assert comparison.missing_terminal_count == 2
    assert comparison.disagreement
    assert plan.boundary_tags == (("first", "second", (10,), (20,)),)


def test_complete_reports_frontier_and_completion_disagreements():
    report = _complete(_plan())

    assert report.authority_mechanism == "execution-graph-projection"
    assert report.cross_check_mechanism == "atlahs-independent-goal"
    assert report.execution_id == "cross-check"
    assert report.step_index == 0
    assert report.ordering_disagreement_count == 1
    assert report.phase_frontier_disagreement_count == 1
    frontier = report.phase_frontier_comparisons[0]
    assert frontier.authority_predecessor_completion_ps == 100
    assert frontier.authority_target_start_ps == 100
    assert frontier.authority_gap_ps == 0
    assert frontier.cross_check_predecessor_completion_ps == 100
    assert frontier.cross_check_target_start_ps == 90
    assert frontier.cross_check_gap_ps == -10
    assert frontier.signed_gap_difference_ps == -10
    assert frontier.evaluated
    assert frontier.disagreement
    assert report.signed_completion_difference_ps == -20
    assert report.completion_tolerance_ps == 0
    assert report.completion_disagreement
    assert report.has_disagreement


def test_completion_tolerance_is_inclusive_and_does_not_hide_other_findings():
    report = _complete(_plan(), tolerance_ps=20)

    assert not report.completion_disagreement
    assert report.phase_frontier_disagreement_count == 1
    assert report.has_disagreement


def test_plan_rejects_semantic_operation_inventory_mismatch():
    altered = GoalTrace(2)
    first_send = altered.rank(0).send(8, 1, tag=10, operation_id="wrong")
    first_recv = altered.rank(1).recv(8, 0, tag=10, operation_id="wrong")
    altered.record_message(
        GoalMessage("wrong", 0, 1, 8, 10, first_send, first_recv)
    )

    with pytest.raises(ValueError, match="semantic operation inventory differs"):
        plan_dependency_cross_check(_graph(), altered, (_edge(),), altered.messages)


def test_plan_rejects_physical_message_inventory_mismatch():
    trace = _direct_trace()
    perturbed = replace(trace.messages[0], payload_bytes=9)

    with pytest.raises(ValueError, match="physical message inventory differs"):
        plan_dependency_cross_check(
            _graph(),
            trace,
            (_edge(),),
            (perturbed, trace.messages[1]),
        )


def test_complete_rejects_incomplete_boundary_rows():
    with pytest.raises(ValueError, match="flow count does not match"):
        _complete(_plan(), cross_check_rows=((10, 0, 100),))


def test_complete_rejects_nonquiescent_evidence():
    with pytest.raises(ValueError, match="must reach quiescence"):
        _complete(_plan(), cross_check_quiescent=False)


def test_frontier_without_message_tags_is_explicitly_unevaluated():
    graph = ExecutionGraph(
        "compute",
        0,
        0,
        (
            ExecutionOperation("first", 0, "queue", _graph().operations[0].work),
            ExecutionOperation("second", 0, "queue", _graph().operations[1].work),
        ),
    )
    trace = GoalTrace(2)
    first0 = trace.rank(0).calc(1, operation_id="first")
    first1 = trace.rank(1).calc(1, operation_id="first")
    second0 = trace.rank(0).calc(1, operation_id="second")
    second1 = trace.rank(1).calc(1, operation_id="second")
    trace.rank(0).requires(second0, first0)
    trace.rank(1).requires(second1, first1)
    plan = plan_dependency_cross_check(graph, trace, (_edge(),), ())

    report = _complete(
        plan,
        authority_rows=(),
        cross_check_rows=(),
        authority_flow_count=0,
        cross_check_flow_count=0,
    )

    assert not report.phase_frontier_comparisons[0].evaluated
    assert report.phase_frontier_comparisons[0].authority_gap_ps is None
    assert report.phase_frontier_disagreement_count == 0
