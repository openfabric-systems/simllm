from dataclasses import FrozenInstanceError, replace
from hashlib import sha256
from types import SimpleNamespace

import pytest

from simllm.backends.goal_session import (
    GoalSessionError,
    GoalSessionExecutor,
    GoalTraceSnapshot,
)
from simllm.goal import GoalMessage, GoalTrace
from simllm.goal.emitter import (
    GoalDependency,
    GoalDependencyKind,
    GoalDependencyProvenance,
    GoalGraphEdge,
    GoalOperation,
)
from simllm.traffic.patterns import ring_allreduce


class FakeSession:
    def __init__(self, durations=10_000, *, simultaneous=False, corrupt=None):
        self.durations = durations
        self.simultaneous = simultaneous
        self.corrupt = corrupt
        self.injections = {}
        self.rows = {}
        self.calls = []
        self.boundary_time_ps = None
        self.floor = None
        self.time = 0
        self.aborted = False
        self.tail_events = [1_000_000]

    @property
    def pending_sequences(self):
        return tuple(seq for seq in self.injections if seq not in self.rows)

    @property
    def completed_sequences(self):
        return tuple(sorted(self.rows))

    @property
    def completion_rows(self):
        return tuple(self.rows.values())

    @property
    def events(self):
        return ()

    def inject(self, **fields):
        assert not self.aborted
        predecessors = fields["predecessor_sequences"]
        assert tuple(sorted(set(predecessors))) == predecessors
        assert set(predecessors) <= self.rows.keys()
        eligible = fields["eligible_at_ps"]
        if self.floor is not None and eligible <= self.floor:
            assert eligible == self.boundary_time_ps and predecessors
        seq = len(self.injections) + 1
        self.injections[seq] = dict(fields)
        self.calls.append(("inject", seq, eligible, predecessors))
        return seq

    def await_completion(self, sequences=None, *, through_ps=None):
        assert not self.aborted
        targets = self.pending_sequences if sequences is None else tuple(sequences)
        assert targets and set(targets) <= set(self.pending_sequences)
        assert through_ps is None or self.floor is None or through_ps >= self.floor
        self.calls.append(("await", targets, through_ps))
        self.boundary_time_ps = None
        times = {}
        for seq in self.pending_sequences:
            fields = self.injections[seq]
            duration = self.durations if isinstance(self.durations, int) else self.durations[seq - 1]
            times[seq] = fields["eligible_at_ps"] + duration
        next_time = min(times[seq] for seq in targets)
        if through_ps is not None and next_time > through_ps:
            self.floor = through_ps
            return SimpleNamespace(reason="horizon", event_time_ps=self.time,
                                   completion_rows=())
        self.time = self.floor = self.boundary_time_ps = next_time
        complete = [seq for seq, time in times.items() if time == next_time]
        if not self.simultaneous:
            complete = complete[:1]
        rows = []
        for seq in complete:
            fields = self.injections[seq]
            row = {name: value for name, value in fields.items()
                   if name not in {"eligible_at_ps", "predecessor_sequences"}}
            row.update(sequence=seq, native_flow_id=seq - 1, wqe_id=seq,
                       start_time_ps=fields["eligible_at_ps"], completion_time_ps=next_time,
                       fct_ps=next_time - fields["eligible_at_ps"], completion_status="success")
            if self.corrupt:
                row.update(self.corrupt)
            self.rows[seq] = row
            rows.append(row)
        return SimpleNamespace(reason="completion", event_time_ps=next_time,
                               completion_rows=tuple(rows))

    def abort(self):
        self.aborted = True


def add_message(trace, source=0, destination=1, *, tag=7, size=4096, owner="network"):
    send = trace.rank(source).send(size, destination, tag, operation_id=owner)
    receive = trace.rank(destination).recv(size, source, tag, operation_id=owner)
    trace.record_message(GoalMessage(owner, source, destination, size, tag, send, receive))
    return send, receive


def execute(trace, session=None, **kwargs):
    session = FakeSession() if session is None else session
    return GoalSessionExecutor(session, "execution").run(
        GoalTraceSnapshot.from_trace(trace), "artifact", 0, **kwargs)


def single_trace():
    trace = GoalTrace(2)
    add_message(trace)
    return trace


def test_snapshot_is_independent_of_mutable_trace_and_protects_rendered_identity():
    trace = single_trace()
    snapshot = GoalTraceSnapshot.from_trace(trace)
    trace.rank(0).calc(10, operation_id="later")
    assert len(snapshot.operations) == 2
    assert snapshot.sha256 == sha256(snapshot.rendered_bytes).hexdigest()
    assert b"later" not in snapshot.rendered_bytes
    with pytest.raises(FrozenInstanceError):
        snapshot.num_ranks = 3
    with pytest.raises(ValueError, match="bytes disagree"):
        replace(snapshot, rendered_bytes=snapshot.rendered_bytes + b"\n")
    with pytest.raises(ValueError, match="SHA-256"):
        replace(snapshot, sha256="0" * 64)


def test_physical_and_logical_rows_are_absolute_and_immutable():
    session = FakeSession()
    result = GoalSessionExecutor(session, "step:1").run(
        GoalTraceSnapshot.from_trace(single_trace()), "a/b", 50_000)
    assert result.completed_at_ps == 60_000
    assert result.accepted_sequences == result.causal_sequences == (1,)
    assert result.flow_sequences == {(0, "r0op0"): 1}
    row = result.completion_rows[0]
    assert row["flow_id"] == "a%2Fb/0/r0op0"
    assert row["start_time_ps"] == 50_000
    assert row["completion_time_ps"] == 60_000
    assert row["fct_ps"] == 10_000
    assert result.operation_frontiers == {"network": {0: 60_000, 1: 60_000}}
    session.rows[1]["completion_time_ps"] = 900_000
    assert row["completion_time_ps"] == 60_000
    with pytest.raises(TypeError):
        row["completion_time_ps"] = 0
    with pytest.raises(TypeError):
        result.flow_sequences[(0, "r0op0")] = 9
    with pytest.raises(TypeError):
        result.operation_frontiers["network"][0] = 9
    assert session.tail_events == [1_000_000]
    assert not any(call[0] in {"advance", "drain", "close"} for call in session.calls)


def test_ring_rounds_follow_declared_receive_completion():
    trace = GoalTrace(2)
    ring_allreduce(trace, ranks=[0, 1], size_bytes=8192, base_tag=10, operation_id="ring")
    session = FakeSession()
    result = execute(trace, session)
    assert [fields["eligible_at_ps"] for fields in session.injections.values()] == [0, 0, 10_000, 10_000]
    assert result.completed_at_ps == 20_000
    assert result.accepted_sequences == (1, 2, 3, 4)
    assert session.injections[3]["predecessor_sequences"] == (1,)
    assert session.injections[4]["predecessor_sequences"] == (2,)
    assert result.operation_frontiers == {"ring": {0: 20_000, 1: 20_000}}


def test_same_callback_rows_release_all_successors_before_next_await():
    trace = GoalTrace(2)
    ring_allreduce(trace, ranks=[0, 1], size_bytes=8192, base_tag=10, operation_id="ring")
    session = FakeSession(simultaneous=True)
    result = execute(trace, session)
    assert result.completed_at_ps == 20_000
    assert [call[0] for call in session.calls] == [
        "inject", "inject", "await", "inject", "inject", "await"]


@pytest.mark.parametrize("local_ns", [5, 10, 20])
def test_local_completion_is_processed_before_equal_native_callback(local_ns):
    trace = GoalTrace(4)
    add_message(trace)
    calc = trace.rank(2).calc(local_ns, operation_id="compute")
    send, _ = add_message(trace, 2, 3, tag=8, owner="second")
    trace.rank(2).requires(send, calc)
    session = FakeSession()
    result = execute(trace, session)
    assert session.injections[2]["eligible_at_ps"] == local_ns * 1000
    assert session.injections[2]["predecessor_sequences"] == ()
    assert session.calls[1] == ("await", (1,), local_ns * 1000 - 1)
    if local_ns <= 10:
        assert session.calls[2] == ("inject", 2, local_ns * 1000, ())
    assert result.completed_at_ps == max(10_000, local_ns * 1000 + 10_000)


def test_late_receive_changes_logical_completion_without_rewriting_physical_row():
    trace = GoalTrace(2)
    calc = trace.rank(1).calc(30, operation_id="compute")
    send, receive = add_message(trace)
    trace.rank(1).requires(receive, calc)
    result = execute(trace)
    assert result.completion_rows[0]["completion_time_ps"] == 10_000
    assert result.completion_rows[0]["fct_ps"] == 10_000
    assert result.action_by_key[(0, send)].completed_at_ps == 30_000
    assert result.action_by_key[(1, receive)].started_at_ps == 30_000
    assert result.action_by_key[(1, receive)].completed_at_ps == 30_000
    assert result.completed_at_ps == 30_000


def test_late_receive_dependency_carries_both_physical_ancestors():
    trace = GoalTrace(3)
    send, receive = add_message(trace, 0, 1)
    _, ready = add_message(trace, 2, 1, tag=8, owner="trigger")
    successor, _ = add_message(trace, 0, 2, tag=9, owner="successor")
    trace.rank(1).requires(receive, ready)
    trace.rank(0).requires(successor, send)
    session = FakeSession(durations=[10_000, 20_000, 10_000])
    result = execute(trace, session)
    assert session.injections[3]["eligible_at_ps"] == 20_000
    assert session.injections[3]["predecessor_sequences"] == (1, 2)
    assert result.action_by_key[(0, send)].completed_at_ps == 20_000
    assert result.completion_rows[0]["completion_time_ps"] == 10_000
    assert result.completed_at_ps == 30_000


def test_pure_compute_serializes_rank_cpu_and_excludes_padding_from_semantic_frontiers():
    trace = GoalTrace(2)
    trace.rank(0).calc(4, operation_id="first")
    trace.rank(0).calc(6, operation_id="second")
    trace.rank(1).calc(0)
    session = FakeSession()
    result = execute(trace, session)
    assert [(a.started_at_ps, a.completed_at_ps) for a in result.actions] == [
        (0, 4000), (4000, 10_000), (0, 1000)]
    assert result.operation_frontiers == {"first": {0: 4000}, "second": {0: 10_000}}
    assert result.accepted_sequences == ()
    assert result.completion_rows == ()
    assert session.calls == []


def test_compute_zero_preserves_one_nanosecond_and_requires_direction():
    trace = GoalTrace(1)
    later = trace.rank(0).calc(2, operation_id="later")
    first = trace.rank(0).calc(0, operation_id="first")
    trace.rank(0).requires(later, first)
    result = execute(trace)
    assert result.action_by_key[(0, first)].completed_at_ps == 1000
    assert result.action_by_key[(0, later)].started_at_ps == 1000
    assert result.completed_at_ps == 3000


def test_compute_between_artifacts_retains_sequences_and_tail_state():
    session = FakeSession()
    executor = GoalSessionExecutor(session, "step")
    snapshot = GoalTraceSnapshot.from_trace(single_trace())
    first = executor.run(snapshot, "first", 0)
    compute = GoalTrace(2)
    compute.rank(0).calc(3, operation_id="compute")
    compute.rank(1).calc(0)
    middle = executor.run(GoalTraceSnapshot.from_trace(compute), "middle", 10_000,
                          predecessor_sequences=first.causal_sequences)
    second = executor.run(snapshot, "second", middle.completed_at_ps,
                          predecessor_sequences=middle.causal_sequences)
    assert second.accepted_sequences == (2,)
    assert [row["sequence"] for row in second.completion_rows] == [2]
    assert second.causal_sequences == (1, 2)
    assert session.injections[2]["eligible_at_ps"] == 13_000
    assert session.injections[2]["predecessor_sequences"] == (1,)
    assert session.tail_events == [1_000_000]


def test_exact_artifact_successor_uses_real_predecessor_and_live_boundary():
    session = FakeSession()
    executor = GoalSessionExecutor(session, "step")
    snapshot = GoalTraceSnapshot.from_trace(single_trace())
    first = executor.run(snapshot, "first", 0)
    second = executor.run(snapshot, "second", first.completed_at_ps,
                          predecessor_sequences=first.causal_sequences)
    assert session.injections[2]["predecessor_sequences"] == (1,)
    assert second.completed_at_ps == 20_000


@pytest.mark.parametrize("text", [
    "calc -1", "calc 1 cpu 1", "calc 1 nic 0", "calc 1 cpu 0 junk", "calc 1\ncalc 2",
    "send 1b to 1 tag 0 nic 1", "send 1b to 1 tag 0 cpu 1", "send 0b to 1 tag 0",
    "send 1b from 1 tag 0", "send 1b to 0 tag 0", "recv 1b to 1 tag 0",
    "recv 1b from 2 tag 0", "send 1b to 1 tag 4294967296", "send 1b to 1", "barrier",
    f"calc {((1 << 64) - 1) // 1000 + 1}",
])
def test_unsupported_grammar_is_rejected_in_snapshot_before_session(text):
    trace = single_trace()
    trace.rank(0)._ops[0] = GoalOperation(0, "r0op0", text, "network")
    with pytest.raises(ValueError):
        GoalTraceSnapshot.from_trace(trace)


def test_exact_zero_selectors_preserve_supported_semantics():
    trace = single_trace()
    trace.rank(0)._ops[0] = replace(trace.rank(0)._ops[0],
                                  text="send 4096b to 1 tag 7 cpu 0 nic 0")
    trace.rank(1)._ops[0] = replace(trace.rank(1)._ops[0],
                                  text="recv 4096b from 0 tag 7 nic 0")
    calc = trace.rank(0).calc(0, cpu=0, operation_id="compute")
    result = execute(trace)
    assert result.action_by_key[(0, calc)].completed_at_ps == 1000
    assert result.completed_at_ps == 10_000


@pytest.mark.parametrize("change", ["duplicate_action", "missing_message", "duplicate_message",
                                    "bad_owner", "bad_payload", "bad_tag", "bad_label",
                                    "unowned_send", "unowned_calc", "bool_rank", "missing_receive"])
def test_complete_inventory_and_owner_validation(change):
    trace = single_trace()
    if change == "duplicate_action":
        trace.rank(0)._ops.append(trace.rank(0)._ops[0])
    elif change == "missing_message":
        trace._messages.clear()
    elif change == "duplicate_message":
        trace._messages.append(trace.messages[0])
    elif change in {"bad_owner", "bad_payload", "bad_tag"}:
        field, value = {"bad_owner": ("operation_id", "other"), "bad_payload": ("payload_bytes", 1),
                        "bad_tag": ("tag", 3)}[change]
        trace._messages[0] = replace(trace.messages[0], **{field: value})
    elif change == "bad_label":
        trace.rank(0)._ops[0] = replace(trace.rank(0)._ops[0], label="bad:label")
    elif change == "unowned_send":
        trace.rank(0)._ops[0] = replace(trace.rank(0)._ops[0], operation_id=None)
    elif change == "unowned_calc":
        trace.rank(0).calc(3)
    elif change == "bool_rank":
        trace.rank(0)._ops[0] = replace(trace.rank(0)._ops[0], rank=False)
    else:
        trace.rank(1)._ops.clear()
    with pytest.raises(ValueError):
        GoalTraceSnapshot.from_trace(trace)


@pytest.mark.parametrize("relation", ["irequires", "cycle", "duplicate", "missing", "owner"])
def test_dependency_preflight_rejects_unsupported_or_inconsistent_edges(relation):
    trace = GoalTrace(1)
    first = trace.rank(0).calc(1, operation_id="first")
    second = trace.rank(0).calc(1, operation_id="second")
    if relation == "irequires":
        trace.rank(0).irequires(second, first)
    elif relation == "cycle":
        trace.rank(0).requires(first, second)
        trace.rank(0).requires(second, first)
    elif relation == "duplicate":
        trace.rank(0).requires(second, first)
        trace.rank(0).requires(second, first)
    elif relation == "missing":
        trace.rank(0)._deps.append(GoalDependency(0, second, "absent", "requires"))
    else:
        provenance = GoalDependencyProvenance(GoalDependencyKind.EXECUTION_GRAPH, "second",
                                            (GoalGraphEdge("wrong", "second", "whole-operation",
                                                           "explicit"),))
        trace.rank(0).requires(second, first, provenance=provenance)
    with pytest.raises(ValueError):
        GoalTraceSnapshot.from_trace(trace)


def test_stalled_receive_dependency_is_terminal_after_physical_callback():
    trace = GoalTrace(2)
    send, receive = add_message(trace)
    other, other_receive = add_message(trace, 1, 0, tag=8, owner="other")
    trace.rank(0).requires(other_receive, send)
    trace.rank(1).requires(receive, other)
    session = FakeSession()
    executor = GoalSessionExecutor(session, "step")
    with pytest.raises(GoalSessionError, match="stalled"):
        executor.run(GoalTraceSnapshot.from_trace(trace), "bad", 0)
    assert session.completed_sequences == (1, 2)
    assert session.aborted
    with pytest.raises(GoalSessionError, match="terminal"):
        executor.run(GoalTraceSnapshot.from_trace(single_trace()), "later", 20_000)


@pytest.mark.parametrize("corrupt", [
    {"completion_status": "transport_error"}, {"flow_id": "wrong"}, {"payload_bytes": 3},
    {"operation_id": "wrong"}, {"fct_ps": 5}, {"start_time_ps": 1},
])
def test_native_row_disagreement_aborts_before_publishing_result(corrupt):
    session = FakeSession(corrupt=corrupt)
    with pytest.raises(GoalSessionError):
        execute(single_trace(), session)
    assert session.aborted


@pytest.mark.parametrize("predecessors", [(0,), (True,), (1,), (2, 1), (1, 1)])
def test_invalid_prior_artifact_sequences_reject_before_injection(predecessors):
    session = FakeSession()
    with pytest.raises((ValueError, GoalSessionError)):
        execute(single_trace(), session, predecessor_sequences=predecessors)
    assert session.calls == []
    assert session.aborted


def test_repeated_message_envelope_rejects_ambiguous_matching_before_execution():
    trace = single_trace()
    add_message(trace)
    with pytest.raises(ValueError, match="ambiguous repeated"):
        GoalTraceSnapshot.from_trace(trace)


def test_unowned_padding_cannot_hide_an_active_rank_action():
    trace = single_trace()
    trace.rank(0).calc(0)
    with pytest.raises(ValueError, match="inactive rank"):
        GoalTraceSnapshot.from_trace(trace)


def test_exact_boundary_without_causal_ancestry_is_rejected_before_injection():
    session = FakeSession()
    executor = GoalSessionExecutor(session, "step")
    snapshot = GoalTraceSnapshot.from_trace(single_trace())
    first = executor.run(snapshot, "first", 0)
    with pytest.raises(GoalSessionError, match="no causal completed predecessor"):
        executor.run(snapshot, "second", first.completed_at_ps)
    assert len(session.injections) == 1
    assert session.aborted


@pytest.mark.parametrize("artifact_id,release", [("first", 10_000), ("second", 9999)])
def test_artifact_identity_and_release_cursor_are_checked(artifact_id, release):
    session = FakeSession()
    executor = GoalSessionExecutor(session, "step")
    snapshot = GoalTraceSnapshot.from_trace(single_trace())
    first = executor.run(snapshot, "first", 0)
    with pytest.raises(GoalSessionError, match="duplicate artifact or backward"):
        executor.run(snapshot, artifact_id, release, first.causal_sequences)
    assert len(session.injections) == 1


def test_original_graph_projection_preserves_owned_compute_and_collective_frontiers():
    from simllm.core import CollectiveWork, ComputeWork, ExecutionGraph, ExecutionOperation
    from simllm.traffic import project_execution_graph_goal

    graph = ExecutionGraph("checked", 0, 0, (
        ExecutionOperation("compute-0", 0, "cpu:0", ComputeWork("gemm", nominal_duration_ps=3000)),
        ExecutionOperation("compute-1", 1, "cpu:1", ComputeWork("gemm", nominal_duration_ps=3000)),
        ExecutionOperation("ring", 0, "nccl", CollectiveWork("all-reduce", (0, 1), 8192, "ring"),
                           participant_local_depends_on=("compute-0", "compute-1")),
    ), ("ring",))
    projection = project_execution_graph_goal(graph, num_goal_ranks=4)
    assert len(projection.artifacts) == 2
    snapshots = tuple(GoalTraceSnapshot.from_trace(artifact.trace)
                      for artifact in projection.artifacts)
    executor = GoalSessionExecutor(FakeSession(), graph.execution_id)
    compute = executor.run(snapshots[0], "compute", 0)
    ring = executor.run(snapshots[1], "ring", compute.completed_at_ps,
                        predecessor_sequences=compute.causal_sequences)
    assert compute.operation_frontiers == {"compute-0": {0: 3000}, "compute-1": {1: 3000}}
    assert ring.operation_frontiers == {"ring": {0: 23_000, 1: 23_000}}
    assert ring.completed_at_ps == 23_000
    assert len(ring.completion_rows) == 4
    assert all(action.operation_id is None for action in ring.actions if action.rank >= 2)


def test_graph_dependency_provenance_matches_checked_semantic_owners():
    trace = GoalTrace(1)
    first = trace.rank(0).calc(1, operation_id="first")
    second = trace.rank(0).calc(2, operation_id="second")
    provenance = GoalDependencyProvenance(GoalDependencyKind.EXECUTION_GRAPH, "second", (
        GoalGraphEdge("first", "second", "participant-local", "logical-queue-fifo", 0),))
    trace.rank(0).requires(second, first, provenance=provenance)
    result = execute(trace)
    assert result.operation_frontiers == {"first": {0: 1000}, "second": {0: 3000}}


def test_native_row_boolean_identity_does_not_equal_integer_rank():
    session = FakeSession(corrupt={"source": False})
    with pytest.raises(GoalSessionError, match="disagrees"):
        execute(single_trace(), session)
    assert session.aborted
