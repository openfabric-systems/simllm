"""Session graph/metric joins with independently scheduled fake completions."""

from dataclasses import replace
from types import MappingProxyType
from typing import ClassVar

import pytest
from test_goal_session import FakeSession

import simllm.backends.step_sink as sink_module
from simllm.backends.flow_session import FlowSessionConfig, FlowSessionDrain
from simllm.backends.goal_session import GoalSessionExecutor, GoalTraceSnapshot
from simllm.backends.session_projection import build_session_evidence
from simllm.backends.step_attribution import HtsimRequestMetricReducer
from simllm.backends.step_sink import HtsimPersistentStepSink, HtsimStepSink, HtsimStepSinkConfig
from simllm.compute import ComputeProvider, DurationEstimate, HostInitiationModel, ModelDims
from simllm.core import (
    CollectiveWork,
    ComputeWork,
    EventPhase,
    ExecutionGraph,
    ExecutionOperation,
    RequestPhase,
    ScheduledRequest,
    StepRecord,
    StepResult,
)
from simllm.core.execution_io import execution_result_to_json
from simllm.core.step_io import step_result_to_json
from simllm.traffic import project_execution_graph_goal


class ObservedSession(FakeSession):
    created: ClassVar[list] = []
    fail_close = False

    def __init__(self, config=None, command=(), *, session_id="test", time_origin_ps=0):
        super().__init__()
        self.config = config
        self.session_id = session_id
        self.transcript = (("request", b"frozen-test-frame"),)
        self.closed = False
        self.created.append(self)

    @property
    def completion_rows(self):
        return tuple(MappingProxyType(self.rows[seq]) for seq in sorted(self.rows))

    @property
    def events(self):
        rows = []
        for row in self.completion_rows:
            for kind in ("accepted", "queued", "started", "completed"):
                at = row["completion_time_ps"] if kind == "completed" else row["start_time_ps"]
                rows.append(MappingProxyType({**row, "kind": kind, "timestamp_ps": at, "sq_id": 1}))
        return tuple(sorted(rows, key=lambda row: row["timestamp_ps"]))

    def close(self):
        if self.fail_close:
            self.abort()
            raise RuntimeError("missing quiescence proof")
        self.closed = True
        return FlowSessionDrain(self.time + 20_000, self.completion_rows,
                                MappingProxyType({"native_posts": len(self.rows)}), (1, 1))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if exc_type:
            self.abort()
        else:
            self.close()


def graph_fixture():
    return ExecutionGraph("checked", 0, 0, (
        ExecutionOperation("compute-0", 0, "cpu:0", ComputeWork("gemm", nominal_duration_ps=3000)),
        ExecutionOperation("compute-1", 1, "cpu:1", ComputeWork("gemm", nominal_duration_ps=3000)),
        ExecutionOperation("ring", 0, "nccl", CollectiveWork("all-reduce", (0, 1), 8192, "ring"),
                           participant_local_depends_on=("compute-0", "compute-1")),
    ), ("ring",))


def projected(*, trailing_compute=False):
    graph = graph_fixture()
    if trailing_compute:
        graph = replace(graph, operations=graph.operations + (
            ExecutionOperation("trailing", 0, "cpu:0", ComputeWork("gemm", nominal_duration_ps=5000),
                               depends_on=("ring",)),), completion_operation_ids=("trailing",))
    snapshots = tuple(GoalTraceSnapshot.from_trace(a.trace)
                      for a in project_execution_graph_goal(graph, num_goal_ranks=4).artifacts)
    session = ObservedSession()
    executor = GoalSessionExecutor(session, graph.execution_id)
    artifacts = []
    release, predecessors = 0, ()
    for index, snapshot in enumerate(snapshots):
        artifact = executor.run(snapshot, f"artifact-{index}", release,
                                predecessor_sequences=predecessors)
        artifacts.append(artifact)
        release = artifact.completed_at_ps
        predecessors = artifact.causal_sequences
    return graph, snapshots, tuple(artifacts), session, session.close()


def test_projection_keeps_original_graph_identity_and_native_quiescence():
    evidence = build_session_evidence(*projected())
    result = evidence.execution_result
    assert result.execution_id == "checked" and result.completed_at_ps == 23_000
    assert result.quiesced_at_ps == evidence.native_drain.quiesced_at_ps == 43_000
    logical = [event for event in result.events if event.phase is EventPhase.COMPLETED
               and event.subject_object_id is None]
    assert {event.operation_id: event.timestamp_ps for event in logical} == {
        "compute-0": 3000, "compute-1": 3000, "ring": 23_000}
    assert len([event for event in result.events if event.subject_object_id is not None]) == 16
    assert execution_result_to_json(result)["quiesced_at_ps"] == 43_000
    evidence.validate_result(StepResult(0, 23_000, 23_000))
    with pytest.raises(ValueError, match="StepResult"):
        evidence.validate_result(StepResult(0, 23_001, 23_001))


def test_graph_quiescence_accounts_for_local_work_without_rewriting_native_time():
    graph, snapshots, artifacts, session, drain = projected(trailing_compute=True)
    drain = replace(drain, quiesced_at_ps=23_000)
    evidence = build_session_evidence(graph, snapshots, artifacts, session, drain)
    assert evidence.native_drain.quiesced_at_ps == 23_000
    assert evidence.execution_result.completed_at_ps == 28_000
    assert evidence.execution_result.quiesced_at_ps == 28_000


@pytest.mark.parametrize("mutation", ("artifact", "action", "owner", "time", "row", "sequence",
                                      "drain", "drain-time", "edge"))
def test_projection_rejects_loss_mutation_or_causal_disagreement(mutation):
    graph, snapshots, artifacts, session, drain = projected()
    if mutation == "artifact":
        artifacts = artifacts[:-1]
    elif mutation in {"action", "owner", "time"}:
        action = artifacts[0].actions[0]
        if mutation == "action":
            actions = artifacts[0].actions[:-1]
        else:
            changed = replace(action, **({"operation_id": "unknown"} if mutation == "owner"
                                         else {"eligible_at_ps": 4000}))
            actions = (changed,) + artifacts[0].actions[1:]
        artifacts = (replace(artifacts[0], actions=actions), artifacts[1])
    elif mutation == "row":
        rows = list(artifacts[1].completion_rows)
        rows[0] = MappingProxyType({**rows[0], "tag": rows[0]["tag"] + 100})
        artifacts = (artifacts[0], replace(artifacts[1], completion_rows=tuple(rows)))
    elif mutation == "sequence":
        artifacts = (artifacts[0], replace(artifacts[1], flow_sequences=MappingProxyType({})))
    elif mutation == "drain":
        drain = replace(drain, completion_rows=drain.completion_rows[:-1])
    elif mutation == "drain-time":
        drain = replace(drain, quiesced_at_ps=22_999)
    else:
        graph = replace(graph, operations=(replace(graph.operations[0], not_before_ps=4000),)
                        + graph.operations[1:])
    with pytest.raises(ValueError):
        build_session_evidence(graph, snapshots, artifacts, session, drain)


class FixedProvider(ComputeProvider):
    def estimate(self, kernel, gpu):
        return DurationEstimate(duration_ps=32_000_000, bound="declared")


DIMS = ModelDims(num_layers=2, hidden_size=256, intermediate_size=1024,
                 num_heads=4, num_kv_heads=4, head_size=64, vocab_size=1024, dtype_bytes=2)


def config(tmp_path, **kwargs):
    values = {"profile": "rnic-nn", "tp_ranks": (0, 4), "dims": DIMS, "workdir": tmp_path,
              "provider": FixedProvider(), "host_model": HostInitiationModel.ideal(), "num_goal_ranks": 8,
              "flow_session": FlowSessionConfig("rnic-nn", 8, 400_000_000_000, "0" * 64, 9001)}
    return HtsimStepSinkConfig(**(values | kwargs))


def record(index=0, release=0):
    phase = RequestPhase.PREFILL if index == 0 else RequestPhase.DECODE
    return StepRecord(index, release, [ScheduledRequest(f"r{i}", phase, num_new_tokens=1, context_length=index + 1)
                                      for i in range(16)])


def fake_child(monkeypatch, tmp_path):
    ObservedSession.created.clear()
    monkeypatch.setattr(sink_module, "FlowSession", ObservedSession)
    monkeypatch.setattr(sink_module, "find_htsim_rnic", lambda: tmp_path / "test-child")


def test_sink_retains_one_session_and_returns_request_metrics(monkeypatch, tmp_path):
    fake_child(monkeypatch, tmp_path)
    sink = HtsimStepSink(config(tmp_path), request_metric_reducer=HtsimRequestMetricReducer(
        {f"r{i}": 0 for i in range(16)}))
    first = sink(record())
    second = sink(record(1, first.completed_at_ps))
    assert first.step_latency_ps == second.step_latency_ps == 32_080_000
    assert len(first.request_metrics) == 16
    assert first.request_metrics[0].ttft_ps == 32_080_000
    assert second.request_metrics[0].tpot_ps == 32_080_000
    assert "request_metrics" in step_result_to_json(first)
    assert len(ObservedSession.created) == 2
    assert all(child.closed and len(child.injections) == 16 for child in ObservedSession.created)
    assert all(outcome.backend_runs == 1 for outcome in sink.locality_outcomes)
    assert len(sink.session_evidence) == 2
    for evidence, result in zip(sink.session_evidence, (first, second), strict=True):
        evidence.validate_result(result)
        assert len(evidence.graph.operations) == 8 and len(evidence.artifacts) == 6


def test_prepared_sessions_publish_proofs_and_metrics_only_when_consumed(monkeypatch, tmp_path):
    fake_child(monkeypatch, tmp_path)
    reducer = HtsimRequestMetricReducer({f"r{i}": 0 for i in range(16)})
    records = [record(), record(1, 32_080_000)]
    with HtsimPersistentStepSink(config(tmp_path), max_workers=2,
                                 request_metric_reducer=reducer) as sink:
        sink.prepare(records)
        assert len(ObservedSession.created) == 2
        assert all(child.closed for child in ObservedSession.created)
        assert sink.session_evidence == sink.outcomes == []
        assert reducer.latest_request_metrics == ()
        with pytest.raises(ValueError, match="next prepared step"):
            sink(records[1])
        assert reducer.latest_request_metrics == ()
        first, second = [sink(item) for item in records]
        assert len(sink.session_evidence) == 2
        assert first.request_metrics[0].ttft_ps == second.request_metrics[0].tpot_ps == 32_080_000
        assert sink.prepared_steps_remaining == 0


@pytest.mark.parametrize("field,value", (("emit_packet_breakdown", True), ("emit_bottleneck_report", True),
                                        ("dependency_cross_check", "atlahs-goal"), ("topology", "custom.topo"),
                                        ("unsafe_disable_child_lifetime_binding", True)))
def test_unsupported_session_compositions_fail_before_child(field, value, monkeypatch, tmp_path):
    fake_child(monkeypatch, tmp_path)
    with pytest.raises(ValueError):
        HtsimStepSink(config(tmp_path, **{field: value}))
    assert not ObservedSession.created


def test_failed_session_publishes_no_metrics_or_outcomes(monkeypatch, tmp_path):
    fake_child(monkeypatch, tmp_path)
    monkeypatch.setattr(ObservedSession, "fail_close", True)
    reducer = HtsimRequestMetricReducer({f"r{i}": 0 for i in range(16)})
    sink = HtsimStepSink(config(tmp_path), request_metric_reducer=reducer)
    with pytest.raises(RuntimeError, match="quiescence"):
        sink(record())
    assert not sink.outcomes and not sink.session_evidence and not sink.locality_outcomes
    assert not reducer.latest_request_metrics
    assert ObservedSession.created[0].aborted


def test_changed_planned_goal_fails_before_native_open(monkeypatch, tmp_path):
    fake_child(monkeypatch, tmp_path)
    sink = HtsimStepSink(config(tmp_path))
    plan = sink._plan_step(record())
    next(a.goal_path for a in plan.artifacts if a.goal_path is not None).write_text("changed")
    with pytest.raises(ValueError, match="changed after"):
        sink._execute_plan(plan)
    assert not ObservedSession.created
