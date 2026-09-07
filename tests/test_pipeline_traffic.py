from __future__ import annotations

import json
from dataclasses import replace

import pytest

from simllm.backends.step_lowerer import SerialStepLowerer, SerialStepLowererConfig
from simllm.compute import ModelDims
from simllm.core import (
    CoarseDeviceRuntime,
    CollectiveWork,
    CompletionReducer,
    ComputeWork,
    EventPhase,
    ExecutionGraph,
    ExecutionObservations,
    ExecutionOperation,
    OperationCorrelation,
    RequestPhase,
    ScheduledRequest,
    StepRecord,
    VirtualClock,
    execution_graph_from_json,
    execution_graph_to_json,
    step_record_to_json,
)
from simllm.core.execution_io import effective_dependency_edges, validate_execution_graph
from simllm.traffic import (
    PipelineStepLowerer,
    compose_pipeline_graph,
    pipeline_stage_membership,
    project_execution_graph_goal,
    render_serial_execution_graph_goal,
    step_pp_activations,
)


@pytest.fixture
def dims():
    return ModelDims(2, 4096, 16384, 32, 8, 128, 32000)


@pytest.fixture
def record():
    return StepRecord(0, 0, [
        ScheduledRequest("a", RequestPhase.PREFILL, 3, context_length=3),
        ScheduledRequest("b", RequestPhase.PREFILL, 5, context_length=5),
    ])


def stage_graph(rank, record, duration=1_000_000):
    operation = ExecutionOperation(
        "compute", rank, f"compute:{rank}",
        ComputeWork(kernel="synthetic", nominal_duration_ps=duration),
        correlation=OperationCorrelation(
            request_ids=tuple(request.request_id for request in record.scheduled),
        ),
    )
    return ExecutionGraph("step-0", record.step_index, record.virtual_time_ps,
                          (operation,), (operation.operation_id,))


def pipeline(record, dims, width):
    ranks = tuple(8 * stage for stage in range(width))
    return compose_pipeline_graph(record, dims, width, ranks,
                                  tuple(stage_graph(rank, record) for rank in ranks))


@pytest.mark.parametrize("width", [1, 2, 4, 8])
@pytest.mark.parametrize("tokens", [0, 1, 8, 128])
def test_forward_inventory(dims, width, tokens):
    record = StepRecord(4, 7000, [ScheduledRequest("a", RequestPhase.PREFILL, tokens, context_length=tokens)])
    ranks = tuple(8 * stage for stage in range(width))
    work = step_pp_activations(record, dims, width, ranks)
    assert len(work) == (width - 1 if tokens else 0)
    assert sum(size for op in work for _, _, size in op.pair_payload_bytes) == (
        (width - 1) * tokens * 4096 * 2
    )
    for boundary, op in enumerate(work):
        assert isinstance(op, CollectiveWork)
        assert op.ranks == (8 * boundary, 8 * (boundary + 1))
        assert op.pair_payload_bytes == ((8 * boundary, 8 * (boundary + 1), tokens * 8192),)
        assert op.collective == "all-to-allv"
        assert op.algorithm_hint == "pairwise"
        assert op.payload_bytes == 0


def test_last_to_first_nested_stage_membership(record, dims):
    work = step_pp_activations(record, dims, 3, ((4, 2), (12, 9), (24,)))
    assert [op.ranks for op in work] == [(2, 12), (9, 24)]
    assert work[0].request_pair_payload_bytes == (
        ("a", 2, 12, 3 * 8192), ("b", 2, 12, 5 * 8192),
    )


@pytest.mark.parametrize(("width", "stages", "error"), [
    (True, (0,), TypeError), (1.0, (0,), TypeError),
    (0, (), ValueError), (-1, (), ValueError),
    (2, (0,), ValueError), (1, (0, 8), ValueError),
    (2, (0, 0), ValueError), (2, ((0, 1), (1, 8)), ValueError),
    (2, ((0,), ()), ValueError), (1, ((True,),), ValueError),
    (1, ((1.5,),), ValueError), (1, (-1,), ValueError),
])
def test_bad_membership(width, stages, error):
    with pytest.raises(error):
        pipeline_stage_membership(width, stages)


@pytest.mark.parametrize(("field", "value"), [("hidden_size", 0), ("dtype_bytes", -1),
                                               ("dtype_bytes", True)])
def test_bad_geometry(record, dims, field, value):
    with pytest.raises(ValueError, match=field):
        step_pp_activations(record, replace(dims, **{field: value}), 2, (0, 8))


def test_repeated_request_refused(record, dims):
    with pytest.raises(ValueError, match="unique"):
        step_pp_activations(replace(record, scheduled=[record.scheduled[0]] * 2), dims, 2, (0, 8))


@pytest.mark.parametrize("width", [2, 4, 8])
def test_graph_causality_and_plan_are_live(record, dims, width):
    graph = pipeline(record, dims, width)
    validate_execution_graph(graph)
    wire = execution_graph_to_json(graph)
    assert execution_graph_from_json(json.loads(json.dumps(wire))) == graph
    assert len(graph.operations) == 2 * width - 1
    assert len(graph.collective_plans) == width - 1
    trace = render_serial_execution_graph_goal(graph, num_goal_ranks=64)
    assert len(trace.messages) == width - 1
    assert sum(message.payload_bytes for message in trace.messages) == (width - 1) * 65536
    edges = effective_dependency_edges(graph)
    for stage in range(width - 1):
        compute = graph.operations[2 * stage]
        transfer = graph.operations[2 * stage + 1]
        next_compute = graph.operations[2 * stage + 2]
        assert any(edge.predecessor_id == compute.operation_id
                   and edge.operation_id == transfer.operation_id for edge in edges)
        assert any(edge.predecessor_id == transfer.operation_id
                   and edge.operation_id == next_compute.operation_id for edge in edges)
    assert not project_execution_graph_goal(graph).boundaries


@pytest.mark.parametrize("width", [2, 4, 8])
def test_completion_events_reach_step_result_and_ttft(record, dims, width):
    graph = pipeline(record, dims, width)
    runtime = CoarseDeviceRuntime()
    result = runtime.execute(graph)
    report = runtime.last_report
    assert report is not None
    step = CompletionReducer(VirtualClock()).reduce(record, graph, result, report)
    expected = width * 1_000_000 + (width - 1) * 1_310_720
    assert step.step_latency_ps == expected
    assert all(metric.ttft_ps == expected for metric in step.request_metrics)
    events = {event.operation_id: event.timestamp_ps for event in result.events
              if event.phase == EventPhase.COMPLETED}
    for prior, current in zip(graph.operations, graph.operations[1:]):
        assert events[prior.operation_id] < events[current.operation_id]
    assert len(report.wqes) == width - 1


@pytest.mark.parametrize("attach_plan", [False, True])
@pytest.mark.parametrize("tokens", [0, 8])
def test_width_one_is_byte_identity(record, dims, attach_plan, tokens):
    record = replace(record, scheduled=[ScheduledRequest("r", RequestPhase.PREFILL, tokens, context_length=tokens)])
    lowerer = SerialStepLowerer(SerialStepLowererConfig(
        dims=dims, tp_ranks=(0, 1), attach_collective_plan=attach_plan,
    ))
    baseline = lowerer.lower(record)
    record_bytes = json.dumps(step_record_to_json(record), sort_keys=True)
    graph_bytes = json.dumps(execution_graph_to_json(baseline), sort_keys=True)
    composed = compose_pipeline_graph(record, dims, 1, ((0, 1),), (baseline,))
    assert composed is baseline
    wrapped = PipelineStepLowerer(dims, 1, ((0, 1),), (lowerer,)).lower(record)
    assert json.dumps(execution_graph_to_json(wrapped), sort_keys=True) == graph_bytes
    assert json.dumps(step_record_to_json(record), sort_keys=True) == record_bytes
    if tokens:
        assert [artifact.trace.render().encode() for artifact in
                project_execution_graph_goal(wrapped).artifacts] == [
                    artifact.trace.render().encode() for artifact in
                    project_execution_graph_goal(baseline).artifacts
                ]
        before_runtime, after_runtime = CoarseDeviceRuntime(), CoarseDeviceRuntime()
        assert before_runtime.execute(baseline) == after_runtime.execute(wrapped)
        assert before_runtime.last_report == after_runtime.last_report


def test_width_one_observations_delegated(record, dims):
    class CapturedLowerer:
        def lower(self, actual_record, observations=None):
            assert actual_record is record
            assert observations is expected_observations
            return expected_graph

    expected_observations = ExecutionObservations()
    expected_graph = stage_graph(0, record)
    wrapped = PipelineStepLowerer(dims, 1, (0,), (CapturedLowerer(),))
    assert wrapped.lower(record, expected_observations) is expected_graph


def test_stage_metadata_and_participants_refused(record, dims):
    graphs = (stage_graph(0, record), stage_graph(8, record))
    for bad in (replace(graphs[1], step_index=1), replace(graphs[1], released_at_ps=1),
                stage_graph(9, record)):
        with pytest.raises(ValueError):
            compose_pipeline_graph(record, dims, 2, (0, 8), (graphs[0], bad))
    with pytest.raises(ValueError, match="one execution graph"):
        compose_pipeline_graph(record, dims, 2, (0, 8), graphs[:1])
    with pytest.raises(ValueError, match="every declared stage rank"):
        compose_pipeline_graph(record, dims, 2, ((0, 1), (8,)), graphs)


def test_multi_rank_stage_completion_is_not_weakened(record, dims):
    graphs = tuple(SerialStepLowerer(SerialStepLowererConfig(dims=dims, tp_ranks=ranks)).lower(record)
                   for ranks in ((0, 1), (8, 9)))
    graph = compose_pipeline_graph(record, dims, 2, ((0, 1), (8, 9)), graphs)
    transfer = next(op for op in graph.operations if op.operation_id == "step-0:pp-forward-0")
    assert transfer.work.pair_payload_bytes == ((1, 8, 65536),)
    assert transfer.depends_on
    assert not transfer.participant_local_depends_on
    assert len(project_execution_graph_goal(graph).artifacts) > 1
    assert CoarseDeviceRuntime().execute(graph).completed_at_ps > 0


def test_stage_background_completion_refused(record, dims):
    first = stage_graph(0, record)
    second = replace(first.operations[0], operation_id="background", logical_queue="other")
    first = replace(first, operations=(*first.operations, second))
    assert compose_pipeline_graph(record, dims, 1, (0,), (first,)) is first
    with pytest.raises(ValueError, match="complete terminal frontier"):
        compose_pipeline_graph(record, dims, 2, (0, 8), (first, stage_graph(8, record)))


def test_drain_has_no_graph_work(record, dims):
    drain = replace(record, scheduled=[])
    empty = ExecutionGraph("step-0", 0, 0)
    assert compose_pipeline_graph(drain, dims, 2, (0, 8), (empty, empty)) is empty
    with pytest.raises(ValueError, match="zero-token"):
        compose_pipeline_graph(drain, dims, 2, (0, 8),
                               (stage_graph(0, drain), stage_graph(8, drain)))


def test_multi_stage_lowerer_preserves_input_graphs(record, dims):
    graphs = tuple(stage_graph(rank, record) for rank in (0, 8))

    class FixedLowerer:
        def __init__(self, graph):
            self.graph = graph

        def lower(self, record, observations=None):
            return self.graph

    snapshots = [execution_graph_to_json(graph) for graph in graphs]
    lowerer = PipelineStepLowerer(dims, 2, (0, 8), tuple(FixedLowerer(g) for g in graphs))
    assert len(lowerer.lower(record).collective_plans) == 1
    assert snapshots == [execution_graph_to_json(graph) for graph in graphs]
    with pytest.raises(ValueError, match="partitioned"):
        lowerer.lower(record, ExecutionObservations())
    with pytest.raises(ValueError, match="one lowerer"):
        PipelineStepLowerer(dims, 2, (0, 8), ())
    with pytest.raises(TypeError, match="ExecutionLowerer"):
        PipelineStepLowerer(dims, 1, (0,), (object(),))


def test_custom_stage_plan_is_not_silently_replaced(record, dims):
    from simllm.traffic import plan_execution_graph_collectives

    stage = SerialStepLowerer(SerialStepLowererConfig(dims=dims, tp_ranks=(0, 1))).lower(record)
    custom = plan_execution_graph_collectives(replace(stage, collective_plans=()), base_tag=5000)
    assert compose_pipeline_graph(record, dims, 1, ((0, 1),), (custom,)) is custom
    with pytest.raises(ValueError, match="canonical"):
        compose_pipeline_graph(record, dims, 2, ((0, 1), (8,)),
                               (custom, stage_graph(8, record)))
