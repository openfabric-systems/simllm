import hashlib
from dataclasses import replace

import pytest

from simllm.backends import (
    HtsimStepSink,
    HtsimStepSinkConfig,
    SerialStepLowerer,
    SerialStepLowererConfig,
)
from simllm.compute import ComputeProvider, DurationEstimate, ModelDims
from simllm.core import (
    CollectiveWork,
    ComputeWork,
    DmaWork,
    ExecutionGraph,
    ExecutionLowerer,
    ExecutionObservations,
    ExecutionOperation,
    RequestPhase,
    ScheduledRequest,
    StepRecord,
    execution_graph_from_json,
    execution_graph_to_json,
)
from simllm.traffic import (
    project_execution_graph_goal,
    render_serial_execution_graph_goal,
    render_step_goal,
)

SMALL_DIMS = ModelDims(
    num_layers=2,
    hidden_size=1024,
    intermediate_size=4096,
    num_heads=8,
    num_kv_heads=8,
    head_size=128,
    vocab_size=32000,
    dtype_bytes=2,
)

SMALL_MOE_DIMS = replace(
    SMALL_DIMS,
    num_experts=8,
    top_k=2,
    moe_intermediate_size=512,
    local_num_experts=2,
)

SAMPLE_DIMS = ModelDims(
    num_layers=2,
    hidden_size=64,
    intermediate_size=128,
    num_heads=4,
    num_kv_heads=4,
    head_size=16,
    vocab_size=256,
    dtype_bytes=2,
)

DENSE_LEGACY_GOAL_SHA256 = (
    "f8aade109ba8e3a581b7d965b3a0c76c1247016a1e37491fa84efbbf377677a5"
)
B1_ABSENT_LEGACY_GOAL_SHA256 = (
    "7087db6780f7e34f5a559a6505eeccc15d984c7b478cd8f0bc5838053825d4b6"
)
MOE_GOAL_SHA256 = (
    "5074699d2bd55a12794bdf39d8181c1475e17ca21a134128e2075470c0eb6543"
)


class FlopProvider(ComputeProvider):
    def estimate(self, kernel, gpu):
        return DurationEstimate(duration_ps=int(kernel.flops), bound="compute")


def prefill_record() -> StepRecord:
    return StepRecord(
        0,
        1_000,
        [ScheduledRequest("prefill", RequestPhase.PREFILL, 256, context_length=256)],
    )


def decode_record() -> StepRecord:
    return StepRecord(
        1,
        2_000,
        [
            ScheduledRequest(f"decode-{index}", RequestPhase.DECODE, 1, context_length=128)
            for index in range(2)
        ],
    )


def _round_trip(graph: ExecutionGraph) -> ExecutionGraph:
    return execution_graph_from_json(execution_graph_to_json(graph))


def _collective_message_rows(graph: ExecutionGraph) -> tuple[tuple[object, ...], ...]:
    projection = project_execution_graph_goal(_round_trip(graph))
    return tuple(
        (
            message.operation_id,
            message.source_rank,
            message.destination_rank,
            message.payload_bytes,
            message.tag,
            message.request_payload_bytes,
        )
        for artifact in projection.artifacts
        for message in artifact.trace.messages
    )


def test_serial_lowerer_is_the_execution_lowerer_protocol():
    lowerer = SerialStepLowerer(SerialStepLowererConfig(SMALL_DIMS, (0, 1)))
    assert isinstance(lowerer, ExecutionLowerer)


def test_dense_serial_lowering_is_per_layer_and_correlated():
    graph = SerialStepLowerer(SerialStepLowererConfig(SMALL_DIMS, (0, 1))).lower(
        prefill_record()
    )
    compute = [operation for operation in graph.operations if isinstance(operation.work, ComputeWork)]
    collectives = [
        operation for operation in graph.operations if isinstance(operation.work, CollectiveWork)
    ]
    assert len(compute) == SMALL_DIMS.num_layers * 2
    assert len(collectives) == SMALL_DIMS.num_layers * 2
    assert {operation.correlation.layer for operation in graph.operations} == {0, 1}
    assert {operation.correlation.request_ids for operation in graph.operations} == {("prefill",)}
    assert {operation.work.nominal_duration_ps for operation in compute} == {12_030_000}
    assert {operation.work.payload_bytes for operation in collectives} == {524_288}
    assert {operation.work.algorithm_hint for operation in collectives} == {"ring"}
    assert not any(operation.depends_on for operation in graph.operations)
    assert [operation.participant_local_depends_on for operation in graph.operations] == [
        (),
        (),
        (
            "step-0:layer-0:rank-0:compute",
            "step-0:layer-0:rank-1:compute",
        ),
        ("step-0:layer-0:tp-attention",),
        ("step-0:layer-0:tp-mlp",),
        ("step-0:layer-0:tp-mlp",),
        (
            "step-0:layer-1:rank-0:compute",
            "step-0:layer-1:rank-1:compute",
        ),
        ("step-0:layer-1:tp-attention",),
    ]
    assert graph.completion_operation_ids == ("step-0:layer-1:tp-mlp",)


def test_dense_graph_projects_to_ordered_artifacts_and_keeps_legacy_diagnostic():
    record = prefill_record()
    graph = SerialStepLowerer(SerialStepLowererConfig(SMALL_DIMS, (0, 1))).lower(record)
    replay_graph = _round_trip(graph)
    projection = project_execution_graph_goal(replay_graph)
    legacy = render_step_goal(
        record,
        SMALL_DIMS,
        (0, 1),
        per_layer_calc_ns=12_030,
    )
    # Post-specified re-acceptance: the unchanged graph wire now splits at
    # every causal level; the legacy diagnostic GOAL remains byte-identical.
    assert [artifact.operation_ids for artifact in projection.artifacts] == [
        (
            "step-0:layer-0:rank-0:compute",
            "step-0:layer-0:rank-1:compute",
        ),
        ("step-0:layer-0:tp-attention",),
        ("step-0:layer-0:tp-mlp",),
        (
            "step-0:layer-1:rank-0:compute",
            "step-0:layer-1:rank-1:compute",
        ),
        ("step-0:layer-1:tp-attention",),
        ("step-0:layer-1:tp-mlp",),
    ]
    assert len(projection.boundaries) == 3
    assert len(projection.serialized_edges) == 12
    assert len(_collective_message_rows(graph)) == 16
    assert (
        hashlib.sha256(legacy.render().encode()).hexdigest()
        == DENSE_LEGACY_GOAL_SHA256
    )


def test_b1_exact_sample_count_changes_analytic_compute_not_collective_artifacts(
    tmp_path,
):
    scheduled = [
        ScheduledRequest(
            "p",
            RequestPhase.PREFILL,
            4,
            num_cached_tokens=4,
            context_length=8,
        ),
        ScheduledRequest("d", RequestPhase.DECODE, 1, context_length=32),
    ]
    exact_record = StepRecord(0, 0, scheduled, num_sampled=1)
    absent_record = replace(exact_record, num_sampled=None)
    provider = FlopProvider()
    sink = HtsimStepSink(
        HtsimStepSinkConfig(
            profile="rnic-nn-fluid",
            tp_ranks=(0, 1),
            dims=SAMPLE_DIMS,
            workdir=tmp_path / "live",
            provider=provider,
        )
    )
    lowerer = SerialStepLowerer(
        SerialStepLowererConfig(SAMPLE_DIMS, (0, 1), provider=provider)
    )

    exact_estimate_ps = sink.compute_estimate_ps(exact_record)
    exact_calc_ns = exact_estimate_ps // (SAMPLE_DIMS.num_layers * 1000)
    exact_graph = lowerer.lower(exact_record)
    exact_live = render_step_goal(
        exact_record,
        SAMPLE_DIMS,
        (0, 1),
        per_layer_calc_ns=exact_calc_ns,
    )
    assert exact_estimate_ps == 880_128
    assert exact_calc_ns == 440

    absent_estimate_ps = sink.compute_estimate_ps(absent_record)
    absent_calc_ns = absent_estimate_ps // (SAMPLE_DIMS.num_layers * 1000)
    absent_graph = lowerer.lower(absent_record)
    absent_live = render_step_goal(
        absent_record,
        SAMPLE_DIMS,
        (0, 1),
        per_layer_calc_ns=absent_calc_ns,
    )
    assert absent_estimate_ps == 912_896
    assert absent_calc_ns == 456
    assert absent_estimate_ps - exact_estimate_ps == 32_768
    assert _collective_message_rows(exact_graph) == _collective_message_rows(
        absent_graph
    )
    assert (
        hashlib.sha256(absent_live.render().encode()).hexdigest()
        == B1_ABSENT_LEGACY_GOAL_SHA256
    )
    assert hashlib.sha256(exact_live.render().encode()).hexdigest() != (
        B1_ABSENT_LEGACY_GOAL_SHA256
    )


def test_dense_tp1_lowers_and_renders_as_compute_only_work():
    graph = SerialStepLowerer(SerialStepLowererConfig(SMALL_DIMS, (0,))).lower(
        prefill_record()
    )
    compute = [operation for operation in graph.operations if isinstance(operation.work, ComputeWork)]
    assert len(compute) == SMALL_DIMS.num_layers
    assert not any(isinstance(operation.work, CollectiveWork) for operation in graph.operations)
    assert sum(operation.work.nominal_duration_ps or 0 for operation in compute) == 24_060_000
    assert graph.completion_operation_ids == ("step-0:layer-1:rank-0:compute",)

    rendered = render_serial_execution_graph_goal(_round_trip(graph)).render()
    assert rendered.count("calc 12030") == SMALL_DIMS.num_layers
    assert "send " not in rendered
    assert "recv " not in rendered


def test_moe_graph_projects_global_serial_fallback_and_direct_diagnostic():
    record = decode_record()
    config = SerialStepLowererConfig(
        SMALL_MOE_DIMS,
        (0,),
        ep_ranks=(0, 1, 2, 3),
    )
    graph = SerialStepLowerer(config).lower(record)
    compute = [operation for operation in graph.operations if isinstance(operation.work, ComputeWork)]
    per_layer_calc_ns = compute[0].work.nominal_duration_ps // 1000
    projection = project_execution_graph_goal(_round_trip(graph))
    legacy = render_step_goal(
        record,
        SMALL_MOE_DIMS,
        (0,),
        per_layer_calc_ns=per_layer_calc_ns,
        ep_ranks=(0, 1, 2, 3),
    )
    # The graph separates per-layer compute, dispatch and combine causal
    # levels while preserving the direct diagnostic's physical sends.
    assert len(projection.artifacts) == 6
    assert len(projection.boundaries) == 3
    assert len(projection.serialized_edges) == 24
    assert sum(
        artifact.trace.render().count(": send ")
        for artifact in projection.artifacts
    ) == legacy.render().count(": send ")
    assert (
        hashlib.sha256(legacy.render().encode()).hexdigest()
        == MOE_GOAL_SHA256
    )
    pairwise = [
        operation.work
        for operation in graph.operations
        if isinstance(operation.work, CollectiveWork)
    ]
    assert len(pairwise) == 2 * SMALL_MOE_DIMS.num_layers
    assert {work.collective for work in pairwise} == {"all-to-allv"}
    assert {work.payload_bytes for work in pairwise} == {0}
    assert {
        sum(size for _, _, size in work.pair_payload_bytes)
        for work in pairwise
    } == {3 * 2048}


def test_serial_renderer_distinguishes_participant_and_operation_dependencies():
    compute_0 = ExecutionOperation(
        "compute-0",
        0,
        "compute-0",
        ComputeWork("rank-0", nominal_duration_ps=1_000),
    )
    compute_1 = ExecutionOperation(
        "compute-1",
        1,
        "compute-1",
        ComputeWork("rank-1", nominal_duration_ps=5_000),
    )
    collective = ExecutionOperation(
        "collective",
        0,
        "nccl",
        CollectiveWork("all-reduce", (0, 1), 8, "ring"),
        participant_local_depends_on=("compute-0", "compute-1"),
    )
    graph = ExecutionGraph(
        "asymmetric",
        0,
        0,
        (compute_0, compute_1, collective),
        ("collective",),
    )

    rendered = render_serial_execution_graph_goal(graph).render()
    assert "r0op0: calc 1" in rendered
    assert "r1op0: calc 5" in rendered
    assert "r0op1 requires r0op0" in rendered
    assert "r1op1 requires r1op0" in rendered

    operation_scoped = replace(
        graph,
        operations=(
            *graph.operations[:2],
            replace(
                collective,
                depends_on=collective.participant_local_depends_on,
                participant_local_depends_on=(),
            ),
        ),
    )
    with pytest.raises(ValueError, match="distributed whole-operation edge"):
        render_serial_execution_graph_goal(operation_scoped)


def test_operation_barrier_is_not_removed_by_local_transitive_reduction():
    distributed = ExecutionOperation(
        "distributed",
        0,
        "nccl",
        CollectiveWork("all-reduce", (0, 1), 8, "ring"),
    )
    local = ExecutionOperation(
        "local",
        0,
        "compute-local",
        ComputeWork("local", nominal_duration_ps=1_000),
        participant_local_depends_on=("distributed",),
    )
    target = ExecutionOperation(
        "target",
        0,
        "compute-target",
        ComputeWork("target", nominal_duration_ps=1_000),
        depends_on=("distributed",),
        participant_local_depends_on=("local",),
    )
    graph = ExecutionGraph(
        "mixed-barrier",
        0,
        0,
        (distributed, local, target),
        ("target",),
    )

    with pytest.raises(ValueError, match="distributed whole-operation edge"):
        render_serial_execution_graph_goal(graph)


def test_explicit_observations_replace_the_serial_fallback_exactly():
    record = prefill_record()
    operation = ExecutionOperation(
        "observed-dma",
        0,
        "cuda:0:copy",
        DmaWork("descriptor", "host", "gpu:0", 64),
    )
    observations = ExecutionObservations((operation,), (operation.operation_id,))
    graph = SerialStepLowerer(SerialStepLowererConfig(SMALL_DIMS, (0, 1))).lower(
        record,
        observations,
    )
    assert graph.operations == (operation,)
    assert graph.completion_operation_ids == ("observed-dma",)
    assert graph.released_at_ps == record.virtual_time_ps


def test_drain_record_lowers_to_an_immediate_empty_graph():
    record = StepRecord(9, 10_000, finished_request_ids=["done"])
    graph = SerialStepLowerer(SerialStepLowererConfig(SMALL_DIMS, (0, 1))).lower(record)
    assert graph == ExecutionGraph("step-9", 9, 10_000)


@pytest.mark.parametrize(
    "graph,match",
    [
        (
            ExecutionGraph(
                "dma",
                0,
                0,
                (ExecutionOperation("dma", 0, "copy", DmaWork("d", "a", "b", 1)),),
            ),
            "unsupported DmaWork",
        ),
        (
            ExecutionGraph(
                "gate",
                0,
                0,
                (
                    ExecutionOperation(
                        "compute",
                        0,
                        "compute",
                        ComputeWork("k", nominal_duration_ps=1_000),
                        not_before_ps=1,
                    ),
                ),
            ),
            "timing gate",
        ),
        (
            ExecutionGraph(
                "precision",
                0,
                0,
                (
                    ExecutionOperation(
                        "compute",
                        0,
                        "compute",
                        ComputeWork("k", nominal_duration_ps=1_001),
                    ),
                ),
            ),
            "whole GOAL nanoseconds",
        ),
    ],
)
def test_serial_goal_renderer_rejects_work_it_cannot_preserve(graph, match):
    expected_error = TypeError if "unsupported DmaWork" in match else ValueError
    with pytest.raises(expected_error, match=match):
        render_serial_execution_graph_goal(graph)


def test_serial_goal_renderer_rejects_zero_payload_pairwise_all_to_allv():
    graph = ExecutionGraph(
        "zero-a2a",
        0,
        0,
        (
            ExecutionOperation(
                "c0",
                0,
                "cuda:0:compute",
                ComputeWork("gemm", nominal_duration_ps=1_000),
            ),
            ExecutionOperation(
                "a2a",
                0,
                "cuda:0:nccl:ep",
                CollectiveWork("all-to-allv", (0, 1), 0, algorithm_hint="pairwise"),
                participant_local_depends_on=("c0",),
            ),
        ),
        ("a2a",),
    )
    with pytest.raises(ValueError, match="zero-payload pairwise"):
        render_serial_execution_graph_goal(graph)


def test_serial_goal_renderer_rejects_single_rank_pairwise_all_to_allv():
    graph = ExecutionGraph(
        "solo-a2a",
        0,
        0,
        (
            ExecutionOperation(
                "a2a",
                0,
                "cuda:0:nccl:ep",
                CollectiveWork("all-to-allv", (0,), 4096, algorithm_hint="pairwise"),
            ),
        ),
        ("a2a",),
    )
    with pytest.raises(ValueError, match="single-rank pairwise"):
        render_serial_execution_graph_goal(graph)


def test_serial_goal_renderer_preserves_uncovered_sparse_rank_frontier():
    graph = ExecutionGraph(
        "uncovered-a2a-rank",
        0,
        0,
        (
            ExecutionOperation(
                "a2a",
                0,
                "cuda:0:nccl:ep",
                CollectiveWork(
                    "all-to-allv",
                    (0, 1, 2),
                    0,
                    algorithm_hint="pairwise",
                    pair_payload_bytes=((0, 1, 4096),),
                ),
                ),
                ExecutionOperation(
                    "after",
                    2,
                    "cuda:2:compute",
                    ComputeWork("after", nominal_duration_ps=1_000),
                    participant_local_depends_on=("a2a",),
                ),
            ),
        ("after",),
    )
    trace = render_serial_execution_graph_goal(graph)

    rank2_operations = [
        operation
        for operation in trace.operations
        if operation.rank == 2
    ]
    assert [operation.text for operation in rank2_operations] == [
        "calc 0",
        "calc 1",
    ]
    assert rank2_operations[0].operation_id == "a2a"
    assert rank2_operations[1].operation_id == "after"


def test_serial_renderer_uses_sparse_pair_payload_sizes_exactly():
    graph = ExecutionGraph(
        "sparse-a2av",
        0,
        0,
        (
            ExecutionOperation(
                "a2av",
                0,
                "cuda:0:nccl:ep",
                CollectiveWork(
                    "all-to-allv",
                    (0, 1),
                    0,
                    "pairwise",
                    pair_payload_bytes=((0, 1, 2048), (1, 0, 4096)),
                ),
            ),
        ),
        ("a2av",),
    )

    text = render_serial_execution_graph_goal(_round_trip(graph)).render()
    assert "send 2048b to 1 tag 1000" in text
    assert "recv 2048b from 0 tag 1000" in text
    assert "send 4096b to 0 tag 1000" in text
    assert "recv 4096b from 1 tag 1000" in text
    assert text.count(": send ") == 2
    assert text.count(": recv ") == 2


def test_serial_lowerer_config_rejects_bad_rank_sets():
    with pytest.raises(ValueError, match="at least one"):
        SerialStepLowererConfig(SMALL_DIMS, ())
    with pytest.raises(ValueError, match="duplicates"):
        SerialStepLowererConfig(SMALL_DIMS, (0, 0))
    with pytest.raises(ValueError, match="non-negative"):
        SerialStepLowererConfig(SMALL_DIMS, (-1,))
