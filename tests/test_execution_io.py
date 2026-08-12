import hashlib
import json
from dataclasses import replace
from types import MappingProxyType

import pytest

from simllm.core import (
    COMPLETION_EVENT_SCHEMA,
    EXECUTION_GRAPH_SCHEMA,
    EXECUTION_RESULT_SCHEMA,
    CollectiveWork,
    CompletionEvent,
    ComputeWork,
    ControlMode,
    ControlWork,
    DmaWork,
    EventPhase,
    ExecutionGraph,
    ExecutionObservations,
    ExecutionOperation,
    ExecutionResult,
    KvCacheAction,
    KvCacheWork,
    OperationCorrelation,
    RequestPhase,
    ResourceKind,
    ResourceRef,
    ScheduledRequest,
    StepRecord,
    completion_event_from_json,
    completion_event_to_json,
    execution_graph_from_json,
    execution_graph_from_observations,
    execution_graph_to_json,
    execution_result_from_json,
    execution_result_to_json,
    validate_execution_graph,
)


def _correlation(layer: int) -> OperationCorrelation:
    return OperationCorrelation(
        request_ids=("request-0", "request-1"),
        batch_id="batch-7",
        layer=layer,
        microbatch=2,
        iteration=3,
    )


def _all_work_graph() -> ExecutionGraph:
    operations = (
        ExecutionOperation(
            "compute",
            0,
            "cuda:0:compute",
            ComputeWork(
                "attention",
                config=(
                    ("tokens", 16),
                    ("dtype", "bf16"),
                    ("scale", 0.5),
                    ("causal", True),
                ),
                flops=1_000,
                hbm_bytes=2_000,
                nominal_duration_ps=300,
                uncertainty_fraction=0.1,
            ),
            correlation=_correlation(0),
            placement_epoch=4,
        ),
        ExecutionOperation(
            "kv",
            0,
            "cuda:0:kv",
            KvCacheWork(
                KvCacheAction.ALLOCATE,
                "gpu:0:kv",
                request_id="request-0",
                block_ids=("block-1",),
                token_start=0,
                token_end=16,
                layer=0,
                dtype="fp8",
                byte_count=4096,
                placement_epoch=4,
                reference_count=1,
                cause="admission",
                correlation_id="kv-1",
            ),
            depends_on=("compute",),
            correlation=_correlation(0),
            placement_epoch=4,
        ),
        ExecutionOperation(
            "dma",
            0,
            "cuda:0:copy",
            DmaWork("descriptor-1", "host:pinned", "gpu:0:hbm", 4096),
            depends_on=("kv",),
            correlation=_correlation(0),
            placement_epoch=4,
        ),
        ExecutionOperation(
            "collective",
            0,
            "cuda:0:nccl",
            CollectiveWork("all-reduce", (0, 1), 4096, "ring", "channel-0"),
            depends_on=("dma",),
            correlation=_correlation(0),
            placement_epoch=4,
        ),
        ExecutionOperation(
            "control",
            0,
            "cuda:0:control",
            ControlWork("barrier", (1,), 64, ControlMode.SYNCHRONOUS),
            depends_on=("collective",),
            priority=7,
            correlation=_correlation(0),
            placement_epoch=4,
        ),
    )
    return ExecutionGraph("execution-7", 7, 100, operations, ("control",))


def test_execution_graph_golden_json_shape():
    graph = ExecutionGraph(
        "execution-1",
        1,
        10,
        (
            ExecutionOperation(
                "compute-0",
                0,
                "cuda:0:compute",
                ComputeWork(
                    "gemm",
                    (("tokens", 8),),
                    100,
                    200,
                    300,
                    0.25,
                ),
                correlation=OperationCorrelation(request_ids=("r0",), layer=0),
            ),
        ),
        ("compute-0",),
    )
    assert execution_graph_to_json(graph) == {
        "schema": EXECUTION_GRAPH_SCHEMA,
        "execution_id": "execution-1",
        "step_index": 1,
        "released_at_ps": 10,
        "operations": [
            {
                "operation_id": "compute-0",
                "rank": 0,
                "logical_queue": "cuda:0:compute",
                "work": {
                    "kind": "compute",
                    "kernel": "gemm",
                    "config": [["tokens", 8]],
                    "flops": 100,
                    "hbm_bytes": 200,
                    "nominal_duration_ps": 300,
                    "uncertainty_fraction": 0.25,
                },
                "depends_on": [],
                "participant_local_depends_on": [],
                "not_before_ps": 0,
                "priority": 0,
                "correlation": {
                    "request_ids": ["r0"],
                    "batch_id": None,
                    "layer": 0,
                    "microbatch": None,
                    "iteration": None,
                },
                "placement_epoch": 0,
            }
        ],
        "completion_operation_ids": ["compute-0"],
    }


def test_all_work_kinds_round_trip_through_real_json():
    graph = _all_work_graph()
    wire = json.loads(json.dumps(execution_graph_to_json(graph)))
    assert execution_graph_from_json(wire) == graph


def _pairwise_graph(work: CollectiveWork) -> ExecutionGraph:
    return ExecutionGraph(
        "core6-uniform",
        7,
        11,
        (
            ExecutionOperation(
                "a2av",
                0,
                "cuda:0:nccl:ep",
                work,
            ),
        ),
        ("a2av",),
    )


def test_uniform_collective_keeps_frozen_v1_json_bytes():
    graph = _pairwise_graph(CollectiveWork("all-to-allv", (0, 1), 2048, "pairwise"))
    wire = (
        json.dumps(
            execution_graph_to_json(graph),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )

    assert len(wire) == 559
    assert hashlib.sha256(wire).hexdigest() == (
        "f4a5a70f5bd4a0c2fed874baa88f3035266a54f386a59927e115872c2bcff0a3"
    )
    assert "pair_payload_bytes" not in wire.decode()
    decoded = execution_graph_from_json(json.loads(wire))
    assert decoded == graph
    assert decoded.operations[0].work.pair_payload_bytes == ()


def test_sparse_pair_payloads_round_trip_as_the_authoritative_table():
    graph = _pairwise_graph(
        CollectiveWork(
            "all-to-allv",
            (0, 1),
            0,
            "pairwise",
            pair_payload_bytes=((0, 1, 2048), (1, 0, 4096)),
        )
    )
    payload = execution_graph_to_json(graph)

    assert payload["schema"] == EXECUTION_GRAPH_SCHEMA
    assert payload["operations"][0]["work"]["payload_bytes"] == 0
    assert payload["operations"][0]["work"]["pair_payload_bytes"] == [
        [0, 1, 2048],
        [1, 0, 4096],
    ]
    assert execution_graph_from_json(json.loads(json.dumps(payload))) == graph


def test_request_pair_partition_round_trips_and_preserves_aggregate_authority():
    work = CollectiveWork(
        "all-to-allv",
        (0, 1),
        0,
        "pairwise",
        pair_payload_bytes=((0, 1, 12), (1, 0, 8)),
        request_pair_payload_bytes=(
            ("alpha", 0, 1, 4),
            ("alpha", 1, 0, 8),
            ("beta", 0, 1, 8),
        ),
    )
    graph = _pairwise_graph(work)
    graph = replace(
        graph,
        operations=(
            replace(
                graph.operations[0],
                correlation=OperationCorrelation(request_ids=("alpha", "beta")),
            ),
        ),
    )

    payload = execution_graph_to_json(graph)

    assert payload["operations"][0]["work"]["request_pair_payload_bytes"] == [
        ["alpha", 0, 1, 4],
        ["alpha", 1, 0, 8],
        ["beta", 0, 1, 8],
    ]
    assert execution_graph_from_json(json.loads(json.dumps(payload))) == graph


@pytest.mark.parametrize(
    ("request_rows", "requests", "match"),
    [
        (
            (("", 0, 1, 12),),
            ("alpha",),
            "must not be blank",
        ),
        (
            (("alpha", 0, 1, 0),),
            ("alpha",),
            "at least 1",
        ),
        (
            (("alpha", 0, 0, 12),),
            ("alpha",),
            "must differ",
        ),
        (
            (("alpha", 0, 2, 12),),
            ("alpha",),
            "belong to ranks",
        ),
        (
            (("alpha", 0, 1, 4), ("alpha", 0, 1, 8)),
            ("alpha",),
            "unique",
        ),
        (
            (("beta", 0, 1, 8), ("alpha", 0, 1, 4)),
            ("alpha", "beta"),
            "request-major",
        ),
        (
            (("alpha", 0, 1, 11),),
            ("alpha",),
            "sums must equal",
        ),
        (
            (("missing", 0, 1, 12),),
            ("alpha",),
            "absent from operation correlation",
        ),
    ],
)
def test_request_pair_partition_rejects_ambiguous_ownership(request_rows, requests, match):
    graph = _pairwise_graph(
        CollectiveWork(
            "all-to-allv",
            (0, 1),
            0,
            "pairwise",
            pair_payload_bytes=((0, 1, 12),),
            request_pair_payload_bytes=request_rows,
        )
    )
    graph = replace(
        graph,
        operations=(
            replace(
                graph.operations[0],
                correlation=OperationCorrelation(request_ids=requests),
            ),
        ),
    )
    with pytest.raises(ValueError, match=match):
        validate_execution_graph(graph)


def test_request_pair_partition_rejects_non_pairwise_collective():
    graph = _pairwise_graph(
        CollectiveWork(
            "all-reduce",
            (0, 1),
            0,
            "ring",
            request_pair_payload_bytes=(("alpha", 0, 1, 12),),
        )
    )
    graph = replace(
        graph,
        operations=(
            replace(
                graph.operations[0],
                correlation=OperationCorrelation(request_ids=("alpha",)),
            ),
        ),
    )

    with pytest.raises(ValueError, match="valid only for pairwise"):
        validate_execution_graph(graph)


@pytest.mark.parametrize(
    ("work", "match"),
    [
        (
            CollectiveWork(
                "all-to-allv",
                (0, 1),
                0,
                "pairwise",
                pair_payload_bytes=((0, 1, 1), (0, 1, 2)),
            ),
            "unique",
        ),
        (
            CollectiveWork(
                "all-to-allv",
                (0, 1),
                0,
                "pairwise",
                pair_payload_bytes=((0, 0, 1),),
            ),
            "must differ",
        ),
        (
            CollectiveWork(
                "all-to-allv",
                (0, 1),
                0,
                "pairwise",
                pair_payload_bytes=((0, 2, 1),),
            ),
            "belong to ranks",
        ),
        (
            CollectiveWork(
                "all-to-allv",
                (0, 1),
                0,
                "pairwise",
                pair_payload_bytes=((0, 1, 0),),
            ),
            "at least 1",
        ),
        (
            CollectiveWork(
                "all-to-allv",
                (0, 1),
                0,
                "pairwise",
                pair_payload_bytes=((1, 0, 1), (0, 1, 1)),
            ),
            "source-major",
        ),
        (
            CollectiveWork(
                "all-to-allv",
                (0, 1),
                1,
                "pairwise",
                pair_payload_bytes=((0, 1, 1),),
            ),
            "cannot both",
        ),
        (
            CollectiveWork(
                "all-reduce",
                (0, 1),
                0,
                "ring",
                pair_payload_bytes=((0, 1, 1),),
            ),
            "only for pairwise",
        ),
    ],
)
def test_sparse_pair_payload_validation_rejects_ambiguous_tables(work, match):
    with pytest.raises(ValueError, match=match):
        validate_execution_graph(_pairwise_graph(work))


def test_observations_are_enveloped_without_reconstructing_policy():
    graph = _all_work_graph()
    observations = ExecutionObservations(graph.operations, graph.completion_operation_ids)
    record = StepRecord(
        step_index=7,
        virtual_time_ps=100,
        scheduled=[ScheduledRequest("request-0", RequestPhase.PREFILL, 16)],
    )
    assert execution_graph_from_observations(record, observations) == replace(
        graph,
        execution_id="step-7",
    )
    assert execution_graph_from_observations(record) == ExecutionGraph("step-7", 7, 100)

    with pytest.raises(ValueError, match="execution_id"):
        execution_graph_from_observations(record, execution_id="")


def test_completion_event_and_result_round_trip():
    events = (
        CompletionEvent("execution-7", "compute", EventPhase.SUBMITTED, 100),
        CompletionEvent(
            "execution-7",
            "compute",
            EventPhase.COMPLETED,
            150,
            ResourceRef(ResourceKind.GPU_SCHEDULER, "gpu:0"),
            completed_bytes=4096,
            subject_object_id="wqe:7",
        ),
    )
    result = ExecutionResult("execution-7", 150, events, quiesced_at_ps=180)
    payload = execution_result_to_json(result)
    assert payload["schema"] == EXECUTION_RESULT_SCHEMA
    assert payload["events"][0]["schema"] == COMPLETION_EVENT_SCHEMA
    assert execution_result_from_json(json.loads(json.dumps(payload))) == result
    assert completion_event_from_json(completion_event_to_json(events[1])) == events[1]
    assert payload["events"][1]["subject_object_id"] == "wqe:7"


@pytest.mark.parametrize(
    "decoder,payload",
    [
        (execution_graph_from_json, {"schema": "simllm-execution-graph-v2"}),
        (completion_event_from_json, {"schema": "simllm-completion-event-v2"}),
        (execution_result_from_json, {"schema": "simllm-execution-result-v2"}),
    ],
)
def test_wire_readers_reject_wrong_schemas(decoder, payload):
    with pytest.raises(ValueError, match="schema"):
        decoder(payload)


def test_core_wire_readers_require_concrete_json_objects_and_arrays():
    payload = execution_graph_to_json(_all_work_graph())
    with pytest.raises(ValueError, match="graph: expected an object"):
        execution_graph_from_json(MappingProxyType(payload))

    payload["operations"] = tuple(payload["operations"])
    with pytest.raises(ValueError, match="graph.operations: expected an array"):
        execution_graph_from_json(payload)


def test_graph_reader_rejects_unknown_fields_and_work_kinds():
    payload = execution_graph_to_json(_all_work_graph())
    payload["typo"] = 1
    with pytest.raises(ValueError, match="unknown fields.*typo"):
        execution_graph_from_json(payload)

    payload = execution_graph_to_json(_all_work_graph())
    payload["operations"][0]["work"]["kind"] = "mystery"
    with pytest.raises(ValueError, match="unknown work kind"):
        execution_graph_from_json(payload)


@pytest.mark.parametrize("bad_value", [True, "7", -1])
def test_graph_reader_rejects_invalid_integer_values(bad_value):
    payload = execution_graph_to_json(_all_work_graph())
    payload["step_index"] = bad_value
    with pytest.raises(ValueError, match="step_index"):
        execution_graph_from_json(payload)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -0.1])
def test_graph_reader_rejects_invalid_uncertainty(bad_value):
    payload = execution_graph_to_json(_all_work_graph())
    payload["operations"][0]["work"]["uncertainty_fraction"] = bad_value
    with pytest.raises(ValueError, match="uncertainty_fraction"):
        execution_graph_from_json(payload)


def test_graph_validation_accepts_forward_cross_queue_dependency():
    graph = ExecutionGraph(
        "forward",
        0,
        0,
        (
            ExecutionOperation(
                "consumer", 0, "queue-a", ComputeWork("a"), depends_on=("producer",)
            ),
            ExecutionOperation("producer", 0, "queue-b", ComputeWork("b")),
        ),
    )
    validate_execution_graph(graph)


def test_participant_local_dependency_requires_a_shared_rank():
    graph = ExecutionGraph(
        "disjoint",
        0,
        0,
        (
            ExecutionOperation("rank-0", 0, "q0", ComputeWork("a")),
            ExecutionOperation(
                "rank-1",
                1,
                "q1",
                ComputeWork("b"),
                participant_local_depends_on=("rank-0",),
            ),
        ),
    )
    with pytest.raises(ValueError, match="no shared rank"):
        validate_execution_graph(graph)


def test_mixed_dependency_edge_modes_round_trip_without_conflation():
    graph = ExecutionGraph(
        "mixed",
        0,
        0,
        (
            ExecutionOperation("global", 0, "q0", ComputeWork("global")),
            ExecutionOperation("local", 0, "q1", ComputeWork("local")),
            ExecutionOperation(
                "consumer",
                0,
                "q2",
                ComputeWork("consumer"),
                depends_on=("global",),
                participant_local_depends_on=("local",),
            ),
        ),
    )

    assert execution_graph_from_json(execution_graph_to_json(graph)) == graph


def test_dependency_cannot_be_both_operation_and_participant_local():
    graph = ExecutionGraph(
        "ambiguous",
        0,
        0,
        (
            ExecutionOperation("producer", 0, "q0", ComputeWork("producer")),
            ExecutionOperation(
                "consumer",
                0,
                "q1",
                ComputeWork("consumer"),
                depends_on=("producer",),
                participant_local_depends_on=("producer",),
            ),
        ),
    )

    with pytest.raises(ValueError, match="two modes"):
        validate_execution_graph(graph)


def test_same_queue_name_on_different_ranks_is_not_one_fifo():
    graph = ExecutionGraph(
        "ranks",
        0,
        0,
        (
            ExecutionOperation("rank-0", 0, "compute", ComputeWork("a"), depends_on=("rank-1",)),
            ExecutionOperation("rank-1", 1, "compute", ComputeWork("b")),
        ),
    )
    validate_execution_graph(graph)


@pytest.mark.parametrize(
    "graph,match",
    [
        (
            ExecutionGraph(
                "dangling",
                0,
                0,
                (ExecutionOperation("a", 0, "a", ComputeWork("a"), depends_on=("missing",)),),
            ),
            "unknown operation ID",
        ),
        (
            ExecutionGraph(
                "explicit-cycle",
                0,
                0,
                (
                    ExecutionOperation("a", 0, "a", ComputeWork("a"), depends_on=("b",)),
                    ExecutionOperation("b", 0, "b", ComputeWork("b"), depends_on=("a",)),
                ),
            ),
            "cycle",
        ),
        (
            ExecutionGraph(
                "fifo-cycle",
                0,
                0,
                (
                    ExecutionOperation("a", 0, "same", ComputeWork("a"), depends_on=("b",)),
                    ExecutionOperation("b", 0, "same", ComputeWork("b")),
                ),
            ),
            "cycle",
        ),
    ],
)
def test_graph_validation_rejects_bad_dependencies(graph, match):
    with pytest.raises(ValueError, match=match):
        validate_execution_graph(graph)


def test_serializers_reject_invalid_in_memory_contracts():
    invalid = ExecutionGraph(
        "bad",
        0,
        0,
        (ExecutionOperation("a", 0, "q", ComputeWork("a", uncertainty_fraction=-1.0)),),
    )
    with pytest.raises(ValueError, match="uncertainty_fraction"):
        execution_graph_to_json(invalid)


def test_graph_reader_normalizes_huge_numeric_overflow_to_validation_error():
    payload = execution_graph_to_json(_all_work_graph())
    payload["operations"][0]["work"]["uncertainty_fraction"] = 10**10_000
    with pytest.raises(ValueError, match="must be finite"):
        execution_graph_from_json(payload)


def test_result_rejects_mismatched_or_out_of_order_events():
    mismatch = ExecutionResult(
        "execution-1",
        10,
        (CompletionEvent("execution-2", "op", EventPhase.COMPLETED, 10),),
    )
    with pytest.raises(ValueError, match="execution_id"):
        execution_result_to_json(mismatch)

    out_of_order = ExecutionResult(
        "execution-1",
        10,
        (
            CompletionEvent("execution-1", "op", EventPhase.COMPLETED, 10),
            CompletionEvent("execution-1", "op", EventPhase.PROGRESS, 9),
        ),
    )
    with pytest.raises(ValueError, match="nondecreasing"):
        execution_result_to_json(out_of_order)

    after_boundary = ExecutionResult(
        "execution-1",
        10,
        (CompletionEvent("execution-1", "op", EventPhase.COMPLETED, 11),),
    )
    with pytest.raises(ValueError, match="boundary"):
        execution_result_to_json(after_boundary)
