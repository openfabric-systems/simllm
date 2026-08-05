from simllm.core import (
    COMPLETION_EVENT_SCHEMA,
    EXECUTION_GRAPH_SCHEMA,
    CollectiveWork,
    CompletionEvent,
    ComputeWork,
    DmaWork,
    EventPhase,
    ExecutionGraph,
    ExecutionOperation,
    ExecutionResult,
    KvCacheAction,
    KvCacheWork,
    ResourceKind,
    ResourceRef,
)


def test_execution_graph_expresses_legal_overlap_without_assigning_resources():
    compute = ExecutionOperation(
        operation_id="compute-0",
        rank=0,
        logical_queue="cuda:0:compute",
        work=ComputeWork(
            kernel="attention",
            config=(("new_tokens", 16), ("context_tokens", 1024)),
            flops=10_000,
            hbm_bytes=4096,
            nominal_duration_ps=100,
        ),
    )
    dma = ExecutionOperation(
        operation_id="dma-0",
        rank=0,
        logical_queue="cuda:0:copy",
        work=DmaWork("descriptor-0", "host:pinned", "gpu:0:hbm", 4096),
    )
    collective = ExecutionOperation(
        operation_id="allreduce-0",
        rank=0,
        logical_queue="cuda:0:nccl",
        work=CollectiveWork("all-reduce", tuple(range(8)), 4096),
        depends_on=("compute-0", "dma-0"),
    )

    graph = ExecutionGraph(
        "step-3",
        3,
        1_000,
        (compute, dma, collective),
        completion_operation_ids=("allreduce-0",),
    )

    assert EXECUTION_GRAPH_SCHEMA == "simllm-execution-graph-v1"
    assert compute.depends_on == dma.depends_on == ()
    assert graph.operations[-1].depends_on == ("compute-0", "dma-0")
    assert graph.completion_operation_ids == ("allreduce-0",)


def test_kv_lifecycle_and_resource_events_share_correlation_ids():
    kv = KvCacheWork(
        action=KvCacheAction.ALLOCATE,
        pool_id="gpu:0:kv",
        request_id="request-7",
        block_ids=("42",),
        token_start=0,
        token_end=16,
        layer=0,
        dtype="fp8",
        byte_count=4096,
        placement_epoch=2,
        reference_count=1,
        correlation_id="kv-7-42",
    )
    resource = ResourceRef(ResourceKind.HBM_QUEUE, "gpu:0:hbm")
    event = CompletionEvent(
        execution_id="step-3",
        operation_id="kv-allocate-0",
        phase=EventPhase.COMPLETED,
        timestamp_ps=1_200,
        resource=resource,
        completed_bytes=kv.byte_count,
    )
    result = ExecutionResult("step-3", completed_at_ps=1_200, events=(event,))

    assert COMPLETION_EVENT_SCHEMA == "simllm-completion-event-v1"
    assert kv.correlation_id == "kv-7-42"
    assert event.resource.kind is ResourceKind.HBM_QUEUE
    assert event.completed_bytes == 4096
    assert result.quiesced_at_ps is None
