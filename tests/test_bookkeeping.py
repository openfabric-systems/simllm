import copy
import json
from dataclasses import replace

import pytest

from simllm.core.bookkeeping import (
    BOOKKEEPING_SCHEMA,
    BookkeepingEntry,
    BookkeepingLedger,
    BookkeepingScope,
    CreatedObjectKind,
    CreatedObjectRecord,
    CreatedObjectRef,
    ObjectOwner,
    ProcessingStage,
    RequestBookkeeper,
    StagePhase,
    StageRecord,
    validate_bookkeeping_ledger,
)
from simllm.core.bookkeeping_io import (
    bookkeeping_ledger_from_json,
    bookkeeping_ledger_to_json,
)
from simllm.core.execution import (
    CompletionEvent,
    ComputeWork,
    EventPhase,
    ExecutionGraph,
    ExecutionOperation,
    ExecutionResult,
    OperationCorrelation,
    ResourceKind,
    ResourceRef,
)
from simllm.core.execution_io import execution_result_from_json, execution_result_to_json


def _scope() -> BookkeepingScope:
    return BookkeepingScope(
        correlation=OperationCorrelation(
            request_ids=("request-0",),
            batch_id="batch-0",
            layer=3,
            microbatch=1,
            iteration=2,
        ),
        step_index=7,
        execution_id="execution-7",
        operation_id="collective-3",
    )


def _record(
    kind: CreatedObjectKind,
    object_id: str,
    timestamp_ps: int,
    *,
    owner: ObjectOwner = ObjectOwner.DEVICE_RUNTIME,
    parents: tuple[CreatedObjectRef, ...] = (),
    native_id: str | None = None,
    metadata: tuple[tuple[str, str | int | float | bool], ...] = (),
    scope: BookkeepingScope | None = None,
) -> CreatedObjectRecord:
    return CreatedObjectRecord(
        CreatedObjectRef(kind, object_id),
        owner,
        timestamp_ps,
        scope or _scope(),
        native_id,
        parents,
        metadata,
    )


def _network_ledger(
    transport_kind: CreatedObjectKind = CreatedObjectKind.DCQCN_QP,
) -> BookkeepingLedger:
    bookkeeper = RequestBookkeeper()
    request = _record(
        CreatedObjectKind.FRAMEWORK_REQUEST,
        "request-object:0",
        0,
        owner=ObjectOwner.FRAMEWORK,
        native_id="request-0",
    )
    vram = _record(
        CreatedObjectKind.VRAM_ALLOCATION,
        "vram:0",
        1,
        owner=ObjectOwner.FRAMEWORK,
        parents=(request.ref,),
        native_id="0x00000000000000ff",
        metadata=(("bytes", 8192), ("resident", True), ("ratio", 0.5), ("pool", "kv")),
    )
    operation = _record(
        CreatedObjectKind.EXECUTION_OPERATION,
        "operation:0",
        2,
        owner=ObjectOwner.CORE,
        parents=(vram.ref,),
        native_id="collective-3",
    )
    nccl = _record(
        CreatedObjectKind.NCCL_COMMAND,
        "nccl:0",
        3,
        owner=ObjectOwner.NCCL,
        parents=(operation.ref,),
        native_id="nccl-command/0007",
    )
    sq = _record(CreatedObjectKind.SEND_QUEUE, "sq:0", 4, parents=(nccl.ref,))
    rq = _record(CreatedObjectKind.RECEIVE_QUEUE, "rq:0", 5, parents=(nccl.ref,))
    cq = _record(CreatedObjectKind.COMPLETION_QUEUE, "cq:0", 6, parents=(nccl.ref,))
    transport = _record(
        transport_kind,
        "transport:0",
        7,
        owner=ObjectOwner.NETWORK_BACKEND,
        parents=(sq.ref, rq.ref, cq.ref),
        native_id="qp=17" if transport_kind is CreatedObjectKind.DCQCN_QP else "0->1",
    )
    wqe = _record(
        CreatedObjectKind.NETWORK_WQE,
        "wqe:0",
        8,
        parents=(nccl.ref, sq.ref, rq.ref, cq.ref, transport.ref),
        native_id="wqe/00000042",
        metadata=(("bytes", 4096),),
    )
    bookkeeper.extend((request, vram, operation, nccl, sq, rq, cq, transport, wqe))
    bookkeeper.append(
        StageRecord(
            ProcessingStage.NETWORK,
            StagePhase.ENTERED,
            9,
            _scope(),
            (wqe.ref,),
        )
    )
    for timestamp, phase in (
        (10, EventPhase.SUBMITTED),
        (11, EventPhase.QUEUED),
        (12, EventPhase.STARTED),
        (18, EventPhase.COMPLETED),
    ):
        bookkeeper.append(
            CompletionEvent(
                "execution-7",
                "collective-3",
                phase,
                timestamp,
                ResourceRef(
                    ResourceKind.COMPLETION_QUEUE
                    if phase is EventPhase.COMPLETED
                    else ResourceKind.NIC_SEND_QUEUE,
                    "cq:0" if phase is EventPhase.COMPLETED else "sq:0",
                ),
                completed_bytes=4096 if phase is EventPhase.COMPLETED else None,
                subject_object_id=wqe.ref.object_id,
            )
        )
    bookkeeper.append(
        StageRecord(
            ProcessingStage.NETWORK,
            StagePhase.COMPLETED,
            19,
            _scope(),
            (wqe.ref,),
        )
    )
    return bookkeeper.snapshot()


def test_graph_registration_is_immutable_and_queryable_by_request():
    graph = ExecutionGraph(
        "execution-7",
        7,
        20,
        (
            ExecutionOperation(
                "compute-a",
                0,
                "cuda:0:compute",
                ComputeWork("gemm-a", nominal_duration_ps=10),
                correlation=OperationCorrelation(request_ids=("request-a",), layer=0),
            ),
            ExecutionOperation(
                "compute-b",
                0,
                "cuda:0:compute",
                ComputeWork("gemm-b", nominal_duration_ps=10),
                correlation=OperationCorrelation(request_ids=("request-b",), layer=1),
            ),
        ),
        ("compute-b",),
    )
    original = replace(graph)
    original_operations = graph.operations

    bookkeeper = RequestBookkeeper()
    entries = bookkeeper.register_graph(graph)
    event_entry = bookkeeper.append(
        CompletionEvent(
            "execution-7",
            "compute-a",
            EventPhase.COMPLETED,
            30,
        )
    )

    assert graph == original
    assert graph.operations is original_operations
    assert len(entries) == 3
    assert [entry.sequence for entry in bookkeeper.for_request("request-a")] == [0, 1, 3]
    assert [entry.sequence for entry in bookkeeper.for_request("request-b")] == [0, 2]
    assert len(bookkeeper.for_execution("execution-7")) == 4
    operation_ref = CreatedObjectRef(
        CreatedObjectKind.EXECUTION_OPERATION,
        "execution-operation:execution-7:compute-a",
    )
    assert event_entry in bookkeeper.for_object(operation_ref)


def test_independent_executions_may_be_appended_out_of_timestamp_order():
    bookkeeper = RequestBookkeeper()
    bookkeeper.append(
        StageRecord(
            ProcessingStage.COMPLETION,
            StagePhase.COMPLETED,
            100,
            BookkeepingScope(execution_id="execution-1"),
        )
    )
    bookkeeper.register_graph(ExecutionGraph("execution-2", 2, 50))

    assert [entry.sequence for entry in bookkeeper.snapshot().entries] == [0, 1]
    assert [
        entry.fact.timestamp_ps
        for entry in bookkeeper.snapshot().entries
        if isinstance(entry.fact, StageRecord)
    ] == [100, 50]


def test_request_and_framework_vram_facts_can_predate_an_execution():
    scope = BookkeepingScope(
        correlation=OperationCorrelation(request_ids=("request-early",))
    )
    request = CreatedObjectRecord(
        CreatedObjectRef(CreatedObjectKind.FRAMEWORK_REQUEST, "request-object:early"),
        ObjectOwner.FRAMEWORK,
        0,
        scope,
        native_id="request-early",
    )
    vram = CreatedObjectRecord(
        CreatedObjectRef(CreatedObjectKind.VRAM_ALLOCATION, "vram:early"),
        ObjectOwner.FRAMEWORK,
        1,
        scope,
        native_id="0x0100",
        parent_refs=(request.ref,),
    )
    bookkeeper = RequestBookkeeper()
    bookkeeper.extend(
        (
            StageRecord(
                ProcessingStage.REQUEST,
                StagePhase.ENTERED,
                0,
                scope,
                (),
            ),
            request,
            vram,
        )
    )

    assert scope.execution_id is None
    assert [entry.sequence for entry in bookkeeper.for_request("request-early")] == [0, 1, 2]


def test_dcqcn_chain_is_typed_and_recursively_queryable():
    ledger = _network_ledger()
    bookkeeper = RequestBookkeeper(ledger)
    vram = CreatedObjectRef(CreatedObjectKind.VRAM_ALLOCATION, "vram:0")

    object_records = [
        entry.fact
        for entry in bookkeeper.for_object(vram, recursive=True)
        if isinstance(entry.fact, CreatedObjectRecord)
    ]
    assert [record.ref.kind for record in object_records] == [
        CreatedObjectKind.VRAM_ALLOCATION,
        CreatedObjectKind.EXECUTION_OPERATION,
        CreatedObjectKind.NCCL_COMMAND,
        CreatedObjectKind.SEND_QUEUE,
        CreatedObjectKind.RECEIVE_QUEUE,
        CreatedObjectKind.COMPLETION_QUEUE,
        CreatedObjectKind.DCQCN_QP,
        CreatedObjectKind.NETWORK_WQE,
    ]
    wqe = object_records[-1]
    assert wqe.parent_refs == (
        CreatedObjectRef(CreatedObjectKind.NCCL_COMMAND, "nccl:0"),
        CreatedObjectRef(CreatedObjectKind.SEND_QUEUE, "sq:0"),
        CreatedObjectRef(CreatedObjectKind.RECEIVE_QUEUE, "rq:0"),
        CreatedObjectRef(CreatedObjectKind.COMPLETION_QUEUE, "cq:0"),
        CreatedObjectRef(CreatedObjectKind.DCQCN_QP, "transport:0"),
    )
    assert len(bookkeeper.for_request("request-0")) == len(ledger.entries)

    wrong_kind = CreatedObjectRef(CreatedObjectKind.RECEIVE_QUEUE, "wqe:0")
    assert bookkeeper.for_object(wrong_kind, recursive=True) == ()


def test_rnic_cn_wqe_uses_a_link_pair_instead_of_a_qp():
    ledger = _network_ledger(CreatedObjectKind.RNIC_CN_LINK_PAIR)
    transport = next(
        entry.fact
        for entry in ledger.entries
        if isinstance(entry.fact, CreatedObjectRecord)
        and entry.fact.ref.kind is CreatedObjectKind.RNIC_CN_LINK_PAIR
    )
    wqe = next(
        entry.fact
        for entry in ledger.entries
        if isinstance(entry.fact, CreatedObjectRecord)
        and entry.fact.ref.kind is CreatedObjectKind.NETWORK_WQE
    )
    assert transport.native_id == "0->1"
    assert transport.ref in wqe.parent_refs
    validate_bookkeeping_ledger(ledger)


def test_wqe_completion_is_the_shared_ledger_and_result_subject():
    ledger = _network_ledger()
    events = tuple(
        entry.fact for entry in ledger.entries if isinstance(entry.fact, CompletionEvent)
    )
    result = ExecutionResult("execution-7", 18, events, quiesced_at_ps=18)

    assert events[-1].phase is EventPhase.COMPLETED
    assert events[-1].subject_object_id == "wqe:0"
    assert events[-1].completed_bytes == 4096
    assert events[-1].resource == ResourceRef(ResourceKind.COMPLETION_QUEUE, "cq:0")
    assert execution_result_from_json(execution_result_to_json(result)) == result


def test_bookkeeping_json_round_trip_is_lossless_and_discriminated():
    ledger = _network_ledger()
    payload = bookkeeping_ledger_to_json(ledger)
    wire = json.loads(json.dumps(payload))

    assert wire["schema"] == BOOKKEEPING_SCHEMA
    assert {entry["fact"]["kind"] for entry in wire["entries"]} == {
        "object",
        "stage",
        "completion",
    }
    vram = next(
        entry["fact"]
        for entry in wire["entries"]
        if entry["fact"]["kind"] == "object"
        and entry["fact"]["ref"]["kind"] == "vram-allocation"
    )
    assert vram["native_id"] == "0x00000000000000ff"
    assert vram["metadata"] == [
        ["bytes", 8192],
        ["resident", True],
        ["ratio", 0.5],
        ["pool", "kv"],
    ]
    assert vram["scope"]["correlation"] == {
        "request_ids": ["request-0"],
        "batch_id": "batch-0",
        "layer": 3,
        "microbatch": 1,
        "iteration": 2,
    }
    assert bookkeeping_ledger_from_json(wire) == ledger


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda payload: payload.update(schema="simllm-request-bookkeeping-v2"), "schema"),
        (lambda payload: payload.update(typo=True), "unknown fields.*typo"),
        (lambda payload: payload.pop("entries"), "missing fields.*entries"),
        (
            lambda payload: payload["entries"][1]["fact"].update(native_id=17),
            "native_id",
        ),
        (
            lambda payload: payload["entries"][1]["fact"]["scope"]["correlation"].update(
                request_ids="request-0"
            ),
            "request_ids",
        ),
        (
            lambda payload: payload["entries"][1]["fact"].update(metadata=[["bad", None]]),
            "finite JSON scalar",
        ),
        (lambda payload: payload["entries"][0].update(sequence=True), "sequence"),
        (
            lambda payload: payload["entries"][0]["fact"].update(kind="packet"),
            "unknown entry kind",
        ),
    ],
)
def test_wire_reader_rejects_schema_shape_and_type_violations(mutation, match):
    payload = copy.deepcopy(bookkeeping_ledger_to_json(_network_ledger()))
    mutation(payload)
    with pytest.raises(ValueError, match=match):
        bookkeeping_ledger_from_json(payload)


def test_wire_reader_rejects_unknown_parent_and_noncontiguous_sequence():
    payload = bookkeeping_ledger_to_json(_network_ledger())
    payload["entries"][1]["fact"]["parent_refs"][0]["object_id"] = "missing"
    with pytest.raises(ValueError, match="unknown object reference"):
        bookkeeping_ledger_from_json(payload)

    payload = bookkeeping_ledger_to_json(_network_ledger())
    payload["entries"][4]["sequence"] = 99
    with pytest.raises(ValueError, match="contiguous value"):
        bookkeeping_ledger_from_json(payload)


def test_ledger_rejects_wrong_typed_parent_and_multiple_wqe_transports():
    ledger = _network_ledger()
    entries = list(ledger.entries)
    wqe_index = next(
        index
        for index, entry in enumerate(entries)
        if isinstance(entry.fact, CreatedObjectRecord)
        and entry.fact.ref.kind is CreatedObjectKind.NETWORK_WQE
    )
    wqe = entries[wqe_index].fact
    assert isinstance(wqe, CreatedObjectRecord)
    wrong_typed_sq = CreatedObjectRef(CreatedObjectKind.RECEIVE_QUEUE, "sq:0")
    entries[wqe_index] = replace(
        entries[wqe_index],
        fact=replace(wqe, parent_refs=(wrong_typed_sq, *wqe.parent_refs[1:])),
    )
    with pytest.raises(ValueError, match="unknown object reference"):
        validate_bookkeeping_ledger(BookkeepingLedger(tuple(entries)))

    prefix = BookkeepingLedger(tuple(ledger.entries[:wqe_index]))
    bookkeeper = RequestBookkeeper(prefix)
    link_pair = _record(
        CreatedObjectKind.RNIC_CN_LINK_PAIR,
        "transport:second",
        8,
        owner=ObjectOwner.NETWORK_BACKEND,
        parents=(
            CreatedObjectRef(CreatedObjectKind.SEND_QUEUE, "sq:0"),
            CreatedObjectRef(CreatedObjectKind.RECEIVE_QUEUE, "rq:0"),
            CreatedObjectRef(CreatedObjectKind.COMPLETION_QUEUE, "cq:0"),
        ),
        native_id="0->1",
    )
    bookkeeper.append(link_pair)
    with pytest.raises(ValueError, match="at most one QP or link pair"):
        bookkeeper.append(
            replace(
                wqe,
                parent_refs=(*wqe.parent_refs, link_pair.ref),
            )
        )


def test_null_network_wqe_can_omit_a_transport_object():
    ledger = _network_ledger()
    entries = list(ledger.entries)
    wqe_index = next(
        index
        for index, entry in enumerate(entries)
        if isinstance(entry.fact, CreatedObjectRecord)
        and entry.fact.ref.kind is CreatedObjectKind.NETWORK_WQE
    )
    wqe = entries[wqe_index].fact
    assert isinstance(wqe, CreatedObjectRecord)
    entries[wqe_index] = replace(
        entries[wqe_index],
        fact=replace(
            wqe,
            parent_refs=tuple(
                parent
                for parent in wqe.parent_refs
                if parent.kind not in {
                    CreatedObjectKind.DCQCN_QP,
                    CreatedObjectKind.RNIC_CN_LINK_PAIR,
                }
            ),
            metadata=(("bytes", 4096), ("transport_kind", "none")),
        ),
    )

    validate_bookkeeping_ledger(BookkeepingLedger(tuple(entries)))


@pytest.mark.parametrize(
    "missing_kind,match",
    [
        (CreatedObjectKind.RECEIVE_QUEUE, "exactly one receive queue"),
        (CreatedObjectKind.COMPLETION_QUEUE, "exactly one completion queue"),
    ],
)
def test_wqe_requires_rq_and_cq_placeholders(missing_kind, match):
    ledger = _network_ledger()
    entries = list(ledger.entries)
    wqe_index = next(
        index
        for index, entry in enumerate(entries)
        if isinstance(entry.fact, CreatedObjectRecord)
        and entry.fact.ref.kind is CreatedObjectKind.NETWORK_WQE
    )
    wqe = entries[wqe_index].fact
    assert isinstance(wqe, CreatedObjectRecord)
    entries[wqe_index] = replace(
        entries[wqe_index],
        fact=replace(
            wqe,
            parent_refs=tuple(parent for parent in wqe.parent_refs if parent.kind is not missing_kind),
        ),
    )
    with pytest.raises(ValueError, match=match):
        validate_bookkeeping_ledger(BookkeepingLedger(tuple(entries)))


def test_transport_free_wqe_must_be_explicit_and_physical_wqe_cannot_claim_none():
    ledger = _network_ledger()
    entries = list(ledger.entries)
    wqe_index = next(
        index
        for index, entry in enumerate(entries)
        if isinstance(entry.fact, CreatedObjectRecord)
        and entry.fact.ref.kind is CreatedObjectKind.NETWORK_WQE
    )
    wqe = entries[wqe_index].fact
    assert isinstance(wqe, CreatedObjectRecord)
    queue_parents = tuple(
        parent
        for parent in wqe.parent_refs
        if parent.kind
        in {
            CreatedObjectKind.SEND_QUEUE,
            CreatedObjectKind.RECEIVE_QUEUE,
            CreatedObjectKind.COMPLETION_QUEUE,
        }
    )
    entries[wqe_index] = replace(
        entries[wqe_index],
        fact=replace(wqe, parent_refs=queue_parents),
    )
    with pytest.raises(ValueError, match="transport_kind=none"):
        validate_bookkeeping_ledger(BookkeepingLedger(tuple(entries)))

    entries = list(ledger.entries)
    entries[wqe_index] = replace(
        entries[wqe_index],
        fact=replace(wqe, metadata=(("transport_kind", "none"),)),
    )
    with pytest.raises(ValueError, match="physical WQE"):
        validate_bookkeeping_ledger(BookkeepingLedger(tuple(entries)))


def test_ledger_rejects_duplicate_wqe_completion_and_scope_mismatch():
    ledger = _network_ledger()
    completion = next(
        entry.fact
        for entry in reversed(ledger.entries)
        if isinstance(entry.fact, CompletionEvent)
        and entry.fact.phase is EventPhase.COMPLETED
    )
    bookkeeper = RequestBookkeeper(ledger)
    with pytest.raises(ValueError, match="more than one completion"):
        bookkeeper.append(replace(completion, timestamp_ps=20))

    entries = list(ledger.entries)
    completion_index = next(
        index
        for index, entry in enumerate(entries)
        if isinstance(entry.fact, CompletionEvent)
    )
    entries[completion_index] = replace(
        entries[completion_index],
        fact=replace(entries[completion_index].fact, operation_id="wrong-operation"),
    )
    with pytest.raises(ValueError, match="does not match subject scope"):
        validate_bookkeeping_ledger(BookkeepingLedger(tuple(entries)))


def test_wqe_completion_must_name_its_own_completion_queue():
    ledger = _network_ledger()
    entries = list(ledger.entries)
    completion_index = next(
        index
        for index, entry in enumerate(entries)
        if isinstance(entry.fact, CompletionEvent)
        and entry.fact.phase is EventPhase.COMPLETED
    )
    completion = entries[completion_index].fact
    assert isinstance(completion, CompletionEvent)
    entries[completion_index] = replace(
        entries[completion_index],
        fact=replace(
            completion,
            resource=ResourceRef(ResourceKind.COMPLETION_QUEUE, "cq:wrong"),
        ),
    )
    with pytest.raises(ValueError, match="must name its completion queue"):
        validate_bookkeeping_ledger(BookkeepingLedger(tuple(entries)))


def test_object_lineage_must_inherit_parent_request_and_execution_scope():
    ledger = _network_ledger()
    entries = list(ledger.entries)
    vram_index = next(
        index
        for index, entry in enumerate(entries)
        if isinstance(entry.fact, CreatedObjectRecord)
        and entry.fact.ref.kind is CreatedObjectKind.VRAM_ALLOCATION
    )
    vram = entries[vram_index].fact
    assert isinstance(vram, CreatedObjectRecord)
    entries[vram_index] = replace(
        entries[vram_index],
        fact=replace(
            vram,
            scope=replace(
                vram.scope,
                correlation=replace(vram.scope.correlation, request_ids=("other-request",)),
            ),
        ),
    )
    with pytest.raises(ValueError, match="introduces request IDs"):
        validate_bookkeeping_ledger(BookkeepingLedger(tuple(entries)))

    entries = list(ledger.entries)
    wqe_index = next(
        index
        for index, entry in enumerate(entries)
        if isinstance(entry.fact, CreatedObjectRecord)
        and entry.fact.ref.kind is CreatedObjectKind.NETWORK_WQE
    )
    wqe = entries[wqe_index].fact
    assert isinstance(wqe, CreatedObjectRecord)
    entries[wqe_index] = replace(
        entries[wqe_index],
        fact=replace(wqe, scope=replace(wqe.scope, execution_id="other-execution")),
    )
    with pytest.raises(ValueError, match="inherit parent execution_id"):
        validate_bookkeeping_ledger(BookkeepingLedger(tuple(entries)))


def test_batched_parent_may_split_but_child_cannot_introduce_a_request():
    parent_scope = BookkeepingScope(
        correlation=OperationCorrelation(
            request_ids=("request-a", "request-b"),
            batch_id="batch-ab",
        ),
        step_index=1,
        execution_id="execution-ab",
        operation_id="operation-ab",
    )
    child_scope = replace(
        parent_scope,
        correlation=replace(parent_scope.correlation, request_ids=("request-a",)),
    )
    parent = _record(
        CreatedObjectKind.EXECUTION_OPERATION,
        "operation:ab",
        0,
        owner=ObjectOwner.CORE,
        scope=parent_scope,
    )
    child = _record(
        CreatedObjectKind.NCCL_COMMAND,
        "nccl:a",
        1,
        owner=ObjectOwner.NCCL,
        parents=(parent.ref,),
        scope=child_scope,
    )

    validate_bookkeeping_ledger(
        BookkeepingLedger(
            (
                BookkeepingEntry(0, parent),
                BookkeepingEntry(1, child),
            )
        )
    )

    introduced = replace(
        child,
        scope=replace(
            child.scope,
            correlation=replace(
                child.scope.correlation,
                request_ids=("request-a", "request-c"),
            ),
        ),
    )
    with pytest.raises(ValueError, match="request-c"):
        validate_bookkeeping_ledger(
            BookkeepingLedger(
                (
                    BookkeepingEntry(0, parent),
                    BookkeepingEntry(1, introduced),
                )
            )
        )


def test_persistent_queues_and_transport_can_serve_later_executions():
    first = _network_ledger()
    objects = {
        entry.fact.ref.kind: entry.fact
        for entry in first.entries
        if isinstance(entry.fact, CreatedObjectRecord)
    }
    second_scope = BookkeepingScope(
        correlation=OperationCorrelation(
            request_ids=("request-1",),
            batch_id="batch-1",
            layer=3,
            microbatch=1,
            iteration=2,
        ),
        step_index=8,
        execution_id="execution-8",
        operation_id="collective-8",
    )
    operation = _record(
        CreatedObjectKind.EXECUTION_OPERATION,
        "operation:8",
        20,
        owner=ObjectOwner.CORE,
        native_id="collective-8",
        scope=second_scope,
    )
    nccl = _record(
        CreatedObjectKind.NCCL_COMMAND,
        "nccl:8",
        21,
        owner=ObjectOwner.NCCL,
        parents=(operation.ref,),
        scope=second_scope,
    )
    shared_refs = tuple(
        objects[kind].ref
        for kind in (
            CreatedObjectKind.SEND_QUEUE,
            CreatedObjectKind.RECEIVE_QUEUE,
            CreatedObjectKind.COMPLETION_QUEUE,
            CreatedObjectKind.DCQCN_QP,
        )
    )
    wqe = _record(
        CreatedObjectKind.NETWORK_WQE,
        "wqe:8",
        22,
        parents=(nccl.ref, *shared_refs),
        native_id="wqe/00000043",
        metadata=(("bytes", 2048),),
        scope=second_scope,
    )

    bookkeeper = RequestBookkeeper(first)
    bookkeeper.extend((operation, nccl, wqe))

    assert wqe.parent_refs[1:] == shared_refs
    shared_wqes = [
        entry.fact.ref.object_id
        for entry in bookkeeper.for_object(shared_refs[0], recursive=True)
        if isinstance(entry.fact, CreatedObjectRecord)
        and entry.fact.ref.kind is CreatedObjectKind.NETWORK_WQE
    ]
    assert shared_wqes == ["wqe:0", "wqe:8"]
    assert [
        entry.fact.ref.object_id
        for entry in bookkeeper.for_execution("execution-8")
        if isinstance(entry.fact, CreatedObjectRecord)
    ] == ["operation:8", "nccl:8", "wqe:8"]


def test_extend_is_atomic_when_a_later_fact_is_invalid():
    bookkeeper = RequestBookkeeper()
    first = CreatedObjectRecord(
        CreatedObjectRef(CreatedObjectKind.FRAMEWORK_REQUEST, "request:atomic"),
        ObjectOwner.FRAMEWORK,
        0,
        BookkeepingScope(
            correlation=OperationCorrelation(request_ids=("request:atomic",))
        ),
    )
    duplicate = replace(first, created_at_ps=1)
    with pytest.raises(ValueError, match="duplicate object ID"):
        bookkeeper.extend((first, duplicate))
    assert bookkeeper.snapshot() == BookkeepingLedger()
