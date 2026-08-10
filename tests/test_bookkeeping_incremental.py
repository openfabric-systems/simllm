import random
from dataclasses import replace

import pytest

from simllm.core.bookkeeping import (
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
from simllm.core.execution import (
    CompletionEvent,
    EventPhase,
    OperationCorrelation,
    ResourceKind,
    ResourceRef,
)

_SEEDS = (7001, 7002, 7003, 7004, 7005, 7006)
_LENGTHS = (8, 32, 128)
_BATCH_WIDTHS = (1, 2, 7, 32)


def _record(
    kind: CreatedObjectKind,
    object_id: str,
    timestamp_ps: int,
    *,
    owner: ObjectOwner = ObjectOwner.DEVICE_RUNTIME,
    scope: BookkeepingScope | None = None,
    parents: tuple[CreatedObjectRef, ...] = (),
    metadata: tuple[tuple[str, str | int | float | bool], ...] = (),
) -> CreatedObjectRecord:
    return CreatedObjectRecord(
        CreatedObjectRef(kind, object_id),
        owner,
        timestamp_ps,
        scope or BookkeepingScope(),
        parent_refs=parents,
        metadata=metadata,
    )


def _valid_fact_stream(seed: int, length: int) -> tuple[object, ...]:
    rng = random.Random(seed)
    empty_scope = BookkeepingScope()
    sq = _record(CreatedObjectKind.SEND_QUEUE, "sq:shared", 0)
    rq = _record(CreatedObjectKind.RECEIVE_QUEUE, "rq:shared", 0)
    cq = _record(CreatedObjectKind.COMPLETION_QUEUE, "cq:shared", 0)
    qp = _record(
        CreatedObjectKind.DCQCN_QP,
        "qp:shared",
        1,
        owner=ObjectOwner.NETWORK_BACKEND,
        parents=(sq.ref, rq.ref, cq.ref),
    )
    link = _record(
        CreatedObjectKind.RNIC_CN_LINK_PAIR,
        "link:shared",
        1,
        owner=ObjectOwner.NETWORK_BACKEND,
        parents=(sq.ref, rq.ref, cq.ref),
    )
    facts: list[object] = [sq, rq, cq, qp, link]

    cycle = 0
    while len(facts) < length:
        timestamp = 100 + cycle * 32
        request_ids = (f"request:{cycle}:a", f"request:{cycle}:b")
        request_scope_a = BookkeepingScope(
            correlation=OperationCorrelation(request_ids=(request_ids[0],))
        )
        request_scope_b = BookkeepingScope(
            correlation=OperationCorrelation(request_ids=(request_ids[1],))
        )
        operation_scope = BookkeepingScope(
            correlation=OperationCorrelation(
                request_ids=request_ids,
                batch_id=f"batch:{cycle}",
                layer=cycle % 8,
                microbatch=cycle % 3,
                iteration=cycle // 3,
            ),
            step_index=cycle,
            execution_id=f"execution:{cycle}",
            operation_id=f"operation:{cycle}",
        )
        narrowed_ids = request_ids if rng.randrange(2) else (rng.choice(request_ids),)
        child_scope = replace(
            operation_scope,
            correlation=replace(operation_scope.correlation, request_ids=narrowed_ids),
        )
        request_a = _record(
            CreatedObjectKind.FRAMEWORK_REQUEST,
            f"request-object:{cycle}:a",
            timestamp,
            owner=ObjectOwner.FRAMEWORK,
            scope=request_scope_a,
        )
        request_b = _record(
            CreatedObjectKind.FRAMEWORK_REQUEST,
            f"request-object:{cycle}:b",
            timestamp,
            owner=ObjectOwner.FRAMEWORK,
            scope=request_scope_b,
        )
        operation = _record(
            CreatedObjectKind.EXECUTION_OPERATION,
            f"operation-object:{cycle}",
            timestamp + 2,
            owner=ObjectOwner.CORE,
            scope=operation_scope,
        )
        nccl = _record(
            CreatedObjectKind.NCCL_COMMAND,
            f"nccl:{cycle}",
            timestamp + 3,
            owner=ObjectOwner.NCCL,
            scope=child_scope,
            parents=(operation.ref,),
        )
        transport_mode = cycle % 3
        transport_refs: tuple[CreatedObjectRef, ...]
        metadata: tuple[tuple[str, str | int | float | bool], ...]
        if transport_mode == 0:
            transport_refs = ()
            metadata = (("bytes", 4096 + cycle), ("transport_kind", "none"))
        elif transport_mode == 1:
            transport_refs = (qp.ref,)
            metadata = (("bytes", 4096 + cycle),)
        else:
            transport_refs = (link.ref,)
            metadata = (("bytes", 4096 + cycle),)
        wqe = _record(
            CreatedObjectKind.NETWORK_WQE,
            f"wqe:{cycle}",
            timestamp + 4,
            scope=child_scope,
            parents=(nccl.ref, sq.ref, rq.ref, cq.ref, *transport_refs),
            metadata=metadata,
        )
        resource = ResourceRef(ResourceKind.NIC_SEND_QUEUE, sq.ref.object_id)
        event_args = (child_scope.execution_id, child_scope.operation_id)
        assert event_args[0] is not None
        assert event_args[1] is not None
        facts.extend(
            (
                request_a,
                request_b,
                operation,
                nccl,
                wqe,
                StageRecord(
                    ProcessingStage.NETWORK,
                    StagePhase.ENTERED,
                    timestamp + 5,
                    child_scope,
                    (wqe.ref,),
                ),
                CompletionEvent(
                    event_args[0],
                    event_args[1],
                    EventPhase.SUBMITTED,
                    timestamp + 6,
                    resource,
                    subject_object_id=wqe.ref.object_id,
                ),
                CompletionEvent(
                    event_args[0],
                    event_args[1],
                    EventPhase.QUEUED,
                    timestamp + 8,
                    resource,
                    subject_object_id=wqe.ref.object_id,
                ),
                CompletionEvent(
                    event_args[0],
                    event_args[1],
                    EventPhase.STARTED,
                    timestamp + 10,
                    resource,
                    subject_object_id=wqe.ref.object_id,
                ),
                CompletionEvent(
                    event_args[0],
                    event_args[1],
                    EventPhase.PROGRESS,
                    timestamp + 12,
                    resource,
                    completed_bytes=2048,
                    subject_object_id=wqe.ref.object_id,
                ),
                CompletionEvent(
                    event_args[0],
                    event_args[1],
                    EventPhase.COMPLETED,
                    timestamp + 14,
                    ResourceRef(ResourceKind.COMPLETION_QUEUE, cq.ref.object_id),
                    completed_bytes=4096 + cycle,
                    subject_object_id=wqe.ref.object_id,
                ),
                StageRecord(
                    ProcessingStage.NETWORK,
                    StagePhase.COMPLETED,
                    timestamp + 15,
                    child_scope,
                    (wqe.ref,),
                ),
                CompletionEvent(
                    event_args[0],
                    event_args[1],
                    EventPhase.COMPLETED,
                    timestamp + 16,
                ),
            )
        )
        cycle += 1

    stream = tuple(facts[:length])
    validate_bookkeeping_ledger(_ledger(stream))
    assert empty_scope == sq.scope
    return stream


def _ledger(facts: tuple[object, ...]) -> BookkeepingLedger:
    return BookkeepingLedger(
        tuple(BookkeepingEntry(index, fact) for index, fact in enumerate(facts))
    )


def _reference_candidate(
    ledger: BookkeepingLedger,
    facts: tuple[object, ...],
) -> tuple[BookkeepingLedger, tuple[BookkeepingEntry, ...]]:
    start = len(ledger.entries)
    additions = tuple(
        BookkeepingEntry(start + offset, fact) for offset, fact in enumerate(facts)
    )
    candidate = BookkeepingLedger((*ledger.entries, *additions))
    validate_bookkeeping_ledger(candidate)
    return candidate, additions


def _assert_append_equivalent(facts: tuple[object, ...]) -> None:
    reference = BookkeepingLedger()
    bookkeeper = RequestBookkeeper()
    for fact in facts:
        before = bookkeeper.snapshot()
        try:
            candidate, additions = _reference_candidate(reference, (fact,))
        except (TypeError, ValueError) as reference_error:
            with pytest.raises(type(reference_error)):
                bookkeeper.append(fact)
            assert bookkeeper.snapshot() == before == reference
        else:
            assert bookkeeper.append(fact) == additions[0]
            reference = candidate
            assert bookkeeper.snapshot() == reference


def _assert_extend_equivalent(facts: tuple[object, ...], seed: int) -> None:
    reference = BookkeepingLedger()
    bookkeeper = RequestBookkeeper()
    candidate, additions = _reference_candidate(reference, ())
    assert bookkeeper.extend(()) == additions == ()
    assert candidate == bookkeeper.snapshot()

    rng = random.Random(seed)
    position = 0
    while position < len(facts):
        width = rng.choice(_BATCH_WIDTHS)
        batch = facts[position : position + width]
        before = bookkeeper.snapshot()
        try:
            candidate, additions = _reference_candidate(reference, batch)
        except (TypeError, ValueError) as reference_error:
            with pytest.raises(type(reference_error)):
                bookkeeper.extend(batch)
            assert bookkeeper.snapshot() == before == reference
        else:
            assert bookkeeper.extend(batch) == additions
            reference = candidate
            assert bookkeeper.snapshot() == reference
        position += len(batch)


def _pick_index(rng: random.Random, facts: list[object], predicate) -> int:
    matches = [index for index, fact in enumerate(facts) if predicate(fact)]
    assert matches
    return rng.choice(matches)


def _replace_timestamp(fact: object, timestamp: object) -> object:
    if isinstance(fact, CreatedObjectRecord):
        return replace(fact, created_at_ps=timestamp)
    return replace(fact, timestamp_ps=timestamp)


def _mutated_stream(seed: int, mutation: str) -> tuple[object, ...]:
    rng = random.Random(seed * 1009 + sum(ord(character) for character in mutation))
    facts = list(_valid_fact_stream(seed, 128))
    object_records = {
        fact.ref.object_id: fact for fact in facts if isinstance(fact, CreatedObjectRecord)
    }

    if mutation == "unsupported-fact":
        facts.insert(rng.randrange(len(facts) + 1), object())
    elif mutation in {"boolean-timestamp", "negative-timestamp"}:
        index = rng.randrange(len(facts))
        timestamp = True if mutation == "boolean-timestamp" else -1
        facts[index] = _replace_timestamp(facts[index], timestamp)
    elif mutation == "invalid-ref-kind":
        index = _pick_index(rng, facts, lambda fact: isinstance(fact, CreatedObjectRecord))
        fact = facts[index]
        facts[index] = replace(fact, ref=replace(fact.ref, kind="not-a-kind"))
    elif mutation == "invalid-owner":
        index = _pick_index(rng, facts, lambda fact: isinstance(fact, CreatedObjectRecord))
        facts[index] = replace(facts[index], owner="not-an-owner")
    elif mutation == "invalid-scope":
        index = _pick_index(
            rng,
            facts,
            lambda fact: isinstance(fact, (CreatedObjectRecord, StageRecord)),
        )
        facts[index] = replace(facts[index], scope="not-a-scope")
    elif mutation == "correlation-list":
        index = _pick_index(
            rng,
            facts,
            lambda fact: isinstance(fact, (CreatedObjectRecord, StageRecord)),
        )
        fact = facts[index]
        correlation = replace(fact.scope.correlation, request_ids=["request:list"])
        facts[index] = replace(fact, scope=replace(fact.scope, correlation=correlation))
    elif mutation == "correlation-duplicate":
        index = _pick_index(
            rng,
            facts,
            lambda fact: isinstance(fact, (CreatedObjectRecord, StageRecord)),
        )
        fact = facts[index]
        correlation = replace(fact.scope.correlation, request_ids=("duplicate", "duplicate"))
        facts[index] = replace(fact, scope=replace(fact.scope, correlation=correlation))
    elif mutation == "boolean-scope-integer":
        index = _pick_index(
            rng,
            facts,
            lambda fact: isinstance(fact, (CreatedObjectRecord, StageRecord)),
        )
        fact = facts[index]
        facts[index] = replace(fact, scope=replace(fact.scope, step_index=True))
    elif mutation == "operation-without-execution":
        index = _pick_index(rng, facts, lambda fact: isinstance(fact, StageRecord))
        fact = facts[index]
        facts[index] = replace(
            fact,
            scope=replace(fact.scope, execution_id=None, operation_id="orphan"),
        )
    elif mutation == "metadata-list":
        index = _pick_index(rng, facts, lambda fact: isinstance(fact, CreatedObjectRecord))
        facts[index] = replace(facts[index], metadata=[])
    elif mutation == "metadata-invalid-scalar":
        index = _pick_index(rng, facts, lambda fact: isinstance(fact, CreatedObjectRecord))
        facts[index] = replace(facts[index], metadata=(("bad", None),))
    elif mutation == "metadata-duplicate":
        index = _pick_index(rng, facts, lambda fact: isinstance(fact, CreatedObjectRecord))
        facts[index] = replace(facts[index], metadata=(("same", 1), ("same", 2)))
    elif mutation == "parent-list":
        index = _pick_index(
            rng,
            facts,
            lambda fact: isinstance(fact, CreatedObjectRecord) and fact.parent_refs,
        )
        facts[index] = replace(facts[index], parent_refs=list(facts[index].parent_refs))
    elif mutation == "duplicate-object":
        indexes = [
            index for index, fact in enumerate(facts) if isinstance(fact, CreatedObjectRecord)
        ]
        first, second = sorted(rng.sample(indexes, 2))
        first_fact = facts[first]
        second_fact = facts[second]
        facts[second] = replace(
            second_fact,
            ref=replace(second_fact.ref, object_id=first_fact.ref.object_id),
        )
    elif mutation == "self-parent":
        index = _pick_index(rng, facts, lambda fact: isinstance(fact, CreatedObjectRecord))
        fact = facts[index]
        facts[index] = replace(fact, parent_refs=(fact.ref,))
    elif mutation in {"unknown-parent", "wrong-typed-parent"}:
        index = _pick_index(
            rng,
            facts,
            lambda fact: isinstance(fact, CreatedObjectRecord) and fact.parent_refs,
        )
        fact = facts[index]
        parent = fact.parent_refs[0]
        replacement = replace(parent, object_id="missing:parent")
        if mutation == "wrong-typed-parent":
            replacement = replace(parent, kind=CreatedObjectKind.FRAMEWORK_REQUEST)
        facts[index] = replace(fact, parent_refs=(replacement, *fact.parent_refs[1:]))
    elif mutation == "child-predates-parent":
        index = _pick_index(
            rng,
            facts,
            lambda fact: isinstance(fact, CreatedObjectRecord)
            and fact.parent_refs
            and object_records[fact.parent_refs[0].object_id].created_at_ps > 0,
        )
        fact = facts[index]
        parent = object_records[fact.parent_refs[0].object_id]
        facts[index] = replace(fact, created_at_ps=parent.created_at_ps - 1)
    elif mutation == "request-introduction":
        index = _pick_index(
            rng,
            facts,
            lambda fact: isinstance(fact, CreatedObjectRecord)
            and fact.ref.kind is CreatedObjectKind.NCCL_COMMAND,
        )
        fact = facts[index]
        correlation = replace(
            fact.scope.correlation,
            request_ids=(*fact.scope.correlation.request_ids, "request:introduced"),
        )
        facts[index] = replace(fact, scope=replace(fact.scope, correlation=correlation))
    elif mutation == "causal-parent-disagreement":
        index = _pick_index(
            rng,
            facts,
            lambda fact: isinstance(fact, CreatedObjectRecord)
            and fact.ref.kind is CreatedObjectKind.NETWORK_WQE
            and any(
                isinstance(prior, CreatedObjectRecord)
                and prior.ref.kind is CreatedObjectKind.NCCL_COMMAND
                and prior.ref not in fact.parent_refs
                for prior in facts[: facts.index(fact)]
            ),
        )
        fact = facts[index]
        other = next(
            prior
            for prior in facts[:index]
            if isinstance(prior, CreatedObjectRecord)
            and prior.ref.kind is CreatedObjectKind.NCCL_COMMAND
            and prior.ref not in fact.parent_refs
        )
        facts[index] = replace(fact, parent_refs=(*fact.parent_refs, other.ref))
    elif mutation == "wqe-missing-queue":
        index = _pick_index(
            rng,
            facts,
            lambda fact: isinstance(fact, CreatedObjectRecord)
            and fact.ref.kind is CreatedObjectKind.NETWORK_WQE,
        )
        fact = facts[index]
        missing_kind = rng.choice(
            (
                CreatedObjectKind.SEND_QUEUE,
                CreatedObjectKind.RECEIVE_QUEUE,
                CreatedObjectKind.COMPLETION_QUEUE,
            )
        )
        facts[index] = replace(
            fact,
            parent_refs=tuple(parent for parent in fact.parent_refs if parent.kind is not missing_kind),
        )
    elif mutation == "wqe-multiple-transports":
        index = _pick_index(
            rng,
            facts,
            lambda fact: isinstance(fact, CreatedObjectRecord)
            and fact.ref.kind is CreatedObjectKind.NETWORK_WQE
            and any(
                parent.kind
                in {CreatedObjectKind.DCQCN_QP, CreatedObjectKind.RNIC_CN_LINK_PAIR}
                for parent in fact.parent_refs
            ),
        )
        fact = facts[index]
        alternate = next(
            ref
            for ref in (
                CreatedObjectRef(CreatedObjectKind.DCQCN_QP, "qp:shared"),
                CreatedObjectRef(CreatedObjectKind.RNIC_CN_LINK_PAIR, "link:shared"),
            )
            if ref not in fact.parent_refs
        )
        facts[index] = replace(fact, parent_refs=(*fact.parent_refs, alternate))
    elif mutation == "transport-free-implicit":
        index = _pick_index(
            rng,
            facts,
            lambda fact: isinstance(fact, CreatedObjectRecord)
            and fact.ref.kind is CreatedObjectKind.NETWORK_WQE
            and not any(
                parent.kind
                in {CreatedObjectKind.DCQCN_QP, CreatedObjectKind.RNIC_CN_LINK_PAIR}
                for parent in fact.parent_refs
            ),
        )
        fact = facts[index]
        facts[index] = replace(
            fact,
            metadata=tuple(item for item in fact.metadata if item[0] != "transport_kind"),
        )
    elif mutation == "physical-transport-none":
        index = _pick_index(
            rng,
            facts,
            lambda fact: isinstance(fact, CreatedObjectRecord)
            and fact.ref.kind is CreatedObjectKind.NETWORK_WQE
            and any(
                parent.kind
                in {CreatedObjectKind.DCQCN_QP, CreatedObjectKind.RNIC_CN_LINK_PAIR}
                for parent in fact.parent_refs
            ),
        )
        fact = facts[index]
        facts[index] = replace(fact, metadata=((*fact.metadata, ("transport_kind", "none"))))
    elif mutation in {"invalid-stage", "invalid-stage-phase"}:
        index = _pick_index(rng, facts, lambda fact: isinstance(fact, StageRecord))
        field = "stage" if mutation == "invalid-stage" else "phase"
        facts[index] = replace(facts[index], **{field: "invalid"})
    elif mutation == "stage-ref-list":
        index = _pick_index(rng, facts, lambda fact: isinstance(fact, StageRecord))
        facts[index] = replace(facts[index], object_refs=list(facts[index].object_refs))
    elif mutation == "stage-unknown-object":
        index = _pick_index(rng, facts, lambda fact: isinstance(fact, StageRecord))
        fact = facts[index]
        facts[index] = replace(
            fact,
            object_refs=(CreatedObjectRef(CreatedObjectKind.NETWORK_WQE, "wqe:missing"),),
        )
    elif mutation == "stage-predates-object":
        index = _pick_index(rng, facts, lambda fact: isinstance(fact, StageRecord))
        fact = facts[index]
        subject = object_records[fact.object_refs[0].object_id]
        facts[index] = replace(fact, timestamp_ps=subject.created_at_ps - 1)
    elif mutation in {
        "invalid-completion-phase",
        "boolean-completed-bytes",
        "invalid-resource",
        "invalid-resource-kind",
    }:
        index = _pick_index(rng, facts, lambda fact: isinstance(fact, CompletionEvent))
        fact = facts[index]
        if mutation == "invalid-completion-phase":
            facts[index] = replace(fact, phase="invalid")
        elif mutation == "boolean-completed-bytes":
            facts[index] = replace(fact, completed_bytes=True)
        elif mutation == "invalid-resource":
            facts[index] = replace(fact, resource="invalid")
        else:
            facts[index] = replace(fact, resource=ResourceRef("invalid", "resource"))
    elif mutation == "unknown-subject":
        index = _pick_index(rng, facts, lambda fact: isinstance(fact, CompletionEvent))
        facts[index] = replace(facts[index], subject_object_id="wqe:missing")
    elif mutation in {"subject-execution-mismatch", "subject-operation-mismatch"}:
        index = _pick_index(
            rng,
            facts,
            lambda fact: isinstance(fact, CompletionEvent) and fact.subject_object_id is not None,
        )
        field = "execution_id" if mutation == "subject-execution-mismatch" else "operation_id"
        facts[index] = replace(facts[index], **{field: "mismatch"})
    elif mutation == "completion-predates-subject":
        index = _pick_index(
            rng,
            facts,
            lambda fact: isinstance(fact, CompletionEvent) and fact.subject_object_id is not None,
        )
        fact = facts[index]
        subject = object_records[fact.subject_object_id]
        facts[index] = replace(fact, timestamp_ps=subject.created_at_ps - 1)
    elif mutation == "decreasing-subject-time":
        index = _pick_index(
            rng,
            facts,
            lambda fact: isinstance(fact, CompletionEvent)
            and fact.phase is EventPhase.QUEUED
            and fact.subject_object_id is not None,
        )
        facts[index] = replace(facts[index], timestamp_ps=facts[index].timestamp_ps - 3)
    elif mutation == "wrong-completion-queue":
        index = _pick_index(
            rng,
            facts,
            lambda fact: isinstance(fact, CompletionEvent)
            and fact.phase is EventPhase.COMPLETED
            and fact.subject_object_id is not None,
        )
        facts[index] = replace(
            facts[index],
            resource=ResourceRef(ResourceKind.COMPLETION_QUEUE, "cq:wrong"),
        )
    elif mutation in {"duplicate-wqe-completion", "event-after-wqe-completion"}:
        index = _pick_index(
            rng,
            facts,
            lambda fact: isinstance(fact, CompletionEvent)
            and fact.phase is EventPhase.COMPLETED
            and fact.subject_object_id is not None,
        )
        completed = facts[index]
        phase = (
            EventPhase.COMPLETED
            if mutation == "duplicate-wqe-completion"
            else EventPhase.PROGRESS
        )
        facts.insert(index + 1, replace(completed, phase=phase, timestamp_ps=completed.timestamp_ps + 1))
    else:
        raise AssertionError(f"unknown mutation {mutation}")

    stream = tuple(facts)
    with pytest.raises((TypeError, ValueError)):
        validate_bookkeeping_ledger(_ledger(stream))
    return stream


_MUTATIONS = (
    "unsupported-fact",
    "boolean-timestamp",
    "negative-timestamp",
    "invalid-ref-kind",
    "invalid-owner",
    "invalid-scope",
    "correlation-list",
    "correlation-duplicate",
    "boolean-scope-integer",
    "operation-without-execution",
    "metadata-list",
    "metadata-invalid-scalar",
    "metadata-duplicate",
    "parent-list",
    "duplicate-object",
    "self-parent",
    "unknown-parent",
    "wrong-typed-parent",
    "child-predates-parent",
    "request-introduction",
    "causal-parent-disagreement",
    "wqe-missing-queue",
    "wqe-multiple-transports",
    "transport-free-implicit",
    "physical-transport-none",
    "invalid-stage",
    "invalid-stage-phase",
    "stage-ref-list",
    "stage-unknown-object",
    "stage-predates-object",
    "invalid-completion-phase",
    "boolean-completed-bytes",
    "invalid-resource",
    "invalid-resource-kind",
    "unknown-subject",
    "subject-execution-mismatch",
    "subject-operation-mismatch",
    "completion-predates-subject",
    "decreasing-subject-time",
    "wrong-completion-queue",
    "duplicate-wqe-completion",
    "event-after-wqe-completion",
)


@pytest.mark.parametrize("seed", _SEEDS)
@pytest.mark.parametrize("length", _LENGTHS)
def test_seeded_valid_append_and_extend_match_full_validator(seed: int, length: int):
    facts = _valid_fact_stream(seed, length)
    _assert_append_equivalent(facts)
    _assert_extend_equivalent(facts, seed + length)


@pytest.mark.parametrize("seed", _SEEDS)
def test_seeded_invalid_injections_match_full_validator(seed: int):
    for mutation in _MUTATIONS:
        facts = _mutated_stream(seed, mutation)
        _assert_append_equivalent(facts)
        _assert_extend_equivalent(facts, seed + len(mutation))


def test_initial_boolean_sequence_rejection_stays_on_reference_validator():
    stage = StageRecord(
        ProcessingStage.REQUEST,
        StagePhase.ENTERED,
        0,
        BookkeepingScope(),
    )
    ledger = BookkeepingLedger(
        (BookkeepingEntry(0, stage), BookkeepingEntry(True, stage))
    )
    with pytest.raises(ValueError, match="sequence: must be an integer"):
        RequestBookkeeper(ledger)
