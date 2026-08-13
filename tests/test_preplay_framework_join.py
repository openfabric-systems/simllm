"""Version 2 framework-capture join, its KV reconciliation, and its boundaries."""

from __future__ import annotations

import copy
import hashlib

import pytest

from simllm.adapters.vllm import ReplayTokenSource
from simllm.core import (
    CreatedObjectKind,
    CreatedObjectRecord,
    ObjectOwner,
    RequestBookkeeper,
)
from simllm.preplay import (
    FRAMEWORK_KV_RECONCILIATION_SCHEMA,
    FRAMEWORK_PREPLAY_TRACE_SCHEMA,
    PREPLAY_REPLAY_RUN_SCHEMA,
    PREPLAY_TRACE_SCHEMA,
    ForwardPhase,
    ForwardTokenTrace,
    FrameworkPreplayTrace,
    FrameworkRequestTrace,
    FrameworkTraceProvenance,
    KvAgreement,
    KvCacheEvent,
    KvDefectCode,
    KvEventKind,
    LayerRouting,
    ObservedLayerDispatch,
    ObservedTokenDispatch,
    PromptFormat,
    RequestArrival,
    RequestTrace,
    SamplingConfig,
    StopReason,
    TraceProvenance,
    framework_kv_reconciliation_from_json,
    framework_kv_reconciliation_to_json,
    join_framework_arrivals,
    join_preplay_arrivals,
    preplay_replay_run_from_json,
    preplay_replay_run_to_json,
    project_preplay_routing,
    read_framework_kv_reconciliation,
    write_framework_kv_reconciliation,
    write_framework_preplay_trace,
    write_preplay_trace,
)

TOP_K = 2
EXPERT_COUNT = 4
LAYERS = (0,)
PROMPTS = {"f0": (11, 12), "f1": (31,)}
OUTPUTS = {"f0": (21, 22, 23), "f1": (41,)}
ARRIVALS_PS = {"f0": 1_000, "f1": 4_000}


def _provenance() -> FrameworkTraceProvenance:
    return FrameworkTraceProvenance(
        model_id="test/framework-model",
        model_revision="test-revision",
        model_class="TestMoeForCausalLM",
        dtype="float32",
        tokenizer_sha256="b" * 64,
        sampling=SamplingConfig.greedy(),
        capture_host="test-host",
        runner="test-framework-runner",
        framework="test-framework",
        framework_version="0.0.1",
        observed_source="test-observed",
        authored_against_source="test-authored",
        torch_version="2.11.0",
        device="cpu",
        torch_num_threads=1,
        engine_seed=0,
        eos_token_id=0,
        top_k=TOP_K,
        expert_count=EXPERT_COUNT,
        moe_layer_indices=LAYERS,
        kv_page_size=1,
        kv_token_capacity=16,
        dispatch_layer_mapping="framework-layer-id",
    )


def _dispatch(phase: ForwardPhase, token_index: int, token_id: int) -> ObservedTokenDispatch:
    first = token_id % EXPERT_COUNT
    second = (first + 1) % EXPERT_COUNT
    return ObservedTokenDispatch(
        phase=phase,
        token_index=token_index,
        token_id=token_id,
        routing=(ObservedLayerDispatch(layer_index=0, expert_ids=(first, second)),),
    )


def _framework_request(request_id: str) -> FrameworkRequestTrace:
    prompt = PROMPTS[request_id]
    output = OUTPUTS[request_id]
    return FrameworkRequestTrace(
        request_id=request_id,
        prompt_sha256=hashlib.sha256(request_id.encode()).hexdigest(),
        prompt_format=PromptFormat.TEXT,
        input_token_ids=prompt,
        max_new_tokens=len(output),
        stop_strings=(),
        output_text=request_id,
        output_token_ids=output,
        output_length=len(output),
        stop_reason=StopReason.LENGTH_CAP,
        matched_stop_string=None,
        framework_cached_tokens=0,
        framework_preemption_count=0,
        prefill_dispatch=tuple(
            _dispatch(ForwardPhase.PREFILL, index, token_id)
            for index, token_id in enumerate(prompt)
        ),
        decode_dispatch=tuple(
            _dispatch(ForwardPhase.DECODE, index, token_id)
            for index, token_id in enumerate(output[:-1])
        ),
    )


def _exact_events() -> tuple[KvCacheEvent, ...]:
    """One prefill allocation plus one per decode forward, per request."""

    events: list[KvCacheEvent] = []
    slot = 0
    for request_id in ("f0", "f1"):
        prompt = len(PROMPTS[request_id])
        events.append(
            KvCacheEvent(
                sequence=len(events),
                kind=KvEventKind.ALLOCATION,
                request_id=request_id,
                framework_step=None,
                token_count=prompt,
                token_slot_ids=tuple(range(slot, slot + prompt)),
            )
        )
        slot += prompt
        for step in range(len(OUTPUTS[request_id]) - 1):
            events.append(
                KvCacheEvent(
                    sequence=len(events),
                    kind=KvEventKind.ALLOCATION,
                    request_id=request_id,
                    framework_step=step + 1,
                    token_count=1,
                    token_slot_ids=(slot,),
                )
            )
            slot += 1
    return tuple(events)


def _trace(events: tuple[KvCacheEvent, ...] | None = None) -> FrameworkPreplayTrace:
    return FrameworkPreplayTrace(
        provenance=_provenance(),
        requests=(_framework_request("f0"), _framework_request("f1")),
        kv_events=_exact_events() if events is None else events,
    )


def _resequenced(events: tuple[KvCacheEvent, ...]) -> tuple[KvCacheEvent, ...]:
    return tuple(
        KvCacheEvent(
            sequence=index,
            kind=event.kind,
            request_id=event.request_id,
            framework_step=event.framework_step,
            token_count=event.token_count,
            block_ids=event.block_ids,
            token_slot_ids=event.token_slot_ids,
            reason=event.reason,
        )
        for index, event in enumerate(events)
    )


def _write(tmp_path, trace: FrameworkPreplayTrace, name: str = "trace.jsonl"):
    return write_framework_preplay_trace(tmp_path / name, trace)


def _arrivals() -> tuple[RequestArrival, ...]:
    return tuple(
        RequestArrival(request_id=request_id, arrived_at_ps=arrived)
        for request_id, arrived in ARRIVALS_PS.items()
    )


def _v1_trace_path(tmp_path):
    provenance = TraceProvenance(
        model_id="test/model",
        model_revision="test-revision",
        model_class="TestMoeForCausalLM",
        dtype="float32",
        tokenizer_sha256="a" * 64,
        sampling=SamplingConfig.greedy(),
        capture_host="test-host",
        runner="test-runner",
        transformers_version="5.14.1",
        torch_version="2.11.0",
        device="cpu",
        torch_num_threads=1,
        eos_token_id=0,
        top_k=1,
        expert_count=1,
        moe_layer_indices=(0,),
    )
    request = RequestTrace(
        request_id="f0",
        prompt_sha256="c" * 64,
        prompt_format=PromptFormat.TEXT,
        input_token_ids=(10,),
        max_new_tokens=1,
        stop_strings=(),
        output_text="f0",
        output_token_ids=(77,),
        stop_reason=StopReason.LENGTH_CAP,
        matched_stop_string=None,
        prefill_tokens=(
            ForwardTokenTrace(
                phase=ForwardPhase.PREFILL,
                token_index=0,
                token_id=10,
                routing=(LayerRouting(layer_index=0, expert_ids=(0,), gate_weights=(1.0,)),),
            ),
        ),
        decode_tokens=(),
    )
    return write_preplay_trace(tmp_path / "v1.jsonl", provenance, (request,))


def _objects(bookkeeper: RequestBookkeeper) -> list[CreatedObjectRecord]:
    return [
        entry.fact
        for entry in bookkeeper.snapshot().entries
        if isinstance(entry.fact, CreatedObjectRecord)
    ]


def test_join_binds_observed_outputs_to_the_replay_identities(tmp_path):
    trace_path = _write(tmp_path, _trace())
    bookkeeper = RequestBookkeeper()
    joined = join_framework_arrivals(_arrivals(), trace_path, bookkeeper)
    run = joined.run

    assert run.schema == PREPLAY_REPLAY_RUN_SCHEMA
    assert run.trace.schema == FRAMEWORK_PREPLAY_TRACE_SCHEMA
    assert run.trace.path == str(trace_path.resolve())
    assert run.trace.sha256 == hashlib.sha256(trace_path.read_bytes()).hexdigest()
    assert [request.request_id for request in run.requests] == ["f0", "f1"]
    assert [request.arrived_at_ps for request in run.requests] == [1_000, 4_000]
    assert [request.output_token_ids for request in run.requests] == [
        OUTPUTS["f0"],
        OUTPUTS["f1"],
    ]
    assert [request.output_length for request in run.requests] == [3, 1]
    assert {request.stop_reason for request in run.requests} == {StopReason.LENGTH_CAP}
    for request in run.requests:
        assert request.routing_reference.trace_schema == FRAMEWORK_PREPLAY_TRACE_SCHEMA
        assert request.routing_reference.trace_sha256 == run.trace.sha256
        assert request.routing_reference.request_id == request.request_id

    objects = _objects(bookkeeper)
    assert len(objects) == 2
    for request, record in zip(run.requests, objects, strict=True):
        metadata = dict(record.metadata)
        assert record.ref.kind is CreatedObjectKind.FRAMEWORK_REQUEST
        assert record.ref.object_id == request.bookkeeping_object_id
        assert record.owner is ObjectOwner.FRAMEWORK
        assert record.created_at_ps == request.arrived_at_ps
        assert record.scope.correlation.request_ids == (request.request_id,)
        assert metadata["preplay_trace_schema"] == FRAMEWORK_PREPLAY_TRACE_SCHEMA


def test_the_replay_seam_accepts_a_version_2_joined_run(tmp_path):
    trace_path = _write(tmp_path, _trace())
    joined = join_framework_arrivals(_arrivals(), trace_path, RequestBookkeeper())
    source = ReplayTokenSource(joined.run, max_model_len=64)
    assert source.trace_sha256 == joined.run.trace.sha256
    assert source.request("f0").output_token_ids == OUTPUTS["f0"]
    assert source.request("f1").output_token_ids == OUTPUTS["f1"]
    with pytest.raises(ValueError, match="beyond max_model_len"):
        ReplayTokenSource(joined.run, max_model_len=3)


def test_each_join_refuses_the_other_trace_schema(tmp_path):
    v2_path = _write(tmp_path, _trace())
    v1_path = _v1_trace_path(tmp_path)
    with pytest.raises(ValueError, match="join_framework_arrivals"):
        join_preplay_arrivals(_arrivals()[:1], v2_path, RequestBookkeeper())
    with pytest.raises(ValueError, match="join_preplay_arrivals"):
        join_framework_arrivals(_arrivals()[:1], v1_path, RequestBookkeeper())


@pytest.mark.parametrize(
    "arrivals, message",
    [
        ((), "must not be empty"),
        (
            (
                RequestArrival(request_id="f0", arrived_at_ps=1),
                RequestArrival(request_id="f0", arrived_at_ps=2),
            ),
            "duplicate request identity",
        ),
        ((RequestArrival(request_id="f0", arrived_at_ps=-1),), "nonnegative"),
        ((RequestArrival(request_id="f0", arrived_at_ps=True),), "integer"),
        ((RequestArrival(request_id="ghost", arrived_at_ps=1),), "missing"),
    ],
)
def test_invalid_framework_join_is_atomic(tmp_path, arrivals, message):
    trace_path = _write(tmp_path, _trace())
    bookkeeper = RequestBookkeeper()
    before = bookkeeper.snapshot()
    with pytest.raises((TypeError, ValueError), match=message):
        join_framework_arrivals(arrivals, trace_path, bookkeeper)
    assert bookkeeper.snapshot() == before


def test_bookkeeping_collision_rejects_the_whole_second_join(tmp_path):
    trace_path = _write(tmp_path, _trace())
    bookkeeper = RequestBookkeeper()
    join_framework_arrivals(_arrivals(), trace_path, bookkeeper)
    before = bookkeeper.snapshot()
    with pytest.raises(ValueError, match="duplicate object ID"):
        join_framework_arrivals(_arrivals(), trace_path, bookkeeper)
    assert bookkeeper.snapshot() == before


def test_the_kv_event_stream_is_not_an_authority(tmp_path):
    with_events = _write(tmp_path, _trace(), "with-events.jsonl")
    without_events = _write(tmp_path, _trace(()), "without-events.jsonl")
    assert with_events.read_bytes() != without_events.read_bytes()

    joined = join_framework_arrivals(_arrivals(), with_events, RequestBookkeeper())
    stripped = join_framework_arrivals(_arrivals(), without_events, RequestBookkeeper())

    def _pinned(run):
        return [
            (
                request.request_id,
                request.arrived_at_ps,
                request.output_length,
                request.stop_reason,
                request.output_token_ids,
                request.routing_reference.request_id,
                request.bookkeeping_object_id,
            )
            for request in run.requests
        ]

    assert _pinned(joined.run) == _pinned(stripped.run)
    assert joined.kv.observed is True
    assert stripped.kv.observed is False
    assert stripped.kv.event_count == 0
    assert stripped.kv.defects == ()
    assert all(
        request.allocated_token_count == 0 for request in stripped.kv.requests
    )


def test_reconciliation_reports_exact_allocation_against_forward_passes(tmp_path):
    trace_path = _write(tmp_path, _trace())
    joined = join_framework_arrivals(_arrivals(), trace_path, RequestBookkeeper())
    kv = joined.kv

    assert kv.schema == FRAMEWORK_KV_RECONCILIATION_SCHEMA
    assert kv.trace_sha256 == joined.run.trace.sha256
    assert kv.defects == ()
    assert kv.unattributed_event_count == 0
    assert kv.unjoined_event_count == 0
    assert kv.peak_live_token_count == 5
    assert kv.final_live_token_count == 5
    f0 = kv.by_request_id("f0")
    assert (f0.forwarded_token_count, f0.allocated_token_count) == (4, 4)
    assert f0.agreement is KvAgreement.EXACT
    f1 = kv.by_request_id("f1")
    assert (f1.forwarded_token_count, f1.allocated_token_count) == (1, 1)
    assert f1.agreement is KvAgreement.EXACT


def test_a_prefix_hit_is_an_admissible_allocation_shortfall(tmp_path):
    request = _framework_request("f0")
    hit = KvCacheEvent(
        sequence=0,
        kind=KvEventKind.PREFIX_HIT,
        request_id="f0",
        framework_step=None,
        token_count=1,
    )
    allocations = tuple(
        event
        for event in _exact_events()
        if event.request_id == "f0" and event.kind is KvEventKind.ALLOCATION
    )
    shortened = KvCacheEvent(
        sequence=0,
        kind=KvEventKind.ALLOCATION,
        request_id="f0",
        framework_step=None,
        token_count=1,
        token_slot_ids=(0,),
    )
    events = _resequenced((hit, shortened, *allocations[1:]))
    trace = FrameworkPreplayTrace(
        provenance=_provenance(),
        requests=(
            FrameworkRequestTrace(
                **{
                    **{
                        field: getattr(request, field)
                        for field in request.__dataclass_fields__
                    },
                    "framework_cached_tokens": 1,
                }
            ),
        ),
        kv_events=events,
    )
    trace_path = _write(tmp_path, trace, "prefix-hit.jsonl")
    joined = join_framework_arrivals(
        (RequestArrival(request_id="f0", arrived_at_ps=0),),
        trace_path,
        RequestBookkeeper(),
    )
    reconciled = joined.kv.by_request_id("f0")
    assert reconciled.prefix_hit_token_count == 1
    assert reconciled.allocated_token_count == 3
    assert reconciled.allocation_surplus == 0
    assert reconciled.agreement is KvAgreement.EXACT
    assert joined.kv.defects == ()


def test_a_preempted_request_may_allocate_more_than_its_forward_passes(tmp_path):
    request = _framework_request("f0")
    preemption = KvCacheEvent(
        sequence=0,
        kind=KvEventKind.PREEMPTION,
        request_id="f0",
        framework_step=1,
        token_count=0,
        reason="kv pressure",
    )
    extra = KvCacheEvent(
        sequence=0,
        kind=KvEventKind.ALLOCATION,
        request_id="f0",
        framework_step=2,
        token_count=2,
        token_slot_ids=(90, 91),
    )
    allocations = tuple(
        event
        for event in _exact_events()
        if event.request_id == "f0" and event.kind is KvEventKind.ALLOCATION
    )
    events = _resequenced((*allocations, preemption, extra))
    trace = FrameworkPreplayTrace(
        provenance=_provenance(),
        requests=(
            FrameworkRequestTrace(
                **{
                    **{
                        field: getattr(request, field)
                        for field in request.__dataclass_fields__
                    },
                    "framework_preemption_count": 1,
                }
            ),
        ),
        kv_events=events,
    )
    trace_path = _write(tmp_path, trace, "preempted.jsonl")
    joined = join_framework_arrivals(
        (RequestArrival(request_id="f0", arrived_at_ps=0),),
        trace_path,
        RequestBookkeeper(),
    )
    reconciled = joined.kv.by_request_id("f0")
    assert reconciled.preemption_event_count == 1
    assert reconciled.allocation_surplus == 2
    assert reconciled.agreement is KvAgreement.RECOMPUTE_SURPLUS
    assert joined.kv.defects == ()


@pytest.mark.parametrize(
    "mutate, code",
    [
        ("extra_allocation", KvDefectCode.UNEXPLAINED_ALLOCATION_TOTAL),
        ("no_allocation", KvDefectCode.MISSING_ALLOCATION),
        ("unreported_preemption", KvDefectCode.PREEMPTION_COUNT_DISAGREEMENT),
        ("unreported_prefix_hit", KvDefectCode.PREFIX_HIT_TOKEN_DISAGREEMENT),
        ("over_release", KvDefectCode.NEGATIVE_OCCUPANCY),
    ],
)
def test_reconciliation_reports_a_capture_that_contradicts_its_record(
    tmp_path, mutate, code
):
    events = list(_exact_events())
    if mutate == "extra_allocation":
        events.append(
            KvCacheEvent(
                sequence=0,
                kind=KvEventKind.ALLOCATION,
                request_id="f0",
                framework_step=9,
                token_count=3,
                token_slot_ids=(80, 81, 82),
            )
        )
    elif mutate == "no_allocation":
        events = [event for event in events if event.request_id != "f1"]
    elif mutate == "unreported_preemption":
        events.append(
            KvCacheEvent(
                sequence=0,
                kind=KvEventKind.PREEMPTION,
                request_id="f0",
                framework_step=1,
                token_count=0,
                reason="kv pressure",
            )
        )
    elif mutate == "unreported_prefix_hit":
        events.insert(
            0,
            KvCacheEvent(
                sequence=0,
                kind=KvEventKind.PREFIX_HIT,
                request_id="f0",
                framework_step=None,
                token_count=2,
            ),
        )
    else:
        events.append(
            KvCacheEvent(
                sequence=0,
                kind=KvEventKind.RELEASE,
                request_id="f0",
                framework_step=9,
                token_count=99,
                token_slot_ids=(70,),
            )
        )
    trace_path = _write(tmp_path, _trace(_resequenced(tuple(events))), f"{mutate}.jsonl")
    joined = join_framework_arrivals(_arrivals(), trace_path, RequestBookkeeper())
    assert code in {defect.code for defect in joined.kv.defects}
    assert [request.request_id for request in joined.run.requests] == ["f0", "f1"]


def test_reconciliation_round_trip_and_rejections(tmp_path):
    trace_path = _write(tmp_path, _trace())
    joined = join_framework_arrivals(_arrivals(), trace_path, RequestBookkeeper())
    first = write_framework_kv_reconciliation(joined.kv, tmp_path / "kv-first.json")
    loaded = read_framework_kv_reconciliation(first)
    second = write_framework_kv_reconciliation(loaded, tmp_path / "kv-second.json")
    assert first.read_bytes() == second.read_bytes()
    assert loaded == joined.kv

    base = framework_kv_reconciliation_to_json(joined.kv)
    unknown = copy.deepcopy(base)
    unknown["unknown"] = True
    with pytest.raises(ValueError, match="unknown fields"):
        framework_kv_reconciliation_from_json(unknown)
    bad_schema = copy.deepcopy(base)
    bad_schema["schema"] = "simllm-preplay-kv-reconciliation-v2"
    with pytest.raises(ValueError, match="unsupported schema"):
        framework_kv_reconciliation_from_json(bad_schema)
    duplicate = copy.deepcopy(base)
    duplicate["requests"].append(copy.deepcopy(duplicate["requests"][0]))
    with pytest.raises(ValueError, match="duplicate request identity"):
        framework_kv_reconciliation_from_json(duplicate)
    bad_agreement = copy.deepcopy(base)
    bad_agreement["requests"][0]["agreement"] = "maybe"
    with pytest.raises(ValueError, match="unsupported value"):
        framework_kv_reconciliation_from_json(bad_agreement)


def test_replay_run_rejects_a_mixed_trace_schema(tmp_path):
    trace_path = _write(tmp_path, _trace())
    joined = join_framework_arrivals(_arrivals(), trace_path, RequestBookkeeper())
    payload = preplay_replay_run_to_json(joined.run)
    assert preplay_replay_run_from_json(copy.deepcopy(payload)) == joined.run
    mixed = copy.deepcopy(payload)
    mixed["requests"][0]["routing_reference"]["trace_schema"] = PREPLAY_TRACE_SCHEMA
    with pytest.raises(ValueError, match="must match run.trace.schema"):
        preplay_replay_run_from_json(mixed)


def test_routing_projection_follows_the_joined_request_order(tmp_path):
    trace_path = _write(tmp_path, _trace())
    forward = join_framework_arrivals(_arrivals(), trace_path, RequestBookkeeper())
    reversed_arrivals = tuple(reversed(_arrivals()))
    backward = join_framework_arrivals(
        reversed_arrivals, trace_path, RequestBookkeeper()
    )

    projected = project_preplay_routing(forward.run)
    assert projected.trace_schema == FRAMEWORK_PREPLAY_TRACE_SCHEMA
    assert projected.trace_sha256 == forward.run.trace.sha256
    assert [request.request_id for request in projected.requests] == ["f0", "f1"]
    assert [request.request_id for request in project_preplay_routing(backward.run).requests] == [
        "f1",
        "f0",
    ]

    source = _trace().by_request_id("f0")
    routed = projected.by_request_id("f0")
    assert routed.prompt_token_count == len(PROMPTS["f0"])
    assert routed.output_token_count == len(OUTPUTS["f0"])
    expected = tuple(
        (dispatch.phase, dispatch.token_index, dispatch.token_id, dispatch.routing[0].expert_ids)
        for dispatch in (*source.prefill_dispatch, *source.decode_dispatch)
    )
    assert (
        tuple(
            (token.phase, token.token_index, token.token_id, token.layers[0].expert_ids)
            for token in routed.tokens
        )
        == expected
    )


def test_a_joined_subset_projects_only_the_joined_requests(tmp_path):
    trace_path = _write(tmp_path, _trace())
    joined = join_framework_arrivals(
        (RequestArrival(request_id="f1", arrived_at_ps=0),),
        trace_path,
        RequestBookkeeper(),
    )
    assert [request.request_id for request in joined.run.requests] == ["f1"]
    assert [request.request_id for request in joined.kv.requests] == ["f1"]
    assert joined.kv.unjoined_event_count == 3
    projected = project_preplay_routing(joined.run)
    assert [request.request_id for request in projected.requests] == ["f1"]
