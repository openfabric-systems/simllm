"""Arrival join tests for the versioned pre-play replay run."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest

from simllm.core import (
    CreatedObjectKind,
    CreatedObjectRecord,
    ObjectOwner,
    RequestBookkeeper,
)
from simllm.preplay import (
    PREPLAY_REPLAY_RUN_SCHEMA,
    PREPLAY_ROUTING_REFERENCE_SCHEMA,
    ForwardPhase,
    ForwardTokenTrace,
    LayerRouting,
    PreplayTrace,
    PromptFormat,
    RequestArrival,
    RequestTrace,
    SamplingConfig,
    StopReason,
    TraceProvenance,
    join_preplay_arrivals,
    preplay_replay_run_from_json,
    preplay_replay_run_to_json,
    read_preplay_replay_run,
    write_preplay_replay_run,
    write_preplay_trace,
)


def _routing(token_id: int, phase: ForwardPhase, token_index: int) -> ForwardTokenTrace:
    return ForwardTokenTrace(
        phase=phase,
        token_index=token_index,
        token_id=token_id,
        routing=(LayerRouting(layer_index=0, expert_ids=(0,), gate_weights=(1.0,)),),
    )


def _request(
    request_id: str,
    output_token_ids: tuple[int, ...],
    stop_reason: StopReason,
) -> RequestTrace:
    return RequestTrace(
        request_id=request_id,
        prompt_sha256=hashlib.sha256(request_id.encode()).hexdigest(),
        prompt_format=PromptFormat.TEXT,
        input_token_ids=(10,),
        max_new_tokens=len(output_token_ids),
        stop_strings=(),
        output_text=request_id,
        output_token_ids=output_token_ids,
        stop_reason=stop_reason,
        matched_stop_string=None,
        prefill_tokens=(_routing(10, ForwardPhase.PREFILL, 0),),
        decode_tokens=tuple(
            _routing(token_id, ForwardPhase.DECODE, index)
            for index, token_id in enumerate(output_token_ids[:-1])
        ),
    )


def _trace() -> PreplayTrace:
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
    return PreplayTrace(
        provenance=provenance,
        requests=(
            _request("alpha", (101, 102, 0), StopReason.EOS),
            _request("beta", (201,), StopReason.LENGTH_CAP),
        ),
    )


@pytest.fixture
def trace_path(tmp_path):
    trace = _trace()
    return write_preplay_trace(
        tmp_path / "trace.jsonl", trace.provenance, trace.requests
    )


def _objects(bookkeeper: RequestBookkeeper) -> list[CreatedObjectRecord]:
    return [
        entry.fact
        for entry in bookkeeper.snapshot().entries
        if isinstance(entry.fact, CreatedObjectRecord)
    ]


def test_join_projects_exact_request_outcomes_and_trace_identity(trace_path):
    bookkeeper = RequestBookkeeper()
    run = join_preplay_arrivals(
        (
            RequestArrival(request_id="alpha", arrived_at_ps=1_000),
            RequestArrival(request_id="beta", arrived_at_ps=4_000),
        ),
        trace_path,
        bookkeeper,
    )

    assert run.schema == PREPLAY_REPLAY_RUN_SCHEMA
    assert run.trace.path == str(trace_path.resolve())
    assert run.trace.sha256 == hashlib.sha256(trace_path.read_bytes()).hexdigest()
    assert [request.request_id for request in run.requests] == ["alpha", "beta"]
    assert [request.arrived_at_ps for request in run.requests] == [1_000, 4_000]
    assert [request.output_token_ids for request in run.requests] == [
        (101, 102, 0),
        (201,),
    ]
    assert [request.output_length for request in run.requests] == [3, 1]
    assert [request.stop_reason for request in run.requests] == [
        StopReason.EOS,
        StopReason.LENGTH_CAP,
    ]
    assert {
        request.routing_reference.schema for request in run.requests
    } == {PREPLAY_ROUTING_REFERENCE_SCHEMA}
    assert {
        request.routing_reference.trace_sha256 for request in run.requests
    } == {run.trace.sha256}

    objects = _objects(bookkeeper)
    assert len(objects) == 2
    for request, record in zip(run.requests, objects, strict=True):
        metadata = dict(record.metadata)
        assert record.ref.kind is CreatedObjectKind.FRAMEWORK_REQUEST
        assert record.ref.object_id == request.bookkeeping_object_id
        assert record.owner is ObjectOwner.FRAMEWORK
        assert record.created_at_ps == request.arrived_at_ps
        assert record.scope.correlation.request_ids == (request.request_id,)
        assert record.native_id == request.request_id
        assert metadata["preplay_arrived_at_ps"] == request.arrived_at_ps
        assert metadata["preplay_output_length"] == request.output_length
        assert metadata["preplay_stop_reason"] == request.stop_reason.value
        assert json.loads(metadata["preplay_output_token_ids"]) == list(
            request.output_token_ids
        )
        assert json.loads(metadata["preplay_routing_reference"])[
            "trace_sha256"
        ] == run.trace.sha256


def test_arrival_shift_changes_only_arrival_fields(trace_path):
    base_bookkeeper = RequestBookkeeper()
    shifted_bookkeeper = RequestBookkeeper()
    base = join_preplay_arrivals(
        (
            RequestArrival(request_id="alpha", arrived_at_ps=1_000),
            RequestArrival(request_id="beta", arrived_at_ps=4_000),
        ),
        trace_path,
        base_bookkeeper,
    )
    shifted = join_preplay_arrivals(
        (
            RequestArrival(request_id="alpha", arrived_at_ps=8_000),
            RequestArrival(request_id="beta", arrived_at_ps=11_000),
        ),
        trace_path,
        shifted_bookkeeper,
    )

    for before, after in zip(base.requests, shifted.requests, strict=True):
        assert after.arrived_at_ps - before.arrived_at_ps == 7_000
        assert before.__class__(**{**before.__dict__, "arrived_at_ps": after.arrived_at_ps}) == after

    for before, after in zip(
        _objects(base_bookkeeper), _objects(shifted_bookkeeper), strict=True
    ):
        assert after.created_at_ps - before.created_at_ps == 7_000
        before_metadata = dict(before.metadata)
        after_metadata = dict(after.metadata)
        assert (
            after_metadata.pop("preplay_arrived_at_ps")
            - before_metadata.pop("preplay_arrived_at_ps")
            == 7_000
        )
        assert before_metadata == after_metadata
        assert before.ref == after.ref
        assert before.scope == after.scope
        assert before.native_id == after.native_id


@pytest.mark.parametrize(
    "arrivals, message",
    [
        ((), "must not be empty"),
        (
            (
                RequestArrival(request_id="alpha", arrived_at_ps=1),
                RequestArrival(request_id="alpha", arrived_at_ps=2),
            ),
            "duplicate request identity",
        ),
        ((RequestArrival(request_id="alpha", arrived_at_ps=-1),), "nonnegative"),
        ((RequestArrival(request_id="alpha", arrived_at_ps=True),), "integer"),
        ((RequestArrival(request_id="missing", arrived_at_ps=1),), "missing"),
    ],
)
def test_invalid_join_is_atomic(trace_path, arrivals, message):
    bookkeeper = RequestBookkeeper()
    before = bookkeeper.snapshot()
    with pytest.raises((TypeError, ValueError), match=message):
        join_preplay_arrivals(arrivals, trace_path, bookkeeper)
    assert bookkeeper.snapshot() == before


def test_bookkeeping_collision_rejects_the_whole_second_join(trace_path):
    arrivals = (
        RequestArrival(request_id="alpha", arrived_at_ps=1_000),
        RequestArrival(request_id="beta", arrived_at_ps=4_000),
    )
    bookkeeper = RequestBookkeeper()
    join_preplay_arrivals(arrivals, trace_path, bookkeeper)
    before = bookkeeper.snapshot()
    with pytest.raises(ValueError, match="duplicate object ID"):
        join_preplay_arrivals(arrivals, trace_path, bookkeeper)
    assert bookkeeper.snapshot() == before


def test_replay_run_strict_round_trip_and_rejections(trace_path, tmp_path):
    run = join_preplay_arrivals(
        (
            RequestArrival(request_id="alpha", arrived_at_ps=1_000),
            RequestArrival(request_id="beta", arrived_at_ps=4_000),
        ),
        trace_path,
        RequestBookkeeper(),
    )
    first = write_preplay_replay_run(run, tmp_path / "first.json")
    loaded = read_preplay_replay_run(first)
    second = write_preplay_replay_run(loaded, tmp_path / "second.json")
    assert first.read_bytes() == second.read_bytes()
    assert preplay_replay_run_from_json(preplay_replay_run_to_json(run)) == run

    base = preplay_replay_run_to_json(run)
    mutations = []
    unknown = copy.deepcopy(base)
    unknown["unknown"] = True
    mutations.append((unknown, "unknown fields"))
    bad_schema = copy.deepcopy(base)
    bad_schema["schema"] = "simllm-preplay-replay-run-v2"
    mutations.append((bad_schema, "unsupported schema"))
    duplicate = copy.deepcopy(base)
    duplicate["requests"].append(copy.deepcopy(duplicate["requests"][0]))
    mutations.append((duplicate, "duplicate request identity"))
    bad_length = copy.deepcopy(base)
    bad_length["requests"][0]["output_length"] = 99
    mutations.append((bad_length, "expected 3"))
    bad_routing = copy.deepcopy(base)
    bad_routing["requests"][0]["routing_reference"]["trace_sha256"] = "f" * 64
    mutations.append((bad_routing, "must match run.trace.sha256"))

    for payload, match in mutations:
        with pytest.raises(ValueError, match=match):
            preplay_replay_run_from_json(payload)
