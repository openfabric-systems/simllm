"""Packed routing-arena format, corruption and mmap lifetime tests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from simllm.core import RequestBookkeeper
from simllm.preplay.arena import (
    ROUTING_ARENA_SCHEMA,
    build_routing_arena,
    open_routing_arena,
    read_routing_arena_index,
)
from simllm.preplay.join import RequestArrival, join_preplay_arrivals
from simllm.preplay.schema import (
    ForwardPhase,
    ForwardTokenTrace,
    LayerRouting,
    PreplayTrace,
    PromptFormat,
    RequestTrace,
    SamplingConfig,
    StopReason,
    TraceProvenance,
)
from simllm.preplay.trace import write_preplay_trace


def _routing(
    phase: ForwardPhase,
    token_index: int,
    token_id: int,
    offset: int,
) -> ForwardTokenTrace:
    return ForwardTokenTrace(
        phase=phase,
        token_index=token_index,
        token_id=token_id,
        routing=(
            LayerRouting(
                layer_index=0,
                expert_ids=(offset % 4, (offset + 1) % 4),
                gate_weights=(0.75, 0.25),
            ),
            LayerRouting(
                layer_index=2,
                expert_ids=((offset + 2) % 4, (offset + 3) % 4),
                gate_weights=(0.6, 0.4),
            ),
        ),
    )


def _request(
    request_id: str,
    input_ids: tuple[int, ...],
    output_ids: tuple[int, ...],
) -> RequestTrace:
    return RequestTrace(
        request_id=request_id,
        prompt_sha256=hashlib.sha256(request_id.encode()).hexdigest(),
        prompt_format=PromptFormat.TEXT,
        input_token_ids=input_ids,
        max_new_tokens=len(output_ids),
        stop_strings=(),
        output_text=request_id,
        output_token_ids=output_ids,
        stop_reason=(
            StopReason.LENGTH_CAP if len(output_ids) == 1 else StopReason.EOS
        ),
        matched_stop_string=None,
        prefill_tokens=tuple(
            _routing(ForwardPhase.PREFILL, index, token_id, index)
            for index, token_id in enumerate(input_ids)
        ),
        decode_tokens=tuple(
            _routing(ForwardPhase.DECODE, index, token_id, index + 1)
            for index, token_id in enumerate(output_ids[:-1])
        ),
    )


def _trace(*, expert_count: int = 4) -> PreplayTrace:
    return PreplayTrace(
        provenance=TraceProvenance(
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
            top_k=2,
            expert_count=expert_count,
            moe_layer_indices=(0, 2),
        ),
        requests=(
            _request("alpha", (10, 11), (20, 21, 0)),
            _request("beta", (30,), (40,)),
        ),
    )


@pytest.fixture
def joined_run(tmp_path):
    trace = _trace()
    trace_path = write_preplay_trace(
        tmp_path / "trace.jsonl",
        trace.provenance,
        trace.requests,
    )
    return join_preplay_arrivals(
        (
            RequestArrival(request_id="beta", arrived_at_ps=100),
            RequestArrival(request_id="alpha", arrived_at_ps=200),
        ),
        trace_path,
        RequestBookkeeper(),
    )


def _canonical(payload: dict[str, object]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()


def _rewrite_index(path, mutation) -> None:
    payload = json.loads(path.read_text())
    mutation(payload)
    path.write_bytes(_canonical(payload))


def test_arena_packs_join_order_and_reads_without_gate_weights(joined_run, tmp_path):
    index_path = tmp_path / "routing-arena.json"
    arena = build_routing_arena(joined_run, index_path)
    try:
        assert arena.index.schema == ROUTING_ARENA_SCHEMA
        assert arena.expert_count == 4
        assert arena.top_k == 2
        assert arena.moe_layer_indices == (0, 2)
        assert [request.request_id for request in arena.requests] == ["beta", "alpha"]
        beta, alpha = arena.requests
        assert (beta.token_offset, beta.token_count) == (0, 1)
        assert (alpha.token_offset, alpha.token_count) == (1, 4)
        assert arena.expert_ids("beta", 0, 0) == (0, 1)
        assert arena.expert_ids("beta", 0, 2) == (2, 3)
        assert arena.expert_ids_at(
            alpha.token_offset,
            alpha.token_count,
            1,
            2,
        ) == (3, 0)
        assert arena.expert_id("alpha", 3, 2, 1) == 1
        assert arena.arena_id == hashlib.sha256(arena.payload_path.read_bytes()).hexdigest()
        assert len(arena.payload_path.read_bytes()) == 20
        assert b"gate" not in index_path.read_bytes()
        assert index_path.read_bytes().endswith(b"\n")
    finally:
        arena.close()


def test_join_can_publish_arena_after_bookkeeping_transaction_prevalidates(
    tmp_path,
) -> None:
    trace = _trace()
    trace_path = write_preplay_trace(
        tmp_path / "trace.jsonl",
        trace.provenance,
        trace.requests,
    )
    index_path = tmp_path / "run.routing.json"
    bookkeeper = RequestBookkeeper()
    run = join_preplay_arrivals(
        (
            RequestArrival(request_id="alpha", arrived_at_ps=100),
            RequestArrival(request_id="beta", arrived_at_ps=200),
        ),
        trace_path,
        bookkeeper,
        routing_arena_index_path=index_path,
    )

    assert len(bookkeeper.snapshot().entries) == 2
    with open_routing_arena(index_path, expected_run=run) as arena:
        assert [request.request_id for request in arena.requests] == [
            "alpha",
            "beta",
        ]


def test_join_removes_new_sidecars_if_bookkeeping_commit_fails(tmp_path) -> None:
    class RejectingBookkeeper(RequestBookkeeper):
        def extend(self, facts):
            del facts
            raise RuntimeError("injected bookkeeping failure")

    trace = _trace()
    trace_path = write_preplay_trace(
        tmp_path / "trace.jsonl",
        trace.provenance,
        trace.requests,
    )
    index_path = tmp_path / "run.routing.json"
    with pytest.raises(RuntimeError, match="injected bookkeeping failure"):
        join_preplay_arrivals(
            (RequestArrival(request_id="alpha", arrived_at_ps=100),),
            trace_path,
            RejectingBookkeeper(),
            routing_arena_index_path=index_path,
        )

    assert not index_path.exists()
    assert not index_path.with_suffix(".bin").exists()


def test_arena_request_view_blocks_close_and_releases_portably(joined_run, tmp_path):
    arena = build_routing_arena(joined_run, tmp_path / "arena.json")
    view = arena.acquire_request("alpha")
    assert arena.live_view_count == 1
    assert view.expert_ids(0, 0) == (0, 1)
    with pytest.raises(BufferError, match="1 live request"):
        arena.close()
    view.release()
    assert arena.live_view_count == 0
    with pytest.raises(RuntimeError, match="released"):
        view.expert_ids(0, 0)
    arena.close()
    assert arena.closed
    with pytest.raises(RuntimeError, match="closed"):
        arena.expert_ids("alpha", 0, 0)


def test_open_validates_join_provenance_and_protects_existing_files(joined_run, tmp_path):
    index_path = tmp_path / "arena.json"
    arena = build_routing_arena(joined_run, index_path)
    arena.close()
    with pytest.raises(FileExistsError):
        build_routing_arena(joined_run, index_path)
    reopened = open_routing_arena(index_path, expected_run=joined_run)
    reopened.close()

    changed_run = replace(
        joined_run,
        trace=replace(joined_run.trace, sha256="f" * 64),
        requests=tuple(
            replace(
                request,
                routing_reference=replace(
                    request.routing_reference,
                    trace_sha256="f" * 64,
                ),
            )
            for request in joined_run.requests
        ),
    )
    with pytest.raises(ValueError, match="trace bytes changed"):
        open_routing_arena(index_path, expected_run=changed_run)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value.update(schema="simllm-routing-arena-index-v2"), "unsupported"),
        (lambda value: value.update(unknown=True), "unknown fields"),
        (
            lambda value: value["requests"][1].update(token_offset=0),
            "overlaps",
        ),
        (
            lambda value: value["requests"][1].update(token_offset=2),
            "leaves a gap",
        ),
        (
            lambda value: value["requests"].append(dict(value["requests"][1])),
            "duplicate request identity",
        ),
        (lambda value: value.update(expert_count=257), "at most 256"),
        (
            lambda value: value.update(moe_layer_indices=list(range(65))),
            "at most 64",
        ),
        (lambda value: value.update(payload_file="../arena.bin"), "sibling file"),
    ],
)
def test_index_rejects_structural_corruption(joined_run, tmp_path, mutation, match):
    index_path = tmp_path / "arena.json"
    arena = build_routing_arena(joined_run, index_path)
    arena.close()
    _rewrite_index(index_path, mutation)
    with pytest.raises((TypeError, ValueError), match=match):
        read_routing_arena_index(index_path)


def test_index_rejects_duplicate_fields_and_noncanonical_json(joined_run, tmp_path):
    index_path = tmp_path / "arena.json"
    arena = build_routing_arena(joined_run, index_path)
    arena.close()
    payload = json.loads(index_path.read_text())
    index_path.write_text(json.dumps(payload, indent=2) + "\n")
    with pytest.raises(ValueError, match="not canonical"):
        read_routing_arena_index(index_path)
    index_path.write_text('{"schema":"first","schema":"second"}\n')
    with pytest.raises(ValueError, match="duplicate object field 'schema'"):
        read_routing_arena_index(index_path)


@pytest.mark.parametrize("corruption", ["truncated", "extra", "hash", "expert"])
def test_open_rejects_payload_corruption(joined_run, tmp_path, corruption):
    index_path = tmp_path / "arena.json"
    arena = build_routing_arena(joined_run, index_path)
    payload_path = arena.payload_path
    arena.close()
    data = payload_path.read_bytes()
    if corruption == "truncated":
        payload_path.write_bytes(data[:-1])
        match = "truncated"
    elif corruption == "extra":
        payload_path.write_bytes(data + b"\0")
        match = "extra bytes"
    elif corruption == "hash":
        payload_path.write_bytes(bytes([data[0] ^ 1]) + data[1:])
        match = "SHA-256"
    else:
        payload_path.write_bytes(bytes([4]) + data[1:])
        _rewrite_index(
            index_path,
            lambda value: value.update(
                payload_sha256=hashlib.sha256(payload_path.read_bytes()).hexdigest()
            ),
        )
        match = "outside"
    with pytest.raises(ValueError, match=match):
        open_routing_arena(index_path)


def test_builder_rejects_more_than_256_experts(tmp_path):
    trace = _trace(expert_count=257)
    trace_path = write_preplay_trace(
        tmp_path / "wide.jsonl",
        trace.provenance,
        trace.requests,
    )
    run = join_preplay_arrivals(
        (RequestArrival(request_id="alpha", arrived_at_ps=0),),
        trace_path,
        RequestBookkeeper(),
    )
    with pytest.raises(ValueError, match="at most 256"):
        build_routing_arena(run, tmp_path / "arena.json")


def test_builder_accepts_exactly_256_experts(tmp_path):
    trace = _trace(expert_count=256)
    trace_path = write_preplay_trace(
        tmp_path / "wide.jsonl",
        trace.provenance,
        trace.requests,
    )
    run = join_preplay_arrivals(
        (RequestArrival(request_id="alpha", arrived_at_ps=0),),
        trace_path,
        RequestBookkeeper(),
    )
    with build_routing_arena(run, tmp_path / "arena.json") as arena:
        assert arena.expert_count == 256
