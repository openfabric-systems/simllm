"""Tests for the strict captured-routing projection."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from simllm.core import RequestBookkeeper
from simllm.preplay import (
    FRAMEWORK_PREPLAY_TRACE_SCHEMA,
    PREPLAY_TRACE_SCHEMA,
    ROUTED_EXPERTS_SCHEMA,
    ForwardPhase,
    ForwardTokenTrace,
    FrameworkPreplayTrace,
    FrameworkRequestTrace,
    FrameworkTraceProvenance,
    LayerRouting,
    ObservedLayerDispatch,
    ObservedTokenDispatch,
    PreplayTrace,
    PromptFormat,
    RequestArrival,
    RequestTrace,
    SamplingConfig,
    StopReason,
    TraceProvenance,
    join_preplay_arrivals,
    project_framework_routing,
    project_preplay_routing,
    read_routed_experts,
    routed_experts_from_json,
    routed_experts_to_json,
    write_framework_preplay_trace,
    write_preplay_trace,
    write_routed_experts,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_GRANITE_FIXTURE = _REPO_ROOT / "examples/preplay_trace_v1/granite_length_cap.jsonl"
_GRANITE_TRACE_SHA256 = (
    "36334f3aaa767c46d5f9c8498e02f6c2805a46e5000a57aea2747e17dd5d1341"
)
_GRANITE_PROJECTION_SHA256 = (
    "e3af45f896ff0a7005c4da0d6b4d3cfba7a00c868653e9aea581f49c37392e7a"
)


def _canonical_bytes(value) -> bytes:
    return (
        json.dumps(
            routed_experts_to_json(value),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )


def test_framework_projection_preserves_returned_top_k_order(tmp_path):
    routes = ((3, 1), (2, 1), (3, 2), (1, 3))
    provenance = FrameworkTraceProvenance(
        model_id="org/model",
        model_revision="test-revision",
        model_class="TestMoeForCausalLM",
        dtype="float32",
        tokenizer_sha256="a" * 64,
        sampling=SamplingConfig.greedy(),
        capture_host="test-host",
        runner="test-runner",
        framework="vllm",
        framework_version="1.0",
        observed_source="b" * 40,
        authored_against_source="c" * 40,
        torch_version="2.0",
        device="cpu",
        torch_num_threads=1,
        engine_seed=1,
        eos_token_id=0,
        top_k=2,
        expert_count=4,
        moe_layer_indices=(0,),
        kv_page_size=1,
        kv_token_capacity=64,
        dispatch_layer_mapping="framework-layer-id",
    )
    request = FrameworkRequestTrace(
        request_id="alpha",
        prompt_sha256="d" * 64,
        prompt_format=PromptFormat.TEXT,
        input_token_ids=(10, 11, 12, 13),
        max_new_tokens=1,
        stop_strings=(),
        output_text="done",
        output_token_ids=(0,),
        output_length=1,
        stop_reason=StopReason.LENGTH_CAP,
        matched_stop_string=None,
        framework_cached_tokens=0,
        framework_preemption_count=0,
        prefill_dispatch=tuple(
            ObservedTokenDispatch(
                phase=ForwardPhase.PREFILL,
                token_index=index,
                token_id=10 + index,
                routing=(
                    ObservedLayerDispatch(layer_index=0, expert_ids=expert_ids),
                ),
            )
            for index, expert_ids in enumerate(routes)
        ),
        decode_dispatch=(),
    )
    path = write_framework_preplay_trace(
        tmp_path / "framework.jsonl",
        FrameworkPreplayTrace(
            provenance=provenance,
            requests=(request,),
            kv_events=(),
        ),
    )

    projection = project_framework_routing(path)
    round_trip = read_routed_experts(
        write_routed_experts(projection, tmp_path / "routing.json")
    )

    assert projection.trace_schema == FRAMEWORK_PREPLAY_TRACE_SCHEMA
    assert projection.trace_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert tuple(
        token.layers[0].expert_ids for token in projection.requests[0].tokens
    ) == routes
    assert round_trip == projection


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


def _synthetic_trace() -> PreplayTrace:
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
            expert_count=4,
            moe_layer_indices=(0, 2),
        ),
        requests=(
            _request("alpha", (10, 11), (20, 21, 0)),
            _request("beta", (30,), (40,)),
        ),
    )


@pytest.fixture
def synthetic_trace_path(tmp_path):
    trace = _synthetic_trace()
    return write_preplay_trace(
        tmp_path / "trace.jsonl",
        trace.provenance,
        trace.requests,
    )


def test_granite_projection_matches_frozen_bytes_and_attribution():
    run = join_preplay_arrivals(
        (RequestArrival(request_id="length-cap", arrived_at_ps=0),),
        _GRANITE_FIXTURE,
        RequestBookkeeper(),
    )

    projection = project_preplay_routing(run)
    payload = _canonical_bytes(projection)
    request = projection.requests[0]

    assert projection.schema == ROUTED_EXPERTS_SCHEMA
    assert projection.trace_schema == PREPLAY_TRACE_SCHEMA
    assert projection.trace_sha256 == _GRANITE_TRACE_SHA256
    assert len(payload) == 30_874
    assert hashlib.sha256(payload).hexdigest() == _GRANITE_PROJECTION_SHA256
    assert projection.expert_count == 32
    assert projection.top_k == 8
    assert projection.moe_layer_indices == tuple(range(24))
    assert request.request_id == "length-cap"
    assert request.prompt_token_count == 22
    assert request.output_token_count == 1
    assert len(request.prefill_tokens) == 22
    assert request.decode_tokens == ()
    assert len(request.tokens) == 22
    assert sum(len(token.layers) for token in request.tokens) == 528
    assert sum(
        len(layer.expert_ids)
        for token in request.tokens
        for layer in token.layers
    ) == 4_224
    assert "gate_weights" not in payload.decode()


def test_projection_preserves_join_order_and_omits_terminal_tokens(
    synthetic_trace_path,
):
    run = join_preplay_arrivals(
        (
            RequestArrival(request_id="beta", arrived_at_ps=100),
            RequestArrival(request_id="alpha", arrived_at_ps=200),
        ),
        synthetic_trace_path,
        RequestBookkeeper(),
    )

    projection = project_preplay_routing(run)

    assert [request.request_id for request in projection.requests] == [
        "beta",
        "alpha",
    ]
    beta, alpha = projection.requests
    assert [token.token_id for token in beta.tokens] == [30]
    assert [token.token_id for token in alpha.prefill_tokens] == [10, 11]
    assert [token.token_id for token in alpha.decode_tokens] == [20, 21]
    assert [token.token_index for token in alpha.decode_tokens] == [0, 1]
    assert 0 not in [token.token_id for token in alpha.tokens]


def test_projection_round_trip_is_canonical_and_protects_existing_path(
    synthetic_trace_path,
    tmp_path,
):
    run = join_preplay_arrivals(
        (RequestArrival(request_id="alpha", arrived_at_ps=0),),
        synthetic_trace_path,
        RequestBookkeeper(),
    )
    projection = project_preplay_routing(run)

    first = write_routed_experts(projection, tmp_path / "first.json")
    loaded = read_routed_experts(first)
    second = write_routed_experts(loaded, tmp_path / "second.json")

    assert loaded == projection
    assert first.read_bytes() == second.read_bytes() == _canonical_bytes(projection)
    assert routed_experts_from_json(routed_experts_to_json(projection)) == projection
    with pytest.raises(FileExistsError):
        write_routed_experts(projection, first)


def test_projection_rejects_changed_trace_and_join_disagreement(
    synthetic_trace_path,
):
    run = join_preplay_arrivals(
        (RequestArrival(request_id="alpha", arrived_at_ps=0),),
        synthetic_trace_path,
        RequestBookkeeper(),
    )
    joined = run.requests[0]
    changed_output = replace(joined, output_token_ids=(20, 22, 0))
    changed_run = replace(run, requests=(changed_output,))
    with pytest.raises(ValueError, match="disagree with trace authority"):
        project_preplay_routing(changed_run)

    synthetic_trace_path.write_bytes(synthetic_trace_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="trace bytes changed"):
        project_preplay_routing(run)


def test_projection_reader_rejects_unknown_or_inconsistent_fields(
    synthetic_trace_path,
):
    run = join_preplay_arrivals(
        (RequestArrival(request_id="alpha", arrived_at_ps=0),),
        synthetic_trace_path,
        RequestBookkeeper(),
    )
    base = routed_experts_to_json(project_preplay_routing(run))
    mutations = []

    unknown = copy.deepcopy(base)
    unknown["unknown"] = True
    mutations.append((unknown, "unknown fields"))
    wrong_schema = copy.deepcopy(base)
    wrong_schema["schema"] = "simllm-routed-experts-v2"
    mutations.append((wrong_schema, "unsupported schema"))
    wrong_hash = copy.deepcopy(base)
    wrong_hash["trace_sha256"] = "A" * 64
    mutations.append((wrong_hash, "lowercase hexadecimal"))
    wrong_count = copy.deepcopy(base)
    wrong_count["requests"][0]["prompt_token_count"] = 1
    mutations.append((wrong_count, "expected 3 forwarded tokens"))
    wrong_phase = copy.deepcopy(base)
    wrong_phase["requests"][0]["tokens"][0]["phase"] = "decode"
    mutations.append((wrong_phase, "expected 'prefill'"))
    wrong_index = copy.deepcopy(base)
    wrong_index["requests"][0]["tokens"][1]["token_index"] = 9
    mutations.append((wrong_index, "expected contiguous index 1"))
    wrong_layer = copy.deepcopy(base)
    wrong_layer["requests"][0]["tokens"][0]["layers"][0]["layer_index"] = 1
    mutations.append((wrong_layer, "expected 0"))
    duplicate_expert = copy.deepcopy(base)
    duplicate_expert["requests"][0]["tokens"][0]["layers"][0][
        "expert_ids"
    ] = [0, 0]
    mutations.append((duplicate_expert, "duplicate values"))
    out_of_range = copy.deepcopy(base)
    out_of_range["requests"][0]["tokens"][0]["layers"][0]["expert_ids"] = [0, 4]
    mutations.append((out_of_range, r"outside \[0, 4\)"))

    for payload, match in mutations:
        with pytest.raises(ValueError, match=match):
            routed_experts_from_json(payload)


def test_projection_reader_rejects_duplicate_json_fields(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema":"first","schema":"second"}\n')

    with pytest.raises(ValueError, match="duplicate object field 'schema'"):
        read_routed_experts(path)


def test_projection_boundary_does_not_import_heavy_runtimes():
    code = (
        "import sys; import simllm.preplay.routing; "
        "assert 'torch' not in sys.modules; "
        "assert 'transformers' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
