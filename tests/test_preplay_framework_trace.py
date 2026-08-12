import base64
import json
from array import array
from contextlib import nullcontext
from dataclasses import replace

import pytest

from simllm.preplay import (
    FRAMEWORK_PREPLAY_TRACE_SCHEMA,
    OBSERVED_DISPATCH_SOURCE,
    ForwardPhase,
    FrameworkOracleRequest,
    FrameworkPreplayTrace,
    FrameworkRequestTrace,
    FrameworkTraceProvenance,
    KvCacheEvent,
    KvEventKind,
    ObservedLayerDispatch,
    ObservedTokenDispatch,
    PromptFormat,
    SamplingConfig,
    StopReason,
    framework_runner,
    read_framework_preplay_trace,
    read_preplay_trace,
    validate_framework_preplay_trace,
    write_framework_preplay_trace,
)
from simllm.preplay.framework_runner import (
    _request_from_response,
    _sidecar_events,
    _vllm_sidecar_projection,
)


def provenance():
    return FrameworkTraceProvenance(
        model_id="org/model",
        model_revision="0123456789abcdef",
        model_class="FrameworkMoeForCausalLM",
        dtype="float32",
        tokenizer_sha256="a" * 64,
        sampling=SamplingConfig.greedy(),
        capture_host="capture-host",
        runner="sglang-cpu",
        framework="sglang",
        framework_version="1.2.3",
        observed_source="f" * 40,
        authored_against_source="e" * 40,
        torch_version="2.11.0",
        device="cpu",
        torch_num_threads=4,
        engine_seed=173,
        eos_token_id=0,
        top_k=2,
        expert_count=4,
        moe_layer_indices=(0, 1),
        kv_page_size=1,
        kv_token_capacity=64,
        dispatch_layer_mapping="framework-layer-id",
    )


def dispatch(phase, token_index, token_id, offset=0):
    return ObservedTokenDispatch(
        phase=phase,
        token_index=token_index,
        token_id=token_id,
        routing=(
            ObservedLayerDispatch(
                layer_index=0,
                expert_ids=(offset % 4, (offset + 1) % 4),
            ),
            ObservedLayerDispatch(
                layer_index=1,
                expert_ids=((offset + 1) % 4, (offset + 2) % 4),
            ),
        ),
    )


def request():
    return FrameworkRequestTrace(
        request_id="request-0",
        prompt_sha256="b" * 64,
        prompt_format=PromptFormat.CHAT,
        input_token_ids=(10, 11),
        max_new_tokens=2,
        stop_strings=(),
        output_text="AB",
        output_token_ids=(20, 21),
        output_length=2,
        stop_reason=StopReason.LENGTH_CAP,
        matched_stop_string=None,
        framework_cached_tokens=0,
        framework_preemption_count=0,
        prefill_dispatch=(
            dispatch(ForwardPhase.PREFILL, 0, 10),
            dispatch(ForwardPhase.PREFILL, 1, 11, 1),
        ),
        decode_dispatch=(dispatch(ForwardPhase.DECODE, 0, 20, 2),),
    )


def events():
    return (
        KvCacheEvent(
            sequence=0,
            kind=KvEventKind.PREFIX_HIT,
            request_id="request-0",
            framework_step=0,
            token_count=0,
        ),
        KvCacheEvent(
            sequence=1,
            kind=KvEventKind.ALLOCATION,
            request_id="request-0",
            framework_step=0,
            token_count=2,
            token_slot_ids=(3, 4),
        ),
        KvCacheEvent(
            sequence=2,
            kind=KvEventKind.EVICTION,
            request_id=None,
            framework_step=1,
            token_count=1,
            token_slot_ids=(8,),
            reason="capacity",
        ),
        KvCacheEvent(
            sequence=3,
            kind=KvEventKind.PREEMPTION,
            request_id="request-0",
            framework_step=2,
            token_count=0,
            reason="decode-pressure",
        ),
        KvCacheEvent(
            sequence=4,
            kind=KvEventKind.RELEASE,
            request_id="request-0",
            framework_step=3,
            token_count=2,
            token_slot_ids=(3, 4),
        ),
    )


def trace():
    return FrameworkPreplayTrace(
        provenance=provenance(),
        requests=(request(),),
        kv_events=events(),
    )


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def write_rows(path, values):
    path.write_text(
        "".join(
            json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
            for value in values
        )
    )


def test_framework_trace_round_trip_is_canonical_and_strict(tmp_path):
    first = write_framework_preplay_trace(tmp_path / "first.jsonl", trace())
    loaded = read_framework_preplay_trace(first)
    second = write_framework_preplay_trace(tmp_path / "second.jsonl", loaded)

    assert loaded == trace()
    assert first.read_bytes() == second.read_bytes()
    assert [value["row_type"] for value in rows(first)] == [
        "header",
        "request",
        "observed-dispatch",
        "observed-dispatch",
        "observed-dispatch",
        "kv-event",
        "kv-event",
        "kv-event",
        "kv-event",
        "kv-event",
        "footer",
    ]
    assert loaded.provenance.schema == FRAMEWORK_PREPLAY_TRACE_SCHEMA
    assert loaded.requests[0].prefill_dispatch[0].routing_source == (
        OBSERVED_DISPATCH_SOURCE
    )


def test_framework_trace_protects_existing_path(tmp_path):
    path = tmp_path / "trace.jsonl"
    path.write_bytes(b"existing\n")

    with pytest.raises(FileExistsError):
        write_framework_preplay_trace(path, trace())
    assert path.read_bytes() == b"existing\n"

    write_framework_preplay_trace(path, trace(), overwrite=True)
    assert read_framework_preplay_trace(path) == trace()


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda values: values[0].__setitem__("unknown", True),
            "unknown fields",
        ),
        (
            lambda values: values[2].__setitem__("routing_source", "recomputed"),
            "observed dispatch",
        ),
        (
            lambda values: values[1].__setitem__("output_length", 1),
            "output_length",
        ),
        (
            lambda values: values[5].__setitem__("sequence", 4),
            "sequence",
        ),
        (
            lambda values: values[-1].__setitem__("kv_event_count", 9),
            "kv_event_count",
        ),
        (
            lambda values: values.pop(),
            "missing",
        ),
    ],
)
def test_framework_reader_rejects_malformed_rows(tmp_path, mutate, match):
    source = write_framework_preplay_trace(tmp_path / "source.jsonl", trace())
    malformed = rows(source)
    mutate(malformed)
    path = tmp_path / "malformed.jsonl"
    write_rows(path, malformed)

    with pytest.raises(ValueError, match=match):
        read_framework_preplay_trace(path)


def test_framework_validation_rejects_action_field_and_dispatch_shape():
    invalid_event = replace(
        events()[1],
        block_ids=(),
        token_slot_ids=(),
    )
    with pytest.raises(ValueError, match="requires block_ids or token_slot_ids"):
        validate_framework_preplay_trace(
            replace(trace(), kv_events=(events()[0], invalid_event))
        )

    invalid_dispatch = replace(
        request().prefill_dispatch[0],
        routing=(ObservedLayerDispatch(layer_index=0, expert_ids=(0, 1)),),
    )
    invalid_request = replace(
        request(),
        prefill_dispatch=(invalid_dispatch, request().prefill_dispatch[1]),
    )
    with pytest.raises(ValueError, match="layer rows"):
        validate_framework_preplay_trace(replace(trace(), requests=(invalid_request,)))

    with pytest.raises(ValueError, match="divisible by kv_page_size"):
        validate_framework_preplay_trace(
            replace(
                trace(),
                provenance=replace(
                    provenance(),
                    kv_page_size=3,
                    kv_token_capacity=64,
                ),
            )
        )


def test_v1_and_v2_readers_do_not_reinterpret_each_other(tmp_path):
    v2 = write_framework_preplay_trace(tmp_path / "v2.jsonl", trace())
    with pytest.raises(ValueError, match="unsupported schema"):
        read_preplay_trace(v2)

    v1 = tmp_path / "v1.jsonl"
    v1.write_text(
        '{"provenance":{},"row_type":"header",'
        '"schema":"simllm-preplay-trace-v1"}\n'
    )
    with pytest.raises(ValueError, match="unsupported schema"):
        read_framework_preplay_trace(v1)


def test_sglang_response_projection_uses_returned_dispatch_and_finish_reason():
    oracle_request = FrameworkOracleRequest(
        request_id="request-0",
        prompt_sha256="b" * 64,
        prompt_format=PromptFormat.CHAT,
        input_token_ids=(10, 11),
        max_new_tokens=2,
    )
    selected = array("i", [0, 1, 1, 2, 2, 3, 3, 0, 0, 2, 1, 3])
    response = {
        "text": "AB",
        "output_ids": [20, 21],
        "meta_info": {
            "id": "request-0",
            "finish_reason": {"type": "length", "length": 2},
            "cached_tokens": 0,
            "num_retractions": 0,
            "routed_experts": base64.b64encode(selected.tobytes()).decode(),
        },
    }

    projected = _request_from_response(response, oracle_request, provenance())

    assert projected.output_token_ids == (20, 21)
    assert projected.stop_reason is StopReason.LENGTH_CAP
    assert projected.prefill_dispatch[0].routing[0].expert_ids == (0, 1)
    assert projected.prefill_dispatch[1].routing[1].expert_ids == (3, 0)
    assert projected.decode_dispatch[0].routing[0].expert_ids == (0, 2)


def test_sidecar_projection_requires_stock_cpu_worker_and_real_allocations():
    values = [
        {
            "kind": "capture-storage-qualified",
            "device": "cpu",
            "pinned": False,
        },
        {"kind": "capture-start", "request_ids": ["request-0"]},
        {
            "kind": "dispatch-layer-qualified",
            "mapping": "granite-model-order",
            "selected_experts_unchanged": True,
        },
        {
            "kind": "worker-qualified",
            "worker_class": "TpModelWorker",
            "model_runner_class": "ModelRunner",
            "model_class": "GraniteMoeForCausalLM",
            "parameter_devices": ["cpu"],
        },
        {
            "kind": "prefix-hit",
            "request_id": "request-0",
            "framework_step": 0,
            "token_count": 0,
            "token_slot_ids": [],
        },
        {
            "kind": "allocation",
            "request_id": "request-0",
            "framework_step": 0,
            "token_count": 2,
            "token_slot_ids": [3, 4],
        },
    ]

    projected = _sidecar_events(values, {"request-0"})

    assert [event.kind for event in projected] == [
        KvEventKind.PREFIX_HIT,
        KvEventKind.ALLOCATION,
    ]
    assert projected[1].token_slot_ids == (3, 4)

    values[3]["worker_class"] = "SimTpModelWorker"
    with pytest.raises(RuntimeError, match="unexpected SGLang worker"):
        _sidecar_events(values, {"request-0"})


def test_vllm_projection_requires_stock_cpu_authorities_and_zero_cuda_delta():
    values = [
        {
            "kind": "kv-manager-qualified",
            "manager_class": "KVCacheManager",
            "block_size": 16,
            "token_capacity": 64,
        },
        {
            "kind": "worker-qualified",
            "worker_class": "CPUWorker",
            "model_runner_class": "CPUModelRunner",
            "model_class": "GraniteMoeForCausalLM",
            "parameter_devices": ["cpu"],
            "cuda_available_before": False,
            "cuda_available_after": False,
            "cuda_memory_allocated_before": 0,
            "cuda_memory_allocated_after": 0,
        },
        {
            "kind": "dispatch-path-qualified",
            "capture_source": "cpu-monolithic-select-experts-return",
            "layer_ids": [0, 1],
            "selected_experts_unchanged": True,
        },
        {"kind": "capture-start", "request_ids": ["request-0"]},
        {
            "kind": "request-mapping",
            "mappings": [
                {
                    "internal_request_id": "internal-0",
                    "request_id": "request-0",
                }
            ],
        },
        {
            "kind": "dispatch-qualified",
            "capture_class": "RoutedExpertsCapturer",
            "capture_source": "post-selection-router-output",
            "selected_experts_unchanged": True,
        },
        {
            "kind": "prefix-hit",
            "request_id": "internal-0",
            "token_count": 0,
            "block_ids": [],
        },
        {
            "kind": "allocation",
            "request_id": "internal-0",
            "token_count": 16,
            "block_ids": [2],
        },
        {
            "kind": "request-final-counters",
            "request_id": "internal-0",
            "num_preemptions": 0,
        },
    ]

    events, block_size, capacity, model_class, preemptions = (
        _vllm_sidecar_projection(values, {"request-0"})
    )

    assert [event.kind for event in events] == [
        KvEventKind.PREFIX_HIT,
        KvEventKind.ALLOCATION,
    ]
    assert (block_size, capacity, model_class) == (
        16,
        64,
        "GraniteMoeForCausalLM",
    )
    assert preemptions == {"request-0": 0}

    values[1]["cuda_memory_allocated_after"] = 16
    with pytest.raises(RuntimeError, match="CPU worker qualification failed"):
        _vllm_sidecar_projection(values, {"request-0"})


def test_missing_sglang_dependency_rejects_before_trace_writer_opens(
    monkeypatch, tmp_path
):
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text("{}\n")
    output = tmp_path / "framework.jsonl"
    raw = tmp_path / "raw.json"
    sidecar = tmp_path / "sidecar.jsonl"
    runner = framework_runner.SglangCpuRunner(
        model_id="org/model",
        model_revision="revision",
        model_path=model_path,
        tokenizer_sha256="a" * 64,
        observed_source="f" * 40,
        authored_against_source="e" * 40,
    )
    oracle_request = FrameworkOracleRequest(
        request_id="request-0",
        prompt_sha256="b" * 64,
        prompt_format=PromptFormat.CHAT,
        input_token_ids=(10,),
        max_new_tokens=1,
    )

    def missing_dependency(name):
        assert name == "sglang"
        raise ModuleNotFoundError("No module named 'sglang'")

    monkeypatch.setattr(framework_runner.importlib, "import_module", missing_dependency)
    with pytest.raises(ModuleNotFoundError, match="sglang"):
        runner.capture(
            (oracle_request,),
            output,
            max_total_tokens=16,
            context_length=16,
            max_running_requests=1,
            observation_path=sidecar,
            raw_response_path=raw,
        )

    assert not output.exists()
    assert not raw.exists()


def test_missing_vllm_dependency_rejects_before_trace_writer_opens(
    monkeypatch, tmp_path
):
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text(
        json.dumps(
            {
                "eos_token_id": 0,
                "hidden_size": 16,
                "num_attention_heads": 2,
                "num_experts_per_tok": 2,
                "num_hidden_layers": 1,
                "num_key_value_heads": 1,
                "num_local_experts": 4,
            }
        )
        + "\n"
    )
    output = tmp_path / "framework.jsonl"
    raw = tmp_path / "raw.json"
    sidecar = tmp_path / "sidecar.jsonl"
    runner = framework_runner.VllmCpuRunner(
        model_id="org/model",
        model_revision="revision",
        model_path=model_path,
        tokenizer_sha256="a" * 64,
        observed_source="f" * 40,
        authored_against_source="e" * 40,
    )
    oracle_request = FrameworkOracleRequest(
        request_id="request-0",
        prompt_sha256="b" * 64,
        prompt_format=PromptFormat.CHAT,
        input_token_ids=(10,),
        max_new_tokens=1,
    )

    def missing_dependency(name):
        assert name == "vllm"
        raise ModuleNotFoundError("No module named 'vllm'")

    monkeypatch.delitem(framework_runner.sys.modules, "vllm", raising=False)
    monkeypatch.setattr(
        framework_runner,
        "_vllm_oracle_environment",
        lambda *_args: nullcontext(),
    )
    monkeypatch.setattr(framework_runner.importlib, "import_module", missing_dependency)
    with pytest.raises(ModuleNotFoundError, match="vllm"):
        runner.capture(
            (oracle_request,),
            output,
            kv_token_capacity=16,
            context_length=16,
            max_running_requests=1,
            block_size=16,
            observation_path=sidecar,
            raw_response_path=raw,
        )

    assert not output.exists()
    assert not raw.exists()
