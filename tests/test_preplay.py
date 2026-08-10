import importlib
import json
from dataclasses import replace

import pytest

from simllm.preplay import (
    PREPLAY_TRACE_SCHEMA,
    LayerRouting,
    PreplayRequest,
    PreplayTrace,
    PreplayTraceReader,
    PreplayTraceWriter,
    PromptFormat,
    RequestTrace,
    SamplingConfig,
    SamplingMode,
    StopReason,
    TokenTrace,
    TraceProvenance,
    TransformersCpuRunner,
    read_preplay_trace,
    trace_provenance_from_json,
    trace_provenance_to_json,
    validate_preplay_trace,
    validate_request_trace,
    validate_sampling_config,
    write_preplay_trace,
)
from simllm.preplay.runner import _classify_stop_reason, _resolve_model_source


def provenance(*, sampling=None):
    return TraceProvenance(
        model_id="org/model",
        model_revision="0123456789abcdef",
        model_class="TestMoeForCausalLM",
        dtype="float32",
        tokenizer_sha256="a" * 64,
        sampling=SamplingConfig.greedy() if sampling is None else sampling,
        capture_host="capture-host",
        runner="transformers-cpu",
        transformers_version="5.0.0",
        torch_version="2.9.0",
        device="cpu",
        torch_num_threads=4,
        eos_token_id=0,
        top_k=2,
        expert_count=4,
        moe_layer_indices=(0, 1),
    )


def routing(layer_index, first_expert=0):
    return LayerRouting(
        layer_index=layer_index,
        expert_ids=(first_expert, (first_expert + 1) % 4),
        gate_weights=(0.75, 0.25),
    )


def token(token_index, token_id):
    return TokenTrace(
        token_index=token_index,
        token_id=token_id,
        routing=(routing(0, token_index), routing(1, token_index + 1)),
    )


def length_cap_request(request_id="request-0"):
    return RequestTrace(
        request_id=request_id,
        prompt_sha256="b" * 64,
        prompt_format=PromptFormat.CHAT,
        input_token_ids=(10, 11, 12),
        max_new_tokens=2,
        stop_strings=(),
        output_text="AB",
        stop_reason=StopReason.LENGTH_CAP,
        matched_stop_string=None,
        tokens=(token(0, 20), token(1, 21)),
    )


def eos_request(request_id="request-eos"):
    return RequestTrace(
        request_id=request_id,
        prompt_sha256="c" * 64,
        prompt_format=PromptFormat.TEXT,
        input_token_ids=(3,),
        max_new_tokens=4,
        stop_strings=(),
        output_text="OK",
        stop_reason=StopReason.EOS,
        matched_stop_string=None,
        tokens=(token(0, 25), token(1, 0)),
    )


def stop_string_request(request_id="request-stop"):
    return RequestTrace(
        request_id=request_id,
        prompt_sha256="d" * 64,
        prompt_format=PromptFormat.CHAT,
        input_token_ids=(5,),
        max_new_tokens=8,
        stop_strings=("STOP",),
        output_text="STOP",
        stop_reason=StopReason.STOP_STRING,
        matched_stop_string="STOP",
        tokens=(token(0, 30),),
    )


def jsonl_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def write_jsonl_rows(path, rows):
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        )
    )


def test_trace_golden_rows_and_round_trip(tmp_path):
    path = write_preplay_trace(
        tmp_path / "trace.jsonl",
        provenance(),
        (length_cap_request(), eos_request()),
    )

    rows = jsonl_rows(path)
    assert [row["row_type"] for row in rows] == [
        "header",
        "request",
        "token",
        "token",
        "request",
        "token",
        "token",
        "footer",
    ]
    assert rows[0] == {
        "schema": PREPLAY_TRACE_SCHEMA,
        "row_type": "header",
        "provenance": trace_provenance_to_json(provenance()),
    }
    assert rows[1]["request_id"] == "request-0"
    assert rows[1]["output_token_count"] == 2
    assert rows[2]["routing"][0] == {
        "layer_index": 0,
        "expert_ids": [0, 1],
        "gate_weights": [0.75, 0.25],
    }
    assert rows[-1] == {
        "schema": PREPLAY_TRACE_SCHEMA,
        "row_type": "footer",
        "request_count": 2,
        "token_count": 4,
    }

    trace = read_preplay_trace(path)
    assert trace == PreplayTrace(
        provenance=provenance(),
        requests=(length_cap_request(), eos_request()),
    )
    assert trace.by_request_id("request-eos").output_token_ids == (25, 0)
    with pytest.raises(KeyError, match="missing"):
        trace.by_request_id("missing")


def test_read_write_round_trip_is_byte_identical(tmp_path):
    first = write_preplay_trace(
        tmp_path / "first.jsonl",
        provenance(sampling=SamplingConfig.seeded(173, temperature=0.8, top_p=0.9)),
        (length_cap_request(), stop_string_request()),
    )
    trace = read_preplay_trace(first)
    second = write_preplay_trace(tmp_path / "second.jsonl", trace.provenance, trace.requests)

    assert first.read_bytes() == second.read_bytes()


def test_writer_flushes_each_request_before_consuming_the_next(tmp_path):
    path = tmp_path / "streamed.jsonl"

    def requests():
        yield length_cap_request()
        on_disk = path.read_text()
        assert '"request_id":"request-0"' in on_disk
        assert '"row_type":"footer"' not in on_disk
        yield eos_request()

    write_preplay_trace(path, provenance(), requests())

    assert tuple(request.request_id for request in read_preplay_trace(path).requests) == (
        "request-0",
        "request-eos",
    )


def test_reader_yields_only_complete_requests_and_validates_footer(tmp_path):
    path = write_preplay_trace(
        tmp_path / "trace.jsonl",
        provenance(),
        (length_cap_request(), eos_request()),
    )

    with PreplayTraceReader(path) as reader:
        first = next(reader)
        assert first.request_id == "request-0"

    rows = jsonl_rows(path)
    rows.pop(3)
    write_jsonl_rows(tmp_path / "missing-token.jsonl", rows)
    with pytest.raises(ValueError, match="missing fields.*routing"):
        read_preplay_trace(tmp_path / "missing-token.jsonl")

    path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
    with pytest.raises(ValueError, match="missing request row or completeness footer"):
        read_preplay_trace(path)


@pytest.mark.parametrize(
    "row_index,location",
    [
        (0, "row"),
        (0, "provenance"),
        (1, "row"),
        (2, "row"),
        (-1, "row"),
    ],
)
def test_reader_rejects_unknown_fields_at_every_level(tmp_path, row_index, location):
    source = write_preplay_trace(
        tmp_path / "source.jsonl",
        provenance(),
        (length_cap_request(),),
    )
    rows = jsonl_rows(source)
    target = rows[row_index]["provenance"] if location == "provenance" else rows[row_index]
    target["typo"] = True
    bad = tmp_path / f"bad-{row_index}-{location}.jsonl"
    write_jsonl_rows(bad, rows)

    with pytest.raises(ValueError, match="unknown fields.*typo"):
        read_preplay_trace(bad)


def test_reader_rejects_wrong_schema_duplicate_ids_and_trailing_content(tmp_path):
    source = write_preplay_trace(
        tmp_path / "source.jsonl",
        provenance(),
        (length_cap_request(),),
    )
    rows = jsonl_rows(source)
    rows[2]["schema"] = "simllm-preplay-trace-v2"
    write_jsonl_rows(tmp_path / "wrong-schema.jsonl", rows)
    with pytest.raises(ValueError, match="unsupported schema"):
        read_preplay_trace(tmp_path / "wrong-schema.jsonl")

    duplicate = tmp_path / "duplicate.jsonl"
    with (
        pytest.raises(ValueError, match="duplicate value"),
        PreplayTraceWriter(duplicate, provenance()) as writer,
    ):
        writer.append(length_cap_request())
        writer.append(length_cap_request())
    with pytest.raises(ValueError, match="missing request row or completeness footer"):
        read_preplay_trace(duplicate)

    with source.open("a") as handle:
        handle.write('{"extra":true}\n')
    with pytest.raises(ValueError, match="content after completeness footer"):
        read_preplay_trace(source)


def test_reader_rejects_duplicate_json_object_fields(tmp_path):
    path = tmp_path / "duplicate-field.jsonl"
    path.write_text(
        '{"schema":"simllm-preplay-trace-v1",'
        '"row_type":"header","row_type":"header","provenance":{}}\n'
    )

    with pytest.raises(ValueError, match="duplicate object field 'row_type'"):
        read_preplay_trace(path)


@pytest.mark.parametrize(
    "sampling,match",
    [
        (
            SamplingConfig(mode=SamplingMode.GREEDY, seed=1),
            "must be null in greedy mode",
        ),
        (
            SamplingConfig(mode=SamplingMode.SEEDED_SAMPLING),
            "seed.*required",
        ),
        (
            SamplingConfig.seeded(1, temperature=0.0),
            "temperature.*greater than zero",
        ),
        (
            SamplingConfig.seeded(1, top_p=1.1),
            "top_p.*at most one",
        ),
    ],
)
def test_sampling_configuration_is_mode_strict(sampling, match):
    with pytest.raises(ValueError, match=match):
        validate_sampling_config(sampling)


def test_provenance_round_trip_is_strict():
    original = provenance(
        sampling=SamplingConfig.seeded(173, temperature=0.8, top_p=0.9)
    )
    payload = trace_provenance_to_json(original)
    assert trace_provenance_from_json(json.loads(json.dumps(payload))) == original

    payload["unknown"] = 1
    with pytest.raises(ValueError, match="unknown fields.*unknown"):
        trace_provenance_from_json(payload)


@pytest.mark.parametrize(
    "bad_route,match",
    [
        (LayerRouting(layer_index=0, expert_ids=(0,), gate_weights=(1.0,)), "top_k=2"),
        (
            LayerRouting(layer_index=0, expert_ids=(0, 4), gate_weights=(0.5, 0.5)),
            "below expert_count=4",
        ),
        (
            LayerRouting(layer_index=0, expert_ids=(0, 1), gate_weights=(0.5, 0.4)),
            "sum to one",
        ),
    ],
)
def test_routing_dimensions_and_values_are_fatal(bad_route, match):
    request = length_cap_request()
    first_token = replace(request.tokens[0], routing=(bad_route, request.tokens[0].routing[1]))
    request = replace(request, tokens=(first_token, request.tokens[1]))

    with pytest.raises(ValueError, match=match):
        validate_request_trace(request, provenance())


def test_every_token_requires_the_exact_provenance_layer_set():
    request = length_cap_request()
    first_token = replace(request.tokens[0], routing=(request.tokens[0].routing[0],))
    request = replace(request, tokens=(first_token, request.tokens[1]))

    with pytest.raises(ValueError, match="exactly match provenance"):
        validate_request_trace(request, provenance())


def test_stop_semantics_are_validated_against_tokens_and_text():
    validate_request_trace(eos_request(), provenance())
    validate_request_trace(length_cap_request(), provenance())
    validate_request_trace(stop_string_request(), provenance())

    with pytest.raises(ValueError, match="eos output must end"):
        validate_request_trace(
            replace(eos_request(), tokens=(token(0, 25), token(1, 26))),
            provenance(),
        )
    with pytest.raises(ValueError, match="must reach max_new_tokens"):
        validate_request_trace(
            replace(length_cap_request(), max_new_tokens=3),
            provenance(),
        )
    with pytest.raises(ValueError, match="must appear in output_text"):
        validate_request_trace(
            replace(stop_string_request(), output_text="not present"),
            provenance(),
        )


def test_stop_classifier_uses_eos_then_string_then_length_cap():
    assert _classify_stop_reason(
        (7, 0),
        "STOP",
        eos_token_id=0,
        max_new_tokens=2,
        stop_strings=("STOP",),
    ) == (StopReason.EOS, None)
    assert _classify_stop_reason(
        (7,),
        "STOP",
        eos_token_id=0,
        max_new_tokens=2,
        stop_strings=("STOP",),
    ) == (StopReason.STOP_STRING, "STOP")
    assert _classify_stop_reason(
        (7, 8),
        "text",
        eos_token_id=0,
        max_new_tokens=2,
        stop_strings=(),
    ) == (StopReason.LENGTH_CAP, None)


def test_materialized_trace_rejects_duplicate_request_identity():
    trace = PreplayTrace(
        provenance=provenance(),
        requests=(length_cap_request(), length_cap_request()),
    )

    with pytest.raises(ValueError, match="duplicate values"):
        validate_preplay_trace(trace)


def test_preplay_request_freezes_and_validates_stop_strings():
    stop_strings = ["DONE"]
    request = PreplayRequest(
        request_id="request",
        prompt="prompt",
        max_new_tokens=2,
        stop_strings=stop_strings,
    )
    stop_strings.append("LATE")
    assert request.stop_strings == ("DONE",)

    with pytest.raises(ValueError, match="duplicate"):
        PreplayRequest(
            request_id="request",
            prompt="prompt",
            max_new_tokens=2,
            stop_strings=("DONE", "DONE"),
        )


def test_runner_dependency_error_is_lazy_and_clear(monkeypatch, tmp_path):
    snapshot = tmp_path / "hub/models--org--model/snapshots/revision"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    real_import_module = importlib.import_module

    def missing_runtime(name, package=None):
        if name in {"torch", "transformers"}:
            raise ModuleNotFoundError(name)
        return real_import_module(name, package)

    monkeypatch.setattr("simllm.preplay.runner.importlib.import_module", missing_runtime)
    with pytest.raises(RuntimeError, match="optional dependencies.*preplay"):
        TransformersCpuRunner(
            model_id="org/model",
            revision="revision",
            cache_dir=tmp_path,
        )


def test_model_source_resolves_exact_hf_home_revision(tmp_path):
    snapshot = tmp_path / "hub/models--org--model/snapshots/pinned-revision"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")

    assert _resolve_model_source("org/model", "pinned-revision", tmp_path) == snapshot

    with pytest.raises(FileNotFoundError, match="not available offline"):
        _resolve_model_source("org/model", "missing-revision", tmp_path)
