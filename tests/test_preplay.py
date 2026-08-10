import importlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from simllm.preplay import (
    PREPLAY_TRACE_SCHEMA,
    ForwardPhase,
    ForwardTokenTrace,
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
from simllm.preplay.runner import (
    _classify_stop_reason,
    _resolve_model_source,
    _tokenizer_sha256,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_WRITER_GOLDEN = _REPO_ROOT / "examples/preplay_trace_v1/writer_golden.jsonl"


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


def forward_token(phase, token_index, token_id, route_offset=0):
    return ForwardTokenTrace(
        phase=phase,
        token_index=token_index,
        token_id=token_id,
        routing=(routing(0, route_offset), routing(1, (route_offset + 1) % 4)),
    )


def prefill_tokens(token_ids):
    return tuple(
        forward_token(ForwardPhase.PREFILL, index, token_id, index % 4)
        for index, token_id in enumerate(token_ids)
    )


def length_cap_request(request_id="request-0"):
    input_token_ids = (10, 11, 12)
    return RequestTrace(
        request_id=request_id,
        prompt_sha256="b" * 64,
        prompt_format=PromptFormat.CHAT,
        input_token_ids=input_token_ids,
        max_new_tokens=2,
        stop_strings=(),
        output_text="AB",
        output_token_ids=(20, 21),
        stop_reason=StopReason.LENGTH_CAP,
        matched_stop_string=None,
        prefill_tokens=prefill_tokens(input_token_ids),
        decode_tokens=(forward_token(ForwardPhase.DECODE, 0, 20),),
    )


def eos_request(request_id="request-eos"):
    input_token_ids = (3,)
    return RequestTrace(
        request_id=request_id,
        prompt_sha256="c" * 64,
        prompt_format=PromptFormat.TEXT,
        input_token_ids=input_token_ids,
        max_new_tokens=4,
        stop_strings=(),
        output_text="OK",
        output_token_ids=(25, 0),
        stop_reason=StopReason.EOS,
        matched_stop_string=None,
        prefill_tokens=prefill_tokens(input_token_ids),
        decode_tokens=(forward_token(ForwardPhase.DECODE, 0, 25),),
    )


def stop_string_request(request_id="request-stop"):
    input_token_ids = (5,)
    return RequestTrace(
        request_id=request_id,
        prompt_sha256="d" * 64,
        prompt_format=PromptFormat.CHAT,
        input_token_ids=input_token_ids,
        max_new_tokens=8,
        stop_strings=("STOP",),
        output_text="STOP",
        output_token_ids=(30,),
        stop_reason=StopReason.STOP_STRING,
        matched_stop_string="STOP",
        prefill_tokens=prefill_tokens(input_token_ids),
        decode_tokens=(),
    )


def golden_request():
    return RequestTrace(
        request_id="request-golden",
        prompt_sha256="b" * 64,
        prompt_format=PromptFormat.CHAT,
        input_token_ids=(10,),
        max_new_tokens=1,
        stop_strings=(),
        output_text="A",
        output_token_ids=(20,),
        stop_reason=StopReason.LENGTH_CAP,
        matched_stop_string=None,
        prefill_tokens=(forward_token(ForwardPhase.PREFILL, 0, 10),),
        decode_tokens=(),
    )


def jsonl_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def write_jsonl_rows(path, rows):
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in rows
        )
    )


def test_writer_matches_frozen_byte_fixture(tmp_path):
    path = write_preplay_trace(tmp_path / "trace.jsonl", provenance(), (golden_request(),))

    assert path.read_bytes() == _WRITER_GOLDEN.read_bytes()
    assert read_preplay_trace(_WRITER_GOLDEN) == PreplayTrace(
        provenance=provenance(),
        requests=(golden_request(),),
    )


def test_writer_protects_existing_path_unless_overwrite_is_explicit(tmp_path):
    path = tmp_path / "trace.jsonl"
    path.write_bytes(b"existing artifact\n")

    with pytest.raises(FileExistsError):
        write_preplay_trace(path, provenance(), (golden_request(),))
    assert path.read_bytes() == b"existing artifact\n"

    write_preplay_trace(path, provenance(), (golden_request(),), overwrite=True)
    assert path.read_bytes() == _WRITER_GOLDEN.read_bytes()


def test_trace_rows_and_round_trip(tmp_path):
    path = write_preplay_trace(
        tmp_path / "trace.jsonl",
        provenance(),
        (length_cap_request(), eos_request()),
    )

    rows = jsonl_rows(path)
    assert [row["row_type"] for row in rows] == [
        "header",
        "request",
        "forward-token",
        "forward-token",
        "forward-token",
        "forward-token",
        "request",
        "forward-token",
        "forward-token",
        "footer",
    ]
    assert rows[0] == {
        "schema": PREPLAY_TRACE_SCHEMA,
        "row_type": "header",
        "provenance": trace_provenance_to_json(provenance()),
    }
    assert rows[1]["output_token_ids"] == [20, 21]
    assert rows[2]["phase"] == "prefill"
    assert rows[5]["phase"] == "decode"
    assert rows[-1] == {
        "schema": PREPLAY_TRACE_SCHEMA,
        "row_type": "footer",
        "request_count": 2,
        "prefill_forward_token_count": 4,
        "decode_forward_token_count": 2,
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
    rows.pop(4)
    write_jsonl_rows(tmp_path / "missing-prefill.jsonl", rows)
    with pytest.raises(ValueError, match="phase|forward-token"):
        read_preplay_trace(tmp_path / "missing-prefill.jsonl")

    rows = jsonl_rows(path)
    rows.pop(5)
    write_jsonl_rows(tmp_path / "missing-decode.jsonl", rows)
    with pytest.raises(ValueError, match="forward-token|missing fields"):
        read_preplay_trace(tmp_path / "missing-decode.jsonl")

    no_footer = tmp_path / "no-footer.jsonl"
    no_footer.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
    with pytest.raises(ValueError, match="missing request row or completeness footer"):
        read_preplay_trace(no_footer)


def test_reader_rejects_terminal_token_forward_row(tmp_path):
    source = write_preplay_trace(
        tmp_path / "source.jsonl",
        provenance(),
        (length_cap_request(),),
    )
    rows = jsonl_rows(source)
    terminal = dict(rows[5])
    terminal["token_index"] = 1
    terminal["token_id"] = 21
    rows.insert(6, terminal)
    target = tmp_path / "terminal-forward.jsonl"
    write_jsonl_rows(target, rows)

    with pytest.raises(ValueError, match="expected 'request' or 'footer'"):
        read_preplay_trace(target)


@pytest.mark.parametrize(
    "row_index,location",
    [
        (0, "row"),
        (0, "provenance"),
        (1, "row"),
        (2, "row"),
        (2, "route"),
        (-1, "row"),
    ],
)
def test_reader_rejects_unknown_fields_at_every_level(
    tmp_path,
    row_index,
    location,
):
    source = write_preplay_trace(
        tmp_path / "source.jsonl",
        provenance(),
        (length_cap_request(),),
    )
    rows = jsonl_rows(source)
    if location == "provenance":
        target = rows[row_index]["provenance"]
    elif location == "route":
        target = rows[row_index]["routing"][0]
    else:
        target = rows[row_index]
    target["typo"] = True
    bad = tmp_path / f"bad-{row_index}-{location}.jsonl"
    write_jsonl_rows(bad, rows)

    with pytest.raises(ValueError, match="unknown fields.*typo"):
        read_preplay_trace(bad)


def test_reader_rejects_mid_line_truncated_json(tmp_path):
    source = write_preplay_trace(
        tmp_path / "source.jsonl",
        provenance(),
        (golden_request(),),
    )
    lines = source.read_text().splitlines(keepends=True)
    lines[2] = lines[2][: len(lines[2]) // 2] + "\n"
    target = tmp_path / "truncated.jsonl"
    target.write_text("".join(lines))

    with pytest.raises(ValueError, match="line 3.*invalid JSON"):
        read_preplay_trace(target)


def test_reader_rejects_header_missing_model_revision(tmp_path):
    source = write_preplay_trace(
        tmp_path / "source.jsonl",
        provenance(),
        (golden_request(),),
    )
    rows = jsonl_rows(source)
    del rows[0]["provenance"]["model_revision"]
    target = tmp_path / "missing-revision.jsonl"
    write_jsonl_rows(target, rows)

    with pytest.raises(ValueError, match="missing fields.*model_revision"):
        read_preplay_trace(target)


def test_reader_rejects_wrong_schema_duplicate_ids_and_trailing_content(tmp_path):
    source = write_preplay_trace(
        tmp_path / "source.jsonl",
        provenance(),
        (length_cap_request(),),
    )
    rows = jsonl_rows(source)
    rows[2]["schema"] = "simllm-preplay-trace-v2"
    wrong_schema = tmp_path / "wrong-schema.jsonl"
    write_jsonl_rows(wrong_schema, rows)
    with pytest.raises(ValueError, match="unsupported schema"):
        read_preplay_trace(wrong_schema)

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
    first_token = replace(
        request.prefill_tokens[0],
        routing=(bad_route, request.prefill_tokens[0].routing[1]),
    )
    request = replace(
        request,
        prefill_tokens=(first_token, *request.prefill_tokens[1:]),
    )

    with pytest.raises(ValueError, match=match):
        validate_request_trace(request, provenance())


def test_every_forward_token_requires_the_exact_provenance_layer_set():
    request = length_cap_request()
    first_token = replace(
        request.prefill_tokens[0],
        routing=(request.prefill_tokens[0].routing[0],),
    )
    request = replace(
        request,
        prefill_tokens=(first_token, *request.prefill_tokens[1:]),
    )

    with pytest.raises(ValueError, match="exactly match provenance"):
        validate_request_trace(request, provenance())


@pytest.mark.parametrize(
    "case_request,match",
    [
        (
            replace(
                length_cap_request(),
                prefill_tokens=length_cap_request().prefill_tokens[:-1],
            ),
            "expected 3 prefill",
        ),
        (
            replace(length_cap_request(), decode_tokens=()),
            "expected 1 decode",
        ),
        (
            replace(
                length_cap_request(),
                decode_tokens=(
                    *length_cap_request().decode_tokens,
                    forward_token(ForwardPhase.DECODE, 1, 21),
                ),
            ),
            "expected 1 decode",
        ),
        (
            replace(
                length_cap_request(),
                decode_tokens=(
                    replace(
                        length_cap_request().decode_tokens[0],
                        phase=ForwardPhase.PREFILL,
                    ),
                ),
            ),
            "expected 'decode'",
        ),
        (
            replace(
                length_cap_request(),
                decode_tokens=(
                    replace(length_cap_request().decode_tokens[0], token_id=21),
                ),
            ),
            "expected forwarded token ID 20",
        ),
    ],
)
def test_forward_coverage_and_attribution_are_exact(case_request, match):
    with pytest.raises(ValueError, match=match):
        validate_request_trace(case_request, provenance())


def test_stop_semantics_are_validated_against_output_tokens_and_text():
    validate_request_trace(eos_request(), provenance())
    validate_request_trace(length_cap_request(), provenance())
    validate_request_trace(stop_string_request(), provenance())

    with pytest.raises(ValueError, match="eos output must end"):
        validate_request_trace(
            replace(eos_request(), output_token_ids=(25, 26)),
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


def test_stop_classifier_is_the_single_priority_order():
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
    assert (
        _classify_stop_reason(
            (7,),
            "text",
            eos_token_id=0,
            max_new_tokens=2,
            stop_strings=(),
        )
        is None
    )


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


def test_tokenizer_hash_covers_pinned_files_and_chat_template_fallback(tmp_path):
    class FakeTokenizer:
        def __init__(self):
            self.backend_tokenizer = None
            self.chat_template = "template-a"
            self.special_tokens_map = {"eos_token": "<eos>"}

        @staticmethod
        def get_vocab():
            return {"token": 1}

        @staticmethod
        def get_added_vocab():
            return {"<eos>": 0}

    tokenizer = FakeTokenizer()
    fallback = _tokenizer_sha256(tokenizer, tmp_path)
    tokenizer.chat_template = "template-b"
    assert _tokenizer_sha256(tokenizer, tmp_path) != fallback

    (tmp_path / "tokenizer.json").write_text("tokenizer-a")
    file_hash = _tokenizer_sha256(tokenizer, tmp_path)
    tokenizer.chat_template = "ignored-when-files-are-pinned"
    assert _tokenizer_sha256(tokenizer, tmp_path) == file_hash
    (tmp_path / "tokenizer.json").write_text("tokenizer-b")
    assert _tokenizer_sha256(tokenizer, tmp_path) != file_hash
