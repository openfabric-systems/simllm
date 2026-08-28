from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
from types import ModuleType

import pytest


def _load_reader() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[1]
        / "examples/deployment_curve_v1/core64_field_reader.py"
    )
    spec = importlib.util.spec_from_file_location("core64_field_reader", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reader = _load_reader()


def _source(payload: bytes):
    return reader._PartialSource(io.BytesIO(payload), len(payload))


def test_core63_projection_stops_before_unselected_tail() -> None:
    payload = json.dumps(
        {
            "calibration_only": {
                "residency_corrected": {
                    "classification": "UNDERCORRECTION",
                    "prediction_tokens_per_second_per_node": "10",
                    "signed_residual_percent": "-1",
                }
            },
            "residency_derivation": {
                "family_decomposition": {"kernel_row_count": 2},
                "step": {"residency_corrected_ps": {"decimal_ps": "20"}},
            },
            "scope": {
                "held_out_mtp_used_in_arithmetic_or_compared": False,
                "parameters_amended_or_refit": False,
                "scored_run_performed": False,
                "zero_free_or_fitted_constants": True,
            },
            "unselected_tail": "FORBIDDEN_TAIL",
        },
        sort_keys=True,
    ).encode() + b"\n"
    source = _source(payload)

    value = reader._extract_top_projection(source, reader.CORE63_PROJECTION)

    assert value["scope"]["scored_run_performed"] is False
    assert source.bytes_accessed <= payload.index(b"FORBIDDEN_TAIL")


def test_standard_case_selector_stops_before_mtp_case() -> None:
    payload = json.dumps(
        {
            "units": [
                {"case_projections": [{"case_id": "prefill"}]},
                {
                    "attention_parallelism": "data-parallel",
                    "case_projections": [
                        {"case_id": "sglang-decode-ep72-b32-c2000"},
                        {"case_id": "FORBIDDEN_MTP", "numeric": 17373},
                    ],
                    "id": "decode",
                },
            ]
        },
        sort_keys=True,
    ).encode() + b"\n"
    source = _source(payload)

    value = reader._extract_first_case(source, 1)

    assert value == {"case_id": "sglang-decode-ep72-b32-c2000"}
    assert source.bytes_accessed <= payload.index(b"FORBIDDEN_MTP")


def test_attention_selector_stops_before_case_array() -> None:
    payload = json.dumps(
        {
            "units": [
                {"case_projections": [{"case_id": "prefill"}]},
                {
                    "attention_parallelism": "data-parallel",
                    "case_projections": [{"case_id": "FORBIDDEN_MTP"}],
                },
            ]
        },
        sort_keys=True,
    ).encode() + b"\n"
    source = _source(payload)

    value = reader._extract_unit_field(source, 1, "attention_parallelism")

    assert value == "data-parallel"
    assert source.bytes_accessed <= payload.index(b"FORBIDDEN_MTP")


def test_whole_file_selector_is_rejected() -> None:
    payload = b'{"other":1}'
    source = _source(payload)

    with pytest.raises(reader.WholeFileAccessRejected):
        reader._extract_top_projection(source, {"missing": reader._CAPTURE})

    assert source.bytes_accessed == len(payload) - 1


def test_access_begin_precedes_source_open(tmp_path: Path) -> None:
    record = tmp_path / "record.json"
    record.write_text('{"selected":1,"tail":2}\n', encoding="utf-8", newline="\n")
    ledger = tmp_path / "ledger.jsonl"
    recorder = reader.AccessRecorder(ledger)

    def extract(source):
        events = [json.loads(line) for line in ledger.read_text().splitlines()]
        assert [event["event"] for event in events] == ["BEGIN"]
        return reader._SUPPORT._extract_top_field(source, "selected")

    value = reader._read_allowlisted(
        record,
        record,
        expected_bytes=record.stat().st_size,
        classification="synthetic",
        label="synthetic.json",
        selector="/selected",
        recorder=recorder,
        extractor=extract,
    )

    assert value == 1
    events = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert [event["event"] for event in events] == ["BEGIN", "END"]
    assert events[-1]["bytes_accessed"] < events[-1]["record_size_bytes"]
