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
        / "examples/deployment_curve_v1/core63_clean_field_reader.py"
    )
    spec = importlib.util.spec_from_file_location("core63_clean_field_reader", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reader = _load_reader()


def _source(payload: bytes):
    return reader._PartialSource(io.BytesIO(payload), len(payload))


def test_top_field_selector_stops_before_end() -> None:
    payload = b'{"calibration_context":{"tokens":32},"forbidden":999}\n'
    source = _source(payload)

    value = reader._extract_top_field(source, "calibration_context")

    assert value == {"tokens": 32}
    assert 0 < source.bytes_accessed < len(payload)


def test_selector_that_would_stream_whole_file_is_rejected() -> None:
    payload = b'{"other":1}'
    source = _source(payload)

    with pytest.raises(reader.WholeFileAccessRejected):
        reader._extract_top_field(source, "missing")

    assert source.bytes_accessed == len(payload) - 1


def test_kernel_selector_stops_at_boundary_prefix() -> None:
    header = ",".join(reader.EXPECTED_HEADER) + "\n"
    selected = (
        "decode,decode_b32_c2000,0,1,fused_moe_kernel,False,,1,1,1,1,0.1,1,1\n"
    )
    collective = (
        "decode,decode_b32_c2000,0,2,collective_kernel,True,all_reduce,1,1,1,1,0.1,1,1\n"
    )
    boundary = "prefill,prefill_s1,0,secret_payload_that_must_not_be_decoded\n"
    trailing = "decode,other,0,unused\n"
    payload = (header + selected + collective + boundary + trailing).encode("utf-8")
    source = _source(payload)

    rows = reader._extract_kernels(source)

    assert len(rows) == 1
    assert rows[0]["name"] == "fused_moe_kernel"
    assert source.bytes_accessed < len(payload)
    assert source.bytes_accessed <= payload.index(b"secret_payload")


def test_access_begin_is_written_before_extractor_runs(tmp_path: Path) -> None:
    record = tmp_path / "record.json"
    record.write_text('{"selected":1,"later":2}\n', encoding="utf-8", newline="\n")
    ledger = tmp_path / "access.jsonl"
    recorder = reader.AccessRecorder(ledger)

    def extract(source):
        entries = [json.loads(line) for line in ledger.read_text().splitlines()]
        assert [entry["event"] for entry in entries] == ["BEGIN"]
        return reader._extract_top_field(source, "selected")

    value = reader._read_allowlisted(
        record,
        record,
        classification="synthetic",
        label="synthetic.json",
        selector="/selected",
        recorder=recorder,
        extractor=extract,
    )

    assert value == 1
    entries = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert [entry["event"] for entry in entries] == ["BEGIN", "END"]
    assert entries[-1]["status"] == "PASS"
    assert entries[-1]["bytes_accessed"] < entries[-1]["record_size_bytes"]


def test_access_ledger_cannot_be_reused(tmp_path: Path) -> None:
    ledger = tmp_path / "access.jsonl"
    reader.AccessRecorder(ledger)

    with pytest.raises(FileExistsError):
        reader.AccessRecorder(ledger)
