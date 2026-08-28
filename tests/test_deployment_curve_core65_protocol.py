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
        / "examples/deployment_curve_v1/core65_field_reader.py"
    )
    spec = importlib.util.spec_from_file_location("core65_field_reader", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reader = _load_reader()


def _source(payload: bytes):
    return reader._PartialSource(io.BytesIO(payload), len(payload))


def test_top_projection_stops_after_last_selected_value() -> None:
    payload = b'{"a":{"value":1},"b":2,"forbidden":999}\n'
    source = _source(payload)

    value = reader._extract_top_projection(
        source,
        {"a": {"value": reader._CAPTURE}, "b": reader._CAPTURE},
    )

    assert value == {"a": {"value": 1}, "b": 2}
    assert 0 < source.bytes_accessed < len(payload)
    assert source.bytes_accessed <= payload.index(b"forbidden")


def test_missing_projection_rejects_complete_byte_coverage() -> None:
    payload = b'{"a":1}'
    source = _source(payload)

    with pytest.raises(reader.WholeFileAccessRejected):
        reader._extract_top_projection(source, {"missing": reader._CAPTURE})

    assert source.bytes_accessed == len(payload) - 1


def test_capture_cell_selector_returns_only_named_cell() -> None:
    payload = (
        b'{"cases":[{"cell":"prefill","secret":1},'
        b'{"cell":"decode_b32_c2000","batch_size":32}],'
        b'"later":999}\n'
    )
    source = _source(payload)

    value = reader._extract_capture_cell(source)

    assert value == {"cell": "decode_b32_c2000", "batch_size": 32}
    assert source.bytes_accessed < len(payload)
    assert source.bytes_accessed <= payload.index(b"later")


def test_access_begin_is_written_before_source_open(tmp_path: Path) -> None:
    record = tmp_path / "record.json"
    record.write_text('{"selected":1,"later":2}\n', encoding="utf-8", newline="\n")
    ledger = tmp_path / "access.jsonl"
    recorder = reader.AccessRecorder(ledger)

    def extract(source):
        rows = [json.loads(line) for line in ledger.read_text().splitlines()]
        assert [row["event"] for row in rows] == ["BEGIN"]
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
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert [row["event"] for row in rows] == ["BEGIN", "END"]
    assert rows[-1]["status"] == "PASS"
    assert rows[-1]["bytes_accessed"] < rows[-1]["record_size_bytes"]


def test_capture_profile_absence_is_logged_without_record_access(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "capture-access.jsonl"

    value = reader.read_capture_profile(
        kernelprobe_root=tmp_path,
        access_ledger=ledger,
    )

    assert value == {"available": False}
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert [row["event"] for row in rows] == ["BEGIN", "END"]
    assert rows[-1]["status"] == "UNAVAILABLE"
    assert rows[-1]["bytes_accessed"] == 0
    assert rows[-1]["record_size_bytes"] is None


def test_committed_freeze_records_protocol_void_and_directions() -> None:
    root = Path(__file__).resolve().parents[1]
    study = root / "examples/deployment_curve_v1"
    expectations = json.loads(
        (study / "core65_expectations.json").read_text(encoding="utf-8")
    )
    incidents = json.loads(
        (study / "core65_forbidden_access_ledger.json").read_text(
            encoding="utf-8"
        )
    )

    assert expectations["status"] == "EXPECTATIONS_ONLY_PROTOCOL_VOID"
    assert expectations["protocol_state"] == {
        "forbidden_access_ledger_empty": False,
        "held_out_mtp_numeric_value_used_or_compared": False,
        "known_pre_reader_incident_count": 2,
        "literal_protocol_closure_possible_in_this_worker": False,
        "whole_file_reader_streams_permitted": False,
    }
    composition = expectations["candidates"]["layer_type_composition"]
    assert composition["dense_only_signed_direction"] == "decrease"
    assert composition["moe_only_signed_direction"] == "increase"
    assert composition["net_signed_direction"].startswith("indeterminate")
    expert = expectations["candidates"]["expert_population"]
    assert expert["assignment_compute_scale"] == "1/9"
    assert expert["expert_count_or_resident_weight_byte_scale"] == "1/64"
    assert len(incidents) == 2
    assert any(row["held_out_numeric_value_exposed"] for row in incidents)
    assert all(not row["used_in_core65_arithmetic"] for row in incidents)


def test_preservation_freeze_extends_core64_to_154_files() -> None:
    root = Path(__file__).resolve().parents[1]
    study = root / "examples/deployment_curve_v1"
    expectations = json.loads(
        (study / "core65_expectations.json").read_text(encoding="utf-8")
    )
    rows = (study / "core65_prior_git_blobs.txt").read_text().splitlines()

    assert len(rows) == 20
    assert expectations["preservation"]["inherited_core64_checked_count"] == 134
    assert expectations["preservation"]["minimum_checked_count"] == 154


def test_all_committed_labels_are_portable() -> None:
    assert reader.CORE63_BASIS_LABEL.startswith("<wave-runs>/")
    assert reader.CAPTURE_PROFILE_LABEL.startswith("<kernelprobe-root>/")
    assert not reader.CORE64_RESULT_LABEL.startswith("/")
    assert not reader.MODEL_EXPECTATIONS_LABEL.startswith("/")
