from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
READER_PATH = ROOT / "examples/deployment_curve_v1/flagship_run4_field_reader.py"


def _reader():
    spec = importlib.util.spec_from_file_location("flagship_run4_field_reader", READER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _encoded(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def test_successor_projection_returns_only_allowlisted_fields() -> None:
    reader = _reader()
    source = {
        "acceptance_status": "candidate",
        "lookup_record_sha256": reader.SUCCESSOR_SHA256,
        "predecessor_lookup_record_sha256": "f" * 64,
        "score": {
            "component_overlay_ledger": {
                "measured_unpriced_mtp": {"DISCLOSED": 1},
                "secret": 99,
            },
            "core61": {"status": "OPEN", "secret": 98},
            "mtp": {
                "evidence_class": "MEASURED",
                "lookup_pricing": "FORBIDDEN_BY_FREEZE",
                "measured_service_ps": 123,
                "secret": 97,
            },
            "task_movement": {"comp74_repeat_inputs": "RETAINED", "secret": 96},
            "secret": 95,
        },
    }
    raw = _encoded(source)

    value, consumed = reader.extract_successor_evidence(io.BytesIO(raw))

    assert value == {
        "lookup_record_sha256": reader.SUCCESSOR_SHA256,
        "predecessor_lookup_record_sha256": "f" * 64,
        "score": {
            "component_overlay_ledger": {"measured_unpriced_mtp": {"DISCLOSED": 1}},
            "core61": {"status": "OPEN"},
            "mtp": {
                "evidence_class": "MEASURED",
                "lookup_pricing": "FORBIDDEN_BY_FREEZE",
                "measured_service_ps": 123,
            },
            "task_movement": {"comp74_repeat_inputs": "RETAINED"},
        },
    }
    assert consumed < len(raw)
    assert "secret" not in json.dumps(value)


def test_anchor_projection_decodes_only_the_selected_row() -> None:
    reader = _reader()
    source = {
        "schema": "fixture",
        "anchors": [
            {"id": "calibration", "role": "calibration", "secret": [1, 2, 3]},
            {
                "id": reader.MTP_ANCHOR_ID,
                "role": "held-out",
                "value": 17,
                "unit": "tokens_per_second_per_node",
            },
            {"id": "later", "role": "held-out", "value": 99},
        ],
        "later_secret": 42,
    }
    raw = _encoded(source)

    value, offset, length, scanned = reader.extract_mtp_anchor(io.BytesIO(raw))

    assert value == source["anchors"][1]
    assert raw[offset : offset + length].startswith(b"{")
    assert scanned < len(raw)


def test_path_allowlists_reject_before_open_and_log(tmp_path: Path) -> None:
    reader = _reader()
    access_log = tmp_path / "access.jsonl"

    with pytest.raises(ValueError, match="non-allowlisted successor"):
        reader.read_successor_mtp_evidence(tmp_path / "wrong.json", access_log)
    with pytest.raises(ValueError, match="non-allowlisted anchor"):
        reader.read_mtp_anchor(tmp_path / "wrong.json", access_log)

    rows = [json.loads(line) for line in access_log.read_text(encoding="utf-8").splitlines()]
    assert [row["status"] for row in rows] == ["REJECTED", "REJECTED"]
    assert all(row["whole_record_loaded"] is False for row in rows)
