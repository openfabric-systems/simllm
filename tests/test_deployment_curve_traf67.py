from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
STUDY_DIR = REPO_ROOT / "examples" / "deployment_curve_v1"


def _module(name: str):
    path = STUDY_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def reader():
    return _module("traf67_field_reader")


def _synthetic_record(rows: list[dict]) -> bytes:
    return json.dumps(
        {
            "accessed_visible_anchor_ids": ["sglang_prefill_1k"],
            "calibration_rows": rows,
            "forbidden_sentinel": "must not be consumed",
        },
        indent=2,
        sort_keys=True,
    ).encode("utf-8")


def test_field_reader_returns_only_visible_row_and_stops_early(reader):
    visible = {
        "anchor_id": "sglang_prefill_1k",
        "per_node_tokens": 131_072,
        "published": {"denominator": 1, "numerator": 57_674},
    }
    payload = _synthetic_record([visible])
    row, consumed = reader.extract_visible_row(io.BytesIO(payload))

    assert row == visible
    assert consumed < len(payload)
    assert payload[consumed:].find(b"must not be consumed") >= 0


def test_field_reader_rejects_a_second_row_without_decoding_it(reader):
    visible = {"anchor_id": "sglang_prefill_1k"}
    forbidden = {"anchor_id": "sglang_prefill_2k", "value": "not decoded"}

    with pytest.raises(ValueError, match="another calibration row"):
        reader.extract_visible_row(io.BytesIO(_synthetic_record([visible, forbidden])))


def test_public_reader_refuses_non_allowlisted_record_and_logs_attempt(
    reader, tmp_path
):
    access_log = tmp_path / "access-ledger.jsonl"
    candidate = tmp_path / "candidate-record.json"

    with pytest.raises(ValueError, match="refuses"):
        reader.read_visible_comp75_row(candidate, access_log)

    entries = [json.loads(line) for line in access_log.read_text().splitlines()]
    assert entries == [
        {
            "classification": "visible_calibration",
            "error": "ValueError",
            "record": "examples/deployment_curve_v1/comp75_calibration_result.json",
            "schema": "simllm-deployment-curve-traf67-access-v1",
            "selector": "/calibration_rows[anchor_id=sglang_prefill_1k]",
            "status": "REJECTED",
            "whole_record_loaded": False,
        }
    ]


def test_expectations_pin_clean_protocol_without_results():
    expectations = json.loads(
        (STUDY_DIR / "traf67_expectations.json").read_text(encoding="utf-8")
    )

    assert expectations["status"] == "EXPECTATIONS_ONLY"
    protocol = expectations["exposure_protocol"]
    assert protocol["permitted_anchor_ids"] == ["sglang_prefill_1k"]
    assert protocol["forbidden_anchor_ids"] == [
        "sglang_prefill_2k",
        "sglang_prefill_4k",
    ]
    assert protocol["expected_held_out_access_ledger"] == []
    assert protocol["whole_record_reads_permitted"] is False
    assert expectations["frozen_repetition"]["parameters_amended_or_refit"] is False
