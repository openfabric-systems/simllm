from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STUDY_DIR = REPOSITORY_ROOT / "examples" / "pd_session_load_delay_v1"


def _reader():
    spec = importlib.util.spec_from_file_location(
        "pd_session_load_delay_field_reader",
        STUDY_DIR / "field_reader.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _entry(batch_size: int, *, split: str = "calibration") -> dict:
    return {
        "implementation_id": f"granite-graph-decode-b{batch_size}-kv16",
        "coverage": "complete-kernel-stream",
        "key": {
            "model_identity": {
                "name": "ibm-granite/granite-3.0-1b-a400m-instruct"
            },
            "pool": "decode",
            "launch_mode": "cuda-graph",
            "parallelism": {"tensor_parallel": 1},
            "shape": {
                "batch_size": batch_size,
                "per_request_kv_lengths": [16] * batch_size,
            },
        },
        "measured_service_ps": batch_size * 10,
        "evidence": {"service_class": "MEASURED", "split": split},
    }


def _record() -> bytes:
    unselected = [
        {
            "forbidden": f"raw-skip-{index}",
            "model": "deepseek" if index < 8 else "other-granite-shape",
        }
        for index in range(14)
    ]
    return json.dumps(
        {
            "acceptance_status": "candidate",
            "campaign_id": "candidate-campaign",
            "capture_protocol": {"unselected": "skipped"},
            "device": {
                "architecture": "sm90",
                "device_kind_id": "nvidia-hopper-sm90",
                "forbidden_device_sentinel": "not returned",
            },
            "entries": [
                *unselected,
                _entry(1),
                _entry(32, split="held-out"),
                _entry(8),
                {"forbidden": "must not be consumed after entry 16"},
            ],
            "forbidden_top_level_sentinel": "must not be consumed",
        },
        indent=2,
        sort_keys=True,
    ).encode("utf-8")


def test_reader_returns_only_two_permitted_rows_and_stops_before_held_out() -> None:
    reader = _reader()
    payload = _record()

    projection, accesses, consumed = reader.extract_surface_projection(
        io.BytesIO(payload)
    )

    assert [row["key"]["shape"]["batch_size"] for row in projection["entries"]] == [
        1,
        8,
    ]
    assert projection["acceptance_status"] == "candidate"
    assert projection["device_kind_id"] == "nvidia-hopper-sm90"
    assert len(accesses) == 5
    assert projection["coverage"] == "complete-kernel-stream"
    assert consumed < len(payload)
    unread = payload[consumed:]
    assert b"must not be consumed after entry 16" in unread
    assert b'"batch_size": 32' not in unread
    assert all("forbidden" not in row for row in projection["entries"])


def test_reader_rejects_if_second_row_is_the_held_out_shape() -> None:
    reader = _reader()
    payload = json.dumps(
        {
            "acceptance_status": "candidate",
            "campaign_id": "candidate-campaign",
            "device": {"device_kind_id": "nvidia-hopper-sm90"},
            "entries": [
                *({"raw": index} for index in range(14)),
                _entry(1),
                {"raw": "held-out-not-decoded"},
                _entry(32, split="held-out"),
            ],
        },
        sort_keys=True,
    ).encode("utf-8")

    with pytest.raises(ValueError, match="disagrees"):
        reader.extract_surface_projection(io.BytesIO(payload))


def test_public_reader_refuses_other_paths_and_logs_rejection(tmp_path: Path) -> None:
    reader = _reader()
    access_log = tmp_path / "access.jsonl"

    with pytest.raises(ValueError, match="refuses"):
        reader.read_surface_projection(tmp_path / "candidate-record.json", access_log)

    rows = [json.loads(line) for line in access_log.read_text().splitlines()]
    assert rows == [
        {
            "error": "ValueError",
            "record": reader.RECORD_LABEL,
            "record_sha256": reader.RECORD_SHA256,
            "schema": reader.ACCESS_SCHEMA,
            "selector": "surface-projection",
            "status": "REJECTED",
            "whole_record_loaded": False,
        }
    ]


def test_access_protocol_freezes_only_calibration_batch_one_and_eight() -> None:
    protocol = json.loads(
        (STUDY_DIR / "access_protocol.json").read_text(encoding="utf-8")
    )

    assert protocol["status"] == "ACCESS_PROTOCOL_ONLY"
    assert [
        row["batch_size"] for row in protocol["permitted_entry_selectors"]
    ] == [1, 8]
    assert all(
        row["service_class"] == "MEASURED" and row["split"] == "calibration"
        for row in protocol["permitted_entry_selectors"]
    )
    assert protocol["forbidden"]["entry_indices"] == [0, 1, 2, 3, 4, 5, 6, 7, 15]
    assert protocol["forbidden"]["model_families"] == ["deepseek"]
    assert protocol["pre_protocol_incident"]["status"] == "CONTAMINATED"
