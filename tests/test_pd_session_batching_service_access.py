from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STUDY_DIR = REPOSITORY_ROOT / "examples" / "pd_session_batching_service_v1"


def _reader():
    spec = importlib.util.spec_from_file_location(
        "pd_session_batching_service_field_reader",
        STUDY_DIR / "field_reader.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record() -> bytes:
    return json.dumps(
        {
            "schema": "fixture-v1",
            "cells": [
                {
                    "load_rps": 210,
                    "observed": {"batching_service_ps": 123},
                    "forbidden": {"raw": [1, 2, 3]},
                },
                {
                    "load_rps": 220,
                    "observed": {"batching_service_ps": 456},
                },
            ],
            "tail": "must not be returned with a field",
        },
        sort_keys=True,
    ).encode("utf-8")


def test_reader_returns_one_scalar_without_a_container() -> None:
    reader = _reader()

    value, consumed = reader.extract_field(
        io.BytesIO(_record()),
        "/cells/1/observed/batching_service_ps",
        "scalar",
    )

    assert value == 456
    assert 0 < consumed <= len(_record())


def test_reader_supports_bounded_structural_navigation() -> None:
    reader = _reader()

    keys, _ = reader.extract_field(io.BytesIO(_record()), "/cells/0", "keys")
    length, _ = reader.extract_field(io.BytesIO(_record()), "/cells", "length")

    assert keys == ["forbidden", "load_rps", "observed"]
    assert length == 2


@pytest.mark.parametrize("pointer", ["", "/"])
def test_reader_rejects_whole_record_pointers(pointer: str) -> None:
    reader = _reader()

    with pytest.raises(ValueError, match="forbidden"):
        reader.extract_field(io.BytesIO(_record()), pointer, "keys")


def test_reader_rejects_container_value_return() -> None:
    reader = _reader()

    with pytest.raises(ValueError, match="container"):
        reader.extract_field(io.BytesIO(_record()), "/cells/0", "scalar")


def test_public_reader_refuses_other_paths_and_logs_rejection(tmp_path: Path) -> None:
    reader = _reader()
    access_log = tmp_path / "access.jsonl"

    with pytest.raises(ValueError, match="refuses"):
        reader.read_field(
            tmp_path / "results.json",
            "/cells/0/load_rps",
            "scalar",
            access_log,
        )

    rows = [json.loads(line) for line in access_log.read_text().splitlines()]
    assert rows == [
        {
            "error": "ValueError",
            "mode": "scalar",
            "pointer": "/cells/0/load_rps",
            "record": reader.RECORD_LABEL,
            "schema": reader.ACCESS_SCHEMA,
            "status": "REJECTED",
            "whole_record_loaded": False,
        }
    ]


def test_protocol_starts_with_an_empty_forbidden_ledger() -> None:
    protocol = json.loads(
        (STUDY_DIR / "access_protocol.json").read_text(encoding="utf-8")
    )
    forbidden = json.loads(
        (STUDY_DIR / "forbidden_access_ledger.json").read_text(encoding="utf-8")
    )

    assert protocol["status"] == "ACCESS_PROTOCOL_ONLY"
    assert protocol["forbidden"]["whole_file_streams"] is True
    assert protocol["forbidden"]["pre_freeze_observed_value_access"] is True
    assert forbidden == []
