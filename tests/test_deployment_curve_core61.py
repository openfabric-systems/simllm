from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STUDY_DIR = REPOSITORY_ROOT / "examples/deployment_curve_v1"


def _reader():
    spec = importlib.util.spec_from_file_location(
        "deployment_curve_core61_depth_field_reader",
        STUDY_DIR / "core61_depth_field_reader.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _selected_entry() -> dict:
    return {
        "coverage": "complete-kernel-stream",
        "distribution": {"forbidden_sentinel": "must not be decoded"},
        "evidence": {"component_class": "DISCLOSED", "service_class": "MEASURED"},
        "implementation_id": "deepseek-v3-reduced4-vllm-ep72-decode-b32-c2000",
        "kernels": [
            {
                "components": {
                    "compute_sm_cycles": 3_376_223,
                    "fixed_overhead_ps": 123,
                    "memory": {"service_ps": 0, "weight_bytes": None},
                    "method": "aggregate retained Nsys noncollective step service",
                },
                "kernel_id": "aggregate_noncollective_step_service",
                "launch_count": 1,
                "measured_elapsed_ps": 1_875_680_000,
                "name": "forbidden selected-entry field",
            }
        ],
        "key": {
            "launch_mode": "cuda-graph",
            "parallelism": {"data_parallel": 72, "expert_parallel": 72},
            "pool": "decode",
            "shape": {"batch_size": 32, "per_request_kv_lengths": [2000] * 32},
        },
        "measured_service_ps": 1_875_680_000,
        "observed_clocks": {"sm_hz": {"median": 1_800_000_000}},
    }


def _synthetic_record(selected: dict) -> bytes:
    prefix = [
        {"implementation_id": f"forbidden-{index}", "secret": "not decoded"}
        for index in range(7)
    ]
    return json.dumps(
        {
            "acceptance_status": "candidate",
            "entries": prefix
            + [selected]
            + [{"forbidden_tail_sentinel": "must remain unread"}],
            "forbidden_top_level_tail": "must remain unread",
        },
        indent=2,
        sort_keys=True,
    ).encode("utf-8")


def test_expectations_freeze_direction_fields_and_single_depth_class():
    expectations = json.loads(
        (STUDY_DIR / "core61_depth_expectations.json").read_text(encoding="utf-8")
    )

    assert expectations["status"] == "EXPECTATIONS_ONLY"
    assert expectations["hypothesis"]["expected_direction"].startswith("strictly smaller")
    protocol = expectations["exposure_protocol"]
    assert protocol["whole_record_reads_permitted"] is False
    assert protocol["unselected_values_may_be_decoded"] is False
    assert protocol["allowed_fields"] == list(_reader().ALLOWED_FIELDS)
    evidence = expectations["evidence_class"]
    assert evidence["corrected_service_class"] == "DECLARED"
    assert evidence["source_service_class_required"] == "MEASURED"
    assert evidence["depth_basis_count"] == 1
    assert expectations["comparison_context"]["may_tune_or_select"] is False
    assert expectations["acceptance"]["core61_closes_on_local_arm"] is False


def test_field_reader_projects_only_allowed_fields_and_stops_before_tail():
    reader = _reader()
    payload = _synthetic_record(_selected_entry())

    selected, consumed = reader.extract_depth_basis(io.BytesIO(payload))

    assert selected["implementation_id"] == reader.EXPECTED_IMPLEMENTATION_ID
    assert "distribution" not in selected
    assert "name" not in selected["kernels"][0]
    assert "weight_bytes" not in selected["kernels"][0]["components"]["memory"]
    assert consumed < len(payload)
    assert payload[consumed:].find(b"must remain unread") >= 0


def test_field_reader_rejects_a_second_kernel_without_decoding_it():
    reader = _reader()
    selected = _selected_entry()
    selected["kernels"].append({"forbidden_kernel": "must not be decoded"})

    with pytest.raises(ValueError, match="unregistered kernel"):
        reader.extract_depth_basis(io.BytesIO(_synthetic_record(selected)))


def test_public_reader_refuses_other_records_and_logs_the_attempt(tmp_path: Path):
    reader = _reader()
    access_log = tmp_path / "access-ledger.jsonl"

    with pytest.raises(ValueError, match="refuses"):
        reader.read_retained_depth_basis(tmp_path / "candidate-record.json", access_log)

    entries = [json.loads(line) for line in access_log.read_text(encoding="utf-8").splitlines()]
    assert len(entries) == 1
    assert entries[0]["status"] == "REJECTED"
    assert entries[0]["whole_record_loaded"] is False
    assert entries[0]["unselected_values_decoded"] is False


def test_merlin_remainder_is_depth8_exact_shape_and_maintenance_gated():
    expectations = json.loads(
        (STUDY_DIR / "core61_depth_expectations.json").read_text(encoding="utf-8")
    )
    deferred = expectations["deferred_merlin"]

    assert deferred["prediction_depth_layers"] == 8
    assert deferred["exact_shape"] == {
        "batch_size": 32,
        "remote_kv_tokens_per_request": 2000,
    }
    assert deferred["maintenance_gate"] == {
        "not_before_local": "2026-08-28T06:30",
        "timezone": "Europe/Zurich",
    }
    for command in (
        deferred["core61_depth8_base_command"],
        deferred["core61_depth8_decode_command"],
    ):
        assert "REDUCED_LAYERS=8" in command
        assert "REVISION=e815299b0bcbac849fa540c768ef21845365c9eb" in command
        assert "--partition=gh-hourly" in command


def test_scope_locks_keep_scored_runs_comp76_and_code_untouched():
    expectations = json.loads(
        (STUDY_DIR / "core61_depth_expectations.json").read_text(encoding="utf-8")
    )

    assert set(expectations["preserved_scope"].values()) == {False}
    assert expectations["acceptance"]["reserved_residual_id"] == "CORE-63"
