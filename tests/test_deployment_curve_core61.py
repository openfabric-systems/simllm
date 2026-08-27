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


def _extrapolation():
    spec = importlib.util.spec_from_file_location(
        "deployment_curve_core61_depth_extrapolation",
        STUDY_DIR / "core61_depth_extrapolation.py",
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
                    "compute_sm_cycles": 3_751_359,
                    "fixed_overhead_ps": 500,
                    "memory": {"service_ps": 0, "weight_bytes": None},
                    "method": (
                        "Retained Nsys additive noncollective service encoded as elapsed "
                        "SM-clock cycles"
                    ),
                },
                "kernel_id": "aggregate_noncollective_step_service",
                "launch_count": 1,
                "measured_elapsed_ps": 1_875_680_000,
                "name": "forbidden selected-entry field",
            }
        ],
        "key": {
            "launch_mode": "cuda-graph",
            "parallelism": {
                "tensor_parallel": 1,
                "pipeline_parallel": 1,
                "data_parallel": 1,
                "expert_parallel": 1,
            },
            "pool": "decode",
            "shape": {"batch_size": 32, "per_request_kv_lengths": [2000] * 32},
        },
        "measured_service_ps": 1_875_680_000,
        "observed_clocks": {"sm_hz": {"median": 2_000_000_000}},
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


def test_separated_extrapolation_conserves_components_and_moves_down():
    extrapolation = _extrapolation()
    expectations = json.loads(
        (STUDY_DIR / "core61_depth_expectations.json").read_text(encoding="utf-8")
    )
    access = {
        "status": "PASS",
        "whole_record_loaded": False,
        "unselected_values_decoded": False,
        "selector": "/entries[7]",
        "record_sha256_from_published_manifest": "fixture-sha256",
    }

    result = extrapolation.derive_result(
        expectations,
        _selected_entry(),
        access,
        expectations_commit="freeze-commit",
    )

    decomposition = result["decomposition"]
    assert decomposition["four_layer_measured_service_ps"] == 1_875_680_000
    assert decomposition["per_step_fixed_ps"] == 500
    assert decomposition["four_layer_repeatable_ps"] == 1_875_679_500
    declared = result["declared_61_layer_step"]
    assert declared["linear_rule"]["published_ps"] == 28_604_120_000
    assert declared["separated_rule"]["published_ps"] == 28_604_112_875
    assert declared["signed_movement_separated_minus_linear"]["published_ps"] == -7_125
    assert result["held_out_depth_prediction"]["published_ps"] == 3_751_359_500
    assert result["evidence_class"]["validated_depth_rule"] is False
    assert result["registry"]["core61"] == "OPEN"


def test_separated_extrapolation_rejects_nonconserving_components():
    extrapolation = _extrapolation()
    expectations = json.loads(
        (STUDY_DIR / "core61_depth_expectations.json").read_text(encoding="utf-8")
    )
    basis = _selected_entry()
    basis["kernels"][0]["components"]["fixed_overhead_ps"] += 1
    access = {
        "status": "PASS",
        "whole_record_loaded": False,
        "unselected_values_decoded": False,
    }

    with pytest.raises(ValueError, match="do not reconstruct"):
        extrapolation.derive_result(
            expectations,
            basis,
            access,
            expectations_commit="freeze-commit",
        )


def test_scope_locks_keep_scored_runs_comp76_and_code_untouched():
    expectations = json.loads(
        (STUDY_DIR / "core61_depth_expectations.json").read_text(encoding="utf-8")
    )

    assert set(expectations["preserved_scope"].values()) == {False}
    assert expectations["acceptance"]["reserved_residual_id"] == "CORE-63"


def test_published_result_is_the_exact_null_magnitude_derivation():
    result = json.loads(
        (STUDY_DIR / "core61_depth_result.json").read_text(encoding="utf-8")
    )

    assert result["status"] == "LOCAL_DERIVATION_COMPLETE_CORE61_OPEN"
    assert result["expectations_commit"] == "a6ba1461655ff4cca553658e613f589d705dc578"
    evidence = result["evidence_class"]
    assert evidence == {
        "description": "DECLARED derivation from a MEASURED decomposition at one depth",
        "measured_depth_count": 1,
        "service_class": "DECLARED",
        "source_component_class": "DISCLOSED",
        "source_service_class": "MEASURED",
        "validated_depth_rule": False,
    }

    decomposition = result["decomposition"]
    assert decomposition["per_step_fixed_ps"] == 489
    assert decomposition["four_layer_repeatable_ps"] == 1_875_679_511
    assert decomposition["per_layer_repeatable"]["numerator"] == 1_875_679_511
    assert decomposition["per_layer_repeatable"]["denominator"] == 4
    assert decomposition["reconstruction_error_ps"] == 0

    declared = result["declared_61_layer_step"]
    assert declared["linear_rule"]["published_ps"] == 28_604_120_000
    assert declared["separated_rule"]["exact"]["numerator"] == 114_416_452_127
    assert declared["separated_rule"]["exact"]["denominator"] == 4
    assert declared["separated_rule"]["published_ps"] == 28_604_113_032
    movement = declared["signed_movement_separated_minus_linear"]
    assert (movement["exact"]["numerator"], movement["exact"]["denominator"]) == (
        -27_873,
        4,
    )
    assert movement["published_ps"] == -6_968
    assert movement["absolute_share_of_linear_ppm"] == "0.243610"

    assert result["held_out_depth_prediction"]["published_ps"] == 3_751_359_511
    assert result["held_out_depth_prediction"]["measured_service_ps"] is None
    assert result["comparison_context"]["used_as_arithmetic_input"] is False
    assert result["comparison_context"]["implied_step_displayed"] is False


def test_published_access_and_candidate_key_preserve_the_freeze():
    result = json.loads(
        (STUDY_DIR / "core61_depth_result.json").read_text(encoding="utf-8")
    )
    source = result["source"]
    access = source["access"]

    assert access["status"] == "PASS"
    assert access["bytes_consumed"] == 21_700
    assert access["fields"] == list(_reader().ALLOWED_FIELDS)
    assert access["whole_record_loaded"] is False
    assert access["unselected_values_decoded"] is False
    key = source["candidate_key"]
    assert key["parallelism"] == {
        "data_parallel": 1,
        "expert_parallel": 1,
        "pipeline_parallel": 1,
        "tensor_parallel": 1,
    }
    assert key["shape"] == {
        "batch_size": 32,
        "per_request_kv_lengths": [2000] * 32,
    }
    assert key["routing"]["availability"] == "not-captured"


def test_registry_records_local_movement_and_exact_merlin_remainder():
    core = (REPOSITORY_ROOT / "docs/modules/core.md").read_text(encoding="utf-8")
    compute = (REPOSITORY_ROOT / "docs/modules/compute.md").read_text(encoding="utf-8")

    assert "- CORE-61 (Precision; P1; M):" in core
    assert "CORE-61 local derivation" in core
    assert "CORE-61 stays" in core
    assert "CORE-63 remains reserved" in core
    assert "REDUCED_LAYERS=8" in compute
    assert "--job-name=gh-core61-d8-base" in compute
    assert "--job-name=gh-core61-d8-decode" in compute
    assert "2026-08-28T06:30" in compute
