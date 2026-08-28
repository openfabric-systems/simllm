from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "examples/deployment_curve_v1"


def _load(name: str) -> dict:
    return json.loads((STUDY / name).read_text(encoding="utf-8"))


def test_protocol_void_result_never_claims_a_capture_or_movement() -> None:
    result = _load("core66_capture_result.json")

    assert result["status"] == "PROTOCOL_VOID_NO_HARDWARE_SUBMISSION"
    assert result["achieved_capture_configuration"] is None
    assert result["hardware"] == {
        "feasible_cell_submission_count": 0,
        "feasible_cell_status": "NOT_SUBMITTED_PROTOCOL_VOID_BEFORE_ALLOCATION",
        "registered_ep72_status": "BLOCKED_IMPOSSIBLE_ON_PROJECT_CLUSTER",
        "shared_gh200_gpu_hours_consumed": 0,
    }
    assert result["calibration_only"]["signed_movement_tokens_per_second_per_node"] is None
    assert result["calibration_only"]["downward_correction_published_alone"] is False
    assert result["identity_and_physics"]["physical_identity_binding_count"] == 0
    assert result["identity_and_physics"]["deep_ep_dispatch_launches"] is None
    assert result["identity_and_physics"]["deep_ep_combine_launches"] is None
    assert result["identity_and_physics"]["hbm_read_write_bytes"] is None


def test_forbidden_access_incidents_are_disclosed_and_not_used() -> None:
    ledger = _load("core66_forbidden_access_ledger.json")
    result = _load("core66_capture_result.json")

    assert ledger["literal_empty_ledger_requirement_met"] is False
    assert ledger["incident_count"] == 4
    assert len(ledger["incidents"]) == 4
    assert all(not row["used_in_core66_arithmetic"] for row in ledger["incidents"])
    assert result["access"]["forbidden_access_ledger_empty"] is False
    assert result["access"]["forbidden_access_incident_count"] == 4
    assert result["access"]["held_out_mtp_numeric_value_exposed_in_worker_output"] is False


def test_bounded_reader_tranches_have_empty_reader_forbidden_ledgers() -> None:
    manifest = _load("core66_reader_access_manifest.json")

    assert manifest["all_reader_forbidden_ledgers_empty"] is True
    assert manifest["reader_access_event_count"] == 10
    assert len(manifest["access_tranches"]) == 4
    for row in manifest["access_tranches"]:
        assert row["forbidden_ledger_bytes"] == 0
        assert row["forbidden_ledger_sha256"] == (
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        )


def test_feasible_freeze_and_deviations_are_explicit() -> None:
    result = _load("core66_capture_result.json")
    capture = result["feasible_capture_freeze"]

    assert capture["expert_parallel_width"] == 12
    assert capture["node_count"] == 3
    assert capture["gpus_per_node"] == 4
    assert capture["logical_experts_per_rank"] == 4
    assert capture["routed_expert_total"] == 48
    assert capture["captured_layer_count"] == 4
    assert capture["batch_per_rank"] == 32
    assert capture["kv_length_per_rank"] == 2000
    assert capture["mtp_enabled"] is False
    assert capture["measured_decode_iterations"] == 1
    assert len(result["declared_deviation_ledger"]) == 5
    assert result["scale_checks"] == {
        "assignment_scale_1_over_9": "UNAVAILABLE_NO_ROUTING_CAPTURE",
        "count_and_weight_scale_1_over_64": "UNAVAILABLE_NO_HBM_OR_ROUTING_CAPTURE",
    }
