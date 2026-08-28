from __future__ import annotations

import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path

import pytest

STUDY_DIR = (
    Path(__file__).resolve().parents[1] / "examples/deployment_curve_v1"
)
sys.path.insert(0, str(STUDY_DIR))

from core63_clean_residency import (
    independent_recompute,
    validate_access_events,
    validate_clean_expectations,
    validate_final_retry_expectations,
    validate_preflight_events,
    validate_registry_preflight_events,
    validate_registry_retry_expectations,
    validate_retry_expectations,
)


def _fraction(row: dict[str, int]) -> Fraction:
    return Fraction(row["numerator"], row["denominator"])


def test_committed_clean_expectations_validate() -> None:
    expectations = json.loads(
        (STUDY_DIR / "core63_clean_expectations.json").read_text(encoding="utf-8")
    )

    validate_clean_expectations(expectations)


def test_committed_retry_and_preflight_validate() -> None:
    retry = json.loads(
        (STUDY_DIR / "core63_clean_retry_expectations.json").read_text(
            encoding="utf-8"
        )
    )
    events = [
        json.loads(line)
        for line in (STUDY_DIR / "core63_clean_access_ledger.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    validate_retry_expectations(retry)
    result = validate_preflight_events(events)

    assert result["end_statuses"] == ["PASS", "PASS", "REJECTED"]
    assert result["forbidden_access_ledger"] == []
    assert result["rejected_before_final_byte"] is True


def test_committed_final_retry_and_sparse_preflight_validate() -> None:
    final_retry = json.loads(
        (STUDY_DIR / "core63_clean_final_retry_expectations.json").read_text(
            encoding="utf-8"
        )
    )
    events = [
        json.loads(line)
        for line in (STUDY_DIR / "core63_clean_access_ledger_retry.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    validate_final_retry_expectations(final_retry)
    result = validate_preflight_events(events)

    assert result["end_statuses"] == ["PASS", "PASS", "REJECTED"]
    assert result["whole_file_streams"] == 0


def test_committed_registry_retry_and_preflight_validate() -> None:
    registry_retry = json.loads(
        (STUDY_DIR / "core63_clean_registry_retry_expectations.json").read_text(
            encoding="utf-8"
        )
    )
    events = [
        json.loads(line)
        for line in (STUDY_DIR / "core63_clean_access_ledger_final.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    validate_registry_retry_expectations(registry_retry)
    result = validate_registry_preflight_events(events)

    assert result["end_statuses"] == ["PASS", "PASS", "PASS", "REJECTED"]
    assert result["forbidden_access_ledger"] == []


def test_access_validation_requires_partial_contemporaneous_pairs() -> None:
    events = []
    for access_number in range(1, 7):
        common = {
            "access_id": f"A{access_number:02d}",
            "classification": "synthetic",
            "held_out_mtp_value_accessed": False,
            "record": f"record-{access_number}",
            "record_size_bytes": 100,
            "schema": "simllm-deployment-curve-core63-clean-access-v1",
            "selector": f"/field-{access_number}",
            "whole_file_streamed": False,
        }
        events.append(
            {
                **common,
                "bytes_accessed": 0,
                "event": "BEGIN",
                "event_index": len(events) + 1,
                "status": "IN_PROGRESS",
            }
        )
        events.append(
            {
                **common,
                "bytes_accessed": 50,
                "event": "END",
                "event_index": len(events) + 1,
                "status": "PASS",
            }
        )

    result = validate_access_events(events)

    assert result["access_count"] == 6
    assert result["forbidden_access_ledger"] == []
    assert result["whole_file_streams"] == 0


def test_access_validation_rejects_full_byte_count() -> None:
    events = []
    for access_number in range(1, 7):
        common = {
            "access_id": f"A{access_number:02d}",
            "classification": "synthetic",
            "held_out_mtp_value_accessed": False,
            "record": f"record-{access_number}",
            "record_size_bytes": 100,
            "schema": "simllm-deployment-curve-core63-clean-access-v1",
            "selector": f"/field-{access_number}",
            "whole_file_streamed": False,
        }
        events.extend(
            [
                {
                    **common,
                    "bytes_accessed": 0,
                    "event": "BEGIN",
                    "event_index": len(events) + 1,
                    "status": "IN_PROGRESS",
                },
                {
                    **common,
                    "bytes_accessed": 100 if access_number == 6 else 50,
                    "event": "END",
                    "event_index": len(events) + 2,
                    "status": "PASS",
                },
            ]
        )

    with pytest.raises(ValueError, match="final source byte"):
        validate_access_events(events)


def test_independent_recompute_uses_only_routed_marker() -> None:
    inputs = {
        "calibration_context": {
            "current_step_ps": 1_000_000,
            "per_node_tokens": 1,
            "published_tokens_per_second_per_node": 2_000_000,
        },
        "component_basis": {
            "kernels": [{"components": {"fixed_overhead_ps": 100}}],
            "measured_service_ps": 2100,
        },
        "kernel_rows": [
            {
                "first_launch_order": "1",
                "name": "attention_kernel",
                "total_duration_per_step_ns": "1.5",
            },
            {
                "first_launch_order": "2",
                "name": "Fused_MoE_Kernel_stage",
                "total_duration_per_step_ns": "0.6",
            },
        ],
    }

    result = independent_recompute(inputs)

    assert [
        row["family"] for row in result["component_classification_ledger"]
    ] == ["retained", "routed_expert"]
    assert _fraction(result["retained_service_ps"]) == 1400
    assert _fraction(result["routed_service_ps"]) == 600
    assert _fraction(result["step"]["corrected_ps"]) == Fraction(67400, 3)


def test_access_ledger_manifest_binds_every_append_only_tranche() -> None:
    manifest = json.loads(
        (STUDY_DIR / "core63_clean_access_ledger_manifest.json").read_text(
            encoding="utf-8"
        )
    )

    assert manifest["forbidden_access_ledger"] == []
    assert manifest["total_access_count"] == 20
    assert manifest["total_event_count"] == 40
    assert manifest["whole_file_streams"] == 0
    repository_root = STUDY_DIR.parents[1]
    for row in manifest["ledgers"]:
        payload = (repository_root / row["path"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == row["sha256"]


def test_registry_movement_closes_only_core63_condition() -> None:
    movement = json.loads(
        (STUDY_DIR / "core63_clean_registry_movement.json").read_text(
            encoding="utf-8"
        )
    )
    core_doc = (STUDY_DIR.parents[1] / "docs/modules/core.md").read_text(
        encoding="utf-8"
    )
    open_tasks = core_doc.split("## Open tasks", 1)[1]

    assert movement["forbidden_access_ledger"] == []
    assert movement["core63"]["movement"] == (
        "COMPLETE_CLEAN_REPRODUCTION_ACCEPTED"
    )
    assert movement["core64"]["movement"] == (
        "OPEN_UNCONDITIONAL_ATTENTION_MLA_FAMILY_GAP"
    )
    assert movement["core65"] == {"free_on_base_main": True, "reserved": True}
    assert "- CORE-63 (Precision" not in open_tasks
    assert "- CORE-64 (Precision" in open_tasks
    assert "now unconditionally promoted by the clean CORE-63" in open_tasks
