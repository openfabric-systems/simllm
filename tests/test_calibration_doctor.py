from __future__ import annotations

from dataclasses import replace

import pytest

from simllm.calibration.doctor import (
    DOCTOR_SCHEMA,
    CapabilityDecision,
    DoctorRecord,
    DoctorState,
    doctor_record_from_obj,
)


def test_inert_blocked_doctor_record_is_strict_and_deterministic() -> None:
    record = DoctorRecord.blocked(
        reason="concrete CUDA, ROCm and simulator probes are not loaded"
    )

    assert record.to_obj() == {
        "schema": DOCTOR_SCHEMA,
        "state": "blocked",
        "capabilities": [],
        "reason_code": "no-concrete-backends",
        "reason": "concrete CUDA, ROCm and simulator probes are not loaded",
    }
    assert DoctorRecord.from_obj(record.to_obj()) == record
    assert doctor_record_from_obj(record.to_obj()) == record


def test_ready_record_requires_null_reason_and_ready_capabilities() -> None:
    ready = CapabilityDecision(
        capability_id="canonical-writer",
        state=DoctorState.READY,
        reason_code=None,
        reason=None,
    )
    record = DoctorRecord(
        state=DoctorState.READY,
        capabilities=(ready,),
        reason_code=None,
        reason=None,
    )
    assert DoctorRecord.from_obj(record.to_obj()) == record

    blocked = CapabilityDecision(
        capability_id="gpu",
        state=DoctorState.BLOCKED,
        reason_code="gpu-absent",
        reason="no GPU was requested or discovered",
    )
    with pytest.raises(ValueError, match="every capability"):
        replace(record, capabilities=(blocked,))
    with pytest.raises(ValueError, match="ready state requires null"):
        replace(record, reason_code="unexpected", reason="unexpected")


def test_nonready_state_requires_complete_reason_pair() -> None:
    with pytest.raises(ValueError, match="requires both reason fields"):
        DoctorRecord(
            state=DoctorState.BLOCKED,
            capabilities=(),
            reason_code="missing-reason",
            reason=None,
        )


def test_capabilities_are_sorted_unique_and_exact() -> None:
    first = CapabilityDecision(
        capability_id="a",
        state=DoctorState.NOT_APPLICABLE,
        reason_code="not-requested",
        reason="capability was not requested",
    )
    second = CapabilityDecision(
        capability_id="b",
        state=DoctorState.REJECTED,
        reason_code="unsupported-target",
        reason="target is outside the backend envelope",
    )
    record = DoctorRecord(
        state=DoctorState.REJECTED,
        capabilities=(first, second),
        reason_code="one-or-more-rejected",
        reason="at least one required capability was rejected",
    )
    assert DoctorRecord.from_obj(record.to_obj()) == record

    with pytest.raises(ValueError, match="must be sorted"):
        replace(record, capabilities=(second, first))
    with pytest.raises(ValueError, match="duplicate"):
        replace(record, capabilities=(first, first))


def test_doctor_reader_rejects_unknown_state_field_and_schema() -> None:
    payload = DoctorRecord.blocked(reason="no backend").to_obj()
    payload["device_uuid"] = "not-in-wave-1a"
    with pytest.raises(ValueError, match="unknown fields"):
        DoctorRecord.from_obj(payload)

    payload = DoctorRecord.blocked(reason="no backend").to_obj()
    payload["state"] = "unavailable"
    with pytest.raises(ValueError, match="expected one of"):
        DoctorRecord.from_obj(payload)

    with pytest.raises(ValueError, match="expected 'simllm-calibration-doctor-v1'"):
        doctor_record_from_obj({"schema": "unknown"})
