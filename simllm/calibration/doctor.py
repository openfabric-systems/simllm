"""Typed, hardware-independent doctor and capability records."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from simllm.compute.device_model import _array, _fields, _object, _require_schema, _string

DOCTOR_SCHEMA = "simllm-calibration-doctor-v1"


class DoctorState(str, Enum):
    """Closed preflight disposition from the Wave 0 capability matrix."""

    READY = "ready"
    BLOCKED = "blocked"
    NOT_APPLICABLE = "not-applicable"
    REJECTED = "rejected"


def _state(value: object, path: str) -> DoctorState:
    raw = _string(value, path)
    try:
        return DoctorState(raw)
    except ValueError as exc:
        choices = ", ".join(repr(member.value) for member in DoctorState)
        raise ValueError(f"{path}: expected one of {choices}") from exc


def _reason_pair(
    *, reason_code: object, reason: object, state: DoctorState, path: str
) -> tuple[str | None, str | None]:
    if state is DoctorState.READY:
        if reason_code is not None or reason is not None:
            raise ValueError(f"{path}: ready state requires null reason fields")
        return None, None
    if reason_code is None or reason is None:
        raise ValueError(f"{path}: non-ready state requires both reason fields")
    return _string(reason_code, f"{path}.reason_code"), _string(
        reason, f"{path}.reason"
    )


@dataclass(frozen=True, slots=True)
class CapabilityDecision:
    """One data-valued backend capability decision."""

    capability_id: str
    state: DoctorState
    reason_code: str | None
    reason: str | None

    def __post_init__(self) -> None:
        _string(self.capability_id, "CapabilityDecision.capability_id")
        if not isinstance(self.state, DoctorState):
            raise TypeError("CapabilityDecision.state: expected DoctorState")
        _reason_pair(
            reason_code=self.reason_code,
            reason=self.reason,
            state=self.state,
            path="CapabilityDecision",
        )

    def to_obj(self) -> dict[str, Any]:
        return {
            "capability_id": self.capability_id,
            "state": self.state.value,
            "reason_code": self.reason_code,
            "reason": self.reason,
        }

    @classmethod
    def from_obj(cls, value: object, path: str = "capability") -> CapabilityDecision:
        payload = _object(value, path)
        _fields(payload, path, {"capability_id", "state", "reason_code", "reason"})
        state = _state(payload["state"], f"{path}.state")
        reason_code, reason = _reason_pair(
            reason_code=payload["reason_code"],
            reason=payload["reason"],
            state=state,
            path=path,
        )
        return cls(
            capability_id=_string(
                payload["capability_id"], f"{path}.capability_id"
            ),
            state=state,
            reason_code=reason_code,
            reason=reason,
        )


@dataclass(frozen=True, slots=True)
class DoctorRecord:
    """One deterministic environment preflight result.

    Concrete CUDA and ROCm snapshots are intentionally absent from Wave 1A.
    Later backends add separately versioned records rather than widening this
    exact object.
    """

    state: DoctorState
    capabilities: tuple[CapabilityDecision, ...]
    reason_code: str | None
    reason: str | None
    schema: str = DOCTOR_SCHEMA

    def __post_init__(self) -> None:
        _require_schema(self.schema, DOCTOR_SCHEMA, "DoctorRecord.schema")
        if not isinstance(self.state, DoctorState):
            raise TypeError("DoctorRecord.state: expected DoctorState")
        if not isinstance(self.capabilities, tuple):
            raise TypeError("DoctorRecord.capabilities: in-memory contract requires a tuple")
        capability_ids: list[str] = []
        for index, capability in enumerate(self.capabilities):
            if not isinstance(capability, CapabilityDecision):
                raise TypeError(
                    f"DoctorRecord.capabilities[{index}]: expected CapabilityDecision"
                )
            capability_ids.append(capability.capability_id)
        if tuple(capability_ids) != tuple(sorted(capability_ids)):
            raise ValueError("DoctorRecord.capabilities: must be sorted by capability_id")
        if len(capability_ids) != len(set(capability_ids)):
            raise ValueError("DoctorRecord.capabilities: duplicate capability_id")
        _reason_pair(
            reason_code=self.reason_code,
            reason=self.reason,
            state=self.state,
            path="DoctorRecord",
        )
        if self.state is DoctorState.READY and any(
            capability.state is not DoctorState.READY
            for capability in self.capabilities
        ):
            raise ValueError(
                "DoctorRecord.state: ready requires every capability to be ready"
            )

    @classmethod
    def blocked(
        cls,
        *,
        reason: str,
        reason_code: str = "no-concrete-backends",
    ) -> DoctorRecord:
        """Build the inert Wave 1A record without probing the environment."""

        return cls(
            state=DoctorState.BLOCKED,
            capabilities=(),
            reason_code=reason_code,
            reason=reason,
        )

    def to_obj(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "state": self.state.value,
            "capabilities": [capability.to_obj() for capability in self.capabilities],
            "reason_code": self.reason_code,
            "reason": self.reason,
        }

    @classmethod
    def from_obj(cls, value: object, path: str = "doctor") -> DoctorRecord:
        payload = _object(value, path)
        _fields(payload, path, {"schema", "state", "capabilities", "reason_code", "reason"})
        _require_schema(payload["schema"], DOCTOR_SCHEMA, f"{path}.schema")
        state = _state(payload["state"], f"{path}.state")
        reason_code, reason = _reason_pair(
            reason_code=payload["reason_code"],
            reason=payload["reason"],
            state=state,
            path=path,
        )
        return cls(
            state=state,
            capabilities=tuple(
                CapabilityDecision.from_obj(item, f"{path}.capabilities[{index}]")
                for index, item in enumerate(
                    _array(payload["capabilities"], f"{path}.capabilities")
                )
            ),
            reason_code=reason_code,
            reason=reason,
        )


def doctor_record_from_obj(value: object) -> DoctorRecord:
    """Strict schema dispatcher used by the generic Wave 1A validator."""

    payload = _object(value, "record")
    if payload.get("schema") != DOCTOR_SCHEMA:
        raise ValueError(f"record.schema: expected {DOCTOR_SCHEMA!r}")
    return DoctorRecord.from_obj(payload, "record")


TYPED_RECORD_READERS = {DOCTOR_SCHEMA: DoctorRecord.from_obj}


def validate_typed_record(value: object) -> DoctorRecord:
    """Validate one doctor record through the generic Wave 1A surface."""

    return doctor_record_from_obj(value)


__all__ = [
    "DOCTOR_SCHEMA",
    "TYPED_RECORD_READERS",
    "CapabilityDecision",
    "DoctorRecord",
    "DoctorState",
    "doctor_record_from_obj",
    "validate_typed_record",
]
