"""Strict immutable envelopes for content-addressed calibration records."""

from __future__ import annotations

import enum
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .canonical import (
    CanonicalError,
    canonical_bytes,
    canonical_loads,
    normalize_json,
    sha256_bytes,
)


class RecordIntent(str, enum.Enum):
    """External closure layer for a record object.

    Intent is manifest metadata, not a member injected into authoritative
    record bytes.  The three values mirror the frozen evidence to fit to
    release dependency direction without defining another public wire schema.
    """

    EVIDENCE = "evidence"
    FIT = "fit"
    RELEASE = "release"


def parse_record_intent(value: Any, path: str = "intent") -> RecordIntent:
    """Parse one closed record intent without enum coercion surprises."""

    if not isinstance(value, str):
        raise CanonicalError(f"{path}: expected a string record intent")
    try:
        return RecordIntent(value)
    except ValueError as error:
        choices = ", ".join(intent.value for intent in RecordIntent)
        raise CanonicalError(f"{path}: expected one of {choices}; found {value!r}") from error


def validate_schema_id(value: Any, path: str = "schema") -> str:
    """Validate the required schema member shared by calibration records."""

    if not isinstance(value, str):
        raise CanonicalError(f"{path}: expected a string")
    normalized = normalize_json(value)
    if normalized != value:
        raise CanonicalError(f"{path}: expected NFC-normalized text")
    if not value or value.strip() != value:
        raise CanonicalError(f"{path}: expected a nonblank string without edge whitespace")
    if any(ord(character) < 0x20 for character in value):
        raise CanonicalError(f"{path}: control characters are forbidden")
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class RecordObject:
    """One strict top-level record plus its external content identity."""

    value: Mapping[str, Any]
    canonical: bytes
    record_id: str
    schema: str

    @classmethod
    def from_value(cls, value: Mapping[str, Any]) -> RecordObject:
        """Validate and freeze a JSON-ready record mapping."""

        normalized = normalize_json(value)
        if not isinstance(normalized, dict):  # pragma: no cover - Mapping input guarantees this
            raise CanonicalError("record: expected a top-level object")
        return cls._from_normalized(normalized)

    @classmethod
    def from_bytes(
        cls,
        raw: str | bytes | bytearray | memoryview,
        *,
        expected_schema: str | None = None,
    ) -> RecordObject:
        """Load one byte-for-byte canonical top-level record."""

        normalized = canonical_loads(raw)
        if not isinstance(normalized, dict):
            raise CanonicalError("record: expected a top-level JSON object")
        record = cls._from_normalized(normalized)
        if expected_schema is not None:
            record.require_schema(expected_schema)
        return record

    @classmethod
    def _from_normalized(cls, normalized: dict[str, Any]) -> RecordObject:
        if "record_id" in normalized:
            raise CanonicalError(
                "record.record_id: content identity is external and cannot be a JSON member"
            )
        if "schema" not in normalized:
            raise CanonicalError("record.schema: missing required field")
        schema = validate_schema_id(normalized["schema"], "record.schema")
        encoded = canonical_bytes(normalized)
        frozen = _freeze(normalized)
        return cls(
            value=frozen,
            canonical=encoded,
            record_id=sha256_bytes(encoded),
            schema=schema,
        )

    def require_schema(self, expected: str) -> RecordObject:
        """Return this object after an exact expected-schema check."""

        expected_schema = validate_schema_id(expected, "expected_schema")
        if self.schema != expected_schema:
            raise CanonicalError(
                f"record.schema: expected {expected_schema!r}; found {self.schema!r}"
            )
        return self


def record_object(value: Mapping[str, Any] | str | bytes | bytearray | memoryview) -> RecordObject:
    """Construct a :class:`RecordObject` from a mapping or canonical bytes."""

    if isinstance(value, Mapping):
        return RecordObject.from_value(value)
    return RecordObject.from_bytes(value)


__all__ = [
    "RecordIntent",
    "RecordObject",
    "parse_record_intent",
    "record_object",
    "validate_schema_id",
]
