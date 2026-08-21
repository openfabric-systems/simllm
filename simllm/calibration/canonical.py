"""Canonical JSON bytes for offline calibration records.

This module is intentionally independent from the repository's historical
JSON writers.  Calibration identities need a full-Unicode, lossless integer
contract and must reject alternate encodings before content is admitted to an
object store.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import unicodedata
from collections.abc import Mapping
from typing import Any

CANONICAL_SCHEMA = "simllm-calibration-canonical-bytes-v1"
CANONICAL_PYTHON = (3, 10)
CANONICAL_UNICODE_DATABASE = "13.0.0"

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_SHORT_ESCAPES = {
    '"': '\\"',
    "\\": "\\\\",
    "\b": "\\b",
    "\t": "\\t",
    "\n": "\\n",
    "\f": "\\f",
    "\r": "\\r",
}


class CanonicalError(ValueError):
    """A value or byte sequence is outside the calibration JSON contract."""


def assert_canonical_runtime() -> None:
    """Reject a runtime that cannot claim the frozen full-Unicode identity."""

    runtime = (sys.version_info.major, sys.version_info.minor)
    if sys.implementation.name != "cpython" or runtime != CANONICAL_PYTHON:
        raise RuntimeError(
            "calibration canonicalization requires CPython 3.10; "
            f"found {sys.implementation.name} {runtime[0]}.{runtime[1]}"
        )
    if unicodedata.unidata_version != CANONICAL_UNICODE_DATABASE:
        raise RuntimeError(
            "calibration canonicalization requires Unicode database "
            f"{CANONICAL_UNICODE_DATABASE}; found {unicodedata.unidata_version}"
        )


def _normalize_string(value: str, path: str) -> str:
    for character in value:
        scalar = ord(character)
        if 0xD800 <= scalar <= 0xDFFF:
            raise CanonicalError(f"{path}: unpaired Unicode surrogate is forbidden")
        if scalar > 0x7F:
            assert_canonical_runtime()
    return unicodedata.normalize("NFC", value)


def _normalize(value: Any, path: str, active_containers: set[int]) -> Any:
    if value is None or type(value) is bool or type(value) is int:
        return value
    if isinstance(value, str):
        return _normalize_string(value, path)
    if isinstance(value, float):
        raise CanonicalError(f"{path}: floating-point values are forbidden")
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active_containers:
            raise CanonicalError(f"{path}: cyclic object is forbidden")
        active_containers.add(identity)
        try:
            result: dict[str, Any] = {}
            normalized_sources: dict[str, str] = {}
            for raw_key, item in value.items():
                if not isinstance(raw_key, str):
                    raise CanonicalError(f"{path}: object key {raw_key!r} is not a string")
                key = _normalize_string(raw_key, f"{path} key")
                if key in result:
                    previous = normalized_sources[key]
                    raise CanonicalError(
                        f"{path}: object keys {previous!r} and {raw_key!r} "
                        "collide after NFC normalization"
                    )
                normalized_sources[key] = raw_key
                result[key] = _normalize(item, f"{path}.{key}", active_containers)
            return result
        finally:
            active_containers.remove(identity)
    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in active_containers:
            raise CanonicalError(f"{path}: cyclic array is forbidden")
        active_containers.add(identity)
        try:
            return [
                _normalize(item, f"{path}[{index}]", active_containers)
                for index, item in enumerate(value)
            ]
        finally:
            active_containers.remove(identity)
    raise CanonicalError(f"{path}: {type(value).__name__} is outside the canonical JSON domain")


def normalize_json(value: Any) -> Any:
    """Return the NFC-normalized JSON value, rejecting unsupported values."""

    return _normalize(value, "$", set())


def _escape_string(value: str) -> str:
    pieces = ['"']
    for character in value:
        replacement = _SHORT_ESCAPES.get(character)
        if replacement is not None:
            pieces.append(replacement)
            continue
        scalar = ord(character)
        if scalar < 0x20:
            pieces.append(f"\\u{scalar:04x}")
        else:
            pieces.append(character)
    pieces.append('"')
    return "".join(pieces)


def _render(value: Any, pieces: list[str]) -> None:
    if value is None:
        pieces.append("null")
    elif type(value) is bool:
        pieces.append("true" if value else "false")
    elif type(value) is int:
        pieces.append(str(value))
    elif isinstance(value, str):
        pieces.append(_escape_string(value))
    elif isinstance(value, list):
        pieces.append("[")
        for index, item in enumerate(value):
            if index:
                pieces.append(",")
            _render(item, pieces)
        pieces.append("]")
    elif isinstance(value, dict):
        pieces.append("{")
        for index, key in enumerate(sorted(value)):
            if index:
                pieces.append(",")
            pieces.append(_escape_string(key))
            pieces.append(":")
            _render(value[key], pieces)
        pieces.append("}")
    else:  # pragma: no cover - normalize_json owns this invariant
        raise AssertionError(f"unhandled normalized value {type(value).__name__}")


def canonical_bytes(value: Any) -> bytes:
    """Encode one value as deterministic UTF-8 calibration JSON without LF."""

    normalized = normalize_json(value)
    pieces: list[str] = []
    _render(normalized, pieces)
    try:
        return "".join(pieces).encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:  # pragma: no cover - scalar validation is prior
        raise CanonicalError("canonical value contains an invalid Unicode scalar") from error


def _parse_integer(token: str) -> int:
    if token == "-0":
        raise CanonicalError("$: negative-zero integer spelling is forbidden")
    return int(token)


def _reject_float(token: str) -> Any:
    raise CanonicalError(f"$: floating-point token {token!r} is forbidden")


def _reject_constant(token: str) -> Any:
    raise CanonicalError(f"$: nonfinite token {token!r} is forbidden")


def _pairs_to_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    raw_keys: set[str] = set()
    normalized_sources: dict[str, str] = {}
    for raw_key, value in pairs:
        if raw_key in raw_keys:
            raise CanonicalError(f"$: duplicate object key {raw_key!r}")
        raw_keys.add(raw_key)
        key = _normalize_string(raw_key, "$ key")
        if key in normalized_sources:
            previous = normalized_sources[key]
            raise CanonicalError(
                f"$: object keys {previous!r} and {raw_key!r} collide after NFC normalization"
            )
        normalized_sources[key] = raw_key
        result[key] = value
    return result


def _source_bytes(raw: str | bytes | bytearray | memoryview) -> bytes:
    if isinstance(raw, str):
        try:
            return raw.encode("utf-8", errors="strict")
        except UnicodeEncodeError as error:
            raise CanonicalError("input contains an invalid Unicode scalar") from error
    if isinstance(raw, bytes):
        return raw
    if isinstance(raw, (bytearray, memoryview)):
        return bytes(raw)
    raise TypeError("raw JSON must be str or bytes-like")


def strict_json_loads(raw: str | bytes | bytearray | memoryview) -> Any:
    """Parse strict calibration JSON and return its normalized Python value.

    This parser accepts insignificant JSON whitespace and noncanonical key
    order so diagnostics can distinguish syntax from canonical-byte failures.
    Use :func:`canonical_loads` at a content-identity boundary.
    """

    source = _source_bytes(raw)
    if source.startswith(b"\xef\xbb\xbf"):
        raise CanonicalError("UTF-8 byte-order marks are forbidden")
    try:
        text = source.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise CanonicalError("input is not strict UTF-8") from error
    if text.startswith("\ufeff"):
        raise CanonicalError("Unicode byte-order marks are forbidden")
    try:
        value = json.loads(
            text,
            object_pairs_hook=_pairs_to_object,
            parse_int=_parse_integer,
            parse_float=_reject_float,
            parse_constant=_reject_constant,
            strict=True,
        )
    except json.JSONDecodeError as error:
        raise CanonicalError(
            f"invalid JSON at line {error.lineno}, column {error.colno}: {error.msg}"
        ) from error
    return _normalize(value, "$", set())


def canonical_loads(raw: str | bytes | bytearray | memoryview) -> Any:
    """Parse a byte-for-byte canonical calibration JSON value."""

    source = _source_bytes(raw)
    value = strict_json_loads(source)
    expected = canonical_bytes(value)
    if source != expected:
        raise CanonicalError("input is valid JSON but is not canonical calibration bytes")
    return value


def sha256_bytes(data: bytes | bytearray | memoryview) -> str:
    """Return lowercase SHA-256 for exact bytes without any transformation."""

    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("SHA-256 input must be bytes-like")
    return hashlib.sha256(bytes(data)).hexdigest()


def canonical_sha256(value: Any) -> str:
    """Return the external identity of one canonical record value."""

    return sha256_bytes(canonical_bytes(value))


def validate_sha256(value: Any, path: str = "record_id") -> str:
    """Validate and return a lowercase hexadecimal SHA-256 identity."""

    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise CanonicalError(f"{path}: expected 64 lowercase hexadecimal digits")
    return value


__all__ = [
    "CANONICAL_PYTHON",
    "CANONICAL_SCHEMA",
    "CANONICAL_UNICODE_DATABASE",
    "CanonicalError",
    "assert_canonical_runtime",
    "canonical_bytes",
    "canonical_loads",
    "canonical_sha256",
    "normalize_json",
    "sha256_bytes",
    "strict_json_loads",
    "validate_sha256",
]
