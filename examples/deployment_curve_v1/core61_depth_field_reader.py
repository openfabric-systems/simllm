"""Field-addressed reader for the retained CORE-61 four-layer decode basis."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, BinaryIO

ACCESS_SCHEMA = "simllm-deployment-curve-core61-depth-access-v1"
SELECTED_ENTRY_INDEX = 7
EXPECTED_IMPLEMENTATION_ID = "deepseek-v3-reduced4-vllm-ep72-decode-b32-c2000"
STUDY_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = STUDY_DIR.parents[1]
ALLOWED_RECORD = (
    REPOSITORY_ROOT / "examples/hopper_kernel_cycle_candidate_v1/candidate-record.json"
)
RECORD_LABEL = "examples/hopper_kernel_cycle_candidate_v1/candidate-record.json"
RECORD_SHA256 = "ff46f6d8a79ddae899da89d4db6eb34373f8042acd06cab50b6336c8fb9a8f52"
SELECTOR = "/entries[7]"
_WHITESPACE = b" \t\r\n"
_DELIMITERS = _WHITESPACE + b",]}"
_CAPTURE = object()

ENTRY_PROJECTION: dict[str, Any] = {
    "coverage": _CAPTURE,
    "evidence": {
        "component_class": _CAPTURE,
        "service_class": _CAPTURE,
    },
    "implementation_id": _CAPTURE,
    "kernels": {
        0: {
            "components": {
                "compute_sm_cycles": _CAPTURE,
                "fixed_overhead_ps": _CAPTURE,
                "memory": {"service_ps": _CAPTURE},
                "method": _CAPTURE,
            },
            "kernel_id": _CAPTURE,
            "launch_count": _CAPTURE,
            "measured_elapsed_ps": _CAPTURE,
        }
    },
    "key": _CAPTURE,
    "measured_service_ps": _CAPTURE,
    "observed_clocks": {"sm_hz": {"median": _CAPTURE}},
}

ALLOWED_FIELDS = (
    "/entries[7]/coverage",
    "/entries[7]/evidence/component_class",
    "/entries[7]/evidence/service_class",
    "/entries[7]/implementation_id",
    "/entries[7]/kernels[0]/components/compute_sm_cycles",
    "/entries[7]/kernels[0]/components/fixed_overhead_ps",
    "/entries[7]/kernels[0]/components/memory/service_ps",
    "/entries[7]/kernels[0]/components/method",
    "/entries[7]/kernels[0]/kernel_id",
    "/entries[7]/kernels[0]/launch_count",
    "/entries[7]/kernels[0]/measured_elapsed_ps",
    "/entries[7]/key",
    "/entries[7]/measured_service_ps",
    "/entries[7]/observed_clocks/sm_hz/median",
)


class _Cursor:
    """One-byte cursor that never loads the whole record."""

    def __init__(self, stream: BinaryIO) -> None:
        self._stream = stream
        self._pending: bytes | None = None
        self.bytes_consumed = 0

    def read(self) -> bytes:
        if self._pending is not None:
            value = self._pending
            self._pending = None
        else:
            value = self._stream.read(1)
        if value:
            self.bytes_consumed += 1
        return value

    def peek(self) -> bytes:
        if self._pending is None:
            self._pending = self._stream.read(1)
        return self._pending


def _skip_space(cursor: _Cursor) -> bytes:
    value = cursor.read()
    while value and value in _WHITESPACE:
        value = cursor.read()
    return value


def _expect(cursor: _Cursor, expected: bytes) -> None:
    observed = _skip_space(cursor)
    if observed != expected:
        raise ValueError(f"expected JSON token {expected!r}, observed {observed!r}")


def _read_string_bytes(cursor: _Cursor, first: bytes | None = None) -> bytes:
    opening = _skip_space(cursor) if first is None else first
    if opening != b'"':
        raise ValueError("expected a JSON string")
    value = bytearray(opening)
    escaped = False
    while True:
        current = cursor.read()
        if not current:
            raise ValueError("unterminated JSON string")
        value.extend(current)
        if escaped:
            escaped = False
        elif current == b"\\":
            escaped = True
        elif current == b'"':
            return bytes(value)


def _read_key(cursor: _Cursor, first: bytes | None = None) -> str:
    value = json.loads(_read_string_bytes(cursor, first).decode("utf-8"))
    if not isinstance(value, str):
        raise TypeError("JSON object key must be a string")
    return value


def _skip_scalar(cursor: _Cursor, first: bytes) -> None:
    token = bytearray(first)
    while cursor.peek() and cursor.peek() not in _DELIMITERS:
        token.extend(cursor.read())
    json.loads(token.decode("utf-8"))


def _skip_value(cursor: _Cursor, first: bytes | None = None) -> None:
    opening = _skip_space(cursor) if first is None else first
    if not opening:
        raise ValueError("missing JSON value")
    if opening == b'"':
        _read_string_bytes(cursor, opening)
        return
    if opening == b"{":
        current = _skip_space(cursor)
        if current == b"}":
            return
        while True:
            _read_key(cursor, current)
            _expect(cursor, b":")
            _skip_value(cursor)
            current = _skip_space(cursor)
            if current == b"}":
                return
            if current != b",":
                raise ValueError("malformed JSON object")
            current = _skip_space(cursor)
    elif opening == b"[":
        current = _skip_space(cursor)
        if current == b"]":
            return
        while True:
            _skip_value(cursor, current)
            current = _skip_space(cursor)
            if current == b"]":
                return
            if current != b",":
                raise ValueError("malformed JSON array")
            current = _skip_space(cursor)
    else:
        _skip_scalar(cursor, opening)


def _capture_value(cursor: _Cursor, first: bytes) -> Any:
    if first == b'"':
        raw = _read_string_bytes(cursor, first)
    elif first in (b"{", b"["):
        raw_value = bytearray(first)
        nesting = [first]
        in_string = False
        escaped = False
        while nesting:
            current = cursor.read()
            if not current:
                raise ValueError("unterminated selected JSON value")
            raw_value.extend(current)
            if in_string:
                if escaped:
                    escaped = False
                elif current == b"\\":
                    escaped = True
                elif current == b'"':
                    in_string = False
                continue
            if current == b'"':
                in_string = True
            elif current in (b"{", b"["):
                nesting.append(current)
            elif current in (b"}", b"]"):
                expected = b"{" if current == b"}" else b"["
                if nesting.pop() != expected:
                    raise ValueError("mismatched JSON container")
        raw = bytes(raw_value)
    else:
        raw_value = bytearray(first)
        while cursor.peek() and cursor.peek() not in _DELIMITERS:
            raw_value.extend(cursor.read())
        raw = bytes(raw_value)
    return json.loads(raw.decode("utf-8"))


def _project_object(cursor: _Cursor, projection: Mapping[str, Any], first: bytes) -> dict:
    if first != b"{":
        raise ValueError("selected field must be a JSON object")
    result = {}
    current = _skip_space(cursor)
    if current == b"}":
        return result
    while True:
        key = _read_key(cursor, current)
        _expect(cursor, b":")
        opening = _skip_space(cursor)
        if key in projection:
            result[key] = _project_value(cursor, projection[key], opening)
        else:
            _skip_value(cursor, opening)
        current = _skip_space(cursor)
        if current == b"}":
            return result
        if current != b",":
            raise ValueError("malformed selected JSON object")
        current = _skip_space(cursor)


def _project_array(cursor: _Cursor, projection: Mapping[int, Any], first: bytes) -> list:
    if first != b"[":
        raise ValueError("selected field must be a JSON array")
    result = []
    current = _skip_space(cursor)
    if current == b"]":
        return result
    index = 0
    while True:
        if index not in projection:
            raise ValueError("selected kernels array contains an unregistered kernel")
        result.append(_project_value(cursor, projection[index], current))
        current = _skip_space(cursor)
        if current == b"]":
            return result
        if current != b",":
            raise ValueError("malformed selected JSON array")
        index += 1
        current = _skip_space(cursor)


def _project_value(cursor: _Cursor, projection: Any, first: bytes) -> Any:
    if projection is _CAPTURE:
        return _capture_value(cursor, first)
    if not isinstance(projection, Mapping):
        raise TypeError("projection must be a mapping or capture marker")
    if all(isinstance(key, str) for key in projection):
        return _project_object(cursor, projection, first)
    if all(isinstance(key, int) for key in projection):
        return _project_array(cursor, projection, first)
    raise TypeError("projection keys must have one type")


def extract_depth_basis(stream: BinaryIO) -> tuple[dict[str, Any], int]:
    """Project the target entry's allowed fields and stop before later entries."""

    cursor = _Cursor(stream)
    _expect(cursor, b"{")
    current = _skip_space(cursor)
    while current != b"}":
        key = _read_key(cursor, current)
        _expect(cursor, b":")
        opening = _skip_space(cursor)
        if key == "entries":
            if opening != b"[":
                raise ValueError("entries must be an array")
            current = _skip_space(cursor)
            for index in range(SELECTED_ENTRY_INDEX + 1):
                if current == b"]":
                    raise ValueError("selected CORE-61 entry is missing")
                if index == SELECTED_ENTRY_INDEX:
                    selected = _project_object(cursor, ENTRY_PROJECTION, current)
                    if selected.get("implementation_id") != EXPECTED_IMPLEMENTATION_ID:
                        raise ValueError("selected CORE-61 implementation identity differs")
                    return selected, cursor.bytes_consumed
                _skip_value(cursor, current)
                delimiter = _skip_space(cursor)
                if delimiter != b",":
                    raise ValueError("selected CORE-61 entry index is missing")
                current = _skip_space(cursor)
        else:
            _skip_value(cursor, opening)
        current = _skip_space(cursor)
        if current == b"}":
            break
        if current != b",":
            raise ValueError("malformed top-level JSON object")
        current = _skip_space(cursor)
    raise ValueError("entries field is missing")


def _append_access(log_path: Path, entry: Mapping[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(dict(entry), sort_keys=True) + "\n")


def read_retained_depth_basis(record_path: Path, access_log: Path) -> dict[str, Any]:
    """Read the sole permitted projection and append one auditable access."""

    entry: dict[str, Any] = {
        "schema": ACCESS_SCHEMA,
        "classification": "retained_measured_decomposition",
        "record": RECORD_LABEL,
        "record_sha256_from_published_manifest": RECORD_SHA256,
        "selector": SELECTOR,
        "fields": list(ALLOWED_FIELDS),
        "whole_record_loaded": False,
        "unselected_values_decoded": False,
    }
    try:
        if record_path.resolve() != ALLOWED_RECORD.resolve():
            raise ValueError("CORE-61 reader refuses every non-allowlisted record")
        with record_path.open("rb", buffering=0) as stream:
            value, consumed = extract_depth_basis(stream)
        entry.update({"bytes_consumed": consumed, "status": "PASS"})
        return value
    except Exception as exc:
        entry.update({"error": type(exc).__name__, "status": "REJECTED"})
        raise
    finally:
        _append_access(access_log, entry)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--access-log", required=True, type=Path)
    args = parser.parse_args()
    value = read_retained_depth_basis(ALLOWED_RECORD, args.access_log)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
