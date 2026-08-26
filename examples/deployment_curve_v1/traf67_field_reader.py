"""Field-addressed reader for TRAF-67's single visible COMP-75 row."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, BinaryIO

ACCESS_SCHEMA = "simllm-deployment-curve-traf67-access-v1"
VISIBLE_ANCHOR_ID = "sglang_prefill_1k"
ROW_FIELD = "calibration_rows"
ID_FIELD = "anchor_id"
STUDY_DIR = Path(__file__).resolve().parent
ALLOWED_RECORD = STUDY_DIR / "comp75_calibration_result.json"
RECORD_LABEL = "examples/deployment_curve_v1/comp75_calibration_result.json"
SELECTOR = "/calibration_rows[anchor_id=sglang_prefill_1k]"
_WHITESPACE = b" \t\r\n"
_DELIMITERS = _WHITESPACE + b",]}"


class _Cursor:
    """One-byte cursor that never loads or returns the whole record."""

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
        raise ValueError(
            f"expected JSON token {expected!r}, observed {observed!r}"
        )


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
    raw = _read_string_bytes(cursor, first)
    value = json.loads(raw.decode("utf-8"))
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


def _capture_value(cursor: _Cursor, first: bytes) -> bytes:
    if first not in (b"{", b"["):
        raise ValueError("TRAF-67 visible row must be a JSON object")
    value = bytearray(first)
    nesting = [first]
    in_string = False
    escaped = False
    while nesting:
        current = cursor.read()
        if not current:
            raise ValueError("unterminated selected JSON value")
        value.extend(current)
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
    return bytes(value)


def extract_visible_row(stream: BinaryIO) -> tuple[dict[str, Any], int]:
    """Extract only the fixed visible row and stop before later record fields."""

    cursor = _Cursor(stream)
    _expect(cursor, b"{")
    current = _skip_space(cursor)
    while current != b"}":
        key = _read_key(cursor, current)
        _expect(cursor, b":")
        if key == ROW_FIELD:
            _expect(cursor, b"[")
            first = _skip_space(cursor)
            if first == b"]":
                raise ValueError("visible calibration row is missing")
            raw = _capture_value(cursor, first)
            row = json.loads(raw.decode("utf-8"))
            if not isinstance(row, dict):
                raise TypeError("visible calibration row must be an object")
            if row.get(ID_FIELD) != VISIBLE_ANCHOR_ID:
                raise ValueError("first calibration row is not the permitted 1K row")
            closing = _skip_space(cursor)
            if closing != b"]":
                raise ValueError("record contains another calibration row")
            return row, cursor.bytes_consumed
        _skip_value(cursor)
        current = _skip_space(cursor)
        if current == b"}":
            break
        if current != b",":
            raise ValueError("malformed top-level JSON object")
        current = _skip_space(cursor)
    raise ValueError("calibration_rows field is missing")


def _append_access(log_path: Path, entry: Mapping[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(dict(entry), sort_keys=True) + "\n")


def read_visible_comp75_row(record_path: Path, access_log: Path) -> dict[str, Any]:
    """Read the sole permitted record row and append one auditable access."""

    entry: dict[str, Any] = {
        "schema": ACCESS_SCHEMA,
        "classification": "visible_calibration",
        "record": RECORD_LABEL,
        "selector": SELECTOR,
        "whole_record_loaded": False,
    }
    try:
        if record_path.resolve() != ALLOWED_RECORD.resolve():
            raise ValueError("TRAF-67 reader refuses every non-allowlisted record")
        with record_path.open("rb", buffering=0) as stream:
            row, consumed = extract_visible_row(stream)
        entry.update({"bytes_consumed": consumed, "status": "PASS"})
        return row
    except Exception as exc:
        entry.update({"error": type(exc).__name__, "status": "REJECTED"})
        raise
    finally:
        _append_access(access_log, entry)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--access-log", required=True, type=Path)
    args = parser.parse_args()
    row = read_visible_comp75_row(ALLOWED_RECORD, args.access_log)
    print(json.dumps(row, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
