"""Field-addressed reader for the guarded VLLM-41 result record."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, BinaryIO

ACCESS_SCHEMA = "simllm-pd-session-batching-service-access-v1"
STUDY_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = STUDY_DIR.parents[1]
ALLOWED_RECORD = (
    REPOSITORY_ROOT / "examples" / "pd_session_queue_onset_v1" / "results.json"
)
RECORD_LABEL = "examples/pd_session_queue_onset_v1/results.json"
FORBIDDEN_RECORD_LABEL = "examples/pd_session_queue_onset_v1/RESULTS.md"
MAX_SCALAR_BYTES = 4096
MAX_OBJECT_KEYS = 512
MAX_KEY_BYTES = 512
_WHITESPACE = b" \t\r\n"
_DELIMITERS = _WHITESPACE + b",]}"


class _Cursor:
    """One-byte cursor that never exposes the complete record."""

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
    raw = _read_string_bytes(cursor, first)
    if len(raw) > MAX_KEY_BYTES:
        raise ValueError("JSON object key exceeds the reader bound")
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


def _parse_pointer(pointer: str) -> tuple[str, ...]:
    if not pointer or pointer == "/":
        raise ValueError("root and empty JSON pointers are forbidden")
    if not pointer.startswith("/"):
        raise ValueError("JSON pointer must start with a slash")
    tokens = []
    for raw in pointer[1:].split("/"):
        token = raw.replace("~1", "/").replace("~0", "~")
        if "~" in token and "~0" not in raw and "~1" not in raw:
            raise ValueError("JSON pointer contains an invalid escape")
        tokens.append(token)
    return tuple(tokens)


def _read_scalar(cursor: _Cursor, opening: bytes) -> Any:
    if opening in (b"{", b"["):
        raise ValueError("container values cannot be returned")
    if opening == b'"':
        raw = _read_string_bytes(cursor, opening)
    else:
        raw_value = bytearray(opening)
        while cursor.peek() and cursor.peek() not in _DELIMITERS:
            raw_value.extend(cursor.read())
            if len(raw_value) > MAX_SCALAR_BYTES:
                raise ValueError("selected scalar exceeds the reader bound")
        raw = bytes(raw_value)
    if len(raw) > MAX_SCALAR_BYTES:
        raise ValueError("selected scalar exceeds the reader bound")
    value = json.loads(raw.decode("utf-8"))
    if isinstance(value, (dict, list)):
        raise TypeError("container values cannot be returned")
    return value


def _read_keys(cursor: _Cursor, opening: bytes) -> list[str]:
    if opening != b"{":
        raise ValueError("keys mode requires an object pointer")
    keys = []
    current = _skip_space(cursor)
    if current == b"}":
        return keys
    while True:
        keys.append(_read_key(cursor, current))
        if len(keys) > MAX_OBJECT_KEYS:
            raise ValueError("selected object exceeds the key-count bound")
        _expect(cursor, b":")
        _skip_value(cursor)
        current = _skip_space(cursor)
        if current == b"}":
            return keys
        if current != b",":
            raise ValueError("malformed selected JSON object")
        current = _skip_space(cursor)


def _read_length(cursor: _Cursor, opening: bytes) -> int:
    if opening != b"[":
        raise ValueError("length mode requires an array pointer")
    length = 0
    current = _skip_space(cursor)
    if current == b"]":
        return length
    while True:
        _skip_value(cursor, current)
        length += 1
        current = _skip_space(cursor)
        if current == b"]":
            return length
        if current != b",":
            raise ValueError("malformed selected JSON array")
        current = _skip_space(cursor)


def _extract_selected(
    cursor: _Cursor,
    tokens: tuple[str, ...],
    mode: str,
    opening: bytes | None = None,
) -> Any:
    current = _skip_space(cursor) if opening is None else opening
    if not current:
        raise ValueError("selected JSON pointer is missing")
    if not tokens:
        if mode == "scalar":
            return _read_scalar(cursor, current)
        if mode == "keys":
            return _read_keys(cursor, current)
        if mode == "length":
            return _read_length(cursor, current)
        raise ValueError(f"unknown reader mode {mode!r}")
    token, remaining = tokens[0], tokens[1:]
    if current == b"{":
        member = _skip_space(cursor)
        if member == b"}":
            raise KeyError(token)
        while True:
            key = _read_key(cursor, member)
            _expect(cursor, b":")
            value_opening = _skip_space(cursor)
            if key == token:
                return _extract_selected(cursor, remaining, mode, value_opening)
            _skip_value(cursor, value_opening)
            member = _skip_space(cursor)
            if member == b"}":
                raise KeyError(token)
            if member != b",":
                raise ValueError("malformed JSON object while selecting pointer")
            member = _skip_space(cursor)
    if current == b"[":
        if not token.isdecimal():
            raise ValueError("array pointer token must be a non-negative index")
        selected_index = int(token)
        index = 0
        item = _skip_space(cursor)
        if item == b"]":
            raise IndexError(selected_index)
        while True:
            if index == selected_index:
                return _extract_selected(cursor, remaining, mode, item)
            _skip_value(cursor, item)
            index += 1
            item = _skip_space(cursor)
            if item == b"]":
                raise IndexError(selected_index)
            if item != b",":
                raise ValueError("malformed JSON array while selecting pointer")
            item = _skip_space(cursor)
    raise ValueError("JSON pointer descends through a scalar")


def extract_field(stream: BinaryIO, pointer: str, mode: str) -> tuple[Any, int]:
    """Return one bounded projection and the number of source bytes consumed."""

    cursor = _Cursor(stream)
    value = _extract_selected(cursor, _parse_pointer(pointer), mode)
    return value, cursor.bytes_consumed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_log(log_path: Path, row: dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")
        stream.flush()


def read_field(
    record_path: Path,
    pointer: str,
    mode: str,
    access_log: Path,
) -> dict[str, Any]:
    """Read one field from the pinned record and log before returning it."""

    base_log = {
        "mode": mode,
        "pointer": pointer,
        "record": RECORD_LABEL,
        "schema": ACCESS_SCHEMA,
        "whole_record_loaded": False,
    }
    try:
        if record_path.resolve() != ALLOWED_RECORD.resolve():
            raise ValueError(
                f"reader refuses every record except {RECORD_LABEL}; "
                f"the Markdown record {FORBIDDEN_RECORD_LABEL} is never permitted"
            )
        record_sha256 = _sha256(record_path)
        with record_path.open("rb") as stream:
            value, consumed = extract_field(stream, pointer, mode)
        projection = {
            "mode": mode,
            "pointer": pointer,
            "record": RECORD_LABEL,
            "value": value,
        }
        _write_log(
            access_log,
            {
                **base_log,
                "bytes_consumed": consumed,
                "record_sha256": record_sha256,
                "status": "ALLOWED",
            },
        )
        return projection
    except Exception as error:
        _write_log(
            access_log,
            {
                **base_log,
                "error": type(error).__name__,
                "status": "REJECTED",
            },
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, default=ALLOWED_RECORD)
    parser.add_argument("--pointer", required=True)
    parser.add_argument("--mode", choices=("scalar", "keys", "length"), required=True)
    parser.add_argument("--access-log", type=Path, required=True)
    args = parser.parse_args()
    projection = read_field(args.record, args.pointer, args.mode, args.access_log)
    print(json.dumps(projection, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
