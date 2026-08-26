"""Field-addressed reader for VLLM-39's two Granite decode surface rows."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, BinaryIO

ACCESS_SCHEMA = "simllm-pd-session-load-delay-access-v1"
RECORD_SHA256 = "ff46f6d8a79ddae899da89d4db6eb34373f8042acd06cab50b6336c8fb9a8f52"
STUDY_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = STUDY_DIR.parents[1]
ALLOWED_RECORD = (
    REPOSITORY_ROOT
    / "examples"
    / "hopper_kernel_cycle_candidate_v1"
    / "candidate-record.json"
)
RECORD_LABEL = "examples/hopper_kernel_cycle_candidate_v1/candidate-record.json"
TOP_LEVEL_FIELDS = (
    "acceptance_status",
    "campaign_id",
    "coverage",
)
DEVICE_FIELD = "device_kind_id"
ENTRY_SELECTORS = (
    "/entries[0:key.model_identity.name=ibm-granite/granite-3.0-1b-a400m-instruct,pool=decode,launch_mode=cuda-graph,batch_size=1,kv=16]",
    "/entries[1:key.model_identity.name=ibm-granite/granite-3.0-1b-a400m-instruct,pool=decode,launch_mode=cuda-graph,batch_size=8,kv=16]",
)
_WHITESPACE = b" \t\r\n"
_DELIMITERS = _WHITESPACE + b",]}"


class _Cursor:
    """One-byte cursor that cannot return the complete record."""

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


def _capture_value(cursor: _Cursor, first: bytes | None = None) -> bytes:
    opening = _skip_space(cursor) if first is None else first
    if not opening:
        raise ValueError("missing selected JSON value")
    if opening == b'"':
        return _read_string_bytes(cursor, opening)
    if opening not in (b"{", b"["):
        value = bytearray(opening)
        while cursor.peek() and cursor.peek() not in _DELIMITERS:
            value.extend(cursor.read())
        return bytes(value)
    value = bytearray(opening)
    nesting = [opening]
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


def _selected_object_field(cursor: _Cursor, field: str) -> tuple[Any, int]:
    _expect(cursor, b"{")
    selected: Any | None = None
    selected_offset: int | None = None
    current = _skip_space(cursor)
    while current != b"}":
        key = _read_key(cursor, current)
        _expect(cursor, b":")
        if key == field:
            raw = _capture_value(cursor)
            selected = json.loads(raw.decode("utf-8"))
            selected_offset = cursor.bytes_consumed
        else:
            _skip_value(cursor)
        current = _skip_space(cursor)
        if current == b"}":
            break
        if current != b",":
            raise ValueError("malformed selected JSON object")
        current = _skip_space(cursor)
    if selected_offset is None:
        raise ValueError(f"selected object field {field!r} is missing")
    return selected, selected_offset


def _validate_entry(entry: Mapping[str, Any], batch_size: int) -> None:
    key = entry.get("key")
    if not isinstance(key, Mapping):
        raise TypeError("permitted surface entry has no key object")
    model = key.get("model_identity")
    parallel = key.get("parallelism")
    shape = key.get("shape")
    evidence = entry.get("evidence")
    if not all(isinstance(value, Mapping) for value in (model, parallel, shape, evidence)):
        raise ValueError("permitted surface entry has malformed key or evidence")
    assert isinstance(model, Mapping)
    assert isinstance(parallel, Mapping)
    assert isinstance(shape, Mapping)
    assert isinstance(evidence, Mapping)
    expected_kv = [16] * batch_size
    checks = {
        "model": model.get("name") == "ibm-granite/granite-3.0-1b-a400m-instruct",
        "pool": key.get("pool") == "decode",
        "launch_mode": key.get("launch_mode") == "cuda-graph",
        "tensor_parallel": parallel.get("tensor_parallel") == 1,
        "batch_size": shape.get("batch_size") == batch_size,
        "per_request_kv_lengths": shape.get("per_request_kv_lengths") == expected_kv,
        "service_class": evidence.get("service_class") == "MEASURED",
        "split": evidence.get("split") == "calibration",
    }
    failed = sorted(name for name, held in checks.items() if not held)
    if failed:
        raise ValueError(f"permitted surface entry disagrees on {failed}")


def extract_surface_projection(
    stream: BinaryIO,
) -> tuple[dict[str, Any], list[tuple[str, int]], int]:
    """Extract the two leading permitted entries and stop at entry 1."""

    cursor = _Cursor(stream)
    projection: dict[str, Any] = {}
    access_offsets: list[tuple[str, int]] = []
    _expect(cursor, b"{")
    current = _skip_space(cursor)
    while current != b"}":
        key = _read_key(cursor, current)
        _expect(cursor, b":")
        if key in TOP_LEVEL_FIELDS:
            raw = _capture_value(cursor)
            projection[key] = json.loads(raw.decode("utf-8"))
            access_offsets.append((f"/{key}", cursor.bytes_consumed))
        elif key == "device":
            value, offset = _selected_object_field(cursor, DEVICE_FIELD)
            projection["device_kind_id"] = value
            access_offsets.append((f"/device/{DEVICE_FIELD}", offset))
        elif key == "entries":
            missing = [name for name in TOP_LEVEL_FIELDS if name not in projection]
            if missing or "device_kind_id" not in projection:
                raise ValueError(f"required provenance fields precede entries: {missing}")
            _expect(cursor, b"[")
            entries = []
            for index, batch_size in enumerate((1, 8)):
                first = _skip_space(cursor)
                if first != b"{":
                    raise ValueError(f"permitted entry {index} is missing")
                raw = _capture_value(cursor, first)
                entry = json.loads(raw.decode("utf-8"))
                if not isinstance(entry, dict):
                    raise TypeError("permitted surface entry must be an object")
                _validate_entry(entry, batch_size)
                entries.append(entry)
                access_offsets.append((ENTRY_SELECTORS[index], cursor.bytes_consumed))
                if index == 0:
                    _expect(cursor, b",")
            projection["entries"] = entries
            return projection, access_offsets, cursor.bytes_consumed
        else:
            _skip_value(cursor)
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


def read_surface_projection(record_path: Path, access_log: Path) -> dict[str, Any]:
    """Read only the allowlisted surface fields and log each field access."""

    if record_path.resolve() != ALLOWED_RECORD.resolve():
        entry = {
            "schema": ACCESS_SCHEMA,
            "record": RECORD_LABEL,
            "record_sha256": RECORD_SHA256,
            "selector": "surface-projection",
            "status": "REJECTED",
            "whole_record_loaded": False,
            "error": "ValueError",
        }
        _append_access(access_log, entry)
        raise ValueError("VLLM-39 reader refuses every non-allowlisted record")
    try:
        with record_path.open("rb", buffering=0) as stream:
            projection, accesses, consumed = extract_surface_projection(stream)
    except Exception as exc:
        _append_access(
            access_log,
            {
                "schema": ACCESS_SCHEMA,
                "record": RECORD_LABEL,
                "record_sha256": RECORD_SHA256,
                "selector": "surface-projection",
                "status": "REJECTED",
                "whole_record_loaded": False,
                "error": type(exc).__name__,
            },
        )
        raise
    for selector, offset in accesses:
        _append_access(
            access_log,
            {
                "schema": ACCESS_SCHEMA,
                "record": RECORD_LABEL,
                "record_sha256": RECORD_SHA256,
                "selector": selector,
                "bytes_consumed_to_field": offset,
                "bytes_consumed_total": consumed,
                "status": "PASS",
                "whole_record_loaded": False,
            },
        )
    return projection


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--access-log", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    projection = read_surface_projection(ALLOWED_RECORD, args.access_log)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(projection, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
