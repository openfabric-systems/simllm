"""Bounded, audited field reader for the CORE-66 hardware remainder.

The reader never materializes a protected record. Each public access is tied to
an exact allowlisted path and selector, writes its BEGIN event before opening
the source, and leaves at least one source byte unread. Numeric MTP selectors
are rejected before the source is opened.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO

SCHEMA = "simllm-deployment-curve-core66-access-v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ALLOWED_RECORDS = {
    "core65-expectations": Path("examples/deployment_curve_v1/core65_expectations.json"),
    "core65-result": Path("examples/deployment_curve_v1/core65_physical_binding_result.json"),
    "hardware-remainder": Path("examples/deployment_curve_v1/core66_hardware_remainder.md"),
    "registry": Path("docs/modules/core.md"),
}
FORBIDDEN_JSON_POINTER_FRAGMENTS = (
    "held_out_mtp",
    "mtp_numeric",
    "simulated_mtp",
)


class WholeFileAccessRejected(ValueError):
    """Raised when a selector cannot finish without consuming the guard byte."""


class SelectorRejected(ValueError):
    """Raised before source open for a forbidden or non-allowlisted selector."""


class _PartialSource:
    """Sequential source that reserves the final byte as an unread guard."""

    def __init__(self, stream: BinaryIO, size: int):
        if size < 2:
            raise WholeFileAccessRejected("record is too small for bounded access")
        self._stream = stream
        self.size = size
        self.bytes_accessed = 0

    @property
    def at_guard(self) -> bool:
        return self._stream.tell() >= self.size - 1

    def read_byte(self) -> bytes:
        if self.at_guard:
            return b""
        value = self._stream.read(1)
        if value:
            self.bytes_accessed += 1
        return value

    def readline(self) -> bytes:
        value = bytearray()
        while True:
            byte = self.read_byte()
            if not byte:
                return bytes(value)
            value.extend(byte)
            if byte == b"\n":
                return bytes(value)


class _Cursor:
    def __init__(self, source: _PartialSource):
        self.source = source
        self._lookahead = b""

    def peek(self) -> bytes:
        if not self._lookahead:
            self._lookahead = self.source.read_byte()
        return self._lookahead

    def take(self) -> bytes:
        value = self.peek()
        self._lookahead = b""
        return value

    def skip_space(self) -> None:
        while self.peek() in {b" ", b"\t", b"\r", b"\n"}:
            self.take()

    def expect(self, expected: bytes) -> None:
        actual = self.take()
        if actual != expected:
            raise ValueError(f"expected {expected!r}, found {actual!r}")


def _heading_level(line: str) -> int | None:
    stripped = line.lstrip()
    hashes = len(stripped) - len(stripped.lstrip("#"))
    if hashes and stripped[hashes : hashes + 1] == " ":
        return hashes
    return None


def extract_markdown_section(source: _PartialSource, heading: str) -> dict[str, object]:
    """Return one atomic heading section and the next heading, if present."""

    selected: list[bytes] = []
    found = False
    while True:
        line = source.readline()
        if not line:
            break
        decoded = line.decode("utf-8")
        level = _heading_level(decoded)
        if not found:
            if decoded.rstrip("\r\n") == heading:
                found = True
                selected.append(line)
            continue
        if level is not None:
            return {
                "heading": heading,
                "text": b"".join(selected).decode("utf-8"),
                "next_heading": decoded.rstrip("\r\n"),
            }
        selected.append(line)
    if not found:
        raise WholeFileAccessRejected(
            f"heading {heading!r} was not found before the unread guard byte"
        )
    return {
        "heading": heading,
        "text": b"".join(selected).decode("utf-8"),
        "next_heading": None,
    }


def extract_markdown_task(source: _PartialSource, task: str) -> str:
    """Return one list-item task and its indented continuation block."""

    prefix = f"- {task} "
    selected: list[bytes] = []
    found = False
    while True:
        line = source.readline()
        if not line:
            break
        decoded = line.decode("utf-8")
        if not found:
            if decoded.startswith((prefix, f"- {task} (")):
                found = True
                selected.append(line)
            continue
        if decoded.startswith("  ") or not decoded.strip():
            selected.append(line)
            continue
        break
    if not found:
        raise WholeFileAccessRejected(f"task {task!r} was not found before the unread guard byte")
    return b"".join(selected).decode("utf-8").rstrip()


def _parse_string(cursor: _Cursor) -> tuple[str, bytes]:
    raw = bytearray()
    cursor.expect(b'"')
    raw.extend(b'"')
    escaped = False
    while True:
        byte = cursor.take()
        if not byte:
            raise WholeFileAccessRejected("JSON string reaches the unread guard byte")
        raw.extend(byte)
        if escaped:
            escaped = False
        elif byte == b"\\":
            escaped = True
        elif byte == b'"':
            break
    encoded = bytes(raw)
    return json.loads(encoded.decode("utf-8")), encoded


def _scan_value(cursor: _Cursor, *, retain: bool) -> bytes:
    cursor.skip_space()
    first = cursor.peek()
    if not first:
        raise WholeFileAccessRejected("JSON value reaches the unread guard byte")
    raw = bytearray()
    if first == b'"':
        _, encoded = _parse_string(cursor)
        return encoded if retain else b""
    if first in {b"{", b"["}:
        stack: list[bytes] = []
        in_string = False
        escaped = False
        pairs = {b"{": b"}", b"[": b"]"}
        while True:
            byte = cursor.take()
            if not byte:
                raise WholeFileAccessRejected("JSON container reaches the unread guard byte")
            if retain:
                raw.extend(byte)
            if in_string:
                if escaped:
                    escaped = False
                elif byte == b"\\":
                    escaped = True
                elif byte == b'"':
                    in_string = False
                continue
            if byte == b'"':
                in_string = True
            elif byte in pairs:
                stack.append(pairs[byte])
            elif stack and byte == stack[-1]:
                stack.pop()
                if not stack:
                    return bytes(raw)

    while cursor.peek() not in {b"", b",", b"}", b"]", b" ", b"\t", b"\r", b"\n"}:
        byte = cursor.take()
        if retain:
            raw.extend(byte)
    if not raw and retain:
        raise ValueError("empty JSON scalar")
    return bytes(raw)


def _select_json(cursor: _Cursor, parts: list[str], depth: int) -> bytes:
    cursor.skip_space()
    opening = cursor.take()
    target = parts[depth]
    if opening == b"{":
        cursor.skip_space()
        while cursor.peek() != b"}":
            key, _ = _parse_string(cursor)
            cursor.skip_space()
            cursor.expect(b":")
            cursor.skip_space()
            if key == target:
                if depth == len(parts) - 1:
                    return _scan_value(cursor, retain=True)
                return _select_json(cursor, parts, depth + 1)
            _scan_value(cursor, retain=False)
            cursor.skip_space()
            separator = cursor.take()
            if separator == b"}":
                break
            if separator != b",":
                raise ValueError("malformed JSON object")
            cursor.skip_space()
        raise KeyError(target)
    if opening == b"[":
        try:
            target_index = int(target)
        except ValueError as error:
            raise ValueError(f"array selector is not an index: {target!r}") from error
        index = 0
        cursor.skip_space()
        while cursor.peek() != b"]":
            if index == target_index:
                if depth == len(parts) - 1:
                    return _scan_value(cursor, retain=True)
                return _select_json(cursor, parts, depth + 1)
            _scan_value(cursor, retain=False)
            index += 1
            cursor.skip_space()
            separator = cursor.take()
            if separator == b"]":
                break
            if separator != b",":
                raise ValueError("malformed JSON array")
            cursor.skip_space()
        raise IndexError(target_index)
    raise ValueError("JSON pointer traverses a scalar")


def extract_json_pointer(source: _PartialSource, pointer: str) -> object:
    """Decode only the value at a JSON pointer and stop immediately after it."""

    if not pointer.startswith("/") or pointer == "/":
        raise SelectorRejected("a non-root JSON pointer is required")
    parts = [part.replace("~1", "/").replace("~0", "~") for part in pointer[1:].split("/")]
    cursor = _Cursor(source)
    try:
        raw = _select_json(cursor, parts, 0)
    except (KeyError, IndexError) as error:
        if source.at_guard:
            raise WholeFileAccessRejected(
                "selector exhausted the record up to its unread guard byte"
            ) from error
        raise
    return json.loads(raw.decode("utf-8"))


class AccessRecorder:
    """Append-and-fsync access events into new per-tranche ledgers."""

    def __init__(self, ledger: Path, forbidden_ledger: Path):
        self.ledger = ledger
        self.forbidden_ledger = forbidden_ledger
        self._event_index = 0
        self._access_index = 0
        for path in (ledger, forbidden_ledger):
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8", newline="\n"):
                pass

    def _append(self, path: Path, event: dict[str, object]) -> None:
        with path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(event, sort_keys=True, separators=(",", ":")))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())

    def reject(self, *, label: str, selector: str, reason: str) -> None:
        self._event_index += 1
        self._append(
            self.forbidden_ledger,
            {
                "event": "DENIED",
                "event_index": self._event_index,
                "reason": reason,
                "record": label,
                "schema": SCHEMA,
                "selector": selector,
                "source_opened": False,
            },
        )

    def begin(self, *, label: str, selector: str) -> str:
        self._event_index += 1
        self._access_index += 1
        access_id = f"A{self._access_index:02d}"
        self._append(
            self.ledger,
            {
                "access_id": access_id,
                "bytes_accessed": 0,
                "event": "BEGIN",
                "event_index": self._event_index,
                "record": label,
                "schema": SCHEMA,
                "selector": selector,
                "status": "IN_PROGRESS",
                "whole_file_streamed": False,
            },
        )
        return access_id

    def end(
        self,
        *,
        access_id: str,
        label: str,
        selector: str,
        source: _PartialSource | None,
        status: str,
        error: str | None = None,
    ) -> None:
        self._event_index += 1
        event: dict[str, object] = {
            "access_id": access_id,
            "bytes_accessed": 0 if source is None else source.bytes_accessed,
            "event": "END",
            "event_index": self._event_index,
            "record": label,
            "record_size_bytes": None if source is None else source.size,
            "schema": SCHEMA,
            "selector": selector,
            "status": status,
            "whole_file_streamed": False,
        }
        if error is not None:
            event["error"] = error
        self._append(self.ledger, event)


def _resolved_rule(label: str, selector: str, recorder: AccessRecorder) -> Path:
    relative = ALLOWED_RECORDS.get(label)
    if relative is None:
        recorder.reject(label=label, selector=selector, reason="record is not allowlisted")
        raise SelectorRejected(f"record {label!r} is not allowlisted")
    if any(fragment in selector.casefold() for fragment in FORBIDDEN_JSON_POINTER_FRAGMENTS):
        recorder.reject(
            label=label,
            selector=selector,
            reason="held-out MTP selector is forbidden",
        )
        raise SelectorRejected("held-out MTP selector is forbidden")
    return REPOSITORY_ROOT / relative


def read_allowlisted(
    *,
    label: str,
    selector: str,
    recorder: AccessRecorder,
    extractor: Callable[[_PartialSource], object],
) -> object:
    """Perform one logged bounded access to an allowlisted record."""

    path = _resolved_rule(label, selector, recorder)
    access_id = recorder.begin(label=label, selector=selector)
    source: _PartialSource | None = None
    try:
        with path.open("rb") as stream:
            source = _PartialSource(stream, path.stat().st_size)
            value = extractor(source)
        if source.bytes_accessed >= source.size:
            raise WholeFileAccessRejected("whole-file access is forbidden")
    except Exception as error:
        recorder.end(
            access_id=access_id,
            label=label,
            selector=selector,
            source=source,
            status="REJECTED",
            error=type(error).__name__,
        )
        raise
    recorder.end(
        access_id=access_id,
        label=label,
        selector=selector,
        source=source,
        status="PASS",
    )
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--access-ledger", required=True, type=Path)
    parser.add_argument("--forbidden-ledger", required=True, type=Path)
    subparsers = parser.add_subparsers(dest="command", required=True)

    section = subparsers.add_parser("markdown-section")
    section.add_argument("--record", required=True)
    section.add_argument("--heading", required=True)

    task = subparsers.add_parser("markdown-task")
    task.add_argument("--record", required=True)
    task.add_argument("--task", required=True)

    pointer = subparsers.add_parser("json-pointer")
    pointer.add_argument("--record", required=True)
    pointer.add_argument("--pointer", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    recorder = AccessRecorder(args.access_ledger, args.forbidden_ledger)
    if args.command == "markdown-section":
        selector = f"markdown-section:{args.heading}"
        value = read_allowlisted(
            label=args.record,
            selector=selector,
            recorder=recorder,
            extractor=lambda source: extract_markdown_section(source, args.heading),
        )
    elif args.command == "markdown-task":
        selector = f"markdown-task:{args.task}"
        value = read_allowlisted(
            label=args.record,
            selector=selector,
            recorder=recorder,
            extractor=lambda source: extract_markdown_task(source, args.task),
        )
    else:
        selector = args.pointer
        value = read_allowlisted(
            label=args.record,
            selector=selector,
            recorder=recorder,
            extractor=lambda source: extract_json_pointer(source, args.pointer),
        )
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
