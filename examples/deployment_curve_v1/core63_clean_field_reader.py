"""Clean field-addressed reader for the CORE-63 residency repetition."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, BinaryIO

ACCESS_SCHEMA = "simllm-deployment-curve-core63-clean-access-v1"
STUDY_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = STUDY_DIR.parents[1]

EXPECTATIONS_RECORD = STUDY_DIR / "core63_expectations.json"
EXPECTATIONS_LABEL = "examples/deployment_curve_v1/core63_expectations.json"
EXPECTATIONS_SELECTOR = "/calibration_context"

CANDIDATE_RECORD = (
    REPOSITORY_ROOT / "examples/hopper_kernel_cycle_candidate_v1/candidate-record.json"
)
CANDIDATE_LABEL = "examples/hopper_kernel_cycle_candidate_v1/candidate-record.json"
CANDIDATE_SELECTOR = "/entries[7]"
EXPECTED_IMPLEMENTATION_ID = "deepseek-v3-reduced4-vllm-ep72-decode-b32-c2000"

KERNEL_SUMMARY_RELATIVE = Path(
    "gh200lane/capture-198891-deepseek-v3-tp1-graph-decode/analysis/kernel-summary.csv"
)
KERNEL_SUMMARY_LABEL = f"<kernelprobe-root>/{KERNEL_SUMMARY_RELATIVE.as_posix()}"
KERNEL_SHAPE_LABEL = "decode_b32_c2000"
KERNEL_SUMMARY_SELECTOR = (
    f"/rows[pool=decode,shape={KERNEL_SHAPE_LABEL},device=0,is_collective=False]"
)
KERNEL_SUMMARY_BYTES = 13_985

CORE_REGISTRY = REPOSITORY_ROOT / "docs/modules/core.md"
CORE_REGISTRY_LABEL = "docs/modules/core.md"
CORE63_SELECTOR = "/task-entry[task_id=CORE-63]"
CORE64_SELECTOR = "/task-entry[task_id=CORE-64]"

VOID_STUDY = STUDY_DIR / "core63_calibration_result.md"
VOID_STUDY_LABEL = "examples/deployment_curve_v1/core63_calibration_result.md"
VOID_PROTOCOL_HEADING = "## Protocol, access and preservation"
VOID_PROTOCOL_SELECTOR = "/section[heading=Protocol, access and preservation]"

EXPECTED_HEADER = (
    "pool",
    "shape",
    "device",
    "first_launch_order",
    "name",
    "is_collective",
    "collective_kind",
    "count_per_step",
    "median_duration_ns",
    "mean_duration_ns",
    "total_duration_per_step_ns",
    "share_of_step_compute",
    "graph_record_count",
    "record_count",
)
LEGACY_HEADER = tuple(
    field for field in EXPECTED_HEADER if field not in {"device", "collective_kind"}
)
CAPTURED_KERNEL_FIELDS = (
    "first_launch_order",
    "name",
    "count_per_step",
    "median_duration_ns",
    "mean_duration_ns",
    "total_duration_per_step_ns",
    "share_of_step_compute",
    "graph_record_count",
    "record_count",
)

_WHITESPACE = b" \t\r\n"
_DELIMITERS = _WHITESPACE + b",]}"
_CAPTURE = object()

CANDIDATE_PROJECTION: dict[str, Any] = {
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


class WholeFileAccessRejected(ValueError):
    """Raised before a selector could consume the final record byte."""


class _PartialSource:
    """Count physical bytes and make a complete byte stream impossible."""

    def __init__(self, stream: BinaryIO, record_size: int) -> None:
        if record_size <= 1:
            raise WholeFileAccessRejected("record is too small for a partial selector")
        self._stream = stream
        self._limit = record_size - 1
        self.bytes_accessed = 0

    def read_byte(self) -> bytes:
        if self.bytes_accessed >= self._limit:
            raise WholeFileAccessRejected(
                "selector would consume a whole-file byte stream"
            )
        value = self._stream.read(1)
        if not value:
            raise WholeFileAccessRejected(
                "selector reached EOF instead of a field boundary"
            )
        self.bytes_accessed += 1
        return value


class _SparseSource:
    """Count sparse random-access bytes and forbid complete file coverage."""

    def __init__(self, stream: BinaryIO, record_size: int) -> None:
        if record_size <= 1:
            raise WholeFileAccessRejected("record is too small for a sparse selector")
        self._stream = stream
        self.record_size = record_size
        self._positions: set[int] = set()
        self.bytes_accessed = 0

    @property
    def unique_bytes_accessed(self) -> int:
        return len(self._positions)

    def read_at(self, position: int) -> bytes:
        if not 0 <= position < self.record_size:
            raise ValueError("sparse byte position is outside the record")
        if position not in self._positions and len(self._positions) >= self.record_size - 1:
            raise WholeFileAccessRejected(
                "sparse selector would cover every byte in the record"
            )
        self._stream.seek(position)
        value = self._stream.read(1)
        if not value:
            raise ValueError("sparse selector could not read the requested byte")
        self.bytes_accessed += 1
        self._positions.add(position)
        return value


class _Cursor:
    """One-byte JSON cursor whose accounting includes lookahead bytes."""

    def __init__(self, source: _PartialSource) -> None:
        self.source = source
        self._pending: bytes | None = None

    def read(self) -> bytes:
        if self._pending is not None:
            value = self._pending
            self._pending = None
            return value
        return self.source.read_byte()

    def peek(self) -> bytes:
        if self._pending is None:
            self._pending = self.source.read_byte()
        return self._pending


class AccessRecorder:
    """Append access events before and immediately after each source read."""

    def __init__(self, ledger_path: Path) -> None:
        if ledger_path.exists():
            raise FileExistsError(f"refusing to append to existing ledger {ledger_path}")
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        ledger_path.touch(exist_ok=False)
        self.path = ledger_path
        self.event_index = 0
        self.access_index = 0

    def _append(self, entry: Mapping[str, Any]) -> None:
        self.event_index += 1
        row = {"event_index": self.event_index, **dict(entry)}
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")

    def begin(
        self,
        *,
        classification: str,
        record: str,
        selector: str,
        record_size_bytes: int,
    ) -> str:
        self.access_index += 1
        access_id = f"A{self.access_index:02d}"
        self._append(
            {
                "access_id": access_id,
                "bytes_accessed": 0,
                "classification": classification,
                "event": "BEGIN",
                "held_out_mtp_value_accessed": False,
                "record": record,
                "record_size_bytes": record_size_bytes,
                "schema": ACCESS_SCHEMA,
                "selector": selector,
                "status": "IN_PROGRESS",
                "whole_file_streamed": False,
            }
        )
        return access_id

    def finish(
        self,
        access_id: str,
        *,
        classification: str,
        record: str,
        selector: str,
        record_size_bytes: int,
        bytes_accessed: int,
        status: str,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        row: dict[str, Any] = {
            "access_id": access_id,
            "bytes_accessed": bytes_accessed,
            "classification": classification,
            "event": "END",
            "held_out_mtp_value_accessed": False,
            "record": record,
            "record_size_bytes": record_size_bytes,
            "schema": ACCESS_SCHEMA,
            "selector": selector,
            "status": status,
            "whole_file_streamed": False,
        }
        if extra:
            row.update(extra)
        self._append(row)


def _skip_space(cursor: _Cursor) -> bytes:
    value = cursor.read()
    while value in _WHITESPACE:
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
    while cursor.peek() not in _DELIMITERS:
        token.extend(cursor.read())
    json.loads(token.decode("utf-8"))


def _skip_value(cursor: _Cursor, first: bytes | None = None) -> None:
    opening = _skip_space(cursor) if first is None else first
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
        while cursor.peek() not in _DELIMITERS:
            raw_value.extend(cursor.read())
        raw = bytes(raw_value)
    return json.loads(raw.decode("utf-8"))


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
        if index in projection:
            result.append(_project_value(cursor, projection[index], current))
        else:
            _skip_value(cursor, current)
        current = _skip_space(cursor)
        if current == b"]":
            return result
        if current != b",":
            raise ValueError("malformed selected JSON array")
        index += 1
        current = _skip_space(cursor)


def _extract_top_field(source: _PartialSource, field: str) -> Any:
    cursor = _Cursor(source)
    _expect(cursor, b"{")
    current = _skip_space(cursor)
    while current != b"}":
        key = _read_key(cursor, current)
        _expect(cursor, b":")
        opening = _skip_space(cursor)
        if key == field:
            return _capture_value(cursor, opening)
        _skip_value(cursor, opening)
        current = _skip_space(cursor)
        if current == b"}":
            break
        if current != b",":
            raise ValueError("malformed top-level JSON object")
        current = _skip_space(cursor)
    raise ValueError(f"top-level field {field!r} is missing")


def _extract_candidate(source: _PartialSource) -> dict[str, Any]:
    cursor = _Cursor(source)
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
            for index in range(8):
                if current == b"]":
                    raise ValueError("selected candidate entry is missing")
                if index == 7:
                    selected = _project_object(cursor, CANDIDATE_PROJECTION, current)
                    if selected.get("implementation_id") != EXPECTED_IMPLEMENTATION_ID:
                        raise ValueError("selected implementation identity differs")
                    return selected
                _skip_value(cursor, current)
                delimiter = _skip_space(cursor)
                if delimiter != b",":
                    raise ValueError("selected candidate entry index is missing")
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


def _read_line(source: _PartialSource, prefix: bytes = b"") -> bytes:
    value = bytearray(prefix)
    while not value or value[-1:] != b"\n":
        value.extend(source.read_byte())
    return bytes(value)


def _csv_values(raw: bytes) -> list[str]:
    try:
        line = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("kernel summary is not UTF-8") from exc
    return next(csv.reader([line]))


def _read_csv_prefix(source: _PartialSource, field_count: int) -> tuple[bytes, list[str]]:
    raw = bytearray()
    fields: list[bytes] = []
    field = bytearray()
    quoted = False
    escaped_quote = False
    while len(fields) < field_count:
        current = source.read_byte()
        raw.extend(current)
        if quoted:
            if current == b'"':
                if escaped_quote:
                    escaped_quote = False
                else:
                    escaped_quote = True
            elif escaped_quote:
                escaped_quote = False
            field.extend(current)
            continue
        if current == b'"' and not field:
            quoted = True
            field.extend(current)
        elif current == b",":
            fields.append(bytes(field))
            field.clear()
        elif current == b"\n":
            raise ValueError("kernel summary row lacks routing fields")
        else:
            field.extend(current)
    decoded = _csv_values(b",".join(fields) + b"\n")
    return bytes(raw), decoded


def _extract_kernels(source: _PartialSource) -> list[dict[str, str]]:
    header_raw = _read_line(source)
    header = tuple(_csv_values(header_raw))
    legacy = header == LEGACY_HEADER
    current = header[:3] == EXPECTED_HEADER[:3] and set(EXPECTED_HEADER) <= set(header)
    if not legacy and not current:
        raise ValueError("kernel summary header differs from the frozen schema")
    prefix_width = 2 if legacy else 3
    selected: list[dict[str, str]] = []
    saw_selected_shape = False
    while True:
        prefix_raw, routing = _read_csv_prefix(source, prefix_width)
        pool, shape = routing[:2]
        route_is_selected = pool == "decode" and shape == KERNEL_SHAPE_LABEL
        if saw_selected_shape and not route_is_selected:
            if not selected:
                raise ValueError("selected kernel family contains no retained rows")
            return selected
        if not route_is_selected:
            _read_line(source, prefix_raw)
            continue
        saw_selected_shape = True
        if not legacy and routing[2] != "0":
            _read_line(source, prefix_raw)
            continue
        raw = _read_line(source, prefix_raw)
        values = _csv_values(raw)
        if len(values) != len(header):
            raise ValueError("kernel summary row width differs")
        row = dict(zip(header, values, strict=True))
        if row["is_collective"] not in {"True", "False"}:
            raise ValueError("kernel summary collective flag differs")
        if row["is_collective"] == "False":
            selected.append({name: row[name] for name in CAPTURED_KERNEL_FIELDS})


def _sparse_header(source: _SparseSource) -> bytes:
    value = bytearray()
    position = 0
    while not value or value[-1:] != b"\n":
        value.extend(source.read_at(position))
        position += 1
    return bytes(value)


def _reverse_nonterminal_lines(source: _SparseSource):
    """Read every byte except the frozen record's terminal separator byte."""

    value = bytearray()
    for position in range(source.record_size - 2, -1, -1):
        current = source.read_at(position)
        if current == b"\n":
            if value:
                yield bytes(reversed(value)) + b"\n"
                value.clear()
        else:
            value.extend(current)
    if value:
        yield bytes(reversed(value))


def _extract_kernels_sparse(source: _SparseSource) -> list[dict[str, str]]:
    """Select legacy-schema rows without accessing the terminal record byte."""

    header = LEGACY_HEADER
    selected_reversed: list[dict[str, str]] = []
    saw_selected_shape = False
    for raw in _reverse_nonterminal_lines(source):
        values = _csv_values(raw)
        if len(values) != len(header):
            raise ValueError("kernel summary row width differs")
        row = dict(zip(header, values, strict=True))
        route_is_selected = (
            row["pool"] == "decode" and row["shape"] == KERNEL_SHAPE_LABEL
        )
        if saw_selected_shape and not route_is_selected:
            if not selected_reversed:
                raise ValueError("selected kernel family contains no retained rows")
            return list(reversed(selected_reversed))
        if not route_is_selected:
            continue
        saw_selected_shape = True
        if row["is_collective"] not in {"True", "False"}:
            raise ValueError("kernel summary collective flag differs")
        if row["is_collective"] == "False":
            selected_reversed.append(
                {name: row[name] for name in CAPTURED_KERNEL_FIELDS}
            )
    raise WholeFileAccessRejected(
        "sparse kernel selector did not find a preceding routing boundary"
    )


def _extract_task_line(source: _PartialSource, task_id: str) -> str:
    marker = task_id.encode("ascii")
    while True:
        raw = _read_line(source)
        if marker in raw:
            return raw.decode("utf-8").rstrip("\n")


def _extract_markdown_section(source: _PartialSource, heading: str) -> str:
    marker = heading.encode("utf-8")
    while True:
        raw = _read_line(source)
        if raw.rstrip(b"\r\n") == marker:
            section = bytearray(raw)
            break
    while True:
        raw = _read_line(source)
        if raw.startswith(b"## "):
            return section.decode("utf-8").rstrip("\n")
        section.extend(raw)


def _read_allowlisted(
    path: Path,
    allowed: Path,
    *,
    classification: str,
    label: str,
    selector: str,
    recorder: AccessRecorder,
    extractor: Any,
) -> Any:
    if path.resolve() != allowed.resolve():
        raise ValueError("clean reader refuses a non-allowlisted path")
    record_size = path.stat().st_size
    access_id = recorder.begin(
        classification=classification,
        record=label,
        selector=selector,
        record_size_bytes=record_size,
    )
    source: _PartialSource | None = None
    try:
        with path.open("rb", buffering=0) as stream:
            source = _PartialSource(stream, record_size)
            value = extractor(source)
        recorder.finish(
            access_id,
            classification=classification,
            record=label,
            selector=selector,
            record_size_bytes=record_size,
            bytes_accessed=source.bytes_accessed,
            status="PASS",
        )
        return value
    except Exception as exc:
        recorder.finish(
            access_id,
            classification=classification,
            record=label,
            selector=selector,
            record_size_bytes=record_size,
            bytes_accessed=0 if source is None else source.bytes_accessed,
            status="REJECTED",
            extra={"error": type(exc).__name__},
        )
        raise


def _read_sparse_kernels(
    record_path: Path,
    kernelprobe_root: Path,
    recorder: AccessRecorder,
) -> list[dict[str, str]]:
    expected = kernelprobe_root.resolve() / KERNEL_SUMMARY_RELATIVE
    if record_path.resolve() != expected.resolve():
        raise ValueError("clean reader refuses a non-allowlisted kernel summary")
    record_size = record_path.stat().st_size
    if record_size != KERNEL_SUMMARY_BYTES:
        raise ValueError("kernel summary size differs from its frozen catalog entry")
    classification = "retained_standard_decode_kernel_decomposition"
    access_id = recorder.begin(
        classification=classification,
        record=KERNEL_SUMMARY_LABEL,
        selector=KERNEL_SUMMARY_SELECTOR,
        record_size_bytes=record_size,
    )
    source: _SparseSource | None = None
    try:
        with record_path.open("rb", buffering=0) as stream:
            source = _SparseSource(stream, record_size)
            rows = _extract_kernels_sparse(source)
        recorder.finish(
            access_id,
            classification=classification,
            record=KERNEL_SUMMARY_LABEL,
            selector=KERNEL_SUMMARY_SELECTOR,
            record_size_bytes=record_size,
            bytes_accessed=source.bytes_accessed,
            status="PASS",
            extra={
                "access_pattern": (
                    "frozen_legacy_schema_plus_reverse_nonterminal_bytes"
                ),
                "selected_row_count": len(rows),
                "unique_bytes_accessed": source.unique_bytes_accessed,
            },
        )
        return rows
    except Exception as exc:
        recorder.finish(
            access_id,
            classification=classification,
            record=KERNEL_SUMMARY_LABEL,
            selector=KERNEL_SUMMARY_SELECTOR,
            record_size_bytes=record_size,
            bytes_accessed=0 if source is None else source.bytes_accessed,
            status="REJECTED",
            extra={
                "access_pattern": (
                    "frozen_legacy_schema_plus_reverse_nonterminal_bytes"
                ),
                "error": type(exc).__name__,
                "unique_bytes_accessed": (
                    0 if source is None else source.unique_bytes_accessed
                ),
            },
        )
        raise


def read_clean_inputs(
    *,
    kernelprobe_root: Path,
    access_ledger: Path,
) -> dict[str, Any]:
    """Perform only the six preregistered partial-field accesses."""

    recorder = AccessRecorder(access_ledger)
    calibration_context = _read_allowlisted(
        EXPECTATIONS_RECORD,
        EXPECTATIONS_RECORD,
        classification="frozen_pre_run_standard_decode_context",
        label=EXPECTATIONS_LABEL,
        selector=EXPECTATIONS_SELECTOR,
        recorder=recorder,
        extractor=lambda source: _extract_top_field(source, "calibration_context"),
    )
    component_basis = _read_allowlisted(
        CANDIDATE_RECORD,
        CANDIDATE_RECORD,
        classification="retained_measured_component_decomposition",
        label=CANDIDATE_LABEL,
        selector=CANDIDATE_SELECTOR,
        recorder=recorder,
        extractor=_extract_candidate,
    )
    kernel_path = kernelprobe_root.resolve() / KERNEL_SUMMARY_RELATIVE
    kernel_rows = _read_sparse_kernels(
        kernel_path,
        kernelprobe_root,
        recorder=recorder,
    )
    core63_entry = _read_allowlisted(
        CORE_REGISTRY,
        CORE_REGISTRY,
        classification="literal_registry_entry",
        label=CORE_REGISTRY_LABEL,
        selector=CORE63_SELECTOR,
        recorder=recorder,
        extractor=lambda source: _extract_task_line(source, "CORE-63"),
    )
    core64_entry = _read_allowlisted(
        CORE_REGISTRY,
        CORE_REGISTRY,
        classification="conditional_registry_entry",
        label=CORE_REGISTRY_LABEL,
        selector=CORE64_SELECTOR,
        recorder=recorder,
        extractor=lambda source: _extract_task_line(source, "CORE-64"),
    )
    void_protocol_section = _read_allowlisted(
        VOID_STUDY,
        VOID_STUDY,
        classification="merged_void_protocol_findings_only",
        label=VOID_STUDY_LABEL,
        selector=VOID_PROTOCOL_SELECTOR,
        recorder=recorder,
        extractor=lambda source: _extract_markdown_section(
            source, VOID_PROTOCOL_HEADING
        ),
    )
    return {
        "calibration_context": calibration_context,
        "component_basis": component_basis,
        "core63_entry": core63_entry,
        "core64_entry": core64_entry,
        "kernel_rows": kernel_rows,
        "void_protocol_section": void_protocol_section,
    }
