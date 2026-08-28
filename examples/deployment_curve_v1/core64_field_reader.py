"""Field-addressed reader for the CORE-64 decode-family shape study."""

from __future__ import annotations

import importlib.util
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ACCESS_SCHEMA = "simllm-deployment-curve-core64-access-v1"
STUDY_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = STUDY_DIR.parents[1]

CORE63_RESULT = STUDY_DIR / "core63_clean_calibration_result.json"
CORE63_RESULT_LABEL = (
    "examples/deployment_curve_v1/core63_clean_calibration_result.json"
)
CORE63_RESULT_BYTES = 54_069
CORE63_RESULT_SELECTOR = (
    "/{calibration_only,residency_derivation,frozen_scope_fields}"
)

PROJECTION_ID = "ee154ed5f07c104269df9cf60d8730b8c6dced0ccf619fb7ff7146ec2ddfd5a2"
PROJECTION = (
    REPOSITORY_ROOT
    / "offline/calibration/deployment-projections"
    / f"{PROJECTION_ID}.json"
)
PROJECTION_LABEL = (
    "offline/calibration/deployment-projections/"
    f"{PROJECTION_ID}.json"
)
PROJECTION_BYTES = 27_611
ATTENTION_SELECTOR = "/units[1]/attention_parallelism"
STANDARD_CASE_SELECTOR = "/units[1]/case_projections[0]"

def _support() -> Any:
    path = STUDY_DIR / "core63_clean_field_reader.py"
    spec = importlib.util.spec_from_file_location("core64_reader_support", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the committed partial JSON reader support")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_SUPPORT = _support()
WholeFileAccessRejected = _SUPPORT.WholeFileAccessRejected
_PartialSource = _SUPPORT._PartialSource
_Cursor = _SUPPORT._Cursor
_CAPTURE = _SUPPORT._CAPTURE

CORE63_PROJECTION: dict[str, Any] = {
    "calibration_only": {
        "residency_corrected": {
            "classification": _CAPTURE,
            "prediction_tokens_per_second_per_node": _CAPTURE,
            "signed_residual_percent": _CAPTURE,
        }
    },
    "residency_derivation": {
        "family_decomposition": _CAPTURE,
        "step": {"residency_corrected_ps": _CAPTURE},
    },
    "scope": {
        "held_out_mtp_used_in_arithmetic_or_compared": _CAPTURE,
        "parameters_amended_or_refit": _CAPTURE,
        "scored_run_performed": _CAPTURE,
        "zero_free_or_fitted_constants": _CAPTURE,
    },
}


class AccessRecorder:
    """Append one BEGIN and one END event around every protected access."""

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


def _extract_top_projection(source: Any, projection: Mapping[str, Any]) -> dict:
    cursor = _Cursor(source)
    _SUPPORT._expect(cursor, b"{")
    result: dict[str, Any] = {}
    current = _SUPPORT._skip_space(cursor)
    while current != b"}":
        key = _SUPPORT._read_key(cursor, current)
        _SUPPORT._expect(cursor, b":")
        opening = _SUPPORT._skip_space(cursor)
        if key in projection:
            result[key] = _SUPPORT._project_value(
                cursor,
                projection[key],
                opening,
            )
            if len(result) == len(projection):
                return result
        else:
            _SUPPORT._skip_value(cursor, opening)
        current = _SUPPORT._skip_space(cursor)
        if current == b"}":
            break
        if current != b",":
            raise ValueError("malformed top-level JSON object")
        current = _SUPPORT._skip_space(cursor)
    raise ValueError("one or more projected top-level fields are missing")


def _unit_cursor(source: Any, unit_index: int) -> tuple[Any, bytes]:
    cursor = _Cursor(source)
    _SUPPORT._expect(cursor, b"{")
    current = _SUPPORT._skip_space(cursor)
    while current != b"}":
        key = _SUPPORT._read_key(cursor, current)
        _SUPPORT._expect(cursor, b":")
        opening = _SUPPORT._skip_space(cursor)
        if key != "units":
            _SUPPORT._skip_value(cursor, opening)
        else:
            if opening != b"[":
                raise ValueError("projection units field must be an array")
            current = _SUPPORT._skip_space(cursor)
            for index in range(unit_index + 1):
                if current == b"]":
                    raise ValueError("selected projection unit is missing")
                if index == unit_index:
                    if current != b"{":
                        raise ValueError("selected projection unit must be an object")
                    return cursor, current
                _SUPPORT._skip_value(cursor, current)
                current = _SUPPORT._skip_space(cursor)
                if current != b",":
                    raise ValueError("selected projection unit index is missing")
                current = _SUPPORT._skip_space(cursor)
        current = _SUPPORT._skip_space(cursor)
        if current == b"}":
            break
        if current != b",":
            raise ValueError("malformed top-level JSON object")
        current = _SUPPORT._skip_space(cursor)
    raise ValueError("projection units field is missing")


def _extract_unit_field(source: Any, unit_index: int, field: str) -> Any:
    cursor, _ = _unit_cursor(source, unit_index)
    current = _SUPPORT._skip_space(cursor)
    while current != b"}":
        key = _SUPPORT._read_key(cursor, current)
        _SUPPORT._expect(cursor, b":")
        opening = _SUPPORT._skip_space(cursor)
        if key == field:
            return _SUPPORT._capture_value(cursor, opening)
        _SUPPORT._skip_value(cursor, opening)
        current = _SUPPORT._skip_space(cursor)
        if current == b"}":
            break
        if current != b",":
            raise ValueError("malformed selected projection unit")
        current = _SUPPORT._skip_space(cursor)
    raise ValueError(f"projection unit field {field!r} is missing")


def _extract_first_case(source: Any, unit_index: int) -> dict[str, Any]:
    cursor, _ = _unit_cursor(source, unit_index)
    current = _SUPPORT._skip_space(cursor)
    while current != b"}":
        key = _SUPPORT._read_key(cursor, current)
        _SUPPORT._expect(cursor, b":")
        opening = _SUPPORT._skip_space(cursor)
        if key == "case_projections":
            if opening != b"[":
                raise ValueError("case projections field must be an array")
            first = _SUPPORT._skip_space(cursor)
            value = _SUPPORT._capture_value(cursor, first)
            if not isinstance(value, dict):
                raise TypeError("selected standard case must be an object")
            return value
        _SUPPORT._skip_value(cursor, opening)
        current = _SUPPORT._skip_space(cursor)
        if current == b"}":
            break
        if current != b",":
            raise ValueError("malformed selected projection unit")
        current = _SUPPORT._skip_space(cursor)
    raise ValueError("case projections field is missing")


def _read_allowlisted(
    path: Path,
    allowed: Path,
    *,
    expected_bytes: int,
    classification: str,
    label: str,
    selector: str,
    recorder: AccessRecorder,
    extractor: Any,
) -> Any:
    if path.resolve() != allowed.resolve():
        raise ValueError("CORE-64 reader refuses a non-allowlisted path")
    record_size = path.stat().st_size
    if record_size != expected_bytes:
        raise ValueError("allowlisted record size differs from the frozen value")
    access_id = recorder.begin(
        classification=classification,
        record=label,
        selector=selector,
        record_size_bytes=record_size,
    )
    source = None
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


def read_core64_inputs(access_ledger: Path) -> dict[str, Any]:
    """Perform only the three preregistered calibration-only accesses."""

    recorder = AccessRecorder(access_ledger)
    core63 = _read_allowlisted(
        CORE63_RESULT,
        CORE63_RESULT,
        expected_bytes=CORE63_RESULT_BYTES,
        classification="published_core63_standard_decode_basis",
        label=CORE63_RESULT_LABEL,
        selector=CORE63_RESULT_SELECTOR,
        recorder=recorder,
        extractor=lambda source: _extract_top_projection(
            source,
            CORE63_PROJECTION,
        ),
    )
    attention_parallelism = _read_allowlisted(
        PROJECTION,
        PROJECTION,
        expected_bytes=PROJECTION_BYTES,
        classification="ep72_attention_parallelism",
        label=PROJECTION_LABEL,
        selector=ATTENTION_SELECTOR,
        recorder=recorder,
        extractor=lambda source: _extract_unit_field(
            source,
            1,
            "attention_parallelism",
        ),
    )
    standard_case = _read_allowlisted(
        PROJECTION,
        PROJECTION,
        expected_bytes=PROJECTION_BYTES,
        classification="ep72_standard_decode_rank_shape",
        label=PROJECTION_LABEL,
        selector=STANDARD_CASE_SELECTOR,
        recorder=recorder,
        extractor=lambda source: _extract_first_case(source, 1),
    )
    return {
        "attention_parallelism": attention_parallelism,
        "core63": core63,
        "standard_case": standard_case,
    }


__all__ = [
    "AccessRecorder",
    "WholeFileAccessRejected",
    "read_core64_inputs",
]
