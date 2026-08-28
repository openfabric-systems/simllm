"""Field-addressed readers for the CORE-65 physical-binding study."""

from __future__ import annotations

import importlib.util
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ACCESS_SCHEMA = "simllm-deployment-curve-core65-access-v1"
STUDY_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = STUDY_DIR.parents[1]

CORE63_BASIS_RELATIVE = Path("core63/reproduction/basis-success.json")
CORE63_BASIS_LABEL = "<wave-runs>/core63/reproduction/basis-success.json"
CORE63_BASIS_BYTES = 27_555

CORE64_RESULT = STUDY_DIR / "core64_shape_result.json"
CORE64_RESULT_LABEL = "examples/deployment_curve_v1/core64_shape_result.json"
CORE64_RESULT_BYTES = 32_044
CORE64_RESULT_SELECTOR = (
    "/{calibration_only,component_classification/"
    "retained_physical_decomposition,preservation_lock,"
    "registry_disposition,scope}"
)

MODEL_EXPECTATIONS = (
    REPOSITORY_ROOT
    / "examples/model_extraction_deepseek_v3_v1/expectations.json"
)
MODEL_EXPECTATIONS_LABEL = (
    "examples/model_extraction_deepseek_v3_v1/expectations.json"
)
MODEL_EXPECTATIONS_BYTES = 22_621
MODEL_FIELDS = (
    "geometry_symbols",
    "inventory_contract",
    "deployment_projection_contract",
    "physical_sanity",
)

CORE_REGISTRY = REPOSITORY_ROOT / "docs/modules/core.md"
CORE_REGISTRY_LABEL = "docs/modules/core.md"
CORE_REGISTRY_BYTES = 102_420
CORE65_SELECTOR = "/task-entry[last-leading-block task_id=CORE-65]"

CAPTURE_PROFILE_RELATIVE = Path(
    "gh200lane/capture-198891-deepseek-v3-tp1-graph-decode/profile.json"
)
CAPTURE_PROFILE_LABEL = f"<kernelprobe-root>/{CAPTURE_PROFILE_RELATIVE.as_posix()}"
CAPTURE_PROFILE_FIELDS = (
    "model",
    "tensor_parallel_size",
    "mode",
    "deepseek_suite",
    "reduced_layers",
    "model_config",
)
CAPTURE_CELL = "decode_b32_c2000"


def _support() -> Any:
    path = STUDY_DIR / "core63_clean_field_reader.py"
    spec = importlib.util.spec_from_file_location("core65_reader_support", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the committed partial-reader support")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_SUPPORT = _support()
WholeFileAccessRejected = _SUPPORT.WholeFileAccessRejected
_PartialSource = _SUPPORT._PartialSource
_SparseSource = _SUPPORT._SparseSource
_Cursor = _SUPPORT._Cursor
_CAPTURE = _SUPPORT._CAPTURE

CORE64_PROJECTION: dict[str, Any] = {
    "calibration_only": _CAPTURE,
    "component_classification": {
        "retained_physical_decomposition": _CAPTURE,
    },
    "preservation_lock": _CAPTURE,
    "registry_disposition": _CAPTURE,
    "scope": _CAPTURE,
}


class AccessRecorder:
    """Write one contemporaneous BEGIN/END pair per protected selector."""

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
        record_size_bytes: int | None,
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
        record_size_bytes: int | None,
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
            result[key] = _SUPPORT._project_value(cursor, projection[key], opening)
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


def _extract_capture_cell(source: Any) -> dict[str, Any]:
    cursor = _Cursor(source)
    _SUPPORT._expect(cursor, b"{")
    current = _SUPPORT._skip_space(cursor)
    while current != b"}":
        key = _SUPPORT._read_key(cursor, current)
        _SUPPORT._expect(cursor, b":")
        opening = _SUPPORT._skip_space(cursor)
        if key != "cases":
            _SUPPORT._skip_value(cursor, opening)
        else:
            if opening != b"[":
                raise ValueError("capture profile cases must be an array")
            current = _SUPPORT._skip_space(cursor)
            while current != b"]":
                value = _SUPPORT._capture_value(cursor, current)
                if not isinstance(value, dict):
                    raise TypeError("capture profile case must be an object")
                if value.get("cell") == CAPTURE_CELL:
                    return value
                current = _SUPPORT._skip_space(cursor)
                if current == b"]":
                    break
                if current != b",":
                    raise ValueError("malformed capture profile cases")
                current = _SUPPORT._skip_space(cursor)
        current = _SUPPORT._skip_space(cursor)
        if current == b"}":
            break
        if current != b",":
            raise ValueError("malformed capture profile")
        current = _SUPPORT._skip_space(cursor)
    raise ValueError(f"capture cell {CAPTURE_CELL!r} is missing")


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
        raise ValueError("CORE-65 reader refuses a non-allowlisted path")
    access_id = recorder.begin(
        classification=classification,
        record=label,
        selector=selector,
        record_size_bytes=expected_bytes,
    )
    source = None
    try:
        record_size = path.stat().st_size
        if record_size != expected_bytes:
            raise ValueError("allowlisted record size differs from the freeze")
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
            record_size_bytes=expected_bytes,
            bytes_accessed=0 if source is None else source.bytes_accessed,
            status="REJECTED",
            extra={"error": type(exc).__name__},
        )
        raise


def _read_registry_block(recorder: AccessRecorder) -> str:
    access_id = recorder.begin(
        classification="literal_core65_registry_entry",
        record=CORE_REGISTRY_LABEL,
        selector=CORE65_SELECTOR,
        record_size_bytes=CORE_REGISTRY_BYTES,
    )
    source = None
    try:
        record_size = CORE_REGISTRY.stat().st_size
        if record_size != CORE_REGISTRY_BYTES:
            raise ValueError("CORE registry size differs from the freeze")
        with CORE_REGISTRY.open("rb", buffering=0) as stream:
            source = _SparseSource(stream, record_size)
            value = _SUPPORT._extract_last_task_block(source, "CORE-65")
        recorder.finish(
            access_id,
            classification="literal_core65_registry_entry",
            record=CORE_REGISTRY_LABEL,
            selector=CORE65_SELECTOR,
            record_size_bytes=record_size,
            bytes_accessed=source.bytes_accessed,
            status="PASS",
            extra={
                "access_pattern": "reverse_nonterminal_task_block",
                "unique_bytes_accessed": source.unique_bytes_accessed,
            },
        )
        return value
    except Exception as exc:
        recorder.finish(
            access_id,
            classification="literal_core65_registry_entry",
            record=CORE_REGISTRY_LABEL,
            selector=CORE65_SELECTOR,
            record_size_bytes=CORE_REGISTRY_BYTES,
            bytes_accessed=0 if source is None else source.bytes_accessed,
            status="REJECTED",
            extra={
                "error": type(exc).__name__,
                "unique_bytes_accessed": (
                    0 if source is None else source.unique_bytes_accessed
                ),
            },
        )
        raise


def read_core65_inputs(
    *,
    wave_runs_root: Path,
    access_ledger: Path,
) -> dict[str, Any]:
    """Perform only the eight preregistered retained-record accesses."""

    recorder = AccessRecorder(access_ledger)
    basis = wave_runs_root.resolve() / CORE63_BASIS_RELATIVE
    component_basis = _read_allowlisted(
        basis,
        basis,
        expected_bytes=CORE63_BASIS_BYTES,
        classification="retained_component_basis",
        label=CORE63_BASIS_LABEL,
        selector="/component_basis",
        recorder=recorder,
        extractor=lambda source: _SUPPORT._extract_top_field(
            source, "component_basis"
        ),
    )
    kernel_rows = _read_allowlisted(
        basis,
        basis,
        expected_bytes=CORE63_BASIS_BYTES,
        classification="retained_total_kernel_inventory",
        label=CORE63_BASIS_LABEL,
        selector="/kernel_rows",
        recorder=recorder,
        extractor=lambda source: _SUPPORT._extract_top_field(source, "kernel_rows"),
    )
    core64 = _read_allowlisted(
        CORE64_RESULT,
        CORE64_RESULT,
        expected_bytes=CORE64_RESULT_BYTES,
        classification="inherited_core64_standard_decode_remainder",
        label=CORE64_RESULT_LABEL,
        selector=CORE64_RESULT_SELECTOR,
        recorder=recorder,
        extractor=lambda source: _extract_top_projection(
            source, CORE64_PROJECTION
        ),
    )
    model_fields = {}
    for field in MODEL_FIELDS:
        model_fields[field] = _read_allowlisted(
            MODEL_EXPECTATIONS,
            MODEL_EXPECTATIONS,
            expected_bytes=MODEL_EXPECTATIONS_BYTES,
            classification="frozen_deepseek_physical_contract",
            label=MODEL_EXPECTATIONS_LABEL,
            selector=f"/{field}",
            recorder=recorder,
            extractor=lambda source, selected=field: _SUPPORT._extract_top_field(
                source, selected
            ),
        )
    core65_entry = _read_registry_block(recorder)
    return {
        "component_basis": component_basis,
        "core64": core64,
        "core65_entry": core65_entry,
        "kernel_rows": kernel_rows,
        "model_fields": model_fields,
    }


def read_capture_profile(
    *,
    kernelprobe_root: Path,
    access_ledger: Path,
) -> dict[str, Any]:
    """Read only the frozen capture-profile fields, or log its absence."""

    recorder = AccessRecorder(access_ledger)
    profile = kernelprobe_root.resolve() / CAPTURE_PROFILE_RELATIVE
    if not profile.exists():
        selector = "/availability"
        access_id = recorder.begin(
            classification="original_capture_profile",
            record=CAPTURE_PROFILE_LABEL,
            selector=selector,
            record_size_bytes=None,
        )
        recorder.finish(
            access_id,
            classification="original_capture_profile",
            record=CAPTURE_PROFILE_LABEL,
            selector=selector,
            record_size_bytes=None,
            bytes_accessed=0,
            status="UNAVAILABLE",
            extra={"error": "FileNotFoundError"},
        )
        return {"available": False}

    record_size = profile.stat().st_size
    values: dict[str, Any] = {"available": True}
    for field in CAPTURE_PROFILE_FIELDS:
        values[field] = _read_allowlisted(
            profile,
            profile,
            expected_bytes=record_size,
            classification="original_capture_profile",
            label=CAPTURE_PROFILE_LABEL,
            selector=f"/{field}",
            recorder=recorder,
            extractor=lambda source, selected=field: _SUPPORT._extract_top_field(
                source, selected
            ),
        )
    values["cell"] = _read_allowlisted(
        profile,
        profile,
        expected_bytes=record_size,
        classification="original_capture_cell",
        label=CAPTURE_PROFILE_LABEL,
        selector=f"/cases[cell={CAPTURE_CELL}]",
        recorder=recorder,
        extractor=_extract_capture_cell,
    )
    return values


__all__ = [
    "ACCESS_SCHEMA",
    "AccessRecorder",
    "WholeFileAccessRejected",
    "read_capture_profile",
    "read_core65_inputs",
]
