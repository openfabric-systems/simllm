"""Field-addressed readers for the fourth CORE-54 scored run."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, BinaryIO

ACCESS_SCHEMA = "simllm-deployment-curve-run4-field-access-v1"
MTP_ANCHOR_ID = "sglang_decode_simulated_mtp"
SUCCESSOR_SHA256 = "d868a4f35d633032daa238168d00f42c2ab47fc569db649b19b907008072e107"
STUDY_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = STUDY_DIR.parents[1]
SUCCESSOR_RESULT = (
    REPOSITORY_ROOT
    / "examples/hopper_kernel_cycle_candidate_v1/successors"
    / SUCCESSOR_SHA256
    / "result.json"
)
ANCHOR_FREEZE = STUDY_DIR / "expectations.json"
RUN3_PUBLICATION = STUDY_DIR / "flagship_run3_result.json"
SUCCESSOR_LABEL = (
    "examples/hopper_kernel_cycle_candidate_v1/successors/"
    f"{SUCCESSOR_SHA256}/result.json"
)
ANCHOR_LABEL = "examples/deployment_curve_v1/expectations.json"
RUN3_LABEL = "examples/deployment_curve_v1/flagship_run3_result.json"
SUCCESSOR_SELECTOR = "/{lookup_record_sha256,score/{component_overlay_ledger,mtp}}"
ANCHOR_SELECTOR = f"/anchors[id={MTP_ANCHOR_ID}]"
RUN3_SELECTOR = (
    "/{schema,status,verdict,scope,allocation,scale_mapping,topology,"
    "pricing_configuration,attenuation_layer,constant_fit,constant_fit_sha256,"
    "held_out_score,anchor_access,anchor_predictions,curves,"
    "offered_load_sweep_requests_per_second,second_legend,"
    "decode_calibration_miss,residuals_required}"
)
RUN4_RESULT_LABEL = "wave-runs/core54run4/attempt-1/result.json"
RUN4_RESULT_SELECTOR = (
    "/{schema,status,verdict,classification,scope,core54_closure,closure_reason,"
    "allocation,shape_observation/{schema,status,anchor_id,allocation,request_count,"
    "requests_per_gpu,total_emitted_tokens,weights_loaded},fit,attenuation_layer,"
    "run3_carry_forward,mtp_score,access,preservation_lock,dominant_contributor,"
    "remaining_work,deployment_frontier,provenance}"
)
_WHITESPACE = b" \t\r\n"
_DELIMITERS = _WHITESPACE + b",]}"
_CAPTURE = object()

SUCCESSOR_PROJECTION: dict[str, Any] = {
    "lookup_record_sha256": _CAPTURE,
    "predecessor_lookup_record_sha256": _CAPTURE,
    "score": {
        "component_overlay_ledger": {
            "measured_unpriced_mtp": {"DISCLOSED": _CAPTURE}
        },
        "core61": {"status": _CAPTURE},
        "mtp": {
            "evidence_class": _CAPTURE,
            "lookup_pricing": _CAPTURE,
            "measured_service_ps": _CAPTURE,
        },
        "task_movement": {"comp74_repeat_inputs": _CAPTURE},
    },
}
RUN3_PROJECTION: dict[str, Any] = {
    name: _CAPTURE
    for name in (
        "schema",
        "status",
        "verdict",
        "scope",
        "allocation",
        "scale_mapping",
        "topology",
        "pricing_configuration",
        "attenuation_layer",
        "constant_fit",
        "constant_fit_sha256",
        "held_out_score",
        "anchor_access",
        "anchor_predictions",
        "curves",
        "offered_load_sweep_requests_per_second",
        "second_legend",
        "decode_calibration_miss",
        "residuals_required",
    )
}
RUN4_RESULT_PROJECTION: dict[str, Any] = {
    name: _CAPTURE
    for name in (
        "schema",
        "status",
        "verdict",
        "classification",
        "scope",
        "core54_closure",
        "closure_reason",
        "allocation",
        "fit",
        "attenuation_layer",
        "run3_carry_forward",
        "mtp_score",
        "access",
        "preservation_lock",
        "dominant_contributor",
        "remaining_work",
        "deployment_frontier",
        "provenance",
    )
}
RUN4_RESULT_PROJECTION["shape_observation"] = {
    name: _CAPTURE
    for name in (
        "schema",
        "status",
        "anchor_id",
        "allocation",
        "request_count",
        "requests_per_gpu",
        "total_emitted_tokens",
        "weights_loaded",
    )
}


class _Cursor:
    """One-byte cursor that never loads a complete record."""

    def __init__(self, stream: BinaryIO) -> None:
        self.stream = stream
        self.pending: bytes | None = None
        self.bytes_consumed = 0

    def read(self) -> bytes:
        if self.pending is not None:
            value = self.pending
            self.pending = None
        else:
            value = self.stream.read(1)
        if value:
            self.bytes_consumed += 1
        return value

    def peek(self) -> bytes:
        if self.pending is None:
            self.pending = self.stream.read(1)
        return self.pending


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


def _project_object(
    cursor: _Cursor,
    projection: Mapping[str, Any],
    first: bytes,
) -> dict[str, Any]:
    if first != b"{":
        raise ValueError("selected field must be a JSON object")
    result: dict[str, Any] = {}
    current = _skip_space(cursor)
    if current == b"}":
        return result
    while True:
        key = _read_key(cursor, current)
        _expect(cursor, b":")
        opening = _skip_space(cursor)
        selected = projection.get(key)
        if selected is _CAPTURE:
            result[key] = _capture_value(cursor, opening)
        elif isinstance(selected, Mapping):
            result[key] = _project_object(cursor, selected, opening)
        else:
            _skip_value(cursor, opening)
        current = _skip_space(cursor)
        if current == b"}":
            return result
        if current != b",":
            raise ValueError("malformed selected JSON object")
        current = _skip_space(cursor)


def extract_successor_evidence(stream: BinaryIO) -> tuple[dict[str, Any], int]:
    """Project only the successor fields permitted by the run-4 protocol."""

    cursor = _Cursor(stream)
    opening = _skip_space(cursor)
    value = _project_object(cursor, SUCCESSOR_PROJECTION, opening)
    return value, cursor.bytes_consumed


def extract_run3_publication(stream: BinaryIO) -> tuple[dict[str, Any], int]:
    """Project only the prior publication fields needed for immutable carry."""

    cursor = _Cursor(stream)
    opening = _skip_space(cursor)
    value = _project_object(cursor, RUN3_PROJECTION, opening)
    return value, cursor.bytes_consumed


def extract_run4_result(stream: BinaryIO) -> tuple[dict[str, Any], int]:
    """Project the scored result without returning its bulk request trace."""

    cursor = _Cursor(stream)
    opening = _skip_space(cursor)
    value = _project_object(cursor, RUN4_RESULT_PROJECTION, opening)
    return value, cursor.bytes_consumed


def _project_object_id(cursor: _Cursor, first: bytes) -> str | None:
    projected = _project_object(cursor, {"id": _CAPTURE}, first)
    identifier = projected.get("id")
    if identifier is not None and not isinstance(identifier, str):
        raise TypeError("anchor id must be a string")
    return identifier


def extract_mtp_anchor(
    stream: BinaryIO,
) -> tuple[dict[str, Any], int, int, int]:
    """Locate and decode only the MTP anchor object from a seekable stream."""

    cursor = _Cursor(stream)
    _expect(cursor, b"{")
    current = _skip_space(cursor)
    while current != b"}":
        key = _read_key(cursor, current)
        _expect(cursor, b":")
        opening = _skip_space(cursor)
        if key != "anchors":
            _skip_value(cursor, opening)
        else:
            if opening != b"[":
                raise ValueError("anchors must be an array")
            current = _skip_space(cursor)
            while current != b"]":
                if current != b"{":
                    raise ValueError("anchor row must be an object")
                start = stream.tell() - 1
                identifier = _project_object_id(cursor, current)
                if identifier == MTP_ANCHOR_ID:
                    stream.seek(start)
                    selected_cursor = _Cursor(stream)
                    selected_opening = selected_cursor.read()
                    selected = _capture_value(selected_cursor, selected_opening)
                    if not isinstance(selected, dict):
                        raise TypeError("MTP anchor must be an object")
                    length = stream.tell() - start
                    return selected, start, length, cursor.bytes_consumed
                current = _skip_space(cursor)
                if current == b"]":
                    break
                if current != b",":
                    raise ValueError("malformed anchors array")
                current = _skip_space(cursor)
            raise ValueError("MTP anchor is missing")
        current = _skip_space(cursor)
        if current == b"}":
            break
        if current != b",":
            raise ValueError("malformed top-level JSON object")
        current = _skip_space(cursor)
    raise ValueError("anchors field is missing")


def _append_access(log_path: Path, entry: Mapping[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(dict(entry), sort_keys=True) + "\n")


def read_successor_mtp_evidence(record_path: Path, access_log: Path) -> dict[str, Any]:
    """Read the allowlisted successor projection and log the access."""

    entry: dict[str, Any] = {
        "schema": ACCESS_SCHEMA,
        "classification": "measured_mtp_evidence",
        "record": SUCCESSOR_LABEL,
        "selector": SUCCESSOR_SELECTOR,
        "whole_record_loaded": False,
        "unselected_values_returned": False,
    }
    try:
        if record_path.resolve() != SUCCESSOR_RESULT.resolve():
            raise ValueError("run-4 reader refuses every non-allowlisted successor")
        with record_path.open("rb", buffering=0) as stream:
            value, consumed = extract_successor_evidence(stream)
        if value.get("lookup_record_sha256") != SUCCESSOR_SHA256:
            raise ValueError("successor identity differs from its content address")
        entry.update({"bytes_consumed": consumed, "status": "PASS"})
        return value
    except Exception as exc:
        entry.update({"error": type(exc).__name__, "status": "REJECTED"})
        raise
    finally:
        _append_access(access_log, entry)


def read_mtp_anchor(record_path: Path, access_log: Path) -> dict[str, Any]:
    """Read the sole run-4 held-out row and append one auditable access."""

    entry: dict[str, Any] = {
        "schema": ACCESS_SCHEMA,
        "classification": "held_out",
        "record": ANCHOR_LABEL,
        "selector": ANCHOR_SELECTOR,
        "anchor_id": MTP_ANCHOR_ID,
        "whole_record_loaded": False,
        "unselected_values_returned": False,
    }
    try:
        if record_path.resolve() != ANCHOR_FREEZE.resolve():
            raise ValueError("run-4 reader refuses every non-allowlisted anchor record")
        with record_path.open("rb", buffering=0) as stream:
            value, offset, length, scanned = extract_mtp_anchor(stream)
        if value.get("id") != MTP_ANCHOR_ID or value.get("role") != "held-out":
            raise ValueError("MTP anchor identity or held-out role differs")
        entry.update(
            {
                "bytes_scanned": scanned,
                "length": length,
                "offset": offset,
                "status": "PASS",
            }
        )
        return value
    except Exception as exc:
        entry.update({"error": type(exc).__name__, "status": "REJECTED"})
        raise
    finally:
        _append_access(access_log, entry)


def read_run3_publication(record_path: Path, access_log: Path) -> dict[str, Any]:
    """Read the allowlisted run-3 carry-forward projection and log it."""

    entry: dict[str, Any] = {
        "schema": ACCESS_SCHEMA,
        "classification": "inherited_run3_publication",
        "record": RUN3_LABEL,
        "selector": RUN3_SELECTOR,
        "whole_record_loaded": False,
        "unselected_values_returned": False,
    }
    try:
        if record_path.resolve() != RUN3_PUBLICATION.resolve():
            raise ValueError("run-4 reader refuses every non-allowlisted run-3 record")
        with record_path.open("rb", buffering=0) as stream:
            value, consumed = extract_run3_publication(stream)
        if value.get("schema") != "simllm-deployment-curve-flagship-run3-publication-v1":
            raise ValueError("run-3 publication schema differs")
        entry.update({"bytes_consumed": consumed, "status": "PASS"})
        return value
    except Exception as exc:
        entry.update({"error": type(exc).__name__, "status": "REJECTED"})
        raise
    finally:
        _append_access(access_log, entry)


def read_run4_result(
    record_path: Path,
    access_log: Path,
    run_root: Path,
) -> dict[str, Any]:
    """Read the sole external attempt through its publication projection."""

    entry: dict[str, Any] = {
        "schema": ACCESS_SCHEMA,
        "classification": "run4_publication",
        "record": RUN4_RESULT_LABEL,
        "selector": RUN4_RESULT_SELECTOR,
        "whole_record_loaded": False,
        "unselected_values_returned": False,
    }
    try:
        resolved_root = run_root.resolve()
        if resolved_root.parts[-2:] != ("wave-runs", "core54run4"):
            raise ValueError("run-4 publication root differs")
        expected = resolved_root / "attempt-1" / "result.json"
        if record_path.resolve() != expected:
            raise ValueError("run-4 reader refuses every non-allowlisted result")
        with record_path.open("rb", buffering=0) as stream:
            value, consumed = extract_run4_result(stream)
        if value.get("schema") != "simllm-deployment-curve-flagship-run4-result-v1":
            raise ValueError("run-4 result schema differs")
        entry.update({"bytes_consumed": consumed, "status": "PASS"})
        return value
    except Exception as exc:
        entry.update({"error": type(exc).__name__, "status": "REJECTED"})
        raise
    finally:
        _append_access(access_log, entry)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("evidence", "run3", "anchor", "result"))
    parser.add_argument("--access-log", required=True, type=Path)
    parser.add_argument("--run-root", type=Path)
    args = parser.parse_args()
    if args.kind == "evidence":
        value = read_successor_mtp_evidence(SUCCESSOR_RESULT, args.access_log)
    elif args.kind == "run3":
        value = read_run3_publication(RUN3_PUBLICATION, args.access_log)
    elif args.kind == "anchor":
        value = read_mtp_anchor(ANCHOR_FREEZE, args.access_log)
    else:
        if args.run_root is None:
            raise SystemExit("--run-root is required for a result projection")
        value = read_run4_result(
            args.run_root / "attempt-1" / "result.json",
            args.access_log,
            args.run_root,
        )
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
