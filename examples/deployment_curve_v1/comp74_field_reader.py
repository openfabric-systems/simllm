"""Logged field-addressed readers for the COMP-74 distribution study."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, BinaryIO

from flagship_run4_field_reader import (
    _CAPTURE,
    _capture_value,
    _Cursor,
    _expect,
    _read_key,
    _skip_space,
    _skip_value,
)

ACCESS_SCHEMA = "simllm-deployment-curve-comp74-field-access-v1"
SUCCESSOR_SHA256 = "d868a4f35d633032daa238168d00f42c2ab47fc569db649b19b907008072e107"
STUDY_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = STUDY_DIR.parents[1]
SUCCESSOR_RESULT = (
    REPOSITORY_ROOT
    / "examples/hopper_kernel_cycle_candidate_v1/successors"
    / SUCCESSOR_SHA256
    / "result.json"
)
RUN4_PUBLICATION = STUDY_DIR / "flagship_run4_result.json"
CURVE_CONFIG = STUDY_DIR / "flagship_run2_config.json"
SUCCESSOR_LABEL = (
    "examples/hopper_kernel_cycle_candidate_v1/successors/"
    f"{SUCCESSOR_SHA256}/result.json"
)
RUN4_LABEL = "examples/deployment_curve_v1/flagship_run4_result.json"
CURVE_CONFIG_LABEL = "examples/deployment_curve_v1/flagship_run2_config.json"
SUCCESSOR_SELECTOR = (
    "/{acceptance_status,lookup_record_sha256,predecessor_lookup_record_sha256,"
    "score/{lookup_service_ledger,priced_repeat_observations[]/"
    "{implementation_suffix,published_point_ps,independent_repeat_ps,"
    "signed_repeat_minus_point_ps,retained_independent_observations,"
    "distribution_propagation},task_movement/{comp74_repeat_inputs}}}"
)
RUN4_SELECTOR = (
    "/{schema,status,verdict,core54_closure,run3_carry_forward/"
    "{status,authority_sha256,held_out_score,anchor_predictions,curves,"
    "decode_calibration_miss},mtp_score,provenance/{candidate_acceptance_status}}"
)
CURVE_CONFIG_SELECTOR = "/{schema,publication_curves}"

REPEAT_ROW_PROJECTION: dict[str, Any] = {
    name: _CAPTURE
    for name in (
        "implementation_suffix",
        "published_point_ps",
        "independent_repeat_ps",
        "signed_repeat_minus_point_ps",
        "retained_independent_observations",
        "distribution_propagation",
    )
}
SUCCESSOR_PROJECTION: dict[str, Any] = {
    "acceptance_status": _CAPTURE,
    "lookup_record_sha256": _CAPTURE,
    "predecessor_lookup_record_sha256": _CAPTURE,
    "score": {
        "lookup_service_ledger": _CAPTURE,
        "priced_repeat_observations": (REPEAT_ROW_PROJECTION,),
        "task_movement": {"comp74_repeat_inputs": _CAPTURE},
    },
}
RUN4_PROJECTION: dict[str, Any] = {
    "schema": _CAPTURE,
    "status": _CAPTURE,
    "verdict": _CAPTURE,
    "core54_closure": _CAPTURE,
    "run3_carry_forward": {
        name: _CAPTURE
        for name in (
            "status",
            "authority_sha256",
            "held_out_score",
            "anchor_predictions",
            "curves",
            "decode_calibration_miss",
        )
    },
    "mtp_score": _CAPTURE,
    "provenance": {"candidate_acceptance_status": _CAPTURE},
}
CURVE_CONFIG_PROJECTION: dict[str, Any] = {
    "schema": _CAPTURE,
    "publication_curves": _CAPTURE,
}


def _project_array(
    cursor: _Cursor,
    projection: Mapping[str, Any],
    first: bytes,
) -> list[dict[str, Any]]:
    if first != b"[":
        raise ValueError("selected field must be a JSON array")
    rows: list[dict[str, Any]] = []
    current = _skip_space(cursor)
    if current == b"]":
        return rows
    while True:
        rows.append(_project_object(cursor, projection, current))
        current = _skip_space(cursor)
        if current == b"]":
            return rows
        if current != b",":
            raise ValueError("malformed selected JSON array")
        current = _skip_space(cursor)


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
        elif (
            isinstance(selected, tuple)
            and len(selected) == 1
            and isinstance(selected[0], Mapping)
        ):
            result[key] = _project_array(cursor, selected[0], opening)
        else:
            _skip_value(cursor, opening)
        current = _skip_space(cursor)
        if current == b"}":
            return result
        if current != b",":
            raise ValueError("malformed selected JSON object")
        current = _skip_space(cursor)


def _extract(
    stream: BinaryIO,
    projection: Mapping[str, Any],
) -> tuple[dict[str, Any], int]:
    cursor = _Cursor(stream)
    opening = _skip_space(cursor)
    value = _project_object(cursor, projection, opening)
    return value, cursor.bytes_consumed


def extract_successor_repeats(stream: BinaryIO) -> tuple[dict[str, Any], int]:
    """Return only the frozen successor repeat and ledger projection."""

    return _extract(stream, SUCCESSOR_PROJECTION)


def extract_run4_publication(stream: BinaryIO) -> tuple[dict[str, Any], int]:
    """Return only the prior scored fields needed for propagation."""

    return _extract(stream, RUN4_PROJECTION)


def extract_curve_config(stream: BinaryIO) -> tuple[dict[str, Any], int]:
    """Return only the curve-construction configuration subtree."""

    return _extract(stream, CURVE_CONFIG_PROJECTION)


def _append_access(path: Path, entry: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(dict(entry), sort_keys=True) + "\n")


def _read_projection(
    path: Path,
    expected: Path,
    access_log: Path,
    *,
    classification: str,
    record_label: str,
    selector: str,
    projection: Mapping[str, Any],
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "schema": ACCESS_SCHEMA,
        "classification": classification,
        "record": record_label,
        "selector": selector,
        "whole_record_loaded": False,
        "unselected_values_returned": False,
    }
    try:
        if path.resolve() != expected.resolve():
            raise ValueError(f"COMP-74 refuses non-allowlisted {classification} input")
        with path.open("rb", buffering=0) as stream:
            value, consumed = _extract(stream, projection)
        entry.update({"bytes_consumed": consumed, "status": "PASS"})
        return value
    except Exception as exc:
        entry.update({"error": type(exc).__name__, "status": "REJECTED"})
        raise
    finally:
        _append_access(access_log, entry)


def read_successor_repeats(path: Path, access_log: Path) -> dict[str, Any]:
    """Read and log only the successor repeat and evidence-ledger fields."""

    value = _read_projection(
        path,
        SUCCESSOR_RESULT,
        access_log,
        classification="successor_repeat_evidence",
        record_label=SUCCESSOR_LABEL,
        selector=SUCCESSOR_SELECTOR,
        projection=SUCCESSOR_PROJECTION,
    )
    if value.get("lookup_record_sha256") != SUCCESSOR_SHA256:
        raise ValueError("successor identity differs from its content address")
    return value


def read_run4_publication(path: Path, access_log: Path) -> dict[str, Any]:
    """Read and log only the preservation-locked run-4 projection."""

    value = _read_projection(
        path,
        RUN4_PUBLICATION,
        access_log,
        classification="prior_scored_publication",
        record_label=RUN4_LABEL,
        selector=RUN4_SELECTOR,
        projection=RUN4_PROJECTION,
    )
    if value.get("schema") != "simllm-deployment-curve-flagship-run4-publication-v1":
        raise ValueError("run-4 publication schema differs")
    return value


def read_curve_config(path: Path, access_log: Path) -> dict[str, Any]:
    """Read and log only the inherited curve configuration."""

    value = _read_projection(
        path,
        CURVE_CONFIG,
        access_log,
        classification="curve_configuration",
        record_label=CURVE_CONFIG_LABEL,
        selector=CURVE_CONFIG_SELECTOR,
        projection=CURVE_CONFIG_PROJECTION,
    )
    if value.get("schema") != "simllm-deployment-curve-flagship-run2-config-v1":
        raise ValueError("curve configuration schema differs")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("successor", "run4", "curve-config"))
    parser.add_argument("--access-log", required=True, type=Path)
    args = parser.parse_args()
    if args.kind == "successor":
        value = read_successor_repeats(SUCCESSOR_RESULT, args.access_log)
    elif args.kind == "run4":
        value = read_run4_publication(RUN4_PUBLICATION, args.access_log)
    else:
        value = read_curve_config(CURVE_CONFIG, args.access_log)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
