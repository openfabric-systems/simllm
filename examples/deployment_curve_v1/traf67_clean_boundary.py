"""Validate TRAF-67's literal clean repetition of the TRAF-66 boundary."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any

from traf66_independent_sign import sign_visible_movement
from traf66_overlap_boundary import (
    calibration_comparison,
    compare_component_inputs,
    verify_preservation_locks,
)
from traf66_overlap_boundary import (
    validate_expectations as validate_traf66_expectations,
)

EXPECTATIONS_SCHEMA = "simllm-deployment-curve-traf67-expectations-v1"
RESULT_SCHEMA = "simllm-deployment-curve-traf67-calibration-v1"
ACCESS_SCHEMA = "simllm-deployment-curve-traf67-access-v1"
VISIBLE_ANCHOR_ID = "sglang_prefill_1k"
SELECTOR = "/calibration_rows[anchor_id=sglang_prefill_1k]"
FROZEN_OPERATOR = "max(C, P) + min(C, P) / 2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fraction(value: Mapping[str, Any], name: str) -> Fraction:
    if set(value) != {"numerator", "denominator"}:
        raise ValueError(f"{name} must contain numerator and denominator")
    numerator = value["numerator"]
    denominator = value["denominator"]
    if type(numerator) is not int or type(denominator) is not int:
        raise TypeError(f"{name} must contain integers")
    if denominator <= 0:
        raise ValueError(f"{name}.denominator must be positive")
    return Fraction(numerator, denominator)


def _percent(value: Fraction) -> str:
    decimal = Decimal(value.numerator * 100) / Decimal(value.denominator)
    return str(decimal.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP))


def validate_expectations(expectations: Mapping[str, Any]) -> None:
    """Validate the expectations-only clean-repetition contract."""

    if expectations.get("schema") != EXPECTATIONS_SCHEMA:
        raise ValueError("TRAF-67 expectations schema disagrees")
    if expectations.get("task") != "TRAF-67":
        raise ValueError("TRAF-67 task identity disagrees")
    if expectations.get("status") != "EXPECTATIONS_ONLY":
        raise ValueError("TRAF-67 expectations must precede record-field access")

    protocol = expectations["exposure_protocol"]
    if protocol["access_log_schema"] != ACCESS_SCHEMA:
        raise ValueError("TRAF-67 access schema disagrees")
    if protocol["permitted_anchor_ids"] != [VISIBLE_ANCHOR_ID]:
        raise ValueError("TRAF-67 visible allowlist disagrees")
    if protocol["forbidden_anchor_ids"] != [
        "sglang_prefill_2k",
        "sglang_prefill_4k",
    ]:
        raise ValueError("TRAF-67 held-out identities disagree")
    if protocol["expected_visible_access_count"] != 1:
        raise ValueError("TRAF-67 must perform exactly one visible access")
    if protocol["expected_held_out_access_ledger"] != []:
        raise ValueError("TRAF-67 held-out ledger must be empty")
    if protocol["whole_record_reads_permitted"]:
        raise ValueError("TRAF-67 cannot permit whole-record reads")
    if protocol["selector"] != SELECTOR:
        raise ValueError("TRAF-67 field selector disagrees")

    frozen = expectations["frozen_repetition"]
    if frozen["operator"] != FROZEN_OPERATOR:
        raise ValueError("TRAF-67 operator disagrees")
    amended = (
        frozen["source_ranges_amended"],
        frozen["event_ledger_amended"],
        frozen["component_service_envelope_amended"],
        frozen["parameters_amended_or_refit"],
    )
    if any(amended):
        raise ValueError("TRAF-67 cannot amend or refit the frozen repetition")
    if frozen["expected_preservation_lock_count"] != 27:
        raise ValueError("TRAF-67 preservation count disagrees")
    if frozen["expected_signed_movement"] != {
        "prediction": "decrease",
        "signed_residual": "more_negative",
    }:
        raise ValueError("TRAF-67 signed movement expectation disagrees")
    if any(expectations["scope_locks"].values()):
        raise ValueError("TRAF-67 scope locks must all remain false")


def verify_frozen_lineage(
    expectations: Mapping[str, Any],
    traf66_expectations: Mapping[str, Any],
    study_dir: Path,
) -> dict[str, Any]:
    """Verify the three reused TRAF-66 artifacts and frozen sections."""

    validate_expectations(expectations)
    validate_traf66_expectations(traf66_expectations)
    frozen = expectations["frozen_repetition"]
    observed = {
        "traf66_expectations_sha256": _sha256(study_dir / "traf66_expectations.json"),
        "traf66_overlap_boundary_sha256": _sha256(
            study_dir / "traf66_overlap_boundary.py"
        ),
        "traf66_independent_sign_sha256": _sha256(
            study_dir / "traf66_independent_sign.py"
        ),
    }
    for key, digest in observed.items():
        if frozen[key] != digest:
            raise ValueError(f"TRAF-67 frozen lineage disagrees for {key}")
    if traf66_expectations["composition"]["operator"] != FROZEN_OPERATOR:
        raise ValueError("TRAF-66 source operator changed")
    if len(traf66_expectations["preservation_locks"]) != 27:
        raise ValueError("TRAF-66 preservation class changed")
    return {
        **observed,
        "component_service_envelope_reused": True,
        "event_ledger_reused": True,
        "parameters_amended_or_refit": False,
        "source_contracts_reused": True,
    }


def validate_visible_access(
    expectations: Mapping[str, Any],
    access_entries: Sequence[Mapping[str, Any]],
    *,
    record_size_bytes: int,
) -> dict[str, Any]:
    """Validate one early-stopping access and an empty held-out ledger."""

    validate_expectations(expectations)
    expected_count = expectations["exposure_protocol"][
        "expected_visible_access_count"
    ]
    if len(access_entries) != expected_count:
        raise ValueError("TRAF-67 visible access count disagrees")
    entry = dict(access_entries[0])
    if entry.get("schema") != ACCESS_SCHEMA:
        raise ValueError("TRAF-67 visible access schema disagrees")
    if entry.get("classification") != "visible_calibration":
        raise ValueError("TRAF-67 access classification disagrees")
    if entry.get("selector") != SELECTOR:
        raise ValueError("TRAF-67 accessed an unpermitted field")
    if entry.get("status") != "PASS" or entry.get("whole_record_loaded") is not False:
        raise ValueError("TRAF-67 visible access was not clean")
    consumed = entry.get("bytes_consumed")
    if type(consumed) is not int or not 0 < consumed < record_size_bytes:
        raise ValueError("TRAF-67 reader did not stop before end of record")
    return {
        "visible_access_count": 1,
        "visible_access_ledger": [entry],
        "held_out_access_ledger": [],
        "held_out_numeric_values_accessed_or_compared": False,
        "record_size_bytes": record_size_bytes,
        "reader_stopped_before_end_of_record": True,
        "whole_record_loaded": False,
    }


def build_result(
    expectations: Mapping[str, Any],
    traf66_expectations: Mapping[str, Any],
    comp75_expectations: Mapping[str, Any],
    visible_row: Mapping[str, Any],
    access_entries: Sequence[Mapping[str, Any]],
    *,
    repository_root: Path,
    study_dir: Path,
    expectations_commit: str,
    record_size_bytes: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the clean result and full event ledger from the permitted row."""

    lineage = verify_frozen_lineage(expectations, traf66_expectations, study_dir)
    access = validate_visible_access(
        expectations,
        access_entries,
        record_size_bytes=record_size_bytes,
    )
    if visible_row.get("anchor_id") != VISIBLE_ANCHOR_ID:
        raise ValueError("TRAF-67 visible projection identity disagrees")
    comp75_projection = {
        "schema": "simllm-deployment-curve-comp75-calibration-v1",
        "accessed_visible_anchor_ids": [VISIBLE_ANCHOR_ID],
        "held_out_numeric_values_accessed": False,
        "calibration_rows": [dict(visible_row)],
    }
    components = compare_component_inputs(traf66_expectations, comp75_expectations)
    if not all(components.values()):
        raise ValueError("TRAF-67 component reuse disagrees")
    row = calibration_comparison(traf66_expectations, comp75_projection)
    composition = traf66_expectations["composition"]
    counts = traf66_expectations["event_conservation"]["counts"]
    independent = sign_visible_movement(
        per_node_tokens=row["per_node_tokens"],
        published_numerator=row["published"]["numerator"],
        published_denominator=row["published"]["denominator"],
        compute_service_ps=composition["candidate_compute_service_ps"],
        packet_service_ps=composition["packet_service_ps"]["selected"],
        children=counts["children"],
    )
    expected_sign = expectations["frozen_repetition"]["expected_signed_movement"]
    if independent["movement"]["direction"] != expected_sign["prediction"]:
        raise ValueError("TRAF-67 independent prediction movement disagrees")
    if (
        independent["signed_residual_movement"]["direction"]
        != expected_sign["signed_residual"]
    ):
        raise ValueError("TRAF-67 independent residual movement disagrees")

    before = _fraction(row["signed_relative_error_before"], "residual before")
    after = _fraction(row["signed_relative_error_after"], "residual after")
    percentages = {
        "residual_before_percent": _percent(before),
        "residual_after_percent": _percent(after),
        "residual_movement_percentage_points": _percent(after - before),
    }
    if percentages != expectations["frozen_repetition"][
        "expected_visible_percentages"
    ]:
        raise ValueError("TRAF-67 visible percentages do not reproduce TRAF-66")

    preservation = verify_preservation_locks(traf66_expectations, repository_root)
    if preservation["checked_count"] != 27:
        raise ValueError("TRAF-67 preservation lock count disagrees")
    result = {
        "schema": RESULT_SCHEMA,
        "status": "PASS_CLEAN_REPETITION",
        "task": "TRAF-67",
        "expectations_commit": expectations_commit,
        "access": access,
        "calibration_rows": [row],
        "composition": composition,
        "event_conservation": traf66_expectations["event_conservation"],
        "frozen_lineage": lineage,
        "independent_signoff": independent,
        "preservation_lock": {
            "checked_count": preservation["checked_count"],
            "prior_records_mutated": preservation["prior_records_mutated"],
            "status": preservation["status"],
        },
        "scope_locks": dict(expectations["scope_locks"]),
        "visible_percentages": percentages,
    }
    event_ledger = {
        "schema": "simllm-deployment-curve-traf67-event-ledger-v1",
        "status": "PASS",
        "task": "TRAF-67",
        "access": access,
        "component_reuse": components,
        "composition": composition,
        "event_conservation": traf66_expectations["event_conservation"],
        "frozen_lineage": lineage,
        "preservation_lock": preservation,
        "source_contracts": traf66_expectations["source_contracts"],
    }
    return result, event_ledger


__all__ = [
    "ACCESS_SCHEMA",
    "EXPECTATIONS_SCHEMA",
    "RESULT_SCHEMA",
    "build_result",
    "validate_expectations",
    "validate_visible_access",
    "verify_frozen_lineage",
]
