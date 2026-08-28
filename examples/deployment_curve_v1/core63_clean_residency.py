"""Frozen clean CORE-63 derivation, access checks, and publication helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from decimal import ROUND_HALF_UP, Decimal, localcontext
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any

from core63_clean_field_reader import ACCESS_SCHEMA
from core63_residency import compare_standard_calibration, derive_residency_step

CLEAN_EXPECTATIONS_SCHEMA = "simllm-deployment-curve-core63-clean-expectations-v1"
CLEAN_RETRY_SCHEMA = "simllm-deployment-curve-core63-clean-retry-v1"
CLEAN_FINAL_RETRY_SCHEMA = "simllm-deployment-curve-core63-clean-final-retry-v1"
CLEAN_REGISTRY_RETRY_SCHEMA = (
    "simllm-deployment-curve-core63-clean-registry-retry-v1"
)
CLEAN_RESULT_SCHEMA = "simllm-deployment-curve-core63-clean-calibration-v1"
EXPECTED_ACCESS_COUNT = 6
EXPECTED_ACCESS_EVENT_COUNT = 12
EXPECTED_CLASSIFICATION = "UNDERCORRECTION"
ROUTED_MARKER = "fused_moe_kernel"
PS_PER_SECOND = 1_000_000_000_000
PS_PER_NANOSECOND = 1_000
SOURCE_LAYERS = 4
TARGET_LAYERS = 61


def _require_mapping(name: str, value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return value


def _require_int(name: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _fraction_row(value: Fraction) -> dict[str, int]:
    return {"denominator": value.denominator, "numerator": value.numerator}


def _fraction_from_row(name: str, value: Mapping[str, Any]) -> Fraction:
    return Fraction(
        _require_int(f"{name}.numerator", value["numerator"]),
        _require_int(f"{name}.denominator", value["denominator"], minimum=1),
    )


def _decimal(value: Fraction, places: int = 6) -> str:
    with localcontext() as context:
        context.prec = 50
        converted = Decimal(value.numerator) / Decimal(value.denominator)
        return str(
            converted.quantize(
                Decimal(1).scaleb(-places),
                rounding=ROUND_HALF_UP,
            )
        )


def validate_clean_expectations(expectations: Mapping[str, Any]) -> None:
    """Validate the committed no-fit and no-score clean freeze."""

    if expectations.get("schema") != CLEAN_EXPECTATIONS_SCHEMA:
        raise ValueError("clean expectations schema differs")
    if expectations.get("task") != "CORE-63":
        raise ValueError("clean task identity differs")
    if expectations.get("status") != "EXPECTATIONS_ONLY":
        raise ValueError("clean expectations must precede source access")
    access = _require_mapping("access_freeze", expectations["access_freeze"])
    if access.get("expected_access_count") != EXPECTED_ACCESS_COUNT:
        raise ValueError("clean access count differs")
    if access.get("expected_access_event_count") != EXPECTED_ACCESS_EVENT_COUNT:
        raise ValueError("clean access event count differs")
    if access.get("expected_forbidden_access_ledger") != []:
        raise ValueError("forbidden access ledger must be empty")
    if access.get("held_out_mtp_numeric_values_accessed_or_compared"):
        raise ValueError("held-out MTP access cannot be frozen true")
    if access.get("whole_file_streams_permitted"):
        raise ValueError("whole-file selectors cannot be permitted")

    architecture = _require_mapping(
        "architecture_arithmetic", expectations["architecture_arithmetic"]
    )
    if architecture.get("assignment_formula") != "256 * 8 * 4 / 288":
        raise ValueError("assignment formula differs")
    formula = Fraction(
        _require_int("batch", architecture["disclosed_batch_per_node"])
        * _require_int("top_k", architecture["top_k"], minimum=1)
        * _require_int(
            "resident slots",
            architecture["resident_physical_slots_per_rank"],
            minimum=1,
        ),
        _require_int(
            "physical slots", architecture["physical_expert_slots"], minimum=1
        ),
    )
    if formula != Fraction(256, 9):
        raise ValueError("expected assignments differ from 256/9")
    scale = _fraction_from_row(
        "routed_assignment_scale", architecture["routed_assignment_scale"]
    )
    if scale != Fraction(1, 9):
        raise ValueError("routed assignment scale differs from 1/9")
    if architecture.get("capture_assignments") != 256:
        raise ValueError("capture assignment count differs")

    component = _require_mapping("component_rule", expectations["component_rule"])
    if component.get("case_insensitive_kernel_name_markers") != [ROUTED_MARKER]:
        raise ValueError("component marker differs")
    if not component.get("fixed_term_kept_once"):
        raise ValueError("fixed term must be kept once")
    if _fraction_from_row(
        "nonmatching scale", component["nonmatching_noncollective_scale"]
    ) != Fraction(1):
        raise ValueError("nonmatching component scale differs")
    if component.get("parameters_amended_or_refit"):
        raise ValueError("parameters cannot be amended or refit")
    if not component.get("zero_free_or_fitted_constants"):
        raise ValueError("clean derivation requires zero free constants")
    if expectations.get("expected_signed_direction") != {
        "corrected_step": "decrease",
        "prediction": "increase",
        "signed_residual": "less_negative_before_any_possible_crossing",
    }:
        raise ValueError("signed direction differs from the clean freeze")
    scope = _require_mapping("scope_locks", expectations["scope_locks"])
    if any(scope.values()):
        raise ValueError("all clean scope locks must remain false")


def validate_retry_expectations(expectations: Mapping[str, Any]) -> None:
    """Validate the access-only retry freeze after the safe preflight rejection."""

    if expectations.get("schema") != CLEAN_RETRY_SCHEMA:
        raise ValueError("clean retry schema differs")
    if expectations.get("task") != "CORE-63":
        raise ValueError("clean retry task identity differs")
    if expectations.get("status") != "EXPECTATIONS_ONLY_ACCESS_RETRY":
        raise ValueError("clean retry must be frozen before the next access")
    if expectations.get("arithmetic_or_direction_amended"):
        raise ValueError("clean retry cannot amend arithmetic or direction")
    prior = _require_mapping("prior_preflight", expectations["prior_preflight"])
    if prior.get("event_count") != 6:
        raise ValueError("preflight event count differs")
    if prior.get("end_statuses") != ["PASS", "PASS", "REJECTED"]:
        raise ValueError("preflight status sequence differs")
    if prior.get("rejection") != "WholeFileAccessRejected":
        raise ValueError("preflight rejection type differs")
    if prior.get("whole_file_streamed"):
        raise ValueError("preflight cannot report a whole-file stream")
    if prior.get("held_out_mtp_numeric_value_accessed"):
        raise ValueError("preflight cannot report held-out MTP exposure")
    retry = _require_mapping("retry_access", expectations["retry_access"])
    if retry.get("access_pattern") != "header_plus_reverse_tail":
        raise ValueError("clean retry access pattern differs")
    if retry.get("expected_access_count") != EXPECTED_ACCESS_COUNT:
        raise ValueError("clean retry access count differs")
    if retry.get("expected_event_count") != EXPECTED_ACCESS_EVENT_COUNT:
        raise ValueError("clean retry event count differs")
    if retry.get("whole_file_streams_permitted"):
        raise ValueError("clean retry cannot permit whole-file streams")
    if expectations.get("expected_forbidden_access_ledger") != []:
        raise ValueError("clean retry forbidden ledger must remain empty")


def validate_final_retry_expectations(expectations: Mapping[str, Any]) -> None:
    """Validate the terminal-byte omission freeze before the final access."""

    if expectations.get("schema") != CLEAN_FINAL_RETRY_SCHEMA:
        raise ValueError("clean final retry schema differs")
    if expectations.get("task") != "CORE-63":
        raise ValueError("clean final retry task identity differs")
    if expectations.get("status") != "EXPECTATIONS_ONLY_FINAL_ACCESS_RETRY":
        raise ValueError("clean final retry must precede the final access")
    if expectations.get("arithmetic_or_direction_amended"):
        raise ValueError("clean final retry cannot amend arithmetic or direction")
    if expectations.get("expected_forbidden_access_ledger") != []:
        raise ValueError("clean final retry forbidden ledger must remain empty")
    prior = expectations.get("prior_safe_rejections")
    if not isinstance(prior, list) or len(prior) != 2:
        raise ValueError("clean final retry must bind two safe rejections")
    for row in prior:
        if row.get("end_statuses") != ["PASS", "PASS", "REJECTED"]:
            raise ValueError("safe rejection status sequence differs")
        if row.get("rejection") != "WholeFileAccessRejected":
            raise ValueError("safe rejection type differs")
        if row.get("whole_file_streamed"):
            raise ValueError("safe rejection cannot report a whole-file stream")
        if row.get("held_out_mtp_numeric_value_accessed"):
            raise ValueError("safe rejection cannot report MTP exposure")
    access = _require_mapping("final_access", expectations["final_access"])
    if access.get("access_pattern") != (
        "frozen_legacy_schema_plus_reverse_nonterminal_bytes"
    ):
        raise ValueError("clean final access pattern differs")
    if access.get("expected_csv_bytes_accessed") != 13_984:
        raise ValueError("clean final CSV byte count differs")
    if access.get("expected_csv_record_size_bytes") != 13_985:
        raise ValueError("clean final CSV record size differs")
    if access.get("terminal_byte_accessed"):
        raise ValueError("clean final access cannot touch the terminal byte")
    if access.get("whole_file_streams_permitted"):
        raise ValueError("clean final access cannot permit whole-file streams")


def validate_registry_retry_expectations(expectations: Mapping[str, Any]) -> None:
    """Validate the syntax-only registry selector correction."""

    if expectations.get("schema") != CLEAN_REGISTRY_RETRY_SCHEMA:
        raise ValueError("clean registry retry schema differs")
    if expectations.get("task") != "CORE-63":
        raise ValueError("clean registry retry task identity differs")
    if expectations.get("status") != "EXPECTATIONS_ONLY_REGISTRY_SELECTOR_RETRY":
        raise ValueError("clean registry retry must precede source access")
    if expectations.get("arithmetic_or_direction_amended"):
        raise ValueError("registry retry cannot amend arithmetic or direction")
    if expectations.get("expected_forbidden_access_ledger") != []:
        raise ValueError("registry retry forbidden ledger must remain empty")
    correction = _require_mapping(
        "registry_selector_correction",
        expectations["registry_selector_correction"],
    )
    if correction.get("before") != "| CORE-63 |":
        raise ValueError("registry retry prior selector differs")
    if correction.get("after") != "CORE-63":
        raise ValueError("registry retry corrected selector differs")
    if correction.get("arithmetic_input"):
        raise ValueError("registry text cannot become an arithmetic input")
    retry = _require_mapping("retry_access", expectations["retry_access"])
    if retry.get("expected_access_count") != EXPECTED_ACCESS_COUNT:
        raise ValueError("registry retry access count differs")
    if retry.get("expected_event_count") != EXPECTED_ACCESS_EVENT_COUNT:
        raise ValueError("registry retry event count differs")
    if retry.get("whole_file_streams_permitted"):
        raise ValueError("registry retry cannot permit whole-file streams")


def arithmetic_expectations(
    clean_expectations: Mapping[str, Any],
    calibration_context: Mapping[str, Any],
) -> dict[str, Any]:
    """Translate only frozen clean fields into the inherited exact calculator."""

    validate_clean_expectations(clean_expectations)
    architecture = clean_expectations["architecture_arithmetic"]
    top_k = architecture["top_k"]
    capture_assignments = architecture["capture_assignments"]
    if capture_assignments % top_k:
        raise ValueError("capture assignments do not divide by top-k")
    preservation = clean_expectations["preservation_freeze"]
    return {
        "architecture_arithmetic": {
            "assignment_scale": dict(architecture["routed_assignment_scale"]),
            "capture_batch_size": capture_assignments // top_k,
            "disclosed_batch_per_node": architecture["disclosed_batch_per_node"],
            "physical_expert_slots": architecture["physical_expert_slots"],
            "resident_physical_slots_per_rank": architecture[
                "resident_physical_slots_per_rank"
            ],
            "top_k": top_k,
            "uniform_routing_assumption": architecture[
                "uniform_routing_assumption"
            ],
        },
        "calibration_context": dict(calibration_context),
        "component_rule": {
            "routed_expert_kernel_name_markers": [ROUTED_MARKER],
            "zero_free_or_fitted_constants": True,
        },
        "expected_signed_direction": dict(
            clean_expectations["expected_signed_direction"]
        ),
        "preservation_lock": {
            "artifact_count": preservation["expected_lock_count"],
            "manifest_path": preservation["manifest_path"],
            "manifest_sha256": preservation["manifest_sha256"],
        },
        "schema": "simllm-deployment-curve-core63-expectations-v1",
        "status": "EXPECTATIONS_ONLY",
        "task": "CORE-63",
    }


def validate_access_events(
    events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Require six ordered begin/end pairs with strictly partial byte counts."""

    if len(events) != EXPECTED_ACCESS_EVENT_COUNT:
        raise ValueError("clean access event count differs")
    completed = []
    for index, event in enumerate(events, start=1):
        if event.get("event_index") != index:
            raise ValueError("access event indices are not contiguous")
        if event.get("schema") != ACCESS_SCHEMA:
            raise ValueError("access schema differs")
        if event.get("held_out_mtp_value_accessed") is not False:
            raise ValueError("access event reports held-out MTP exposure")
        if event.get("whole_file_streamed") is not False:
            raise ValueError("access event reports a whole-file stream")
    for access_number in range(1, EXPECTED_ACCESS_COUNT + 1):
        begin = events[2 * (access_number - 1)]
        end = events[2 * (access_number - 1) + 1]
        expected_id = f"A{access_number:02d}"
        if begin.get("access_id") != expected_id or end.get("access_id") != expected_id:
            raise ValueError("access identifiers differ")
        if begin.get("event") != "BEGIN" or begin.get("status") != "IN_PROGRESS":
            raise ValueError("access did not begin contemporaneously")
        if begin.get("bytes_accessed") != 0:
            raise ValueError("begin event must precede byte access")
        if end.get("event") != "END" or end.get("status") != "PASS":
            raise ValueError("clean access did not complete successfully")
        if begin.get("record") != end.get("record"):
            raise ValueError("access record changed within a pair")
        if begin.get("selector") != end.get("selector"):
            raise ValueError("access selector changed within a pair")
        consumed = end.get("bytes_accessed")
        size = end.get("record_size_bytes")
        if type(consumed) is not int or type(size) is not int:
            raise TypeError("access byte accounting must use integers")
        if not 0 < consumed < size:
            raise ValueError("selector did not stop before the final source byte")
        completed.append(dict(end))
    return {
        "access_count": EXPECTED_ACCESS_COUNT,
        "access_event_count": EXPECTED_ACCESS_EVENT_COUNT,
        "completed_accesses": completed,
        "contemporaneous_begin_before_open": True,
        "forbidden_access_ledger": [],
        "held_out_mtp_numeric_values_accessed_or_compared": False,
        "whole_file_streams": 0,
    }


def validate_preflight_events(
    events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate the logged safe rejection without treating it as input evidence."""

    if len(events) != 6:
        raise ValueError("preflight must contain three begin/end pairs")
    statuses = []
    for index, event in enumerate(events, start=1):
        if event.get("event_index") != index:
            raise ValueError("preflight event indices are not contiguous")
        if event.get("schema") != ACCESS_SCHEMA:
            raise ValueError("preflight access schema differs")
        if event.get("held_out_mtp_value_accessed") is not False:
            raise ValueError("preflight reports held-out MTP exposure")
        if event.get("whole_file_streamed") is not False:
            raise ValueError("preflight reports a whole-file stream")
    for access_number in range(1, 4):
        begin = events[2 * (access_number - 1)]
        end = events[2 * (access_number - 1) + 1]
        if begin.get("event") != "BEGIN" or begin.get("bytes_accessed") != 0:
            raise ValueError("preflight begin event was not contemporaneous")
        if end.get("event") != "END":
            raise ValueError("preflight end event is missing")
        consumed = end.get("bytes_accessed")
        size = end.get("record_size_bytes")
        if type(consumed) is not int or type(size) is not int:
            raise TypeError("preflight byte accounting must use integers")
        if not 0 < consumed < size:
            raise ValueError("preflight selector reached a whole-file stream")
        statuses.append(end.get("status"))
    if statuses != ["PASS", "PASS", "REJECTED"]:
        raise ValueError("preflight status sequence differs")
    final = events[-1]
    if final.get("error") != "WholeFileAccessRejected":
        raise ValueError("preflight did not reject the whole-file selector")
    return {
        "access_count": 3,
        "access_event_count": 6,
        "end_statuses": statuses,
        "forbidden_access_ledger": [],
        "held_out_mtp_numeric_values_accessed_or_compared": False,
        "rejected_before_final_byte": True,
        "rejection": "WholeFileAccessRejected",
        "whole_file_streams": 0,
    }


def validate_registry_preflight_events(
    events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate three clean source passes followed by a registry rejection."""

    if len(events) != 8:
        raise ValueError("registry preflight must contain four begin/end pairs")
    statuses = []
    for index, event in enumerate(events, start=1):
        if event.get("event_index") != index:
            raise ValueError("registry preflight event indices differ")
        if event.get("schema") != ACCESS_SCHEMA:
            raise ValueError("registry preflight access schema differs")
        if event.get("held_out_mtp_value_accessed") is not False:
            raise ValueError("registry preflight reports held-out MTP exposure")
        if event.get("whole_file_streamed") is not False:
            raise ValueError("registry preflight reports a whole-file stream")
    for access_number in range(1, 5):
        begin = events[2 * (access_number - 1)]
        end = events[2 * (access_number - 1) + 1]
        if begin.get("event") != "BEGIN" or begin.get("bytes_accessed") != 0:
            raise ValueError("registry preflight begin was not contemporaneous")
        if end.get("event") != "END":
            raise ValueError("registry preflight end is missing")
        consumed = end.get("bytes_accessed")
        size = end.get("record_size_bytes")
        if type(consumed) is not int or type(size) is not int:
            raise TypeError("registry preflight byte accounting must use integers")
        if not 0 < consumed < size:
            raise ValueError("registry preflight reached a whole-file stream")
        statuses.append(end.get("status"))
    if statuses != ["PASS", "PASS", "PASS", "REJECTED"]:
        raise ValueError("registry preflight status sequence differs")
    if events[-1].get("error") != "WholeFileAccessRejected":
        raise ValueError("registry preflight rejection type differs")
    return {
        "access_count": 4,
        "access_event_count": 8,
        "end_statuses": statuses,
        "forbidden_access_ledger": [],
        "held_out_mtp_numeric_values_accessed_or_compared": False,
        "rejected_before_final_byte": True,
        "rejection": "WholeFileAccessRejected",
        "whole_file_streams": 0,
    }


def independent_recompute(inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute step and calibration movement without calling CORE-63 helpers."""

    basis = _require_mapping("component_basis", inputs["component_basis"])
    kernels = basis.get("kernels")
    if not isinstance(kernels, list) or len(kernels) != 1:
        raise ValueError("independent basis must contain one aggregate kernel")
    kernel = _require_mapping("aggregate kernel", kernels[0])
    components = _require_mapping("components", kernel["components"])
    fixed = Fraction(_require_int("fixed service", components["fixed_overhead_ps"]))
    rows = inputs.get("kernel_rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("independent kernel row set is empty")
    raw = Fraction()
    routed = Fraction()
    classification = []
    for row in rows:
        name = row.get("name")
        duration = row.get("total_duration_per_step_ns")
        if not isinstance(name, str) or not isinstance(duration, str):
            raise TypeError("independent kernel fields differ")
        service = Fraction(duration) * PS_PER_NANOSECOND
        family = "routed_expert" if ROUTED_MARKER in name.lower() else "retained"
        raw += service
        if family == "routed_expert":
            routed += service
        classification.append(
            {
                "family": family,
                "first_launch_order": int(row["first_launch_order"]),
                "name": name,
                "service_ps": _fraction_row(service),
            }
        )
    measured = Fraction(_require_int("measured service", basis["measured_service_ps"]))
    if abs(raw - measured) > 1:
        raise ValueError("independent kernel total does not reconstruct the step")
    retained = raw - routed - fixed
    corrected = fixed + Fraction(TARGET_LAYERS, SOURCE_LAYERS) * (
        retained + routed / 9
    )
    context = _require_mapping("calibration_context", inputs["calibration_context"])
    tokens = _require_int("tokens", context["per_node_tokens"], minimum=1)
    anchor = _require_int(
        "standard anchor",
        context["published_tokens_per_second_per_node"],
        minimum=1,
    )
    current_step = _require_int("current step", context["current_step_ps"], minimum=1)
    current_prediction = Fraction(tokens * PS_PER_SECOND, current_step)
    corrected_prediction = Fraction(tokens * PS_PER_SECOND, 1) / corrected
    current_residual = current_prediction / anchor - 1
    corrected_residual = corrected_prediction / anchor - 1
    classification_name = (
        "PASS"
        if abs(corrected_residual) <= Fraction(5, 100)
        else "UNDERCORRECTION"
        if corrected_residual < 0
        else "OVERCORRECTION"
    )
    return {
        "calibration": {
            "classification": classification_name,
            "corrected_prediction": _fraction_row(corrected_prediction),
            "corrected_signed_residual": _fraction_row(corrected_residual),
            "prediction_movement": _fraction_row(
                corrected_prediction - current_prediction
            ),
            "signed_residual_movement": _fraction_row(
                corrected_residual - current_residual
            ),
        },
        "component_classification_ledger": classification,
        "fixed_service_ps": _fraction_row(fixed),
        "raw_service_ps": _fraction_row(raw),
        "retained_service_ps": _fraction_row(retained),
        "routed_service_ps": _fraction_row(routed),
        "step": {
            "corrected_ps": _fraction_row(corrected),
            "formula": "F + 61/4 * (retained_4 + routed_4/9)",
        },
    }


def cross_check_recomputation(
    derivation: Mapping[str, Any],
    calibration: Mapping[str, Any],
    independent: Mapping[str, Any],
) -> None:
    """Require exact equality between the two implementations."""

    primary_step = _fraction_from_row(
        "primary step", derivation["step"]["residency_corrected_ps"]
    )
    independent_step = _fraction_from_row(
        "independent step", independent["step"]["corrected_ps"]
    )
    if primary_step != independent_step:
        raise ValueError("independent corrected step differs")
    primary_prediction = _fraction_from_row(
        "primary prediction",
        calibration["residency_corrected"]["prediction_exact"],
    )
    independent_prediction = _fraction_from_row(
        "independent prediction",
        independent["calibration"]["corrected_prediction"],
    )
    if primary_prediction != independent_prediction:
        raise ValueError("independent corrected prediction differs")
    if (
        calibration["residency_corrected"]["classification"]
        != independent["calibration"]["classification"]
    ):
        raise ValueError("independent classification differs")


def verify_preservation(
    clean_expectations: Mapping[str, Any], repository_root: Path
) -> dict[str, Any]:
    """Verify the frozen 93-file manifest without decoding artifact values."""

    freeze = clean_expectations["preservation_freeze"]
    manifest = repository_root / str(freeze["manifest_path"])
    payload = manifest.read_bytes()
    observed_manifest_sha = hashlib.sha256(payload).hexdigest()
    if observed_manifest_sha != freeze["manifest_sha256"]:
        raise ValueError("preservation manifest digest differs")
    rows = payload.decode("utf-8").splitlines()
    if len(rows) != freeze["expected_lock_count"]:
        raise ValueError("preservation manifest count differs")
    mismatches = []
    for row in rows:
        expected, relative_text = row.split("  ", 1)
        relative = PurePosixPath(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("preservation path is not repository-relative")
        path = repository_root.joinpath(*relative.parts)
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed != expected:
            mismatches.append(relative.as_posix())
    if mismatches:
        raise ValueError("prior artifacts differ from the preservation freeze")
    return {
        "checked_count": len(rows),
        "hash_only_values_decoded": False,
        "manifest_sha256": observed_manifest_sha,
        "mismatches": [],
        "prior_artifacts_mutated": False,
        "status": "PASS",
    }


def build_clean_result(
    clean_expectations: Mapping[str, Any],
    retry_expectations: Mapping[str, Any],
    final_retry_expectations: Mapping[str, Any],
    registry_retry_expectations: Mapping[str, Any],
    inputs: Mapping[str, Any],
    access_events: Sequence[Mapping[str, Any]],
    preflight_events: Sequence[Mapping[str, Any]],
    sparse_preflight_events: Sequence[Mapping[str, Any]],
    registry_preflight_events: Sequence[Mapping[str, Any]],
    *,
    repository_root: Path,
    expectations_commit: str,
    final_retry_expectations_commit: str,
    registry_retry_expectations_commit: str,
    retry_expectations_commit: str,
    runner_commit: str,
    base_commit: str,
) -> dict[str, Any]:
    """Build the clean candidate evidence from in-memory selected fields."""

    validate_clean_expectations(clean_expectations)
    validate_retry_expectations(retry_expectations)
    validate_final_retry_expectations(final_retry_expectations)
    validate_registry_retry_expectations(registry_retry_expectations)
    if retry_expectations["arithmetic_expectations_commit"] != expectations_commit:
        raise ValueError("retry does not bind the original arithmetic freeze commit")
    if final_retry_expectations["arithmetic_expectations_commit"] != expectations_commit:
        raise ValueError("final retry does not bind the arithmetic freeze commit")
    if registry_retry_expectations[
        "arithmetic_expectations_commit"
    ] != expectations_commit:
        raise ValueError("registry retry does not bind the arithmetic freeze commit")
    access = validate_access_events(access_events)
    preflight = validate_preflight_events(preflight_events)
    sparse_preflight = validate_preflight_events(sparse_preflight_events)
    registry_preflight = validate_registry_preflight_events(
        registry_preflight_events
    )
    csv_access = access["completed_accesses"][2]
    final_access = final_retry_expectations["final_access"]
    if csv_access.get("access_pattern") != final_access["access_pattern"]:
        raise ValueError("final CSV access pattern differs from its freeze")
    if csv_access.get("bytes_accessed") != final_access[
        "expected_csv_bytes_accessed"
    ]:
        raise ValueError("final CSV physical byte count differs from its freeze")
    if csv_access.get("unique_bytes_accessed") != final_access[
        "expected_csv_bytes_accessed"
    ]:
        raise ValueError("final CSV unique byte count differs from its freeze")
    if csv_access.get("record_size_bytes") != final_access[
        "expected_csv_record_size_bytes"
    ]:
        raise ValueError("final CSV record size differs from its freeze")
    arithmetic = arithmetic_expectations(
        clean_expectations,
        _require_mapping("calibration_context", inputs["calibration_context"]),
    )
    extracted = {
        "component_basis": inputs["component_basis"],
        "kernel_rows": inputs["kernel_rows"],
    }
    derivation = derive_residency_step(arithmetic, extracted)
    calibration = compare_standard_calibration(arithmetic, derivation)
    independent = independent_recompute(inputs)
    cross_check_recomputation(derivation, calibration, independent)
    classification = calibration["residency_corrected"]["classification"]
    if classification != EXPECTED_CLASSIFICATION:
        raise ValueError("clean result does not reproduce the frozen undercorrection")
    if calibration["movement"]["direction"] != "increase":
        raise ValueError("clean prediction direction differs from the freeze")
    preservation = verify_preservation(clean_expectations, repository_root)
    return {
        "access": {
            **access,
            "cumulative_access_count": (
                preflight["access_count"]
                + sparse_preflight["access_count"]
                + registry_preflight["access_count"]
                + access["access_count"]
            ),
            "cumulative_access_event_count": (
                preflight["access_event_count"]
                + sparse_preflight["access_event_count"]
                + registry_preflight["access_event_count"]
                + access["access_event_count"]
            ),
            "preflight_attempts": [
                preflight,
                sparse_preflight,
                registry_preflight,
            ],
            "successful_tranche": access["completed_accesses"],
        },
        "base_commit": base_commit,
        "calibration_only": calibration,
        "expectations_commit": expectations_commit,
        "final_retry_expectations_commit": final_retry_expectations_commit,
        "independent_recomputation": independent,
        "preservation_lock": preservation,
        "registry_retry_expectations_commit": registry_retry_expectations_commit,
        "registry_source_entries": {
            "core63_verbatim": inputs["core63_entry"],
            "core64_verbatim": inputs["core64_entry"],
        },
        "residency_derivation": derivation,
        "runner_commit": runner_commit,
        "retry_expectations_commit": retry_expectations_commit,
        "schema": CLEAN_RESULT_SCHEMA,
        "scope": {
            "calibration_only": True,
            "held_out_mtp_used_in_arithmetic_or_compared": False,
            "model_weights_downloaded": False,
            "parameters_amended_or_refit": False,
            "scored_run_performed": False,
            "void_post_derivation_numbers_used_as_inputs": False,
            "web_pages_fetched": False,
            "zero_free_or_fitted_constants": True,
        },
        "status": "PASS_CLEAN_REPRODUCTION_UNDERCORRECTION",
        "task": "CORE-63",
        "void_protocol_findings": {
            "ambient_record_inspection_repeated": False,
            "contemporaneous_reader_logging": True,
            "protocol_section_accessed_without_numerical_sections": bool(
                inputs["void_protocol_section"]
            ),
            "whole_file_csv_stream_repeated": False,
        },
    }


def render_markdown(result: Mapping[str, Any]) -> str:
    """Render the clean candidate evidence without consulting source files."""

    derivation = result["residency_derivation"]
    families = derivation["family_decomposition"]
    step = derivation["step"]["residency_corrected_ps"]
    calibration = result["calibration_only"]
    corrected = calibration["residency_corrected"]
    movement = calibration["movement"]
    access = result["access"]
    preservation = result["preservation_lock"]
    return f"""# CORE-63 clean expert-residency repetition

Status: **{result['status']}**. The clean repetition independently reproduces
an honest **{corrected['classification']}** calibration-only result.

## Empty forbidden-access ledger

The forbidden-access ledger is exactly `[]`. No held-out MTP numeric value was
read, copied, compared, or scored. The fifth scored run retains sole ownership
of scoring.

## Reproduced residency step and signed movement

The frozen assignment arithmetic is `256 * 8 * 4 / 288 = 256/9`, relative to
the 256-assignment capture, so routed expert work alone is scaled by `1/9`.
The fixed service is kept once and every other noncollective family stays at
scale one.

```text
T63 = F + 61/4 * (retained_4 + routed_4/9)
    = {step['decimal_ps']} ps
```

The round-half-up corrected step is **{step['published_ps_round_half_up']:,}
ps**. The standard-decode prediction moves by
**+{movement['prediction_tokens_per_second_per_node']} tokens/s/node** to
**{corrected['prediction_tokens_per_second_per_node']} tokens/s/node**. The
signed residual moves by **+{movement['signed_residual_percentage_points']}
percentage points** to **{corrected['signed_residual_percent']} percent**.
The result remains an undercorrection, so the entry's literal acceptance bar
governs how far CORE-63 may move.

## Component classification

The clean reader selected {families['kernel_row_count']} standard-decode
noncollective rows. Exactly {families['routed_kernel_row_count']} row matched
the frozen case-insensitive marker `fused_moe_kernel` and was classified as
routed expert work. Every attention, MLA, router/top-k, shared-expert, dense
MLP, normalization, elementwise, and other nonmatching row stayed retained at
scale one. The committed JSON companion carries the complete classification
ledger and the independently recomputed exact fractions.

## Access and preservation

All {access['access_count']} allowlisted field accesses have contemporaneous
`BEGIN` and `END` events. Every completed byte count is strictly below the
source size, and the final CSV selector left the terminal record byte unread.
The earlier forward and header-plus-reverse preflights were both safely
rejected at 13,984 of 13,985 bytes before full coverage. A third preflight
passed the terminal-byte CSV selector and then rejected an over-specific
registry spelling before EOF. Across all tranches there were
{access['cumulative_access_count']} logged accesses and
{access['cumulative_access_event_count']} contemporaneous events. Whole-file
semantic streams: **{access['whole_file_streams']}**.

All {preservation['checked_count']} inherited preservation artifacts are
byte-identical. Hash verification decoded no artifact values. The void
result's post-derivation numbers were not used as inputs, and no parameter was
amended or refit.

## Registry disposition

The exact pre-run CORE-63 and CORE-64 entries are retained in the JSON result
for literal acceptance review. This clean reproduction stands as candidate
evidence. CORE-64's attention-and-MLA-family gap may be registered
unconditionally because the clean repetition preserved all nonmatching
families and still produced the standard-decode undercorrection.
"""


def write_new_json(path: Path, value: object) -> None:
    """Write one new JSON artifact with pinned POSIX newlines."""

    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_new_text(path: Path, value: str) -> None:
    """Write one new text artifact with pinned POSIX newlines."""

    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(value)


__all__ = [
    "arithmetic_expectations",
    "build_clean_result",
    "cross_check_recomputation",
    "independent_recompute",
    "render_markdown",
    "validate_access_events",
    "validate_clean_expectations",
    "validate_final_retry_expectations",
    "validate_preflight_events",
    "validate_registry_preflight_events",
    "validate_registry_retry_expectations",
    "validate_retry_expectations",
    "verify_preservation",
    "write_new_json",
    "write_new_text",
]
