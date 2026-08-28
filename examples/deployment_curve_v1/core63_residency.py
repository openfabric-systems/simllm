"""Exact CORE-63 expert-residency derivation and calibration comparison."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from decimal import ROUND_HALF_UP, Decimal, localcontext
from fractions import Fraction
from pathlib import Path
from typing import Any

from core63_independent_sign import residency_sign

EXPECTATIONS_SCHEMA = "simllm-deployment-curve-core63-expectations-v1"
RESULT_SCHEMA = "simllm-deployment-curve-core63-calibration-v1"
EXPECTED_IMPLEMENTATION_ID = "deepseek-v3-reduced4-vllm-ep72-decode-b32-c2000"
PS_PER_SECOND = 1_000_000_000_000
PS_PER_NANOSECOND = 1_000
SOURCE_LAYERS = 4
TARGET_LAYERS = 61
ROUTED_MARKER = "fused_moe_kernel"


def _require_int(name: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _require_mapping(name: str, value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return value


def _ceil_fraction(numerator: int, denominator: int) -> int:
    if numerator < 0 or denominator <= 0:
        raise ValueError("ceil fraction requires nonnegative service")
    return -(-numerator // denominator)


def _round_half_up(value: Fraction) -> int:
    if value < 0:
        raise ValueError("service must be nonnegative")
    return (2 * value.numerator + value.denominator) // (2 * value.denominator)


def _decimal(value: Fraction, places: int) -> str:
    with localcontext() as context:
        context.prec = 50
        converted = Decimal(value.numerator) / Decimal(value.denominator)
        quantum = Decimal(1).scaleb(-places)
        return str(converted.quantize(quantum, rounding=ROUND_HALF_UP))


def _fraction_json(value: Fraction) -> dict[str, Any]:
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
    }


def _service_json(value: Fraction) -> dict[str, Any]:
    return {
        **_fraction_json(value),
        "decimal_ps": _decimal(value, 6),
        "decimal_ms": _decimal(value / 1_000_000_000, 9),
        "published_ps_round_half_up": _round_half_up(value),
    }


def _validate_expectations(expectations: Mapping[str, Any]) -> Fraction:
    if expectations.get("schema") != EXPECTATIONS_SCHEMA:
        raise ValueError("CORE-63 expectations schema disagrees")
    if expectations.get("task") != "CORE-63":
        raise ValueError("CORE-63 task identity disagrees")
    if expectations.get("status") != "EXPECTATIONS_ONLY":
        raise ValueError("CORE-63 expectations must remain expectations only")
    component = _require_mapping("component_rule", expectations["component_rule"])
    if not component.get("zero_free_or_fitted_constants"):
        raise ValueError("CORE-63 requires zero free or fitted constants")
    if component.get("routed_expert_kernel_name_markers") != [ROUTED_MARKER]:
        raise ValueError("CORE-63 routed classifier differs from its freeze")
    architecture = _require_mapping(
        "architecture_arithmetic", expectations["architecture_arithmetic"]
    )
    batch = _require_int("batch_per_node", architecture["disclosed_batch_per_node"])
    top_k = _require_int("top_k", architecture["top_k"], minimum=1)
    residents = _require_int(
        "resident slots", architecture["resident_physical_slots_per_rank"], minimum=1
    )
    slots = _require_int(
        "physical expert slots", architecture["physical_expert_slots"], minimum=1
    )
    capture_batch = _require_int(
        "capture batch", architecture["capture_batch_size"], minimum=1
    )
    expected_assignments = Fraction(batch * top_k * residents, slots)
    captured_assignments = Fraction(capture_batch * top_k)
    scale = expected_assignments / captured_assignments
    frozen_scale = _require_mapping("assignment_scale", architecture["assignment_scale"])
    if scale != Fraction(
        _require_int("assignment numerator", frozen_scale["numerator"]),
        _require_int("assignment denominator", frozen_scale["denominator"], minimum=1),
    ):
        raise ValueError("CORE-63 assignment scale differs from architecture arithmetic")
    if expected_assignments != Fraction(256, 9) or scale != Fraction(1, 9):
        raise ValueError("CORE-63 exact residency arithmetic disagrees")
    if expectations["expected_signed_direction"] != {
        "corrected_step": "decrease",
        "prediction": "increase",
        "signed_residual": "less_negative_before_any_possible_crossing",
    }:
        raise ValueError("CORE-63 signed direction differs from its freeze")
    return scale


def _validate_component_basis(basis: Mapping[str, Any]) -> dict[str, Any]:
    if basis.get("implementation_id") != EXPECTED_IMPLEMENTATION_ID:
        raise ValueError("CORE-63 implementation identity differs")
    if basis.get("coverage") != "complete-kernel-stream":
        raise ValueError("CORE-63 requires complete kernel coverage")
    evidence = _require_mapping("evidence", basis.get("evidence"))
    if evidence.get("service_class") != "MEASURED":
        raise ValueError("CORE-63 source service must be measured")
    if evidence.get("component_class") != "DISCLOSED":
        raise ValueError("CORE-63 source components must be disclosed")
    key = _require_mapping("key", basis.get("key"))
    if key.get("pool") != "decode" or key.get("launch_mode") != "cuda-graph":
        raise ValueError("CORE-63 source key differs")
    parallelism = _require_mapping("parallelism", key.get("parallelism"))
    if dict(parallelism) != {
        "data_parallel": 1,
        "expert_parallel": 1,
        "pipeline_parallel": 1,
        "tensor_parallel": 1,
    }:
        raise ValueError("CORE-63 source must be the TP1 full-model shard")
    shape = _require_mapping("shape", key.get("shape"))
    if shape.get("batch_size") != 32:
        raise ValueError("CORE-63 capture batch differs")
    if shape.get("per_request_kv_lengths") != [2000] * 32:
        raise ValueError("CORE-63 capture KV shape differs")

    kernels = basis.get("kernels")
    if not isinstance(kernels, list) or len(kernels) != 1:
        raise ValueError("CORE-63 requires one aggregate component kernel")
    kernel = _require_mapping("kernel", kernels[0])
    if kernel.get("kernel_id") != "aggregate_noncollective_step_service":
        raise ValueError("CORE-63 aggregate kernel identity differs")
    if _require_int("launch_count", kernel.get("launch_count")) != 1:
        raise ValueError("CORE-63 aggregate kernel must launch once")
    components = _require_mapping("components", kernel.get("components"))
    compute_cycles = _require_int("compute_sm_cycles", components.get("compute_sm_cycles"))
    memory = _require_mapping("memory", components.get("memory"))
    memory_ps = _require_int("memory service", memory.get("service_ps"))
    fixed_ps = _require_int("fixed overhead", components.get("fixed_overhead_ps"))
    clocks = _require_mapping("observed clocks", basis.get("observed_clocks"))
    sm_clock = _require_mapping("SM clock", clocks.get("sm_hz"))
    sm_hz = _require_int("median SM Hz", sm_clock.get("median"), minimum=1)
    compute_ps = _ceil_fraction(compute_cycles * PS_PER_SECOND, sm_hz)
    reconstructed_ps = max(compute_ps, memory_ps) + fixed_ps
    measured_kernel_ps = _require_int(
        "measured kernel service", kernel.get("measured_elapsed_ps")
    )
    measured_service_ps = _require_int(
        "measured step service", basis.get("measured_service_ps")
    )
    if reconstructed_ps != measured_kernel_ps or measured_kernel_ps != measured_service_ps:
        raise ValueError("CORE-63 component reconstruction disagrees")
    return {
        "compute_service_ps": compute_ps,
        "compute_sm_cycles": compute_cycles,
        "fixed_service_ps": fixed_ps,
        "measured_service_ps": measured_service_ps,
        "memory_service_ps": memory_ps,
        "sm_hz": sm_hz,
    }


def _kernel_decomposition(
    rows: Sequence[Mapping[str, Any]],
    *,
    measured_service_ps: int,
    fixed_service_ps: int,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("CORE-63 kernel rows are empty")
    ledger = []
    raw_total = Fraction()
    routed_total = Fraction()
    routed_count = 0
    for row in rows:
        name = row.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("CORE-63 kernel name is missing")
        order = row.get("first_launch_order")
        if not isinstance(order, str) or not order.isdigit():
            raise ValueError("CORE-63 kernel launch order differs")
        raw_ns = row.get("total_duration_per_step_ns")
        if not isinstance(raw_ns, str):
            raise TypeError("CORE-63 kernel service must be a decimal string")
        try:
            service_ps = Fraction(raw_ns) * PS_PER_NANOSECOND
        except (ValueError, ZeroDivisionError) as exc:
            raise ValueError("CORE-63 kernel service is not an exact decimal") from exc
        if service_ps < 0:
            raise ValueError("CORE-63 kernel service must be nonnegative")
        family = "routed_expert" if ROUTED_MARKER in name.lower() else "retained"
        raw_total += service_ps
        if family == "routed_expert":
            routed_total += service_ps
            routed_count += 1
        ledger.append(
            {
                "family": family,
                "first_launch_order": int(order),
                "name": name,
                "service_ps": _fraction_json(service_ps),
            }
        )
    tolerance = Fraction(1)
    reconstruction_error = raw_total - measured_service_ps
    if abs(reconstruction_error) > tolerance:
        raise ValueError("CORE-63 per-kernel service does not reconstruct the step")
    if routed_total <= 0 or routed_count == 0:
        raise ValueError("CORE-63 frozen routed kernel family is absent")
    retained_repeatable = raw_total - fixed_service_ps - routed_total
    if retained_repeatable < 0:
        raise ValueError("CORE-63 retained repeatable service is negative")
    return {
        "kernel_classification_ledger": ledger,
        "kernel_row_count": len(ledger),
        "raw_kernel_service_ps": raw_total,
        "reconstruction_error_ps": reconstruction_error,
        "retained_repeatable_four_layer_ps": retained_repeatable,
        "routed_four_layer_ps": routed_total,
        "routed_kernel_row_count": routed_count,
    }


def derive_residency_step(
    expectations: Mapping[str, Any],
    extracted_basis: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive the corrected step without accepting any calibration anchor."""

    routed_scale = _validate_expectations(expectations)
    component_basis = _require_mapping(
        "component_basis", extracted_basis.get("component_basis")
    )
    rows = extracted_basis.get("kernel_rows")
    if not isinstance(rows, list):
        raise TypeError("kernel_rows must be a list")
    component = _validate_component_basis(component_basis)
    families = _kernel_decomposition(
        rows,
        measured_service_ps=component["measured_service_ps"],
        fixed_service_ps=component["fixed_service_ps"],
    )
    fixed = Fraction(component["fixed_service_ps"])
    retained = families["retained_repeatable_four_layer_ps"]
    routed = families["routed_four_layer_ps"]
    current_separated = fixed + Fraction(TARGET_LAYERS, SOURCE_LAYERS) * (
        retained + routed
    )
    corrected = fixed + Fraction(TARGET_LAYERS, SOURCE_LAYERS) * (
        retained + routed_scale * routed
    )
    correction = corrected - current_separated
    if correction != Fraction(TARGET_LAYERS, SOURCE_LAYERS) * (
        routed_scale - 1
    ) * routed:
        raise AssertionError("CORE-63 correction identity failed")
    sign = residency_sign(
        retained_service_ps=_round_half_up(retained),
        routed_service_ps=_round_half_up(routed),
        fixed_service_ps=component["fixed_service_ps"],
        routed_scale=routed_scale,
    )
    expected_sign = expectations["expected_signed_direction"]
    if (
        sign["corrected_step"] != expected_sign["corrected_step"]
        or sign["predicted_throughput"] != expected_sign["prediction"]
        or sign["signed_residual"] != expected_sign["signed_residual"]
    ):
        raise ValueError("CORE-63 independent sign check disagrees")
    architecture = expectations["architecture_arithmetic"]
    return {
        "architecture_arithmetic": {
            "captured_routed_assignments": 256,
            "ep72_expected_assignments_per_rank": _fraction_json(Fraction(256, 9)),
            "formula": "256 * 8 * 4 / 288",
            "routed_assignment_scale": _fraction_json(routed_scale),
            "uniform_routing_assumption": architecture["uniform_routing_assumption"],
        },
        "component_reconstruction": {
            **component,
            "raw_kernel_service_ps": _service_json(
                families["raw_kernel_service_ps"]
            ),
            "reconstruction_error_ps": _fraction_json(
                families["reconstruction_error_ps"]
            ),
        },
        "family_decomposition": {
            "fixed_service_ps": _service_json(fixed),
            "retained_repeatable_four_layer_ps": _service_json(retained),
            "routed_four_layer_ps": _service_json(routed),
            "routed_scaled_four_layer_ps": _service_json(routed_scale * routed),
            "kernel_row_count": families["kernel_row_count"],
            "routed_kernel_row_count": families["routed_kernel_row_count"],
            "kernel_classification_ledger": families[
                "kernel_classification_ledger"
            ],
        },
        "step": {
            "formula": "F + 61/4 * (retained_4 + routed_4/9)",
            "current_separated_ps": _service_json(current_separated),
            "residency_corrected_ps": _service_json(corrected),
            "signed_correction_ps": _service_json(abs(correction)),
            "signed_correction_direction": "decrease",
        },
        "independent_signoff": sign,
        "zero_free_or_fitted_constants": True,
    }


def compare_standard_calibration(
    expectations: Mapping[str, Any],
    derivation: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare the already-derived step to the sole visible standard anchor."""

    context = _require_mapping("calibration_context", expectations["calibration_context"])
    tokens = _require_int("per_node_tokens", context["per_node_tokens"], minimum=1)
    anchor = _require_int(
        "published standard anchor",
        context["published_tokens_per_second_per_node"],
        minimum=1,
    )
    current_step = _require_int("current step", context["current_step_ps"], minimum=1)
    corrected_row = _require_mapping(
        "corrected step", derivation["step"]["residency_corrected_ps"]
    )
    corrected_step = Fraction(
        corrected_row["numerator"], corrected_row["denominator"]
    )
    current_prediction = Fraction(tokens * PS_PER_SECOND, current_step)
    corrected_prediction = Fraction(tokens * PS_PER_SECOND, 1) / corrected_step
    movement = corrected_prediction - current_prediction
    current_residual = current_prediction / anchor - 1
    corrected_residual = corrected_prediction / anchor - 1
    residual_movement = corrected_residual - current_residual
    pass_bar = Fraction(5, 100)
    if abs(corrected_residual) <= pass_bar:
        classification = "PASS"
    elif corrected_residual < 0:
        classification = "UNDERCORRECTION"
    else:
        classification = "OVERCORRECTION"
    return {
        "anchor_id": context["anchor_id"],
        "role": "calibration-only",
        "published_tokens_per_second_per_node": anchor,
        "per_node_tokens": tokens,
        "current": {
            "step_ps": current_step,
            "prediction_exact": _fraction_json(current_prediction),
            "prediction_tokens_per_second_per_node": _decimal(
                current_prediction, 6
            ),
            "published_display": context[
                "current_prediction_tokens_per_second_per_node_display"
            ],
            "signed_residual_percent": _decimal(current_residual * 100, 6),
        },
        "residency_corrected": {
            "step_ps": dict(corrected_row),
            "prediction_exact": _fraction_json(corrected_prediction),
            "prediction_tokens_per_second_per_node": _decimal(
                corrected_prediction, 6
            ),
            "signed_residual_percent": _decimal(corrected_residual * 100, 6),
            "classification": classification,
        },
        "movement": {
            "direction": "increase",
            "prediction_tokens_per_second_per_node": _decimal(movement, 6),
            "prediction_relative_percent": _decimal(
                movement / current_prediction * 100, 6
            ),
            "signed_residual_percentage_points": _decimal(
                residual_movement * 100, 6
            ),
        },
        "pass_bar_absolute_percent": "5.000000",
    }


def verify_preservation_lock(
    expectations: Mapping[str, Any], repository_root: Path
) -> dict[str, Any]:
    lock = _require_mapping("preservation_lock", expectations["preservation_lock"])
    manifest_path = repository_root / str(lock["manifest_path"])
    rows = [line.split("  ", 1) for line in manifest_path.read_text().splitlines()]
    if len(rows) != lock["artifact_count"]:
        raise ValueError("CORE-63 preservation manifest count differs")
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if manifest_sha != lock["manifest_sha256"]:
        raise ValueError("CORE-63 preservation manifest identity differs")
    mismatches = []
    for expected, relative in rows:
        observed = hashlib.sha256((repository_root / relative).read_bytes()).hexdigest()
        if observed != expected:
            mismatches.append(relative)
    return {
        "checked_count": len(rows),
        "manifest_sha256": manifest_sha,
        "mismatches": mismatches,
        "prior_records_mutated": bool(mismatches),
        "status": "PASS" if not mismatches else "FAIL",
    }


def build_result(
    expectations: Mapping[str, Any],
    extracted_basis: Mapping[str, Any],
    successful_access_entries: Sequence[Mapping[str, Any]],
    cumulative_access_entries: Sequence[Mapping[str, Any]],
    protocol_incidents: Sequence[Mapping[str, Any]],
    *,
    repository_root: Path,
    expectations_commit: str,
) -> dict[str, Any]:
    """Build the honest calibration-only record, including protocol incidents."""

    derivation = derive_residency_step(expectations, extracted_basis)
    calibration = compare_standard_calibration(expectations, derivation)
    if len(successful_access_entries) != 2:
        raise ValueError("CORE-63 successful access tranche must contain two entries")
    if any(entry.get("status") != "PASS" for entry in successful_access_entries):
        raise ValueError("CORE-63 successful access tranche contains a rejection")
    if any(entry.get("held_out_numeric_value_accessed") for entry in cumulative_access_entries):
        raise ValueError("CORE-63 reader ledger reports a held-out access")
    preservation = verify_preservation_lock(expectations, repository_root)
    if preservation["status"] != "PASS":
        raise ValueError("CORE-63 preservation lock failed")
    violation = any(incident.get("protocol_violation") for incident in protocol_incidents)
    classification = calibration["residency_corrected"]["classification"]
    if not violation:
        raise ValueError("this publication must retain the disclosed protocol incident")
    return {
        "schema": RESULT_SCHEMA,
        "task": "CORE-63",
        "status": f"PROTOCOL_VOID_CALIBRATION_ONLY_{classification}",
        "expectations_commit": expectations_commit,
        "residency_derivation": derivation,
        "calibration_only": calibration,
        "access": {
            "successful_tranche": [dict(entry) for entry in successful_access_entries],
            "successful_tranche_count": 2,
            "cumulative_reader_ledger": [
                dict(entry) for entry in cumulative_access_entries
            ],
            "cumulative_reader_access_count": len(cumulative_access_entries),
            "cumulative_pass_count": sum(
                entry.get("status") == "PASS" for entry in cumulative_access_entries
            ),
            "cumulative_rejected_count": sum(
                entry.get("status") == "REJECTED"
                for entry in cumulative_access_entries
            ),
            "reader_held_out_access_ledger": [],
            "reader_held_out_numeric_values_accessed_or_compared": False,
            "protocol_incident_ledger": [dict(row) for row in protocol_incidents],
            "literal_clean_protocol": False,
        },
        "overlap_ruling": {
            "communication_term_in_current_decode_pricing": False,
            "binding_mechanism_now": False,
            "follow_on": (
                "derive decode-side overlap only after a decode communication "
                "service term exists"
            ),
        },
        "preservation_lock": preservation,
        "scope": {
            "calibration_only": True,
            "held_out_mtp_used_in_arithmetic_or_compared": False,
            "model_weights_downloaded": False,
            "scored_run_performed": False,
            "web_pages_fetched": False,
            "zero_free_or_fitted_constants": True,
        },
        "registry": {
            "core63": "OPEN_PROTOCOL_VOID_REQUIRES_CLEAN_REPETITION",
            "core64": (
                "OPEN_ON_EXACT_STANDARD_DECODE_UNDERCORRECTION_RESIDUAL"
                if classification == "UNDERCORRECTION"
                else f"OPEN_ON_EXACT_STANDARD_DECODE_{classification}_RESIDUAL"
            ),
        },
    }


__all__ = [
    "build_result",
    "compare_standard_calibration",
    "derive_residency_step",
    "verify_preservation_lock",
]
