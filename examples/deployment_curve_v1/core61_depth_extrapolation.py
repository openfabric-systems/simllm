"""Separated fixed-plus-per-layer extrapolation for the CORE-61 local arm."""

from __future__ import annotations

from decimal import Decimal, localcontext
from fractions import Fraction
from typing import Any

PS_PER_SECOND = 1_000_000_000_000
PS_PER_MILLISECOND = 1_000_000_000
SOURCE_LAYERS = 4
TARGET_LAYERS = 61
HELD_OUT_LAYERS = 8
EXPECTED_IMPLEMENTATION_ID = "deepseek-v3-reduced4-vllm-ep72-decode-b32-c2000"


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    return value


def _ceil_fraction(numerator: int, denominator: int) -> int:
    if numerator < 0 or denominator <= 0:
        raise ValueError("ceil fraction requires a nonnegative numerator and positive denominator")
    return -(-numerator // denominator)


def _round_half_up(value: Fraction) -> int:
    if value < 0:
        raise ValueError("CORE-61 service values must be nonnegative")
    return (2 * value.numerator + value.denominator) // (2 * value.denominator)


def _decimal(value: Fraction, places: int) -> str:
    with localcontext() as context:
        context.prec = 40
        decimal = Decimal(value.numerator) / Decimal(value.denominator)
        return f"{decimal:.{places}f}"


def _exact(value: Fraction) -> dict[str, Any]:
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "decimal_ps": _decimal(value, 6),
        "decimal_ms": _decimal(value / PS_PER_MILLISECOND, 9),
    }


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object")
    return value


def _validate_basis(basis: dict[str, Any]) -> dict[str, Any]:
    if basis.get("implementation_id") != EXPECTED_IMPLEMENTATION_ID:
        raise ValueError("CORE-61 selected implementation identity differs")
    if basis.get("coverage") != "complete-kernel-stream":
        raise ValueError("CORE-61 requires complete-kernel-stream coverage")

    evidence = _require_mapping(basis.get("evidence"), "evidence")
    if evidence.get("service_class") != "MEASURED":
        raise ValueError("CORE-61 source service must be MEASURED")
    if evidence.get("component_class") != "DISCLOSED":
        raise ValueError("CORE-61 source components must retain DISCLOSED class")

    key = _require_mapping(basis.get("key"), "key")
    if key.get("pool") != "decode" or key.get("launch_mode") != "cuda-graph":
        raise ValueError("CORE-61 source must be the CUDA-graph decode key")
    parallelism = _require_mapping(key.get("parallelism"), "parallelism")
    expected_parallelism = {
        "tensor_parallel": 1,
        "pipeline_parallel": 1,
        "data_parallel": 1,
        "expert_parallel": 1,
    }
    if parallelism != expected_parallelism:
        raise ValueError("CORE-61 reduced physical key parallelism differs")
    shape = _require_mapping(key.get("shape"), "shape")
    if shape.get("batch_size") != 32:
        raise ValueError("CORE-61 source batch size differs")
    kv_lengths = shape.get("per_request_kv_lengths")
    if not isinstance(kv_lengths, list) or kv_lengths != [2000] * 32:
        raise ValueError("CORE-61 source remote-KV-2000 vector differs")

    clocks = _require_mapping(basis.get("observed_clocks"), "observed_clocks")
    sm_clock = _require_mapping(clocks.get("sm_hz"), "observed_clocks.sm_hz")
    sm_hz = _integer(sm_clock.get("median"), "median SM clock")
    if sm_hz <= 0:
        raise ValueError("median SM clock must be positive")

    kernels = basis.get("kernels")
    if not isinstance(kernels, list) or len(kernels) != 1:
        raise ValueError("CORE-61 requires exactly one frozen aggregate kernel")
    kernel = _require_mapping(kernels[0], "kernel")
    if kernel.get("kernel_id") != "aggregate_noncollective_step_service":
        raise ValueError("CORE-61 kernel is not the whole-step aggregate")
    launch_count = _integer(kernel.get("launch_count"), "launch_count")
    if launch_count != 1:
        raise ValueError("CORE-61 whole-step aggregate must have one launch")

    components = _require_mapping(kernel.get("components"), "components")
    method = components.get("method")
    if not isinstance(method, str) or "Retained Nsys additive noncollective service" not in method:
        raise ValueError("CORE-61 aggregate method does not identify retained step service")
    compute_cycles = _integer(components.get("compute_sm_cycles"), "compute_sm_cycles")
    memory = _require_mapping(components.get("memory"), "components.memory")
    memory_ps = _integer(memory.get("service_ps"), "memory service")
    fixed_ps = _integer(components.get("fixed_overhead_ps"), "fixed overhead")
    if min(compute_cycles, memory_ps, fixed_ps) < 0:
        raise ValueError("CORE-61 components must be nonnegative")

    compute_ps = _ceil_fraction(compute_cycles * PS_PER_SECOND, sm_hz)
    repeatable_ps = max(compute_ps, memory_ps) * launch_count
    fixed_step_ps = fixed_ps * launch_count
    reconstructed_kernel_ps = max(compute_ps, memory_ps) + fixed_ps
    measured_kernel_ps = _integer(kernel.get("measured_elapsed_ps"), "kernel elapsed service")
    if reconstructed_kernel_ps != measured_kernel_ps:
        raise ValueError("CORE-61 kernel components do not reconstruct elapsed service")
    measured_service_ps = _integer(basis.get("measured_service_ps"), "measured service")
    if measured_kernel_ps * launch_count != measured_service_ps:
        raise ValueError("CORE-61 kernel stream does not reconstruct measured service")
    if fixed_step_ps + repeatable_ps != measured_service_ps:
        raise ValueError("CORE-61 fixed-plus-repeatable decomposition does not conserve T(4)")

    return {
        "candidate_key": key,
        "component_method": method,
        "compute_sm_cycles": compute_cycles,
        "compute_service_ps": compute_ps,
        "fixed_step_ps": fixed_step_ps,
        "launch_count": launch_count,
        "measured_service_ps": measured_service_ps,
        "memory_service_ps": memory_ps,
        "repeatable_four_layer_ps": repeatable_ps,
        "sm_hz": sm_hz,
    }


def derive_result(
    expectations: dict[str, Any],
    basis: dict[str, Any],
    access_entry: dict[str, Any],
    *,
    expectations_commit: str,
) -> dict[str, Any]:
    """Validate the frozen basis and emit exact separated extrapolations."""

    if expectations.get("status") != "EXPECTATIONS_ONLY":
        raise ValueError("CORE-61 expectations must retain EXPECTATIONS_ONLY status")
    if access_entry.get("status") != "PASS":
        raise ValueError("CORE-61 retained access must pass")
    if access_entry.get("whole_record_loaded") is not False:
        raise ValueError("CORE-61 refuses a whole-record access")
    if access_entry.get("unselected_values_decoded") is not False:
        raise ValueError("CORE-61 refuses decoded unselected values")

    component = _validate_basis(basis)
    t4 = Fraction(component["measured_service_ps"])
    fixed = Fraction(component["fixed_step_ps"])
    repeatable = Fraction(component["repeatable_four_layer_ps"])
    per_layer = repeatable / SOURCE_LAYERS
    linear_61 = t4 * TARGET_LAYERS / SOURCE_LAYERS
    separated_61 = fixed + TARGET_LAYERS * per_layer
    separated_8 = fixed + HELD_OUT_LAYERS * per_layer
    correction = separated_61 - linear_61
    expected_correction = -Fraction(57, 4) * fixed
    if correction != expected_correction:
        raise AssertionError("CORE-61 signed correction identity failed")

    fixed_share_ppm = fixed * 1_000_000 / t4
    correction_share_ppm = abs(correction) * 1_000_000 / linear_61
    corrected_ps = _round_half_up(separated_61)
    linear_ps = _round_half_up(linear_61)
    held_out_ps = _round_half_up(separated_8)
    materiality = (
        "NULL_ZERO_FIXED_COMPONENT"
        if fixed == 0
        else "NONZERO_BUT_REPORT_EXACT_MAGNITUDE_WITHOUT_TUNED_THRESHOLD"
    )

    return {
        "schema": "simllm-deployment-curve-core61-depth-result-v1",
        "task": "CORE-61",
        "status": "LOCAL_DERIVATION_COMPLETE_CORE61_OPEN",
        "expectations_commit": expectations_commit,
        "evidence_class": {
            "service_class": "DECLARED",
            "description": "DECLARED derivation from a MEASURED decomposition at one depth",
            "source_service_class": "MEASURED",
            "source_component_class": "DISCLOSED",
            "measured_depth_count": 1,
            "validated_depth_rule": False,
        },
        "source": {
            "implementation_id": EXPECTED_IMPLEMENTATION_ID,
            "candidate_key": component["candidate_key"],
            "selector": access_entry["selector"],
            "published_record_sha256": access_entry[
                "record_sha256_from_published_manifest"
            ],
            "access": access_entry,
        },
        "decomposition": {
            "four_layer_measured_service_ps": component["measured_service_ps"],
            "per_step_fixed_ps": component["fixed_step_ps"],
            "four_layer_repeatable_ps": component["repeatable_four_layer_ps"],
            "per_layer_repeatable": _exact(per_layer),
            "fixed_share_of_t4_ppm": _decimal(fixed_share_ppm, 6),
            "materiality_verdict": materiality,
            "kernel": {
                "kernel_id": "aggregate_noncollective_step_service",
                "launch_count": component["launch_count"],
                "median_sm_hz": component["sm_hz"],
                "compute_sm_cycles": component["compute_sm_cycles"],
                "compute_service_ps": component["compute_service_ps"],
                "memory_service_ps": component["memory_service_ps"],
                "fixed_overhead_ps": component["fixed_step_ps"],
                "method": component["component_method"],
            },
            "reconstruction_error_ps": 0,
        },
        "declared_61_layer_step": {
            "linear_rule": {
                "exact": _exact(linear_61),
                "published_ps": linear_ps,
            },
            "separated_rule": {
                "exact": _exact(separated_61),
                "published_ps": corrected_ps,
            },
            "signed_movement_separated_minus_linear": {
                "direction": "decrease" if correction < 0 else "unchanged",
                "exact": _exact(correction),
                "published_ps": corrected_ps - linear_ps,
                "absolute_share_of_linear_ppm": _decimal(correction_share_ppm, 6),
            },
        },
        "held_out_depth_prediction": {
            "depth_layers": HELD_OUT_LAYERS,
            "shape": {"batch_size": 32, "remote_kv_tokens_per_request": 2000},
            "service": _exact(separated_8),
            "published_ps": held_out_ps,
            "evidence_class": "DECLARED",
            "measured_service_ps": None,
            "signed_residual_percent": None,
            "status": "BLOCKED_ON_MERLIN_MAINTENANCE",
        },
        "comparison_context": {
            "published_tokens_per_second": expectations["comparison_context"][
                "published_tokens_per_second"
            ],
            "used_as_arithmetic_input": False,
            "used_for_selection_or_classification": False,
            "implied_step_displayed": False,
        },
        "signed_residual_ledger": [
            {
                "term": "depth scaling correction",
                "owner": "CORE-61",
                "signed_ps": corrected_ps - linear_ps,
                "direction": "decrease" if correction < 0 else "unchanged",
                "state": "DECLARED_SINGLE_DEPTH",
            },
            {
                "term": "finite compute and communication overlap",
                "owner": "TRAF-66",
                "signed_ps": None,
                "direction": "unchanged_not_recomputed",
                "state": "OUT_OF_SCOPE",
            },
            {
                "term": "held-out depth prediction residual",
                "owner": "CORE-61",
                "signed_ps": None,
                "direction": "unknown_until_second_measurement",
                "state": "BLOCKED_ON_COMP-72_MERLIN_REMAINDER",
            },
        ],
        "registry": {
            "core61": "OPEN",
            "local_movement": "separated extrapolation derived from one retained measured depth",
            "merlin_remainder": "measure the frozen eight-layer batch-32 remote-KV-2000 cell and score the held-out residual within 5 percent",
            "comp72": "OPEN_WITH_CORE61_DEPTH8_COMPANION_COMMANDS",
            "comp76": "UNCHANGED",
            "reserved_residual_id": "CORE-63",
        },
    }
