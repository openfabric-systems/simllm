"""Event-derived finite two-batch overlap boundary service."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any

EXPECTATIONS_SCHEMA = "simllm-deployment-curve-traf66-expectations-v1"
RESULT_SCHEMA = "simllm-deployment-curve-traf66-calibration-v1"
PS_PER_SECOND = 1_000_000_000_000


def _require_int(name: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _interval(row: Mapping[str, Any], name: str) -> dict[str, int]:
    value = row[name]
    if set(value) != {"lower", "selected", "upper"}:
        raise ValueError(f"{name} must contain lower, selected and upper")
    result = {
        edge: _require_int(f"{name}.{edge}", value[edge])
        for edge in ("lower", "selected", "upper")
    }
    if not result["lower"] <= result["selected"] <= result["upper"]:
        raise ValueError(f"{name} interval is not ordered")
    return result


def finite_two_batch_service(
    compute_service_ps: int,
    packet_service_ps: int,
) -> dict[str, Fraction]:
    """Return the exact two-child prologue, steady and epilogue service.

    Each child conserves one half of the aggregate compute service and one
    half of the aggregate packet service. One child's second resource overlaps
    the other child's first resource in the steady window. The remaining
    half-service is exposed at the finite prologue and epilogue boundary.
    """

    compute = Fraction(_require_int("compute_service_ps", compute_service_ps, minimum=1))
    packet = Fraction(_require_int("packet_service_ps", packet_service_ps, minimum=1))
    child_compute = compute / 2
    child_packet = packet / 2
    steady = max(child_compute, child_packet)
    total = child_compute + steady + child_packet
    steady_state_total = max(compute, packet)
    return {
        "child_compute_service_ps": child_compute,
        "child_packet_service_ps": child_packet,
        "prologue_service_ps": child_compute,
        "steady_interleave_service_ps": steady,
        "epilogue_service_ps": child_packet,
        "steady_state_total_service_ps": steady_state_total,
        "boundary_service_ps": total - steady_state_total,
        "total_service_ps": total,
    }


def conserved_event_counts(moe_layers: int) -> dict[str, int]:
    """Return exact child-stage and async communication event counts."""

    layers = _require_int("moe_layers", moe_layers, minimum=1)
    children = 2
    yields_per_child = 2 * layers
    stages_per_child = yields_per_child + 1
    return {
        "children": children,
        "moe_layers": layers,
        "dispatch_launches_per_child": layers,
        "dispatch_completions_per_child": layers,
        "combine_launches_per_child": layers,
        "combine_completions_per_child": layers,
        "yield_boundaries_per_child": yields_per_child,
        "stages_per_child": stages_per_child,
        "dispatch_launches_total": children * layers,
        "dispatch_completions_total": children * layers,
        "combine_launches_total": children * layers,
        "combine_completions_total": children * layers,
        "yield_boundaries_total": children * yields_per_child,
        "stage_advances_total": children * stages_per_child,
    }


def validate_expectations(expectations: Mapping[str, Any]) -> None:
    """Validate the expectations-only TRAF-66 protocol-void freeze."""

    if expectations.get("schema") != EXPECTATIONS_SCHEMA:
        raise ValueError("TRAF-66 expectations schema disagrees")
    if expectations.get("task") != "TRAF-66":
        raise ValueError("TRAF-66 task identity disagrees")
    if expectations.get("status") != "EXPECTATIONS_ONLY_PROTOCOL_VOID":
        raise ValueError("TRAF-66 must record its held-out access protocol void")

    chronology = expectations["chronology"]
    if chronology != {
        "allowlist_extension_commit": "dcf6be1cf82b23a989d45b1d5ff50d15aa71ac5f",
        "allowlist_extension_preceded_source_inspection": True,
        "boundary_form_derived_before_visible_comparison": True,
        "framework_evaluation_tables_accessed": False,
        "held_out_component_record_accessed": True,
        "scored_comparison_performed": False,
        "scored_flagship_rerun_performed": False,
        "visible_calibration_comparison_performed": False,
        "web_pages_fetched": False,
    }:
        raise ValueError("TRAF-66 chronology disagrees")

    split = expectations["calibration_split"]
    if split["visible_anchor_ids"] != ["sglang_prefill_1k"]:
        raise ValueError("TRAF-66 visible split disagrees")
    if split["forbidden_anchor_ids"] != ["sglang_prefill_2k", "sglang_prefill_4k"]:
        raise ValueError("TRAF-66 forbidden split disagrees")
    ledger = split["held_out_access_ledger"]
    if len(ledger) != 1 or ledger[0]["disposition"] != "not_used_or_compared":
        raise ValueError("TRAF-66 must retain the accidental held-out access")

    parameters = expectations["parameters"]
    if parameters["free"] or parameters["fitted"]:
        raise ValueError("TRAF-66 cannot contain free or fitted parameters")
    if parameters["child_fraction"] != {"numerator": 1, "denominator": 2}:
        raise ValueError("TRAF-66 child fraction must follow two-child conservation")

    events = expectations["event_conservation"]
    expected_counts = conserved_event_counts(events["moe_layers"])
    if events["counts"] != expected_counts:
        raise ValueError("TRAF-66 event conservation disagrees")
    if events["tbo_delta_stages"] != 0:
        raise ValueError("TRAF-66 pinned prefill stage offset must be zero")
    if events["executor_prologue_stage_advances"] != 0:
        raise ValueError("TRAF-66 executor prologue count disagrees")
    if events["executor_epilogue_stage_advances"] != 0:
        raise ValueError("TRAF-66 executor epilogue count disagrees")
    if events["central_interleave_iterations"] != expected_counts["stages_per_child"]:
        raise ValueError("TRAF-66 central interleave count disagrees")

    composition = expectations["composition"]
    if composition["operator"] != "max(C, P) + min(C, P) / 2":
        raise ValueError("TRAF-66 boundary operator disagrees")
    compute = _require_int(
        "candidate_compute_service_ps",
        composition["candidate_compute_service_ps"],
        minimum=1,
    )
    packet = _interval(composition, "packet_service_ps")
    total = _interval(composition, "total_service_ps")
    boundary = _interval(composition, "boundary_service_ps")
    for edge in ("lower", "selected", "upper"):
        derived = finite_two_batch_service(compute, packet[edge])
        if derived["total_service_ps"].denominator != 1:
            raise ValueError("TRAF-66 frozen total service must be integral ps")
        if derived["boundary_service_ps"].denominator != 1:
            raise ValueError("TRAF-66 frozen boundary service must be integral ps")
        if total[edge] != derived["total_service_ps"].numerator:
            raise ValueError(f"TRAF-66 {edge} total service disagrees")
        if boundary[edge] != derived["boundary_service_ps"].numerator:
            raise ValueError(f"TRAF-66 {edge} boundary service disagrees")
    if composition["selected_regime"] != "packet_dominant":
        raise ValueError("TRAF-66 selected regime disagrees")
    if not all(packet[edge] > compute for edge in packet):
        raise ValueError("TRAF-66 packet-dominant envelope does not hold")

    signed = expectations["signed_movement_expectation"]
    if signed != {
        "predicted_throughput_relative_to_comp75": "decrease",
        "service_relative_to_comp75": "increase",
        "visible_calibration_residual_movement": "more_negative",
    }:
        raise ValueError("TRAF-66 signed movement expectation disagrees")

    structural = expectations["held_out_structural_prediction"]
    if structural["numeric_values_accessed_for_prediction"]:
        raise ValueError("TRAF-66 held-out structural statement cannot use values")
    if structural["service_term"] != "packet_service_ps + compute_service_ps / 2":
        raise ValueError("TRAF-66 held-out structural service term disagrees")


def verify_preservation_locks(
    expectations: Mapping[str, Any], repository_root: Path
) -> dict[str, Any]:
    """Verify every prior record named by the preservation-lock class."""

    validate_expectations(expectations)
    checked = []
    for row in expectations["preservation_locks"]:
        relative = PurePosixPath(str(row["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("preservation paths must be repository-relative")
        path = repository_root.joinpath(*relative.parts)
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed != row["sha256"]:
            raise ValueError(f"preservation digest disagrees for {row['path']}")
        checked.append({"path": row["path"], "sha256": observed})
    return {
        "status": "PASS",
        "checked_artifacts": checked,
        "checked_count": len(checked),
        "prior_records_mutated": False,
    }


def compare_component_inputs(
    expectations: Mapping[str, Any], comp75_expectations: Mapping[str, Any]
) -> dict[str, bool]:
    """Check that TRAF-66 reuses COMP-75 compute and packet services exactly."""

    validate_expectations(expectations)
    composition = expectations["composition"]
    previous = comp75_expectations["composition"]
    selected_arm = previous["selected_arm"]
    arms = {str(row["id"]): row for row in previous["service_arms"]}
    return {
        "candidate_compute_service_reused": (
            composition["candidate_compute_service_ps"]
            == previous["candidate_compute_service_ps"]
        ),
        "packet_service_envelope_reused": (
            composition["packet_service_ps"]
            == arms[selected_arm]["communication_service_ps"]
        ),
        "moe_layer_count_reused": (
            expectations["event_conservation"]["moe_layers"]
            == comp75_expectations["traffic"]["moe_layers"]
        ),
    }


def fraction_json(value: Fraction) -> dict[str, int]:
    """Render one exact fraction without a floating-point conversion."""

    return {"numerator": value.numerator, "denominator": value.denominator}


def _movement(updated: Fraction, reference: Fraction) -> dict[str, Any]:
    delta = updated - reference
    direction = "decrease" if delta < 0 else "increase" if delta > 0 else "unchanged"
    return {
        "absolute_tokens_per_second_per_node": fraction_json(delta),
        "direction": direction,
        "relative_to_reference": fraction_json(delta / reference),
    }


def calibration_comparison(
    expectations: Mapping[str, Any],
    comp75_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare the frozen form with COMP-75's visible-only calibration row."""

    validate_expectations(expectations)
    if comp75_result.get("schema") != "simllm-deployment-curve-comp75-calibration-v1":
        raise ValueError("TRAF-66 requires the clean COMP-75 calibration record")
    if comp75_result.get("held_out_numeric_values_accessed"):
        raise ValueError("TRAF-66 cannot consume a held-out COMP-75 result")
    if comp75_result.get("accessed_visible_anchor_ids") != ["sglang_prefill_1k"]:
        raise ValueError("TRAF-66 COMP-75 visible row disagrees")
    rows = comp75_result.get("calibration_rows")
    if not isinstance(rows, list) or len(rows) != 1:
        raise ValueError("TRAF-66 requires exactly one visible calibration row")
    row = rows[0]
    if row.get("anchor_id") != "sglang_prefill_1k":
        raise ValueError("TRAF-66 visible calibration identity disagrees")

    composition = expectations["composition"]
    previous_service = _require_int(
        "comp75 total_service_ps", row["total_service_ps"], minimum=1
    )
    if previous_service != composition["packet_service_ps"]["selected"]:
        raise ValueError("TRAF-66 COMP-75 service does not match the frozen packet service")
    tokens = _require_int("per_node_tokens", row["per_node_tokens"], minimum=1)
    published = row["published"]
    target = Fraction(
        _require_int("published.numerator", published["numerator"], minimum=1),
        _require_int("published.denominator", published["denominator"], minimum=1),
    )
    previous_prediction = Fraction(tokens * PS_PER_SECOND, previous_service)
    totals = _interval(composition, "total_service_ps")
    predictions = {
        edge: Fraction(tokens * PS_PER_SECOND, totals[edge])
        for edge in ("lower", "selected", "upper")
    }
    updated = predictions["selected"]
    previous_error = previous_prediction / target - 1
    updated_error = updated / target - 1
    target_service = Fraction(tokens * PS_PER_SECOND, target)
    previous_surplus = Fraction(previous_service) - target_service
    updated_surplus = Fraction(totals["selected"]) - target_service
    boundary = Fraction(composition["boundary_service_ps"]["selected"])
    return {
        "anchor_id": "sglang_prefill_1k",
        "per_node_tokens": tokens,
        "published": fraction_json(target),
        "comp75_prediction": fraction_json(previous_prediction),
        "prediction": {
            "lower": fraction_json(min(predictions.values())),
            "point": fraction_json(updated),
            "upper": fraction_json(max(predictions.values())),
        },
        "comp75_total_service_ps": previous_service,
        "boundary_service_ps": boundary.numerator,
        "total_service_ps": totals["selected"],
        "signed_movement_from_comp75": _movement(updated, previous_prediction),
        "signed_relative_error_before": fraction_json(previous_error),
        "signed_relative_error_after": fraction_json(updated_error),
        "signed_residual_movement": fraction_json(updated_error - previous_error),
        "service_surplus_before_ps": fraction_json(previous_surplus),
        "service_surplus_after_ps": fraction_json(updated_surplus),
        "boundary_to_prior_surplus_ratio": fraction_json(boundary / previous_surplus),
    }


def validate_result(
    result: Mapping[str, Any],
    expectations: Mapping[str, Any],
    comp75_result: Mapping[str, Any],
) -> None:
    """Validate the published calibration-only protocol-void result."""

    if result.get("schema") != RESULT_SCHEMA:
        raise ValueError("TRAF-66 result schema disagrees")
    if result.get("status") != "PROTOCOL_VOID_HELD_OUT_COMPONENT_ACCESS":
        raise ValueError("TRAF-66 result must retain its protocol-void status")
    if result.get("task") != "TRAF-66":
        raise ValueError("TRAF-66 result task identity disagrees")
    expected_row = calibration_comparison(expectations, comp75_result)
    if result.get("calibration_rows") != [expected_row]:
        raise ValueError("TRAF-66 calibration row disagrees")
    if result.get("event_conservation") != expectations["event_conservation"]:
        raise ValueError("TRAF-66 published event conservation disagrees")
    if result.get("held_out_access_ledger") != expectations["calibration_split"][
        "held_out_access_ledger"
    ]:
        raise ValueError("TRAF-66 result must retain the access ledger")
    if result.get("held_out_numeric_values_used_or_compared"):
        raise ValueError("TRAF-66 result cannot use or compare a held-out value")
    if result.get("scored_flagship_rerun_performed"):
        raise ValueError("TRAF-66 result cannot rerun the scored flagship")
    if result.get("decode_pricing_changed"):
        raise ValueError("TRAF-66 result cannot change decode pricing")
    if result.get("nvlink_scope_touched"):
        raise ValueError("TRAF-66 result cannot touch TRAF-65 scope")
