"""Event-derived finite two-batch overlap boundary service."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any

EXPECTATIONS_SCHEMA = "simllm-deployment-curve-traf66-expectations-v1"
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
