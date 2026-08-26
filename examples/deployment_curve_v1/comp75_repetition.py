"""Clean COMP-75 repetition of the CORE-60 physical composition."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from fractions import Fraction
from math import comb
from pathlib import Path, PurePosixPath
from typing import Any

from core59_role_mechanisms import (
    prediction_at_role_shape_service as core59_prediction,
)
from curve_tools import as_fraction, fraction_json

EXPECTATIONS_SCHEMA = "simllm-deployment-curve-comp75-expectations-v1"
CALIBRATION_SCHEMA = "simllm-deployment-curve-comp75-calibration-v1"
PS_PER_SECOND = 1_000_000_000_000
PREFILL_ANCHOR_ID = "sglang_prefill_1k"


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


def _arms(expectations: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = expectations["composition"]["service_arms"]
    result = {str(row["id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("COMP-75 service arm IDs must be unique")
    return result


def _fraction_matches(row: Mapping[str, Any], prefix: str, expected: Fraction) -> bool:
    return expected == Fraction(
        _require_int(f"{prefix}_numerator", row[f"{prefix}_numerator"]),
        _require_int(f"{prefix}_denominator", row[f"{prefix}_denominator"], minimum=1),
    )


def validate_expectations(expectations: Mapping[str, Any]) -> None:
    """Validate the expectations-only, no-fit COMP-75 freeze."""

    if expectations.get("schema") != EXPECTATIONS_SCHEMA:
        raise ValueError("COMP-75 expectations schema disagrees")
    if expectations.get("task") != "COMP-75":
        raise ValueError("COMP-75 task identity disagrees")
    if expectations.get("status") != "EXPECTATIONS_ONLY":
        raise ValueError("COMP-75 expectations status disagrees")

    chronology = expectations["chronology"]
    required_false = (
        "framework_evaluation_tables_accessed",
        "held_out_numeric_values_accessed",
        "scored_comparison_performed",
        "scored_flagship_rerun_performed",
        "visible_calibration_comparison_performed",
        "visible_calibration_numeric_values_accessed",
        "void_core60_external_source_values_accessed",
        "web_pages_fetched",
    )
    if not chronology["allowlist_committed_before_source_inspection"]:
        raise ValueError("source inspection preceded the allowlist")
    if not chronology["mechanism_service_derived_before_comparison"]:
        raise ValueError("mechanism service was not derived before comparison")
    if any(chronology[name] for name in required_false):
        raise ValueError("COMP-75 expectations chronology is exposed")

    split = expectations["calibration_split"]
    if split["visible_anchor_ids"] != [PREFILL_ANCHOR_ID]:
        raise ValueError("COMP-75 visible calibration split disagrees")
    if set(split["forbidden_anchor_ids"]) != {
        "sglang_prefill_2k",
        "sglang_prefill_4k",
        "sglang_decode_simulated_mtp",
    }:
        raise ValueError("COMP-75 forbidden calibration split disagrees")
    if split["held_out_access_ledger"] != []:
        raise ValueError("COMP-75 held-out access ledger must be empty")

    parameters = expectations["parameters"]
    if parameters["fitted"] or parameters["free"]:
        raise ValueError("COMP-75 cannot contain fitted or free parameters")
    if set(parameters["rejected"]) != {
        "communication_scale_factor",
        "overlap_fraction",
        "destination_dedup_factor",
    }:
        raise ValueError("COMP-75 rejected parameter inventory disagrees")

    traffic = expectations["traffic"]
    experts = _require_int("logical_experts", traffic["logical_experts"], minimum=1)
    ep = _require_int("expert_parallel", traffic["expert_parallel"], minimum=1)
    top_k = _require_int("top_k", traffic["top_k"], minimum=1)
    if experts % ep:
        raise ValueError("logical experts must divide evenly over EP ranks")
    experts_per_rank = experts // ep
    probability = Fraction(
        comb(experts, top_k) - comb(experts - experts_per_rank, top_k),
        comb(experts, top_k),
    )
    probability_row = traffic["dedup_probability"]
    if not _fraction_matches(probability_row, "p_rank", probability):
        raise ValueError("COMP-75 rank incidence probability disagrees")
    if not _fraction_matches(
        probability_row,
        "expected_unique_destinations",
        ep * probability,
    ):
        raise ValueError("COMP-75 unique destination count disagrees")
    if not _fraction_matches(
        probability_row,
        "expected_remote_destinations",
        (ep - 1) * probability,
    ):
        raise ValueError("COMP-75 remote destination count disagrees")
    if not traffic["same_destination_deduplication"]:
        raise ValueError("COMP-75 requires same-destination deduplication")
    if traffic["local_peer_count"] + traffic["remote_peer_count"] != ep - 1:
        raise ValueError("COMP-75 peer partition disagrees")

    hidden = _require_int("hidden_size", traffic["hidden_size"], minimum=1)
    group = _require_int("fp8_group_elements", traffic["fp8_group_elements"], minimum=1)
    if hidden % group:
        raise ValueError("FP8 groups must divide the hidden width")
    dispatch_width = (
        hidden * _require_int("fp8_bytes_per_element", traffic["fp8_bytes_per_element"], minimum=1)
        + hidden // group * _require_int("fp8_scale_bytes", traffic["fp8_scale_bytes"], minimum=1)
    )
    combine_width = hidden * 2
    if traffic["dispatch"]["vector_bytes"] != dispatch_width:
        raise ValueError("COMP-75 dispatch vector width disagrees")
    if traffic["combine"]["vector_bytes"] != combine_width:
        raise ValueError("COMP-75 combine vector width disagrees")

    tokens = _require_int("new_tokens_per_rank", traffic["new_tokens_per_rank"], minimum=1)
    for phase, width in (("dispatch", dispatch_width), ("combine", combine_width)):
        row = traffic[phase]
        exact = Fraction(tokens * width) * probability
        pair = row["per_pair_bytes"]
        if not _fraction_matches(pair, "exact", exact):
            raise ValueError(f"COMP-75 {phase} exact pair bytes disagree")
        lower = exact.numerator // exact.denominator
        if (pair["lower"], pair["selected"], pair["upper"]) != (
            lower,
            lower,
            lower + 1,
        ):
            raise ValueError(f"COMP-75 {phase} pair-byte envelope disagrees")
        for locality, peers in (
            ("local", traffic["local_peer_count"]),
            ("fabric", traffic["remote_peer_count"]),
        ):
            interval = _interval(row, f"{locality}_bytes_per_phase")
            expected = {
                "lower": peers * lower,
                "selected": peers * lower,
                "upper": peers * (lower + 1),
            }
            if interval != expected:
                raise ValueError(f"COMP-75 {phase} {locality} bytes disagree")

    composition = expectations["composition"]
    if composition["operator"] != "max":
        raise ValueError("COMP-75 composition must be max-like")
    if composition["selected_arm"] != "point":
        raise ValueError("COMP-75 selected service arm disagrees")
    interleaving = composition["two_batch_interleaving"]
    if interleaving != {
        "copies_of_operation_sequence": 2,
        "stage_offsets": [0, "tbo_delta_stages"],
    }:
        raise ValueError("COMP-75 two-batch interleaving disagrees")
    compute = _require_int(
        "candidate_compute_service_ps",
        composition["candidate_compute_service_ps"],
        minimum=1,
    )
    arms = _arms(expectations)
    if set(arms) != {"point", "sensitivity"}:
        raise ValueError("COMP-75 service arm inventory disagrees")
    for arm_id, rate in (("point", 400_000_000_000), ("sensitivity", 200_000_000_000)):
        arm = arms[arm_id]
        if arm["fabric_link_rate_bits_per_second"] != rate:
            raise ValueError(f"COMP-75 {arm_id} fabric rate disagrees")
        dispatch = _interval(arm, "dispatch_phase_service_ps")
        combine = _interval(arm, "combine_phase_service_ps")
        communication = _interval(arm, "communication_service_ps")
        total = _interval(arm, "total_step_service_ps")
        exposed = _interval(arm, "exposed_incremental_service_ps")
        for edge in ("lower", "selected", "upper"):
            expected_comm = traffic["moe_layers"] * (dispatch[edge] + combine[edge])
            if communication[edge] != expected_comm:
                raise ValueError(f"COMP-75 {arm_id} communication service disagrees")
            if total[edge] != max(compute, expected_comm):
                raise ValueError(f"COMP-75 {arm_id} max composition disagrees")
            if exposed[edge] != total[edge] - compute:
                raise ValueError(f"COMP-75 {arm_id} exposed service disagrees")
        if arm["hidden_compute_service_ps"] != min(compute, communication["selected"]):
            raise ValueError(f"COMP-75 {arm_id} hiding budget disagrees")

    if expectations["decode_disposition"] != {
        "mechanism_count": 0,
        "owner": "SGL-38",
        "pricing_changed": False,
        "signed_movement": "unchanged",
    }:
        raise ValueError("COMP-75 decode disposition disagrees")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _locked_path(repository_root: Path, row: Mapping[str, Any]) -> Path:
    relative = PurePosixPath(str(row["path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("preservation paths must be repository-relative")
    return repository_root.joinpath(*relative.parts)


def verify_evidence_locks(
    expectations: Mapping[str, Any], repository_root: Path
) -> dict[str, Any]:
    """Verify component, allowlist and void-record preservation locks."""

    validate_expectations(expectations)
    checked: list[dict[str, str]] = []
    allowlist = expectations["source_allowlist"]
    allowlist_path = _locked_path(repository_root, allowlist)
    observed_allowlist = _sha256(allowlist_path)
    if observed_allowlist != allowlist["portable_rendering_sha256"]:
        raise ValueError("portable source allowlist digest disagrees")
    checked.append({"path": allowlist["path"], "sha256": observed_allowlist})

    component_keys = (
        "candidate_compute_record",
        "core59_expectations",
        "fabric_sensitivity",
        "model_extraction",
        "placement_point",
    )
    for key in component_keys:
        row = expectations["component_evidence"][key]
        path = _locked_path(repository_root, row)
        observed = _sha256(path)
        if observed != row["sha256"]:
            raise ValueError(f"component evidence digest disagrees for {row['path']}")
        checked.append({"path": row["path"], "sha256": observed})

    void_checked = []
    for row in expectations["preservation_locks"]["void_core60_artifacts"]:
        path = _locked_path(repository_root, row)
        observed = _sha256(path)
        if observed != row["sha256"]:
            raise ValueError(f"void CORE-60 digest disagrees for {row['path']}")
        entry = {"path": row["path"], "sha256": observed}
        checked.append(entry)
        void_checked.append(entry)
    return {
        "status": "PASS",
        "checked_artifacts": checked,
        "void_core60_artifacts": void_checked,
        "void_core60_mutated": False,
    }


def compare_core60_contracts(
    expectations: Mapping[str, Any], core60: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare only physical CORE-60 fields, never its external-source fields."""

    validate_expectations(expectations)
    traffic = expectations["traffic"]
    void_traffic = core60["traffic"]
    probability_keys = (
        "p_rank_numerator",
        "p_rank_denominator",
        "expected_unique_destinations_numerator",
        "expected_unique_destinations_denominator",
        "expected_remote_destinations_numerator",
        "expected_remote_destinations_denominator",
    )
    destination = all(
        traffic["dedup_probability"][key] == void_traffic["dedup_probability"][key]
        for key in probability_keys
    ) and all(
        traffic[key] == void_traffic[key]
        for key in (
            "moe_layers",
            "new_tokens_per_rank",
            "local_peer_count",
            "remote_peer_count",
        )
    )

    phase_keys = (
        "vector_bytes",
        "per_pair_bytes",
        "local_bytes_per_phase",
        "fabric_bytes_per_phase",
    )
    packet_arithmetic = all(
        traffic[phase][key] == void_traffic[phase][key]
        for phase in ("dispatch", "combine")
        for key in phase_keys
    )
    composition = expectations["composition"]
    void_composition = core60["composition"]
    clean_arms = _arms(expectations)
    void_arms = {str(row["id"]): row for row in void_composition["service_arms"]}
    arm_keys = (
        "fabric_link_rate_bits_per_second",
        "dispatch_phase_service_ps",
        "combine_phase_service_ps",
        "communication_service_ps",
        "total_step_service_ps",
        "exposed_incremental_service_ps",
        "hidden_compute_service_ps",
    )
    packet_services = packet_arithmetic and all(
        clean_arms[arm][key] == void_arms[arm][key]
        for arm in ("point", "sensitivity")
        for key in arm_keys[:3]
    )
    max_like = (
        composition["candidate_compute_service_ps"]
        == void_composition["candidate_compute_service_ps"]
        and composition["selected_arm"] == void_composition["selected_arm"]
        and all(
            clean_arms[arm][key] == void_arms[arm][key]
            for arm in ("point", "sensitivity")
            for key in arm_keys[3:]
        )
    )
    contracts = {
        "destination_arithmetic": "REPRODUCED" if destination else "REFUTED",
        "packet_services": "REPRODUCED" if packet_services else "REFUTED",
        "max_like_composition": "REPRODUCED" if max_like else "REFUTED",
    }
    reproduced = all(value == "REPRODUCED" for value in contracts.values())
    return {
        "verdict": "REPRODUCED" if reproduced else "REFUTED",
        "contracts": contracts,
        "core60_external_source_values_accessed": False,
        "core60_record_status_unchanged": "VOID",
        "core60_promoted": False,
    }


def _select_visible(
    rows: list[Mapping[str, Any]], id_field: str, visible_ids: tuple[str, ...]
) -> dict[str, Mapping[str, Any]]:
    allowed = set(visible_ids)
    selected = {
        str(row.get(id_field)): row
        for row in rows
        if row.get(id_field) in allowed
    }
    if set(selected) != allowed:
        raise ValueError("COMP-75 visible allowlist did not resolve")
    return selected


def _signed_error(predicted: Fraction, target: Fraction) -> Fraction:
    return (predicted - target) / target


def _movement(updated: Fraction, reference: Fraction) -> dict[str, Any]:
    delta = updated - reference
    direction = "decrease" if delta < 0 else "increase" if delta > 0 else "unchanged"
    return {
        "absolute_tokens_per_second_per_node": fraction_json(delta),
        "direction": direction,
        "relative_to_reference": fraction_json(delta / reference),
    }


def calibration_comparison(
    anchor_freeze: Mapping[str, Any],
    scored_freeze: Mapping[str, Any],
    core59_expectations: Mapping[str, Any],
    expectations: Mapping[str, Any],
) -> dict[str, Any]:
    """Price only the preregistered visible 1K row with no fitted parameter."""

    validate_expectations(expectations)
    visible = tuple(expectations["calibration_split"]["visible_anchor_ids"])
    anchors = _select_visible(anchor_freeze["anchors"], "id", visible)
    rows = _select_visible(
        scored_freeze["pre_tuning_predicted_bands"], "anchor_id", visible
    )
    traffic = expectations["traffic"]
    shape = {
        "role": "prefill",
        "expert_parallel": traffic["expert_parallel"],
        "hidden_size": traffic["hidden_size"],
        "moe_layers": traffic["moe_layers"],
        "new_tokens_per_rank": traffic["new_tokens_per_rank"],
        "top_k": traffic["top_k"],
        "vector_bytes_per_element": 2,
    }
    arm = _arms(expectations)[expectations["composition"]["selected_arm"]]
    total_service = _interval(arm, "total_step_service_ps")
    result_rows = []
    for anchor_id in visible:
        frozen_row = rows[anchor_id]
        tokens = _require_int("per_node_tokens", frozen_row["per_node_tokens"], minimum=1)
        candidate_service = _require_int(
            "candidate_service_ps", frozen_row["candidate_service_ps"], minimum=1
        )
        candidate = Fraction(tokens * PS_PER_SECOND, candidate_service)
        previous = core59_prediction(frozen_row, core59_expectations, shape)
        predictions = {
            edge: Fraction(tokens * PS_PER_SECOND, total_service[edge])
            for edge in ("lower", "selected", "upper")
        }
        updated = predictions["selected"]
        target = as_fraction(anchors[anchor_id]["value"], f"{anchor_id}.value")
        result_rows.append(
            {
                "anchor_id": anchor_id,
                "baseline_candidate_only": fraction_json(candidate),
                "core59_point": fraction_json(previous),
                "published": fraction_json(target),
                "prediction": {
                    "lower": fraction_json(min(predictions.values())),
                    "point": fraction_json(updated),
                    "upper": fraction_json(max(predictions.values())),
                },
                "candidate_service_ps": candidate_service,
                "core59_total_service_ps": tokens * PS_PER_SECOND // previous,
                "per_node_tokens": tokens,
                "total_service_ps": total_service["selected"],
                "signed_movement_from_candidate_only": _movement(updated, candidate),
                "signed_movement_from_core59": _movement(updated, previous),
                "signed_relative_error_before": fraction_json(
                    _signed_error(candidate, target)
                ),
                "signed_relative_error_core59": fraction_json(
                    _signed_error(previous, target)
                ),
                "signed_relative_error_after": fraction_json(
                    _signed_error(updated, target)
                ),
                "absolute_remaining_error_tokens_per_second_per_node": fraction_json(
                    updated - target
                ),
            }
        )

    expected = expectations["signed_movement_expectations"][0]
    observed = result_rows[0]
    if observed["signed_movement_from_candidate_only"]["direction"] != expected[
        "direction_relative_to_candidate_only"
    ]:
        raise ValueError("candidate-only movement sign disagrees")
    if observed["signed_movement_from_core59"]["direction"] != expected[
        "direction_relative_to_core59"
    ]:
        raise ValueError("CORE-59 movement sign disagrees")
    return {
        "schema": CALIBRATION_SCHEMA,
        "status": "PASS",
        "classification": "CALIBRATION_ONLY_NOT_SCORED_CLEAN_REPETITION",
        "accessed_visible_anchor_ids": list(visible),
        "held_out_access_ledger": [],
        "held_out_numeric_values_accessed": False,
        "held_out_score_performed": False,
        "scored_flagship_rerun_performed": False,
        "fitted_parameters": [],
        "free_parameters": [],
        "decode_pricing_changed": False,
        "calibration_rows": result_rows,
    }


__all__ = [
    "CALIBRATION_SCHEMA",
    "EXPECTATIONS_SCHEMA",
    "calibration_comparison",
    "compare_core60_contracts",
    "validate_expectations",
    "verify_evidence_locks",
]
