"""Frozen role and shape mechanisms for the CORE-54 flagship successor."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any

from curve_tools import as_fraction, fraction_json

EXPECTATIONS_SCHEMA = "simllm-deployment-curve-core59-expectations-v1"
CALIBRATION_SCHEMA = "simllm-deployment-curve-core59-calibration-v1"
PS_PER_SECOND = 1_000_000_000_000
PREFILL_MECHANISM_ID = "prefill_ep32_moe_dispatch_combine_v1"


def _require_int(name: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _constant_map(expectations: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = expectations["constants"]["declared"]
    selected = {str(row["id"]): row for row in rows}
    if len(selected) != len(rows):
        raise ValueError("declared constant IDs must be unique")
    return selected


def _selected_constants(expectations: Mapping[str, Any]) -> dict[str, int]:
    return {
        constant_id: _require_int(
            f"constants.declared[{constant_id}].selected",
            row["selected"],
            minimum=1,
        )
        for constant_id, row in _constant_map(expectations).items()
    }


def _ceil_local_service_ps(payload_bytes: int, rate_bytes_per_second: int) -> int:
    service_ns = (
        payload_bytes * 1_000_000_000 + rate_bytes_per_second - 1
    ) // rate_bytes_per_second
    return service_ns * 1_000


def _arm_map(mechanism: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    arms = mechanism["physical_service"]["arms"]
    selected = {str(arm["id"]): arm for arm in arms}
    if len(selected) != len(arms):
        raise ValueError("physical service arm IDs must be unique")
    return selected


def validate_expectations(expectations: Mapping[str, Any]) -> None:
    """Validate the no-fit mechanism freeze and its physical arithmetic."""

    if expectations.get("schema") != EXPECTATIONS_SCHEMA:
        raise ValueError("CORE-59 expectations schema disagrees")
    if expectations.get("status") != "EXPECTATIONS_ONLY":
        raise ValueError("CORE-59 expectations must remain expectations-only")
    if expectations.get("task") != "CORE-59":
        raise ValueError("CORE-59 task identity disagrees")

    chronology = expectations["chronology"]
    if chronology != {
        "calibration_numeric_values_accessed": False,
        "held_out_numeric_values_accessed": False,
        "mechanism_service_derived_before_comparison": True,
        "scored_comparison_performed": False,
    }:
        raise ValueError("CORE-59 freeze chronology disagrees")

    split = expectations["calibration_split"]
    if split["visible_anchor_ids"] != [
        "sglang_prefill_1k",
        "sglang_decode_standard",
    ]:
        raise ValueError("CORE-59 calibration split disagrees")
    if set(split["forbidden_anchor_ids"]) != {
        "sglang_prefill_2k",
        "sglang_prefill_4k",
        "sglang_decode_simulated_mtp",
    }:
        raise ValueError("CORE-59 forbidden anchor set disagrees")

    constants = expectations["constants"]
    if constants["new_free_constants"] or constants["tunable"]:
        raise ValueError("CORE-59 cannot contain free or tunable constants")
    selected = _selected_constants(expectations)
    expected_selected = {
        "deepseek_v3_moe_layers": 58,
        "moe_collective_phases_per_layer": 2,
        "prefill_expert_parallel_ranks": 32,
        "prefill_new_tokens_per_rank": 16_384,
        "routed_experts_per_token": 8,
        "hidden_vector_elements": 7_168,
        "routed_vector_bytes_per_element": 2,
        "nvlink_endpoint_bandwidth_bytes_per_second": 450_000_000_000,
        "fabric_link_rate_bits_per_second": 400_000_000_000,
    }
    if selected != expected_selected:
        raise ValueError("CORE-59 declared constants disagree")
    for constant_id, row in _constant_map(expectations).items():
        envelope = row["envelope"]
        lower = _require_int(f"{constant_id}.envelope.lower", envelope["lower"])
        upper = _require_int(f"{constant_id}.envelope.upper", envelope["upper"])
        if not lower <= row["selected"] <= upper:
            raise ValueError(f"{constant_id} is outside its physical envelope")
        if not isinstance(row.get("justification"), str) or not row["justification"].strip():
            raise ValueError(f"{constant_id} needs a physical justification")

    residual = expectations["shared_residual_disposition"]
    if (
        residual["historical_id"] != "intra_node_collective_surcharge_ps"
        or residual["next_run_application_count"] != 0
        or residual["next_run_selected_ps"] is not None
    ):
        raise ValueError("the shared residual was not retired from the new path")

    mechanisms = expectations["mechanisms"]
    if len(mechanisms) != 1 or mechanisms[0]["id"] != PREFILL_MECHANISM_ID:
        raise ValueError("CORE-59 must freeze exactly one prefill mechanism")
    mechanism = mechanisms[0]
    if mechanism["role"] != "prefill" or mechanism["tunable"]:
        raise ValueError("the prefill mechanism role or fit disposition drifted")
    if mechanism["data_parallel_attention_synchronization_count"] != 0:
        raise ValueError("DP attention synchronization must remain absent")
    gate = mechanism["shape_gate"]
    if gate != {
        "expert_parallel": 32,
        "hidden_size": 7_168,
        "moe_layers": 58,
        "new_tokens_per_rank": 16_384,
        "top_k": 8,
        "vector_bytes_per_element": 2,
    }:
        raise ValueError("the prefill mechanism shape gate drifted")

    traffic = mechanism["traffic_arithmetic"]
    per_pair = (
        gate["new_tokens_per_rank"]
        * gate["top_k"]
        * gate["hidden_size"]
        * gate["vector_bytes_per_element"]
        // gate["expert_parallel"]
    )
    if traffic["per_pair_bytes"] != per_pair:
        raise ValueError("prefill per-pair byte arithmetic disagrees")
    if (traffic["local_peer_count"], traffic["remote_peer_count"]) != (7, 24):
        raise ValueError("the four-node EP32 peer partition disagrees")
    local_bytes = traffic["local_peer_count"] * per_pair
    fabric_bytes = traffic["remote_peer_count"] * per_pair
    if (
        traffic["local_bytes_per_phase"] != local_bytes
        or traffic["fabric_bytes_per_phase"] != fabric_bytes
        or traffic["total_directed_bytes_per_phase"] != local_bytes + fabric_bytes
    ):
        raise ValueError("prefill phase bytes are not conserved")
    applications = gate["moe_layers"] * 2
    if traffic["application_count"] != applications:
        raise ValueError("prefill mechanism application count disagrees")
    expected_local_ps = _ceil_local_service_ps(
        local_bytes,
        selected["nvlink_endpoint_bandwidth_bytes_per_second"],
    )

    arms = _arm_map(mechanism)
    if set(arms) != {"point", "sensitivity"}:
        raise ValueError("CORE-59 physical service arms disagree")
    if mechanism["physical_service"]["selected_arm"] != "point":
        raise ValueError("the PLACE-5 point arm must remain selected")
    for arm_id, link_rate in (
        ("point", 400_000_000_000),
        ("sensitivity", 200_000_000_000),
    ):
        arm = arms[arm_id]
        if arm["fabric_link_rate_bits_per_second"] != link_rate:
            raise ValueError(f"{arm_id} fabric link rate disagrees")
        if arm["local_phase_service_ps"] != expected_local_ps:
            raise ValueError(f"{arm_id} local serializer arithmetic disagrees")
        if arm["composed_phase_service_ps"] != max(
            arm["local_phase_service_ps"], arm["fabric_phase_service_ps"]
        ):
            raise ValueError(f"{arm_id} phase composition disagrees")
        if arm["total_mechanism_service_ps"] != (arm["composed_phase_service_ps"] * applications):
            raise ValueError(f"{arm_id} total mechanism service disagrees")
    service = mechanism["physical_service"]
    endpoints = sorted(arm["total_mechanism_service_ps"] for arm in arms.values())
    if service["envelope"] != {"lower": endpoints[0], "upper": endpoints[1]}:
        raise ValueError("prefill mechanism service envelope disagrees")

    decode = expectations["decode_disposition"]
    if (
        decode["mechanism_count"] != 0
        or decode["data_parallel_attention_synchronization_count"] != 0
        or decode["signed_movement"] != "zero"
    ):
        raise ValueError("decode must retain zero CORE-59 mechanisms")
    if decode["shape_gate"] != {
        "batch_size": 32,
        "expert_parallel": 72,
        "moe_layers": 58,
        "new_tokens_per_rank": 32,
        "per_request_kv_length": 2_000,
    }:
        raise ValueError("the standard-decode shape gate drifted")

    movements = {
        row["anchor_id"]: row["direction"] for row in expectations["signed_movement_expectations"]
    }
    if movements != {
        "sglang_prefill_1k": "decrease",
        "sglang_decode_standard": "unchanged",
    }:
        raise ValueError("frozen signed movement expectations disagree")


def verify_historical_refutation(
    expectations: Mapping[str, Any],
    repository_root: Path,
) -> dict[str, Any]:
    """Verify every first-run artifact remains byte-identical."""

    validate_expectations(expectations)
    checked = []
    for row in expectations["historical_refutation_lock"]["tracked_artifacts"]:
        relative = PurePosixPath(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("historical artifact paths must stay repository-relative")
        path = repository_root.joinpath(*relative.parts)
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed != row["sha256"]:
            raise ValueError(f"historical artifact digest disagrees for {row['path']}")
        checked.append({"path": row["path"], "sha256": observed})
    return {
        "status": "PASS",
        "checked_artifacts": checked,
        "first_scored_run_mutated": False,
    }


def calibration_shapes(expectations: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Return only the two shapes named by the frozen calibration split."""

    validate_expectations(expectations)
    mechanism = expectations["mechanisms"][0]
    return {
        "sglang_prefill_1k": {
            "role": "prefill",
            **dict(mechanism["shape_gate"]),
        },
        "sglang_decode_standard": {
            "role": "decode",
            **dict(expectations["decode_disposition"]["shape_gate"]),
        },
    }


def mechanism_service_ps(
    expectations: Mapping[str, Any],
    shape: Mapping[str, Any],
    *,
    arm_id: str = "point",
) -> int:
    """Return frozen added service for one exact role and shape."""

    validate_expectations(expectations)
    role = shape.get("role")
    if role == "prefill":
        mechanism = expectations["mechanisms"][0]
        gate = mechanism["shape_gate"]
        if any(shape.get(name) != expected for name, expected in gate.items()):
            raise ValueError("prefill shape is outside the frozen CORE-59 gate")
        arms = _arm_map(mechanism)
        if arm_id not in arms:
            raise ValueError(f"unknown CORE-59 service arm {arm_id!r}")
        return int(arms[arm_id]["total_mechanism_service_ps"])
    if role == "decode":
        gate = expectations["decode_disposition"]["shape_gate"]
        if any(shape.get(name) != expected for name, expected in gate.items()):
            raise ValueError("decode shape is outside the frozen CORE-59 gate")
        if arm_id not in {"point", "sensitivity"}:
            raise ValueError(f"unknown CORE-59 service arm {arm_id!r}")
        return 0
    raise ValueError("CORE-59 shape role must be prefill or decode")


def prediction_at_role_shape_service(
    frozen_row: Mapping[str, Any],
    expectations: Mapping[str, Any],
    shape: Mapping[str, Any],
    *,
    arm_id: str = "point",
) -> Fraction:
    """Price one candidate row with its frozen role and shape mechanism."""

    candidate_service_ps = _require_int(
        "candidate_service_ps", frozen_row["candidate_service_ps"], minimum=1
    )
    per_node_tokens = _require_int("per_node_tokens", frozen_row["per_node_tokens"], minimum=1)
    added_service_ps = mechanism_service_ps(expectations, shape, arm_id=arm_id)
    return Fraction(
        per_node_tokens * PS_PER_SECOND,
        candidate_service_ps + added_service_ps,
    )


def role_shape_prediction_interval(
    frozen_row: Mapping[str, Any],
    expectations: Mapping[str, Any],
    shape: Mapping[str, Any],
) -> dict[str, Any]:
    """Propagate the two physical link arms around the selected point."""

    point = prediction_at_role_shape_service(frozen_row, expectations, shape, arm_id="point")
    sensitivity = prediction_at_role_shape_service(
        frozen_row, expectations, shape, arm_id="sensitivity"
    )
    lower = min(point, sensitivity)
    upper = max(point, sensitivity)
    point_service = mechanism_service_ps(expectations, shape, arm_id="point")
    sensitivity_service = mechanism_service_ps(expectations, shape, arm_id="sensitivity")
    return {
        "lower": fraction_json(lower),
        "point": fraction_json(point),
        "upper": fraction_json(upper),
        "contributions": [
            {
                "source_kind": "candidate-record",
                "source_id": expectations["evidence"]["candidate_record"]["sha256"],
                "point_service_ps": int(frozen_row["candidate_service_ps"]),
            },
            {
                "source_kind": "role-shape-mechanism",
                "source_id": (PREFILL_MECHANISM_ID if shape["role"] == "prefill" else None),
                "lower_service_ps": min(point_service, sensitivity_service),
                "selected_service_ps": point_service,
                "upper_service_ps": max(point_service, sensitivity_service),
            },
        ],
    }


def _anchor_subset(
    anchor_freeze: Mapping[str, Any],
    allowed_ids: tuple[str, ...],
) -> dict[str, Mapping[str, Any]]:
    """Read numeric anchor fields only after an ID passes the allowlist."""

    allowed = set(allowed_ids)
    selected = {
        str(row.get("id")): row for row in anchor_freeze["anchors"] if row.get("id") in allowed
    }
    if set(selected) != allowed:
        raise ValueError("CORE-59 calibration anchor allowlist did not resolve")
    return selected


def _signed_error(predicted: Fraction, target: Fraction) -> Fraction:
    return (predicted - target) / target


def fit_calibration_only(
    anchor_freeze: Mapping[str, Any],
    scored_freeze: Mapping[str, Any],
    expectations: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze the no-parameter projection on calibration anchors only."""

    validate_expectations(expectations)
    visible = tuple(expectations["calibration_split"]["visible_anchor_ids"])
    anchors = _anchor_subset(anchor_freeze, visible)
    rows = {
        row["anchor_id"]: row
        for row in scored_freeze["pre_tuning_predicted_bands"]
        if row["anchor_id"] in set(visible)
    }
    if set(rows) != set(visible):
        raise ValueError("CORE-59 calibration projection rows disagree")
    shapes = calibration_shapes(expectations)

    calibration_rows = []
    for anchor_id in visible:
        target = as_fraction(anchors[anchor_id]["value"], f"{anchor_id}.value")
        frozen_row = rows[anchor_id]
        baseline = Fraction(
            int(frozen_row["per_node_tokens"]) * PS_PER_SECOND,
            int(frozen_row["candidate_service_ps"]),
        )
        interval = role_shape_prediction_interval(
            frozen_row,
            expectations,
            shapes[anchor_id],
        )
        updated = as_fraction(interval["point"], f"{anchor_id}.point")
        delta = updated - baseline
        direction = "decrease" if delta < 0 else "increase" if delta > 0 else "unchanged"
        calibration_rows.append(
            {
                "anchor_id": anchor_id,
                "baseline_candidate_only": fraction_json(baseline),
                "published": fraction_json(target),
                "prediction": interval,
                "mechanism_service_ps": mechanism_service_ps(expectations, shapes[anchor_id]),
                "signed_movement": {
                    "absolute_tokens_per_second_per_node": fraction_json(delta),
                    "direction": direction,
                    "relative_to_baseline": fraction_json(delta / baseline),
                },
                "signed_relative_error_before": fraction_json(_signed_error(baseline, target)),
                "signed_relative_error_after": fraction_json(_signed_error(updated, target)),
            }
        )

    expected_directions = {
        row["anchor_id"]: row["direction"] for row in expectations["signed_movement_expectations"]
    }
    observed_directions = {
        row["anchor_id"]: row["signed_movement"]["direction"] for row in calibration_rows
    }
    if observed_directions != expected_directions:
        raise ValueError("CORE-59 signed calibration movement disagrees with the freeze")
    return {
        "schema": CALIBRATION_SCHEMA,
        "status": "FROZEN",
        "classification": "CALIBRATION_ONLY_NOT_SCORED",
        "accessed_anchor_ids": list(visible),
        "forbidden_anchor_ids_accessed": [],
        "held_out_numeric_values_accessed": False,
        "held_out_score_performed": False,
        "fitted_parameters": [],
        "shared_collective_surcharge_application_count": 0,
        "calibration_rows": calibration_rows,
    }


__all__ = [
    "CALIBRATION_SCHEMA",
    "EXPECTATIONS_SCHEMA",
    "PREFILL_MECHANISM_ID",
    "calibration_shapes",
    "fit_calibration_only",
    "mechanism_service_ps",
    "prediction_at_role_shape_service",
    "role_shape_prediction_interval",
    "validate_expectations",
    "verify_historical_refutation",
]
