"""Frozen CORE-60 EP32 prefill service composition."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from fractions import Fraction
from math import comb
from pathlib import Path, PurePosixPath
from typing import Any

from core59_role_mechanisms import (
    calibration_shapes as core59_calibration_shapes,
)
from core59_role_mechanisms import (
    prediction_at_role_shape_service as core59_prediction,
)
from curve_tools import as_fraction, fraction_json

EXPECTATIONS_SCHEMA = "simllm-deployment-curve-core60-expectations-v1"
CALIBRATION_SCHEMA = "simllm-deployment-curve-core60-calibration-v1"
PS_PER_SECOND = 1_000_000_000_000
PREFILL_ANCHOR_ID = "sglang_prefill_1k"
DECODE_ANCHOR_ID = "sglang_decode_standard"


def _require_int(name: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _service_arms(expectations: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = expectations["composition"]["service_arms"]
    arms = {str(row["id"]): row for row in rows}
    if len(arms) != len(rows):
        raise ValueError("CORE-60 service arm IDs must be unique")
    return arms


def _interval_value(row: Mapping[str, Any], name: str) -> dict[str, int]:
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


def validate_expectations(expectations: Mapping[str, Any]) -> None:
    """Validate the expectations-only, no-fit CORE-60 freeze."""

    if expectations.get("schema") != EXPECTATIONS_SCHEMA:
        raise ValueError("CORE-60 expectations schema disagrees")
    if expectations.get("task") != "CORE-60":
        raise ValueError("CORE-60 task identity disagrees")
    if expectations.get("status") != "EXPECTATIONS_ONLY_VOID":
        raise ValueError("CORE-60 expectations must retain their protocol-void status")

    chronology = expectations["chronology"]
    if chronology != {
        "core59_public_result_known": True,
        "core60_calibration_comparison_performed": False,
        "held_out_numeric_values_accessed": True,
        "mechanism_service_derived_before_comparison": True,
        "scored_comparison_performed": False,
        "visible_calibration_values_already_public_from_core59": True,
    }:
        raise ValueError("CORE-60 freeze chronology disagrees")
    invalidation = expectations["invalidation"]
    if invalidation["external_source_exposed_forbidden_anchor_ids"] != [
        "sglang_prefill_2k",
        "sglang_prefill_4k",
    ] or invalidation["replacement_owner"] != "COMP-75":
        raise ValueError("CORE-60 protocol invalidation disagrees")

    split = expectations["calibration_split"]
    if split["visible_anchor_ids"] != [PREFILL_ANCHOR_ID, DECODE_ANCHOR_ID]:
        raise ValueError("CORE-60 visible calibration split disagrees")
    if set(split["forbidden_anchor_ids"]) != {
        "sglang_prefill_2k",
        "sglang_prefill_4k",
        "sglang_decode_simulated_mtp",
    }:
        raise ValueError("CORE-60 forbidden calibration split disagrees")

    parameters = expectations["parameters"]
    if parameters["fitted"] or parameters["free"]:
        raise ValueError("CORE-60 cannot contain fitted or free parameters")
    if set(parameters["rejected"]) != {
        "communication_scale_factor",
        "overlap_fraction",
        "destination_dedup_factor",
    }:
        raise ValueError("CORE-60 rejected parameter inventory disagrees")

    contracts = {row["contract"]: row for row in expectations["contracts"]}
    if set(contracts) != {
        "per-rank token ownership",
        "routed wire precision",
        "same-destination expert deduplication",
        "framework-supported compute and communication overlap",
    }:
        raise ValueError("CORE-60 physical contract inventory disagrees")
    if not all(row["adopted"] for row in contracts.values()):
        raise ValueError("every frozen CORE-60 contract must be adopted")
    if contracts["per-rank token ownership"]["signed_effect_relative_to_core59"] != (
        "unchanged"
    ):
        raise ValueError("token ownership must preserve CORE-59's population")
    for name in (
        "routed wire precision",
        "same-destination expert deduplication",
        "framework-supported compute and communication overlap",
    ):
        if contracts[name]["signed_effect_relative_to_core59"] != "increase-throughput":
            raise ValueError(f"{name} signed effect disagrees")

    traffic = expectations["traffic"]
    if (traffic["moe_layers"], traffic["new_tokens_per_rank"]) != (58, 16_384):
        raise ValueError("CORE-60 traffic shape disagrees")
    if (traffic["local_peer_count"], traffic["remote_peer_count"]) != (7, 24):
        raise ValueError("CORE-60 placement peer partition disagrees")
    probability = Fraction(comb(256, 8) - comb(248, 8), comb(256, 8))
    probability_row = traffic["dedup_probability"]
    if probability != Fraction(
        probability_row["p_rank_numerator"],
        probability_row["p_rank_denominator"],
    ):
        raise ValueError("CORE-60 destination probability disagrees")
    if Fraction(32) * probability != Fraction(
        probability_row["expected_unique_destinations_numerator"],
        probability_row["expected_unique_destinations_denominator"],
    ):
        raise ValueError("CORE-60 expected unique destination count disagrees")
    if Fraction(31) * probability != Fraction(
        probability_row["expected_remote_destinations_numerator"],
        probability_row["expected_remote_destinations_denominator"],
    ):
        raise ValueError("CORE-60 expected remote destination count disagrees")

    expected_vectors = {"dispatch": 7_392, "combine": 14_336}
    for phase, vector_bytes in expected_vectors.items():
        phase_row = traffic[phase]
        if phase_row["vector_bytes"] != vector_bytes:
            raise ValueError(f"CORE-60 {phase} vector width disagrees")
        exact = Fraction(16_384 * vector_bytes) * probability
        pair = phase_row["per_pair_bytes"]
        if exact != Fraction(pair["exact_numerator"], pair["exact_denominator"]):
            raise ValueError(f"CORE-60 {phase} exact pair bytes disagree")
        lower = exact.numerator // exact.denominator
        if (pair["lower"], pair["selected"], pair["upper"]) != (
            lower,
            lower,
            lower + 1,
        ):
            raise ValueError(f"CORE-60 {phase} byte rounding envelope disagrees")
        for locality, peers in (("local", 7), ("fabric", 24)):
            interval = _interval_value(phase_row, f"{locality}_bytes_per_phase")
            expected = {
                "lower": peers * lower,
                "selected": peers * lower,
                "upper": peers * (lower + 1),
            }
            if interval != expected:
                raise ValueError(f"CORE-60 {phase} {locality} bytes disagree")

    composition = expectations["composition"]
    compute_ps = _require_int(
        "candidate_compute_service_ps",
        composition["candidate_compute_service_ps"],
        minimum=1,
    )
    if compute_ps != 1_363_249_960_000:
        raise ValueError("CORE-60 candidate compute service disagrees")
    if composition["selected_arm"] != "point":
        raise ValueError("CORE-60 must select the PLACE-5 point arm")
    arms = _service_arms(expectations)
    if set(arms) != {"point", "sensitivity"}:
        raise ValueError("CORE-60 service arm inventory disagrees")
    for arm_id, rate in (("point", 400_000_000_000), ("sensitivity", 200_000_000_000)):
        arm = arms[arm_id]
        if arm["fabric_link_rate_bits_per_second"] != rate:
            raise ValueError(f"CORE-60 {arm_id} fabric rate disagrees")
        dispatch = _interval_value(arm, "dispatch_phase_service_ps")
        combine = _interval_value(arm, "combine_phase_service_ps")
        communication = _interval_value(arm, "communication_service_ps")
        total = _interval_value(arm, "total_step_service_ps")
        exposed = _interval_value(arm, "exposed_incremental_service_ps")
        for edge in ("lower", "selected", "upper"):
            if communication[edge] != 58 * (dispatch[edge] + combine[edge]):
                raise ValueError(f"CORE-60 {arm_id} communication service disagrees")
            if total[edge] != max(compute_ps, communication[edge]):
                raise ValueError(f"CORE-60 {arm_id} max composition disagrees")
            if exposed[edge] != total[edge] - compute_ps:
                raise ValueError(f"CORE-60 {arm_id} exposed service disagrees")
        if arm["hidden_compute_service_ps"] != min(
            compute_ps, communication["selected"]
        ):
            raise ValueError(f"CORE-60 {arm_id} hiding budget disagrees")

    decode = expectations["decode_disposition"]
    if decode != {
        "mechanism_count": 0,
        "owner": "SGL-38",
        "pricing_changed": False,
        "signed_movement": "unchanged",
    }:
        raise ValueError("CORE-60 decode disposition disagrees")


def verify_preservation_locks(
    expectations: Mapping[str, Any],
    repository_root: Path,
) -> dict[str, Any]:
    """Verify CORE-59 and the first scored run remain byte-identical."""

    validate_expectations(expectations)
    checked: list[dict[str, str]] = []
    for group in ("core59_artifacts", "first_scored_run_artifacts"):
        for row in expectations["locks"][group]:
            relative = PurePosixPath(row["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("preservation lock paths must be repository-relative")
            path = repository_root.joinpath(*relative.parts)
            observed = hashlib.sha256(path.read_bytes()).hexdigest()
            if observed != row["sha256"]:
                raise ValueError(f"preservation digest disagrees for {row['path']}")
            checked.append({"path": row["path"], "sha256": observed})
    return {
        "status": "PASS",
        "checked_artifacts": checked,
        "core59_mutated": False,
        "first_scored_run_mutated": False,
    }


def calibration_shapes(
    expectations: Mapping[str, Any],
    core59_expectations: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Reuse the exact CORE-59 role and shape gates."""

    validate_expectations(expectations)
    return core59_calibration_shapes(core59_expectations)


def total_service_ps(
    expectations: Mapping[str, Any],
    shape: Mapping[str, Any],
    *,
    arm_id: str = "point",
    edge: str = "selected",
) -> int:
    """Return frozen whole-step service for an exact role and shape."""

    validate_expectations(expectations)
    if edge not in {"lower", "selected", "upper"}:
        raise ValueError(f"unknown CORE-60 service edge {edge!r}")
    role = shape.get("role")
    if role == "prefill":
        gate = {
            "expert_parallel": 32,
            "hidden_size": 7_168,
            "moe_layers": 58,
            "new_tokens_per_rank": 16_384,
            "top_k": 8,
            "vector_bytes_per_element": 2,
        }
        if any(shape.get(name) != value for name, value in gate.items()):
            raise ValueError("prefill shape is outside the frozen CORE-60 gate")
        arms = _service_arms(expectations)
        if arm_id not in arms:
            raise ValueError(f"unknown CORE-60 service arm {arm_id!r}")
        return int(arms[arm_id]["total_step_service_ps"][edge])
    if role == "decode":
        gate = {
            "batch_size": 32,
            "expert_parallel": 72,
            "moe_layers": 58,
            "new_tokens_per_rank": 32,
            "per_request_kv_length": 2_000,
        }
        if any(shape.get(name) != value for name, value in gate.items()):
            raise ValueError("decode shape is outside the frozen CORE-60 gate")
        if arm_id not in {"point", "sensitivity"}:
            raise ValueError(f"unknown CORE-60 service arm {arm_id!r}")
        return _require_int("candidate_service_ps", shape["candidate_service_ps"], minimum=1)
    raise ValueError("CORE-60 shape role must be prefill or decode")


def prediction_at_composed_service(
    frozen_row: Mapping[str, Any],
    expectations: Mapping[str, Any],
    shape: Mapping[str, Any],
    *,
    arm_id: str = "point",
    edge: str = "selected",
) -> Fraction:
    """Price one row using the frozen total service composition."""

    row_shape = dict(shape)
    if row_shape.get("role") == "decode":
        row_shape["candidate_service_ps"] = _require_int(
            "candidate_service_ps", frozen_row["candidate_service_ps"], minimum=1
        )
    service_ps = total_service_ps(
        expectations,
        row_shape,
        arm_id=arm_id,
        edge=edge,
    )
    tokens = _require_int("per_node_tokens", frozen_row["per_node_tokens"], minimum=1)
    return Fraction(tokens * PS_PER_SECOND, service_ps)


def prediction_interval(
    frozen_row: Mapping[str, Any],
    expectations: Mapping[str, Any],
    shape: Mapping[str, Any],
) -> dict[str, Any]:
    """Propagate bandwidth and integer-byte rounding around the point."""

    service_shape = dict(shape)
    if service_shape.get("role") == "decode":
        service_shape["candidate_service_ps"] = _require_int(
            "candidate_service_ps", frozen_row["candidate_service_ps"], minimum=1
        )
    point = prediction_at_composed_service(frozen_row, expectations, shape)
    endpoints = [
        prediction_at_composed_service(
            frozen_row,
            expectations,
            shape,
            arm_id=arm_id,
            edge=edge,
        )
        for arm_id in ("point", "sensitivity")
        for edge in ("lower", "upper")
    ]
    return {
        "lower": fraction_json(min(endpoints)),
        "point": fraction_json(point),
        "upper": fraction_json(max(endpoints)),
        "contributions": [
            {
                "source_kind": "candidate-record",
                "source_id": expectations["evidence"]["candidate_record"]["sha256"],
                "compute_service_ps": int(frozen_row["candidate_service_ps"]),
            },
            {
                "source_kind": "physical-composition",
                "source_id": "core60_ep32_prefill_composition_v1",
                "lower_total_service_ps": min(
                    total_service_ps(
                        expectations,
                        service_shape,
                        arm_id=arm_id,
                        edge=edge,
                    )
                    for arm_id in ("point", "sensitivity")
                    for edge in ("lower", "upper")
                ),
                "selected_total_service_ps": total_service_ps(
                    expectations, service_shape
                ),
                "upper_total_service_ps": max(
                    total_service_ps(
                        expectations,
                        service_shape,
                        arm_id=arm_id,
                        edge=edge,
                    )
                    for arm_id in ("point", "sensitivity")
                    for edge in ("lower", "upper")
                ),
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
        str(row.get("id")): row
        for row in anchor_freeze["anchors"]
        if row.get("id") in allowed
    }
    if set(selected) != allowed:
        raise ValueError("CORE-60 calibration anchor allowlist did not resolve")
    return selected


def _signed_error(predicted: Fraction, target: Fraction) -> Fraction:
    return (predicted - target) / target


def _movement(updated: Fraction, baseline: Fraction) -> dict[str, Any]:
    delta = updated - baseline
    direction = "decrease" if delta < 0 else "increase" if delta > 0 else "unchanged"
    return {
        "absolute_tokens_per_second_per_node": fraction_json(delta),
        "direction": direction,
        "relative_to_reference": fraction_json(delta / baseline),
    }


def fit_calibration_only(
    anchor_freeze: Mapping[str, Any],
    scored_freeze: Mapping[str, Any],
    core59_expectations: Mapping[str, Any],
    expectations: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute the no-parameter projection on the two visible rows only."""

    validate_expectations(expectations)
    visible = tuple(expectations["calibration_split"]["visible_anchor_ids"])
    anchors = _anchor_subset(anchor_freeze, visible)
    visible_set = set(visible)
    rows = {
        row["anchor_id"]: row
        for row in scored_freeze["pre_tuning_predicted_bands"]
        if row["anchor_id"] in visible_set
    }
    if set(rows) != visible_set:
        raise ValueError("CORE-60 calibration projection rows disagree")
    shapes = calibration_shapes(expectations, core59_expectations)

    result_rows = []
    for anchor_id in visible:
        frozen_row = rows[anchor_id]
        target = as_fraction(anchors[anchor_id]["value"], f"{anchor_id}.value")
        candidate = Fraction(
            int(frozen_row["per_node_tokens"]) * PS_PER_SECOND,
            int(frozen_row["candidate_service_ps"]),
        )
        previous = core59_prediction(
            frozen_row,
            core59_expectations,
            shapes[anchor_id],
        )
        interval = prediction_interval(frozen_row, expectations, shapes[anchor_id])
        updated = as_fraction(interval["point"], f"{anchor_id}.point")
        result_rows.append(
            {
                "anchor_id": anchor_id,
                "baseline_candidate_only": fraction_json(candidate),
                "core59_point": fraction_json(previous),
                "published": fraction_json(target),
                "prediction": interval,
                "total_service_ps": total_service_ps(
                    expectations,
                    {
                        **shapes[anchor_id],
                        "candidate_service_ps": int(frozen_row["candidate_service_ps"]),
                    },
                ),
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
            }
        )

    expected = {
        row["anchor_id"]: (
            row["direction_relative_to_candidate_only"],
            row["direction_relative_to_core59"],
        )
        for row in expectations["signed_movement_expectations"]
    }
    observed = {
        row["anchor_id"]: (
            row["signed_movement_from_candidate_only"]["direction"],
            row["signed_movement_from_core59"]["direction"],
        )
        for row in result_rows
    }
    if observed != expected:
        raise ValueError("CORE-60 signed calibration movement disagrees with the freeze")

    return {
        "schema": CALIBRATION_SCHEMA,
        "status": "VOID",
        "classification": "CALIBRATION_ONLY_NOT_SCORED_PROTOCOL_VOID",
        "accessed_anchor_ids": list(visible),
        "forbidden_anchor_ids_accessed": [],
        "held_out_numeric_values_accessed": True,
        "externally_exposed_held_out_anchor_ids": list(
            expectations["invalidation"][
                "external_source_exposed_forbidden_anchor_ids"
            ]
        ),
        "held_out_score_performed": False,
        "scored_flagship_rerun_performed": False,
        "fitted_parameters": [],
        "decode_pricing_changed": False,
        "calibration_rows": result_rows,
    }


__all__ = [
    "CALIBRATION_SCHEMA",
    "EXPECTATIONS_SCHEMA",
    "PS_PER_SECOND",
    "calibration_shapes",
    "fit_calibration_only",
    "prediction_at_composed_service",
    "prediction_interval",
    "total_service_ps",
    "validate_expectations",
    "verify_preservation_locks",
]
