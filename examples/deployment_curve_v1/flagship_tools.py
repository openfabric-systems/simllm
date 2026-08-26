"""Pure fitting, scoring and curve construction for the CORE-54 flagship."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterable
from fractions import Fraction
from pathlib import Path
from typing import Any

from curve_tools import as_fraction, fraction_json

SCORED_EXPECTATIONS_SCHEMA = "simllm-deployment-curve-scored-expectations-v1"
FLAGSHIP_CONFIG_SCHEMA = "simllm-deployment-curve-flagship-config-v1"
FIT_SCHEMA = "simllm-deployment-curve-flagship-fit-v1"
SCORE_SCHEMA = "simllm-deployment-curve-flagship-score-v1"
CURVE_SCHEMA = "simllm-deployment-curve-v1"
POINT_SCHEMA = "simllm-deployment-curve-point-v1"
PS_PER_SECOND = 1_000_000_000_000


def load_json(path: Path) -> dict[str, Any]:
    """Read one UTF-8 JSON object."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def write_json(path: Path, value: object) -> None:
    """Write deterministic UTF-8 JSON with LF line endings."""

    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def sha256(path: Path) -> str:
    """Return one file's hexadecimal SHA-256 digest."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_scored_expectations(value: dict[str, Any]) -> None:
    """Validate the chronology, split and non-imputation rules."""

    if value.get("schema") != SCORED_EXPECTATIONS_SCHEMA:
        raise ValueError("scored expectations schema disagrees")
    if value.get("status") != "EXPECTATIONS_ONLY":
        raise ValueError("scored expectations must retain expectations-only status")
    chronology = value["chronology"]
    if any(
        chronology[name]
        for name in (
            "scored_runner_existed_before_this_freeze",
            "fitted_constants_existed_before_this_freeze",
            "held_out_score_existed_before_this_freeze",
            "flagship_figure_existed_before_this_freeze",
        )
    ):
        raise ValueError("scored output must not predate the freeze")
    allocation = value["allocation_ruling"]
    if (
        allocation["prefill_experiment"]["simultaneous_with_decode_experiment"]
        or allocation["decode_experiment"]["simultaneous_with_prefill_experiment"]
        or allocation["structural_comparator_only"]["may_be_called_96_gpu_system"]
    ):
        raise ValueError("the separate-experiment allocation ruling drifted")
    score = value["scoring_rule"]
    scorable = set(score["scorable_held_out_anchor_ids"])
    blocked = set(score["blocked_held_out_anchor_ids"])
    if scorable != {"sglang_prefill_2k", "sglang_prefill_4k"}:
        raise ValueError("scorable held-out anchor set disagrees")
    if blocked != {"sglang_decode_simulated_mtp"}:
        raise ValueError("MTP must be the only blocked held-out anchor")
    if scorable & blocked:
        raise ValueError("scorable and blocked anchors must be disjoint")


def validate_flagship_config(value: dict[str, Any]) -> None:
    """Validate the compact execution configuration without importing SGLang."""

    if value.get("schema") != FLAGSHIP_CONFIG_SCHEMA:
        raise ValueError("flagship configuration schema disagrees")
    if value["study"]["classification"] != "scored":
        raise ValueError("flagship configuration must be scored")
    live = value["live_session"]
    if (
        live["prefill_engines"],
        live["decode_engines"],
        live["simulated_gpus_per_engine"],
    ) != (1, 1, 8):
        raise ValueError("live session scale disagrees with the freeze")
    observations = value["exact_shape_observations"]
    ids = [row["anchor_id"] for row in observations]
    if ids != [
        "sglang_prefill_1k",
        "sglang_prefill_2k",
        "sglang_prefill_4k",
        "sglang_decode_standard",
    ]:
        raise ValueError("exact-shape observation order disagrees")
    if any(
        row["prompt_tokens"] * row["requests"] != 16_384
        for row in observations
        if row["pool"] == "prefill"
    ):
        raise ValueError("prefill observations must conserve 16384 tokens per rank")
    if live["context_length"] <= max(row["prompt_tokens"] for row in observations):
        raise ValueError("live context must preserve every prompt plus scheduler margin")
    for curve in value["publication_curves"]:
        loads = curve["offered_load_requests_per_second"]
        if loads != sorted(set(loads)):
            raise ValueError("curve loads must be unique and increasing")
        if any(PS_PER_SECOND % load for load in loads):
            raise ValueError("curve loads must map to exact picosecond intervals")


def _anchor_subset(
    anchor_freeze: dict[str, Any],
    allowed_ids: Iterable[str],
) -> dict[str, dict[str, Any]]:
    """Read anchor objects only after their IDs pass the caller's allowlist."""

    allowed = set(allowed_ids)
    selected: dict[str, dict[str, Any]] = {}
    for anchor in anchor_freeze["anchors"]:
        anchor_id = anchor.get("id")
        if anchor_id in allowed:
            selected[str(anchor_id)] = anchor
    if set(selected) != allowed:
        raise ValueError(f"anchor allowlist did not resolve exactly: {sorted(allowed)}")
    return selected


def prediction_at_surcharge(
    frozen_row: dict[str, Any],
    surcharge_ps: int,
    *,
    application_count: int = 116,
) -> Fraction:
    """Evaluate the frozen component projection at one physical surcharge."""

    if type(surcharge_ps) is not int or surcharge_ps < 0:
        raise ValueError("surcharge_ps must be a nonnegative integer")
    service_ps = int(frozen_row["candidate_service_ps"])
    per_node_tokens = int(frozen_row["per_node_tokens"])
    return Fraction(
        per_node_tokens * PS_PER_SECOND,
        service_ps + application_count * surcharge_ps,
    )


def prediction_interval(
    frozen_row: dict[str, Any],
    constant: dict[str, Any],
    point_surcharge_ps: int,
) -> dict[str, Any]:
    """Propagate the frozen record, zero distribution and constant envelope."""

    count = int(constant["application_count_per_step"])
    lower_constant = int(constant["envelope"]["lower"])
    upper_constant = int(constant["envelope"]["upper"])
    point = prediction_at_surcharge(
        frozen_row,
        point_surcharge_ps,
        application_count=count,
    )
    lower = prediction_at_surcharge(
        frozen_row,
        upper_constant,
        application_count=count,
    )
    upper = prediction_at_surcharge(
        frozen_row,
        lower_constant,
        application_count=count,
    )
    return {
        "lower": fraction_json(lower),
        "point": fraction_json(point),
        "upper": fraction_json(upper),
        "contributions": [
            {
                "source_kind": "candidate-record",
                "source_id": "ff46f6d8a79ddae899da89d4db6eb34373f8042acd06cab50b6336c8fb9a8f52",
                "point_service_ps": int(frozen_row["candidate_service_ps"]),
            },
            {
                "source_kind": "distribution",
                "source_id": "single-retained-seed-insufficient-replays",
                "relative_half_width": 0,
                "stability_claim": False,
            },
            {
                "source_kind": "constant-envelope",
                "source_id": constant["id"],
                "lower_ps": lower_constant,
                "selected_ps": point_surcharge_ps,
                "upper_ps": upper_constant,
                "application_count": count,
            },
        ],
    }


def _relative_error(predicted: Fraction, published: Fraction) -> Fraction:
    return abs(predicted - published) / published


def _objective(
    surcharge_ps: int,
    rows: dict[str, dict[str, Any]],
    targets: dict[str, Fraction],
    application_count: int,
) -> float:
    return math.fsum(
        float(
            _relative_error(
                prediction_at_surcharge(
                    rows[anchor_id],
                    surcharge_ps,
                    application_count=application_count,
                ),
                target,
            )
        )
        ** 2
        for anchor_id, target in targets.items()
    )


def _integer_minimum(
    lower: int,
    upper: int,
    objective: Callable[[int], float],
) -> int:
    """Find the deterministic integer minimum of the frozen unimodal objective."""

    while upper - lower > 24:
        third = (upper - lower) // 3
        left = lower + third
        right = upper - third
        if objective(left) <= objective(right):
            upper = right - 1
        else:
            lower = left + 1
    return min(range(lower, upper + 1), key=lambda value: (objective(value), value))


def fit_frozen_constant(
    anchor_freeze: dict[str, Any],
    scored_freeze: dict[str, Any],
) -> dict[str, Any]:
    """Fit only calibration anchors inside the preregistered envelope."""

    validate_scored_expectations(scored_freeze)
    fit_rule = scored_freeze["fit_rule"]
    visible = tuple(fit_rule["visible_anchor_ids"])
    calibration = _anchor_subset(anchor_freeze, visible)
    targets = {
        anchor_id: as_fraction(anchor["value"], f"{anchor_id}.value")
        for anchor_id, anchor in calibration.items()
    }
    rows = {
        row["anchor_id"]: row
        for row in scored_freeze["pre_tuning_predicted_bands"]
        if row["anchor_id"] in set(visible)
    }
    if set(rows) != set(visible):
        raise ValueError("calibration prediction rows disagree with the fit split")
    constant = scored_freeze["constants"]["tunable"][0]
    lower = int(constant["envelope"]["lower"])
    upper = int(constant["envelope"]["upper"])
    count = int(constant["application_count_per_step"])
    objective = lambda value: _objective(value, rows, targets, count)
    fitted = _integer_minimum(lower, upper, objective)
    calibration_rows = []
    for anchor_id in visible:
        predicted = prediction_at_surcharge(
            rows[anchor_id],
            fitted,
            application_count=count,
        )
        calibration_rows.append(
            {
                "anchor_id": anchor_id,
                "published": fraction_json(targets[anchor_id]),
                "predicted": fraction_json(predicted),
                "absolute_relative_error": fraction_json(
                    _relative_error(predicted, targets[anchor_id])
                ),
            }
        )
    return {
        "schema": FIT_SCHEMA,
        "status": "FROZEN",
        "constant_id": constant["id"],
        "fitted_ps": fitted,
        "envelope": dict(constant["envelope"]),
        "objective": fit_rule["objective"],
        "objective_value": objective(fitted),
        "accessed_anchor_ids": sorted(visible),
        "forbidden_anchor_ids_accessed": [],
        "calibration_rows": calibration_rows,
        "disposition": "fitted-parameter-not-measurement",
    }


def score_frozen_fit(
    anchor_freeze: dict[str, Any],
    scored_freeze: dict[str, Any],
    fit: dict[str, Any],
) -> dict[str, Any]:
    """Read and score only priced held-out anchors after the fit is frozen."""

    if fit.get("schema") != FIT_SCHEMA or fit.get("status") != "FROZEN":
        raise ValueError("held-out scoring requires a frozen fit artifact")
    rule = scored_freeze["scoring_rule"]
    scorable_ids = tuple(rule["scorable_held_out_anchor_ids"])
    held_out = _anchor_subset(anchor_freeze, scorable_ids)
    prediction_rows = {
        row["anchor_id"]: row
        for row in scored_freeze["pre_tuning_predicted_bands"]
        if row["anchor_id"] in set(scorable_ids)
    }
    constant = scored_freeze["constants"]["tunable"][0]
    bar = as_fraction(rule["maximum_absolute_relative_error"], "score.bar")
    rows = []
    for anchor_id in scorable_ids:
        published = as_fraction(held_out[anchor_id]["value"], f"{anchor_id}.value")
        interval = prediction_interval(
            prediction_rows[anchor_id],
            constant,
            int(fit["fitted_ps"]),
        )
        point = as_fraction(interval["point"], f"{anchor_id}.point")
        error = _relative_error(point, published)
        rows.append(
            {
                "anchor_id": anchor_id,
                "published": fraction_json(published),
                "predicted": interval,
                "absolute_relative_error": fraction_json(error),
                "point_passes_5_percent": error <= bar,
            }
        )
    maximum = max(
        as_fraction(row["absolute_relative_error"], "score.error") for row in rows
    )
    status = "PASS" if maximum <= bar else "REFUTED"
    return {
        "schema": SCORE_SCHEMA,
        "status": status,
        "scope": "priced held-out prefill anchors only",
        "maximum_absolute_relative_error": fraction_json(maximum),
        "acceptance_bar": fraction_json(bar),
        "accessed_anchor_ids": sorted(scorable_ids),
        "forbidden_anchor_ids_accessed": [],
        "rows": rows,
        "blocked_rows": [
            {
                "anchor_id": "sglang_decode_simulated_mtp",
                "status": "BLOCKED",
                "published": None,
                "prediction": None,
                "reason": "candidate record has no EP72 MTP batch-16 KV-4000 cell",
                "dependency": "COMP-72 resumable Merlin execution",
            }
        ],
    }


def _queue_point(
    offered_requests_per_second: int,
    output_tokens_per_request: int,
    capacity: Fraction,
    base_delay_ps: int,
) -> tuple[Fraction, Fraction]:
    offered_tokens = Fraction(
        offered_requests_per_second * output_tokens_per_request
    )
    throughput = min(offered_tokens, capacity)
    utilization = min(offered_tokens / capacity, Fraction(19, 20))
    delay = Fraction(base_delay_ps, 1) / (1 - utilization)
    return throughput, delay


def build_publication_curve(
    curve_config: dict[str, Any],
    capacity_interval_per_node: dict[str, Any],
) -> dict[str, Any]:
    """Build one exact role-scaled queue curve in the shared curve schema."""

    nodes = int(curve_config["target_nodes"])
    capacities = {
        name: as_fraction(capacity_interval_per_node[name], f"capacity.{name}") * nodes
        for name in ("lower", "point", "upper")
    }
    output_tokens = int(curve_config["output_tokens_per_request"])
    base_delay = int(curve_config["base_per_token_delay_ps"])
    points = []
    for load in curve_config["offered_load_requests_per_second"]:
        x = {}
        delay = {}
        inverse = {}
        for name in ("lower", "point", "upper"):
            throughput, per_token_delay = _queue_point(
                int(load),
                output_tokens,
                capacities[name],
                base_delay,
            )
            x[name] = fraction_json(throughput)
            delay[name] = fraction_json(per_token_delay)
            inverse[name] = fraction_json(Fraction(PS_PER_SECOND, 1) / per_token_delay)
        point_throughput = as_fraction(x["point"], "point.throughput")
        output_token_count = point_throughput.numerator
        request_count = math.ceil(output_token_count / output_tokens)
        observation_duration_ps = point_throughput.denominator * PS_PER_SECOND
        points.append(
            {
                "schema": POINT_SCHEMA,
                "offered_load_requests_per_second": fraction_json(Fraction(load)),
                "aggregated_output_throughput_tokens_per_second": x["point"],
                "per_token_request_delay_ps": delay["point"],
                "request_count": request_count,
                "output_token_count": output_token_count,
                "first_admitted_at_ps": 0,
                "last_completed_at_ps": observation_duration_ps,
                "point_kind": "analytic-capacity-projection",
                "uncertainty": {
                    "schema": "simllm-deployment-curve-point-interval-v1",
                    "method": "deterministic-additive-interval-v1",
                    "aggregated_output_throughput_tokens_per_second": x,
                    "per_token_request_delay_ps": delay,
                    "inverse_per_token_request_delay_tokens_per_second": inverse,
                    "source_contributions": capacity_interval_per_node[
                        "contributions"
                    ],
                },
            }
        )
    return {
        "schema": CURVE_SCHEMA,
        "configuration_id": curve_config["configuration_id"],
        "configuration_label": curve_config["configuration_label"],
        "prefill_engines": 0,
        "decode_engines": nodes,
        "prompt_tokens": 2000,
        "orientation": {
            "x": "aggregated-output-throughput-rightward",
            "y": "inverse-per-token-request-delay-upward",
        },
        "evidence_class": curve_config["evidence_class"],
        "target_nodes": nodes,
        "points": points,
    }


def stable_request_projection(value: dict[str, Any]) -> dict[str, Any]:
    """Project one request onto the preregistered cross-run stable field set."""

    timeline_fields = (
        "request_id",
        "admitted_at_ps",
        "prefill_eligible_at_ps",
        "prefill_completed_at_ps",
        "handoff",
        "decode_eligible_at_ps",
        "decode_token_completed_at_ps",
    )
    stable = {name: value[name] for name in timeline_fields}
    stable.update(
        {
            name: value[name]
            for name in (
                "prefill_engine_id",
                "decode_engine_id",
                "bootstrap_token_id",
                "decode_token_ids",
                "prefill_step_count",
                "decode_step_count",
            )
        }
    )
    join = dict(value["join_metadata"])
    join.pop("prefill_process_id", None)
    join.pop("decode_process_id", None)
    stable["join_metadata"] = join
    if "compute_pricing" in value:
        stable["compute_pricing"] = value["compute_pricing"]
    return stable
