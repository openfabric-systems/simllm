"""Repeat-derived distribution propagation for the CORE-54 flagship bands."""

from __future__ import annotations

import csv
import hashlib
import json
from copy import deepcopy
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any

from curve_tools import (
    add_symmetric_relative_spread,
    as_fraction,
    fraction_json,
)
from flagship_tools import build_publication_curve

EXPECTATIONS_SCHEMA = "simllm-deployment-curve-comp74-expectations-v1"
RESULT_SCHEMA = "simllm-deployment-curve-comp74-distribution-result-v1"
LAYER_NAMES = (
    "physics_only",
    "physics_plus_boundary",
    "physics_plus_boundary_plus_attenuation",
)
ZERO_WIDTH_SOURCE = "comp74-zero-width-insufficient-replays"


def file_sha256(path: Path) -> str:
    """Return a streaming SHA-256 without loading the whole file."""

    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    """Write deterministic UTF-8 JSON with explicit LF newlines."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def validate_expectations(frozen: dict[str, Any]) -> None:
    """Validate the committed statistic, access and no-pooling contract."""

    if frozen.get("schema") != EXPECTATIONS_SCHEMA:
        raise ValueError("COMP-74 expectations schema differs")
    if frozen.get("status") != "EXPECTATIONS_ONLY":
        raise ValueError("COMP-74 freeze must remain expectations only")
    chronology = frozen["chronology"]
    for name in (
        "additional_retained_repetition_values_accessed",
        "comp74_result_existed_before_freeze",
        "comp74_runner_existed_before_freeze",
        "distribution_intervals_computed",
        "published_throughput_values_used",
    ):
        if chronology[name]:
            raise ValueError(f"COMP-74 chronology flag {name} differs")
    if not chronology["field_reader_must_be_committed_before_record_access"]:
        raise ValueError("COMP-74 field reader chronology differs")
    pooling = frozen["estimation_rule"]["pooling"]
    if any(pooling.values()):
        raise ValueError("COMP-74 forbids every registered pooling dimension")
    mappings = frozen["key_mapping"]
    if len(mappings) != 4:
        raise ValueError("COMP-74 requires exactly four priced key mappings")
    if len({row["implementation_suffix"] for row in mappings}) != 4:
        raise ValueError("COMP-74 implementation suffixes must be unique")
    if len({row["anchor_id"] for row in mappings}) != 4:
        raise ValueError("COMP-74 anchor mappings must be unique")


def verify_source_digests(
    repository_root: Path,
    frozen: dict[str, Any],
) -> list[dict[str, str]]:
    """Verify frozen source identities without decoding their records."""

    validate_expectations(frozen)
    checked = []
    for item in frozen["sources"]:
        relative = PurePosixPath(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("COMP-74 source paths must be repository-relative")
        path = repository_root.joinpath(*relative.parts)
        actual = file_sha256(path)
        if actual != item["sha256"]:
            raise ValueError(f"COMP-74 source digest differs for {item['path']}")
        checked.append({"path": item["path"], "sha256": actual})
    return checked


def verify_preservation_lock(
    repository_root: Path,
    frozen: dict[str, Any],
) -> list[dict[str, str]]:
    """Verify every prior scored publication artifact byte for byte."""

    validate_expectations(frozen)
    checked = []
    for item in frozen["preservation_lock"]["artifacts"]:
        relative = PurePosixPath(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("COMP-74 preservation paths must be repository-relative")
        path = repository_root.joinpath(*relative.parts)
        actual = file_sha256(path)
        if actual != item["sha256"]:
            raise ValueError(f"COMP-74 preservation digest differs for {item['path']}")
        checked.append({"path": item["path"], "sha256": actual})
    return checked


def estimate_key_intervals(
    frozen: dict[str, Any],
    successor: dict[str, Any],
) -> list[dict[str, Any]]:
    """Estimate the exact frozen repeat statistic independently for each key."""

    validate_expectations(frozen)
    invariants = frozen["evidence_invariants"]
    if successor.get("acceptance_status") != invariants[
        "lookup_record_acceptance_status"
    ]:
        raise ValueError("COMP-74 successor candidate status differs")
    score = successor["score"]
    if score["task_movement"]["comp74_repeat_inputs"] != (
        "RETAINED_FOR_ALL_FOUR_PRICED_KEYS"
    ):
        raise ValueError("COMP-74 repeat-input movement differs")
    observations = {
        row["implementation_suffix"]: row
        for row in score["priced_repeat_observations"]
    }
    expected = {row["implementation_suffix"] for row in frozen["key_mapping"]}
    if set(observations) != expected or len(observations) != len(
        score["priced_repeat_observations"]
    ):
        raise ValueError("COMP-74 retained repeat key set differs")

    intervals = []
    for mapping in frozen["key_mapping"]:
        row = observations[mapping["implementation_suffix"]]
        point = int(row["published_point_ps"])
        repeat = int(row["independent_repeat_ps"])
        if point <= 0 or repeat <= 0:
            raise ValueError("COMP-74 service observations must be positive")
        if int(row["retained_independent_observations"]) < 2:
            raise ValueError("COMP-74 priced keys require two independent observations")
        signed = repeat - point
        if int(row["signed_repeat_minus_point_ps"]) != signed:
            raise ValueError("COMP-74 signed repeat movement differs")
        if row["distribution_propagation"] != "DEFERRED_TO_COMP-74":
            raise ValueError("COMP-74 source propagation marker differs")
        delta = abs(signed)
        relative = Fraction(delta, point)
        service_lower = point - delta
        service_upper = point + delta
        if service_lower <= 0:
            raise ValueError("COMP-74 service envelope crossed zero")
        if not service_lower <= min(point, repeat) <= max(point, repeat) <= service_upper:
            raise ValueError("COMP-74 observed-repeat envelope lost an observation")
        intervals.append(
            {
                **mapping,
                "observation_count": int(row["retained_independent_observations"]),
                "published_point_ps": point,
                "independent_repeat_ps": repeat,
                "signed_repeat_minus_point_ps": signed,
                "absolute_repeat_deviation_ps": delta,
                "relative_half_width": fraction_json(relative),
                "relative_half_width_percent": 100 * float(relative),
                "service_interval_ps": {
                    "lower": service_lower,
                    "point": point,
                    "upper": service_upper,
                },
                "observed_min_ps": min(point, repeat),
                "observed_max_ps": max(point, repeat),
                "nonzero": delta != 0,
                "stability_claim": False,
            }
        )
    return intervals


def propagate_prediction_interval(
    interval: dict[str, Any],
    estimate: dict[str, Any],
    *,
    enabled: bool,
) -> dict[str, Any]:
    """Add one repeat-derived spread through the shared interval engine."""

    inherited = deepcopy(interval)
    if not enabled:
        return inherited
    lower = as_fraction(inherited["lower"], "prediction.lower")
    point = as_fraction(inherited["point"], "prediction.point")
    upper = as_fraction(inherited["upper"], "prediction.upper")
    lower, upper, delta = add_symmetric_relative_spread(
        lower,
        point,
        upper,
        estimate["relative_half_width"],
        f"{estimate['anchor_id']}.distribution",
    )
    inherited.update(
        {
            "lower": fraction_json(lower),
            "point": fraction_json(point),
            "upper": fraction_json(upper),
        }
    )
    replacement = {
        "source_kind": "distribution",
        "source_id": (
            f"comp74-repeat-envelope:{estimate['implementation_suffix']}"
        ),
        "relative_half_width": estimate["relative_half_width"],
        "prediction_delta": {
            "lower": fraction_json(-delta),
            "upper": fraction_json(delta),
        },
        "observation_count": estimate["observation_count"],
        "service_interval_ps": estimate["service_interval_ps"],
        "stability_claim": False,
    }
    contributions = []
    replaced = False
    for contribution in inherited.get("contributions", []):
        if (
            contribution.get("source_kind") == "distribution"
            and contribution.get("source_id") == ZERO_WIDTH_SOURCE
        ):
            contributions.append(replacement)
            replaced = True
        else:
            contributions.append(contribution)
    if not replaced:
        contributions.append(replacement)
    inherited["contributions"] = contributions
    return inherited


def _contact(interval: dict[str, Any], boundary: Fraction) -> bool:
    lower = as_fraction(interval["lower"], "contact.lower")
    upper = as_fraction(interval["upper"], "contact.upper")
    return lower <= boundary <= upper


def _movement_row(
    anchor_id: str,
    layer: str,
    published: object,
    bar: object,
    prior: dict[str, Any],
    propagated: dict[str, Any],
    frozen_verdict: str,
    distribution_status: str,
    relative_half_width: object,
) -> dict[str, Any]:
    published_fraction = as_fraction(published, f"{anchor_id}.published")
    bar_fraction = as_fraction(bar, f"{anchor_id}.bar")
    accepted_lower = published_fraction * (1 - bar_fraction)
    accepted_upper = published_fraction * (1 + bar_fraction)
    prior_lower = as_fraction(prior["lower"], f"{anchor_id}.prior.lower")
    prior_upper = as_fraction(prior["upper"], f"{anchor_id}.prior.upper")
    new_lower = as_fraction(propagated["lower"], f"{anchor_id}.new.lower")
    new_upper = as_fraction(propagated["upper"], f"{anchor_id}.new.upper")
    prior_lower_contact = _contact(prior, accepted_lower)
    prior_upper_contact = _contact(prior, accepted_upper)
    new_lower_contact = _contact(propagated, accepted_lower)
    new_upper_contact = _contact(propagated, accepted_upper)
    return {
        "anchor_id": anchor_id,
        "layer": layer,
        "frozen_point_verdict": frozen_verdict,
        "distribution_status": distribution_status,
        "relative_half_width": relative_half_width,
        "prior": deepcopy(prior),
        "propagated": deepcopy(propagated),
        "movement": {
            "lower": fraction_json(new_lower - prior_lower),
            "upper": fraction_json(new_upper - prior_upper),
        },
        "acceptance_interval": {
            "lower": fraction_json(accepted_lower),
            "upper": fraction_json(accepted_upper),
        },
        "prior_touches_lower_boundary": prior_lower_contact,
        "prior_touches_upper_boundary": prior_upper_contact,
        "propagated_touches_lower_boundary": new_lower_contact,
        "propagated_touches_upper_boundary": new_upper_contact,
        "new_boundary_contact": (
            (new_lower_contact and not prior_lower_contact)
            or (new_upper_contact and not prior_upper_contact)
        ),
        "propagated_intersects_acceptance_interval": (
            new_lower <= accepted_upper and new_upper >= accepted_lower
        ),
        "rescore_performed": False,
    }


def propagate_anchor_bands(
    run4: dict[str, Any],
    estimates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Propagate all priced key bands and retain MTP as single-seed context."""

    by_anchor = {row["anchor_id"]: row for row in estimates}
    inherited = run4["run3_carry_forward"]["anchor_predictions"]
    propagated_rows = deepcopy(inherited)
    off_proof = []
    movement = []
    bar = run4["run3_carry_forward"]["held_out_score"]["acceptance_bar"]
    held_out = {
        row["anchor_id"]: row
        for row in run4["run3_carry_forward"]["held_out_score"]["rows"]
    }
    for prior_row, propagated_row in zip(inherited, propagated_rows, strict=True):
        anchor_id = prior_row["anchor_id"]
        if prior_row.get("status") != "PREDICTED":
            continue
        estimate = by_anchor[anchor_id]
        frozen_verdict = (
            "PASS"
            if anchor_id in held_out
            and held_out[anchor_id]["layers"]
            ["physics_plus_boundary_plus_attenuation"]["point_passes_5_percent"]
            else "CALIBRATION_CONTEXT"
        )
        for layer in LAYER_NAMES:
            prior_interval = prior_row["layers"][layer]["prediction"]
            off_interval = propagate_prediction_interval(
                prior_interval,
                estimate,
                enabled=False,
            )
            new_interval = propagate_prediction_interval(
                prior_interval,
                estimate,
                enabled=True,
            )
            propagated_row["layers"][layer]["prediction"] = new_interval
            exact = off_interval == prior_interval
            off_proof.append(
                {
                    "anchor_id": anchor_id,
                    "layer": layer,
                    "prior_point": prior_interval["point"],
                    "distribution_off_point": off_interval["point"],
                    "interval_object_equal": exact,
                    "point_exact": off_interval["point"] == prior_interval["point"],
                }
            )
            movement.append(
                _movement_row(
                    anchor_id,
                    layer,
                    prior_row["published"],
                    bar,
                    prior_interval,
                    new_interval,
                    frozen_verdict,
                    "REPEAT_DERIVED",
                    estimate["relative_half_width"],
                )
            )

    mtp = run4["mtp_score"]
    for layer in LAYER_NAMES:
        prior_interval = mtp["layers"][layer]["prediction"]
        off_proof.append(
            {
                "anchor_id": mtp["anchor_id"],
                "layer": layer,
                "prior_point": prior_interval["point"],
                "distribution_off_point": deepcopy(prior_interval["point"]),
                "interval_object_equal": True,
                "point_exact": True,
            }
        )
        movement.append(
            _movement_row(
                mtp["anchor_id"],
                layer,
                mtp["published"],
                mtp["acceptance_bar"],
                prior_interval,
                prior_interval,
                mtp["layers"][layer]["status"],
                "SINGLE_SEED_NOT_PROPAGATED",
                0,
            )
        )
    return propagated_rows, movement, off_proof


def propagate_flagship_curves(
    run4: dict[str, Any],
    curve_config: dict[str, Any],
    propagated_anchors: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Rebuild the flagship curves through the inherited capacity engine."""

    prior_anchor = next(
        row
        for row in run4["run3_carry_forward"]["anchor_predictions"]
        if row["anchor_id"] == "sglang_decode_standard"
    )
    new_anchor = next(
        row
        for row in propagated_anchors
        if row["anchor_id"] == "sglang_decode_standard"
    )
    layer = "physics_plus_boundary_plus_attenuation"
    prior_capacity = prior_anchor["layers"][layer]["prediction"]
    new_capacity = new_anchor["layers"][layer]["prediction"]
    rebuilt_prior = [
        build_publication_curve(config, prior_capacity)
        for config in curve_config["publication_curves"]
    ]
    stored = run4["run3_carry_forward"]["curves"]
    if rebuilt_prior != stored:
        raise ValueError("distribution-OFF curve reproduction differs")
    propagated = [
        build_publication_curve(config, new_capacity)
        for config in curve_config["publication_curves"]
    ]
    proof = [
        {
            "configuration_id": prior["configuration_id"],
            "point_count": len(prior["points"]),
            "stored_curve_equal": prior == stored_curve,
        }
        for prior, stored_curve in zip(rebuilt_prior, stored, strict=True)
    ]
    movement = []
    for prior_curve, new_curve in zip(stored, propagated, strict=True):
        for prior_point, new_point in zip(
            prior_curve["points"], new_curve["points"], strict=True
        ):
            movement.append(
                {
                    "configuration_id": prior_curve["configuration_id"],
                    "offered_load_requests_per_second": prior_point[
                        "offered_load_requests_per_second"
                    ],
                    "prior_throughput_interval": prior_point["uncertainty"][
                        "aggregated_output_throughput_tokens_per_second"
                    ],
                    "propagated_throughput_interval": new_point["uncertainty"][
                        "aggregated_output_throughput_tokens_per_second"
                    ],
                    "prior_inverse_delay_interval": prior_point["uncertainty"][
                        "inverse_per_token_request_delay_tokens_per_second"
                    ],
                    "propagated_inverse_delay_interval": new_point["uncertainty"][
                        "inverse_per_token_request_delay_tokens_per_second"
                    ],
                    "point_predictions_equal": (
                        prior_point["aggregated_output_throughput_tokens_per_second"]
                        == new_point["aggregated_output_throughput_tokens_per_second"]
                        and prior_point["per_token_request_delay_ps"]
                        == new_point["per_token_request_delay_ps"]
                    ),
                }
            )
    return propagated, movement, proof


def build_result(
    frozen: dict[str, Any],
    successor: dict[str, Any],
    run4: dict[str, Any],
    curve_config: dict[str, Any],
    source_checks: list[dict[str, str]],
    preservation: list[dict[str, str]],
    access: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the complete COMP-74 result without rescoring any point."""

    estimates = estimate_key_intervals(frozen, successor)
    anchors, band_movement, off_proof = propagate_anchor_bands(run4, estimates)
    curves, curve_movement, curve_proof = propagate_flagship_curves(
        run4,
        curve_config,
        anchors,
    )
    all_points_exact = all(row["point_exact"] for row in off_proof)
    all_intervals_equal = all(row["interval_object_equal"] for row in off_proof)
    all_curves_equal = all(row["stored_curve_equal"] for row in curve_proof)
    if not (all_points_exact and all_intervals_equal and all_curves_equal):
        raise ValueError("COMP-74 distribution-OFF reproduction failed")
    if any(not row["nonzero"] for row in estimates):
        raise ValueError("COMP-74 varying retained observations require nonzero intervals")
    if successor["acceptance_status"] != "candidate":
        raise ValueError("COMP-74 candidate status changed")
    if run4["verdict"] != "ALL_SCORABLE_HELD_OUT_REFUTED":
        raise ValueError("COMP-74 inherited run-4 verdict differs")
    if any(row.get("rescore_performed") for row in band_movement):
        raise ValueError("COMP-74 must never rescore")
    new_contacts = [row for row in band_movement if row["new_boundary_contact"]]
    return {
        "schema": RESULT_SCHEMA,
        "status": "PASS",
        "task_id": "COMP-74",
        "classification": "REPEAT_DERIVED_DISTRIBUTION_PROPAGATION",
        "statistic": deepcopy(frozen["estimation_rule"]),
        "interval_method": frozen["propagation_rule"]["method"],
        "key_intervals": estimates,
        "distribution_off_reproduction": {
            "status": "PASS",
            "current_point_prediction_count": len(off_proof),
            "all_points_exact": all_points_exact,
            "all_interval_objects_equal": all_intervals_equal,
            "anchor_layers": off_proof,
            "all_stored_curves_equal": all_curves_equal,
            "curves": curve_proof,
        },
        "band_movement": {
            "rows": band_movement,
            "new_boundary_contact_count": len(new_contacts),
            "new_boundary_contacts": [
                {"anchor_id": row["anchor_id"], "layer": row["layer"]}
                for row in new_contacts
            ],
            "rescore_performed": False,
        },
        "propagated_anchor_predictions": anchors,
        "propagated_curves": curves,
        "curve_band_movement": curve_movement,
        "verdicts": {
            "run3_prefill": run4["run3_carry_forward"]["held_out_score"]["status"],
            "run4_mtp": run4["mtp_score"]["status"],
            "combined": run4["verdict"],
            "changed": False,
            "rule": "band widening never flips or recomputes a frozen point verdict",
        },
        "single_seed_keys": [
            {
                "anchor_id": "sglang_decode_simulated_mtp",
                "status": "SINGLE_SEED_NOT_PROPAGATED",
                "reason": "pooling standard decode into simulated MTP is forbidden",
                "residual_id": "COMP-79",
            }
        ],
        "evidence": {
            "lookup_record_sha256": successor["lookup_record_sha256"],
            "predecessor_lookup_record_sha256": successor[
                "predecessor_lookup_record_sha256"
            ],
            "acceptance_status_before": successor["acceptance_status"],
            "acceptance_status_after": successor["acceptance_status"],
            "candidate_promotion_performed": False,
            "lookup_service_ledger_before": successor["score"][
                "lookup_service_ledger"
            ],
            "lookup_service_ledger_after": deepcopy(
                successor["score"]["lookup_service_ledger"]
            ),
            "ledger_equal": True,
        },
        "preservation_lock": {
            "status": "PASS",
            "class": frozen["preservation_lock"]["class"],
            "artifacts": preservation,
        },
        "source_checks": source_checks,
        "access": {
            "whole_record_loaded": False,
            "rows": access,
            "successful_projection_count": sum(
                row.get("status") == "PASS" for row in access
            ),
        },
        "closure": {
            "comp74": "CLOSED_LITERAL",
            "priced_key_count": len(estimates),
            "all_priced_keys_have_two_observations": all(
                row["observation_count"] >= 2 for row in estimates
            ),
            "all_varying_keys_have_nonzero_intervals": all(
                row["nonzero"] for row in estimates
            ),
            "residuals_registered": deepcopy(frozen["closure_rule"]["residuals"]),
        },
        "provenance": {
            "published_throughput_values_used_for_distribution": False,
            "model_weights_loaded_or_downloaded": False,
            "web_pages_fetched": False,
            "merlin_submissions_performed": False,
            "traffic_module_touched": False,
        },
    }


def write_key_table(path: Path, result: dict[str, Any]) -> None:
    """Write the compact per-key repeat interval table with LF newlines."""

    columns = (
        "anchor_id",
        "implementation_suffix",
        "published_point_ps",
        "independent_repeat_ps",
        "signed_repeat_minus_point_ps",
        "service_lower_ps",
        "service_upper_ps",
        "relative_half_width_fraction",
        "relative_half_width_percent",
        "observation_count",
        "candidate_status",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in result["key_intervals"]:
            relative = as_fraction(row["relative_half_width"], "relative")
            writer.writerow(
                {
                    "anchor_id": row["anchor_id"],
                    "implementation_suffix": row["implementation_suffix"],
                    "published_point_ps": row["published_point_ps"],
                    "independent_repeat_ps": row["independent_repeat_ps"],
                    "signed_repeat_minus_point_ps": row[
                        "signed_repeat_minus_point_ps"
                    ],
                    "service_lower_ps": row["service_interval_ps"]["lower"],
                    "service_upper_ps": row["service_interval_ps"]["upper"],
                    "relative_half_width_fraction": (
                        f"{relative.numerator}/{relative.denominator}"
                    ),
                    "relative_half_width_percent": (
                        f"{row['relative_half_width_percent']:.9f}"
                    ),
                    "observation_count": row["observation_count"],
                    "candidate_status": result["evidence"][
                        "acceptance_status_after"
                    ],
                }
            )


def _fraction_text(value: object, name: str) -> str:
    fraction = as_fraction(value, name)
    return f"{fraction.numerator}/{fraction.denominator}"


def write_band_table(path: Path, result: dict[str, Any]) -> None:
    """Write the required per-anchor band movement artifact."""

    columns = (
        "anchor_id",
        "layer",
        "frozen_point_verdict",
        "distribution_status",
        "prior_lower_fraction",
        "point_fraction",
        "prior_upper_fraction",
        "propagated_lower_fraction",
        "propagated_upper_fraction",
        "new_boundary_contact",
        "intersects_acceptance_interval",
        "rescore_performed",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in result["band_movement"]["rows"]:
            writer.writerow(
                {
                    "anchor_id": row["anchor_id"],
                    "layer": row["layer"],
                    "frozen_point_verdict": row["frozen_point_verdict"],
                    "distribution_status": row["distribution_status"],
                    "prior_lower_fraction": _fraction_text(
                        row["prior"]["lower"], "prior.lower"
                    ),
                    "point_fraction": _fraction_text(
                        row["prior"]["point"], "prior.point"
                    ),
                    "prior_upper_fraction": _fraction_text(
                        row["prior"]["upper"], "prior.upper"
                    ),
                    "propagated_lower_fraction": _fraction_text(
                        row["propagated"]["lower"], "propagated.lower"
                    ),
                    "propagated_upper_fraction": _fraction_text(
                        row["propagated"]["upper"], "propagated.upper"
                    ),
                    "new_boundary_contact": row["new_boundary_contact"],
                    "intersects_acceptance_interval": row[
                        "propagated_intersects_acceptance_interval"
                    ],
                    "rescore_performed": row["rescore_performed"],
                }
            )
