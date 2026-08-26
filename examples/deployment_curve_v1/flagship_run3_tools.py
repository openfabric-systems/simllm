"""Pure fitting, layered scoring and curve construction for CORE-54 run three."""

from __future__ import annotations

import hashlib
import json
import math
from fractions import Fraction
from math import comb
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

from curve_tools import as_fraction, fraction_json
from flagship_tools import (
    build_publication_curve,
    load_json,
    sha256,
    stable_request_projection,
    write_json,
)

EXPECTATIONS_SCHEMA = "simllm-deployment-curve-scored-run3-expectations-v1"
CONFIG_SCHEMA = "simllm-deployment-curve-flagship-run2-config-v1"
FIT_SCHEMA = "simllm-deployment-curve-flagship-run3-fit-v1"
SCORE_SCHEMA = "simllm-deployment-curve-flagship-run3-score-v1"
ACCESS_SCHEMA = "simllm-deployment-curve-run3-anchor-access-v1"
ANCHOR_SCHEMA = "simllm-deployment-curve-anchor-freeze-v1"
ANCHOR_SHA256 = "b1a918ed02329a242d033943fb18b93fd9be8fdaa18093477e6abb8298540df5"
RUN2_CONFIG_SHA256 = "3e5cca6693be05d9bd93870158ee24f7bee9092c2ce981c287fd94765d2d1970"
PS_PER_SECOND = 1_000_000_000_000

ANCHOR_SPANS = {
    "sglang_prefill_1k": (4862, 344, "calibration"),
    "sglang_prefill_2k": (5212, 341, "held_out"),
    "sglang_prefill_4k": (5559, 341, "held_out"),
    "sglang_decode_standard": (5906, 457, "calibration"),
}


def _fraction(value: object, name: str) -> Fraction:
    return as_fraction(value, name)


def _factor_interval(frozen: dict[str, Any]) -> dict[str, Fraction]:
    row = frozen["attenuation_layer"]["factors"][0]["uncertainty"]["interval"]
    return {edge: _fraction(row[edge], f"attenuation.{edge}") for edge in row}


def validate_expectations(value: dict[str, Any]) -> None:
    """Validate chronology, envelopes, attenuation and the scoring boundary."""

    if value.get("schema") != EXPECTATIONS_SCHEMA:
        raise ValueError("third-run expectations schema disagrees")
    if value.get("status") != "EXPECTATIONS_ONLY":
        raise ValueError("third-run expectations must remain expectations only")
    chronology = value["chronology"]
    for name in (
        "third_scored_runner_existed_before_this_freeze",
        "third_fitted_constants_existed_before_this_freeze",
        "third_held_out_score_existed_before_this_freeze",
        "third_flagship_figure_existed_before_this_freeze",
        "held_out_anchor_numeric_values_accessed",
        "web_pages_fetched",
        "model_weights_loaded_or_downloaded",
    ):
        if chronology[name]:
            raise ValueError(f"third-run chronology flag {name} must be false")
    allocation = value["inherited_rulings"]["allocation"]
    if (
        allocation["prefill_experiment"]["simultaneous_with_decode_experiment"]
        or allocation["decode_experiment"]["simultaneous_with_prefill_experiment"]
        or allocation["structural_comparator_only"]["may_be_called_96_gpu_system"]
    ):
        raise ValueError("the separate-experiment ruling drifted")

    constants = {row["id"]: row for row in value["constants"]["tunable"]}
    if set(constants) != {
        "intra_node_collective_surcharge_ps",
        "overlap_exposed_fraction",
    }:
        raise ValueError("third-run tunable inventory disagrees")
    if value["constants"]["new_unbounded_or_free"]:
        raise ValueError("third-run constants must remain physically bounded")
    exposed = constants["overlap_exposed_fraction"]
    if (
        _fraction(exposed["envelope"]["lower"], "f.lower") != 0
        or _fraction(exposed["envelope"]["upper"], "f.upper") != Fraction(1, 2)
    ):
        raise ValueError("overlap exposure must retain the clean [0, 1/2] bracket")

    prefill = value["pricing_configuration"]["prefill"]
    if prefill["physics_only_operator"] != "max(C, P)":
        raise ValueError("physics-only composition drifted")
    if prefill["boundary_operator"] != "max(C, P) + f * min(C, P)":
        raise ValueError("boundary composition drifted")
    refinements = {
        row["id"]: row
        for row in value["pricing_configuration"]["derived_refinements"]
    }
    locality = refinements["ep32_rank_layout_locality_split"]
    if (
        locality["status"] != "FROZEN_DERIVED"
        or locality["fit_allowed"]
        or locality["anchor_numeric_input_count"]
        or locality["topology"]["same_node_destination_peers"] != 7
        or locality["topology"]["fabric_destination_peers"] != 24
    ):
        raise ValueError("the topology-derived locality split drifted")
    if refinements["a100_three_module_packet_candidate_substitution"]["status"] != (
        "REJECTED"
    ):
        raise ValueError("the cross-architecture packet candidate must stay rejected")

    attenuation = value["attenuation_layer"]
    if attenuation["admitted_factor_count"] != 1 or len(attenuation["factors"]) != 1:
        raise ValueError("exactly one attenuation factor must be admitted")
    factor = attenuation["factors"][0]
    touched = factor["applies_to_anchor_ids"]
    if set(touched) != {
        "sglang_prefill_1k",
        "sglang_prefill_2k",
        "sglang_prefill_4k",
    }:
        raise ValueError("routing attenuation anchor scope drifted")
    if factor["anchor_numeric_input_count"] or len(attenuation["factors"]) >= len(touched):
        raise ValueError("attenuation independence or non-vacuity drifted")
    magnitude = factor["magnitude"]
    experts = int(magnitude["logical_experts"])
    per_rank = int(magnitude["experts_per_rank"])
    top_k = int(magnitude["top_k"])
    ranks = int(magnitude["expert_parallel_ranks"])
    probability = Fraction(
        comb(experts, top_k) - comb(experts - per_rank, top_k),
        comb(experts, top_k),
    )
    expected_factor = ranks * probability / top_k
    if _fraction(magnitude["factor"], "attenuation.factor") != expected_factor:
        raise ValueError("routing attenuation magnitude disagrees")
    interval = _factor_interval(value)
    if not interval["lower"] <= interval["point"] <= interval["upper"]:
        raise ValueError("routing attenuation interval is unordered")
    rejected = {row["id"]: row for row in attenuation["rejected_candidates"]}
    if rejected["exact_length_packing_vs_per_request_overhead"]["factor"] is not None:
        raise ValueError("exact-length packing cannot gain a fitted factor")
    if rejected["decode_depth_attenuation"]["status"] != (
        "FORBIDDEN_BY_POLICY_RULE_FIVE"
    ):
        raise ValueError("decode attenuation must remain forbidden")

    scorable = set(value["scoring_rule"]["scorable_held_out_anchor_ids"])
    blocked = set(value["scoring_rule"]["blocked_held_out_anchor_ids"])
    if scorable != {"sglang_prefill_2k", "sglang_prefill_4k"}:
        raise ValueError("the priced held-out set disagrees")
    if blocked != {"sglang_decode_simulated_mtp"}:
        raise ValueError("MTP must remain the only blocked held-out anchor")
    if value["pricing_configuration"]["decode"]["attenuation_allowed"]:
        raise ValueError("decode cannot receive benchmark-bias attenuation")


def validate_execution_config(config: dict[str, Any], frozen: dict[str, Any]) -> None:
    """Validate that the inherited second-run configuration is unchanged."""

    validate_expectations(frozen)
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("inherited execution configuration schema disagrees")
    if config["study"]["classification"] != "scored":
        raise ValueError("the inherited configuration must remain scored")
    live = config["live_session"]
    if (
        live["prefill_engines"],
        live["decode_engines"],
        live["simulated_gpus_per_engine"],
    ) != (1, 1, 8):
        raise ValueError("the inherited live scale disagrees")
    if live["project_remote_kv_length"] is not True:
        raise ValueError("remote-KV projection must remain enabled")
    observations = config["exact_shape_observations"]
    if [row["anchor_id"] for row in observations] != [
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
    expected_loads = frozen["offered_load_sweep_requests_per_second"]
    if config["publication_curves"][0]["offered_load_requests_per_second"] != (
        expected_loads["sglang_decode_standard"]
    ):
        raise ValueError("standard-decode sweep drifted")
    for curve in config["publication_curves"]:
        loads = curve["offered_load_requests_per_second"]
        if loads != sorted(set(loads)) or any(PS_PER_SECOND % load for load in loads):
            raise ValueError("publication loads must be unique exact-picosecond rates")


def verify_preservation_lock(
    repository_root: Path,
    frozen: dict[str, Any],
) -> list[dict[str, str]]:
    """Verify and return every preregistered byte-identity row."""

    validate_expectations(frozen)
    checked = []
    for artifact in frozen["preservation_lock"]["artifacts"]:
        relative = PurePosixPath(str(artifact["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("preservation paths must be repository-relative")
        path = repository_root.joinpath(*relative.parts)
        actual = sha256(path)
        if actual != artifact["sha256"]:
            raise ValueError(f"preservation digest disagrees for {artifact['path']}")
        checked.append({"path": artifact["path"], "sha256": actual})
    return checked


def _append_access(log_path: Path, entry: dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(entry, sort_keys=True) + "\n")


def _read_span(stream: BinaryIO, offset: int, length: int) -> dict[str, Any]:
    stream.seek(offset)
    raw = stream.read(length)
    if len(raw) != length:
        raise ValueError("anchor span ended before its frozen length")
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise TypeError("anchor span must contain one JSON object")
    return value


def read_anchor_subset(
    anchor_path: Path,
    anchor_ids: tuple[str, ...],
    access_log: Path,
    *,
    classification: str,
) -> dict[str, dict[str, Any]]:
    """Read only fixed-digest byte spans for the explicitly allowed anchors."""

    allowed = set(anchor_ids)
    if classification == "calibration":
        expected = {"sglang_prefill_1k", "sglang_decode_standard"}
    elif classification == "held_out":
        expected = {"sglang_prefill_2k", "sglang_prefill_4k"}
    else:
        raise ValueError("anchor access classification must be calibration or held_out")
    if allowed != expected:
        raise ValueError(f"{classification} anchor allowlist disagrees")
    if sha256(anchor_path) != ANCHOR_SHA256:
        raise ValueError("anchor freeze digest disagrees before field access")
    selected = {}
    with anchor_path.open("rb", buffering=0) as stream:
        for anchor_id in anchor_ids:
            offset, length, frozen_classification = ANCHOR_SPANS[anchor_id]
            if frozen_classification != classification:
                raise ValueError("anchor span classification disagrees")
            row = _read_span(stream, offset, length)
            if row.get("id") != anchor_id or row.get("role") != classification.replace(
                "_", "-"
            ):
                raise ValueError("field-addressed anchor identity or role disagrees")
            selected[anchor_id] = row
            _append_access(
                access_log,
                {
                    "schema": ACCESS_SCHEMA,
                    "status": "PASS",
                    "classification": classification,
                    "anchor_id": anchor_id,
                    "offset": offset,
                    "length": length,
                    "whole_record_loaded": False,
                },
            )
    return selected


def _prediction_rows(frozen: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        row["anchor_id"]: row
        for row in frozen["pre_fit_prediction_layers"]
        if row.get("status") != "BLOCKED"
    }


def _throughput(per_node_tokens: int, service_ps: Fraction) -> Fraction:
    if service_ps <= 0:
        raise ValueError("service must be positive")
    return Fraction(per_node_tokens * PS_PER_SECOND, 1) / service_ps


def prediction_layers(
    frozen_row: dict[str, Any],
    frozen: dict[str, Any],
    fitted_exposed_fraction: Fraction,
    fitted_surcharge_ps: int,
) -> dict[str, dict[str, Any]]:
    """Return the three exact prediction layers at one frozen fit."""

    constants = {row["id"]: row for row in frozen["constants"]["tunable"]}
    surcharge = constants["intra_node_collective_surcharge_ps"]
    if not surcharge["envelope"]["lower"] <= fitted_surcharge_ps <= surcharge[
        "envelope"
    ]["upper"]:
        raise ValueError("fitted surcharge leaves its closed envelope")
    exposed = constants["overlap_exposed_fraction"]["envelope"]
    if not _fraction(exposed["lower"], "f.lower") <= fitted_exposed_fraction <= _fraction(
        exposed["upper"], "f.upper"
    ):
        raise ValueError("fitted exposed fraction leaves its clean envelope")

    if "candidate_compute_service_ps" not in frozen_row:
        point = _fraction(frozen_row["physics_only"]["point"], "decode.point")
        exact = {
            "lower": fraction_json(point),
            "point": fraction_json(point),
            "upper": fraction_json(point),
            "contributions": [
                {
                    "source_kind": "candidate-record",
                    "source_id": (
                        "ff46f6d8a79ddae899da89d4db6eb34373f8042acd06cab50b6336c8fb9a8f52"
                    ),
                },
                {
                    "source_kind": "benchmark-bias-attenuation",
                    "applied": False,
                    "reason": "policy rule five registered modeling residual",
                },
            ],
        }
        return {
            "physics_only": dict(exact),
            "physics_plus_boundary": dict(exact),
            "physics_plus_boundary_plus_attenuation": dict(exact),
        }

    prefill = frozen["pricing_configuration"]["prefill"]
    packet = prefill["point_arm"]["communication_service_ps"]
    compute = int(frozen_row["candidate_compute_service_ps"])
    tokens = int(frozen_row["per_node_tokens"])
    physics_lower = _throughput(tokens, Fraction(max(compute, packet["upper"])))
    physics_point = _throughput(tokens, Fraction(max(compute, packet["selected"])))
    physics_upper = _throughput(tokens, Fraction(max(compute, packet["lower"])))
    f_lower = _fraction(exposed["lower"], "f.lower")
    f_upper = _fraction(exposed["upper"], "f.upper")
    boundary_lower = _throughput(
        tokens,
        Fraction(max(compute, packet["upper"]))
        + f_upper * min(compute, packet["upper"]),
    )
    boundary_point = _throughput(
        tokens,
        Fraction(max(compute, packet["selected"]))
        + fitted_exposed_fraction * min(compute, packet["selected"]),
    )
    boundary_upper = _throughput(
        tokens,
        Fraction(max(compute, packet["lower"]))
        + f_lower * min(compute, packet["lower"]),
    )
    factor = _factor_interval(frozen)
    physics = {
        "lower": fraction_json(physics_lower),
        "point": fraction_json(physics_point),
        "upper": fraction_json(physics_upper),
        "contributions": [
            {
                "source_kind": "comp75-clean-composition-record",
                "source_id": prefill["composition_authority"]["sha256"],
                "service_ps": dict(packet),
            },
            {
                "source_kind": "constant-envelope",
                "source_id": surcharge["id"],
                "selected_ps": fitted_surcharge_ps,
                "application_count": surcharge[
                    "third_run_mechanism_path_application_count_per_step"
                ],
            },
            {
                "source_kind": "distribution",
                "source_id": "comp74-zero-width-insufficient-replays",
                "relative_half_width": 0,
                "stability_claim": False,
            },
        ],
    }
    boundary = {
        "lower": fraction_json(boundary_lower),
        "point": fraction_json(boundary_point),
        "upper": fraction_json(boundary_upper),
        "contributions": [
            *physics["contributions"],
            {
                "source_kind": "overlap-exposure-envelope",
                "source_id": prefill["boundary_authority"]["sha256"],
                "lower_fraction": fraction_json(f_lower),
                "selected_fraction": fraction_json(fitted_exposed_fraction),
                "upper_fraction": fraction_json(f_upper),
            },
        ],
    }
    attenuated = {
        "lower": fraction_json(boundary_lower * factor["lower"]),
        "point": fraction_json(boundary_point * factor["point"]),
        "upper": fraction_json(boundary_upper * factor["upper"]),
        "contributions": [
            *boundary["contributions"],
            {
                "source_kind": "benchmark-bias-attenuation",
                "source_id": frozen["attenuation_layer"]["factors"][0]["id"],
                "factor": {
                    edge: fraction_json(value) for edge, value in factor.items()
                },
                "anchor_numeric_input_count": 0,
            },
        ],
    }
    return {
        "physics_only": physics,
        "physics_plus_boundary": boundary,
        "physics_plus_boundary_plus_attenuation": attenuated,
    }


def _signed_error(predicted: Fraction, published: Fraction) -> Fraction:
    return predicted / published - 1


def _layer_comparison(
    layers: dict[str, dict[str, Any]],
    published: Fraction,
    bar: Fraction,
) -> dict[str, dict[str, Any]]:
    result = {}
    for name, interval in layers.items():
        point = _fraction(interval["point"], f"{name}.point")
        signed = _signed_error(point, published)
        result[name] = {
            "prediction": interval,
            "signed_relative_error": fraction_json(signed),
            "absolute_relative_error": fraction_json(abs(signed)),
            "point_passes_5_percent": abs(signed) <= bar,
        }
    return result


def fit_constants(
    calibration: dict[str, dict[str, Any]],
    frozen: dict[str, Any],
) -> dict[str, Any]:
    """Fit both inherited constants using only the two calibration objects."""

    validate_expectations(frozen)
    visible = tuple(frozen["fit_rule"]["visible_anchor_ids"])
    if set(calibration) != set(visible):
        raise ValueError("calibration input set disagrees")
    rows = _prediction_rows(frozen)
    prefill_row = rows["sglang_prefill_1k"]
    published = Fraction(int(calibration["sglang_prefill_1k"]["value"]))
    attenuation = _factor_interval(frozen)["point"]
    packet = frozen["pricing_configuration"]["prefill"]["point_arm"][
        "communication_service_ps"
    ]["selected"]
    compute = int(prefill_row["candidate_compute_service_ps"])
    tokens = int(prefill_row["per_node_tokens"])
    raw_exposed = (
        attenuation * tokens * PS_PER_SECOND / published - max(compute, packet)
    ) / min(compute, packet)
    exposed_envelope = {
        edge: _fraction(
            next(
                row
                for row in frozen["constants"]["tunable"]
                if row["id"] == "overlap_exposed_fraction"
            )["envelope"][edge],
            f"f.{edge}",
        )
        for edge in ("lower", "upper")
    }
    fitted_exposed = min(
        exposed_envelope["upper"], max(exposed_envelope["lower"], raw_exposed)
    )
    fitted_surcharge = 0
    bar = _fraction(
        frozen["scoring_rule"]["maximum_absolute_relative_error"], "fit.bar"
    )
    calibration_rows = []
    objective_terms = []
    for anchor_id in visible:
        target = Fraction(int(calibration[anchor_id]["value"]))
        layers = prediction_layers(
            rows[anchor_id], frozen, fitted_exposed, fitted_surcharge
        )
        comparisons = _layer_comparison(layers, target, bar)
        scored_error = _fraction(
            comparisons["physics_plus_boundary_plus_attenuation"][
                "signed_relative_error"
            ],
            "fit.scored_error",
        )
        objective_terms.append(float(scored_error) ** 2)
        calibration_rows.append(
            {
                "anchor_id": anchor_id,
                "published": fraction_json(target),
                "layers": comparisons,
            }
        )
    return {
        "schema": FIT_SCHEMA,
        "status": "FROZEN",
        "objective": frozen["fit_rule"]["objective"],
        "objective_value": math.fsum(objective_terms),
        "constants": [
            {
                "id": "intra_node_collective_surcharge_ps",
                "fitted": fitted_surcharge,
                "envelope": {"lower": 0, "upper": 30_128_029},
                "application_count_per_step": 0,
                "tie_break_applied": True,
            },
            {
                "id": "overlap_exposed_fraction",
                "unclamped_solution": fraction_json(raw_exposed),
                "fitted": fraction_json(fitted_exposed),
                "envelope": {
                    edge: fraction_json(value)
                    for edge, value in exposed_envelope.items()
                },
                "clamped_to_envelope": fitted_exposed != raw_exposed,
                "tie_break_applied": False,
            },
        ],
        "accessed_anchor_ids": sorted(visible),
        "forbidden_anchor_ids_accessed": [],
        "calibration_rows": calibration_rows,
        "attenuation_factor_fitted": False,
        "in_run_adjustment_allowed": False,
    }


def _fit_values(fit: dict[str, Any]) -> tuple[Fraction, int]:
    constants = {row["id"]: row for row in fit["constants"]}
    return (
        _fraction(constants["overlap_exposed_fraction"]["fitted"], "fit.f"),
        int(constants["intra_node_collective_surcharge_ps"]["fitted"]),
    )


def score_frozen_fit(
    held_out: dict[str, dict[str, Any]],
    frozen: dict[str, Any],
    fit: dict[str, Any],
    *,
    fit_sha256: str,
) -> dict[str, Any]:
    """Score the two held-out prefill anchors after the fit is addressed."""

    if fit.get("schema") != FIT_SCHEMA or fit.get("status") != "FROZEN":
        raise ValueError("held-out scoring requires a frozen third-run fit")
    if len(fit_sha256) != 64 or any(char not in "0123456789abcdef" for char in fit_sha256):
        raise ValueError("held-out scoring requires the serialized fit SHA-256")
    if fit["forbidden_anchor_ids_accessed"]:
        raise ValueError("fit crossed the held-out access boundary")
    scorable = tuple(frozen["scoring_rule"]["scorable_held_out_anchor_ids"])
    if set(held_out) != set(scorable):
        raise ValueError("held-out input set disagrees")
    prediction_rows = _prediction_rows(frozen)
    fitted_exposed, fitted_surcharge = _fit_values(fit)
    bar = _fraction(
        frozen["scoring_rule"]["maximum_absolute_relative_error"], "score.bar"
    )
    rows = []
    for anchor_id in scorable:
        published = Fraction(int(held_out[anchor_id]["value"]))
        layers = prediction_layers(
            prediction_rows[anchor_id], frozen, fitted_exposed, fitted_surcharge
        )
        rows.append(
            {
                "anchor_id": anchor_id,
                "published": fraction_json(published),
                "layers": _layer_comparison(layers, published, bar),
            }
        )
    scored_name = "physics_plus_boundary_plus_attenuation"
    unattenuated_name = "physics_plus_boundary"
    attenuated_max = max(
        _fraction(row["layers"][scored_name]["absolute_relative_error"], "score.error")
        for row in rows
    )
    unattenuated_max = max(
        _fraction(
            row["layers"][unattenuated_name]["absolute_relative_error"],
            "score.unattenuated_error",
        )
        for row in rows
    )
    status = "PASS" if attenuated_max <= bar else "REFUTED"
    unattenuated_status = "PASS" if unattenuated_max <= bar else "REFUTED"
    return {
        "schema": SCORE_SCHEMA,
        "status": status,
        "unattenuated_status": unattenuated_status,
        "scope": (
            "priced held-out prefill anchors under the declared benchmark-bias model"
        ),
        "fit_sha256": fit_sha256,
        "maximum_attenuated_absolute_relative_error": fraction_json(attenuated_max),
        "maximum_unattenuated_absolute_relative_error": fraction_json(unattenuated_max),
        "acceptance_bar": fraction_json(bar),
        "accessed_anchor_ids": sorted(scorable),
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
                "numeric_anchor_read": False,
            }
        ],
    }


def load_access_log(path: Path) -> list[dict[str, Any]]:
    """Load the external append-only anchor access ledger."""

    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError("access ledger row must be an object")
            rows.append(value)
    return rows


def access_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize the exact calibration-before-held-out access sequence."""

    expected = [
        ("calibration", "sglang_prefill_1k"),
        ("calibration", "sglang_decode_standard"),
        ("held_out", "sglang_prefill_2k"),
        ("held_out", "sglang_prefill_4k"),
    ]
    observed = [(row["classification"], row["anchor_id"]) for row in rows]
    if observed != expected or not all(row["whole_record_loaded"] is False for row in rows):
        raise ValueError("anchor access sequence disagrees")
    return {
        "status": "PASS",
        "sequence": rows,
        "whole_anchor_record_loaded": False,
        "calibration_access_count": 2,
        "held_out_access_count": 2,
        "mtp_numeric_access_count": 0,
    }


def file_sha256(path: Path) -> str:
    """Return a local SHA-256 without changing the shared helper surface."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = [
    "ACCESS_SCHEMA",
    "ANCHOR_SHA256",
    "FIT_SCHEMA",
    "RUN2_CONFIG_SHA256",
    "SCORE_SCHEMA",
    "access_summary",
    "build_publication_curve",
    "file_sha256",
    "fit_constants",
    "load_access_log",
    "load_json",
    "prediction_layers",
    "read_anchor_subset",
    "score_frozen_fit",
    "sha256",
    "stable_request_projection",
    "validate_execution_config",
    "validate_expectations",
    "verify_preservation_lock",
    "write_json",
]
