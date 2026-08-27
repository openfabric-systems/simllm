"""Pure configuration, arithmetic and scoring for CORE-54 run four."""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any

EXPECTATIONS_SCHEMA = "simllm-deployment-curve-scored-run4-expectations-v1"
CONFIG_SCHEMA = "simllm-deployment-curve-flagship-run4-config-v1"
SCORE_SCHEMA = "simllm-deployment-curve-flagship-run4-score-v1"
RESULT_SCHEMA = "simllm-deployment-curve-flagship-run4-result-v1"
PS_PER_SECOND = 1_000_000_000_000


def fraction_json(value: Fraction) -> dict[str, int]:
    """Return one exact fraction in the repository wire form."""

    return {"numerator": value.numerator, "denominator": value.denominator}


def as_fraction(value: object, name: str) -> Fraction:
    """Decode an integer or exact fraction object."""

    if type(value) is int:
        return Fraction(value)
    if not isinstance(value, dict) or set(value) != {"numerator", "denominator"}:
        raise TypeError(f"{name} must be an exact fraction")
    return Fraction(int(value["numerator"]), int(value["denominator"]))


def write_json(path: Path, value: object) -> None:
    """Write stable UTF-8 JSON with pinned LF newlines."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def sha256(path: Path) -> str:
    """Return the SHA-256 of one local artifact."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_expectations(frozen: dict[str, Any]) -> None:
    """Validate the immutable pre-fit MTP scoring boundary."""

    if frozen.get("schema") != EXPECTATIONS_SCHEMA or frozen.get("status") != (
        "EXPECTATIONS_ONLY"
    ):
        raise ValueError("run-4 expectations identity differs")
    chronology = frozen["chronology"]
    if chronology["held_out_mtp_anchor_numeric_value_accessed"]:
        raise ValueError("run-4 freeze crossed the MTP held-out boundary")
    if any(
        chronology[name]
        for name in (
            "run4_fit_performed",
            "run4_runner_existed_before_this_freeze",
            "run4_score_existed_before_this_freeze",
            "run4_figure_existed_before_this_freeze",
        )
    ):
        raise ValueError("run-4 chronology differs")
    if frozen["attenuation_layer"]["admitted_factor_count"] != 0:
        raise ValueError("run-4 decode attenuation must remain absent")
    if frozen["fit_rule"]["refit_allowed"]:
        raise ValueError("run-4 must inherit the run-3 fit")
    rule = frozen["scoring_rule"]
    if rule["anchor_id"] != "sglang_decode_simulated_mtp" or not rule["score_once"]:
        raise ValueError("run-4 one-shot anchor differs")


def validate_config(config: dict[str, Any], frozen: dict[str, Any]) -> None:
    """Validate the disclosed separate MTP experiment shape."""

    validate_expectations(frozen)
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("run-4 configuration identity differs")
    shape = config["shape"]
    allocation = config["allocation"]
    if (
        allocation["nodes"],
        allocation["gpus_per_node"],
        allocation["expert_parallel"],
        allocation["separate_from_prefill"],
    ) != (9, 8, 72, True):
        raise ValueError("run-4 separate decode allocation differs")
    if (
        shape["batch_per_node"],
        shape["batch_per_gpu"],
        shape["kv_tokens_per_request"],
        shape["base_tokens_per_request"],
        shape["simulated_speculative_tokens_per_request"],
        shape["emitted_tokens_per_request"],
    ) != (128, 16, 4000, 1, 1, 2):
        raise ValueError("run-4 disclosed MTP shape differs")
    if shape["batch_per_gpu"] * allocation["gpus_per_node"] != shape[
        "batch_per_node"
    ]:
        raise ValueError("run-4 per-GPU batch mapping does not conserve")
    if config["model"]["weights_required"]:
        raise ValueError("run-4 must not require model weights")


def verify_preservation_lock(
    repository_root: Path,
    frozen: dict[str, Any],
) -> list[dict[str, str]]:
    """Verify every frozen prior artifact byte for byte."""

    validate_expectations(frozen)
    checked = []
    for artifact in frozen["preservation_lock"]["artifacts"]:
        relative = PurePosixPath(artifact["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("run-4 preservation path is not repository-relative")
        path = repository_root.joinpath(*relative.parts)
        actual = sha256(path)
        if actual != artifact["sha256"]:
            raise ValueError(f"preservation digest differs for {artifact['path']}")
        checked.append({"path": artifact["path"], "sha256": actual})
    if len(checked) < frozen["preservation_lock"]["minimum_artifact_count"]:
        raise ValueError("run-4 preservation class is undersized")
    return checked


def build_shape_observation(config: dict[str, Any], frozen: dict[str, Any]) -> dict[str, Any]:
    """Realize the exact disclosed one-node MTP request shape."""

    validate_config(config, frozen)
    shape = config["shape"]
    allocation = config["allocation"]
    requests = [
        {
            "request_ordinal": ordinal,
            "gpu_ordinal": ordinal // shape["batch_per_gpu"],
            "kv_tokens": shape["kv_tokens_per_request"],
            "base_tokens": shape["base_tokens_per_request"],
            "simulated_speculative_tokens": shape[
                "simulated_speculative_tokens_per_request"
            ],
            "emitted_tokens": shape["emitted_tokens_per_request"],
        }
        for ordinal in range(shape["batch_per_node"])
    ]
    per_gpu = [
        sum(row["gpu_ordinal"] == gpu for row in requests)
        for gpu in range(allocation["gpus_per_node"])
    ]
    return {
        "schema": "simllm-deployment-curve-run4-mtp-shape-observation-v1",
        "status": "PASS",
        "anchor_id": shape["anchor_id"],
        "allocation": dict(allocation),
        "request_count": len(requests),
        "requests_per_gpu": per_gpu,
        "total_emitted_tokens": sum(row["emitted_tokens"] for row in requests),
        "requests": requests,
        "weights_loaded": False,
    }


def _comparison(
    prediction: dict[str, Any],
    published: Fraction,
    bar: Fraction,
) -> dict[str, Any]:
    point = as_fraction(prediction["point"], "prediction.point")
    signed = point / published - 1
    return {
        "prediction": prediction,
        "signed_relative_error": fraction_json(signed),
        "absolute_relative_error": fraction_json(abs(signed)),
        "point_passes_5_percent": abs(signed) <= bar,
        "status": "PASS" if abs(signed) <= bar else "REFUTED",
    }


def score_mtp_anchor(
    anchor: dict[str, Any],
    frozen: dict[str, Any],
    *,
    prediction_sha256: str,
) -> dict[str, Any]:
    """Score the sole MTP row once against the serialized frozen prediction."""

    validate_expectations(frozen)
    if len(prediction_sha256) != 64:
        raise ValueError("run-4 scoring requires an addressed prediction")
    if anchor.get("id") != frozen["scoring_rule"]["anchor_id"]:
        raise ValueError("run-4 held-out anchor identity differs")
    if anchor.get("role") != "held-out":
        raise ValueError("run-4 anchor must remain held out")
    published = Fraction(int(anchor["value"]))
    bar = as_fraction(frozen["scoring_rule"]["maximum_absolute_relative_error"], "bar")
    frozen_row = frozen["pre_fit_prediction_layers"][0]
    layers = {
        name: _comparison(frozen_row[name], published, bar)
        for name in (
            "physics_only",
            "physics_plus_boundary",
            "physics_plus_boundary_plus_attenuation",
        )
    }
    scored = layers[frozen["scoring_rule"]["scored_layer"]]
    return {
        "schema": SCORE_SCHEMA,
        "status": scored["status"],
        "anchor_id": anchor["id"],
        "published": fraction_json(published),
        "layers": layers,
        "scored_layer": frozen["scoring_rule"]["scored_layer"],
        "acceptance_bar": fraction_json(bar),
        "prediction_sha256": prediction_sha256,
        "attenuation_applied": False,
        "score_attempt_count": 1,
        "in_run_adjustment_performed": False,
    }


def validate_run3_carry(value: dict[str, Any], frozen: dict[str, Any]) -> None:
    """Require the exact run-3 publication and immutable two-row PASS scope."""

    expected = frozen["inherited_run3_rows"]
    if value.get("schema") != "simllm-deployment-curve-flagship-run3-publication-v1":
        raise ValueError("run-3 publication schema differs")
    if value.get("status") != "PASS" or value.get("verdict") != (
        "SCORABLE_HELD_OUT_PASS_MTP_BLOCKED"
    ):
        raise ValueError("run-3 verdict differs")
    rows = value["held_out_score"]["rows"]
    if [row["anchor_id"] for row in rows] != [
        "sglang_prefill_2k",
        "sglang_prefill_4k",
    ]:
        raise ValueError("run-3 carried row order differs")
    scored_layer = "physics_plus_boundary_plus_attenuation"
    if not all(row["layers"][scored_layer]["point_passes_5_percent"] for row in rows):
        raise ValueError("run-3 carried PASS rows differ")
    if sha256(Path(__file__).resolve().parent / "flagship_run3_result.json") != expected[
        "authority_sha256"
    ]:
        raise ValueError("run-3 carry-forward authority digest differs")


def build_result(
    frozen: dict[str, Any],
    config: dict[str, Any],
    run3: dict[str, Any],
    shape: dict[str, Any],
    score: dict[str, Any],
    preservation: list[dict[str, str]],
    access: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compose the complete fourth-run result without changing prior rows."""

    validate_config(config, frozen)
    validate_run3_carry(run3, frozen)
    combined_status = "PASS" if score["status"] == "PASS" else "REFUTED"
    return {
        "schema": RESULT_SCHEMA,
        "status": combined_status,
        "verdict": f"ALL_SCORABLE_HELD_OUT_{combined_status}",
        "classification": "FOURTH_SCORED_FLAGSHIP",
        "scope": (
            "all three scorable held-out anchors: run-3 prefill rows carried "
            "forward unchanged and MTP scored unattenuated"
        ),
        "core54_closure": False,
        "closure_reason": (
            "CORE-54 retains decode calibration reproduction, COMP-74 "
            "distribution propagation, the Granite campaign arm and depth linearity."
        ),
        "allocation": config["allocation"],
        "shape_observation": shape,
        "fit": frozen["fit_rule"],
        "attenuation_layer": frozen["attenuation_layer"],
        "run3_carry_forward": {
            "authority_sha256": frozen["inherited_run3_rows"]["authority_sha256"],
            "status": "BYTE_IDENTICAL_NOT_RESCORED",
            "held_out_score": run3["held_out_score"],
            "anchor_predictions": run3["anchor_predictions"],
            "curves": run3["curves"],
            "second_legend": run3["second_legend"],
            "decode_calibration_miss": run3["decode_calibration_miss"],
        },
        "mtp_score": score,
        "combined_held_out_rows": [*run3["held_out_score"]["rows"], score],
        "access": {
            "whole_record_loaded": False,
            "rows": access,
            "mtp_anchor_access_count": sum(
                row.get("classification") == "held_out"
                and row.get("anchor_id") == "sglang_decode_simulated_mtp"
                and row.get("status") == "PASS"
                for row in access
            ),
        },
        "preservation_lock": {
            "status": "PASS",
            "artifacts": preservation,
        },
        "dominant_contributor": (
            "The 61-over-4 linear extrapolation of measured four-layer service "
            "dominates the low MTP prediction. Incomplete data-parallel-attention "
            "integration remains disclosed and is not attenuated."
        ),
        "remaining_work": [
            "decode calibration reproduction",
            "COMP-74 distribution propagation from retained repeats",
            "Granite campaign arm",
            "depth linearity",
        ],
        "deployment_frontier": {
            "status": "UNCHANGED_FROZEN_CONTRACT",
            "reason": (
                "The v2 contract fixes a standard-decode paired marker and a "
                "y-only H800 anchor; it has no MTP marker slot and byte-locks "
                "the existing figure."
            ),
        },
    }
