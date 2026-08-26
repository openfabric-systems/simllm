"""Pure fitting, scoring and curve construction for CORE-54 run two."""

from __future__ import annotations

import math
from collections.abc import Iterable
from fractions import Fraction
from pathlib import Path
from typing import Any

from curve_tools import as_fraction, fraction_json
from flagship_tools import (
    build_publication_curve,
    load_json,
    sha256,
    stable_request_projection,
    write_json,
)

EXPECTATIONS_SCHEMA = "simllm-deployment-curve-scored-run2-expectations-v1"
CONFIG_SCHEMA = "simllm-deployment-curve-flagship-run2-config-v1"
FIT_SCHEMA = "simllm-deployment-curve-flagship-run2-fit-v1"
SCORE_SCHEMA = "simllm-deployment-curve-flagship-run2-score-v1"
PS_PER_SECOND = 1_000_000_000_000


def validate_expectations(value: dict[str, Any]) -> None:
    """Validate the second-run chronology and scoring boundary."""

    if value.get("schema") != EXPECTATIONS_SCHEMA:
        raise ValueError("second-run expectations schema disagrees")
    if value.get("status") != "EXPECTATIONS_ONLY":
        raise ValueError("second-run expectations must remain expectations only")
    chronology = value["chronology"]
    if any(
        chronology[name]
        for name in (
            "second_scored_runner_existed_before_this_freeze",
            "second_fitted_constants_existed_before_this_freeze",
            "second_held_out_score_existed_before_this_freeze",
            "second_flagship_figure_existed_before_this_freeze",
        )
    ):
        raise ValueError("a second-run output predates the freeze")
    allocation = value["inherited_rulings"]["allocation"]
    if (
        allocation["prefill_experiment"]["simultaneous_with_decode_experiment"]
        or allocation["decode_experiment"]["simultaneous_with_prefill_experiment"]
        or allocation["structural_comparator_only"]["may_be_called_96_gpu_system"]
    ):
        raise ValueError("the separate-experiment ruling drifted")
    composition = value["pricing_configuration"]["prefill"]
    if composition["new_free_or_fitted_tunables"] != 0:
        raise ValueError("the clean composition must add zero tunables")
    if composition["operator"] != "max":
        raise ValueError("the clean composition must remain max-like")
    decode = value["pricing_configuration"]["decode"]
    if decode["mechanism_count"] != 0:
        raise ValueError("no evidence-backed decode mechanism is frozen")
    if not decode["remote_kv_projection"]["enabled_for_this_run"]:
        raise ValueError("the second scored run must enable remote-KV projection")
    scorable = set(value["scoring_rule"]["scorable_held_out_anchor_ids"])
    blocked = set(value["scoring_rule"]["blocked_held_out_anchor_ids"])
    if scorable != {"sglang_prefill_2k", "sglang_prefill_4k"}:
        raise ValueError("the priced held-out set disagrees")
    if blocked != {"sglang_decode_simulated_mtp"}:
        raise ValueError("MTP must remain the only blocked held-out anchor")


def validate_config(config: dict[str, Any], frozen: dict[str, Any]) -> None:
    """Validate the compact execution configuration without importing SGLang."""

    validate_expectations(frozen)
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("second-run configuration schema disagrees")
    if config["study"]["classification"] != "scored":
        raise ValueError("second-run configuration must be scored")
    live = config["live_session"]
    if (
        live["prefill_engines"],
        live["decode_engines"],
        live["simulated_gpus_per_engine"],
    ) != (1, 1, 8):
        raise ValueError("second-run live scale disagrees")
    if live["project_remote_kv_length"] is not True:
        raise ValueError("remote-KV projection must be enabled")
    observations = config["exact_shape_observations"]
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
        raise ValueError("live context must preserve every prompt plus margin")
    expected_loads = frozen["offered_load_sweep_requests_per_second"]
    for curve in config["publication_curves"]:
        loads = curve["offered_load_requests_per_second"]
        if loads != sorted(set(loads)):
            raise ValueError("curve loads must be unique and increasing")
        if any(PS_PER_SECOND % load for load in loads):
            raise ValueError("curve loads must map to exact picosecond intervals")
    if config["publication_curves"][0]["offered_load_requests_per_second"] != (
        expected_loads["sglang_decode_standard"]
    ):
        raise ValueError("standard-decode sweep drifted from the freeze")


def verify_preservation_lock(
    repository_root: Path,
    frozen: dict[str, Any],
) -> list[dict[str, str]]:
    """Verify and return the preregistered byte-identity ledger."""

    checked = []
    for artifact in frozen["preservation_lock"]["artifacts"]:
        path = repository_root / artifact["path"]
        actual = sha256(path)
        if actual != artifact["sha256"]:
            raise ValueError(f"preservation digest disagrees for {artifact['path']}")
        checked.append({"path": artifact["path"], "sha256": actual})
    return checked


def _anchor_subset(
    anchor_freeze: dict[str, Any],
    allowed_ids: Iterable[str],
) -> dict[str, dict[str, Any]]:
    """Read anchor objects only after their IDs pass the caller allowlist."""

    allowed = set(allowed_ids)
    selected: dict[str, dict[str, Any]] = {}
    for anchor in anchor_freeze["anchors"]:
        anchor_id = anchor.get("id")
        if anchor_id in allowed:
            selected[str(anchor_id)] = anchor
    if set(selected) != allowed:
        raise ValueError(f"anchor allowlist did not resolve exactly: {sorted(allowed)}")
    return selected


def _throughput(per_node_tokens: int, service_ps: int) -> Fraction:
    if type(service_ps) is not int or service_ps <= 0:
        raise ValueError("service_ps must be a positive integer")
    return Fraction(per_node_tokens * PS_PER_SECOND, service_ps)


def prediction_interval(
    frozen_row: dict[str, Any],
    frozen: dict[str, Any],
    fitted_surcharge_ps: int,
) -> dict[str, Any]:
    """Propagate the record, inherited constant and distribution intervals."""

    constant = frozen["constants"]["tunable"][0]
    envelope = constant["envelope"]
    if not envelope["lower"] <= fitted_surcharge_ps <= envelope["upper"]:
        raise ValueError("fitted surcharge leaves its inherited envelope")
    count = constant["second_run_mechanism_path_application_count_per_step"]
    per_node_tokens = int(frozen_row["per_node_tokens"])
    if "composed_service_ps" in frozen_row:
        service = frozen_row["composed_service_ps"]
        record_kind = "comp75-clean-composition-record"
    else:
        point = int(frozen_row["candidate_service_ps"])
        service = {"lower": point, "point": point, "upper": point}
        record_kind = "candidate-record"
    lower = _throughput(
        per_node_tokens,
        int(service["upper"]) + count * int(envelope["upper"]),
    )
    point = _throughput(
        per_node_tokens,
        int(service["point"]) + count * fitted_surcharge_ps,
    )
    upper = _throughput(
        per_node_tokens,
        int(service["lower"]) + count * int(envelope["lower"]),
    )
    return {
        "lower": fraction_json(lower),
        "point": fraction_json(point),
        "upper": fraction_json(upper),
        "contributions": [
            {
                "source_kind": record_kind,
                "source_id": (
                    frozen["pricing_configuration"]["prefill"]
                    ["composition_authority"]["sha256"]
                    if record_kind == "comp75-clean-composition-record"
                    else "ff46f6d8a79ddae899da89d4db6eb34373f8042acd06cab50b6336c8fb9a8f52"
                ),
                "service_ps": dict(service),
            },
            {
                "source_kind": "constant-envelope",
                "source_id": constant["id"],
                "lower_ps": int(envelope["lower"]),
                "selected_ps": fitted_surcharge_ps,
                "upper_ps": int(envelope["upper"]),
                "application_count": count,
            },
            {
                "source_kind": "distribution",
                "source_id": "comp74-zero-width-insufficient-replays",
                "relative_half_width": 0,
                "stability_claim": False,
            },
        ],
    }


def _prediction_rows(frozen: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        row["anchor_id"]: row
        for row in frozen["pre_fit_predicted_bands"]
        if row.get("status") != "BLOCKED"
    }


def _relative_error(predicted: Fraction, published: Fraction) -> Fraction:
    return abs(predicted - published) / published


def fit_inherited_constant(
    anchor_freeze: dict[str, Any],
    frozen: dict[str, Any],
) -> dict[str, Any]:
    """Fit only calibration anchors under the zero-application successor path."""

    validate_expectations(frozen)
    rule = frozen["fit_rule"]
    visible = tuple(rule["visible_anchor_ids"])
    calibration = _anchor_subset(anchor_freeze, visible)
    prediction_rows = _prediction_rows(frozen)
    constant = frozen["constants"]["tunable"][0]
    fitted = int(constant["envelope"]["lower"])
    rows = []
    squared_errors = []
    for anchor_id in visible:
        published = as_fraction(calibration[anchor_id]["value"], f"{anchor_id}.value")
        interval = prediction_interval(prediction_rows[anchor_id], frozen, fitted)
        predicted = as_fraction(interval["point"], f"{anchor_id}.point")
        error = _relative_error(predicted, published)
        squared_errors.append(float(error) ** 2)
        rows.append(
            {
                "anchor_id": anchor_id,
                "published": fraction_json(published),
                "predicted": fraction_json(predicted),
                "absolute_relative_error": fraction_json(error),
            }
        )
    return {
        "schema": FIT_SCHEMA,
        "status": "FROZEN",
        "constant_id": constant["id"],
        "fitted_ps": fitted,
        "envelope": dict(constant["envelope"]),
        "application_count_per_step": constant[
            "second_run_mechanism_path_application_count_per_step"
        ],
        "objective": rule["objective"],
        "objective_value": math.fsum(squared_errors),
        "tie_break_applied": True,
        "accessed_anchor_ids": sorted(visible),
        "forbidden_anchor_ids_accessed": [],
        "calibration_rows": rows,
        "disposition": "inherited-fitted-parameter-not-measurement",
    }


def score_frozen_fit(
    anchor_freeze: dict[str, Any],
    frozen: dict[str, Any],
    fit: dict[str, Any],
    *,
    fit_sha256: str,
) -> dict[str, Any]:
    """Read priced held-out anchors only after the serialized fit is addressed."""

    if fit.get("schema") != FIT_SCHEMA or fit.get("status") != "FROZEN":
        raise ValueError("held-out scoring requires a frozen second-run fit")
    if len(fit_sha256) != 64 or any(char not in "0123456789abcdef" for char in fit_sha256):
        raise ValueError("held-out scoring requires the serialized fit SHA-256")
    if fit["forbidden_anchor_ids_accessed"]:
        raise ValueError("fit crossed the held-out access boundary")
    rule = frozen["scoring_rule"]
    scorable_ids = tuple(rule["scorable_held_out_anchor_ids"])
    held_out = _anchor_subset(anchor_freeze, scorable_ids)
    prediction_rows = _prediction_rows(frozen)
    bar = as_fraction(rule["maximum_absolute_relative_error"], "score.bar")
    rows = []
    for anchor_id in scorable_ids:
        published = as_fraction(held_out[anchor_id]["value"], f"{anchor_id}.value")
        interval = prediction_interval(
            prediction_rows[anchor_id],
            frozen,
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
    maximum = max(as_fraction(row["absolute_relative_error"], "score.error") for row in rows)
    status = "PASS" if maximum <= bar else "REFUTED"
    return {
        "schema": SCORE_SCHEMA,
        "status": status,
        "scope": "priced held-out prefill anchors only",
        "fit_sha256": fit_sha256,
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


__all__ = [
    "build_publication_curve",
    "fit_inherited_constant",
    "load_json",
    "prediction_interval",
    "score_frozen_fit",
    "sha256",
    "stable_request_projection",
    "validate_config",
    "validate_expectations",
    "verify_preservation_lock",
    "write_json",
]
