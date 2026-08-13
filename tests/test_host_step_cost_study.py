"""Regression checks for the held-out host-step sensitivity study."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
STUDY_PATH = REPOSITORY / "examples/host_step_cost_v1/run_study.py"
EXPECTATIONS_PATH = REPOSITORY / "examples/host_step_cost_v1/expectations.json"
CALIBRATION_PATH = REPOSITORY / "examples/host_step_cost_v1/calibration.json"
ATTEMPT_TWO_PATH = REPOSITORY / "examples/host_step_cost_v1/results.json"
HOLDOUT_PATH = REPOSITORY / "examples/host_step_cost_v1/holdout_results.json"


def _study_module():
    spec = importlib.util.spec_from_file_location("host_step_cost_v1", STUDY_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        json.loads(EXPECTATIONS_PATH.read_text(encoding="utf-8")),
        json.loads(CALIBRATION_PATH.read_text(encoding="utf-8")),
    )


def _request_fields(step_latencies: list[int]) -> dict[str, Any]:
    tpot = Fraction(step_latencies[1] + step_latencies[2], 2)
    return {
        "ttft_ps": step_latencies[0],
        "tpot_numerator": tpot.numerator,
        "tpot_denominator": tpot.denominator,
        "fallback_calls": 0,
        "runtime_step_count": 3,
    }


def _holdout_rows(
    expectations: dict[str, Any],
    calibration: dict[str, Any],
    network_mode: str,
) -> list[dict[str, Any]]:
    fixture = expectations["live_attempt_three"]["fixture"]
    ideal_steps = fixture["prior_observed_ideal_step_ps"]
    ideal_network = fixture["prior_observed_network_ps"]
    physical_network = fixture["network_physical_bounds_ps"]
    structural = [
        {
            "total_directed_bytes": 10_000 + index,
            "num_flows": 100 + index,
            "artifact_operation_ids": [[index, 0], [index, 1]],
        }
        for index in range(3)
    ]

    def steps(compute_ps: list[int], network_ps: list[int], floor_ps: int):
        released_at = fixture["arrival_ps"]
        result = []
        for index, (compute, network) in enumerate(
            zip(compute_ps, network_ps, strict=True)
        ):
            latency = compute + network
            result.append(
                {
                    "step_index": index,
                    "released_at_ps": released_at,
                    "step_latency_ps": latency,
                    "completed_at_ps": released_at + latency,
                    "compute_estimate_ps": (
                        fixture["fixed_compute_ps"][index]
                        if floor_ps == 0
                        else floor_ps
                    ),
                    "compute_service_ps": compute,
                    "provider_compute_ps": fixture["fixed_compute_ps"][index],
                    "host_launch_floor_ps": floor_ps,
                    "observed_schedule_provider_compute_ps": fixture[
                        "fixed_compute_ps"
                    ][index],
                    "observed_schedule_represented_compute_ps": compute,
                    "network_service_ps": network,
                    **structural[index],
                }
            )
            released_at += latency
        return result

    ideal_row_steps = steps(fixture["fixed_compute_ps"], ideal_network, 0)
    rows = [
        {
            "profile": "ideal",
            "launch_count": 0,
            "steps": ideal_row_steps,
            **_request_fields(ideal_steps),
        }
    ]
    for profile in ("turing-cuda-graph", "turing-eager-host"):
        profile_values = calibration["profiles"][profile]
        for count in (440, 567):
            floor_ps = count * profile_values["point_ps_per_launch"]
            represented = ((floor_ps + 999) // 1000) * 1000
            if network_mode == "prior":
                network = ideal_network
            elif network_mode == "floor":
                network = [bounds[0] for bounds in physical_network]
            elif network_mode == "ratio_countermodel":
                network = [physical_network[0][0]] + [
                    bounds[1] for bounds in physical_network[1:]
                ]
            else:
                raise AssertionError(f"unknown network mode: {network_mode}")
            row_steps = steps([represented] * 3, network, floor_ps)
            latencies = [step["step_latency_ps"] for step in row_steps]
            rows.append(
                {
                    "profile": profile,
                    "launch_count": count,
                    "host_model": {
                        "point_ps_per_launch": profile_values[
                            "point_ps_per_launch"
                        ],
                        "empirical_min_ps_per_launch": profile_values[
                            "empirical_min_ps_per_launch"
                        ],
                        "empirical_max_ps_per_launch": profile_values[
                            "empirical_max_ps_per_launch"
                        ],
                    },
                    "steps": row_steps,
                    **_request_fields(latencies),
                }
            )
    return rows


def test_frozen_holdout_inventory_and_budget_arithmetic():
    study = _study_module()
    expectations, calibration = _inputs()
    holdout = expectations["live_attempt_three"]

    assert expectations["live_attempt"] == 3
    assert sum(holdout["scored_relations"].values()) == 12
    assert holdout["fatal_unscored_guards"] == list(study.HOLDOUT_FATAL_GUARDS)
    assert expectations["live_attempt_two"]["genuine_risk_instances"] == 0
    assert sum(expectations["attempt_two_relations_originally_scored"].values()) == 12
    rows = _holdout_rows(expectations, calibration, "prior")
    budget = study._budget(rows, expectations)

    assert budget["passed"]
    assert [budget["minimum"], budget["maximum"]] == expectations[
        "live_attempt_two"
    ]["point_budget_expected"]
    assert [budget["empirical_minimum"], budget["empirical_maximum"]] == (
        expectations["live_attempt_two"]["empirical_budget_expected"]
    )
    assert budget["b100_host_step_cost"] == "unknown"


def test_fixed_provider_selects_distinct_step_service_by_context():
    from simllm.compute import GPU_ENVELOPES, KernelSpec

    study = _study_module()
    provider = study._fixed_provider({19: 99_024_000, 20: 99_048_000})

    first = provider.estimate(
        KernelSpec(
            name="fixed",
            flops=0.0,
            bytes_moved=0.0,
            config=(("kv_tokens", 19),),
        ),
        GPU_ENVELOPES["b100"],
    )
    second = provider.estimate(
        KernelSpec(
            name="fixed",
            flops=0.0,
            bytes_moved=0.0,
            config=(("kv_tokens", 20),),
        ),
        GPU_ENVELOPES["b100"],
    )

    assert first.duration_ps == 99_024_000
    assert second.duration_ps == 99_048_000
    with pytest.raises(KeyError, match="kv_tokens=21"):
        provider.estimate(
            KernelSpec(
                name="fixed",
                flops=0.0,
                bytes_moved=0.0,
                config=(("kv_tokens", 21),),
            ),
            GPU_ENVELOPES["b100"],
        )


def test_attempt_two_result_is_immutable_entailed_evidence():
    expectations, _ = _inputs()
    frozen = expectations["live_attempt_three"]["prior_attempt_two"]
    payload = ATTEMPT_TWO_PATH.read_bytes()
    result = json.loads(payload)

    assert hashlib.sha256(payload).hexdigest() == frozen["result_sha256"]
    assert result["live_attempt"] == 2
    assert result["fatal_guard_failures"] == []
    assert len(result["scored_relations"]) == frozen["entailed_findings"] == 12
    assert frozen["genuine_risk_instances"] == 0


def test_tracked_holdout_result_is_accepted_when_present():
    if not HOLDOUT_PATH.exists():
        pytest.skip("held-out host-step study has not run yet")
    result = json.loads(HOLDOUT_PATH.read_text(encoding="utf-8"))

    assert result["schema"] == "simllm-host-step-holdout-v1-results-v1"
    assert result["run_status"] == "accepted"
    assert result["behavioral_score_interpretable"] is True
    assert result["fatal_guard_failures"] == []
    assert result["genuine_risk_instances"] == 12
    assert len(result["scored_relations"]) == 12
    assert all(row["passed"] for row in result["scored_relations"])


def test_point_composition_passes_row_bands_and_increment_ratio():
    study = _study_module()
    expectations, calibration = _inputs()
    rows = _holdout_rows(expectations, calibration, "prior")

    scores, directions = study._score_rows(rows, expectations)

    assert len(scores) == 12
    assert all(row["passed"] for row in scores)
    assert directions["id"] == "HOLD-D1_two_parameter_directions"
    assert directions["passed"]
    live3 = [row for row in scores if row["id"].startswith("LIVE-3")]
    assert all(0.38 <= row["increment_ratio"] <= 0.4 for row in live3)


def test_fatal_valid_countermodels_can_fail_every_scored_family():
    study = _study_module()
    expectations, calibration = _inputs()

    frozen_countermodels = study._non_entailment_checks(expectations, calibration)
    assert frozen_countermodels["passed"]
    assert len(frozen_countermodels["rows"]) == 4
    assert all(row["fatal_physical_bounds_hold"] for row in frozen_countermodels["rows"])
    assert all(row["live1_countermodel_misses"] for row in frozen_countermodels["rows"])
    assert all(row["live2_countermodel_misses"] for row in frozen_countermodels["rows"])
    assert all(row["live3_countermodel_misses"] for row in frozen_countermodels["rows"])

    floor_rows = _holdout_rows(expectations, calibration, "floor")
    floor_scores, _ = study._score_rows(floor_rows, expectations)
    assert study._physical_checks(floor_rows, expectations, calibration)["passed"]
    assert study._conservation_checks(floor_rows, expectations)["passed"]
    assert study._quantized_compute_checks(floor_rows)["passed"]
    assert study._observed_schedule_checks(floor_rows, expectations)["passed"]
    assert study._budget(floor_rows, expectations)["passed"]
    assert not any(
        row["passed"]
        for row in floor_scores
        if row["id"] in {
            "LIVE-1_decode_multiplier_in_band",
            "LIVE-2_tpot_multiplier_in_band",
        }
    )

    ratio_rows = _holdout_rows(expectations, calibration, "ratio_countermodel")
    ratio_scores, _ = study._score_rows(ratio_rows, expectations)
    assert study._physical_checks(ratio_rows, expectations, calibration)["passed"]
    assert study._conservation_checks(ratio_rows, expectations)["passed"]
    assert study._quantized_compute_checks(ratio_rows)["passed"]
    assert study._observed_schedule_checks(ratio_rows, expectations)["passed"]
    assert study._budget(ratio_rows, expectations)["passed"]
    assert not any(
        row["passed"]
        for row in ratio_scores
        if row["id"] == "LIVE-3_ttft_relative_increase_is_smaller"
    )


def test_network_equality_is_survivable_and_not_a_conservation_guard():
    study = _study_module()
    expectations, calibration = _inputs()
    rows = _holdout_rows(expectations, calibration, "floor")

    conservation = study._conservation_checks(rows, expectations)
    diagnostic = study._network_service_diagnostic(rows)

    assert conservation["passed"]
    assert all(
        "network_is_host_invariant" not in step["checks"]
        for row in conservation["rows"]
        for step in row["step_checks"]
    )
    assert diagnostic["id"] == "HOLD-D2_network_service_matches_ideal"
    assert diagnostic["survivable"] is True
    assert diagnostic["passed"] is False
