"""Regression checks for the live host-step sensitivity study."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
STUDY_PATH = REPOSITORY / "examples/host_step_cost_v1/run_study.py"
EXPECTATIONS_PATH = REPOSITORY / "examples/host_step_cost_v1/expectations.json"
CALIBRATION_PATH = REPOSITORY / "examples/host_step_cost_v1/calibration.json"
RESULTS_PATH = REPOSITORY / "examples/host_step_cost_v1/results.json"


def _study_module():
    spec = importlib.util.spec_from_file_location("host_step_cost_v1", STUDY_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_live_inventory_and_budget_arithmetic():
    study = _study_module()
    expectations = json.loads(EXPECTATIONS_PATH.read_text(encoding="utf-8"))
    calibration = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))

    assert sum(expectations["scored_live_relations"].values()) == 12
    assert expectations["representative_step"]["acceptance_decode_multiplier"] == [
        1.8,
        7.75,
    ]
    rows = [
        {
            "profile": profile,
            "launch_count": count,
            "host_model": {
                "point_ps_per_launch": calibration["profiles"][profile][
                    "point_ps_per_launch"
                ],
                "empirical_min_ps_per_launch": calibration["profiles"][profile][
                    "empirical_min_ps_per_launch"
                ],
                "empirical_max_ps_per_launch": calibration["profiles"][profile][
                    "empirical_max_ps_per_launch"
                ],
            },
        }
        for profile in ("turing-cuda-graph", "turing-eager-host")
        for count in (440, 567)
    ]
    budget = study._budget(rows, expectations)

    assert budget["passed"]
    assert [budget["minimum"], budget["maximum"]] == expectations[
        "live_attempt_two"
    ]["point_budget_expected"]
    assert [budget["empirical_minimum"], budget["empirical_maximum"]] == (
        expectations["live_attempt_two"]["empirical_budget_expected"]
    )
    assert budget["point"]["matches_frozen"]
    assert budget["empirical"]["matches_frozen"]
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


def test_tracked_live_result_is_accepted_when_present():
    if not RESULTS_PATH.exists():
        pytest.skip("live host-step study has not run yet")
    result = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))

    if result["run_status"] == "accepted":
        assert result["exact_ideal_offpath"]["passed"] is True
    else:
        assert result["run_status"] == "accepted_pending_offpath"
        assert result["exact_ideal_offpath"]["status"].startswith("pending")
    assert result["behavioral_score_interpretable"] is True
    assert result["fatal_guard_failures"] == []
    assert len(result["scored_relations"]) == 12
    assert all(row["passed"] for row in result["scored_relations"])


def test_attempt_two_guard_accepts_only_the_frozen_repair_rows():
    study = _study_module()
    expectations = json.loads(EXPECTATIONS_PATH.read_text(encoding="utf-8"))
    prior = json.loads(
        (REPOSITORY / "examples/host_step_cost_v1/live_attempt1.json").read_text(
            encoding="utf-8"
        )
    )
    prior_by_key = {
        (row["profile"], row["launch_count"]): row for row in prior["rows"]
    }
    rows = []
    for expected in expectations["live_attempt_two"]["expected_rows"]:
        old = prior_by_key[(expected["profile"], expected["launch_count"])]
        steps = []
        for index, old_step in enumerate(old["steps"]):
            steps.append(
                {
                    **old_step,
                    "released_at_ps": (
                        0 if index == 0 else expected["completion_ps"][index - 1]
                    ),
                    "compute_estimate_ps": expected["floor_ps"],
                    "compute_service_ps": expected["service_ps"],
                    "provider_compute_ps": expectations["live_attempt_two"][
                        "raw_provider_ps"
                    ][index],
                    "host_launch_floor_ps": expected["floor_ps"],
                    "step_latency_ps": expected["step_latency_ps"][index],
                    "completed_at_ps": expected["completion_ps"][index],
                    "observed_schedule_provider_compute_ps": expectations[
                        "live_attempt_two"
                    ]["raw_provider_ps"][index],
                    "observed_schedule_represented_compute_ps": expected[
                        "service_ps"
                    ],
                }
            )
        rows.append(
            {
                **old,
                "steps": steps,
                "ttft_ps": expected["ttft_ps"],
                "tpot_numerator": expected["tpot_ps"],
                "tpot_denominator": 1,
            }
        )

    assert study._attempt_two_checks(rows, expectations, prior)["passed"]
    assert study._quantized_compute_checks(rows)["passed"]
    assert study._observed_schedule_checks(rows, expectations)["passed"]

    rows[0]["steps"][0]["compute_service_ps"] -= 1000
    assert not study._attempt_two_checks(rows, expectations, prior)["passed"]
    assert not study._quantized_compute_checks(rows)["passed"]
