from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "examples/deployment_curve_v1"


def _expectations() -> dict:
    return json.loads(
        (STUDY / "core66_decode_kernel_ladder_expectations.json").read_text(
            encoding="utf-8"
        )
    )


def test_freeze_precedes_confirmation_and_pins_exact_shape() -> None:
    frozen = _expectations()

    assert frozen["task"] == "CORE-66"
    assert frozen["chronology"] == {
        "base_commit": "598e12463f5b9b6355078e2da3f85828a01dad1a",
        "confirmatory_run_must_follow_this_commit": True,
        "confirmatory_run_observed": False,
        "scratch_layer_is_nonpublished_calibration": True,
        "status": "EXPECTATIONS_ONLY",
    }
    assert frozen["shape"] == {
        "batch_size": 32,
        "dummy_weights": True,
        "expert_parallel": 1,
        "gpu_count": 1,
        "kv_tokens_per_request": 2000,
        "network_enabled": False,
        "pipeline_parallel": 1,
        "tensor_parallel": 1,
    }
    assert frozen["runtime_substitution"]["sglang_physical_binding"] == (
        "DECLARED_LIMITATION_NOT_A_GOAL"
    )


def test_confirmatory_composition_and_relations_are_exact() -> None:
    frozen = _expectations()["confirmatory_capture"]

    assert frozen["composition"] == {
        "dense_layer_count": 3,
        "dense_layer_service_ps": 308_256_000,
        "graph_fixed_service_ps": 3_456_000,
        "independent_stream_contention_factor": "1",
        "moe_layer_count": 1,
        "moe_layer_service_ps": 415_648_000,
        "predicted_graph_service_ps": 1_343_872_000,
    }
    assert frozen["scored_relations"] == {
        "composition_residual_absolute_ps_at_most": 67_193_600,
        "native_repeat_difference_fraction_at_most": "0.05",
        "physical_service_ceiling_ps": 20_000_000_000,
        "physical_service_floor_ps": 524_000_000,
    }
    assert len(frozen["fatal_guards"]) == 8


def test_layer_model_closes_full_services_and_keeps_ep_unpriced() -> None:
    frozen = _expectations()
    symbols = frozen["model_freeze"]["symbols_ps"]

    assert symbols["C_common"] + symbols["D_dense_specific"] == 308_256_000
    assert (
        symbols["C_common"]
        + symbols["N_moe_nonrouted_specific"]
        + symbols["G_routed_group_at_R256_A256"]
        == 415_648_000
    )
    projection = frozen["calibration_projection_freeze"]
    assert projection["compute_only_prediction_interval_tokens_per_second_per_node"] == {
        "maximum": "18003.485222",
        "minimum": "16707.262995",
    }
    assert projection["expert_parallel_communication"] == {
        "measured_in_this_study": False,
        "modeled_term": "E_ep_ps >= 0",
        "publication_disposition": "ABSENT_AND_UNPRICED",
        "signed_effect": (
            "adding E_ep_ps increases step service and moves throughput downward"
        ),
    }


def test_inherited_inputs_match_preservation_manifest() -> None:
    lines = (
        STUDY / "core66_decode_kernel_ladder_prior_sha256.txt"
    ).read_text(encoding="utf-8").splitlines()
    assert len(lines) == _expectations()["preservation"]["required_count"] == 7
    for line in lines:
        expected, relative = line.split("  ", maxsplit=1)
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert actual == expected
