from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def _load_shape() -> ModuleType:
    root = Path(__file__).resolve().parents[1]
    reader_path = root / "examples/deployment_curve_v1/core64_field_reader.py"
    reader_spec = importlib.util.spec_from_file_location(
        "core64_field_reader",
        reader_path,
    )
    assert reader_spec is not None
    assert reader_spec.loader is not None
    reader = importlib.util.module_from_spec(reader_spec)
    reader_spec.loader.exec_module(reader)

    import sys

    sys.modules["core64_field_reader"] = reader
    path = root / "examples/deployment_curve_v1/core64_shape.py"
    spec = importlib.util.spec_from_file_location("core64_shape", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


shape = _load_shape()
ROOT = Path(__file__).resolve().parents[1]


def _expectations():
    import json

    return json.loads(
        (
            ROOT / "examples/deployment_curve_v1/core64_expectations.json"
        ).read_text(encoding="utf-8")
    )


def _events():
    rows = []
    for access in range(1, 4):
        common = {
            "access_id": f"A{access:02d}",
            "classification": "synthetic",
            "held_out_mtp_value_accessed": False,
            "record": "synthetic.json",
            "record_size_bytes": 100,
            "schema": "simllm-deployment-curve-core64-access-v1",
            "selector": f"/synthetic/{access}",
            "whole_file_streamed": False,
        }
        rows.extend(
            [
                {
                    **common,
                    "bytes_accessed": 0,
                    "event": "BEGIN",
                    "event_index": 2 * access - 1,
                    "status": "IN_PROGRESS",
                },
                {
                    **common,
                    "bytes_accessed": 50,
                    "event": "END",
                    "event_index": 2 * access,
                    "status": "PASS",
                },
            ]
        )
    return rows


def _inputs(expectations):
    invariant = []
    for row in expectations["component_classification"]["logical_families"]:
        if row["family_id"] in {
            "moe_routed_experts",
            "multi_token_prediction_head",
        }:
            continue
        invariant.append(
            {
                "family_id": row["family_id"],
                "shape_vector": {"values": row["expected_shape_values"]},
            }
        )
    physical = {
        "kernel_classification_ledger": [
            {"family": "retained", "name": "attention"},
            {"family": "routed_expert", "name": "fused_moe_kernel"},
        ],
        "kernel_row_count": 2,
        "routed_kernel_row_count": 1,
    }
    return {
        "attention_parallelism": "data-parallel",
        "core63": {
            "calibration_only": {
                "residency_corrected": {
                    "classification": "UNDERCORRECTION",
                    "prediction_tokens_per_second_per_node": "9544.657796",
                    "signed_residual_percent": "-57.164268",
                }
            },
            "residency_derivation": {
                "family_decomposition": physical,
                "step": {
                    "residency_corrected_ps": {
                        "published_ps_round_half_up": 26821286365
                    }
                },
            },
            "scope": {
                "held_out_mtp_used_in_arithmetic_or_compared": False,
                "parameters_amended_or_refit": False,
                "scored_run_performed": False,
                "zero_free_or_fitted_constants": True,
            },
        },
        "standard_case": {
            "case_id": "sglang-decode-ep72-b32-c2000",
            "new_tokens_per_rank": 32,
            "phase": "decode",
            "rank_invariant_family_projections": invariant,
        },
    }


def test_null_shape_result_retains_the_exact_remainder() -> None:
    expectations = _expectations()
    result = shape.build_result(
        expectations,
        {
            "arithmetic_or_direction_amended": False,
            "component_classification_amended": False,
            "expected_forbidden_access_ledger": [],
            "prior_structural_rejection": {
                "access_count": 3,
                "end_statuses": ["PASS", "PASS", "PASS"],
                "event_count": 6,
                "held_out_mtp_numeric_value_accessed": False,
                "whole_file_streamed": False,
            },
            "schema": "simllm-deployment-curve-core64-structural-retry-expectations-v1",
            "status": "EXPECTATIONS_ONLY_STRUCTURAL_RETRY",
            "task": "CORE-64",
        },
        _inputs(expectations),
        _events(),
        _events(),
        {"checked_count": 134},
        base_commit="base",
        expectations_commit="expectations",
        runner_commit="runner",
    )

    assert result["per_rank_shape"]["requests_per_rank"] == 32
    assert result["per_rank_shape"]["kv_tokens_per_rank"] == 64_000
    assert result["per_rank_shape"]["shape_mismatch_count"] == 0
    assert result["calibration_only"] == {
        "anchor_tokens_per_second_per_node": "22282",
        "classification": "UNDERCORRECTION",
        "corrected_step_ps_round_half_up": 26_821_286_365,
        "final_prediction_tokens_per_second_per_node": "9544.657796",
        "final_signed_residual_percent": "-57.164268",
        "prediction_movement_tokens_per_second_per_node": "0.000000",
        "signed_difference_from_anchor_tokens_per_second_per_node": "-12737.342204",
        "signed_residual_movement_percentage_points": "0.000000",
    }
    assert result["registry_disposition"]["exact_remainder_id"] == "CORE-65"
    assert result["access"]["cumulative_access_count"] == 6


def test_access_validation_rejects_complete_byte_coverage() -> None:
    events = _events()
    events[-1]["bytes_accessed"] = events[-1]["record_size_bytes"]

    import pytest

    with pytest.raises(ValueError, match="whole-file"):
        shape.validate_access_events(events)
