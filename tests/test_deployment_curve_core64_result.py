from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "examples/deployment_curve_v1"


def _result():
    return json.loads(
        (STUDY / "core64_shape_result.json").read_text(encoding="utf-8")
    )


def test_core64_result_publishes_the_null_movement_and_exact_remainder() -> None:
    result = _result()

    assert result["status"] == "PASS_NULL_SHAPE_MOVEMENT_EXACT_REMAINDER"
    assert result["per_rank_shape"] == {
        "attention_parallelism": "data-parallel-72",
        "batch_per_node": 256,
        "decode_nodes": 9,
        "global_requests": 2304,
        "kv_tokens_per_rank": 64000,
        "kv_tokens_per_request": 2000,
        "requests_per_rank": 32,
        "requests_per_rank_formula": "9 * 256 / 72",
        "shape_mismatch_count": 0,
    }
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
    assert result["registry_disposition"] == {
        "core64": "REMAINS_OPEN_LITERAL_GAP_UNRESOLVED_BY_NULL_MOVEMENT",
        "exact_remainder_id": "CORE-65",
        "reserved_id_free_on_base_main": True,
    }


def test_core64_result_enumerates_all_families_without_guessing_names() -> None:
    classification = _result()["component_classification"]
    ledger = classification["logical_family_ledger"]

    assert len(ledger) == 14
    assert sum(row["standard_decode_state"] == "present" for row in ledger) == 13
    assert all(row["shape_match"] for row in ledger)
    assert ledger[-1]["family_id"] == "multi_token_prediction_head"
    assert ledger[-1]["standard_decode_state"] == "absent_and_unread"
    assert classification["semantic_physical_binding_state"] == (
        "ABSENT_TOTAL_BINDING_NON_NUMERIC_FOR_NULL_SCALE"
    )
    assert classification["physical_kernel_rule"] == {
        "case_insensitive_routed_marker": "fused_moe_kernel",
        "nonmatching_noncollective_scale": "1",
        "semantic_name_guessing_permitted": False,
    }


def test_core64_access_and_preservation_are_literal() -> None:
    result = _result()

    assert result["access"]["access_count"] == 3
    assert result["access"]["access_event_count"] == 6
    assert result["access"]["cumulative_access_count"] == 6
    assert result["access"]["cumulative_access_event_count"] == 12
    assert result["access"]["whole_file_streams"] == 0
    assert result["access"]["forbidden_access_ledger"] == []
    assert json.loads(
        (STUDY / "core64_forbidden_access_ledger.json").read_text(encoding="utf-8")
    ) == []
    assert result["preservation_lock"] == {
        "additional_git_blob_count": 41,
        "checked_count": 134,
        "hash_verification_decoded_artifact_values": False,
        "inherited_sha256_count": 93,
        "prior_artifacts_mutated": False,
    }


def test_core64_markdown_and_registry_publish_the_exact_outcome() -> None:
    report = (STUDY / "core64_shape_result.md").read_text(encoding="utf-8")
    registry = (ROOT / "docs/modules/core.md").read_text(encoding="utf-8")

    assert report.index("## Enumerated shape mismatches") < report.index(
        "## Derived correction and signed movement"
    )
    assert "9 * 256 / 72 = 32" in report
    assert "64,000 aggregate KV-token" in report
    assert "CORE-64 prediction movement = 0.000000 tokens/s/node" in report
    assert "signed difference = -12737.342204 tokens/s/node" in report
    assert "final signed residual = -57.164268 percent" in report
    assert "CORE-64 remains open" in registry
    assert "- CORE-65 (Precision; P0; L):" in registry
    assert "-12,737.342204" in registry
