from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "examples/deployment_curve_v1"
sys.path.insert(0, str(STUDY))

import core65_physical_binding as binding


def _synthetic_rows() -> list[dict[str, object]]:
    rows = []
    for order in sorted(binding.FAMILY_BY_ORDER):
        rows.append(
            {
                "count_per_step": 1.0,
                "first_launch_order": float(order),
                "graph_record_count": 1,
                "name": f"kernel_{order}",
                "record_count": 1,
                "share_of_step_compute": 1 / 46,
                "total_duration_per_step_ns": 1.0,
            }
        )
    return rows


def test_total_inventory_partition_covers_all_46_orders() -> None:
    rows = _synthetic_rows()
    inventory = binding.build_inventory(rows, 46_000)

    assert inventory["coverage"] == "TOTAL_46_OF_46_ROWS_NO_UNMAPPED_KERNEL"
    assert inventory["kernel_row_count"] == 46
    assert inventory["measured_service_ps"] == 46_000
    assert inventory["bucket_row_counts"] == {
        "all_layer_common": 9,
        "dense_only": 5,
        "first_layer_only": 3,
        "later_layer_attention": 3,
        "moe_only": 14,
        "step_or_output_once": 12,
    }
    assert {row["first_launch_order"] for row in inventory["rows"]} == set(
        binding.FAMILY_BY_ORDER
    )


def test_vllm_routed_rows_are_not_claimed_as_sglang_ep72_identities() -> None:
    inventory = binding.build_inventory(_synthetic_rows(), 46_000)
    rows = {row["first_launch_order"]: row for row in inventory["rows"]}

    for order in (96, 98, 100, 101, 104):
        assert rows[order]["ep72_physical_binding_state"] == (
            "NO_ONE_TO_ONE_COUNTERPART"
        )
        assert "DeepEP" in rows[order]["ep72_comparison"]
    assert rows[108]["ep72_physical_binding_state"] == (
        "SEMANTIC_COUNTERPART_WITH_DP_LM_HEAD_REQUIRED"
    )


def test_hardware_remainder_pins_the_exact_ep72_cell() -> None:
    remainder = binding.build_hardware_remainder()

    assert remainder["task"] == "CORE-66"
    assert remainder["configuration"] == {
        "attention": "DP72",
        "batch_per_rank": 32,
        "context_tokens_per_request": 2000,
        "decode_nodes": 9,
        "expert_parallel_size": 72,
        "gpus_per_node": 8,
        "lm_head": "DP72_enabled",
        "model": "pre_staged_DeepSeek_V3_no_download",
        "moe_a2a_backend": "deepep",
        "mtp": "disabled",
        "output_length": 2,
        "profile_start_step": 0,
        "profile_steps": 1,
        "tensor_parallel_size": 72,
    }
    command = remainder["command"]
    assert "--tp-size 72 --dp-size 72 --ep-size 72" in command
    assert "--batch-size 32 --input-len 2000 --output-len 2" in command
    assert "--profile-start-step 0 --profile-steps 1" in command
    assert "--enable-dp-lm-head" in command
    assert "--moe-a2a-backend deepep" in command
    assert "dram__bytes_read.sum" in remainder["counter_pass"]["requirement"]


def test_expectations_and_preservation_are_protocol_void_and_literal() -> None:
    expectations = json.loads(
        (STUDY / "core65_expectations.json").read_text(encoding="utf-8")
    )

    binding.validate_expectations(expectations)
    preservation = binding.verify_preservation(expectations, ROOT)
    assert preservation == {
        "additional_core64_git_blob_count": 20,
        "checked_count": 154,
        "inherited_core64_checked_count": 134,
        "prior_artifacts_mutated": False,
    }
