"""CORE-65 retained-kernel inventory and EP72 physical-binding publication."""

from __future__ import annotations

import hashlib
import html
import json
from collections.abc import Mapping, Sequence
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any

import core64_shape
from core65_field_reader import ACCESS_SCHEMA

EXPECTATIONS_SCHEMA = "simllm-deployment-curve-core65-expectations-v1"
RESULT_SCHEMA = "simllm-deployment-curve-core65-physical-binding-result-v1"
INVENTORY_SCHEMA = "simllm-deployment-curve-core65-kernel-inventory-v1"
REMAINDER_SCHEMA = "simllm-deployment-curve-core66-hardware-remainder-v1"

STEP_ORDERS = frozenset({0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 107, 108})
FIRST_ORDERS = frozenset({13, 14, 17})
COMMON_ORDERS = frozenset({11, 12, 16, 18, 19, 20, 21, 22, 24})
LATER_ORDERS = frozenset({34, 35, 38})
DENSE_ORDERS = frozenset({25, 27, 28, 30, 31})
MOE_ORDERS = frozenset({88, 90, 91, 92, 93, 94, 95, 96, 98, 99, 100, 101, 104, 105})

FAMILY_BY_ORDER = {
    0: "step_setup",
    1: "step_setup",
    2: "step_setup",
    3: "step_setup",
    4: "step_setup",
    6: "kv_slot_mapping",
    7: "step_setup",
    8: "mla_attention_setup",
    9: "step_setup",
    10: "embedding_and_input_norm",
    11: "mixed_projection_quantization",
    12: "mla_q_and_kv_compression",
    13: "first_layer_mla_transform",
    14: "first_layer_mla_transform",
    16: "mla_q_decompression",
    17: "first_layer_rotary_transform",
    18: "mla_kv_cache_write",
    19: "mla_attention_projection",
    20: "mla_attention",
    21: "mla_attention_combine",
    22: "mla_attention_projection",
    24: "mla_output_projection",
    25: "dense_input_norm",
    27: "dense_gate_up_projection",
    28: "dense_activation",
    30: "dense_down_projection",
    31: "dense_residual_and_norm",
    34: "later_layer_mla_transform",
    35: "later_layer_mla_transform",
    38: "later_layer_rotary_transform",
    88: "moe_input_norm",
    90: "moe_router",
    91: "moe_shared_expert_gate_up",
    92: "moe_router_reduction",
    93: "moe_topk",
    94: "moe_mixed_quantization",
    95: "moe_shared_expert_activation",
    96: "vllm_routed_expert_alignment",
    98: "vllm_routed_expert_sort",
    99: "moe_shared_expert_down",
    100: "vllm_routed_expert_compute",
    101: "vllm_routed_expert_activation",
    104: "vllm_routed_expert_sum",
    105: "moe_residual_and_norm",
    107: "output_token_gather",
    108: "lm_head",
}


def _mapping(name: str, value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return value


def _integer(name: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if type(value) is float and value.is_integer():
        value = int(value)
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def validate_expectations(expectations: Mapping[str, Any]) -> None:
    """Require the committed protocol-void, no-fit direction freeze."""

    if expectations.get("schema") != EXPECTATIONS_SCHEMA:
        raise ValueError("CORE-65 expectations schema differs")
    if expectations.get("task") != "CORE-65":
        raise ValueError("CORE-65 task identity differs")
    if expectations.get("status") != "EXPECTATIONS_ONLY_PROTOCOL_VOID":
        raise ValueError("CORE-65 expectations must record the protocol void")
    state = _mapping("protocol_state", expectations.get("protocol_state"))
    if state.get("forbidden_access_ledger_empty") is not False:
        raise ValueError("CORE-65 cannot claim an empty forbidden ledger")
    if state.get("known_pre_reader_incident_count") != 2:
        raise ValueError("CORE-65 incident count differs")
    if state.get("literal_protocol_closure_possible_in_this_worker") is not False:
        raise ValueError("CORE-65 cannot claim literal closure")
    if any(_mapping("scope_locks", expectations.get("scope_locks")).values()):
        raise ValueError("every CORE-65 scope lock must remain false")


def validate_access_events(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate the eight field-addressed retained-record accesses."""

    if len(events) != 16:
        raise ValueError("CORE-65 access event count differs")
    completed = []
    for index, event in enumerate(events, start=1):
        if event.get("event_index") != index or event.get("schema") != ACCESS_SCHEMA:
            raise ValueError("CORE-65 access event identity differs")
        if event.get("held_out_mtp_value_accessed") is not False:
            raise ValueError("CORE-65 reader reports held-out MTP exposure")
        if event.get("whole_file_streamed") is not False:
            raise ValueError("CORE-65 reader reports a whole-file stream")
    for offset in range(0, 16, 2):
        begin, end = events[offset : offset + 2]
        if begin.get("access_id") != end.get("access_id"):
            raise ValueError("CORE-65 access pair differs")
        if begin.get("event") != "BEGIN" or begin.get("bytes_accessed") != 0:
            raise ValueError("CORE-65 BEGIN event is not contemporaneous")
        if end.get("event") != "END" or end.get("status") != "PASS":
            raise ValueError("CORE-65 END event did not pass")
        consumed = _integer("bytes_accessed", end.get("bytes_accessed"), minimum=1)
        size = _integer("record_size_bytes", end.get("record_size_bytes"), minimum=2)
        if consumed >= size:
            raise ValueError("CORE-65 selector reached a whole-file stream")
        completed.append(dict(end))
    return {
        "access_count": 8,
        "access_event_count": 16,
        "completed_accesses": completed,
        "held_out_mtp_numeric_values_accessed_or_compared_by_reader": False,
        "whole_file_reader_streams": 0,
    }


def validate_capture_events(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate the logged, unavailable optional original capture profile."""

    if len(events) != 2:
        raise ValueError("CORE-65 optional-capture event count differs")
    begin, end = events
    if begin.get("event") != "BEGIN" or begin.get("bytes_accessed") != 0:
        raise ValueError("CORE-65 optional-capture BEGIN differs")
    if end.get("event") != "END" or end.get("status") != "UNAVAILABLE":
        raise ValueError("CORE-65 optional capture was not logged unavailable")
    if end.get("bytes_accessed") != 0:
        raise ValueError("unavailable CORE-65 capture consumed bytes")
    return {"access_event_count": 2, "available": False, "bytes_accessed": 0}


def _git_blob_sha1(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()


def verify_preservation(expectations: Mapping[str, Any], repository_root: Path) -> dict[str, Any]:
    """Extend the inherited 134-path lock with all 20 CORE-64 artifacts."""

    core64_path = repository_root / "examples/deployment_curve_v1/core64_expectations.json"
    with core64_path.open(encoding="utf-8") as stream:
        inherited = core64_shape.verify_preservation(json.load(stream), repository_root)
    if inherited.get("checked_count") != 134:
        raise ValueError("inherited CORE-64 preservation class differs")
    freeze = _mapping("preservation", expectations.get("preservation"))
    manifest = repository_root / PurePosixPath(
        str(freeze["additional_core64_git_blob_manifest_path"])
    )
    rows = []
    with manifest.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            fields = line.rstrip("\n").split("  ", maxsplit=1)
            if len(fields) != 2 or not all(fields):
                raise ValueError(f"{manifest.name}:{line_number}: malformed lock row")
            rows.append((fields[0], fields[1]))
    if len(rows) != freeze.get("additional_core64_git_blob_count"):
        raise ValueError("CORE-65 additional preservation count differs")
    for expected_blob, relative in rows:
        payload = (repository_root / PurePosixPath(relative)).read_bytes()
        if _git_blob_sha1(payload) != expected_blob:
            raise ValueError(f"CORE-65 preservation mismatch: {relative}")
    checked_count = inherited["checked_count"] + len(rows)
    if checked_count < freeze.get("minimum_checked_count", 0):
        raise ValueError("CORE-65 preservation class is too small")
    return {
        "additional_core64_git_blob_count": len(rows),
        "checked_count": checked_count,
        "inherited_core64_checked_count": inherited["checked_count"],
        "prior_artifacts_mutated": False,
    }


def _bucket(order: int) -> tuple[str, str, Fraction]:
    if order in STEP_ORDERS:
        return "step_or_output_once", "once_per_step", Fraction(1)
    if order in FIRST_ORDERS:
        return "first_layer_only", "once_in_first_layer", Fraction(1)
    if order in COMMON_ORDERS:
        return "all_layer_common", "per_captured_layer", Fraction(1, 4)
    if order in LATER_ORDERS:
        return "later_layer_attention", "per_later_captured_layer", Fraction(1, 3)
    if order in DENSE_ORDERS:
        return "dense_only", "per_dense_layer", Fraction(1, 3)
    if order in MOE_ORDERS:
        return "moe_only", "per_moe_layer", Fraction(1)
    raise ValueError(f"unmapped first launch order {order}")


def _ep72_binding(order: int) -> dict[str, str]:
    if order in {96, 98, 100, 101, 104}:
        return {
            "state": "NO_ONE_TO_ONE_COUNTERPART",
            "comparison": (
                "vLLM EP1 routed-expert path; SGLang EP72 uses DeepEP "
                "dispatch, a local expert runner, and DeepEP combine"
            ),
        }
    if order in {18, 20, 21}:
        return {
            "state": "BACKEND_DEPENDENT_SEMANTIC_COUNTERPART_ONLY",
            "comparison": "required MLA phase; selected SGLang backend sets its identity",
        }
    if order == 108:
        return {
            "state": "SEMANTIC_COUNTERPART_WITH_DP_LM_HEAD_REQUIRED",
            "comparison": "registered EP72 capture pins --enable-dp-lm-head",
        }
    return {
        "state": "SEMANTIC_COUNTERPART_PHYSICAL_IDENTITY_UNBOUND",
        "comparison": "required semantic phase; no captured SGLang EP72 physical identity",
    }


def build_inventory(kernel_rows: object, measured_service_ps: object) -> dict[str, Any]:
    """Enumerate every retained kernel row with an exhaustive family binding."""

    if not isinstance(kernel_rows, list) or len(kernel_rows) != 46:
        raise ValueError("CORE-65 requires exactly 46 retained kernel rows")
    inventory = []
    seen_orders = set()
    bucket_service: dict[str, int] = {}
    bucket_rows: dict[str, int] = {}
    for raw in kernel_rows:
        row = _mapping("kernel row", raw)
        order = _integer("first_launch_order", row.get("first_launch_order"))
        if order in seen_orders or order not in FAMILY_BY_ORDER:
            raise ValueError(f"duplicate or unmapped first launch order {order}")
        seen_orders.add(order)
        bucket, denominator, divisor = _bucket(order)
        count = Decimal(str(row.get("count_per_step")))
        service = Decimal(str(row.get("total_duration_per_step_ns"))) * 1000
        if service != service.to_integral_value():
            raise ValueError("kernel service is not integral picoseconds")
        per_layer = Fraction(str(count)) * divisor
        binding = _ep72_binding(order)
        output = {
            "captured_count_per_step": str(count),
            "captured_count_per_layer_denominator": denominator,
            "captured_count_per_layer_fraction": (f"{per_layer.numerator}/{per_layer.denominator}"),
            "captured_service_ps": int(service),
            "captured_service_share": f"{Decimal(str(row.get('share_of_step_compute'))):.12f}",
            "ep72_comparison": binding["comparison"],
            "ep72_physical_binding_state": binding["state"],
            "family": FAMILY_BY_ORDER[order],
            "first_launch_order": order,
            "graph_record_count": row.get("graph_record_count"),
            "name": row.get("name"),
            "record_count": row.get("record_count"),
            "stream_bucket": bucket,
        }
        inventory.append(output)
        bucket_service[bucket] = bucket_service.get(bucket, 0) + int(service)
        bucket_rows[bucket] = bucket_rows.get(bucket, 0) + 1
    expected_orders = (
        STEP_ORDERS | FIRST_ORDERS | COMMON_ORDERS | LATER_ORDERS | DENSE_ORDERS | MOE_ORDERS
    )
    if seen_orders != expected_orders:
        raise ValueError("CORE-65 kernel order inventory is not total")
    total_service = sum(row["captured_service_ps"] for row in inventory)
    if total_service != _integer("measured_service_ps", measured_service_ps, minimum=1):
        raise ValueError("CORE-65 inventory does not reconstruct measured service")
    return {
        "bucket_row_counts": dict(sorted(bucket_rows.items())),
        "bucket_service_ps": dict(sorted(bucket_service.items())),
        "captured_framework": "vllm-0.27.1+cu129",
        "captured_parallelism": "TP1_DP1_EP1_PP1",
        "coverage": "TOTAL_46_OF_46_ROWS_NO_UNMAPPED_KERNEL",
        "kernel_row_count": 46,
        "measured_service_ps": total_service,
        "rows": inventory,
        "schema": INVENTORY_SCHEMA,
    }


def _family_map(contract: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = contract.get("families")
    if not isinstance(rows, list):
        raise TypeError("model family inventory must be an array")
    return {str(_mapping("family", row)["id"]): _mapping("family", row) for row in rows}


def _model_analysis(model_fields: Mapping[str, Any]) -> dict[str, Any]:
    geometry = _mapping("geometry_symbols", model_fields.get("geometry_symbols"))
    contract = _mapping("inventory_contract", model_fields.get("inventory_contract"))
    projection = _mapping(
        "deployment_projection_contract", model_fields.get("deployment_projection_contract")
    )
    sanity = _mapping("physical_sanity", model_fields.get("physical_sanity"))
    if (geometry.get("L"), geometry.get("Ld"), geometry.get("Lm")) != (61, 3, 58):
        raise ValueError("DeepSeek layer schedule differs")
    families = _family_map(contract)
    attention_ids = (
        "mla_q_compression",
        "mla_q_decompression",
        "mla_kv_compression",
        "mla_kv_decompression",
        "mla_rotary_split",
        "mla_attention",
        "mla_output_projection",
    )
    attention = sum(
        int(families[family_id].get("static_hbm_bytes_per_layer", 0)) for family_id in attention_ids
    )
    dense = int(families["dense_early_mlp"]["static_hbm_bytes_per_layer"])
    router = int(families["moe_router"]["static_hbm_bytes_per_layer"])
    shared = int(families["moe_shared_expert"]["static_hbm_bytes_per_layer"])
    routed = int(families["moe_routed_experts"]["static_hbm_bytes_per_layer"])
    lm_head = int(families["lm_head"]["static_hbm_bytes"])
    captured_resident = 4 * attention + 3 * dense + router + shared + routed + lm_head
    ep72_static = int(sanity["ep72_physical_static_hbm_bytes_per_rank"])
    common = int(sanity["base_common_nonrouted_static_hbm_bytes"])
    per_expert_all_layers = int(projection["per_expert_static_hbm_bytes_all_base_moe_layers"])
    ep72_routed = 4 * per_expert_all_layers
    if common + ep72_routed != ep72_static:
        raise ValueError("EP72 static byte decomposition differs")
    return {
        "captured_layer_schedule": {
            "dense_layers": 3,
            "moe_layers": 1,
            "proof": (
                "effective num_hidden_layers=4 with first_k_dense_replace=3 and moe_layer_freq=1"
            ),
            "state": "DECIDED_THREE_DENSE_THEN_ONE_MOE",
        },
        "expert_population": {
            "assignment_tracked_compute_scale": "1/9_inherited_not_refit",
            "capture_logical_experts_resident_in_moe_layer": 256,
            "ep72_physical_slots_per_rank": 4,
            "expert_count_or_per_layer_resident_weight_scale": "1/64",
            "full_model_layer_adjusted_resident_routed_weight_scale": "29/32",
            "full_model_layer_adjusted_scale_formula": "58 * 4 / 256",
            "unique_active_expert_read_scale": "UNDECIDABLE_ROUTING_NOT_CAPTURED",
        },
        "static_weight_bytes": {
            "attention_per_layer": attention,
            "captured_reduced4_tp1_resident": captured_resident,
            "captured_reduced4_tp1_routed_all256_one_layer": routed,
            "dense_per_layer": dense,
            "ep72_common_nonrouted_per_rank": common,
            "ep72_declared_physical_per_rank": ep72_static,
            "ep72_routed_four_slots_all58_layers": ep72_routed,
            "lm_head": lm_head,
            "moe_router_per_layer": router,
            "moe_shared_expert_per_layer": shared,
            "naive_captured_resident_scaled_by_61_over_4": captured_resident * 61 // 4,
            "resident_bytes_are_not_per_step_reads": True,
            "actual_capture_per_step_read_bytes": "UNAVAILABLE",
            "actual_ep72_per_step_read_bytes": "UNAVAILABLE",
        },
    }


def _round_half_up(value: Fraction) -> int:
    decimal = Decimal(value.numerator) / Decimal(value.denominator)
    return int(decimal.quantize(Decimal(1), rounding=ROUND_HALF_UP))


def _conditional_diagnostic(inventory: Mapping[str, Any], fixed_ps: int) -> dict[str, Any]:
    services = _mapping("bucket_service_ps", inventory.get("bucket_service_ps"))
    step = Fraction(int(services["step_or_output_once"]))
    first = Fraction(int(services["first_layer_only"]))
    common = Fraction(int(services["all_layer_common"]))
    later = Fraction(int(services["later_layer_attention"]))
    dense = Fraction(int(services["dense_only"]))
    moe = Fraction(int(services["moe_only"]))
    routed = next(
        Fraction(row["captured_service_ps"])
        for row in inventory["rows"]
        if row["first_launch_order"] == 100
    )
    diagnostic = (
        fixed_ps
        + step
        - fixed_ps
        + first
        + Fraction(61, 4) * common
        + 20 * later
        + dense
        + 58 * (moe - routed)
        + Fraction(58, 9) * routed
    )
    prediction = Decimal(256 * 10**12) / (
        Decimal(diagnostic.numerator) / Decimal(diagnostic.denominator)
    )
    movement = prediction - Decimal("9544.657796")
    residual = (prediction - Decimal(22282)) / Decimal(22282) * 100
    return {
        "admissible_for_calibration": False,
        "formula": (
            "fixed + (step_once - fixed) + first_once + 61/4*common + "
            "20*later_variants + dense + 58*(moe-routed) + 58/9*routed"
        ),
        "prediction_movement_tokens_per_second_per_node": f"{movement:.6f}",
        "prediction_tokens_per_second_per_node": f"{prediction:.6f}",
        "rejection_reasons": [
            "row 11 aggregates attention, dense, and shared-expert quantization without per-launch durations",
            "the vLLM TP1 physical launches are not bound to SGLang EP72 launches",
            "DeepEP dispatch and combine are absent from the retained stream",
            "routing identities and actual HBM read bytes were not captured",
        ],
        "signed_residual_percent": f"{residual:.6f}",
        "step_ps": {
            "denominator": diagnostic.denominator,
            "numerator": diagnostic.numerator,
            "round_half_up": _round_half_up(diagnostic),
        },
    }


def build_hardware_remainder() -> dict[str, Any]:
    """Specify the literal EP72 cell and two-pass per-rank capture."""

    command = """export PYTHONPATH=\"$SIMLLM_SGLANG_SOURCE/python\"
export SIMLLM_CORE66_RUN_ROOT=\"$SIMLLM_WAVE_RUNS/core66/ep72-b32-c2000\"
nsys profile --force-overwrite=true --trace=cuda,nvtx,osrt,cublas --sample=none --gpu-metrics-device=all --trace-fork-before-exec=true --output \"$SIMLLM_CORE66_RUN_ROOT/node-$NODE_RANK\" \\
  \"$SIMLLM_SGLANG_PYTHON\" -m sglang.benchmark.one_batch \\
  --model-path \"$SIMLLM_DEEPSEEK_V3_LOCAL_MODEL\" \\
  --tp-size 72 --dp-size 72 --ep-size 72 \\
  --nnodes 9 --node-rank \"$NODE_RANK\" --dist-init-addr \"$MASTER_ADDR:$MASTER_PORT\" \\
  --enable-dp-attention --enable-dp-lm-head --moe-a2a-backend deepep \\
  --batch-size 32 --input-len 2000 --output-len 2 \\
  --profile --profile-stage decode --profile-start-step 0 --profile-steps 1 \\
  --profile-record-shapes --profile-activities CUDA_PROFILER \\
  --result-filename \"$SIMLLM_CORE66_RUN_ROOT/result-node-$NODE_RANK.jsonl\"
"""
    return {
        "capture_cell": "decode_b32_c2000_one_decode_iteration",
        "command": command,
        "configuration": {
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
        },
        "counter_pass": {
            "requirement": (
                "repeat the identical cell with rank-aware CUPTI or Nsight Compute "
                "application replay and dram__bytes_read.sum plus dram__bytes_write.sum"
            ),
            "why_separate": (
                "the stock one_batch profiler marker is not an all-rank HBM-counter capture"
            ),
        },
        "must_capture": [
            "all 72 rank identities and devices",
            "every CUDA launch name, order, duration, grid, block, stream, and correlation ID",
            "NVTX semantic phase and resolved attention, MoE, and LM-head backend",
            "per-layer routed expert IDs, assignment counts, and local physical slot IDs",
            "DeepEP dispatch and combine launches, peers, payload bytes, and durations",
            "actual per-kernel and per-step HBM read and write bytes",
            "fusion flags and the local resident weight byte inventory",
        ],
        "minimum_rank_classes": [
            "one of ranks 0-39 with four logical experts in four slots",
            "one of ranks 40-71 with three logical experts plus one redundant slot",
        ],
        "pinned_sglang": {
            "implementation": "DeepseekV2AttentionMLA_and_DeepseekV2MoE",
            "source_commit": "bfeae4e79a8dc4600e006f1a5fbc85321a01c1a3",
            "version": "0.5.19.dev345+gbfeae4e79",
        },
        "schema": REMAINDER_SCHEMA,
        "task": "CORE-66",
    }


def build_result(
    *,
    expectations: Mapping[str, Any],
    selected: Mapping[str, Any],
    access_events: Sequence[Mapping[str, Any]],
    capture_events: Sequence[Mapping[str, Any]],
    forbidden_access_ledger: object,
    preservation: Mapping[str, Any],
    expectations_commit: str,
    runner_commit: str,
    core66_free_on_base_main: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build the total inventory, null publication, and hardware remainder."""

    validate_expectations(expectations)
    access = validate_access_events(access_events)
    capture_access = validate_capture_events(capture_events)
    if not isinstance(forbidden_access_ledger, list) or len(forbidden_access_ledger) != 2:
        raise ValueError("CORE-65 forbidden-access incident ledger differs")
    if any(not row.get("protocol_violation") for row in forbidden_access_ledger):
        raise ValueError("CORE-65 forbidden-access incident is not explicit")
    inputs = _mapping("inputs", selected.get("inputs"))
    if _mapping("profile", selected.get("profile")).get("available") is not False:
        raise ValueError("unexpected original capture profile state")
    component = _mapping("component_basis", inputs.get("component_basis"))
    key = _mapping("component key", component.get("key"))
    parallelism = _mapping("capture parallelism", key.get("parallelism"))
    if parallelism != {
        "data_parallel": 1,
        "expert_parallel": 1,
        "pipeline_parallel": 1,
        "tensor_parallel": 1,
    }:
        raise ValueError("retained capture is not TP1/DP1/EP1/PP1")
    routing = _mapping("capture routing", key.get("routing"))
    if routing.get("availability") != "not-captured" or routing.get("expert_loads") is not None:
        raise ValueError("unexpected retained routing evidence")
    inventory = build_inventory(inputs.get("kernel_rows"), component.get("measured_service_ps"))
    model = _model_analysis(_mapping("model_fields", inputs.get("model_fields")))
    core64 = _mapping("core64", inputs.get("core64"))
    calibration = _mapping("calibration_only", core64.get("calibration_only"))
    classification = _mapping("component_classification", core64.get("component_classification"))
    decomposition = _mapping(
        "retained_physical_decomposition",
        classification.get("retained_physical_decomposition"),
    )
    fixed = _mapping("fixed_service_ps", decomposition.get("fixed_service_ps"))
    fixed_ps = _integer("fixed published ps", fixed.get("published_ps_round_half_up"))
    diagnostic = _conditional_diagnostic(inventory, fixed_ps)
    remainder = build_hardware_remainder()
    result = {
        "access": {
            **access,
            "capture_profile": capture_access,
            "forbidden_access_incident_count": len(forbidden_access_ledger),
            "forbidden_access_ledger_empty": False,
            "literal_protocol_satisfied": False,
        },
        "base_commit": expectations.get("base_commit"),
        "calibration_only": {
            "anchor_tokens_per_second_per_node": calibration["anchor_tokens_per_second_per_node"],
            "classification": "UNDERCORRECTION",
            "corrected_step_ps_round_half_up": calibration["corrected_step_ps_round_half_up"],
            "final_prediction_tokens_per_second_per_node": calibration[
                "final_prediction_tokens_per_second_per_node"
            ],
            "final_signed_residual_percent": calibration["final_signed_residual_percent"],
            "prediction_movement_tokens_per_second_per_node": "0.000000",
            "signed_difference_from_anchor_tokens_per_second_per_node": calibration[
                "signed_difference_from_anchor_tokens_per_second_per_node"
            ],
            "signed_residual_movement_percentage_points": "0.000000",
        },
        "candidate_verdicts": {
            "layer_type_composition": {
                "expected_direction": "dense_decrease_and_moe_increase_net_indeterminate",
                "finding": model["captured_layer_schedule"],
                "numeric_calibration_movement_admissible": False,
                "verdict": "DECIDED_THREE_DENSE_PLUS_ONE_MOE",
            },
            "expert_population": {
                "expected_direction": "decrease_for_count_or_weight_bytes",
                "finding": model["expert_population"],
                "numeric_calibration_movement_admissible": False,
                "verdict": "COMPUTE_AND_MEMORY_TRACKING_VARIABLES_NOT_INTERCHANGEABLE",
            },
            "weight_read_volume": {
                "expected_direction": "decrease_if_capture_reads_more_increase_if_less",
                "finding": model["static_weight_bytes"],
                "numeric_calibration_movement_admissible": False,
                "verdict": "UNDECIDABLE_NO_CAPTURE_OR_EP72_HBM_COUNTERS",
            },
            "other_kernel_counterparts": {
                "expected_direction": "capture_only_decrease_ep72_required_absent_increase",
                "finding": {
                    "capture_only_vllm_routed_rows": [96, 98, 100, 101, 104],
                    "ep72_required_absent_families": [
                        "deepep_dispatch_a",
                        "deepep_dispatch_b",
                        "deepep_combine_a",
                        "deepep_combine_b",
                    ],
                },
                "numeric_calibration_movement_admissible": False,
                "verdict": "DECIDED_IDENTITY_MISMATCH_SERVICE_UNDECIDABLE",
            },
        },
        "conditional_frequency_only_diagnostic": diagnostic,
        "expectations_commit": expectations_commit,
        "hardware_remainder": {
            "artifact": "examples/deployment_curve_v1/core66_hardware_remainder.json",
            "task": "CORE-66",
        },
        "inventory_summary": {name: value for name, value in inventory.items() if name != "rows"},
        "layer_composition_arithmetic": {
            "all_layer_common_multiplier": "61/4",
            "dense_correct_multiplier": "3/3=1",
            "dense_naive_overpricing_multiplier_relative_to_correct": "61/4",
            "moe_correct_multiplier": "58/1=58",
            "moe_naive_underpricing_multiplier_relative_to_correct": "61/232",
            "net_service_direction": "INDETERMINATE_WITHOUT_COMPONENT_SERVICE_SPLIT",
            "step_or_output_correct_multiplier": "1",
            "step_or_output_naive_overpricing_multiplier_relative_to_correct": "61/4",
        },
        "physical_binding": {
            "capture": "vLLM_TP1_DP1_EP1_PP1_reduced4",
            "capture_original_profile_available": False,
            "coverage": inventory["coverage"],
            "ep72_target": "SGLang_DP72_EP72_DeepEP_3_dense_plus_58_MoE",
            "state": "TOTAL_RETAINED_INVENTORY_BUT_NOT_TOTAL_EP72_PHYSICAL_BINDING",
        },
        "preservation_lock": dict(preservation),
        "registry_disposition": {
            "core65": "REMAINS_OPEN_PROTOCOL_VOID_AND_PHYSICAL_SERVICE_UNRESOLVED",
            "exact_remainder_id": "CORE-66",
            "reserved_id_free_on_base_main": core66_free_on_base_main,
        },
        "runner_commit": runner_commit,
        "schema": RESULT_SCHEMA,
        "scope": {
            "calibration_constant_fitted": False,
            "decode_overlap_term_added": False,
            "held_out_mtp_numeric_value_compared": False,
            "held_out_mtp_numeric_value_used_in_arithmetic": False,
            "model_weights_downloaded": False,
            "scored_run_performed": False,
            "web_pages_fetched": False,
        },
        "source_pins": {
            "sglang": remainder["pinned_sglang"],
            "vllm": {
                "deepseek_source_sha256": (
                    "f22d4458a604875fbc3fb194119519734b3f2b2460276ef2091419ed6988f1be"
                ),
                "source_commit": "6e448d0ea9bf3d88d898b65449ca6dc2aec170ac",
            },
        },
        "status": "PROTOCOL_VOID_NULL_MOVEMENT_EXACT_HARDWARE_REMAINDER",
        "task": "CORE-65",
    }
    return inventory, result, remainder


def render_markdown(result: Mapping[str, Any], inventory: Mapping[str, Any]) -> str:
    """Render the total inventory first, followed by verdicts and movement."""

    table_rows = []
    for row in inventory["rows"]:
        name = html.escape(str(row["name"])).replace("|", "&#124;")
        share = Decimal(row["captured_service_share"]) * 100
        table_rows.append(
            "| {order} | <code>{name}</code> | {family} | {count} | {per_layer} | "
            "{share:.6f}% | {state} |".format(
                order=row["first_launch_order"],
                name=name,
                family=row["family"],
                count=row["captured_count_per_step"],
                per_layer=row["captured_count_per_layer_fraction"],
                share=share,
                state=row["ep72_physical_binding_state"],
            )
        )
    calibration = result["calibration_only"]
    diagnostic = result["conditional_frequency_only_diagnostic"]
    weight = result["candidate_verdicts"]["weight_read_volume"]["finding"]
    return f"""# CORE-65 retained kernel inventory and EP72 physical binding

Status: **{result["status"]}**. The retained stream is enumerated totally, but
it cannot be bound totally to SGLang EP72 physical launches. CORE-65 therefore
publishes a null calibration movement and registers the literal EP72 capture as
CORE-66.

## Total retained kernel inventory and EP72 comparison

All **{inventory["kernel_row_count"]} of {inventory["kernel_row_count"]}**
retained rows are named below. Their services sum exactly to
**{inventory["measured_service_ps"]:,} ps**. No row is unmapped.

| Order | Retained physical kernel | Family | Count/step | Count/layer basis | Service share | EP72 binding |
| ---: | --- | --- | ---: | ---: | ---: | --- |
{chr(10).join(table_rows)}

The stream is vLLM TP1/DP1/EP1/PP1. The real target is SGLang DP72/EP72 with
DeepEP. Orders 96, 98, 100, 101 and 104 are vLLM EP1 routed-expert scheduling,
compute, activation and sum rows; they have no one-to-one physical SGLang
DeepEP counterpart. Conversely, real EP72 requires DeepEP dispatch A/B and
combine A/B launches that are absent from this noncollective stream. Attention
and LM-head semantic work exists in both, but their SGLang physical identities
remain backend-dependent and uncaptured.

## Candidate verdicts

1. **Layer-type composition: decided.** The effective capture has four layers,
   and the pinned vLLM rule with `first_k_dense_replace=3` and frequency one
   makes them exactly three dense layers followed by one MoE layer. Common work
   legitimately scales by `61/4`; dense-only work scales by `3/3 = 1`, not
   `61/4`; MoE-only work scales by `58/1 = 58`, not `61/4`; step/output work
   stays once. Thus naive depth scaling overprices dense and step/output work,
   underprices MoE work, and has no preregisterable net sign without a valid
   component service split.
2. **Expert population: partly decided, service movement undecidable.** The
   retained MoE layer is resident over 256 logical experts, while an EP72 rank
   has four physical slots. A per-layer expert-count or resident-weight term
   scales by `4/256 = 1/64`; after replacing one captured MoE layer by 58 real
   MoE layers, its full-model resident routed-weight ratio is `58*4/256 =
   29/32`. The inherited `1/9` remains only on the previously classified
   assignment-tracked `fused_moe_kernel` service. It is not silently reused as
   an expert-count or weight-byte scale. Routing identities are absent, so the
   unique active expert weights actually read are undecidable.
3. **Weight-read volume: undecidable.** The reduced TP1 model's static resident
   inventory is {weight["captured_reduced4_tp1_resident"]:,} bytes; naive
   `61/4` scaling gives
   {weight["naive_captured_resident_scaled_by_61_over_4"]:,} bytes. The declared
   EP72 per-rank inventory is {weight["ep72_declared_physical_per_rank"]:,}
   bytes: {weight["ep72_common_nonrouted_per_rank"]:,} common bytes plus
   {weight["ep72_routed_four_slots_all58_layers"]:,} routed bytes. Static
   residency is not a per-step read count. The retained record has no HBM
   counter attribution and no routing trace, so neither side's actual bytes
   read per step is known.
4. **Other counterparts: identity mismatch decided, service undecidable.** The
   five vLLM routed rows above are capture-only physical identities and point
   downward if removed. Missing DeepEP dispatch/combine points upward. Exact
   services and overlap are absent, so neither is priced.

## Conditional arithmetic, rejected for calibration

For audit only, a frequency-only regrouping gives
**{diagnostic["step_ps"]["numerator"]}/{diagnostic["step_ps"]["denominator"]} ps**
({diagnostic["step_ps"]["round_half_up"]:,} ps), predicting
**{diagnostic["prediction_tokens_per_second_per_node"]} tokens/s/node**, a
movement of **{diagnostic["prediction_movement_tokens_per_second_per_node"]}
tokens/s/node**. It is rejected because row 11 mixes attention, dense and
shared-expert quantization without per-launch durations; the vLLM launches are
not physically bound to SGLang EP72; DeepEP communication is missing; and
routing plus actual HBM reads were not captured. It is not the publication.

## Published signed movement

```text
CORE-65 prediction movement = {calibration["prediction_movement_tokens_per_second_per_node"]} tokens/s/node
final standard-decode prediction = {calibration["final_prediction_tokens_per_second_per_node"]} tokens/s/node
calibration anchor = {calibration["anchor_tokens_per_second_per_node"]} tokens/s/node
signed difference = {calibration["signed_difference_from_anchor_tokens_per_second_per_node"]} tokens/s/node
signed residual movement = {calibration["signed_residual_movement_percentage_points"]} percentage points
final signed residual = {calibration["final_signed_residual_percent"]} percent
```

No parameter was fitted, no overlap term was added, and no held-out MTP value
was used or compared.

## Protocol and registry disposition

The committed reader made eight partial, contemporaneously logged accesses;
the optional original profile was logged unavailable without reading bytes.
However, two pre-reader incidents make the forbidden-access ledger nonempty:
one held-out numeric exposure, redacted and unused, and one unlogged broad
registry inspection. Literal CORE-65 closure is therefore impossible in this
worker even independently of the missing physical evidence.

CORE-65 remains open. CORE-66 is the exact hardware remainder: run the pinned
SGLang EP72 `b32/c2000`, MTP-disabled cell across all 72 ranks, capturing both
expert-residency rank classes, every launch and semantic correlation, routing
and physical slot identities, DeepEP payloads, and per-kernel HBM read/write
bytes. The exact configuration and command are in
`core66_hardware_remainder.json`.
"""


def render_hardware_remainder(remainder: Mapping[str, Any]) -> str:
    """Render the exact CORE-66 EP72 capture contract."""

    requirements = "\n".join(f"- {item}" for item in remainder["must_capture"])
    rank_classes = "\n".join(f"- {item}" for item in remainder["minimum_rank_classes"])
    return f"""# CORE-66 EP72 physical capture remainder

Run the pinned SGLang source at commit
`{remainder["pinned_sglang"]["source_commit"]}` with the pre-staged local
DeepSeek-V3 model. Do not download weights. The cell is DP72/EP72, batch 32
and KV length 2,000 per rank, MTP disabled, and one measured decode iteration.

## Required rank coverage

{rank_classes}

The preferred capture covers all 72 ranks, not only these representatives.

## Kernel-trace command

Run the following once on each of nine nodes with node-specific `NODE_RANK`.

```bash
{remainder["command"]}
```

## Required evidence

{requirements}

Repeat the identical cell with rank-aware CUPTI or Nsight Compute application
replay and `dram__bytes_read.sum` plus `dram__bytes_write.sum`. The stock
`one_batch` profiler marker is not by itself an all-rank HBM-counter capture;
the counter pass must instrument every child rank and preserve rank identity.
"""


def write_new_json(path: Path, value: object) -> None:
    """Write one new JSON artifact with pinned POSIX newlines."""

    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_new_text(path: Path, value: str) -> None:
    """Write one new text artifact with pinned POSIX newlines."""

    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(value)


__all__ = [
    "build_hardware_remainder",
    "build_inventory",
    "build_result",
    "render_hardware_remainder",
    "render_markdown",
    "validate_access_events",
    "validate_capture_events",
    "validate_expectations",
    "verify_preservation",
    "write_new_json",
    "write_new_text",
]
