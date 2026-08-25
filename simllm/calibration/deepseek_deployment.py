"""Exact rank-class projections for disclosed DeepSeek-V3 deployments."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .canonical import canonical_sha256
from .model_inventory import ModelKernelInventory

DEEPSEEK_DEPLOYMENT_PROJECTION_SCHEMA = (
    "simllm-deepseek-deployment-projection-v1"
)
_BASE_ROUTED_FAMILY = "moe_routed_experts"
_MTP_FAMILY = "multi_token_prediction_head"
_COMPRESSED_KV_READ_FAMILY = "mla_compressed_kv_read"


def _positive(value: object, path: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{path}: expected a positive integer")
    return value


def _rank_classes(value: object, path: str) -> tuple[dict[str, int], ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path}: expected a nonempty rank-class array")
    result = []
    for index, item in enumerate(value):
        item_path = f"{path}[{index}]"
        expected = {
            "rank_count",
            "logical_experts_per_rank",
            "redundant_slots_per_rank",
            "physical_slots_per_rank",
        }
        if not isinstance(item, dict) or set(item) != expected:
            raise ValueError(f"{item_path}: invalid rank-class fields")
        row = {
            "rank_count": _positive(item["rank_count"], f"{item_path}.rank_count"),
            "logical_experts_per_rank": _positive(
                item["logical_experts_per_rank"],
                f"{item_path}.logical_experts_per_rank",
            ),
            "redundant_slots_per_rank": item["redundant_slots_per_rank"],
            "physical_slots_per_rank": _positive(
                item["physical_slots_per_rank"],
                f"{item_path}.physical_slots_per_rank",
            ),
        }
        if (
            type(row["redundant_slots_per_rank"]) is not int
            or row["redundant_slots_per_rank"] < 0
            or row["logical_experts_per_rank"]
            + row["redundant_slots_per_rank"]
            != row["physical_slots_per_rank"]
        ):
            raise ValueError(f"{item_path}: invalid redundancy arithmetic")
        result.append(row)
    return tuple(result)


def _projection_by_family(case: Any) -> dict[str, Any]:
    return {item.family_id: item for item in case.kernel_projections}


def _family_projection_obj(item: Any) -> dict[str, Any]:
    return {
        "family_id": item.family_id,
        "shape_vector": item.shape_vector.to_obj(),
        "logical_visit_count": item.logical_launch_count,
        "aggregate_flops_per_rank": item.aggregate_flops,
        "aggregate_hbm_bytes_per_rank": item.aggregate_hbm_bytes,
    }


def _static_rank_classes(
    classes: tuple[dict[str, int], ...],
    *,
    base_common_bytes: int,
    per_expert_base_bytes: int,
    mtp_common_bytes: int,
    per_expert_mtp_bytes: int,
) -> list[dict[str, Any]]:
    rows = []
    for index, item in enumerate(classes):
        unique = item["logical_experts_per_rank"]
        physical = item["physical_slots_per_rank"]
        rows.append(
            {
                "class_id": f"rank-class-{index}",
                **item,
                "base": {
                    "common_hbm_bytes_per_rank": base_common_bytes,
                    "logical_routed_hbm_bytes_per_rank": (
                        unique * per_expert_base_bytes
                    ),
                    "physical_routed_hbm_bytes_per_rank": (
                        physical * per_expert_base_bytes
                    ),
                    "logical_total_hbm_bytes_per_rank": (
                        base_common_bytes + unique * per_expert_base_bytes
                    ),
                    "physical_total_hbm_bytes_per_rank": (
                        base_common_bytes + physical * per_expert_base_bytes
                    ),
                },
                "mtp_when_enabled": {
                    "common_hbm_bytes_per_rank": mtp_common_bytes,
                    "logical_routed_hbm_bytes_per_rank": (
                        unique * per_expert_mtp_bytes
                    ),
                    "physical_routed_hbm_bytes_per_rank": (
                        physical * per_expert_mtp_bytes
                    ),
                    "logical_total_hbm_bytes_per_rank": (
                        mtp_common_bytes + unique * per_expert_mtp_bytes
                    ),
                    "physical_total_hbm_bytes_per_rank": (
                        mtp_common_bytes + physical * per_expert_mtp_bytes
                    ),
                },
            }
        )
    return rows


def _case_projection(
    case: Any,
    *,
    expert_parallel: int,
    logical_experts: int,
    classes: tuple[dict[str, int], ...],
    top_k: int,
    moe_layers: int,
    per_expert_layer_flops: int,
    per_expert_base_bytes: int,
    per_expert_mtp_bytes: int,
) -> dict[str, Any]:
    by_family = _projection_by_family(case)
    routed = by_family[_BASE_ROUTED_FAMILY]
    mtp = by_family[_MTP_FAMILY]
    new_tokens = routed.shape_vector.values[0]
    global_routed_visits = expert_parallel * new_tokens * top_k
    if global_routed_visits % logical_experts:
        raise ValueError(
            f"case {case.case_id!r}: routed visits do not divide unique experts"
        )
    visits_per_expert = global_routed_visits // logical_experts
    per_expert_base_flops = moe_layers * per_expert_layer_flops
    expected_local_routed = new_tokens * top_k * per_expert_base_flops
    if routed.aggregate_flops != expected_local_routed:
        raise ValueError(f"case {case.case_id!r}: routed family FLOPs changed")
    invariant = [
        _family_projection_obj(item)
        for item in case.kernel_projections
        if item.family_id not in {_BASE_ROUTED_FAMILY, _MTP_FAMILY}
    ]
    invariant_base_flops = sum(
        item["aggregate_flops_per_rank"] for item in invariant
    )
    invariant_base_bytes = sum(
        item["aggregate_hbm_bytes_per_rank"] for item in invariant
    )
    mtp_enabled = mtp.logical_launch_count
    if mtp_enabled not in {0, 1}:
        raise ValueError(f"case {case.case_id!r}: invalid MTP visit count")
    mtp_routed_local_flops = (
        mtp_enabled * new_tokens * top_k * per_expert_layer_flops
    )
    mtp_common_flops = mtp.aggregate_flops - mtp_routed_local_flops
    mtp_common_bytes = mtp.aggregate_hbm_bytes - (
        mtp_enabled * logical_experts * per_expert_mtp_bytes
    )
    if mtp_common_flops < 0 or mtp_common_bytes < 0:
        raise ValueError(f"case {case.case_id!r}: invalid MTP routed split")
    rank_rows = []
    for index, item in enumerate(classes):
        unique = item["logical_experts_per_rank"]
        physical = item["physical_slots_per_rank"]
        base_routed_flops = unique * visits_per_expert * per_expert_base_flops
        mtp_routed_flops = (
            mtp_enabled * unique * visits_per_expert * per_expert_layer_flops
        )
        rank_rows.append(
            {
                "class_id": f"rank-class-{index}",
                "rank_count": item["rank_count"],
                "logical_experts_per_rank": unique,
                "redundant_slots_per_rank": item["redundant_slots_per_rank"],
                "physical_slots_per_rank": physical,
                "base_routed_visits_per_rank_per_moe_layer": (
                    unique * visits_per_expert
                ),
                "base_routed_flops_per_rank": base_routed_flops,
                "base_logical_routed_hbm_bytes_per_rank": (
                    unique * per_expert_base_bytes
                ),
                "base_physical_routed_hbm_bytes_per_rank": (
                    physical * per_expert_base_bytes
                ),
                "base_total_flops_per_rank": (
                    invariant_base_flops + base_routed_flops
                ),
                "base_logical_hbm_bytes_per_rank": (
                    invariant_base_bytes + unique * per_expert_base_bytes
                ),
                "base_physical_hbm_bytes_per_rank": (
                    invariant_base_bytes + physical * per_expert_base_bytes
                ),
                "mtp_routed_flops_per_rank": mtp_routed_flops,
                "mtp_logical_routed_hbm_bytes_per_rank": (
                    mtp_enabled * unique * per_expert_mtp_bytes
                ),
                "mtp_physical_routed_hbm_bytes_per_rank": (
                    mtp_enabled * physical * per_expert_mtp_bytes
                ),
                "mtp_total_flops_per_rank": (
                    mtp_common_flops + mtp_routed_flops
                ),
                "mtp_logical_hbm_bytes_per_rank": (
                    mtp_common_bytes
                    + mtp_enabled * unique * per_expert_mtp_bytes
                ),
                "mtp_physical_hbm_bytes_per_rank": (
                    mtp_common_bytes
                    + mtp_enabled * physical * per_expert_mtp_bytes
                ),
            }
        )
    rank_base_flops = sum(
        row["rank_count"] * row["base_total_flops_per_rank"]
        for row in rank_rows
    )
    reference_base_flops = expert_parallel * (
        invariant_base_flops + routed.aggregate_flops
    )
    rank_mtp_flops = sum(
        row["rank_count"] * row["mtp_total_flops_per_rank"]
        for row in rank_rows
    )
    reference_mtp_flops = expert_parallel * mtp.aggregate_flops
    if rank_base_flops != reference_base_flops:
        raise ValueError(f"case {case.case_id!r}: base rank FLOPs do not conserve")
    if rank_mtp_flops != reference_mtp_flops:
        raise ValueError(f"case {case.case_id!r}: MTP rank FLOPs do not conserve")
    return {
        "case_id": case.case_id,
        "phase": case.phase,
        "new_tokens_per_rank": new_tokens,
        "global_routed_visits_per_moe_layer": global_routed_visits,
        "routed_visits_per_unique_expert_per_moe_layer": visits_per_expert,
        "rank_invariant_family_projections": invariant,
        "rank_classes": rank_rows,
        "conservation": {
            "reference_base_flops": reference_base_flops,
            "rank_class_base_flops": rank_base_flops,
            "reference_mtp_flops": reference_mtp_flops,
            "rank_class_mtp_flops": rank_mtp_flops,
        },
    }


def build_deepseek_deployment_projection(
    suite: Mapping[str, Any],
    inventory: ModelKernelInventory,
) -> dict[str, Any]:
    """Build one framework-neutral exact projection from a total inventory."""

    if suite.get("suite") != inventory.suite.suite_id:
        raise ValueError("deployment suite and inventory identity differ")
    model = suite.get("reference_model")
    if not isinstance(model, dict) or model.get("name") != "deepseek-ai/DeepSeek-V3":
        raise ValueError("deployment projection requires the pinned DeepSeek-V3 model")
    stack = model.get("deepseek_stack")
    if not isinstance(stack, dict):
        raise TypeError("deployment projection requires a DeepSeek stack")
    logical_experts = inventory.model.geometry.num_experts
    top_k = inventory.model.geometry.top_k
    moe_layers = inventory.model.geometry.layers - stack["first_k_dense_replace"]
    cases = {case.case_id: case for case in inventory.cases}
    first_case = inventory.cases[0]
    first_by_family = _projection_by_family(first_case)
    routed = first_by_family[_BASE_ROUTED_FAMILY]
    per_expert_base_bytes, remainder = divmod(
        routed.aggregate_hbm_bytes, logical_experts
    )
    if remainder:
        raise ValueError("base routed weights do not divide unique experts")
    new_tokens = routed.shape_vector.values[0]
    per_expert_base_flops, remainder = divmod(
        routed.aggregate_flops, new_tokens * top_k
    )
    if remainder or per_expert_base_flops % moe_layers:
        raise ValueError("base routed FLOPs do not divide expert dispatches")
    per_expert_layer_flops = per_expert_base_flops // moe_layers
    per_expert_mtp_bytes = per_expert_base_bytes // moe_layers
    if per_expert_mtp_bytes * moe_layers != per_expert_base_bytes:
        raise ValueError("base routed bytes do not divide MoE layers")
    enabled_cases = [
        case
        for case in inventory.cases
        if _projection_by_family(case)[_MTP_FAMILY].logical_launch_count == 1
    ]
    if len(enabled_cases) != 1:
        raise ValueError("deployment projection requires one enabled MTP case")
    enabled_mtp = _projection_by_family(enabled_cases[0])[_MTP_FAMILY]
    mtp_common_bytes = (
        enabled_mtp.aggregate_hbm_bytes
        - logical_experts * per_expert_mtp_bytes
    )
    base_common_bytes = sum(
        item.aggregate_hbm_bytes
        for item in first_case.kernel_projections
        if item.family_id
        not in {
            _BASE_ROUTED_FAMILY,
            _MTP_FAMILY,
            _COMPRESSED_KV_READ_FAMILY,
        }
    )
    declarations = suite.get("deployment_projections")
    if not isinstance(declarations, list) or len(declarations) != 4:
        raise ValueError("suite must declare four deployment projections")
    units = []
    for unit_index, declaration in enumerate(declarations):
        if not isinstance(declaration, dict):
            raise TypeError(
                f"deployment_projections[{unit_index}] is not an object"
            )
        expert_parallel = _positive(
            declaration["expert_parallel"],
            f"deployment_projections[{unit_index}].expert_parallel",
        )
        classes = _rank_classes(
            declaration["rank_classes"],
            f"deployment_projections[{unit_index}].rank_classes",
        )
        logical_sum = sum(
            row["rank_count"] * row["logical_experts_per_rank"]
            for row in classes
        )
        physical_sum = sum(
            row["rank_count"] * row["physical_slots_per_rank"]
            for row in classes
        )
        if logical_sum != logical_experts or physical_sum != 288:
            raise ValueError(f"deployment {declaration['id']!r} expert sums changed")
        dynamic_claimed = declaration["id"].startswith("sglang-")
        case_rows = []
        if dynamic_claimed:
            case_rows = [
                _case_projection(
                    cases[case_id],
                    expert_parallel=expert_parallel,
                    logical_experts=logical_experts,
                    classes=classes,
                    top_k=top_k,
                    moe_layers=moe_layers,
                    per_expert_layer_flops=per_expert_layer_flops,
                    per_expert_base_bytes=per_expert_base_bytes,
                    per_expert_mtp_bytes=per_expert_mtp_bytes,
                )
                for case_id in declaration["source_case_ids"]
            ]
        units.append(
            {
                "id": declaration["id"],
                "disclosure": declaration["disclosure"],
                "phase": declaration.get("phase", declaration.get("source_phase")),
                "expert_parallel": expert_parallel,
                "attention_parallelism": declaration["attention_parallelism"],
                "source_case_ids": declaration["source_case_ids"],
                "dynamic_projection_state": (
                    "exact-disclosed-workload"
                    if dynamic_claimed
                    else "not-claimed-no-disclosed-workload-shape"
                ),
                "static_rank_classes": _static_rank_classes(
                    classes,
                    base_common_bytes=base_common_bytes,
                    per_expert_base_bytes=per_expert_base_bytes,
                    mtp_common_bytes=mtp_common_bytes,
                    per_expert_mtp_bytes=per_expert_mtp_bytes,
                ),
                "case_projections": case_rows,
            }
        )
    neutral = inventory.to_obj()
    neutral.pop("framework")
    neutral["implementation_identity"].pop("join_tasks")
    return {
        "schema": DEEPSEEK_DEPLOYMENT_PROJECTION_SCHEMA,
        "suite": {
            "id": inventory.suite.suite_id,
            "sha256": inventory.suite.suite_sha256,
        },
        "model": inventory.model.to_obj(),
        "inventory_structure_sha256": canonical_sha256(neutral),
        "expert_contract": {
            "logical_experts": logical_experts,
            "redundant_physical_slots": 32,
            "physical_slots": 288,
            "top_k": top_k,
            "base_moe_layers": moe_layers,
            "per_expert_layer_flops": per_expert_layer_flops,
            "per_expert_base_static_hbm_bytes": per_expert_base_bytes,
            "per_expert_mtp_static_hbm_bytes": per_expert_mtp_bytes,
            "redundancy_rule": (
                "physical residency only; duplicate slots create no logical work"
            ),
        },
        "units": units,
    }


__all__ = [
    "DEEPSEEK_DEPLOYMENT_PROJECTION_SCHEMA",
    "build_deepseek_deployment_projection",
]
