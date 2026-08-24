from __future__ import annotations

from dataclasses import replace

import pytest

from simllm.backends.step_lowerer import SerialStepLowerer, SerialStepLowererConfig
from simllm.calibration.canonical import canonical_loads
from simllm.calibration.graph_identity import (
    EXECUTION_GRAPH_TEMPLATE_SCHEMA,
    GraphIdentityError,
    execution_graph_template_record,
    unbound_execution_graph_record,
)
from simllm.compute import ModelDims
from simllm.core import (
    ComputeWork,
    ControlMode,
    ControlWork,
    DmaWork,
    ExecutionGraph,
    ExecutionOperation,
    KvCacheAction,
    KvCacheWork,
    RequestPhase,
    ScheduledRequest,
    StepRecord,
)


def _variant_graph() -> ExecutionGraph:
    operations = (
        ExecutionOperation(
            operation_id="compute-a",
            rank=7,
            logical_queue="queue-a",
            work=ComputeWork(
                kernel="gemm",
                config=(("m", 8),),
                flops=100,
                hbm_bytes=20,
                nominal_duration_ps=30,
                uncertainty_fraction=0.25,
            ),
        ),
        ExecutionOperation(
            operation_id="kv-a",
            rank=7,
            logical_queue="queue-a",
            work=KvCacheWork(
                action=KvCacheAction.READ,
                pool_id="pool-a",
                request_id="request-a",
                block_ids=("block-a",),
                byte_count=64,
            ),
            depends_on=("compute-a",),
        ),
        ExecutionOperation(
            operation_id="dma-a",
            rank=7,
            logical_queue="queue-b",
            work=DmaWork(
                descriptor_id="descriptor-a",
                source="gpu:7:hbm",
                destination="host:pinned",
                byte_count=64,
            ),
            participant_local_depends_on=("kv-a",),
        ),
        ExecutionOperation(
            operation_id="control-a",
            rank=7,
            logical_queue="queue-c",
            work=ControlWork(
                message="barrier",
                destination_ranks=(9,),
                payload_bytes=8,
                mode=ControlMode.SYNCHRONOUS,
            ),
            depends_on=("dma-a",),
            priority=2,
        ),
    )
    return ExecutionGraph(
        execution_id="graph-a",
        step_index=3,
        released_at_ps=10,
        operations=operations,
        completion_operation_ids=("control-a",),
    )


def test_unbound_instance_graph_nulls_service_but_preserves_exact_graph() -> None:
    record = unbound_execution_graph_record(_variant_graph())
    value = canonical_loads(record.canonical)
    compute = value["operations"][0]["work"]
    assert compute["nominal_duration_ps"] is None
    assert compute["uncertainty_fraction"] is None
    assert compute["config"] == [["m", 8]]
    assert compute["flops"] == 100
    assert compute["hbm_bytes"] == 20


def test_unbound_instance_graph_rejects_float_config() -> None:
    graph = _variant_graph()
    operation = graph.operations[0]
    changed = replace(
        graph,
        operations=(
            replace(operation, work=replace(operation.work, config=(("ratio", 0.5),))),
            *graph.operations[1:],
        ),
    )
    with pytest.raises(GraphIdentityError, match="cannot contain a float"):
        unbound_execution_graph_record(changed)


def test_template_projects_all_noncollective_work_variants_exactly() -> None:
    record = execution_graph_template_record(_variant_graph())
    value = canonical_loads(record.canonical)
    assert value["schema"] == EXECUTION_GRAPH_TEMPLATE_SCHEMA
    assert value["completion_operation_ordinals"] == [3]
    assert value["collective_plans"] == []
    assert [operation["work"] for operation in value["operations"]] == [
        {"kind": "compute", "kernel": "gemm"},
        {"kind": "kv-cache", "action": "read"},
        {
            "kind": "dma",
            "source_role": "gpu:0:hbm",
            "destination_role": "host:pinned",
        },
        {
            "kind": "control",
            "mode": "synchronous",
            "message": "barrier",
            "destination_rank_ordinals": [1],
        },
    ]
    assert [operation["logical_queue_ordinal"] for operation in value["operations"]] == [
        0,
        0,
        1,
        2,
    ]


def test_template_is_invariant_to_excluded_instance_fields_and_identity_renames() -> None:
    original = _variant_graph()
    operations = []
    rename = {
        "compute-a": "x0",
        "kv-a": "x1",
        "dma-a": "x2",
        "control-a": "x3",
    }
    for operation in original.operations:
        work = operation.work
        if isinstance(work, ComputeWork):
            work = replace(
                work,
                config=(("different", 99),),
                flops=999,
                hbm_bytes=888,
                nominal_duration_ps=777,
                uncertainty_fraction=0.5,
            )
        operations.append(
            replace(
                operation,
                operation_id=rename[operation.operation_id],
                rank={7: 70, 9: 90}[operation.rank],
                logical_queue={
                    "queue-a": "renamed-a",
                    "queue-b": "renamed-b",
                    "queue-c": "renamed-c",
                }[operation.logical_queue],
                work=(
                    replace(work, source="gpu:70:hbm")
                    if isinstance(work, DmaWork)
                    else replace(work, destination_ranks=(90,))
                    if isinstance(work, ControlWork)
                    else work
                ),
                depends_on=tuple(rename[item] for item in operation.depends_on),
                participant_local_depends_on=tuple(
                    rename[item] for item in operation.participant_local_depends_on
                ),
                not_before_ps=12345,
                placement_epoch=6,
            )
        )
    changed = replace(
        original,
        execution_id="renamed",
        step_index=99,
        released_at_ps=888,
        operations=tuple(operations),
        completion_operation_ids=("x3",),
    )
    assert (
        execution_graph_template_record(changed).record_id
        == execution_graph_template_record(original).record_id
    )


def test_template_changes_when_retained_priority_or_dependency_changes() -> None:
    original = _variant_graph()
    priority = replace(
        original,
        operations=(
            *original.operations[:-1],
            replace(original.operations[-1], priority=3),
        ),
    )
    dependency = replace(
        original,
        operations=(
            original.operations[0],
            replace(original.operations[1], depends_on=()),
            *original.operations[2:],
        ),
    )
    identity = execution_graph_template_record(original).record_id
    assert execution_graph_template_record(priority).record_id != identity
    assert execution_graph_template_record(dependency).record_id != identity


def test_empty_completion_normalizes_to_explicit_all() -> None:
    graph = _variant_graph()
    all_ids = tuple(operation.operation_id for operation in graph.operations)
    empty = replace(graph, completion_operation_ids=())
    explicit = replace(graph, completion_operation_ids=tuple(reversed(all_ids)))
    assert (
        execution_graph_template_record(empty).record_id
        == execution_graph_template_record(explicit).record_id
    )


def test_template_rejects_unknown_dma_endpoint_roles() -> None:
    graph = _variant_graph()
    dma = graph.operations[2]
    changed = replace(
        graph,
        operations=(
            *graph.operations[:2],
            replace(dma, work=replace(dma.work, source="device-seven")),
            *graph.operations[3:],
        ),
    )
    with pytest.raises(GraphIdentityError, match="endpoint-role"):
        execution_graph_template_record(changed)


def test_collective_plan_projection_rewrites_every_nested_reference() -> None:
    dims = ModelDims(
        num_layers=2,
        hidden_size=64,
        intermediate_size=32,
        num_heads=4,
        num_kv_heads=2,
        head_size=16,
        vocab_size=128,
        num_experts=8,
        top_k=2,
        moe_intermediate_size=32,
        local_num_experts=2,
    )
    record = StepRecord(
        step_index=0,
        virtual_time_ps=0,
        scheduled=[
            ScheduledRequest(
                request_id="request",
                phase=RequestPhase.DECODE,
                num_new_tokens=1,
                context_length=16,
            )
        ],
        num_sampled=1,
    )
    graph = SerialStepLowerer(
        SerialStepLowererConfig(
            dims=dims,
            tp_ranks=(0,),
            ep_ranks=(0, 1, 2, 3),
        )
    ).lower(record)
    template = canonical_loads(execution_graph_template_record(graph).canonical)
    assert template["collective_plans"]
    for plan in template["collective_plans"]:
        assert set(plan) == {
            "operation_ordinal",
            "algorithm",
            "channel_ordinal",
            "rank_order",
            "rounds",
            "actions",
            "extents",
            "entry_action_ordinals",
            "terminal_action_ordinals",
        }
        assert all(set(round_) == {"transfer_channel_ordinal"} for round_ in plan["rounds"])
        assert all(
            set(action)
            == {
                "rank_ordinal",
                "kind",
                "extent_ordinal",
                "depends_on_action_ordinals",
            }
            for action in plan["actions"]
        )
        assert all(
            set(extent)
            == {
                "round_ordinal",
                "source_rank_ordinal",
                "destination_rank_ordinal",
                "send_action_ordinal",
                "receive_action_ordinal",
            }
            for extent in plan["extents"]
        )
        assert all(
            set(frontier) == {"rank_ordinal", "action_ordinals"}
            for frontier in plan["entry_action_ordinals"]
            + plan["terminal_action_ordinals"]
        )
    encoded = execution_graph_template_record(graph).canonical.decode()
    for forbidden in ("operation_id", "action_id", "extent_id", "integrity_sha256"):
        assert forbidden not in encoded
