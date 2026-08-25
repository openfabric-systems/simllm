from __future__ import annotations

from dataclasses import replace

import pytest

from simllm.calibration.bindings import (
    CollectiveDeviceRankFrontier,
    CollectiveDeviceStageBinding,
    DeviceDispatchContext,
    DispatchSignature,
    DispatchTraitValueType,
    OperationImplementationBinding,
    RankDeviceAssignment,
    ResolutionSource,
    ResolvedCollectiveDeviceStage,
    ResolvedCollectiveDeviceStageSet,
    ResolvedDeviceBindingClosure,
    ResolvedOperationServiceBinding,
    ResolvedOperationServiceBindingSet,
    SelectedDeviceModel,
    TypedDispatchTrait,
    binding_record_from_obj,
    validate_observed_collective_stage_bindings,
    validate_observed_operation_bindings,
    validate_resolved_binding_closure,
    validate_resolved_collective_stage_plan,
    validate_resolved_operation_bindings,
)
from simllm.compute.device_model import (
    BinaryImplementationRef,
    ShapeVector,
)

GRAPH = "1" * 64
CONTEXT = "2" * 64
MODEL = "3" * 64
SIGNATURE = "4" * 64
PLAN = "5" * 64
ENTRY = "6" * 64
OBSERVED = "7" * 64
OPERATION_SET = "8" * 64
COLLECTIVE_SET = "9" * 64
MODULE = "a" * 64
FUNCTION = "b" * 64


def implementation() -> BinaryImplementationRef:
    return BinaryImplementationRef(
        implementation_id="nccl-ring-sm80",
        vendor_id="nvidia",
        device_isa="sm_80",
        module_sha256=MODULE,
        function_sha256=FUNCTION,
        function_symbol=None,
        backend_id="cuda",
        algorithm_id="nccl-ring",
        launch_formula_id="one-grid-per-rank",
    )


def shape() -> ShapeVector:
    return ShapeVector(shape_schema_id="collective-stage-v1", values=(4096, 2))


def signature() -> DispatchSignature:
    return DispatchSignature(
        framework_id="vllm",
        framework_version="0.10",
        backend_id="cuda",
        backend_version="12.4",
        kernel_library_id="nccl",
        kernel_library_version="2.23",
        algorithm_policy_id="framework-observed",
        device_isa="sm_80",
        numeric_traits=(
            TypedDispatchTrait(
                trait_id="dtype",
                value_type=DispatchTraitValueType.STRING,
                value="bf16",
            ),
            TypedDispatchTrait(
                trait_id="tensor_parallel",
                value_type=DispatchTraitValueType.INTEGER,
                value=2,
            ),
        ),
        layout_traits=(
            TypedDispatchTrait(
                trait_id="contiguous",
                value_type=DispatchTraitValueType.BOOLEAN,
                value=True,
            ),
        ),
    )


def context() -> DeviceDispatchContext:
    return DeviceDispatchContext(
        instance_graph_sha256=GRAPH,
        rank_device_assignments=(
            RankDeviceAssignment(rank=0, device_instance_id="gpu-0"),
            RankDeviceAssignment(rank=1, device_instance_id="gpu-1"),
        ),
        selected_device_models=(
            SelectedDeviceModel(
                device_instance_id="gpu-0",
                device_model_id="a100-v1",
                device_model_sha256=MODEL,
                dispatch_signature_sha256=SIGNATURE,
            ),
            SelectedDeviceModel(
                device_instance_id="gpu-1",
                device_model_id="a100-v1",
                device_model_sha256=MODEL,
                dispatch_signature_sha256=SIGNATURE,
            ),
        ),
    )


def resolved_operation() -> ResolvedOperationServiceBinding:
    return ResolvedOperationServiceBinding(
        instance_graph_sha256=GRAPH,
        operation_id="op-0",
        launch_ordinal=0,
        device_instance_id="gpu-0",
        device_model_sha256=MODEL,
        semantic_key="compute:gemm",
        shape_vector=shape(),
        implementation_ref=implementation(),
        service_entry_id=ENTRY,
        resolution_source=ResolutionSource.SELECTOR,
        observed_implementation_binding_sha256=None,
    )


def resolved_stage(*, rank: int = 0, device: str = "gpu-0") -> ResolvedCollectiveDeviceStage:
    return ResolvedCollectiveDeviceStage(
        instance_graph_sha256=GRAPH,
        collective_operation_id="all-reduce-0",
        collective_plan_integrity_sha256=PLAN,
        rank=rank,
        launch_ordinal=0,
        device_instance_id=device,
        device_model_sha256=MODEL,
        implementation_ref=implementation(),
        shape_vector=shape(),
        service_entry_id=ENTRY,
    )


def frontier(*, rank: int = 0) -> CollectiveDeviceRankFrontier:
    return CollectiveDeviceRankFrontier(
        collective_operation_id="all-reduce-0",
        collective_plan_integrity_sha256=PLAN,
        rank=rank,
        ordered_stage_ordinals=(0,),
        entry_action_ids=(f"entry-{rank}",),
        terminal_action_ids=(f"terminal-{rank}",),
    )


def test_typed_dispatch_signature_round_trips_and_forbids_launch_mode() -> None:
    record = signature()

    assert DispatchSignature.from_obj(record.to_obj()) == record
    assert binding_record_from_obj(record.to_obj()) == record

    payload = record.to_obj()
    payload["launch_mode"] = "cuda-graph"
    with pytest.raises(ValueError, match="unknown fields"):
        DispatchSignature.from_obj(payload)


@pytest.mark.parametrize(
    ("value_type", "value"),
    [
        ("integer", True),
        ("boolean", 1),
        ("string", 1),
    ],
)
def test_typed_trait_rejects_cross_type_values(value_type: str, value: object) -> None:
    with pytest.raises(TypeError, match="does not match"):
        TypedDispatchTrait.from_obj(
            {"trait_id": "x", "value_type": value_type, "value": value}
        )


def test_signature_requires_sorted_unique_trait_ids() -> None:
    trait_a = TypedDispatchTrait(
        trait_id="z", value_type=DispatchTraitValueType.INTEGER, value=1
    )
    trait_b = TypedDispatchTrait(
        trait_id="a", value_type=DispatchTraitValueType.INTEGER, value=1
    )
    with pytest.raises(ValueError, match="must be sorted"):
        replace(signature(), numeric_traits=(trait_a, trait_b))
    with pytest.raises(ValueError, match="duplicate"):
        replace(signature(), numeric_traits=(trait_a, trait_a))


def test_dispatch_context_is_total_sorted_and_strict() -> None:
    record = context()
    assert DeviceDispatchContext.from_obj(record.to_obj()) == record
    record.validate_graph_ranks((1, 0, 1))
    assert record.device_for_rank(1) == "gpu-1"

    with pytest.raises(ValueError, match="total graph rank coverage"):
        record.validate_graph_ranks((0, 1, 2))
    with pytest.raises(ValueError, match="must be sorted"):
        replace(record, rank_device_assignments=tuple(reversed(record.rank_device_assignments)))
    with pytest.raises(ValueError, match="cover exactly"):
        replace(record, selected_device_models=record.selected_device_models[:1])


def test_observed_bindings_are_binary_only_and_have_exact_totality() -> None:
    operation = OperationImplementationBinding(
        instance_graph_sha256=GRAPH,
        operation_id="op-0",
        launch_ordinal=0,
        implementation_ref=implementation(),
        shape_vector=shape(),
    )
    collective = CollectiveDeviceStageBinding(
        instance_graph_sha256=GRAPH,
        collective_operation_id="all-reduce-0",
        collective_plan_integrity_sha256=PLAN,
        rank=0,
        launch_ordinal=0,
        implementation_ref=implementation(),
        shape_vector=shape(),
    )

    assert OperationImplementationBinding.from_obj(operation.to_obj()) == operation
    assert CollectiveDeviceStageBinding.from_obj(collective.to_obj()) == collective
    validate_observed_operation_bindings((operation,), (operation.key,))
    validate_observed_collective_stage_bindings((collective,), (collective.key,))
    with pytest.raises(ValueError, match="must match expected"):
        validate_observed_operation_bindings((operation,), ())


def test_resolved_operation_source_and_observed_hash_are_coupled() -> None:
    selector = replace(resolved_operation(), service_entry_id="entry-gemm-sm80")
    assert ResolvedOperationServiceBinding.from_obj(selector.to_obj()) == selector

    observed = replace(
        selector,
        resolution_source=ResolutionSource.OBSERVED_BINDING,
        observed_implementation_binding_sha256=OBSERVED,
    )
    assert ResolvedOperationServiceBinding.from_obj(observed.to_obj()) == observed
    with pytest.raises(ValueError, match="observed-binding source requires"):
        replace(selector, resolution_source=ResolutionSource.OBSERVED_BINDING)


def test_resolved_operation_set_rejects_graph_splice_and_duplicates() -> None:
    binding = resolved_operation()
    record = ResolvedOperationServiceBindingSet(
        instance_graph_sha256=GRAPH,
        dispatch_context_sha256=CONTEXT,
        bindings=(binding,),
    )
    assert ResolvedOperationServiceBindingSet.from_obj(record.to_obj()) == record

    with pytest.raises(ValueError, match="graph splice"):
        replace(record, bindings=(replace(binding, instance_graph_sha256="f" * 64),))
    with pytest.raises(ValueError, match="duplicate"):
        replace(record, bindings=(binding, binding))


def test_resolved_operation_totality_uses_graph_order_and_rank_assignment() -> None:
    binding = resolved_operation()
    record = ResolvedOperationServiceBindingSet(
        instance_graph_sha256=GRAPH,
        dispatch_context_sha256=CONTEXT,
        bindings=(binding,),
    )
    validate_resolved_operation_bindings(
        record,
        expected_keys=(binding.key,),
        operation_rank_by_id={"op-0": 0},
        dispatch_context=context(),
    )
    with pytest.raises(ValueError, match="missing, extra"):
        validate_resolved_operation_bindings(
            record,
            expected_keys=(),
            operation_rank_by_id={"op-0": 0},
            dispatch_context=context(),
        )
    with pytest.raises(ValueError, match="rank/device assignment splice"):
        validate_resolved_operation_bindings(
            replace(record, bindings=(replace(binding, device_instance_id="gpu-1"),)),
            expected_keys=(binding.key,),
            operation_rank_by_id={"op-0": 0},
            dispatch_context=context(),
        )

    second = replace(
        binding,
        operation_id="op-1",
        device_instance_id="gpu-1",
        semantic_key="compute:activation",
    )
    reordered = replace(record, bindings=(second, binding))
    with pytest.raises(ValueError, match="reordered launch"):
        validate_resolved_operation_bindings(
            reordered,
            expected_keys=(binding.key, second.key),
            operation_rank_by_id={"op-0": 0, "op-1": 1},
            dispatch_context=context(),
        )


def test_collective_stage_set_requires_one_stage_and_frontier_per_rank() -> None:
    stage0 = resolved_stage()
    stage1 = resolved_stage(rank=1, device="gpu-1")
    record = ResolvedCollectiveDeviceStageSet(
        instance_graph_sha256=GRAPH,
        dispatch_context_sha256=CONTEXT,
        stages=(stage0, stage1),
        rank_frontiers=(frontier(), frontier(rank=1)),
    )

    assert ResolvedCollectiveDeviceStageSet.from_obj(record.to_obj()) == record
    with pytest.raises(ValueError, match="missing or extra rank"):
        replace(record, rank_frontiers=(frontier(),))
    with pytest.raises(ValueError, match="exactly one stage"):
        replace(frontier(), ordered_stage_ordinals=(0, 1))
    with pytest.raises(ValueError, match="must not be empty"):
        ResolvedCollectiveDeviceStageSet(
            instance_graph_sha256=GRAPH,
            dispatch_context_sha256=CONTEXT,
            stages=(),
            rank_frontiers=(),
        )

    validate_resolved_collective_stage_plan(
        record,
        expected_stage_keys=(stage0.key, stage1.key),
        expected_rank_frontiers=(frontier(), frontier(rank=1)),
        dispatch_context=context(),
    )
    with pytest.raises(ValueError, match="do not match the traffic plan"):
        validate_resolved_collective_stage_plan(
            record,
            expected_stage_keys=(stage0.key, stage1.key),
            expected_rank_frontiers=(
                replace(frontier(), entry_action_ids=("different",)),
                frontier(rank=1),
            ),
            dispatch_context=context(),
        )
    reordered = replace(
        record,
        stages=tuple(reversed(record.stages)),
        rank_frontiers=tuple(reversed(record.rank_frontiers)),
    )
    with pytest.raises(ValueError, match="reordered stage"):
        validate_resolved_collective_stage_plan(
            reordered,
            expected_stage_keys=(stage0.key, stage1.key),
            expected_rank_frontiers=(frontier(), frontier(rank=1)),
            dispatch_context=context(),
        )


def test_resolved_collective_stage_requires_typed_implementation_ref() -> None:
    with pytest.raises(TypeError, match="expected ImplementationRef"):
        replace(resolved_stage(), implementation_ref=object())


def test_resolved_closure_accepts_one_graph_context_and_model_tuple() -> None:
    operation_set = ResolvedOperationServiceBindingSet(
        instance_graph_sha256=GRAPH,
        dispatch_context_sha256=CONTEXT,
        bindings=(resolved_operation(),),
    )
    collective_set = ResolvedCollectiveDeviceStageSet(
        instance_graph_sha256=GRAPH,
        dispatch_context_sha256=CONTEXT,
        stages=(resolved_stage(), resolved_stage(rank=1, device="gpu-1")),
        rank_frontiers=(frontier(), frontier(rank=1)),
    )
    closure = ResolvedDeviceBindingClosure(
        instance_graph_sha256=GRAPH,
        operation_service_binding_set_sha256=OPERATION_SET,
        collective_device_stage_set_sha256=COLLECTIVE_SET,
    )

    validate_resolved_binding_closure(
        closure=closure,
        operation_set=operation_set,
        operation_set_sha256=OPERATION_SET,
        dispatch_context=context(),
        dispatch_context_sha256=CONTEXT,
        collective_set=collective_set,
        collective_set_sha256=COLLECTIVE_SET,
    )


def test_resolved_closure_rejects_graph_context_and_model_splices() -> None:
    operation_set = ResolvedOperationServiceBindingSet(
        instance_graph_sha256=GRAPH,
        dispatch_context_sha256=CONTEXT,
        bindings=(resolved_operation(),),
    )
    closure = ResolvedDeviceBindingClosure(
        instance_graph_sha256=GRAPH,
        operation_service_binding_set_sha256=OPERATION_SET,
        collective_device_stage_set_sha256=None,
    )
    kwargs = {
        "closure": closure,
        "operation_set": operation_set,
        "operation_set_sha256": OPERATION_SET,
        "dispatch_context": context(),
        "dispatch_context_sha256": CONTEXT,
    }

    with pytest.raises(ValueError, match="operation-set digest splice"):
        validate_resolved_binding_closure(**(kwargs | {"operation_set_sha256": "f" * 64}))
    with pytest.raises(ValueError, match="cross-graph splice"):
        validate_resolved_binding_closure(
            **(kwargs | {"closure": replace(closure, instance_graph_sha256="f" * 64)})
        )
    spliced = replace(
        operation_set,
        bindings=(replace(resolved_operation(), device_model_sha256="f" * 64),),
    )
    with pytest.raises(ValueError, match="model splice"):
        validate_resolved_binding_closure(**(kwargs | {"operation_set": spliced}))


def test_binding_dispatcher_rejects_unknown_schema() -> None:
    with pytest.raises(ValueError, match="unsupported binding record"):
        binding_record_from_obj({"schema": "unknown"})
