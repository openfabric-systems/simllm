"""Regression guards for the observed-overlap control arm."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from simllm.adapters.vllm.schedule import (
    VllmBatchSlice,
    build_granite_execution_observations,
)
from simllm.compute import GPU_ENVELOPES, HostInitiationModel, ModelDims, RooflineProvider
from simllm.core import (
    CollectiveWork,
    RequestPhase,
    ScheduledRequest,
    StepRecord,
    execution_graph_from_observations,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
STUDY_PATH = REPO_ROOT / "examples" / "vllm_observed_overlap_v1" / "run_study.py"
DP_SIZE = 8
NUM_LAYERS = 24


def _load_study():
    spec = importlib.util.spec_from_file_location(
        "vllm_observed_overlap_study",
        STUDY_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _dims() -> ModelDims:
    return ModelDims(
        num_layers=NUM_LAYERS,
        hidden_size=1_024,
        intermediate_size=512,
        num_heads=16,
        num_kv_heads=8,
        head_size=64,
        vocab_size=49_155,
        dtype_bytes=2,
        num_experts=32,
        top_k=8,
        moe_intermediate_size=512,
        local_num_experts=32 // DP_SIZE,
    )


def _observations(request_ids: tuple[str, ...], step_index: int = 3):
    record = StepRecord(
        step_index,
        0,
        [
            ScheduledRequest(request_id, RequestPhase.DECODE, 1, context_length=20)
            for request_id in request_ids
        ],
        num_sampled=len(request_ids),
    )
    if len(request_ids) == 1:
        slices = (VllmBatchSlice(None, request_ids, 1),)
    else:
        split = len(request_ids) // 2
        slices = (
            VllmBatchSlice(0, request_ids[:split], split),
            VllmBatchSlice(1, request_ids[split:], len(request_ids) - split),
        )
    observations, _ = build_granite_execution_observations(
        record,
        _dims(),
        tuple(range(DP_SIZE)),
        slices,
        RooflineProvider(efficiency=0.7),
        GPU_ENVELOPES["b100"],
        HostInitiationModel(),
    )
    return record, observations


def test_control_arm_adds_only_edges_and_stays_acyclic() -> None:
    study = _load_study()
    record, observed = _observations(("a", "b", "c"))
    control, added = study._control_observations(observed, record.step_index)

    assert added == study.CONTROL_EDGES_PER_DBO_STEP == 760
    assert study._tuple_fields_agree(observed, control)
    for source, target in zip(observed.operations, control.operations, strict=True):
        assert set(source.depends_on) <= set(target.depends_on)
    assert sum(
        len(target.depends_on) - len(source.depends_on)
        for source, target in zip(observed.operations, control.operations, strict=True)
    ) == added

    # the control graph must remain a legal DAG once queue FIFO edges are added
    graph = execution_graph_from_observations(
        record,
        control,
        execution_id=f"step-{record.step_index}",
    )
    assert len(graph.operations) == len(observed.operations)
    assert graph.completion_operation_ids == observed.completion_operation_ids
    assert sum(
        isinstance(operation.work, CollectiveWork) for operation in graph.operations
    ) == study.DBO_MOE_INVOCATIONS


def test_control_arm_is_a_no_op_without_a_second_microbatch() -> None:
    study = _load_study()
    record, observed = _observations(("solo",))
    control, added = study._control_observations(observed, record.step_index)

    assert added == 0
    assert control is observed


def test_reversed_serialization_would_be_rejected_as_a_cycle() -> None:
    """The originally frozen edge direction closes a cycle with queue FIFO."""

    from dataclasses import replace

    from simllm.core import ExecutionObservations

    record, observed = _observations(("a", "b", "c"))
    step = record.step_index
    operations = []
    for operation in observed.operations:
        if operation.operation_id.endswith(":pre-dispatch") and (
            f"step-{step}:ubatch-1:layer-0:" in operation.operation_id
        ):
            operation = replace(
                operation,
                depends_on=(
                    *operation.depends_on,
                    f"step-{step}:ubatch-0:layer-0:ep-combine",
                ),
            )
        operations.append(operation)
    reversed_control = ExecutionObservations(
        operations=tuple(operations),
        completion_operation_ids=observed.completion_operation_ids,
    )
    with pytest.raises(ValueError, match="cycle"):
        execution_graph_from_observations(
            record,
            reversed_control,
            execution_id=f"step-{step}",
        )
