"""Regression guards for the VLLM-22 qualification harness."""

from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

from simllm.adapters.vllm import ObservationStepSink
from simllm.adapters.vllm.schedule import (
    VllmBatchSlice,
    build_granite_execution_observations,
)
from simllm.compute import GPU_ENVELOPES, HostInitiationModel, ModelDims, RooflineProvider
from simllm.core import (
    ExecutionObservations,
    RequestPhase,
    ScheduledRequest,
    StepRecord,
    execution_graph_from_observations,
    execution_graph_to_json,
    step_record_to_json,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
STUDY_PATH = (
    REPO_ROOT / "examples" / "vllm_producer_qualification_v1" / "run_study.py"
)
DRIVER_PATH = REPO_ROOT / "examples" / "vllm_observed_schedule_v1" / "live_driver.py"
DP_SIZE = 8


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _dims() -> ModelDims:
    return ModelDims(
        num_layers=24,
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
        local_num_experts=4,
    )


def _record(step_index: int, request_ids: tuple[str, ...]) -> StepRecord:
    return StepRecord(
        step_index,
        0,
        [
            ScheduledRequest(request_id, RequestPhase.DECODE, 1, context_length=20)
            for request_id in request_ids
        ],
        num_sampled=len(request_ids),
    )


def _observations(step_index: int, request_ids: tuple[str, ...]):
    record = _record(step_index, request_ids)
    if len(request_ids) == 1:
        slices = (VllmBatchSlice(None, request_ids, 1),)
    else:
        slices = (
            VllmBatchSlice(0, request_ids[:1], 1),
            VllmBatchSlice(1, request_ids[1:], len(request_ids) - 1),
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
    return observations


def _inventory_row(record: StepRecord, observations: ExecutionObservations):
    graph = execution_graph_from_observations(
        record,
        observations,
        execution_id=f"source-step-{record.step_index}",
    )
    return {
        "record": step_record_to_json(record),
        "source_graph": execution_graph_to_json(graph),
        "step_result": {
            "request_metrics": [
                {"request_id": request.request_id} for request in record.scheduled
            ]
        },
    }


def test_frozen_submission_order_grammar_matches_the_real_producer():
    study = _load(STUDY_PATH, "vllm_producer_qualification_study")
    for step_index, request_ids in ((3, ("r0", "r1", "r2")), (24, ("r2",))):
        observations = _observations(step_index, request_ids)
        assert tuple(
            operation.operation_id for operation in observations.operations
        ) == study._expected_operation_ids(step_index)


def test_independent_dependency_and_correlation_grammar_can_fail(tmp_path):
    study = _load(STUDY_PATH, "vllm_producer_qualification_grammar")
    record = _record(3, ("r0", "r1", "r2"))
    observations = _observations(3, ("r0", "r1", "r2"))

    path = tmp_path / "valid.jsonl"
    path.write_text(json.dumps(_inventory_row(record, observations)) + "\n")
    inventory = study._source_inventory(path)
    assert all(inventory["dependency_grammar_checks"])
    assert all(inventory["correlation_grammar_checks"])

    dependency_operations = list(observations.operations)
    dependency_index = next(
        index
        for index, operation in enumerate(dependency_operations)
        if operation.operation_id.endswith("ubatch-0:layer-1:rank-0:pre-dispatch")
    )
    dependency_operations[dependency_index] = replace(
        dependency_operations[dependency_index],
        participant_local_depends_on=("step-3:ubatch-0:layer-0:ep-dispatch",),
    )
    changed_dependencies = replace(
        observations,
        operations=tuple(dependency_operations),
    )
    path.write_text(json.dumps(_inventory_row(record, changed_dependencies)) + "\n")
    inventory = study._source_inventory(path)
    assert not all(inventory["dependency_grammar_checks"])

    correlation_operations = list(observations.operations)
    correlation_operations[0] = replace(
        correlation_operations[0],
        correlation=replace(
            correlation_operations[0].correlation,
            request_ids=("replacement",),
        ),
    )
    changed_correlations = replace(
        observations,
        operations=tuple(correlation_operations),
    )
    path.write_text(json.dumps(_inventory_row(record, changed_correlations)) + "\n")
    inventory = study._source_inventory(path)
    assert not all(inventory["correlation_grammar_checks"])


def test_behavioral_registry_scores_raw_serial_to_observed_tpot():
    study = _load(STUDY_PATH, "vllm_producer_qualification_behavior")

    def result(value: int):
        return {"r0_tpot_ps": study.overlap._fraction_to_json(Fraction(value))}

    modes = {
        "single-node:serial": result(100_000_000),
        "single-node:observed": result(98_560_000),
        "cross-node:serial": result(112_000_000),
        "cross-node:observed": result(99_040_000),
    }
    behavior = study._evaluate_behavioral(modes)

    assert behavior["genuine_risk_passed"] == 3
    assert behavior["genuine_risk_executed"] == 3
    assert behavior["b2_absolute_reduction_ratio"] == 9.0

    modes["single-node:observed"] = result(100_000_000)
    failed = study._evaluate_behavioral(modes)
    assert failed["genuine_risk_passed"] < failed["genuine_risk_executed"]


def test_disabled_live_sink_remains_a_one_argument_legacy_contract():
    driver = _load(DRIVER_PATH, "vllm_producer_qualification_driver")
    sink = object.__new__(driver._LiveLegacySink)

    assert not isinstance(sink, ObservationStepSink)
    assert tuple(inspect.signature(driver._LiveLegacySink.__call__).parameters) == (
        "self",
        "record",
    )
