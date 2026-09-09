"""Corruption controls for the post-specified retained-evidence audit."""

import json
from dataclasses import replace

import pytest

from examples.kimi_k3_structure_v1.audit_retained import (
    event_guards,
    frozen_grid,
    partition_guards,
    projection_guards,
    record_guards,
    rejection_guards,
    request_guards,
)
from examples.kimi_k3_structure_v1.run_study import HERE, ROOT, SUITE_ID, Evidence
from simllm.backends.kimi_k3_lowerer import KimiK3Lowerer, KimiK3LowererConfig
from simllm.calibration.extraction import case_records_from_suite, load_extraction_suite
from simllm.calibration.kimi_k3 import build_kimi_k3_inventory
from simllm.calibration.model_inventory import FrameworkIdentity
from simllm.compute.device_model import ShapeVector
from simllm.core import CoarseDeviceRuntime
from simllm.core.execution import (
    ComputeWork,
    EventPhase,
    ExecutionGraph,
    ExecutionOperation,
    OperationCorrelation,
    ResourceKind,
)
from simllm.core.step import (
    AdditiveVisitTotals,
    LatencyAttribution,
    RequestMetric,
    RequestPhase,
    StepResult,
    step_record_to_json,
)


def inputs():
    raw = (ROOT / "offline/calibration/suites" / SUITE_ID / "suite.json").read_bytes()
    suite, model = load_extraction_suite(raw)
    suite["graph_cells"] = suite["graph_cells"][:1]
    records = case_records_from_suite(suite)
    graph = KimiK3Lowerer(KimiK3LowererConfig(model.geometry, "vllm")).lower(records[0])
    declared = suite["frameworks"][0]
    inventory = build_kimi_k3_inventory(
        suite_raw=json.dumps(suite).encode(),
        suite=suite,
        model=model,
        framework=FrameworkIdentity(
            "vllm", declared["version"], declared["source_commit"], None, "unit-seam"
        ),
        records=records,
    )
    return graph, inventory, json.loads((HERE / "expectations.json").read_bytes())


def test_frozen_grid_keeps_both_parameters_and_phases():
    frozen = json.loads((HERE / "expectations.json").read_bytes())
    grid = frozen_grid(frozen)
    assert len(grid) == 12
    assert {row[0] for row in grid} == {"prefill", "decode"}
    assert {row[1] for row in grid} == {1, 3}
    assert ("prefill", 3, 16, 16) in grid
    assert ("decode", 3, 1, 257) in grid


def test_wrong_operator_name_cannot_hide_behind_preserved_visit_count():
    graph, _, frozen = inputs()
    baseline = Evidence()
    partition_guards(graph, "prefill", frozen, baseline, "unit")
    assert baseline.valid
    target = next(op for op in graph.operations if op.work.kernel == "kimi_k3.kda.q_projection")
    changed = replace(
        graph,
        operations=tuple(
            replace(op, work=replace(op.work, kernel="kimi_k3.kda.fake_projection"))
            if op is target
            else op
            for op in graph.operations
        ),
    )
    corrupted = Evidence()
    partition_guards(changed, "prefill", frozen, corrupted, "unit")
    assert not corrupted.valid


def test_inventory_arithmetic_must_join_original_graph():
    graph, inventory, _ = inputs()
    baseline = Evidence()
    projection_guards(graph, inventory.cases[0], inventory, baseline, "unit")
    assert baseline.valid
    case = inventory.cases[0]
    target = next(
        p
        for p in case.kernel_projections
        if p.aggregate_flops is not None and p.aggregate_flops > 0
    )
    changed = replace(
        case,
        kernel_projections=tuple(
            replace(p, aggregate_flops=p.aggregate_flops + 1) if p is target else p
            for p in case.kernel_projections
        ),
    )
    corrupted = Evidence()
    projection_guards(graph, changed, inventory, corrupted, "unit")
    assert not corrupted.valid


def test_unbound_negative_controls_reject_at_supported_consumers():
    evidence = Evidence()
    rejection_guards(evidence)
    assert evidence.valid
    assert {row["name"] for row in evidence.guards} == {
        "reject-logical:runtime",
        "reject-logical:GOAL",
        "reject-logical:bottleneck",
    }


def test_extra_operator_is_not_hidden_outside_the_completion_frontier():
    graph, _, frozen = inputs()
    extra = ExecutionOperation(
        "extra",
        0,
        "extra",
        ComputeWork("kimi_k3.unknown", flops=None, hbm_bytes=None, scope="logical-operator"),
        correlation=OperationCorrelation(layer=0),
    )
    graph = replace(graph, operations=(*graph.operations, extra))
    evidence = Evidence()
    partition_guards(graph, "prefill", frozen, evidence, "unit")
    assert not evidence.valid


def test_inventory_cannot_delete_an_axis_owned_by_graph_work():
    graph, inventory, _ = inputs()
    target = next(p for p in inventory.cases[0].kernel_projections if p.logical_launch_count)
    schemas = tuple(
        replace(s, axes=s.axes[:-1])
        if s.shape_schema_id == target.shape_vector.shape_schema_id
        else s
        for s in inventory.shape_schemas
    )
    case = replace(
        inventory.cases[0],
        kernel_projections=tuple(
            replace(
                p,
                shape_vector=ShapeVector(
                    p.shape_vector.shape_schema_id, p.shape_vector.values[:-1]
                ),
            )
            if p is target
            else p
            for p in inventory.cases[0].kernel_projections
        ),
    )
    changed = replace(inventory, shape_schemas=schemas, cases=(case,))
    evidence = Evidence()
    projection_guards(graph, case, changed, evidence, "unit")
    assert not evidence.valid


def test_duplicate_and_acausal_events_fail_without_replaying_the_model():
    graph = ExecutionGraph(
        "small",
        0,
        0,
        (
            ExecutionOperation("a", 0, "a", ComputeWork("a", nominal_duration_ps=7)),
            ExecutionOperation(
                "b", 0, "b", ComputeWork("b", nominal_duration_ps=7), depends_on=("a",)
            ),
        ),
    )
    result = CoarseDeviceRuntime(serial_compute=True).execute(graph)
    clean = Evidence()
    event_guards(graph, result, 7, clean, "unit")
    assert clean.valid
    duplicated = replace(result, events=(result.events[0], *result.events))
    corrupt = Evidence()
    event_guards(graph, duplicated, 7, corrupt, "unit")
    assert not corrupt.valid
    changed = replace(
        result,
        events=tuple(
            replace(e, timestamp_ps=0)
            if e.operation_id == "b"
            and e.resource.kind is ResourceKind.GPU_WORK_QUEUE
            and e.phase is EventPhase.QUEUED
            else e
            for e in result.events
        ),
    )
    corrupt = Evidence()
    event_guards(graph, changed, 7, corrupt, "unit")
    assert not corrupt.valid


@pytest.mark.parametrize(
    "corruption",
    [
        "different-resource-duplicate",
        "wrong-resource",
        "missing-resource",
        "nonzero-progress",
        "invented-submission-bytes",
    ],
)
def test_event_resource_and_byte_corruption_voids_the_audit(corruption):
    graph = ExecutionGraph(
        "small", 0, 0, (ExecutionOperation("a", 0, "a", ComputeWork("a", nominal_duration_ps=7)),)
    )
    result = CoarseDeviceRuntime(serial_compute=True).execute(graph)
    first = result.events[0]
    wrong = replace(first, resource=replace(first.resource, resource_id="unowned-resource"))
    if corruption == "different-resource-duplicate":
        events = (*result.events, wrong)
    elif corruption == "wrong-resource":
        events = (wrong, *result.events[1:])
    elif corruption == "missing-resource":
        events = (replace(first, resource=None), *result.events[1:])
    elif corruption == "nonzero-progress":
        events = tuple(
            replace(event, completed_bytes=1) if event.phase is EventPhase.PROGRESS else event
            for event in result.events
        )
    else:
        events = (replace(first, completed_bytes=0), *result.events[1:])
    evidence = Evidence()
    event_guards(graph, replace(result, events=events), 7, evidence, "unit")
    assert not evidence.valid


def test_retained_inputs_and_inventory_hash_must_match_the_frozen_record():
    _, inventory, _ = inputs()
    suite = json.loads((ROOT / "offline/calibration/suites" / SUITE_ID / "suite.json").read_bytes())
    records = case_records_from_suite(suite)[:1]
    payload = step_record_to_json(records[0])
    clean = Evidence()
    record_guards(json.dumps(payload).encode(), inventory.cases, records, clean, "unit")
    assert clean.valid
    corrupted_input = Evidence()
    record_guards(
        json.dumps({**payload, "virtual_time_ps": 1}).encode(),
        inventory.cases,
        records,
        corrupted_input,
        "unit",
    )
    assert not corrupted_input.valid
    corrupted_hash = Evidence()
    record_guards(
        json.dumps(payload).encode(),
        (replace(inventory.cases[0], step_record_sha256="0" * 64),),
        records,
        corrupted_hash,
        "unit",
    )
    assert not corrupted_hash.valid


@pytest.mark.parametrize("corruption", ["early-completion", "wrong-latency", "missing-request"])
def test_request_boundary_must_join_retained_completion(corruption):
    from simllm.core.execution import ExecutionResult

    boundary = 3244000
    metric = RequestMetric(
        "request",
        RequestPhase.PREFILL,
        1,
        boundary,
        boundary,
        boundary,
        None,
        LatencyAttribution(kernel_ps=boundary),
        AdditiveVisitTotals(),
    )
    step = StepResult(0, boundary, boundary, (metric,))
    events = ExecutionResult("small", boundary)
    clean = Evidence()
    request_guards(step, events, 0, 0, 1, 1000, clean, "unit")
    assert clean.valid
    if corruption == "early-completion":
        metrics = (replace(metric, completed_at_ps=0),)
    elif corruption == "wrong-latency":
        metrics = (replace(metric, latency_ps=0, attribution=LatencyAttribution()),)
    else:
        metrics = ()
    corrupted = Evidence()
    request_guards(replace(step, request_metrics=metrics), events, 0, 0, 1, 1000, corrupted, "unit")
    assert not corrupted.valid
