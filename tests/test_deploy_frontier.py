from __future__ import annotations

import hashlib
import json
import subprocess
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

from examples.deployment_frontier_v1.plot_study import prepare_plot
from simllm.deploy import (
    FRONTIER_RECORD_SCHEMA,
    PIPELINE_PARALLEL_UNPRICED,
    PLOT_CONTRACT_V3_SCHEMA,
    BudgetSpec,
    DeploymentCandidate,
    EnvelopeSpec,
    EstimateStamp,
    EstimatorInputs,
    EvidenceClass,
    ExternalAnchor,
    FabricSpec,
    FrontierPoint,
    ModelRef,
    ModelWork,
    NamedTermEstimate,
    PointClass,
    PoolSpec,
    ScanInputs,
    SimDerivedTerms,
    SlaSpec,
    TermEstimate,
    WorkloadPoint,
    frontier_record_from_json,
    frontier_record_to_json,
    pareto_front,
    prepare_plot_v3,
    scan,
    weak_dominance_pareto,
)

ROOT = Path(__file__).resolve().parents[1]
LEGACY_RESULT = ROOT / "examples" / "deployment_frontier_v1" / "result.json"
INVENTORY_SHA256 = "a" * 64
RECORD_SHA256 = "b" * 64

PUBLISHED_STEPS = {
    "b100-one-node-intra": (
        3_448_398_380,
        3_465_966_380,
        3_501_102_380,
        3_571_374_380,
        3_711_918_380,
        4_523_298_348,
    ),
    "h100-two-node-serialized": (
        8_234_981_205,
        8_276_934_638,
        8_360_841_504,
        8_528_655_235,
        8_864_282_698,
        9_535_537_623,
    ),
    "h100-nine-node-incast": (
        8_234_981_205,
        8_276_934_638,
        8_360_841_504,
        8_528_655_235,
        8_864_282_698,
        9_535_537_623,
    ),
}
BATCHES = (1, 2, 4, 8, 16, 32)


def _pool(*, pipeline_parallel: int = 1) -> PoolSpec:
    return PoolSpec(
        role="decode",
        engines=1,
        gpus_per_engine=8,
        tensor_parallel=8,
        pipeline_parallel=pipeline_parallel,
        expert_parallel=1,
        data_parallel=1,
        device="b100",
    )


def _candidate(
    candidate_id: str = "synthetic-frontier",
    *,
    pipeline_parallel: int = 1,
) -> DeploymentCandidate:
    return DeploymentCandidate(
        candidate_id=candidate_id,
        model=ModelRef(
            framework="synthetic",
            model_id="synthetic/model",
            inventory_sha256=INVENTORY_SHA256,
        ),
        pools=(_pool(pipeline_parallel=pipeline_parallel),),
        fabric=FabricSpec(
            inter_node_bits_per_second=400_000_000_000,
            intra_node_bytes_per_second=450_000_000_000,
        ),
        workload=WorkloadPoint(
            arrival_rate_rps=100,
            prompt_tokens=16,
            output_tokens=4,
            kv_context_tokens=2_000,
        ),
        sla=SlaSpec(tpot_target_ps=None, ttft_target_ps=None),
        budget=BudgetSpec(max_gpus=None, max_nodes=None),
    )


def _estimator_inputs(*, simulated: bool = False) -> EstimatorInputs:
    return EstimatorInputs(
        model_work=ModelWork(
            kernel_name="synthetic-decode-rank",
            flops_per_batch_item=8_000_000_000_000,
            static_logical_hbm_bytes=10_000_000_000,
            dynamic_hbm_bytes_per_batch_item=0,
            logical_collective_bytes_per_gpu_per_batch_item=0,
            inventory_sha256=INVENTORY_SHA256,
            source="synthetic model-work declaration",
        ),
        envelopes={
            "b100": EnvelopeSpec(
                device="b100",
                peak_flops_per_second=8_000_000_000_000_000,
                hbm_bytes_per_second=8_000_000_000_000,
                efficiency=1.0,
                source="synthetic declared GPU envelope",
            )
        },
        sim_derived=(
            SimDerivedTerms(
                fabric_excess_ps=1,
                intra_excess_ps=0,
                record_path="examples/deployment_frontier_v1/result.json",
                record_sha256=RECORD_SHA256,
            )
            if simulated
            else None
        ),
    )


def _scan_inputs(*, simulated: bool = False) -> ScanInputs:
    return ScanInputs(
        estimator_inputs=_estimator_inputs(simulated=simulated),
        static_rank_bytes_per_pool={"decode": 10_000_000_000},
        device_hbm_capacity_bytes={"b100": 192_000_000_000},
        anchors=(
            ExternalAnchor(
                anchor_id="published-paired",
                label="Published paired anchor",
                x_tokens_per_second_per_request=Fraction(22_282, 256),
                y_tokens_per_second_per_gpu=Fraction(22_282, 8),
            ),
            ExternalAnchor(
                anchor_id="published-production",
                label="Published production y-only anchor",
                y_tokens_per_second_per_gpu=Fraction(14_800, 8),
            ),
        ),
    )


def _published_point(
    configuration_id: str,
    batch_per_gpu: int,
    step_ps: int,
) -> FrontierPoint:
    key = hashlib.sha256(configuration_id.encode("ascii")).hexdigest()
    stamp = EstimateStamp(
        candidate_key=key,
        terms=(
            NamedTermEstimate(
                "kernel_floor",
                TermEstimate(step_ps, EvidenceClass.ROOFLINE, "published step literal"),
            ),
            NamedTermEstimate(
                "fabric_floor",
                TermEstimate(0, EvidenceClass.DECLARED, "published fabric floor"),
            ),
            NamedTermEstimate(
                "intra_floor",
                TermEstimate(0, EvidenceClass.DECLARED, "published intra floor"),
            ),
            NamedTermEstimate(
                "fabric_excess",
                TermEstimate(
                    0,
                    EvidenceClass.SIM_DERIVED,
                    "examples/deployment_frontier_v1/result.json",
                ),
            ),
        ),
    )
    x_value = Fraction(1_000_000_000_000, step_ps)
    return FrontierPoint(
        candidate_key=key,
        configuration_id=configuration_id,
        batch_per_gpu=batch_per_gpu,
        x_tokens_per_second_per_request=x_value,
        y_tokens_per_second_per_gpu=batch_per_gpu * x_value,
        point_class=PointClass.SIMULATED,
        step_ps=step_ps,
        stamp=stamp,
    )


def _compatibility_grid() -> tuple[FrontierPoint, ...]:
    return tuple(
        _published_point(configuration_id, batch, step)
        for configuration_id, steps in PUBLISHED_STEPS.items()
        for batch, step in zip(BATCHES, steps, strict=True)
    )


def test_scan_prices_in_declaration_order_and_preserves_exact_coordinates() -> None:
    first = _candidate("first")
    second = _candidate("second")

    record = scan((first, second), (4, 1), _scan_inputs())

    assert [candidate.configuration_id for candidate in record.candidates] == [
        "first",
        "second",
    ]
    assert [point.batch_per_gpu for point in record.points] == [4, 1, 4, 1]
    assert all(point.point_class is PointClass.ESTIMATE for point in record.points)
    for point in record.points:
        assert point.x_tokens_per_second_per_request == Fraction(
            1_000_000_000_000,
            point.step_ps,
        )
        assert point.y_tokens_per_second_per_gpu == (
            point.batch_per_gpu * point.x_tokens_per_second_per_request
        )
        assert point.stamp.candidate_key == point.candidate_key


def test_scan_rejected_candidate_has_reason_and_no_points() -> None:
    rejected = _candidate("pipeline-two", pipeline_parallel=2)

    record = scan((rejected,), (1, 2), _scan_inputs())

    candidate = record.candidates[0]
    assert not candidate.accepted
    assert candidate.rejection_reasons == (PIPELINE_PARALLEL_UNPRICED,)
    assert candidate.points == ()
    assert record.points == ()


def test_scan_marks_points_that_consume_tracked_simulation_terms() -> None:
    record = scan((_candidate(),), (1, 2), _scan_inputs(simulated=True))

    assert all(point.point_class is PointClass.SIMULATED for point in record.points)
    assert all(point.stamp.consumes_sim_derived for point in record.points)


def test_scan_does_not_create_a_process(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"process creation attempted: {args!r} {kwargs!r}")

    monkeypatch.setattr(subprocess, "Popen", forbidden)

    record = scan((_candidate(),), (1, 2, 4, 8, 16, 32), _scan_inputs())

    assert len(record.points) == 6


def test_frontier_record_strict_round_trip_and_y_only_anchor() -> None:
    record = scan((_candidate(),), (1, 2), _scan_inputs())

    rendered = frontier_record_to_json(record)

    assert rendered["schema"] == FRONTIER_RECORD_SCHEMA
    assert frontier_record_from_json(rendered) == record
    assert record.anchors[1].point_class is PointClass.MEASURED
    assert record.anchors[1].x_tokens_per_second_per_request is None


@pytest.mark.parametrize(
    "path",
    [(), ("candidates", 0), ("candidates", 0, "points", 0), ("anchors", 0)],
)
def test_frontier_record_rejects_unknown_fields(path: tuple[str | int, ...]) -> None:
    payload = deepcopy(
        frontier_record_to_json(scan((_candidate(),), (1,), _scan_inputs()))
    )
    target: object = payload
    for component in path:
        target = target[component]  # type: ignore[index]
    target["unexpected"] = 1  # type: ignore[index]

    with pytest.raises(ValueError, match="unknown fields"):
        frontier_record_from_json(payload)


def test_frontier_record_checks_schema_before_unknown_fields() -> None:
    payload = frontier_record_to_json(scan((_candidate(),), (1,), _scan_inputs()))
    payload["schema"] = "simllm-deployment-frontier-record-v0"
    payload["unexpected"] = 1

    with pytest.raises(ValueError, match="unsupported schema") as error:
        frontier_record_from_json(payload)

    assert "unknown fields" not in str(error.value)


def test_frozen_compatibility_grid_has_exact_six_point_b100_front() -> None:
    front = pareto_front(_compatibility_grid())

    assert [(point.configuration_id, point.batch_per_gpu) for point in front] == [
        ("b100-one-node-intra", 1),
        ("b100-one-node-intra", 2),
        ("b100-one-node-intra", 4),
        ("b100-one-node-intra", 8),
        ("b100-one-node-intra", 16),
        ("b100-one-node-intra", 32),
    ]


def test_pareto_front_keeps_equal_coordinates_and_single_point() -> None:
    first = _published_point("h100-two-node-serialized", 1, 8_234_981_205)
    equal = _published_point("h100-nine-node-incast", 1, 8_234_981_205)

    assert pareto_front((first,)) == (first,)
    assert pareto_front((first, equal)) == (equal, first)


def test_pareto_front_is_deterministic_under_input_permutation() -> None:
    grid = _compatibility_grid()

    assert pareto_front(reversed(grid)) == pareto_front(grid)


def test_prepare_plot_v3_preserves_every_legacy_v2_series_value() -> None:
    result = json.loads(LEGACY_RESULT.read_text(encoding="utf-8"))
    expected = prepare_plot(result)

    observed = prepare_plot_v3(result)

    for key, value in expected.items():
        assert observed[key] == value
    assert observed["schema"] == PLOT_CONTRACT_V3_SCHEMA
    assert observed["axes"] == {
        "x_scale": "log",
        "y_scale": "log",
        "optimal_corner": "upper-right",
    }
    assert observed["series_styles"]["measured_paired"] == {
        "kind": "marker",
        "marker": "diamond",
        "fill": "white",
        "edge": "black",
    }
    assert observed["series_styles"]["measured_y_only"] == {
        "kind": "horizontal-line",
        "style": "dashed",
    }
    assert observed["pareto_emphasis"]["points"] == [
        {"configuration_id": "b100-one-node-intra", "batch_per_gpu": batch}
        for batch in BATCHES
    ]


def test_prepare_plot_v3_marks_classes_and_front_without_inventing_anchor_x() -> None:
    estimate_record = scan((_candidate(),), (1, 2), _scan_inputs())
    simulated_record = scan((_candidate(),), (1, 2), _scan_inputs(simulated=True))

    estimate_plot = prepare_plot_v3(estimate_record)
    simulated_plot = prepare_plot_v3(simulated_record)

    assert {point["point_class"] for point in estimate_plot["point_classes"]} == {
        PointClass.ESTIMATE.value
    }
    assert {point["point_class"] for point in simulated_plot["point_classes"]} == {
        PointClass.SIMULATED.value
    }
    assert estimate_plot["paired_marker"] == {
        "label": "Published paired anchor",
        "x": 22_282 / 256,
        "y": 22_282 / 8,
    }
    assert estimate_plot["y_only_anchor"] == {
        "label": "Published production y-only anchor",
        "y": 14_800 / 8,
    }
    assert "x" not in estimate_plot["y_only_anchor"]


def test_point_class_cannot_mislabel_simulation_evidence() -> None:
    simulated = _published_point("simulated", 1, 10)

    with pytest.raises(ValueError, match="must reflect"):
        replace(simulated, point_class=PointClass.ESTIMATE)


def test_single_axis_tie_uses_weak_dominance() -> None:
    dominant = _published_point("tie-dominant", 1, 1_000)
    dominated = _published_point("tie-dominated", 2, 2_000)
    assert dominated.y_tokens_per_second_per_gpu == dominant.y_tokens_per_second_per_gpu
    assert dominant.x_tokens_per_second_per_request > dominated.x_tokens_per_second_per_request
    assert pareto_front((dominant, dominated)) == (dominant,)
    assert pareto_front((dominated, dominant)) == (dominant,)


def test_generic_weak_dominance_front_deduplicates_by_stable_identity() -> None:
    points = (
        {"id": "z", "x": Fraction(1), "y": Fraction(3)},
        {"id": "a", "x": Fraction(1), "y": Fraction(3)},
        {"id": "b", "x": Fraction(2), "y": Fraction(2)},
        {"id": "dominated", "x": Fraction(1), "y": Fraction(1)},
    )

    front = weak_dominance_pareto(
        reversed(points),
        coordinate=lambda point: (point["x"], point["y"]),
        identity=lambda point: point["id"],
    )

    assert [point["id"] for point in front] == ["a", "b"]
