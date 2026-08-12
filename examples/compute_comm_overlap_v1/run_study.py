"""Run the frozen compute and communication overlap study."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import subprocess
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

D_PS = 41_943_040
COMPUTE_PS = (20_971_520, 83_886_080)
SHAPES = ("independent", "pipeline", "serial")
FROZEN_JCT_PS = {
    20_971_520: {
        "independent": 41_943_040,
        "pipeline": 52_428_800,
        "serial": 62_914_560,
    },
    83_886_080: {
        "independent": 83_886_080,
        "pipeline": 104_857_600,
        "serial": 125_829_120,
    },
}
SERIAL_GRAPH_SHA256 = (
    "aa3c836fe559973a7bf0940384c2e8a84e6af84e0fbd2c02d3b89774ee0c8e2d"
)
SERIAL_GOAL_SHA256 = (
    "7087db6780f7e34f5a559a6505eeccc15d984c7b478cd8f0bc5838053825d4b6"
)
EXPECTATIONS_COMMIT = "cf3ae9c"
EVIDENCE_FAMILIES = ("A", "B", "C", "D")
BEHAVIORAL = "behavioral_relation"
EXACT = "exact_oracle"
FATAL = "fatal_guard"
RANKS = (0, 8)
PAYLOAD_BYTES = 524_288


@dataclass(frozen=True)
class EvidenceRow:
    family: str
    evidence_class: str
    case: str
    expected: Any
    measured: Any
    passed: bool
    genuine_risk: bool | None


@dataclass(frozen=True)
class ScheduleResult:
    compute_ps: int
    shape: str
    step_latency_ps: int
    ttft_ps: int
    tpot_ps: Fraction
    step_latencies_ps: tuple[int, ...]
    event_counts: tuple[int, ...]
    attribution_totals_ps: tuple[int, ...]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    return parser


def _validate_frozen_literals() -> None:
    assert len(SERIAL_GRAPH_SHA256) == 64
    assert len(SERIAL_GOAL_SHA256) == 64
    assert EVIDENCE_FAMILIES == ("A", "B", "C", "D")
    for compute_ps in COMPUTE_PS:
        observed = FROZEN_JCT_PS[compute_ps]
        half_compute = compute_ps // 2
        half_communication = D_PS // 2
        assert observed["independent"] == max(compute_ps, D_PS)
        assert observed["serial"] == compute_ps + D_PS
        assert observed["pipeline"] == (
            half_compute
            + max(half_compute, half_communication)
            + half_communication
        )
        assert observed["independent"] < observed["pipeline"] < observed["serial"]
        assert observed["pipeline"] * 6 == observed["serial"] * 5
    assert 4_003 - 3_004 == 999


def _dims():
    from simllm.compute import ModelDims

    return ModelDims(
        num_layers=2,
        hidden_size=262_144,
        intermediate_size=1,
        num_heads=1,
        num_kv_heads=1,
        head_size=1,
        vocab_size=1,
        dtype_bytes=2,
    )


def _record(step_index: int, release_ps: int):
    from simllm.core import RequestPhase, ScheduledRequest, StepRecord

    phase = RequestPhase.PREFILL if step_index == 0 else RequestPhase.DECODE
    return StepRecord(
        step_index,
        release_ps,
        [ScheduledRequest("request", phase, 1, context_length=step_index + 1)],
        num_sampled=1,
        sampled_request_ids=["request"],
    )


def _compute_id(step_index: int, layer: int, rank: int) -> str:
    return f"step-{step_index}:layer-{layer}:rank-{rank}:compute"


def _collective_id(step_index: int, layer: int, site: str) -> str:
    return f"step-{step_index}:layer-{layer}:tp-{site}"


def _observations(record, compute_ps: int, shape: str):
    from simllm.core import (
        CollectiveWork,
        ComputeWork,
        ExecutionObservations,
        ExecutionOperation,
        OperationCorrelation,
    )

    if shape not in SHAPES:
        raise ValueError(f"unknown schedule shape {shape!r}")
    operations = []
    per_layer_ps = compute_ps // 2
    for layer in range(2):
        correlation = OperationCorrelation(
            request_ids=("request",),
            batch_id=f"step-{record.step_index}",
            layer=layer,
        )
        for rank in RANKS:
            local_dependencies = ()
            if layer > 0 and shape == "serial":
                local_dependencies = (
                    _collective_id(record.step_index, layer - 1, "mlp"),
                )
            operations.append(
                ExecutionOperation(
                    operation_id=_compute_id(record.step_index, layer, rank),
                    rank=rank,
                    logical_queue=f"cuda:{rank}:compute",
                    work=ComputeWork(
                        kernel=f"layer-{layer}",
                        nominal_duration_ps=per_layer_ps,
                    ),
                    correlation=correlation,
                    participant_local_depends_on=local_dependencies,
                )
            )
        for site in ("attention", "mlp"):
            local_dependencies = ()
            if shape != "independent" and site == "attention":
                local_dependencies = tuple(
                    _compute_id(record.step_index, layer, rank) for rank in RANKS
                )
            operations.append(
                ExecutionOperation(
                    operation_id=_collective_id(record.step_index, layer, site),
                    rank=RANKS[0],
                    logical_queue="cuda:0:nccl:tp",
                    work=CollectiveWork(
                        "all-reduce",
                        RANKS,
                        PAYLOAD_BYTES,
                        channel_hint=site,
                    ),
                    correlation=correlation,
                    participant_local_depends_on=local_dependencies,
                )
            )
    return ExecutionObservations(
        operations=tuple(operations),
        completion_operation_ids=(
            _compute_id(record.step_index, 1, RANKS[0]),
            _compute_id(record.step_index, 1, RANKS[1]),
            _collective_id(record.step_index, 1, "mlp"),
        ),
    )


def _run_schedule(compute_ps: int, shape: str) -> ScheduleResult:
    from simllm.backends import ObservedStepLowerer, SerialStepLowererConfig
    from simllm.core import (
        CoarseDeviceProfile,
        CoarseDeviceRuntime,
        CompletionReducer,
        VirtualClock,
    )

    dims = _dims()
    lowerer = ObservedStepLowerer(SerialStepLowererConfig(dims, RANKS))
    clock = VirtualClock(0)
    reducer = CompletionReducer(clock)
    runtime = CoarseDeviceRuntime(CoarseDeviceProfile())
    step_latencies = []
    event_counts = []
    attribution_totals = []
    ttft_ps = None
    tpot_ps = None
    for step_index in range(3):
        record = _record(step_index, clock.now_ps)
        observations = _observations(record, compute_ps, shape)
        graph = lowerer.lower(record, observations)
        streamed = []
        result = runtime.execute(graph, on_event=streamed.append)
        report = runtime.last_report
        assert report is not None
        assert tuple(streamed) == result.events
        step = reducer.reduce(record, graph, result, report)
        metric = step.request_metrics[0]
        step_latencies.append(step.step_latency_ps)
        event_counts.append(len(streamed))
        attribution_totals.append(metric.attribution.total_ps)
        ttft_ps = metric.ttft_ps
        if metric.tpot_ps is not None:
            tpot_ps = metric.tpot_ps
    assert ttft_ps is not None
    assert tpot_ps is not None
    return ScheduleResult(
        compute_ps=compute_ps,
        shape=shape,
        step_latency_ps=step_latencies[0],
        ttft_ps=ttft_ps,
        tpot_ps=tpot_ps,
        step_latencies_ps=tuple(step_latencies),
        event_counts=tuple(event_counts),
        attribution_totals_ps=tuple(attribution_totals),
    )


def _row(
    rows: list[EvidenceRow],
    family: str,
    evidence_class: str,
    case: str,
    expected: Any,
    measured: Any,
    *,
    genuine_risk: bool | None,
) -> None:
    rows.append(
        EvidenceRow(
            family=family,
            evidence_class=evidence_class,
            case=case,
            expected=expected,
            measured=measured,
            passed=measured == expected,
            genuine_risk=genuine_risk,
        )
    )


def _family_a_relations(
    rows: list[EvidenceRow],
    raw: dict[tuple[int, str], ScheduleResult],
) -> None:
    for compute_ps in COMPUTE_PS:
        observed = {shape: raw[(compute_ps, shape)].step_latency_ps for shape in SHAPES}
        pipeline_form = (
            compute_ps // 2
            + max(compute_ps // 2, D_PS // 2)
            + D_PS // 2
        )
        _row(
            rows,
            "A",
            BEHAVIORAL,
            f"independent=max:C={compute_ps}",
            max(compute_ps, D_PS),
            observed["independent"],
            genuine_risk=True,
        )
        _row(
            rows,
            "A",
            BEHAVIORAL,
            f"serial=sum:C={compute_ps}",
            compute_ps + D_PS,
            observed["serial"],
            genuine_risk=True,
        )
        _row(
            rows,
            "A",
            BEHAVIORAL,
            f"pipeline=form:C={compute_ps}",
            pipeline_form,
            observed["pipeline"],
            genuine_risk=True,
        )
        _row(
            rows,
            "A",
            BEHAVIORAL,
            f"pipeline=strictly-between:C={compute_ps}",
            True,
            observed["independent"] < observed["pipeline"] < observed["serial"],
            genuine_risk=True,
        )
        expected_gap = min(compute_ps, D_PS) // 2
        _row(
            rows,
            "A",
            BEHAVIORAL,
            f"symmetric-pipeline-gaps:C={compute_ps}",
            (expected_gap, expected_gap),
            (
                observed["pipeline"] - observed["independent"],
                observed["serial"] - observed["pipeline"],
            ),
            genuine_risk=True,
        )


def _family_b_relations(
    rows: list[EvidenceRow],
    raw: dict[tuple[int, str], ScheduleResult],
) -> None:
    for compute_ps in COMPUTE_PS:
        serial = raw[(compute_ps, "serial")]
        pipeline = raw[(compute_ps, "pipeline")]
        expected_reduction = min(compute_ps, D_PS) // 2
        _row(
            rows,
            "B",
            BEHAVIORAL,
            f"pipeline-ttft-reduction:C={compute_ps}",
            (expected_reduction, Fraction(5, 6)),
            (
                serial.ttft_ps - pipeline.ttft_ps,
                Fraction(pipeline.ttft_ps, serial.ttft_ps),
            ),
            genuine_risk=True,
        )
        _row(
            rows,
            "B",
            BEHAVIORAL,
            f"pipeline-tpot-reduction:C={compute_ps}",
            (expected_reduction, Fraction(5, 6)),
            (
                serial.tpot_ps - pipeline.tpot_ps,
                pipeline.tpot_ps / serial.tpot_ps,
            ),
            genuine_risk=True,
        )


def _channel_graph(channel_ids: tuple[str, str]):
    from simllm.core import CollectiveWork, ExecutionGraph, ExecutionOperation

    return ExecutionGraph(
        execution_id=f"channels-{channel_ids[0]}-{channel_ids[1]}",
        step_index=0,
        released_at_ps=0,
        operations=tuple(
            ExecutionOperation(
                operation_id=f"collective-{index}",
                rank=0,
                logical_queue=f"logical-{index}",
                work=CollectiveWork(
                    "all-reduce",
                    RANKS,
                    2,
                    "ring",
                    channel_id,
                ),
            )
            for index, channel_id in enumerate(channel_ids)
        ),
        completion_operation_ids=("collective-0", "collective-1"),
    )


def _run_channels(channel_ids: tuple[str, str]) -> dict[str, int]:
    from simllm.core import (
        CoarseDeviceProfile,
        CoarseDeviceRuntime,
        ResourceKind,
    )

    graph = _channel_graph(channel_ids)
    runtime = CoarseDeviceRuntime(
        CoarseDeviceProfile(
            rnic_rate_bps=8_000_000_000_000,
            nccl_channel_service_ps=1_000,
        )
    )
    result = runtime.execute(graph)
    report = runtime.last_report
    assert report is not None
    visits = {
        operation_id: tuple(
            visit
            for visit in report.visits
            if visit.operation_id == operation_id
            and visit.resource.kind is ResourceKind.NCCL_CHANNEL
        )
        for operation_id in ("collective-0", "collective-1")
    }
    return {
        "jct_ps": result.completed_at_ps,
        "first_0_started_ps": min(visit.started_at_ps for visit in visits["collective-0"]),
        "last_0_finished_ps": max(
            visit.finished_at_ps for visit in visits["collective-0"]
        ),
        "first_1_started_ps": min(visit.started_at_ps for visit in visits["collective-1"]),
    }


def _family_c_relations(
    rows: list[EvidenceRow],
    shared: dict[str, int],
    split: dict[str, int],
) -> None:
    _row(
        rows,
        "C",
        BEHAVIORAL,
        "shared-versus-split-first-channel-grant",
        (shared["last_0_finished_ps"], 0),
        (shared["first_1_started_ps"], split["first_1_started_ps"]),
        genuine_risk=True,
    )
    _row(
        rows,
        "C",
        BEHAVIORAL,
        "split-channel-jct-reduction",
        999,
        shared["jct_ps"] - split["jct_ps"],
        genuine_risk=True,
    )


def _exact_rows(
    rows: list[EvidenceRow],
    raw: dict[tuple[int, str], ScheduleResult],
    shared: dict[str, int],
    split: dict[str, int],
) -> None:
    for compute_ps in COMPUTE_PS:
        for shape in SHAPES:
            result = raw[(compute_ps, shape)]
            expected = FROZEN_JCT_PS[compute_ps][shape]
            for metric, measured in (
                ("jct", result.step_latency_ps),
                ("ttft", result.ttft_ps),
                ("tpot", result.tpot_ps),
            ):
                _row(
                    rows,
                    "A" if metric == "jct" else "B",
                    EXACT,
                    f"{metric}:C={compute_ps}:shape={shape}",
                    Fraction(expected, 1) if metric == "tpot" else expected,
                    measured,
                    genuine_risk=None,
                )
    for case, expected, measured in (
        ("shared-jct", 4_003, shared["jct_ps"]),
        ("split-jct", 3_004, split["jct_ps"]),
        ("shared-second-channel-start", 2_001, shared["first_1_started_ps"]),
        ("split-second-channel-start", 0, split["first_1_started_ps"]),
    ):
        _row(rows, "C", EXACT, case, expected, measured, genuine_risk=None)


def _serial_identity(rows: list[EvidenceRow]) -> None:
    from simllm.backends import ObservedStepLowerer, SerialStepLowererConfig
    from simllm.compute import ComputeProvider, DurationEstimate, ModelDims
    from simllm.core import (
        RequestPhase,
        ScheduledRequest,
        StepRecord,
        execution_graph_to_json,
    )
    from simllm.traffic import render_serial_execution_graph_goal

    class FlopProvider(ComputeProvider):
        def estimate(self, kernel, gpu):
            return DurationEstimate(duration_ps=int(kernel.flops), bound="compute")

    dims = ModelDims(2, 64, 128, 4, 4, 16, 256, 2)
    record = StepRecord(
        0,
        0,
        [
            ScheduledRequest(
                "p",
                RequestPhase.PREFILL,
                4,
                num_cached_tokens=4,
                context_length=8,
            ),
            ScheduledRequest("d", RequestPhase.DECODE, 1, context_length=32),
        ],
        num_sampled=None,
    )
    graph = ObservedStepLowerer(
        SerialStepLowererConfig(dims, (0, 1), provider=FlopProvider())
    ).lower(record)
    wire = (
        json.dumps(
            execution_graph_to_json(graph),
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()
    goal = render_serial_execution_graph_goal(graph).render().encode()
    for case, expected, measured in (
        ("serial-graph-bytes", 4_127, len(wire)),
        ("serial-graph-sha256", SERIAL_GRAPH_SHA256, hashlib.sha256(wire).hexdigest()),
        ("serial-goal-bytes", 1_880, len(goal)),
        ("serial-goal-sha256", SERIAL_GOAL_SHA256, hashlib.sha256(goal).hexdigest()),
    ):
        _row(rows, "D", FATAL, case, expected, measured, genuine_risk=None)


def _structural_guards(
    rows: list[EvidenceRow],
    raw: dict[tuple[int, str], ScheduleResult],
) -> None:
    from simllm.backends import ObservedStepLowerer, SerialStepLowererConfig
    from simllm.core import CollectiveWork, ComputeWork
    from simllm.traffic import lower_step_observations

    record = _record(0, 0)
    observations = _observations(record, COMPUTE_PS[0], "pipeline")
    graph = ObservedStepLowerer(SerialStepLowererConfig(_dims(), RANKS)).lower(
        record,
        observations,
    )
    preserved = all(
        (
            lowered.operation_id,
            lowered.rank,
            lowered.logical_queue,
            lowered.depends_on,
            lowered.participant_local_depends_on,
            lowered.not_before_ps,
            lowered.priority,
            lowered.correlation,
        )
        == (
            observed.operation_id,
            observed.rank,
            observed.logical_queue,
            observed.depends_on,
            observed.participant_local_depends_on,
            observed.not_before_ps,
            observed.priority,
            observed.correlation,
        )
        and (
            not isinstance(observed.work, ComputeWork)
            or lowered.work == observed.work
        )
        for observed, lowered in zip(
            observations.operations,
            graph.operations,
            strict=True,
        )
    )
    collectives = [
        operation.work
        for operation in graph.operations
        if isinstance(operation.work, CollectiveWork)
    ]
    guards = {
        "schedule-fields-preserved": preserved,
        "completion-frontier-preserved": (
            graph.completion_operation_ids == observations.completion_operation_ids
        ),
        "traffic-selected-ring": (
            len(collectives) == 4
            and all(work.algorithm_hint == "ring" for work in collectives)
        ),
        "traffic-payload-preserved": (
            all(work.payload_bytes == PAYLOAD_BYTES for work in collectives)
        ),
        "no-overlap-knob": not any(
            "overlap" in name or "discount" in name
            for name in inspect.signature(lower_step_observations).parameters
        ),
        "three-step-latencies-stable": all(
            len(set(result.step_latencies_ps)) == 1 for result in raw.values()
        ),
        "events-produced-every-step": all(
            all(count > 0 for count in result.event_counts) for result in raw.values()
        ),
        "request-attribution-conserves": all(
            result.attribution_totals_ps == result.step_latencies_ps
            for result in raw.values()
        ),
    }
    for case, measured in guards.items():
        _row(rows, "D", FATAL, case, True, measured, genuine_risk=None)


def _json_value(value: Any) -> Any:
    if isinstance(value, Fraction):
        return f"{value.numerator}/{value.denominator}"
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


def _observed_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _summary(rows: list[EvidenceRow]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for evidence_class in (EXACT, BEHAVIORAL, FATAL):
        selected = [row for row in rows if row.evidence_class == evidence_class]
        result[evidence_class] = {
            "passed": sum(row.passed for row in selected),
            "total": len(selected),
        }
    return result


def _run(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    raw = {
        (compute_ps, shape): _run_schedule(compute_ps, shape)
        for compute_ps in COMPUTE_PS
        for shape in SHAPES
    }
    shared = _run_channels(("shared", "shared"))
    split = _run_channels(("a", "b"))
    rows: list[EvidenceRow] = []

    # Entailment discipline: score raw relations before applying exact literals.
    _family_a_relations(rows, raw)
    _family_b_relations(rows, raw)
    _family_c_relations(rows, shared, split)
    _exact_rows(rows, raw, shared, split)
    _serial_identity(rows)
    _structural_guards(rows, raw)

    summary = _summary(rows)
    payload = {
        "expectations_commit": EXPECTATIONS_COMMIT,
        "observed_repository_commit": _observed_commit(),
        "relation_evaluation_order": "raw relations before entailing exact oracles",
        "configurations": [
            _json_value(asdict(raw[(compute_ps, shape)]))
            for compute_ps in COMPUTE_PS
            for shape in SHAPES
        ]
        + [
            {"family": "C", "channels": ["shared", "shared"], **shared},
            {"family": "C", "channels": ["a", "b"], **split},
        ],
        "evidence": [_json_value(asdict(row)) for row in rows],
        "summary": summary,
    }
    output_path = output_dir / "results.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if any(not row.passed for row in rows):
        failed = [row.case for row in rows if not row.passed]
        raise AssertionError(f"study evidence failed: {failed}")
    print(
        "study passed: "
        + ", ".join(
            f"{evidence_class}={counts['passed']}/{counts['total']}"
            for evidence_class, counts in summary.items()
        )
    )
    print(f"wrote {output_path}")


def main() -> int:
    args = _parser().parse_args()
    _validate_frozen_literals()
    if args.check_only:
        print("check-only: frozen overlap literals and CLI are valid")
        return 0
    _run(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
