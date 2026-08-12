"""Observation-aware step lowering and compute/communication overlap tests."""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import replace
from fractions import Fraction

import pytest

from simllm.backends import (
    ObservedStepLowerer,
    SerialStepLowererConfig,
)
from simllm.compute import ComputeProvider, DurationEstimate, ModelDims
from simllm.core import (
    CoarseDeviceProfile,
    CoarseDeviceRuntime,
    CollectiveWork,
    CompletionReducer,
    ComputeWork,
    DmaWork,
    ExecutionGraph,
    ExecutionLowerer,
    ExecutionObservations,
    ExecutionOperation,
    OperationCorrelation,
    RequestPhase,
    ResourceKind,
    ScheduledRequest,
    StepRecord,
    VirtualClock,
    execution_graph_to_json,
)
from simllm.traffic import lower_step_observations, render_serial_execution_graph_goal

RANKS = (0, 8)
PAYLOAD_BYTES = 524_288
D_PS = 41_943_040
COMPUTE_PS = (20_971_520, 83_886_080)
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

DIMS = ModelDims(
    num_layers=2,
    hidden_size=262_144,
    intermediate_size=1,
    num_heads=1,
    num_kv_heads=1,
    head_size=1,
    vocab_size=1,
    dtype_bytes=2,
)


class FlopProvider(ComputeProvider):
    def estimate(self, kernel, gpu):
        return DurationEstimate(duration_ps=int(kernel.flops), bound="compute")


def _record(
    step_index: int = 0,
    release_ps: int = 0,
    phase: RequestPhase = RequestPhase.PREFILL,
) -> StepRecord:
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


def _observations(
    record: StepRecord,
    compute_ps: int,
    shape: str,
) -> ExecutionObservations:
    if shape not in {"independent", "pipeline", "serial"}:
        raise ValueError(f"unknown schedule shape {shape!r}")
    operations = []
    per_layer_ps = compute_ps // DIMS.num_layers
    for layer in range(DIMS.num_layers):
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


def _lower(record: StepRecord, compute_ps: int, shape: str):
    return ObservedStepLowerer(SerialStepLowererConfig(DIMS, RANKS)).lower(
        record,
        _observations(record, compute_ps, shape),
    )


def test_observed_lowerer_uses_the_standard_protocol_and_preserves_schedule():
    record = _record()
    observations = _observations(record, COMPUTE_PS[0], "pipeline")
    lowerer = ObservedStepLowerer(SerialStepLowererConfig(DIMS, RANKS))

    graph = lowerer.lower(record, observations)

    assert isinstance(lowerer, ExecutionLowerer)
    assert tuple(operation.operation_id for operation in graph.operations) == tuple(
        operation.operation_id for operation in observations.operations
    )
    assert graph.completion_operation_ids == observations.completion_operation_ids
    for observed, lowered in zip(observations.operations, graph.operations, strict=True):
        assert lowered.operation_id == observed.operation_id
        assert lowered.rank == observed.rank
        assert lowered.logical_queue == observed.logical_queue
        assert lowered.depends_on == observed.depends_on
        assert lowered.participant_local_depends_on == observed.participant_local_depends_on
        assert lowered.not_before_ps == observed.not_before_ps
        assert lowered.priority == observed.priority
        assert lowered.correlation == observed.correlation
        if isinstance(observed.work, ComputeWork):
            assert lowered is observed
        else:
            assert isinstance(lowered.work, CollectiveWork)
            assert lowered.work.algorithm_hint == "ring"
            assert lowered.work.payload_bytes == PAYLOAD_BYTES


@pytest.mark.parametrize("compute_ps", COMPUTE_PS)
@pytest.mark.parametrize("shape", ("independent", "pipeline", "serial"))
def test_observed_step_runtime_matches_frozen_dependency_shapes(compute_ps, shape):
    graph = _lower(_record(), compute_ps, shape)

    result = CoarseDeviceRuntime(CoarseDeviceProfile()).execute(graph)

    assert result.completed_at_ps - graph.released_at_ps == FROZEN_JCT_PS[compute_ps][shape]


@pytest.mark.parametrize("compute_ps", COMPUTE_PS)
@pytest.mark.parametrize("shape", ("pipeline", "serial"))
def test_observed_step_overlap_reaches_ttft_and_tpot(compute_ps, shape):
    clock = VirtualClock(0)
    reducer = CompletionReducer(clock)
    runtime = CoarseDeviceRuntime(CoarseDeviceProfile())
    expected = FROZEN_JCT_PS[compute_ps][shape]

    for step_index in range(3):
        phase = RequestPhase.PREFILL if step_index == 0 else RequestPhase.DECODE
        record = _record(step_index, clock.now_ps, phase)
        graph = _lower(record, compute_ps, shape)
        result = runtime.execute(graph)
        assert runtime.last_report is not None
        step = reducer.reduce(record, graph, result, runtime.last_report)
        metric = step.request_metrics[0]

        assert step.step_latency_ps == expected
        assert metric.ttft_ps == expected
        assert metric.tpot_ps == (
            None if step_index == 0 else Fraction(expected, 1)
        )


def _channel_graph(channel_ids: tuple[str, str]) -> ExecutionGraph:
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


def _run_channels(channel_ids: tuple[str, str]):
    runtime = CoarseDeviceRuntime(
        CoarseDeviceProfile(
            rnic_rate_bps=8_000_000_000_000,
            nccl_channel_service_ps=1_000,
        )
    )
    result = runtime.execute(_channel_graph(channel_ids))
    assert runtime.last_report is not None
    visits = {
        operation_id: tuple(
            visit
            for visit in runtime.last_report.visits
            if visit.operation_id == operation_id
            and visit.resource.kind is ResourceKind.NCCL_CHANNEL
        )
        for operation_id in ("collective-0", "collective-1")
    }
    return result.completed_at_ps, visits


def test_nccl_channel_contention_reduces_legal_overlap_exactly():
    shared_jct, shared = _run_channels(("shared", "shared"))
    split_jct, split = _run_channels(("a", "b"))

    shared_last_first = max(
        visit.finished_at_ps for visit in shared["collective-0"]
    )
    shared_first_second = min(
        visit.started_at_ps for visit in shared["collective-1"]
    )
    split_first_second = min(
        visit.started_at_ps for visit in split["collective-1"]
    )
    assert shared_last_first == shared_first_second == 2_001
    assert split_first_second == 0
    assert shared_jct == 4_003
    assert split_jct == 3_004
    assert shared_jct - split_jct == 999


def test_lowering_rejects_missing_duplicate_and_mismatched_collectives():
    record = _record()
    observations = _observations(record, COMPUTE_PS[0], "pipeline")
    collective_index = next(
        index
        for index, operation in enumerate(observations.operations)
        if isinstance(operation.work, CollectiveWork)
    )
    collective = observations.operations[collective_index]

    missing = replace(
        observations,
        operations=observations.operations[:collective_index]
        + observations.operations[collective_index + 1 :],
    )
    with pytest.raises(ValueError, match="missing planned collective sites"):
        lower_step_observations(record, DIMS, RANKS, missing)

    duplicate = replace(
        observations,
        operations=(*observations.operations, collective),
    )
    with pytest.raises(ValueError, match="duplicate collective site"):
        lower_step_observations(record, DIMS, RANKS, duplicate)

    assert isinstance(collective.work, CollectiveWork)
    wrong_payload = replace(
        observations,
        operations=(
            *observations.operations[:collective_index],
            replace(
                collective,
                work=replace(collective.work, payload_bytes=PAYLOAD_BYTES + 1),
            ),
            *observations.operations[collective_index + 1 :],
        ),
    )
    with pytest.raises(ValueError, match="payload disagrees"):
        lower_step_observations(record, DIMS, RANKS, wrong_payload)

    wrong_site = replace(
        observations,
        operations=(
            *observations.operations[:collective_index],
            replace(
                collective,
                work=replace(collective.work, channel_hint="unknown"),
            ),
            *observations.operations[collective_index + 1 :],
        ),
    )
    with pytest.raises(ValueError, match="absent from the step plan"):
        lower_step_observations(record, DIMS, RANKS, wrong_site)


def test_lowering_rejects_non_compute_non_collective_observations():
    record = StepRecord(
        0,
        0,
        [ScheduledRequest("request", RequestPhase.PREFILL, 1)],
    )
    observations = ExecutionObservations(
        (
            ExecutionOperation(
                "dma",
                0,
                "cuda:0:copy",
                DmaWork("dma", "host", "gpu:0", 1),
            ),
        )
    )

    with pytest.raises(TypeError, match="supports only ComputeWork and CollectiveWork"):
        lower_step_observations(record, replace(DIMS, num_layers=0), (0,), observations)


def test_overlap_api_has_no_percentage_or_duration_discount_parameter():
    names = set(inspect.signature(lower_step_observations).parameters)
    assert not any("overlap" in name or "discount" in name for name in names)


def test_observed_lowerer_absent_observations_preserves_serial_artifacts():
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

    assert len(wire) == 4_127
    assert hashlib.sha256(wire).hexdigest() == (
        "aa3c836fe559973a7bf0940384c2e8a84e6af84e0fbd2c02d3b89774ee0c8e2d"
    )
    assert len(goal) == 1_880
    assert hashlib.sha256(goal).hexdigest() == (
        "7087db6780f7e34f5a559a6505eeccc15d984c7b478cd8f0bc5838053825d4b6"
    )
