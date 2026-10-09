"""Deferred pipeline steps on the existing persistent device runtime."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from simllm.backends.step_lowerer import SerialStepLowerer, SerialStepLowererConfig
from simllm.compute import HostInitiationModel
from simllm.core import (
    CoarseDeviceRuntime,
    ExecutionGraph,
    ExecutionResult,
    ResourceKind,
    RuntimeReport,
    StepRecord,
    StepResult,
)
from simllm.traffic.pipeline import compose_pipeline_graph


@dataclass(frozen=True)
class PipelineStageOutcome:
    """Stage timing projected from the runtime, with parent-step identity."""

    step_index: int
    stage_index: int
    layer_range: tuple[int, int]
    ranks: tuple[int, ...]
    request_ids: tuple[str, ...]
    eligible_at_ps: int
    started_at_ps: int
    completed_at_ps: int


@dataclass(frozen=True)
class PipelineStepOutcome:
    record: StepRecord
    graph: ExecutionGraph
    execution_result: ExecutionResult
    runtime_report: RuntimeReport
    step_result: StepResult
    stages: tuple[PipelineStageOutcome, ...]


class PipelineRuntimeStepSink:
    """Reserve work without advancing the adapter's clock.

    The adapter binds its stage geometry once, then consumes the returned
    completion through its FIFO futures. ``rank_map`` maps the logical world
    into the selected runtime's endpoints. The default coarse runtime has
    eight slots per node; this is a declared topology, not an AMD profile.
    """

    def __init__(
        self,
        *,
        runtime: CoarseDeviceRuntime | None = None,
        rank_map: Sequence[int] | None = None,
        host_model: HostInitiationModel | None = None,
    ) -> None:
        self.runtime = runtime if runtime is not None else CoarseDeviceRuntime()
        self.rank_map = None if rank_map is None else tuple(rank_map)
        if self.rank_map is not None and (
            len(set(self.rank_map)) != len(self.rank_map)
            or any(type(rank) is not int or rank < 0 for rank in self.rank_map)
        ):
            raise ValueError("rank_map must contain distinct nonnegative integers")
        self.host_model = host_model or HostInitiationModel.ideal()
        self._lowerers: tuple[SerialStepLowerer, ...] = ()
        self._outcomes: list[PipelineStepOutcome] = []

    @property
    def outcomes(self) -> tuple[PipelineStepOutcome, ...]:
        """Immutable projections in submission order, including pending work."""
        return tuple(self._outcomes)

    def bind_pipeline(
        self,
        configs: Sequence[SerialStepLowererConfig],
        layer_ranges: Sequence[tuple[int, int]],
        boundary_builder: Callable[[StepRecord], Sequence[ExecutionGraph]],
    ) -> None:
        if self._lowerers:
            raise RuntimeError("pipeline sink is already bound to an executor")
        configs = tuple(configs)
        ranges = tuple(layer_ranges)
        if len(configs) < 2 or len(configs) != len(ranges):
            raise ValueError("one configuration and layer range per PP stage is required")
        if any(config.host_model != self.host_model for config in configs):
            raise ValueError("pipeline and adapter host models must agree")
        self._lowerers = tuple(SerialStepLowerer(config) for config in configs)
        self._ranges = ranges
        self._boundary_builder = boundary_builder

    def __call__(self, record: StepRecord) -> StepResult:
        raise RuntimeError("PP steps require prepare() and FIFO completion consumption")

    def prepare(self, record: StepRecord) -> PipelineStepOutcome:
        if not self._lowerers:
            raise RuntimeError("bind_pipeline() is required before submission")
        if not record.total_new_tokens:
            raise ValueError("drains carry no pipeline work")
        stages = tuple(tuple(lowerer.config.tp_ranks) for lowerer in self._lowerers)
        graph = compose_pipeline_graph(
            record, self._lowerers[0].config.dims, len(stages), stages,
            tuple(lowerer.lower(record) for lowerer in self._lowerers),
            boundary_graphs=self._boundary_builder(record),
        )
        result = self.runtime.execute(graph)
        report = self.runtime.last_report
        assert report is not None
        stage_outcomes = []
        for index, (ranks, layer_range) in enumerate(zip(stages, self._ranges)):
            prefix = f"{graph.execution_id}:pp-stage-{index}:"
            operations = [op for op in report.operations if op.operation_id.startswith(prefix)]
            stage_outcomes.append(PipelineStageOutcome(
                record.step_index, index, layer_range, ranks,
                tuple(request.request_id for request in record.scheduled),
                min(op.eligible_at_ps for op in operations),
                min(visit.started_at_ps for visit in report.visits
                    if visit.operation_id.startswith(prefix)
                    and visit.resource.kind is ResourceKind.GPU_WORK_QUEUE),
                max(op.completed_at_ps for op in operations),
            ))
        outcome = PipelineStepOutcome(
            record, graph, result, report,
            StepResult(record.step_index, result.completed_at_ps - record.virtual_time_ps,
                       result.completed_at_ps),
            tuple(stage_outcomes),
        )
        self._outcomes.append(outcome)
        return outcome
