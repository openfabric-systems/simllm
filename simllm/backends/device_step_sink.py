"""Observation-aware step sink backed by the shared coarse device runtime."""

from __future__ import annotations

from dataclasses import dataclass

from simllm.backends.step_lowerer import ObservedStepLowerer, SerialStepLowererConfig
from simllm.core import (
    CoarseDeviceRuntime,
    CompletionReducer,
    ExecutionGraph,
    ExecutionObservations,
    ExecutionResult,
    RuntimeReport,
    StepRecord,
    StepResult,
    VirtualClock,
)


@dataclass(frozen=True)
class DeviceStepOutcome:
    """Read-only evidence for one call through :class:`DeviceRuntimeStepSink`."""

    record: StepRecord
    observations: ExecutionObservations | None
    graph: ExecutionGraph
    execution_result: ExecutionResult
    runtime_report: RuntimeReport
    step_result: StepResult


class DeviceRuntimeStepSink:
    """Run a framework step through observed lowering and completion reduction.

    This sink owns one persistent runtime and one request-metric history. The
    adapter supplies the same ``VirtualClock`` used to timestamp the
    ``StepRecord``. Observations select traffic binding; ``None`` delegates to
    the serial lowerer through :class:`ObservedStepLowerer`.
    """

    def __init__(
        self,
        config: SerialStepLowererConfig,
        *,
        runtime: CoarseDeviceRuntime | None = None,
    ) -> None:
        if not isinstance(config, SerialStepLowererConfig):
            raise TypeError("config must be a SerialStepLowererConfig")
        if runtime is not None and not isinstance(runtime, CoarseDeviceRuntime):
            raise TypeError("runtime must be a CoarseDeviceRuntime")
        self.lowerer = ObservedStepLowerer(config)
        self.runtime = runtime or CoarseDeviceRuntime()
        self._clock: VirtualClock | None = None
        self._reducer: CompletionReducer | None = None
        self._outcomes: list[DeviceStepOutcome] = []

    @property
    def clock(self) -> VirtualClock | None:
        """The adapter clock after binding, otherwise ``None``."""

        return self._clock

    @property
    def outcomes(self) -> tuple[DeviceStepOutcome, ...]:
        """Completed calls in submission order."""

        return tuple(self._outcomes)

    def bind_clock(self, clock: VirtualClock) -> None:
        """Bind the adapter's sole clock before the first step."""

        if not isinstance(clock, VirtualClock):
            raise TypeError("clock must be a VirtualClock")
        if self._clock is clock:
            return
        if self._clock is not None:
            raise RuntimeError("DeviceRuntimeStepSink is already bound to another clock")
        if self._outcomes:
            raise RuntimeError("cannot bind a clock after step execution")
        self._clock = clock
        self._reducer = CompletionReducer(clock)

    def __call__(
        self,
        record: StepRecord,
        observations: ExecutionObservations | None,
    ) -> StepResult:
        """Lower, execute, and reduce one authoritative step."""

        if not isinstance(record, StepRecord):
            raise TypeError("record must be a StepRecord")
        if observations is not None and not isinstance(
            observations, ExecutionObservations
        ):
            raise TypeError("observations must be ExecutionObservations or None")
        if self._clock is None or self._reducer is None:
            raise RuntimeError("bind_clock() must be called before step execution")
        if self._clock.now_ps != record.virtual_time_ps:
            raise ValueError("bound clock does not equal StepRecord virtual time")

        graph = self.lowerer.lower(record, observations)
        execution_result = self.runtime.execute(graph)
        runtime_report = self.runtime.last_report
        if runtime_report is None:
            raise RuntimeError("device runtime returned no RuntimeReport")
        step_result = self._reducer.reduce(
            record,
            graph,
            execution_result,
            runtime_report,
        )
        self._outcomes.append(
            DeviceStepOutcome(
                record=record,
                observations=observations,
                graph=graph,
                execution_result=execution_result,
                runtime_report=runtime_report,
                step_result=step_result,
            )
        )
        return step_result


__all__ = ["DeviceRuntimeStepSink", "DeviceStepOutcome"]
