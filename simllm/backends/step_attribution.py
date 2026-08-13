"""Per-request TTFT and TPOT attribution for the packet-level step sink.

:class:`~simllm.backends.step_sink.HtsimStepSink` runs the fabric on
``htsim_rnic`` and publishes one whole-step makespan plus, per step, the
ordered service of every artifact it executed. It publishes no per-request
endpoint, so this module does not invent one: every request co-scheduled in a
step is charged that step's whole service, exactly as
``per_request_fidelity_v1`` froze, and the decomposition is of the request's
own elapsed interval rather than a division of the makespan.

The reducer is a read-only projection. Every timestamp it reports comes from
the sink that executed the backend; it introduces no second timing authority
and advances no object. It reuses ``LatencyAttribution`` and ``RequestMetric``
unchanged, and it mirrors :class:`~simllm.core.completion.CompletionReducer`'s
interval rule: a scheduler gap charged to ``queue_ps``, a pending attribution
carried for a co-scheduled request that does not sample this step, and a hard
conservation check on every sampled interval.

Scope: the all-remote compatibility level, where every routed byte crosses the
fabric. A run whose locality projection reports NVLink bytes or NVLink service
is refused rather than approximated, because the composed service of a mixed
artifact is a maximum over two resources and this module would have to choose
one of them without evidence.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from fractions import Fraction

from simllm.backends.step_sink import StepLocalityOutcome
from simllm.core import (
    AdditiveVisitTotals,
    LatencyAttribution,
    RequestMetric,
    StepRecord,
    StepResult,
    sampled_request_ids,
)

#: The packet-level sink records no ``QueueVisit``, so the additive work sum
#: stays empty rather than being fabricated from wall-clock services. A caller
#: that needs the work sum must use a runtime authority that publishes visits.
_NO_VISIT_TOTALS = AdditiveVisitTotals()


def attribute_step(
    result: StepResult,
    locality: StepLocalityOutcome | None,
) -> LatencyAttribution:
    """Partition one packet-level step's makespan over its executed artifacts.

    Each executed artifact contributes its composed service to exactly one
    component: an artifact with nonzero fabric service is collective time, an
    artifact with zero fabric service is kernel time. The partition is complete
    and disjoint by construction, so the returned total equals
    ``result.step_latency_ps`` with no unattributed remainder.

    ``locality is None`` describes a step the sink did not simulate, which the
    adapter settles with its own fallback latency. That latency is charged to
    ``control_ps`` because no artifact accounts for it.
    """

    if not isinstance(result, StepResult):
        raise TypeError("result must be a StepResult")
    if locality is None:
        return LatencyAttribution(control_ps=result.step_latency_ps)
    if not isinstance(locality, StepLocalityOutcome):
        raise TypeError("locality must be a StepLocalityOutcome or None")
    if locality.step_index != result.step_index:
        raise ValueError("locality outcome belongs to another step")
    if locality.nvlink_directed_bytes or locality.nvlink_service_ps:
        raise ValueError(
            "packet-level request attribution requires the all-remote level; "
            "a mixed NVLink and fabric artifact has no single-resource owner"
        )
    composed = locality.composed_phase_service_ps
    fabric = locality.fabric_phase_service_ps
    if len(composed) != len(fabric):
        raise ValueError("composed and fabric artifact services disagree in length")
    kernel_ps = 0
    collective_ps = 0
    for composed_ps, fabric_ps in zip(composed, fabric, strict=True):
        if fabric_ps:
            collective_ps += composed_ps
        else:
            kernel_ps += composed_ps
    attribution = LatencyAttribution(kernel_ps=kernel_ps, collective_ps=collective_ps)
    if attribution.total_ps != result.step_latency_ps:
        raise ValueError(
            "executed artifact services do not conserve the step makespan"
        )
    return attribution


@dataclass
class _RequestState:
    arrived_at_ps: int
    accounted_through_ps: int
    first_token_at_ps: int | None = None
    last_token_at_ps: int | None = None
    token_count: int = 0
    inter_token_sum_ps: int = 0
    inter_token_count: int = 0
    pending: LatencyAttribution = field(default_factory=LatencyAttribution)
    ttft_attribution: LatencyAttribution = field(default_factory=LatencyAttribution)
    decode_attribution: LatencyAttribution = field(default_factory=LatencyAttribution)
    latest_metric: RequestMetric | None = None


@dataclass(frozen=True)
class RequestLatencyTotals:
    """One request's completed TTFT and TPOT with their conserved partitions."""

    request_id: str
    arrived_at_ps: int
    first_token_at_ps: int
    last_token_at_ps: int
    token_count: int
    ttft_ps: int
    tpot_ps: Fraction | None
    ttft_attribution: LatencyAttribution
    decode_attribution: LatencyAttribution

    def __post_init__(self) -> None:
        if self.ttft_attribution.total_ps != self.ttft_ps:
            raise ValueError("TTFT attribution does not conserve TTFT")
        expected_decode_ps = self.last_token_at_ps - self.first_token_at_ps
        if self.decode_attribution.total_ps != expected_decode_ps:
            raise ValueError("decode attribution does not conserve the decode span")
        if self.tpot_ps is None:
            if self.token_count != 1:
                raise ValueError("only a single-token request may omit TPOT")
        elif self.tpot_ps * (self.token_count - 1) != expected_decode_ps:
            raise ValueError("TPOT does not reproduce the attributed decode span")


class HtsimRequestMetricReducer:
    """Project packet-level step outcomes into per-request TTFT and TPOT.

    ``arrivals`` are the declared framework-entry timestamps that the admission
    gate released against, so a request's TTFT is queue plus service and never
    starts before the request exists. Feeding the study's own step-completion
    times without those arrivals is what produced negative TTFT in the earlier
    unregistered run.
    """

    def __init__(self, arrivals: Mapping[str, int]) -> None:
        parsed: dict[str, int] = {}
        for request_id, arrived_at_ps in arrivals.items():
            if not isinstance(request_id, str) or not request_id.strip():
                raise ValueError("arrival keys must be nonblank request identities")
            if isinstance(arrived_at_ps, bool) or not isinstance(arrived_at_ps, int):
                raise TypeError(f"arrival {request_id!r} must be an integer")
            if arrived_at_ps < 0:
                raise ValueError(f"arrival {request_id!r} must be nonnegative")
            parsed[request_id] = arrived_at_ps
        self._arrivals = parsed
        self._requests: dict[str, _RequestState] = {}
        self._consumed_step_indices: set[int] = set()

    @property
    def latest_request_metrics(self) -> tuple[RequestMetric, ...]:
        """The most recent completed metric for each observed request."""

        return tuple(
            state.latest_metric
            for state in self._requests.values()
            if state.latest_metric is not None
        )

    def totals(self) -> tuple[RequestLatencyTotals, ...]:
        """Completed per-request TTFT and TPOT with their conserved partitions."""

        rows = []
        for request_id, state in self._requests.items():
            if state.first_token_at_ps is None or state.last_token_at_ps is None:
                continue
            rows.append(
                RequestLatencyTotals(
                    request_id=request_id,
                    arrived_at_ps=state.arrived_at_ps,
                    first_token_at_ps=state.first_token_at_ps,
                    last_token_at_ps=state.last_token_at_ps,
                    token_count=state.token_count,
                    ttft_ps=state.first_token_at_ps - state.arrived_at_ps,
                    tpot_ps=(
                        Fraction(state.inter_token_sum_ps, state.inter_token_count)
                        if state.inter_token_count
                        else None
                    ),
                    ttft_attribution=state.ttft_attribution,
                    decode_attribution=state.decode_attribution,
                )
            )
        return tuple(rows)

    def consume(
        self,
        record: StepRecord,
        result: StepResult,
        locality: StepLocalityOutcome | None,
    ) -> tuple[RequestMetric, ...]:
        """Reduce one executed step and commit its request-metric history."""

        if not isinstance(record, StepRecord):
            raise TypeError("record must be a StepRecord")
        if not isinstance(result, StepResult):
            raise TypeError("result must be a StepResult")
        if result.step_index != record.step_index:
            raise ValueError("StepResult belongs to another StepRecord")
        if record.step_index in self._consumed_step_indices:
            raise ValueError(f"step index has already been reduced: {record.step_index}")
        released_at_ps = record.virtual_time_ps
        if result.completed_at_ps != released_at_ps + result.step_latency_ps:
            raise ValueError("StepResult completion disagrees with its own makespan")
        step_attribution = attribute_step(result, locality)

        sampled = sampled_request_ids(record)
        states = {
            request_id: replace(state)
            for request_id, state in self._requests.items()
        }
        metrics: list[RequestMetric] = []
        for scheduled in record.scheduled:
            request_id = scheduled.request_id
            state = states.get(request_id)
            if state is None:
                arrived_at_ps = self._arrivals.get(request_id)
                if arrived_at_ps is None:
                    raise ValueError(
                        f"scheduled request {request_id!r} has no declared arrival"
                    )
                if arrived_at_ps > released_at_ps:
                    raise ValueError(
                        f"scheduled request {request_id!r} predates its arrival at "
                        f"{arrived_at_ps} ps"
                    )
                state = _RequestState(
                    arrived_at_ps=arrived_at_ps,
                    accounted_through_ps=arrived_at_ps,
                )
                states[request_id] = state
            if released_at_ps < state.accounted_through_ps:
                raise ValueError(f"request {request_id!r} would move backward in time")
            interval = (
                state.pending
                + LatencyAttribution(
                    queue_ps=released_at_ps - state.accounted_through_ps
                )
                + step_attribution
            )
            state.accounted_through_ps = result.completed_at_ps

            if request_id not in sampled:
                state.pending = interval
                continue

            if state.first_token_at_ps is None:
                latency_ps = result.completed_at_ps - state.arrived_at_ps
                state.first_token_at_ps = result.completed_at_ps
                state.ttft_attribution = interval
                ttft_ps = latency_ps
            else:
                assert state.last_token_at_ps is not None
                latency_ps = result.completed_at_ps - state.last_token_at_ps
                ttft_ps = state.first_token_at_ps - state.arrived_at_ps
                state.inter_token_sum_ps += latency_ps
                state.inter_token_count += 1
                state.decode_attribution = state.decode_attribution + interval
            if interval.total_ps != latency_ps:
                raise ValueError(
                    f"request {request_id!r} interval attribution does not conserve "
                    "elapsed time"
                )
            state.token_count += 1
            state.last_token_at_ps = result.completed_at_ps
            metric = RequestMetric(
                request_id=request_id,
                phase=scheduled.phase,
                token_index=state.token_count,
                completed_at_ps=result.completed_at_ps,
                latency_ps=latency_ps,
                ttft_ps=ttft_ps,
                tpot_ps=(
                    Fraction(state.inter_token_sum_ps, state.inter_token_count)
                    if state.inter_token_count
                    else None
                ),
                attribution=interval,
                additive_visit_totals=_NO_VISIT_TOTALS,
            )
            state.latest_metric = metric
            state.pending = LatencyAttribution()
            metrics.append(metric)

        self._requests = states
        self._consumed_step_indices.add(record.step_index)
        return tuple(metrics)


__all__ = [
    "HtsimRequestMetricReducer",
    "RequestLatencyTotals",
    "attribute_step",
]
