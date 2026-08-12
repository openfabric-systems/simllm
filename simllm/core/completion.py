"""Completion feedback from one runtime authority to scheduler metrics."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from fractions import Fraction
from itertools import pairwise

from simllm.core.bookkeeping import (
    BookkeepingLedger,
    framework_request_arrivals,
)
from simllm.core.clock import VirtualClock
from simllm.core.execution import (
    EventPhase,
    ExecutionGraph,
    ExecutionResult,
)
from simllm.core.execution_io import operation_participant_ranks
from simllm.core.request_lifetime import RequestLifetimeRegistry
from simllm.core.runtime import (
    QueueVisit,
    RuntimeCriticalSegment,
    RuntimeOperationRecord,
    RuntimeReport,
)
from simllm.core.step import (
    AdditiveVisitTotals,
    LatencyAttribution,
    RequestMetric,
    RequestPhase,
    StepRecord,
    StepResult,
)


@dataclass
class _RequestMetricState:
    first_observed_at_ps: int
    accounted_through_ps: int
    first_token_at_ps: int | None = None
    last_token_at_ps: int | None = None
    token_count: int = 0
    inter_token_sum_ps: int = 0
    inter_token_count: int = 0
    pending_attribution: LatencyAttribution = field(
        default_factory=LatencyAttribution
    )
    pending_additive: AdditiveVisitTotals = field(
        default_factory=AdditiveVisitTotals
    )
    latest_metric: RequestMetric | None = None


def _visit_totals(visits: tuple[QueueVisit, ...]) -> AdditiveVisitTotals:
    return AdditiveVisitTotals(
        queue_wait_ps=sum(visit.queue_wait_ps for visit in visits),
        service_ps=sum(visit.service_ps for visit in visits),
        visibility_ps=sum(visit.visibility_ps for visit in visits),
        visit_count=len(visits),
    )


def _validate_scalar_projection(
    record: RuntimeOperationRecord,
    by_id: dict[str, RuntimeOperationRecord],
    released_at_ps: int,
) -> None:
    """Check one operation's scalar fields against its critical segments.

    The segments are the conservation authority. The operation-level
    completion, causal boundary, additive predecessor and critical-path
    breakdown are compatibility projections of that authority, so each one is
    derived here from the segments and required to agree exactly. An
    asynchronous operation may release the framework at a participant
    completion earlier than its physical maximum, but it may never report a
    timestamp the segments do not carry.
    """

    name = record.operation_id
    segments = record.critical_segments
    if not segments:
        raise ValueError(f"operation {name!r} has no critical segment")
    completions = [segment.completed_at_ps for segment in segments]
    if record.physical_completed_at_ps != max(completions):
        raise ValueError(
            f"operation {name!r} physical completion disagrees with its "
            "critical segments"
        )
    if record.completed_at_ps not in completions:
        raise ValueError(
            f"operation {name!r} completion is not a participant critical-segment "
            "completion"
        )
    causal_id = record.causal_predecessor_id
    boundary_ps = record.causal_predecessor_completed_at_ps
    if (causal_id is None) != (boundary_ps is None):
        raise ValueError(
            f"operation {name!r} causal predecessor identity and boundary disagree"
        )
    if causal_id is None:
        if record.critical_predecessor_id is not None:
            raise ValueError(
                f"operation {name!r} names an additive predecessor without a causal one"
            )
        segment_start_ps = released_at_ps
    else:
        predecessor = by_id.get(causal_id)
        if predecessor is None:
            raise ValueError(
                f"operation {name!r} names an unknown causal predecessor"
            )
        if boundary_ps not in {
            segment.completed_at_ps for segment in predecessor.critical_segments
        }:
            raise ValueError(
                f"operation {name!r} causal boundary is not a participant completion "
                f"of {causal_id!r}"
            )
        if not any(
            segment.predecessor_operation_id == causal_id
            and segment.started_at_ps == boundary_ps
            for segment in segments
        ):
            raise ValueError(
                f"operation {name!r} has no critical segment admitted from "
                f"{causal_id!r} at its causal boundary"
            )
        expected_additive_id = (
            causal_id if boundary_ps == predecessor.completed_at_ps else None
        )
        if record.critical_predecessor_id != expected_additive_id:
            raise ValueError(
                f"operation {name!r} additive predecessor disagrees with the "
                "critical segment that admitted it"
            )
        segment_start_ps = (
            boundary_ps if expected_additive_id is not None else released_at_ps
        )
    if record.breakdown.operation_latency_ps != record.completed_at_ps - segment_start_ps:
        raise ValueError(
            f"operation {name!r} critical-path breakdown does not span its own "
            "projected segment"
        )


class CompletionReducer:
    """Reduce authoritative runtime evidence and advance one virtual clock.

    The reducer owns request-metric history only. It neither schedules an
    operation nor advances a physical object. All timestamps and realized
    predecessor choices come from the supplied runtime result and report.
    """

    def __init__(
        self,
        clock: VirtualClock,
        *,
        bookkeeping: BookkeepingLedger | None = None,
        lifetimes: RequestLifetimeRegistry | None = None,
    ) -> None:
        if not isinstance(clock, VirtualClock):
            raise TypeError("clock must be a VirtualClock")
        if bookkeeping is not None and not isinstance(bookkeeping, BookkeepingLedger):
            raise TypeError("bookkeeping must be a BookkeepingLedger or None")
        if lifetimes is not None and not isinstance(
            lifetimes, RequestLifetimeRegistry
        ):
            raise TypeError("lifetimes must be a RequestLifetimeRegistry or None")
        self.clock = clock
        self.lifetimes = lifetimes
        self._request_arrivals = (
            None
            if bookkeeping is None
            else {
                arrival.request_id: arrival.arrived_at_ps
                for arrival in framework_request_arrivals(bookkeeping)
            }
        )
        self._requests: dict[str, _RequestMetricState] = {}
        self._consumed_execution_ids: set[str] = set()

    @property
    def latest_request_metrics(self) -> tuple[RequestMetric, ...]:
        """Return the latest completed metric for each observed request."""

        return tuple(
            state.latest_metric
            for state in self._requests.values()
            if state.latest_metric is not None
        )

    @staticmethod
    def _sampled_request_ids(record: StepRecord) -> set[str]:
        scheduled_ids = [request.request_id for request in record.scheduled]
        if len(scheduled_ids) != len(set(scheduled_ids)):
            raise ValueError("StepRecord.scheduled contains duplicate request IDs")

        explicit = record.sampled_request_ids
        if explicit is not None:
            if len(explicit) != len(set(explicit)):
                raise ValueError("sampled_request_ids must be unique")
            unknown = sorted(set(explicit) - set(scheduled_ids))
            if unknown:
                raise ValueError(
                    f"sampled_request_ids contains unscheduled requests: {unknown}"
                )
            if record.num_sampled is not None and len(explicit) != record.num_sampled:
                raise ValueError(
                    "sampled_request_ids cardinality must equal num_sampled"
                )
            return set(explicit)

        if record.num_sampled is None:
            return set(scheduled_ids)
        if record.num_sampled == 0:
            return set()
        if record.num_sampled == len(scheduled_ids):
            return set(scheduled_ids)

        decode_ids = {
            request.request_id
            for request in record.scheduled
            if request.phase is RequestPhase.DECODE
        }
        if record.num_sampled < len(decode_ids):
            raise ValueError("num_sampled is smaller than the scheduled decode set")
        if record.num_sampled == len(decode_ids):
            return decode_ids
        raise ValueError(
            "partial sampled-request identity is ambiguous; provide "
            "StepRecord.sampled_request_ids (CORE-17)"
        )

    @staticmethod
    def _validate_inputs(
        record: StepRecord,
        graph: ExecutionGraph,
        result: ExecutionResult,
        report: RuntimeReport,
        clock: VirtualClock,
    ) -> tuple[
        dict[str, RuntimeOperationRecord],
        tuple[str, ...],
        dict[tuple[str, int], RuntimeCriticalSegment],
    ]:
        if not isinstance(record, StepRecord):
            raise TypeError("record must be a StepRecord")
        if not isinstance(graph, ExecutionGraph):
            raise TypeError("graph must be an ExecutionGraph")
        if not isinstance(result, ExecutionResult):
            raise TypeError("result must be an ExecutionResult")
        if not isinstance(report, RuntimeReport):
            raise TypeError("report must be a RuntimeReport")
        if graph.step_index != record.step_index:
            raise ValueError("graph step index disagrees with StepRecord")
        if graph.released_at_ps != record.virtual_time_ps:
            raise ValueError("graph release disagrees with StepRecord virtual time")
        if clock.now_ps != record.virtual_time_ps:
            raise ValueError("VirtualClock does not equal StepRecord virtual time")
        if result.execution_id != graph.execution_id:
            raise ValueError("ExecutionResult execution ID disagrees with graph")
        if report.execution_id != graph.execution_id:
            raise ValueError("RuntimeReport execution ID disagrees with graph")
        if result.completed_at_ps < graph.released_at_ps:
            raise ValueError("ExecutionResult completes before graph release")
        if (
            result.quiesced_at_ps is not None
            and result.quiesced_at_ps < result.completed_at_ps
        ):
            raise ValueError("ExecutionResult quiescence precedes completion")

        operation_ids = tuple(operation.operation_id for operation in graph.operations)
        report_ids = tuple(operation.operation_id for operation in report.operations)
        if report_ids != operation_ids:
            raise ValueError("RuntimeReport operation order or cardinality disagrees")
        by_id = {operation.operation_id: operation for operation in report.operations}
        if len(by_id) != len(report.operations):
            raise ValueError("RuntimeReport contains duplicate operation IDs")
        graph_operation_by_id = {
            operation.operation_id: operation for operation in graph.operations
        }
        critical_segments: dict[tuple[str, int], RuntimeCriticalSegment] = {}
        for operation_record in report.operations:
            if (
                operation_record.attribution.total_ps
                != operation_record.breakdown.operation_latency_ps
            ):
                raise ValueError(
                    f"operation {operation_record.operation_id!r} attribution does not conserve"
                )
            predecessor = operation_record.critical_predecessor_id
            if predecessor is not None and predecessor not in by_id:
                raise ValueError(
                    f"operation {operation_record.operation_id!r} names an unknown predecessor"
                )
            graph_operation = graph_operation_by_id[operation_record.operation_id]
            expected_ranks = operation_participant_ranks(graph_operation)
            segment_ranks = tuple(
                segment.participant_rank
                for segment in operation_record.critical_segments
            )
            if segment_ranks != expected_ranks:
                raise ValueError(
                    f"operation {operation_record.operation_id!r} critical segments "
                    "disagree with canonical participants"
                )
            participant_completions = dict(
                operation_record.participant_completed_at_ps
            )
            if tuple(sorted(participant_completions)) != expected_ranks:
                raise ValueError(
                    f"operation {operation_record.operation_id!r} participant "
                    "completions disagree with canonical participants"
                )
            for segment in operation_record.critical_segments:
                if segment.operation_id != operation_record.operation_id:
                    raise ValueError("critical segment operation identity disagrees")
                if (
                    segment.completed_at_ps
                    != participant_completions[segment.participant_rank]
                ):
                    raise ValueError("critical segment participant completion disagrees")
                key = (segment.operation_id, segment.participant_rank)
                if key in critical_segments:
                    raise ValueError("RuntimeReport contains duplicate critical segments")
                critical_segments[key] = segment

        for segment in critical_segments.values():
            if segment.predecessor_operation_id is None:
                if segment.started_at_ps != graph.released_at_ps:
                    raise ValueError(
                        "root critical segment does not start at graph release"
                    )
                continue
            assert segment.predecessor_participant_rank is not None
            predecessor_key = (
                segment.predecessor_operation_id,
                segment.predecessor_participant_rank,
            )
            predecessor_segment = critical_segments.get(predecessor_key)
            if predecessor_segment is None:
                raise ValueError("critical segment names an unknown predecessor segment")
            if segment.started_at_ps != predecessor_segment.completed_at_ps:
                raise ValueError("critical segment predecessor timestamp disagrees")

        # CORE-46: the scalar operation-level fields are a read-only projection
        # of the participant-keyed segments above, so they are joined by the
        # stable (operation, participant rank) identity and checked for loss,
        # duplication and timestamp disagreement rather than trusted.
        for operation_record in report.operations:
            _validate_scalar_projection(operation_record, by_id, graph.released_at_ps)

        report_segment_chain = report.realized_critical_path_segments
        if any(key not in critical_segments for key in report_segment_chain):
            raise ValueError("RuntimeReport critical path names an unknown segment")
        if tuple(key[0] for key in report_segment_chain) != (
            report.realized_critical_path_operation_ids
        ):
            raise ValueError("RuntimeReport critical path projections disagree")
        for predecessor_key, segment_key in pairwise(report_segment_chain):
            segment = critical_segments[segment_key]
            if (
                segment.predecessor_operation_id,
                segment.predecessor_participant_rank,
            ) != predecessor_key:
                raise ValueError("RuntimeReport critical segment chain is discontinuous")
        if report_segment_chain:
            report_chain_latency_ps = sum(
                critical_segments[key].breakdown.operation_latency_ps
                for key in report_segment_chain
            )
            report_endpoint = critical_segments[report_segment_chain[-1]]
            if (
                report_chain_latency_ps
                != report_endpoint.completed_at_ps - graph.released_at_ps
            ):
                raise ValueError("RuntimeReport critical segment chain does not conserve")
        report_queue_ps = sum(
            critical_segments[key].breakdown.critical_path_queue_ps
            for key in report_segment_chain
        )
        if report_queue_ps != report.critical_path_queue_ps:
            raise ValueError("RuntimeReport critical-path queue total disagrees")

        if any(visit.execution_id != graph.execution_id for visit in report.visits):
            raise ValueError("RuntimeReport visit execution ID disagrees with graph")
        if any(visit.operation_id not in by_id for visit in report.visits):
            raise ValueError("RuntimeReport visit names an unknown operation")
        visit_totals = _visit_totals(report.visits)
        if visit_totals.queue_wait_ps != report.sum_visit_wait_ps:
            raise ValueError("RuntimeReport additive queue-wait total disagrees")

        last_timestamp = -1
        for event in result.events:
            if event.execution_id != graph.execution_id:
                raise ValueError("completion event execution ID disagrees with graph")
            if event.operation_id not in by_id:
                raise ValueError("completion event names an unknown operation")
            if event.timestamp_ps < last_timestamp:
                raise ValueError("completion events are not timestamp ordered")
            if result.quiesced_at_ps is not None and event.timestamp_ps > result.quiesced_at_ps:
                raise ValueError("completion event occurs after physical quiescence")
            last_timestamp = event.timestamp_ps

        required_ids = graph.completion_operation_ids or operation_ids
        boundary = max(
            (by_id[operation_id].completed_at_ps for operation_id in required_ids),
            default=graph.released_at_ps,
        )
        if boundary != result.completed_at_ps:
            raise ValueError("ExecutionResult boundary disagrees with required operations")
        for operation_id in operation_ids:
            expected = by_id[operation_id].completed_at_ps
            logical_events = [
                event
                for event in result.events
                if event.operation_id == operation_id
                and event.subject_object_id is None
                and event.phase is EventPhase.COMPLETED
            ]
            if len(logical_events) != 1 or logical_events[0].timestamp_ps != expected:
                raise ValueError(
                    f"operation {operation_id!r} has no unique matching completion event"
                )
        return by_id, required_ids, critical_segments

    @staticmethod
    def _critical_chain(
        endpoint_id: str,
        by_id: dict[str, RuntimeOperationRecord],
        critical_segments: dict[tuple[str, int], RuntimeCriticalSegment],
        released_at_ps: int,
    ) -> tuple[tuple[tuple[str, int], ...], LatencyAttribution]:
        endpoint_record = by_id[endpoint_id]
        endpoint_segment = max(
            (
                segment
                for segment in endpoint_record.critical_segments
                if segment.completed_at_ps == endpoint_record.completed_at_ps
            ),
            key=lambda segment: segment.participant_rank,
        )
        reverse_chain: list[tuple[str, int]] = []
        seen: set[tuple[str, int]] = set()
        current: tuple[str, int] | None = (
            endpoint_segment.operation_id,
            endpoint_segment.participant_rank,
        )
        while current is not None:
            if current in seen:
                raise ValueError("request critical predecessor chain contains a cycle")
            seen.add(current)
            reverse_chain.append(current)
            segment = critical_segments[current]
            if segment.predecessor_operation_id is None:
                current = None
            else:
                assert segment.predecessor_participant_rank is not None
                current = (
                    segment.predecessor_operation_id,
                    segment.predecessor_participant_rank,
                )
        chain = tuple(reversed(reverse_chain))
        attribution = LatencyAttribution()
        for key in chain:
            attribution += critical_segments[key].attribution
        expected_latency = by_id[endpoint_id].completed_at_ps - released_at_ps
        if attribution.total_ps != expected_latency:
            raise ValueError(
                f"request endpoint {endpoint_id!r} critical chain does not conserve"
            )
        return chain, attribution

    @staticmethod
    def _request_additive_totals(
        request_id: str,
        graph: ExecutionGraph,
        report: RuntimeReport,
    ) -> AdditiveVisitTotals:
        operation_ids = {
            operation.operation_id
            for operation in graph.operations
            if request_id in operation.correlation.request_ids
        }
        return _visit_totals(
            tuple(
                visit for visit in report.visits if visit.operation_id in operation_ids
            )
        )

    def reduce(
        self,
        record: StepRecord,
        graph: ExecutionGraph,
        result: ExecutionResult,
        report: RuntimeReport,
    ) -> StepResult:
        """Return one scheduler result and atomically commit metric history."""

        if (
            isinstance(graph, ExecutionGraph)
            and graph.execution_id in self._consumed_execution_ids
        ):
            raise ValueError(
                f"execution ID has already been reduced: {graph.execution_id!r}"
            )
        by_id, required_ids, critical_segments = self._validate_inputs(
            record,
            graph,
            result,
            report,
            self.clock,
        )
        sampled_ids = self._sampled_request_ids(record)
        operation_index = {
            operation.operation_id: index
            for index, operation in enumerate(graph.operations)
        }
        graph_operation_by_id = {
            operation.operation_id: operation for operation in graph.operations
        }
        states = {
            request_id: replace(state)
            for request_id, state in self._requests.items()
        }
        metrics: list[RequestMetric] = []

        if self._request_arrivals is not None:
            for scheduled in record.scheduled:
                if scheduled.request_id in states:
                    continue
                arrived_at_ps = self._request_arrivals.get(scheduled.request_id)
                if arrived_at_ps is None:
                    raise ValueError(
                        f"scheduled request {scheduled.request_id!r} has no "
                        "framework-request bookkeeping origin"
                    )
                if arrived_at_ps > record.virtual_time_ps:
                    raise ValueError(
                        f"scheduled request {scheduled.request_id!r} predates its "
                        f"bookkeeping arrival at {arrived_at_ps} ps"
                    )

        for scheduled in record.scheduled:
            endpoint_candidates = [
                operation_id
                for operation_id in required_ids
                if scheduled.request_id
                in graph_operation_by_id[operation_id].correlation.request_ids
            ]
            if not endpoint_candidates:
                raise ValueError(
                    f"scheduled request {scheduled.request_id!r} has no required endpoint"
                )
            endpoint_id = max(
                endpoint_candidates,
                key=lambda operation_id: (
                    by_id[operation_id].completed_at_ps,
                    operation_index[operation_id],
                ),
            )
            _, graph_attribution = self._critical_chain(
                endpoint_id,
                by_id,
                critical_segments,
                graph.released_at_ps,
            )
            endpoint_at_ps = by_id[endpoint_id].completed_at_ps
            state = states.get(scheduled.request_id)
            if state is None:
                first_observed_at_ps = record.virtual_time_ps
                if self._request_arrivals is not None:
                    first_observed_at_ps = self._request_arrivals[
                        scheduled.request_id
                    ]
                state = _RequestMetricState(
                    first_observed_at_ps=first_observed_at_ps,
                    accounted_through_ps=first_observed_at_ps,
                )
                states[scheduled.request_id] = state
            if graph.released_at_ps < state.accounted_through_ps:
                raise ValueError(
                    f"request {scheduled.request_id!r} would move backward in time"
                )
            scheduler_gap_ps = graph.released_at_ps - state.accounted_through_ps
            interval_attribution = (
                state.pending_attribution
                + LatencyAttribution(queue_ps=scheduler_gap_ps)
                + graph_attribution
            )
            interval_additive = (
                state.pending_additive
                + self._request_additive_totals(scheduled.request_id, graph, report)
            )
            state.accounted_through_ps = endpoint_at_ps

            if scheduled.request_id not in sampled_ids:
                state.pending_attribution = interval_attribution
                state.pending_additive = interval_additive
                continue

            if state.first_token_at_ps is None:
                latency_ps = endpoint_at_ps - state.first_observed_at_ps
                state.first_token_at_ps = endpoint_at_ps
                ttft_ps = latency_ps
            else:
                assert state.last_token_at_ps is not None
                latency_ps = endpoint_at_ps - state.last_token_at_ps
                ttft_ps = state.first_token_at_ps - state.first_observed_at_ps
                state.inter_token_sum_ps += latency_ps
                state.inter_token_count += 1
            if interval_attribution.total_ps != latency_ps:
                raise ValueError(
                    f"request {scheduled.request_id!r} interval attribution does not "
                    "conserve elapsed time"
                )
            state.token_count += 1
            state.last_token_at_ps = endpoint_at_ps
            tpot_ps = (
                Fraction(state.inter_token_sum_ps, state.inter_token_count)
                if state.inter_token_count
                else None
            )
            metric = RequestMetric(
                request_id=scheduled.request_id,
                phase=scheduled.phase,
                token_index=state.token_count,
                completed_at_ps=endpoint_at_ps,
                latency_ps=latency_ps,
                ttft_ps=ttft_ps,
                tpot_ps=tpot_ps,
                attribution=interval_attribution,
                additive_visit_totals=interval_additive,
            )
            state.latest_metric = metric
            state.pending_attribution = LatencyAttribution()
            state.pending_additive = AdditiveVisitTotals()
            metrics.append(metric)

        step_result = StepResult(
            step_index=record.step_index,
            step_latency_ps=result.completed_at_ps - graph.released_at_ps,
            completed_at_ps=result.completed_at_ps,
            request_metrics=tuple(metrics),
            additive_visit_totals=_visit_totals(report.visits),
        )
        if self.lifetimes is not None:
            self.lifetimes.consume_step(record, graph, result)
        self.clock.advance_to(result.completed_at_ps)
        self._requests = states
        self._consumed_execution_ids.add(graph.execution_id)
        return step_result


__all__ = ["CompletionReducer"]
