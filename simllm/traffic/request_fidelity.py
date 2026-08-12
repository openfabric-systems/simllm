"""Loss-checked per-request partitions of rendered aggregate traffic."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from simllm.goal import GoalMessage

RequestPairRow = tuple[str, str, int, int, int]
AggregatePairRow = tuple[str, int, int, int]


class RequestFidelityError(ValueError):
    """A rendered request partition disagrees with its routed authority."""


@dataclass(frozen=True)
class RequestFidelityReport:
    """Raw aggregate and per-request comparison without outcome mutation."""

    aggregate_matches: bool
    aggregate_mismatch_count: int
    aggregate_l1_error_bytes: int
    per_request_matches: bool
    mismatch_count: int
    l1_error_bytes: int
    request_delta_bytes: tuple[tuple[str, int], ...]
    expected_request_rows: tuple[RequestPairRow, ...]
    observed_request_rows: tuple[RequestPairRow, ...]
    expected_aggregate_rows: tuple[AggregatePairRow, ...]
    observed_aggregate_rows: tuple[AggregatePairRow, ...]

    def require_match(self) -> RequestFidelityReport:
        """Return this report or fail closed with quantitative diagnostics."""

        if not self.aggregate_matches or not self.per_request_matches:
            raise RequestFidelityError(
                "rendered MoE request fidelity failed: "
                f"aggregate_mismatches={self.aggregate_mismatch_count}, "
                f"request_mismatches={self.mismatch_count}, "
                f"request_l1_error_bytes={self.l1_error_bytes}"
            )
        return self


def _request_map(rows: Sequence[RequestPairRow]) -> dict[tuple[str, str, int, int], int]:
    result: dict[tuple[str, str, int, int], int] = defaultdict(int)
    for operation_id, request_id, source, destination, size in rows:
        result[(operation_id, request_id, source, destination)] += size
    return dict(result)


def _aggregate_map(rows: Sequence[AggregatePairRow]) -> dict[tuple[str, int, int], int]:
    result: dict[tuple[str, int, int], int] = defaultdict(int)
    for operation_id, source, destination, size in rows:
        result[(operation_id, source, destination)] += size
    return dict(result)


def _request_rows(values: dict[tuple[str, str, int, int], int]) -> tuple[RequestPairRow, ...]:
    return tuple(
        (operation_id, request_id, source, destination, size)
        for (operation_id, request_id, source, destination), size in sorted(values.items())
    )


def _aggregate_rows(values: dict[tuple[str, int, int], int]) -> tuple[AggregatePairRow, ...]:
    return tuple(
        (operation_id, source, destination, size)
        for (operation_id, source, destination), size in sorted(values.items())
    )


def compare_goal_request_attribution(
    expected_request_rows: Sequence[RequestPairRow],
    expected_aggregate_rows: Sequence[AggregatePairRow],
    messages: Sequence[GoalMessage],
) -> RequestFidelityReport:
    """Compare raw rendered messages with one authoritative request partition."""

    expected_request = _request_map(expected_request_rows)
    expected_aggregate = _aggregate_map(expected_aggregate_rows)
    expected_operation_ids = {key[0] for key in expected_aggregate}

    observed_request_values: dict[tuple[str, str, int, int], int] = defaultdict(int)
    observed_aggregate_values: dict[tuple[str, int, int], int] = defaultdict(int)
    for message in messages:
        if not isinstance(message, GoalMessage):
            raise TypeError("messages: expected GoalMessage entries")
        attributed = bool(message.request_payload_bytes)
        if message.operation_id in expected_operation_ids or attributed:
            operation_id = message.operation_id
            if operation_id is None:
                raise RequestFidelityError(
                    "request-attributed GOAL message has no operation identity"
                )
            observed_aggregate_values[
                (operation_id, message.source_rank, message.destination_rank)
            ] += message.payload_bytes
            for request_id, size in message.request_payload_bytes:
                observed_request_values[
                    (
                        operation_id,
                        request_id,
                        message.source_rank,
                        message.destination_rank,
                    )
                ] += size

    observed_request = dict(observed_request_values)
    observed_aggregate = dict(observed_aggregate_values)
    request_keys = set(expected_request) | set(observed_request)
    aggregate_keys = set(expected_aggregate) | set(observed_aggregate)
    request_mismatches = {
        key for key in request_keys if expected_request.get(key, 0) != observed_request.get(key, 0)
    }
    aggregate_mismatches = {
        key
        for key in aggregate_keys
        if expected_aggregate.get(key, 0) != observed_aggregate.get(key, 0)
    }
    request_ids = sorted({key[1] for key in request_keys})
    request_deltas = tuple(
        (
            request_id,
            sum(
                observed_request.get(key, 0) - expected_request.get(key, 0)
                for key in request_keys
                if key[1] == request_id
            ),
        )
        for request_id in request_ids
    )
    return RequestFidelityReport(
        aggregate_matches=not aggregate_mismatches,
        aggregate_mismatch_count=len(aggregate_mismatches),
        aggregate_l1_error_bytes=sum(
            abs(expected_aggregate.get(key, 0) - observed_aggregate.get(key, 0))
            for key in aggregate_keys
        ),
        per_request_matches=not request_mismatches,
        mismatch_count=len(request_mismatches),
        l1_error_bytes=sum(
            abs(expected_request.get(key, 0) - observed_request.get(key, 0)) for key in request_keys
        ),
        request_delta_bytes=request_deltas,
        expected_request_rows=_request_rows(expected_request),
        observed_request_rows=_request_rows(observed_request),
        expected_aggregate_rows=_aggregate_rows(expected_aggregate),
        observed_aggregate_rows=_aggregate_rows(observed_aggregate),
    )
