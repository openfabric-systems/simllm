"""Canonical unbound instance and normalized template graph identities."""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

from simllm.core.execution import (
    CollectivePlan,
    CollectiveWork,
    ComputeWork,
    ControlWork,
    DmaWork,
    ExecutionGraph,
    ExecutionOperation,
    KvCacheWork,
)
from simllm.core.execution_io import execution_graph_to_json, validate_execution_graph

from .record_types import RecordObject

EXECUTION_GRAPH_TEMPLATE_SCHEMA = "simllm-execution-graph-template-v1"

_DEVICE_ENDPOINT = re.compile(r"(gpu|cuda):(0|[1-9][0-9]*)(:hbm)?")
_HOST_ENDPOINTS = frozenset({"host", "host:pinned", "host:pageable"})


class GraphIdentityError(ValueError):
    """An execution graph cannot be represented by the frozen identities."""


def unbound_execution_graph_record(graph: ExecutionGraph) -> RecordObject:
    """Return exact graph-v1 bytes with service fields explicitly unbound."""

    validate_execution_graph(graph)
    operations = []
    for operation_index, operation in enumerate(graph.operations):
        work = operation.work
        if isinstance(work, ComputeWork):
            for config_index, (_, value) in enumerate(work.config):
                if type(value) is float:
                    raise GraphIdentityError(
                        "graph.operations"
                        f"[{operation_index}].work.config[{config_index}]: "
                        "calibration graph config cannot contain a float"
                    )
            work = replace(
                work,
                nominal_duration_ps=None,
                uncertainty_fraction=None,
            )
            operation = replace(operation, work=work)
        operations.append(operation)
    unbound = replace(graph, operations=tuple(operations))
    return RecordObject.from_value(execution_graph_to_json(unbound))


def _endpoint_rank(endpoint: str, path: str) -> int | None:
    if endpoint in _HOST_ENDPOINTS:
        return None
    match = _DEVICE_ENDPOINT.fullmatch(endpoint)
    if match is None:
        raise GraphIdentityError(
            f"{path}: expected a simllm-device-endpoint-role-v1 value"
        )
    if match.group(1) == "cuda" and match.group(3) is not None:
        raise GraphIdentityError(f"{path}: cuda endpoints cannot carry an HBM suffix")
    return int(match.group(2))


def _rank_set(graph: ExecutionGraph) -> tuple[int, ...]:
    ranks: set[int] = set()
    for operation_index, operation in enumerate(graph.operations):
        ranks.add(operation.rank)
        work = operation.work
        if isinstance(work, CollectiveWork):
            ranks.update(work.ranks)
            for source, destination, _ in work.pair_payload_bytes:
                ranks.update((source, destination))
        elif isinstance(work, ControlWork):
            ranks.update(work.destination_ranks)
        elif isinstance(work, DmaWork):
            for field_name in ("source", "destination"):
                rank = _endpoint_rank(
                    getattr(work, field_name),
                    f"graph.operations[{operation_index}].work.{field_name}",
                )
                if rank is not None:
                    ranks.add(rank)
    for plan in graph.collective_plans:
        ranks.update(plan.rank_order)
        ranks.update(action.rank for action in plan.actions)
        for extent in plan.extents:
            ranks.update((extent.source_rank, extent.destination_rank))
        ranks.update(rank for rank, _ in plan.entry_action_ids)
        ranks.update(rank for rank, _ in plan.terminal_action_ids)
    return tuple(sorted(ranks))


def _endpoint_role(endpoint: str, ranks: dict[int, int], path: str) -> str:
    rank = _endpoint_rank(endpoint, path)
    if rank is None:
        return endpoint
    try:
        ordinal = ranks[rank]
    except KeyError as error:  # pragma: no cover - rank collection is total
        raise GraphIdentityError(f"{path}: unresolved endpoint rank {rank}") from error
    if endpoint.startswith("cuda:"):
        return f"cuda:{ordinal}"
    suffix = ":hbm" if endpoint.endswith(":hbm") else ""
    return f"gpu:{ordinal}{suffix}"


def _ordinal_tuple(
    values: tuple[str, ...],
    identities: dict[str, int],
    path: str,
) -> list[int]:
    try:
        return sorted({identities[value] for value in values})
    except KeyError as error:
        raise GraphIdentityError(f"{path}: unresolved identity {error.args[0]!r}") from error


def _semantic_channels(graph: ExecutionGraph) -> dict[str, int]:
    channels: dict[str, int] = {}
    for operation in graph.operations:
        if isinstance(operation.work, CollectiveWork):
            channel = operation.work.channel_hint or "default"
            channels.setdefault(channel, len(channels))
    return channels


def _work_template(
    work: object,
    *,
    ranks: dict[int, int],
    channels: dict[str, int],
    path: str,
) -> dict[str, Any]:
    if isinstance(work, ComputeWork):
        return {"kind": "compute", "kernel": work.kernel}
    if isinstance(work, KvCacheWork):
        return {"kind": "kv-cache", "action": work.action.value}
    if isinstance(work, DmaWork):
        return {
            "kind": "dma",
            "source_role": _endpoint_role(work.source, ranks, f"{path}.source"),
            "destination_role": _endpoint_role(
                work.destination, ranks, f"{path}.destination"
            ),
        }
    if isinstance(work, CollectiveWork):
        channel = work.channel_hint or "default"
        try:
            channel_ordinal = channels[channel]
        except KeyError as error:  # pragma: no cover - namespace is precomputed
            raise GraphIdentityError(f"{path}: unresolved channel {channel!r}") from error
        return {
            "kind": "collective",
            "collective": work.collective,
            "algorithm_hint": work.algorithm_hint,
            "rank_ordinals": [ranks[rank] for rank in work.ranks],
            "channel_ordinal": channel_ordinal,
            "pair_rank_ordinals": [
                [ranks[source], ranks[destination]]
                for source, destination, _ in work.pair_payload_bytes
            ],
        }
    if isinstance(work, ControlWork):
        return {
            "kind": "control",
            "mode": work.mode.value,
            "message": work.message,
            "destination_rank_ordinals": [
                ranks[rank] for rank in work.destination_ranks
            ],
        }
    raise GraphIdentityError(f"{path}: unsupported work type {type(work).__name__}")


def _queue_ordinals(
    operations: tuple[ExecutionOperation, ...],
    ranks: dict[int, int],
) -> dict[tuple[int, str], int]:
    queues: dict[tuple[int, str], int] = {}
    counts: dict[int, int] = {}
    for operation in operations:
        rank = ranks[operation.rank]
        key = (rank, operation.logical_queue)
        if key not in queues:
            queues[key] = counts.get(rank, 0)
            counts[rank] = queues[key] + 1
    return queues


def _frontier_template(
    plan: CollectivePlan,
    values: tuple[tuple[int, tuple[str, ...]], ...],
    *,
    ranks: dict[int, int],
    actions: dict[str, int],
    path: str,
) -> list[dict[str, Any]]:
    by_rank = dict(values)
    if set(by_rank) != set(plan.rank_order):
        raise GraphIdentityError(f"{path}: frontier ranks do not match plan rank order")
    return [
        {
            "rank_ordinal": ranks[rank],
            "action_ordinals": _ordinal_tuple(
                by_rank[rank], actions, f"{path}[rank={rank}]"
            ),
        }
        for rank in plan.rank_order
    ]


def _plan_template(
    plan: CollectivePlan,
    *,
    operations: dict[str, int],
    ranks: dict[int, int],
    channels: dict[str, int],
    transfer_channels: dict[str, int],
    path: str,
) -> dict[str, Any]:
    try:
        operation_ordinal = operations[plan.operation_id]
        channel_ordinal = channels[plan.channel_id]
    except KeyError as error:
        raise GraphIdentityError(f"{path}: unresolved plan identity {error.args[0]!r}") from error
    action_ordinals = {action.action_id: index for index, action in enumerate(plan.actions)}
    extent_ordinals = {extent.extent_id: index for index, extent in enumerate(plan.extents)}
    round_ordinals = {
        round_.round_index: index for index, round_ in enumerate(plan.rounds)
    }
    if len(action_ordinals) != len(plan.actions) or len(extent_ordinals) != len(
        plan.extents
    ):
        raise GraphIdentityError(f"{path}: duplicate action or extent identity")

    rounds = []
    for round_ in plan.rounds:
        transfer_channels.setdefault(round_.channel_id, len(transfer_channels))
        rounds.append(
            {"transfer_channel_ordinal": transfer_channels[round_.channel_id]}
        )
    actions = []
    for action_index, action in enumerate(plan.actions):
        try:
            extent_ordinal = extent_ordinals[action.extent_id]
        except KeyError as error:
            raise GraphIdentityError(
                f"{path}.actions[{action_index}]: unresolved extent {action.extent_id!r}"
            ) from error
        actions.append(
            {
                "rank_ordinal": ranks[action.rank],
                "kind": action.kind.value,
                "extent_ordinal": extent_ordinal,
                "depends_on_action_ordinals": _ordinal_tuple(
                    action.depends_on,
                    action_ordinals,
                    f"{path}.actions[{action_index}].depends_on",
                ),
            }
        )
    extents = []
    for extent_index, extent in enumerate(plan.extents):
        try:
            projected = {
                "round_ordinal": round_ordinals[extent.round_index],
                "source_rank_ordinal": ranks[extent.source_rank],
                "destination_rank_ordinal": ranks[extent.destination_rank],
                "send_action_ordinal": action_ordinals[extent.send_action_id],
                "receive_action_ordinal": action_ordinals[extent.receive_action_id],
            }
        except KeyError as error:
            raise GraphIdentityError(
                f"{path}.extents[{extent_index}]: unresolved reference {error.args[0]!r}"
            ) from error
        extents.append(projected)
    return {
        "operation_ordinal": operation_ordinal,
        "algorithm": plan.algorithm,
        "channel_ordinal": channel_ordinal,
        "rank_order": [ranks[rank] for rank in plan.rank_order],
        "rounds": rounds,
        "actions": actions,
        "extents": extents,
        "entry_action_ordinals": _frontier_template(
            plan,
            plan.entry_action_ids,
            ranks=ranks,
            actions=action_ordinals,
            path=f"{path}.entry_action_ids",
        ),
        "terminal_action_ordinals": _frontier_template(
            plan,
            plan.terminal_action_ids,
            ranks=ranks,
            actions=action_ordinals,
            path=f"{path}.terminal_action_ids",
        ),
    }


def execution_graph_template_record(graph: ExecutionGraph) -> RecordObject:
    """Project one graph into the frozen identity-only template schema."""

    validate_execution_graph(graph)
    operation_ordinals = {
        operation.operation_id: index
        for index, operation in enumerate(graph.operations)
    }
    rank_ordinals = {
        rank: index for index, rank in enumerate(_rank_set(graph))
    }
    queue_ordinals = _queue_ordinals(graph.operations, rank_ordinals)
    channels = _semantic_channels(graph)
    operations = []
    for index, operation in enumerate(graph.operations):
        rank_ordinal = rank_ordinals[operation.rank]
        operations.append(
            {
                "rank_ordinal": rank_ordinal,
                "logical_queue_ordinal": queue_ordinals[
                    (rank_ordinal, operation.logical_queue)
                ],
                "priority": operation.priority,
                "work": _work_template(
                    operation.work,
                    ranks=rank_ordinals,
                    channels=channels,
                    path=f"graph.operations[{index}].work",
                ),
                "depends_on_operation_ordinals": _ordinal_tuple(
                    operation.depends_on,
                    operation_ordinals,
                    f"graph.operations[{index}].depends_on",
                ),
                "participant_local_depends_on_operation_ordinals": _ordinal_tuple(
                    operation.participant_local_depends_on,
                    operation_ordinals,
                    f"graph.operations[{index}].participant_local_depends_on",
                ),
            }
        )
    if graph.completion_operation_ids:
        completion = _ordinal_tuple(
            graph.completion_operation_ids,
            operation_ordinals,
            "graph.completion_operation_ids",
        )
    else:
        completion = list(range(len(graph.operations)))
    transfer_channels: dict[str, int] = {}
    plans = [
        _plan_template(
            plan,
            operations=operation_ordinals,
            ranks=rank_ordinals,
            channels=channels,
            transfer_channels=transfer_channels,
            path=f"graph.collective_plans[{index}]",
        )
        for index, plan in enumerate(graph.collective_plans)
    ]
    return RecordObject.from_value(
        {
            "schema": EXECUTION_GRAPH_TEMPLATE_SCHEMA,
            "operations": operations,
            "completion_operation_ordinals": completion,
            "collective_plans": plans,
        }
    )


__all__ = [
    "EXECUTION_GRAPH_TEMPLATE_SCHEMA",
    "GraphIdentityError",
    "execution_graph_template_record",
    "unbound_execution_graph_record",
]
