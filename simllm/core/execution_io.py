"""Validation and JSON wire forms for execution and completion contracts."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict, deque
from typing import Any

from simllm.core._wire import (
    _array,
    _correlation_from_json,
    _correlation_to_json,
    _enum_value,
    _fail,
    _fields,
    _int_tuple,
    _integer,
    _number,
    _object,
    _optional_integer,
    _optional_number,
    _optional_string,
    _require_tuple,
    _string,
    _string_tuple,
    _validate_correlation,
    _validate_unique,
)
from simllm.core.execution import (
    COLLECTIVE_PLAN_SCHEMA,
    COMPLETION_EVENT_SCHEMA,
    EXECUTION_GRAPH_SCHEMA,
    EXECUTION_RESULT_SCHEMA,
    CollectivePlan,
    CollectivePlanAction,
    CollectivePlanActionKind,
    CollectivePlanExtent,
    CollectivePlanRound,
    CollectiveWork,
    CompletionEvent,
    ComputeWork,
    ControlMode,
    ControlWork,
    DependencyOrigin,
    DependencyScope,
    DmaWork,
    EffectiveDependencyEdge,
    EventPhase,
    ExecutionGraph,
    ExecutionObservations,
    ExecutionOperation,
    ExecutionResult,
    KvCacheAction,
    KvCacheWork,
    ResourceKind,
    ResourceRef,
    WorkPayload,
)
from simllm.core.step import StepRecord

_DEVICE_PARTICIPANT_RE = re.compile(
    r"^(?:gpu|cuda|rank):(\d+)(?::|$)",
    re.IGNORECASE,
)


def operation_participant_ranks(operation: ExecutionOperation) -> tuple[int, ...]:
    """Return the canonical participant ranks of one graph operation."""

    if not isinstance(operation, ExecutionOperation):
        _fail("operation", "expected ExecutionOperation")
    ranks = {operation.rank}
    work = operation.work
    if isinstance(work, CollectiveWork):
        ranks.update(work.ranks)
    elif isinstance(work, ControlWork):
        ranks.update(work.destination_ranks)
    elif isinstance(work, DmaWork):
        for endpoint in (work.source, work.destination):
            match = _DEVICE_PARTICIPANT_RE.match(endpoint.strip())
            if match is not None:
                ranks.add(int(match.group(1)))
    return tuple(sorted(ranks))


def _effective_dependency_edges_unchecked(
    graph: ExecutionGraph,
) -> tuple[EffectiveDependencyEdge, ...]:
    operation_by_id = {
        operation.operation_id: operation for operation in graph.operations
    }
    queue_tail: dict[tuple[int, str], str] = {}
    result: list[EffectiveDependencyEdge] = []
    for operation in graph.operations:
        for predecessor_id in operation.depends_on:
            result.append(
                EffectiveDependencyEdge(
                    predecessor_id=predecessor_id,
                    operation_id=operation.operation_id,
                    scope=DependencyScope.WHOLE_OPERATION,
                    origin=DependencyOrigin.EXPLICIT,
                )
            )
        for predecessor_id in operation.participant_local_depends_on:
            predecessor = operation_by_id.get(predecessor_id)
            if predecessor is None:
                continue
            shared_ranks = sorted(
                set(operation_participant_ranks(operation))
                & set(operation_participant_ranks(predecessor))
            )
            result.extend(
                EffectiveDependencyEdge(
                    predecessor_id=predecessor_id,
                    operation_id=operation.operation_id,
                    scope=DependencyScope.PARTICIPANT_LOCAL,
                    origin=DependencyOrigin.EXPLICIT,
                    participant_rank=rank,
                )
                for rank in shared_ranks
            )
        queue = (operation.rank, operation.logical_queue)
        previous = queue_tail.get(queue)
        if previous is not None:
            result.append(
                EffectiveDependencyEdge(
                    predecessor_id=previous,
                    operation_id=operation.operation_id,
                    scope=DependencyScope.WHOLE_OPERATION,
                    origin=DependencyOrigin.LOGICAL_QUEUE_FIFO,
                )
            )
        queue_tail[queue] = operation.operation_id
    return tuple(result)


def _validate_config(config: tuple[tuple[str, Any], ...], path: str) -> None:
    _require_tuple(config, path)
    names: list[str] = []
    for index, entry in enumerate(config):
        entry_path = f"{path}[{index}]"
        if not isinstance(entry, tuple) or len(entry) != 2:
            _fail(entry_path, "expected a two-item tuple")
        name = _string(entry[0], f"{entry_path}[0]")
        scalar = entry[1]
        if isinstance(scalar, bool) or type(scalar) is int:
            pass
        elif type(scalar) is float:
            if not math.isfinite(scalar):
                _fail(f"{entry_path}[1]", "float must be finite")
        elif isinstance(scalar, str):
            pass
        else:
            _fail(f"{entry_path}[1]", "expected a JSON scalar")
        names.append(name)
    if len(names) != len(set(names)):
        _fail(path, "configuration keys must be unique")


def _validate_work(work: WorkPayload, path: str) -> None:
    if isinstance(work, ComputeWork):
        _string(work.kernel, f"{path}.kernel")
        _validate_config(work.config, f"{path}.config")
        _integer(work.flops, f"{path}.flops", nonnegative=True)
        _integer(work.hbm_bytes, f"{path}.hbm_bytes", nonnegative=True)
        if work.nominal_duration_ps is not None:
            _integer(
                work.nominal_duration_ps,
                f"{path}.nominal_duration_ps",
                nonnegative=True,
            )
        if work.uncertainty_fraction is not None:
            _number(
                work.uncertainty_fraction,
                f"{path}.uncertainty_fraction",
                nonnegative=True,
            )
        return
    if isinstance(work, KvCacheWork):
        if not isinstance(work.action, KvCacheAction):
            _fail(f"{path}.action", "expected KvCacheAction")
        _string(work.pool_id, f"{path}.pool_id")
        for field_name in ("request_id", "dtype", "cause", "correlation_id"):
            value = getattr(work, field_name)
            if value is not None:
                _string(value, f"{path}.{field_name}")
        block_ids = _require_tuple(work.block_ids, f"{path}.block_ids")
        for index, block_id in enumerate(block_ids):
            _string(block_id, f"{path}.block_ids[{index}]")
        _validate_unique(block_ids, f"{path}.block_ids")
        if (work.token_start is None) != (work.token_end is None):
            _fail(path, "token_start and token_end must appear together")
        if work.token_start is not None and work.token_end is not None:
            _integer(work.token_start, f"{path}.token_start", nonnegative=True)
            _integer(work.token_end, f"{path}.token_end", nonnegative=True)
            if work.token_end < work.token_start:
                _fail(path, "token_end must be greater than or equal to token_start")
        for field_name in ("layer", "reference_count"):
            value = getattr(work, field_name)
            if value is not None:
                _integer(value, f"{path}.{field_name}", nonnegative=True)
        _integer(work.byte_count, f"{path}.byte_count", nonnegative=True)
        _integer(work.placement_epoch, f"{path}.placement_epoch", nonnegative=True)
        return
    if isinstance(work, DmaWork):
        _string(work.descriptor_id, f"{path}.descriptor_id")
        _string(work.source, f"{path}.source")
        _string(work.destination, f"{path}.destination")
        _integer(work.byte_count, f"{path}.byte_count", nonnegative=True)
        return
    if isinstance(work, CollectiveWork):
        _string(work.collective, f"{path}.collective")
        ranks = _require_tuple(work.ranks, f"{path}.ranks")
        if not ranks:
            _fail(f"{path}.ranks", "must not be empty")
        for index, rank in enumerate(ranks):
            _integer(rank, f"{path}.ranks[{index}]", nonnegative=True)
        _validate_unique(ranks, f"{path}.ranks")
        _integer(work.payload_bytes, f"{path}.payload_bytes", nonnegative=True)
        pair_payloads = _require_tuple(
            work.pair_payload_bytes,
            f"{path}.pair_payload_bytes",
        )
        pair_keys: list[tuple[int, int]] = []
        for index, entry in enumerate(pair_payloads):
            entry_path = f"{path}.pair_payload_bytes[{index}]"
            if not isinstance(entry, tuple) or len(entry) != 3:
                _fail(entry_path, "expected a three-item tuple")
            source_rank = _integer(entry[0], f"{entry_path}[0]", nonnegative=True)
            destination_rank = _integer(
                entry[1],
                f"{entry_path}[1]",
                nonnegative=True,
            )
            _integer(entry[2], f"{entry_path}[2]", minimum=1)
            if source_rank == destination_rank:
                _fail(entry_path, "source and destination ranks must differ")
            if source_rank not in ranks or destination_rank not in ranks:
                _fail(entry_path, "source and destination must belong to ranks")
            pair_keys.append((source_rank, destination_rank))
        if len(pair_keys) != len(set(pair_keys)):
            _fail(f"{path}.pair_payload_bytes", "ordered rank pairs must be unique")
        if pair_keys != sorted(pair_keys):
            _fail(
                f"{path}.pair_payload_bytes",
                "entries must be in source-major order",
            )
        if pair_payloads:
            if (work.collective, work.algorithm_hint) != (
                "all-to-allv",
                "pairwise",
            ):
                _fail(
                    f"{path}.pair_payload_bytes",
                    "table is valid only for pairwise all-to-allv",
                )
            if work.payload_bytes != 0:
                _fail(
                    path,
                    "pair_payload_bytes and payload_bytes cannot both be authoritative",
                )
        request_pair_payloads = _require_tuple(
            work.request_pair_payload_bytes,
            f"{path}.request_pair_payload_bytes",
        )
        request_pair_keys: list[tuple[str, int, int]] = []
        attributed_pairs: dict[tuple[int, int], int] = defaultdict(int)
        for index, entry in enumerate(request_pair_payloads):
            entry_path = f"{path}.request_pair_payload_bytes[{index}]"
            if not isinstance(entry, tuple) or len(entry) != 4:
                _fail(entry_path, "expected a four-item tuple")
            request_id = _string(entry[0], f"{entry_path}[0]")
            source_rank = _integer(entry[1], f"{entry_path}[1]", nonnegative=True)
            destination_rank = _integer(
                entry[2],
                f"{entry_path}[2]",
                nonnegative=True,
            )
            size = _integer(entry[3], f"{entry_path}[3]", minimum=1)
            if source_rank == destination_rank:
                _fail(entry_path, "source and destination ranks must differ")
            if source_rank not in ranks or destination_rank not in ranks:
                _fail(entry_path, "source and destination must belong to ranks")
            request_pair_keys.append((request_id, source_rank, destination_rank))
            attributed_pairs[(source_rank, destination_rank)] += size
        if len(request_pair_keys) != len(set(request_pair_keys)):
            _fail(
                f"{path}.request_pair_payload_bytes",
                "request and ordered rank pairs must be unique",
            )
        if request_pair_keys != sorted(request_pair_keys):
            _fail(
                f"{path}.request_pair_payload_bytes",
                "entries must be in request-major order",
            )
        if request_pair_payloads:
            if (work.collective, work.algorithm_hint) != (
                "all-to-allv",
                "pairwise",
            ):
                _fail(
                    f"{path}.request_pair_payload_bytes",
                    "partition is valid only for pairwise all-to-allv",
                )
            expected_pairs = {
                (source, destination): size for source, destination, size in pair_payloads
            }
            if dict(attributed_pairs) != expected_pairs:
                _fail(
                    f"{path}.request_pair_payload_bytes",
                    "per-pair sums must equal pair_payload_bytes exactly",
                )
        for field_name in ("algorithm_hint", "channel_hint"):
            value = getattr(work, field_name)
            if value is not None:
                _string(value, f"{path}.{field_name}")
        return
    if isinstance(work, ControlWork):
        _string(work.message, f"{path}.message")
        ranks = _require_tuple(work.destination_ranks, f"{path}.destination_ranks")
        for index, rank in enumerate(ranks):
            _integer(rank, f"{path}.destination_ranks[{index}]", nonnegative=True)
        _validate_unique(ranks, f"{path}.destination_ranks")
        _integer(work.payload_bytes, f"{path}.payload_bytes", nonnegative=True)
        if not isinstance(work.mode, ControlMode):
            _fail(f"{path}.mode", "expected ControlMode")
        return
    _fail(path, f"unsupported work payload {type(work).__name__}")


def _collective_plan_identity_to_json(plan: CollectivePlan) -> dict[str, Any]:
    return {
        "schema": COLLECTIVE_PLAN_SCHEMA,
        "operation_id": plan.operation_id,
        "collective": plan.collective,
        "algorithm": plan.algorithm,
        "channel_id": plan.channel_id,
        "rank_order": list(plan.rank_order),
        "payload_bytes": plan.payload_bytes,
        "pair_payload_bytes": [list(entry) for entry in plan.pair_payload_bytes],
        "request_pair_payload_bytes": [
            list(entry) for entry in plan.request_pair_payload_bytes
        ],
        "rounds": [
            {
                "round_index": round_.round_index,
                "tag": round_.tag,
                "channel_id": round_.channel_id,
            }
            for round_ in plan.rounds
        ],
        "actions": [
            {
                "action_id": action.action_id,
                "rank": action.rank,
                "kind": action.kind.value,
                "extent_id": action.extent_id,
                "depends_on": list(action.depends_on),
            }
            for action in plan.actions
        ],
        "extents": [
            {
                "extent_id": extent.extent_id,
                "round_index": extent.round_index,
                "source_rank": extent.source_rank,
                "destination_rank": extent.destination_rank,
                "payload_bytes": extent.payload_bytes,
                "send_action_id": extent.send_action_id,
                "receive_action_id": extent.receive_action_id,
                "request_payload_bytes": [
                    list(entry) for entry in extent.request_payload_bytes
                ],
            }
            for extent in plan.extents
        ],
        "entry_action_ids": [
            [rank, list(action_ids)] for rank, action_ids in plan.entry_action_ids
        ],
        "terminal_action_ids": [
            [rank, list(action_ids)] for rank, action_ids in plan.terminal_action_ids
        ],
    }


def collective_plan_integrity_sha256(plan: CollectivePlan) -> str:
    """Return the canonical integrity identity of one plan's immutable content."""

    if not isinstance(plan, CollectivePlan):
        raise TypeError("plan must be a CollectivePlan")
    wire = json.dumps(
        _collective_plan_identity_to_json(plan),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(wire).hexdigest()


def _validate_rank_action_ids(
    entries: tuple[tuple[int, tuple[str, ...]], ...],
    path: str,
    *,
    rank_order: tuple[int, ...],
    action_by_id: dict[str, CollectivePlanAction],
) -> dict[int, tuple[str, ...]]:
    values = _require_tuple(entries, path)
    if len(values) != len(rank_order):
        _fail(path, "must contain one entry for every rank in rank_order")
    result: dict[int, tuple[str, ...]] = {}
    for index, entry in enumerate(values):
        entry_path = f"{path}[{index}]"
        if not isinstance(entry, tuple) or len(entry) != 2:
            _fail(entry_path, "expected a two-item tuple")
        rank = _integer(entry[0], f"{entry_path}[0]", nonnegative=True)
        if rank != rank_order[index]:
            _fail(entry_path, "rank entries must preserve rank_order exactly")
        action_ids = _require_tuple(entry[1], f"{entry_path}[1]")
        for action_index, action_id in enumerate(action_ids):
            action_id = _string(action_id, f"{entry_path}[1][{action_index}]")
            action = action_by_id.get(action_id)
            if action is None:
                _fail(f"{entry_path}[1][{action_index}]", "unknown action ID")
            if action.rank != rank:
                _fail(f"{entry_path}[1][{action_index}]", "action belongs to another rank")
        _validate_unique(action_ids, f"{entry_path}[1]")
        result[rank] = action_ids
    return result


def _validate_collective_plan(
    plan: CollectivePlan,
    path: str,
    *,
    work: CollectiveWork,
) -> None:
    if not isinstance(plan, CollectivePlan):
        _fail(path, "expected CollectivePlan")
    _string(plan.operation_id, f"{path}.operation_id")
    _string(plan.collective, f"{path}.collective")
    _string(plan.algorithm, f"{path}.algorithm")
    _string(plan.channel_id, f"{path}.channel_id")
    rank_order = _require_tuple(plan.rank_order, f"{path}.rank_order")
    if not rank_order:
        _fail(f"{path}.rank_order", "must not be empty")
    for index, rank in enumerate(rank_order):
        _integer(rank, f"{path}.rank_order[{index}]", nonnegative=True)
    _validate_unique(rank_order, f"{path}.rank_order")
    _integer(plan.payload_bytes, f"{path}.payload_bytes", nonnegative=True)

    if plan.collective != work.collective:
        _fail(path, "collective disagrees with semantic work")
    if plan.algorithm != work.algorithm_hint:
        _fail(path, "algorithm disagrees with semantic work")
    if plan.channel_id != (work.channel_hint or "default"):
        _fail(path, "channel disagrees with semantic work")
    if plan.rank_order != work.ranks:
        _fail(path, "rank order disagrees with semantic work")
    if plan.payload_bytes != work.payload_bytes:
        _fail(path, "payload_bytes disagrees with semantic work")
    if plan.pair_payload_bytes != work.pair_payload_bytes:
        _fail(path, "pair_payload_bytes disagrees with semantic work")
    if plan.request_pair_payload_bytes != work.request_pair_payload_bytes:
        _fail(path, "request_pair_payload_bytes disagrees with semantic work")

    semantic_copy = CollectiveWork(
        collective=plan.collective,
        ranks=plan.rank_order,
        payload_bytes=plan.payload_bytes,
        algorithm_hint=plan.algorithm,
        channel_hint=work.channel_hint,
        pair_payload_bytes=plan.pair_payload_bytes,
        request_pair_payload_bytes=plan.request_pair_payload_bytes,
    )
    _validate_work(semantic_copy, path)

    rounds = _require_tuple(plan.rounds, f"{path}.rounds")
    if not rounds:
        _fail(f"{path}.rounds", "must not be empty")
    round_tags = []
    for index, round_ in enumerate(rounds):
        round_path = f"{path}.rounds[{index}]"
        if not isinstance(round_, CollectivePlanRound):
            _fail(round_path, "expected CollectivePlanRound")
        if round_.round_index != index:
            _fail(f"{round_path}.round_index", "round indices must be contiguous")
        _integer(round_.tag, f"{round_path}.tag", nonnegative=True)
        _string(round_.channel_id, f"{round_path}.channel_id")
        round_tags.append(round_.tag)
    _validate_unique(round_tags, f"{path}.rounds tags")

    actions = _require_tuple(plan.actions, f"{path}.actions")
    action_by_id: dict[str, CollectivePlanAction] = {}
    action_index_by_id: dict[str, int] = {}
    for index, action in enumerate(actions):
        action_path = f"{path}.actions[{index}]"
        if not isinstance(action, CollectivePlanAction):
            _fail(action_path, "expected CollectivePlanAction")
        action_id = _string(action.action_id, f"{action_path}.action_id")
        if action_id in action_by_id:
            _fail(f"{action_path}.action_id", "duplicate action ID")
        action_by_id[action_id] = action
        action_index_by_id[action_id] = index
        _integer(action.rank, f"{action_path}.rank", nonnegative=True)
        if action.rank not in rank_order:
            _fail(f"{action_path}.rank", "action rank is outside rank_order")
        if not isinstance(action.kind, CollectivePlanActionKind):
            _fail(f"{action_path}.kind", "expected CollectivePlanActionKind")
        _string(action.extent_id, f"{action_path}.extent_id")
        dependencies = _require_tuple(action.depends_on, f"{action_path}.depends_on")
        for dep_index, dependency in enumerate(dependencies):
            _string(dependency, f"{action_path}.depends_on[{dep_index}]")
        _validate_unique(dependencies, f"{action_path}.depends_on")

    for index, action in enumerate(actions):
        action_path = f"{path}.actions[{index}]"
        for dep_index, dependency in enumerate(action.depends_on):
            predecessor_index = action_index_by_id.get(dependency)
            if predecessor_index is None:
                _fail(f"{action_path}.depends_on[{dep_index}]", "unknown action ID")
            if predecessor_index >= index:
                _fail(
                    f"{action_path}.depends_on[{dep_index}]",
                    "internal dependencies must point to an earlier action",
                )
            if action_by_id[dependency].rank != action.rank:
                _fail(
                    f"{action_path}.depends_on[{dep_index}]",
                    "internal dependency must stay on one rank",
                )

    extents = _require_tuple(plan.extents, f"{path}.extents")
    extent_by_id: dict[str, CollectivePlanExtent] = {}
    extent_action_ids: set[str] = set()
    previous_round = -1
    for index, extent in enumerate(extents):
        extent_path = f"{path}.extents[{index}]"
        if not isinstance(extent, CollectivePlanExtent):
            _fail(extent_path, "expected CollectivePlanExtent")
        extent_id = _string(extent.extent_id, f"{extent_path}.extent_id")
        if extent_id in extent_by_id:
            _fail(f"{extent_path}.extent_id", "duplicate extent ID")
        extent_by_id[extent_id] = extent
        _integer(extent.round_index, f"{extent_path}.round_index", nonnegative=True)
        if extent.round_index >= len(rounds):
            _fail(f"{extent_path}.round_index", "round index is outside the plan")
        if extent.round_index < previous_round:
            _fail(f"{extent_path}.round_index", "extents must be round-major")
        previous_round = extent.round_index
        _integer(extent.source_rank, f"{extent_path}.source_rank", nonnegative=True)
        _integer(
            extent.destination_rank,
            f"{extent_path}.destination_rank",
            nonnegative=True,
        )
        if extent.source_rank == extent.destination_rank:
            _fail(extent_path, "extent source and destination must differ")
        if (
            extent.source_rank not in rank_order
            or extent.destination_rank not in rank_order
        ):
            _fail(extent_path, "extent endpoint is outside rank_order")
        _integer(extent.payload_bytes, f"{extent_path}.payload_bytes", minimum=1)
        for field_name, action_id, kind, rank in (
            (
                "send_action_id",
                extent.send_action_id,
                CollectivePlanActionKind.SEND,
                extent.source_rank,
            ),
            (
                "receive_action_id",
                extent.receive_action_id,
                CollectivePlanActionKind.RECEIVE,
                extent.destination_rank,
            ),
        ):
            action_id = _string(action_id, f"{extent_path}.{field_name}")
            action = action_by_id.get(action_id)
            if action is None:
                _fail(f"{extent_path}.{field_name}", "unknown action ID")
            if action.kind is not kind or action.rank != rank:
                _fail(f"{extent_path}.{field_name}", "action kind or rank disagrees")
            if action.extent_id != extent_id:
                _fail(f"{extent_path}.{field_name}", "action extent identity disagrees")
            if action_id in extent_action_ids:
                _fail(f"{extent_path}.{field_name}", "action is reused by another extent")
            extent_action_ids.add(action_id)
        request_ids = []
        request_bytes = 0
        request_payloads = _require_tuple(
            extent.request_payload_bytes,
            f"{extent_path}.request_payload_bytes",
        )
        for request_index, entry in enumerate(request_payloads):
            entry_path = f"{extent_path}.request_payload_bytes[{request_index}]"
            if not isinstance(entry, tuple) or len(entry) != 2:
                _fail(entry_path, "expected a two-item tuple")
            request_ids.append(_string(entry[0], f"{entry_path}[0]"))
            request_bytes += _integer(entry[1], f"{entry_path}[1]", minimum=1)
        _validate_unique(request_ids, f"{extent_path}.request_payload_bytes request IDs")
        if request_ids != sorted(request_ids):
            _fail(
                f"{extent_path}.request_payload_bytes",
                "entries must be request-major",
            )
        if request_payloads and request_bytes != extent.payload_bytes:
            _fail(
                f"{extent_path}.request_payload_bytes",
                "partition must sum to extent payload_bytes",
            )
    if extent_action_ids != set(action_by_id):
        _fail(f"{path}.actions", "actions must be paired exactly to plan extents")

    entries = _validate_rank_action_ids(
        plan.entry_action_ids,
        f"{path}.entry_action_ids",
        rank_order=rank_order,
        action_by_id=action_by_id,
    )
    terminals = _validate_rank_action_ids(
        plan.terminal_action_ids,
        f"{path}.terminal_action_ids",
        rank_order=rank_order,
        action_by_id=action_by_id,
    )
    expected_entries = {
        rank: tuple(
            action.action_id
            for action in actions
            if action.rank == rank and not action.depends_on
        )
        for rank in rank_order
    }
    if entries != expected_entries:
        _fail(f"{path}.entry_action_ids", "does not match dependency roots")
    predecessor_ids = {
        dependency for action in actions for dependency in action.depends_on
    }
    expected_terminals = {
        rank: tuple(
            action.action_id
            for action in actions
            if action.rank == rank and action.action_id not in predecessor_ids
        )
        for rank in rank_order
    }
    if terminals != expected_terminals:
        _fail(f"{path}.terminal_action_ids", "does not match dependency terminals")

    if not re.fullmatch(r"[0-9a-f]{64}", plan.integrity_sha256):
        _fail(f"{path}.integrity_sha256", "expected a lowercase SHA-256 digest")
    if collective_plan_integrity_sha256(plan) != plan.integrity_sha256:
        _fail(f"{path}.integrity_sha256", "collective plan integrity mismatch")


def validate_execution_graph(graph: ExecutionGraph) -> None:
    """Validate graph structure, FIFO semantics and local payload values."""
    if not isinstance(graph, ExecutionGraph):
        _fail("graph", "expected ExecutionGraph")
    _string(graph.execution_id, "graph.execution_id")
    _integer(graph.step_index, "graph.step_index", nonnegative=True)
    _integer(graph.released_at_ps, "graph.released_at_ps", nonnegative=True)
    operations = _require_tuple(graph.operations, "graph.operations")
    completion_ids = _require_tuple(
        graph.completion_operation_ids,
        "graph.completion_operation_ids",
    )
    collective_plans = _require_tuple(
        graph.collective_plans,
        "graph.collective_plans",
    )
    operation_by_id: dict[str, ExecutionOperation] = {}
    for index, operation in enumerate(operations):
        path = f"graph.operations[{index}]"
        if not isinstance(operation, ExecutionOperation):
            _fail(path, "expected ExecutionOperation")
        operation_id = _string(operation.operation_id, f"{path}.operation_id")
        if operation_id in operation_by_id:
            _fail(f"{path}.operation_id", f"duplicate operation ID {operation_id!r}")
        operation_by_id[operation_id] = operation
        _integer(operation.rank, f"{path}.rank", nonnegative=True)
        _string(operation.logical_queue, f"{path}.logical_queue")
        dependencies = _require_tuple(operation.depends_on, f"{path}.depends_on")
        local_dependencies = _require_tuple(
            operation.participant_local_depends_on,
            f"{path}.participant_local_depends_on",
        )
        for field_name, values in (
            ("depends_on", dependencies),
            ("participant_local_depends_on", local_dependencies),
        ):
            for dep_index, dependency in enumerate(values):
                _string(dependency, f"{path}.{field_name}[{dep_index}]")
            _validate_unique(values, f"{path}.{field_name}")
            if operation_id in values:
                _fail(f"{path}.{field_name}", "operation cannot depend on itself")
        overlap = sorted(set(dependencies).intersection(local_dependencies))
        if overlap:
            _fail(path, f"dependency edges cannot have two modes: {overlap}")
        _integer(operation.not_before_ps, f"{path}.not_before_ps", nonnegative=True)
        _integer(operation.priority, f"{path}.priority")
        _validate_correlation(operation.correlation, f"{path}.correlation")
        _integer(operation.placement_epoch, f"{path}.placement_epoch", nonnegative=True)
        _validate_work(operation.work, f"{path}.work")
        if (
            isinstance(operation.work, CollectiveWork)
            and operation.rank not in operation.work.ranks
        ):
            _fail(
                f"{path}.rank",
                "collective anchor rank must be one of its participant ranks",
            )
        if isinstance(operation.work, CollectiveWork) and operation.work.request_pair_payload_bytes:
            attributed_requests = {entry[0] for entry in operation.work.request_pair_payload_bytes}
            unknown_requests = sorted(attributed_requests - set(operation.correlation.request_ids))
            if unknown_requests:
                _fail(
                    f"{path}.work.request_pair_payload_bytes",
                    f"request identities are absent from operation correlation: {unknown_requests}",
                )

    for index, operation in enumerate(operations):
        for field_name, dependencies in (
            ("depends_on", operation.depends_on),
            ("participant_local_depends_on", operation.participant_local_depends_on),
        ):
            for dep_index, dependency in enumerate(dependencies):
                if dependency not in operation_by_id:
                    _fail(
                        f"graph.operations[{index}].{field_name}[{dep_index}]",
                        f"unknown operation ID {dependency!r}",
                    )
        for dep_index, dependency in enumerate(operation.participant_local_depends_on):
            predecessor = operation_by_id[dependency]
            target_ranks = set(operation_participant_ranks(operation))
            predecessor_ranks = set(operation_participant_ranks(predecessor))
            if target_ranks.isdisjoint(predecessor_ranks):
                _fail(
                    f"graph.operations[{index}].participant_local_depends_on[{dep_index}]",
                    "participant-local dependency has no shared rank",
                )

    for index, completion_id in enumerate(completion_ids):
        _string(completion_id, f"graph.completion_operation_ids[{index}]")
        if completion_id not in operation_by_id:
            _fail(
                f"graph.completion_operation_ids[{index}]",
                f"unknown operation ID {completion_id!r}",
            )
    _validate_unique(completion_ids, "graph.completion_operation_ids")

    if collective_plans:
        collective_operations = tuple(
            operation
            for operation in operations
            if isinstance(operation.work, CollectiveWork)
        )
        if len(collective_plans) != len(collective_operations):
            _fail(
                "graph.collective_plans",
                "explicit plans must cover every collective operation exactly",
            )
        all_tags = []
        for index, (plan, operation) in enumerate(
            zip(collective_plans, collective_operations, strict=True)
        ):
            path = f"graph.collective_plans[{index}]"
            if not isinstance(plan, CollectivePlan):
                _fail(path, "expected CollectivePlan")
            if plan.operation_id != operation.operation_id:
                _fail(path, "plan order or operation identity disagrees with graph")
            assert isinstance(operation.work, CollectiveWork)
            _validate_collective_plan(plan, path, work=operation.work)
            all_tags.extend(round_.tag for round_ in plan.rounds)
        _validate_unique(all_tags, "graph.collective_plans round tags")

    edges = {
        (edge.predecessor_id, edge.operation_id)
        for edge in _effective_dependency_edges_unchecked(graph)
    }

    outgoing: dict[str, set[str]] = defaultdict(set)
    indegree = {operation_id: 0 for operation_id in operation_by_id}
    for source, destination in edges:
        if destination not in outgoing[source]:
            outgoing[source].add(destination)
            indegree[destination] += 1
    ready = deque(operation_id for operation_id, degree in indegree.items() if degree == 0)
    visited = 0
    while ready:
        source = ready.popleft()
        visited += 1
        for destination in outgoing[source]:
            indegree[destination] -= 1
            if indegree[destination] == 0:
                ready.append(destination)
    if visited != len(operation_by_id):
        cycle_ids = sorted(operation_id for operation_id, degree in indegree.items() if degree)
        _fail("graph", f"dependency and FIFO edges contain a cycle involving {cycle_ids}")


def effective_dependency_edges(
    graph: ExecutionGraph,
) -> tuple[EffectiveDependencyEdge, ...]:
    """Return every explicit and implicit ordering edge in canonical order.

    Participant-local dependencies are expanded by shared rank. Logical queue
    FIFO contributes a whole-operation edge between each pair of adjacent
    operations on the same rank and queue.
    """

    validate_execution_graph(graph)
    return _effective_dependency_edges_unchecked(graph)


def execution_graph_from_observations(
    record: StepRecord,
    observations: ExecutionObservations | None = None,
    *,
    execution_id: str | None = None,
) -> ExecutionGraph:
    """Envelope normalized adapter observations in one validated graph."""
    if not isinstance(record, StepRecord):
        _fail("record", "expected StepRecord")
    _integer(record.step_index, "record.step_index", nonnegative=True)
    _integer(record.virtual_time_ps, "record.virtual_time_ps", nonnegative=True)
    if observations is None:
        observations = ExecutionObservations()
    if not isinstance(observations, ExecutionObservations):
        _fail("observations", "expected ExecutionObservations")
    _require_tuple(observations.operations, "observations.operations")
    _require_tuple(
        observations.completion_operation_ids,
        "observations.completion_operation_ids",
    )
    graph = ExecutionGraph(
        execution_id=f"step-{record.step_index}" if execution_id is None else execution_id,
        step_index=record.step_index,
        released_at_ps=record.virtual_time_ps,
        operations=observations.operations,
        completion_operation_ids=observations.completion_operation_ids,
    )
    validate_execution_graph(graph)
    return graph


def _work_to_json(work: WorkPayload) -> dict[str, Any]:
    if isinstance(work, ComputeWork):
        return {
            "kind": "compute",
            "kernel": work.kernel,
            "config": [[name, value] for name, value in work.config],
            "flops": work.flops,
            "hbm_bytes": work.hbm_bytes,
            "nominal_duration_ps": work.nominal_duration_ps,
            "uncertainty_fraction": work.uncertainty_fraction,
        }
    if isinstance(work, KvCacheWork):
        return {
            "kind": "kv-cache",
            "action": work.action.value,
            "pool_id": work.pool_id,
            "request_id": work.request_id,
            "block_ids": list(work.block_ids),
            "token_start": work.token_start,
            "token_end": work.token_end,
            "layer": work.layer,
            "dtype": work.dtype,
            "byte_count": work.byte_count,
            "placement_epoch": work.placement_epoch,
            "reference_count": work.reference_count,
            "cause": work.cause,
            "correlation_id": work.correlation_id,
        }
    if isinstance(work, DmaWork):
        return {
            "kind": "dma",
            "descriptor_id": work.descriptor_id,
            "source": work.source,
            "destination": work.destination,
            "byte_count": work.byte_count,
        }
    if isinstance(work, CollectiveWork):
        payload = {
            "kind": "collective",
            "collective": work.collective,
            "ranks": list(work.ranks),
            "payload_bytes": work.payload_bytes,
            "algorithm_hint": work.algorithm_hint,
            "channel_hint": work.channel_hint,
        }
        if work.pair_payload_bytes:
            payload["pair_payload_bytes"] = [list(entry) for entry in work.pair_payload_bytes]
        if work.request_pair_payload_bytes:
            payload["request_pair_payload_bytes"] = [
                list(entry) for entry in work.request_pair_payload_bytes
            ]
        return payload
    if isinstance(work, ControlWork):
        return {
            "kind": "control",
            "message": work.message,
            "destination_ranks": list(work.destination_ranks),
            "payload_bytes": work.payload_bytes,
            "mode": work.mode.value,
        }
    _fail("work", f"unsupported work payload {type(work).__name__}")


def _config_from_json(value: Any, path: str) -> tuple[tuple[str, Any], ...]:
    entries: list[tuple[str, Any]] = []
    for index, raw_entry in enumerate(_array(value, path)):
        entry_path = f"{path}[{index}]"
        pair = _array(raw_entry, entry_path)
        if len(pair) != 2:
            _fail(entry_path, "expected a two-item array")
        name = _string(pair[0], f"{entry_path}[0]")
        scalar = pair[1]
        if (
            isinstance(scalar, bool)
            or type(scalar) is int
            or isinstance(scalar, str)
            or (type(scalar) is float and math.isfinite(scalar))
        ):
            pass
        else:
            _fail(f"{entry_path}[1]", "expected a finite JSON scalar")
        entries.append((name, scalar))
    result = tuple(entries)
    _validate_config(result, path)
    return result


def _work_from_json(value: Any, path: str) -> WorkPayload:
    payload = _object(value, path)
    if "kind" not in payload:
        _fail(path, "missing fields ['kind']")
    kind = _string(payload["kind"], f"{path}.kind")
    if kind == "compute":
        _fields(
            payload,
            path,
            required={"kind", "kernel"},
            optional={
                "config",
                "flops",
                "hbm_bytes",
                "nominal_duration_ps",
                "uncertainty_fraction",
            },
        )
        return ComputeWork(
            kernel=_string(payload["kernel"], f"{path}.kernel"),
            config=_config_from_json(payload.get("config", []), f"{path}.config"),
            flops=_integer(payload.get("flops", 0), f"{path}.flops", nonnegative=True),
            hbm_bytes=_integer(payload.get("hbm_bytes", 0), f"{path}.hbm_bytes", nonnegative=True),
            nominal_duration_ps=_optional_integer(
                payload.get("nominal_duration_ps"),
                f"{path}.nominal_duration_ps",
                nonnegative=True,
            ),
            uncertainty_fraction=_optional_number(
                payload.get("uncertainty_fraction"),
                f"{path}.uncertainty_fraction",
                nonnegative=True,
            ),
        )
    if kind == "kv-cache":
        _fields(
            payload,
            path,
            required={"kind", "action", "pool_id"},
            optional={
                "request_id",
                "block_ids",
                "token_start",
                "token_end",
                "layer",
                "dtype",
                "byte_count",
                "placement_epoch",
                "reference_count",
                "cause",
                "correlation_id",
            },
        )
        work = KvCacheWork(
            action=_enum_value(KvCacheAction, payload["action"], f"{path}.action"),
            pool_id=_string(payload["pool_id"], f"{path}.pool_id"),
            request_id=_optional_string(payload.get("request_id"), f"{path}.request_id"),
            block_ids=_string_tuple(payload.get("block_ids", []), f"{path}.block_ids"),
            token_start=_optional_integer(
                payload.get("token_start"), f"{path}.token_start", nonnegative=True
            ),
            token_end=_optional_integer(
                payload.get("token_end"), f"{path}.token_end", nonnegative=True
            ),
            layer=_optional_integer(payload.get("layer"), f"{path}.layer", nonnegative=True),
            dtype=_optional_string(payload.get("dtype"), f"{path}.dtype"),
            byte_count=_integer(
                payload.get("byte_count", 0), f"{path}.byte_count", nonnegative=True
            ),
            placement_epoch=_integer(
                payload.get("placement_epoch", 0),
                f"{path}.placement_epoch",
                nonnegative=True,
            ),
            reference_count=_optional_integer(
                payload.get("reference_count"),
                f"{path}.reference_count",
                nonnegative=True,
            ),
            cause=_optional_string(payload.get("cause"), f"{path}.cause"),
            correlation_id=_optional_string(
                payload.get("correlation_id"), f"{path}.correlation_id"
            ),
        )
        _validate_work(work, path)
        return work
    if kind == "dma":
        _fields(
            payload,
            path,
            required={"kind", "descriptor_id", "source", "destination", "byte_count"},
        )
        return DmaWork(
            descriptor_id=_string(payload["descriptor_id"], f"{path}.descriptor_id"),
            source=_string(payload["source"], f"{path}.source"),
            destination=_string(payload["destination"], f"{path}.destination"),
            byte_count=_integer(payload["byte_count"], f"{path}.byte_count", nonnegative=True),
        )
    if kind == "collective":
        _fields(
            payload,
            path,
            required={"kind", "collective", "ranks", "payload_bytes"},
            optional={
                "algorithm_hint",
                "channel_hint",
                "pair_payload_bytes",
                "request_pair_payload_bytes",
            },
        )
        raw_pair_payloads = _array(
            payload.get("pair_payload_bytes", []),
            f"{path}.pair_payload_bytes",
        )
        pair_payloads: list[tuple[int, int, int]] = []
        for index, raw_entry in enumerate(raw_pair_payloads):
            entry_path = f"{path}.pair_payload_bytes[{index}]"
            entry = _array(raw_entry, entry_path)
            if len(entry) != 3:
                _fail(entry_path, "expected a three-item array")
            pair_payloads.append(
                (
                    _integer(entry[0], f"{entry_path}[0]", nonnegative=True),
                    _integer(entry[1], f"{entry_path}[1]", nonnegative=True),
                    _integer(entry[2], f"{entry_path}[2]", minimum=1),
                )
            )
        raw_request_pair_payloads = _array(
            payload.get("request_pair_payload_bytes", []),
            f"{path}.request_pair_payload_bytes",
        )
        request_pair_payloads: list[tuple[str, int, int, int]] = []
        for index, raw_entry in enumerate(raw_request_pair_payloads):
            entry_path = f"{path}.request_pair_payload_bytes[{index}]"
            entry = _array(raw_entry, entry_path)
            if len(entry) != 4:
                _fail(entry_path, "expected a four-item array")
            request_pair_payloads.append(
                (
                    _string(entry[0], f"{entry_path}[0]"),
                    _integer(entry[1], f"{entry_path}[1]", nonnegative=True),
                    _integer(entry[2], f"{entry_path}[2]", nonnegative=True),
                    _integer(entry[3], f"{entry_path}[3]", minimum=1),
                )
            )
        work = CollectiveWork(
            collective=_string(payload["collective"], f"{path}.collective"),
            ranks=_int_tuple(payload["ranks"], f"{path}.ranks"),
            payload_bytes=_integer(
                payload["payload_bytes"], f"{path}.payload_bytes", nonnegative=True
            ),
            algorithm_hint=_optional_string(
                payload.get("algorithm_hint"), f"{path}.algorithm_hint"
            ),
            channel_hint=_optional_string(payload.get("channel_hint"), f"{path}.channel_hint"),
            pair_payload_bytes=tuple(pair_payloads),
            request_pair_payload_bytes=tuple(request_pair_payloads),
        )
        _validate_work(work, path)
        return work
    if kind == "control":
        _fields(
            payload,
            path,
            required={"kind", "message"},
            optional={"destination_ranks", "payload_bytes", "mode"},
        )
        work = ControlWork(
            message=_string(payload["message"], f"{path}.message"),
            destination_ranks=_int_tuple(
                payload.get("destination_ranks", []), f"{path}.destination_ranks"
            ),
            payload_bytes=_integer(
                payload.get("payload_bytes", 0), f"{path}.payload_bytes", nonnegative=True
            ),
            mode=_enum_value(
                ControlMode,
                payload.get("mode", ControlMode.ASYNCHRONOUS.value),
                f"{path}.mode",
            ),
        )
        _validate_work(work, path)
        return work
    _fail(f"{path}.kind", f"unknown work kind {kind!r}")


def _collective_plan_to_json(plan: CollectivePlan) -> dict[str, Any]:
    payload = _collective_plan_identity_to_json(plan)
    payload["integrity_sha256"] = plan.integrity_sha256
    return payload


def _rank_action_ids_from_json(
    value: Any,
    path: str,
) -> tuple[tuple[int, tuple[str, ...]], ...]:
    result = []
    for index, raw_entry in enumerate(_array(value, path)):
        entry_path = f"{path}[{index}]"
        entry = _array(raw_entry, entry_path)
        if len(entry) != 2:
            _fail(entry_path, "expected a two-item array")
        result.append(
            (
                _integer(entry[0], f"{entry_path}[0]", nonnegative=True),
                _string_tuple(entry[1], f"{entry_path}[1]"),
            )
        )
    return tuple(result)


def _collective_plan_from_json(value: Any, path: str) -> CollectivePlan:
    payload = _object(value, path)
    _fields(
        payload,
        path,
        required={
            "schema",
            "operation_id",
            "collective",
            "algorithm",
            "channel_id",
            "rank_order",
            "payload_bytes",
            "pair_payload_bytes",
            "request_pair_payload_bytes",
            "rounds",
            "actions",
            "extents",
            "entry_action_ids",
            "terminal_action_ids",
            "integrity_sha256",
        },
    )
    schema = _string(payload["schema"], f"{path}.schema")
    if schema != COLLECTIVE_PLAN_SCHEMA:
        _fail(
            f"{path}.schema",
            f"unsupported schema {schema!r}; expected {COLLECTIVE_PLAN_SCHEMA!r}",
        )

    pair_payloads = []
    for index, raw_entry in enumerate(
        _array(payload["pair_payload_bytes"], f"{path}.pair_payload_bytes")
    ):
        entry_path = f"{path}.pair_payload_bytes[{index}]"
        entry = _array(raw_entry, entry_path)
        if len(entry) != 3:
            _fail(entry_path, "expected a three-item array")
        pair_payloads.append(
            (
                _integer(entry[0], f"{entry_path}[0]", nonnegative=True),
                _integer(entry[1], f"{entry_path}[1]", nonnegative=True),
                _integer(entry[2], f"{entry_path}[2]", minimum=1),
            )
        )

    request_pair_payloads = []
    for index, raw_entry in enumerate(
        _array(
            payload["request_pair_payload_bytes"],
            f"{path}.request_pair_payload_bytes",
        )
    ):
        entry_path = f"{path}.request_pair_payload_bytes[{index}]"
        entry = _array(raw_entry, entry_path)
        if len(entry) != 4:
            _fail(entry_path, "expected a four-item array")
        request_pair_payloads.append(
            (
                _string(entry[0], f"{entry_path}[0]"),
                _integer(entry[1], f"{entry_path}[1]", nonnegative=True),
                _integer(entry[2], f"{entry_path}[2]", nonnegative=True),
                _integer(entry[3], f"{entry_path}[3]", minimum=1),
            )
        )

    rounds = []
    for index, raw_round in enumerate(_array(payload["rounds"], f"{path}.rounds")):
        round_path = f"{path}.rounds[{index}]"
        round_payload = _object(raw_round, round_path)
        _fields(
            round_payload,
            round_path,
            required={"round_index", "tag", "channel_id"},
        )
        rounds.append(
            CollectivePlanRound(
                round_index=_integer(
                    round_payload["round_index"],
                    f"{round_path}.round_index",
                    nonnegative=True,
                ),
                tag=_integer(
                    round_payload["tag"],
                    f"{round_path}.tag",
                    nonnegative=True,
                ),
                channel_id=_string(
                    round_payload["channel_id"],
                    f"{round_path}.channel_id",
                ),
            )
        )

    actions = []
    for index, raw_action in enumerate(_array(payload["actions"], f"{path}.actions")):
        action_path = f"{path}.actions[{index}]"
        action_payload = _object(raw_action, action_path)
        _fields(
            action_payload,
            action_path,
            required={"action_id", "rank", "kind", "extent_id", "depends_on"},
        )
        actions.append(
            CollectivePlanAction(
                action_id=_string(
                    action_payload["action_id"],
                    f"{action_path}.action_id",
                ),
                rank=_integer(
                    action_payload["rank"],
                    f"{action_path}.rank",
                    nonnegative=True,
                ),
                kind=_enum_value(
                    CollectivePlanActionKind,
                    action_payload["kind"],
                    f"{action_path}.kind",
                ),
                extent_id=_string(
                    action_payload["extent_id"],
                    f"{action_path}.extent_id",
                ),
                depends_on=_string_tuple(
                    action_payload["depends_on"],
                    f"{action_path}.depends_on",
                ),
            )
        )

    extents = []
    for index, raw_extent in enumerate(_array(payload["extents"], f"{path}.extents")):
        extent_path = f"{path}.extents[{index}]"
        extent_payload = _object(raw_extent, extent_path)
        _fields(
            extent_payload,
            extent_path,
            required={
                "extent_id",
                "round_index",
                "source_rank",
                "destination_rank",
                "payload_bytes",
                "send_action_id",
                "receive_action_id",
                "request_payload_bytes",
            },
        )
        request_payloads = []
        for request_index, raw_entry in enumerate(
            _array(
                extent_payload["request_payload_bytes"],
                f"{extent_path}.request_payload_bytes",
            )
        ):
            entry_path = f"{extent_path}.request_payload_bytes[{request_index}]"
            entry = _array(raw_entry, entry_path)
            if len(entry) != 2:
                _fail(entry_path, "expected a two-item array")
            request_payloads.append(
                (
                    _string(entry[0], f"{entry_path}[0]"),
                    _integer(entry[1], f"{entry_path}[1]", minimum=1),
                )
            )
        extents.append(
            CollectivePlanExtent(
                extent_id=_string(
                    extent_payload["extent_id"],
                    f"{extent_path}.extent_id",
                ),
                round_index=_integer(
                    extent_payload["round_index"],
                    f"{extent_path}.round_index",
                    nonnegative=True,
                ),
                source_rank=_integer(
                    extent_payload["source_rank"],
                    f"{extent_path}.source_rank",
                    nonnegative=True,
                ),
                destination_rank=_integer(
                    extent_payload["destination_rank"],
                    f"{extent_path}.destination_rank",
                    nonnegative=True,
                ),
                payload_bytes=_integer(
                    extent_payload["payload_bytes"],
                    f"{extent_path}.payload_bytes",
                    minimum=1,
                ),
                send_action_id=_string(
                    extent_payload["send_action_id"],
                    f"{extent_path}.send_action_id",
                ),
                receive_action_id=_string(
                    extent_payload["receive_action_id"],
                    f"{extent_path}.receive_action_id",
                ),
                request_payload_bytes=tuple(request_payloads),
            )
        )

    return CollectivePlan(
        operation_id=_string(payload["operation_id"], f"{path}.operation_id"),
        collective=_string(payload["collective"], f"{path}.collective"),
        algorithm=_string(payload["algorithm"], f"{path}.algorithm"),
        channel_id=_string(payload["channel_id"], f"{path}.channel_id"),
        rank_order=_int_tuple(payload["rank_order"], f"{path}.rank_order"),
        payload_bytes=_integer(
            payload["payload_bytes"],
            f"{path}.payload_bytes",
            nonnegative=True,
        ),
        pair_payload_bytes=tuple(pair_payloads),
        request_pair_payload_bytes=tuple(request_pair_payloads),
        rounds=tuple(rounds),
        actions=tuple(actions),
        extents=tuple(extents),
        entry_action_ids=_rank_action_ids_from_json(
            payload["entry_action_ids"],
            f"{path}.entry_action_ids",
        ),
        terminal_action_ids=_rank_action_ids_from_json(
            payload["terminal_action_ids"],
            f"{path}.terminal_action_ids",
        ),
        integrity_sha256=_string(
            payload["integrity_sha256"],
            f"{path}.integrity_sha256",
        ),
    )


def execution_graph_to_json(graph: ExecutionGraph) -> dict[str, Any]:
    """Return the canonical JSON-ready form of one execution graph."""
    validate_execution_graph(graph)
    payload = {
        "schema": EXECUTION_GRAPH_SCHEMA,
        "execution_id": graph.execution_id,
        "step_index": graph.step_index,
        "released_at_ps": graph.released_at_ps,
        "operations": [
            {
                "operation_id": operation.operation_id,
                "rank": operation.rank,
                "logical_queue": operation.logical_queue,
                "work": _work_to_json(operation.work),
                "depends_on": list(operation.depends_on),
                "participant_local_depends_on": list(operation.participant_local_depends_on),
                "not_before_ps": operation.not_before_ps,
                "priority": operation.priority,
                "correlation": _correlation_to_json(operation.correlation),
                "placement_epoch": operation.placement_epoch,
            }
            for operation in graph.operations
        ],
        "completion_operation_ids": list(graph.completion_operation_ids),
    }
    if graph.collective_plans:
        payload["collective_plans"] = [
            _collective_plan_to_json(plan) for plan in graph.collective_plans
        ]
    return payload


def execution_graph_from_json(value: Any) -> ExecutionGraph:
    """Parse and validate one canonical execution-graph payload."""
    payload = _object(value, "graph")
    if "schema" not in payload:
        _fail("graph", "missing fields ['schema']")
    schema = _string(payload["schema"], "graph.schema")
    if schema != EXECUTION_GRAPH_SCHEMA:
        _fail(
            "graph.schema",
            f"unsupported schema {schema!r}; expected {EXECUTION_GRAPH_SCHEMA!r}",
        )
    _fields(
        payload,
        "graph",
        required={"schema", "execution_id", "step_index", "released_at_ps"},
        optional={"operations", "completion_operation_ids", "collective_plans"},
    )
    operations: list[ExecutionOperation] = []
    for index, raw_operation in enumerate(
        _array(payload.get("operations", []), "graph.operations")
    ):
        path = f"graph.operations[{index}]"
        operation = _object(raw_operation, path)
        _fields(
            operation,
            path,
            required={"operation_id", "rank", "logical_queue", "work"},
            optional={
                "depends_on",
                "participant_local_depends_on",
                "not_before_ps",
                "priority",
                "correlation",
                "placement_epoch",
            },
        )
        operations.append(
            ExecutionOperation(
                operation_id=_string(operation["operation_id"], f"{path}.operation_id"),
                rank=_integer(operation["rank"], f"{path}.rank", nonnegative=True),
                logical_queue=_string(operation["logical_queue"], f"{path}.logical_queue"),
                work=_work_from_json(operation["work"], f"{path}.work"),
                depends_on=_string_tuple(operation.get("depends_on", []), f"{path}.depends_on"),
                not_before_ps=_integer(
                    operation.get("not_before_ps", 0),
                    f"{path}.not_before_ps",
                    nonnegative=True,
                ),
                priority=_integer(operation.get("priority", 0), f"{path}.priority"),
                correlation=_correlation_from_json(
                    operation.get("correlation", {}), f"{path}.correlation"
                ),
                placement_epoch=_integer(
                    operation.get("placement_epoch", 0),
                    f"{path}.placement_epoch",
                    nonnegative=True,
                ),
                participant_local_depends_on=_string_tuple(
                    operation.get("participant_local_depends_on", []),
                    f"{path}.participant_local_depends_on",
                ),
            )
        )
    graph = ExecutionGraph(
        execution_id=_string(payload["execution_id"], "graph.execution_id"),
        step_index=_integer(payload["step_index"], "graph.step_index", nonnegative=True),
        released_at_ps=_integer(
            payload["released_at_ps"], "graph.released_at_ps", nonnegative=True
        ),
        operations=tuple(operations),
        completion_operation_ids=_string_tuple(
            payload.get("completion_operation_ids", []),
            "graph.completion_operation_ids",
        ),
        collective_plans=tuple(
            _collective_plan_from_json(entry, f"graph.collective_plans[{index}]")
            for index, entry in enumerate(
                _array(payload.get("collective_plans", []), "graph.collective_plans")
            )
        ),
    )
    validate_execution_graph(graph)
    return graph


def _validate_resource(resource: ResourceRef, path: str) -> None:
    if not isinstance(resource, ResourceRef):
        _fail(path, "expected ResourceRef")
    if not isinstance(resource.kind, ResourceKind):
        _fail(f"{path}.kind", "expected ResourceKind")
    _string(resource.resource_id, f"{path}.resource_id")


def _validate_completion_event(event: CompletionEvent, path: str = "event") -> None:
    if not isinstance(event, CompletionEvent):
        _fail(path, "expected CompletionEvent")
    _string(event.execution_id, f"{path}.execution_id")
    _string(event.operation_id, f"{path}.operation_id")
    if not isinstance(event.phase, EventPhase):
        _fail(f"{path}.phase", "expected EventPhase")
    _integer(event.timestamp_ps, f"{path}.timestamp_ps", nonnegative=True)
    if event.resource is not None:
        _validate_resource(event.resource, f"{path}.resource")
    if event.completed_bytes is not None:
        _integer(event.completed_bytes, f"{path}.completed_bytes", nonnegative=True)
    if event.subject_object_id is not None:
        _string(event.subject_object_id, f"{path}.subject_object_id")


def completion_event_to_json(event: CompletionEvent) -> dict[str, Any]:
    """Return the canonical JSON-ready form of one completion event."""
    _validate_completion_event(event)
    resource = None
    if event.resource is not None:
        resource = {
            "kind": event.resource.kind.value,
            "resource_id": event.resource.resource_id,
        }
    return {
        "schema": COMPLETION_EVENT_SCHEMA,
        "execution_id": event.execution_id,
        "operation_id": event.operation_id,
        "phase": event.phase.value,
        "timestamp_ps": event.timestamp_ps,
        "resource": resource,
        "completed_bytes": event.completed_bytes,
        "subject_object_id": event.subject_object_id,
    }


def completion_event_from_json(value: Any) -> CompletionEvent:
    """Parse and validate one canonical completion-event payload."""
    payload = _object(value, "event")
    if "schema" not in payload:
        _fail("event", "missing fields ['schema']")
    schema = _string(payload["schema"], "event.schema")
    if schema != COMPLETION_EVENT_SCHEMA:
        _fail(
            "event.schema",
            f"unsupported schema {schema!r}; expected {COMPLETION_EVENT_SCHEMA!r}",
        )
    _fields(
        payload,
        "event",
        required={"schema", "execution_id", "operation_id", "phase", "timestamp_ps"},
        optional={"resource", "completed_bytes", "subject_object_id"},
    )
    resource = None
    raw_resource = payload.get("resource")
    if raw_resource is not None:
        resource_payload = _object(raw_resource, "event.resource")
        _fields(
            resource_payload,
            "event.resource",
            required={"kind", "resource_id"},
        )
        resource = ResourceRef(
            kind=_enum_value(ResourceKind, resource_payload["kind"], "event.resource.kind"),
            resource_id=_string(resource_payload["resource_id"], "event.resource.resource_id"),
        )
    event = CompletionEvent(
        execution_id=_string(payload["execution_id"], "event.execution_id"),
        operation_id=_string(payload["operation_id"], "event.operation_id"),
        phase=_enum_value(EventPhase, payload["phase"], "event.phase"),
        timestamp_ps=_integer(payload["timestamp_ps"], "event.timestamp_ps", nonnegative=True),
        resource=resource,
        completed_bytes=_optional_integer(
            payload.get("completed_bytes"), "event.completed_bytes", nonnegative=True
        ),
        subject_object_id=_optional_string(
            payload.get("subject_object_id"), "event.subject_object_id"
        ),
    )
    _validate_completion_event(event)
    return event


def _validate_execution_result(result: ExecutionResult) -> None:
    if not isinstance(result, ExecutionResult):
        _fail("result", "expected ExecutionResult")
    _string(result.execution_id, "result.execution_id")
    _integer(result.completed_at_ps, "result.completed_at_ps", nonnegative=True)
    events = _require_tuple(result.events, "result.events")
    previous_timestamp = -1
    for index, event in enumerate(events):
        path = f"result.events[{index}]"
        _validate_completion_event(event, path)
        if event.execution_id != result.execution_id:
            _fail(path, "execution_id does not match result.execution_id")
        if event.timestamp_ps < previous_timestamp:
            _fail(path, "event timestamps must be nondecreasing")
        previous_timestamp = event.timestamp_ps
    if result.quiesced_at_ps is not None:
        _integer(result.quiesced_at_ps, "result.quiesced_at_ps", nonnegative=True)
        if result.quiesced_at_ps < result.completed_at_ps:
            _fail("result.quiesced_at_ps", "must not precede completed_at_ps")
        event_bound = result.quiesced_at_ps
    else:
        event_bound = result.completed_at_ps
    if events and events[-1].timestamp_ps > event_bound:
        _fail("result.events", "event timestamp exceeds the reported result boundary")


def execution_result_to_json(result: ExecutionResult) -> dict[str, Any]:
    """Return the canonical JSON-ready form of one execution result."""
    _validate_execution_result(result)
    return {
        "schema": EXECUTION_RESULT_SCHEMA,
        "execution_id": result.execution_id,
        "completed_at_ps": result.completed_at_ps,
        "events": [completion_event_to_json(event) for event in result.events],
        "quiesced_at_ps": result.quiesced_at_ps,
    }


def execution_result_from_json(value: Any) -> ExecutionResult:
    """Parse and validate one canonical execution-result payload."""
    payload = _object(value, "result")
    if "schema" not in payload:
        _fail("result", "missing fields ['schema']")
    schema = _string(payload["schema"], "result.schema")
    if schema != EXECUTION_RESULT_SCHEMA:
        _fail(
            "result.schema",
            f"unsupported schema {schema!r}; expected {EXECUTION_RESULT_SCHEMA!r}",
        )
    _fields(
        payload,
        "result",
        required={"schema", "execution_id", "completed_at_ps"},
        optional={"events", "quiesced_at_ps"},
    )
    events = tuple(
        completion_event_from_json(entry)
        for entry in _array(payload.get("events", []), "result.events")
    )
    result = ExecutionResult(
        execution_id=_string(payload["execution_id"], "result.execution_id"),
        completed_at_ps=_integer(
            payload["completed_at_ps"], "result.completed_at_ps", nonnegative=True
        ),
        events=events,
        quiesced_at_ps=_optional_integer(
            payload.get("quiesced_at_ps"), "result.quiesced_at_ps", nonnegative=True
        ),
    )
    _validate_execution_result(result)
    return result
