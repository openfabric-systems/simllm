"""Validation and JSON wire forms for execution and completion contracts."""

from __future__ import annotations

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
    COMPLETION_EVENT_SCHEMA,
    EXECUTION_GRAPH_SCHEMA,
    EXECUTION_RESULT_SCHEMA,
    CollectiveWork,
    CompletionEvent,
    ComputeWork,
    ControlMode,
    ControlWork,
    DmaWork,
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


def _operation_participant_ranks(operation: ExecutionOperation) -> set[int]:
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
    return ranks


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
            target_ranks = _operation_participant_ranks(operation)
            predecessor_ranks = _operation_participant_ranks(predecessor)
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

    edges: set[tuple[str, str]] = set()
    for operation in operations:
        edges.update((dependency, operation.operation_id) for dependency in operation.depends_on)
        edges.update(
            (dependency, operation.operation_id)
            for dependency in operation.participant_local_depends_on
        )
    queue_tail: dict[tuple[int, str], str] = {}
    for operation in operations:
        queue = (operation.rank, operation.logical_queue)
        previous = queue_tail.get(queue)
        if previous is not None:
            edges.add((previous, operation.operation_id))
        queue_tail[queue] = operation.operation_id

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


def execution_graph_to_json(graph: ExecutionGraph) -> dict[str, Any]:
    """Return the canonical JSON-ready form of one execution graph."""
    validate_execution_graph(graph)
    return {
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
        optional={"operations", "completion_operation_ids"},
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
