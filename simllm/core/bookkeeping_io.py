"""Strict JSON wire form for central request bookkeeping."""

from __future__ import annotations

import enum
import math
from collections.abc import Mapping
from typing import Any, NoReturn, TypeVar

from simllm.core.bookkeeping import (
    BOOKKEEPING_SCHEMA,
    BookkeepingEntry,
    BookkeepingLedger,
    BookkeepingScope,
    CreatedObjectKind,
    CreatedObjectRecord,
    CreatedObjectRef,
    ObjectMetadata,
    ObjectOwner,
    ProcessingStage,
    StagePhase,
    StageRecord,
    validate_bookkeeping_ledger,
)
from simllm.core.execution import (
    CompletionEvent,
    EventPhase,
    OperationCorrelation,
    ResourceKind,
    ResourceRef,
)

_EnumT = TypeVar("_EnumT", bound=enum.Enum)


def _fail(path: str, message: str) -> NoReturn:
    raise ValueError(f"{path}: {message}")


def _object(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        _fail(path, "expected an object")
    if any(not isinstance(key, str) for key in value):
        _fail(path, "object keys must be strings")
    return value


def _fields(
    value: Mapping[str, Any],
    path: str,
    *,
    required: set[str],
) -> None:
    missing = sorted(required - value.keys())
    if missing:
        _fail(path, f"missing fields {missing}")
    unknown = sorted(value.keys() - required)
    if unknown:
        _fail(path, f"unknown fields {unknown}")


def _array(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(path, "expected an array")
    return value


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str):
        _fail(path, "expected a string")
    if not value.strip():
        _fail(path, "must not be blank")
    return value


def _optional_string(value: Any, path: str) -> str | None:
    if value is None:
        return None
    return _string(value, path)


def _integer(value: Any, path: str, *, nonnegative: bool = False) -> int:
    if type(value) is not int:
        _fail(path, "expected an integer")
    if nonnegative and value < 0:
        _fail(path, "must be nonnegative")
    return value


def _optional_integer(value: Any, path: str) -> int | None:
    if value is None:
        return None
    return _integer(value, path, nonnegative=True)


def _enum_value(cls: type[_EnumT], value: Any, path: str) -> _EnumT:
    raw = _string(value, path)
    try:
        return cls(raw)
    except ValueError as exc:
        choices = [member.value for member in cls]
        _fail(path, f"unknown value {raw!r}; expected one of {choices}")
        raise AssertionError from exc


def _scalar(value: Any, path: str) -> str | int | float | bool:
    if isinstance(value, bool) or type(value) is int or isinstance(value, str):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    _fail(path, "expected a finite JSON scalar")


def _correlation_to_json(correlation: OperationCorrelation) -> dict[str, Any]:
    return {
        "request_ids": list(correlation.request_ids),
        "batch_id": correlation.batch_id,
        "layer": correlation.layer,
        "microbatch": correlation.microbatch,
        "iteration": correlation.iteration,
    }


def _correlation_from_json(value: Any, path: str) -> OperationCorrelation:
    obj = _object(value, path)
    _fields(
        obj,
        path,
        required={"request_ids", "batch_id", "layer", "microbatch", "iteration"},
    )
    request_ids = tuple(
        _string(item, f"{path}.request_ids[{index}]")
        for index, item in enumerate(_array(obj["request_ids"], f"{path}.request_ids"))
    )
    return OperationCorrelation(
        request_ids=request_ids,
        batch_id=_optional_string(obj["batch_id"], f"{path}.batch_id"),
        layer=_optional_integer(obj["layer"], f"{path}.layer"),
        microbatch=_optional_integer(obj["microbatch"], f"{path}.microbatch"),
        iteration=_optional_integer(obj["iteration"], f"{path}.iteration"),
    )


def _scope_to_json(scope: BookkeepingScope) -> dict[str, Any]:
    return {
        "correlation": _correlation_to_json(scope.correlation),
        "step_index": scope.step_index,
        "execution_id": scope.execution_id,
        "operation_id": scope.operation_id,
    }


def _scope_from_json(value: Any, path: str) -> BookkeepingScope:
    obj = _object(value, path)
    _fields(
        obj,
        path,
        required={"correlation", "step_index", "execution_id", "operation_id"},
    )
    return BookkeepingScope(
        correlation=_correlation_from_json(obj["correlation"], f"{path}.correlation"),
        step_index=_optional_integer(obj["step_index"], f"{path}.step_index"),
        execution_id=_optional_string(obj["execution_id"], f"{path}.execution_id"),
        operation_id=_optional_string(obj["operation_id"], f"{path}.operation_id"),
    )


def _ref_to_json(ref: CreatedObjectRef) -> dict[str, str]:
    return {"kind": ref.kind.value, "object_id": ref.object_id}


def _ref_from_json(value: Any, path: str) -> CreatedObjectRef:
    obj = _object(value, path)
    _fields(obj, path, required={"kind", "object_id"})
    return CreatedObjectRef(
        kind=_enum_value(CreatedObjectKind, obj["kind"], f"{path}.kind"),
        object_id=_string(obj["object_id"], f"{path}.object_id"),
    )


def _refs_from_json(value: Any, path: str) -> tuple[CreatedObjectRef, ...]:
    return tuple(
        _ref_from_json(item, f"{path}[{index}]")
        for index, item in enumerate(_array(value, path))
    )


def _metadata_to_json(metadata: ObjectMetadata) -> list[list[Any]]:
    return [[name, value] for name, value in metadata]


def _metadata_from_json(value: Any, path: str) -> ObjectMetadata:
    result: list[tuple[str, str | int | float | bool]] = []
    for index, item in enumerate(_array(value, path)):
        item_path = f"{path}[{index}]"
        pair = _array(item, item_path)
        if len(pair) != 2:
            _fail(item_path, "expected a two-item array")
        result.append(
            (
                _string(pair[0], f"{item_path}[0]"),
                _scalar(pair[1], f"{item_path}[1]"),
            )
        )
    return tuple(result)


def _resource_to_json(resource: ResourceRef | None) -> dict[str, str] | None:
    if resource is None:
        return None
    return {"kind": resource.kind.value, "resource_id": resource.resource_id}


def _resource_from_json(value: Any, path: str) -> ResourceRef | None:
    if value is None:
        return None
    obj = _object(value, path)
    _fields(obj, path, required={"kind", "resource_id"})
    return ResourceRef(
        kind=_enum_value(ResourceKind, obj["kind"], f"{path}.kind"),
        resource_id=_string(obj["resource_id"], f"{path}.resource_id"),
    )


def _fact_to_json(fact: CreatedObjectRecord | StageRecord | CompletionEvent) -> dict[str, Any]:
    if isinstance(fact, CreatedObjectRecord):
        return {
            "kind": "object",
            "ref": _ref_to_json(fact.ref),
            "owner": fact.owner.value,
            "created_at_ps": fact.created_at_ps,
            "scope": _scope_to_json(fact.scope),
            "native_id": fact.native_id,
            "parent_refs": [_ref_to_json(ref) for ref in fact.parent_refs],
            "metadata": _metadata_to_json(fact.metadata),
        }
    if isinstance(fact, StageRecord):
        return {
            "kind": "stage",
            "stage": fact.stage.value,
            "phase": fact.phase.value,
            "timestamp_ps": fact.timestamp_ps,
            "scope": _scope_to_json(fact.scope),
            "object_refs": [_ref_to_json(ref) for ref in fact.object_refs],
        }
    if isinstance(fact, CompletionEvent):
        return {
            "kind": "completion",
            "execution_id": fact.execution_id,
            "operation_id": fact.operation_id,
            "phase": fact.phase.value,
            "timestamp_ps": fact.timestamp_ps,
            "resource": _resource_to_json(fact.resource),
            "completed_bytes": fact.completed_bytes,
            "subject_object_id": fact.subject_object_id,
        }
    _fail("ledger.fact", f"unsupported fact {type(fact).__name__}")


def _object_fact_from_json(obj: Mapping[str, Any], path: str) -> CreatedObjectRecord:
    _fields(
        obj,
        path,
        required={
            "kind",
            "ref",
            "owner",
            "created_at_ps",
            "scope",
            "native_id",
            "parent_refs",
            "metadata",
        },
    )
    return CreatedObjectRecord(
        ref=_ref_from_json(obj["ref"], f"{path}.ref"),
        owner=_enum_value(ObjectOwner, obj["owner"], f"{path}.owner"),
        created_at_ps=_integer(
            obj["created_at_ps"], f"{path}.created_at_ps", nonnegative=True
        ),
        scope=_scope_from_json(obj["scope"], f"{path}.scope"),
        native_id=_optional_string(obj["native_id"], f"{path}.native_id"),
        parent_refs=_refs_from_json(obj["parent_refs"], f"{path}.parent_refs"),
        metadata=_metadata_from_json(obj["metadata"], f"{path}.metadata"),
    )


def _stage_fact_from_json(obj: Mapping[str, Any], path: str) -> StageRecord:
    _fields(
        obj,
        path,
        required={"kind", "stage", "phase", "timestamp_ps", "scope", "object_refs"},
    )
    return StageRecord(
        stage=_enum_value(ProcessingStage, obj["stage"], f"{path}.stage"),
        phase=_enum_value(StagePhase, obj["phase"], f"{path}.phase"),
        timestamp_ps=_integer(
            obj["timestamp_ps"], f"{path}.timestamp_ps", nonnegative=True
        ),
        scope=_scope_from_json(obj["scope"], f"{path}.scope"),
        object_refs=_refs_from_json(obj["object_refs"], f"{path}.object_refs"),
    )


def _completion_fact_from_json(obj: Mapping[str, Any], path: str) -> CompletionEvent:
    _fields(
        obj,
        path,
        required={
            "kind",
            "execution_id",
            "operation_id",
            "phase",
            "timestamp_ps",
            "resource",
            "completed_bytes",
            "subject_object_id",
        },
    )
    return CompletionEvent(
        execution_id=_string(obj["execution_id"], f"{path}.execution_id"),
        operation_id=_string(obj["operation_id"], f"{path}.operation_id"),
        phase=_enum_value(EventPhase, obj["phase"], f"{path}.phase"),
        timestamp_ps=_integer(
            obj["timestamp_ps"], f"{path}.timestamp_ps", nonnegative=True
        ),
        resource=_resource_from_json(obj["resource"], f"{path}.resource"),
        completed_bytes=_optional_integer(
            obj["completed_bytes"], f"{path}.completed_bytes"
        ),
        subject_object_id=_optional_string(
            obj["subject_object_id"], f"{path}.subject_object_id"
        ),
    )


def _fact_from_json(value: Any, path: str) -> CreatedObjectRecord | StageRecord | CompletionEvent:
    obj = _object(value, path)
    if "kind" not in obj:
        _fail(path, "missing fields ['kind']")
    kind = _string(obj["kind"], f"{path}.kind")
    if kind == "object":
        return _object_fact_from_json(obj, path)
    if kind == "stage":
        return _stage_fact_from_json(obj, path)
    if kind == "completion":
        return _completion_fact_from_json(obj, path)
    _fail(f"{path}.kind", "unknown entry kind; expected one of ['object', 'stage', 'completion']")


def bookkeeping_ledger_to_json(ledger: BookkeepingLedger) -> dict[str, Any]:
    """Return a validated JSON-compatible bookkeeping ledger."""

    validate_bookkeeping_ledger(ledger)
    return {
        "schema": BOOKKEEPING_SCHEMA,
        "entries": [
            {"sequence": entry.sequence, "fact": _fact_to_json(entry.fact)}
            for entry in ledger.entries
        ],
    }


def bookkeeping_ledger_from_json(value: Any) -> BookkeepingLedger:
    """Parse a strict schema-first JSON bookkeeping ledger."""

    obj = _object(value, "ledger")
    _fields(obj, "ledger", required={"schema", "entries"})
    schema = _string(obj["schema"], "ledger.schema")
    if schema != BOOKKEEPING_SCHEMA:
        _fail("ledger.schema", f"expected {BOOKKEEPING_SCHEMA!r}, got {schema!r}")
    entries: list[BookkeepingEntry] = []
    for index, value_entry in enumerate(_array(obj["entries"], "ledger.entries")):
        path = f"ledger.entries[{index}]"
        entry = _object(value_entry, path)
        _fields(entry, path, required={"sequence", "fact"})
        entries.append(
            BookkeepingEntry(
                sequence=_integer(entry["sequence"], f"{path}.sequence", nonnegative=True),
                fact=_fact_from_json(entry["fact"], f"{path}.fact"),
            )
        )
    ledger = BookkeepingLedger(tuple(entries))
    validate_bookkeeping_ledger(ledger)
    return ledger
