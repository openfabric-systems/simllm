"""Arrival and oracle join projected into request bookkeeping.

The trace remains the immutable oracle authority. A joined run is a strict,
versioned projection that names the trace bytes it consumed. Each projected
request is also appended as one framework-request object to the caller's
existing :class:`simllm.core.RequestBookkeeper`.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from simllm.core import (
    BookkeepingScope,
    CreatedObjectKind,
    CreatedObjectRecord,
    CreatedObjectRef,
    ObjectOwner,
    OperationCorrelation,
    RequestBookkeeper,
)
from simllm.core._wire import _fields, _integer, _object, _string
from simllm.preplay.schema import PREPLAY_TRACE_SCHEMA, StopReason
from simllm.preplay.trace import read_preplay_trace

PREPLAY_REPLAY_RUN_SCHEMA = "simllm-preplay-replay-run-v1"
PREPLAY_ROUTING_REFERENCE_SCHEMA = "simllm-preplay-routing-reference-v1"

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True, kw_only=True)
class RequestArrival:
    """One realized framework-entry arrival in integer picoseconds."""

    request_id: str
    arrived_at_ps: int


@dataclass(frozen=True, kw_only=True)
class TraceArtifactReference:
    """Stable identity of the exact trace bytes consumed by a replay run."""

    path: str
    sha256: str
    schema: str = PREPLAY_TRACE_SCHEMA


@dataclass(frozen=True, kw_only=True)
class RoutingReference:
    """Versioned pointer to one request's routing rows in a trace."""

    trace_sha256: str
    request_id: str
    schema: str = PREPLAY_ROUTING_REFERENCE_SCHEMA
    trace_schema: str = PREPLAY_TRACE_SCHEMA


@dataclass(frozen=True, kw_only=True)
class JoinedRequest:
    """Immutable request outcome pinned before the first scheduler step."""

    request_id: str
    arrived_at_ps: int
    output_length: int
    stop_reason: StopReason
    output_token_ids: tuple[int, ...]
    routing_reference: RoutingReference
    bookkeeping_object_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_token_ids", tuple(self.output_token_ids))


@dataclass(frozen=True, kw_only=True)
class PreplayReplayRun:
    """Versioned run record naming its trace and all joined requests."""

    trace: TraceArtifactReference
    requests: tuple[JoinedRequest, ...]
    schema: str = PREPLAY_REPLAY_RUN_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "requests", tuple(self.requests))

    def by_request_id(self, request_id: str) -> JoinedRequest:
        """Return one joined request or fail with the stable request identity."""

        for request in self.requests:
            if request.request_id == request_id:
                return request
        raise KeyError(f"request ID {request_id!r} not in joined pre-play run")


def _require_text(value: object, path: str) -> str:
    text = _string(value, path)
    if not text.strip():
        raise ValueError(f"{path}: must be a nonblank string")
    return text


def _require_sha256(value: object, path: str) -> str:
    digest = _string(value, path)
    if _SHA256_RE.fullmatch(digest) is None:
        raise ValueError(f"{path}: expected 64 lowercase hexadecimal digits")
    return digest


def _validate_arrival(value: RequestArrival, path: str) -> None:
    if not isinstance(value, RequestArrival):
        raise TypeError(f"{path}: expected RequestArrival")
    _require_text(value.request_id, f"{path}.request_id")
    _integer(value.arrived_at_ps, f"{path}.arrived_at_ps", nonnegative=True)


def _validate_trace_reference(value: TraceArtifactReference, path: str) -> None:
    if not isinstance(value, TraceArtifactReference):
        raise TypeError(f"{path}: expected TraceArtifactReference")
    if value.schema != PREPLAY_TRACE_SCHEMA:
        raise ValueError(f"{path}.schema: unsupported schema {value.schema!r}")
    _require_text(value.path, f"{path}.path")
    _require_sha256(value.sha256, f"{path}.sha256")


def _validate_routing_reference(value: RoutingReference, path: str) -> None:
    if not isinstance(value, RoutingReference):
        raise TypeError(f"{path}: expected RoutingReference")
    if value.schema != PREPLAY_ROUTING_REFERENCE_SCHEMA:
        raise ValueError(f"{path}.schema: unsupported schema {value.schema!r}")
    if value.trace_schema != PREPLAY_TRACE_SCHEMA:
        raise ValueError(
            f"{path}.trace_schema: unsupported schema {value.trace_schema!r}"
        )
    _require_sha256(value.trace_sha256, f"{path}.trace_sha256")
    _require_text(value.request_id, f"{path}.request_id")


def _validate_joined_request(value: JoinedRequest, path: str) -> None:
    if not isinstance(value, JoinedRequest):
        raise TypeError(f"{path}: expected JoinedRequest")
    _require_text(value.request_id, f"{path}.request_id")
    _integer(value.arrived_at_ps, f"{path}.arrived_at_ps", nonnegative=True)
    _integer(value.output_length, f"{path}.output_length", nonnegative=True)
    if not isinstance(value.stop_reason, StopReason):
        raise TypeError(f"{path}.stop_reason: expected StopReason")
    if not isinstance(value.output_token_ids, tuple):
        raise TypeError(f"{path}.output_token_ids: in-memory contract requires a tuple")
    for index, token_id in enumerate(value.output_token_ids):
        _integer(token_id, f"{path}.output_token_ids[{index}]", nonnegative=True)
    if value.output_length != len(value.output_token_ids):
        raise ValueError(
            f"{path}.output_length: expected {len(value.output_token_ids)} "
            "from output_token_ids"
        )
    if value.output_length == 0:
        raise ValueError(f"{path}.output_length: pre-play output must not be empty")
    _validate_routing_reference(value.routing_reference, f"{path}.routing_reference")
    if value.routing_reference.request_id != value.request_id:
        raise ValueError(
            f"{path}.routing_reference.request_id: must match request_id"
        )
    _require_text(value.bookkeeping_object_id, f"{path}.bookkeeping_object_id")


def validate_preplay_replay_run(value: PreplayReplayRun) -> None:
    """Validate trace identity, request uniqueness and routing consistency."""

    if not isinstance(value, PreplayReplayRun):
        raise TypeError("run: expected PreplayReplayRun")
    if value.schema != PREPLAY_REPLAY_RUN_SCHEMA:
        raise ValueError(f"run.schema: unsupported schema {value.schema!r}")
    _validate_trace_reference(value.trace, "run.trace")
    if not isinstance(value.requests, tuple):
        raise TypeError("run.requests: in-memory contract requires a tuple")
    if not value.requests:
        raise ValueError("run.requests: must not be empty")
    request_ids: list[str] = []
    object_ids: list[str] = []
    for index, request in enumerate(value.requests):
        path = f"run.requests[{index}]"
        _validate_joined_request(request, path)
        if request.routing_reference.trace_sha256 != value.trace.sha256:
            raise ValueError(
                f"{path}.routing_reference.trace_sha256: must match run.trace.sha256"
            )
        request_ids.append(request.request_id)
        object_ids.append(request.bookkeeping_object_id)
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("run.requests: duplicate request identity")
    if len(object_ids) != len(set(object_ids)):
        raise ValueError("run.requests: duplicate bookkeeping object identity")


def _routing_reference_json(value: RoutingReference) -> dict[str, str]:
    return {
        "schema": value.schema,
        "trace_schema": value.trace_schema,
        "trace_sha256": value.trace_sha256,
        "request_id": value.request_id,
    }


def _routing_reference_from_json(value: object, path: str) -> RoutingReference:
    payload = _object(value, path)
    _fields(
        payload,
        path,
        required={"schema", "trace_schema", "trace_sha256", "request_id"},
    )
    reference = RoutingReference(
        schema=_string(payload["schema"], f"{path}.schema"),
        trace_schema=_string(payload["trace_schema"], f"{path}.trace_schema"),
        trace_sha256=_string(payload["trace_sha256"], f"{path}.trace_sha256"),
        request_id=_string(payload["request_id"], f"{path}.request_id"),
    )
    _validate_routing_reference(reference, path)
    return reference


def preplay_replay_run_to_json(value: PreplayReplayRun) -> dict[str, Any]:
    """Return the strict JSON object for a joined replay run."""

    validate_preplay_replay_run(value)
    return {
        "schema": value.schema,
        "trace": {
            "schema": value.trace.schema,
            "path": value.trace.path,
            "sha256": value.trace.sha256,
        },
        "requests": [
            {
                "request_id": request.request_id,
                "arrived_at_ps": request.arrived_at_ps,
                "output_length": request.output_length,
                "stop_reason": request.stop_reason.value,
                "output_token_ids": list(request.output_token_ids),
                "routing_reference": _routing_reference_json(
                    request.routing_reference
                ),
                "bookkeeping_object_id": request.bookkeeping_object_id,
            }
            for request in value.requests
        ],
    }


def preplay_replay_run_from_json(value: object) -> PreplayReplayRun:
    """Parse and validate a strict joined replay-run object."""

    payload = _object(value, "run")
    _fields(payload, "run", required={"schema", "trace", "requests"})
    trace_payload = _object(payload["trace"], "run.trace")
    _fields(
        trace_payload,
        "run.trace",
        required={"schema", "path", "sha256"},
    )
    trace = TraceArtifactReference(
        schema=_string(trace_payload["schema"], "run.trace.schema"),
        path=_string(trace_payload["path"], "run.trace.path"),
        sha256=_string(trace_payload["sha256"], "run.trace.sha256"),
    )
    requests_value = payload["requests"]
    if not isinstance(requests_value, list):
        raise TypeError("run.requests: expected an array")
    requests: list[JoinedRequest] = []
    for index, item in enumerate(requests_value):
        path = f"run.requests[{index}]"
        request_payload = _object(item, path)
        _fields(
            request_payload,
            path,
            required={
                "request_id",
                "arrived_at_ps",
                "output_length",
                "stop_reason",
                "output_token_ids",
                "routing_reference",
                "bookkeeping_object_id",
            },
        )
        token_values = request_payload["output_token_ids"]
        if not isinstance(token_values, list):
            raise TypeError(f"{path}.output_token_ids: expected an array")
        try:
            stop_reason = StopReason(
                _string(request_payload["stop_reason"], f"{path}.stop_reason")
            )
        except ValueError as exc:
            raise ValueError(f"{path}.stop_reason: unsupported value") from exc
        requests.append(
            JoinedRequest(
                request_id=_string(
                    request_payload["request_id"], f"{path}.request_id"
                ),
                arrived_at_ps=_integer(
                    request_payload["arrived_at_ps"], f"{path}.arrived_at_ps"
                ),
                output_length=_integer(
                    request_payload["output_length"], f"{path}.output_length"
                ),
                stop_reason=stop_reason,
                output_token_ids=tuple(
                    _integer(token_id, f"{path}.output_token_ids[{token_index}]")
                    for token_index, token_id in enumerate(token_values)
                ),
                routing_reference=_routing_reference_from_json(
                    request_payload["routing_reference"],
                    f"{path}.routing_reference",
                ),
                bookkeeping_object_id=_string(
                    request_payload["bookkeeping_object_id"],
                    f"{path}.bookkeeping_object_id",
                ),
            )
        )
    run = PreplayReplayRun(
        schema=_string(payload["schema"], "run.schema"),
        trace=trace,
        requests=tuple(requests),
    )
    validate_preplay_replay_run(run)
    return run


def write_preplay_replay_run(
    value: PreplayReplayRun,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write one canonical joined run record with protective creation."""

    payload = preplay_replay_run_to_json(value)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    with target.open(mode, encoding="utf-8", newline="\n") as stream:
        json.dump(
            payload,
            stream,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        stream.write("\n")
    return target


def read_preplay_replay_run(path: str | Path) -> PreplayReplayRun:
    """Read one strict joined run record."""

    source = Path(path)
    with source.open("r", encoding="utf-8") as stream:
        try:
            payload = json.load(stream)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{source}: invalid replay-run JSON: {exc}") from exc
    return preplay_replay_run_from_json(payload)


def _bookkeeping_object_id(request_id: str) -> str:
    digest = hashlib.sha256(request_id.encode("utf-8")).hexdigest()
    return f"preplay-request-v1:{digest}"


def _canonical_json(value: Mapping[str, object]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _bookkeeping_record(
    request: JoinedRequest,
    trace: TraceArtifactReference,
) -> CreatedObjectRecord:
    routing_json = _canonical_json(_routing_reference_json(request.routing_reference))
    tokens_json = json.dumps(
        list(request.output_token_ids),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    return CreatedObjectRecord(
        ref=CreatedObjectRef(
            kind=CreatedObjectKind.FRAMEWORK_REQUEST,
            object_id=request.bookkeeping_object_id,
        ),
        owner=ObjectOwner.FRAMEWORK,
        created_at_ps=request.arrived_at_ps,
        scope=BookkeepingScope(
            correlation=OperationCorrelation(request_ids=(request.request_id,)),
        ),
        native_id=request.request_id,
        metadata=(
            ("preplay_run_schema", PREPLAY_REPLAY_RUN_SCHEMA),
            ("preplay_trace_schema", trace.schema),
            ("preplay_trace_path", trace.path),
            ("preplay_trace_sha256", trace.sha256),
            ("preplay_arrived_at_ps", request.arrived_at_ps),
            ("preplay_output_length", request.output_length),
            ("preplay_stop_reason", request.stop_reason.value),
            ("preplay_output_token_ids", tokens_json),
            ("preplay_routing_reference", routing_json),
        ),
    )


def join_preplay_arrivals(
    arrivals: Iterable[RequestArrival],
    trace_path: str | Path,
    bookkeeper: RequestBookkeeper,
    *,
    routing_arena_index_path: str | Path | None = None,
) -> PreplayReplayRun:
    """Join realized arrivals to a trace and append all records atomically.

    The trace is parsed and hashed, then every arrival and join result is
    validated before the bookkeeper receives one atomic ``extend`` call.
    """

    if not isinstance(bookkeeper, RequestBookkeeper):
        raise TypeError("bookkeeper: expected RequestBookkeeper")
    source = Path(trace_path)
    trace_bytes = source.read_bytes()
    trace = read_preplay_trace(source)
    trace_reference = TraceArtifactReference(
        path=str(source.resolve()),
        sha256=hashlib.sha256(trace_bytes).hexdigest(),
    )
    _validate_trace_reference(trace_reference, "trace")

    realized = tuple(arrivals)
    if not realized:
        raise ValueError("arrivals: must not be empty")
    arrival_ids: list[str] = []
    for index, arrival in enumerate(realized):
        _validate_arrival(arrival, f"arrivals[{index}]")
        arrival_ids.append(arrival.request_id)
    if len(arrival_ids) != len(set(arrival_ids)):
        raise ValueError("arrivals: duplicate request identity")

    trace_requests = {request.request_id: request for request in trace.requests}
    missing = [request_id for request_id in arrival_ids if request_id not in trace_requests]
    if missing:
        raise ValueError(
            "arrivals: request IDs missing from pre-play trace: "
            + ", ".join(repr(request_id) for request_id in missing)
        )

    joined: list[JoinedRequest] = []
    for arrival in realized:
        request_trace = trace_requests[arrival.request_id]
        joined.append(
            JoinedRequest(
                request_id=arrival.request_id,
                arrived_at_ps=arrival.arrived_at_ps,
                output_length=len(request_trace.output_token_ids),
                stop_reason=request_trace.stop_reason,
                output_token_ids=request_trace.output_token_ids,
                routing_reference=RoutingReference(
                    trace_sha256=trace_reference.sha256,
                    request_id=arrival.request_id,
                ),
                bookkeeping_object_id=_bookkeeping_object_id(arrival.request_id),
            )
        )
    run = PreplayReplayRun(trace=trace_reference, requests=tuple(joined))
    validate_preplay_replay_run(run)
    facts = tuple(
        _bookkeeping_record(request, trace_reference) for request in run.requests
    )
    published_arena_paths: tuple[Path, Path] | None = None
    if routing_arena_index_path is not None:
        # Validate the ledger transaction before publishing either sidecar. The
        # real append sees the same immutable snapshot after the arena build.
        RequestBookkeeper(bookkeeper.snapshot()).extend(facts)
        from simllm.preplay.arena import build_routing_arena

        arena = build_routing_arena(run, routing_arena_index_path)
        published_arena_paths = (arena.index_path, arena.payload_path)
        arena.close()
    try:
        bookkeeper.extend(facts)
    except BaseException:
        if published_arena_paths is not None:
            for published_path in published_arena_paths:
                published_path.unlink(missing_ok=True)
        raise
    return run
