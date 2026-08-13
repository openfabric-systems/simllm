"""Join an observed framework trace v2 into the existing replay identities.

The version 1 arrival join stays exactly as it was. This module adds the
version 2 half of the same seam: a serving framework's own capture pins each
replayed request's arrival, output length, stop reason, output token IDs and
expert routing, and the result is the same ``simllm-preplay-replay-run-v1``
record the vLLM replay seam, the bookkeeping projection and the routed-expert
projection already consume. Only the named trace schema differs.

The capture's KV event stream is deliberately not part of that record. The
framework scheduler that produced the capture was the sole KV authority, and
its events are evidence of what it decided, never an input to a replay
decision. They are projected into a separate read-only
``simllm-preplay-kv-reconciliation-v1`` record, which reports where the event
stream and the capture's own per-request oracle record agree, where they
disagree admissibly, and where they contradict each other.

Reconciliation is defined against one reference: a request with ``P`` prompt
tokens and oracle output length ``L`` executed exactly ``P + L - 1`` forward
passes, which is also the number of dispatch rows the same capture carries.

Admissible disagreements, which are recorded and not treated as defects:

- a prefix hit lets a request allocate fewer than ``P + L - 1`` tokens, short
  by exactly the hit;
- a preempted request allocates more than ``P + L - 1`` tokens, because a
  recomputed request re-allocates slots it already held;
- an eviction that names no request, which is a pool-level decision about
  tokens no live request owns;
- a capture with no KV events at all, which yields an unobserved
  reconciliation and no claim.

Defects, which mean the capture disagrees with its own oracle record:

- the preemption events naming a request disagree with its reported
  ``framework_preemption_count``;
- the prefix-hit token total for a request disagrees with its reported
  ``framework_cached_tokens``;
- a request allocates fewer tokens than its prompt, which cannot run a prefill;
- a joined request has no allocation event at all;
- a request with neither a preemption nor a prefix hit does not allocate
  exactly ``P + L - 1`` tokens;
- replaying the stream drives live occupancy below zero.

A defect is reported, not raised. The KV stream has no authority over the
replay, so an internally inconsistent capture cannot invalidate an output
length the same capture states explicitly. Callers that want to fail closed
read :attr:`FrameworkKvReconciliation.defects`.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from simllm.core import RequestBookkeeper
from simllm.core._wire import _array, _fields, _integer, _object, _optional_string, _string
from simllm.preplay.framework_schema import (
    FRAMEWORK_PREPLAY_TRACE_SCHEMA,
    FrameworkPreplayTrace,
    KvEventKind,
)
from simllm.preplay.framework_trace import read_framework_preplay_trace
from simllm.preplay.join import (
    JoinedRequest,
    PreplayReplayRun,
    RequestArrival,
    RoutingReference,
    TraceArtifactReference,
    _bookkeeping_object_id,
    _bookkeeping_record,
    _validate_arrival,
    _validate_trace_reference,
    peek_trace_schema,
    validate_preplay_replay_run,
)
from simllm.preplay.schema import PREPLAY_TRACE_SCHEMA

FRAMEWORK_KV_RECONCILIATION_SCHEMA = "simllm-preplay-kv-reconciliation-v1"


class KvAgreement(str, Enum):
    """How one request's allocation total relates to its forward passes."""

    #: allocation equals forward passes minus the prefix hit
    EXACT = "exact"
    #: allocation exceeds that, explained by at least one preemption event
    RECOMPUTE_SURPLUS = "recompute-surplus"
    #: allocation differs with nothing in the stream that explains it
    UNEXPLAINED = "unexplained"


class KvDefectCode(str, Enum):
    """Stable identities for a capture that contradicts its own oracle record."""

    MISSING_ALLOCATION = "missing-allocation"
    PREFILL_ALLOCATION_SHORTFALL = "prefill-allocation-shortfall"
    UNEXPLAINED_ALLOCATION_TOTAL = "unexplained-allocation-total"
    PREEMPTION_COUNT_DISAGREEMENT = "preemption-count-disagreement"
    PREFIX_HIT_TOKEN_DISAGREEMENT = "prefix-hit-token-disagreement"
    NEGATIVE_OCCUPANCY = "negative-occupancy"


@dataclass(frozen=True, kw_only=True)
class KvDefect:
    """One reported contradiction between the event stream and the record."""

    code: KvDefectCode
    request_id: str | None
    detail: str


@dataclass(frozen=True, kw_only=True)
class KvRequestReconciliation:
    """Read-only KV accounting for one joined request."""

    request_id: str
    forwarded_token_count: int
    allocated_token_count: int
    released_token_count: int
    evicted_token_count: int
    prefix_hit_token_count: int
    preemption_event_count: int
    declared_cached_tokens: int
    declared_preemption_count: int
    allocation_surplus: int
    agreement: KvAgreement


@dataclass(frozen=True, kw_only=True)
class FrameworkKvReconciliation:
    """Read-only projection of one capture's KV event stream."""

    trace_sha256: str
    kv_page_size: int
    kv_token_capacity: int
    observed: bool
    event_count: int
    unattributed_event_count: int
    unjoined_event_count: int
    peak_live_token_count: int
    final_live_token_count: int
    requests: tuple[KvRequestReconciliation, ...]
    defects: tuple[KvDefect, ...]
    trace_schema: str = FRAMEWORK_PREPLAY_TRACE_SCHEMA
    schema: str = FRAMEWORK_KV_RECONCILIATION_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "requests", tuple(self.requests))
        object.__setattr__(self, "defects", tuple(self.defects))

    def by_request_id(self, request_id: str) -> KvRequestReconciliation:
        """Return one request's accounting or fail with its stable identity."""

        for request in self.requests:
            if request.request_id == request_id:
                return request
        raise KeyError(f"request ID {request_id!r} not in KV reconciliation")


@dataclass(frozen=True, kw_only=True)
class FrameworkReplayJoin:
    """A joined version 2 run plus the read-only KV evidence beside it."""

    run: PreplayReplayRun
    kv: FrameworkKvReconciliation


def validate_framework_kv_reconciliation(value: FrameworkKvReconciliation) -> None:
    """Validate schema identity, counter signs and per-request uniqueness."""

    if not isinstance(value, FrameworkKvReconciliation):
        raise TypeError("reconciliation: expected FrameworkKvReconciliation")
    if value.schema != FRAMEWORK_KV_RECONCILIATION_SCHEMA:
        raise ValueError(
            f"reconciliation.schema: unsupported schema {value.schema!r}"
        )
    if value.trace_schema != FRAMEWORK_PREPLAY_TRACE_SCHEMA:
        raise ValueError(
            f"reconciliation.trace_schema: unsupported schema {value.trace_schema!r}"
        )
    _string(value.trace_sha256, "reconciliation.trace_sha256")
    if len(value.trace_sha256) != 64:
        raise ValueError("reconciliation.trace_sha256: expected a SHA-256 digest")
    _integer(value.kv_page_size, "reconciliation.kv_page_size", minimum=1)
    _integer(value.kv_token_capacity, "reconciliation.kv_token_capacity", minimum=1)
    if not isinstance(value.observed, bool):
        raise TypeError("reconciliation.observed: expected a boolean")
    for name in (
        "event_count",
        "unattributed_event_count",
        "unjoined_event_count",
        "peak_live_token_count",
    ):
        _integer(getattr(value, name), f"reconciliation.{name}", nonnegative=True)
    _integer(value.final_live_token_count, "reconciliation.final_live_token_count")
    if value.observed != (value.event_count > 0):
        raise ValueError(
            "reconciliation.observed: must say whether the capture carried events"
        )
    if not isinstance(value.requests, tuple):
        raise TypeError("reconciliation.requests: in-memory contract requires a tuple")
    request_ids: list[str] = []
    for index, request in enumerate(value.requests):
        path = f"reconciliation.requests[{index}]"
        if not isinstance(request, KvRequestReconciliation):
            raise TypeError(f"{path}: expected KvRequestReconciliation")
        _string(request.request_id, f"{path}.request_id")
        for name in (
            "forwarded_token_count",
            "allocated_token_count",
            "released_token_count",
            "evicted_token_count",
            "prefix_hit_token_count",
            "preemption_event_count",
            "declared_cached_tokens",
            "declared_preemption_count",
        ):
            _integer(getattr(request, name), f"{path}.{name}", nonnegative=True)
        _integer(request.allocation_surplus, f"{path}.allocation_surplus")
        if not isinstance(request.agreement, KvAgreement):
            raise TypeError(f"{path}.agreement: expected KvAgreement")
        request_ids.append(request.request_id)
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("reconciliation.requests: duplicate request identity")
    if not isinstance(value.defects, tuple):
        raise TypeError("reconciliation.defects: in-memory contract requires a tuple")
    for index, defect in enumerate(value.defects):
        path = f"reconciliation.defects[{index}]"
        if not isinstance(defect, KvDefect):
            raise TypeError(f"{path}: expected KvDefect")
        if not isinstance(defect.code, KvDefectCode):
            raise TypeError(f"{path}.code: expected KvDefectCode")
        if defect.request_id is not None:
            _string(defect.request_id, f"{path}.request_id")
        _string(defect.detail, f"{path}.detail")


def framework_kv_reconciliation_to_json(
    value: FrameworkKvReconciliation,
) -> dict[str, Any]:
    """Return the strict JSON object for one KV reconciliation record."""

    validate_framework_kv_reconciliation(value)
    return {
        "schema": value.schema,
        "trace_schema": value.trace_schema,
        "trace_sha256": value.trace_sha256,
        "kv_page_size": value.kv_page_size,
        "kv_token_capacity": value.kv_token_capacity,
        "observed": value.observed,
        "event_count": value.event_count,
        "unattributed_event_count": value.unattributed_event_count,
        "unjoined_event_count": value.unjoined_event_count,
        "peak_live_token_count": value.peak_live_token_count,
        "final_live_token_count": value.final_live_token_count,
        "requests": [
            {
                "request_id": request.request_id,
                "forwarded_token_count": request.forwarded_token_count,
                "allocated_token_count": request.allocated_token_count,
                "released_token_count": request.released_token_count,
                "evicted_token_count": request.evicted_token_count,
                "prefix_hit_token_count": request.prefix_hit_token_count,
                "preemption_event_count": request.preemption_event_count,
                "declared_cached_tokens": request.declared_cached_tokens,
                "declared_preemption_count": request.declared_preemption_count,
                "allocation_surplus": request.allocation_surplus,
                "agreement": request.agreement.value,
            }
            for request in value.requests
        ],
        "defects": [
            {
                "code": defect.code.value,
                "request_id": defect.request_id,
                "detail": defect.detail,
            }
            for defect in value.defects
        ],
    }


def framework_kv_reconciliation_from_json(value: object) -> FrameworkKvReconciliation:
    """Parse and validate one strict KV reconciliation object."""

    payload = _object(value, "reconciliation")
    _fields(
        payload,
        "reconciliation",
        required={
            "schema",
            "trace_schema",
            "trace_sha256",
            "kv_page_size",
            "kv_token_capacity",
            "observed",
            "event_count",
            "unattributed_event_count",
            "unjoined_event_count",
            "peak_live_token_count",
            "final_live_token_count",
            "requests",
            "defects",
        },
    )
    observed = payload["observed"]
    if not isinstance(observed, bool):
        raise TypeError("reconciliation.observed: expected a boolean")
    requests: list[KvRequestReconciliation] = []
    for index, item in enumerate(_array(payload["requests"], "reconciliation.requests")):
        path = f"reconciliation.requests[{index}]"
        request = _object(item, path)
        _fields(
            request,
            path,
            required={
                "request_id",
                "forwarded_token_count",
                "allocated_token_count",
                "released_token_count",
                "evicted_token_count",
                "prefix_hit_token_count",
                "preemption_event_count",
                "declared_cached_tokens",
                "declared_preemption_count",
                "allocation_surplus",
                "agreement",
            },
        )
        try:
            agreement = KvAgreement(_string(request["agreement"], f"{path}.agreement"))
        except ValueError as exc:
            raise ValueError(f"{path}.agreement: unsupported value") from exc
        requests.append(
            KvRequestReconciliation(
                request_id=_string(request["request_id"], f"{path}.request_id"),
                forwarded_token_count=_integer(
                    request["forwarded_token_count"], f"{path}.forwarded_token_count"
                ),
                allocated_token_count=_integer(
                    request["allocated_token_count"], f"{path}.allocated_token_count"
                ),
                released_token_count=_integer(
                    request["released_token_count"], f"{path}.released_token_count"
                ),
                evicted_token_count=_integer(
                    request["evicted_token_count"], f"{path}.evicted_token_count"
                ),
                prefix_hit_token_count=_integer(
                    request["prefix_hit_token_count"], f"{path}.prefix_hit_token_count"
                ),
                preemption_event_count=_integer(
                    request["preemption_event_count"], f"{path}.preemption_event_count"
                ),
                declared_cached_tokens=_integer(
                    request["declared_cached_tokens"], f"{path}.declared_cached_tokens"
                ),
                declared_preemption_count=_integer(
                    request["declared_preemption_count"],
                    f"{path}.declared_preemption_count",
                ),
                allocation_surplus=_integer(
                    request["allocation_surplus"], f"{path}.allocation_surplus"
                ),
                agreement=agreement,
            )
        )
    defects: list[KvDefect] = []
    for index, item in enumerate(_array(payload["defects"], "reconciliation.defects")):
        path = f"reconciliation.defects[{index}]"
        defect = _object(item, path)
        _fields(defect, path, required={"code", "request_id", "detail"})
        try:
            code = KvDefectCode(_string(defect["code"], f"{path}.code"))
        except ValueError as exc:
            raise ValueError(f"{path}.code: unsupported value") from exc
        defects.append(
            KvDefect(
                code=code,
                request_id=_optional_string(defect["request_id"], f"{path}.request_id"),
                detail=_string(defect["detail"], f"{path}.detail"),
            )
        )
    reconciliation = FrameworkKvReconciliation(
        schema=_string(payload["schema"], "reconciliation.schema"),
        trace_schema=_string(payload["trace_schema"], "reconciliation.trace_schema"),
        trace_sha256=_string(payload["trace_sha256"], "reconciliation.trace_sha256"),
        kv_page_size=_integer(payload["kv_page_size"], "reconciliation.kv_page_size"),
        kv_token_capacity=_integer(
            payload["kv_token_capacity"], "reconciliation.kv_token_capacity"
        ),
        observed=observed,
        event_count=_integer(payload["event_count"], "reconciliation.event_count"),
        unattributed_event_count=_integer(
            payload["unattributed_event_count"], "reconciliation.unattributed_event_count"
        ),
        unjoined_event_count=_integer(
            payload["unjoined_event_count"], "reconciliation.unjoined_event_count"
        ),
        peak_live_token_count=_integer(
            payload["peak_live_token_count"], "reconciliation.peak_live_token_count"
        ),
        final_live_token_count=_integer(
            payload["final_live_token_count"], "reconciliation.final_live_token_count"
        ),
        requests=tuple(requests),
        defects=tuple(defects),
    )
    validate_framework_kv_reconciliation(reconciliation)
    return reconciliation


def write_framework_kv_reconciliation(
    value: FrameworkKvReconciliation,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write one canonical KV reconciliation record."""

    payload = framework_kv_reconciliation_to_json(value)
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


def read_framework_kv_reconciliation(path: str | Path) -> FrameworkKvReconciliation:
    """Read one strict KV reconciliation record."""

    source = Path(path)
    with source.open("r", encoding="utf-8") as stream:
        try:
            payload = json.load(stream)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{source}: invalid reconciliation JSON: {exc}") from exc
    return framework_kv_reconciliation_from_json(payload)


def _occupancy(trace: FrameworkPreplayTrace) -> tuple[int, int, list[KvDefect]]:
    """Replay the whole stream in sequence order into live token occupancy."""

    live = 0
    peak = 0
    defects: list[KvDefect] = []
    for event in trace.kv_events:
        if event.kind is KvEventKind.ALLOCATION:
            live += event.token_count
        elif event.kind in (KvEventKind.RELEASE, KvEventKind.EVICTION):
            live -= event.token_count
        if live < 0:
            defects.append(
                KvDefect(
                    code=KvDefectCode.NEGATIVE_OCCUPANCY,
                    request_id=event.request_id,
                    detail=(
                        f"event {event.sequence} drives live occupancy to {live}"
                    ),
                )
            )
            break
        peak = max(peak, live)
    return peak, live, defects


def reconcile_framework_kv_events(
    trace: FrameworkPreplayTrace,
    run: PreplayReplayRun,
) -> FrameworkKvReconciliation:
    """Project one capture's KV events beside the requests a run joined.

    Read only in both directions: nothing here changes the run, and nothing in
    the run changes the capture. See the module docstring for which
    disagreements are admissible and which are reported as defects.
    """

    if not isinstance(trace, FrameworkPreplayTrace):
        raise TypeError("trace: expected FrameworkPreplayTrace")
    validate_preplay_replay_run(run)
    if run.trace.schema != FRAMEWORK_PREPLAY_TRACE_SCHEMA:
        raise ValueError(
            "run.trace.schema: KV reconciliation needs a version 2 joined run"
        )
    trace_requests = {request.request_id: request for request in trace.requests}
    joined_ids = [request.request_id for request in run.requests]
    missing = [request_id for request_id in joined_ids if request_id not in trace_requests]
    if missing:
        raise ValueError(
            "run.requests: joined request IDs absent from the capture: "
            + ", ".join(repr(request_id) for request_id in missing)
        )

    allocated = dict.fromkeys(joined_ids, 0)
    released = dict.fromkeys(joined_ids, 0)
    evicted = dict.fromkeys(joined_ids, 0)
    prefix_hit = dict.fromkeys(joined_ids, 0)
    preemptions = dict.fromkeys(joined_ids, 0)
    allocation_events = dict.fromkeys(joined_ids, 0)
    unattributed = 0
    unjoined = 0
    for event in trace.kv_events:
        request_id = event.request_id
        if request_id is None:
            unattributed += 1
            continue
        if request_id not in allocated:
            unjoined += 1
            continue
        if event.kind is KvEventKind.ALLOCATION:
            allocated[request_id] += event.token_count
            allocation_events[request_id] += 1
        elif event.kind is KvEventKind.RELEASE:
            released[request_id] += event.token_count
        elif event.kind is KvEventKind.EVICTION:
            evicted[request_id] += event.token_count
        elif event.kind is KvEventKind.PREFIX_HIT:
            prefix_hit[request_id] += event.token_count
        elif event.kind is KvEventKind.PREEMPTION:
            preemptions[request_id] += 1

    peak, final, defects = _occupancy(trace)
    reconciled: list[KvRequestReconciliation] = []
    for joined in run.requests:
        request_id = joined.request_id
        source = trace_requests[request_id]
        prompt_length = len(source.input_token_ids)
        forwarded = prompt_length + joined.output_length - 1
        hit = prefix_hit[request_id]
        expected = forwarded - hit
        surplus = allocated[request_id] - expected
        preempted = preemptions[request_id]
        if surplus == 0:
            agreement = KvAgreement.EXACT
        elif surplus > 0 and preempted > 0:
            agreement = KvAgreement.RECOMPUTE_SURPLUS
        else:
            agreement = KvAgreement.UNEXPLAINED
        if trace.kv_events and allocation_events[request_id] == 0:
            defects.append(
                KvDefect(
                    code=KvDefectCode.MISSING_ALLOCATION,
                    request_id=request_id,
                    detail="the capture carries KV events but never allocates for it",
                )
            )
        elif allocation_events[request_id] and allocated[request_id] < prompt_length - hit:
            defects.append(
                KvDefect(
                    code=KvDefectCode.PREFILL_ALLOCATION_SHORTFALL,
                    request_id=request_id,
                    detail=(
                        f"allocated {allocated[request_id]} tokens for a "
                        f"{prompt_length}-token prompt with a {hit}-token prefix hit"
                    ),
                )
            )
        if agreement is KvAgreement.UNEXPLAINED and allocation_events[request_id]:
            defects.append(
                KvDefect(
                    code=KvDefectCode.UNEXPLAINED_ALLOCATION_TOTAL,
                    request_id=request_id,
                    detail=(
                        f"allocated {allocated[request_id]} tokens against "
                        f"{expected} expected from {forwarded} forward passes "
                        f"and a {hit}-token prefix hit, with {preempted} "
                        "preemption events"
                    ),
                )
            )
        if preempted != source.framework_preemption_count:
            defects.append(
                KvDefect(
                    code=KvDefectCode.PREEMPTION_COUNT_DISAGREEMENT,
                    request_id=request_id,
                    detail=(
                        f"{preempted} preemption events against a reported "
                        f"framework_preemption_count of "
                        f"{source.framework_preemption_count}"
                    ),
                )
            )
        if hit != source.framework_cached_tokens:
            defects.append(
                KvDefect(
                    code=KvDefectCode.PREFIX_HIT_TOKEN_DISAGREEMENT,
                    request_id=request_id,
                    detail=(
                        f"{hit} prefix-hit tokens against a reported "
                        f"framework_cached_tokens of {source.framework_cached_tokens}"
                    ),
                )
            )
        reconciled.append(
            KvRequestReconciliation(
                request_id=request_id,
                forwarded_token_count=forwarded,
                allocated_token_count=allocated[request_id],
                released_token_count=released[request_id],
                evicted_token_count=evicted[request_id],
                prefix_hit_token_count=hit,
                preemption_event_count=preempted,
                declared_cached_tokens=source.framework_cached_tokens,
                declared_preemption_count=source.framework_preemption_count,
                allocation_surplus=surplus,
                agreement=agreement,
            )
        )

    reconciliation = FrameworkKvReconciliation(
        trace_sha256=run.trace.sha256,
        kv_page_size=trace.provenance.kv_page_size,
        kv_token_capacity=trace.provenance.kv_token_capacity,
        observed=bool(trace.kv_events),
        event_count=len(trace.kv_events),
        unattributed_event_count=unattributed,
        unjoined_event_count=unjoined,
        peak_live_token_count=peak,
        final_live_token_count=final,
        requests=tuple(reconciled),
        defects=tuple(defects),
    )
    validate_framework_kv_reconciliation(reconciliation)
    return reconciliation


def join_framework_arrivals(
    arrivals: Iterable[RequestArrival],
    trace_path: str | Path,
    bookkeeper: RequestBookkeeper,
) -> FrameworkReplayJoin:
    """Join realized arrivals to a version 2 capture, atomically.

    The capture is parsed and hashed, then every arrival and join result is
    validated before the bookkeeper receives one atomic ``extend`` call, the
    same order the version 1 join uses. The returned run is the ordinary
    replay-run record; the KV reconciliation beside it is evidence only.
    """

    if not isinstance(bookkeeper, RequestBookkeeper):
        raise TypeError("bookkeeper: expected RequestBookkeeper")
    source = Path(trace_path)
    if peek_trace_schema(source) == PREPLAY_TRACE_SCHEMA:
        raise ValueError(
            f"trace_path: {PREPLAY_TRACE_SCHEMA} is joined by "
            "join_preplay_arrivals, not by the framework join"
        )
    trace_bytes = source.read_bytes()
    trace = read_framework_preplay_trace(source)
    trace_reference = TraceArtifactReference(
        path=str(source.resolve()),
        sha256=hashlib.sha256(trace_bytes).hexdigest(),
        schema=FRAMEWORK_PREPLAY_TRACE_SCHEMA,
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
            "arrivals: request IDs missing from framework pre-play trace: "
            + ", ".join(repr(request_id) for request_id in missing)
        )

    joined: list[JoinedRequest] = []
    for arrival in realized:
        observed = trace_requests[arrival.request_id]
        joined.append(
            JoinedRequest(
                request_id=arrival.request_id,
                arrived_at_ps=arrival.arrived_at_ps,
                output_length=observed.output_length,
                stop_reason=observed.stop_reason,
                output_token_ids=observed.output_token_ids,
                routing_reference=RoutingReference(
                    trace_sha256=trace_reference.sha256,
                    request_id=arrival.request_id,
                    trace_schema=FRAMEWORK_PREPLAY_TRACE_SCHEMA,
                ),
                bookkeeping_object_id=_bookkeeping_object_id(arrival.request_id),
            )
        )
    run = PreplayReplayRun(trace=trace_reference, requests=tuple(joined))
    validate_preplay_replay_run(run)
    reconciliation = reconcile_framework_kv_events(trace, run)
    facts = tuple(
        _bookkeeping_record(request, trace_reference) for request in run.requests
    )
    bookkeeper.extend(facts)
    return FrameworkReplayJoin(run=run, kv=reconciliation)
