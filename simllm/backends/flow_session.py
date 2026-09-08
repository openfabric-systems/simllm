"""Strict framed client for one native RNIC flow-session authority."""

from __future__ import annotations

import json
import math
import re
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from ._child_process import OwnedBinaryProcess, OwnedBinaryReadError

_SCHEMA = "simllm-htsim-flow-session-v1"
_AUTHORITY = "simllm-native-rnic-session"
_FRAME_LIMIT = 1 << 20
_U64 = (1 << 64) - 1
_U32 = (1 << 32) - 1
_COMMON = {"schema", "status", "verb"}
_COUNTERS = {"legacy_aborts", "legacy_ledger_constructed", "legacy_mutations",
             "legacy_posts", "native_posts", "native_session_constructed"}
_IDENTITY = {"sequence", "execution_id", "operation_id", "flow_id", "source",
             "destination", "tag", "payload_bytes", "native_flow_id"}
_ALIASES = {"wqe_id", "sq_id", "sq_post_sequence", "cq_id", "transport_kind",
            "transport_object_id"}
_CQ = {"cq_post_sequence", "cq_consume_sequence"}
_EVENT_FIELDS = _IDENTITY | _ALIASES | _CQ | {
    "kind", "timestamp_ps", "policy_context_token"}
_ROW_FIELDS = _IDENTITY | _ALIASES | _CQ | {
    "sq_dispatch_sequence", "start_time_ps", "completion_time_ps", "fct_ps"}
_PHASES = ("accepted", "queued", "started", "completed")
_UPDATE_FIELDS = {
    "reason", "event_time_ps", "fully_processed_horizon_ps",
    "ordinary_injection_floor_ps", "events_executed", "events", "completion_rows",
    "last_accepted_sequence", "authority_counters", "quiescent", "boundary_time_ps",
    "boundary_id"}


class FlowSessionError(RuntimeError):
    """A terminal transport, protocol, or native-authority failure."""


def _uint(value: Any, name: str, minimum: int = 0, maximum: int = _U64) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise FlowSessionError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _string(value: Any, name: str) -> str:
    if type(value) is not str or not value:
        raise FlowSessionError(f"{name} must be a nonempty string")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise FlowSessionError(f"{name} is not valid UTF-8") from error
    return value


def _object(value: Any, fields: set[str], name: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != fields:
        raise FlowSessionError(f"{name} has missing, unknown, or invalid fields")
    return value


def _array(value: Any, name: str) -> list[Any]:
    if type(value) is not list:
        raise FlowSessionError(f"{name} must be an array")
    return value


def _boolean(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise FlowSessionError(f"{name} must be boolean")
    return value


def _nullable_uint(value: Any, name: str, minimum: int = 0) -> int | None:
    return None if value is None else _uint(value, name, minimum)


def _sequences(values: Sequence[int], name: str) -> tuple[int, ...]:
    result = tuple(_uint(value, name, 1, _U32) for value in values)
    if tuple(sorted(set(result))) != result:
        raise FlowSessionError(f"{name} must be strictly increasing")
    return result


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FlowSessionError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_number(value: str) -> None:
    raise FlowSessionError(f"noninteger JSON number {value}")


@dataclass(frozen=True)
class FlowSessionConfig:
    profile: str
    node_count: int
    link_rate_bps: int
    effective_hardware_sha256: str
    policy_context_token: int
    seed: int = 1
    wall_timeout_s: float = 60.0
    max_events: int = 1_000_000
    simulation_budget_ps: int = 10_000_000_000

    def __post_init__(self) -> None:
        if self.profile not in {"rnic-nn", "rnic-cn"}:
            raise ValueError("flow sessions require rnic-nn or rnic-cn")
        try:
            _uint(self.node_count, "node_count", 2, (1 << 31) - 1)
            _uint(self.link_rate_bps, "link_rate_bps", 1)
            _uint(self.seed, "seed")
            _uint(self.policy_context_token, "policy_context_token", 1)
            _uint(self.max_events, "max_events", 1)
            _uint(self.simulation_budget_ps, "simulation_budget_ps", 1)
        except FlowSessionError as error:
            raise ValueError(str(error)) from error
        if (type(self.effective_hardware_sha256) is not str or
                re.fullmatch(r"[0-9a-f]{64}", self.effective_hardware_sha256) is None):
            raise ValueError("effective_hardware_sha256 must be supplied as lowercase SHA-256")
        if (isinstance(self.wall_timeout_s, bool) or
                not isinstance(self.wall_timeout_s, (int, float)) or
                not math.isfinite(self.wall_timeout_s) or self.wall_timeout_s <= 0):
            raise ValueError("wall_timeout_s must be finite and positive")
        if self.profile == "rnic-cn":
            radix = math.isqrt(2 * self.node_count)
            if radix % 2 or radix * radix != 2 * self.node_count:
                raise ValueError("generated CN node count must equal K^2/2 for even K")


@dataclass(frozen=True)
class FlowSessionUpdate:
    reason: str
    event_time_ps: int
    fully_processed_horizon_ps: int | None
    ordinary_injection_floor_ps: int | None
    events_executed: int
    events: tuple[Mapping[str, Any], ...]
    completion_rows: tuple[Mapping[str, Any], ...]
    last_accepted_sequence: int
    authority_counters: Mapping[str, int]
    quiescent: bool
    boundary_time_ps: int | None
    boundary_id: int | None


@dataclass(frozen=True)
class FlowSessionDrain:
    quiesced_at_ps: int
    completion_rows: tuple[Mapping[str, Any], ...]
    authority_counters: Mapping[str, int]
    sq_high_watermarks: tuple[int, ...]


class FlowSession:
    """One framed authority; all exposed rows are validated read-only projections."""

    def __init__(
        self, config: FlowSessionConfig, command: Sequence[str], *, session_id: str,
        time_origin_ps: int = 0, environment: Mapping[str, str] | None = None,
    ) -> None:
        self.config = config
        self.command = tuple(str(item) for item in command)
        if not self.command:
            raise ValueError("flow session command must not be empty")
        self.session_id = _string(session_id, "session_id")
        self.time_origin_ps = _uint(time_origin_ps, "time_origin_ps")
        self.max_time_ps = _uint(time_origin_ps + config.simulation_budget_ps, "max_time_ps", 1)
        self._environment = None if environment is None else dict(environment)
        self._stream: OwnedBinaryProcess | None = None
        self._opened = False
        self._closed = False
        self._poisoned = False
        self._injections: dict[int, dict[str, Any]] = {}
        self._identities: set[tuple[str, str, str]] = set()
        self._events: list[Mapping[str, Any]] = []
        self._rows: dict[int, Mapping[str, Any]] = {}
        self._phases: dict[int, list[Mapping[str, Any]]] = {}
        self._aliases: dict[tuple[int, int], int] = {}
        self._queue_aliases: dict[tuple[str, int, int, int], int] = {}
        self._transcript: list[tuple[str, bytes]] = []
        self._event_time_ps = 0
        self._horizon: int | None = None
        self._floor: int | None = None
        self._boundary_id: int | None = None
        self._boundary_time: int | None = None
        self._last_boundary_id = 0
        self._quiesced_at: int | None = None
        self._drain: FlowSessionDrain | None = None

    @property
    def last_accepted_sequence(self) -> int:
        return len(self._injections)

    @property
    def pending_sequences(self) -> tuple[int, ...]:
        return tuple(sequence for sequence in self._injections if sequence not in self._rows)

    @property
    def completed_sequences(self) -> tuple[int, ...]:
        return tuple(sorted(self._rows))

    @property
    def boundary_id(self) -> int | None:
        return self._boundary_id

    @property
    def boundary_time_ps(self) -> int | None:
        return self._boundary_time

    @property
    def fully_processed_horizon_ps(self) -> int | None:
        return self._horizon

    @property
    def ordinary_injection_floor_ps(self) -> int | None:
        return self._floor

    @property
    def event_time_ps(self) -> int:
        return self._event_time_ps

    @property
    def transcript(self) -> tuple[tuple[str, bytes], ...]:
        return tuple(self._transcript)

    @property
    def events(self) -> tuple[Mapping[str, Any], ...]:
        if self._poisoned:
            raise FlowSessionError("poisoned session has no publishable lifecycle evidence")
        return tuple(self._events)

    @property
    def completion_rows(self) -> tuple[Mapping[str, Any], ...]:
        if self._poisoned:
            raise FlowSessionError("poisoned session has no publishable completion rows")
        return tuple(self._rows[sequence] for sequence in sorted(self._rows))

    @property
    def stderr(self) -> bytes:
        return b"" if self._stream is None else self._stream.stderr

    @property
    def poisoned(self) -> bool:
        return self._poisoned

    def _guard(self) -> None:
        if self._poisoned or self._closed:
            raise FlowSessionError("flow session is terminal")

    def abort(self) -> None:
        self._poisoned = True
        if self._stream is not None:
            self._stream.abort()

    def _failure(self, error: BaseException) -> FlowSessionError:
        self.abort()
        return FlowSessionError(str(error))

    def _exchange(self, verb: str, fields: dict[str, Any], expected: set[str]) -> dict[str, Any]:
        self._guard()
        if self._stream is None:
            raise FlowSessionError("flow session has not been opened")
        request = _canonical({"schema": _SCHEMA, "verb": verb, **fields})
        if not 0 < len(request) <= _FRAME_LIMIT:
            raise FlowSessionError("request exceeds the frame limit")
        frame = struct.pack(">I", len(request)) + request
        self._transcript.append(("request", frame))
        self._stream.write(frame)
        try:
            header = self._stream.read_exact(4)
        except OwnedBinaryReadError as error:
            self._transcript.append(("response", error.partial))
            raise
        size = struct.unpack(">I", header)[0]
        if not 0 < size <= _FRAME_LIMIT:
            self._transcript.append(("response", header))
            raise FlowSessionError("response has invalid frame length")
        try:
            body = self._stream.read_exact(size)
        except OwnedBinaryReadError as error:
            self._transcript.append(("response", header + error.partial))
            raise
        self._transcript.append(("response", header + body))
        response = json.loads(body.decode("utf-8"), object_pairs_hook=_pairs,
                              parse_float=_reject_number, parse_constant=_reject_number)
        if _canonical(response) != body:
            raise FlowSessionError("response is not canonical JSON")
        if type(response) is not dict or response.get("schema") != _SCHEMA:
            raise FlowSessionError("response schema is invalid")
        if response.get("status") == "error":
            _object(response, _COMMON | {"error", "terminal", "authority_counters"}, "error")
            detail = _object(response["error"], {"code", "message"}, "error detail")
            if response["verb"] != "error" or response["terminal"] is not True:
                raise FlowSessionError("invalid terminal error response")
            self._counters(response["authority_counters"])
            raise FlowSessionError(f"native {_string(detail['code'], 'code')}: "
                                   f"{_string(detail['message'], 'message')}")
        _object(response, _COMMON | expected, verb)
        if response["status"] != "ok" or response["verb"] != verb:
            raise FlowSessionError("response status or verb disagrees with the request")
        return response

    def open(self) -> FlowSession:
        self._guard()
        if self._opened:
            return self
        try:
            cfg = self.config
            fields = {
                "session_id": self.session_id, "profile": cfg.profile,
                "node_count": cfg.node_count, "link_rate_bps": cfg.link_rate_bps,
                "seed": cfg.seed, "topology_identity": f"{cfg.profile}:nodes={cfg.node_count}",
                "wqe_authority": _AUTHORITY,
                "effective_hardware_sha256": cfg.effective_hardware_sha256,
            }
            self._stream = OwnedBinaryProcess(self.command, timeout_s=cfg.wall_timeout_s,
                                              environment=self._environment)
            response = self._exchange("open", fields, set(fields) | {"sequence"})
            for name, value in fields.items():
                if type(response[name]) is not type(value) or response[name] != value:
                    raise FlowSessionError(f"open response changed {name}")
            if _uint(response["sequence"], "sequence") != 0:
                raise FlowSessionError("open response has a nonzero cursor")
            self._opened = True
            return self
        except BaseException as error:
            if not isinstance(error, Exception):
                self.abort()
                raise
            raise self._failure(error) from error

    def inject(
        self, *, execution_id: str, operation_id: str, flow_id: str, source: int,
        destination: int, tag: int, payload_bytes: int, eligible_at_ps: int,
        predecessor_sequences: Sequence[int] = (),
    ) -> int:
        self._guard()
        try:
            if not self._opened:
                raise FlowSessionError("flow session has not been opened")
            sequence = _uint(self.last_accepted_sequence + 1, "sequence", 1, _U32)
            fields = {
                "sequence": sequence,
                "execution_id": _string(execution_id, "execution_id"),
                "operation_id": _string(operation_id, "operation_id"),
                "flow_id": _string(flow_id, "flow_id"),
                "source": _uint(source, "source", 0, self.config.node_count - 1),
                "destination": _uint(destination, "destination", 0, self.config.node_count - 1),
                "tag": _uint(tag, "tag", 0, _U32),
                "payload_bytes": _uint(payload_bytes, "payload_bytes", 1),
                "eligible_at_ps": _uint(eligible_at_ps, "eligible_at_ps"),
                "policy_context_token": self.config.policy_context_token,
            }
            identity = (execution_id, operation_id, flow_id)
            if source == destination or identity in self._identities:
                raise FlowSessionError("self flow or duplicate flow identity")
            if not self.time_origin_ps <= eligible_at_ps <= self.max_time_ps:
                raise FlowSessionError("injection eligibility exceeds the session budget")
            predecessors = _sequences(predecessor_sequences, "predecessor_sequences")
            if any(value not in self._rows for value in predecessors):
                raise FlowSessionError("injection predecessor is not successfully completed")
            boundary = bool(predecessors) and eligible_at_ps == self._boundary_time
            if self._floor is not None and eligible_at_ps <= self._floor and not boundary:
                raise FlowSessionError("ordinary injection is inside the exclusion floor")
            expected = {"accepted_sequence", "eligible_at_ps"}
            request = dict(fields)
            verb = "inject"
            if boundary:
                if self._boundary_id is None:
                    raise FlowSessionError("injection has no open boundary")
                verb = "inject_at_boundary"
                request.update(boundary_id=self._boundary_id,
                               predecessor_sequences=list(predecessors))
                expected.add("boundary_id")
            response = self._exchange(verb, request, expected)
            if (_uint(response["accepted_sequence"], "accepted_sequence") != sequence or
                    _uint(response["eligible_at_ps"], "eligible_at_ps") != eligible_at_ps):
                raise FlowSessionError("injection acknowledgement changed its cursor or time")
            if boundary and _uint(response["boundary_id"], "boundary_id", 1) != self._boundary_id:
                raise FlowSessionError("injection acknowledgement changed its boundary")
            self._injections[sequence] = fields
            self._identities.add(identity)
            self._quiesced_at = None
            return sequence
        except BaseException as error:
            if not isinstance(error, Exception):
                self.abort()
                raise
            raise self._failure(error) from error

    def _counters(self, value: Any) -> Mapping[str, int]:
        counters = _object(value, _COUNTERS, "authority counters")
        for name in _COUNTERS:
            _uint(counters[name], name)
        if any(counters[name] for name in _COUNTERS if name.startswith("legacy_")):
            raise FlowSessionError("legacy authority mutated in a native flow session")
        if counters["native_session_constructed"] != 1:
            raise FlowSessionError("session does not have exactly one native authority")
        if counters["native_posts"] > self.last_accepted_sequence:
            raise FlowSessionError("native posts exceed accepted injections")
        return MappingProxyType(dict(counters))

    def _identity(self, value: dict[str, Any]) -> int:
        sequence = _uint(value["sequence"], "sequence", 1, _U32)
        if sequence not in self._injections:
            raise FlowSessionError("native observation names an unaccepted sequence")
        injection = self._injections[sequence]
        for name in _IDENTITY - {"native_flow_id"}:
            if type(value[name]) is not type(injection[name]) or value[name] != injection[name]:
                raise FlowSessionError(f"native observation changed {name}")
        native_id = (injection["source"] << 32) | (sequence - 1)
        if _uint(value["native_flow_id"], "native_flow_id") != native_id:
            raise FlowSessionError("native flow identity disagrees with its source and sequence")
        expected_kind = "none" if self.config.profile == "rnic-nn" else "rnic-cn-link-pair"
        if value["transport_kind"] != expected_kind:
            raise FlowSessionError("native transport kind disagrees with profile")
        for name in _ALIASES - {"transport_kind"}:
            _uint(value[name], name)
        return sequence

    def _observations(self, response: dict[str, Any], now: int) -> tuple[
        tuple[Mapping[str, Any], ...], tuple[Mapping[str, Any], ...]
    ]:
        events = []
        ordering = []
        for raw in _array(response["events"], "events"):
            event = _object(raw, _EVENT_FIELDS, "lifecycle event")
            sequence = self._identity(event)
            phase = event["kind"]
            history = self._phases.setdefault(sequence, [])
            if len(history) >= len(_PHASES) or phase != _PHASES[len(history)]:
                raise FlowSessionError("lifecycle event is missing, duplicated, or out of order")
            timestamp = _uint(event["timestamp_ps"], "timestamp_ps")
            if not self._injections[sequence]["eligible_at_ps"] <= timestamp <= now:
                raise FlowSessionError("lifecycle timestamp is outside eligibility and response time")
            if phase == "accepted" and timestamp != self._injections[sequence]["eligible_at_ps"]:
                raise FlowSessionError("native submission did not occur at accepted eligibility")
            if history and timestamp < history[-1]["timestamp_ps"]:
                raise FlowSessionError("lifecycle time moved backward")
            if _uint(event["policy_context_token"], "policy_context_token", 1) != self.config.policy_context_token:
                raise FlowSessionError("lifecycle policy identity disagrees")
            for name in _CQ:
                if phase == "completed":
                    _uint(event[name], name)
                elif event[name] is not None:
                    raise FlowSessionError("unfinished lifecycle carries completion-queue aliases")
            if history and any(event[name] != history[0][name] for name in _ALIASES):
                raise FlowSessionError("native object aliases changed during its lifecycle")
            key = (event["source"], event["wqe_id"])
            if key in self._aliases and self._aliases[key] != sequence:
                raise FlowSessionError("native WQE aliases distinct flow identities")
            self._aliases[key] = sequence
            self._queue_alias(event, "sq_post_sequence", "sq_id", sequence)
            if phase == "completed":
                for name in _CQ:
                    self._queue_alias(event, name, "cq_id", sequence)
            frozen = MappingProxyType(dict(event))
            history.append(frozen)
            events.append(frozen)
            ordering.append((timestamp, sequence, _PHASES.index(phase)))
        if ordering != sorted(ordering):
            raise FlowSessionError("lifecycle response ordering is not canonical")
        rows = []
        previous = 0
        for raw in _array(response["completion_rows"], "completion_rows"):
            row = _object(raw, _ROW_FIELDS | {"completion_status"}, "completion row")
            sequence = self._identity(row)
            if sequence <= previous or sequence in self._rows:
                raise FlowSessionError("completion sequence is duplicated or out of order")
            previous = sequence
            if row["completion_status"] != "success":
                raise FlowSessionError(f"native completion failed: {row['completion_status']!r}")
            for name in _CQ | {"sq_dispatch_sequence", "start_time_ps", "completion_time_ps", "fct_ps"}:
                _uint(row[name], name)
            history = self._phases.get(sequence, [])
            if len(history) != 4:
                raise FlowSessionError("completion row has no complete lifecycle")
            if any(row[name] != history[-1][name] for name in _ALIASES | _CQ):
                raise FlowSessionError("completion row changed native lifecycle aliases")
            if (row["start_time_ps"] != self._injections[sequence]["eligible_at_ps"] or
                    row["completion_time_ps"] != history[3]["timestamp_ps"] or
                    row["fct_ps"] != row["completion_time_ps"] - row["start_time_ps"]):
                raise FlowSessionError("completion timestamps disagree with native lifecycle")
            self._queue_alias(row, "sq_dispatch_sequence", "sq_id", sequence)
            frozen = MappingProxyType(dict(row))
            self._rows[sequence] = frozen
            rows.append(frozen)
        if any(len(history) == 4 and sequence not in self._rows
               for sequence, history in self._phases.items()):
            raise FlowSessionError("completed lifecycle omitted its completion row")
        self._events.extend(events)
        return tuple(events), tuple(rows)

    def _queue_alias(self, row: dict[str, Any], sequence_name: str,
                     queue_name: str, sequence: int) -> None:
        key = (sequence_name, row["source"], row[queue_name], row[sequence_name])
        if key in self._queue_aliases and self._queue_aliases[key] != sequence:
            raise FlowSessionError("native queue sequence aliases distinct flow identities")
        self._queue_aliases[key] = sequence

    def await_completion(
        self, sequences: Sequence[int] | None = None, *, through_ps: int | None = None,
    ) -> FlowSessionUpdate:
        return self._await(self.pending_sequences if sequences is None else sequences,
                           through_ps=through_ps, until_quiescent=False)

    def await_quiescence(self) -> FlowSessionUpdate:
        return self._await((), through_ps=None, until_quiescent=True)

    def _await(self, sequences: Sequence[int], *, through_ps: int | None,
               until_quiescent: bool) -> FlowSessionUpdate:
        self._guard()
        try:
            targets = _sequences(sequences, "completion_sequences")
            if (not targets and not until_quiescent) or any(
                    value not in self.pending_sequences for value in targets):
                raise FlowSessionError("await requires pending accepted targets")
            through_ps = _nullable_uint(through_ps, "through_ps")
            if through_ps is not None and (through_ps > self.max_time_ps or
                    (self._floor is not None and through_ps < self._floor)):
                raise FlowSessionError("scheduling horizon moved backward or exceeds budget")
            response = self._exchange("await_completion", {
                "through_sequence": self.last_accepted_sequence,
                "completion_sequences": list(targets), "until_quiescent": until_quiescent,
                "through_ps": through_ps, "max_time_ps": self.max_time_ps,
                "max_events": self.config.max_events,
            }, _UPDATE_FIELDS)
            now = _uint(response["event_time_ps"], "event_time_ps")
            if not self._event_time_ps <= now <= self.max_time_ps:
                raise FlowSessionError("event time moved backward or exceeds budget")
            cursor = _uint(response["last_accepted_sequence"], "last_accepted_sequence")
            if cursor != self.last_accepted_sequence:
                raise FlowSessionError("await response changed accepted cursor")
            executed = _uint(response["events_executed"], "events_executed", 0, self.config.max_events)
            horizon = _nullable_uint(response["fully_processed_horizon_ps"], "horizon")
            floor = _nullable_uint(response["ordinary_injection_floor_ps"], "floor")
            if floor is not None and floor > self.max_time_ps:
                raise FlowSessionError("response frontier exceeds simulated time budget")
            for previous, current in ((self._horizon, horizon), (self._floor, floor)):
                if previous is not None and (current is None or current < previous):
                    raise FlowSessionError("response frontier moved backward")
            if horizon is not None and (floor is None or horizon > floor):
                raise FlowSessionError("processed horizon exceeds the injection exclusion floor")
            boundary_time = _nullable_uint(response["boundary_time_ps"], "boundary_time_ps")
            boundary_id = _nullable_uint(response["boundary_id"], "boundary_id", 1)
            quiescent = _boolean(response["quiescent"], "quiescent")
            reason = response["reason"]
            if reason == "completion":
                if until_quiescent or executed == 0 or boundary_time != now or boundary_id is None:
                    raise FlowSessionError("completion response has an invalid boundary")
                if boundary_id <= self._last_boundary_id or floor != now or horizon != (now - 1 if now else None):
                    raise FlowSessionError("completion response has invalid frontier or token order")
                if through_ps is not None and now > through_ps:
                    raise FlowSessionError("completion overshot the scheduling horizon")
            else:
                if boundary_time is not None or boundary_id is not None:
                    raise FlowSessionError("non-completion response opened a boundary")
                if reason == "horizon":
                    if through_ps is None or horizon != through_ps or floor != through_ps or now > through_ps:
                        raise FlowSessionError("horizon response disagrees with requested horizon")
                elif reason != "quiescence" or not until_quiescent or not quiescent:
                    raise FlowSessionError("await response has an invalid reason")
            counters = self._counters(response["authority_counters"])
            events, rows = self._observations(response, now)
            if counters["native_posts"] != len(self._phases):
                raise FlowSessionError("native post counter disagrees with observed lifecycles")
            if horizon is not None and any(
                    request["eligible_at_ps"] <= horizon and sequence not in self._phases
                    for sequence, request in self._injections.items()):
                raise FlowSessionError("processed horizon skipped an accepted injection")
            hit = any(row["sequence"] in targets for row in rows)
            if (reason == "completion") != hit:
                raise FlowSessionError("await reason disagrees with target completion")
            if any(row["sequence"] in targets and row["completion_time_ps"] != now
                   for row in rows):
                raise FlowSessionError("target completion was exposed after its exact boundary")
            if quiescent and (self.pending_sequences or counters["native_posts"] != cursor):
                raise FlowSessionError("quiescence omitted accepted physical work")
            self._event_time_ps, self._horizon, self._floor = now, horizon, floor
            self._boundary_id, self._boundary_time = boundary_id, boundary_time
            if boundary_id is not None:
                self._last_boundary_id = boundary_id
            if quiescent:
                self._quiesced_at = now
            return FlowSessionUpdate(reason, now, horizon, floor, executed, events, rows,
                                     cursor, counters, quiescent, boundary_time, boundary_id)
        except BaseException as error:
            if not isinstance(error, Exception):
                self.abort()
                raise
            raise self._failure(error) from error

    def close(self) -> FlowSessionDrain:
        if self._closed and self._drain is not None:
            return self._drain
        self._guard()
        try:
            self.await_quiescence()
            response = self._exchange("drain", {"through_sequence": self.last_accepted_sequence},
                                      {"authority_counters", "completion_rows", "events",
                                       "last_accepted_sequence", "quiescent", "sq_high_watermarks"})
            counters = self._counters(response["authority_counters"])
            if (_uint(response["last_accepted_sequence"], "last_accepted_sequence") != self.last_accepted_sequence or
                    response["quiescent"] is not True or response["events"] != []):
                raise FlowSessionError("terminal drain changed the verified completion boundary")
            rows = _array(response["completion_rows"], "completion_rows")
            expected = [{key: value for key, value in row.items() if key != "completion_status"}
                        for row in self.completion_rows]
            if len(rows) != len(expected):
                raise FlowSessionError("terminal drain lost a completion row")
            for row, previous in zip(rows, expected, strict=True):
                _object(row, _ROW_FIELDS, "legacy completion row")
                if any(type(row[key]) is not type(value) or row[key] != value
                       for key, value in previous.items()):
                    raise FlowSessionError("terminal drain changed an authoritative completion row")
            high_water = tuple(_uint(value, "sq_high_watermark")
                               for value in _array(response["sq_high_watermarks"], "sq_high_watermarks"))
            if len(high_water) != self.config.node_count:
                raise FlowSessionError("terminal drain omitted endpoint high-water marks")
            for endpoint, peak in enumerate(high_water):
                posts = sum(request["source"] == endpoint for request in self._injections.values())
                if not int(posts > 0) <= peak <= posts:
                    raise FlowSessionError("terminal queue high-water mark disagrees with posts")
            closed = self._exchange("close", {"through_sequence": self.last_accepted_sequence},
                                    {"last_accepted_sequence", "terminal"})
            if (_uint(closed["last_accepted_sequence"], "last_accepted_sequence") != self.last_accepted_sequence or
                    closed["terminal"] is not True):
                raise FlowSessionError("close response changed its terminal cursor")
            assert self._stream is not None and self._quiesced_at is not None
            if self._stream.finish() != 0:
                raise FlowSessionError("native session exited unsuccessfully after close")
            self._drain = FlowSessionDrain(self._quiesced_at, self.completion_rows, counters, high_water)
            self._closed = True
            return self._drain
        except BaseException as error:
            if not isinstance(error, Exception):
                self.abort()
                raise
            raise self._failure(error) from error

    def __enter__(self) -> FlowSession:  # noqa: PYI034 (Python 3.10 compatibility)
        return self.open()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()
