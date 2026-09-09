"""Framework-neutral timing records for a disaggregated serving session.

The frontend adapters own their schedulers. This module owns only the narrow
join between a completed prefill leg and an admitted decode leg: one KV-cache
handoff event on the session's existing :class:`VirtualClock`, followed by an
exact per-request timing reduction. It imports no serving framework.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from simllm.core.clock import VirtualClock

PD_SESSION_SCHEMA = "simllm-pd-session-result-v1"
KV_HANDOFF_AUTHORITY = "simllm-declared-kv-handoff-v1"
PACKET_KV_HANDOFF_AUTHORITY = "simllm-packet-kv-handoff-v1"
SHARED_PACKET_KV_HANDOFF_AUTHORITY = "simllm-shared-packet-kv-handoff-v1"
KV_HANDOFF_AUTHORITIES = (KV_HANDOFF_AUTHORITY, PACKET_KV_HANDOFF_AUTHORITY)
KV_HANDOFF_ARMS = ("off", "declared-constant", "packet")


def _nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _positive_int(name: str, value: object) -> int:
    result = _nonnegative_int(name, value)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result


class ServingPoolRole(enum.Enum):
    """The role declared by one engine instance in a P/D session."""

    PREFILL = "prefill"
    DECODE = "decode"


@dataclass(frozen=True)
class KvHandoffGeometry:
    """Model geometry needed to size one prompt's key-value cache.

    The byte count covers keys plus values over the complete model. Tensor
    parallelism changes which rank owns each byte, not the aggregate handoff.
    """

    num_layers: int
    num_kv_heads: int
    head_size: int
    element_bytes: int

    def __post_init__(self) -> None:
        for name in ("num_layers", "num_kv_heads", "head_size", "element_bytes"):
            _positive_int(name, getattr(self, name))

    @property
    def bytes_per_token(self) -> int:
        return (
            2
            * self.num_layers
            * self.num_kv_heads
            * self.head_size
            * self.element_bytes
        )

    def bytes_for_prompt(self, prompt_tokens: int) -> int:
        return self.bytes_per_token * _positive_int("prompt_tokens", prompt_tokens)


@dataclass(frozen=True)
class KvHandoffEvent:
    """One queue-contract visit for a prefill-to-decode KV handoff."""

    request_id: str
    kv_bytes: int
    submitted_at_ps: int
    eligible_at_ps: int
    started_at_ps: int
    finished_at_ps: int
    completed_at_ps: int
    pricing_arm: str
    authority: str = KV_HANDOFF_AUTHORITY

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id.strip():
            raise ValueError("request_id must be a nonblank string")
        _positive_int("kv_bytes", self.kv_bytes)
        times = (
            self.submitted_at_ps,
            self.eligible_at_ps,
            self.started_at_ps,
            self.finished_at_ps,
            self.completed_at_ps,
        )
        for name, value in zip(
            (
                "submitted_at_ps",
                "eligible_at_ps",
                "started_at_ps",
                "finished_at_ps",
                "completed_at_ps",
            ),
            times,
            strict=True,
        ):
            _nonnegative_int(name, value)
        if tuple(sorted(times)) != times:
            raise ValueError("KV handoff timestamps must be monotonic")
        if self.pricing_arm not in KV_HANDOFF_ARMS:
            raise ValueError(f"pricing_arm must be one of {KV_HANDOFF_ARMS}")
        if self.authority not in KV_HANDOFF_AUTHORITIES:
            raise ValueError(f"authority must be one of {KV_HANDOFF_AUTHORITIES}")
        if self.queue_wait_ps != 0:
            raise ValueError("the KV handoff has no internal queue")
        if self.visibility_ps != 0:
            raise ValueError("the KV handoff has no visibility tail")
        if self.pricing_arm == "off":
            if self.authority != KV_HANDOFF_AUTHORITY or self.total_ps != 0:
                raise ValueError("the off KV handoff arm must be a declared identity")
        elif self.pricing_arm == "declared-constant":
            if self.authority != KV_HANDOFF_AUTHORITY:
                raise ValueError("the constant arm must retain declared authority")
            if self.submission_ps != 0 or self.service_ps == 0:
                raise ValueError("the declared-constant handoff must be pure service")
        else:
            if self.authority != PACKET_KV_HANDOFF_AUTHORITY:
                raise ValueError("the packet arm must carry packet authority")
            if self.submission_ps == 0 or self.service_ps == 0:
                raise ValueError("the packet arm needs submission and packet service")

    @property
    def submission_ps(self) -> int:
        return self.eligible_at_ps - self.submitted_at_ps

    @property
    def queue_wait_ps(self) -> int:
        return self.started_at_ps - self.eligible_at_ps

    @property
    def service_ps(self) -> int:
        return self.finished_at_ps - self.started_at_ps

    @property
    def visibility_ps(self) -> int:
        return self.completed_at_ps - self.finished_at_ps

    @property
    def total_ps(self) -> int:
        return self.completed_at_ps - self.submitted_at_ps


@runtime_checkable
class KvHandoffPolicy(Protocol):
    """Narrow scheduling surface shared by constant and packet arms."""

    def apply(
        self,
        clock: VirtualClock,
        *,
        request_id: str,
        kv_bytes: int,
    ) -> KvHandoffEvent:
        """Apply one handoff to the caller's sole virtual clock."""

    def schedule(
        self,
        *,
        submitted_at_ps: int,
        request_id: str,
        kv_bytes: int,
    ) -> KvHandoffEvent:
        """Schedule one handoff without advancing a shared clock."""


def _identifier(name: str, value: object) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a nonblank string")
    return value


@dataclass(frozen=True)
class PendingKvHandoff:
    """An accepted shard inventory, with no predicted completion time."""

    request_id: str
    source_engine_id: str
    destination_engine_id: str
    kv_bytes: int
    submitted_at_ps: int
    eligible_at_ps: int
    sequences: tuple[int, ...]

    def __post_init__(self) -> None:
        for name in ("request_id", "source_engine_id", "destination_engine_id"):
            _identifier(name, getattr(self, name))
        _positive_int("kv_bytes", self.kv_bytes)
        _nonnegative_int("submitted_at_ps", self.submitted_at_ps)
        _positive_int("eligible_at_ps", self.eligible_at_ps)
        if self.eligible_at_ps <= self.submitted_at_ps:
            raise ValueError("shared handoff requires positive submission delay")
        if type(self.sequences) is not tuple or not self.sequences:
            raise ValueError("pending handoff requires an immutable sequence inventory")
        for sequence in self.sequences:
            _positive_int("sequence", sequence)
        if self.sequences != tuple(range(self.sequences[0], self.sequences[0] + len(self.sequences))):
            raise ValueError("handoff sequences must be contiguous and ordered")


@dataclass(frozen=True)
class KvHandoffShardCompletion:
    """Complete native flow observations, without an invented release event."""

    row: Mapping[str, object]
    events: tuple[Mapping[str, object], ...]

    def __post_init__(self) -> None:
        def freeze(value: Mapping[str, object]) -> Mapping[str, object]:
            if not isinstance(value, Mapping) or any(
                type(key) is not str or type(item) not in (str, int, type(None))
                for key, item in value.items()
            ):
                raise TypeError("native shard observations must contain flat typed values")
            return MappingProxyType(dict(value))

        row = freeze(self.row)
        events = tuple(freeze(event) for event in self.events)
        if tuple(event.get("kind") for event in events) != ("accepted", "queued", "started", "completed"):
            raise ValueError("shard completion requires its complete native lifecycle")
        for name in ("sequence", "payload_bytes"):
            _positive_int(name, row.get(name))
        for name in ("source", "destination", "start_time_ps", "completion_time_ps", "fct_ps"):
            _nonnegative_int(name, row.get(name))
        for name in ("execution_id", "operation_id", "flow_id"):
            _identifier(name, row.get(name))
        if row.get("completion_status") != "success" or row["source"] == row["destination"]:
            raise ValueError("shard completion must be a successful nonlocal flow")
        times = tuple(_nonnegative_int("timestamp_ps", event.get("timestamp_ps")) for event in events)
        if times != tuple(sorted(times)) or (times[0], times[-1]) != (
            row["start_time_ps"], row["completion_time_ps"]
        ) or row["fct_ps"] != times[-1] - times[0]:
            raise ValueError("shard lifecycle and completion times disagree")
        identity = ("sequence", "execution_id", "operation_id", "flow_id", "source", "destination", "payload_bytes")
        if any(any(type(event.get(key)) is not type(row[key]) or event.get(key) != row[key]
                   for key in identity) for event in events):
            raise ValueError("shard lifecycle identity disagrees with its flow")
        object.__setattr__(self, "row", row)
        object.__setattr__(self, "events", events)

    @property
    def sequence(self) -> int:
        return self.row["sequence"]

    @property
    def completed_at_ps(self) -> int:
        return self.row["completion_time_ps"]

    def to_json(self) -> dict[str, object]:
        return {"row": dict(self.row), "events": [dict(event) for event in self.events]}


@dataclass(frozen=True)
class KvHandoffJoin:
    """One consumer-visible all-shard join, distinct from a resource visit."""

    submission: PendingKvHandoff
    shards: tuple[KvHandoffShardCompletion, ...]

    authority = SHARED_PACKET_KV_HANDOFF_AUTHORITY
    pricing_arm = "shared-packet"

    def __post_init__(self) -> None:
        if type(self.submission) is not PendingKvHandoff:
            raise TypeError("join requires its accepted pending handoff")
        self.submission.__post_init__()
        if type(self.shards) is not tuple or any(type(shard) is not KvHandoffShardCompletion for shard in self.shards):
            raise TypeError("join requires immutable shard completions")
        if tuple(shard.sequence for shard in self.shards) != self.submission.sequences:
            raise ValueError("join must cover every accepted shard exactly once")
        if sum(shard.row["payload_bytes"] for shard in self.shards) != self.kv_bytes:
            raise ValueError("join shard bytes do not conserve")
        if any(shard.row["start_time_ps"] != self.eligible_at_ps for shard in self.shards):
            raise ValueError("join shard eligibility disagrees")

    @property
    def request_id(self) -> str:
        return self.submission.request_id

    @property
    def kv_bytes(self) -> int:
        return self.submission.kv_bytes

    @property
    def submitted_at_ps(self) -> int:
        return self.submission.submitted_at_ps

    @property
    def eligible_at_ps(self) -> int:
        return self.submission.eligible_at_ps

    @property
    def completed_at_ps(self) -> int:
        return max(shard.completed_at_ps for shard in self.shards)

    @property
    def total_ps(self) -> int:
        return self.completed_at_ps - self.submitted_at_ps

    @property
    def critical_shard_sequences(self) -> tuple[int, ...]:
        return tuple(shard.sequence for shard in self.shards if shard.completed_at_ps == self.completed_at_ps)

    def to_json(self) -> dict[str, object]:
        return {
            "authority": self.authority, "pricing_arm": self.pricing_arm,
            "source_engine_id": self.submission.source_engine_id,
            "destination_engine_id": self.submission.destination_engine_id,
            "kv_bytes": self.kv_bytes, "submitted_at_ps": self.submitted_at_ps,
            "eligible_at_ps": self.eligible_at_ps, "completed_at_ps": self.completed_at_ps,
            "critical_shard_sequences": list(self.critical_shard_sequences),
            "shards": [shard.to_json() for shard in self.shards],
        }


@runtime_checkable
class PendingKvHandoffPolicy(Protocol):
    """A shared handoff capability that progresses its existing native owner."""

    def validate_engines(self, prefill_ids: Sequence[str], decode_ids: Sequence[str], width: int) -> None: ...

    def bind(self, clock: VirtualClock, engine_local_ranks: Mapping[str, tuple[int, ...]]) -> None: ...

    def submit(self, *, request_id: str, source_engine_id: str, destination_engine_id: str,
               kv_bytes: int, submitted_at_ps: int) -> PendingKvHandoff: ...

    def progress(self, pending: Sequence[PendingKvHandoff], *, through_ps: int | None) -> int | None: ...

    def complete_due(self, pending: Sequence[PendingKvHandoff]) -> tuple[KvHandoffJoin, ...]: ...

    def close(self) -> None: ...

    def abort(self) -> None: ...


@dataclass(frozen=True)
class DeclaredKvHandoffPolicy:
    """Identity-off or declared-constant pricing for the KV join."""

    duration_ps: int
    enabled: bool = True

    def __post_init__(self) -> None:
        _nonnegative_int("duration_ps", self.duration_ps)
        if type(self.enabled) is not bool:
            raise TypeError("enabled must be a boolean")
        if self.enabled and self.duration_ps == 0:
            raise ValueError("an enabled declared handoff must have positive duration")
        if not self.enabled and self.duration_ps != 0:
            raise ValueError("a disabled handoff must have zero duration")

    @classmethod
    def off(cls) -> DeclaredKvHandoffPolicy:
        return cls(duration_ps=0, enabled=False)

    def apply(
        self,
        clock: VirtualClock,
        *,
        request_id: str,
        kv_bytes: int,
    ) -> KvHandoffEvent:
        """Advance the caller's sole clock and return its immutable projection."""

        if not isinstance(clock, VirtualClock):
            raise TypeError("clock must be a VirtualClock")
        event = self.schedule(
            submitted_at_ps=clock.now_ps,
            request_id=request_id,
            kv_bytes=kv_bytes,
        )
        clock.advance_to(event.completed_at_ps)
        return event

    def schedule(
        self,
        *,
        submitted_at_ps: int,
        request_id: str,
        kv_bytes: int,
    ) -> KvHandoffEvent:
        """Schedule one independent handoff without advancing a shared clock."""

        submitted_at_ps = _nonnegative_int("submitted_at_ps", submitted_at_ps)
        completed_at_ps = submitted_at_ps + self.duration_ps
        event = KvHandoffEvent(
            request_id=request_id,
            kv_bytes=kv_bytes,
            submitted_at_ps=submitted_at_ps,
            eligible_at_ps=submitted_at_ps,
            started_at_ps=submitted_at_ps,
            finished_at_ps=completed_at_ps,
            completed_at_ps=completed_at_ps,
            pricing_arm="declared-constant" if self.enabled else "off",
        )
        return event


@dataclass(frozen=True)
class DisaggregatedRequestTimeline:
    """One request's exact critical-path timing through both pools."""

    request_id: str
    admitted_at_ps: int
    prefill_eligible_at_ps: int
    prefill_completed_at_ps: int
    handoff: KvHandoffEvent | KvHandoffJoin
    decode_eligible_at_ps: int
    decode_token_completed_at_ps: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id.strip():
            raise ValueError("request_id must be a nonblank string")
        if self.handoff.request_id != self.request_id:
            raise ValueError("handoff request identity disagrees with the timeline")
        for name in (
            "admitted_at_ps",
            "prefill_eligible_at_ps",
            "prefill_completed_at_ps",
            "decode_eligible_at_ps",
        ):
            _nonnegative_int(name, getattr(self, name))
        tokens = tuple(self.decode_token_completed_at_ps)
        if not tokens:
            raise ValueError("decode_token_completed_at_ps must not be empty")
        for value in tokens:
            _nonnegative_int("decode token completion", value)
        object.__setattr__(self, "decode_token_completed_at_ps", tokens)
        ordered = (
            self.admitted_at_ps,
            self.prefill_eligible_at_ps,
            self.prefill_completed_at_ps,
            self.handoff.submitted_at_ps,
            self.handoff.completed_at_ps,
            self.decode_eligible_at_ps,
            *tokens,
        )
        if tuple(sorted(ordered)) != ordered:
            raise ValueError("session timestamps must be monotonic")
        if self.prefill_completed_at_ps != self.handoff.submitted_at_ps:
            raise ValueError("handoff must be submitted at prefill completion")
        if self.decomposition_total_ps != self.ttft_ps:
            raise ValueError("TTFT decomposition does not conserve")

    @property
    def prefill_queue_ps(self) -> int:
        return self.prefill_eligible_at_ps - self.admitted_at_ps

    @property
    def prefill_service_ps(self) -> int:
        return self.prefill_completed_at_ps - self.prefill_eligible_at_ps

    @property
    def decode_admission_wait_ps(self) -> int:
        return self.decode_eligible_at_ps - self.handoff.completed_at_ps

    @property
    def decode_first_token_service_ps(self) -> int:
        return self.decode_token_completed_at_ps[0] - self.decode_eligible_at_ps

    @property
    def decomposition_total_ps(self) -> int:
        return (
            self.prefill_queue_ps
            + self.prefill_service_ps
            + self.handoff.total_ps
            + self.decode_admission_wait_ps
            + self.decode_first_token_service_ps
        )

    @property
    def ttft_ps(self) -> int:
        return self.decode_token_completed_at_ps[0] - self.admitted_at_ps

    @property
    def tpot_ps(self) -> Fraction | None:
        if len(self.decode_token_completed_at_ps) == 1:
            return None
        return Fraction(
            self.decode_token_completed_at_ps[-1]
            - self.decode_token_completed_at_ps[0],
            len(self.decode_token_completed_at_ps) - 1,
        )

    def to_json(self) -> dict[str, object]:
        tpot = self.tpot_ps
        return {
            "schema": PD_SESSION_SCHEMA,
            "request_id": self.request_id,
            "admitted_at_ps": self.admitted_at_ps,
            "prefill_eligible_at_ps": self.prefill_eligible_at_ps,
            "prefill_completed_at_ps": self.prefill_completed_at_ps,
            "handoff": self.handoff.to_json() if isinstance(self.handoff, KvHandoffJoin) else {
                "authority": self.handoff.authority,
                "pricing_arm": self.handoff.pricing_arm,
                "kv_bytes": self.handoff.kv_bytes,
                "submitted_at_ps": self.handoff.submitted_at_ps,
                "eligible_at_ps": self.handoff.eligible_at_ps,
                "started_at_ps": self.handoff.started_at_ps,
                "finished_at_ps": self.handoff.finished_at_ps,
                "completed_at_ps": self.handoff.completed_at_ps,
            },
            "decode_eligible_at_ps": self.decode_eligible_at_ps,
            "decode_token_completed_at_ps": list(
                self.decode_token_completed_at_ps
            ),
            "ttft_ps": self.ttft_ps,
            "tpot_ps": (
                None
                if tpot is None
                else {"numerator": tpot.numerator, "denominator": tpot.denominator}
            ),
            "decomposition": {
                "prefill_queue_ps": self.prefill_queue_ps,
                "prefill_service_ps": self.prefill_service_ps,
                "handoff_ps": self.handoff.total_ps,
                "decode_admission_wait_ps": self.decode_admission_wait_ps,
                "decode_first_token_service_ps": self.decode_first_token_service_ps,
                "total_ps": self.decomposition_total_ps,
            },
        }


__all__ = [
    "KV_HANDOFF_ARMS",
    "KV_HANDOFF_AUTHORITIES",
    "KV_HANDOFF_AUTHORITY",
    "PACKET_KV_HANDOFF_AUTHORITY",
    "PD_SESSION_SCHEMA",
    "SHARED_PACKET_KV_HANDOFF_AUTHORITY",
    "DeclaredKvHandoffPolicy",
    "DisaggregatedRequestTimeline",
    "KvHandoffEvent",
    "KvHandoffGeometry",
    "KvHandoffJoin",
    "KvHandoffPolicy",
    "KvHandoffShardCompletion",
    "PendingKvHandoff",
    "PendingKvHandoffPolicy",
    "ServingPoolRole",
]
