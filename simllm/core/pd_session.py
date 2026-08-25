"""Framework-neutral timing records for a disaggregated serving session.

The frontend adapters own their schedulers. This module owns only the narrow
join between a completed prefill leg and an admitted decode leg: one KV-cache
handoff event on the session's existing :class:`VirtualClock`, followed by an
exact per-request timing reduction. It imports no serving framework.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from fractions import Fraction

from simllm.core.clock import VirtualClock

PD_SESSION_SCHEMA = "simllm-pd-session-result-v1"
KV_HANDOFF_AUTHORITY = "simllm-declared-kv-handoff-v1"
KV_HANDOFF_ARMS = ("off", "declared-constant")


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
        if self.authority != KV_HANDOFF_AUTHORITY:
            raise ValueError(f"authority must be {KV_HANDOFF_AUTHORITY!r}")
        if self.queue_wait_ps != 0:
            raise ValueError("the declared KV handoff has no internal queue")
        if self.visibility_ps != 0:
            raise ValueError("the declared KV handoff has no visibility tail")
        if self.pricing_arm == "off" and self.service_ps != 0:
            raise ValueError("the off KV handoff arm must have zero service")
        if self.pricing_arm == "declared-constant" and self.service_ps == 0:
            raise ValueError("the declared-constant KV handoff must have service")

    @property
    def queue_wait_ps(self) -> int:
        return self.started_at_ps - self.eligible_at_ps

    @property
    def service_ps(self) -> int:
        return self.finished_at_ps - self.started_at_ps

    @property
    def visibility_ps(self) -> int:
        return self.completed_at_ps - self.finished_at_ps


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
        submitted_at_ps = clock.now_ps
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
        clock.advance_to(event.completed_at_ps)
        return event


@dataclass(frozen=True)
class DisaggregatedRequestTimeline:
    """One request's exact critical-path timing through both pools."""

    request_id: str
    admitted_at_ps: int
    prefill_eligible_at_ps: int
    prefill_completed_at_ps: int
    handoff: KvHandoffEvent
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
            + self.handoff.service_ps
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
            "handoff": {
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
                "handoff_ps": self.handoff.service_ps,
                "decode_admission_wait_ps": self.decode_admission_wait_ps,
                "decode_first_token_service_ps": self.decode_first_token_service_ps,
                "total_ps": self.decomposition_total_ps,
            },
        }


__all__ = [
    "KV_HANDOFF_ARMS",
    "KV_HANDOFF_AUTHORITY",
    "PD_SESSION_SCHEMA",
    "DeclaredKvHandoffPolicy",
    "DisaggregatedRequestTimeline",
    "KvHandoffEvent",
    "KvHandoffGeometry",
    "ServingPoolRole",
]
