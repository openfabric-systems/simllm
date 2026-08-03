"""Scheduler-step records: the contract between frontend adapters and the core.

One :class:`StepRecord` describes what the framework's scheduler decided to run
in a single engine step; one :class:`StepResult` carries the simulated outcome
back. Adapters (vLLM, SGLang) translate their native scheduler outputs into
these records so the core never depends on a specific framework.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field


class RequestPhase(enum.Enum):
    PREFILL = "prefill"
    DECODE = "decode"


@dataclass
class ScheduledRequest:
    """One request's share of a scheduler step."""

    request_id: str
    phase: RequestPhase
    #: new tokens computed this step (prefill chunk size, or 1 for decode)
    num_new_tokens: int
    #: prompt tokens served from the prefix cache (no compute, no prefill traffic)
    num_cached_tokens: int = 0
    #: total context length after this step
    context_length: int = 0


@dataclass
class StepRecord:
    """What the frontend scheduler decided to run in one engine step."""

    step_index: int
    #: virtual time at which the scheduler released this step, picoseconds
    virtual_time_ps: int
    scheduled: list[ScheduledRequest] = field(default_factory=list)
    preempted_request_ids: list[str] = field(default_factory=list)
    finished_request_ids: list[str] = field(default_factory=list)

    @property
    def total_new_tokens(self) -> int:
        return sum(r.num_new_tokens for r in self.scheduled)


@dataclass
class StepResult:
    """Simulated outcome of one step, produced by the core."""

    step_index: int
    #: simulated wall time this step took, picoseconds
    step_latency_ps: int
    #: virtual time at which the step completed
    completed_at_ps: int
