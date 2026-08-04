"""Scheduler-step records: the contract between frontend adapters and the core.

One :class:`StepRecord` describes what the framework's scheduler decided to run
in a single engine step; one :class:`StepResult` carries the simulated outcome
back. Adapters (vLLM, SGLang) translate their native scheduler outputs into
these records so the core never depends on a specific framework.

In closed-loop mode the same contract crosses a process boundary as versioned
JSON manifests: a step manifest (schema ``atlahs-closed-loop-step-v1``, what
the scheduler ran plus the virtual time) goes to the simulator, and a result
manifest (schema ``atlahs-closed-loop-result-v1``, ``simulated_time_us`` plus
per-flow completions) comes back. Per-step subprocess invocation is the
diagnostic mode; a persistent co-simulator process is planned (BRIDGE-1).
"""

from __future__ import annotations

import enum
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: closed-loop wire-format schema names for StepRecord / StepResult
STEP_SCHEMA = "atlahs-closed-loop-step-v1"
RESULT_SCHEMA = "atlahs-closed-loop-result-v1"


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


def step_record_to_json(record: StepRecord) -> dict[str, Any]:
    """One record as a JSON-ready dict, tagged with :data:`STEP_SCHEMA`.

    This is *the* JSON form of a step record: the offline JSONL dump and the
    closed-loop step manifest both use it, so the two cannot drift.

    Attribution note: ``finished_request_ids`` on record N lists the requests
    the scheduler reported finished when it released step N, i.e. they
    completed during the *previous* step (vLLM's scheduler rebinds its
    finished set after constructing the step, so the ids arrive one step
    later). ``preempted_request_ids`` is same-step. A consumer computing
    completion times joins a finished id to the preceding record's virtual
    time.
    """
    return {
        "schema": STEP_SCHEMA,
        "step_index": record.step_index,
        "virtual_time_ps": record.virtual_time_ps,
        "scheduled": [
            {
                "request_id": request.request_id,
                "phase": request.phase.value,
                "num_new_tokens": request.num_new_tokens,
                "num_cached_tokens": request.num_cached_tokens,
                "context_length": request.context_length,
            }
            for request in record.scheduled
        ],
        "preempted_request_ids": list(record.preempted_request_ids),
        "finished_request_ids": list(record.finished_request_ids),
    }


def step_records_to_json(records: Sequence[StepRecord]) -> list[dict[str, Any]]:
    """Step records as plain JSON-ready dicts (phase enums become strings)."""
    return [step_record_to_json(record) for record in records]


def write_step_records(records: Sequence[StepRecord], path: str | Path) -> Path:
    """Write one JSON object per step record, newline delimited."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        handle.writelines(json.dumps(entry) + "\n" for entry in step_records_to_json(records))
    return path


class StepRecordStream:
    """Append records to a JSONL path the moment each step completes.

    Frontend engines do not reliably route in-process teardown through any
    callback the adapter can see (observed on vLLM v0.26.0), so a dump that
    waits for shutdown loses everything; this writer truncates the file on
    the first append and makes every record durable immediately.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._started = False

    @property
    def path(self) -> Path:
        return self._path

    def append(self, record: StepRecord) -> None:
        if not self._started:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text("")
            self._started = True
        with open(self._path, "a") as handle:
            handle.write(json.dumps(step_record_to_json(record)) + "\n")
