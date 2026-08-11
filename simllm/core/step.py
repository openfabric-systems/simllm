"""Scheduler-step records: the contract between frontend adapters and the core.

One :class:`StepRecord` describes what the framework's scheduler decided to run
in a single engine step; one :class:`StepResult` carries the simulated outcome
back. Adapters (vLLM, SGLang) translate their native scheduler outputs into
these records so the core never depends on a specific framework.

In closed-loop mode the same contract crosses a process boundary as versioned
JSON objects. The step-record schema remains ``atlahs-closed-loop-step-v1``.
The full result schema is ``simllm-step-result-v2`` and preserves every CORE-5
metric without floating-point conversion. Per-step subprocess invocation is
the diagnostic mode; BRIDGE-2 owns the persistent graph-level client.
"""

from __future__ import annotations

import enum
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

#: closed-loop wire-format schema names for StepRecord / StepResult
STEP_SCHEMA = "atlahs-closed-loop-step-v1"
LEGACY_RESULT_SCHEMA = "atlahs-closed-loop-result-v1"
RESULT_SCHEMA = "simllm-step-result-v2"


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
    #: exact number of scheduled requests that sample this step; None means unknown
    num_sampled: int | None = None
    #: physical model-input tokens after framework padding; None means unobserved
    num_tokens_after_padding: int | None = None
    #: exact identities sampled this step; None retains the legacy count-only form
    sampled_request_ids: list[str] | None = None

    def __post_init__(self) -> None:
        if self.num_sampled is not None:
            if isinstance(self.num_sampled, bool) or not isinstance(self.num_sampled, int):
                raise TypeError("num_sampled must be an integer or None")
            if not 0 <= self.num_sampled <= len(self.scheduled):
                raise ValueError(
                    f"num_sampled={self.num_sampled} must be between 0 and "
                    f"len(scheduled)={len(self.scheduled)}"
                )
        if self.num_tokens_after_padding is not None:
            value = self.num_tokens_after_padding
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError("num_tokens_after_padding must be an integer or None")
            if value < self.total_new_tokens:
                raise ValueError(
                    f"num_tokens_after_padding={value} must be at least "
                    f"total_new_tokens={self.total_new_tokens}"
                )
        if self.sampled_request_ids is not None:
            if not isinstance(self.sampled_request_ids, list):
                raise TypeError("sampled_request_ids must be a list or None")
            if any(
                not isinstance(request_id, str) or not request_id.strip()
                for request_id in self.sampled_request_ids
            ):
                raise ValueError("sampled_request_ids must contain nonblank strings")
            if len(self.sampled_request_ids) != len(set(self.sampled_request_ids)):
                raise ValueError("sampled_request_ids must be unique")
            scheduled_ids = {request.request_id for request in self.scheduled}
            unknown = sorted(set(self.sampled_request_ids) - scheduled_ids)
            if unknown:
                raise ValueError(
                    f"sampled_request_ids contains unscheduled requests: {unknown}"
                )
            if (
                self.num_sampled is not None
                and len(self.sampled_request_ids) != self.num_sampled
            ):
                raise ValueError(
                    "sampled_request_ids cardinality must equal num_sampled"
                )

    @property
    def total_new_tokens(self) -> int:
        return sum(r.num_new_tokens for r in self.scheduled)


def _require_nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


@dataclass(frozen=True)
class LatencyAttribution:
    """One realized request interval partitioned by semantic owner."""

    queue_ps: int = 0
    kv_ps: int = 0
    kernel_ps: int = 0
    dma_ps: int = 0
    collective_ps: int = 0
    nic_ps: int = 0
    control_ps: int = 0

    def __post_init__(self) -> None:
        for name in (
            "queue_ps",
            "kv_ps",
            "kernel_ps",
            "dma_ps",
            "collective_ps",
            "nic_ps",
            "control_ps",
        ):
            _require_nonnegative_int(name, getattr(self, name))

    @property
    def total_ps(self) -> int:
        """Return the conserved elapsed time represented by this partition."""

        return (
            self.queue_ps
            + self.kv_ps
            + self.kernel_ps
            + self.dma_ps
            + self.collective_ps
            + self.nic_ps
            + self.control_ps
        )

    def __add__(self, other: object) -> LatencyAttribution:
        if not isinstance(other, LatencyAttribution):
            return NotImplemented
        return LatencyAttribution(
            queue_ps=self.queue_ps + other.queue_ps,
            kv_ps=self.kv_ps + other.kv_ps,
            kernel_ps=self.kernel_ps + other.kernel_ps,
            dma_ps=self.dma_ps + other.dma_ps,
            collective_ps=self.collective_ps + other.collective_ps,
            nic_ps=self.nic_ps + other.nic_ps,
            control_ps=self.control_ps + other.control_ps,
        )


@dataclass(frozen=True)
class AdditiveVisitTotals:
    """Work sums over visits, never an additive wall-latency decomposition."""

    queue_wait_ps: int = 0
    service_ps: int = 0
    visibility_ps: int = 0
    visit_count: int = 0

    def __post_init__(self) -> None:
        for name in (
            "queue_wait_ps",
            "service_ps",
            "visibility_ps",
            "visit_count",
        ):
            _require_nonnegative_int(name, getattr(self, name))

    @property
    def total_ps(self) -> int:
        """Return additive visit work, which may exceed wall time."""

        return self.queue_wait_ps + self.service_ps + self.visibility_ps

    def __add__(self, other: object) -> AdditiveVisitTotals:
        if not isinstance(other, AdditiveVisitTotals):
            return NotImplemented
        return AdditiveVisitTotals(
            queue_wait_ps=self.queue_wait_ps + other.queue_wait_ps,
            service_ps=self.service_ps + other.service_ps,
            visibility_ps=self.visibility_ps + other.visibility_ps,
            visit_count=self.visit_count + other.visit_count,
        )


@dataclass(frozen=True)
class RequestMetric:
    """Latest sampled-token boundary and exact per-request latency metrics."""

    request_id: str
    phase: RequestPhase
    token_index: int
    completed_at_ps: int
    latency_ps: int
    ttft_ps: int
    tpot_ps: Fraction | None
    attribution: LatencyAttribution
    additive_visit_totals: AdditiveVisitTotals

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id.strip():
            raise ValueError("request_id must be a nonblank string")
        if not isinstance(self.phase, RequestPhase):
            raise TypeError("phase must be a RequestPhase")
        _require_nonnegative_int("token_index", self.token_index)
        if self.token_index == 0:
            raise ValueError("token_index must be positive")
        for name in ("completed_at_ps", "latency_ps", "ttft_ps"):
            _require_nonnegative_int(name, getattr(self, name))
        if self.tpot_ps is not None:
            if not isinstance(self.tpot_ps, Fraction):
                raise TypeError("tpot_ps must be a Fraction or None")
            if self.tpot_ps < 0:
                raise ValueError("tpot_ps must be nonnegative")
        if not isinstance(self.attribution, LatencyAttribution):
            raise TypeError("attribution must be a LatencyAttribution")
        if self.attribution.total_ps != self.latency_ps:
            raise ValueError("request attribution does not conserve latency_ps")
        if not isinstance(self.additive_visit_totals, AdditiveVisitTotals):
            raise TypeError("additive_visit_totals must be AdditiveVisitTotals")


@dataclass
class StepResult:
    """Simulated outcome of one step, produced by the core."""

    step_index: int
    #: simulated wall time this step took, picoseconds
    step_latency_ps: int
    #: virtual time at which the step completed
    completed_at_ps: int
    #: sampled request boundaries completed in this step
    request_metrics: tuple[RequestMetric, ...] = ()
    #: graph-wide visit work, kept separate from request critical paths
    additive_visit_totals: AdditiveVisitTotals | None = None

    def __post_init__(self) -> None:
        for name in ("step_index", "step_latency_ps", "completed_at_ps"):
            _require_nonnegative_int(name, getattr(self, name))
        if not isinstance(self.request_metrics, tuple):
            raise TypeError("request_metrics must be a tuple")
        if any(not isinstance(metric, RequestMetric) for metric in self.request_metrics):
            raise TypeError("request_metrics must contain RequestMetric records")
        request_ids = [metric.request_id for metric in self.request_metrics]
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("request_metrics must contain at most one row per request")
        if any(metric.completed_at_ps > self.completed_at_ps for metric in self.request_metrics):
            raise ValueError("request metric completes after its StepResult")
        if self.additive_visit_totals is not None and not isinstance(
            self.additive_visit_totals,
            AdditiveVisitTotals,
        ):
            raise TypeError("additive_visit_totals must be AdditiveVisitTotals or None")


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
    payload = {
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
    if record.num_sampled is not None:
        payload["num_sampled"] = record.num_sampled
    if record.num_tokens_after_padding is not None:
        payload["num_tokens_after_padding"] = record.num_tokens_after_padding
    if record.sampled_request_ids is not None:
        payload["sampled_request_ids"] = list(record.sampled_request_ids)
    return payload


def step_records_to_json(records: Sequence[StepRecord]) -> list[dict[str, Any]]:
    """Step records as plain JSON-ready dicts (phase enums become strings)."""
    return [step_record_to_json(record) for record in records]


def step_record_from_json(payload: dict[str, Any]) -> StepRecord:
    """Inverse of :func:`step_record_to_json`.

    Validates the schema tag first and rejects anything else loudly: a
    record stream is a wire format, so silently accepting an untagged or
    differently-versioned payload would let two ends drift apart without
    either noticing.
    """
    schema = payload.get("schema")
    if schema != STEP_SCHEMA:
        raise ValueError(
            f"unsupported step-record schema {schema!r}; this reader speaks {STEP_SCHEMA!r}"
        )
    scheduled = [
        ScheduledRequest(
            request_id=entry["request_id"],
            phase=RequestPhase(entry["phase"]),
            num_new_tokens=entry["num_new_tokens"],
            num_cached_tokens=entry.get("num_cached_tokens", 0),
            context_length=entry.get("context_length", 0),
        )
        for entry in payload.get("scheduled", [])
    ]
    return StepRecord(
        step_index=payload["step_index"],
        virtual_time_ps=payload["virtual_time_ps"],
        scheduled=scheduled,
        preempted_request_ids=list(payload.get("preempted_request_ids", [])),
        finished_request_ids=list(payload.get("finished_request_ids", [])),
        num_sampled=payload.get("num_sampled"),
        num_tokens_after_padding=payload.get("num_tokens_after_padding"),
        sampled_request_ids=(
            list(payload["sampled_request_ids"])
            if payload.get("sampled_request_ids") is not None
            else None
        ),
    )


def step_records_from_jsonl(path: str | Path) -> list[StepRecord]:
    """Load a newline-delimited step-record dump (the adapters' JSONL format).

    Every non-blank line must be one schema-tagged record; a bad line names
    its line number so a truncated or foreign file fails with a usable error.
    """
    records: list[StepRecord] = []
    with open(Path(path)) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(step_record_from_json(json.loads(line)))
            except (TypeError, ValueError, KeyError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return records


def write_step_records(records: Sequence[StepRecord], path: str | Path) -> Path:
    """Write one JSON object per step record, newline delimited."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="\n") as handle:
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
        with open(self._path, "a", newline="\n") as handle:
            handle.write(json.dumps(step_record_to_json(record)) + "\n")
