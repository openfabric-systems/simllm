"""Open-loop client driver for SGLang's native streaming generate endpoint.

The module uses only the standard library and is importable without SGLang.
It observes wall-clock submission and token visibility relative to one local
origin. Those timestamps are external evidence, not simulated virtual time.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from http.client import HTTPConnection, HTTPSConnection
from numbers import Integral
from typing import Any, Protocol
from urllib.parse import SplitResult, urlsplit

from simllm.workload.serving import (
    PICOSECONDS_PER_SECOND,
    GenerationRequest,
    TransportRequestObservation,
)

DEFAULT_GENERATE_PATH = "/generate"
_SSE_DONE = object()


class MonotonicClock(Protocol):
    """Nanosecond monotonic clock used by the pacing loop."""

    def __call__(self) -> int: ...


class Sleeper(Protocol):
    """Injectable wall-clock sleeper."""

    def __call__(self, seconds: float) -> None: ...


@dataclass(frozen=True)
class PreparedSglangSubmission:
    """One started transport operation whose response may remain in flight."""

    request_id: str
    submitted_at_ps: int
    await_observation: Callable[[], TransportRequestObservation]


class SglangRequestSubmitter(Protocol):
    """Start one transport in plan order without waiting for its response."""

    def start(
        self,
        request: GenerationRequest,
        now_ps: Callable[[], int],
    ) -> PreparedSglangSubmission: ...


def sglang_generate_payload(request: GenerationRequest) -> dict[str, Any]:
    """Map one realized request to pinned SGLang's native `/generate` form."""

    if not isinstance(request, GenerationRequest):
        raise TypeError("request must be a GenerationRequest")
    return {
        "rid": request.request_id,
        "input_ids": list(request.prompt_token_ids),
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": request.output_token_count,
            "ignore_eos": True,
        },
        "stream": True,
        "return_logprob": False,
    }


def _sse_payload(line: bytes) -> bytes | object | None:
    stripped = line.strip()
    if not stripped:
        return None
    if stripped.startswith(b"data:"):
        stripped = stripped[5:].lstrip()
    if stripped == b"[DONE]":
        return _SSE_DONE
    return stripped


def _http_connection(url: str, timeout_s: float) -> tuple[HTTPConnection, str]:
    parsed: SplitResult = urlsplit(url)
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("credentials must be supplied through headers, not base_url")
    if parsed.hostname is None:
        raise ValueError("SGLang generate URL has no host")
    connection_type = HTTPSConnection if parsed.scheme == "https" else HTTPConnection
    connection = connection_type(parsed.hostname, parsed.port, timeout=timeout_s)
    target = parsed.path or "/"
    if parsed.query:
        target += "?" + parsed.query
    return connection, target


def token_completion_times_from_sglang_chunks(
    chunks: Iterable[tuple[Mapping[str, Any], int]],
    *,
    expected_count: int,
    expected_request_id: str | None = None,
) -> tuple[int, ...]:
    """Convert cumulative SGLang stream chunks to observable token frontiers."""

    if isinstance(expected_count, bool) or not isinstance(expected_count, Integral):
        raise TypeError("expected_count must be an integer")
    if expected_count <= 0:
        raise ValueError("expected_count must be positive")
    result: list[int] = []
    cumulative = 0
    last_timestamp = -1
    saw_length_finish = False
    for chunk, timestamp in chunks:
        if saw_length_finish:
            raise RuntimeError("SGLang stream returned data after its terminal frame")
        if chunk.get("error") is not None:
            raise RuntimeError(
                "SGLang stream returned an error frame: "
                + json.dumps(chunk["error"], sort_keys=True)
            )
        if isinstance(timestamp, bool) or not isinstance(timestamp, Integral):
            raise TypeError("chunk timestamps must be integers")
        timestamp = int(timestamp)
        if timestamp < 0 or timestamp < last_timestamp:
            raise ValueError("chunk timestamps must be nonnegative and nondecreasing")
        last_timestamp = timestamp
        meta = chunk.get("meta_info") or {}
        if not isinstance(meta, Mapping):
            raise TypeError("SGLang meta_info must be a JSON object")
        if expected_request_id is not None:
            returned_request_id = meta.get("id")
            if returned_request_id != expected_request_id:
                raise ValueError(
                    f"SGLang stream request id {returned_request_id!r} does not match "
                    f"planned request {expected_request_id!r}"
                )
        count = meta.get("completion_tokens")
        if count is None:
            if meta.get("finish_reason") is None and not chunk.get("output_ids"):
                continue
            raise ValueError("SGLang stream chunk is missing cumulative completion_tokens")
        if isinstance(count, bool) or not isinstance(count, Integral):
            raise TypeError("SGLang completion_tokens must be an integer")
        count = int(count)
        if count < cumulative:
            raise ValueError("SGLang cumulative completion count decreased")
        if count > expected_count:
            raise ValueError(
                f"SGLang reported {count} completions, expected at most {expected_count}"
            )
        result.extend([timestamp] * (count - cumulative))
        cumulative = count
        finish_reason = meta.get("finish_reason")
        if finish_reason is not None:
            if not isinstance(finish_reason, Mapping):
                raise TypeError("SGLang finish_reason must be a JSON object")
            if finish_reason.get("type") != "length":
                raise RuntimeError(
                    "SGLang request did not finish at its requested length: "
                    + json.dumps(finish_reason, sort_keys=True)
                )
            finish_length = finish_reason.get("length")
            if (
                isinstance(finish_length, bool)
                or not isinstance(finish_length, Integral)
                or int(finish_length) != expected_count
                or count != expected_count
            ):
                raise RuntimeError(
                    "SGLang length finish does not match the requested output count: "
                    + json.dumps(finish_reason, sort_keys=True)
                )
            saw_length_finish = True
    if cumulative != expected_count:
        raise ValueError(
            f"SGLang stream ended with {cumulative} completions, expected {expected_count}"
        )
    if expected_request_id is not None and not saw_length_finish:
        raise RuntimeError("SGLang stream ended without a length finish reason")
    return tuple(result)


@dataclass(frozen=True)
class SglangHttpSubmitter:
    """Standard-library HTTP submitter for a native SGLang server."""

    base_url: str
    timeout_s: float = 600.0
    headers: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive")

    @property
    def generate_url(self) -> str:
        return self.base_url.rstrip("/") + DEFAULT_GENERATE_PATH

    def __call__(
        self,
        request: GenerationRequest,
        submitted_at_ps: int,
        now_ps: Callable[[], int],
    ) -> TransportRequestObservation:
        del submitted_at_ps
        return self.start(request, now_ps).await_observation()

    def start(
        self,
        request: GenerationRequest,
        now_ps: Callable[[], int],
    ) -> PreparedSglangSubmission:
        """Send the HTTP request now, then return a response-drain callback."""

        payload = json.dumps(
            sglang_generate_payload(request), separators=(",", ":")
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.headers:
            headers.update(self.headers)
        connection, target = _http_connection(self.generate_url, self.timeout_s)
        try:
            connection.request("POST", target, body=payload, headers=headers)
        except BaseException:
            connection.close()
            raise
        submitted_at_ps = now_ps()

        def await_observation() -> TransportRequestObservation:
            chunks: list[tuple[Mapping[str, Any], int]] = []
            saw_done = False
            try:
                response = connection.getresponse()
            except BaseException:
                connection.close()
                raise
            status = response.status
            if status != 200:
                response.close()
                connection.close()
                raise RuntimeError(f"SGLang /generate returned HTTP {status}")
            try:
                for line in response:
                    raw = _sse_payload(line)
                    if raw is None:
                        continue
                    if raw is _SSE_DONE:
                        saw_done = True
                        break
                    assert isinstance(raw, bytes)
                    decoded = json.loads(raw)
                    if not isinstance(decoded, Mapping):
                        raise TypeError("SGLang stream chunk must be a JSON object")
                    chunks.append((decoded, now_ps()))
            finally:
                response.close()
                connection.close()
            if not saw_done:
                raise RuntimeError("SGLang stream ended without a [DONE] frame")
            return TransportRequestObservation(
                request_id=request.request_id,
                submitted_at_ps=submitted_at_ps,
                token_completed_at_ps=token_completion_times_from_sglang_chunks(
                    chunks,
                    expected_count=request.output_token_count,
                    expected_request_id=request.request_id,
                ),
            )

        return PreparedSglangSubmission(
            request_id=request.request_id,
            submitted_at_ps=submitted_at_ps,
            await_observation=await_observation,
        )


@dataclass(frozen=True)
class SglangOpenLoopDriver:
    """Pace a frozen plan while earlier requests continue in flight."""

    submit: SglangRequestSubmitter
    monotonic_ns: MonotonicClock = time.perf_counter_ns
    sleep: Sleeper = time.sleep
    max_workers: int = 64

    def _validated_plan(
        self, requests: tuple[GenerationRequest, ...]
    ) -> tuple[GenerationRequest, ...]:
        plan = tuple(requests)
        ids: set[str] = set()
        ordinals: set[int] = set()
        previous_key: tuple[int, int] | None = None
        for request in plan:
            if not isinstance(request, GenerationRequest):
                raise TypeError("requests must contain GenerationRequest values")
            if request.request_id in ids:
                raise ValueError(f"duplicate request ID {request.request_id!r}")
            if request.ordinal in ordinals:
                raise ValueError(f"duplicate request ordinal {request.ordinal}")
            key = (request.arrived_at_ps, request.ordinal)
            if previous_key is not None and key < previous_key:
                raise ValueError("requests must be ordered by arrival then ordinal")
            ids.add(request.request_id)
            ordinals.add(request.ordinal)
            previous_key = key
        return plan

    def run_open_loop(
        self, requests: tuple[GenerationRequest, ...]
    ) -> tuple[TransportRequestObservation, ...]:
        plan = self._validated_plan(requests)
        if not plan:
            return ()
        if isinstance(self.max_workers, bool) or not isinstance(self.max_workers, Integral):
            raise TypeError("max_workers must be an integer")
        worker_count = int(self.max_workers)
        if worker_count <= 0:
            raise ValueError("max_workers must be positive")
        if worker_count < len(plan):
            raise ValueError(
                "the frozen plan exceeds max_workers; use a small smoke or an async "
                "transport so delayed response reads cannot distort token visibility"
            )
        origin_ns = self.monotonic_ns()

        def now_ps() -> int:
            elapsed_ns = self.monotonic_ns() - origin_ns
            if elapsed_ns < 0:
                raise RuntimeError("monotonic clock moved backwards")
            return elapsed_ns * 1_000

        futures: dict[
            Future[TransportRequestObservation], tuple[GenerationRequest, int]
        ] = {}
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            for request in plan:
                while True:
                    remaining_ps = request.arrived_at_ps - now_ps()
                    if remaining_ps <= 0:
                        break
                    self.sleep(remaining_ps / PICOSECONDS_PER_SECOND)
                prepared = self.submit.start(request, now_ps)
                if not isinstance(prepared, PreparedSglangSubmission):
                    raise TypeError("SGLang submitter start() returned an invalid operation")
                if prepared.request_id != request.request_id:
                    raise ValueError(
                        f"submitter started request {prepared.request_id!r} "
                        f"for {request.request_id!r}"
                    )
                prepared_at = prepared.submitted_at_ps
                if isinstance(prepared_at, bool) or not isinstance(prepared_at, Integral):
                    raise TypeError("prepared submission timestamp must be an integer")
                prepared_at = int(prepared_at)
                if prepared_at < request.arrived_at_ps:
                    raise ValueError(
                        f"request {request.request_id!r} was prepared before arrival"
                    )
                future = executor.submit(prepared.await_observation)
                futures[future] = (request, prepared_at)

            completed: dict[str, TransportRequestObservation] = {}
            for future in as_completed(futures):
                request, prepared_at = futures[future]
                observation = future.result()
                if observation.request_id != request.request_id:
                    raise ValueError(
                        f"submitter returned request {observation.request_id!r} "
                        f"for {request.request_id!r}"
                    )
                if observation.submitted_at_ps != prepared_at:
                    raise ValueError(
                        f"submitter changed request {request.request_id!r} submission "
                        f"timestamp from {prepared_at} to {observation.submitted_at_ps}"
                    )
                completed[request.request_id] = observation
        return tuple(completed[request.request_id] for request in plan)
