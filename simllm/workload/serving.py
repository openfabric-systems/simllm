"""Deterministic generation workloads and client-observed request timing.

Realization is deliberately separate from transport. A caller can inspect and
persist the exact request plan before a real framework sees it, while an
adapter-specific open-loop transport owns only submission and streamed-token
observation. Neither surface schedules simulated GPU work.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from fractions import Fraction
from itertools import pairwise
from numbers import Integral, Real
from typing import Any, Protocol

PICOSECONDS_PER_SECOND = 1_000_000_000_000


class ArrivalProvider(Protocol):
    """An existing workload arrival source with absolute seconds."""

    def times(self, n: int) -> Any: ...


class LengthProvider(Protocol):
    """An existing workload token-count sampler."""

    def sample(self, n: int) -> Any: ...


class PromptBuilder(Protocol):
    """Build exactly one prompt after its length has been sampled."""

    def __call__(
        self, request_id: str, prompt_length: int, ordinal: int
    ) -> Iterable[int]: ...


@dataclass(frozen=True)
class GenerationRequest:
    """One fully realized open-loop generation request."""

    ordinal: int
    request_id: str
    arrived_at_ps: int
    prompt_token_ids: tuple[int, ...]
    output_token_count: int

    def __post_init__(self) -> None:
        if isinstance(self.ordinal, bool) or not isinstance(self.ordinal, Integral):
            raise TypeError("request ordinal must be an integer")
        if self.ordinal < 0:
            raise ValueError("request ordinal must be nonnegative")
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("request_id must be a nonempty string")
        if isinstance(self.arrived_at_ps, bool) or not isinstance(
            self.arrived_at_ps, Integral
        ):
            raise TypeError("arrived_at_ps must be an integer")
        if self.arrived_at_ps < 0:
            raise ValueError("arrived_at_ps must be nonnegative")
        prompt = tuple(self.prompt_token_ids)
        if not prompt:
            raise ValueError("prompt_token_ids must not be empty")
        for token_id in prompt:
            if isinstance(token_id, bool) or not isinstance(token_id, Integral):
                raise TypeError("prompt token IDs must be integers")
            if token_id < 0:
                raise ValueError("prompt token IDs must be nonnegative")
        if isinstance(self.output_token_count, bool) or not isinstance(
            self.output_token_count, Integral
        ):
            raise TypeError("output_token_count must be an integer")
        if self.output_token_count <= 0:
            raise ValueError("output_token_count must be positive")
        object.__setattr__(self, "ordinal", int(self.ordinal))
        object.__setattr__(self, "arrived_at_ps", int(self.arrived_at_ps))
        object.__setattr__(self, "prompt_token_ids", tuple(int(token) for token in prompt))
        object.__setattr__(self, "output_token_count", int(self.output_token_count))


@dataclass(frozen=True)
class HashedTokenPrompts:
    """Stable synthetic prompts keyed independently by request and position.

    The keyed hash avoids dependence on process hash randomization, NumPy RNG
    versions, or request callback order. It intentionally does not introduce
    shared prefixes; WORK-1 owns controlled prefix structure.
    """

    vocab_size: int
    seed: int = 0
    first_token_id: int = 0

    def __post_init__(self) -> None:
        for name, value in (
            ("vocab_size", self.vocab_size),
            ("seed", self.seed),
            ("first_token_id", self.first_token_id),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if not 0 <= self.first_token_id < self.vocab_size:
            raise ValueError("first_token_id must be inside the vocabulary")

    def __call__(
        self, request_id: str, prompt_length: int, ordinal: int
    ) -> tuple[int, ...]:
        span = self.vocab_size - self.first_token_id
        if span <= 0:
            raise ValueError("the configured token range is empty")
        tokens: list[int] = []
        for position in range(prompt_length):
            key = f"{self.seed}\0{ordinal}\0{request_id}\0{position}".encode()
            digest = hashlib.blake2b(
                key, digest_size=8, person=b"simllm-prompt"
            ).digest()
            tokens.append(
                self.first_token_id + int.from_bytes(digest, "big", signed=False) % span
            )
        return tuple(tokens)


def _exact_sequence(name: str, values: Any, count: int) -> tuple[Any, ...]:
    try:
        materialized = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{name} must return an iterable") from exc
    if len(materialized) != count:
        raise ValueError(
            f"{name} returned {len(materialized)} values for {count} requests"
        )
    return materialized


def _arrival_decimal(value: Any) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (Real, Decimal)):
        raise TypeError(f"arrival times must be real numbers, got {value!r}")
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"arrival time is not numeric: {value!r}") from exc
    if not result.is_finite() or not math.isfinite(float(result)):
        raise ValueError(f"arrival time must be finite, got {value!r}")
    if result < 0:
        raise ValueError(f"arrival time must be nonnegative, got {value!r}")
    return result


def _arrival_ps(value: Decimal) -> int:
    return int(
        (value * PICOSECONDS_PER_SECOND).to_integral_value(rounding=ROUND_HALF_UP)
    )


def _positive_length(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} values must be integers, got {value!r}")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} values must be positive, got {result}")
    return result


def realize_generation_requests(
    request_ids: Iterable[str],
    *,
    arrivals: ArrivalProvider,
    prompt_lengths: LengthProvider,
    output_lengths: LengthProvider,
    prompt_builder: PromptBuilder,
) -> tuple[GenerationRequest, ...]:
    """Freeze arrival, length, and prompt randomness into immutable requests."""

    ids = tuple(request_ids)
    if any(not isinstance(request_id, str) or not request_id for request_id in ids):
        raise ValueError("generation request IDs must be nonempty strings")
    if len(set(ids)) != len(ids):
        raise ValueError("generation request IDs must be unique")
    count = len(ids)
    raw_arrivals = _exact_sequence("arrivals.times", arrivals.times(count), count)
    raw_prompt_lengths = _exact_sequence(
        "prompt_lengths.sample", prompt_lengths.sample(count), count
    )
    raw_output_lengths = _exact_sequence(
        "output_lengths.sample", output_lengths.sample(count), count
    )

    arrival_decimals = tuple(_arrival_decimal(value) for value in raw_arrivals)
    if any(
        current < previous
        for previous, current in pairwise(arrival_decimals)
    ):
        raise ValueError("arrival times must be nondecreasing")
    arrivals_ps = tuple(_arrival_ps(value) for value in arrival_decimals)

    requests: list[GenerationRequest] = []
    for ordinal, (request_id, arrived_at_ps, prompt_raw, output_raw) in enumerate(
        zip(
            ids,
            arrivals_ps,
            raw_prompt_lengths,
            raw_output_lengths,
            strict=True,
        )
    ):
        prompt_length = _positive_length("prompt length", prompt_raw)
        output_length = _positive_length("output length", output_raw)
        prompt = tuple(prompt_builder(request_id, prompt_length, ordinal))
        if len(prompt) != prompt_length:
            raise ValueError(
                f"prompt builder returned {len(prompt)} tokens for request "
                f"{request_id!r}, expected {prompt_length}"
            )
        requests.append(
            GenerationRequest(
                ordinal=ordinal,
                request_id=request_id,
                arrived_at_ps=arrived_at_ps,
                prompt_token_ids=prompt,
                output_token_count=output_length,
            )
        )
    return tuple(requests)


@dataclass(frozen=True)
class TransportRequestObservation:
    """One transport's submission and streamed-token observations."""

    request_id: str
    submitted_at_ps: int
    token_completed_at_ps: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "token_completed_at_ps", tuple(self.token_completed_at_ps))


class OpenLoopTransport(Protocol):
    """Adapter transport that does not wait for one request before the next."""

    def run_open_loop(
        self, requests: tuple[GenerationRequest, ...]
    ) -> Iterable[TransportRequestObservation]: ...


@dataclass(frozen=True)
class ObservedRequestTiming:
    """Client-observed per-request timing, kept distinct from core metrics."""

    ordinal: int
    request_id: str
    arrived_at_ps: int
    submitted_at_ps: int
    first_token_at_ps: int
    completed_at_ps: int
    output_token_count: int
    submission_lateness_ps: int
    ttft_ps: int
    tpot_ps: Fraction | None


def _observation_integer(name: str, value: Any, request_id: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} for request {request_id!r} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} for request {request_id!r} must be nonnegative")
    return result


def reduce_transport_observations(
    requests: Sequence[GenerationRequest],
    observations: Iterable[TransportRequestObservation],
) -> tuple[ObservedRequestTiming, ...]:
    """Validate and reduce open-loop observations to exact TTFT and TPOT."""

    plan = tuple(requests)
    by_id: dict[str, GenerationRequest] = {}
    ordinals: set[int] = set()
    for request in plan:
        if not isinstance(request, GenerationRequest):
            raise TypeError("requests must contain GenerationRequest values")
        if request.request_id in by_id:
            raise ValueError(f"duplicate planned request ID {request.request_id!r}")
        if request.ordinal in ordinals:
            raise ValueError(f"duplicate planned request ordinal {request.ordinal}")
        by_id[request.request_id] = request
        ordinals.add(request.ordinal)

    observed: dict[str, TransportRequestObservation] = {}
    for observation in observations:
        if not isinstance(observation, TransportRequestObservation):
            raise TypeError("observations must contain TransportRequestObservation values")
        request_id = observation.request_id
        if request_id not in by_id:
            raise ValueError(f"foreign transport observation for request {request_id!r}")
        if request_id in observed:
            raise ValueError(f"duplicate transport observation for request {request_id!r}")
        observed[request_id] = observation

    missing = [request_id for request_id in by_id if request_id not in observed]
    if missing:
        raise ValueError(f"missing transport observations for request IDs {missing}")

    timings: list[ObservedRequestTiming] = []
    for request in sorted(plan, key=lambda item: item.ordinal):
        observation = observed[request.request_id]
        submitted = _observation_integer(
            "submitted_at_ps", observation.submitted_at_ps, request.request_id
        )
        if submitted < request.arrived_at_ps:
            raise ValueError(f"request {request.request_id!r} was submitted before arrival")
        tokens = tuple(
            _observation_integer("token completion", value, request.request_id)
            for value in observation.token_completed_at_ps
        )
        if len(tokens) != request.output_token_count:
            raise ValueError(
                f"request {request.request_id!r} observed {len(tokens)} output tokens, "
                f"expected {request.output_token_count}"
            )
        if tokens[0] < submitted:
            raise ValueError(
                f"request {request.request_id!r} completed a token before submission"
            )
        if any(current < previous for previous, current in pairwise(tokens)):
            raise ValueError(
                f"request {request.request_id!r} has decreasing token timestamps"
            )
        tpot = (
            None
            if len(tokens) == 1
            else Fraction(tokens[-1] - tokens[0], len(tokens) - 1)
        )
        timings.append(
            ObservedRequestTiming(
                ordinal=request.ordinal,
                request_id=request.request_id,
                arrived_at_ps=request.arrived_at_ps,
                submitted_at_ps=submitted,
                first_token_at_ps=tokens[0],
                completed_at_ps=tokens[-1],
                output_token_count=len(tokens),
                submission_lateness_ps=submitted - request.arrived_at_ps,
                ttft_ps=tokens[0] - request.arrived_at_ps,
                tpot_ps=tpot,
            )
        )
    return tuple(timings)
