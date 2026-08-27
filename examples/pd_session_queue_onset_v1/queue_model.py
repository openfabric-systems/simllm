"""Pre-run deterministic bulk-service queue model for VLLM-41."""

from __future__ import annotations

from fractions import Fraction
from itertools import pairwise

from simllm.calibration.batch_service_surface import (
    BatchServicePoint,
    interpolate_batch_service_ps,
)

PS_PER_SECOND = 1_000_000_000_000
OUTPUT_TOKENS = 4
REQUESTS_PER_CELL = 64
MAX_BATCH_SIZE = 8
OFFERED_LOADS = (
    50,
    100,
    150,
    175,
    200,
    210,
    220,
    225,
    230,
    235,
    240,
    245,
    250,
)
POOL_RATIOS = ((1, 1), (1, 2), (2, 1))
PROMPT_LENGTHS = (8, 16)
HELD_OUT_LOADS = (240,)
HELD_OUT_POOL_RATIOS = ((2, 1),)
THREE_SIGMA_MULTIPLIER = 3
SURFACE_SCENARIOS = ("lower", "central", "upper")


def fraction_json(value: Fraction | int) -> dict[str, int]:
    """Render one exact rational without float conversion."""

    result = Fraction(value)
    return {"numerator": result.numerator, "denominator": result.denominator}


def fraction_from_json(value: dict[str, int]) -> Fraction:
    """Load one exact rational emitted by :func:`fraction_json`."""

    return Fraction(value["numerator"], value["denominator"])


def surface_cv_envelope_ppm(points: tuple[BatchServicePoint, ...]) -> int:
    """Return the conservative three-CV surface envelope in ppm."""

    if not points:
        raise ValueError("surface points must not be empty")
    maximum_cv_ppm = max(round(point.uncertainty_fraction * 1_000_000) for point in points)
    envelope = THREE_SIGMA_MULTIPLIER * maximum_cv_ppm
    if envelope >= 1_000_000:
        raise ValueError("surface uncertainty envelope must remain below one")
    return envelope


def scenario_service_ps(
    points: tuple[BatchServicePoint, ...],
    batch_size: int,
    scenario: str,
) -> int:
    """Return an integer service time under one frozen surface scenario."""

    if scenario not in SURFACE_SCENARIOS:
        raise ValueError(f"unknown surface scenario {scenario!r}")
    central = interpolate_batch_service_ps(points, batch_size)
    if scenario == "central":
        return central
    envelope_ppm = surface_cv_envelope_ppm(points)
    numerator = 1_000_000 + (envelope_ppm if scenario == "upper" else -envelope_ppm)
    scaled = Fraction(central * numerator, 1_000_000)
    if scenario == "lower":
        return scaled.numerator // scaled.denominator
    return -(-scaled.numerator // scaled.denominator)


def isolated_queue_onset_requests_per_second(
    points: tuple[BatchServicePoint, ...],
    scenario: str = "central",
) -> Fraction:
    """Return the rate at which one-request decode demand equals interarrival."""

    return Fraction(
        PS_PER_SECOND,
        OUTPUT_TOKENS * scenario_service_ps(points, 1, scenario),
    )


def onset_rate_band(
    points: tuple[BatchServicePoint, ...],
) -> dict[str, dict[str, int]]:
    """Return the surface-only onset-rate band, ordered low to high."""

    return {
        "lower": fraction_json(
            isolated_queue_onset_requests_per_second(points, "upper")
        ),
        "central": fraction_json(
            isolated_queue_onset_requests_per_second(points, "central")
        ),
        "upper": fraction_json(
            isolated_queue_onset_requests_per_second(points, "lower")
        ),
    }


def _simulate_point(
    points: tuple[BatchServicePoint, ...],
    *,
    prefill_engines: int,
    decode_engines: int,
    offered_load: int,
    scenario: str,
) -> dict[str, Fraction | int]:
    """Drive the frozen zero-cost-boundary bulk-service queue abstraction."""

    if (prefill_engines, decode_engines) not in POOL_RATIOS:
        raise ValueError("pool ratio is outside the frozen study")
    if offered_load not in OFFERED_LOADS:
        raise ValueError("offered load is outside the frozen study")
    interarrival_ps = PS_PER_SECOND // offered_load
    arrivals = [index * interarrival_ps for index in range(REQUESTS_PER_CELL)]
    prefill_queues: list[list[int]] = [[] for _ in range(prefill_engines)]
    decode_queues: list[list[int]] = [[] for _ in range(decode_engines)]
    remaining_visits = [OUTPUT_TOKENS] * REQUESTS_PER_CELL
    prefill_waits: list[int | None] = [None] * REQUESTS_PER_CELL
    decode_waits: list[int | None] = [None] * REQUESTS_PER_CELL
    decode_ready_at: list[int | None] = [None] * REQUESTS_PER_CELL
    batch_sizes: list[int] = []
    batch_services: list[int] = []
    next_arrival = 0
    cursor = 0
    now_ps = 0

    def admit_arrivals() -> int:
        current = next_arrival
        while current < REQUESTS_PER_CELL and arrivals[current] <= now_ps:
            prefill_queues[current % prefill_engines].append(current)
            current += 1
        return current

    while any(remaining_visits):
        next_arrival = admit_arrivals()
        active_driver_index: int | None = None
        for offset in range(prefill_engines + decode_engines):
            driver_index = (cursor + offset) % (prefill_engines + decode_engines)
            queue = (
                prefill_queues[driver_index]
                if driver_index < prefill_engines
                else decode_queues[driver_index - prefill_engines]
            )
            if queue:
                active_driver_index = driver_index
                break
        if active_driver_index is None:
            if next_arrival >= REQUESTS_PER_CELL:
                raise RuntimeError("queue model made no progress")
            now_ps = arrivals[next_arrival]
            continue

        cursor = (active_driver_index + 1) % (prefill_engines + decode_engines)
        if active_driver_index < prefill_engines:
            prefill_index = active_driver_index
            batch = prefill_queues[prefill_index][:MAX_BATCH_SIZE]
            del prefill_queues[prefill_index][: len(batch)]
            for request_index in batch:
                prefill_waits[request_index] = now_ps - arrivals[request_index]
                decode_ready_at[request_index] = now_ps
                decode_queues[request_index % decode_engines].append(request_index)
            continue

        decode_index = active_driver_index - prefill_engines
        batch = decode_queues[decode_index][:MAX_BATCH_SIZE]
        for request_index in batch:
            ready_at = decode_ready_at[request_index]
            if ready_at is None:
                raise RuntimeError("decode request has no modeled ready time")
            if decode_waits[request_index] is None:
                decode_waits[request_index] = now_ps - ready_at
            remaining_visits[request_index] -= 1
        decode_queues[decode_index] = [
            request_index
            for request_index in decode_queues[decode_index]
            if remaining_visits[request_index]
        ]
        service_ps = scenario_service_ps(points, len(batch), scenario)
        batch_sizes.append(len(batch))
        batch_services.append(service_ps)
        now_ps += service_ps

    if any(value is None for value in (*prefill_waits, *decode_waits)):
        raise RuntimeError("queue model left an admission wait unset")
    exact_prefill_waits = [int(value) for value in prefill_waits if value is not None]
    exact_decode_waits = [int(value) for value in decode_waits if value is not None]
    scheduled_visits = sum(batch_sizes)
    if scheduled_visits != REQUESTS_PER_CELL * OUTPUT_TOKENS:
        raise RuntimeError("queue model did not conserve decode visits")
    mean_prefill_wait = Fraction(sum(exact_prefill_waits), REQUESTS_PER_CELL)
    mean_decode_wait = Fraction(sum(exact_decode_waits), REQUESTS_PER_CELL)
    return {
        "mean_prefill_queue_ps": mean_prefill_wait,
        "mean_decode_admission_wait_ps": mean_decode_wait,
        "mean_scheduler_queue_wait_ps": mean_prefill_wait + mean_decode_wait,
        "amortized_decode_batch_service_per_token_ps": Fraction(
            sum(batch_services), scheduled_visits
        ),
        "maximum_decode_batch_size": max(batch_sizes),
        "decode_batch_count": len(batch_sizes),
    }


def predict_point(
    points: tuple[BatchServicePoint, ...],
    *,
    prefill_engines: int,
    decode_engines: int,
    prompt_tokens: int,
    offered_load: int,
) -> dict[str, object]:
    """Predict separate service and admission-wait components with bands."""

    if prompt_tokens not in PROMPT_LENGTHS:
        raise ValueError("prompt length is outside the frozen study")
    scenarios = {
        scenario: _simulate_point(
            points,
            prefill_engines=prefill_engines,
            decode_engines=decode_engines,
            offered_load=offered_load,
            scenario=scenario,
        )
        for scenario in SURFACE_SCENARIOS
    }
    central = scenarios["central"]
    service_values = [
        Fraction(row["amortized_decode_batch_service_per_token_ps"])
        for row in scenarios.values()
    ]
    wait_values = [
        Fraction(row["mean_scheduler_queue_wait_ps"])
        for row in scenarios.values()
    ]
    phase_envelope_ps = 2 * scenario_service_ps(
        points, MAX_BATCH_SIZE, "upper"
    )
    return {
        "configuration": [prefill_engines, decode_engines, prompt_tokens],
        "offered_load_requests_per_second": offered_load,
        "interarrival_ps": PS_PER_SECOND // offered_load,
        "predicted_mean_prefill_queue_ps": fraction_json(
            Fraction(central["mean_prefill_queue_ps"])
        ),
        "predicted_mean_decode_admission_wait_ps": fraction_json(
            Fraction(central["mean_decode_admission_wait_ps"])
        ),
        "predicted_mean_scheduler_queue_wait_ps": fraction_json(
            Fraction(central["mean_scheduler_queue_wait_ps"])
        ),
        "scheduler_queue_wait_band_ps": {
            "lower": fraction_json(max(Fraction(), min(wait_values) - phase_envelope_ps)),
            "upper": fraction_json(max(wait_values) + phase_envelope_ps),
        },
        "predicted_batch_service_per_token_ps": fraction_json(
            Fraction(central["amortized_decode_batch_service_per_token_ps"])
        ),
        "batch_service_per_token_band_ps": {
            "lower": fraction_json(min(service_values)),
            "upper": fraction_json(max(service_values)),
        },
        "phase_envelope_ps": phase_envelope_ps,
        "predicted_maximum_decode_batch_size": central[
            "maximum_decode_batch_size"
        ],
        "predicted_decode_batch_count": central["decode_batch_count"],
    }


def _component_total(point: dict[str, object]) -> Fraction:
    service = fraction_from_json(point["predicted_batch_service_per_token_ps"])
    wait = fraction_from_json(point["predicted_mean_scheduler_queue_wait_ps"])
    return service + wait / OUTPUT_TOKENS


def predicted_segments(
    points: tuple[BatchServicePoint, ...],
    configuration: tuple[int, int, int],
    *,
    scenario: str = "central",
) -> list[dict[str, object]]:
    """Classify adjacent segments by separate wait and service deltas."""

    prefill_engines, decode_engines, prompt_tokens = configuration
    if prompt_tokens not in PROMPT_LENGTHS:
        raise ValueError("prompt length is outside the frozen study")
    rows = []
    for load in OFFERED_LOADS:
        modeled = _simulate_point(
            points,
            prefill_engines=prefill_engines,
            decode_engines=decode_engines,
            offered_load=load,
            scenario=scenario,
        )
        rows.append(
            {
                "load": load,
                "wait": Fraction(modeled["mean_scheduler_queue_wait_ps"]),
                "service": Fraction(
                    modeled["amortized_decode_batch_service_per_token_ps"]
                ),
            }
        )
    segments = []
    for left, right in pairwise(rows):
        wait_delta = right["wait"] - left["wait"]
        service_delta = right["service"] - left["service"]
        wait_delta_per_token = wait_delta / OUTPUT_TOKENS
        net_component_delta = wait_delta_per_token + service_delta
        segments.append(
            {
                "configuration": list(configuration),
                "from_load": left["load"],
                "to_load": right["load"],
                "predicted_scheduler_wait_delta_ps": fraction_json(wait_delta),
                "predicted_scheduler_wait_delta_per_token_ps": fraction_json(
                    wait_delta_per_token
                ),
                "predicted_batch_service_per_token_delta_ps": fraction_json(
                    service_delta
                ),
                "predicted_component_total_delta_per_token_ps": fraction_json(
                    net_component_delta
                ),
                "queue_dominated": wait_delta > 0 and net_component_delta > 0,
            }
        )
    return segments


def first_queue_dominated_segment(
    points: tuple[BatchServicePoint, ...],
    configuration: tuple[int, int, int],
    *,
    scenario: str = "central",
) -> tuple[int, int] | None:
    """Return the first segment whose wait increase outweighs batching gain."""

    return next(
        (
            (row["from_load"], row["to_load"])
            for row in predicted_segments(points, configuration, scenario=scenario)
            if row["queue_dominated"]
        ),
        None,
    )


def predicted_onset_segments(
    points: tuple[BatchServicePoint, ...],
) -> dict[str, object]:
    """Freeze central and uncertainty-admitted first onset segments."""

    reference = (1, 1, 8)
    by_scenario = {
        scenario: first_queue_dominated_segment(
            points, reference, scenario=scenario
        )
        for scenario in SURFACE_SCENARIOS
    }
    admitted = sorted({segment for segment in by_scenario.values() if segment})
    return {
        "reference_configuration": list(reference),
        "central": list(by_scenario["central"] or ()),
        "by_surface_scenario": {
            scenario: None if segment is None else list(segment)
            for scenario, segment in by_scenario.items()
        },
        "inclusive_admitted_segments": [list(segment) for segment in admitted],
    }


def held_out_points(
    points: tuple[BatchServicePoint, ...],
) -> list[dict[str, object]]:
    """Return the union of the frozen held-out load and pool-ratio cells."""

    return [
        predict_point(
            points,
            prefill_engines=prefill,
            decode_engines=decode,
            prompt_tokens=prompt,
            offered_load=load,
        )
        for prefill, decode in POOL_RATIOS
        for prompt in PROMPT_LENGTHS
        for load in OFFERED_LOADS
        if load in HELD_OUT_LOADS or (prefill, decode) in HELD_OUT_POOL_RATIOS
    ]


__all__ = [
    "HELD_OUT_LOADS",
    "HELD_OUT_POOL_RATIOS",
    "MAX_BATCH_SIZE",
    "OFFERED_LOADS",
    "OUTPUT_TOKENS",
    "POOL_RATIOS",
    "PROMPT_LENGTHS",
    "PS_PER_SECOND",
    "REQUESTS_PER_CELL",
    "SURFACE_SCENARIOS",
    "THREE_SIGMA_MULTIPLIER",
    "first_queue_dominated_segment",
    "fraction_from_json",
    "fraction_json",
    "held_out_points",
    "isolated_queue_onset_requests_per_second",
    "onset_rate_band",
    "predict_point",
    "predicted_onset_segments",
    "predicted_segments",
    "scenario_service_ps",
    "surface_cv_envelope_ppm",
]
