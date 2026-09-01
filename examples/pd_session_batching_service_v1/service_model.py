"""Pre-run service-only predictor for the VLLM-42 successor study."""

from __future__ import annotations

from fractions import Fraction

from simllm.calibration.batch_service_surface import (
    BatchServicePoint,
    interpolate_batch_service_ps,
)

PS_PER_SECOND = 1_000_000_000_000
OUTPUT_TOKENS = 4
REQUESTS_PER_CELL = 64
MAX_BATCH_SIZE = 8
HANDOFF_PS = 100_000_000
PREFILL_SERVICE_PS = {8: 95_424_000, 16: 114_936_000}
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
TIMING_SCENARIOS = ("lower", "central", "upper")


def fraction_json(value: Fraction | int) -> dict[str, int]:
    """Render one exact rational without float conversion."""

    result = Fraction(value)
    return {"numerator": result.numerator, "denominator": result.denominator}


def fraction_from_json(value: dict[str, int]) -> Fraction:
    """Load one exact rational emitted by :func:`fraction_json`."""

    return Fraction(value["numerator"], value["denominator"])


def surface_cv_envelope_ppm(points: tuple[BatchServicePoint, ...]) -> int:
    """Return the conservative three-CV service-clock envelope in ppm."""

    if not points:
        raise ValueError("surface points must not be empty")
    maximum_cv_ppm = max(round(point.uncertainty_fraction * 1_000_000) for point in points)
    envelope = THREE_SIGMA_MULTIPLIER * maximum_cv_ppm
    if envelope >= 1_000_000:
        raise ValueError("surface uncertainty envelope must remain below one")
    return envelope


def central_service_ps(
    points: tuple[BatchServicePoint, ...],
    batch_size: int,
) -> int:
    """Return the independently measured and interpolated batch service."""

    if not 1 <= batch_size <= MAX_BATCH_SIZE:
        raise ValueError("batch size is outside the frozen scheduler limit")
    return interpolate_batch_service_ps(points, batch_size)


def timing_service_ps(
    points: tuple[BatchServicePoint, ...],
    batch_size: int,
    scenario: str,
) -> int:
    """Return service used only to place the next scheduler grant in time."""

    if scenario not in TIMING_SCENARIOS:
        raise ValueError(f"unknown timing scenario {scenario!r}")
    central = central_service_ps(points, batch_size)
    if scenario == "central":
        return central
    envelope_ppm = surface_cv_envelope_ppm(points)
    numerator = 1_000_000 + (envelope_ppm if scenario == "upper" else -envelope_ppm)
    scaled = Fraction(central * numerator, 1_000_000)
    if scenario == "lower":
        return scaled.numerator // scaled.denominator
    return -(-scaled.numerator // scaled.denominator)


def is_held_out(
    prefill_engines: int,
    decode_engines: int,
    offered_load: int,
) -> bool:
    """Return the frozen union split for successor disclosure order."""

    return offered_load in HELD_OUT_LOADS or (
        prefill_engines,
        decode_engines,
    ) in HELD_OUT_POOL_RATIOS


def _simulate_point(
    points: tuple[BatchServicePoint, ...],
    *,
    prefill_engines: int,
    decode_engines: int,
    prompt_tokens: int,
    offered_load: int,
    timing_scenario: str,
    phase_complete: bool,
) -> dict[str, object]:
    """Simulate only arrival phasing and scheduler-authored batch membership."""

    if (prefill_engines, decode_engines) not in POOL_RATIOS:
        raise ValueError("pool ratio is outside the frozen study")
    if prompt_tokens not in PROMPT_LENGTHS:
        raise ValueError("prompt length is outside the frozen study")
    if offered_load not in OFFERED_LOADS:
        raise ValueError("offered load is outside the frozen study")
    if timing_scenario not in TIMING_SCENARIOS:
        raise ValueError("timing scenario is outside the frozen study")

    interarrival_ps = PS_PER_SECOND // offered_load
    arrivals = [index * interarrival_ps for index in range(REQUESTS_PER_CELL)]
    prefill_queues: list[list[int]] = [[] for _ in range(prefill_engines)]
    decode_queues: list[list[int]] = [[] for _ in range(decode_engines)]
    pending_handoffs: list[tuple[int, int]] = []
    remaining_visits = [OUTPUT_TOKENS] * REQUESTS_PER_CELL
    decode_batches: list[tuple[int, ...]] = []
    prefill_batches: list[tuple[int, ...]] = []
    central_batch_services: list[int] = []
    next_arrival = 0
    cursor = 0
    now_ps = 0

    def admit_ready() -> None:
        nonlocal next_arrival
        while (
            next_arrival < REQUESTS_PER_CELL
            and arrivals[next_arrival] <= now_ps
        ):
            prefill_queues[next_arrival % prefill_engines].append(next_arrival)
            next_arrival += 1
        ready = [row for row in pending_handoffs if row[0] <= now_ps]
        if ready:
            for _, request_index in ready:
                decode_queues[request_index % decode_engines].append(request_index)
            pending_handoffs[:] = [
                row for row in pending_handoffs if row[0] > now_ps
            ]

    while any(remaining_visits):
        admit_ready()
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
            candidates = []
            if next_arrival < REQUESTS_PER_CELL:
                candidates.append(arrivals[next_arrival])
            candidates.extend(ready_at for ready_at, _ in pending_handoffs)
            if not candidates:
                raise RuntimeError("service model made no progress")
            now_ps = min(candidates)
            continue

        cursor = (active_driver_index + 1) % (prefill_engines + decode_engines)
        if active_driver_index < prefill_engines:
            prefill_index = active_driver_index
            batch = tuple(prefill_queues[prefill_index][:MAX_BATCH_SIZE])
            del prefill_queues[prefill_index][: len(batch)]
            prefill_batches.append(batch)
            prefill_service = PREFILL_SERVICE_PS[prompt_tokens] if phase_complete else 0
            handoff_service = HANDOFF_PS if phase_complete else 0
            now_ps += prefill_service
            ready_at = now_ps + handoff_service
            pending_handoffs.extend((ready_at, request_index) for request_index in batch)
            continue

        decode_index = active_driver_index - prefill_engines
        batch = tuple(decode_queues[decode_index][:MAX_BATCH_SIZE])
        decode_batches.append(batch)
        for request_index in batch:
            remaining_visits[request_index] -= 1
        decode_queues[decode_index] = [
            request_index
            for request_index in decode_queues[decode_index]
            if remaining_visits[request_index]
        ]
        central_batch_services.append(central_service_ps(points, len(batch)))
        now_ps += timing_service_ps(points, len(batch), timing_scenario)

    scheduled_visits = sum(map(len, decode_batches))
    if scheduled_visits != REQUESTS_PER_CELL * OUTPUT_TOKENS:
        raise RuntimeError("service model did not conserve decode visits")
    if sum(map(len, prefill_batches)) != REQUESTS_PER_CELL:
        raise RuntimeError("service model did not conserve prefill admissions")
    histogram = {
        str(batch_size): sum(len(batch) == batch_size for batch in decode_batches)
        for batch_size in range(1, MAX_BATCH_SIZE + 1)
    }
    return {
        "amortized_batch_service_per_token_ps": Fraction(
            sum(central_batch_services),
            scheduled_visits,
        ),
        "decode_batch_count": len(decode_batches),
        "decode_batch_histogram": histogram,
        "maximum_decode_batch_size": max(map(len, decode_batches)),
        "maximum_prefill_batch_size": max(map(len, prefill_batches)),
        "scheduled_decode_visits": scheduled_visits,
        "modeled_finish_ps": now_ps,
    }


def predict_point(
    points: tuple[BatchServicePoint, ...],
    *,
    prefill_engines: int,
    decode_engines: int,
    prompt_tokens: int,
    offered_load: int,
) -> dict[str, object]:
    """Predict one service field from independent service and arrival inputs."""

    scenarios = {
        scenario: _simulate_point(
            points,
            prefill_engines=prefill_engines,
            decode_engines=decode_engines,
            prompt_tokens=prompt_tokens,
            offered_load=offered_load,
            timing_scenario=scenario,
            phase_complete=True,
        )
        for scenario in TIMING_SCENARIOS
    }
    zero_phase = _simulate_point(
        points,
        prefill_engines=prefill_engines,
        decode_engines=decode_engines,
        prompt_tokens=prompt_tokens,
        offered_load=offered_load,
        timing_scenario="central",
        phase_complete=False,
    )
    central = scenarios["central"]
    service_values = [
        Fraction(row["amortized_batch_service_per_token_ps"])
        for row in scenarios.values()
    ]
    central_service = Fraction(central["amortized_batch_service_per_token_ps"])
    zero_phase_service = Fraction(
        zero_phase["amortized_batch_service_per_token_ps"]
    )
    return {
        "configuration": [prefill_engines, decode_engines, prompt_tokens],
        "offered_load_requests_per_second": offered_load,
        "interarrival_ps": PS_PER_SECOND // offered_load,
        "split": (
            "held-out"
            if is_held_out(prefill_engines, decode_engines, offered_load)
            else "non-held-out"
        ),
        "predicted_batch_service_per_token_ps": fraction_json(central_service),
        "batch_service_per_token_band_ps": {
            "lower": fraction_json(min(service_values)),
            "upper": fraction_json(max(service_values)),
        },
        "central_decode_batch_count": central["decode_batch_count"],
        "central_decode_batch_histogram": central["decode_batch_histogram"],
        "central_maximum_decode_batch_size": central["maximum_decode_batch_size"],
        "central_maximum_prefill_batch_size": central["maximum_prefill_batch_size"],
        "zero_phase_counterfactual_service_per_token_ps": fraction_json(
            zero_phase_service
        ),
        "phase_completion_signed_delta_ps": fraction_json(
            central_service - zero_phase_service
        ),
        "timing_scenario_services_per_token_ps": {
            scenario: fraction_json(
                Fraction(row["amortized_batch_service_per_token_ps"])
            )
            for scenario, row in scenarios.items()
        },
    }


def all_predictions(
    points: tuple[BatchServicePoint, ...],
) -> list[dict[str, object]]:
    """Return the complete frozen successor prediction registry."""

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
    ]


def physical_service_bounds_ps(
    points: tuple[BatchServicePoint, ...],
) -> tuple[Fraction, Fraction]:
    """Return service-per-visit floor and ceiling from measured batch work."""

    per_visit = [
        Fraction(central_service_ps(points, batch_size), batch_size)
        for batch_size in range(1, MAX_BATCH_SIZE + 1)
    ]
    return min(per_visit), max(per_visit)


__all__ = [
    "HANDOFF_PS",
    "HELD_OUT_LOADS",
    "HELD_OUT_POOL_RATIOS",
    "MAX_BATCH_SIZE",
    "OFFERED_LOADS",
    "OUTPUT_TOKENS",
    "POOL_RATIOS",
    "PREFILL_SERVICE_PS",
    "PROMPT_LENGTHS",
    "PS_PER_SECOND",
    "REQUESTS_PER_CELL",
    "THREE_SIGMA_MULTIPLIER",
    "TIMING_SCENARIOS",
    "all_predictions",
    "central_service_ps",
    "fraction_from_json",
    "fraction_json",
    "is_held_out",
    "physical_service_bounds_ps",
    "predict_point",
    "surface_cv_envelope_ppm",
    "timing_service_ps",
]
