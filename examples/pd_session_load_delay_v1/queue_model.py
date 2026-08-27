"""Frozen analytical queue model for the VLLM-39 load-delay study."""

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
OFFERED_LOADS = (250, 500, 1_000, 2_000, 4_000, 8_000)
PREFILL_SERVICE_PS = {8: 95_424_000, 16: 114_936_000}
POOL_RATIOS = ((1, 1), (1, 2), (2, 1))
PROMPT_LENGTHS = (8, 16)
CALIBRATION_CONFIGURATIONS = ((1, 1, 8), (1, 2, 8))
HELD_OUT_CONFIGURATIONS = (
    (1, 1, 16),
    (1, 2, 16),
    (2, 1, 8),
    (2, 1, 16),
)


def fraction_json(value: Fraction | int) -> dict[str, int]:
    """Render one exact rational using the repository curve convention."""

    result = Fraction(value)
    return {"numerator": result.numerator, "denominator": result.denominator}


def decode_capacity_requests_per_second(
    points: tuple[BatchServicePoint, ...],
    decode_engines: int,
) -> Fraction:
    """Return max-batch capacity for four decode steps per request."""

    service_ps = interpolate_batch_service_ps(points, MAX_BATCH_SIZE)
    return Fraction(
        decode_engines * MAX_BATCH_SIZE * PS_PER_SECOND,
        OUTPUT_TOKENS * service_ps,
    )


def predicted_batch_size(
    points: tuple[BatchServicePoint, ...],
    offered_load: int,
    decode_engines: int,
) -> int:
    """Map max-batch utilization to a deterministic occupancy bucket."""

    service_ps = interpolate_batch_service_ps(points, MAX_BATCH_SIZE)
    numerator = offered_load * OUTPUT_TOKENS * service_ps
    denominator = decode_engines * PS_PER_SECOND
    occupancy = -(-numerator // denominator)
    return min(MAX_BATCH_SIZE, max(1, occupancy))


def mean_overload_queue_wait_ps(
    points: tuple[BatchServicePoint, ...],
    offered_load: int,
    decode_engines: int,
) -> Fraction:
    """Finite D/D/c mean wait beyond the max-batch service knee."""

    max_service_ps = interpolate_batch_service_ps(points, MAX_BATCH_SIZE)
    service_interval = Fraction(
        OUTPUT_TOKENS * max_service_ps,
        MAX_BATCH_SIZE * decode_engines,
    )
    arrival_interval = Fraction(PS_PER_SECOND, offered_load)
    excess = max(Fraction(), service_interval - arrival_interval)
    return Fraction(REQUESTS_PER_CELL - 1, 2) * excess


def predict_point(
    points: tuple[BatchServicePoint, ...],
    *,
    prefill_engines: int,
    decode_engines: int,
    prompt_tokens: int,
    offered_load: int,
) -> dict[str, object]:
    """Predict one per-token delay and its pre-run uncertainty band."""

    if (prefill_engines, decode_engines) not in POOL_RATIOS:
        raise ValueError("pool ratio is outside the frozen study")
    if prompt_tokens not in PROMPT_LENGTHS:
        raise ValueError("prompt length is outside the frozen study")
    if offered_load not in OFFERED_LOADS:
        raise ValueError("offered load is outside the frozen study")
    batch_size = predicted_batch_size(points, offered_load, decode_engines)
    batch_service_ps = interpolate_batch_service_ps(points, batch_size)
    decode_service_total = Fraction(OUTPUT_TOKENS * batch_service_ps, batch_size)
    queue_wait = mean_overload_queue_wait_ps(points, offered_load, decode_engines)
    prefill_service = Fraction(PREFILL_SERVICE_PS[prompt_tokens])
    request_total = prefill_service + HANDOFF_PS + decode_service_total + queue_wait
    prediction = request_total / OUTPUT_TOKENS

    surface_uncertainty = decode_service_total * Fraction(15, 100)
    scheduler_phase_envelope = Fraction(
        interpolate_batch_service_ps(points, MAX_BATCH_SIZE),
        MAX_BATCH_SIZE,
    )
    queue_uncertainty = queue_wait * Fraction(1, 4)
    prefill_uncertainty = prefill_service * Fraction(1, 10)
    half_width = (
        surface_uncertainty
        + scheduler_phase_envelope
        + queue_uncertainty
        + prefill_uncertainty
    ) / OUTPUT_TOKENS
    lower = max(Fraction(1), prediction - half_width)
    upper = prediction + half_width
    return {
        "configuration": [prefill_engines, decode_engines, prompt_tokens],
        "offered_load_requests_per_second": offered_load,
        "predicted_batch_size": batch_size,
        "batch_service_ps": batch_service_ps,
        "batching_gain_service_per_request_ps": fraction_json(
            decode_service_total
        ),
        "scheduler_queue_wait_ps": fraction_json(queue_wait),
        "predicted_per_token_request_delay_ps": fraction_json(prediction),
        "prediction_band_ps": {
            "lower": fraction_json(lower),
            "upper": fraction_json(upper),
        },
    }


def direction(left: Fraction, right: Fraction) -> str:
    """Return an exact signed direction label."""

    if right > left:
        return "increase"
    if right < left:
        return "decrease"
    return "flat"


def predicted_segments(
    points: tuple[BatchServicePoint, ...],
    configuration: tuple[int, int, int],
) -> list[dict[str, object]]:
    """Predict every adjacent signed movement for one frozen curve."""

    prefill_engines, decode_engines, prompt_tokens = configuration
    rows = [
        predict_point(
            points,
            prefill_engines=prefill_engines,
            decode_engines=decode_engines,
            prompt_tokens=prompt_tokens,
            offered_load=load,
        )
        for load in OFFERED_LOADS
    ]
    predictions = [
        Fraction(
            row["predicted_per_token_request_delay_ps"]["numerator"],
            row["predicted_per_token_request_delay_ps"]["denominator"],
        )
        for row in rows
    ]
    return [
        {
            "configuration": list(configuration),
            "from_load": left_load,
            "to_load": right_load,
            "expected_direction": direction(left, right),
        }
        for left_load, right_load, left, right in zip(
            OFFERED_LOADS[:-1],
            OFFERED_LOADS[1:],
            predictions[:-1],
            predictions[1:],
            strict=True,
        )
    ]


__all__ = [
    "CALIBRATION_CONFIGURATIONS",
    "HANDOFF_PS",
    "HELD_OUT_CONFIGURATIONS",
    "MAX_BATCH_SIZE",
    "OFFERED_LOADS",
    "OUTPUT_TOKENS",
    "POOL_RATIOS",
    "PREFILL_SERVICE_PS",
    "PROMPT_LENGTHS",
    "PS_PER_SECOND",
    "REQUESTS_PER_CELL",
    "decode_capacity_requests_per_second",
    "direction",
    "fraction_json",
    "mean_overload_queue_wait_ps",
    "predict_point",
    "predicted_batch_size",
    "predicted_segments",
]
