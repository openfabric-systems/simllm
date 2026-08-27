from __future__ import annotations

import importlib.util
from fractions import Fraction
from pathlib import Path

from simllm.calibration.batch_service_surface import BatchServicePoint

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STUDY_DIR = REPOSITORY_ROOT / "examples" / "pd_session_load_delay_v1"


def _model():
    spec = importlib.util.spec_from_file_location(
        "pd_session_load_delay_queue_model",
        STUDY_DIR / "queue_model.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _points() -> tuple[BatchServicePoint, ...]:
    return (
        BatchServicePoint(1, 1_110_576_000, 0.004232, "1" * 64),
        BatchServicePoint(8, 1_892_831_500, 0.001538, "8" * 64),
    )


def test_surface_derived_knees_are_between_frozen_load_points() -> None:
    model = _model()

    one = model.decode_capacity_requests_per_second(_points(), 1)
    two = model.decode_capacity_requests_per_second(_points(), 2)

    assert Fraction(1_000) < one < Fraction(2_000)
    assert Fraction(2_000) < two < Fraction(4_000)
    assert two == 2 * one


def test_batching_gain_precedes_overload_queue_wait() -> None:
    model = _model()
    points = _points()

    batches = [
        model.predicted_batch_size(points, load, 1)
        for load in model.OFFERED_LOADS
    ]
    waits = [
        model.mean_overload_queue_wait_ps(points, load, 1)
        for load in model.OFFERED_LOADS
    ]

    assert batches == [2, 4, 8, 8, 8, 8]
    assert waits[:3] == [0, 0, 0]
    assert waits[3:] == sorted(waits[3:])
    assert waits[3] > 0


def test_each_curve_has_five_frozen_signed_segments() -> None:
    model = _model()
    points = _points()
    configurations = tuple(
        (prefill, decode, prompt)
        for prefill, decode in model.POOL_RATIOS
        for prompt in model.PROMPT_LENGTHS
    )

    rows = [
        row
        for configuration in configurations
        for row in model.predicted_segments(points, configuration)
    ]

    assert len(rows) == 30
    assert {row["expected_direction"] for row in rows} == {
        "decrease",
        "increase",
    }


def test_held_out_band_contains_its_surface_queue_prediction() -> None:
    model = _model()
    row = model.predict_point(
        _points(),
        prefill_engines=2,
        decode_engines=1,
        prompt_tokens=16,
        offered_load=2_000,
    )
    prediction = Fraction(**row["predicted_per_token_request_delay_ps"])
    lower = Fraction(**row["prediction_band_ps"]["lower"])
    upper = Fraction(**row["prediction_band_ps"]["upper"])

    assert lower < prediction < upper
    assert Fraction(**row["scheduler_queue_wait_ps"]) > 0
