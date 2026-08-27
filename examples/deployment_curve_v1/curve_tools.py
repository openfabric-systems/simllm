"""Anchor, calibration, scoring and uncertainty helpers for CORE-54."""

from __future__ import annotations

import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path
from typing import Any

ANCHOR_SCHEMA = "simllm-deployment-curve-anchor-freeze-v1"
POINT_INTERVAL_SCHEMA = "simllm-deployment-curve-point-interval-v1"
INTERVAL_METHOD = "deterministic-additive-interval-v1"
PS_PER_SECOND = 1_000_000_000_000


def load_json(path: Path) -> dict[str, Any]:
    """Read one UTF-8 JSON object."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def write_json(path: Path, value: object) -> None:
    """Write deterministic UTF-8 JSON with LF line endings."""

    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def fraction_json(value: Fraction) -> dict[str, int]:
    """Encode an exact fraction using the established curve representation."""

    return {"numerator": value.numerator, "denominator": value.denominator}


def as_fraction(value: object, name: str) -> Fraction:
    """Read an integer, finite decimal or numerator/denominator pair exactly."""

    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return Fraction(str(value))
    if isinstance(value, dict) and set(value) == {"numerator", "denominator"}:
        numerator = value["numerator"]
        denominator = value["denominator"]
        if type(numerator) is not int or type(denominator) is not int:
            raise TypeError(f"{name} fraction members must be integers")
        if denominator == 0:
            raise ValueError(f"{name} denominator must be nonzero")
        return Fraction(numerator, denominator)
    raise TypeError(f"{name} must be an integer, decimal or exact fraction")


def _anchor_map(freeze: dict[str, Any], role: str) -> dict[str, dict[str, Any]]:
    split_key = {
        "calibration": "calibration_anchor_ids",
        "held-out": "held_out_anchor_ids",
        "context-only": "context_only_anchor_ids",
    }[role]
    allowed = set(freeze["calibration_split"][split_key])
    return {anchor["id"]: anchor for anchor in freeze["anchors"] if anchor["id"] in allowed}


def validate_anchor_freeze(freeze: dict[str, Any]) -> None:
    """Validate the immutable disclosure split and curve contract."""

    if freeze.get("schema") != ANCHOR_SCHEMA:
        raise ValueError("anchor freeze schema disagrees")
    axis = freeze["axis_contract"]
    if (
        axis["x"]["quantity"] != "aggregated_output_throughput"
        or axis["x"]["direction"] != "rightward"
        or axis["y"]["quantity"] != "inverse_per_token_request_delay"
        or axis["y"]["direction"] != "upward"
        or axis["optimal_corner"] != "upper-right"
    ):
        raise ValueError("anchor freeze axis contract disagrees")
    acceptance = freeze["acceptance"]
    if acceptance["maximum_absolute_relative_error"] != 0.05:
        raise ValueError("held-out acceptance bar must remain 5 percent")
    if acceptance["pricing_dependent_bands_frozen_here"] is not False:
        raise ValueError("the scaffold freeze must not claim pricing bands")

    anchors = freeze["anchors"]
    ids = [anchor["id"] for anchor in anchors]
    if len(ids) != len(set(ids)):
        raise ValueError("anchor IDs must be unique")
    source_ids = set(freeze["sources"])
    if any(anchor["source_id"] not in source_ids for anchor in anchors):
        raise ValueError("every anchor must resolve to a frozen source")
    if any(not anchor.get("source_locator") for anchor in anchors):
        raise ValueError("every anchor must carry a source locator")

    split = freeze["calibration_split"]
    partition = (
        set(split["calibration_anchor_ids"]),
        set(split["held_out_anchor_ids"]),
        set(split["context_only_anchor_ids"]),
    )
    if any(
        left & right for index, left in enumerate(partition) for right in partition[index + 1 :]
    ):
        raise ValueError("anchor split sets must be disjoint")
    if set(ids) != set().union(*partition):
        raise ValueError("anchor split must cover every anchor exactly once")
    expected_roles = {
        "calibration_anchor_ids": "calibration",
        "held_out_anchor_ids": "held-out",
        "context_only_anchor_ids": "context-only",
    }
    by_id = {anchor["id"]: anchor for anchor in anchors}
    for split_name, expected_role in expected_roles.items():
        if any(by_id[anchor_id]["role"] != expected_role for anchor_id in split[split_name]):
            raise ValueError(f"{split_name} disagrees with anchor roles")


def validate_constant_declarations(
    declarations: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Validate physically bounded tuning declarations and selected values."""

    if not declarations:
        raise ValueError("at least one tunable constant must be declared")
    by_id: dict[str, dict[str, Any]] = {}
    for declaration in declarations:
        constant_id = declaration.get("id")
        if not isinstance(constant_id, str) or not constant_id.strip():
            raise ValueError("constant id must be a nonblank string")
        if constant_id in by_id:
            raise ValueError(f"duplicate constant id {constant_id!r}")
        if declaration.get("tunable") is not True:
            raise ValueError(f"{constant_id} must explicitly be tunable")
        if declaration.get("unit") != "ps":
            raise ValueError(f"{constant_id} must use picoseconds")
        provenance = declaration.get("provenance")
        if not isinstance(provenance, dict) or not all(
            provenance.get(key) for key in ("source", "locator", "physical_basis")
        ):
            raise ValueError(f"{constant_id} needs complete physical provenance")
        envelope = declaration.get("envelope")
        if not isinstance(envelope, dict):
            raise TypeError(f"{constant_id} needs an envelope")
        lower = as_fraction(envelope.get("lower"), f"{constant_id}.lower")
        upper = as_fraction(envelope.get("upper"), f"{constant_id}.upper")
        selected = as_fraction(declaration.get("selected"), f"{constant_id}.selected")
        if lower < 0 or upper <= lower:
            raise ValueError(f"{constant_id} envelope must be nonnegative and nonempty")
        if not lower <= selected <= upper:
            raise ValueError(
                f"{constant_id} selected value {selected} is outside [{lower}, {upper}]"
            )
        by_id[constant_id] = declaration
    return by_id


def fit_tunable_constants(
    freeze: dict[str, Any],
    declarations: list[dict[str, Any]],
    fit_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Fit independent linear constant projections on calibration anchors only.

    Each row defines ``prediction = baseline + sensitivity * constant`` for one
    observable. Multiple rows for the same constant are reduced by least squares
    on relative residuals. A fitted value outside its declared physical envelope
    is refused rather than clipped.
    """

    validate_anchor_freeze(freeze)
    constants = validate_constant_declarations(declarations)
    calibration = _anchor_map(freeze, "calibration")
    by_constant: dict[str, list[dict[str, Any]]] = {constant_id: [] for constant_id in constants}
    for row in fit_rows:
        constant_id = row.get("constant_id")
        if constant_id not in constants:
            raise ValueError(f"fit row names unknown constant {constant_id!r}")
        anchor_id = row.get("anchor_id")
        if anchor_id not in calibration:
            raise ValueError(f"fit row anchor {anchor_id!r} is not in the calibration split")
        by_constant[constant_id].append(row)

    fits = []
    accessed: set[str] = set()
    for constant_id, rows in by_constant.items():
        if not rows:
            raise ValueError(f"no calibration fit row supplied for {constant_id}")
        numerator = Fraction()
        denominator = Fraction()
        for index, row in enumerate(rows):
            anchor_id = row["anchor_id"]
            anchor = calibration[anchor_id]
            target = as_fraction(anchor["value"], f"{anchor_id}.value")
            baseline = as_fraction(row["baseline"], f"fit_rows[{index}].baseline")
            sensitivity = as_fraction(
                row["sensitivity_per_ps"],
                f"fit_rows[{index}].sensitivity_per_ps",
            )
            if sensitivity == 0:
                raise ValueError("fit sensitivity must be nonzero")
            normalized_sensitivity = sensitivity / target
            normalized_residual = (target - baseline) / target
            numerator += normalized_sensitivity * normalized_residual
            denominator += normalized_sensitivity * normalized_sensitivity
            accessed.add(anchor_id)
        fitted = numerator / denominator
        declaration = constants[constant_id]
        lower = as_fraction(declaration["envelope"]["lower"], "envelope.lower")
        upper = as_fraction(declaration["envelope"]["upper"], "envelope.upper")
        if not lower <= fitted <= upper:
            raise ValueError(f"fitted {constant_id} value {fitted} is outside [{lower}, {upper}]")
        fits.append(
            {
                "constant_id": constant_id,
                "fitted": fraction_json(fitted),
                "envelope": {
                    "lower": fraction_json(lower),
                    "upper": fraction_json(upper),
                },
                "calibration_anchor_ids": sorted({row["anchor_id"] for row in rows}),
                "disposition": "fitted-parameter-not-measurement",
            }
        )
    return {
        "schema": "simllm-deployment-constant-fit-v1",
        "accessed_anchor_ids": sorted(accessed),
        "forbidden_anchor_ids_accessed": [],
        "fits": fits,
    }


def score_held_out_predictions(
    freeze: dict[str, Any],
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Score exactly the held-out anchors without exposing calibration values."""

    validate_anchor_freeze(freeze)
    held_out = _anchor_map(freeze, "held-out")
    prediction_ids = [row.get("anchor_id") for row in predictions]
    if len(prediction_ids) != len(set(prediction_ids)):
        raise ValueError("held-out prediction IDs must be unique")
    if set(prediction_ids) != set(held_out):
        raise ValueError("scoring requires exactly the frozen held-out anchor IDs")
    bar = as_fraction(
        freeze["acceptance"]["maximum_absolute_relative_error"],
        "maximum_absolute_relative_error",
    )
    rows = []
    for prediction in predictions:
        anchor_id = prediction["anchor_id"]
        published = as_fraction(held_out[anchor_id]["value"], f"{anchor_id}.value")
        point = as_fraction(prediction["point"], f"{anchor_id}.point")
        lower = as_fraction(prediction.get("lower", prediction["point"]), f"{anchor_id}.lower")
        upper = as_fraction(prediction.get("upper", prediction["point"]), f"{anchor_id}.upper")
        if not 0 < lower <= point <= upper:
            raise ValueError(f"{anchor_id} prediction interval is invalid")
        accepted_lower = published * (1 - bar)
        accepted_upper = published * (1 + bar)
        intersects = lower <= accepted_upper and upper >= accepted_lower
        rows.append(
            {
                "anchor_id": anchor_id,
                "published": fraction_json(published),
                "predicted": {
                    "lower": fraction_json(lower),
                    "point": fraction_json(point),
                    "upper": fraction_json(upper),
                },
                "absolute_relative_error": fraction_json(abs(point - published) / published),
                "acceptance_interval": {
                    "lower": fraction_json(accepted_lower),
                    "upper": fraction_json(accepted_upper),
                },
                "interval_intersects_acceptance": intersects,
            }
        )
    return {
        "schema": "simllm-deployment-held-out-score-v1",
        "status": "PASS"
        if all(row["interval_intersects_acceptance"] for row in rows)
        else "REFUTED",
        "accessed_anchor_ids": sorted(prediction_ids),
        "forbidden_anchor_ids_accessed": [],
        "rows": rows,
    }


def _relative_delta(point: Fraction, width: object, name: str) -> Fraction:
    half_width = as_fraction(width, name)
    if half_width < 0:
        raise ValueError(f"{name} must be nonnegative")
    return point * half_width


def add_symmetric_relative_spread(
    lower: Fraction,
    point: Fraction,
    upper: Fraction,
    relative_half_width: object,
    name: str,
) -> tuple[Fraction, Fraction, Fraction]:
    """Minkowski-add one symmetric point-relative spread to an interval."""

    if not 0 < lower <= point <= upper:
        raise ValueError(f"{name} base interval must be positive and ordered")
    delta = _relative_delta(point, relative_half_width, name)
    widened_lower = lower - delta
    widened_upper = upper + delta
    if widened_lower <= 0:
        raise ValueError(f"{name} spread crossed a nonphysical zero bound")
    return widened_lower, widened_upper, delta


def propagate_curve_interval(
    point: dict[str, Any],
    uncertainty: dict[str, Any],
    declarations: list[dict[str, Any]],
) -> dict[str, Any]:
    """Propagate record, distribution and constant intervals to both axes.

    The method is deterministic interval arithmetic. Record bounds establish
    the first interval. Distribution half-widths and tuned-constant envelope
    effects are additive intervals whose Minkowski sum forms the final hull.
    Inverting delay reverses its endpoints exactly.
    """

    constants = validate_constant_declarations(declarations)
    throughput = as_fraction(
        point["aggregated_output_throughput_tokens_per_second"],
        "point.throughput",
    )
    delay = as_fraction(point["per_token_request_delay_ps"], "point.delay")
    if throughput <= 0 or delay <= 0:
        raise ValueError("curve point axes must be positive")

    record = uncertainty.get("record_bounds", {})
    throughput_bounds = record.get("aggregated_output_throughput_tokens_per_second")
    delay_bounds = record.get("per_token_request_delay_ps")
    if throughput_bounds is None:
        throughput_lower = throughput_upper = throughput
    else:
        throughput_lower = as_fraction(throughput_bounds["lower"], "record.throughput.lower")
        throughput_upper = as_fraction(throughput_bounds["upper"], "record.throughput.upper")
    if delay_bounds is None:
        delay_lower = delay_upper = delay
    else:
        delay_lower = as_fraction(delay_bounds["lower"], "record.delay.lower")
        delay_upper = as_fraction(delay_bounds["upper"], "record.delay.upper")
    if not 0 < throughput_lower <= throughput <= throughput_upper:
        raise ValueError("record throughput bounds must contain the point")
    if not 0 < delay_lower <= delay <= delay_upper:
        raise ValueError("record delay bounds must contain the point")

    contributions = [
        {
            "source_kind": "record-bounds",
            "source_id": record.get("source_id", "exact-curve-record"),
            "throughput_delta": {
                "lower": fraction_json(throughput_lower - throughput),
                "upper": fraction_json(throughput_upper - throughput),
            },
            "delay_ps_delta": {
                "lower": fraction_json(delay_lower - delay),
                "upper": fraction_json(delay_upper - delay),
            },
        }
    ]

    for index, spread in enumerate(uncertainty.get("distribution_spreads", [])):
        throughput_lower, throughput_upper, x_delta = add_symmetric_relative_spread(
            throughput_lower,
            throughput,
            throughput_upper,
            spread["throughput_relative_half_width"],
            f"distribution_spreads[{index}].throughput_relative_half_width",
        )
        delay_lower, delay_upper, d_delta = add_symmetric_relative_spread(
            delay_lower,
            delay,
            delay_upper,
            spread["delay_relative_half_width"],
            f"distribution_spreads[{index}].delay_relative_half_width",
        )
        contributions.append(
            {
                "source_kind": "distribution-spread",
                "source_id": spread["source_id"],
                "throughput_delta": {
                    "lower": fraction_json(-x_delta),
                    "upper": fraction_json(x_delta),
                },
                "delay_ps_delta": {
                    "lower": fraction_json(-d_delta),
                    "upper": fraction_json(d_delta),
                },
            }
        )

    for index, effect in enumerate(uncertainty.get("tuned_constant_envelopes", [])):
        constant_id = effect["constant_id"]
        if constant_id not in constants:
            raise ValueError(f"uncertainty effect names unknown constant {constant_id!r}")
        declaration = constants[constant_id]
        selected = as_fraction(declaration["selected"], f"{constant_id}.selected")
        lower = as_fraction(declaration["envelope"]["lower"], f"{constant_id}.lower")
        upper = as_fraction(declaration["envelope"]["upper"], f"{constant_id}.upper")
        x_sensitivity = as_fraction(
            effect["throughput_tokens_per_second_per_ps"],
            f"tuned_constant_envelopes[{index}].throughput_sensitivity",
        )
        d_sensitivity = as_fraction(
            effect["delay_ps_per_ps"],
            f"tuned_constant_envelopes[{index}].delay_sensitivity",
        )
        x_endpoints = (
            (lower - selected) * x_sensitivity,
            (upper - selected) * x_sensitivity,
        )
        d_endpoints = (
            (lower - selected) * d_sensitivity,
            (upper - selected) * d_sensitivity,
        )
        x_low, x_high = min(x_endpoints), max(x_endpoints)
        d_low, d_high = min(d_endpoints), max(d_endpoints)
        throughput_lower += x_low
        throughput_upper += x_high
        delay_lower += d_low
        delay_upper += d_high
        contributions.append(
            {
                "source_kind": "tuned-constant-envelope",
                "source_id": constant_id,
                "throughput_delta": {
                    "lower": fraction_json(x_low),
                    "upper": fraction_json(x_high),
                },
                "delay_ps_delta": {
                    "lower": fraction_json(d_low),
                    "upper": fraction_json(d_high),
                },
            }
        )

    if throughput_lower <= 0 or delay_lower <= 0:
        raise ValueError("propagated curve interval crossed a nonphysical zero axis")
    inverse_point = Fraction(PS_PER_SECOND, 1) / delay
    inverse_lower = Fraction(PS_PER_SECOND, 1) / delay_upper
    inverse_upper = Fraction(PS_PER_SECOND, 1) / delay_lower
    return {
        "schema": POINT_INTERVAL_SCHEMA,
        "method": INTERVAL_METHOD,
        "aggregated_output_throughput_tokens_per_second": {
            "lower": fraction_json(throughput_lower),
            "point": fraction_json(throughput),
            "upper": fraction_json(throughput_upper),
        },
        "per_token_request_delay_ps": {
            "lower": fraction_json(delay_lower),
            "point": fraction_json(delay),
            "upper": fraction_json(delay_upper),
        },
        "inverse_per_token_request_delay_tokens_per_second": {
            "lower": fraction_json(inverse_lower),
            "point": fraction_json(inverse_point),
            "upper": fraction_json(inverse_upper),
        },
        "source_contributions": contributions,
    }
