"""Score the B200 NVLink envelope measurements against the frozen cells.

Every bound below is transcribed from ``expectations.md`` and
``expectations.json``, which were committed before the benchmark lane existed.
This script reads the stage results, evaluates each frozen cell the available
data allow, keeps the evidence classes apart, and writes ``scored.json``. It
never edits the freeze and it never imports torch.

Usage::

    python score_expectations.py --measurements measurements

Evidence classes, from the freeze: the scored denominator is the holdout rows
of E2, E3 and E5. E1 and E8 are fatal and unscored, E4 is reported and decides
between validation and refit, E6 and E7 are structural. Counts in different
classes are never added, and one violated fatal guard voids the run.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

STUDY = "b200_nvlink_envelope_v1"
SCHEMA = "simllm-b200-nvlink-envelope-scored-v1"
NEW_PROFILE_ID = "b200-nccl-2.27-local-firstparty-v1"

PICOSECONDS_PER_SECOND = 1_000_000_000_000
FIT_WINDOW_BYTES = (1_048_576, 1_073_741_824)
ASYMPTOTE_HOLDOUT_BYTES = 67_108_864
REFIT_HOLDOUT_BYTES = 4_096
REFIT_ROWS_BYTES = (8, 1_024, 2_048, 8_192, 16_384, 32_768, 65_536, 131_072, 262_144)
BEFORE_ERROR_ROWS_BYTES = (8, 1_024, 4_096, 65_536, 262_144)
R2_MIN = 0.99
HOLDOUT_TOLERANCE_FRACTION = 0.10
TOLERANCE_FLOOR_PS = 1_000_000
UNIDIRECTIONAL_CEILING_BPS = 900_000_000_000
BIDIRECTIONAL_CEILING_BPS = 1_800_000_000_000
BUSBW_CEILING_BPS = 900_000_000_000
MIN_TIMED_ROW_NS = 1_000
P1_BETA_BAND_BPS = (500_000_000_000, 900_000_000_000)
P2_BETA_BAND_BPS = (400_000_000_000, 900_000_000_000)
P1_SMALL_PAYLOAD_BAND_US = (1.0, 30.0)
P2_SMALL_PAYLOAD_BAND_US = (3.0, 60.0)
SOURCE_PAYLOAD_BYTES = (8, 262_144)
PROPAGATION_REFERENCE_PS = 2_000_000
ASYMPTOTE_FRACTION = 0.90
SERIALIZER_QUARTER = 0.25
ALL_PAIRS_SPREAD_MAX = 0.15
DISJOINT_SLOWDOWN_MAX = 0.15
CONCURRENT_PAYLOAD_BYTES = 67_108_864
ALL_PAIRS_STRUCTURE_BYTES = 16_777_216

IDENTITY_GUARDS = (
    ("E8-placement-records", "the five vLLM reference manifest digests of the PLACE-13 freeze"),
    ("E8-floor-study-check", "the tracked results of examples/collective_latency_floor_v1"),
    ("E8-existing-tests", "every existing test"),
    ("E8-profile-by-name", "b200-nccl-2.27-local-v1 resolves to the same constants"),
)


@dataclass
class Outcome:
    """One evaluated cell, carrying the evidence class it belongs to."""

    ident: str
    cls: str
    passed: bool | None
    detail: str
    observed: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.ident,
            "class": self.cls,
            "passed": self.passed,
            "detail": self.detail,
            "observed": self.observed,
        }


def ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def busbw_factor(op: str, width: int) -> float:
    if op == "all_reduce":
        return 2.0 * (width - 1) / width
    return float(width - 1) / width


def endpoint_bytes(width: int, payload_bytes: int) -> int:
    """Return the ring endpoint load the floor study charges."""

    return 2 * (width - 1) * payload_bytes // width


def tolerance_ps(observed_ps: float) -> float:
    """Return the frozen ``max(10 percent, 1 microsecond)`` tolerance."""

    return max(float(TOLERANCE_FLOOR_PS), abs(observed_ps) * HOLDOUT_TOLERANCE_FRACTION)


def ols_fit(sizes: list[float], seconds: list[float]) -> tuple[float, float, float]:
    """Return ``(alpha_seconds, beta_bytes_per_second, r_squared)``.

    The model is ``t = alpha + S / beta`` fitted by ordinary least squares of
    ``t`` on ``S``, the same form the A100 envelope used.
    """

    count = len(sizes)
    mean_x = sum(sizes) / count
    mean_y = sum(seconds) / count
    sxx = sum((x - mean_x) ** 2 for x in sizes)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(sizes, seconds))
    slope = sxy / sxx if sxx > 0 else float("nan")
    alpha = mean_y - slope * mean_x
    predicted = [alpha + slope * x for x in sizes]
    ss_res = sum((y - p) ** 2 for y, p in zip(seconds, predicted))
    ss_tot = sum((y - mean_y) ** 2 for y in seconds)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    beta = 1.0 / slope if slope > 0 else float("inf")
    return alpha, beta, r_squared


def requested_bytes(row: dict[str, Any]) -> int:
    return int(row.get("requested_bytes", row["bytes"]))


def measured(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("status") == "measured"]


def copy_series(result: dict[str, Any], cell: str, direction: str) -> list[dict[str, Any]]:
    rows = [
        row
        for row in measured(result.get("p1", []))
        if row.get("cell") == cell and row.get("direction") == direction
    ]
    return sorted(rows, key=requested_bytes)


def p2_series(result: dict[str, Any], cell: str, direction: str) -> list[dict[str, Any]]:
    rows = [
        row
        for row in measured(result.get("p2", []))
        if row.get("cell") == cell and row.get("direction") == direction
    ]
    return sorted(rows, key=requested_bytes)


def collective_series(result: dict[str, Any], op: str, width: int) -> list[dict[str, Any]]:
    rows = [
        row
        for row in measured(result.get("p3", []))
        if row.get("op") == op and row.get("width") == width
    ]
    return sorted(rows, key=requested_bytes)


def point(rows: list[dict[str, Any]], size_bytes: int) -> dict[str, Any] | None:
    for row in rows:
        if requested_bytes(row) == size_bytes:
            return row
    return None


def fit_window(rows: list[dict[str, Any]], exclude: int) -> list[dict[str, Any]]:
    low, high = FIT_WINDOW_BYTES
    return [
        row
        for row in rows
        if low <= requested_bytes(row) <= high and requested_bytes(row) != exclude
    ]


def score_e1(stages: dict[str, dict[str, Any]]) -> tuple[Outcome, list[Outcome]]:
    """Evaluate the physical ceilings, the frozen fatal guard."""

    violations: list[Outcome] = []
    observed: dict[str, Any] = {}
    peaks = {"p1_unidirectional": 0.0, "p1_bidirectional": 0.0, "p2_unidirectional": 0.0}
    peaks.update({"p2_bidirectional": 0.0, "busbw": 0.0, "fanin_aggregate": 0.0})
    fastest_ns = float("inf")

    for name, result in stages.items():
        for lane in ("p1", "p2"):
            for row in measured(result.get(lane, [])):
                fastest_ns = min(fastest_ns, float(row["time_ns"]))
                if float(row["time_ns"]) < MIN_TIMED_ROW_NS:
                    violations.append(
                        Outcome(
                            "E1",
                            "fatal",
                            False,
                            f"{name} {lane} {row.get('cell')} {row.get('direction')} at "
                            f"{row['bytes']} B reports {fmt(float(row['time_ns']), 1)} ns, "
                            f"below the {MIN_TIMED_ROW_NS} ns floor",
                            {"lane": lane, "bytes": row["bytes"], "time_ns": row["time_ns"]},
                        )
                    )
                cell = row.get("cell")
                rate = float(row.get("bytes_per_second", 0.0))
                aggregate = float(row.get("aggregate_bytes_per_second", rate))
                if cell in ("unidirectional", "all_pairs", "stage2_pair"):
                    key = f"{lane}_unidirectional"
                    peaks[key] = max(peaks[key], rate)
                    if rate > UNIDIRECTIONAL_CEILING_BPS:
                        violations.append(
                            Outcome(
                                "E1",
                                "fatal",
                                False,
                                f"{name} {lane} {row.get('direction')} at {row['bytes']} B "
                                f"reaches {fmt(rate / 1e9, 2)} GB/s, above the "
                                f"{UNIDIRECTIONAL_CEILING_BPS / 1e9} GB/s ceiling",
                                {"lane": lane, "bytes": row["bytes"], "bytes_per_second": rate},
                            )
                        )
                elif cell == "bidirectional":
                    key = f"{lane}_bidirectional"
                    peaks[key] = max(peaks[key], aggregate)
                    if aggregate > BIDIRECTIONAL_CEILING_BPS:
                        violations.append(
                            Outcome(
                                "E1",
                                "fatal",
                                False,
                                f"{name} {lane} bidirectional at {row['bytes']} B reaches "
                                f"{fmt(aggregate / 1e9, 2)} GB/s aggregate, above the "
                                f"{BIDIRECTIONAL_CEILING_BPS / 1e9} GB/s ceiling",
                                {"lane": lane, "bytes": row["bytes"], "aggregate": aggregate},
                            )
                        )
                elif cell == "fanin":
                    peaks["fanin_aggregate"] = max(peaks["fanin_aggregate"], aggregate)
                    if aggregate > UNIDIRECTIONAL_CEILING_BPS:
                        violations.append(
                            Outcome(
                                "E1",
                                "fatal",
                                False,
                                f"{name} fan-in of {row.get('donors')} donors reaches "
                                f"{fmt(aggregate / 1e9, 2)} GB/s into GPU 0, above the "
                                f"{UNIDIRECTIONAL_CEILING_BPS / 1e9} GB/s ceiling",
                                {"donors": row.get("donors"), "aggregate": aggregate},
                            )
                        )
        for row in measured(result.get("p3", [])):
            fastest_ns = min(fastest_ns, float(row["time_ns"]))
            busbw = float(row.get("busbw_bytes_per_second", 0.0))
            peaks["busbw"] = max(peaks["busbw"], busbw)
            if float(row["time_ns"]) < MIN_TIMED_ROW_NS:
                violations.append(
                    Outcome(
                        "E1",
                        "fatal",
                        False,
                        f"{name} p3 {row['op']} width {row['width']} at {row['bytes']} B "
                        f"reports {fmt(float(row['time_ns']), 1)} ns, below the "
                        f"{MIN_TIMED_ROW_NS} ns floor",
                        {"op": row["op"], "bytes": row["bytes"], "time_ns": row["time_ns"]},
                    )
                )
            if busbw > BUSBW_CEILING_BPS:
                violations.append(
                    Outcome(
                        "E1",
                        "fatal",
                        False,
                        f"{name} p3 {row['op']} width {row['width']} at {row['bytes']} B "
                        f"reaches {fmt(busbw / 1e9, 2)} GB/s bus bandwidth, above the "
                        f"{BUSBW_CEILING_BPS / 1e9} GB/s ceiling",
                        {"op": row["op"], "bytes": row["bytes"], "busbw": busbw},
                    )
                )
            if row["op"] == "all_reduce" and row.get("all_reduce_correct") is False:
                violations.append(
                    Outcome(
                        "E1",
                        "fatal",
                        False,
                        f"{name} all-reduce width {row['width']} at {row['bytes']} B failed "
                        "its correctness check",
                        {"width": row["width"], "bytes": row["bytes"]},
                    )
                )

    observed["peak_bytes_per_second"] = peaks
    observed["fastest_row_ns"] = None if math.isinf(fastest_ns) else fastest_ns
    detail = (
        "every physical ceiling held: peak unidirectional "
        f"{fmt(max(peaks['p1_unidirectional'], peaks['p2_unidirectional']) / 1e9, 2)} GB/s, "
        f"peak bus bandwidth {fmt(peaks['busbw'] / 1e9, 2)} GB/s, fastest row "
        f"{fmt(observed['fastest_row_ns'] or 0.0, 1)} ns"
    )
    if violations:
        detail = f"{len(violations)} physical ceiling violations, the run is void"
    return Outcome("E1", "fatal", not violations, detail, observed), violations


def _asymptote_cell(
    ident: str,
    rows: list[dict[str, Any]],
    beta_band: tuple[float, float],
    small_band: tuple[float, float],
    label: str,
) -> tuple[list[Outcome], dict[str, Any]]:
    """Fit one asymptote cell and return its outcomes plus the fit record."""

    window = fit_window(rows, ASYMPTOTE_HOLDOUT_BYTES)
    holdout = point(rows, ASYMPTOTE_HOLDOUT_BYTES)
    small = point(rows, SOURCE_PAYLOAD_BYTES[0])
    record: dict[str, Any] = {"fit_points": len(window)}
    outcomes: list[Outcome] = []

    if len(window) < 3 or holdout is None:
        outcomes.append(
            Outcome(
                f"{ident}-fit",
                "structural",
                None,
                f"{label}: not enough rows in the 1 MiB to 1 GiB window to fit",
                record,
            )
        )
        outcomes.append(
            Outcome(
                f"{ident}-holdout",
                "scored",
                None,
                f"{label}: the 64 MiB holdout is not available",
                record,
            )
        )
        return outcomes, record

    alpha, beta, r_squared = ols_fit(
        [float(row["bytes"]) for row in window],
        [float(row["time_ns"]) * 1e-9 for row in window],
    )
    predicted_s = alpha + float(holdout["bytes"]) / beta
    observed_s = float(holdout["time_ns"]) * 1e-9
    error = abs(predicted_s - observed_s) / observed_s if observed_s > 0 else float("inf")
    record.update(
        {
            "alpha_us": alpha * 1e6,
            "beta_bytes_per_second": beta,
            "r_squared": r_squared,
            "holdout_bytes": holdout["bytes"],
            "holdout_observed_us": observed_s * 1e6,
            "holdout_predicted_us": predicted_s * 1e6,
            "holdout_error_fraction": error,
        }
    )
    outcomes.append(
        Outcome(
            f"{ident}-fit",
            "structural",
            bool(r_squared >= R2_MIN and beta_band[0] <= beta <= beta_band[1]),
            f"{label}: beta {fmt(beta / 1e9, 2)} GB/s in "
            f"[{beta_band[0] / 1e9}, {beta_band[1] / 1e9}] with R^2 {fmt(r_squared, 5)}, "
            f"floor {R2_MIN}",
            dict(record),
        )
    )
    outcomes.append(
        Outcome(
            f"{ident}-holdout",
            "scored",
            bool(error <= HOLDOUT_TOLERANCE_FRACTION),
            f"{label}: the 64 MiB holdout predicts {fmt(predicted_s * 1e6, 2)} us against "
            f"{fmt(observed_s * 1e6, 2)} us observed, error {fmt(error * 100, 2)} percent, "
            f"limit {HOLDOUT_TOLERANCE_FRACTION * 100} percent",
            dict(record),
        )
    )
    if small is not None:
        small_us = float(small["time_ns"]) * 1e-3
        record["small_payload_us"] = small_us
        outcomes.append(
            Outcome(
                f"{ident}-small-payload",
                "reported",
                bool(small_band[0] <= small_us <= small_band[1]),
                f"{label}: the 8 B point is {fmt(small_us, 3)} us, recorded against the "
                f"[{small_band[0]}, {small_band[1]}] us expectation and not scored",
                {"small_payload_us": small_us},
            )
        )
    return outcomes, record


def score_e2(stage1: dict[str, Any]) -> tuple[list[Outcome], dict[str, Any]]:
    outcomes: list[Outcome] = []
    records: dict[str, Any] = {}
    for direction in ("0->1", "1->0"):
        rows = copy_series(stage1, "unidirectional", direction)
        cell, record = _asymptote_cell(
            f"E2-{direction}",
            rows,
            P1_BETA_BAND_BPS,
            P1_SMALL_PAYLOAD_BAND_US,
            f"peer copy {direction}",
        )
        outcomes.extend(cell)
        records[direction] = record
    return outcomes, records


def score_e3(stage1: dict[str, Any]) -> tuple[list[Outcome], dict[str, Any]]:
    rows = p2_series(stage1, "unidirectional", "0->1")
    outcomes, record = _asymptote_cell(
        "E3",
        rows,
        P2_BETA_BAND_BPS,
        P2_SMALL_PAYLOAD_BAND_US,
        "NCCL point to point 0->1",
    )
    return outcomes, record


def score_e4(
    stages: dict[str, dict[str, Any]],
    predictions: dict[str, dict[str, int]],
) -> tuple[list[Outcome], dict[str, Any]]:
    """Report the current profile's before error at every measured width."""

    outcomes: list[Outcome] = []
    table: dict[str, Any] = {}
    for width_key, frozen in sorted(predictions.items(), key=lambda item: int(item[0])):
        width = int(width_key)
        rows: list[dict[str, Any]] = []
        for result in stages.values():
            rows.extend(collective_series(result, "all_reduce", width))
        rows = sorted({requested_bytes(row): row for row in rows}.items())
        rows = [row for _, row in rows]
        if not rows:
            outcomes.append(
                Outcome(
                    f"E4-w{width}",
                    "reported",
                    None,
                    f"width {width} was not measured, so the before error is not available",
                    {},
                )
            )
            continue
        entries = []
        validated = True
        for size_bytes in BEFORE_ERROR_ROWS_BYTES:
            row = point(rows, size_bytes)
            predicted_ps = frozen.get(str(size_bytes))
            if row is None or predicted_ps is None:
                validated = False
                entries.append({"bytes": size_bytes, "status": "missing"})
                continue
            observed_ps = float(row["time_ns"]) * 1_000.0
            delta_ps = observed_ps - float(predicted_ps)
            within = abs(delta_ps) <= tolerance_ps(observed_ps)
            validated = validated and within
            entries.append(
                {
                    "bytes": size_bytes,
                    "observed_ps": observed_ps,
                    "predicted_ps": predicted_ps,
                    "delta_ps": delta_ps,
                    "delta_fraction_of_observed": (
                        delta_ps / observed_ps if observed_ps > 0 else None
                    ),
                    "within_tolerance": within,
                }
            )
        table[width_key] = entries
        worst = max(
            (abs(entry.get("delta_ps", 0.0)) for entry in entries if "delta_ps" in entry),
            default=0.0,
        )
        outcomes.append(
            Outcome(
                f"E4-w{width}",
                "reported",
                validated,
                f"width {width} before error: worst row misses the frozen prediction by "
                f"{fmt(worst / 1e6, 3)} us; the profile is "
                f"{'validated' if validated else 'not validated'} at this width",
                {"rows": entries},
            )
        )
    return outcomes, table


def score_e5(stages: dict[str, dict[str, Any]]) -> tuple[list[Outcome], dict[str, Any]]:
    """Refit one intercept per measured width under one shared slope."""

    widths: list[int] = []
    rows_by_width: dict[int, list[dict[str, Any]]] = {}
    for result in stages.values():
        for row in measured(result.get("p3", [])):
            if row.get("op") != "all_reduce":
                continue
            width = int(row["width"])
            rows_by_width.setdefault(width, [])
            if point(rows_by_width[width], requested_bytes(row)) is None:
                rows_by_width[width].append(row)
    widths = sorted(rows_by_width)

    design: list[list[float]] = []
    target: list[float] = []
    used: dict[int, list[tuple[int, int, float]]] = {}
    for index, width in enumerate(widths):
        series = sorted(rows_by_width[width], key=requested_bytes)
        used[width] = []
        for size_bytes in REFIT_ROWS_BYTES:
            row = point(series, size_bytes)
            if row is None:
                continue
            load = endpoint_bytes(width, size_bytes)
            observed_ps = float(row["time_ns"]) * 1_000.0
            columns = [0.0] * len(widths) + [float(load)]
            columns[index] = 1.0
            design.append(columns)
            target.append(observed_ps)
            used[width].append((size_bytes, load, observed_ps))

    record: dict[str, Any] = {"widths": widths, "fit_rows": sum(len(v) for v in used.values())}
    if len(design) < len(widths) + 2:
        return (
            [
                Outcome(
                    "E5-fit",
                    "structural",
                    None,
                    "not enough all-reduce rows to refit an intercept and a shared slope",
                    record,
                )
            ],
            record,
        )

    matrix = np.array(design, dtype=float)
    values = np.array(target, dtype=float)
    solution, _, _, _ = np.linalg.lstsq(matrix, values, rcond=None)
    intercepts_ps = {width: float(solution[index]) for index, width in enumerate(widths)}
    ps_per_byte = float(solution[-1])
    slope_bytes_per_second = (
        PICOSECONDS_PER_SECOND / ps_per_byte if ps_per_byte > 0 else float("inf")
    )
    predicted = matrix @ solution
    residuals = values - predicted
    ss_tot = float(((values - values.mean()) ** 2).sum())
    r_squared = 1.0 - float((residuals**2).sum()) / ss_tot if ss_tot > 0 else float("nan")

    bandwidth = round(slope_bytes_per_second)
    constants = {width: round(value) for width, value in intercepts_ps.items()}
    bands: dict[int, tuple[int, int]] = {}
    row_index = 0
    residual_by_width: dict[int, list[float]] = {width: [] for width in widths}
    for width in widths:
        for _ in used[width]:
            residual_by_width[width].append(float(residuals[row_index]))
            row_index += 1
    for width in widths:
        spread = round(max((abs(value) for value in residual_by_width[width]), default=0.0))
        low = max(0, constants[width] - spread)
        bands[width] = (low, constants[width] + spread)

    record.update(
        {
            "intercept_ps": {str(width): constants[width] for width in widths},
            "bandwidth_bytes_per_second": bandwidth,
            "r_squared": r_squared,
            "band_ps": {str(width): list(bands[width]) for width in widths},
            "max_abs_residual_ps": {
                str(width): max((abs(v) for v in residual_by_width[width]), default=0.0)
                for width in widths
            },
        }
    )

    outcomes = [
        Outcome(
            "E5-fit",
            "structural",
            bool(bandwidth > 0),
            f"refit over {record['fit_rows']} rows at widths {widths}: shared slope "
            f"{bandwidth} bytes per second with intercepts "
            + ", ".join(f"{width}:{constants[width]} ps" for width in widths)
            + f", R^2 {fmt(r_squared, 5)}",
            dict(record),
        )
    ]

    holdouts: dict[str, Any] = {}
    for width in widths:
        series = sorted(rows_by_width[width], key=requested_bytes)
        row = point(series, REFIT_HOLDOUT_BYTES)
        if row is None:
            outcomes.append(
                Outcome(
                    f"E5-holdout-w{width}",
                    "scored",
                    None,
                    f"width {width} has no 4 KiB holdout row",
                    {},
                )
            )
            continue
        load = endpoint_bytes(width, REFIT_HOLDOUT_BYTES)
        predicted_ps = constants[width] + ceil_div(load * PICOSECONDS_PER_SECOND, bandwidth)
        observed_ps = float(row["time_ns"]) * 1_000.0
        allowed = tolerance_ps(observed_ps)
        error = abs(predicted_ps - observed_ps)
        holdouts[str(width)] = {
            "endpoint_bytes": load,
            "predicted_ps": predicted_ps,
            "observed_ps": observed_ps,
            "error_ps": error,
            "allowed_ps": allowed,
        }
        outcomes.append(
            Outcome(
                f"E5-holdout-w{width}",
                "scored",
                bool(error <= allowed),
                f"width {width} 4 KiB holdout predicts {predicted_ps} ps against "
                f"{observed_ps:.0f} ps observed, error {fmt(error / 1e6, 3)} us, allowed "
                f"{fmt(allowed / 1e6, 3)} us",
                holdouts[str(width)],
            )
        )
    record["holdouts"] = holdouts
    return outcomes, record


def score_e6(
    stage1: dict[str, Any],
    refit: dict[str, Any],
    e2_records: dict[str, Any],
) -> tuple[list[Outcome], dict[str, Any]]:
    """Report the serializer question side by side, structural and unscored."""

    rows = collective_series(stage1, "all_reduce", 2)
    window = fit_window(rows, ASYMPTOTE_HOLDOUT_BYTES)
    record: dict[str, Any] = {}
    if len(window) < 3 or "bandwidth_bytes_per_second" not in refit:
        return (
            [
                Outcome(
                    "E6",
                    "structural",
                    None,
                    "the collective asymptote or the refit slope is not available",
                    record,
                )
            ],
            record,
        )
    _, beta_collective, r_squared = ols_fit(
        [float(row["bytes"]) for row in window],
        [float(row["time_ns"]) * 1e-9 for row in window],
    )
    factor = busbw_factor("all_reduce", 2)
    target = ASYMPTOTE_FRACTION * beta_collective * factor
    crossover = None
    for row in rows:
        if float(row.get("busbw_bytes_per_second", 0.0)) >= target:
            crossover = requested_bytes(row)
            break
    window_slope = float(refit["bandwidth_bytes_per_second"])
    beta_pair = e2_records.get("0->1", {}).get("beta_bytes_per_second")
    record = {
        "e5_slope_bytes_per_second": window_slope,
        "all_reduce_asymptote_bytes_per_second": beta_collective,
        "all_reduce_asymptote_r_squared": r_squared,
        "busbw_90_percent_payload_bytes": crossover,
        "e2_beta_bytes_per_second": beta_pair,
        "slope_over_asymptote": window_slope / beta_collective if beta_collective else None,
    }
    passed = bool(window_slope < SERIALIZER_QUARTER * beta_collective)
    return (
        [
            Outcome(
                "E6",
                "structural",
                passed,
                f"the window slope is {fmt(window_slope / 1e9, 2)} GB/s against a large-payload "
                f"all-reduce asymptote of {fmt(beta_collective / 1e9, 2)} GB/s, a ratio of "
                f"{fmt(window_slope / beta_collective, 4)} against the frozen quarter; bus "
                f"bandwidth first reaches 90 percent of the asymptote at {crossover} B",
                record,
            )
        ],
        record,
    )


def score_e7(stage2: dict[str, Any] | None, e2_records: dict[str, Any]) -> list[Outcome]:
    """Evaluate the stage 2 placement structure, structural and unscored."""

    if stage2 is None:
        return [
            Outcome("E7", "structural", None, "stage 2 did not run", {}),
        ]
    outcomes: list[Outcome] = []
    pairs = [
        row
        for row in measured(stage2.get("p1", []))
        if row.get("cell") == "all_pairs"
        and requested_bytes(row) == ALL_PAIRS_STRUCTURE_BYTES
    ]
    if pairs:
        times = [float(row["time_ns"]) for row in pairs]
        spread = (max(times) - min(times)) / min(times) if min(times) > 0 else float("inf")
        outcomes.append(
            Outcome(
                "E7-all-pairs",
                "structural",
                bool(spread <= ALL_PAIRS_SPREAD_MAX),
                f"over {len(pairs)} ordered pairs at 16 MiB the slowest is "
                f"{fmt(spread * 100, 2)} percent above the fastest, limit "
                f"{ALL_PAIRS_SPREAD_MAX * 100} percent",
                {"pairs": len(pairs), "spread_fraction": spread},
            )
        )
    isolated = point(copy_series(stage2, "unidirectional", "0->1"), CONCURRENT_PAYLOAD_BYTES)
    disjoint = next(
        (row for row in measured(stage2.get("p1", [])) if row.get("cell") == "disjoint_pairs"),
        None,
    )
    if isolated is not None and disjoint is not None:
        ratio = float(disjoint["time_ns"]) / float(isolated["time_ns"])
        outcomes.append(
            Outcome(
                "E7-disjoint-pairs",
                "structural",
                bool(ratio <= 1.0 + DISJOINT_SLOWDOWN_MAX),
                f"four disjoint pairs at 64 MiB take {fmt(ratio, 4)} times the isolated pair, "
                f"limit {1.0 + DISJOINT_SLOWDOWN_MAX}",
                {"ratio": ratio},
            )
        )
    fanin = next(
        (
            row
            for row in measured(stage2.get("p1", []))
            if row.get("cell") == "fanin" and row.get("donors") == 7
        ),
        None,
    )
    beta_pair = e2_records.get("0->1", {}).get("beta_bytes_per_second")
    if fanin is not None and isolated is not None and beta_pair:
        floor_ns = (
            7.0 * float(isolated["time_ns"]) * beta_pair / float(UNIDIRECTIONAL_CEILING_BPS)
        )
        outcomes.append(
            Outcome(
                "E7-fanin",
                "structural",
                bool(float(fanin["time_ns"]) >= floor_ns),
                f"the seven-donor fan-in takes {fmt(float(fanin['time_ns']) / 1e6, 3)} ms "
                f"against a receiver-limited floor of {fmt(floor_ns / 1e6, 3)} ms",
                {"fanin_ns": fanin["time_ns"], "floor_ns": floor_ns},
            )
        )
    if not outcomes:
        outcomes.append(
            Outcome("E7", "structural", None, "stage 2 carries no placement rows", {})
        )
    return outcomes


def score_e8() -> list[Outcome]:
    """Emit the identity guards the orchestrator fills in from its own runs."""

    return [
        Outcome(
            ident,
            "fatal-external",
            None,
            f"run separately: {description}",
            {},
        )
        for ident, description in IDENTITY_GUARDS
    ]


def proposed_profile(
    refit: dict[str, Any],
    header: dict[str, Any],
    expectations: dict[str, Any],
) -> dict[str, Any] | None:
    """Return the constants of the new profile in the repository's shape."""

    if "bandwidth_bytes_per_second" not in refit:
        return None
    widths = [int(width) for width in refit["widths"]]
    constants = {int(width): int(value) for width, value in refit["intercept_ps"].items()}
    bands = {int(width): list(band) for width, band in refit["band_ps"].items()}
    substrate = expectations.get("substrate", {})
    holdouts = refit.get("holdouts", {})
    errors = ", ".join(
        f"{fmt(float(holdouts[str(width)]['error_ps']) / 1e6, 3)} us at width {width}"
        for width in widths
        if str(width) in holdouts
    )
    return {
        "profile_id": NEW_PROFILE_ID,
        "bandwidth_bytes_per_second": int(refit["bandwidth_bytes_per_second"]),
        "participant_latency_ps": [[width, constants[width]] for width in widths],
        "source_payload_bytes_min": SOURCE_PAYLOAD_BYTES[0],
        "source_payload_bytes_max": SOURCE_PAYLOAD_BYTES[1],
        "propagation_reference_ps": PROPAGATION_REFERENCE_PS,
        "supported_participant_counts": widths,
        "provenance": {
            "evidence_class": "calibrated",
            "source": (
                f"first-party float32 sum ALL-REDUCE capture of {substrate.get('marketplace')} "
                f"B200 GPUs in the {substrate.get('image')} container, NCCL "
                f"{header.get('nccl_version')} through PyTorch {header.get('torch_version')} "
                f"on driver {header.get('driver_version')}, taken by "
                f"examples/{STUDY} between {header.get('started_utc')} and "
                f"{header.get('finished_utc')}"
            ),
            "locator": (
                f"the refit intercepts of examples/{STUDY}, one intercept per measured width "
                f"under one shared slope, held out at 4 KiB with errors of {errors or 'none'}"
            ),
            "transfer": (
                "the intra-node ALL-REDUCE intercept is charged unchanged as the "
                "per-collective surcharge of any supported collective, and the profile "
                "refuses every width this study did not measure"
            ),
            "participant_latency_band_ps": [
                [width, bands[width][0], bands[width][1]] for width in widths
            ],
        },
    }


def build_report(
    stages: dict[str, dict[str, Any]],
    expectations: dict[str, Any],
) -> dict[str, Any]:
    stage1 = stages["stage1"]
    stage2 = stages.get("stage2")

    outcomes: list[Outcome] = []
    e1, fatal = score_e1(stages)
    outcomes.append(e1)

    e2_outcomes, e2_records = score_e2(stage1)
    outcomes.extend(e2_outcomes)
    e3_outcomes, e3_record = score_e3(stage1)
    outcomes.extend(e3_outcomes)

    predictions = expectations["baseline"]["profile_predictions_ps"]
    available = {
        key: value
        for key, value in predictions.items()
        if any(collective_series(result, "all_reduce", int(key)) for result in stages.values())
    }
    e4_outcomes, e4_table = score_e4(stages, available or predictions)
    outcomes.extend(e4_outcomes)

    e5_outcomes, refit = score_e5(stages)
    outcomes.extend(e5_outcomes)
    e6_outcomes, e6_record = score_e6(stage1, refit, e2_records)
    outcomes.extend(e6_outcomes)
    outcomes.extend(score_e7(stage2, e2_records))
    outcomes.extend(score_e8())

    scored = [outcome for outcome in outcomes if outcome.cls == "scored"]
    passed = sum(1 for outcome in scored if outcome.passed)
    return {
        "schema": SCHEMA,
        "study": STUDY,
        "stages_present": sorted(stages),
        "expectations_task": expectations.get("task"),
        "void": bool(fatal),
        "scored_passed": passed,
        "scored_total": len(scored),
        "fatal_violations": len(fatal),
        "expectations": [outcome.as_dict() for outcome in outcomes],
        "fatal": [outcome.as_dict() for outcome in fatal],
        "before_error": e4_table,
        "asymptotes": {"e2": e2_records, "e3": e3_record, "e6": e6_record},
        "refit": refit,
        "proposed_profile": proposed_profile(refit, stage1.get("header", {}), expectations),
        "identity_guards_pending": [ident for ident, _ in IDENTITY_GUARDS],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Score the B200 NVLink envelope measurements")
    parser.add_argument("--measurements", type=Path, default=here / "measurements")
    parser.add_argument("--expectations", type=Path, default=here / "expectations.json")
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    expectations = json.loads(args.expectations.read_text())
    stages: dict[str, dict[str, Any]] = {}
    for index in (1, 2):
        path = args.measurements / f"stage{index}_result.json"
        if path.is_file():
            stages[f"stage{index}"] = json.loads(path.read_text())
    if "stage1" not in stages:
        print(f"no stage 1 result under {args.measurements}")
        return 2

    report = build_report(stages, expectations)
    out = args.out or (args.measurements / "scored.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print("Fatal guards")
    if not report["fatal"]:
        print("  every evaluated fatal guard held")
    for entry in report["fatal"]:
        print(f"  VIOLATED {entry['id']}: {entry['detail']}")
    print()
    print(f"Scored holdouts {report['scored_passed']} of {report['scored_total']}")
    for entry in report["expectations"]:
        if entry["passed"] is None:
            mark = "n/a "
        else:
            mark = "pass" if entry["passed"] else "FAIL"
        print(f"  [{entry['class']}] {mark} {entry['id']}: {entry['detail']}")
    print()
    print(f"wrote {out}")
    if report["void"]:
        print("the run is void: a fatal guard was violated")
    return 1 if report["void"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
