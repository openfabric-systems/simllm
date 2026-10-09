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
FANIN_DONORS = 7
#: amendment c: a placement row of record is a graph row at every payload, and
#: an eager placement row this slow whose captured twin is less than half of it
#: was timed into a device hosting a spinning peer rank, so it measures the
#: wait and not the link.
NOT_MEANINGFUL_ABOVE_NS = 2_000_000.0
NOT_MEANINGFUL_TWIN_FRACTION = 0.5
EAGER_AGREEMENT_FRACTION = 0.05
PLACEMENT_CELLS = ("all_pairs", "disjoint_pairs", "fanout", "fanin")
#: Width 2 is measured in both stages. Its rows of record are stage 1's, the
#: pinned pair the freeze's E5 text describes; the wider communicators exist
#: only in stage 2. Pooling the stages would let one silently shadow the other,
#: which is how a refit can end up carrying a board its provenance does not
#: name.
PINNED_PAIR_WIDTH = 2
PINNED_PAIR_STAGE = "stage1"

#: amendment of 2026-09-15: at or below this payload the row of record is the
#: CUDA graph replay, above it the eager row, and the other method is kept
#: beside it as an unscored control.
GRAPH_ROW_MAX_BYTES = 1_048_576
METHOD_GRAPH = "graph"
METHOD_EAGER = "eager"

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


@dataclass
class Fit:
    """One ``t = alpha + S / beta`` fit, degenerate cases included.

    A fit is degenerate when the rows carry no usable payload term: too few
    points, no spread in the payloads, or a slope that is zero or negative
    because the window is latency dominated and the measured times do not grow
    with the payload. A degenerate fit has no bandwidth, so every consumer
    reports it rather than rounding an infinity into a profile constant.
    """

    alpha_seconds: float
    beta_bytes_per_second: float
    r_squared: float
    slope_seconds_per_byte: float
    points: int
    degenerate: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "alpha_us": _finite(self.alpha_seconds * 1e6),
            "beta_bytes_per_second": _finite(self.beta_bytes_per_second),
            "r_squared": _finite(self.r_squared),
            "slope_seconds_per_byte": _finite(self.slope_seconds_per_byte),
            "fit_points": self.points,
            "fit_degenerate": self.degenerate,
            "fit_reason": self.reason,
        }


def _finite(value: Any) -> Any:
    """Return ``value`` when it is a finite number, else ``None``.

    Infinity and NaN are not JSON, and a consumer that reads them back as a
    bandwidth would silently carry a nonsense constant.
    """

    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _sanitize(payload: Any) -> Any:
    """Recursively replace non-finite floats so the report is strict JSON."""

    if isinstance(payload, dict):
        return {key: _sanitize(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [_sanitize(value) for value in payload]
    return _finite(payload)


def ols_fit(sizes: list[float], seconds: list[float]) -> Fit:
    """Fit ``t = alpha + S / beta`` by ordinary least squares of ``t`` on ``S``.

    The form is the A100 envelope's and the floor study's. The result is always
    a :class:`Fit`; a window that cannot identify a bandwidth comes back
    degenerate instead of raising or returning an infinite constant.
    """

    count = len(sizes)
    if count < 3:
        return Fit(
            float("nan"),
            float("inf"),
            float("nan"),
            float("nan"),
            count,
            True,
            f"only {count} rows in the window, at least 3 are needed",
        )
    mean_x = sum(sizes) / count
    mean_y = sum(seconds) / count
    sxx = sum((x - mean_x) ** 2 for x in sizes)
    if sxx <= 0:
        return Fit(
            mean_y,
            float("inf"),
            float("nan"),
            float("nan"),
            count,
            True,
            "every row in the window carries the same payload",
        )
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(sizes, seconds))
    slope = sxy / sxx
    alpha = mean_y - slope * mean_x
    predicted = [alpha + slope * x for x in sizes]
    ss_res = sum((y - value) ** 2 for y, value in zip(seconds, predicted))
    ss_tot = sum((y - mean_y) ** 2 for y in seconds)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    if not math.isfinite(slope) or slope <= 0:
        return Fit(
            alpha,
            float("inf"),
            r_squared,
            slope,
            count,
            True,
            (
                "the fitted slope is not positive, so the window identifies no "
                "serialization term: completion time does not grow with the payload here"
            ),
        )
    return Fit(alpha, 1.0 / slope, r_squared, slope, count, False, "")


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


def fit_window(rows: list[dict[str, Any]], exclude: int | None) -> list[dict[str, Any]]:
    low, high = FIT_WINDOW_BYTES
    return [
        row
        for row in rows
        if low <= requested_bytes(row) <= high and requested_bytes(row) != exclude
    ]


def row_method(row: dict[str, Any]) -> str:
    """Return a row's timing method, defaulting to the pre-amendment eager."""

    return str(row.get("method", METHOD_EAGER))


def method_of_record(size_bytes: int) -> str:
    """Return the amendment's method of record for one payload."""

    return METHOD_GRAPH if size_bytes <= GRAPH_ROW_MAX_BYTES else METHOD_EAGER


def rows_of_record(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select one row per payload: graph at or below 1 MiB, eager above.

    When the run carries no row for the method of record, for instance a
    container that could not capture or a capture taken before the amendment,
    the other method is used and the payload is listed as a fallback so the
    report never presents a control row as a row of record silently.
    """

    by_payload: dict[int, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_payload.setdefault(requested_bytes(row), {})[row_method(row)] = row
    chosen: list[dict[str, Any]] = []
    fallbacks: list[int] = []
    for size_bytes, by_method in sorted(by_payload.items()):
        wanted = method_of_record(size_bytes)
        row = by_method.get(wanted)
        if row is None:
            alternative = METHOD_EAGER if wanted == METHOD_GRAPH else METHOD_GRAPH
            row = by_method.get(alternative)
            if row is None:
                continue
            fallbacks.append(size_bytes)
        chosen.append(row)
    record = {
        "rows_selected": len(chosen),
        "fallback_payload_bytes": fallbacks,
        "methods_present": sorted({row_method(row) for row in rows}),
    }
    return chosen, record


def all_reduce_rows_by_width(
    stages: dict[str, dict[str, Any]],
) -> dict[int, dict[str, list[dict[str, Any]]]]:
    """Return the measured all-reduce rows as ``{width: {stage: rows}}``."""

    by_width: dict[int, dict[str, list[dict[str, Any]]]] = {}
    for name, result in stages.items():
        for row in measured(result.get("p3", [])):
            if row.get("op") != "all_reduce":
                continue
            by_width.setdefault(int(row["width"]), {}).setdefault(name, []).append(row)
    return by_width


def source_stage_for_width(width: int, by_stage: dict[str, list[dict[str, Any]]]) -> str:
    """Return the stage whose rows are this width's rows of record."""

    if width == PINNED_PAIR_WIDTH and PINNED_PAIR_STAGE in by_stage:
        return PINNED_PAIR_STAGE
    return min(by_stage)


def placement_rows_of_record(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select the graph row of every placement cell, at every payload.

    Amendment c moves the placement cells to graph rows throughout, because
    their eager rows were timed while seven idle ranks span in an NCCL barrier
    on the very devices being written to.
    """

    graph_rows = [row for row in rows if row_method(row) == METHOD_GRAPH]
    fallbacks = sorted({requested_bytes(row) for row in rows}) if not graph_rows else []
    record = {
        "rows_selected": len(graph_rows) or len(rows),
        "method_of_record": METHOD_GRAPH,
        "fallback_payload_bytes": fallbacks,
    }
    return (graph_rows or rows), record


def mark_not_meaningful(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Report the eager placement rows that timed a spinning peer, not a link."""

    marked: list[dict[str, Any]] = []
    for row in rows:
        if row_method(row) != METHOD_EAGER or row.get("cell") not in PLACEMENT_CELLS:
            continue
        observed = float(row["time_ns"])
        if observed <= NOT_MEANINGFUL_ABOVE_NS:
            continue
        twin = next(
            (
                other
                for other in rows
                if row_method(other) == METHOD_GRAPH
                and other.get("cell") == row.get("cell")
                and other.get("direction") == row.get("direction")
                and requested_bytes(other) == requested_bytes(row)
                and other.get("donors") == row.get("donors")
            ),
            None,
        )
        if twin is None:
            continue
        if float(twin["time_ns"]) < NOT_MEANINGFUL_TWIN_FRACTION * observed:
            marked.append(
                {
                    "cell": row.get("cell"),
                    "direction": row.get("direction"),
                    "bytes": row["bytes"],
                    "eager_time_ns": observed,
                    "graph_time_ns": float(twin["time_ns"]),
                }
            )
    return {
        "not_meaningful_eager_rows": len(marked),
        "reason": (
            "the eager row is above 2 ms while its captured twin is less than half of "
            "it, which is the signature of a copy timed into a device hosting a peer "
            "rank spinning in an NCCL barrier"
        ),
        "rows": marked,
    }


def control_row(
    rows: list[dict[str, Any]],
    size_bytes: int,
    method: str,
) -> dict[str, Any] | None:
    """Return the row for one payload measured by one named method."""

    for row in rows:
        if requested_bytes(row) == size_bytes and row_method(row) == method:
            return row
    return None


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
            if row["op"] == "all_reduce" and row.get("all_reduce_correct") is not True:
                recorded = row.get("all_reduce_correct")
                violations.append(
                    Outcome(
                        "E1",
                        "fatal",
                        False,
                        f"{name} all-reduce width {row['width']} at {row['bytes']} B "
                        + (
                            "failed its correctness check"
                            if recorded is False
                            else "carries no correctness result, which the freeze requires "
                            "on every all-reduce point"
                        ),
                        {
                            "width": row["width"],
                            "bytes": row["bytes"],
                            "all_reduce_correct": recorded,
                        },
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
    """Fit one asymptote cell and return its outcomes plus the fit record.

    ``rows`` are already the rows of record. A window that cannot identify a
    bandwidth is reported as a degenerate fit and fails its holdout, because
    the freeze scores that holdout against a prediction the fit cannot make.
    """

    selected, selection = rows_of_record(rows)
    window = fit_window(selected, ASYMPTOTE_HOLDOUT_BYTES)
    holdout = point(selected, ASYMPTOTE_HOLDOUT_BYTES)
    small = point(selected, SOURCE_PAYLOAD_BYTES[0])
    record: dict[str, Any] = {"selection": selection}
    outcomes: list[Outcome] = []

    if holdout is None:
        record["fit_points"] = len(window)
        record["holdout_missing"] = True
        outcomes.append(
            Outcome(
                f"{ident}-fit",
                "structural",
                None,
                f"{label}: no rows to fit in the 1 MiB to 1 GiB window",
                record,
            )
        )
        outcomes.append(
            Outcome(
                f"{ident}-holdout",
                "scored",
                False,
                f"{label}: the 64 MiB holdout row is missing, so the cell cannot be "
                "satisfied; a scored row that was not measured is not a row that passed",
                record,
            )
        )
        return outcomes, record

    fit = ols_fit(
        [float(row["bytes"]) for row in window],
        [float(row["time_ns"]) * 1e-9 for row in window],
    )
    record.update(fit.as_dict())
    record["window_payload_bytes"] = [requested_bytes(row) for row in window]
    observed_s = float(holdout["time_ns"]) * 1e-9
    record["holdout_bytes"] = holdout["bytes"]
    record["holdout_observed_us"] = observed_s * 1e6

    if fit.degenerate:
        outcomes.append(
            Outcome(
                f"{ident}-fit",
                "structural",
                False,
                f"{label}: degenerate fit over {fit.points} rows, {fit.reason}",
                dict(record),
            )
        )
        outcomes.append(
            Outcome(
                f"{ident}-holdout",
                "scored",
                False,
                f"{label}: the 64 MiB holdout cannot be predicted because the fit is "
                f"degenerate ({fit.reason})",
                dict(record),
            )
        )
    else:
        predicted_s = fit.alpha_seconds + float(holdout["bytes"]) / fit.beta_bytes_per_second
        error = abs(predicted_s - observed_s) / observed_s if observed_s > 0 else float("inf")
        record["holdout_predicted_us"] = predicted_s * 1e6
        record["holdout_error_fraction"] = error
        outcomes.append(
            Outcome(
                f"{ident}-fit",
                "structural",
                bool(
                    fit.r_squared >= R2_MIN
                    and beta_band[0] <= fit.beta_bytes_per_second <= beta_band[1]
                ),
                f"{label}: beta {fmt(fit.beta_bytes_per_second / 1e9, 2)} GB/s in "
                f"[{beta_band[0] / 1e9}, {beta_band[1] / 1e9}] with R^2 "
                f"{fmt(fit.r_squared, 5)}, floor {R2_MIN}",
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
        record["small_payload_method"] = row_method(small)
        outcomes.append(
            Outcome(
                f"{ident}-small-payload",
                "reported",
                bool(small_band[0] <= small_us <= small_band[1]),
                f"{label}: the 8 B point is {fmt(small_us, 3)} us by the "
                f"{row_method(small)} method, recorded against the "
                f"[{small_band[0]}, {small_band[1]}] us expectation and not scored",
                {"small_payload_us": small_us, "method": row_method(small)},
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
    """Report the current profile's before error at every measured width.

    The amendment reads the before error from the graph rows and keeps the
    eager row of the same payload beside it, so the size of the dispatch floor
    the first capture found stays on the record.
    """

    outcomes: list[Outcome] = []
    table: dict[str, Any] = {}
    for width_key, frozen in sorted(predictions.items(), key=lambda item: int(item[0])):
        width = int(width_key)
        by_stage = all_reduce_rows_by_width(stages).get(width, {})
        source = source_stage_for_width(width, by_stage) if by_stage else ""
        all_rows = sorted(by_stage.get(source, []), key=requested_bytes)
        if not all_rows:
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
        selected, selection = rows_of_record(all_rows)
        selection["source_stage"] = source
        entries = []
        validated = True
        for size_bytes in BEFORE_ERROR_ROWS_BYTES:
            row = point(selected, size_bytes)
            predicted_ps = frozen.get(str(size_bytes))
            if row is None or predicted_ps is None:
                validated = False
                entries.append({"bytes": size_bytes, "status": "missing"})
                continue
            observed_ps = float(row["time_ns"]) * 1_000.0
            delta_ps = observed_ps - float(predicted_ps)
            within = abs(delta_ps) <= tolerance_ps(observed_ps)
            validated = validated and within
            control = control_row(all_rows, size_bytes, METHOD_EAGER)
            entry: dict[str, Any] = {
                "bytes": size_bytes,
                "method": row_method(row),
                "observed_ps": observed_ps,
                "predicted_ps": predicted_ps,
                "delta_ps": delta_ps,
                "delta_fraction_of_observed": (
                    delta_ps / observed_ps if observed_ps > 0 else None
                ),
                "within_tolerance": within,
            }
            if control is not None and control is not row:
                entry["eager_control_ps"] = float(control["time_ns"]) * 1_000.0
                entry["dispatch_floor_ps"] = entry["eager_control_ps"] - observed_ps
            entries.append(entry)
        table[width_key] = {"rows": entries, "selection": selection}
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
                {"rows": entries, "selection": selection},
            )
        )
    return outcomes, table


def score_e5(stages: dict[str, dict[str, Any]]) -> tuple[list[Outcome], dict[str, Any]]:
    """Refit one intercept per measured width under one shared slope.

    The fit rows run from 8 B to 256 KiB, so every one of them is a row of
    record by graph replay under the amendment. A slope that comes back zero or
    negative means the window carries no serialization term; that is reported
    as a degenerate fit whose holdout fails, and no profile is proposed.
    """

    by_width = all_reduce_rows_by_width(stages)
    rows_by_width: dict[int, list[dict[str, Any]]] = {}
    selection_by_width: dict[str, Any] = {}
    for width, by_stage in by_width.items():
        source = source_stage_for_width(width, by_stage)
        selected, selection = rows_of_record(by_stage[source])
        selection["source_stage"] = source
        selection["stages_measuring_this_width"] = sorted(by_stage)
        rows_by_width[width] = selected
        selection_by_width[str(width)] = selection
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

    record: dict[str, Any] = {
        "widths": widths,
        "fit_rows": sum(len(value) for value in used.values()),
        "selection": selection_by_width,
        "fit_degenerate": False,
    }
    if len(design) < len(widths) + 2:
        record["fit_degenerate"] = True
        record["fit_reason"] = "not enough all-reduce rows to refit"
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
    predicted = matrix @ solution
    residuals = values - predicted
    ss_tot = float(((values - values.mean()) ** 2).sum())
    r_squared = 1.0 - float((residuals**2).sum()) / ss_tot if ss_tot > 0 else float("nan")
    record["r_squared"] = _finite(r_squared)
    record["ps_per_byte"] = _finite(ps_per_byte)
    record["intercept_ps"] = {
        str(width): _finite(intercepts_ps[width]) for width in widths
    }
    record["fit_rows_by_width"] = {
        str(width): [size for size, _, _ in used[width]] for width in widths
    }

    degenerate_reason = ""
    if not all(math.isfinite(value) for value in solution):
        degenerate_reason = "the least squares solution is not finite"
    elif ps_per_byte <= 0:
        degenerate_reason = (
            "the shared slope is not positive, so the 8 B to 256 KiB window identifies no "
            "endpoint serializer: completion time does not grow with the payload there"
        )
    if degenerate_reason:
        record["fit_degenerate"] = True
        record["fit_reason"] = degenerate_reason
        outcomes = [
            Outcome(
                "E5-fit",
                "structural",
                False,
                f"degenerate refit over {record['fit_rows']} rows at widths {widths}: "
                f"{degenerate_reason}",
                dict(record),
            )
        ]
        for width in widths:
            outcomes.append(
                Outcome(
                    f"E5-holdout-w{width}",
                    "scored",
                    False,
                    f"width {width} 4 KiB holdout cannot be predicted: {degenerate_reason}",
                    {"fit_degenerate": True, "fit_reason": degenerate_reason},
                )
            )
        return outcomes, record

    slope_bytes_per_second = PICOSECONDS_PER_SECOND / ps_per_byte
    bandwidth = round(slope_bytes_per_second)
    constants = {width: round(value) for width, value in intercepts_ps.items()}
    holdout_error_by_width: dict[int, float] = {}
    for width in widths:
        row = point(sorted(rows_by_width[width], key=requested_bytes), REFIT_HOLDOUT_BYTES)
        if row is None:
            continue
        load = endpoint_bytes(width, REFIT_HOLDOUT_BYTES)
        predicted_ps = constants[width] + ceil_div(load * PICOSECONDS_PER_SECOND, bandwidth)
        holdout_error_by_width[width] = abs(predicted_ps - float(row["time_ns"]) * 1_000.0)
    bands: dict[int, tuple[int, int]] = {}
    row_index = 0
    residual_by_width: dict[int, list[float]] = {width: [] for width in widths}
    for width in widths:
        for _ in used[width]:
            residual_by_width[width].append(float(residuals[row_index]))
            row_index += 1
    for width in widths:
        # The band is the inclusive minimum and maximum of the intercept plus
        # the fit residuals, so it brackets every row the fit saw, widened to
        # the holdout error when that reaches further than any residual.
        spread = residual_by_width[width] or [0.0]
        low = constants[width] + round(min(spread))
        high = constants[width] + round(max(spread))
        holdout_error = holdout_error_by_width.get(width, 0.0)
        low = max(0, min(low, constants[width] - round(holdout_error)))
        high = max(high, constants[width] + round(holdout_error))
        bands[width] = (low, high)

    record.update(
        {
            "intercept_ps": {str(width): constants[width] for width in widths},
            "bandwidth_bytes_per_second": bandwidth,
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
            True,
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
                    False,
                    f"width {width} has no 4 KiB holdout row, so its refit cannot be "
                    "declared eligible",
                    {"holdout_missing": True},
                )
            )
            continue
        load = endpoint_bytes(width, REFIT_HOLDOUT_BYTES)
        predicted_ps = constants[width] + ceil_div(load * PICOSECONDS_PER_SECOND, bandwidth)
        observed_ps = float(row["time_ns"]) * 1_000.0
        allowed = tolerance_ps(observed_ps)
        error = abs(predicted_ps - observed_ps)
        holdouts[str(width)] = {
            "source_stage": selection_by_width[str(width)]["source_stage"],
            "endpoint_bytes": load,
            "predicted_ps": predicted_ps,
            "observed_ps": observed_ps,
            "error_ps": error,
            "allowed_ps": allowed,
            "method": row_method(row),
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
    """Report the serializer question side by side, structural and unscored.

    The amendment reads the window slope from the graph rows, which is what the
    refit returns, and the large-payload asymptote from the eager rows, which
    are the rows of record above 1 MiB.
    """

    rows = [
        row
        for row in collective_series(stage1, "all_reduce", 2)
        if row_method(row) == METHOD_EAGER
    ] or collective_series(stage1, "all_reduce", 2)
    # E6 reports an asymptote rather than scoring a holdout, so it fits the whole
    # 1 MiB to 1 GiB window; the 64 MiB row is held out only where it is scored.
    window = fit_window(rows, None)
    record: dict[str, Any] = {}
    fit = ols_fit(
        [float(row["bytes"]) for row in window],
        [float(row["time_ns"]) * 1e-9 for row in window],
    )
    record["all_reduce_asymptote"] = fit.as_dict()
    window_slope = refit.get("bandwidth_bytes_per_second")
    record["e5_slope_bytes_per_second"] = window_slope
    record["e5_fit_degenerate"] = bool(refit.get("fit_degenerate"))
    record["e2_beta_bytes_per_second"] = e2_records.get("0->1", {}).get(
        "beta_bytes_per_second"
    )
    crossover = None
    if not fit.degenerate:
        factor = busbw_factor("all_reduce", 2)
        target = ASYMPTOTE_FRACTION * fit.beta_bytes_per_second * factor
        for row in rows:
            if float(row.get("busbw_bytes_per_second", 0.0)) >= target:
                crossover = requested_bytes(row)
                break
    record["busbw_90_percent_payload_bytes"] = crossover

    if fit.degenerate or window_slope is None:
        missing = "the refit slope" if window_slope is None else "the collective asymptote"
        return (
            [
                Outcome(
                    "E6",
                    "structural",
                    None,
                    f"the side by side is incomplete: {missing} is not available"
                    + (f" ({fit.reason})" if fit.degenerate else ""),
                    record,
                )
            ],
            record,
        )
    ratio = window_slope / fit.beta_bytes_per_second
    record["slope_over_asymptote"] = ratio
    return (
        [
            Outcome(
                "E6",
                "structural",
                bool(window_slope < SERIALIZER_QUARTER * fit.beta_bytes_per_second),
                f"the window slope is {fmt(window_slope / 1e9, 2)} GB/s against a large-payload "
                f"all-reduce asymptote of {fmt(fit.beta_bytes_per_second / 1e9, 2)} GB/s, a "
                f"ratio of {fmt(ratio, 4)} against the frozen quarter; bus bandwidth first "
                f"reaches 90 percent of the asymptote at {crossover} B",
                record,
            )
        ],
        record,
    )


def score_e7(stage2: dict[str, Any] | None, e2_records: dict[str, Any]) -> list[Outcome]:
    """Evaluate the stage 2 placement structure, structural and unscored.

    Amendment c makes the graph-replay row the row of record in every placement
    cell at every payload, keeps the eager rows as a control with the ones
    timed into a spinning peer's device marked not meaningful, and replaces the
    fan-in rule, which as frozen inverted its fraction and asked the seven-donor
    fan-in to be slower than the nameplate allows.
    """

    if stage2 is None:
        return [
            Outcome("E7", "structural", None, "stage 2 did not run", {}),
        ]
    outcomes: list[Outcome] = []
    placement = [
        row
        for row in measured(stage2.get("p1", []))
        if row.get("cell") in PLACEMENT_CELLS
    ]
    control = mark_not_meaningful(placement)
    outcomes.append(
        Outcome(
            "E7-eager-control",
            "reported",
            None,
            f"{control['not_meaningful_eager_rows']} of "
            f"{sum(1 for row in placement if row_method(row) == METHOD_EAGER)} eager "
            "placement rows are marked not meaningful: " + control["reason"],
            control,
        )
    )

    pairs, pairs_selection = placement_rows_of_record(
        [
            row
            for row in placement
            if row.get("cell") == "all_pairs"
            and requested_bytes(row) == ALL_PAIRS_STRUCTURE_BYTES
        ]
    )
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
                {
                    "pairs": len(pairs),
                    "spread_fraction": spread,
                    "fastest_us": min(times) / 1e3,
                    "slowest_us": max(times) / 1e3,
                    "selection": pairs_selection,
                },
            )
        )

    isolated_rows, _ = placement_rows_of_record(
        [
            row
            for row in measured(stage2.get("p1", []))
            if row.get("cell") == "unidirectional"
            and row.get("direction") == "0->1"
            and requested_bytes(row) == CONCURRENT_PAYLOAD_BYTES
        ]
    )
    isolated = isolated_rows[0] if isolated_rows else None
    disjoint_rows, _ = placement_rows_of_record(
        [row for row in placement if row.get("cell") == "disjoint_pairs"]
    )
    disjoint = disjoint_rows[0] if disjoint_rows else None
    if isolated is not None and disjoint is not None:
        ratio = float(disjoint["time_ns"]) / float(isolated["time_ns"])
        outcomes.append(
            Outcome(
                "E7-disjoint-pairs",
                "structural",
                bool(ratio <= 1.0 + DISJOINT_SLOWDOWN_MAX),
                f"four disjoint pairs at 64 MiB take {fmt(ratio, 4)} times the isolated pair, "
                f"limit {1.0 + DISJOINT_SLOWDOWN_MAX}",
                {
                    "ratio": ratio,
                    "disjoint_us": float(disjoint["time_ns"]) / 1e3,
                    "isolated_us": float(isolated["time_ns"]) / 1e3,
                },
            )
        )

    fanout_rows, _ = placement_rows_of_record(
        [row for row in placement if row.get("cell") == "fanout"]
    )
    if fanout_rows:
        fanout = fanout_rows[0]
        aggregate = float(fanout.get("aggregate_bytes_per_second", 0.0))
        outcomes.append(
            Outcome(
                "E7-fanout",
                "reported",
                None,
                f"the fan-out to seven peers completes in "
                f"{fmt(float(fanout['time_ns']) / 1e6, 3)} ms at "
                f"{fmt(aggregate / 1e9, 1)} GB/s aggregate, "
                f"{fmt(aggregate / UNIDIRECTIONAL_CEILING_BPS * 100, 1)} percent of the "
                "nameplate",
                {
                    "time_ns": fanout["time_ns"],
                    "aggregate_bytes_per_second": aggregate,
                    "fraction_of_nameplate": aggregate / UNIDIRECTIONAL_CEILING_BPS,
                },
            )
        )

    fanin_rows, _ = placement_rows_of_record(
        [
            row
            for row in placement
            if row.get("cell") == "fanin" and row.get("donors") == FANIN_DONORS
        ]
    )
    if fanin_rows:
        fanin = fanin_rows[0]
        observed_ns = float(fanin["time_ns"])
        # Amendment c: the nameplate floor for seven donors into one receiver.
        floor_ns = (
            FANIN_DONORS * CONCURRENT_PAYLOAD_BYTES / float(UNIDIRECTIONAL_CEILING_BPS) * 1e9
        )
        aggregate = float(fanin.get("aggregate_bytes_per_second", 0.0))
        fraction = aggregate / UNIDIRECTIONAL_CEILING_BPS
        outcomes.append(
            Outcome(
                "E7-fanin",
                "structural",
                bool(observed_ns >= floor_ns),
                f"the seven-donor fan-in takes {fmt(observed_ns / 1e6, 3)} ms against the "
                f"nameplate floor of {fmt(floor_ns / 1e6, 3)} ms, receiving at "
                f"{fmt(aggregate / 1e9, 1)} GB/s, {fmt(fraction * 100, 1)} percent of the "
                "900 GB/s nameplate",
                {
                    "fanin_ns": observed_ns,
                    "floor_ns": floor_ns,
                    "aggregate_bytes_per_second": aggregate,
                    "fraction_of_nameplate": fraction,
                },
            )
        )

    if len(outcomes) <= 1:
        outcomes.append(
            Outcome("E7", "structural", None, "stage 2 carries no placement rows", {})
        )
    return outcomes


def score_eager_agreement(stages: dict[str, dict[str, Any]]) -> Outcome:
    """Compare the eager control against its captured twin above 1 MiB.

    Amendment c expects the two methods to agree within 5 percent above 1 MiB
    once the idle ranks wait on the host. The criterion is frozen and the
    outcome is reported as it falls; what the check must not do is count rows
    an earlier amendment already declared not meaningful, because a row with no
    meaning cannot disagree with anything. The offsets are reported per payload
    as well as the fractions, since a constant offset and a constant fraction
    are different claims about the method.
    """

    compared = 0
    outside = 0
    worst: dict[str, Any] | None = None
    excluded = 0
    by_payload: dict[int, list[tuple[float, float]]] = {}
    per_stage: dict[str, dict[str, int]] = {}

    for name, result in stages.items():
        rows = measured(result.get("p1", []))
        meaningless = {
            (entry["cell"], entry["direction"], entry["bytes"])
            for entry in mark_not_meaningful(rows)["rows"]
        }
        stage_record = per_stage.setdefault(name, {"compared": 0, "outside": 0})
        for row in rows:
            if row_method(row) != METHOD_EAGER or requested_bytes(row) <= GRAPH_ROW_MAX_BYTES:
                continue
            twin = next(
                (
                    other
                    for other in rows
                    if row_method(other) == METHOD_GRAPH
                    and other.get("cell") == row.get("cell")
                    and other.get("direction") == row.get("direction")
                    and requested_bytes(other) == requested_bytes(row)
                    and other.get("donors") == row.get("donors")
                ),
                None,
            )
            if twin is None or float(twin["time_ns"]) <= 0:
                continue
            key = (row.get("cell"), row.get("direction"), row["bytes"])
            if key in meaningless or _spinning_peer_row(row, twin):
                excluded += 1
                continue
            offset = float(row["time_ns"]) - float(twin["time_ns"])
            gap = abs(offset) / float(twin["time_ns"])
            compared += 1
            stage_record["compared"] += 1
            by_payload.setdefault(requested_bytes(row), []).append((offset, gap))
            if gap > EAGER_AGREEMENT_FRACTION:
                outside += 1
                stage_record["outside"] += 1
            if worst is None or gap > worst["gap_fraction"]:
                worst = {
                    "stage": name,
                    "cell": row.get("cell"),
                    "direction": row.get("direction"),
                    "bytes": row["bytes"],
                    "eager_time_ns": row["time_ns"],
                    "graph_time_ns": twin["time_ns"],
                    "offset_ns": offset,
                    "gap_fraction": gap,
                }

    table = []
    for size_bytes in sorted(by_payload):
        offsets = sorted(value for value, _ in by_payload[size_bytes])
        gaps = sorted(value for _, value in by_payload[size_bytes])
        middle = len(offsets) // 2
        table.append(
            {
                "bytes": size_bytes,
                "rows": len(offsets),
                "median_offset_ns": offsets[middle],
                "median_gap_fraction": gaps[middle],
                "max_gap_fraction": gaps[-1],
            }
        )

    observed = {
        "compared": compared,
        "outside": outside,
        "excluded_not_meaningful": excluded,
        "per_stage": per_stage,
        "offset_by_payload": table,
        "worst": worst,
    }
    if not compared:
        return Outcome(
            "E7-eager-agreement",
            "structural",
            None,
            "no payload above 1 MiB carries both methods, so the control cannot be compared",
            observed,
        )
    offsets = [entry["median_offset_ns"] for entry in table]
    return Outcome(
        "E7-eager-agreement",
        "structural",
        bool(outside == 0),
        f"above 1 MiB {compared - outside} of {compared} eager rows agree with their "
        f"captured row within {EAGER_AGREEMENT_FRACTION * 100} percent "
        f"({excluded} rows excluded as carrying the spinning-peer signature); the "
        f"median eager minus graph "
        f"offset runs from {fmt(min(offsets) / 1e3, 1)} to {fmt(max(offsets) / 1e3, 1)} us "
        "across the payloads, which is the per-iteration dispatch the capture removes",
        observed,
    )


def _spinning_peer_row(row: dict[str, Any], twin: dict[str, Any]) -> bool:
    """Return whether this eager row carries the spinning-peer artifact."""

    observed = float(row["time_ns"])
    return (
        observed > NOT_MEANINGFUL_ABOVE_NS
        and float(twin["time_ns"]) < NOT_MEANINGFUL_TWIN_FRACTION * observed
    )


def score_e8() -> list[Outcome]:
    """Emit the identity guards the orchestrator fills in from its own runs."""

    return [
        Outcome(
            ident,
            "fatal-external",
            None,
            f"run separately, see RESULTS.md: {description}",
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


def graph_availability(stages: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], str]:
    """Report, per lane, whether any graph row of record survived.

    A lane whose captures all failed silently falls back to its eager control,
    and every cell downstream would then be scoring the dispatch floor. That
    has to be visible at the top of the report, not buried per row.
    """

    flags: dict[str, Any] = {}
    missing: list[str] = []
    reasons: dict[str, str] = {}
    for lane in ("p1", "p2", "p3"):
        rows = [row for result in stages.values() for row in result.get(lane, [])]
        if not rows:
            continue
        graph_rows = [
            row
            for row in rows
            if row_method(row) == METHOD_GRAPH and row.get("status") == "measured"
        ]
        flags[f"{lane}_graph_rows_available"] = bool(graph_rows)
        if graph_rows:
            continue
        missing.append(lane.upper())
        reason = next(
            (str(row["graph_skip_reason"]) for row in rows if row.get("graph_skip_reason")),
            "no reason recorded",
        )
        reasons[lane] = reason.strip().splitlines()[0]
    if not missing:
        summary = "every lane carries graph rows of record"
    else:
        detail = "; ".join(f"{lane}: {reasons[lane.lower()]}" for lane in missing)
        lanes = ", ".join(missing)
        verb = "has" if len(missing) == 1 else "have"
        summary = (
            f"lane {lanes} {verb} no graph rows, so those rows at or below 1 MiB fall "
            f"back to the eager control ({detail})"
        )
    flags["graph_rows_summary"] = summary
    return flags, summary


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
    outcomes.append(score_eager_agreement(stages))
    outcomes.extend(score_e8())

    scored = [outcome for outcome in outcomes if outcome.cls == "scored"]
    passed = sum(1 for outcome in scored if outcome.passed)
    availability, _ = graph_availability(stages)
    return {
        "schema": SCHEMA,
        "study": STUDY,
        "stages_present": sorted(stages),
        "expectations_task": expectations.get("task"),
        "void": bool(fatal),
        "void_scope": (
            "void is decided by cell E1, the physical ceilings, over the rows in this "
            "result. The E8 identity guards are equally fatal but are run separately and "
            "reported in RESULTS.md, so a false value here never means E8 held"
        ),
        "scored_passed": passed,
        "scored_total": len(scored),
        "fatal_violations": len(fatal),
        "expectations": [outcome.as_dict() for outcome in outcomes],
        "fatal": [outcome.as_dict() for outcome in fatal],
        "before_error": e4_table,
        "asymptotes": {"e2": e2_records, "e3": e3_record, "e6": e6_record},
        "refit": refit,
        "proposed_profile": proposed_profile(refit, stage1.get("header", {}), expectations),
        "proposed_profile_reason": (
            refit.get("fit_reason", "")
            if refit.get("fit_degenerate")
            else "the refit produced the constants below"
        ),
        "timing_methods": sorted(
            {
                row_method(row)
                for result in stages.values()
                for lane in ("p1", "p2", "p3")
                for row in result.get(lane, [])
            }
        ),
        "identity_guards_pending": [ident for ident, _ in IDENTITY_GUARDS],
        **availability,
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
    out.write_text(json.dumps(_sanitize(report), indent=2, sort_keys=True) + "\n")

    print(f"Timing methods: {report['graph_rows_summary']}")
    print()
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
