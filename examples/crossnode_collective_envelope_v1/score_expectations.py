"""Score the cross-node collective envelope measurements against the freeze.

Every bound, band and relation checked here is transcribed from
expectations.md, which was committed before this script and before the harness
existed. The script reads the raw per-cell results, evaluates each scored
relation, evaluates each fatal guard separately, derives the frozen bandwidth
curve anchors, and prints a report. It never edits the freeze.

Usage::

    python score_expectations.py \\
        --w2-default measurements/crossnode_w2_default.json \\
        --w2-fournic measurements/crossnode_w2_fournic.json \\
        --w8-default measurements/crossnode_w8_default.json \\
        --out measurements/scored.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Port and host constants, derived in the freeze from the phase 1 discovery and
# from the A100 hardware envelope's own measurements of this node type.
CASSINI_PORT_GBPS = 25.0
CASSINI_FOUR_PORT_GBPS = 100.0
PCIE_D2H_GBPS = 26.19
PCIE_H2D_GBPS = 26.78

# First-party intra-node A100 anchors this study compares against.
INTRA_W2_FLOOR_US = 9.1136
INTRA_W4_FLOOR_US = 12.9536
INTRA_W2_BUSBW_GBPS = 72.774312725

# Shipped repository constants the measurement is scored against.
B200_LOCAL_W2_US = 10.722112
B200_LOCAL_W8_US = 30.128029
B200_CROSS_W2_US = 13.487792
B200_CROSS_W8_US = 49.487789

# The frozen curve-anchor window.
CURVE_WINDOW_LOW = 131072
CURVE_WINDOW_HIGH = 8388608
CURVE_HELD_OUT_BAR_PERCENT = 15.0


@dataclass
class Outcome:
    """One relation or fatal guard.

    ``evaluated`` is False when the cell the relation needs never ran. An
    absent measurement is an absence of evidence, not a refuted prediction, so
    it is reported separately and never counted in the scored denominator. The
    freeze did not anticipate an unschedulable cell; treating one as a failure
    would understate the model and treating it as a pass would be worse.
    """

    ident: str
    passed: bool
    detail: str
    observed: dict[str, Any] = field(default_factory=dict)
    evaluated: bool = True


def band(value: float, low: float, high: float) -> bool:
    return low <= value <= high


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


class Lane:
    """One measured cell, indexed by operation and payload."""

    def __init__(self, payload: dict[str, Any], label: str) -> None:
        self.label = label
        self.raw = payload
        self.width = int(payload["world"])
        self.arm = str(payload["arm"])
        self.cells: dict[tuple[str, int], dict[str, Any]] = {}
        for cell in payload["cells"]:
            self.cells[(str(cell["op"]), int(cell["bytes"]))] = cell

    def payloads(self, op: str) -> list[int]:
        return sorted(byte for (name, byte) in self.cells if name == op)

    def has(self, op: str) -> bool:
        return any(name == op for (name, _) in self.cells)

    def time_us(self, op: str, payload_bytes: int) -> float:
        return float(self.cells[(op, payload_bytes)]["time_us"])

    def field(self, op: str, payload_bytes: int, name: str) -> float:
        return float(self.cells[(op, payload_bytes)][name])

    def floor_us(self, op: str) -> float:
        return self.time_us(op, 8)


def endpoint_bytes(op: str, payload_bytes: int, width: int) -> int:
    """Return one rank's endpoint load, matching the traffic module exactly.

    `critical_collective_endpoint_bytes` expands a ring all-reduce into
    ``2(W-1)`` chunks of ``payload // W`` and a pairwise all-to-allv into
    ``(W-1)`` whole payloads. The two are different functions and using the
    ring expansion for an all-to-allv would misprice it by a factor of
    ``W / 2``.
    """

    if payload_bytes == 0:
        return 0
    if op == "allreduce":
        chunk = max(1, payload_bytes // width)
        return 2 * (width - 1) * chunk
    if op == "alltoallv":
        return (width - 1) * payload_bytes
    # A point-to-point message is its own endpoint load.
    return payload_bytes


def serialization_bandwidth(lane: Lane, op: str) -> list[tuple[int, float]]:
    """Return ``(payload_bytes, serialization GB/s)`` with the floor removed.

    The serialization bandwidth is what a profile's bandwidth term has to
    represent: endpoint bytes over the measured time with the width's own base
    latency already subtracted. Storing total algorithm bandwidth instead would
    double-count the floor once a consumer adds the base latency back on.
    """

    floor_us = lane.floor_us(op)
    rows: list[tuple[int, float]] = []
    for payload in lane.payloads(op):
        residual_us = lane.time_us(op, payload) - floor_us
        if residual_us <= 0.0:
            continue
        endpoint = endpoint_bytes(op, payload, lane.width)
        if endpoint <= 0:
            continue
        rows.append((payload, endpoint / (residual_us * 1e-6) / 1e9))
    return rows


def frozen_curve_anchors(rows: list[tuple[int, float]]) -> list[int]:
    """Return the anchor payloads under the rule frozen in expectations.md.

    The rule, verbatim from the freeze: the smallest measured endpoint load; the
    payload at which the measured serialization bandwidth first reaches 50
    percent of its maximum over the sweep; the payload of the local minimum of
    the measured serialization bandwidth inside the 128 KiB to 8 MiB window if
    one exists, and otherwise the geometric midpoint of that window; and the
    largest measured endpoint load.
    """

    if not rows:
        return []
    payloads = [payload for payload, _ in rows]
    rates = {payload: rate for payload, rate in rows}
    peak = max(rates.values())

    anchors = [payloads[0]]

    half = next(
        (payload for payload in payloads if rates[payload] >= 0.5 * peak),
        payloads[0],
    )
    anchors.append(half)

    window = [p for p in payloads if CURVE_WINDOW_LOW <= p <= CURVE_WINDOW_HIGH]
    local_min: int | None = None
    for index in range(1, len(window) - 1):
        previous, current, following = window[index - 1], window[index], window[index + 1]
        dips = rates[current] < rates[previous] and rates[current] < rates[following]
        if dips and (local_min is None or rates[current] < rates[local_min]):
            local_min = current
    if local_min is None:
        midpoint = math.sqrt(CURVE_WINDOW_LOW * CURVE_WINDOW_HIGH)
        local_min = min(payloads, key=lambda p: abs(math.log(p) - math.log(midpoint)))
    anchors.append(local_min)

    anchors.append(payloads[-1])
    return sorted(set(anchors))


def curve_rate(anchors: list[tuple[int, float]], load_bytes: int) -> float:
    """Interpolate geometrically, the same law ``CollectiveBandwidthCurve`` uses."""

    if load_bytes <= anchors[0][0]:
        return anchors[0][1]
    if load_bytes >= anchors[-1][0]:
        return anchors[-1][1]
    for (low_bytes, low_rate), (high_bytes, high_rate) in itertools.pairwise(anchors):
        if low_bytes <= load_bytes <= high_bytes:
            span = math.log(high_bytes / low_bytes)
            position = math.log(load_bytes / low_bytes) / span
            return low_rate * (high_rate / low_rate) ** position
    raise AssertionError("unreachable: endpoint bytes fell outside every span")


def held_out_error(lane: Lane, op: str) -> dict[str, Any]:
    """Return the frozen curve's anchors and its worst signed held-out error."""

    rows = serialization_bandwidth(lane, op)
    if len(rows) < 4:
        return {"available": False}
    rate_at = dict(rows)
    anchor_payloads = frozen_curve_anchors(rows)
    anchors = [
        (endpoint_bytes(op, payload, lane.width), rate_at[payload])
        for payload in anchor_payloads
    ]
    floor_us = lane.floor_us(op)

    worst_signed = 0.0
    worst_payload = 0
    held_out: list[dict[str, Any]] = []
    for payload, _ in rows:
        if payload in anchor_payloads:
            continue
        endpoint = endpoint_bytes(op, payload, lane.width)
        modeled_us = floor_us + endpoint / curve_rate(anchors, endpoint) / 1e9 * 1e6
        measured_us = lane.time_us(op, payload)
        signed = (modeled_us - measured_us) / measured_us * 100.0
        held_out.append(
            {
                "payload_bytes": payload,
                "modeled_us": modeled_us,
                "measured_us": measured_us,
                "signed_error_percent": signed,
            }
        )
        if abs(signed) > abs(worst_signed):
            worst_signed = signed
            worst_payload = payload

    # What the shipped single-slope form would have done on the same held-out
    # split, anchored the way the A100 and GH200 studies anchored it: the
    # asymptotic serialization bandwidth at the largest measured payload. This
    # is the comparison that says whether the curve earns its complexity.
    flat_rate = rows[-1][1]
    flat_worst = 0.0
    flat_worst_payload = 0
    for payload, _ in rows:
        if payload in anchor_payloads:
            continue
        load = endpoint_bytes(op, payload, lane.width)
        modeled_us = floor_us + load / flat_rate / 1e9 * 1e6
        measured_us = lane.time_us(op, payload)
        signed = (modeled_us - measured_us) / measured_us * 100.0
        if abs(signed) > abs(flat_worst):
            flat_worst = signed
            flat_worst_payload = payload

    return {
        "available": True,
        "flat_slope_gbps": flat_rate,
        "flat_worst_signed_error_percent": flat_worst,
        "flat_worst_payload_bytes": flat_worst_payload,
        "anchor_payloads": anchor_payloads,
        "anchor_endpoint_bytes": [entry[0] for entry in anchors],
        "anchor_rates_gbps": [entry[1] for entry in anchors],
        "held_out": held_out,
        "worst_signed_error_percent": worst_signed,
        "worst_payload_bytes": worst_payload,
        "clears_bar": abs(worst_signed) <= CURVE_HELD_OUT_BAR_PERCENT,
    }


def score_guards(lanes: list[Lane]) -> list[Outcome]:
    """Return one Outcome per violated fatal guard. An empty list means held."""

    violations: list[Outcome] = []

    # G3 value conservation, and G2 timer sanity, over every measured cell.
    for lane in lanes:
        for (op, payload), cell in sorted(lane.cells.items()):
            mismatch = float(cell.get("max_rank_mismatches", 0.0))
            if mismatch != 0.0:
                violations.append(
                    Outcome(
                        "G3",
                        False,
                        f"{lane.label} {op} at {payload} B reported "
                        f"{mismatch:g} mismatching probes",
                        {"lane": lane.label, "op": op, "bytes": payload},
                    )
                )
            time_us = float(cell["time_us"])
            if not (time_us > 0.0 and math.isfinite(time_us)):
                violations.append(
                    Outcome(
                        "G2",
                        False,
                        f"{lane.label} {op} at {payload} B reported a "
                        f"nonpositive or nonfinite time of {time_us}",
                        {"lane": lane.label, "op": op, "bytes": payload},
                    )
                )

    # G5 no rate above the ceiling of the ports in play.
    for lane in lanes:
        ceiling = CASSINI_FOUR_PORT_GBPS if lane.width > 2 or lane.arm == "fournic" else CASSINI_PORT_GBPS
        for (op, payload), cell in sorted(lane.cells.items()):
            for key in ("algbw_gbps", "busbw_gbps", "per_rank_egress_gbps"):
                if key not in cell:
                    continue
                rate = float(cell[key])
                if rate > ceiling:
                    violations.append(
                        Outcome(
                            "G5",
                            False,
                            f"{lane.label} {op} at {payload} B reported "
                            f"{fmt(rate)} GB/s for {key}, above the "
                            f"{fmt(ceiling)} GB/s ceiling of the ports in play",
                            {"lane": lane.label, "op": op, "bytes": payload},
                        )
                    )

    # G6 declared rank placement.
    for lane in lanes:
        declared = {"w2-default": 2, "w2-fournic": 2, "w8-default": 8, "w4-default": 4}
        if lane.label in declared and lane.width != declared[lane.label]:
            violations.append(
                Outcome(
                    "G6",
                    False,
                    f"{lane.label} realized width {lane.width}, not "
                    f"{declared[lane.label]}",
                    {"lane": lane.label},
                )
            )

    return violations


def score(
    w2: Lane,
    w2_four: Lane | None,
    w8: Lane | None,
    w4: Lane | None,
) -> list[Outcome]:
    out: list[Outcome] = []

    # ---------------------------------------------------------------- block P
    p2p_floor = w2.floor_us("p2p")
    out.append(
        Outcome(
            "E-P-1",
            band(p2p_floor, 10.0, 200.0),
            f"8 B cross-node send and receive completion is {fmt(p2p_floor)} us, "
            f"band [10, 200]",
            {"value_us": p2p_floor},
        )
    )

    p2p_top = w2.field("p2p", 134217728, "algbw_gbps")
    out.append(
        Outcome(
            "E-P-2",
            band(p2p_top, 1.0, 15.0),
            f"point-to-point algorithm bandwidth at 128 MiB is {fmt(p2p_top)} GB/s, "
            f"band [1.0, 15.0], which is "
            f"{fmt(p2p_top / CASSINI_PORT_GBPS * 100.0, 1)} percent of the "
            f"{fmt(CASSINI_PORT_GBPS, 1)} GB/s port rate",
            {"value_gbps": p2p_top, "port_fraction_percent": p2p_top / CASSINI_PORT_GBPS * 100.0},
        )
    )

    ramp = [p for p in w2.payloads("p2p") if p >= 65536]
    times = [w2.time_us("p2p", p) for p in ramp]
    increasing = all(later > earlier for earlier, later in itertools.pairwise(times))
    first_break = next(
        (
            ramp[index + 1]
            for index in range(len(times) - 1)
            if times[index + 1] <= times[index]
        ),
        None,
    )
    out.append(
        Outcome(
            "E-P-3",
            increasing,
            "point-to-point completion is strictly increasing from 64 KiB to "
            "128 MiB"
            if increasing
            else f"point-to-point completion is not strictly increasing; the "
            f"first break is at {first_break} B",
            {"payloads": ramp, "times_us": times},
        )
    )

    # ---------------------------------------------------------------- block A
    ar2 = w2.floor_us("allreduce")
    ratio_intra = ar2 / INTRA_W2_FLOOR_US
    out.append(
        Outcome(
            "E-A-1",
            band(ar2, 15.0, 150.0) and ratio_intra >= 1.5,
            f"width-2 cross-node 8 B all-reduce floor is {fmt(ar2)} us, band "
            f"[15, 150], and {fmt(ratio_intra)} times the intra-node "
            f"{fmt(INTRA_W2_FLOOR_US)} us floor, required at least 1.5",
            {"value_us": ar2, "ratio_over_intra": ratio_intra},
        )
    )

    if w8 is not None:
        ar8 = w8.floor_us("allreduce")
        out.append(
            Outcome(
                "E-A-2",
                ar8 > ar2,
                f"width-8 floor {fmt(ar8)} us against width-2 floor "
                f"{fmt(ar2)} us",
                {"w8_us": ar8, "w2_us": ar2},
            )
        )
        out.append(
            Outcome(
                "E-A-3",
                band(ar8, 20.0, 250.0),
                f"width-8 cross-node 8 B all-reduce floor is {fmt(ar8)} us, "
                f"band [20, 250]",
                {"value_us": ar8},
            )
        )
    else:
        out.append(Outcome("E-A-2", False, "width-8 cell absent, relation unevaluated", {}, evaluated=False))
        out.append(Outcome("E-A-3", False, "width-8 cell absent, relation unevaluated", {}, evaluated=False))

    ratio_cross_w2 = ar2 / B200_CROSS_W2_US
    out.append(
        Outcome(
            "E-A-4",
            ar2 > B200_CROSS_W2_US and band(ratio_cross_w2, 1.2, 12.0),
            f"width-2 measured floor {fmt(ar2)} us over the shipped "
            f"cross-node provisional {fmt(B200_CROSS_W2_US)} us is "
            f"{fmt(ratio_cross_w2)}, band [1.2, 12], signed prediction was "
            f"underestimate",
            {"ratio": ratio_cross_w2, "measured_us": ar2, "shipped_us": B200_CROSS_W2_US},
        )
    )

    busbw2 = w2.field("allreduce", 134217728, "busbw_gbps")
    intra_fraction = busbw2 / INTRA_W2_BUSBW_GBPS * 100.0
    out.append(
        Outcome(
            "E-A-5",
            band(busbw2, 0.5, 15.0) and intra_fraction <= 25.0,
            f"width-2 asymptotic all-reduce bus bandwidth is {fmt(busbw2)} GB/s, "
            f"band [0.5, 15.0], and {fmt(intra_fraction, 1)} percent of the "
            f"intra-node {fmt(INTRA_W2_BUSBW_GBPS)} GB/s, required at most 25",
            {"value_gbps": busbw2, "intra_fraction_percent": intra_fraction},
        )
    )

    if w8 is not None:
        busbw8 = w8.field("allreduce", 134217728, "busbw_gbps")
        width_gain = busbw8 / busbw2
        out.append(
            Outcome(
                "E-A-6",
                band(width_gain, 1.5, 8.0),
                f"width-8 asymptotic bus bandwidth {fmt(busbw8)} GB/s over "
                f"width-2 {fmt(busbw2)} GB/s is {fmt(width_gain)}, band "
                f"[1.5, 8.0]",
                {"w8_gbps": busbw8, "w2_gbps": busbw2, "gain": width_gain},
            )
        )
        ar8 = w8.floor_us("allreduce")
        signed = (B200_CROSS_W8_US - ar8) / ar8 * 100.0
        out.append(
            Outcome(
                "E-A-7",
                band(ar8, 20.0, 250.0),
                f"width-8 floor {fmt(ar8)} us in band [20, 250]; the shipped "
                f"cross-node provisional {fmt(B200_CROSS_W8_US)} us sits "
                f"{fmt(signed, 1)} percent from it, sign reported and not scored",
                {"value_us": ar8, "shipped_us": B200_CROSS_W8_US, "shipped_signed_percent": signed},
            )
        )
    else:
        out.append(Outcome("E-A-6", False, "width-8 cell absent, relation unevaluated", {}, evaluated=False))
        out.append(Outcome("E-A-7", False, "width-8 cell absent, relation unevaluated", {}, evaluated=False))

    # ---------------------------------------------------------------- block T
    a2a2 = w2.floor_us("alltoallv")
    shape_ratio = a2a2 / ar2
    out.append(
        Outcome(
            "E-T-1",
            band(shape_ratio, 0.5, 2.0),
            f"width-2 all-to-allv floor {fmt(a2a2)} us over all-reduce floor "
            f"{fmt(ar2)} us is {fmt(shape_ratio)}, band [0.5, 2.0]",
            {"ratio": shape_ratio, "a2a_us": a2a2, "ar_us": ar2},
        )
    )

    if w8 is not None:
        a2a8 = w8.floor_us("alltoallv")
        ar8 = w8.floor_us("allreduce")
        ratio8 = a2a8 / ar8
        out.append(
            Outcome(
                "E-T-2",
                band(ratio8, 1.0, 6.0),
                f"width-8 all-to-allv floor {fmt(a2a8)} us over all-reduce "
                f"floor {fmt(ar8)} us is {fmt(ratio8)}, band [1.0, 6.0]",
                {"ratio": ratio8, "a2a_us": a2a8, "ar_us": ar8},
            )
        )
        # The 4 MiB per rank that leaves each node's four ranks for the other
        # node, at a 1 MiB per-pair payload and width 8.
        time_s = w8.time_us("alltoallv", 1048576) * 1e-6
        cross_rate = 4 * 1048576 / time_s / 1e9
        out.append(
            Outcome(
                "E-T-3",
                band(cross_rate, 1.0, 40.0),
                f"width-8 aggregate cross-node rate at a 1 MiB per-pair "
                f"payload is {fmt(cross_rate)} GB/s, band [1.0, 40.0]",
                {"value_gbps": cross_rate},
            )
        )
    else:
        out.append(Outcome("E-T-2", False, "width-8 cell absent, relation unevaluated", {}, evaluated=False))
        out.append(Outcome("E-T-3", False, "width-8 cell absent, relation unevaluated", {}, evaluated=False))

    # ---------------------------------------------------------------- block N
    if w2_four is not None:
        four_top = w2_four.field("p2p", 134217728, "algbw_gbps")
        port_gain = four_top / p2p_top
        out.append(
            Outcome(
                "E-N-1",
                band(port_gain, 1.2, 4.0),
                f"four-port point-to-point bandwidth {fmt(four_top)} GB/s over "
                f"single-port {fmt(p2p_top)} GB/s is {fmt(port_gain)}, band "
                f"[1.2, 4.0]",
                {"four_gbps": four_top, "default_gbps": p2p_top, "gain": port_gain},
            )
        )
        four_floor = w2_four.floor_us("p2p")
        drift = abs(four_floor - p2p_floor) / p2p_floor * 100.0
        out.append(
            Outcome(
                "E-N-2",
                drift <= 25.0,
                f"8 B point-to-point floor moves from {fmt(p2p_floor)} to "
                f"{fmt(four_floor)} us across the arms, {fmt(drift, 1)} percent, "
                f"allowed 25",
                {"default_us": p2p_floor, "fournic_us": four_floor, "drift_percent": drift},
            )
        )
    else:
        out.append(Outcome("E-N-1", False, "four-port arm absent, relation unevaluated", {}, evaluated=False))
        out.append(Outcome("E-N-2", False, "four-port arm absent, relation unevaluated", {}, evaluated=False))

    # ---------------------------------------------------------------- block C
    rows = serialization_bandwidth(w2, "allreduce")
    window = [(p, r) for p, r in rows if 262144 <= p <= 4194304]
    breaks = [
        (window[index + 1][0], window[index][1], window[index + 1][1])
        for index in range(len(window) - 1)
        if window[index + 1][1] < window[index][1] * 0.95
    ]
    out.append(
        Outcome(
            "E-C-1",
            not breaks,
            "width-2 serialization bandwidth is monotone within 5 percent from "
            "256 KiB to 4 MiB, so the cross-node socket path does not reproduce "
            "the intra-node dip"
            if not breaks
            else "width-2 serialization bandwidth is NOT monotone from 256 KiB "
            "to 4 MiB; the dip survives a change of transport. Breaks at "
            + ", ".join(
                f"{payload} B ({fmt(before)} to {fmt(after)} GB/s)"
                for payload, before, after in breaks
            ),
            {
                "window": [{"payload_bytes": p, "gbps": r} for p, r in window],
                "breaks": [
                    {"payload_bytes": p, "before_gbps": b, "after_gbps": a}
                    for p, b, a in breaks
                ],
            },
        )
    )

    # ---------------------------------------------------------------- block M
    if w8 is not None:
        ar8 = w8.floor_us("allreduce")
        ratio_local8 = ar8 / B200_LOCAL_W8_US
        out.append(
            Outcome(
                "E-M-1",
                ar8 > B200_LOCAL_W8_US and band(ratio_local8, 1.2, 8.0),
                f"width-8 measured floor {fmt(ar8)} us over the shipped local "
                f"width-8 intercept {fmt(B200_LOCAL_W8_US)} us is "
                f"{fmt(ratio_local8)}, band [1.2, 8.0], signed prediction was "
                f"underestimate",
                {"ratio": ratio_local8, "measured_us": ar8},
            )
        )
    else:
        out.append(Outcome("E-M-1", False, "width-8 cell absent, relation unevaluated", {}, evaluated=False))

    ratio_local2 = ar2 / B200_LOCAL_W2_US
    out.append(
        Outcome(
            "E-M-2",
            ar2 > B200_LOCAL_W2_US and band(ratio_local2, 1.5, 12.0),
            f"width-2 measured floor {fmt(ar2)} us over the shipped local "
            f"width-2 intercept {fmt(B200_LOCAL_W2_US)} us is "
            f"{fmt(ratio_local2)}, band [1.5, 12.0], signed prediction was "
            f"underestimate",
            {"ratio": ratio_local2, "measured_us": ar2},
        )
    )

    # Reported, not scored. The freeze's Block P header says "point-to-point ramp
    # at width 2" and gives the four-port arm its own block, so E-P-3 is scored
    # on the default arm. The same statement is false on the four-port arm, at
    # exactly the protocol boundary, and that belongs in the record rather than
    # only in the prose.
    if w2_four is not None:
        four_ramp = [payload for payload in w2_four.payloads("p2p") if payload >= 65536]
        four_times = [w2_four.time_us("p2p", payload) for payload in four_ramp]
        four_breaks = [
            four_ramp[index + 1]
            for index in range(len(four_times) - 1)
            if four_times[index + 1] <= four_times[index]
        ]
        out.append(
            Outcome(
                "REPORTED-fournic-p2p-monotonicity",
                not four_breaks,
                "the four-port point-to-point ramp is strictly increasing"
                if not four_breaks
                else "the four-port point-to-point ramp is NOT strictly "
                f"increasing; it falls at {four_breaks}, which is the same "
                "LL to SIMPLE boundary the all-reduce falls at",
                {"breaks": four_breaks},
            )
        )
        # Port gain is not one number per stack: it depends on the operation.
        gains = {}
        for op, key in (
            ("p2p", "algbw_gbps"),
            ("allreduce", "busbw_gbps"),
            ("alltoallv", "per_rank_egress_gbps"),
        ):
            gains[op] = w2_four.field(op, 134217728, key) / w2.field(op, 134217728, key)
        out.append(
            Outcome(
                "REPORTED-port-gain-by-operation",
                True,
                "four ports over one, at 128 MiB: "
                + ", ".join(f"{op} {fmt(gain)}x" for op, gain in gains.items())
                + ", so the gain from adding ports is a property of the "
                "operation and not one efficiency scalar per stack",
                {"gains": gains},
            )
        )

    if w4 is not None:
        out.append(
            Outcome(
                "OPTIONAL-w4",
                True,
                f"the optional four-node width-4 cell ran: 8 B all-reduce floor "
                f"{fmt(w4.floor_us('allreduce'))} us",
                {"value_us": w4.floor_us("allreduce")},
            )
        )

    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--w2-default", type=Path, required=True)
    parser.add_argument("--w2-fournic", type=Path)
    parser.add_argument("--w8-default", type=Path)
    parser.add_argument("--w4-default", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    w2 = Lane(json.loads(args.w2_default.read_text()), "w2-default")
    lanes = [w2]
    w2_four = None
    if args.w2_fournic is not None and args.w2_fournic.exists():
        w2_four = Lane(json.loads(args.w2_fournic.read_text()), "w2-fournic")
        lanes.append(w2_four)
    w8 = None
    if args.w8_default is not None and args.w8_default.exists():
        w8 = Lane(json.loads(args.w8_default.read_text()), "w8-default")
        lanes.append(w8)
    w4 = None
    if args.w4_default is not None and args.w4_default.exists():
        w4 = Lane(json.loads(args.w4_default.read_text()), "w4-default")
        lanes.append(w4)

    fatal = score_guards(lanes)
    outcomes = score(w2, w2_four, w8, w4)
    scored = [entry for entry in outcomes if entry.ident.startswith("E-")]
    extras = [entry for entry in outcomes if not entry.ident.startswith("E-")]

    curves = {
        lane.label: {op: held_out_error(lane, op) for op in ("allreduce", "alltoallv")}
        for lane in lanes
    }

    evaluated = [entry for entry in scored if entry.evaluated]
    unevaluated = [entry for entry in scored if not entry.evaluated]
    passed = sum(1 for entry in evaluated if entry.passed)
    print("Fatal guards")
    if not fatal:
        print("  every fatal guard held")
    for entry in fatal:
        print(f"  VIOLATED {entry.ident}: {entry.detail}")
    print()
    print(f"Scored relations {passed} of {len(evaluated)} evaluated")
    for entry in evaluated:
        mark = "pass" if entry.passed else "FAIL"
        print(f"  {mark} {entry.ident}: {entry.detail}")
    if unevaluated:
        print()
        print(
            f"Unevaluated for want of a measurement: {len(unevaluated)} of the "
            f"{len(scored)} frozen relations"
        )
        for entry in unevaluated:
            print(f"  none {entry.ident}: {entry.detail}")
    if extras:
        print()
        print("Reported, not scored")
        for entry in extras:
            print(f"  {entry.ident}: {entry.detail}")
    print()
    print("Frozen curve anchors and held-out error")
    for label, per_op in curves.items():
        for op, info in per_op.items():
            if not info.get("available"):
                continue
            print(
                f"  {label} {op}: anchors {info['anchor_payloads']}, worst "
                f"held-out {fmt(info['worst_signed_error_percent'], 2)} percent "
                f"at {info['worst_payload_bytes']} B, "
                f"{'clears' if info['clears_bar'] else 'misses'} the "
                f"{CURVE_HELD_OUT_BAR_PERCENT:.0f} percent bar; the single "
                f"slope would be {fmt(info['flat_worst_signed_error_percent'], 2)} "
                f"percent at {info['flat_worst_payload_bytes']} B"
            )

    report = {
        "study": "crossnode_collective_envelope_v1",
        "frozen_relation_total": len(scored),
        "scored_evaluated": len(evaluated),
        "scored_passed": passed,
        "unevaluated": [entry.ident for entry in unevaluated],
        "fatal_violations": len(fatal),
        "void": bool(fatal),
        "relations": [
            {
                "id": entry.ident,
                "passed": entry.passed,
                "evaluated": entry.evaluated,
                "detail": entry.detail,
                "observed": entry.observed,
            }
            for entry in scored
        ],
        "reported_not_scored": [
            {"id": entry.ident, "detail": entry.detail, "observed": entry.observed}
            for entry in extras
        ],
        "fatal": [
            {"id": entry.ident, "detail": entry.detail, "observed": entry.observed}
            for entry in fatal
        ],
        "curves": curves,
    }
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print()
    print(f"wrote {args.out}")
    return 0 if not fatal else 1


if __name__ == "__main__":
    raise SystemExit(main())
