"""Score the A100 hardware envelope measurements against the frozen freeze.

The bounds and relations checked here are transcribed from expectations.md,
which was committed before the harness existed. This script reads the raw lane
results, evaluates each expectation, evaluates each fatal guard separately, and
prints a report. It never edits the freeze.

Usage::

    python score_expectations.py --lane-a lane_a_result.json \\
        --lane-b lane_b_result.json --out scored.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Nameplate constants derived in the freeze from the recon inventory.
HBM_PEAK_GBPS = 2039.04
BF16_PEAK_TFLOPS = 311.87
NVLINK_PAIR_GBPS = 100.0
NVLINK_EGRESS_GBPS = 300.0
PCIE_GBPS = 31.5
IDEAL_CROSSOVER_M = 158.9

# Repository surrogate constants the comparison lane scores against.
DEFAULT_NVLINK_BANDWIDTH_GBPS = 450.0
B200_LOCAL_BANDWIDTH_GBPS = 70.027079100
B200_LOCAL_ALPHA_W4_US = 15.745167


@dataclass
class Outcome:
    """One evaluated expectation or fatal guard."""

    ident: str
    passed: bool
    detail: str
    observed: dict[str, Any] = field(default_factory=dict)


def band(value: float, low: float, high: float) -> bool:
    return low <= value <= high


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def ols_fit(sizes: list[float], times_s: list[float]) -> tuple[float, float, float]:
    """Return ``(alpha_seconds, beta_bytes_per_second, r_squared)``.

    The model is ``t = alpha + S / beta`` fitted by ordinary least squares of
    ``t`` on ``S``.
    """

    n = len(sizes)
    mean_x = sum(sizes) / n
    mean_y = sum(times_s) / n
    sxx = sum((x - mean_x) ** 2 for x in sizes)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(sizes, times_s))
    slope = sxy / sxx
    alpha = mean_y - slope * mean_x
    predicted = [alpha + slope * x for x in sizes]
    ss_res = sum((y - p) ** 2 for y, p in zip(times_s, predicted))
    ss_tot = sum((y - mean_y) ** 2 for y in times_s)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    beta = 1.0 / slope if slope > 0 else float("inf")
    return alpha, beta, r_squared


def hbm_point(lane_a: dict, size_mib: int) -> dict:
    for row in lane_a["hbm"]:
        if row["size_mib"] == size_mib:
            return row
    raise KeyError(f"no HBM point at {size_mib} MiB")


def gemm_points(lane_a: dict, sweep: str) -> list[dict]:
    return [row for row in lane_a["gemm"] if row["sweep"] == sweep]


def collective(lane_b: dict, op: str, width: int, size_bytes: int) -> dict | None:
    for row in lane_b["collectives"]:
        if (
            row.get("op") == op
            and row.get("width") == width
            and row.get("bytes") == size_bytes
            and row.get("status") == "measured"
        ):
            return row
    return None


def collective_series(lane_b: dict, op: str, width: int) -> list[dict]:
    rows = [
        row
        for row in lane_b["collectives"]
        if row.get("op") == op and row.get("width") == width and row.get("status") == "measured"
    ]
    return sorted(rows, key=lambda row: row["bytes"])


def score_lane_a(lane_a: dict) -> tuple[list[Outcome], list[Outcome]]:
    scored: list[Outcome] = []
    fatal: list[Outcome] = []

    big = hbm_point(lane_a, 4096)
    scored.append(
        Outcome(
            "E-A1-1",
            band(big["read_gbps"], 1400.0, HBM_PEAK_GBPS),
            f"4 GiB read {fmt(big['read_gbps'])} GB/s in [1400, {HBM_PEAK_GBPS}]",
            {"read_gbps": big["read_gbps"]},
        )
    )
    scored.append(
        Outcome(
            "E-A1-2",
            band(big["write_gbps"], 1000.0, HBM_PEAK_GBPS),
            f"4 GiB write {fmt(big['write_gbps'])} GB/s in [1000, {HBM_PEAK_GBPS}]",
            {"write_gbps": big["write_gbps"]},
        )
    )
    scored.append(
        Outcome(
            "E-A1-3",
            band(big["copy_gbps"], 1200.0, HBM_PEAK_GBPS),
            f"4 GiB copy {fmt(big['copy_gbps'])} GB/s in [1200, {HBM_PEAK_GBPS}]",
            {"copy_gbps": big["copy_gbps"]},
        )
    )

    plateau = [hbm_point(lane_a, mib)["copy_gbps"] for mib in (1024, 2048, 4096)]
    plateau_median = sorted(plateau)[1]
    spread = max(abs(v - plateau_median) / plateau_median for v in plateau)
    scored.append(
        Outcome(
            "E-A1-4",
            spread <= 0.05,
            f"copy flatness over 1/2/4 GiB is {fmt(spread * 100, 2)} percent, limit 5",
            {"values_gbps": plateau, "spread": spread},
        )
    )

    l2 = hbm_point(lane_a, 8)
    ratio = l2["copy_gbps"] / big["copy_gbps"]
    scored.append(
        Outcome(
            "E-A1-5",
            ratio >= 1.2 and l2["copy_gbps"] > HBM_PEAK_GBPS,
            f"8 MiB copy {fmt(l2['copy_gbps'])} GB/s is {fmt(ratio, 2)}x the 4 GiB point "
            f"and above the {HBM_PEAK_GBPS} GB/s HBM ceiling",
            {"l2_copy_gbps": l2["copy_gbps"], "ratio": ratio},
        )
    )

    for row in lane_a["hbm"]:
        if row["bytes"] >= (256 << 20):
            for key, moved in (("read_gbps", 1), ("write_gbps", 1), ("copy_gbps", 2)):
                if row[key] > HBM_PEAK_GBPS:
                    fatal.append(
                        Outcome(
                            "F2",
                            False,
                            f"HBM {key} at {row['size_mib']} MiB is {fmt(row[key])} GB/s, "
                            f"above the {HBM_PEAK_GBPS} GB/s ceiling",
                            {"size_mib": row["size_mib"], key: row[key], "moved": moved},
                        )
                    )

    decode = gemm_points(lane_a, "decode")
    for row in lane_a["gemm"]:
        floor_ms = max(row["t_mem_ms"], row["t_flop_ms"])
        if row["time_ms"] < floor_ms:
            fatal.append(
                Outcome(
                    "F3",
                    False,
                    f"GEMM {row['m']}x{row['n']}x{row['k']} took {fmt(row['time_ms'], 4)} ms, "
                    f"below its floor {fmt(floor_ms, 4)} ms",
                    {"m": row["m"], "time_ms": row["time_ms"], "floor_ms": floor_ms},
                )
            )

    small = [row for row in decode if row["m"] <= 64]
    worst_tflops = max(row["tflops_achieved"] for row in small)
    scored.append(
        Outcome(
            "E-A2-1",
            worst_tflops < 5.0,
            f"memory-bound plateau peaks at {fmt(worst_tflops, 3)} TFLOP/s, limit 5",
            {"max_tflops": worst_tflops},
        )
    )
    times = [row["time_ms"] for row in small]
    plateau_ratio = max(times) / min(times)
    scored.append(
        Outcome(
            "E-A2-2",
            plateau_ratio <= 1.6,
            f"plateau max/min over M<=64 is {fmt(plateau_ratio, 3)}, limit 1.6",
            {"ratio": plateau_ratio},
        )
    )
    within = all(row["t_mem_ms"] <= row["time_ms"] <= 3.0 * row["t_mem_ms"] for row in small)
    worst_multiple = max(row["time_ms"] / row["t_mem_ms"] for row in small)
    scored.append(
        Outcome(
            "E-A2-3",
            within,
            f"every M<=64 time is between 1.0x and 3.0x its memory floor; worst is "
            f"{fmt(worst_multiple, 3)}x",
            {"worst_multiple": worst_multiple},
        )
    )

    sorted_times = sorted(times)
    plateau_med = (
        sorted_times[len(sorted_times) // 2]
        if len(sorted_times) % 2 == 1
        else 0.5 * (sorted_times[len(sorted_times) // 2 - 1] + sorted_times[len(sorted_times) // 2])
    )
    crossover = None
    for row in sorted(decode, key=lambda r: r["m"]):
        if row["m"] > 64 and row["time_ms"] > 1.5 * plateau_med:
            crossover = row["m"]
            break
    scored.append(
        Outcome(
            "E-A2-4",
            crossover is not None and 128 <= crossover <= 1024,
            f"measured crossover M={crossover} in [128, 1024], ideal {IDEAL_CROSSOVER_M}",
            {"crossover_m": crossover, "plateau_median_ms": plateau_med},
        )
    )

    large = [row for row in decode if row["m"] >= 2048]
    worst_large = min(row["tflops_achieved"] for row in large)
    scored.append(
        Outcome(
            "E-A2-5",
            worst_large >= 150.0,
            f"worst M>=2048 rate is {fmt(worst_large, 2)} TFLOP/s, floor 150",
            {"min_tflops": worst_large},
        )
    )
    m4096 = next(row for row in decode if row["m"] == 4096)
    m8192 = next(row for row in decode if row["m"] == 8192)
    scale = m8192["time_ms"] / m4096["time_ms"]
    scored.append(
        Outcome(
            "E-A2-6",
            band(scale, 1.7, 2.3),
            f"time ratio M=8192 over M=4096 is {fmt(scale, 3)} in [1.7, 2.3]",
            {"ratio": scale},
        )
    )
    square8192 = next(row for row in gemm_points(lane_a, "square") if row["m"] == 8192)
    scored.append(
        Outcome(
            "E-A2-7",
            band(square8192["tflops_achieved"], 180.0, 300.0),
            f"square 8192 achieves {fmt(square8192['tflops_achieved'], 2)} TFLOP/s in [180, 300]",
            {"tflops": square8192["tflops_achieved"]},
        )
    )

    launch = lane_a["launch"]
    scored.append(
        Outcome(
            "E-A3-1",
            band(launch["pipelined_period_us"], 0.5, 10.0),
            f"pipelined launch period {fmt(launch['pipelined_period_us'], 3)} us in [0.5, 10]",
            {"us": launch["pipelined_period_us"]},
        )
    )
    scored.append(
        Outcome(
            "E-A3-2",
            band(launch["roundtrip_us"], 3.0, 40.0),
            f"launch roundtrip {fmt(launch['roundtrip_us'], 3)} us in [3, 40]",
            {"us": launch["roundtrip_us"]},
        )
    )
    scored.append(
        Outcome(
            "E-A3-3",
            launch["roundtrip_us"] > launch["pipelined_period_us"],
            f"roundtrip {fmt(launch['roundtrip_us'], 3)} us exceeds pipelined "
            f"{fmt(launch['pipelined_period_us'], 3)} us",
            {},
        )
    )
    scored.append(
        Outcome(
            "E-A3-4",
            launch["graph_period_us"] < launch["pipelined_period_us"],
            f"graph replay period {fmt(launch['graph_period_us'], 3)} us is below pipelined "
            f"{fmt(launch['pipelined_period_us'], 3)} us",
            {"graph_us": launch["graph_period_us"]},
        )
    )

    pcie = lane_a["pcie"]
    scored.append(
        Outcome(
            "E-A4-1",
            band(pcie["h2d_gbps"], 15.0, PCIE_GBPS),
            f"host to device {fmt(pcie['h2d_gbps'], 3)} GB/s in [15, {PCIE_GBPS}]",
            {"gbps": pcie["h2d_gbps"]},
        )
    )
    scored.append(
        Outcome(
            "E-A4-2",
            band(pcie["d2h_gbps"], 15.0, PCIE_GBPS),
            f"device to host {fmt(pcie['d2h_gbps'], 3)} GB/s in [15, {PCIE_GBPS}]",
            {"gbps": pcie["d2h_gbps"]},
        )
    )
    for key in ("h2d_gbps", "d2h_gbps"):
        if pcie[key] > PCIE_GBPS:
            fatal.append(
                Outcome("F2", False, f"PCIe {key} {fmt(pcie[key], 3)} above {PCIE_GBPS}", {}))

    if lane_a["visible_device_count"] != 1:
        fatal.append(
            Outcome(
                "F1",
                False,
                f"lane A saw {lane_a['visible_device_count']} devices, expected 1",
                {},
            )
        )
    if "A100-SXM4-80GB" not in lane_a["device_name"]:
        fatal.append(Outcome("F1", False, f"unexpected device {lane_a['device_name']}", {}))

    missing_clocks = 0
    for row in lane_a["hbm"] + lane_a["gemm"]:
        if not row["clocks_before"]["valid"] or not row["clocks_after"]["valid"]:
            missing_clocks += 1
    for block in ("launch", "pcie"):
        blk = lane_a[block]
        if not blk["clocks_before"]["valid"] or not blk["clocks_after"]["valid"]:
            missing_clocks += 1
    if missing_clocks:
        fatal.append(
            Outcome("F7", False, f"{missing_clocks} timed blocks lack clock observations", {}))

    return scored, fatal


def score_lane_b(lane_b: dict) -> tuple[list[Outcome], list[Outcome]]:
    scored: list[Outcome] = []
    fatal: list[Outcome] = []

    uni = lane_b["p2p_unidirectional"]
    values = [row["gbps"] for row in uni]
    scored.append(
        Outcome(
            "E-B1-1",
            all(band(v, 75.0, NVLINK_PAIR_GBPS) for v in values),
            f"12 ordered pairs span [{fmt(min(values), 2)}, {fmt(max(values), 2)}] GB/s, "
            f"band [75, {NVLINK_PAIR_GBPS}]",
            {"min": min(values), "max": max(values)},
        )
    )
    med = sorted(values)[len(values) // 2]
    spread = max(abs(v - med) / med for v in values)
    scored.append(
        Outcome(
            "E-B1-2",
            spread <= 0.10,
            f"pair symmetry spread {fmt(spread * 100, 2)} percent, limit 10",
            {"median": med, "spread": spread},
        )
    )
    bidir = lane_b["p2p_bidirectional_0_1"]
    scored.append(
        Outcome(
            "E-B1-3",
            band(bidir["aggregate_gbps"], 150.0, 200.0),
            f"bidirectional pair sum {fmt(bidir['aggregate_gbps'], 2)} GB/s in [150, 200]",
            {"gbps": bidir["aggregate_gbps"]},
        )
    )
    fan = lane_b["p2p_fanout_0_to_all"]
    scored.append(
        Outcome(
            "E-B1-4",
            band(fan["aggregate_gbps"], 225.0, NVLINK_EGRESS_GBPS),
            f"fan-out egress {fmt(fan['aggregate_gbps'], 2)} GB/s in [225, {NVLINK_EGRESS_GBPS}]",
            {"gbps": fan["aggregate_gbps"]},
        )
    )
    fan_ratio = fan["aggregate_gbps"] / med
    scored.append(
        Outcome(
            "E-B1-5",
            fan_ratio >= 2.4,
            f"fan-out is {fmt(fan_ratio, 3)}x one pair, floor 2.4",
            {"ratio": fan_ratio},
        )
    )
    for row in uni:
        if row["gbps"] > NVLINK_PAIR_GBPS:
            fatal.append(
                Outcome(
                    "F2",
                    False,
                    f"pair {row['src']}->{row['dst']} at {fmt(row['gbps'], 2)} GB/s exceeds "
                    f"{NVLINK_PAIR_GBPS}",
                    {},
                )
            )
    if bidir["aggregate_gbps"] > 200.0:
        fatal.append(Outcome("F2", False, "bidirectional pair sum above 200 GB/s", {}))
    if fan["aggregate_gbps"] > NVLINK_EGRESS_GBPS:
        fatal.append(Outcome("F2", False, "fan-out egress above 300 GB/s", {}))

    one_gib = 1 << 30
    ar2 = collective(lane_b, "allreduce", 2, one_gib)
    ar4 = collective(lane_b, "allreduce", 4, one_gib)
    scored.append(
        Outcome(
            "E-B2-1",
            band(ar2["busbw_gbps"], 70.0, 100.0),
            f"width 2 all-reduce busbw at 1 GiB {fmt(ar2['busbw_gbps'], 2)} GB/s in [70, 100]",
            {"busbw": ar2["busbw_gbps"]},
        )
    )
    scored.append(
        Outcome(
            "E-B2-2",
            band(ar4["busbw_gbps"], 130.0, 300.0),
            f"width 4 all-reduce busbw at 1 GiB {fmt(ar4['busbw_gbps'], 2)} GB/s in [130, 300]",
            {"busbw": ar4["busbw_gbps"]},
        )
    )
    width_ratio = ar4["busbw_gbps"] / ar2["busbw_gbps"]
    scored.append(
        Outcome(
            "E-B2-3",
            width_ratio >= 1.4,
            f"width 4 over width 2 busbw is {fmt(width_ratio, 3)}x, floor 1.4",
            {"ratio": width_ratio},
        )
    )

    agreement_detail = []
    agreement_ok = True
    for width in (2, 4):
        trio = []
        for op in ("allreduce", "allgather", "reducescatter"):
            row = collective(lane_b, op, width, one_gib)
            trio.append(row["busbw_gbps"])
        mean = sum(trio) / len(trio)
        dev = max(abs(v - mean) / mean for v in trio)
        agreement_detail.append(f"width {width} deviation {fmt(dev * 100, 2)} percent")
        agreement_ok = agreement_ok and dev <= 0.30
    scored.append(
        Outcome(
            "E-B2-4",
            agreement_ok,
            "collective busbw agreement at 1 GiB: " + ", ".join(agreement_detail) + ", limit 30",
            {},
        )
    )

    fits: dict[int, tuple[float, float, float]] = {}
    for width in (2, 4):
        rows = [
            row
            for row in collective_series(lane_b, "allreduce", width)
            if (1 << 20) <= row["bytes"] <= one_gib
        ]
        sizes = [float(row["bytes"]) for row in rows]
        times = [row["time_us"] * 1e-6 for row in rows]
        fits[width] = ols_fit(sizes, times)
    scored.append(
        Outcome(
            "E-B2-5",
            all(fits[w][2] >= 0.99 for w in (2, 4)),
            f"linear fit R squared is {fmt(fits[2][2], 5)} at width 2 and "
            f"{fmt(fits[4][2], 5)} at width 4, floor 0.99",
            {"r2_w2": fits[2][2], "r2_w4": fits[4][2]},
        )
    )
    alpha2_us = fits[2][0] * 1e6
    alpha4_us = fits[4][0] * 1e6
    scored.append(
        Outcome(
            "E-B2-6",
            band(alpha2_us, 2.0, 20.0) and band(alpha4_us, 3.0, 40.0) and alpha4_us > alpha2_us,
            f"fitted alpha is {fmt(alpha2_us, 3)} us at width 2 and {fmt(alpha4_us, 3)} us at "
            f"width 4, bands [2, 20] and [3, 40] with width 4 larger",
            {"alpha2_us": alpha2_us, "alpha4_us": alpha4_us},
        )
    )
    tiny2 = collective(lane_b, "allreduce", 2, 8)
    tiny4 = collective(lane_b, "allreduce", 4, 8)
    scored.append(
        Outcome(
            "E-B2-7",
            band(tiny2["time_us"], 2.0, 40.0)
            and band(tiny4["time_us"], 2.0, 40.0)
            and tiny4["time_us"] > tiny2["time_us"],
            f"8 B all-reduce is {fmt(tiny2['time_us'], 3)} us at width 2 and "
            f"{fmt(tiny4['time_us'], 3)} us at width 4",
            {"w2_us": tiny2["time_us"], "w4_us": tiny4["time_us"]},
        )
    )
    ratio_detail = []
    ratio_ok = True
    for width in (2, 4):
        half = collective(lane_b, "allreduce", width, 512 << 20)
        full = collective(lane_b, "allreduce", width, one_gib)
        ratio = full["time_us"] / half["time_us"]
        ratio_detail.append(f"width {width} ratio {fmt(ratio, 3)}")
        ratio_ok = ratio_ok and band(ratio, 1.8, 2.2)
    scored.append(
        Outcome(
            "E-B2-8", ratio_ok, "1 GiB over 512 MiB: " + ", ".join(ratio_detail) + ", band [1.8, 2.2]", {}
        )
    )
    quarter = 256 << 20
    ar = collective(lane_b, "allreduce", 4, quarter)
    rs = collective(lane_b, "reducescatter", 4, quarter)
    ag = collective(lane_b, "allgather", 4, quarter)
    composition = (rs["time_us"] + ag["time_us"])
    comp_ratio = ar["time_us"] / composition
    scored.append(
        Outcome(
            "E-B2-9",
            band(comp_ratio, 0.7, 1.3),
            f"all-reduce over reduce-scatter plus all-gather is {fmt(comp_ratio, 3)} in [0.7, 1.3]",
            {"allreduce_us": ar["time_us"], "composition_us": composition},
        )
    )

    for row in lane_b["collectives"]:
        # The contention cell is a timing comparison, not a bandwidth point, so
        # it carries no bus bandwidth to check against a ceiling.
        if row.get("status") != "measured" or "busbw_gbps" not in row:
            continue
        ceiling = NVLINK_PAIR_GBPS if row["width"] == 2 else NVLINK_EGRESS_GBPS
        if row["busbw_gbps"] > ceiling:
            fatal.append(
                Outcome(
                    "F2",
                    False,
                    f"{row['op']} width {row['width']} at {row['bytes']} B has busbw "
                    f"{fmt(row['busbw_gbps'], 2)} GB/s above {ceiling}",
                    {},
                )
            )
        if row["op"] == "allreduce" and row.get("allreduce_mismatching_ranks", 0) != 0:
            fatal.append(
                Outcome(
                    "F4",
                    False,
                    f"all-reduce width {row['width']} at {row['bytes']} B returned a wrong sum on "
                    f"{row['allreduce_mismatching_ranks']} ranks",
                    {},
                )
            )

    contention = next(
        (row for row in lane_b["collectives"] if row.get("op") == "contention"), None
    )
    t0 = contention["comm_alone_us"]
    g0 = contention["gemm_alone_us"]
    t1 = contention["comm_together_us"]
    g1 = contention["gemm_together_us"]
    scored.append(
        Outcome(
            "E-B3-1",
            t1 >= 1.02 * t0 and g1 >= 1.02 * g0,
            f"under contention the collective grows {fmt(t1 / t0, 3)}x and the GEMM grows "
            f"{fmt(g1 / g0, 3)}x, floor 1.02 each",
            {"comm_ratio": t1 / t0, "gemm_ratio": g1 / g0},
        )
    )
    scored.append(
        Outcome(
            "E-B3-2",
            t1 <= 3.0 * t0 and g1 <= 2.0 * g0,
            f"contention growth stays under the 3.0x collective and 2.0x GEMM ceilings at "
            f"{fmt(t1 / t0, 3)}x and {fmt(g1 / g0, 3)}x",
            {},
        )
    )
    total_alone = contention["alone_sum_us"]
    comm_total_alone = t0 * contention["comm_iters"]
    gemm_total_alone = g0 * contention["gemm_iters"]
    makespan = contention["makespan_us"]
    scored.append(
        Outcome(
            "E-B3-3",
            makespan >= max(comm_total_alone, gemm_total_alone) and makespan <= 0.95 * total_alone,
            f"makespan {fmt(makespan, 1)} us sits between max alone "
            f"{fmt(max(comm_total_alone, gemm_total_alone), 1)} us and 0.95 of the serial sum "
            f"{fmt(0.95 * total_alone, 1)} us",
            {
                "makespan_us": makespan,
                "comm_total_alone_us": comm_total_alone,
                "gemm_total_alone_us": gemm_total_alone,
            },
        )
    )

    scored.append(
        Outcome(
            "E-C-1",
            fan["aggregate_gbps"] * 1.5 <= DEFAULT_NVLINK_BANDWIDTH_GBPS,
            f"the {DEFAULT_NVLINK_BANDWIDTH_GBPS} GB/s flat surrogate is "
            f"{fmt(DEFAULT_NVLINK_BANDWIDTH_GBPS / fan['aggregate_gbps'], 3)}x the measured "
            f"{fmt(fan['aggregate_gbps'], 2)} GB/s egress, floor 1.5x",
            {"surrogate_over_measured": DEFAULT_NVLINK_BANDWIDTH_GBPS / fan["aggregate_gbps"]},
        )
    )
    small_rows = [
        row
        for row in collective_series(lane_b, "allreduce", 4)
        if row["bytes"] <= (1 << 18)
    ]
    small_alpha, small_beta, small_r2 = ols_fit(
        [float(row["bytes"]) for row in small_rows],
        [row["time_us"] * 1e-6 for row in small_rows],
    )
    small_beta_gbps = small_beta / 1e9
    scored.append(
        Outcome(
            "E-C-2",
            band(small_beta_gbps, 2.0, B200_LOCAL_BANDWIDTH_GBPS),
            f"small-message beta on this node is {fmt(small_beta_gbps, 3)} GB/s, band "
            f"[2, {B200_LOCAL_BANDWIDTH_GBPS}] set by the B200 profile",
            {"beta_gbps": small_beta_gbps, "alpha_us": small_alpha * 1e6, "r2": small_r2},
        )
    )
    scored.append(
        Outcome(
            "E-C-3",
            alpha4_us > B200_LOCAL_ALPHA_W4_US,
            f"width 4 alpha here is {fmt(alpha4_us, 3)} us against the B200 profile's "
            f"{B200_LOCAL_ALPHA_W4_US} us",
            {"alpha4_us": alpha4_us},
        )
    )

    if lane_b["visible_device_count"] != 4:
        fatal.append(
            Outcome(
                "F1",
                False,
                f"lane B saw {lane_b['visible_device_count']} devices, expected 4",
                {},
            )
        )
    for dev in lane_b["devices"]:
        if "A100-SXM4-80GB" not in dev["name"]:
            fatal.append(Outcome("F1", False, f"unexpected device {dev['name']}", {}))

    return scored, fatal


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane-a", type=Path, required=True)
    parser.add_argument("--lane-b", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    lane_a = json.loads(args.lane_a.read_text())
    lane_b = json.loads(args.lane_b.read_text())

    scored_a, fatal_a = score_lane_a(lane_a)
    scored_b, fatal_b = score_lane_b(lane_b)
    scored = scored_a + scored_b
    fatal = fatal_a + fatal_b

    passed = sum(1 for outcome in scored if outcome.passed)
    print("Fatal guards")
    if not fatal:
        print("  every fatal guard held")
    for outcome in fatal:
        print(f"  VIOLATED {outcome.ident}: {outcome.detail}")
    print()
    print(f"Scored expectations {passed} of {len(scored)}")
    for outcome in scored:
        mark = "pass" if outcome.passed else "FAIL"
        print(f"  {mark} {outcome.ident}: {outcome.detail}")

    report = {
        "study": "a100_hardware_envelope_v1",
        "scored_total": len(scored),
        "scored_passed": passed,
        "fatal_violations": len(fatal),
        "void": bool(fatal),
        "expectations": [
            {
                "id": outcome.ident,
                "passed": outcome.passed,
                "detail": outcome.detail,
                "observed": outcome.observed,
            }
            for outcome in scored
        ],
        "fatal": [
            {"id": outcome.ident, "detail": outcome.detail, "observed": outcome.observed}
            for outcome in fatal
        ],
    }
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print()
    print(f"wrote {args.out}")
    return 0 if not fatal else 1


if __name__ == "__main__":
    raise SystemExit(main())
