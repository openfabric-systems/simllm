"""Score the GH200 hardware envelope measurements against the frozen freeze.

The bounds and relations checked here are transcribed from expectations.md,
which was committed before the ported harness was committed and before any
GH200 job ran. This script reads the raw lane results, evaluates each
expectation, evaluates each fatal guard separately, and prints a report. It
never edits the freeze.

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

# Nameplate constants derived in the freeze from the GH200 recon inventory.
HBM_PEAK_GBPS = 4022.78
BF16_PEAK_TFLOPS = 1070.53
BF16_FLOP_PER_SM_CYCLE = 4096.0
NVLINK_PAIR_GBPS = 159.375
NVLINK_EGRESS_GBPS = 478.125
NVLINK_BIDIR_PAIR_GBPS = 318.75
C2C_GBPS = 450.0
IDEAL_CROSSOVER_M = 284.6

# Repository surrogate constant the comparison lane scores against.
DEFAULT_NVLINK_BANDWIDTH_GBPS = 450.0

# Previously published A100 measurements, quoted by the freeze for comparison.
A100_PIPELINED_LAUNCH_US = 1.806
A100_H2D_GBPS = 26.776
A100_D2H_GBPS = 26.189
A100_ALLREDUCE_W4_BUSBW_GBPS = 212.89
A100_RING_EFFICIENCY_W4 = 0.710
A100_TINY_ALLREDUCE_W4_US = 12.954


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
    """Return ``(alpha_seconds, beta_bytes_per_second, r_squared)``."""

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
    for ident, key, low in (
        ("E-A1-1", "read_gbps", 2800.0),
        ("E-A1-2", "write_gbps", 2000.0),
        ("E-A1-3", "copy_gbps", 2400.0),
    ):
        scored.append(
            Outcome(
                ident,
                band(big[key], low, HBM_PEAK_GBPS),
                f"4 GiB {key.split('_')[0]} {fmt(big[key])} GB/s in [{low}, {HBM_PEAK_GBPS}]",
                {key: big[key]},
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

    one_mib = hbm_point(lane_a, 1)["read_ms"]
    two_mib = hbm_point(lane_a, 2)["read_ms"]
    roundtrip_ms = lane_a["launch"]["roundtrip_us"] * 1e-3
    time_ratio = max(one_mib, two_mib) / min(one_mib, two_mib)
    floor_multiples = (one_mib / roundtrip_ms, two_mib / roundtrip_ms)
    scored.append(
        Outcome(
            "E-A1-5",
            time_ratio <= 1.25 and all(0.5 <= m <= 2.0 for m in floor_multiples),
            f"1 MiB and 2 MiB reads take {fmt(one_mib * 1e3, 2)} and {fmt(two_mib * 1e3, 2)} us, "
            f"a ratio of {fmt(time_ratio, 3)} against a factor two in payload, and sit at "
            f"{fmt(floor_multiples[0], 2)} and {fmt(floor_multiples[1], 2)} times the "
            f"{fmt(lane_a['launch']['roundtrip_us'], 2)} us launch roundtrip",
            {"ratio": time_ratio, "floor_multiples": list(floor_multiples)},
        )
    )

    for row in lane_a["hbm"]:
        if row["bytes"] >= (256 << 20):
            for key in ("read_gbps", "write_gbps", "copy_gbps"):
                if row[key] > HBM_PEAK_GBPS:
                    fatal.append(
                        Outcome(
                            "F2",
                            False,
                            f"HBM {key} at {row['size_mib']} MiB is {fmt(row[key])} GB/s, "
                            f"above the {HBM_PEAK_GBPS} GB/s ceiling",
                            {"size_mib": row["size_mib"], key: row[key]},
                        )
                    )

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

    decode = gemm_points(lane_a, "decode")
    m1 = next(row for row in decode if row["m"] == 1)
    scored.append(
        Outcome(
            "E-A2-1",
            m1["tflops_achieved"] < 5.0,
            f"the M=1 point achieves {fmt(m1['tflops_achieved'], 3)} TFLOP/s, limit 5",
            {"tflops": m1["tflops_achieved"]},
        )
    )

    small = [row for row in decode if row["m"] <= 64]
    times = [row["time_ms"] for row in small]
    plateau_ratio = max(times) / min(times)
    scored.append(
        Outcome(
            "E-A2-2",
            plateau_ratio <= 1.8,
            f"plateau max/min over M<=64 is {fmt(plateau_ratio, 3)}, limit 1.8",
            {"ratio": plateau_ratio},
        )
    )
    worst_multiple = max(row["time_ms"] / row["t_mem_ms"] for row in small)
    scored.append(
        Outcome(
            "E-A2-3",
            all(row["t_mem_ms"] <= row["time_ms"] <= 3.0 * row["t_mem_ms"] for row in small),
            f"every M<=64 time is between 1.0x and 3.0x its memory floor; worst is "
            f"{fmt(worst_multiple, 3)}x",
            {"worst_multiple": worst_multiple},
        )
    )

    sorted_times = sorted(times)
    mid = len(sorted_times) // 2
    plateau_med = (
        sorted_times[mid]
        if len(sorted_times) % 2 == 1
        else 0.5 * (sorted_times[mid - 1] + sorted_times[mid])
    )
    crossover = None
    for row in sorted(decode, key=lambda r: r["m"]):
        if row["m"] > 64 and row["time_ms"] > 1.5 * plateau_med:
            crossover = row["m"]
            break
    scored.append(
        Outcome(
            "E-A2-4",
            crossover is not None and 256 <= crossover <= 2048,
            f"measured crossover M={crossover} in [256, 2048], ideal {IDEAL_CROSSOVER_M}",
            {"crossover_m": crossover, "plateau_median_ms": plateau_med},
        )
    )

    large = [row for row in decode if row["m"] >= 2048]
    worst_large = min(row["tflops_achieved"] for row in large)
    scored.append(
        Outcome(
            "E-A2-5",
            worst_large >= 450.0,
            f"worst M>=2048 rate is {fmt(worst_large, 2)} TFLOP/s, floor 450",
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
    observed_mhz = square8192["clocks_after"]["sm_mhz"] or square8192["clocks_before"]["sm_mhz"]
    peak_at_clock = (
        lane_a["sm_count"] * observed_mhz * 1e6 * lane_a["bf16_flop_per_sm_cycle"] / 1e12
    )
    efficiency = square8192["tflops_achieved"] / peak_at_clock if peak_at_clock else 0.0
    scored.append(
        Outcome(
            "E-A2-7",
            band(efficiency, 0.75, 1.00),
            f"square 8192 reaches {fmt(square8192['tflops_achieved'], 2)} TFLOP/s, "
            f"{fmt(efficiency * 100, 2)} percent of the {fmt(peak_at_clock, 2)} TFLOP/s peak at "
            f"the {observed_mhz} MHz observed there, band [75, 100] percent",
            {
                "tflops": square8192["tflops_achieved"],
                "observed_sm_mhz": observed_mhz,
                "peak_at_clock_tflops": peak_at_clock,
                "efficiency": efficiency,
            },
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
    launch_ratio = launch["pipelined_period_us"] / A100_PIPELINED_LAUNCH_US
    scored.append(
        Outcome(
            "E-A3-5",
            0.4 <= launch_ratio <= 2.5,
            f"pipelined launch is {fmt(launch_ratio, 3)}x the A100's "
            f"{A100_PIPELINED_LAUNCH_US} us, band [0.4, 2.5]",
            {"ratio": launch_ratio},
        )
    )

    host = lane_a["pcie"]
    scored.append(
        Outcome(
            "E-A4-1",
            band(host["h2d_gbps"], 150.0, C2C_GBPS),
            f"host to device {fmt(host['h2d_gbps'], 2)} GB/s in [150, {C2C_GBPS}]",
            {"gbps": host["h2d_gbps"]},
        )
    )
    scored.append(
        Outcome(
            "E-A4-2",
            band(host["d2h_gbps"], 150.0, C2C_GBPS),
            f"device to host {fmt(host['d2h_gbps'], 2)} GB/s in [150, {C2C_GBPS}]",
            {"gbps": host["d2h_gbps"]},
        )
    )
    h2d_gain = host["h2d_gbps"] / A100_H2D_GBPS
    d2h_gain = host["d2h_gbps"] / A100_D2H_GBPS
    scored.append(
        Outcome(
            "E-A4-3",
            h2d_gain >= 5.0 and d2h_gain >= 5.0,
            f"the host link beats the A100 PCIe link by {fmt(h2d_gain, 2)}x inbound and "
            f"{fmt(d2h_gain, 2)}x outbound, floor 5x each",
            {"h2d_gain": h2d_gain, "d2h_gain": d2h_gain},
        )
    )
    for key in ("h2d_gbps", "d2h_gbps"):
        if host[key] > C2C_GBPS:
            fatal.append(
                Outcome("F2", False, f"host link {key} {fmt(host[key], 2)} above {C2C_GBPS}", {})
            )

    if lane_a["visible_device_count"] != 1:
        fatal.append(
            Outcome(
                "F1", False, f"lane A saw {lane_a['visible_device_count']} devices, expected 1", {}
            )
        )
    if "GH200" not in lane_a["device_name"]:
        fatal.append(Outcome("F1", False, f"unexpected device {lane_a['device_name']}", {}))

    if lane_a.get("bf16_flop_per_sm_cycle") != BF16_FLOP_PER_SM_CYCLE:
        fatal.append(
            Outcome(
                "F8",
                False,
                f"harness reported {lane_a.get('bf16_flop_per_sm_cycle')} BF16 FLOP per SM "
                f"cycle, expected {BF16_FLOP_PER_SM_CYCLE}",
                {},
            )
        )
    for name, reported, expected in (
        (
            "hbm_peak_bytes_per_second",
            lane_a.get("hbm_peak_bytes_per_second"),
            HBM_PEAK_GBPS * 1e9,
        ),
        ("bf16_peak_flops", lane_a.get("bf16_peak_flops"), BF16_PEAK_TFLOPS * 1e12),
    ):
        if reported is None or abs(reported - expected) / expected > 0.001:
            fatal.append(
                Outcome(
                    "F8",
                    False,
                    f"harness derived {name} of {reported}, freeze tabulated {expected}",
                    {},
                )
            )

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
            Outcome("F7", False, f"{missing_clocks} timed blocks lack clock observations", {})
        )

    return scored, fatal


def score_lane_b(lane_b: dict) -> tuple[list[Outcome], list[Outcome]]:
    scored: list[Outcome] = []
    fatal: list[Outcome] = []

    uni = [row for row in lane_b["p2p_unidirectional"] if "gbps" in row]
    values = [row["gbps"] for row in uni]
    scored.append(
        Outcome(
            "E-B1-1",
            len(values) == 12 and all(band(v, 120.0, NVLINK_PAIR_GBPS) for v in values),
            f"{len(values)} ordered pairs span [{fmt(min(values), 2)}, {fmt(max(values), 2)}] "
            f"GB/s, band [120, {NVLINK_PAIR_GBPS}]",
            {"min": min(values), "max": max(values), "count": len(values)},
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
            bidir.get("status") == "measured"
            and band(bidir.get("aggregate_gbps", 0.0), 200.0, NVLINK_BIDIR_PAIR_GBPS),
            f"bidirectional pair sum {fmt(bidir.get('aggregate_gbps', 0.0), 2)} GB/s in "
            f"[200, {NVLINK_BIDIR_PAIR_GBPS}]",
            {"gbps": bidir.get("aggregate_gbps")},
        )
    )
    fan = lane_b["p2p_fanout_0_to_all"]
    scored.append(
        Outcome(
            "E-B1-4",
            fan.get("status") == "measured"
            and band(fan.get("aggregate_gbps", 0.0), 330.0, NVLINK_EGRESS_GBPS),
            f"fan-out egress {fmt(fan.get('aggregate_gbps', 0.0), 2)} GB/s in "
            f"[330, {NVLINK_EGRESS_GBPS}]",
            {"gbps": fan.get("aggregate_gbps")},
        )
    )
    fan_ratio = fan.get("aggregate_gbps", 0.0) / med
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
    if bidir.get("aggregate_gbps", 0.0) > NVLINK_BIDIR_PAIR_GBPS:
        fatal.append(Outcome("F2", False, "bidirectional pair sum above its ceiling", {}))
    if fan.get("aggregate_gbps", 0.0) > NVLINK_EGRESS_GBPS:
        fatal.append(Outcome("F2", False, "fan-out egress above its ceiling", {}))

    one_gib = 1 << 30
    ar2 = collective(lane_b, "allreduce", 2, one_gib)
    ar4 = collective(lane_b, "allreduce", 4, one_gib)
    scored.append(
        Outcome(
            "E-B2-1",
            band(ar2["busbw_gbps"], 90.0, NVLINK_PAIR_GBPS),
            f"width 2 all-reduce busbw at 1 GiB {fmt(ar2['busbw_gbps'], 2)} GB/s in "
            f"[90, {NVLINK_PAIR_GBPS}]",
            {"busbw": ar2["busbw_gbps"]},
        )
    )
    scored.append(
        Outcome(
            "E-B2-2",
            band(ar4["busbw_gbps"], 200.0, NVLINK_EGRESS_GBPS),
            f"width 4 all-reduce busbw at 1 GiB {fmt(ar4['busbw_gbps'], 2)} GB/s in "
            f"[200, {NVLINK_EGRESS_GBPS}]",
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
        trio = [
            collective(lane_b, op, width, one_gib)["busbw_gbps"]
            for op in ("allreduce", "allgather", "reducescatter")
        ]
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

    tiny2 = collective(lane_b, "allreduce", 2, 8)
    tiny4 = collective(lane_b, "allreduce", 4, 8)
    scored.append(
        Outcome(
            "E-B2-5",
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
            "E-B2-6",
            ratio_ok,
            "1 GiB over 512 MiB: " + ", ".join(ratio_detail) + ", band [1.8, 2.2]",
            {},
        )
    )
    quarter = 256 << 20
    ar = collective(lane_b, "allreduce", 4, quarter)
    rs = collective(lane_b, "reducescatter", 4, quarter)
    ag = collective(lane_b, "allgather", 4, quarter)
    composition = rs["time_us"] + ag["time_us"]
    comp_ratio = ar["time_us"] / composition
    scored.append(
        Outcome(
            "E-B2-7",
            band(comp_ratio, 0.7, 1.3),
            f"all-reduce over reduce-scatter plus all-gather is {fmt(comp_ratio, 3)} in "
            f"[0.7, 1.3]",
            {"allreduce_us": ar["time_us"], "composition_us": composition},
        )
    )

    # E-B2-8, the forward prediction carried over from the A100 finding: the
    # two-anchor model is optimistic in the middle of the payload range.
    worst_by_width: dict[int, tuple[float, int]] = {}
    for width in (2, 4):
        series = collective_series(lane_b, "allreduce", width)
        alpha_s = series[0]["time_us"] * 1e-6
        largest = series[-1]
        beta = largest["bytes"] / (largest["time_us"] * 1e-6 - alpha_s)
        worst_err, worst_bytes = 0.0, 0
        for row in series:
            predicted = (alpha_s + row["bytes"] / beta) * 1e6
            err = (predicted - row["time_us"]) / row["time_us"]
            if err < worst_err:
                worst_err, worst_bytes = err, row["bytes"]
        worst_by_width[width] = (worst_err, worst_bytes)
    window = ((256 << 10), (32 << 20))
    scored.append(
        Outcome(
            "E-B2-8",
            all(worst_by_width[w][0] < -0.20 for w in (2, 4))
            and all(window[0] <= worst_by_width[w][1] <= window[1] for w in (2, 4)),
            "two-anchor model worst signed error is "
            + ", ".join(
                f"width {w} {fmt(worst_by_width[w][0] * 100, 1)} percent at "
                f"{worst_by_width[w][1]} B"
                for w in (2, 4)
            )
            + "; required more negative than -20 percent inside [256 KiB, 32 MiB]",
            {str(w): list(worst_by_width[w]) for w in (2, 4)},
        )
    )

    half_by_width: dict[int, int] = {}
    for width in (2, 4):
        series = collective_series(lane_b, "allreduce", width)
        target = series[-1]["busbw_gbps"] / 2.0
        found = 0
        for row in series:
            if row["busbw_gbps"] >= target:
                found = row["bytes"]
                break
        half_by_width[width] = found
    scored.append(
        Outcome(
            "E-B2-9",
            all((1 << 20) <= half_by_width[w] <= (32 << 20) for w in (2, 4))
            and half_by_width[4] > half_by_width[2],
            "half-bandwidth payload is "
            + ", ".join(f"width {w} at {fmt(half_by_width[w] / 2**20, 2)} MiB" for w in (2, 4))
            + "; required inside [1, 32] MiB and larger at width 4",
            {str(w): half_by_width[w] for w in (2, 4)},
        )
    )

    fits: dict[int, tuple[float, float, float]] = {}
    for width in (2, 4):
        rows = [
            row
            for row in collective_series(lane_b, "allreduce", width)
            if (1 << 20) <= row["bytes"] <= one_gib
        ]
        fits[width] = ols_fit(
            [float(row["bytes"]) for row in rows], [row["time_us"] * 1e-6 for row in rows]
        )
    alpha2_us = fits[2][0] * 1e6
    alpha4_us = fits[4][0] * 1e6
    scored.append(
        Outcome(
            "E-B2-10",
            alpha2_us > 3.0 * tiny2["time_us"] and fits[2][2] >= 0.99,
            f"the wide-window fit puts alpha at {fmt(alpha2_us, 2)} us against a measured "
            f"{fmt(tiny2['time_us'], 2)} us floor at width 2, a factor "
            f"{fmt(alpha2_us / tiny2['time_us'], 2)}, with R squared {fmt(fits[2][2], 5)}",
            {"alpha2_us": alpha2_us, "alpha4_us": alpha4_us, "r2_w2": fits[2][2]},
        )
    )

    for row in lane_b["collectives"]:
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
                    f"all-reduce width {row['width']} at {row['bytes']} B returned a wrong sum "
                    f"on {row['allreduce_mismatching_ranks']} ranks",
                    {},
                )
            )

    contention = next((row for row in lane_b["collectives"] if row.get("op") == "contention"), None)
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
    comm_total_alone = t0 * contention["comm_iters"]
    gemm_total_alone = g0 * contention["gemm_iters"]
    total_alone = contention["alone_sum_us"]
    makespan = contention["makespan_us"]
    scored.append(
        Outcome(
            "E-B3-3",
            makespan >= max(comm_total_alone, gemm_total_alone) and makespan <= 0.95 * total_alone,
            f"makespan {fmt(makespan, 1)} us sits between max alone "
            f"{fmt(max(comm_total_alone, gemm_total_alone), 1)} us and 0.95 of the serial sum "
            f"{fmt(0.95 * total_alone, 1)} us",
            {"makespan_us": makespan},
        )
    )

    surrogate_ratio = DEFAULT_NVLINK_BANDWIDTH_GBPS / fan.get("aggregate_gbps", 1.0)
    scored.append(
        Outcome(
            "E-C-1",
            band(surrogate_ratio, 1 / 1.5, 1.5),
            f"the {DEFAULT_NVLINK_BANDWIDTH_GBPS} GB/s flat surrogate is "
            f"{fmt(surrogate_ratio, 3)}x the measured {fmt(fan.get('aggregate_gbps', 0.0), 2)} "
            f"GB/s egress here, against 1.598x on the A100; band [0.667, 1.5]",
            {"surrogate_over_measured": surrogate_ratio},
        )
    )
    a100_gain = ar4["busbw_gbps"] / A100_ALLREDUCE_W4_BUSBW_GBPS
    scored.append(
        Outcome(
            "E-C-2",
            a100_gain >= 1.3,
            f"width 4 busbw is {fmt(a100_gain, 3)}x the A100's "
            f"{A100_ALLREDUCE_W4_BUSBW_GBPS} GB/s, floor 1.3x",
            {"gain": a100_gain},
        )
    )
    efficiency = ar4["busbw_gbps"] / NVLINK_EGRESS_GBPS
    scored.append(
        Outcome(
            "E-C-3",
            abs(efficiency - A100_RING_EFFICIENCY_W4) <= 0.15,
            f"ring efficiency against per-GPU egress is {fmt(efficiency * 100, 2)} percent here "
            f"against {fmt(A100_RING_EFFICIENCY_W4 * 100, 1)} percent on the A100, limit 15 "
            f"percentage points",
            {"efficiency": efficiency},
        )
    )
    latency_ratio = tiny4["time_us"] / A100_TINY_ALLREDUCE_W4_US
    scored.append(
        Outcome(
            "E-C-4",
            0.5 <= latency_ratio <= 2.0,
            f"the 8 B width-4 floor is {fmt(latency_ratio, 3)}x the A100's "
            f"{A100_TINY_ALLREDUCE_W4_US} us, band [0.5, 2.0]",
            {"ratio": latency_ratio},
        )
    )

    if lane_b["visible_device_count"] != 4:
        fatal.append(
            Outcome(
                "F1", False, f"lane B saw {lane_b['visible_device_count']} devices, expected 4", {}
            )
        )
    for dev in lane_b["devices"]:
        if "GH200" not in dev["name"]:
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
        "study": "gh200_hardware_envelope_v1",
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
