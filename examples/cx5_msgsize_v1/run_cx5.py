"""CX-5 message-size calibration: run the checks of expectations.md.

Four experiments on the RoCEv2 DCQCN packet path, configured from
``simllm.backends.nic_profile``: the 2 B latency anchor that calibrates the
topology, a message-size sweep at Q = 1 and Q = 16 for the measured
ConnectX-5 profile and its declared ConnectX-7 scaling, an MTU pair at 1 MiB,
and a 2 to 1 fan-in plus 1 to 2 fan-out at two switch buffer sizes and three
seeds. Emits summary.csv (one row per registered check), latency.csv, msg.csv,
mtu.csv and incast.csv for the plot script.

Usage:
    SIMLLM_HTSIM_DCQCN=... SIMLLM_TXT2BIN=... SIMLLM_DATA_ROOT=... \\
    python examples/cx5_msgsize_v1/run_cx5.py
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from pathlib import Path

from simllm._local_config import path_from_env
from simllm.backends import HtsimDcqcnConfig, RnicRunResult, run_htsim_dcqcn
from simllm.backends.nic_profile import PROFILES, NicProfile, dcqcn_flags
from simllm.goal import GoalTrace, to_binary

HERE = Path(__file__).resolve().parent
TOPOLOGIES = HERE / "topologies"
RANKS = 3
SENDER, RECEIVER, THIRD = 0, 1, 2

SIZES = [4096, 16384, 65536, 262144, 1048576, 4194304]
DEPTHS = [1, 16]
INCAST_SIZE = 1 << 20
INCAST_MESSAGES = 32
SEEDS = [1, 2, 3]
BUFFERS = [33554432, 262016]
# Post-specified diagnostic arm, reported and never scored: the registered
# 262016 B arm is refused by the comparator, whose configuration guard
# requires the egress buffer to exceed ecn_kmax_bytes, so this is the
# smallest round buffer above the registered 409600 B Kmax.
DIAGNOSTIC_BUFFERS = [524288]
DEFAULT_BUFFER = 33554432
FATAL_GUARD = "invalid DCQCN ATLAHS model config"

# Measured anchors, from the campaign records named in the profile provenance.
MEASURED_T_EFF_US = 4.48
MEASURED_LATENCY_FLOOR_US = 2.08
MEASURED_MTU_TAX = 0.056
MEASURED_DEPTH_RATIO_64K = 78.09 / 49.63
MEASURED_INCAST_GOODPUT_GBPS = 73.9
MEASURED_INCAST_WIRE_FRACTION = 0.994
MEASURED_INCAST_AMPLIFICATION = 15.6
MEASURED_Q1_CURVE_GBPS = {
    4096: 7.04,
    16384: 21.95,
    65536: 49.63,
    262144: 79.23,
    1048576: 73.95,
    4194304: 77.03,
}

G = 1_000_000_000


# Topology and flags ---------------------------------------------------------


def topology_for(profile: NicProfile) -> Path:
    return TOPOLOGIES / f"{profile.name}_3node.topo"


def topology_link_bps(topology: Path) -> int:
    matches = re.findall(
        r"^Downlink_speed_Gbps\s+(\d+)", topology.read_text(), re.MULTILINE)
    rates = {int(match) * G for match in matches}
    if len(rates) != 1:
        raise ValueError(f"{topology.name}: expected one link rate, found {sorted(rates)}")
    return rates.pop()


def study_flags(profile: NicProfile, *, buffer_bytes: int, seed: int,
                mtu_bytes: int | None = None) -> tuple[int, dict[str, str]]:
    """Profile flags plus the fabric and policy parameters the study owns."""
    flags = dcqcn_flags(profile)
    link_bps = int(flags.pop("-link_bps"))
    if mtu_bytes is not None:
        flags["-max_wire_packet_bytes"] = str(mtu_bytes)
    flags["-shared_buffer_bytes"] = str(buffer_bytes)
    flags["-egress_buffer_bytes"] = str(buffer_bytes)
    flags["-seed"] = str(seed)
    flags["-ecn_seed"] = str(seed)
    return link_bps, flags


# GOAL traces ----------------------------------------------------------------


def _pad(trace: GoalTrace, used: set[int]) -> GoalTrace:
    for rank in set(range(RANKS)) - used:
        trace.rank(rank).calc(0)
    return trace


def msg_goal(size: int, count: int) -> GoalTrace:
    trace = GoalTrace(RANKS)
    for index in range(count):
        trace.rank(SENDER).send(size, to=RECEIVER, tag=1 + index)
        trace.rank(RECEIVER).recv(size, source=SENDER, tag=1 + index)
    return _pad(trace, {SENDER, RECEIVER})


def incast_goal(size: int, count: int) -> GoalTrace:
    trace = GoalTrace(RANKS)
    for slot, source in enumerate((SENDER, THIRD)):
        for index in range(count):
            tag = 1 + slot * 1000 + index
            trace.rank(source).send(size, to=RECEIVER, tag=tag)
            trace.rank(RECEIVER).recv(size, source=source, tag=tag)
    return _pad(trace, {SENDER, RECEIVER, THIRD})


def fanout_goal(size: int, count: int) -> GoalTrace:
    trace = GoalTrace(RANKS)
    for slot, destination in enumerate((RECEIVER, THIRD)):
        for index in range(count):
            tag = 1 + slot * 1000 + index
            trace.rank(SENDER).send(size, to=destination, tag=tag)
            trace.rank(destination).recv(size, source=SENDER, tag=tag)
    return _pad(trace, {SENDER, RECEIVER, THIRD})


# Running --------------------------------------------------------------------


def manifest_counters(result: RnicRunResult) -> dict[str, int]:
    counters: dict[str, int] = {}
    for line in result.manifest:
        for key, value in re.findall(r"(\w+)=(\d+)\b", line):
            counters[key] = int(value)
    return counters


def state_trace_totals(path: Path) -> dict[int, tuple[int, int]]:
    """Final (new, retransmitted) packet counts per flow id."""
    totals: dict[int, tuple[int, int]] = {}
    if not path.is_file():
        return totals
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            new = row["new_packets_sent"]
            rtx = row["rtx_packets_sent"]
            if not new:
                continue
            totals[int(row["flow_id"])] = (int(new), int(rtx or 0))
    return totals


def run(profile: NicProfile, goal_bin: Path, out: Path, stem: str, *,
        buffer_bytes: int = DEFAULT_BUFFER, seed: int = 1,
        mtu_bytes: int | None = None,
        state_trace: bool = False) -> tuple[RnicRunResult, dict[str, int], Path]:
    topology = topology_for(profile)
    link_bps, flags = study_flags(
        profile, buffer_bytes=buffer_bytes, seed=seed, mtu_bytes=mtu_bytes)
    if topology_link_bps(topology) != link_bps:
        raise ValueError(
            f"{topology.name} rate does not match the rendered -link_bps {link_bps}")
    trace_path = out / f"{stem}.state.csv"
    if state_trace:
        flags["-state_trace_csv"] = str(trace_path)
    result = run_htsim_dcqcn(HtsimDcqcnConfig(
        goal_bin=goal_bin, topology=topology, link_bps=link_bps,
        completion_csv=out / f"{stem}.csv", extra_flags=flags))
    return result, manifest_counters(result), trace_path


# Analysis -------------------------------------------------------------------


def fit_offset_law(points: list[tuple[int, float]]) -> tuple[float, float]:
    """Fit FCT = t_eff + bits/C. Returns (t_eff seconds, C bits per second)."""
    xs = [8.0 * size for size, _ in points]
    ys = [fct for _, fct in points]
    n = float(len(xs))
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    intercept = (sy - slope * sx) / n
    return intercept, 1.0 / slope


def law_goodput_gbps(size: int, t_eff_s: float, c_bps: float) -> float:
    return 8.0 * size / (t_eff_s + 8.0 * size / c_bps) / G


def wire_packets_and_bytes(size: int, mss: int, mtu: int) -> tuple[int, int]:
    """Packets and wire bytes one message of `size` costs, with no loss."""
    full, tail = divmod(size, mss)
    packets = full + (1 if tail else 0)
    wire = full * mtu + (tail + (mtu - mss) if tail else 0)
    return packets, wire


def relative(measured: float, reference: float) -> float:
    return abs(measured - reference) / reference


def msg_reference_gbps(out: Path, profile_name: str, depth: int, size: int) -> float | None:
    """The E-MSG goodput of one cell, for cross-experiment references."""
    path = out / "msg.csv"
    if not path.is_file():
        return None
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            if (row["profile"] == profile_name and int(row["q"]) == depth
                    and int(row["size_bytes"]) == size):
                return float(row["goodput_gbps"])
    return None


# Experiments ----------------------------------------------------------------


def experiment_latency(out: Path, emit) -> None:
    profile = PROFILES["cx5_100g"]
    goal_bin = to_binary(msg_goal(2, 1).write(out / "lat2.goal"))
    result, _, _ = run(profile, goal_bin, out, "lat2")
    fct_us = result.job_completion_time_ps() * 1e-6
    with open(out / "latency.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["profile", "size_bytes", "fct_us", "measured_us"])
        writer.writerow([profile.name, 2, round(fct_us, 4), MEASURED_LATENCY_FLOOR_US])
    emit(check="M3", fct_us=round(fct_us, 4), measured_us=MEASURED_LATENCY_FLOOR_US,
         ratio=round(fct_us / MEASURED_LATENCY_FLOOR_US, 4),
         ok=relative(fct_us, MEASURED_LATENCY_FLOOR_US) <= 0.15)


def experiment_msg(out: Path, emit) -> None:
    rows: list[dict] = []
    for name in ("cx5_100g", "cx7_400g"):
        profile = PROFILES[name]
        for depth in DEPTHS:
            for size in SIZES:
                stem = f"msg-{name}-q{depth}-{size}"
                goal_bin = to_binary(msg_goal(size, depth).write(out / f"{stem}.goal"))
                result, counters, _ = run(profile, goal_bin, out, stem)
                jct_s = result.job_completion_time_ps() * 1e-12
                rows.append({
                    "profile": name, "q": depth, "size_bytes": size,
                    "jct_us": round(jct_s * 1e6, 4),
                    "goodput_gbps": round(8.0 * depth * size / jct_s / G, 4),
                    "dropped_packets": counters.get("ns_tm3_dropped_packets", 0),
                    "ecn_marked_packets": counters.get("ecn_marked_packets", 0),
                })
    with open(out / "msg.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    by = {(row["profile"], row["q"], row["size_bytes"]): row for row in rows}
    fits = {}
    for name in ("cx5_100g", "cx7_400g"):
        points = [(size, by[(name, 1, size)]["jct_us"] * 1e-6) for size in SIZES]
        fits[name] = fit_offset_law(points)

    # M1: every Q=1 point within 10 percent of its own profile's fitted law.
    worst_name, worst = None, 0.0
    for name, (t_eff, c_bps) in fits.items():
        for size in SIZES:
            deviation = relative(by[(name, 1, size)]["goodput_gbps"],
                                 law_goodput_gbps(size, t_eff, c_bps))
            if deviation > worst:
                worst_name, worst = f"{name}@{size}", deviation
    emit(check="M1", worst_cell=worst_name, worst_deviation=round(worst, 4),
         ok=worst <= 0.10)

    # M2: fitted C within 3 percent of the rendered link rate.
    for name, (t_eff, c_bps) in fits.items():
        rendered = topology_link_bps(topology_for(PROFILES[name]))
        emit(check=f"M2-{name}", fitted_c_gbps=round(c_bps / G, 3),
             rendered_gbps=rendered // G,
             point_4mib_gbps=by[(name, 1, 4194304)]["goodput_gbps"],
             ratio=round(c_bps / rendered, 4),
             ok=relative(c_bps, rendered) <= 0.03)

    # M4: the scaled profile is the same curve, four times faster.
    t5, c5 = fits["cx5_100g"]
    t7, c7 = fits["cx7_400g"]
    emit(check="M4", c_ratio=round(c7 / c5, 4), t_eff_ratio=round(t7 / t5, 4),
         cx5_t_eff_us=round(t5 * 1e6, 4), cx7_t_eff_us=round(t7 * 1e6, 4),
         ok=relative(c7 / c5, 4.0) <= 0.03 and relative(t7, t5) <= 0.15)

    # T1: fitted T_eff against the measured 4.48 us (registered expected FAIL).
    fitted_us = t5 * 1e6
    emit(check="T1", fitted_t_eff_us=round(fitted_us, 4), measured_us=MEASURED_T_EFF_US,
         ratio=round(fitted_us / MEASURED_T_EFF_US, 4),
         ok=relative(fitted_us, MEASURED_T_EFF_US) <= 0.15)

    # T2: Q amortization against the measured depth ratio (expected FAIL).
    model_ratio = (by[("cx5_100g", 16, 65536)]["goodput_gbps"]
                   / by[("cx5_100g", 1, 65536)]["goodput_gbps"])
    aggregate_equivalent = relative(by[("cx5_100g", 16, 65536)]["goodput_gbps"],
                                    by[("cx5_100g", 1, 1048576)]["goodput_gbps"])
    emit(check="T2", model_ratio=round(model_ratio, 4),
         measured_ratio=round(MEASURED_DEPTH_RATIO_64K, 4),
         q16_vs_q1_at_16x_size=round(aggregate_equivalent, 4),
         ok=relative(model_ratio, MEASURED_DEPTH_RATIO_64K) <= 0.20)

    # T3: the curve against the measured depth-1 rows, reported without a bar.
    deviations = {
        size: round(by[("cx5_100g", 1, size)]["goodput_gbps"] / MEASURED_Q1_CURVE_GBPS[size], 4)
        for size in SIZES
    }
    comparable = [deviations[size] for size in SIZES if size <= 262144]
    emit(check="T3", ratios_by_size=";".join(f"{s}:{deviations[s]}" for s in SIZES),
         median_ratio_at_or_below_256kib=round(statistics.median(comparable), 4),
         model_above_measurement=all(value > 1.0 for value in comparable),
         ok="reported")


def experiment_mtu(out: Path, emit) -> None:
    profile = PROFILES["cx5_100g"]
    rows = []
    for mtu in (4096, 1024):
        stem = f"mtu-{mtu}"
        goal_bin = to_binary(msg_goal(INCAST_SIZE, 1).write(out / f"{stem}.goal"))
        result, counters, _ = run(profile, goal_bin, out, stem, mtu_bytes=mtu)
        jct_s = result.job_completion_time_ps() * 1e-12
        rows.append({
            "mtu_bytes": mtu, "size_bytes": INCAST_SIZE,
            "goodput_gbps": round(8.0 * INCAST_SIZE / jct_s / G, 4),
            "dropped_packets": counters.get("ns_tm3_dropped_packets", 0),
        })
    with open(out / "mtu.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    big, small = rows[0]["goodput_gbps"], rows[1]["goodput_gbps"]
    tax = 1.0 - small / big
    emit(check="M5", mtu4096_gbps=big, mtu1024_gbps=small, tax_pp=round(tax * 100, 3),
         measured_tax_pp=round(MEASURED_MTU_TAX * 100, 3),
         ok=abs(tax - MEASURED_MTU_TAX) <= 0.02)


def _incast_row(profile: NicProfile, result: RnicRunResult, counters: dict[str, int],
                trace_path: Path, destinations: tuple[int, ...]) -> dict:
    flows = [flow for flow in result.flows if flow.destination in destinations]
    payload = sum(flow.payload_bytes for flow in flows)
    completions = sorted(flow.completion_time_ps for flow in flows)
    makespan_s = completions[-1] * 1e-12
    p50_s = statistics.median(completions) * 1e-12
    p95_s = completions[min(len(completions) - 1, int(0.95 * len(completions)))] * 1e-12
    mss = profile.mss_bytes
    per_message = {}
    for flow in flows:
        per_message[flow.flow_id] = wire_packets_and_bytes(
            flow.payload_bytes, mss, profile.mtu_bytes)
    totals = state_trace_totals(trace_path)
    new_packets = sum(totals.get(flow.flow_id, (per_message[flow.flow_id][0], 0))[0]
                      for flow in flows)
    rtx_packets = sum(totals.get(flow.flow_id, (0, 0))[1] for flow in flows)
    payload_wire_bytes = sum(per_message[flow.flow_id][1] for flow in flows)
    offered_bytes = payload_wire_bytes + rtx_packets * profile.mtu_bytes
    offered_packets = new_packets + rtx_packets
    dropped = counters.get("ns_tm3_dropped_packets", 0)
    # ns_tm3_dropped_packets is switch-wide and counts control packets too, so
    # it cannot be subtracted as if every drop were a data packet: every flow
    # completed, so the receiver accepted all payload-carrying packets, and the
    # only uncertainty is how many of the retransmitted duplicates also
    # arrived. Bracket it, and use the upper bound for the reported wire rate.
    delivered_upper = offered_bytes
    delivered_lower = (payload_wire_bytes
                       + max(0, rtx_packets - dropped) * profile.mtu_bytes)
    goodput_gbps = 8.0 * payload / makespan_s / G
    wire_gbps = 8.0 * delivered_upper / makespan_s / G
    wire_gbps_lower = 8.0 * delivered_lower / makespan_s / G
    per_source: dict[int, float] = {}
    for flow in flows:
        key = flow.source if len(destinations) == 1 else flow.destination
        per_source[key] = per_source.get(key, 0.0) + flow.payload_bytes
    shares = sorted(value / payload for value in per_source.values())
    return {
        "goodput_gbps": round(goodput_gbps, 4),
        "wire_gbps": round(wire_gbps, 4),
        "wire_gbps_lower": round(wire_gbps_lower, 4),
        "makespan_us": round(makespan_s * 1e6, 3),
        # Reported context, not part of any registered check: the makespan of
        # a short episode is dominated by any flow that waits a full silent
        # RTO, so the completion distribution separates the tail from the rate.
        "p50_completion_us": round(p50_s * 1e6, 3),
        "p95_completion_us": round(p95_s * 1e6, 3),
        "goodput_p95_gbps": round(8.0 * payload / p95_s / G, 4),
        "offered_packets": offered_packets,
        "rtx_packets": rtx_packets,
        "dropped_packets": dropped,
        "rtx_per_drop": round(rtx_packets / dropped, 4) if dropped else 0.0,
        "loss_rate": round(dropped / offered_packets, 6) if offered_packets else 0.0,
        "goodput_tax": round(1.0 - goodput_gbps / wire_gbps, 6) if wire_gbps else 0.0,
        "share_low_pct": round(100 * shares[0], 3),
        "share_high_pct": round(100 * shares[-1], 3),
        "silent_rtos": counters.get("silent_rtos", 0),
        "ecn_marked_packets": counters.get("ecn_marked_packets", 0),
    }


def experiment_incast(out: Path, emit) -> None:
    profile = PROFILES["cx5_100g"]
    rows: list[dict] = []
    voided: dict[int, str] = {}
    for pattern in ("incast", "fanout"):
        builder = incast_goal if pattern == "incast" else fanout_goal
        destinations = (RECEIVER,) if pattern == "incast" else (RECEIVER, THIRD)
        goal_bin = to_binary(
            builder(INCAST_SIZE, INCAST_MESSAGES).write(out / f"{pattern}.goal"))
        for buffer_bytes in BUFFERS + DIAGNOSTIC_BUFFERS:
            for seed in SEEDS:
                stem = f"{pattern}-b{buffer_bytes}-s{seed}"
                try:
                    result, counters, trace_path = run(
                        profile, goal_bin, out, stem,
                        buffer_bytes=buffer_bytes, seed=seed, state_trace=True)
                except RuntimeError as error:
                    if FATAL_GUARD not in str(error):
                        raise
                    voided[buffer_bytes] = (
                        "the comparator requires the egress buffer to exceed "
                        "ecn_kmax_bytes, so this arm cannot be configured"
                    )
                    print(f"VOID {stem}: {voided[buffer_bytes]}")
                    continue
                row = {"pattern": pattern, "buffer_bytes": buffer_bytes, "seed": seed,
                       "registered": buffer_bytes in BUFFERS}
                row.update(_incast_row(profile, result, counters, trace_path, destinations))
                rows.append(row)
                print(row)
    with open(out / "incast.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def cells(pattern: str, buffer_bytes: int | None = None,
              registered_only: bool = True) -> list[dict]:
        return [row for row in rows if row["pattern"] == pattern
                and (row["registered"] or not registered_only)
                and (buffer_bytes is None or row["buffer_bytes"] == buffer_bytes)]

    def spread(selected: list[dict], key: str) -> tuple[float, float, float]:
        values = [row[key] for row in selected]
        return (round(statistics.median(values), 4), round(min(values), 4),
                round(max(values), 4))

    # M6: fair share in every 2 to 1 cell.
    incast_cells = cells("incast")
    worst_share = max(abs(row["share_high_pct"] - 50.0) for row in incast_cells)
    emit(check="M6", worst_share_deviation_pp=round(worst_share, 3),
         ok=worst_share <= 2.0)

    # M7: fan-out is clean, against the E-MSG Q=16 aggregate at the same size.
    fanout_cells = cells("fanout")
    fan_share = max(abs(row["share_high_pct"] - 50.0) for row in fanout_cells)
    fan_dirty = max(row["dropped_packets"] + row["rtx_packets"] for row in fanout_cells)
    fan_median, fan_min, fan_max = spread(fanout_cells, "goodput_gbps")
    reference = msg_reference_gbps(out, profile.name, 16, INCAST_SIZE)
    emit(check="M7", goodput_gbps=fan_median, goodput_min=fan_min, goodput_max=fan_max,
         q16_reference_gbps=reference,
         fraction_of_reference=round(fan_median / reference, 4) if reference else None,
         worst_share_deviation_pp=round(fan_share, 3), drops_plus_rtx=fan_dirty,
         ok=(fan_share <= 2.0 and fan_dirty == 0 and reference is not None
             and fan_median >= 0.95 * reference))

    # M8: the go-back-N amplification identity at the small-buffer arm.
    def amplification_row(check: str, selected: list[dict], scored: bool) -> None:
        tax_median, tax_min, tax_max = spread(selected, "goodput_tax")
        loss_median, loss_min, loss_max = spread(selected, "loss_rate")
        literal = loss_median * INCAST_SIZE / profile.mtu_bytes
        amplification = tax_median / loss_median if loss_median else 0.0
        emit(check=check, goodput_tax=tax_median, tax_min=tax_min, tax_max=tax_max,
             loss_rate=loss_median, loss_min=loss_min, loss_max=loss_max,
             rtx_per_drop=spread(selected, "rtx_per_drop")[0],
             literal_identity_tax=round(literal, 4),
             amplification_packets=round(amplification, 3),
             measured_amplification=MEASURED_INCAST_AMPLIFICATION,
             amplification_within_2x=(
                 0.5 <= amplification / MEASURED_INCAST_AMPLIFICATION <= 2.0),
             ok=(relative(tax_median, literal) <= 0.25 if literal else False)
             if scored else "reported")

    if 262016 in voided:
        emit(check="M8", ok="void", reason=voided[262016])
    else:
        amplification_row("M8", cells("incast", 262016), scored=True)
    for buffer_bytes in DIAGNOSTIC_BUFFERS:
        amplification_row(f"M8-diagnostic-b{buffer_bytes}",
                          cells("incast", buffer_bytes, registered_only=False),
                          scored=False)

    # T4 and T5: the measured incast pair, per buffer arm.
    rendered_gbps = topology_link_bps(topology_for(profile)) / G
    for buffer_bytes in BUFFERS + DIAGNOSTIC_BUFFERS:
        registered = buffer_bytes in BUFFERS
        if registered and buffer_bytes in voided:
            emit(check=f"T4-b{buffer_bytes}", ok="void", reason=voided[buffer_bytes])
            emit(check=f"T5-b{buffer_bytes}", ok="void", reason=voided[buffer_bytes])
            continue
        selected = cells("incast", buffer_bytes, registered_only=registered)
        good_median, good_min, good_max = spread(selected, "goodput_gbps")
        wire_median, wire_min, wire_max = spread(selected, "wire_gbps")
        p95_median = spread(selected, "goodput_p95_gbps")[0]
        wire_fraction = wire_median / rendered_gbps
        suffix = "" if registered else "-diagnostic"
        emit(check=f"T4{suffix}-b{buffer_bytes}", goodput_gbps=good_median,
             goodput_min=good_min, goodput_max=good_max,
             goodput_p95_gbps=p95_median,
             measured_gbps=MEASURED_INCAST_GOODPUT_GBPS,
             ratio=round(good_median / MEASURED_INCAST_GOODPUT_GBPS, 4),
             ok=(relative(good_median, MEASURED_INCAST_GOODPUT_GBPS) <= 0.15
                 if registered else "reported"))
        emit(check=f"T5{suffix}-b{buffer_bytes}", wire_gbps=wire_median,
             wire_min=wire_min, wire_max=wire_max,
             wire_fraction_pct=round(100 * wire_fraction, 3),
             measured_pct=round(100 * MEASURED_INCAST_WIRE_FRACTION, 3),
             ok=(abs(wire_fraction - MEASURED_INCAST_WIRE_FRACTION) <= 0.02
                 if registered else "reported"))


EXPERIMENTS = {
    "latency": experiment_latency,
    "msg": experiment_msg,
    "mtu": experiment_mtu,
    "incast": experiment_incast,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    parser.add_argument("--experiments", nargs="+", choices=sorted(EXPERIMENTS),
                        default=sorted(EXPERIMENTS))
    args = parser.parse_args()
    if args.out is None:
        data_root = path_from_env("SIMLLM_DATA_ROOT")
        if data_root is None:
            parser.error("--out is required when SIMLLM_DATA_ROOT is not set")
        args.out = data_root / "cx5_msgsize_v1"
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    checks: list[dict] = []

    def emit(**row) -> None:
        checks.append(row)
        print(" ".join(f"{key}={value}" for key, value in row.items()))

    for name in ("latency", "msg", "mtu", "incast"):
        if name in args.experiments:
            EXPERIMENTS[name](out, emit)

    with open(out / "summary.csv", "w", newline="") as handle:
        fields: list[str] = []
        for row in checks:
            fields += [key for key in row if key not in fields]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(checks)
    print(f"\n{len(checks)} check rows -> {out / 'summary.csv'}")


if __name__ == "__main__":
    main()
