"""HACC fabric model: run the checks of expectations.md.

Four experiments on the RoCEv2 DCQCN packet path with the measured HACC leaf
rendered onto it by ``simllm.backends.fabric_profile``: the 2 B latency
anchor, a single-flow message-size sweep against the rendered link rate, the
per-port buffer identity read from the first go-back-N NACK over a sender
count by buffer size sweep, and a 2 to 1 fan-in that carries both the fairness
check and the comparator baseline. Emits summary.csv (one row per registered
check and guard), latency.csv, msg.csv, buffer.csv and incast.csv.

Usage:
    SIMLLM_HTSIM_DCQCN=... SIMLLM_TXT2BIN=... SIMLLM_DATA_ROOT=... \\
    python examples/hacc_fabric_v1/run_hacc_fabric.py
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import replace
from pathlib import Path

from simllm._local_config import path_from_env
from simllm.backends import HtsimDcqcnConfig, RnicRunResult, run_htsim_dcqcn
from simllm.backends.fabric_profile import (
    HACC_LEAF_4X100G,
    FabricProfile,
    render_dcqcn,
    render_topology,
)
from simllm.backends.nic_profile import CX5_100G
from simllm.goal import GoalTrace, to_binary

HERE = Path(__file__).resolve().parent
TOPOLOGIES = HERE / "topologies"
TOPOLOGY = TOPOLOGIES / "hacc_leaf_4x100g.topo"

NIC = CX5_100G
RANKS = HACC_LEAF_4X100G.host_ports
RECEIVER = 0
SENDERS = (1, 2, 3)

G = 1_000_000_000
RUN_TIMEOUT_S = 1800

#: The 2.6 MB arm is a declared sensitivity variant, not a second measurement.
HALF_BUFFER_BYTES = 2_600_000


def half_buffer_profile() -> FabricProfile:
    evidence = dict(HACC_LEAF_4X100G.evidence)
    evidence["egress_buffer_bytes"] = "declared"
    return replace(
        HACC_LEAF_4X100G,
        name="hacc_leaf_4x100g_half_buffer",
        egress_buffer_bytes=HALF_BUFFER_BYTES,
        evidence=evidence,
        provenance=(
            "Declared sensitivity arm: the measured HACC leaf with its per-port egress "
            "buffer halved, to test the buffer identity at a second point. "
            f"Base provenance: {HACC_LEAF_4X100G.provenance}"
        ),
    )


FABRICS_BY_BUFFER = {
    HACC_LEAF_4X100G.egress_buffer_bytes: HACC_LEAF_4X100G,
    HALF_BUFFER_BYTES: half_buffer_profile(),
}
BUFFERS = sorted(FABRICS_BY_BUFFER, reverse=True)
FULL_BUFFER = HACC_LEAF_4X100G.egress_buffer_bytes

MSG_SIZES = [4096, 65536, 1048576, 4194304]
MSG_ANCHOR = 4194304
BUF_SIZE = 32 << 20
BUF_SENDER_COUNTS = [2, 3]
INCAST_SIZES = [262144, 1048576]
INCAST_ANCHOR = 1048576
INCAST_MESSAGES = 32
INCAST_SENDERS = 2
SEED = 1

# Measured anchors, from the campaign records named in the fabric provenance.
MEASURED_LATENCY_FLOOR_US = 2.08
#: The cx5 study's 2 to 1 receiver goodput on this same packet path.
COMPARATOR_BASELINE_GBPS = 7.351
#: B1 registers the fan-in as collapsed rather than taxed.
COLLAPSE_CEILING_GBPS = 20.0


# Configuration --------------------------------------------------------------


def study_flags(fabric: FabricProfile, *, state_trace: Path | None = None) -> dict[str, str]:
    """Rendered flags plus the seeds and traces the study owns."""
    _, flags = render_dcqcn(NIC, fabric)
    flags["-seed"] = str(SEED)
    flags["-ecn_seed"] = str(SEED)
    if state_trace is not None:
        flags["-state_trace_csv"] = str(state_trace)
    return flags


def link_bps() -> int:
    return HACC_LEAF_4X100G.port_bps


def check_topology_file() -> None:
    """The committed topology must be exactly what the profile renders."""
    expected = render_topology(HACC_LEAF_4X100G)
    if not TOPOLOGY.is_file():
        raise FileNotFoundError(f"missing generated topology {TOPOLOGY.name}")
    if TOPOLOGY.read_text() != expected:
        raise ValueError(
            f"{TOPOLOGY.name} is not what the {HACC_LEAF_4X100G.name} profile renders; "
            "regenerate it"
        )
    rates = {
        int(match) * G
        for match in re.findall(r"^Downlink_speed_Gbps (\d+)", expected, re.MULTILINE)
    }
    if rates != {link_bps()}:
        raise ValueError(f"{TOPOLOGY.name} rate does not match the rendered link rate")


# GOAL traces ----------------------------------------------------------------


def _pad(trace: GoalTrace, used: set[int]) -> GoalTrace:
    for rank in set(range(RANKS)) - used:
        trace.rank(rank).calc(0)
    return trace


def fan_in_goal(size: int, count: int, sender_count: int) -> GoalTrace:
    """`sender_count` senders each post `count` messages of `size` to rank 0."""
    trace = GoalTrace(RANKS)
    used = {RECEIVER}
    for slot, source in enumerate(SENDERS[:sender_count]):
        used.add(source)
        for index in range(count):
            tag = 1 + slot * 1000 + index
            trace.rank(source).send(size, to=RECEIVER, tag=tag)
            trace.rank(RECEIVER).recv(size, source=source, tag=tag)
    return _pad(trace, used)


# Running --------------------------------------------------------------------


def manifest_counters(result: RnicRunResult) -> dict[str, int]:
    counters: dict[str, int] = {}
    for line in result.manifest:
        if line.startswith("[DCQCN manifest] ns_tm3_ingress_drops "):
            continue
        for key, value in re.findall(r"(\w+)=(\d+)\b", line):
            counters[key] = int(value)
    return counters


INGRESS_DROPS = re.compile(
    r"ns_tm3_ingress_drops switch=\S+ ingress=(\d+) "
    r"ns_tm3_ingress_admitted_packets=(\d+) "
    r"ns_tm3_ingress_dropped_packets=(\d+) "
    r"ns_tm3_unreacted_ingress_dropped_packets=(\d+)"
)


def ingress_drops(result: RnicRunResult,
                  ports: tuple[int, ...]) -> tuple[dict[int, int], dict[int, int]]:
    """Switch-side loss on each sender's physical ingress: over the whole run,
    and over the window that ends when the first loss notification crossed the
    switch.

    The second window is the one the sharing question is about. Before it no
    source has reacted, so every source is still offering at the rate it
    started with and admission is the only thing that can make the loss
    unequal. After it the sources are no longer equal-rate, because go-back-N
    re-offers everything after a hole and the rate cuts land at different
    times.

    `ports` are the physical ingresses of the senders. The rendered leaf puts
    host `h` on downlink port `h`, so they are the sender ranks. The receiver's
    port carries only acknowledgements and is not a competing sender, so it is
    not part of the split. A sender that offered nothing at all would be
    missing from the manifest, which is an error rather than a clean source.
    """
    dropped: dict[int, int] = {}
    unreacted: dict[int, int] = {}
    for line in result.manifest:
        match = INGRESS_DROPS.search(line)
        if match is None:
            continue
        ingress = int(match.group(1))
        dropped[ingress] = dropped.get(ingress, 0) + int(match.group(3))
        unreacted[ingress] = unreacted.get(ingress, 0) + int(match.group(4))
    if not dropped:
        return {}, {}
    missing = sorted(port for port in ports if port not in dropped)
    if missing:
        raise ValueError(f"no ns_tm3_ingress_drops line for physical ingress {missing}")
    return ({port: dropped[port] for port in ports},
            {port: unreacted[port] for port in ports})


def split_deviation_pct(counts: dict[int, int]) -> float | None:
    """Largest deviation from an even split, as a percentage of the mean."""
    values = list(counts.values())
    if len(values) < 2 or sum(values) == 0:
        return None
    mean = sum(values) / len(values)
    return round(100.0 * max(abs(value - mean) for value in values) / mean, 3)


def render_counts(counts: dict[int, int]) -> str:
    return ";".join(f"{ingress}:{counts[ingress]}" for ingress in sorted(counts))


def state_trace_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def first_event_ps(rows: list[dict[str, str]], event: str) -> int | None:
    times = [int(row["time_ps"]) for row in rows if row["event"] == event]
    return min(times) if times else None


def run(fabric: FabricProfile, goal_bin: Path, out: Path, stem: str, *,
        state_trace: bool = False) -> tuple[RnicRunResult, dict[str, int], list[dict[str, str]]]:
    trace_path = out / f"{stem}.state.csv"
    flags = study_flags(fabric, state_trace=trace_path if state_trace else None)
    result = run_htsim_dcqcn(
        HtsimDcqcnConfig(
            goal_bin=goal_bin,
            topology=TOPOLOGY,
            link_bps=int(flags.pop("-link_bps")),
            completion_csv=out / f"{stem}.csv",
            extra_flags=flags,
        ),
        timeout_s=RUN_TIMEOUT_S,
    )
    return result, manifest_counters(result), state_trace_rows(trace_path) if state_trace else []


# Guards ---------------------------------------------------------------------


class Guards:
    """Per-cell guard observations, emitted once at the end of the run."""

    def __init__(self) -> None:
        self.marked: dict[str, int] = {}
        self.shared_drops: dict[str, int] = {}
        self.pause_frames: dict[str, int] = {}
        self.conservation: dict[str, tuple[int, int, int, int]] = {}

    def observe(self, stem: str, result: RnicRunResult, counters: dict[str, int],
                offered_bytes: int, offered_flows: int) -> None:
        self.marked[stem] = counters.get("ecn_marked_packets", -1)
        self.shared_drops[stem] = counters.get("ns_tm3_shared_pool_dropped_packets", -1)
        self.pause_frames[stem] = counters.get("dcqcn_pfc_pause_frames", -1)
        delivered = sum(flow.payload_bytes for flow in result.flows)
        self.conservation[stem] = (
            offered_bytes, delivered, offered_flows, counters.get("completed_flows", -1)
        )

    def emit(self, emit) -> None:
        bad_conservation = {
            stem: values
            for stem, values in self.conservation.items()
            if values[0] != values[1] or values[2] != values[3]
        }
        emit(check="G2", cells=len(self.conservation),
             offending_cells=";".join(sorted(bad_conservation)) or "none",
             ok=not bad_conservation)
        worst_marked = max(self.marked.values(), default=-1)
        emit(check="G3", max_ecn_marked_packets=worst_marked,
             cells=len(self.marked), ok=worst_marked == 0)
        worst_shared = max(self.shared_drops.values(), default=-1)
        emit(check="G4", max_shared_pool_dropped_packets=worst_shared,
             cells=len(self.shared_drops), ok=worst_shared == 0)
        emit(check="G5-pfc-still", max_pause_frames=max(self.pause_frames.values(), default=-1),
             ok="reported")


# Experiments ----------------------------------------------------------------


def experiment_latency(out: Path, emit, guards: Guards) -> None:
    goal_bin = to_binary(fan_in_goal(2, 1, 1).write(out / "lat2.goal"))
    result, counters, _ = run(HACC_LEAF_4X100G, goal_bin, out, "lat2")
    guards.observe("lat2", result, counters, 2, 1)
    fct_us = result.job_completion_time_ps() * 1e-6
    with open(out / "latency.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["fabric", "size_bytes", "fct_us", "measured_us"])
        writer.writerow([HACC_LEAF_4X100G.name, 2, round(fct_us, 4), MEASURED_LATENCY_FLOOR_US])
    emit(check="M1", fct_us=round(fct_us, 4), measured_us=MEASURED_LATENCY_FLOOR_US,
         ratio=round(fct_us / MEASURED_LATENCY_FLOOR_US, 4),
         ok=abs(fct_us - MEASURED_LATENCY_FLOOR_US) / MEASURED_LATENCY_FLOOR_US <= 0.15)


def experiment_msg(out: Path, emit, guards: Guards) -> None:
    rows: list[dict] = []
    for buffer_bytes in BUFFERS:
        fabric = FABRICS_BY_BUFFER[buffer_bytes]
        for size in MSG_SIZES:
            stem = f"msg-b{buffer_bytes}-{size}"
            goal_bin = to_binary(fan_in_goal(size, 1, 1).write(out / f"{stem}.goal"))
            result, counters, _ = run(fabric, goal_bin, out, stem)
            guards.observe(stem, result, counters, size, 1)
            jct_ps = result.job_completion_time_ps()
            rows.append({
                "buffer_bytes": buffer_bytes,
                "size_bytes": size,
                "jct_ps": jct_ps,
                "goodput_gbps": round(8.0 * size / (jct_ps * 1e-12) / G, 4),
                "dropped_packets": counters.get("ns_tm3_dropped_packets", 0),
                "ecn_marked_packets": counters.get("ecn_marked_packets", 0),
            })
    with open(out / "msg.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    by = {(row["buffer_bytes"], row["size_bytes"]): row for row in rows}
    rendered_gbps = link_bps() / G
    anchor = by[(FULL_BUFFER, MSG_ANCHOR)]["goodput_gbps"]
    emit(check="M2", goodput_gbps=anchor, rendered_gbps=rendered_gbps,
         shortfall_pct=round(100 * (1 - anchor / rendered_gbps), 3),
         ok=abs(anchor - rendered_gbps) / rendered_gbps <= 0.03)

    # M2-buffer: an idle port never queues, so the two arms must be identical.
    differences = {
        size: by[(BUFFERS[0], size)]["jct_ps"] - by[(BUFFERS[1], size)]["jct_ps"]
        for size in MSG_SIZES
    }
    emit(check="M2-buffer", max_abs_jct_difference_ps=max(abs(v) for v in differences.values()),
         curve_gbps=";".join(f"{s}:{by[(FULL_BUFFER, s)]['goodput_gbps']}" for s in MSG_SIZES),
         ok="reported")


def _buffer_row(fabric: FabricProfile, sender_count: int, result: RnicRunResult,
                counters: dict[str, int], rows: list[dict[str, str]]) -> dict:
    rendered = link_bps()
    excess = (sender_count - 1) * rendered
    nack_ps = first_event_ps(rows, "gbn-nack")
    rto_ps = first_event_ps(rows, "silent-rto")
    buffer_estimate = (
        (nack_ps * 1e-12) * excess * rendered / (rendered + excess) / 8.0
        if nack_ps is not None else None
    )
    offered_packets = sum(
        int(row["new_packets_sent"] or 0) + int(row["rtx_packets_sent"] or 0)
        for row in rows if row["event"] == "completion"
    )
    dropped = counters.get("ns_tm3_dropped_packets", 0)
    rates = [
        int(row["configured_rate_bps"])
        for row in rows
        if row["configured_rate_bps"] and nack_ps is not None and int(row["time_ps"]) >= nack_ps
    ]
    makespan_s = result.job_completion_time_ps() * 1e-12
    payload = sum(flow.payload_bytes for flow in result.flows)
    # Reported diagnostic, not a registered check: which sender paid the loss.
    # The measurement lost within 0.5 percent of the same count on every
    # concurrent stream, so an uneven split here is a mechanism difference.
    rtx_by_source = {
        int(row["source"]): int(row["rtx_packets_sent"] or 0)
        for row in rows if row["event"] == "completion"
    }
    total_rtx = sum(rtx_by_source.values())
    # Reported diagnostic beside it: the switch's own view of the same
    # question. Retransmissions are an amplified proxy, because go-back-N
    # re-sends everything after a hole.
    drops_by_ingress, unreacted_by_ingress = ingress_drops(result, SENDERS[:sender_count])
    return {
        "senders": sender_count,
        "buffer_bytes": fabric.egress_buffer_bytes,
        "excess_gbps": excess / G,
        "first_gbn_nack_us": round(nack_ps * 1e-6, 3) if nack_ps is not None else None,
        "first_silent_rto_us": round(rto_ps * 1e-6, 3) if rto_ps is not None else None,
        "buffer_estimate_bytes": round(buffer_estimate) if buffer_estimate is not None else None,
        "buffer_ratio": (
            round(buffer_estimate / fabric.egress_buffer_bytes, 4)
            if buffer_estimate is not None else None
        ),
        "makespan_us": round(makespan_s * 1e6, 3),
        "aggregate_gbps": round(8.0 * payload / makespan_s / G, 4),
        "offered_packets": offered_packets,
        "dropped_packets": dropped,
        "egress_domain_dropped": counters.get("ns_tm3_egress_domain_dropped_packets", 0),
        "loss_fraction": round(dropped / offered_packets, 6) if offered_packets else 0.0,
        "min_configured_rate_gbps": round(min(rates) / G, 4) if rates else None,
        "loss_rate_cuts": counters.get("loss_rate_cuts", 0),
        "silent_rtos": counters.get("silent_rtos", 0),
        "rtx_by_source": ";".join(
            f"{source}:{rtx_by_source[source]}" for source in sorted(rtx_by_source)),
        "worst_source_rtx_share_pct": (
            round(100.0 * max(rtx_by_source.values()) / total_rtx, 3) if total_rtx else 0.0),
        "clean_sources": sum(1 for value in rtx_by_source.values() if value == 0),
        "drops_by_ingress": render_counts(drops_by_ingress),
        "unreacted_drops_by_ingress": render_counts(unreacted_by_ingress),
        "drop_split_deviation_pct": split_deviation_pct(drops_by_ingress),
        "unreacted_drop_split_deviation_pct": split_deviation_pct(unreacted_by_ingress),
        "clean_ingresses": sum(1 for value in drops_by_ingress.values() if value == 0),
    }


def experiment_buf(out: Path, emit, guards: Guards) -> None:
    rows: list[dict] = []
    repeat: dict | None = None
    for sender_count in BUF_SENDER_COUNTS:
        goal = fan_in_goal(BUF_SIZE, 1, sender_count)
        goal_bin = to_binary(goal.write(out / f"buf-n{sender_count}.goal"))
        for buffer_bytes in BUFFERS:
            fabric = FABRICS_BY_BUFFER[buffer_bytes]
            stem = f"buf-n{sender_count}-b{buffer_bytes}"
            result, counters, trace = run(fabric, goal_bin, out, stem, state_trace=True)
            guards.observe(stem, result, counters, BUF_SIZE * sender_count, sender_count)
            row = _buffer_row(fabric, sender_count, result, counters, trace)
            rows.append(row)
            print(row)
            if sender_count == 2 and buffer_bytes == FULL_BUFFER:
                rerun_stem = f"{stem}-repeat"
                rerun, rerun_counters, rerun_trace = run(
                    fabric, goal_bin, out, rerun_stem, state_trace=True)
                guards.observe(
                    rerun_stem, rerun, rerun_counters, BUF_SIZE * sender_count, sender_count)
                repeat = _buffer_row(fabric, sender_count, rerun, rerun_counters, rerun_trace)
    with open(out / "buffer.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # G1: the repeated cell must be identical in every scored quantity.
    baseline = next(
        row for row in rows if row["senders"] == 2 and row["buffer_bytes"] == FULL_BUFFER)
    keys = ("makespan_us", "dropped_packets", "offered_packets", "first_gbn_nack_us")
    mismatch = (
        "no repeat run" if repeat is None
        else ";".join(k for k in keys if baseline[k] != repeat[k]) or "none"
    )
    emit(check="G1", mismatched_fields=mismatch, ok=mismatch == "none")

    # M3a: the buffer identity, from the first go-back-N NACK.
    voided = [row for row in rows if row["buffer_estimate_bytes"] is None]
    scored = [row for row in rows if row["buffer_estimate_bytes"] is not None]
    if voided:
        emit(check="M3a", ok="void", cells=len(rows), void_cells=len(voided),
             reason="no gbn-nack row, so the first-drop estimator has no input")
    else:
        worst = max(abs(row["buffer_ratio"] - 1.0) for row in scored)
        emit(check="M3a", cells=len(scored),
             ratios=";".join(f"n{r['senders']}b{r['buffer_bytes']}:{r['buffer_ratio']}"
                             for r in scored),
             nack_us=";".join(f"n{r['senders']}b{r['buffer_bytes']}:{r['first_gbn_nack_us']}"
                              for r in scored),
             worst_deviation=round(worst, 4), ok=worst <= 0.20)

    # Reported diagnostic beside M3a: how the loss split across the senders.
    emit(check="M3a-loss-split",
         rtx_by_source=";".join(f"n{r['senders']}b{r['buffer_bytes']}:{r['rtx_by_source']}"
                                for r in rows),
         worst_source_share_pct=";".join(
             f"n{r['senders']}b{r['buffer_bytes']}:{r['worst_source_rtx_share_pct']}"
             for r in rows),
         clean_sources=";".join(f"n{r['senders']}b{r['buffer_bytes']}:{r['clean_sources']}"
                                for r in rows),
         ok="reported")

    # Reported diagnostic beside it: the same question asked of the switch.
    emit(check="M3a-drop-split",
         unreacted_drops_by_ingress=";".join(
             f"n{r['senders']}b{r['buffer_bytes']}:{r['unreacted_drops_by_ingress']}"
             for r in rows),
         worst_unreacted_deviation_pct=max(
             (r["unreacted_drop_split_deviation_pct"] for r in rows
              if r["unreacted_drop_split_deviation_pct"] is not None), default=None),
         drops_by_ingress=";".join(
             f"n{r['senders']}b{r['buffer_bytes']}:{r['drops_by_ingress']}" for r in rows),
         worst_run_deviation_pct=max(
             (r["drop_split_deviation_pct"] for r in rows
              if r["drop_split_deviation_pct"] is not None), default=None),
         ok="reported")

    # M3b: void by construction, with the numbers reported beside it.
    emit(check="M3b", ok="void",
         reason=("the measured identity needs an open-loop paced sender; this path has only "
                 "a DCQCN closed-loop source with loss_rate_cut on, so the excess collapses"),
         loss_fractions=";".join(f"n{r['senders']}b{r['buffer_bytes']}:{r['loss_fraction']}"
                                 for r in rows),
         min_rate_gbps=";".join(f"n{r['senders']}b{r['buffer_bytes']}:"
                                f"{r['min_configured_rate_gbps']}" for r in rows),
         loss_rate_cuts=";".join(f"n{r['senders']}b{r['buffer_bytes']}:{r['loss_rate_cuts']}"
                                 for r in rows))


def _incast_row(size: int, buffer_bytes: int, result: RnicRunResult,
                counters: dict[str, int], rows: list[dict[str, str]]) -> dict:
    payload_by_source: dict[int, int] = {}
    span_by_source: dict[int, int] = {}
    for flow in result.flows:
        payload_by_source[flow.source] = (
            payload_by_source.get(flow.source, 0) + flow.payload_bytes)
        span_by_source[flow.source] = max(
            span_by_source.get(flow.source, 0), flow.completion_time_ps)
    throughput = {
        source: payload_by_source[source] / (span_by_source[source] * 1e-12)
        for source in payload_by_source
    }
    total = sum(throughput.values())
    shares = sorted(100.0 * value / total for value in throughput.values())
    payload = sum(payload_by_source.values())
    makespan_s = result.job_completion_time_ps() * 1e-12
    dropped = counters.get("ns_tm3_dropped_packets", 0)
    drops_by_ingress, unreacted_by_ingress = ingress_drops(result, SENDERS[:INCAST_SENDERS])
    return {
        "size_bytes": size,
        "buffer_bytes": buffer_bytes,
        "goodput_gbps": round(8.0 * payload / makespan_s / G, 4),
        "makespan_us": round(makespan_s * 1e6, 3),
        "share_low_pct": round(shares[0], 3),
        "share_high_pct": round(shares[-1], 3),
        "dropped_packets": dropped,
        "egress_domain_dropped": counters.get("ns_tm3_egress_domain_dropped_packets", 0),
        "silent_rtos": counters.get("silent_rtos", 0),
        "gbn_nacks": sum(1 for row in rows if row["event"] == "gbn-nack"),
        "loss_rate_cuts": counters.get("loss_rate_cuts", 0),
        "ecn_marked_packets": counters.get("ecn_marked_packets", 0),
        "drops_by_ingress": render_counts(drops_by_ingress),
        "unreacted_drops_by_ingress": render_counts(unreacted_by_ingress),
        "unreacted_drop_split_deviation_pct": split_deviation_pct(unreacted_by_ingress),
    }


def experiment_incast(out: Path, emit, guards: Guards) -> None:
    rows: list[dict] = []
    for size in INCAST_SIZES:
        goal = fan_in_goal(size, INCAST_MESSAGES, INCAST_SENDERS)
        goal_bin = to_binary(goal.write(out / f"incast-{size}.goal"))
        offered = size * INCAST_MESSAGES * INCAST_SENDERS
        flows = INCAST_MESSAGES * INCAST_SENDERS
        for buffer_bytes in BUFFERS:
            fabric = FABRICS_BY_BUFFER[buffer_bytes]
            stem = f"incast-{size}-b{buffer_bytes}"
            result, counters, trace = run(fabric, goal_bin, out, stem, state_trace=True)
            guards.observe(stem, result, counters, offered, flows)
            row = _incast_row(size, buffer_bytes, result, counters, trace)
            rows.append(row)
            print(row)
    with open(out / "incast.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    worst_share = max(abs(row["share_high_pct"] - 50.0) for row in rows)
    emit(check="M4", worst_share_deviation_pp=round(worst_share, 3),
         shares=";".join(f"{r['size_bytes']}b{r['buffer_bytes']}:"
                         f"{r['share_low_pct']}/{r['share_high_pct']}" for r in rows),
         ok=worst_share <= 2.0)

    anchor = next(
        row for row in rows
        if row["size_bytes"] == INCAST_ANCHOR and row["buffer_bytes"] == FULL_BUFFER)
    emit(check="B1", goodput_gbps=anchor["goodput_gbps"],
         ceiling_gbps=COLLAPSE_CEILING_GBPS,
         comparator_baseline_gbps=COMPARATOR_BASELINE_GBPS,
         makespan_us=anchor["makespan_us"], silent_rtos=anchor["silent_rtos"],
         gbn_nacks=anchor["gbn_nacks"], dropped_packets=anchor["dropped_packets"],
         ok=anchor["goodput_gbps"] <= COLLAPSE_CEILING_GBPS)


EXPERIMENTS = {
    "latency": experiment_latency,
    "msg": experiment_msg,
    "buf": experiment_buf,
    "incast": experiment_incast,
}
ORDER = ("latency", "msg", "buf", "incast")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    parser.add_argument("--experiments", nargs="+", choices=sorted(EXPERIMENTS),
                        default=sorted(EXPERIMENTS))
    parser.add_argument("--write-topology", action="store_true",
                        help="regenerate the committed topology from the fabric profile")
    args = parser.parse_args()
    if args.write_topology:
        TOPOLOGIES.mkdir(parents=True, exist_ok=True)
        TOPOLOGY.write_text(render_topology(HACC_LEAF_4X100G))
        print(f"wrote {TOPOLOGY}")
        return
    if args.out is None:
        data_root = path_from_env("SIMLLM_DATA_ROOT")
        if data_root is None:
            parser.error("--out is required when SIMLLM_DATA_ROOT is not set")
        args.out = data_root / "hacc_fabric_v1"
    check_topology_file()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    checks: list[dict] = []
    guards = Guards()

    def emit(**row) -> None:
        checks.append(row)
        print(" ".join(f"{key}={value}" for key, value in row.items()))

    for name in ORDER:
        if name in args.experiments:
            EXPERIMENTS[name](out, emit, guards)
    guards.emit(emit)

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
