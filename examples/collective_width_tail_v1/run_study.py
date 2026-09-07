"""Frozen packet-level collective sweep and supported step attribution."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path

from simllm.backends import (
    HtsimRequestMetricReducer,
    HtsimRnicConfig,
    HtsimStepSink,
    HtsimStepSinkConfig,
    attribute_step_detail,
    run_htsim_rnic,
)
from simllm.backends.fct import (
    earliest_completion_byte_floors,
    normalized_fct,
    normalized_phase_makespan,
)
from simllm.backends.htsim_rnic import parse_completion_csv
from simllm.compute import ComputeProvider, DurationEstimate, ModelDims
from simllm.core import RequestPhase, ScheduledRequest, StepRecord
from simllm.goal import GoalTrace, to_binary
from simllm.placement import declared_manifest
from simllm.traffic import pairwise_all_to_allv, ring_allreduce

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
TOPOLOGY = REPO / "examples/m1/topologies/clos_64_400g.topo"
WIDTHS = (8, 16, 32, 64)
RATES = (400, 200)
PROFILES = ("rnic-nn", "rnic-cn")
PAYLOAD = 1_048_576
PAIR_BYTES = 65_536
COMPUTE_PS = 100_000_000
PROPAGATION_PS = 2_000_000
FREEZE = "aae9adf"
AMENDMENT = "3d818f7"
FROZEN_RING_PS = {
    (8, 400): 66_438_400, (8, 200): 104_876_800,
    (16, 400): 102_432_000, (16, 200): 144_864_000,
    (32, 400): 170_425_600, (32, 200): 216_851_200,
    (64, 400): 304_416_000, (64, 200): 356_832_000,
}


@dataclass(frozen=True)
class Cell:
    pattern: str
    width: int
    rate_gbps: int
    profile: str
    mode: str = "collective"

    @property
    def name(self):
        return f"{self.mode}-{self.pattern}-w{self.width}-{self.rate_gbps}g-{self.profile}"

    @property
    def ps_per_byte(self):
        return 8000 // self.rate_gbps

    @property
    def slot_ps(self):
        return 4160 * self.ps_per_byte

    @property
    def remote_peers(self):
        return 7 * self.width // 8


def ranks_for(width):
    if width not in WIDTHS:
        raise ValueError("width must be one of the frozen widths")
    manifest = declared_manifest(tp=64, nodes=8, gpus_per_node=8)
    return [r.global_rank for r in sorted(
        manifest.ranks, key=lambda r: (r.local_rank, r.global_rank)
    )][:width]


def expected_messages(cell):
    ranks = ranks_for(cell.width)
    if cell.mode == "step":
        raise ValueError("standalone message oracle cannot describe a step")
    if cell.pattern == "ring":
        return Counter({
            (r, ranks[(i + 1) % cell.width], 1000 + rnd, PAYLOAD // cell.width): 1
            for rnd in range(2 * (cell.width - 1)) for i, r in enumerate(ranks)
        })
    return Counter({(s, d, 1000, PAIR_BYTES): 1
                    for s in ranks for d in ranks if s // 8 != d // 8})


def build_trace(cell):
    trace = GoalTrace(64)
    ranks = ranks_for(cell.width)
    if cell.pattern == "ring":
        ring_allreduce(trace, ranks, PAYLOAD, 1000, operation_id="ring",
                       exact_frontier=True)
    else:
        pairs = {(s, d): PAIR_BYTES for s in ranks for d in ranks if s // 8 != d // 8}
        pairwise_all_to_allv(trace, ranks, pairs, 1000,
                            operation_id="all-to-all", exact_frontier=True)
    return trace


def nearest_rank(values, percentile):
    if not values or not 0 < percentile <= 1:
        raise ValueError("a nonempty sample and percentile in (0,1] are required")
    return sorted(values)[math.ceil(percentile * len(values)) - 1]


def bounds(cell):
    b = PAYLOAD // cell.width if cell.pattern == "ring" else PAIR_BYTES
    propagation = PROPAGATION_PS if cell.profile == "rnic-nn" else 4_000_000
    if cell.pattern == "ring":
        count = 2 * (cell.width - 1)
        phase_floor = count * (b * cell.ps_per_byte + propagation)
        ceiling = FROZEN_RING_PS[(cell.width, cell.rate_gbps)] + count * cell.slot_ps
    else:
        phase_floor = cell.remote_peers * b * cell.ps_per_byte + propagation
        ceiling = (cell.width * cell.remote_peers * 16 + 1) * cell.slot_ps + propagation
    return {
        "flow_payload_floor_ps": b * cell.ps_per_byte,
        "flow_propagation_floor_ps": b * cell.ps_per_byte + propagation,
        "flow_ceiling_ps": ceiling if cell.profile == "rnic-nn" else None,
        "phase_floor_ps": phase_floor,
        "phase_ceiling_ps": ceiling if cell.profile == "rnic-nn" else None,
        "propagation_ps": propagation,
    }


def check(family, ok, **evidence):
    return {"family": family, "ok": bool(ok), **evidence}


def flow_guards(cell, flows):
    actual = Counter((f.source, f.destination, f.tag, f.payload_bytes) for f in flows)
    limits = bounds(cell)
    return [
        check("message_identity", actual == expected_messages(cell)),
        check("unique_flow_id", len({f.flow_id for f in flows}) == len(flows)),
        check("timestamps", all(f.start_time_ps >= 0 and f.fct_ps > 0
                                and f.completion_time_ps - f.start_time_ps == f.fct_ps
                                for f in flows)),
        check("fabric_locality", all(f.source // 8 != f.destination // 8 for f in flows)),
        check("payload_floor", all(f.fct_ps >= limits["flow_payload_floor_ps"] for f in flows)),
        check("propagation_floor", all(f.fct_ps >= limits["flow_propagation_floor_ps"]
                                       for f in flows)),
    ]


def flow_summary(flows):
    fcts = [f.fct_ps for f in flows]
    if not fcts:
        raise ValueError("no completed flows")
    return {
        "flow_count": len(flows),
        "payload_bytes_total": sum(f.payload_bytes for f in flows),
        "fct_p50_ps": nearest_rank(fcts, .5),
        "fct_p99_ps": nearest_rank(fcts, .99),
        "fct_min_ps": min(fcts), "fct_max_ps": max(fcts),
        "phase_makespan_ps": max(f.completion_time_ps for f in flows)
        - min(f.start_time_ps for f in flows),
        "first_start_ps": min(f.start_time_ps for f in flows),
        "last_completion_ps": max(f.completion_time_ps for f in flows),
    }


def aligned_comparison(flows, baseline):
    def keyed(rows):
        result = {(f.source, f.destination, f.tag): f for f in rows}
        if len(result) != len(rows):
            raise ValueError("duplicate flow key in normalization")
        return result

    physical, ideal = keyed(flows), keyed(baseline)
    if physical.keys() != ideal.keys():
        raise ValueError("normalization requires identical message identities")
    if any(f.payload_bytes != ideal[key].payload_bytes for key, f in physical.items()):
        raise ValueError("normalization requires identical payloads")
    keys = [key for key, f in physical.items()
            if f.start_time_ps == ideal[key].start_time_ps]
    normalized = normalized_fct([physical[k] for k in keys], [ideal[k] for k in keys])
    rows = [{**asdict(n), "slowdown": n.slowdown} for n in normalized]
    return rows, len(flows) - len(rows)


def digest(path):
    if path.suffix in (".md", ".py", ".topo", ".goal"):
        return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, content):
    path.write_bytes((json.dumps(content, indent=2, sort_keys=True) + "\n").encode())


def write_csv(path, rows):
    if rows:
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        path.write_bytes(stream.getvalue().encode())


def receiver_byte_floors(flows, rate_gbps, propagation_ps):
    rows = []
    for destination in sorted({f.destination for f in flows}):
        receiver = [f for f in flows if f.destination == destination]
        rows.extend({**asdict(p), "slack_ps": p.slack_ps, "ok": p.ok} for p in
                    earliest_completion_byte_floors(
                        receiver, link_rate_bps=rate_gbps * 1_000_000_000,
                        propagation_ps=propagation_ps))
    return rows


def topology_for(out, rate):
    text = TOPOLOGY.read_text()
    if text.count("Downlink_speed_Gbps 400") != 2:
        raise ValueError("reference topology tier rates changed")
    target = out / f"clos_64_{rate}g.topo"
    target.write_bytes(text.replace("Downlink_speed_Gbps 400",
                                    f"Downlink_speed_Gbps {rate}").encode())
    return target


def run_collective(cell, directory, topology):
    goal = directory / "collective.goal"
    goal.write_bytes(build_trace(cell).render().encode())
    write_json(directory / "bounds.json", bounds(cell))
    completion = directory / "completion.csv"
    run = run_htsim_rnic(HtsimRnicConfig(
        goal_bin=to_binary(goal), profile=cell.profile,
        linkspeed_bps=cell.rate_gbps * 1_000_000_000,
        topology=topology if cell.profile == "rnic-cn" else None,
        completion_csv=completion,
    ))
    write_json(directory / "manifest.json", run.manifest)
    summary = flow_summary(run.flows)
    guards = flow_guards(cell, run.flows)
    prefixes = receiver_byte_floors(run.flows, cell.rate_gbps, bounds(cell)["propagation_ps"])
    write_csv(directory / "receiver_prefix_floors.csv", prefixes)
    guards.append(check("receiver_prefix_floor", all(p["ok"] for p in prefixes),
                        rows=len(prefixes), minimum_slack_ps=min(p["slack_ps"] for p in prefixes),
                        failed_rows=[p for p in prefixes if not p["ok"]]))
    guards.append(check("quiescence", run.quiescent))
    guards.append(check("phase_floor", summary["phase_makespan_ps"] >= bounds(cell)["phase_floor_ps"]))
    return {**summary, "goal_sha256": digest(goal), "guards": guards,
            "backend_manifest": [line.replace(str(directory.parent) + "/", "")
                                 for line in run.manifest], "bounds": bounds(cell)}


class FixedCompute(ComputeProvider):
    def estimate(self, kernel, gpu):
        return DurationEstimate(duration_ps=COMPUTE_PS, bound="measured")


def make_sink(cell, directory, topology):
    expert = cell.pattern == "all-to-all"
    ranks = ranks_for(cell.width)
    dims = ModelDims(
        num_layers=1, hidden_size=1024, intermediate_size=4096,
        num_heads=16, num_kv_heads=8, head_size=64, vocab_size=4096, dtype_bytes=2,
        num_experts=64 if expert else 0, top_k=1 if expert else 0,
        moe_intermediate_size=4096 if expert else 0,
        local_num_experts=64 // cell.width if expert else 0,
    )
    return HtsimStepSink(HtsimStepSinkConfig(
        profile=cell.profile, tp_ranks=(ranks[0],) if expert else ranks,
        ep_ranks=sorted(ranks) if expert else None, dims=dims, workdir=directory,
        linkspeed_bps=cell.rate_gbps * 1_000_000_000,
        topology=topology if cell.profile == "rnic-cn" else None,
        provider=FixedCompute(), num_goal_ranks=64,
        placement_manifest=declared_manifest(tp=64, nodes=8, gpus_per_node=8),
    ))


def run_step(cell, directory, topology):
    sink = make_sink(cell, directory, topology)
    tokens = cell.width * 32 if cell.pattern == "all-to-all" else 512
    record = StepRecord(0, 0, [ScheduledRequest(
        "request", RequestPhase.PREFILL, tokens, context_length=tokens)], num_sampled=1)
    result = sink(record)
    if result is None:
        raise ValueError("step unexpectedly bypassed network")
    locality = sink.locality_outcomes[0]
    detail = attribute_step_detail(result, locality)
    reducer = HtsimRequestMetricReducer({"request": 0})
    reducer.consume(record, result, locality)
    request, = reducer.totals()
    files = sorted(directory.glob("*.csv"))
    flows = [f for path in files for f in parse_completion_csv(path)]
    raw_summary = flow_summary(flows)
    # Backend timestamps restart per artifact; do not label their union a phase.
    raw_summary.pop("phase_makespan_ps")
    raw_summary.pop("first_start_ps")
    raw_summary.pop("last_completion_ps")
    quantum = cell.ps_per_byte
    flow_floor = (PAIR_BYTES if cell.pattern == "all-to-all" else PAYLOAD // cell.width) * quantum
    guards = [
        check("step_partition", detail.media.total_ps == result.step_latency_ps),
        check("ttft_reduction", request.ttft_ps == result.step_latency_ps
              and request.ttft_attribution == detail.attribution),
        check("fixed_compute", detail.media.kernel_ps == COMPUTE_PS),
        check("fabric_locality", all(f.source // 8 != f.destination // 8 for f in flows)),
        check("payload_floor", all(f.fct_ps >= flow_floor for f in flows)),
        check("step_bytes", sum(f.payload_bytes for f in flows) == locality.fabric_directed_bytes),
    ]
    return {
        **raw_summary, "guards": guards,
        "step_latency_ps": result.step_latency_ps, "ttft_ps": request.ttft_ps,
        "collective_share": detail.media.collective_ps / result.step_latency_ps,
        "fabric_share": detail.media.fabric_ps / result.step_latency_ps,
        "media": asdict(detail.media), "masked_service": asdict(detail.masked),
        "artifact_fabric_ps": list(locality.fabric_phase_service_ps),
        "artifact_composed_ps": list(locality.composed_phase_service_ps),
        "artifact_goal_sha256": {p.name: digest(p) for p in sorted(directory.glob("*.goal"))},
        "flow_payload_floor_ps": flow_floor,
        "critical_path_breakdown": None,
        "reducer": "HtsimRequestMetricReducer/attribute_step_detail",
        "workload": "single-engine-dispatch-combine" if cell.pattern == "all-to-all" else "two-rings",
    }



def audit_step_row(row, directory):
    cell = Cell(**{k: row[k] for k in Cell.__dataclass_fields__})
    raw = [f for path in sorted(directory.glob("*.csv")) for f in parse_completion_csv(path)]
    ranks = ranks_for(cell.width)
    if cell.pattern == "ring":
        expected = Counter({(r, ranks[(i + 1) % cell.width], PAYLOAD // cell.width):
                            4 * (cell.width - 1) for i, r in enumerate(ranks)})
    else:
        expected = Counter({pair: 1 for r in ranks if r // 8 != 0
                            for pair in ((0, r, PAIR_BYTES), (r, 0, PAIR_BYTES))})
    actual = Counter((f.source, f.destination, f.payload_bytes) for f in raw)
    limits = bounds(cell)
    network_floor = 2 * limits["phase_floor_ps"]
    row["step_floor_ps"] = COMPUTE_PS + network_floor
    row["step_ceiling_ps"] = COMPUTE_PS + 2 * limits["phase_ceiling_ps"]
    row["collective_share_floor"] = network_floor / (COMPUTE_PS + network_floor)
    row["guards"].extend([
        check("step_message_identity", actual == expected),
        check("step_timestamps", all(f.start_time_ps >= 0 and f.fct_ps > 0 and
                                     f.completion_time_ps - f.start_time_ps == f.fct_ps
                                     for f in raw)),
        check("step_physical_floor", row["step_latency_ps"] >= row["step_floor_ps"]),
        check("step_share_bounds", row["collective_share_floor"] <= row["collective_share"] <= 1),
    ])

def collect_checks(rows, out):
    for row in rows:
        directory = out / row["name"]
        if "backend_manifest" in row:
            row["backend_manifest"] = [line.replace(str(out) + "/", "")
                                       for line in row["backend_manifest"]]
        if (directory / "original-attempt.json").exists():
            row["earlier_attempt"] = "source-major input rejected before backend execution"
        if row["status"] == "error":
            error = (directory / "error.txt").read_text()
            if "fabric dropped control lifecycle" in error:
                row["failure_detail"] = error.splitlines()[0]
                row["guards"] = [check("quiescence", False,
                                        reason="fatal control lifecycle loss",
                                        survivable=(row["mode"] == "collective" and
                                                    row["pattern"] == "all-to-all" and
                                                    row["width"] == 64 and
                                                    row["profile"] == "rnic-cn"))]
        if row["status"] == "complete" and row["mode"] == "step":
            audit_step_row(row, directory)
    completed = [r for r in rows if r["status"] == "complete"]
    by_key = {(r["mode"], r["pattern"], r["width"], r["rate_gbps"], r["profile"]): r
              for r in completed}
    exact, behavioral, normalization = [], [], []
    for row in completed:
        cell = Cell(**{k: row[k] for k in Cell.__dataclass_fields__})
        if cell.profile == "rnic-nn":
            measured = row.get("phase_makespan_ps", row.get("step_latency_ps"))
            if cell.pattern == "ring":
                point = FROZEN_RING_PS[(cell.width, cell.rate_gbps)]
                expected = point if cell.mode == "collective" else COMPUTE_PS + 2 * point
                exact.append(check("ideal_ring", measured == expected, cell=cell.name,
                                   measured_ps=measured, expected_ps=expected,
                                   residual_ps=measured - expected))
            if cell.mode == "collective":
                ceiling = row["bounds"]["phase_ceiling_ps"]
                behavioral.append(check("ideal_calendar_envelope", measured <= ceiling,
                                        cell=cell.name, measured_ps=measured, ceiling_ps=ceiling))
        if cell.mode == "collective" and cell.profile == "rnic-cn":
            baseline = by_key.get((cell.mode, cell.pattern, cell.width, cell.rate_gbps, "rnic-nn"))
            if baseline is None:
                continue
            if row["goal_sha256"] != baseline["goal_sha256"]:
                row["guards"].append(check("identical_goal", False))
                continue
            physical = parse_completion_csv(out / cell.name / "completion.csv")
            ideal = parse_completion_csv(out / baseline["name"] / "completion.csv")
            ratios, excluded = aligned_comparison(physical, ideal)
            write_csv(out / cell.name / "aligned_normalization.csv", ratios)
            phase = normalized_phase_makespan(physical, ideal)
            row["guards"].append(check("phase_baseline_floor", phase.slowdown >= 1,
                                        physical_ps=phase.makespan_ps,
                                        ideal_ps=phase.baseline_makespan_ps))
            if cell.pattern == "ring":
                row["guards"].append(check("unshared_aligned_baseline_floor",
                                            all(r["slowdown"] >= 1 for r in ratios)))
            slowdowns = [r["slowdown"] for r in ratios]
            normalization.append({
                "cell": cell.name, "aligned_flows": len(ratios), "unaligned_flows": excluded,
                "slowdown_p50": nearest_rank(slowdowns, .5) if ratios else None,
                "slowdown_p99": nearest_rank(slowdowns, .99) if ratios else None,
                "slowdown_max": max(slowdowns) if ratios else None,
                "slowdown_min": min(slowdowns) if ratios else None,
                "below_1x": sum(s < 1 for s in slowdowns),
                "above_2x": sum(s > 2 for s in slowdowns),
                "above_1_2x": sum(s > 1.2 for s in slowdowns),
                "phase_ratio": phase.slowdown,
                "per_flow_interpretation": "unshared-aligned-bound" if cell.pattern == "ring"
                else "shared-receiver-diagnostic",
            })
            if ratios:
                behavioral.append(check("aligned_physical_2x", max(slowdowns) <= 2,
                                        cell=cell.name, max_slowdown=max(slowdowns)))
    for mode in ("collective", "step"):
        metric = "phase_makespan_ps" if mode == "collective" else "step_latency_ps"
        for pattern in ("ring", "all-to-all"):
            for profile in PROFILES:
                for rate in RATES:
                    selected = [by_key.get((mode, pattern, w, rate, profile)) for w in WIDTHS]
                    for left, right in pairwise(selected):
                        if left is not None and right is not None:
                            behavioral.append(check("width_growth", right[metric] > left[metric],
                                                    left=left["name"], right=right["name"]))
                for width in WIDTHS:
                    fast = by_key.get((mode, pattern, width, 400, profile))
                    slow = by_key.get((mode, pattern, width, 200, profile))
                    if fast is None or slow is None:
                        continue
                    a, b = fast[metric], slow[metric]
                    behavioral.append(check("rate_direction", b >= a,
                                            fast=fast["name"], slow=slow["name"], ratio=b / a))
                    if profile == "rnic-nn" and pattern == "ring":
                        fixed = 2 * (width - 1) * PROPAGATION_PS
                        if mode == "step":
                            fixed = COMPUTE_PS + 2 * fixed
                        exact.append(check("ring_serialization_2x", b - fixed == 2 * (a - fixed),
                                           fast=fast["name"], slow=slow["name"],
                                           residual_ps=(b - fixed) - 2 * (a - fixed)))
                    if profile == "rnic-nn" and pattern == "all-to-all":
                        behavioral.append(check("all_to_all_rate_band", a <= b <= 2 * a + 332800,
                                                fast=fast["name"], slow=slow["name"], ratio=b / a))
    return exact, behavioral, normalization


def verdict(rows, exact, behavioral, expected_count=None):
    violations = [{"cell": r["name"], **g} for r in rows
                  for g in r.get("guards", []) if not g["ok"]]
    survivable = [v for v in violations if v.get("survivable", False)]
    fatal = [v for v in violations if not v.get("survivable", False)]
    survivable_cells = {v["cell"] for v in survivable}
    failed_cells = [r["name"] for r in rows if r["status"] == "error"]
    unexpected_errors = [name for name in failed_cells if name not in survivable_cells]
    missing = 0 if expected_count is None else max(0, expected_count - len(rows))
    return {
        "status": "void" if fatal else "incomplete" if unexpected_errors or missing else
        "component-evidence-with-survivable-voids" if survivable else "component-evidence-only",
        "survivable_void_cells": sorted(survivable_cells),
        "missing_configurations": missing,
        "fatal_violations": violations, "failed_cells": failed_cells,
        "rejected_steps": [r["name"] for r in rows if r["status"] == "unsupported"],
        "exact_oracles": {"instances": len(exact), "failed": sum(not c["ok"] for c in exact)},
        "behavioral_relations": {
            "families": sorted({c["family"] for c in behavioral}),
            "instances": len(behavioral),
            "failed": None if fatal else sum(not c["ok"] for c in behavioral),
            "interpretable_for_closure": not fatal and not failed_cells and not missing,
            "completed_component_interpretable": not fatal and not unexpected_errors and not missing,
        },
        "comp9_closed": False,
    }


def plot_results(report, destination):
    """Six views of the sweep: makespans, the per-flow tail, ratios and step shares.

    Top row: ring and all-to-all phase makespans against width with the
    frozen phase floors, then the per-flow completion-time tail at 400 Gbit/s
    (p50 and p99 for both profiles, with the payload floor). Bottom row: the
    supported ideal step shares for the two-ring and expert steps, then the
    physical-over-ideal phase ratio against the 2x comparator target. Missing
    physical width-64 all-to-all points are the fatal control-loss exits.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in report["configurations"] if r["status"] == "complete"]
    colors = {"rnic-nn": "#1f77b4", "rnic-cn": "#d62728"}

    def pick(mode, pattern, profile, rate):
        return sorted([r for r in rows if r["mode"] == mode and r["pattern"] == pattern
                       and r["profile"] == profile and r["rate_gbps"] == rate],
                      key=lambda r: r["width"])

    titles = {"ring": "Ring all-reduce", "all-to-all": "Remote all-to-all"}
    from matplotlib.lines import Line2D

    with plt.rc_context({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
                         "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "lines.markersize": 4, "lines.linewidth": 1.3,
                         "legend.fontsize": 7, "legend.framealpha": 1}):
        fig, axes = plt.subplots(2, 3, figsize=(7.2, 6.0))
        fig.subplots_adjust(left=.08, right=.985, bottom=.085, top=.80,
                            wspace=.43, hspace=.48)
        for col, pattern in enumerate(("ring", "all-to-all")):
            ax = axes[0, col]
            for profile in PROFILES:
                for rate in RATES:
                    selected = pick("collective", pattern, profile, rate)
                    ax.plot([r["width"] for r in selected],
                            [r["phase_makespan_ps"] / 1e6 for r in selected],
                            marker="o", color=colors[profile],
                            linestyle="-" if rate == 400 else "--")
            for rate in RATES:
                selected = pick("collective", pattern, "rnic-nn", rate)
                ax.plot([r["width"] for r in selected],
                        [r["bounds"]["phase_floor_ps"] / 1e6 for r in selected],
                        color="gray", linewidth=.9, linestyle="-" if rate == 400 else "--")
            ax.set(title=f"({chr(97 + col)}) {titles[pattern]}\nPhase makespan",
                   ylabel="Phase makespan (µs)", yscale="log")
        axes[0, 1].text(.97, .04, "CN width 64 omitted:\nfatal control loss",
                        transform=axes[0, 1].transAxes, ha="right", fontsize=7)

        ax = axes[0, 2]
        for profile in PROFILES:
            selected = pick("collective", "all-to-all", profile, 400)
            widths = [r["width"] for r in selected]
            ax.plot(widths, [r["fct_p50_ps"] / 1e6 for r in selected], marker="o",
                    color=colors[profile], linestyle="-")
            ax.plot(widths, [r["fct_p99_ps"] / 1e6 for r in selected], marker="^",
                    color=colors[profile], linestyle=":", markerfacecolor="white")
        selected = pick("collective", "all-to-all", "rnic-nn", 400)
        floor = selected[0]["bounds"]["flow_propagation_floor_ps"] / 1e6
        ax.axhline(floor, color="gray", linewidth=.9)
        ax.axhline(floor + 2, color="gray", linewidth=.9, linestyle=":")
        ax.text(.04, .08, f"Byte + path floors:\nNN {floor:.5f}, CN {floor + 2:.5f} µs", transform=ax.transAxes,
                fontsize=7, va="bottom")
        ax.set(title="(c) All-to-all flow tail\n400 Gbit/s; CN width 64 void",
               ylabel="Flow completion time (µs)", yscale="log", ylim=(.8, 110))
        ax.legend(handles=[Line2D([], [], color="black", marker="^", linestyle=":",
                                  markerfacecolor="white", label="p99"),
                           Line2D([], [], color="black", marker="o", label="p50")],
                  loc="center right", bbox_to_anchor=(1, .50), ncols=2,
                  columnspacing=.8, handlelength=1.4)

        for col, pattern in enumerate(("ring", "all-to-all")):
            ax = axes[1, col]
            for rate in RATES:
                selected = pick("step", pattern, "rnic-nn", rate)
                ax.plot([r["width"] for r in selected],
                        [100 * r["fabric_share"] for r in selected], marker="o",
                        color=colors["rnic-nn"], linestyle="-" if rate == 400 else "--")
            ax.set(title="(d) Two-ring step\nIdeal fabric share" if pattern == "ring"
                   else "(e) Single-engine expert step\nIdeal fabric share",
                   ylabel="Fabric / step latency (%)", ylim=(0, 100))
            last = pick("step", pattern, "rnic-nn", 400)[-1]
            ax.text(.96, .07, f"Width 64, 400G:\n{100 * last['fabric_share']:.1f}%",
                    transform=ax.transAxes, ha="right", fontsize=8, color=colors["rnic-nn"])

        ax = axes[1, 2]
        for pattern, marker in (("ring", "o"), ("all-to-all", "s")):
            for rate in RATES:
                ideal = {r["width"]: r for r in pick("collective", pattern, "rnic-nn", rate)}
                physical = pick("collective", pattern, "rnic-cn", rate)
                ax.plot([r["width"] for r in physical],
                        [r["phase_makespan_ps"] / ideal[r["width"]]["phase_makespan_ps"]
                         for r in physical], marker=marker, color=colors["rnic-cn"],
                        linestyle="-" if rate == 400 else "--")
        ax.axhline(2.0, color="black", linewidth=.8)
        ax.text(.98, 2.12, "2× comparator", transform=ax.get_yaxis_transform(),
                ha="right", fontsize=7)
        ax.axhline(1.0, color="gray", linewidth=.6)
        ax.set(title="(f) Phase makespan ratio\nPhysical / ideal",
               ylabel="CN / NN (dimensionless)", ylim=(.7, 7.6))
        ax.legend(handles=[Line2D([], [], color=colors["rnic-cn"], marker="o",
                                  linestyle="none", label="Ring"),
                           Line2D([], [], color=colors["rnic-cn"], marker="s",
                                  linestyle="none", label="All-to-all")],
                  loc="upper left", ncols=2, columnspacing=.5, handletextpad=.3,
                  handlelength=.8, borderpad=.3)
        ax.text(.97, .28, "All-to-all CN width 64:\nfatal control loss",
                transform=ax.transAxes, ha="right", fontsize=7)
        for ax in axes.flat:
            ax.set(xlabel="Participating ranks (count)", xticks=WIDTHS, xlim=(5, 67))
            ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle("Collective width tail: completed components; two void cells", fontsize=11, y=.985)
        fig.text(.5, .94, "64-rank reference; physical two-tier Clos at 400 / 200 Gbit/s",
                 ha="center", fontsize=8)
        fig.legend(handles=[
            Line2D([], [], color=colors["rnic-nn"], label="NN: rnic-nn ideal"),
            Line2D([], [], color=colors["rnic-cn"], label="CN: rnic-cn physical"),
            Line2D([], [], color="gray", label="NN phase floor (a, b)"),
            Line2D([], [], color="black", linestyle="-", label="400 Gbit/s"),
            Line2D([], [], color="black", linestyle="--", label="200 Gbit/s"),
        ], loc="upper center", bbox_to_anchor=(.5, .923), ncols=3,
            frameon=False, columnspacing=1.2, handlelength=2)
        destination.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination / "collective_tail.png", dpi=240)
        fig.savefig(destination / "collective_tail.pdf", metadata={"CreationDate": None})
        plt.close(fig)


def morning_reproduction(report, original):
    """Compare every original ideal number and exact-oracle row, with zero tolerance."""
    def numeric_leaves(value, path=()):
        if isinstance(value, dict):
            for key, item in value.items():
                if key != "guards":
                    yield from numeric_leaves(item, (*path, key))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                yield from numeric_leaves(item, (*path, index))
        elif isinstance(value, (int, float)):
            yield path, value

    current = {r["name"]: r for r in report["configurations"]}
    mismatches, count, cells = [], 0, 0
    for old in original["configurations"]:
        if old["profile"] != "rnic-nn":
            continue
        cells += 1
        row = current[old["name"]]
        leaves = dict(numeric_leaves(row))
        for path, value in numeric_leaves(old):
            count += 1
            if leaves.get(path) != value:
                mismatches.append({"cell": old["name"], "path": list(path),
                                   "expected": value, "actual": leaves.get(path)})
        for key in ("status", "goal_sha256", "artifact_goal_sha256"):
            if old.get(key) != row.get(key):
                mismatches.append({"cell": old["name"], "path": [key],
                                   "expected": old.get(key), "actual": row.get(key)})
    exact_equal = report["exact_oracles"] == original["exact_oracles"]
    return {"source_commit": "80eef42", "ideal_configurations": cells,
            "ideal_numeric_fields": count, "mismatches": mismatches,
            "all_exact_oracle_rows_identical": exact_equal,
            "ok": not mismatches and exact_equal}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--publish", action="store_true", help="copy summary and figures into study")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    raw_out = args.out or (Path(os.environ["SIMLLM_DATA_ROOT"]) / "collective_width_tail_v1"
                           if os.environ.get("SIMLLM_DATA_ROOT") else None)
    if raw_out is None:
        parser.error("set SIMLLM_DATA_ROOT or pass --out on external bulk storage")
    out = raw_out.resolve()
    if out == REPO or REPO in out.parents:
        parser.error("bulk --out must be outside the repository")
    for name in ("SIMLLM_HTSIM_BUILD", "SIMLLM_HTSIM_RNIC", "SIMLLM_TXT2BIN"):
        if not os.environ.get(name) or not Path(os.environ[name]).exists():
            parser.error(f"configure {name} in local environment")
    out.mkdir(parents=True, exist_ok=True)
    provenance = {
        "expectations_commit": subprocess.check_output(
            ["git", "rev-parse", FREEZE], cwd=REPO, text=True).strip(),
        "guard_amendment_commit": subprocess.check_output(
            ["git", "rev-parse", AMENDMENT], cwd=REPO, text=True).strip(),
        "expectations_sha256": digest(HERE / "expectations.md"),
        "script_sha256": digest(Path(__file__)),
        "report_script_sha256": digest(HERE / "report.py"),
        "htsim_pin": subprocess.check_output(
            ["git", "rev-parse", "HEAD:third_party/htsim"], cwd=REPO, text=True).strip(),
        "binaries_sha256": {name: digest(Path(os.environ[name]))
                            for name in ("SIMLLM_HTSIM_RNIC", "SIMLLM_TXT2BIN")},
        "topology_sha256": digest(TOPOLOGY),
    }
    previous_path = out / "provenance.json"
    if previous_path.exists():
        previous = json.loads(previous_path.read_text())
        if not (args.resume or args.summarize_only):
            parser.error("existing results require --resume or a new --out")
        for key in ("expectations_commit", "guard_amendment_commit", "expectations_sha256", "htsim_pin",
                    "binaries_sha256", "topology_sha256"):
            compatible = {provenance[key]} if isinstance(provenance[key], str) else None
            source = {"expectations_sha256": HERE / "expectations.md",
                      "topology_sha256": TOPOLOGY}.get(key)
            if source is not None:
                compatible.add(hashlib.sha256(source.read_bytes()).hexdigest())
            if (previous[key] not in compatible if compatible is not None
                    else previous[key] != provenance[key]):
                parser.error(f"cannot reuse evidence after changing {key}; select a new --out")
    write_json(previous_path, provenance)
    topologies = {rate: topology_for(out, rate) for rate in RATES}
    rows = []
    for mode in ("collective", "step"):
        for pattern in ("ring", "all-to-all"):
            for width in WIDTHS:
                for rate in RATES:
                    for profile in PROFILES:
                        cell = Cell(pattern, width, rate, profile, mode)
                        directory = out / cell.name
                        directory.mkdir(exist_ok=True)
                        saved = directory / "cell.json"
                        if saved.exists() and (args.resume or args.summarize_only):
                            row = json.loads(saved.read_text())
                            if row["expectations_sha256"] not in {
                                provenance["expectations_sha256"],
                                hashlib.sha256((HERE / "expectations.md").read_bytes()).hexdigest()}:
                                raise ValueError("cannot resume changed expectations")
                        elif args.summarize_only:
                            continue
                        elif saved.exists():
                            parser.error("existing results require --resume or a new --out")
                        else:
                            row = {**asdict(cell), "name": cell.name,
                                   "expectations_sha256": provenance["expectations_sha256"],
                                   "script_sha256": provenance["script_sha256"]}
                            try:
                                run = run_collective if mode == "collective" else run_step
                                row.update(run(cell, directory, topologies[rate]))
                                row["status"] = "complete"
                            except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as exc:
                                (directory / "error.txt").write_bytes((str(exc) + "\n").encode())
                                unsupported = (mode == "step" and profile == "rnic-cn"
                                               and "complete BACK-38" in str(exc))
                                row.update(status="unsupported" if unsupported else "error",
                                           reason="BACK-38 state-preserving execution required"
                                           if unsupported else type(exc).__name__,
                                           step_latency_ps=None, collective_share=None,
                                           fabric_share=None, ttft_ps=None)
                            write_json(saved, row)
                        rows.append(row)
                        print(f"{cell.name}: {row['status']} "
                              f"{row.get('phase_makespan_ps', row.get('step_latency_ps'))}", flush=True)
    exact, behavioral, normalization = collect_checks(rows, out)
    report = {"provenance": provenance, "configurations": rows,
              "exact_oracles": exact, "behavioral_relations": behavioral,
              "normalization": normalization, "verdict": verdict(rows, exact, behavioral, expected_count=64)}
    original_bytes = subprocess.check_output(
        ["git", "show", "80eef42:examples/collective_width_tail_v1/results.json"], cwd=REPO)
    original = json.loads(original_bytes)
    reproduction = morning_reproduction(report, original)
    reproduction["source_sha256"] = hashlib.sha256(original_bytes).hexdigest()
    report["morning_reproduction"] = reproduction
    if not reproduction["ok"]:
        report["verdict"]["status"] = "void"
        report["verdict"]["fatal_violations"].append(check("morning_reproduction", False))
        report["verdict"]["behavioral_relations"].update(
            failed=None, interpretable_for_closure=False, completed_component_interpretable=False)
    write_json(out / "results.json", report)
    spec = importlib.util.spec_from_file_location("width_tail_report", HERE / "report.py")
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    markdown = renderer.render(report).encode()
    (out / "RESULTS.md").write_bytes(markdown)
    plot_results(report, out / "figures")
    if args.publish:
        write_json(HERE / "results.json", report)
        (HERE / "RESULTS.md").write_bytes(markdown)
        plot_results(report, HERE / "figures")
    print(json.dumps(report["verdict"], indent=2), flush=True)
    if report["verdict"]["status"] in ("void", "incomplete"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
