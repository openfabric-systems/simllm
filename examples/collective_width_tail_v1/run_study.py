"""Frozen packet-level collective sweep and supported step attribution."""

from __future__ import annotations

import argparse
import csv
import hashlib
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
from simllm.backends.fct import normalized_fct
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
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, content):
    path.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n")


def write_csv(path, rows):
    if rows:
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def topology_for(out, rate):
    text = TOPOLOGY.read_text()
    if text.count("Downlink_speed_Gbps 400") != 2:
        raise ValueError("reference topology tier rates changed")
    target = out / f"clos_64_{rate}g.topo"
    target.write_text(text.replace("Downlink_speed_Gbps 400", f"Downlink_speed_Gbps {rate}"))
    return target


def run_collective(cell, directory, topology):
    goal = build_trace(cell).write(directory / "collective.goal")
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
                                        reason="fatal control lifecycle loss")]
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
            row["guards"].append(check("aligned_baseline_floor", all(r["slowdown"] >= 1 for r in ratios)))
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
                "phase_ratio": row["phase_makespan_ps"] / baseline["phase_makespan_ps"],
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
    failed_cells = [r["name"] for r in rows if r["status"] == "error"]
    missing = 0 if expected_count is None else max(0, expected_count - len(rows))
    return {
        "status": "void" if violations else "incomplete" if failed_cells or missing else "component-evidence-only",
        "missing_configurations": missing,
        "fatal_violations": violations, "failed_cells": failed_cells,
        "rejected_steps": [r["name"] for r in rows if r["status"] == "unsupported"],
        "exact_oracles": {"instances": len(exact), "failed": sum(not c["ok"] for c in exact)},
        "behavioral_relations": {
            "families": sorted({c["family"] for c in behavioral}),
            "instances": len(behavioral),
            "failed": None if violations else sum(not c["ok"] for c in behavioral),
            "interpretable_for_closure": not violations and not failed_cells and not missing,
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

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.2), layout="constrained")
    titles = {"ring": "Ring all-reduce", "all-to-all": "Remote all-to-all"}
    for col, pattern in enumerate(("ring", "all-to-all")):
        ax = axes[0, col]
        for profile in PROFILES:
            for rate in RATES:
                selected = pick("collective", pattern, profile, rate)
                ax.plot([r["width"] for r in selected],
                        [r["phase_makespan_ps"] / 1e6 for r in selected],
                        marker="o", color=colors[profile],
                        linestyle="-" if rate == 400 else "--", label=f"{profile}, {rate}G")
        for rate in RATES:
            selected = pick("collective", pattern, "rnic-nn", rate)
            ax.plot([r["width"] for r in selected],
                    [r["bounds"]["phase_floor_ps"] / 1e6 for r in selected],
                    color="gray", linewidth=0.8, linestyle="-" if rate == 400 else "--",
                    label=f"phase floor, {rate}G")
        ax.set(title=f"{titles[pattern]}: phase makespan", ylabel="Phase makespan (us)",
               yscale="log", xlabel="Participating ranks", xticks=WIDTHS)
        ax.legend(fontsize=7, loc="upper left", ncols=2)
    axes[0, 1].text(.97, .05, "rnic-cn width 64: fatal control loss",
                    transform=axes[0, 1].transAxes, ha="right", fontsize=7)

    ax = axes[0, 2]
    for profile in PROFILES:
        selected = pick("collective", "all-to-all", profile, 400)
        widths = [r["width"] for r in selected]
        ax.plot(widths, [r["fct_p50_ps"] / 1e6 for r in selected], marker="o",
                color=colors[profile], linestyle="-", label=f"{profile} p50")
        ax.plot(widths, [r["fct_p99_ps"] / 1e6 for r in selected], marker="^",
                color=colors[profile], linestyle=":", label=f"{profile} p99")
    selected = pick("collective", "all-to-all", "rnic-nn", 400)
    ax.plot([r["width"] for r in selected],
            [r["bounds"]["flow_payload_floor_ps"] / 1e6 for r in selected],
            color="gray", linewidth=0.8, label="payload floor")
    ax.set(title="All-to-all per-flow FCT at 400G", ylabel="Flow completion time (us)",
           yscale="log", xlabel="Participating ranks", xticks=WIDTHS)
    ax.legend(fontsize=7, loc="lower right")

    for col, pattern in enumerate(("ring", "all-to-all")):
        ax = axes[1, col]
        for rate in RATES:
            selected = pick("step", pattern, "rnic-nn", rate)
            ax.plot([r["width"] for r in selected],
                    [100 * r["collective_share"] for r in selected], marker="o",
                    color=colors["rnic-nn"], linestyle="-" if rate == 400 else "--",
                    label=f"rnic-nn, {rate}G")
        ax.set(title="Two-ring step: fabric share of TTFT" if pattern == "ring"
               else "Expert step: fabric share of TTFT",
               ylabel="Collective share of step latency (%)", ylim=(0, 100),
               xlabel="Participating ranks", xticks=WIDTHS)
        ax.legend(fontsize=7, loc="lower right")

    ax = axes[1, 2]
    for pattern, marker in (("ring", "o"), ("all-to-all", "s")):
        for rate in RATES:
            ideal = {r["width"]: r for r in pick("collective", pattern, "rnic-nn", rate)}
            physical = pick("collective", pattern, "rnic-cn", rate)
            ax.plot([r["width"] for r in physical],
                    [r["phase_makespan_ps"] / ideal[r["width"]]["phase_makespan_ps"]
                     for r in physical], marker=marker, color=colors["rnic-cn"],
                    linestyle="-" if rate == 400 else "--", label=f"{pattern}, {rate}G")
    ax.axhline(2.0, color="black", linewidth=0.8, label="2x comparator target")
    ax.axhline(1.0, color="gray", linewidth=0.6)
    ax.set(title="Physical over ideal phase makespan", ylabel="rnic-cn / rnic-nn (dimensionless)",
           xlabel="Participating ranks", xticks=WIDTHS)
    ax.legend(fontsize=7, loc="upper left")
    for ax in axes.flat:
        ax.tick_params(labelsize=8)
    fig.suptitle("Collective width tail (void study with findings): 64 ranks on the two-tier "
                 "400G Clos, rnic-nn ideal versus rnic-cn physical", fontsize=10)
    destination.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination / "collective_tail.png", dpi=180)
    fig.savefig(destination / "collective_tail.pdf", metadata={"CreationDate": None})
    plt.close(fig)


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
        "expectations_sha256": digest(HERE / "expectations.md"),
        "script_sha256": digest(Path(__file__)),
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
        for key in ("expectations_commit", "expectations_sha256", "htsim_pin",
                    "binaries_sha256", "topology_sha256"):
            if previous[key] != provenance[key]:
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
                            if row["expectations_sha256"] != provenance["expectations_sha256"]:
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
                                (directory / "error.txt").write_text(str(exc) + "\n")
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
    write_json(out / "results.json", report)
    plot_results(report, out / "figures")
    if args.publish:
        write_json(HERE / "results.json", report)
        plot_results(report, HERE / "figures")
    print(json.dumps(report["verdict"], indent=2), flush=True)
    if report["verdict"]["status"] in ("void", "incomplete"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
