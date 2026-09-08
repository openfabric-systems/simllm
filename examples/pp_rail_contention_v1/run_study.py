"""Run the frozen contention grid, retaining one concurrent GOAL per cell.

Set SIMLLM_DATA_ROOT and the SIMLLM_HTSIM_RNIC/SIMLLM_TXT2BIN executables.
The accepted study owns the unchanged workload and completion interpretation.
This driver adds capacity bounds, explicit topology variants and live metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
from collections import Counter
from dataclasses import asdict
from enum import Enum
from itertools import product
from pathlib import Path

from simllm.backends import HtsimRnicConfig, HtsimStepSink, HtsimStepSinkConfig, run_htsim_rnic
from simllm.backends.htsim_rnic import find_htsim_rnic
from simllm.backends.rail_topology import project_declared_clos
from simllm.core import (
    AdditiveVisitTotals,
    CompletionEvent,
    ComputeWork,
    EventPhase,
    LatencyAttribution,
    RequestMetric,
    StepResult,
    execution_graph_to_json,
)
from simllm.goal import find_txt2bin, to_binary
from simllm.placement import declared_pipeline_placement, declared_rail_fabric
from simllm.traffic import render_serial_execution_graph_goal

HERE = Path(__file__).resolve().parent
_reference_spec = importlib.util.spec_from_file_location(
    "pp_rail_reference", HERE.parent / "pp_rail_topology_v1" / "run_study.py")
reference = importlib.util.module_from_spec(_reference_spec)
_reference_spec.loader.exec_module(reference)
EXPECTATIONS_COMMIT = "94b7f1fb126ee5c295b39b52d7d7f322b737fb05"
SPINES = (8, 2)
VARIANTS = reference.VARIANTS
WIDTHS = reference.WIDTHS
EP_WIDTHS = reference.EP_WIDTHS
PROFILES = reference.PROFILES
B = reference.B
M = reference.EP_BYTES
RATE = reference.RATE
COMPUTE_PS = reference.COMPUTE_PS
PACKET_PS = 83_200
REFERENCE_DIGESTS = {
    "run_study.py": "daf10376c5a4a301115c6d118c4d2b94410b8a948d2b7182a8623faec7182271",
    "results.json": "cadd567bfb6fa46284490bdc6209615a3e4bf629a54b0edc1072e7a53e3ac317",
    "expectations.md": "9aab8fa1bdb29c4e416e71088ebb267c475b5f9c7e2cbcbef6703af9a84e227b",
}
DEFAULT_MANIFEST_DIGESTS = {
    "rail": "595920bd5de6f4481a687e9d53b37942ef47b129b3cd062862af89223f39d5b4",
    "node-local": "083fd43dd2818d4b747412aa1dc2e1c65eb708cfe203ef7aa78df3c8c282b69b",
}
DEFAULT_TOPOLOGY = (
    "Nodes 64\nTiers 2\nPodsize 64\n\n"
    "Tier 0\nDownlink_speed_Gbps 400\nRadix_Down 8\nRadix_Up 8\n"
    "Downlink_Latency_ns 1000\nSwitch_Latency_ns 0\n\n"
    "Tier 1\nDownlink_speed_Gbps 400\nRadix_Down 8\n"
    "Downlink_Latency_ns 1000\nSwitch_Latency_ns 0\n"
)


def write_json(path: Path, value) -> None:
    """Use LF bytes on every platform, including tracked artifacts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    def enum_value(value):
        if isinstance(value, Enum):
            return value.value
        raise TypeError(f"unsupported JSON value: {type(value).__name__}")
    path.write_bytes((json.dumps(value, indent=2, sort_keys=True, default=enum_value) + "\n").encode())


def text_digest_matches(raw: bytes, expected: str) -> bool:
    return expected in {
        hashlib.sha256(raw).hexdigest(),
        hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest(),
    }


def reference_inputs() -> dict:
    for name, expected in REFERENCE_DIGESTS.items():
        if not text_digest_matches((reference.HERE / name).read_bytes(), expected):
            raise ValueError(f"accepted reference input changed: {name}")
    return json.loads((reference.HERE / "results.json").read_bytes())


def identity_guards() -> dict:
    """Golden bytes were recorded after the freeze, before production edits."""
    checked = {}
    for variant in VARIANTS:
        placement = declared_pipeline_placement(8)
        fabric = declared_rail_fabric(placement, variant=variant)
        raw = (json.dumps(fabric.to_dict(), indent=2) + "\n").encode()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != DEFAULT_MANIFEST_DIGESTS[variant]:
            raise ValueError(f"default {variant} manifest changed")
        projection = project_declared_clos(fabric)
        expected_map = tuple(rank % 8 * 8 + rank // 8 if variant == "rail" else rank
                             for rank in range(64))
        if projection.topology_text != DEFAULT_TOPOLOGY or projection.endpoint_by_rank != expected_map:
            raise ValueError(f"default {variant} topology or permutation changed")
        checked[variant] = {"manifest_sha256": digest, "topology_and_permutation": "exact"}
    return checked


def leaf_for(rank: int, variant: str) -> int:
    return rank % 8 if variant == "rail" else rank // 8


def bounds(variant: str, spines: int, width: int, ep_width: int) -> dict:
    """Directional cuts are counted from the actual semantic EP pair inventory.

    The whole-phase cut is not a tagged-hop queue. Zero is the only guaranteed
    ahead-of-PP byte count available without packet queue event evidence.
    """
    if (variant, spines, width, ep_width) not in product(VARIANTS, SPINES, WIDTHS, EP_WIDTHS):
        raise ValueError("bounds require a frozen grid cell")
    outgoing, incoming = Counter(), Counter()
    endpoint_out, endpoint_in = Counter(), Counter()
    for source, destination, size in reference.ep_pairs(ep_width):
        endpoint_out[source] += size
        endpoint_in[destination] += size
        source_leaf, destination_leaf = leaf_for(source, variant), leaf_for(destination, variant)
        if source_leaf != destination_leaf:
            outgoing[source_leaf] += size
            incoming[destination_leaf] += size
    endpoint_bytes = max((*endpoint_out.values(), *endpoint_in.values()), default=0)
    cut_bytes = max((*outgoing.values(), *incoming.values()), default=0)
    endpoint_serialization = endpoint_bytes * 20
    cut_serialization = cut_bytes * 20 // spines
    ep_floor = max(endpoint_serialization + 2_000_000 if endpoint_bytes else 0,
                   cut_serialization + 4_000_000 if cut_bytes else 0)
    hop_floor = B * 20 + (2 if variant == "rail" else 4) * 1_000_000
    hop_cuts = []
    for stage in range(width - 1):
        source_leaf = leaf_for(8 * stage, variant)
        destination_leaf = leaf_for(8 * (stage + 1), variant)
        source_bytes = outgoing[source_leaf] if source_leaf != destination_leaf else 0
        destination_bytes = incoming[destination_leaf] if source_leaf != destination_leaf else 0
        hop_cuts.append({
            "source": 8 * stage, "destination": 8 * (stage + 1),
            "source_leaf_ep_uplink_bytes": source_bytes,
            "destination_leaf_ep_downlink_bytes": destination_bytes,
            "potential_shared_cut_drain_floor_ps": max(source_bytes, destination_bytes) * 20 // spines,
            "guaranteed_ep_bytes_ahead": 0, "additional_queue_floor_ps": 0,
            "hop_data_arrival_floor_ps": hop_floor,
        })
    step_floor = width * COMPUTE_PS + (width - 1) * hop_floor
    return {
        **reference.bounds(width, ep_width, variant),
        "spine_count": spines,
        "uplink_capacity_bps": spines * RATE,
        "shared_throughput_ceiling_bytes_per_second": spines * RATE // 8,
        "ep_endpoint_bytes": endpoint_bytes,
        "ep_busiest_leaf_uplink_bytes": cut_bytes,
        "ep_endpoint_serialization_floor_ps": endpoint_serialization,
        "ep_cut_drain_floor_ps": cut_serialization,
        "ep_phase_floor_ps": ep_floor,
        "ep_effective_serialization_floor_ps": max(endpoint_serialization, cut_serialization),
        "pp_hop_cut_bounds": hop_cuts,
        "pp_noncompute_share_floor": 1 - width * COMPUTE_PS / step_floor,
        "share_ceiling": 1,
        "sender_fct_ceiling_ps": None,
    }


def final_stage_event(graph, pp) -> CompletionEvent:
    """Project the accepted final calc(0) gate; never schedule a second runtime."""
    final_compute = [operation for operation in graph.operations
                     if isinstance(operation.work, ComputeWork)][-1]
    timestamp = pp[-1].completion_time_ps // 1000 * 1000 + 1000 + COMPUTE_PS
    return CompletionEvent(graph.execution_id, final_compute.operation_id,
                           EventPhase.COMPLETED, timestamp)


def packet_step_result(record, event: CompletionEvent, width: int, pp) -> StepResult:
    """Return a one-token metric from the packet authority's final-stage event.

    This projection is restricted to the frozen one-request serial PP chain.
    EP is unrelated background work; it contributes neither request latency
    nor an additive wait. Gate/quantization is separate from measured hop FCT.
    """
    completion = event.timestamp_ps
    service = sum(flow.fct_ps for flow in pp)
    compute = width * COMPUTE_PS
    overhead = completion - compute - service
    if overhead < 0 or len(record.scheduled) != 1 or record.virtual_time_ps != 0:
        raise ValueError("packet metric projection requires the frozen serial request")
    attribution = LatencyAttribution(kernel_ps=compute, nic_ps=service, control_ps=overhead)
    request = record.scheduled[0]
    metric = RequestMetric(request.request_id, request.phase, 1, completion, completion,
                           completion, None, attribution, AdditiveVisitTotals())
    return StepResult(record.step_index, completion, completion, (metric,))


class ContentionStepSink(HtsimStepSink):
    """Study-local single-program adapter over the sink's native execution seam.

    The ordinary sink planner orders graph artifacts. That changes concurrent
    EP traffic and is inappropriate for this probe. The call and native seam
    are specialized for one program and the frozen seed, which the ordinary
    sink config does not expose. Binary conversion and execution use the same
    production tools. The config supplies the native seam's physical settings.
    This does not extend the production sink's supported graph contract.
    """

    def __init__(self, out: Path, variant: str, spines: int, width: int, ep_width: int, profile: str):
        self.variant, self.spines = variant, spines
        self.width, self.ep_width = width, ep_width
        self.graph = reference.build_graph(width, ep_width)
        self.cell_bounds = bounds(variant, spines, width, ep_width)
        write_json(out / "pre_run_bounds.json", self.cell_bounds)
        placement = declared_pipeline_placement(width)
        fabric = declared_rail_fabric(placement, variant=variant, spine_count=spines)
        self.projection = project_declared_clos(fabric)
        write_json(out / "placement.json", asdict(placement))
        write_json(out / "fabric.json", fabric.to_dict())
        write_json(out / "endpoint_by_rank.json", self.projection.endpoint_by_rank)
        write_json(out / "semantic_graph.json", execution_graph_to_json(self.graph))
        self.topology_path = out / "clos.topo"
        self.topology_path.write_bytes(self.projection.topology_text.encode())
        super().__init__(HtsimStepSinkConfig(
            profile=profile, tp_ranks=(0,), dims=reference.dimensions(), workdir=out,
            linkspeed_bps=RATE, num_goal_ranks=64,
            topology=self.topology_path if profile == "rnic-cn" else None,
        ))
        self.row = None

    def _run_goal(self, plan, goal_path, completion_csv):
        # The accepted study explicitly selects seed 1. The native default is
        # different; never substitute it just because the sink lacks a knob.
        return run_htsim_rnic(HtsimRnicConfig(
            goal_bin=to_binary(goal_path), profile=plan.profile,
            linkspeed_bps=plan.linkspeed_bps, topology=plan.topology,
            completion_csv=completion_csv,
            extra_flags={"-rnic_cn_prbs_seed": "1"} if plan.profile == "rnic-cn" else {},
        ), timeout_s=120)

    def __call__(self, record) -> StepResult | None:
        if record != reference.step_record():
            raise ValueError("contention sink accepts only the frozen step record")
        out = self.config.workdir
        semantic_trace = render_serial_execution_graph_goal(self.graph, num_goal_ranks=64)
        projected_graph = self.projection.project_graph(self.graph)
        trace = render_serial_execution_graph_goal(projected_graph, num_goal_ranks=64)
        goal_path = out / "step.goal"
        goal_path.write_bytes(trace.render().encode())
        run = self._run_goal(self.config, goal_path, out / "completion.csv")
        (out / "manifest.log").write_bytes(("\n".join(run.manifest) + "\n").encode())
        flows = [self.projection.semantic_flow(flow) for flow in run.flows]
        write_json(out / "semantic_flows.json", [asdict(flow) for flow in flows])
        row, pp = reference.evaluate_cell(self.variant, self.width, self.ep_width,
                                          self.config.profile, semantic_trace, flows, run, self.cell_bounds)
        row.pop("exact_rail_oracle", None)
        row.update(spine_count=self.spines, oversubscription=f"{8 // self.spines}:1",
                   goal_sha256=reference.digest(goal_path), physical_quiescence=run.quiescent)
        findings = row["fatal_findings"]
        guard = lambda condition, message: reference.require_guard(condition, message, findings)
        guard(all(type(value) is int and value >= 0 for flow in flows for value in
                  (flow.start_time_ps, flow.completion_time_ps, flow.fct_ps)), "noninteger flow time")
        manifest = [" ".join(token for token in line.split()
                              if not token.startswith(("goal=", "completion_csv=", "topology=")))
                    for line in run.manifest]
        if self.config.profile == "rnic-cn":
            guard(any("global_seed=1 " in line for line in manifest), "unexpected native PRBS seed")
            guard(row.get("ep_phase_makespan_ps", 0) >= self.cell_bounds["ep_phase_floor_ps"],
                  "EP completion violates physical endpoint or shared-cut floor")
        row["manifest"] = manifest
        row["pp_fct_samples_ps"] = [flow.fct_ps for flow in pp]
        ep = [flow for flow in flows if flow.source % 8 != 0]
        # Flow lifetime overlap proves concurrent work, not queue ordering or
        # an exact count of bytes transmitted during this interval.
        row["pp_hop_ep_overlap_counts"] = [sum(
            other.start_time_ps < flow.completion_time_ps and other.completion_time_ps > flow.start_time_ps
            for other in ep) for flow in pp]
        row["pp_hop_same_cut_overlap_payload_bytes"] = [sum(
            other.payload_bytes for other in ep
            if other.start_time_ps < flow.completion_time_ps and other.completion_time_ps > flow.start_time_ps
            and leaf_for(flow.source, self.variant) != leaf_for(flow.destination, self.variant)
            and leaf_for(other.source, self.variant) != leaf_for(other.destination, self.variant)
            and (leaf_for(other.source, self.variant) == leaf_for(flow.source, self.variant)
                 or leaf_for(other.destination, self.variant) == leaf_for(flow.destination, self.variant))
        ) for flow in pp]
        result = None
        if not findings:
            event = final_stage_event(self.graph, pp)
            result = packet_step_result(record, event, self.width, pp)
            guard(result.completed_at_ps == row["pp_final_stage_projection_ps"], "StepResult boundary mismatch")
            guard(result.request_metrics[0].ttft_ps == result.step_latency_ps, "request TTFT mismatch")
            guard(result.request_metrics[0].attribution.total_ps == result.step_latency_ps,
                  "request metric attribution does not conserve")
            row["pp_hop_critical_path_share"] = sum(flow.fct_ps for flow in pp) / result.step_latency_ps
            row["packet_metric_projection"] = {
                "authority": "htsim-single-concurrent-goal",
                "completed_event_ps": event.timestamp_ps,
                "step_latency_ps": result.step_latency_ps,
                "request_ttft_ps": result.request_metrics[0].ttft_ps,
                "request_tpot_ps": None,
                "attribution": asdict(result.request_metrics[0].attribution),
                "scope": "study-local declared singleton-stage one-token projection",
            }
            write_json(out / "final_stage_event.json", asdict(event))
            write_json(out / "step_result.json", asdict(result))
        row["cell_status"] = "void" if findings else "clear"
        write_json(out / "cell.json", row)
        # Full flow records stay in bulk storage; compact samples retain the
        # exact nearest-rank quantiles in the public result artifact.
        row.pop("pp_hops", None)
        self.row = row
        return result if not findings else None


def cell_key(row: dict) -> tuple:
    return row["variant"], row["spine_count"], row["pp_width"], row["ep_width"], row["profile"]


def native_error_row(cell: tuple, error_type: str, message: str) -> dict:
    """Retain a portable failure class and pre-run bounds, never partial FCT."""
    if "fabric dropped control lifecycle" in message:
        failure = "fatal-control-lifecycle-drop"
    elif error_type == "TimeoutExpired":
        failure = "operational-timeout"
    else:
        failure = "native-or-validation-error"
    return {"cell": list(cell), "error_type": error_type, "cell_status": "void",
            "failure_class": failure, "bounds": bounds(*cell[:4]),
            "timing_claims": "void; no partial tail, phase or request score"}


def relation_rows(rows: list[dict], accepted: dict) -> list[dict]:
    """Keep exact oracles and behavioral hypotheses separate from fatal guards."""
    indexed = {cell_key(row): row for row in rows}
    old = {(row["variant"], row["pp_width"], row["ep_width"], row["profile"]): row
           for row in accepted["configurations"]}
    relations = []

    def relation(name, keys, metric, predicate, **extra):
        cells = [indexed.get(key) for key in keys]
        entry = {"family": name, "cells": [list(key) for key in keys], "metric": metric, **extra}
        if any(cell is None or cell["cell_status"] != "clear" for cell in cells):
            entry.update(status="unscored-void-or-missing", matched=None)
        else:
            values = [cell[metric] for cell in cells]
            matched = bool(predicate(values))
            entry.update(values=values, matched=matched, status="holds" if matched else "refuted")
        relations.append(entry)

    for width in WIDTHS:
        keys = [(variant, spines, width, 0, "rnic-nn") for variant, spines in product(VARIANTS, SPINES)]
        relation("R1-null-unloaded-equality", keys, "pp_fct_p99_ps", lambda values: len(set(values)) == 1)
        for variant in VARIANTS:
            for metric in ("pp_fct_p50_ps", "pp_fct_p99_ps"):
                expected = old[(variant, width, 0, "rnic-cn")][metric]
                relation("R2-reference-exact", [(variant, 8, width, 0, "rnic-cn")], metric,
                         lambda values, expected=expected: values == [expected], expected_ps=expected)
        keys = [("node-local", 2, width, ep, "rnic-cn") for ep in EP_WIDTHS]
        for metric in ("pp_fct_p99_ps", "pp_communication_projection_share"):
            relation("R3-node-local-load-growth", keys, metric,
                     lambda values: values[0] <= values[1] <= values[2] and values[0] < values[2])
        for ep in EP_WIDTHS:
            keys = [("rail", spines, width, ep, "rnic-cn") for spines in SPINES]
            relation("R4-rail-packet-bound", keys, "pp_fct_p99_ps",
                     lambda values: abs(values[1] - values[0]) <= PACKET_PS, tolerance_ps=PACKET_PS)
        for variant, ep in product(VARIANTS, (8, 32)):
            keys = [(variant, spines, width, ep, "rnic-cn") for spines in SPINES]
            b8, b2 = [bounds(variant, spines, width, ep) for spines in SPINES]
            ratio = b2["ep_effective_serialization_floor_ps"] / b8["ep_effective_serialization_floor_ps"]
            relation("R5-EP-effective-floor-ratio", keys, "ep_phase_makespan_ps",
                     lambda values, ratio=ratio: values[1] >= ratio * values[0], expected_minimum_ratio=ratio)
            if b8["ep_busiest_leaf_uplink_bytes"]:
                relation("R5-EP-raw-cut-ratio", keys, "ep_phase_makespan_ps",
                         lambda values: values[1] >= 4 * values[0], expected_minimum_ratio=4)
    return relations


def draw(summary: dict, directory: Path) -> None:
    """Plain quantitative views from clear cells only; missing cells leave gaps."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [row for row in summary["configurations"]
            if row["cell_status"] == "clear" and row["profile"] == "rnic-cn"]
    directory.mkdir(parents=True, exist_ok=True)
    for name, metric, ylabel, scale in (
        ("pp_tail", "pp_fct_p99_ps", "PP hop p99 FCT (µs)", 1e6),
        ("phase_share", "ep_phase_makespan_ps", "EP phase makespan (ms)", 1e9),
    ):
        figure, axes = plt.subplots(2, 3, figsize=(7.2, 6.4), sharex=True)
        figure.subplots_adjust(left=0.085, right=0.985, top=0.89, bottom=0.20,
                               wspace=0.48, hspace=0.58)
        for column, width in enumerate(WIDTHS):
            for row_index, variant in enumerate(VARIANTS):
                ax = axes[row_index, column]
                for spines, linestyle in ((2, "-"), (8, "--")):
                    selected = sorted((row for row in rows if row["variant"] == variant
                                       and row["pp_width"] == width and row["spine_count"] == spines),
                                      key=lambda row: row["ep_width"])
                    ax.plot([row["ep_width"] for row in selected],
                            [row[metric] / scale for row in selected], marker="o",
                            markersize=5 if spines == 2 else 3,
                            markerfacecolor="white" if spines == 2 else "C0",
                            markeredgewidth=1.1, linewidth=1.5,
                            linestyle=linestyle, color="C1" if spines == 2 else "C0",
                            label=f"{8 // spines}:1 ({spines} spines)")
                ax.set_title(f"{variant}, P={width}", fontsize=9)
                ax.set_xticks(EP_WIDTHS)
                ax.set_xlabel("EP participants, W (count)", fontsize=8)
                ax.set_ylabel(ylabel, fontsize=8)
                ax.tick_params(labelsize=8, labelbottom=True)
                ax.margins(x=0.08, y=0.18)
                if name == "pp_tail":
                    floor = selected[0]["bounds"]["data_arrival_floor_ps"] / scale
                    ax.axhline(floor, linestyle=":", color="0.45", label="PP data floor")
                    ax.set_ylim(0, 20)
                    ax.set_yticks((0, 5, 10, 15, 20))
                    unloaded = {row["spine_count"]: row[metric] for row in rows
                                if row["variant"] == variant and row["pp_width"] == width
                                and row["ep_width"] == 0}
                    if {2, 8} <= set(unloaded):
                        offset = (unloaded[2] - unloaded[8]) / 1e6
                        ax.text(0.04, unloaded[8] / 1e6 / 20 - 0.06,
                                "4:1 minus 1:1, W=0\n"
                                f"+{offset:.2f} µs (unloaded)", transform=ax.transAxes,
                                va="top", fontsize=7.5, color="0.2")
                    changes = []
                    for spines in (2, 8):
                        by_ep = {row["ep_width"]: row[metric] for row in rows
                                 if row["variant"] == variant and row["pp_width"] == width
                                 and row["spine_count"] == spines}
                        changes.append(by_ep[8] - by_ep[0])
                    delta = (f"{changes[0]:,} ps (both fabrics)" if changes[0] == changes[1]
                             else f"{changes[0]:,} / {changes[1]:,} ps (4:1 / 1:1)")
                    ax.text(0.04, 0.14, f"p99 change, W=0 to 8\n{delta}",
                            transform=ax.transAxes, va="top", fontsize=7.5)
                else:
                    floors = [row["bounds"]["ep_phase_floor_ps"] / scale for row in selected]
                    ax.plot(EP_WIDTHS, floors, linestyle=":", marker="_", markersize=7,
                            color="0.45", label="1:1 EP phase floor")
                    shares = [row for row in rows if row["variant"] == variant
                              and row["pp_width"] == width and row["spine_count"] == 2]
                    if shares:
                        shares.sort(key=lambda row: row["ep_width"])
                        text = ", ".join(f"{100 * row['pp_hop_critical_path_share']:.1f}" for row in shares)
                        ax.text(0.04, 0.95, f"4:1 PP hop share (%)\nW=0,8: {text}",
                                transform=ax.transAxes, va="top", fontsize=7.5)
                    ax.set_ylim(-0.035, 1.3)
                    ax.set_yticks((0, 0.4, 0.8, 1.2))
                ax.text(0.98, 0.10 if name == "phase_share" else 0.94, "4:1 W=32: void",
                        transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color="0.35")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.09),
                      ncols=len(handles), fontsize=8, frameon=False)
        figure.text(0.5, 0.045, "Void cells have no timing value. TP=1; TP phase is zero.",
                    ha="center", fontsize=8)
        figure.suptitle("Physical Clos: " + ("PP finite-chain maximum" if name == "pp_tail"
                                            else "EP phase and PP critical path"), fontsize=11)
        figure.savefig(directory / f"{name}.png", dpi=220)
        figure.savefig(directory / f"{name}.pdf", metadata={"CreationDate": None})
        plt.close(figure)


def make_summary(rows, errors, accepted, executables, identities, grid) -> dict:
    manifests = {}
    for row in rows:
        manifest = row.pop("manifest", [])
        digest = hashlib.sha256("\n".join(manifest).encode()).hexdigest()
        manifests[digest] = manifest
        row["manifest_sha256"] = digest
    return {
        "expectations_commit": EXPECTATIONS_COMMIT,
        "htsim_build_commit": "617ce20",
        "seed": 1,
        "reference_input_lf_sha256": REFERENCE_DIGESTS,
        "default_identity_guards": identities,
        "executable_sha256": {binary.name: reference.digest(binary) for binary in executables},
        "requested_configuration_count": len(grid), "configuration_count": len(rows),
        "configurations": rows, "execution_errors": errors,
        "backend_manifests_by_sha256": manifests,
        "behavioral_and_exact_relations": relation_rows(rows, accepted),
        "fatal_status": "void" if errors or any(row["cell_status"] == "void" for row in rows) else "clear",
        "evidence_class": "declared-packet-simulator",
        "metric_scope": "study-local final-stage event to StepResult and one-token TTFT",
        "queue_byte_scope": "whole-phase cut and temporal overlap; no per-packet queue ordering",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--summary", type=Path, default=HERE / "results.json")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--cell", nargs=5, metavar=("VARIANT", "SPINES", "P", "EP", "PROFILE"))
    args = parser.parse_args()
    if args.plot_only:
        draw(json.loads(args.summary.read_bytes()), HERE / "figures")
        return
    root = args.out or (Path(os.environ["SIMLLM_DATA_ROOT"]) / "pp_rail_contention_v1"
                        if os.environ.get("SIMLLM_DATA_ROOT") else None)
    if root is None or root.resolve().is_relative_to(HERE.parents[1]):
        parser.error("set SIMLLM_DATA_ROOT or --out to an external bulk directory")
    executables = (find_htsim_rnic(), find_txt2bin())
    if any(binary is None for binary in executables):
        parser.error("set SIMLLM_HTSIM_RNIC and SIMLLM_TXT2BIN to the pinned binaries")
    accepted = reference_inputs()
    identities = identity_guards()
    grid = list(product(VARIANTS, SPINES, WIDTHS, EP_WIDTHS, PROFILES))
    if args.cell:
        variant, spines, width, ep, profile = args.cell
        selected = (variant, int(spines), int(width), int(ep), profile)
        if selected not in grid:
            parser.error("--cell must select a member of the frozen grid")
        grid = [selected]
    # Write every cell's byte accounting before the first native invocation.
    write_json(root / "all_pre_run_bounds.json", [
        {"cell": list(cell), "bounds": bounds(*cell[:4])} for cell in grid
    ])
    rows, errors = [], []
    for variant, spines, width, ep, profile in grid:
        name = f"{variant}-s{spines}-pp{width}-ep{ep}-{profile}"
        print(f"RUN {name}", flush=True)
        try:
            sink = ContentionStepSink(root / name, variant, spines, width, ep, profile)
            result = sink(reference.step_record())
            row = sink.row
            if row is None:
                raise ValueError("sink did not publish its evidence row")
            rows.append(row)
            print(f"  {row['cell_status']}: p99={row.get('pp_fct_p99_ps')} ps; "
                  f"step={result.step_latency_ps if result else None} ps; "
                  f"guards={row['fatal_findings']}", flush=True)
        except (RuntimeError, ValueError, OSError, subprocess.TimeoutExpired) as error:
            # Do not copy native diagnostic paths into tracked artifacts.
            log = root / name / "error.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_bytes((str(error) + "\n").encode())
            errors.append(native_error_row((variant, spines, width, ep, profile),
                                           type(error).__name__, str(error)))
            print(f"  VOID: {type(error).__name__}; diagnostic retained in bulk", flush=True)
    summary = make_summary(rows, errors, accepted, executables, identities, grid)
    write_json(args.summary, summary)
    if not args.cell:
        draw(summary, HERE / "figures")
    print(f"CONFIGURATIONS {len(rows)}/{len(grid)}; FATAL STATUS {summary['fatal_status']}", flush=True)


if __name__ == "__main__":
    main()
