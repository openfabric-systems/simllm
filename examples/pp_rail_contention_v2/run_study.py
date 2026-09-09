"""Execute the frozen queue-attribution study through the native RNIC seam.

Inputs, physical bounds and source identities are retained before execution.
The native runtime owns every timestamp; this runner only audits and projects.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing
import os
import platform
import shutil
import subprocess
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from functools import partial
from itertools import pairwise, product
from pathlib import Path

from examples.pp_rail_contention_v1 import run_study as previous
from simllm.backends._child_process import run_owned_process
from simllm.backends.fct import earliest_completion_byte_floors, normalized_fct
from simllm.backends.htsim_rnic import (
    HtsimRnicConfig,
    RnicRunResult,
    _parse_goal_completion_time_ps,
    build_htsim_rnic_command,
    parse_completion_csv,
    parse_control_recovery_manifest,
    parse_data_recovery_manifest,
    prepare_htsim_child_lifetime,
)
from simllm.backends.rail_topology import project_declared_clos
from simllm.core import execution_graph_to_json
from simllm.goal import to_binary
from simllm.placement import declared_pipeline_placement, declared_rail_fabric
from simllm.traffic import render_serial_execution_graph_goal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FREEZE = "d13a593ce3f347f6b3ed4b499d170411d609f39e"
HIGH_WINDOW_PS = 10_566_400
LOW_WINDOW_PS = 6_572_800
TICK_PS = 16_000
RATE = 400_000_000_000
BUFFER_BYTES = 1_048_576
reference = previous.reference
write_json = previous.write_json


@dataclass(frozen=True)
class Cell:
    population: str
    variant: str
    spines: int
    width: int
    ep_width: int
    window_ps: int | None
    profile: str = "rnic-cn"

    @property
    def reference_name(self):
        return (f"{self.variant}-s{self.spines}-pp{self.width}"
                f"-ep{self.ep_width}-{self.profile}")

    @property
    def name(self):
        window = "auto" if self.window_ps is None else str(self.window_ps)
        return f"{self.population}-{self.reference_name}-d{window}"


def cells():
    physical = [Cell("physical", variant, spines, 2, ep, HIGH_WINDOW_PS)
                for variant, spines, ep in product(reference.VARIANTS, (8, 2), (0, 8, 32))]
    physical += [Cell("physical", variant, 2, width, ep, HIGH_WINDOW_PS)
                 for variant, width, ep in product(reference.VARIANTS, (4, 8), (0, 32))]
    physical += [Cell("physical", variant, spines, 2, 0, LOW_WINDOW_PS)
                 for variant, spines in product(reference.VARIANTS, (8, 2))]
    legacy = [Cell("legacy", variant, 8, width, ep, None)
              for variant, width, ep in product(reference.VARIANTS, (2, 4, 8), (0, 8))]
    ideal = [Cell("ideal", variant, spines, width, ep, None, "rnic-nn")
             for variant, spines, width, ep in product(reference.VARIANTS, (8, 2),
                                                      (2, 4, 8), (0, 8, 32))]
    return physical + legacy + ideal


def digest(path):
    with Path(path).open("rb") as handle:
        result = hashlib.sha256()
        for block in iter(lambda: handle.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def git(*args, cwd=REPO):
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def physical_bounds(cell):
    base = previous.bounds(cell.variant, cell.spines, cell.width, cell.ep_width)
    receiver_bytes = Counter()
    for source, destination, size in reference.ep_pairs(cell.ep_width):
        receiver_bytes[destination] += size
    for stage in range(cell.width - 1):
        receiver_bytes[8 * (stage + 1)] += reference.B
    service = max(max(receiver_bytes.values()) * 20,
                  base["ep_effective_serialization_floor_ps"], base["pp_step_data_floor_ps"])
    queue_allowance = 3 * BUFFER_BYTES * 20 + 8_000_000 + 4160 * 20
    return {**base, "receiver_payload_bytes": dict(sorted(receiver_bytes.items())),
            "budget_service_ps": service, "budget_queue_allowance_ps": queue_allowance,
            "engineering_budget_ps": 9 * service + 8 * (40_000_000 + queue_allowance),
            "budget_metric": "max(all completion)-min(all start)",
            "physical_path_floors_apply": cell.profile == "rnic-cn",
            "ring_window_ps": cell.window_ps, "ring_tick_ps": TICK_PS,
            "initial_window_bytes": None, "unconditional_fct_ceiling_ps": None}


def reference_locks():
    """Use already-published digest oracles, never mutable bulk CSVs alone."""
    result = json.loads((HERE.parent / "data_recovery_v1/results.json").read_bytes())
    locks = {}
    for row in result["cells"]:
        if row["cell"].startswith("pipeline-") and row["arm"] == "legacy-none":
            key = row["cell"].removeprefix("pipeline-")
            if key in locks:
                raise ValueError("duplicate published pipeline reference lock")
            locks[key] = row["inputs"]
    if len(locks) != 36:
        raise ValueError("published pipeline reference population changed")
    return locks


def require_digest(path, accepted):
    accepted = [accepted] if isinstance(accepted, str) else accepted
    raw = Path(path).read_bytes()
    candidates = {hashlib.sha256(raw).hexdigest(),
                  hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()}
    if not candidates.intersection(accepted):
        raise ValueError(f"reference bytes changed: {Path(path).name}")


def prepare_cell(cell, root, references, locks, txt2bin):
    out = root / cell.name
    out.mkdir()
    write_json(out / "pre_run_bounds.json", physical_bounds(cell))
    placement = declared_pipeline_placement(cell.width)
    fabric = declared_rail_fabric(placement, variant=cell.variant, spine_count=cell.spines)
    projection = project_declared_clos(fabric)
    graph = reference.build_graph(cell.width, cell.ep_width)
    write_json(out / "placement.json", asdict(placement))
    write_json(out / "fabric.json", fabric.to_dict())
    write_json(out / "endpoint_by_rank.json", projection.endpoint_by_rank)
    write_json(out / "semantic_graph.json", execution_graph_to_json(graph))
    (out / "clos.topo").write_bytes(projection.topology_text.encode())
    trace = render_serial_execution_graph_goal(projection.project_graph(graph), num_goal_ranks=64)
    (out / "step.goal").write_bytes(trace.render().encode())
    to_binary(out / "step.goal", tool=txt2bin)
    physical_name = cell.reference_name.replace("rnic-nn", "rnic-cn")
    lock = locks[physical_name]
    retained = references / physical_name
    for name, field in (("step.goal", "goal_text_sha256"), ("step.bin", "goal_sha256"),
                        ("clos.topo", "topology_sha256")):
        require_digest(out / name, lock[field])
        require_digest(retained / name, lock[field])
    for name in ("fabric.json", "placement.json", "semantic_graph.json", "endpoint_by_rank.json"):
        if json.loads((out / name).read_bytes()) != json.loads((retained / name).read_bytes()):
            raise ValueError(f"declared reference changed: {name}")
    ideal_path = references / physical_name.replace("rnic-cn", "rnic-nn") / "completion.csv"
    require_digest(ideal_path, lock["ideal_completion_sha256"])
    if cell.population == "legacy":
        require_digest(retained / "completion.csv", lock["reference_completion_sha256"])
    write_json(out / "inputs.json", {
        "cell": asdict(cell), "expectations_commit": FREEZE,
        "published_reference": lock,
        "input_sha256": {name: digest(out / name) for name in (
            "step.goal", "step.bin", "clos.topo", "fabric.json", "placement.json",
            "semantic_graph.json", "endpoint_by_rank.json")},
    })


def configuration(cell, inputs, out, traced):
    if traced and cell.population != "physical":
        raise ValueError("compatibility populations have no trace arm")
    flags = {"-rnic_cn_prbs_seed": "1"} if cell.profile == "rnic-cn" else {}
    if cell.population == "physical":
        flags.update({"-rnic_cn_ring_window_ps": str(cell.window_ps),
                      "-rnic_cn_ring_tick_ps": str(TICK_PS),
                      "-rnic_cn_ring_capacity_bytes": str(BUFFER_BYTES),
                      "-rnic_cn_ns_tm3_buffer_bytes": str(BUFFER_BYTES),
                      "-rnic_cn_control_headroom_bytes": "131072",
                      "-rnic_cn_control_deadline_ps": "10000000",
                      "-rnic_cn_margin_ppm": "900000",
                      "-rnic_cn_control_wire_bytes": "64",
                      "-rnic_cn_max_retransmissions": "8",
                      "-rnic_cn_retransmission_rto_ps": "50000000000"})
    if traced:
        flags["-rnic_cn_trace_dir"] = str(out / "trace")
    return HtsimRnicConfig(
        inputs / "step.bin", cell.profile, RATE, completion_csv=out / "completion.csv",
        topology=inputs / "clos.topo" if cell.profile == "rnic-cn" else None,
        extra_flags=flags,
        control_recovery="headroom" if cell.population == "physical" else "none",
        data_recovery="exponential" if cell.population == "physical" else "none",
    )


def manifest_value(lines, key):
    values = [token.partition("=")[2] for line in lines for token in line.split()
              if token.startswith(key + "=")]
    if len(values) != 1:
        raise ValueError(f"missing or repeated manifest field: {key}")
    return values[0]


def phase(flows):
    return max(f.completion_time_ps for f in flows) - min(f.start_time_ps for f in flows)


def normalized_rows(flows, ideal, out):
    with (out / "per_flow_normalization.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("source", "destination", "tag", "payload_bytes", "fct_ps",
                         "baseline_fct_ps", "normalized_fct", "interpretation"))
        for flow in normalized_fct(flows, ideal):
            writer.writerow((flow.source, flow.destination, flow.tag, flow.payload_bytes,
                             flow.fct_ps, flow.baseline_fct_ps, flow.slowdown,
                             "diagnostic-shared-or-dynamic-phase"))


def exact_goal_timing(pp, width, findings):
    """Check the retained calc gates independently of the metric projection.

    Native calc(0) consumes one nanosecond. Compute events schedule relative
    to the current physical picosecond, preserving a receive's sub-nanosecond
    remainder. Only the printed final GOAL boundary rounds down to nanoseconds.
    """
    guard = lambda condition, message: reference.require_guard(condition, message, findings)
    guard(len(pp) == width - 1, "exact GOAL timing lacks the full pipeline chain")
    if not pp:
        return None
    guard(pp[0].start_time_ps == reference.COMPUTE_PS + 1000,
          "first PP release differs from compute plus one calc gate")
    for previous_flow, following in pairwise(pp):
        guard(following.start_time_ps == previous_flow.completion_time_ps + reference.COMPUTE_PS + 2000,
              "PP release differs from preceding delivery plus compute and two calc gates")
    return ((pp[-1].completion_time_ps + reference.COMPUTE_PS + 1000) // 1000) * 1000


def trace_boundary_guard(audit, manifest):
    if audit.get("schema") != "pp-queue-audit-v1" or audit.get("status") != "valid":
        raise ValueError("trace audit is not a valid known-schema result")
    if audit.get("physical_quiescence_time_ps") != int(manifest_value(manifest, "physical_quiescence_time_ps")):
        raise ValueError("trace and process quiescence boundaries differ")


def analyze_execution(cell, inputs, out, references, execution, stdout, traced):
    """Interpret only after run.log, completion.csv and execution.json exist."""
    row = {"cell": cell.name, **asdict(cell), "traced": traced, "cell_status": "void",
           "fatal_findings": [], "request_ttft_ps": None, "pp_hop_maximum_ps": None,
           "request_tpot_ps": None, "trace_audit": None, "completion_sha256": None}
    if execution["returncode"] != 0:
        row["fatal_findings"].append(f"native execution exited {execution['returncode']}")
        return row
    flows = parse_completion_csv(out / "completion.csv")
    manifest = [line for line in stdout.splitlines() if line.startswith("[RNIC manifest]")]
    run = RnicRunResult(flows, manifest,
                        manifest_value(manifest, "physical_quiescence") == "verified",
                        _parse_goal_completion_time_ps(stdout),
                        parse_control_recovery_manifest(manifest), parse_data_recovery_manifest(manifest))
    projection = project_declared_clos(declared_rail_fabric(
        declared_pipeline_placement(cell.width), variant=cell.variant, spine_count=cell.spines))
    semantic_flows = [projection.semantic_flow(flow) for flow in flows]
    graph = reference.build_graph(cell.width, cell.ep_width)
    semantic_trace = render_serial_execution_graph_goal(graph, num_goal_ranks=64)
    b = physical_bounds(cell)
    checked, pp = reference.evaluate_cell(cell.variant, cell.width, cell.ep_width,
                                           cell.profile, semantic_trace, semantic_flows, run, b)
    findings = row["fatal_findings"] = checked["fatal_findings"]
    guard = lambda condition, message: reference.require_guard(condition, message, findings)
    final_goal_time = exact_goal_timing(pp, cell.width, findings)
    whole_phase = phase(flows)
    guard(whole_phase <= b["engineering_budget_ps"], "complete-flow phase exceeds frozen budget")
    guard(all(type(value) is int and value >= 0 for flow in flows for value in
              (flow.start_time_ps, flow.completion_time_ps, flow.fct_ps)), "invalid flow time")
    ep = [flow for flow in semantic_flows if flow.source % 8 != 0]
    ep_phase = phase(ep) if ep else 0
    ideal_path = references / cell.reference_name.replace("rnic-cn", "rnic-nn") / "completion.csv"
    ideal = parse_completion_csv(ideal_path)
    if cell.profile == "rnic-cn":
        guard(manifest_value(manifest, "global_seed") == "1", "native seed changed")
        guard(ep_phase >= b["ep_phase_floor_ps"], "EP phase beats directional capacity floor")
        guard(whole_phase >= phase(ideal), "physical complete phase beats identical ideal phase")
        guard(all(flow.fct_ps >= flow.payload_bytes * 20
                  + (2_000_000 if flow.source // 8 == flow.destination // 8 else 4_000_000)
                  for flow in flows), "flow beats payload service and physical path propagation")
        for destination in {flow.destination for flow in flows}:
            received = [flow for flow in flows if flow.destination == destination]
            propagation = min(2_000_000 if flow.source // 8 == destination // 8 else 4_000_000
                              for flow in received)
            guard(all(bound.ok for bound in earliest_completion_byte_floors(
                received, link_rate_bps=RATE, propagation_ps=propagation)),
                f"receiver {destination} completion prefix beats byte floor")
        normalized_rows(flows, ideal, out)
    if cell.population == "physical":
        for key, expected in (("ring_window_ps", str(cell.window_ps)), ("ring_tick_ps", str(TICK_PS)),
                              ("rnic_cn_data_recovery", "exponential"),
                              ("rnic_cn_control_recovery", "headroom"),
                              ("rnic_cn_initial_window", "none"),
                              ("rnic_cn_retry_probe_windows", "4")):
            guard(manifest_value(manifest, key) == expected, f"fixed selection changed: {key}")
        guard(int(manifest_value(manifest, "rnic_cn_maximum_retry_attempt_observed")) <= 8,
              "retry limit exceeded")
    row["completion_sha256"] = digest(out / "completion.csv")
    if cell.population != "physical":
        lock = reference_locks()[cell.reference_name.replace("rnic-nn", "rnic-cn")]
        expected = lock["ideal_completion_sha256" if cell.population == "ideal"
                        else "reference_completion_sha256"]
        guard(row["completion_sha256"] == expected, "protected completion CSV changed")
        row["exact_oracle"] = {"family": cell.population + "-completion-csv",
                               "expected_sha256": expected, "matches": row["completion_sha256"] == expected}
    if traced:
        from examples.pp_rail_contention_v2.trace_analysis import audit_trace

        audit = audit_trace(out / "trace", completion_rows=[asdict(flow) for flow in flows],
                            endpoint_by_rank=projection.endpoint_by_rank,
                            spine_count=cell.spines,
                            delta_ps=cell.window_ps, tick_ps=TICK_PS)
        write_json(out / "trace_audit.json", audit)
        trace_boundary_guard(audit, manifest)
        row["trace_audit"] = audit
    row.update(physical_quiescence=run.quiescent,
               physical_quiescence_time_ps=int(manifest_value(manifest, "physical_quiescence_time_ps")),
               bounds=b, complete_flow_phase_ps=whole_phase, ep_phase_ps=ep_phase,
               ideal_phase_ps=phase(ideal), normalized_phase_makespan=whole_phase / phase(ideal),
               job_completion_ps=run.job_completion_time_ps(),
               native_manifest=[" ".join(token for token in line.split()
                                if not token.startswith(("goal=", "completion_csv=", "topology=")))
                                for line in manifest])
    if not findings:
        event = previous.final_stage_event(graph, pp)
        result = previous.packet_step_result(reference.step_record(), event, cell.width, pp)
        guard(event.timestamp_ps == checked["pp_final_stage_projection_ps"], "final stage event changed")
        guard(event.timestamp_ps == final_goal_time, "request event differs from exact GOAL timing")
        guard(result.request_metrics[0].ttft_ps == event.timestamp_ps, "TTFT differs from final stage")
        guard(result.request_metrics[0].attribution.total_ps == result.step_latency_ps,
              "request attribution does not conserve")
        guard(checked["pp_goal_gate_and_quantization_ps"]
              == 2000 * (cell.width - 1) - pp[-1].completion_time_ps % 1000,
              "GOAL gates and printed nanosecond quantization do not conserve exactly")
        if not findings:
            write_json(out / "final_stage_event.json", asdict(event))
            write_json(out / "step_result.json", asdict(result))
            row.update(cell_status="valid", request_ttft_ps=result.request_metrics[0].ttft_ps,
                       pp_hop_maximum_ps=max(flow.fct_ps for flow in pp),
                       pp_fct_samples_ps=[flow.fct_ps for flow in pp],
                       request_attribution=asdict(result.request_metrics[0].attribution),
                       pp_hops=checked["pp_hops"])
    return row


def execute(cell, traced, root, references, binary, timeout_s):
    inputs = root / cell.name
    out = inputs / ("trace-on" if traced else "trace-off")
    out.mkdir()
    command = build_htsim_rnic_command(binary, configuration(cell, inputs, out, traced))
    write_json(out / "command.json", {"argv": command})
    try:
        native = run_owned_process(command, timeout_s=timeout_s)
        stdout, returncode = native.stdout + "\n" + native.stderr, native.returncode
    except subprocess.TimeoutExpired as error:
        decode = lambda value: value.decode(errors="replace") if isinstance(value, bytes) else value or ""
        stdout, returncode = decode(error.output) + "\n" + decode(error.stderr), 124
    (out / "run.log").write_bytes(stdout.encode())
    execution = {"returncode": returncode, "run_log_sha256": digest(out / "run.log"),
                 "completion_sha256": digest(out / "completion.csv")
                 if (out / "completion.csv").exists() else None}
    write_json(out / "execution.json", execution)
    try:
        row = analyze_execution(cell, inputs, out, references, execution, stdout, traced)
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as error:
        (out / "audit_error.txt").write_text(f"{type(error).__name__}: {error}\n")
        row = {"cell": cell.name, **asdict(cell), "traced": traced, "cell_status": "void",
               "fatal_findings": [f"{type(error).__name__}: {error}"],
               "request_ttft_ps": None, "pp_hop_maximum_ps": None, "request_tpot_ps": None,
               "trace_audit": None, "completion_sha256": execution["completion_sha256"]}
    write_json(out / "cell.json", row)
    print(f"{cell.name} trace={int(traced)} {row['cell_status']} "
          f"ttft_ps={row['request_ttft_ps']} findings={len(row['fatal_findings'])}", flush=True)
    return row


def compare_windows(low, high):
    def packets(row):
        return {(flow["source_rank"], flow["destination_rank"], packet["packet_index"]): packet
                for flow in row["trace_audit"]["pp_flows"] for packet in flow["packets"]
                if packet["attempt"] == 0}

    left, right = packets(low), packets(high)
    if not left or left.keys() != right.keys():
        raise ValueError("unloaded window comparison lacks identical original packet inventory")
    differences = [{"source_rank": key[0], "destination_rank": key[1], "packet_index": key[2],
                    "arrival_delta_ps": right[key]["arrival_ps"] - left[key]["arrival_ps"],
                    "release_delta_ps": right[key]["logical_release_ps"] - left[key]["logical_release_ps"],
                    "holding_delta_ps": right[key]["ring_holding_ps"] - left[key]["ring_holding_ps"]}
                   for key in sorted(left)]
    return {"holding_nondecreasing": all(row["holding_delta_ps"] >= 0 for row in differences),
            "packet_differences": differences,
            "request_delta_ps": high["request_ttft_ps"] - low["request_ttft_ps"],
            "hop_maximum_delta_ps": high["pp_hop_maximum_ps"] - low["pp_hop_maximum_ps"]}


def load_holding_comparisons(low, high):
    def originals(row):
        return {(flow["source_rank"], flow["destination_rank"], packet["packet_index"]): packet
                for flow in row["trace_audit"]["pp_flows"] for packet in flow["packets"]
                if packet["attempt"] == 0}

    left, right = originals(low), originals(high)
    rows = []
    fields = ("source_start_ps", "eta_ps", "arrival_ps", "logical_release_ps", "ring_holding_ps",
              "rx_service_start_ps", "rx_service_end_ps", "delivery_ps")
    for key in sorted(left.keys() & right.keys()):
        a, b = left[key], right[key]
        if any(a.get(field) is None or b.get(field) is None for field in fields):
            continue
        differences = {field.removesuffix("_ps") + "_delta_ps": b[field] - a[field] for field in fields}
        queue_delta = (sum(v["queue_wait_ps"] for v in b["visits"])
                       - sum(v["queue_wait_ps"] for v in a["visits"]))
        absorbed = (queue_delta > 0 and differences["arrival_delta_ps"] == queue_delta
                    and differences["ring_holding_delta_ps"] == -queue_delta
                    and all(differences[name + "_delta_ps"] == 0 for name in (
                        "source_start", "eta", "logical_release", "rx_service_start", "rx_service_end", "delivery")))
        rows.append({"source_rank": key[0], "destination_rank": key[1], "packet_index": key[2],
                     **differences, "switch_queue_wait_delta_ps": queue_delta,
                     "extra_queue_wait_absorbed_by_holding": absorbed})
    return {"comparable_original_count": len(rows),
            "originals_with_changed_or_missing_admission": len(left.keys() | right.keys()) - len(rows),
            "absorbed_original_count": sum(row["extra_queue_wait_absorbed_by_holding"] for row in rows),
            "packets": rows}


def behavioral_relations(physical):
    indexed = {(row["variant"], row["spines"], row["width"], row["ep_width"], row["window_ps"]): row
               for row in physical}
    relations, diagnostics = [], []

    def relation(family, keys, evaluate):
        selected = [indexed.get(key) for key in keys]
        entry = {"family": family, "cells": [row["cell"] if row else None for row in selected]}
        if any(row is None or row["cell_status"] != "valid" for row in selected):
            entry.update(status="uninterpretable", matched=None)
        else:
            details, matched = evaluate(selected)
            entry.update(details=details, matched=bool(matched), status="holds" if matched else "refuted")
        relations.append(entry)

    for variant, spines in product(reference.VARIANTS, (8, 2)):
        def window(rows):
            result = compare_windows(*rows)
            return result, result["holding_nondecreasing"]

        relation("R1-window-holding", [(variant, spines, 2, 0, value)
                                       for value in (LOW_WINDOW_PS, HIGH_WINDOW_PS)], window)
    for width in (2, 4, 8):
        keys = [("node-local", 2, width, ep, HIGH_WINDOW_PS)
                for ep in ((0, 8, 32) if width == 2 else (0, 32))]

        def queue(rows):
            values = [row["trace_audit"]["queue_work"]["ep_data_service_ahead_ps"] for row in rows]
            lower_bounds = [sum(flow["ep_data_bound_ps"] for flow in row["trace_audit"]["pp_flows"])
                            for row in rows]
            return {"ep_widths": [row["ep_width"] for row in rows],
                    "ep_data_service_ahead_work_ps": values,
                    "strictly_earlier_ep_data_bound_work_ps": lower_bounds}, values[-1] > 0

        relation("R2-expert-service-ahead", keys, queue)

        def penalty(rows):
            hop_delta = rows[-1]["pp_hop_maximum_ps"] - rows[0]["pp_hop_maximum_ps"]
            request_delta = rows[-1]["request_ttft_ps"] - rows[0]["request_ttft_ps"]
            return {"hop_maximum_delta_ps": hop_delta, "request_ttft_delta_ps": request_delta,
                    "unloaded_ttft_ps": rows[0]["request_ttft_ps"],
                    "loaded_ttft_ps": rows[-1]["request_ttft_ps"],
                    "holding_comparison": load_holding_comparisons(rows[0], rows[-1])}, hop_delta > 0 and request_delta > 0

        relation("R3-request-penalty", [keys[0], keys[-1]], penalty)
    for variant in reference.VARIANTS:
        selected = [indexed[(variant, spines, 2, 32, HIGH_WINDOW_PS)] for spines in (8, 2)]
        valid = all(row["cell_status"] == "valid" for row in selected)
        diagnostics.append({"family": "R4-cut-response", "variant": variant,
                            "status": "interpretable" if valid else "uninterpretable",
                            "spines_2_over_8_phase_ratio": selected[1]["ep_phase_ps"] / selected[0]["ep_phase_ps"]
                            if valid else None,
                            "phase_ps": [row["ep_phase_ps"] for row in selected] if valid else None,
                            "phase_floors_ps": [row["bounds"]["ep_phase_floor_ps"] for row in selected]
                            if valid else None})
    return relations, diagnostics


def summarize(rows, provenance):
    expected = {(cell.name, traced) for cell in cells() for traced in
                ((False, True) if cell.population == "physical" else (False,))}
    keys = [(row["cell"], row["traced"]) for row in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("execution population is missing, duplicated or outside the freeze")
    indexed = dict(zip(keys, rows))
    physical, controls, exact_pairs = [], [], []
    findings = []
    pair_fields = ("completion_sha256", "request_ttft_ps", "pp_hop_maximum_ps",
                   "physical_quiescence", "physical_quiescence_time_ps", "complete_flow_phase_ps",
                   "ep_phase_ps", "job_completion_ps", "request_attribution", "pp_fct_samples_ps",
                   "native_manifest")
    for cell in cells():
        off = indexed[(cell.name, False)]
        if cell.population != "physical":
            controls.append(off)
            if off["cell_status"] != "valid":
                findings.append({"cell": cell.name, "findings": off["fatal_findings"]})
            continue
        on = indexed[(cell.name, True)]
        row = dict(on)
        fatal = list(off["fatal_findings"]) + list(on["fatal_findings"])
        both_valid = off["cell_status"] == on["cell_status"] == "valid"
        differences = [field for field in pair_fields if field not in off or field not in on
                       or off[field] != on[field]] if both_valid else []
        if differences:
            fatal.append("trace selection changed " + ", ".join(differences))
        if not both_valid and not fatal:
            fatal.append("trace pair lacks two valid executions")
        exact_pairs.append({"cell": cell.name, "family": "trace-selection-identity",
                            "status": "exact" if both_valid and not fatal else "void",
                            "mismatched_fields": differences,
                            "completion_sha256": off.get("completion_sha256")})
        row["fatal_findings"] = fatal
        if fatal:
            findings.append({"cell": cell.name, "findings": fatal})
            row.update(cell_status="void", request_ttft_ps=None, pp_hop_maximum_ps=None,
                       request_attribution=None, trace_audit=None)
        physical.append(row)
    relations, diagnostics = behavioral_relations(physical)
    refuted = [row["family"] for row in relations if row["status"] == "refuted"]
    verdict = "void" if findings else "valid-refutation" if refuted else "valid-positive-penalty"
    return {"schema": "pp-queue-attribution-v1", "verdict": verdict, "provenance": provenance,
            "execution_count": len(rows), "physical_pair_count": len(physical),
            "legacy_control_count": sum(row["population"] == "legacy" for row in controls),
            "ideal_control_count": sum(row["population"] == "ideal" for row in controls),
            "physical_configurations": physical, "compatibility_controls": controls,
            "trace_identity_oracles": exact_pairs, "behavioral_relations": relations,
            "behavioral_families": sorted({row["family"] for row in relations}),
            "diagnostics": diagnostics, "fatal_findings": findings,
            "refuted_families": sorted(set(refuted)),
            "traf88_acceptance": "met" if verdict == "valid-positive-penalty" else "not-met",
            "attribution_scope": "R2 queue work and R3 request changes are separate tests; "
            "trigger-packet timelines are not marginal latency contributions",
            "scope": "forward-only singleton request; no TPOT or hardware calibration"}


def execute_job(job, *, root, references, binary, timeout_s):
    return execute(*job, root, references, binary, timeout_s)


def main():
    parser = argparse.ArgumentParser(description="Audit pipeline switch service and receiver holding")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout-s", type=int, default=3600)
    args = parser.parse_args()
    if args.workers < 1 or args.timeout_s < 1:
        parser.error("workers and timeout must be positive")
    if args.out.resolve().is_relative_to(REPO):
        parser.error("bulk --out must be outside the repository")
    required = ("SIMLLM_HTSIM_RNIC", "SIMLLM_HTSIM_SOURCE", "SIMLLM_TXT2BIN")
    if any(not os.getenv(name) for name in required):
        parser.error("configure " + ", ".join(required))
    binary, source, txt2bin = (Path(os.environ[name]).resolve() for name in required)
    if git("status", "--porcelain") or git("status", "--porcelain", cwd=source):
        parser.error("commit source and runner changes before recording execution identities")
    git("merge-base", "--is-ancestor", FREEZE, "HEAD")
    if subprocess.check_output(["git", "show", f"{FREEZE}:examples/pp_rail_contention_v2/expectations.md"],
                               cwd=REPO) != (HERE / "expectations.md").read_bytes():
        parser.error("frozen expectations bytes changed")
    previous.reference_inputs()
    identities = previous.identity_guards()
    locks = reference_locks()
    args.out.mkdir(parents=True, exist_ok=False)
    planned = cells()
    write_json(args.out / "physical_bounds.json", {cell.name: physical_bounds(cell) for cell in planned})
    provenance = {"expectations_commit": FREEZE, "simllm_commit": git("rev-parse", "HEAD"),
                  "htsim_commit": git("rev-parse", "HEAD", cwd=source),
                  "binary_sha256": digest(binary), "txt2bin_sha256": digest(txt2bin),
                  "python_version": platform.python_version(), "platform": platform.system(),
                  "worker_count": args.workers, "wall_timeout_s": args.timeout_s,
                  "runner_sha256": digest(__file__), "trace_auditor_sha256": digest(HERE / "trace_analysis.py"),
                  "expectations_sha256": digest(HERE / "expectations.md"),
                  "reference_result_sha256": digest(HERE.parent / "data_recovery_v1/results.json"),
                  "reference_name": args.reference.name, "default_identity_guards": identities}
    write_json(args.out / "provenance.json", provenance)
    snapshots = args.out / "runner-snapshots"
    snapshots.mkdir()
    for path in (Path(__file__), HERE / "trace_analysis.py", HERE / "expectations.md",
                 Path(previous.__file__), Path(reference.__file__),
                 REPO / "simllm/backends/htsim_rnic.py"):
        shutil.copyfile(path, snapshots / (digest(path) + path.suffix))
    for cell in planned:
        prepare_cell(cell, args.out, args.reference, locks, txt2bin)
    jobs = [(cell, traced) for cell in planned for traced in
            ((False, True) if cell.population == "physical" else (False,))]
    write_json(args.out / "planned_executions.json", [
        {"cell": cell.name, "traced": traced} for cell, traced in jobs])
    execute_one = partial(execute_job, root=args.out, references=args.reference,
                          binary=binary, timeout_s=args.timeout_s)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=prepare_htsim_child_lifetime) as pool:
        rows = list(pool.map(execute_one, jobs))
    summary = summarize(rows, provenance)
    write_json(args.out / "results.json", summary)
    print(f"verdict={summary['verdict']} executions={len(rows)}", flush=True)
    return 2 if summary["verdict"] == "void" else 0


if __name__ == "__main__":
    raise SystemExit(main())
