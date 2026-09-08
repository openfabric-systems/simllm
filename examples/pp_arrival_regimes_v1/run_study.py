"""Execute the frozen early-burst pipeline qualification through the native seam.

Declared stage work changes first communication eligibility. Native completions
remain the only timing authority; observations and request metrics are projections.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import platform
import shutil
import subprocess
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from functools import partial
from itertools import pairwise, product
from pathlib import Path

from examples.pp_rail_contention_v2 import run_study as predecessor
from simllm.backends._child_process import run_owned_process
from simllm.backends.fct import earliest_completion_byte_floors
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
from simllm.core import (
    AdditiveVisitTotals,
    ComputeWork,
    LatencyAttribution,
    RequestMetric,
    StepResult,
    execution_graph_to_json,
)
from simllm.goal import to_binary
from simllm.placement import declared_pipeline_placement, declared_rail_fabric
from simllm.traffic import render_serial_execution_graph_goal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FREEZE = "540bf8ddc89b3c0fe23991fcccf100e3dbd7b49b"
ARRIVALS_PS = (16_000, 80_000)
PENALTY_BAND_PS = (2_400_000_000, 2_800_000_000)
PROBE_WINDOWS_PS = tuple(40_000_000 * 2**power for power in range(6))
HIGH_WINDOW_PS, TICK_PS = predecessor.HIGH_WINDOW_PS, predecessor.TICK_PS
RATE, BUFFER_BYTES = predecessor.RATE, predecessor.BUFFER_BYTES
reference, previous = predecessor.reference, predecessor.previous
write_json, digest = predecessor.write_json, predecessor.digest
require_digest, manifest_value = predecessor.require_digest, predecessor.manifest_value
PAIR_FIELDS = (
    "completion_sha256", "request_ttft_ps", "pp_hop_maximum_ps", "physical_quiescence",
    "physical_quiescence_time_ps", "complete_flow_phase_ps", "ep_phase_ps", "job_completion_ps",
    "request_attribution", "pp_fct_samples_ps", "pp_hops", "native_manifest",
)
REGRESSION_FIELDS = PAIR_FIELDS + ("request_tpot_ps", "normalized_phase_makespan")


@dataclass(frozen=True)
class Cell:
    population: str
    variant: str
    spines: int
    width: int
    ep_width: int
    arrival_ps: int
    profile: str = "rnic-cn"

    @property
    def window_ps(self):
        return HIGH_WINDOW_PS if self.population in {"physical", "regression"} else None

    @property
    def reference_name(self):
        return f"{self.variant}-s{self.spines}-pp{self.width}-ep{self.ep_width}-{self.profile}"

    @property
    def name(self):
        return f"{self.population}-{self.reference_name}-a{self.arrival_ps}"

    def predecessor_cell(self):
        population = "physical" if self.population == "regression" else self.population
        return predecessor.Cell(population, self.variant, self.spines, self.width,
                                self.ep_width, self.window_ps, self.profile)


def cells():
    held_out = list(product(reference.VARIANTS, (4, 8), (0, 32), ARRIVALS_PS))
    physical = [Cell("physical", variant, 2, width, ep, arrival)
                for variant, width, ep, arrival in held_out]
    ideal = [Cell("ideal", variant, 2, width, ep, arrival, "rnic-nn")
             for variant, width, ep, arrival in held_out]
    regressions = [Cell("regression", variant, 2, width, ep, 0)
                   for variant, width, ep in product(reference.VARIANTS, (2, 4, 8), (0, 32))]
    legacy = [Cell("legacy", variant, 8, width, ep, 0)
              for variant, width, ep in product(reference.VARIANTS, (2, 4, 8), (0, 8))]
    return physical + ideal + regressions + legacy


def jobs():
    return [(cell, traced) for cell in cells() for traced in
            ((False, True) if cell.population == "physical" else (False,))]


def build_graph(cell):
    graph = reference.build_graph(cell.width, cell.ep_width)
    computes = [op for op in graph.operations if isinstance(op.work, ComputeWork)]
    if len(computes) != cell.width or computes[0].rank != 0:
        raise ValueError("declared pipeline compute inventory changed")
    first = computes[0]
    shifted = replace(first, work=replace(first.work,
                                         nominal_duration_ps=reference.COMPUTE_PS + cell.arrival_ps))
    return replace(graph, operations=tuple(shifted if op == first else op for op in graph.operations))


def physical_bounds(cell):
    bounds = predecessor.physical_bounds(cell.predecessor_cell())
    bounds["pp_step_data_floor_ps"] += cell.arrival_ps
    bounds["declared_compute_ps"] = cell.width * reference.COMPUTE_PS + cell.arrival_ps
    bounds["first_stage_additional_compute_ps"] = cell.arrival_ps
    bounds["pp_noncompute_share_floor"] = (
        1 - bounds["declared_compute_ps"] / bounds["pp_step_data_floor_ps"])
    bounds["budget_service_ps"] = max(
        max(bounds["receiver_payload_bytes"].values()) * 20,
        bounds["ep_effective_serialization_floor_ps"], bounds["pp_step_data_floor_ps"])
    bounds["engineering_budget_ps"] = (
        9 * bounds["budget_service_ps"] + 8 * (40_000_000 + bounds["budget_queue_allowance_ps"]))
    return bounds


def configuration(cell, inputs, out, traced) -> HtsimRnicConfig:
    if traced and cell.population != "physical":
        raise ValueError("only held-out physical cells have a trace arm")
    return predecessor.configuration(cell.predecessor_cell(), inputs, out, traced)


def prior_locks(prior_root):
    """Bind retained observations to the public result before using any as controls."""
    published = json.loads((HERE.parent / "pp_rail_contention_v2/results.json").read_bytes())
    require_digest(prior_root / "results.json", published["bulk_results_sha256"])
    complete = json.loads((prior_root / "results.json").read_bytes())
    if complete["schema"] != "pp-queue-attribution-v1" or complete["verdict"] != "valid-refutation":
        raise ValueError("prior study is not the published valid refutation")
    full_rows = {row["cell"]: row for row in complete["physical_configurations"]}
    public_rows = {row["cell"]: row for row in published["physical_configurations"]}
    locks = {}
    for cell in cells():
        if cell.population != "regression":
            continue
        name = cell.predecessor_cell().name
        row, public = full_rows[name], public_rows[name]
        for traced in ("trace-off", "trace-on"):
            require_digest(prior_root / name / traced / "completion.csv", public["completion_sha256"])
        audit_path = prior_root / name / "trace-on/trace_audit.json"
        require_digest(audit_path, public["trace_audit_sha256"])
        if json.loads(audit_path.read_bytes()) != row["trace_audit"]:
            raise ValueError("prior full audit disagrees with its locked result")
        if any(row[field] != public[field] for field in REGRESSION_FIELDS):
            raise ValueError("prior public metrics disagree with the locked full result")
        locks[name] = {**{field: public[field] for field in REGRESSION_FIELDS},
                       "prior_trace_audit_sha256": public["trace_audit_sha256"]}
    if len(locks) != 12:
        raise ValueError("prior regression population changed")
    return published, locks


def projection_for(cell):
    return project_declared_clos(declared_rail_fabric(
        declared_pipeline_placement(cell.width), variant=cell.variant, spine_count=cell.spines))


def require_rendered_goal(raw, accepted):
    accepted = [accepted] if isinstance(accepted, str) else accepted
    if hashlib.sha256(raw).hexdigest() not in accepted:
        raise ValueError("zero-offset rendered GOAL differs from the published reference")


def prepare_cell(cell, root, references, locks, txt2bin):
    out = root / cell.name
    out.mkdir()
    write_json(out / "pre_run_bounds.json", physical_bounds(cell))
    placement = declared_pipeline_placement(cell.width)
    fabric = declared_rail_fabric(placement, variant=cell.variant, spine_count=cell.spines)
    projection = project_declared_clos(fabric)
    graph, baseline_graph = build_graph(cell), build_graph(replace(cell, arrival_ps=0))
    physical_name = cell.reference_name.replace("rnic-nn", "rnic-cn")
    retained, lock = references / physical_name, locks[physical_name]
    baseline_trace = render_serial_execution_graph_goal(projection.project_graph(baseline_graph), num_goal_ranks=64)
    raw = baseline_trace.render().encode()
    require_rendered_goal(raw, lock["goal_text_sha256"])
    for name, field in (("step.goal", "goal_text_sha256"), ("step.bin", "goal_sha256"),
                        ("clos.topo", "topology_sha256")):
        require_digest(retained / name, lock[field])
    expected_json = {"placement.json": asdict(placement), "fabric.json": asdict(fabric),
                     "endpoint_by_rank.json": projection.endpoint_by_rank,
                     "semantic_graph.json": execution_graph_to_json(baseline_graph)}
    for name, value in expected_json.items():
        if json.loads(json.dumps(value)) != json.loads((retained / name).read_bytes()):
            raise ValueError(f"unchanged declared reference differs: {name}")
        write_json(out / name, execution_graph_to_json(graph) if name == "semantic_graph.json" else value)
    (out / "clos.topo").write_bytes(projection.topology_text.encode())
    require_digest(out / "clos.topo", lock["topology_sha256"])
    trace = render_serial_execution_graph_goal(projection.project_graph(graph), num_goal_ranks=64)
    (out / "step.goal").write_bytes(trace.render().encode())
    to_binary(out / "step.goal", tool=txt2bin)
    if cell.arrival_ps == 0:
        require_digest(out / "step.goal", lock["goal_text_sha256"])
        require_digest(out / "step.bin", lock["goal_sha256"])
    ideal_path = references / physical_name.replace("rnic-cn", "rnic-nn") / "completion.csv"
    require_digest(ideal_path, lock["ideal_completion_sha256"])
    if cell.population == "legacy":
        require_digest(retained / "completion.csv", lock["reference_completion_sha256"])
    write_json(out / "inputs.json", {
        "cell": asdict(cell), "expectations_commit": FREEZE, "published_reference": lock,
        "transformation": "only first declared compute duration gains arrival_ps",
        "input_sha256": {name: digest(out / name) for name in (
            "step.goal", "step.bin", "clos.topo", *expected_json)},
    })


def shifted_ideal(cell, references):
    path = references / cell.reference_name.replace("rnic-cn", "rnic-nn") / "completion.csv"
    projection = projection_for(cell)
    flows = parse_completion_csv(path)
    return [replace(flow, start_time_ps=flow.start_time_ps + cell.arrival_ps,
                    completion_time_ps=flow.completion_time_ps + cell.arrival_ps)
            if projection.semantic_flow(flow).source % 8 == 0 else flow for flow in flows]


def flow_index(flows):
    indexed = {reference.flow_key(flow): flow for flow in flows}
    if len(indexed) != len(flows) or len({flow.flow_id for flow in flows}) != len(flows):
        raise ValueError("completion identity is duplicated")
    return indexed


def ideal_transform_guard(flows, expected):
    actual, transformed = flow_index(flows), flow_index(expected)
    fields = ("profile", "start_time_ps", "completion_time_ps", "fct_ps")
    return actual.keys() == transformed.keys() and all(
        all(getattr(actual[key], field) == getattr(transformed[key], field) for field in fields)
        for key in transformed)


def exact_goal_timing(pp, cell, findings):
    guard = lambda condition, message: reference.require_guard(condition, message, findings)
    guard(len(pp) == cell.width - 1, "exact GOAL timing lacks the full pipeline chain")
    if not pp:
        return None
    guard(pp[0].start_time_ps == reference.COMPUTE_PS + cell.arrival_ps + 1000,
          "first PP release differs from declared compute plus one calc gate")
    for before, after in pairwise(pp):
        guard(after.start_time_ps == before.completion_time_ps + reference.COMPUTE_PS + 2000,
              "PP release differs from preceding delivery plus compute and two calc gates")
    return ((pp[-1].completion_time_ps + reference.COMPUTE_PS + 1000) // 1000) * 1000


def packet_step_result(record, event, cell, pp):
    compute = cell.width * reference.COMPUTE_PS + cell.arrival_ps
    service = sum(flow.fct_ps for flow in pp)
    overhead = event.timestamp_ps - compute - service
    expected = 2000 * (cell.width - 1) - pp[-1].completion_time_ps % 1000
    if (overhead != expected or len(record.scheduled) != 1 or record.virtual_time_ps != 0):
        raise ValueError("arrival request projection violates the frozen compute and GOAL decomposition")
    attribution = LatencyAttribution(kernel_ps=compute, nic_ps=service, control_ps=overhead)
    request = record.scheduled[0]
    metric = RequestMetric(request.request_id, request.phase, 1, event.timestamp_ps,
                           event.timestamp_ps, event.timestamp_ps, None, attribution, AdditiveVisitTotals())
    return StepResult(record.step_index, event.timestamp_ps, event.timestamp_ps, (metric,))


def empty_row(cell, traced, findings):
    return {"cell": cell.name, **asdict(cell), "window_ps": cell.window_ps, "traced": traced,
            "cell_status": "void", "fatal_findings": findings, "request_ttft_ps": None,
            "pp_hop_maximum_ps": None, "request_tpot_ps": None, "trace_audit": None,
            "completion_sha256": None}


def verify_inputs(inputs, cell, manifest_sha256):
    if digest(inputs / "inputs.json") != manifest_sha256:
        raise ValueError("frozen input manifest changed")
    record = json.loads((inputs / "inputs.json").read_bytes())
    expected = {"step.goal", "step.bin", "clos.topo", "fabric.json", "placement.json",
                "semantic_graph.json", "endpoint_by_rank.json"}
    if (record["cell"] != asdict(cell) or record["expectations_commit"] != FREEZE
            or record["input_sha256"].keys() != expected):
        raise ValueError("frozen input inventory changed")
    for name, expected_digest in record["input_sha256"].items():
        if digest(inputs / name) != expected_digest:
            raise ValueError(f"frozen input bytes changed: {name}")


def analyze_execution(cell, out, references, regression_locks, execution, stdout, traced):
    row = empty_row(cell, traced, [])
    if execution["returncode"] != 0:
        row["fatal_findings"].append(f"native execution exited {execution['returncode']}")
        return row
    flows = parse_completion_csv(out / "completion.csv")
    flow_index(flows)
    manifest = [line for line in stdout.splitlines() if line.startswith("[RNIC manifest]")]
    run = RnicRunResult(flows, manifest, manifest_value(manifest, "physical_quiescence") == "verified",
                        _parse_goal_completion_time_ps(stdout), parse_control_recovery_manifest(manifest),
                        parse_data_recovery_manifest(manifest))
    projection, graph, bounds = projection_for(cell), build_graph(cell), physical_bounds(cell)
    semantic = [projection.semantic_flow(flow) for flow in flows]
    trace = render_serial_execution_graph_goal(graph, num_goal_ranks=64)
    checked, pp = reference.evaluate_cell(cell.variant, cell.width, cell.ep_width, cell.profile,
                                           trace, semantic, run, bounds)
    findings = row["fatal_findings"] = checked["fatal_findings"]
    guard = lambda condition, message: reference.require_guard(condition, message, findings)
    final_time = exact_goal_timing(pp, cell, findings)
    ideal = shifted_ideal(cell, references)
    whole_phase = predecessor.phase(flows)
    ep = [flow for flow in semantic if flow.source % 8 != 0]
    ep_phase = predecessor.phase(ep) if ep else 0
    guard(whole_phase <= bounds["engineering_budget_ps"], "complete-flow phase exceeds frozen budget")
    guard(all(type(value) is int and value >= 0 for flow in flows for value in
              (flow.start_time_ps, flow.completion_time_ps, flow.fct_ps)), "invalid flow time")
    if cell.profile == "rnic-cn":
        guard(manifest_value(manifest, "global_seed") == "1", "native seed changed")
        guard(ep_phase >= bounds["ep_phase_floor_ps"], "EP phase beats directional capacity floor")
        guard(whole_phase >= predecessor.phase(ideal), "physical complete phase beats identical ideal phase")
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
        predecessor.normalized_rows(flows, ideal, out)
    if cell.window_ps is not None:
        for key, expected in (("ring_window_ps", str(cell.window_ps)), ("ring_tick_ps", str(TICK_PS)),
                              ("rnic_cn_data_recovery", "exponential"),
                              ("rnic_cn_control_recovery", "headroom"), ("rnic_cn_initial_window", "none"),
                              ("rnic_cn_retry_probe_windows", "4")):
            guard(manifest_value(manifest, key) == expected, f"fixed selection changed: {key}")
        guard(int(manifest_value(manifest, "rnic_cn_maximum_retry_attempt_observed")) <= 8,
              "retry limit exceeded")
    row["completion_sha256"] = digest(out / "completion.csv")
    if cell.population in {"ideal", "legacy"}:
        if cell.population == "ideal":
            matches = ideal_transform_guard(flows, ideal)
            oracle = {"family": "ideal-declared-compute-shift", "arrival_ps": cell.arrival_ps,
                      "complete_unique_flow_count": len(flows)}
        else:
            expected = predecessor.reference_locks()[cell.reference_name]["reference_completion_sha256"]
            matches = row["completion_sha256"] == expected
            oracle = {"family": "legacy-completion-csv", "expected_sha256": expected}
        guard(matches, "protected completion control changed")
        row["exact_oracle"] = {**oracle, "matches": matches}
    if traced:
        from examples.pp_rail_contention_v2.trace_analysis import audit_trace

        audit = audit_trace(out / "trace", completion_rows=[asdict(flow) for flow in flows],
                            endpoint_by_rank=projection.endpoint_by_rank, spine_count=cell.spines,
                            delta_ps=cell.window_ps, tick_ps=TICK_PS, capture_drop_occupancy=True)
        write_json(out / "trace_audit.json", audit)
        predecessor.trace_boundary_guard(audit, manifest)
        row["trace_audit"] = audit
    row.update(physical_quiescence=run.quiescent,
               physical_quiescence_time_ps=int(manifest_value(manifest, "physical_quiescence_time_ps")),
               bounds=bounds, complete_flow_phase_ps=whole_phase, ep_phase_ps=ep_phase,
               ideal_phase_ps=predecessor.phase(ideal), normalized_phase_makespan=whole_phase / predecessor.phase(ideal),
               job_completion_ps=run.job_completion_time_ps(),
               native_manifest=[" ".join(token for token in line.split()
                                if not token.startswith(("goal=", "completion_csv=", "topology=")))
                                for line in manifest])
    if not findings:
        event = previous.final_stage_event(graph, pp)
        result = packet_step_result(reference.step_record(), event, cell, pp)
        guard(event.timestamp_ps == final_time, "request event differs from exact GOAL timing")
        if cell.profile == "rnic-cn":
            guard(event.timestamp_ps >= bounds["pp_step_data_floor_ps"],
                  "request beats compute and physical floor")
        guard(result.request_metrics[0].attribution.total_ps == result.step_latency_ps,
              "request attribution does not conserve")
        row.update(request_ttft_ps=result.request_metrics[0].ttft_ps,
                   pp_hop_maximum_ps=max(flow.fct_ps for flow in pp),
                   pp_fct_samples_ps=[flow.fct_ps for flow in pp], pp_hops=checked["pp_hops"],
                   request_attribution=asdict(result.request_metrics[0].attribution))
        if cell.population == "regression":
            lock = regression_locks[cell.predecessor_cell().name]
            differences = [field for field in REGRESSION_FIELDS if row[field] != lock[field]]
            guard(not differences, "fixed-window regression changed " + ", ".join(differences))
            row["exact_oracle"] = {"family": "v2-csv-and-request-regression", "mismatched_fields": differences,
                                   "matches": not differences, "prior_cell": cell.predecessor_cell().name,
                                   "prior_trace_audit_sha256": lock["prior_trace_audit_sha256"]}
        if not findings:
            write_json(out / "final_stage_event.json", asdict(event))
            write_json(out / "step_result.json", asdict(result))
            row["cell_status"] = "valid"
    if findings:
        row.update(request_ttft_ps=None, pp_hop_maximum_ps=None)
    return row


def execute(job, *, root, references, regression_locks, input_locks, binary, timeout_s):
    cell, traced = job
    inputs = root / cell.name
    out = inputs / ("trace-on" if traced else "trace-off")
    out.mkdir()
    command = build_htsim_rnic_command(binary, configuration(cell, inputs, out, traced))
    write_json(out / "command.json", {"argv": command})
    try:
        verify_inputs(inputs, cell, input_locks[cell.name])
        native = run_owned_process(command, timeout_s=timeout_s)
        stdout, returncode = native.stdout + "\n" + native.stderr, native.returncode
    except subprocess.TimeoutExpired as error:
        decode = lambda value: value.decode(errors="replace") if isinstance(value, bytes) else value or ""
        stdout, returncode = decode(error.output) + "\n" + decode(error.stderr), 124
    except (OSError, ValueError, KeyError, TypeError) as error:
        stdout, returncode = f"{type(error).__name__}: {error}\n", 125
    (out / "run.log").write_bytes(stdout.encode())
    execution = {"returncode": returncode, "run_log_sha256": digest(out / "run.log"),
                 "completion_sha256": digest(out / "completion.csv") if (out / "completion.csv").exists() else None}
    write_json(out / "execution.json", execution)
    try:
        verify_inputs(inputs, cell, input_locks[cell.name])
        row = analyze_execution(cell, out, references, regression_locks, execution, stdout, traced)
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as error:
        (out / "audit_error.txt").write_text(f"{type(error).__name__}: {error}\n")
        row = empty_row(cell, traced, [f"{type(error).__name__}: {error}"])
        row["completion_sha256"] = execution["completion_sha256"]
    write_json(out / "cell.json", row)
    print(f"{cell.name} trace={int(traced)} {row['cell_status']} ttft_ps={row['request_ttft_ps']}", flush=True)
    return row


def retry_mechanism(row):
    flow = max(row["trace_audit"]["pp_flows"], key=lambda value: value["fct_ps"])
    trigger = next(packet for packet in flow["packets"]
                   if packet["lifecycle_id"] == flow["completion_trigger_lifecycle_id"])
    attempts = [packet for packet in flow["packets"] if packet["packet_index"] == trigger["packet_index"]]
    by_attempt = {packet["attempt"]: packet for packet in attempts}
    if len(by_attempt) != len(attempts):
        raise ValueError("duplicate logical retry attempt in valid audit")
    shape = (trigger["attempt"] == 7 and trigger["terminal"] == "endpoint_consumed"
             and set(range(8)) <= by_attempt.keys())
    if any(packet["attempt"] > 0 and packet["retry_authorization"] is None for packet in attempts):
        raise ValueError("transmitted retry lacks authorization observation")
    observations, intervals = [], []
    for number in range(7):
        packet = by_attempt.get(number)
        if packet is None:
            shape = False
            continue
        dropped = packet["terminal"] == "fabric_drop"
        snapshot = packet.get("drop_occupancy")
        if dropped and snapshot is None:
            raise ValueError("fabric-dropped critical attempt lacks occupancy observation")
        ep_bytes = snapshot["shared_partition_bytes"]["ep_data"] if snapshot else 0
        sufficient = snapshot["admits_without_ep_data"] if snapshot else False
        observations.append({"attempt": number, "lifecycle_id": packet["lifecycle_id"],
                             "terminal": packet["terminal"], "drop_occupancy": snapshot})
        shape = shape and dropped and ep_bytes > 0 and sufficient
    first = by_attempt.get(1)
    first_authorization = first["retry_authorization"] if first else None
    shape = shape and first_authorization is not None and first_authorization["cause"] == "gap_nack"
    for number, expected in zip(range(2, 8), PROBE_WINDOWS_PS):
        packet, before = by_attempt.get(number), by_attempt.get(number - 1)
        if packet is None or before is None:
            shape = False
            continue
        authorization = packet["retry_authorization"]
        if authorization is None:
            raise ValueError("transmitted retry lacks authorization observation")
        interval = authorization["time_ps"] - before["source_end_ps"]
        match = (authorization["cause"] == "probe_timeout" and interval == expected
                 and authorization["deadline_ps"] == authorization["time_ps"]
                 and authorization["origin_attempt"] == number - 1)
        intervals.append({"attempt": number, "expected_interval_ps": expected, "interval_ps": interval,
                          "authorization": authorization,
                          "dispatch_offset_ps": packet["source_start_ps"] - authorization["time_ps"]})
        shape = shape and match
    return {"slowest_hop": {field: flow[field] for field in (
                "flow_id", "source_rank", "destination_rank", "fct_ps", "completion_trigger_lifecycle_id")},
            "trigger_packet": {key: value for key, value in trigger.items() if key != "visits"},
            "drop_attempts": observations, "first_retry_authorization": first_authorization,
            "first_retry_dispatch_offset_ps": None if first_authorization is None else
            first["source_start_ps"] - first_authorization["time_ps"],
            "probe_intervals": intervals, "expected_probe_window_sum_ps": sum(PROBE_WINDOWS_PS),
            "trigger_packet_timeline": flow["trigger_packet_timeline"],
            "completion_dependency_witness": flow["completion_dependency_witness"]}, bool(shape)


def rail_identity(unloaded, loaded):
    fields = ("source_start_ps", "source_end_ps", "eta_ps", "arrival_ps", "logical_release_ps",
              "rx_service_start_ps", "rx_service_end_ps", "delivery_ps")

    def originals(row):
        packets = [(flow["source_rank"], flow["destination_rank"], packet)
                   for flow in row["trace_audit"]["pp_flows"] for packet in flow["packets"]
                   if packet["attempt"] == 0]
        indexed = {(source, destination, packet["packet_index"]): tuple(packet[field] for field in fields)
                   for source, destination, packet in packets}
        if len(indexed) != len(packets) or len(packets) != (row["width"] - 1) * 16:
            raise ValueError("rail control lacks the complete unique original PP inventory")
        return indexed

    left, right = originals(unloaded), originals(loaded)
    same = left == right and unloaded["pp_hops"] == loaded["pp_hops"]
    no_ep = all(row["trace_audit"]["queue_work"]["ep_data_service_ahead_ps"] == 0
                for row in (unloaded, loaded))
    return {"original_packet_count": len(left), "packet_fields": fields,
            "all_original_times_exact": left == right, "pp_flow_times_exact": unloaded["pp_hops"] == loaded["pp_hops"],
            "zero_ep_service_ahead": no_ep, "matches": same and no_ep}


def behavioral_relations(physical):
    indexed = {(row["variant"], row["width"], row["ep_width"], row["arrival_ps"]): row for row in physical}
    relations, rail_oracles, findings = [], [], []
    for width, arrival in product((4, 8), ARRIVALS_PS):
        unloaded, loaded = [indexed[("node-local", width, ep, arrival)] for ep in (0, 32)]
        valid = all(row["cell_status"] == "valid" for row in (unloaded, loaded))
        values = []
        if valid:
            hop_delta = loaded["pp_hop_maximum_ps"] - unloaded["pp_hop_maximum_ps"]
            ttft_delta = loaded["request_ttft_ps"] - unloaded["request_ttft_ps"]
            values.append(({"hop_maximum_delta_ps": hop_delta, "request_ttft_delta_ps": ttft_delta,
                            "acceptance_band_ps": PENALTY_BAND_PS}, all(
                                PENALTY_BAND_PS[0] <= value <= PENALTY_BAND_PS[1]
                                for value in (hop_delta, ttft_delta))))
            visits = [{"flow_id": flow["flow_id"], "source_rank": flow["source_rank"],
                       "destination_rank": flow["destination_rank"], "lifecycle_id": packet["lifecycle_id"],
                       "packet_index": packet["packet_index"], "attempt": packet["attempt"], "visit": visit}
                      for flow in loaded["trace_audit"]["pp_flows"] for packet in flow["packets"]
                      for visit in packet["visits"]]
            witnesses = [item for item in visits if item["visit"]["service_ahead_ps"]["ep_data"] > 0
                         and item["visit"]["ep_data_bound_ps"] > 0]
            values.append(({"positive_independent_visit_count": len(witnesses),
                            "ep_data_service_ahead_work_ps": loaded["trace_audit"]["queue_work"]["ep_data_service_ahead_ps"],
                            "strictly_earlier_ep_bound_work_ps": sum(item["visit"]["ep_data_bound_ps"] for item in visits),
                            "first_positive_visit": witnesses[0] if witnesses else None}, bool(witnesses)))
            try:
                values.append(retry_mechanism(loaded))
            except (ValueError, KeyError, TypeError, StopIteration) as error:
                findings.append({"cell": loaded["cell"], "findings": [f"retry witness is incomplete: {error}"]})
                values.append((None, None))
        for index, family in enumerate(("B1-visible-penalty", "B2-independent-contention", "B3-retry-mechanism")):
            details, matched = values[index] if valid else (None, None)
            relations.append({"family": family, "width": width, "arrival_ps": arrival,
                              "cells": [unloaded["cell"], loaded["cell"]], "details": details,
                              "matched": matched, "status": ("holds" if matched else "refuted")
                              if matched is not None else "uninterpretable"})
        pair = [indexed[("rail", width, ep, arrival)] for ep in (0, 32)]
        valid = all(row["cell_status"] == "valid" for row in pair)
        oracle = rail_identity(*pair) if valid else {"matches": False}
        rail_oracles.append({"family": "rail-pp-load-identity", "width": width, "arrival_ps": arrival,
                             "cells": [row["cell"] for row in pair], **oracle,
                             "status": "exact" if valid and oracle["matches"] else "void"})
        if valid and not oracle["matches"]:
            findings.append({"cells": [row["cell"] for row in pair], "findings": ["rail off-path identity changed"]})
    return relations, rail_oracles, findings


def summarize(rows, provenance):
    expected = {(cell.name, traced): cell for cell, traced in jobs()}
    keys = [(row["cell"], row["traced"]) for row in rows]
    if len(set(keys)) != len(keys) or set(keys) != expected.keys():
        raise ValueError("execution population is missing, duplicated or outside the freeze")
    for key, row in zip(keys, rows):
        if any(row.get(field) != value for field, value in asdict(expected[key]).items()):
            raise ValueError("execution metadata differs from the frozen cell")
    indexed = dict(zip(keys, rows))
    physical, controls, exact_pairs, findings = [], [], [], []
    for cell in cells():
        off = indexed[(cell.name, False)]
        if cell.population != "physical":
            controls.append(off)
            if off["cell_status"] != "valid" or off["fatal_findings"]:
                findings.append({"cell": cell.name, "findings": off["fatal_findings"] or ["invalid control"]})
            continue
        on = indexed[(cell.name, True)]
        row = dict(on)
        fatal = list(off["fatal_findings"]) + list(on["fatal_findings"])
        valid = off["cell_status"] == on["cell_status"] == "valid"
        differences = [field for field in PAIR_FIELDS if field not in off or field not in on
                       or off[field] != on[field]] if valid else []
        if differences:
            fatal.append("trace selection changed " + ", ".join(differences))
        if not valid and not fatal:
            fatal.append("trace pair lacks two valid executions")
        exact_pairs.append({"cell": cell.name, "family": "trace-selection-identity",
                            "status": "exact" if valid and not fatal else "void",
                            "mismatched_fields": differences, "completion_sha256": off.get("completion_sha256")})
        if fatal:
            findings.append({"cell": cell.name, "findings": fatal})
            row.update(cell_status="void", fatal_findings=fatal, request_ttft_ps=None,
                       pp_hop_maximum_ps=None, trace_audit=None)
        physical.append(row)
    try:
        relations, rail_oracles, rail_findings = behavioral_relations(physical)
    except (ValueError, KeyError, TypeError, StopIteration) as error:
        findings.append({"findings": [f"required behavioral observation is incomplete: {error}"]})
        uninterpretable = [{**row, "cell_status": "void"} for row in physical]
        relations, rail_oracles, rail_findings = behavioral_relations(uninterpretable)
    findings.extend(rail_findings)
    refuted = sorted({row["family"] for row in relations if row["status"] == "refuted"})
    verdict = "void" if findings else "valid-refutation" if refuted else "valid-qualified-penalty"
    return {"schema": "pp-arrival-regimes-v1", "verdict": verdict, "provenance": provenance,
            "execution_count": len(rows), "physical_pair_count": len(physical),
            "control_counts": dict(Counter(row["population"] for row in controls)),
            "physical_configurations": physical, "compatibility_controls": controls,
            "trace_identity_oracles": exact_pairs, "rail_identity_oracles": rail_oracles,
            "behavioral_relations": relations, "behavioral_families": sorted({row["family"] for row in relations}),
            "fatal_findings": findings, "refuted_families": refuted,
            "traf88_acceptance": "met" if verdict == "valid-qualified-penalty" else "not-met",
            "attribution_scope": "declared compute, measured hop FCT and exact GOAL gates conserve TTFT; "
            "queue work and dispatch offsets are not marginal latency contributions",
            "scope": "finite seed-one early-burst forward prefill; no TPOT, captured serving or hardware calibration"}


def frozen_sources(binary, source, txt2bin, published):
    if predecessor.git("status", "--porcelain") or predecessor.git("status", "--porcelain", cwd=source):
        raise ValueError("commit source and runner changes before recording execution identities")
    predecessor.git("merge-base", "--is-ancestor", FREEZE, "HEAD")
    for path in (HERE / "expectations.md", HERE.parent / "pp_rail_contention_v2/results.json",
                 HERE.parent / "data_recovery_v1/results.json"):
        frozen = subprocess.check_output(["git", "show", f"{FREEZE}:{path.relative_to(REPO).as_posix()}"], cwd=REPO)
        if path.read_bytes() != frozen:
            raise ValueError("frozen expectations or published reference bytes changed")
    for path, field in ((binary, "binary_sha256"), (txt2bin, "txt2bin_sha256")):
        if digest(path) != published["provenance"][field]:
            raise ValueError(f"frozen executable changed: {field}")
    native_commit = published["provenance"]["htsim_commit"]
    predecessor.git("merge-base", "--is-ancestor", native_commit, "HEAD", cwd=source)
    if predecessor.git("diff", "--name-only", native_commit, "HEAD", "--", "htsim/sim", "htsim/CMakeLists.txt",
                       "CMakeLists.txt", cwd=source):
        raise ValueError("native source differs from the frozen predecessor implementation")


def main():
    parser = argparse.ArgumentParser(description="Qualify the frozen early-burst pipeline penalty")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout-s", type=int, default=3600)
    args = parser.parse_args()
    if args.workers < 1 or args.timeout_s < 1 or args.out.resolve().is_relative_to(REPO):
        parser.error("positive workers/timeout and an external bulk --out are required")
    required = ("SIMLLM_HTSIM_RNIC", "SIMLLM_HTSIM_SOURCE", "SIMLLM_TXT2BIN")
    if any(not os.getenv(name) for name in required):
        parser.error("configure " + ", ".join(required))
    binary, source, txt2bin = (Path(os.environ[name]).resolve() for name in required)
    published, regression_locks = prior_locks(args.prior)
    frozen_sources(binary, source, txt2bin, published)
    previous.reference_inputs()
    identities, locks = previous.identity_guards(), predecessor.reference_locks()
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "physical_bounds.json", {cell.name: physical_bounds(cell) for cell in cells()})
    auditor = HERE.parent / "pp_rail_contention_v2/trace_analysis.py"
    provenance = {"expectations_commit": FREEZE, "simllm_commit": predecessor.git("rev-parse", "HEAD"),
                  "htsim_commit": predecessor.git("rev-parse", "HEAD", cwd=source),
                  "binary_sha256": digest(binary), "txt2bin_sha256": digest(txt2bin),
                  "python_version": platform.python_version(), "platform": platform.system(),
                  "worker_count": args.workers, "wall_timeout_s": args.timeout_s,
                  "runner_sha256": digest(__file__), "trace_auditor_sha256": digest(auditor),
                  "expectations_sha256": digest(HERE / "expectations.md"),
                  "reference_result_sha256": digest(HERE.parent / "data_recovery_v1/results.json"),
                  "prior_public_result_sha256": digest(HERE.parent / "pp_rail_contention_v2/results.json"),
                  "prior_bulk_result_sha256": published["bulk_results_sha256"],
                  "prior_expectations_commit": published["provenance"]["expectations_commit"],
                  "prior_trace_audits": {name: lock["prior_trace_audit_sha256"] for name, lock in regression_locks.items()},
                  "default_identity_guards": identities}
    write_json(args.out / "provenance.json", provenance)
    snapshots = args.out / "runner-snapshots"
    snapshots.mkdir()
    for path in (Path(__file__), auditor, HERE / "expectations.md", Path(predecessor.__file__),
                 Path(previous.__file__), Path(reference.__file__), REPO / "simllm/backends/htsim_rnic.py"):
        shutil.copyfile(path, snapshots / (digest(path) + path.suffix))
    for cell in cells():
        prepare_cell(cell, args.out, args.reference, locks, txt2bin)
    input_locks = {cell.name: digest(args.out / cell.name / "inputs.json") for cell in cells()}
    provenance["input_manifest_sha256"] = input_locks
    write_json(args.out / "provenance.json", provenance)
    write_json(args.out / "planned_executions.json", [{"cell": cell.name, "traced": traced} for cell, traced in jobs()])
    execute_one = partial(execute, root=args.out, references=args.reference, regression_locks=regression_locks,
                          input_locks=input_locks, binary=binary, timeout_s=args.timeout_s)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=prepare_htsim_child_lifetime) as pool:
        rows = list(pool.map(execute_one, jobs()))
    summary = summarize(rows, provenance)
    write_json(args.out / "results.json", summary)
    print(f"verdict={summary['verdict']} executions={len(rows)}", flush=True)
    return 2 if summary["verdict"] == "void" else 0


if __name__ == "__main__":
    raise SystemExit(main())
