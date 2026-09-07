"""Run the frozen PP activation and rail-topology packet study.

Set SIMLLM_HTSIM_RNIC, SIMLLM_TXT2BIN and SIMLLM_DATA_ROOT, or pass --out.
Bulk programs and completion rows live below --out; --summary selects the
small portable JSON artifact. --plot-only redraws the figure from that JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from collections import Counter
from dataclasses import asdict, replace
from itertools import pairwise, product
from pathlib import Path

from simllm.backends import HtsimRnicConfig, run_htsim_rnic
from simllm.backends.fct import normalized_fct
from simllm.backends.htsim_rnic import FlowCompletion, find_htsim_rnic
from simllm.backends.rail_topology import project_declared_clos
from simllm.compute import ModelDims
from simllm.core import (
    CoarseDeviceRuntime,
    CollectiveWork,
    CompletionReducer,
    ComputeWork,
    ExecutionGraph,
    ExecutionOperation,
    OperationCorrelation,
    RequestPhase,
    ScheduledRequest,
    StepRecord,
    VirtualClock,
    execution_graph_to_json,
)
from simllm.goal import find_txt2bin, to_binary
from simllm.placement import declared_pipeline_placement, declared_rail_fabric
from simllm.traffic import (
    compose_pipeline_graph,
    plan_execution_graph_collectives,
    render_serial_execution_graph_goal,
)

EXPECTATIONS_COMMIT = "8356fa6cf4fe10f77757227eb59856830084adff"
VARIANTS = ("rail", "node-local")
WIDTHS = (2, 4, 8)
EP_WIDTHS = (0, 8, 32)
PROFILES = ("rnic-nn", "rnic-cn")
B = 65_536
EP_BYTES = 1_048_576
RATE = 400_000_000_000
COMPUTE_PS = 1_000_000
LINK_DELAY_PS = 1_000_000
RAIL_ORACLE_PS = 3_310_720
SEED = 1
HERE = Path(__file__).resolve().parent


def dimensions() -> ModelDims:
    return ModelDims(1, 4096, 16384, 32, 8, 128, 32000)


def step_record() -> StepRecord:
    return StepRecord(0, 0, [ScheduledRequest("pp-request", RequestPhase.PREFILL, 8,
                                             context_length=8)])


def build_pp_graph(width: int) -> ExecutionGraph:
    record = step_record()
    placement = declared_pipeline_placement(width)
    ranks = tuple(placement.group_ranks(0, "pp"))
    graphs = []
    for rank in ranks:
        compute = ExecutionOperation(
            "compute", rank, f"compute:{rank}",
            ComputeWork(kernel="declared-stage", nominal_duration_ps=COMPUTE_PS),
            correlation=OperationCorrelation(request_ids=("pp-request",)),
        )
        graphs.append(ExecutionGraph("step-0", 0, 0, (compute,), ("compute",)))
    return compose_pipeline_graph(record, dimensions(), width, ranks, graphs)


def ep_pairs(width: int) -> tuple[tuple[int, int, int], ...]:
    if width not in EP_WIDTHS:
        raise ValueError("EP width must be 0, 8 or 32")
    ranks = tuple(8 * node + nic for nic in range(1, 5) for node in range(8))[:width]
    return tuple(sorted((source, destination, EP_BYTES)
                        for source in ranks for destination in ranks
                        if source // 8 != destination // 8))


def build_graph(width: int, ep_width: int) -> ExecutionGraph:
    graph = build_pp_graph(width)
    pairs = ep_pairs(ep_width)
    if not pairs:
        return graph
    ranks = tuple(sorted({rank for source, destination, _ in pairs for rank in (source, destination)}))
    background = ExecutionOperation(
        "background-ep", ranks[0], "background:ep",
        CollectiveWork("all-to-allv", ranks, 0, "pairwise", "ep-contention",
                       pair_payload_bytes=pairs),
    )
    return plan_execution_graph_collectives(replace(
        graph, operations=(background, *graph.operations), collective_plans=(),
    ))


def quantile(values: list[int], percentile: float) -> int:
    if not values or not 0 < percentile <= 1:
        raise ValueError("nearest-rank quantiles need samples and 0 < percentile <= 1")
    return sorted(values)[math.ceil(percentile * len(values)) - 1]


def bounds(width: int, ep_width: int, variant: str) -> dict:
    hops = 2 if variant == "rail" else 4
    serialization = B * 8 * 10**12 // RATE
    propagation = hops * LINK_DELAY_PS
    payload_budget = len(ep_pairs(ep_width)) * EP_BYTES + (width - 1) * B
    queue_budget = 0 if variant == "rail" else hops * payload_budget * 20
    return {
        "physical_link_count": hops,
        "serialization_floor_ps": serialization,
        "propagation_floor_ps": propagation,
        "data_arrival_floor_ps": serialization + propagation,
        "unloaded_data_store_forward_ceiling_ps": hops * serialization + propagation,
        "payload_only_queue_envelope_ps": queue_budget,
        "payload_only_data_envelope_ps": hops * serialization + propagation + queue_budget,
        "pp_step_data_floor_ps": width * COMPUTE_PS + (width - 1) * (serialization + propagation),
        "control_and_ack_included_in_ceiling": False,
    }


def flow_key(flow: FlowCompletion) -> tuple:
    return flow.source, flow.destination, flow.tag, flow.payload_bytes


def require_guard(condition: bool, description: str, findings: list[str]) -> None:
    if not condition:
        findings.append(description)


def evaluate_cell(variant, width, ep_width, profile, trace, flows, run, cell_bounds):
    findings: list[str] = []
    expected = Counter((message.source_rank, message.destination_rank, message.tag,
                        message.payload_bytes) for message in trace.messages)
    actual = Counter(flow_key(flow) for flow in flows)
    require_guard(expected == actual, "completion inventory differs from semantic GOAL", findings)
    require_guard(len({flow.flow_id for flow in flows}) == len(flows), "duplicate flow ID", findings)
    require_guard(run.quiescent, "backend did not verify physical quiescence", findings)
    require_guard(all(flow.completion_time_ps - flow.start_time_ps == flow.fct_ps
                      and flow.fct_ps >= flow.payload_bytes * 20 for flow in flows),
                  "flow time identity or serialization floor violated", findings)
    pp = sorted((flow for flow in flows if flow.source % 8 == flow.destination % 8 == 0),
                key=lambda flow: flow.source)
    ep = [flow for flow in flows if flow not in pp]
    require_guard(len(pp) == width - 1 and sum(flow.payload_bytes for flow in pp) == (width - 1) * B,
                  "PP count or bytes differ from frozen inventory", findings)
    require_guard(len(ep) == len(ep_pairs(ep_width)), "EP remote count differs", findings)
    require_guard(all(flow.source + 8 == flow.destination for flow in pp), "nonforward PP hop", findings)
    require_guard(bool(pp) and pp[0].start_time_ps >= COMPUTE_PS,
                  "first PP hop starts before stage compute", findings)
    for previous, following in pairwise(pp):
        require_guard(following.start_time_ps + 1000 >= previous.completion_time_ps + COMPUTE_PS,
                      "next stage starts before activation and compute", findings)
    if profile == "rnic-cn":
        require_guard(all(flow.fct_ps >= cell_bounds["data_arrival_floor_ps"] for flow in pp),
                      "physical hop floor violated", findings)
    if not pp:
        return {"variant": variant, "pp_width": width, "ep_width": ep_width,
                "profile": profile, "fatal_findings": findings}, pp

    # Match the existing backend's max(calc_ns, 1) convention. The last
    # receive is followed by its collective frontier calc(0), then stage
    # compute. This reduction was corrected after the initial run and is a
    # post-specified regression check, not an additional frozen oracle.
    final_stage_projection = pp[-1].completion_time_ps // 1000 * 1000 + 1000 + COMPUTE_PS
    if ep_width == 0:
        require_guard(final_stage_projection == run.job_completion_time_ps(),
                      "post-specified final-stage projection differs from isolated GOAL completion",
                      findings)
    times = [flow.fct_ps for flow in pp]
    return {
        "variant": variant, "pp_width": width, "ep_width": ep_width, "profile": profile,
        "topology_timing_authority": "physical-clos" if profile == "rnic-cn" else "null-network",
        "pp_flow_count": len(pp), "pp_bytes": sum(flow.payload_bytes for flow in pp),
        "ep_remote_flow_count": len(ep), "ep_remote_peers_per_rank": len(ep) // ep_width if ep_width else 0,
        "pp_fct_p50_ps": quantile(times, 0.5), "pp_fct_p99_ps": quantile(times, 0.99),
        "pp_hops": [{key: value for key, value in asdict(flow).items() if key in
                     {"source", "destination", "tag", "payload_bytes", "start_time_ps",
                      "completion_time_ps", "fct_ps"}} for flow in pp],
        "tp_phase_makespan_ps": 0,
        "ep_phase_makespan_ps": max((flow.completion_time_ps for flow in ep), default=0),
        "pp_last_transfer_completed_ps": pp[-1].completion_time_ps,
        "pp_final_stage_projection_ps": final_stage_projection,
        "pp_communication_projection_share": (final_stage_projection - width * COMPUTE_PS) / final_stage_projection,
        "pp_flow_service_sum_ps": sum(times),
        "pp_goal_gate_and_quantization_ps": final_stage_projection - width * COMPUTE_PS - sum(times),
        "job_completion_ps": run.job_completion_time_ps(),
        "isolated_goal_projection_delta_ps": (run.job_completion_time_ps() - final_stage_projection)
        if ep_width == 0 else None,
        "exact_rail_oracle": all(value == RAIL_ORACLE_PS for value in times) if variant == "rail" else None,
        "fatal_findings": findings,
        "bounds": cell_bounds,
    }, pp


def run_cell(out: Path, variant: str, width: int, ep_width: int, profile: str):
    name = f"{variant}-pp{width}-ep{ep_width}-{profile}"
    workdir = out / name
    workdir.mkdir(parents=True, exist_ok=True)
    cell_bounds = bounds(width, ep_width, variant)
    write_json(workdir / "pre_run_bounds.json", cell_bounds)
    placement = declared_pipeline_placement(width)
    fabric = declared_rail_fabric(placement, variant=variant)
    placement.save(workdir / "placement.json")
    fabric.save(workdir / "fabric.json")
    projection = project_declared_clos(fabric)
    topology = workdir / "clos.topo"
    topology.write_text(projection.topology_text)
    graph = build_graph(width, ep_width)
    write_json(workdir / "semantic_graph.json", execution_graph_to_json(graph))
    trace = render_serial_execution_graph_goal(graph, num_goal_ranks=64)
    projected = projection.project_graph(graph)
    physical_trace = render_serial_execution_graph_goal(projected, num_goal_ranks=64)
    write_json(workdir / "endpoint_by_rank.json", projection.endpoint_by_rank)
    goal_path = physical_trace.write(workdir / "step.goal")
    binary = to_binary(goal_path)
    run = run_htsim_rnic(HtsimRnicConfig(
        goal_bin=binary, profile=profile, linkspeed_bps=RATE,
        completion_csv=workdir / "completion.csv",
        topology=topology if profile == "rnic-cn" else None,
        extra_flags={"-rnic_cn_prbs_seed": str(SEED)} if profile == "rnic-cn" else {},
    ), timeout_s=120)
    (workdir / "manifest.log").write_text("\n".join(run.manifest) + "\n")
    flows = [projection.semantic_flow(flow) for flow in run.flows]
    write_json(workdir / "semantic_flows.json", [asdict(flow) for flow in flows])
    row, pp = evaluate_cell(variant, width, ep_width, profile, trace, flows, run, cell_bounds)
    row["goal_sha256"] = digest(goal_path)
    row["manifest"] = [
        " ".join(token for token in line.split()
                 if not token.startswith(("goal=", "completion_csv=", "topology=")))
        for line in run.manifest
    ]
    row["physical_quiescence"] = run.quiescent
    write_json(workdir / "cell.json", row)
    return row, pp


def coarse_metric_check(width: int) -> dict:
    graph = build_pp_graph(width)
    record = step_record()
    runtime = CoarseDeviceRuntime()
    execution = runtime.execute(graph)
    report = runtime.last_report
    if report is None:
        raise AssertionError("missing runtime report")
    result = CompletionReducer(VirtualClock()).reduce(record, graph, execution, report)
    expected = width * COMPUTE_PS + (width - 1) * B * 20
    return {
        "pp_width": width,
        "authority": report.authority,
        "ttft_ps": result.request_metrics[0].ttft_ps,
        "step_latency_ps": result.step_latency_ps,
        "serialization_only_expected_ps": expected,
        "completion_event_count": len(execution.events),
        "matched": result.step_latency_ps == expected,
        "meaning": "coarse-runtime reachability; no packet or topology calibration",
    }


def relation_rows(rows: list[dict]) -> list[dict]:
    indexed = {(row["variant"], row["pp_width"], row["ep_width"], row["profile"]): row for row in rows}
    relations = []
    for profile, width in product(PROFILES, WIDTHS):
        for variant in VARIANTS:
            keys = [(variant, width, ep, profile) for ep in EP_WIDTHS]
            if not all(key in indexed and "pp_fct_p99_ps" in indexed[key] for key in keys):
                continue
            cells = [indexed[key] for key in keys]
            values = [cell["pp_fct_p99_ps"] for cell in cells]
            if variant == "rail":
                name, matched = "R2-rail-independence", len(set(values)) == 1
            else:
                name = "R3-node-local-load-growth"
                matched = values[0] <= values[1] <= values[2] and values[0] < values[2]
            relations.append({"family": name, "pp_width": width, "profile": profile,
                              "values_ps": values, "matched": matched})
            if variant == "node-local":
                shares = [cell["pp_communication_projection_share"] for cell in cells]
                relations.append({"family": "R5-node-local-share-growth", "pp_width": width,
                                  "profile": profile, "values": shares,
                                  "matched": shares[0] <= shares[1] <= shares[2] and shares[0] < shares[2]})
    for width in WIDTHS:
        keys = [(variant, width, 0, "rnic-nn") for variant in VARIANTS]
        if all(key in indexed and "pp_fct_p99_ps" in indexed[key] for key in keys):
            values = [indexed[key]["pp_fct_p99_ps"] for key in keys]
            relations.append({"family": "R4-null-zero-load-equality", "pp_width": width,
                              "values_ps": values, "matched": values[0] == values[1]})
    for profile in PROFILES:
        keys = [("rail", width, 0, profile) for width in WIDTHS]
        if all(key in indexed and "pp_fct_p99_ps" in indexed[key] for key in keys):
            cells = [indexed[key] for key in keys]
            per_added_stage = [(b["pp_final_stage_projection_ps"] - a["pp_final_stage_projection_ps"])
                               / (b["pp_width"] - a["pp_width"]) for a, b in pairwise(cells)]
            predicted = COMPUTE_PS + cells[0]["pp_fct_p99_ps"]
            relations.append({"family": "R5-rail-stage-scaling", "profile": profile,
                              "values_ps": per_added_stage, "predicted_ps": predicted,
                              "matched": all(abs(value - predicted) <= 2000 for value in per_added_stage)})
    return relations


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def draw(summary: dict, directory: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    figure, axes = plt.subplots(2, 2, figsize=(7, 5.5), sharex=True)
    for column, profile in enumerate(PROFILES):
        for variant, style in (("node-local", "-"), ("rail", "--")):
            for width in WIDTHS:
                color = f"C{WIDTHS.index(width)}"
                cells = sorted((row for row in summary["configurations"]
                                if row["variant"] == variant and row["pp_width"] == width
                                and row["profile"] == profile and "pp_fct_p99_ps" in row),
                               key=lambda row: row["ep_width"])
                if not cells:
                    continue
                label = f"{variant}, P={width}"
                axes[0, column].plot([r["ep_width"] for r in cells],
                                     [r["pp_fct_p99_ps"] / 1e6 for r in cells],
                                     style, color=color, marker="o", label=label)
                axes[1, column].plot([r["ep_width"] for r in cells],
                                     [100 * r["pp_communication_projection_share"] for r in cells],
                                     style, color=color, marker="o", label=label)
        axes[0, column].set_title(profile + (" (topology-free)" if column == 0 else " (Clos)"))
        axes[0, column].set_ylabel("PP hop p99 FCT (us)")
        axes[1, column].set_ylabel("PP communication projection (%)")
        axes[1, column].set_xlabel("EP participants")
        axes[1, column].set_xticks(EP_WIDTHS)
        axes[0, column].margins(y=0.15)
        axes[1, column].margins(y=0.15)
    handles = [Line2D([], [], color=f"C{WIDTHS.index(width)}", marker="o", label=f"P={width}")
               for width in reversed(WIDTHS)]
    handles += [Line2D([], [], color="black", linestyle=style, label=variant)
                for variant, style in (("node-local", "-"), ("rail", "--"))]
    figure.legend(handles=handles, loc="lower center", ncol=5, fontsize=8)
    axes[0, 0].text(0.5, 0.06, "All P and both fabrics overlap", transform=axes[0, 0].transAxes,
                    ha="center", fontsize=8)
    axes[0, 1].text(0.5, 0.98, "Node-local P=4,8 overlap", transform=axes[0, 1].transAxes,
                    ha="center", va="top", fontsize=8)
    figure.tight_layout(rect=(0, 0.12, 1, 1))
    directory.mkdir(parents=True, exist_ok=True)
    figure.savefig(directory / "pp_rail.png", dpi=160)
    figure.savefig(directory / "pp_rail.pdf")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    parser.add_argument("--summary", type=Path, default=HERE / "results.json")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--cell", nargs=4, metavar=("VARIANT", "P", "EP", "PROFILE"))
    args = parser.parse_args()
    if args.plot_only:
        draw(json.loads(args.summary.read_text()), HERE / "figures")
        return
    root = args.out or (Path(os.environ["SIMLLM_DATA_ROOT"]) if os.environ.get("SIMLLM_DATA_ROOT") else None)
    if root is None:
        parser.error("set SIMLLM_DATA_ROOT or pass --out for bulk evidence")
    if root.resolve().is_relative_to(HERE.parents[1]):
        parser.error("bulk evidence must be outside the repository")
    executables = (find_htsim_rnic(), find_txt2bin())
    if any(binary is None for binary in executables):
        parser.error("set SIMLLM_HTSIM_RNIC and SIMLLM_TXT2BIN to the pinned binaries")
    grid = list(product(VARIANTS, WIDTHS, EP_WIDTHS, PROFILES))
    if args.cell:
        variant, width, ep_width, profile = args.cell
        selected = (variant, int(width), int(ep_width), profile)
        if selected not in grid:
            parser.error("--cell must select a member of the frozen grid")
        grid = [selected]
    rows, keyed_pp, errors = [], {}, []
    for variant, width, ep_width, profile in grid:
        print(f"RUN {variant} P={width} EP={ep_width} {profile}", flush=True)
        try:
            row, pp = run_cell(root, variant, width, ep_width, profile)
            rows.append(row)
            keyed_pp[(variant, width, ep_width, profile)] = pp
            print(f"  PP p99={row.get('pp_fct_p99_ps')} ps; fatal={row['fatal_findings']}", flush=True)
        except (RuntimeError, ValueError, OSError, subprocess.TimeoutExpired) as error:
            # The full local error remains in bulk storage. Public artifacts
            # avoid copying executable or personal filesystem paths from it.
            name = f"{variant}-pp{width}-ep{ep_width}-{profile}"
            root.mkdir(parents=True, exist_ok=True)
            (root / f"{name}-error.log").write_text(str(error) + "\n")
            errors.append({"cell": name, "error_type": type(error).__name__})
            print(f"  VOID: {type(error).__name__}; see bulk error log", flush=True)
    for row in rows:
        if row["profile"] != "rnic-cn":
            continue
        key = (row["variant"], row["pp_width"], row["ep_width"])
        physical, baseline = keyed_pp.get((*key, "rnic-cn"), []), keyed_pp.get((*key, "rnic-nn"), [])
        if physical and baseline and physical[0].start_time_ps == baseline[0].start_time_ps:
            normalized = normalized_fct(physical[:1], baseline[:1])[0]
            row["first_aligned_hop_normalized_fct"] = normalized.slowdown
            row["first_aligned_hop_baseline_fct_ps"] = normalized.baseline_fct_ps
            require_guard(normalized.slowdown >= 1, "aligned physical FCT is below null baseline",
                          row["fatal_findings"])
        row["later_hop_normalization"] = "omitted: starts are model-dependent"
    coarse = [coarse_metric_check(width) for width in (1, *WIDTHS)]
    profile_manifests = {}
    for row in rows:
        manifest = row.pop("manifest", [])
        profile_manifests.setdefault(row["profile"], manifest)
        row["manifest_sha256"] = hashlib.sha256("\n".join(manifest).encode()).hexdigest()
    summary = {
        "backend_profile_manifests": profile_manifests,
        "expectations_commit": EXPECTATIONS_COMMIT,
        "seed": SEED, "htsim_build_commit": "617ce20",
        "executable_sha256": {binary.name: digest(binary) for binary in executables},
        "configuration_count": len(rows), "requested_configuration_count": len(grid),
        "configurations": rows, "execution_errors": errors,
        "behavioral_and_exact_relations": relation_rows(rows),
        "coarse_metric_reachability": coarse,
        "fatal_status": "void" if errors or any(row["fatal_findings"] for row in rows)
        or not all(row["matched"] for row in coarse) else "clear",
        "claim_status": "component evidence; TRAF-8 and PLACE-1 remain open",
    }
    write_json(args.summary, summary)
    if not args.cell:
        draw(summary, HERE / "figures")
    print(f"CONFIGURATIONS {len(rows)}/{len(grid)}; FATAL STATUS {summary['fatal_status']}", flush=True)


if __name__ == "__main__":
    main()
