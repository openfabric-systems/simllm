"""Execute frozen bottleneck cells or replay compact retained backend extrema."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from collections import defaultdict
from dataclasses import asdict, replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

from simllm.backends import HtsimRequestMetricReducer, HtsimStepSink, HtsimStepSinkConfig
from simllm.compute import GPU_ENVELOPES, step_kernel
from simllm.core import (
    CoarseDeviceRuntime,
    CompletionReducer,
    VirtualClock,
    step_record_to_json,
    step_result_to_json,
)
from simllm.core.bookkeeping import (
    BookkeepingEntry,
    BookkeepingLedger,
    BookkeepingScope,
    CreatedObjectKind,
    CreatedObjectRecord,
    CreatedObjectRef,
    ObjectOwner,
)
from simllm.core.bottleneck import (
    bottleneck_report_from_json,
    bottleneck_report_to_json,
    bottleneck_step_from_json,
    bottleneck_step_to_json,
    classify_runtime,
)
from simllm.core.execution import OperationCorrelation

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FREEZE = "525b74e"


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


M4 = load("bottleneck_m4", "examples/m4/run_m4.py")
BREAKDOWN = load("bottleneck_breakdown", "examples/breakdown/run_breakdown.py")
WIDTH = load("bottleneck_width", "examples/collective_width_tail_v1/run_study.py")
CORE5 = load("bottleneck_core5", "examples/core5_reduction/run_study.py")
PACKET = load("bottleneck_packet", "examples/packet_breakdown_v1/run_study.py")


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def digest(data):
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())


def require(condition, message):
    if not condition:
        raise ValueError(f"VOID: {message}")


def cases():
    rows = []
    for kind, rates in (("m4", (200, 400)), ("breakdown", (100, 400))):
        for width in (2, 8):
            for rate in rates:
                rows.append({"kind": kind, "width": width, "rate": rate, "pattern": "ring",
                                 "profile": "rnic-nn-fluid", "name": f"{kind}-w{width}-{rate}g"})
    for pattern in ("ring", "all-to-all"):
        for width in (8, 64):
            for rate in (200, 400):
                rows.append({"kind": "width", "width": width, "rate": rate, "pattern": pattern,
                                 "profile": "rnic-nn", "name": f"width-{pattern}-w{width}-{rate}g"})
    return rows


def configuration(case, directory, enabled=None):
    if case["kind"] == "width":
        cell = WIDTH.Cell(case["pattern"], case["width"], case["rate"], case["profile"], "step")
        cfg = WIDTH.make_sink(cell, directory, None).config
    else:
        cfg = HtsimStepSinkConfig(profile=case["profile"], tp_ranks=tuple(range(case["width"])),
                                  dims=M4.dims_tp(case["width"]), workdir=directory,
                                  linkspeed_bps=case["rate"] * 10**9)
    if enabled is not None:
        cfg.emit_bottleneck_report = enabled
    return cfg


def records_for(case):
    if case["kind"] == "width":
        from simllm.core import RequestPhase, ScheduledRequest, StepRecord
        tokens = 512 if case["pattern"] == "ring" else case["width"] * 32
        return [StepRecord(0, 0, [ScheduledRequest("request", RequestPhase.PREFILL,
                                                 tokens, context_length=tokens)], num_sampled=1)]
    if case["kind"] == "breakdown":
        return BREAKDOWN.request_steps()
    # m4 freezes distinct prefill and eight-request decode shapes.
    return [M4.prefill_record(), M4.decode_record(1), M4.decode_record(2)]


def physical_bounds(case, record, config):
    """Derive transport and work limits before executing or reading outcomes."""
    kernel = step_kernel(config.dims, record, num_sampled=record.num_sampled if record.num_sampled is not None else len(record.scheduled))
    physical_kernel_floor = int(max(kernel.flops / config.gpu.peak_flops,
                                    kernel.bytes_moved / config.gpu.mem_bandwidth) * 1e12)
    if case["kind"] == "width":
        limits = WIDTH.bounds(WIDTH.Cell(case["pattern"], case["width"], case["rate"], case["profile"], "step"))
        fabric_floor = 2 * limits["phase_floor_ps"]
        fabric_ceiling = 2 * limits["phase_ceiling_ps"]
        compute_floor = compute_ceiling = WIDTH.COMPUTE_PS
    else:
        payload = record.total_new_tokens * M4.HIDDEN * M4.DTYPE_BYTES
        rounds = 2 * M4.LAYERS * 2 * (case["width"] - 1)
        fabric_floor = rounds * (payload // case["width"] * (8000 // case["rate"]) + 2_000_000)
        fabric_ceiling = fabric_floor
        duration = config.provider.estimate(kernel, config.gpu).duration_ps
        compute_floor = compute_ceiling = M4.LAYERS * max(duration // (M4.LAYERS * 1000), 1) * 1000
    return {"fabric_floor_ps": fabric_floor, "fabric_ceiling_ps": fabric_ceiling,
                "compute_floor_ps": compute_floor, "compute_ceiling_ps": compute_ceiling,
                "physical_kernel_floor_ps": physical_kernel_floor,
                "step_floor_ps": fabric_floor + compute_floor,
                "step_ceiling_ps": fabric_ceiling + compute_ceiling}


def install_capture(sink, retained):
    original = sink._run_goal

    def run(plan, goal, csv):
        value = original(plan, goal, csv)
        # Keep actual witness rows for all extrema the projection consumes.
        witnesses = sorted({(f.start_time_ps, f.completion_time_ps, f.fct_ps) for f in (
            min(value.flows, key=lambda f: f.start_time_ps),
            max(value.flows, key=lambda f: f.completion_time_ps),
            min(value.flows, key=lambda f: f.fct_ps),
            max(value.flows, key=lambda f: f.fct_ps),
        )})
        retained[goal.name] = [value.job_completion_time_ps(), len(value.flows), witnesses]
        return value
    sink._run_goal = run


def install_replay(sink, retained, source=None):
    def run(plan, goal, csv):
        job, count, witnesses = retained[goal.name]
        require(count >= len(witnesses), "retained witness count")
        if source is not None:
            require(goal.read_bytes() == (source / goal.name).read_bytes(), "replay GOAL bytes")
            for target in (csv, goal.with_suffix(".bin")):
                if (source / target.name).exists():
                    shutil.copyfile(source / target.name, target)
        flows = [SimpleNamespace(start_time_ps=s, completion_time_ps=c, fct_ps=f)
                 for s, c, f in witnesses]
        flows += [flows[0]] * (count - len(flows))
        return SimpleNamespace(flows=flows, quiescent=True, job_completion_time_ps=lambda: job)
    sink._run_goal = run


def pack(retained):
    groups = defaultdict(list)
    values = {}
    for name, value in retained.items():
        key = canonical(value).decode()
        groups[key].append(name)
        values[key] = value
    return [{"timing": values[key], "artifacts": sorted(names)} for key, names in sorted(groups.items())]


def unpack(groups):
    return {name: group["timing"] for group in groups for name in group["artifacts"]}


def artifacts(path):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(path.iterdir()) if p.suffix in {".goal", ".csv", ".bin"}}


def ranked_summary(report):
    return [{"class": c.class_name, "width": c.participant_width, "span_ps": c.span_ps,
             "share_percent": float(Fraction(c.span_ps * 100, report.span_ps))} for c in report.classes]


def packet_cell(case, directory, fixture=None):
    cfg = configuration(case, directory / "native", True)
    sink = HtsimStepSink(cfg)
    retained = {} if fixture is None else unpack(fixture["backend_extrema"])
    if fixture is None:
        install_capture(sink, retained)
    else:
        install_replay(sink, retained)
    arrivals = {s.request_id: 0 for record in records_for(case) for s in record.scheduled}
    reducer = HtsimRequestMetricReducer(arrivals)
    records, results, request_metrics, rows = [], [], [], []
    release = 0
    for raw in records_for(case):
        record = replace(raw, virtual_time_ps=release)
        bounds = physical_bounds(case, record, cfg)
        result = sink(record)
        require(result is not None, "packet step unexpectedly bypassed")
        report = sink.bottleneck_reports[-1]
        media = sink.packet_breakdowns[-1].detail.media
        require(bounds["fabric_floor_ps"] <= media.fabric_ps <= bounds["fabric_ceiling_ps"], "fabric physical bounds")
        require(bounds["compute_floor_ps"] == media.kernel_ps == bounds["compute_ceiling_ps"], "declared compute quantization")
        require(media.kernel_ps + M4.LAYERS * 1000 >= bounds["physical_kernel_floor_ps"], "kernel physical floor")
        require(bounds["step_floor_ps"] <= result.step_latency_ps <= bounds["step_ceiling_ps"], "request physical bounds")
        require(report.step.span_ps == result.step_latency_ps, "step conservation")
        require(report.step.share("fabric") * report.step.span_ps == media.fabric_ps, "fabric selected owner")
        metrics = reducer.consume(record, result, sink.locality_outcomes[-1],
                                  packet_breakdown=sink.packet_breakdowns[-1], bottleneck_report=report)
        request_report = reducer.bottleneck_reports[-1]
        with_metrics = replace(result, request_metrics=metrics)
        payload = bottleneck_step_to_json(with_metrics, request_report)
        decoded, decoded_report = bottleneck_step_from_json(json.loads(canonical(payload)))
        require(decoded == with_metrics and decoded_report == request_report, "strict JSON round trip")
        require(bottleneck_report_from_json(bottleneck_report_to_json(report)) == report, "step record round trip")
        for request in request_report.requests:
            metric = next(m for m in metrics if m.request_id == request.request_id)
            require(request.ttft.latency_ps == metric.ttft_ps, "TTFT conservation")
            require((None if request.tpot is None else request.tpot.latency_ps) == metric.tpot_ps, "TPOT conservation")
        write_json(directory / f"report-{record.step_index}.json", payload)
        rows.append({"step_index": record.step_index, "phase": record.scheduled[0].phase.value,
                         "latency_ps": result.step_latency_ps, "ranked": ranked_summary(report.step),
                         "kernel_bound_classes": sorted({k.bound_class for k in report.step.kernels}),
                         "measured_fraction_state": "absent-by-design", "bounds": bounds,
                         "flow_artifacts": len(report.step.flow_tails),
                         "max_fct_tail_share": max((float(t.tail_share) for t in report.step.flow_tails), default=None)})
        records.append(record)
        results.append(result)
        request_metrics.extend(metrics)
        release = result.completed_at_ps
    snapshot = PACKET.exact_snapshot(sink, results)
    metric_bytes = canonical([asdict(m) | {"phase": m.phase.value,
                                         "tpot_ps": None if m.tpot_ps is None else [m.tpot_ps.numerator, m.tpot_ps.denominator]}
                              for m in request_metrics])
    for mode in (None, False):
        replay_dir = directory / ("default" if mode is None else "false")
        replay_sink = HtsimStepSink(configuration(case, replay_dir, mode))
        install_replay(replay_sink, retained, source=cfg.workdir)
        replay_reducer = HtsimRequestMetricReducer(arrivals)
        replay_results, replay_metrics = [], []
        for record in records:
            result = replay_sink(record)
            replay_results.append(result)
            replay_metrics.extend(replay_reducer.consume(record, result, replay_sink.locality_outcomes[-1]))
            require(bottleneck_step_to_json(result) == step_result_to_json(result), "absent wire identity")
        require(PACKET.exact_snapshot(replay_sink, replay_results) == snapshot, "accepted snapshot identity")
        require(artifacts(replay_dir) == artifacts(cfg.workdir), "accepted artifact identity")
        require(replay_metrics == request_metrics, "all request timestamps and metrics identical")
    snapshot_hash = digest(snapshot)
    if fixture is not None:
        require(fixture["case"] == case, "retained case identity")
        require(fixture["records"] == [step_record_to_json(r) for r in records], "retained record identity")
        require(snapshot_hash == fixture["snapshot_sha256"], "retained accepted snapshot")
        require(digest(metric_bytes) == fixture["metric_sha256"], "retained request metrics")
    fixture = {"case": case, "records": [step_record_to_json(r) for r in records],
                   "backend_extrema": pack(retained), "snapshot_sha256": snapshot_hash,
                   "metric_sha256": digest(metric_bytes)}
    return {"case": case, "steps": rows, "total_ps": sum(r["latency_ps"] for r in rows),
                "ranked": ranked_summary(_combine_steps(sink.bottleneck_reports)),
                "fatal_guards": "held", "identity_ps": 0, "snapshot_sha256": snapshot_hash}, fixture


def _combine_steps(reports):
    from simllm.core.bottleneck import combine
    return combine(*(r.step for r in reports))


def coarse_cell(rate, gap, enabled):
    arrivals = BookkeepingLedger(tuple(BookkeepingEntry(i, CreatedObjectRecord(
        CreatedObjectRef(CreatedObjectKind.FRAMEWORK_REQUEST, f"request-{i}"),
        ObjectOwner.FRAMEWORK, CORE5.T0,
        BookkeepingScope(correlation=OperationCorrelation(request_ids=(f"request-{i}",))),
    )) for i in range(2)))
    runtime = CoarseDeviceRuntime(CORE5._profile(rate))
    clock = VirtualClock(CORE5.T0 + gap)
    reducer = CompletionReducer(clock, bookkeeping=arrivals, emit_bottleneck_report=enabled)
    results, rows = [], []
    for index in range(3):
        source = CORE5._step_record(index, clock.now_ps)
        graph = CORE5._two_request_graph(index, clock.now_ps, False)
        # First-principles model ceiling and floor: selected overlap critical
        # chain is two 4096-byte handoffs plus kernel, collective and control.
        expected_service = 2 * CORE5.CONTROL_BYTES * 8000 // rate + CORE5.C_PS + CORE5.A_PS + CORE5.H_PS
        result = runtime.execute(graph)
        segments = (classify_runtime(graph, runtime.last_report, runtime.selected_critical_visits,
                                     GPU_ENVELOPES["b100"]) if enabled else None)
        output = reducer.reduce(source, graph, result, runtime.last_report, bottleneck_segments=segments)
        require(output.step_latency_ps == expected_service, "coarse physical chain bounds")
        results.append(step_result_to_json(output))
        if enabled:
            report = reducer.bottleneck_reports[-1]
            report.validate_result(output)
            require(bottleneck_report_from_json(bottleneck_report_to_json(report)) == report, "coarse round trip")
            rows.append({"step_index": index, "latency_ps": output.step_latency_ps,
                             "ranked": ranked_summary(report.step),
                             "requests": [{"request_id": r.request_id, "ttft_ps": r.ttft.span_ps,
                                            "ttft_ranked": ranked_summary(r.ttft),
                                            "tpot_ps": None if r.tpot is None else [r.tpot.latency_ps.numerator, r.tpot.latency_ps.denominator]}
                                       for r in report.requests]})
    return results, rows, reducer


def behavioral_checks(rows, coarse):
    checks = []

    def check(name, ok, evidence):
        checks.append({"relation": name, "held": bool(ok), "evidence": evidence})

    def share(row, cls="fabric"):
        return sum(r["share_percent"] for r in row["ranked"] if r["class"] == cls)

    for row in rows:
        c = row["case"]
        if c["kind"] == "m4" and c["width"] == 8:
            check("m4-fabric-first", all(s["ranked"][0]["class"] == "fabric" for s in row["steps"]), c["name"])
        if c["kind"] == "breakdown":
            check("breakdown-kernel-regimes", all(s["kernel_bound_classes"] == (["compute-bound"] if s["phase"] == "prefill" else ["hbm-bound"])
                                                   for s in row["steps"]), c["name"])
    for kind in ("m4", "breakdown", "width"):
        subset = [r for r in rows if r["case"]["kind"] == kind]
        for pattern in sorted({r["case"]["pattern"] for r in subset}):
            select = [r for r in subset if r["case"]["pattern"] == pattern]
            widths = sorted({r["case"]["width"] for r in select})
            rates = sorted({r["case"]["rate"] for r in select})
            lookup = {(r["case"]["width"], r["case"]["rate"]): r for r in select}
            for rate in rates:
                check("fabric-share-rises-with-width", share(lookup[widths[1], rate]) > share(lookup[widths[0], rate]),
                      {"study": kind, "pattern": pattern, "rate": rate,
                           "shares": [share(lookup[w, rate]) for w in widths]})
            for width in widths:
                check("fabric-share-rises-at-lower-rate", share(lookup[width, rates[0]]) > share(lookup[width, rates[1]]),
                      {"study": kind, "pattern": pattern, "width": width,
                           "shares": [share(lookup[width, rate]) for rate in rates]})
    for kind, rates in (("m4", (200, 400)), ("breakdown", (100, 400)), ("width", (200, 400))):
        select = [r for r in rows if r["case"]["kind"] == kind and r["case"]["pattern"] == "ring"]
        widths = sorted({r["case"]["width"] for r in select})
        for width in widths:
            slow, fast = (next(r for r in select if r["case"]["width"] == width and r["case"]["rate"] == rate)
                          for rate in rates)
            factor = rates[1] // rates[0]
            propagation = 4 * (width - 1) * 2_000_000 * (1 if kind == "width" else M4.LAYERS)
            errors = []
            for a, b in zip(slow["steps"], fast["steps"], strict=True):
                fa, fb = (sum(c["span_ps"] for c in r["ranked"] if c["class"] == "fabric") for r in (a, b))
                errors.append((fa - propagation) - factor * (fb - propagation))
            check("serialization-rate-scaling", all(error == 0 for error in errors),
                  {"study": kind, "width": width, "factor": factor, "errors_ps": errors})
    for pattern, anchor in (("ring", 85.892), ("all-to-all", 60.515)):
        row = next(r for r in rows if r["case"] == {"kind": "width", "width": 64, "rate": 400, "pattern": pattern,
                                                       "profile": "rnic-nn", "name": f"width-{pattern}-w64-400g"})
        check("width64-rounded-anchor", abs(share(row) - anchor) <= .0005,
              {"pattern": pattern, "expected_percent": anchor, "observed_percent": share(row)})
    for row in coarse:
        if row["gap_ps"]:
            check("coarse-batching-first", all(r["ttft_ranked"][0]["class"] == "batching-queue" for r in row["steps"][0]["requests"]),
                  {"rate": row["rate"], "gap_ps": row["gap_ps"]})
    return checks


def exact_oracles(rows, coarse):
    result = []
    for row in rows:
        c = row["case"]
        if c["kind"] == "breakdown":
            sums = {name: sum(r["span_ps"] for r in row["ranked"] if r["class"] == name)
                    for name in ("compute-bound", "hbm-bound", "fabric")}
            observed = [sums["compute-bound"], sums["hbm-bound"], sums["fabric"], row["total_ps"]]
            expected = list(BREAKDOWN.FROZEN_F[c["width"], f'{c["rate"]}G'])
            result.append({"cell": c["name"], "expected": expected, "observed": observed, "held": expected == observed})
            require(expected == observed, "breakdown accepted exact component oracle")
    for rate in (200, 400):
        low, high = (next(r for r in coarse if r["rate"] == rate and r["gap_ps"] == gap) for gap in (0, 10**9))
        for a, b in zip(low["steps"], high["steps"], strict=True):
            require(a["latency_ps"] == b["latency_ps"], "admission preserves step service")
            for ra, rb in zip(a["requests"], b["requests"], strict=True):
                require(rb["ttft_ps"] - ra["ttft_ps"] == 10**9 and ra["tpot_ps"] == rb["tpot_ps"], "admission exact metric shift")
    return result


def plot(report, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [(r["case"]["name"], r["ranked"]) for r in report["packet_cells"]]
    for row in report["coarse_cells"]:
        rows.append((f'core5-{row["rate"]}g-gap{row["gap_ps"] // 10**6}us', row["steps"][0]["requests"][0]["ttft_ranked"]))
    classes = sorted({c["class"] for _, cs in rows for c in cs})
    colors = {name: plt.get_cmap("tab10")(i % 10) for i, name in enumerate(classes)}
    fig, ax = plt.subplots(figsize=(9, 8))
    for index, (_, cs) in enumerate(rows):
        left = 0
        for c in cs:
            ax.barh(index, c["share_percent"], left=left, color=colors[c["class"]], height=.7)
            left += c["share_percent"]
    ax.set_yticks(range(len(rows)), [name for name, _ in rows], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Selected critical-path latency share (%)")
    ax.set_title("Bottleneck classes, largest contribution first")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=colors[c], label=c) for c in classes],
              loc="upper center", bbox_to_anchor=(.5, -.09), ncol=3, frameon=False, fontsize=9)
    fig.tight_layout()
    destination.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination / "ranked_shares.png", dpi=150)
    fig.savefig(destination / "ranked_shares.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def results_markdown(report):
    anchors = [c["evidence"] for c in report["behavioral_relations"] if c["relation"] == "width64-rounded-anchor"]
    misses = [c for c in report["behavioral_relations"] if not c["held"]]
    lines = ["# Bottleneck report v1 results", "",
             "What ran: 16 packet configurations and four coarse-runtime configurations exercise the optional selected-path report beside time to first token (TTFT) and time per output token (TPOT).", "",
             f'What came out: {report["state"].upper()}. Every conservation and timing-identity comparison has 0 ps error. At width 64 and 400 Gbit/s, fabric contributes ' +
             " and ".join(f'{a["observed_percent"]:.6f}% ({a["pattern"]})' for a in anchors) + f'. Behavioral relation misses: {len(misses)}.', "",
             ("What it changes: CORE-67 meets its functional study acceptance, and its entry is removed for integration. Strict step and request bottleneck records, packet flow-tail diagnostics and the exact kernel name/config/GPU measurement join are implemented. The orchestrator must reconcile the task index before merging." if not misses else "What it changes: CORE-67 remains open on the frozen behavioral relations listed in Findings. The strict optional record is implemented."), "",
             "What it does not change: no runtime service, default, packet byte, TTFT or TPOT changes. COMP-90 measurements retain their evidence and void state; no kernel calibration, physical fabric calibration, or new deployment validity is claimed.", "",
             f'Frozen expectations commit: `{FREEZE}`. The freeze precedes implementation and the first study run. Existing published outcomes informed the predictions.', "",
             "## Evidence classes", "", "| Class | Result |", "|---|---|",
             f'| Run configurations | {len(report["packet_cells"])} packet, {len(report["coarse_cells"])} coarse |',
             f'| Exact published component oracle rows | {len(report["exact_oracles"])} held |',
             f'| Behavioral relation instances | {len(report["behavioral_relations"])} evaluated, {len(misses)} misses |',
             "| Fatal structural, physical and identity guards | Held, unscored |",
             "| GPU or network-dependent test executables | None |", "",
             "These counts are separate evidence classes, never a combined score. A fatal guard failure voids the study. Inherited void measurements are diagnostic only.", "",
             "## Ranked cells", "", "| Cell | First class | Share (%) | Total (ps) |", "|---|---|---:|---:|"]
    for row in report["packet_cells"]:
        first = row["ranked"][0]
        lines.append(f'| {row["case"]["name"]} | {first["class"]} | {first["share_percent"]:.6f} | {row["total_ps"]} |')
    for row in report["coarse_cells"]:
        req = row["steps"][0]["requests"][0]
        first = req["ttft_ranked"][0]
        lines.append(f'| core5 {row["rate"]}G gap {row["gap_ps"]} ps | {first["class"]} | {first["share_percent"]:.6f} | {req["ttft_ps"]} |')
    lines += ["", "Packet table totals sum the disjoint steps of each declared cell. The coarse rows show the first request's TTFT. Each underlying step and sampled request has its own strict report in the bulk run. The figure orders contributions within each row by descending share.", "",
              "![Ranked critical-path shares](figures/ranked_shares.png)", "",
              "## Physical interpretation", "",
              "The GPU must fetch bytes and execute arithmetic before releasing the next communication phase. Prefill reuses weights across many tokens and its arithmetic work binds the ideal roof. Decode reuses them less and memory bandwidth binds it. The record compares declared arithmetic intensity with the device ridge point; it does not infer the measured cause of a slow kernel. All named serving cells have absent-by-design measured fractions because no matching measured kernel/config cell is supplied.", "",
              "The packet floors use payload bytes divided by link rate and propagation per dependent ring hop. The expert floor uses bytes received from all remote peers. The ideal ceilings serialize the declared packet slots conservatively. Each row retains the limits computed before execution. All observations lie inside their bounds, and the four published breakdown component tuples agree exactly. The fixed 100-microsecond width compute interval exceeds its declared-work roofline floors: 9.250770 microseconds for rings, 27.066368 for width-8 expert work and 40.568240 for width-64 expert work.", "",
              "The core5 chain is two 4096-byte handoffs plus 20000 ps kernel, 8000 ps collective and 1000 ps control. Its service is 356680 ps at 200G and 192840 ps at 400G. Adding 1000000000 ps of declared admission shifts only TTFT by that amount; it makes batching queue the first class without changing service or TPOT.", "",
              "Flow completion time (FCT) tail share is `(maximum FCT - minimum FCT) / maximum FCT` within each selected fabric artifact. It is a spread diagnostic, not a sum of flow waits and not a switch queue measurement. Full packet rows remain in bulk. Replay retains actual witness rows for extrema and counts, not the full FCT distribution.", "",
              "The m4 source uses distinct prefill and decode request identities. This replay declares all arrivals at zero, so the decode requests correctly attribute the preceding prefill to batching delay in their TTFT. The m4 fabric-first prediction concerns each executed step. Request rankings conserve the declared request history independently; they do not copy the step winner.", "",
              "These are model-side classifications. Matching a deterministic service model is not evidence of deployment token throughput. No hardware number or new calibration is inferred from this study.", "",
              "## Integration gate", "",
              "The numerical study holds, but the repository merge gate is blocked by the generated task index in `docs/README_PRO.md`: core has 26 open tasks and backends has 51, while the index still says 27 and 52. The backend difference is inherited from BACK-69. The worker contract forbids editing that file and reserves cross-file reconciliation for the orchestrator. Closure bookkeeping for CORE-67 and BACK-69 also belongs in `docs/task-ledger.json` during integration.", "",
              "The configured wave binaries also differ from the historical hashes required by two optional collective-floor study checks. Those checks are separate from this study's frozen backend cells. Ordinary continuous integration leaves those machine-specific binary selections unset and skips the historical identity checks; the retained-data checks remain enabled. No numerical guard is relaxed, and a passing study is not presented as a green repository gate.", "",
              "## Using the record", "",
              "Set `emit_bottleneck_report=True` on `HtsimStepSinkConfig` or `DeviceRuntimeStepSink`. Their `bottleneck_reports` list publishes only completed calls. The coarse sink includes request rankings directly. For packet requests, pass the step report as `bottleneck_report=` to `HtsimRequestMetricReducer.consume`; its own `bottleneck_reports` then includes TTFT and TPOT. The persistent packet sink publishes reports only when a prepared call is consumed.", "",
              "`bottleneck_step_to_json(result, report)` writes the optional field beside the original result. `bottleneck_step_from_json(payload)` returns the result and report, or `None` for a legacy absent report. The strict report codec also works independently. Measured cells use `bottleneck_kernel_cells`, keyed exactly like profile calibration by `(kernel.name, kernel.config, gpu.name)`. Each value is `KernelEvidence` with the original ledger cell identity, work, measured duration, roofs, evidence class and void state. A missing exact key remains absent-by-design; no family-only or cross-GPU fallback occurs.", "",
              "## Identity and reproduction", "",
              "Default absence and explicit false regenerate the identical GOAL plan and replay the same retained authoritative backend output. Their original StepResult, outcome, locality and request metric bytes match the enabled run after omitting only the optional report. GOAL, binary GOAL and native completion CSV bytes also match. Runtime records use exact integer picoseconds; TPOT divides cumulative class spans by the same exact interval count as the metric.", "",
              "```bash", ". ./.env.local.sh", ".venv/bin/python examples/bottleneck_report_v1/run_study.py --live", ".venv/bin/python examples/bottleneck_report_v1/run_study.py --replay", "```", "",
              "Bulk output defaults to `SIMLLM_DATA_ROOT/bottleneck_report_v1`. Backend binaries use `SIMLLM_HTSIM_RNIC` and `SIMLLM_TXT2BIN`, configured from `SIMLLM_HTSIM_BUILD`. Tracked text uses LF bytes. Replay accepts LF-normalized source digests and requires no native backend, GPU or network.", ""]
    if misses:
        lines += ["## Findings", "", *[f'- {r["relation"]}: {r["evidence"]}' for r in misses], ""]
    return "\n".join(lines)


def run(out, *, replay=False, artifact_dir=HERE, render=True):
    retained = json.loads((HERE / "fixtures.json").read_bytes()) if replay else None
    sources = {str(Path(module.__file__).relative_to(ROOT)): digest(Path(module.__file__).read_bytes())
               for module in (M4, BREAKDOWN, WIDTH, CORE5)}
    if retained is not None:
        require(retained["source_sha256"] == sources, "source digests")
    rows, fixtures, coarse = [], [], []
    for index, case in enumerate(cases()):
        print(f'Running {case["name"]}', flush=True)
        row, fixture = packet_cell(case, out / case["name"], None if retained is None else retained["cells"][index])
        rows.append(row)
        fixtures.append(fixture)
    example = None
    for rate in (200, 400):
        for gap in (0, 10**9):
            disabled, _, _ = coarse_cell(rate, gap, False)
            enabled, steps, reducer = coarse_cell(rate, gap, True)
            require(canonical(disabled) == canonical(enabled), "coarse identity")
            coarse.append({"rate": rate, "gap_ps": gap, "steps": steps, "identity_ps": 0})
            example = bottleneck_report_to_json(reducer.bottleneck_reports[-1])
    exact = exact_oracles(rows, coarse)
    relations = behavioral_checks(rows, coarse)
    report = {"schema": "simllm-bottleneck-study-v1", "expectations_commit": FREEZE,
                  "state": "held" if all(c["held"] for c in relations) else "findings",
                  "source_sha256": sources, "packet_cells": rows, "coarse_cells": coarse,
                  "fatal_guards": "held", "exact_oracles": exact, "behavioral_relations": relations,
                  "record_example": example}
    write_json(out / "results.json", report)
    write_json(artifact_dir / "results.json", report)
    if not replay:
        write_json(artifact_dir / "fixtures.json", {"source_sha256": sources, "cells": fixtures})
    (artifact_dir / "RESULTS.md").write_bytes(results_markdown(report).encode())
    if render:
        plot(report, artifact_dir / "figures")
    print(f'{report["state"].upper()}: 20 configurations; fatal guards held; identity error 0 ps; '
          f'{sum(not c["held"] for c in relations)} behavioral misses', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--live", action="store_true")
    selection.add_argument("--replay", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--artifact-dir", type=Path, default=HERE)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    root = os.environ.get("SIMLLM_DATA_ROOT")
    if args.out is None and not root:
        parser.error("configure SIMLLM_DATA_ROOT or --out")
    out = args.out or Path(root) / "bottleneck_report_v1"
    out.mkdir(parents=True, exist_ok=True)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    try:
        run(out, replay=args.replay, artifact_dir=args.artifact_dir, render=not args.no_plot)
    except ValueError as exc:
        write_json(out / "void.json", {"state": "void", "findings": str(exc), "expectations_commit": FREEZE})
        raise


if __name__ == "__main__":
    main()
