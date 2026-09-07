"""Frozen live packet study and deterministic replay of retained visit inputs."""

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
from pathlib import Path
from types import MethodType, SimpleNamespace

from simllm.backends import (
    HtsimPersistentStepSink,
    HtsimRequestMetricReducer,
    HtsimStepSink,
    HtsimStepSinkConfig,
    attribute_step_detail,
)
from simllm.backends.packet_breakdown import packet_step_from_json, packet_step_to_json
from simllm.core import (
    RequestPhase,
    ScheduledRequest,
    StepRecord,
    completion_event_to_json,
    step_record_to_json,
    step_records_from_jsonl,
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FREEZE = "7b39399"
PROPAGATION_PS = 2_000_000


def _load_study(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_width_study = _load_study("packet_width_source", HERE.parent / "collective_width_tail_v1/run_study.py")
_breakdown_study = _load_study("packet_breakdown_source", HERE.parent / "breakdown/run_breakdown.py")
Cell, bounds, make_sink = _width_study.Cell, _width_study.bounds, _width_study.make_sink
dims_tp, request_steps = _breakdown_study.dims_tp, _breakdown_study.request_steps


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def digest(data):
    return hashlib.sha256(data).hexdigest()


def text_digest(path):
    return digest(path.read_bytes().replace(b"\r\n", b"\n"))


def artifact_digest(directory, suffixes):
    return digest(canonical({
        p.name: digest(p.read_bytes().replace(b"\r\n", b"\n")
                       if p.suffix in (".goal", ".csv") else p.read_bytes())
        for p in sorted(directory.iterdir()) if p.suffix in suffixes
    }))


def cases():
    result = []
    for pattern in ("ring", "all-to-all"):
        for width in (64, 8):
            for rate in (400, 200):
                result.append({"kind": "width", "pattern": pattern, "width": width,
                               "rate": rate, "profile": "rnic-nn",
                               "name": f"width-{pattern}-{width}-{rate}"})
    for tp in (2, 8):
        for profile in ("rnic-nn-fluid", "rnic-nn"):
            result.append({"kind": "breakdown", "width": tp, "rate": 400,
                           "profile": profile, "name": f"breakdown-{tp}-{profile}"})
    for fixture in ("vllm-m2", "sglang-m3"):
        result.append({"kind": "m4", "fixture": fixture, "width": 8,
                       "rate": 400, "profile": "rnic-nn-fluid", "name": f"m4-{fixture}"})
    return result


def configuration(case, directory, *, enabled=None):
    if case["kind"] == "width":
        cell = Cell(case["pattern"], case["width"], case["rate"], case["profile"], "step")
        cfg = make_sink(cell, directory, None).config
    else:
        cfg = HtsimStepSinkConfig(
            profile=case["profile"], tp_ranks=tuple(range(case["width"])),
            dims=dims_tp(case["width"]), workdir=directory,
        )
    if enabled is not None:
        cfg.emit_packet_breakdown = enabled
    return cfg


def records_for(case):
    if case["kind"] == "width":
        tokens = 512 if case["pattern"] == "ring" else case["width"] * 32
        return [StepRecord(0, 0, [ScheduledRequest(
            "request", RequestPhase.PREFILL, tokens, context_length=tokens)], num_sampled=1)]
    if case["kind"] == "breakdown":
        return request_steps()
    return step_records_from_jsonl(
        REPO / "examples/m4/fixtures" / f"{case['fixture']}-steps.jsonl")


def physical_bounds(case, record):
    """Compute bounds from declared bytes, rates and propagation before execution."""
    if case["kind"] == "width":
        cell = Cell(case["pattern"], case["width"], case["rate"], case["profile"], "step")
        b = bounds(cell)
        return {"fabric_floor_ps": 2 * b["phase_floor_ps"],
                "fabric_ceiling_ps": 2 * b["phase_ceiling_ps"],
                "propagation_ps": (4 * (cell.width - 1) if cell.pattern == "ring" else 2)
                * PROPAGATION_PS}
    width = case["width"]
    payload = record.total_new_tokens * 4096 * 2
    chunk = payload // width
    rounds = 64 * 2 * (width - 1)
    floor = rounds * (chunk * 20 + PROPAGATION_PS)
    if case["profile"] == "rnic-nn-fluid":
        ceiling = floor
    else:
        packets = (chunk + 4095) // 4096
        ceiling = rounds * ((packets + 2) * 4160 * 20 + PROPAGATION_PS)
    return {"fabric_floor_ps": floor, "fabric_ceiling_ps": ceiling,
            "propagation_ps": rounds * PROPAGATION_PS}


def install_capture(sink, retained):
    original = sink._run_goal

    def run(self, plan, goal, csv):
        result = original(plan, goal, csv)
        row = [min(f.start_time_ps for f in result.flows),
               max(f.completion_time_ps for f in result.flows),
               result.job_completion_time_ps(), len(result.flows)]
        retained[goal.name] = row
        return result

    sink._run_goal = MethodType(run, sink)


def install_replay(sink, retained, *, source=None):
    """Retained extrema are sufficient for this artifact-granularity projection.

    Full native CSVs stay in bulk. Deterministic tests replay only published
    extrema and counts, with no claim to reconstruct individual flow identities.
    """
    def run(self, plan, goal, csv):
        first, last, job, count = retained[goal.name]
        if source is not None:
            if goal.read_bytes() != (source / goal.name).read_bytes():
                raise ValueError("replay GOAL bytes changed")
            for target in (csv, goal.with_suffix(".bin")):
                original = source / target.name
                if original.exists():
                    shutil.copyfile(original, target)
        flow = SimpleNamespace(start_time_ps=first, completion_time_ps=last,
                               fct_ps=last - first)
        return SimpleNamespace(flows=[flow] * count, quiescent=True,
                               job_completion_time_ps=lambda: job)

    sink._run_goal = MethodType(run, sink)


def pack_runs(retained):
    """Deduplicate equal timing rows without discarding artifact identities."""
    groups = defaultdict(list)
    for name, row in sorted(retained.items()):
        groups[tuple(row)].append(name)
    return [{"timing": list(row), "artifacts": names} for row, names in groups.items()]


def unpack_runs(groups):
    return {name: group["timing"] for group in groups for name in group["artifacts"]}


def replay(case, directory, records, retained, *, enabled, persistent=False, source=None):
    cfg = configuration(case, directory, enabled=enabled)
    sink = (HtsimPersistentStepSink(cfg, max_workers=2) if persistent else HtsimStepSink(cfg))
    install_replay(sink, retained, source=source)
    arrivals = {s.request_id: 0 for r in records for s in r.scheduled}
    reducer = HtsimRequestMetricReducer(arrivals)
    results, metrics = [], []
    try:
        if persistent:
            # Two prepared batches check publication and surviving worker state.
            split = max(1, len(records) // 2)
            batches = (records[:split], records[split:])
        else:
            batches = (records,)
        for batch in batches:
            if not batch:
                continue
            if persistent:
                published = len(sink.outcomes)
                sink.prepare(batch)
                assert len(sink.outcomes) == published
                assert len(sink.packet_breakdowns) == (published if enabled else 0)
            for record in batch:
                result = sink(record)
                assert result is not None
                projection = sink.packet_breakdowns[-1] if enabled else None
                results.append(result)
                metrics.extend(reducer.consume(
                    record, result, sink.locality_outcomes[-1], packet_breakdown=projection))
        return sink, results, metrics, reducer
    finally:
        if persistent:
            sink.close()


def exact_snapshot(sink, results):
    return canonical({
        "results": [packet_step_to_json(r) for r in results],
        "outcomes": [asdict(r) for r in sink.outcomes],
        "locality": [asdict(r) for r in sink.locality_outcomes],
        "collective_timing": [asdict(r) for r in sink.collective_timing_outcomes],
        "floor": [asdict(r) for r in sink.collective_floor_timing_outcomes],
        "registration": [asdict(r) for r in sink.collective_registration_outcomes],
        "cross_check": [asdict(r) for r in sink.dependency_cross_check_reports],
    })


def run_case(case, directory):
    native = directory / "native"
    sink = HtsimStepSink(configuration(case, native, enabled=True))
    retained = {}
    install_capture(sink, retained)
    records, results, rows = [], [], []
    release = 0
    for raw_record in records_for(case):
        record = replace(raw_record, virtual_time_ps=release)
        limits = physical_bounds(case, record)
        result = sink(record)
        if result is None:
            raise ValueError("registered cell unexpectedly bypassed packet execution")
        projection = sink.packet_breakdowns[-1]
        detail = attribute_step_detail(result, sink.locality_outcomes[-1],
                                       packet_breakdown=projection)
        breakdown = projection.breakdown
        # Floors apply to the transport including separately named visibility.
        fabric = detail.media.fabric_ps
        if not limits["fabric_floor_ps"] <= fabric <= limits["fabric_ceiling_ps"]:
            raise ValueError("fatal physical service envelope failure")
        if breakdown.external_dependency_ps != sink.locality_outcomes[-1].compute_service_ps:
            raise ValueError("fatal external compute conservation failure")
        if packet_step_from_json(packet_step_to_json(result, projection)) != (result, projection):
            raise ValueError("fatal strict wire round-trip failure")
        write_json(directory / "projections" / f"step-{record.step_index}.json",
                   packet_step_to_json(result, projection))
        events = projection.execution_result.events
        (directory / "projections" / f"step-{record.step_index}.events.jsonl").write_bytes(
            b"".join(canonical(completion_event_to_json(event)) for event in events))
        rows.append({"step": record.step_index, **asdict(breakdown), **limits,
                     "media": asdict(detail.media), "masked": asdict(detail.masked),
                     "visit_count": len(projection.visits), "event_count": len(events)})
        records.append(record)
        results.append(result)
        release = result.completed_at_ps
    reference = exact_snapshot(sink, results)
    modes = (("omitted", None, False), ("disabled", False, False),
             ("enabled", True, False), ("prepared-off", False, True),
             ("prepared-on", True, True))
    baseline_metrics = baseline_totals = None
    byte_locks = []
    for label, enabled, persistent in modes:
        directory_mode = directory / label
        replayed, step_results, metrics, reducer = replay(
            case, directory_mode, records, retained, enabled=enabled,
            persistent=persistent, source=native,
        )
        if exact_snapshot(replayed, step_results) != reference:
            raise ValueError(f"fatal accepted outcome byte mismatch: {label}")
        goal_match = artifact_digest(native, {".goal"}) == artifact_digest(directory_mode, {".goal"})
        csv_match = artifact_digest(native, {".csv"}) == artifact_digest(directory_mode, {".csv"})
        if not goal_match or not csv_match:
            raise ValueError(f"fatal accepted artifact byte mismatch: {label}")
        if baseline_metrics is None:
            baseline_metrics, baseline_totals = metrics, reducer.totals()
        if metrics != baseline_metrics or reducer.totals() != baseline_totals:
            raise ValueError(f"fatal reducer equality mismatch: {label}")
        if enabled:
            for row in reducer.totals():
                ttft, decode = reducer.critical_path_breakdowns()[row.request_id]
                assert ttft.operation_latency_ps == row.ttft_ps
                assert decode.operation_latency_ps == row.decode_attribution.total_ps
        byte_locks.append({"mode": label, "outcomes": True, "goals": goal_match,
                           "completions": csv_match, "reducer": True})
    return {**case, "status": "complete", "steps": rows, "byte_locks": byte_locks,
            "accepted_snapshot_sha256": digest(reference),
            "goal_sha256": artifact_digest(native, {".goal"}),
            "csv_sha256": artifact_digest(native, {".csv"})}, {
                "case": case, "records": [step_record_to_json(r) for r in records],
                "runs": pack_runs(retained), "snapshot_sha256": digest(reference),
                "goal_sha256": artifact_digest(native, {".goal"}),
            }


def audit_retained(case, directory, row):
    """Join regenerated GOAL bytes to the accepted width study and bulk evidence."""
    native = directory / "native"
    for lock in row["byte_locks"]:
        mode = directory / lock["mode"]
        binary_equal = artifact_digest(native, {".bin"}) == artifact_digest(mode, {".bin"})
        if not binary_equal:
            raise ValueError("binary GOAL replay bytes differ")
        lock["binary_goals"] = binary_equal
    if case["kind"] == "width":
        accepted = json.loads((HERE.parent / "collective_width_tail_v1/results.json").read_bytes())
        original = next(r for r in accepted["configurations"] if
                        r["mode"] == "step" and r["pattern"] == case["pattern"]
                        and r["width"] == case["width"] and r["rate_gbps"] == case["rate"]
                        and r["profile"] == case["profile"])
        observed = row["steps"][0]
        assert observed["operation_latency_ps"] == original["step_latency_ps"]
        assert observed["media"] == original["media"]
        assert observed["masked"] == original["masked_service"]
        for name, expected in original["artifact_goal_sha256"].items():
            payload = (native / name).read_bytes()
            assert expected in {digest(payload), digest(payload.replace(b"\r\n", b"\n"))}
        row["accepted_width_equal"] = True
    elif case["kind"] == "breakdown":
        compute, memory, fluid_network, fluid_total = _breakdown_study.FROZEN_F[(case["width"], "400G")]
        observed = row["steps"]
        assert observed[0]["external_dependency_ps"] == compute
        assert sum(s["external_dependency_ps"] for s in observed[1:]) == memory
        network = sum(s["service_ps"] for s in observed)
        if case["profile"] == "rnic-nn-fluid":
            assert network == fluid_network
            assert sum(s["operation_latency_ps"] for s in observed) == fluid_total
            reference = fluid_network
        else:
            reference, band = _breakdown_study.FROZEN_N[case["width"]]
            assert abs(network - reference) <= band
        row["accepted_breakdown_network_residual_ps"] = network - reference


def relations(rows):
    complete = {r["name"]: r for r in rows if r["status"] == "complete"}
    checks = []
    for pattern in ("ring", "all-to-all"):
        for rate in (400, 200):
            pair = [complete.get(f"width-{pattern}-{w}-{rate}") for w in (8, 64)]
            if all(pair):
                a, b = [p["steps"][0] for p in pair]
                checks.append({"family": "width_service_share", "pattern": pattern,
                               "rate": rate, "ok": a["service_ps"] * b["operation_latency_ps"]
                               < b["service_ps"] * a["operation_latency_ps"]})
        for width in (8, 64):
            pair = [complete.get(f"width-{pattern}-{width}-{rate}") for rate in (400, 200)]
            if all(pair):
                a, b = [p["steps"][0] for p in pair]
                residual = b["service_ps"] - a["propagation_ps"] - 2 * (
                    a["service_ps"] - a["propagation_ps"])
                checks.append({"family": "serialization_doubling", "pattern": pattern,
                               "width": width, "residual_ps": residual,
                               "propagation_ps": a["propagation_ps"], "ok": residual == 0})
    return checks


def plot(rows, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(7, 5.5), sharex=True, sharey=True)
    segments = ("launch_queue_ps", "device_queue_ps", "service_ps",
                "completion_delivery_ps", "external_dependency_ps")
    for axis, (pattern, rate) in zip(axes.flat, (
        ("ring", 400), ("ring", 200), ("all-to-all", 400), ("all-to-all", 200)), strict=True):
        selected = sorted((r for r in rows if r["kind"] == "width"
                           and r["status"] == "complete" and r["pattern"] == pattern
                           and r["rate"] == rate), key=lambda r: r["width"])
        bottom = [0.] * len(selected)
        labels = [str(r["width"]) for r in selected]
        for field in segments:
            values = [r["steps"][0][field] / 1e9 for r in selected]
            name = field.removesuffix("_ps").replace("_", " ")
            if all(v == 0 for r in rows if r["kind"] == "width" and r["status"] == "complete"
                   for v in [r["steps"][0][field]]):
                name += " (0 ps in every cell)"
            axis.bar(labels, values, bottom=bottom, label=name)
            for x, (base, value) in enumerate(zip(bottom, values, strict=True)):
                if value >= 0.06:
                    axis.text(x, base + value / 2, f"{value:.3f}", ha="center", va="center",
                              fontsize=7, color="white")
                elif value > 0:
                    axis.text(x + 0.43, base + value / 2, f"{value:.3f}", ha="left",
                              va="center", fontsize=7, color="C2")
            bottom = [a + b for a, b in zip(bottom, values, strict=True)]
        for x, total in enumerate(bottom):
            axis.text(x, total + 0.012, f"{total:.3f} ms", ha="center", va="bottom", fontsize=7)
        axis.set_title(f"{pattern}, {rate} Gbit/s")
        axis.set_xlabel("Participating ranks")
        axis.tick_params(axis="x", labelbottom=True)
        axis.set_ylabel("Step latency (ms)")
        axis.set_ylim(0, 0.95)
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="upper center", ncol=3,
               bbox_to_anchor=(.5, .995), fontsize=7.5)
    fig.text(0.5, 0.905, "Ideal profile (rnic-nn): the external dependency is the 100 ms fixed compute of every step;"
             " service is the fabric time; labels are ms", ha="center", fontsize=7.5)
    fig.tight_layout(rect=(0, 0, 1, .89))
    fig.savefig(directory / "packet_breakdown.png", dpi=160)
    fig.savefig(directory / "packet_breakdown.pdf", metadata={"CreationDate": None})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    parser.add_argument("--case", action="append")
    parser.add_argument("--collect", action="store_true")
    args = parser.parse_args()
    root = args.out or (Path(os.environ["SIMLLM_DATA_ROOT"]) / "packet_breakdown_v1")
    root.mkdir(parents=True, exist_ok=True)
    selected = [c for c in cases() if not args.case or c["name"] in args.case]
    rows, fixtures = [], []
    for case in selected:
        directory = root / case["name"]
        directory.mkdir(parents=True, exist_ok=True)
        if args.collect:
            row = json.loads((directory / "result.json").read_bytes())
            fixture = json.loads((directory / "fixture.json").read_bytes()) if row["status"] == "complete" else None
        else:
            print(f"RUN {case['name']}", flush=True)
            try:
                row, fixture = run_case(case, directory)
            except (ValueError, RuntimeError, AssertionError) as error:
                (directory / "error.txt").write_bytes(str(error).encode())
                row, fixture = {**case, "status": "void", "reason": type(error).__name__}, None
            write_json(directory / "result.json", row)
            if fixture:
                write_json(directory / "fixture.json", fixture)
            print(f"{row['status'].upper()} {case['name']}", flush=True)
        if row["status"] == "complete":
            audit_retained(case, directory, row)
        rows.append(row)
        if fixture:
            fixtures.append(fixture)
    if not args.case or args.collect:
        checks = relations(rows)
        write_json(HERE / "results.json", {
            "expectations_commit": FREEZE, "cells": rows, "relations": checks,
            "backend_binaries": {
                name: digest(Path(os.environ[name]).read_bytes())
                for name in ("SIMLLM_HTSIM_RNIC", "SIMLLM_TXT2BIN")
            },
            "evidence": "native ideal runs; five deterministic artifact replays per cell",
        })
        write_json(HERE / "fixtures.json", {
            "expectations_commit": FREEZE, "cells": fixtures,
            "inputs": {str(p.relative_to(REPO)): text_digest(p) for p in (
                HERE / "expectations.md",
                HERE.parent / "collective_width_tail_v1/results.json",
                HERE.parent / "breakdown/run_breakdown.py",
                REPO / "examples/m4/fixtures/vllm-m2-steps.jsonl",
                REPO / "examples/m4/fixtures/sglang-m3-steps.jsonl",
            )},
        })
        plot(rows, HERE / "figures")
        print(f"{sum(r['status'] == 'complete' for r in rows)}/{len(rows)} cells complete; "
              f"{sum(c['ok'] for c in checks)}/{len(checks)} relations pass", flush=True)


if __name__ == "__main__":
    main()
