#!/usr/bin/env python3
"""Attribute the unchanged conformance live loop with exclusive host timers."""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
import os
import platform
import pstats
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
STUDY = Path(__file__).resolve().parent
FREEZE = "3c7d2a090bfa12f747087c66e22612ebe4e4e8ca"
AMENDMENT = "c7a0da5fad517135f0e40f500ad32db2b3b34c17"
CONTROLLED_FREEZE = "becdddc65852b3b4d988238ad723f807f4f25af0"
THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS")
PHASES = (
    "scheduler", "kv_allocate", "kv_free", "executor", "scheduler_output",
    "frontend_output", "engine_residual", "admission", "output_collection",
    "driver_residual",
)


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def text_matches(raw, frozen):
    """Accept checkout CRLF conversion without accepting a content change."""
    return raw.replace(b"\r\n", b"\n") == frozen.replace(b"\r\n", b"\n")


def text_hashes(raw):
    return {"raw": hashlib.sha256(raw).hexdigest(),
            "lf": hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()}


def host_snapshot(torch):
    # nproc honors OpenMP overrides; remove them only in this inventory command.
    environment = {k: v for k, v in os.environ.items()
                   if k not in ("OMP_NUM_THREADS", "OMP_THREAD_LIMIT")}
    return {"unix_time_ns": time.time_ns(), "load_average": list(os.getloadavg()),
            "taskset_output": subprocess.check_output(
                ["taskset", "-p", str(os.getpid())], text=True).strip(),
            "nproc": int(subprocess.check_output(["nproc"], env=environment, text=True)),
            "nproc_environment": "OpenMP count overrides removed for CPU inventory only",
            "affinity": sorted(os.sched_getaffinity(0)),
            "thread_environment": {name: os.getenv(name) for name in THREAD_ENV},
            "torch_intraop_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads()}


def host_guard(before, after):
    return all(snapshot["affinity"] == list(range(24, 32))
               and snapshot["nproc"] == 8
               and snapshot["torch_intraop_threads"] == 1
               and snapshot["torch_interop_threads"] == 1
               and all(snapshot["thread_environment"][name] == "1" for name in THREAD_ENV)
               for snapshot in (before, after))


def write_once(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical(value))


def stability(values):
    if len(values) != 7 or min(values) <= 0:
        raise ValueError("stability requires seven positive elapsed durations")
    whole = statistics.median(values)
    first, last = statistics.median(values[:3]), statistics.median(values[-3:])
    deviation = max(abs(first - whole), abs(last - whole), abs(first - last)) / whole
    return {"median_ns": whole, "first_three_median_ns": first,
            "last_three_median_ns": last, "maximum_deviation": deviation,
            "passed": deviation <= 0.10}


class Timers:
    def __init__(self, clock=time.perf_counter_ns):
        self.clock = clock
        self.stack = []
        self.events = []
        self.step = -1

    def call(self, phase, function, *args, **kwargs):
        frame = [self.clock(), 0]
        self.stack.append(frame)
        try:
            return function(*args, **kwargs)
        finally:
            elapsed = self.clock() - frame[0]
            if self.stack.pop() is not frame:
                raise RuntimeError("phase stack corruption")
            if self.stack:
                self.stack[-1][1] += elapsed
            exclusive = elapsed - frame[1]
            if exclusive < 0:
                raise RuntimeError("negative exclusive phase interval")
            self.events.append({"phase": phase, "step": self.step,
                                "inclusive_ns": elapsed, "exclusive_ns": exclusive})

    def wrap(self, stack, obj, name, phase, before=None, after=None):
        original = getattr(obj, name)

        def timed(*args, **kwargs):
            metadata = before(*args, **kwargs) if before else {}
            result = self.call(phase, original, *args, **kwargs)
            event = self.events[-1]
            event.update(metadata)
            if after:
                event.update(after(result))
            return result

        stack.enter_context(patch.object(obj, name, timed))


def cells(oracle, config):
    reference = oracle.wall_cell(config)
    result = [replace(reference, cell_id="reference-n16-b512")]
    for cap in (32, 64, 128):
        for budget in (256, 512, 1024):
            result.append(replace(reference, cell_id=f"n{cap}-b{budget}",
                                  engine={**reference.engine, "max_num_seqs": cap,
                                          "budget": budget}))
    return result


def install_timers(timers, stack, oracle, llm, worker):
    engine = llm.llm_engine
    core = engine.engine_core.engine_core
    scheduler = core.scheduler
    manager = scheduler.kv_cache_manager

    def begin_step():
        timers.step += 1
        return {}

    def schedule_before(throttle_prefills=False):
        return {"running_before": len(scheduler.running)}

    def schedule_after(output):
        return {"scheduled": len(output.num_scheduled_tokens),
                "scheduled_tokens": sum(output.num_scheduled_tokens.values())}

    def free_before(request):
        return {"blocks": sum(len(group) for group in
                              manager.get_blocks(request.request_id).blocks)}

    def allocate_after(output):
        return {"blocks": 0 if output is None else sum(map(len, output.blocks)),
                "allocation_failed": output is None}

    timers.wrap(stack, engine, "step", "engine_residual", before=begin_step)
    timers.wrap(stack, scheduler, "schedule", "scheduler",
                before=schedule_before, after=schedule_after)
    timers.wrap(stack, manager, "allocate_slots", "kv_allocate", after=allocate_after)
    timers.wrap(stack, manager, "free", "kv_free", before=free_before)
    for name in ("execute_model", "sample_tokens"):
        timers.wrap(stack, core.model_executor, name, "executor")
    timers.wrap(stack, scheduler, "update_from_output", "scheduler_output")
    timers.wrap(stack, engine.output_processor, "process_outputs", "frontend_output")
    timers.wrap(stack, engine, "add_request", "admission")
    timers.wrap(stack, oracle, "_observe_outputs", "output_collection")


def run_once(oracle, config, cell, model, directory, arm):
    import torch

    directory.mkdir(parents=True, exist_ok=False)
    llm = None
    timers = Timers()
    profiler = cProfile.Profile() if arm == "profile" else None
    try:
        with oracle._engine_environment(directory, capture_oracle=False,
                                        raw_native_path=None):
            llm, worker, sampling, _, _ = oracle._construct_live_engine(
                cell, config, model, directory, capture=False)
            resolved = oracle._resolved_tuple(llm, cell)
            device_before = device_is_absent(getattr(worker, "device", None),
                                             torch.cuda.is_initialized())
            with ExitStack() as stack:
                if arm == "timed":
                    install_timers(timers, stack, oracle, llm, worker)
                if profiler:
                    profiler.enable()
                started = time.perf_counter_ns()
                if arm == "timed":
                    admission, outputs, stopped = timers.call(
                        "driver_residual", oracle._drive_live, llm, worker, sampling, cell)
                else:
                    admission, outputs, stopped = oracle._drive_live(
                        llm, worker, sampling, cell)
                elapsed = time.perf_counter_ns() - started
                if profiler:
                    profiler.disable()
            records = [oracle.step_record_to_json(row) for row in worker.step_records]
            phase_totals = {phase: sum(e["exclusive_ns"] for e in timers.events
                                      if e["phase"] == phase) for phase in PHASES}
            record = {
                "arm": arm, "elapsed_ns": elapsed, "step_count": len(records),
                "output_tokens": sum(map(len, outputs.values())),
                "admission": list(admission), "stopped": stopped, "resolved": resolved,
                "records_sha256": digest(records), "outputs_sha256": digest(outputs),
                "token_identity_ok": all(len(tokens) == 32 and set(tokens) == {512}
                                         for tokens in outputs.values()),
                "phase_totals_ns": phase_totals,
                "reconstruction_relative_error": abs(sum(phase_totals.values())
                                                     - elapsed) / elapsed,
                "device_absent": device_before and device_is_absent(
                    getattr(worker, "device", None), torch.cuda.is_initialized()),
                "worker_device": str(worker.device) if worker.device is not None else None,
                "cuda_initialized": torch.cuda.is_initialized(),
                "torch_version": torch.__version__, "torch_cuda_build": torch.version.cuda,
            }
            for event in timers.events:
                if event["phase"] == "scheduler":
                    row = records[event["step"]]
                    event["record_phase"] = (
                        "decode" if row["scheduled"] and all(
                            q["phase"] == "decode" for q in row["scheduled"]) else "prefill")
            write_once(directory / "summary.json", record)
            write_once(directory / "records.json", records)
            write_once(directory / "outputs.json", outputs)
            if timers.events:
                write_once(directory / "events.json", timers.events)
            if profiler:
                profiler.dump_stats(str(directory / "functions.prof"))
                record["functions"] = profile_rows(profiler)
                write_once(directory / "functions.json", record["functions"])
            return record, timers.events
    finally:
        if llm is not None:
            llm.llm_engine.engine_core.shutdown()
        oracle.reset_configuration()


def device_is_absent(worker_device, cuda_initialized):
    return worker_device is None and not cuda_initialized


def portable_function(filename, line, name):
    if filename.startswith("<") or filename == "~":
        return f"builtin:{name}"
    path = Path(filename)
    parts = path.parts
    positions = [parts.index(marker) for marker in ("vllm", "simllm", "examples")
                 if marker in parts]
    if positions:
        return f"{'/'.join(parts[min(positions):])}:{line}:{name}"
    return f"{path.name}:{line}:{name}"


def profile_rows(profiler):
    rows = []
    for (filename, line, name), (primitive, calls, own, cumulative, _) in (
            pstats.Stats(profiler).stats.items()):
        rows.append({"function": portable_function(filename, line, name),
                     "calls": calls, "primitive_calls": primitive,
                     "self_ns": round(own * 1e9), "cumulative_ns": round(cumulative * 1e9)})
    return sorted(rows, key=lambda row: row["self_ns"], reverse=True)


def attribute_floor():
    obj = SimpleNamespace(value=1)
    batches = []
    for _ in range(7):
        started = time.perf_counter_ns()
        for _ in range(1_000_000):
            _ = obj.value
        batches.append((time.perf_counter_ns() - started) / 1_000_000)
    return {"batch_ns_per_access": batches, "minimum_ns_per_access": min(batches)}


def summarize_cell(cell, runs, events):
    stability_rows = {arm: stability([r["elapsed_ns"] for r in runs[arm]])
                      for arm in ("plain", "timed")}
    all_runs = runs["plain"] + runs["timed"]
    hashes = {(r["records_sha256"], r["outputs_sha256"]) for r in all_runs}
    resolved = all_runs[0]["resolved"]
    expected = {"max_num_seqs": cell.engine["max_num_seqs"],
                "max_num_batched_tokens": cell.engine["budget"],
                "resolved_max_num_scheduled_tokens": cell.engine["budget"],
                "scheduler_block_size": 16, "num_kv_blocks": 2048,
                "enable_prefix_caching": False, "enable_chunked_prefill": True,
                "async_scheduling": False, "pipeline_parallel_size": 1,
                "tensor_parallel_size": 1, "speculative_decoding": False,
                "lora": False, "multimodal": False, "max_model_len": 128,
                "queue_policy": "fcfs", "reserve_mode": "full-isl", "watermark": 0.0,
                "long_prefill_token_threshold": 0, "prefill_schedule_interval": 1,
                "construction_validation_bypass": False,
                "construction_capacity_validation_bypass": False}
    guards = {
        "stability": all(row["passed"] for row in stability_rows.values()),
        "byte_identity": len(hashes) == 1,
        "resolved_configuration": all(r["resolved"] == resolved for r in all_runs)
        and all(resolved[key] == value for key, value in expected.items()),
        "tokens_and_completion": all(r["output_tokens"] == 4096 and
                                     r["token_identity_ok"] and not r["stopped"]
                                     for r in all_runs),
        "admission_identity": all(r["admission"] == [q.request_id for q in cell.requests]
                                  for r in all_runs),
        "device_absent": all(r["device_absent"] for r in all_runs),
        "phase_partition": all(r["reconstruction_relative_error"] <= .05
                               for r in runs["timed"]),
    }
    phase_medians = {p: statistics.median([r["phase_totals_ns"][p] for r in runs["timed"]])
                     for p in PHASES}
    schedules = [e for e in events if e["phase"] == "scheduler"]
    at_cap = [e for e in schedules if e["record_phase"] == "decode"
              and e["running_before"] == cell.engine["max_num_seqs"]
              and e["scheduled"] == cell.engine["max_num_seqs"]]
    groups = defaultdict(list)
    for e in events:
        if e["phase"] in ("kv_allocate", "kv_free"):
            groups[(e["phase"], e["blocks"])].append(e["inclusive_ns"])
    kv_groups = [{"phase": phase, "blocks": blocks, "calls": len(values),
                  "median_ns": statistics.median(values),
                  "median_ns_per_block": statistics.median(values) / blocks if blocks else None}
                 for (phase, blocks), values in sorted(groups.items())]
    plain_median = stability_rows["plain"]["median_ns"]
    overhead = stability_rows["timed"]["median_ns"] / plain_median - 1
    return {"cell_id": cell.cell_id, "cap": cell.engine["max_num_seqs"],
            "budget": cell.engine["budget"], "resolved": resolved,
            "fatal_guards": guards, "stability": stability_rows,
            "plain_ns": [r["elapsed_ns"] for r in runs["plain"]],
            "timed_ns": [r["elapsed_ns"] for r in runs["timed"]],
            "step_count": all_runs[0]["step_count"], "timer_overhead_fraction": overhead,
            "absolute_attribution_valid": abs(overhead) <= .05,
            "phase_medians_ns": phase_medians, "kv_groups": kv_groups,
            "records_sha256": all_runs[0]["records_sha256"],
            "outputs_sha256": all_runs[0]["outputs_sha256"],
            "decode_at_cap_samples": len(at_cap),
            "decode_at_cap_scheduler_ns": statistics.median(
                [e["inclusive_ns"] for e in at_cap]) if at_cap else None,
            "scheduler_ns_per_scheduled_request": sum(e["inclusive_ns"] for e in schedules)
            / sum(e["scheduled"] for e in schedules),
            "max_reconstruction_relative_error": max(
                r["reconstruction_relative_error"] for r in runs["timed"])}


def evaluate_relations(rows):
    relations = []
    indexed = {(r["cap"], r["budget"]): r for r in rows}
    for budget in (256, 512, 1024):
        for low, high in ((32, 64), (64, 128)):
            a, b = [indexed[(n, budget)]["decode_at_cap_scheduler_ns"] for n in (low, high)]
            ratio = b / a if a and b else None
            relations.append({"family": "R1", "instance": f"b{budget}:{low}-to-{high}",
                              "ratio": ratio, "status": "unevaluated" if ratio is None
                              else "pass" if 1 < ratio <= 3 else "fail"})
    for row in rows:
        for phase in ("kv_allocate", "kv_free"):
            groups = [g for g in row["kv_groups"] if g["phase"] == phase and g["blocks"]]
            comparisons = 0
            for a in groups:
                for b in groups:
                    if b["blocks"] < 2 * a["blocks"]:
                        continue
                    comparisons += 1
                    ratio = b["median_ns"] / a["median_ns"]
                    limit = 2 * b["blocks"] / a["blocks"]
                    relations.append({"family": "R2", "instance":
                                      f"{row['cell_id']}:{phase}:{a['blocks']}-to-{b['blocks']}",
                                      "ratio": ratio, "upper_bound": limit,
                                      "status": "pass" if 1 < ratio <= limit else "fail"})
            if not comparisons:
                relations.append({"family": "R2", "instance": f"{row['cell_id']}:{phase}",
                                  "status": "unevaluated", "reason": "no twofold block groups"})
    return relations


def require_versions(distribution, module):
    if (distribution, module) != ("0.27.1+cpu", "0.27.1"):
        raise RuntimeError(f"vLLM pin mismatch: distribution={distribution}, module={module}")


def preflight(oracle, config, model):
    import importlib.metadata

    import vllm
    from vllm.v1.core.sched import scheduler

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    files = ["examples/surrogate_conformance_v1/run_study.py",
             "examples/surrogate_conformance_v1/study_config.json"]
    for name in files:
        frozen = subprocess.check_output(["git", "show", f"{FREEZE}:{name}"], cwd=ROOT)
        if not text_matches((ROOT / name).read_bytes(), frozen):
            raise RuntimeError(f"frozen source changed: {name}")
    subprocess.run(["git", "merge-base", "--is-ancestor", FREEZE, "HEAD"], cwd=ROOT, check=True)
    subprocess.run(["git", "merge-base", "--is-ancestor", AMENDMENT, "HEAD"], cwd=ROOT, check=True)
    subprocess.run(["git", "merge-base", "--is-ancestor", CONTROLLED_FREEZE, "HEAD"],
                   cwd=ROOT, check=True)
    actual = sha(Path(scheduler.__file__))
    if config["oracle"]["scheduler_sha256"] not in text_hashes(
            Path(scheduler.__file__).read_bytes()).values():
        raise RuntimeError("scheduler source hash mismatch")
    distribution = importlib.metadata.version("vllm")
    require_versions(distribution, vllm.__version__)
    if config["oracle"]["model_config_sha256"] not in text_hashes(
            (model / "config.json").read_bytes()).values():
        raise RuntimeError("model configuration hash mismatch")
    if model.name != config["oracle"]["model_revision"]:
        raise RuntimeError("select the exact pinned model revision snapshot")
    return {"distribution_version": distribution, "module_version": vllm.__version__,
            "scheduler_sha256": actual, "source_sha256": {n: sha(ROOT / n) for n in files},
            "source_text_hashes": {n: text_hashes((ROOT / n).read_bytes()) for n in files},
            "workload_sha256": digest([vars(r) for r in oracle.wall_cell(config).requests]),
            "model_revision": model.name, "model_config_sha256": sha(model / "config.json")}


def run(args):
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from examples.surrogate_conformance_v1 import run_study as oracle

    config = oracle.load_config()
    attempt = args.run_root / args.attempt
    attempt.mkdir(parents=True, exist_ok=False)
    provenance = preflight(oracle, config, args.model)
    provenance["implementation_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    host_before = host_snapshot(torch)
    write_once(attempt / "host_before.json", host_before)
    floor = attribute_floor()
    write_once(attempt / "attribute_floor.json", floor)
    print("Physical floor: N *", floor["minimum_ns_per_access"], "ns per attribute access.",
          "Ceiling: each exclusive phase <= its enclosing interval.", flush=True)
    summaries = []
    for cell in cells(oracle, config):
        runs = {"plain": [], "timed": []}
        events = []
        for arm, repetitions in runs.items():
            for index in range(9):
                name = f"warmup-{index + 1}" if index < 2 else f"run-{index - 1}"
                record, observed = run_once(oracle, config, cell, args.model,
                                            attempt / cell.cell_id / arm / name, arm)
                if index >= 2:
                    repetitions.append(record)
                    events.extend(observed)
            print(cell.cell_id, arm, stability([r["elapsed_ns"] for r in repetitions]), flush=True)
        summary = summarize_cell(cell, runs, events)
        write_once(attempt / cell.cell_id / "cell.json", summary)
        summaries.append(summary)
    profile, _ = run_once(oracle, config, cells(oracle, config)[0], args.model,
                          attempt / "profile", "profile")
    host_after = host_snapshot(torch)
    write_once(attempt / "host_after.json", host_after)
    violations = [f"{r['cell_id']}:{g}" for r in summaries
                  for g, passed in r["fatal_guards"].items() if not passed]
    if not host_guard(host_before, host_after):
        violations.append("controlled_host")
    relations = evaluate_relations(summaries)
    cpu_model = next((line.split(":", 1)[1].strip() for line in
                      Path("/proc/cpuinfo").read_text().splitlines()
                      if line.startswith("model name")), "undisclosed")
    result = {"schema": "simllm-vllm-step-loop-cost-v1",
              "expectation_commit": subprocess.check_output(
                  ["git", "rev-parse", CONTROLLED_FREEZE], cwd=ROOT, text=True).strip(),
              "distribution_amendment_commit": AMENDMENT,
              "original_expectation_commit": FREEZE,
              "host_control": {"protocol": "eight-core-affinity-single-thread-v1",
                               "before": host_before, "after": host_after,
                               "passed": host_guard(host_before, host_after),
                               "no_concurrent_local_suite": args.no_concurrent_local_suite,
                               "background": "Other wave workers use different cores; "
                               "other users' background jobs remain unpinned."},
              "attempt": args.attempt, "provenance": provenance,
              "status": "void" if violations else "nonvoid",
              "fatal_violations": violations, "attribute_floor": floor,
              "machine": {"cpu_model": cpu_model, "python": platform.python_version(),
                          "logical_cpus": os.cpu_count(),
                          "affinity": sorted(os.sched_getaffinity(0)),
                          "clock": vars(time.get_clock_info("perf_counter"))},
              "cells": summaries, "relation_observations": relations,
              "behavioral_score": None if violations else {
                  family: {status: sum(r["family"] == family and r["status"] == status
                                       for r in relations)
                           for status in ("pass", "fail", "unevaluated")}
                  for family in ("R1", "R2")},
              "profile_top_self": profile["functions"][:25],
              "profile_top_cumulative": sorted(profile["functions"],
                                                key=lambda r: r["cumulative_ns"],
                                                reverse=True)[:25]}
    write_once(attempt / "results.json", result)
    print(json.dumps({"status": result["status"], "fatal_violations": violations,
                      "behavioral_score": result["behavioral_score"]}), flush=True)
    return 2 if violations else 0


PHASE_LABELS = {
    "executor": "simulated executor and\nmodel-runner stub",
    "scheduler": "scheduler, excluding\ncache allocate/free",
    "kv_allocate": "KV cache allocation",
    "frontend_output": "frontend output processing",
    "scheduler_output": "scheduler output update,\nexcluding cache free",
    "admission": "request admission",
    "driver_residual": "driver residual",
    "engine_residual": "engine-step residual",
    "output_collection": "driver output collection\nand validation",
    "kv_free": "KV cache free",
}


def _median(value):
    """Median of a repetition list, or the median field of a stability record."""
    if isinstance(value, dict):
        for key in ("median_ns", "median"):
            if key in value:
                return value[key]
    if isinstance(value, (list, tuple)):
        ordered = sorted(value)
        middle = len(ordered) // 2
        return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
    return value


def _arm_drift(loops):
    """Frozen R4 statistic: the largest separation among the first-three,
    last-three and seven-run medians, divided by the seven-run median."""
    first = statistics.median(loops[:3])
    last = statistics.median(loops[-3:])
    whole = statistics.median(loops)
    return max(abs(first - whole), abs(last - whole), abs(first - last)) / whole


def plot(result, output):
    """Draw exclusive loop costs, scheduler scaling and the repeated-loop drift.

    The upper axis of panel A divides complete-loop phase medians by the
    engine step count. Panel C shows the seven measured whole-workload loops
    of the arm with the largest frozen drift statistic next to the reference
    plain arm, so the reason for a void status is visible in the figure.
    Retained values from a void run are diagnostic only.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reference = result["cells"][0]
    steps = reference["step_count"]
    phases = sorted(reference["phase_medians_ns"].items(), key=lambda kv: kv[1])
    total = sum(ns for _, ns in phases)
    fig = plt.figure(figsize=(7, 11.2))
    left = fig.add_axes((0.38, 0.685, 0.60, 0.225))
    right = fig.add_axes((0.12, 0.395, 0.86, 0.195))
    bottom = fig.add_axes((0.12, 0.115, 0.86, 0.17))
    labels = [PHASE_LABELS.get(name, name.replace("_", " ")) for name, _ in phases]
    values = [ns / 1e6 for _, ns in phases]
    bars = left.barh(labels, values, color="#1f77b4")
    for bar, (_, ns) in zip(bars, phases):
        left.text(bar.get_width() + 0.4, bar.get_y() + bar.get_height() / 2,
                  f"{100 * ns / total:.1f}%", va="center", fontsize=8.5)
    left.set_xlabel(f"Exclusive time per 128-request loop (ms)\n{steps} engine steps per loop",
                    fontsize=9)
    upper = left.secondary_xaxis("top", functions=(lambda ms: ms * 1e3 / steps,
                                                   lambda us: us * steps / 1e3))
    upper.set_xlabel("Amortized time per engine step (µs)", fontsize=9)
    upper.tick_params(labelsize=8.5)
    fig.text(0.04, 0.955, f"A  Ranked exclusive phases: cap {reference['cap']}, "
             f"token budget {reference['budget']}", fontsize=10, weight="bold")
    left.tick_params(labelsize=8.5)
    left.set_ylim(-0.5, len(phases) - 0.5)
    for label in left.get_yticklabels():
        label.set_linespacing(1.0)
    left.set_xlim(0, max(values) * 1.18)
    left.spines[["right"]].set_visible(False)

    cells = sorted(result["cells"], key=lambda cell: (cell["budget"], cell["cap"]))
    budgets = sorted({cell["budget"] for cell in cells})
    colors = dict(zip(budgets, ("#d62728", "#1f77b4", "#2ca02c")))
    markers = dict(zip(budgets, ("s", "o", "^")))
    caps = sorted({cell["cap"] for cell in cells})
    for budget in budgets:
        selected = [cell for cell in cells if cell["budget"] == budget
                    and cell.get("decode_at_cap_scheduler_ns") is not None]
        right.plot([cell["cap"] for cell in selected],
                   [cell["decode_at_cap_scheduler_ns"] / 1e3 for cell in selected],
                   marker=markers[budget], markersize=6, markerfacecolor="white",
                   markeredgewidth=1.2, linewidth=1.4, color=colors[budget],
                   label=f"Token budget {budget}", zorder=3)
    anchor = reference["decode_at_cap_scheduler_ns"] / 1e3
    right.plot(caps, [anchor * cap / reference["cap"] for cap in caps], color="gray",
               linewidth=1, linestyle="--", label="Linear reference", zorder=2)
    right.set(xscale="log", yscale="log", xticks=caps,
              yticks=(200, 400, 800, 1600), ylim=(170, 1900))
    right.set_xlabel("Concurrency cap (running requests)", fontsize=9)
    right.set_ylabel("Inclusive scheduler time\nper all-decode step at cap (µs)", fontsize=9)
    right.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    right.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    right.get_xaxis().set_minor_formatter(plt.NullFormatter())
    right.get_yaxis().set_minor_formatter(plt.NullFormatter())
    handles, legend_labels = right.get_legend_handles_labels()
    order = sorted(range(len(budgets)), key=lambda i: (budgets[i] != 512, budgets[i]))
    order.append(len(budgets))
    right.legend([handles[i] for i in order], [legend_labels[i] for i in order],
                 fontsize=8, loc="upper left", ncol=2, frameon=False)
    fig.text(0.04, 0.615, "B  Scheduler cost grows with running requests", fontsize=10,
             weight="bold")
    right.tick_params(labelsize=8.5)
    right.grid(True, which="major", alpha=0.25)
    right.spines[["top", "right"]].set_visible(False)

    arms = []
    for cell in result["cells"]:
        for arm, key in (("plain", "plain_ns"), ("instrumented", "timed_ns")):
            loops = [ns / 1e6 for ns in cell.get(key, ())]
            if len(loops) >= 7:
                arms.append((_arm_drift(loops), cell, arm, loops))
    if arms:
        worst = max(arms, key=lambda item: item[0])
        drift, cell, arm, loops = worst
        ref_loops = [ns / 1e6 for ns in reference["plain_ns"]]
        index = list(range(1, len(loops) + 1))
        median = statistics.median(loops)
        bottom.axhspan(0.9 * median, 1.1 * median, color="#d62728", alpha=0.10,
                       zorder=1, label="Frozen 10% band around the seven-run median")
        bottom.axhline(median, color="#d62728", linewidth=0.8, linestyle=":", zorder=2)
        bottom.plot(index, loops, marker="o", color="#d62728", linewidth=1.4,
                    markersize=6, markerfacecolor="white", markeredgewidth=1.2, zorder=3,
                    label=f"Worst arm: {cell['cap']} requests, budget {cell['budget']}, "
                          f"{arm} ({100 * drift:.2f}% drift)")
        bottom.plot(index[:len(ref_loops)], ref_loops, marker="s", color="#1f77b4",
                    linewidth=1.2, markersize=5, markerfacecolor="white",
                    markeredgewidth=1.1, zorder=3,
                    label=f"Reference arm: cap {reference['cap']}, budget "
                          f"{reference['budget']}, plain "
                          f"({100 * _arm_drift(ref_loops):.2f}% drift)")
        first = statistics.median(loops[:3])
        last = statistics.median(loops[-3:])
        bottom.annotate(f"first three: median {first:.1f} ms",
                        xy=(2, first), xytext=(3.4, first + 0.12 * median), fontsize=8,
                        color="#d62728", arrowprops={"arrowstyle": "-", "color": "#d62728",
                                    "linewidth": 0.7})
        bottom.annotate(f"last three: median {last:.1f} ms",
                        xy=(6, last), xytext=(4.2, last - 0.2 * median), fontsize=8,
                        color="#d62728", arrowprops={"arrowstyle": "-", "color": "#d62728",
                                    "linewidth": 0.7})
        bottom.set_xticks(index)
        bottom.set_xlabel("Measured whole-workload loop, in execution order", fontsize=9)
        bottom.set_ylabel("Loop wall time (ms)", fontsize=9)
        low = min(loops + ref_loops)
        high = max(loops + ref_loops)
        bottom.set_ylim(0.72 * low, 1.14 * high)
        bottom.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2),
                      ncol=1, frameon=False)
        bottom.tick_params(labelsize=8.5)
        bottom.grid(True, which="major", alpha=0.25)
        bottom.spines[["top", "right"]].set_visible(False)
    fig.text(0.04, 0.305, "C  Repeated-loop drift on the worst arm against the fatal guard",
             fontsize=10, weight="bold")
    fig.suptitle("vLLM 0.27.1 CPU engine loop: 128-request workload\n"
                 f"Run status {result['status'].upper()}: diagnostic timings", fontsize=11,
                 y=0.992)
    output.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(output / f"phase_cost.{extension}", dpi=220)
    plt.close(fig)


def publish(attempt, output, retained=()):
    raw = attempt / "results.json"
    result = json.loads(raw.read_text())
    functions = profile_rows(str(attempt / "profile" / "functions.prof"))
    result["profile_top_self"] = functions[:25]
    result["profile_top_cumulative"] = sorted(
        functions, key=lambda row: row["cumulative_ns"], reverse=True)[:25]
    result["publication"] = {
        "raw_results_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
        "profile_sha256": hashlib.sha256(
            (attempt / "profile" / "functions.prof").read_bytes()).hexdigest(),
        "correction": "Preserve simllm/adapters/vllm as a project-qualified profile path; "
                      "reproject the retained profile without a new run or any timing change.",
    }
    result["publication"]["retained_attempts"] = [
        {"artifact": path.name, "text_sha256": text_hashes(path.read_bytes()),
         **{key: value for key, value in json.loads(path.read_bytes()).items()
            if key in ("attempt", "status", "expectation_commit", "fatal_violations")}}
        for path in retained]
    write_once(output, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    execute = commands.add_parser("run")
    execute.add_argument("--run-root", type=Path, default=os.getenv("SIMLLM_DATA_ROOT"))
    execute.add_argument("--model", type=Path, default=os.getenv("SIMLLM_VLLM_MODEL"))
    execute.add_argument("--attempt", default="attempt-001")
    execute.add_argument("--no-concurrent-local-suite", action="store_true", required=True,
                         help="attest no local suite, plot or other study runs during timing")
    render = commands.add_parser("plot")
    render.add_argument("--results", type=Path, required=True)
    render.add_argument("--output", type=Path, default=STUDY / "figures")
    publication = commands.add_parser("publish")
    publication.add_argument("--attempt-dir", type=Path, required=True)
    publication.add_argument("--output", type=Path, default=STUDY / "controlled_results.json")
    publication.add_argument("--retained-result", type=Path, action="append",
                             default=[STUDY / "results.json"])
    args = parser.parse_args()
    if args.command == "plot":
        plot(json.loads(args.results.read_text()), args.output)
        return 0
    if args.command == "publish":
        publish(args.attempt_dir, args.output, args.retained_result)
        return 0
    if args.run_root is None or args.model is None:
        parser.error("configure SIMLLM_DATA_ROOT and SIMLLM_VLLM_MODEL or pass --run-root/--model")
    interpreter = os.getenv("SIMLLM_VLLM_PYTHON")
    if not interpreter:
        parser.error("configure SIMLLM_VLLM_PYTHON with the supplied pinned interpreter")
    os.environ.update({name: "1" for name in THREAD_ENV})
    if os.path.abspath(sys.executable) != os.path.abspath(interpreter):
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        return subprocess.call([interpreter, "-B", str(Path(__file__)), *sys.argv[1:]],
                               env=environment)
    args.run_root = args.run_root.resolve()
    if args.run_root == ROOT or ROOT in args.run_root.parents:
        parser.error("--run-root must be outside the repository")
    temp = args.run_root / "temporary"
    temp.mkdir(parents=True, exist_ok=True)
    os.environ.update(TMPDIR=str(temp), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      VLLM_CACHE_ROOT=str(args.run_root / "cache"))
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
