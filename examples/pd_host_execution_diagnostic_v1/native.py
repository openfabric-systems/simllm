"""Run the unchanged native session with optional passive host observation."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
import traceback
import weakref
from dataclasses import asdict
from pathlib import Path

from examples.pd_host_execution_diagnostic_v1.timing import (
    PhaseCounters,
    counter_delta,
    install,
    profiled_call,
)
from examples.pd_session_identity_v1.run_study import exact_json_bytes, sha, state_snapshot, write
from examples.pd_session_target_scale_v1 import native as target
from examples.pd_session_v1 import run_study as baseline

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def source_snapshot(frozen, package):
    origins = {name: str(Path(importlib.import_module(
        name.removesuffix(".py").replace("/", ".")).__file__).resolve())
        for name in frozen["source_sha256"]}
    native_origins = {name: str(Path(importlib.import_module(
        name.removesuffix(".py").replace("/", ".")).__file__).resolve())
        for name in frozen["native_source_sha256"]}
    return {
        "origins": origins,
        "native_origins": native_origins,
        "repository": {name: sha(Path(path).read_bytes()) for name, path in origins.items()},
        "native": {name: sha(Path(path).read_bytes()) for name, path in native_origins.items()},
    }


def source_admission(snapshot, frozen, package):
    if snapshot["repository"] != frozen["source_sha256"]:
        raise ValueError("repository source differs from diagnostic freeze")
    if snapshot["native"] != frozen["native_source_sha256"]:
        raise ValueError("native source differs from diagnostic freeze")
    if any(Path(path) != (ROOT / name).resolve() for name, path in snapshot["origins"].items()):
        raise ValueError("native process imported another repository checkout")
    if any(Path(path) != (package.parent / name).resolve() for name, path in snapshot["native_origins"].items()):
        raise ValueError("native process imported another vLLM source module")


def selection_snapshot(session):
    rows = []
    for engine in target.all_engines(session):
        config = engine.step_sink.config
        frontend = engine.llm.llm_engine.vllm_config
        provider = config.provider
        rows.append({"engine_id": engine.engine_id,
            "provider_type": type(provider).__module__ + "." + type(provider).__qualname__,
            "provider_state": vars(provider).copy(), "gpu": asdict(config.gpu),
            "host_model": asdict(config.host_model),
            "collective_envelope": config.collective_fixed_cost_envelope,
            "collective_arm": config.collective_fixed_cost_arm,
            "resolved_collective_profile": state_snapshot(config.resolved_collective_latency_profile),
            "max_num_seqs": frontend.scheduler_config.max_num_seqs,
            "tensor_parallel_size": frontend.parallel_config.tensor_parallel_size,
            "executor_mode": engine.executor.config.mode})
    return rows


def engine_identity(engine):
    value = target.engine_identity(engine)
    frontend = engine.llm.llm_engine
    value.update(scheduler_object_id=id(frontend.engine_core.engine_core.scheduler),
                 output_processor_object_id=id(frontend.output_processor))
    return value


def sink_snapshot(session, frozen):
    return [{"engine_id": engine.engine_id,
             "collections": {name: [asdict(row) for row in getattr(engine.step_sink, name)]
                             for name in frozen["sink_collections"]}}
            for engine in target.all_engines(session)]


def execute(args):
    import torch
    import vllm
    from huggingface_hub import hf_hub_download

    from simllm.adapters.vllm.pd_session import VllmDisaggregatedSession
    from simllm.core.step import step_record_to_json

    frozen = json.loads((HERE / "expectations.json").read_bytes())
    package = Path(vllm.__file__).resolve().parent
    if package != args.vllm_source.resolve() or vllm.__version__ != frozen["frontend"]["version"]:
        raise ValueError("native package origin/version differs")
    config = Path(hf_hub_download(baseline.MODEL_ID, "config.json",
                                 revision=baseline.MODEL_REVISION, local_files_only=True)).resolve()
    if sha(config.read_bytes()) != frozen["frontend"]["model_config_sha256"]:
        raise ValueError("native model config differs")
    sources_before = source_snapshot(frozen, package)
    source_admission(sources_before, frozen, package)
    if sys.getprofile() is not None:
        raise RuntimeError("native process already has a profile hook")
    cuda_before = torch.cuda.is_initialized()
    clock = target.ObservedClock()
    construction, weak_objects = [], []
    construction_start = time.perf_counter_ns()

    def observe(engine):
        weak_objects.extend(weakref.ref(value) for value in (
            engine, engine.llm, engine.executor, *engine.executor._workers))
        row = {"index": len(construction), "identity": engine_identity(engine),
               "elapsed_ns": time.perf_counter_ns() - construction_start,
               "current_rss_kib": baseline._rss_kib(), "clock_ps": clock.now_ps,
               "all_observed_objects_alive": all(ref() is not None for ref in weak_objects)}
        construction.append(row)
        target.append_progress(args.output_root / "construction-progress.jsonl", row)
        if row["current_rss_kib"] >= frozen["limits"]["rss_cap_kib"]:
            raise RuntimeError("frozen resident-memory stop reached")

    counters = PhaseCounters() if args.arm == "instrumented" else None
    (args.output_root / "profiles").mkdir()
    with VllmDisaggregatedSession(
        baseline._session_config(args.output_root / "engine-work"),
        clock=clock, construction_observer=observe,
    ) as session:
        before = [engine_identity(engine) for engine in target.all_engines(session)]
        selections_before = selection_snapshot(session)
        try:
            if counters is not None:
                install(counters, session, frozen["instrumentation"]["entries"])
                counters.check_bindings()
            initial = {} if counters is None else counters.snapshot()
            baseline_wall, baseline_cpu = time.perf_counter_ns(), time.process_time_ns()
            controls = target.baseline_control(
                session, {"serial_cells": frozen["historical_control_specs"]}, args.output_root)
            baseline_cpu = time.process_time_ns() - baseline_cpu
            baseline_wall = time.perf_counter_ns() - baseline_wall
            control_counters = ({} if counters is None
                                else counter_delta(initial, counters.snapshot()))
            cells, profiles = [], []
            for spec in frozen["cells"]:
                if counters is not None:
                    counters.check_bindings()
                prior = {} if counters is None else counters.snapshot()
                host = {}

                def call(spec=spec, host=host):
                    wall, cpu = time.perf_counter_ns(), time.process_time_ns()
                    try:
                        return target.run_cell(session, spec, args.output_root)
                    finally:
                        host["process_cpu_ns"] = time.process_time_ns() - cpu
                        host["wall_ns"] = time.perf_counter_ns() - wall

                selected = counters is not None and spec["profile_in_instrumented_arm"]
                profile_path = args.output_root / "profiles" / (spec["id"] + ".pstats")
                cell, functions = profiled_call(call, profile_path=profile_path if selected else None)
                profile = None
                if selected:
                    export = profile_path.with_suffix(".json")
                    write(export, {"schema": "simllm-host-call-profile-v1", "cell_id": spec["id"],
                                   "functions": functions})
                    profile = {"cell_id": spec["id"],
                               "raw_path": profile_path.relative_to(args.output_root).as_posix(),
                               "raw_sha256": sha(profile_path.read_bytes()),
                               "export_path": export.relative_to(args.output_root).as_posix(),
                               "export_sha256": sha(export.read_bytes()), "functions": len(functions)}
                    profiles.append(profile)
                cells.append({"cell": cell, "host_call": host,
                              "phase_counters": {} if counters is None else counter_delta(prior, counters.snapshot()),
                              "profile": profile})
            totals = {} if counters is None else counters.snapshot()
        finally:
            if counters is not None and not counters.closed:
                try:
                    counters.close()
                finally:
                    write(args.output_root / "probe-restoration.json", counters.restored)
        after = [engine_identity(engine) for engine in target.all_engines(session)]
        steps = [{"engine_id": engine.engine_id, "record": step_record_to_json(record),
                  "result": asdict(result)}
                 for engine in target.all_engines(session)
                 for record, result in zip(engine.executor.step_records,
                                           engine.executor.step_results, strict=True)]
        sinks = sink_snapshot(session, frozen)
        sources_after = source_snapshot(frozen, package)
        source_admission(sources_after, frozen, package)
        value = {
            "schema": "simllm-pd-host-execution-native-v1", "arm": args.arm,
            "identity": {"pid": os.getpid(), "interpreter": sys.executable,
                         "package": str(package), "version": vllm.__version__,
                         "config_path": str(config), "config_sha256": sha(config.read_bytes()),
                         "sources_before": sources_before, "sources_after": sources_after,
                         "selections_before": selections_before, "selections_after": selection_snapshot(session)},
            "construction": construction, "engines_before": before, "engines_after": after,
            "weak_objects_alive_after": all(ref() is not None for ref in weak_objects),
            "baseline_controls": controls,
            "baseline_host_call": {"wall_ns": baseline_wall, "process_cpu_ns": baseline_cpu},
            "baseline_counters": control_counters, "cells": cells,
            "unique_steps": steps, "clock_advances": clock.advances,
            "sink_outcomes": sinks, "phase_counters": totals, "profiles": profiles,
            "probe_restoration": [] if counters is None else counters.restored,
            "probe_stack_empty": counters is None or not counters.stack,
            "profile_hook_restored": sys.getprofile() is None,
            "cuda_initialized_before": cuda_before,
            "cuda_initialized_after": torch.cuda.is_initialized(),
            "backend_runs": sum(row.backend_runs for engine in target.all_engines(session)
                                for row in engine.step_sink.locality_outcomes),
            "final_clock_ps": clock.now_ps,
        }
        # Dataclass tuples have JSON array projections, as in the native baseline.
        value = json.loads(json.dumps(value, allow_nan=False))
        write(args.output_root / "native.json", value)
        write(args.output_root / "receipt.json",
              {"pid": os.getpid(), "native_sha256": sha(exact_json_bytes(value))})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("uninstrumented", "instrumented"), required=True)
    parser.add_argument("--vllm-source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        execute(args)
    except Exception:
        (args.output_root / "native-failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
