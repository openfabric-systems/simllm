"""Capture fresh native processes for the frozen independent-engine sweep."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import weakref
from dataclasses import asdict, replace
from pathlib import Path

from examples.independent_engine_completion_v1.common import (
    HERE,
    PUBLICATIONS,
    ROOT,
    exact_json_bytes,
    json_value,
    read,
    sha,
    sink_publications,
    source_snapshot,
    write,
)
from examples.pd_session_target_scale_v1.native import (
    ObservedClock,
    all_engines,
    append_progress,
    engine_identity,
    request_row,
)
from examples.pd_session_v1 import run_study as baseline


def selections(session):
    rows = []
    for engine in all_engines(session):
        config = engine.step_sink.config
        native = engine.llm.llm_engine.vllm_config
        rows.append({"engine_id": engine.engine_id, "provider_type": type(config.provider).__module__ + "." + type(config.provider).__qualname__,
                     "provider_state": vars(config.provider).copy(), "gpu": asdict(config.gpu),
                     "host_model": asdict(config.host_model), "selected_precision": dict(config.selected_precision_levels),
                     "collective_profile": asdict(config.resolved_collective_latency_profile),
                     "collective_arm": config.collective_fixed_cost_arm,
                     "collective_envelope": config.collective_fixed_cost_envelope,
                     "max_num_seqs": native.scheduler_config.max_num_seqs,
                     "tensor_parallel_size": native.parallel_config.tensor_parallel_size,
                     "pipeline_parallel_size": native.parallel_config.pipeline_parallel_size,
                     "data_parallel_size": native.parallel_config.data_parallel_size,
                     "async_scheduling": native.scheduler_config.async_scheduling,
                     "policy": native.scheduler_config.policy, "executor_mode": engine.executor.config.mode})
    return rows


def run_cell(session, spec, output):
    from simllm.adapters.vllm.pd_session import VllmPdRequest
    from simllm.core import DeclaredKvHandoffPolicy

    start_ps, start_ns = session.clock.now_ps, time.perf_counter_ns()
    advance_start = len(session.clock.advances)
    record_starts = {engine.engine_id: len(engine.executor.step_records) for engine in all_engines(session)}
    runtime = session.engine_runtime
    event_start = len(runtime.events) if runtime else 0
    visit_start = len(runtime.visits) if runtime else 0
    requests = [VllmPdRequest(
        spec["id"] + ":request-" + str(index), baseline._prompt_tokens()[:prompt],
        spec["decode_output_tokens"], start_ps + spec["arrival_offsets_ps"][index],
    ) for index, prompt in enumerate(spec["prompt_tokens"])]
    result = session.run_requests(requests, handoff_policy=DeclaredKvHandoffPolicy(spec["handoff_ps"]))
    rows = [request_row(request) for request in result.requests]
    value = {"id": spec["id"], "start_ps": start_ps, "end_ps": session.clock.now_ps,
             "host_wall_ns": time.perf_counter_ns() - start_ns, "requests": rows,
             "record_starts": record_starts,
             "clock_advance_indices": list(range(advance_start, len(session.clock.advances))),
             "record_stops": {engine.engine_id: len(engine.executor.step_records) for engine in all_engines(session)},
             "event_indices": list(range(event_start, len(runtime.events))) if runtime else [],
             "visit_indices": list(range(visit_start, len(runtime.visits))) if runtime else [],
             "prefill_batches": [list(row) for row in result.prefill_batches],
             "decode_batches": [list(row) for row in result.decode_batches]}
    write(output / (spec["id"] + ".json"), value)
    for row in rows:
        append_progress(output / "request-progress.jsonl", row)
    return value


def execute(args):
    import torch
    import vllm
    from huggingface_hub import hf_hub_download

    from simllm.adapters.vllm.pd_session import VllmDisaggregatedSession
    from simllm.core import DeclaredKvHandoffPolicy
    from simllm.core.step import step_record_to_json

    frozen = json.loads((HERE / "expectations.json").read_bytes())
    process = next(row for row in frozen["native_processes"] if row["id"] == args.process)
    package = Path(vllm.__file__).resolve().parent
    config_path = Path(hf_hub_download(baseline.MODEL_ID, "config.json", revision=baseline.MODEL_REVISION,
                                      local_files_only=True)).resolve()
    sources = source_snapshot(frozen)
    expected_sources = read(args.source_manifest)
    if (package != args.vllm_source.resolve() or vllm.__version__ != frozen["native_environment"]["vllm_version"]
            or sources["native"]["sha256"] != frozen["native_source_sha256"]
            or sources["repository"]["sha256"] != expected_sources
            or any(Path(path) != (ROOT / name).resolve() for name, path in sources["repository"]["origins"].items())
            or any(Path(path) != (package.parent / name).resolve() for name, path in sources["native"]["origins"].items())
            or sha(config_path.read_bytes()) != frozen["native_environment"]["model_config_sha256"]):
        raise ValueError("native source, origin or model config differs from admitted protocol")
    clock, constructions, checkpoints, weak_objects = ObservedClock(), [], [], []
    cuda_before = torch.cuda.is_initialized()
    construction_start = time.perf_counter_ns()

    def construction(engine):
        weak_objects.extend(weakref.ref(value) for value in (engine, engine.llm, engine.executor, *engine.executor._workers))
        row = {"identity": engine_identity(engine), "elapsed_ns": time.perf_counter_ns() - construction_start,
               "current_rss_kib": baseline._rss_kib(), "peak_rss_kib": baseline._peak_rss_kib(),
               "clock_ps": clock.now_ps, "all_observed_alive": all(ref() is not None for ref in weak_objects)}
        constructions.append(row)
        append_progress(args.output_root / "construction-progress.jsonl", row)

    def observe(phase, engine, native):
        publications = sink_publications(engine)
        row = {"index": len(checkpoints), "phase": phase, "engine_id": engine.engine_id, "at_ps": clock.now_ps,
               "step_index": engine.executor._runtime.step_index if phase == "before-submit" else engine.executor.step_records[-1].step_index,
               "record_count": len(engine.executor.step_records), "result_count": len(engine.executor.step_results),
               "sink_counts": {name: len(publications[name]) for name in PUBLICATIONS},
               "sink_sha256": {name: sha(exact_json_bytes(publications[name])) for name in PUBLICATIONS}, "native": native}
        checkpoints.append(row)
        append_progress(args.output_root / "native-checkpoints.jsonl", row)

    session_config = replace(baseline._session_config(
        args.output_root / "engine-work", prefill_engines=process["prefill_engines"], decode_engines=process["decode_engines"]),
        engine_timing=process["mode"], max_num_seqs=1)
    with VllmDisaggregatedSession(session_config, clock=clock, construction_observer=construction,
                                  completion_observer=observe if process["mode"] == "independent" else None) as session:
        retained_before = [engine_identity(engine) for engine in all_engines(session)]
        selected_before = selections(session)
        controls = []
        if process["baseline_controls_first"]:
            for spec in frozen["baseline_control"]["accepted_cells"]:
                name = f"prompt-{spec['prompt_tokens']}-handoff-{spec['handoff_ps']}"
                result = session.run_request(name, baseline._prompt_tokens()[:spec["prompt_tokens"]], decode_output_tokens=4,
                                             handoff_policy=DeclaredKvHandoffPolicy(spec["handoff_ps"]))
                comparison = result.to_comparison_json()
                controls.append({"id": name, "comparison": comparison, "comparison_sha256": sha(exact_json_bytes(comparison)),
                                 **request_row(result)})
                append_progress(args.output_root / "request-progress.jsonl", controls[-1])
        cells = [run_cell(session, spec, args.output_root) for spec in process["cells"]]
        steps = [{"engine_id": engine.engine_id, "record": step_record_to_json(record), "result": asdict(result)}
                 for engine in all_engines(session) for record, result in zip(
                     engine.executor.step_records, engine.executor.step_results, strict=True)]
        runtime = session.engine_runtime
        # Capture actual receipt/event/visit projections before teardown.
        projections = {"events": [asdict(row) for row in runtime.events] if runtime else [],
                       "visits": [asdict(row) for row in runtime.visits] if runtime else [],
                       "completed": [{"receipt": asdict(receipt), "result": asdict(result)}
                                     for receipt, result in runtime.results] if runtime else []}
        value = {"schema": "simllm-independent-engine-completion-native-v1", "process_id": process["id"],
                 "authority": session.timing_authority, "sources_before": sources, "sources_after": source_snapshot(frozen),
                 "identity": {"pid": os.getpid(), "interpreter": sys.executable, "package": str(package),
                              "version": vllm.__version__, "config_sha256": sha(config_path.read_bytes()),
                              "config_path": str(config_path), "clock_object_id": id(clock),
                              "cuda_before": cuda_before, "cuda_after": torch.cuda.is_initialized(),
                              "packet_backend_runs": sum(row.backend_runs for engine in all_engines(session) for row in engine.step_sink.locality_outcomes),
                              "all_observed_alive": all(ref() is not None for ref in weak_objects),
                              "unfinished_after": [engine.llm.llm_engine.has_unfinished_requests() for engine in all_engines(session)],
                              "native_has_work_after": [engine.llm.llm_engine.engine_core.engine_core.scheduler.has_requests()
                                                        for engine in all_engines(session)]},
                 "retained_before": retained_before, "retained_after": [engine_identity(engine) for engine in all_engines(session)],
                 "selected_before": selected_before, "selected_after": selections(session), "construction": constructions,
                 "controls": controls, "cells": cells, "steps": steps, "clock_advances": clock.advances,
                 "checkpoints": checkpoints, "projections": projections,
                 "sinks": {engine.engine_id: sink_publications(engine) for engine in all_engines(session)},
                 "final_clock_ps": clock.now_ps, "runtime_failure": runtime.failure if runtime else None}
    write(args.output_root / "native.json", json_value(value))
    write(args.output_root / "receipt.json", {"pid": os.getpid(), "native_sha256": sha((args.output_root / "native.json").read_bytes())})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--process", required=True)
    parser.add_argument("--vllm-source", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    execute(parser.parse_args())


if __name__ == "__main__":
    main()
