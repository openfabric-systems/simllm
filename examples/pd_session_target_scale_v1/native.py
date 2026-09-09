"""Observe retained native frontend objects without owning their lifecycle."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
import traceback
import weakref
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from examples.pd_session_identity_v1.run_study import exact_json_bytes, sha, write
from examples.pd_session_v1 import run_study as baseline
from simllm.core import VirtualClock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


class ObservedClock(VirtualClock):
    """Record the existing clock's advances without introducing another clock."""

    def __init__(self):
        super().__init__()
        self.advances = []

    def advance_to(self, at_ps):
        before = self.now_ps
        super().advance_to(at_ps)
        self.advances.append({"before_ps": before, "after_ps": self.now_ps})


def worker_identity(worker):
    """Execute through the frontend's callable collective RPC."""
    return {
        "object_id": id(worker), "rank": worker.rank, "local_rank": worker.local_rank,
        "config_id": id(worker.vllm_config), "device_absent": worker.device is None,
        "model_runner_absent": worker.model_runner is None,
    }


def engine_identity(engine):
    executor = engine.executor
    core = engine.llm.llm_engine.engine_core.engine_core
    return {
        "engine_id": engine.engine_id, "role": engine.role.value, "ordinal": engine.ordinal,
        "engine_object_id": id(engine), "frontend_object_id": id(engine.llm),
        "core_object_id": id(core), "executor_object_id": id(executor),
        "frontend_executor_object_id": id(core.model_executor),
        "sink_object_id": id(engine.step_sink), "clock_object_id": id(executor.clock),
        "runtime_clock_object_id": id(executor._runtime.clock),
        "config_object_id": id(executor.vllm_config),
        "frontend_config_object_id": id(engine.llm.llm_engine.vllm_config),
        "world_size": engine.simulated_worker_count,
        "executor_role": executor.config.pool_role,
        "connector_role": engine.llm.llm_engine.vllm_config.kv_transfer_config.kv_role,
        "workers": engine.llm.collective_rpc(worker_identity),
        "executor_workers": [worker_identity(worker) for worker in executor._workers],
        "local_placement": [
            {"rank": row.global_rank, "local_rank": row.local_rank, "role": row.pool_role,
             "gpu_uuid": row.gpu_uuid, "tp_ranks": row.groups["tp"].global_ranks}
            for row in engine.placement.ranks
        ],
    }


def append_progress(path, row):
    with path.open("ab") as stream:
        stream.write(exact_json_bytes(row) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def source_snapshot(package, frozen):
    return {name: sha((package / name.removeprefix("vllm/")).read_bytes())
            for name in frozen["native_source_sha256"]}


def request_row(result):
    return {
        "result": result.to_json(),
        "prefill_record_indices": [row.step_index for row in result.prefill_records],
        "decode_record_indices": [row.step_index for row in result.decode_records],
    }


def all_engines(session):
    return (*session.prefill_engines, *session.decode_engines)


def step_keys(session):
    return [(engine.engine_id, row.step_index)
            for engine in all_engines(session) for row in engine.executor.step_records]


def run_cell(session, spec, output, *, burst=False):
    from simllm.adapters.vllm.pd_session import VllmPdRequest
    from simllm.core import DeclaredKvHandoffPolicy

    prompt = baseline._prompt_tokens()[:spec["prompt_tokens"]]
    prior_steps = set(step_keys(session))
    first_advance = len(session.clock.advances)
    start_ps = session.clock.now_ps
    start_ns = time.perf_counter_ns()
    rows = []
    if burst:
        result = session.run_requests(
            [VllmPdRequest(
                request_id=f"{spec['id']}:request-{index}", prompt_token_ids=prompt,
                admitted_at_ps=start_ps, decode_output_tokens=spec["decode_output_tokens"],
            ) for index in range(spec["requests"])],
            handoff_policy=DeclaredKvHandoffPolicy(spec["handoff_ps"]),
        )
        rows = [request_row(row) for row in result.requests]
        for row in rows:
            append_progress(output / "request-progress.jsonl", row)
    else:
        for index in range(spec["requests"]):
            result = session.run_request(
                f"{spec['id']}:request-{index}", prompt,
                decode_output_tokens=spec["decode_output_tokens"],
                handoff_policy=DeclaredKvHandoffPolicy(spec["handoff_ps"]),
            )
            row = request_row(result)
            rows.append(row)
            append_progress(output / "request-progress.jsonl", row)
    wall_ns = time.perf_counter_ns() - start_ns
    value = {
        "id": spec["id"], "requests": rows, "start_ps": start_ps,
        "end_ps": session.clock.now_ps, "wall_seconds": wall_ns / 1e9,
        "wall_time_ns": wall_ns,
        "arrival_mode": "simultaneous-burst" if burst else "serial-completion-admission",
        "offered_rate_requests_per_second": None,
        "engine_request_counts": {
            role: dict(Counter(row["result"][role + "_engine_id"] for row in rows))
            for role in ("prefill", "decode")
        },
        "step_identities": [list(key) for key in step_keys(session) if key not in prior_steps],
        "clock_advance_indices": list(range(first_advance, len(session.clock.advances))),
    }
    write(output / (spec["id"] + ".json"), value)
    return value


def baseline_control(session, frozen, output):
    from simllm.core import DeclaredKvHandoffPolicy

    rows = []
    for spec in frozen["serial_cells"]:
        result = session.run_request(
            spec["id"], baseline._prompt_tokens()[:spec["prompt_tokens"]],
            decode_output_tokens=spec["decode_output_tokens"],
            handoff_policy=DeclaredKvHandoffPolicy(spec["handoff_ps"]),
        )
        comparison = result.to_comparison_json()
        row = {"id": spec["id"], "comparison": comparison,
               "comparison_sha256": sha(exact_json_bytes(comparison)), **request_row(result)}
        rows.append(row)
        append_progress(output / "request-progress.jsonl", row)
    return rows


def placement_projection(session, scale, output):
    from simllm.placement import disaggregated_manifests

    manifests = disaggregated_manifests(
        prefill_nodes=scale["prefill_engines"], decode_nodes=scale["decode_engines"],
        render_physical_topology=False,
    )
    manifests.placement.save(output / "placement.json")
    manifests.fabric.save(output / "fabric.json")
    gpus = {gpu.global_rank: gpu for node in manifests.fabric.nodes for gpu in node.gpus}
    nics = {nic.nic_id: nic for node in manifests.fabric.nodes for nic in node.nics}
    rows = []
    for engine in all_engines(session):
        offset = 0 if engine.role.value == "prefill" else scale["prefill_engines"]
        for worker in engine.llm.collective_rpc(worker_identity):
            rank_id = 8 * (offset + engine.ordinal) + worker["local_rank"]
            rank = manifests.placement.by_rank(rank_id)
            gpu = gpus[rank_id]
            nic = nics[gpu.nic_id]
            rows.append({
                "engine_id": engine.engine_id, "role": engine.role.value,
                "ordinal": engine.ordinal, "worker_object_id": worker["object_id"],
                "local_rank": worker["local_rank"], "global_rank": rank.global_rank,
                "manifest_role": rank.pool_role, "manifest_local_rank": rank.local_rank,
                "hostname": rank.hostname, "gpu_id": gpu.gpu_id, "gpu_uuid": rank.gpu_uuid,
                "gpu_node_id": gpu.node_id, "nic_id": nic.nic_id,
                "nic_node_id": nic.node_id, "nic_affine_rank": nic.affine_gpu_rank,
                "tp_ranks": rank.groups["tp"].global_ranks,
            })
    return rows


def execute(args):
    import torch
    import vllm
    from huggingface_hub import hf_hub_download

    from simllm.adapters.vllm.pd_session import VllmDisaggregatedSession
    from simllm.core.step import step_record_to_json

    frozen = json.loads((HERE / "expectations.json").read_bytes())
    scale = next(row for row in frozen["scales"] if row["id"] == args.scale)
    package = Path(vllm.__file__).resolve().parent
    config_path = Path(hf_hub_download(
        baseline.MODEL_ID, "config.json", revision=baseline.MODEL_REVISION,
        local_files_only=True,
    ))
    before = source_snapshot(package, frozen)
    if (package != args.vllm_source.resolve() or vllm.__version__ != frozen["frontend"]["version"]
            or before != frozen["native_source_sha256"]
            or sha(config_path.read_bytes()) != frozen["frontend"]["model_config_sha256"]):
        raise ValueError("native source, version or configuration differs from freeze")
    simllm_origins = {
        name: str(Path(importlib.import_module(name.removesuffix(".py").replace("/", ".")).__file__).resolve())
        for name in frozen["current_source_sha256"]
    }
    simllm_before = {name: sha(Path(path).read_bytes()) for name, path in simllm_origins.items()}
    clock = ObservedClock()
    weak_objects = []
    construction = []
    current_before = baseline._rss_kib()
    peak_before = baseline._peak_rss_kib()
    cuda_before = torch.cuda.is_initialized()
    construction_start = time.perf_counter_ns()

    def observe(engine):
        weak_objects.extend(weakref.ref(value) for value in (
            engine, engine.llm, engine.executor, *engine.executor._workers,
        ))
        row = {
            "index": len(construction), "identity": engine_identity(engine),
            "native_constructor_seconds": engine.construction_seconds,
            "elapsed_ns": time.perf_counter_ns() - construction_start,
            "current_rss_kib": baseline._rss_kib(), "peak_rss_kib": baseline._peak_rss_kib(),
            "clock_ps": clock.now_ps,
            "all_observed_objects_alive": all(ref() is not None for ref in weak_objects),
        }
        construction.append(row)
        append_progress(args.output_root / "construction-progress.jsonl", row)
        if row["current_rss_kib"] >= frozen["construction"]["stop_rss_kib"]:
            raise RuntimeError("frozen resident-memory stop reached")

    with VllmDisaggregatedSession(
        baseline._session_config(args.output_root / "engine-work",
                                 prefill_engines=scale["prefill_engines"],
                                 decode_engines=scale["decode_engines"]),
        clock=clock, construction_observer=observe,
    ) as session:
        construction_ns = time.perf_counter_ns() - construction_start
        retention_before = [engine_identity(engine) for engine in all_engines(session)]
        current_constructed = baseline._rss_kib()
        controls = (baseline_control(session, frozen, args.output_root)
                    if scale["id"] == frozen["baseline_control"]["scale"] else [])
        serial = [run_cell(session, spec, args.output_root) for spec in frozen["serial_cells"]]
        burst = run_cell(session, frozen["burst"], args.output_root, burst=True)
        retention_after = [engine_identity(engine) for engine in all_engines(session)]
        projection = placement_projection(session, scale, args.output_root)
        unique_steps = []
        for engine in all_engines(session):
            for record, result in zip(engine.executor.step_records,
                                      engine.executor.step_results, strict=True):
                unique_steps.append({"engine_id": engine.engine_id,
                                     "record": step_record_to_json(record),
                                     "result": json.loads(json.dumps(asdict(result)))})
        backend_runs = sum(row.backend_runs for engine in all_engines(session)
                           for row in engine.step_sink.locality_outcomes)
        collective_arms = sorted({(row.envelope_id, row.arm) for engine in all_engines(session)
                                  for row in engine.step_sink.collective_timing_outcomes})
        value = {
            "schema": "simllm-pd-session-target-scale-native-v1", "scale_id": scale["id"],
            "native_identity": {
                "pid": os.getpid(), "interpreter": sys.executable, "package": str(package),
                "version": vllm.__version__, "config_sha256": sha(config_path.read_bytes()),
                "config_path": str(config_path.resolve()),
                "freeze_sha256": sha((HERE / "expectations.json").read_bytes()),
                "cuda_initialized_before": cuda_before,
                "cuda_initialized_after": torch.cuda.is_initialized(),
                "clock_object_id": id(clock), "packet_backend_runs": backend_runs,
                "native_origins": {
                    name: str(Path(sys.modules[name.removesuffix(".py").replace("/", ".")].__file__).resolve())
                    for name in frozen["native_source_sha256"]
                },
                "simllm_origins": simllm_origins,
                "simllm_source_before": simllm_before,
                "simllm_source_after": {name: sha(Path(path).read_bytes()) for name, path in simllm_origins.items()},
                "unfinished_requests_after": [engine.llm.llm_engine.has_unfinished_requests()
                                              for engine in all_engines(session)],
                "collective_arms": [list(row) for row in collective_arms],
                "observer_weak_references_only": all(isinstance(ref, weakref.ReferenceType)
                                                     for ref in weak_objects),
                "all_observed_objects_alive": all(ref() is not None for ref in weak_objects),
            },
            "source_before": before, "source_after": source_snapshot(package, frozen),
            "construction_rows": construction, "retention_before": retention_before,
            "retention_after": retention_after, "worker_manifest_projection": projection,
            "baseline_controls": controls, "serial_cells": serial, "burst_cell": burst,
            "unique_steps": unique_steps, "clock_advances": clock.advances,
            "host_measurements": {
                "current_rss_before_kib": current_before, "peak_rss_before_kib": peak_before,
                "current_rss_constructed_kib": current_constructed,
                "current_rss_after_requests_kib": baseline._rss_kib(),
                "peak_rss_after_requests_kib": baseline._peak_rss_kib(),
                "construction_wall_time_ns": construction_ns,
                "request_wall_time_ns": sum(row["wall_time_ns"] for row in [*serial, burst]),
                "request_count": sum(len(row["requests"]) for row in [*serial, burst]),
                "final_clock_ps": clock.now_ps,
            },
        }
    write(args.output_root / "native.json", value)
    write(args.output_root / "receipt.json", {
        "pid": os.getpid(), "native_sha256": sha((args.output_root / "native.json").read_bytes()),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", required=True)
    parser.add_argument("--vllm-source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        execute(args)
    except Exception as error:  # noqa: BLE001
        # Keep the exact failed execution and already flushed progress for a void report.
        (args.output_root / "exception.txt").write_text(traceback.format_exc(), encoding="utf-8")
        write(args.output_root / "failure.json", {"type": type(error).__name__, "error": str(error)})
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
