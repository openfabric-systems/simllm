"""Capture one frozen serving process and its persistent packet authority."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import weakref
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

from examples.completion_boundary_v1.run_study import save_transcript
from examples.independent_engine_completion_v1.common import exact_json_bytes
from examples.independent_engine_completion_v1.native import selections
from examples.pd_session_target_scale_v1.native import (
    all_engines,
    append_progress,
    engine_identity,
    request_row,
)
from examples.pd_session_v1 import run_study as baseline
from examples.shared_kv_handoff_v1.capture import (
    CaptureClock,
    EvidenceJournal,
    ObservationOrder,
    clock_state,
)
from examples.shared_kv_handoff_v1.common import (
    HERE,
    PUBLICATIONS,
    native_configuration,
    network_snapshot,
    plain,
    read,
    sha,
    sink_publications,
    source_snapshot,
    validate_sources,
    write,
)


def recorded_owner(args, spec, frozen, deadline, observations, order=None):
    from simllm.backends.flow_session import FlowSessionConfig
    from simllm.placement import disaggregated_manifests
    from simllm.traffic import SharedKvHandoffRuntime

    class ObservedHandoff(SharedKvHandoffRuntime):
        """Observe the existing owner after calls without scheduling any work."""

        def capture(self, action, call=None, result=None):
            first, failures = None, []
            try:
                row = {"index": len(observations), "action": action,
                       "at_ps": None if self._clock is None else self._clock.now_ps,
                       "call": call, "result": result, "clock_after": clock_state(self._clock),
                       "transcript_stop": len(self.transcript),
                       "snapshot": network_snapshot(self)}
                observations.append(row)
                append_progress(args.output_root / "network-observations.jsonl", row)
                if order is not None:
                    order.append("network", row["index"], row["at_ps"])
            except BaseException as error:  # noqa: BLE001, retain the separate wire evidence.
                first = error
                failures.append({"surface": "observation", "type": type(error).__name__, "message": str(error)})
            try:
                save_transcript(args.output_root / "network-transcript", self.transcript, self.stderr)
            except BaseException as error:  # noqa: BLE001, preserve the first capture failure.
                first = first or error
                failures.append({"surface": "transcript", "type": type(error).__name__, "message": str(error)})
            if first is not None:
                self._capture_failures = failures
                raise first

        def invoke(self, action, function, *positional, **keywords):
            clock = positional[0] if action == "bind" else self._clock
            if action == "bind":
                arguments = {"clock_id": id(clock), "engine_local_ranks": plain(positional[1])}
            elif action in ("progress", "complete_due"):
                arguments = {"pending": plain(positional[0]), **plain(keywords)}
            else:
                arguments = plain(keywords)
            call = {"arguments": arguments, "clock_before": clock_state(clock),
                    "transcript_start": len(self.transcript), "order_start": None if order is None else len(order.rows)}
            try:
                result = function(*positional, **keywords)
            except BaseException as primary:
                try:
                    self.capture(action + ":failed", call, {"failure": {"type": type(primary).__name__, "message": str(primary)}})
                except BaseException as error:  # noqa: BLE001, keep the original native failure.
                    try:
                        append_progress(args.output_root / "capture-failures.jsonl", {
                            "action": action, "type": type(error).__name__, "message": str(error)})
                    except BaseException as capture_error:  # noqa: BLE001, preserve the native failure.
                        self._capture_write_failure = {"type": type(capture_error).__name__, "message": str(capture_error)}
                raise
            captured = ([{"request_id": join.request_id, "join": join.to_json()} for join in result]
                        if action == "complete_due" else plain(result))
            self.capture(action, call, captured)
            return result

        def bind(self, *positional, **keywords):
            return self.invoke("bind", super().bind, *positional, **keywords)

        def submit(self, **keywords):
            return self.invoke("submit", super().submit, **keywords)

        def progress(self, *positional, **keywords):
            return self.invoke("progress", super().progress, *positional, **keywords)

        def complete_due(self, *positional, **keywords):
            return self.invoke("complete_due", super().complete_due, *positional, **keywords)

        def close(self):
            return self.invoke("close", super().close)

        def abort(self):
            return self.invoke("abort", super().abort)

    backend = frozen["backend"]
    deployment = disaggregated_manifests(prefill_nodes=spec["prefill_engines"],
                                        decode_nodes=spec["decode_engines"], gpus_per_node=8,
                                        render_physical_topology=False)
    bindings = {f"simllm-{role}-{index}": f"{role}-node-{index}"
                for role in ("prefill", "decode") for index in range(spec[role + "_engines"])}
    config = FlowSessionConfig(backend["profile"], 8 * len(bindings), spec["link_rate_bps"],
                               backend["effective_hardware_sha256"], backend["policy_context_token"],
                               seed=backend["seed"], **deadline["shared_flow_session"],
                               max_events=backend["max_events_per_call"],
                               simulation_budget_ps=backend["simulation_budget_ps"])
    return ObservedHandoff(config, (str(args.htsim_rnic), "--flow-session"),
                            session_id="shared-kv:" + spec["id"], deployment=deployment,
                            engine_nodes=bindings, pcie_submission_ps=backend["pcie_submission_ps"])


@contextmanager
def observe_serialized(session, mode, output, rows, bindings, journal, order=None):
    """Read actual native states around each unchanged synchronous step call."""
    from simllm.adapters.vllm.independent import native_engine_state

    installed = []

    def bind(engine):
        frontend = engine.llm.llm_engine
        original = frontend.step
        had_instance = "step" in vars(frontend)
        prior_instance = vars(frontend).get("step")
        function = original.__func__
        witness = {"engine_id": engine.engine_id, "frontend_id": id(frontend),
                   "function_id": id(function), "code_id": id(function.__code__),
                   "module": function.__module__, "qualname": function.__qualname__,
                   "source": function.__code__.co_filename,
                   "source_sha256": sha(Path(function.__code__.co_filename)),
                   "had_instance_binding": had_instance, "restored": False}
        bindings.append(witness)

        def capture(phase, outputs=()):
            state = native_engine_state(frontend)
            state["emitted"] = [{"request_id": item.request_id, "finished": item.finished,
                                  "token_ids": list(item.outputs[0].token_ids),
                                  "kv_transfer_params": deepcopy(item.kv_transfer_params)} for item in outputs]
            row = {"index": len(rows), "engine_id": engine.engine_id, "phase": phase,
                   "at_ps": session.clock.now_ps, "record_count": len(engine.executor.step_records),
                   "result_count": len(engine.executor.step_results), "native": state}
            rows.append(row)
            append_progress(output / "serialized-observations.jsonl", row)
            if order is not None:
                order.append("serialized", row["index"], session.clock.now_ps)
            journal.engine(engine, phase, session.clock.now_ps)
            journal.runtime(session, phase)

        def observed(*positional, **keywords):
            capture("before-step")
            try:
                result = original(*positional, **keywords)
            except BaseException:
                try:
                    capture("failed-step")
                except BaseException as capture_error:  # noqa: BLE001, preserve the native failure.
                    witness["failure_capture_error"] = {"type": type(capture_error).__name__, "message": str(capture_error)}
                raise
            capture("after-step", result)
            return result

        frontend.step = observed
        installed.append((frontend, original, had_instance, prior_instance, witness))

    try:
        if mode == "serialized":
            for engine in all_engines(session):
                bind(engine)
        yield
    finally:
        primary, cleanup = sys.exc_info()[1], None
        for frontend, original, had_instance, prior_instance, witness in reversed(installed):
            try:
                if had_instance:
                    frontend.step = prior_instance
                else:
                    del frontend.step
                current = frontend.step
                witness["restored"] = (current.__self__ is original.__self__
                                       and current.__func__ is original.__func__
                                       and current.__func__.__code__ is original.__func__.__code__)
            except BaseException as error:  # noqa: BLE001, restore every installed owner.
                witness["restore_error"] = {"type": type(error).__name__, "message": str(error)}
                cleanup = cleanup or error
        try:
            if bindings:
                write(output / "serialized-bindings.json", bindings)
        except BaseException as error:  # noqa: BLE001, retain the first error.
            cleanup = cleanup or error
        if primary is None and cleanup is not None:
            raise cleanup


def run_batch(session, spec, ordinal, admission, output, order):
    from simllm.adapters.vllm.pd_session import VllmPdRequest

    name = (spec["id"].split("-", 1)[1] if spec["kind"] == "compatibility" else spec["id"]) + f":batch-{ordinal}"
    runtime = session.engine_runtime
    first_advance = len(session.clock.advances)
    first_event = 0 if runtime is None else len(runtime.events)
    first_visit = 0 if runtime is None else len(runtime.visits)
    record_starts = {engine.engine_id: len(engine.executor.step_records) for engine in all_engines(session)}
    start = session.clock.now_ps
    first_observation = len(order.rows)
    requests = [VllmPdRequest(name + f":request-{index}", baseline._prompt_tokens()[:spec["prompt_tokens"]],
                              spec["decode_output_tokens"], admission) for index in range(spec["requests_per_batch"])]
    write(output / f"batch-{ordinal}-inputs.json", plain(requests))
    result = session.run_requests(requests)
    row = {"id": name, "start_ps": start, "admission_ps": admission, "end_ps": session.clock.now_ps,
           "observation_start": first_observation, "observation_stop": len(order.rows),
           "inputs": plain(requests),
           "requests": [request_row(item) for item in result.requests],
           "comparisons": [item.to_comparison_json() for item in result.requests],
           "record_starts": record_starts,
           "record_stops": {engine.engine_id: len(engine.executor.step_records) for engine in all_engines(session)},
           "clock_advance_indices": list(range(first_advance, len(session.clock.advances))),
           "event_indices": [] if runtime is None else list(range(first_event, len(runtime.events))),
           "visit_indices": [] if runtime is None else list(range(first_visit, len(runtime.visits))),
           "prefill_batches": result.prefill_batches, "decode_batches": result.decode_batches,
           "network_after": network_snapshot(session._shared_handoff) if spec["kind"] != "compatibility" else {"enabled": False}}
    row = plain(row)
    write(output / f"batch-{ordinal}.json", row)
    for request in row["requests"]:
        append_progress(output / "request-progress.jsonl", request)
    return row


def execute(args):
    import torch
    import vllm
    from huggingface_hub import hf_hub_download

    from simllm.adapters.vllm.pd_session import VllmDisaggregatedSession
    from simllm.core import DeclaredKvHandoffPolicy
    from simllm.core.step import step_record_to_json

    frozen = json.loads((HERE / "expectations.json").read_bytes())
    aliases = json.loads((HERE / "alias-expectations.json").read_bytes())
    frozen["native_source_sha256"].update(aliases["native_source_sha256"])
    deadline = json.loads((HERE / "deadline-expectations.json").read_bytes())
    spec = next(row for row in frozen["native_processes"] if row["id"] == args.process)
    mode = spec.get("engine_timing", "independent")
    config_path = Path(hf_hub_download(baseline.MODEL_ID, "config.json", revision=baseline.MODEL_REVISION,
                                      local_files_only=True)).resolve()
    if (Path(vllm.__file__).resolve().parent != args.vllm_source.resolve()
            or vllm.__version__ != frozen["native_environment"]["vllm_version"]
            or sha(config_path) != frozen["native_environment"]["model_config_sha256"]):
        raise ValueError("native environment or model configuration changed")
    order = ObservationOrder(args.output_root)
    clock, constructions, checkpoints, observations, weak_objects = CaptureClock(order), [], [], [], []
    serialized, serialized_bindings = [], []
    journal = EvidenceJournal(args.output_root)
    cuda_before = torch.cuda.is_initialized()
    construction_start = time.perf_counter_ns()

    def construction(engine):
        weak_objects.extend(weakref.ref(value) for value in (engine, engine.llm, engine.executor, *engine.executor._workers))
        row = {"identity": engine_identity(engine), "elapsed_ns": time.perf_counter_ns() - construction_start,
               "current_rss_kib": baseline._rss_kib(), "peak_rss_kib": baseline._peak_rss_kib(),
               "clock_ps": clock.now_ps, "all_observed_alive": all(ref() is not None for ref in weak_objects),
               "native_configuration": native_configuration(engine)}
        constructions.append(row)
        append_progress(args.output_root / "construction-progress.jsonl", row)

    def observe(phase, engine, native):
        publications = sink_publications(engine)
        row = {"index": len(checkpoints), "phase": phase, "engine_id": engine.engine_id, "at_ps": clock.now_ps,
               "step_index": engine.executor._runtime.step_index if phase == "before-submit" else engine.executor.step_records[-1].step_index,
               "record_count": len(engine.executor.step_records), "result_count": len(engine.executor.step_results),
               "sink_counts": {name: len(publications[name]) for name in PUBLICATIONS},
               "sink_sha256": {name: hashlib.sha256(exact_json_bytes(publications[name])).hexdigest()
                               for name in PUBLICATIONS}, "native": native}
        checkpoints.append(row)
        append_progress(args.output_root / "native-checkpoints.jsonl", row)
        order.append("engine", row["index"], clock.now_ps)
        journal.engine(engine, phase, clock.now_ps, publications)
        journal.runtime(session, phase)

    owner = None if spec["kind"] == "compatibility" else recorded_owner(args, spec, frozen, deadline, observations, order)
    policy = owner
    if owner is None:
        policy = DeclaredKvHandoffPolicy(spec["handoff_ps"]) if spec["handoff_ps"] else DeclaredKvHandoffPolicy.off()
    config = replace(baseline._session_config(args.output_root / "engine-work",
                     prefill_engines=spec["prefill_engines"], decode_engines=spec["decode_engines"]),
                     engine_timing=mode, max_num_seqs=1, handoff_policy=policy)
    sources = source_snapshot(args.repository, frozen)
    validate_sources(sources, args.repository, read(args.source_manifest), args.vllm_source, frozen)
    write(args.output_root / "sources-before.json", sources)
    with VllmDisaggregatedSession(config, clock=clock, construction_observer=construction,
                                  completion_observer=observe if mode == "independent" else None) as session:
        retained_before = [engine_identity(engine) for engine in all_engines(session)]
        selected_before, network_before = selections(session), network_snapshot(owner)
        try:
            with observe_serialized(session, mode, args.output_root, serialized, serialized_bindings, journal, order):
                cells = [run_batch(session, spec, ordinal, admission, args.output_root, order)
                         for ordinal, admission in enumerate(spec["admission_times_ps"])]
        except BaseException:
            try:
                journal.partial(session, all_engines(session))
            except BaseException as error:  # noqa: BLE001, preserve the first native error.
                journal.failure = {"type": type(error).__name__, "message": str(error)}
            raise
        journal.partial(session, all_engines(session))
        steps = [{"engine_id": engine.engine_id, "record": step_record_to_json(record), "result": asdict(result)}
                 for engine in all_engines(session) for record, result in zip(
                     engine.executor.step_records, engine.executor.step_results, strict=True)]
        runtime = session.engine_runtime
        value = {"schema": "simllm-shared-kv-handoff-native-v1", "process_id": spec["id"], "mode": mode,
                 "authority": session.timing_authority, "sources_before": sources,
                 "identity": {"pid": os.getpid(), "interpreter": sys.executable,
                              "package": str(Path(vllm.__file__).resolve().parent), "version": vllm.__version__,
                              "config_sha256": sha(config_path), "config_path": str(config_path),
                              "clock_object_id": id(clock), "cuda_before": cuda_before,
                              "cuda_after": torch.cuda.is_initialized(),
                              "packet_backend_runs": sum(row.backend_runs for engine in all_engines(session)
                                                         for row in engine.step_sink.locality_outcomes),
                              "all_observed_alive": all(ref() is not None for ref in weak_objects),
                              "unfinished_after": [engine.llm.llm_engine.has_unfinished_requests() for engine in all_engines(session)],
                              "native_has_work_after": [engine.llm.llm_engine.engine_core.engine_core.scheduler.has_requests()
                                                        for engine in all_engines(session)]},
                 "retained_before": retained_before, "retained_after": [engine_identity(engine) for engine in all_engines(session)],
                 "selected_before": selected_before, "selected_after": selections(session),
                 "native_configurations": [native_configuration(engine) for engine in all_engines(session)],
                 "construction": constructions, "controls": [], "cells": cells, "steps": steps,
                 "clock_advances": clock.advances, "checkpoints": checkpoints,
                 "serialized_observations": serialized, "serialized_bindings": serialized_bindings,
                 "engine_evidence": journal.engine_rows, "runtime_evidence": journal.runtime_rows,
                 "observation_order": order.rows,
                 "projections": {"events": [] if runtime is None else plain(runtime.events),
                                 "visits": [] if runtime is None else plain(runtime.visits),
                                 "completed": [] if runtime is None else [
                                     {"receipt": asdict(receipt), "result": asdict(result)} for receipt, result in runtime.results]},
                 "sinks": {engine.engine_id: sink_publications(engine) for engine in all_engines(session)},
                 "final_clock_ps": clock.now_ps, "runtime_failure": None if runtime is None else runtime.failure,
                 "network_before": network_before, "network_before_close": network_snapshot(owner)}
        write(args.output_root / "pre-close-native.json", plain(value))
    value.update(sources_after=source_snapshot(args.repository, frozen), network_after=network_snapshot(owner),
                 network_observations=observations, clock_after_close_ps=clock.now_ps,
                 clock_owner_released=not hasattr(clock, "_engine_step_owner"))
    validate_sources(value["sources_after"], args.repository, read(args.source_manifest), args.vllm_source, frozen)
    write(args.output_root / "native.json", plain(value))
    write(args.output_root / "receipt.json", {"pid": os.getpid(), "native_sha256": sha(args.output_root / "native.json")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--process", required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--vllm-source", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--htsim-rnic", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        execute(args)
    except BaseException as error:
        try:
            write(args.output_root / "failure.json", {"type": type(error).__name__, "message": str(error)})
        except BaseException as capture_error:  # noqa: BLE001, preserve the original exception.
            error.__dict__["capture_failure"] = {"type": type(capture_error).__name__, "message": str(capture_error)}
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
