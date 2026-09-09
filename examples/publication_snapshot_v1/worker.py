"""Capture actual sink readers and local lifecycles from one selected source."""

from __future__ import annotations

import argparse
import cProfile
import json
import math
import os
import sys
import traceback
from contextlib import ExitStack
from copy import copy, deepcopy
from dataclasses import dataclass, fields, replace
from pathlib import Path
from unittest.mock import patch

from examples.publication_snapshot_v1.common import (
    HERE,
    InvalidObservation,
    nodes,
    pack,
    packages,
    profile_rows,
    sha,
    study_origins,
    write,
)
from simllm.backends.step_sink import HtsimStepSink, HtsimStepSinkConfig
from simllm.compute import ModelDims, RooflineProvider
from simllm.core.clock import VirtualClock
from simllm.core.engine_steps import EngineStepRuntime
from simllm.core.step import RequestPhase, ScheduledRequest, StepRecord
from simllm.core.value_snapshot import value_snapshot
from simllm.placement import declared_manifest


@dataclass(frozen=True)
class SyntheticRow:
    payload: tuple[int, ...]


def make_sink(frozen, workdir, layers=1):
    dimensions = dict(frozen["dimensions"], num_layers=layers)
    return HtsimStepSink(HtsimStepSinkConfig(
        profile="rnic-nn-fluid", tp_ranks=(0, 1), dims=ModelDims(**dimensions), workdir=workdir,
        provider=RooflineProvider(efficiency=0.7),
        placement_manifest=declared_manifest(tp=2, nodes=1, gpus_per_node=2),
        collective_fixed_cost_envelope="intra-node-fixed-cost-v1", collective_fixed_cost_arm="lower"))


def record(index, at_ps):
    return StepRecord(index, at_ps, [ScheduledRequest(
        "request-" + str(index), RequestPhase.DECODE, 1, context_length=16)], num_sampled=1)


def publication_values(sink, frozen):
    return {name: getattr(sink, name) for name in frozen["publication_names"]}


def state_values(sink):
    config = {field.name: getattr(sink.config, field.name) for field in fields(sink.config) if field.name != "provider"}
    provider = sink.config.provider
    return (config, type(provider).__module__, type(provider).__qualname__, vars(provider),
            vars(sink._rank_mapper), vars(sink._registration_ledger))


def binding_names(sink):
    result = []
    for owner, function, code in sink._deferred_bindings():
        if owner is not sink or code is not function.__code__:
            raise ValueError("source binding is not the selected sink implementation")
        result.append(function.__name__)
    return result


def capture_case(spec, frozen, workdir, output):
    sink = make_sink(frozen, workdir)
    for name in frozen["synthetic_populated_names"]:
        getattr(sink, name).extend(SyntheticRow(tuple(range(spec["width"]))) for _ in range(spec["history"]))
    price = sink.prepare_deferred(record(0, 0))
    binding = sink.deferred_publication(price)
    state, published = state_values(sink), publication_values(sink, frozen)
    state_snapshot, publication_snapshot = sink._deferred_state(), sink._deferred_publications()
    before = (value_snapshot(price.record), value_snapshot(price.simulation), state_snapshot, publication_snapshot)
    expected = {"state_visits": nodes(state), "publication_visits": nodes(published),
                "state_encoded_visits": nodes(state_snapshot), "publication_encoded_visits": nodes(publication_snapshot),
                "outer_input_visits": nodes(binding.read_values(binding.payload))}
    capture = {"id": spec["id"], "spec": spec, "expected": expected, "bindings": binding_names(sink),
               "state_snapshot": pack(state_snapshot), "publication_snapshot": pack(publication_snapshot),
               "state_wrapper_exact": state_snapshot == value_snapshot(state),
               "publication_wrapper_exact": publication_snapshot == value_snapshot(published),
               "before": pack(before)}
    write(output / (spec["id"] + "-before.json"), capture)
    if sys.getprofile() is not None:
        raise ValueError("profile hook already occupied")
    profiler = cProfile.Profile()
    profiler.runcall(lambda: value_snapshot(binding.read_values(binding.payload)))
    profile = output / (spec["id"] + ".pstats")
    profiler.dump_stats(str(profile))
    capture.update(profiles=profile_rows(profile), profile_hook_restored=sys.getprofile() is None,
                   after=pack((value_snapshot(price.record), value_snapshot(price.simulation),
                               sink._deferred_state(), sink._deferred_publications())))
    write(output / (spec["id"] + ".json"), capture)
    return capture


def bind(sink, runtime, price, duplicate=False):
    receipt = None

    def publish(result):
        if result is not price.result or sink.publish_deferred(price, runtime, receipt) is not result:
            raise ValueError("owned result identity changed")
        if duplicate:
            sink.publish_deferred(price, runtime, receipt)

    receipt = runtime.submit("engine", price.record, price.result, guard=price.validate, publish=publish,
                             publications=(sink.deferred_publication(price),))
    return receipt


def capture_job(spec, frozen, workdir):
    ordinary, deferred = make_sink(frozen, workdir, spec["layers"]), make_sink(frozen, workdir, spec["layers"])
    records, results, now = [], [], 0
    for index in range(spec["history"] + 1):
        current = record(index, now)
        expected = ordinary(deepcopy(current))
        records.append(current)
        results.append(expected)
        if index < spec["history"] and deferred(deepcopy(current)) != expected:
            raise ValueError("ordinary history disagrees")
        now = expected.completed_at_ps
    runtime = EngineStepRuntime(VirtualClock(records[-1].virtual_time_ps))
    before = pack(value_snapshot(publication_values(deferred, frozen)))
    price = deferred.prepare_deferred(deepcopy(records[-1]))
    receipt = bind(deferred, runtime, price)
    runtime.advance_to(price.result.completed_at_ps - 1)
    early = {"completed": list(runtime.complete_due()), "at_ps": runtime.clock.now_ps,
             "publications": pack(value_snapshot(publication_values(deferred, frozen))),
             "results": pack(value_snapshot(runtime.results)), "visits": pack(value_snapshot(runtime.visits))}
    runtime.advance_to(price.result.completed_at_ps)
    retired = runtime.complete_due()
    result = {"id": spec["id"], "spec": spec, "records": pack(value_snapshot(records)),
              "ordinary_results": pack(value_snapshot(results)), "service_ps": [item.step_latency_ps for item in results],
              "completed_at_ps": now, "before": before, "early": early,
              "retired": pack(value_snapshot(retired)), "receipt": pack(value_snapshot(receipt)),
              "deferred_result": pack(value_snapshot(price.result)),
              "prepared_simulation": pack(value_snapshot(price.simulation)),
              "ordinary_publications": pack(value_snapshot(publication_values(ordinary, frozen))),
              "deferred_publications": pack(value_snapshot(publication_values(deferred, frozen))),
              "events": pack(value_snapshot(runtime.events)), "visits": pack(value_snapshot(runtime.visits)),
              "runtime_results": pack(value_snapshot(runtime.results)), "clock_at_ps": runtime.clock.now_ps,
              "selection": pack(value_snapshot(state_values(deferred)))}
    runtime.close()
    return result


def mutation_observation(sink, price, frozen, runtime, receipt):
    """Capture full values, marking only the three declared invalid locations."""
    state = state_values(sink)
    evidence = state[0]["resolved_collective_evidence_class"]
    if type(evidence) is float and math.isnan(evidence):
        state[0]["resolved_collective_evidence_class"] = InvalidObservation("float.nan")
    publications = {key: list(value) for key, value in publication_values(sink, frozen).items()}
    tail = publications["outcomes"][-1]
    if type(tail) is float and math.isnan(tail):
        publications["outcomes"][-1] = InvalidObservation("float.nan")
    elif type(tail) is object:
        publications["outcomes"][-1] = InvalidObservation("builtins.object")
    elif type(tail) is list and len(tail) == 1 and tail[0] is tail:
        publications["outcomes"][-1] = InvalidObservation("self-cycle-list")
    simulation = price.simulation
    if type(simulation.outcome) is object:
        simulation = replace(simulation, outcome=InvalidObservation("builtins.object"))
    return {"clock_at_ps": runtime.clock.now_ps, "receipt": pack(value_snapshot(receipt)),
            "record": pack(value_snapshot(price.record)), "result": pack(value_snapshot(price.result)),
            "state": pack(value_snapshot(state)), "simulation": pack(value_snapshot(simulation)),
            "publications": pack(value_snapshot(publications)), "price_state": pack(price.state),
            "price_record_state": pack(price.record_state), "price_simulation_state": pack(price.simulation_state),
            "price_publications": pack(price.publications)}


def capture_mutation(name, frozen, workdir, output):
    sink = make_sink(frozen, workdir)
    first = sink(record(0, 0))
    price = sink.prepare_deferred(record(1, first.completed_at_ps))
    runtime = EngineStepRuntime(VirtualClock(first.completed_at_ps))
    receipt = bind(sink, runtime, price, duplicate=name == "duplicate-publication")
    initial = pack(value_snapshot(publication_values(sink, frozen)))
    saved_config, saved_provider = sink.config, sink.config.provider
    config_values = dict(vars(sink.config))
    provider_values = dict(vars(saved_provider))
    saved_publications = deepcopy(publication_values(sink, frozen))
    saved_simulation, saved_simulation_state = price.simulation, price.simulation_state
    saved_outcome = price.simulation.outcome
    fixture = mutation_observation(sink, price, frozen, runtime, receipt)
    with ExitStack() as changes:
        if name == "prior-row":
            sink.outcomes[0] = replace(sink.outcomes[0], host_profile="changed")
        elif name == "provider-values":
            saved_provider.efficiency = 0.5
        elif name == "config-evidence":
            sink.config.resolved_collective_evidence_class = "changed"
        elif name == "config-replacement":
            sink.config = copy(sink.config)
        elif name == "provider-replacement":
            sink.config.provider = RooflineProvider(efficiency=0.7)
        elif name == "prepared-payload-forgery":
            changed = replace(price.simulation, outcome=replace(saved_outcome, makespan_ps=saved_outcome.makespan_ps + 1))
            object.__setattr__(price, "simulation", changed)
            object.__setattr__(price, "simulation_state", value_snapshot(changed))
        elif name == "typed-config":
            sink.config.emit_packet_breakdown = 0
        elif name in ("nonfinite-state-before-simulation", "nonfinite-publication-before-simulation"):
            if name.startswith("nonfinite-state"):
                sink.config.resolved_collective_evidence_class = float("nan")
            else:
                sink.outcomes.append(float("nan"))
            object.__setattr__(price.simulation, "outcome", object())
        elif name == "cycle-publication":
            cyclic = []
            cyclic.append(cyclic)
            sink.outcomes.append(cyclic)
        elif name == "unsupported-publication":
            sink.outcomes.append(object())
        elif name == "binding-forgery":
            original = sink._publish
            changes.enter_context(patch.object(sink, "_deferred_bindings", lambda: price.bindings))
            changes.enter_context(patch.object(sink, "_publish", lambda simulation, record=None: original(simulation, record)))
        elif name in frozen["new_helper_corruptions"]:
            method = frozen["new_helpers"][int(name.startswith("publications"))]
            if name.endswith("binding-forgery"):
                old_values = deepcopy(getattr(sink, method)())
                changes.enter_context(patch.object(sink, method, lambda: old_values))
                changes.enter_context(patch.object(sink, "_deferred_bindings", lambda: price.bindings))
                if name.startswith("state"):
                    sink.config.resolved_collective_evidence_class = "changed"
                else:
                    sink.outcomes[0] = replace(sink.outcomes[0], host_profile="changed")
            elif "class" in name:
                changes.enter_context(patch.object(HtsimStepSink, method, lambda self: None))
            else:
                changes.enter_context(patch.object(sink, method, lambda: None))
        elif name not in ("early-publication", "duplicate-publication"):
            raise ValueError("unknown mutation")
        # Observable selected deltas precede the trigger, including invalid values.
        delta = mutation_delta(name, sink, price, saved_config, saved_provider, saved_outcome, frozen)
        helper_return = None
        if name in frozen["new_helper_corruptions"]:
            method = frozen["new_helpers"][int(name.startswith("publications"))]
            helper_return = pack(value_snapshot(getattr(sink, method)()))
        before_trigger = mutation_observation(sink, price, frozen, runtime, receipt)
        capture = {"name": name, "delta": delta, "fixture": fixture,
                   "pre_trigger": before_trigger, "helper_return": helper_return}
        write(output / (name + "-before.json"), capture)
        before_trigger_counts = {key: len(value) for key, value in publication_values(sink, frozen).items()}
        try:
            if name == "early-publication":
                sink.publish_deferred(price, runtime, receipt)
            else:
                runtime.advance_to(price.result.completed_at_ps)
                runtime.complete_due()
        except (TypeError, ValueError, RuntimeError) as error:
            rejection = {"type": type(error).__name__, "message": str(error)}
        else:
            rejection = None
        after_trigger_counts = {key: len(value) for key, value in publication_values(sink, frozen).items()}
        after_trigger = mutation_observation(sink, price, frozen, runtime, receipt)
        failure, visits, results = runtime.failure, len(runtime.visits), len(runtime.results)
    sink.config = saved_config
    vars(saved_config).clear()
    vars(saved_config).update(config_values)
    vars(saved_provider).clear()
    vars(saved_provider).update(provider_values)
    object.__setattr__(saved_simulation, "outcome", saved_outcome)
    object.__setattr__(price, "simulation", saved_simulation)
    object.__setattr__(price, "simulation_state", saved_simulation_state)
    for key, value in saved_publications.items():
        setattr(sink, key, value)
    price.validate()
    restored = pack(value_snapshot(publication_values(sink, frozen)))
    try:
        runtime.complete_due()
    except RuntimeError as error:
        retry = str(error)
    else:
        retry = None
    result = {**capture, "post_trigger": after_trigger, "exception": rejection, "failure": failure,
            "before_counts": before_trigger_counts, "after_counts": after_trigger_counts,
            "initial_publications": initial, "restored_publications": restored,
            "completed_visits": visits, "completed_results": results, "retry": retry}
    write(output / (name + ".json"), result)
    return result


def mutation_delta(name, sink, price, old_config, old_provider, old_outcome, frozen):
    """Read actual changed values and callables, not just the recipe label."""
    return {"name": name, "config_same": sink.config is old_config,
            "provider_same": sink.config.provider is old_provider,
            "efficiency": sink.config.provider.efficiency,
            "evidence_changed": sink.config.resolved_collective_evidence_class == "changed",
            "evidence_nonfinite": type(sink.config.resolved_collective_evidence_class) is float and math.isnan(sink.config.resolved_collective_evidence_class),
            "typed_integer_zero": type(sink.config.emit_packet_breakdown) is int and sink.config.emit_packet_breakdown == 0,
            "prior_host_changed": sink.outcomes[0].host_profile == "changed",
            "payload_plain_object": type(price.simulation.outcome) is object,
            "payload_added_ps": price.simulation.outcome.makespan_ps - old_outcome.makespan_ps if type(price.simulation.outcome) is type(old_outcome) else None,
            "coherent_payload": price.simulation_state == value_snapshot(price.simulation) if type(price.simulation.outcome) is type(old_outcome) else False,
            "tail_type": type(sink.outcomes[-1]).__name__,
            "tail_nonfinite": type(sink.outcomes[-1]) is float and math.isnan(sink.outcomes[-1]),
            "tail_cycle": type(sink.outcomes[-1]) is list and sink.outcomes[-1][0] is sink.outcomes[-1],
            "instance_shadows": sorted(key for key in vars(sink) if key in (*frozen["new_helpers"], "_deferred_bindings", "_publish")),
            "class_helper_names": {key: getattr(HtsimStepSink, key).__name__ for key in frozen["new_helpers"] if hasattr(HtsimStepSink, key)}}


def execute(args):
    root, output = args.repository.resolve(), args.output.resolve()
    frozen = json.loads(args.expectations.read_bytes())
    source_before = packages(root)
    workdir = output.parent / "sink-work"
    cases = [capture_case(spec, frozen, workdir, output) for spec in frozen["cases"]]
    jobs = []
    for spec in frozen["jobs"]:
        job = capture_job(spec, frozen, workdir)
        write(output / (job["id"] + "-job.json"), job)
        jobs.append(job)
    mutations = [capture_mutation(name, frozen, workdir, output) for name in frozen["common_corruptions"]]
    helpers = [capture_mutation(name, frozen, workdir, output) for name in frozen["new_helper_corruptions"]] if args.arm == "after" else []
    origins = {name: str(Path(module.__file__).resolve()) for name, module in sys.modules.items()
               if name == "simllm" or name.startswith("simllm.") if getattr(module, "__file__", None)}
    write(output / "worker.json", {"schema": "simllm-publication-snapshot-worker-v1", "arm": args.arm,
          "pid": os.getpid(), "interpreter": sys.executable, "repository": str(root),
          "worker_sha256": sha(__file__), "common_sha256": sha(HERE / "common.py"),
          "expectations_sha256": sha(args.expectations), "sources_before": source_before, "sources_after": packages(root),
          "origins": origins, "study_origins": study_origins(), "worker_path": str(Path(__file__).resolve()),
          "cases": cases, "jobs": jobs, "mutations": mutations, "helpers": helpers,
          "workdir": str(workdir), "workdir_empty": workdir.is_dir() and not list(workdir.iterdir())})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expectations", type=Path, required=True)
    parser.add_argument("--arm", choices=("before", "after"), required=True)
    args = parser.parse_args()
    try:
        execute(args)
    except Exception:
        (args.output / "exception.txt").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
