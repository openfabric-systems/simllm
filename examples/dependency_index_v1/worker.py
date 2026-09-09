"""Run one installed source tree under the frozen dependency lookup protocol."""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
import os
import pstats
import subprocess
import sys
import traceback
from copy import deepcopy
from dataclasses import asdict, is_dataclass, replace
from enum import Enum
from pathlib import Path


def plain(value):
    if isinstance(value, Enum):
        return plain(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return plain(asdict(value))
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    if type(value) in (str, int, float, bool, type(None)):
        return value
    raise TypeError("unsupported evidence value: " + type(value).__name__)


def encoded(value):
    return json.dumps(plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")


def sha(value):
    return hashlib.sha256(value).hexdigest()


def write(path, value):
    path.write_bytes(encoded(value))


def package_sources(root):
    paths = subprocess.check_output(["git", "ls-files", "--", "simllm"], cwd=root, text=True).splitlines()
    return {name: sha((root / name).read_bytes()) for name in paths}


def profile_rows(path):
    stats = pstats.Stats(str(path))
    return [{"function": list(key), "primitive_calls": primitive, "calls": calls,
             "self_seconds": self_seconds, "cumulative_seconds": cumulative_seconds,
             "callers": [{"function": list(caller), "counts": list(counts)}
                         for caller, counts in sorted(callers.items())]}
            for key, (primitive, calls, self_seconds, cumulative_seconds, callers) in sorted(stats.stats.items())]


def graph_case(depth, width, *, payload=64, service=1000):
    from simllm.core import CollectiveWork, ComputeWork, ExecutionGraph, ExecutionOperation

    def ring(index, dependencies=()):
        return ExecutionOperation("ring-" + str(index), 0, "collective",
                                  CollectiveWork("all-reduce", tuple(range(width)), payload, "ring"),
                                  participant_local_depends_on=dependencies)

    operations = [ring(0)]
    for layer in range(1, depth + 1):
        names = tuple(f"compute-{layer}-{rank}" for rank in range(width))
        operations.extend(ExecutionOperation(name, rank, name, ComputeWork(name, nominal_duration_ps=service),
                                            participant_local_depends_on=("ring-" + str(layer - 1),))
                          for rank, name in enumerate(names))
        operations.append(ring(layer, names))
    return ExecutionGraph(f"depth-{depth}-width-{width}", 0, 0, tuple(operations), ("ring-" + str(depth),))


def key_control(name):
    from simllm.core import CollectiveWork, ComputeWork, ExecutionGraph, ExecutionOperation

    if name == "same-endpoints-distinct-origin":
        first = ExecutionOperation("first", 0, "shared", ComputeWork("first", nominal_duration_ps=1000))
        second = ExecutionOperation("second", 0, "shared", ComputeWork("second", nominal_duration_ps=1000), depends_on=("first",))
    elif name == "same-endpoints-distinct-rank":
        work = CollectiveWork("all-reduce", (0, 1, 2, 3), 64, "ring")
        first = ExecutionOperation("first", 0, "shared", work)
        second = ExecutionOperation("second", 0, "shared", work, participant_local_depends_on=("first",))
    else:
        raise ValueError("unknown valid key control")
    return ExecutionGraph(name, 0, 0, (first, second), ("second",))


def projection_value(projection):
    return {"execution_id": projection.execution_id, "num_goal_ranks": projection.num_goal_ranks,
            "base_tag": projection.base_tag, "boundaries": plain(projection.boundaries),
            "serialized_edges": plain(projection.serialized_edges),
            "artifacts": [{"operation_ids": plain(artifact.operation_ids), "num_ranks": artifact.trace.num_ranks,
                           "text": artifact.trace.render(), "operations": plain(artifact.trace.operations),
                           "messages": plain(artifact.trace.messages), "dependencies": plain(artifact.trace.dependencies)}
                          for artifact in projection.artifacts]}


def input_value(graph, projection):
    from simllm.core.execution_io import effective_dependency_edges, execution_graph_to_json

    return {"graph": execution_graph_to_json(graph), "edges": plain(effective_dependency_edges(graph)),
            "projection": projection_value(projection)}


def corrupt(projection, name):
    from simllm.traffic import ExecutionGoalArtifactBoundary

    value = deepcopy(projection)
    first = value.serialized_edges[0]
    changed = {
        "predecessor": {"predecessor_id": "absent"}, "target": {"operation_id": "absent"},
        "scope": {"scope": "whole-operation", "participant_rank": None},
        "origin": {"origin": "logical-queue-fifo"},
        "participant-rank": {"participant_rank": (first.participant_rank + 1) % value.num_goal_ranks},
    }
    if name in changed:
        return replace(value, serialized_edges=(replace(first, **changed[name]), *value.serialized_edges[1:]))
    if name == "invalid-type":
        return replace(value, serialized_edges=("invalid-edge", *value.serialized_edges[1:]))
    if name == "delete":
        return replace(value, serialized_edges=value.serialized_edges[1:])
    if name == "duplicate":
        return replace(value, serialized_edges=(*value.serialized_edges, first))
    if name == "reverse":
        return replace(value, serialized_edges=tuple(reversed(value.serialized_edges)))
    if name == "boundary-as-serialized":
        return replace(value, boundaries=value.boundaries[1:], serialized_edges=(*value.serialized_edges, value.boundaries[0].edge))
    if name == "serialized-also-boundary":
        owner = {operation: index for index, artifact in enumerate(value.artifacts) for operation in artifact.operation_ids}
        boundary = ExecutionGoalArtifactBoundary(owner[first.predecessor_id], owner[first.operation_id], first)
        return replace(value, boundaries=(*value.boundaries, boundary))
    if name == "canonical-text":
        operations = value.artifacts[1].trace.rank(0)._ops
        if operations[0].text != "calc 1":
            raise ValueError("frozen compute mutation has no one-nanosecond operation")
        operations[0] = replace(operations[0], text="calc 2")
        return value
    if name == "boundary-index-before-invalid-type":
        value = corrupt(value, "invalid-type")
        return replace(value, boundaries=(replace(value.boundaries[0], predecessor_artifact_index=1), *value.boundaries[1:]))
    if name == "edge-count-before-canonical-text":
        return corrupt(corrupt(value, "canonical-text"), "delete")
    raise ValueError("unknown frozen projection corruption")


def run_graph(spec, frozen, output):
    from simllm.traffic import project_execution_graph_goal, verify_execution_goal_projection

    graph = graph_case(spec["depth"], spec["width"], payload=frozen["ring_payload_bytes"], service=frozen["compute_service_ps"])
    projection = project_execution_graph_goal(graph)
    before = input_value(graph, projection)
    if sys.getprofile() is not None:
        raise ValueError("profile hook already installed")
    profile = cProfile.Profile()
    path = output / (spec["id"] + ".pstats")
    try:
        profile.enable()
        result = verify_execution_goal_projection(graph, projection)
    finally:
        profile.disable()
        profile.dump_stats(str(path))
    profiles = profile_rows(path)
    write(path.with_suffix(".json"), profiles)
    rejected = []
    for name in frozen["projection_corruptions"]:
        candidate = corrupt(projection, name)
        prior = input_value(graph, candidate)
        try:
            verify_execution_goal_projection(graph, candidate)
        except (TypeError, ValueError) as error:
            outcome = {"type": type(error).__name__, "message": str(error)}
        else:
            outcome = None
        rejected.append({"name": name, "before": prior, "after": input_value(graph, candidate), "exception": outcome})
    return {"id": spec["id"], "before": before, "after": input_value(graph, projection), "accepted": result is None,
            "profile_hook_restored": sys.getprofile() is None, "profiles": profiles, "rejections": rejected}


def run_sink(spec, frozen, output):
    from examples.pd_session_v1 import run_study as baseline
    from simllm.backends import HtsimStepSink, HtsimStepSinkConfig
    from simllm.compute import RooflineProvider
    from simllm.core import RequestPhase, ScheduledRequest, StepRecord
    from simllm.core.step import step_record_to_json
    from simllm.placement import declared_manifest

    sink = HtsimStepSink(HtsimStepSinkConfig(
        profile="rnic-nn-fluid", tp_ranks=tuple(range(8)), dims=baseline._granite_dims(), workdir=output / spec["id"],
        provider=RooflineProvider(efficiency=0.7), placement_manifest=declared_manifest(tp=8, nodes=1, gpus_per_node=8),
        collective_fixed_cost_envelope="intra-node-fixed-cost-v1", collective_fixed_cost_arm="lower"))
    prompt, now, steps = spec["prompt_tokens"], 0, []
    for request in range(spec["requests"]):
        for visit in range(5):
            record = StepRecord(len(steps), now, [ScheduledRequest(
                "request-" + str(request), RequestPhase.PREFILL if visit < 2 else RequestPhase.DECODE,
                prompt if visit == 0 else 1, num_cached_tokens=prompt if visit == 1 else 0,
                context_length=prompt + visit)], num_sampled=1)
            result = sink(record)
            steps.append({"record": step_record_to_json(record), "result": plain(result)})
            now = result.completed_at_ps
    config = sink.config
    selection = {"gpu": plain(config.gpu), "host": plain(config.host_model), "dims": plain(config.dims),
                 "provider": vars(config.provider), "tp_ranks": plain(config.tp_ranks),
                 "placement": plain(config.placement_manifest), "precision": plain(config.selected_precision_levels),
                 "collective_profile": plain(config.resolved_collective_latency_profile)}
    return {"id": spec["id"], "steps": steps, "completed_at_ps": now, "selection": selection,
            "publications": {name: plain(getattr(sink, name)) for name in frozen["sink_collections"]}}


def execute(args):
    import simllm
    from examples.pd_session_v1 import run_study as baseline
    from simllm.traffic import project_execution_graph_goal, verify_execution_goal_projection

    root = args.repository_root.resolve()
    if Path(simllm.__file__).resolve() != root / "simllm/__init__.py":
        raise ValueError("worker imported another source tree")
    frozen = json.loads(args.expectations.read_bytes())
    before = package_sources(root)
    helper_path = Path(baseline.__file__).resolve()
    helper_before = sha(helper_path.read_bytes())
    graphs = []
    for spec in frozen["graph_cases"]:
        value = run_graph(spec, frozen, args.output_root)
        graphs.append(value)
        write(args.output_root / (spec["id"] + "-graph.json"), value)
    controls = []
    for name in frozen["valid_key_controls"]:
        graph = key_control(name)
        projection = project_execution_graph_goal(graph)
        value = input_value(graph, projection)
        result = verify_execution_goal_projection(graph, projection)
        controls.append({"id": name, "before": value, "after": input_value(graph, projection), "accepted": result is None})
    sinks = []
    for spec in frozen["sink_cases"]:
        value = run_sink(spec, frozen, args.output_root)
        sinks.append(value)
        write(args.output_root / (spec["id"] + "-sink.json"), value)
    value = {"schema": "simllm-dependency-index-worker-v1", "arm": args.arm, "pid": os.getpid(),
             "interpreter": sys.executable, "package_sources_before": before, "package_sources_after": package_sources(root),
             "module_origins": {name: str(Path(module.__file__).resolve()) for name, module in sorted(sys.modules.items())
                                if name.startswith("simllm") and getattr(module, "__file__", None)},
             "worker_sha256": sha(Path(__file__).read_bytes()), "expectations_sha256": sha(args.expectations.read_bytes()),
             "helper_origin": str(helper_path), "helper_source_before": helper_before,
             "helper_source_after": sha(helper_path.read_bytes()),
             "graphs": graphs, "controls": controls, "sinks": sinks}
    write(args.output_root / "worker.json", value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("before", "after"), required=True)
    for name in ("repository-root", "expectations", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        execute(args)
    except Exception:
        (args.output_root / "exception.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
