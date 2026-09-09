"""Separate exact lookup work from unchanged supported-path completion."""

from __future__ import annotations

import ast
import json
from copy import deepcopy
from pathlib import Path

from examples.dependency_index_v1.worker import encoded, profile_rows
from examples.pd_session_target_scale_v1.checks import GuardFailure, integer


def read(path):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise GuardFailure("duplicate JSON field: " + key)
            value[key] = item
        return value

    raw = path.read_bytes()
    value = json.loads(raw, object_pairs_hook=unique)
    if raw != encoded(value):
        raise GuardFailure("noncanonical evidence: " + path.name)
    return value


def formula(depth, width):
    edges = depth * (2 * width + 1)
    boundaries, serialized = depth, 2 * depth * width
    search = width * (2 * width + 1) * depth * depth
    return {"edges": edges, "boundaries": boundaries, "serialized": serialized,
            "artifacts": 2 * depth + 1, "search": search,
            "floor": edges + boundaries + serialized,
            "ceiling": edges + boundaries + serialized + edges * serialized,
            "before": edges + boundaries + serialized + search, "after": edges + boundaries + serialized}


def conversion_calls(rows, source, evidence, label):
    selected = [row for row in rows if row["function"][2] == "_goal_edge"]
    evidence.equal(label + ":function-domain", len(selected), 1)
    row = selected[0]
    definitions = [node for node in ast.parse(source.read_bytes()).body
                   if isinstance(node, ast.FunctionDef) and node.name == "_goal_edge"]
    evidence.check(label + ":source-definition", len(definitions) == 1 and not definitions[0].decorator_list)
    evidence.equal(label + ":source-origin", row["function"], [str(source), definitions[0].lineno, "_goal_edge"])
    evidence.check(label + ":integer-count", integer(row["calls"], 1)
                   and row["primitive_calls"] == row["calls"])
    return row["calls"]


def frozen_input(spec, frozen, *, control=None):
    """Construct the declared graph independently of the capture worker."""
    from examples.dependency_index_v1.worker import input_value
    from simllm.core import CollectiveWork, ComputeWork, ExecutionGraph, ExecutionOperation
    from simllm.traffic import project_execution_graph_goal

    if control:
        if control == "same-endpoints-distinct-origin":
            operations = (
                ExecutionOperation("first", 0, "shared", ComputeWork("first", nominal_duration_ps=1000)),
                ExecutionOperation("second", 0, "shared", ComputeWork("second", nominal_duration_ps=1000), depends_on=("first",)))
        else:
            work = CollectiveWork("all-reduce", (0, 1, 2, 3), 64, "ring")
            operations = (ExecutionOperation("first", 0, "shared", work),
                          ExecutionOperation("second", 0, "shared", work, participant_local_depends_on=("first",)))
        graph = ExecutionGraph(control, 0, 0, operations, ("second",))
    else:
        depth, width = spec["depth"], spec["width"]
        operations = []
        for layer in range(depth + 1):
            computes = tuple(f"compute-{layer}-{rank}" for rank in range(width)) if layer else ()
            for rank, name in enumerate(computes):
                operations.append(ExecutionOperation(name, rank, name, ComputeWork(name, nominal_duration_ps=frozen["compute_service_ps"]),
                                  participant_local_depends_on=(f"ring-{layer - 1}",)))
            operations.append(ExecutionOperation(f"ring-{layer}", 0, "collective",
                              CollectiveWork("all-reduce", tuple(range(width)), frozen["ring_payload_bytes"], "ring"),
                              participant_local_depends_on=computes))
        graph = ExecutionGraph(f"depth-{depth}-width-{width}", 0, 0, tuple(operations), (f"ring-{depth}",))
    return input_value(graph, project_execution_graph_goal(graph))


def declared_corruption(value, name):
    """Apply the frozen deltas to JSON, separately from worker object mutations."""
    result = deepcopy(value)
    projection = result["projection"]
    serialized, boundaries = projection["serialized_edges"], projection["boundaries"]
    first = serialized[0]
    if name == "predecessor":
        first["predecessor_id"] = "absent"
    elif name == "target":
        first["operation_id"] = "absent"
    elif name == "scope":
        first.update(scope="whole-operation", participant_rank=None)
    elif name == "origin":
        first["origin"] = "logical-queue-fifo"
    elif name == "participant-rank":
        first["participant_rank"] = (first["participant_rank"] + 1) % projection["num_goal_ranks"]
    elif name == "invalid-type":
        serialized[0] = "invalid-edge"
    elif name == "delete":
        serialized.pop(0)
    elif name == "duplicate":
        serialized.append(deepcopy(first))
    elif name == "reverse":
        serialized.reverse()
    elif name == "boundary-as-serialized":
        serialized.append(boundaries.pop(0)["edge"])
    elif name == "serialized-also-boundary":
        owner = {operation: index for index, artifact in enumerate(projection["artifacts"]) for operation in artifact["operation_ids"]}
        boundaries.append({"predecessor_artifact_index": owner[first["predecessor_id"]],
                           "operation_artifact_index": owner[first["operation_id"]], "edge": deepcopy(first)})
    elif name == "canonical-text":
        artifact = projection["artifacts"][1]
        operation = artifact["operations"][0]
        operation["text"] = "calc 2"
        artifact["text"] = artifact["text"].replace(operation["label"] + ": calc 1\n", operation["label"] + ": calc 2\n", 1)
    elif name == "boundary-index-before-invalid-type":
        serialized[0] = "invalid-edge"
        boundaries[0]["predecessor_artifact_index"] = 1
    elif name == "edge-count-before-canonical-text":
        return declared_corruption(declared_corruption(value, "canonical-text"), "delete")
    else:
        raise GuardFailure("unknown declared corruption")
    return result


def check_graphs(data, frozen, root, path, evidence):
    prefix = data["arm"]
    evidence.equal(prefix + ":graph-domain", [row["id"] for row in data["graphs"]], [row["id"] for row in frozen["graph_cases"]])
    calls = {}
    for row, spec in zip(data["graphs"], frozen["graph_cases"], strict=True):
        label = prefix + ":graph:" + spec["id"]
        evidence.fields(label + ":fields", row, "id before after accepted profile_hook_restored profiles rejections")
        expected = formula(spec["depth"], spec["width"])
        evidence.check(label + ":accepted", row["accepted"] is True and row["profile_hook_restored"] is True)
        evidence.equal(label + ":immutable", row["after"], row["before"])
        value = row["before"]
        evidence.fields(label + ":input-fields", value, "graph edges projection")
        evidence.equal(label + ":frozen-input", value, frozen_input(spec, frozen))
        projection = value["projection"]
        evidence.equal(label + ":edge-count", len(value["edges"]), expected["edges"])
        for collection, field in (("artifacts", "artifacts"), ("boundaries", "boundaries"), ("serialized_edges", "serialized")):
            evidence.equal(label + ":" + collection, len(projection[collection]), expected[field])
        raw_profile = path / (spec["id"] + ".pstats")
        evidence.equal(label + ":profile-export", read(raw_profile.with_suffix(".json")), row["profiles"])
        evidence.equal(label + ":binary-profile", profile_rows(raw_profile), row["profiles"])
        observed = conversion_calls(row["profiles"], root / "simllm/traffic/execution_goal.py", evidence, label)
        evidence.check(label + ":work-bounds", expected["floor"] <= observed <= expected["ceiling"])
        evidence.equal(label + ":work-oracle", observed, expected[prefix], kind="oracles")
        calls[spec["id"]] = observed
        evidence.equal(label + ":rejection-domain", [item["name"] for item in row["rejections"]], frozen["projection_corruptions"])
        for item in row["rejections"]:
            tag = label + ":reject:" + item["name"]
            evidence.fields(tag + ":fields", item, "name before after exception")
            evidence.equal(tag + ":immutable", item["after"], item["before"])
            evidence.equal(tag + ":declared-delta", item["before"], declared_corruption(value, item["name"]))
            evidence.check(tag + ":changed-input", encoded(item["before"]) != encoded(value))
            evidence.check(tag + ":rejected", type(item["exception"]) is dict
                           and item["exception"].get("type") in ("TypeError", "ValueError")
                           and type(item["exception"].get("message")) is str and bool(item["exception"]["message"]))
            if item["name"] == "boundary-index-before-invalid-type":
                evidence.equal(tag + ":precedence", item["exception"], {
                    "type": "ValueError", "message": "artifact boundary indexes do not match graph operations"})
            elif item["name"] == "edge-count-before-canonical-text":
                evidence.check(tag + ":precedence", item["exception"]["type"] == "ValueError"
                               and item["exception"]["message"].startswith("GOAL projection edge mismatch:"))
    evidence.equal(prefix + ":valid-control-domain", [row["id"] for row in data["controls"]], frozen["valid_key_controls"])
    for row in data["controls"]:
        label = prefix + ":control:" + row["id"]
        evidence.fields(label + ":fields", row, "id before after accepted")
        evidence.check(label + ":accepted", row["accepted"] is True)
        evidence.equal(label + ":immutable", row["before"], row["after"])
        evidence.equal(label + ":frozen-input", row["before"], frozen_input(None, frozen, control=row["id"]))
        edges = row["before"]["edges"]
        evidence.check(label + ":same-endpoints", len({(edge["predecessor_id"], edge["operation_id"]) for edge in edges}) == 1)
        if row["id"] == "same-endpoints-distinct-origin":
            evidence.equal(label + ":origin-domain", sorted(edge["origin"] for edge in edges), ["explicit", "logical-queue-fifo"])
        else:
            evidence.equal(label + ":rank-domain", sorted(edge["participant_rank"] for edge in edges if edge["participant_rank"] is not None), list(range(4)))
    return calls


def check_sinks(data, frozen, evidence):
    from examples.pd_session_v1 import run_study as baseline
    from simllm.compute import GPU_ENVELOPES, HostInitiationModel, RooflineProvider
    from simllm.placement import declared_manifest
    from simllm.traffic import resolve_collective_fixed_cost_envelope

    envelope = resolve_collective_fixed_cost_envelope("intra-node-fixed-cost-v1")
    expected_selection = {"gpu": GPU_ENVELOPES["b100"], "host": HostInitiationModel.ideal(), "dims": baseline._granite_dims(),
                          "provider": vars(RooflineProvider(efficiency=0.7)), "tp_ranks": list(range(8)),
                          "placement": declared_manifest(tp=8, nodes=1, gpus_per_node=8),
                          "precision": {"compute": "roofline", "dependency": "serial", "locality": "analytic-nvlink", "network": "rnic-nn-fluid"},
                          "collective_profile": envelope.arm_profile("lower")}
    evidence.equal(data["arm"] + ":sink-domain", [row["id"] for row in data["sinks"]], [spec["id"] for spec in frozen["sink_cases"]])
    for row, spec in zip(data["sinks"], frozen["sink_cases"], strict=True):
        label = data["arm"] + ":sink:" + spec["id"]
        evidence.fields(label + ":fields", row, "id steps completed_at_ps selection publications")
        evidence.check(label + ":source-selection", encoded(row["selection"]) == encoded(expected_selection))
        evidence.equal(label + ":step-count", len(row["steps"]), 5 * spec["requests"])
        services = frozen["known_services_ps"][str(spec["prompt_tokens"])]
        now = 0
        for index, step in enumerate(row["steps"]):
            tag = label + ":step:" + str(index)
            visit, prompt = index % 5, spec["prompt_tokens"]
            service = services["prefill" if visit == 0 else "decode"]
            evidence.fields(tag + ":fields", step, "record result")
            evidence.equal(tag + ":input", step["record"], {"schema": "atlahs-closed-loop-step-v1",
                "step_index": index, "virtual_time_ps": now, "num_sampled": 1,
                "preempted_request_ids": [], "finished_request_ids": [],
                "scheduled": [{"request_id": "request-" + str(index // 5), "phase": "prefill" if visit < 2 else "decode",
                               "num_new_tokens": prompt if visit == 0 else 1, "num_cached_tokens": prompt if visit == 1 else 0,
                               "context_length": prompt + visit}]})
            result = step["result"]
            evidence.check(tag + ":physical-floor", integer(result["step_latency_ps"], frozen["resident_surrogate_floor_ps"]))
            evidence.equal(tag + ":known-service", result, {"step_index": index, "step_latency_ps": service,
                           "completed_at_ps": now + service, "request_metrics": [], "additive_visit_totals": None})
            now += service
        evidence.equal(label + ":job-completion", row["completed_at_ps"], now)
        evidence.equal(label + ":serial-oracle", now, spec["requests"] * (services["prefill"] + 4 * services["decode"]))
        publications = row["publications"]
        evidence.equal(label + ":publication-domain", sorted(publications), sorted(frozen["sink_collections"]))
        for name, rows in publications.items():
            expected = list(range(5 * spec["requests"])) if name in ("outcomes", "locality_outcomes", "collective_timing_outcomes") else []
            evidence.equal(label + ":published-steps:" + name, [item["step_index"] for item in rows], expected)
        for index, outcome in enumerate(publications["outcomes"]):
            evidence.equal(label + ":result-join:" + str(index), outcome["makespan_ps"], row["steps"][index]["result"]["step_latency_ps"])
            evidence.check(label + ":no-fabric:" + str(index), outcome["num_flows"] == 0
                           and publications["locality_outcomes"][index]["backend_runs"] == 0)


def admit(data, frozen, arm, root, path, sources, worker_sha, monitor, evidence):
    evidence.fields(arm + ":fields", data, "schema arm pid interpreter package_sources_before package_sources_after module_origins worker_sha256 expectations_sha256 helper_origin helper_source_before helper_source_after graphs controls sinks")
    evidence.equal(arm + ":schema", data["schema"], "simllm-dependency-index-worker-v1")
    evidence.equal(arm + ":arm", data["arm"], arm)
    evidence.equal(arm + ":pid", data["pid"], monitor["pid"])
    evidence.equal(arm + ":sources-before", data["package_sources_before"], sources)
    evidence.equal(arm + ":sources-after", data["package_sources_after"], sources)
    evidence.equal(arm + ":worker-source", data["worker_sha256"], worker_sha)
    evidence.equal(arm + ":helper-origin", data["helper_origin"], str(root / "examples/pd_session_v1/run_study.py"))
    evidence.equal(arm + ":helper-before", data["helper_source_before"], frozen["granite_helper_sha256"])
    evidence.equal(arm + ":helper-after", data["helper_source_after"], frozen["granite_helper_sha256"])
    evidence.check(arm + ":required-imports", all(name in data["module_origins"] for name in (
        "simllm", "simllm.traffic.execution_goal", "simllm.backends.step_sink", "simllm.compute.provider", "simllm.compute.transformer")))
    for name, origin in data["module_origins"].items():
        module = Path(*name.split("."))
        candidates = (root / module.with_suffix(".py"), root / module / "__init__.py")
        evidence.check(arm + ":origin:" + name, Path(origin) in candidates and Path(origin).relative_to(root).as_posix() in sources)
    evidence.check(arm + ":completed-process", monitor["exit_code"] == 0 and monitor["stopping_reason"] is None)
    evidence.check(arm + ":host-bounds", 0 < monitor["rss_kib"] < frozen["limits"]["rss_cap_kib"]
                   and 0 < monitor["wall_seconds"] < frozen["limits"]["process_timeout_seconds"])
    calls = check_graphs(data, frozen, root, path, evidence)
    check_sinks(data, frozen, evidence)
    return calls


def compare(before, after, calls, frozen, evidence):
    evidence.equal("pair:key-controls", before["controls"], after["controls"])
    evidence.equal("pair:sink-jobs", before["sinks"], after["sinks"])
    for first, second, spec in zip(before["graphs"], after["graphs"], frozen["graph_cases"], strict=True):
        label = "pair:" + spec["id"]
        for name in ("before", "after", "rejections"):
            evidence.equal(label + ":" + name, first[name], second[name])
        factor = 2 + spec["width"] * spec["depth"]
        evidence.check(label, 2 * calls["before"][spec["id"]] == factor * calls["after"][spec["id"]],
                       kind="relations", family="lookup-conversion-reduction")
    evidence.finish("paired-relations")
