"""Admit full source, profile, input, mutation and completion evidence."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import fields
from pathlib import Path

from examples.pd_session_target_scale_v1.checks import Evidence, GuardFailure
from examples.publication_snapshot_v1.common import (
    HERE,
    ROOT,
    InvalidObservation,
    nodes,
    original_nodes,
    pack,
    plain,
    profile_rows,
    sha,
    snapshot_functions,
    study_sources,
    typed_fields,
    unpack,
)
from simllm.backends.step_sink import (
    CollectiveArtifactTiming,
    HtsimStepSink,
    HtsimStepSinkConfig,
    StepCollectiveTimingOutcome,
    StepLocalityOutcome,
    StepNetworkOutcome,
    _SimulatedStep,
)
from simllm.compute import ModelDims, RooflineProvider
from simllm.core.engine_steps import EngineStepReceipt
from simllm.core.execution import CompletionEvent, EventPhase, ResourceKind, ResourceRef
from simllm.core.runtime import QueueVisit
from simllm.core.step import RequestPhase, ScheduledRequest, StepRecord, StepResult
from simllm.core.value_snapshot import value_snapshot
from simllm.placement import declared_manifest

BASE_BINDINGS = ["_plan_step", "_execute_plan", "_simulate_step", "_publish", "_publish_result",
                 "_deferred_state", "_deferred_publications", "_deferred_bindings",
                 "deferred_publication", "prepare_deferred", "publish_deferred", "validate_deferred_mode"]


def expected_state(frozen, workdir, layers=1):
    """Construct only the declared selection, independently of the worker factory."""
    sink = HtsimStepSink(HtsimStepSinkConfig(
        profile=frozen["configuration"]["profile"], tp_ranks=tuple(frozen["configuration"]["tp_ranks"]),
        dims=ModelDims(**dict(frozen["dimensions"], num_layers=layers)), workdir=workdir,
        provider=RooflineProvider(efficiency=frozen["configuration"]["provider_efficiency"]),
        placement_manifest=declared_manifest(tp=2, nodes=1, gpus_per_node=2),
        collective_fixed_cost_envelope=frozen["configuration"]["collective_envelope"],
        collective_fixed_cost_arm=frozen["configuration"]["collective_arm"]))
    config = sink.config
    raw = ({field.name: getattr(config, field.name) for field in fields(config) if field.name != "provider"},
           type(config.provider).__module__, type(config.provider).__qualname__, vars(config.provider),
           vars(sink._rank_mapper), vars(sink._registration_ledger))
    return value_snapshot(raw), nodes(raw)


def expected_publications(spec, frozen, module="__main__"):
    payload = ("builtins", "tuple", tuple(("builtins", "int", number) for number in range(spec["width"])))
    row = (module, "SyntheticRow", (("payload", payload),))
    return ("builtins", "dict", tuple(
        (("builtins", "str", name), ("builtins", "list", tuple(
            row for _ in range(spec["history"] if name in frozen["synthetic_populated_names"] else 0))))
        for name in frozen["publication_names"]))


def expected_delta(name, arm, frozen):
    unsupported = name.startswith("nonfinite-")
    helper_names = {key: key for key in frozen["new_helpers"]} if arm == "after" else {}
    shadows = []
    if name == "binding-forgery":
        shadows = ["_deferred_bindings", "_publish"]
    elif name in frozen["new_helper_corruptions"]:
        helper = frozen["new_helpers"][int(name.startswith("publications"))]
        if "class" in name:
            helper_names[helper] = "<lambda>"
        else:
            shadows = [helper]
            if name.endswith("binding-forgery"):
                shadows.append("_deferred_bindings")
    return {
        "name": name, "config_same": name != "config-replacement",
        "provider_same": name != "provider-replacement",
        "efficiency": 0.5 if name == "provider-values" else 0.7,
        "evidence_changed": name in ("config-evidence", "state-binding-forgery"),
        "evidence_nonfinite": name == "nonfinite-state-before-simulation",
        "typed_integer_zero": name == "typed-config",
        "prior_host_changed": name in ("prior-row", "publications-binding-forgery"),
        "payload_plain_object": unsupported,
        "payload_added_ps": None if unsupported else int(name == "prepared-payload-forgery"),
        "coherent_payload": not unsupported,
        "tail_type": {"nonfinite-publication-before-simulation": "float", "cycle-publication": "list",
                      "unsupported-publication": "object"}.get(name, "StepNetworkOutcome"),
        "tail_nonfinite": name == "nonfinite-publication-before-simulation",
        "tail_cycle": name == "cycle-publication", "instance_shadows": sorted(shadows),
        "class_helper_names": helper_names,
    }


def member(snapshot, name):
    return next(value for key, value in snapshot[2] if key == name)


def entry(snapshot, name):
    return next(value for key, value in snapshot[2] if key[2] == name)


def change_member(snapshot, name, value):
    selected = [pair for pair in snapshot[2] if pair[0] == name]
    if len(selected) != 1:
        raise GuardFailure("mutation field domain disagrees")
    selected[0][1] = value


def change_entry(snapshot, name, value):
    selected = [pair for pair in snapshot[2] if pair[0][2] == name]
    if len(selected) != 1:
        raise GuardFailure("mutation entry domain disagrees")
    selected[0][1] = value


def mutation_fixture(job):
    record, result = job["records"][2][-1], job["ordinary_results"][2][-1]
    values = {"clock_at_ps": job["service_ps"][0], "receipt": job["receipt"], "record": record, "result": result,
            "state": job["selection"], "simulation": job["prepared_simulation"], "publications": job["before"],
            "price_state": job["selection"], "price_record_state": record,
            "price_simulation_state": job["prepared_simulation"], "price_publications": job["before"]}
    return {key: deepcopy(value) for key, value in values.items()}


def mutated_fixture(fixture, name):
    changed = deepcopy(fixture)
    config, provider = changed["state"][2][0], changed["state"][2][3]
    outcomes = entry(changed["publications"], "outcomes")[2]
    encode = lambda value: pack(value_snapshot(value))
    if name in ("prior-row", "publications-binding-forgery"):
        change_member(outcomes[0], "host_profile", encode("changed"))
    elif name == "provider-values":
        change_entry(provider, "efficiency", encode(0.5))
    elif name in ("config-evidence", "state-binding-forgery"):
        change_entry(config, "resolved_collective_evidence_class", encode("changed"))
    elif name == "typed-config":
        change_entry(config, "emit_packet_breakdown", encode(0))
    elif name == "prepared-payload-forgery":
        outcome = member(changed["simulation"], "outcome")
        change_member(outcome, "makespan_ps", encode(member(outcome, "makespan_ps")[2] + 1))
        changed["price_simulation_state"] = deepcopy(changed["simulation"])
    elif name.startswith("nonfinite-"):
        invalid = encode(InvalidObservation("float.nan"))
        if name.startswith("nonfinite-state"):
            change_entry(config, "resolved_collective_evidence_class", invalid)
        else:
            outcomes.append(invalid)
        change_member(changed["simulation"], "outcome", encode(InvalidObservation("builtins.object")))
    elif name in ("cycle-publication", "unsupported-publication"):
        outcomes.append(encode(InvalidObservation("self-cycle-list" if name == "cycle-publication" else "builtins.object")))
    return changed


def check_mutations(rows, names, frozen, arm, evidence, label, fixture_job):
    evidence.equal(label + ":inventory", [row["name"] for row in rows], names)
    for row in rows:
        name, key = row["name"], label + ":" + row["name"]
        evidence.fields(key, row, "name delta fixture pre_trigger post_trigger helper_return exception failure "
                        "before_counts after_counts initial_publications restored_publications completed_visits completed_results retry")
        fixture = mutation_fixture(fixture_job)
        evidence.equal(key + ":complete-fixture", row["fixture"], fixture)
        changed = mutated_fixture(fixture, name)
        evidence.equal(key + ":complete-mutation", row["pre_trigger"], changed)
        helper = None
        if name in frozen["new_helper_corruptions"]:
            helper = fixture["state" if name.startswith("state") else "publications"] if name.endswith("binding-forgery") else pack(value_snapshot(None))
        evidence.equal(key + ":actual-helper-return", row["helper_return"], helper)
        changed["clock_at_ps"] = fixture["clock_at_ps"] if name == "early-publication" else fixture_job["completed_at_ps"]
        if name == "duplicate-publication":
            changed["publications"] = fixture_job["ordinary_publications"]
        evidence.equal(key + ":complete-post-trigger", row["post_trigger"], changed)
        evidence.equal(key + ":initial-publications", row["initial_publications"], fixture["publications"])
        expected_error = frozen["common_errors"].get(name, frozen["new_helper_error"])
        evidence.equal(key + ":declared-delta", row["delta"], expected_delta(name, arm, frozen))
        evidence.equal(key + ":rejection", row["exception"], expected_error)
        evidence.check(key + ":poison", type(row["failure"]) is str and bool(row["failure"]))
        evidence.equal(key + ":restored", row["initial_publications"], row["restored_publications"])
        evidence.equal(key + ":no-core-completion", [row["completed_visits"], row["completed_results"]], [0, 0])
        evidence.check(key + ":retry-rejected", type(row["retry"]) is str and row["retry"].startswith("independent engine runtime is poisoned:"))
        counts = {name: int(name in frozen["synthetic_populated_names"]) for name in frozen["publication_names"]}
        if name in ("nonfinite-publication-before-simulation", "cycle-publication", "unsupported-publication"):
            counts["outcomes"] += 1
        evidence.equal(key + ":before-counts", row["before_counts"], counts)
        if name == "duplicate-publication":
            counts = {name: 2 * count for name, count in counts.items()}
        evidence.equal(key + ":after-counts", row["after_counts"], counts)


def check_publication_rows(published, spec, services, selection, frozen, evidence, label):
    kinds = dict(zip(frozen["synthetic_populated_names"],
                     (StepNetworkOutcome, StepLocalityOutcome, StepCollectiveTimingOutcome), strict=True))
    for name, kind in kinds.items():
        for row in entry(published, name)[2]:
            typed_fields(row, kind)
    rows = plain(published)
    config = selection[2][0]
    profile = plain(entry(config, "resolved_collective_latency_profile"))
    for index, service in enumerate(services):
        key = label + ":row:" + str(index)
        outcome, locality, timing = (rows[name][index] for name in kinds)
        evidence.equal(key + ":identity", [outcome["step_index"], locality["step_index"], timing["step_index"]], [index] * 3)
        evidence.equal(key + ":step-result", outcome["makespan_ps"], service)
        evidence.equal(key + ":sample", [outcome["num_sampled"], outcome["sample_count_exact"]], [1, True])
        evidence.equal(key + ":host", [outcome["host_profile"], outcome["host_launch_class"], outcome["exposed_host_ps"]], ["ideal", "none", 0])
        evidence.equal(key + ":graph", [locality["authority"], locality["ordering_authority"], locality["graph_execution_id"]],
                       ["placement-manifest", "execution-graph", "step-" + str(index)])
        evidence.equal(key + ":local-only", [outcome["num_flows"], locality["backend_runs"], locality["fabric_directed_bytes"], locality["fabric_segments"]], [0] * 4)
        evidence.equal(key + ":floor-registration-off", [locality["collective_floor_phase_ps"], locality["registration_phase_cost_ps"]], [[], []])
        for name in ("profile_id", "bandwidth_bytes_per_second", "participant_latency_ps", "propagation_reference_ps"):
            evidence.equal(key + ":profile:" + name, timing[name], profile[name])
        evidence.equal(key + ":envelope", [timing["envelope_id"], timing["arm"], timing["evidence_class"]],
                       [frozen["configuration"]["collective_envelope"], frozen["configuration"]["collective_arm"],
                        plain(entry(config, "resolved_collective_evidence_class"))])
        timing_snapshot = entry(published, "collective_timing_outcomes")[2][index]
        for artifact in member(timing_snapshot, "artifacts")[2]:
            typed_fields(artifact, CollectiveArtifactTiming)
        artifacts = timing["artifacts"]
        StepCollectiveTimingOutcome(**{**timing, "artifacts": tuple(CollectiveArtifactTiming(**row) for row in artifacts)})
        evidence.check(key + ":artifact-identity", bool(artifacts) and len({row["artifact_id"] for row in artifacts}) == len(artifacts))
        evidence.equal(key + ":artifact-count", locality["artifact_count"], len(artifacts))
        for field, artifact_field in (("artifact_operation_ids", "operation_ids"), ("local_phase_service_ps", "local_service_ps"),
                                      ("fabric_phase_service_ps", "fabric_transport_ps"), ("base_phase_latency_ps", "collective_base_latency_ps"),
                                      ("composed_phase_service_ps", "composed_service_ps")):
            evidence.equal(key + ":projection:" + field, locality[field], [row[artifact_field] for row in artifacts])
        for ordinal, row in enumerate(artifacts):
            evidence.equal(key + ":artifact-local:" + str(ordinal), [row["fabric_transport_ps"], row["registration_cost_ps"]], [0, 0])
            evidence.equal(key + ":artifact-service:" + str(ordinal), row["composed_service_ps"], row["collective_base_latency_ps"] + row["local_service_ps"])
        evidence.equal(key + ":step-sum", sum(row["composed_service_ps"] for row in artifacts), service)
        collective = [row for row in artifacts if row["collective_operation_id"] is not None]
        evidence.equal(key + ":width", [row["participant_count"] for row in collective], [2] * len(collective))
        evidence.equal(key + ":media", locality["local_phase_medium"], ["nvlink" if row in collective else "gpu-compute" for row in artifacts])
        evidence.equal(key + ":local-service", locality["nvlink_service_ps"], sum(row["local_service_ps"] for row in collective))
        evidence.equal(key + ":compute-service", locality["compute_service_ps"], sum(row["local_service_ps"] for row in artifacts if row not in collective))
        evidence.equal(key + ":compute-layers", len(outcome["layer_calc_ns"]), spec["layers"])
        evidence.equal(key + ":compute-quantization", locality["compute_service_ps"], sum(max(value, 1) * 1000 for value in outcome["layer_calc_ns"]))
    return {"outcome": entry(published, "outcomes")[2][-1],
            "locality_outcome": entry(published, "locality_outcomes")[2][-1],
            "collective_timing_outcome": entry(published, "collective_timing_outcomes")[2][-1]}


def check_job(row, spec, frozen, workdir, evidence, label):
    evidence.fields(label, row, "id spec records ordinary_results service_ps completed_at_ps before early retired receipt "
                    "deferred_result prepared_simulation ordinary_publications deferred_publications events visits runtime_results clock_at_ps selection")
    evidence.equal(label + ":spec", row["spec"], spec)
    evidence.equal(label + ":id", row["id"], spec["id"])
    services = row["service_ps"]
    evidence.equal(label + ":steps", len(services), spec["history"] + 1)
    for index, service in enumerate(services):
        evidence.check(label + f":bounds:{index}", type(service) is int and
                       frozen["step_floor_per_layer_ps"] * spec["layers"] <= service <=
                       frozen["step_ceiling_per_layer_ps"] * spec["layers"])
    evidence.equal(label + ":constant-service", services, [services[0]] * (spec["history"] + 1))
    service, release, completion = services[0], sum(services[:-1]), sum(services)
    records = [StepRecord(index, index * service, [ScheduledRequest(
        "request-" + str(index), RequestPhase.DECODE, 1, context_length=16)], num_sampled=1)
        for index in range(spec["history"] + 1)]
    results = [StepResult(index, service, (index + 1) * service) for index in range(spec["history"] + 1)]
    receipt = EngineStepReceipt("engine", spec["history"], 0, release, completion)
    resource = ResourceRef(ResourceKind.GPU_WORK_QUEUE, "engine")
    events = tuple(CompletionEvent("engine-service:engine", "step-" + str(spec["history"]), phase,
                                  release if index < 3 else completion, resource=resource)
                   for index, phase in enumerate((EventPhase.SUBMITTED, EventPhase.QUEUED, EventPhase.STARTED,
                                                   EventPhase.PROGRESS, EventPhase.COMPLETED)))
    visits = (QueueVisit("engine-service:engine", "step-" + str(spec["history"]), resource,
                         release, release, release, completion, completion, stage="declared-whole-engine-service"),)
    expected = {"records": records, "ordinary_results": results, "receipt": receipt, "retired": (receipt,),
                "deferred_result": results[-1], "events": events, "visits": visits,
                "runtime_results": ((receipt, results[-1]),)}
    for key, value in expected.items():
        evidence.equal(label + ":" + key, row[key], pack(value_snapshot(value)))
    evidence.equal(label + ":clock", [row["completed_at_ps"], row["clock_at_ps"]], [completion, completion])
    evidence.equal(label + ":selection", row["selection"], pack(expected_state(frozen, workdir, spec["layers"])[0]))
    evidence.equal(label + ":full-publications", row["ordinary_publications"], row["deferred_publications"])
    published = unpack(row["ordinary_publications"])
    evidence.equal(label + ":publication-container", pack(published[:2]), ["builtins", "dict"])
    evidence.equal(label + ":publication-domain", [key[2] for key, _ in published[2]], frozen["publication_names"])
    prefix = []
    for key, values in published[2]:
        evidence.equal(label + ":publication-list:" + key[2], pack(values[:2]), ["builtins", "list"])
        count = spec["history"] + 1 if key[2] in frozen["synthetic_populated_names"] else 0
        evidence.equal(label + ":publication-count:" + key[2], len(values[2]), count)
        original_nodes(values)
        prefix.append((key, (*values[:2], values[2][:spec["history"]])))
    evidence.equal(label + ":history-prefix", row["before"], pack((*published[:2], tuple(prefix))))
    components = check_publication_rows(published, spec, services, unpack(row["selection"]), frozen, evidence, label)
    components["result"] = value_snapshot(results[-1])
    simulation = (_SimulatedStep.__module__, _SimulatedStep.__qualname__, tuple(
        (field.name, components.get(field.name, value_snapshot(None))) for field in fields(_SimulatedStep)))
    evidence.equal(label + ":prepared-simulation", row["prepared_simulation"], pack(simulation))
    evidence.equal(label + ":early", row["early"], {"completed": [], "at_ps": completion - 1,
                   "publications": row["before"], "results": pack(value_snapshot(())), "visits": pack(value_snapshot(()))})


def admit(data, arm, root, output, frozen, sources, monitor, evidence):
    label = arm
    evidence.fields(label, data, "schema arm pid interpreter repository worker_sha256 common_sha256 expectations_sha256 "
                    "sources_before sources_after origins study_origins worker_path cases jobs mutations helpers workdir workdir_empty")
    evidence.equal(label + ":schema", data["schema"], "simllm-publication-snapshot-worker-v1")
    evidence.equal(label + ":arm", data["arm"], arm)
    evidence.equal(label + ":source-before", data["sources_before"], sources)
    evidence.equal(label + ":source-after", data["sources_after"], sources)
    evidence.equal(label + ":repository", data["repository"], str(root))
    evidence.equal(label + ":pid", data["pid"], monitor["pid"])
    evidence.equal(label + ":worker", data["worker_sha256"], sha(HERE / "worker.py"))
    evidence.equal(label + ":common", data["common_sha256"], sha(HERE / "common.py"))
    evidence.equal(label + ":freeze", data["expectations_sha256"], sha(HERE / "expectations.json"))
    required = {"simllm", "simllm.backends.step_sink", "simllm.core.engine_steps", "simllm.core.value_snapshot",
                "simllm.compute.provider", "simllm.core.step", "simllm.placement"}
    evidence.check(label + ":required-origins", required <= data["origins"].keys())
    for name, path in data["origins"].items():
        base = root / name.replace(".", "/")
        expected_path = base / "__init__.py" if (base / "__init__.py").is_file() else base.with_suffix(".py")
        evidence.equal(label + ":origin:" + name, path, str(expected_path))
        relative = expected_path.relative_to(root).as_posix()
        evidence.check(label + ":tracked-origin:" + name, relative in sources)
        evidence.equal(label + ":origin-bytes:" + name, sha(expected_path), sources[relative])
    study_lock = study_sources(ROOT)
    evidence.check(label + ":study-origin-domain", "examples.publication_snapshot_v1.common" in data["study_origins"])
    for name, path in data["study_origins"].items():
        relative = name.replace(".", "/") + ".py"
        evidence.equal(label + ":study-origin:" + name, path, str(ROOT / relative))
        evidence.check(label + ":tracked-study-origin:" + name, relative in study_lock)
        evidence.equal(label + ":study-origin-bytes:" + name, sha(Path(path)), study_lock[relative])
    evidence.equal(label + ":worker-origin", data["worker_path"], str(HERE / "worker.py"))
    workdir = output.parent / "sink-work"
    evidence.equal(label + ":workdir", data["workdir"], str(workdir))
    evidence.equal(label + ":empty-workdir", data["workdir_empty"], True)
    evidence.check(label + ":actual-empty-workdir", workdir.is_dir() and not list(workdir.iterdir()))
    evidence.equal(label + ":case-domain", [row["id"] for row in data["cases"]], [spec["id"] for spec in frozen["cases"]])
    state, state_count = expected_state(frozen, workdir)
    functions = snapshot_functions(root)
    counts = {}
    for row, spec in zip(data["cases"], frozen["cases"], strict=True):
        key = label + ":case:" + spec["id"]
        evidence.fields(key, row, "id spec expected bindings state_snapshot publication_snapshot state_wrapper_exact "
                        "publication_wrapper_exact before profiles profile_hook_restored after")
        evidence.equal(key + ":spec", row["spec"], spec)
        evidence.equal(key + ":state", row["state_snapshot"], pack(state))
        evidence.equal(key + ":publications", row["publication_snapshot"], pack(expected_publications(spec, frozen)))
        evidence.equal(key + ":input-identity", row["before"], row["after"])
        before = unpack(row["before"])
        evidence.equal(key + ":input-domain", len(before), 4)
        frozen_record = StepRecord(0, 0, [ScheduledRequest("request-0", RequestPhase.DECODE, 1, context_length=16)], num_sampled=1)
        evidence.equal(key + ":frozen-record", pack(before[0]), pack(value_snapshot(frozen_record)))
        evidence.equal(key + ":input-state", pack(before[2]), row["state_snapshot"])
        evidence.equal(key + ":input-publications", pack(before[3]), row["publication_snapshot"])
        evidence.equal(key + ":wrappers", [row["state_wrapper_exact"], row["publication_wrapper_exact"]], [True, True])
        expected_bindings = list(BASE_BINDINGS)
        if arm == "after":
            expected_bindings[8:8] = frozen["new_helpers"]
        evidence.equal(key + ":bindings", row["bindings"], expected_bindings)
        actual = profile_rows(output / (spec["id"] + ".pstats"))
        evidence.equal(key + ":full-profile", row["profiles"], actual)
        selected = {name: [item for item in actual if item["function"] == identity] for name, identity in functions.items()}
        evidence.equal(key + ":profile-function-domain", {name: len(rows) for name, rows in selected.items()}, {"value_snapshot": 1, "visit": 1})
        evidence.equal(key + ":hook-restored", row["profile_hook_restored"], True)
        expected = row["expected"]
        evidence.fields(key + ":expected", expected, "state_visits publication_visits state_encoded_visits publication_encoded_visits outer_input_visits")
        h, q = spec["history"], spec["width"]
        evidence.equal(key + ":structural-counts", {name: expected[name] for name in expected if name != "outer_input_visits"},
                       {"state_visits": state_count, "publication_visits": 21 + 3 * h * (2 + q),
                        "state_encoded_visits": nodes(state), "publication_encoded_visits": 94 + 3 * h * (10 + 4 * q)})
        fixed = 5 + nodes(frozen_record) + original_nodes(before[1]) + 4 * len(expected_bindings)
        outer = fixed + (nodes(state) + expected["publication_encoded_visits"] if arm == "before"
                         else state_count + expected["publication_visits"])
        evidence.equal(key + ":outer-count", expected["outer_input_visits"], outer)
        observed = selected["visit"][0]["calls"]
        evidence.check(key + ":visitor-floor", type(expected["outer_input_visits"]) is int and
                       expected["outer_input_visits"] > 0 and observed >= state_count + expected["publication_visits"])
        expected_calls = expected["outer_input_visits"] + (state_count + expected["publication_visits"] if arm == "before" else 0)
        evidence.equal(key + ":visitor-count", observed, expected_calls)
        evidence.equal(key + ":top-level-snapshots", selected["value_snapshot"][0]["calls"], frozen["top_level_snapshot_calls"][arm], kind="oracles")
        counts[spec["id"]] = observed
    evidence.equal(label + ":job-domain", [row["id"] for row in data["jobs"]], [spec["id"] for spec in frozen["jobs"]])
    for row, spec in zip(data["jobs"], frozen["jobs"], strict=True):
        check_job(row, spec, frozen, workdir, evidence, label + ":job:" + spec["id"])
    first_job = next(row for row in data["jobs"] if row["id"] == "l1-h0")
    for row in data["cases"]:
        evidence.equal(label + ":prepared-case:" + row["id"], row["before"][1], first_job["prepared_simulation"])
    one, two = (next(row["service_ps"][0] for row in data["jobs"] if row["id"] == name) for name in ("l1-h0", "l2-h0"))
    evidence.check(label + ":layer-relation", one < two <= 2 * one + 1000)
    fixture_job = next(row for row in data["jobs"] if row["id"] == "l1-h1")
    check_mutations(data["mutations"], frozen["common_corruptions"], frozen, arm, evidence, label + ":mutations", fixture_job)
    check_mutations(data["helpers"], frozen["new_helper_corruptions"] if arm == "after" else [], frozen, arm, evidence, label + ":helpers", fixture_job)
    return counts


def compare(before, after, counts, frozen, evidence):
    evidence.equal("paired:jobs", before["jobs"], after["jobs"])
    for left, right, spec in zip(before["cases"], after["cases"], frozen["cases"], strict=True):
        label = "paired:" + spec["id"]
        for name in ("state_snapshot", "publication_snapshot", "before", "after"):
            evidence.equal(label + ":" + name, left[name], right[name])
        reduction = left["expected"]["state_encoded_visits"] + left["expected"]["publication_encoded_visits"] - frozen["new_binding_visits"]
        evidence.check(label + ":positive-reduction", reduction > 0)
        actual = counts["before"][spec["id"]] - counts["after"][spec["id"]]
        evidence.check(label, actual == reduction, kind="relations", family="secondary-encoding-work")
    for left, right in zip(before["mutations"], after["mutations"], strict=True):
        evidence.equal("paired:mutation:" + left["name"], {key: value for key, value in left.items() if key != "delta"},
                       {key: value for key, value in right.items() if key != "delta"})
    evidence.finish("paired-relations")


__all__ = ["Evidence", "GuardFailure", "admit", "compare"]
