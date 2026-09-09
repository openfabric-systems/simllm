"""Exact diagnostic admission, with no scored host performance relation."""

from __future__ import annotations

import math
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, fields

from examples.pd_session_identity_v1.run_study import exact_json_bytes, sha, state_snapshot
from examples.pd_session_target_scale_v1.checks import (
    Evidence,
    GuardFailure,
    integer,
    request_metrics,
)
from simllm.backends.step_sink import (
    CollectiveArtifactTiming,
    StepCollectiveTimingOutcome,
    StepLocalityOutcome,
    StepNetworkOutcome,
)
from simllm.compute import GPU_ENVELOPES, HostInitiationModel
from simllm.traffic import resolve_collective_fixed_cost_envelope

ROOT_FIELDS = """schema arm identity construction engines_before engines_after
weak_objects_alive_after baseline_controls baseline_host_call baseline_counters cells
unique_steps clock_advances sink_outcomes phase_counters profiles probe_restoration
probe_stack_empty profile_hook_restored cuda_initialized_before cuda_initialized_after
backend_runs final_clock_ps"""
COUNTER_FIELDS = "calls wall_ns cpu_ns child_wall_ns child_cpu_ns exclusive_wall_ns exclusive_cpu_ns"
ENGINE_FIELDS = """engine_id role ordinal engine_object_id frontend_object_id core_object_id
executor_object_id frontend_executor_object_id sink_object_id clock_object_id
runtime_clock_object_id config_object_id frontend_config_object_id world_size
executor_role connector_role workers executor_workers local_placement
scheduler_object_id output_processor_object_id"""


def collective_selection(frozen):
    selected = frozen["session"]
    envelope = resolve_collective_fixed_cost_envelope(selected["collective_envelope"])
    return envelope, envelope.arm_profile(selected["collective_arm"])


def expected_selections(frozen):
    selected = frozen["session"]
    _, profile = collective_selection(frozen)
    return [{"engine_id": "simllm-" + role + "-0",
             "provider_type": "simllm.compute.provider.RooflineProvider",
             "provider_state": {"efficiency": 0.7, "enable_layer_breakdown": False},
             "gpu": asdict(GPU_ENVELOPES[selected["gpu"]]),
             "host_model": asdict(HostInitiationModel.ideal()),
             "collective_envelope": selected["collective_envelope"],
             "collective_arm": selected["collective_arm"],
             "resolved_collective_profile": state_snapshot(profile),
             "max_num_seqs": selected["max_num_seqs"],
             "tensor_parallel_size": selected["tensor_parallel_size"], "executor_mode": "virtual"}
            for role in ("prefill", "decode")]


def compare_request(raw, frozen):
    value = deepcopy(raw)
    for field in frozen["baseline_control"]["excluded_root_fields"]:
        if not isinstance(value.get(field), str) or not value[field].strip():
            raise GuardFailure("invalid opaque request identity")
        del value[field]
    return {"schema": frozen["baseline_control"]["comparison_schema"],
            "excluded_root_fields": frozen["baseline_control"]["excluded_root_fields"],
            "request": value}


def request_rows(data):
    return [*data["baseline_controls"],
            *(row for entry in data["cells"] for row in entry["cell"]["requests"])]


def normalized_steps(data):
    mapping = {}
    targets = set()
    for row in request_rows(data):
        raw = row["result"]
        for role in ("prefill", "decode"):
            engine = raw[role + "_engine_id"]
            key = (engine, raw[role + "_internal_request_id"])
            target = (engine, raw["request_id"])
            if key in mapping or target in targets:
                raise GuardFailure("step identity mapping is not an engine-qualified bijection")
            mapping[key] = target
            targets.add(target)
    result = deepcopy(data["unique_steps"])
    for step in result:
        record, engine = step["record"], step["engine_id"]
        if "sampled_request_ids" in record:
            raise GuardFailure("unexpected sampled_request_ids field")

        def resolve(identifier, engine=engine):
            key = (engine, identifier)
            if key not in mapping:
                raise GuardFailure("foreign engine-qualified step identity")
            return mapping[key][1]

        for row in record["scheduled"]:
            row["request_id"] = resolve(row["request_id"])
        for key in ("finished_request_ids", "preempted_request_ids"):
            record[key] = [resolve(identifier) for identifier in record[key]]
    return result


def check_counter_rows(rows, phases, evidence, label):
    evidence.equal(label + ":phases", sorted(rows), sorted(phases))
    for phase, row in rows.items():
        name = label + ":" + phase
        evidence.fields(name + ":fields", row, COUNTER_FIELDS)
        evidence.check(name + ":values", all(integer(value) for value in row.values()))
        for kind in ("wall", "cpu"):
            evidence.equal(name + ":" + kind, row["exclusive_" + kind + "_ns"],
                           row[kind + "_ns"] - row["child_" + kind + "_ns"])


def check_profile(profile, evidence, label):
    evidence.fields(label + ":fields", profile, "schema cell_id functions")
    evidence.equal(label + ":schema", profile["schema"], "simllm-host-call-profile-v1")
    functions = profile["functions"]
    evidence.check(label + ":function-table", type(functions) is list and bool(functions))
    keys = []
    for index, row in enumerate(functions):
        name = f"{label}:function:{index}"
        evidence.fields(name + ":fields", row,
                        "function primitive_calls calls self_seconds cumulative_seconds callers")
        key = row["function"]
        evidence.check(name + ":identity", type(key) is list and len(key) == 3
                       and type(key[0]) is str and bool(key[0])
                       and integer(key[1]) and type(key[2]) is str and bool(key[2]))
        keys.append(tuple(key))
        evidence.check(name + ":counts", integer(row["primitive_calls"])
                       and integer(row["calls"], row["primitive_calls"]))
        times = [row["self_seconds"], row["cumulative_seconds"]]
        evidence.check(name + ":time", all(type(v) is float and math.isfinite(v) and v >= 0 for v in times)
                       and times[1] + 1e-12 >= times[0])
        caller_keys = []
        for caller_index, caller in enumerate(row["callers"]):
            caller_name = name + ":caller:" + str(caller_index)
            evidence.fields(caller_name + ":fields", caller, "function counts")
            caller_keys.append(tuple(caller["function"]))
            counts = caller["counts"]
            evidence.check(caller_name + ":counts", type(counts) is list and len(counts) in (1, 4)
                           and all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in counts))
        evidence.check(name + ":unique-callers", len(caller_keys) == len(set(caller_keys)))
    evidence.check(label + ":unique-functions", len(keys) == len(set(keys)))
    evidence.check(label + ":caller-domain", all(tuple(caller["function"]) in set(keys)
                   for row in functions for caller in row["callers"]))


def check_engines(data, evidence):
    before = data["engines_before"]
    evidence.equal("engines:order", [row["engine_id"] for row in before],
                   ["simllm-prefill-0", "simllm-decode-0"])
    evidence.equal("engines:retained", data["engines_after"], before)
    evidence.check("engines:weak", data["weak_objects_alive_after"] is True)
    clock = before[0]["clock_object_id"]
    workers = []
    for role, row in zip(("prefill", "decode"), before, strict=True):
        name = "engine:" + role
        evidence.fields(name + ":fields", row, ENGINE_FIELDS)
        evidence.equal(name + ":role", [row["role"], row["executor_role"]], [role, role])
        evidence.equal(name + ":connector", row["connector_role"],
                       "kv_producer" if role == "prefill" else "kv_consumer")
        evidence.equal(name + ":ordinal", row["ordinal"], 0)
        evidence.equal(name + ":width", row["world_size"], 8)
        evidence.equal(name + ":clock", [row["clock_object_id"], row["runtime_clock_object_id"]], [clock, clock])
        evidence.equal(name + ":executor", row["frontend_executor_object_id"], row["executor_object_id"])
        evidence.equal(name + ":config", row["frontend_config_object_id"], row["config_object_id"])
        evidence.equal(name + ":rpc", row["workers"], row["executor_workers"])
        evidence.equal(name + ":ranks", [w["rank"] for w in row["workers"]], list(range(8)))
        evidence.equal(name + ":local", row["local_placement"], [
            {"rank": i, "local_rank": i, "role": role, "gpu_uuid": f"sim-{role}-gpu-{i}",
             "tp_ranks": list(range(8))} for i in range(8)])
        for index, worker in enumerate(row["workers"]):
            worker_name = name + ":worker:" + str(index)
            evidence.fields(worker_name + ":fields", worker,
                            "object_id rank local_rank config_id device_absent model_runner_absent")
            evidence.equal(worker_name + ":config", worker["config_id"], row["config_object_id"])
            evidence.equal(worker_name + ":local", worker["local_rank"], index)
            evidence.check(worker_name + ":simulated", worker["device_absent"] is True
                           and worker["model_runner_absent"] is True and integer(worker["object_id"], 1))
            workers.append(worker["object_id"])
        evidence.check(name + ":id-types", all(integer(value, 1) for key, value in row.items()
                       if key.endswith("object_id")))
    evidence.check("engines:distinct-workers", len(workers) == len(set(workers)) == 16)
    for field in ("engine_object_id", "frontend_object_id", "core_object_id",
                  "executor_object_id", "sink_object_id", "config_object_id"):
        evidence.check("engines:distinct:" + field, len({row[field] for row in before}) == 2)
    evidence.equal("construction:count", len(data["construction"]), 2)
    for index, row in enumerate(data["construction"]):
        name = "construction:" + str(index)
        evidence.fields(name + ":fields", row,
                        "index identity elapsed_ns current_rss_kib clock_ps all_observed_objects_alive")
        evidence.equal(name + ":index", row["index"], index)
        evidence.equal(name + ":identity", row["identity"], before[index])
        evidence.equal(name + ":clock", row["clock_ps"], 0)
        evidence.check(name + ":host-values", integer(row["elapsed_ns"], 1)
                       and integer(row["current_rss_kib"], 1) and row["all_observed_objects_alive"] is True)


def check_requests_and_steps(data, frozen, evidence):
    rows = request_rows(data)
    expected_specs = [(s["id"], s) for s in frozen["historical_control_specs"]]
    expected_specs += [(f"{s['id']}:request-{i}", s) for s in frozen["cells"] for i in range(s["requests"])]
    evidence.equal("requests:domain", [row["result"]["request_id"] for row in rows],
                   [name for name, _ in expected_specs])
    expected_steps = {"prefill": [], "decode": []}
    expected_advances = []
    previous_finished = {"prefill": [], "decode": []}
    previous = 0
    for index, (row, (name, spec)) in enumerate(zip(rows, expected_specs, strict=True)):
        extra = "id comparison comparison_sha256 " if index < 4 else ""
        evidence.fields(name + ":row-fields", row,
                        extra + "result prefill_record_indices decode_record_indices")
        raw = row["result"]
        request_metrics(raw, name, evidence)
        prompt = spec["prompt_tokens"]
        service = frozen["known_services_ps"][str(prompt)]
        prefill, decode, handoff = service["prefill"], service["decode"], spec["handoff_ps"]
        evidence.equal(name + ":admitted", raw["admitted_at_ps"], previous)
        evidence.equal(name + ":prefill-start", raw["prefill_eligible_at_ps"], previous)
        evidence.equal(name + ":prefill-complete", raw["prefill_completed_at_ps"], previous + prefill)
        cursor = previous + prefill + handoff
        evidence.equal(name + ":handoff-complete", raw["handoff"]["completed_at_ps"], cursor)
        evidence.equal(name + ":decode-start", raw["decode_eligible_at_ps"], cursor)
        completions = []
        for duration in decode:
            cursor += duration
            completions.append(cursor)
        evidence.equal(name + ":token-times", raw["decode_token_completed_at_ps"], completions, kind="oracles")
        evidence.equal(name + ":bytes", raw["handoff"]["kv_bytes"], prompt * 49152)
        evidence.equal(name + ":connector", raw["kv_transfer_params"], {
            "schema": "simllm-pd-kv-params-v1", "remote_request_id": name, "session_request_id": name,
            "remote_num_tokens": prompt, "bootstrap_token_id": 512, "do_remote_prefill": True,
            "do_remote_decode": False, "remote_engine_id": "simllm-prefill-0",
            "worker_tensor_transfer": False, "timing_authority": "simllm-declared-kv-handoff-v1"})
        expected_advances += [{"before_ps": previous, "after_ps": previous},
                              {"before_ps": previous, "after_ps": previous + prefill},
                              {"before_ps": previous + prefill, "after_ps": previous + prefill + handoff}]
        expected_advances += [{"before_ps": end - duration, "after_ps": end}
                              for end, duration in zip(completions, decode, strict=True)]
        for role in ("prefill", "decode"):
            engine = "simllm-" + role + "-0"
            evidence.equal(name + ":engine:" + role, raw[role + "_engine_id"], engine)
            identifier = raw[role + "_internal_request_id"]
            evidence.check(name + ":internal:" + role, type(identifier) is str and bool(identifier.strip()))
            durations = [prefill] if role == "prefill" else decode
            start = previous if role == "prefill" else previous + prefill + handoff
            first = len(expected_steps[role])
            evidence.equal(name + ":indices:" + role, row[role + "_record_indices"], list(range(first, first + len(durations))))
            evidence.equal(name + ":step-count:" + role, raw[role + "_step_count"], len(durations))
            for visit, duration in enumerate(durations):
                step_index = len(expected_steps[role])
                scheduled = {"request_id": identifier, "phase": "prefill" if role == "prefill" or visit == 0 else "decode",
                             "num_new_tokens": prompt if role == "prefill" else 1,
                             "num_cached_tokens": prompt if role == "decode" and visit == 0 else 0,
                             "context_length": prompt if role == "prefill" else prompt + visit + 1}
                record = {"schema": "atlahs-closed-loop-step-v1", "step_index": step_index,
                          "virtual_time_ps": start, "scheduled": [scheduled], "num_sampled": 1,
                          "finished_request_ids": previous_finished[role], "preempted_request_ids": []}
                result = {"step_index": step_index, "step_latency_ps": duration,
                          "completed_at_ps": start + duration, "request_metrics": [], "additive_visit_totals": None}
                expected_steps[role].append({"engine_id": engine, "record": record, "result": result})
                previous_finished[role] = [identifier] if visit == len(durations) - 1 else []
                start += duration
        if index < 4:
            comparison = compare_request(raw, frozen)
            evidence.equal(name + ":comparison", row["comparison"], comparison)
            evidence.equal(name + ":comparison-sha", row["comparison_sha256"], sha(exact_json_bytes(comparison)))
            evidence.equal(name + ":historical", row["comparison_sha256"],
                           frozen["baseline_control"]["comparisons"][index]["comparison_sha256"], kind="oracles")
        previous = completions[-1]
    internal_ids = [row["result"][role + "_internal_request_id"] for row in rows for role in ("prefill", "decode")]
    evidence.check("requests:distinct-native-ids", len(internal_ids) == len(set(internal_ids)))
    evidence.equal("steps:complete", data["unique_steps"], expected_steps["prefill"] + expected_steps["decode"])
    evidence.equal("clock:complete", data["clock_advances"], expected_advances)
    evidence.equal("clock:final", data["final_clock_ps"], previous)
    normalized_steps(data)


def admit(data, frozen, evidence):
    evidence.fields("native:fields", data, ROOT_FIELDS)
    evidence.equal("native:schema", data["schema"], "simllm-pd-host-execution-native-v1")
    evidence.check("native:arm", data["arm"] in frozen["arms"])
    for key in ("probe_stack_empty", "profile_hook_restored", "weak_objects_alive_after"):
        evidence.check(key, data[key] is True)
    for key in ("cuda_initialized_before", "cuda_initialized_after"):
        evidence.check(key, data[key] is False)
    evidence.equal("native:backend-off", data["backend_runs"], 0)
    check_engines(data, evidence)
    check_requests_and_steps(data, frozen, evidence)
    phases = [row["phase"] for row in frozen["instrumentation"]["entries"]] if data["arm"] == "instrumented" else []
    check_counter_rows(data["phase_counters"], phases, evidence, "total-counters")
    check_counter_rows(data["baseline_counters"], phases, evidence, "baseline-counters")
    evidence.fields("baseline:host-fields", data["baseline_host_call"], "wall_ns process_cpu_ns")
    evidence.check("baseline:host-values", all(integer(v, 1) for v in data["baseline_host_call"].values()))
    evidence.equal("cells:domain", [entry["cell"]["id"] for entry in data["cells"]], [s["id"] for s in frozen["cells"]])
    previous = data["baseline_controls"][-1]["result"]["decode_token_completed_at_ps"][-1]
    first_advance = 28
    for entry, spec in zip(data["cells"], frozen["cells"], strict=True):
        name, cell = spec["id"], entry["cell"]
        evidence.fields(name + ":entry", entry, "cell host_call phase_counters profile")
        evidence.fields(name + ":cell", cell, "id requests start_ps end_ps wall_seconds wall_time_ns arrival_mode "
                        "offered_rate_requests_per_second engine_request_counts step_identities clock_advance_indices")
        evidence.fields(name + ":host", entry["host_call"], "wall_ns process_cpu_ns")
        evidence.check(name + ":host-values", all(integer(v, 1) for v in entry["host_call"].values()))
        evidence.check(name + ":wall", integer(cell["wall_time_ns"], 1) and type(cell["wall_seconds"]) is float
                       and cell["wall_seconds"] == cell["wall_time_ns"] / 1e9)
        evidence.equal(name + ":arrival", cell["arrival_mode"], "serial-completion-admission")
        evidence.equal(name + ":rate", cell["offered_rate_requests_per_second"], None)
        evidence.equal(name + ":start", cell["start_ps"], previous)
        previous = cell["requests"][-1]["result"]["decode_token_completed_at_ps"][-1]
        evidence.equal(name + ":end", cell["end_ps"], previous)
        evidence.equal(name + ":counts", cell["engine_request_counts"], {
            role: {"simllm-" + role + "-0": spec["requests"]} for role in ("prefill", "decode")})
        expected_keys = [[step["engine_id"], step["record"]["step_index"]] for step in data["unique_steps"]
                         if cell["start_ps"] <= step["record"]["virtual_time_ps"] < cell["end_ps"]]
        evidence.equal(name + ":steps", cell["step_identities"], expected_keys)
        evidence.equal(name + ":advances", cell["clock_advance_indices"], list(range(first_advance, first_advance + spec["requests"] * 7)))
        first_advance += spec["requests"] * 7
        check_counter_rows(entry["phase_counters"], phases, evidence, name + ":counters")
    if phases:
        parts = [data["baseline_counters"], *(entry["phase_counters"] for entry in data["cells"])]
        for phase in phases:
            evidence.equal("counters:sum:" + phase, data["phase_counters"][phase],
                           {key: sum(part[phase][key] for part in parts) for key in COUNTER_FIELDS.split()})
            evidence.check("counters:reachable:" + phase, data["phase_counters"][phase]["calls"] > 0)
        counts = Counter(row["phase"] for row in data["probe_restoration"])
        expected_counts = Counter({row["phase"]: 2 if row["binding"] == "native-instance" else 1
                                   for row in frozen["instrumentation"]["entries"]})
        evidence.equal("probes:domain", dict(counts), dict(expected_counts))
        for index, row in enumerate(data["probe_restoration"]):
            name = "probe:" + str(index)
            evidence.fields(name + ":fields", row, "phase attribute original_own_attribute restored source owner_object_id")
            evidence.check(name + ":restored", row["restored"] is True and type(row["original_own_attribute"]) is bool
                           and integer(row["owner_object_id"], 1))
            entry = next(item for item in frozen["instrumentation"]["entries"] if item["phase"] == row["phase"])
            evidence.equal(name + ":attribute", row["attribute"], entry["target"].split(".")[-1].split(":")[-1])
            evidence.equal(name + ":binding-structure", row["original_own_attribute"], entry["binding"] != "native-instance")
        for entry in frozen["instrumentation"]["entries"]:
            if entry["binding"] == "native-instance":
                field = "output_processor_object_id" if entry["phase"] == "native-output" else "scheduler_object_id"
                evidence.equal("probes:owners:" + entry["phase"],
                               sorted(row["owner_object_id"] for row in data["probe_restoration"] if row["phase"] == entry["phase"]),
                               sorted(engine[field] for engine in data["engines_before"]))
        for phase in ("native-schedule", "native-update", "native-output"):
            evidence.equal("counters:native-calls:" + phase, data["phase_counters"][phase]["calls"], 70)
        evidence.equal("counters:persistence-calls", data["phase_counters"]["progress-persist"]["calls"], 14)
    else:
        evidence.equal("probes:off", data["probe_restoration"], [])
        evidence.equal("profiles:off", data["profiles"], [])
    evidence.equal("sinks:engine-domain", [row["engine_id"] for row in data["sink_outcomes"]],
                   ["simllm-prefill-0", "simllm-decode-0"])
    sink_types = {"outcomes": StepNetworkOutcome, "locality_outcomes": StepLocalityOutcome,
                  "collective_timing_outcomes": StepCollectiveTimingOutcome}
    envelope, profile = collective_selection(frozen)
    for row in data["sink_outcomes"]:
        name = "sink:" + row["engine_id"]
        evidence.fields(name + ":fields", row, "engine_id collections")
        evidence.equal(name + ":collections", sorted(row["collections"]), sorted(frozen["sink_collections"]))
        steps = [step for step in data["unique_steps"] if step["engine_id"] == row["engine_id"]]
        for key in ("outcomes", "locality_outcomes", "collective_timing_outcomes"):
            for index, outcome in enumerate(row["collections"][key]):
                label = f"{name}:{key}:{index}"
                evidence.fields(label + ":fields", outcome, " ".join(field.name for field in fields(sink_types[key])))
                if key == "collective_timing_outcomes":
                    for artifact_index, artifact in enumerate(outcome["artifacts"]):
                        evidence.fields(label + ":artifact:" + str(artifact_index), artifact,
                                        " ".join(field.name for field in fields(CollectiveArtifactTiming)))
                    try:
                        StepCollectiveTimingOutcome(**{**outcome, "artifacts": tuple(
                            CollectiveArtifactTiming(**artifact) for artifact in outcome["artifacts"])})
                        valid = True
                    except (TypeError, ValueError):
                        valid = False
                    evidence.check(label + ":typed-profile", valid)
                    expected_profile = {"profile_id": profile.profile_id,
                        "bandwidth_bytes_per_second": profile.bandwidth_bytes_per_second,
                        "participant_latency_ps": [list(row) for row in profile.participant_latency_ps],
                        "propagation_reference_ps": profile.propagation_reference_ps,
                        "envelope_id": frozen["session"]["collective_envelope"],
                        "arm": frozen["session"]["collective_arm"],
                        "evidence_class": envelope.arm_evidence_class(frozen["session"]["collective_arm"])}
                    evidence.equal(label + ":frozen-profile", {key: outcome[key] for key in expected_profile}, expected_profile)
                elif key == "outcomes":
                    evidence.equal(label + ":host-selection", outcome["host_profile"], frozen["session"]["host_model"])
            evidence.equal(name + ":step-indices:" + key, [x["step_index"] for x in row["collections"][key]],
                           [step["record"]["step_index"] for step in steps])
        evidence.equal(name + ":service", [x["makespan_ps"] for x in row["collections"]["outcomes"]],
                       [step["result"]["step_latency_ps"] for step in steps])
        for key in set(frozen["sink_collections"]) - {"outcomes", "locality_outcomes", "collective_timing_outcomes"}:
            evidence.equal(name + ":inactive:" + key, row["collections"][key], [])


def compare_pair(first, second, frozen, evidence):
    evidence.equal("pair:arms", [first["arm"], second["arm"]], frozen["arms"])
    evidence.equal("pair:requests", [compare_request(row["result"], frozen) for row in request_rows(first)],
                   [compare_request(row["result"], frozen) for row in request_rows(second)], kind="oracles")
    evidence.equal("pair:steps", normalized_steps(first), normalized_steps(second))
    evidence.equal("pair:clock", first["clock_advances"], second["clock_advances"])
    evidence.equal("pair:sinks", first["sink_outcomes"], second["sink_outcomes"])


def fresh_evidence():
    return Evidence(())
