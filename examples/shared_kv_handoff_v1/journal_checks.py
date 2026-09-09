"""Reconstruct complete append-only captures and synchronous native output."""

from __future__ import annotations

from collections import defaultdict

from examples.independent_engine_completion_v1.checks import (
    check_cache_state,
    check_waiting_admissions,
)
from examples.pd_session_target_scale_v1.checks import integer

from .aliases import request_rows
from .common import PUBLICATIONS


def reconstruct(rows, expected, label, evidence, *, engine=False):
    positions = {owner: dict.fromkeys(values, 0) for owner, values in expected.items()}
    previous_time = 0
    for index, row in enumerate(rows):
        name = label + ":" + str(index)
        fields = "index phase at_ps starts stops values" + (" engine_id" if engine else "")
        evidence.fields(name + ":fields", row, fields)
        evidence.equal(name + ":index", row["index"], index)
        evidence.check(name + ":time-forward", integer(row["at_ps"], previous_time))
        previous_time = row["at_ps"]
        owner = row["engine_id"] if engine else "runtime"
        evidence.check(name + ":owner", owner in expected)
        values, cursor = expected[owner], positions[owner]
        evidence.equal(name + ":start-cursors", row["starts"], cursor)
        evidence.equal(name + ":stop-domain", sorted(row["stops"]), sorted(values))
        evidence.equal(name + ":value-domain", sorted(row["values"]), sorted(values))
        for key, complete in values.items():
            stop = row["stops"][key]
            evidence.check(name + ":stop:" + key, integer(stop, cursor[key]) and stop <= len(complete))
            evidence.equal(name + ":full-values:" + key, row["values"][key], complete[cursor[key]:stop])
            cursor[key] = stop
    evidence.equal(label + ":all-completed-prefixes", positions, {
        owner: {key: len(rows) for key, rows in values.items()} for owner, values in expected.items()})


def check_journals(data, evidence):
    label = data["process_id"] + ":journals"
    engines = {}
    for owner in (row["engine_id"] for row in data["retained_before"]):
        steps = [step for step in data["steps"] if step["engine_id"] == owner]
        engines[owner] = {"records": [step["record"] for step in steps], "results": [step["result"] for step in steps],
                          **data["sinks"][owner]}
    reconstruct(data["engine_evidence"], engines, label + ":engine", evidence, engine=True)
    reconstruct(data["runtime_evidence"], {"runtime": {**data["projections"], "clock_advances": data["clock_advances"]}},
                label + ":runtime", evidence)
    checkpoints = data["checkpoints"] if data["mode"] == "independent" else data["serialized_observations"]
    expected = [[row["engine_id"], row["phase"], row["at_ps"]] for row in checkpoints]
    expected.extend([owner, "partial", data["final_clock_ps"]] for owner in engines)
    evidence.equal(label + ":engine-capture-domain", [[row["engine_id"], row["phase"], row["at_ps"]]
                   for row in data["engine_evidence"]], expected)
    evidence.equal(label + ":runtime-capture-domain", [[row["phase"], row["at_ps"]] for row in data["runtime_evidence"]],
                   [[row["phase"], row["at_ps"]] for row in checkpoints] + [["partial", data["final_clock_ps"]]])
    for index, (row, checkpoint) in enumerate(zip(data["engine_evidence"], checkpoints)):
        name = label + ":observed-prefix:" + str(index)
        evidence.equal(name + ":records", row["stops"]["records"], checkpoint["record_count"])
        evidence.equal(name + ":results", row["stops"]["results"], checkpoint["result_count"])
        if data["mode"] == "independent":
            evidence.equal(name + ":publications", {key: row["stops"][key] for key in PUBLICATIONS}, checkpoint["sink_counts"])
    check_runtime_visibility(data, evidence)


def check_runtime_visibility(data, evidence):
    """The retired native callback precedes its core retirement projections."""
    label = data["process_id"] + ":runtime-publication"
    submitted = retired = 0
    checkpoints = data["checkpoints"] if data["mode"] == "independent" else data["serialized_observations"]
    for index, row in enumerate(data["runtime_evidence"]):
        checkpoint = checkpoints[index] if index < len(checkpoints) else None
        phase = row["phase"]
        if data["mode"] == "independent" and phase == "submitted":
            submitted += 1
        name = label + ":" + str(index)
        evidence.equal(name + ":phase-exact-prefix", {key: row["stops"][key] for key in ("events", "visits", "completed")},
                       {"events": 3 * submitted + 2 * retired, "visits": retired, "completed": retired})
        count = row["stops"]["clock_advances"]
        advances = data["clock_advances"][:count]
        evidence.check(name + ":no-future-clock", all(advance["after_ps"] <= row["at_ps"] for advance in advances))
        evidence.equal(name + ":current-clock", advances[-1]["after_ps"] if advances else 0, row["at_ps"])
        if data["mode"] == "independent" and checkpoint is not None and phase == "retired":
            retired += 1


def check_serialized(data, steps, owners, evidence):
    label = data["process_id"] + ":serialized-observer"
    rows, bindings = data["serialized_observations"], data["serialized_bindings"]
    if data["mode"] == "independent":
        evidence.equal(label + ":off", [rows, bindings], [[], []])
        return
    engines = {row["engine_id"]: row for row in data["retained_before"]}
    native_ids = {row["engine_id"]: row["llm_engine_object_id"] for row in data["native_configurations"]}
    evidence.equal(label + ":binding-domain", [row["engine_id"] for row in bindings], list(engines))
    source = data["sources_before"]["native"]["vllm/v1/engine/llm_engine.py"]
    for index, row in enumerate(bindings):
        name = label + ":binding:" + str(index)
        evidence.fields(name + ":fields", row, "engine_id frontend_id function_id code_id module qualname source source_sha256 had_instance_binding restored")
        evidence.equal(name + ":frontend", row["frontend_id"], native_ids[row["engine_id"]])
        evidence.check(name + ":outer-wrapper-distinct", row["frontend_id"] != engines[row["engine_id"]]["frontend_object_id"])
        evidence.equal(name + ":source", [row["source"], row["source_sha256"]], [source["origin"], source["sha256"]])
        evidence.equal(name + ":method", [row["module"], row["qualname"]], ["vllm.v1.engine.llm_engine", "LLMEngine.step"])
        evidence.check(name + ":restored", row["had_instance_binding"] is False and row["restored"] is True
                       and integer(row["function_id"], 1) and integer(row["code_id"], 1))
    evidence.equal(label + ":one-implementation", len({(row["function_id"], row["code_id"]) for row in bindings}), 1)
    evidence.equal(label + ":capture-count", len(rows), 2 * len(steps))
    requests = {row["result"]["request_id"]: row["result"] for row in request_rows(data)}
    visits, observations, previous = defaultdict(list), defaultdict(list), {}
    for key, step in steps.items():
        for member in step["record"]["scheduled"]:
            visits[(key[0], member["request_id"])].append(key)
    for index, row in enumerate(rows):
        name = label + ":capture:" + str(index)
        evidence.fields(name + ":fields", row, "index engine_id phase at_ps record_count result_count native")
        evidence.equal(name + ":index", row["index"], index)
        evidence.equal(name + ":phase", row["phase"], "before-step" if index % 2 == 0 else "after-step")
        engine = row["engine_id"]
        key = (engine, row["record_count"] - index % 2)
        evidence.check(name + ":step-key", key in steps)
        observations[key].append(row)
        evidence.equal(name + ":result-count", row["result_count"], row["record_count"])
        step = steps[key]
        evidence.equal(name + ":time", row["at_ps"], step["record"]["virtual_time_ps"] if index % 2 == 0 else step["result"]["completed_at_ps"])
        check_cache_state(row["native"], owners, engine, name, evidence, 64)
        for alias in row["native"]["requests"]:
            raw = requests[owners[(engine, alias)]]
            release = raw["admitted_at_ps"] if engine == raw["prefill_engine_id"] else raw["handoff"]["completed_at_ps"]
            evidence.check(name + ":native-release:" + alias, row["at_ps"] >= release)
        if index % 2 == 0:
            evidence.equal(name + ":no-early-output", row["native"]["emitted"], [])
            if engine in previous:
                check_waiting_admissions(previous[engine], row["native"], name + ":between-steps", evidence)
        else:
            previous[engine] = row["native"]
    for key, step in steps.items():
        name = label + ":step:" + key[0] + ":" + str(key[1])
        evidence.equal(name + ":pair", len(observations[key]), 2)
        before, after = observations[key]
        expected = []
        scheduled = {member["request_id"] for member in step["record"]["scheduled"]}
        evidence.check(name + ":no-unexplained-removal", set(after["native"]["requests"]) <= set(before["native"]["requests"]))
        for alias, request in before["native"]["requests"].items():
            if alias not in scheduled:
                evidence.equal(name + ":unscheduled-request:" + alias, after["native"]["requests"].get(alias), request)
                evidence.equal(name + ":unscheduled-visible:" + alias, after["native"]["visible"].get(alias), before["native"]["visible"].get(alias))
                for group, (old, current) in enumerate(zip(before["native"]["cache"]["groups"], after["native"]["cache"]["groups"], strict=True)):
                    evidence.equal(name + f":unscheduled-cache:{group}:" + alias, current.get(alias), old.get(alias))
        for queue in ("waiting", "running"):
            evidence.equal(name + ":unscheduled-order:" + queue,
                [alias for alias in after["native"]["queues"][queue] if alias not in scheduled],
                [alias for alias in before["native"]["queues"][queue] if alias not in scheduled])
        for member in step["record"]["scheduled"]:
            alias = member["request_id"]
            raw = requests[owners[(key[0], alias)]]
            ordinal = visits[(key[0], alias)].index(key)
            finished = key == visits[(key[0], alias)][-1]
            for section in ("requests", "visible"):
                evidence.check(name + ":input-presence:" + section + ":" + alias, alias in before["native"][section])
            cell = next(cell for cell in data["cells"] if any(item["result"]["request_id"] == raw["request_id"] for item in cell["requests"]))
            submitted = next(item for item in cell["inputs"] if item["request_id"] == raw["request_id"])
            prefill = key[0] == raw["prefill_engine_id"]
            prompt = submitted["prompt_token_ids"] + ([] if prefill else [512])
            params = {"schema": "simllm-pd-kv-params-v1", "session_request_id": raw["request_id"]} if prefill else raw["kv_transfer_params"]
            expected_request = {"status": "WAITING" if ordinal == 0 else "RUNNING",
                "num_computed_tokens": 0 if ordinal == 0 else len(prompt) + ordinal - 1, "num_in_flight_tokens": 0,
                "output_token_ids": [512] * ordinal, "all_token_ids": prompt + [512] * ordinal,
                "stop_reason": None, "kv_transfer_params": params}
            expected_visible = {"external_request_id": raw["request_id"], "output_token_ids": [512] * ordinal,
                                "is_prefilling": ordinal == 0, "sent_tokens_offset": 0}
            evidence.equal(name + ":before-request:" + alias, before["native"]["requests"][alias], expected_request)
            evidence.equal(name + ":before-visible:" + alias, before["native"]["visible"][alias], expected_visible)
            queue = "waiting" if ordinal == 0 else "running"
            evidence.check(name + ":before-queue:" + alias, alias in before["native"]["queues"][queue])
            expected.append({"request_id": raw["request_id"], "finished": finished, "token_ids": [512] * (ordinal + 1),
                             "kv_transfer_params": raw["kv_transfer_params"] if key[0] == raw["prefill_engine_id"] else None})
            if finished:
                evidence.check(name + ":terminal-removal:" + alias, alias not in after["native"]["requests"] and alias not in after["native"]["visible"])
            else:
                expected_request.update(status="RUNNING", num_computed_tokens=len(prompt) + ordinal,
                    output_token_ids=[512] * (ordinal + 1), all_token_ids=prompt + [512] * (ordinal + 1))
                expected_visible.update(is_prefilling=False, output_token_ids=[512] * (ordinal + 1))
                evidence.equal(name + ":after-request:" + alias, after["native"]["requests"][alias], expected_request)
                evidence.equal(name + ":after-visible:" + alias, after["native"]["visible"][alias], expected_visible)
                evidence.check(name + ":after-queue:" + alias, alias in after["native"]["queues"]["running"])
        evidence.equal(name + ":emitted", after["native"]["emitted"], expected)
