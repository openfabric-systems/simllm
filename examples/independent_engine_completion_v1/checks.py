"""Admit native ownership, causal timelines and separate evidence classes."""

from __future__ import annotations

from collections import Counter, defaultdict
from fractions import Fraction
from itertools import pairwise

from examples.independent_engine_completion_v1.common import PUBLICATIONS, exact_json_bytes, sha
from examples.pd_session_target_scale_v1.checks import integer, request_metrics
from simllm.core.step import StepResult, step_record_from_json, step_record_to_json


def check_components(rows, frozen, evidence):
    evidence.equal("component:domain", [row["id"] for row in rows], [row["id"] for row in frozen["component_cases"]])
    by_id = {row["id"]: row for row in rows}
    for row in rows:
        label = "component:" + row["id"]
        evidence.equal(label + ":failure", row["failure"], None)
        check_component_checkpoints(row, next(spec for spec in frozen["component_cases"] if spec["id"] == row["id"]), evidence)
        evidence.equal(label + ":event-count", len(row["events"]), len(row["visits"]) * 5)
        evidence.check(label + ":zero-forced-waits", all(
            visit["submitted_at_ps"] == visit["eligible_at_ps"] == visit["started_at_ps"]
            and visit["finished_at_ps"] == visit["completed_at_ps"] for visit in row["visits"]))
    for instance in frozen["behavioral_families"]["component-independence"]["instances"]:
        independent, serial = (by_id["equal-" + instance + "-" + mode] for mode in ("independent", "serial"))
        spec = next(row for row in frozen["component_cases"] if row["id"] == independent["id"])
        evidence.check(instance, independent["end_ps"] == spec["service_ps"]
                       and serial["end_ps"] == spec["work_items"] * spec["service_ps"],
                       kind="relations", family="component-independence")
    for spec in frozen["component_cases"]:
        row = by_id[spec["id"]]
        if spec["kind"] == "unequal-arrival":
            evidence.equal(spec["id"].removeprefix("unequal-"), row["outputs"],
                           [{"id": "short", "at_ps": spec["short_arrival_ps"] + spec["short_service_ps"]},
                            {"id": "long", "at_ps": spec["long_service_ps"]}],
                           kind="relations", family="component-unequal-order")
        elif spec["kind"] == "tie":
            evidence.equal("component:tie", row["outputs"],
                           [{"id": name, "at_ps": at} for name, at in (("a", 4000), ("b", 4000), ("c", 5000))],
                           kind="oracles")
        elif spec["kind"] == "drain":
            evidence.equal("component:drain", row["outputs"], [{"id": "drain", "at_ps": 3000}], kind="oracles")
            evidence.equal("component:drain-zero", row["visits"][0]["completed_at_ps"], row["visits"][0]["started_at_ps"])
    evidence.finish("component-cases")


def check_sources(data, process, frozen, sources, args, monitor, evidence):
    label = process["id"]
    identity = data["identity"]
    evidence.equal(label + ":source-stable", data["sources_before"], data["sources_after"])
    for kind, expected, root in (("repository", sources, args.repository_root),
                                  ("native", frozen["native_source_sha256"], args.vllm_source.parent)):
        current = data["sources_before"][kind]
        evidence.equal(label + ":source:" + kind, current["sha256"], expected)
        evidence.equal(label + ":source-origins:" + kind, current["origins"],
                       {name: str((root / name).resolve()) for name in expected})
    evidence.equal(label + ":pid", identity["pid"], monitor["pid"])
    evidence.equal(label + ":interpreter", identity["interpreter"], str(args.native_python))
    evidence.equal(label + ":package", identity["package"], str(args.vllm_source.resolve()))
    evidence.equal(label + ":version", identity["version"], frozen["native_environment"]["vllm_version"])
    evidence.equal(label + ":model-config", identity["config_sha256"], frozen["native_environment"]["model_config_sha256"])
    evidence.check(label + ":no-gpu-backend", identity["cuda_before"] is False and identity["cuda_after"] is False
                   and type(identity["packet_backend_runs"]) is int and identity["packet_backend_runs"] == 0)
    evidence.equal(label + ":authority", data["authority"],
                   frozen["authority"]["enabled" if process["mode"] == "independent" else "disabled"])
    evidence.equal(label + ":runtime-failure", data["runtime_failure"], None)
    evidence.check(label + ":host-envelope", monitor["exit_code"] == 0 and monitor["stopping_reason"] is None
                   and monitor["wall_seconds"] < frozen["limits"]["native_timeout_seconds"]
                   and 0 < monitor["sampled_max_current_rss_kib"] < frozen["limits"]["native_rss_cap_kib"])


def check_engines(data, process, frozen, evidence):
    label = process["id"]
    expected_ids = [f"simllm-{role}-{index}" for role in ("prefill", "decode")
                    for index in range(process[role + "_engines"])]
    evidence.equal(label + ":engine-domain", [row["engine_id"] for row in data["retained_before"]], expected_ids)
    evidence.equal(label + ":retained-identity", data["retained_before"], data["retained_after"])
    evidence.equal(label + ":construction-domain", [row["identity"] for row in data["construction"]], data["retained_before"])
    evidence.equal(label + ":selected-stable", data["selected_before"], data["selected_after"])
    evidence.equal(label + ":selected-domain", [row["engine_id"] for row in data["selected_before"]], expected_ids)
    evidence.check(label + ":retention", data["identity"]["all_observed_alive"] is True
                   and all(row["all_observed_alive"] is True and row["clock_ps"] == 0 for row in data["construction"]))
    for key in ("engine_object_id", "frontend_object_id", "core_object_id", "executor_object_id", "sink_object_id", "config_object_id"):
        evidence.equal(label + ":distinct:" + key, len({row[key] for row in data["retained_before"]}), len(expected_ids))
    worker_ids = []
    identity_fields = (
        "engine_object_id", "frontend_object_id", "core_object_id", "executor_object_id",
        "frontend_executor_object_id", "sink_object_id", "clock_object_id", "runtime_clock_object_id",
        "config_object_id", "frontend_config_object_id",
    )
    for row, selected in zip(data["retained_before"], data["selected_before"], strict=True):
        name = label + ":" + row["engine_id"]
        role, ordinal = row["engine_id"].removeprefix("simllm-").rsplit("-", 1)
        evidence.equal(name + ":role", [row["role"], row["executor_role"], row["ordinal"], row["connector_role"]],
                       [role, role, int(ordinal), "kv_producer" if role == "prefill" else "kv_consumer"])
        evidence.check(name + ":identity-types", all(integer(row[field], 1) for field in identity_fields))
        evidence.equal(name + ":placement", row["local_placement"], [
            {"rank": index, "local_rank": index, "role": role, "gpu_uuid": f"sim-{role}-gpu-{index}",
             "tp_ranks": list(range(8))} for index in range(8)])
        evidence.equal(name + ":precision", selected["selected_precision"],
                       {"compute": "roofline", "dependency": "serial", "locality": "analytic-nvlink", "network": "rnic-nn-fluid"})
        evidence.check(name + ":clock-owner", row["clock_object_id"] == row["runtime_clock_object_id"] == data["identity"]["clock_object_id"])
        evidence.equal(name + ":executor-owner", row["executor_object_id"], row["frontend_executor_object_id"])
        evidence.equal(name + ":config-owner", row["config_object_id"], row["frontend_config_object_id"])
        evidence.equal(name + ":worker-rpc", row["workers"], row["executor_workers"])
        evidence.equal(name + ":worker-ranks", [worker["rank"] for worker in row["workers"]], list(range(8)))
        evidence.equal(name + ":worker-local-ranks", [worker["local_rank"] for worker in row["workers"]], list(range(8)))
        evidence.equal(name + ":world", row["world_size"], 8)
        evidence.check(name + ":unloaded-workers", all(worker["device_absent"] is True and worker["model_runner_absent"] is True
                       and worker["config_id"] == row["config_object_id"] for worker in row["workers"]))
        evidence.check(name + ":worker-types", all(integer(worker["object_id"], 1) and integer(worker["config_id"], 1)
                       and integer(worker["rank"]) and integer(worker["local_rank"]) for worker in row["workers"]))
        worker_ids.extend(worker["object_id"] for worker in row["workers"])
        evidence.equal(name + ":provider", selected["provider_type"], "simllm.compute.provider.RooflineProvider")
        evidence.equal(name + ":provider-values", selected["provider_state"], {"efficiency": 0.7, "enable_layer_breakdown": False})
        evidence.check(name + ":selected-envelope", selected["collective_arm"] == "lower"
                       and selected["collective_envelope"] == "intra-node-fixed-cost-v1"
                       and selected["max_num_seqs"] == 1 and selected["tensor_parallel_size"] == 8
                       and selected["pipeline_parallel_size"] == selected["data_parallel_size"] == 1
                       and selected["async_scheduling"] is False and selected["policy"] == "fcfs"
                       and selected["executor_mode"] == "virtual")
        evidence.equal(name + ":memory-bandwidth", selected["gpu"]["mem_bandwidth"], float(frozen["bounds"]["memory_bytes_per_second"]))
    evidence.equal(label + ":unique-workers", len(set(worker_ids)), 8 * len(expected_ids))
    evidence.equal(label + ":unfinished", data["identity"]["unfinished_after"], [False] * len(expected_ids))
    if process["mode"] == "independent":
        evidence.equal(label + ":drained-native", data["identity"]["native_has_work_after"], [False] * len(expected_ids))


def check_baselines(data, frozen, evidence):
    rows = data["controls"]
    expected = frozen["baseline_control"]["comparisons"]
    evidence.equal("historical:domain", [row["id"] for row in rows], [row["cell"] for row in expected])
    for row, old in zip(rows, expected, strict=True):
        label = "historical:" + row["id"]
        comparison = row["comparison"]
        excluded = ["prefill_internal_request_id", "decode_internal_request_id"]
        evidence.equal(label + ":comparison", comparison, {
            "schema": frozen["baseline_control"]["comparison_schema"], "excluded_root_fields": excluded,
            "request": {key: value for key, value in row["result"].items() if key not in excluded}})
        evidence.equal(label + ":receipt", row["comparison_sha256"], sha(exact_json_bytes(comparison)))
        evidence.equal(label + ":accepted", row["comparison_sha256"], old["comparison_sha256"], kind="oracles")
        request_metrics(row["result"], label, evidence)
    evidence.finish("historical-baselines")


def request_times(row):
    return {key: row[key] for key in ("admitted_at_ps", "prefill_eligible_at_ps", "prefill_completed_at_ps",
                                    "decode_eligible_at_ps", "decode_token_completed_at_ps", "ttft_ps", "tpot_ps")}


def expected_times(row, start):
    tokens = [start + value for value in row["tokens"]]
    tpot = Fraction(tokens[-1] - tokens[0], 3)
    return {"admitted_at_ps": start + row["arrival"], "prefill_eligible_at_ps": start + row["prefill_start"],
            "prefill_completed_at_ps": start + row["prefill_end"], "decode_eligible_at_ps": start + row["decode_start"],
            "decode_token_completed_at_ps": tokens, "ttft_ps": row["tokens"][0] - row["arrival"],
            "tpot_ps": {"numerator": tpot.numerator, "denominator": tpot.denominator}}


def check_cells(data, process, frozen, references, evidence):
    prefix = process["id"]
    evidence.equal(prefix + ":cell-domain", [row["id"] for row in data["cells"]], [row["id"] for row in process["cells"]])
    all_rows = [*data["controls"], *[row for cell in data["cells"] for row in cell["requests"]]]
    public_ids = [row["result"]["request_id"] for row in all_rows]
    evidence.equal(prefix + ":unique-public-requests", len(set(public_ids)), len(public_ids))
    owners = {}
    for row in all_rows:
        raw = row["result"]
        for role in ("prefill", "decode"):
            key = (raw[role + "_engine_id"], raw[role + "_internal_request_id"])
            evidence.check(prefix + ":opaque:" + raw["request_id"] + ":" + role,
                           key not in owners and type(key[1]) is str and bool(key[1]))
            owners[key] = raw["request_id"]
    steps, used = {}, set()
    for step in data["steps"]:
        record, result = step["record"], step["result"]
        key = (step["engine_id"], record["step_index"])
        label = prefix + ":step:" + ":".join(map(str, key))
        evidence.check(label + ":unique", key not in steps)
        steps[key] = step
        evidence.equal(label + ":record-roundtrip", step_record_to_json(step_record_from_json(record)), record)
        evidence.fields(label + ":result-fields", result,
                        "step_index step_latency_ps completed_at_ps request_metrics additive_visit_totals")
        evidence.equal(label + ":result-reducer-off", result["request_metrics"], [])
        evidence.equal(label + ":visit-reducer-off", result["additive_visit_totals"], None)
        StepResult(**{**result, "request_metrics": ()})
        evidence.check(label + ":time", integer(record["virtual_time_ps"]) and result["step_index"] == record["step_index"]
                       and result["completed_at_ps"] == record["virtual_time_ps"] + result["step_latency_ps"])
        for member in ("finished_request_ids", "preempted_request_ids"):
            evidence.check(label + ":" + member, all((key[0], item) in owners for item in record[member]))
        if record["scheduled"]:
            evidence.equal(label + ":one-native-sequence", len(record["scheduled"]), 1)
            request_id = owners.get((key[0], record["scheduled"][0]["request_id"]))
            evidence.check(label + ":request-join", request_id is not None)
            used.add(key)
        else:
            evidence.check(label + ":genuine-drain", bool(record["finished_request_ids"] or record["preempted_request_ids"])
                           and result["step_latency_ps"] == 0)
    evidence.equal(prefix + ":positive-step-count", len(used), len(all_rows) * 5)
    for cell, spec in zip(data["cells"], process["cells"], strict=True):
        label = prefix + ":cell:" + cell["id"]
        reference = references[process["id"]][cell["id"]]
        start = cell["start_ps"]
        per_engine = Counter()
        for step in reference["service_steps"]:
            per_engine[step["engine_id"]] += step["end_ps"] - step["start_ps"]
        causal = [arrival + frozen["known_services_ps"][str(prompt)]["prefill"]
                  + sum(frozen["known_services_ps"][str(prompt)]["decode"]) + spec["handoff_ps"]
                  for arrival, prompt in zip(spec["arrival_offsets_ps"], spec["prompt_tokens"], strict=True)]
        floor = max(*causal, *per_engine.values())
        ceiling = max(spec["arrival_offsets_ps"]) + sum(per_engine.values()) + spec["requests"] * spec["handoff_ps"]
        evidence.check(label + ":physical-bounds", floor <= cell["end_ps"] - start <= ceiling)
        evidence.equal(label + ":makespan", cell["end_ps"] - start, reference["makespan_ps"])
        evidence.equal(label + ":request-domain", [row["result"]["request_id"] for row in cell["requests"]],
                       [spec["id"] + ":request-" + str(index) for index in range(spec["requests"])])
        expected = [expected_times(row, start) for row in reference["requests"]]
        evidence.equal(label + ":full-timelines", [request_times(row["result"]) for row in cell["requests"]], expected, kind="oracles")
        advances = [{key: start + value for key, value in row.items()} for row in reference["clock_advances"]]
        evidence.equal(label + ":clock-advances", [data["clock_advances"][index] for index in cell["clock_advance_indices"]], advances)
        for index, row in enumerate(cell["requests"]):
            raw, name = row["result"], label + ":request:" + str(index)
            request_metrics(raw, name, evidence)
            prompt = spec["prompt_tokens"][index]
            evidence.equal(name + ":handoff-bytes", raw["handoff"]["kv_bytes"], prompt * frozen["bounds"]["cache_bytes_per_prompt_token"])
            evidence.equal(name + ":handoff-duration", raw["handoff"]["completed_at_ps"] - raw["prefill_completed_at_ps"], spec["handoff_ps"])
            evidence.equal(name + ":connector", raw["kv_transfer_params"], {
                "schema": "simllm-pd-kv-params-v1", "do_remote_prefill": True, "do_remote_decode": False,
                "remote_engine_id": raw["prefill_engine_id"], "remote_request_id": raw["request_id"],
                "session_request_id": raw["request_id"], "remote_num_tokens": prompt, "bootstrap_token_id": 512,
                "worker_tensor_transfer": False, "timing_authority": "simllm-declared-kv-handoff-v1"})
            for role in ("prefill", "decode"):
                engine = f"simllm-{role}-{index % process[role + '_engines']}"
                evidence.equal(name + ":routing:" + role, raw[role + "_engine_id"], engine)
                indices = row[role + "_record_indices"]
                evidence.check(name + ":slice-indices:" + role, len(indices) > 0 and indices == list(range(indices[0], indices[-1] + 1)))
                evidence.equal(name + ":slice-count:" + role, raw[role + "_step_count"], len(indices))
                records = [steps[(engine, number)]["record"] for number in indices]
                boundary = raw["admitted_at_ps"] if role == "prefill" else raw["handoff"]["completed_at_ps"]
                evidence.check(name + ":slice-release:" + role, all(record["virtual_time_ps"] >= boundary for record in records))
                own = [steps[(engine, number)] for number in indices if any(
                    item["request_id"] == raw[role + "_internal_request_id"]
                    for item in steps[(engine, number)]["record"]["scheduled"])]
                expected_steps = [step for step in reference["service_steps"] if step["role"] == role and step["request_index"] == index]
                evidence.equal(name + ":owned-step-times:" + role,
                               [[step["record"]["virtual_time_ps"], step["result"]["completed_at_ps"]] for step in own],
                               [[start + step["start_ps"], start + step["end_ps"]] for step in expected_steps])
                phases = [step["record"]["scheduled"][0]["phase"] for step in own]
                evidence.equal(name + ":native-phases:" + role, phases,
                               ["prefill"] if role == "prefill" else ["prefill", "decode", "decode", "decode"])
    evidence.equal(prefix + ":final-clock", data["final_clock_ps"], data["cells"][-1]["end_ps"])
    check_native_memberships(data, process, steps, owners, evidence)
    return steps, owners


def check_projections(data, process, steps, evidence):
    label = process["id"] + ":projection"
    projections = data["projections"]
    evidence.fields(label + ":fields", projections, "events visits completed")
    if process["mode"] == "serialized":
        evidence.equal(label + ":off", projections, {"events": [], "visits": [], "completed": []})
        evidence.equal(label + ":observer-off", data["checkpoints"], [])
        return
    completed, events, visits = (projections[name] for name in ("completed", "events", "visits"))
    evidence.equal(label + ":counts", [len(completed), len(events), len(visits)], [len(steps), 5 * len(steps), len(steps)])
    receipts, sequences = {}, set()
    for index, (entry, visit) in enumerate(zip(completed, visits, strict=True)):
        receipt = entry["receipt"]
        key = (receipt["engine_id"], receipt["step_index"])
        name = label + ":receipt:" + str(index)
        evidence.fields(name + ":fields", receipt, "engine_id step_index sequence submitted_at_ps completed_at_ps")
        evidence.check(name + ":join", key in steps and key not in receipts
                       and integer(receipt["sequence"]) and receipt["sequence"] not in sequences)
        receipts[key] = receipt
        sequences.add(receipt["sequence"])
        step = steps[key]
        evidence.equal(name + ":result", entry["result"], step["result"])
        evidence.equal(name + ":input-start", receipt["submitted_at_ps"], step["record"]["virtual_time_ps"])
        evidence.equal(name + ":result-end", receipt["completed_at_ps"], step["result"]["completed_at_ps"])
        evidence.check(name + ":time-types", integer(receipt["submitted_at_ps"])
                       and integer(receipt["completed_at_ps"], receipt["submitted_at_ps"])
                       and integer(step["result"]["step_latency_ps"]))
        evidence.equal(name + ":service-interval", step["result"]["step_latency_ps"],
                       receipt["completed_at_ps"] - receipt["submitted_at_ps"])
        evidence.equal(name + ":visit", visit, {
            "execution_id": "engine-service:" + key[0], "operation_id": "step-" + str(key[1]),
            "resource": {"kind": "gpu-work-queue", "resource_id": key[0]},
            "submitted_at_ps": receipt["submitted_at_ps"], "eligible_at_ps": receipt["submitted_at_ps"],
            "started_at_ps": receipt["submitted_at_ps"], "finished_at_ps": receipt["completed_at_ps"],
            "completed_at_ps": receipt["completed_at_ps"], "service_bytes": 0, "subject_object_id": None,
            "stage": "declared-whole-engine-service"})
    evidence.equal(label + ":sequence-domain", sorted(sequences), list(range(len(steps))))
    pending, last_end, retired, submitted = {}, {}, [], []
    cursor, now = 0, 0
    while cursor < len(events):
        event = events[cursor]
        key = (event["resource"]["resource_id"], int(event["operation_id"].removeprefix("step-")))
        evidence.check(label + ":event-key:" + str(cursor), key in receipts)
        receipt = receipts[key]
        at = event["timestamp_ps"]
        evidence.check(label + ":monotonic:" + str(cursor), integer(at, now))
        now = at
        submit = event["phase"] == "submitted"
        phases = ("submitted", "queued", "started") if submit else ("progress", "completed")
        if submit:
            evidence.check(label + ":ready:" + str(cursor), key not in submitted and key[0] not in pending
                           and all(other["completed_at_ps"] > at for other in pending.values())
                           and at >= last_end.get(key[0], 0) and at == receipt["submitted_at_ps"])
            evidence.equal(label + ":insertion:" + str(cursor), receipt["sequence"], len(submitted))
            pending[key[0]] = receipt
            submitted.append(key)
        else:
            next_due = min(pending.values(), key=lambda item: (item["completed_at_ps"], item["sequence"]), default=None)
            evidence.equal(label + ":due:" + str(cursor), receipt, next_due)
            evidence.equal(label + ":due-time:" + str(cursor), at, receipt["completed_at_ps"])
            del pending[key[0]]
            last_end[key[0]] = at
            retired.append(key)
        for offset, phase in enumerate(phases):
            evidence.equal(label + ":event:" + str(cursor + offset), events[cursor + offset], {
                "execution_id": "engine-service:" + key[0], "operation_id": "step-" + str(key[1]),
                "phase": phase, "timestamp_ps": at, "resource": {"kind": "gpu-work-queue", "resource_id": key[0]},
                "completed_bytes": None, "subject_object_id": None})
        cursor += len(phases)
    evidence.check(label + ":all-retired", retired == list(receipts))
    evidence.equal(label + ":empty-pending", pending, {})


def check_cache_state(state, owners, engine, label, evidence, expected_blocks):
    evidence.fields(label + ":native-fields", state, "requests visible cache queues emitted")
    cache, requests, visible, queues = (state[name] for name in ("cache", "requests", "visible", "queues"))
    evidence.fields(label + ":cache-fields", cache, "groups blocks free_queue")
    blocks = dict(cache["blocks"])
    evidence.check(label + ":blocks", len(blocks) == len(cache["blocks"]) > 0
                   and all(integer(key) and integer(refs) for key, refs in cache["blocks"]))
    evidence.equal(label + ":block-domain", list(blocks), list(range(expected_blocks)))
    evidence.equal(label + ":group-domain", len(cache["groups"]), 1)
    owned = Counter(block for group in cache["groups"] for ids in group.values() for block in ids)
    # The pinned native pool reserves block zero as its unreferenced null block.
    evidence.check(label + ":no-null-owner", 0 not in owned and blocks[0] == 0)
    evidence.check(label + ":reference-conservation", all(blocks[block] == owned[block] for block in blocks if block != 0))
    free = cache["free_queue"]
    evidence.equal(label + ":free-domain", sorted(free), sorted(block for block, count in blocks.items() if block != 0 and count == 0))
    evidence.check(label + ":free-queue", len(free) == len(set(free))
                   and all(type(block) is int and block in blocks and blocks[block] == 0 for block in free))
    evidence.check(label + ":owner-domain", all((engine, key) in owners for key in requests)
                   and all((engine, key) in owners for key in visible))
    for index, group in enumerate(cache["groups"]):
        evidence.check(label + ":group:" + str(index), all(
            (engine, request) in owners and len(ids) == len(set(ids))
            and all(type(block) is int and block in blocks for block in ids)
            for request, ids in group.items()))
    evidence.fields(label + ":queue-fields", queues, "waiting running")
    queued = [*queues["waiting"], *queues["running"]]
    evidence.check(label + ":queues", len(queued) == len(set(queued))
                   and all(key in requests for key in queued) and len(queues["running"]) <= 1)
    for request, row in requests.items():
        evidence.fields(label + ":request-fields:" + request, row,
                        "status num_computed_tokens num_in_flight_tokens output_token_ids all_token_ids stop_reason kv_transfer_params")
        evidence.check(label + ":token-types:" + request, integer(row["num_computed_tokens"]) and integer(row["num_in_flight_tokens"]))
    for request, row in visible.items():
        evidence.fields(label + ":visible-fields:" + request, row, "external_request_id output_token_ids is_prefilling sent_tokens_offset")
        evidence.equal(label + ":visible-owner:" + request, row["external_request_id"], owners[(engine, request)])


def check_checkpoints(data, process, steps, owners, evidence, *, expected_cache_blocks=64):
    if process["mode"] != "independent":
        return
    prefix = process["id"] + ":checkpoint"
    rows = data["checkpoints"]
    evidence.equal(prefix + ":count", len(rows), 4 * len(steps))
    public_rows = {row["result"]["request_id"]: row["result"] for row in
                   [*data.get("controls", []), *[row for cell in data.get("cells", []) for row in cell["requests"]]]}
    grouped, checkpoint_events = defaultdict(list), []
    last_retired = {}
    memberships = defaultdict(list)
    for key, step in steps.items():
        for item in step["record"]["scheduled"]:
            memberships[(key[0], item["request_id"])].append(key)
    for index, row in enumerate(rows):
        name = prefix + ":" + str(index)
        evidence.fields(name + ":fields", row,
                        "index phase engine_id at_ps step_index record_count result_count sink_counts sink_sha256 native")
        evidence.equal(name + ":index", row["index"], index)
        key = (row["engine_id"], row["step_index"])
        evidence.check(name + ":key", key in steps)
        grouped[key].append(row)
        if row["phase"] == "before-submit" and key[0] in last_retired:
            check_waiting_admissions(last_retired[key[0]]["native"], row["native"], name + ":between-steps", evidence)
        if row["phase"] == "retired":
            last_retired[key[0]] = row
        else:
            evidence.equal(name + ":no-early-native-output", row["native"]["emitted"], [])
        check_cache_state(row["native"], owners, key[0], name, evidence, expected_cache_blocks)
        for request in row["native"]["requests"]:
            if owners[(key[0], request)] in public_rows:
                raw = public_rows[owners[(key[0], request)]]
                earliest = raw["admitted_at_ps"] if raw["prefill_engine_id"] == key[0] else raw["handoff"]["completed_at_ps"]
                evidence.check(name + ":native-admission:" + request, row["at_ps"] >= earliest)
        for index, output in enumerate(row["native"]["emitted"]):
            evidence.fields(name + ":emitted-fields:" + str(index), output, "request_id finished token_ids kv_transfer_params")
            if output["request_id"] in public_rows:
                raw = public_rows[output["request_id"]]
                expected_params = raw["kv_transfer_params"] if raw["prefill_engine_id"] == key[0] else None
                evidence.equal(name + ":emitted-handoff:" + str(index), output["kv_transfer_params"], expected_params)
        evidence.equal(name + ":sink-domain", sorted(row["sink_counts"]), sorted(PUBLICATIONS))
        evidence.equal(name + ":sink-hash-domain", sorted(row["sink_sha256"]), sorted(PUBLICATIONS))
        for collection in PUBLICATIONS:
            count = row["sink_counts"][collection]
            all_rows = data["sinks"][key[0]][collection]
            evidence.check(name + ":sink-count:" + collection, integer(count) and count <= len(all_rows))
            evidence.equal(name + ":sink-prefix:" + collection, row["sink_sha256"][collection],
                           sha(exact_json_bytes(all_rows[:count])))
        if row["phase"] in ("submitted", "retired"):
            phases = ("submitted", "queued", "started") if row["phase"] == "submitted" else ("progress", "completed")
            checkpoint_events.extend([key[0], key[1], phase, row["at_ps"]] for phase in phases)
    evidence.equal(prefix + ":event-order", checkpoint_events, [
        [row["resource"]["resource_id"], int(row["operation_id"].removeprefix("step-")), row["phase"], row["timestamp_ps"]]
        for row in data["projections"]["events"]])
    for key, step in steps.items():
        name = prefix + ":" + key[0] + ":" + str(key[1])
        captures = grouped[key]
        evidence.equal(name + ":phases", [row["phase"] for row in captures],
                       ["before-submit", "submitted", "before-retire", "retired"])
        before, submitted, due, retired = captures
        start, end = step["record"]["virtual_time_ps"], step["result"]["completed_at_ps"]
        evidence.equal(name + ":times", [row["at_ps"] for row in captures], [start, start, end, end])
        evidence.equal(name + ":record-count", [row["record_count"] for row in captures], [key[1], key[1] + 1, key[1] + 1, key[1] + 1])
        evidence.equal(name + ":result-count", [row["result_count"] for row in captures], [key[1], key[1], key[1], key[1] + 1])
        for field in ("sink_counts", "sink_sha256"):
            evidence.check(name + ":unpublished:" + field, before[field] == submitted[field] == due[field])
        # Native scheduling may reserve new blocks and in-flight input tokens.
        evidence.equal(name + ":no-submit-visible-output", before["native"]["visible"], submitted["native"]["visible"])
        for request, old in before["native"]["requests"].items():
            current = submitted["native"]["requests"].get(request)
            evidence.check(name + ":no-submit-completion:" + request, current is not None
                           and all(current[field] == old[field] for field in ("output_token_ids", "all_token_ids", "stop_reason")))
        old_blocks = dict(before["native"]["cache"]["blocks"])
        new_blocks = dict(submitted["native"]["cache"]["blocks"])
        evidence.check(name + ":no-submit-release", all(new_blocks[block] >= refs for block, refs in old_blocks.items()))
        # New waiting requests may arrive while this engine's receipt is pending.
        frozen, current = submitted["native"], due["native"]
        evidence.equal(name + ":pending-cache", current["cache"], frozen["cache"])
        for section in ("requests", "visible"):
            evidence.equal(name + ":pending:" + section,
                           {key: current[section].get(key) for key in frozen[section]}, frozen[section])
            added = set(current[section]) - set(frozen[section])
            evidence.check(name + ":new-waiters:" + section, all(
                current[section][request]["output_token_ids"] == [] for request in added))
        added = set(current["requests"]) - set(frozen["requests"])
        evidence.check(name + ":waiting-reservations", all(
            current["requests"][request]["status"] == "WAITING"
            and current["requests"][request]["num_computed_tokens"] == current["requests"][request]["num_in_flight_tokens"] == 0
            for request in added))
        evidence.equal(name + ":pending-running", current["queues"]["running"], frozen["queues"]["running"])
        evidence.equal(name + ":pending-waiting", [request for request in current["queues"]["waiting"] if request not in added],
                       frozen["queues"]["waiting"])
        positive = bool(step["record"]["scheduled"])
        for collection in PUBLICATIONS:
            expected_delta = int(positive and collection in ("outcomes", "locality_outcomes", "collective_timing_outcomes"))
            evidence.equal(name + ":retired-publication:" + collection,
                           retired["sink_counts"][collection] - due["sink_counts"][collection], expected_delta)
        scheduled = [row["request_id"] for row in step["record"]["scheduled"]]
        emitted = []
        for capture in (before, submitted, due):
            for section in ("requests", "visible"):
                evidence.check(name + ":scheduled-presence:" + capture["phase"] + ":" + section,
                               all(request in capture["native"][section] for request in scheduled))
        for item in step["record"]["scheduled"]:
            request = item["request_id"]
            old = current["requests"][request]
            evidence.equal(name + ":scheduled-in-flight:" + request, old["num_in_flight_tokens"], item["num_new_tokens"])
            after = retired["native"]["requests"].get(request)
            visible = retired["native"]["visible"].get(request)
            visits = memberships[(key[0], request)]
            visit = visits.index(key)
            finished = key == visits[-1]
            evidence.equal(name + ":prior-token-count:" + request, old["output_token_ids"], [512] * visit)
            if finished:
                evidence.check(name + ":terminal-removal:" + request, after is None and visible is None)
            else:
                evidence.check(name + ":live-retired-token:" + request, after is not None and visible is not None
                               and after["output_token_ids"] == visible["output_token_ids"] == [512] * (visit + 1)
                               and after["num_in_flight_tokens"] == 0)
            emitted.append({"request_id": owners[(key[0], request)], "finished": finished, "token_ids": [512] * (visit + 1)})
        for request, old in current["requests"].items():
            after = retired["native"]["requests"].get(request)
            visible = retired["native"]["visible"].get(request)
            if request not in scheduled:
                evidence.equal(name + ":other-request:" + request, after, old)
                evidence.equal(name + ":other-visible:" + request, visible, current["visible"].get(request))
        evidence.equal(name + ":native-emitted", [{field: row[field] for field in ("request_id", "finished", "token_ids")}
                       for row in retired["native"]["emitted"]], emitted)

        if not positive:
            evidence.equal(name + ":drain-no-state-change", retired["native"], current)


def check_selected_envelopes(data, label, evidence):
    from dataclasses import asdict

    from examples.independent_engine_completion_v1.common import json_value
    from simllm.compute import GPU_ENVELOPES, HostInitiationModel
    from simllm.traffic.collective_latency import resolve_collective_fixed_cost_envelope

    envelope = resolve_collective_fixed_cost_envelope("intra-node-fixed-cost-v1")
    profile = envelope.arm_profile("lower")
    for selected in data["selected_before"]:
        name = label + ":selection:" + selected["engine_id"]
        evidence.equal(name + ":host", selected["host_model"], json_value(asdict(HostInitiationModel.ideal())))
        evidence.equal(name + ":gpu", selected["gpu"], json_value(asdict(GPU_ENVELOPES["b100"])))
        evidence.equal(name + ":collective", selected["collective_profile"], json_value(asdict(profile)))
    return profile, envelope


def check_sinks_and_domains(data, process, steps, evidence):
    label = process["id"] + ":accounting"
    engines = [row["engine_id"] for row in data["retained_before"]]
    evidence.equal(label + ":sink-engines", sorted(data["sinks"]), sorted(engines))
    profile, envelope = check_selected_envelopes(data, label, evidence)
    for engine in engines:
        local = [step for (owner, index), step in steps.items() if owner == engine]
        name = label + ":" + engine
        evidence.equal(name + ":dense-step-indices", [row["record"]["step_index"] for row in local], list(range(len(local))))
        positive = [row for row in local if row["record"]["scheduled"]]
        publication = data["sinks"][engine]
        evidence.equal(name + ":publication-domain", sorted(publication), sorted(PUBLICATIONS))
        for collection in PUBLICATIONS:
            expected = [row["record"]["step_index"] for row in positive] if collection in (
                "outcomes", "locality_outcomes", "collective_timing_outcomes") else []
            evidence.equal(name + ":published-steps:" + collection, [row["step_index"] for row in publication[collection]], expected)
        for index, (step, outcome, locality) in enumerate(zip(positive, publication["outcomes"], publication["locality_outcomes"], strict=True)):
            tag = name + ":outcome:" + str(index)
            check_sink_decomposition(step, outcome, locality, publication["collective_timing_outcomes"][index],
                                     profile, envelope, tag, evidence)
            evidence.equal(tag + ":owned-service", outcome["makespan_ps"], step["result"]["step_latency_ps"])
            evidence.check(tag + ":isolated-host-identity", outcome["host_profile"] == "ideal"
                           and outcome["host_launch_count"] == outcome["host_launch_floor_ps"] == outcome["exposed_host_ps"] == 0
                           and outcome["num_flows"] == locality["backend_runs"] == 0 and outcome["quiescent"] is True)
        previous = 0
        for step in local:
            evidence.check(name + ":no-overlap:" + str(step["record"]["step_index"]),
                           step["record"]["virtual_time_ps"] >= previous)
            previous = step["result"]["completed_at_ps"]
        cursor = data["cells"][0]["record_starts"][engine]
        for index, cell in enumerate(data["cells"]):
            evidence.equal(name + ":cell-start:" + str(index), cell["record_starts"][engine], cursor)
            cursor = cell["record_stops"][engine]
            evidence.check(name + ":cell-stop:" + str(index), integer(cursor) and cursor <= len(local))
        evidence.equal(name + ":cell-coverage", cursor, len(local))
    advances = data["clock_advances"]
    previous = 0
    for index, row in enumerate(advances):
        evidence.fields(label + ":advance-fields:" + str(index), row, "before_ps after_ps")
        evidence.equal(label + ":clock-continuity:" + str(index), row["before_ps"], previous)
        evidence.check(label + ":clock-forward:" + str(index), integer(row["after_ps"], previous))
        previous = row["after_ps"]
    evidence.equal(label + ":clock-final", previous, data["final_clock_ps"])
    cell_advances = [index for cell in data["cells"] for index in cell["clock_advance_indices"]]
    control_advances = []
    for control in data["controls"]:
        raw = control["result"]
        control_advances.append({"before_ps": raw["admitted_at_ps"], "after_ps": raw["admitted_at_ps"]})
        times = [raw["admitted_at_ps"], raw["prefill_completed_at_ps"], raw["handoff"]["completed_at_ps"],
                 *raw["decode_token_completed_at_ps"]]
        control_advances.extend({"before_ps": a, "after_ps": b} for a, b in pairwise(times))
    evidence.equal(label + ":control-advances", advances[:len(control_advances)], control_advances)
    evidence.equal(label + ":all-advance-indices", cell_advances, list(range(len(control_advances), len(advances))))
    expected_start = data["controls"][-1]["result"]["decode_token_completed_at_ps"][-1] if data["controls"] else 0
    for index, cell in enumerate(data["cells"]):
        evidence.equal(label + ":cell-continuity:" + str(index), cell["start_ps"], expected_start)
        expected_start = cell["end_ps"]
    if process["mode"] == "independent":
        for member, plural in (("event_indices", "events"), ("visit_indices", "visits")):
            evidence.equal(label + ":" + member, [index for cell in data["cells"] for index in cell[member]],
                           list(range(len(data["projections"][plural]))))


def check_native_relations(all_data, frozen, evidence):
    def cell(width, prompt, handoff):
        process = all_data[f"independent-p{width}-d{width}"]
        return next(row for row in process["cells"] if row["id"] == f"burst-prompt{prompt}-handoff{handoff}")

    def relative_tokens(row, start):
        return [at - start for at in row["result"]["decode_token_completed_at_ps"]]

    def makespan(row):
        return row["end_ps"] - row["start_ps"]

    for a, b in ((1, 2), (2, 4)):
        for prompt in (8, 16):
            service = frozen["known_services_ps"][str(prompt)]["decode"][0]
            for handoff in (100000000, 200000000):
                before, after = cell(a, prompt, handoff), cell(b, prompt, handoff)
                first, last = makespan(before), makespan(after)
                expected_first = frozen["known_services_ps"][str(prompt)]["prefill"] + handoff + 16 // a * service
                expected_last = frozen["known_services_ps"][str(prompt)]["prefill"] + handoff + 16 // b * service
                evidence.check(f"p{a}-to-p{b}-prompt{prompt}-handoff{handoff}",
                               first - last == (16 // a - 16 // b) * service
                               and Fraction(4 * 10**12, first) == Fraction(4 * 10**12, expected_first)
                               and Fraction(4 * 10**12, last) == Fraction(4 * 10**12, expected_last),
                               kind="relations", family="native-width")
    for width in (1, 2, 4):
        for prompt in (8, 16):
            first, last = cell(width, prompt, 100000000), cell(width, prompt, 200000000)
            shifts = [makespan(last) - makespan(first)]
            for a, b in zip(first["requests"], last["requests"], strict=True):
                shifts.append(b["result"]["ttft_ps"] - a["result"]["ttft_ps"])
                shifts.extend(y - x for x, y in zip(relative_tokens(a, first["start_ps"]), relative_tokens(b, last["start_ps"]), strict=True))
                evidence.equal(f"handoff-tpot:p{width}:prompt{prompt}:" + a["result"]["request_id"],
                               a["result"]["tpot_ps"], b["result"]["tpot_ps"])
            evidence.check(f"p{width}-prompt{prompt}", all(shift == 100000000 for shift in shifts),
                           kind="relations", family="native-handoff")
    burst = cell(1, 8, 100000000)
    for entry in all_data["independent-p1-d1"]["cells"]:
        if not entry["id"].startswith("arrival-"):
            continue
        row, baseline = entry["requests"][1], burst["requests"][1]
        arrival = row["result"]["admitted_at_ps"] - entry["start_ps"]
        evidence.check("arrival-" + str(arrival),
                       relative_tokens(row, entry["start_ps"]) == relative_tokens(baseline, burst["start_ps"])
                       and row["result"]["ttft_ps"] == baseline["result"]["ttft_ps"] - arrival
                       and row["result"]["prefill_eligible_at_ps"] - entry["start_ps"]
                       == max(arrival, frozen["known_services_ps"]["8"]["prefill"]),
                       kind="relations", family="native-arrival")
    mixed = {}
    for mode in ("serialized", "independent"):
        entry = next(row for row in all_data[f"{mode}-p2-d2"]["cells"] if row["id"] == "later-short-prompt")
        mixed[mode] = [relative_tokens(row, entry["start_ps"])[0] for row in entry["requests"]]
    evidence.check("later-short-prompt", mixed["independent"] == [292912000, 274376000]
                   and mixed["serialized"] == [292912000, 448840000],
                   kind="relations", family="native-mixed-order")
    evidence.finish("native-relations")


def check_native_memberships(data, process, steps, owners, evidence):
    prefix = process["id"] + ":membership"
    requests = {row["result"]["request_id"]: row for row in [
        *data["controls"], *[row for cell in data["cells"] for row in cell["requests"]]]}
    memberships = defaultdict(list)
    for key, step in steps.items():
        record = step["record"]
        name = prefix + ":" + key[0] + ":" + str(key[1])
        evidence.equal(name + ":sample-count", record["num_sampled"], len(record["scheduled"]))
        evidence.equal(name + ":no-preemption", record["preempted_request_ids"], [])
        evidence.check(name + ":finished-unique", len(set(record["finished_request_ids"])) == len(record["finished_request_ids"]))
        for item in record["scheduled"]:
            identifier = (key[0], item["request_id"])
            raw = requests[owners[identifier]]["result"]
            role = "prefill" if raw["prefill_engine_id"] == key[0] else "decode"
            visit = len(memberships[identifier])
            prompt = raw["kv_transfer_params"]["remote_num_tokens"]
            evidence.equal(name + ":token-contract", item, {
                "request_id": identifier[1], "phase": "prefill" if role == "prefill" or visit == 0 else "decode",
                "num_new_tokens": prompt if role == "prefill" else 1,
                "num_cached_tokens": prompt if role == "decode" and visit == 0 else 0,
                "context_length": prompt if role == "prefill" else prompt + visit + 1})
            memberships[identifier].append(key)
    for engine in data["retained_before"]:
        name = engine["engine_id"]
        previous_finished = []
        for key, step in steps.items():
            if key[0] != name:
                continue
            record = step["record"]
            evidence.equal(prefix + ":finished-lag:" + name + ":" + str(key[1]), record["finished_request_ids"], previous_finished)
            previous_finished = sorted(item["request_id"] for item in record["scheduled"]
                                       if memberships[(name, item["request_id"])][-1] == key)
        if process["mode"] == "independent":
            evidence.equal(prefix + ":final-drain:" + name, previous_finished, [])
    for identifier, public in owners.items():
        row, engine = requests[public], identifier[0]
        raw = row["result"]
        role = "prefill" if raw["prefill_engine_id"] == engine else "decode"
        keys = memberships[identifier]
        name = prefix + ":" + public + ":" + role
        evidence.equal(name + ":count", len(keys), 1 if role == "prefill" else 4)
        indices = row[role + "_record_indices"]
        evidence.check(name + ":all-owned-in-slice", all(key[1] in indices for key in keys))
        cell = next((cell for cell in data["cells"] if any(item["result"]["request_id"] == public for item in cell["requests"])), None)
        first_in_cell = cell["record_starts"][engine] if cell is not None else 0
        earliest_admission = raw["admitted_at_ps"] if role == "prefill" else raw["handoff"]["completed_at_ps"]
        first = min(index for (owner, index), step in steps.items() if owner == engine and index >= first_in_cell
                    and step["record"]["virtual_time_ps"] >= earliest_admission)
        evidence.equal(name + ":exact-slice", indices, list(range(first, keys[-1][1] + 1)))
        starts = [steps[key]["record"]["virtual_time_ps"] for key in keys]
        completions = [steps[key]["result"]["completed_at_ps"] for key in keys]
        evidence.equal(name + ":release", starts[0], raw[role + "_eligible_at_ps"])
        evidence.equal(name + ":visible", completions, [raw["prefill_completed_at_ps"]] if role == "prefill"
                       else raw["decode_token_completed_at_ps"])
        earliest = raw["admitted_at_ps"] if role == "prefill" else raw["handoff"]["completed_at_ps"]
        evidence.check(name + ":bounded-slice", all(
            earliest <= steps[(engine, index)]["record"]["virtual_time_ps"]
            <= steps[(engine, index)]["result"]["completed_at_ps"] <= completions[-1] for index in indices))


def check_waiting_admissions(before, after, label, evidence):
    """Between owned steps, only freshly appended waiting requests may appear."""
    evidence.equal(label + ":cache", after["cache"], before["cache"])
    added = set(after["requests"]) - set(before["requests"])
    evidence.check(label + ":new-visible-domain", set(after["visible"]) - set(before["visible"]) == added)
    for section in ("requests", "visible"):
        evidence.equal(label + ":retained:" + section,
                       {key: after[section].get(key) for key in before[section]}, before[section])
    evidence.check(label + ":new-waiting", all(
        after["requests"][key]["status"] == "WAITING"
        and after["requests"][key]["num_computed_tokens"] == after["requests"][key]["num_in_flight_tokens"] == 0
        and after["requests"][key]["output_token_ids"] == after["visible"][key]["output_token_ids"] == []
        for key in added))
    evidence.equal(label + ":running", after["queues"]["running"], before["queues"]["running"])
    waiting = after["queues"]["waiting"]
    previous = before["queues"]["waiting"]
    evidence.equal(label + ":waiting-prefix", waiting[:len(previous)], previous)
    evidence.check(label + ":waiting-suffix", set(waiting[len(previous):]) == added
                   and len(waiting) == len(previous) + len(added))


def check_component_checkpoints(row, spec, evidence):
    label = "component-detail:" + spec["id"]
    inputs, checkpoints = row["inputs"], row["checkpoints"]
    if spec["kind"] == "equal-work":
        expected_ids = [str(index) for index in range(spec["work_items"])]
        jobs = [(name, name if spec["resource_mode"] == "independent" else "serial",
                 0 if spec["resource_mode"] == "independent" else index * spec["service_ps"], spec["service_ps"])
                for index, name in enumerate(expected_ids)]
        expected_outputs = [{"id": name, "at_ps": spec["service_ps"] * (index + 1 if spec["resource_mode"] == "serial" else 1)}
                            for index, name in enumerate(expected_ids)]
    elif spec["kind"] == "unequal-arrival":
        expected_ids = ["long", "short"]
        jobs = [("long", "long", 0, spec["long_service_ps"]),
                ("short", "short", spec["short_arrival_ps"], spec["short_service_ps"])]
        expected_outputs = [{"id": "short", "at_ps": spec["short_arrival_ps"] + spec["short_service_ps"]},
                            {"id": "long", "at_ps": spec["long_service_ps"]}]
    elif spec["kind"] == "tie":
        expected_ids, expected_outputs = ["a", "b", "c"], [{"id": name, "at_ps": at} for name, at in (("a", 4000), ("b", 4000), ("c", 5000))]
        jobs = [(job["id"], job["engine"], job["arrival_ps"], job["service_ps"]) for job in spec["jobs"]]
    else:
        expected_ids, expected_outputs = ["drain"], [{"id": "drain", "at_ps": 3000}]
        jobs = [("drain", "drain", spec["at_ps"], 0)]
    evidence.equal(label + ":input-domain", [entry["job_id"] for entry in inputs], expected_ids)
    for index, (job, entry) in enumerate(zip(jobs, inputs, strict=True)):
        identifier, engine, start, service = job
        evidence.equal(label + ":frozen-input:" + identifier, entry, {
            "job_id": identifier, "engine_id": engine,
            "record": {"schema": "atlahs-closed-loop-step-v1", "step_index": index,
                       "virtual_time_ps": start,
                       "scheduled": [] if spec["kind"] == "drain" else [{"request_id": identifier,
                           "phase": "decode", "num_new_tokens": 1, "num_cached_tokens": 0, "context_length": 0}],
                       "preempted_request_ids": [], "finished_request_ids": spec.get("finished_request_ids", [])},
            "result": {"step_index": index, "step_latency_ps": service, "completed_at_ps": start + service,
                       "request_metrics": [], "additive_visit_totals": None},
            "receipt": {"engine_id": engine, "step_index": index, "sequence": index,
                        "submitted_at_ps": start, "completed_at_ps": start + service}})
    evidence.equal(label + ":output-bijection", row["outputs"], expected_outputs)
    steps = {(entry["engine_id"], entry["record"]["step_index"]): entry for entry in inputs}
    check_projections({"projections": {key: row[key] for key in ("events", "visits", "completed")}},
                      {"id": label, "mode": "independent"}, steps, evidence)
    evidence.equal(label + ":checkpoint-count", len(checkpoints), 3 * len(inputs))
    expected_end = {entry["id"]: entry["at_ps"] for entry in expected_outputs}
    submitted_count, retired_sequences = 0, []
    for index, capture in enumerate(checkpoints):
        if capture["phase"] == "submitted":
            submitted_count += 1
        elif capture["phase"] == "retired":
            retired_sequences.append(capture["sequence"])
        evidence.equal(label + ":checkpoint-events:" + str(index), capture["event_count"],
                       3 * submitted_count + 2 * len(retired_sequences))
        evidence.equal(label + ":checkpoint-results:" + str(index), capture["result_sequences"], retired_sequences)
    for index, entry in enumerate(inputs):
        receipt = entry["receipt"]
        name = label + ":" + entry["job_id"]
        evidence.equal(name + ":owned-receipt", receipt, next(item["receipt"] for item in row["completed"]
                       if item["receipt"]["sequence"] == index))
        evidence.equal(name + ":due", receipt["completed_at_ps"], expected_end[entry["job_id"]])
        captures = [capture for capture in checkpoints if capture["sequence"] == index]
        evidence.equal(name + ":checkpoint-domain", [capture["phase"] for capture in captures], ["submitted", "before-due", "retired"])
        for capture in captures:
            at = capture["at_ps"]
            if capture["phase"] == "submitted":
                evidence.equal(name + ":submitted-time", at, receipt["submitted_at_ps"])
            elif capture["phase"] == "before-due":
                evidence.equal(name + ":before-time", at, max(receipt["submitted_at_ps"], receipt["completed_at_ps"] - 1))
            else:
                evidence.equal(name + ":retired-time", at, receipt["completed_at_ps"])
            completed = capture["result_sequences"]
            evidence.check(name + ":" + capture["phase"] + ":own-visibility",
                           (index in completed) == (capture["phase"] == "retired"))
            evidence.equal(name + ":" + capture["phase"] + ":publication-counts",
                           [capture["visit_count"], len(capture["outputs"])], [len(completed)] * 2)
            evidence.equal(name + ":" + capture["phase"] + ":outputs", capture["outputs"], [
                {"id": inputs[number]["job_id"], "at_ps": inputs[number]["receipt"]["completed_at_ps"]} for number in completed])
            evidence.check(name + ":" + capture["phase"] + ":due-only",
                           len(completed) == len(set(completed)) and all(inputs[number]["receipt"]["completed_at_ps"] <= at for number in completed))
            submitted_by_now = sum(other["receipt"]["submitted_at_ps"] < at for other in inputs)
            evidence.check(name + ":" + capture["phase"] + ":event-count",
                           capture["event_count"] >= 3 * submitted_by_now + 2 * len(completed)
                           and capture["event_count"] <= 3 * len(inputs) + 2 * len(completed))


def check_sink_decomposition(step, outcome, locality, timing, profile, envelope, label, evidence):
    from dataclasses import fields

    from examples.pd_session_v1 import run_study as baseline
    from simllm.backends.step_sink import (
        CollectiveArtifactTiming,
        StepCollectiveTimingOutcome,
        StepLocalityOutcome,
        StepNetworkOutcome,
    )
    from simllm.compute import GPU_ENVELOPES, HostInitiationModel, RooflineProvider, step_kernel

    for name, row, cls in (("network", outcome, StepNetworkOutcome), ("locality", locality, StepLocalityOutcome),
                           ("collective", timing, StepCollectiveTimingOutcome)):
        evidence.fields(label + ":fields:" + name, row, " ".join(field.name for field in fields(cls)))
    expected_profile = {
        "profile_id": profile.profile_id, "bandwidth_bytes_per_second": profile.bandwidth_bytes_per_second,
        "participant_latency_ps": [list(row) for row in profile.participant_latency_ps],
        "propagation_reference_ps": profile.propagation_reference_ps,
        "envelope_id": "intra-node-fixed-cost-v1", "arm": "lower", "evidence_class": envelope.arm_evidence_class("lower"),
    }
    evidence.equal(label + ":profile", {key: timing[key] for key in expected_profile}, expected_profile)
    for index, artifact in enumerate(timing["artifacts"]):
        evidence.fields(label + ":artifact-fields:" + str(index), artifact, " ".join(field.name for field in fields(CollectiveArtifactTiming)))
    StepCollectiveTimingOutcome(**{**timing, "artifacts": tuple(CollectiveArtifactTiming(**row) for row in timing["artifacts"])})
    artifacts = timing["artifacts"]
    evidence.check(label + ":artifact-identity", len(artifacts) > 0
                   and len({row["artifact_id"] for row in artifacts}) == len(artifacts))
    evidence.equal(label + ":artifact-count", locality["artifact_count"], len(artifacts))
    projections = {
        "artifact_operation_ids": [row["operation_ids"] for row in artifacts],
        "local_phase_service_ps": [row["local_service_ps"] for row in artifacts],
        "fabric_phase_service_ps": [row["fabric_transport_ps"] for row in artifacts],
        "base_phase_latency_ps": [row["collective_base_latency_ps"] for row in artifacts],
        "composed_phase_service_ps": [row["composed_service_ps"] for row in artifacts],
        "local_phase_medium": ["gpu-compute" if row["collective_operation_id"] is None else "nvlink" for row in artifacts],
    }
    for field, values in projections.items():
        evidence.equal(label + ":artifact-projection:" + field, locality[field], values)
    evidence.equal(label + ":composed-service", sum(row["composed_service_ps"] for row in artifacts), step["result"]["step_latency_ps"])
    evidence.check(label + ":isolated-terms", all(row["fabric_transport_ps"] == row["registration_cost_ps"] == 0 for row in artifacts))
    evidence.equal(label + ":floor-off", locality["collective_floor_phase_ps"], [])
    evidence.equal(label + ":registration-off", locality["registration_phase_cost_ps"], [])
    evidence.equal(label + ":local-order", [locality["authority"], locality["ordering_authority"], locality["graph_execution_id"]],
                   ["placement-manifest", "execution-graph", "step-" + str(step["record"]["step_index"])])
    evidence.check(label + ":local-path", locality["compatibility_fast_path"] is False
                   and locality["fabric_directed_bytes"] == locality["fabric_segments"] == locality["backend_runs"] == 0
                   and locality["total_directed_bytes"] == locality["nvlink_directed_bytes"])
    evidence.equal(label + ":bandwidth", locality["nvlink_bandwidth_bytes_per_second"], profile.bandwidth_bytes_per_second)
    groups = defaultdict(list)
    for row in artifacts:
        if row["collective_operation_id"] is not None:
            groups[row["collective_operation_id"]].append(row)
    collective = [row for rows in groups.values() for row in rows]
    evidence.equal(label + ":local-service-sum", locality["nvlink_service_ps"], sum(row["local_service_ps"] for row in collective))
    evidence.equal(label + ":compute-service-sum", locality["compute_service_ps"],
                   sum(row["local_service_ps"] for row in artifacts if row["collective_operation_id"] is None))
    evidence.equal(label + ":phase-count", locality["phase_count"], len(collective))
    endpoint_bytes = 0
    for index, rows in enumerate(groups.values()):
        endpoint = rows[0]["critical_endpoint_bytes"]
        evidence.equal(label + ":width:" + str(index), rows[0]["participant_count"], 8)
        endpoint_bytes += endpoint
        numerator = endpoint * 10**12
        floor = (numerator + profile.bandwidth_bytes_per_second - 1) // profile.bandwidth_bytes_per_second
        service = sum(row["local_service_ps"] for row in rows)
        # The accepted local serializer rounds each phase to a whole nanosecond.
        ceiling = ((endpoint * 10**9 + profile.bandwidth_bytes_per_second - 1)
                   // profile.bandwidth_bytes_per_second + len(rows) - 1) * 1000
        evidence.check(label + ":serializer-bound:" + str(index), floor <= service <= ceiling)
    evidence.equal(label + ":balanced-ring-bytes", locality["nvlink_directed_bytes"], endpoint_bytes * 8)
    evidence.equal(label + ":ring-segments", locality["nvlink_segments"], len(collective) * 8)
    record = step_record_from_json(step["record"])
    dims, gpu = baseline._granite_dims(), GPU_ENVELOPES["b100"]
    estimate = RooflineProvider(efficiency=0.7).estimate(step_kernel(dims, record, num_sampled=record.num_sampled), gpu)
    host = HostInitiationModel.ideal().represented_estimate(estimate, gpu)
    expected_host = {
        "compute_estimate_ps": host.duration_ps, "provider_compute_ps": host.provider_duration_ps,
        "host_profile": "ideal", "host_launch_class": "none", "host_launch_count": 0,
        "host_launch_floor_ps": 0, "host_launch_floor_lower_ps": 0, "host_launch_floor_upper_ps": 0,
        "host_empirical_lower_ps": host.empirical_lower_ps, "host_empirical_upper_ps": host.empirical_upper_ps,
        "exposed_host_ps": 0, "represented_bound": host.bound,
        "num_sampled": 1, "sample_count_exact": True, "routing_mode": "none", "placement_epoch": None,
    }
    evidence.equal(label + ":host-service", {key: outcome[key] for key in expected_host}, expected_host)
    layer_ns = host.duration_ps // (dims.num_layers * 1000)
    evidence.equal(label + ":uniform-layer-work", outcome["layer_calc_ns"], [layer_ns] * dims.num_layers)
    evidence.equal(label + ":uniform-layer-price", outcome["per_layer_calc_ns"], layer_ns)
    evidence.equal(label + ":compute-quantization", locality["compute_service_ps"], dims.num_layers * max(layer_ns, 1) * 1000)
    evidence.check(label + ":graph-counts", all(integer(locality[field], 1) for field in (
        "effective_dependency_edge_count", "graph_artifact_count", "boundary_edge_count", "serialized_edge_count")))
