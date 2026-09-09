"""Reconcile native packet joins with complete serving steps and request time."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from fractions import Fraction

from examples.independent_engine_completion_v1.checks import (
    check_checkpoints,
    check_engines,
    check_native_memberships,
    check_projections,
    check_sinks_and_domains,
    expected_times,
    request_times,
)
from examples.independent_engine_completion_v1.reference import native_reference
from examples.pd_session_target_scale_v1.checks import integer
from examples.pd_session_v1 import run_study as baseline
from simllm.core.step import StepResult, step_record_from_json, step_record_to_json

from .aliases import admit_bindings, comparison, request_rows
from .journal_checks import check_journals, check_serialized
from .network_checks import check_network, geometry
from .order_checks import check_order


def request_metrics(raw, spec, frozen, label, evidence):
    evidence.fields(label + ":fields", raw, "schema request_id admitted_at_ps prefill_eligible_at_ps "
        "prefill_completed_at_ps handoff decode_eligible_at_ps decode_token_completed_at_ps ttft_ps tpot_ps "
        "decomposition prefill_engine_id decode_engine_id prefill_internal_request_id decode_internal_request_id "
        "bootstrap_token_id decode_token_ids kv_transfer_params prefill_step_count decode_step_count")
    evidence.equal(label + ":schema", raw["schema"], "simllm-pd-session-result-v1")
    handoff, tokens = raw["handoff"], raw["decode_token_completed_at_ps"]
    times = [raw["admitted_at_ps"], raw["prefill_eligible_at_ps"], raw["prefill_completed_at_ps"],
             handoff["submitted_at_ps"], handoff["eligible_at_ps"], handoff["completed_at_ps"],
             raw["decode_eligible_at_ps"], *tokens]
    evidence.check(label + ":causal-times", all(integer(value) for value in times)
                   and times == sorted(times) and len(tokens) == 4)
    evidence.equal(label + ":tokens", raw["decode_token_ids"], [512] * 4)
    evidence.equal(label + ":bootstrap", raw["bootstrap_token_id"], 512)
    evidence.equal(label + ":handoff-submission", handoff["submitted_at_ps"], raw["prefill_completed_at_ps"])
    expected = {"prefill_queue_ps": raw["prefill_eligible_at_ps"] - raw["admitted_at_ps"],
                "prefill_service_ps": raw["prefill_completed_at_ps"] - raw["prefill_eligible_at_ps"],
                "handoff_ps": handoff["completed_at_ps"] - handoff["submitted_at_ps"],
                "decode_admission_wait_ps": raw["decode_eligible_at_ps"] - handoff["completed_at_ps"],
                "decode_first_token_service_ps": tokens[0] - raw["decode_eligible_at_ps"]}
    expected["total_ps"] = sum(expected.values())
    evidence.equal(label + ":decomposition", raw["decomposition"], expected)
    evidence.equal(label + ":ttft", raw["ttft_ps"], tokens[0] - raw["admitted_at_ps"])
    evidence.equal(label + ":critical-path-sum", raw["ttft_ps"], expected["total_ps"])
    tpot = Fraction(tokens[-1] - tokens[0], 3)
    evidence.equal(label + ":tpot", raw["tpot_ps"], {"numerator": tpot.numerator, "denominator": tpot.denominator})
    total = spec["prompt_tokens"] * 2 * frozen["geometry"]["num_layers"] * frozen["geometry"]["num_kv_heads"] * frozen["geometry"]["head_size"] * frozen["geometry"]["element_bytes"]
    evidence.equal(label + ":cache-bytes", handoff["kv_bytes"], total)
    params = {"schema": "simllm-pd-kv-params-v1", "do_remote_prefill": True, "do_remote_decode": False,
              "remote_engine_id": raw["prefill_engine_id"], "remote_request_id": raw["request_id"],
              "session_request_id": raw["request_id"], "remote_num_tokens": spec["prompt_tokens"],
              "bootstrap_token_id": 512, "worker_tensor_transfer": False,
              "timing_authority": frozen["authority"]["legacy"]}
    if spec["kind"] == "compatibility":
        finish = raw["prefill_completed_at_ps"] + spec["handoff_ps"]
        evidence.equal(label + ":declared-handoff", handoff, {
            "authority": frozen["authority"]["legacy"], "pricing_arm": "declared-constant" if spec["handoff_ps"] else "off",
            "kv_bytes": total, "submitted_at_ps": raw["prefill_completed_at_ps"],
            "eligible_at_ps": raw["prefill_completed_at_ps"], "started_at_ps": raw["prefill_completed_at_ps"],
            "finished_at_ps": finish, "completed_at_ps": finish})
    else:
        params.update(timing_authority=frozen["authority"]["shared"], pricing_arm="shared-packet",
                      handoff_completed_at_ps=handoff["completed_at_ps"])
    evidence.equal(label + ":connector-params", raw["kv_transfer_params"], params)


def collect_steps(data, owners, evidence):
    steps, positive = {}, 0
    for ordinal, step in enumerate(data["steps"]):
        record, result = step["record"], step["result"]
        key = (step["engine_id"], record["step_index"])
        label = data["process_id"] + ":step:" + str(ordinal)
        evidence.fields(label + ":fields", step, "engine_id record result")
        evidence.check(label + ":unique", key not in steps)
        steps[key] = step
        evidence.equal(label + ":record-roundtrip", step_record_to_json(step_record_from_json(record)), record)
        evidence.fields(label + ":result-fields", result, "step_index step_latency_ps completed_at_ps request_metrics additive_visit_totals")
        evidence.equal(label + ":request-reducer-off", result["request_metrics"], [])
        evidence.equal(label + ":visit-reducer-off", result["additive_visit_totals"], None)
        StepResult(**{**result, "request_metrics": ()})
        evidence.check(label + ":time", integer(record["virtual_time_ps"])
                       and result["step_index"] == record["step_index"]
                       and result["completed_at_ps"] == record["virtual_time_ps"] + result["step_latency_ps"])
        if record["scheduled"]:
            evidence.equal(label + ":one-native-sequence", len(record["scheduled"]), 1)
            evidence.check(label + ":owner", (key[0], record["scheduled"][0]["request_id"]) in owners)
            positive += 1
        else:
            evidence.check(label + ":genuine-drain", bool(record["finished_request_ids"] or record["preempted_request_ids"])
                           and result["step_latency_ps"] == 0)
    evidence.equal(data["process_id"] + ":positive-step-count", positive, len(request_rows(data)) * 5)
    return steps


def native_schedule(cell, spec, frozen, flows):
    """Use only admitted native flow completions to release each decode queue."""
    service = frozen["known_services_ps"][str(spec["prompt_tokens"])]
    admission, count = cell["admission_ps"], spec["requests_per_batch"]
    joins = [max(flow["completion_time_ps"] for flow in flows if flow["source"] // 8 == index)
             for index in range(count)]
    ready, starts = [0] * spec["decode_engines"], [None] * count
    for index in sorted(range(count), key=lambda i: (joins[i], i)):
        engine = index % spec["decode_engines"]
        starts[index] = max(joins[index], ready[engine])
        ready[engine] = starts[index] + sum(service["decode"])
    rows = []
    for start in starts:
        tokens = [start + sum(service["decode"][:index]) for index in range(1, 5)]
        rows.append({"admitted_at_ps": admission, "prefill_eligible_at_ps": admission,
                     "prefill_completed_at_ps": admission + service["prefill"], "decode_eligible_at_ps": start,
                     "decode_token_completed_at_ps": tokens, "ttft_ps": tokens[0] - admission,
                     "tpot_ps": {"numerator": service["decode"][0], "denominator": 1}})
    return rows


def check_cells(data, spec, frozen, steps, owners, evidence):
    label = spec["id"] + ":cells"
    evidence.equal(label + ":count", len(data["cells"]), len(spec["admission_times_ps"]))
    evidence.equal(label + ":no-controls", data["controls"], [])
    service = frozen["known_services_ps"][str(spec["prompt_tokens"])]
    for ordinal, (cell, admission) in enumerate(zip(data["cells"], spec["admission_times_ps"], strict=True)):
        name = label + ":" + str(ordinal)
        prefix = (spec["id"].split("-", 1)[1] if spec["kind"] == "compatibility" else spec["id"]) + f":batch-{ordinal}"
        evidence.fields(name + ":fields", cell, "id start_ps admission_ps end_ps observation_start observation_stop inputs requests comparisons record_starts record_stops "
                        "clock_advance_indices event_indices visit_indices prefill_batches decode_batches network_after")
        evidence.equal(name + ":id", cell["id"], prefix)
        evidence.equal(name + ":admission", cell["admission_ps"], admission)
        evidence.check(name + ":arrival-after-prior-batch", admission >= cell["start_ps"])
        evidence.equal(name + ":inputs", cell["inputs"], [
            {"request_id": prefix + f":request-{index}", "prompt_token_ids": list(baseline._prompt_tokens()[:spec["prompt_tokens"]]),
             "decode_output_tokens": 4, "admitted_at_ps": admission} for index in range(spec["requests_per_batch"])])
        evidence.equal(name + ":requests", [row["result"]["request_id"] for row in cell["requests"]],
                       [row["request_id"] for row in cell["inputs"]])
        evidence.equal(name + ":comparison-count", len(cell["comparisons"]), len(cell["requests"]))
        if spec["kind"] == "compatibility":
            reference = native_reference(spec, {"arrival_offsets_ps": [0, 0], "prompt_tokens": [spec["prompt_tokens"]] * 2,
                "handoff_ps": spec["handoff_ps"], "decode_output_tokens": 4}, frozen["known_services_ps"])
            timeline = [expected_times(row, admission) for row in reference["requests"]]
            kind = "guards"
            floor = service["prefill"] + spec["handoff_ps"] + sum(service["decode"])
            ceiling = 2 * (service["prefill"] + spec["handoff_ps"] + sum(service["decode"]))
        else:
            flows = [row for row in data["network_after"]["rows"] if ordinal * 16 < row["sequence"] <= (ordinal + 1) * 16]
            timeline, kind = native_schedule(cell, spec, frozen, flows), "oracles"
            _, _, n, q = geometry(spec, frozen)
            fixed = frozen["backend"]["pcie_submission_ps"] + frozen["backend"]["propagation_ps"]
            floor = service["prefill"] + fixed + (n + 1) * q + sum(service["decode"])
            ceiling = service["prefill"] + fixed + (16 * n + 1) * q + 2 * sum(service["decode"])
        evidence.check(name + ":physical-bounds", floor <= cell["end_ps"] - admission <= ceiling)
        evidence.equal(name + ":full-causal-schedule", [request_times(row["result"]) for row in cell["requests"]], timeline, kind=kind)
        evidence.equal(name + ":last-completion", cell["end_ps"], max(row["decode_token_completed_at_ps"][-1] for row in timeline))
        for index, (row, compared, expected) in enumerate(zip(cell["requests"], cell["comparisons"], timeline, strict=True)):
            tag, raw = name + ":request:" + str(index), row["result"]
            evidence.fields(tag + ":row-fields", row, "result prefill_record_indices decode_record_indices")
            evidence.fields(tag + ":comparison-fields", compared, "schema excluded_root_fields request")
            evidence.equal(tag + ":comparison-schema", compared["schema"], "simllm-vllm-pd-request-comparison-v1")
            request_metrics(raw, spec, frozen, tag, evidence)
            for role in ("prefill", "decode"):
                engine = f"simllm-{role}-{index % spec[role + '_engines']}"
                evidence.equal(tag + ":route:" + role, raw[role + "_engine_id"], engine)
                indices = row[role + "_record_indices"]
                evidence.equal(tag + ":slice-count:" + role, raw[role + "_step_count"], len(indices))
                owned = [step for (owner, _), step in steps.items() if owner == engine and any(
                    item["request_id"] == raw[role + "_internal_request_id"] for item in step["record"]["scheduled"])]
                starts = [expected["prefill_eligible_at_ps"]] if role == "prefill" else [expected["decode_eligible_at_ps"], *expected["decode_token_completed_at_ps"][:-1]]
                ends = [expected["prefill_completed_at_ps"]] if role == "prefill" else expected["decode_token_completed_at_ps"]
                evidence.equal(tag + ":native-step-vector:" + role,
                    [[step["record"]["virtual_time_ps"], step["result"]["completed_at_ps"], step["result"]["step_latency_ps"]] for step in owned],
                    [[start, end, end - start] for start, end in zip(starts, ends, strict=True)])
    evidence.equal(label + ":final-clock", data["final_clock_ps"], data["cells"][-1]["end_ps"])
    check_native_memberships(data, spec, steps, owners, evidence)


def check_native_state(data, spec, steps, owners, evidence):
    """Keep raw shared metadata and project only the producer's earlier view."""
    projected = deepcopy(data)
    if spec["kind"] != "compatibility":
        params = {row["result"]["request_id"]: row["result"]["kv_transfer_params"] for row in request_rows(data)}
        seen = set()
        for index, checkpoint in enumerate(data["checkpoints"]):
            engine = checkpoint["engine_id"]
            if engine.startswith("simllm-decode"):
                for alias, request in checkpoint["native"]["requests"].items():
                    public = owners[(engine, alias)]
                    evidence.equal(spec["id"] + f":consumer-join-params:{index}:{alias}", request["kv_transfer_params"], params[public])
                    seen.add(public)
        evidence.equal(spec["id"] + ":all-consumer-join-params", sorted(seen), sorted(params))
        for row in request_rows(projected):
            params = row["result"]["kv_transfer_params"]
            del params["pricing_arm"], params["handoff_completed_at_ps"]
            params["timing_authority"] = "simllm-declared-kv-handoff-v1"
    check_checkpoints(projected, spec, steps, owners, evidence)


def admit(data, spec, frozen, dependency, deadline, evidence, wire):
    label = spec["id"]
    spec = {**spec, "mode": spec.get("engine_timing", "independent")}
    evidence.fields(label + ":fields", data, "schema process_id mode authority sources_before sources_after identity "
        "retained_before retained_after selected_before selected_after native_configurations construction controls cells steps "
        "clock_advances checkpoints serialized_observations serialized_bindings engine_evidence runtime_evidence observation_order projections "
        "sinks final_clock_ps runtime_failure network_before network_before_close network_after network_observations "
        "clock_after_close_ps clock_owner_released")
    evidence.equal(label + ":schema", data["schema"], "simllm-shared-kv-handoff-native-v1")
    evidence.equal(label + ":process", data["process_id"], label)
    evidence.equal(label + ":mode", data["mode"], spec["mode"])
    evidence.equal(label + ":authority", data["authority"], dependency["authority"]["enabled" if spec["mode"] == "independent" else "disabled"])
    evidence.equal(label + ":runtime-failure", data["runtime_failure"], None)
    evidence.equal(label + ":no-teardown-time", data["clock_after_close_ps"], data["final_clock_ps"])
    evidence.equal(label + ":clock-released", data["clock_owner_released"], True)
    check_engines(data, spec, dependency, evidence)
    constructed = {row["native_configuration"]["engine_id"]: row["native_configuration"] for row in data["construction"]}
    native_ids = [constructed[row["engine_id"]]["llm_engine_object_id"] for row in data["retained_before"]]
    evidence.check(label + ":distinct-native-frontends", len(set(native_ids)) == len(native_ids)
                   and all(integer(value, 1) and value not in (row["frontend_object_id"], row["core_object_id"])
                           for value, row in zip(native_ids, data["retained_before"], strict=True)))
    configs = [{"engine_id": row["engine_id"], "dtype": "torch.bfloat16", "max_model_len": 64,
                "num_gpu_blocks_override": 64, "llm_engine_object_id": value}
               for row, value in zip(data["retained_before"], native_ids, strict=True)]
    evidence.equal(label + ":native-configs", data["native_configurations"], configs)
    evidence.equal(label + ":constructed-configs", [row["native_configuration"] for row in data["construction"]], configs)
    bindings = admit_bindings(data, evidence)
    owners = {key: value[1] for key, value in bindings.items()}
    steps = collect_steps(data, owners, evidence)
    durations = check_network(data, spec, frozen, deadline, evidence)
    check_cells(data, spec, frozen, steps, owners, evidence)
    check_projections(data, spec, steps, evidence)
    check_sinks_and_domains(data, spec, steps, evidence)
    check_native_state(data, spec, steps, owners, evidence)
    check_serialized(data, steps, owners, evidence)
    check_journals(data, evidence)
    check_order(data, spec, wire, evidence)
    projected = comparison(data, bindings, evidence)
    return {"durations": durations, "comparison": projected, "steps": steps, "owners": owners}


def check_relations(campaign, frozen, evidence):
    fixed = frozen["backend"]["pcie_submission_ps"] + frozen["backend"]["propagation_ps"]

    def cell(decode, prompt, rate):
        return campaign[f"shared-p2-d{decode}-t{prompt}-r{rate}"]

    def durations(value):
        return [raw["result"]["handoff"]["completed_at_ps"] - raw["result"]["prefill_completed_at_ps"]
                for raw in request_rows(value)]

    for decode in (1, 2):
        for prompt in (8, 16):
            a, b = durations(cell(decode, prompt, 200)), durations(cell(decode, prompt, 400))
            evidence.check(f"rate:d{decode}:t{prompt}", all(x > y and abs((x - fixed) - 2 * (y - fixed)) <= 2 * 166400
                           for x, y in zip(a, b, strict=True)), kind="relations", family="rate")
        for rate in (200, 400):
            a, b = durations(cell(decode, 16, rate)), durations(cell(decode, 8, rate))
            q = 166400 if rate == 200 else 83200
            evidence.check(f"size:d{decode}:r{rate}", all(x > y and abs((x - fixed) - 2 * (y - fixed)) <= 2 * q
                           for x, y in zip(a, b, strict=True)), kind="relations", family="size")
    for prompt in (8, 16):
        for rate in (200, 400):
            a, b = (request_rows(cell(decode, prompt, rate)) for decode in (1, 2))
            delta = min(row["result"]["ttft_ps"] for row in a) - min(row["result"]["ttft_ps"] for row in b)
            join_delta = min(row["result"]["handoff"]["completed_at_ps"] for row in a) - min(row["result"]["handoff"]["completed_at_ps"] for row in b)
            n, q = prompt * 3 // 2, 166400 if rate == 200 else 83200
            evidence.check(f"sharing:t{prompt}:r{rate}", 0 < (n - 1) * q <= delta <= n * q and delta == join_delta,
                           kind="relations", family="sharing")
    evidence.equal("relations:families", dict(Counter(row["family"] for row in evidence.relations)), frozen["behavioral_families"])
