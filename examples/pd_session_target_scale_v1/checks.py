"""Reconcile actual retained objects, unique service and request visibility."""

from __future__ import annotations

import math
from collections import Counter
from fractions import Fraction

from examples.pd_session_identity_v1.run_study import exact_json_bytes, sha


class GuardFailure(ValueError):
    """A failed precondition voids this campaign instead of losing a point."""


class Evidence:
    def __init__(self, stages):
        self.expected_stages = set(stages)
        self.finished_stages = set()
        self.names = set()
        self.guards, self.oracles, self.relations = [], [], []

    def check(self, name, condition, *, kind="guards", family=None):
        if name in self.names or kind not in ("guards", "oracles", "relations"):
            raise GuardFailure("duplicate check or invalid evidence class: " + name)
        self.names.add(name)
        row = {"name": name, "passed": condition is True}
        if family is not None:
            row["family"] = family
        getattr(self, kind).append(row)
        if condition is not True:
            raise GuardFailure(name)

    def equal(self, name, actual, expected, **kwargs):
        self.check(name, exact_json_bytes(actual) == exact_json_bytes(expected), **kwargs)

    def fields(self, name, value, fields):
        self.check(name, type(value) is dict and set(value) == set(fields.split()))

    def finish(self, stage):
        self.check("stage:" + stage,
                   stage in self.expected_stages and stage not in self.finished_stages)
        self.finished_stages.add(stage)


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def finite_positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def engine_names(scale):
    return {role: [f"simllm-{role}-{ordinal}" for ordinal in range(scale[role + "_engines"])]
            for role in ("prefill", "decode")}


def check_retention(data, scale, frozen, evidence):
    prefix = scale["id"]
    names = engine_names(scale)
    ordered = names["prefill"] + names["decode"]
    before, after = data["retention_before"], data["retention_after"]
    evidence.equal(prefix + ":retention-order", [row["engine_id"] for row in before], ordered)
    evidence.equal(prefix + ":retention-identity", after, before)
    identity = data["native_identity"]
    evidence.check(prefix + ":weak-observer", identity["observer_weak_references_only"] is True
                   and identity["all_observed_objects_alive"] is True)
    for key in ("engine_object_id", "frontend_object_id", "core_object_id",
                "executor_object_id", "sink_object_id", "config_object_id"):
        values = [row[key] for row in before]
        evidence.check(prefix + ":distinct:" + key,
                       len(set(values)) == scale["engines"]
                       and all(integer(value, 1) for value in values))
    workers = []
    for row in before:
        label = prefix + ":" + row["engine_id"]
        evidence.fields(label + ":fields", row,
                        "engine_id role ordinal engine_object_id frontend_object_id core_object_id "
                        "executor_object_id frontend_executor_object_id sink_object_id clock_object_id "
                        "runtime_clock_object_id config_object_id frontend_config_object_id world_size "
                        "executor_role connector_role workers executor_workers local_placement")
        role = row["role"]
        evidence.check(label + ":role", role in names and row["engine_id"] in names[role])
        evidence.equal(label + ":ordinal", row["ordinal"], names[role].index(row["engine_id"]))
        evidence.equal(label + ":world", row["world_size"], 8)
        evidence.equal(label + ":executor-role", row["executor_role"], role)
        evidence.equal(label + ":connector", row["connector_role"],
                       "kv_producer" if role == "prefill" else "kv_consumer")
        evidence.equal(label + ":frontend-executor", row["frontend_executor_object_id"],
                       row["executor_object_id"])
        evidence.equal(label + ":frontend-config", row["frontend_config_object_id"],
                       row["config_object_id"])
        evidence.equal(label + ":shared-clock", [row["clock_object_id"], row["runtime_clock_object_id"]],
                       [identity["clock_object_id"]] * 2)
        evidence.equal(label + ":rpc-workers", row["workers"], row["executor_workers"])
        evidence.equal(label + ":local-ranks", [worker["local_rank"] for worker in row["workers"]],
                       list(range(8)))
        for index, worker in enumerate(row["workers"]):
            evidence.fields(label + ":worker-fields:" + str(index), worker,
                            "object_id rank local_rank config_id device_absent model_runner_absent")
        for index, rank in enumerate(row["local_placement"]):
            evidence.fields(label + ":placement-fields:" + str(index), rank,
                            "rank local_rank role gpu_uuid tp_ranks")
        evidence.check(label + ":worker-contract", all(
            integer(worker["object_id"], 1) and worker["rank"] == worker["local_rank"]
            and type(worker["rank"]) is int and integer(worker["config_id"], 1)
            and worker["config_id"] == row["config_object_id"]
            and worker["device_absent"] is True and worker["model_runner_absent"] is True
            for worker in row["workers"]))
        evidence.check(label + ":local-placement", len(row["local_placement"]) == 8 and all(
            integer(rank["rank"]) and integer(rank["local_rank"])
            and rank["rank"] == index and rank["local_rank"] == index and rank["role"] == role
            and rank["gpu_uuid"] == f"sim-{role}-gpu-{index}"
            and all(integer(value) for value in rank["tp_ranks"])
            and rank["tp_ranks"] == list(range(8))
            for index, rank in enumerate(row["local_placement"])))
        workers.extend(worker["object_id"] for worker in row["workers"])
    evidence.check(prefix + ":distinct-workers", len(workers) == len(set(workers)) == scale["workers"])
    rows = data["construction_rows"]
    evidence.equal(prefix + ":construction-order", [row["identity"] for row in rows], before)
    previous_ns = 0
    for index, row in enumerate(rows):
        label = f"{prefix}:construction:{index}"
        evidence.fields(label + ":fields", row,
                        "index identity native_constructor_seconds elapsed_ns current_rss_kib "
                        "peak_rss_kib clock_ps all_observed_objects_alive")
        evidence.equal(label + ":index", row["index"], index)
        evidence.equal(label + ":clock", row["clock_ps"], 0)
        evidence.check(label + ":alive", row["all_observed_objects_alive"] is True)
        evidence.check(label + ":host", integer(row["current_rss_kib"], 1)
                       and row["current_rss_kib"] < frozen["construction"]["stop_rss_kib"]
                       and integer(row["peak_rss_kib"], row["current_rss_kib"])
                       and integer(row["elapsed_ns"], previous_ns + 1)
                       and finite_positive(row["native_constructor_seconds"]))
        previous_ns = row["elapsed_ns"]
    evidence.finish(prefix + ":construction")
    projection = data["worker_manifest_projection"]
    evidence.equal(prefix + ":global-rank-domain", [row["global_rank"] for row in projection],
                   list(range(scale["workers"])))
    evidence.equal(prefix + ":projected-worker-domain", [row["worker_object_id"] for row in projection],
                   workers)
    for row in projection:
        rank = row["global_rank"]
        evidence.fields(f"{prefix}:projection-fields:{rank}", row,
                        "engine_id role ordinal worker_object_id local_rank global_rank manifest_role "
                        "manifest_local_rank hostname gpu_id gpu_uuid gpu_node_id nic_id nic_node_id "
                        "nic_affine_rank tp_ranks")
        global_node, local = divmod(rank, 8)
        role = "prefill" if global_node < scale["prefill_engines"] else "decode"
        ordinal = global_node if role == "prefill" else global_node - scale["prefill_engines"]
        evidence.check(f"{prefix}:projection:{rank}",
                       row["role"] == row["manifest_role"] == role
                       and row["ordinal"] == ordinal
                       and row["engine_id"] == f"simllm-{role}-{ordinal}"
                       and row["local_rank"] == row["manifest_local_rank"] == local
                       and row["gpu_id"] == row["gpu_uuid"] == f"sim-gpu-{rank:04d}"
                       and row["nic_id"] == f"sim-nic-{rank:04d}"
                       and row["hostname"] == row["gpu_node_id"] == row["nic_node_id"]
                       == f"{role}-node-{ordinal}"
                       and row["nic_affine_rank"] == rank
                       and all(integer(row[key]) for key in ("ordinal", "local_rank", "manifest_local_rank", "nic_affine_rank"))
                       and row["tp_ranks"] == list(range(8 * global_node, 8 * (global_node + 1))))
        evidence.check(f"{prefix}:projection-tp-types:{rank}", all(integer(value) for value in row["tp_ranks"]))
    evidence.finish(prefix + ":retention")


def request_metrics(raw, label, evidence):
    keys = {"schema", "request_id", "admitted_at_ps", "prefill_eligible_at_ps",
            "prefill_completed_at_ps", "handoff", "decode_eligible_at_ps",
            "decode_token_completed_at_ps", "ttft_ps", "tpot_ps", "decomposition",
            "prefill_engine_id", "decode_engine_id", "prefill_internal_request_id",
            "decode_internal_request_id", "bootstrap_token_id", "decode_token_ids",
            "kv_transfer_params", "prefill_step_count", "decode_step_count"}
    evidence.equal(label + ":result-fields", sorted(raw), sorted(keys))
    handoff, decomposition = raw["handoff"], raw["decomposition"]
    evidence.equal(label + ":schema", raw["schema"], "simllm-pd-session-result-v1")
    evidence.equal(label + ":declared-handoff", handoff, {
        "authority": "simllm-declared-kv-handoff-v1", "pricing_arm": "declared-constant",
        "kv_bytes": handoff["kv_bytes"],
        "submitted_at_ps": raw["prefill_completed_at_ps"],
        "eligible_at_ps": raw["prefill_completed_at_ps"],
        "started_at_ps": raw["prefill_completed_at_ps"],
        "finished_at_ps": handoff["completed_at_ps"], "completed_at_ps": handoff["completed_at_ps"],
    })
    times = [raw["admitted_at_ps"], raw["prefill_eligible_at_ps"], raw["prefill_completed_at_ps"],
             *[handoff[key] for key in ("submitted_at_ps", "eligible_at_ps", "started_at_ps",
                                       "finished_at_ps", "completed_at_ps")],
             raw["decode_eligible_at_ps"], *raw["decode_token_completed_at_ps"]]
    evidence.check(label + ":causal-time", all(integer(value) for value in times)
                   and times == sorted(times) and len(raw["decode_token_completed_at_ps"]) == 4)
    evidence.equal(label + ":tokens", raw["decode_token_ids"], [512] * 4)
    evidence.equal(label + ":bootstrap", raw["bootstrap_token_id"], 512)
    evidence.equal(label + ":handoff-release", handoff["submitted_at_ps"], raw["prefill_completed_at_ps"])
    expected = {
        "prefill_queue_ps": raw["prefill_eligible_at_ps"] - raw["admitted_at_ps"],
        "prefill_service_ps": raw["prefill_completed_at_ps"] - raw["prefill_eligible_at_ps"],
        "handoff_ps": handoff["completed_at_ps"] - handoff["submitted_at_ps"],
        "decode_admission_wait_ps": raw["decode_eligible_at_ps"] - handoff["completed_at_ps"],
        "decode_first_token_service_ps": raw["decode_token_completed_at_ps"][0] - raw["decode_eligible_at_ps"],
    }
    expected["total_ps"] = sum(expected.values())
    evidence.equal(label + ":decomposition", decomposition, expected)
    evidence.equal(label + ":ttft", raw["ttft_ps"], raw["decode_token_completed_at_ps"][0] - raw["admitted_at_ps"])
    evidence.equal(label + ":causal-sum", raw["ttft_ps"], decomposition["total_ps"])
    tpot = Fraction(raw["decode_token_completed_at_ps"][-1] - raw["decode_token_completed_at_ps"][0], 3)
    evidence.equal(label + ":tpot", raw["tpot_ps"],
                   {"numerator": tpot.numerator, "denominator": tpot.denominator})
    return {"ttft_ps": raw["ttft_ps"], "tpot_ps": tpot,
            "jct_ps": raw["decode_token_completed_at_ps"][-1] - raw["admitted_at_ps"],
            "prefill_service_ps": decomposition["prefill_service_ps"],
            "decode_first_token_service_ps": decomposition["decode_first_token_service_ps"],
            "kv_bytes": handoff["kv_bytes"]}


def check_cells(data, scale, frozen, evidence):
    prefix = scale["id"]
    evidence.equal(prefix + ":serial-domain", [row["id"] for row in data["serial_cells"]],
                   [row["id"] for row in frozen["serial_cells"]])
    names, requests, metrics = engine_names(scale), {}, {}
    cells = [*data["serial_cells"], data["burst_cell"]]
    for cell, spec in zip(cells, [*frozen["serial_cells"], frozen["burst"]], strict=True):
        label = prefix + ":" + spec["id"]
        burst = spec is frozen["burst"]
        evidence.equal(label + ":id", cell["id"], spec["id"])
        evidence.check(label + ":components", set(frozen["evidence_domains"]["required_cell_components"])
                       | {"wall_time_ns", "arrival_mode", "offered_rate_requests_per_second"} == set(cell))
        evidence.equal(label + ":request-domain",
                       [row["result"]["request_id"] for row in cell["requests"]],
                       [f"{spec['id']}:request-{index}" for index in range(80)])
        evidence.check(label + ":wall", integer(cell["wall_time_ns"], 1)
                       and finite_positive(cell["wall_seconds"])
                       and cell["wall_seconds"] == cell["wall_time_ns"] / 1e9)
        evidence.equal(label + ":arrival-mode", cell["arrival_mode"],
                       "simultaneous-burst" if burst else "serial-completion-admission")
        evidence.equal(label + ":no-invented-rate", cell["offered_rate_requests_per_second"], None)
        cell_metrics = []
        previous = cell["start_ps"]
        for index, row in enumerate(cell["requests"]):
            raw = row["result"]
            request_id = raw["request_id"]
            key = label + ":" + str(index)
            evidence.fields(key + ":row-fields", row, "result prefill_record_indices decode_record_indices")
            value = request_metrics(raw, key, evidence)
            evidence.equal(key + ":admission", raw["admitted_at_ps"], cell["start_ps"] if burst else previous)
            previous = raw["decode_token_completed_at_ps"][-1]
            for role in names:
                evidence.equal(key + ":route:" + role, raw[role + "_engine_id"],
                               names[role][index % len(names[role])])
            evidence.equal(key + ":handoff", raw["handoff"]["completed_at_ps"] - raw["handoff"]["submitted_at_ps"],
                           spec["handoff_ps"])
            evidence.equal(key + ":bytes", value["kv_bytes"],
                           spec["prompt_tokens"] * frozen["physical_bounds"]["kv_bytes_per_prompt_token"])
            params = raw["kv_transfer_params"]
            evidence.equal(key + ":connector", params, {
                "schema": "simllm-pd-kv-params-v1", "do_remote_prefill": True, "do_remote_decode": False,
                "remote_engine_id": raw["prefill_engine_id"], "remote_request_id": request_id,
                "session_request_id": request_id, "remote_num_tokens": spec["prompt_tokens"],
                "bootstrap_token_id": raw["bootstrap_token_id"], "worker_tensor_transfer": False,
                "timing_authority": "simllm-declared-kv-handoff-v1",
            })
            requests[request_id] = row
            cell_metrics.append(value)
        evidence.equal(label + ":terminal-clock", cell["end_ps"],
                       max(row["result"]["decode_token_completed_at_ps"][-1] for row in cell["requests"]))
        for role in names:
            expected_counts = {name: 80 // len(names[role]) for name in names[role]}
            evidence.equal(label + ":balanced:" + role, cell["engine_request_counts"][role], expected_counts)
            evidence.equal(label + ":actual-routing:" + role,
                           dict(Counter(row["result"][role + "_engine_id"] for row in cell["requests"])),
                           expected_counts)
        evidence.fields(label + ":role-domain", cell["engine_request_counts"], "prefill decode")
        metrics[spec["id"]] = cell_metrics
        if not burst:
            accepted = next(row for row in frozen["baseline_control"]["accepted_cells"]
                            if row["prompt_tokens"] == spec["prompt_tokens"] and row["handoff_ps"] == spec["handoff_ps"])
            evidence.check(label + ":historical-cell", all(
                all(value[key] == accepted[key] for key in ("ttft_ps", "tpot_ps", "prefill_service_ps",
                                                            "decode_first_token_service_ps", "kv_bytes"))
                for value in cell_metrics), kind="oracles")
    for row in data["baseline_controls"]:
        request_id = row["result"]["request_id"]
        evidence.check(prefix + ":baseline-unique:" + request_id, request_id not in requests)
        requests[request_id] = row
    evidence.finish(prefix + ":serial")
    evidence.finish(prefix + ":burst")
    return requests, metrics


def interval_union(intervals):
    ordered = sorted(intervals)
    if not ordered:
        return 0
    total, start, stop = 0, *ordered[0]
    for lower, upper in ordered[1:]:
        if lower > stop:
            total += stop - start
            start, stop = lower, upper
        else:
            stop = max(stop, upper)
    return total + stop - start


def check_accounting(data, scale, frozen, requests, evidence):
    prefix = scale["id"]
    by_internal = {}
    for request_id, row in requests.items():
        raw = row["result"]
        for role in ("prefill", "decode"):
            identifier = raw[role + "_internal_request_id"]
            evidence.check(f"{prefix}:internal-id:{request_id}:{role}",
                           type(identifier) is str and bool(identifier.strip()) and identifier not in by_internal)
            by_internal[identifier] = (request_id, role, raw[role + "_engine_id"])
    steps, memberships, intervals = {}, {}, []
    names = engine_names(scale)
    declared_engines = names["prefill"] + names["decode"]
    identities = [(row["engine_id"], row["record"]["step_index"]) for row in data["unique_steps"]]
    evidence.check(prefix + ":unique-step-domain", len(identities) == len(set(identities)))
    for row in data["unique_steps"]:
        record, result, engine = row["record"], row["result"], row["engine_id"]
        index = record["step_index"]
        key = (engine, index)
        label = f"{prefix}:step:{engine}:{index}"
        evidence.fields(label + ":fields", row, "engine_id record result")
        evidence.fields(label + ":record-fields", record,
                        "schema step_index virtual_time_ps scheduled preempted_request_ids finished_request_ids num_sampled")
        evidence.equal(label + ":record-schema", record["schema"], "atlahs-closed-loop-step-v1")
        evidence.fields(label + ":result-fields", result,
                        "step_index step_latency_ps completed_at_ps request_metrics additive_visit_totals")
        evidence.equal(label + ":inactive-request-metrics", result["request_metrics"], [])
        evidence.equal(label + ":inactive-visit-totals", result["additive_visit_totals"], None)
        evidence.check(label + ":engine-domain", engine in declared_engines)
        evidence.check(label + ":unique", key not in steps and integer(index))
        evidence.check(label + ":result", integer(result["step_index"]) and result["step_index"] == index
                       and integer(record["virtual_time_ps"]) and integer(result["step_latency_ps"])
                       and integer(result["completed_at_ps"])
                       and result["completed_at_ps"] == record["virtual_time_ps"] + result["step_latency_ps"])
        steps[key] = row
        scheduled = record["scheduled"]
        evidence.check(label + ":scheduled-unique", len({item["request_id"] for item in scheduled}) == len(scheduled))
        evidence.equal(label + ":sampled", record["num_sampled"], len(scheduled))
        evidence.equal(label + ":no-preemption", record["preempted_request_ids"], [])
        finished = record["finished_request_ids"]
        evidence.check(label + ":finished-domain", type(finished) is list
                       and len(finished) == len(set(finished)) and all(
                           identifier in by_internal and by_internal[identifier][2] == engine for identifier in finished))
        for item in scheduled:
            identifier = item["request_id"]
            evidence.check(label + ":member:" + identifier,
                           identifier in by_internal and by_internal[identifier][2] == engine)
            memberships.setdefault(identifier, []).append(key)
            request_id, role, _ = by_internal[identifier]
            prompt = requests[request_id]["result"]["kv_transfer_params"]["remote_num_tokens"]
            visit = len(memberships[identifier]) - 1
            evidence.equal(label + ":token-contract:" + identifier, item, {
                "request_id": identifier, "phase": "prefill" if role == "prefill" or visit == 0 else "decode",
                "num_new_tokens": prompt if role == "prefill" else 1,
                "num_cached_tokens": prompt if role == "decode" and visit == 0 else 0,
                "context_length": prompt if role == "prefill" else prompt + visit + 1,
            })
        if scheduled:
            bounds = frozen["physical_bounds"]["nonempty_step_service_ps"]
            evidence.check(label + ":service-bound", bounds["floor"] <= result["step_latency_ps"] <= bounds["ceiling"])
            intervals.append((record["virtual_time_ps"], result["completed_at_ps"]))
        else:
            evidence.equal(label + ":empty-service", result["step_latency_ps"], 0)
            evidence.check(label + ":justified-drain", bool(finished))
    for engine in engine_names(scale).values():
        for name in engine:
            indices = [index for candidate, index in steps if candidate == name]
            evidence.check(prefix + ":step-sequence:" + name,
                           bool(indices) and indices == list(range(len(indices))))
            previous_finished = []
            for index in indices:
                record = steps[(name, index)]["record"]
                evidence.equal(f"{prefix}:finished-lag:{name}:{index}", record["finished_request_ids"], previous_finished)
                previous_finished = sorted(item["request_id"] for item in record["scheduled"]
                                           if memberships[item["request_id"]][-1] == (name, index))
    for identifier, (request_id, role, engine) in by_internal.items():
        raw = requests[request_id]["result"]
        keys = memberships.get(identifier, [])
        label = f"{prefix}:membership:{request_id}:{role}"
        evidence.equal(label + ":count", len(keys), 1 if role == "prefill" else 4)
        indices = requests[request_id][role + "_record_indices"]
        evidence.check(label + ":slice-domain", bool(indices) and all(integer(index) for index in indices)
                       and indices == list(range(indices[0], indices[-1] + 1))
                       and all((engine, index) in steps for index in indices)
                       and all(index in indices for _, index in keys))
        evidence.equal(label + ":slice-count", raw[role + "_step_count"], len(indices))
        starts = [steps[key]["record"]["virtual_time_ps"] for key in keys]
        completions = [steps[key]["result"]["completed_at_ps"] for key in keys]
        evidence.equal(label + ":release", starts[0], raw[role + "_eligible_at_ps"])
        evidence.equal(label + ":visibility", completions,
                       [raw["prefill_completed_at_ps"]] if role == "prefill" else raw["decode_token_completed_at_ps"])
        latest = completions[-1]
        earliest = raw["admitted_at_ps"] if role == "prefill" else raw["handoff"]["completed_at_ps"]
        evidence.check(label + ":slice-bounds", all(
            earliest <= steps[(engine, index)]["record"]["virtual_time_ps"]
            <= steps[(engine, index)]["result"]["completed_at_ps"] <= latest for index in indices))
        if role == "decode":
            floor = frozen["physical_bounds"]["compatibility_resident_streaming_floor_ps"]
            evidence.check(label + ":conditional-memory-floor", all(steps[key]["result"]["step_latency_ps"] >= floor
                                                                      for key in keys))
    for cell in [*data["serial_cells"], data["burst_cell"], {"id": "baseline", "requests": data["baseline_controls"]}]:
        burst = cell is data["burst_cell"]
        previous_stop = {}
        for row in cell["requests"]:
            raw = row["result"]
            for role in ("prefill", "decode"):
                engine = raw[role + "_engine_id"]
                own = memberships[raw[role + "_internal_request_id"]]
                if cell["id"] == "baseline":
                    first = previous_stop.get(engine, 0)
                else:
                    cell_indices = [index for candidate, index in cell["step_identities"] if candidate == engine]
                    if burst and role == "decode":
                        first = min(index for index in cell_indices
                                    if steps[(engine, index)]["record"]["virtual_time_ps"] >= raw["handoff"]["completed_at_ps"])
                    else:
                        first = min(cell_indices) if burst else previous_stop.get(engine, min(cell_indices))
                stop = own[-1][1] + 1
                evidence.equal(f"{prefix}:exact-slice:{raw['request_id']}:{role}",
                               row[role + "_record_indices"], list(range(first, stop)))
                previous_stop[engine] = stop
        if burst:
            evidence.check(prefix + ":burst-no-empty-driven-step", all(
                bool(steps[tuple(key)]["record"]["scheduled"]) for key in cell["step_identities"]))
    advances = data["clock_advances"]
    previous = 0
    positive_advances = []
    for index, row in enumerate(advances):
        evidence.fields(f"{prefix}:clock-fields:{index}", row, "before_ps after_ps")
        evidence.check(f"{prefix}:clock:{index}", integer(row["before_ps"])
                       and integer(row["after_ps"], row["before_ps"]) and row["before_ps"] == previous)
        previous = row["after_ps"]
        if row["after_ps"] > row["before_ps"]:
            positive_advances.append((row["before_ps"], row["after_ps"]))
    evidence.equal(prefix + ":final-clock", previous, data["host_measurements"]["final_clock_ps"])
    service_sum = sum(stop - start for start, stop in intervals)
    evidence.equal(prefix + ":serialized-union", interval_union(intervals), service_sum)
    remaining = Counter(positive_advances)
    for interval in intervals:
        evidence.check(prefix + ":service-advance:" + str(interval), remaining[interval] == 1)
        remaining[interval] -= 1
    burst_ids = {row["result"]["request_id"] for row in data["burst_cell"]["requests"]}
    handoff_sum = 0
    for request_id, row in requests.items():
        if request_id in burst_ids:
            continue
        handoff = row["result"]["handoff"]
        interval = (handoff["submitted_at_ps"], handoff["completed_at_ps"])
        evidence.check(prefix + ":serial-handoff-advance:" + request_id, remaining[interval] == 1)
        remaining[interval] -= 1
        handoff_sum += interval[1] - interval[0]
    idle_sum = 0
    for (start, stop), count in remaining.items():
        if not count:
            continue
        candidates = [row["result"]["handoff"]["completed_at_ps"]
                      for row in data["burst_cell"]["requests"]
                      if row["result"]["handoff"]["submitted_at_ps"] <= start
                      < row["result"]["handoff"]["completed_at_ps"]]
        evidence.check(prefix + ":burst-idle:" + str((start, stop)), count == 1
                       and bool(candidates) and stop == min(candidates)
                       and data["burst_cell"]["start_ps"] <= start < stop <= data["burst_cell"]["end_ps"])
        evidence.check(prefix + ":burst-idle-no-ready-work:" + str((start, stop)), not any(
            raw["admitted_at_ps"] <= start < raw["prefill_completed_at_ps"]
            or raw["handoff"]["completed_at_ps"] <= start < raw["decode_token_completed_at_ps"][-1]
            for raw in (row["result"] for row in data["burst_cell"]["requests"])))
        idle_sum += stop - start
    evidence.equal(prefix + ":clock-partition", service_sum + handoff_sum + idle_sum, previous)
    covered = []
    next_advance = data["serial_cells"][0]["clock_advance_indices"][0]
    for cell in [*data["serial_cells"], data["burst_cell"]]:
        indices = cell["clock_advance_indices"]
        evidence.check(prefix + ":cell-clock-indices:" + cell["id"],
                       bool(indices) and all(integer(index) for index in indices)
                       and indices == list(range(next_advance, next_advance + len(indices)))
                       and indices[-1] < len(advances))
        evidence.equal(prefix + ":cell-clock-start:" + cell["id"],
                       advances[indices[0]]["before_ps"], cell["start_ps"])
        evidence.equal(prefix + ":cell-clock-stop:" + cell["id"],
                       advances[indices[-1]]["after_ps"], cell["end_ps"])
        next_advance = indices[-1] + 1
        keys = [tuple(key) for key in cell["step_identities"]]
        evidence.check(prefix + ":cell-steps:" + cell["id"], len(keys) == len(set(keys))
                       and all(key in steps for key in keys))
        cell_internal = {row["result"][role + "_internal_request_id"]
                         for row in cell["requests"] for role in ("prefill", "decode")}
        evidence.check(prefix + ":cell-step-ownership:" + cell["id"], all(
            cell["start_ps"] <= steps[key]["record"]["virtual_time_ps"]
            <= steps[key]["result"]["completed_at_ps"] <= cell["end_ps"]
            and all(item["request_id"] in cell_internal for item in steps[key]["record"]["scheduled"])
            for key in keys))
        covered.extend(keys)
        service = sum(steps[key]["result"]["step_latency_ps"] for key in keys)
        duration = cell["end_ps"] - cell["start_ps"]
        if cell is data["burst_cell"]:
            ceiling = service + frozen["burst"]["requests"] * frozen["burst"]["handoff_ps"]
            evidence.check(prefix + ":burst-bound", service <= duration <= ceiling)
        else:
            handoff = next(spec["handoff_ps"] for spec in frozen["serial_cells"] if spec["id"] == cell["id"])
            evidence.equal(prefix + ":serial-partition:" + cell["id"], duration, service + 80 * handoff)
    for row in data["baseline_controls"]:
        for role in ("prefill", "decode"):
            covered.extend((row["result"][role + "_engine_id"], index)
                           for index in row[role + "_record_indices"])
    evidence.check(prefix + ":complete-step-partition", Counter(covered) == Counter(dict.fromkeys(steps, 1)))
    evidence.equal(prefix + ":complete-cell-clock-suffix", next_advance, len(advances))
    evidence.finish(prefix + ":accounting")
    return {"unique_service_ps": service_sum, "serial_handoff_ps": handoff_sum,
            "burst_wall_idle_ps": idle_sum, "final_clock_ps": previous,
            "unique_steps": len(steps), "unique_scheduled_steps": len(intervals)}


def check_relations(scale, frozen, metrics, cells, evidence):
    by_cell = {row["id"]: row for row in cells}
    for prompt in (8, 16):
        labels = [f"prompt-{prompt}-handoff-{handoff}" for handoff in (100000000, 200000000)]
        pairs = list(zip(metrics[labels[0]], metrics[labels[1]], strict=True))
        name = f"{scale['id']}:handoff-response:{prompt}"
        evidence.check(name + ":fixed", all(a["tpot_ps"] == b["tpot_ps"]
                       and a["prefill_service_ps"] == b["prefill_service_ps"]
                       and a["decode_first_token_service_ps"] == b["decode_first_token_service_ps"]
                       for a, b in pairs))
        durations = [by_cell[label]["end_ps"] - by_cell[label]["start_ps"] for label in labels]
        evidence.check(name, all(b["ttft_ps"] - a["ttft_ps"] == b["jct_ps"] - a["jct_ps"] == 100000000
                                for a, b in pairs) and durations[1] - durations[0] == 8000000000,
                       kind="relations", family="handoff-response")
    for handoff in (100000000, 200000000):
        pairs = list(zip(metrics[f"prompt-8-handoff-{handoff}"], metrics[f"prompt-16-handoff-{handoff}"], strict=True))
        name = f"{scale['id']}:prompt-response:{handoff}"
        evidence.check(name + ":fixed", all(b["kv_bytes"] == 2 * a["kv_bytes"] and b["tpot_ps"] >= a["tpot_ps"]
                                            for a, b in pairs))
        evidence.check(name, all(all(b[key] > a[key] for key in ("prefill_service_ps", "ttft_ps", "jct_ps"))
                                for a, b in pairs), kind="relations", family="prompt-response")


def check_baseline(data, frozen, evidence):
    rows = data["baseline_controls"]
    expected = frozen["baseline_control"]["comparisons"]
    evidence.equal("baseline:domain", [row["id"] for row in rows], [row["cell"] for row in expected])
    for row, reference in zip(rows, expected, strict=True):
        evidence.fields("baseline:row-fields:" + row["id"], row,
                        "id comparison comparison_sha256 result prefill_record_indices decode_record_indices")
        comparison = row["comparison"]
        evidence.fields("baseline:comparison-fields:" + row["id"], comparison,
                        "schema excluded_root_fields request")
        evidence.equal("baseline:comparison-schema:" + row["id"], comparison["schema"],
                       frozen["baseline_control"]["comparison_schema"])
        evidence.equal("baseline:declared-exclusion:" + row["id"], comparison["excluded_root_fields"],
                       frozen["baseline_control"]["excluded_root_fields"])
        evidence.equal("baseline:complete-projection:" + row["id"], comparison["request"],
                       {key: value for key, value in row["result"].items()
                        if key not in frozen["baseline_control"]["excluded_root_fields"]})
        evidence.equal("baseline:hash-receipt:" + row["id"], row["comparison_sha256"], sha(exact_json_bytes(comparison)))
        evidence.equal("baseline:accepted:" + row["id"], row["comparison_sha256"], reference["comparison_sha256"],
                       kind="oracles")
    evidence.finish("baseline-control")
