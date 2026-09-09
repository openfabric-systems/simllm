"""Join packet ownership and physical serialization to native cache releases."""

from __future__ import annotations

from collections import defaultdict
from urllib.parse import quote

from .aliases import request_rows


def geometry(spec, frozen):
    shape, backend = frozen["geometry"], frozen["backend"]
    total = (spec["prompt_tokens"] * 2 * shape["num_layers"] * shape["num_kv_heads"]
             * shape["head_size"] * shape["element_bytes"])
    shard, remainder = divmod(total, shape["tensor_parallel_size"])
    packet_payload = backend["max_wire_packet_bytes"] - backend["data_header_bytes"]
    packets, partial = divmod(shard, packet_payload)
    serialization, fractional = divmod(backend["max_wire_packet_bytes"] * 8 * 10**12, spec["link_rate_bps"])
    if remainder or partial or fractional or min(total, packets, serialization) <= 0:
        raise ValueError("frozen packet geometry is not exactly integral")
    return total, shard, packets, serialization


def check_network(data, spec, frozen, deadline, evidence):
    label = spec["id"] + ":network"
    if spec["kind"] == "compatibility":
        for key in ("network_before", "network_before_close", "network_after"):
            evidence.equal(label + ":off:" + key, data[key], {"enabled": False})
        evidence.equal(label + ":observer-off", data["network_observations"], [])
        return {}
    backend = frozen["backend"]
    before, after, closing = (data[key] for key in ("network_before", "network_after", "network_before_close"))
    requests = request_rows(data)
    total, shard_bytes, packets, q = geometry(spec, frozen)
    p, delay = backend["propagation_ps"], backend["pcie_submission_ps"]
    ranks = {f"simllm-{role}-{index}": list(range(8 * offset, 8 * offset + 8))
             for role, count, base in (("prefill", spec["prefill_engines"], 0),
                                      ("decode", spec["decode_engines"], spec["prefill_engines"]))
             for index in range(count) for offset in (base + index,)}
    roles = {name: "prefill" if name.startswith("simllm-prefill") else "decode" for name in ranks}
    evidence.equal(label + ":global-ranks", after["engine_ranks"], ranks)
    evidence.equal(label + ":roles", after["engine_roles"], roles)
    evidence.equal(label + ":clock-owner", after["clock_id"], data["identity"]["clock_object_id"])
    expected_config = {"profile": backend["profile"], "node_count": 8 * len(ranks),
                       "link_rate_bps": spec["link_rate_bps"],
                       "effective_hardware_sha256": backend["effective_hardware_sha256"],
                       "policy_context_token": backend["policy_context_token"], "seed": backend["seed"],
                       "max_events": backend["max_events_per_call"],
                       "simulation_budget_ps": backend["simulation_budget_ps"], **deadline["shared_flow_session"]}
    evidence.equal(label + ":config", after["config"], expected_config)
    evidence.check(label + ":clean-close", after["enabled"] is True and after["closed"] is True
                   and after["poisoned"] is False and after["child_exit_code"] == 0
                   and closing["closed"] is False and closing["drain"] is None)
    evidence.equal(label + ":empty-terminal", [after["pending"], after["staged"], closing["pending"], closing["staged"]], [[], [], [], []])
    evidence.equal(label + ":initial-inventory", [before[key] for key in ("accepted", "events", "rows", "joins", "pending", "staged")],
                   [[], [], [], [], [], []])
    for key in ("owner_id", "clock_id", "session_id", "child_pid", "config", "engine_ranks", "engine_roles"):
        evidence.equal(label + ":owner-stable:" + key, before[key], after[key])
        evidence.equal(label + ":close-owner-stable:" + key, closing[key], after[key])
    for key in ("accepted", "events", "rows", "joins"):
        evidence.equal(label + ":close-retained:" + key, closing[key], after[key])
    expected, request_sequences = [], {}
    for row in requests:
        raw = row["result"]
        operation = "kv/" + quote(raw["request_id"], safe="")
        sequences = []
        for local, (source, destination) in enumerate(zip(ranks[raw["prefill_engine_id"]], ranks[raw["decode_engine_id"]], strict=True)):
            sequence = len(expected) + 1
            sequences.append(sequence)
            expected.append({"request_id": raw["request_id"], "execution_id": "shared-kv:" + spec["id"],
                             "operation_id": operation, "flow_id": operation + f"/shard/{local}",
                             "source": source, "destination": destination, "payload_bytes": shard_bytes, "tag": 6200,
                             "eligible_at_ps": raw["prefill_completed_at_ps"] + delay, "sequence": sequence})
        request_sequences[raw["request_id"]] = sequences
    evidence.equal(label + ":accepted-global-bindings", after["accepted"], expected)
    evidence.equal(label + ":sequence-count", after["last_sequence"], len(expected))
    evidence.equal(label + ":flow-sequences", [row["sequence"] for row in after["rows"]], list(range(1, len(expected) + 1)))
    evidence.equal(label + ":four-phases-per-flow", len(after["events"]), 4 * len(expected))
    flows = {row["sequence"]: row for row in after["rows"]}
    phases = defaultdict(list)
    for event in after["events"]:
        phases[event["sequence"]].append(event)
    joins, durations = [], {}
    for row in requests:
        raw = row["result"]
        name, sequences = label + ":request:" + raw["request_id"], request_sequences[raw["request_id"]]
        shards = [{"row": flows[sequence], "events": phases[sequence]} for sequence in sequences]
        completed = max(flows[sequence]["completion_time_ps"] for sequence in sequences)
        expected_join = {"authority": frozen["authority"]["shared"], "pricing_arm": frozen["authority"]["shared_pricing_arm"],
                         "source_engine_id": raw["prefill_engine_id"], "destination_engine_id": raw["decode_engine_id"],
                         "kv_bytes": total, "submitted_at_ps": raw["prefill_completed_at_ps"],
                         "eligible_at_ps": raw["prefill_completed_at_ps"] + delay, "completed_at_ps": completed,
                         "critical_shard_sequences": [sequence for sequence in sequences if flows[sequence]["completion_time_ps"] == completed],
                         "shards": shards}
        evidence.equal(name + ":complete-join", raw["handoff"], expected_join)
        joins.append(expected_join)
        durations[raw["request_id"]] = completed - raw["prefill_completed_at_ps"]
        evidence.check(name + ":decode-release", raw["decode_eligible_at_ps"] >= completed)
        for sequence in sequences:
            flow = flows[sequence]
            prefix = name + ":shard:" + str(sequence)
            admitted = expected[sequence - 1]
            for key in ("execution_id", "operation_id", "flow_id", "source", "destination", "payload_bytes", "tag", "sequence"):
                evidence.equal(prefix + ":" + key, flow[key], admitted[key])
            evidence.equal(prefix + ":eligibility", flow["start_time_ps"], admitted["eligible_at_ps"])
            evidence.equal(prefix + ":fct", flow["fct_ps"], flow["completion_time_ps"] - flow["start_time_ps"])
            evidence.equal(prefix + ":lifecycle", [item["kind"] for item in phases[sequence]], ["accepted", "queued", "started", "completed"])
            lower = ((packets + 1) if spec["decode_engines"] == 2 else 2 * packets) * q + p
            upper = ((packets + 1) if spec["decode_engines"] == 2 else 2 * packets + 1) * q + p
            evidence.check(prefix + ":packet-service-bounds", lower <= flow["fct_ps"] <= upper)
    evidence.equal(label + ":join-publication-order", after["joins"], sorted(joins, key=lambda row: (
        row["completed_at_ps"], row["shards"][0]["row"]["sequence"])))
    first_seen, previous_rows = set(), set()
    previous_sizes = dict.fromkeys(("accepted", "events", "joins"), 0)
    for ordinal, observation in enumerate(data["network_observations"]):
        snapshot, at = observation["snapshot"], observation["at_ps"]
        name = label + ":observation:" + str(ordinal)
        evidence.equal(name + ":index", observation["index"], ordinal)
        for key in ("owner_id", "clock_id", "session_id", "child_pid", "config", "engine_ranks", "engine_roles"):
            evidence.equal(name + ":stable:" + key, snapshot[key], after[key])
        for key in ("accepted", "events", "joins"):
            evidence.check(name + ":growing:" + key, len(snapshot[key]) >= previous_sizes[key])
            evidence.equal(name + ":prefix:" + key, snapshot[key], after[key][:len(snapshot[key])])
            previous_sizes[key] = len(snapshot[key])
        sequences = [row["sequence"] for row in snapshot["rows"]]
        evidence.check(name + ":row-conservation", len(sequences) == len(set(sequences))
                       and previous_rows <= set(sequences))
        evidence.check(name + ":known-rows", all(flows.get(row["sequence"]) == row for row in snapshot["rows"]))
        previous_rows = set(sequences)
        for join in snapshot["joins"]:
            identity = join["shards"][0]["row"]["sequence"]
            if identity not in first_seen:
                evidence.equal(name + ":exact-visibility:" + str(identity), at, join["completed_at_ps"])
                first_seen.add(identity)
    evidence.equal(label + ":all-visible", len(first_seen), len(requests))
    for ordinal, cell in enumerate(data["cells"]):
        sequence_domain = [sequence for row in cell["requests"] for sequence in request_sequences[row["result"]["request_id"]]]
        by_receiver = defaultdict(list)
        for sequence in sequence_domain:
            by_receiver[flows[sequence]["destination"]].append(flows[sequence])
        for endpoint, values in by_receiver.items():
            for k, flow in enumerate(sorted(values, key=lambda row: row["completion_time_ps"]), 1):
                evidence.check(label + f":cell:{ordinal}:receiver:{endpoint}:earliest:{k}", flow["fct_ps"] >= (k * packets + 1) * q + p)
        if spec["decode_engines"] == 2:
            evidence.equal(label + f":cell:{ordinal}:unshared-vector", [flows[sequence]["fct_ps"] for sequence in sequence_domain],
                           [(packets + 1) * q + p] * len(sequence_domain), kind="oracles")
        snapshot = cell["network_after"]
        evidence.equal(label + f":cell:{ordinal}:persistent-sequence", snapshot["last_sequence"], (ordinal + 1) * 16)
        evidence.equal(label + f":cell:{ordinal}:no-drain", snapshot["drain"], None)
        evidence.check(label + f":cell:{ordinal}:still-open", snapshot["closed"] is False and not snapshot["pending"] and not snapshot["staged"])
    return durations
