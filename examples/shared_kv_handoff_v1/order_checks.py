"""Join exact packet-call horizons and publications to the single public clock."""

from __future__ import annotations

from collections import defaultdict

from examples.pd_session_target_scale_v1.checks import integer

from .aliases import request_rows


def packet_call(data, spec, wire, row, position, state, pending_heap, retiring, published, previous_wire, evidence):
    label = spec["id"] + ":ordered-network:" + str(row["index"])
    action, call, snapshot = row["action"], row["call"], row["snapshot"]
    evidence.fields(label + ":fields", row, "index action at_ps call result clock_after transcript_stop snapshot")
    evidence.fields(label + ":call-fields", call, "arguments clock_before transcript_start order_start")
    evidence.check(label + ":action", action in ("bind", "submit", "progress", "complete_due", "close"))
    evidence.equal(label + ":call-order", call["order_start"], position)
    next_due = min((value[0] for value in pending_heap.values()), default=None)
    closing = action == "close"
    expected_clock = {"at_ps": state["now"], "next_engine_completion_ps": next_due,
        "engine_owner_present": True, "advance_count": state["clock"],
        "event_count": state["events"], "visit_count": state["retired"],
        "result_count": state["retired"]}
    evidence.equal(label + ":before-clock", call["clock_before"], expected_clock)
    evidence.equal(label + ":nonadvancing-call", row["clock_after"], expected_clock)
    evidence.equal(label + ":wire-start", call["transcript_start"], previous_wire)
    stop = row["transcript_stop"]
    evidence.check(label + ":wire-stop", integer(stop, previous_wire) and stop % 2 == 0 and stop in wire["prefixes"])
    current, prior = wire["prefixes"][stop], wire["prefixes"][previous_wire]
    requests = wire["requests"][previous_wire // 2:stop // 2]
    responses = wire["responses"][previous_wire // 2:stop // 2]
    arguments = call["arguments"]
    raw = {row["result"]["request_id"]: row["result"] for row in request_rows(data)}
    accepted = data["network_after"]["accepted"]
    by_request = defaultdict(list)
    for flow in accepted:
        by_request[flow["request_id"]].append(flow["sequence"])
    active = [name for name, sequences in by_request.items() if sequences[-1] <= prior["last_sequence"] and name not in published]
    receipts = {name: {"request_id": name, "source_engine_id": value["prefill_engine_id"],
        "destination_engine_id": value["decode_engine_id"], "kv_bytes": value["handoff"]["kv_bytes"],
        "submitted_at_ps": value["prefill_completed_at_ps"], "eligible_at_ps": value["handoff"]["eligible_at_ps"],
        "sequences": by_request[name]} for name, value in raw.items()}
    if action == "bind":
        evidence.equal(label + ":first-call", [position, previous_wire, stop], [0, 0, 2])
        evidence.equal(label + ":arguments", arguments, {"clock_id": data["identity"]["clock_object_id"],
                       "engine_local_ranks": {name: list(range(8)) for name in data["sinks"]}})
        evidence.equal(label + ":verbs", [request["verb"] for request in requests], ["open"])
        evidence.equal(label + ":result", row["result"], None)
    elif action == "submit":
        name = arguments["request_id"]
        evidence.check(label + ":request", name in receipts and name not in active and name not in published)
        expected = {key: value for key, value in receipts[name].items() if key not in ("eligible_at_ps", "sequences")}
        evidence.equal(label + ":arguments", arguments, expected)
        evidence.equal(label + ":at-producer-completion", state["now"], raw[name]["prefill_completed_at_ps"])
        evidence.check(label + ":inside-producer-retirement", retiring is not None and retiring[0] == raw[name]["prefill_engine_id"])
        evidence.equal(label + ":verbs", [request["verb"] for request in requests], ["inject"] * 8)
        evidence.equal(label + ":result", row["result"], receipts[name])
    elif action in ("progress", "complete_due"):
        evidence.check(label + ":all-engine-completions-first", retiring is None and (next_due is None or next_due > state["now"]))
        evidence.equal(label + ":pending-inputs", arguments["pending"], [receipts[name] for name in active])
        evidence.check(label + ":await-only", all(request["verb"] == "await_completion" and request["until_quiescent"] is False for request in requests))
        if action == "progress":
            evidence.fields(label + ":argument-fields", arguments, "pending through_ps")
            cell = next(cell for cell in data["cells"] if cell["observation_start"] <= position < cell["observation_stop"])
            candidates = [item["admitted_at_ps"] for item in cell["inputs"] if item["admitted_at_ps"] > state["now"]]
            if next_due is not None:
                candidates.append(next_due)
            horizon = min(candidates) - 1 if candidates else None
            evidence.equal(label + ":next-local-deadline", arguments["through_ps"], horizon)
            evidence.equal(label + ":one-exchange", len(requests), int(bool(prior["pending_sequences"])))
            result = responses[-1]["event_time_ps"] if responses and responses[-1]["reason"] == "completion" else None
            evidence.equal(label + ":native-progress-result", row["result"], result)
            evidence.check(label + ":progress-future", result is None or result > state["now"])
        else:
            evidence.fields(label + ":argument-fields", arguments, "pending")
            horizon = state["now"]
            prior_completed = {event["sequence"] for event in prior["events"] if event["kind"] == "completed"}
            future_staged = any(set(by_request[name]) <= prior_completed
                                and raw[name]["handoff"]["completed_at_ps"] > horizon for name in active)
            blocked = future_staged or (prior["injection_floor_ps"] is not None
                                       and prior["injection_floor_ps"] > horizon)
            if blocked:
                evidence.equal(label + ":future-boundary-short-circuit", [requests, row["result"]], [[], []])
            else:
                evidence.equal(label + ":drain-initial-pending", bool(requests), bool(prior["pending_sequences"]))
                for index, response in enumerate(responses):
                    prefix = wire["prefixes"][previous_wire + 2 * index]
                    evidence.check(label + ":drain-has-work:" + str(index), bool(prefix["pending_sequences"]))
                    evidence.check(label + ":drain-reason:" + str(index), response["reason"] in ("completion", "horizon"))
                    if response["reason"] == "horizon":
                        evidence.equal(label + ":drain-horizon-is-terminal:" + str(index), index, len(responses) - 1)
                        evidence.equal(label + ":drain-complete-time-prefix", response["fully_processed_horizon_ps"], horizon)
                if current["pending_sequences"]:
                    evidence.check(label + ":drain-exhausts-same-time-callbacks",
                                   bool(responses) and responses[-1]["reason"] == "horizon")
        for index, request in enumerate(requests):
            evidence.equal(label + ":horizon:" + str(index), request["through_ps"], horizon)
            prefix = wire["prefixes"][previous_wire + 2 * index]
            evidence.equal(label + ":all-pending-flow-targets:" + str(index),
                           request["completion_sequences"], prefix["pending_sequences"])
    else:
        evidence.check(label + ":empty-close", retiring is None and not pending_heap and not active)
        evidence.equal(label + ":arguments", arguments, {})
        evidence.equal(label + ":verbs", [request["verb"] for request in requests], ["await_completion", "drain", "close"])
        evidence.equal(label + ":result", row["result"], None)
    for key in ("events", "rows", "last_sequence", "drain"):
        evidence.equal(label + ":exact-wire-prefix:" + key, snapshot[key], current[key])
    evidence.equal(label + ":accepted-prefix", snapshot["accepted"], accepted[:current["last_sequence"]])
    evidence.check(label + ":whole-shard-submissions", current["last_sequence"] % 8 == 0)
    completed, ready_order = set(), []
    flow_owners = {sequence: name for name, sequences in by_request.items() for sequence in sequences}
    for event in current["events"]:
        if event["kind"] == "completed":
            completed.add(event["sequence"])
            name = flow_owners[event["sequence"]]
            if name not in ready_order and set(by_request[name]) <= completed:
                ready_order.append(name)
    now_active = [name for name, sequences in by_request.items() if sequences[-1] <= current["last_sequence"] and name not in published]
    if action == "complete_due":
        ready = [name for name in now_active if name in ready_order and raw[name]["handoff"]["completed_at_ps"] == state["now"]]
        evidence.equal(label + ":all-due-joins", row["result"], [{"request_id": name, "join": raw[name]["handoff"]} for name in ready])
        published.extend(ready)
    evidence.equal(label + ":published-owner-prefix", snapshot["joins"], [raw[name]["handoff"] for name in published])
    evidence.equal(label + ":pending-owner-prefix", snapshot["pending"], [name for name in now_active if name not in published])
    evidence.equal(label + ":staged-owner-prefix", snapshot["staged"], [name for name in ready_order if name not in published])
    evidence.equal(label + ":closed", snapshot["closed"], closing)
    evidence.equal(label + ":no-poison", snapshot["poisoned"], False)
    return stop


def check_order(data, spec, wire, evidence):
    label = spec["id"] + ":observation-order"
    state = {"now": 0, "clock": 0, "engine": 0, "serialized": 0, "network": 0, "events": 0, "retired": 0}
    boundaries, heap, retiring, published, last_wire, last_network = [], {}, None, [], 0, None
    receipts = {(row["receipt"]["engine_id"], row["receipt"]["step_index"]): row["receipt"] for row in data["projections"]["completed"]}
    for position, row in enumerate(data["observation_order"]):
        boundaries.append({**state, "last_network": last_network})
        name = label + ":" + str(position)
        evidence.fields(name + ":fields", row, "sequence kind index at_ps")
        evidence.equal(name + ":sequence", row["sequence"], position)
        kind = row["kind"]
        evidence.check(name + ":kind", kind in ("clock", "engine", "serialized", "network"))
        evidence.equal(name + ":index", row["index"], state[kind])
        if kind == "clock":
            advance = data["clock_advances"][row["index"]]
            evidence.equal(name + ":before-clock", advance["before_ps"], state["now"])
            evidence.check(name + ":no-clock-rewind", integer(advance["after_ps"], state["now"]))
            evidence.check(name + ":no-clock-inside-retirement", retiring is None)
            evidence.check(name + ":no-skipped-engine-event", all(advance["after_ps"] <= value[0] for value in heap.values()))
            if spec["kind"] != "compatibility":
                cell = next(cell for cell in data["cells"] if cell["observation_start"] <= position < cell["observation_stop"])
                candidates = [value[0] for value in heap.values()]
                candidates.extend(item["admitted_at_ps"] for item in cell["inputs"] if item["admitted_at_ps"] > state["now"])
                previous = data["observation_order"][position - 1] if position else None
                if previous is not None and previous["kind"] == "network":
                    call = data["network_observations"][previous["index"]]
                    if call["action"] == "progress" and call["result"] is not None:
                        candidates.append(call["result"])
                evidence.check(name + ":driver-has-deadline", bool(candidates))
                evidence.equal(name + ":next-causal-deadline", advance["after_ps"], min(candidates))
            state["now"] = advance["after_ps"]
        evidence.equal(name + ":public-time", row["at_ps"], state["now"])
        if kind == "engine":
            checkpoint = data["checkpoints"][row["index"]]
            evidence.equal(name + ":checkpoint-time", checkpoint["at_ps"], state["now"])
            key, phase = (checkpoint["engine_id"], checkpoint["step_index"]), checkpoint["phase"]
            receipt = receipts[key]
            if phase == "before-submit":
                evidence.check(name + ":submission-legal", retiring is None and key[0] not in heap
                               and all(value[0] > state["now"] for value in heap.values()))
            elif phase == "submitted":
                evidence.check(name + ":new-reservation", key[0] not in heap)
                heap[key[0]] = (receipt["completed_at_ps"], receipt["sequence"], key)
                state["events"] += 3
            elif phase == "before-retire":
                expected = min(heap.values())
                evidence.equal(name + ":next-retirement", [*key, state["now"]], [*expected[2], expected[0]])
                evidence.equal(name + ":single-retiring-callback", retiring, None)
                del heap[key[0]]
                retiring = key
            else:
                evidence.equal(name + ":retiring-callback", list(retiring) if retiring else None, list(key))
            capture = data["runtime_evidence"][row["index"]]
            evidence.equal(name + ":runtime-clock-prefix", capture["stops"]["clock_advances"], state["clock"])
            evidence.equal(name + ":runtime-event-prefix", capture["stops"]["events"], state["events"])
            evidence.equal(name + ":runtime-retirement-prefix", [capture["stops"]["visits"], capture["stops"]["completed"]], [state["retired"]] * 2)
            if phase == "retired":
                state["events"] += 2
                state["retired"] += 1
                retiring = None
        elif kind == "serialized":
            checkpoint = data["serialized_observations"][row["index"]]
            evidence.equal(name + ":serialized-time", checkpoint["at_ps"], state["now"])
            evidence.equal(name + ":serialized-clock-prefix", data["runtime_evidence"][row["index"]]["stops"]["clock_advances"], state["clock"])
        elif kind == "network":
            observation = data["network_observations"][row["index"]]
            evidence.equal(name + ":network-time", observation["at_ps"], state["now"])
            last_wire = packet_call(data, spec, wire, observation, position, state, heap, retiring, published, last_wire, evidence)
            last_network = row["index"]
        state[kind] += 1
    boundaries.append({**state, "last_network": last_network})
    evidence.equal(label + ":complete-inventory", {key: state[key] for key in ("clock", "engine", "serialized", "network")},
        {"clock": len(data["clock_advances"]), "engine": len(data["checkpoints"]),
         "serialized": len(data["serialized_observations"]), "network": len(data["network_observations"])})
    evidence.equal(label + ":empty-engine-heap", [list(heap), retiring], [[], None])
    evidence.equal(label + ":full-wire", last_wire, len(wire["requests"]) * 2)
    start = 1 if spec["kind"] != "compatibility" else 0
    for ordinal, cell in enumerate(data["cells"]):
        name = label + ":cell:" + str(ordinal)
        evidence.equal(name + ":start", cell["observation_start"], start)
        stop = cell["observation_stop"]
        evidence.check(name + ":stop", integer(stop, start + 1) and stop <= len(data["observation_order"]))
        before, after = boundaries[start], boundaries[stop]
        evidence.equal(name + ":times", [before["now"], after["now"]], [cell["start_ps"], cell["end_ps"]])
        for field, counter in (("clock_advance_indices", "clock"), ("event_indices", "events"), ("visit_indices", "retired")):
            evidence.equal(name + ":" + field, cell[field], list(range(before[counter], after[counter])))
        if spec["kind"] != "compatibility":
            evidence.equal(name + ":exact-network-snapshot", cell["network_after"], data["network_observations"][after["last_network"]]["snapshot"])
        start = stop
    evidence.equal(label + ":terminal-call-domain", len(data["observation_order"]) - start, int(spec["kind"] != "compatibility"))
