"""Admit native aliases before projecting the three frozen identity fields."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy


def request_rows(data):
    return [row for cell in data["cells"] for row in cell["requests"]]


def admit_bindings(data, evidence):
    """Build a bijection, then require the complete scheduled membership."""
    label = data["process_id"] + ":aliases"
    roles = {row["engine_id"]: row["role"] for row in data["retained_before"]}
    bindings, public, aliases = {}, set(), set()
    for ordinal, row in enumerate(request_rows(data)):
        raw = row["result"]
        name = label + ":request:" + str(ordinal)
        identifier = raw["request_id"]
        evidence.check(name + ":public", type(identifier) is str and bool(identifier) and identifier not in public)
        public.add(identifier)
        for role in ("prefill", "decode"):
            engine, alias = raw[role + "_engine_id"], raw[role + "_internal_request_id"]
            evidence.equal(name + ":role:" + role, roles.get(engine), role)
            evidence.check(name + ":unique:" + role, type(alias) is str and bool(alias)
                           and alias != identifier and alias not in aliases and (engine, alias) not in bindings)
            aliases.add(alias)
            bindings[(engine, alias)] = [engine, identifier, role]
    scheduled = Counter()
    for ordinal, step in enumerate(data["steps"]):
        engine, record = step["engine_id"], step["record"]
        name = label + ":record:" + str(ordinal)
        evidence.check(name + ":engine", engine in roles)
        for field in ("finished_request_ids", "preempted_request_ids"):
            values = record[field]
            evidence.check(name + ":unique:" + field, len(values) == len(set(values)))
            for index, alias in enumerate(values):
                evidence.check(name + ":owner:" + field + ":" + str(index), (engine, alias) in bindings)
        values = [item["request_id"] for item in record["scheduled"]]
        evidence.check(name + ":scheduled-unique", len(values) == len(set(values)))
        for index, alias in enumerate(values):
            evidence.check(name + ":scheduled-owner:" + str(index), (engine, alias) in bindings)
            scheduled[(engine, alias)] += 1
    evidence.equal(label + ":complete-memberships", sorted(
        [engine, alias, count] for (engine, alias), count in scheduled.items()), sorted(
        [engine, alias, 1 if value[2] == "prefill" else 4] for (engine, alias), value in bindings.items()))
    observed = set()
    captures = [*data["checkpoints"], *data["serialized_observations"]]
    for ordinal, capture in enumerate(captures):
        engine, state = capture["engine_id"], capture["native"]
        name = label + ":native:" + str(ordinal)
        locations = {"requests": list(state["requests"]), "visible": list(state["visible"]),
                     "waiting": state["queues"]["waiting"], "running": state["queues"]["running"]}
        locations.update({"cache-group-" + str(index): list(group) for index, group in enumerate(state["cache"]["groups"])})
        for location, values in locations.items():
            for index, alias in enumerate(values):
                evidence.check(name + ":" + location + ":" + str(index), (engine, alias) in bindings)
        for index, (alias, row) in enumerate(state["visible"].items()):
            evidence.equal(name + ":external:" + str(index), row["external_request_id"], bindings[(engine, alias)][1])
            if alias in state["requests"]:
                observed.add((engine, alias))
        for index, row in enumerate(state["emitted"]):
            evidence.check(name + ":emitted:" + str(index), any(
                value[0] == engine and value[1] == row["request_id"] for value in bindings.values()))
    evidence.equal(label + ":observed-membership-domain", [list(key) for key in sorted(observed)],
                   [list(key) for key in sorted(bindings)])
    return bindings


def _no_aliases(value, aliases, path, evidence):
    """Reject undeclared locations; this walk never rewrites any string."""
    if type(value) is str:
        evidence.check(path, not any(alias in value for alias in aliases))
    elif type(value) is dict:
        for index, (key, item) in enumerate(value.items()):
            _no_aliases(key, aliases, path + f":key:{index}", evidence)
            _no_aliases(item, aliases, path + f":value:{index}", evidence)
    elif type(value) in (tuple, list):
        for index, item in enumerate(value):
            _no_aliases(item, aliases, path + f":item:{index}", evidence)


def comparison(data, bindings, evidence):
    """Project admitted aliases without removing fields or changing raw data."""
    label = data["process_id"] + ":comparison"
    aliases = {alias for _, alias in bindings}
    cells, steps = deepcopy(data["cells"]), deepcopy(data["steps"])
    excluded = ["prefill_internal_request_id", "decode_internal_request_id"]
    for ordinal, cell in enumerate(cells):
        evidence.equal(label + ":comparison-count:" + str(ordinal), len(cell["comparisons"]), len(cell["requests"]))
        for index, (row, compared) in enumerate(zip(cell["requests"], cell["comparisons"], strict=True)):
            name = label + f":cell:{ordinal}:request:{index}"
            evidence.equal(name + ":excluded", compared["excluded_root_fields"], excluded)
            expected = {key: value for key, value in row["result"].items() if key not in excluded}
            evidence.equal(name + ":request", compared["request"], expected)
            row["result"] = expected
    for ordinal, step in enumerate(steps):
        name = label + ":step:" + str(ordinal)
        engine, record = step["engine_id"], step["record"]
        for index, member in enumerate(record["scheduled"]):
            key = (engine, member["request_id"])
            evidence.check(name + ":scheduled:" + str(index), key in bindings)
            member["request_id"] = list(bindings[key])
        for field in ("finished_request_ids", "preempted_request_ids"):
            projected = []
            for index, alias in enumerate(record[field]):
                key = (engine, alias)
                evidence.check(name + ":" + field + ":" + str(index), key in bindings)
                projected.append(list(bindings[key]))
            record[field] = projected
    value = {"authority": data["authority"], "cells": cells, "steps": steps,
             "sinks": data["sinks"], "projections": data["projections"],
             "clock_advances": data["clock_advances"], "final_clock_ps": data["final_clock_ps"],
             "selected_before": data["selected_before"], "selected_after": data["selected_after"]}
    _no_aliases(value, aliases, label + ":undeclared-alias", evidence)
    return deepcopy(value)
