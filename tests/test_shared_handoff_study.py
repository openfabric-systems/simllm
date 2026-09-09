"""Corrupt the frozen native alias boundary without running a serving engine."""

from __future__ import annotations

from copy import deepcopy

import pytest

from examples.pd_session_target_scale_v1.checks import Evidence, GuardFailure
from examples.shared_kv_handoff_v1.aliases import admit_bindings, comparison


def alias_fixture(suffix):
    raw = {"request_id": "public", "prefill_engine_id": "prefill-0", "decode_engine_id": "decode-0",
           "prefill_internal_request_id": "prefill:" + suffix,
           "decode_internal_request_id": "decode:" + suffix, "ttft_ps": 23}
    excluded = ["prefill_internal_request_id", "decode_internal_request_id"]
    compared = {"schema": "fixture", "excluded_root_fields": excluded,
                "request": {key: value for key, value in raw.items() if key not in excluded}}
    data = {"process_id": "fixture", "authority": "independent", "retained_before": [], "steps": [],
            "checkpoints": [], "serialized_observations": [],
            "cells": [{"requests": [{"result": raw}], "comparisons": [compared]}],
            "sinks": {"prefill-0": {"latencies": [11]}, "decode-0": {"latencies": [12]}},
            "projections": {"events": [{"at_ps": 23}], "visits": [], "completed": []},
            "clock_advances": [{"before_ps": 0, "after_ps": 23}], "final_clock_ps": 23,
            "selected_before": [{"precision": "exact"}], "selected_after": [{"precision": "exact"}]}
    for role, count in (("prefill", 1), ("decode", 4)):
        engine, alias = role + "-0", raw[role + "_internal_request_id"]
        data["retained_before"].append({"engine_id": engine, "role": role})
        for index in range(count):
            data["steps"].append({"engine_id": engine, "record": {
                "step_index": index, "scheduled": [{"request_id": alias, "tokens": 1}],
                "finished_request_ids": [], "preempted_request_ids": [], "virtual_time_ps": index * 12},
                "result": {"step_latency_ps": 12}})
        data["checkpoints"].append({"engine_id": engine, "native": {
            "requests": {alias: {"status": "RUNNING"}}, "visible": {alias: {"external_request_id": "public"}},
            "cache": {"groups": [{alias: [1]}]}, "queues": {"waiting": [], "running": [alias]},
            "emitted": [{"request_id": "public"}]}})
    return data


def project(data):
    evidence = Evidence([])
    return comparison(data, admit_bindings(data, evidence), evidence)


def test_alias_projection_permits_only_coherent_bijective_renaming():
    before, after = alias_fixture("before"), alias_fixture("after")
    retained = deepcopy((before, after))
    assert project(before) == project(after)
    assert (before, after) == retained
    projected = project(before)
    assert projected["steps"][0]["record"]["scheduled"][0]["request_id"] == ["prefill-0", "public", "prefill"]
    assert set(projected["steps"][0]["record"]) == set(before["steps"][0]["record"])


@pytest.mark.parametrize("kind", ["alias-collision", "foreign-alias", "wrong-engine-binding",
                                  "missing-record-member", "unrecognized-alias-location", "partial-renaming"])
def test_alias_fatal_controls_reject_before_comparison(kind):
    data = alias_fixture("native")
    raw = data["cells"][0]["requests"][0]["result"]
    if kind == "alias-collision":
        raw["decode_internal_request_id"] = raw["prefill_internal_request_id"]
    elif kind == "foreign-alias":
        data["steps"][0]["record"]["scheduled"][0]["request_id"] = "foreign"
    elif kind == "wrong-engine-binding":
        raw["prefill_engine_id"] = "decode-0"
    elif kind == "missing-record-member":
        data["steps"].pop()
    elif kind == "unrecognized-alias-location":
        data["sinks"]["prefill-0"]["opaque"] = "unexpected=" + raw["prefill_internal_request_id"]
    else:
        raw["prefill_internal_request_id"] = "coherently-changed-projection"
        data["steps"][0]["record"]["scheduled"][0]["request_id"] = raw["prefill_internal_request_id"]
    with pytest.raises(GuardFailure):
        project(data)


@pytest.mark.parametrize("member", ["sinks", "projections", "clock_advances", "selected_before", "selected_after"])
def test_projection_retains_every_nonidentity_value(member):
    data = alias_fixture("native")
    changed = deepcopy(data)
    if type(changed[member]) is dict:
        changed[member]["new_value"] = 99
    else:
        changed[member].append({"new_value": 99})
    assert project(changed) != project(data)


@pytest.mark.parametrize("field", ["finished_request_ids", "preempted_request_ids"])
def test_projection_preserves_terminal_identity_field_and_order(field):
    data = alias_fixture("native")
    record = data["steps"][0]["record"]
    record[field] = ["prefill:native"]
    assert project(data)["steps"][0]["record"][field] == [["prefill-0", "public", "prefill"]]
    record[field] *= 2
    with pytest.raises(GuardFailure):
        project(data)


def test_serialized_aliases_require_the_same_native_observation_binding():
    data = alias_fixture("native")
    data["serialized_observations"], data["checkpoints"] = data["checkpoints"], []
    assert project(data)
    data["serialized_observations"].pop()
    with pytest.raises(GuardFailure, match="observed-membership-domain"):
        project(data)
