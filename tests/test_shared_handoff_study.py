"""Corrupt the frozen native alias boundary without running a serving engine."""

from __future__ import annotations

import json
from copy import deepcopy
from itertools import pairwise

import pytest

from examples.independent_engine_completion_v1.checks import expected_times
from examples.independent_engine_completion_v1.reference import native_reference
from examples.pd_session_target_scale_v1.checks import Evidence, GuardFailure
from examples.shared_kv_handoff_v1.aliases import admit_bindings, comparison
from examples.shared_kv_handoff_v1.checks import check_service_vector
from examples.shared_kv_handoff_v1.common import HERE

SERVICE_CONTROLS = json.loads((HERE / "service-vector-expectations.json").read_bytes())["software_controls"]


def service_fixture(mode, handoff, duration, admission):
    process = {"mode": mode, "prefill_engines": 2, "decode_engines": 2}
    cell = {"arrival_offsets_ps": [0, 0], "prompt_tokens": [8, 8],
            "handoff_ps": handoff, "decode_output_tokens": 4}
    reference = native_reference(process, cell, {"8": {"prefill": 13, "decode": [duration] * 4}})
    owned = [{"engine_id": row["engine_id"], "record": {"virtual_time_ps": admission + row["start_ps"]},
              "result": {"completed_at_ps": admission + row["end_ps"], "step_latency_ps": row["end_ps"] - row["start_ps"]}}
             for row in reference["service_steps"] if row["request_index"] == 0 and row["role"] == "decode"]
    return reference, owned


@pytest.mark.parametrize("mode", SERVICE_CONTROLS["timing"])
@pytest.mark.parametrize("handoff", SERVICE_CONTROLS["handoff_ps"])
@pytest.mark.parametrize("duration", SERVICE_CONTROLS["decode_service_ps"])
@pytest.mark.parametrize("admission", SERVICE_CONTROLS["admission_ps"])
def test_compatibility_service_vectors_separate_waiting_from_service(mode, handoff, duration, admission):
    reference, owned = service_fixture(mode, handoff, duration, admission)
    expected = expected_times(reference["requests"][0], admission)
    check_service_vector(owned, expected, reference, request_index=0, role="decode",
                         engine="simllm-decode-0", admission=admission, label="fixture", evidence=Evidence([]))
    assert [row["result"]["step_latency_ps"] for row in owned] == [duration] * 4
    gaps = [right["result"]["completed_at_ps"] - left["result"]["completed_at_ps"]
            for left, right in pairwise(owned)]
    if mode == "serialized":
        assert 2 * duration in gaps
    else:
        assert gaps == [duration] * 3
        check_service_vector(owned, expected, None, request_index=0, role="decode",
                             engine="simllm-decode-0", admission=admission, label="shared", evidence=Evidence([]))
    if handoff == 0 and mode == "serialized":
        assert reference["makespan_ps"] == 2 * 13 + 8 * duration


@pytest.mark.parametrize("mutation", SERVICE_CONTROLS["fatal_mutations"])
def test_service_vector_corruptions_remain_fatal(mutation):
    reference, owned = service_fixture("serialized", 0, 7, 101)
    expected = expected_times(reference["requests"][0], 101)
    if mutation == "missing":
        owned.pop()
    elif mutation == "duplicate":
        owned.append(deepcopy(owned[-1]))
    elif mutation == "wrong-engine":
        owned[1]["engine_id"] = "simllm-decode-1"
    elif mutation == "early-start":
        owned[1]["record"]["virtual_time_ps"] -= 1
    elif mutation == "changed-finish":
        owned[1]["result"]["completed_at_ps"] += 1
    else:
        owned[1]["record"]["virtual_time_ps"] = owned[0]["result"]["completed_at_ps"]
        owned[1]["result"]["step_latency_ps"] *= 2
    with pytest.raises(GuardFailure, match="native-step"):
        check_service_vector(owned, expected, reference, request_index=0, role="decode",
                             engine="simllm-decode-0", admission=101, label="fixture", evidence=Evidence([]))


def test_service_vector_rejects_reference_engine_misbinding():
    reference, owned = service_fixture("serialized", 0, 7, 0)
    next(row for row in reference["service_steps"] if row["role"] == "decode")["engine_id"] = "foreign"
    with pytest.raises(GuardFailure, match="reference-engine"):
        check_service_vector(owned, {}, reference, request_index=0, role="decode",
                             engine="simllm-decode-0", admission=0, label="fixture", evidence=Evidence([]))


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
