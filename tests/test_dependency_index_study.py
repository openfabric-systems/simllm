"""Frozen lookup and source-compatibility controls, separate from study scores."""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.dependency_index_v1.checks import (
    check_graphs,
    check_sinks,
    conversion_calls,
    formula,
    read,
)
from examples.dependency_index_v1.run_study import HERE, receipts, reread
from examples.dependency_index_v1.worker import (
    encoded,
    graph_case,
    input_value,
    key_control,
    profile_rows,
    projection_value,
    run_graph,
    run_sink,
    sha,
    write,
)
from examples.pd_session_target_scale_v1.checks import Evidence, GuardFailure
from simllm.core.execution_io import effective_dependency_edges
from simllm.traffic import project_execution_graph_goal, verify_execution_goal_projection

FROZEN = json.loads((HERE / "expectations.json").read_bytes())
VERIFIER = HERE.parents[1] / "simllm/traffic/execution_goal.py"


@pytest.mark.parametrize("spec", FROZEN["graph_cases"], ids=lambda row: row["id"])
def test_frozen_graph_geometry_has_the_declared_occurrence_counts(spec):
    graph = graph_case(spec["depth"], spec["width"])
    projection = project_execution_graph_goal(graph)
    expected = formula(spec["depth"], spec["width"])
    assert len(effective_dependency_edges(graph)) == expected["edges"]
    assert len(projection.artifacts) == expected["artifacts"]
    assert len(projection.boundaries) == expected["boundaries"]
    assert len(projection.serialized_edges) == expected["serialized"]


@pytest.mark.parametrize("spec", [FROZEN["graph_cases"][0], FROZEN["graph_cases"][-1]], ids=lambda row: row["id"])
def test_actual_profile_and_distinct_rejections_match_the_frozen_contract(tmp_path, spec):
    row = run_graph(spec, FROZEN, tmp_path)
    count = conversion_calls(row["profiles"], VERIFIER, Evidence([]), "profile")
    assert count == formula(spec["depth"], spec["width"])["after"]
    assert profile_rows(tmp_path / (spec["id"] + ".pstats")) == row["profiles"]
    assert row["before"] == row["after"]
    assert row["profile_hook_restored"] is True
    controls = row["rejections"]
    assert len({sha(encoded(item["before"])) for item in controls}) == 14
    assert all(item["before"] == item["after"] and item["exception"] for item in controls)
    errors = {item["name"]: item["exception"] for item in controls}
    assert errors["invalid-type"]["type"] == "TypeError"
    assert errors["boundary-index-before-invalid-type"] == {
        "type": "ValueError", "message": "artifact boundary indexes do not match graph operations"}
    assert errors["edge-count-before-canonical-text"]["message"].startswith("GOAL projection edge mismatch:")


@pytest.mark.parametrize("name", FROZEN["valid_key_controls"])
def test_shared_endpoints_keep_distinct_origin_and_rank_keys(name):
    graph = key_control(name)
    projection = project_execution_graph_goal(graph)
    before = projection_value(projection)
    verify_execution_goal_projection(graph, projection)
    assert projection_value(projection) == before
    assert len({(edge.predecessor_id, edge.operation_id) for edge in effective_dependency_edges(graph)}) == 1
    if name.endswith("origin"):
        assert len(projection.serialized_edges) == 2
        assert len({edge.origin for edge in projection.serialized_edges}) == 2
    else:
        assert [edge.participant_rank for edge in projection.serialized_edges] == list(range(4))


def test_actual_granite_sink_job_and_corrupted_result(tmp_path):
    spec = FROZEN["sink_cases"][0]
    row = run_sink(spec, FROZEN, tmp_path)
    frozen = {**FROZEN, "sink_cases": [spec]}
    data = {"arm": "after", "sinks": [row]}
    check_sinks(data, frozen, Evidence([]))
    corrupted = deepcopy(data)
    corrupted["sinks"][0]["steps"][0]["result"]["step_latency_ps"] += 1
    with pytest.raises(GuardFailure, match="known-service"):
        check_sinks(corrupted, frozen, Evidence([]))


@pytest.mark.parametrize("raw", [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}', b'{"x":1}\n', b'{ "x":1}'])
def test_reader_rejects_ambiguous_or_changed_writer_bytes(tmp_path, raw):
    path = tmp_path / "bad.json"
    path.write_bytes(raw)
    with pytest.raises((GuardFailure, ValueError)):
        read(path)


def test_raw_receipts_lock_byte_content_and_file_domain(tmp_path):
    path = tmp_path / "value.json"
    write(path, {"x": 1})
    first = receipts(tmp_path)
    reread(tmp_path, first, Evidence([]), "unchanged")
    write(path, {"x": 2})
    with pytest.raises(GuardFailure):
        reread(tmp_path, first, Evidence([]), "changed")
    write(path, {"x": 1})
    (tmp_path / "extra").write_bytes(b"unreceipted")
    with pytest.raises(GuardFailure):
        reread(tmp_path, first, Evidence([]), "added")


def test_binary_profile_identity_cannot_be_replaced_with_a_count():
    rows = [{"function": [str(Path("foreign.py")), 1, "_goal_edge"], "calls": 1, "primitive_calls": 1}]
    with pytest.raises(GuardFailure, match="source-origin"):
        conversion_calls(rows, VERIFIER, Evidence([]), "profile")


def test_profile_function_line_must_match_the_source_definition():
    rows = [{"function": [str(VERIFIER), 1, "_goal_edge"], "calls": 1, "primitive_calls": 1}]
    with pytest.raises(GuardFailure, match="source-origin"):
        conversion_calls(rows, VERIFIER, Evidence([]), "profile")


@pytest.mark.parametrize("mutation", [None, "release", "same-rejections", "control-release"])
def test_full_frozen_input_and_each_declared_corruption_are_bound(tmp_path, mutation):
    spec = FROZEN["graph_cases"][0]
    row = run_graph(spec, FROZEN, tmp_path)
    controls = []
    for name in FROZEN["valid_key_controls"]:
        graph = key_control(name)
        value = input_value(graph, project_execution_graph_goal(graph))
        controls.append({"id": name, "before": value, "after": deepcopy(value), "accepted": True})
    data = {"arm": "after", "graphs": [row], "controls": controls}
    if mutation == "release":
        row["before"]["graph"]["released_at_ps"] = row["after"]["graph"]["released_at_ps"] = 1
    elif mutation == "same-rejections":
        first = row["rejections"][0]
        row["rejections"] = [{**deepcopy(first), "name": name} for name in FROZEN["projection_corruptions"]]
    elif mutation == "control-release":
        controls[0]["before"]["graph"]["released_at_ps"] = controls[0]["after"]["graph"]["released_at_ps"] = 1
    frozen = {**FROZEN, "graph_cases": [spec]}
    if mutation is None:
        check_graphs(data, frozen, HERE.parents[1], tmp_path, Evidence([]))
    else:
        with pytest.raises(GuardFailure, match="frozen-input|declared-delta"):
            check_graphs(data, frozen, HERE.parents[1], tmp_path, Evidence([]))


def test_failed_worker_retains_first_receipts_and_original_failure(tmp_path, monkeypatch):
    from examples.dependency_index_v1 import run_study

    failure = GuardFailure("worker exited")

    def fail(arm, root, args, frozen, monitors):
        path = args.output_root / arm
        path.mkdir()
        (path / "process.log").write_bytes(b"failure retained")
        write(path / "process.json", {"exit_code": 1})
        raise failure

    monkeypatch.setattr(run_study, "launch", fail)
    raw, root_receipts = {}, {}
    with pytest.raises(GuardFailure) as caught:
        run_study.capture_worker("before", HERE.parents[1], SimpleNamespace(output_root=tmp_path), FROZEN, {}, raw, root_receipts)
    assert caught.value is failure
    assert read(tmp_path / "before-raw-receipts.json") == raw["before"]
    assert root_receipts["before-raw-receipts.json"]["sha256"] == sha((tmp_path / "before-raw-receipts.json").read_bytes())
    (tmp_path / "before/process.log").write_bytes(b"changed after failure")
    with pytest.raises(GuardFailure):
        reread(tmp_path / "before", raw["before"], Evidence([]), "changed")
