"""Evidence integrity checks for the coarse receiver consumer study."""

from __future__ import annotations

import copy
import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "examples/receiver_ingress_v1/run_study.py"
SPEC = importlib.util.spec_from_file_location("receiver_ingress_study", SCRIPT)
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def combine_case():
    return next(c for c in study.cases() if c["family"] == "main" and
                c["pattern"] == "combine" and c["width"] == 8 and
                c["payload_bytes"] == 4096 and c["rate_gbps"] == 400)


@pytest.fixture(scope="module")
def observed(tmp_path_factory):
    root = tmp_path_factory.mktemp("receiver-study")
    case = combine_case()
    rows = {}
    for mode in ("baseline", "disabled", "enabled"):
        directory = root / mode
        directory.mkdir()
        rows[mode] = study.observe(case, mode, directory)
    return case, root, rows


def test_frozen_matrix_has_exactly_102_distinct_workloads():
    cases = study.cases()
    assert len(cases) == len({c["name"] for c in cases}) == 102
    assert Counter(c["family"] for c in cases) == {
        "main": 72, "ring": 16, "sentinel": 2, "asymmetric": 4, "complete": 8}
    for case in cases:
        bounds = study.bounds(case)
        assert bounds["phase_floor_ps"] <= bounds["expected_enabled_ps"] <= bounds["fixture_ceiling_ps"]


def test_bounds_use_ring_chunks_and_keep_the_unphysical_baseline_explicit():
    sentinel = next(c for c in study.cases() if c["family"] == "sentinel" and c["rate_gbps"] == 400)
    assert study.bounds(sentinel)["phase_floor_ps"] == 120
    case = combine_case()
    bounds = study.bounds(case)
    assert bounds["phase_floor_ps"] == bounds["expected_enabled_ps"] == 655360
    assert bounds["expected_disabled_ps"] == 81920


def test_raw_observation_survives_a_later_oracle_failure(observed):
    case, root, rows = observed
    path = root / "enabled" / "observations.json"
    before = path.read_bytes()
    wrong = {**study.bounds(case), "expected_enabled_ps": 1}
    result = study.apply_guards(rows["enabled"], wrong, disabled=rows["disabled"])
    assert "exact fixture latency differs from its frozen relation" in result["fatal_findings"]
    assert path.read_bytes() == before
    assert json.loads(before)["ttft_ps"] == 655360


def test_current_default_and_explicit_false_keep_full_snapshot_bytes(observed):
    _, root, rows = observed
    assert rows["baseline"]["status"] == rows["disabled"]["status"] == "complete"
    assert (root / "baseline/snapshot.json").read_bytes() == (root / "disabled/snapshot.json").read_bytes()


def test_enabled_metric_chain_and_wait_reductions_are_distinct(observed):
    case, _, rows = observed
    result = study.apply_guards(rows["enabled"], study.bounds(case), disabled=rows["disabled"])
    assert result["fatal_findings"] == []
    assert result["ttft_ps"] == study.integral(result["tpot_ps"]) == 655360
    assert result["jct_ps"] == 1966080
    for step in result["steps"]:
        assert step["sum_receiver_wait_ps"] == 2293760
        assert step["critical_queue_ps"] == 573440
        assert step["critical_breakdown"]["service_ps"] == 81920
        assert step["latency_ps"] == 655360


@pytest.mark.parametrize("projection", ["source_port_reservations", "receiver_port_reservations"])
def test_missing_endpoint_reservation_is_fatal(observed, projection):
    case, _, rows = observed
    row = copy.deepcopy(rows["enabled"])
    row["endpoint_projections"][projection].pop()
    checked = study.apply_guards(row, study.bounds(case), disabled=rows["disabled"])
    assert any("port projection lost or duplicated" in f for f in checked["fatal_findings"])


def test_disabled_snapshot_mutation_is_not_absorbed(observed):
    case, _, rows = observed
    row = {**rows["disabled"], "snapshot_sha256": "0"*64}
    checked = study.apply_guards(row, study.bounds(case), baseline=rows["baseline"])
    assert "disabled snapshot differs from immutable baseline" in checked["fatal_findings"]


def test_changed_release_is_the_only_workload_graph_identity_exemption(observed):
    case, _, rows = observed
    assert rows["disabled"]["graph_sha256"][1] != rows["enabled"]["graph_sha256"][1]
    assert rows["disabled"]["workload_graph_sha256"] == rows["enabled"]["workload_graph_sha256"]
    assert rows["disabled"]["goal_sha256"] == rows["enabled"]["goal_sha256"]
    row = copy.deepcopy(rows["enabled"])
    row["workload_graph_sha256"][1] = "0"*64
    checked = study.apply_guards(row, study.bounds(case), disabled=rows["disabled"])
    assert "receiver selection changed the normalized workload graph" in checked["fatal_findings"]


def test_failed_execution_retains_null_metrics(tmp_path, monkeypatch):
    from simllm.core import CoarseDeviceRuntime

    def fail(self, *args, **kwargs):
        raise RuntimeError("injected runtime failure")

    monkeypatch.setattr(CoarseDeviceRuntime, "execute", fail)
    row = study.observe(combine_case(), "enabled", tmp_path)
    assert row["status"] == "failed"
    assert row["ttft_ps"] is row["tpot_ps"] is row["jct_ps"] is None
    assert json.loads((tmp_path / "observations.json").read_bytes()) == row


def test_immutable_records_reject_overwrite(tmp_path):
    path = tmp_path / "record.json"
    study.write_once(path, {"original": True})
    with pytest.raises(FileExistsError):
        study.write_once(path, {"original": False})
    assert json.loads(path.read_bytes()) == {"original": True}


def test_void_and_missing_matrix_are_not_behavioral_scores(observed):
    case, _, rows = observed
    row = study.apply_guards(rows["enabled"], study.bounds(case), disabled=rows["disabled"])
    result = study.summarize([row], [], ("enabled",))
    assert result["verdict"] == "void"
    assert result["fatal_findings"][-1]["finding"] == "matrix missing or duplicated"
    assert "behavioral_score" not in result


def test_zero_disabled_and_duplex_effects_are_not_scored(observed):
    _, _, rows = observed
    relations = study.behavioral_relations([rows["disabled"], rows["enabled"]])
    assert len(relations) == 1
    assert relations[0]["family"] == "receiver_latency_increase"
    assert relations[0]["within_band"]


def test_missing_baseline_never_silently_skips_disabled_identity(observed):
    case, _, rows = observed
    checked = study.apply_guards(rows["disabled"], study.bounds(case))
    assert "disabled configuration has no immutable baseline evidence" in checked["fatal_findings"]


@pytest.mark.parametrize("mutation", ["none", "missing", "duplicate", "unexpected",
                                      "changed-case", "changed-mode", "snapshot", "script"])
def test_baseline_loader_requires_exact_population_and_snapshots(
    tmp_path, monkeypatch, observed, mutation,
):
    case, root, rows = observed
    monkeypatch.setattr(study, "cases", lambda: [case])
    row = study.apply_guards(rows["baseline"], study.bounds(case))
    provenance = {"runtime_commit": study.BASELINE_COMMIT, "mode": "baseline",
                  "script_sha256": "script", "expectations_sha256": "expectations"}
    baseline = {"schema": "receiver-ingress-v1", "verdict": "valid", "executions": 1,
                "request_steps": 3, "cells": [copy.deepcopy(row)], "provenance": provenance}
    snapshot = tmp_path / case["name"] / "baseline/snapshot.json"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes((root / "baseline/snapshot.json").read_bytes())
    if mutation == "missing":
        baseline["cells"] = []
    elif mutation == "duplicate":
        baseline["cells"].append(copy.deepcopy(row))
    elif mutation == "unexpected":
        baseline["cells"][0]["case"]["name"] = "unregistered-case"
    elif mutation == "changed-case":
        baseline["cells"][0]["case"]["rate_gbps"] = 1
    elif mutation == "changed-mode":
        baseline["cells"][0]["mode"] = "disabled"
    elif mutation == "snapshot":
        snapshot.write_bytes(b"[]\n")
    elif mutation == "script":
        provenance["script_sha256"] = "changed"
    study.write_once(tmp_path / "results.json", baseline)
    expected = {"script_sha256": "script", "expectations_sha256": "expectations"}
    if mutation == "none":
        loaded, _ = study.load_baseline(tmp_path, expected)
        assert loaded == {case["name"]: row}
    else:
        with pytest.raises(ValueError, match="baseline"):
            study.load_baseline(tmp_path, expected)
