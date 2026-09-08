"""Guards for the fixed queue-attribution experiment and its evidence classes."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from itertools import product
from types import SimpleNamespace

import pytest

from examples.pp_rail_contention_v2 import run_study as study


def test_matrix_has_exactly_the_frozen_populations():
    cells = study.cells()
    physical = [cell for cell in cells if cell.population == "physical"]
    assert len(cells) == len({cell.name for cell in cells}) == 72
    assert len(physical) == 24
    assert sum(cell.population == "legacy" for cell in cells) == 12
    assert sum(cell.population == "ideal" for cell in cells) == 36
    assert sum(2 if cell.population == "physical" else 1 for cell in cells) == 96
    assert sum(cell.window_ps == study.LOW_WINDOW_PS for cell in physical) == 4
    formerly_void = {(cell.variant, cell.width) for cell in physical
                     if cell.spines == 2 and cell.ep_width == 32}
    assert formerly_void == set(product(("rail", "node-local"), (2, 4, 8)))
    assert all(cell.spines == 2 for cell in physical if cell.width != 2)


@pytest.mark.parametrize("cell", study.cells(), ids=lambda cell: cell.name)
def test_fixed_policy_is_independent_of_load_and_parallel_width(cell, tmp_path):
    cfg = study.configuration(cell, tmp_path, tmp_path, False)
    assert cfg.initial_window_bytes is cfg.initial_window_fan_in is None
    assert cfg.retry_probe_windows == 4
    if cell.population == "physical":
        assert cfg.control_recovery == "headroom" and cfg.data_recovery == "exponential"
        assert int(cfg.extra_flags["-rnic_cn_ring_window_ps"]) == cell.window_ps
        traced = study.configuration(cell, tmp_path, tmp_path, True)
        flags = dict(traced.extra_flags)
        assert flags.pop("-rnic_cn_trace_dir") == str(tmp_path / "trace")
        assert flags == cfg.extra_flags
    else:
        assert cfg.control_recovery == cfg.data_recovery == "none"
        assert cfg.extra_flags == ({"-rnic_cn_prbs_seed": "1"} if cell.profile == "rnic-cn" else {})
        with pytest.raises(ValueError, match="no trace arm"):
            study.configuration(cell, tmp_path, tmp_path, True)


@pytest.mark.parametrize("variant,budget", [("rail", 16_742_451_200), ("node-local", 11_457_628_160)])
@pytest.mark.parametrize("width", (2, 4, 8))
def test_formerly_void_operational_budget_is_not_enlarged(variant, budget, width):
    cell = study.Cell("physical", variant, 2, width, 32, study.HIGH_WINDOW_PS)
    bounds = study.physical_bounds(cell)
    assert bounds["engineering_budget_ps"] == budget
    assert bounds["budget_queue_allowance_ps"] == 70_997_760
    assert bounds["unconditional_fct_ceiling_ps"] is None
    assert study.physical_bounds(replace(cell, window_ps=study.LOW_WINDOW_PS))["engineering_budget_ps"] == budget


@pytest.mark.parametrize("variant,cut", [("rail", 176_160_768), ("node-local", 117_440_512)])
def test_physics_is_per_direction_and_per_egress(variant, cut):
    cell = study.Cell("physical", variant, 2, 2, 32, study.HIGH_WINDOW_PS)
    two = study.physical_bounds(cell)
    eight = study.physical_bounds(replace(cell, spines=8))
    assert two["ep_busiest_leaf_uplink_bytes"] == cut
    assert two["ep_cut_drain_floor_ps"] == 4 * eight["ep_cut_drain_floor_ps"]
    assert two["ep_endpoint_serialization_floor_ps"] == 587_202_560
    assert two["data_arrival_floor_ps"] == 65_536 * 20 + (2 if variant == "rail" else 4) * 1_000_000
    assert two["pp_hop_cut_bounds"][0]["guaranteed_ep_bytes_ahead"] == 0


def test_published_reference_locks_cover_the_complete_controls():
    locks = study.reference_locks()
    assert len(locks) == 36
    for cell in study.cells():
        lock = locks[cell.reference_name.replace("rnic-nn", "rnic-cn")]
        assert len(lock["goal_sha256"]) == len(lock["ideal_completion_sha256"]) == 64
        if cell.population == "legacy":
            assert len(lock["reference_completion_sha256"]) == 64


def test_text_line_endings_are_the_only_reference_digest_normalization(tmp_path):
    path = tmp_path / "input.goal"
    expected = hashlib.sha256(b"num_ranks 64\n").hexdigest()
    path.write_bytes(b"num_ranks 64\r\n")
    study.require_digest(path, [expected])
    path.write_bytes(b"num_ranks 63\n")
    with pytest.raises(ValueError, match="reference bytes changed"):
        study.require_digest(path, [expected])


@pytest.mark.parametrize("lines", [[], ["x=1 x=1"], ["x=1", "x=2"]])
def test_manifest_identity_requires_exactly_one_authority(lines):
    with pytest.raises(ValueError, match="missing or repeated"):
        study.manifest_value(lines, "x")


def test_native_failure_cannot_publish_partial_request_metrics(tmp_path):
    cell = study.cells()[0]
    row = study.analyze_execution(cell, tmp_path, tmp_path, tmp_path,
                                  {"returncode": 124}, "partial flow output", False)
    assert row["cell_status"] == "void"
    assert row["request_ttft_ps"] is row["pp_hop_maximum_ps"] is None
    assert row["fatal_findings"] == ["native execution exited 124"]


def void_rows():
    return [{"cell": cell.name, **study.asdict(cell), "traced": traced,
             "cell_status": "void", "fatal_findings": ["synthetic interrupted execution"],
             "request_ttft_ps": None, "pp_hop_maximum_ps": None}
            for cell in study.cells() for traced in
            ((False, True) if cell.population == "physical" else (False,))]


@pytest.mark.parametrize("mutation", ("missing", "duplicate", "extra"))
def test_population_loss_is_fatal_before_any_relation_denominator(mutation):
    rows = void_rows()
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(rows[0])
    else:
        rows[-1] = {**rows[-1], "cell": "unregistered-cell"}
    with pytest.raises(ValueError, match="execution population"):
        study.summarize(rows, {})


def test_void_aggregate_has_no_behavioral_score_or_task_closure():
    result = study.summarize(void_rows(), {})
    assert result["verdict"] == "void" and result["traf88_acceptance"] == "not-met"
    assert all(row["matched"] is None for row in result["behavioral_relations"])
    assert all(row["status"] == "void" for row in result["trace_identity_oracles"])
    assert len(result["behavioral_families"]) == 3
    assert len(result["diagnostics"]) == 2


def test_raw_execution_record_precedes_auditor_failure(monkeypatch, tmp_path):
    cell = study.cells()[0]
    (tmp_path / cell.name).mkdir()
    monkeypatch.setattr(study, "run_owned_process", lambda *args, **kwargs:
                        type("Result", (), {"stdout": "retained native evidence", "stderr": "", "returncode": 0})())

    def fail_after_raw(cell, inputs, out, *args):
        execution = json.loads((out / "execution.json").read_bytes())
        assert execution["run_log_sha256"] == study.digest(out / "run.log")
        assert "retained native evidence" in (out / "run.log").read_text()
        raise ValueError("synthetic identity violation")

    monkeypatch.setattr(study, "analyze_execution", fail_after_raw)
    row = study.execute(cell, True, tmp_path, tmp_path, tmp_path / "binary", 1)
    assert row["cell_status"] == "void" and row["request_ttft_ps"] is None
    assert row["fatal_findings"] == ["ValueError: synthetic identity violation"]


def valid_rows(*, positive_penalty):
    rows = []
    for row in void_rows():
        load_penalty = 1 if positive_penalty and row["variant"] == "node-local" and row["ep_width"] == 32 else 0
        holding = row["window_ps"] or 0
        row.update(cell_status="valid", fatal_findings=[], completion_sha256="a" * 64,
                   request_ttft_ps=100_000_000 + holding + load_penalty,
                   pp_hop_maximum_ps=10_000_000 + holding + load_penalty,
                   physical_quiescence=True, physical_quiescence_time_ps=200_000_000,
                   complete_flow_phase_ps=100_000_000, ep_phase_ps=90_000_000,
                   job_completion_ps=100_000_000, request_attribution={"kernel_ps": 2_000_000},
                   pp_fct_samples_ps=[10_000_000 + holding + load_penalty],
                   bounds={"ep_phase_floor_ps": 1_000_000},
                   trace_audit={"queue_work": {"ep_data_service_ahead_ps": row["ep_width"]},
                                "pp_flows": [{"source_rank": 0, "destination_rank": 8,
                                              "packets": [{"packet_index": 0, "attempt": 0,
                                                           "arrival_ps": 1_000_000,
                                                           "logical_release_ps": 1_000_000 + holding,
                                                           "ring_holding_ps": holding}]}]})
        rows.append(row)
    return rows


def test_positive_queue_service_with_flat_request_is_a_valid_refutation():
    result = study.summarize(valid_rows(positive_penalty=False), {})
    assert result["verdict"] == "valid-refutation" and result["traf88_acceptance"] == "not-met"
    assert result["refuted_families"] == ["R3-request-penalty"]
    assert all(row["status"] == "holds" for row in result["behavioral_relations"]
               if row["family"] == "R2-expert-service-ahead")
    assert len(result["behavioral_relations"]) == 10
    assert len(result["trace_identity_oracles"]) == 24
    assert len(result["compatibility_controls"]) == 48


def test_closure_requires_positive_live_penalty_and_all_identity_guards():
    rows = valid_rows(positive_penalty=True)
    result = study.summarize(rows, {})
    assert result["verdict"] == "valid-positive-penalty" and result["traf88_acceptance"] == "met"
    corrupted = deepcopy(rows)
    corrupted[1]["physical_quiescence_time_ps"] += 1
    result = study.summarize(corrupted, {})
    assert result["verdict"] == "void" and result["traf88_acceptance"] == "not-met"
    assert result["physical_configurations"][0]["request_ttft_ps"] is None
    assert result["trace_identity_oracles"][0]["mismatched_fields"] == ["physical_quiescence_time_ps"]


@pytest.mark.parametrize("shifted_hop", (0, 1))
def test_exact_goal_timing_rejects_a_shift_that_preserves_fct(shifted_hop):
    pp = [SimpleNamespace(start_time_ps=1_001_000, completion_time_ps=11_088_200),
          SimpleNamespace(start_time_ps=12_090_200, completion_time_ps=22_259_400)]
    findings = []
    assert study.exact_goal_timing(pp, 3, findings) == 23_260_000
    assert findings == []
    pp[shifted_hop].start_time_ps += 1000
    pp[shifted_hop].completion_time_ps += 1000
    study.exact_goal_timing(pp, 3, findings)
    assert findings and any("release differs" in finding for finding in findings)


@pytest.mark.parametrize("mutation", ("boundary", "schema", "status"))
def test_trace_publication_must_bind_the_runtime_boundary(mutation):
    audit = {"schema": "pp-queue-audit-v1", "status": "valid", "physical_quiescence_time_ps": 100}
    manifest = ["[RNIC manifest] physical_quiescence=verified physical_quiescence_time_ps=100"]
    study.trace_boundary_guard(audit, manifest)
    if mutation == "boundary":
        audit["physical_quiescence_time_ps"] += 1
    else:
        audit[mutation] = "unknown"
    with pytest.raises(ValueError):
        study.trace_boundary_guard(audit, manifest)
