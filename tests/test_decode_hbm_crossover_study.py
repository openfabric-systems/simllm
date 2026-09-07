"""Regression and mutation checks for the frozen decode crossover evidence."""

from __future__ import annotations

import copy
import json
import subprocess
from fractions import Fraction

import pytest

from examples.decode_hbm_crossover_v1 import run_study as study


@pytest.fixture(scope="module")
def evidence():
    return study.run_study()


@pytest.fixture(scope="module")
def frozen():
    return json.loads((study.STUDY / "expectations.json").read_text())


def evaluate(rows, frozen):
    return study.audit(rows, frozen, study.read_csv(study.STUDY / "expected_cells.csv"))[0]


def select(rows, model="dense7b-tp1", device="h200", batch=1, context=1, scale="1"):
    return next(r for r in rows if study.identity(r) == (model, context, device, scale, batch))


def test_full_frozen_grid_matches_without_rounding_band_on_api(evidence):
    rows, summary, relations = evidence
    assert len(rows) == 781
    assert summary["configurations"] == {"sweep": 675, "knee": 105, "external": 1}
    assert summary["verdict"] == "PASS"
    assert summary["fatal_findings"] == []
    assert summary["exact_oracle"]["matched"] == 781
    assert summary["exact_oracle"]["max_tpot_residual_ps"] == 0
    assert len(summary["behavioral_families"]) == 6
    assert all(r["passed"] for r in relations)
    assert summary["pricing_subprocess_attempts"] == 0
    assert summary["tasks_closed"] == []


def test_retained_results_replay_exactly(evidence, tmp_path):
    rows, summary, relations = evidence
    study.write_results(tmp_path, rows, summary, relations)
    assert study.read_csv(tmp_path / "results.csv") == rows
    assert json.loads((tmp_path / "results.json").read_text()) == summary
    assert json.loads((study.STUDY / "results.json").read_text()) == summary
    assert study.read_csv(study.STUDY / "results.csv") == rows


def test_nonzero_cache_refutes_exact_flatness(evidence):
    rows = evidence[0]
    first = select(rows)
    second = select(rows, batch=2)
    assert first["bound"] == second["bound"] == "memory"
    assert second["tpot_ps"] > first["tpot_ps"]
    assert first["weight_bytes"] == second["weight_bytes"]
    assert first["kv_bytes"] * 2 == second["kv_bytes"]
    assert Fraction(second["cache_to_weight_ratio"]) < Fraction(1, 100)


def test_long_context_can_remove_crossover(evidence):
    for device in ("a100", "h100", "h200", "b100", "b200"):
        row = select(evidence[0], device=device, batch=256, context=2048)
        assert row["crossover_batch"] == "none"
        assert row["relative_to_crossover"] == "no-finite-crossover"
        assert row["bound"] == "memory"


def test_tensor_parallel_output_count_and_deployment_projection(evidence):
    row = select(evidence[0], model="dense70b-tp8", batch=1)
    speed = Fraction(row["request_tokens_per_second"])
    assert Fraction(row["throughput_tokens_per_second_per_gpu"]) == speed / 8
    assert row["deployment_step_ps"] == row["tpot_ps"]
    assert row["deployment_guards_ok"]
    single = select(evidence[0])
    assert single["deployment_step_ps"] == "unavailable"


def test_ep72_uses_declared_compressed_cache_and_local_batch(evidence):
    row = select(evidence[0], model="deepseek-v3-ep72", batch=32, context=2000)
    assert row["flops"] == 3_594_330_365_632
    assert row["weight_bytes"] + row["kv_bytes"] == 31_944_051_040
    assert Fraction(row["throughput_tokens_per_second_per_gpu"]) == Fraction(
        32 * study.PS, row["tpot_ps"])
    assert row["deployment_step_ps"] == "unavailable"


def test_external_row_remains_unscored_and_unfitted(evidence):
    comparison = evidence[1]["external_comparison_unscored"]
    assert comparison["roofline_tpot_ps"] == 5_379_515_733
    assert comparison["external_tpot_ps"] == 9_179_000_000
    assert comparison["roofline_below_external"]
    external = next(r for r in evidence[0] if r["scope"] == "external")
    assert external["peak_flops"] == 1_979_000_000_000_000
    assert external["bound"] == "memory"


@pytest.mark.parametrize("mutation", ["drop", "duplicate", "work", "normalization", "stamp"])
def test_fatal_mutations_void_scores(evidence, frozen, mutation):
    rows = copy.deepcopy(evidence[0])
    if mutation == "drop":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif mutation == "work":
        rows[0]["flops"] += 1
    elif mutation == "normalization":
        rows[0]["throughput_tokens_per_second_per_gpu"] = "1"
    else:
        rows[0]["deployment_guards_ok"] = False
    summary = evaluate(rows, frozen)
    assert summary["verdict"] == "VOID"
    assert summary["fatal_findings"]
    assert summary["behavioral_families"] is None
    assert summary["exact_oracle"]["matched"] is None


def test_one_picosecond_mismatch_is_not_hidden_by_physical_rounding_band(evidence, frozen):
    rows = copy.deepcopy(evidence[0])
    row = select(rows)
    row["tpot_ps"] += 1
    row["request_tokens_per_second"] = str(Fraction(study.PS, row["tpot_ps"]))
    row["throughput_tokens_per_second_per_gpu"] = row["request_tokens_per_second"]
    summary = evaluate(rows, frozen)
    assert summary["nonvoid"]
    assert summary["verdict"] == "FAIL"
    assert summary["exact_oracle"]["matched"] == 780
    assert summary["exact_oracle"]["max_tpot_residual_ps"] == 1


def test_physics_violation_is_void(evidence, frozen):
    rows = copy.deepcopy(evidence[0])
    rows[0]["tpot_ps"] = 1
    summary = evaluate(rows, frozen)
    assert summary["verdict"] == "VOID"
    assert any("physical bounds" in f for f in summary["fatal_findings"])


def test_process_creation_is_blocked_and_guard_restores(monkeypatch):
    original = subprocess.Popen

    def process_attempt(*args):
        subprocess.run(["not-an-actual-executable"], check=True)

    monkeypatch.setattr(study, "price_cell", process_attempt)
    rows, summary, _ = study.run_study()
    assert rows == []
    assert summary["verdict"] == "VOID"
    assert summary["pricing_subprocess_attempts"] == 1
    assert subprocess.Popen is original


def test_digest_failure_prevents_pricing(monkeypatch):
    monkeypatch.setattr(study, "frozen_findings", lambda frozen: ["changed frozen input"])
    monkeypatch.setattr(study, "price_cell", lambda *args: pytest.fail("pricing was attempted"))
    rows, summary, _ = study.run_study()
    assert rows == []
    assert summary["verdict"] == "VOID"
    assert "changed frozen input" in summary["fatal_findings"]


def test_failure_retains_completed_cells(monkeypatch):
    original = study.price_cell
    calls = 0

    def stop_after_two(*args):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise ValueError("injected pricing defect")
        return original(*args)

    monkeypatch.setattr(study, "price_cell", stop_after_two)
    rows, summary, _ = study.run_study()
    assert len(rows) == 2
    assert summary["verdict"] == "VOID"
    assert any("injected pricing defect" in f for f in summary["fatal_findings"])
