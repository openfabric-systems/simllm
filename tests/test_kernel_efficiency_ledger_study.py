from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "examples/kernel_efficiency_ledger_v1"
SPEC = importlib.util.spec_from_file_location("kernel_efficiency_study", HERE / "run_study.py")
assert SPEC is not None and SPEC.loader is not None
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


@pytest.fixture(scope="module")
def evidence():
    return study.evaluate(ROOT)


def test_retained_analysis_reproduces_committed_artifacts(evidence, tmp_path):
    rows, summary = evidence
    study.write_artifacts(rows, summary, tmp_path, plot=False)
    for name in ("results.csv", "results.json"):
        assert (tmp_path / name).read_bytes() == (HERE / name).read_bytes()
    assert study.evaluate(ROOT) == evidence
    committed = json.loads((HERE / "results.json").read_text())
    assert committed["ledger_sha256"] == hashlib.sha256((HERE / "results.csv").read_bytes()).hexdigest()


def test_population_preserves_arm_identity_void_state_and_missing_work(evidence):
    rows, summary = evidence
    assert len({r["cell_id"] for r in rows}) == 385
    constants = [r for r in rows if r["cell_id"].startswith("constants/")]
    assert len(constants) == 264
    assert all(r["source_void"] and r["verdict_void"] for r in constants)
    assert all(r["source_guard_reasons"] == ["G9R", "G10", "G11R"] for r in constants)
    assert sum(r["host_issue_bound"] for r in constants) == 1
    missing = [r for r in rows if r["work_state"] == "missing"]
    assert len(missing) == 5
    assert all(r["views"] == {"measured": None, "datasheet": None} for r in missing)
    assert all(r["bytes_declared"] is None and r["flops_declared"] is None for r in missing)
    assert all(r["source_state"] == "nonvoid" and r["verdict"] == "unavailable-work" for r in missing)
    assert all(r["distribution_verdict"] == "insufficient-replays" for r in missing)
    assert summary["behavioral_score"] is None
    assert summary["closure"]["keeps_open"] == ["COMP-45", "COMP-46"]
    assert summary["closure"]["closes"] == []


def test_hand_computed_work_and_units():
    # A tiny matrix multiply reads two operands and writes the output, all BF16.
    raw = {"family": "gemm", "m": 2, "n": 3, "k": 4}
    assert study.declared_work(raw) == (52, 48)
    decode = {"family": "attn_decode", "m": 4, "n": 16, "k": 64, "length": 512}
    assert study.declared_work(decode) == (4_194_304, 8_388_608)
    prefill = {"family": "attn_prefill_granite", "m": 128, "n": 16, "k": 64}
    assert study.declared_work(prefill) == (2_097_152, 67_108_864)
    result = study.roofline(52, 48, 2e-6, 1e9, 1e12)
    assert result["bytes_s"] == 26e6
    assert result["flops_s"] == 24e6
    assert result["ridge_class"] == "HBM-bound"
    assert result["fraction"] == pytest.approx(0.026)


def test_ridge_tie_and_memory_peak_invariance():
    assert study.roofline(8, 16, 1, 4, 8)["ridge_class"] == "compute-bound"
    fractions = [study.roofline(128, 32, 2, 64, peak)["fraction"] for peak in (32, 64, 128)]
    assert fractions == [1, 1, 1]
    assert study.roofline(128, 32, 2, 64, 8)["fraction"] == 2


@pytest.mark.parametrize("parameter,value", [("time", 0), ("time", float("nan")),
                                              ("bandwidth", -1), ("peak", float("inf"))])
def test_bad_quantities_void_analysis(parameter, value):
    args = {"work_bytes": 10, "flops": 1, "time": 1, "bandwidth": 100, "peak": 100}
    args[parameter] = value
    with pytest.raises(ValueError, match="VOID analysis"):
        study.roofline(**args)


def test_shape_disagreement_is_fatal():
    raw = {"family": "gemm", "m": 2, "n": 3, "k": 4, "total_bytes": 53,
           "flops": 48, "distinct_bytes": 52, "constant_s": 1}
    with pytest.raises(ValueError, match="work declaration mismatch"):
        study.make_cell("bad", raw, device="a100", source="synthetic", arm="test",
                        state="nonvoid", evidence="fixture", sm_low=1410, sm_high=1410,
                        memory_mhz=1593)


def test_source_edit_is_rejected_before_analysis(tmp_path):
    path = tmp_path / "retained.json"
    path.write_text("changed")
    freeze = {"source_sha256": {"retained.json": hashlib.sha256(b"original").hexdigest()}}
    with pytest.raises(ValueError, match="source digest mismatch"):
        study.verify_sources(tmp_path, freeze)


def test_launch_override_is_not_inferred_from_low_efficiency(evidence):
    rows, summary = evidence
    env = summary["envelopes"]["a100"]
    source = next(r for r in rows if r["cell_id"] == "constants/boosted/gemm_G4_m1")
    slow = deepcopy(source)
    slow["time_s"] *= 100
    result = study.analyze_cell(slow, env)
    assert result["class"] == "HBM-bound"
    assert result["views"]["measured"]["diagnostic_flags"]["0.5"]
    forced = study.analyze_cell({**slow, "host_issue_bound": True}, env)
    assert forced["class"] == "launch-bound"
    assert not any(forced["views"]["measured"]["diagnostic_flags"].values())


def test_envelope_breach_cannot_be_rehabilitated_by_datasheet(evidence):
    rows, summary = evidence
    original = next(r for r in rows if r["cell_id"] == "envelope/a100/hbm_copy_4096mib")
    env = summary["envelopes"]["a100"]
    changed = {**original, "time_s": original["bytes_declared"] / (env["measured_hbm_bytes_s"] * 1.01)}
    result = study.analyze_cell(changed, env)
    assert result["views"]["datasheet"]["fraction"] < 1
    assert result["cell_void"] and result["verdict_void"]
    assert result["verdict"] == "void-with-findings"
    assert "nominal-byte-rate-above-measured-envelope" in result["cell_guard_findings"]


def test_sweep_nests_candidates_and_preserves_each_threshold_order(evidence):
    rows, summary = evidence
    for source in study.SOURCES:
        configs = [c for c in summary["configurations"] if c["envelope"] == source]
        assert len({c["rank_order_sha256"] for c in configs}) == 1
        flags = [set(c["diagnostic_candidates"]) for c in configs]
        assert flags[0] <= flags[1] <= flags[2]
        invalid = {r["cell_id"] for r in rows if r.get("verdict_void")}
        assert all(not invalid.intersection(c["accepted_source_candidates"]) for c in configs)
    comparison = summary["rank_comparison"]
    assert comparison["pairwise_inversions"] > 0
    assert comparison["spearman"] >= 0.98
    assert summary["structural_guards"]["scored"] is False
    assert all(r["accepted_score"] is None for r in summary["relations"].values())


def test_ledger_is_a_ranked_table_with_explicit_missing_fields(evidence):
    rows, _ = evidence
    table = list(csv.DictReader(io.StringIO(study.ledger_csv(rows))))
    assert [int(r["measured_rank"]) for r in table[:380]] == list(range(1, 381))
    assert all(r["measured_rank"] == r["measured_bytes_s"] == "" for r in table[380:])
    assert all(r["verdict"] == "unavailable-work" for r in table[380:])
    assert any(r["flop_accounting"] == "source-zero-scalar-work-uninventoried" for r in table)


def test_clock_and_envelope_sources_are_not_silently_substituted(evidence):
    rows, summary = evidence
    base_boosted = next(r for r in rows if r["cell_id"] == "constants/base/attn_decode_b256_l8192")
    assert base_boosted["compute_peak_flops_s"] == 311_869_440_000_000
    env = summary["envelopes"]
    assert env["a100"]["measured_anchor"] == "hbm_write_4096mib"
    assert env["a100"]["measured_hbm_bytes_s"] < 1.8e12
    assert env["gh200"]["model_comparison_exact_device"] is False
    assert env["gh200"]["model_hbm_bytes_s"] != env["gh200"]["datasheet_hbm_bytes_s"]
