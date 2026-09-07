"""Offline live-metric and artifact checks for the BACK-68 experiments."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

from simllm.backends.htsim_rnic import FlowCompletion

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "aligned_baseline_study", ROOT / "examples/aligned_baseline_v1/run_study.py")
STUDY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = STUDY
SPEC.loader.exec_module(STUDY)


def test_frozen_matrix_uses_distinct_senders_and_exact_pair_bytes():
    cells = list(STUDY.cells())
    assert len(cells) == len({c.name for c in cells}) == 56
    for c in cells:
        assert len(c.sources) == len(set(c.sources)) == c.fan_in
        assert 0 not in c.sources
        assert STUDY.build_trace(c).render().count(": send ") == c.fan_in
        if c.kind == "pair":
            assert [m[3] for m in c.messages] == [c.payload, 4 * c.payload]
        if c.kind == "incast":
            assert {s // 8 for s in c.sources} == set(range(8))


def test_all_requested_fluid_pair_predictions_are_frozen_in_ps():
    expected = {(262144, 400): (12485760, 28214400),
                (262144, 200): (22971520, 54428800),
                (1048576, 400): (43943040, 106857600),
                (1048576, 200): (85886080, 211715200)}
    for (size, rate), pair in expected.items():
        bounds = STUDY.limits(STUDY.Cell("pair", 2, size, "local", rate, "rnic-nn"))
        assert (bounds["fluid_small_ps"], bounds["fluid_phase_ps"]) == pair


def test_decision_keeps_fatal_evidence_void_and_incomplete_matrix_unscored():
    rows = [{"name": "shared", "status": "complete", "fan_in": 32,
             "guards": [], "comparison": {"below_one": 1}}]
    assert STUDY.decide(rows, expected=1)["decision"] == "H-sched"
    assert STUDY.decide(rows)["decision"] == "incomplete"
    rows[0]["guards"] = [STUDY.check("prefix_floor", False)]
    result = STUDY.decide(rows, expected=1)
    assert result["decision"] == "H-credit" and result["status"] == "void"
    assert not result["behavioral_score_interpretable"]
    rows[0].update(fan_in=1, guards=[])
    assert STUDY.decide(rows, expected=1)["decision"] == "inconclusive"


def test_published_evidence_is_recomputed_through_live_metrics():
    result = json.loads((STUDY.HERE / "results.json").read_bytes())
    rows = result["configurations"]
    by_name = {r["name"]: r for r in rows}
    for row in rows:
        c = STUDY.Cell(**{k: row[k] for k in STUDY.Cell.__dataclass_fields__})
        flows = [FlowCompletion(profile=c.profile, **f) for f in row["flows"]]
        live = STUDY.measure(c, flows, True)
        assert live["prefix_rows"] == row["prefix_rows"]
        assert live["phase_makespan_ps"] == row["phase_makespan_ps"]
        if c.profile == "rnic-cn":
            base = by_name[c.name.replace("rnic-cn", "rnic-nn")]
            ideal = [FlowCompletion(profile="rnic-nn", **f) for f in base["flows"]]
            assert STUDY.compare(flows, ideal) == row["comparison"]
    assert STUDY.decide(rows) == result["verdict"]
    assert result["verdict"]["decision"] == "H-sched"


def test_text_artifacts_accept_lf_digest_and_emit_lf_bytes(tmp_path):
    text = tmp_path / "input.md"
    text.write_bytes(b"first\r\nsecond\r\n")
    assert STUDY.digest(text, True) == hashlib.sha256(b"first\nsecond\n").hexdigest()
    target = tmp_path / "out.json"
    STUDY.write_json(target, {"a": 1})
    assert b"\r" not in target.read_bytes()
    STUDY.write_csv(tmp_path / "out.csv", [{"a": 1}])
    assert (tmp_path / "out.csv").read_bytes() == b"a\n1\n"
