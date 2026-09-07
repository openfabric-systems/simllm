"""Offline evidence checks for the bounded control recovery study."""

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

from simllm.backends.htsim_rnic import FlowCompletion

HERE = Path(__file__).resolve().parents[1] / "examples/control_recovery_v1"
SPEC = importlib.util.spec_from_file_location("control_recovery_study", HERE / "run_study.py")
STUDY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = STUDY
SPEC.loader.exec_module(STUDY)


def flow(identity, completed, payload=100, start=0):
    return FlowCompletion("rnic-cn", identity, identity, 8, 1, payload,
                          start, completed, completed - start)


def test_text_digests_accept_checkout_line_endings_and_outputs_use_lf(tmp_path):
    path = tmp_path / "input.txt"
    path.write_bytes(b"first\nsecond\n")
    expected = STUDY.text_digests(path)
    path.write_bytes(b"first\r\nsecond\r\n")
    assert set(expected) <= set(STUDY.text_digests(path))
    STUDY.write_json(tmp_path / "result.json", {"scope": "test", "values": [1, 2]})
    assert b"\r" not in (tmp_path / "result.json").read_bytes()


def test_prefix_floor_detects_aggregate_impossibility_when_each_flow_is_plausible():
    # One 100-byte transfer needs 2000 ps at 400 Gbit/s. Each of these
    # finishes after that, but together they require 4000 ps.
    rows = [flow(1, 4_003_000), flow(2, 4_003_000)]
    findings, ratio = STUDY.receiver_prefix_findings(rows, 400)
    assert findings == ["receiver 8 prefix 2 beats byte floor"]
    assert ratio < 1
    rows[1] = flow(2, 4_004_000)
    assert STUDY.receiver_prefix_findings(rows, 400) == ([], 1)


def test_prefix_guard_refuses_to_treat_shifted_starts_as_aligned():
    with pytest.raises(ValueError, match="aligned"):
        STUDY.receiver_prefix_findings([flow(1, 5_000_000, start=1000)], 400)


def test_nearest_rank_is_within_cell_and_uses_upper_order_statistic():
    assert STUDY.nearest_rank([1, 4, 8, 20], .5) == 4
    assert STUDY.nearest_rank([1, 4, 8, 20], .99) == 20
    with pytest.raises(ValueError):
        STUDY.nearest_rank([], .99)


def test_fatal_failure_voids_study_without_becoming_a_score():
    rows = [{"cell": "example", "arm": "candidate", "mode": "headroom",
             "source": "pipeline", "pattern": "pipeline", "fatal_findings": ["lost flow"],
             "status": "failed", "phase_makespan_ps": None, "compatibility_oracle": None}]
    result = STUDY.summarize(rows, {})
    assert result["verdict"] == "void"
    assert result["completed_configurations"] == 0
    assert result["compatibility_oracles"] == 0
    assert result["behavioral_relations"] == []
    assert "score" not in result
    assert len(result["fatal_findings"]) == 1


def test_expected_off_mode_loss_has_no_manufactured_completion(tmp_path):
    cell = STUDY.Cell("collective", tmp_path, 64, 400, "all-to-all",
                      tmp_path / "topology", "collective", "1", True)
    manifest = "[RNIC manifest] rnic_cn_control_recovery=none rnic_cn_control_headroom_admissions=0\n"
    row = STUDY.analyze(cell, tmp_path, manifest + "htsim_rnic: rnic-cn fabric dropped control lifecycle 7",
                        2, "none", "candidate", {})
    assert row["status"] == "expected-control-loss"
    assert row["phase_makespan_ps"] is None
    assert row["compatibility_oracle"] is None
    assert row["fatal_findings"] == []
    wrong = STUDY.analyze(cell, tmp_path, manifest + "unrelated error", 2, "none", "candidate", {})
    assert wrong["fatal_findings"] == ["expected control-loss identity exit missing"]


def test_goal_message_identity_counts_duplicates_and_bytes(tmp_path):
    goal = tmp_path / "input.goal"
    goal.write_text("num_ranks 16\nrank 0 {\na: send 64b to 8 tag 7\nb: send 64b to 8 tag 7\n}\n")
    assert STUDY.messages(goal) == Counter({(0, 8, 7, 64): 2})


def test_published_record_keeps_provenance_and_evidence_classes_separate():
    path = HERE / "results.json"
    if not path.exists():
        pytest.skip("study artifacts have not been published")
    result = json.loads(path.read_text())
    assert set(STUDY.text_digests(HERE / "expectations.md")) & set(
        result["provenance"]["expectations_sha256"])
    assert result["provenance"]["expectations_commit"].startswith(STUDY.FREEZE)
    assert result["provenance"]["htsim_expectations_commit"].startswith(STUDY.HTSIM_FREEZE)
    assert len(result["provenance"]["binary_sha256"]) == 64
    assert result["run_configurations"] == len(result["cells"])
    assert result["verdict"] == ("void" if result["fatal_findings"] else "valid")
    for row in result["cells"]:
        if row["status"] == "expected-control-loss":
            assert row["phase_makespan_ps"] is None
            assert row["fct_p99_ps"] is None
        if row["compatibility_oracle"] is not None:
            assert row["compatibility_oracle"]["byte_identical"] == (
                row["completion_sha256"] == row["compatibility_oracle"]["reference_sha256"])


def test_resume_accepts_lf_digest_without_loosening_binary_identity():
    saved = {"goal_text_sha256": ["lf"], "binary_sha256": "one"}
    current = {"goal_text_sha256": ["crlf", "lf"], "binary_sha256": "one"}
    assert STUDY.compatible_locks(saved, current)
    current["binary_sha256"] = "two"
    assert not STUDY.compatible_locks(saved, current)
