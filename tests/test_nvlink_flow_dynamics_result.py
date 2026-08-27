import hashlib
import json
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "examples" / "nvlink_flow_dynamics_v1"
RESULT_PATH = STUDY / "results.json"
REPORT_PATH = STUDY / "RESULTS.md"


def _result():
    return json.loads(RESULT_PATH.read_text(encoding="utf-8"))


def test_exact_transitions_and_schedule_verdicts_are_published():
    result = _result()

    assert result["study_verdict"] == "PASS_WITH_EXPECTED_FANOUT_REFUTATION"
    assert result["convergence_1_to_2"] == {
        "expected_open_ps": 13886,
        "observed_open_ps": 13886,
        "residual_ps": 0,
        "verdict": "PASS",
    }
    assert result["divergence_2_to_1"]["expected_time_to_target_ps"] == 16684
    assert result["divergence_2_to_1"]["observed_time_to_target_ps"] == 16684
    assert result["divergence_2_to_1"]["residual_ps"] == 0
    assert result["divergence_2_to_1"]["verdict"] == "PASS"
    assert result["overall_schedule"]["completion_order"] == [
        "flow-c",
        "flow-b",
        "flow-a",
    ]
    assert result["overall_schedule"]["steady_rate_checks"] == 219
    assert result["overall_schedule"]["steady_rate_failures"] == 0


def test_every_fct_rung_and_incast_degree_has_a_pass_verdict():
    result = _result()
    cdf = result["fct_cdf"]
    expected_sizes = {256, 1024, 4096, 16384, 65536, 262144, 524288}

    assert cdf["seed_count"] == 9
    assert cdf["band"] == "pointwise minimum to maximum empirical CDF across seeds"
    assert len(cdf["verdicts"]) == 21
    assert {(row["degree"], row["size_bytes"]) for row in cdf["verdicts"]} == {
        (degree, size_bytes)
        for degree in (1, 2, 3)
        for size_bytes in expected_sizes
    }
    assert all(row["verdict"] == "PASS" for row in cdf["verdicts"])

    incast = result["incast"]
    assert [row["degree"] for row in incast] == [1, 2, 3]
    assert [row["expected_binding_module"] for row in incast] == [
        "tx_pair_links",
        "tx_pair_links",
        "rx",
    ]
    assert all(row["verdict"] == "PASS" for row in incast)
    assert all(row["simulated_payload_gbps"] <= row["payload_ceiling_gbps"] for row in incast)


def test_guards_preservation_and_honest_fanout_refutation_are_published():
    result = _result()

    assert len(result["fatal_guards"]) == 13
    assert all(row["verdict"] == "PASS" for row in result["fatal_guards"])
    assert result["fatal_guard_verdict"] == "PASS"
    assert result["authority"]["preservation_artifacts_checked"] == 60
    assert result["authority"]["static_identity_sha256"] == (
        "2f2af64619ed3c6341b209d877d9f1e6984a67e44b97b5eb176a157294a6c252"
    )
    assert result["fanout_separate_check"]["verdict"] == "REFUTED"
    assert result["fanout_separate_check"]["relative_error"] > 0.46


def test_attempt_history_retains_identical_complete_run_evidence():
    result = _result()

    assert [row["commit"] for row in result["attempt_history"]] == [
        "b808a6b",
        "97cb90d",
        "4f4022e",
    ]
    assert len(result["raw_evidence_sha256"]) == 8
    assert all(len(digest) == 64 for digest in result["raw_evidence_sha256"].values())


def test_all_published_figures_match_their_manifest_and_are_readable():
    result = _result()

    assert len(result["figure_artifacts"]) == 10
    for artifact in result["figure_artifacts"]:
        path = STUDY / "figures" / artifact["path"]
        payload = path.read_bytes()
        assert len(payload) == artifact["bytes"]
        assert hashlib.sha256(payload).hexdigest() == artifact["sha256"]
        if path.suffix == ".pdf":
            assert payload.startswith(b"%PDF-")
        else:
            assert payload.startswith(b"\x89PNG\r\n\x1a\n")
            width, height = struct.unpack(">II", payload[16:24])
            assert width >= 1400
            assert height >= 1300


def test_report_is_portable_and_traf69_is_closed():
    report_bytes = REPORT_PATH.read_bytes()
    report = report_bytes.decode("utf-8")
    traffic = (ROOT / "docs" / "modules" / "traffic.md").read_text(encoding="utf-8")
    ledger = json.loads((ROOT / "docs" / "task-ledger.json").read_text(encoding="utf-8"))

    assert b"\r" not in report_bytes
    assert "\N{EM DASH}" not in report
    assert "/data3/" not in report
    assert "/home/" not in report
    assert "0 + 1,692 + 10,880 + 1,314 = 13,886 ps" in report
    assert "0 + 2 * 10,880 - 3 * 1,692 + 0 = 16,684 ps" in report
    assert "- TRAF-69 (" not in traffic
    assert "TRAF-69" in ledger["closed"]
