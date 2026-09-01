import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "examples" / "nvlink_mechanism_alignment_v1"
RESULT = STUDY / "results.json"
REPORT = STUDY / "RESULTS.md"
RESULT_SHA256 = "579acea13bd8c899e1c7c00a752dc23397136d44a95827abccf6a8294283e32d"


def _result() -> dict[str, object]:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def test_published_result_is_the_final_pass_attempt() -> None:
    result = _result()

    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == RESULT_SHA256
    assert result["schema"] == "simllm-nvlink-mechanism-alignment-result-v1"
    assert result["study_verdict"] == "PASS"
    assert result["fatal_guard_verdict"] == "PASS"
    assert len(result["fatal_guards"]) == 13
    assert all(row["verdict"] == "PASS" for row in result["fatal_guards"])


def test_void_attempt_is_retained_before_the_final_pass() -> None:
    attempts = _result()["attempt_history"]

    assert [row["attempt"] for row in attempts] == ["0001", "0002", "0003"]
    assert [row["verdict"] for row in attempts] == ["VOID", "PASS", "PASS"]
    assert len({row["result_sha256"] for row in attempts}) == 3
    assert "historical source lock" in attempts[0]["finding"]


def test_published_oracles_and_signed_shifts_are_complete() -> None:
    result = _result()
    cells = result["sanity_cells"]
    shifts = result["inherited_envelope_shifts"]

    assert len(cells) == 4
    assert {(row["packet_flits"], row["per_link_bytes_per_second"]) for row in cells} == {
        (17, 12_500_000_000),
        (17, 25_000_000_000),
        (18, 12_500_000_000),
        (18, 25_000_000_000),
    }
    assert all(row["serialization_verdict"] == "PASS" for row in cells)
    assert len(shifts) == 6
    assert all(row["signed_shift"].startswith("+0") for row in shifts)


def test_result_is_lf_portable() -> None:
    payload = RESULT.read_bytes()
    text = payload.decode("utf-8")
    compact_plus_minus = "+/" + "-"

    assert b"\r" not in payload
    assert "\N{EM DASH}" not in text
    assert "/data3/" not in text
    assert "/home/" not in text
    assert compact_plus_minus not in text


def test_report_publishes_handoff_limits_and_every_signed_shift() -> None:
    payload = REPORT.read_bytes()
    report = payload.decode("utf-8")
    compact_plus_minus = "+/" + "-"
    traffic = (ROOT / "docs" / "modules" / "traffic.md").read_text(
        encoding="utf-8"
    )
    ledger = json.loads(
        (ROOT / "docs" / "task-ledger.json").read_text(encoding="utf-8")
    )

    assert "What ran:" in report
    assert "What came out:" in report
    assert "What it changes:" in report
    assert "What it does not change:" in report
    assert "94.117647 GB/s" in report
    assert "88.888889 GB/s" in report
    assert "+655,737 ps" in report
    assert "+1,311,097 ps" in report
    assert report.count("+0") >= 6
    assert "TRAF-73 stays open" in report
    assert "- TRAF-80 (" not in traffic
    assert "TRAF-80" in ledger["closed"]
    assert "integrator-owned" in report
    assert b"\r" not in payload
    assert "\N{EM DASH}" not in report
    assert "/data3/" not in report
    assert "/home/" not in report
    assert compact_plus_minus not in report
