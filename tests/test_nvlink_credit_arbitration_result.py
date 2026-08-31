import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "examples" / "nvlink_credit_arbitration_v1"
RESULT_PATH = RESULT_DIR / "results.json"
RESULT_MARKDOWN_PATH = RESULT_DIR / "RESULTS.md"
TRAFFIC_DOC_PATH = ROOT / "docs" / "modules" / "traffic.md"


@pytest.fixture(scope="module")
def result() -> dict[str, object]:
    return json.loads(RESULT_PATH.read_text(encoding="utf-8"))


def _row(
    result: dict[str, object],
    degree: int,
    policy: str,
) -> dict[str, object]:
    rows = result["simulation_rows"]
    assert isinstance(rows, list)
    return next(
        row
        for row in rows
        if row["degree"] == degree and row["policy"] == policy
    )


def test_result_keeps_evidence_classes_separate(result: dict[str, object]) -> None:
    assert result["hardware_status"] == "REGISTERED_NOT_RUN"
    assert result["behavioral_summary"] == {
        "family_count": 3,
        "instance_count": 15,
        "passed_instances": 15,
        "refuted_instances": 0,
        "verdict": "PASS",
    }
    assert result["fatal_summary"] == {
        "guard_instances": 105,
        "violations": 0,
        "verdict": "PASS",
    }
    evidence = result["evidence_classes"]
    assert isinstance(evidence, dict)
    assert evidence["hardware"] == "registered_not_run"
    assert evidence["behavioral"] != evidence["fatal"]


def test_physical_degree_three_predictions_separate_policies(
    result: dict[str, object],
) -> None:
    round_robin = _row(result, 3, "release_aware_round_robin")
    static = _row(result, 3, "static_interleave")
    greedy = _row(result, 3, "greedy_capture")
    assert round_robin["wire_gbps_per_source"] == pytest.approx(
        [87.158536, 59.921493, 59.921493]
    )
    assert static["wire_gbps_per_source"] == pytest.approx(
        [59.999945, 59.999945, 59.999945]
    )
    assert greedy["wire_gbps_per_source"] == pytest.approx(
        [99.75977, 53.620876, 53.620876]
    )
    assert all(
        row["topology_class"] == "PHYSICAL_NV4"
        for row in (round_robin, static, greedy)
    )


def test_extrapolations_are_disclosed_and_every_row_passes(
    result: dict[str, object],
) -> None:
    rows = result["simulation_rows"]
    assert isinstance(rows, list)
    assert len(rows) == 15
    assert all(row["behavioral_verdict"] == "PASS" for row in rows)
    assert {
        row["degree"]
        for row in rows
        if row["topology_class"] == "SIMULATED_MESH_EXTRAPOLATION"
    } == {4, 8, 16}


def test_result_preserves_profile_and_merged_artifacts(
    result: dict[str, object],
) -> None:
    authority = result["authority"]
    assert isinstance(authority, dict)
    preservation = authority["preservation"]
    assert preservation == {
        "candidate_profile_sha256": (
            "d33ef5b2c6fa87cc97e1e7b45a43a841a5da45f5462311e3981fbc903c56deb2"
        ),
        "path_content_digest_sha256": (
            "61af15faf7c7080f40a33f8f9d5503b3b0278f15be15997e90c6895cddf85c72"
        ),
        "tracked_bytes": 6429838,
        "tracked_file_count": 89,
    }


def test_scorer_correction_chronology_is_retained(result: dict[str, object]) -> None:
    chronology = result["scoring_chronology"]
    assert isinstance(chronology, dict)
    assert chronology["correction_class"] == (
        "post_specified_aggregate_quantization_bound"
    )
    assert chronology["first_run_commit"] == "0888344"
    assert chronology["first_run_result_sha256"] == (
        "b5713abb3902795cffd8e86ef1a9a4b40bd420a253928f36ab2d2c33442143ad"
    )


def test_report_keeps_hardware_gate_and_task_open() -> None:
    report = RESULT_MARKDOWN_PATH.read_text(encoding="utf-8")
    traffic = TRAFFIC_DOC_PATH.read_text(encoding="utf-8")
    assert "no hardware cell ran" in report
    assert "TRAF-73 stays\nopen" in report
    assert "- TRAF-73 (Precision; P1; M):" in traffic
    assert chr(0x2014) not in report
