import hashlib
import importlib.util
import json
import struct
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "examples" / "nvlink_incast_validation_v1"


def _load(name: str, filename: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, STUDY / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


score_study = _load("nvlink_incast_validation_result_score", "score_study.py")


def _result() -> dict[str, object]:
    return json.loads((STUDY / "results.json").read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_result_binds_the_one_completed_hardware_attempt() -> None:
    result = _result()

    assert result["status"] == "VOID_FATAL_GUARD"
    assert result["task_status"] == "OPEN"
    assert result["scheduler_job"] == "200456"
    assert result["execution_head"] == (
        "9cf29a1046933604dcad0efae2b2a57d9d07fa74"
    )
    assert result["expectations_commit"] == (
        "092080e682acaee9d68779c6ebb2195e97d0d6fb"
    )
    assert result["expectations_sha256"] == (
        "9f50aadba0085a54e78c156d61837e4c7db19a498d8fef9c1aba7b32e0a163b4"
    )
    assert result["attempt_manifest_sha256"] == (
        "3dae2e3829f9937ca8d66527043fa52ffe7d8566fe4c597cf32687448955d830"
    )
    assert result["producer_binary_sha256"] == (
        "fbd09baf75e3ec65ecd2a374a57ef64a645b09360aa467ca4b3d29e9d3651a08"
    )
    assert result["residual_task"] is None


def test_fatal_launch_skew_voids_the_whole_behavioral_comparison() -> None:
    result = _result()
    guards = {guard["id"]: guard for guard in result["fatal_guards"]["guards"]}

    assert result["measurement_validity"] == "VOID_FATAL_GUARD"
    assert result["fatal_guards"]["verdict"] == "VOID"
    assert guards["FG11"]["status"] == "FAIL"
    assert len(guards["FG11"]["findings"]) == 4
    assert all(
        finding["degree"] == 3 and finding["size_bytes"] == 262144
        for finding in guards["FG11"]["findings"]
    )
    assert all(
        guards[f"FG{index:02d}"]["status"] == "PASS"
        for index in (*range(1, 11), 12, 13)
    )
    assert result["summary"]["maximum_launch_skew_fraction"] == pytest.approx(
        0.10500671822482531
    )
    assert result["summary"]["launch_skew_fraction_high"] == 0.1
    assert result["summary"]["pass_cells"] == 0
    assert result["summary"]["miss_cells"] == 0
    assert result["summary"]["void_cells"] == 6


def test_all_six_raw_hardware_simulation_rows_are_published_but_unscored() -> None:
    comparisons = _result()["comparisons"]
    expected_aggregate = {
        (1, 262144): 2.2260868998109657,
        (2, 262144): 4.571428491709185,
        (3, 262144): 6.620689487960766,
        (1, 524288): 3.1801241464449697,
        (2, 524288): 6.13173631180753,
        (3, 524288): 9.365853825847713,
    }

    assert len(comparisons) == 6
    for row in comparisons:
        key = (row["degree"], row["size_bytes"])
        assert row["hardware_aggregate_gbps"] == pytest.approx(
            expected_aggregate[key]
        )
        assert row["physical_sanity"] == "PASS"
        assert row["verdict"] == "VOID"
        assert row["responsible_parameter"] == "undecidable_under_void_run"
        assert len(row["hardware_completion_us_by_source"]) == row["degree"]
        assert len(row["simulation_completion_us_by_source"]) == row["degree"]


def test_compact_result_keeps_all_repetitions_and_preservation_evidence() -> None:
    result = _result()

    assert result["coverage"] == {
        "expected_rows": 42,
        "hardware_cells": 6,
        "observed_rows": 42,
        "repetitions_per_cell": 7,
        "sample_rows": 42,
    }
    assert len(result["hardware_samples"]) == 42
    assert result["preservation"] == {
        "artifact_count": 59,
        "artifacts_sha256": (
            "dea2439d208a29101a561ec98f6318133c6c3b587dea33b01f59f4e1d3220a59"
        ),
        "verdict": "PASS",
    }
    assert not (STUDY / "results.jsonl").exists()


def test_report_leads_with_table_and_preserves_the_scope_boundary() -> None:
    result = _result()
    report = (STUDY / "RESULTS.md").read_text(encoding="utf-8")
    normalized = " ".join(report.split())

    assert report == score_study.render_markdown(result)
    assert report.index("## Hardware against simulation") < report.index("## What ran")
    assert "None of the six hardware cells" in normalized
    assert "receives a pass or miss verdict" in normalized
    assert "TRAF-73 stays open" in normalized
    assert "Degrees 4, 8 and 16 remain DECLARED SIMULATION" in normalized
    assert "This result covers long flows only" in normalized
    assert "supports but does not prove" in normalized
    assert "no small-flow hardware validity claim" in normalized
    assert "\u2014" not in report


def test_comparison_and_figures_are_stable_publication_artifacts() -> None:
    csv_path = STUDY / "comparison.csv"
    pdf_path = STUDY / "figures" / "nvlink-incast-hardware-simulation.pdf"
    png_path = STUDY / "figures" / "nvlink-incast-hardware-simulation.png"

    assert _sha256(csv_path) == (
        "874af0453fa673b37575ce8c03ef0fcf28eb5aad1b23161d2ff5cdebd41052bd"
    )
    assert csv_path.read_bytes().count(b"\n") == 7
    assert pdf_path.read_bytes().startswith(b"%PDF")
    png = png_path.read_bytes()
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert struct.unpack(">II", png[16:24]) == (1260, 702)
