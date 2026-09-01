import hashlib
import importlib.util
import json
import struct
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "examples" / "nvlink_incast_validation_v1"


def _load(name: str, filename: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, STUDY / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


score_study = _load(
    "nvlink_incast_validation_run2_result_score",
    "score_study_run2.py",
)


def _result() -> dict[str, object]:
    return json.loads((STUDY / "results_run2.json").read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_run2_result_binds_the_one_completed_hardware_attempt() -> None:
    result = _result()

    assert result["status"] == "VALID_0_PASS_6_MISS"
    assert result["task_status"] == "CLOSED_WITH_FINDING"
    assert result["task_id"] == score_study.TASK_ID == "TRAF-74"
    assert result["residual_task"] == score_study.RESIDUAL_TASK_ID == "TRAF-86"
    assert result["scheduler_job"] == "202466"
    assert result["execution_head"] == (
        "2389e00545a83af898d64fdde3c9b47c7199e3d3"
    )
    assert result["expectations_commit"] == (
        "b21ba822707d2d7c80b83ee2d3fb87f4fa93178d"
    )
    assert result["expectations_sha256"] == (
        "5465271e9909cebc214c153209316a6f266ec142d7e578b3279935b1c6a10a53"
    )
    assert result["attempt_manifest_sha256"] == (
        "4ebe25ccd49dca3047916af0c08c7302d3111c7ff6fbd0d6c41aa60ff1e72ee0"
    )
    assert result["producer_binary_sha256"] == (
        "a30df800c3bc2cf725b1e8678d561a27c78bf55625367b5039759aa58f910a54"
    )


def test_run2_all_fatal_guards_pass_and_launch_skew_has_margin() -> None:
    result = _result()
    guards = {guard["id"]: guard for guard in result["fatal_guards"]["guards"]}

    assert result["measurement_validity"] == "VALID_FOR_FROZEN_COMPARISON"
    assert result["fatal_guards"]["verdict"] == "PASS"
    assert all(guards[f"FG{index:02d}"]["status"] == "PASS" for index in range(1, 14))
    assert result["summary"] == {
        "launch_skew_fraction_high": 0.1,
        "maximum_launch_skew_fraction": pytest.approx(0.011289739769680661),
        "miss_cells": 6,
        "pass_cells": 0,
        "void_cells": 0,
        "worst_absolute_signed_relative_error": pytest.approx(20.111748331889775),
    }
    assert len(result["fatal_guards"]["launch_skew_rows"]) == 42
    assert all(
        row["launch_skew_fraction"] <= 0.1
        for row in result["fatal_guards"]["launch_skew_rows"]
    )


def test_run2_all_six_cells_publish_literal_packetization_misses() -> None:
    comparisons = _result()["comparisons"]
    expected_aggregate = {
        (1, 4 << 20): 4.461873614611664,
        (2, 4 << 20): 8.914036873594686,
        (3, 4 << 20): 13.760358508193,
        (1, 8 << 20): 4.561247077147439,
        (2, 8 << 20): 9.0619467509202,
        (3, 8 << 20): 14.288372571660371,
    }

    assert len(comparisons) == 6
    for row in comparisons:
        key = (row["degree"], row["size_bytes"])
        assert row["hardware_aggregate_gbps"] == pytest.approx(
            expected_aggregate[key]
        )
        assert row["physical_sanity"] == "PASS"
        assert row["verdict"] == "MISS"
        assert row["responsible_parameter"] == "packetization"
        assert row["within_frozen_band"] is False
        assert row["launch_skew_fraction_high"] == 0.1
        assert row["maximum_launch_skew_fraction"] <= 0.1
        assert len(row["hardware_completion_us_by_source"]) == row["degree"]
        assert len(row["simulation_completion_us_by_source"]) == row["degree"]
        assert all(
            error < -0.92
            for error in row["completion_signed_relative_error_by_source"]
        )


def test_run2_physical_sanity_covers_three_independent_angles() -> None:
    sanity = _result()["physical_sanity"]

    assert sanity["verdict"] == "PASS"
    assert sanity["minimum_hardware_over_floor"] > 13.5
    assert sanity["maximum_completion_over_ceiling"] < 0.38
    assert 1.93 < sanity["minimum_eight_mib_over_four_mib_completion"] < 2.0
    assert 1.93 < sanity["maximum_eight_mib_over_four_mib_completion"] < 2.0
    assert 4.45 < sanity["minimum_hardware_source_goodput_gbps"] < 4.5
    assert 4.8 < sanity["maximum_hardware_source_goodput_gbps"] < 4.81


def test_run2_compact_result_keeps_all_rows_and_raw_evidence_identity() -> None:
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
        "artifact_count": 71,
        "artifacts_sha256": (
            "68fd24d949424ca580877e1cfc693383921fd8df5c14e2c413bf6222560dc79a"
        ),
        "verdict": "PASS",
    }
    raw = result["raw_evidence"]
    assert raw["attempt_manifest_sha256"] == result["attempt_manifest_sha256"]
    assert raw["row_count"] == 42
    assert len(raw["row_digests"]) == len(set(raw["row_digests"])) == 42
    required = " ".join(raw["required_observables"])
    for term in (
        "checksum",
        "ordering",
        "per-link data and raw counters",
        "replay",
        "recovery",
        "throttle",
        "topology",
        "competing-process",
    ):
        assert term in required


def test_run2_result_names_the_exact_base_model_and_flow_policy() -> None:
    identity = _result()["simulation_identity"]

    assert identity == {
        "flow_policy": "release_aware_round_robin",
        "implementation": "simllm-htsim-nvlink-domain-v1",
        "model_sha256": (
            "498de9ca7e81bd59679ce55242eeae516624d9be495e802b5c1ee959ee213f47"
        ),
        "module_version_commit": "65593131a0448d2b33f51018d5972c918dad3493",
        "profile_sha256": (
            "d33ef5b2c6fa87cc97e1e7b45a43a841a5da45f5462311e3981fbc903c56deb2"
        ),
        "release_ps": 0,
    }


def test_run2_report_leads_with_table_and_states_project_effect() -> None:
    result = _result()
    report = (STUDY / "RESULTS_RUN2.md").read_text(encoding="utf-8")
    normalized = " ".join(report.split())

    assert report == score_study.render_markdown(result)
    assert report.index("## Hardware against simulation") < report.index("## What ran")
    assert "Maximum launch skew" in report
    assert "0 of 6 cells pass and 6 miss" in normalized
    assert "Every miss names `packetization`" in normalized
    assert "TRAF-74 closes" in normalized
    assert "TRAF-86 owns" in normalized
    assert "Degrees 4, 8 and 16 remain DECLARED SIMULATION" in normalized
    assert "first frozen capture remains byte-identical and void" in normalized
    assert "supports but does not prove" in normalized
    assert "no small-flow hardware validity claim" in normalized
    assert "+" + "/-" not in report
    assert "\u2014" not in report
    assert "/" + "data3" not in report
    assert "/" + "home/" not in report


def test_run2_comparison_and_figures_are_stable_publication_artifacts() -> None:
    csv_path = STUDY / "comparison_run2.csv"
    pdf_path = STUDY / "figures" / "nvlink-incast-hardware-simulation-run2.pdf"
    png_path = STUDY / "figures" / "nvlink-incast-hardware-simulation-run2.png"

    assert _sha256(csv_path) == (
        "c128aba51105008da1ce8c81dbc6c186ceaaa0b02bb5e853b1da251d4df96049"
    )
    assert csv_path.read_bytes().count(b"\n") == 7
    assert pdf_path.read_bytes().startswith(b"%PDF")
    png = png_path.read_bytes()
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert struct.unpack(">II", png[16:24]) == (1260, 702)
