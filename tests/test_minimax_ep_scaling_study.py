from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "examples/minimax_ep_scaling_v1"
RECORD = STUDY / "record.json"
RESULTS_CSV = STUDY / "results.csv"
RECORD_SHA256 = "ed14ca4f65bd504a185f6214f0310ba63bcf10cc9aec65a77bc20a87b426791d"
RESULTS_CSV_SHA256 = "c59b185eaac7895fd5eb7eef8224a1362d62ba25454166b6e4db726faadacfbb"
PNG_SHA256 = "13c080a3a2608da2b283f9d14b6bbf10716ac667c48d7ac3adcd7fb9e303669d"
PDF_SHA256 = "9f8d529e4230967cf83c33066db222788b8cc32aa956b1925023d800587fea4d"
METADATA_SHA256 = "2abe27ce621898c6a35438f5c41faec50fa496e8d6e126eefec62b7ebe925dcd"
WIDTHS = (8, 32, 128, 256)
FIRST_FREEZE_SHA256 = "9b355278c779c7834d18eaf3b19d16929f7b1800926e0ba1ba271f14a5d613ed"
CORRECTED_FREEZE_SHA256 = "b237945a945e1b1500ab299cf81faf20e704541f6c3e591b1cf90c418b5bb116"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _runner():
    spec = importlib.util.spec_from_file_location(
        "simllm_minimax_ep_scaling_runner",
        STUDY / "run_study.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_both_expectation_freezes_are_immutable() -> None:
    assert _sha256(STUDY / "expectations.md") == FIRST_FREEZE_SHA256
    assert _sha256(STUDY / "expectations_v2.md") == CORRECTED_FREEZE_SHA256


def test_corrected_dense_population_and_extrapolation_are_predeclared() -> None:
    runner = _runner()
    config = runner._load_config()
    assert config["chronology"]["corrected_expectations_commit"] == "4d1e41c"
    assert config["void_first_run"]["published_headline_ratio"] == 0.2742607736975033

    ep8 = runner._dense_packet_phases(config, width=8)
    assert len(ep8) == 2
    assert all(len(phase.phase.segments) == 56 for phase in ep8)
    assert all(phase.fabric_segments == () for phase in ep8)
    assert all(phase.nvlink_bytes == 56 * 24_576 for phase in ep8)

    ep32 = runner._dense_packet_phases(config, width=32)
    assert all(len(phase.phase.segments) == 992 for phase in ep32)
    assert all(len(phase.fabric_segments) == 768 for phase in ep32)
    assert all(len(phase.nvlink_segments) == 224 for phase in ep32)

    extrapolated = runner._extrapolate_dense_packet_width(
        config,
        width=256,
        anchor={
            "expert_parallel": 128,
            "layer_packet_ms": 0.5,
            "population_status": "measured full rank and message population",
        },
    )
    assert extrapolated["layer_packet_ms"] == 0.5 * 31 / 15
    assert extrapolated["simulated_messages_per_layer"] == 0
    assert extrapolated["represented_messages"] == 2 * 256 * 255 * 65
    assert extrapolated["extrapolation"]["anchor_expert_parallel"] == 128
    assert extrapolated["population_scored"] is False
    assert extrapolated["extrapolation"]["rule_commit"] == "a6ba97f"
    assert extrapolated["extrapolation"]["frozen_before_implementation"] is False
    assert extrapolated["extrapolation"]["scored"] is False


def test_family_d_generator_classifies_cost_models_and_diagnostic() -> None:
    runner = _runner()
    measured = runner._family_d_assessment(
        width=128,
        gpus_per_node=8,
        ratio=0.8026183885459625,
        population_scored=True,
    )
    assert measured == {
        "contention_comparison": False,
        "cross_node_contention_present": True,
        "score_status": "scored measured cell",
        "outcome": "REFUTED",
        "passed": False,
        "scored": True,
    }

    diagnostic = runner._family_d_assessment(
        width=256,
        gpus_per_node=8,
        ratio=1.187022158460092,
        population_scored=False,
    )
    assert diagnostic == {
        "contention_comparison": False,
        "cross_node_contention_present": True,
        "score_status": "unscored post-specified diagnostic",
        "outcome": "UNSCORED DIAGNOSTIC",
        "passed": None,
        "scored": False,
    }


def test_fg4_inspector_rejects_mutated_results_table(tmp_path: Path) -> None:
    runner = _runner()
    report = (STUDY / "RESULTS.md").read_text(encoding="utf-8")
    required = (
        "external NCCL-table cost model: dense SM90 fallback, generic "
        "half-precision all-gather plus reduce-scatter"
    )
    assert required in report
    mutated = tmp_path / "RESULTS.md"
    mutated.write_text(
        report.replace(required, "external NCCL-table cost model: traffic", 1),
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(RuntimeError, match="FG-4 RESULTS.md Family D row 0"):
        runner._inspect_artifact_disclosures(
            record_path=RECORD,
            csv_path=RESULTS_CSV,
            figures_dir=STUDY / "figures",
            results_path=mutated,
        )


def test_figure_rendering_uses_the_declared_external_python(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    interpreter = tmp_path / "external-venv" / "bin" / "python"
    runner._render_figures(
        tmp_path / "record.json",
        tmp_path / "figures",
        python_executable=interpreter,
    )

    assert calls[0][0][0] == os.fspath(interpreter)


def test_tracked_minimax_record_is_locked_and_nonvoid() -> None:
    assert _sha256(RECORD) == RECORD_SHA256
    assert _sha256(RESULTS_CSV) == RESULTS_CSV_SHA256
    assert _sha256(STUDY / "figures/minimax_ep_scaling.png") == PNG_SHA256
    assert _sha256(STUDY / "figures/minimax_ep_scaling.pdf") == PDF_SHA256
    assert (
        _sha256(STUDY / "figures/minimax_ep_scaling.metadata.json")
        == METADATA_SHA256
    )
    assert b"\r" not in RECORD.read_bytes()
    assert b"\r" not in RESULTS_CSV.read_bytes()

    record = _record()
    assert record["schema"] == "simllm-minimax-ep-scaling-record-v2"
    assert record["run_state"] == "nonvoid"
    assert record["attempt"] == "attempt-0001"
    assert record["run_commit"] == "66687d57275c336c11cbab048c678a1fb92e34ae"
    assert record["freeze_commits"] == ["61b66c4", "5a29bb0", "4d1e41c"]
    assert all(record["fatal_guards"].values())
    assert record["fresh_evaluations"] == {
        "bit_equal": True,
        "count": 2,
        "first_sha256": "e2da060655f9efa73f87cf126728e1ec731628ebe43c21dd3a0765b28fd45f33",
        "second_sha256": "e2da060655f9efa73f87cf126728e1ec731628ebe43c21dd3a0765b28fd45f33",
    }

    families = record["family_tallies"]
    assert {
        name: (families[name]["passed"], families[name]["denominator"])
        for name in ("E", "C", "D", "W")
    } == {"E": (4, 4), "C": (4, 4), "D": (0, 3), "W": (1, 1)}
    assert families["S"]["scored"] is False
    assert families["W"]["elapsed_seconds"] == 1039.514329513535
    assert record["artifact_disclosure_inspection"] == {
        "csv_rows_inspected": 4,
        "figure_caption_inspected": True,
        "figure_series_inspected": 5,
        "pdf_text_inspected": True,
        "record_rows_inspected": 4,
        "results_table_rows_inspected": 12,
    }

    report = (STUDY / "RESULTS.md").read_text(encoding="utf-8")
    opening = report[:2_000]
    for term in (
        "VOID against FG-4",
        "0.2742607736975033",
        "strategy comparison",
        "does not know which strategy",
    ):
        assert term in opening
    for first_run_value in (
        "0.8643398194341548",
        "0.4262782480503487",
        "0.3048657016451342",
        "0.2742607736975033",
        "0.02496",
        "4.0350336",
        "5.5890432",
        "7.1043648",
        "16,320 of 130,560",
        "3,258,777,600",
    ):
        assert first_run_value in report


def test_frozen_dispatch_and_composition_cells_are_locked() -> None:
    record = _record()
    families = record["family_tallies"]
    assert [cell["expert_parallel"] for cell in families["E"]["cells"]] == list(
        WIDTHS
    )
    assert [cell["actual_hex"] for cell in families["E"]["cells"]] == [
        "0x1.ec0b780346dc6p+0",
        "0x1.3d27bdfef25dcp+4",
        "0x1.263c1785d279dp+5",
        "0x1.9b29e147ae148p+5",
    ]
    assert all(
        cell["actual_hex"] == cell["expected_hex"] and cell["passed"]
        for cell in families["E"]["cells"]
    )
    assert [cell["quotient"] for cell in families["C"]["cells"]] == [
        1.0,
        1.0,
        1.0,
        1.0,
    ]
    assert all(cell["passed"] for cell in families["C"]["cells"])
    assert families["C"]["interpretation"] == (
        "end-to-end parity reusing the dispatch code validated by E"
    )


def test_corrected_dense_refutations_and_sparse_geometry_are_locked() -> None:
    record = _record()
    rows = record["rows"]
    assert [row["expert_parallel"] for row in rows] == list(WIDTHS)
    assert [row["family_d_ratio"] for row in rows] == [
        0.02590463307406155,
        0.3530150565741419,
        0.8026183885459625,
        1.187022158460092,
    ]
    assert [row["family_d_contention_comparison"] for row in rows] == [
        False,
        False,
        False,
        False,
    ]
    assert [row["family_d_score_status"] for row in rows] == [
        "scored measured cell",
        "scored measured cell",
        "scored measured cell",
        "unscored post-specified diagnostic",
    ]
    assert [row["family_d_outcome"] for row in rows] == [
        "REFUTED",
        "REFUTED",
        "REFUTED",
        "UNSCORED DIAGNOSTIC",
    ]
    assert [row["family_d_packet_ms"] for row in rows] == [
        0.04979,
        6.997536,
        29.519776,
        61.00753706666667,
    ]
    assert [row["family_d_simulated_messages_per_layer"] for row in rows] == [
        112,
        1_984,
        32_512,
        0,
    ]
    assert [row["family_d_represented_messages"] for row in rows] == [
        7_280,
        128_960,
        2_113_280,
        8_486_400,
    ]
    assert [cell["passed"] for cell in record["family_tallies"]["D"]["cells"]] == [
        False,
        False,
        False,
        None,
    ]
    assert rows[-1]["family_d_extrapolation"]["cross_node_bytes_per_rank_factor"] == (
        31 / 15
    )
    assert rows[-1]["family_d_extrapolation"]["frozen_before_implementation"] is False
    assert rows[-1]["family_d_extrapolation"]["scored"] is False

    assert [row["family_s_simulated_messages_per_layer"] for row in rows] == [
        112,
        1_340,
        7_444,
        15_640,
    ]
    assert all(
        row["family_s_population_status"]
        == "full rank and realized-message population"
        for row in rows
    )
    assert [row["family_s_payload_bytes_per_rank"] for row in rows] == [
        258_048.0,
        285_696.0,
        292_608.0,
        293_760.0,
    ]
    widest_geometry = rows[-1]["family_s_routing_geometry"]
    assert widest_geometry == {
        "expected_cross_node_senders_per_receiver": 29.57691192626953,
        "expected_distinct_destinations_per_source": 30.411744117736816,
        "maximum_cross_node_senders_per_receiver": 32,
        "realized_cross_node_senders_per_receiver": 29.78125,
        "realized_distinct_destinations_per_source": 30.546875,
        "realized_geometry_source": (
            "simulator completion rows for cross-node flows plus analytically "
            "completed same-node segments"
        ),
    }
    assert [row["void_first_run_ratio"] for row in rows] == [
        0.8643398194341548,
        0.4262782480503487,
        0.3048657016451342,
        0.2742607736975033,
    ]
    assert all(row["void_first_run_status"] == "VOID against FG-4" for row in rows)


def test_physical_ledger_and_portable_paths_are_locked() -> None:
    record = _record()
    assert record["physical_sanity"] == {
        "dense_dispatch_plus_combine_fabric_bytes_per_rank": 12_189_696,
        "dense_dispatch_plus_combine_wire_bytes_per_rank": 12_533_760,
        "dense_half_buffer_bytes_per_rank": 6_291_456,
        "dense_serialization_floor_microseconds_per_layer": 243.79391999999999,
        "link_bytes_per_second": 50_000_000_000,
        "sparse_combine_bytes_per_rank": 195_840.0,
        "sparse_dispatch_bytes_per_rank": 97_920.0,
        "sparse_dispatch_plus_combine_bytes_per_rank": 293_760.0,
        "sparse_dispatch_plus_combine_fabric_bytes_per_rank": 286_524.0,
        "sparse_serialization_floor_microseconds_per_layer": 5.73048,
        "widest_expert_parallel": 256,
    }
    assert record["external_table_identification"] == {
        "caller_argument_interpretation": (
            "the caller passes tokens times hidden times expert-parallel width as "
            "an element count, so the table axis unit is unresolved between its "
            "byte name and the caller's element semantics"
        ),
        "dtype_identifier": "half",
        "dtype_interpretation": (
            "generic half precision; the table does not identify BF16"
        ),
        "message_axis_name": "message_bytes",
    }
    assert record["family_d_fixed_overhead_analysis"] == {
        "caller_argument_semantics": "half-precision element count",
        "donor_latency_microseconds_per_layer": 55.445,
        "dtype_identifier": "half",
        "dtype_interpretation": (
            "generic half precision; the source table does not identify BF16"
        ),
        "expert_parallel": 128,
        "external_minus_packet_gap_ms": 7.259565741071427,
        "fixed_and_algorithmic_residual_microseconds_per_layer": 43.21161333333334,
        "ideal_ring_serialization_microseconds_per_layer": 12.233386666666666,
        "inflated_residual_exceeds_gap": True,
        "inflated_residual_ms_over_65_layers": 28.664346541071428,
        "message_axis_name": "message_bytes",
        "rank_factor": 10.205357142857142,
    }
    assert record["bulk_evidence"] == "${SIMLLM_MINIMAX_FIX_BULK_ROOT}/attempt-0001"
    text = RECORD.read_text(encoding="utf-8")
    slash = chr(47)
    for forbidden in (
        f"{slash}data3{slash}",
        f"{slash}home{slash}",
        f"~{slash}",
    ):
        assert forbidden not in text

    with RESULTS_CSV.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert [int(row["expert_parallel"]) for row in rows] == list(WIDTHS)
    assert all(
        row["family_d_external_traffic_definition"]
        == row["family_d_packet_traffic_definition"]
        for row in rows
    )
    assert all(row["family_d_external_strategy"] for row in rows)
    assert [row["family_d_contention_comparison"] for row in rows] == [
        "False",
        "False",
        "False",
        "False",
    ]
    assert [row["family_d_outcome"] for row in rows] == [
        "REFUTED",
        "REFUTED",
        "REFUTED",
        "UNSCORED DIAGNOSTIC",
    ]
    assert all(row["family_s_sparse_strategy"] for row in rows)
    assert all(row["void_first_run_status"] == "VOID against FG-4" for row in rows)

    metadata = json.loads(
        (STUDY / "figures/minimax_ep_scaling.metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert metadata["point_annotations"] == {
        "ep8": "not a contention comparison: no cross-node traffic",
        "ep256": "unscored post-specified diagnostic",
    }
    assert metadata["series"][-1]["panel"] == "family-d-cost-model-ratio"


def test_runner_validates_tracked_record_without_pythonpath(tmp_path: Path) -> None:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [
            sys.executable,
            os.fspath(STUDY / "run_study.py"),
            "--validate-tracked",
        ],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "run_state=nonvoid"


def test_live_sdk_family_matches_when_environment_is_available(
    tmp_path: Path,
) -> None:
    raw_venv = os.environ.get("SIMLLM_EXTERNAL_AIC_VENV")
    if not raw_venv:
        pytest.skip(
            "live SDK family requires SIMLLM_EXTERNAL_AIC_VENV; "
            "the tracked frozen cells remain covered"
        )
    venv = Path(raw_venv)
    candidates = (venv / "bin/python", venv / "Scripts/python.exe")
    python = next((candidate for candidate in candidates if candidate.is_file()), None)
    assert python is not None, "SIMLLM_EXTERNAL_AIC_VENV has no Python interpreter"
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [os.fspath(python), os.fspath(STUDY / "run_study.py"), "--live-sdk-worker"],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    live = json.loads(completed.stdout.splitlines()[-1])
    frozen = json.loads((STUDY / "study_config.json").read_text(encoding="utf-8"))
    assert [row["decode_step_hex"] for row in live["widths"]] == [
        row["live_decode_step_hex"] for row in frozen["widths"]
    ]
    assert [row["dispatch_hex"] for row in live["widths"]] == [
        row["live_dispatch_hex"] for row in frozen["widths"]
    ]
