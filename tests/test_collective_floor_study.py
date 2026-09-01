"""Lock the corrected aggregate collective-floor study artifacts.

These tests read committed outputs only. Native-tool checks are separate and
carry named skip reasons, so ordinary continuous integration still locks every
published relation without pretending it reproduced an unavailable backend.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "examples/collective_floor_calibration_v1"
CONFIG_PATH = STUDY / "study_config.json"
RECORD_PATH = STUDY / "record.json"
CSV_PATH = STUDY / "results.csv"
RESULTS_PATH = STUDY / "RESULTS.md"
RUNNER = STUDY / "run_study.py"
RECORD_SHA256 = "3e41e6ec80e67eed851ca68884da0244ac8f79c338bef06272d8ccb97113026f"
CSV_SHA256 = "9dc7e5abfb955e6fc731e86fd51c9cc44e24d8b921506e76063f0e9ca723d3e5"


@pytest.fixture(scope="module")
def config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def record() -> dict:
    return json.loads(RECORD_PATH.read_text(encoding="utf-8"))


def test_first_attempt_void_opens_the_published_chronology() -> None:
    text = RESULTS_PATH.read_text(encoding="utf-8")
    assert text.index("## Chronology") < text.index("## Outcome")
    chronology = text[text.index("## Chronology") : text.index("## Outcome")]
    assert "Attempt 0001 is void" in chronology
    assert "stopped before regime selection,\nfitting or implementation" in chronology
    assert "fatal axis guard worked as designed" in chronology
    assert "No attempt-0001 directory exists by construction" in chronology
    assert "worker report is its only evidence" in chronology
    assert "Attempt 0002 is superseded" in chronology
    assert "Attempt 0003 is void" in chronology
    assert "Attempt 0004 is the corrected publication" in chronology


def test_corrected_axis_and_three_sdk_citations_are_locked(config: dict) -> None:
    axis = config["axis"]
    assert axis["source_coordinate"] == "ELEMENTS"
    assert axis["physical_fit_axis"] == "BYTES"
    assert axis["conversion"] == (
        "true_bytes = source_elements * dtype_width_bytes"
    )
    assert axis["dtype_width_bytes"] == {"half": 2, "int8": 1}
    assert [row["path"] for row in axis["sdk_citations"]] == [
        "aiconfigurator_core/sdk/operations/communication.py",
        "aiconfigurator_core/sdk/operations/communication.py",
        "aiconfigurator_core/sdk/common.py",
    ]
    assert [row.get("line", row.get("lines")) for row in axis["sdk_citations"]] == [
        516,
        "394-395",
        "1112-1113",
    ]


def test_equal_byte_guard_resolved_distinct_cells(record: dict) -> None:
    axis = record["axis"]
    assert axis["passed"]
    assert axis["equal_byte_cells_are_distinct"]
    assert axis["observed"] == {
        "half": {
            "latency_ms": "0.00517",
            "matches_frozen_cell": True,
            "source_elements": 256,
            "true_bytes": 512,
        },
        "int8": {
            "latency_ms": "0.00519",
            "matches_frozen_cell": True,
            "source_elements": 512,
            "true_bytes": 512,
        },
    }


def test_all_fatal_guards_held_and_determinism_is_fresh_process(record: dict) -> None:
    guards = record["fatal_guards"]
    assert [guard["id"] for guard in guards] == [f"FG-{index}" for index in range(1, 8)]
    assert all(guard["held"] for guard in guards)
    assert record["verdict"] == "interpretable"
    determinism = guards[5]["evaluated"]
    assert determinism["fresh_processes"] == 2
    assert determinism["excluded_field_name"] == "wall_time_seconds"
    assert determinism["evaluation_1_sha256"] == determinism["evaluation_2_sha256"]


def test_training_only_true_byte_boundaries_match_the_freeze(
    config: dict,
    record: dict,
) -> None:
    frozen = {
        (row["dtype"], row["operation"], row["ranks"]): row[
            "lower_bounds_of_following_regimes"
        ]
        for row in config["fit"]["regime_boundaries_true_bytes"]
    }
    observed = {}
    for regime in record["fit"]["regimes"]:
        key = (regime["dtype"], regime["operation"], regime["ranks"])
        if regime["regime_index"]:
            observed.setdefault(key, []).append(regime["lower_bytes"])
        else:
            observed.setdefault(key, [])
        assert int(regime["floor_ps"].split("/")[0]) > 0
        assert int(regime["slope_ps_per_byte"].split("/")[0]) > 0
    assert observed == frozen
    assert record["fit"]["training_cells"] == 63
    assert record["fit"]["holdout_cells"] == 63


def test_family_h_refutation_and_error_summaries_are_locked(record: dict) -> None:
    family = record["families"]["H"]
    assert family["status"] == "REFUTED"
    assert family["passed"] == 51
    assert family["denominator"] == 63
    assert family["summary"] == {
        "after_median_relative_error": 0.03267105960264901,
        "after_p95_relative_error_nearest_rank": 0.1979721778791334,
        "before_median_relative_error": 0.9161607875307629,
        "before_p95_relative_error_nearest_rank": 0.9995792426367461,
        "median_improvement_factor": 28.041967376426307,
    }
    failures = [row["cell_id"] for row in family["rows"] if not row["passed"]]
    assert len(failures) == 12
    assert "half/all_gather/r8/i07" in failures
    assert max(row["after_relative_error"] for row in family["rows"]) == (
        0.4180805668016194
    )


def test_family_b_is_the_generated_byte_exact_bypass(record: dict) -> None:
    family = record["families"]["B"]
    assert family["status"] == "PASS"
    assert family["golden_generating_commit"] == (
        "06fc199783e364c2eaa6a7c917a1f9f2c84d79ac"
    )
    assert family["golden_file_sha256"] == (
        "1303b6bffa6f345dad6b374e1507314fe18c9b895cc03c85f12aa16e76a2616b"
    )
    assert family["golden_record_sha256"] == family[
        "post_wave_default_off_sha256"
    ]
    assert family["golden_record_sha256"] == (
        "ac952c0c0f3e9f427fb892711d716c1f93d826a86faabca70221b7b767f03f2d"
    )
    assert family["first_divergent_field"] is None
    assert family["checked_fields"] == [
        "phase and step timestamps",
        "local and fabric segment tuples",
        "application and wire byte counts",
        "completion order",
        "backend invocation order",
        "random-generator state",
    ]
    plans = family["observed"]["plans"]
    assert [plan["application_bytes"]["total_directed_bytes"] for plan in plans] == [
        1_572_864,
        24_576,
    ]
    assert [plan["application_bytes"]["local_directed_bytes"] for plan in plans] == [
        786_432,
        12_288,
    ]
    assert [plan["application_bytes"]["fabric_directed_bytes"] for plan in plans] == [
        786_432,
        12_288,
    ]
    assert family["observed"]["wire_bytes"]["fabric_goal_send_bytes"] == 798_720
    assert family["observed"]["completion_order"] == [0, 1]
    assert len(family["observed"]["backend_invocation_order"]) == 24
    rng = family["observed"]["random_generator_state"]
    assert rng["before"] == rng["after"]


def test_family_d8_refutes_at_the_matched_coordinate(record: dict) -> None:
    family = record["families"]["D8"]
    assert family["status"] == "REFUTED"
    assert family["passed"] == 0
    assert family["before_quotient"] == 0.02590463307406155
    assert family["physical_endpoint_query_bytes"] == 172_032
    assert family["physical_endpoint_packet_ms"] == 2.06052353
    assert family["physical_endpoint_quotient"] == 1.072044707473791
    assert family["matched_operation_buffer_elements"] == 98_304
    assert family["matched_operation_buffer_bytes"] == 196_608
    assert family["calibrated_packet_ms"] == 2.1318284
    assert family["calibrated_quotient"] == 1.1091430503889075
    assert family["attempt_0002_wrong_query_bytes"] == 344_064
    assert family["calibrated_quotient"] > family["band"][1]
    widths = {row["expert_parallel"]: row for row in family["widths"]}
    assert widths[32]["calibrated_dispatch_combine_ms"] == 8.64087354
    assert widths[128]["calibrated_dispatch_combine_ms"] == 31.16311354
    for width in (32, 128):
        assert not widths[width]["scored"]
        assert widths[width]["execution_mode"] == (
            "parallel-independent-semantic-halves"
        )
        assert all(
            phase["evidence_class"] == "transferred-at-use"
            for phase in widths[width]["phases"]
        )


def test_family_m_moves_ttft_and_tpot_by_the_published_arithmetic(
    record: dict,
) -> None:
    family = record["families"]["M"]
    assert family["status"] == "PASS"
    assert family["off_reproduces_feature_absent"]
    assert family["ttft_delta_ps"] == 2_634_233_088
    assert family["tpot_delta_ps"] == 731_433_664
    assert family["tpot_delta_ps"] == family["expected_tpot_delta_ps"]
    assert [row["observed_delta_ps"] for row in family["step_arithmetic"]] == [
        2_634_233_088,
        731_433_664,
        731_433_664,
    ]
    assert all(row["equation_holds"] for row in family["step_arithmetic"])


def test_family_w_uses_the_slower_evaluation_and_passes(record: dict) -> None:
    family = record["families"]["W"]
    assert family["status"] == "PASS"
    assert family["wall_time_seconds"] == 358.23272075131536
    assert family["wall_time_seconds"] <= family["bound_seconds"] == 600.0


def test_families_remain_separate_and_are_never_summed(record: dict) -> None:
    assert record["family_tallies"] == {
        "B": {"denominator": 1, "passed": 1, "status": "PASS"},
        "D8": {"denominator": 1, "passed": 0, "status": "REFUTED"},
        "H": {"denominator": 63, "passed": 51, "status": "REFUTED"},
        "M": {"denominator": 1, "passed": 1, "status": "PASS"},
        "W": {"denominator": 1, "passed": 1, "status": "PASS"},
    }
    assert "total" not in record["family_tallies"]


def test_record_csv_and_report_digests_are_current(record: dict) -> None:
    assert hashlib.sha256(RECORD_PATH.read_bytes()).hexdigest() == RECORD_SHA256
    assert hashlib.sha256(CSV_PATH.read_bytes()).hexdigest() == CSV_SHA256
    assert b"\r" not in RECORD_PATH.read_bytes()
    assert b"\r" not in CSV_PATH.read_bytes()
    with CSV_PATH.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 67
    report = RESULTS_PATH.read_text(encoding="utf-8")
    assert RECORD_SHA256 in report
    assert CSV_SHA256 in report
    assert "TRAF-77 is untouched" in report
    assert "No task closes" in report
    assert record["chronology"]["attempt_0001"]["status"] == "VOID"
    assert record["chronology"]["attempt_0002"]["status"] == "SUPERSEDED"
    assert record["chronology"]["attempt_0003"]["status"] == "VOID"
    assert record["chronology"]["attempt_0004"]["status"] == "INTERPRETABLE"


def test_runner_checks_the_committed_artifacts_without_pythonpath() -> None:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, os.fspath(RUNNER), "--check"],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert RECORD_SHA256 in completed.stdout


@pytest.mark.parametrize(
    ("environment_name", "record_key", "reason"),
    [
        (
            "SIMLLM_HTSIM_RNIC",
            "htsim",
            "Family B and mixed D8 artifact identity require the pinned htsim_rnic",
        ),
        (
            "SIMLLM_TXT2BIN",
            "txt2bin",
            "Family B and mixed D8 artifact identity require the pinned txt2bin",
        ),
    ],
)
def test_native_tool_identity_when_environment_is_available(
    record: dict,
    environment_name: str,
    record_key: str,
    reason: str,
) -> None:
    raw = os.environ.get(environment_name)
    if not raw:
        pytest.skip(reason)
    path = Path(raw)
    if not path.is_file():
        pytest.skip(f"{reason}; configured path is unavailable")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record["environment"][
        record_key
    ]["sha256"]


def test_sdk_axis_lines_when_external_environment_is_available() -> None:
    raw = os.environ.get("SIMLLM_EXTERNAL_AIC_VENV")
    if not raw:
        pytest.skip("corrected ELEMENT axis citations require the pinned SDK environment")
    roots = sorted(Path(raw).glob("lib/python*/site-packages/aiconfigurator_core/sdk"))
    if not roots:
        pytest.skip("corrected ELEMENT axis citations require installed pinned SDK sources")
    communication = (roots[0] / "operations/communication.py").read_text(
        encoding="utf-8"
    ).splitlines()
    common = (roots[0] / "common.py").read_text(encoding="utf-8").splitlines()
    assert "_num_elements_per_token" in communication[515]
    assert "dtype.value.memory * message_size" in communication[394]
    assert "half = QuantMapping(2" in common[1111]
    assert "int8 = QuantMapping(1" in common[1112]
