import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "examples" / "nvlink_credit_arbitration_v1"
RESULT_PATH = STUDY / "hardware_identification.json"
MARKDOWN_PATH = STUDY / "RESULTS_HARDWARE.md"
CSV_PATH = STUDY / "aggregate_outstanding_bytes.csv"
SCORER_PATH = STUDY / "score_hardware_identification.py"


def _load_scorer():
    name = "test_traf73_published_hardware_scorer"
    spec = importlib.util.spec_from_file_location(name, SCORER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def result():
    return json.loads(RESULT_PATH.read_text(encoding="utf-8"))


def test_published_result_is_the_literal_void_hardware_record(result):
    assert result["schema"] == "simllm-nvlink-credit-identification-score-v1"
    assert result["task_id"] == "TRAF-73"
    assert result["status"] == "VOID"
    assert result["measurement_validity"] == "VOID_FATAL_GUARD"
    assert result["expectations_commit"] == (
        "f3f2624e7a96efe3ad67eac5940fee8746e40b98"
    )
    assert result["execution_head"] == (
        "f39d24cfec081e3ef17cf0fbe990a92706a18dec"
    )
    assert result["scheduler_jobs"] == {
        "h1": "202778",
        "h2": "202796",
        "h3": "202813",
    }
    assert result["scorer_sha256"] == hashlib.sha256(
        SCORER_PATH.read_bytes()
    ).hexdigest()
    assert result["scoring_audit"]["classification_rules"] == (
        "frozen_aligned_expectations_unchanged"
    )


def test_only_the_frozen_receiver_ceiling_guard_voids_the_score(result):
    guards = result["fatal_guards"]["guards"]
    assert result["fatal_guards"]["verdict"] == "VOID"
    assert [row["id"] for row in guards if row["status"] == "FAIL"] == [
        "FG15"
    ]
    assert all(row["decidable"] is True for row in guards)
    physical = result["fatal_guards"]["physical_sanity"]
    assert physical["status"] == "FAIL"
    assert physical["minimum_completion_over_wire_floor"] == pytest.approx(
        258.363969110444
    )
    assert physical["maximum_completion_over_loose_ceiling"] == pytest.approx(
        7.587253222508061
    )


def test_h1_and_h2_publish_no_identified_knee_or_pool(result):
    h1 = result["h1_credit_window_and_return"]
    assert h1["verdict"] == "VOID"
    assert h1["effective_window_payload_bytes"] is None
    assert h1["effective_return_delay_ps"] is None
    assert len(h1["pair_fits"]) == 12
    assert all(row["payload_bytes"] is None for row in h1["pair_fits"])

    h2 = result["h2_pool_scope"]
    assert h2["verdict"] == "VOID"
    assert h2["aggregate_outstanding_discriminator"] == [
        {
            "sender_count": count,
            "per_sender_knee_payload_bytes": [],
            "aggregate_outstanding_payload_bytes": None,
        }
        for count in (1, 2, 3)
    ]


def test_h3_retains_the_unscored_fair_shape_and_fatal_aggregate(result):
    h3 = result["h3_arbitration"]
    assert h3["verdict"] == "VOID"
    assert h3["identified_policy"] is None
    assert [row["selected_policy"] for row in h3["rotations"]] == [
        "release_aware_round_robin",
        "release_aware_round_robin",
        "release_aware_round_robin",
    ]
    aggregates = [
        row["aggregate_achieved_raw_gbps"] for row in h3["rotations"]
    ]
    assert aggregates == pytest.approx(
        [211.354230784, 213.109096448, 211.908790272]
    )
    assert min(aggregates) > 207.101921876
    small_rates = [
        rate
        for row in h3["rotations"]
        for rate in row["achieved_raw_gbps_by_source_order"][1:]
    ]
    assert min(small_rates) == pytest.approx(58.784913408)
    assert max(small_rates) == pytest.approx(59.545339904)


def test_void_result_registers_no_traf85_promotion(result):
    assert result["traf85_residual"] == {
        "required": False,
        "task_id": None,
        "exact_promotion_cells": [],
    }
    traffic = (ROOT / "docs/modules/traffic.md").read_text(encoding="utf-8")
    assert "TRAF-73" in traffic
    assert "TRAF-85" not in traffic


def test_public_renderings_are_reproducible_and_complete(result):
    scorer = _load_scorer()
    markdown = MARKDOWN_PATH.read_text(encoding="utf-8")
    assert markdown == scorer.render_markdown(result)
    assert "213.109096448 GB/s" in markdown
    assert "TRAF-73 stays open" in markdown
    assert "exact promotion-cell set is empty" in markdown
    assert "+/-" not in markdown
    assert "\N{EM DASH}" not in markdown

    with open(CSV_PATH, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {
            "sender_count": str(count),
            "per_sender_knee_payload_bytes": "",
            "aggregate_outstanding_payload_bytes": "",
        }
        for count in (1, 2, 3)
    ]


def test_raw_evidence_ledger_has_only_the_three_scored_attempts(result):
    evidence = result["raw_evidence"]
    assert {family: row["row_count"] for family, row in evidence.items()} == {
        "h1": 372,
        "h2": 93,
        "h3": 3,
    }
    assert all(
        len(row["attempt_manifest_sha256"]) == 64
        and len(row["row_sha256"]) == 64
        for row in evidence.values()
    )
