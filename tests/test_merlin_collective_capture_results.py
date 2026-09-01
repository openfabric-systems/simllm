"""Locked publication tests for the TRAF-77 Merlin collective capture."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "examples" / "merlin_collective_capture_v1"
RUNNER = STUDY / "run_study.py"
RECORD = STUDY / "record.json"
RESULTS = STUDY / "results.csv"
EVIDENCE_ENV = "SIMLLM_TRAF77_EVIDENCE_ROOT"
EVIDENCE_SKIP_REASON = (
    "TRAF-77 fetched hardware evidence is absent; set "
    "SIMLLM_TRAF77_EVIDENCE_ROOT to run evidence-dependent reproduction"
)


def _record() -> dict:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _csv_rows() -> list[dict[str, str]]:
    with RESULTS.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _evidence_root_or_skip() -> Path:
    configured = os.environ.get(EVIDENCE_ENV)
    if not configured:
        pytest.skip(EVIDENCE_SKIP_REASON)
    root = Path(configured)
    if not root.is_dir():
        pytest.skip(f"{EVIDENCE_SKIP_REASON}; configured directory is unavailable")
    return root


def test_tracked_record_locks_nonvoid_run_and_cell_scoped_family_outcomes() -> None:
    record = _record()
    assert record["schema"] == "simllm-merlin-collective-scored-record-v1"
    assert record["run_state"] == "nonvoid"
    assert record["verdict"] == (
        "NONVOID_FG_2_CELL_SCOPED_CONCENTRATION_CONTROL_REFUTED"
    )
    assert {key: row["status"] for key, row in record["fatal_guards"].items()} == {
        "FG-1": "PASS",
        "FG-2": "FAIL",
        "FG-3": "PASS",
        "FG-4": "PASS",
        "FG-5": "PASS",
        "FG-6": "PASS",
        "FG-7": "PASS",
    }
    assert record["families"]["C"]["status"] == "PASS"
    assert record["families"]["C"]["passed"] == 3
    assert record["fatal_guards"]["FG-2"]["cell_status_counts"] == {
        "CONTRADICTED": 317,
        "INSUFFICIENT-SIGNAL": 35,
        "PROVEN": 0,
    }
    assert record["families"]["R"]["status"] == "UNEVALUABLE"
    assert record["families"]["R"]["denominator"] == 0
    assert record["families"]["R"]["scoreable_rows"] == 0
    assert record["families"]["R"]["void_rows"] == 129
    assert record["families"]["R"]["unevaluable_rows"] == 21
    assert record["families"]["L"]["status"] == "PASS"
    assert record["families"]["L"]["passed"] == 352
    assert record["families"]["W"]["status"] == "PASS"


def test_tracked_record_locks_capture_identity_and_anchor_values() -> None:
    record = _record()
    record_text = RECORD.read_text(encoding="utf-8")
    assert "/data3/" not in record_text
    assert "/usr/" not in record_text
    assert record["evidence"]["normalized_sha256"] == (
        "80a7852b42ad756493b1bdc1d91f314f766483d9f937823b30f64d219334d6aa"
    )
    assert record["evidence"]["normalized_outputs_byte_identical"]
    anchors = record["families"]["C"]["rows"]
    assert [row["observed_over_anchor"] for row in anchors] == [
        1.504778218291071,
        1.410986414648553,
        1.4075719614720783,
    ]
    assert all(not row["path"].startswith("/") for row in record["evidence"]["files"])
    assert record["task_effect"]["TRAF-77"] == "NARROWED, remains open"
    assert record["task_effect"]["TRAF-87"].startswith("OPENED")


def test_tracked_record_locks_socket_mechanism_and_achieved_ports() -> None:
    record = _record()
    mechanisms = {
        row["attempt_id"]: row for row in record["mechanism"]["attempts"]
    }
    assert mechanisms["w2_one-port_job202415"]["selected_network_plugins"] == [
        "Socket"
    ]
    assert mechanisms["w8_one-port_job202417"]["socket_ifname_settings"] == [
        "=hsn0"
    ]
    assert mechanisms["w8_one-port_job202417"]["gdr_states"] == [0]
    assert mechanisms["w8_one-port_job202417"]["logical_network_connections"] == {
        "collective": 8,
        "shared_p2p": 64,
        "total": 72,
    }
    assert mechanisms["w8_four-port_job202418"]["logical_network_connections"] == {
        "collective": 32,
        "shared_p2p": 64,
        "total": 96,
    }

    summaries = {
        (row["attempt_id"], row["node"]): row
        for row in record["achieved_concentration"]["node_summaries"]
    }
    gpu101 = summaries[("w8_one-port_job202417", "gpu101")]
    gpu105 = summaries[("w8_one-port_job202417", "gpu105")]
    assert gpu101["dominant_rx_port"] == "hsn0"
    assert gpu101["dominant_tx_port"] == "hsn2"
    assert gpu101["tx_total_bytes"] == 262_776_768_014
    assert gpu101["rx_total_bytes"] == 25_527_859_035
    assert gpu105["dominant_rx_port"] == "hsn0"
    assert gpu105["dominant_tx_port"] == "hsn0"
    assert gpu105["tx_total_bytes"] == 262_100_471_362
    assert gpu105["rx_total_bytes"] == 25_595_714_529
    assert len(record["achieved_concentration"]["rows"]) == 32

    direction_verdicts = {
        (row["attempt_id"], row["node"]): (
            row["directional_routing"]["tx"]["status"],
            row["directional_routing"]["rx"]["status"],
        )
        for row in record["achieved_concentration"]["node_summaries"]
    }
    assert direction_verdicts == {
        ("w2_four-port_job202416", "gpu101"): (
            "CONTRADICTED",
            "CONTRADICTED",
        ),
        ("w2_four-port_job202416", "gpu102"): (
            "CONTRADICTED",
            "CONTRADICTED",
        ),
        ("w2_one-port_job202415", "gpu101"): (
            "CONTRADICTED",
            "CONTRADICTED",
        ),
        ("w2_one-port_job202415", "gpu102"): (
            "CONTRADICTED",
            "CONTRADICTED",
        ),
        ("w8_four-port_job202418", "gpu101"): ("CONTRADICTED", "PROVEN"),
        ("w8_four-port_job202418", "gpu105"): ("CONTRADICTED", "PROVEN"),
        ("w8_one-port_job202417", "gpu101"): ("CONTRADICTED", "PROVEN"),
        ("w8_one-port_job202417", "gpu105"): ("PROVEN", "PROVEN"),
    }

    cell_routing = record["achieved_concentration"]["cell_routing"]
    assert cell_routing["signal_scope"] == "node-level TX-plus-RX total"
    assert "w8/four-port/all_gather/65536" in cell_routing[
        "cell_ids_by_status"
    ]["CONTRADICTED"]


def test_results_csv_separates_scored_void_and_unevaluable_evidence() -> None:
    rows = _csv_rows()
    assert len(rows) == 506
    counts = {}
    for family in ("C", "R/E1", "R/E2", "R/E3", "R/E4", "L", "W"):
        counts[family] = sum(row["family"] == family for row in rows)
    assert counts == {
        "C": 3,
        "R/E1": 88,
        "R/E2": 44,
        "R/E3": 12,
        "R/E4": 6,
        "L": 352,
        "W": 1,
    }
    routing_rows = [row for row in rows if row["family"].startswith("R/")]
    assert Counter(row["status"] for row in routing_rows) == {
        "VOID": 129,
        "UNEVALUABLE": 21,
    }
    assert all(
        "forbids relabeling" in row["reason"]
        for row in routing_rows
        if row["status"] == "VOID"
    )
    assert all(
        "1 MiB signal minimum" in row["reason"]
        for row in routing_rows
        if row["status"] == "UNEVALUABLE"
    )
    assert {row["status"] for row in rows if row["family"] == "L"} == {"PASS"}


def test_runner_reproduces_without_pythonpath_when_evidence_is_available() -> None:
    evidence_root = _evidence_root_or_skip()
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--evidence-root",
            str(evidence_root),
            "--check",
        ],
        cwd=evidence_root.parent,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stderr
    assert "reproduce byte for byte" in completed.stdout
