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
STUDY = ROOT / "examples/minimax_ep_scaling_v1"
RECORD = STUDY / "record.json"
RESULTS_CSV = STUDY / "results.csv"
RECORD_SHA256 = "6f980eaea513bb532723e5a0cd66740002a5f7d4b3c78317c95c745aa0921f68"
RESULTS_CSV_SHA256 = "b806306e2b2bc9ff81f4a0895fbc5694845e423fd37171d32c45ff98a5250467"
PNG_SHA256 = "e7c2b72fa85bf4cd9bf1169e6371bb164382e39016c0613d3e8029d9c81ebf85"
PDF_SHA256 = "3efe36eda7d08706176e6169162827e8cfcb7e15e9d3d3d913df46ffaebd01f1"
WIDTHS = (8, 32, 128, 256)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_tracked_minimax_record_is_locked_and_nonvoid() -> None:
    assert _sha256(RECORD) == RECORD_SHA256
    assert _sha256(RESULTS_CSV) == RESULTS_CSV_SHA256
    assert _sha256(STUDY / "figures/minimax_ep_scaling.png") == PNG_SHA256
    assert _sha256(STUDY / "figures/minimax_ep_scaling.pdf") == PDF_SHA256
    assert b"\r" not in RECORD.read_bytes()
    assert b"\r" not in RESULTS_CSV.read_bytes()

    record = _record()
    assert record["schema"] == "simllm-minimax-ep-scaling-record-v1"
    assert record["run_state"] == "nonvoid"
    assert record["attempt"] == "attempt-0002"
    assert record["run_commit"] == "df47c6532c71312d36eb96ed528f6ebd772e5952"
    assert record["freeze_commits"] == ["61b66c4", "5a29bb0"]
    assert all(record["fatal_guards"].values())
    assert record["fresh_evaluations"] == {
        "bit_equal": True,
        "count": 2,
        "first_sha256": "31c092cf2e1264c55820b58a2d942c428c2c33f5414026952357ccadb725c461",
        "second_sha256": "31c092cf2e1264c55820b58a2d942c428c2c33f5414026952357ccadb725c461",
    }

    families = record["family_tallies"]
    assert {
        name: (entry["passed"], entry["denominator"])
        for name, entry in families.items()
    } == {"E": (4, 4), "C": (4, 4), "N": (1, 3), "W": (1, 1)}
    assert families["W"]["elapsed_seconds"] == 298.81469979602844


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


def test_network_refutations_and_fanin_are_locked() -> None:
    record = _record()
    rows = record["rows"]
    assert [row["expert_parallel"] for row in rows] == list(WIDTHS)
    assert [row["ratio"] for row in rows] == [
        0.8643398194341548,
        0.4262782480503487,
        0.3048657016451342,
        0.2742607736975033,
    ]
    assert [row["packet_dispatch_combine_ms"] for row in rows] == [
        0.02496,
        4.0350336,
        5.5890432,
        7.1043648,
    ]
    assert [row["peer_subset"] for row in rows] == [False, False, False, True]
    assert rows[-1]["simulated_message_fraction"] == 0.125
    assert rows[-1]["simulated_messages_per_sampled_layer"] == 16_320
    assert rows[-1]["represented_messages"] == 8_486_400
    assert rows[-1]["represented_bytes"] == 3_258_777_600

    bands = record["family_tallies"]["N"]["bands"]
    assert bands["N1"]["passed"] is False
    assert bands["N2"] == {
        "actual": 0.2742607736975033,
        "lower": 1.25,
        "passed": False,
        "rule": "widest ratio is at least 1.25",
    }
    assert record["n3"]["maximum_receiver_ingress_occupancy_ps"] == 54_648_960
    assert record["n3"]["maximum_simultaneous_senders_per_receiver"] == 248
    assert all(phase["sender_count"] == 248 for phase in record["n3"]["phases"])


def test_physical_ledger_and_portable_paths_are_locked() -> None:
    record = _record()
    assert record["physical_sanity"] == {
        "dispatch_plus_combine_bytes_per_rank": 195_840,
        "full_rank_serialization_floor_microseconds": 3.9168000000000003,
        "link_bytes_per_second": 50_000_000_000,
        "routed_fp8_payload_per_rank_pair_bytes": 384,
        "widest_expert_parallel": 256,
    }
    assert record["bulk_evidence"] == (
        "${SIMLLM_MINIMAX_T1_BULK_ROOT}/attempt-0002"
    )
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
    assert rows[-1]["sample_label"].startswith("sampled layer and peers:")


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
