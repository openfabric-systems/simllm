import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "examples" / "a100_nvlink_packet_v1"
FREEZE_SHA256 = "212a7a26f54e444c9b18f1e528bd0d00b5a28e4f9e005b0dc137f477ad642571"


def run_study(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        (sys.executable, str(STUDY / "run_study.py"), *arguments),
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )


def test_freeze_digest_and_catalog_are_immutable():
    digest = hashlib.sha256((STUDY / "expectations.json").read_bytes()).hexdigest()
    freeze = json.loads((STUDY / "expectations.json").read_text())

    assert digest == FREEZE_SHA256
    assert len(freeze["catalog"]) == 80
    assert [case["ordinal"] for case in freeze["catalog"]] == list(range(1, 81))
    assert len({case["stable_name"] for case in freeze["catalog"]}) == 80
    assert all("expected_band" in case for case in freeze["catalog"])
    assert all(set(case["identifies"]) == {"tx", "switch", "rx"} for case in freeze["catalog"])


def test_cell_registry_has_80_isolated_5_corner_and_1_all_frame():
    completed = run_study("--list-cells")

    assert completed.returncode == 0, completed.stderr
    cells = json.loads(completed.stdout)
    assert len(cells) == 86
    assert [cell["index"] for cell in cells] == list(range(86))
    assert sum(cell["frame"] == "isolated" for cell in cells) == 80
    assert sum(cell["frame"] == "corner_frame" for cell in cells) == 5
    assert cells[-1]["cell_id"] == "all-corners-frame"
    assert len(cells[-1]["case_names"]) == 80


def test_empty_hardware_root_reports_every_cell_pending(tmp_path):
    completed = run_study(
        "--pending-indices",
        "--output-root",
        str(tmp_path),
        "--expected-head",
        "0123456789abcdef0123456789abcdef01234567",
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip().split(",") == [str(index) for index in range(86)]


def test_dry_run_expands_every_payload_byte_and_three_producers(tmp_path):
    completed = run_study(
        "--mode",
        "mock",
        "--binary",
        str(tmp_path / "not-built-in-dry-run"),
        "--output-root",
        str(tmp_path),
        "--array-index",
        "0",
        "--dry-run",
    )

    assert completed.returncode == 0, completed.stderr
    plan = json.loads(completed.stdout)
    assert plan["freeze_sha256"] == FREEZE_SHA256
    assert plan["cells"][0]["cell_id"] == "isolated-001"
    assert plan["cells"][0]["point_count"] == 512 * 3
    assert not list(tmp_path.iterdir())


@pytest.fixture(scope="module")
def mock_binary(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler is unavailable")
    executable_suffix = ".exe" if os.name == "nt" else ""
    output = tmp_path_factory.mktemp("traf65-build") / f"nvlink-packet-mock{executable_suffix}"
    completed = subprocess.run(
        (
            compiler,
            "-x",
            "c++",
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Wpedantic",
            "-DSIMLLM_NVLINK_MOCK",
            str(STUDY / "nvlink_packet_lane.cu"),
            "-o",
            str(output),
        ),
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return output


def test_mock_cell_is_digest_complete_and_resumable(tmp_path, mock_binary):
    arguments = (
        "--mode",
        "mock",
        "--binary",
        str(mock_binary),
        "--output-root",
        str(tmp_path),
        "--array-index",
        "14",
    )
    first = run_study(*arguments)
    second = run_study(*arguments)

    assert first.returncode == 0, first.stderr
    assert "complete" in first.stdout
    assert second.returncode == 0, second.stderr
    assert "already complete and digest verified" in second.stdout

    cell = tmp_path / FREEZE_SHA256 / "cells" / "isolated-015"
    attempts = list(cell.glob("attempt-*"))
    assert len(attempts) == 1
    attempt = attempts[0]
    manifest = json.loads((attempt / "manifest.json").read_text())
    complete = json.loads((attempt / "COMPLETE.json").read_text())
    manifest_digest = hashlib.sha256((attempt / "manifest.json").read_bytes()).hexdigest()
    assert complete["manifest_sha256"] == manifest_digest
    for payload in manifest["payloads"]:
        path = attempt / payload["path"]
        assert path.stat().st_size == payload["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == payload["sha256"]

    rows = [
        json.loads(line) for line in (attempt / "results.jsonl").read_text().splitlines() if line
    ]
    assert {row["producer"] for row in rows} == {
        "persistent_sm_peer_write",
        "dependent_sm_peer_read",
        "copy_engine_reference",
    }
    assert all(row["mode"] == "mock" for row in rows)
    assert all(row["measurement_claim"] is False for row in rows)
    assert all(row["checksum_ok"] is True for row in rows)


def test_candidate_handoff_and_submission_are_pinned():
    profile = json.loads((STUDY / "candidate-profile.json").read_text())
    sbatch = (STUDY / "run_merlin_cell.sbatch").read_text()
    source = (STUDY / "nvlink_packet_lane.cu").read_text()

    assert profile["freeze_sha256"] == FREEZE_SHA256
    assert profile["status"] == "candidate"
    assert profile["handoff"]["measurement_claim"] is False
    assert "#SBATCH --partition=a100-hourly" in sbatch
    assert "#SBATCH --gres=gpu:4" in sbatch
    assert "#SBATCH --array=0-85%1" in sbatch
    assert FREEZE_SHA256 in sbatch
    assert "persistent_peer_write" in source
    assert "dependent_peer_read" in source
    assert "cudaMemcpyPeerAsync" in source
    assert "ncclSend" in source and "ncclRecv" in source


def test_tracked_local_validation_makes_no_hardware_claim():
    result = json.loads((STUDY / "local-validation.json").read_text())

    assert result["status"] == "VALID_LOCAL_MOCK_86_OF_86"
    assert result["task_status"] == "OPEN"
    assert result["freeze_sha256"] == FREEZE_SHA256
    assert result["cell_count"] == 86
    assert result["isolated_cell_count"] == 80
    assert result["corner_frame_count"] == 5
    assert result["all_corners_frame_count"] == 1
    assert result["result_row_count"] == 14_035
    assert result["measurement_claim"] is False
    assert result["hardware_remainder"] == {
        "available_after": "2026-08-28T06:30",
        "reason": "Merlin maintenance reservation SD26082026",
        "status": "not_run",
    }
