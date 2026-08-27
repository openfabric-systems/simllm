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
EXECUTION_HEAD = "2ab092f9255d77c00c547446b65534a3b273ec82"
PRODUCER_BINARY_SHA256 = "96b4c544de54457d1fbed8e56b0a1cbe61344bcdab02d6445c07a0ab637277a4"
PROTECTED_CANDIDATE_PROFILE_SHA256 = (
    "899712c4734f7a6b410d80231291663a404511528d46aab7497b73831e0e354f"
)
PUBLISHED_CANDIDATE_PROFILE_SHA256 = (
    "d33ef5b2c6fa87cc97e1e7b45a43a841a5da45f5462311e3981fbc903c56deb2"
)
TRAF70_FREEZE_SHA256 = (
    "f0ab026e054873a56614af63ab3a7ae3219dc0b045423808cb41522910fa6da6"
)


def run_study(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        (sys.executable, str(STUDY / "run_study.py"), *arguments),
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )


def run_score(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        (sys.executable, str(STUDY / "score_hardware.py"), *arguments),
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )


def write_text_lf(path: Path, text: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def write_json_lf(path: Path, payload: object) -> None:
    write_text_lf(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def make_hardware_attempt(bulk_root: Path) -> Path:
    attempt = bulk_root / FREEZE_SHA256 / "cells" / "isolated-001" / "attempt-0001"
    attempt.mkdir(parents=True)
    plan = {
        "schema": "simllm-a100-nvlink-packet-cell-v1",
        "cell": {"index": 0},
        "mode": "hardware",
        "freeze_sha256": FREEZE_SHA256,
        "candidate_profile_sha256": (
            "899712c4734f7a6b410d80231291663a404511528d46aab7497b73831e0e354f"
        ),
        "implementation_sha256": (
            "af6801d25f105b612dfa5ca475f33d03d1306bf0e3c80c72089310d0de53b643"
        ),
        "producer_binary_sha256": PRODUCER_BINARY_SHA256,
        "expected_head": EXECUTION_HEAD,
        "point_count": 1,
    }
    environment = {
        "schema": "simllm-a100-nvlink-packet-cell-v1",
        "mode": "hardware",
        "source_head": EXECUTION_HEAD,
        "slurm_partition": "a100-hourly",
    }
    summary = {
        "status": "hardware_unscored",
        "row_count": 1,
    }
    result = {
        "schema": "simllm-a100-nvlink-packet-observation-v1",
        "mode": "hardware",
        "case_name": "CORNER_NVPKT_001_payload_bytes",
        "point_id": "CORNER_NVPKT_001_payload_bytes:bytes=256:copy_engine_reference",
        "producer": "copy_engine_reference",
        "payload_rate_gbps": 94.01,
        "checksum_ok": True,
        "measurement_claim": "unscored",
    }
    write_json_lf(attempt / "plan.json", plan)
    write_json_lf(attempt / "environment.json", environment)
    write_json_lf(attempt / "summary.json", summary)
    write_text_lf(attempt / "results.jsonl", json.dumps(result, sort_keys=True) + "\n")
    for name, text in (
        ("guards_before.txt", "qualified\n"),
        ("guards_after.txt", "qualified\n"),
        ("points.tsv", "case_name\nCORNER_NVPKT_001_payload_bytes\n"),
        ("stderr.txt", ""),
        ("stdout.txt", ""),
    ):
        write_text_lf(attempt / name, text)
    payloads = []
    for path in sorted(attempt.iterdir()):
        payloads.append(
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    manifest = {
        "schema": "simllm-a100-nvlink-packet-attempt-manifest-v1",
        "cell_id": "isolated-001",
        "freeze_sha256": FREEZE_SHA256,
        "payloads": payloads,
    }
    write_json_lf(attempt / "manifest.json", manifest)
    write_json_lf(
        attempt / "COMPLETE.json",
        {
            "schema": "simllm-a100-nvlink-packet-attempt-manifest-v1",
            "status": "complete",
            "cell_id": "isolated-001",
            "manifest_sha256": hashlib.sha256((attempt / "manifest.json").read_bytes()).hexdigest(),
        },
    )
    return attempt


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


def test_scored_handoff_and_historical_submission_are_pinned():
    profile_path = STUDY / "candidate-profile.json"
    profile = json.loads(profile_path.read_text())
    sbatch = (STUDY / "run_merlin_cell.sbatch").read_text()
    source = (STUDY / "nvlink_packet_lane.cu").read_text()

    assert profile["freeze_sha256"] == TRAF70_FREEZE_SHA256
    assert profile["status"] == "scored_mixed_parameter_evidence"
    assert profile["handoff"]["measurement_claim"] is True
    assert "hardware_scoring" not in profile
    assert (
        profile["traf70_score_publication"]["protected_candidate_before_sha256"]
        == PROTECTED_CANDIDATE_PROFILE_SHA256
    )
    assert (
        hashlib.sha256(profile_path.read_bytes()).hexdigest()
        == PUBLISHED_CANDIDATE_PROFILE_SHA256
    )
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


def test_tracked_hardware_score_publishes_the_complete_void_result():
    score = json.loads((STUDY / "hardware-score.json").read_text())

    assert score["status"] == "COMPLETE_VOID_86_OF_86"
    assert score["task_status"] == "OPEN"
    assert score["freeze_sha256"] == FREEZE_SHA256
    assert score["coverage"]["completed_indices"] == list(range(86))
    assert score["coverage"]["completed_prefix_indices"] == list(range(86))
    assert score["coverage"]["pending_array"] == ""
    assert score["coverage"]["rejected_attempts"] == []
    assert score["coverage"]["result_row_count"] == 14_035
    assert score["coverage"]["protocol_validation_row_count"] == 4
    assert [corner["verdict"] for corner in score["corner_verdicts"]] == [
        "UNSCORABLE_RUN_VOID",
        "MEASURED_BAND_REFUTED_BUT_RUN_VOID",
        "MEASURED_BAND_REFUTED_BUT_RUN_VOID",
        "UNSCORABLE_RUN_VOID",
        "UNSCORABLE_RUN_VOID",
    ]
    assert score["producer_binary_audit"]["status"] == (
        "BUILD_REPRODUCIBILITY_MISMATCH"
    )
    assert score["capture_contract_audit"]["status"] == (
        "REFUTED_AS_IDENTIFICATION_CAPTURE"
    )
    assert score["candidate_profile_decision"]["parameter_value_changes"] == []


def test_hardware_scorer_reports_verified_prefix_and_exact_remainder(tmp_path):
    make_hardware_attempt(tmp_path)
    score_path = tmp_path / "lean" / "score.json"
    report_path = tmp_path / "lean" / "RESULTS.md"
    candidate_before = (STUDY / "candidate-profile.json").read_bytes()
    completed = run_score(
        "--bulk-root",
        str(tmp_path),
        "--json-out",
        str(score_path),
        "--markdown-out",
        str(report_path),
        "--scheduler-job",
        "198968",
    )

    assert completed.returncode == 0, completed.stderr
    score = json.loads(score_path.read_text())
    assert score["status"] == "PARTIAL_VOID_1_OF_86"
    assert score["measurement_validity"] == "VOID_FATAL_GUARD_COVERAGE_INCOMPLETE"
    assert score["coverage"]["completed_prefix_indices"] == [0]
    assert score["coverage"]["pending_array"] == "1-85"
    assert score["case_scores"][0]["metric_score"]["status"] == "UNSCORABLE"
    assert score["candidate_profile_decision"]["parameter_value_changes"] == []
    capture = score["capture_contract_audit"]
    assert capture["status"] == "REFUTED_AS_IDENTIFICATION_CAPTURE"
    assert "outstanding" in capture["parsed_but_not_applied_hardware_controls"]
    assert capture["copy_engine_batch_contract"].startswith("REFUTED")
    report = report_path.read_text()
    assert "Per-corner verdicts" in report
    assert "Module-parameter identification" in report
    assert "reservation lifted early" in report
    assert (STUDY / "candidate-profile.json").read_bytes() == candidate_before


def test_hardware_scorer_rejects_a_changed_payload(tmp_path):
    attempt = make_hardware_attempt(tmp_path)
    with open(attempt / "results.jsonl", "a", encoding="utf-8", newline="\n") as handle:
        handle.write("{}\n")
    score_path = tmp_path / "score.json"
    completed = run_score(
        "--bulk-root",
        str(tmp_path),
        "--json-out",
        str(score_path),
    )

    assert completed.returncode == 0, completed.stderr
    score = json.loads(score_path.read_text())
    assert score["status"] == "PENDING_0_OF_86"
    assert score["coverage"]["completed_indices"] == []
    assert score["coverage"]["pending_array"] == "0-85"
    assert score["coverage"]["rejected_attempts"][0]["reason"].startswith(
        "payload digest mismatch"
    )
