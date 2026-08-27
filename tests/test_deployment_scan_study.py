from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from examples.deployment_scan_v1.run_study import (
    EXPECTATIONS_COMMIT,
    ProcessGuard,
    run_study,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STUDY_DIR = REPOSITORY_ROOT / "examples" / "deployment_scan_v1"
RESULT_SHA256 = "4e70a01a1f5c229db8b1807c049ddd35c9632eea42a4dd07b8a18a8c1ffde2f2"
CSV_SHA256 = "e6d3f43fe76f2c23cc2ad099e7b3931d32803ee1de668bfd9b9e320b8e861a37"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_default_scan_artifacts_are_byte_locked() -> None:
    result_path = STUDY_DIR / "results.json"
    csv_path = STUDY_DIR / "results.csv"

    assert _sha256(result_path) == RESULT_SHA256
    assert _sha256(csv_path) == CSV_SHA256
    assert b"\r" not in result_path.read_bytes()
    assert b"\r" not in csv_path.read_bytes()
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["status"] == "PASS"
    assert all(guard["status"] == "PASS" for guard in result["fatal_guards"])


def test_default_scan_rearms_the_zero_subprocess_guard() -> None:
    outer_guard = ProcessGuard()
    with outer_guard:
        result = run_study(implementation_commit="f" * 40)

    assert result["status"] == "PASS"
    assert outer_guard.attempts == []
    process_guard = next(
        guard for guard in result["fatal_guards"] if guard["id"] == "FG-1"
    )
    assert process_guard["status"] == "PASS"
    assert process_guard["interceptions_fired"] == 0
    assert len(result["records"]["analytical"]["candidates"]) == 3
    guards = {guard["id"]: guard for guard in result["fatal_guards"]}
    assert guards["FG-2"]["status"] == "ENFORCED_BY_CONSTRUCTION"
    assert guards["FG-2"]["runtime_evidence"] is False
    assert "mutation negative control was rejected" in guards["FG-4"]["detail"]
    assert guards["FG-6"]["status"] == "VERIFIED_OUT_OF_PROCESS"
    post_check = result["post_specified_regressions"]["checks"][0]
    assert post_check["status"] == "PASS"
    assert post_check["observed"]["floor_division_candidate_ps"] == 177_384_106
    assert post_check["observed"]["rounded_scaling_candidate_ps"] == 177_384_107


def test_expectations_commit_precedes_deployment_implementation() -> None:
    probe = subprocess.run(
        ["git", "cat-file", "-e", f"{EXPECTATIONS_COMMIT}^{{commit}}"],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        pytest.skip(
            f"expectations commit {EXPECTATIONS_COMMIT} is absent from this shallow clone"
        )

    changed = subprocess.run(
        ["git", "show", "--format=", "--name-only", EXPECTATIONS_COMMIT],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert tuple(line for line in changed.stdout.splitlines() if line) == (
        "examples/deployment_scan_v1/expectations.md",
    )
    deploy_tree = subprocess.run(
        ["git", "cat-file", "-e", f"{EXPECTATIONS_COMMIT}:simllm/deploy"],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert deploy_tree.returncode != 0
    assert EXPECTATIONS_COMMIT in (STUDY_DIR / "RESULTS.md").read_text(
        encoding="utf-8"
    )
