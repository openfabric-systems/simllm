"""Dry-run contract for the dispatch-sequence study."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_dispatch_sequence_check_only_creates_no_artifacts(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    output = tmp_path / "must-not-exist"
    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "examples" / "dispatch_sequence_v1" / "run_study.py"),
            "--out",
            str(output),
            "--granite-root",
            str(tmp_path / "granite"),
            "--htsim-rnic",
            str(tmp_path / "htsim_rnic"),
            "--txt2bin",
            str(tmp_path / "txt2bin"),
            "--check-only",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "produced no artifacts" in completed.stdout
    assert not output.exists()

def test_dispatch_sequence_v2_check_only_creates_no_artifacts(tmp_path):
    """The TRAF-22 requalification harness must dry-run without side effects."""

    repo_root = Path(__file__).resolve().parents[1]
    output = tmp_path / "must-not-exist"
    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "examples" / "dispatch_sequence_v2" / "run_study.py"),
            "--out",
            str(output),
            "--granite-root",
            str(tmp_path / "granite"),
            "--htsim-rnic",
            str(tmp_path / "htsim_rnic"),
            "--txt2bin",
            str(tmp_path / "txt2bin"),
            "--check-only",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "produced no artifacts" in completed.stdout
    assert not output.exists()


def test_collective_plan_default_check_only_creates_no_artifacts(tmp_path):
    """The TRAF-28 harness must dry-run without side effects."""

    repo_root = Path(__file__).resolve().parents[1]
    output = tmp_path / "must-not-exist"
    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "examples" / "collective_plan_default_v1" / "run_study.py"),
            "--out",
            str(output),
            "--granite-root",
            str(tmp_path / "granite"),
            "--check-only",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "produced no artifacts" in completed.stdout
    assert not output.exists()
