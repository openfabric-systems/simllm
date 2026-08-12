"""Study-entrypoint regression checks for per-request replay fidelity."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_check_only_validates_registry_without_inspecting_paths_or_writing(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    output = tmp_path / "must-not-exist"
    missing = tmp_path / "also-missing"

    result = subprocess.run(
        [
            sys.executable,
            str(repository / "examples/per_request_fidelity_v1/run_study.py"),
            "--out",
            str(output),
            "--source-root",
            str(missing),
            "--htsim-rnic",
            str(missing),
            "--txt2bin",
            str(missing),
            "--check-only",
        ],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "no artifacts produced" in result.stdout
    assert not output.exists()
