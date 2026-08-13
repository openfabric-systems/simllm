"""Study-entrypoint regression checks for request routing lifetimes."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_check_only_validates_registry_without_inspecting_paths_or_writing(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    output = tmp_path / "must-not-exist"
    missing = tmp_path / "also-missing"

    result = subprocess.run(
        [
            sys.executable,
            str(repository / "examples/routing_lifetime_v1/run_study.py"),
            "--out",
            str(output),
            "--source-root",
            str(missing),
            "--check-only",
        ],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "six scored families, 14 scored instances" in result.stdout
    assert "two unscored duplicate views" in result.stdout
    assert "no artifacts produced" in result.stdout
    assert not output.exists()
