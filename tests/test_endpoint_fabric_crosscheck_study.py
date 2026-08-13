"""Study-entrypoint regression checks for the endpoint versus fabric cross-check."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

STUDY = "examples/endpoint_fabric_crosscheck_v1/run_study.py"


def test_check_only_validates_registry_without_inspecting_paths_or_writing(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    output = tmp_path / "must-not-exist"
    missing = tmp_path / "also-missing"

    result = subprocess.run(
        [
            sys.executable,
            str(repository / STUDY),
            "--out",
            str(output),
            "--source-root",
            str(missing),
            "--htsim-rnic",
            str(missing / "htsim_rnic"),
            "--txt2bin",
            str(missing / "txt2bin"),
            "--check-only",
        ],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "two scored families and two entailed fatal families" in result.stdout
    assert "no artifacts produced" in result.stdout
    assert not output.exists()
