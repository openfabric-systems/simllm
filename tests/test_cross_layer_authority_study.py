"""Study-entrypoint regression checks for the cross-layer authority."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

STUDY = "examples/cross_layer_authority_v1/run_study.py"


def test_check_only_validates_the_frozen_table_without_writing(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    output = tmp_path / "must-not-exist"

    result = subprocess.run(
        [sys.executable, str(repository / STUDY), "--out", str(output), "--check-only"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "13 registered contradictions" in result.stdout
    assert not output.exists()


def test_study_reproduces_the_frozen_sweep_and_rejects_every_contradiction(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    output = tmp_path / "run"

    subprocess.run(
        [sys.executable, str(repository / STUDY), "--out", str(output)],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["failures"] == []
    assert summary["scored"]["CLA-B1_contradictions_rejected"] == "13/13"
    assert summary["scored"]["CLA-B2_sweep_cells_matching_frozen_jct"] == "4/4"
    assert {cell["completed_at_ps"] for cell in summary["cells"]} == {
        296_980,
        586_260,
        297_780,
        587_060,
    }
