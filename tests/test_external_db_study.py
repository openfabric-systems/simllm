from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "examples/external_db_parity_v1"
ARTIFACT = (
    ROOT
    / "offline/calibration/external-databases"
    / "85e72f990f00ea457de522d0b773e678f5e067740689912df5646f6296273284"
)


def test_local_worker_runs_directly_without_pythonpath(tmp_path: Path) -> None:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [
            sys.executable,
            os.fspath(STUDY / "run_study.py"),
            "--worker",
            "local",
            "--artifact",
            os.fspath(ARTIFACT),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    result = json.loads(completed.stdout)
    assert result["row_count"] == 284717
    assert len(result["i2"]) == 26
    assert len(result["p1"]) == 4
    assert result["evidence"]["all_match"] is True
