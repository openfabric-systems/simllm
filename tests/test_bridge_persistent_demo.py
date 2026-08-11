"""Checks for the BRIDGE-1 registered study command."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
STUDY = REPO_ROOT / "examples" / "bridge_persistent_v1" / "run_study.py"


def test_registered_study_check_only_produces_no_artifacts(tmp_path):
    out = tmp_path / "must-remain-absent"
    environment = os.environ.copy()
    environment.update(
        SIMLLM_HTSIM_RNIC=sys.executable,
        SIMLLM_TXT2BIN=sys.executable,
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(STUDY),
            "--out",
            str(out),
            "--fixtures",
            "vllm,sglang",
            "--workers",
            "4,8",
            "--check-only",
        ],
        cwd=REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    plan = json.loads(completed.stdout)
    assert not plan["artifacts_created"]
    assert [row["fixture"] for row in plan["matrix"]] == ["vllm", "sglang"]
    assert all(
        row["modes"] == ["diagnostic", "prepared-4", "prepared-8"]
        for row in plan["matrix"]
    )
    assert not out.exists()
