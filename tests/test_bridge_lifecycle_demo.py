import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "examples" / "bridge_lifecycle_v1" / "run_study.py"


def test_bridge_lifecycle_check_only_creates_no_artifacts(tmp_path):
    out = tmp_path / "check-only-output"
    environment = os.environ.copy()
    environment["SIMLLM_WAVE5_RUN_ROOT"] = str(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--out",
            str(out),
            "--real-htsim",
            sys.executable,
            "--real-txt2bin",
            sys.executable,
            "--check-only",
        ],
        cwd=REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(completed.stdout)
    observed_gitlink = subprocess.run(
        ["git", "rev-parse", "HEAD:third_party/htsim"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    assert report["artifacts_created"] is False
    assert report["authoring_htsim_commit"] == (
        "4885c647eecdfdf81479d1df052223c016ad086b"
    )
    assert report["observed_htsim_gitlink"] == observed_gitlink
    assert not out.exists()
