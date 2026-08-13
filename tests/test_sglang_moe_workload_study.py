"""Regression lock for the retained void SGLang MoE workload study."""

import json
import os
import subprocess
import sys
from pathlib import Path


def test_sglang_moe_workload_study_retains_the_refuted_fatal_guard(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repository)
    completed = subprocess.run(
        [
            sys.executable,
            str(repository / "examples/sglang_moe_workload_v1/run_study.py"),
            "--run-dir",
            str(tmp_path),
        ],
        cwd=repository,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 1
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "VOID"
    assert summary["scored_passed_retained"] == summary["scored_total_retained"] == 10
    assert summary["fatal_guards"] == 42
    failed = [guard for guard in summary["guards"] if not guard["passed"]]
    assert [guard["guard"] for guard in failed] == [
        "workload-short-length-trace-rejected"
    ]
    assert len(summary["post_specified_component_sanity"]) == 4
