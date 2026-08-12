"""Exercise the frozen HTSIM-8 validation-gate controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
FIXTURES = (
    "passing-control",
    "child-exit-defect",
    "zero-completion-defect",
)


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def _source_root() -> Path:
    raw = os.environ.get("HTSIM_SOURCE_ROOT")
    if not raw:
        raise ValueError("HTSIM_SOURCE_ROOT must name the backend source checkout")
    root = Path(raw).resolve()
    required = (
        root / "htsim" / "sim" / "datacenter" / "commit_check.sh",
        root / "htsim" / "sim" / "datacenter" / "validate.py",
    )
    if not all(path.is_file() for path in required):
        raise ValueError(f"HTSIM_SOURCE_ROOT is not an HTSIM checkout: {root}")
    return root


def _write_executable(path: Path, body: str) -> None:
    path.write_text(body, newline="\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _make_fixture(root: Path, name: str) -> Path:
    fixture = root / name
    fixture.mkdir()
    matrix = fixture / "one.cm"
    matrix.write_text("Nodes 2\nConnections 1\n", newline="\n")
    executable = fixture / "simulator.sh"
    if name == "passing-control":
        body = """#!/usr/bin/env bash
printf '%s\n' 'Flow gate-flow flowId 1 src0 finished at 10 total messages 1 total packets 1'
printf '%s\n' 'New: 1 Rtx: 0 RTS: 0'
"""
    elif name == "child-exit-defect":
        body = """#!/usr/bin/env bash
exit 23
"""
    elif name == "zero-completion-defect":
        body = """#!/usr/bin/env bash
printf '%s\n' 'New: 0 Rtx: 0 RTS: 0'
"""
    else:
        raise AssertionError(f"unknown fixture {name}")
    _write_executable(executable, body)
    plan = fixture / "plan.txt"
    plan.write_text(
        "\n".join(
            (
                str(matrix),
                f"!Experiment {name}",
                f"!Binary {executable}",
                "!tailFCT 20",
                "!FCT gate-flow 20",
                "",
            )
        ),
        newline="\n",
    )
    return plan


def _invoke_gate(gate: Path, plan: Path) -> dict[str, Any]:
    run = subprocess.run(
        [str(gate), str(plan)],
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "returncode": run.returncode,
        "stderr": run.stderr,
        "stdout": run.stdout,
    }


def _check_only(args: argparse.Namespace, source_root: Path) -> None:
    plan = {
        "artifacts_created": False,
        "fixtures": FIXTURES,
        "gate": str(source_root / "htsim" / "sim" / "datacenter" / "commit_check.sh"),
        "out": str(args.out),
        "raw_relation_evaluation": "return code before diagnostic parsing",
        "source_root": str(source_root),
    }
    print(json.dumps(plan, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    source_root = _source_root()
    if args.check_only:
        _check_only(args, source_root)
        return
    if args.out.exists():
        parser.error("--out must not exist")
    args.out.mkdir(parents=True)

    gate = source_root / "htsim" / "sim" / "datacenter" / "commit_check.sh"
    fixture_root = Path(tempfile.mkdtemp(prefix="htsim-gate-fixtures-"))
    try:
        observations = {
            name: _invoke_gate(gate, _make_fixture(fixture_root, name)) for name in FIXTURES
        }
    finally:
        shutil.rmtree(fixture_root)

    passing = observations["passing-control"]
    passing_control = {
        "exit_zero": passing["returncode"] == 0,
        "one_completion_reported": "[PASS] Connection count 1" in passing["stdout"],
    }
    fatal_failures = [key for key, passed in passing_control.items() if not passed]

    child = observations["child-exit-defect"]
    zero = observations["zero-completion-defect"]
    relation_rows = [
        {
            "checks": {
                "nonzero_gate_status": child["returncode"] != 0,
                "status_23_named": "status 23" in (child["stdout"] + child["stderr"]),
            },
            "fixture": "child-exit-defect",
            "gate_returncode": child["returncode"],
            "genuine_risk": True,
        },
        {
            "checks": {
                "empty_case_named": (
                    "zero flows completed while 1 were expected"
                    in (zero["stdout"] + zero["stderr"])
                ),
                "no_traceback": "Traceback" not in (zero["stdout"] + zero["stderr"]),
                "no_zero_division": ("ZeroDivisionError" not in (zero["stdout"] + zero["stderr"])),
                "nonzero_gate_status": zero["returncode"] != 0,
            },
            "fixture": "zero-completion-defect",
            "gate_returncode": zero["returncode"],
            "genuine_risk": True,
        },
    ]
    for row in relation_rows:
        row["passed"] = all(row["checks"].values())

    observed_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source_root, text=True
    ).strip()
    summary = {
        "authored_against_htsim_commit": ("fc4400e4ca619223481536632074045cb6af2756"),
        "backend_gate_sha256": hashlib.sha256(gate.read_bytes()).hexdigest(),
        "fatal_unscored": {
            "failures": fatal_failures,
            "passing_control": passing_control,
            "valid": not fatal_failures,
        },
        "observed_htsim_commit": observed_commit,
        "raw_gate_rejection": {
            "genuine_risk_passed": sum(row["passed"] for row in relation_rows),
            "genuine_risk_total": len(relation_rows),
            "instances": relation_rows,
        },
        "schema": "simllm-htsim-commit-gate-study-v1",
    }
    (args.out / "summary.json").write_bytes(_canonical(summary))
    for name, observation in observations.items():
        (args.out / f"{name}.stdout").write_text(observation["stdout"], newline="\n")
        (args.out / f"{name}.stderr").write_text(observation["stderr"], newline="\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if fatal_failures or not all(row["passed"] for row in relation_rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
