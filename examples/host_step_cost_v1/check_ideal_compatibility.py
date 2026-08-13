"""Verify the frozen ideal-profile identity against a fresh mission run."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXPECTATIONS = HERE / "expectations.json"
DEFAULT_RESULT = HERE / "ideal_compatibility.json"
GUARD_NAME = "OFF-G1_ideal_named_study_exact_identity"
CANONICALIZATION = (
    "remove wall_seconds from each cell.json, serialize all cells by name with "
    "sorted keys and compact separators, append LF"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path.name}")
    return value


def _validate_expectations(expectations: dict[str, Any]) -> dict[str, Any]:
    if expectations.get("schema") != "simllm-host-step-cost-v1-expectations-v1":
        raise AssertionError("host-step expectation schema drifted")
    frozen = expectations.get("ideal_compatibility")
    if not isinstance(frozen, dict):
        raise TypeError("ideal compatibility freeze must be a JSON object")
    if frozen.get("study") != "end_to_end_replay_v1":
        raise AssertionError("named ideal compatibility study drifted")
    if frozen.get("canonicalization") != CANONICALIZATION:
        raise AssertionError("OFF-G1 cell canonicalization drifted")
    step_digests = frozen.get("step_record_sha256")
    if not isinstance(step_digests, dict) or len(step_digests) != 5:
        raise AssertionError("OFF-G1 must freeze exactly five step streams")
    digests = [frozen.get("aggregate_cell_sha256"), *step_digests.values()]
    if any(
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        for digest in digests
    ):
        raise AssertionError("OFF-G1 contains a malformed SHA-256 identity")
    fatal_guards = expectations.get("fatal_unscored_guards")
    if not isinstance(fatal_guards, list) or GUARD_NAME not in fatal_guards:
        raise AssertionError("OFF-G1 is not registered as a fatal unscored guard")
    return frozen


def _cell_paths(run_dir: Path) -> dict[str, Path]:
    cells_dir = run_dir / "cells"
    if not cells_dir.is_dir():
        raise FileNotFoundError(f"mission cells directory is missing: {cells_dir}")
    return {
        path.parent.name: path
        for path in sorted(cells_dir.glob("*/cell.json"))
        if path.is_file()
    }


def _canonical_cell_digest(cell_paths: dict[str, Path]) -> tuple[str, list[str]]:
    """Hash all named cells after deleting only their top-level wall time."""

    cells: dict[str, Any] = {}
    missing_wall_seconds: list[str] = []
    for name in sorted(cell_paths):
        cell = _load_json(cell_paths[name])
        if "wall_seconds" not in cell:
            missing_wall_seconds.append(name)
        else:
            del cell["wall_seconds"]
        cells[name] = cell
    payload = json.dumps(cells, sort_keys=True, separators=(",", ":")) + "\n"
    return _sha256_bytes(payload.encode("utf-8")), missing_wall_seconds


def _mission_preconditions(summary: dict[str, Any]) -> dict[str, Any]:
    relations = summary.get("exact_oracle_relations")
    relation_count = len(relations) if isinstance(relations, dict) else 0
    every_relation_passed = bool(relations) and all(
        isinstance(detail, dict) and detail.get("passed") is True
        for detail in relations.values()
    )
    violated = summary.get("violated_fatal_guards")
    nonvoid = summary.get("void") is False and violated == []
    exact_oracles_passed = (
        relation_count == 13
        and every_relation_passed
        and summary.get("exact_oracle_passed") == 13
        and summary.get("exact_oracle_total") == 13
    )
    return {
        "nonvoid": nonvoid,
        "violated_fatal_guards": violated,
        "exact_oracle_relations_observed": relation_count,
        "exact_oracle_relations_expected": 13,
        "all_exact_oracles_passed": exact_oracles_passed,
    }


def _evaluate(run_dir: Path, frozen: dict[str, Any]) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"mission summary is missing: {summary_path}")
    summary_bytes = summary_path.read_bytes()
    summary = json.loads(summary_bytes)
    if not isinstance(summary, dict):
        raise TypeError("mission summary must be a JSON object")

    expected_steps = frozen["step_record_sha256"]
    cell_paths = _cell_paths(run_dir)
    actual_names = set(cell_paths)
    expected_names = set(expected_steps)
    aggregate_digest, missing_wall_seconds = _canonical_cell_digest(cell_paths)

    findings: list[str] = []
    if actual_names != expected_names:
        findings.append("mission cell names differ from the five frozen identities")
    if missing_wall_seconds:
        findings.append("one or more cell records lack the removable wall_seconds field")
    aggregate_matches = aggregate_digest == frozen["aggregate_cell_sha256"]
    if not aggregate_matches:
        findings.append("aggregate canonical cell identity differs from the freeze")

    step_identity: dict[str, dict[str, Any]] = {}
    for name in sorted(expected_names):
        path = run_dir / "cells" / name / "steps.jsonl"
        actual = _sha256_file(path) if path.is_file() else None
        expected = expected_steps[name]
        matched = actual == expected
        step_identity[name] = {
            "expected_sha256": expected,
            "actual_sha256": actual,
            "matched": matched,
        }
        if not matched:
            findings.append(f"{name} steps.jsonl identity differs from the freeze")

    preconditions = _mission_preconditions(summary)
    if not preconditions["nonvoid"]:
        findings.append("named mission study is void")
    if not preconditions["all_exact_oracles_passed"]:
        findings.append("named mission study did not pass all 13 exact oracles")

    return {
        "summary_sha256": _sha256_bytes(summary_bytes),
        "mission_preconditions": preconditions,
        "aggregate_identity": {
            "canonicalization": CANONICALIZATION,
            "expected_sha256": frozen["aggregate_cell_sha256"],
            "actual_sha256": aggregate_digest,
            "matched": aggregate_matches,
        },
        "step_record_identity": step_identity,
        "findings": findings,
    }


def _git_output(*arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _clean_head() -> str:
    status = _git_output("status", "--porcelain", "--untracked-files=all")
    if status:
        raise RuntimeError(
            "OFF-G1 production evidence requires a clean tracked and untracked worktree"
        )
    return _git_output("rev-parse", "HEAD")


def _result_document(observed_head: str, evaluation: dict[str, Any]) -> dict[str, Any]:
    held = not evaluation["findings"]
    return {
        "schema": "simllm-host-step-ideal-compatibility-v1",
        "study": "host_step_cost_v1",
        "task": "COMP-2",
        "observed_git_head": observed_head,
        "input_summary": {
            "artifact": "summary.json",
            "sha256": evaluation["summary_sha256"],
        },
        "named_study": "end_to_end_replay_v1",
        "mission_preconditions": evaluation["mission_preconditions"],
        "aggregate_cell_identity": evaluation["aggregate_identity"],
        "step_record_identity": evaluation["step_record_identity"],
        "evidence": {
            "class": "fatal_unscored_guard",
            "guard": GUARD_NAME,
            "held": held,
            "findings": evaluation["findings"],
        },
        "run_status": "nonvoid" if held else "void",
    }


def main() -> None:
    args = _parse_args()
    expectations = _load_json(EXPECTATIONS)
    frozen = _validate_expectations(expectations)
    if args.check_only:
        print(
            json.dumps(
                {
                    "check_only": True,
                    "fatal_unscored_guard": GUARD_NAME,
                    "expected_cell_count": len(frozen["step_record_sha256"]),
                },
                sort_keys=True,
            )
        )
        return

    if args.result.exists():
        raise FileExistsError(f"result already exists: {args.result}")
    observed_head = _clean_head()
    evaluation = _evaluate(args.run_dir, frozen)
    if _clean_head() != observed_head:
        raise RuntimeError("git HEAD changed while OFF-G1 inputs were being evaluated")
    result = _result_document(observed_head, evaluation)
    args.result.parent.mkdir(parents=True, exist_ok=True)
    with args.result.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "guard": GUARD_NAME,
                "held": result["evidence"]["held"],
                "run_status": result["run_status"],
            },
            sort_keys=True,
        )
    )
    if not result["evidence"]["held"]:
        raise SystemExit("OFF-G1 failed; the compatibility run is void")


if __name__ == "__main__":
    main()
