"""Exercise the frozen BACK-2 LogGOPSim invocation-helper expectations."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]

# Frozen sweep. Two parameters vary: the message size and the per-byte gap.
PAYLOAD_BYTES = (262144, 524288)
BYTE_GAP_NS = (3.0, 6.0)

# Frozen LogGOPS constants, the tool defaults named in simulator.ggo.
LATENCY_NS = 2500
OVERHEAD_NS = 1500
MESSAGE_GAP_NS = 1000
BYTE_OVERHEAD_NS = 0
RENDEZVOUS_THRESHOLD_BYTES = 65535

# Frozen physical bracket. The floor is the message's own byte serialization
# and the ceiling adds the largest constant term a rendezvous exchange can
# charge at the tool defaults.
CEILING_SLACK_NS = 15000

# Frozen scaling cross-check for the serialization-dominated cells.
GAP_DOUBLING_RATIO_RANGE = (1.99, 2.01)


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def _required_executable(environment_name: str) -> Path:
    raw = os.environ.get(environment_name)
    if not raw:
        raise ValueError(f"{environment_name} must name an executable")
    path = Path(raw).expanduser()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError(f"{environment_name} is not an executable file: {path}")
    return path.resolve()


def _expected_gap_difference_ns(payload_bytes: int) -> float:
    return (payload_bytes - 1) * (BYTE_GAP_NS[1] - BYTE_GAP_NS[0])


def _expected_size_difference_ns(byte_gap_ns: float) -> float:
    return (PAYLOAD_BYTES[1] - PAYLOAD_BYTES[0]) * byte_gap_ns


def _cells() -> list[dict[str, Any]]:
    return [
        {
            "byte_gap_ns": byte_gap_ns,
            "cell": name,
            "floor_ns": (payload_bytes - 1) * byte_gap_ns,
            "payload_bytes": payload_bytes,
        }
        for name, (payload_bytes, byte_gap_ns) in zip(
            ("A", "B", "C", "D"),
            [(size, gap) for size in PAYLOAD_BYTES for gap in BYTE_GAP_NS],
            strict=True,
        )
    ]


def _write_pair_goal(path: Path, payload_bytes: int) -> Path:
    from simllm.goal import GoalTrace

    trace = GoalTrace(2)
    trace.rank(0).send(payload_bytes, to=1, tag=0)
    trace.rank(1).recv(payload_bytes, source=0, tag=0)
    return trace.write(path)


def _run_cell(
    cell: dict[str, Any],
    goal_bin: Path,
    binary: Path,
    batch_mode: bool,
) -> dict[str, Any]:
    from simllm.backends.loggopsim import LogGopsimConfig, run_loggopsim

    config = LogGopsimConfig(
        goal_bin=goal_bin,
        latency_ns=LATENCY_NS,
        overhead_ns=OVERHEAD_NS,
        message_gap_ns=MESSAGE_GAP_NS,
        byte_gap_ns=cell["byte_gap_ns"],
        byte_overhead_ns=BYTE_OVERHEAD_NS,
        rendezvous_threshold_bytes=RENDEZVOUS_THRESHOLD_BYTES,
        batch_mode=batch_mode,
    )
    result = run_loggopsim(config, binary=binary)
    finish_ns = result.job_completion_time_ps() // 1000
    floor_ns = cell["floor_ns"]
    return {
        "byte_gap_ns": cell["byte_gap_ns"],
        "cell": cell["cell"],
        "ceiling_ns": floor_ns + CEILING_SLACK_NS,
        "finish_ns": finish_ns,
        "floor_ns": floor_ns,
        "inside_physical_bracket": floor_ns <= finish_ns <= floor_ns + CEILING_SLACK_NS,
        "intercept_ns": finish_ns - floor_ns,
        "payload_bytes": cell["payload_bytes"],
        "rank_count": result.rank_count,
        "strictly_positive": finish_ns > 0,
    }


def _check_only(args: argparse.Namespace, executables: dict[str, Path]) -> None:
    plan = {
        "artifacts_created": False,
        "cells": _cells(),
        "ceiling_slack_ns": CEILING_SLACK_NS,
        "executables": {name: str(path) for name, path in executables.items()},
        "expected_gap_difference_ns": {
            str(size): _expected_gap_difference_ns(size) for size in PAYLOAD_BYTES
        },
        "expected_size_difference_ns": {
            str(gap): _expected_size_difference_ns(gap) for gap in BYTE_GAP_NS
        },
        "gap_doubling_ratio_range": list(GAP_DOUBLING_RATIO_RANGE),
        "out": str(args.out),
        "raw_relation_evaluation": "finishing times parsed before any relation is read",
    }
    print(json.dumps(plan, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    executables = {
        "loggopsim": _required_executable("SIMLLM_LOGGOPSIM"),
        "txt2bin": _required_executable("SIMLLM_TXT2BIN"),
    }
    if args.check_only:
        _check_only(args, executables)
        return
    if args.out.exists():
        parser.error("--out must not exist")
    args.out.mkdir(parents=True)

    from simllm.goal import to_binary

    goal_bins: dict[int, Path] = {}
    for payload_bytes in PAYLOAD_BYTES:
        goal_path = _write_pair_goal(args.out / f"pair-{payload_bytes}.goal", payload_bytes)
        goal_bins[payload_bytes] = to_binary(
            goal_path,
            args.out / f"pair-{payload_bytes}.bin",
            tool=executables["txt2bin"],
        )

    rows = [
        _run_cell(cell, goal_bins[cell["payload_bytes"]], executables["loggopsim"], batch_mode=True)
        for cell in _cells()
    ]
    by_cell = {row["cell"]: row for row in rows}

    fatal = {
        "physical_bracket": {
            row["cell"]: row["inside_physical_bracket"] for row in rows
        },
        "rank_count_is_two": {row["cell"]: row["rank_count"] == 2 for row in rows},
        "strictly_positive": {row["cell"]: row["strictly_positive"] for row in rows},
    }
    fatal_failures = [
        f"{name}:{cell}"
        for name, checks in fatal.items()
        for cell, passed in checks.items()
        if not passed
    ]

    gap_family = []
    for size, low, high in ((PAYLOAD_BYTES[0], "A", "B"), (PAYLOAD_BYTES[1], "C", "D")):
        expected = _expected_gap_difference_ns(size)
        observed = by_cell[high]["finish_ns"] - by_cell[low]["finish_ns"]
        ratio = by_cell[high]["finish_ns"] / by_cell[low]["finish_ns"]
        gap_family.append(
            {
                "expected_difference_ns": expected,
                "genuine_risk": True,
                "instance": f"payload_bytes={size}",
                "observed_difference_ns": observed,
                "passed": observed == expected,
                "ratio": ratio,
                "ratio_in_range": (
                    GAP_DOUBLING_RATIO_RANGE[0] <= ratio <= GAP_DOUBLING_RATIO_RANGE[1]
                ),
            }
        )

    size_family = []
    for gap, low, high in ((BYTE_GAP_NS[0], "A", "C"), (BYTE_GAP_NS[1], "B", "D")):
        expected = _expected_size_difference_ns(gap)
        observed = by_cell[high]["finish_ns"] - by_cell[low]["finish_ns"]
        size_family.append(
            {
                "expected_difference_ns": expected,
                "genuine_risk": True,
                "instance": f"byte_gap_ns={gap}",
                "observed_difference_ns": observed,
                "passed": observed == expected,
            }
        )

    intercepts = sorted({row["intercept_ns"] for row in rows})
    summary = {
        "cells": rows,
        "entailed_unscored": {
            "intercept_invariant": len(intercepts) == 1,
            "intercepts_ns": intercepts,
        },
        "fatal_unscored": {
            "checks": fatal,
            "failures": fatal_failures,
            "valid": not fatal_failures,
        },
        "gap_family": {
            "genuine_risk_passed": sum(row["passed"] for row in gap_family),
            "genuine_risk_total": len(gap_family),
            "instances": gap_family,
        },
        "observed_simllm_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "schema": "simllm-loggopsim-helper-study-v1",
        "size_family": {
            "genuine_risk_passed": sum(row["passed"] for row in size_family),
            "genuine_risk_total": len(size_family),
            "instances": size_family,
        },
    }
    (args.out / "summary.json").write_bytes(_canonical(summary))
    print(json.dumps(summary, indent=2, sort_keys=True))
    scored = gap_family + size_family
    if fatal_failures or not all(row["passed"] for row in scored):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
