"""Recompute PLAY-B3 GOAL pair tables directly from raw trace rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

TRACE_SCHEMA = "simllm-preplay-trace-v1"
STEP_SCHEMA = "atlahs-closed-loop-step-v1"
RESULT_SCHEMA = "simllm-play5-raw-goal-recheck-v1"
LAYERS = 24
EXPERTS = 32
TOP_K = 8
VECTOR_BYTES = 2_048
RANKS = (0, 1)
CELL_NAMES = ("200g", "400g")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            row = json.loads(line)
            if not isinstance(row, dict):
                raise AssertionError(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
    return rows


def _raw_trace_index(
    path: Path,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[tuple[str, str, int], dict[str, Any]],
]:
    rows = _jsonl(path)
    if not rows or any(row.get("schema") != TRACE_SCHEMA for row in rows):
        raise AssertionError("raw trace schema changed")
    headers = [row for row in rows if row.get("row_type") == "header"]
    footers = [row for row in rows if row.get("row_type") == "footer"]
    if len(headers) != 1 or len(footers) != 1:
        raise AssertionError("raw trace must have one header and footer")
    provenance = headers[0]["provenance"]
    if (
        provenance.get("top_k") != TOP_K
        or provenance.get("expert_count") != EXPERTS
        or provenance.get("moe_layer_indices") != list(range(LAYERS))
    ):
        raise AssertionError("raw trace routing geometry changed")

    request_rows = [row for row in rows if row.get("row_type") == "request"]
    requests = {row["request_id"]: row for row in request_rows}
    if len(requests) != len(request_rows) or not requests:
        raise AssertionError("raw trace request identities are empty or duplicated")

    tokens: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in rows:
        if row.get("row_type") != "forward-token":
            continue
        key = (row["request_id"], row["phase"], row["token_index"])
        if key in tokens:
            raise AssertionError(f"duplicate raw forward-token row {key}")
        routes = row["routing"]
        if [route["layer_index"] for route in routes] != list(range(LAYERS)):
            raise AssertionError(f"raw forward-token row {key} changed layer coverage")
        for route in routes:
            expert_ids = route["expert_ids"]
            if (
                len(expert_ids) != TOP_K
                or len(set(expert_ids)) != TOP_K
                or any(expert < 0 or expert >= EXPERTS for expert in expert_ids)
            ):
                raise AssertionError(f"raw forward-token row {key} has invalid experts")
        tokens[key] = row

    expected_keys = set()
    for request_id, request in requests.items():
        for index, token_id in enumerate(request["input_token_ids"]):
            key = (request_id, "prefill", index)
            expected_keys.add(key)
            if tokens.get(key, {}).get("token_id") != token_id:
                raise AssertionError(f"raw prefill row {key} disagrees with request tokens")
        for index, token_id in enumerate(request["output_token_ids"][:-1]):
            key = (request_id, "decode", index)
            expected_keys.add(key)
            if tokens.get(key, {}).get("token_id") != token_id:
                raise AssertionError(f"raw decode row {key} disagrees with request tokens")
    if set(tokens) != expected_keys:
        raise AssertionError("raw forward-token coverage disagrees with request rows")

    footer = footers[0]
    expected_prefill = sum(len(row["input_token_ids"]) for row in requests.values())
    expected_decode = sum(len(row["output_token_ids"]) - 1 for row in requests.values())
    if (
        footer.get("request_count") != len(requests)
        or footer.get("prefill_forward_token_count") != expected_prefill
        or footer.get("decode_forward_token_count") != expected_decode
    ):
        raise AssertionError("raw trace footer counts changed")
    return requests, tokens


def _scheduled_raw_rows(
    steps_path: Path,
    requests: dict[str, dict[str, Any]],
    tokens: dict[tuple[str, str, int], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]]]:
    steps = _jsonl(steps_path)
    if [row.get("step_index") for row in steps] != list(range(len(steps))):
        raise AssertionError("scheduler step indices are not contiguous")
    if any(row.get("schema") != STEP_SCHEMA for row in steps):
        raise AssertionError("scheduler step schema changed")
    cursors = {(request_id, phase): 0 for request_id in requests for phase in ("prefill", "decode")}
    by_step = []
    for step in steps:
        selected = []
        for scheduled in step["scheduled"]:
            request_id = scheduled["request_id"]
            phase = scheduled["phase"]
            if request_id not in requests or phase not in ("prefill", "decode"):
                raise AssertionError("scheduler named an unknown request or phase")
            count = scheduled["num_new_tokens"]
            if not isinstance(count, int) or count <= 0:
                raise AssertionError("scheduler raw-trace check requires positive token visits")
            cursor_key = (request_id, phase)
            start = cursors[cursor_key]
            end = start + count
            prompt_count = len(requests[request_id]["input_token_ids"])
            expected_context = end if phase == "prefill" else prompt_count + end
            if scheduled["context_length"] != expected_context:
                raise AssertionError("scheduler context does not identify the next raw rows")
            for index in range(start, end):
                key = (request_id, phase, index)
                if key not in tokens:
                    raise AssertionError(f"scheduler requested absent raw trace row {key}")
                selected.append(tokens[key])
            cursors[cursor_key] = end
        by_step.append(selected)
    consumed = {
        (request_id, phase, index)
        for (request_id, phase), count in cursors.items()
        for index in range(count)
    }
    if consumed != set(tokens):
        raise AssertionError("scheduler did not consume every raw forward-token row exactly once")
    return steps, by_step


def _expected_pairs(
    raw_rows: list[dict[str, Any]],
) -> dict[int, dict[tuple[int, int], int]]:
    result = {}
    for layer in range(LAYERS):
        dispatch: dict[tuple[int, int], int] = {}
        for source in RANKS:
            for token in raw_rows:
                route = token["routing"][layer]
                if route["layer_index"] != layer:
                    raise AssertionError("raw routing row order changed")
                destinations = {0 if expert < 16 else 1 for expert in route["expert_ids"]}
                for destination in destinations:
                    if source != destination:
                        pair = (source, destination)
                        dispatch[pair] = dispatch.get(pair, 0) + VECTOR_BYTES
        combine = {(destination, source): size for (source, destination), size in dispatch.items()}
        result[1000 + 2 * layer] = dispatch
        result[1001 + 2 * layer] = combine
    return result


def _goal_pairs(path: Path) -> dict[int, dict[tuple[int, int], int]]:
    result: dict[int, dict[tuple[int, int], int]] = {}
    rank = None
    num_ranks = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("num_ranks "):
            num_ranks = int(line.split()[1])
        elif line.startswith("rank "):
            rank = int(line.split()[1])
        elif ": send " in line:
            if rank is None:
                raise AssertionError("GOAL send appeared outside a rank block")
            words = line.split()
            pair = (rank, int(words[4]))
            tag = int(words[6])
            if pair in result.setdefault(tag, {}):
                raise AssertionError(f"duplicate GOAL send for tag {tag}, pair {pair}")
            result[tag][pair] = int(words[2].removesuffix("b"))
    if num_ranks != len(RANKS):
        raise AssertionError("GOAL rank count changed")
    return result


def _table_rows(table: dict[int, dict[tuple[int, int], int]]) -> list[dict[str, int]]:
    return [
        {"tag": tag, "source": source, "destination": destination, "bytes": size}
        for tag, pairs in sorted(table.items())
        for (source, destination), size in sorted(pairs.items())
    ]


def _table_sha256(table: dict[int, dict[tuple[int, int], int]]) -> str:
    payload = json.dumps(_table_rows(table), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _cell(
    name: str,
    steps_path: Path,
    goals_dir: Path,
    requests: dict[str, dict[str, Any]],
    tokens: dict[tuple[str, str, int], dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    steps, by_step = _scheduled_raw_rows(steps_path, requests, tokens)
    goal_paths = sorted(goals_dir.glob("step-*.goal"))
    expected_paths = [goals_dir / f"step-{index:06d}.goal" for index in range(len(steps))]
    if goal_paths != expected_paths:
        raise AssertionError(f"{name} GOAL file set disagrees with scheduler steps")
    rows = []
    for step, raw_rows, goal_path in zip(steps, by_step, goal_paths, strict=True):
        expected = _expected_pairs(raw_rows)
        actual = _goal_pairs(goal_path)
        rows.append(
            {
                "step_index": step["step_index"],
                "raw_forward_token_count": len(raw_rows),
                "expected_table_sha256": _table_sha256(expected),
                "actual_table_sha256": _table_sha256(actual),
                "goal_sha256": file_sha256(goal_path),
                "send_count": len(_table_rows(actual)),
                "passed": actual == expected,
            }
        )
    identity = [
        {
            "step_index": step["step_index"],
            "scheduled": step["scheduled"],
            "finished_request_ids": step["finished_request_ids"],
            "num_sampled": step["num_sampled"],
        }
        for step in steps
    ]
    return {
        "steps_sha256": file_sha256(steps_path),
        "goal_rows": rows,
        "passed": all(row["passed"] for row in rows),
    }, identity


def check_only(args: argparse.Namespace) -> None:
    inputs = (
        args.trace,
        args.steps_200,
        args.goals_200,
        args.steps_400,
        args.goals_400,
    )
    if not args.trace.is_file() or not args.steps_200.is_file() or not args.steps_400.is_file():
        raise SystemExit("raw trace or scheduler step input is missing")
    if not args.goals_200.is_dir() or not args.goals_400.is_dir():
        raise SystemExit("GOAL input directory is missing")
    if not args.out.parent.is_dir():
        raise SystemExit("output parent directory is missing")
    if VECTOR_BYTES != 2_048 or LAYERS != 24 or EXPERTS != 32 or TOP_K != 8:
        raise AssertionError("raw recomputation geometry changed")
    if len(inputs) != 5 or CELL_NAMES != ("200g", "400g"):
        raise AssertionError("raw recomputation input surface changed")
    print(f"check-only out={args.out}; validated raw GOAL recheck inputs and produced no artifacts")


def run(args: argparse.Namespace) -> dict[str, Any]:
    requests, tokens = _raw_trace_index(args.trace)
    cells = {}
    identities = {}
    for name, steps, goals in (
        ("200g", args.steps_200, args.goals_200),
        ("400g", args.steps_400, args.goals_400),
    ):
        cells[name], identities[name] = _cell(name, steps, goals, requests, tokens)
    scheduler_identity = identities["200g"] == identities["400g"]
    expected_identity = [
        row["expected_table_sha256"] for row in cells["200g"]["goal_rows"]
    ] == [row["expected_table_sha256"] for row in cells["400g"]["goal_rows"]]
    if not scheduler_identity or not expected_identity:
        raise AssertionError("bandwidth changed scheduler membership or raw expected tables")
    executed = sum(len(cell["goal_rows"]) for cell in cells.values())
    passed = sum(row["passed"] for cell in cells.values() for row in cell["goal_rows"])
    summary = {
        "schema": RESULT_SCHEMA,
        "classification": "post-specified-review-check",
        "input_authority": "raw-forward-token-jsonl-rows",
        "routed_experts_projection_used": False,
        "trace_sha256": file_sha256(args.trace),
        "scheduler_identity": scheduler_identity,
        "expected_table_identity": expected_identity,
        "cells": cells,
        "scored": {
            "executed": executed,
            "passed": passed,
            "genuine_risk_numerator": executed,
            "genuine_risk_denominator": executed,
        },
        "complete": passed == executed and executed == 10,
    }
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not summary["complete"]:
        raise AssertionError("raw GOAL recheck failed")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--steps-200", type=Path, required=True)
    parser.add_argument("--goals-200", type=Path, required=True)
    parser.add_argument("--steps-400", type=Path, required=True)
    parser.add_argument("--goals-400", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    check_only(args)
    if args.check_only:
        return
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
