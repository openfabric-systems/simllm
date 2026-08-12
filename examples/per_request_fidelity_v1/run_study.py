"""Run the frozen per-request replay fidelity study."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

EXPECTATIONS_COMMIT = "eaa8b23860c7a5e357dc509fcf0897176a40df66"

SYNTHETIC_GOALS = {
    (0, 1): (744, "1eb2bbff8a981523b5f6733420aa9d5d3509aa473ed991409b8d455e619e5864", 6, 48),
    (0, 2): (952, "78a8e80589b156374b965634dd82251931219398c1e2cf2454b06cbe3629916c", 8, 80),
    (0, 3): (964, "8e38bf44631b9f3d7020452886552502fa567ec44559d05b5401a5dbbc825ab6", 8, 128),
    (1, 1): (744, "8c1738dbd01f320b0f5f005b9ea6acd19145c77db67af89eaac4a78219d494de", 6, 48),
    (1, 2): (960, "3023c39e472980ed6c689410a21fa626db3a73cf8a3d83bde425d8d41cfd4361", 8, 112),
    (1, 3): (964, "60cb32ca80a57d03b627de51d01fd292c0e87da3ec1482760faa8d304b075440", 8, 176),
}

SYNTHETIC_PERMUTATIONS = {
    (0, 2): (0, 12, 96, -16),
    (0, 3): (0, 12, 96, -16),
    (1, 2): (0, 4, 32, 16),
    (1, 3): (0, 4, 32, 16),
}

SYNTHETIC_JCT_PS = {
    (0, 1, 200_000_000_000): 8_003_280,
    (0, 1, 400_000_000_000): 8_002_640,
    (0, 2, 200_000_000_000): 8_003_920,
    (0, 2, 400_000_000_000): 8_002_960,
    (0, 3, 200_000_000_000): 8_004_560,
    (0, 3, 400_000_000_000): 8_003_280,
    (1, 1, 200_000_000_000): 8_003_280,
    (1, 1, 400_000_000_000): 8_002_640,
    (1, 2, 200_000_000_000): 8_004_560,
    (1, 2, 400_000_000_000): 8_003_280,
    (1, 3, 200_000_000_000): 8_005_840,
    (1, 3, 400_000_000_000): 8_003_920,
}

SYNTHETIC_DISPATCH = {
    (0, "alpha", 0): ((0, 1, 8), (1, 0, 8)),
    (0, "alpha", 1): ((1, 0, 8),),
    (0, "beta", 0): ((1, 0, 8),),
    (0, "beta", 1): ((0, 1, 8),),
    (0, "gamma", 0): ((0, 1, 8),),
    (0, "gamma", 1): ((0, 1, 8), (1, 0, 8)),
    (1, "alpha", 0): ((1, 0, 8),),
    (1, "alpha", 1): ((0, 1, 8), (1, 0, 8)),
    (1, "beta", 0): ((0, 1, 8), (1, 0, 8)),
    (1, "beta", 1): ((0, 1, 8), (1, 0, 8)),
    (1, "gamma", 0): ((0, 1, 8), (1, 0, 8)),
    (1, "gamma", 1): ((0, 1, 8), (1, 0, 8)),
}

GRANITE_REQUESTS = {
    "r0": (
        2_688,
        84_439_040,
        80_824,
        "d2d5564c0507ae8e9946e377dfd9df0fca3eab20910d150faba03b1576e5e75a",
    ),
    "r1": (
        2_688,
        46_190_592,
        80_516,
        "5f7603ec085e76e86b022b688404c428c90344115ac675ef40b59e609b90f568",
    ),
    "r2": (
        2_688,
        76_869_632,
        80_810,
        "c441be8e81936ef0d32d32d59dfaf20f08bf496d588836edfee84058dbe0c89f",
    ),
}

GRANITE_ALL = (
    8_064,
    207_499_264,
    242_146,
    "bcb21232c6f433e64ca0efb9bbfdaab4c008b087249f5d4b849dfb9bc646c077",
)

GRANITE_PERMUTATION = {
    "aggregate_mismatch_count": 0,
    "request_mismatch_count": 5_348,
    "l1_error_bytes": 76_496_896,
    "request_delta_bytes": {
        "r0": -38_248_448,
        "r1": 38_248_448,
        "r2": 0,
    },
}

SOURCE_ARTIFACTS = {
    "capture": (
        "capture/granite-greedy.jsonl",
        "5f55fccc265ee5519430a0f73d6631e49aa547ec07fcb81034e9fc2b4d9fead6",
    ),
    "steps": (
        "replay-400g/steps.jsonl",
        "824cd9557293328bb42b593ac893b6a067302e545b087c9219195ccb8031d755",
    ),
    "routing": (
        "replay-400g/routed-experts.json",
        "24e986e989e21f1bfe7e758d4470928c82c3bbaec06072a839743b9b17d7cf5f",
    ),
    "aggregate_goal": (
        "replay-400g/htsim/step-000000.goal",
        "08a0403af66ff8a9d6b18f93afd15ae0bc925cc85555acf8a0593438a3d7bc92",
    ),
}

_GOAL_SEND_RE = re.compile(
    r"^r(?P<source>\d+)op\d+: send (?P<size>\d+)b to "
    r"(?P<destination>\d+) tag (?P<tag>\d+)(?:\s|$)"
)


def _check_frozen_registry() -> None:
    if set(SYNTHETIC_GOALS) != {(epoch, count) for epoch in (0, 1) for count in (1, 2, 3)}:
        raise AssertionError("synthetic GOAL registry is incomplete")
    if set(SYNTHETIC_PERMUTATIONS) != {(epoch, count) for epoch in (0, 1) for count in (2, 3)}:
        raise AssertionError("permutation registry is incomplete")
    if len(SYNTHETIC_JCT_PS) != 12:
        raise AssertionError("fluid JCT registry must contain twelve cells")
    if sum(value[1] for value in GRANITE_REQUESTS.values()) != GRANITE_ALL[1]:
        raise AssertionError("Granite request bytes do not conserve the total")
    digests = [value[1] for value in SYNTHETIC_GOALS.values()]
    digests += [value[3] for value in GRANITE_REQUESTS.values()]
    digests.append(GRANITE_ALL[3])
    if any(len(digest) != 64 for digest in digests):
        raise AssertionError("frozen SHA-256 values must contain 64 digits")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--htsim-rnic", required=True, type=Path)
    parser.add_argument("--txt2bin", required=True, type=Path)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _path_observation(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "bytes": len(data),
        "sha256": _sha256(data),
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _goal_observation(text: str) -> dict[str, Any]:
    sends = []
    for line in text.splitlines():
        match = _GOAL_SEND_RE.match(line)
        if match is not None:
            sends.append(
                (
                    int(match.group("source")),
                    int(match.group("destination")),
                    int(match.group("tag")),
                    int(match.group("size")),
                )
            )
    data = text.encode()
    return {
        "goal_bytes": len(data),
        "goal_sha256": _sha256(data),
        "send_count": len(sends),
        "total_send_bytes": sum(row[3] for row in sends),
    }


def _git_object(revision: str) -> str:
    result = subprocess.run(
        ["git", "rev-parse", revision],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _validate_inputs(arguments: argparse.Namespace) -> dict[str, Any]:
    observations = {}
    for name, (relative_path, expected_sha256) in SOURCE_ARTIFACTS.items():
        path = arguments.source_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"missing source artifact: {path}")
        observation = _path_observation(path)
        observation["relative_path"] = relative_path
        observation["expected_sha256"] = expected_sha256
        observation["matches_frozen"] = observation["sha256"] == expected_sha256
        if not observation["matches_frozen"]:
            raise AssertionError(f"source artifact changed: {relative_path}")
        observations[name] = observation
    capture = arguments.source_root / SOURCE_ARTIFACTS["capture"][0]
    observations["capture"]["line_count"] = len(capture.read_bytes().splitlines())
    if observations["capture"]["line_count"] != 120:
        raise AssertionError("capture row count changed")
    goal = arguments.source_root / SOURCE_ARTIFACTS["aggregate_goal"][0]
    goal_observation = _goal_observation(goal.read_text(encoding="utf-8"))
    observations["aggregate_goal"].update(goal_observation)
    if goal_observation != {
        "goal_bytes": 334_432,
        "goal_sha256": SOURCE_ARTIFACTS["aggregate_goal"][1],
        "send_count": 2_688,
        "total_send_bytes": 207_499_264,
    }:
        raise AssertionError("archived aggregate GOAL changed")
    for label, binary in (
        ("htsim_rnic", arguments.htsim_rnic),
        ("txt2bin", arguments.txt2bin),
    ):
        if not binary.is_file():
            raise FileNotFoundError(f"missing native binary: {binary}")
        observations[label] = _path_observation(binary)
        observations[label]["supplied_path"] = str(binary)
    return observations


def _fixed_provider(duration_ps: int) -> Any:
    from simllm.compute import ComputeProvider, DurationEstimate

    class FixedProvider(ComputeProvider):
        def estimate(self, kernel: Any, gpu: Any) -> DurationEstimate:
            return DurationEstimate(duration_ps=duration_ps, bound="measured")

    return FixedProvider()


def _synthetic_inputs() -> tuple[Any, Any]:
    from simllm.compute import ModelDims
    from simllm.preplay import (
        PREPLAY_TRACE_SCHEMA,
        ForwardPhase,
        RoutedExperts,
        RoutedLayer,
        RoutedRequest,
        RoutedToken,
    )
    from simllm.traffic import ExpertPlacementSnapshot, RoutedMoeSupply

    dims = ModelDims(
        num_layers=2,
        hidden_size=4,
        intermediate_size=8,
        num_heads=2,
        num_kv_heads=2,
        head_size=2,
        vocab_size=16,
        dtype_bytes=2,
        num_experts=4,
        top_k=2,
        moe_intermediate_size=4,
        local_num_experts=2,
    )
    routes = {
        "alpha": ((0, 2), (0, 1)),
        "beta": ((0, 1), (2, 3)),
        "gamma": ((2, 3), (1, 3)),
    }
    requests = tuple(
        RoutedRequest(
            request_id=request_id,
            prompt_token_count=1,
            output_token_count=1,
            tokens=(
                RoutedToken(
                    phase=ForwardPhase.PREFILL,
                    token_index=0,
                    token_id=10 + index,
                    layers=tuple(
                        RoutedLayer(layer_index=layer, expert_ids=experts)
                        for layer, experts in enumerate(request_routes)
                    ),
                ),
            ),
        )
        for index, (request_id, request_routes) in enumerate(routes.items())
    )
    routed = RoutedExperts(
        trace_schema=PREPLAY_TRACE_SCHEMA,
        trace_sha256="a" * 64,
        expert_count=4,
        top_k=2,
        moe_layer_indices=(0, 1),
        requests=requests,
    )
    epoch0 = ExpertPlacementSnapshot(
        placement_epoch=0,
        expert_owners=tuple(
            (layer, expert, 0 if expert < 2 else 1) for layer in range(2) for expert in range(4)
        ),
    )
    epoch1_owners = (
        {0: 0, 1: 1, 2: 0, 3: 1},
        {0: 0, 1: 1, 2: 1, 3: 0},
    )
    epoch1 = ExpertPlacementSnapshot(
        placement_epoch=1,
        expert_owners=tuple(
            (layer, expert, epoch1_owners[layer][expert])
            for layer in range(2)
            for expert in range(4)
        ),
    )
    supply = RoutedMoeSupply(
        routed_experts=routed,
        placements=(epoch0, epoch1),
        step_placement_epochs=tuple(
            (epoch * 3 + count - 1, epoch) for epoch in (0, 1) for count in (1, 2, 3)
        ),
    )
    return dims, supply


def _synthetic_record(epoch: int, count: int) -> Any:
    from simllm.core import RequestPhase, ScheduledRequest, StepRecord

    return StepRecord(
        step_index=epoch * 3 + count - 1,
        virtual_time_ps=0,
        scheduled=[
            ScheduledRequest(
                request_id,
                RequestPhase.PREFILL,
                1,
                context_length=1,
            )
            for request_id in ("alpha", "beta", "gamma")[:count]
        ],
        num_sampled=count,
    )


def _swap_requests(messages: tuple[Any, ...], left: str, right: str) -> tuple[Any, ...]:
    result = []
    for message in messages:
        partition = tuple(
            sorted(
                (
                    (
                        right
                        if request_id == left
                        else left
                        if request_id == right
                        else request_id,
                        size,
                    )
                    for request_id, size in message.request_payload_bytes
                )
            )
        )
        result.append(replace(message, request_payload_bytes=partition))
    return tuple(result)


def _report_observation(report: Any) -> dict[str, Any]:
    return {
        "aggregate_matches": report.aggregate_matches,
        "aggregate_mismatch_count": report.aggregate_mismatch_count,
        "aggregate_l1_error_bytes": report.aggregate_l1_error_bytes,
        "per_request_matches": report.per_request_matches,
        "request_mismatch_count": report.mismatch_count,
        "request_l1_error_bytes": report.l1_error_bytes,
        "request_delta_bytes": dict(report.request_delta_bytes),
        "expected_request_row_count": len(report.expected_request_rows),
        "observed_request_row_count": len(report.observed_request_rows),
    }


def _synthetic_exact_rows(epoch: int, count: int, step_index: int) -> tuple[Any, ...]:
    rows = []
    for request_id in ("alpha", "beta", "gamma")[:count]:
        for layer in range(2):
            operation = f"step-{step_index}:layer-{layer}:ep-dispatch"
            dispatch = SYNTHETIC_DISPATCH[(epoch, request_id, layer)]
            rows.extend(
                (operation, request_id, source, destination, size)
                for source, destination, size in dispatch
            )
            operation = f"step-{step_index}:layer-{layer}:ep-combine"
            rows.extend(
                (operation, request_id, destination, source, size)
                for source, destination, size in dispatch
            )
    return tuple(sorted(rows))


def _gate_rejects(
    record: Any,
    dims: Any,
    ep_ranks: tuple[int, ...],
    supply: Any,
    messages: tuple[Any, ...],
) -> bool:
    from simllm.traffic import RequestFidelityError, validate_request_moe_fidelity

    try:
        validate_request_moe_fidelity(
            record,
            dims,
            ep_ranks,
            supply,
            messages,
        )
    except RequestFidelityError:
        return True
    return False


def _run_synthetic(out: Path) -> tuple[dict[str, Any], dict[tuple[int, int], Any]]:
    from simllm.backends import SerialStepLowerer, SerialStepLowererConfig
    from simllm.core import execution_graph_from_json, execution_graph_to_json
    from simllm.traffic import (
        compare_request_moe_fidelity,
        render_serial_execution_graph_goal,
        render_step_goal,
    )

    dims, supply = _synthetic_inputs()
    provider = _fixed_provider(2_000)
    traces = {}
    graph_traces = {}
    positive = []
    permutations = []
    fatal_rows = []
    output_dir = out / "synthetic"
    output_dir.mkdir(parents=True, exist_ok=False)

    for epoch in (0, 1):
        for count in (1, 2, 3):
            record = _synthetic_record(epoch, count)
            trace = render_step_goal(
                record,
                dims,
                (0,),
                1,
                ep_ranks=(0, 1),
                routed_supply=supply,
            )
            direct_report = compare_request_moe_fidelity(
                record,
                dims,
                (0, 1),
                supply,
                trace.messages,
            )
            lowerer = SerialStepLowerer(
                SerialStepLowererConfig(
                    dims=dims,
                    tp_ranks=(0,),
                    ep_ranks=(0, 1),
                    provider=provider,
                    routed_moe_supply=supply,
                )
            )
            graph_json = execution_graph_to_json(lowerer.lower(record))
            graph = execution_graph_from_json(graph_json)
            graph_trace = render_serial_execution_graph_goal(graph)
            graph_report = compare_request_moe_fidelity(
                record,
                dims,
                (0, 1),
                supply,
                graph_trace.messages,
            )
            label = f"epoch-{epoch}-requests-{count}"
            trace.write(output_dir / f"{label}.goal")
            _write_json(output_dir / f"{label}.graph.json", graph_json)
            graph_trace.write(output_dir / f"{label}.graph.goal")
            positive.append(
                {
                    "epoch": epoch,
                    "request_count": count,
                    "direct": _report_observation(direct_report),
                    "graph_round_trip": _report_observation(graph_report),
                    "classification": "fatal-unscored-entailment",
                    "reason": (
                        "both renderers require the same fidelity relation before "
                        "returning the trace observed here"
                    ),
                }
            )
            traces[(epoch, count)] = trace
            graph_traces[(epoch, count)] = graph_trace

            if count >= 2:
                permuted = _swap_requests(trace.messages, "alpha", "beta")
                report = compare_request_moe_fidelity(
                    record,
                    dims,
                    (0, 1),
                    supply,
                    permuted,
                )
                expected = SYNTHETIC_PERMUTATIONS[(epoch, count)]
                delta = dict(report.request_delta_bytes)
                gate_rejected = _gate_rejects(
                    record,
                    dims,
                    (0, 1),
                    supply,
                    permuted,
                )
                passed = (
                    report.aggregate_mismatch_count == expected[0]
                    and report.aggregate_matches
                    and report.mismatch_count == expected[1]
                    and report.l1_error_bytes == expected[2]
                    and delta["alpha"] == expected[3]
                    and delta["beta"] == -expected[3]
                    and gate_rejected
                )
                permutations.append(
                    {
                        "epoch": epoch,
                        "request_count": count,
                        **_report_observation(report),
                        "expected": {
                            "aggregate_mismatch_count": expected[0],
                            "request_mismatch_count": expected[1],
                            "request_l1_error_bytes": expected[2],
                            "alpha_delta_bytes": expected[3],
                        },
                        "gate_rejected": gate_rejected,
                        "passed": passed,
                    }
                )

    if not all(row["passed"] for row in permutations):
        raise AssertionError("PLAY-B2 failed before fatal exact-oracle evaluation")

    for epoch in (0, 1):
        for count in (1, 2, 3):
            trace = traces[(epoch, count)]
            graph_trace = graph_traces[(epoch, count)]
            record = _synthetic_record(epoch, count)
            report = compare_request_moe_fidelity(
                record,
                dims,
                (0, 1),
                supply,
                trace.messages,
            )
            goal = _goal_observation(trace.render())
            expected_goal = SYNTHETIC_GOALS[(epoch, count)]
            exact_rows = _synthetic_exact_rows(epoch, count, record.step_index)
            physical_passed = goal == {
                "goal_bytes": expected_goal[0],
                "goal_sha256": expected_goal[1],
                "send_count": expected_goal[2],
                "total_send_bytes": expected_goal[3],
            }
            row_passed = report.observed_request_rows == exact_rows
            graph_report = compare_request_moe_fidelity(
                record,
                dims,
                (0, 1),
                supply,
                graph_trace.messages,
            )
            graph_rows_passed = graph_report.observed_request_rows == exact_rows
            graph_goal_matches = graph_trace.render() == trace.render()
            written_goal = (output_dir / f"epoch-{epoch}-requests-{count}.goal").read_text(
                encoding="utf-8"
            )
            fatal_rows.append(
                {
                    "epoch": epoch,
                    "request_count": count,
                    **goal,
                    "request_rows_match_frozen": row_passed,
                    "graph_request_rows_match_frozen": graph_rows_passed,
                    "physical_goal_matches_frozen": physical_passed,
                    "graph_goal_matches_direct": graph_goal_matches,
                    "written_goal_matches_direct": written_goal == trace.render(),
                    "passed": (
                        row_passed
                        and graph_rows_passed
                        and physical_passed
                        and graph_goal_matches
                        and written_goal == trace.render()
                    ),
                }
            )
    if not all(row["passed"] for row in fatal_rows):
        raise AssertionError("synthetic fatal exact oracle failed")

    return (
        {
            "run_configurations": [
                {"placement_epoch": epoch, "request_count": count}
                for epoch in (0, 1)
                for count in (1, 2, 3)
            ],
            "positive_identity": positive,
            "permutation_control": permutations,
            "fatal_exact_oracles": fatal_rows,
        },
        traces,
    )


def _granite_dims() -> Any:
    from simllm.compute import ModelDims

    return ModelDims(
        num_layers=24,
        hidden_size=1024,
        intermediate_size=512,
        num_heads=16,
        num_kv_heads=8,
        head_size=64,
        vocab_size=49_152,
        dtype_bytes=2,
        num_experts=32,
        top_k=8,
        moe_intermediate_size=512,
        local_num_experts=4,
    )


def _granite_supply(arguments: argparse.Namespace, record: Any) -> Any:
    from simllm.preplay import read_routed_experts
    from simllm.traffic import ExpertPlacementSnapshot, RoutedMoeSupply

    routed = read_routed_experts(arguments.source_root / SOURCE_ARTIFACTS["routing"][0])
    placement = ExpertPlacementSnapshot(
        placement_epoch=0,
        expert_owners=tuple(
            (layer, expert, expert % 8) for layer in range(24) for expert in range(32)
        ),
    )
    return RoutedMoeSupply(
        routed_experts=routed,
        placements=(placement,),
        step_placement_epochs=((record.step_index, 0),),
    )


def _canonical_granite_rows(rows: tuple[Any, ...]) -> bytes:
    phase_order = {"dispatch": 0, "combine": 1}
    result = []
    for operation_id, request_id, source, destination, size in rows:
        parts = operation_id.split(":")
        if len(parts) != 3 or not parts[1].startswith("layer-"):
            raise AssertionError(f"unexpected operation identity {operation_id!r}")
        layer = int(parts[1].removeprefix("layer-"))
        phase = parts[2].removeprefix("ep-")
        if phase not in phase_order:
            raise AssertionError(f"unexpected MoE phase {phase!r}")
        result.append([request_id, layer, phase, source, destination, size])
    result.sort(
        key=lambda row: (
            row[0],
            row[1],
            phase_order[row[2]],
            row[3],
            row[4],
        )
    )
    return (json.dumps(result, separators=(",", ":")) + "\n").encode()


def _run_granite(arguments: argparse.Namespace, out: Path) -> dict[str, Any]:
    from simllm.core import step_records_from_jsonl
    from simllm.traffic import compare_request_moe_fidelity, render_step_goal

    record = step_records_from_jsonl(arguments.source_root / SOURCE_ARTIFACTS["steps"][0])[0]
    dims = _granite_dims()
    supply = _granite_supply(arguments, record)
    trace = render_step_goal(
        record,
        dims,
        (0,),
        4_139,
        ep_ranks=tuple(range(8)),
        routed_supply=supply,
        num_goal_ranks=8,
    )
    report = compare_request_moe_fidelity(
        record,
        dims,
        tuple(range(8)),
        supply,
        trace.messages,
    )
    output_dir = out / "granite"
    output_dir.mkdir(parents=True, exist_ok=False)
    trace.write(output_dir / "step-000000.goal")

    permuted = _swap_requests(trace.messages, "r0", "r1")
    permutation_report = compare_request_moe_fidelity(
        record,
        dims,
        tuple(range(8)),
        supply,
        permuted,
    )
    gate_rejected = _gate_rejects(
        record,
        dims,
        tuple(range(8)),
        supply,
        permuted,
    )
    goal = _goal_observation(trace.render())
    expected_permutation = GRANITE_PERMUTATION
    permutation_passed = (
        permutation_report.aggregate_matches
        and permutation_report.aggregate_mismatch_count
        == expected_permutation["aggregate_mismatch_count"]
        and permutation_report.mismatch_count == expected_permutation["request_mismatch_count"]
        and permutation_report.l1_error_bytes == expected_permutation["l1_error_bytes"]
        and dict(permutation_report.request_delta_bytes)
        == expected_permutation["request_delta_bytes"]
        and goal["send_count"] == 2_688
        and goal["total_send_bytes"] == 207_499_264
        and gate_rejected
    )
    permutation = {
        **_report_observation(permutation_report),
        "physical_send_count": goal["send_count"],
        "physical_send_bytes": goal["total_send_bytes"],
        "expected": expected_permutation,
        "gate_rejected": gate_rejected,
        "passed": permutation_passed,
    }
    if not permutation_passed:
        raise AssertionError("PLAY-B3 failed before fatal exact-oracle evaluation")

    canonical_all = _canonical_granite_rows(report.observed_request_rows)
    canonical_rows = json.loads(canonical_all)
    request_oracles = {}
    for request_id, expected in GRANITE_REQUESTS.items():
        request_data = (
            json.dumps(
                [row for row in canonical_rows if row[0] == request_id],
                separators=(",", ":"),
            )
            + "\n"
        ).encode()
        observation = {
            "positive_rows": len(json.loads(request_data)),
            "total_bytes": sum(row[-1] for row in json.loads(request_data)),
            "canonical_bytes": len(request_data),
            "canonical_sha256": _sha256(request_data),
        }
        observation["matches_frozen"] = observation == {
            "positive_rows": expected[0],
            "total_bytes": expected[1],
            "canonical_bytes": expected[2],
            "canonical_sha256": expected[3],
        }
        request_oracles[request_id] = observation
        (output_dir / f"{request_id}.canonical.json").write_bytes(request_data)
    all_observation = {
        "positive_rows": len(canonical_rows),
        "total_bytes": sum(row[-1] for row in canonical_rows),
        "canonical_bytes": len(canonical_all),
        "canonical_sha256": _sha256(canonical_all),
    }
    all_observation["matches_frozen"] = all_observation == {
        "positive_rows": GRANITE_ALL[0],
        "total_bytes": GRANITE_ALL[1],
        "canonical_bytes": GRANITE_ALL[2],
        "canonical_sha256": GRANITE_ALL[3],
    }
    (output_dir / "all.canonical.json").write_bytes(canonical_all)
    physical_matches = goal == {
        "goal_bytes": 334_432,
        "goal_sha256": SOURCE_ARTIFACTS["aggregate_goal"][1],
        "send_count": 2_688,
        "total_send_bytes": 207_499_264,
    }
    if not (
        all(item["matches_frozen"] for item in request_oracles.values())
        and all_observation["matches_frozen"]
        and physical_matches
    ):
        raise AssertionError("Granite fatal exact oracle failed")
    return {
        "run_configuration": {
            "step_index": record.step_index,
            "scheduled_request_ids": [item.request_id for item in record.scheduled],
            "placement_rule": "expert_id modulo 8",
            "ep_ranks": list(range(8)),
        },
        "positive_identity": {
            **_report_observation(report),
            "classification": "fatal-unscored-entailment",
            "reason": (
                "the direct renderer requires the same fidelity relation before "
                "returning the trace observed here"
            ),
        },
        "permutation_control": permutation,
        "fatal_request_oracles": request_oracles,
        "fatal_all_request_oracle": all_observation,
        "fatal_physical_goal": {**goal, "matches_frozen": physical_matches},
    }


def _run_native_sanity(arguments: argparse.Namespace, out: Path) -> dict[str, Any]:
    from simllm.backends import HtsimStepSink, HtsimStepSinkConfig

    dims, supply = _synthetic_inputs()
    provider = _fixed_provider(2_000)
    rows = []
    for epoch in (0, 1):
        for count in (1, 2, 3):
            for bandwidth in (200_000_000_000, 400_000_000_000):
                record = _synthetic_record(epoch, count)
                workdir = (
                    out
                    / "native"
                    / f"epoch-{epoch}-requests-{count}"
                    / f"{bandwidth // 1_000_000_000}g"
                )
                sink = HtsimStepSink(
                    HtsimStepSinkConfig(
                        profile="rnic-nn-fluid",
                        tp_ranks=(0,),
                        dims=dims,
                        workdir=workdir,
                        ep_ranks=(0, 1),
                        linkspeed_bps=bandwidth,
                        provider=provider,
                        routed_moe_supply=supply,
                    )
                )
                result = sink(record)
                if result is None or len(sink.outcomes) != 1:
                    raise AssertionError("native sanity cell produced no outcome")
                outcome = sink.outcomes[0]
                expected = SYNTHETIC_JCT_PS[(epoch, count, bandwidth)]
                goal_observation = _goal_observation(
                    (workdir / f"step-{record.step_index:06d}.goal").read_text(encoding="utf-8")
                )
                expected_goal = SYNTHETIC_GOALS[(epoch, count)]
                structural_passed = (
                    result.completed_at_ps == result.step_latency_ps
                    and outcome.quiescent
                    and outcome.routing_mode == "captured"
                    and outcome.placement_epoch == epoch
                    and goal_observation["goal_sha256"] == expected_goal[1]
                )
                matches_frozen_jct = result.step_latency_ps == expected
                rows.append(
                    {
                        "epoch": epoch,
                        "request_count": count,
                        "bandwidth_bps": bandwidth,
                        "expected_jct_ps": expected,
                        "observed_jct_ps": result.step_latency_ps,
                        "residual_ps": result.step_latency_ps - expected,
                        "num_flows": outcome.num_flows,
                        "quiescent": outcome.quiescent,
                        "routing_mode": outcome.routing_mode,
                        "placement_epoch": outcome.placement_epoch,
                        "goal_sha256": goal_observation["goal_sha256"],
                        "matches_frozen_jct": matches_frozen_jct,
                        "structural_passed": structural_passed,
                    }
                )
    if not all(row["structural_passed"] for row in rows):
        raise AssertionError("whole-step native structural guard failed")
    deviations = [row for row in rows if not row["matches_frozen_jct"]]
    return {
        "classification": "sanity-unscored",
        "frozen_relation_status": (
            "matched" if not deviations else "unexpected-post-freeze-deviation"
        ),
        "claim_boundary": (
            "whole-step JCT proves live metric reachability but does not assign latency to requests"
        ),
        "frozen_formula_assumption": (
            "sum the larger directed payload in each globally serialized phase"
        ),
        "observed_schedule_semantics": (
            "participant-local completion frontiers allow adjacent asymmetric "
            "phases to overlap without a global collective barrier"
        ),
        "deviation_analysis": (
            "the frozen sum-of-phase-maxima formula overestimates cells where "
            "the critical direction changes; exact physical GOAL identity shows "
            "the attribution implementation did not change the schedule"
        ),
        "cells": rows,
        "completed_cells": len(rows),
        "frozen_matching_cells": len(rows) - len(deviations),
        "frozen_deviation_cells": len(deviations),
        "frozen_exact_relation_passed": not deviations,
        "structural_guards_passed": True,
        "timing_result_is_scored": False,
    }


def _run(arguments: argparse.Namespace) -> None:
    if arguments.out.exists():
        raise FileExistsError(f"output path already exists: {arguments.out}")
    inputs = _validate_inputs(arguments)
    os.environ["SIMLLM_HTSIM_RNIC"] = str(arguments.htsim_rnic)
    os.environ["SIMLLM_TXT2BIN"] = str(arguments.txt2bin)
    arguments.out.mkdir(parents=True)

    synthetic, _ = _run_synthetic(arguments.out)
    granite = _run_granite(arguments, arguments.out)
    native = _run_native_sanity(arguments, arguments.out)

    synthetic_instances = synthetic["permutation_control"]
    granite_instance = granite["permutation_control"]
    scored_families = [
        {
            "family": "PLAY-B2",
            "passed_instances": sum(row["passed"] for row in synthetic_instances),
            "total_instances": len(synthetic_instances),
            "passed": all(row["passed"] for row in synthetic_instances),
        },
        {
            "family": "PLAY-B3",
            "passed_instances": int(granite_instance["passed"]),
            "total_instances": 1,
            "passed": granite_instance["passed"],
        },
    ]
    summary = {
        "schema": "simllm-per-request-fidelity-study-v1",
        "expectations_commit": EXPECTATIONS_COMMIT,
        "provenance": {
            "run_head": _git_object("HEAD"),
            "run_observed_htsim_gitlink": _git_object("HEAD:third_party/htsim"),
            "expectations_authored_against_htsim_gitlink": _git_object(
                f"{EXPECTATIONS_COMMIT}:third_party/htsim"
            ),
            "gitlink_equality_required": False,
            "native_binaries": {
                "htsim_rnic": inputs.pop("htsim_rnic"),
                "txt2bin": inputs.pop("txt2bin"),
            },
        },
        "source_artifacts": inputs,
        "synthetic": synthetic,
        "granite": granite,
        "native_whole_step_sanity": native,
        "evidence_accounting": {
            "scored_families": scored_families,
            "passed_families": sum(row["passed"] for row in scored_families),
            "total_families": len(scored_families),
            "passed_instances": sum(row["passed_instances"] for row in scored_families),
            "total_instances": sum(row["total_instances"] for row in scored_families),
            "withdrawn_frozen_families": [
                {
                    "family": "PLAY-B1",
                    "frozen_instances": 6,
                    "classification": "fatal-unscored-entailment",
                    "reason": (
                        "render_step_goal requires the identity before exposing "
                        "the study observation"
                    ),
                },
                {
                    "family": "CORE-B1",
                    "frozen_instances": 6,
                    "classification": "fatal-unscored-entailment",
                    "reason": (
                        "render_serial_execution_graph_goal requires the identity "
                        "before exposing the study observation"
                    ),
                },
            ],
            "fatal_exact_or_structural_counts_are_scored": False,
            "sanity_counts_are_scored": False,
        },
        "claim_boundary": {
            "covered": (
                "captured MoE dispatch and combine bytes by scheduled request, "
                "layer and directed pair under the selected placement"
            ),
            "not_covered": [
                "per-request latency attribution",
                "KV-cache behavior",
                "expert compute fidelity",
                "gate weights",
                "TP collective attribution",
                "packet-level calibration",
            ],
            "step_latency_remains_whole_step_makespan": True,
        },
    }
    if not (
        summary["evidence_accounting"]["passed_families"]
        == summary["evidence_accounting"]["total_families"]
        and summary["evidence_accounting"]["passed_instances"]
        == summary["evidence_accounting"]["total_instances"]
    ):
        raise AssertionError("scored acceptance bar failed")
    _write_json(arguments.out / "summary.json", summary)
    print(
        "per-request fidelity: 2/2 genuine-risk families and 5/5 instances "
        "passed; native whole-step sanity completed with "
        f"{native['frozen_matching_cells']}/12 frozen JCT matches unscored"
    )


def main() -> None:
    arguments = _parse_args()
    _check_frozen_registry()
    if arguments.check_only:
        print(
            "check-only: parsed all paths and validated 6 GOAL, 4 permutation, "
            "12 JCT and 4 Granite literal rows; no artifacts produced"
        )
        return
    _run(arguments)


if __name__ == "__main__":
    main()
