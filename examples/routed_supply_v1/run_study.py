"""Run the three-section routed supply study after all freezes land."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SECTIONS = ("core", "play", "traffic")
CORE_UNIFORM_WIRE_SHA256 = (
    "f4a5a70f5bd4a0c2fed874baa88f3035266a54f386a59927e115872c2bcff0a3"
)
CORE_UNIFORM_GOAL_SHA256 = (
    "46ca1ea42952c5e0c66ea9eebb8947e770f7090f6cbdea6c711b4e764b412f5b"
)
PLAY_TRACE_SHA256 = {
    "tracked": "36334f3aaa767c46d5f9c8498e02f6c2805a46e5000a57aea2747e17dd5d1341",
    "full": "5d0ee3a1af045c404f9aa9baa7d063dc446584da60282f4492a1e72f08e081b5",
}
PLAY_PROJECTION_ORACLES = {
    "tracked": (
        30_874,
        "e3af45f896ff0a7005c4da0d6b4d3cfba7a00c868653e9aea581f49c37392e7a",
    ),
    "full": (
        87_845,
        "7d1875ac46de07f7ed2ed814dc8596ecc500a74f51c626a9b98b2ecb38d949d5",
    ),
    "reversed": (
        87_845,
        "18a5f737d1680aac22df3ca4a095d2f4ef5205c2433379de86ed96afc77687c1",
    ),
}
PLAY_ROW_COUNTS = {
    "tracked": (1, 22, 0, 528, 4224),
    "full": (3, 57, 6, 1512, 12096),
}


def parse_sections(value: str) -> tuple[str, ...]:
    sections = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = tuple(section for section in sections if section not in SECTIONS)
    if not sections or unknown or len(set(sections)) != len(sections):
        raise argparse.ArgumentTypeError(
            "sections must be a unique comma-separated subset of core,play,traffic"
        )
    return sections


def check_only(args: argparse.Namespace) -> None:
    if any(
        len(value) != 64
        for value in (CORE_UNIFORM_WIRE_SHA256, CORE_UNIFORM_GOAL_SHA256)
    ):
        raise AssertionError("frozen CORE SHA-256 literals must contain 64 digits")
    if any(len(value) != 64 for value in PLAY_TRACE_SHA256.values()):
        raise AssertionError("frozen PLAY trace hashes must contain 64 digits")
    if any(len(value[1]) != 64 or value[0] <= 0 for value in PLAY_PROJECTION_ORACLES.values()):
        raise AssertionError("frozen PLAY projection oracles are malformed")
    if any(any(count < 0 for count in counts) for counts in PLAY_ROW_COUNTS.values()):
        raise AssertionError("frozen PLAY row counts must be nonnegative")
    if "play" in args.sections and args.decode_trace is None:
        raise SystemExit("--decode-trace is required for the PLAY section")
    print(
        f"check-only sections={','.join(args.sections)} out={args.out}; "
        "validated frozen literals and produced no artifacts"
    )


def _core_graph(second_payload_bytes: int):
    from simllm.core import CollectiveWork, ExecutionGraph, ExecutionOperation

    return ExecutionGraph(
        "core6-variable",
        7,
        11,
        (
            ExecutionOperation(
                "a2av",
                0,
                "cuda:0:nccl:ep",
                CollectiveWork(
                    "all-to-allv",
                    (0, 1),
                    0,
                    "pairwise",
                    pair_payload_bytes=(
                        (0, 1, 2048),
                        (1, 0, second_payload_bytes),
                    ),
                ),
            ),
        ),
        ("a2av",),
    )


def _uniform_core_graph():
    from simllm.core import CollectiveWork, ExecutionGraph, ExecutionOperation

    return ExecutionGraph(
        "core6-uniform",
        7,
        11,
        (
            ExecutionOperation(
                "a2av",
                0,
                "cuda:0:nccl:ep",
                CollectiveWork("all-to-allv", (0, 1), 2048, "pairwise"),
            ),
        ),
        ("a2av",),
    )


def _goal_send_sizes(text: str) -> dict[tuple[int, int], int]:
    result: dict[tuple[int, int], int] = {}
    rank = -1
    for line in text.splitlines():
        if line.startswith("rank "):
            rank = int(line.split()[1])
        if ": send " not in line:
            continue
        words = line.split()
        size = int(words[2].removesuffix("b"))
        destination = int(words[4])
        result[(rank, destination)] = size
    return result


def run_core(out: Path) -> dict[str, object]:
    from simllm.core import (
        CoarseDeviceRuntime,
        EventPhase,
        ResourceKind,
        execution_graph_from_json,
        execution_graph_to_json,
    )
    from simllm.traffic import render_serial_execution_graph_goal

    out.mkdir(parents=True, exist_ok=True)
    uniform = _uniform_core_graph()
    uniform_wire = (
        json.dumps(
            execution_graph_to_json(uniform),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )
    uniform_round_trip = execution_graph_from_json(json.loads(uniform_wire))
    uniform_goal = render_serial_execution_graph_goal(uniform_round_trip).render().encode()
    uniform_checks = {
        "wire_bytes": len(uniform_wire),
        "wire_sha256": hashlib.sha256(uniform_wire).hexdigest(),
        "wire_matches_frozen": (
            hashlib.sha256(uniform_wire).hexdigest() == CORE_UNIFORM_WIRE_SHA256
        ),
        "goal_bytes": len(uniform_goal),
        "goal_sha256": hashlib.sha256(uniform_goal).hexdigest(),
        "goal_matches_frozen": (
            hashlib.sha256(uniform_goal).hexdigest() == CORE_UNIFORM_GOAL_SHA256
        ),
        "table_absent": b"pair_payload_bytes" not in uniform_wire,
    }
    uniform_checks["passed"] = all(
        uniform_checks[key]
        for key in ("wire_matches_frozen", "goal_matches_frozen", "table_absent")
    )

    instances: list[dict[str, object]] = []
    for second_payload_bytes in (4096, 6144):
        graph = _core_graph(second_payload_bytes)
        round_trip = execution_graph_from_json(execution_graph_to_json(graph))
        goal_sizes = _goal_send_sizes(
            render_serial_execution_graph_goal(round_trip).render()
        )
        runtime = CoarseDeviceRuntime()
        result = runtime.execute(round_trip)
        assert runtime.last_report is not None
        transfer_sizes = sorted(
            visit.service_bytes
            for visit in runtime.last_report.visits
            if visit.resource.kind is ResourceKind.NVLINK
        )
        completed_bytes = [
            event.completed_bytes
            for event in result.events
            if event.operation_id == "a2av"
            and event.phase is EventPhase.COMPLETED
            and event.subject_object_id is None
        ]
        instances.append(
            {
                "second_payload_bytes": second_payload_bytes,
                "goal_send_bytes": {
                    f"{source}->{destination}": size
                    for (source, destination), size in sorted(goal_sizes.items())
                },
                "runtime_transfer_bytes": transfer_sizes,
                "runtime_completed_bytes": completed_bytes,
                "passed": (
                    goal_sizes
                    == {(0, 1): 2048, (1, 0): second_payload_bytes}
                    and transfer_sizes == [2048, second_payload_bytes]
                    and completed_bytes == [2048 + second_payload_bytes]
                ),
            }
        )
    summary = {
        "freeze_commit": "91bed6fd201e4fa1d810e3322905632bb54714c6",
        "behavioral": {
            "CORE-B1": {
                "instances": instances,
                "passed": all(instance["passed"] for instance in instances),
            }
        },
        "exact_oracle": {"CORE-E1": uniform_checks},
    }
    (out / "core_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    if not summary["behavioral"]["CORE-B1"]["passed"] or not uniform_checks[
        "passed"
    ]:
        raise AssertionError("CORE section failed its frozen acceptance bar")
    return summary


def _canonical_projection_bytes(projection) -> bytes:
    from simllm.preplay import routed_experts_to_json

    return (
        json.dumps(
            routed_experts_to_json(projection),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )


def _join_projection(trace_path: Path, request_ids: tuple[str, ...]):
    from simllm.core import RequestBookkeeper
    from simllm.preplay import (
        RequestArrival,
        join_preplay_arrivals,
        project_preplay_routing,
    )

    run = join_preplay_arrivals(
        tuple(
            RequestArrival(request_id=request_id, arrived_at_ps=index * 1_000)
            for index, request_id in enumerate(request_ids)
        ),
        trace_path,
        RequestBookkeeper(),
    )
    return project_preplay_routing(run)


def _projection_counts(projection) -> tuple[int, int, int, int, int]:
    requests = len(projection.requests)
    prefill = sum(len(request.prefill_tokens) for request in projection.requests)
    decode = sum(len(request.decode_tokens) for request in projection.requests)
    layers = sum(
        len(token.layers)
        for request in projection.requests
        for token in request.tokens
    )
    assignments = sum(
        len(layer.expert_ids)
        for request in projection.requests
        for token in request.tokens
        for layer in token.layers
    )
    return requests, prefill, decode, layers, assignments


def _projection_oracle(projection) -> dict[str, object]:
    payload = _canonical_projection_bytes(projection)
    return {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "counts": list(_projection_counts(projection)),
    }


def run_play(out: Path, decode_trace: Path) -> dict[str, object]:
    from simllm.preplay import read_preplay_trace, routed_experts_to_json

    out.mkdir(parents=True, exist_ok=True)
    tracked_trace = (
        Path(__file__).resolve().parents[1]
        / "preplay_trace_v1/granite_length_cap.jsonl"
    )
    full_trace = read_preplay_trace(decode_trace)
    full_order = tuple(request.request_id for request in full_trace.requests)
    reversed_order = tuple(reversed(full_order))
    tracked = _join_projection(tracked_trace, ("length-cap",))
    full = _join_projection(decode_trace, full_order)
    reversed_projection = _join_projection(decode_trace, reversed_order)
    projections = {
        "tracked": tracked,
        "full": full,
        "reversed": reversed_projection,
    }
    oracles = {
        name: _projection_oracle(projection)
        for name, projection in projections.items()
    }
    for name, projection in projections.items():
        (out / f"play_{name}.json").write_bytes(
            _canonical_projection_bytes(projection)
        )

    expected_counts = {
        "tracked": PLAY_ROW_COUNTS["tracked"],
        "full": PLAY_ROW_COUNTS["full"],
    }
    exact_checks = {
        name: (
            oracles[name]["bytes"] == PLAY_PROJECTION_ORACLES[name][0]
            and oracles[name]["sha256"] == PLAY_PROJECTION_ORACLES[name][1]
        )
        for name in projections
    }
    exact_checks["tracked_trace_sha256"] = (
        tracked.trace_sha256 == PLAY_TRACE_SHA256["tracked"]
    )
    exact_checks["full_trace_sha256"] = (
        full.trace_sha256 == PLAY_TRACE_SHA256["full"]
    )
    exact_checks["tracked_counts"] = (
        _projection_counts(tracked) == expected_counts["tracked"]
    )
    exact_checks["full_counts"] = (
        _projection_counts(full) == expected_counts["full"]
    )

    terminal_rows = []
    for request in full.requests:
        terminal_rows.append(
            {
                "request_id": request.request_id,
                "prompt_tokens": request.prompt_token_count,
                "output_tokens": request.output_token_count,
                "prefill_forwards": len(request.prefill_tokens),
                "decode_forwards": len(request.decode_tokens),
                "passed": (
                    len(request.prefill_tokens) == request.prompt_token_count
                    and len(request.decode_tokens)
                    == request.output_token_count - 1
                ),
            }
        )

    full_payloads = {
        request["request_id"]: request
        for request in routed_experts_to_json(full)["requests"]
    }
    reversed_payloads = {
        request["request_id"]: request
        for request in routed_experts_to_json(reversed_projection)["requests"]
    }
    association = {
        "full_order": list(full_order),
        "reversed_order": list(reversed_order),
        "reversed_projection_order": [
            request.request_id for request in reversed_projection.requests
        ],
        "per_request_payloads_equal": full_payloads == reversed_payloads,
    }
    association["passed"] = (
        association["reversed_projection_order"] == list(reversed_order)
        and association["per_request_payloads_equal"]
    )
    summary = {
        "freeze_commit": "a778cd9fea3e0ed1d2ec5250c148c068396aa497",
        "behavioral": {
            "PLAY-B1": {
                "rows": terminal_rows,
                "passed": all(row["passed"] for row in terminal_rows),
            },
            "PLAY-B2": association,
        },
        "exact_oracle": {
            "PLAY-E1": {
                "projections": oracles,
                "checks": exact_checks,
                "passed": all(exact_checks.values()),
            }
        },
    }
    (out / "play_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    passed = (
        summary["behavioral"]["PLAY-B1"]["passed"]
        and summary["behavioral"]["PLAY-B2"]["passed"]
        and summary["exact_oracle"]["PLAY-E1"]["passed"]
    )
    if not passed:
        raise AssertionError("PLAY section failed its frozen acceptance bar")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sections", type=parse_sections, default=SECTIONS)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--decode-trace", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if "play" in args.sections and args.decode_trace is None:
        parser.error("--decode-trace is required for the PLAY section")
    if args.check_only:
        check_only(args)
        return
    summaries = {}
    if "core" in args.sections:
        summaries["core"] = run_core(args.out)
    if "play" in args.sections:
        summaries["play"] = run_play(args.out, args.decode_trace)
    unsupported = tuple(section for section in args.sections if section == "traffic")
    if unsupported:
        raise SystemExit(
            "result-producing sections land only after their expectation freezes: "
            + ",".join(unsupported)
        )
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
