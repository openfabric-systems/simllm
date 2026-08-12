"""Run the frozen routing-lifetime representation and close-out study."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

EXPECTATIONS_COMMIT = "6fa7c4acc059a16ac2b1054f9538358404dc74ce"


SOURCE_ARTIFACTS = {
    "capture": (
        "capture/granite-greedy.jsonl",
        "5f55fccc265ee5519430a0f73d6631e49aa547ec07fcb81034e9fc2b4d9fead6",
        None,
    ),
    "run": (
        "replay-400g/run.json",
        "b4d38a09011caf6de159c22133264d62a2727063496953f4337b17d79cfde93e",
        1_831,
    ),
    "steps": (
        "replay-400g/steps.jsonl",
        "824cd9557293328bb42b593ac893b6a067302e545b087c9219195ccb8031d755",
        12_666,
    ),
    "routing": (
        "replay-400g/routed-experts.json",
        "24e986e989e21f1bfe7e758d4470928c82c3bbaec06072a839743b9b17d7cf5f",
        159_957,
    ),
    "prefill_goal": (
        "replay-400g/htsim/step-000000.goal",
        "08a0403af66ff8a9d6b18f93afd15ae0bc925cc85555acf8a0593438a3d7bc92",
        334_432,
    ),
}

MEMORY_CELLS = {
    1: {
        "tokens": 45,
        "legacy_bytes_per_token_band": (6_000, 6_600),
        "arena_bytes_per_token_max": 192,
        "reduction_band": (32, 34),
    },
    3: {
        "tokens": 115,
        "legacy_bytes_per_token_band": (6_000, 6_600),
        "arena_bytes_per_token_max": 192,
        "reduction_band": (32, 34),
    },
}

LIFECYCLE_CELLS = {
    "clean-one": {"request_count": 1, "closed": 1, "live": 0, "views": 0},
    "clean-three": {"request_count": 3, "closed": 3, "live": 0, "views": 0},
    "suppress-dispatch": {
        "request_id": "r0",
        "phase": "dispatch",
        "layer": 7,
    },
    "suppress-combine": {
        "request_id": "r2",
        "phase": "combine",
        "layer": 19,
    },
}


def _check_frozen_registry() -> None:
    if set(MEMORY_CELLS) != {1, 3}:
        raise AssertionError("memory registry must contain one and three requests")
    if [MEMORY_CELLS[count]["tokens"] for count in (1, 3)] != [45, 115]:
        raise AssertionError("memory token counts disagree with the frozen capture")
    if set(LIFECYCLE_CELLS) != {
        "clean-one",
        "clean-three",
        "suppress-dispatch",
        "suppress-combine",
    }:
        raise AssertionError("lifecycle registry is incomplete")
    if any(len(digest) != 64 for _, digest, _ in SOURCE_ARTIFACTS.values()):
        raise AssertionError("source digests must contain 64 hexadecimal digits")
    suppressions = {
        (value["request_id"], value["phase"], value["layer"])
        for name, value in LIFECYCLE_CELLS.items()
        if name.startswith("suppress-")
    }
    if suppressions != {("r0", "dispatch", 7), ("r2", "combine", 19)}:
        raise AssertionError("suppression registry disagrees with expectations")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _path_observation(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"bytes": len(data), "sha256": _sha256(data)}


def _git_object(revision: str) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", revision],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _validate_inputs(arguments: argparse.Namespace) -> dict[str, Any]:
    observations = {}
    for name, (relative_path, digest, size) in SOURCE_ARTIFACTS.items():
        path = arguments.source_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"missing source artifact: {path}")
        observation = _path_observation(path)
        observation.update(
            relative_path=relative_path,
            expected_sha256=digest,
            matches_sha256=observation["sha256"] == digest,
        )
        if size is not None:
            observation["expected_bytes"] = size
            observation["matches_bytes"] = observation["bytes"] == size
        if not observation["matches_sha256"] or not observation.get(
            "matches_bytes", True
        ):
            raise AssertionError(f"source artifact changed: {relative_path}")
        observations[name] = observation
    capture_path = arguments.source_root / SOURCE_ARTIFACTS["capture"][0]
    observations["capture"]["line_count"] = len(capture_path.read_bytes().splitlines())
    if observations["capture"]["line_count"] != 120:
        raise AssertionError("capture row count changed")
    return observations


def _retained_size(root: object) -> int:
    seen: set[int] = set()

    def visit(value: object) -> int:
        identity = id(value)
        if identity in seen:
            return 0
        seen.add(identity)
        size = sys.getsizeof(value)
        if isinstance(value, dict):
            size += sum(visit(key) + visit(item) for key, item in value.items())
        elif isinstance(value, (tuple, list, set, frozenset)):
            size += sum(visit(item) for item in value)
        elif hasattr(value, "__dict__"):
            size += visit(vars(value))
        return size

    return visit(root)


def _fixed_provider(duration_ps: int) -> Any:
    from simllm.compute import ComputeProvider, DurationEstimate

    class FixedProvider(ComputeProvider):
        def estimate(self, kernel: Any, gpu: Any) -> DurationEstimate:
            return DurationEstimate(duration_ps=duration_ps, bound="measured")

    return FixedProvider()


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


def _placement() -> Any:
    from simllm.traffic import ExpertPlacementSnapshot

    return ExpertPlacementSnapshot(
        placement_epoch=0,
        expert_owners=tuple(
            (layer, expert, expert % 8)
            for layer in range(24)
            for expert in range(32)
        ),
    )


def _join_cell(
    arguments: argparse.Namespace,
    output: Path,
    external_run: Any,
    request_count: int,
) -> tuple[Any, Any]:
    from simllm.core import RequestBookkeeper
    from simllm.preplay import (
        RequestArrival,
        join_preplay_arrivals,
        open_routing_arena,
        write_preplay_replay_run,
    )

    directory = output / f"requests-{request_count}"
    directory.mkdir(parents=True, exist_ok=False)
    arrivals = tuple(
        RequestArrival(
            request_id=request.request_id,
            arrived_at_ps=request.arrived_at_ps,
        )
        for request in external_run.requests[:request_count]
    )
    index_path = directory / "run.routing.json"
    joined = join_preplay_arrivals(
        arrivals,
        arguments.source_root / SOURCE_ARTIFACTS["capture"][0],
        RequestBookkeeper(),
        routing_arena_index_path=index_path,
    )
    write_preplay_replay_run(joined, directory / "run.json")
    arena = open_routing_arena(index_path, expected_run=joined)
    return joined, arena


def _memory_rows(
    routed: Any,
    cells: dict[int, tuple[Any, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for request_count in (1, 3):
        _, arena = cells[request_count]
        legacy = replace(routed, requests=routed.requests[:request_count])
        tokens = sum(len(request.tokens) for request in legacy.requests)
        legacy_bytes = _retained_size(legacy)
        arena_bytes = arena.index.payload_bytes
        legacy_per_token = legacy_bytes / tokens
        arena_per_token = arena_bytes / tokens
        reduction = legacy_bytes / arena_bytes
        expected = MEMORY_CELLS[request_count]
        legacy_band_passed = (
            tokens == expected["tokens"]
            and expected["legacy_bytes_per_token_band"][0]
            <= legacy_per_token
            <= expected["legacy_bytes_per_token_band"][1]
        )
        reduction_passed = (
            expected["reduction_band"][0]
            <= reduction
            <= expected["reduction_band"][1]
        )
        rows.append(
            {
                "request_count": request_count,
                "forwarded_tokens": tokens,
                "legacy_retained_bytes": legacy_bytes,
                "arena_payload_bytes": arena_bytes,
                "legacy_bytes_per_token": legacy_per_token,
                "arena_bytes_per_token": arena_per_token,
                "reduction": reduction,
                "legacy_band_passed": legacy_band_passed,
                "reduction_passed": reduction_passed,
                "arena_bound_passed": (
                    arena_per_token <= expected["arena_bytes_per_token_max"]
                ),
                "direction_passed": arena_per_token < legacy_per_token,
                "passed": legacy_band_passed and reduction_passed,
            }
        )
    return rows


def _supply(
    routed: Any,
    steps: tuple[Any, ...],
    *,
    arena: Any | None = None,
    lifetimes: Any | None = None,
) -> Any:
    from simllm.traffic import RoutedMoeSupply

    keywords = (
        {"routed_experts": routed}
        if arena is None
        else {"routing_arena": arena, "lifetimes": lifetimes}
    )
    return RoutedMoeSupply(
        placements=(_placement(),),
        step_placement_epochs=tuple((record.step_index, 0) for record in steps),
        **keywords,
    )


def _traffic_identity(
    arguments: argparse.Namespace,
    routed: Any,
    steps: tuple[Any, ...],
    arena: Any,
    lifetimes: Any,
) -> tuple[dict[str, Any], Any, Any]:
    from simllm.backends import SerialStepLowerer, SerialStepLowererConfig
    from simllm.core import execution_graph_to_json
    from simllm.traffic import (
        render_serial_execution_graph_goal,
        render_step_goal,
        step_moe_alltoalls,
    )

    dims = _granite_dims()
    legacy_supply = _supply(routed, steps)
    arena_supply = _supply(
        routed,
        steps,
        arena=arena,
        lifetimes=lifetimes,
    )
    provider = _fixed_provider(24 * 4_139_000)
    legacy_lowerer = SerialStepLowerer(
        SerialStepLowererConfig(
            dims=dims,
            tp_ranks=(0,),
            ep_ranks=tuple(range(8)),
            provider=provider,
            routed_moe_supply=legacy_supply,
        )
    )
    arena_lowerer = SerialStepLowerer(
        SerialStepLowererConfig(
            dims=dims,
            tp_ranks=(0,),
            ep_ranks=tuple(range(8)),
            provider=provider,
            routed_moe_supply=arena_supply,
        )
    )
    rows = []
    for record in steps:
        legacy_operations = step_moe_alltoalls(
            record,
            dims,
            tuple(range(8)),
            routed_supply=legacy_supply,
        )
        arena_operations = step_moe_alltoalls(
            record,
            dims,
            tuple(range(8)),
            routed_supply=arena_supply,
        )
        operations_match = legacy_operations == arena_operations
        legacy_goal = render_step_goal(
            record,
            dims,
            (0,),
            4_139,
            ep_ranks=tuple(range(8)),
            routed_supply=legacy_supply,
            num_goal_ranks=8,
        ).render().encode()
        arena_goal = render_step_goal(
            record,
            dims,
            (0,),
            4_139,
            ep_ranks=tuple(range(8)),
            routed_supply=arena_supply,
            num_goal_ranks=8,
        ).render().encode()
        direct_goal_match = legacy_goal == arena_goal
        legacy_graph = legacy_lowerer.lower(record)
        arena_graph = arena_lowerer.lower(record)
        graph_match = (
            execution_graph_to_json(legacy_graph)
            == execution_graph_to_json(arena_graph)
        )
        legacy_graph_goal = render_serial_execution_graph_goal(
            legacy_graph,
            num_goal_ranks=8,
        ).render().encode()
        arena_graph_goal = render_serial_execution_graph_goal(
            arena_graph,
            num_goal_ranks=8,
        ).render().encode()
        graph_goal_match = legacy_graph_goal == arena_graph_goal
        rows.append(
            {
                "step_index": record.step_index,
                "operation_count": len(legacy_operations),
                "operations_match": operations_match,
                "direct_goal_bytes": len(legacy_goal),
                "direct_goal_sha256": _sha256(legacy_goal),
                "direct_goal_match": direct_goal_match,
                "graph_match": graph_match,
                "graph_goal_bytes": len(legacy_graph_goal),
                "graph_goal_sha256": _sha256(legacy_graph_goal),
                "graph_goal_match": graph_goal_match,
                "passed": (
                    operations_match
                    and direct_goal_match
                    and graph_match
                    and graph_goal_match
                ),
            }
        )
    archived_goal = (
        arguments.source_root / SOURCE_ARTIFACTS["prefill_goal"][0]
    ).read_bytes()
    step_zero_matches_archive = rows[0]["direct_goal_sha256"] == _sha256(
        archived_goal
    ) and rows[0]["direct_goal_bytes"] == len(archived_goal)
    result = {
        "classification": "fatal-unscored",
        "rows": rows,
        "step_zero_matches_archived_goal": step_zero_matches_archive,
        "passed": all(row["passed"] for row in rows)
        and step_zero_matches_archive,
    }
    return result, legacy_supply, arena_supply


def _filtered_record(source: Any, request_ids: set[str], virtual_time_ps: int) -> Any:
    scheduled = [
        replace(request)
        for request in source.scheduled
        if request.request_id in request_ids
    ]
    sampled = source.sampled_request_ids
    return replace(
        source,
        virtual_time_ps=virtual_time_ps,
        scheduled=scheduled,
        preempted_request_ids=[
            request_id
            for request_id in source.preempted_request_ids
            if request_id in request_ids
        ],
        finished_request_ids=[
            request_id
            for request_id in source.finished_request_ids
            if request_id in request_ids
        ],
        num_sampled=len(scheduled),
        num_tokens_after_padding=None,
        sampled_request_ids=(
            None
            if sampled is None
            else [request_id for request_id in sampled if request_id in request_ids]
        ),
    )


def _runtime_report_compatible_graph(graph: Any) -> Any:
    """Tighten local frontiers to barriers for the coarse report reducer.

    The coarse runtime can execute participant-local stagger, but its additive
    operation report currently rejects a later rank whose selected local path
    starts before the operation's one global critical predecessor. This study
    makes no timing claim, so it preserves every operation, request, layer,
    work item and completion boundary while tightening only those dependency
    frontiers to whole-operation barriers.
    """

    from simllm.core import ExecutionGraph, validate_execution_graph

    operations = tuple(
        replace(
            operation,
            depends_on=tuple(
                dict.fromkeys(
                    (*operation.depends_on, *operation.participant_local_depends_on)
                )
            ),
            participant_local_depends_on=(),
        )
        for operation in graph.operations
    )
    tightened = ExecutionGraph(
        execution_id=graph.execution_id,
        step_index=graph.step_index,
        released_at_ps=graph.released_at_ps,
        operations=operations,
        completion_operation_ids=graph.completion_operation_ids,
    )
    validate_execution_graph(tightened)
    return tightened


def _run_clean_lifecycle(
    joined: Any,
    arena: Any,
    supply: Any,
    source_steps: tuple[Any, ...],
    request_ids: tuple[str, ...],
) -> tuple[dict[str, Any], list[tuple[Any, Any, Any, Any]]]:
    from simllm.backends import SerialStepLowerer, SerialStepLowererConfig
    from simllm.core import (
        CoarseDeviceProfile,
        CoarseDeviceRuntime,
        CompletionReducer,
        StepRecord,
        VirtualClock,
    )

    del joined
    dims = _granite_dims()
    lowerer = SerialStepLowerer(
        SerialStepLowererConfig(
            dims=dims,
            tp_ranks=(0,),
            ep_ranks=tuple(range(8)),
            provider=_fixed_provider(24 * 4_139_000),
            routed_moe_supply=supply,
        )
    )
    lifetimes = supply.lifetimes
    clock = VirtualClock(0)
    reducer = CompletionReducer(clock, lifetimes=lifetimes)
    runtime = CoarseDeviceRuntime(CoarseDeviceProfile())
    wanted = set(request_ids)
    selected = source_steps[:25] if request_ids == ("r0",) else source_steps
    evidence: list[tuple[Any, Any, Any, Any]] = []
    states = []
    for source in selected:
        record = _filtered_record(source, wanted, clock.now_ps)
        graph = _runtime_report_compatible_graph(lowerer.lower(record))
        execution = runtime.execute(graph)
        report = runtime.last_report
        assert report is not None
        reducer.reduce(record, graph, execution, report)
        evidence.append((record, graph, execution, report))
        states.append(
            {
                "step_index": record.step_index,
                "requests": [
                    {
                        "request_id": lifetime.request_id,
                        "state": lifetime.state.value,
                        "cursor": lifetime.consumption_cursor,
                        "dispatch_mask": lifetime.dispatch_end_mask,
                        "combine_mask": lifetime.combine_end_mask,
                        "view_released": lifetime.view_released,
                    }
                    for lifetime in lifetimes.requests
                ],
            }
        )
    if request_ids != ("r0",):
        drain_source = StepRecord(
            step_index=32,
            virtual_time_ps=clock.now_ps,
            finished_request_ids=["r2"],
            num_sampled=0,
            sampled_request_ids=[],
        )
        graph = _runtime_report_compatible_graph(lowerer.lower(drain_source))
        execution = runtime.execute(graph)
        report = runtime.last_report
        assert report is not None
        reducer.reduce(drain_source, graph, execution, report)
        evidence.append((drain_source, graph, execution, report))
        states.append(
            {
                "step_index": 32,
                "requests": [
                    {
                        "request_id": lifetime.request_id,
                        "state": lifetime.state.value,
                        "cursor": lifetime.consumption_cursor,
                        "dispatch_mask": lifetime.dispatch_end_mask,
                        "combine_mask": lifetime.combine_end_mask,
                        "view_released": lifetime.view_released,
                    }
                    for lifetime in lifetimes.requests
                ],
            }
        )
    raw = {
        "closed": lifetimes.closed_request_count,
        "live": lifetimes.live_request_count,
        "views": lifetimes.live_view_count,
    }
    expected = LIFECYCLE_CELLS[
        "clean-one" if request_ids == ("r0",) else "clean-three"
    ]
    passed = raw == {
        "closed": expected["closed"],
        "live": expected["live"],
        "views": expected["views"],
    }
    # LIFE-B1 is evaluated from raw counts before this entailing fatal audit.
    lifetimes.audit_closed()
    arena.close()
    return {
        "request_ids": list(request_ids),
        "raw_exit": raw,
        "state_trace": states,
        "passed": passed,
    }, evidence


def _suppression_registry(
    joined: Any,
    arena: Any,
    target: tuple[str, str, int],
) -> Any:
    from simllm.core import (
        JoinProvenance,
        RequestLifetimeRegistry,
        RoutingViewDescriptor,
    )

    class SuppressingRegistry(RequestLifetimeRegistry):
        def _mark_end_flag(self, record: Any, phase: str, layer: int) -> None:
            if (record.request_id, phase, layer) == target:
                return
            super()._mark_end_flag(record, phase, layer)

    registry = SuppressingRegistry(arena.moe_layer_indices)
    provenance = JoinProvenance(
        run_schema=joined.schema,
        trace_schema=joined.trace.schema,
        trace_sha256=joined.trace.sha256,
    )
    for joined_request in joined.requests:
        view = arena.acquire_request(joined_request.request_id)
        registry.register(
            joined_request.request_id,
            provenance,
            joined_request.arrived_at_ps,
            RoutingViewDescriptor(
                arena_id=view.arena_id,
                token_offset=view.token_offset,
                token_count=view.token_count,
                prompt_token_count=view.prompt_token_count,
                release_callback=view.release,
            ),
        )
    return registry


def _run_suppression(
    joined: Any,
    arena: Any,
    evidence: list[tuple[Any, Any, Any, Any]],
    target: tuple[str, str, int],
) -> dict[str, Any]:
    from simllm.core import (
        CollectiveWork,
        CompletionReducer,
        EventPhase,
        VirtualClock,
    )

    registry = _suppression_registry(joined, arena, target)
    clock = VirtualClock(0)
    reducer = CompletionReducer(clock, lifetimes=registry)
    for record, graph, execution, report in evidence:
        reducer.reduce(record, graph, execution, report)
    request_id, phase, layer = target
    target_token_count = registry.by_request_id(request_id).token_count
    final_execution_ids = {
        graph.execution_id
        for record, graph, _, _ in evidence
        if any(
            scheduled.request_id == request_id
            and scheduled.num_new_tokens > 0
            and scheduled.context_length == target_token_count
            for scheduled in record.scheduled
        )
    }
    matching_ids = {
        operation.operation_id
        for _, graph, _, _ in evidence
        if graph.execution_id in final_execution_ids
        for operation in graph.operations
        if request_id in operation.correlation.request_ids
        and operation.correlation.layer == layer
        and isinstance(operation.work, CollectiveWork)
        and operation.work.channel_hint == phase
    }
    raw_event_count = sum(
        event.phase is EventPhase.COMPLETED
        and event.subject_object_id is None
        and event.operation_id in matching_ids
        for _, _, execution, _ in evidence
        for event in execution.events
    )
    diagnostic = None
    try:
        registry.audit_closed()
    except RuntimeError as exc:
        diagnostic = str(exc)
    lifetime = registry.by_request_id(request_id)
    passed = (
        raw_event_count >= 1
        and diagnostic is not None
        and request_id in diagnostic
        and phase in diagnostic
        and str(layer) in diagnostic
        and not lifetime.view_released
    )
    # The intentionally live view prevents arena.close(), which is itself a
    # fatal-unscored ownership guard. Process exit releases the read-only mmap.
    close_rejected = False
    try:
        arena.close()
    except BufferError:
        close_rejected = True
    return {
        "request_id": request_id,
        "phase": phase,
        "layer": layer,
        "raw_subjectless_completion_count": raw_event_count,
        "diagnostic": diagnostic,
        "state": lifetime.state.value,
        "view_released": lifetime.view_released,
        "arena_close_with_live_view_rejected": close_rejected,
        "passed": passed,
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _run(arguments: argparse.Namespace) -> dict[str, Any]:
    from simllm.core import step_records_from_jsonl
    from simllm.preplay import (
        build_routing_arena,
        create_request_lifetimes,
        read_preplay_replay_run,
        read_routed_experts,
    )

    inputs = _validate_inputs(arguments)
    arguments.out.mkdir(parents=True, exist_ok=False)
    external_run = read_preplay_replay_run(
        arguments.source_root / SOURCE_ARTIFACTS["run"][0]
    )
    steps = step_records_from_jsonl(
        arguments.source_root / SOURCE_ARTIFACTS["steps"][0]
    )
    if len(steps) != 32:
        raise AssertionError("expected 32 recorded scheduler steps")
    routed = read_routed_experts(
        arguments.source_root / SOURCE_ARTIFACTS["routing"][0]
    )
    cells = {
        count: _join_cell(arguments, arguments.out, external_run, count)
        for count in (1, 3)
    }

    # Raw retained sizes are evaluated before the explicit layout oracle below.
    memory = _memory_rows(routed, cells)
    memory_passed = all(row["passed"] for row in memory)
    layout_rows = []
    for request_count, (_, arena) in cells.items():
        expected_bytes = (
            MEMORY_CELLS[request_count]["tokens"]
            * len(arena.moe_layer_indices)
            * arena.top_k
        )
        layout_rows.append(
            {
                "request_count": request_count,
                "expected_payload_bytes": expected_bytes,
                "observed_payload_bytes": arena.index.payload_bytes,
                "passed": arena.index.payload_bytes == expected_bytes,
            }
        )

    joined_three, arena_three = cells[3]
    lifetimes_three = create_request_lifetimes(joined_three, arena_three)
    traffic, _, arena_supply_three = _traffic_identity(
        arguments,
        routed,
        steps,
        arena_three,
        lifetimes_three,
    )

    joined_one, arena_one = cells[1]
    lifetimes_one = create_request_lifetimes(joined_one, arena_one)
    one_supply = _supply(
        replace(routed, requests=routed.requests[:1]),
        steps,
        arena=arena_one,
        lifetimes=lifetimes_one,
    )
    clean_one, evidence_one = _run_clean_lifecycle(
        joined_one,
        arena_one,
        one_supply,
        steps,
        ("r0",),
    )
    clean_three, evidence_three = _run_clean_lifecycle(
        joined_three,
        arena_three,
        arena_supply_three,
        steps,
        ("r0", "r1", "r2"),
    )

    suppression_root = arguments.out / "suppression"
    suppression_root.mkdir()
    suppression_rows = []
    for name, joined, evidence in (
        ("suppress-dispatch", joined_one, evidence_one),
        ("suppress-combine", joined_three, evidence_three),
    ):
        target_value = LIFECYCLE_CELLS[name]
        target = (
            str(target_value["request_id"]),
            str(target_value["phase"]),
            int(target_value["layer"]),
        )
        arena = build_routing_arena(
            joined,
            suppression_root / f"{name}.routing.json",
        )
        suppression_rows.append(_run_suppression(joined, arena, evidence, target))

    scored = {
        "MEM-B1": {
            "classification": "scored-behavioral",
            "genuine_risk": "2/2",
            "rows": memory,
            "passed": memory_passed,
        },
        "LIFE-B1": {
            "classification": "scored-behavioral",
            "genuine_risk": "2/2",
            "rows": [clean_one, clean_three],
            "passed": clean_one["passed"] and clean_three["passed"],
        },
        "LIFE-B2": {
            "classification": "scored-behavioral",
            "genuine_risk": "2/2",
            "rows": suppression_rows,
            "passed": all(row["passed"] for row in suppression_rows),
        },
    }
    fatal = {
        "input_identity": {
            "classification": "fatal-unscored-configuration",
            "passed": True,
        },
        "uint8_layout": {
            "classification": "fatal-unscored-structural",
            "rows": layout_rows,
            "passed": all(row["passed"] for row in layout_rows),
        },
        "traffic_identity": traffic,
        "suppression_view_retention": {
            "classification": "fatal-unscored-structural",
            "passed": all(
                row["arena_close_with_live_view_rejected"]
                for row in suppression_rows
            ),
        },
    }
    result = {
        "schema": "simllm-routing-lifetime-study-results-v1",
        "provenance": {
            "expectations_commit": EXPECTATIONS_COMMIT,
            "run_head": _git_object("HEAD"),
            "observed_htsim_gitlink": _git_object("HEAD:third_party/htsim"),
            "gitlink_equality_required": False,
            "python": sys.version,
        },
        "inputs": inputs,
        "run_configuration": {
            "request_prefixes": [1, 3],
            "recorded_steps": 32,
            "explicit_drain_step": 32,
            "moe_layers": 24,
            "experts": 32,
            "top_k": 8,
            "ep_ranks": list(range(8)),
            "placement_rule": "expert_id modulo 8",
            "lifecycle_dependency_projection": (
                "participant-local frontiers tightened to whole-operation barriers "
                "for coarse additive-report compatibility after run-1 exposed its "
                "overlap rejection; operation, work, request, layer and completion "
                "identities are unchanged"
            ),
        },
        "scored_relations": scored,
        "fatal_unscored": fatal,
        "evidence_class_counts": {
            "run_configurations": 2,
            "scored_behavioral_families": 3,
            "scored_behavioral_instances": 6,
            "fatal_exact_step_rows": len(traffic["rows"]),
            "fatal_structural_families": 3,
            "native_test_executables": 0,
        },
    }
    _write_json(arguments.out / "results.json", result)
    if not all(value["passed"] for value in scored.values()):
        raise AssertionError("a scored behavioral relation failed")
    if not all(value["passed"] for value in fatal.values()):
        raise AssertionError("a fatal unscored oracle failed")
    return result


def main() -> None:
    arguments = _parse_args()
    _check_frozen_registry()
    if arguments.check_only:
        print(
            "check-only validated two memory cells, two clean lifecycle cells, "
            "two suppression cells and five source records; no artifacts produced"
        )
        return
    result = _run(arguments)
    print(
        "routing lifetime study passed "
        f"{result['evidence_class_counts']['scored_behavioral_instances']} scored "
        "instances with all fatal oracles"
    )


if __name__ == "__main__":
    main()
