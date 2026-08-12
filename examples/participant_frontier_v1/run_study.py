"""Run the frozen participant-frontier conservation study for CORE-35.

The study executes the Granite three-request replay twice per request count,
once on the lowerer's unchanged participant-local graph and once on the
previously accepted whole-operation barrier projection, and checks that the
coarse runtime report conserves participant-keyed critical segments instead of
one scalar predecessor per operation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

EXPECTATIONS_COMMIT = "242e4d88aa949eb62691f5e43b78a971311d9df4"

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
}

TARGET_OPERATION_ID = "step-0:layer-1:rank-1:compute"
TARGET_PARTICIPANT_RANK = 1
DECISION_STEP_INDEX = 0

MOE_LAYERS = 24
NUM_EXPERTS = 32
TOP_K = 8
EP_RANKS = tuple(range(8))
LAYER_COMPUTE_PS = 4_139_000
NVLINK_BITS_PER_SECOND = 900_000_000_000

SHAPES = ("participant-local", "barrier")

# Frozen expectations, examples/participant_frontier_v1/expectations.md at
# commit 242e4d8. Nothing below is recomputed from a run.
CELL_EXPECTATIONS = {
    (1, "participant-local"): {
        "executions": 25,
        "all_events": 110_416,
        "completions": 5_760,
        "result_bytes": 30_399_320,
        "result_sha256": (
            "00cff9f56b550a166548e9c44e98d4dffe26c8102eb17b7a1fcdeda6e863fb94"
        ),
        "completion_bytes": 288_300,
        "completion_sha256": (
            "73b7415729185e9b4481561da8e6caff23487b68bb6a962f401bfe7052beb8b4"
        ),
        "predecessor_boundary_ps": 6_341_742,
        "target_completed_at_ps": 10_480_742,
        "step_completed_at_ps": 154_568_365,
    },
    (1, "barrier"): {
        "executions": 25,
        "all_events": 110_416,
        "completions": 5_760,
        "result_bytes": 30_399_320,
        "result_sha256": (
            "38cb6503f5475f2acd8071771c09119ddfa7ae4dc7af875169612b9375347420"
        ),
        "completion_bytes": 288_300,
        "completion_sha256": (
            "6f70c590af674dea6f9f24860e16fda3cf1f9a20eda4869ec0a34b027cb637af"
        ),
        "predecessor_boundary_ps": 6_651_217,
        "target_completed_at_ps": 10_790_217,
        "step_completed_at_ps": 154_568_365,
    },
    (3, "participant-local"): {
        "executions": 33,
        "all_events": 160_416,
        "completions": 7_680,
        "result_bytes": 44_179_494,
        "result_sha256": (
            "f58841e7747ae08fb41355e48a1aba30fdf9b12bb3b2e68642241550cf36115f"
        ),
        "completion_bytes": 386_327,
        "completion_sha256": (
            "1fcaf34da306efac867c27d45d0e2d0ae8975c7c692a34cafbf650b68adec6c7"
        ),
        "predecessor_boundary_ps": 9_673_156,
        "target_completed_at_ps": 13_812_156,
        "step_completed_at_ps": 234_886_380,
    },
    (3, "barrier"): {
        "executions": 33,
        "all_events": 160_416,
        "completions": 7_680,
        "result_bytes": 44_179_502,
        "result_sha256": (
            "66668afa531ab34054d2e4a3b3dc476d539600cc945ceb6188b87cbeb233f1a5"
        ),
        "completion_bytes": 386_327,
        "completion_sha256": (
            "dd2356365d657d9c0c1e4056b1677bf184d14060bac0827ac8c34cbbbb18125e"
        ),
        "predecessor_boundary_ps": 10_346_720,
        "target_completed_at_ps": 14_485_720,
        "step_completed_at_ps": 234_886_380,
    },
}

LIFETIME_EXITS = {1: (1, 0, 0), 3: (3, 0, 0)}

TIMESTAMP_AGREEMENT = {1: (4_455, 1_305), 3: (5_127, 2_553)}

TARGET_SHAPE_GAP_PS = {1: 309_475, 3: 673_564}

STEP_SCALING = {
    "one_request_ps": 154_568_365,
    "three_request_ps": 234_886_380,
    "increase_ps": 80_318_015,
    "ratio": 1.519627771,
}

PHYSICAL_EXPECTATIONS = {
    1: {
        "directed_pairs": 336,
        "total_bytes": 10_403_840,
        "peak_rank_egress_bytes": 5_201_920,
        "peak_egress_floor_ps": 46_239_289,
        "compute_floor_ps": 99_336_000,
        "serialized_work_ceiling_ps": 191_814_578,
        "observed_step_ps": 154_568_365,
    },
    3: {
        "directed_pairs": 336,
        "total_bytes": 25_563_136,
        "peak_rank_egress_bytes": 12_781_568,
        # The freeze records 113,613,049 ps here. That literal is not the
        # serialization of its own frozen peak-egress byte count, which is
        # 113,613,938 ps at 900 Gbit/s. Both values are recorded below and the
        # bound is enforced against the recomputed one; see RESULTS.md.
        "peak_egress_floor_ps": 113_613_049,
        "compute_floor_ps": 99_336_000,
        "serialized_work_ceiling_ps": 326_563_876,
        "observed_step_ps": 234_886_380,
    },
}

BYTE_SCALING_RATIO = 2.457086614

NEGATIVE_CONTROL = {
    "operation_id": TARGET_OPERATION_ID,
    "segment_rank": TARGET_PARTICIPANT_RANK,
    "declared_predecessor_rank": 0,
    "true_predecessor_rank": 1,
    "diagnostic": "critical segment predecessor timestamp disagrees",
}


def _check_frozen_registry() -> None:
    if set(CELL_EXPECTATIONS) != {
        (count, shape) for count in (1, 3) for shape in SHAPES
    }:
        raise AssertionError("the frozen sweep must be two request counts by two shapes")
    for (count, shape), cell in CELL_EXPECTATIONS.items():
        if len(cell["result_sha256"]) != 64 or len(cell["completion_sha256"]) != 64:
            raise AssertionError("frozen digests must contain 64 hexadecimal digits")
        if cell["target_completed_at_ps"] - cell["predecessor_boundary_ps"] != (
            LAYER_COMPUTE_PS
        ):
            raise AssertionError(
                f"frozen cell {(count, shape)} target segment is not one layer of compute"
            )
    for count, gap in TARGET_SHAPE_GAP_PS.items():
        observed_gap = (
            CELL_EXPECTATIONS[(count, "barrier")]["target_completed_at_ps"]
            - CELL_EXPECTATIONS[(count, "participant-local")]["target_completed_at_ps"]
        )
        if observed_gap != gap or gap <= 0:
            raise AssertionError("frozen target gap disagrees with the frozen cells")
        boundary_gap = (
            CELL_EXPECTATIONS[(count, "barrier")]["predecessor_boundary_ps"]
            - CELL_EXPECTATIONS[(count, "participant-local")]["predecessor_boundary_ps"]
        )
        if boundary_gap != gap:
            raise AssertionError("frozen boundary gap disagrees with the frozen cells")
    if (
        STEP_SCALING["three_request_ps"] - STEP_SCALING["one_request_ps"]
        != STEP_SCALING["increase_ps"]
    ):
        raise AssertionError("frozen step-scaling increase is not the frozen difference")
    if (
        round(
            STEP_SCALING["three_request_ps"] / STEP_SCALING["one_request_ps"],
            9,
        )
        != STEP_SCALING["ratio"]
    ):
        raise AssertionError("frozen step-scaling ratio disagrees with its endpoints")
    for count, agreement in TIMESTAMP_AGREEMENT.items():
        if sum(agreement) != CELL_EXPECTATIONS[(count, "barrier")]["completions"]:
            raise AssertionError("frozen timestamp agreement does not cover every row")
    for count, physical in PHYSICAL_EXPECTATIONS.items():
        floor_ps = max(physical["compute_floor_ps"], physical["peak_egress_floor_ps"])
        if not floor_ps < physical["observed_step_ps"] < (
            physical["serialized_work_ceiling_ps"]
        ):
            raise AssertionError(
                f"frozen {count}-request step does not lie inside its frozen bounds"
            )
        if physical["compute_floor_ps"] != MOE_LAYERS * LAYER_COMPUTE_PS:
            raise AssertionError("frozen compute floor is not the serial layer sum")
    byte_ratio = round(
        PHYSICAL_EXPECTATIONS[3]["total_bytes"]
        / PHYSICAL_EXPECTATIONS[1]["total_bytes"],
        9,
    )
    if byte_ratio != BYTE_SCALING_RATIO:
        raise AssertionError("frozen byte scaling disagrees with the frozen totals")
    if NEGATIVE_CONTROL["declared_predecessor_rank"] == (
        NEGATIVE_CONTROL["true_predecessor_rank"]
    ):
        raise AssertionError("the negative control must name a different predecessor rank")
    if any(len(digest) != 64 for _, digest, _ in SOURCE_ARTIFACTS.values()):
        raise AssertionError("source digests must contain 64 hexadecimal digits")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _compact(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _git_object(revision: str) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", revision],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _serialization_floor_ps(payload_bytes: int) -> int:
    return -(-payload_bytes * 8 * 1_000_000_000_000 // NVLINK_BITS_PER_SECOND)


def _validate_inputs(arguments: argparse.Namespace) -> dict[str, Any]:
    observations: dict[str, Any] = {}
    for name, (relative_path, digest, size) in SOURCE_ARTIFACTS.items():
        path = arguments.source_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"missing source artifact: {path}")
        data = path.read_bytes()
        observation = {
            "relative_path": relative_path,
            "bytes": len(data),
            "sha256": _sha256(data),
            "expected_sha256": digest,
        }
        observation["matches_sha256"] = observation["sha256"] == digest
        if size is not None:
            observation["expected_bytes"] = size
            observation["matches_bytes"] = observation["bytes"] == size
        if not observation["matches_sha256"] or not observation.get(
            "matches_bytes", True
        ):
            raise AssertionError(f"source artifact changed: {relative_path}")
        observations[name] = observation
    return observations


def _fixed_provider(duration_ps: int) -> Any:
    from simllm.compute import ComputeProvider, DurationEstimate

    class FixedProvider(ComputeProvider):
        def estimate(self, kernel: Any, gpu: Any) -> DurationEstimate:
            return DurationEstimate(duration_ps=duration_ps, bound="measured")

    return FixedProvider()


def _granite_dims() -> Any:
    from simllm.compute import ModelDims

    return ModelDims(
        num_layers=MOE_LAYERS,
        hidden_size=1024,
        intermediate_size=512,
        num_heads=16,
        num_kv_heads=8,
        head_size=64,
        vocab_size=49_152,
        dtype_bytes=2,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        moe_intermediate_size=512,
        local_num_experts=4,
    )


def _placement() -> Any:
    from simllm.traffic import ExpertPlacementSnapshot

    return ExpertPlacementSnapshot(
        placement_epoch=0,
        expert_owners=tuple(
            (layer, expert, expert % len(EP_RANKS))
            for layer in range(MOE_LAYERS)
            for expert in range(NUM_EXPERTS)
        ),
    )


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


def _barrier_projection(graph: Any) -> Any:
    """Return the accepted whole-operation barrier projection of ``graph``.

    This is the compatibility projection the routing-lifetime study registered
    when the scalar report rejected a participant-local frontier. It moves every
    explicit participant-local predecessor into ``depends_on`` and preserves
    operation order, work, request correlation, layer identity and completion
    IDs exactly.
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


def _open_cell(
    arguments: argparse.Namespace,
    directory: Path,
    external_run: Any,
    steps: tuple[Any, ...],
    request_count: int,
) -> tuple[Any, Any, Any, Any]:
    from simllm.core import RequestBookkeeper
    from simllm.preplay import (
        RequestArrival,
        create_request_lifetimes,
        join_preplay_arrivals,
        open_routing_arena,
    )
    from simllm.traffic import RoutedMoeSupply

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
    arena = open_routing_arena(index_path, expected_run=joined)
    lifetimes = create_request_lifetimes(joined, arena)
    supply = RoutedMoeSupply(
        engine_rank=0,
        placements=(_placement(),),
        step_placement_epochs=tuple((record.step_index, 0) for record in steps),
        routing_arena=arena,
        lifetimes=lifetimes,
    )
    return joined, arena, lifetimes, supply


def _lowerer(supply: Any) -> Any:
    from simllm.backends import SerialStepLowerer, SerialStepLowererConfig

    return SerialStepLowerer(
        SerialStepLowererConfig(
            dims=_granite_dims(),
            tp_ranks=(0,),
            ep_ranks=EP_RANKS,
            provider=_fixed_provider(MOE_LAYERS * LAYER_COMPUTE_PS),
            routed_moe_supply=supply,
        )
    )


def _graph_physics(graph: Any) -> dict[str, Any]:
    from simllm.core import CollectiveWork

    pair_entries = 0
    total_bytes = 0
    egress: dict[int, int] = {}
    for operation in graph.operations:
        work = operation.work
        if not isinstance(work, CollectiveWork) or not work.pair_payload_bytes:
            continue
        for source_rank, _destination_rank, payload_bytes in work.pair_payload_bytes:
            if payload_bytes <= 0:
                continue
            pair_entries += 1
            total_bytes += payload_bytes
            egress[source_rank] = egress.get(source_rank, 0) + payload_bytes
    return {
        "directed_pairs": pair_entries,
        "total_bytes": total_bytes,
        "peak_rank_egress_bytes": max(egress.values()) if egress else 0,
    }


def _segment_audit(graph: Any, report: Any, released_at_ps: int) -> dict[str, int]:
    """Recheck the registered conservation identities from the raw report."""

    from simllm.core.execution_io import operation_participant_ranks

    graph_by_id = {
        operation.operation_id: operation for operation in graph.operations
    }
    segments: dict[tuple[str, int], Any] = {}
    for record in report.operations:
        expected_ranks = operation_participant_ranks(graph_by_id[record.operation_id])
        ranks = tuple(segment.participant_rank for segment in record.critical_segments)
        if ranks != expected_ranks:
            raise AssertionError(
                f"operation {record.operation_id!r} segments disagree with participants"
            )
        completions = dict(record.participant_completed_at_ps)
        for segment in record.critical_segments:
            if segment.completed_at_ps != completions[segment.participant_rank]:
                raise AssertionError("segment completion disagrees with its participant")
            latency_ps = segment.completed_at_ps - segment.started_at_ps
            if segment.breakdown.operation_latency_ps != latency_ps:
                raise AssertionError("segment breakdown does not conserve its span")
            if segment.attribution.total_ps != latency_ps:
                raise AssertionError("segment attribution does not conserve its span")
            key = (segment.operation_id, segment.participant_rank)
            if key in segments:
                raise AssertionError("duplicate participant segment in one report")
            segments[key] = segment

    local_edges = 0
    for segment in segments.values():
        if segment.predecessor_operation_id is None:
            if segment.started_at_ps != released_at_ps:
                raise AssertionError("root segment does not start at graph release")
            continue
        predecessor = segments.get(
            (segment.predecessor_operation_id, segment.predecessor_participant_rank)
        )
        if predecessor is None:
            raise AssertionError("segment names an unknown predecessor segment")
        if segment.started_at_ps != predecessor.completed_at_ps:
            raise AssertionError("segment start disagrees with its predecessor")
        if segment.predecessor_participant_rank == segment.participant_rank:
            local_edges += 1

    chain = report.realized_critical_path_segments
    if len(set(chain)) != len(chain):
        raise AssertionError("realized critical segment chain is not acyclic")
    if tuple(key[0] for key in chain) != report.realized_critical_path_operation_ids:
        raise AssertionError("critical path projections disagree")
    if chain:
        first = segments[chain[0]]
        if first.started_at_ps != released_at_ps:
            raise AssertionError("critical segment chain does not start at release")
        chain_latency_ps = sum(
            segments[key].breakdown.operation_latency_ps for key in chain
        )
        endpoint = segments[chain[-1]]
        if chain_latency_ps != endpoint.completed_at_ps - released_at_ps:
            raise AssertionError("critical segment chain does not conserve")
    return {
        "operations": len(report.operations),
        "segments": len(segments),
        "same_rank_edges": local_edges,
        "chain_segments": len(chain),
    }


def _run_cell(
    arguments: argparse.Namespace,
    external_run: Any,
    steps: tuple[Any, ...],
    request_count: int,
    shape: str,
) -> dict[str, Any]:
    from simllm.core import (
        CoarseDeviceProfile,
        CoarseDeviceRuntime,
        CompletionReducer,
        EventPhase,
        StepRecord,
        VirtualClock,
        execution_result_to_json,
    )

    directory = arguments.out / f"{shape}-requests-{request_count}"
    _joined, arena, lifetimes, supply = _open_cell(
        arguments,
        directory,
        external_run,
        steps,
        request_count,
    )
    lowerer = _lowerer(supply)
    clock = VirtualClock(0)
    reducer = CompletionReducer(clock, lifetimes=lifetimes)
    runtime = CoarseDeviceRuntime(CoarseDeviceProfile())
    wanted = {f"r{index}" for index in range(request_count)}
    selected = steps[:25] if request_count == 1 else steps

    payloads: list[dict[str, Any]] = []
    completion_rows: list[list[Any]] = []
    step_rows: list[dict[str, Any]] = []
    identity_timestamps: dict[tuple[int, str, str | None], int] = {}
    identity_order: list[tuple[int, str, str | None]] = []
    all_event_count = 0
    segment_totals = {
        "operations": 0,
        "segments": 0,
        "same_rank_edges": 0,
        "chain_segments": 0,
    }
    decision: dict[str, Any] | None = None
    physics: dict[str, Any] | None = None
    negative_control_seed: dict[str, Any] | None = None

    def drain_record() -> Any:
        return StepRecord(
            step_index=32,
            virtual_time_ps=clock.now_ps,
            finished_request_ids=["r2"],
            num_sampled=0,
            sampled_request_ids=[],
        )

    pending: list[Any] = [
        lambda source=source: _filtered_record(source, wanted, clock.now_ps)
        for source in selected
    ]
    if request_count != 1:
        pending.append(drain_record)
    for make_record in pending:
        record = make_record()
        graph = lowerer.lower(record)
        if shape == "barrier":
            graph = _barrier_projection(graph)
        execution = runtime.execute(graph)
        report = runtime.last_report
        assert report is not None
        step_result = reducer.reduce(record, graph, execution, report)

        audit = _segment_audit(graph, report, graph.released_at_ps)
        for name, value in audit.items():
            segment_totals[name] += value
        payloads.append(execution_result_to_json(execution))
        all_event_count += len(execution.events)
        for event in execution.events:
            if event.phase is not EventPhase.COMPLETED:
                continue
            completion_rows.append(
                [event.operation_id, event.subject_object_id, event.timestamp_ps]
            )
            identity = (record.step_index, event.operation_id, event.subject_object_id)
            if identity in identity_timestamps:
                raise AssertionError("duplicate completion identity in one cell")
            identity_timestamps[identity] = event.timestamp_ps
            identity_order.append(identity)
        step_rows.append(
            {
                "step_index": record.step_index,
                "execution_completed_at_ps": execution.completed_at_ps,
                "step_result_completed_at_ps": step_result.completed_at_ps,
                "quiesced_at_ps": execution.quiesced_at_ps,
            }
        )
        if record.step_index == DECISION_STEP_INDEX:
            physics = _graph_physics(graph)
            by_id = {
                operation.operation_id: operation for operation in report.operations
            }
            target = by_id[TARGET_OPERATION_ID]
            segment = next(
                item
                for item in target.critical_segments
                if item.participant_rank == TARGET_PARTICIPANT_RANK
            )
            decision = {
                "operation_id": TARGET_OPERATION_ID,
                "participant_rank": segment.participant_rank,
                "predecessor_operation_id": segment.predecessor_operation_id,
                "predecessor_participant_rank": segment.predecessor_participant_rank,
                "predecessor_boundary_ps": segment.started_at_ps,
                "target_completed_at_ps": target.completed_at_ps,
                "segment_latency_ps": segment.breakdown.operation_latency_ps,
                "step_completed_at_ps": execution.completed_at_ps,
            }
            if shape == "participant-local" and request_count == 3:
                predecessor_record = by_id[segment.predecessor_operation_id]
                negative_control_seed = {
                    "record": record,
                    "graph": graph,
                    "execution": execution,
                    "report": report,
                    "target": target,
                    "segment": segment,
                    "predecessor_completions": dict(
                        predecessor_record.participant_completed_at_ps
                    ),
                }

    raw_exit = (
        lifetimes.closed_request_count,
        lifetimes.live_request_count,
        lifetimes.live_view_count,
    )
    assert decision is not None
    assert physics is not None
    cell = {
        "request_count": request_count,
        "shape": shape,
        "executions": len(payloads),
        "all_events": all_event_count,
        "completions": len(completion_rows),
        "raw_exit": {
            "closed": raw_exit[0],
            "live": raw_exit[1],
            "views": raw_exit[2],
        },
        "decision_row": decision,
        "physics": physics,
        "segments": segment_totals,
        "step_rows": step_rows,
        "target_operation_present": True,
    }
    private = {
        "result_bytes": (_compact(payloads) + "\n").encode(),
        "completion_bytes": (_compact(completion_rows) + "\n").encode(),
        "identity_timestamps": identity_timestamps,
        "identity_order": tuple(identity_order),
        "negative_control_seed": negative_control_seed,
        "lifetimes": lifetimes,
        "arena": arena,
    }
    return {"cell": cell, "private": private}


def _negative_control(
    arguments: argparse.Namespace,
    external_run: Any,
    steps: tuple[Any, ...],
    seed: dict[str, Any],
) -> dict[str, Any]:
    from simllm.core import CompletionReducer, VirtualClock

    directory = arguments.out / "negative-control"
    _joined, arena, lifetimes, _supply = _open_cell(
        arguments,
        directory,
        external_run,
        steps,
        3,
    )
    clock = VirtualClock(0)
    reducer = CompletionReducer(clock, lifetimes=lifetimes)

    def lifetime_state() -> list[dict[str, Any]]:
        return [
            {
                "request_id": lifetime.request_id,
                "state": lifetime.state.value,
                "cursor": lifetime.consumption_cursor,
                "dispatch_mask": lifetime.dispatch_end_mask,
                "combine_mask": lifetime.combine_end_mask,
                "view_released": lifetime.view_released,
            }
            for lifetime in lifetimes.requests
        ]

    before = lifetime_state()
    segment = seed["segment"]
    target = seed["target"]
    report = seed["report"]
    predecessor_completions = seed["predecessor_completions"]
    declared_rank = NEGATIVE_CONTROL["declared_predecessor_rank"]
    true_rank = segment.predecessor_participant_rank
    distinct_predecessor = (
        predecessor_completions[declared_rank] != predecessor_completions[true_rank]
    )
    mutated_target = replace(
        target,
        critical_segments=tuple(
            replace(item, predecessor_participant_rank=declared_rank)
            if item.participant_rank == segment.participant_rank
            else item
            for item in target.critical_segments
        ),
    )
    mutated_report = replace(
        report,
        operations=tuple(
            mutated_target
            if operation.operation_id == target.operation_id
            else operation
            for operation in report.operations
        ),
    )
    diagnostic: str | None = None
    rejected = False
    try:
        reducer.reduce(
            seed["record"],
            seed["graph"],
            seed["execution"],
            mutated_report,
        )
    except ValueError as error:
        rejected = True
        diagnostic = str(error)
    after = lifetime_state()
    unchanged = (
        clock.now_ps == 0
        and reducer.latest_request_metrics == ()
        and before == after
    )
    accepted_unmutated = False
    accepted_boundary_ps: int | None = None
    if rejected:
        accepted = reducer.reduce(
            seed["record"],
            seed["graph"],
            seed["execution"],
            report,
        )
        accepted_unmutated = True
        accepted_boundary_ps = accepted.completed_at_ps
    # This cell replays only the decision step, so its request views stay live
    # by construction and the arena refuses to close. That refusal is the
    # ownership guard working, not a leak; process exit releases the read-only
    # mapping.
    close_rejected = False
    try:
        arena.close()
    except BufferError:
        close_rejected = True
    passed = (
        rejected
        and diagnostic == NEGATIVE_CONTROL["diagnostic"]
        and distinct_predecessor
        and true_rank == NEGATIVE_CONTROL["true_predecessor_rank"]
    )
    return {
        "operation_id": target.operation_id,
        "segment_rank": segment.participant_rank,
        "true_predecessor_rank": true_rank,
        "declared_predecessor_rank": declared_rank,
        "true_predecessor_completed_at_ps": predecessor_completions[true_rank],
        "declared_predecessor_completed_at_ps": predecessor_completions[declared_rank],
        "distinct_predecessor_completion": distinct_predecessor,
        "rejected": rejected,
        "diagnostic": diagnostic,
        "expected_diagnostic": NEGATIVE_CONTROL["diagnostic"],
        "state_unchanged_before_rejection": unchanged,
        "unmutated_report_accepted": accepted_unmutated,
        "unmutated_boundary_ps": accepted_boundary_ps,
        "partial_replay_arena_close_rejected": close_rejected,
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
    from simllm.preplay import read_preplay_replay_run

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

    runs: dict[tuple[int, str], dict[str, Any]] = {}
    for request_count in (1, 3):
        for shape in SHAPES:
            runs[(request_count, shape)] = _run_cell(
                arguments,
                external_run,
                steps,
                request_count,
                shape,
            )

    # PF-B1 through PF-B3 read raw runtime and reducer observations. No exact
    # timestamp, byte or hash oracle has been consulted at this point.
    admission_rows = []
    for request_count in (1, 3):
        cell = runs[(request_count, "participant-local")]["cell"]
        expected_exit = LIFETIME_EXITS[request_count]
        expected_cell = CELL_EXPECTATIONS[(request_count, "participant-local")]
        row = {
            "request_count": request_count,
            "raw_exit": cell["raw_exit"],
            "expected_exit": {
                "closed": expected_exit[0],
                "live": expected_exit[1],
                "views": expected_exit[2],
            },
            "target_operation_id": cell["decision_row"]["operation_id"],
            "target_participant_rank": cell["decision_row"]["participant_rank"],
            "target_predecessor_operation_id": (
                cell["decision_row"]["predecessor_operation_id"]
            ),
            "target_predecessor_participant_rank": (
                cell["decision_row"]["predecessor_participant_rank"]
            ),
            "completions": cell["completions"],
            "expected_completions": expected_cell["completions"],
        }
        row["passed"] = (
            row["raw_exit"] == row["expected_exit"]
            and row["target_participant_rank"] == TARGET_PARTICIPANT_RANK
            and row["target_predecessor_participant_rank"] == TARGET_PARTICIPANT_RANK
            and row["completions"] == expected_cell["completions"]
        )
        admission_rows.append(row)

    gap_rows = []
    for request_count in (1, 3):
        local = runs[(request_count, "participant-local")]["cell"]
        barrier = runs[(request_count, "barrier")]["cell"]
        gap_ps = (
            barrier["decision_row"]["target_completed_at_ps"]
            - local["decision_row"]["target_completed_at_ps"]
        )
        local_steps = {
            row["step_index"]: row["execution_completed_at_ps"]
            for row in local["step_rows"]
        }
        barrier_steps = {
            row["step_index"]: row["execution_completed_at_ps"]
            for row in barrier["step_rows"]
        }
        local_result_steps = {
            row["step_index"]: row["step_result_completed_at_ps"]
            for row in local["step_rows"]
        }
        barrier_result_steps = {
            row["step_index"]: row["step_result_completed_at_ps"]
            for row in barrier["step_rows"]
        }
        row = {
            "request_count": request_count,
            "target_gap_ps": gap_ps,
            "expected_target_gap_ps": TARGET_SHAPE_GAP_PS[request_count],
            "step_boundaries_equal": local_steps == barrier_steps,
            "step_result_boundaries_equal": (
                local_result_steps == barrier_result_steps
            ),
            "local_target_completed_at_ps": (
                local["decision_row"]["target_completed_at_ps"]
            ),
            "barrier_target_completed_at_ps": (
                barrier["decision_row"]["target_completed_at_ps"]
            ),
        }
        row["passed"] = (
            gap_ps == TARGET_SHAPE_GAP_PS[request_count]
            and gap_ps > 0
            and row["step_boundaries_equal"]
            and row["step_result_boundaries_equal"]
        )
        gap_rows.append(row)

    one_step_ps = runs[(1, "participant-local")]["cell"]["decision_row"][
        "step_completed_at_ps"
    ]
    three_step_ps = runs[(3, "participant-local")]["cell"]["decision_row"][
        "step_completed_at_ps"
    ]
    scaling = {
        "one_request_ps": one_step_ps,
        "three_request_ps": three_step_ps,
        "increase_ps": three_step_ps - one_step_ps,
        "ratio": round(three_step_ps / one_step_ps, 9),
        "expected": dict(STEP_SCALING),
    }
    scaling["passed"] = (
        one_step_ps == STEP_SCALING["one_request_ps"]
        and three_step_ps == STEP_SCALING["three_request_ps"]
        and scaling["increase_ps"] == STEP_SCALING["increase_ps"]
        and scaling["ratio"] == STEP_SCALING["ratio"]
    )

    control = _negative_control(
        arguments,
        external_run,
        steps,
        runs[(3, "participant-local")]["private"]["negative_control_seed"],
    )

    # Fatal-unscored oracles start here.
    physical_rows = []
    for request_count in (1, 3):
        expected = PHYSICAL_EXPECTATIONS[request_count]
        physics = runs[(request_count, "participant-local")]["cell"]["physics"]
        barrier_physics = runs[(request_count, "barrier")]["cell"]["physics"]
        observed_step_ps = runs[(request_count, "participant-local")]["cell"][
            "decision_row"
        ]["step_completed_at_ps"]
        recomputed_egress_floor_ps = _serialization_floor_ps(
            physics["peak_rank_egress_bytes"]
        )
        recomputed_ceiling_ps = _serialization_floor_ps(
            physics["total_bytes"]
        ) + MOE_LAYERS * LAYER_COMPUTE_PS
        floor_ps = max(recomputed_egress_floor_ps, MOE_LAYERS * LAYER_COMPUTE_PS)
        row = {
            "request_count": request_count,
            "directed_pairs": physics["directed_pairs"],
            "total_bytes": physics["total_bytes"],
            "peak_rank_egress_bytes": physics["peak_rank_egress_bytes"],
            "shapes_agree_on_bytes": physics == barrier_physics,
            "recomputed_peak_egress_floor_ps": recomputed_egress_floor_ps,
            "frozen_peak_egress_floor_ps": expected["peak_egress_floor_ps"],
            "frozen_floor_matches_frozen_bytes": (
                recomputed_egress_floor_ps == expected["peak_egress_floor_ps"]
            ),
            "compute_floor_ps": MOE_LAYERS * LAYER_COMPUTE_PS,
            "recomputed_ceiling_ps": recomputed_ceiling_ps,
            "frozen_ceiling_ps": expected["serialized_work_ceiling_ps"],
            "observed_step_ps": observed_step_ps,
            "inside_bounds": floor_ps < observed_step_ps < recomputed_ceiling_ps,
        }
        row["passed"] = (
            row["directed_pairs"] == expected["directed_pairs"]
            and row["total_bytes"] == expected["total_bytes"]
            and row["peak_rank_egress_bytes"] == expected["peak_rank_egress_bytes"]
            and row["shapes_agree_on_bytes"]
            and row["recomputed_ceiling_ps"] == expected["serialized_work_ceiling_ps"]
            and row["inside_bounds"]
        )
        physical_rows.append(row)
    byte_ratio = round(
        physical_rows[1]["total_bytes"] / physical_rows[0]["total_bytes"], 9
    )
    egress_ratio = round(
        physical_rows[1]["peak_rank_egress_bytes"]
        / physical_rows[0]["peak_rank_egress_bytes"],
        9,
    )

    digest_rows = []
    for (request_count, shape), run in runs.items():
        expected = CELL_EXPECTATIONS[(request_count, shape)]
        private = run["private"]
        cell = run["cell"]
        row = {
            "request_count": request_count,
            "shape": shape,
            "executions": cell["executions"],
            "all_events": cell["all_events"],
            "completions": cell["completions"],
            "result_bytes": len(private["result_bytes"]),
            "result_sha256": _sha256(private["result_bytes"]),
            "completion_bytes": len(private["completion_bytes"]),
            "completion_sha256": _sha256(private["completion_bytes"]),
            "expected": {
                key: expected[key]
                for key in (
                    "executions",
                    "all_events",
                    "completions",
                    "result_bytes",
                    "result_sha256",
                    "completion_bytes",
                    "completion_sha256",
                )
            },
        }
        row["passed"] = all(
            row[key] == expected[key]
            for key in (
                "executions",
                "all_events",
                "completions",
                "result_bytes",
                "result_sha256",
                "completion_bytes",
                "completion_sha256",
            )
        )
        digest_rows.append(row)
    digest_rows.sort(key=lambda row: (row["request_count"], row["shape"]))

    identity_rows = []
    for request_count in (1, 3):
        local = runs[(request_count, "participant-local")]["private"]
        barrier = runs[(request_count, "barrier")]["private"]
        local_ids = local["identity_order"]
        barrier_ids = barrier["identity_order"]
        equal = sum(
            1
            for identity in local_ids
            if barrier["identity_timestamps"].get(identity)
            == local["identity_timestamps"][identity]
        )
        differ = len(local_ids) - equal
        expected_equal, expected_differ = TIMESTAMP_AGREEMENT[request_count]

        def identity_key(identity: tuple[int, str, str | None]) -> tuple[int, str, str]:
            return (identity[0], identity[1], identity[2] or "")

        row = {
            "request_count": request_count,
            "identity_multiset_equal": (
                sorted(local_ids, key=identity_key)
                == sorted(barrier_ids, key=identity_key)
            ),
            # Identity carries its step index, so multiset equality above is
            # exactly the registered per-step claim. Emission order is a
            # different question and is recorded, not required: sequencing
            # follows realized timestamps, and the barrier arm moves them.
            "identity_emission_order_equal": local_ids == barrier_ids,
            "equal_timestamps": equal,
            "differing_timestamps": differ,
            "expected_equal_timestamps": expected_equal,
            "expected_differing_timestamps": expected_differ,
            "barrier_never_earlier": all(
                barrier["identity_timestamps"][identity]
                >= local["identity_timestamps"][identity]
                for identity in local_ids
            ),
        }
        row["passed"] = (
            row["identity_multiset_equal"]
            and equal == expected_equal
            and differ == expected_differ
            and row["barrier_never_earlier"]
        )
        identity_rows.append(row)

    segment_rows = []
    for (request_count, shape), run in sorted(
        runs.items(), key=lambda item: (item[0][0], item[0][1])
    ):
        totals = run["cell"]["segments"]
        segment_rows.append(
            {
                "request_count": request_count,
                "shape": shape,
                "operations": totals["operations"],
                "segments": totals["segments"],
                "same_rank_predecessor_edges": totals["same_rank_edges"],
                "chain_segments": totals["chain_segments"],
                "segments_cover_every_participant": (
                    totals["segments"] >= totals["operations"]
                ),
            }
        )
    local_same_rank = sum(
        row["same_rank_predecessor_edges"]
        for row in segment_rows
        if row["shape"] == "participant-local"
    )
    barrier_same_rank = sum(
        row["same_rank_predecessor_edges"]
        for row in segment_rows
        if row["shape"] == "barrier"
    )

    for run in runs.values():
        run["private"]["arena"].close()

    scored = {
        "PF-B1": {
            "classification": "scored-behavioral",
            "genuine_risk": "2/2",
            "rows": admission_rows,
            "passed": all(row["passed"] for row in admission_rows),
        },
        "PF-B2": {
            "classification": "scored-behavioral",
            "genuine_risk": "2/2",
            "rows": gap_rows,
            "passed": all(row["passed"] for row in gap_rows),
        },
        "PF-B3": {
            "classification": "scored-behavioral",
            "genuine_risk": "2/2",
            "rows": [scaling],
            "passed": scaling["passed"],
        },
        "PF-B4": {
            "classification": "scored-behavioral",
            "genuine_risk": "1/1",
            "rows": [control],
            "passed": control["passed"],
        },
    }
    fatal = {
        "input_identity": {
            "classification": "fatal-unscored-configuration",
            "passed": True,
        },
        "exact_completion_preservation": {
            "classification": "fatal-unscored-exact-oracle",
            "rows": digest_rows,
            "passed": all(row["passed"] for row in digest_rows),
        },
        "completion_identity_agreement": {
            "classification": "fatal-unscored-exact-oracle",
            "rows": identity_rows,
            "passed": all(row["passed"] for row in identity_rows),
        },
        "segment_inventory": {
            "classification": "fatal-unscored-structural",
            "note": (
                "the registered per-rank inventory, segment conservation and "
                "endpoint-chain identities are enforced by exception inside "
                "_segment_audit on every step of every cell; the counts below are "
                "the by-construction census that survived"
            ),
            "rows": segment_rows,
            "local_same_rank_predecessor_edges": local_same_rank,
            "barrier_same_rank_predecessor_edges": barrier_same_rank,
            "passed": all(
                row["segments_cover_every_participant"] for row in segment_rows
            ),
        },
        "physical_bounds": {
            "classification": "fatal-unscored-structural",
            "rows": physical_rows,
            "byte_scaling_ratio": byte_ratio,
            "peak_egress_scaling_ratio": egress_ratio,
            "expected_scaling_ratio": BYTE_SCALING_RATIO,
            "passed": all(row["passed"] for row in physical_rows)
            and byte_ratio == BYTE_SCALING_RATIO
            and egress_ratio == BYTE_SCALING_RATIO,
        },
        "negative_control_atomicity": {
            "classification": "fatal-unscored-structural",
            "state_unchanged_before_rejection": (
                control["state_unchanged_before_rejection"]
            ),
            "unmutated_report_accepted": control["unmutated_report_accepted"],
            "passed": control["state_unchanged_before_rejection"]
            and control["unmutated_report_accepted"],
        },
    }
    result = {
        "schema": "simllm-participant-frontier-study-results-v1",
        "provenance": {
            "expectations_commit": EXPECTATIONS_COMMIT,
            "run_head": _git_object("HEAD"),
            "observed_htsim_gitlink": _git_object("HEAD:third_party/htsim"),
            "gitlink_equality_required": False,
            "python": sys.version,
        },
        "inputs": inputs,
        "run_configuration": {
            "request_counts": [1, 3],
            "graph_shapes": list(SHAPES),
            "one_request_steps": 25,
            "three_request_steps": 33,
            "moe_layers": MOE_LAYERS,
            "experts": NUM_EXPERTS,
            "top_k": TOP_K,
            "ep_ranks": list(EP_RANKS),
            "layer_compute_ps": LAYER_COMPUTE_PS,
            "placement_rule": "expert_id modulo 8",
            "barrier_projection": (
                "explicit participant-local predecessors moved into depends_on; "
                "operation order, work, request correlation, layer identity and "
                "completion IDs unchanged"
            ),
            "completion_row_encoding": (
                "compact key-sorted JSON array of "
                "[operation_id, subject_object_id, timestamp_ps] rows plus LF; "
                "operation IDs already carry their step prefix"
            ),
        },
        "scored_relations": scored,
        "fatal_unscored": fatal,
        "evidence_class_counts": {
            "run_configurations": 4,
            "scored_behavioral_families": 4,
            "scored_behavioral_instances": 7,
            "fatal_exact_cells": len(digest_rows),
            "fatal_structural_families": 4,
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
            "check-only validated four sweep cells, four scored relation families, "
            "seven scored instances and four source records; no artifacts produced"
        )
        return
    result = _run(arguments)
    print(
        "participant frontier study passed "
        f"{result['evidence_class_counts']['scored_behavioral_instances']} scored "
        "instances with all fatal oracles"
    )


if __name__ == "__main__":
    main()
