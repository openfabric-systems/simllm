"""Dry-run registry for the frozen dispatch sequence study."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import tracemalloc
from dataclasses import asdict
from pathlib import Path
from typing import Any

GROUPINGS = ("aggregate", "per-expert-group", "per-token")
PROFILES = ("rnic-nn", "rnic-nn-fluid")
RATES_BPS = (200_000_000_000, 400_000_000_000)
VECTOR_BYTES = 2_048
PACKET_WIRE_BYTES = 4_096
HEADER_BYTES = 64
ROUTES = ((3, 1), (2, 1), (3, 2), (1, 3))
MESSAGE_COUNTS = {
    "aggregate": 18,
    "per-expert-group": 18,
    "per-token": 48,
}
TOTAL_BYTES = 98_304
PACKET_DELTA_BANDS_PS = {
    200_000_000_000: (15_360, 61_440),
    400_000_000_000: (7_680, 30_720),
}
PHYSICAL_BOUNDS_PS = {
    200_000_000_000: (1_474_560, 9_000_000),
    400_000_000_000: (737_280, 4_500_000),
}
FLUID_ABS_DELTA_MAX_PS = 1_000
RATE_SCALING_TOLERANCE_PS = 2_000
SCORED_FAMILIES = 3
SCORED_INSTANCES = 10
GRANITE_ROUTING_SHA256 = (
    "24e986e989e21f1bfe7e758d4470928c82c3bbaec06072a839743b9b17d7cf5f"
)
GRANITE_STEPS_SHA256 = (
    "824cd9557293328bb42b593ac893b6a067302e545b087c9219195ccb8031d755"
)
GRANITE_AGGREGATE_GOAL = (
    334_432,
    "08a0403af66ff8a9d6b18f93afd15ae0bc925cc85555acf8a0593438a3d7bc92",
)
REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_TIMEOUT_S = 600
GRANITE_RENDER_COMPILE_LIMIT_S = 30.0
GRANITE_BACKEND_LIMIT_S = 60.0
GRANITE_MEMORY_LIMIT_BYTES = 1 << 30
GRANITE_GOAL_LIMIT_BYTES = 64 << 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--granite-root", type=Path, required=True)
    parser.add_argument("--htsim-rnic", type=Path, required=True)
    parser.add_argument("--txt2bin", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def check_only(args: argparse.Namespace) -> None:
    if GROUPINGS != ("aggregate", "per-expert-group", "per-token"):
        raise AssertionError("grouping registry drifted")
    if PROFILES != ("rnic-nn", "rnic-nn-fluid"):
        raise AssertionError("profile registry drifted")
    if RATES_BPS != (200_000_000_000, 400_000_000_000):
        raise AssertionError("rate registry drifted")
    if VECTOR_BYTES != 2_048 or PACKET_WIRE_BYTES != 4_096 or HEADER_BYTES != 64:
        raise AssertionError("packet arithmetic inputs drifted")
    if ROUTES != ((3, 1), (2, 1), (3, 2), (1, 3)):
        raise AssertionError("observed route fixture drifted")
    if MESSAGE_COUNTS != {
        "aggregate": 18,
        "per-expert-group": 18,
        "per-token": 48,
    }:
        raise AssertionError("message-count registry drifted")
    if TOTAL_BYTES != 2 * 24 * VECTOR_BYTES:
        raise AssertionError("fixture byte arithmetic drifted")
    if PACKET_DELTA_BANDS_PS[200_000_000_000] != tuple(
        2 * value for value in PACKET_DELTA_BANDS_PS[400_000_000_000]
    ):
        raise AssertionError("packet delta bands do not scale inversely with rate")
    if any(floor <= 0 or ceiling < floor for floor, ceiling in PHYSICAL_BOUNDS_PS.values()):
        raise AssertionError("physical bounds are malformed")
    if FLUID_ABS_DELTA_MAX_PS != 1_000 or RATE_SCALING_TOLERANCE_PS != 2_000:
        raise AssertionError("relation tolerances drifted")
    if (SCORED_FAMILIES, SCORED_INSTANCES) != (3, 10):
        raise AssertionError("evidence accounting drifted")
    if any(len(value) != 64 for value in (GRANITE_ROUTING_SHA256, GRANITE_STEPS_SHA256)):
        raise AssertionError("Granite input hash is malformed")
    if GRANITE_AGGREGATE_GOAL[0] <= 0 or len(GRANITE_AGGREGATE_GOAL[1]) != 64:
        raise AssertionError("Granite aggregate oracle is malformed")
    if any(not str(path) for path in (args.out, args.granite_root, args.htsim_rnic, args.txt2bin)):
        raise AssertionError("registered path argument is empty")
    print(
        "check-only validated frozen dispatch-sequence registries and arithmetic; "
        "produced no artifacts"
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _path_observation(path: Path) -> dict[str, Any]:
    return {
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _git_object(revision: str) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", revision],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _validate_result_inputs(args: argparse.Namespace) -> dict[str, Any]:
    configured_root = os.environ.get("SIMLLM_WAVE6_RUN_ROOT")
    if not configured_root:
        raise RuntimeError(
            "SIMLLM_WAVE6_RUN_ROOT must name the external wave-6 run root"
        )
    run_root = Path(configured_root).resolve()
    try:
        args.out.resolve().relative_to(run_root)
    except ValueError as exc:
        raise ValueError("study output must remain under SIMLLM_WAVE6_RUN_ROOT") from exc
    if args.out.exists():
        raise FileExistsError(f"study output already exists: {args.out}")
    for label, path in (
        ("Granite root", args.granite_root),
        ("htsim_rnic", args.htsim_rnic),
        ("txt2bin", args.txt2bin),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")

    source_paths = {
        "routing": args.granite_root / "replay-400g" / "routed-experts.json",
        "steps": args.granite_root / "replay-400g" / "steps.jsonl",
        "aggregate_goal": (
            args.granite_root / "replay-400g" / "htsim" / "step-000000.goal"
        ),
    }
    for label, path in source_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing Granite {label}: {path}")
    observations = {
        label: {
            **_path_observation(path),
            "relative_path": str(path.relative_to(args.granite_root)),
        }
        for label, path in source_paths.items()
    }
    expected = {
        "routing": GRANITE_ROUTING_SHA256,
        "steps": GRANITE_STEPS_SHA256,
        "aggregate_goal": GRANITE_AGGREGATE_GOAL[1],
    }
    for label, digest in expected.items():
        observations[label]["authored_against_sha256"] = digest
        observations[label]["matches_authored_against"] = (
            observations[label]["sha256"] == digest
        )
        if not observations[label]["matches_authored_against"]:
            raise AssertionError(f"Granite {label} changed before the study")
    if observations["aggregate_goal"]["bytes"] != GRANITE_AGGREGATE_GOAL[0]:
        raise AssertionError("accepted Granite aggregate GOAL size changed")
    return {
        "artifacts": observations,
        "observed_simllm_commit": _git_object("HEAD"),
        "observed_htsim_gitlink": _git_object("HEAD:third_party/htsim"),
        "htsim_binary": _path_observation(args.htsim_rnic),
        "txt2bin_binary": _path_observation(args.txt2bin),
    }


def _synthetic_fixture(output_dir: Path) -> tuple[Any, Any, Any, Any]:
    from simllm.compute import ModelDims
    from simllm.core import RequestPhase, ScheduledRequest, StepRecord
    from simllm.preplay import (
        ForwardPhase,
        FrameworkPreplayTrace,
        FrameworkRequestTrace,
        FrameworkTraceProvenance,
        ObservedLayerDispatch,
        ObservedTokenDispatch,
        PromptFormat,
        SamplingConfig,
        StopReason,
        project_framework_routing,
        write_framework_preplay_trace,
    )
    from simllm.traffic import ExpertPlacementSnapshot, RoutedMoeSupply

    provenance = FrameworkTraceProvenance(
        model_id="study/sequence-fixture",
        model_revision="frozen-v1",
        model_class="SequenceFixtureMoeForCausalLM",
        dtype="float16",
        tokenizer_sha256="a" * 64,
        sampling=SamplingConfig.greedy(),
        capture_host="registered-fixture",
        runner="fixture-runner",
        framework="vllm",
        framework_version="fixture-v1",
        observed_source="b" * 40,
        authored_against_source="c" * 40,
        torch_version="fixture-v1",
        device="cpu",
        torch_num_threads=1,
        engine_seed=1,
        eos_token_id=0,
        top_k=2,
        expert_count=4,
        moe_layer_indices=(0,),
        kv_page_size=1,
        kv_token_capacity=64,
        dispatch_layer_mapping="framework-layer-id",
    )
    request = FrameworkRequestTrace(
        request_id="alpha",
        prompt_sha256="d" * 64,
        prompt_format=PromptFormat.TEXT,
        input_token_ids=(10, 11, 12, 13),
        max_new_tokens=1,
        stop_strings=(),
        output_text="done",
        output_token_ids=(0,),
        output_length=1,
        stop_reason=StopReason.LENGTH_CAP,
        matched_stop_string=None,
        framework_cached_tokens=0,
        framework_preemption_count=0,
        prefill_dispatch=tuple(
            ObservedTokenDispatch(
                phase=ForwardPhase.PREFILL,
                token_index=index,
                token_id=10 + index,
                routing=(
                    ObservedLayerDispatch(layer_index=0, expert_ids=expert_ids),
                ),
            )
            for index, expert_ids in enumerate(ROUTES)
        ),
        decode_dispatch=(),
    )
    trace_path = write_framework_preplay_trace(
        output_dir / "framework-trace.jsonl",
        FrameworkPreplayTrace(
            provenance=provenance,
            requests=(request,),
            kv_events=(),
        ),
    )
    routing = project_framework_routing(trace_path)
    dims = ModelDims(
        num_layers=1,
        hidden_size=1024,
        intermediate_size=2048,
        num_heads=16,
        num_kv_heads=8,
        head_size=64,
        vocab_size=1024,
        dtype_bytes=2,
        num_experts=4,
        top_k=2,
        moe_intermediate_size=1024,
        local_num_experts=1,
    )
    placement = ExpertPlacementSnapshot(
        placement_epoch=0,
        expert_owners=tuple((0, expert, expert) for expert in range(4)),
    )
    supply = RoutedMoeSupply(
        routed_experts=routing,
        placements=(placement,),
        step_placement_epochs=((0, 0),),
    )
    record = StepRecord(
        step_index=0,
        virtual_time_ps=0,
        scheduled=[
            ScheduledRequest(
                "alpha",
                RequestPhase.PREFILL,
                4,
                context_length=4,
            )
        ],
        num_sampled=1,
    )
    return record, dims, supply, routing


def _goal_observation(trace: Any) -> dict[str, Any]:
    payload = trace.render().encode()
    return {
        "goal_bytes": len(payload),
        "goal_sha256": hashlib.sha256(payload).hexdigest(),
        "message_count": len(trace.messages),
        "message_bytes": sum(message.payload_bytes for message in trace.messages),
    }


def _render_synthetic(
    args: argparse.Namespace,
    output_dir: Path,
    record: Any,
    dims: Any,
    supply: Any,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Path],
    dict[str, Any],
]:
    from simllm.goal import to_binary
    from simllm.traffic import (
        MoeMessageGrouping,
        render_sequenced_step_goal,
        render_step_goal,
        step_moe_alltoalls,
        step_moe_message_sequences,
    )

    plans = {
        "aggregate": step_moe_alltoalls(
            record,
            dims,
            (0, 1, 2, 3),
            routed_supply=supply,
        ),
        "per-expert-group": step_moe_message_sequences(
            record,
            dims,
            (0, 1, 2, 3),
            routed_supply=supply,
            grouping=MoeMessageGrouping.PER_EXPERT_GROUP,
        ),
        "per-token": step_moe_message_sequences(
            record,
            dims,
            (0, 1, 2, 3),
            routed_supply=supply,
            grouping=MoeMessageGrouping.PER_TOKEN,
        ),
    }
    traces = {
        "aggregate": render_step_goal(
            record,
            dims,
            (0,),
            0,
            ep_ranks=(0, 1, 2, 3),
            routed_supply=supply,
        ),
        "per-expert-group": render_sequenced_step_goal(
            record,
            dims,
            (0,),
            0,
            ep_ranks=(0, 1, 2, 3),
            routed_supply=supply,
            message_grouping=MoeMessageGrouping.PER_EXPERT_GROUP,
        ),
        "per-token": render_sequenced_step_goal(
            record,
            dims,
            (0,),
            0,
            ep_ranks=(0, 1, 2, 3),
            routed_supply=supply,
            message_grouping=MoeMessageGrouping.PER_TOKEN,
        ),
    }
    observations = {}
    binaries = {}
    for grouping in GROUPINGS:
        goal_path = traces[grouping].write(output_dir / f"{grouping}.goal")
        binary_path = output_dir / f"{grouping}.bin"
        start = time.perf_counter_ns()
        to_binary(goal_path, binary_path, tool=args.txt2bin)
        compile_ns = time.perf_counter_ns() - start
        observations[grouping] = {
            **_goal_observation(traces[grouping]),
            "binary_bytes": binary_path.stat().st_size,
            "compile_ns": compile_ns,
        }
        binaries[grouping] = binary_path
    return plans, traces, binaries, observations


def _run_backend_cell(
    args: argparse.Namespace,
    goal_bin: Path,
    *,
    profile: str,
    rate_bps: int,
    completion_csv: Path,
    include_raw_flows: bool,
) -> dict[str, Any]:
    from simllm.backends import HtsimRnicConfig, run_htsim_rnic

    extra_flags = {"-rnic_nn_propagation_ps": "0"}
    if profile == "rnic-nn":
        extra_flags = {
            "-rnic_max_wire_bytes": str(PACKET_WIRE_BYTES),
            "-rnic_data_header_bytes": str(HEADER_BYTES),
            "-rnic_nn_propagation_ps": "0",
        }
    start = time.perf_counter_ns()
    result = run_htsim_rnic(
        HtsimRnicConfig(
            goal_bin=goal_bin,
            profile=profile,
            linkspeed_bps=rate_bps,
            completion_csv=completion_csv,
            extra_flags=extra_flags,
        ),
        binary=args.htsim_rnic,
        timeout_s=BACKEND_TIMEOUT_S,
    )
    wall_ns = time.perf_counter_ns() - start
    flow_rows = [asdict(flow) for flow in result.flows]
    observation = {
        "profile": profile,
        "rate_bps": rate_bps,
        "job_completion_time_ps": result.job_completion_time_ps(),
        "goal_completion_time_ps": result.goal_completion_time_ps,
        "quiescent": result.quiescent,
        "flow_count": len(flow_rows),
        "flow_payload_bytes": sum(flow.payload_bytes for flow in result.flows),
        "fct_min_ps": min(flow.fct_ps for flow in result.flows),
        "fct_max_ps": max(flow.fct_ps for flow in result.flows),
        "backend_wall_ns": wall_ns,
        "manifest": result.manifest,
        "completion_csv": str(completion_csv.name),
    }
    if include_raw_flows:
        observation["raw_flows"] = flow_rows
    return observation


def _evaluate_behavior(cells: dict[str, dict[str, Any]]) -> dict[str, Any]:
    def completion(grouping: str, profile: str, rate: int) -> int:
        key = f"{grouping}.{profile}.{rate}"
        return int(cells[key]["job_completion_time_ps"])

    packet_signed = []
    for rate in RATES_BPS:
        subject = completion("per-token", "rnic-nn", rate)
        floor, ceiling = PACKET_DELTA_BANDS_PS[rate]
        for comparator in ("per-expert-group", "aggregate"):
            delta = subject - completion(comparator, "rnic-nn", rate)
            packet_signed.append(
                {
                    "rate_bps": rate,
                    "comparison": f"per-token-minus-{comparator}",
                    "raw_delta_ps": delta,
                    "expected_closed_band_ps": [floor, ceiling],
                    "passed": floor <= delta <= ceiling,
                }
            )

    inverse_rate = []
    for comparator in ("per-expert-group", "aggregate"):
        delta_200 = completion("per-token", "rnic-nn", RATES_BPS[0]) - completion(
            comparator,
            "rnic-nn",
            RATES_BPS[0],
        )
        delta_400 = completion("per-token", "rnic-nn", RATES_BPS[1]) - completion(
            comparator,
            "rnic-nn",
            RATES_BPS[1],
        )
        error = abs(delta_200 - 2 * delta_400)
        inverse_rate.append(
            {
                "comparison": f"per-token-minus-{comparator}",
                "delta_200g_ps": delta_200,
                "delta_400g_ps": delta_400,
                "absolute_scaling_error_ps": error,
                "maximum_error_ps": RATE_SCALING_TOLERANCE_PS,
                "passed": error <= RATE_SCALING_TOLERANCE_PS,
            }
        )

    fluid = []
    for grouping in ("per-expert-group", "per-token"):
        for rate in RATES_BPS:
            delta = completion(grouping, "rnic-nn-fluid", rate) - completion(
                "aggregate",
                "rnic-nn-fluid",
                rate,
            )
            fluid.append(
                {
                    "grouping": grouping,
                    "rate_bps": rate,
                    "raw_delta_ps": delta,
                    "maximum_absolute_delta_ps": FLUID_ABS_DELTA_MAX_PS,
                    "passed": abs(delta) <= FLUID_ABS_DELTA_MAX_PS,
                }
            )
    instances = packet_signed + inverse_rate + fluid
    return {
        "packet_signed_delta": packet_signed,
        "packet_inverse_rate": inverse_rate,
        "fluid_grouping": fluid,
        "registered_family_classes": SCORED_FAMILIES,
        "registered_instances": SCORED_INSTANCES,
        "passed_instances": sum(bool(row["passed"]) for row in instances),
        "all_passed": all(bool(row["passed"]) for row in instances),
    }


def _evaluate_physical_bounds(cells: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for key, cell in sorted(cells.items()):
        floor, ceiling = PHYSICAL_BOUNDS_PS[int(cell["rate_bps"])]
        completion = int(cell["job_completion_time_ps"])
        rows.append(
            {
                "cell": key,
                "completion_time_ps": completion,
                "floor_ps": floor,
                "ceiling_ps": ceiling,
                "passed": floor <= completion <= ceiling,
            }
        )
    return {
        "rows": rows,
        "all_passed": all(bool(row["passed"]) for row in rows),
    }


def _evaluate_synthetic_exact(
    plans: dict[str, Any],
    traces: dict[str, Any],
    routing: Any,
) -> dict[str, Any]:
    dispatch = plans["per-token"][0]
    combine = plans["per-token"][1]
    grouped_dispatch = plans["per-expert-group"][0]
    grouped_combine = plans["per-expert-group"][1]
    aggregate_dispatch = plans["aggregate"][0]
    aggregate_combine = plans["aggregate"][1]
    source_sequences = {
        str(source): [
            message.destination_rank
            for message in dispatch.messages
            if message.source_rank == source
        ]
        for source in range(4)
    }
    expected_sequences = {
        "0": [3, 1, 2, 1, 3, 2, 1, 3],
        "1": [3, 2, 3, 2, 3],
        "2": [3, 1, 1, 3, 1, 3],
        "3": [1, 2, 1, 2, 1],
    }
    routing_routes = tuple(
        token.layers[0].expert_ids for token in routing.requests[0].tokens
    )
    checks = {
        "v2_top_k_tuple_order": routing_routes == ROUTES,
        "per_token_message_count": (
            len(dispatch.messages) + len(combine.messages)
            == MESSAGE_COUNTS["per-token"]
        ),
        "expert_group_message_count": (
            len(grouped_dispatch.messages) + len(grouped_combine.messages)
            == MESSAGE_COUNTS["per-expert-group"]
        ),
        "aggregate_message_count": (
            len(traces["aggregate"].messages) == MESSAGE_COUNTS["aggregate"]
        ),
        "source_sequences": source_sequences == expected_sequences,
        "total_bytes": (
            sum(message.payload_bytes for message in dispatch.messages)
            + sum(message.payload_bytes for message in combine.messages)
            == TOTAL_BYTES
        ),
        "per_token_pair_projection": (
            dispatch.pair_payload_bytes == aggregate_dispatch.pair_payload_bytes
            and combine.pair_payload_bytes == aggregate_combine.pair_payload_bytes
        ),
        "expert_group_request_projection": (
            grouped_dispatch.request_pair_payload_bytes
            == aggregate_dispatch.request_pair_payload_bytes
            and grouped_combine.request_pair_payload_bytes
            == aggregate_combine.request_pair_payload_bytes
        ),
        "combine_transpose": all(
            (
                combine_message.source_rank,
                combine_message.destination_rank,
                combine_message.routing_ordinals,
            )
            == (
                dispatch_message.destination_rank,
                dispatch_message.source_rank,
                dispatch_message.routing_ordinals,
            )
            for dispatch_message, combine_message in zip(
                dispatch.messages,
                combine.messages,
                strict=True,
            )
        ),
    }
    return {
        "checks": checks,
        "source_sequences": source_sequences,
        "all_passed": all(checks.values()),
    }


def _measure_call(function: Any) -> tuple[Any, int, int]:
    tracemalloc.start()
    start = time.perf_counter_ns()
    try:
        value = function()
        elapsed_ns = time.perf_counter_ns() - start
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return value, elapsed_ns, peak_bytes


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


def _run_granite(
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    from simllm.core import step_records_from_jsonl
    from simllm.goal import to_binary
    from simllm.preplay import read_routed_experts
    from simllm.traffic import (
        ExpertPlacementSnapshot,
        MoeMessageGrouping,
        RoutedMoeSupply,
        render_sequenced_step_goal,
        render_step_goal,
        step_moe_alltoalls,
        step_moe_message_sequences,
    )

    record = step_records_from_jsonl(
        args.granite_root / "replay-400g" / "steps.jsonl"
    )[0]
    dims = _granite_dims()
    routing = read_routed_experts(
        args.granite_root / "replay-400g" / "routed-experts.json"
    )
    supply = RoutedMoeSupply(
        routed_experts=routing,
        placements=(
            ExpertPlacementSnapshot(
                placement_epoch=0,
                expert_owners=tuple(
                    (layer, expert, expert % 8)
                    for layer in range(24)
                    for expert in range(32)
                ),
            ),
        ),
        step_placement_epochs=((record.step_index, 0),),
    )
    plan_functions = {
        "aggregate": lambda: step_moe_alltoalls(
            record,
            dims,
            tuple(range(8)),
            routed_supply=supply,
        ),
        "per-expert-group": lambda: step_moe_message_sequences(
            record,
            dims,
            tuple(range(8)),
            routed_supply=supply,
            grouping=MoeMessageGrouping.PER_EXPERT_GROUP,
        ),
        "per-token": lambda: step_moe_message_sequences(
            record,
            dims,
            tuple(range(8)),
            routed_supply=supply,
            grouping=MoeMessageGrouping.PER_TOKEN,
        ),
    }
    render_functions = {
        "aggregate": lambda: render_step_goal(
            record,
            dims,
            (0,),
            4_139,
            ep_ranks=tuple(range(8)),
            routed_supply=supply,
            num_goal_ranks=8,
        ),
        "per-expert-group": lambda: render_sequenced_step_goal(
            record,
            dims,
            (0,),
            4_139,
            ep_ranks=tuple(range(8)),
            routed_supply=supply,
            message_grouping=MoeMessageGrouping.PER_EXPERT_GROUP,
            num_goal_ranks=8,
        ),
        "per-token": lambda: render_sequenced_step_goal(
            record,
            dims,
            (0,),
            4_139,
            ep_ranks=tuple(range(8)),
            routed_supply=supply,
            message_grouping=MoeMessageGrouping.PER_TOKEN,
            num_goal_ranks=8,
        ),
    }

    observations = {}
    for grouping in GROUPINGS:
        plan, plan_ns, plan_peak = _measure_call(plan_functions[grouping])
        trace, render_ns, render_peak = _measure_call(render_functions[grouping])
        goal_path = trace.write(output_dir / f"{grouping}.goal")
        binary_path = output_dir / f"{grouping}.bin"
        start = time.perf_counter_ns()
        to_binary(goal_path, binary_path, tool=args.txt2bin)
        compile_ns = time.perf_counter_ns() - start
        goal = _goal_observation(trace)
        if grouping == "aggregate" and (
            goal["goal_bytes"],
            goal["goal_sha256"],
        ) != GRANITE_AGGREGATE_GOAL:
            raise AssertionError("generated aggregate Granite GOAL changed")

        backends = {}
        for profile in PROFILES:
            completion_csv = output_dir / f"{grouping}.{profile}.csv"
            try:
                backends[profile] = _run_backend_cell(
                    args,
                    binary_path,
                    profile=profile,
                    rate_bps=400_000_000_000,
                    completion_csv=completion_csv,
                    include_raw_flows=False,
                )
                backends[profile]["timed_out"] = False
            except subprocess.TimeoutExpired:
                backends[profile] = {
                    "profile": profile,
                    "rate_bps": 400_000_000_000,
                    "timed_out": True,
                    "timeout_s": BACKEND_TIMEOUT_S,
                    "completion_csv": str(completion_csv.name),
                }

        backend_within_limit = all(
            not result["timed_out"]
            and int(result["backend_wall_ns"])
            <= int(GRANITE_BACKEND_LIMIT_S * 1_000_000_000)
            for result in backends.values()
        )
        peak_bytes = max(plan_peak, render_peak)
        practical = (
            (render_ns + compile_ns)
            <= int(GRANITE_RENDER_COMPILE_LIMIT_S * 1_000_000_000)
            and peak_bytes <= GRANITE_MEMORY_LIMIT_BYTES
            and goal["goal_bytes"] <= GRANITE_GOAL_LIMIT_BYTES
            and backend_within_limit
        )
        observations[grouping] = {
            **goal,
            "trace_schema": routing.trace_schema,
            "order_authority": "reconstructed-v1",
            "plan_item_count": len(plan),
            "plan_ns": plan_ns,
            "render_ns": render_ns,
            "plan_peak_traced_bytes": plan_peak,
            "render_peak_traced_bytes": render_peak,
            "peak_traced_bytes": peak_bytes,
            "binary_bytes": binary_path.stat().st_size,
            "compile_ns": compile_ns,
            "render_plus_compile_ns": render_ns + compile_ns,
            "backends": backends,
            "practical_for_large_sweeps": practical,
        }
        _write_json(output_dir / "progress.json", observations)
    return {
        "step_index": record.step_index,
        "total_new_tokens": record.total_new_tokens,
        "trace_schema": routing.trace_schema,
        "order_authority": "reconstructed-v1, not framework-observed",
        "thresholds": {
            "render_plus_compile_s": GRANITE_RENDER_COMPILE_LIMIT_S,
            "peak_traced_bytes": GRANITE_MEMORY_LIMIT_BYTES,
            "goal_bytes": GRANITE_GOAL_LIMIT_BYTES,
            "backend_wall_s": GRANITE_BACKEND_LIMIT_S,
            "backend_attempt_timeout_s": BACKEND_TIMEOUT_S,
        },
        "groupings": observations,
    }


def run_study(args: argparse.Namespace) -> None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    input_observations = _validate_result_inputs(args)
    args.out.mkdir(parents=True, exist_ok=False)
    synthetic_dir = args.out / "synthetic"
    granite_dir = args.out / "granite"
    synthetic_dir.mkdir()
    granite_dir.mkdir()

    record, dims, supply, routing = _synthetic_fixture(synthetic_dir)
    plans, traces, binaries, render_observations = _render_synthetic(
        args,
        synthetic_dir,
        record,
        dims,
        supply,
    )
    raw = {
        "schema": "simllm-dispatch-sequence-v1-raw",
        "input_observations": input_observations,
        "pre_measurement_physical_bounds_ps": {
            str(rate): list(bounds) for rate, bounds in PHYSICAL_BOUNDS_PS.items()
        },
        "synthetic_renderers": render_observations,
        "synthetic_cells": {},
    }
    raw_path = args.out / "raw_observations.json"
    _write_json(raw_path, raw)
    for grouping in GROUPINGS:
        for profile in PROFILES:
            for rate in RATES_BPS:
                key = f"{grouping}.{profile}.{rate}"
                raw["synthetic_cells"][key] = _run_backend_cell(
                    args,
                    binaries[grouping],
                    profile=profile,
                    rate_bps=rate,
                    completion_csv=synthetic_dir / f"{key}.csv",
                    include_raw_flows=True,
                )
                _write_json(raw_path, raw)

    behavior = _evaluate_behavior(raw["synthetic_cells"])
    raw["behavior_evaluated_before_fatal_oracles"] = behavior
    _write_json(raw_path, raw)

    exact = _evaluate_synthetic_exact(plans, traces, routing)
    physical = _evaluate_physical_bounds(raw["synthetic_cells"])
    granite = _run_granite(args, granite_dir)
    summary = {
        "schema": "simllm-dispatch-sequence-v1-summary",
        "expectation_commit": "7efd71e7e54fc6faecde17c5faebab9430a2e847",
        "behavior": behavior,
        "fatal_unscored": {
            "synthetic_exact": exact,
            "physical_bounds": physical,
            "input_identity": input_observations,
        },
        "granite_scale_and_cost": granite,
        "success": (
            behavior["all_passed"]
            and exact["all_passed"]
            and physical["all_passed"]
        ),
    }
    _write_json(args.out / "summary.json", summary)
    if not summary["success"]:
        raise AssertionError("dispatch sequence study did not meet frozen acceptance")
    print(
        f"dispatch sequence study passed {behavior['passed_instances']}/"
        f"{behavior['registered_instances']} genuine-risk instances"
    )


def main() -> None:
    args = parse_args()
    if args.check_only:
        check_only(args)
        return
    run_study(args)


if __name__ == "__main__":
    main()
