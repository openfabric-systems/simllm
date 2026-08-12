"""TRAF-22 sequenced timing requalification on the corrected full-duplex floor.

This is a fresh qualification. The void run under ``examples/dispatch_sequence_v1``
is untouched: its record, its raw observations and its refuted floor stay as
chronology. Every bound below is derived from the rendered endpoint loads and
the packet backend's full-envelope calendar, and the held-out fixture has never
been rendered or executed before this study.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
import tracemalloc
from dataclasses import asdict
from pathlib import Path
from typing import Any

GROUPINGS = ("aggregate", "per-expert-group", "per-token")
PROFILES = ("rnic-nn", "rnic-nn-fluid")
RATES_BPS = (200_000_000_000, 400_000_000_000)
PACKET_WIRE_BYTES = 4_096
HEADER_BYTES = 64
PACKET_DATA_BYTES = PACKET_WIRE_BYTES - HEADER_BYTES

EXPECTATIONS = "examples/dispatch_sequence_v2/expectations.md"
REPO_ROOT = Path(__file__).resolve().parents[2]

# --- frozen fixture registries ------------------------------------------------

PRIMARY = {
    "name": "primary",
    "engine_rank": 0,
    "ep_ranks": (0, 1, 2, 3),
    "hidden_size": 1_024,
    "dtype_bytes": 2,
    "vector_bytes": 2_048,
    "top_k": 2,
    "num_experts": 4,
    "token_count": 4,
    "routes": ((3, 1), (2, 1), (3, 2), (1, 3)),
    "source_sequence": (3, 1, 2, 1, 3, 2, 1, 3),
    "pair_payload_bytes": {
        "dispatch": ((0, 1, 6_144), (0, 2, 4_096), (0, 3, 6_144)),
        "combine": ((1, 0, 6_144), (2, 0, 4_096), (3, 0, 6_144)),
    },
    "message_counts": {"aggregate": 6, "per-expert-group": 6, "per-token": 16},
    "total_bytes": 32_768,
    "hop_count": 16,
    "hop_ceiling": 16,
    "home_load_bytes": 16_384,
    "floor_ps": {200_000_000_000: 655_360, 400_000_000_000: 327_680},
    "packet_excess_bytes": 8_192,
}

HELD_OUT = {
    "name": "held-out",
    "engine_rank": 0,
    "ep_ranks": (0, 1, 2, 3),
    "hidden_size": 512,
    "dtype_bytes": 2,
    "vector_bytes": 1_024,
    "top_k": 2,
    "num_experts": 4,
    "token_count": 6,
    "routes": ((1, 2), (1, 3), (2, 3), (1, 2), (3, 1), (2, 1)),
    "source_sequence": (1, 2, 1, 3, 2, 3, 1, 2, 3, 1, 2, 1),
    "pair_payload_bytes": {
        "dispatch": ((0, 1, 5_120), (0, 2, 4_096), (0, 3, 3_072)),
        "combine": ((1, 0, 5_120), (2, 0, 4_096), (3, 0, 3_072)),
    },
    "message_counts": {"aggregate": 6, "per-expert-group": 6, "per-token": 24},
    "total_bytes": 24_576,
    "hop_count": 24,
    "hop_ceiling": 24,
    "home_load_bytes": 12_288,
    "floor_ps": {200_000_000_000: 491_520, 400_000_000_000: 245_760},
    "packet_excess_bytes": 28_672,
}

FIXTURES = (PRIMARY, HELD_OUT)

# --- frozen behavioral relations ---------------------------------------------

FROZEN_PACKET_BAND_MULTIPLIER = 4
FROZEN_RATE_SCALING_TOLERANCE_PS = 6_000
FROZEN_FLUID_EQUAL_SET_TOLERANCE_PS = 2_000
FROZEN_GRANITE_RATIO_BAND = (1.95, 2.05)
FROZEN_SCORED_FAMILIES = 5
FROZEN_SCORED_INSTANCES = 34

# --- frozen Granite registry --------------------------------------------------

GRANITE_ROUTING_SHA256 = (
    "24e986e989e21f1bfe7e758d4470928c82c3bbaec06072a839743b9b17d7cf5f"
)
GRANITE_STEPS_SHA256 = (
    "824cd9557293328bb42b593ac893b6a067302e545b087c9219195ccb8031d755"
)
GRANITE = {
    "engine_rank": 0,
    "ep_width": 8,
    "num_layers": 24,
    "token_count": 54,
    "top_k": 8,
    "layer_calc_ns": 4_139,
    "compute_ps": 24 * 4_139 * 1_000,
    "message_counts": {
        "aggregate": 336,
        "per-expert-group": 1_008,
        "per-token": 12_482,
    },
    "total_bytes": 25_563_136,
    "hop_ceiling": 54 * 8 * 24 * 2,
    "peak_egress_bytes": 12_781_568,
    "aggregate_goal": (
        47_399,
        "6bb83366a3936bcf1e435cab008bb55c2966777528a9e8d885dd44d47f5a4943",
    ),
    "retained_aggregate_completion_ps": {
        "rnic-nn": 503_658_600,
        "rnic-nn-fluid": 489_235_306,
    },
    "floor_ps": {200_000_000_000: 610_598_720, 400_000_000_000: 354_967_360},
}

GRANITE_RENDER_COMPILE_LIMIT_S = 30.0
GRANITE_BACKEND_LIMIT_S = 60.0
GRANITE_MEMORY_LIMIT_BYTES = 1 << 30
GRANITE_GOAL_LIMIT_BYTES = 64 << 20
GRANITE_RENDER_ATTEMPT_TIMEOUT_S = 60
BACKEND_TIMEOUT_S = 600


class MeasurementTimedOut(RuntimeError):
    """A scale measurement exceeded its declared wall-time attempt."""

    def __init__(self, elapsed_ns: int, peak_bytes: int):
        super().__init__("scale measurement timed out")
        self.elapsed_ns = elapsed_ns
        self.peak_bytes = peak_bytes


# --- frozen arithmetic --------------------------------------------------------


def serialization_ps(byte_count: int, rate_bps: int) -> int:
    """Whole-picosecond serialization of ``byte_count`` at ``rate_bps``."""

    return -(-byte_count * 8 * 1_000_000_000_000 // rate_bps)


def envelope_bytes(payload_bytes: int, profile: str) -> int:
    """Calendar bytes one message reserves on one serializer."""

    if profile == "rnic-nn":
        packets = -(-payload_bytes // PACKET_DATA_BYTES)
        return packets * PACKET_WIRE_BYTES
    return payload_bytes


def cell_ceiling_ps(payloads: tuple[int, ...], profile: str, rate_bps: int) -> int:
    """Serialize every envelope at both serializers, plus 1 ns per message."""

    total = sum(envelope_bytes(payload, profile) for payload in payloads)
    return 2 * serialization_ps(total, rate_bps) + 1_000 * len(payloads)


def endpoint_loads(
    messages: tuple[tuple[int, int, int], ...],
) -> dict[int, tuple[int, int]]:
    """Return ``{rank: (egress_bytes, ingress_bytes)}`` for directed messages."""

    loads: dict[int, list[int]] = {}
    for source, destination, payload in messages:
        loads.setdefault(source, [0, 0])[0] += payload
        loads.setdefault(destination, [0, 0])[1] += payload
    return {rank: (value[0], value[1]) for rank, value in sorted(loads.items())}


def peak_endpoint_load(messages: tuple[tuple[int, int, int], ...]) -> int:
    loads = endpoint_loads(messages)
    return max((max(pair) for pair in loads.values()), default=0)


def fixture_dispatch_messages(fixture: dict[str, Any]) -> tuple[tuple[int, int], ...]:
    """Frozen ``(destination, count)`` expansion of one fixture's route table."""

    counts: dict[int, int] = {}
    for token_routes in fixture["routes"]:
        for destination in token_routes:
            counts[destination] = counts.get(destination, 0) + 1
    return tuple(sorted(counts.items()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--granite-root", type=Path, required=True)
    parser.add_argument("--htsim-rnic", type=Path, required=True)
    parser.add_argument("--txt2bin", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def check_only(args: argparse.Namespace) -> None:
    """Validate the frozen registries and their arithmetic, and nothing else."""

    if GROUPINGS != ("aggregate", "per-expert-group", "per-token"):
        raise AssertionError("grouping registry drifted")
    if PROFILES != ("rnic-nn", "rnic-nn-fluid"):
        raise AssertionError("profile registry drifted")
    if RATES_BPS != (200_000_000_000, 400_000_000_000):
        raise AssertionError("rate registry drifted")
    if PACKET_DATA_BYTES != 4_032:
        raise AssertionError("packet envelope arithmetic drifted")

    for fixture in FIXTURES:
        name = fixture["name"]
        expansion = fixture_dispatch_messages(fixture)
        vector = fixture["vector_bytes"]
        if vector != fixture["hidden_size"] * fixture["dtype_bytes"]:
            raise AssertionError(f"{name}: vector byte arithmetic drifted")
        if len(fixture["routes"]) != fixture["token_count"]:
            raise AssertionError(f"{name}: token count disagrees with route table")
        if any(len(row) != fixture["top_k"] for row in fixture["routes"]):
            raise AssertionError(f"{name}: route width disagrees with top_k")
        expected_dispatch = tuple(
            (fixture["engine_rank"], destination, count * vector)
            for destination, count in expansion
        )
        if expected_dispatch != fixture["pair_payload_bytes"]["dispatch"]:
            raise AssertionError(f"{name}: dispatch pair registry drifted")
        expected_combine = tuple(
            (destination, source, payload)
            for source, destination, payload in expected_dispatch
        )
        if expected_combine != fixture["pair_payload_bytes"]["combine"]:
            raise AssertionError(f"{name}: combine transpose registry drifted")
        hops = 2 * fixture["token_count"] * fixture["top_k"]
        if hops != fixture["hop_count"] or hops != fixture["hop_ceiling"]:
            raise AssertionError(f"{name}: hop registry drifted")
        if fixture["message_counts"]["per-token"] != hops:
            raise AssertionError(f"{name}: per-token message registry drifted")
        if fixture["message_counts"]["aggregate"] != 2 * len(expansion):
            raise AssertionError(f"{name}: aggregate message registry drifted")
        if (
            fixture["message_counts"]["per-expert-group"]
            != fixture["message_counts"]["aggregate"]
        ):
            raise AssertionError(f"{name}: expert-group message registry drifted")
        if fixture["total_bytes"] != hops * vector:
            raise AssertionError(f"{name}: byte arithmetic drifted")
        if fixture["home_load_bytes"] != fixture["total_bytes"] // 2:
            raise AssertionError(f"{name}: home endpoint load drifted")
        for rate in RATES_BPS:
            expected_floor = serialization_ps(fixture["home_load_bytes"], rate)
            if expected_floor != fixture["floor_ps"][rate]:
                raise AssertionError(f"{name}: full-duplex floor drifted at {rate}")
        aggregate_envelope = sum(
            envelope_bytes(count * vector, "rnic-nn") for _, count in expansion
        )
        per_token_envelope = envelope_bytes(vector, "rnic-nn") * (hops // 2)
        if per_token_envelope - aggregate_envelope != fixture["packet_excess_bytes"]:
            raise AssertionError(f"{name}: envelope excess registry drifted")

    if PRIMARY["floor_ps"][200_000_000_000] != 655_360:
        raise AssertionError("the corrected 200 Gbit/s full-duplex floor drifted")
    if PRIMARY["floor_ps"][400_000_000_000] != 327_680:
        raise AssertionError("the corrected 400 Gbit/s full-duplex floor drifted")
    if 2 * PRIMARY["floor_ps"][400_000_000_000] != PRIMARY["floor_ps"][
        200_000_000_000
    ]:
        raise AssertionError("floors are not inverse in the rate")
    if set(HELD_OUT["routes"]) & set(PRIMARY["routes"]) == set(HELD_OUT["routes"]):
        raise AssertionError("the held-out route table is not a new shape")
    if HELD_OUT["vector_bytes"] == PRIMARY["vector_bytes"]:
        raise AssertionError("the held-out payload is not a new payload")

    if FROZEN_PACKET_BAND_MULTIPLIER != 4:
        raise AssertionError("packet band multiplier drifted")
    if FROZEN_RATE_SCALING_TOLERANCE_PS != 6_000:
        raise AssertionError("rate scaling tolerance drifted")
    if FROZEN_FLUID_EQUAL_SET_TOLERANCE_PS != 2_000:
        raise AssertionError("fluid equal-set tolerance drifted")
    if FROZEN_GRANITE_RATIO_BAND != (1.95, 2.05):
        raise AssertionError("Granite ratio band drifted")
    packet_instances = len(FIXTURES) * len(RATES_BPS) * 2
    rate_instances = len(FIXTURES) * len(GROUPINGS) * len(PROFILES)
    fluid_direction_instances = len(FIXTURES) * len(RATES_BPS)
    fluid_equal_instances = len(FIXTURES) * len(RATES_BPS)
    granite_instances = len(GROUPINGS) * len(PROFILES)
    registered = (
        packet_instances
        + rate_instances
        + fluid_direction_instances
        + fluid_equal_instances
        + granite_instances
    )
    if (FROZEN_SCORED_FAMILIES, registered) != (5, FROZEN_SCORED_INSTANCES):
        raise AssertionError("evidence accounting drifted")

    if GRANITE["compute_ps"] != 99_336_000:
        raise AssertionError("Granite represented compute drifted")
    if GRANITE["message_counts"]["per-token"] > GRANITE["hop_ceiling"]:
        raise AssertionError("Granite hops exceed the independent ceiling")
    if (
        GRANITE["total_bytes"]
        != GRANITE["message_counts"]["per-token"] * PRIMARY["vector_bytes"]
    ):
        raise AssertionError("Granite byte arithmetic drifted")
    if GRANITE["hop_ceiling"] != 54 * 8 * 24 * 2:
        raise AssertionError("Granite hop ceiling drifted")
    for rate in RATES_BPS:
        expected = GRANITE["compute_ps"] + serialization_ps(
            GRANITE["peak_egress_bytes"],
            rate,
        )
        if expected != GRANITE["floor_ps"][rate]:
            raise AssertionError(f"Granite step floor drifted at {rate}")
    if len(GRANITE["aggregate_goal"][1]) != 64:
        raise AssertionError("Granite aggregate GOAL digest is malformed")
    if any(len(value) != 64 for value in (GRANITE_ROUTING_SHA256, GRANITE_STEPS_SHA256)):
        raise AssertionError("Granite input digest is malformed")

    if not (REPO_ROOT / EXPECTATIONS).is_file():
        raise AssertionError("the frozen expectations record is missing")
    if any(
        not str(path)
        for path in (args.out, args.granite_root, args.htsim_rnic, args.txt2bin)
    ):
        raise AssertionError("registered path argument is empty")
    print(
        "check-only validated the frozen dispatch-sequence-v2 registries, "
        "full-duplex floors, envelope ceilings and evidence accounting; "
        "produced no artifacts"
    )


# --- run helpers --------------------------------------------------------------


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
    return {"bytes": path.stat().st_size, "sha256": _sha256(path)}


def _git_object(revision: str) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", revision],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _measure_call(function: Any, *, timeout_s: int | None = None) -> tuple[Any, int, int]:
    if timeout_s is not None and not hasattr(signal, "setitimer"):
        raise RuntimeError("timed scale measurements require POSIX interval timers")

    def timed_out(_signum: int, _frame: Any) -> None:
        raise TimeoutError

    previous_handler: Any = None
    if timeout_s is not None:
        previous_handler = signal.signal(signal.SIGALRM, timed_out)
        signal.setitimer(signal.ITIMER_REAL, timeout_s)
    tracemalloc.start()
    start = time.perf_counter_ns()
    try:
        value = function()
        elapsed_ns = time.perf_counter_ns() - start
        _, peak_bytes = tracemalloc.get_traced_memory()
    except TimeoutError:
        elapsed_ns = time.perf_counter_ns() - start
        _, peak_bytes = tracemalloc.get_traced_memory()
        raise MeasurementTimedOut(elapsed_ns, peak_bytes) from None
    finally:
        tracemalloc.stop()
        if timeout_s is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)
    return value, elapsed_ns, peak_bytes


def _validate_result_inputs(args: argparse.Namespace) -> dict[str, Any]:
    configured_root = os.environ.get("SIMLLM_WAVE10_RUN_ROOT")
    if not configured_root:
        raise RuntimeError(
            "SIMLLM_WAVE10_RUN_ROOT must name the external wave-10 run root"
        )
    run_root = Path(configured_root).resolve()
    try:
        args.out.resolve().relative_to(run_root)
    except ValueError as exc:
        raise ValueError(
            "study output must remain under SIMLLM_WAVE10_RUN_ROOT"
        ) from exc
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
    }
    for label, path in source_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing Granite {label}: {path}")
    expected = {"routing": GRANITE_ROUTING_SHA256, "steps": GRANITE_STEPS_SHA256}
    observations = {}
    for label, path in source_paths.items():
        observation = _path_observation(path)
        observation["relative_path"] = str(path.relative_to(args.granite_root))
        observation["authored_against_sha256"] = expected[label]
        observation["matches_authored_against"] = (
            observation["sha256"] == expected[label]
        )
        observations[label] = observation
    return {
        "artifacts": observations,
        "observed_simllm_commit": _git_object("HEAD"),
        "observed_htsim_gitlink": _git_object("HEAD:third_party/htsim"),
        "htsim_binary": _path_observation(args.htsim_rnic),
        "txt2bin_binary": _path_observation(args.txt2bin),
        "all_passed": all(
            bool(observation["matches_authored_against"])
            for observation in observations.values()
        ),
    }


def _build_fixture(fixture: dict[str, Any], output_dir: Path) -> tuple[Any, Any, Any, Any]:
    """Materialize one frozen fixture as a record, dims, supply and routing."""

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

    token_count = fixture["token_count"]
    provenance = FrameworkTraceProvenance(
        model_id=f"study/sequence-{fixture['name']}",
        model_revision="frozen-v2",
        model_class="SequenceFixtureMoeForCausalLM",
        dtype="float16",
        tokenizer_sha256="a" * 64,
        sampling=SamplingConfig.greedy(),
        capture_host="registered-fixture",
        runner="fixture-runner",
        framework="vllm",
        framework_version="fixture-v2",
        observed_source="b" * 40,
        authored_against_source="c" * 40,
        torch_version="fixture-v2",
        device="cpu",
        torch_num_threads=1,
        engine_seed=1,
        eos_token_id=0,
        top_k=fixture["top_k"],
        expert_count=fixture["num_experts"],
        moe_layer_indices=(0,),
        kv_page_size=1,
        kv_token_capacity=64,
        dispatch_layer_mapping="framework-layer-id",
    )
    request = FrameworkRequestTrace(
        request_id="alpha",
        prompt_sha256="d" * 64,
        prompt_format=PromptFormat.TEXT,
        input_token_ids=tuple(10 + index for index in range(token_count)),
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
                routing=(ObservedLayerDispatch(layer_index=0, expert_ids=expert_ids),),
            )
            for index, expert_ids in enumerate(fixture["routes"])
        ),
        decode_dispatch=(),
    )
    trace_path = write_framework_preplay_trace(
        output_dir / f"{fixture['name']}-framework-trace.jsonl",
        FrameworkPreplayTrace(provenance=provenance, requests=(request,), kv_events=()),
    )
    routing = project_framework_routing(trace_path)
    dims = ModelDims(
        num_layers=1,
        hidden_size=fixture["hidden_size"],
        intermediate_size=2 * fixture["hidden_size"],
        num_heads=16,
        num_kv_heads=8,
        head_size=fixture["hidden_size"] // 16,
        vocab_size=1_024,
        dtype_bytes=fixture["dtype_bytes"],
        num_experts=fixture["num_experts"],
        top_k=fixture["top_k"],
        moe_intermediate_size=fixture["hidden_size"],
        local_num_experts=1,
    )
    placement = ExpertPlacementSnapshot(
        placement_epoch=0,
        expert_owners=tuple((0, expert, expert) for expert in range(fixture["num_experts"])),
    )
    supply = RoutedMoeSupply(
        engine_rank=fixture["engine_rank"],
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
                token_count,
                context_length=token_count,
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


def _plan_and_render(
    fixture: dict[str, Any],
    record: Any,
    dims: Any,
    supply: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from simllm.traffic import (
        MoeMessageGrouping,
        render_sequenced_step_goal,
        render_step_goal,
        step_moe_alltoalls,
        step_moe_message_sequences,
    )

    ep_ranks = fixture["ep_ranks"]
    plans = {
        "aggregate": step_moe_alltoalls(record, dims, ep_ranks, routed_supply=supply),
        "per-expert-group": step_moe_message_sequences(
            record,
            dims,
            ep_ranks,
            routed_supply=supply,
            grouping=MoeMessageGrouping.PER_EXPERT_GROUP,
        ),
        "per-token": step_moe_message_sequences(
            record,
            dims,
            ep_ranks,
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
            ep_ranks=ep_ranks,
            routed_supply=supply,
        ),
        "per-expert-group": render_sequenced_step_goal(
            record,
            dims,
            (0,),
            0,
            ep_ranks=ep_ranks,
            routed_supply=supply,
            message_grouping=MoeMessageGrouping.PER_EXPERT_GROUP,
        ),
        "per-token": render_sequenced_step_goal(
            record,
            dims,
            (0,),
            0,
            ep_ranks=ep_ranks,
            routed_supply=supply,
            message_grouping=MoeMessageGrouping.PER_TOKEN,
        ),
    }
    return plans, traces


def _compile_traces(
    args: argparse.Namespace,
    output_dir: Path,
    prefix: str,
    traces: dict[str, Any],
) -> tuple[dict[str, Path], dict[str, Any]]:
    from simllm.goal import to_binary

    binaries = {}
    observations = {}
    for grouping in GROUPINGS:
        goal_path = traces[grouping].write(output_dir / f"{prefix}.{grouping}.goal")
        binary_path = output_dir / f"{prefix}.{grouping}.bin"
        start = time.perf_counter_ns()
        to_binary(goal_path, binary_path, tool=args.txt2bin)
        compile_ns = time.perf_counter_ns() - start
        observations[grouping] = {
            **_goal_observation(traces[grouping]),
            "binary_bytes": binary_path.stat().st_size,
            "compile_ns": compile_ns,
        }
        binaries[grouping] = binary_path
    return binaries, observations


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


# --- evidence -----------------------------------------------------------------


def _trace_directed_messages(trace: Any) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        (message.source_rank, message.destination_rank, message.payload_bytes)
        for message in trace.messages
    )


def _plan_pair_table(plans: dict[str, Any], grouping: str) -> dict[str, Any]:
    """Ordered-pair table keyed by ``(layer, phase, source, destination)``."""

    table: dict[str, int] = {}
    requests: dict[str, int] = {}
    if grouping == "aggregate":
        for operation in plans[grouping]:
            for source, destination, payload in operation.pair_payload_bytes:
                key = f"{operation.layer}:{operation.phase}:{source}:{destination}"
                table[key] = table.get(key, 0) + payload
            for request_id, source, destination, payload in (
                operation.request_pair_payload_bytes
            ):
                key = (
                    f"{operation.layer}:{operation.phase}:"
                    f"{request_id}:{source}:{destination}"
                )
                requests[key] = requests.get(key, 0) + payload
        return {"pairs": table, "requests": requests}
    for sequence in plans[grouping]:
        for message in sequence.messages:
            key = (
                f"{sequence.layer}:{sequence.phase}:"
                f"{message.source_rank}:{message.destination_rank}"
            )
            table[key] = table.get(key, 0) + message.payload_bytes
            request_key = (
                f"{sequence.layer}:{sequence.phase}:{message.request_id}:"
                f"{message.source_rank}:{message.destination_rank}"
            )
            requests[request_key] = requests.get(request_key, 0) + message.payload_bytes
    return {"pairs": table, "requests": requests}


def _fixture_exact_rows(
    fixture: dict[str, Any],
    plans: dict[str, Any],
    traces: dict[str, Any],
    routing: Any,
) -> dict[str, Any]:
    engine = fixture["engine_rank"]
    vector = fixture["vector_bytes"]
    aggregate_table = _plan_pair_table(plans, "aggregate")
    dispatch = plans["per-token"][0]
    combine = plans["per-token"][1]
    routing_routes = tuple(
        token.layers[0].expert_ids for token in routing.requests[0].tokens
    )
    source_sequence = tuple(
        message.destination_rank
        for message in dispatch.messages
        if message.source_rank == engine
    )
    total_bytes = {
        grouping: sum(message.payload_bytes for message in traces[grouping].messages)
        for grouping in GROUPINGS
    }
    message_counts = {
        grouping: len(traces[grouping].messages) for grouping in GROUPINGS
    }
    hops = sum(total_bytes.values()) // (len(GROUPINGS) * vector)
    checks = {
        "route_tuple_order": routing_routes == fixture["routes"],
        "source_sequence": source_sequence == fixture["source_sequence"],
        "rendered_message_counts": message_counts == fixture["message_counts"],
        "rendered_total_bytes": all(
            value == fixture["total_bytes"] for value in total_bytes.values()
        ),
        "hop_ceiling": hops == fixture["hop_count"] <= fixture["hop_ceiling"],
        "exact_pair_payloads": (
            plans["aggregate"][0].pair_payload_bytes
            == fixture["pair_payload_bytes"]["dispatch"]
            and plans["aggregate"][1].pair_payload_bytes
            == fixture["pair_payload_bytes"]["combine"]
        ),
        "sequenced_pair_equality": all(
            _plan_pair_table(plans, grouping)["pairs"] == aggregate_table["pairs"]
            for grouping in ("per-expert-group", "per-token")
        ),
        "sequenced_request_equality": all(
            _plan_pair_table(plans, grouping)["requests"]
            == aggregate_table["requests"]
            for grouping in ("per-expert-group", "per-token")
        ),
        "engine_ownership": all(
            message.source_rank == engine for message in dispatch.messages
        )
        and all(message.destination_rank == engine for message in combine.messages),
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
        "rendered_endpoint_load": all(
            peak_endpoint_load(_trace_directed_messages(traces[grouping]))
            == fixture["home_load_bytes"]
            for grouping in GROUPINGS
        ),
    }
    return {
        "fixture": fixture["name"],
        "checks": checks,
        "source_sequence": list(source_sequence),
        "observed_hops": hops,
        "hop_ceiling": fixture["hop_ceiling"],
        "message_counts": message_counts,
        "all_passed": all(checks.values()),
    }


def _fixture_bounds_rows(
    fixture: dict[str, Any],
    traces: dict[str, Any],
    cells: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for grouping in GROUPINGS:
        payloads = tuple(
            message.payload_bytes for message in traces[grouping].messages
        )
        for profile in PROFILES:
            for rate in RATES_BPS:
                cell = cells[f"{fixture['name']}.{grouping}.{profile}.{rate}"]
                completion = int(cell["job_completion_time_ps"])
                floor = fixture["floor_ps"][rate]
                ceiling = cell_ceiling_ps(payloads, profile, rate)
                rows.append(
                    {
                        "cell": f"{fixture['name']}.{grouping}.{profile}.{rate}",
                        "completion_time_ps": completion,
                        "floor_ps": floor,
                        "ceiling_ps": ceiling,
                        "passed": floor <= completion <= ceiling,
                    }
                )
    return rows


def _evaluate_behavior(
    cells: dict[str, Any],
    granite_cells: dict[str, Any],
) -> dict[str, Any]:
    def completion(fixture: str, grouping: str, profile: str, rate: int) -> int:
        return int(cells[f"{fixture}.{grouping}.{profile}.{rate}"]["job_completion_time_ps"])

    packet_band = []
    for fixture in FIXTURES:
        excess = fixture["packet_excess_bytes"]
        for rate in RATES_BPS:
            low = serialization_ps(excess, rate)
            high = FROZEN_PACKET_BAND_MULTIPLIER * low
            subject = completion(fixture["name"], "per-token", "rnic-nn", rate)
            for comparator in ("per-expert-group", "aggregate"):
                delta = subject - completion(
                    fixture["name"], comparator, "rnic-nn", rate
                )
                packet_band.append(
                    {
                        "fixture": fixture["name"],
                        "rate_bps": rate,
                        "comparison": f"per-token-minus-{comparator}",
                        "raw_delta_ps": delta,
                        "expected_closed_band_ps": [low, high],
                        "passed": low <= delta <= high,
                    }
                )

    rate_scaling = []
    for fixture in FIXTURES:
        for grouping in GROUPINGS:
            for profile in PROFILES:
                slow = completion(fixture["name"], grouping, profile, RATES_BPS[0])
                fast = completion(fixture["name"], grouping, profile, RATES_BPS[1])
                error = abs(slow - 2 * fast)
                rate_scaling.append(
                    {
                        "fixture": fixture["name"],
                        "grouping": grouping,
                        "profile": profile,
                        "completion_200g_ps": slow,
                        "completion_400g_ps": fast,
                        "absolute_scaling_error_ps": error,
                        "maximum_error_ps": FROZEN_RATE_SCALING_TOLERANCE_PS,
                        "passed": error <= FROZEN_RATE_SCALING_TOLERANCE_PS,
                    }
                )

    fluid_direction = []
    fluid_equal_set = []
    for fixture in FIXTURES:
        for rate in RATES_BPS:
            base = completion(fixture["name"], "aggregate", "rnic-nn-fluid", rate)
            per_token = completion(fixture["name"], "per-token", "rnic-nn-fluid", rate)
            grouped = completion(
                fixture["name"], "per-expert-group", "rnic-nn-fluid", rate
            )
            fluid_direction.append(
                {
                    "fixture": fixture["name"],
                    "rate_bps": rate,
                    "raw_delta_ps": per_token - base,
                    "passed": per_token > base,
                }
            )
            fluid_equal_set.append(
                {
                    "fixture": fixture["name"],
                    "rate_bps": rate,
                    "raw_delta_ps": grouped - base,
                    "maximum_absolute_delta_ps": FROZEN_FLUID_EQUAL_SET_TOLERANCE_PS,
                    "passed": abs(grouped - base)
                    <= FROZEN_FLUID_EQUAL_SET_TOLERANCE_PS,
                }
            )

    granite_scaling = []
    low_band, high_band = FROZEN_GRANITE_RATIO_BAND
    compute_ps = GRANITE["compute_ps"]
    for grouping in GROUPINGS:
        for profile in PROFILES:
            slow_cell = granite_cells.get(f"{grouping}.{profile}.{RATES_BPS[0]}")
            fast_cell = granite_cells.get(f"{grouping}.{profile}.{RATES_BPS[1]}")
            if slow_cell is None or fast_cell is None:
                granite_scaling.append(
                    {
                        "grouping": grouping,
                        "profile": profile,
                        "ratio": None,
                        "expected_band": [low_band, high_band],
                        "passed": False,
                        "note": "cell not measured",
                    }
                )
                continue
            slow = int(slow_cell["job_completion_time_ps"]) - compute_ps
            fast = int(fast_cell["job_completion_time_ps"]) - compute_ps
            ratio = slow / fast if fast > 0 else None
            granite_scaling.append(
                {
                    "grouping": grouping,
                    "profile": profile,
                    "network_term_200g_ps": slow,
                    "network_term_400g_ps": fast,
                    "ratio": ratio,
                    "expected_band": [low_band, high_band],
                    "passed": ratio is not None and low_band <= ratio <= high_band,
                }
            )

    families = {
        "packet_envelope_band": packet_band,
        "synthetic_rate_scaling": rate_scaling,
        "fluid_granularity_direction": fluid_direction,
        "fluid_equal_message_set": fluid_equal_set,
        "granite_rate_scaling": granite_scaling,
    }
    instances = [row for rows in families.values() for row in rows]
    family_results = {
        name: all(bool(row["passed"]) for row in rows) for name, rows in families.items()
    }
    return {
        **families,
        "registered_family_classes": FROZEN_SCORED_FAMILIES,
        "registered_instances": FROZEN_SCORED_INSTANCES,
        "observed_instances": len(instances),
        "passed_family_classes": sum(family_results.values()),
        "passed_instances": sum(bool(row["passed"]) for row in instances),
        "family_results": family_results,
        "all_passed": all(bool(row["passed"]) for row in instances),
    }


def _granite_dims() -> Any:
    from simllm.compute import ModelDims

    return ModelDims(
        num_layers=24,
        hidden_size=1_024,
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


def _run_granite(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
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

    record = step_records_from_jsonl(args.granite_root / "replay-400g" / "steps.jsonl")[0]
    dims = _granite_dims()
    routing = read_routed_experts(
        args.granite_root / "replay-400g" / "routed-experts.json"
    )
    ep_ranks = tuple(range(GRANITE["ep_width"]))
    supply = RoutedMoeSupply(
        engine_rank=GRANITE["engine_rank"],
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
            record, dims, ep_ranks, routed_supply=supply
        ),
        "per-expert-group": lambda: step_moe_message_sequences(
            record,
            dims,
            ep_ranks,
            routed_supply=supply,
            grouping=MoeMessageGrouping.PER_EXPERT_GROUP,
        ),
        "per-token": lambda: step_moe_message_sequences(
            record,
            dims,
            ep_ranks,
            routed_supply=supply,
            grouping=MoeMessageGrouping.PER_TOKEN,
        ),
    }
    render_functions = {
        "aggregate": lambda: render_step_goal(
            record,
            dims,
            (0,),
            GRANITE["layer_calc_ns"],
            ep_ranks=ep_ranks,
            routed_supply=supply,
            num_goal_ranks=GRANITE["ep_width"],
        ),
        "per-expert-group": lambda: render_sequenced_step_goal(
            record,
            dims,
            (0,),
            GRANITE["layer_calc_ns"],
            ep_ranks=ep_ranks,
            routed_supply=supply,
            message_grouping=MoeMessageGrouping.PER_EXPERT_GROUP,
            num_goal_ranks=GRANITE["ep_width"],
        ),
        "per-token": lambda: render_sequenced_step_goal(
            record,
            dims,
            (0,),
            GRANITE["layer_calc_ns"],
            ep_ranks=ep_ranks,
            routed_supply=supply,
            message_grouping=MoeMessageGrouping.PER_TOKEN,
            num_goal_ranks=GRANITE["ep_width"],
        ),
    }

    observations: dict[str, Any] = {}
    cells: dict[str, Any] = {}
    for grouping in GROUPINGS:
        plan, plan_ns, plan_peak = _measure_call(plan_functions[grouping])
        if grouping == "aggregate":
            planned_message_count = sum(
                len(operation.pair_payload_bytes) for operation in plan
            )
            planned_message_bytes = sum(
                sum(size for _, _, size in operation.pair_payload_bytes)
                for operation in plan
            )
        else:
            planned_message_count = sum(len(sequence.messages) for sequence in plan)
            planned_message_bytes = sum(
                sum(message.payload_bytes for message in sequence.messages)
                for sequence in plan
            )
        try:
            trace, render_ns, render_peak = _measure_call(
                render_functions[grouping],
                timeout_s=GRANITE_RENDER_ATTEMPT_TIMEOUT_S,
            )
        except MeasurementTimedOut as exc:
            observations[grouping] = {
                "plan_item_count": len(plan),
                "planned_message_count": planned_message_count,
                "planned_message_bytes": planned_message_bytes,
                "plan_ns": plan_ns,
                "plan_peak_traced_bytes": plan_peak,
                "render_timed_out": True,
                "render_attempt_ns": exc.elapsed_ns,
                "render_attempt_peak_traced_bytes": exc.peak_bytes,
                "backends": {},
            }
            _write_json(output_dir / "granite-progress.json", observations)
            continue
        goal_path = trace.write(output_dir / f"granite.{grouping}.goal")
        binary_path = output_dir / f"granite.{grouping}.bin"
        start = time.perf_counter_ns()
        to_binary(goal_path, binary_path, tool=args.txt2bin)
        compile_ns = time.perf_counter_ns() - start
        goal = _goal_observation(trace)
        peak_load = peak_endpoint_load(_trace_directed_messages(trace))
        packet_envelope_total = sum(
            envelope_bytes(message.payload_bytes, "rnic-nn")
            for message in trace.messages
        )

        backends = {}
        for profile in PROFILES:
            for rate in RATES_BPS:
                completion_csv = output_dir / f"granite.{grouping}.{profile}.{rate}.csv"
                cell = _run_backend_cell(
                    args,
                    binary_path,
                    profile=profile,
                    rate_bps=rate,
                    completion_csv=completion_csv,
                    include_raw_flows=False,
                )
                backends[f"{profile}.{rate}"] = cell
                cells[f"{grouping}.{profile}.{rate}"] = cell
        observations[grouping] = {
            "plan_item_count": len(plan),
            "planned_message_count": planned_message_count,
            "planned_message_bytes": planned_message_bytes,
            "plan_ns": plan_ns,
            "plan_peak_traced_bytes": plan_peak,
            "render_ns": render_ns,
            "render_peak_traced_bytes": render_peak,
            "compile_ns": compile_ns,
            "render_plus_compile_s": (render_ns + compile_ns) / 1e9,
            "peak_traced_bytes": max(plan_peak, render_peak),
            "peak_endpoint_load_bytes": peak_load,
            "packet_envelope_bytes": packet_envelope_total,
            "render_timed_out": False,
            **goal,
            "backends": backends,
        }
        _write_json(output_dir / "granite-progress.json", observations)
    return {"groupings": observations, "cells": cells}


def _granite_exact_rows(granite: dict[str, Any]) -> dict[str, Any]:
    groupings = granite["groupings"]
    cells = granite["cells"]
    checks: dict[str, Any] = {}
    for grouping in GROUPINGS:
        observation = groupings.get(grouping)
        if observation is None or observation.get("render_timed_out", True):
            checks[f"{grouping}_rendered"] = False
            continue
        checks[f"{grouping}_rendered"] = True
        checks[f"{grouping}_message_count"] = (
            observation["message_count"] == GRANITE["message_counts"][grouping]
        )
        checks[f"{grouping}_directed_bytes"] = (
            observation["message_bytes"] == GRANITE["total_bytes"]
        )
        checks[f"{grouping}_peak_endpoint_load"] = (
            observation["peak_endpoint_load_bytes"] == GRANITE["peak_egress_bytes"]
        )
        checks[f"{grouping}_render_compile_limit"] = (
            observation["render_plus_compile_s"] <= GRANITE_RENDER_COMPILE_LIMIT_S
        )
        checks[f"{grouping}_memory_limit"] = (
            observation["peak_traced_bytes"] <= GRANITE_MEMORY_LIMIT_BYTES
        )
        checks[f"{grouping}_goal_limit"] = (
            observation["goal_bytes"] <= GRANITE_GOAL_LIMIT_BYTES
        )
        checks[f"{grouping}_backend_limit"] = all(
            cell["backend_wall_ns"] / 1e9 <= GRANITE_BACKEND_LIMIT_S
            for cell in observation["backends"].values()
        )
    checks["hop_ceiling"] = (
        GRANITE["message_counts"]["per-token"] <= GRANITE["hop_ceiling"]
    )
    aggregate = groupings.get("aggregate")
    if aggregate is not None and not aggregate.get("render_timed_out", True):
        checks["aggregate_goal_identity"] = (
            aggregate["goal_bytes"],
            aggregate["goal_sha256"],
        ) == GRANITE["aggregate_goal"]
        for profile, retained in GRANITE["retained_aggregate_completion_ps"].items():
            cell = cells.get(f"aggregate.{profile}.{RATES_BPS[1]}")
            checks[f"aggregate_retained_{profile}"] = (
                cell is not None
                and int(cell["job_completion_time_ps"]) == retained
            )
    else:
        checks["aggregate_goal_identity"] = False
    return {"checks": checks, "all_passed": all(bool(v) for v in checks.values())}


def _granite_bounds_rows(granite: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for grouping in GROUPINGS:
        observation = granite["groupings"].get(grouping)
        if observation is None or observation.get("render_timed_out", True):
            continue
        message_count = observation["message_count"]
        for profile in PROFILES:
            for rate in RATES_BPS:
                cell = granite["cells"].get(f"{grouping}.{profile}.{rate}")
                if cell is None:
                    continue
                completion = int(cell["job_completion_time_ps"])
                floor = GRANITE["floor_ps"][rate]
                if profile == "rnic-nn":
                    envelope_total = observation["packet_envelope_bytes"]
                else:
                    envelope_total = observation["message_bytes"]
                ceiling = (
                    GRANITE["compute_ps"]
                    + 2 * serialization_ps(envelope_total, rate)
                    + 1_000 * message_count
                )
                rows.append(
                    {
                        "cell": f"granite.{grouping}.{profile}.{rate}",
                        "completion_time_ps": completion,
                        "floor_ps": floor,
                        "ceiling_ps": ceiling,
                        "passed": floor <= completion <= ceiling,
                    }
                )
    return rows


def _quiescence_rows(cells: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for key, cell in sorted(cells.items()):
        # The backend manifest is a list of raw "[RNIC manifest]" lines.
        manifest = cell.get("manifest") or []
        verified = any("physical_quiescence=verified" in line for line in manifest)
        rows.append(
            {
                "cell": key,
                "quiescent": bool(cell.get("quiescent")),
                "physical_quiescence_verified": verified,
                "passed": bool(cell.get("quiescent")) and verified,
            }
        )
    return {"rows": rows, "all_passed": all(row["passed"] for row in rows)}


def run_study(args: argparse.Namespace) -> None:
    inputs = _validate_result_inputs(args)
    output_dir = args.out
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "inputs.json", inputs)

    synthetic_cells: dict[str, Any] = {}
    exact_rows = []
    bounds_rows = []
    goal_observations: dict[str, Any] = {}
    for fixture in FIXTURES:
        record, dims, supply, routing = _build_fixture(fixture, output_dir)
        plans, traces = _plan_and_render(fixture, record, dims, supply)
        _, observations = _compile_traces(args, output_dir, fixture["name"], traces)
        goal_observations[fixture["name"]] = observations
        for grouping in GROUPINGS:
            binary_path = output_dir / f"{fixture['name']}.{grouping}.bin"
            for profile in PROFILES:
                for rate in RATES_BPS:
                    key = f"{fixture['name']}.{grouping}.{profile}.{rate}"
                    completion_csv = output_dir / f"{key}.csv"
                    synthetic_cells[key] = _run_backend_cell(
                        args,
                        binary_path,
                        profile=profile,
                        rate_bps=rate,
                        completion_csv=completion_csv,
                        include_raw_flows=True,
                    )
        exact_rows.append(_fixture_exact_rows(fixture, plans, traces, routing))
        bounds_rows.extend(_fixture_bounds_rows(fixture, traces, synthetic_cells))
    _write_json(output_dir / "synthetic-cells.json", synthetic_cells)

    granite = _run_granite(args, output_dir)
    _write_json(output_dir / "granite.json", granite)

    # Scored relations are computed from raw completions before any fatal
    # oracle below is consulted.
    behavior = _evaluate_behavior(synthetic_cells, granite["cells"])

    all_cells = dict(synthetic_cells)
    for key, cell in granite["cells"].items():
        all_cells[f"granite.{key}"] = cell
    granite_bounds = _granite_bounds_rows(granite)
    fatal = {
        "input_identity": inputs["all_passed"],
        "fixture_exact": all(row["all_passed"] for row in exact_rows),
        "synthetic_bounds": all(row["passed"] for row in bounds_rows),
        "granite_bounds": all(row["passed"] for row in granite_bounds),
        "granite_exact": _granite_exact_rows(granite)["all_passed"],
        "quiescence": _quiescence_rows(all_cells)["all_passed"],
    }
    void = not all(fatal.values())
    summary = {
        "study": "dispatch_sequence_v2",
        "expectations": EXPECTATIONS,
        "inputs": inputs,
        "goal_observations": goal_observations,
        "fixture_exact_rows": exact_rows,
        "synthetic_bounds_rows": bounds_rows,
        "granite_bounds_rows": granite_bounds,
        "granite_exact": _granite_exact_rows(granite),
        "quiescence": _quiescence_rows(all_cells),
        "behavior": behavior,
        "fatal_guards": fatal,
        "void": void,
        "outcome": (
            "void"
            if void
            else ("passed" if behavior["all_passed"] else "failed")
        ),
    }
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("outcome", "void", "fatal_guards")}, indent=2))
    print(
        json.dumps(
            {
                "passed_instances": behavior["passed_instances"],
                "registered_instances": behavior["registered_instances"],
                "family_results": behavior["family_results"],
            },
            indent=2,
        )
    )
    if void:
        raise SystemExit(2)
    if not behavior["all_passed"]:
        raise SystemExit(3)


def main() -> None:
    args = parse_args()
    if args.check_only:
        check_only(args)
        return
    run_study(args)


if __name__ == "__main__":
    main()
