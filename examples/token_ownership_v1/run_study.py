"""Run the frozen TRAF-25 token-ownership study after implementation."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

EXPECTATIONS_COMMIT = "cdf03d2"
EVIDENCE_AUTHORED_AGAINST = "cede92930a469bd0be2f2c588866885c9e0e3618"

EP_WORLDS = (2, 4, 8)
BANDWIDTHS_BPS = (200_000_000_000, 400_000_000_000)
PROFILES = ("rnic-nn-fluid", "rnic-nn")
VECTOR_BYTES = 2_048
TOKEN_COUNT = 54
TOP_K = 8
LAYER_COUNT = 24
PHASE_COUNT = 48
PROPAGATION_PS = 2_000_000
COMPUTE_PS = 99_360_000
PUBLISHED_FLUID_JCT_PS = 974_838_253

BYTE_CELLS = {
    2: {
        "legacy_total": 10_612_736,
        "corrected_total": 5_304_320,
        "legacy_peak": 5_306_368,
        "corrected_peak": 2_652_160,
    },
    4: {
        "legacy_total": 58_773_504,
        "corrected_total": 14_594_048,
        "legacy_peak": 14_792_704,
        "corrected_peak": 7_297_024,
    },
    8: {
        "legacy_total": 207_499_264,
        "corrected_total": 25_563_136,
        "legacy_peak": 27_060_224,
        "corrected_peak": 12_781_568,
    },
}

CORRECTED_FLUID_JCT_PS = {
    (2, 200_000_000_000): 407_532_800,
    (2, 400_000_000_000): 301_446_400,
    (4, 200_000_000_000): 779_121_920,
    (4, 400_000_000_000): 487_240_960,
    (8, 200_000_000_000): 1_217_885_440,
    (8, 400_000_000_000): 706_622_720,
}

REQUEST_BYTES_W8 = {
    "r0": 10_403_840,
    "r1": 5_701_632,
    "r2": 9_457_664,
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
    "archived_goal": (
        "replay-400g/htsim/step-000000.goal",
        "08a0403af66ff8a9d6b18f93afd15ae0bc925cc85555acf8a0593438a3d7bc92",
    ),
}

TOTAL_RATIO_BANDS = {
    2: (1.95, 2.05),
    4: (3.9, 4.2),
    8: (7.9, 8.3),
}
PEAK_RATIO_BANDS = {
    2: (1.95, 2.05),
    4: (1.95, 2.15),
    8: (1.95, 2.30),
}


def _check_frozen_registry() -> None:
    if tuple(BYTE_CELLS) != EP_WORLDS:
        raise AssertionError("EP-width byte registry is incomplete")
    if set(CORRECTED_FLUID_JCT_PS) != {
        (world, bandwidth)
        for world in EP_WORLDS
        for bandwidth in BANDWIDTHS_BPS
    }:
        raise AssertionError("fluid JCT registry is incomplete")
    if sum(REQUEST_BYTES_W8.values()) != BYTE_CELLS[8]["corrected_total"]:
        raise AssertionError("request bytes do not conserve corrected total")
    hop_bound = TOKEN_COUNT * TOP_K * LAYER_COUNT * 2
    corrected_hops = BYTE_CELLS[8]["corrected_total"] // VECTOR_BYTES
    legacy_hops = BYTE_CELLS[8]["legacy_total"] // VECTOR_BYTES
    if not corrected_hops <= hop_bound < legacy_hops:
        raise AssertionError("hop bound does not discriminate the defect")
    total_ratios = []
    peak_ratios = []
    for world in EP_WORLDS:
        cell = BYTE_CELLS[world]
        total_ratios.append(cell["legacy_total"] / cell["corrected_total"])
        peak_ratios.append(cell["legacy_peak"] / cell["corrected_peak"])
    if not total_ratios[0] < total_ratios[1] < total_ratios[2]:
        raise AssertionError("total-byte ratios must increase with EP width")
    if not all(1.9 < ratio < 2.3 for ratio in peak_ratios):
        raise AssertionError("peak-rank ratios lost the frozen factor-two shape")
    if not peak_ratios[-1] < total_ratios[-1] / 3:
        raise AssertionError("critical-rank and population responses are conflated")
    for world in EP_WORLDS:
        corrected_bytes = BYTE_CELLS[world]["corrected_total"]
        for bandwidth in BANDWIDTHS_BPS:
            expected = (
                COMPUTE_PS
                + PHASE_COUNT * PROPAGATION_PS
                + corrected_bytes * 8 * 10**12 // bandwidth
            )
            if CORRECTED_FLUID_JCT_PS[(world, bandwidth)] != expected:
                raise AssertionError("fluid JCT arithmetic drifted")
    floor_ps = BYTE_CELLS[8]["corrected_peak"] * 8 * 10**12 // BANDWIDTHS_BPS[1]
    if floor_ps != 255_631_360:
        raise AssertionError("serialization floor drifted")
    if CORRECTED_FLUID_JCT_PS[(8, BANDWIDTHS_BPS[1])] <= floor_ps:
        raise AssertionError("fluid JCT must remain above serialization floor")
    if PUBLISHED_FLUID_JCT_PS <= CORRECTED_FLUID_JCT_PS[(8, BANDWIDTHS_BPS[1])]:
        raise AssertionError("published-to-corrected direction drifted")


def check_only(args: argparse.Namespace) -> None:
    _check_frozen_registry()
    print(
        f"check-only source_root={args.source_root} out={args.out}; "
        "validated frozen literals and produced no artifacts"
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_revision(revision: str) -> str:
    return subprocess.run(
        ("git", "rev-parse", revision),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _file_observation(path: Path) -> dict[str, Any]:
    return {
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _validate_source_artifacts(source_root: Path) -> dict[str, Any]:
    observations = {}
    for name, (relative, expected_hash) in SOURCE_ARTIFACTS.items():
        path = source_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"required source artifact is missing: {relative}")
        observed = _file_observation(path)
        observed["relative_path"] = relative
        observed["matches_frozen_sha256"] = observed["sha256"] == expected_hash
        if not observed["matches_frozen_sha256"]:
            raise AssertionError(f"source artifact changed: {relative}")
        observations[name] = observed
    return observations


def _dims(world: int) -> Any:
    from simllm.compute import ModelDims

    return ModelDims(
        num_layers=LAYER_COUNT,
        hidden_size=1_024,
        intermediate_size=512,
        num_heads=16,
        num_kv_heads=8,
        head_size=64,
        vocab_size=49_152,
        dtype_bytes=2,
        num_experts=32,
        top_k=TOP_K,
        moe_intermediate_size=512,
        local_num_experts=32 // world,
    )


def _supply(routed: Any, record: Any, world: int) -> Any:
    from simllm.traffic import ExpertPlacementSnapshot, RoutedMoeSupply

    arguments: dict[str, Any] = {
        "routed_experts": routed,
        "placements": (
            ExpertPlacementSnapshot(
                placement_epoch=0,
                expert_owners=tuple(
                    (layer, expert, expert % world)
                    for layer in range(LAYER_COUNT)
                    for expert in range(32)
                ),
            ),
        ),
        "step_placement_epochs": ((record.step_index, 0),),
    }
    if "engine_rank" in RoutedMoeSupply.__dataclass_fields__:
        arguments["engine_rank"] = 0
    return RoutedMoeSupply(**arguments)


def _scheduled_tokens(record: Any, routed: Any) -> tuple[tuple[str, Any], ...]:
    selected = []
    for scheduled in record.scheduled:
        request = routed.by_request_id(scheduled.request_id)
        if scheduled.phase.value == "prefill":
            end = scheduled.context_length
            start = end - scheduled.num_new_tokens
            candidates = request.prefill_tokens
        else:
            start = (
                scheduled.context_length
                - scheduled.num_new_tokens
                - request.prompt_token_count
            )
            end = start + scheduled.num_new_tokens
            candidates = request.decode_tokens
        if start < 0 or end > len(candidates) or start > end:
            raise AssertionError("independent scheduled token slice is invalid")
        selected.extend((scheduled.request_id, token) for token in candidates[start:end])
    if len(selected) != record.total_new_tokens:
        raise AssertionError("independent scheduled token count disagrees with record")
    return tuple(selected)


def _independent_projection(record: Any, routed: Any, world: int) -> dict[str, Any]:
    tokens = _scheduled_tokens(record, routed)
    total_bytes = 0
    request_bytes: dict[str, int] = defaultdict(int)
    layer_hops = []
    dispatch_rows = []
    combine_rows = []
    for layer in range(LAYER_COUNT):
        dispatch: dict[tuple[int, int], int] = defaultdict(int)
        request_dispatch: dict[tuple[str, int, int], int] = defaultdict(int)
        hops = 0
        for request_id, token in tokens:
            destinations = {
                expert % world for expert in token.layers[layer].expert_ids
            }
            for destination in destinations:
                if destination == 0:
                    continue
                dispatch[(0, destination)] += VECTOR_BYTES
                request_dispatch[(request_id, 0, destination)] += VECTOR_BYTES
                request_bytes[request_id] += 2 * VECTOR_BYTES
                total_bytes += 2 * VECTOR_BYTES
                hops += 2
        layer_hops.append(hops)
        dispatch_rows.append(
            tuple(
                (source, destination, size)
                for (source, destination), size in sorted(dispatch.items())
            )
        )
        combine_rows.append(
            tuple(
                (destination, source, size)
                for (source, destination), size in sorted(dispatch.items())
            )
        )
    return {
        "total_bytes": total_bytes,
        "hops": total_bytes // VECTOR_BYTES,
        "layer_hops": layer_hops,
        "request_bytes": dict(sorted(request_bytes.items())),
        "dispatch_rows": dispatch_rows,
        "combine_rows": combine_rows,
    }


def _physical_rows(operation: Any) -> tuple[tuple[int, int, int], ...]:
    if operation.pair_payload_bytes:
        return operation.pair_payload_bytes
    return tuple(
        (source, destination, operation.per_pair_bytes)
        for source in operation.ranks
        for destination in operation.ranks
        if source != destination
    )


def _renderer_observation(record: Any, dims: Any, supply: Any, world: int) -> dict[str, Any]:
    from simllm.traffic import step_moe_alltoalls, step_tp_allreduces

    operations = step_moe_alltoalls(
        record,
        dims,
        tuple(range(world)),
        routed_supply=supply,
    )
    egress: dict[int, int] = defaultdict(int)
    request_bytes: dict[str, int] = defaultdict(int)
    layer_hops: dict[int, int] = defaultdict(int)
    dispatch_sources = set()
    total_bytes = 0
    positive_rows = 0
    for operation in operations:
        rows = _physical_rows(operation)
        total_bytes += sum(size for _, _, size in rows)
        positive_rows += len(rows)
        for source, _, size in rows:
            egress[source] += size
        for request_id, source, _, size in operation.request_pair_payload_bytes:
            request_bytes[request_id] += size
            if operation.phase == "dispatch":
                dispatch_sources.add(source)
        layer_hops[operation.layer] += sum(size for _, _, size in rows) // VECTOR_BYTES
    tp_operations = step_tp_allreduces(record, dims, (0, 1))
    uniform_operations = step_moe_alltoalls(record, dims, tuple(range(world)))
    uniform_total = sum(
        size
        for operation in uniform_operations
        for _, _, size in _physical_rows(operation)
    )
    uniform_sources = sorted(
        {
            source
            for operation in uniform_operations
            if operation.phase == "dispatch"
            for source, _, _ in _physical_rows(operation)
        }
    )
    return {
        "operation_count": len(operations),
        "positive_rows": positive_rows,
        "total_bytes": total_bytes,
        "hops": total_bytes // VECTOR_BYTES,
        "layer_hops": [layer_hops[layer] for layer in range(LAYER_COUNT)],
        "egress_bytes": {str(rank): egress[rank] for rank in sorted(egress)},
        "peak_rank_egress_bytes": max(egress.values(), default=0),
        "request_bytes": dict(sorted(request_bytes.items())),
        "dispatch_sources": sorted(dispatch_sources),
        "tp_payload_bytes": sorted(
            {operation.payload_bytes for operation in tp_operations}
        ),
        "uniform_total_bytes": uniform_total,
        "uniform_dispatch_sources": uniform_sources,
    }


def _run_goal(
    goal_path: Path,
    *,
    profile: str,
    bandwidth_bps: int,
) -> dict[str, Any]:
    from simllm.backends import HtsimRnicConfig, run_htsim_rnic
    from simllm.goal import to_binary

    goal_bin = to_binary(goal_path)
    completion_csv = goal_path.with_suffix(f".{profile}.{bandwidth_bps}.csv")
    result = run_htsim_rnic(
        HtsimRnicConfig(
            goal_bin=goal_bin,
            profile=profile,
            linkspeed_bps=bandwidth_bps,
            completion_csv=completion_csv,
        )
    )
    return {
        "jct_ps": result.job_completion_time_ps(),
        "flow_count": len(result.flows),
        "quiescent": result.quiescent,
        "goal": _file_observation(goal_path),
        "goal_bin": _file_observation(goal_bin),
        "completion_csv": _file_observation(completion_csv),
    }


def _run_direct_native(
    out: Path,
    record: Any,
    routed: Any,
    *,
    profile: str,
    bandwidth_bps: int,
    world: int,
) -> dict[str, Any]:
    from simllm.traffic import render_step_goal

    trace = render_step_goal(
        record,
        _dims(world),
        (0,),
        COMPUTE_PS // (LAYER_COUNT * 1_000),
        ep_ranks=tuple(range(world)),
        routed_supply=_supply(routed, record, world),
        num_goal_ranks=world,
    )
    cell = out / "direct" / f"w{world}-{bandwidth_bps}-{profile}"
    cell.mkdir(parents=True, exist_ok=False)
    return _run_goal(
        trace.write(cell / "step-000000.goal"),
        profile=profile,
        bandwidth_bps=bandwidth_bps,
    )


def _fixed_provider() -> Any:
    from simllm.compute import ComputeProvider, DurationEstimate

    class FixedProvider(ComputeProvider):
        def estimate(self, kernel: Any, gpu: Any) -> Any:
            return DurationEstimate(duration_ps=COMPUTE_PS, bound="declared-fixed")

    return FixedProvider()


def _run_live_sink(
    out: Path,
    record: Any,
    routed: Any,
    *,
    profile: str,
    bandwidth_bps: int,
) -> dict[str, Any]:
    from simllm.backends import HtsimStepSink, HtsimStepSinkConfig

    world = 8
    cell = out / "live" / f"w{world}-{bandwidth_bps}-{profile}"
    sink = HtsimStepSink(
        HtsimStepSinkConfig(
            profile=profile,
            tp_ranks=(0,),
            dims=_dims(world),
            workdir=cell,
            ep_ranks=tuple(range(world)),
            linkspeed_bps=bandwidth_bps,
            provider=_fixed_provider(),
            routed_moe_supply=_supply(routed, record, world),
            num_goal_ranks=world,
        )
    )
    result = sink(record)
    if result is None or len(sink.outcomes) != 1 or len(sink.locality_outcomes) != 1:
        raise AssertionError("live sink produced no unique result")
    network = sink.outcomes[0]
    locality = sink.locality_outcomes[0]
    return {
        "step_result_jct_ps": result.step_latency_ps,
        "completed_at_ps": result.completed_at_ps,
        "routing_mode": network.routing_mode,
        "flow_count": network.num_flows,
        "quiescent": network.quiescent,
        "total_directed_bytes": locality.total_directed_bytes,
        "fabric_directed_bytes": locality.fabric_directed_bytes,
        "backend_runs": locality.backend_runs,
        "graph_artifact_count": locality.graph_artifact_count,
    }


def _native_provenance() -> dict[str, Any]:
    from simllm.backends import find_htsim_rnic
    from simllm.goal import find_txt2bin

    binaries = {
        "htsim_rnic": find_htsim_rnic(),
        "txt2bin": find_txt2bin(),
    }
    result = {}
    for name, path in binaries.items():
        if path is None:
            raise FileNotFoundError(f"required native binary is unavailable: {name}")
        result[name] = _file_observation(path)
    return result


def _collect_snapshot(
    out: Path,
    source_root: Path,
    *,
    include_live: bool,
    include_archived: bool,
) -> dict[str, Any]:
    from simllm.core import step_records_from_jsonl
    from simllm.preplay import read_routed_experts

    if out.exists():
        raise FileExistsError(f"refusing to overwrite snapshot directory: {out}")
    out.mkdir(parents=True)
    source_observations = _validate_source_artifacts(source_root)
    records = step_records_from_jsonl(source_root / SOURCE_ARTIFACTS["steps"][0])
    if not records or records[0].total_new_tokens != TOKEN_COUNT:
        raise AssertionError("Granite step 0 token population changed")
    record = records[0]
    routed = read_routed_experts(source_root / SOURCE_ARTIFACTS["routing"][0])
    renderer = {}
    independent = {}
    for world in EP_WORLDS:
        dims = _dims(world)
        supply = _supply(routed, record, world)
        renderer[str(world)] = _renderer_observation(record, dims, supply, world)
        independent[str(world)] = _independent_projection(record, routed, world)
    direct_native = {}
    for world in EP_WORLDS:
        for bandwidth in BANDWIDTHS_BPS:
            key = f"w{world}:{bandwidth}:rnic-nn-fluid"
            direct_native[key] = _run_direct_native(
                out,
                record,
                routed,
                profile="rnic-nn-fluid",
                bandwidth_bps=bandwidth,
                world=world,
            )
    direct_native["w8:400000000000:rnic-nn"] = _run_direct_native(
        out,
        record,
        routed,
        profile="rnic-nn",
        bandwidth_bps=400_000_000_000,
        world=8,
    )
    archived_native = {}
    if include_archived:
        archived_goal = source_root / SOURCE_ARTIFACTS["archived_goal"][0]
        for profile in PROFILES:
            archived_dir = out / "archived" / profile
            archived_dir.mkdir(parents=True, exist_ok=False)
            copied_goal = archived_dir / "step-000000.goal"
            copied_goal.write_bytes(archived_goal.read_bytes())
            archived_native[profile] = _run_goal(
                copied_goal,
                profile=profile,
                bandwidth_bps=400_000_000_000,
            )
    live = {}
    if include_live:
        for profile, bandwidth in (
            ("rnic-nn-fluid", 200_000_000_000),
            ("rnic-nn-fluid", 400_000_000_000),
            ("rnic-nn", 400_000_000_000),
        ):
            live[f"{bandwidth}:{profile}"] = _run_live_sink(
                out,
                record,
                routed,
                profile=profile,
                bandwidth_bps=bandwidth,
            )
    return {
        "provenance": {
            "expectations_commit": EXPECTATIONS_COMMIT,
            "evidence_authored_against": EVIDENCE_AUTHORED_AGAINST,
            "observed_simllm_revision": _git_revision("HEAD"),
            "observed_htsim_gitlink": _git_revision("HEAD:third_party/htsim"),
            "python": sys.version,
            "platform": platform.platform(),
            "native": _native_provenance(),
            "source_artifacts": source_observations,
        },
        "configuration": {
            "engine_rank": 0,
            "ep_worlds": list(EP_WORLDS),
            "bandwidths_bps": list(BANDWIDTHS_BPS),
            "profiles": list(PROFILES),
            "compute_ps": COMPUTE_PS,
            "token_population": "one-engine-step",
            "peer_scheduled_tokens": 0,
        },
        "renderer": renderer,
        "independent_projection": independent,
        "direct_native": direct_native,
        "archived_native": archived_native,
        "live_step_result": live,
    }


def _inside(value: float, band: tuple[float, float]) -> bool:
    return band[0] <= value <= band[1]


def _score_behavioral(legacy: dict[str, Any], corrected: dict[str, Any]) -> dict[str, Any]:
    population_rows = []
    peak_rows = []
    for world in EP_WORLDS:
        old = legacy["renderer"][str(world)]
        new = corrected["renderer"][str(world)]
        total_ratio = old["total_bytes"] / new["total_bytes"]
        peak_ratio = old["peak_rank_egress_bytes"] / new["peak_rank_egress_bytes"]
        population_rows.append(
            {
                "ep_world": world,
                "legacy_total_bytes": old["total_bytes"],
                "corrected_total_bytes": new["total_bytes"],
                "ratio": total_ratio,
                "passed": _inside(total_ratio, TOTAL_RATIO_BANDS[world]),
            }
        )
        peak_rows.append(
            {
                "ep_world": world,
                "legacy_peak_bytes": old["peak_rank_egress_bytes"],
                "corrected_peak_bytes": new["peak_rank_egress_bytes"],
                "ratio": peak_ratio,
                "passed": _inside(peak_ratio, PEAK_RATIO_BANDS[world]),
            }
        )
    population_pass = all(row["passed"] for row in population_rows) and all(
        population_rows[index]["ratio"] < population_rows[index + 1]["ratio"]
        for index in range(len(population_rows) - 1)
    )
    peak_pass = all(row["passed"] for row in peak_rows) and (
        peak_rows[-1]["ratio"] < population_rows[-1]["ratio"] / 3
    )
    native_rows = []
    for bandwidth in BANDWIDTHS_BPS:
        key = f"{bandwidth}:rnic-nn-fluid"
        live = corrected["live_step_result"][key]
        observed = live["step_result_jct_ps"]
        expected = CORRECTED_FLUID_JCT_PS[(8, bandwidth)]
        flow_count = live["flow_count"]
        passed = expected <= observed <= expected + flow_count
        if bandwidth == 400_000_000_000:
            published_ratio = observed / PUBLISHED_FLUID_JCT_PS
            passed = passed and _inside(published_ratio, (0.69, 0.85))
        else:
            published_ratio = None
        native_rows.append(
            {
                "profile": "rnic-nn-fluid",
                "bandwidth_bps": bandwidth,
                "observed_jct_ps": observed,
                "expected_jct_ps": expected,
                "flow_count": flow_count,
                "ratio_to_published_400g": published_ratio,
                "passed": passed,
            }
        )
    packet = corrected["live_step_result"]["400000000000:rnic-nn"]
    archived_packet = legacy["archived_native"]["rnic-nn"]
    fluid_400 = corrected["live_step_result"]["400000000000:rnic-nn-fluid"]
    packet_ratio = packet["step_result_jct_ps"] / archived_packet["jct_ps"]
    packet_pass = (
        _inside(packet_ratio, (0.60, 0.85))
        and fluid_400["step_result_jct_ps"]
        <= packet["step_result_jct_ps"]
        <= 850_000_000
    )
    native_rows.append(
        {
            "profile": "rnic-nn",
            "bandwidth_bps": 400_000_000_000,
            "observed_jct_ps": packet["step_result_jct_ps"],
            "archived_defective_jct_ps": archived_packet["jct_ps"],
            "ratio_to_archived": packet_ratio,
            "fluid_floor_jct_ps": fluid_400["step_result_jct_ps"],
            "passed": packet_pass,
        }
    )
    return {
        "TRAF-B1": {
            "classification": "scored-genuine-risk",
            "instances": population_rows,
            "passed": population_pass,
        },
        "TRAF-B2": {
            "classification": "scored-genuine-risk",
            "instances": peak_rows,
            "passed": peak_pass,
        },
        "TRAF-B3": {
            "classification": "scored-genuine-risk",
            "instances": native_rows,
            "passed": all(row["passed"] for row in native_rows),
        },
    }


def _fatal_checks(corrected: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for world in EP_WORLDS:
        observed = corrected["renderer"][str(world)]
        independent = corrected["independent_projection"][str(world)]
        frozen = BYTE_CELLS[world]
        expected_request = REQUEST_BYTES_W8 if world == 8 else independent["request_bytes"]
        uniform_per_pair = TOKEN_COUNT * TOP_K * VECTOR_BYTES // world
        expected_uniform_total = PHASE_COUNT * (world - 1) * uniform_per_pair
        passed = (
            observed["total_bytes"] == frozen["corrected_total"]
            and observed["peak_rank_egress_bytes"] == frozen["corrected_peak"]
            and observed["total_bytes"] == independent["total_bytes"]
            and observed["hops"] == independent["hops"]
            and observed["layer_hops"] == independent["layer_hops"]
            and observed["request_bytes"] == expected_request
            and observed["dispatch_sources"] == [0]
            and observed["uniform_dispatch_sources"] == [0]
            and observed["uniform_total_bytes"] == expected_uniform_total
            and observed["tp_payload_bytes"] == [TOKEN_COUNT * VECTOR_BYTES]
            and observed["hops"] <= TOKEN_COUNT * TOP_K * LAYER_COUNT * 2
        )
        rows.append(
            {
                "ep_world": world,
                "observed_total_bytes": observed["total_bytes"],
                "independent_total_bytes": independent["total_bytes"],
                "observed_peak_bytes": observed["peak_rank_egress_bytes"],
                "observed_hops": observed["hops"],
                "hop_bound": TOKEN_COUNT * TOP_K * LAYER_COUNT * 2,
                "request_bytes": observed["request_bytes"],
                "dispatch_sources": observed["dispatch_sources"],
                "uniform_dispatch_sources": observed["uniform_dispatch_sources"],
                "uniform_total_bytes": observed["uniform_total_bytes"],
                "tp_payload_bytes": observed["tp_payload_bytes"],
                "passed": passed,
            }
        )
    live_rows = []
    for key, observation in corrected["live_step_result"].items():
        bandwidth, profile = key.split(":", 1)
        direct = corrected["direct_native"][f"w8:{bandwidth}:{profile}"]
        floor_ps = BYTE_CELLS[8]["corrected_peak"] * 8 * 10**12 // int(bandwidth)
        passed = (
            observation["routing_mode"] == "captured"
            and observation["quiescent"]
            and observation["total_directed_bytes"] == BYTE_CELLS[8]["corrected_total"]
            and observation["fabric_directed_bytes"] == BYTE_CELLS[8]["corrected_total"]
            and observation["step_result_jct_ps"] >= floor_ps
            and direct["quiescent"]
        )
        live_rows.append(
            {
                "profile": profile,
                "bandwidth_bps": int(bandwidth),
                "step_result_jct_ps": observation["step_result_jct_ps"],
                "direct_jct_ps": direct["jct_ps"],
                "serialization_floor_ps": floor_ps,
                "passed": passed,
            }
        )
    return {
        "renderer_rows": rows,
        "live_step_result_rows": live_rows,
        "passed": all(row["passed"] for row in rows)
        and all(row["passed"] for row in live_rows),
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def capture_legacy(args: argparse.Namespace) -> None:
    legacy = _collect_snapshot(
        args.out / "legacy",
        args.source_root,
        include_live=False,
        include_archived=True,
    )
    for world in EP_WORLDS:
        observed = legacy["renderer"][str(world)]
        frozen = BYTE_CELLS[world]
        if (
            observed["total_bytes"] != frozen["legacy_total"]
            or observed["peak_rank_egress_bytes"] != frozen["legacy_peak"]
        ):
            raise AssertionError("legacy renderer disagrees with frozen defect census")
    if legacy["archived_native"]["rnic-nn-fluid"]["jct_ps"] != PUBLISHED_FLUID_JCT_PS:
        raise AssertionError("archived fluid GOAL no longer reproduces published JCT")
    _write_json(args.out / "legacy" / "summary.json", legacy)
    print(json.dumps({"legacy_snapshot": "recorded"}, sort_keys=True))


def run_final(args: argparse.Namespace) -> None:
    legacy_path = args.out / "legacy" / "summary.json"
    if not legacy_path.is_file():
        raise FileNotFoundError(
            "legacy summary is missing; run the frozen harness with --capture-legacy "
            "before implementing TRAF-25"
        )
    legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
    corrected = _collect_snapshot(
        args.out / "corrected",
        args.source_root,
        include_live=True,
        include_archived=False,
    )
    behavioral = _score_behavioral(legacy, corrected)
    fatal = _fatal_checks(corrected)
    score = {
        "passed_families": sum(family["passed"] for family in behavioral.values()),
        "total_families": len(behavioral),
        "passed_instances": sum(
            sum(instance["passed"] for instance in family["instances"])
            for family in behavioral.values()
        ),
        "total_instances": sum(
            len(family["instances"]) for family in behavioral.values()
        ),
    }
    summary = {
        "schema": "simllm-token-ownership-study-v1",
        "legacy": legacy,
        "corrected": corrected,
        "behavioral": behavioral,
        "behavioral_score": score,
        "fatal_unscored": fatal,
        "entailment": (
            "TRAF-B1 through TRAF-B3 were evaluated from raw renderer totals, "
            "raw peak egress and native JCTs before exact byte, request, hop, "
            "source, fluid-point, quiescence or provenance checks."
        ),
    }
    _write_json(args.out / "summary.json", summary)
    print(json.dumps(score, sort_keys=True))
    if not all(family["passed"] for family in behavioral.values()) or not fatal["passed"]:
        raise AssertionError("token ownership study failed its frozen acceptance bar")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--capture-legacy", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.check_only:
        check_only(args)
        return
    if args.capture_legacy:
        capture_legacy(args)
        return
    run_final(args)


if __name__ == "__main__":
    main()
