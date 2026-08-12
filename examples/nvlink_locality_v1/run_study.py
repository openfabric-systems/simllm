"""Run the frozen TRAF-10 NVLink locality study after implementation lands."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

TRACE_SHA256 = "36334f3aaa767c46d5f9c8498e02f6c2805a46e5000a57aea2747e17dd5d1341"
NVLINK_BYTES_PER_SECOND = 450_000_000_000
VECTOR_BYTES = (1_024, 2_048)
PLACEMENTS = {
    "AAAA": ("a", "a", "a", "a"),
    "AABB": ("a", "a", "b", "b"),
    "ABCD": ("a", "b", "c", "d"),
}
#: CORE-42 requalification: the all-local services are the maximum endpoint
#: load charged by CORE-41, not the superseded 4,538,000 ps and 9,047,000 ps
#: maximum source egress. Every AABB and ABCD row is unchanged.
EXPECTED_CELLS = {
    (1_024, "AAAA"): (2_983_936, 0, 2_983_936, 6_652_000),
    (1_024, "AABB"): (2_983_936, 2_011_136, 972_800, 2_194_000),
    (1_024, "ABCD"): (2_983_936, 2_983_936, 0, 0),
    (2_048, "AAAA"): (5_967_872, 0, 5_967_872, 13_286_000),
    (2_048, "AABB"): (5_967_872, 4_022_272, 1_945_600, 4_358_000),
    (2_048, "ABCD"): (5_967_872, 5_967_872, 0, 0),
}
#: whole-nanosecond quantization window allowed over the exact serialization
#: floor of an all-local cell: one nanosecond per serial phase
NVLINK_QUANTIZATION_WINDOW_PS = 48_000
#: exact serialization floors of the all-local cells, in picoseconds
NVLINK_SERVICE_FLOORS_PS = {1_024: 6_630_969, 2_048: 13_261_938}
DIRECT_GOAL_ORACLES = {
    1_024: (
        20_392,
        "917961edf996753223857d64010fc61e4f6b08672f18dcadf42c70d60ee36c4a",
    ),
    2_048: (
        20_392,
        "16ee686eda4634886b117788b3893c893f5e12ea819736e0afdbdf63bab0e826",
    ),
}
EXPECTED_JCT_BANDS = {
    (1_024, "AAAA"): (6_676_000, 6_676_000),
    (1_024, "AABB"): (136_246_720, 136_246_720),
    (1_024, "ABCD"): (155_702_720, 155_702_864),
    (2_048, "AAAA"): (13_310_000, 13_310_000),
    (2_048, "AABB"): (176_469_440, 176_469_440),
    (2_048, "ABCD"): (215_381_440, 215_381_584),
}
#: fixed compute estimate carried by every cell, in picoseconds
COMPUTE_ESTIMATE_PS = 24_000
EXPECTED_PHASES = 48
EXPECTED_POSITIVE_PAIRS = 144
EXPECTATIONS_COMMIT = "dd1eefebc84091c547d8cad10225b21ab85a7706"
REQUALIFICATION_EXPECTATIONS_COMMIT = "a455bc4581b79fcd8d3c0021a50e449276afb477"
EVIDENCE_AUTHORED_AGAINST = "6973bd0e3ed6091c403c7055ee01c2d8ae0ae970"


def _check_frozen_registry() -> None:
    if len(TRACE_SHA256) != 64:
        raise AssertionError("trace SHA-256 must contain 64 hexadecimal digits")
    if NVLINK_BYTES_PER_SECOND <= 0:
        raise AssertionError("NVLink rate must be positive")
    expected_keys = {
        (vector_bytes, placement)
        for vector_bytes in VECTOR_BYTES
        for placement in PLACEMENTS
    }
    if set(EXPECTED_CELLS) != expected_keys:
        raise AssertionError("expected-cell grid is incomplete")
    if set(EXPECTED_JCT_BANDS) != expected_keys:
        raise AssertionError("JCT-band grid is incomplete")
    for key, (total, fabric, local, service_ps) in EXPECTED_CELLS.items():
        if min(total, fabric, local, service_ps) < 0:
            raise AssertionError(f"cell {key} contains a negative literal")
        if fabric + local != total:
            raise AssertionError(f"cell {key} does not conserve directed bytes")
    for vector_bytes in VECTOR_BYTES:
        one = EXPECTED_CELLS[(vector_bytes, "AAAA")]
        two = EXPECTED_CELLS[(vector_bytes, "AABB")]
        remote = EXPECTED_CELLS[(vector_bytes, "ABCD")]
        if not (one[1] < two[1] < remote[1]):
            raise AssertionError("fabric-byte direction is not strictly increasing")
        if not (one[2] > two[2] > remote[2]):
            raise AssertionError("NVLink-byte direction is not strictly decreasing")
        low, high = EXPECTED_JCT_BANDS[(vector_bytes, "ABCD")]
        if high - low != EXPECTED_POSITIVE_PAIRS:
            raise AssertionError("all-remote quantization band drifted")
        ordered = [EXPECTED_JCT_BANDS[(vector_bytes, name)][0] for name in PLACEMENTS]
        if not ordered[0] < ordered[1] < ordered[2]:
            raise AssertionError("JCT direction is not strictly increasing")
    if EXPECTED_CELLS[(2_048, "ABCD")][0] != 2 * EXPECTED_CELLS[(1_024, "ABCD")][0]:
        raise AssertionError("payload sweep no longer doubles directed bytes")
    for vector_bytes, floor_ps in NVLINK_SERVICE_FLOORS_PS.items():
        service_ps = EXPECTED_CELLS[(vector_bytes, "AAAA")][3]
        if not floor_ps <= service_ps <= floor_ps + NVLINK_QUANTIZATION_WINDOW_PS:
            raise AssertionError(
                "registered all-local service leaves its serialization window"
            )
        jct_low, jct_high = EXPECTED_JCT_BANDS[(vector_bytes, "AAAA")]
        if jct_low != jct_high or jct_low != service_ps + COMPUTE_ESTIMATE_PS:
            raise AssertionError(
                "registered all-local JCT is not service plus the fixed compute term"
            )
    doubling_slack_ps = (
        2 * EXPECTED_CELLS[(1_024, "AAAA")][3] - EXPECTED_CELLS[(2_048, "AAAA")][3]
    )
    if not 0 <= doubling_slack_ps <= NVLINK_QUANTIZATION_WINDOW_PS:
        raise AssertionError("registered payload doubling leaves its quantization window")
    if any(size <= 0 or len(digest) != 64 for size, digest in DIRECT_GOAL_ORACLES.values()):
        raise AssertionError("direct GOAL oracle is malformed")
    if EXPECTED_PHASES <= 0 or EXPECTED_POSITIVE_PAIRS <= 0:
        raise AssertionError("phase and pair counts must be positive")


def check_only(args: argparse.Namespace) -> None:
    _check_frozen_registry()
    print(
        f"check-only out={args.out}; validated the frozen TRAF-10 registry and "
        "its CORE-42 requalification literals, and produced no artifacts"
    )


def _git_revision(*args: str) -> str:
    return subprocess.run(
        ("git", "rev-parse", *args),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _trace_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "preplay_trace_v1/granite_length_cap.jsonl"
    )


def _routed_projection(trace_path: Path):
    from simllm.core import RequestBookkeeper
    from simllm.preplay import (
        RequestArrival,
        join_preplay_arrivals,
        project_preplay_routing,
    )

    joined = join_preplay_arrivals(
        (RequestArrival(request_id="length-cap", arrived_at_ps=0),),
        trace_path,
        RequestBookkeeper(),
    )
    return project_preplay_routing(joined)


def _expert_manifest():
    from simllm.placement import PlacementManifest, RankPlacement

    return PlacementManifest(
        ranks=[
            RankPlacement(
                global_rank=rank,
                hostname="expert-owner",
                local_rank=rank,
                local_expert_ids={
                    layer: list(range(rank * 8, (rank + 1) * 8))
                    for layer in range(24)
                },
                placement_epoch=0,
            )
            for rank in range(4)
        ]
    )


def _routed_supply(projection):
    from simllm.traffic import ExpertPlacementSnapshot, RoutedMoeSupply

    return RoutedMoeSupply(
        engine_rank=0,
        routed_experts=projection,
        placements=(
            ExpertPlacementSnapshot.from_manifest(
                _expert_manifest(),
                (0, 1, 2, 3),
            ),
        ),
        step_placement_epochs=((0, 0), (1, 0), (2, 0)),
    )


def _dims(vector_bytes: int):
    from simllm.compute import ModelDims

    return ModelDims(
        num_layers=24,
        hidden_size=vector_bytes // 2,
        intermediate_size=512,
        num_heads=8,
        num_kv_heads=4,
        head_size=64,
        vocab_size=49_152,
        dtype_bytes=2,
        num_experts=32,
        top_k=8,
        moe_intermediate_size=512,
        local_num_experts=8,
    )


def _physical_manifest(hosts: tuple[str, ...]):
    from simllm.placement import PlacementManifest, RankPlacement

    next_local_rank: dict[str, int] = {}
    ranks = []
    for global_rank, hostname in enumerate(hosts):
        local_rank = next_local_rank.get(hostname, 0)
        next_local_rank[hostname] = local_rank + 1
        ranks.append(
            RankPlacement(
                global_rank=global_rank,
                hostname=hostname,
                local_rank=local_rank,
            )
        )
    return PlacementManifest(ranks=ranks)


def _record(step_index: int, virtual_time_ps: int):
    from simllm.core import RequestPhase, ScheduledRequest, StepRecord

    return StepRecord(
        step_index=step_index,
        virtual_time_ps=virtual_time_ps,
        scheduled=[
            ScheduledRequest(
                "length-cap",
                RequestPhase.PREFILL,
                22,
                context_length=22,
            )
        ],
        num_sampled=1,
    )


def _run_cell(
    out: Path,
    *,
    vector_bytes: int,
    placement_name: str,
    hosts: tuple[str, ...] | None,
    supply,
) -> tuple[dict[str, object], object, object]:
    from simllm.backends import HtsimStepSink, HtsimStepSinkConfig
    from simllm.compute import ComputeProvider, DurationEstimate

    class FixedProvider(ComputeProvider):
        def estimate(self, kernel, gpu):
            return DurationEstimate(
                duration_ps=COMPUTE_ESTIMATE_PS,
                bound="declared-fixed",
            )

    workdir = out / f"vector-{vector_bytes}" / placement_name
    sink = HtsimStepSink(
        HtsimStepSinkConfig(
            profile="rnic-nn-fluid",
            tp_ranks=(0,),
            dims=_dims(vector_bytes),
            workdir=workdir,
            ep_ranks=(0, 1, 2, 3),
            linkspeed_bps=400_000_000_000,
            provider=FixedProvider(),
            routed_moe_supply=supply,
            placement_manifest=(
                _physical_manifest(hosts) if hosts is not None else None
            ),
            nvlink_bandwidth_bytes_per_second=NVLINK_BYTES_PER_SECOND,
            num_goal_ranks=4,
        )
    )
    virtual_time_ps = 0
    results = []
    for step_index in range(3):
        result = sink(_record(step_index, virtual_time_ps))
        results.append(result)
        if result is not None:
            virtual_time_ps = result.completed_at_ps

    complete_results = tuple(result for result in results if result is not None)
    complete_outcomes = tuple(sink.outcomes)
    locality_outcomes = tuple(sink.locality_outcomes)
    latencies = tuple(result.step_latency_ps for result in complete_results)
    tpot_ps = (
        sum(latencies[1:]) // len(latencies[1:])
        if len(latencies) > 1
        else None
    )
    cell = {
        "vector_bytes": vector_bytes,
        "placement": placement_name,
        "step_results_reached": len(complete_results),
        "step_outcomes_reached": len(complete_outcomes),
        "locality_outcomes_reached": len(locality_outcomes),
        "jct_ps": latencies[0] if latencies else None,
        "ttft_ps": latencies[0] if latencies else None,
        "tpot_ps": tpot_ps,
        "replay_latencies_ps": list(latencies),
        "locality": (
            asdict(locality_outcomes[0]) if locality_outcomes else None
        ),
        "network": asdict(complete_outcomes[0]) if complete_outcomes else None,
    }
    return cell, sink, complete_results


def _raw_behavioral(cells: dict[tuple[int, str], dict[str, object]]):
    b1_instances = []
    for vector_bytes in VECTOR_BYTES:
        localities = [
            cells[(vector_bytes, name)]["locality"] for name in PLACEMENTS
        ]
        fabric = [row["fabric_directed_bytes"] for row in localities]
        nvlink = [row["nvlink_directed_bytes"] for row in localities]
        b1_instances.append(
            {
                "vector_bytes": vector_bytes,
                "raw_fabric_bytes": fabric,
                "raw_nvlink_bytes": nvlink,
                "passed": fabric[0] < fabric[1] < fabric[2]
                and nvlink[0] > nvlink[1] > nvlink[2],
            }
        )

    b2_instances = []
    for vector_bytes in VECTOR_BYTES:
        for placement in ("AAAA", "AABB"):
            observed = cells[(vector_bytes, placement)]["locality"][
                "nvlink_service_ps"
            ]
            expected = EXPECTED_CELLS[(vector_bytes, placement)][3]
            b2_instances.append(
                {
                    "vector_bytes": vector_bytes,
                    "placement": placement,
                    "raw_nvlink_service_ps": observed,
                    "expected_nvlink_service_ps": expected,
                    "passed": observed == expected,
                }
            )

    b3_instances = []
    for vector_bytes in VECTOR_BYTES:
        raw = [cells[(vector_bytes, name)]["jct_ps"] for name in PLACEMENTS]
        bands = [EXPECTED_JCT_BANDS[(vector_bytes, name)] for name in PLACEMENTS]
        band_pass = all(
            value is not None and low <= value <= high
            for value, (low, high) in zip(raw, bands, strict=True)
        )
        b3_instances.append(
            {
                "vector_bytes": vector_bytes,
                "raw_jct_ps": raw,
                "expected_bands_ps": [list(band) for band in bands],
                "step_results_reached": [
                    cells[(vector_bytes, name)]["step_results_reached"]
                    for name in PLACEMENTS
                ],
                "passed": band_pass
                and raw[0] < raw[1] < raw[2]
                and all(
                    cells[(vector_bytes, name)]["step_results_reached"] == 3
                    for name in PLACEMENTS
                ),
            }
        )

    return {
        "TRAF-B1": {
            "instances": b1_instances,
            "passed": all(row["passed"] for row in b1_instances),
        },
        "TRAF-B2": {
            "instances": b2_instances,
            "passed": all(row["passed"] for row in b2_instances),
        },
        "TRAF-B3": {
            "instances": b3_instances,
            "passed": all(row["passed"] for row in b3_instances),
        },
    }


def _goal_oracle(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _direct_goal_oracle(vector_bytes: int, supply) -> dict[str, object]:
    from simllm.traffic import render_step_goal

    trace = render_step_goal(
        _record(0, 0),
        _dims(vector_bytes),
        (0,),
        1,
        ep_ranks=(0, 1, 2, 3),
        routed_supply=supply,
        num_goal_ranks=4,
    )
    payload = trace.render().encode()
    return {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _artifact_inventory(workdir: Path, suffix: str) -> dict[str, object]:
    paths = sorted(workdir.glob(f"step-000000.*.{suffix}"))
    digest = hashlib.sha256()
    total_bytes = 0
    for path in paths:
        payload = path.read_bytes()
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(payload)
        total_bytes += len(payload)
    return {
        "files": len(paths),
        "bytes": total_bytes,
        "sha256": digest.hexdigest(),
    }


def _tp_width_guards():
    from simllm.compute import ModelDims
    from simllm.core import RequestPhase, ScheduledRequest, StepRecord
    from simllm.placement import RankMapper
    from simllm.traffic import plan_step_locality

    dims = ModelDims(
        num_layers=1,
        hidden_size=8,
        intermediate_size=16,
        num_heads=2,
        num_kv_heads=2,
        head_size=4,
        vocab_size=32,
        dtype_bytes=2,
    )
    record = StepRecord(
        0,
        0,
        [ScheduledRequest("tp", RequestPhase.DECODE, 1, context_length=2)],
    )
    rows = []
    for width in range(1, 9):
        hosts = tuple("single-node" for _ in range(width))
        plan = plan_step_locality(
            record,
            dims,
            tuple(range(width)),
            rank_mapper=RankMapper(_physical_manifest(hosts)),
        )
        rows.append(
            {
                "tp_width": width,
                "fabric_bytes": plan.fabric_bytes,
                "nvlink_bytes": plan.nvlink_bytes,
                "passed": plan.fabric_bytes == 0,
            }
        )
    return rows


def _phase_partition_guards(vector_bytes: int, supply):
    from collections import Counter

    from simllm.placement import RankMapper
    from simllm.traffic import plan_step_locality

    plan = plan_step_locality(
        _record(0, 0),
        _dims(vector_bytes),
        (0,),
        ep_ranks=(0, 1, 2, 3),
        routed_supply=supply,
        rank_mapper=RankMapper(_physical_manifest(PLACEMENTS["AABB"])),
        nvlink_bandwidth_bytes_per_second=NVLINK_BYTES_PER_SECOND,
    )
    partition_exact = all(
        Counter(phase.phase.segments)
        == Counter((*phase.fabric_segments, *phase.nvlink_segments))
        for phase in plan.phases
    )
    transposes = []
    for layer in range(24):
        dispatch = plan.phases[layer * 2].phase.segments
        combine = plan.phases[layer * 2 + 1].phase.segments
        transposes.append(
            Counter(
                (segment.destination_rank, segment.source_rank, segment.payload_bytes)
                for segment in dispatch
            )
            == Counter(
                (segment.source_rank, segment.destination_rank, segment.payload_bytes)
                for segment in combine
            )
        )
    tags = [phase.phase.segments[0].tag for phase in plan.phases]
    return {
        "phase_count": len(plan.phases),
        "positive_pairs": sum(len(phase.phase.segments) for phase in plan.phases),
        "partition_exact": partition_exact,
        "dispatch_combine_transpose": all(transposes),
        "tags": tags,
        "passed": len(plan.phases) == EXPECTED_PHASES
        and sum(len(phase.phase.segments) for phase in plan.phases)
        == EXPECTED_POSITIVE_PAIRS
        and partition_exact
        and all(transposes)
        and tags == list(range(1000, 1000 + EXPECTED_PHASES)),
    }


def _fatal_checks(
    out: Path,
    cells: dict[tuple[int, str], dict[str, object]],
    sinks: dict[tuple[int, str], object],
    omitted: dict[int, tuple[dict[str, object], object]],
    supply,
) -> dict[str, object]:
    exact_cells = []
    identity = []
    for vector_bytes in VECTOR_BYTES:
        for placement in PLACEMENTS:
            locality = cells[(vector_bytes, placement)]["locality"]
            observed = (
                locality["total_directed_bytes"],
                locality["fabric_directed_bytes"],
                locality["nvlink_directed_bytes"],
                locality["nvlink_service_ps"],
            )
            expected = EXPECTED_CELLS[(vector_bytes, placement)]
            exact_cells.append(
                {
                    "vector_bytes": vector_bytes,
                    "placement": placement,
                    "observed": list(observed),
                    "expected": list(expected),
                    "passed": observed == expected,
                }
            )

        explicit_dir = out / f"vector-{vector_bytes}" / "ABCD"
        omitted_dir = out / f"vector-{vector_bytes}" / "omitted"
        direct_goal = _direct_goal_oracle(vector_bytes, supply)
        frozen_size, frozen_digest = DIRECT_GOAL_ORACLES[vector_bytes]
        explicit_goals = _artifact_inventory(explicit_dir, "goal")
        omitted_goals = _artifact_inventory(omitted_dir, "goal")
        explicit_csvs = _artifact_inventory(explicit_dir, "rnic-nn-fluid.csv")
        omitted_csvs = _artifact_inventory(omitted_dir, "rnic-nn-fluid.csv")
        explicit_network = sinks[(vector_bytes, "ABCD")].outcomes[0]
        omitted_network = omitted[vector_bytes][1].outcomes[0]
        identity.append(
            {
                "vector_bytes": vector_bytes,
                "direct_goal": direct_goal,
                "frozen_direct_goal": {
                    "bytes": frozen_size,
                    "sha256": frozen_digest,
                },
                "explicit_goal_artifacts": explicit_goals,
                "omitted_goal_artifacts": omitted_goals,
                "explicit_csv_artifacts": explicit_csvs,
                "omitted_csv_artifacts": omitted_csvs,
                "passed": direct_goal
                == {"bytes": frozen_size, "sha256": frozen_digest}
                and explicit_goals == omitted_goals
                and explicit_csvs == omitted_csvs
                and explicit_network == omitted_network,
            }
        )

    all_intra = []
    all_remote = []
    metric_visibility = []
    for vector_bytes in VECTOR_BYTES:
        one = cells[(vector_bytes, "AAAA")]
        remote = cells[(vector_bytes, "ABCD")]
        one_locality = one["locality"]
        remote_locality = remote["locality"]
        one_sink = sinks[(vector_bytes, "AAAA")]
        all_intra.append(
            {
                "vector_bytes": vector_bytes,
                "fabric_bytes": one_locality["fabric_directed_bytes"],
                "fabric_segments": one_locality["fabric_segments"],
                "backend_runs": one_locality["backend_runs"],
                "num_flows": one["network"]["num_flows"],
                "goal_files": len(
                    list(
                        (out / f"vector-{vector_bytes}" / "AAAA").glob("*.goal")
                    )
                ),
            }
        )
        all_intra[-1]["passed"] = all(
            all_intra[-1][key] == 0
            for key in (
                "fabric_bytes",
                "fabric_segments",
                "backend_runs",
                "num_flows",
                "goal_files",
            )
        )
        all_remote.append(
            {
                "vector_bytes": vector_bytes,
                "nvlink_bytes": remote_locality["nvlink_directed_bytes"],
                "nvlink_segments": remote_locality["nvlink_segments"],
                "compatibility_fast_path": remote_locality[
                    "compatibility_fast_path"
                ],
                "passed": remote_locality["nvlink_directed_bytes"] == 0
                and remote_locality["nvlink_segments"] == 0
                and remote_locality["compatibility_fast_path"],
            }
        )
        metric_visibility.append(
            {
                "vector_bytes": vector_bytes,
                "ttft_ps": [
                    cells[(vector_bytes, name)]["ttft_ps"] for name in PLACEMENTS
                ],
                "tpot_ps": [
                    cells[(vector_bytes, name)]["tpot_ps"] for name in PLACEMENTS
                ],
                "passed": all(
                    cells[(vector_bytes, name)]["ttft_ps"]
                    == cells[(vector_bytes, name)]["tpot_ps"]
                    == cells[(vector_bytes, name)]["jct_ps"]
                    for name in PLACEMENTS
                ),
            }
        )
        if len(one_sink.outcomes) != 3:
            all_intra[-1]["passed"] = False

    quiescence = {
        f"{vector_bytes}:{placement}": all(
            outcome.quiescent for outcome in sink.outcomes
        )
        for (vector_bytes, placement), sink in sinks.items()
    }
    quiescence.update(
        {
            f"{vector_bytes}:omitted": all(
                outcome.quiescent for outcome in sink.outcomes
            )
            for vector_bytes, (_, sink) in omitted.items()
        }
    )
    physical_bounds = []
    for vector_bytes in VECTOR_BYTES:
        floor_ps = NVLINK_SERVICE_FLOORS_PS[vector_bytes]
        ceiling_ps = floor_ps + NVLINK_QUANTIZATION_WINDOW_PS
        locality = cells[(vector_bytes, "AAAA")]["locality"]
        observed_service_ps = locality["nvlink_service_ps"]
        observed_jct_ps = cells[(vector_bytes, "AAAA")]["jct_ps"]
        physical_bounds.append(
            {
                "vector_bytes": vector_bytes,
                "serialization_floor_ps": floor_ps,
                "quantization_ceiling_ps": ceiling_ps,
                "observed_nvlink_service_ps": observed_service_ps,
                "observed_jct_ps": observed_jct_ps,
                "passed": floor_ps <= observed_service_ps <= ceiling_ps
                and observed_jct_ps == observed_service_ps + COMPUTE_ESTIMATE_PS,
            }
        )
    observed_doubling_slack_ps = (
        2 * cells[(1_024, "AAAA")]["locality"]["nvlink_service_ps"]
        - cells[(2_048, "AAAA")]["locality"]["nvlink_service_ps"]
    )
    payload_doubling = {
        "observed_slack_ps": observed_doubling_slack_ps,
        "window_ps": NVLINK_QUANTIZATION_WINDOW_PS,
        "passed": 0 <= observed_doubling_slack_ps <= NVLINK_QUANTIZATION_WINDOW_PS,
    }
    phase_guards = [
        _phase_partition_guards(vector_bytes, supply)
        for vector_bytes in VECTOR_BYTES
    ]
    tp_widths = _tp_width_guards()
    checks = {
        "exact_cells": exact_cells,
        "all_remote_identity": identity,
        "all_intra_zero_fabric": all_intra,
        "all_remote_zero_nvlink": all_remote,
        "metric_visibility": metric_visibility,
        "physical_bounds": physical_bounds,
        "payload_doubling": payload_doubling,
        "phase_partition": phase_guards,
        "single_node_tp_widths": tp_widths,
        "quiescence": quiescence,
    }
    checks["passed"] = (
        all(row["passed"] for row in exact_cells)
        and all(row["passed"] for row in identity)
        and all(row["passed"] for row in all_intra)
        and all(row["passed"] for row in all_remote)
        and all(row["passed"] for row in metric_visibility)
        and all(row["passed"] for row in physical_bounds)
        and payload_doubling["passed"]
        and all(row["passed"] for row in phase_guards)
        and all(row["passed"] for row in tp_widths)
        and all(quiescence.values())
    )
    return checks


def run_study(out: Path) -> dict[str, object]:
    _check_frozen_registry()
    trace_path = _trace_path()
    trace_digest = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    if trace_digest != TRACE_SHA256:
        raise AssertionError("tracked Granite trace does not match the frozen input")
    out.mkdir(parents=True, exist_ok=False)
    projection = _routed_projection(trace_path)
    supply = _routed_supply(projection)

    cells: dict[tuple[int, str], dict[str, object]] = {}
    sinks: dict[tuple[int, str], object] = {}
    for vector_bytes in VECTOR_BYTES:
        for placement, hosts in PLACEMENTS.items():
            cell, sink, _ = _run_cell(
                out,
                vector_bytes=vector_bytes,
                placement_name=placement,
                hosts=hosts,
                supply=supply,
            )
            cells[(vector_bytes, placement)] = cell
            sinks[(vector_bytes, placement)] = sink

    omitted: dict[int, tuple[dict[str, object], object]] = {}
    for vector_bytes in VECTOR_BYTES:
        cell, sink, _ = _run_cell(
            out,
            vector_bytes=vector_bytes,
            placement_name="omitted",
            hosts=None,
            supply=supply,
        )
        omitted[vector_bytes] = (cell, sink)

    # Evaluate scored raw observations before any exact byte, conservation,
    # digest, zero or transpose oracle below.
    behavioral = _raw_behavioral(cells)
    fatal_unscored = _fatal_checks(out, cells, sinks, omitted, supply)
    summary = {
        "schema": "simllm-nvlink-locality-study-v1",
        "provenance": {
            "expectations_commit": EXPECTATIONS_COMMIT,
            "requalification_expectations_commit": (
                REQUALIFICATION_EXPECTATIONS_COMMIT
            ),
            "evidence_authored_against": EVIDENCE_AUTHORED_AGAINST,
            "observed_simllm_revision": _git_revision("HEAD"),
            "observed_htsim_gitlink": _git_revision("HEAD:third_party/htsim"),
            "python": sys.version,
            "platform": platform.platform(),
            "trace_path": str(
                trace_path.relative_to(Path(__file__).resolve().parents[2])
            ),
            "trace_sha256": trace_digest,
        },
        "configuration": {
            "profile": "rnic-nn-fluid",
            "linkspeed_bps": 400_000_000_000,
            "nvlink_bandwidth_bytes_per_second": NVLINK_BYTES_PER_SECOND,
            "nvlink_model": "declared-flat-maximum-endpoint-load",
            "nvlink_calibrated": False,
            "compute_estimate_ps": COMPUTE_ESTIMATE_PS,
            "vector_bytes": list(VECTOR_BYTES),
            "placements": {key: list(value) for key, value in PLACEMENTS.items()},
        },
        "cells": {
            f"{vector_bytes}:{placement}": cell
            for (vector_bytes, placement), cell in cells.items()
        },
        "omitted_locality_cells": {
            str(vector_bytes): cell for vector_bytes, (cell, _) in omitted.items()
        },
        "behavioral": behavioral,
        "behavioral_score": {
            "passed_families": sum(
                bool(family["passed"]) for family in behavioral.values()
            ),
            "total_families": 3,
            "passed_instances": sum(
                sum(bool(instance["passed"]) for instance in family["instances"])
                for family in behavioral.values()
            ),
            "total_instances": 8,
        },
        "fatal_unscored": fatal_unscored,
        "entailment": (
            "TRAF-B1 through TRAF-B3 were evaluated from raw locality counters, "
            "analytic service, and StepResult values before the exact split, "
            "conservation, zero, transpose, digest, and identity checks."
        ),
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    scored_passed = all(family["passed"] for family in behavioral.values())
    if not scored_passed or not fatal_unscored["passed"]:
        raise AssertionError("TRAF-10 study failed its frozen acceptance bar")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.check_only:
        check_only(args)
        return
    summary = run_study(args.out)
    print(json.dumps(summary["behavioral_score"], sort_keys=True))


if __name__ == "__main__":
    main()
