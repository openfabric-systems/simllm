"""Cross-check the analytic endpoint charge against the fluid fabric (CORE-43).

The frozen registry and the check-only dry run below precede every result. The
observation path replays the recorded Granite capture through the supported
metric chain twice, once with every EP rank on one host and once with every
rank on its own host, and compares the analytic intra-node serializer against
the fluid manifold's realized serialization on identical directed traffic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

#: recorded scheduler steps replayed from the capture tree
REPLAY_STEPS = 32
#: MoE layers, and one dispatch plus one combine phase per layer
MOE_LAYERS = 24
PHASES_PER_STEP = 2 * MOE_LAYERS
#: graph artifacts per step: one compute artifact per layer plus the phases
ARTIFACTS_PER_STEP = MOE_LAYERS + PHASES_PER_STEP
EP_WIDTH = 8
#: largest directed segment count any phase of this capture carries
MAX_SEGMENTS_PER_PHASE = EP_WIDTH - 1
#: fixed propagation the fluid manifold adds once after the last serviced bit
PROPAGATION_PS = 2_000_000
#: whole-nanosecond GOAL calc quantum of the analytic charge
ANALYTIC_QUANTUM_PS = 1_000
#: declared NVLink rate, used only for the deployment TTFT and TPOT arm
NVLINK_BYTES_PER_SECOND = 450_000_000_000

#: matched rate pairs: fluid bits/s, analytic bytes/s, picoseconds per byte
MATCHED_RATES = {
    400_000_000_000: (50_000_000_000, 20),
    200_000_000_000: (25_000_000_000, 40),
}

#: pre-freeze input characterization; see expectations.md
CAPTURE_LEDGER = {
    "prefill_step": 0,
    "prefill_peak_endpoint_bytes": 25_563_136,
    "prefill_peak_egress_bytes": 15_249_408,
    "prefill_segments": 336,
    "prefill_dispatch_bytes": 12_781_568,
    "prefill_combine_peak_egress_bytes": 2_467_840,
    "total_peak_endpoint_bytes": 54_218_752,
    "total_peak_egress_bytes": 32_567_296,
    "total_segments": 9_108,
}

#: frozen serialization and analytic-charge literals, picoseconds
FROZEN_SERIALIZATION_PS = {
    ("prefill", 400_000_000_000): 511_262_720,
    ("prefill", 200_000_000_000): 1_022_525_440,
    ("total", 400_000_000_000): 1_084_375_040,
    ("total", 200_000_000_000): 2_168_750_080,
}
FROZEN_ANALYTIC_PS = {
    ("prefill", 400_000_000_000): 511_290_000,
    ("prefill", 200_000_000_000): 1_022_550_000,
    ("total", 400_000_000_000): 1_084_962_000,
    ("total", 200_000_000_000): 2_169_586_000,
}

#: scored behavioral registry after the evidence-accounting corrections
SCORED_FAMILIES = {
    "CORE-F1": PHASES_PER_STEP * REPLAY_STEPS * len(MATCHED_RATES),
    "CORE-F2": REPLAY_STEPS,
}

#: entailed identities that remain fatal-unscored evidence
ENTAILED_FATAL_FAMILIES = {
    "CORE-F3": REPLAY_STEPS * len(MATCHED_RATES),
    "analytic_scaling_identity": REPLAY_STEPS,
}

#: recorded source artifacts under the capture tree
SOURCE_ARTIFACTS = {
    "steps": "replay-400g/steps.jsonl",
    "routing": "replay-400g/routed-experts.json",
}

EXPECTATIONS_COMMIT = "9f2cb0999c3ddbb79561f7c4ce760e73072b1804"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

#: executed arms: name -> (placement, fluid bits/s, analytic bytes/s)
ARMS = {
    "local-400": ("all-local", 400_000_000_000, 50_000_000_000),
    "remote-400": ("all-remote", 400_000_000_000, 50_000_000_000),
    "local-200": ("all-local", 200_000_000_000, 25_000_000_000),
    "remote-200": ("all-remote", 200_000_000_000, 25_000_000_000),
    "local-nvlink": ("all-local", 400_000_000_000, NVLINK_BYTES_PER_SECOND),
    "remote-400-control": ("all-remote", 400_000_000_000, NVLINK_BYTES_PER_SECOND),
}
#: the all-remote identity control differs from its reference only in the
#: analytic serializer's configuration, which the fabric path never reads
REMOTE_IDENTITY_PAIR = ("remote-400", "remote-400-control")


def _analytic_charge_ps(peak_endpoint_bytes: int, ps_per_byte: int) -> int:
    """Whole-nanosecond analytic phase service for one peak endpoint load."""

    ideal_ps = ps_per_byte * peak_endpoint_bytes
    quanta = -(-ideal_ps // ANALYTIC_QUANTUM_PS)
    return quanta * ANALYTIC_QUANTUM_PS


def _analytic_service_ps(peak_endpoint_bytes: int, analytic_bps: int) -> int:
    """The analytic model's own closed form, for any declared bytes-per-second.

    Written out here so the guard compares the production charge against a
    second construction rather than against itself.
    """

    whole_ns = -(-(peak_endpoint_bytes * 1_000_000_000) // analytic_bps)
    return whole_ns * ANALYTIC_QUANTUM_PS


def _check_frozen_registry() -> None:
    if PHASES_PER_STEP != 48 or ARTIFACTS_PER_STEP != 72:
        raise AssertionError("phase and artifact inventory drifted")
    if set(MATCHED_RATES) != {400_000_000_000, 200_000_000_000}:
        raise AssertionError("matched rate registry drifted")
    for linkspeed_bps, (analytic_bps, ps_per_byte) in MATCHED_RATES.items():
        if analytic_bps * 8 != linkspeed_bps:
            raise AssertionError("analytic bytes/s does not match the fluid bits/s")
        if ps_per_byte * linkspeed_bps != 8 * 1_000_000_000_000:
            raise AssertionError("picoseconds per byte is not the matched rate")
    if MATCHED_RATES[200_000_000_000][1] != 2 * MATCHED_RATES[400_000_000_000][1]:
        raise AssertionError("halving the rate must double picoseconds per byte")

    ledger = CAPTURE_LEDGER
    if ledger["prefill_peak_endpoint_bytes"] <= ledger["prefill_peak_egress_bytes"]:
        raise AssertionError("the capture-scale correction lost its direction")
    expected_ratio = 1.676336
    observed_ratio = (
        ledger["prefill_peak_endpoint_bytes"] / ledger["prefill_peak_egress_bytes"]
    )
    if abs(observed_ratio - expected_ratio) > 1e-6:
        raise AssertionError("the capture-scale undercharge literal drifted")
    dispatch = ledger["prefill_dispatch_bytes"]
    if ledger["prefill_peak_endpoint_bytes"] != 2 * dispatch:
        raise AssertionError("prefill dispatch and combine loads are not equal")
    if (
        ledger["prefill_peak_egress_bytes"]
        != dispatch + ledger["prefill_combine_peak_egress_bytes"]
    ):
        raise AssertionError("prefill egress-only aggregate does not decompose")
    if ledger["prefill_segments"] > MAX_SEGMENTS_PER_PHASE * PHASES_PER_STEP:
        raise AssertionError("prefill segment count exceeds the star bound")
    if ledger["total_segments"] > MAX_SEGMENTS_PER_PHASE * PHASES_PER_STEP * REPLAY_STEPS:
        raise AssertionError("total segment count exceeds the star bound")
    if ledger["total_peak_endpoint_bytes"] <= ledger["prefill_peak_endpoint_bytes"]:
        raise AssertionError("the prefill step cannot dominate every step")

    for cell, bytes_key in (
        ("prefill", "prefill_peak_endpoint_bytes"),
        ("total", "total_peak_endpoint_bytes"),
    ):
        for linkspeed_bps, (_, ps_per_byte) in MATCHED_RATES.items():
            ideal_ps = ps_per_byte * ledger[bytes_key]
            if FROZEN_SERIALIZATION_PS[(cell, linkspeed_bps)] != ideal_ps:
                raise AssertionError("frozen serialization literal drifted")
            analytic_ps = FROZEN_ANALYTIC_PS[(cell, linkspeed_bps)]
            if analytic_ps < ideal_ps:
                raise AssertionError("analytic charge fell below its own floor")
            phases = PHASES_PER_STEP if cell == "prefill" else PHASES_PER_STEP * REPLAY_STEPS
            if analytic_ps - ideal_ps >= phases * ANALYTIC_QUANTUM_PS:
                raise AssertionError("analytic quantization exceeds one quantum per phase")
            if analytic_ps % ANALYTIC_QUANTUM_PS:
                raise AssertionError("analytic charge is not whole-nanosecond")
        doubled = FROZEN_SERIALIZATION_PS[(cell, 400_000_000_000)] * 2
        if FROZEN_SERIALIZATION_PS[(cell, 200_000_000_000)] != doubled:
            raise AssertionError("serialization does not scale as one over bandwidth")

    if _analytic_charge_ps(1, 20) != ANALYTIC_QUANTUM_PS:
        raise AssertionError("analytic quantization identity drifted")
    if _analytic_charge_ps(50, 20) != ANALYTIC_QUANTUM_PS:
        raise AssertionError("a whole-nanosecond load must not be rounded up")
    if _analytic_charge_ps(51, 20) != 2 * ANALYTIC_QUANTUM_PS:
        raise AssertionError("a load past the quantum must take the next nanosecond")

    if SCORED_FAMILIES != {"CORE-F1": 3_072, "CORE-F2": 32}:
        raise AssertionError("scored evidence registry drifted")
    if ENTAILED_FATAL_FAMILIES != {
        "CORE-F3": 64,
        "analytic_scaling_identity": 32,
    }:
        raise AssertionError("entailed fatal evidence registry drifted")
    step_propagation_ps = PHASES_PER_STEP * PROPAGATION_PS
    if step_propagation_ps != 96_000_000:
        raise AssertionError("per-step propagation literal drifted")
    lower = step_propagation_ps - PHASES_PER_STEP * (ANALYTIC_QUANTUM_PS - 1)
    if lower != 95_952_048:
        raise AssertionError("CORE-F3 lower bound drifted")
    if set(SOURCE_ARTIFACTS) != {"steps", "routing"}:
        raise AssertionError("source artifact registry drifted")

    if set(ARMS) != {
        "local-400",
        "remote-400",
        "local-200",
        "remote-200",
        "local-nvlink",
        "remote-400-control",
    }:
        raise AssertionError("executed arm registry drifted")
    for name, (placement, linkspeed_bps, analytic_bps) in ARMS.items():
        if placement not in {"all-local", "all-remote"}:
            raise AssertionError(f"arm {name} has an unsupported placement")
        if linkspeed_bps not in MATCHED_RATES:
            raise AssertionError(f"arm {name} uses an unregistered link rate")
        if analytic_bps <= 0:
            raise AssertionError(f"arm {name} has a nonpositive analytic rate")
    reference, control = REMOTE_IDENTITY_PAIR
    if ARMS[reference][:2] != ARMS[control][:2]:
        raise AssertionError("the identity control must differ only in the analytic rate")
    if ARMS[reference][2] == ARMS[control][2]:
        raise AssertionError("the identity control must change the analytic rate")
    if len(EXPECTATIONS_COMMIT) != 40:
        raise AssertionError("revision provenance is malformed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--htsim-rnic", required=True, type=Path)
    parser.add_argument("--txt2bin", required=True, type=Path)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def check_only(args: argparse.Namespace) -> None:
    _check_frozen_registry()
    if any(
        not str(path)
        for path in (args.out, args.source_root, args.htsim_rnic, args.txt2bin)
    ):
        raise AssertionError("registered path argument is empty")
    print(
        "check-only validated the matched rate pairs, the capture ledger, the "
        "frozen serialization and analytic literals, two scored families and "
        "two entailed fatal families; no artifacts produced"
    )


def _git_revision(revision: str) -> str:
    return subprocess.run(
        ("git", "rev-parse", revision),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_clean_worktree() -> None:
    status = subprocess.run(
        ("git", "status", "--porcelain", "--untracked-files=all"),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise RuntimeError(
            "production evidence requires a clean worktree so the recorded "
            "SimLLM revision identifies the executed source"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    """Project observations onto the exact shapes a JSON reload returns."""

    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, (int, str, float)):
        return value
    raise TypeError(f"observation field is not JSON canonical: {type(value)!r}")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=1, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _validate_production_args(args: argparse.Namespace) -> dict[str, object]:
    _require_clean_worktree()
    configured_root = os.environ.get("SIMLLM_WAVE10_RUN_ROOT")
    if not configured_root:
        raise RuntimeError("SIMLLM_WAVE10_RUN_ROOT must name this branch's external run root")
    run_root = Path(configured_root).resolve()
    try:
        args.out.resolve().relative_to(run_root)
    except ValueError as exc:
        raise ValueError("study output must remain under SIMLLM_WAVE10_RUN_ROOT") from exc
    if args.out.exists():
        raise FileExistsError(f"study output already exists: {args.out}")
    for label, path in (("htsim_rnic", args.htsim_rnic), ("txt2bin", args.txt2bin)):
        if not path.is_file() or not os.access(path, os.X_OK):
            raise FileNotFoundError(f"{label} is not an executable file: {path}")
    provenance = {}
    for label, path in (("htsim_rnic", args.htsim_rnic), ("txt2bin", args.txt2bin)):
        provenance[label] = {
            "name": path.name,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    for label, relative in SOURCE_ARTIFACTS.items():
        source = args.source_root / relative
        if not source.is_file():
            raise FileNotFoundError(f"capture artifact {label} is missing: {source}")
        provenance[label] = {
            "name": relative,
            "bytes": source.stat().st_size,
            "sha256": _sha256(source),
        }
    os.environ["SIMLLM_HTSIM_RNIC"] = str(args.htsim_rnic.resolve())
    os.environ["SIMLLM_TXT2BIN"] = str(args.txt2bin.resolve())
    return provenance


def _dims():
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
        num_experts=32,
        top_k=8,
        moe_intermediate_size=512,
        local_num_experts=32 // EP_WIDTH,
    )


def _manifest(placement: str):
    from simllm.placement import PlacementManifest, RankPlacement

    if placement == "all-local":
        hosts = ("node-0",) * EP_WIDTH
    else:
        hosts = tuple(f"node-{rank}" for rank in range(EP_WIDTH))
    next_local_rank: dict[str, int] = {}
    ranks = []
    for global_rank, hostname in enumerate(hosts):
        local_rank = next_local_rank.get(hostname, 0)
        next_local_rank[hostname] = local_rank + 1
        ranks.append(RankPlacement(global_rank, hostname, local_rank))
    return PlacementManifest(ranks=ranks)


def _supply(source_root: Path):
    from simllm.preplay import read_routed_experts
    from simllm.traffic import ExpertPlacementSnapshot, RoutedMoeSupply

    routed = read_routed_experts(source_root / SOURCE_ARTIFACTS["routing"])
    snapshot = ExpertPlacementSnapshot(
        placement_epoch=0,
        expert_owners=tuple(
            (layer, expert, expert % EP_WIDTH)
            for layer in range(MOE_LAYERS)
            for expert in range(32)
        ),
    )
    return RoutedMoeSupply(
        engine_rank=0,
        routed_experts=routed,
        placements=(snapshot,),
        step_placement_epochs=tuple((step, 0) for step in range(REPLAY_STEPS + 1)),
    )


def _independent_endpoint_ledger(segments) -> tuple[tuple[int, int, int], ...]:
    """Second construction of the per-endpoint ledger, built inside the study."""

    egress: Counter[int] = Counter()
    ingress: Counter[int] = Counter()
    for segment in segments:
        egress[segment.source_rank] += segment.payload_bytes
        ingress[segment.destination_rank] += segment.payload_bytes
    return tuple(
        (rank, egress[rank], ingress[rank]) for rank in sorted(set(egress) | set(ingress))
    )


def _segment_key(segment) -> tuple[int, int, int, int]:
    return (
        segment.source_rank,
        segment.destination_rank,
        segment.payload_bytes,
        segment.tag,
    )


def _phase_inventory(records, source_root: Path, placement: str, analytic_bps: int):
    """Independently lower and classify every step for one placement."""

    from simllm.backends.step_lowerer import SerialStepLowerer, SerialStepLowererConfig
    from simllm.compute import RooflineProvider
    from simllm.placement import RankMapper
    from simllm.traffic.step_comm import plan_execution_graph_locality

    lowerer = SerialStepLowerer(
        SerialStepLowererConfig(
            dims=_dims(),
            tp_ranks=(0,),
            ep_ranks=tuple(range(EP_WIDTH)),
            provider=RooflineProvider(),
            routed_moe_supply=_supply(source_root),
        )
    )
    mapper = RankMapper(_manifest(placement))
    steps = []
    for record in records:
        plan = plan_execution_graph_locality(
            lowerer.lower(record),
            rank_mapper=mapper,
            nvlink_bandwidth_bytes_per_second=analytic_bps,
        )
        rows = []
        for phase in plan.phases:
            local = placement == "all-local"
            segments = phase.nvlink_segments if local else phase.fabric_segments
            ledger = _independent_endpoint_ledger(segments)
            peak_endpoint = max((max(e, i) for _, e, i in ledger), default=0)
            rows.append(
                {
                    "phase_id": phase.phase.phase_id,
                    "operation_id": phase.phase.operation_id,
                    "segments": [list(_segment_key(item)) for item in segments],
                    "segment_count": len(segments),
                    "directed_bytes": sum(item.payload_bytes for item in segments),
                    "ledger": [list(row) for row in ledger],
                    "ledger_egress_sum": sum(row[1] for row in ledger),
                    "ledger_ingress_sum": sum(row[2] for row in ledger),
                    "peak_endpoint_bytes": peak_endpoint,
                    "peak_egress_bytes": max((e for _, e, _ in ledger), default=0),
                    "hub_ranks": [
                        rank for rank, e, i in ledger if max(e, i) == peak_endpoint
                    ],
                    "analytic_service_ps": phase.nvlink_service_ps,
                }
            )
        steps.append(
            {
                "step_index": record.step_index,
                "phase_count": len(plan.phases),
                "local_bytes": plan.nvlink_bytes,
                "fabric_bytes": plan.fabric_bytes,
                "phases": rows,
            }
        )
    return steps


def _artifact_inventory(workdir: Path) -> list[dict[str, object]]:
    rows = []
    for path in sorted(candidate for candidate in workdir.rglob("*") if candidate.is_file()):
        rows.append(
            {
                "path": path.relative_to(workdir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return rows


def _run_arm(out: Path, name: str, records, source_root: Path) -> dict[str, Any]:
    from simllm.backends import HtsimStepSink, HtsimStepSinkConfig
    from simllm.compute import RooflineProvider

    placement, linkspeed_bps, analytic_bps = ARMS[name]
    workdir = out / "arms" / name
    sink = HtsimStepSink(
        HtsimStepSinkConfig(
            profile="rnic-nn-fluid",
            tp_ranks=(0,),
            dims=_dims(),
            workdir=workdir,
            ep_ranks=tuple(range(EP_WIDTH)),
            linkspeed_bps=linkspeed_bps,
            provider=RooflineProvider(),
            placement_manifest=_manifest(placement),
            nvlink_bandwidth_bytes_per_second=analytic_bps,
            num_goal_ranks=EP_WIDTH,
            routed_moe_supply=_supply(source_root),
        )
    )
    virtual_time_ps = 0
    steps = []
    for record in records:
        result = sink(replace(record, virtual_time_ps=virtual_time_ps))
        if result is None:
            raise AssertionError(f"arm {name} produced no StepResult")
        steps.append(
            {
                "step_index": record.step_index,
                "step_latency_ps": result.step_latency_ps,
                "completed_at_ps": result.completed_at_ps,
                "scheduled_request_ids": [
                    request.request_id for request in record.scheduled
                ],
            }
        )
        virtual_time_ps = result.completed_at_ps
    for index, (outcome, locality) in enumerate(
        zip(sink.outcomes, sink.locality_outcomes)
    ):
        steps[index]["network_outcome"] = asdict(outcome)
        steps[index]["locality_outcome"] = asdict(locality)
    return {
        "arm": name,
        "placement": placement,
        "linkspeed_bps": linkspeed_bps,
        "analytic_bytes_per_second": analytic_bps,
        "steps": steps,
        "artifacts": _artifact_inventory(workdir),
    }


def _collective_service_ps(step: dict[str, Any], placement: str) -> list[int]:
    """Per-collective realized service in executed artifact order."""

    locality = step["locality_outcome"]
    operation_ids = locality["artifact_operation_ids"]
    fabric = locality["fabric_phase_service_ps"]
    composed = locality["composed_phase_service_ps"]
    services = []
    for index, ids in enumerate(operation_ids):
        if len(ids) != 1 or not ids[0].endswith((":ep-dispatch", ":ep-combine")):
            continue
        services.append(
            fabric[index] if placement == "all-remote" else composed[index]
        )
    return services


def _collective_operation_ids(step: dict[str, Any]) -> list[str]:
    return [
        ids[0]
        for ids in step["locality_outcome"]["artifact_operation_ids"]
        if len(ids) == 1 and ids[0].endswith((":ep-dispatch", ":ep-combine"))
    ]


def _compute_service_ps(step: dict[str, Any]) -> int:
    locality = step["locality_outcome"]
    total = 0
    for index, ids in enumerate(locality["artifact_operation_ids"]):
        if len(ids) == 1 and ids[0].endswith((":ep-dispatch", ":ep-combine")):
            continue
        total += locality["composed_phase_service_ps"][index]
    return total


def _request_metrics(arm: dict[str, Any], arrivals: dict[str, int]) -> dict[str, Any]:
    """TTFT and TPOT for every replayed request, from live StepResults."""

    first: dict[str, int] = {}
    last: dict[str, int] = {}
    tokens: Counter[str] = Counter()
    for step in arm["steps"]:
        for request_id in step["scheduled_request_ids"]:
            tokens[request_id] += 1
            if request_id not in first:
                first[request_id] = step["completed_at_ps"]
            last[request_id] = step["completed_at_ps"]
    metrics = {}
    for request_id, arrival_ps in sorted(arrivals.items()):
        steps_scheduling = tokens[request_id]
        metrics[request_id] = {
            "arrival_ps": arrival_ps,
            "ttft_ps": first[request_id] - arrival_ps,
            "tpot_ps": (
                (last[request_id] - first[request_id]) // (steps_scheduling - 1)
                if steps_scheduling > 1
                else None
            ),
            "scheduled_steps": steps_scheduling,
        }
    return metrics


def _phase_rows(
    inventories: dict[str, Any],
    arms: dict[str, Any],
    linkspeed_bps: int,
) -> list[dict[str, Any]]:
    """Raw per-phase comparison rows for one matched rate."""

    _, ps_per_byte = MATCHED_RATES[linkspeed_bps]
    local_name = "local-400" if linkspeed_bps == 400_000_000_000 else "local-200"
    remote_name = "remote-400" if linkspeed_bps == 400_000_000_000 else "remote-200"
    local_arm, remote_arm = arms[local_name], arms[remote_name]
    local_inventory = inventories[local_name]
    remote_inventory = inventories[remote_name]
    rows = []
    for index in range(REPLAY_STEPS):
        local_services = _collective_service_ps(local_arm["steps"][index], "all-local")
        remote_services = _collective_service_ps(
            remote_arm["steps"][index], "all-remote"
        )
        local_phases = local_inventory[index]["phases"]
        remote_phases = remote_inventory[index]["phases"]
        for phase_index, local_phase in enumerate(local_phases):
            remote_phase = remote_phases[phase_index]
            peak = local_phase["peak_endpoint_bytes"]
            segments = local_phase["segment_count"]
            analytic_ps = local_services[phase_index]
            fluid_ps = remote_services[phase_index]
            serialization_ps = fluid_ps - PROPAGATION_PS
            difference_ps = analytic_ps - serialization_ps
            rows.append(
                {
                    "linkspeed_bps": linkspeed_bps,
                    "step_index": index,
                    "phase_id": local_phase["phase_id"],
                    "peak_endpoint_bytes": peak,
                    "segment_count": segments,
                    "ideal_serialization_ps": ps_per_byte * peak,
                    "analytic_service_ps": analytic_ps,
                    "fluid_artifact_ps": fluid_ps,
                    "fluid_serialization_ps": serialization_ps,
                    "difference_ps": difference_ps,
                    "band_low_ps": -segments,
                    "band_high_ps": ANALYTIC_QUANTUM_PS - 1,
                    "fluid_above_floor": serialization_ps >= ps_per_byte * peak,
                    "fluid_below_ceiling": (
                        serialization_ps <= ps_per_byte * peak + segments
                    ),
                    "segments_identical": (
                        local_phase["segments"] == remote_phase["segments"]
                    ),
                    "passed": -segments <= difference_ps <= ANALYTIC_QUANTUM_PS - 1,
                }
            )
    return rows


def _raw_behavioral(
    inventories: dict[str, Any],
    arms: dict[str, Any],
) -> dict[str, Any]:
    phase_rows = []
    for linkspeed_bps in sorted(MATCHED_RATES, reverse=True):
        phase_rows.extend(_phase_rows(inventories, arms, linkspeed_bps))

    scaling_rows = []
    analytic_scaling_rows = []
    for index in range(REPLAY_STEPS):
        fast = [row for row in phase_rows if row["step_index"] == index and row["linkspeed_bps"] == 400_000_000_000]
        slow = [row for row in phase_rows if row["step_index"] == index and row["linkspeed_bps"] == 200_000_000_000]
        segment_total = sum(row["segment_count"] for row in slow)
        fluid_fast = sum(row["fluid_serialization_ps"] for row in fast)
        fluid_slow = sum(row["fluid_serialization_ps"] for row in slow)
        analytic_fast = sum(row["analytic_service_ps"] for row in fast)
        analytic_slow = sum(row["analytic_service_ps"] for row in slow)
        quantization = PHASES_PER_STEP * ANALYTIC_QUANTUM_PS
        scaling_rows.append(
            {
                "step_index": index,
                "arm": "fluid",
                "fast_ps": fluid_fast,
                "slow_ps": fluid_slow,
                "doubled_fast_ps": 2 * fluid_fast,
                "allowance_ps": segment_total,
                "propagation_fast_ps": PHASES_PER_STEP * PROPAGATION_PS,
                "propagation_slow_ps": PHASES_PER_STEP * PROPAGATION_PS,
                "passed": abs(fluid_slow - 2 * fluid_fast) <= segment_total,
            }
        )
        analytic_scaling_rows.append(
            {
                "step_index": index,
                "arm": "analytic",
                "fast_ps": analytic_fast,
                "slow_ps": analytic_slow,
                "doubled_fast_ps": 2 * analytic_fast,
                "allowance_ps": quantization,
                "passed": abs(analytic_slow - 2 * analytic_fast) <= quantization,
            }
        )

    composition_rows = []
    for linkspeed_bps in sorted(MATCHED_RATES, reverse=True):
        local_name = "local-400" if linkspeed_bps == 400_000_000_000 else "local-200"
        remote_name = "remote-400" if linkspeed_bps == 400_000_000_000 else "remote-200"
        for index in range(REPLAY_STEPS):
            local_step = arms[local_name]["steps"][index]
            remote_step = arms[remote_name]["steps"][index]
            segment_total = sum(
                phase["segment_count"]
                for phase in inventories[local_name][index]["phases"]
            )
            delta = remote_step["step_latency_ps"] - local_step["step_latency_ps"]
            low = PHASES_PER_STEP * PROPAGATION_PS - PHASES_PER_STEP * (
                ANALYTIC_QUANTUM_PS - 1
            )
            high = PHASES_PER_STEP * PROPAGATION_PS + segment_total
            composition_rows.append(
                {
                    "linkspeed_bps": linkspeed_bps,
                    "step_index": index,
                    "local_step_latency_ps": local_step["step_latency_ps"],
                    "remote_step_latency_ps": remote_step["step_latency_ps"],
                    "difference_ps": delta,
                    "band_low_ps": low,
                    "band_high_ps": high,
                    "passed": low <= delta <= high,
                }
            )

    families = {
        "CORE-F1": {"instances": phase_rows},
        "CORE-F2": {"instances": scaling_rows},
    }
    for name, family in families.items():
        family["passed"] = all(row["passed"] for row in family["instances"])
        family["instance_count"] = len(family["instances"])
        if family["instance_count"] != SCORED_FAMILIES[name]:
            raise AssertionError(f"{name} instance count disagrees with the freeze")
    entailed = {
        "CORE-F3": composition_rows,
        "analytic_scaling_identity": analytic_scaling_rows,
    }
    for name, rows in entailed.items():
        if len(rows) != ENTAILED_FATAL_FAMILIES[name]:
            raise AssertionError(f"{name} instance count disagrees with the registry")
    return families, entailed


def _fatal_checks(
    inventories: dict[str, Any],
    arms: dict[str, Any],
) -> dict[str, Any]:
    population_rows = []
    ledger_rows = []
    quantization_rows = []
    structure_rows = []
    projection_rows = []
    for name, arm in sorted(arms.items()):
        placement = arm["placement"]
        analytic_bps = arm["analytic_bytes_per_second"]
        steps = inventories[name]
        for index, step in enumerate(steps):
            observed = arm["steps"][index]["locality_outcome"]
            local_bytes = sum(
                phase["directed_bytes"] for phase in step["phases"]
            )
            population_rows.append(
                {
                    "arm": name,
                    "step_index": index,
                    "phase_count": step["phase_count"],
                    "reconstructed_bytes": local_bytes,
                    "observed_local_bytes": observed["nvlink_directed_bytes"],
                    "observed_fabric_bytes": observed["fabric_directed_bytes"],
                    "passed": (
                        step["phase_count"] == PHASES_PER_STEP
                        and observed["phase_count"] == PHASES_PER_STEP
                        and observed["artifact_count"] == ARTIFACTS_PER_STEP
                        and (
                            observed["fabric_directed_bytes"] == local_bytes
                            and observed["nvlink_directed_bytes"] == 0
                            if placement == "all-remote"
                            else observed["nvlink_directed_bytes"] == local_bytes
                            and observed["fabric_directed_bytes"] == 0
                        )
                        and observed["backend_runs"]
                        == (PHASES_PER_STEP if placement == "all-remote" else 0)
                    ),
                }
            )
            for phase in step["phases"]:
                bytes_total = phase["directed_bytes"]
                ledger_rows.append(
                    {
                        "arm": name,
                        "step_index": index,
                        "phase_id": phase["phase_id"],
                        "passed": (
                            phase["ledger_egress_sum"]
                            == phase["ledger_ingress_sum"]
                            == bytes_total
                            and [row[0] for row in phase["ledger"]]
                            == sorted({row[0] for row in phase["ledger"]})
                            and all(row[1] >= 0 and row[2] >= 0 for row in phase["ledger"])
                        ),
                    }
                )
                structure_rows.append(
                    {
                        "arm": name,
                        "step_index": index,
                        "phase_id": phase["phase_id"],
                        "segment_count": phase["segment_count"],
                        "hub_count": len(phase["hub_ranks"]),
                        "passed": (
                            phase["segment_count"] <= MAX_SEGMENTS_PER_PHASE
                            and len(phase["hub_ranks"]) == 1
                            and phase["hub_ranks"] == [0]
                            and phase["peak_endpoint_bytes"] == bytes_total
                        ),
                    }
                )
                if placement == "all-local":
                    expected = _analytic_service_ps(
                        phase["peak_endpoint_bytes"], analytic_bps
                    )
                    quantization_rows.append(
                        {
                            "arm": name,
                            "step_index": index,
                            "phase_id": phase["phase_id"],
                            "observed_ps": phase["analytic_service_ps"],
                            "expected_ps": expected,
                            "passed": phase["analytic_service_ps"] == expected,
                        }
                    )
            if placement == "all-local":
                executed = _collective_service_ps(arm["steps"][index], placement)
                projection_rows.append(
                    {
                        "arm": name,
                        "step_index": index,
                        "passed": executed
                        == [phase["analytic_service_ps"] for phase in step["phases"]],
                    }
                )

    identity_rows = []
    for linkspeed_bps in sorted(MATCHED_RATES, reverse=True):
        local_name = "local-400" if linkspeed_bps == 400_000_000_000 else "local-200"
        remote_name = "remote-400" if linkspeed_bps == 400_000_000_000 else "remote-200"
        for index in range(REPLAY_STEPS):
            local_phases = inventories[local_name][index]["phases"]
            remote_phases = inventories[remote_name][index]["phases"]
            identity_rows.append(
                {
                    "linkspeed_bps": linkspeed_bps,
                    "step_index": index,
                    "passed": (
                        [phase["phase_id"] for phase in local_phases]
                        == [phase["phase_id"] for phase in remote_phases]
                        and [
                            sorted(map(tuple, phase["segments"])) for phase in local_phases
                        ]
                        == [
                            sorted(map(tuple, phase["segments"])) for phase in remote_phases
                        ]
                        and _collective_operation_ids(arms[local_name]["steps"][index])
                        == _collective_operation_ids(arms[remote_name]["steps"][index])
                    ),
                }
            )

    quiescence_rows = []
    for name, arm in sorted(arms.items()):
        if arm["placement"] != "all-remote":
            continue
        for index, step in enumerate(arm["steps"]):
            expected_flows = sum(
                phase["segment_count"] for phase in inventories[name][index]["phases"]
            )
            quiescence_rows.append(
                {
                    "arm": name,
                    "step_index": index,
                    "num_flows": step["network_outcome"]["num_flows"],
                    "expected_flows": expected_flows,
                    "quiescent": step["network_outcome"]["quiescent"],
                    "passed": (
                        step["network_outcome"]["quiescent"]
                        and step["network_outcome"]["num_flows"] == expected_flows
                    ),
                }
            )

    reference, control = REMOTE_IDENTITY_PAIR
    control_rows = [
        {
            "surface": "artifact inventory",
            "passed": arms[reference]["artifacts"] == arms[control]["artifacts"],
        },
        {
            "surface": "step latencies and outcomes",
            "passed": [
                {key: step[key] for key in step if key != "locality_outcome"}
                for step in arms[reference]["steps"]
            ]
            == [
                {key: step[key] for key in step if key != "locality_outcome"}
                for step in arms[control]["steps"]
            ],
        },
        {
            "surface": "fabric phase services",
            "passed": [
                step["locality_outcome"]["fabric_phase_service_ps"]
                for step in arms[reference]["steps"]
            ]
            == [
                step["locality_outcome"]["fabric_phase_service_ps"]
                for step in arms[control]["steps"]
            ],
        },
    ]

    compute_rows = []
    for index in range(REPLAY_STEPS):
        observed = {
            name: _compute_service_ps(arm["steps"][index])
            for name, arm in sorted(arms.items())
        }
        compute_rows.append(
            {
                "step_index": index,
                "compute_service_ps": observed,
                "passed": len(set(observed.values())) == 1,
            }
        )

    guards = {
        "byte_population_identity": population_rows + identity_rows,
        "endpoint_ledger_conservation": ledger_rows,
        "analytic_quantization_identity": quantization_rows,
        "structural_star_identity": structure_rows,
        "backend_quiescence": quiescence_rows,
        "all_remote_exactness": control_rows,
        "compute_identity": compute_rows,
        "analytic_projection_identity": projection_rows,
    }
    return {
        name: {"rows": rows, "passed": all(row["passed"] for row in rows)}
        for name, rows in guards.items()
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    from simllm.core.step import step_records_from_jsonl

    provenance = _validate_production_args(args)
    args.out.mkdir(parents=True, exist_ok=False)
    records = step_records_from_jsonl(args.source_root / SOURCE_ARTIFACTS["steps"])
    if len(records) != REPLAY_STEPS:
        raise AssertionError("the recorded capture does not carry the frozen step count")

    inventories = {}
    arms = {}
    for name, (placement, _, analytic_bps) in ARMS.items():
        inventories[name] = _phase_inventory(
            records, args.source_root, placement, analytic_bps
        )
        arms[name] = _run_arm(args.out, name, records, args.source_root)

    behavioral, entailed = _raw_behavioral(inventories, arms)
    fatal = _fatal_checks(inventories, arms)
    for name, rows in entailed.items():
        fatal[name] = {
            "rows": rows,
            "passed": all(row["passed"] for row in rows),
        }

    arrivals = {"r0": 0, "r1": 2_000_000_000, "r2": 4_000_000_000}
    metrics = {
        name: _request_metrics(arm, arrivals) for name, arm in sorted(arms.items())
    }

    summary = {
        "schema": "simllm-endpoint-fabric-crosscheck-results-v1",
        "provenance": {
            "expectations_commit": EXPECTATIONS_COMMIT,
            "run_head": _git_revision("HEAD"),
            "observed_htsim_gitlink": _git_revision("HEAD:third_party/htsim"),
            "gitlink_equality_required": False,
            "python": sys.version,
            "platform": platform.platform(),
            "tools": provenance,
        },
        "capture_ledger": CAPTURE_LEDGER,
        "scored": behavioral,
        "fatal": fatal,
        "request_metrics": metrics,
        "arms": arms,
        "phase_inventories": inventories,
    }
    _write_json(args.out / "summary.json", _jsonable(summary))
    return summary


def main() -> int:
    args = parse_args()
    if args.check_only:
        check_only(args)
        return 0
    _check_frozen_registry()
    summary = _run(args)
    scored_failures = [
        name for name, family in summary["scored"].items() if not family["passed"]
    ]
    fatal_failures = [
        name for name, guard in summary["fatal"].items() if not guard["passed"]
    ]
    for name, guard in sorted(summary["fatal"].items()):
        print(f"fatal {name}: {'passed' if guard['passed'] else 'VIOLATED'}")
    for name, family in sorted(summary["scored"].items()):
        passed = sum(1 for row in family["instances"] if row["passed"])
        print(f"scored {name}: {passed}/{family['instance_count']}")
    if fatal_failures:
        raise AssertionError(f"fatal guard violated, run is void: {fatal_failures}")
    if scored_failures:
        raise AssertionError(f"a scored behavioral relation failed: {scored_failures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
