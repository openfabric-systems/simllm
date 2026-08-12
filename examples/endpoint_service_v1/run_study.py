"""Run the frozen CORE-41 endpoint-service study after implementation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

EP_WIDTHS = (2, 4, 8)
PAYLOAD_BYTES = (1_024, 2_048)
SHAPES = ("symmetric", "dispatch-star", "combine-star")
NVLINK_BYTES_PER_SECOND = 450_000_000_000
COMPUTE_PS = 10_000
REPLAY_STEPS = 3
SCORED_FAMILIES = 4
SCORED_INSTANCES = 20
EXPECTATIONS_COMMIT = "3879fb01a7249bbe92fe4342ad9e163570c2da1d"
EVIDENCE_AUTHORED_AGAINST = "76223875557a552deb5aa2c2c529a07f000135ba"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

EXPECTED_ENDPOINT_SERVICE_PS = {
    (1_024, 2): 3_000,
    (1_024, 4): 7_000,
    (1_024, 8): 16_000,
    (2_048, 2): 5_000,
    (2_048, 4): 14_000,
    (2_048, 8): 32_000,
}
EXPECTED_COMBINE_CHANGE_PS = {
    (1_024, 2): 0,
    (1_024, 4): 4_000,
    (1_024, 8): 13_000,
    (2_048, 2): 0,
    (2_048, 4): 9_000,
    (2_048, 8): 27_000,
}
DEPENDENCY_AUTHORITY_REFREEZE = {
    1_024: {
        "old_service_ps": 4_538_000,
        "new_service_ps": 6_652_000,
        "old_jct_ps": 4_562_000,
        "new_jct_ps": 6_676_000,
    },
    2_048: {
        "old_service_ps": 9_047_000,
        "new_service_ps": 13_286_000,
        "old_jct_ps": 9_071_000,
        "new_jct_ps": 13_310_000,
    },
}
SEQGEN_ENDPOINT_BYTES = 16_384
SEQGEN_FLOORS_PS = {
    200_000_000_000: 655_360,
    400_000_000_000: 327_680,
}


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def _service_ps(payload_bytes: int) -> int:
    return (
        _ceil_div(
            payload_bytes * 1_000_000_000,
            NVLINK_BYTES_PER_SECOND,
        )
        * 1_000
    )


def _check_frozen_registry() -> None:
    keys = {(payload_bytes, width) for payload_bytes in PAYLOAD_BYTES for width in EP_WIDTHS}
    if set(EXPECTED_ENDPOINT_SERVICE_PS) != keys:
        raise AssertionError("endpoint-service grid is incomplete")
    if set(EXPECTED_COMBINE_CHANGE_PS) != keys:
        raise AssertionError("combine-change grid is incomplete")
    if SHAPES != ("symmetric", "dispatch-star", "combine-star"):
        raise AssertionError("fixture-shape registry drifted")

    for payload_bytes, width in sorted(keys):
        endpoint_bytes = (width - 1) * payload_bytes
        expected_service = _service_ps(endpoint_bytes)
        if EXPECTED_ENDPOINT_SERVICE_PS[(payload_bytes, width)] != expected_service:
            raise AssertionError("endpoint-service arithmetic drifted")
        old_combine_service = _service_ps(payload_bytes)
        expected_change = expected_service - old_combine_service
        if EXPECTED_COMBINE_CHANGE_PS[(payload_bytes, width)] != expected_change:
            raise AssertionError("combine-change arithmetic drifted")

        floor_numerator = endpoint_bytes * 1_000_000_000_000
        service_numerator = expected_service * NVLINK_BYTES_PER_SECOND
        if not (
            floor_numerator <= service_numerator < floor_numerator + 1_000 * NVLINK_BYTES_PER_SECOND
        ):
            raise AssertionError("service lies outside its physical bounds")
        if expected_service % 1_000:
            raise AssertionError("analytic service is not whole-nanosecond")

    if any(EXPECTED_COMBINE_CHANGE_PS[(payload_bytes, 2)] != 0 for payload_bytes in PAYLOAD_BYTES):
        raise AssertionError("width-two compatibility drifted")
    if any(
        EXPECTED_COMBINE_CHANGE_PS[(payload_bytes, width)] <= 0
        for payload_bytes in PAYLOAD_BYTES
        for width in (4, 8)
    ):
        raise AssertionError("combine response lost its positive direction")
    for width in EP_WIDTHS:
        if EXPECTED_ENDPOINT_SERVICE_PS[(2_048, width)] != _service_ps(2 * (width - 1) * 1_024):
            raise AssertionError("payload scaling arithmetic drifted")

    for row in DEPENDENCY_AUTHORITY_REFREEZE.values():
        service_change = row["new_service_ps"] - row["old_service_ps"]
        jct_change = row["new_jct_ps"] - row["old_jct_ps"]
        if service_change <= 0 or jct_change != service_change:
            raise AssertionError("dependency-authority refreeze arithmetic drifted")
        if row["old_jct_ps"] - row["old_service_ps"] != 24_000:
            raise AssertionError("old dependency compute term drifted")
        if row["new_jct_ps"] - row["new_service_ps"] != 24_000:
            raise AssertionError("new dependency compute term drifted")

    for rate_bps, expected_floor_ps in SEQGEN_FLOORS_PS.items():
        observed_floor_ps = SEQGEN_ENDPOINT_BYTES * 8 * 1_000_000_000_000 // rate_bps
        if observed_floor_ps != expected_floor_ps:
            raise AssertionError("seqgen endpoint floor arithmetic drifted")
    if (SCORED_FAMILIES, SCORED_INSTANCES) != (4, 20):
        raise AssertionError("behavioral evidence registry drifted")
    if len(EXPECTATIONS_COMMIT) != 40 or len(EVIDENCE_AUTHORED_AGAINST) != 40:
        raise AssertionError("revision provenance is malformed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("baseline", "corrected"), required=True)
    parser.add_argument("--baseline-summary", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--htsim-rnic", type=Path, required=True)
    parser.add_argument("--txt2bin", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def check_only(args: argparse.Namespace) -> None:
    _check_frozen_registry()
    if args.mode == "baseline" and args.baseline_summary is not None:
        raise ValueError("baseline mode does not accept --baseline-summary")
    if args.mode == "corrected" and args.baseline_summary is None:
        raise ValueError("corrected mode requires --baseline-summary")
    if any(not str(path) for path in (args.out, args.htsim_rnic, args.txt2bin)):
        raise AssertionError("registered path argument is empty")
    print(
        f"check-only mode={args.mode} out={args.out}; "
        "validated frozen literals and produced no artifacts"
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
    """Project observations onto the exact shapes a JSON reload returns.

    Baseline observations reach the corrected run through ``summary.json``, so
    a tuple written by ``dataclasses.asdict`` comes back as a list. Comparing
    identity across modes only means something when both sides live in the
    same value space, so every observation is canonicalized once here.
    """

    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return value
    raise TypeError(f"observation field is not JSON canonical: {type(value)!r}")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _validate_production_args(args: argparse.Namespace) -> dict[str, object]:
    _require_clean_worktree()
    configured_root = os.environ.get("SIMLLM_WAVE6_RUN_ROOT")
    if not configured_root:
        raise RuntimeError("SIMLLM_WAVE6_RUN_ROOT must name this branch's external run root")
    run_root = Path(configured_root).resolve()
    try:
        args.out.resolve().relative_to(run_root)
    except ValueError as exc:
        raise ValueError("study output must remain under SIMLLM_WAVE6_RUN_ROOT") from exc
    if args.out.exists():
        raise FileExistsError(f"study output already exists: {args.out}")
    if args.mode == "baseline" and args.baseline_summary is not None:
        raise ValueError("baseline mode does not accept --baseline-summary")
    if args.mode == "corrected":
        if args.baseline_summary is None:
            raise ValueError("corrected mode requires --baseline-summary")
        try:
            args.baseline_summary.resolve().relative_to(run_root)
        except ValueError as exc:
            raise ValueError("baseline summary must remain under SIMLLM_WAVE6_RUN_ROOT") from exc
        if not args.baseline_summary.is_file():
            raise FileNotFoundError(f"baseline summary does not exist: {args.baseline_summary}")
    for label, path in (
        ("htsim_rnic", args.htsim_rnic),
        ("txt2bin", args.txt2bin),
    ):
        if not path.is_file() or not os.access(path, os.X_OK):
            raise FileNotFoundError(f"{label} is not an executable file: {path}")
    os.environ["SIMLLM_HTSIM_RNIC"] = str(args.htsim_rnic.resolve())
    os.environ["SIMLLM_TXT2BIN"] = str(args.txt2bin.resolve())
    return {
        "htsim_rnic": {
            "name": args.htsim_rnic.name,
            "bytes": args.htsim_rnic.stat().st_size,
            "sha256": _sha256(args.htsim_rnic),
        },
        "txt2bin": {
            "name": args.txt2bin.name,
            "bytes": args.txt2bin.stat().st_size,
            "sha256": _sha256(args.txt2bin),
        },
    }


def _manifest(hosts: tuple[str, ...]):
    from simllm.placement import PlacementManifest, RankPlacement

    next_local_rank: dict[str, int] = {}
    ranks = []
    for global_rank, hostname in enumerate(hosts):
        local_rank = next_local_rank.get(hostname, 0)
        next_local_rank[hostname] = local_rank + 1
        ranks.append(RankPlacement(global_rank, hostname, local_rank))
    return PlacementManifest(ranks=ranks)


def _phase(shape: str, payload_bytes: int, width: int):
    from simllm.traffic import (
        CollectiveCommunicationPhase,
        DirectedCollectiveSegment,
    )

    if shape == "symmetric":
        pairs = (
            (source, destination)
            for source in range(width)
            for destination in range(width)
            if source != destination
        )
    elif shape == "dispatch-star":
        pairs = ((0, destination) for destination in range(1, width))
    elif shape == "combine-star":
        pairs = ((source, 0) for source in range(1, width))
    else:
        raise ValueError(f"unsupported fixture shape: {shape}")
    return CollectiveCommunicationPhase(
        phase_id=f"{shape}:p{payload_bytes}:w{width}",
        layer=0,
        participants=tuple(range(width)),
        segments=tuple(
            DirectedCollectiveSegment(source, destination, payload_bytes, 7)
            for source, destination in pairs
        ),
    )


def _independent_endpoint_ledger(segments) -> tuple[tuple[int, int, int], ...]:
    egress: Counter[int] = Counter()
    ingress: Counter[int] = Counter()
    for segment in segments:
        egress[segment.source_rank] += segment.payload_bytes
        ingress[segment.destination_rank] += segment.payload_bytes
    return tuple((rank, egress[rank], ingress[rank]) for rank in sorted(set(egress) | set(ingress)))


def _component_cells() -> dict[str, dict[str, object]]:
    from simllm.placement import RankMapper
    from simllm.traffic import classify_step_locality

    cells = {}
    for payload_bytes in PAYLOAD_BYTES:
        for width in EP_WIDTHS:
            mapper = RankMapper(_manifest(("local-node",) * width))
            for shape in SHAPES:
                phase = _phase(shape, payload_bytes, width)
                classified = classify_step_locality(
                    (phase,),
                    rank_mapper=mapper,
                    bandwidth_bytes_per_second=NVLINK_BYTES_PER_SECOND,
                ).phases[0]
                independent_ledger = _independent_endpoint_ledger(classified.nvlink_segments)
                production_ledger = getattr(
                    classified,
                    "nvlink_endpoint_bytes",
                    None,
                )
                source_services = getattr(
                    classified,
                    "nvlink_source_service_ns",
                    None,
                )
                key = f"{payload_bytes}:{width}:{shape}"
                cells[key] = {
                    "payload_bytes": payload_bytes,
                    "ep_width": width,
                    "shape": shape,
                    "segments": [
                        [
                            segment.source_rank,
                            segment.destination_rank,
                            segment.payload_bytes,
                            segment.tag,
                        ]
                        for segment in classified.nvlink_segments
                    ],
                    "total_group_bytes": classified.nvlink_bytes,
                    "independent_endpoint_ledger": [list(row) for row in independent_ledger],
                    "production_endpoint_ledger": (
                        None
                        if production_ledger is None
                        else [list(row) for row in production_ledger]
                    ),
                    "legacy_source_service_ns": (
                        None if source_services is None else [list(row) for row in source_services]
                    ),
                    "nvlink_service_ps": classified.nvlink_service_ps,
                }
    return cells


def _dims(payload_bytes: int, width: int):
    from simllm.compute import ModelDims

    return ModelDims(
        num_layers=1,
        hidden_size=payload_bytes // 2,
        intermediate_size=8,
        num_heads=1,
        num_kv_heads=1,
        head_size=1,
        vocab_size=16,
        dtype_bytes=2,
        num_experts=width,
        top_k=1,
        moe_intermediate_size=4,
        local_num_experts=1,
    )


def _record(step_index: int, virtual_time_ps: int, width: int):
    from simllm.core import RequestPhase, ScheduledRequest, StepRecord

    return StepRecord(
        step_index=step_index,
        virtual_time_ps=virtual_time_ps,
        scheduled=[
            ScheduledRequest(
                "endpoint-fixture",
                RequestPhase.PREFILL,
                width,
                context_length=width,
            )
        ],
        num_sampled=1,
    )


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


def _live_cell(
    out: Path,
    *,
    payload_bytes: int,
    width: int,
    placement: str,
) -> dict[str, object]:
    from simllm.backends import HtsimStepSink, HtsimStepSinkConfig
    from simllm.compute import ComputeProvider, DurationEstimate

    class FixedProvider(ComputeProvider):
        def estimate(self, kernel, gpu):
            return DurationEstimate(duration_ps=COMPUTE_PS, bound="declared-fixed")

    if placement == "all-local":
        hosts: tuple[str, ...] | None = ("local-node",) * width
    elif placement == "all-remote-explicit":
        hosts = tuple(f"node-{rank}" for rank in range(width))
    elif placement == "all-remote-omitted":
        hosts = None
    else:
        raise ValueError(f"unsupported live placement: {placement}")

    workdir = out / "live" / placement / f"p{payload_bytes}-w{width}"
    sink = HtsimStepSink(
        HtsimStepSinkConfig(
            profile="rnic-nn-fluid",
            tp_ranks=(0,),
            dims=_dims(payload_bytes, width),
            workdir=workdir,
            ep_ranks=tuple(range(width)),
            provider=FixedProvider(),
            placement_manifest=None if hosts is None else _manifest(hosts),
            nvlink_bandwidth_bytes_per_second=NVLINK_BYTES_PER_SECOND,
            num_goal_ranks=width,
        )
    )
    virtual_time_ps = 0
    results = []
    for step_index in range(REPLAY_STEPS):
        result = sink(_record(step_index, virtual_time_ps, width))
        if result is None:
            raise AssertionError("live endpoint fixture produced no StepResult")
        results.append(result)
        virtual_time_ps = result.completed_at_ps
    latencies = tuple(result.step_latency_ps for result in results)
    return {
        "payload_bytes": payload_bytes,
        "ep_width": width,
        "placement": placement,
        "jct_ps": latencies[0],
        "ttft_ps": latencies[0],
        "tpot_ps": sum(latencies[1:]) // len(latencies[1:]),
        "replay_latencies_ps": list(latencies),
        "step_results": [asdict(result) for result in results],
        "network_outcomes": [asdict(outcome) for outcome in sink.outcomes],
        "locality_outcomes": [asdict(locality) for locality in sink.locality_outcomes],
        "artifacts": _artifact_inventory(workdir),
    }


def _observe(out: Path) -> dict[str, object]:
    component = _component_cells()
    local = {}
    remote = {}
    for payload_bytes in PAYLOAD_BYTES:
        for width in EP_WIDTHS:
            key = f"{payload_bytes}:{width}"
            local[key] = _live_cell(
                out,
                payload_bytes=payload_bytes,
                width=width,
                placement="all-local",
            )
            for placement in (
                "all-remote-explicit",
                "all-remote-omitted",
            ):
                remote[f"{key}:{placement}"] = _live_cell(
                    out,
                    payload_bytes=payload_bytes,
                    width=width,
                    placement=placement,
                )
    return {
        "component_cells": component,
        "live_local_cells": local,
        "live_remote_cells": remote,
    }


def _raw_behavioral(
    baseline: dict[str, object],
    corrected: dict[str, object],
) -> dict[str, object]:
    baseline_component = baseline["component_cells"]
    corrected_component = corrected["component_cells"]

    symmetric = []
    dispatch = []
    combine = []
    live_jct = []
    for payload_bytes in PAYLOAD_BYTES:
        for width in EP_WIDTHS:
            for shape, rows in (
                ("symmetric", symmetric),
                ("dispatch-star", dispatch),
            ):
                key = f"{payload_bytes}:{width}:{shape}"
                old = baseline_component[key]["nvlink_service_ps"]
                new = corrected_component[key]["nvlink_service_ps"]
                rows.append(
                    {
                        "payload_bytes": payload_bytes,
                        "ep_width": width,
                        "raw_baseline_service_ps": old,
                        "raw_corrected_service_ps": new,
                        "raw_signed_change_ps": new - old,
                        "expected_signed_change_ps": 0,
                        "passed": new - old == 0,
                    }
                )
            if width in (4, 8):
                key = f"{payload_bytes}:{width}:combine-star"
                old_service = baseline_component[key]["nvlink_service_ps"]
                new_service = corrected_component[key]["nvlink_service_ps"]
                expected_change = EXPECTED_COMBINE_CHANGE_PS[(payload_bytes, width)]
                combine.append(
                    {
                        "payload_bytes": payload_bytes,
                        "ep_width": width,
                        "raw_baseline_service_ps": old_service,
                        "raw_corrected_service_ps": new_service,
                        "raw_signed_change_ps": new_service - old_service,
                        "expected_signed_change_ps": expected_change,
                        "passed": new_service - old_service == expected_change,
                    }
                )
                live_key = f"{payload_bytes}:{width}"
                old_jct = baseline["live_local_cells"][live_key]["jct_ps"]
                new_jct = corrected["live_local_cells"][live_key]["jct_ps"]
                live_jct.append(
                    {
                        "payload_bytes": payload_bytes,
                        "ep_width": width,
                        "raw_baseline_jct_ps": old_jct,
                        "raw_corrected_jct_ps": new_jct,
                        "raw_signed_change_ps": new_jct - old_jct,
                        "expected_signed_change_ps": expected_change,
                        "passed": new_jct - old_jct == expected_change,
                    }
                )

    families = {
        "CORE-B1": {"instances": symmetric},
        "CORE-B2": {"instances": dispatch},
        "CORE-B3": {"instances": combine},
        "CORE-B4": {"instances": live_jct},
    }
    for family in families.values():
        family["passed"] = all(row["passed"] for row in family["instances"])
    return families


def _remote_timing_projection(cell: dict[str, object]) -> dict[str, object]:
    return {
        "jct_ps": cell["jct_ps"],
        "ttft_ps": cell["ttft_ps"],
        "tpot_ps": cell["tpot_ps"],
        "replay_latencies_ps": cell["replay_latencies_ps"],
        "step_results": cell["step_results"],
        "network_outcomes": cell["network_outcomes"],
        "artifacts": cell["artifacts"],
    }


def _expected_component_service(
    mode: str,
    payload_bytes: int,
    width: int,
    shape: str,
) -> int:
    if mode == "baseline" and shape == "combine-star":
        return _service_ps(payload_bytes)
    return EXPECTED_ENDPOINT_SERVICE_PS[(payload_bytes, width)]


def _fatal_checks(
    *,
    mode: str,
    observations: dict[str, object],
    tool_provenance: dict[str, object],
    baseline: dict[str, object] | None,
) -> dict[str, object]:
    component_rows = []
    physical_rows = []
    for payload_bytes in PAYLOAD_BYTES:
        for width in EP_WIDTHS:
            for shape in SHAPES:
                key = f"{payload_bytes}:{width}:{shape}"
                cell = observations["component_cells"][key]
                ledger = cell["independent_endpoint_ledger"]
                egress_sum = sum(row[1] for row in ledger)
                ingress_sum = sum(row[2] for row in ledger)
                expected_total = (
                    width * (width - 1) * payload_bytes
                    if shape == "symmetric"
                    else (width - 1) * payload_bytes
                )
                expected_service = _expected_component_service(
                    mode,
                    payload_bytes,
                    width,
                    shape,
                )
                production_ledger = cell["production_endpoint_ledger"]
                ledger_surface_ok = (
                    production_ledger is None if mode == "baseline" else production_ledger == ledger
                )
                source_surface_ok = (
                    cell["legacy_source_service_ns"] is not None
                    if mode == "baseline"
                    else cell["legacy_source_service_ns"] is None
                )
                component_rows.append(
                    {
                        "payload_bytes": payload_bytes,
                        "ep_width": width,
                        "shape": shape,
                        "egress_sum": egress_sum,
                        "ingress_sum": ingress_sum,
                        "total_group_bytes": cell["total_group_bytes"],
                        "expected_total_group_bytes": expected_total,
                        "observed_service_ps": cell["nvlink_service_ps"],
                        "expected_service_ps": expected_service,
                        "ledger_surface_ok": ledger_surface_ok,
                        "source_surface_ok": source_surface_ok,
                        "passed": egress_sum
                        == ingress_sum
                        == cell["total_group_bytes"]
                        == expected_total
                        and ledger_surface_ok
                        and source_surface_ok
                        and cell["nvlink_service_ps"] == expected_service,
                    }
                )
                if mode == "corrected" and shape == "combine-star":
                    endpoint_bytes = (width - 1) * payload_bytes
                    service_ps = cell["nvlink_service_ps"]
                    floor_numerator = endpoint_bytes * 1_000_000_000_000
                    service_numerator = service_ps * NVLINK_BYTES_PER_SECOND
                    physical_rows.append(
                        {
                            "payload_bytes": payload_bytes,
                            "ep_width": width,
                            "peak_endpoint_bytes": endpoint_bytes,
                            "service_ps": service_ps,
                            "above_floor": service_numerator >= floor_numerator,
                            "below_strict_ceiling": service_numerator
                            < floor_numerator + 1_000 * NVLINK_BYTES_PER_SECOND,
                            "passed": floor_numerator
                            <= service_numerator
                            < floor_numerator + 1_000 * NVLINK_BYTES_PER_SECOND,
                        }
                    )

    live_rows = []
    for payload_bytes in PAYLOAD_BYTES:
        for width in EP_WIDTHS:
            key = f"{payload_bytes}:{width}"
            cell = observations["live_local_cells"][key]
            endpoint_service = EXPECTED_ENDPOINT_SERVICE_PS[(payload_bytes, width)]
            combine_service = _service_ps(payload_bytes) if mode == "baseline" else endpoint_service
            expected_jct = COMPUTE_PS + endpoint_service + combine_service
            locality = cell["locality_outcomes"][0]
            expected_nvlink_service = endpoint_service + combine_service
            replay_equal = cell["replay_latencies_ps"] == [expected_jct] * REPLAY_STEPS
            live_rows.append(
                {
                    "payload_bytes": payload_bytes,
                    "ep_width": width,
                    "observed_jct_ps": cell["jct_ps"],
                    "expected_jct_ps": expected_jct,
                    "observed_nvlink_service_ps": locality["nvlink_service_ps"],
                    "expected_nvlink_service_ps": expected_nvlink_service,
                    "step_results_reached": len(cell["step_results"]),
                    "passed": cell["jct_ps"] == cell["ttft_ps"] == cell["tpot_ps"] == expected_jct
                    and replay_equal
                    and locality["nvlink_service_ps"] == expected_nvlink_service
                    and len(cell["step_results"]) == REPLAY_STEPS,
                }
            )

    remote_identity_rows = []
    for payload_bytes in PAYLOAD_BYTES:
        for width in EP_WIDTHS:
            prefix = f"{payload_bytes}:{width}"
            explicit = observations["live_remote_cells"][f"{prefix}:all-remote-explicit"]
            omitted = observations["live_remote_cells"][f"{prefix}:all-remote-omitted"]
            within_mode_identity = _remote_timing_projection(explicit) == _remote_timing_projection(
                omitted
            )
            baseline_identity = True
            if baseline is not None:
                baseline_identity = all(
                    observations["live_remote_cells"][f"{prefix}:{placement}"]
                    == baseline["live_remote_cells"][f"{prefix}:{placement}"]
                    for placement in (
                        "all-remote-explicit",
                        "all-remote-omitted",
                    )
                )
            quiescent = all(
                outcome["quiescent"]
                for cell in (explicit, omitted)
                for outcome in cell["network_outcomes"]
            )
            remote_identity_rows.append(
                {
                    "payload_bytes": payload_bytes,
                    "ep_width": width,
                    "within_mode_identity": within_mode_identity,
                    "baseline_identity": baseline_identity,
                    "quiescent": quiescent,
                    "passed": within_mode_identity and baseline_identity and quiescent,
                }
            )

    renderer_identity_rows = []
    if baseline is not None:
        for key, cell in observations["component_cells"].items():
            previous = baseline["component_cells"][key]
            renderer_identity_rows.append(
                {
                    "cell": key,
                    "segments_equal": cell["segments"] == previous["segments"],
                    "total_group_bytes_equal": cell["total_group_bytes"]
                    == previous["total_group_bytes"],
                    "independent_ledger_equal": cell["independent_endpoint_ledger"]
                    == previous["independent_endpoint_ledger"],
                    "passed": cell["segments"] == previous["segments"]
                    and cell["total_group_bytes"] == previous["total_group_bytes"]
                    and cell["independent_endpoint_ledger"]
                    == previous["independent_endpoint_ledger"],
                }
            )

    tool_identity = True
    if baseline is not None:
        tool_identity = tool_provenance == baseline["provenance"]["native_tools"]
    checks = {
        "component_exactness_and_conservation": component_rows,
        "physical_sanity": physical_rows,
        "live_metric_exactness": live_rows,
        "all_remote_identity": remote_identity_rows,
        "renderer_identity": renderer_identity_rows,
        "native_tool_identity": tool_identity,
    }
    checks["passed"] = (
        all(row["passed"] for row in component_rows)
        and all(row["passed"] for row in physical_rows)
        and all(row["passed"] for row in live_rows)
        and all(row["passed"] for row in remote_identity_rows)
        and all(row["passed"] for row in renderer_identity_rows)
        and tool_identity
    )
    return checks


def _load_baseline(path: Path) -> tuple[dict[str, object], str]:
    digest = _sha256(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("baseline summary must contain a JSON object")
    if value.get("schema") != "simllm-endpoint-service-study-v1":
        raise ValueError("baseline summary has the wrong schema")
    if value.get("mode") != "baseline":
        raise ValueError("baseline summary has the wrong mode")
    if value.get("run_status") != "baseline-observation":
        raise ValueError("baseline summary was not accepted as an observation")
    if value.get("provenance", {}).get("expectations_commit") != EXPECTATIONS_COMMIT:
        raise ValueError("baseline summary uses a different expectations commit")
    return value, digest


def run_study(args: argparse.Namespace) -> dict[str, object]:
    _check_frozen_registry()
    tool_provenance = _validate_production_args(args)
    baseline_summary = None
    baseline_digest = None
    if args.mode == "corrected":
        assert args.baseline_summary is not None
        baseline_summary, baseline_digest = _load_baseline(args.baseline_summary)

    args.out.mkdir(parents=True, exist_ok=False)
    observations = _jsonable(_observe(args.out))

    behavioral = None
    if baseline_summary is not None:
        behavioral = _raw_behavioral(baseline_summary, observations)
    fatal = _fatal_checks(
        mode=args.mode,
        observations=observations,
        tool_provenance=tool_provenance,
        baseline=baseline_summary,
    )

    behavioral_score = None
    if behavioral is not None and fatal["passed"]:
        passed_families = sum(bool(family["passed"]) for family in behavioral.values())
        passed_instances = sum(
            sum(bool(instance["passed"]) for instance in family["instances"])
            for family in behavioral.values()
        )
        behavioral_score = {
            "passed_families": passed_families,
            "total_families": SCORED_FAMILIES,
            "passed_instances": passed_instances,
            "total_instances": SCORED_INSTANCES,
        }
    if not fatal["passed"]:
        run_status = "void"
    elif args.mode == "baseline":
        run_status = "baseline-observation"
    elif behavioral_score != {
        "passed_families": SCORED_FAMILIES,
        "total_families": SCORED_FAMILIES,
        "passed_instances": SCORED_INSTANCES,
        "total_instances": SCORED_INSTANCES,
    }:
        run_status = "failed"
    else:
        run_status = "accepted"

    summary = {
        "schema": "simllm-endpoint-service-study-v1",
        "mode": args.mode,
        "run_status": run_status,
        "provenance": {
            "expectations_commit": EXPECTATIONS_COMMIT,
            "evidence_authored_against": EVIDENCE_AUTHORED_AGAINST,
            "observed_simllm_revision": _git_revision("HEAD"),
            "observed_htsim_gitlink": _git_revision("HEAD:third_party/htsim"),
            "baseline_summary_sha256": baseline_digest,
            "native_tools": tool_provenance,
            "python": sys.version,
            "platform": platform.platform(),
        },
        "configuration": {
            "ep_widths": list(EP_WIDTHS),
            "payload_bytes": list(PAYLOAD_BYTES),
            "shapes": list(SHAPES),
            "nvlink_bytes_per_second": NVLINK_BYTES_PER_SECOND,
            "nvlink_duplex_model": "full-duplex-max-egress-ingress",
            "rejected_alternative": "shared-half-duplex-egress-plus-ingress",
            "compute_ps": COMPUTE_PS,
            "replay_steps": REPLAY_STEPS,
            "remote_profile": "rnic-nn-fluid",
        },
        **observations,
        "behavioral": behavioral,
        "behavioral_score": behavioral_score,
        "fatal_unscored": fatal,
        "entailment": (
            "CORE-B1 through CORE-B4 were evaluated from raw baseline and "
            "corrected component services and StepResult JCT values before "
            "exact service, endpoint ledger, conservation, physical bound, "
            "renderer, artifact, timestamp or native-tool identity checks."
        ),
        "dependency_authority_refreeze": DEPENDENCY_AUTHORITY_REFREEZE,
        "seqgen_safe_endpoint_floors_ps": SEQGEN_FLOORS_PS,
    }
    _write_json(args.out / "summary.json", summary)
    if run_status in {"void", "failed"}:
        raise RuntimeError(f"endpoint-service study status is {run_status}")
    return summary


def main() -> None:
    args = parse_args()
    if args.check_only:
        check_only(args)
        return
    summary = run_study(args)
    print(
        json.dumps(
            {
                "mode": summary["mode"],
                "run_status": summary["run_status"],
                "behavioral_score": summary["behavioral_score"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
