"""Run the frozen calibrated collective-latency-floor study."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict
from itertools import pairwise
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

PROFILE = "b200-nccl-2.27-local-v1"
EFFECTIVE_BANDWIDTH_BYTES_PER_SECOND = 70_027_079_100
PARTICIPANT_LATENCY_PS = {
    2: 10_722_112,
    4: 15_745_167,
    8: 30_128_029,
}
PROPAGATION_REFERENCE_PS = 2_000_000
COLLECTIVES_PER_STEP = 48
EXPECTED_STEP_FLOOR_PS = 1_446_145_392
HELD_OUT_ROWS = (
    (2, 4_096, 4_096, 10_520_000, 10_780_604, 1_052_000),
    (4, 4_096, 6_144, 16_180_000, 15_832_905, 1_618_000),
    (8, 4_096, 7_168, 30_310_000, 30_230_390, 3_031_000),
)
SENSITIVITY_ENDPOINT_BYTES = 200_704
SENSITIVITY_TRANSPORT_PS = {
    400_000_000_000: 6_014_081,
    200_000_000_000: 10_028_161,
}
EXPECTATIONS_COMMIT = "3a6126e174e859d5c222e137dc9e2d94ead6db29"
EXPECTATIONS_SHA256 = "986c5952099bb6200736618a33580217b58ab3bbf9cb9ec97df1efc8b6962a28"
STUDY_ATTEMPT = 2
PRIOR_VOID_REVISION = "cec7109adeb9656de92f9ef5ea54572accdc3208"
SOURCE_ZIP_SHA256 = "91629a3b4a6eff4ac2e8bbc2261a928dbcca42f07c02d7f1fe15f9d981d0713f"
SOURCE_LOCAL_TSV_SHA256 = "639348d43e625d8b7199c45db29ec7a848165974142016fab7b85ca34564a8f3"
SOURCE_URL = "https://github.com/user-attachments/files/21326711/nccl-test-result.zip"
SOURCE_ISSUE_URL = "https://github.com/NVIDIA/nccl-tests/issues/333"
SOURCE_SYSTEM_URL = (
    "https://docs.nvidia.com/dgx/dgxb200-user-guide/introduction-to-dgxb200.html"
)
PHYSICAL_BANDWIDTH_BYTES_PER_SECOND = 900_000_000_000
EXPECTED_SCORED_FAMILIES = 4
EXPECTED_HELDOUT_INSTANCES = 3
PREFILL_NEW_TOKENS = 32
DECODE_NEW_TOKENS = 1
MODEL_LAYERS = 24
MODEL_HIDDEN_SIZE = 1_024
MODEL_DTYPE_BYTES = 2
MODEL_EXPERTS = 32
MODEL_TOP_K = 8
MODEL_MOE_INTERMEDIATE_SIZE = 512
MODEL_LOCAL_EXPERTS = 4
RUN_ROOT_ENV = "SIMLLM_COLLECTIVE_FLOOR_RUN_ROOT"


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def _prediction_ps(width: int, endpoint_bytes: int) -> int:
    return PARTICIPANT_LATENCY_PS[width] + _ceil_div(
        endpoint_bytes * 1_000_000_000_000,
        EFFECTIVE_BANDWIDTH_BYTES_PER_SECOND,
    )


def check_only(args: argparse.Namespace) -> None:
    """Validate frozen inputs and arithmetic without producing artifacts."""

    if not args.htsim_rnic.is_file() or not os.access(args.htsim_rnic, os.X_OK):
        raise SystemExit(f"htsim_rnic is missing or not executable: {args.htsim_rnic}")
    txt2bin = os.environ.get("SIMLLM_TXT2BIN")
    if txt2bin is not None:
        converter = Path(txt2bin)
        if not converter.is_file() or not os.access(converter, os.X_OK):
            raise SystemExit(f"SIMLLM_TXT2BIN is not executable: {converter}")
    if args.run_dir.exists():
        raise SystemExit(f"check-only run directory must not exist: {args.run_dir}")
    if tuple(PARTICIPANT_LATENCY_PS) != (2, 4, 8):
        raise AssertionError("participant calibration widths changed")
    if EFFECTIVE_BANDWIDTH_BYTES_PER_SECOND >= 900_000_000_000:
        raise AssertionError("effective bandwidth must stay below the physical bound")
    for width, payload_bytes, endpoint_bytes, observed, expected, allowed in HELD_OUT_ROWS:
        if endpoint_bytes != 2 * (width - 1) * payload_bytes // width:
            raise AssertionError("held-out endpoint-byte relation changed")
        predicted = _prediction_ps(width, endpoint_bytes)
        if predicted != expected:
            raise AssertionError("held-out integer prediction changed")
        if allowed != max(1_000_000, observed // 10):
            raise AssertionError("held-out acceptance tolerance changed")
        if abs(predicted - observed) > allowed:
            raise AssertionError("a frozen held-out prediction misses its bar")
        physical_floor_ps = _ceil_div(endpoint_bytes * 1_000_000_000_000, 900_000_000_000)
        if not physical_floor_ps < predicted < 50_000_000:
            raise AssertionError("held-out physical sanity envelope changed")
    if COLLECTIVES_PER_STEP * PARTICIPANT_LATENCY_PS[8] != EXPECTED_STEP_FLOOR_PS:
        raise AssertionError("flagship collective-count arithmetic changed")
    fast = SENSITIVITY_TRANSPORT_PS[400_000_000_000]
    slow = SENSITIVITY_TRANSPORT_PS[200_000_000_000]
    if 2 * fast - slow != PROPAGATION_REFERENCE_PS + 1:
        raise AssertionError("frozen propagation cancellation changed")
    if slow - fast != SENSITIVITY_ENDPOINT_BYTES * 20:
        raise AssertionError("frozen bandwidth sensitivity changed")
    enabled_ratio = (slow + PARTICIPANT_LATENCY_PS[8]) / (
        fast + PARTICIPANT_LATENCY_PS[8]
    )
    if not 1.10 <= enabled_ratio <= 1.12:
        raise AssertionError("enabled sensitivity ratio left its frozen band")
    print(
        f"check-only attempt={STUDY_ATTEMPT}; profile={PROFILE}; "
        f"run-dir={args.run_dir}; validated "
        "frozen calibration and produced no artifacts"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(
        ("git", *args),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_clean_worktree() -> None:
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError(
            "production evidence requires a clean worktree so its revision "
            "identifies the executed source"
        )


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _artifact_manifest(workdir: Path) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "name": path.relative_to(workdir).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(workdir.rglob("*"))
        if path.is_file()
    )


def _dims() -> Any:
    from simllm.compute import ModelDims

    return ModelDims(
        num_layers=MODEL_LAYERS,
        hidden_size=MODEL_HIDDEN_SIZE,
        intermediate_size=MODEL_MOE_INTERMEDIATE_SIZE,
        num_heads=16,
        num_kv_heads=8,
        head_size=64,
        vocab_size=49_152,
        dtype_bytes=MODEL_DTYPE_BYTES,
        num_experts=MODEL_EXPERTS,
        top_k=MODEL_TOP_K,
        moe_intermediate_size=MODEL_MOE_INTERMEDIATE_SIZE,
        local_num_experts=MODEL_LOCAL_EXPERTS,
    )


def _sensitivity_dims() -> Any:
    dims = _dims()
    return dims.__class__(
        num_layers=1,
        hidden_size=dims.hidden_size,
        intermediate_size=dims.intermediate_size,
        num_heads=dims.num_heads,
        num_kv_heads=dims.num_kv_heads,
        head_size=dims.head_size,
        vocab_size=dims.vocab_size,
        dtype_bytes=dims.dtype_bytes,
        num_experts=dims.num_experts,
        top_k=dims.top_k,
        moe_intermediate_size=dims.moe_intermediate_size,
        local_num_experts=dims.local_num_experts,
    )


def _record(step_index: int, virtual_time_ps: int, new_tokens: int) -> Any:
    from simllm.core import RequestPhase, ScheduledRequest, StepRecord

    return StepRecord(
        step_index=step_index,
        virtual_time_ps=virtual_time_ps,
        scheduled=[
            ScheduledRequest(
                request_id="fixture",
                phase=(RequestPhase.PREFILL if step_index == 0 else RequestPhase.DECODE),
                num_new_tokens=new_tokens,
                context_length=64 + step_index,
            )
        ],
        num_sampled=1,
    )


def _sink_cell(
    args: argparse.Namespace,
    *,
    name: str,
    linkspeed_bps: int,
    collective_latency_profile: object = None,
    placement_hosts: tuple[str, ...] | None = None,
    step_specs: tuple[tuple[int, int], ...] | None = None,
    dims: object | None = None,
) -> dict[str, object]:
    from simllm.backends import HtsimStepSink, HtsimStepSinkConfig
    from simllm.placement import PlacementManifest, RankPlacement

    workdir = args.run_dir / "cells" / name
    placement = (
        PlacementManifest(
            ranks=[
                RankPlacement(global_rank=rank, hostname=host, local_rank=rank)
                for rank, host in enumerate(placement_hosts)
            ]
        )
        if placement_hosts is not None
        else None
    )
    sink = HtsimStepSink(
        HtsimStepSinkConfig(
            profile="rnic-nn-fluid",
            tp_ranks=(0,),
            ep_ranks=tuple(range(8)),
            dims=_dims() if dims is None else dims,
            workdir=workdir,
            linkspeed_bps=linkspeed_bps,
            placement_manifest=placement,
            collective_latency_profile=collective_latency_profile,
        )
    )
    selected_steps = step_specs or (
        (0, PREFILL_NEW_TOKENS),
        (1, DECODE_NEW_TOKENS),
        (2, DECODE_NEW_TOKENS),
    )
    results = []
    virtual_time_ps = 0
    for step_index, new_tokens in selected_steps:
        record = _record(step_index, virtual_time_ps, new_tokens)
        result = sink(record)
        if result is None:
            raise AssertionError("registered fixture unexpectedly has no collective work")
        results.append(result)
        virtual_time_ps = result.completed_at_ps
    return {
        "name": name,
        "linkspeed_bps": linkspeed_bps,
        "enabled": bool(sink.collective_timing_outcomes),
        "step_results": [asdict(result) for result in results],
        "network_outcomes": [asdict(outcome) for outcome in sink.outcomes],
        "locality_outcomes": [asdict(outcome) for outcome in sink.locality_outcomes],
        "collective_timing_outcomes": [
            asdict(outcome) for outcome in sink.collective_timing_outcomes
        ],
        "artifact_manifest": _artifact_manifest(workdir),
    }


def _local_width_cell(
    args: argparse.Namespace,
    *,
    width: int,
    enabled: bool,
) -> dict[str, object]:
    from simllm.backends import HtsimStepSink, HtsimStepSinkConfig
    from simllm.compute import ComputeProvider, DurationEstimate, ModelDims
    from simllm.core import RequestPhase, ScheduledRequest, StepRecord
    from simllm.placement import PlacementManifest, RankPlacement
    from simllm.traffic import B200_NCCL_2_27_LOCAL_PROFILE

    class FixedProvider(ComputeProvider):
        def estimate(self, kernel: object, gpu: object) -> DurationEstimate:
            return DurationEstimate(duration_ps=2_000, bound="measured")

    dims = ModelDims(
        num_layers=1,
        hidden_size=4_096,
        intermediate_size=8_192,
        num_heads=8,
        num_kv_heads=8,
        head_size=512,
        vocab_size=256,
        dtype_bytes=1,
    )
    manifest = PlacementManifest(
        ranks=[RankPlacement(rank, "node", rank) for rank in range(width)]
    )
    name = f"local-w{width}-{'enabled' if enabled else 'off'}"
    workdir = args.run_dir / "cells" / name
    sink = HtsimStepSink(
        HtsimStepSinkConfig(
            profile="rnic-nn-fluid",
            tp_ranks=tuple(range(width)),
            dims=dims,
            workdir=workdir,
            placement_manifest=manifest,
            provider=FixedProvider(),
            collective_latency_profile=(
                B200_NCCL_2_27_LOCAL_PROFILE if enabled else None
            ),
        )
    )
    result = sink(
        StepRecord(
            step_index=0,
            virtual_time_ps=0,
            scheduled=[
                ScheduledRequest(
                    "width",
                    RequestPhase.DECODE,
                    num_new_tokens=1,
                    context_length=8,
                )
            ],
        )
    )
    if result is None:
        raise AssertionError("width fixture unexpectedly produced no result")
    locality = sink.locality_outcomes[0]
    timing = (
        sink.collective_timing_outcomes[0]
        if sink.collective_timing_outcomes
        else None
    )
    collective_ids = (
        {
            row.collective_operation_id
            for row in timing.artifacts
            if row.collective_operation_id is not None
        }
        if timing is not None
        else set()
    )
    return {
        "name": name,
        "width": width,
        "enabled": enabled,
        "step_latency_ps": result.step_latency_ps,
        "collective_count": (
            len(collective_ids) if timing is not None else 2
        ),
        "base_total_ps": (
            sum(row.collective_base_latency_ps for row in timing.artifacts)
            if timing is not None
            else 0
        ),
        "positive_base_charges": (
            sum(
                row.collective_base_latency_ps > 0
                for row in timing.artifacts
            )
            if timing is not None
            else 0
        ),
        "nvlink_service_ps": locality.nvlink_service_ps,
        "nvlink_bandwidth_bytes_per_second": (
            locality.nvlink_bandwidth_bytes_per_second
        ),
        "fabric_directed_bytes": locality.fabric_directed_bytes,
        "artifact_manifest": _artifact_manifest(workdir),
    }


def _validate_production_args(args: argparse.Namespace) -> dict[str, object]:
    _require_clean_worktree()
    configured_root = os.environ.get(RUN_ROOT_ENV)
    if not configured_root:
        raise RuntimeError(f"{RUN_ROOT_ENV} must name this branch's external run root")
    run_root = Path(configured_root).resolve()
    try:
        args.run_dir.resolve().relative_to(run_root)
    except ValueError as exc:
        raise ValueError(f"study output must remain under {RUN_ROOT_ENV}") from exc
    if args.run_dir.exists():
        raise FileExistsError(f"study output already exists: {args.run_dir}")
    if not args.htsim_rnic.is_file() or not os.access(args.htsim_rnic, os.X_OK):
        raise FileNotFoundError(
            f"htsim_rnic is not an executable file: {args.htsim_rnic}"
        )
    txt2bin_value = os.environ.get("SIMLLM_TXT2BIN")
    if not txt2bin_value:
        raise RuntimeError("SIMLLM_TXT2BIN must name the registered converter")
    txt2bin = Path(txt2bin_value)
    if not txt2bin.is_file() or not os.access(txt2bin, os.X_OK):
        raise FileNotFoundError(f"SIMLLM_TXT2BIN is not executable: {txt2bin}")
    expectations_path = Path(__file__).with_name("expectations.md")
    if _sha256(expectations_path) != EXPECTATIONS_SHA256:
        raise RuntimeError("expectations.md differs from the final freeze")
    frozen_expectations = _git(
        "show",
        f"{EXPECTATIONS_COMMIT}:examples/collective_latency_floor_v1/expectations.md",
    )
    if frozen_expectations != expectations_path.read_text(encoding="utf-8").rstrip(
        "\n"
    ):
        raise RuntimeError("the checked-out expectations do not match their freeze commit")
    os.environ["SIMLLM_HTSIM_RNIC"] = str(args.htsim_rnic.resolve())
    return {
        "simllm_revision": _git("rev-parse", "HEAD"),
        "expectations_commit": EXPECTATIONS_COMMIT,
        "expectations_sha256": EXPECTATIONS_SHA256,
        "htsim_gitlink": _git("rev-parse", "HEAD:third_party/htsim"),
        "htsim_rnic_sha256": _sha256(args.htsim_rnic),
        "txt2bin_sha256": _sha256(txt2bin),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def _step_latencies(cell: dict[str, Any]) -> tuple[int, ...]:
    return tuple(row["step_latency_ps"] for row in cell["step_results"])


def _collective_rows(cell: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    timing = cell["collective_timing_outcomes"]
    if len(timing) != len(cell["step_results"]):
        return ()
    return tuple(
        artifact
        for step in timing
        for artifact in step["artifacts"]
        if artifact["collective_operation_id"] is not None
    )


def _first_fabric_collective_service(cell: dict[str, Any]) -> int:
    locality = cell["locality_outcomes"][0]
    for operation_ids, service_ps in zip(
        locality["artifact_operation_ids"],
        locality["fabric_phase_service_ps"],
        strict=True,
    ):
        if any(":ep-" in operation_id for operation_id in operation_ids):
            return service_ps
    raise AssertionError(f"cell {cell['name']!r} has no fabric collective artifact")


def _first_enabled_collective_service(cell: dict[str, Any]) -> int:
    for row in _collective_rows(cell):
        if row["fabric_transport_ps"]:
            return row["composed_service_ps"]
    raise AssertionError(f"cell {cell['name']!r} has no enabled fabric collective")


def _behavioral_results(
    *,
    heldout: tuple[dict[str, object], ...],
    local_rows: tuple[dict[str, object], ...],
    sensitivity_off_fast: dict[str, Any],
    sensitivity_off_slow: dict[str, Any],
    sensitivity_enabled_fast: dict[str, Any],
    sensitivity_enabled_slow: dict[str, Any],
    flagship_off: dict[str, Any],
    flagship_enabled: dict[str, Any],
) -> dict[str, dict[str, object]]:
    c1 = {
        "instances": heldout,
        "passed": all(
            row["absolute_error_ps"] <= row["allowed_error_ps"]
            for row in heldout
        ),
    }
    local_by_width = {(row["width"], row["enabled"]): row for row in local_rows}
    enabled_latencies = [
        local_by_width[(width, True)]["step_latency_ps"] for width in (2, 4, 8)
    ]
    deltas = [
        local_by_width[(width, True)]["step_latency_ps"]
        - local_by_width[(width, False)]["step_latency_ps"]
        for width in (2, 4, 8)
    ]
    c2 = {
        "widths": [2, 4, 8],
        "enabled_step_latency_ps": enabled_latencies,
        "enabled_minus_off_ps": deltas,
        "passed": enabled_latencies[0] < enabled_latencies[1] < enabled_latencies[2]
        and deltas[0] < deltas[1] < deltas[2],
    }
    off_fast = _first_fabric_collective_service(sensitivity_off_fast)
    off_slow = _first_fabric_collective_service(sensitivity_off_slow)
    enabled_fast = _first_enabled_collective_service(sensitivity_enabled_fast)
    enabled_slow = _first_enabled_collective_service(sensitivity_enabled_slow)
    off_ratio = off_slow / off_fast
    enabled_ratio = enabled_slow / enabled_fast
    c3 = {
        "off_transport_ps": [off_fast, off_slow],
        "enabled_service_ps": [enabled_fast, enabled_slow],
        "off_ratio": off_ratio,
        "enabled_ratio": enabled_ratio,
        "passed": 1.10 <= enabled_ratio <= 1.12
        and abs(enabled_ratio - 1.0) < abs(off_ratio - 1.0),
    }
    off_steps = _step_latencies(flagship_off)
    enabled_steps = _step_latencies(flagship_enabled)
    ttft_off = off_steps[0]
    ttft_enabled = enabled_steps[0]
    tpot_off = sum(off_steps[1:]) // len(off_steps[1:])
    tpot_enabled = sum(enabled_steps[1:]) // len(enabled_steps[1:])
    ttft_delta = ttft_enabled - ttft_off
    tpot_delta = tpot_enabled - tpot_off
    c4 = {
        "ttft_off_ps": ttft_off,
        "ttft_enabled_ps": ttft_enabled,
        "tpot_off_ps": tpot_off,
        "tpot_enabled_ps": tpot_enabled,
        "ttft_delta_ps": ttft_delta,
        "tpot_delta_ps": tpot_delta,
        "ttft_relative_increase": ttft_delta / ttft_off,
        "tpot_relative_increase": tpot_delta / tpot_off,
        "passed": ttft_enabled > ttft_off
        and tpot_enabled > tpot_off
        and ttft_delta == EXPECTED_STEP_FLOOR_PS
        and tpot_delta == EXPECTED_STEP_FLOOR_PS
        and ttft_delta / ttft_off < tpot_delta / tpot_off,
    }
    return {"C1": c1, "C2": c2, "C3": c3, "C4": c4}


def _unsupported_width_guard(args: argparse.Namespace) -> dict[str, object]:
    from simllm.backends import HtsimStepSink, HtsimStepSinkConfig
    from simllm.compute import ModelDims
    from simllm.core import RequestPhase, ScheduledRequest, StepRecord
    from simllm.traffic import B200_NCCL_2_27_LOCAL_PROFILE

    workdir = args.run_dir / "cells" / "unsupported-width"
    dims = ModelDims(
        num_layers=1,
        hidden_size=4_096,
        intermediate_size=8_192,
        num_heads=8,
        num_kv_heads=8,
        head_size=512,
        vocab_size=256,
        dtype_bytes=1,
    )
    sink = HtsimStepSink(
        HtsimStepSinkConfig(
            profile="rnic-nn-fluid",
            tp_ranks=(0, 1, 2),
            dims=dims,
            workdir=workdir,
            collective_latency_profile=B200_NCCL_2_27_LOCAL_PROFILE,
        )
    )
    message = None
    try:
        sink(
            StepRecord(
                0,
                0,
                [
                    ScheduledRequest(
                        "unsupported",
                        RequestPhase.DECODE,
                        num_new_tokens=1,
                        context_length=8,
                    )
                ],
            )
        )
    except ValueError as exc:
        message = str(exc)
    files = tuple(path.relative_to(workdir).as_posix() for path in workdir.rglob("*") if path.is_file())
    passed = (
        message is not None
        and "participant count 3" in message
        and not files
        and not sink.outcomes
        and not sink.locality_outcomes
        and not sink.collective_timing_outcomes
    )
    return {"passed": passed, "message": message, "files": files}


def _goal_only_manifest(cell: dict[str, Any]) -> tuple[dict[str, object], ...]:
    return tuple(
        row for row in cell["artifact_manifest"] if row["name"].endswith(".goal")
    )


def _identity_payload(cell: dict[str, Any]) -> dict[str, object]:
    return {
        "step_results": cell["step_results"],
        "network_outcomes": cell["network_outcomes"],
        "locality_outcomes": cell["locality_outcomes"],
        "artifact_manifest": cell["artifact_manifest"],
    }


def _locality_without_authority(cell: dict[str, Any]) -> tuple[dict[str, object], ...]:
    rows = []
    for locality in cell["locality_outcomes"]:
        projected = dict(locality)
        projected.pop("authority")
        rows.append(projected)
    return tuple(rows)


def _network_without_composed_makespan(
    cell: dict[str, Any],
) -> tuple[dict[str, object], ...]:
    rows = []
    for outcome in cell["network_outcomes"]:
        projected = dict(outcome)
        projected.pop("makespan_ps")
        rows.append(projected)
    return tuple(rows)


def _fatal_guards(
    *,
    flagship_off: dict[str, Any],
    flagship_legacy: dict[str, Any],
    flagship_enabled: dict[str, Any],
    sensitivity_off_fast: dict[str, Any],
    sensitivity_off_slow: dict[str, Any],
    sensitivity_enabled_fast: dict[str, Any],
    sensitivity_enabled_slow: dict[str, Any],
    explicit_off_fast: dict[str, Any],
    explicit_enabled_fast: dict[str, Any],
    local_rows: tuple[dict[str, object], ...],
    unsupported: dict[str, object],
) -> dict[str, bool]:
    from simllm.traffic import B200_NCCL_2_27_LOCAL_PROFILE

    profile = B200_NCCL_2_27_LOCAL_PROFILE
    default_legacy_identity = _identity_payload(flagship_off) == _identity_payload(
        flagship_legacy
    )
    off_remote_identity = (
        sensitivity_off_fast["step_results"] == explicit_off_fast["step_results"]
        and sensitivity_off_fast["network_outcomes"]
        == explicit_off_fast["network_outcomes"]
        and _locality_without_authority(sensitivity_off_fast)
        == _locality_without_authority(explicit_off_fast)
        and sensitivity_off_fast["artifact_manifest"]
        == explicit_off_fast["artifact_manifest"]
    )
    enabled_remote_identity = (
        sensitivity_enabled_fast["step_results"]
        == explicit_enabled_fast["step_results"]
        and sensitivity_enabled_fast["network_outcomes"]
        == explicit_enabled_fast["network_outcomes"]
        and sensitivity_enabled_fast["collective_timing_outcomes"]
        == explicit_enabled_fast["collective_timing_outcomes"]
        and _locality_without_authority(sensitivity_enabled_fast)
        == _locality_without_authority(explicit_enabled_fast)
        and sensitivity_enabled_fast["artifact_manifest"]
        == explicit_enabled_fast["artifact_manifest"]
    )
    active_cells = (
        flagship_enabled,
        sensitivity_enabled_fast,
        sensitivity_enabled_slow,
        explicit_enabled_fast,
    )
    charge_once = True
    field_equations = True
    report_fields = True
    step_sums = True
    operation_inventory = True
    for cell in active_cells:
        if len(cell["collective_timing_outcomes"]) != len(cell["step_results"]):
            report_fields = False
            continue
        for result, timing in zip(
            cell["step_results"],
            cell["collective_timing_outcomes"],
            strict=True,
        ):
            report_fields = report_fields and (
                timing["profile_id"] == profile.profile_id
                and timing["bandwidth_bytes_per_second"]
                == profile.bandwidth_bytes_per_second
                and tuple(tuple(row) for row in timing["participant_latency_ps"])
                == profile.participant_latency_ps
                and timing["propagation_reference_ps"]
                == profile.propagation_reference_ps
            )
            grouped: dict[str, list[dict[str, Any]]] = {}
            for row in timing["artifacts"]:
                field_equations = field_equations and row["composed_service_ps"] == (
                    row["collective_base_latency_ps"]
                    + max(row["local_service_ps"], row["fabric_transport_ps"])
                )
                operation_id = row["collective_operation_id"]
                if operation_id is not None:
                    grouped.setdefault(operation_id, []).append(row)
            for rows in grouped.values():
                width = rows[0]["participant_count"]
                endpoint_bytes = rows[0]["critical_endpoint_bytes"]
                observed_bases = [row["collective_base_latency_ps"] for row in rows]
                charge_once = charge_once and all(
                    row["participant_count"] == width
                    and row["critical_endpoint_bytes"] == endpoint_bytes
                    for row in rows
                )
                if endpoint_bytes:
                    charge_once = charge_once and (
                        sum(observed_bases) == profile.base_latency_ps(width)
                        and sum(value > 0 for value in observed_bases) == 1
                    )
                else:
                    charge_once = charge_once and not any(observed_bases)
            step_sums = step_sums and sum(
                row["composed_service_ps"] for row in timing["artifacts"]
            ) == result["step_latency_ps"]
            operation_inventory = operation_inventory and len(grouped) == (
                COLLECTIVES_PER_STEP if cell is flagship_enabled else 2
            )
    fast_off = _first_fabric_collective_service(sensitivity_off_fast)
    slow_off = _first_fabric_collective_service(sensitivity_off_slow)
    fast_enabled_transport = _first_fabric_collective_service(
        sensitivity_enabled_fast
    )
    slow_enabled_transport = _first_fabric_collective_service(
        sensitivity_enabled_slow
    )
    fast_enabled = _first_enabled_collective_service(sensitivity_enabled_fast)
    slow_enabled = _first_enabled_collective_service(sensitivity_enabled_slow)
    sensitivity_rows = (
        _collective_rows(sensitivity_enabled_fast),
        _collective_rows(sensitivity_enabled_slow),
    )
    sensitivity_fixture_identity = (
        all(len(rows) == 2 for rows in sensitivity_rows)
        and all(
            row["participant_count"] == 8
            and row["critical_endpoint_bytes"] == SENSITIVITY_ENDPOINT_BYTES
            for rows in sensitivity_rows
            for row in rows
        )
        and fast_off == SENSITIVITY_TRANSPORT_PS[400_000_000_000]
        and slow_off == SENSITIVITY_TRANSPORT_PS[200_000_000_000]
        and fast_enabled_transport == SENSITIVITY_TRANSPORT_PS[400_000_000_000]
        and slow_enabled_transport == SENSITIVITY_TRANSPORT_PS[200_000_000_000]
        and slow_off - fast_off == 4_014_080
        and slow_enabled_transport - fast_enabled_transport == 4_014_080
    )
    propagation_cancellation = (
        2 * fast_off - slow_off == PROPAGATION_REFERENCE_PS + 1
        and 2 * fast_enabled_transport - slow_enabled_transport
        == PROPAGATION_REFERENCE_PS + 1
        and 2 * fast_enabled - slow_enabled - profile.base_latency_ps(8)
        == PROPAGATION_REFERENCE_PS + 1
    )
    byte_conservation = all(
        locality["fabric_directed_bytes"] + locality["nvlink_directed_bytes"]
        == locality["total_directed_bytes"]
        for cell in (
            flagship_off,
            flagship_enabled,
            sensitivity_off_fast,
            sensitivity_off_slow,
            sensitivity_enabled_fast,
            sensitivity_enabled_slow,
        )
        for locality in cell["locality_outcomes"]
    ) and all(row["fabric_directed_bytes"] == 0 for row in local_rows)
    local_charge_once = all(
        (
            row["base_total_ps"]
            == 2 * profile.base_latency_ps(row["width"])
            and row["positive_base_charges"] == 2
            and row["collective_count"] == 2
            and row["nvlink_bandwidth_bytes_per_second"]
            == profile.bandwidth_bytes_per_second
        )
        if row["enabled"]
        else (
            row["base_total_ps"] == 0
            and row["positive_base_charges"] == 0
        )
        for row in local_rows
    )
    quiescent_positive = all(
        outcome["quiescent"]
        for cell in (
            flagship_off,
            flagship_enabled,
            sensitivity_off_fast,
            sensitivity_off_slow,
            sensitivity_enabled_fast,
            sensitivity_enabled_slow,
        )
        for outcome in cell["network_outcomes"]
    )
    for cell in (flagship_off, flagship_enabled):
        completions = [row["completed_at_ps"] for row in cell["step_results"]]
        quiescent_positive = quiescent_positive and all(
            result["step_latency_ps"] > 0
            and result["completed_at_ps"] >= result["step_latency_ps"]
            for result in cell["step_results"]
        ) and all(
            later > earlier
            for earlier, later in pairwise(completions)
        )
    goal_identity = all(
        _goal_only_manifest(off) == _goal_only_manifest(enabled)
        for off, enabled in (
            (flagship_off, flagship_enabled),
            (sensitivity_off_fast, sensitivity_enabled_fast),
            (sensitivity_off_slow, sensitivity_enabled_slow),
        )
    )
    backend_artifact_identity = all(
        off["artifact_manifest"] == enabled["artifact_manifest"]
        for off, enabled in (
            (flagship_off, flagship_enabled),
            (sensitivity_off_fast, sensitivity_enabled_fast),
            (sensitivity_off_slow, sensitivity_enabled_slow),
        )
    )
    backend_outcome_identity = all(
        _network_without_composed_makespan(off)
        == _network_without_composed_makespan(enabled)
        for off, enabled in (
            (flagship_off, flagship_enabled),
            (sensitivity_off_fast, sensitivity_enabled_fast),
            (sensitivity_off_slow, sensitivity_enabled_slow),
        )
    )
    parameter_identity = (
        profile.profile_id == PROFILE
        and profile.bandwidth_bytes_per_second
        == EFFECTIVE_BANDWIDTH_BYTES_PER_SECOND
        and dict(profile.participant_latency_ps) == PARTICIPANT_LATENCY_PS
        and profile.propagation_reference_ps == PROPAGATION_REFERENCE_PS
        and SOURCE_URL.endswith("nccl-test-result.zip")
        and SOURCE_ISSUE_URL.endswith("/333")
        and "dgxb200" in SOURCE_SYSTEM_URL
        and SOURCE_ZIP_SHA256
        == "91629a3b4a6eff4ac2e8bbc2261a928dbcca42f07c02d7f1fe15f9d981d0713f"
        and SOURCE_LOCAL_TSV_SHA256
        == "639348d43e625d8b7199c45db29ec7a848165974142016fab7b85ca34564a8f3"
    )
    return {
        "default_and_explicit_legacy_identity": default_legacy_identity,
        "all_remote_identity_off": off_remote_identity,
        "all_remote_identity_enabled": enabled_remote_identity,
        "enabled_goal_byte_identity": goal_identity,
        "enabled_backend_artifact_identity": backend_artifact_identity,
        "enabled_backend_outcome_identity": backend_outcome_identity,
        "one_base_charge_per_semantic_collective": charge_once
        and local_charge_once,
        "artifact_field_equations": field_equations,
        "step_service_conservation": step_sums,
        "run_record_decomposition": report_fields,
        "sensitivity_fixture_identity": sensitivity_fixture_identity,
        "propagation_rate_cancellation": propagation_cancellation,
        "unsupported_width_preflight": bool(unsupported["passed"]),
        "operation_inventory": operation_inventory,
        "byte_conservation": byte_conservation,
        "backend_quiescence_and_positive_time": quiescent_positive,
        "source_and_parameter_identity": parameter_identity,
    }


def run_study(args: argparse.Namespace) -> dict[str, object]:
    provenance = _validate_production_args(args)
    args.run_dir.mkdir(parents=True, exist_ok=False)

    from simllm.traffic import (
        B200_NCCL_2_27_LOCAL_PROFILE,
        LEGACY_COLLECTIVE_LATENCY_PROFILE,
    )

    profile = B200_NCCL_2_27_LOCAL_PROFILE
    heldout = tuple(
        {
            "width": width,
            "payload_bytes": payload_bytes,
            "endpoint_bytes": endpoint_bytes,
            "physical_floor_ps": _ceil_div(
                endpoint_bytes * 1_000_000_000_000,
                PHYSICAL_BANDWIDTH_BYTES_PER_SECOND,
            ),
            "physical_ceiling_ps": 50_000_000,
            "observed_ps": observed_ps,
            "predicted_ps": profile.total_service_ps(width, endpoint_bytes),
            "absolute_error_ps": abs(
                profile.total_service_ps(width, endpoint_bytes) - observed_ps
            ),
            "allowed_error_ps": allowed_ps,
        }
        for (
            width,
            payload_bytes,
            endpoint_bytes,
            observed_ps,
            _expected_ps,
            allowed_ps,
        ) in HELD_OUT_ROWS
    )
    local_rows = tuple(
        _local_width_cell(args, width=width, enabled=enabled)
        for width in (2, 4, 8)
        for enabled in (False, True)
    )
    flagship_off = _sink_cell(
        args,
        name="flagship-off",
        linkspeed_bps=400_000_000_000,
    )
    flagship_legacy = _sink_cell(
        args,
        name="flagship-legacy",
        linkspeed_bps=400_000_000_000,
        collective_latency_profile=LEGACY_COLLECTIVE_LATENCY_PROFILE,
    )
    flagship_enabled = _sink_cell(
        args,
        name="flagship-enabled",
        linkspeed_bps=400_000_000_000,
        collective_latency_profile=profile,
    )
    sensitivity_dims = _sensitivity_dims()
    sensitivity_cells = {
        (enabled, rate): _sink_cell(
            args,
            name=f"sensitivity-{'enabled' if enabled else 'off'}-{rate // 1_000_000_000}g",
            linkspeed_bps=rate,
            collective_latency_profile=profile if enabled else None,
            step_specs=((0, 14),),
            dims=sensitivity_dims,
        )
        for enabled in (False, True)
        for rate in (400_000_000_000, 200_000_000_000)
    }
    explicit_off_fast = _sink_cell(
        args,
        name="all-remote-explicit-off-400g",
        linkspeed_bps=400_000_000_000,
        placement_hosts=tuple(f"node-{rank}" for rank in range(8)),
        step_specs=((0, 14),),
        dims=sensitivity_dims,
    )
    explicit_enabled_fast = _sink_cell(
        args,
        name="all-remote-explicit-enabled-400g",
        linkspeed_bps=400_000_000_000,
        collective_latency_profile=profile,
        placement_hosts=tuple(f"node-{rank}" for rank in range(8)),
        step_specs=((0, 14),),
        dims=sensitivity_dims,
    )
    unsupported = _unsupported_width_guard(args)
    behavioral = _behavioral_results(
        heldout=heldout,
        local_rows=local_rows,
        sensitivity_off_fast=sensitivity_cells[(False, 400_000_000_000)],
        sensitivity_off_slow=sensitivity_cells[(False, 200_000_000_000)],
        sensitivity_enabled_fast=sensitivity_cells[(True, 400_000_000_000)],
        sensitivity_enabled_slow=sensitivity_cells[(True, 200_000_000_000)],
        flagship_off=flagship_off,
        flagship_enabled=flagship_enabled,
    )
    fatal = _fatal_guards(
        flagship_off=flagship_off,
        flagship_legacy=flagship_legacy,
        flagship_enabled=flagship_enabled,
        sensitivity_off_fast=sensitivity_cells[(False, 400_000_000_000)],
        sensitivity_off_slow=sensitivity_cells[(False, 200_000_000_000)],
        sensitivity_enabled_fast=sensitivity_cells[(True, 400_000_000_000)],
        sensitivity_enabled_slow=sensitivity_cells[(True, 200_000_000_000)],
        explicit_off_fast=explicit_off_fast,
        explicit_enabled_fast=explicit_enabled_fast,
        local_rows=local_rows,
        unsupported=unsupported,
    )
    violated = tuple(name for name, passed in fatal.items() if not passed)
    behavioral_family_passed = (
        None
        if violated
        else sum(bool(relation["passed"]) for relation in behavioral.values())
    )
    summary: dict[str, object] = {
        "study_attempt": STUDY_ATTEMPT,
        "prior_void_revision": PRIOR_VOID_REVISION,
        "attempt_two_change_kind": (
            "fatal harness oracle refreeze; no modeled behavior change"
        ),
        "void": bool(violated),
        "violated_fatal_guards": violated,
        "behavioral_relations": behavioral,
        "behavioral_score_interpretable": not violated,
        "behavioral_family_total": (
            None if violated else EXPECTED_SCORED_FAMILIES
        ),
        "behavioral_family_passed": behavioral_family_passed,
        "heldout_instance_total": EXPECTED_HELDOUT_INSTANCES,
        "fatal_guards": fatal,
        "heldout": heldout,
        "local_width_cells": local_rows,
        "flagship": {
            "off": flagship_off,
            "legacy": flagship_legacy,
            "enabled": flagship_enabled,
        },
        "sensitivity": {
            f"{'enabled' if enabled else 'off'}-{rate // 1_000_000_000}g": cell
            for (enabled, rate), cell in sensitivity_cells.items()
        },
        "all_remote_explicit": {
            "off-400g": explicit_off_fast,
            "enabled-400g": explicit_enabled_fast,
        },
        "unsupported_width": unsupported,
        "error_budget_ps": {
            "mission_network_before": 106_000_000,
            "calibrated_floor": EXPECTED_STEP_FLOOR_PS,
            "mission_network_after": 1_552_145_392,
            "mission_step_before": 205_000_000,
            "mission_step_after": 1_651_145_392,
            "plausible_real_floor": 1_100_000_000,
            "plausible_real_ceiling": 4_500_000_000,
        },
        "sources": {
            "issue_url": SOURCE_ISSUE_URL,
            "capture_url": SOURCE_URL,
            "system_url": SOURCE_SYSTEM_URL,
            "zip_sha256": SOURCE_ZIP_SHA256,
            "local_tsv_sha256": SOURCE_LOCAL_TSV_SHA256,
            "source_kind": "user-supplied public capture",
        },
        "provenance": provenance,
    }
    _write_json(args.run_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--htsim-rnic", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    check_only(args)
    if args.check_only:
        return
    summary = run_study(args)
    headline = {
        "void": summary["void"],
        "violated_fatal_guards": summary["violated_fatal_guards"],
    }
    if not summary["void"]:
        headline.update(
            {
                "behavioral_family_passed": summary[
                    "behavioral_family_passed"
                ],
                "behavioral_family_total": summary[
                    "behavioral_family_total"
                ],
            }
        )
    print(json.dumps(headline, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
