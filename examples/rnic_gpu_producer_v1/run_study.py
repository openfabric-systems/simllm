"""Run the frozen BACK-27 GPU producer coupling study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = REPO_ROOT / "simllm" / "backends" / "rnic"
RESULTS = Path(__file__).with_name("results.csv")
SIMLLM_BASE_COMMIT = "b74629b4b4da1addda9ff21226cfabf5c09aad87"
NCCL_COMMIT = "5067397c2676d5aed50042fc39e5c8ee96eb0027"

PRODUCER_SHAPES = (
    "host_cpu_driver",
    "cpu_proxy",
    "gpu_initiated",
)
OCCUPANCY_SHARED_BYTES = {
    "idle": 0,
    "half": 32 * 1024,
    "saturated": 64 * 1024,
}
COUPLING_MODES = ("disabled", "enabled")
BASELINE_SUBMISSION_CYCLE = 16
BACKGROUND_COMPLETION_CYCLE = 32
ISOLATED_COMPLETION_CYCLES = {
    "cpu_proxy": 4,
    "gpu_initiated": 7,
}
EXPECTED_TASK_COMPLETIONS = {
    (shape, occupancy): (
        isolated
        if occupancy == "idle"
        else isolated + 1
        if occupancy == "half"
        else BACKGROUND_COMPLETION_CYCLE + isolated
    )
    for shape, isolated in ISOLATED_COMPLETION_CYCLES.items()
    for occupancy in OCCUPANCY_SHARED_BYTES
}
OWNER_IDS = {
    "cpu_proxy": 7202,
    "gpu_initiated": 7103,
}
PS_PER_FIXTURE_CYCLE = 1_000

FROZEN_ARTIFACT_DIGESTS = {
    "examples/rnic_wq_v1/results.csv": (
        "598f0e10ca4e5a83a9dfb8ed8289e25cdc4c80fc24f92f2f70db967724be5682"
    ),
    "examples/rnic_pcie_v1/results.csv": (
        "464b92fd5327287db6b5e71a5449add5b893285bc3c4bcdf6a4950355339a5e2"
    ),
    "examples/rnic_device_v1/results.csv": (
        "7a0b8423d0a99de9538047f307bb7fd2f20c8d19bd408ef90fe02199da868934"
    ),
    "examples/rnic_device_v1/native_tests.csv": (
        "969963477314bfb723770556a02e4f038c7220820d522ae60dfa8c80744a202d"
    ),
    "examples/rnic_session_records_v1/results.json": (
        "d83575d1c873d3375bc24819c4d6eca0b85ea3a414fe8578f30262268a39fdf6"
    ),
    "examples/rnic_hostmem_v1/results.csv": (
        "1bc7bcc8e72b7aef9fda1ed7e6ca2078d60c48a00377cbf8dfded75ff4d2fa53"
    ),
    "examples/rnic_submission_v1/results.csv": (
        "8f74c6fd92d012f2c70c1c2b09d6f49a4d99bcc35fd418a239f7b577777edbc7"
    ),
    "examples/gpu_service_model/results.csv": (
        "c6e98d8cdca82d72a0ff82a60f6880246849e327c4de4ff7c59f563d52b03032"
    ),
    "examples/gpu_task_mix/results.csv": (
        "cc6a6e18d574be9a3fe5f52d1a78b235342d57fbd68595d51883f6840f4c8611"
    ),
    "examples/gpu_task_mix/diagnostics.csv": (
        "1c3767eef14241cf4e5ccf3bad925c5674101b16631fec36549f850910c3a3b5"
    ),
    "examples/gpu_task_mix/nccl_convergence.csv": (
        "a45a3dac202f12603fb3aa004db6467f8e194be5451570cdd1244d3f2dea58a2"
    ),
}

ROW_FIELDS = (
    "coupling",
    "producer_shape",
    "sm_occupancy",
    "background_shared_bytes",
    "baseline_submission_cycle",
    "task_present",
    "task_submitted_cycle",
    "task_eligible_cycle",
    "task_started_cycle",
    "task_finished_cycle",
    "task_completed_cycle",
    "task_issued_instructions",
    "task_hbm_requested_bytes",
    "task_hbm_transacted_bytes",
    "task_hbm_request_instructions",
    "task_nvlink_requested_bytes",
    "task_nvlink_transacted_bytes",
    "task_nvlink_request_instructions",
    "effective_submission_cycle",
    "submission_delay_cycles",
    "native_linked_records",
    "native_record_submission_ps",
    "native_invariants_valid",
)

NATIVE_LINK_FIELDS = (
    "producer_shape",
    "task_id",
    "task_owner_kind",
    "task_owner_id",
    "task_submitted_at_ps",
    "task_eligible_at_ps",
    "task_started_at_ps",
    "task_finished_at_ps",
    "task_completed_at_ps",
    "record_submitted_at_ps",
    "linked_records",
    "invariants_valid",
)


def _wave3_root() -> Path:
    configured = os.environ.get("SIMLLM_WAVE3_RUN_ROOT")
    if not configured:
        raise RuntimeError(
            "SIMLLM_WAVE3_RUN_ROOT must name the external wave-3 run root"
        )
    return Path(configured).resolve()


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _expected_completion(shape: str, occupancy: str) -> int:
    return EXPECTED_TASK_COMPLETIONS[(shape, occupancy)]


def _validate_registry(out: Path) -> None:
    cells = {
        (coupling, shape, occupancy)
        for coupling in COUPLING_MODES
        for shape in PRODUCER_SHAPES
        for occupancy in OCCUPANCY_SHARED_BYTES
    }
    if len(cells) != 18:
        raise AssertionError("BACK-27 registry must contain eighteen unique cells")
    if len(ROW_FIELDS) != len(set(ROW_FIELDS)):
        raise AssertionError("BACK-27 result fields must be unique")
    if len(NATIVE_LINK_FIELDS) != len(set(NATIVE_LINK_FIELDS)):
        raise AssertionError("BACK-27 native link fields must be unique")
    for commit in (SIMLLM_BASE_COMMIT, NCCL_COMMIT):
        if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
            raise AssertionError("BACK-27 source audit commits must be full hashes")
    subprocess.run(
        ["git", "cat-file", "-e", f"{SIMLLM_BASE_COMMIT}^{{commit}}"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    if _expected_completion("cpu_proxy", "half") != 5:
        raise AssertionError("BACK-27 CPU-proxy half-occupancy literal drifted")
    if _expected_completion("gpu_initiated", "half") != 8:
        raise AssertionError("BACK-27 GPU half-occupancy literal drifted")
    if (
        _expected_completion("cpu_proxy", "saturated")
        - BASELINE_SUBMISSION_CYCLE
        != 20
    ):
        raise AssertionError("BACK-27 CPU-proxy saturated delay drifted")
    if (
        _expected_completion("gpu_initiated", "saturated")
        - BASELINE_SUBMISSION_CYCLE
        != 23
    ):
        raise AssertionError("BACK-27 GPU saturated delay drifted")
    for relative, expected in FROZEN_ARTIFACT_DIGESTS.items():
        actual = _digest(REPO_ROOT / relative)
        if actual != expected:
            raise AssertionError(
                f"frozen artifact digest drifted for {relative}: {actual}"
            )
    try:
        out.resolve().relative_to(_wave3_root())
    except ValueError as error:
        raise ValueError(
            "BACK-27 output must remain under SIMLLM_WAVE3_RUN_ROOT"
        ) from error


def _native_executable(build_dir: Path) -> Path:
    name = "simllm_rnic_submission_test"
    candidates = (
        build_dir / name,
        build_dir / f"{name}.exe",
        build_dir / "Release" / name,
        build_dir / "Release" / f"{name}.exe",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    joined = ", ".join(str(candidate) for candidate in candidates)
    raise RuntimeError(f"native executable not found; checked {joined}")


def _build(build_dir: Path) -> tuple[Path, dict[str, int]]:
    subprocess.run(
        [
            "cmake",
            "-S",
            str(SOURCE_DIR),
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DSIMLLM_RNIC_BUILD_TESTS=ON",
            "-DSIMLLM_RNIC_BUILD_TOOLS=ON",
            "-DSIMLLM_RNIC_WARNINGS_AS_ERRORS=ON",
        ],
        check=True,
    )
    subprocess.run(
        [
            "cmake",
            "--build",
            str(build_dir),
            "--config",
            "Release",
            "--parallel",
        ],
        check=True,
    )
    completed = subprocess.run(
        [
            "ctest",
            "--test-dir",
            str(build_dir),
            "-C",
            "Release",
            "--output-on-failure",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    print(completed.stdout, end="")
    match = re.search(
        r"100% tests passed, (\d+) tests failed out of (\d+)",
        completed.stdout,
    )
    if match is None:
        raise RuntimeError("could not parse the native CTest summary")
    failed = int(match.group(1))
    total = int(match.group(2))
    return _native_executable(build_dir), {
        "passed": total - failed,
        "failed": failed,
        "total": total,
    }


def _architecture() -> Any:
    from simllm.compute import (
        GpuArchitectureProfile,
        GpuCalibrationProfile,
        GpuModelProvenance,
        MemoryHierarchyProfile,
        NvlinkProfile,
        PipelineKind,
        PipelineProfile,
        WarpSchedulerPolicy,
    )

    profile_id = "rnic-producer-synthetic-v1"
    return GpuArchitectureProfile(
        profile_id=profile_id,
        gpu_name="rnic-producer-synthetic",
        sm_count=1,
        warp_size=32,
        scheduler_count_per_sm=1,
        dispatch_width_per_scheduler=1,
        max_blocks_per_sm=16,
        max_warps_per_sm=64,
        max_threads_per_sm=2048,
        max_threads_per_block=1024,
        registers_per_sm=65_536,
        max_registers_per_thread=255,
        register_allocation_granularity_per_warp=1,
        shared_memory_per_sm=64 * 1024,
        max_static_shared_memory_per_block=64 * 1024,
        max_shared_memory_per_block=64 * 1024,
        shared_memory_allocation_granularity=1,
        calibration=GpuCalibrationProfile(
            calibration_id=f"{profile_id}-calibration",
            target_architecture_profile_id=profile_id,
            provenance=GpuModelProvenance(
                source="synthetic BACK-27 fixture, no silicon claim",
                version="1",
                gpu="rnic-producer-synthetic",
                created="2026-08-11",
            ),
            core_clock_hz=1_000_000_000,
            target_memory_clock_hz=None,
            pipelines=(
                PipelineProfile(
                    kind=PipelineKind.ALU,
                    opcodes=("ALU",),
                    latency_cycles=1,
                    issue_width_per_sm=1,
                ),
                PipelineProfile(
                    kind=PipelineKind.LOAD_STORE,
                    opcodes=("LDG", "STG"),
                    latency_cycles=1,
                    issue_width_per_sm=1,
                ),
                PipelineProfile(
                    kind=PipelineKind.CONTROL,
                    opcodes=("CONTROL",),
                    latency_cycles=1,
                    issue_width_per_sm=1,
                ),
            ),
            memory=MemoryHierarchyProfile(
                hbm_latency_cycles=2,
                hbm_bandwidth_bytes_per_cycle=64,
                l2_latency_cycles=1,
                l1_latency_cycles=1,
                shared_latency_cycles=1,
            ),
            nvlink=NvlinkProfile(
                latency_cycles=2,
                bandwidth_bytes_per_cycle=64,
            ),
            copy_engines=(),
            warp_scheduler_policy=WarpSchedulerPolicy.LOOSE_ROUND_ROBIN,
            relative_uncertainty=0.0,
        ),
        aliases=(),
    )


def _background_task(occupancy: str) -> Any | None:
    if occupancy == "idle":
        return None
    from simllm.compute import (
        CtaTrace,
        GpuTask,
        GpuTaskKind,
        KernelLaunch,
        MemorySpace,
        PipelineKind,
        SassInstruction,
        SassWarpTrace,
    )

    instructions = (
        SassInstruction(
            opcode="LDG",
            pipeline=PipelineKind.LOAD_STORE,
            memory_space=MemorySpace.HBM,
            requested_bytes=64,
            transacted_bytes=64,
            destination_registers=("r0",),
        ),
        SassInstruction(
            opcode="STG",
            pipeline=PipelineKind.LOAD_STORE,
            memory_space=MemorySpace.NVLINK,
            requested_bytes=64,
            transacted_bytes=64,
            source_registers=("r0",),
        ),
        SassInstruction(
            opcode="ALU",
            pipeline=PipelineKind.ALU,
            repeat=26,
            dependent=True,
        ),
    )
    launch = KernelLaunch(
        implementation_id=f"rnic-producer-background-{occupancy}",
        trace_id=f"rnic-producer-background-{occupancy}-trace",
        grid_blocks=1,
        threads_per_block=32,
        registers_per_thread=0,
        static_shared_memory_bytes=OCCUPANCY_SHARED_BYTES[occupancy],
        dynamic_shared_memory_bytes=0,
        cta_traces=(
            CtaTrace(
                trace_class_id=f"rnic-producer-background-{occupancy}-cta",
                block_ids=(0,),
                warp_traces=(SassWarpTrace(warp_id=0, instructions=instructions),),
            ),
        ),
    )
    return GpuTask(
        task_id=f"background-{occupancy}",
        kind=GpuTaskKind.NETWORK,
        launch=launch,
    )


def _native_link_row(
    executable: Path,
    *,
    shape: str,
    task_id: str,
    owner_id: int,
    submitted_cycle: int,
    eligible_cycle: int,
    started_cycle: int,
    finished_cycle: int,
    completed_cycle: int,
    record_submission_cycle: int,
) -> dict[str, str]:
    completed = subprocess.run(
        [
            str(executable),
            "--producer-link-csv",
            shape,
            task_id,
            str(owner_id),
            str(submitted_cycle * PS_PER_FIXTURE_CYCLE),
            str(eligible_cycle * PS_PER_FIXTURE_CYCLE),
            str(started_cycle * PS_PER_FIXTURE_CYCLE),
            str(finished_cycle * PS_PER_FIXTURE_CYCLE),
            str(completed_cycle * PS_PER_FIXTURE_CYCLE),
            str(record_submission_cycle * PS_PER_FIXTURE_CYCLE),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    reader = csv.DictReader(io.StringIO(completed.stdout))
    if tuple(reader.fieldnames or ()) != NATIVE_LINK_FIELDS:
        raise AssertionError("BACK-27 native task-link schema drifted")
    rows = list(reader)
    if len(rows) != 1:
        raise AssertionError("BACK-27 native task-link probe must emit one row")
    return rows[0]


def _integer(value: str, field: str) -> int:
    try:
        return int(value)
    except ValueError as error:
        raise AssertionError(f"BACK-27 field {field} must be an integer") from error


def _schedule_cell(
    *,
    executable: Path,
    coupling: str,
    shape: str,
    occupancy: str,
) -> dict[str, str | int]:
    from simllm.compute import (
        RnicProducerCoupling,
        RnicProducerRequest,
        RnicProducerShape,
        SmSchedulerModel,
    )

    enabled = coupling == "enabled"
    producer_shape = RnicProducerShape(shape)
    background = _background_task(occupancy)
    concurrent_tasks = () if background is None else (background,)
    scheduler = SmSchedulerModel(_architecture()) if enabled else None
    service = RnicProducerCoupling(enabled=enabled, scheduler=scheduler)
    schedule = service.schedule(
        (
            RnicProducerRequest(
                task_id=f"producer-{shape}-{occupancy}",
                producer_shape=producer_shape,
                wqe_count=1,
                submitted_cycle=0,
                eligible_cycle=0,
                baseline_submission_cycle=BASELINE_SUBMISSION_CYCLE,
            ),
        ),
        concurrent_tasks=concurrent_tasks,
    )
    if len(schedule.entries) != 1:
        raise AssertionError("BACK-27 schedule must return one ordered entry")
    entry = schedule.entries[0]
    link = entry.producer_task
    task = None
    if link is not None:
        if schedule.estimate is None:
            raise AssertionError("BACK-27 linked task requires a concurrent estimate")
        task = next(
            item for item in schedule.estimate.tasks if item.task_id == link.task_id
        )

    expected_task = enabled and shape != "host_cpu_driver"
    if (link is not None) != expected_task:
        raise AssertionError("BACK-27 producer task presence drifted")
    expected_completion = (
        _expected_completion(shape, occupancy) if expected_task else None
    )
    expected_submission = max(
        BASELINE_SUBMISSION_CYCLE,
        expected_completion or BASELINE_SUBMISSION_CYCLE,
    )
    if entry.effective_submission_cycle != expected_submission:
        raise AssertionError(
            f"BACK-27 effective submission drifted for {coupling}/{shape}/{occupancy}"
        )
    if link is not None and link.completed_cycle != expected_completion:
        raise AssertionError(
            f"BACK-27 task completion drifted for {shape}/{occupancy}: "
            f"{link.completed_cycle}"
        )

    native_linked_records = 0
    native_record_submission_ps = 0
    native_invariants_valid = 0
    if link is not None:
        native = _native_link_row(
            executable,
            shape=shape,
            task_id=link.task_id,
            owner_id=OWNER_IDS[shape],
            submitted_cycle=link.submitted_cycle,
            eligible_cycle=link.eligible_cycle,
            started_cycle=link.started_cycle,
            finished_cycle=link.finished_cycle,
            completed_cycle=link.completed_cycle,
            record_submission_cycle=entry.effective_submission_cycle,
        )
        native_expected = {
            "producer_shape": shape,
            "task_id": link.task_id,
            "task_owner_kind": "gpu",
            "task_owner_id": str(OWNER_IDS[shape]),
            "task_submitted_at_ps": str(
                link.submitted_cycle * PS_PER_FIXTURE_CYCLE
            ),
            "task_eligible_at_ps": str(link.eligible_cycle * PS_PER_FIXTURE_CYCLE),
            "task_started_at_ps": str(link.started_cycle * PS_PER_FIXTURE_CYCLE),
            "task_finished_at_ps": str(link.finished_cycle * PS_PER_FIXTURE_CYCLE),
            "task_completed_at_ps": str(
                link.completed_cycle * PS_PER_FIXTURE_CYCLE
            ),
            "record_submitted_at_ps": str(
                entry.effective_submission_cycle * PS_PER_FIXTURE_CYCLE
            ),
            "linked_records": "1",
            "invariants_valid": "1",
        }
        if native != native_expected:
            raise AssertionError(
                f"BACK-27 native linkage drifted for {shape}/{occupancy}: {native}"
            )
        native_linked_records = _integer(native["linked_records"], "linked_records")
        native_record_submission_ps = _integer(
            native["record_submitted_at_ps"], "record_submitted_at_ps"
        )
        native_invariants_valid = _integer(
            native["invariants_valid"], "invariants_valid"
        )

    return {
        "coupling": coupling,
        "producer_shape": shape,
        "sm_occupancy": occupancy,
        "background_shared_bytes": OCCUPANCY_SHARED_BYTES[occupancy],
        "baseline_submission_cycle": BASELINE_SUBMISSION_CYCLE,
        "task_present": int(link is not None),
        "task_submitted_cycle": "" if link is None else link.submitted_cycle,
        "task_eligible_cycle": "" if link is None else link.eligible_cycle,
        "task_started_cycle": "" if link is None else link.started_cycle,
        "task_finished_cycle": "" if link is None else link.finished_cycle,
        "task_completed_cycle": "" if link is None else link.completed_cycle,
        "task_issued_instructions": "" if task is None else task.issued_instructions,
        "task_hbm_requested_bytes": "" if task is None else task.hbm_requested_bytes,
        "task_hbm_transacted_bytes": "" if task is None else task.hbm_transacted_bytes,
        "task_hbm_request_instructions": (
            "" if task is None else task.hbm_request_instructions
        ),
        "task_nvlink_requested_bytes": (
            "" if task is None else task.nvlink_requested_bytes
        ),
        "task_nvlink_transacted_bytes": (
            "" if task is None else task.nvlink_transacted_bytes
        ),
        "task_nvlink_request_instructions": (
            "" if task is None else task.nvlink_request_instructions
        ),
        "effective_submission_cycle": entry.effective_submission_cycle,
        "submission_delay_cycles": (
            entry.effective_submission_cycle - BASELINE_SUBMISSION_CYCLE
        ),
        "native_linked_records": native_linked_records,
        "native_record_submission_ps": native_record_submission_ps,
        "native_invariants_valid": native_invariants_valid,
    }


def _csv_bytes(rows: list[dict[str, str | int]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=ROW_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def _timeline_bytes(shape: str, submission_cycle: int) -> bytes:
    payload = {
        "producer_shape": shape,
        "submission_cycles": [submission_cycle],
    }
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _validate_rows(rows: list[dict[str, str | int]]) -> dict[str, int]:
    indexed = {
        (str(row["coupling"]), str(row["producer_shape"]), str(row["sm_occupancy"])): row
        for row in rows
    }
    if len(rows) != 18 or len(indexed) != 18:
        raise AssertionError("BACK-27 measured grid differs from the frozen grid")

    half_passed = 0
    saturated_passed = 0
    idle_identity_passed = 0
    structural_failures: list[str] = []
    for shape, isolated in ISOLATED_COMPLETION_CYCLES.items():
        idle = indexed[("enabled", shape, "idle")]
        half = indexed[("enabled", shape, "half")]
        saturated = indexed[("enabled", shape, "saturated")]
        disabled_idle = indexed[("disabled", shape, "idle")]
        half_passed += int(
            int(half["task_completed_cycle"]) == isolated + 1
        )
        saturated_passed += int(
            int(saturated["submission_delay_cycles"])
            == BACKGROUND_COMPLETION_CYCLE + isolated - BASELINE_SUBMISSION_CYCLE
        )
        idle_identity_passed += int(
            _timeline_bytes(shape, int(idle["effective_submission_cycle"]))
            == _timeline_bytes(
                shape, int(disabled_idle["effective_submission_cycle"])
            )
        )

        expected_counters = (
            (2, 64, 1) if shape == "cpu_proxy" else (3, 68, 2)
        )
        for occupancy in OCCUPANCY_SHARED_BYTES:
            row = indexed[("enabled", shape, occupancy)]
            actual_counters = (
                int(row["task_issued_instructions"]),
                int(row["task_hbm_transacted_bytes"]),
                int(row["task_hbm_request_instructions"]),
            )
            if actual_counters != expected_counters:
                structural_failures.append(
                    f"{shape}/{occupancy}: task counters {actual_counters}"
                )
            if (
                int(row["task_nvlink_transacted_bytes"]) != 0
                or int(row["task_nvlink_request_instructions"]) != 0
                or int(row["native_linked_records"]) != 1
                or int(row["native_invariants_valid"]) != 1
            ):
                structural_failures.append(
                    f"{shape}/{occupancy}: linkage or NVLink projection failed"
                )

    for coupling in COUPLING_MODES:
        for occupancy in OCCUPANCY_SHARED_BYTES:
            host = indexed[(coupling, "host_cpu_driver", occupancy)]
            if int(host["task_present"]) != 0:
                structural_failures.append(
                    f"{coupling}/host_cpu_driver/{occupancy}: task was constructed"
                )
    for shape in PRODUCER_SHAPES:
        for occupancy in OCCUPANCY_SHARED_BYTES:
            disabled = indexed[("disabled", shape, occupancy)]
            if (
                int(disabled["task_present"]) != 0
                or int(disabled["effective_submission_cycle"])
                != BASELINE_SUBMISSION_CYCLE
            ):
                structural_failures.append(
                    f"disabled/{shape}/{occupancy}: identity path drifted"
                )

    if half_passed != 2:
        raise AssertionError("BACK-27 half-occupancy relation failed")
    if saturated_passed != 2:
        raise AssertionError("BACK-27 saturated-cadence relation failed")
    if idle_identity_passed != 2:
        raise AssertionError("BACK-27 idle timeline identity failed")
    if structural_failures:
        raise AssertionError(
            "BACK-27 fatal structural failures: " + "; ".join(structural_failures)
        )
    return {
        "half_occupancy_passed": half_passed,
        "half_occupancy_total": 2,
        "saturated_cadence_passed": saturated_passed,
        "saturated_cadence_total": 2,
        "idle_identity_passed": idle_identity_passed,
        "idle_identity_total": 2,
    }


def _run(out: Path) -> dict[str, Any]:
    before = {
        relative: _digest(REPO_ROOT / relative)
        for relative in FROZEN_ARTIFACT_DIGESTS
    }
    out.mkdir(parents=True, exist_ok=True)
    executable, ctest = _build(out / "build")
    baseline = subprocess.run(
        [str(executable), "--study-csv"],
        check=True,
        capture_output=True,
    ).stdout
    accepted_back20 = (REPO_ROOT / "examples/rnic_submission_v1/results.csv").read_bytes()
    if baseline != accepted_back20:
        raise AssertionError("BACK-27 changed the accepted BACK-20 study bytes")

    rows = [
        _schedule_cell(
            executable=executable,
            coupling=coupling,
            shape=shape,
            occupancy=occupancy,
        )
        for coupling in COUPLING_MODES
        for shape in PRODUCER_SHAPES
        for occupancy in OCCUPANCY_SHARED_BYTES
    ]
    checked = _validate_rows(rows)
    rendered = _csv_bytes(rows)
    RESULTS.write_bytes(rendered)
    (out / "results.csv").write_bytes(rendered)
    (out / "back20_results.csv").write_bytes(baseline)

    after = {
        relative: _digest(REPO_ROOT / relative)
        for relative in FROZEN_ARTIFACT_DIGESTS
    }
    identity = {
        relative: before[relative] == after[relative] == expected
        for relative, expected in FROZEN_ARTIFACT_DIGESTS.items()
    }
    if not all(identity.values()):
        raise AssertionError("BACK-27 accepted artifact byte identity failed")
    summary: dict[str, Any] = {
        **checked,
        "artifact_identity": identity,
        "artifact_identity_passed": sum(identity.values()),
        "artifact_identity_total": len(identity),
        "back20_live_output_identity": baseline == accepted_back20,
        "ctest": ctest,
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="validate the frozen registry without creating outputs",
    )
    arguments = parser.parse_args()
    _validate_registry(arguments.out)
    if arguments.check_only:
        print(
            "RNIC GPU producer registry check passed; "
            "no artifacts were produced"
        )
        return
    summary = _run(arguments.out.resolve())
    print(
        "BACK-27 passed "
        f"{summary['half_occupancy_passed']}/"
        f"{summary['half_occupancy_total']} half-occupancy relations, "
        f"{summary['saturated_cadence_passed']}/"
        f"{summary['saturated_cadence_total']} saturated-cadence relations, "
        f"{summary['idle_identity_passed']}/"
        f"{summary['idle_identity_total']} idle identities and "
        f"{summary['artifact_identity_passed']}/"
        f"{summary['artifact_identity_total']} artifact identities"
    )


if __name__ == "__main__":
    main()
