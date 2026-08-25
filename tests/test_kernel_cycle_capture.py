from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from simllm.calibration.canonical import canonical_bytes, canonical_loads, sha256_bytes
from simllm.calibration.kernel_cycle_lut import analyze_kernel_cycle_capture

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "offline" / "calibration" / "kernel_cycle_capture.py"
SUITE = ROOT / "offline" / "calibration" / "suites" / "kernel-cycle-v1" / "suite.json"
FIXTURE = ROOT / "tests" / "fixtures" / "kernel_cycle_lut_v1"

SPEC = importlib.util.spec_from_file_location("kernel_cycle_capture", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
capture = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(capture)


@pytest.fixture(scope="module")
def campaign() -> dict:
    return capture.load_campaign(SUITE)


@pytest.fixture(scope="module")
def plan(campaign: dict) -> dict:
    return capture.render_plan(
        campaign,
        model_name="Qwen/Qwen3.8-27B",
        model_revision="model-revision",
        model_family="dense",
        max_context_tokens=262_144,
    )


def test_campaign_freezes_full_pool_mode_and_component_protocol(campaign: dict) -> None:
    capture.validate_campaign(campaign)

    assert campaign["pools"] == ["decode", "prefill"]
    assert campaign["launch_modes"] == ["cuda-graph", "eager"]
    assert len(campaign["kv_grid_basis_points"]) == 16
    assert campaign["replays"] == {"cuda-graph": 256, "eager": 64}
    assert campaign["ncu"]["clock_control"] == "none"
    assert set(campaign["ncu"]["metrics"]) >= {
        "gpu__cycles_elapsed.max",
        "gpu__time_duration.sum",
        "dram__bytes_read.sum",
        "dram__bytes_write.sum",
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    }
    assert campaign["ncu"]["program_counter_sections"] == ["SourceCounters"]
    assert campaign["code_object_harvest"]["clean_runs"] == 2
    assert campaign["code_object_harvest"]["require_byte_identical_manifests"] is True


def test_plan_expands_every_framework_parallelism_pool_and_mode(plan: dict) -> None:
    capture.validate_plan(plan)

    assert plan["schema"] == "simllm-kernel-cycle-plan-v1"
    assert len(plan["cells"]) == 1_212
    assert plan["slurm"] == {
        "aware": True,
        "job_id_environment_variable": "SLURM_JOB_ID",
        "array_id_environment_variable": "SLURM_ARRAY_TASK_ID",
        "site_options_are_caller_supplied": True,
    }
    frameworks = {cell["framework"] for cell in plan["cells"]}
    assert frameworks == {"sglang", "vllm"}
    assert {cell["launch_mode"] for cell in plan["cells"]} == {"cuda-graph", "eager"}
    assert {cell["pool"] for cell in plan["cells"]} == {"decode", "prefill"}


def test_decode_plan_carries_sixteen_kv_lengths_and_two_placements(plan: dict) -> None:
    cells = [
        cell
        for cell in plan["cells"]
        if cell["framework"] == "vllm"
        and cell["pool"] == "decode"
        and cell["launch_mode"] == "cuda-graph"
        and cell["parallelism"]["tensor_parallel"] == 1
        and cell["shape"]["batch_size"] == 1
    ]
    lengths = sorted({cell["shape"]["per_request_kv_lengths"][0] for cell in cells})

    assert len(cells) == 32
    assert len(lengths) == 16
    assert lengths[0] == 2_622
    assert lengths[-1] == 262_144
    assert {cell["kv_placement"] for cell in cells} == {
        "fresh-contiguous",
        "deliberately-fragmented",
    }
    assert all(
        len(cell["shape"]["per_request_kv_lengths"]) == cell["shape"]["batch_size"]
        for cell in cells
    )


def test_routed_plan_requires_per_cell_routing_evidence(campaign: dict) -> None:
    routed = capture.render_plan(
        campaign,
        model_name="routed-model",
        model_revision="model-revision",
        model_family="routed",
        max_context_tokens=4_096,
    )

    assert all(cell["routing_evidence_required"] for cell in routed["cells"])


def test_plan_rendering_is_byte_deterministic(campaign: dict, plan: dict) -> None:
    second = capture.render_plan(
        campaign,
        model_name="Qwen/Qwen3.8-27B",
        model_revision="model-revision",
        model_family="dense",
        max_context_tokens=262_144,
    )

    assert canonical_bytes(second) == canonical_bytes(plan)
    assert plan["campaign_sha256"] == capture.canonical_sha256(campaign)


def test_dry_commands_cover_nsys_components_and_program_counter_pass(plan: dict) -> None:
    cell = next(
        cell
        for cell in plan["cells"]
        if cell["framework"] == "vllm"
        and cell["pool"] == "decode"
        and cell["launch_mode"] == "cuda-graph"
    )
    commands = capture.commands_for_cell(
        plan,
        cell_id=cell["cell_id"],
        output_dir=Path("configured-run-root") / "cell",
        environment={"SIMLLM_VLLM_KERNEL_CAPTURE_TARGET": "pinned-vllm-target"},
    )

    assert len(commands) == 3
    assert commands[0][0] == "nsys"
    assert "--cuda-graph-trace=node" in commands[0]
    assert "--trace=cuda,nvtx,cublas,cudnn" in commands[0]
    assert commands[1][0] == "ncu"
    assert commands[1][commands[1].index("--clock-control") + 1] == "none"
    assert "dram__bytes_read.sum" in commands[1][commands[1].index("--metrics") + 1]
    assert commands[2][commands[2].index("--section") + 1] == "SourceCounters"
    assert all(
        command[-3:] == ["pinned-vllm-target", "--cell-spec", "configured-run-root/cell/cell.json"]
        for command in commands
    )


def test_command_rendering_requires_exact_framework_target(plan: dict) -> None:
    with pytest.raises(RuntimeError, match="SIMLLM_SGLANG_KERNEL_CAPTURE_TARGET"):
        capture.commands_for_cell(
            plan,
            cell_id=plan["cells"][0]["cell_id"],
            output_dir="configured-run-root",
            environment={},
        )


def test_campaign_validation_rejects_short_kv_grid(campaign: dict) -> None:
    mutated = json.loads(json.dumps(campaign))
    mutated["kv_grid_basis_points"].pop()

    with pytest.raises(ValueError, match="expected 16 points"):
        capture.validate_campaign(mutated)


def test_code_object_harvest_hashes_ptx_sass_and_configuration(tmp_path: Path) -> None:
    root = tmp_path / "objects"
    root.mkdir()
    (root / "kernel.ptx").write_bytes(b"ptx-v1\n")
    (root / "kernel.sass").write_bytes(b"sass-v1\n")
    index = {
        "schema": "simllm-kernel-code-object-index-v1",
        "entries": [
            {
                "kernel_id": "kernel-a",
                "implementation_class": "triton-jit",
                "ptx_path": "kernel.ptx",
                "sass_path": "kernel.sass",
                "compile_configuration": {"arch": "sm80", "autotune": "pinned"},
            }
        ],
    }
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps(index), encoding="utf-8")

    manifest = capture.harvest_code_objects(index_path, root)
    row = manifest["entries"][0]

    assert row["ptx_sha256"] == sha256_bytes(b"ptx-v1\n")
    assert row["sass_sha256"] == sha256_bytes(b"sass-v1\n")
    assert row["compile_configuration_sha256"] == capture.canonical_sha256(
        index["entries"][0]["compile_configuration"]
    )


def test_double_harvest_requires_exact_canonical_byte_identity(tmp_path: Path) -> None:
    manifest = {"schema": "simllm-kernel-code-object-manifest-v1", "entries": []}
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    encoded = canonical_bytes(manifest)
    first.write_bytes(encoded)
    second.write_bytes(encoded)

    assert capture.compare_code_harvests(first, second) == sha256_bytes(encoded)
    second.write_bytes(canonical_bytes({**manifest, "entries": [{"changed": True}]}))
    with pytest.raises(ValueError, match="not byte-identical"):
        capture.compare_code_harvests(first, second)


def test_pair_harvest_checks_two_clean_code_object_roots(tmp_path: Path) -> None:
    index = {
        "schema": "simllm-kernel-code-object-index-v1",
        "entries": [
            {
                "kernel_id": "kernel-a",
                "implementation_class": "triton-jit",
                "ptx_path": "kernel.ptx",
                "sass_path": None,
                "compile_configuration": {"arch": "sm80"},
            }
        ],
    }
    roots = [tmp_path / "clean-run-1", tmp_path / "clean-run-2"]
    indexes = [tmp_path / "index-1.json", tmp_path / "index-2.json"]
    for root, index_path in zip(roots, indexes, strict=True):
        root.mkdir()
        (root / "kernel.ptx").write_bytes(b"stable-ptx\n")
        index_path.write_text(json.dumps(index), encoding="utf-8")

    result = capture.harvest_code_object_pair(indexes[0], roots[0], indexes[1], roots[1])

    assert result["clean_runs"] == 2
    assert result["byte_identical"] is True
    assert result["manifest_sha256"] == sha256_bytes(canonical_bytes(result["manifest"]))
    (roots[1] / "kernel.ptx").write_bytes(b"changed-ptx\n")
    with pytest.raises(ValueError, match="not byte-identical"):
        capture.harvest_code_object_pair(indexes[0], roots[0], indexes[1], roots[1])


def test_plan_cli_writes_canonical_bytes(tmp_path: Path) -> None:
    output = tmp_path / "plan.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "plan",
            "--suite",
            str(SUITE),
            "--model",
            "Qwen/Qwen3.8-27B",
            "--model-revision",
            "model-revision",
            "--model-family",
            "dense",
            "--max-context-tokens",
            "262144",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    parsed = canonical_loads(output.read_bytes())
    capture.validate_plan(parsed)


def test_validate_record_cli_checks_canonical_shape(tmp_path: Path) -> None:
    record = analyze_kernel_cycle_capture(FIXTURE)
    path = tmp_path / "lookup.json"
    path.write_bytes(record.canonical)

    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "validate-record", str(path)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stdout.strip() == record.record_id
