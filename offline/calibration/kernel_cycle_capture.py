#!/usr/bin/env python3
"""Render and drive the portable COMP-64 kernel-cycle capture campaign."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from simllm.calibration.canonical import (
    canonical_bytes,
    canonical_loads,
    canonical_sha256,
    sha256_bytes,
)
from simllm.calibration.kernel_cycle_lut import validate_kernel_cycle_lut

CAMPAIGN_SCHEMA = "simllm-kernel-cycle-campaign-v1"
PLAN_SCHEMA = "simllm-kernel-cycle-plan-v1"
CODE_INDEX_SCHEMA = "simllm-kernel-code-object-index-v1"
CODE_MANIFEST_SCHEMA = "simllm-kernel-code-object-manifest-v1"


def _object(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path}: expected an object")
    return value


def _array(value: object, path: str) -> Sequence[Any]:
    if not isinstance(value, list | tuple):
        raise TypeError(f"{path}: expected an array")
    return value


def _string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{path}: expected a nonblank string without edge whitespace")
    return value


def _integer(value: object, path: str, *, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{path}: expected an integer of at least {minimum}")
    return value


def _fields(value: Mapping[str, Any], path: str, expected: set[str]) -> None:
    missing = sorted(expected - value.keys())
    unknown = sorted(value.keys() - expected)
    if missing:
        raise ValueError(f"{path}: missing fields {missing}")
    if unknown:
        raise ValueError(f"{path}: unknown fields {unknown}")


def load_campaign(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    campaign = dict(_object(value, "campaign"))
    validate_campaign(campaign)
    return campaign


def validate_campaign(campaign: Mapping[str, Any]) -> None:
    """Reject campaign drift before a GPU allocation is requested."""

    _fields(
        campaign,
        "campaign",
        {
            "schema",
            "frameworks",
            "pools",
            "launch_modes",
            "parallelism",
            "decode_batches",
            "kv_grid_basis_points",
            "kv_placement_modes",
            "prefill_shapes",
            "replays",
            "nsys",
            "ncu",
            "code_object_harvest",
        },
    )
    if campaign["schema"] != CAMPAIGN_SCHEMA:
        raise ValueError(f"campaign.schema: expected {CAMPAIGN_SCHEMA!r}")
    frameworks = _array(campaign["frameworks"], "campaign.frameworks")
    framework_names: list[str] = []
    for index, raw in enumerate(frameworks):
        path = f"campaign.frameworks[{index}]"
        framework = _object(raw, path)
        _fields(framework, path, {"name", "target_environment_variable"})
        framework_names.append(_string(framework["name"], f"{path}.name"))
        environment_name = _string(
            framework["target_environment_variable"],
            f"{path}.target_environment_variable",
        )
        if not environment_name.startswith("SIMLLM_"):
            raise ValueError(f"{path}.target_environment_variable: expected SIMLLM_ prefix")
    if framework_names != sorted(framework_names) or len(framework_names) != len(
        set(framework_names)
    ):
        raise ValueError("campaign.frameworks: names must be sorted and unique")
    if list(campaign["pools"]) != ["decode", "prefill"]:
        raise ValueError("campaign.pools: expected decode and prefill")
    if list(campaign["launch_modes"]) != ["cuda-graph", "eager"]:
        raise ValueError("campaign.launch_modes: expected cuda-graph and eager")
    parallelism_rows = _array(campaign["parallelism"], "campaign.parallelism")
    if not parallelism_rows:
        raise ValueError("campaign.parallelism: must not be empty")
    for index, raw in enumerate(parallelism_rows):
        path = f"campaign.parallelism[{index}]"
        row = _object(raw, path)
        _fields(
            row,
            path,
            {"tensor_parallel", "pipeline_parallel", "data_parallel", "expert_parallel"},
        )
        for name, value in row.items():
            _integer(value, f"{path}.{name}")
    batches = [_integer(value, "campaign.decode_batches[]") for value in campaign["decode_batches"]]
    if batches != sorted(set(batches)):
        raise ValueError("campaign.decode_batches: must be sorted and unique")
    grid = [
        _integer(value, "campaign.kv_grid_basis_points[]")
        for value in campaign["kv_grid_basis_points"]
    ]
    if len(grid) != 16 or grid[0] != 100 or grid[-1] != 10_000:
        raise ValueError("campaign.kv_grid_basis_points: expected 16 points from 100 to 10000")
    if grid != sorted(set(grid)) or any(value > 10_000 for value in grid):
        raise ValueError("campaign.kv_grid_basis_points: must be sorted, unique and bounded")
    if list(campaign["kv_placement_modes"]) != [
        "fresh-contiguous",
        "deliberately-fragmented",
    ]:
        raise ValueError("campaign.kv_placement_modes: unexpected placement matrix")
    shapes = _array(campaign["prefill_shapes"], "campaign.prefill_shapes")
    if not shapes:
        raise ValueError("campaign.prefill_shapes: must not be empty")
    for index, raw in enumerate(shapes):
        path = f"campaign.prefill_shapes[{index}]"
        shape = _object(raw, path)
        _fields(shape, path, {"computed_new_tokens", "existing_context_tokens"})
        _integer(shape["computed_new_tokens"], f"{path}.computed_new_tokens")
        _integer(shape["existing_context_tokens"], f"{path}.existing_context_tokens", minimum=0)
    replays = _object(campaign["replays"], "campaign.replays")
    _fields(replays, "campaign.replays", {"cuda-graph", "eager"})
    if _integer(replays["cuda-graph"], "campaign.replays.cuda-graph") < 256:
        raise ValueError("campaign.replays.cuda-graph: expected at least 256")
    if _integer(replays["eager"], "campaign.replays.eager") < 64:
        raise ValueError("campaign.replays.eager: expected at least 64")
    ncu = _object(campaign["ncu"], "campaign.ncu")
    _fields(
        ncu,
        "campaign.ncu",
        {"clock_control", "replay_mode", "metrics", "program_counter_sections"},
    )
    if ncu["clock_control"] != "none":
        raise ValueError("campaign.ncu.clock_control: expected none")
    metrics = set(ncu["metrics"])
    required_metrics = {
        "gpu__cycles_elapsed.max",
        "gpu__time_duration.sum",
        "dram__bytes_read.sum",
        "dram__bytes_write.sum",
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    }
    if not required_metrics <= metrics:
        raise ValueError("campaign.ncu.metrics: missing component metrics")
    if not ncu["program_counter_sections"]:
        raise ValueError("campaign.ncu.program_counter_sections: must not be empty")
    harvest = _object(campaign["code_object_harvest"], "campaign.code_object_harvest")
    _fields(
        harvest,
        "campaign.code_object_harvest",
        {"clean_runs", "require_byte_identical_manifests", "classes", "digests"},
    )
    if harvest["clean_runs"] != 2 or harvest["require_byte_identical_manifests"] is not True:
        raise ValueError("campaign.code_object_harvest: expected two byte-identical runs")


def _kv_length(max_context_tokens: int, basis_points: int) -> int:
    return max(1, -(-(max_context_tokens * basis_points) // 10_000))


def render_plan(
    campaign: Mapping[str, Any],
    *,
    model_name: str,
    model_revision: str,
    model_family: str,
    max_context_tokens: int,
) -> dict[str, Any]:
    """Expand the complete protocol into deterministic framework cells."""

    validate_campaign(campaign)
    _string(model_name, "model_name")
    _string(model_revision, "model_revision")
    if model_family not in {"dense", "routed"}:
        raise ValueError("model_family: expected dense or routed")
    _integer(max_context_tokens, "max_context_tokens")
    cells: list[dict[str, Any]] = []
    for framework in campaign["frameworks"]:
        for parallelism in campaign["parallelism"]:
            parallel_id = "-".join(
                f"{name[:2]}{parallelism[name]}"
                for name in (
                    "tensor_parallel",
                    "pipeline_parallel",
                    "data_parallel",
                    "expert_parallel",
                )
            )
            for launch_mode in campaign["launch_modes"]:
                for batch in campaign["decode_batches"]:
                    for basis_points in campaign["kv_grid_basis_points"]:
                        kv_length = _kv_length(max_context_tokens, basis_points)
                        for placement in campaign["kv_placement_modes"]:
                            cell_id = (
                                f"{framework['name']}-decode-{launch_mode}-{parallel_id}"
                                f"-b{batch}-kv{kv_length}-{placement}"
                            )
                            cells.append(
                                _cell(
                                    cell_id=cell_id,
                                    framework=framework,
                                    model_name=model_name,
                                    model_revision=model_revision,
                                    model_family=model_family,
                                    pool="decode",
                                    launch_mode=launch_mode,
                                    parallelism=parallelism,
                                    shape={
                                        "batch_size": batch,
                                        "per_request_kv_lengths": [kv_length] * batch,
                                    },
                                    placement=placement,
                                    campaign=campaign,
                                )
                            )
                for shape_index, shape in enumerate(campaign["prefill_shapes"]):
                    cell_id = (
                        f"{framework['name']}-prefill-{launch_mode}-{parallel_id}"
                        f"-s{shape_index:02d}"
                    )
                    cells.append(
                        _cell(
                            cell_id=cell_id,
                            framework=framework,
                            model_name=model_name,
                            model_revision=model_revision,
                            model_family=model_family,
                            pool="prefill",
                            launch_mode=launch_mode,
                            parallelism=parallelism,
                            shape=shape,
                            placement=None,
                            campaign=campaign,
                        )
                    )
    cells.sort(key=lambda cell: cell["cell_id"])
    plan = {
        "schema": PLAN_SCHEMA,
        "campaign_sha256": canonical_sha256(campaign),
        "model": {
            "name": model_name,
            "revision": model_revision,
            "family": model_family,
            "max_context_tokens": max_context_tokens,
        },
        "slurm": {
            "aware": True,
            "job_id_environment_variable": "SLURM_JOB_ID",
            "array_id_environment_variable": "SLURM_ARRAY_TASK_ID",
            "site_options_are_caller_supplied": True,
        },
        "cells": cells,
    }
    validate_plan(plan)
    return plan


def _cell(
    *,
    cell_id: str,
    framework: Mapping[str, Any],
    model_name: str,
    model_revision: str,
    model_family: str,
    pool: str,
    launch_mode: str,
    parallelism: Mapping[str, Any],
    shape: Mapping[str, Any],
    placement: str | None,
    campaign: Mapping[str, Any],
) -> dict[str, Any]:
    ncu = campaign["ncu"]
    return {
        "cell_id": cell_id,
        "framework": framework["name"],
        "target_environment_variable": framework["target_environment_variable"],
        "model": {
            "name": model_name,
            "revision": model_revision,
            "family": model_family,
        },
        "pool": pool,
        "launch_mode": launch_mode,
        "parallelism": dict(parallelism),
        "shape": dict(shape),
        "kv_placement": placement,
        "replays": campaign["replays"][launch_mode],
        "collectors": {
            "nsys": campaign["nsys"],
            "ncu": {
                "clock_control": ncu["clock_control"],
                "replay_mode": ncu["replay_mode"],
                "metrics": ncu["metrics"],
            },
            "program_counter_sampling": {
                "attempt": True,
                "sections": ncu["program_counter_sections"],
                "record_status": ["granted", "denied", "unavailable"],
            },
            "observed_clocks": ["sm-clock-hz", "memory-clock-hz"],
        },
        "routing_evidence_required": model_family == "routed",
        "code_object_harvest": campaign["code_object_harvest"],
    }


def validate_plan(plan: Mapping[str, Any]) -> None:
    _fields(plan, "plan", {"schema", "campaign_sha256", "model", "slurm", "cells"})
    if plan["schema"] != PLAN_SCHEMA:
        raise ValueError(f"plan.schema: expected {PLAN_SCHEMA!r}")
    cells = _array(plan["cells"], "plan.cells")
    if not cells:
        raise ValueError("plan.cells: must not be empty")
    ids = [cell["cell_id"] for cell in cells]
    if ids != sorted(ids) or len(ids) != len(set(ids)):
        raise ValueError("plan.cells: IDs must be sorted and unique")
    for index, raw in enumerate(cells):
        path = f"plan.cells[{index}]"
        cell = _object(raw, path)
        _fields(
            cell,
            path,
            {
                "cell_id",
                "framework",
                "target_environment_variable",
                "model",
                "pool",
                "launch_mode",
                "parallelism",
                "shape",
                "kv_placement",
                "replays",
                "collectors",
                "routing_evidence_required",
                "code_object_harvest",
            },
        )
        pool = cell["pool"]
        if pool == "decode":
            shape = cell["shape"]
            if len(shape["per_request_kv_lengths"]) != shape["batch_size"]:
                raise ValueError(f"{path}.shape: KV vector length must equal batch")
            if cell["kv_placement"] not in {
                "fresh-contiguous",
                "deliberately-fragmented",
            }:
                raise ValueError(f"{path}.kv_placement: invalid decode placement")
        elif pool == "prefill":
            if cell["kv_placement"] is not None:
                raise ValueError(f"{path}.kv_placement: prefill requires null")
        else:
            raise ValueError(f"{path}.pool: unsupported pool")
        if cell["model"]["family"] == "routed" and not cell["routing_evidence_required"]:
            raise ValueError(f"{path}.routing_evidence_required: routed cell requires evidence")


def commands_for_cell(
    plan: Mapping[str, Any],
    *,
    cell_id: str,
    output_dir: str | Path,
    environment: Mapping[str, str] | None = None,
) -> list[list[str]]:
    """Render profiler commands without launching them."""

    validate_plan(plan)
    environ = os.environ if environment is None else environment
    matches = [cell for cell in plan["cells"] if cell["cell_id"] == cell_id]
    if len(matches) != 1:
        raise ValueError(f"cell_id: expected one plan cell, found {len(matches)}")
    cell = matches[0]
    target_variable = cell["target_environment_variable"]
    target = environ.get(target_variable)
    if not target:
        raise RuntimeError(f"{target_variable} must name the pinned framework target")
    output = Path(output_dir)
    target_command = [target, "--cell-spec", str(output / "cell.json")]
    nsys = cell["collectors"]["nsys"]
    ncu = cell["collectors"]["ncu"]
    commands = [
        [
            "nsys",
            "profile",
            f"--output={output / 'nsys'}",
            "--force-overwrite=false",
            f"--cuda-graph-trace={nsys['cuda_graph_trace']}",
            f"--trace={','.join(nsys['trace'])}",
            "--trace-fork-before-exec=true",
            f"--sample={nsys['sample']}",
            "--cpuctxsw=none",
            f"--wait={nsys['wait']}",
            "--export=sqlite",
            *target_command,
        ],
        [
            "ncu",
            "--target-processes",
            "all",
            "--profile-from-start",
            "off",
            "--clock-control",
            ncu["clock_control"],
            "--graph-profiling",
            "node",
            "--replay-mode",
            ncu["replay_mode"],
            "--metrics",
            ",".join(ncu["metrics"]),
            "--export",
            str(output / "ncu-components"),
            *target_command,
        ],
        [
            "ncu",
            "--target-processes",
            "all",
            "--profile-from-start",
            "off",
            "--clock-control",
            "none",
            "--graph-profiling",
            "node",
            "--replay-mode",
            "kernel",
            "--section",
            ",".join(cell["collectors"]["program_counter_sampling"]["sections"]),
            "--export",
            str(output / "ncu-pc-sampling"),
            *target_command,
        ],
    ]
    return commands


def run_cell(
    plan: Mapping[str, Any],
    *,
    cell_id: str,
    output_dir: str | Path,
    environment: Mapping[str, str] | None = None,
) -> None:
    """Run one planned cell, preserving a denied PC pass as explicit status."""

    environ = dict(os.environ if environment is None else environment)
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("output_dir must be absent or empty")
    output.mkdir(parents=True, exist_ok=True)
    cell = next(cell for cell in plan["cells"] if cell["cell_id"] == cell_id)
    (output / "cell.json").write_bytes(canonical_bytes(cell))
    metadata = {
        "slurm_job_id": environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": environ.get("SLURM_ARRAY_TASK_ID"),
    }
    (output / "scheduler.json").write_bytes(canonical_bytes(metadata))
    commands = commands_for_cell(
        plan,
        cell_id=cell_id,
        output_dir=output,
        environment=environ,
    )
    for command in commands[0][:1] + commands[1][:1] + ["nvidia-smi"]:
        if shutil.which(command) is None:
            raise RuntimeError(f"required capture tool is unavailable: {command}")
    clock_path = output / "clocks.csv"
    sidecar = subprocess.Popen(
        [
            "nvidia-smi",
            (
                "--query-gpu=timestamp,index,uuid,name,pstate,clocks.current.sm,"
                "clocks.current.memory,power.draw,temperature.gpu,utilization.gpu,"
                "memory.used"
            ),
            "--format=csv",
            "--loop-ms=200",
            f"--filename={clock_path}",
        ],
        env=environ,
    )
    try:
        for index, command in enumerate(commands):
            completed = subprocess.run(command, check=False, env=environ)
            if index < 2 and completed.returncode != 0:
                raise RuntimeError(
                    f"capture command failed with status {completed.returncode}: "
                    f"{shlex.join(command)}"
                )
            if index == 2:
                pc_status = "granted" if completed.returncode == 0 else "denied"
                (output / "program-counter-status.json").write_bytes(
                    canonical_bytes({"status": pc_status, "returncode": completed.returncode})
                )
    finally:
        sidecar.terminate()
        sidecar.wait(timeout=15)


def harvest_code_objects(index_path: str | Path, root: str | Path) -> dict[str, Any]:
    """Hash per-kernel PTX and SASS references from a producer-owned index."""

    index = json.loads(Path(index_path).read_text(encoding="utf-8"))
    payload = _object(index, "code_index")
    _fields(payload, "code_index", {"schema", "entries"})
    if payload["schema"] != CODE_INDEX_SCHEMA:
        raise ValueError(f"code_index.schema: expected {CODE_INDEX_SCHEMA!r}")
    root_path = Path(root)
    entries = []
    for row_index, raw in enumerate(payload["entries"]):
        path = f"code_index.entries[{row_index}]"
        row = _object(raw, path)
        _fields(
            row,
            path,
            {
                "kernel_id",
                "implementation_class",
                "ptx_path",
                "sass_path",
                "compile_configuration",
            },
        )
        result = {
            "kernel_id": _string(row["kernel_id"], f"{path}.kernel_id"),
            "implementation_class": _string(
                row["implementation_class"],
                f"{path}.implementation_class",
            ),
            "ptx_sha256": _digest_relative(root_path, row["ptx_path"], f"{path}.ptx_path"),
            "sass_sha256": _digest_relative(
                root_path,
                row["sass_path"],
                f"{path}.sass_path",
            ),
            "compile_configuration_sha256": canonical_sha256(
                _object(row["compile_configuration"], f"{path}.compile_configuration")
            ),
        }
        if result["ptx_sha256"] is None and result["sass_sha256"] is None:
            raise ValueError(f"{path}: at least one code object must be present")
        entries.append(result)
    entries.sort(key=lambda row: row["kernel_id"])
    if len(entries) != len({row["kernel_id"] for row in entries}):
        raise ValueError("code_index.entries: kernel IDs must be unique")
    return {"schema": CODE_MANIFEST_SCHEMA, "entries": entries}


def _digest_relative(root: Path, value: object, path: str) -> str | None:
    if value is None:
        return None
    relative = Path(_string(value, path))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{path}: must stay below the code-object root")
    return sha256_bytes((root / relative).read_bytes())


def compare_code_harvests(first: str | Path, second: str | Path) -> str:
    """Require exact canonical byte identity across two clean harvest runs."""

    first_bytes = Path(first).read_bytes()
    second_bytes = Path(second).read_bytes()
    canonical_loads(first_bytes)
    canonical_loads(second_bytes)
    if first_bytes != second_bytes:
        raise ValueError("code-object harvest manifests are not byte-identical")
    return sha256_bytes(first_bytes)


def harvest_code_object_pair(
    first_index: str | Path,
    first_root: str | Path,
    second_index: str | Path,
    second_root: str | Path,
) -> dict[str, Any]:
    """Harvest two clean producer runs and require canonical byte identity."""

    first = harvest_code_objects(first_index, first_root)
    second = harvest_code_objects(second_index, second_root)
    first_bytes = canonical_bytes(first)
    if canonical_bytes(second) != first_bytes:
        raise ValueError("code-object harvest manifests are not byte-identical")
    return {
        "schema": "simllm-kernel-code-object-double-harvest-v1",
        "clean_runs": 2,
        "byte_identical": True,
        "manifest_sha256": sha256_bytes(first_bytes),
        "manifest": first,
    }


def _write_or_stdout(value: Mapping[str, Any], output: Path | None) -> None:
    encoded = canonical_bytes(value)
    if output is None:
        sys.stdout.buffer.write(encoded + b"\n")
    else:
        output.write_bytes(encoded)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan")
    plan.add_argument("--suite", type=Path, required=True)
    plan.add_argument("--model", required=True)
    plan.add_argument("--model-revision", required=True)
    plan.add_argument("--model-family", choices=("dense", "routed"), required=True)
    plan.add_argument("--max-context-tokens", type=int, required=True)
    plan.add_argument("--output", type=Path)
    commands = subparsers.add_parser("commands")
    commands.add_argument("--plan", type=Path, required=True)
    commands.add_argument("--cell-id", required=True)
    commands.add_argument("--output-dir", type=Path, required=True)
    run = subparsers.add_parser("run-cell")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--cell-id", required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    validate = subparsers.add_parser("validate-record")
    validate.add_argument("record", type=Path)
    harvest = subparsers.add_parser("harvest-code")
    harvest.add_argument("--index", type=Path, required=True)
    harvest.add_argument("--root", type=Path, required=True)
    harvest.add_argument("--output", type=Path, required=True)
    compare = subparsers.add_parser("compare-harvests")
    compare.add_argument("first", type=Path)
    compare.add_argument("second", type=Path)
    pair = subparsers.add_parser("harvest-pair")
    pair.add_argument("--first-index", type=Path, required=True)
    pair.add_argument("--first-root", type=Path, required=True)
    pair.add_argument("--second-index", type=Path, required=True)
    pair.add_argument("--second-root", type=Path, required=True)
    pair.add_argument("--output", type=Path, required=True)
    return parser


def _load_plan(path: Path) -> Mapping[str, Any]:
    value = canonical_loads(path.read_bytes())
    plan = _object(value, "plan")
    validate_plan(plan)
    return plan


def main() -> None:
    args = _parser().parse_args()
    if args.command == "plan":
        _write_or_stdout(
            render_plan(
                load_campaign(args.suite),
                model_name=args.model,
                model_revision=args.model_revision,
                model_family=args.model_family,
                max_context_tokens=args.max_context_tokens,
            ),
            args.output,
        )
    elif args.command == "commands":
        for command in commands_for_cell(
            _load_plan(args.plan),
            cell_id=args.cell_id,
            output_dir=args.output_dir,
        ):
            print(shlex.join(command))
    elif args.command == "run-cell":
        run_cell(
            _load_plan(args.plan),
            cell_id=args.cell_id,
            output_dir=args.output_dir,
        )
    elif args.command == "validate-record":
        print(validate_kernel_cycle_lut(args.record.read_bytes()).record_id)
    elif args.command == "harvest-code":
        _write_or_stdout(
            harvest_code_objects(args.index, args.root),
            args.output,
        )
    elif args.command == "compare-harvests":
        print(compare_code_harvests(args.first, args.second))
    elif args.command == "harvest-pair":
        _write_or_stdout(
            harvest_code_object_pair(
                args.first_index,
                args.first_root,
                args.second_index,
                args.second_root,
            ),
            args.output,
        )
    else:  # pragma: no cover
        raise AssertionError(f"unhandled command {args.command!r}")


if __name__ == "__main__":
    main()
