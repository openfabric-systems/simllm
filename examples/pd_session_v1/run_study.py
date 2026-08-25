"""Run the frozen first-slice disaggregated serving study.

The expectations commit predates both the session mechanism and this harness.
Check-only validates the complete frozen input registry without importing the
SimLLM target modules or creating the selected run directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import time
from fractions import Fraction
from pathlib import Path, PurePath
from typing import Any

STUDY_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = STUDY_DIR.parents[1]
EXPECTATIONS_PATH = STUDY_DIR / "expectations.json"
TRACE_PATH = REPOSITORY_ROOT / "examples/preplay_trace_v1/granite_length_cap.jsonl"

EXPECTATIONS_COMMIT = "303a958f80062726573ab0717decb84895cad8f9"
IMPLEMENTATION_COMMIT = "f25fc8fe6612c938fcef4a6c62f6d709f7bf77a0"
RESULT_SCHEMA = "simllm-pd-session-study-result-v1"
SCALE_SCHEMA = "simllm-pd-session-engine-scale-v1"
MODEL_ID = "ibm-granite/granite-3.0-1b-a400m-instruct"
MODEL_REVISION = "ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445"
VLLM_VERSION = "0.27.1"
DECODE_OUTPUT_TOKENS = 4
PROMPT_LENGTHS = (8, 16)
HANDOFF_DURATIONS_PS = (100_000_000, 200_000_000)
SCALE_POINTS = ((1, 1), (2, 2))
PS_PER_SECOND = 1_000_000_000_000


def render_cli_path(path: PurePath) -> str:
    """Render every executed path with POSIX separators on every host."""

    return path.as_posix()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _git_blob(commit: str, path: str) -> bytes:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
    )
    return completed.stdout


def _git_head() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _require_implementation_ancestor() -> None:
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", IMPLEMENTATION_COMMIT, "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(
            f"implementation commit {IMPLEMENTATION_COMMIT} is not an ancestor of HEAD"
        )


def _validate_frozen_arithmetic(frozen: dict[str, Any]) -> None:
    deployment = frozen["deployment"]
    geometry = frozen["model_geometry"]
    sweep = frozen["request_sweep"]
    evidence = frozen["evidence_classes"]
    if EXPECTATIONS_COMMIT != "303a958f80062726573ab0717decb84895cad8f9":
        raise SystemExit("expectations commit literal drifted")
    if deployment["target_prefill_engines"] != 16:
        raise SystemExit("target prefill count drifted")
    if deployment["target_decode_engines"] != 40:
        raise SystemExit("target decode count drifted")
    if deployment["target_simulated_ranks"] != 448:
        raise SystemExit("target rank count drifted")
    if tuple(PROMPT_LENGTHS) != tuple(sweep["prompt_tokens"]):
        raise SystemExit("prompt sweep drifted")
    if tuple(HANDOFF_DURATIONS_PS) != tuple(sweep["handoff_ps"]):
        raise SystemExit("handoff sweep drifted")
    if DECODE_OUTPUT_TOKENS != sweep["decode_output_tokens"]:
        raise SystemExit("decode output length drifted")
    bytes_per_token = (
        2
        * geometry["num_layers"]
        * geometry["num_kv_heads"]
        * geometry["head_size"]
        * geometry["kv_element_bytes"]
    )
    if bytes_per_token != geometry["kv_bytes_per_prompt_token"]:
        raise SystemExit("KV geometry arithmetic drifted")
    if evidence != {
        "behavioral_families": 2,
        "behavioral_instances": 6,
        "exact_oracle_rows": 4,
        "structural_invariants": "unscored",
        "native_engine_smoke": "separate",
        "scale_measurements": "descriptive",
    }:
        raise SystemExit("evidence class registry drifted")


def _source_audit(
    frozen: dict[str, Any],
    vllm_source: Path,
) -> list[dict[str, str]]:
    rows = []
    as_of = frozen["as_of_commit"]
    for name, expected in frozen["source_audit_sha256"].items():
        if name.startswith("simllm/"):
            actual = _sha256_bytes(_git_blob(as_of, name))
            scope = f"git-blob:{as_of}"
        else:
            relative = name.removeprefix("vllm/")
            actual = _sha256(vllm_source / relative)
            scope = "installed-vllm-source"
        if actual != expected:
            raise SystemExit(
                f"source audit hash disagrees for {name}: {actual} != {expected}"
            )
        rows.append(
            {
                "path": name,
                "sha256": actual,
                "validation_scope": scope,
            }
        )
    return rows


def _vllm_version(vllm_python: Path) -> str:
    completed = subprocess.run(
        [render_cli_path(vllm_python), "-c", "import vllm; print(vllm.__version__)"],
        check=True,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    return completed.stdout.strip().splitlines()[-1]


def check_registry(args: argparse.Namespace) -> dict[str, Any]:
    """Validate frozen inputs and return the provenance rows."""

    if args.run_dir.exists():
        raise SystemExit(f"selected run directory already exists: {args.run_dir}")
    frozen = _load_json(EXPECTATIONS_PATH)
    _validate_frozen_arithmetic(frozen)
    _require_implementation_ancestor()
    frontend = frozen["frontend"]
    if frontend["name"] != "vllm" or frontend["version"] != VLLM_VERSION:
        raise SystemExit("frontend identity drifted")
    if frontend["model_id"] != MODEL_ID:
        raise SystemExit("model identity drifted")
    if frontend["model_revision"] != MODEL_REVISION:
        raise SystemExit("model revision drifted")
    if _sha256(TRACE_PATH) != frontend["fixture_sha256"]:
        raise SystemExit("prompt fixture hash disagrees")
    if _sha256(args.model_config) != frontend["model_config_sha256"]:
        raise SystemExit("model config hash disagrees")
    if _vllm_version(args.vllm_python) != VLLM_VERSION:
        raise SystemExit("installed vLLM version disagrees")
    source_rows = _source_audit(frozen, args.vllm_source)
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise SystemExit("HF_HUB_OFFLINE=1 is required")
    if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") != "0":
        raise SystemExit("VLLM_ENABLE_V1_MULTIPROCESSING=0 is required")
    return {
        "expectations_commit": EXPECTATIONS_COMMIT,
        "implementation_commit": IMPLEMENTATION_COMMIT,
        "run_head": _git_head(),
        "fixture_sha256": frontend["fixture_sha256"],
        "model_config_sha256": frontend["model_config_sha256"],
        "source_audit": source_rows,
        "vllm_version": VLLM_VERSION,
    }


def _granite_dims() -> Any:
    from simllm.compute import ModelDims

    return ModelDims(
        num_layers=24,
        hidden_size=1024,
        intermediate_size=64,
        num_heads=2,
        num_kv_heads=1,
        head_size=64,
        vocab_size=49155,
        dtype_bytes=2,
        num_experts=32,
        top_k=8,
        moe_intermediate_size=64,
        local_num_experts=32,
    )


def _session_config(
    workdir: Path,
    *,
    prefill_engines: int = 1,
    decode_engines: int = 1,
) -> Any:
    from simllm.adapters.vllm.pd_session import VllmPdSessionConfig
    from simllm.core import DeclaredKvHandoffPolicy, KvHandoffGeometry

    return VllmPdSessionConfig(
        model=MODEL_ID,
        model_revision=MODEL_REVISION,
        workdir=workdir,
        dims=_granite_dims(),
        handoff_geometry=KvHandoffGeometry(24, 8, 64, 2),
        handoff_policy=DeclaredKvHandoffPolicy(HANDOFF_DURATIONS_PS[0]),
        prefill_engines=prefill_engines,
        decode_engines=decode_engines,
        tensor_parallel_size=8,
        max_model_len=64,
        num_gpu_blocks_override=64,
        max_num_seqs=8,
        token_id=512,
    )


def _prompt_tokens() -> tuple[int, ...]:
    for line in TRACE_PATH.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("row_type") == "request":
            tokens = tuple(row["input_token_ids"])
            if len(tokens) < max(PROMPT_LENGTHS):
                raise RuntimeError("frozen prompt fixture is too short")
            return tokens
    raise RuntimeError("frozen prompt fixture has no request row")


def _manifest_summary(prefill_nodes: int, decode_nodes: int) -> dict[str, Any]:
    from simllm.placement import RankMapper, disaggregated_manifests

    manifests = disaggregated_manifests(
        prefill_nodes=prefill_nodes,
        decode_nodes=decode_nodes,
        gpus_per_node=8,
        framework="vllm",
        framework_version=VLLM_VERSION,
    )
    ranks = manifests.placement.ranks
    mapper = RankMapper(manifests.placement)
    expected_prefill = prefill_nodes * 8
    expected_total = (prefill_nodes + decode_nodes) * 8
    findings = []
    if len(ranks) != expected_total:
        findings.append("rank cardinality")
    if [rank.global_rank for rank in ranks] != list(range(expected_total)):
        findings.append("dense rank order")
    if {rank.pool_role for rank in ranks[:expected_prefill]} != {"prefill"}:
        findings.append("prefill role interval")
    if {rank.pool_role for rank in ranks[expected_prefill:]} != {"decode"}:
        findings.append("decode role interval")
    if len(manifests.fabric.nodes) != prefill_nodes + decode_nodes:
        findings.append("fabric node cardinality")
    if any(len(node.gpus) != 8 or len(node.nics) != 8 for node in manifests.fabric.nodes):
        findings.append("per-node GPU or NIC cardinality")
    if any(
        not nic.fabric_location
        for node in manifests.fabric.nodes
        for nic in node.nics
    ):
        findings.append("physical fabric location")
    if [mapper.goal_rank(rank) for rank in range(expected_total)] != list(
        range(expected_total)
    ):
        findings.append("gpu-rank GOAL identity")
    for rank in ranks:
        if any(
            manifests.placement.by_rank(peer).pool_role != rank.pool_role
            for peer in rank.groups["dp"].global_ranks
        ):
            findings.append("cross-role data parallel group")
            break
    return {
        "prefill_nodes": prefill_nodes,
        "decode_nodes": decode_nodes,
        "ranks": len(ranks),
        "gpus": sum(len(node.gpus) for node in manifests.fabric.nodes),
        "nics": sum(len(node.nics) for node in manifests.fabric.nodes),
        "held": not findings,
        "findings": findings,
    }


def _fraction_json(value: Fraction | None) -> dict[str, int] | None:
    if value is None:
        return None
    return {"numerator": value.numerator, "denominator": value.denominator}


def _cell_key(prompt_tokens: int, handoff_ps: int) -> str:
    return f"prompt-{prompt_tokens}-handoff-{handoff_ps}"


def run_behavior(run_dir: Path) -> dict[str, Any]:
    from simllm.adapters.vllm.pd_session import VllmDisaggregatedSession
    from simllm.core import DeclaredKvHandoffPolicy

    prompt = _prompt_tokens()
    cells = []
    with VllmDisaggregatedSession(
        _session_config(run_dir / "behavior")
    ) as session:
        for prompt_length in PROMPT_LENGTHS:
            for handoff_ps in HANDOFF_DURATIONS_PS:
                request_id = _cell_key(prompt_length, handoff_ps)
                result = session.run_request(
                    request_id,
                    prompt[:prompt_length],
                    decode_output_tokens=DECODE_OUTPUT_TOKENS,
                    handoff_policy=DeclaredKvHandoffPolicy(handoff_ps),
                )
                timeline = result.timeline
                cells.append(
                    {
                        "label": request_id,
                        "prompt_tokens": prompt_length,
                        "handoff_ps": handoff_ps,
                        "kv_bytes": timeline.handoff.kv_bytes,
                        "ttft_ps": timeline.ttft_ps,
                        "tpot_ps": _fraction_json(timeline.tpot_ps),
                        "prefill_service_ps": timeline.prefill_service_ps,
                        "decode_admission_wait_ps": timeline.decode_admission_wait_ps,
                        "decode_first_token_service_ps": (
                            timeline.decode_first_token_service_ps
                        ),
                        "decomposition_total_ps": timeline.decomposition_total_ps,
                        "decomposition_residual_ps": (
                            timeline.ttft_ps - timeline.decomposition_total_ps
                        ),
                        "decode_token_ids": list(result.decode_token_ids),
                        "prefill_engine_id": result.prefill_engine_id,
                        "decode_engine_id": result.decode_engine_id,
                        "prefill_internal_request_id": (
                            result.prefill_internal_request_id
                        ),
                        "decode_internal_request_id": result.decode_internal_request_id,
                        "kv_transfer_params": result.kv_transfer_params,
                    }
                )
        engines = (*session.prefill_engines, *session.decode_engines)
        clock_shared = all(engine.executor.clock is session.clock for engine in engines)
        engine_roles = [
            {
                "engine_id": engine.engine_id,
                "pool_role": engine.role.value,
                "executor_pool_role": engine.executor.config.pool_role,
                "simulated_workers": engine.simulated_worker_count,
                "connector_role": (
                    engine.llm.llm_engine.vllm_config.kv_transfer_config.kv_role
                ),
            }
            for engine in engines
        ]
        backend_runs = sum(
            outcome.backend_runs
            for engine in engines
            for outcome in engine.step_sink.locality_outcomes
        )
        locality_rows = sum(
            len(engine.step_sink.locality_outcomes) for engine in engines
        )
        collective_arms = sorted(
            {
                (row.envelope_id, row.arm)
                for engine in engines
                for row in engine.step_sink.collective_timing_outcomes
            }
        )
        step_latencies = [
            result.step_latency_ps
            for engine in engines
            for record, result in zip(
                engine.executor.step_records,
                engine.executor.step_results,
                strict=True,
            )
            if record.scheduled
        ]
    return {
        "cells": cells,
        "clock_shared": clock_shared,
        "engine_roles": engine_roles,
        "handoff_events": len(cells),
        "backend_runs": backend_runs,
        "locality_rows": locality_rows,
        "collective_arms": [list(row) for row in collective_arms],
        "step_latencies_ps": step_latencies,
    }


def _cell_map(cells: list[dict[str, Any]]) -> dict[tuple[int, int], dict[str, Any]]:
    return {(row["prompt_tokens"], row["handoff_ps"]): row for row in cells}


def analyze_behavior(behavior: dict[str, Any]) -> dict[str, Any]:
    cells = _cell_map(behavior["cells"])
    exact_rows = []
    for row in behavior["cells"]:
        exact_rows.append(
            {
                "label": row["label"],
                "held": row["decomposition_residual_ps"] == 0,
                "ttft_ps": row["ttft_ps"],
                "decomposition_total_ps": row["decomposition_total_ps"],
                "residual_ps": row["decomposition_residual_ps"],
            }
        )
    behavioral = []
    for prompt_length in PROMPT_LENGTHS:
        lower = cells[(prompt_length, HANDOFF_DURATIONS_PS[0])]
        upper = cells[(prompt_length, HANDOFF_DURATIONS_PS[1])]
        behavioral.append(
            {
                "family": "handoff-movement",
                "instance": f"prompt-{prompt_length}",
                "held": (
                    upper["ttft_ps"] - lower["ttft_ps"] == 100_000_000
                    and upper["prefill_service_ps"] == lower["prefill_service_ps"]
                    and upper["decode_admission_wait_ps"]
                    == lower["decode_admission_wait_ps"]
                    and upper["decode_first_token_service_ps"]
                    == lower["decode_first_token_service_ps"]
                    and upper["decode_token_ids"] == lower["decode_token_ids"]
                    and upper["tpot_ps"] == lower["tpot_ps"]
                ),
                "ttft_delta_ps": upper["ttft_ps"] - lower["ttft_ps"],
            }
        )
    for handoff_ps in HANDOFF_DURATIONS_PS:
        short = cells[(PROMPT_LENGTHS[0], handoff_ps)]
        long = cells[(PROMPT_LENGTHS[1], handoff_ps)]
        tpot_short = Fraction(**short["tpot_ps"])
        tpot_long = Fraction(**long["tpot_ps"])
        common = (
            long["kv_bytes"] == 2 * short["kv_bytes"]
            and tpot_long >= tpot_short
        )
        behavioral.extend(
            (
                {
                    "family": "prompt-movement",
                    "instance": f"prefill-service-handoff-{handoff_ps}",
                    "held": common
                    and long["prefill_service_ps"] > short["prefill_service_ps"],
                    "delta_ps": (
                        long["prefill_service_ps"] - short["prefill_service_ps"]
                    ),
                },
                {
                    "family": "prompt-movement",
                    "instance": f"ttft-handoff-{handoff_ps}",
                    "held": common and long["ttft_ps"] > short["ttft_ps"],
                    "delta_ps": long["ttft_ps"] - short["ttft_ps"],
                },
            )
        )
    return {
        "exact_oracle_rows": exact_rows,
        "behavioral_relations": behavioral,
        "behavioral_instances": len(behavioral),
        "behavioral_held": sum(row["held"] for row in behavioral),
    }


def _rss_kib() -> int:
    status = Path("/proc/self/status").read_text(encoding="utf-8")
    for line in status.splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1])
    raise RuntimeError("/proc/self/status has no VmRSS row")


def _peak_rss_kib() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def run_scale_child(args: argparse.Namespace) -> int:
    from simllm.adapters.vllm.pd_session import VllmDisaggregatedSession

    prefill_engines, decode_engines = args.scale_child
    started = time.perf_counter()
    baseline_current = _rss_kib()
    baseline_peak = _peak_rss_kib()
    rows = []

    def observe(engine: Any) -> None:
        rows.append(
            {
                "ordinal": len(rows),
                "engine_id": engine.engine_id,
                "pool_role": engine.role.value,
                "simulated_workers": engine.simulated_worker_count,
                "construction_seconds": engine.construction_seconds,
                "elapsed_seconds": time.perf_counter() - started,
                "current_rss_kib": _rss_kib(),
                "peak_rss_kib": _peak_rss_kib(),
            }
        )

    session = VllmDisaggregatedSession(
        _session_config(
            args.run_dir,
            prefill_engines=prefill_engines,
            decode_engines=decode_engines,
        ),
        construction_observer=observe,
    )
    total_seconds = time.perf_counter() - started
    result = {
        "schema": SCALE_SCHEMA,
        "prefill_engines": prefill_engines,
        "decode_engines": decode_engines,
        "retained_engines": len(session.prefill_engines) + len(session.decode_engines),
        "baseline_current_rss_kib": baseline_current,
        "baseline_peak_rss_kib": baseline_peak,
        "final_current_rss_kib": _rss_kib(),
        "final_peak_rss_kib": _peak_rss_kib(),
        "total_construction_seconds": total_seconds,
        "engines": rows,
    }
    _write_json(args.scale_out, result)
    session.shutdown()
    return 0


def scale_child_command(
    args: argparse.Namespace,
    *,
    prefill_engines: int,
    decode_engines: int,
    run_dir: PurePath,
    output: PurePath,
) -> list[str]:
    return [
        render_cli_path(args.vllm_python),
        render_cli_path(Path(__file__).resolve()),
        "--run-dir",
        render_cli_path(run_dir),
        "--vllm-source",
        render_cli_path(args.vllm_source),
        "--model-config",
        render_cli_path(args.model_config),
        "--vllm-python",
        render_cli_path(args.vllm_python),
        "--scale-child",
        str(prefill_engines),
        str(decode_engines),
        "--scale-out",
        render_cli_path(output),
    ]


def run_scale_points(args: argparse.Namespace) -> list[dict[str, Any]]:
    scale_dir = args.run_dir / "scale"
    scale_dir.mkdir(parents=True, exist_ok=False)
    results = []
    for prefill_engines, decode_engines in SCALE_POINTS:
        label = f"{prefill_engines}p-{decode_engines}d"
        output = scale_dir / f"{label}.json"
        command = scale_child_command(
            args,
            prefill_engines=prefill_engines,
            decode_engines=decode_engines,
            run_dir=scale_dir / label,
            output=output,
        )
        completed = subprocess.run(command, check=False, env=os.environ.copy())
        if completed.returncode != 0:
            raise RuntimeError(
                f"scale child {label} failed with code {completed.returncode}"
            )
        results.append(_load_json(output))
    return results


def summarize_scale(cells: list[dict[str, Any]]) -> dict[str, Any]:
    summaries = []
    per_engine_memory = []
    per_engine_peak_memory = []
    per_engine_seconds = []
    for cell in cells:
        engines = cell["retained_engines"]
        peak_delta = cell["final_peak_rss_kib"] - cell["baseline_peak_rss_kib"]
        current_delta = (
            cell["final_current_rss_kib"] - cell["baseline_current_rss_kib"]
        )
        memory_per_engine = Fraction(peak_delta, engines)
        current_memory_per_engine = Fraction(current_delta, engines)
        seconds_per_engine = cell["total_construction_seconds"] / engines
        per_engine_memory.append(current_memory_per_engine)
        per_engine_peak_memory.append(memory_per_engine)
        per_engine_seconds.append(seconds_per_engine)
        summaries.append(
            {
                "prefill_engines": cell["prefill_engines"],
                "decode_engines": cell["decode_engines"],
                "retained_engines": engines,
                "peak_rss_delta_kib": peak_delta,
                "current_rss_delta_kib": current_delta,
                "current_rss_kib_per_engine": float(current_memory_per_engine),
                "peak_rss_kib_per_engine": float(memory_per_engine),
                "total_construction_seconds": cell["total_construction_seconds"],
                "construction_seconds_per_engine": seconds_per_engine,
            }
        )
    return {
        "cells": summaries,
        "target_engine_count": 56,
        "memory_extrapolation_basis": "current-resident-set-size",
        "target_incremental_current_rss_kib_range": [
            float(min(per_engine_memory) * 56),
            float(max(per_engine_memory) * 56),
        ],
        "target_incremental_peak_rss_kib_range": [
            float(min(per_engine_peak_memory) * 56),
            float(max(per_engine_peak_memory) * 56),
        ],
        "peak_high_watermark_censored": all(
            value == 0 for value in per_engine_peak_memory
        ),
        "target_sequential_construction_seconds_range": [
            min(per_engine_seconds) * 56,
            max(per_engine_seconds) * 56,
        ],
        "assumptions": [
            "model metadata and cached revision remain unchanged",
            "each engine retains 64 KV blocks at tensor parallel width eight",
            "allocator and in-process topology scale like the measured cells",
            "engines construct sequentially and remain resident",
        ],
        "fit_claim": False,
        "reporting_chronology": (
            "Post-specified after scored-v1 exposed an import-time peak above "
            "every retained-engine sample; acceptance relations are unchanged."
        ),
    }


def fatal_guards(
    behavior: dict[str, Any],
    analysis: dict[str, Any],
    manifests: list[dict[str, Any]],
    scale_cells: list[dict[str, Any]],
    frozen: dict[str, Any],
) -> dict[str, Any]:
    findings = []

    def require(name: str, condition: bool, detail: object) -> None:
        if not condition:
            findings.append({"guard": name, "detail": detail})

    require("shared-clock", behavior["clock_shared"], behavior["engine_roles"])
    expected_roles = [
        ("prefill", "prefill", "kv_producer", 8),
        ("decode", "decode", "kv_consumer", 8),
    ]
    actual_roles = [
        (
            row["pool_role"],
            row["executor_pool_role"],
            row["connector_role"],
            row["simulated_workers"],
        )
        for row in behavior["engine_roles"]
    ]
    require("engine-roles", actual_roles == expected_roles, actual_roles)
    request_ids = [row["label"] for row in behavior["cells"]]
    remote_ids = [
        row["kv_transfer_params"]["remote_request_id"]
        for row in behavior["cells"]
    ]
    require("request-identity", request_ids == remote_ids, remote_ids)
    require(
        "one-handoff-per-request",
        behavior["handoff_events"] == len(behavior["cells"]),
        behavior["handoff_events"],
    )
    require(
        "ttft-decomposition",
        all(row["held"] for row in analysis["exact_oracle_rows"]),
        analysis["exact_oracle_rows"],
    )
    bounds = frozen["physical_bounds"]["handoff_bounds_ps"]
    for row in behavior["cells"]:
        interval = bounds[str(row["prompt_tokens"])]
        require(
            "handoff-physical-interval",
            interval["floor"] <= row["handoff_ps"] <= interval["ceiling"],
            {"label": row["label"], "interval": interval},
        )
    require("no-packet-backend", behavior["backend_runs"] == 0, behavior["backend_runs"])
    require(
        "declared-local-collective-arm",
        behavior["collective_arms"] == [["intra-node-fixed-cost-v1", "lower"]],
        behavior["collective_arms"],
    )
    service_bounds = frozen["physical_bounds"]["per_step_service_ps"]
    require(
        "step-service-bounds",
        bool(behavior["step_latencies_ps"])
        and all(
            service_bounds["floor"] <= value <= service_bounds["ceiling"]
            for value in behavior["step_latencies_ps"]
        ),
        behavior["step_latencies_ps"],
    )
    require("placement-structure", all(row["held"] for row in manifests), manifests)
    for cell in scale_cells:
        expected_engines = cell["prefill_engines"] + cell["decode_engines"]
        require(
            "scale-cardinality",
            cell["retained_engines"] == expected_engines
            and len(cell["engines"]) == expected_engines,
            cell,
        )
        roles = [row["pool_role"] for row in cell["engines"]]
        require(
            "scale-roles",
            roles
            == ["prefill"] * cell["prefill_engines"]
            + ["decode"] * cell["decode_engines"],
            roles,
        )
        require(
            "scale-worker-count",
            all(row["simulated_workers"] == 8 for row in cell["engines"]),
            cell["engines"],
        )
        peaks = [cell["baseline_peak_rss_kib"]] + [
            row["peak_rss_kib"] for row in cell["engines"]
        ]
        require(
            "scale-monotonic-peak",
            peaks == sorted(peaks),
            peaks,
        )
    return {"status": "PASS" if not findings else "VOID", "findings": findings}


def _decode_rates(behavior: dict[str, Any]) -> list[float]:
    return [
        float(Fraction(PS_PER_SECOND, 1) / Fraction(**row["tpot_ps"]))
        for row in behavior["cells"]
    ]


def run_study(args: argparse.Namespace, provenance: dict[str, Any]) -> int:
    args.run_dir.mkdir(parents=True, exist_ok=False)
    frozen = _load_json(EXPECTATIONS_PATH)
    try:
        manifests = [_manifest_summary(1, 1), _manifest_summary(16, 40)]
        behavior = run_behavior(args.run_dir)
        analysis = analyze_behavior(behavior)
        scale_cells = run_scale_points(args)
        scale = summarize_scale(scale_cells)
        guards = fatal_guards(behavior, analysis, manifests, scale_cells, frozen)
        decode_rates = _decode_rates(behavior)
        rate_held = all(10 <= value <= 100_000 for value in decode_rates)
        if not rate_held:
            guards["status"] = "VOID"
            guards["findings"].append(
                {"guard": "decode-rate", "detail": decode_rates}
            )
        behavioral_pass = (
            analysis["behavioral_held"] == analysis["behavioral_instances"]
        )
        status = guards["status"]
        if status != "VOID":
            status = "PASS" if behavioral_pass else "FAIL"
        result = {
            "schema": RESULT_SCHEMA,
            "status": status,
            "provenance": provenance,
            "behavior": behavior,
            "analysis": analysis,
            "fatal_guards": guards,
            "placement": manifests,
            "scale": scale,
            "decode_tokens_per_second": decode_rates,
            "scope": {
                "live_prefill_engines": 1,
                "live_decode_engines": 1,
                "simulated_workers_per_engine": 8,
                "target_manifest_only": True,
                "packet_rendered_kv": False,
                "lookup_pricing": False,
                "sglang": False,
            },
        }
        _write_json(args.run_dir / "result.json", result)
        print(
            f"{status}: exact={sum(row['held'] for row in analysis['exact_oracle_rows'])}/"
            f"{len(analysis['exact_oracle_rows'])}; behavioral="
            f"{analysis['behavioral_held']}/{analysis['behavioral_instances']}; "
            f"ttft_ps={behavior['cells'][0]['ttft_ps']}"
        )
        return 0 if status == "PASS" else (2 if status == "VOID" else 1)
    except BaseException as exc:
        void = {
            "schema": RESULT_SCHEMA,
            "status": "VOID",
            "provenance": provenance,
            "fatal_guards": {
                "status": "VOID",
                "findings": [
                    {
                        "guard": "study-execution",
                        "detail": f"{type(exc).__name__}: {exc}",
                    }
                ],
            },
        }
        _write_json(args.run_dir / "result.json", void)
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--vllm-source", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--vllm-python", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--scale-child", nargs=2, type=int, metavar=("PREFILL", "DECODE"))
    parser.add_argument("--scale-out", type=Path)
    args = parser.parse_args()
    if args.scale_child is not None:
        if args.scale_out is None:
            parser.error("--scale-child requires --scale-out")
        if any(value < 1 for value in args.scale_child):
            parser.error("--scale-child counts must be positive")
    elif args.scale_out is not None:
        parser.error("--scale-out requires --scale-child")
    return args


def main() -> int:
    args = _parse_args()
    provenance = check_registry(args)
    if args.check_only:
        print(
            "check-only: frozen disaggregated session registry passed; "
            "no artifacts written"
        )
        return 0
    if args.scale_child is not None:
        return run_scale_child(args)
    return run_study(args, provenance)


if __name__ == "__main__":
    raise SystemExit(main())
