"""Run the frozen concurrent disaggregated serving study."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from fractions import Fraction
from pathlib import Path, PurePath
from typing import Any

STUDY_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = STUDY_DIR.parents[1]
EXPECTATIONS_PATH = STUDY_DIR / "expectations.json"
TRACE_PATH = REPOSITORY_ROOT / "examples/preplay_trace_v1/granite_length_cap.jsonl"

EXPECTATIONS_COMMIT = "7536e08b32009951470f310e4f459216c7212dbc"
IMPLEMENTATION_COMMIT = "d6bd2cd520dfc731bc59e25928128d6b77918045"
RESULT_SCHEMA = "simllm-pd-session-concurrent-study-result-v1"
MODEL_ID = "ibm-granite/granite-3.0-1b-a400m-instruct"
MODEL_REVISION = "ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445"
VLLM_VERSION = "0.27.1"
PROMPT_LENGTHS = (8, 16)
OFFERED_LOADS = (8, 16, 32)
INTERARRIVAL_PS = (125_000_000_000, 62_500_000_000, 31_250_000_000)
POOL_RATIOS = ((1, 1), (1, 2), (2, 1))
REQUESTS_PER_CELL = 8
DECODE_OUTPUT_TOKENS = 4
HANDOFF_PS = 100_000_000
PS_PER_SECOND = 1_000_000_000_000
RUN_ROOT_ENV = "SIMLLM_VLLM35_RUN_ROOT"


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
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


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


def _require_clean_worktree() -> None:
    completed = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    if completed.stdout:
        raise SystemExit("the scored run requires a clean tracked worktree")


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


def _fraction_json(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _fraction(value: dict[str, int]) -> Fraction:
    return Fraction(value["numerator"], value["denominator"])


def _validate_frozen_arithmetic(frozen: dict[str, Any]) -> None:
    deployment = frozen["deployment"]
    sweep = frozen["request_sweep"]
    curve = frozen["curve_record"]
    evidence = frozen["evidence_classes"]
    if EXPECTATIONS_COMMIT != "7536e08b32009951470f310e4f459216c7212dbc":
        raise SystemExit("expectations commit literal drifted")
    if tuple(
        (row["prefill_engines"], row["decode_engines"])
        for row in deployment["pool_ratios"]
    ) != POOL_RATIOS:
        raise SystemExit("pool ratio sweep drifted")
    if tuple(sweep["offered_load_requests_per_second"]) != OFFERED_LOADS:
        raise SystemExit("offered load sweep drifted")
    if tuple(sweep["interarrival_ps"]) != INTERARRIVAL_PS:
        raise SystemExit("interarrival sweep drifted")
    if any(load * interval != PS_PER_SECOND for load, interval in zip(
        OFFERED_LOADS,
        INTERARRIVAL_PS,
        strict=True,
    )):
        raise SystemExit("offered load and interarrival arithmetic disagrees")
    if tuple(sweep["prompt_tokens"]) != PROMPT_LENGTHS:
        raise SystemExit("prompt sweep drifted")
    if sweep["requests_per_cell"] != REQUESTS_PER_CELL:
        raise SystemExit("requests per cell drifted")
    if sweep["decode_output_tokens_per_request"] != DECODE_OUTPUT_TOKENS:
        raise SystemExit("decode output length drifted")
    if sweep["handoff_ps"] != HANDOFF_PS:
        raise SystemExit("handoff constant drifted")
    if sweep["cells"] != len(POOL_RATIOS) * len(PROMPT_LENGTHS) * len(OFFERED_LOADS):
        raise SystemExit("cell count drifted")
    if curve["schema"] != "simllm-deployment-curve-v1":
        raise SystemExit("curve schema drifted")
    if evidence != {
        "behavioral_families": 4,
        "curve_families": 6,
        "curve_points": 18,
        "exact_conservation": "fatal-unscored",
        "baseline_byte_identity": "fatal-unscored",
        "source_and_runtime_identity": "fatal-unscored",
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
        rows.append({"path": name, "sha256": actual, "scope": scope})
    return rows


def _baseline_audit(frozen: dict[str, Any]) -> list[dict[str, str]]:
    rows = []
    for name, expected in frozen["core51_baseline_sha256"].items():
        actual = _sha256(REPOSITORY_ROOT / name)
        if actual != expected:
            raise SystemExit(
                f"CORE-51 baseline hash disagrees for {name}: {actual} != {expected}"
            )
        rows.append({"path": name, "sha256": actual})
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
    """Validate frozen inputs without importing SimLLM behavior modules."""

    if args.run_dir.exists():
        raise SystemExit(f"selected run directory already exists: {args.run_dir}")
    frozen = _load_json(EXPECTATIONS_PATH)
    _validate_frozen_arithmetic(frozen)
    _require_implementation_ancestor()
    frontend = frozen["frontend"]
    if frontend["name"] != "vllm" or frontend["version"] != VLLM_VERSION:
        raise SystemExit("frontend identity drifted")
    if frontend["model_id"] != MODEL_ID or frontend["model_revision"] != MODEL_REVISION:
        raise SystemExit("model identity drifted")
    if _sha256(TRACE_PATH) != frontend["fixture_sha256"]:
        raise SystemExit("prompt fixture hash disagrees")
    if _sha256(args.model_config) != frontend["model_config_sha256"]:
        raise SystemExit("model configuration hash disagrees")
    if _vllm_version(args.vllm_python) != VLLM_VERSION:
        raise SystemExit("installed vLLM version disagrees")
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise SystemExit("HF_HUB_OFFLINE=1 is required")
    if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") != "0":
        raise SystemExit("VLLM_ENABLE_V1_MULTIPROCESSING=0 is required")
    return {
        "expectations_commit": EXPECTATIONS_COMMIT,
        "implementation_commit": IMPLEMENTATION_COMMIT,
        "run_head": _git_head(),
        "source_audit": _source_audit(frozen, args.vllm_source),
        "baseline_audit": _baseline_audit(frozen),
        "vllm_version": VLLM_VERSION,
        "model_config_sha256": frontend["model_config_sha256"],
        "fixture_sha256": frontend["fixture_sha256"],
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
    prefill_engines: int,
    decode_engines: int,
) -> Any:
    from simllm.adapters.vllm.pd_session import VllmPdSessionConfig
    from simllm.core import DeclaredKvHandoffPolicy, KvHandoffGeometry

    return VllmPdSessionConfig(
        model=MODEL_ID,
        model_revision=MODEL_REVISION,
        workdir=workdir,
        dims=_granite_dims(),
        handoff_geometry=KvHandoffGeometry(24, 8, 64, 2),
        handoff_policy=DeclaredKvHandoffPolicy(HANDOFF_PS),
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


def _request_id(
    prefill_engines: int,
    decode_engines: int,
    prompt_tokens: int,
    offered_load: int,
    index: int,
) -> str:
    return (
        f"p{prefill_engines}-d{decode_engines}-prompt{prompt_tokens}-"
        f"load{offered_load}-request{index}"
    )


def _cell_observation(
    session: Any,
    *,
    prompt: tuple[int, ...],
    prompt_tokens: int,
    offered_load: int,
    interarrival_ps: int,
    prefill_engines: int,
    decode_engines: int,
) -> tuple[dict[str, Any], Any]:
    from simllm.adapters.vllm.pd_session import VllmPdRequest

    admitted_start = session.clock.now_ps
    requests = tuple(
        VllmPdRequest(
            request_id=_request_id(
                prefill_engines,
                decode_engines,
                prompt_tokens,
                offered_load,
                index,
            ),
            prompt_token_ids=prompt[:prompt_tokens],
            decode_output_tokens=DECODE_OUTPUT_TOKENS,
            admitted_at_ps=admitted_start + index * interarrival_ps,
        )
        for index in range(REQUESTS_PER_CELL)
    )
    result = session.run_requests(requests)
    point = result.curve_point(Fraction(offered_load))
    rows = []
    for expected, observed in zip(requests, result.requests, strict=True):
        timeline = observed.timeline
        tpot = timeline.tpot_ps
        rows.append(
            {
                "request_id": timeline.request_id,
                "expected_admitted_at_ps": expected.admitted_at_ps,
                "timeline": timeline.to_json(),
                "prefill_engine_id": observed.prefill_engine_id,
                "decode_engine_id": observed.decode_engine_id,
                "prefill_internal_request_id": observed.prefill_internal_request_id,
                "decode_internal_request_id": observed.decode_internal_request_id,
                "decode_token_ids": list(observed.decode_token_ids),
                "kv_transfer_params": dict(observed.kv_transfer_params),
                "prefill_step_count": len(observed.prefill_records),
                "decode_step_count": len(observed.decode_records),
                "tpot_ps": None if tpot is None else _fraction_json(tpot),
            }
        )
    return (
        {
            "prefill_engines": prefill_engines,
            "decode_engines": decode_engines,
            "prompt_tokens": prompt_tokens,
            "offered_load_requests_per_second": offered_load,
            "interarrival_ps": interarrival_ps,
            "requests": rows,
            "prefill_batches": [list(batch) for batch in result.prefill_batches],
            "decode_batches": [list(batch) for batch in result.decode_batches],
            "maximum_prefill_batch_size": result.maximum_prefill_batch_size,
            "maximum_decode_batch_size": result.maximum_decode_batch_size,
            "curve_point": point.to_json(),
        },
        point,
    )


def run_observation(run_dir: Path) -> dict[str, Any]:
    from simllm.adapters.vllm.pd_session import (
        VllmDisaggregatedSession,
        VllmPdCurveRecord,
    )

    prompt = _prompt_tokens()
    cells = []
    curves = []
    baseline = None
    for prefill_engines, decode_engines in POOL_RATIOS:
        ratio_dir = run_dir / f"p{prefill_engines}-d{decode_engines}"
        with VllmDisaggregatedSession(
            _session_config(
                ratio_dir,
                prefill_engines=prefill_engines,
                decode_engines=decode_engines,
            )
        ) as session:
            if (prefill_engines, decode_engines) == (1, 1):
                baseline_result = session.run_request(
                    "core51-live-baseline-control",
                    prompt[:8],
                    decode_output_tokens=DECODE_OUTPUT_TOKENS,
                )
                baseline = baseline_result.to_json()
            for prompt_tokens in PROMPT_LENGTHS:
                points = []
                for offered_load, interarrival_ps in zip(
                    OFFERED_LOADS,
                    INTERARRIVAL_PS,
                    strict=True,
                ):
                    cell, point = _cell_observation(
                        session,
                        prompt=prompt,
                        prompt_tokens=prompt_tokens,
                        offered_load=offered_load,
                        interarrival_ps=interarrival_ps,
                        prefill_engines=prefill_engines,
                        decode_engines=decode_engines,
                    )
                    cells.append(cell)
                    points.append(point)
                curves.append(
                    VllmPdCurveRecord(
                        configuration_id=(
                            f"p{prefill_engines}-d{decode_engines}-"
                            f"prompt{prompt_tokens}"
                        ),
                        prefill_engines=prefill_engines,
                        decode_engines=decode_engines,
                        prompt_tokens=prompt_tokens,
                        points=tuple(points),
                    ).to_json()
                )
    if baseline is None:
        raise RuntimeError("the one-request baseline control did not run")
    return {"baseline": baseline, "cells": cells, "curves": curves}


def _cell_key(cell: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        cell["prefill_engines"],
        cell["decode_engines"],
        cell["prompt_tokens"],
        cell["offered_load_requests_per_second"],
    )


def analyze_observation(observation: dict[str, Any]) -> dict[str, Any]:
    """Apply the frozen fatal guards and scored relations."""

    fatal_findings = []
    baseline = observation["baseline"]
    baseline_expected = {
        "ttft_ps": 273_376_000,
        "tpot_ps": {"numerator": 77_952_000, "denominator": 1},
        "prefill_service_ps": 95_424_000,
        "handoff_ps": HANDOFF_PS,
        "decode_first_token_service_ps": 77_952_000,
    }
    baseline_observed = {
        "ttft_ps": baseline["ttft_ps"],
        "tpot_ps": baseline["tpot_ps"],
        "prefill_service_ps": baseline["decomposition"]["prefill_service_ps"],
        "handoff_ps": baseline["decomposition"]["handoff_ps"],
        "decode_first_token_service_ps": baseline["decomposition"][
            "decode_first_token_service_ps"
        ],
    }
    if baseline_observed != baseline_expected:
        fatal_findings.append(
            {
                "guard": "core51-live-baseline-timestamps",
                "expected": baseline_expected,
                "observed": baseline_observed,
            }
        )

    cells = observation["cells"]
    if len(cells) != 18 or len({_cell_key(cell) for cell in cells}) != 18:
        fatal_findings.append({"guard": "complete-unique-cell-registry"})
    exact_rows = []
    all_prefill_internal = []
    all_decode_internal = []
    minimum_step_ps = None
    maximum_step_ps = None
    minimum_decode_rate = None
    maximum_decode_rate = None
    for cell in cells:
        request_rows = cell["requests"]
        request_ids = [row["request_id"] for row in request_rows]
        handoff_ids = [row["timeline"]["request_id"] for row in request_rows]
        terminal_tokens = sum(len(row["decode_token_ids"]) for row in request_rows)
        residuals = [
            row["timeline"]["decomposition"]["total_ps"]
            - row["timeline"]["ttft_ps"]
            for row in request_rows
        ]
        stable = (
            len(request_rows) == REQUESTS_PER_CELL
            and len(request_ids) == len(set(request_ids))
            and request_ids == handoff_ids
            and all(
                row["timeline"]["admitted_at_ps"]
                == row["expected_admitted_at_ps"]
                for row in request_rows
            )
        )
        tokens = (
            terminal_tokens == REQUESTS_PER_CELL * DECODE_OUTPUT_TOKENS
            and all(
                len(row["decode_token_ids"]) == DECODE_OUTPUT_TOKENS
                for row in request_rows
            )
        )
        handoffs = all(
            row["timeline"]["handoff"]["kv_bytes"]
            == cell["prompt_tokens"] * 49_152
            for row in request_rows
        )
        local_pairs = [
            (
                row["prefill_internal_request_id"],
                row["decode_internal_request_id"],
            )
            for row in request_rows
        ]
        local_identity = all(left != right for left, right in local_pairs)
        all_prefill_internal.extend(left for left, _ in local_pairs)
        all_decode_internal.extend(right for _, right in local_pairs)
        curve = cell["curve_point"]
        curve_conservation = (
            curve["request_count"] == REQUESTS_PER_CELL
            and curve["output_token_count"]
            == REQUESTS_PER_CELL * DECODE_OUTPUT_TOKENS
        )
        held = (
            stable
            and tokens
            and handoffs
            and local_identity
            and curve_conservation
            and all(residual == 0 for residual in residuals)
        )
        exact_rows.append(
            {
                "cell": list(_cell_key(cell)),
                "held": held,
                "admissions": len(request_rows),
                "handoffs": len(handoff_ids),
                "terminals": len(request_rows),
                "terminal_tokens": terminal_tokens,
                "maximum_ttft_residual_ps": max(map(abs, residuals), default=0),
            }
        )
        if not held:
            fatal_findings.append(
                {"guard": "cell-conservation", "cell": list(_cell_key(cell))}
            )
        for row in request_rows:
            timeline = row["timeline"]
            tpot = _fraction(row["tpot_ps"])
            decode_rate = Fraction(PS_PER_SECOND, 1) / tpot
            services = (
                timeline["decomposition"]["prefill_service_ps"],
                timeline["decomposition"]["decode_first_token_service_ps"],
                tpot,
            )
            for service in services:
                minimum_step_ps = service if minimum_step_ps is None else min(
                    minimum_step_ps,
                    service,
                )
                maximum_step_ps = service if maximum_step_ps is None else max(
                    maximum_step_ps,
                    service,
                )
            minimum_decode_rate = (
                decode_rate
                if minimum_decode_rate is None
                else min(minimum_decode_rate, decode_rate)
            )
            maximum_decode_rate = (
                decode_rate
                if maximum_decode_rate is None
                else max(maximum_decode_rate, decode_rate)
            )

    if len(all_prefill_internal) != len(set(all_prefill_internal)):
        fatal_findings.append({"guard": "prefill-local-identity-reuse"})
    if len(all_decode_internal) != len(set(all_decode_internal)):
        fatal_findings.append({"guard": "decode-local-identity-reuse"})
    if minimum_step_ps is None or maximum_step_ps is None:
        fatal_findings.append({"guard": "missing-step-service"})
    elif not (1_000_000 <= minimum_step_ps <= maximum_step_ps <= 100_000_000_000):
        fatal_findings.append(
            {
                "guard": "step-service-physical-bounds",
                "minimum": _fraction_json(Fraction(minimum_step_ps)),
                "maximum": _fraction_json(Fraction(maximum_step_ps)),
            }
        )
    if minimum_decode_rate is None or maximum_decode_rate is None:
        fatal_findings.append({"guard": "missing-decode-rate"})
    elif not (
        Fraction(10) <= minimum_decode_rate <= maximum_decode_rate <= Fraction(100_000)
    ):
        fatal_findings.append(
            {
                "guard": "decode-rate-physical-bounds",
                "minimum": _fraction_json(minimum_decode_rate),
                "maximum": _fraction_json(maximum_decode_rate),
            }
        )

    by_key = {_cell_key(cell): cell for cell in cells}
    batching_rows = []
    for prefill_engines, decode_engines in POOL_RATIOS:
        high = [
            by_key[(prefill_engines, decode_engines, prompt_tokens, OFFERED_LOADS[-1])]
            for prompt_tokens in PROMPT_LENGTHS
        ]
        prefill_max = max(cell["maximum_prefill_batch_size"] for cell in high)
        decode_max = max(cell["maximum_decode_batch_size"] for cell in high)
        held = prefill_max >= 2 and decode_max >= 2
        batching_rows.append(
            {
                "family": "genuine-multi-request-batching",
                "pool_ratio": [prefill_engines, decode_engines],
                "maximum_prefill_batch_size": prefill_max,
                "maximum_decode_batch_size": decode_max,
                "held": held,
            }
        )
        if not held:
            fatal_findings.append(
                {
                    "guard": "missing-multi-request-batch",
                    "pool_ratio": [prefill_engines, decode_engines],
                }
            )

    throughput_rows = []
    delay_rows = []
    for prefill_engines, decode_engines in POOL_RATIOS:
        for prompt_tokens in PROMPT_LENGTHS:
            curve_cells = [
                by_key[(prefill_engines, decode_engines, prompt_tokens, load)]
                for load in OFFERED_LOADS
            ]
            throughput = [
                _fraction(
                    cell["curve_point"][
                        "aggregated_output_throughput_tokens_per_second"
                    ]
                )
                for cell in curve_cells
            ]
            delays = [
                _fraction(cell["curve_point"]["per_token_request_delay_ps"])
                for cell in curve_cells
            ]
            throughput_rows.append(
                {
                    "family": "throughput-nondecreasing-with-load",
                    "configuration": [prefill_engines, decode_engines, prompt_tokens],
                    "values": [_fraction_json(value) for value in throughput],
                    "held": throughput == sorted(throughput),
                }
            )
            delay_rows.append(
                {
                    "family": "delay-nondecreasing-with-load",
                    "configuration": [prefill_engines, decode_engines, prompt_tokens],
                    "values": [_fraction_json(value) for value in delays],
                    "held": delays == sorted(delays),
                }
            )

    prompt_rows = []
    for prefill_engines, decode_engines in POOL_RATIOS:
        for load in OFFERED_LOADS:
            short = by_key[(prefill_engines, decode_engines, 8, load)]
            long = by_key[(prefill_engines, decode_engines, 16, load)]
            short_bytes = {
                row["timeline"]["handoff"]["kv_bytes"] for row in short["requests"]
            }
            long_bytes = {
                row["timeline"]["handoff"]["kv_bytes"] for row in long["requests"]
            }
            held = short_bytes == {393_216} and long_bytes == {786_432}
            prompt_rows.append(
                {
                    "family": "prompt-kv-byte-doubling",
                    "pool_ratio": [prefill_engines, decode_engines],
                    "offered_load": load,
                    "held": held,
                }
            )

    behavioral = batching_rows + throughput_rows + delay_rows + prompt_rows
    behavioral_held = sum(row["held"] for row in behavioral)
    status = (
        "VOID"
        if fatal_findings
        else "PASS"
        if behavioral_held == len(behavioral)
        else "REFUTED"
    )
    return {
        "status": status,
        "fatal_guards": {
            "status": "HELD" if not fatal_findings else "VIOLATED",
            "findings": fatal_findings,
        },
        "exact_oracle_rows": exact_rows,
        "behavioral_relations": behavioral,
        "behavioral_held": behavioral_held,
        "behavioral_total": len(behavioral),
        "physical_sanity": {
            "minimum_step_service_ps": _fraction_json(Fraction(minimum_step_ps)),
            "maximum_step_service_ps": _fraction_json(Fraction(maximum_step_ps)),
            "minimum_decode_tokens_per_second": _fraction_json(minimum_decode_rate),
            "maximum_decode_tokens_per_second": _fraction_json(maximum_decode_rate),
        },
    }


def _validate_run_dir(run_dir: Path) -> None:
    configured = os.environ.get(RUN_ROOT_ENV)
    if not configured:
        raise SystemExit(f"{RUN_ROOT_ENV} must name the external run root")
    root = Path(configured).resolve()
    try:
        run_dir.resolve().relative_to(root)
    except ValueError as exc:
        raise SystemExit(f"run directory must remain under {RUN_ROOT_ENV}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--vllm-source", required=True, type=Path)
    parser.add_argument("--model-config", required=True, type=Path)
    parser.add_argument("--vllm-python", required=True, type=Path)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    provenance = check_registry(args)
    if args.check_only:
        print(
            "check-only validated 18 frozen cells, 6 curve records, source and "
            "runtime identity, and all CORE-51 baseline digests; no artifacts produced"
        )
        return
    _require_clean_worktree()
    _validate_run_dir(args.run_dir)
    args.run_dir.mkdir(parents=True, exist_ok=False)
    observation = run_observation(args.run_dir)
    analysis = analyze_observation(observation)
    result = {
        "schema": RESULT_SCHEMA,
        "provenance": provenance,
        "observation": observation,
        "analysis": analysis,
    }
    _write_json(args.run_dir / "result.json", result)
    print(json.dumps(analysis, indent=2, sort_keys=True))
    if analysis["status"] != "PASS":
        raise SystemExit(f"study status is {analysis['status']}")


if __name__ == "__main__":
    main()
