"""Run the frozen VLLM-39 concurrent-session load-delay study."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
from fractions import Fraction
from itertools import pairwise
from pathlib import Path, PurePath
from typing import Any

from queue_model import (
    HANDOFF_PS,
    HELD_OUT_CONFIGURATIONS,
    OFFERED_LOADS,
    OUTPUT_TOKENS,
    POOL_RATIOS,
    PROMPT_LENGTHS,
    PS_PER_SECOND,
    REQUESTS_PER_CELL,
    direction,
    fraction_json,
)

from simllm.calibration.batch_service_surface import (
    BatchServicePoint,
    compile_pool_local_batch_service_provider,
    interpolate_batch_service_ps,
)

STUDY_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = STUDY_DIR.parents[1]
EXPECTATIONS_PATH = STUDY_DIR / "expectations.json"
SURFACE_PATH = STUDY_DIR / "surface.json"
ACCESS_LEDGER_PATH = STUDY_DIR / "access_ledger.jsonl"
TRACE_PATH = REPOSITORY_ROOT / "examples/preplay_trace_v1/granite_length_cap.jsonl"
COMPARATOR_RUNNER_PATH = (
    REPOSITORY_ROOT / "examples/pd_session_concurrent_v1/run_study.py"
)

FREEZE_COMMIT = "121345e950b12a36018404084c7dcf9bd507f962"
EXPECTATIONS_SHA256 = "28cee81deffe771836b5c38d7fe605185f4dc31a953087c80288ceb7a3a84e22"
SURFACE_SHA256 = "26fc547d8b47ccec7108872e05fbedfe71ebb6229b88799ca254089d3f2b6e9d"
ACCESS_LEDGER_SHA256 = "0394d2789a11e8dc68c6d3a18c563d19f493d1d27c21d53b3ea74f37b3d14fec"
RESULT_SCHEMA = "simllm-pd-session-load-delay-result-v1"
MODEL_ID = "ibm-granite/granite-3.0-1b-a400m-instruct"
MODEL_REVISION = "ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445"
RUN_ROOT_ENV = "SIMLLM_VLLM39_RUN_ROOT"


def render_cli_path(path: PurePath) -> str:
    """Render executed paths with POSIX separators on every host."""

    return path.as_posix()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _require_freeze_ancestor() -> None:
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", FREEZE_COMMIT, "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(f"freeze commit {FREEZE_COMMIT} is not an ancestor")


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tracked_paths(prefix: str) -> tuple[Path, ...]:
    completed = subprocess.run(
        ["git", "ls-files", prefix],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return tuple(
        REPOSITORY_ROOT / line
        for line in completed.stdout.splitlines()
        if line
    )


def _lock(paths: tuple[Path, ...]) -> dict[str, Any]:
    rows = [
        {
            "path": path.relative_to(REPOSITORY_ROOT).as_posix(),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(paths)
    ]
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return {
        "artifact_count": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "manifest_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def preservation_locks() -> dict[str, Any]:
    """Recompute the three frozen artifact classes."""

    deployment = _tracked_paths("examples/deployment_curve_v1")
    flagship = tuple(
        path
        for path in deployment
        if path.name.startswith("flagship")
        or path.name.startswith("RESULTS")
        or "figures" in path.parts
    )
    return {
        "core51_one_request_control": _lock(
            _tracked_paths("examples/pd_session_v1")
        ),
        "deterministic_concurrent_comparator": _lock(
            _tracked_paths("examples/pd_session_concurrent_v1")
        ),
        "scored_flagship_artifacts": _lock(flagship),
        "flagship_selection_rule": "tracked deployment_curve_v1 basenames beginning flagship or RESULTS, plus every tracked figures member",
    }


def _validate_freeze(freeze: dict[str, Any]) -> None:
    if freeze["status"] != "EXPECTATIONS_ONLY":
        raise SystemExit("VLLM-39 freeze status drifted")
    if _sha256(EXPECTATIONS_PATH) != EXPECTATIONS_SHA256:
        raise SystemExit("VLLM-39 expectations bytes drifted")
    if _sha256(SURFACE_PATH) != SURFACE_SHA256:
        raise SystemExit("VLLM-39 surface bytes drifted")
    if _sha256(ACCESS_LEDGER_PATH) != ACCESS_LEDGER_SHA256:
        raise SystemExit("VLLM-39 access ledger bytes drifted")
    sweep = freeze["sweep"]
    if tuple(sweep["offered_load_requests_per_second"]) != OFFERED_LOADS:
        raise SystemExit("offered-load sweep drifted")
    if tuple(sweep["prompt_tokens"]) != PROMPT_LENGTHS:
        raise SystemExit("prompt sweep drifted")
    if tuple(map(tuple, sweep["pool_ratios"])) != POOL_RATIOS:
        raise SystemExit("pool-ratio sweep drifted")
    if sweep["requests_per_cell"] != REQUESTS_PER_CELL:
        raise SystemExit("requests per cell drifted")
    if sweep["decode_output_tokens_per_request"] != OUTPUT_TOKENS:
        raise SystemExit("output tokens drifted")
    if len(freeze["expected_segments"]) != 30:
        raise SystemExit("expected segment registry drifted")
    if len(freeze["held_out_prediction_bands"]) != 24:
        raise SystemExit("held-out band registry drifted")
    if freeze["exposure"]["clean_close_permitted"] is not False:
        raise SystemExit("contaminated exposure cannot permit clean closure")


def check_registry(args: argparse.Namespace) -> dict[str, Any]:
    """Validate all frozen inputs without constructing a vLLM engine."""

    if args.run_dir.exists():
        raise SystemExit(f"selected run directory already exists: {args.run_dir}")
    _require_freeze_ancestor()
    freeze = _load_json(EXPECTATIONS_PATH)
    _validate_freeze(freeze)
    observed_locks = preservation_locks()
    if observed_locks != freeze["preservation_locks"]:
        raise SystemExit("preservation locks changed before the run")
    comparator = _module(COMPARATOR_RUNNER_PATH, "vllm39_comparator_registry")
    comparator_provenance = comparator.check_registry(args)
    return {
        "freeze_commit": FREEZE_COMMIT,
        "expectations_sha256": EXPECTATIONS_SHA256,
        "surface_sha256": SURFACE_SHA256,
        "access_ledger_sha256": ACCESS_LEDGER_SHA256,
        "run_head": _git_head(),
        "exposure_status": freeze["exposure"]["status"],
        "candidate_acceptance_status": freeze["surface"]["acceptance_status"],
        "calibration_claim": False,
        "preservation_locks": observed_locks,
        "comparator_registry": comparator_provenance,
    }


def _surface_points(surface: dict[str, Any]) -> tuple[BatchServicePoint, ...]:
    return tuple(
        BatchServicePoint(
            batch_size=row["batch_size"],
            duration_ps=row["measured_service_ps"],
            uncertainty_fraction=(
                row["trimmed_coefficient_of_variation_ppm"] / 1_000_000
            ),
            entry_key_sha256=row["entry_key_sha256"],
            evidence_class=row["evidence_class"],
            split=row["split"],
        )
        for row in surface["points"]
    )


def _surface_provider(surface: dict[str, Any]):
    from simllm.compute import RooflineProvider

    return compile_pool_local_batch_service_provider(
        _surface_points(surface),
        record_sha256=surface["record_sha256"],
        acceptance_status=surface["acceptance_status"],
        campaign_id=surface["campaign_id"],
        coverage=surface["coverage"],
        record_device_kind_id=surface["record_device_kind_id"],
        pool="decode",
        comparator=RooflineProvider(efficiency=0.7),
    )


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
    decode_provider: Any | None,
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
        decode_provider=decode_provider,
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
        f"vllm39-p{prefill_engines}-d{decode_engines}-prompt{prompt_tokens}-"
        f"load{offered_load}-request{index}"
    )


def _cell_observation(
    session: Any,
    *,
    prompt: tuple[int, ...],
    prompt_tokens: int,
    offered_load: int,
    prefill_engines: int,
    decode_engines: int,
    points: tuple[BatchServicePoint, ...],
) -> tuple[dict[str, Any], Any]:
    from simllm.adapters.vllm.pd_session import VllmPdRequest

    admitted_start = session.clock.now_ps
    interarrival_ps = PS_PER_SECOND // offered_load
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
            decode_output_tokens=OUTPUT_TOKENS,
            admitted_at_ps=admitted_start + index * interarrival_ps,
        )
        for index in range(REQUESTS_PER_CELL)
    )
    result = session.run_requests(requests)
    point = result.curve_point(Fraction(offered_load))
    rows = []
    for expected, observed in zip(requests, result.requests, strict=True):
        timeline = observed.timeline
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
            }
        )
    decode_visits = sum(len(batch) for batch in result.decode_batches)
    amortized_service = Fraction(
        sum(
            interpolate_batch_service_ps(points, len(batch))
            for batch in result.decode_batches
        ),
        decode_visits,
    )
    pricing = result.requests[0].compute_pricing
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
            "amortized_decode_batch_service_per_token_ps": fraction_json(
                amortized_service
            ),
            "compute_pricing": pricing,
            "curve_point": point.to_json(),
        },
        point,
    )


def _core51_control(run_dir: Path, prompt: tuple[int, ...]) -> dict[str, Any]:
    from simllm.adapters.vllm.pd_session import VllmDisaggregatedSession

    with VllmDisaggregatedSession(
        _session_config(
            run_dir / "core51-control",
            prefill_engines=1,
            decode_engines=1,
            decode_provider=None,
        )
    ) as session:
        return session.run_request(
            "vllm39-core51-one-request-control",
            prompt[:8],
            decode_output_tokens=OUTPUT_TOKENS,
        ).to_json()


def run_observation(run_dir: Path) -> dict[str, Any]:
    from simllm.adapters.vllm.pd_session import (
        VllmDisaggregatedSession,
        VllmPdCurveRecord,
    )

    prompt = _prompt_tokens()
    surface = _load_json(SURFACE_PATH)
    points = _surface_points(surface)
    cells = []
    curves = []
    control = _core51_control(run_dir, prompt)
    for prefill_engines, decode_engines in POOL_RATIOS:
        provider = _surface_provider(surface)
        ratio_dir = run_dir / f"p{prefill_engines}-d{decode_engines}"
        with VllmDisaggregatedSession(
            _session_config(
                ratio_dir,
                prefill_engines=prefill_engines,
                decode_engines=decode_engines,
                decode_provider=provider,
            )
        ) as session:
            for prompt_tokens in PROMPT_LENGTHS:
                curve_points = []
                for offered_load in OFFERED_LOADS:
                    cell, point = _cell_observation(
                        session,
                        prompt=prompt,
                        prompt_tokens=prompt_tokens,
                        offered_load=offered_load,
                        prefill_engines=prefill_engines,
                        decode_engines=decode_engines,
                        points=points,
                    )
                    cells.append(cell)
                    curve_points.append(point)
                curves.append(
                    VllmPdCurveRecord(
                        configuration_id=(
                            f"p{prefill_engines}-d{decode_engines}-"
                            f"prompt{prompt_tokens}"
                        ),
                        prefill_engines=prefill_engines,
                        decode_engines=decode_engines,
                        prompt_tokens=prompt_tokens,
                        points=tuple(curve_points),
                    ).to_json()
                )
    return {"core51_control": control, "cells": cells, "curves": curves}


def _fraction(value: dict[str, int]) -> Fraction:
    return Fraction(value["numerator"], value["denominator"])


def _cell_key(cell: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        cell["prefill_engines"],
        cell["decode_engines"],
        cell["prompt_tokens"],
        cell["offered_load_requests_per_second"],
    )


def _control_verdict(control: dict[str, Any]) -> dict[str, Any]:
    observed = {
        "ttft_ps": control["ttft_ps"],
        "tpot_ps": control["tpot_ps"],
        "prefill_service_ps": control["decomposition"]["prefill_service_ps"],
        "handoff_ps": control["decomposition"]["handoff_ps"],
        "decode_first_token_service_ps": control["decomposition"][
            "decode_first_token_service_ps"
        ],
    }
    expected = {
        "ttft_ps": 273_376_000,
        "tpot_ps": {"numerator": 77_952_000, "denominator": 1},
        "prefill_service_ps": 95_424_000,
        "handoff_ps": HANDOFF_PS,
        "decode_first_token_service_ps": 77_952_000,
    }
    return {"expected": expected, "observed": observed, "held": observed == expected}


def analyze_observation(
    observation: dict[str, Any],
    freeze: dict[str, Any],
) -> dict[str, Any]:
    """Apply frozen conservation, direction and held-out-band rules."""

    fatal_findings = []
    control = _control_verdict(observation["core51_control"])
    if not control["held"]:
        fatal_findings.append({"guard": "core51-one-request-control"})
    cells = observation["cells"]
    expected_cell_count = len(POOL_RATIOS) * len(PROMPT_LENGTHS) * len(OFFERED_LOADS)
    if len(cells) != expected_cell_count:
        fatal_findings.append({"guard": "complete-cell-registry"})
    if len({_cell_key(cell) for cell in cells}) != expected_cell_count:
        fatal_findings.append({"guard": "unique-cell-registry"})
    exact_rows = []
    all_prefill_ids = []
    all_decode_ids = []
    for cell in cells:
        requests = cell["requests"]
        residuals = [
            row["timeline"]["decomposition"]["total_ps"]
            - row["timeline"]["ttft_ps"]
            for row in requests
        ]
        terminal_tokens = sum(len(row["decode_token_ids"]) for row in requests)
        all_prefill_ids.extend(row["prefill_internal_request_id"] for row in requests)
        all_decode_ids.extend(row["decode_internal_request_id"] for row in requests)
        pricing = cell["compute_pricing"]
        decode_pricing = None if pricing is None else pricing["decode"]
        pricing_held = (
            pricing is not None
            and pricing["prefill"] is None
            and decode_pricing["record_sha256"]
            == freeze["surface"]["record_sha256"]
            and decode_pricing["acceptance_status"] == "candidate"
            and decode_pricing["calibration_claim"] is False
        )
        held = (
            len(requests) == REQUESTS_PER_CELL
            and len({row["request_id"] for row in requests}) == REQUESTS_PER_CELL
            and all(
                row["timeline"]["admitted_at_ps"]
                == row["expected_admitted_at_ps"]
                for row in requests
            )
            and terminal_tokens == REQUESTS_PER_CELL * OUTPUT_TOKENS
            and all(residual == 0 for residual in residuals)
            and cell["curve_point"]["request_count"] == REQUESTS_PER_CELL
            and cell["curve_point"]["output_token_count"]
            == REQUESTS_PER_CELL * OUTPUT_TOKENS
            and pricing_held
        )
        mean_prefill_queue = Fraction(
            sum(
                row["timeline"]["decomposition"]["prefill_queue_ps"]
                for row in requests
            ),
            len(requests),
        )
        mean_decode_wait = Fraction(
            sum(
                row["timeline"]["decomposition"]["decode_admission_wait_ps"]
                for row in requests
            ),
            len(requests),
        )
        exact_rows.append(
            {
                "cell": list(_cell_key(cell)),
                "held": held,
                "admissions": len(requests),
                "handoffs": len(requests),
                "terminals": len(requests),
                "terminal_tokens": terminal_tokens,
                "maximum_ttft_residual_ps": max(map(abs, residuals), default=0),
                "mean_prefill_queue_ps": fraction_json(mean_prefill_queue),
                "mean_decode_admission_wait_ps": fraction_json(mean_decode_wait),
                "mean_scheduler_queue_wait_ps": fraction_json(
                    mean_prefill_queue + mean_decode_wait
                ),
                "amortized_decode_batch_service_per_token_ps": cell[
                    "amortized_decode_batch_service_per_token_ps"
                ],
            }
        )
        if not held:
            fatal_findings.append(
                {"guard": "cell-conservation-or-pricing", "cell": list(_cell_key(cell))}
            )
    if len(all_prefill_ids) != len(set(all_prefill_ids)):
        fatal_findings.append({"guard": "prefill-local-identity-reuse"})
    if len(all_decode_ids) != len(set(all_decode_ids)):
        fatal_findings.append({"guard": "decode-local-identity-reuse"})

    by_key = {_cell_key(cell): cell for cell in cells}
    expected_segments = {
        (
            *row["configuration"],
            row["from_load"],
            row["to_load"],
        ): row["expected_direction"]
        for row in freeze["expected_segments"]
    }
    segment_rows = []
    for prefill, decode in POOL_RATIOS:
        for prompt in PROMPT_LENGTHS:
            for left_load, right_load in pairwise(OFFERED_LOADS):
                left = _fraction(
                    by_key[(prefill, decode, prompt, left_load)]["curve_point"][
                        "per_token_request_delay_ps"
                    ]
                )
                right = _fraction(
                    by_key[(prefill, decode, prompt, right_load)]["curve_point"][
                        "per_token_request_delay_ps"
                    ]
                )
                observed_direction = direction(left, right)
                expected_direction = expected_segments[
                    (prefill, decode, prompt, left_load, right_load)
                ]
                segment_rows.append(
                    {
                        "configuration": [prefill, decode, prompt],
                        "from_load": left_load,
                        "to_load": right_load,
                        "expected_direction": expected_direction,
                        "observed_direction": observed_direction,
                        "signed_movement_ps": fraction_json(right - left),
                        "held": observed_direction == expected_direction,
                    }
                )

    band_registry = {
        (*row["configuration"], row["offered_load_requests_per_second"]): row
        for row in freeze["held_out_prediction_bands"]
    }
    band_rows = []
    for configuration in HELD_OUT_CONFIGURATIONS:
        for load in OFFERED_LOADS:
            expected = band_registry[(*configuration, load)]
            observed = _fraction(
                by_key[(*configuration, load)]["curve_point"][
                    "per_token_request_delay_ps"
                ]
            )
            lower = _fraction(expected["prediction_band_ps"]["lower"])
            upper = _fraction(expected["prediction_band_ps"]["upper"])
            band_rows.append(
                {
                    "configuration": list(configuration),
                    "offered_load_requests_per_second": load,
                    "predicted_per_token_request_delay_ps": expected[
                        "predicted_per_token_request_delay_ps"
                    ],
                    "band_ps": expected["prediction_band_ps"],
                    "observed_per_token_request_delay_ps": fraction_json(observed),
                    "held": lower <= observed <= upper,
                }
            )

    observed_knees = []
    for configuration in (
        (prefill, decode, prompt)
        for prefill, decode in POOL_RATIOS
        for prompt in PROMPT_LENGTHS
    ):
        rows = [row for row in segment_rows if row["configuration"] == list(configuration)]
        first_increase = next(
            (row for row in rows if row["observed_direction"] == "increase"),
            None,
        )
        observed_knees.append(
            {
                "configuration": list(configuration),
                "predicted_requests_per_second": freeze[
                    "predicted_knees_requests_per_second"
                ][str(configuration[1])],
                "observed_bracket": (
                    None
                    if first_increase is None
                    else [first_increase["from_load"], first_increase["to_load"]]
                ),
            }
        )

    monotonic_held = all(
        row["observed_direction"] == "increase" for row in segment_rows
    )
    decreases = sum(row["observed_direction"] == "decrease" for row in segment_rows)
    increases = sum(row["observed_direction"] == "increase" for row in segment_rows)
    if monotonic_held:
        mechanism = "scheduler queue wait dominates every observed segment"
    elif increases and decreases:
        mechanism = "batch amortization dominates below a queue-wait knee; scheduler wait dominates above it"
    else:
        mechanism = "batch amortization dominates the frozen range; no queue-wait knee was observed"
    if fatal_findings:
        status = "VOID"
    elif all(row["held"] for row in segment_rows) and all(
        row["held"] for row in band_rows
    ):
        status = "PASS"
    else:
        status = "REFUTED"
    return {
        "status": status,
        "fatal_guards": {
            "status": "HELD" if not fatal_findings else "VIOLATED",
            "findings": fatal_findings,
        },
        "core51_control": control,
        "exact_conservation_rows": exact_rows,
        "conservation": {
            "cells": len(cells),
            "admissions": sum(row["admissions"] for row in exact_rows),
            "handoffs": sum(row["handoffs"] for row in exact_rows),
            "terminals": sum(row["terminals"] for row in exact_rows),
            "terminal_decode_tokens": sum(
                row["terminal_tokens"] for row in exact_rows
            ),
            "maximum_ttft_residual_ps": max(
                (row["maximum_ttft_residual_ps"] for row in exact_rows),
                default=0,
            ),
        },
        "segment_verdicts": segment_rows,
        "direction_summary": {
            "matched": sum(row["held"] for row in segment_rows),
            "evaluated": len(segment_rows),
            "observed_decreases": decreases,
            "observed_increases": increases,
            "observed_flats": len(segment_rows) - decreases - increases,
        },
        "held_out_band_verdicts": band_rows,
        "held_out_band_summary": {
            "held": sum(row["held"] for row in band_rows),
            "evaluated": len(band_rows),
        },
        "knees": observed_knees,
        "monotonic_delay_claim": "VALIDATED" if monotonic_held else "WITHDRAWN",
        "measured_mechanism": mechanism,
        "exposure_ruling": {
            "status": freeze["exposure"]["status"],
            "vllm39_clean_close_permitted": False,
            "clean_repetition_residual": "VLLM-40",
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


def main() -> int:
    args = parse_args()
    provenance = check_registry(args)
    if args.check_only:
        print(
            "check-only validated the immutable surface, 36-cell sweep, 30 "
            "segment signs, 24 held-out bands and all preservation locks; no "
            "artifacts produced"
        )
        return 0
    _require_clean_worktree()
    _validate_run_dir(args.run_dir)
    args.run_dir.mkdir(parents=True, exist_ok=False)
    observation = run_observation(args.run_dir)
    analysis = analyze_observation(observation, _load_json(EXPECTATIONS_PATH))
    result = {
        "schema": RESULT_SCHEMA,
        "provenance": provenance,
        "observation": observation,
        "analysis": analysis,
    }
    _write_json(args.run_dir / "result.json", result)
    print(json.dumps(analysis, indent=2, sort_keys=True))
    if analysis["status"] == "VOID":
        raise SystemExit("study is VOID because a fatal guard failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
