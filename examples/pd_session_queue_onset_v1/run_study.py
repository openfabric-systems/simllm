"""Run and score the frozen VLLM-41 lower-load concurrent session."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
from fractions import Fraction
from itertools import pairwise
from pathlib import Path, PurePath
from typing import Any

from simllm.calibration.batch_service_surface import (
    BatchServicePoint,
    interpolate_batch_service_ps,
)

STUDY_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = STUDY_DIR.parents[1]


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_QUEUE_MODEL = _module(STUDY_DIR / "queue_model.py", "vllm41_runtime_queue_model")
OFFERED_LOADS = _QUEUE_MODEL.OFFERED_LOADS
OUTPUT_TOKENS = _QUEUE_MODEL.OUTPUT_TOKENS
POOL_RATIOS = _QUEUE_MODEL.POOL_RATIOS
PROMPT_LENGTHS = _QUEUE_MODEL.PROMPT_LENGTHS
PS_PER_SECOND = _QUEUE_MODEL.PS_PER_SECOND
REQUESTS_PER_CELL = _QUEUE_MODEL.REQUESTS_PER_CELL
fraction_from_json = _QUEUE_MODEL.fraction_from_json
fraction_json = _QUEUE_MODEL.fraction_json

EXPECTATIONS_PATH = STUDY_DIR / "expectations.json"
ACCESS_LEDGER_PATH = STUDY_DIR / "access_ledger.jsonl"
QUEUE_MODEL_PATH = STUDY_DIR / "queue_model.py"
FREEZE_BUILDER_PATH = STUDY_DIR / "freeze_expectations.py"
REFERENCE_STUDY_DIR = REPOSITORY_ROOT / "examples" / "pd_session_load_delay_v1"
SURFACE_PATH = REFERENCE_STUDY_DIR / "surface.json"
BASE_RUNNER_PATH = REFERENCE_STUDY_DIR / "run_study.py"
BASE_QUEUE_MODEL_PATH = REFERENCE_STUDY_DIR / "queue_model.py"

FREEZE_COMMIT = "b3e225e6a4b97280c86536bef136e9945cc239fb"
EXPECTATIONS_SHA256 = (
    "859efc475534bd461761a0e34a039594bd52877520a232a91f2f9c4309c73308"
)
ACCESS_LEDGER_SHA256 = (
    "8a61a6e0b58a213259a19a593b8b3f4ec08cea6a7c854f5481ccbd7bc2dc5914"
)
SURFACE_SHA256 = "26fc547d8b47ccec7108872e05fbedfe71ebb6229b88799ca254089d3f2b6e9d"
QUEUE_MODEL_SHA256 = "d3b63d1a50e3615c7c65d0396a6dc038bbdcab569d43c5e8620babb9fbbce1e3"
RESULT_SCHEMA = "simllm-pd-session-queue-onset-result-v1"
RUN_ROOT_ENV = "SIMLLM_VLLM41_RUN_ROOT"
EXPECTED_VLLM_VERSION = "0.27.1"


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
        raise SystemExit("the VLLM-41 run requires a clean tracked worktree")


def _require_freeze_ancestor() -> None:
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", FREEZE_COMMIT, "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(f"freeze commit {FREEZE_COMMIT} is not an ancestor")


def _validate_freeze(freeze: dict[str, Any]) -> None:
    if freeze["status"] != "EXPECTATIONS_ONLY" or freeze["task"] != "VLLM-41":
        raise SystemExit("VLLM-41 expectations status drifted")
    for path, expected in (
        (EXPECTATIONS_PATH, EXPECTATIONS_SHA256),
        (ACCESS_LEDGER_PATH, ACCESS_LEDGER_SHA256),
        (SURFACE_PATH, SURFACE_SHA256),
        (QUEUE_MODEL_PATH, QUEUE_MODEL_SHA256),
    ):
        if _sha256(path) != expected:
            raise SystemExit(f"frozen path drifted: {path.relative_to(REPOSITORY_ROOT)}")
    sweep = freeze["sweep"]
    if tuple(sweep["offered_load_requests_per_second"]) != OFFERED_LOADS:
        raise SystemExit("lower offered-load ladder drifted")
    if tuple(map(tuple, sweep["pool_ratios"])) != POOL_RATIOS:
        raise SystemExit("pool-ratio ladder drifted")
    if tuple(sweep["prompt_tokens"]) != PROMPT_LENGTHS:
        raise SystemExit("prompt ladder drifted")
    if sweep["requests_per_cell"] != REQUESTS_PER_CELL:
        raise SystemExit("request count drifted")
    if sweep["decode_output_tokens_per_request"] != OUTPUT_TOKENS:
        raise SystemExit("output-token count drifted")
    if len(freeze["expected_segments"]) != 72:
        raise SystemExit("expected segment registry drifted")
    if len(freeze["held_out"]["prediction_bands"]) != 30:
        raise SystemExit("held-out band registry drifted")
    if freeze["chronology"]["lower_ladder_run_existed_before_freeze"] is not False:
        raise SystemExit("expectations no longer precede the run")
    if freeze["queue_model"]["observed_curve_inputs"]:
        raise SystemExit("observed curve entered the frozen queue model")


def check_registry(args: argparse.Namespace) -> dict[str, Any]:
    """Validate the complete freeze without importing or constructing vLLM."""

    if args.run_dir.exists():
        raise SystemExit(f"selected run directory already exists: {args.run_dir}")
    _require_freeze_ancestor()
    freeze = _load_json(EXPECTATIONS_PATH)
    _validate_freeze(freeze)
    freeze_builder = _module(FREEZE_BUILDER_PATH, "vllm41_frozen_builder")
    observed_locks = freeze_builder.preservation_locks()
    if observed_locks != freeze["preservation_locks"]:
        raise SystemExit("preservation locks changed before the VLLM-41 run")
    return {
        "freeze_commit": FREEZE_COMMIT,
        "expectations_sha256": EXPECTATIONS_SHA256,
        "access_ledger_sha256": ACCESS_LEDGER_SHA256,
        "surface_sha256": SURFACE_SHA256,
        "queue_model_sha256": QUEUE_MODEL_SHA256,
        "run_head": _git_head(),
        "candidate_acceptance_status": freeze["surface"]["acceptance_status"],
        "calibration_claim": False,
        "preservation_locks": observed_locks,
        "total_delay_direction_scored": False,
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


def _base_runner_module():
    """Load the historical helpers with their sibling frozen queue model."""

    previous = sys.modules.get("queue_model")
    base_queue_model = _module(
        BASE_QUEUE_MODEL_PATH,
        "vllm41_historical_queue_model",
    )
    sys.modules["queue_model"] = base_queue_model
    try:
        return _module(BASE_RUNNER_PATH, "vllm41_base_session_helpers")
    finally:
        if previous is None:
            del sys.modules["queue_model"]
        else:
            sys.modules["queue_model"] = previous


def _request_id(
    prefill_engines: int,
    decode_engines: int,
    prompt_tokens: int,
    offered_load: int,
    index: int,
) -> str:
    return (
        f"vllm41-p{prefill_engines}-d{decode_engines}-prompt{prompt_tokens}-"
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
) -> dict[str, Any]:
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
                "compute_pricing": observed.compute_pricing,
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
    return {
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
    }


def _validate_runtime() -> str:
    if sys.version_info[:2] != (3, 10):
        raise SystemExit("VLLM-41 requires the worktree Python 3.10 environment")
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise SystemExit("HF_HUB_OFFLINE=1 is required; model downloads are forbidden")
    version = importlib.metadata.version("vllm")
    if version != EXPECTED_VLLM_VERSION:
        raise SystemExit(f"vLLM {EXPECTED_VLLM_VERSION} required, observed {version}")
    return version


def run_observation(run_dir: Path) -> dict[str, Any]:
    """Run all frozen cells without reducing them to total-delay curves."""

    from simllm.adapters.vllm.pd_session import VllmDisaggregatedSession

    vllm_version = _validate_runtime()
    base = _base_runner_module()
    prompt = base._prompt_tokens()
    surface = _load_json(SURFACE_PATH)
    points = _surface_points(surface)
    cells = []
    for prefill_engines, decode_engines in POOL_RATIOS:
        provider = base._surface_provider(surface)
        ratio_dir = run_dir / f"p{prefill_engines}-d{decode_engines}"
        with VllmDisaggregatedSession(
            base._session_config(
                ratio_dir,
                prefill_engines=prefill_engines,
                decode_engines=decode_engines,
                decode_provider=provider,
            )
        ) as session:
            for prompt_tokens in PROMPT_LENGTHS:
                for offered_load in OFFERED_LOADS:
                    cells.append(
                        _cell_observation(
                            session,
                            prompt=prompt,
                            prompt_tokens=prompt_tokens,
                            offered_load=offered_load,
                            prefill_engines=prefill_engines,
                            decode_engines=decode_engines,
                            points=points,
                        )
                    )
    return {
        "runtime": {
            "python": sys.version.split()[0],
            "vllm": vllm_version,
            "offline": True,
        },
        "cells": cells,
        "total_delay_curves": None,
        "total_delay_direction_scored": False,
    }


def _cell_key(cell: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        cell["prefill_engines"],
        cell["decode_engines"],
        cell["prompt_tokens"],
        cell["offered_load_requests_per_second"],
    )


def _decomposition_row(cell: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    findings = []
    requests = cell["requests"]
    residuals = [
        row["timeline"]["decomposition"]["total_ps"] - row["timeline"]["ttft_ps"]
        for row in requests
    ]
    terminal_tokens = sum(len(row["decode_token_ids"]) for row in requests)
    pricing_rows = [row["compute_pricing"] for row in requests]
    pricing_held = all(
        pricing is not None
        and pricing["prefill"] is None
        and pricing["decode"]["record_sha256"]
        == "ff46f6d8a79ddae899da89d4db6eb34373f8042acd06cab50b6336c8fb9a8f52"
        and pricing["decode"]["acceptance_status"] == "candidate"
        and pricing["decode"]["calibration_claim"] is False
        for pricing in pricing_rows
    )
    held = (
        len(requests) == REQUESTS_PER_CELL
        and len({row["request_id"] for row in requests}) == REQUESTS_PER_CELL
        and all(
            row["timeline"]["admitted_at_ps"] == row["expected_admitted_at_ps"]
            for row in requests
        )
        and terminal_tokens == REQUESTS_PER_CELL * OUTPUT_TOKENS
        and all(residual == 0 for residual in residuals)
        and pricing_held
    )
    if not held:
        findings.append("cell-conservation-or-pricing")
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
    return (
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
            "maximum_prefill_batch_size": cell["maximum_prefill_batch_size"],
            "maximum_decode_batch_size": cell["maximum_decode_batch_size"],
        },
        findings,
    )


def _segment_rows(decompositions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {tuple(row["cell"]): row for row in decompositions}
    rows = []
    for prefill, decode in POOL_RATIOS:
        for prompt in PROMPT_LENGTHS:
            for left_load, right_load in pairwise(OFFERED_LOADS):
                left = by_key[(prefill, decode, prompt, left_load)]
                right = by_key[(prefill, decode, prompt, right_load)]
                wait_delta = fraction_from_json(
                    right["mean_scheduler_queue_wait_ps"]
                ) - fraction_from_json(left["mean_scheduler_queue_wait_ps"])
                service_delta = fraction_from_json(
                    right["amortized_decode_batch_service_per_token_ps"]
                ) - fraction_from_json(
                    left["amortized_decode_batch_service_per_token_ps"]
                )
                wait_delta_per_token = wait_delta / OUTPUT_TOKENS
                component_total = wait_delta_per_token + service_delta
                rows.append(
                    {
                        "configuration": [prefill, decode, prompt],
                        "from_load": left_load,
                        "to_load": right_load,
                        "observed_scheduler_wait_delta_ps": fraction_json(wait_delta),
                        "observed_scheduler_wait_delta_per_token_ps": fraction_json(
                            wait_delta_per_token
                        ),
                        "observed_batch_service_per_token_delta_ps": fraction_json(
                            service_delta
                        ),
                        "observed_component_total_delta_per_token_ps": fraction_json(
                            component_total
                        ),
                        "queue_dominated": wait_delta > 0 and component_total > 0,
                    }
                )
    return rows


def _first_onsets(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "configuration": [prefill, decode, prompt],
            "observed_first_queue_dominated_segment": (
                None
                if first is None
                else [first["from_load"], first["to_load"]]
            ),
            "preceding_non_queue_dominated_segments": (
                0
                if first is None
                else sum(
                    not row["queue_dominated"]
                    for row in configuration_rows[: configuration_rows.index(first)]
                )
            ),
        }
        for prefill, decode in POOL_RATIOS
        for prompt in PROMPT_LENGTHS
        for configuration_rows in (
            [
                row
                for row in segments
                if row["configuration"] == [prefill, decode, prompt]
            ],
        )
        for first in (next((row for row in configuration_rows if row["queue_dominated"]), None),)
    ]


def _held_out_verdicts(
    freeze: dict[str, Any], decompositions: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    observed = {tuple(row["cell"]): row for row in decompositions}
    rows = []
    for expected in freeze["held_out"]["prediction_bands"]:
        key = (
            *expected["configuration"],
            expected["offered_load_requests_per_second"],
        )
        row = observed[key]
        wait = fraction_from_json(row["mean_scheduler_queue_wait_ps"])
        service = fraction_from_json(
            row["amortized_decode_batch_service_per_token_ps"]
        )
        wait_lower = fraction_from_json(
            expected["scheduler_queue_wait_band_ps"]["lower"]
        )
        wait_upper = fraction_from_json(
            expected["scheduler_queue_wait_band_ps"]["upper"]
        )
        service_lower = fraction_from_json(
            expected["batch_service_per_token_band_ps"]["lower"]
        )
        service_upper = fraction_from_json(
            expected["batch_service_per_token_band_ps"]["upper"]
        )
        wait_held = wait_lower <= wait <= wait_upper
        service_held = service_lower <= service <= service_upper
        rows.append(
            {
                "configuration": expected["configuration"],
                "offered_load_requests_per_second": key[-1],
                "predicted_scheduler_queue_wait_ps": expected[
                    "predicted_mean_scheduler_queue_wait_ps"
                ],
                "scheduler_queue_wait_band_ps": expected[
                    "scheduler_queue_wait_band_ps"
                ],
                "observed_scheduler_queue_wait_ps": fraction_json(wait),
                "queue_wait_held": wait_held,
                "predicted_batch_service_per_token_ps": expected[
                    "predicted_batch_service_per_token_ps"
                ],
                "batch_service_per_token_band_ps": expected[
                    "batch_service_per_token_band_ps"
                ],
                "observed_batch_service_per_token_ps": fraction_json(service),
                "batch_service_held": service_held,
                "joint_held": wait_held and service_held,
            }
        )
    return rows


def analyze_observation(
    observation: dict[str, Any], freeze: dict[str, Any]
) -> dict[str, Any]:
    """Apply only the frozen decomposition, band, onset, and closure rules."""

    fatal_findings = []
    cells = observation["cells"]
    expected_cell_count = len(POOL_RATIOS) * len(PROMPT_LENGTHS) * len(OFFERED_LOADS)
    if len(cells) != expected_cell_count:
        fatal_findings.append({"guard": "complete-cell-registry"})
    if len({_cell_key(cell) for cell in cells}) != expected_cell_count:
        fatal_findings.append({"guard": "unique-cell-registry"})
    decompositions = []
    all_prefill_ids = []
    all_decode_ids = []
    for cell in cells:
        decomposition, findings = _decomposition_row(cell)
        decompositions.append(decomposition)
        all_prefill_ids.extend(
            row["prefill_internal_request_id"] for row in cell["requests"]
        )
        all_decode_ids.extend(
            row["decode_internal_request_id"] for row in cell["requests"]
        )
        fatal_findings.extend(
            {"guard": finding, "cell": decomposition["cell"]}
            for finding in findings
        )
    if len(all_prefill_ids) != len(set(all_prefill_ids)):
        fatal_findings.append({"guard": "prefill-local-identity-reuse"})
    if len(all_decode_ids) != len(set(all_decode_ids)):
        fatal_findings.append({"guard": "decode-local-identity-reuse"})

    segments = _segment_rows(decompositions)
    expected_segments = {
        (
            *row["configuration"],
            row["from_load"],
            row["to_load"],
        ): row["queue_dominated"]
        for row in freeze["expected_segments"]
    }
    for row in segments:
        key = (
            *row["configuration"],
            row["from_load"],
            row["to_load"],
        )
        row["predicted_queue_dominated"] = expected_segments[key]
        row["prediction_held"] = row["queue_dominated"] == expected_segments[key]

    onsets = _first_onsets(segments)
    admitted_segments = {
        tuple(segment)
        for segment in freeze["queue_model"][
            "first_queue_dominated_segment_prediction"
        ]["inclusive_admitted_segments"]
    }
    for row in onsets:
        observed = row["observed_first_queue_dominated_segment"]
        row["inside_predicted_segment_band"] = (
            observed is not None and tuple(observed) in admitted_segments
        )
    onset_segments = {
        tuple(row["observed_first_queue_dominated_segment"])
        for row in onsets
        if row["observed_first_queue_dominated_segment"] is not None
    }
    vllm41_close = all(
        row["observed_first_queue_dominated_segment"] is not None
        and row["observed_first_queue_dominated_segment"][1] < 250
        and row["preceding_non_queue_dominated_segments"] > 0
        for row in onsets
    )
    held_out = _held_out_verdicts(freeze, decompositions)
    vllm42_required = any(not row["joint_held"] for row in held_out)
    vllm43_required = len(onset_segments) != 1 or not vllm41_close
    if fatal_findings:
        status = "VOID"
    elif vllm41_close:
        status = "IDENTIFIED"
    else:
        status = "UNRESOLVED"
    return {
        "status": status,
        "fatal_guards": {
            "status": "HELD" if not fatal_findings else "VIOLATED",
            "findings": fatal_findings,
        },
        "conservation": {
            "cells": len(cells),
            "admissions": sum(row["admissions"] for row in decompositions),
            "handoffs": sum(row["handoffs"] for row in decompositions),
            "terminals": sum(row["terminals"] for row in decompositions),
            "terminal_decode_tokens": sum(
                row["terminal_tokens"] for row in decompositions
            ),
            "maximum_ttft_residual_ps": max(
                (row["maximum_ttft_residual_ps"] for row in decompositions),
                default=0,
            ),
        },
        "decomposition_rows": decompositions,
        "segment_decompositions": segments,
        "observed_onsets": onsets,
        "onset_summary": {
            "predicted_central_segment": freeze["queue_model"][
                "first_queue_dominated_segment_prediction"
            ]["central"],
            "predicted_inclusive_segments": [
                list(segment) for segment in sorted(admitted_segments)
            ],
            "distinct_observed_segments": [
                list(segment) for segment in sorted(onset_segments)
            ],
            "configurations_resolved": sum(
                row["observed_first_queue_dominated_segment"] is not None
                for row in onsets
            ),
            "configurations_inside_prediction_band": sum(
                row["inside_predicted_segment_band"] for row in onsets
            ),
        },
        "held_out_band_verdicts": held_out,
        "held_out_band_summary": {
            "evaluated": len(held_out),
            "queue_wait_held": sum(row["queue_wait_held"] for row in held_out),
            "batch_service_held": sum(row["batch_service_held"] for row in held_out),
            "joint_held": sum(row["joint_held"] for row in held_out),
        },
        "total_delay_direction_scored": False,
        "prior_250_to_8000_monotonic_direction": "PRESERVED_NOT_REOPENED",
        "closure": {
            "VLLM-41": "CLOSED" if vllm41_close else "OPEN",
            "VLLM-42": "REGISTER_RESIDUAL" if vllm42_required else "UNUSED_RESERVED",
            "VLLM-43": "REGISTER_RESIDUAL" if vllm43_required else "UNUSED_RESERVED",
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
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    provenance = check_registry(args)
    if args.check_only:
        print(
            "check-only validated the committed VLLM-41 freeze, 78-cell lower "
            "ladder, 72 segment predictions, 30 held-out bands and every "
            "preservation lock; no session was constructed"
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
    print(
        json.dumps(
            {
                "status": analysis["status"],
                "conservation": analysis["conservation"],
                "onset_summary": analysis["onset_summary"],
                "held_out_band_summary": analysis["held_out_band_summary"],
                "closure": analysis["closure"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if analysis["status"] == "VOID":
        raise SystemExit("VLLM-41 is VOID because a fatal guard failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
