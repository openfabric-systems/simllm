"""Run and score one frozen VLLM-42 successor split."""

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
from pathlib import Path, PurePath
from typing import Any

from service_model import (
    OFFERED_LOADS,
    OUTPUT_TOKENS,
    POOL_RATIOS,
    PROMPT_LENGTHS,
    PS_PER_SECOND,
    REQUESTS_PER_CELL,
    fraction_from_json,
    fraction_json,
    is_held_out,
)

from simllm.calibration.batch_service_surface import (
    BatchServicePoint,
    interpolate_batch_service_ps,
)

STUDY_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = STUDY_DIR.parents[1]
EXPECTATIONS_PATH = STUDY_DIR / "expectations.json"
SERVICE_MODEL_PATH = STUDY_DIR / "service_model.py"
FIELD_READER_PATH = STUDY_DIR / "field_reader.py"
ACCESS_PROTOCOL_PATH = STUDY_DIR / "access_protocol.json"
FORBIDDEN_ACCESS_LEDGER_PATH = STUDY_DIR / "forbidden_access_ledger.json"
NON_HELD_OUT_PUBLICATION_PATH = STUDY_DIR / "non_held_out_results.json"
SURFACE_PATH = (
    REPOSITORY_ROOT / "examples" / "pd_session_load_delay_v1" / "surface.json"
)
VLLM41_RUNNER_PATH = (
    REPOSITORY_ROOT / "examples" / "pd_session_queue_onset_v1" / "run_study.py"
)

FREEZE_COMMIT = "375639c147f39fe4f01ea212855ef9e8efb5d7fa"
EXPECTATIONS_SHA256 = "95a5921d2075136073189ead7ca7fdc9eca4c8fcb6482cffda7e04eee35989da"
SERVICE_MODEL_SHA256 = "10ae0207581302f1e685c2374ce8e2348f4fd589e08368178b90ad51d56b3a3f"
FIELD_READER_SHA256 = "73d9d5891efbbbe2d0e521da6343be9f9cac9cdf0ce65ce4975559b761b360f6"
ACCESS_PROTOCOL_SHA256 = "0706303d2d69502a06a362dec0f9bf83ff2306b02dfad8a0e842b5167aab769a"
FORBIDDEN_LEDGER_SHA256 = "37517e5f3dc66819f61f5a7bb8ace1921282415f10551d2defa5c3eb0985b570"
SURFACE_SHA256 = "26fc547d8b47ccec7108872e05fbedfe71ebb6229b88799ca254089d3f2b6e9d"
VLLM41_RUNNER_SHA256 = "71b5c9ef4017f6b9943223978d06f21304a56b78082e5180691e6e21a4e3782a"
EXPECTED_VLLM_VERSION = "0.27.1"
RESULT_SCHEMA = "simllm-pd-session-batching-service-split-result-v1"
RUN_ROOT_ENV = "SIMLLM_VLLM42_RUN_ROOT"
SPLITS = ("non-held-out", "held-out")


def render_cli_path(path: PurePath) -> str:
    """Render executed paths with POSIX separators on every host."""

    return path.as_posix()


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=REPOSITORY_ROOT,
        check=check,
        capture_output=True,
        text=True,
    )


def _git_head() -> str:
    return _git("rev-parse", "HEAD").stdout.strip()


def _require_clean_worktree() -> None:
    if _git("status", "--porcelain", "--untracked-files=all").stdout:
        raise SystemExit("the VLLM-42 run requires a clean tracked worktree")


def _require_freeze_ancestor() -> None:
    if _git("merge-base", "--is-ancestor", FREEZE_COMMIT, "HEAD", check=False).returncode:
        raise SystemExit(f"freeze commit {FREEZE_COMMIT} is not an ancestor")


def _current_preservation_rows() -> list[dict[str, Any]]:
    rows = []
    for line in _git("ls-files", "-s", "examples/pd_session*").stdout.splitlines():
        metadata, path = line.split("\t", 1)
        _, indexed_blob, stage = metadata.split()
        if path.startswith("examples/pd_session_batching_service_v1/"):
            continue
        worktree_blob = _git("hash-object", "--", path).stdout.strip()
        rows.append(
            {
                "path": path,
                "blob_sha1": indexed_blob,
                "stage": int(stage),
                "worktree_blob_sha1": worktree_blob,
            }
        )
    return rows


def _validate_preservation(freeze: dict[str, Any]) -> dict[str, Any]:
    expected = freeze["preservation"]["rows"]
    current = _current_preservation_rows()
    projection = [
        {key: row[key] for key in ("path", "blob_sha1", "stage")}
        for row in current
    ]
    if projection != expected:
        raise SystemExit("the tracked earlier pd_session registry changed after freeze")
    drifted = [
        row["path"]
        for row in current
        if row["blob_sha1"] != row["worktree_blob_sha1"]
    ]
    if drifted:
        raise SystemExit(f"earlier pd_session worktree bytes drifted: {drifted}")
    return {
        "artifact_count": len(current),
        "queue_onset_artifact_count": sum(
            row["path"].startswith("examples/pd_session_queue_onset_v1/")
            for row in current
        ),
        "manifest_sha256": freeze["preservation"]["manifest_sha256"],
        "worktree_bytes_held": True,
    }


def _validate_freeze(freeze: dict[str, Any]) -> None:
    if freeze["status"] != "EXPECTATIONS_ONLY" or freeze["task"] != "VLLM-42":
        raise SystemExit("VLLM-42 expectations status drifted")
    for path, expected in (
        (EXPECTATIONS_PATH, EXPECTATIONS_SHA256),
        (SERVICE_MODEL_PATH, SERVICE_MODEL_SHA256),
        (FIELD_READER_PATH, FIELD_READER_SHA256),
        (ACCESS_PROTOCOL_PATH, ACCESS_PROTOCOL_SHA256),
        (FORBIDDEN_ACCESS_LEDGER_PATH, FORBIDDEN_LEDGER_SHA256),
        (SURFACE_PATH, SURFACE_SHA256),
        (VLLM41_RUNNER_PATH, VLLM41_RUNNER_SHA256),
    ):
        if _sha256(path) != expected:
            relative = path.relative_to(REPOSITORY_ROOT).as_posix()
            raise SystemExit(f"frozen path drifted: {relative}")
    if freeze["predictor"]["observed_curve_inputs"]:
        raise SystemExit("an observed curve entered the frozen predictor")
    if freeze["predictor"]["fit_parameters"]:
        raise SystemExit("a fitted parameter entered the frozen predictor")
    if len(freeze["prediction_bands"]) != 78:
        raise SystemExit("prediction registry drifted")
    if freeze["holdout"]["non_held_out_cell_count"] != 48:
        raise SystemExit("non-held-out split drifted")
    if freeze["holdout"]["held_out_cell_count"] != 30:
        raise SystemExit("held-out split drifted")
    if freeze["source_access"]["forbidden_access_ledger"]:
        raise SystemExit("forbidden-access ledger is not empty")


def _require_non_held_out_publication() -> dict[str, str]:
    relative = NON_HELD_OUT_PUBLICATION_PATH.relative_to(REPOSITORY_ROOT).as_posix()
    if not NON_HELD_OUT_PUBLICATION_PATH.exists():
        raise SystemExit("held-out execution requires a committed non-held-out publication")
    if _git("ls-files", "--error-unmatch", relative, check=False).returncode:
        raise SystemExit("non-held-out publication is not tracked")
    if _git("status", "--porcelain", "--", relative).stdout:
        raise SystemExit("non-held-out publication has uncommitted changes")
    commit = _git("log", "-1", "--format=%H", "--", relative).stdout.strip()
    if not commit or _git(
        "merge-base", "--is-ancestor", commit, "HEAD", check=False
    ).returncode:
        raise SystemExit("non-held-out publication commit is not an ancestor")
    publication = _load_json(NON_HELD_OUT_PUBLICATION_PATH)
    if publication["split"] != "non-held-out" or publication["status"] == "VOID":
        raise SystemExit("non-held-out publication cannot release the holdout")
    if publication["provenance"]["freeze_commit"] != FREEZE_COMMIT:
        raise SystemExit("non-held-out publication uses another freeze")
    return {"commit": commit, "sha256": _sha256(NON_HELD_OUT_PUBLICATION_PATH)}


def check_registry(split: str, run_dir: Path) -> dict[str, Any]:
    """Validate the frozen registry without constructing a vLLM engine."""

    if split not in SPLITS:
        raise SystemExit(f"unknown split {split!r}")
    if run_dir.exists():
        raise SystemExit(f"selected run directory already exists: {run_dir}")
    _require_freeze_ancestor()
    freeze = _load_json(EXPECTATIONS_PATH)
    _validate_freeze(freeze)
    preservation = _validate_preservation(freeze)
    non_held_out_publication = (
        _require_non_held_out_publication() if split == "held-out" else None
    )
    return {
        "freeze_commit": FREEZE_COMMIT,
        "expectations_sha256": EXPECTATIONS_SHA256,
        "service_model_sha256": SERVICE_MODEL_SHA256,
        "surface_sha256": SURFACE_SHA256,
        "field_reader_sha256": FIELD_READER_SHA256,
        "access_protocol_sha256": ACCESS_PROTOCOL_SHA256,
        "forbidden_access_ledger_sha256": FORBIDDEN_LEDGER_SHA256,
        "run_head": _git_head(),
        "split": split,
        "preservation": preservation,
        "non_held_out_publication": non_held_out_publication,
        "onset_rescored": False,
        "monotonic_direction_rescored": False,
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


def _vllm41_base_module():
    runner = _module(VLLM41_RUNNER_PATH, "vllm42_vllm41_harness_lineage")
    return runner._base_runner_module()


def _request_id(
    split: str,
    prefill_engines: int,
    decode_engines: int,
    prompt_tokens: int,
    offered_load: int,
    index: int,
) -> str:
    split_label = split.replace("-", "")
    return (
        f"vllm42-{split_label}-p{prefill_engines}-d{decode_engines}-"
        f"prompt{prompt_tokens}-load{offered_load}-request{index}"
    )


def _cell_observation(
    session: Any,
    *,
    split: str,
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
                split,
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
    decode_visits = sum(map(len, result.decode_batches))
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
        "amortized_batching_service_per_token_ps": fraction_json(
            amortized_service
        ),
    }


def _validate_runtime() -> str:
    if sys.version_info[:2] != (3, 10):
        raise SystemExit("VLLM-42 requires the worktree Python 3.10 environment")
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise SystemExit("HF_HUB_OFFLINE=1 is required; model downloads are forbidden")
    version = importlib.metadata.version("vllm")
    if version != EXPECTED_VLLM_VERSION:
        raise SystemExit(f"vLLM {EXPECTED_VLLM_VERSION} required, observed {version}")
    return version


def _cell_selected(
    split: str,
    prefill_engines: int,
    decode_engines: int,
    offered_load: int,
) -> bool:
    held_out = is_held_out(prefill_engines, decode_engines, offered_load)
    return held_out if split == "held-out" else not held_out


def run_observation(split: str, run_dir: Path) -> dict[str, Any]:
    """Run only one frozen disclosure split through the local serving harness."""

    from simllm.adapters.vllm.pd_session import VllmDisaggregatedSession

    version = _validate_runtime()
    base = _vllm41_base_module()
    prompt = base._prompt_tokens()
    surface = _load_json(SURFACE_PATH)
    points = _surface_points(surface)
    cells = []
    for prefill_engines, decode_engines in POOL_RATIOS:
        selected = [
            (prompt_tokens, offered_load)
            for prompt_tokens in PROMPT_LENGTHS
            for offered_load in OFFERED_LOADS
            if _cell_selected(
                split,
                prefill_engines,
                decode_engines,
                offered_load,
            )
        ]
        if not selected:
            continue
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
            for prompt_tokens, offered_load in selected:
                cells.append(
                    _cell_observation(
                        session,
                        split=split,
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
            "vllm": version,
            "offline": True,
            "cluster_time": False,
        },
        "split": split,
        "cells": cells,
        "onset_scored": False,
        "monotonic_direction_scored": False,
    }


def _cell_key(cell: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        cell["prefill_engines"],
        cell["decode_engines"],
        cell["prompt_tokens"],
        cell["offered_load_requests_per_second"],
    )


def _decomposition_row(
    cell: dict[str, Any],
    expected: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    findings = []
    requests = cell["requests"]
    residuals = [
        row["timeline"]["decomposition"]["total_ps"] - row["timeline"]["ttft_ps"]
        for row in requests
    ]
    terminal_tokens = sum(len(row["decode_token_ids"]) for row in requests)
    pricing_held = all(
        (pricing := row["compute_pricing"]) is not None
        and pricing["prefill"] is None
        and pricing["decode"]["record_sha256"]
        == "ff46f6d8a79ddae899da89d4db6eb34373f8042acd06cab50b6336c8fb9a8f52"
        and pricing["decode"]["acceptance_status"] == "candidate"
        and pricing["decode"]["calibration_claim"] is False
        for row in requests
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
    arrival_to_prefill = Fraction(
        sum(
            row["timeline"]["decomposition"]["prefill_queue_ps"]
            for row in requests
        ),
        len(requests),
    )
    handoff_to_decode = Fraction(
        sum(
            row["timeline"]["decomposition"]["decode_admission_wait_ps"]
            for row in requests
        ),
        len(requests),
    )
    service = fraction_from_json(
        cell["amortized_batching_service_per_token_ps"]
    )
    lower = fraction_from_json(expected["batch_service_per_token_band_ps"]["lower"])
    upper = fraction_from_json(expected["batch_service_per_token_band_ps"]["upper"])
    return (
        {
            "cell": list(_cell_key(cell)),
            "split": expected["split"],
            "held": lower <= service <= upper,
            "admissions": len(requests),
            "handoffs": len(requests),
            "terminals": len(requests),
            "terminal_decode_tokens": terminal_tokens,
            "maximum_ttft_residual_ps": max(map(abs, residuals), default=0),
            "arrival_to_prefill_wait_ps": fraction_json(arrival_to_prefill),
            "handoff_to_decode_admission_wait_ps": fraction_json(
                handoff_to_decode
            ),
            "predicted_batching_service_per_token_ps": expected[
                "predicted_batch_service_per_token_ps"
            ],
            "batching_service_per_token_band_ps": expected[
                "batch_service_per_token_band_ps"
            ],
            "observed_batching_service_per_token_ps": fraction_json(service),
            "maximum_prefill_batch_size": cell["maximum_prefill_batch_size"],
            "maximum_decode_batch_size": cell["maximum_decode_batch_size"],
        },
        findings,
    )


def analyze_observation(
    observation: dict[str, Any],
    freeze: dict[str, Any],
) -> dict[str, Any]:
    """Apply only frozen service bands and fatal guards to one split."""

    split = observation["split"]
    expected_rows = [
        row for row in freeze["prediction_bands"] if row["split"] == split
    ]
    expected = {
        (*row["configuration"], row["offered_load_requests_per_second"]): row
        for row in expected_rows
    }
    cells = observation["cells"]
    fatal_findings = []
    if len(cells) != len(expected_rows):
        fatal_findings.append({"guard": "complete-cell-registry"})
    if len({_cell_key(cell) for cell in cells}) != len(expected_rows):
        fatal_findings.append({"guard": "unique-cell-registry"})
    rows = []
    all_prefill_ids = []
    all_decode_ids = []
    for cell in cells:
        key = _cell_key(cell)
        if key not in expected:
            fatal_findings.append({"guard": "split-membership", "cell": list(key)})
            continue
        row, findings = _decomposition_row(cell, expected[key])
        rows.append(row)
        all_prefill_ids.extend(
            request["prefill_internal_request_id"] for request in cell["requests"]
        )
        all_decode_ids.extend(
            request["decode_internal_request_id"] for request in cell["requests"]
        )
        fatal_findings.extend(
            {"guard": finding, "cell": list(key)} for finding in findings
        )
    if len(all_prefill_ids) != len(set(all_prefill_ids)):
        fatal_findings.append({"guard": "prefill-local-identity-reuse"})
    if len(all_decode_ids) != len(set(all_decode_ids)):
        fatal_findings.append({"guard": "decode-local-identity-reuse"})
    floor = fraction_from_json(freeze["physical_bounds"]["floor_service_per_token_ps"])
    ceiling = fraction_from_json(
        freeze["physical_bounds"]["ceiling_service_per_token_ps"]
    )
    if any(
        not floor
        <= fraction_from_json(row["observed_batching_service_per_token_ps"])
        <= ceiling
        for row in rows
    ):
        fatal_findings.append({"guard": "physical-service-bounds"})
    if fatal_findings:
        status = "VOID"
    elif all(row["held"] for row in rows):
        status = "PASS"
    else:
        status = "REFUTED"
    return {
        "status": status,
        "split": split,
        "fatal_guards": {
            "status": "HELD" if not fatal_findings else "VIOLATED",
            "findings": fatal_findings,
        },
        "conservation": {
            "cells": len(rows),
            "admissions": sum(row["admissions"] for row in rows),
            "handoffs": sum(row["handoffs"] for row in rows),
            "terminals": sum(row["terminals"] for row in rows),
            "terminal_decode_tokens": sum(
                row["terminal_decode_tokens"] for row in rows
            ),
            "maximum_ttft_residual_ps": max(
                (row["maximum_ttft_residual_ps"] for row in rows),
                default=0,
            ),
        },
        "service_band_verdicts": rows,
        "service_band_summary": {
            "held": sum(row["held"] for row in rows),
            "missed": sum(not row["held"] for row in rows),
            "evaluated": len(rows),
        },
        "arrival_to_prefill_published": True,
        "handoff_to_decode_published": True,
        "batching_service_published": True,
        "onset_claim": "PRESERVED_NOT_RESCORED",
        "monotonic_250_to_8000_claim": "PRESERVED_NOT_RESCORED",
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
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    provenance = check_registry(args.split, args.run_dir)
    if args.check_only:
        expected_cells = 48 if args.split == "non-held-out" else 30
        print(
            f"check-only validated the committed VLLM-42 freeze, {expected_cells} "
            f"{args.split} cells, preservation locks, and disclosure order"
        )
        return 0
    _require_clean_worktree()
    _validate_run_dir(args.run_dir)
    args.run_dir.mkdir(parents=True, exist_ok=False)
    observation = run_observation(args.split, args.run_dir)
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
                "split": args.split,
                "conservation": analysis["conservation"],
                "service_band_summary": analysis["service_band_summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if analysis["status"] == "VOID":
        raise SystemExit("VLLM-42 split is VOID because a fatal guard failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
