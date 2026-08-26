#!/usr/bin/env python3
"""Run the frozen second CORE-54 scored DeepSeek-V3 flagship study."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path, PurePath
from typing import Any

from flagship_run2_tools import (
    build_publication_curve,
    fit_inherited_constant,
    load_json,
    prediction_interval,
    score_frozen_fit,
    sha256,
    stable_request_projection,
    validate_config,
    validate_expectations,
    verify_preservation_lock,
    write_json,
)
from run_flagship import _packet_observation, _topology_summary

STUDY_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = STUDY_DIR.parents[1]
ANCHOR_PATH = STUDY_DIR / "expectations.json"
EXPECTATIONS_PATH = STUDY_DIR / "scored_run2_expectations.json"
DEFAULT_CONFIG_PATH = STUDY_DIR / "flagship_run2_config.json"
RESULT_SCHEMA = "simllm-deployment-curve-flagship-run2-result-v1"
VOID_SCHEMA = "simllm-deployment-curve-flagship-run2-void-v1"
RUN_ROOT_ENV = "SIMLLM_CORE54RUN2_RUN_ROOT"
SGLANG_VERSION = "0.5.19.dev345+gbfeae4e79"
SGLANG_COMMIT = "bfeae4e79a8dc4600e006f1a5fbc85321a01c1a3"
HTSIM_RNIC_SHA256 = "388415f92d6ef54c84bb5d2b7f7dabcaad27574ec235d62260f08175f3958bd9"
TXT2BIN_SHA256 = "f3745f34ad86febe9c9eebef10aee5fae00b8865cb29943344fb75b0f142495b"


def render_cli_path(path: PurePath) -> str:
    """Render command-line paths with POSIX separators on every host."""

    return path.as_posix()


def _git_head(path: Path = REPOSITORY_ROOT) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _require_ancestor(commit: str) -> None:
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=False,
    )
    if completed.returncode:
        raise SystemExit(f"required commit {commit} is not an ancestor of HEAD")


def _require_clean_tracked_worktree() -> None:
    completed = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    if completed.stdout.strip():
        raise SystemExit("the second scored flagship requires a clean tracked worktree")


def _validate_run_dir(run_dir: Path) -> None:
    configured = os.environ.get(RUN_ROOT_ENV)
    if not configured:
        raise SystemExit(f"{RUN_ROOT_ENV} must name the external run root")
    root = Path(configured).resolve()
    try:
        run_dir.resolve().relative_to(root)
    except ValueError as exc:
        raise SystemExit(f"run directory must remain under {RUN_ROOT_ENV}") from exc


def _verify_frozen_inputs(
    frozen: dict[str, Any],
    config: dict[str, Any],
) -> list[dict[str, str]]:
    if sha256(EXPECTATIONS_PATH) != config["study"]["expectations_sha256"]:
        raise SystemExit("second-run expectations digest disagrees")
    for key in ("candidate_record", "composition_record"):
        item = config[key]
        if sha256(REPOSITORY_ROOT / item["path"]) != item["sha256"]:
            raise SystemExit(f"configured {key} digest disagrees")
    candidate = load_json(REPOSITORY_ROOT / config["candidate_record"]["path"])
    if candidate.get("acceptance_status") != config["candidate_record"][
        "acceptance_status"
    ]:
        raise SystemExit("candidate acceptance status disagrees")
    composition = load_json(REPOSITORY_ROOT / config["composition_record"]["path"])
    if (
        composition.get("status") != config["composition_record"]["status"]
        or composition.get("record_comparison", {}).get("verdict")
        != config["composition_record"]["verdict"]
        or composition.get("held_out_numeric_values_accessed") is not False
        or composition.get("fitted_parameters") != []
        or composition.get("free_parameters") != []
    ):
        raise SystemExit("clean COMP-75 composition authority disagrees")
    try:
        return verify_preservation_lock(REPOSITORY_ROOT, frozen)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def check_registry(
    args: argparse.Namespace,
    *,
    require_runtime: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, str]]]:
    """Validate every frozen input before importing SGLang or writing output."""

    anchors = load_json(ANCHOR_PATH)
    frozen = load_json(EXPECTATIONS_PATH)
    config = load_json(args.config)
    validate_expectations(frozen)
    validate_config(config, frozen)
    for dependency in frozen["dependency_gate"]["must_all_be_ancestors_of_run_head"]:
        _require_ancestor(dependency["merge_commit"])
    _require_ancestor(config["study"]["expectations_commit"])
    preservation = _verify_frozen_inputs(frozen, config)
    if require_runtime:
        _require_clean_tracked_worktree()
        _validate_run_dir(args.run_dir)
        for path in (
            args.run_dir,
            args.model_path,
            args.sglang_source,
            args.txt2bin,
            args.htsim_rnic,
        ):
            if not path.is_absolute():
                raise SystemExit("all runtime paths must be explicit absolute paths")
        if args.run_dir.exists():
            raise SystemExit(f"selected run directory already exists: {args.run_dir}")
        if not args.model_path.is_dir():
            raise SystemExit("the local DeepSeek configuration directory is missing")
        if sha256(args.model_path / "config.json") != config["model"]["config_sha256"]:
            raise SystemExit("the local DeepSeek configuration digest disagrees")
        if not args.sglang_source.is_dir():
            raise SystemExit("the pinned SGLang source directory is missing")
        if _git_head(args.sglang_source) != SGLANG_COMMIT:
            raise SystemExit("the pinned SGLang source commit disagrees")
        if importlib.metadata.version("sglang") != SGLANG_VERSION:
            raise SystemExit("the installed SGLang version disagrees")
        if sys.version_info[:3] != (3, 10, 18):
            raise SystemExit("the second flagship run requires Python 3.10.18")
        if sha256(args.txt2bin) != TXT2BIN_SHA256:
            raise SystemExit("the accepted txt2bin digest disagrees")
        if sha256(args.htsim_rnic) != HTSIM_RNIC_SHA256:
            raise SystemExit("the accepted htsim_rnic digest disagrees")
        required_environment = {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "SIMLLM_SGLANG_ENABLE": "1",
        }
        for name, expected in required_environment.items():
            if os.environ.get(name) != expected:
                raise SystemExit(f"{name}={expected} is required")
        pythonpath = os.environ.get("PYTHONPATH", "").split(os.pathsep)
        if not pythonpath or Path(pythonpath[0]).resolve() != REPOSITORY_ROOT:
            raise SystemExit("PYTHONPATH must begin with the selected worktree")
    return anchors, frozen, config, preservation


def _deepseek_dims(config: dict[str, Any]) -> Any:
    from simllm.compute import ModelDims

    return ModelDims(**config["model"]["dims"])


def _arrangement(size: int) -> Any:
    from simllm.placement import SglangPoolArrangement

    return SglangPoolArrangement(
        enable_data_parallel_attention=True,
        attention_data_parallel_size=size,
        dense_data_parallel_size=size,
        expert_parallel_size=size,
    )


def _session_config(
    config: dict[str, Any],
    model_path: Path,
    workdir: Path,
    *,
    candidate: bytes,
    observation: dict[str, Any],
) -> Any:
    from simllm.adapters.sglang import SglangPdSessionConfig
    from simllm.calibration.kernel_cycle_lut import compile_session_profile_provider
    from simllm.compute import GPU_ENVELOPES, RooflineProvider
    from simllm.core import DeclaredKvHandoffPolicy, KvHandoffGeometry

    live = config["live_session"]
    comparator = RooflineProvider(efficiency=live["roofline_comparator_efficiency"])
    selected = compile_session_profile_provider(
        candidate,
        expected_sha256=config["candidate_record"]["sha256"],
        pool=observation["pool"],
        comparator=comparator,
        selection_entry_index=observation["candidate_entry_index"],
    )
    providers: dict[str, Any] = {
        "provider": comparator,
        "prefill_provider": selected if observation["pool"] == "prefill" else None,
        "decode_provider": selected if observation["pool"] == "decode" else None,
    }
    return SglangPdSessionConfig(
        model_path=model_path,
        workdir=workdir,
        dims=_deepseek_dims(config),
        handoff_geometry=KvHandoffGeometry(**config["model"]["kv_handoff_geometry"]),
        handoff_policy=DeclaredKvHandoffPolicy(live["handoff_constant_ps"]),
        prefill_arrangement=_arrangement(8),
        decode_arrangement=_arrangement(8),
        prefill_engines=live["prefill_engines"],
        decode_engines=live["decode_engines"],
        simulated_gpus_per_engine=live["simulated_gpus_per_engine"],
        context_length=live["context_length"],
        max_total_tokens=live["max_total_tokens"],
        max_running_requests=live["max_running_requests"],
        token_id=live["token_id"],
        random_seed=live["random_seed"],
        gpu=GPU_ENVELOPES[live["gpu"]],
        project_remote_kv_length=live["project_remote_kv_length"],
        **providers,
    )


def _prompt(length: int, ordinal: int) -> tuple[int, ...]:
    return tuple(1_000 + (ordinal + index) % 97 for index in range(length))


def _runtime_observation(
    config: dict[str, Any],
    model_path: Path,
    run_dir: Path,
    candidate: bytes,
    observation: dict[str, Any],
    *,
    suffix: str,
) -> dict[str, Any]:
    from simllm.adapters.sglang import SglangDisaggregatedSession, SglangPdRequest

    anchor_id = observation["anchor_id"]
    requests = tuple(
        SglangPdRequest(
            request_id=f"{anchor_id}-request-{index}",
            prompt_token_ids=_prompt(observation["prompt_tokens"], index),
            decode_output_tokens=observation["decode_output_tokens_per_request"],
            admitted_at_ps=0,
        )
        for index in range(observation["requests"])
    )
    with SglangDisaggregatedSession(
        _session_config(
            config,
            model_path,
            run_dir / f"{anchor_id}-{suffix}",
            candidate=candidate,
            observation=observation,
        )
    ) as session:
        result = session.run_requests(requests)
        prefill_ranks, decode_ranks = session.packet_rank_sets()
        stable = [stable_request_projection(row.to_json()) for row in result.requests]
        pricing = result.requests[-1].compute_pricing
        batches = {
            "prefill": [list(batch) for batch in result.prefill_batches],
            "decode": [list(batch) for batch in result.decode_batches],
        }
    stable_bytes = json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()
    return {
        "anchor_id": anchor_id,
        "pool": observation["pool"],
        "candidate_entry_index": observation["candidate_entry_index"],
        "admissions": len(requests),
        "terminals": len(stable),
        "prompt_tokens_per_request": observation["prompt_tokens"],
        "total_prompt_tokens": observation["prompt_tokens"] * len(requests),
        "remote_kv_projection_enabled": config["live_session"][
            "project_remote_kv_length"
        ],
        "stable_projection_sha256": hashlib.sha256(stable_bytes).hexdigest(),
        "stable_requests": stable,
        "pricing_provenance": pricing,
        "batches": batches,
        "prefill_ranks": list(prefill_ranks),
        "decode_ranks": list(decode_ranks),
    }


def _selection_summary(observation: dict[str, Any]) -> dict[str, Any]:
    pricing = observation["pricing_provenance"]
    pool = observation["pool"]
    selected = None if pricing is None else pricing[pool]
    return {
        "anchor_id": observation["anchor_id"],
        "pool": pool,
        "record_sha256": None if selected is None else selected["record_sha256"],
        "acceptance_status": None if selected is None else selected["acceptance_status"],
        "lookup_hits": None if selected is None else selected["lookup_hits"],
        "lookup_misses": None if selected is None else selected["lookup_misses"],
        "selected_entry_key_sha256s": (
            [] if selected is None else selected["selected_entry_key_sha256s"]
        ),
        "selected": bool(selected and selected["lookup_hits"] > 0),
    }


def _anchor_predictions(
    frozen: dict[str, Any],
    fit: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for prediction in frozen["pre_fit_predicted_bands"]:
        if prediction.get("status") == "BLOCKED":
            rows.append(
                {
                    "anchor_id": prediction["anchor_id"],
                    "status": "BLOCKED",
                    "prediction": None,
                    "dependency": prediction["dependency"],
                }
            )
            continue
        rows.append(
            {
                "anchor_id": prediction["anchor_id"],
                "status": "PREDICTED",
                "prediction": prediction_interval(
                    prediction,
                    frozen,
                    int(fit["fitted_ps"]),
                ),
            }
        )
    return rows


def _pre_score_guards(
    frozen: dict[str, Any],
    observations: list[dict[str, Any]],
    identity_b: dict[str, Any],
    packet: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    selections = [_selection_summary(row) for row in observations]
    violations = []
    if not all(row["selected"] for row in selections):
        violations.append("one or more exact candidate rows did not select")
    expected_decode = frozen["pricing_configuration"]["decode"][
        "remote_kv_projection"
    ]
    decode = next(row for row in selections if row["anchor_id"] == "sglang_decode_standard")
    if (
        decode["lookup_hits"] != expected_decode["expected_lookup_hits"]
        or decode["lookup_misses"] != expected_decode["expected_lookup_misses"]
        or decode["selected_entry_key_sha256s"]
        != [expected_decode["exact_candidate_key_sha256"]]
    ):
        violations.append("the exact SGL-38 decode key did not bind")
    identity_a = observations[-1]
    if identity_a["stable_requests"] != identity_b["stable_requests"]:
        violations.append("the CORE-58 stable identity projection drifted")
    if not packet["byte_conserved"] or not packet["endpoint_conserved"]:
        violations.append("the packet handoff did not conserve bytes or endpoints")
    if not all(row["quiescent"] for row in packet["rows"]):
        violations.append("a packet handoff arm did not reach quiescence")
    return selections, violations


def _write_void_before_score(
    args: argparse.Namespace,
    fit_sha256: str,
    violations: list[str],
    observations: list[dict[str, Any]],
) -> dict[str, Any]:
    value = {
        "schema": VOID_SCHEMA,
        "status": "VOID",
        "held_out_numeric_values_accessed": False,
        "held_out_score_performed": False,
        "fit_sha256": fit_sha256,
        "violations": violations,
        "session_observations": observations,
    }
    write_json(args.run_dir / "void-before-score.json", value)
    return value


def run_study(
    anchors: dict[str, Any],
    frozen: dict[str, Any],
    config: dict[str, Any],
    preservation: list[dict[str, str]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Execute the fit, guards, one-shot score and publication curves."""

    args.run_dir.mkdir(parents=True, exist_ok=False)
    fit = fit_inherited_constant(anchors, frozen)
    fit_path = args.run_dir / "frozen-fit.json"
    write_json(fit_path, fit)
    fit_sha256 = sha256(fit_path)

    candidate_path = REPOSITORY_ROOT / config["candidate_record"]["path"]
    candidate = candidate_path.read_bytes()
    observations = [
        _runtime_observation(
            config,
            args.model_path,
            args.run_dir / "sessions",
            candidate,
            observation,
            suffix="run-a",
        )
        for observation in config["exact_shape_observations"]
    ]
    identity_b = _runtime_observation(
        config,
        args.model_path,
        args.run_dir / "sessions",
        candidate,
        config["exact_shape_observations"][-1],
        suffix="run-b",
    )
    packet = _packet_observation(config, args)
    selections, violations = _pre_score_guards(
        frozen,
        observations,
        identity_b,
        packet,
    )
    if violations:
        return _write_void_before_score(args, fit_sha256, violations, observations)

    score = score_frozen_fit(
        anchors,
        frozen,
        fit,
        fit_sha256=fit_sha256,
    )
    score_path = args.run_dir / "held-out-score.json"
    write_json(score_path, score)
    score_sha256 = sha256(score_path)
    predictions = _anchor_predictions(frozen, fit)
    standard_capacity = next(
        row["prediction"]
        for row in predictions
        if row["anchor_id"] == "sglang_decode_standard"
    )
    curves = [
        build_publication_curve(curve, standard_capacity)
        for curve in config["publication_curves"]
    ]
    decode_calibration = next(
        row
        for row in fit["calibration_rows"]
        if row["anchor_id"] == "sglang_decode_standard"
    )
    dominant = max(
        score["rows"],
        key=lambda row: row["absolute_relative_error"]["numerator"]
        / row["absolute_relative_error"]["denominator"],
    )
    passed = score["status"] == "PASS"
    result = {
        "schema": RESULT_SCHEMA,
        "status": score["status"],
        "verdict": f"SCORABLE_HELD_OUT_{score['status']}_MTP_BLOCKED",
        "classification": "SECOND_SCORED_FLAGSHIP",
        "scored_flagship": True,
        "scope": score["scope"],
        "core54_closure": False,
        "closure_reason": (
            "The priced held-out prefill scope passes, but MTP, decode "
            "reproduction and distribution evidence remain open."
            if passed
            else "The priced held-out prefill scope misses the 5 percent bar, "
            "and MTP, decode reproduction and distribution evidence remain open."
        ),
        "provenance": {
            "run_head": _git_head(),
            "worktree": frozen["dispatch_identity"]["worktree"],
            "branch": frozen["dispatch_identity"]["branch"],
            "base_main_sha": frozen["dispatch_identity"]["base_main_sha"],
            "expectations_commit": config["study"]["expectations_commit"],
            "expectations_sha256": sha256(EXPECTATIONS_PATH),
            "anchor_sha256": sha256(ANCHOR_PATH),
            "configuration_sha256": sha256(args.config),
            "candidate_record_sha256": config["candidate_record"]["sha256"],
            "candidate_acceptance_status": config["candidate_record"][
                "acceptance_status"
            ],
            "composition_record_sha256": config["composition_record"]["sha256"],
            "composition_record_status": config["composition_record"]["status"],
            "composition_record_verdict": config["composition_record"]["verdict"],
            "composition_new_tunables": 0,
            "sglang_version": SGLANG_VERSION,
            "sglang_commit": SGLANG_COMMIT,
            "python_version": ".".join(map(str, sys.version_info[:3])),
            "model_config_sha256": config["model"]["config_sha256"],
            "model_weights_loaded": False,
            "web_pages_fetched": False,
            "htsim_rnic_sha256": HTSIM_RNIC_SHA256,
            "txt2bin_sha256": TXT2BIN_SHA256,
        },
        "allocation": frozen["inherited_rulings"]["allocation"],
        "scale_mapping": frozen["inherited_rulings"]["largest_faithful_live_scale"],
        "topology": _topology_summary(),
        "pricing_configuration": frozen["pricing_configuration"],
        "constant_fit": fit,
        "constant_fit_sha256": fit_sha256,
        "held_out_score": score,
        "held_out_score_sha256": score_sha256,
        "anchor_predictions": predictions,
        "curves": curves,
        "offered_load_sweep_requests_per_second": frozen[
            "offered_load_sweep_requests_per_second"
        ],
        "session_observations": observations,
        "candidate_selections": selections,
        "stable_identity_guard": {
            "field_set": "CORE-58 unchanged",
            "first_sha256": observations[-1]["stable_projection_sha256"],
            "second_sha256": identity_b["stable_projection_sha256"],
            "equal": True,
            "whole_request_bytes_compared": False,
        },
        "packet_observation": packet,
        "decode_calibration_miss": {
            "mechanism_count": 0,
            "remote_kv_projection_enabled": True,
            "candidate_key_sha256": frozen["pricing_configuration"]["decode"]
            ["remote_kv_projection"]["exact_candidate_key_sha256"],
            "declared_step_ps": frozen["pricing_configuration"]["decode"]
            ["declared_full_depth_service_ps"],
            "published_throughput_implied_step_ps": frozen[
                "pricing_configuration"
            ]["decode"]["visible_calibration_implied_step_ps"],
            "predicted": decode_calibration["predicted"],
            "published": decode_calibration["published"],
            "absolute_relative_error": decode_calibration[
                "absolute_relative_error"
            ],
            "signed_direction": "prediction low",
            "in_run_adjustment_performed": False,
        },
        "dominant_held_out_contributor": {
            "anchor_id": dominant["anchor_id"],
            "finding": (
                "The shared communication record dominates every measured "
                "prefill compute row, so max-like composition flattens the 1K, "
                "2K and 4K point capacities. The largest residual is therefore "
                "the longest-prompt held-out row."
            ),
        },
        "preservation_lock": {
            "class": frozen["preservation_lock"]["class"],
            "status": "PASS",
            "artifacts": preservation,
        },
        "residuals_required": [
            "COMP-72 exact MTP and measured DeepSeek cells",
            "COMP-74 repeat-derived distributions",
            "COMP-76 standard-decode reproduction",
            "CORE-61 depth-extrapolation validity",
            "TRAF-66 finite-overlap residual",
            "SGL-36 physical steady-state load surface",
            "TRAF-64 full PLACE-5 path qualification",
        ],
    }
    write_json(args.run_dir / "result.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--sglang-source", type=Path)
    parser.add_argument("--txt2bin", type=Path)
    parser.add_argument("--htsim-rnic", type=Path)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runtime_paths = (
        args.run_dir,
        args.model_path,
        args.sglang_source,
        args.txt2bin,
        args.htsim_rnic,
    )
    if not args.check_only and any(path is None for path in runtime_paths):
        raise SystemExit("all runtime path arguments are required for a scored run")
    anchors, frozen, config, preservation = check_registry(
        args,
        require_runtime=not args.check_only,
    )
    if args.check_only:
        print(
            "check-only: second freeze, clean composition, remote-KV decode, "
            "preservation locks, exact load grids and blocked MTP passed; "
            "no artifact written"
        )
        return 0
    result = run_study(anchors, frozen, config, preservation, args)
    if result["status"] == "VOID":
        print(
            "VOID before held-out scoring; wrote "
            f"{render_cli_path(args.run_dir / 'void-before-score.json')}"
        )
        return 2
    score = result["held_out_score"]
    print(
        f"{result['verdict']}: max_error="
        f"{score['maximum_absolute_relative_error']['numerator']}/"
        f"{score['maximum_absolute_relative_error']['denominator']}; wrote "
        f"{render_cli_path(args.run_dir / 'result.json')}"
    )
    return 0 if score["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
