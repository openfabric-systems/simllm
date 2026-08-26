#!/usr/bin/env python3
"""Run the frozen third CORE-54 scored DeepSeek-V3 flagship study."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import subprocess
import sys
from fractions import Fraction
from pathlib import Path, PurePath
from typing import Any

from curve_tools import as_fraction, fraction_json
from flagship_run3_tools import (
    RUN2_CONFIG_SHA256,
    access_summary,
    build_publication_curve,
    fit_constants,
    load_access_log,
    load_json,
    read_anchor_subset,
    score_frozen_fit,
    sha256,
    validate_execution_config,
    validate_expectations,
    verify_preservation_lock,
    write_json,
)
from run_flagship import _packet_observation, _topology_summary
from run_flagship_run2 import _pre_score_guards, _runtime_observation

STUDY_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = STUDY_DIR.parents[1]
ANCHOR_PATH = STUDY_DIR / "expectations.json"
EXPECTATIONS_PATH = STUDY_DIR / "scored_run3_expectations.json"
DEFAULT_CONFIG_PATH = STUDY_DIR / "flagship_run2_config.json"
EXPECTATIONS_COMMIT = "45251494fa7c9dc0b872bf5324619380cf516a7b"
EXPECTATIONS_SHA256 = "9764f4c910c2ac7410c8ac447936b5ca48964096cf6240521c3f7888754fe637"
RESULT_SCHEMA = "simllm-deployment-curve-flagship-run3-result-v1"
VOID_SCHEMA = "simllm-deployment-curve-flagship-run3-void-v1"
RUN_ROOT_ENV = "SIMLLM_CORE54RUN3_RUN_ROOT"
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
        raise SystemExit("the third scored flagship requires a clean tracked worktree")


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
    config_path: Path,
) -> list[dict[str, str]]:
    if sha256(EXPECTATIONS_PATH) != EXPECTATIONS_SHA256:
        raise SystemExit("third-run expectations digest disagrees")
    if sha256(config_path) != RUN2_CONFIG_SHA256:
        raise SystemExit("the inherited second-run configuration changed")
    candidate_item = config["candidate_record"]
    composition_item = config["composition_record"]
    for item, label in (
        (candidate_item, "candidate record"),
        (composition_item, "composition record"),
    ):
        if sha256(REPOSITORY_ROOT / item["path"]) != item["sha256"]:
            raise SystemExit(f"configured {label} digest disagrees")
    candidate = load_json(REPOSITORY_ROOT / candidate_item["path"])
    if candidate.get("acceptance_status") != candidate_item["acceptance_status"]:
        raise SystemExit("candidate acceptance status disagrees")
    composition = load_json(REPOSITORY_ROOT / composition_item["path"])
    if (
        composition.get("status") != composition_item["status"]
        or composition.get("record_comparison", {}).get("verdict")
        != composition_item["verdict"]
        or composition.get("held_out_numeric_values_accessed") is not False
        or composition.get("fitted_parameters") != []
        or composition.get("free_parameters") != []
    ):
        raise SystemExit("clean COMP-75 composition authority disagrees")
    boundary = frozen["pricing_configuration"]["prefill"]["boundary_authority"]
    boundary_record = REPOSITORY_ROOT / boundary["path"]
    if sha256(boundary_record) != boundary["sha256"]:
        raise SystemExit("clean TRAF-67 boundary digest disagrees")
    if load_json(boundary_record).get("status") != boundary["status"]:
        raise SystemExit("clean TRAF-67 boundary status disagrees")
    try:
        return verify_preservation_lock(REPOSITORY_ROOT, frozen)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def check_registry(
    args: argparse.Namespace,
    *,
    require_runtime: bool,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, str]]]:
    """Validate every frozen input without loading the disclosure anchors."""

    frozen = load_json(EXPECTATIONS_PATH)
    config = load_json(args.config)
    validate_expectations(frozen)
    validate_execution_config(config, frozen)
    for dependency in frozen["dependency_gate"]["must_all_be_ancestors_of_run_head"]:
        _require_ancestor(dependency["merge_commit"])
    _require_ancestor(EXPECTATIONS_COMMIT)
    preservation = _verify_frozen_inputs(frozen, config, args.config)
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
        if not args.sglang_source.is_dir() or _git_head(args.sglang_source) != SGLANG_COMMIT:
            raise SystemExit("the pinned SGLang source directory or commit disagrees")
        if importlib.metadata.version("sglang") != SGLANG_VERSION:
            raise SystemExit("the installed SGLang version disagrees")
        if sys.version_info[:3] != (3, 10, 18):
            raise SystemExit("the third flagship run requires Python 3.10.18")
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
    return frozen, config, preservation


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


def _anchor_predictions(
    frozen: dict[str, Any],
    fit: dict[str, Any],
    score: dict[str, Any],
) -> list[dict[str, Any]]:
    calibration = {row["anchor_id"]: row for row in fit["calibration_rows"]}
    held_out = {row["anchor_id"]: row for row in score["rows"]}
    rows = []
    for frozen_row in frozen["pre_fit_prediction_layers"]:
        anchor_id = frozen_row["anchor_id"]
        if frozen_row.get("status") == "BLOCKED":
            rows.append(
                {
                    "anchor_id": anchor_id,
                    "status": "BLOCKED",
                    "prediction": None,
                    "dependency": frozen_row["dependency"],
                }
            )
        else:
            comparison = calibration.get(anchor_id, held_out.get(anchor_id))
            rows.append(
                {
                    "anchor_id": anchor_id,
                    "status": "PREDICTED",
                    "published": comparison["published"],
                    "layers": comparison["layers"],
                }
            )
    return rows


def _decode_calibration_miss(
    frozen: dict[str, Any],
    fit: dict[str, Any],
) -> dict[str, Any]:
    row = next(
        value
        for value in fit["calibration_rows"]
        if value["anchor_id"] == "sglang_decode_standard"
    )
    layer = row["layers"]["physics_plus_boundary_plus_attenuation"]
    published = as_fraction(row["published"], "decode.published")
    implied_step = Fraction(
        frozen["pricing_configuration"]["decode"]["per_node_tokens"] * 10**12,
        1,
    ) / published
    return {
        "mechanism_count": 0,
        "remote_kv_projection_enabled": True,
        "candidate_key_sha256": frozen["pricing_configuration"]["decode"]
        ["remote_kv_projection"]["exact_candidate_key_sha256"],
        "declared_step_ps": frozen["pricing_configuration"]["decode"]
        ["declared_full_depth_service_ps"],
        "published_throughput_implied_step_ps": fraction_json(implied_step),
        "predicted": layer["prediction"]["point"],
        "published": row["published"],
        "signed_relative_error": layer["signed_relative_error"],
        "absolute_relative_error": layer["absolute_relative_error"],
        "signed_direction": "prediction low",
        "attenuation_applied": False,
        "in_run_adjustment_performed": False,
    }


def run_study(
    frozen: dict[str, Any],
    config: dict[str, Any],
    preservation: list[dict[str, str]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Execute calibration access, fit, guards, one-shot score and curves."""

    args.run_dir.mkdir(parents=True, exist_ok=False)
    access_log_path = args.run_dir / "anchor-access-ledger.jsonl"
    calibration = read_anchor_subset(
        ANCHOR_PATH,
        tuple(frozen["fit_rule"]["visible_anchor_ids"]),
        access_log_path,
        classification="calibration",
    )
    fit = fit_constants(calibration, frozen)
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

    held_out = read_anchor_subset(
        ANCHOR_PATH,
        tuple(frozen["scoring_rule"]["scorable_held_out_anchor_ids"]),
        access_log_path,
        classification="held_out",
    )
    score = score_frozen_fit(held_out, frozen, fit, fit_sha256=fit_sha256)
    score_path = args.run_dir / "held-out-score.json"
    write_json(score_path, score)
    score_sha256 = sha256(score_path)
    access = access_summary(load_access_log(access_log_path))
    predictions = _anchor_predictions(frozen, fit, score)
    decode_prediction = next(
        row["layers"]["physics_plus_boundary_plus_attenuation"]["prediction"]
        for row in predictions
        if row["anchor_id"] == "sglang_decode_standard"
    )
    curves = [
        build_publication_curve(curve, decode_prediction)
        for curve in config["publication_curves"]
    ]
    scored_layer = "physics_plus_boundary_plus_attenuation"
    dominant = max(
        score["rows"],
        key=lambda row: float(
            as_fraction(
                row["layers"][scored_layer]["absolute_relative_error"],
                "dominant.error",
            )
        ),
    )
    passed = score["status"] == "PASS"
    result = {
        "schema": RESULT_SCHEMA,
        "status": score["status"],
        "verdict": f"SCORABLE_HELD_OUT_{score['status']}_MTP_BLOCKED",
        "classification": "THIRD_SCORED_FLAGSHIP",
        "scored_flagship": True,
        "scope": score["scope"],
        "core54_closure": False,
        "closure_reason": (
            "The priced held-out prefill scope passes under the declared "
            "benchmark-bias model, but MTP and decode reproduction remain open."
            if passed
            else "The priced held-out prefill scope misses the 5 percent bar, "
            "and MTP and decode reproduction remain open."
        ),
        "provenance": {
            "run_head": _git_head(),
            "worktree": frozen["dispatch_identity"]["worktree"],
            "branch": frozen["dispatch_identity"]["branch"],
            "base_main_sha": frozen["dispatch_identity"]["base_main_sha"],
            "expectations_commit": EXPECTATIONS_COMMIT,
            "expectations_sha256": sha256(EXPECTATIONS_PATH),
            "anchor_sha256": sha256(ANCHOR_PATH),
            "inherited_configuration_sha256": sha256(args.config),
            "candidate_record_sha256": config["candidate_record"]["sha256"],
            "candidate_acceptance_status": config["candidate_record"][
                "acceptance_status"
            ],
            "composition_record_sha256": config["composition_record"]["sha256"],
            "boundary_record_sha256": frozen["pricing_configuration"]["prefill"]
            ["boundary_authority"]["sha256"],
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
        "attenuation_layer": frozen["attenuation_layer"],
        "constant_fit": fit,
        "constant_fit_sha256": fit_sha256,
        "held_out_score": score,
        "held_out_score_sha256": score_sha256,
        "anchor_access": access,
        "anchor_predictions": predictions,
        "curves": curves,
        "offered_load_sweep_requests_per_second": frozen[
            "offered_load_sweep_requests_per_second"
        ],
        "second_legend": config["second_legend"],
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
        "decode_calibration_miss": _decode_calibration_miss(frozen, fit),
        "dominant_held_out_contributor": {
            "anchor_id": dominant["anchor_id"],
            "finding": (
                "The unattenuated shared communication floor remains flat across "
                "prompt lengths. The independently frozen routing-balance factor "
                "corrects that benchmark-condition bias without changing physics."
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
    frozen, config, preservation = check_registry(
        args,
        require_runtime=not args.check_only,
    )
    if args.check_only:
        print(
            "check-only: third freeze, clean boundary, attenuation arithmetic, "
            "inherited execution, preservation locks and blocked MTP passed; "
            "no anchor or output was read or written"
        )
        return 0
    result = run_study(frozen, config, preservation, args)
    if result["status"] == "VOID":
        print(
            "VOID before held-out scoring; wrote "
            f"{render_cli_path(args.run_dir / 'void-before-score.json')}"
        )
        return 2
    score = result["held_out_score"]
    maximum = score["maximum_attenuated_absolute_relative_error"]
    print(
        f"{result['verdict']}: max_attenuated_error="
        f"{maximum['numerator']}/{maximum['denominator']}; wrote "
        f"{render_cli_path(args.run_dir / 'result.json')}"
    )
    return 0 if score["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
