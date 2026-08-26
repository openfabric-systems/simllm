#!/usr/bin/env python3
"""Run the frozen CORE-54 scored DeepSeek-V3 flagship study."""

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

from flagship_tools import (
    build_publication_curve,
    fit_frozen_constant,
    load_json,
    prediction_interval,
    score_frozen_fit,
    sha256,
    stable_request_projection,
    validate_flagship_config,
    validate_scored_expectations,
    write_json,
)

STUDY_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = STUDY_DIR.parents[1]
ANCHOR_PATH = STUDY_DIR / "expectations.json"
SCORED_EXPECTATIONS_PATH = STUDY_DIR / "scored_expectations.json"
DEFAULT_CONFIG_PATH = STUDY_DIR / "flagship_config.json"
RESULT_SCHEMA = "simllm-deployment-curve-flagship-result-v1"
RUN_ROOT_ENV = "SIMLLM_CORE54_RUN_ROOT"
SGLANG_VERSION = "0.5.19.dev345+gbfeae4e79"
SGLANG_COMMIT = "bfeae4e79a8dc4600e006f1a5fbc85321a01c1a3"
HTSIM_RNIC_SHA256 = "388415f92d6ef54c84bb5d2b7f7dabcaad27574ec235d62260f08175f3958bd9"
TXT2BIN_SHA256 = "f3745f34ad86febe9c9eebef10aee5fae00b8865cb29943344fb75b0f142495b"
PS_PER_SECOND = 1_000_000_000_000


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
        raise SystemExit("the scored flagship requires a clean tracked worktree")


def _validate_run_dir(run_dir: Path) -> None:
    configured = os.environ.get(RUN_ROOT_ENV)
    if not configured:
        raise SystemExit(f"{RUN_ROOT_ENV} must name the external run root")
    root = Path(configured).resolve()
    try:
        run_dir.resolve().relative_to(root)
    except ValueError as exc:
        raise SystemExit(f"run directory must remain under {RUN_ROOT_ENV}") from exc


def _validate_dependency_gate(scored: dict[str, Any]) -> None:
    for dependency in scored["dependency_gate"]["must_all_be_ancestors_of_run_head"]:
        _require_ancestor(dependency["merge_commit"])


def _verify_frozen_inputs(scored: dict[str, Any], config: dict[str, Any]) -> None:
    for item in scored["frozen_inputs"].values():
        path = item.get("path")
        expected = item.get("sha256")
        if path is None or expected is None:
            continue
        actual = sha256(REPOSITORY_ROOT / path)
        if actual != expected:
            raise SystemExit(f"frozen input digest disagrees for {path}: {actual}")
    candidate = config["candidate_record"]
    if sha256(REPOSITORY_ROOT / candidate["path"]) != candidate["sha256"]:
        raise SystemExit("configured candidate record digest disagrees")


def check_registry(
    args: argparse.Namespace,
    *,
    require_runtime: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate all frozen inputs before importing SGLang or writing artifacts."""

    anchors = load_json(ANCHOR_PATH)
    scored = load_json(SCORED_EXPECTATIONS_PATH)
    config = load_json(args.config)
    validate_scored_expectations(scored)
    validate_flagship_config(config)
    _validate_dependency_gate(scored)
    _require_ancestor(config["study"]["expectations_commit"])
    _verify_frozen_inputs(scored, config)
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
        if args.sglang_source.is_dir() is False:
            raise SystemExit("the pinned SGLang source directory is missing")
        if _git_head(args.sglang_source) != SGLANG_COMMIT:
            raise SystemExit("the pinned SGLang source commit disagrees")
        if importlib.metadata.version("sglang") != SGLANG_VERSION:
            raise SystemExit("the installed SGLang version disagrees")
        if sys.version_info[:3] != (3, 10, 18):
            raise SystemExit("the flagship run requires Python 3.10.18")
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
    return anchors, scored, config


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


def _topology_summary() -> dict[str, Any]:
    from simllm.adapters.sglang.pd_session import SGLANG_VERSION as LIVE_VERSION
    from simllm.placement import sglang_disaggregated_manifests

    separate = sglang_disaggregated_manifests(
        prefill_nodes=4,
        decode_nodes=9,
        gpus_per_node=8,
        prefill_arrangement=_arrangement(32),
        decode_arrangement=_arrangement(72),
        framework_version=LIVE_VERSION,
    )
    what_if = sglang_disaggregated_manifests(
        prefill_nodes=16,
        decode_nodes=40,
        gpus_per_node=8,
        prefill_arrangement=_arrangement(128),
        decode_arrangement=_arrangement(320),
        framework_version=LIVE_VERSION,
    )
    ranks = separate.placement.ranks
    return {
        "separate_experiment_authority": {
            "structural_comparator_ranks": len(ranks),
            "may_be_called_96_gpu_system": False,
            "prefill_rank_set": [
                rank.global_rank for rank in ranks if rank.pool_role == "prefill"
            ],
            "decode_rank_set": [
                rank.global_rank for rank in ranks if rank.pool_role == "decode"
            ],
            "simultaneous_disclosure_claim": False,
        },
        "place5_16p40d_what_if": {
            "ranks": len(what_if.placement.ranks),
            "nodes": len(what_if.fabric.nodes),
            "gpus": sum(len(node.gpus) for node in what_if.fabric.nodes),
            "nics": sum(len(node.nics) for node in what_if.fabric.nodes),
            "physical_rendering_enabled": what_if.fabric.physical_rendering_enabled,
            "topology_name": what_if.fabric.topology_name,
            "evidence_class": what_if.fabric.evidence_class,
        },
    }


def _packet_observation(
    config: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    from simllm.core import KvHandoffGeometry
    from simllm.traffic import PacketKvHandoffPolicy

    packet = config["packet_observation"]
    geometry = KvHandoffGeometry(**config["model"]["kv_handoff_geometry"])
    kv_bytes = geometry.bytes_for_prompt(packet["prompt_tokens"])
    rows = []
    for label, rate in (
        ("point", packet["point_link_rate_bps"]),
        ("sensitivity", packet["sensitivity_link_rate_bps"]),
    ):
        policy = PacketKvHandoffPolicy(
            artifact_dir=args.run_dir / "packet" / label,
            linkspeed_bps=rate,
            txt2bin=args.txt2bin,
            htsim_rnic=args.htsim_rnic,
            pcie_submission_ps=packet["pcie_submission_ps"],
            prefill_ranks=tuple(packet["prefill_ranks"]),
            decode_ranks=tuple(packet["decode_ranks"]),
        )
        event = policy.schedule(
            submitted_at_ps=0,
            request_id=f"{packet['request_id']}-{label}",
            kv_bytes=kv_bytes,
        )
        artifact = policy.artifacts[0]
        rows.append(
            {
                "arm": label,
                "link_rate_bps": rate,
                "event": event.to_json(),
                "packet_service_ps": artifact.packet_service_ps,
                "aggregate_kv_bytes": artifact.aggregate_kv_bytes,
                "chunk_bytes": list(artifact.chunk_bytes),
                "message_pairs": [
                    [message.source_rank, message.destination_rank]
                    for message in artifact.messages
                ],
                "quiescent": artifact.quiescent,
            }
        )
    return {
        "topology_authority": packet["topology_authority"],
        "prompt_tokens": packet["prompt_tokens"],
        "kv_bytes": kv_bytes,
        "prefill_ranks": packet["prefill_ranks"],
        "decode_ranks": packet["decode_ranks"],
        "rows": rows,
        "byte_conserved": all(row["aggregate_kv_bytes"] == kv_bytes for row in rows),
        "endpoint_conserved": all(len(row["message_pairs"]) == 8 for row in rows),
    }


def _anchor_predictions(
    scored: dict[str, Any],
    fit: dict[str, Any],
) -> list[dict[str, Any]]:
    constant = scored["constants"]["tunable"][0]
    rows = []
    for prediction in scored["pre_tuning_predicted_bands"]:
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
                    constant,
                    fit["fitted_ps"],
                ),
            }
        )
    return rows


def run_study(
    anchors: dict[str, Any],
    scored: dict[str, Any],
    config: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Execute fit, live guards, packet arms, one-shot score and curves."""

    args.run_dir.mkdir(parents=True, exist_ok=False)
    fit = fit_frozen_constant(anchors, scored)
    fit_path = args.run_dir / "frozen-fit.json"
    write_json(fit_path, fit)
    fit_sha256 = sha256(fit_path)

    candidate_path = REPOSITORY_ROOT / config["candidate_record"]["path"]
    candidate = candidate_path.read_bytes()
    observations = []
    for observation in config["exact_shape_observations"]:
        observations.append(
            _runtime_observation(
                config,
                args.model_path,
                args.run_dir / "sessions",
                candidate,
                observation,
                suffix="run-a",
            )
        )
    identity_source = config["exact_shape_observations"][-1]
    identity_b = _runtime_observation(
        config,
        args.model_path,
        args.run_dir / "sessions",
        candidate,
        identity_source,
        suffix="run-b",
    )
    identity_a = observations[-1]
    stable_identity_equal = (
        identity_a["stable_requests"] == identity_b["stable_requests"]
    )

    packet = _packet_observation(config, args)

    score = score_frozen_fit(anchors, scored, fit)
    score["fit_sha256"] = fit_sha256
    score_path = args.run_dir / "held-out-score.json"
    write_json(score_path, score)
    score_sha256 = sha256(score_path)

    predictions = _anchor_predictions(scored, fit)
    standard_capacity = next(
        row["prediction"]
        for row in predictions
        if row["anchor_id"] == "sglang_decode_standard"
    )
    curves = [
        build_publication_curve(curve, standard_capacity)
        for curve in config["publication_curves"]
    ]
    selections = [_selection_summary(row) for row in observations]
    runtime_selected = {
        row["anchor_id"]: row["selected"] for row in selections
    }
    decode_selected = runtime_selected["sglang_decode_standard"]
    runtime_finding = (
        "The driver-level SGLang decode join did not expose remote KV length "
        "2000 in the worker request shape, so the exact present decode row "
        "delegated to the roofline comparator."
        if not decode_selected
        else "The exact standard-decode candidate row was selected in the live session."
    )
    result = {
        "schema": RESULT_SCHEMA,
        "status": score["status"],
        "verdict": f"SCORABLE_HELD_OUT_{score['status']}_MTP_BLOCKED",
        "classification": "scored",
        "scored_flagship": True,
        "scope": score["scope"],
        "core54_closure": False,
        "closure_reason": (
            "CORE-54 literal acceptance is not met because the two priced "
            "held-out anchors are refuted and MTP remains blocked."
        ),
        "provenance": {
            "run_head": _git_head(),
            "worktree": scored["dispatch_identity"]["worktree"],
            "branch": scored["dispatch_identity"]["branch"],
            "base_main_sha": scored["dispatch_identity"]["base_main_sha"],
            "expectations_commit": config["study"]["expectations_commit"],
            "expectations_sha256": sha256(SCORED_EXPECTATIONS_PATH),
            "anchor_sha256": sha256(ANCHOR_PATH),
            "configuration_sha256": sha256(args.config),
            "candidate_record_sha256": config["candidate_record"]["sha256"],
            "candidate_acceptance_status": config["candidate_record"][
                "acceptance_status"
            ],
            "candidate_calibration_claim": False,
            "sglang_version": SGLANG_VERSION,
            "sglang_commit": SGLANG_COMMIT,
            "python_version": ".".join(map(str, sys.version_info[:3])),
            "model_config_sha256": config["model"]["config_sha256"],
            "model_weights_loaded": False,
            "htsim_rnic_sha256": HTSIM_RNIC_SHA256,
            "txt2bin_sha256": TXT2BIN_SHA256,
        },
        "allocation": scored["allocation_ruling"],
        "scale_mapping": scored["largest_faithful_live_scale"],
        "topology": _topology_summary(),
        "constant_fit": fit,
        "constant_fit_sha256": fit_sha256,
        "held_out_score": score,
        "held_out_score_sha256": score_sha256,
        "anchor_predictions": predictions,
        "curves": curves,
        "session_observations": observations,
        "candidate_selections": selections,
        "stable_identity_guard": {
            "field_set": scored["stable_cross_run_identity_fields"],
            "first_sha256": identity_a["stable_projection_sha256"],
            "second_sha256": identity_b["stable_projection_sha256"],
            "equal": stable_identity_equal,
            "whole_request_bytes_compared": False,
        },
        "packet_observation": packet,
        "runtime_finding": runtime_finding,
        "dominant_held_out_contributor": (
            "The candidate-only EP32 capacity is 69.20 percent high at 2K. "
            "The shared collective fit selects its physical floor because the "
            "standard-decode calibration is already underpredicted, so the "
            "single bounded term cannot reconcile the opposing role errors."
        ),
        "residuals_required": [
            "COMP-72 exact MTP campaign cell",
            "SGL-38 remote-KV decode shape binding",
            "CORE-59 role-specific mechanistic residual calibration",
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
    anchors, scored, config = check_registry(args, require_runtime=not args.check_only)
    if args.check_only:
        print(
            "check-only: scored freeze, separate allocation, candidate digest, "
            "load grids and blocked MTP rule passed; no artifact written"
        )
        return 0
    result = run_study(anchors, scored, config, args)
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
