"""Run the frozen fabric-rendered KV handoff study."""

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
CORE51_EXPECTATIONS_PATH = REPOSITORY_ROOT / "examples/pd_session_v1/expectations.json"
TRACE_PATH = REPOSITORY_ROOT / "examples/preplay_trace_v1/granite_length_cap.jsonl"

EXPECTATIONS_COMMIT = "7536e08b32009951470f310e4f459216c7212dbc"
CONVERTER_CORRECTION_COMMIT = "a685a80b991d199a6de7ee8a39c490d8dd173aee"
IMPLEMENTATION_COMMIT = "8ce91b5644908b3c0a0fc6cdbc11c5ef369ccb6a"
RESULT_SCHEMA = "simllm-pd-session-fabric-handoff-study-result-v1"
MODEL_ID = "ibm-granite/granite-3.0-1b-a400m-instruct"
MODEL_REVISION = "ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445"
VLLM_VERSION = "0.27.1"
PROMPT_LENGTHS = (8, 16)
LINK_BANDWIDTHS_BPS = (200_000_000_000, 400_000_000_000)
DECODE_OUTPUT_TOKENS = 4
CONSTANT_HANDOFF_PS = 100_000_000
PCIE_SUBMISSION_PS = 20_000_000
RUN_ROOT_ENV = "SIMLLM_VLLM35_RUN_ROOT"


def render_cli_path(path: PurePath) -> str:
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


def _require_ancestor(commit: str, label: str) -> None:
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(f"{label} commit {commit} is not an ancestor of HEAD")


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


def _vllm_version(vllm_python: Path) -> str:
    completed = subprocess.run(
        [render_cli_path(vllm_python), "-c", "import vllm; print(vllm.__version__)"],
        check=True,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    return completed.stdout.strip().splitlines()[-1]


def _fraction_json(value: Fraction | None) -> dict[str, int] | None:
    if value is None:
        return None
    return {"numerator": value.numerator, "denominator": value.denominator}


def _fraction(value: dict[str, int]) -> Fraction:
    return Fraction(value["numerator"], value["denominator"])


def _validate_frozen_arithmetic(frozen: dict[str, Any]) -> None:
    transfer = frozen["transfer"]
    sweep = frozen["sweep"]
    if tuple(sweep["prompt_tokens"]) != PROMPT_LENGTHS:
        raise SystemExit("prompt sweep drifted")
    if tuple(sweep["link_bandwidth_bps"]) != LINK_BANDWIDTHS_BPS:
        raise SystemExit("bandwidth sweep drifted")
    if sweep["cells"] != len(PROMPT_LENGTHS) * len(LINK_BANDWIDTHS_BPS):
        raise SystemExit("packet cell count drifted")
    if transfer["pcie_submission_ps"] != PCIE_SUBMISSION_PS:
        raise SystemExit("PCIe submission term drifted")
    if transfer["constant_arm_ps"] != CONSTANT_HANDOFF_PS:
        raise SystemExit("constant arm drifted")
    if transfer["prefill_goal_ranks"] != list(range(8)):
        raise SystemExit("prefill endpoint interval drifted")
    if transfer["decode_goal_ranks"] != list(range(8, 16)):
        raise SystemExit("decode endpoint interval drifted")
    for prompt_tokens in PROMPT_LENGTHS:
        aggregate = prompt_tokens * frozen["model_geometry"][
            "kv_bytes_per_prompt_token"
        ]
        if sweep["aggregate_kv_bytes"][str(prompt_tokens)] != aggregate:
            raise SystemExit("aggregate KV byte arithmetic drifted")
        if sweep["per_pair_chunk_bytes"][str(prompt_tokens)] * 8 != aggregate:
            raise SystemExit("per-pair KV byte arithmetic drifted")


def _source_audit(frozen: dict[str, Any]) -> list[dict[str, str]]:
    rows = []
    as_of = frozen["as_of_commit"]
    for name, expected in frozen["source_audit_sha256"].items():
        actual = _sha256_bytes(_git_blob(as_of, name))
        if actual != expected:
            raise SystemExit(f"source audit hash disagrees for {name}")
        rows.append({"path": name, "sha256": actual, "scope": f"git-blob:{as_of}"})
    return rows


def _baseline_audit(frozen: dict[str, Any]) -> list[dict[str, str]]:
    rows = []
    for name, expected in frozen["core51_baseline_sha256"].items():
        actual = _sha256(REPOSITORY_ROOT / name)
        if actual != expected:
            raise SystemExit(f"CORE-51 baseline hash disagrees for {name}")
        rows.append({"path": name, "sha256": actual})
    return rows


def _gitlink(commit: str, path: str) -> str:
    completed = subprocess.run(
        ["git", "ls-tree", commit, path],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    fields = completed.stdout.split()
    if len(fields) < 3 or fields[0] != "160000":
        raise SystemExit(f"{path} is not a gitlink at {commit}")
    return fields[2]


def check_registry(args: argparse.Namespace) -> dict[str, Any]:
    if args.run_dir.exists():
        raise SystemExit(f"selected run directory already exists: {args.run_dir}")
    frozen = _load_json(EXPECTATIONS_PATH)
    core51 = _load_json(CORE51_EXPECTATIONS_PATH)
    _validate_frozen_arithmetic(frozen)
    _require_ancestor(EXPECTATIONS_COMMIT, "expectations")
    _require_ancestor(CONVERTER_CORRECTION_COMMIT, "converter correction")
    _require_ancestor(IMPLEMENTATION_COMMIT, "implementation")
    runtime = frozen["runtime"]
    if _gitlink(frozen["as_of_commit"], "third_party/htsim") != runtime[
        "htsim_gitlink"
    ]:
        raise SystemExit("frozen htsim gitlink disagrees")
    if _sha256(args.htsim_rnic) != runtime["htsim_rnic_sha256"]:
        raise SystemExit("htsim_rnic executable hash disagrees")
    if _sha256(args.txt2bin) != runtime["txt2bin_sha256"]:
        raise SystemExit("txt2bin executable hash disagrees")
    frontend = core51["frontend"]
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
        "converter_correction_commit": CONVERTER_CORRECTION_COMMIT,
        "implementation_commit": IMPLEMENTATION_COMMIT,
        "run_head": _git_head(),
        "source_audit": _source_audit(frozen),
        "baseline_audit": _baseline_audit(frozen),
        "htsim_gitlink": runtime["htsim_gitlink"],
        "htsim_rnic_sha256": runtime["htsim_rnic_sha256"],
        "txt2bin_sha256": runtime["txt2bin_sha256"],
        "model_config_sha256": frontend["model_config_sha256"],
        "fixture_sha256": frontend["fixture_sha256"],
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
        vocab_size=49_155,
        dtype_bytes=2,
        num_experts=32,
        top_k=8,
        moe_intermediate_size=64,
        local_num_experts=32,
    )


def _session_config(workdir: Path) -> Any:
    from simllm.adapters.vllm.pd_session import VllmPdSessionConfig
    from simllm.core import DeclaredKvHandoffPolicy, KvHandoffGeometry

    return VllmPdSessionConfig(
        model=MODEL_ID,
        model_revision=MODEL_REVISION,
        workdir=workdir,
        dims=_granite_dims(),
        handoff_geometry=KvHandoffGeometry(24, 8, 64, 2),
        handoff_policy=DeclaredKvHandoffPolicy(CONSTANT_HANDOFF_PS),
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


def _result_row(result: Any) -> dict[str, Any]:
    return {
        "timeline": result.timeline.to_json(),
        "decode_token_ids": list(result.decode_token_ids),
        "kv_transfer_params": dict(result.kv_transfer_params),
        "tpot_ps": _fraction_json(result.timeline.tpot_ps),
    }


def run_observation(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    from simllm.adapters.vllm.pd_session import VllmDisaggregatedSession
    from simllm.core import DeclaredKvHandoffPolicy
    from simllm.traffic import PacketKvHandoffPolicy

    prompt = _prompt_tokens()
    controls: dict[str, dict[str, Any]] = {}
    cells = []
    packet_root = run_dir / "packet-artifacts"
    with VllmDisaggregatedSession(_session_config(run_dir / "session")) as session:
        for prompt_tokens in PROMPT_LENGTHS:
            packet_artifacts_before = (
                0 if not packet_root.exists() else len(tuple(packet_root.rglob("*")))
            )
            off = session.run_request(
                f"prompt{prompt_tokens}-off",
                prompt[:prompt_tokens],
                decode_output_tokens=DECODE_OUTPUT_TOKENS,
                handoff_policy=DeclaredKvHandoffPolicy.off(),
            )
            constant = session.run_request(
                f"prompt{prompt_tokens}-constant",
                prompt[:prompt_tokens],
                decode_output_tokens=DECODE_OUTPUT_TOKENS,
                handoff_policy=DeclaredKvHandoffPolicy(CONSTANT_HANDOFF_PS),
            )
            controls[str(prompt_tokens)] = {
                "off": _result_row(off),
                "constant": _result_row(constant),
                "packet_artifact_count_before_controls": packet_artifacts_before,
                "packet_artifact_count_after_controls": (
                    0 if not packet_root.exists() else len(tuple(packet_root.rglob("*")))
                ),
            }
            for bandwidth in LINK_BANDWIDTHS_BPS:
                request_id = f"prompt{prompt_tokens}-bandwidth{bandwidth}-packet"
                policy = PacketKvHandoffPolicy(
                    artifact_dir=packet_root,
                    linkspeed_bps=bandwidth,
                    txt2bin=args.txt2bin,
                    htsim_rnic=args.htsim_rnic,
                    pcie_submission_ps=PCIE_SUBMISSION_PS,
                )
                packet = session.run_request(
                    request_id,
                    prompt[:prompt_tokens],
                    decode_output_tokens=DECODE_OUTPUT_TOKENS,
                    handoff_policy=policy,
                )
                artifact = policy.artifacts[0]
                cells.append(
                    {
                        "prompt_tokens": prompt_tokens,
                        "link_bandwidth_bps": bandwidth,
                        "packet": _result_row(packet),
                        "artifact": artifact.to_json(),
                        "artifact_sha256": {
                            "goal": _sha256(artifact.goal_path),
                            "goal_binary": _sha256(artifact.goal_binary_path),
                            "completion_csv": _sha256(artifact.completion_csv_path),
                            "manifest": _sha256(artifact.manifest_path),
                        },
                    }
                )
    return {
        "controls": controls,
        "cells": cells,
        "packet_artifact_request_directories": (
            0
            if not packet_root.exists()
            else len(tuple(path for path in packet_root.iterdir() if path.is_dir()))
        ),
    }


def analyze_observation(
    observation: dict[str, Any],
    frozen: dict[str, Any],
) -> dict[str, Any]:
    fatal_findings = []
    controls = observation["controls"]
    expected_controls = frozen["constant_arm_timing_control"]
    for prompt_tokens in PROMPT_LENGTHS:
        control = controls[str(prompt_tokens)]
        constant = control["constant"]
        off = control["off"]
        expected = expected_controls[str(prompt_tokens)]
        constant_timeline = constant["timeline"]
        off_timeline = off["timeline"]
        if (
            constant_timeline["ttft_ps"] != expected["ttft_ps"]
            or _fraction(constant["tpot_ps"]) != expected["tpot_ps"]
            or constant_timeline["decomposition"]["handoff_ps"]
            != CONSTANT_HANDOFF_PS
            or off_timeline["ttft_ps"] != expected["ttft_ps"] - CONSTANT_HANDOFF_PS
            or _fraction(off["tpot_ps"]) != expected["tpot_ps"]
            or off_timeline["decomposition"]["handoff_ps"] != 0
        ):
            fatal_findings.append(
                {"guard": "constant-or-off-control", "prompt_tokens": prompt_tokens}
            )
        if (
            control["packet_artifact_count_before_controls"]
            != control["packet_artifact_count_after_controls"]
        ):
            fatal_findings.append(
                {"guard": "constant-or-off-artifact-emission", "prompt_tokens": prompt_tokens}
            )

    cells = observation["cells"]
    if len(cells) != 4 or len(
        {(cell["prompt_tokens"], cell["link_bandwidth_bps"]) for cell in cells}
    ) != 4:
        fatal_findings.append({"guard": "complete-unique-cell-registry"})
    exact_rows = []
    for cell in cells:
        prompt_tokens = cell["prompt_tokens"]
        bandwidth = cell["link_bandwidth_bps"]
        packet = cell["packet"]
        timeline = packet["timeline"]
        handoff = timeline["handoff"]
        artifact = cell["artifact"]
        expected_bytes = frozen["sweep"]["aggregate_kv_bytes"][str(prompt_tokens)]
        expected_chunk = frozen["sweep"]["per_pair_chunk_bytes"][str(prompt_tokens)]
        expected_pairs = [(index, index + 8) for index in range(8)]
        message_pairs = [
            (message["source_rank"], message["destination_rank"])
            for message in artifact["messages"]
        ]
        flow_pairs = [
            (flow["source"], flow["destination"]) for flow in artifact["flows"]
        ]
        last_arrival = max(flow["completion_time_ps"] for flow in artifact["flows"])
        packet_duration = handoff["completed_at_ps"] - handoff["submitted_at_ps"]
        packet_service = handoff["finished_at_ps"] - handoff["started_at_ps"]
        constant = controls[str(prompt_tokens)]["constant"]
        metric_residual = (
            timeline["ttft_ps"]
            - constant["timeline"]["ttft_ps"]
            - (packet_duration - CONSTANT_HANDOFF_PS)
        )
        bounds = frozen["physical_bounds_ps"][str(prompt_tokens)][str(bandwidth)]
        held = (
            handoff["kv_bytes"] == expected_bytes
            and artifact["aggregate_kv_bytes"] == expected_bytes
            and artifact["chunk_bytes"] == [expected_chunk] * 8
            and sum(artifact["chunk_bytes"]) == expected_bytes
            and message_pairs == expected_pairs
            and flow_pairs == expected_pairs
            and sum(message["payload_bytes"] for message in artifact["messages"])
            == expected_bytes
            and sum(flow["payload_bytes"] for flow in artifact["flows"])
            == expected_bytes
            and handoff["eligible_at_ps"] - handoff["submitted_at_ps"]
            == PCIE_SUBMISSION_PS
            and handoff["started_at_ps"] == handoff["eligible_at_ps"]
            and handoff["completed_at_ps"] == handoff["finished_at_ps"]
            and handoff["finished_at_ps"] - handoff["submitted_at_ps"]
            == last_arrival
            and artifact["last_required_arrival_ps"] == last_arrival
            and artifact["quiescent"] is True
            and metric_residual == 0
            and _fraction(packet["tpot_ps"]) == _fraction(constant["tpot_ps"])
            and packet["decode_token_ids"] == constant["decode_token_ids"]
            and bounds["parallel_serialization_floor"]
            <= packet_service
            <= bounds["packet_service_ceiling"]
        )
        exact_rows.append(
            {
                "prompt_tokens": prompt_tokens,
                "link_bandwidth_bps": bandwidth,
                "aggregate_kv_bytes": expected_bytes,
                "chunks": len(artifact["chunk_bytes"]),
                "flows": len(artifact["flows"]),
                "packet_service_ps": packet_service,
                "packet_duration_ps": packet_duration,
                "ttft_ps": timeline["ttft_ps"],
                "constant_ttft_ps": constant["timeline"]["ttft_ps"],
                "signed_ttft_difference_ps": (
                    timeline["ttft_ps"] - constant["timeline"]["ttft_ps"]
                ),
                "signed_handoff_difference_ps": (
                    packet_duration - CONSTANT_HANDOFF_PS
                ),
                "metric_residual_ps": metric_residual,
                "tpot_ps": packet["tpot_ps"],
                "held": held,
            }
        )
        if not held:
            fatal_findings.append(
                {
                    "guard": "packet-cell-exact-oracle",
                    "prompt_tokens": prompt_tokens,
                    "link_bandwidth_bps": bandwidth,
                }
            )

    by_key = {
        (row["prompt_tokens"], row["link_bandwidth_bps"]): row for row in exact_rows
    }
    behavioral = []
    for prompt_tokens in PROMPT_LENGTHS:
        slow = by_key[(prompt_tokens, LINK_BANDWIDTHS_BPS[0])]
        fast = by_key[(prompt_tokens, LINK_BANDWIDTHS_BPS[1])]
        behavioral.append(
            {
                "family": "halving-bandwidth-does-not-reduce-service",
                "prompt_tokens": prompt_tokens,
                "held": slow["packet_service_ps"] >= fast["packet_service_ps"],
            }
        )
    for bandwidth in LINK_BANDWIDTHS_BPS:
        short = by_key[(8, bandwidth)]
        long = by_key[(16, bandwidth)]
        behavioral.append(
            {
                "family": "doubling-context-does-not-reduce-service",
                "link_bandwidth_bps": bandwidth,
                "held": long["packet_service_ps"] >= short["packet_service_ps"],
            }
        )
    for row in exact_rows:
        behavioral.append(
            {
                "family": "only-packet-arm-emits-backend-artifacts",
                "prompt_tokens": row["prompt_tokens"],
                "link_bandwidth_bps": row["link_bandwidth_bps"],
                "held": observation["packet_artifact_request_directories"] == 4,
            }
        )
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
    parser.add_argument("--htsim-rnic", required=True, type=Path)
    parser.add_argument("--txt2bin", required=True, type=Path)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    provenance = check_registry(args)
    if args.check_only:
        print(
            "check-only validated four packet cells, source, runtime, backend, "
            "converter and all CORE-51 baseline digests; no artifacts produced"
        )
        return
    _require_clean_worktree()
    _validate_run_dir(args.run_dir)
    args.run_dir.mkdir(parents=True, exist_ok=False)
    frozen = _load_json(EXPECTATIONS_PATH)
    observation = run_observation(args.run_dir, args)
    analysis = analyze_observation(observation, frozen)
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
