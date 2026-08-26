#!/usr/bin/env python3
"""Run the frozen SGL-33 SGLang disaggregated-session study."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXPECTATIONS_PATH = Path(__file__).with_name("expectations.json")
TRACE_PATH = REPOSITORY_ROOT / "examples/preplay_trace_v1/granite_length_cap.jsonl"

EXPECTATIONS_COMMIT = "191c237700431507639f963a0ee69e94292611c6"
SESSION_IMPLEMENTATION_COMMIT = "f6bd963b9d92d786e9363d3e646364a329d7bc69"
CONVERTER_CORRECTION_COMMIT = "a685a80b991d199a6de7ee8a39c490d8dd173aee"
SGLANG_COMMIT = "bfeae4e79a8dc4600e006f1a5fbc85321a01c1a3"
SGLANG_VERSION = "0.5.19.dev345+gbfeae4e79"
MODEL_REVISION = "ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445"
MODEL_CONFIG_SHA256 = "ca4bb3a5c1bdef988ab413e0d731640446da65316e4ed16de3666cd96ecc3a0b"
TRACE_SHA256 = "36334f3aaa767c46d5f9c8498e02f6c2805a46e5000a57aea2747e17dd5d1341"
HTSIM_GITLINK = "1dcbfec36a33753bf978cf6323bade1a6645fe4f"
HTSIM_RNIC_SHA256 = "388415f92d6ef54c84bb5d2b7f7dabcaad27574ec235d62260f08175f3958bd9"
TXT2BIN_SHA256 = "f3745f34ad86febe9c9eebef10aee5fae00b8865cb29943344fb75b0f142495b"

POOL_RATIOS = ((1, 1), (1, 2), (2, 1))
PROMPT_LENGTHS = (8, 16)
OFFERED_LOADS = (8_000, 16_000, 32_000)
INTERARRIVAL_PS = (125_000_000, 62_500_000, 31_250_000)
REQUESTS_PER_CELL = 8
DECODE_OUTPUT_TOKENS = 4
CONSTANT_HANDOFF_PS = 100_000_000
SENSITIVITY_HANDOFF_PS = 200_000_000
PCIE_SUBMISSION_PS = 20_000_000
LINKSPEED_BPS = 400_000_000_000
PS_PER_SECOND = 1_000_000_000_000
STEP_FLOOR_PS = 1_000_000
STEP_CEILING_PS = 100_000_000_000
DECODE_RATE_FLOOR = 10
DECODE_RATE_CEILING = 100_000
PACKET_SHARD_FLOOR_PS = 983_040
PACKET_SERVICE_CEILING_PS = 81_457_280


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_head(path: Path = REPOSITORY_ROOT) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _git_blob(commit: str, path: str) -> bytes:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
    )
    return completed.stdout


def _gitlink(commit: str, path: str) -> str:
    completed = subprocess.run(
        ["git", "ls-tree", commit, path],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    fields = completed.stdout.strip().split()
    if len(fields) < 3 or fields[1] != "commit":
        raise SystemExit(f"{path} is not a gitlink at {commit}")
    return fields[2]


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
        raise SystemExit("the scored run requires a clean tracked worktree")


def _fraction_json(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _fraction(value: dict[str, int] | None) -> Fraction | None:
    if value is None:
        return None
    return Fraction(value["numerator"], value["denominator"])


def _source_audit(frozen: dict[str, Any], sglang_source: Path) -> list[dict[str, str]]:
    rows = []
    as_of = frozen["as_of_commit"]
    for name, expected in frozen["source_audit_sha256"].items():
        if name.startswith("simllm/"):
            actual = _sha256_bytes(_git_blob(as_of, name))
            scope = f"git-blob:{as_of}"
        else:
            actual = _sha256(sglang_source / name)
            scope = "pinned-sglang-source"
        if actual != expected:
            raise SystemExit(f"source audit disagrees for {name}: {actual} != {expected}")
        rows.append({"path": name, "sha256": actual, "scope": scope})
    return rows


def _baseline_audit(frozen: dict[str, Any]) -> list[dict[str, str]]:
    rows = []
    for name, expected in frozen["baseline_sha256"].items():
        actual = _sha256(REPOSITORY_ROOT / name)
        if actual != expected:
            raise SystemExit(f"baseline audit disagrees for {name}: {actual} != {expected}")
        rows.append({"path": name, "sha256": actual})
    return rows


def _validate_frozen_registry(frozen: dict[str, Any]) -> None:
    if frozen["schema"] != "simllm-sglang-pd-session-expectations-v1":
        raise SystemExit("expectations schema drifted")
    if frozen["frontend"]["commit"] != SGLANG_COMMIT:
        raise SystemExit("SGLang commit drifted")
    if frozen["frontend"]["version"] != SGLANG_VERSION:
        raise SystemExit("SGLang version drifted")
    if tuple(
        (row["prefill_engines"], row["decode_engines"])
        for row in frozen["deployment"]["pool_ratios"]
    ) != POOL_RATIOS:
        raise SystemExit("pool ratios drifted")
    sweep = frozen["request_sweep"]
    if tuple(sweep["offered_load_requests_per_second"]) != OFFERED_LOADS:
        raise SystemExit("offered loads drifted")
    if tuple(sweep["interarrival_ps"]) != INTERARRIVAL_PS:
        raise SystemExit("interarrival intervals drifted")
    if any(
        load * interval != PS_PER_SECOND
        for load, interval in zip(OFFERED_LOADS, INTERARRIVAL_PS, strict=True)
    ):
        raise SystemExit("offered load arithmetic drifted")
    expected = (
        tuple(sweep["prompt_tokens"]),
        sweep["requests_per_cell"],
        sweep["decode_output_tokens_per_request"],
        sweep["declared_handoff_ps"],
        sweep["cells"],
    )
    if expected != (PROMPT_LENGTHS, 8, 4, CONSTANT_HANDOFF_PS, 18):
        raise SystemExit("request sweep drifted")


def check_registry(args: argparse.Namespace, *, require_clean: bool) -> dict[str, Any]:
    """Validate every frozen input before importing session behavior."""

    frozen = _load_json(EXPECTATIONS_PATH)
    _validate_frozen_registry(frozen)
    _require_ancestor(EXPECTATIONS_COMMIT)
    _require_ancestor(SESSION_IMPLEMENTATION_COMMIT)
    _require_ancestor(CONVERTER_CORRECTION_COMMIT)
    if require_clean:
        _require_clean_tracked_worktree()
    paths = (args.run_dir, args.model_path, args.sglang_source, args.txt2bin, args.htsim_rnic)
    if any(not path.is_absolute() for path in paths):
        raise SystemExit("all runtime paths must be explicit absolute paths")
    if args.run_dir.exists():
        raise SystemExit(f"run directory already exists: {args.run_dir}")
    if not args.model_path.is_dir():
        raise SystemExit("the cached model snapshot is missing")
    model_config = args.model_path / "config.json"
    if _sha256(model_config) != MODEL_CONFIG_SHA256:
        raise SystemExit("the cached model configuration hash disagrees")
    if not args.sglang_source.is_dir() or _git_head(args.sglang_source) != SGLANG_COMMIT:
        raise SystemExit("the pinned SGLang source commit disagrees")
    if importlib.metadata.version("sglang") != SGLANG_VERSION:
        raise SystemExit("the installed SGLang version disagrees")
    if sys.version_info[:3] != (3, 10, 18):
        raise SystemExit("the SGLang study requires Python 3.10.18")
    if _sha256(TRACE_PATH) != TRACE_SHA256:
        raise SystemExit("the Granite prompt fixture hash disagrees")
    if _sha256(args.htsim_rnic) != HTSIM_RNIC_SHA256:
        raise SystemExit("the accepted htsim_rnic binary hash disagrees")
    if _sha256(args.txt2bin) != TXT2BIN_SHA256:
        raise SystemExit("the accepted txt2bin binary hash disagrees")
    if _gitlink(frozen["as_of_commit"], "third_party/htsim") != HTSIM_GITLINK:
        raise SystemExit("the htsim gitlink disagrees")
    required_environment = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "SIMLLM_SGLANG_ENABLE": "1",
    }
    for name, expected_value in required_environment.items():
        if os.environ.get(name) != expected_value:
            raise SystemExit(f"{name}={expected_value} is required")
    pythonpath = os.environ.get("PYTHONPATH", "").split(os.pathsep)
    if not pythonpath or Path(pythonpath[0]).resolve() != REPOSITORY_ROOT:
        raise SystemExit("PYTHONPATH must begin with the selected worktree")
    return {
        "expectations_commit": EXPECTATIONS_COMMIT,
        "session_implementation_commit": SESSION_IMPLEMENTATION_COMMIT,
        "converter_correction_commit": CONVERTER_CORRECTION_COMMIT,
        "run_head": _git_head(),
        "python_version": ".".join(map(str, sys.version_info[:3])),
        "sglang_version": SGLANG_VERSION,
        "sglang_commit": SGLANG_COMMIT,
        "model_config_sha256": MODEL_CONFIG_SHA256,
        "fixture_sha256": TRACE_SHA256,
        "htsim_gitlink": HTSIM_GITLINK,
        "htsim_rnic_sha256": HTSIM_RNIC_SHA256,
        "txt2bin_sha256": TXT2BIN_SHA256,
        "source_audit": _source_audit(frozen, args.sglang_source),
        "baseline_audit": _baseline_audit(frozen),
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


def _arrangement(size: int, *, dense_size: int | None = None) -> Any:
    from simllm.placement import SglangPoolArrangement

    return SglangPoolArrangement(
        enable_data_parallel_attention=True,
        attention_data_parallel_size=size,
        dense_data_parallel_size=size if dense_size is None else dense_size,
        expert_parallel_size=size,
    )


def _session_config(
    model_path: Path,
    workdir: Path,
    *,
    prefill_engines: int,
    decode_engines: int,
) -> Any:
    from simllm.adapters.sglang import SglangPdSessionConfig
    from simllm.core import DeclaredKvHandoffPolicy, KvHandoffGeometry

    return SglangPdSessionConfig(
        model_path=model_path,
        workdir=workdir,
        dims=_granite_dims(),
        handoff_geometry=KvHandoffGeometry(24, 8, 64, 2),
        handoff_policy=DeclaredKvHandoffPolicy(CONSTANT_HANDOFF_PS),
        prefill_arrangement=_arrangement(prefill_engines * 8),
        decode_arrangement=_arrangement(decode_engines * 8),
        prefill_engines=prefill_engines,
        decode_engines=decode_engines,
        simulated_gpus_per_engine=8,
        context_length=64,
        max_total_tokens=64,
        max_running_requests=8,
        token_id=512,
        random_seed=173,
    )


def _base_prompt() -> tuple[int, ...]:
    for line in TRACE_PATH.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("row_type") == "request":
            tokens = tuple(row["input_token_ids"])
            if len(tokens) < max(PROMPT_LENGTHS):
                raise RuntimeError("the frozen prompt fixture is too short")
            return tokens
    raise RuntimeError("the frozen prompt fixture has no request row")


def _distinct_prompt(base: tuple[int, ...], length: int, seed: int) -> tuple[int, ...]:
    return (1_000 + seed, *base[1:length])


def _request_row(result: Any) -> dict[str, Any]:
    tpot = result.timeline.tpot_ps
    return {
        "request_id": result.timeline.request_id,
        "timeline": result.timeline.to_json(),
        "prefill_engine_id": result.prefill_engine_id,
        "decode_engine_id": result.decode_engine_id,
        "prefill_internal_request_id": result.prefill_internal_request_id,
        "decode_internal_request_id": result.decode_internal_request_id,
        "bootstrap_token_id": result.bootstrap_token_id,
        "decode_token_ids": list(result.decode_token_ids),
        "join_metadata": dict(result.join_metadata),
        "prefill_step_latencies_ps": [
            row.step_latency_ps for row in result.prefill_results
        ],
        "decode_step_latencies_ps": [
            row.step_latency_ps for row in result.decode_results
        ],
        "tpot_ps": None if tpot is None else _fraction_json(tpot),
    }


def _cell(
    session: Any,
    *,
    base_prompt: tuple[int, ...],
    cell_ordinal: int,
    prefill_engines: int,
    decode_engines: int,
    prompt_tokens: int,
    offered_load: int,
    interarrival_ps: int,
) -> tuple[dict[str, Any], Any]:
    from simllm.adapters.sglang import SglangPdRequest

    start = session.clock.now_ps
    prompt = _distinct_prompt(base_prompt, prompt_tokens, cell_ordinal)
    requests = tuple(
        SglangPdRequest(
            request_id=(
                f"p{prefill_engines}-d{decode_engines}-prompt{prompt_tokens}-"
                f"load{offered_load}-request{index}"
            ),
            prompt_token_ids=prompt,
            decode_output_tokens=DECODE_OUTPUT_TOKENS,
            admitted_at_ps=start + index * interarrival_ps,
        )
        for index in range(REQUESTS_PER_CELL)
    )
    result = session.run_requests(requests)
    point = result.curve_point(Fraction(offered_load))
    return (
        {
            "prefill_engines": prefill_engines,
            "decode_engines": decode_engines,
            "prompt_tokens": prompt_tokens,
            "offered_load_requests_per_second": offered_load,
            "interarrival_ps": interarrival_ps,
            "admitted_request_ids": [row.request_id for row in requests],
            "requests": [_request_row(row) for row in result.requests],
            "handoff_request_ids": [row.request_id for row in session.handoffs[-8:]],
            "prefill_batches": [list(batch) for batch in result.prefill_batches],
            "decode_batches": [list(batch) for batch in result.decode_batches],
            "maximum_prefill_batch_size": result.maximum_prefill_batch_size,
            "maximum_decode_batch_size": result.maximum_decode_batch_size,
            "curve_point": point.to_json(),
        },
        point,
    )


def _single_request(
    session: Any,
    *,
    base_prompt: tuple[int, ...],
    request_id: str,
    seed: int,
    policy: Any,
) -> dict[str, Any]:
    from simllm.adapters.sglang import SglangPdRequest

    request = SglangPdRequest(
        request_id=request_id,
        prompt_token_ids=_distinct_prompt(base_prompt, 8, seed),
        decode_output_tokens=DECODE_OUTPUT_TOKENS,
        admitted_at_ps=session.clock.now_ps,
    )
    result = session.run_requests((request,), handoff_policy=policy)
    return _request_row(result.requests[0])


def _pool_metadata(session: Any) -> dict[str, Any]:
    engines = (*session.prefill_engines, *session.decode_engines)
    return {
        "engine_count": len(engines),
        "process_ids": [engine.process_id for engine in engines],
        "engine_ids": [engine.engine_id for engine in engines],
        "roles": [engine.role.value for engine in engines],
        "scheduler_types": [engine.scheduler_type for engine in engines],
        "worker_types": [engine.worker_type for engine in engines],
        "simulated_worker_counts": [engine.simulated_worker_count for engine in engines],
        "tensor_parallel_sizes": [
            len(engine.tensor_parallel_ranks) for engine in engines
        ],
        "attention_data_parallel_sizes": [
            len(engine.attention_data_parallel_ranks) for engine in engines
        ],
        "dense_data_parallel_sizes": [
            len(engine.dense_data_parallel_ranks) for engine in engines
        ],
        "expert_parallel_sizes": [
            len(engine.expert_parallel_ranks) for engine in engines
        ],
        "construction_seconds": [engine.construction_seconds for engine in engines],
    }


def _flagship_render() -> dict[str, Any]:
    from simllm.placement import sglang_disaggregated_manifests

    manifests = sglang_disaggregated_manifests(
        prefill_nodes=4,
        decode_nodes=9,
        gpus_per_node=8,
        prefill_arrangement=_arrangement(32),
        decode_arrangement=_arrangement(72, dense_size=1),
        framework_version=SGLANG_VERSION,
    )
    placement = manifests.placement
    return {
        "nodes": len(manifests.fabric.nodes),
        "ranks": len(placement.ranks),
        "gpus": sum(len(node.gpus) for node in manifests.fabric.nodes),
        "nics": sum(len(node.nics) for node in manifests.fabric.nodes),
        "prefill_attn_dp": placement.group_ranks(0, "attn_dp"),
        "prefill_tp": placement.group_ranks(0, "tp"),
        "prefill_dense_dp": placement.group_ranks(0, "dense_dp"),
        "prefill_ep": placement.group_ranks(0, "ep"),
        "decode_attn_dp": placement.group_ranks(32, "attn_dp"),
        "decode_tp": placement.group_ranks(32, "tp"),
        "decode_dense_dp": placement.group_ranks(32, "dense_dp"),
        "decode_ep": placement.group_ranks(32, "ep"),
        "core54_claimed_ranks": 96,
    }


def run_observation(args: argparse.Namespace, provenance: dict[str, Any]) -> dict[str, Any]:
    from simllm.adapters.sglang import (
        SglangDisaggregatedSession,
        SglangPdCurveRecord,
    )
    from simllm.core import DeclaredKvHandoffPolicy
    from simllm.traffic import PacketKvHandoffPolicy

    os.environ["SIMLLM_HTSIM_RNIC"] = args.htsim_rnic.as_posix()
    os.environ["SIMLLM_TXT2BIN"] = args.txt2bin.as_posix()
    base_prompt = _base_prompt()
    cells = []
    curves = []
    pools = []
    controls = None
    packet = None
    cell_ordinal = 1
    for prefill_engines, decode_engines in POOL_RATIOS:
        ratio_dir = args.run_dir / f"p{prefill_engines}-d{decode_engines}"
        with SglangDisaggregatedSession(
            _session_config(
                args.model_path,
                ratio_dir,
                prefill_engines=prefill_engines,
                decode_engines=decode_engines,
            )
        ) as session:
            pools.append(
                {
                    "prefill_engines": prefill_engines,
                    "decode_engines": decode_engines,
                    **_pool_metadata(session),
                }
            )
            for prompt_tokens in PROMPT_LENGTHS:
                points = []
                for offered_load, interarrival_ps in zip(
                    OFFERED_LOADS,
                    INTERARRIVAL_PS,
                    strict=True,
                ):
                    observation, point = _cell(
                        session,
                        base_prompt=base_prompt,
                        cell_ordinal=cell_ordinal,
                        prefill_engines=prefill_engines,
                        decode_engines=decode_engines,
                        prompt_tokens=prompt_tokens,
                        offered_load=offered_load,
                        interarrival_ps=interarrival_ps,
                    )
                    cell_ordinal += 1
                    cells.append(observation)
                    points.append(point)
                curves.append(
                    SglangPdCurveRecord(
                        configuration_id=(
                            f"sglang-p{prefill_engines}-d{decode_engines}-"
                            f"prompt{prompt_tokens}"
                        ),
                        prefill_engines=prefill_engines,
                        decode_engines=decode_engines,
                        prompt_tokens=prompt_tokens,
                        points=tuple(points),
                    ).to_json()
                )
            if (prefill_engines, decode_engines) == (1, 1):
                packet_root = args.run_dir / "packet-control"
                artifact_count_before = 0 if not packet_root.exists() else len(
                    tuple(packet_root.iterdir())
                )
                constant = _single_request(
                    session,
                    base_prompt=base_prompt,
                    request_id="constant-100us-control",
                    seed=100,
                    policy=DeclaredKvHandoffPolicy(CONSTANT_HANDOFF_PS),
                )
                sensitivity = _single_request(
                    session,
                    base_prompt=base_prompt,
                    request_id="constant-200us-control",
                    seed=101,
                    policy=DeclaredKvHandoffPolicy(SENSITIVITY_HANDOFF_PS),
                )
                artifact_count_after_constants = (
                    0 if not packet_root.exists() else len(tuple(packet_root.iterdir()))
                )
                prefill_ranks, decode_ranks = session.packet_rank_sets()
                packet_policy = PacketKvHandoffPolicy(
                    artifact_dir=packet_root,
                    linkspeed_bps=LINKSPEED_BPS,
                    txt2bin=args.txt2bin,
                    htsim_rnic=args.htsim_rnic,
                    pcie_submission_ps=PCIE_SUBMISSION_PS,
                    prefill_ranks=prefill_ranks,
                    decode_ranks=decode_ranks,
                )
                packet_row = _single_request(
                    session,
                    base_prompt=base_prompt,
                    request_id="packet-400g-control",
                    seed=102,
                    policy=packet_policy,
                )
                controls = {
                    "constant": constant,
                    "sensitivity": sensitivity,
                    "packet_artifact_count_before": artifact_count_before,
                    "packet_artifact_count_after_constants": (
                        artifact_count_after_constants
                    ),
                }
                packet = {
                    "request": packet_row,
                    "artifact": packet_policy.artifacts[0].to_json(),
                    "prefill_ranks": list(prefill_ranks),
                    "decode_ranks": list(decode_ranks),
                }
    if controls is None or packet is None:
        raise RuntimeError("the one-plus-one controls did not run")
    return {
        "schema": "simllm-sglang-pd-session-raw-result-v1",
        "provenance": provenance,
        "pools": pools,
        "cells": cells,
        "curves": curves,
        "controls": controls,
        "packet": packet,
        "flagship": _flagship_render(),
    }


def _cell_key(cell: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        cell["prefill_engines"],
        cell["decode_engines"],
        cell["prompt_tokens"],
        cell["offered_load_requests_per_second"],
    )


def _curve_fraction(point: dict[str, Any], name: str) -> Fraction:
    return Fraction(point[name]["numerator"], point[name]["denominator"])


def analyze_observation(observation: dict[str, Any]) -> dict[str, Any]:
    """Apply the frozen fatal guards and behavioral relations."""

    fatal = []
    exact_rows = []
    step_latencies = []
    decode_rates = []
    all_prefill_internal = []
    all_decode_internal = []
    cells = observation["cells"]
    if len(cells) != 18 or len({_cell_key(cell) for cell in cells}) != 18:
        fatal.append({"guard": "complete-unique-cell-registry"})
    for cell in cells:
        requests = cell["requests"]
        admitted = cell["admitted_request_ids"]
        handed_off = cell["handoff_request_ids"]
        terminal = [row["request_id"] for row in requests]
        residuals = [
            row["timeline"]["decomposition"]["total_ps"]
            - row["timeline"]["ttft_ps"]
            for row in requests
        ]
        token_count = sum(len(row["decode_token_ids"]) for row in requests)
        prefill_internal = [row["prefill_internal_request_id"] for row in requests]
        decode_internal = [row["decode_internal_request_id"] for row in requests]
        all_prefill_internal.extend(prefill_internal)
        all_decode_internal.extend(decode_internal)
        expected_widths = {
            "prefill": cell["prefill_engines"] * 8,
            "decode": cell["decode_engines"] * 8,
        }
        arrangements_hold = all(
            row["join_metadata"][f"{role}_arrangement"][
                "attention_data_parallel_size"
            ]
            == width
            and row["join_metadata"][f"{role}_arrangement"][
                "dense_data_parallel_size"
            ]
            == width
            and row["join_metadata"][f"{role}_arrangement"]["expert_parallel_size"]
            == width
            and len(row["join_metadata"][f"{role}_expert_parallel_ranks"])
            == width
            for row in requests
            for role, width in expected_widths.items()
        )
        identities_hold = (
            admitted == handed_off == terminal
            and len(set(admitted)) == REQUESTS_PER_CELL
            and len(set(prefill_internal)) == REQUESTS_PER_CELL
            and len(set(decode_internal)) == REQUESTS_PER_CELL
            and all(left != right for left, right in zip(
                prefill_internal,
                decode_internal,
                strict=True,
            ))
        )
        curve = cell["curve_point"]
        held = (
            len(requests) == REQUESTS_PER_CELL
            and identities_hold
            and token_count == REQUESTS_PER_CELL * DECODE_OUTPUT_TOKENS
            and all(len(row["decode_token_ids"]) == 4 for row in requests)
            and all(residual == 0 for residual in residuals)
            and all(
                row["timeline"]["handoff"]["kv_bytes"]
                == cell["prompt_tokens"] * 49_152
                for row in requests
            )
            and arrangements_hold
            and curve["schema"] == "simllm-deployment-curve-point-v1"
            and curve["request_count"] == REQUESTS_PER_CELL
            and curve["output_token_count"] == 32
        )
        exact_rows.append(
            {
                "cell": list(_cell_key(cell)),
                "held": held,
                "admissions": len(admitted),
                "handoffs": len(handed_off),
                "terminals": len(terminal),
                "decode_tokens": token_count,
                "maximum_ttft_residual_ps": max(map(abs, residuals), default=0),
            }
        )
        if not held:
            fatal.append({"guard": "cell-exact-conservation", "cell": list(_cell_key(cell))})
        for row in requests:
            step_latencies.extend(row["prefill_step_latencies_ps"])
            step_latencies.extend(row["decode_step_latencies_ps"])
            tpot = _fraction(row["tpot_ps"])
            if tpot is None or tpot <= 0:
                fatal.append({"guard": "missing-positive-tpot", "request": row["request_id"]})
            else:
                decode_rates.append(Fraction(PS_PER_SECOND, 1) / tpot)
    if len(set(all_prefill_internal)) != len(all_prefill_internal):
        fatal.append({"guard": "prefill-internal-identity-reuse"})
    if len(set(all_decode_internal)) != len(all_decode_internal):
        fatal.append({"guard": "decode-internal-identity-reuse"})

    for pool in observation["pools"]:
        process_ids = pool["process_ids"]
        expected_engine_count = pool["prefill_engines"] + pool["decode_engines"]
        pool_held = (
            pool["engine_count"] == expected_engine_count
            and len(process_ids) == len(set(process_ids)) == expected_engine_count
            and set(pool["scheduler_types"]) == {"Scheduler"}
            and set(pool["worker_types"]) == {"SimTpModelWorker"}
            and set(pool["simulated_worker_counts"]) == {8}
            and set(pool["tensor_parallel_sizes"]) == {1}
        )
        if not pool_held:
            fatal.append({"guard": "pool-process-or-runtime-identity", "pool": pool})

    controls = observation["controls"]
    constant = controls["constant"]
    sensitivity = controls["sensitivity"]
    constant_timeline = constant["timeline"]
    sensitivity_timeline = sensitivity["timeline"]
    constant_other_terms = dict(constant_timeline["decomposition"])
    sensitivity_other_terms = dict(sensitivity_timeline["decomposition"])
    constant_other_terms.pop("handoff_ps")
    constant_other_terms.pop("total_ps")
    sensitivity_other_terms.pop("handoff_ps")
    sensitivity_other_terms.pop("total_ps")
    sensitivity_row = {
        "ttft_delta_ps": sensitivity_timeline["ttft_ps"] - constant_timeline["ttft_ps"],
        "tpot_delta_ps": _fraction(sensitivity["tpot_ps"]) - _fraction(constant["tpot_ps"]),
        "other_terms_identical": constant_other_terms == sensitivity_other_terms,
    }
    sensitivity_held = (
        sensitivity_row["ttft_delta_ps"] == 100_000_000
        and sensitivity_row["tpot_delta_ps"] == 0
        and sensitivity_row["other_terms_identical"]
        and controls["packet_artifact_count_before"]
        == controls["packet_artifact_count_after_constants"]
    )
    if not sensitivity_held:
        fatal.append({"guard": "constant-sensitivity-exact-relation"})

    packet = observation["packet"]
    packet_request = packet["request"]
    packet_timeline = packet_request["timeline"]
    artifact = packet["artifact"]
    packet_duration = (
        packet_timeline["handoff"]["completed_at_ps"]
        - packet_timeline["handoff"]["submitted_at_ps"]
    )
    packet_service = artifact["packet_service_ps"]
    signed_ttft_delta = packet_timeline["ttft_ps"] - constant_timeline["ttft_ps"]
    signed_handoff_delta = packet_duration - CONSTANT_HANDOFF_PS
    metric_residual = signed_ttft_delta - signed_handoff_delta
    expected_pairs = [(index, index + 8) for index in range(8)]
    message_pairs = [
        (row["source_rank"], row["destination_rank"])
        for row in artifact["messages"]
    ]
    packet_held = (
        packet["prefill_ranks"] == list(range(8))
        and packet["decode_ranks"] == list(range(8, 16))
        and message_pairs == expected_pairs
        and artifact["aggregate_kv_bytes"] == 393_216
        and artifact["chunk_bytes"] == [49_152] * 8
        and len(artifact["flows"]) == 8
        and artifact["quiescent"] is True
        and metric_residual == 0
        and _fraction(packet_request["tpot_ps"]) == _fraction(constant["tpot_ps"])
        and packet_request["decode_token_ids"] == constant["decode_token_ids"]
        and PACKET_SHARD_FLOOR_PS <= packet_service <= PACKET_SERVICE_CEILING_PS
    )
    if not packet_held:
        fatal.append({"guard": "packet-exact-relation"})

    flagship = observation["flagship"]
    flagship_held = (
        flagship["nodes"] == 13
        and flagship["ranks"] == flagship["gpus"] == flagship["nics"] == 104
        and flagship["prefill_attn_dp"] == list(range(32))
        and flagship["prefill_tp"] == [0]
        and flagship["prefill_dense_dp"] == list(range(32))
        and flagship["prefill_ep"] == list(range(32))
        and flagship["decode_attn_dp"] == list(range(32, 104))
        and flagship["decode_tp"] == [32]
        and flagship["decode_dense_dp"] == [32]
        and flagship["decode_ep"] == list(range(32, 104))
        and flagship["core54_claimed_ranks"] == 96
    )
    if not flagship_held:
        fatal.append({"guard": "flagship-structural-render"})

    if not step_latencies or not all(
        STEP_FLOOR_PS <= value <= STEP_CEILING_PS for value in step_latencies
    ):
        fatal.append({"guard": "step-physical-bounds"})
    if not decode_rates or not all(
        DECODE_RATE_FLOOR <= value <= DECODE_RATE_CEILING for value in decode_rates
    ):
        fatal.append({"guard": "decode-rate-physical-bounds"})

    batching = []
    for prefill_engines, decode_engines in POOL_RATIOS:
        high_cells = [
            cell
            for cell in cells
            if cell["prefill_engines"] == prefill_engines
            and cell["decode_engines"] == decode_engines
            and cell["offered_load_requests_per_second"] == OFFERED_LOADS[-1]
        ]
        row = {
            "prefill_engines": prefill_engines,
            "decode_engines": decode_engines,
            "maximum_prefill_batch_size": max(
                cell["maximum_prefill_batch_size"] for cell in high_cells
            ),
            "maximum_decode_batch_size": max(
                cell["maximum_decode_batch_size"] for cell in high_cells
            ),
        }
        row["held"] = (
            row["maximum_prefill_batch_size"] >= 2
            and row["maximum_decode_batch_size"] >= 2
        )
        batching.append(row)

    curve_relations = []
    for curve in observation["curves"]:
        points = curve["points"]
        throughput = [
            _curve_fraction(point, "aggregated_output_throughput_tokens_per_second")
            for point in points
        ]
        delays = [
            _curve_fraction(point, "per_token_request_delay_ps")
            for point in points
        ]
        curve_relations.append(
            {
                "configuration_id": curve["configuration_id"],
                "throughput_nondecreasing": throughput == sorted(throughput),
                "delay_direction": (
                    "nondecreasing" if delays == sorted(delays) else "not-nondecreasing"
                ),
            }
        )
    prompt_relation = all(
        all(
            row["timeline"]["handoff"]["kv_bytes"] == prompt * 49_152
            for row in cell["requests"]
        )
        for cell in cells
        for prompt in (cell["prompt_tokens"],)
    )
    behavioral_held = (
        all(row["held"] for row in batching)
        and all(row["throughput_nondecreasing"] for row in curve_relations)
        and prompt_relation
    )
    if not behavioral_held:
        fatal.append({"guard": "frozen-behavioral-relations"})

    maximum_residual = max(
        row["maximum_ttft_residual_ps"] for row in exact_rows
    )
    return {
        "status": "PASS" if not fatal else "VOID",
        "fatal_guards": {"status": "HELD" if not fatal else "VIOLATED", "findings": fatal},
        "conservation": {
            "cells": len(exact_rows),
            "admissions": sum(row["admissions"] for row in exact_rows),
            "handoffs": sum(row["handoffs"] for row in exact_rows),
            "terminals": sum(row["terminals"] for row in exact_rows),
            "decode_tokens": sum(row["decode_tokens"] for row in exact_rows),
            "maximum_ttft_residual_ps": maximum_residual,
            "all_rows_held": all(row["held"] for row in exact_rows),
        },
        "constant_sensitivity": {
            **sensitivity_row,
            "tpot_delta_ps": _fraction_json(sensitivity_row["tpot_delta_ps"]),
            "held": sensitivity_held,
        },
        "packet_exact": {
            "aggregate_kv_bytes": artifact["aggregate_kv_bytes"],
            "chunks": len(artifact["chunk_bytes"]),
            "flows": len(artifact["flows"]),
            "packet_service_ps": packet_service,
            "packet_duration_ps": packet_duration,
            "constant_ttft_ps": constant_timeline["ttft_ps"],
            "packet_ttft_ps": packet_timeline["ttft_ps"],
            "signed_ttft_delta_ps": signed_ttft_delta,
            "signed_handoff_delta_ps": signed_handoff_delta,
            "metric_residual_ps": metric_residual,
            "tpot_identical": (
                _fraction(packet_request["tpot_ps"]) == _fraction(constant["tpot_ps"])
            ),
            "held": packet_held,
        },
        "batching": batching,
        "curve_relations": curve_relations,
        "curves": observation["curves"],
        "flagship": {
            "nodes": flagship["nodes"],
            "ranks": flagship["ranks"],
            "core54_claimed_ranks": flagship["core54_claimed_ranks"],
            "arithmetic_difference": flagship["ranks"] - flagship["core54_claimed_ranks"],
            "held": flagship_held,
        },
        "physical_sanity": {
            "minimum_step_ps": min(step_latencies),
            "maximum_step_ps": max(step_latencies),
            "minimum_decode_tokens_per_second": _fraction_json(min(decode_rates)),
            "maximum_decode_tokens_per_second": _fraction_json(max(decode_rates)),
            "resident_weight_read_floor_ps": 52_691_712,
            "packet_shard_serialization_floor_ps": PACKET_SHARD_FLOOR_PS,
            "packet_service_ps": packet_service,
        },
    }


def run(args: argparse.Namespace) -> None:
    provenance = check_registry(args, require_clean=True)
    args.run_dir.mkdir(parents=True, exist_ok=False)
    observation = run_observation(args, provenance)
    analysis = analyze_observation(observation)
    raw = {**observation, "analysis": analysis}
    raw_path = args.run_dir / "raw-result.json"
    raw_path.write_text(
        json.dumps(raw, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({"status": analysis["status"], "raw_result": raw_path.as_posix()}))
    if analysis["status"] != "PASS":
        raise SystemExit("SGL-33 study is void; see retained raw findings")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--sglang-source", type=Path, required=True)
    parser.add_argument("--txt2bin", type=Path, required=True)
    parser.add_argument("--htsim-rnic", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.check_only:
        checked = check_registry(args, require_clean=False)
        print(json.dumps({"check_only": True, "provenance": checked}, sort_keys=True))
        return
    run(args)


if __name__ == "__main__":
    main()
