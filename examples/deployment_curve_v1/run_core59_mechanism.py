#!/usr/bin/env python3
"""Validate and reproduce the frozen CORE-59 calibration mechanisms."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path, PurePath
from typing import Any

from core59_role_mechanisms import (
    fit_calibration_only,
    validate_expectations,
    verify_historical_refutation,
)

STUDY_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = STUDY_DIR.parents[1]
EXPECTATIONS_PATH = STUDY_DIR / "core59_expectations.json"
ANCHOR_PATH = STUDY_DIR / "expectations.json"
SCORED_EXPECTATIONS_PATH = STUDY_DIR / "scored_expectations.json"
RUN_ROOT_ENV = "SIMLLM_CORE59_RUN_ROOT"
MECHANISM_EVIDENCE_SCHEMA = "simllm-deployment-curve-core59-mechanism-evidence-v1"


def render_cli_path(path: PurePath) -> str:
    """Render command-line paths with POSIX separators on every host."""

    return path.as_posix()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_digest(path: Path, expected: str, label: str) -> None:
    observed = _sha256(path)
    if observed != expected:
        raise SystemExit(f"{label} digest disagrees: {observed}")


def _require_external_run_dir(run_dir: Path) -> None:
    configured = os.environ.get(RUN_ROOT_ENV)
    if not configured:
        raise SystemExit(f"{RUN_ROOT_ENV} must name the external run root")
    root = Path(configured).resolve()
    try:
        run_dir.resolve().relative_to(root)
    except ValueError as exc:
        raise SystemExit(f"run directory must remain under {RUN_ROOT_ENV}") from exc
    if run_dir.exists():
        raise SystemExit(f"selected run directory already exists: {run_dir}")


def _selected_constants(expectations: dict[str, Any]) -> dict[str, int]:
    return {row["id"]: int(row["selected"]) for row in expectations["constants"]["declared"]}


def _deepseek_moe_dims(expectations: dict[str, Any]) -> Any:
    from simllm.compute import ModelDims

    selected = _selected_constants(expectations)
    return ModelDims(
        num_layers=1,
        hidden_size=selected["hidden_vector_elements"],
        intermediate_size=18_432,
        num_heads=128,
        num_kv_heads=128,
        head_size=192,
        vocab_size=129_280,
        dtype_bytes=selected["routed_vector_bytes_per_element"],
        num_experts=256,
        top_k=selected["routed_experts_per_token"],
        moe_intermediate_size=2_048,
        local_num_experts=32,
    )


def _placement(expectations: dict[str, Any]) -> Any:
    from simllm.placement import (
        SglangPoolArrangement,
        sglang_disaggregated_manifests,
    )

    selected = _selected_constants(expectations)
    ep_size = selected["prefill_expert_parallel_ranks"]
    arrangement = SglangPoolArrangement(
        enable_data_parallel_attention=True,
        attention_data_parallel_size=ep_size,
        dense_data_parallel_size=ep_size,
        expert_parallel_size=ep_size,
    )
    return sglang_disaggregated_manifests(
        prefill_nodes=4,
        decode_nodes=1,
        gpus_per_node=8,
        prefill_arrangement=arrangement,
        decode_arrangement=SglangPoolArrangement.identity(),
        framework_version="0.5.19.dev345+gbfeae4e79",
    ).placement


def _record(expectations: dict[str, Any]) -> Any:
    from simllm.core import RequestPhase, ScheduledRequest, StepRecord

    tokens = _selected_constants(expectations)["prefill_new_tokens_per_rank"]
    return StepRecord(
        step_index=0,
        virtual_time_ps=0,
        scheduled=[
            ScheduledRequest(
                request_id="core59-prefill-calibration-shape",
                phase=RequestPhase.PREFILL,
                num_new_tokens=tokens,
                context_length=tokens,
            )
        ],
    )


def _run_arm(
    expectations: dict[str, Any],
    run_dir: Path,
    *,
    arm: dict[str, Any],
) -> dict[str, Any]:
    from simllm.backends import HtsimStepSink, HtsimStepSinkConfig
    from simllm.backends.step_sink import NVLINK_MEDIUM
    from simllm.compute import ComputeProvider, DurationEstimate

    class ZeroProvider(ComputeProvider):
        def estimate(self, kernel, gpu):
            del kernel, gpu
            return DurationEstimate(duration_ps=0, bound="measured")

    ep_size = _selected_constants(expectations)["prefill_expert_parallel_ranks"]
    sink = HtsimStepSink(
        HtsimStepSinkConfig(
            profile=expectations["evidence"]["htsim"]["profile"],
            tp_ranks=(0,),
            dims=_deepseek_moe_dims(expectations),
            workdir=run_dir / arm["id"],
            ep_ranks=tuple(range(ep_size)),
            linkspeed_bps=arm["fabric_link_rate_bits_per_second"],
            provider=ZeroProvider(),
            placement_manifest=_placement(expectations),
        )
    )
    result = sink(_record(expectations))
    if result is None:
        raise AssertionError("the EP32 mechanism produced no collective result")
    locality = sink.locality_outcomes[0]
    outcome = sink.outcomes[0]
    collective_indexes = tuple(
        index for index, medium in enumerate(locality.local_phase_medium) if medium == NVLINK_MEDIUM
    )
    if len(collective_indexes) != 2 or locality.phase_count != 2:
        raise AssertionError("one MoE layer must render dispatch and combine")
    fabric_service = tuple(locality.fabric_phase_service_ps[index] for index in collective_indexes)
    local_service = tuple(locality.local_phase_service_ps[index] for index in collective_indexes)
    composed_service = tuple(
        locality.composed_phase_service_ps[index] for index in collective_indexes
    )
    if len(set(fabric_service)) != 1 or len(set(local_service)) != 1:
        raise AssertionError("dispatch and combine service must agree for this matrix")
    if len(set(composed_service)) != 1:
        raise AssertionError("dispatch and combine composition must agree")
    expected = next(
        row
        for row in expectations["mechanisms"][0]["physical_service"]["arms"]
        if row["id"] == arm["id"]
    )
    observed = {
        "id": arm["id"],
        "fabric_link_rate_bits_per_second": arm["fabric_link_rate_bits_per_second"],
        "fabric_phase_service_ps": fabric_service[0],
        "local_phase_service_ps": local_service[0],
        "composed_phase_service_ps": composed_service[0],
        "total_mechanism_service_ps": (
            composed_service[0]
            * expectations["mechanisms"][0]["traffic_arithmetic"]["application_count"]
        ),
    }
    if observed != expected:
        raise AssertionError(f"{arm['id']} mechanism evidence disagrees")
    return {
        **observed,
        "backend_runs": locality.backend_runs,
        "fabric_bytes_for_dispatch_and_combine": locality.fabric_directed_bytes,
        "fabric_segments_for_dispatch_and_combine": locality.fabric_segments,
        "goal_flow_count": outcome.num_flows,
        "nvlink_bytes_for_dispatch_and_combine": locality.nvlink_directed_bytes,
        "nvlink_segments_for_dispatch_and_combine": locality.nvlink_segments,
        "phase_count": locality.phase_count,
        "quiescent": outcome.quiescent,
        "workdir": render_cli_path((run_dir / arm["id"]).relative_to(run_dir)),
    }


def derive_mechanism_evidence(
    expectations: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Run only the two frozen communication arms, without anchor access."""

    _require_external_run_dir(args.run_dir)
    native = expectations["evidence"]["htsim"]
    _require_digest(args.htsim_rnic, native["binary_sha256"], "htsim_rnic")
    _require_digest(args.txt2bin, native["txt2bin_sha256"], "txt2bin")
    os.environ["SIMLLM_HTSIM_RNIC"] = str(args.htsim_rnic)
    os.environ["SIMLLM_TXT2BIN"] = str(args.txt2bin)
    args.run_dir.mkdir(parents=True, exist_ok=False)
    arms = [
        _run_arm(expectations, args.run_dir, arm=arm)
        for arm in expectations["mechanisms"][0]["physical_service"]["arms"]
    ]
    result = {
        "schema": MECHANISM_EVIDENCE_SCHEMA,
        "status": "PASS",
        "task": "CORE-59",
        "expectations_sha256": _sha256(EXPECTATIONS_PATH),
        "anchor_numeric_values_accessed": False,
        "held_out_numeric_values_accessed": False,
        "held_out_score_performed": False,
        "model_weights_loaded": False,
        "mechanism_id": expectations["mechanisms"][0]["id"],
        "arms": arms,
    }
    _write_json(args.run_dir / "mechanism-evidence.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--check-only", action="store_true")
    modes.add_argument("--derive-service", action="store_true")
    modes.add_argument("--calibration-only", action="store_true")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--htsim-rnic", type=Path)
    parser.add_argument("--txt2bin", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    expectations = _load_json(EXPECTATIONS_PATH)
    validate_expectations(expectations)
    history = verify_historical_refutation(expectations, REPOSITORY_ROOT)
    if args.derive_service:
        if args.run_dir is None or args.htsim_rnic is None or args.txt2bin is None:
            raise SystemExit("--derive-service requires --run-dir, --htsim-rnic and --txt2bin")
        result = derive_mechanism_evidence(expectations, args)
    elif args.calibration_only:
        if any(value is not None for value in (args.run_dir, args.htsim_rnic, args.txt2bin)):
            raise SystemExit("--calibration-only does not accept native-run arguments")
        result = fit_calibration_only(
            _load_json(ANCHOR_PATH),
            _load_json(SCORED_EXPECTATIONS_PATH),
            expectations,
        )
        result["historical_refutation_lock"] = history
    else:
        if any(value is not None for value in (args.run_dir, args.htsim_rnic, args.txt2bin)):
            raise SystemExit("--check-only does not accept native-run arguments")
        result = {
            "status": "PASS",
            "task": "CORE-59",
            "expectations_sha256": _sha256(EXPECTATIONS_PATH),
            "historical_refutation_lock": history,
            "calibration_numeric_values_accessed": False,
            "held_out_numeric_values_accessed": False,
            "held_out_score_performed": False,
        }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
