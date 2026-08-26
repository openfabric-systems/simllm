#!/usr/bin/env python3
"""Validate and reproduce the clean COMP-75 composition record."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path, PurePath
from typing import Any

from comp75_independent_sign import sign_visible_movement
from comp75_repetition import (
    calibration_comparison,
    compare_core60_contracts,
    validate_expectations,
    verify_evidence_locks,
)

STUDY_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = STUDY_DIR.parents[1]
EXPECTATIONS_PATH = STUDY_DIR / "comp75_expectations.json"
CORE59_EXPECTATIONS_PATH = STUDY_DIR / "core59_expectations.json"
CORE60_EXPECTATIONS_PATH = STUDY_DIR / "core60_expectations.json"
ANCHOR_PATH = STUDY_DIR / "expectations.json"
SCORED_EXPECTATIONS_PATH = STUDY_DIR / "scored_expectations.json"
RUN_ROOT_ENV = "SIMLLM_COMP75_RUN_ROOT"
MECHANISM_SCHEMA = "simllm-deployment-curve-comp75-mechanism-evidence-v1"


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


def _require_external_run_dir(run_dir: Path) -> Path:
    configured = os.environ.get(RUN_ROOT_ENV)
    if not configured:
        raise SystemExit(f"{RUN_ROOT_ENV} must name the external run root")
    root = Path(configured).resolve()
    try:
        relative = run_dir.resolve().relative_to(root)
    except ValueError as exc:
        raise SystemExit(f"run directory must remain under {RUN_ROOT_ENV}") from exc
    if run_dir.exists():
        raise SystemExit(f"selected run directory already exists: {run_dir}")
    return relative


def _placement() -> Any:
    from simllm.placement import SglangPoolArrangement, sglang_disaggregated_manifests

    arrangement = SglangPoolArrangement(
        enable_data_parallel_attention=True,
        attention_data_parallel_size=32,
        dense_data_parallel_size=32,
        expert_parallel_size=32,
    )
    return sglang_disaggregated_manifests(
        prefill_nodes=4,
        decode_nodes=1,
        gpus_per_node=8,
        prefill_arrangement=arrangement,
        decode_arrangement=SglangPoolArrangement.identity(),
        framework_version="0.5.19.dev345+gbfeae4e79",
    ).placement


def _record() -> Any:
    from simllm.core import RequestPhase, ScheduledRequest, StepRecord

    return StepRecord(
        step_index=0,
        virtual_time_ps=0,
        scheduled=[
            ScheduledRequest(
                request_id="comp75-prefill-equivalent-pair-matrix",
                phase=RequestPhase.PREFILL,
                num_new_tokens=32,
                context_length=32,
            )
        ],
    )


def _equivalent_dims(pair_bytes: int) -> Any:
    from simllm.compute import ModelDims

    return ModelDims(
        num_layers=1,
        hidden_size=pair_bytes,
        intermediate_size=1,
        num_heads=1,
        num_kv_heads=1,
        head_size=1,
        vocab_size=1,
        dtype_bytes=1,
        num_experts=32,
        top_k=1,
        moe_intermediate_size=1,
        local_num_experts=1,
    )


def _run_phase(
    run_dir: Path,
    *,
    phase_name: str,
    pair_bytes: int,
    arm_id: str,
    link_rate: int,
) -> dict[str, Any]:
    from simllm.backends import HtsimStepSink, HtsimStepSinkConfig
    from simllm.backends.step_sink import NVLINK_MEDIUM
    from simllm.compute import ComputeProvider, DurationEstimate

    class ZeroProvider(ComputeProvider):
        def estimate(self, kernel, gpu):
            del kernel, gpu
            return DurationEstimate(duration_ps=0, bound="measured")

    workdir = run_dir / arm_id / phase_name
    sink = HtsimStepSink(
        HtsimStepSinkConfig(
            profile="rnic-nn",
            tp_ranks=(0,),
            dims=_equivalent_dims(pair_bytes),
            workdir=workdir,
            ep_ranks=tuple(range(32)),
            linkspeed_bps=link_rate,
            provider=ZeroProvider(),
            placement_manifest=_placement(),
        )
    )
    result = sink(_record())
    if result is None:
        raise AssertionError("the equivalent EP32 pair matrix produced no result")
    locality = sink.locality_outcomes[0]
    indexes = tuple(
        index
        for index, medium in enumerate(locality.local_phase_medium)
        if medium == NVLINK_MEDIUM
    )
    if len(indexes) != 2:
        raise AssertionError("the equivalent matrix must render two identical phases")
    first = indexes[0]
    for values in (
        locality.fabric_phase_service_ps,
        locality.local_phase_service_ps,
        locality.composed_phase_service_ps,
    ):
        if len({values[index] for index in indexes}) != 1:
            raise AssertionError("equivalent synthetic phases disagree")
    return {
        "arm": arm_id,
        "phase": phase_name,
        "pair_bytes": pair_bytes,
        "fabric_phase_service_ps": locality.fabric_phase_service_ps[first],
        "local_phase_service_ps": locality.local_phase_service_ps[first],
        "composed_phase_service_ps": locality.composed_phase_service_ps[first],
        "fabric_bytes_per_phase": locality.fabric_directed_bytes // 2,
        "local_bytes_per_phase": locality.nvlink_directed_bytes // 2,
        "fabric_segments_per_phase": locality.fabric_segments // 2,
        "local_segments_per_phase": locality.nvlink_segments // 2,
        "backend_runs_for_equivalent_pair": locality.backend_runs,
        "quiescent": sink.outcomes[0].quiescent,
        "workdir": render_cli_path(workdir.relative_to(run_dir)),
    }


def derive_mechanism_evidence(
    expectations: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    """Exercise every frozen packet-service endpoint."""

    relative_run_dir = _require_external_run_dir(args.run_dir)
    native = expectations["component_evidence"]["htsim"]
    _require_digest(args.htsim_rnic, native["binary_sha256"], "htsim_rnic")
    _require_digest(args.txt2bin, native["txt2bin_sha256"], "txt2bin")
    os.environ["SIMLLM_HTSIM_RNIC"] = str(args.htsim_rnic)
    os.environ["SIMLLM_TXT2BIN"] = str(args.txt2bin)
    args.run_dir.mkdir(parents=True, exist_ok=False)

    services = []
    arms = {row["id"]: row for row in expectations["composition"]["service_arms"]}
    for arm_id in ("point", "sensitivity"):
        arm = arms[arm_id]
        for phase_name in ("dispatch", "combine"):
            pair = expectations["traffic"][phase_name]["per_pair_bytes"]
            for edge in ("lower", "upper"):
                observed = _run_phase(
                    args.run_dir,
                    phase_name=f"{phase_name}-{edge}",
                    pair_bytes=pair[edge],
                    arm_id=arm_id,
                    link_rate=arm["fabric_link_rate_bits_per_second"],
                )
                if observed["composed_phase_service_ps"] != arm[
                    f"{phase_name}_phase_service_ps"
                ][edge]:
                    raise AssertionError(
                        f"{arm_id} {phase_name} {edge} service disagrees"
                    )
                services.append({**observed, "rounding_edge": edge})

    result = {
        "schema": MECHANISM_SCHEMA,
        "status": "PASS",
        "task": "COMP-75",
        "expectations_sha256": _sha256(EXPECTATIONS_PATH),
        "run_dir_relative_to_external_root": render_cli_path(relative_run_dir),
        "anchor_numeric_values_accessed": False,
        "held_out_access_ledger": [],
        "held_out_numeric_values_accessed": False,
        "held_out_score_performed": False,
        "scored_flagship_rerun_performed": False,
        "model_weights_loaded": False,
        "services": services,
    }
    _write_json(args.run_dir / "mechanism-evidence.json", result)
    return result


def _calibration_result(expectations: dict[str, Any]) -> dict[str, Any]:
    result = calibration_comparison(
        _load_json(ANCHOR_PATH),
        _load_json(SCORED_EXPECTATIONS_PATH),
        _load_json(CORE59_EXPECTATIONS_PATH),
        expectations,
    )
    result["expectations_sha256"] = _sha256(EXPECTATIONS_PATH)
    result["source_allowlist_preregistered_sha256"] = expectations["source_allowlist"][
        "preregistered_sha256"
    ]
    result["preservation_lock"] = verify_evidence_locks(expectations, REPOSITORY_ROOT)
    result["record_comparison"] = compare_core60_contracts(
        expectations, _load_json(CORE60_EXPECTATIONS_PATH)
    )
    row = result["calibration_rows"][0]
    published = row["published"]
    if published["denominator"] != 1:
        raise ValueError("visible published value must be an integer")
    result["independent_signoff"] = sign_visible_movement(
        per_node_tokens=row["per_node_tokens"],
        candidate_service_ps=row["candidate_service_ps"],
        core59_total_service_ps=row["core59_total_service_ps"],
        comp75_total_service_ps=row["total_service_ps"],
        published_tokens_per_second_per_node=published["numerator"],
    )
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
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    expectations = _load_json(EXPECTATIONS_PATH)
    validate_expectations(expectations)
    preservation = verify_evidence_locks(expectations, REPOSITORY_ROOT)
    if args.derive_service:
        if args.run_dir is None or args.htsim_rnic is None or args.txt2bin is None:
            raise SystemExit(
                "--derive-service requires --run-dir, --htsim-rnic and --txt2bin"
            )
        if args.output is not None:
            raise SystemExit("--derive-service writes inside --run-dir")
        result = derive_mechanism_evidence(expectations, args)
    elif args.calibration_only:
        if any(value is not None for value in (args.run_dir, args.htsim_rnic, args.txt2bin)):
            raise SystemExit("--calibration-only does not accept native-run arguments")
        result = _calibration_result(expectations)
        if args.output is not None:
            _write_json(args.output, result)
    else:
        if any(value is not None for value in (args.run_dir, args.htsim_rnic, args.txt2bin)):
            raise SystemExit("--check-only does not accept native-run arguments")
        if args.output is not None:
            raise SystemExit("--check-only does not write output")
        result = {
            "status": "PASS",
            "task": "COMP-75",
            "expectations_sha256": _sha256(EXPECTATIONS_PATH),
            "preservation_lock": preservation,
            "held_out_access_ledger": [],
            "held_out_numeric_values_accessed": False,
            "held_out_score_performed": False,
            "scored_flagship_rerun_performed": False,
        }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
