#!/usr/bin/env python3
"""Run the frozen CORE-53 session pricing study."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from dataclasses import replace
from fractions import Fraction
from pathlib import Path, PurePath
from typing import Any

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[1]
FREEZE_PATH = HERE / "expectations.json"
BASELINE_FREEZE_PATH = REPOSITORY_ROOT / "examples/pd_session_v1/expectations.json"
TRACE_PATH = REPOSITORY_ROOT / "examples/preplay_trace_v1/granite_length_cap.jsonl"

EXPECTATION_COMMIT = "fda6eed557aef037bf1794da1c1d8556a10a1ee0"
IMPLEMENTATION_COMMIT = "6817019376d153be2a4b6cdd972bbec36dfa23e6"
RESULT_SCHEMA = "simllm-pd-session-kernel-cycle-study-result-v1"
MODEL_ID = "ibm-granite/granite-3.0-1b-a400m-instruct"
MODEL_REVISION = "ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445"
VLLM_VERSION = "0.27.1"
PROMPT_LENGTHS = (8, 16)
HANDOFF_DURATIONS_PS = (100_000_000, 200_000_000)
DECODE_OUTPUT_TOKENS = 4
PS_PER_SECOND = 1_000_000_000_000


def render_cli_path(path: PurePath) -> str:
    """Render command paths with POSIX separators on every platform."""

    return path.as_posix()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _git_head() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
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


def _require_ancestor(commit: str, label: str) -> None:
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(f"{label} commit {commit} is not an ancestor of HEAD")


def _vllm_version(vllm_python: Path) -> str:
    completed = subprocess.run(
        [render_cli_path(vllm_python), "-c", "import vllm; print(vllm.__version__)"],
        check=True,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    return completed.stdout.strip().splitlines()[-1]


def _source_audit(
    baseline_freeze: dict[str, Any],
    vllm_source: Path,
) -> list[dict[str, str]]:
    rows = []
    as_of = baseline_freeze["as_of_commit"]
    for name, expected in baseline_freeze["source_audit_sha256"].items():
        if name.startswith("simllm/"):
            actual = _sha256_bytes(_git_blob(as_of, name))
            scope = f"git-blob:{as_of}"
        else:
            actual = _sha256(vllm_source / name.removeprefix("vllm/"))
            scope = "installed-vllm-source"
        if actual != expected:
            raise SystemExit(
                f"source audit hash disagrees for {name}: {actual} != {expected}"
            )
        rows.append({"path": name, "sha256": actual, "scope": scope})
    return rows


def check_registry(args: argparse.Namespace) -> dict[str, Any]:
    """Validate every frozen input without creating a run directory."""

    if args.run_dir.exists():
        raise SystemExit(f"selected run directory already exists: {args.run_dir}")
    _require_ancestor(EXPECTATION_COMMIT, "expectation")
    _require_ancestor(IMPLEMENTATION_COMMIT, "implementation")
    frozen = _load_json(FREEZE_PATH)
    baseline_freeze = _load_json(BASELINE_FREEZE_PATH)
    if frozen["authored_against"] != "4f7a316926ecd55fb00d376e5aae1bcfc01c1929":
        raise SystemExit("freeze base commit drifted")
    for relative, expected in frozen["baseline"]["tracked_sha256"].items():
        actual = _sha256(REPOSITORY_ROOT / relative)
        if actual != expected:
            raise SystemExit(
                f"accepted baseline file drifted for {relative}: {actual} != {expected}"
            )
    frontend = baseline_freeze["frontend"]
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
        "expectation_commit": EXPECTATION_COMMIT,
        "implementation_commit": IMPLEMENTATION_COMMIT,
        "run_head": _git_head(),
        "freeze_sha256": _sha256(FREEZE_PATH),
        "model_config_sha256": frontend["model_config_sha256"],
        "source_audit": _source_audit(baseline_freeze, args.vllm_source),
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
        vocab_size=49155,
        dtype_bytes=2,
        num_experts=32,
        top_k=8,
        moe_intermediate_size=64,
        local_num_experts=32,
    )


def _session_config(workdir: Path) -> Any:
    from simllm.adapters.vllm.pd_session import VllmPdSessionConfig
    from simllm.compute import RooflineProvider
    from simllm.core import DeclaredKvHandoffPolicy, KvHandoffGeometry

    return VllmPdSessionConfig(
        model=MODEL_ID,
        model_revision=MODEL_REVISION,
        workdir=workdir,
        dims=_granite_dims(),
        handoff_geometry=KvHandoffGeometry(24, 8, 64, 2),
        handoff_policy=DeclaredKvHandoffPolicy(HANDOFF_DURATIONS_PS[0]),
        tensor_parallel_size=8,
        max_model_len=64,
        num_gpu_blocks_override=64,
        max_num_seqs=8,
        token_id=512,
        provider=RooflineProvider(efficiency=0.7),
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


def _cell_key(prompt_tokens: int, handoff_ps: int) -> str:
    return f"prompt-{prompt_tokens}-handoff-{handoff_ps}"


def _fraction_int(value: Fraction) -> int:
    if value.denominator != 1:
        raise RuntimeError(f"expected an integral picosecond value, found {value}")
    return value.numerator


def _canonical_result(value: dict[str, Any]) -> bytes:
    from simllm.calibration.canonical import canonical_bytes

    return canonical_bytes(value)


def _run_arm(
    run_dir: Path,
    *,
    record_canonical: bytes | None,
    record_sha256: str | None,
) -> dict[str, Any]:
    from simllm.adapters.vllm.pd_session import VllmDisaggregatedSession
    from simllm.calibration.kernel_cycle_lut import compile_session_profile_provider
    from simllm.compute import RooflineProvider
    from simllm.core import DeclaredKvHandoffPolicy

    config = _session_config(run_dir)
    if record_canonical is not None:
        if record_sha256 is None:
            raise ValueError("record_sha256 is required with record_canonical")
        prefill = compile_session_profile_provider(
            record_canonical,
            expected_sha256=record_sha256,
            pool="prefill",
            comparator=RooflineProvider(efficiency=0.7),
        )
        decode = compile_session_profile_provider(
            record_canonical,
            expected_sha256=record_sha256,
            pool="decode",
            comparator=RooflineProvider(efficiency=0.7),
        )
        config = replace(
            config,
            prefill_provider=prefill,
            decode_provider=decode,
        )
    prompt = _prompt_tokens()
    cells = []
    with VllmDisaggregatedSession(config) as session:
        for prompt_length in PROMPT_LENGTHS:
            for handoff_ps in HANDOFF_DURATIONS_PS:
                label = _cell_key(prompt_length, handoff_ps)
                result = session.run_request(
                    label,
                    prompt[:prompt_length],
                    decode_output_tokens=DECODE_OUTPUT_TOKENS,
                    handoff_policy=DeclaredKvHandoffPolicy(handoff_ps),
                )
                serialized = result.to_json()
                canonical = _canonical_result(serialized)
                timeline = result.timeline
                cells.append(
                    {
                        "label": label,
                        "prompt_tokens": prompt_length,
                        "handoff_ps": handoff_ps,
                        "kv_bytes": timeline.handoff.kv_bytes,
                        "prefill_service_ps": timeline.prefill_service_ps,
                        "decode_first_token_service_ps": (
                            timeline.decode_first_token_service_ps
                        ),
                        "ttft_ps": timeline.ttft_ps,
                        "tpot_ps": _fraction_int(timeline.tpot_ps),
                        "decomposition_total_ps": timeline.decomposition_total_ps,
                        "decomposition_residual_ps": (
                            timeline.ttft_ps - timeline.decomposition_total_ps
                        ),
                        "request_result_sha256": _sha256_bytes(canonical),
                        "request_result": serialized,
                    }
                )
        engines = (*session.prefill_engines, *session.decode_engines)
        step_latencies = [
            step_result.step_latency_ps
            for engine in engines
            for record, step_result in zip(
                engine.executor.step_records,
                engine.executor.step_results,
                strict=True,
            )
            if record.scheduled
        ]
        pricing = {
            engine.role.value: engine.executor.compute_provider.pricing_provenance()
            for engine in engines
        }
    return {
        "cells": cells,
        "step_latencies_ps": step_latencies,
        "pricing_provenance": pricing,
    }


def _compact(cells: list[dict[str, Any]]) -> list[dict[str, int]]:
    fields = (
        "prompt_tokens",
        "handoff_ps",
        "kv_bytes",
        "prefill_service_ps",
        "decode_first_token_service_ps",
        "ttft_ps",
        "tpot_ps",
    )
    return [{field: int(row[field]) for field in fields} for row in cells]


def _cell_map(cells: list[dict[str, Any]]) -> dict[tuple[int, int], dict[str, Any]]:
    return {(row["prompt_tokens"], row["handoff_ps"]): row for row in cells}


def _guard(name: str, passed: bool, detail: object) -> dict[str, Any]:
    return {"name": name, "passed": passed, "detail": detail}


def evaluate(
    frozen: dict[str, Any],
    record: Any,
    absent_a: dict[str, Any],
    absent_b: dict[str, Any],
    lookup: dict[str, Any],
) -> dict[str, Any]:
    accepted = frozen["baseline"]["cells"]
    absent_a_cells = absent_a["cells"]
    absent_b_cells = absent_b["cells"]
    lookup_cells = lookup["cells"]
    absent_a_map = _cell_map(absent_a_cells)
    lookup_map = _cell_map(lookup_cells)
    movement_rows = []
    for expected in frozen["movement_oracle"]["cells"]:
        key = (expected["prompt_tokens"], expected["handoff_ps"])
        baseline = absent_a_map[key]
        observed = lookup_map[key]
        movement_rows.append(
            {
                "prompt_tokens": key[0],
                "handoff_ps": key[1],
                "expected_ttft_ps": expected["expected_ttft_ps"],
                "observed_ttft_ps": observed["ttft_ps"],
                "signed_ttft_delta_ps": observed["ttft_ps"] - baseline["ttft_ps"],
                "expected_signed_ttft_delta_ps": expected["signed_ttft_delta_ps"],
                "expected_tpot_ps": expected["expected_tpot_ps"],
                "observed_tpot_ps": observed["tpot_ps"],
                "signed_tpot_delta_ps": observed["tpot_ps"] - baseline["tpot_ps"],
                "expected_signed_tpot_delta_ps": expected["signed_tpot_delta_ps"],
                "passed": (
                    observed["ttft_ps"] == expected["expected_ttft_ps"]
                    and observed["tpot_ps"] == expected["expected_tpot_ps"]
                    and observed["ttft_ps"] - baseline["ttft_ps"]
                    == expected["signed_ttft_delta_ps"]
                    and observed["tpot_ps"] - baseline["tpot_ps"]
                    == expected["signed_tpot_delta_ps"]
                ),
            }
        )
    handoff_rows = []
    for prompt_length in PROMPT_LENGTHS:
        lower = lookup_map[(prompt_length, HANDOFF_DURATIONS_PS[0])]
        upper = lookup_map[(prompt_length, HANDOFF_DURATIONS_PS[1])]
        handoff_rows.append(
            {
                "prompt_tokens": prompt_length,
                "ttft_delta_ps": upper["ttft_ps"] - lower["ttft_ps"],
                "tpot_delta_ps": upper["tpot_ps"] - lower["tpot_ps"],
                "passed": (
                    upper["ttft_ps"] - lower["ttft_ps"] == 100_000_000
                    and upper["tpot_ps"] == lower["tpot_ps"]
                ),
            }
        )
    byte_rows = []
    for first, second in zip(absent_a_cells, absent_b_cells, strict=True):
        byte_rows.append(
            {
                "label": first["label"],
                "first_sha256": first["request_result_sha256"],
                "second_sha256": second["request_result_sha256"],
                "passed": _canonical_result(first["request_result"])
                == _canonical_result(second["request_result"]),
            }
        )
    complete_bytes_identical = all(row["passed"] for row in byte_rows)
    reproducibility_rows = [
        {
            "observation": name,
            "request_result_sha256s": [
                row["request_result_sha256"] for row in arm["cells"]
            ],
            "passed": _compact(arm["cells"]) == accepted
            and complete_bytes_identical,
        }
        for name, arm in (("record-absent-a", absent_a), ("record-absent-b", absent_b))
    ]
    prefill_provenance = lookup["pricing_provenance"]["prefill"]
    decode_provenance = lookup["pricing_provenance"]["decode"]
    bounds = frozen["physical_bounds"]
    live_steps = (
        absent_a["step_latencies_ps"]
        + absent_b["step_latencies_ps"]
        + lookup["step_latencies_ps"]
    )
    decode_rates = [PS_PER_SECOND / row["tpot_ps"] for row in lookup_cells]
    all_absent_rows = absent_a_cells + absent_b_cells
    guards = [
        _guard(
            "candidate-source-or-content-digest-disagreement",
            record.record_id == frozen["candidate_record"]["sha256"],
            record.record_id,
        ),
        _guard(
            "record-schema-or-candidate-status-disagreement",
            record.schema == frozen["candidate_record"]["schema"]
            and record.value["acceptance_status"] == "candidate",
            {
                "schema": record.schema,
                "acceptance_status": record.value["acceptance_status"],
            },
        ),
        _guard(
            "nonexact-or-duplicate-entry-selection",
            decode_provenance["lookup_hits"] == 2
            and prefill_provenance["lookup_hits"] == 0
            and decode_provenance["selected_entry_key_sha256s"]
            == [
                "d178524fd1bbc3719f7d45065e3a35b41ef9abfec3965d1db453152a30e303e1"
            ],
            lookup["pricing_provenance"],
        ),
        _guard(
            "provider-chain-provenance-loss",
            all("compute_pricing" in row["request_result"] for row in lookup_cells)
            and decode_provenance["record_sha256"] == record.record_id,
            lookup["pricing_provenance"],
        ),
        _guard(
            "accepted-pd-session-v1-tracked-byte-drift",
            _compact(absent_a_cells) == accepted,
            _compact(absent_a_cells),
        ),
        _guard(
            "record-absent-cell-or-request-byte-drift",
            _compact(absent_a_cells) == _compact(absent_b_cells)
            and all(row["passed"] for row in byte_rows)
            and all(
                "compute_pricing" not in row["request_result"]
                for row in all_absent_rows
            ),
            byte_rows,
        ),
        _guard(
            "ttft-decomposition-or-kv-byte-disagreement",
            all(row["decomposition_residual_ps"] == 0 for row in lookup_cells)
            and [row["kv_bytes"] for row in lookup_cells]
            == [row["kv_bytes"] for row in absent_a_cells],
            _compact(lookup_cells),
        ),
        _guard(
            "candidate-or-live-service-outside-physical-bounds",
            bounds["candidate_kernel_subset_service_ps"]["floor"]
            <= frozen["candidate_record"]["measured_service_ps"]
            <= bounds["candidate_kernel_subset_service_ps"]["ceiling"]
            and all(
                bounds["live_nonempty_step_service_ps"]["floor"]
                <= value
                <= bounds["live_nonempty_step_service_ps"]["ceiling"]
                for value in live_steps
            )
            and all(
                bounds["decode_tokens_per_second"]["floor"]
                <= value
                <= bounds["decode_tokens_per_second"]["ceiling"]
                for value in decode_rates
            ),
            {
                "candidate_service_ps": frozen["candidate_record"][
                    "measured_service_ps"
                ],
                "live_step_min_ps": min(live_steps),
                "live_step_max_ps": max(live_steps),
                "decode_tokens_per_second": decode_rates,
            },
        ),
        _guard(
            "candidate-status-promoted-to-calibration",
            decode_provenance["acceptance_status"] == "candidate"
            and decode_provenance["calibration_claim"] is False,
            decode_provenance,
        ),
    ]
    voiding = [row["name"] for row in guards if not row["passed"]]
    scored = {
        "R1-signed-selected-row-movement": movement_rows,
        "R2-handoff-orthogonality": handoff_rows,
        "R3-record-absent-byte-identity": reproducibility_rows,
    }
    scored_instances = [row for rows in scored.values() for row in rows]
    status = "VOID" if voiding else (
        "PASS" if all(row["passed"] for row in scored_instances) else "FAIL"
    )
    return {
        "schema": RESULT_SCHEMA,
        "status": status,
        "fatal_guards": guards,
        "voiding_guards": voiding,
        "scored_relation_families": 3,
        "scored_parameterized_instances": len(scored_instances),
        "scored_relations": scored,
        "record_absent": {
            "first": absent_a,
            "second": absent_b,
            "compact_accepted": _compact(absent_a_cells) == accepted,
            "complete_request_bytes_identical": complete_bytes_identical,
        },
        "lookup": lookup,
        "lookup_compact": _compact(lookup_cells),
        "physical_sanity": {
            "candidate_service_ps": frozen["candidate_record"][
                "measured_service_ps"
            ],
            "candidate_ceiling_ps": bounds["candidate_kernel_subset_service_ps"][
                "ceiling"
            ],
            "live_step_min_ps": min(live_steps),
            "live_step_max_ps": max(live_steps),
            "decode_tokens_per_second": decode_rates,
        },
        "closure": {
            "core_53_literal": False,
            "reason": (
                "the accepted candidate record covers one decode shape and no "
                "prefill shape"
            ),
            "required_residual_id": "COMP-73",
        },
        "scope": frozen["closure_rule"]["does_not_claim"],
    }


def run_study(args: argparse.Namespace, provenance: dict[str, Any]) -> int:
    from simllm.calibration.kernel_cycle_lut import analyze_kernel_cycle_capture

    args.run_dir.mkdir(parents=True, exist_ok=False)
    frozen = _load_json(FREEZE_PATH)
    try:
        record = analyze_kernel_cycle_capture(args.fixture)
        absent_a = _run_arm(
            args.run_dir / "record-absent-a",
            record_canonical=None,
            record_sha256=None,
        )
        absent_b = _run_arm(
            args.run_dir / "record-absent-b",
            record_canonical=None,
            record_sha256=None,
        )
        lookup = _run_arm(
            args.run_dir / "candidate-lookup",
            record_canonical=record.canonical,
            record_sha256=record.record_id,
        )
        result = evaluate(frozen, record, absent_a, absent_b, lookup)
        result["provenance"] = provenance
        _write_json(args.run_dir / "result.json", result)
        print(
            f"{result['status']}: lookup_hits="
            f"{lookup['pricing_provenance']['decode']['lookup_hits']}; "
            f"ttft_delta_ps={result['scored_relations']['R1-signed-selected-row-movement'][2]['signed_ttft_delta_ps']}"
        )
        if result["status"] == "VOID":
            return 2
        return 0 if result["status"] == "PASS" else 1
    except BaseException as exc:
        _write_json(
            args.run_dir / "result.json",
            {
                "schema": RESULT_SCHEMA,
                "status": "VOID",
                "provenance": provenance,
                "fatal_guards": [
                    {
                        "name": "study-execution",
                        "passed": False,
                        "detail": f"{type(exc).__name__}: {exc}",
                    }
                ],
            },
        )
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--vllm-source", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--vllm-python", type=Path, required=True)
    parser.add_argument(
        "--fixture",
        type=Path,
        default=REPOSITORY_ROOT / "tests/fixtures/kernel_cycle_lut_v1",
    )
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    provenance = check_registry(args)
    if args.check_only:
        print("check-only: CORE-53 frozen registry passed; no artifacts written")
        return 0
    return run_study(args, provenance)


if __name__ == "__main__":
    raise SystemExit(main())
