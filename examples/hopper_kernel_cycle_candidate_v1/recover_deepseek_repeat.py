#!/usr/bin/env python3
"""Retain exact priced DeepSeek repeat service from profiler sources."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from recover_mtp_service import (
    EXPECTED_CONFIG_SHA256,
    EXPECTED_MODEL,
    EXPECTED_REVISION,
    _nvtx_service,
    _read_json,
    _sha256,
    _source_manifest,
)

SCHEMA = "simllm-hopper-deepseek-repeat-recovery-v1"
PREFILL_CELLS = {
    "prefill_r16_l1024_t16384": "ep32-prefill-r16-l1024-t16384",
    "prefill_r4_l4096_t16384": "ep32-prefill-r4-l4096-t16384",
    "prefill_r8_l2048_t16384": "ep32-prefill-r8-l2048-t16384",
}
DECODE_CELL = "decode_b32_c2000"
DECODE_SUFFIX = "ep72-decode-b32-c2000"
DECODE_LABEL = "execute_context_0(0)_generation_32(32)"


def _validate_common(profile: dict[str, Any], suite: str) -> None:
    expected = {
        "model": EXPECTED_MODEL,
        "model_key": "deepseek-v3",
        "tensor_parallel_size": 1,
        "mode": "graph",
        "shape_set": "deepseek",
        "deepseek_suite": suite,
        "reduced_layers": 4,
        "phase": "profile",
    }
    for name, value in expected.items():
        if profile.get(name) != value:
            raise ValueError(f"profile.{name}: expected {value!r}, found {profile.get(name)!r}")
    config = profile.get("model_config")
    if not isinstance(config, dict):
        raise TypeError("profile.model_config: expected an object")
    for name, value in {
        "requested_revision": EXPECTED_REVISION,
        "resolved_revision": EXPECTED_REVISION,
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "effective_num_hidden_layers": 4,
    }.items():
        if config.get(name) != value:
            raise ValueError(
                f"profile.model_config.{name}: expected {value!r}, found {config.get(name)!r}"
            )


def _prefill_services(run_dir: Path, profile: dict[str, Any]) -> list[dict[str, Any]]:
    cases = {
        str(case["cell"]): case
        for case in profile["cases"]
        if str(case["cell"]).startswith("prefill_")
    }
    if set(cases) != set(PREFILL_CELLS):
        raise ValueError("profile.cases: expected the exact three registered prefill cells")
    for cell, case in cases.items():
        expected_shape = {
            "prefill_r16_l1024_t16384": (16, 1024),
            "prefill_r8_l2048_t16384": (8, 2048),
            "prefill_r4_l4096_t16384": (4, 4096),
        }[cell]
        if (case["batch_size"], case["input_len"]) != expected_shape:
            raise ValueError(f"profile.cases[{cell}]: shape changed")

    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    with (run_dir / "analysis/ordered-kernels.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        for row in csv.DictReader(stream):
            cell = str(row["cell"])
            if cell not in PREFILL_CELLS:
                continue
            totals[cell][0] += 1
            duration_ns = int(row["duration_ns"])
            if row["is_collective"] == "False":
                totals[cell][1] += duration_ns
            elif row["is_collective"] == "True":
                totals[cell][2] += duration_ns
            else:
                raise ValueError(f"ordered-kernels.csv: invalid collective flag {row['is_collective']!r}")
    if set(totals) != set(PREFILL_CELLS):
        raise ValueError("ordered-kernels.csv: one or more prefill cells are absent")
    result = []
    for cell, implementation_suffix in sorted(PREFILL_CELLS.items()):
        record_count, noncollective_ns, collective_ns = totals[cell]
        if record_count <= 0 or noncollective_ns <= 0:
            raise ValueError(f"{cell}: expected positive retained kernel service")
        result.append(
            {
                "cell": cell,
                "implementation_suffix": implementation_suffix,
                "measured_service_ps": noncollective_ns * 1000,
                "collective_service_ps": collective_ns * 1000,
                "kernel_record_count": record_count,
                "boundary_basis": "exact-profile-case-window",
            }
        )
    return result


def _decode_service(run_dir: Path, profile: dict[str, Any]) -> list[dict[str, Any]]:
    matches = [case for case in profile["cases"] if case["cell"] == DECODE_CELL]
    if len(matches) != 1:
        raise ValueError("profile.cases: expected one registered decode cell")
    case = matches[0]
    expected_shape = ("decode", 32, 2000, 2, 1)
    actual_shape = tuple(
        case[name]
        for name in ("pool", "batch_size", "input_len", "output_len", "decode_steps")
    )
    if actual_shape != expected_shape:
        raise ValueError(f"profile.cases[{DECODE_CELL}]: shape changed")
    noncollective_ns, collective_ns, boundary = _nvtx_service(
        run_dir,
        case,
        DECODE_LABEL,
    )
    return [
        {
            "cell": DECODE_CELL,
            "implementation_suffix": DECODE_SUFFIX,
            "measured_service_ps": noncollective_ns * 1000,
            "collective_service_ps": collective_ns * 1000,
            "kernel_record_count": boundary["kernel_record_count"],
            "boundary_basis": boundary["basis"],
            "boundary_label": boundary["label"],
            "runtime_correlation_count": boundary["runtime_correlation_count"],
        }
    ]


def recover_repeat(run_dir: Path, suite: str) -> dict[str, Any]:
    """Recover one base or decode repeat without pooling their roles."""

    profile = _read_json(run_dir / "profile.json")
    _validate_common(profile, suite)
    if (run_dir / "weight_files.txt").read_text(encoding="utf-8").strip():
        raise ValueError("weight_files.txt: dummy-weight isolation was violated")
    services = (
        _prefill_services(run_dir, profile)
        if suite == "base"
        else _decode_service(run_dir, profile)
    )
    analysis_status_path = run_dir / "analysis_status.txt"
    analysis_status = analysis_status_path.read_text(encoding="utf-8").strip()
    if analysis_status not in {"0", "1"}:
        raise ValueError(
            f"analysis_status.txt: expected 0 or 1, found {analysis_status!r}"
        )
    compact_analysis = (
        {"status": "PASSED"}
        if analysis_status == "0"
        else {
            "status": "BLOCKED",
            "reason": "the staged all-cells analyzer could not resolve its decode boundary",
        }
    )
    sources = _source_manifest(run_dir)
    sources.append(
        {
            "name": "analysis_status.txt",
            "bytes": analysis_status_path.stat().st_size,
            "sha256": _sha256(analysis_status_path),
        }
    )
    return {
        "schema": SCHEMA,
        "suite": suite,
        "evidence_class": "MEASURED",
        "services": services,
        "sources": sources,
        "original_compact_analysis": compact_analysis,
        "distribution_propagation": "DEFERRED_TO_COMP-74",
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--suite", choices=("base", "decode"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = recover_repeat(args.run_dir.resolve(), args.suite)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
