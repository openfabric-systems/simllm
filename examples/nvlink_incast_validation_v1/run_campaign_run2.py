#!/usr/bin/env python3
"""Run or resume the second frozen NV4 cell registered as TRAF-74."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TRAF70_ROOT = HERE.parent / "a100_nvlink_packet_v2"
FIRST_RUNNER_PATH = HERE / "run_campaign.py"


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_runner = _load_module("_traf74_first_campaign_reused_for_run2", FIRST_RUNNER_PATH)

EXPECTATIONS_PATH = HERE / "expectations_run2.json"
EXPECTATIONS_COMMIT = "b21ba822707d2d7c80b83ee2d3fb87f4fa93178d"
EXPECTATIONS_SHA256 = "5465271e9909cebc214c153209316a6f266ec142d7e578b3279935b1c6a10a53"
REGISTRY_TASK_ID = "TRAF-74"
CELL_ID = "nv4-long-flow-incast-run2"
CELL_SCHEMA = "simllm-nvlink-incast-validation-cell-v2"
MANIFEST_SCHEMA = "simllm-nvlink-incast-validation-attempt-manifest-v2"
OBSERVATION_SCHEMA = _runner.OBSERVATION_SCHEMA
CELL_TIMEOUT_SECONDS = 10 * 60
BULK_ROOT_ENV = "SIMLLM_NVINC_RUN2_BULK_ROOT"

traf70_cases = _runner.traf70_cases
traf70_run = _runner.traf70_run

IMPLEMENTATION_PATHS = (
    Path(__file__),
    HERE / "run_merlin_cell_run2.sbatch",
    FIRST_RUNNER_PATH,
    TRAF70_ROOT / "case_matrix.py",
    TRAF70_ROOT / "nvlink_packet_lane.cu",
    TRAF70_ROOT / "run_study.py",
    TRAF70_ROOT / "sha256.h",
)


def campaign_points(frozen: dict[str, Any]) -> tuple[Any, ...]:
    """Expand the six run-two cells without changing the TRAF-70 producer."""
    arm = frozen["hardware_arm"]
    rows = []
    for size_bytes in arm["flow_sizes_bytes"]:
        if size_bytes % arm["producer_payload_bytes"]:
            raise RuntimeError("flow size is not divisible by the producer payload")
        message_count = size_bytes // arm["producer_payload_bytes"]
        for degree in arm["degrees"]:
            sources = ",".join(
                str(source) for source in arm["senders_by_degree"][str(degree)]
            )
            for repetition in range(arm["repetitions_per_cell"]):
                rows.append(
                    traf70_cases.SweepPoint(
                        case_name=f"TRAF74_NVINC_RUN2_LONG_D{degree}",
                        point_id=(
                            f"TRAF74_NVINC_RUN2_LONG_D{degree}:size={size_bytes}:"
                            f"repeat={repetition:02d}"
                        ),
                        producer=arm["producer"],
                        payload_bytes=arm["producer_payload_bytes"],
                        message_count=message_count,
                        source=1,
                        destination=arm["receiver"],
                        sources=sources,
                        destinations=str(arm["receiver"]),
                        pattern=(
                            "one_source"
                            if degree == 1
                            else "two_source_simultaneous"
                            if degree == 2
                            else "three_source_simultaneous"
                        ),
                    )
                )
    expected = (
        len(arm["flow_sizes_bytes"])
        * len(arm["degrees"])
        * arm["repetitions_per_cell"]
    )
    if len(rows) != expected:
        raise RuntimeError(f"{REGISTRY_TASK_ID} run-two point expansion changed")
    return tuple(rows)


for name, value in {
    "EXPECTATIONS_PATH": EXPECTATIONS_PATH,
    "EXPECTATIONS_COMMIT": EXPECTATIONS_COMMIT,
    "EXPECTATIONS_SHA256": EXPECTATIONS_SHA256,
    "REGISTRY_TASK_ID": REGISTRY_TASK_ID,
    "CELL_ID": CELL_ID,
    "CELL_SCHEMA": CELL_SCHEMA,
    "MANIFEST_SCHEMA": MANIFEST_SCHEMA,
    "CELL_TIMEOUT_SECONDS": CELL_TIMEOUT_SECONDS,
    "BULK_ROOT_ENV": BULK_ROOT_ENV,
    "IMPLEMENTATION_PATHS": IMPLEMENTATION_PATHS,
    "campaign_points": campaign_points,
}.items():
    setattr(_runner, name, value)

parse_args = _runner.parse_args
_first_load_expectations = _runner.load_expectations


def load_expectations(
    expected_digest: str = EXPECTATIONS_SHA256,
) -> dict[str, Any]:
    """Load only the committed second-capture freeze."""
    return _first_load_expectations(expected_digest)


_runner.load_expectations = load_expectations
verify_preservation = _runner.verify_preservation
resolve_output_root = _runner.resolve_output_root
check_arguments = _runner.check_arguments
cell_root = _runner.cell_root
complete_attempt = _runner.complete_attempt
verify_attempt = _runner.verify_attempt
implementation_sha256 = _runner.implementation_sha256
sha256 = _runner.sha256


def main(argv: list[str] | None = None) -> int:
    """Run the second capture through the separately pinned configuration."""
    return _runner.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
